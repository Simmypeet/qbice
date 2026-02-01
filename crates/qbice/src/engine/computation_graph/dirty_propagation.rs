use std::sync::Arc;

use futures::lock::Mutex;
use tokio::{sync::Semaphore, task::JoinSet};

use crate::{
    Engine, ExecutionStyle, config::Config,
    engine::computation_graph::QueryKind, query::QueryID,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Edge {
    pub caller: QueryID,
    pub callee: QueryID,
}

impl<C: Config> Engine<C> {
    #[allow(clippy::manual_async_fn, clippy::await_holding_lock)]
    fn dirty_propagate_async<'a>(
        self: &'a Arc<Self>,
        query_id: &'a QueryID,
        dirty_list: &'a Arc<Mutex<Vec<Edge>>>,
        bound: &'a Arc<Semaphore>,
    ) -> impl Future<Output = ()> + 'a + Send {
        async move {
            // has already been marked dirty
            if !self.insert_dirty_query(query_id) {
                return;
            }

            let backward_edges =
                unsafe { self.get_backward_edges_unchecked(query_id).await };

            let mut join_sets = JoinSet::new();

            for caller in backward_edges.0.read().iter() {
                let permit_attempt = bound.clone().try_acquire_owned();

                // can spawn a new task
                if let Ok(permit) = permit_attempt {
                    let engine = self.clone();
                    let dirty_list = dirty_list.clone();
                    let bound = bound.clone();
                    let query_id = *query_id;

                    join_sets.spawn(async move {
                        let _permit = permit;

                        if engine
                            .stem_child(&caller, &query_id, &dirty_list)
                            .await
                        {
                            engine
                                .dirty_propagate_async(
                                    &caller,
                                    &dirty_list,
                                    &bound,
                                )
                                .await;
                        }
                    });
                }
                // has to run in current task
                else if self.stem_child(&caller, query_id, dirty_list).await {
                    Box::pin(
                        self.dirty_propagate_async(&caller, dirty_list, bound),
                    )
                    .await;
                }
            }

            while let Some(res) = join_sets.join_next().await {
                res.expect("Task panicked during dirty propagation");
            }
        }
    }

    #[allow(clippy::similar_names)]
    async fn stem_child(
        self: &Arc<Self>,
        caller: &QueryID,
        callee: &QueryID,
        dirty_list: &Arc<Mutex<Vec<Edge>>>,
    ) -> bool {
        dirty_list.lock().await.push(Edge { caller: *caller, callee: *callee });

        let query_kind =
            unsafe { self.get_query_kind_unchecked(caller).await.unwrap() };

        // if this is a firewall node or projection node, then we stop
        // propagation here.
        !matches!(
            query_kind,
            QueryKind::Executable(
                ExecutionStyle::Firewall | ExecutionStyle::Projection
            )
        )
    }

    pub(super) async fn get_dirty_propagate_list_from_batch(
        self: &Arc<Self>,
        query_id: impl IntoIterator<Item = QueryID>,
    ) -> Vec<Edge> {
        // NOTE: we spawn a join handle here because dirty propagation can't be
        // CANCELLED. If it could be cancelled, then we might leave the
        // computation graph in an inconsistent state.
        let bound = std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(8)
            * 8;

        let engine = self.clone();
        let semaphore = Arc::new(Semaphore::new(bound));
        let dirty_list = Arc::new(Mutex::new(Vec::new()));

        let mut join_set = JoinSet::new();

        for qid in query_id {
            match semaphore.clone().try_acquire_owned() {
                // can spawn a new task
                Ok(permit) => {
                    let engine = engine.clone();
                    let dirty_list = dirty_list.clone();
                    let semaphore = semaphore.clone();

                    join_set.spawn(async move {
                        let _permit = permit;

                        engine
                            .dirty_propagate_async(
                                &qid,
                                &dirty_list,
                                &semaphore,
                            )
                            .await;
                    });
                }
                // has to run in current task
                Err(_) => {
                    engine
                        .dirty_propagate_async(&qid, &dirty_list, &semaphore)
                        .await;
                }
            }
        }

        while let Some(res) = join_set.join_next().await {
            res.expect("Task panicked during dirty propagation");
        }

        Arc::try_unwrap(dirty_list)
            .expect("No other references to dirty_list exist")
            .into_inner()
    }

    pub(super) async fn get_dirty_propagate_list(
        self: &Arc<Self>,
        query_id: &QueryID,
    ) -> Vec<Edge> {
        // NOTE: we spawn a join handle here because dirty propagation can't be
        // CANCELLED. If it could be cancelled, then we might leave the
        // computation graph in an inconsistent state.
        let bound = std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(8)
            * 8;

        let engine = self.clone();
        let semaphore = Arc::new(Semaphore::new(bound));
        let dirty_list = Arc::new(Mutex::new(Vec::new()));

        engine.dirty_propagate_async(query_id, &dirty_list, &semaphore).await;

        Arc::try_unwrap(dirty_list)
            .expect("No other references to dirty_list exist")
            .into_inner()
    }

    pub(super) fn insert_dirty_query(&self, query_id: &QueryID) -> bool {
        self.computation_graph.dirtied_queries.insert(*query_id)
    }

    pub(super) fn clear_dirtied_queries(&self) {
        self.computation_graph.dirtied_queries.clear();
    }
}
