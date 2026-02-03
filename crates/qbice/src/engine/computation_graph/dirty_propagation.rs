use std::sync::Arc;

use tokio::{
    sync::{Mutex, Semaphore},
    task::JoinSet,
};

use crate::{
    Engine, ExecutionStyle,
    config::{Config, WriteTransaction},
    engine::computation_graph::QueryKind,
    query::QueryID,
};

impl<C: Config> Engine<C> {
    #[allow(clippy::manual_async_fn, clippy::await_holding_lock)]
    fn dirty_propagate_async<'a>(
        self: &'a Arc<Self>,
        query_id: &'a QueryID,
        bound: &'a Arc<Semaphore>,
        tx: &'a Arc<Mutex<WriteTransaction<C>>>,
    ) -> impl Future<Output = ()> + 'a + Send {
        async move {
            // has already been marked dirty
            if !self.insert_dirty_query(query_id) {
                return;
            }

            let backward_edges = self.get_backward_edges(query_id).await;

            let mut join_sets = JoinSet::new();

            for caller in backward_edges {
                let permit_attempt = bound.clone().try_acquire_owned();

                // can spawn a new task
                if let Ok(permit) = permit_attempt {
                    let engine = self.clone();
                    let bound = bound.clone();
                    let query_id = *query_id;
                    let tx = tx.clone();

                    join_sets.spawn(async move {
                        let _permit = permit;

                        if engine.stem_child(&caller, &query_id, &tx).await {
                            engine
                                .dirty_propagate_async(&caller, &bound, &tx)
                                .await;
                        }
                    });
                }
                // has to run in current task
                else if self.stem_child(&caller, query_id, tx).await {
                    Box::pin(self.dirty_propagate_async(&caller, bound, tx))
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
        tx: &Arc<Mutex<WriteTransaction<C>>>,
    ) -> bool {
        self.mark_dirty_forward_edge(*caller, *callee, &mut *tx.lock().await)
            .await;

        let query_kind = self.get_query_kind(caller).await.unwrap();

        // if this is a firewall node or projection node, then we stop
        // propagation here.
        !matches!(
            query_kind,
            QueryKind::Executable(
                ExecutionStyle::Firewall | ExecutionStyle::Projection
            )
        )
    }

    pub(super) async fn dirty_propagate_from_batch(
        self: &Arc<Self>,
        query_id: impl IntoIterator<Item = QueryID>,
        trasnaction: WriteTransaction<C>,
    ) -> WriteTransaction<C> {
        // NOTE: we spawn a join handle here because dirty propagation can't be
        // CANCELLED. If it could be cancelled, then we might leave the
        // computation graph in an inconsistent state.
        let bound = std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(8)
            * 8;

        let tx = Arc::new(Mutex::new(trasnaction));
        let engine = self.clone();
        let semaphore = Arc::new(Semaphore::new(bound));

        let mut join_set = JoinSet::new();

        for qid in query_id {
            match semaphore.clone().try_acquire_owned() {
                // can spawn a new task
                Ok(permit) => {
                    let engine = engine.clone();
                    let semaphore = semaphore.clone();
                    let tx = tx.clone();

                    join_set.spawn(async move {
                        let _permit = permit;

                        engine
                            .dirty_propagate_async(&qid, &semaphore, &tx)
                            .await;
                    });
                }
                // has to run in current task
                Err(_) => {
                    engine.dirty_propagate_async(&qid, &semaphore, &tx).await;
                }
            }
        }

        while let Some(res) = join_set.join_next().await {
            res.expect("Task panicked during dirty propagation");
        }

        Arc::try_unwrap(tx)
            .unwrap_or_else(|_| {
                panic!("the transaction is still held elsewhere")
            })
            .into_inner()
    }

    pub(super) fn insert_dirty_query(&self, query_id: &QueryID) -> bool {
        self.computation_graph.dirtied_queries.insert(*query_id)
    }

    pub(super) fn clear_dirtied_queries(&self) {
        self.computation_graph.dirtied_queries.clear();
    }
}
