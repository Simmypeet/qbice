use std::{sync::Arc, thread::available_parallelism};

use tokio::task::JoinSet;

use crate::{
    Engine,
    config::Config,
    engine::computation_graph::{
        CallerInformation, caller::CallerKind,
        lock::BackwardProjectionLockGuard,
    },
    query::QueryID,
};

impl<C: Config> Engine<C> {
    #[allow(clippy::await_holding_lock)]
    pub(super) async fn invoke_backward_projections(
        self: &Arc<Self>,
        current_query_id: &QueryID,
        caller_information: &CallerInformation,
        backward_projection_lock_guard: BackwardProjectionLockGuard<C>,
    ) {
        let backward_edges = self.get_backward_edges(current_query_id).await;

        let mut backward_projections = Vec::new();
        for query_id in backward_edges {
            let query_kind = self.get_query_kind(&query_id).await.unwrap();

            if query_kind.is_projection() {
                backward_projections.push(query_id);
            }
        }

        let expected_parallelism = available_parallelism()
            .map_or_else(|_| 1, std::num::NonZero::get)
            * 4;

        let chunk_size = std::cmp::max(
            (backward_projections.len()) / expected_parallelism,
            1,
        );

        let mut join_set = JoinSet::new();

        let timestamp = caller_information.timestamp();
        for chunk in backward_projections.chunks(chunk_size) {
            let engine = Arc::clone(self);
            let chunk = chunk.to_vec();
            let active_computation_graph =
                caller_information.clone_active_computation_guard();

            join_set.spawn(async move {
                for query_id in chunk {
                    let entry =
                        engine.executor_registry.get_executor_entry_by_type_id(
                            &query_id.stable_type_id(),
                        );

                    let _ = entry
                        .repair_query_from_query_id(
                            &engine,
                            &query_id.compact_hash_128(),
                            &CallerInformation::new(
                                CallerKind::BackwardProjectionPropagation,
                                timestamp,
                                active_computation_graph.clone(),
                            ),
                        )
                        .await;
                }
            });
        }

        while let Some(res) = join_set.join_next().await {
            match res {
                Ok(()) => {}

                Err(er) => match er.try_into_panic() {
                    Ok(panic_reason) => {
                        std::panic::resume_unwind(panic_reason);
                    }
                    Err(er) => {
                        panic!(
                            "Backward projection task failed without \
                             panicking: {er}"
                        );
                    }
                },
            }
        }

        self.done_backward_projection(
            current_query_id,
            backward_projection_lock_guard,
        )
        .await;
    }

    pub(super) async fn try_do_backward_projections(
        self: &Arc<Self>,
        query_id: &QueryID,
        caller_information: &CallerInformation,
    ) {
        loop {
            // no more pending backward projection
            if self
                .get_pending_backward_projection(query_id)
                .await
                .is_none_or(|x| x != caller_information.timestamp())
            {
                return;
            }

            let Some(lock_guard) =
                self.get_backward_projection_lock_guard(query_id)
            else {
                // lock is not available, need to wait

                let Some(pending_lock) = self
                    .computation_graph
                    .lock
                    .try_get_pending_backward_projection_lock(query_id)
                else {
                    // during this short amount of time, the lock is now
                    // available, try again
                    continue;
                };

                let notified_owned = pending_lock.notified_owned();
                drop(pending_lock);

                notified_owned.await;

                continue;
            };

            // the lock is acquired, do the backward propagations
            self.invoke_backward_projections(
                query_id,
                caller_information,
                lock_guard,
            )
            .await;
        }
    }
}
