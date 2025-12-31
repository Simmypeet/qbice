use std::{sync::Arc, thread::available_parallelism};

use crate::{
    Engine,
    config::Config,
    engine::computation_graph::{
        CallerInformation, lock::BackwardProjectionLockGuard,
    },
    query::QueryID,
};

impl<C: Config> Engine<C> {
    pub(super) async fn invoke_backward_projections(
        self: &Arc<Self>,
        current_query_id: QueryID,
        backward_projection_lock_guard: BackwardProjectionLockGuard<'_, C>,
    ) {
        let backward_edges =
            self.computation_graph.get_backward_edges(current_query_id);

        let backward_projections = backward_edges
            .iter()
            .map(|x| *x)
            .filter(|x| {
                let query_kind =
                    self.computation_graph.get_query_kind(*x).unwrap();

                query_kind.is_projection()
            })
            .collect::<Vec<_>>();

        let expected_parallelism = available_parallelism()
            .map_or_else(|_| 1, std::num::NonZero::get)
            * 4;

        let chunk_size = std::cmp::max(
            (backward_projections.len()) / expected_parallelism,
            1,
        );

        let mut handles = Vec::new();

        for chunk in backward_projections.chunks(chunk_size) {
            let engine = Arc::clone(self);
            let chunk = chunk.to_vec();

            handles.push(tokio::spawn(async move {
                for query_id in chunk {
                    let entry =
                        engine.executor_registry.get_executor_entry_by_type_id(
                            &query_id.stable_type_id(),
                        );

                    let _ = entry
                        .repair_query_from_query_id(
                            &engine,
                            query_id.compact_hash_128(),
                            CallerInformation::BackwardProjectionPropagation,
                        )
                        .await;
                }
            }));
        }

        for handle in handles {
            let _ = handle.await;
        }

        self.done_backward_projection(
            &current_query_id,
            backward_projection_lock_guard,
        );
    }

    pub(super) async fn try_do_backward_projections(
        self: &Arc<Self>,
        query_id: QueryID,
    ) {
        let current_timestamp =
            self.computation_graph.timestamp_manager.get_current();

        loop {
            // no more pending backward projection
            if self
                .computation_graph
                .get_pending_backward_projection(query_id)
                .is_none_or(|x| x != current_timestamp)
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
            self.invoke_backward_projections(query_id, lock_guard).await;
        }
    }
}
