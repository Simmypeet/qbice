use std::thread::available_parallelism;

use tokio::task::JoinSet;

use crate::{
    Query,
    config::Config,
    engine::computation_graph::{
        CallerInformation, caller::CallerKind,
        computing::BackwardProjectionLockGuard, database::Snapshot,
    },
};

impl<C: Config, Q: Query> Snapshot<C, Q> {
    pub(super) async fn invoke_backward_projections(
        self,
        caller_information: &CallerInformation,
        backward_projection_lock_guard: BackwardProjectionLockGuard<C>,
    ) {
        // SAFETY: We are reading our own backward edges, which we've already
        // acquired the lock for.
        let backward_edges = unsafe {
            self.engine().get_backward_edges_unchecked(self.query_id()).await
        };

        let mut backward_projections = Vec::new();
        for query_id in backward_edges {
            let query_kind = self.engine().get_query_kind(&query_id).await;

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
            let engine = self.engine().clone();
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

        self.done_backward_projection(backward_projection_lock_guard).await;
    }
}
