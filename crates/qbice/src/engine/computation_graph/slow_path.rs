use crossbeam::sync::WaitGroup;
use thread_local::ThreadLocal;

use crate::{
    Query, TrackedEngine,
    config::Config,
    engine::{
        computation_graph::{
            caller::{
                CallerInformation, CallerKind, CallerReason, QueryCaller,
            },
            computing::{ComputingLockGuard, ComputingMode, WriteGuard},
            database::Snapshot,
        },
        guard::GuardExt,
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SlowPath {
    Computing,
    BaackwardProjection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ExecuteQueryFor {
    FreshQuery,
    RecomputeQuery,
}

pub struct GuardedTrackedEngine<C: Config> {
    tracked_engine: TrackedEngine<C>,
}

impl<C: Config> GuardedTrackedEngine<C> {
    /// Creates a new `GuardedTrackedEngine` wrapping the given
    /// `TrackedEngine`.
    ///
    /// When dropped, this will wait for any spawned tasks associated with the
    /// tracked engine to complete.
    pub(crate) const fn new(tracked_engine: TrackedEngine<C>) -> Self {
        Self { tracked_engine }
    }

    pub const fn tracked_engine(&self) -> &TrackedEngine<C> {
        &self.tracked_engine
    }
}

impl<C: Config> Drop for GuardedTrackedEngine<C> {
    fn drop(&mut self) {
        if let Some(wait_group) = self.tracked_engine.caller.get_wait_group() {
            wait_group.wait();
        }
    }
}

impl<C: Config, Q: Query> Snapshot<C, Q> {
    #[allow(clippy::too_many_lines)]
    pub(super) async fn execute_query(
        mut self,
        query: &Q,
        execute_query_for: ExecuteQueryFor,
        caller_information: &CallerInformation,
        lock_guard: ComputingLockGuard<C>,
    ) {
        let wait_group = WaitGroup::new();

        let pedantic_repair = match caller_information.kind() {
            CallerKind::Query(query_caller) => query_caller.pedantic_repair(),
            CallerKind::BackwardProjectionPropagation => true,

            _ => false,
        };

        let tracked_engine = TrackedEngine {
            engine: self.engine().clone(),
            cache: ThreadLocal::new(),
            caller: CallerInformation::new(
                CallerKind::Query(QueryCaller::new_with_pedantic_repair(
                    *self.query_id(),
                    CallerReason::RequireValue(Some(wait_group)),
                    lock_guard.query_computing().clone(),
                    pedantic_repair,
                )),
                caller_information.timestamp(),
                caller_information.clone_active_computation_guard(),
            ),
        };
        let guarded_tracked_engine = GuardedTrackedEngine { tracked_engine };

        let entry = self.engine().executor_registry.get_executor_entry::<Q>();

        let result =
            entry.invoke_executor::<Q>(query, &guarded_tracked_engine).await;

        // WAIT POINT: We must wait all the potentially spawned threads that
        // might hold references to the tracked engine to finish before
        // proceeding, to ensure that there are no more references that can
        // modify the query's state.
        drop(guarded_tracked_engine);

        let is_in_scc = lock_guard.query_computing().is_in_scc();

        // if `is_in_scc` is `true`, it means that the query is part of
        // a strongly connected component (SCC) and the
        // value should be an error, otherwise, it
        // should be a valid value.

        let value = if is_in_scc {
            // obtain the SCC value
            entry.obtain_scc_value::<Q>()
        } else {
            match result {
                Ok(value) => value,
                Err(panic) => panic.resume_unwind(),
            }
        };

        // TRANSACTIONAL: from now on, we are starting to modify the query's
        // internal state, so we need to make sure that this whole block is
        // polled to the completion without being cancelled.
        //
        // We must also hold `caller_information` alive until the end of this
        // async block, because it's needed to hold `ActiveComputationGuard`

        let timestamp = caller_information.timestamp();
        let query = query.clone();

        async move {
            let old_kind = self.query_kind().await;
            let existing_forward_edges = self.forward_edge_order().await;

            // if the old node info is a firewall or projection node, we compare
            // the old and new value fingerprints to determine if we need to
            // do dirty propagation.
            let (
                continuing_tx,
                query_value_fingerprint,
                need_backward_projection_propagation,
            ) = if let Some(old_kind) = old_kind
                && (old_kind.is_firewall() || old_kind.is_projection())
                && execute_query_for == ExecuteQueryFor::RecomputeQuery
            {
                let old_node_info = self.node_info().await.expect(
                    "old node info should exist for recomputed firewall or \
                     projection",
                );

                let fingerprint = self.engine().hash(&value);
                let updated = old_node_info.value_fingerprint() != fingerprint;

                let mut write_buffer = self.engine().new_write_transaction();

                // if fingerprint has changed, we do dirty propagation
                if updated {
                    write_buffer = self
                        .engine()
                        .dirty_propagate_from_batch(
                            std::iter::once(*self.query_id()),
                            write_buffer,
                        )
                        .await;
                }

                (
                    write_buffer,
                    Some(fingerprint),
                    // if the query is a firewall and its value has changed, it
                    // needs to invoke projection queries in the backward
                    // direction and propagate dirtiness as
                    // needed.
                    old_kind.is_firewall() && updated,
                )
            } else {
                (self.engine().new_write_transaction(), None, false)
            };

            self.computing_lock_to_computed(
                query.clone(),
                value,
                query_value_fingerprint,
                lock_guard,
                need_backward_projection_propagation,
                timestamp,
                existing_forward_edges.as_ref().map(|x| x.0.as_ref()),
                continuing_tx,
            )
            .await;
        }
        .guarded() // make sure the future will eventually complete
        .await;
    }
}

impl<C: Config, Q: Query> Snapshot<C, Q> {
    pub async fn continuation(
        self,
        query: &Q,
        caller_information: &CallerInformation,
        lock_guard: WriteGuard<C>,
    ) {
        match lock_guard {
            WriteGuard::ComputingLockGuard(lock_guard) => {
                if lock_guard.computing_mode() == ComputingMode::Execute {
                    self.execute_query(
                        query,
                        ExecuteQueryFor::FreshQuery,
                        caller_information,
                        lock_guard,
                    )
                    .await;
                } else {
                    self.repair_query(query, caller_information, lock_guard)
                        .await;
                }
            }

            WriteGuard::BackwardProjectionLockGuard(lock_guard) => {
                self.invoke_backward_projections(
                    caller_information,
                    lock_guard,
                )
                .await;
            }
        }
    }
}
