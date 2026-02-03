use std::sync::Arc;

use crossbeam::sync::WaitGroup;
use dashmap::DashSet;

use crate::{
    Engine, Query, TrackedEngine,
    config::Config,
    engine::{
        computation_graph::{
            QueryWithID,
            caller::{
                CallerInformation, CallerKind, CallerReason, QueryCaller,
            },
            lock::{ComputingLockGuard, ComputingMode, LockGuard},
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

impl<C: Config> Engine<C> {
    #[allow(clippy::too_many_lines)]
    pub(super) async fn execute_query<Q: Query>(
        self: &Arc<Self>,
        query: &QueryWithID<'_, Q>,
        caller_information: &CallerInformation,
        execute_query_for: ExecuteQueryFor,
        lock_guard: ComputingLockGuard<C>,
    ) {
        // create a new tracked engine
        let cache = Arc::new(DashSet::default());

        let wait_group = WaitGroup::new();

        let tracked_engine = TrackedEngine {
            engine: self.clone(),
            cache: cache.clone(),
            caller: CallerInformation::new(
                CallerKind::Query(QueryCaller::new(
                    query.id,
                    CallerReason::RequireValue(Some(wait_group)),
                )),
                caller_information.timestamp(),
                caller_information.clone_active_computation_guard(),
            ),
        };
        let guarded_tracked_engine = GuardedTrackedEngine { tracked_engine };

        let entry = self.executor_registry.get_executor_entry::<Q>();

        let result = entry
            .invoke_executor::<Q>(query.query, &guarded_tracked_engine)
            .await;

        // WAIT POINT: We must wait all the potentially spawned threads that
        // might hold references to the tracked engine to finish before
        // proceeding, to ensure that there are no more references that can
        // modify the query's state.
        drop(guarded_tracked_engine);

        let is_in_scc =
            self.computation_graph.lock.get_lock(&query.id).is_in_scc();

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

        let engine = self.clone();
        let query_id = query.id;
        let query = query.query.clone();
        let timestamp = caller_information.timestamp();
        let caller_information = caller_information.clone();

        async move {
            let old_kind = engine.get_query_kind(&query_id).await;
            let existing_forward_edges =
                engine.get_forward_edges_order(&query_id).await;

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
                let old_node_info =
                    engine.get_node_info(&query_id).await.expect(
                        "old node info should exist for recomputed firewall \
                         or projection",
                    );

                let fingerprint = engine.hash(&value);
                let updated = old_node_info.value_fingerprint() != fingerprint;

                let mut write_buffer = engine.new_write_transaction();

                // if fingerprint has changed, we do dirty propagation
                if updated {
                    write_buffer = engine
                        .dirty_propagate_from_batch(
                            std::iter::once(query_id),
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
                (engine.new_write_transaction(), None, false)
            };

            engine
                .computing_lock_to_computed(
                    query,
                    query_id,
                    value,
                    query_value_fingerprint,
                    lock_guard,
                    need_backward_projection_propagation,
                    timestamp,
                    existing_forward_edges
                        .as_ref()
                        .map(std::convert::AsRef::as_ref),
                    continuing_tx,
                )
                .await;

            // if the firewall is being repaired, and it has pending backward
            // projections, we need to do backward projections now.
            if matches!(caller_information.kind(), CallerKind::RepairFirewall {
                invoke_backward_projection: true
            }) && need_backward_projection_propagation
            {
                engine
                    .try_do_backward_projections(&query_id, &caller_information)
                    .await;
            }
        }
        .guarded() // make sure the future will eventually complete
        .await;
    }

    pub(super) async fn continuation<Q: Query>(
        self: &Arc<Self>,
        query: &QueryWithID<'_, Q>,
        caller_information: &CallerInformation,
        lock_guard: LockGuard<C>,
    ) {
        match lock_guard {
            LockGuard::ComputingLockGuard(lock_guard) => {
                if lock_guard.computing_mode() == ComputingMode::Execute {
                    self.execute_query(
                        query,
                        caller_information,
                        ExecuteQueryFor::FreshQuery,
                        lock_guard,
                    )
                    .await;
                } else {
                    self.repair_query(query, caller_information, lock_guard)
                        .await;
                }
            }

            LockGuard::BackwardProjectionLockGuard(lock_guard) => {
                self.invoke_backward_projections(
                    &query.id,
                    caller_information,
                    lock_guard,
                )
                .await;
            }
        }
    }
}
