use qbice_stable_hash::Compact128;
use thread_local::ThreadLocal;
use tracing::instrument;

use crate::{
    Query, TrackedEngine,
    config::Config,
    engine::{
        computation_graph::{
            caller::{
                CallerInformation, CallerKind, CallerReason, QueryCaller,
            },
            computing::{ComputingLockGuard, ComputingMode, WriteGuard},
            database::{ForwardEdgeObservation, NodeInfo, Snapshot},
            statistics::Event,
        },
        guard::GuardExt,
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SlowPath {
    /// No entry found in the database, we need to compute the query and store
    /// it to the database.
    Compute,

    /// The query is found but stale due to timestamp check, we need to see
    /// whether we can repair it or need to recompute it.
    Repair,

    /// The query has "pending" backward projection to do.
    BaackwardProjection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ExecuteQueryFor {
    FreshQuery,
    RecomputeQuery,
}

impl<C: Config, Q: Query> Snapshot<C, Q> {
    #[allow(clippy::too_many_lines)]
    #[instrument(
        skip(self, execute_query_for, caller_information, lock_guard),
        level = "debug",
        name = "execute_query",
        target = "qbice"
    )]
    pub(super) async fn execute_query(
        mut self,
        query: &Q,
        execute_query_for: ExecuteQueryFor,
        previous_caller_forward_edges: Option<ForwardEdgeObservation<C>>,
        caller_information: &CallerInformation<C>,
        lock_guard: ComputingLockGuard<C>,
    ) {
        let wait_group = waitgroup::WaitGroup::new();

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
                    CallerReason::RequireValue {
                        _wait_guard: Some(wait_group.worker()),
                        previous_caller_forward_edges,
                    },
                    lock_guard.query_computing().clone(),
                    pedantic_repair,
                )),
                caller_information.timestamp(),
                caller_information.clone_active_computation_guard(),
            ),
        };

        let entry = self.engine().executor_registry.get_executor_entry::<Q>();

        let result = entry.invoke_executor::<Q>(query, &tracked_engine).await;

        // WAIT POINT: We must wait all the potentially spawned threads that
        // might hold references to the tracked engine to finish before
        // proceeding, to ensure that there are no more references that can
        // modify the query's state.
        drop(tracked_engine);
        wait_group.wait().await;

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
            // owned, because the snapshot is given away below while the
            // edges are still looked at
            let existing_forward_edges =
                self.forward_edge_order().await.cloned();

            // if the old node info is a firewall or projection node, we compare
            // the old and new value fingerprints to determine if we need to
            // do dirty propagation.
            let (
                continuing_tx,
                query_value_fingerprint,
                need_backward_projection_propagation,
            ) = if let Some(old_kind) = old_kind
                && (old_kind.is_firewall() || old_kind.is_projection())
                && matches!(execute_query_for, ExecuteQueryFor::RecomputeQuery)
            {
                let fingerprint = self.engine().hash(&value);

                let old_node_info = self.node_info().await.expect(
                    "old node info should exist for recomputed firewall or \
                     projection",
                );

                let updated = old_node_info.value_fingerprint() != fingerprint;

                self.record(if updated {
                    Event::RecomputedChanged
                } else {
                    Event::RecomputedUnchanged
                });

                // a changed firewall or projection is a root of dirty
                // propagation; an unchanged one stops it
                if updated {
                    tracing::debug!(
                        target: "qbice::dirty",
                        kind = ?old_kind,
                        query = ?query,
                        "recomputed to a new value, propagating dirtiness"
                    );
                } else {
                    tracing::trace!(
                        target: "qbice::dirty",
                        kind = ?old_kind,
                        query = ?query,
                        "recomputed to the same value, stopping dirtiness"
                    );
                }

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
                    (old_kind.is_firewall() || old_kind.is_projection())
                        && updated,
                )
            } else {
                let fingerprint =
                    self.record_execution(&value, execute_query_for).await;

                (self.engine().new_write_transaction(), fingerprint, false)
            };

            self.computing_lock_to_computed(
                query.clone(),
                value,
                query_value_fingerprint,
                lock_guard,
                need_backward_projection_propagation,
                timestamp,
                existing_forward_edges.as_ref().map(|x| x.0.as_ref()),
                // recompute query means, there were dirty edges from the old
                // value, so we need to clean them up. But for fresh query,
                // there is no dirty edge, so we can skip cleaning up.
                matches!(execute_query_for, ExecuteQueryFor::RecomputeQuery),
                continuing_tx,
            )
            .await;
        }
        .guarded() // make sure the future will eventually complete
        .await;
    }
}

impl<C: Config, Q: Query> Snapshot<C, Q> {
    /// Records `event` for the type of this query in the statistics.
    pub(super) fn record(&self, event: Event) {
        self.engine()
            .computation_graph
            .statistics_recorder
            .record(self.query_id().stable_type_id(), event);
    }

    /// Records the execution of this query, which gave `value`, in the
    /// statistics, unless it is a firewall or a projection executed again,
    /// whose change is recorded where it is decided.
    ///
    /// Returns the fingerprint of `value` if it had to compute it, to tell
    /// whether a query executed again changed, so that storing the value
    /// does not compute it again.
    async fn record_execution(
        &mut self,
        value: &Q::Value,
        execute_query_for: ExecuteQueryFor,
    ) -> Option<Compact128> {
        if !self.engine().computation_graph.statistics_recorder.is_enabled() {
            return None;
        }

        match execute_query_for {
            ExecuteQueryFor::FreshQuery => {
                self.record(Event::Fresh);
                None
            }

            ExecuteQueryFor::RecomputeQuery => {
                let fingerprint = self.engine().hash(value);
                let old_fingerprint =
                    self.node_info().await.map(NodeInfo::value_fingerprint);

                self.record(if old_fingerprint == Some(fingerprint) {
                    Event::RecomputedUnchanged
                } else {
                    Event::RecomputedChanged
                });

                Some(fingerprint)
            }
        }
    }

    #[instrument(
        skip(self, caller_information, lock_guard),
        level = "debug",
        target = "qbice"
    )]
    pub async fn process_query(
        self,
        query: &Q,
        caller_information: &CallerInformation<C>,
        lock_guard: WriteGuard<C>,
    ) {
        match lock_guard {
            WriteGuard::ComputingLockGuard(lock_guard) => {
                if lock_guard.computing_mode() == ComputingMode::Execute {
                    self.execute_query(
                        query,
                        ExecuteQueryFor::FreshQuery,
                        None,
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
