use std::sync::Arc;

use dashmap::DashMap;
use qbice_storage::kv_database::KvDatabase;

use crate::{
    Engine, Query, TrackedEngine,
    config::Config,
    engine::computation_graph::{
        QueryWithID,
        caller::{CallerInformation, CallerReason, QueryCaller},
        lock::{ComputingLockGuard, ComputingMode, LockGuard},
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

impl<C: Config> Engine<C> {
    pub(super) async fn execute_query<Q: Query>(
        self: &Arc<Self>,
        query: &QueryWithID<'_, Q>,
        caller_information: &CallerInformation,
        execute_query_for: ExecuteQueryFor,
        lock_guard: ComputingLockGuard<'_, C>,
    ) {
        // create a new tracked engine
        let cache = Arc::new(DashMap::default());

        let mut tracked_engine = TrackedEngine {
            engine: self.clone(),
            cache: cache.clone(),
            caller: CallerInformation::Query(QueryCaller::new(
                query.id,
                CallerReason::RequireValue,
            )),
        };

        let entry = self.executor_registry.get_executor_entry::<Q>();

        let result =
            entry.invoke_executor::<Q>(query.query, &mut tracked_engine).await;

        // use the `cache`'s strong count to determine if the tracked
        // engine is still held elsewhere other than the
        // current call stack.
        //
        // if there're still references to the `TrackedEngine`, it means
        // that there's some dangling references to the
        // `TrackedEngine` on some other threads that
        // the implementation of the query is not aware of.
        //
        // in this case, we'll panic to avoid silent bugs in the query
        // implementation.
        assert!(
            // 2 one for aliving tracked engine, and one for cache
            Arc::strong_count(&tracked_engine.cache) == 2,
            "`TrackedEngine` is still held elsewhere, this is a bug in the \
             query implementation which violates the query system's contract. \
             It's possible that the `TrackedEngine` is being sent to a \
             different thread and the query implementation hasn't properly \
             joined the thread before returning the value. Key: `{}`",
            std::any::type_name::<Q>()
        );

        let is_in_scc =
            self.computation_graph.lock.get_lock(query.id).is_in_scc();

        // if `is_in_scc` is `true`, it means that the query is part of
        // a strongly connected component (SCC) and the
        // value should be an error, otherwise, it
        // should be a valid value.

        assert_eq!(
            is_in_scc,
            result.is_err(),
            "Cyclic dependency state mismatch: expected {}, got {}",
            result.is_err(),
            is_in_scc
        );

        let value = result.unwrap_or_else(|_| entry.obtain_scc_value::<Q>());

        let old_kind = self.computation_graph.get_query_kind(query.id);

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
                self.computation_graph.get_node_info(query.id).expect(
                    "old node info should exist for recomputed firewall or \
                     projection",
                );

            let tx = self.database.write_transaction();

            let fingerprint = self.hash(&value);
            let updated = old_node_info.value_fingerprint() != fingerprint;

            // if fingerprint has changed, we do dirty propagation
            if updated {
                self.dirty_propagate(query.id, &tx);
            }

            (
                Some(tx),
                Some(fingerprint),
                // if the query is a firewall and its value has changed, it
                // needs to invoke projection queries in the backward direction
                // and propagate dirtiness as needed.
                old_kind.is_firewall() && updated,
            )
        } else {
            (None, None, false)
        };

        self.computing_lock_to_computed(
            query,
            value,
            query_value_fingerprint,
            lock_guard,
            need_backward_projection_propagation,
            continuing_tx,
        );

        // if the firewall is being repaired, and it has pending backward
        // projections, we need to do backward projections now.
        if matches!(caller_information, CallerInformation::RepairFirewall {
            invoke_backward_projection: true
        }) && need_backward_projection_propagation
        {
            self.try_do_backward_projections(query.id).await;
        }
    }

    pub(super) async fn continuation<Q: Query>(
        self: &Arc<Self>,
        query: &QueryWithID<'_, Q>,
        caller_information: &CallerInformation,
        lock_guard: LockGuard<'_, C>,
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
                self.invoke_backward_projections(query.id, lock_guard).await;
            }
        }
    }
}
