use std::sync::Arc;

use dashmap::DashMap;

use crate::{
    Engine, Query, TrackedEngine,
    config::Config,
    engine::computation_graph::{
        QueryWithID,
        caller::{CallerInformation, CallerReason, QueryCaller},
        computing_lock::{ComputingLockGuard, ComputingMode},
    },
};

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
        lock_guard: ComputingLockGuard<'_>,
    ) {
        // create a new tracked engine
        let cache = Arc::new(DashMap::default());

        let reason = if *caller_information
            == CallerInformation::BackwardProjectionPropagation
        {
            CallerReason::ProjectionRecomputingDueToBackwardPropagation
        } else {
            CallerReason::RequireValue
        };

        let mut tracked_engine = TrackedEngine {
            engine: self.clone(),
            cache: cache.clone(),
            caller: CallerInformation::Query(QueryCaller::new(
                query.id, reason,
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

        let is_in_scc = self
            .computation_graph
            .computing_lock
            .get_lock(&query.id)
            .is_in_scc();

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

        match execute_query_for {
            ExecuteQueryFor::FreshQuery => {
                self.computing_lock_to_computed(query, value, lock_guard);
            }
            ExecuteQueryFor::RecomputeQuery => todo!("handle recompute query"),
        }
    }

    pub(super) async fn continuation<Q: Query>(
        self: &Arc<Self>,
        query: &QueryWithID<'_, Q>,
        caller_information: &CallerInformation,
        lock_guard: ComputingLockGuard<'_>,
    ) {
        if lock_guard.computing_mode() == ComputingMode::Execute {
            self.execute_query(
                query,
                caller_information,
                ExecuteQueryFor::FreshQuery,
                lock_guard,
            )
            .await;
        } else {
            todo!("haven't implemented")
        }
    }
}
