use std::sync::Arc;

use crate::{
    Config,
    engine::computation_graph::{
        ActiveComputationGuard,
        computing::QueryComputing,
        database::{ForwardEdgeObservation, Timestamp},
    },
    query::QueryID,
};

#[derive(Clone)]
pub enum CallerReason<C: Config> {
    RequireValue {
        _wait_guard: Option<waitgroup::Worker>,
        previous_caller_forward_edges: Option<ForwardEdgeObservation<C>>,
    },
    Repair,
}

impl<C: Config> std::fmt::Debug for CallerReason<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RequireValue { .. } => {
                f.debug_struct("RequireValue").finish_non_exhaustive()
            }
            Self::Repair => f.debug_tuple("Repair").finish(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct QueryCaller<C: Config> {
    query_id: QueryID,
    computing: Option<Arc<QueryComputing>>,
    reason: CallerReason<C>,

    /// Pedantic repair forces the repair process to eagerly recursively invoke
    /// `repair` on its callees, instead of checking the dirtiness of the
    /// callee and only invoking `repair` when the callee is dirty.
    pedantic_repair: bool,
}

impl<C: Config> QueryCaller<C> {
    pub const fn new_with_pedantic_repair(
        query_id: QueryID,
        reason: CallerReason<C>,
        computing: Arc<QueryComputing>,
        pedantic_repair: bool,
    ) -> Self {
        Self { query_id, computing: Some(computing), reason, pedantic_repair }
    }

    pub const fn new_external_input(
        query_id: QueryID,
        worker: waitgroup::Worker,
    ) -> Self {
        Self {
            query_id,
            computing: None,
            reason: CallerReason::RequireValue {
                _wait_guard: Some(worker),
                previous_caller_forward_edges: None,
            },
            pedantic_repair: false,
        }
    }

    pub fn is_in_the_previous_forward_edges(&self, callee: QueryID) -> bool {
        match &self.reason {
            CallerReason::RequireValue {
                _wait_guard,
                previous_caller_forward_edges: previous_forward_edges,
            } => previous_forward_edges
                .as_ref()
                .is_some_and(|x| x.0.contains_key(&callee)),
            // repairing checks the caller's old edges only; those callees'
            // firewalls are covered by the root user query's TFC repair
            CallerReason::Repair => true,
        }
    }

    pub const fn pedantic_repair(&self) -> bool { self.pedantic_repair }

    pub const fn try_computing(&self) -> Option<&Arc<QueryComputing>> {
        self.computing.as_ref()
    }

    #[must_use]
    pub const fn computing(&self) -> &Arc<QueryComputing> {
        self.computing
            .as_ref()
            .expect("`ExternalInput` cannot call other queries")
    }

    #[must_use]
    pub const fn query_id(&self) -> QueryID { self.query_id }

    #[must_use]
    pub const fn require_value(&self) -> bool {
        matches!(self.reason, CallerReason::RequireValue { .. })
    }
}

#[derive(Debug, Clone)]
pub struct CallerInformation<C: Config> {
    kind: CallerKind<C>,
    timestamp: Timestamp,
    active_computation_guard: Option<ActiveComputationGuard>,
}

impl<C: Config> CallerInformation<C> {
    pub const fn new(
        kind: CallerKind<C>,
        timestamp: Timestamp,
        active_computation_guard: Option<ActiveComputationGuard>,
    ) -> Self {
        Self { kind, timestamp, active_computation_guard }
    }

    #[must_use]
    pub fn clone_active_computation_guard(
        &self,
    ) -> Option<ActiveComputationGuard> {
        self.active_computation_guard.clone()
    }

    #[must_use]
    pub const fn active_computation_guard(
        &self,
    ) -> Option<&ActiveComputationGuard> {
        self.active_computation_guard.as_ref()
    }

    pub const fn get_query_caller(&self) -> Option<&QueryCaller<C>> {
        match &self.kind {
            CallerKind::RepairFirewall
            | CallerKind::BackwardProjectionPropagation
            | CallerKind::Tracing
            | CallerKind::User => None,

            CallerKind::Query(q) => Some(q),
        }
    }

    pub const fn require_value(&self) -> bool {
        match &self.kind {
            CallerKind::RepairFirewall
            | CallerKind::BackwardProjectionPropagation
            | CallerKind::Tracing => false,

            CallerKind::User => true,
            CallerKind::Query(q) => {
                matches!(q.reason, CallerReason::RequireValue { .. })
            }
        }
    }

    pub const fn timestamp(&self) -> Timestamp { self.timestamp }

    pub const fn kind(&self) -> &CallerKind<C> { &self.kind }
}

#[derive(Debug, Clone)]
pub enum CallerKind<C: Config> {
    User,
    Query(QueryCaller<C>),
    Tracing,

    /// The caller is either a firewall or projection query. The caller calls
    /// the query when the caller itself has changed its value and has to
    /// invoke all of its backward projections (projections that use the caller)
    /// in order to propagate the dirtiness.
    ///
    /// ```txt
    ///         Firewall <-- caller has changed its value
    ///          ^    ^
    ///         /      \
    ///   Projection1  Projection2 <-- both got called with
    ///                                 `BackwardProjectionPropagation`
    /// ```
    BackwardProjectionPropagation,

    RepairFirewall,
}
