use std::sync::Arc;

use crossbeam::sync::WaitGroup;

use crate::{
    engine::computation_graph::{
        ActiveComputationGuard, computing::QueryComputing, database::Timestamp,
    },
    query::QueryID,
};

#[derive(Debug, Clone)]
pub enum CallerReason {
    RequireValue(Option<WaitGroup>),
    Repair,
}

#[derive(Debug, Clone)]
pub struct QueryCaller {
    query_id: QueryID,
    computing: Option<Arc<QueryComputing>>,
    reason: CallerReason,
}

impl QueryCaller {
    pub const fn new(
        query_id: QueryID,
        reason: CallerReason,
        computing: Arc<QueryComputing>,
    ) -> Self {
        Self { query_id, computing: Some(computing), reason }
    }

    pub const fn new_external_input(
        query_id: QueryID,
        wait_group: WaitGroup,
    ) -> Self {
        Self {
            query_id,
            computing: None,
            reason: CallerReason::RequireValue(Some(wait_group)),
        }
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
pub struct CallerInformation {
    kind: CallerKind,
    timestamp: Timestamp,
    active_computation_guard: Option<ActiveComputationGuard>,
}

impl CallerInformation {
    pub const fn new(
        kind: CallerKind,
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

    pub const fn get_wait_group(&mut self) -> Option<WaitGroup> {
        match &mut self.kind {
            CallerKind::RepairFirewall { .. }
            | CallerKind::BackwardProjectionPropagation
            | CallerKind::Tracing
            | CallerKind::User => None,

            CallerKind::Query(q) => match &mut q.reason {
                CallerReason::RequireValue(wait_group) => wait_group.take(),
                CallerReason::Repair => None,
            },
        }
    }

    pub const fn get_query_caller(&self) -> Option<&QueryCaller> {
        match &self.kind {
            CallerKind::RepairFirewall { .. }
            | CallerKind::BackwardProjectionPropagation
            | CallerKind::Tracing
            | CallerKind::User => None,

            CallerKind::Query(q) => Some(q),
        }
    }

    pub const fn get_caller(&self) -> Option<&QueryID> {
        match &self.kind {
            CallerKind::RepairFirewall { .. }
            | CallerKind::BackwardProjectionPropagation
            | CallerKind::Tracing
            | CallerKind::User => None,

            CallerKind::Query(q) => Some(&q.query_id),
        }
    }

    pub const fn require_value(&self) -> bool {
        match &self.kind {
            CallerKind::RepairFirewall { .. }
            | CallerKind::BackwardProjectionPropagation
            | CallerKind::Tracing => false,

            CallerKind::User => true,
            CallerKind::Query(q) => {
                matches!(q.reason, CallerReason::RequireValue { .. })
            }
        }
    }

    pub const fn timestamp(&self) -> Timestamp { self.timestamp }

    pub const fn kind(&self) -> &CallerKind { &self.kind }
}

#[derive(Debug, Clone)]
pub enum CallerKind {
    User,
    Query(QueryCaller),
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

    RepairFirewall {
        invoke_backward_projection: bool,
    },
}
