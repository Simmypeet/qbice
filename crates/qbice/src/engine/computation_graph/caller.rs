use crate::{engine::computation_graph::persist::Timestamp, query::QueryID};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallerReason {
    RequireValue,
    Repair,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueryCaller {
    query_id: QueryID,
    reason: CallerReason,
}

impl QueryCaller {
    pub const fn new(query_id: QueryID, reason: CallerReason) -> Self {
        Self { query_id, reason }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CallerInformation {
    kind: CallerKind,
    timestamp: Timestamp,
}

impl CallerInformation {
    pub const fn new(kind: CallerKind, timestamp: Timestamp) -> Self {
        Self { kind, timestamp }
    }

    pub const fn get_caller(&self) -> Option<QueryID> {
        match &self.kind {
            CallerKind::RepairFirewall { .. }
            | CallerKind::BackwardProjectionPropagation
            | CallerKind::User => None,

            CallerKind::Query(q) => Some(q.query_id),
        }
    }

    pub const fn require_value(&self) -> bool {
        match &self.kind {
            CallerKind::RepairFirewall { .. }
            | CallerKind::BackwardProjectionPropagation => false,

            CallerKind::User => true,
            CallerKind::Query(q) => {
                matches!(q.reason, CallerReason::RequireValue)
            }
        }
    }

    pub fn has_a_caller_requiring_value(&self) -> Option<&QueryID> {
        match &self.kind {
            // it does require value, but the caller is not another query
            CallerKind::RepairFirewall { .. }
            | CallerKind::User
            | CallerKind::BackwardProjectionPropagation => None,

            CallerKind::Query(q) => {
                matches!(q.reason, CallerReason::RequireValue)
                    .then_some(&q.query_id)
            }
        }
    }

    pub const fn timestamp(&self) -> Timestamp { self.timestamp }

    pub const fn kind(&self) -> &CallerKind { &self.kind }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallerKind {
    User,
    Query(QueryCaller),

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
