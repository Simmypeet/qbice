use crate::query::QueryID;

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
pub enum CallerInformation {
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

impl CallerInformation {
    pub const fn get_caller(&self) -> Option<QueryID> {
        match self {
            Self::RepairFirewall { .. }
            | Self::BackwardProjectionPropagation
            | Self::User => None,

            Self::Query(q) => Some(q.query_id),
        }
    }

    pub const fn require_value(&self) -> bool {
        match self {
            Self::RepairFirewall { .. }
            | Self::BackwardProjectionPropagation => false,

            Self::User => true,
            Self::Query(q) => matches!(q.reason, CallerReason::RequireValue),
        }
    }

    pub fn has_a_caller_requiring_value(&self) -> Option<&QueryID> {
        match self {
            // it does require value, but the caller is not another query
            Self::RepairFirewall { .. }
            | Self::User
            | Self::BackwardProjectionPropagation => None,

            Self::Query(q) => matches!(q.reason, CallerReason::RequireValue)
                .then_some(&q.query_id),
        }
    }
}
