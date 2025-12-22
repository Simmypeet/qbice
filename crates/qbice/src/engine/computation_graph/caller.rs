use crate::query::QueryID;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallerReason {
    RequireValue,
    Repair,

    /// This occurs when a projection got invoked with
    /// `BackwardProjectionPropagation` and now the projection itself has to
    /// recompute and in the process it has to call another query with this
    /// flag.
    ///
    /// ```txt
    ///       Firewall1                Firewall2
    ///          ^                        ^
    ///          |                        |
    ///          +------- Projection  ----+
    ///                    ^^^^^^^
    ///                    Got backward propagated and now have to recompute
    ///                    itself and call Firewall1 and Firewall2 with
    ///                    this flag.
    /// ```
    ProjectionRecomputingDueToBackwardPropagation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueryCaller {
    query_id: QueryID,
    reason: CallerReason,
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

    RepairFirewall,
}

impl CallerInformation {
    pub const fn get_caller(&self) -> Option<&QueryID> {
        match self {
            Self::RepairFirewall
            | Self::BackwardProjectionPropagation
            | Self::User => None,

            Self::Query(q) => Some(&q.query_id),
        }
    }

    pub const fn require_value(&self) -> bool {
        match self {
            Self::RepairFirewall | Self::BackwardProjectionPropagation => false,

            Self::User => true,
            Self::Query(q) => matches!(q.reason, CallerReason::RequireValue
                | CallerReason::ProjectionRecomputingDueToBackwardPropagation
            ),
        }
    }

    pub fn has_a_caller_requiring_value(&self) -> Option<&QueryID> {
        match self {
            // it does require value, but the caller is not another query
            Self::RepairFirewall
            | Self::User
            | Self::BackwardProjectionPropagation => None,

            Self::Query(q) => {
                matches!(q.reason,
                    CallerReason::RequireValue
                    | CallerReason::ProjectionRecomputingDueToBackwardPropagation
                ).then_some(&q.query_id)
            }
        }
    }
}
