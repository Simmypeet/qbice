use crate::{
    Engine,
    config::Config,
    engine::computation_graph::{
        caller::{CallerInformation, CallerKind, QueryCaller},
        database::{NodeInfo, Snapshot, ValueSlot},
        slow_path::SlowPath,
    },
    query::QueryID,
};

/// The result of attempting a fast-path query execution.
pub enum FastPathResult {
    ToSlowPath(SlowPath),

    /// The query is up to date. If the caller requires the value, it has been
    /// read into the value slot.
    Hit,
}

impl<C: Config> Engine<C> {}

impl<C: Config> Snapshot<C> {
    /// `value_slot` must be given if the caller requires the value.
    pub async fn fast_path(
        &mut self,
        caller: &CallerInformation<C>,
        value_slot: Option<&mut ValueSlot<'_, C>>,
    ) -> FastPathResult {
        // check if we have the existing query info
        let Some(last_verified) = self.last_verified().await.map(|x| x.0)
        else {
            return FastPathResult::ToSlowPath(SlowPath::Compute);
        };

        if self.node_info().await.is_none() {
            return FastPathResult::ToSlowPath(SlowPath::Compute);
        }

        // check if the query is up-to-date
        if last_verified != caller.timestamp() {
            return FastPathResult::ToSlowPath(SlowPath::Repair);
        }

        // check if the query was called with repairing firewall and
        // has pending backward projection to do
        if matches!(
            caller.kind(),
            CallerKind::RepairFirewall
                | CallerKind::BackwardProjectionPropagation
        ) && self
            .pending_backward_projection()
            .await
            .is_some_and(|x| x.0 == caller.timestamp())
        {
            return FastPathResult::ToSlowPath(SlowPath::BaackwardProjection);
        }

        // gets the result
        if caller.require_value() {
            let value_slot = value_slot
                .expect("a caller that requires the value gives a slot for it");

            if !self.read_query_value(value_slot).await {
                return FastPathResult::ToSlowPath(SlowPath::Compute);
            }
        }

        if let Some(query_caller) = caller.get_query_caller()
            && query_caller.require_value()
        {
            let query_id = *self.query_id();
            let node_info = self
                .node_info()
                .await
                .expect("the node info has been found above");

            Self::observe_callee_fingerprint(
                query_caller,
                node_info,
                &query_id,
            );
        }

        FastPathResult::Hit
    }

    /// Add both forward and backward dependencies for both caller and callee.
    fn observe_callee_fingerprint(
        query_caller: &QueryCaller<C>,
        query_info: &NodeInfo,
        query_id: &QueryID,
    ) {
        // add dependency for the caller
        query_caller.computing().observe_callee(
            query_id,
            query_info.value_fingerprint(),
            query_info.transitive_firewall_callees_fingerprint(),
        );

        query_caller.computing().caller_observe_tfc_callees(
            query_info,
            query_info.kind(),
            *query_id,
        );
    }
}
