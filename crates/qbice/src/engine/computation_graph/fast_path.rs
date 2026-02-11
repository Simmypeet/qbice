use crate::{
    Engine, Query,
    config::Config,
    engine::computation_graph::{
        QueryKind,
        caller::{CallerInformation, CallerKind, QueryCaller},
        database::{NodeInfo, Snapshot},
        slow_path::SlowPath,
    },
};

/// The result of attempting a fast-path query execution.
pub enum FastPathResult<V> {
    ToSlowPath(SlowPath),
    Hit(Option<V>),
}

impl<C: Config> Engine<C> {}

impl<C: Config, Q: Query> Snapshot<C, Q> {
    pub async fn fast_path(
        &mut self,
        caller: &CallerInformation,
    ) -> FastPathResult<Q::Value> {
        // check if we have the existing query info
        let last_verified = self.last_verified().await;
        let node_info = self.node_info().await;

        let (Some(node_info), Some(last_verified)) = (node_info, last_verified)
        else {
            return FastPathResult::ToSlowPath(SlowPath::Compute);
        };

        // check if the query is up-to-date
        if last_verified.0 != caller.timestamp() {
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
        let query_result = if caller.require_value() {
            let Some(query_result) = self.query_result().await else {
                return FastPathResult::ToSlowPath(SlowPath::Compute);
            };

            Some(query_result)
        } else {
            None
        };

        if let Some(query_caller) = caller.get_query_caller()
            && query_caller.require_value()
        {
            let kind = self.query_kind().await.unwrap();

            self.observe_callee_fingerprint(query_caller, &node_info, kind);
        }

        FastPathResult::Hit(query_result)
    }

    /// Add both forward and backward dependencies for both caller and callee.
    fn observe_callee_fingerprint(
        &mut self,
        query_caller: &QueryCaller,
        query_info: &NodeInfo,
        query_kind: QueryKind,
    ) {
        // add dependency for the caller
        query_caller.computing().observe_callee(
            self.query_id(),
            query_info.value_fingerprint(),
            query_info.transitive_firewall_callees_fingerprint(),
        );

        query_caller.computing().caller_observe_tfc_callees(
            query_info,
            query_kind,
            *self.query_id(),
        );
    }
}
