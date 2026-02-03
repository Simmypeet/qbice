use std::sync::Arc;

use crate::{
    Engine, Query,
    config::Config,
    engine::computation_graph::{
        QueryKind,
        caller::{CallerInformation, CallerKind},
        lock::Computing,
        persist::NodeInfo,
        slow_path::SlowPath,
    },
    executor::CyclicError,
    query::QueryID,
};

/// The result of attempting a fast-path query execution.
pub enum FastPathResult<V> {
    TryAgain,
    ToSlowPath(SlowPath),
    Hit(Option<V>),
}

impl<C: Config> Engine<C> {
    /// Add both forward and backward dependencies for both caller and callee.
    fn observe_callee_fingerprint(
        &self,
        callee_info: &NodeInfo,
        callee_target: &QueryID,
        callee_kind: QueryKind,
        caller_source: &QueryID,
    ) {
        // add dependency for the caller
        let caller_computing =
            self.computation_graph.lock.get_lock(caller_source);

        caller_computing.observe_callee(
            callee_target,
            callee_info.value_fingerprint(),
            callee_info.transitive_firewall_callees_fingerprint(),
        );

        Self::caller_observe_tfc_callees(
            &caller_computing,
            callee_info,
            callee_kind,
            *callee_target,
        );
    }

    /// Exit early if a cyclic dependency is detected.
    fn exit_scc(
        &self,
        called_from: Option<&QueryID>,
        running_state: &Computing,
    ) -> Result<(), CyclicError> {
        // if there is no caller, we are at the root.
        let Some(called_from) = called_from else {
            return Ok(());
        };

        let is_in_scc = self.check_cyclic(running_state, called_from);

        // mark the caller as being in scc
        if is_in_scc {
            let computing = self.computation_graph.lock.get_lock(called_from);

            computing.mark_scc();

            return Err(CyclicError);
        }

        Ok(())
    }

    pub(super) async fn fast_path<Q: Query>(
        self: &Arc<Self>,
        query_id: &QueryID,
        caller: &CallerInformation,
    ) -> Result<FastPathResult<Q::Value>, CyclicError> {
        if let Some((notified_owned, computing)) =
            self.computation_graph.lock.try_get_lock_for_fast_path(query_id)
        {
            // exit out of the scc query to avoid circular waits
            self.exit_scc(caller.get_caller(), &computing)?;

            notified_owned.await;

            Ok(FastPathResult::TryAgain)
        } else {
            // check if we have the existing query info
            let (Some(query_info), Some(last_verified)) = (
                self.get_node_info(query_id).await,
                self.get_last_verified(query_id).await,
            ) else {
                return Ok(FastPathResult::ToSlowPath(SlowPath::Computing));
            };

            // check if the query is up-to-date
            if last_verified != caller.timestamp() {
                return Ok(FastPathResult::ToSlowPath(SlowPath::Computing));
            }

            // check if the query was called with repairing firewall and
            // has pending backward projection to do
            if matches!(caller.kind(), CallerKind::RepairFirewall {
                invoke_backward_projection: true
            }) {
                if let Some(pending_lock) = self
                    .computation_graph
                    .lock
                    .try_get_pending_backward_projection_lock(query_id)
                {
                    let notified_owned = pending_lock.notified_owned();
                    drop(pending_lock);

                    notified_owned.await;

                    return Ok(FastPathResult::TryAgain);
                } else if self
                    .get_pending_backward_projection(query_id)
                    .await
                    .is_some_and(|x| x == caller.timestamp())
                {
                    return Ok(FastPathResult::ToSlowPath(
                        SlowPath::BaackwardProjection,
                    ));
                }
            }

            // gets the result
            let query_result = if caller.require_value() {
                let Some(query_result) = self
                    .get_query_result::<Q>(&query_id.hash_128().into())
                    .await
                else {
                    return Ok(FastPathResult::ToSlowPath(SlowPath::Computing));
                };

                Some(query_result)
            } else {
                None
            };

            if let Some(caller_requiring_value) =
                caller.has_a_caller_requiring_value()
            {
                let kind = self.get_query_kind(query_id).await.unwrap();

                self.observe_callee_fingerprint(
                    &query_info,
                    query_id,
                    kind,
                    caller_requiring_value,
                );
            }

            Ok(FastPathResult::Hit(query_result))
        }
    }
}
