use std::sync::Arc;

use crate::{
    Engine, Query, config::Config,
    engine::computation_graph::caller::CallerInformation,
    executor::CyclicError, query::QueryID,
};

/// The result of attempting a fast-path query execution.
pub enum FastPathResult<V> {
    TryAgain,
    ToSlowPath,
    Hit(Option<V>),
}

impl<C: Config> Engine<C> {
    pub(super) async fn fast_path<Q: Query>(
        self: &Arc<Self>,
        query_id: &QueryID,
        caller: &CallerInformation,
    ) -> Result<FastPathResult<Q::Value>, CyclicError> {
        if let Some(computing) =
            self.computation_graph.computing_lock.get_lock(query_id)
        {
            let notified_owned = computing.notified_owned();
            drop(computing);

            notified_owned.await;

            Ok(FastPathResult::TryAgain)
        } else {
            // check if we have the existing query info
            let Some(query_info) =
                self.computation_graph.node_info.get_normal(query_id).await
            else {
                return Ok(FastPathResult::ToSlowPath);
            };

            // check if the query is up-to-date
            if query_info.last_verified != self.computation_graph.timestamp {
                return Ok(FastPathResult::ToSlowPath);
            }

            // gets the result
            if caller.require_value() {
                let Some(query_result) = self
                    .computation_graph
                    .query_store
                    .get_value::<Q>(&query_id.hash_128().into())
                    .await
                else {
                    return Ok(FastPathResult::ToSlowPath);
                };

                Ok(FastPathResult::Hit(Some(query_result)))
            } else {
                Ok(FastPathResult::Hit(None))
            }
        }
    }
}
