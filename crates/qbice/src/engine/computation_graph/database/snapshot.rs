use std::sync::Arc;

use qbice_stable_hash::Compact128;
use qbice_storage::{dynamic_map::DynamicMap as _, single_map::SingleMap as _};

use crate::{
    Config, Engine, Query,
    engine::computation_graph::{
        QueryKind,
        database::{
            Edge, ForwardEdgeObservation, ForwardEdgeOrder, LastVerified,
            NodeInfo, PendingBackwardProjection, QueryResult,
        },
        query_lock_manager::QueryLock,
    },
    query::QueryID,
};

/// A snapshot for repeatingly reading query data.
///
/// It holds either shared or exclusive lock as long as the snapshot is alive.
///
/// This allows us to minimize lock acquisition to the main database of the
/// engine.
#[allow(clippy::option_option)]
pub struct Snapshot<C: Config, Q: Query> {
    engine: Arc<Engine<C>>,
    lock: Option<QueryLock>,
    query_id: QueryID,

    last_verified: Option<Option<LastVerified>>,
    query_kind: Option<Option<QueryKind>>,
    node_info: Option<Option<NodeInfo>>,
    pending_backward_projection: Option<Option<PendingBackwardProjection>>,
    forward_edge_order: Option<Option<ForwardEdgeOrder>>,
    forward_edge_observation: Option<Option<ForwardEdgeObservation<C>>>,
    query_input: Option<Option<Q>>,
    query_result: Option<Option<Q::Value>>,
}

impl<C: Config> Engine<C> {
    /// Acquires a read snapshot for the given query.
    pub(crate) async fn get_read_snapshot<Q: Query>(
        self: &Arc<Self>,
        hash128: Compact128,
    ) -> Snapshot<C, Q> {
        let query_id = QueryID::new::<Q>(hash128);
        let lock = self
            .computation_graph
            .lock_manager
            .acquire_shared_lock(&query_id)
            .await;

        Snapshot {
            engine: self.clone(),
            lock: Some(lock),
            query_id,

            last_verified: None,
            query_kind: None,
            node_info: None,
            pending_backward_projection: None,
            forward_edge_order: None,
            forward_edge_observation: None,
            query_input: None,
            query_result: None,
        }
    }

    /// Acquires a read snapshot for the given query.
    pub(crate) async fn get_exclusive_snapshot<Q: Query>(
        self: &Arc<Self>,
        hash128: Compact128,
    ) -> Snapshot<C, Q> {
        let query_id = QueryID::new::<Q>(hash128);
        let lock = self
            .computation_graph
            .lock_manager
            .acquire_exclusive_lock(&query_id)
            .await;

        Snapshot {
            engine: self.clone(),
            lock: Some(lock),
            query_id,

            last_verified: None,
            query_kind: None,
            node_info: None,
            pending_backward_projection: None,
            forward_edge_order: None,
            forward_edge_observation: None,
            query_input: None,
            query_result: None,
        }
    }
}

impl<C: Config, Q: Query> Snapshot<C, Q> {
    pub async fn last_verified(&mut self) -> Option<LastVerified> {
        if let Some(opt) = &self.last_verified {
            return opt.clone();
        }

        let value = self
            .engine
            .computation_graph
            .database
            .last_verified
            .get(&self.query_id)
            .await;

        self.last_verified = Some(value.clone());

        value
    }

    pub async fn query_kind(&mut self) -> Option<QueryKind> {
        if let Some(opt) = &self.query_kind {
            return *opt;
        }

        let value = self
            .engine
            .computation_graph
            .database
            .query_kind
            .get(&self.query_id)
            .await;

        self.query_kind = Some(value);

        value
    }

    pub async fn node_info(&mut self) -> Option<NodeInfo> {
        if let Some(opt) = &self.node_info {
            return opt.clone();
        }

        let value = self
            .engine
            .computation_graph
            .database
            .node_info
            .get(&self.query_id)
            .await;

        self.node_info = Some(value.clone());

        value
    }

    pub async fn pending_backward_projection(
        &mut self,
    ) -> Option<PendingBackwardProjection> {
        if let Some(opt) = &self.pending_backward_projection {
            return opt.clone();
        }

        let value = self
            .engine
            .computation_graph
            .database
            .pending_backward_projection
            .get(&self.query_id)
            .await;

        self.pending_backward_projection = Some(value.clone());

        value
    }

    pub async fn forward_edge_order(&mut self) -> Option<ForwardEdgeOrder> {
        if let Some(opt) = &self.forward_edge_order {
            return opt.clone();
        }

        let value = self
            .engine
            .computation_graph
            .database
            .forward_edge_order
            .get(&self.query_id)
            .await;

        self.forward_edge_order = Some(value.clone());

        value
    }

    pub async fn forward_edge_observation(
        &mut self,
    ) -> Option<ForwardEdgeObservation<C>> {
        if let Some(opt) = &self.forward_edge_observation {
            return opt.clone();
        }

        let value = self
            .engine
            .computation_graph
            .database
            .forward_edge_observation
            .get(&self.query_id)
            .await;

        self.forward_edge_observation = Some(value.clone());

        value
    }

    pub async fn query_result(&mut self) -> Option<Q::Value> {
        if let Some(opt) = &self.query_result {
            return opt.clone();
        }

        let value = self
            .engine
            .computation_graph
            .database
            .query_store
            .get::<QueryResult<Q>>(&self.query_id.compact_hash_128())
            .await;

        self.query_result = Some(value.as_ref().map(|x| x.0.clone()));

        value.map(|x| x.0)
    }

    pub const fn query_id(&self) -> &QueryID { &self.query_id }

    pub const fn engine(&self) -> &Arc<Engine<C>> { &self.engine }

    pub async fn value_fingerprint(&mut self) -> Option<Compact128> {
        if let Some(opt) = &self.node_info {
            return opt.as_ref().map(|x| x.fingerprint);
        }

        let node_info = self.node_info().await;
        node_info.map(|x| x.fingerprint)
    }

    pub async fn is_edge_dirty(&mut self, callee: QueryID) -> bool {
        self.engine
            .computation_graph
            .database
            .dirty_edge_set
            .get(&Edge { from: self.query_id, to: callee })
            .await
            .is_some()
    }

    pub async fn upgrade_to_exclusive(&mut self) {
        if matches!(self.lock.as_ref(), Some(QueryLock::Exclusive(_))) {
            return;
        }

        let lock = self.lock.take().expect("snapshot lock must exist");
        drop(lock);

        let exclusive_lock = self
            .engine
            .computation_graph
            .lock_manager
            .acquire_exclusive_lock(&self.query_id)
            .await;

        self.lock = Some(exclusive_lock);

        // clear cached values, as they might be stale now
        self.last_verified = None;
        self.query_kind = None;
        self.node_info = None;
        self.pending_backward_projection = None;
        self.forward_edge_order = None;
        self.forward_edge_observation = None;
        self.query_input = None;
        self.query_result = None;
    }
}
