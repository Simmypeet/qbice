use std::sync::Arc;

use qbice_stable_hash::Compact128;
use qbice_storage::single_map::SingleMap as _;

use crate::{
    Config, Engine,
    engine::computation_graph::{
        QueryKind,
        database::{
            ForwardEdgeObservation, ForwardEdgeOrder, ForwardEdges,
            LastVerified, NodeInfo, PendingBackwardProjection, ValueSlot,
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
///
/// # Queries that have never been computed
///
/// Everything the engine records about a query node is first written by one
/// write batch, under the exclusive lock of the query, and that batch always
/// writes the last verified timestamp and the node info. A snapshot that
/// finds either of the two missing therefore knows that none of the rest has
/// been written, and answers for it without asking the database again. A
/// query that is computed for the first time would otherwise ask the database
/// once for every piece it reads, and every one of those reads comes back
/// empty.
///
/// # No query type
///
/// The snapshot does not know the type of its query, and neither does the
/// code that runs a query with it. The steps that need the type go through the
/// registry entry of the query type, or through a function that the caller of
/// the query passes along. Being generic over the query type would copy all of
/// that code for every query type, in every crate that calls the query.
#[allow(clippy::option_option)]
pub struct Snapshot<C: Config> {
    engine: Arc<Engine<C>>,
    lock: Option<QueryLock>,
    query_id: QueryID,

    last_verified: Option<Option<LastVerified>>,
    node_info: Option<Option<NodeInfo>>,
    pending_backward_projection: Option<Option<PendingBackwardProjection>>,
    forward_edges: Option<Option<ForwardEdges<C>>>,
}

impl<C: Config> Engine<C> {
    /// Acquires a read snapshot for the given query.
    pub(crate) async fn get_read_snapshot(
        self: &Arc<Self>,
        query_id: QueryID,
    ) -> Snapshot<C> {
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
            node_info: None,
            pending_backward_projection: None,
            forward_edges: None,
        }
    }

    /// Acquires a read snapshot for the given query.
    pub(crate) async fn get_exclusive_snapshot(
        self: &Arc<Self>,
        query_id: QueryID,
    ) -> Snapshot<C> {
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
            node_info: None,
            pending_backward_projection: None,
            forward_edges: None,
        }
    }
}

impl<C: Config> Snapshot<C> {
    /// Records that nothing has been written about the query node, so that
    /// none of it is looked up.
    fn set_never_computed(&mut self) {
        self.last_verified = Some(None);
        self.node_info = Some(None);
        self.pending_backward_projection = Some(None);
        self.forward_edges = Some(None);
    }

    pub async fn last_verified(&mut self) -> Option<&LastVerified> {
        if self.last_verified.is_none() {
            let value = self
                .engine
                .computation_graph
                .database
                .last_verified
                .get(&self.query_id)
                .await;

            if value.is_none() {
                self.set_never_computed();
            } else {
                self.last_verified = Some(value);
            }
        }

        self.last_verified.as_ref().and_then(Option::as_ref)
    }

    /// The kind of the query, which is a part of its node info.
    pub async fn query_kind(&mut self) -> Option<QueryKind> {
        self.node_info().await.map(NodeInfo::kind)
    }

    pub async fn node_info(&mut self) -> Option<&NodeInfo> {
        if self.node_info.is_none() {
            let value = self
                .engine
                .computation_graph
                .database
                .node_info
                .get(&self.query_id)
                .await;

            if value.is_none() {
                self.set_never_computed();
            } else {
                self.node_info = Some(value);
            }
        }

        self.node_info.as_ref().and_then(Option::as_ref)
    }

    pub async fn pending_backward_projection(
        &mut self,
    ) -> Option<&PendingBackwardProjection> {
        if self.pending_backward_projection.is_none() {
            let value = self
                .engine
                .computation_graph
                .database
                .pending_backward_projection
                .get(&self.query_id)
                .await;

            self.pending_backward_projection = Some(value);
        }

        self.pending_backward_projection.as_ref().and_then(Option::as_ref)
    }

    /// The dependencies of the query and what it saw of them, which are
    /// stored together.
    async fn forward_edges(&mut self) -> Option<&ForwardEdges<C>> {
        if self.forward_edges.is_none() {
            let value = self
                .engine
                .computation_graph
                .database
                .forward_edges
                .get(&self.query_id)
                .await;

            self.forward_edges = Some(value);
        }

        self.forward_edges.as_ref().and_then(Option::as_ref)
    }

    pub async fn forward_edge_order(&mut self) -> Option<&ForwardEdgeOrder> {
        self.forward_edges().await.map(|x| &x.order)
    }

    pub async fn forward_edge_observation(
        &mut self,
    ) -> Option<&ForwardEdgeObservation<C>> {
        self.forward_edges().await.map(|x| &x.observation)
    }

    /// Reads the value of the query into `slot`. Returns whether the value was
    /// found.
    pub async fn read_query_value(&self, slot: &mut ValueSlot<'_, C>) -> bool {
        slot.read(&self.engine, &self.query_id.compact_hash_128()).await
    }

    pub const fn query_id(&self) -> &QueryID { &self.query_id }

    pub const fn engine(&self) -> &Arc<Engine<C>> { &self.engine }

    pub async fn value_fingerprint(&mut self) -> Option<Compact128> {
        self.node_info().await.map(NodeInfo::value_fingerprint)
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
        self.node_info = None;
        self.pending_backward_projection = None;
        self.forward_edges = None;
    }
}
