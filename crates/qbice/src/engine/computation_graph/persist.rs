use std::{
    borrow::Borrow, collections::HashMap, hash::Hash, marker::PhantomData,
    sync::Arc,
};

use dashmap::{DashMap, DashSet, setref::multiple::RefMulti};
use qbice_serialize::{Decode, Encode};
use qbice_stable_hash::{Compact128, StableHash};
use qbice_stable_type_id::{Identifiable, StableTypeID};
use qbice_storage::{
    intern::Interned,
    kv_database::{
        DiscriminantEncoding, KeyOfSetColumn, WideColumn, WideColumnValue,
    },
    sieve::{
        BackgroundWriter, KeyOfSetContainer, KeyOfSetSieve,
        RemoveElementFromSet, WideColumnSieve, WriteBuffer,
    },
};
use rayon::iter::IntoParallelRefIterator;

use crate::{
    Engine, ExecutionStyle, Query,
    config::Config,
    engine::computation_graph::{
        ComputationGraph, QueryKind, QueryWithID, Sieve,
        lock::BackwardProjectionLockGuard,
        tfc_achetype::TransitiveFirewallCallees,
    },
    query::QueryID,
};

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Identifiable,
)]
pub struct TimestampColumn;

impl WideColumn for TimestampColumn {
    type Key = ();

    type Discriminant = ();

    fn discriminant_encoding() -> DiscriminantEncoding {
        DiscriminantEncoding::Prefixed
    }
}

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Encode,
    Decode,
    Identifiable,
)]
pub struct Timestamp(u64);

impl WideColumnValue<TimestampColumn> for Timestamp {
    fn discriminant() {}
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Encode, Decode,
)]
pub enum QueryNodeDiscriminant {
    LastVerified,
    ForwardEdgeOrder,
    ForwardEdgeObservation,
    QueryKind,
    NodeInfo,
    PendingBackwardProjection,
}

impl QueryKind {
    pub const fn is_projection(self) -> bool {
        matches!(self, Self::Executable(ExecutionStyle::Projection))
    }
}

pub struct BackwardEdge<C: Config>(Arc<DashSet<QueryID, C::BuildHasher>>);

impl<C: Config> BackwardEdge<C> {
    pub fn par_iter(&self) -> dashmap::rayon::set::Iter<'_, QueryID> {
        self.0.par_iter()
    }

    pub fn iter(
        &self,
    ) -> dashmap::iter_set::Iter<
        '_,
        QueryID,
        C::BuildHasher,
        DashMap<QueryID, (), C::BuildHasher>,
    > {
        self.0.iter()
    }
}

impl<C: Config> std::fmt::Debug for BackwardEdge<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BackwardEdge")
            .field("edges", &self.0)
            .finish_non_exhaustive()
    }
}

impl<C: Config> Clone for BackwardEdge<C> {
    fn clone(&self) -> Self { Self(self.0.clone()) }
}

impl<C: Config> Encode for BackwardEdge<C> {
    fn encode<E: qbice_serialize::Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        plugin: &qbice_serialize::Plugin,
        session: &mut qbice_serialize::session::Session,
    ) -> std::io::Result<()> {
        (*self.0).encode(encoder, plugin, session)
    }
}

impl<C: Config> Decode for BackwardEdge<C> {
    fn decode<D: qbice_serialize::Decoder + ?Sized>(
        decoder: &mut D,
        plugin: &qbice_serialize::Plugin,
        session: &mut qbice_serialize::session::Session,
    ) -> std::io::Result<Self> {
        let set = DashSet::decode(decoder, plugin, session)?;
        Ok(Self(Arc::new(set)))
    }
}

impl<C: Config> Default for BackwardEdge<C> {
    fn default() -> Self { Self(Arc::new(DashSet::default())) }
}

impl<C: Config> Extend<QueryID> for BackwardEdge<C> {
    fn extend<T: IntoIterator<Item = QueryID>>(&mut self, iter: T) {
        for query_id in iter {
            self.0.insert(query_id);
        }
    }
}

impl<C: Config> FromIterator<QueryID> for BackwardEdge<C> {
    fn from_iter<T: IntoIterator<Item = QueryID>>(iter: T) -> Self {
        Self(Arc::new(FromIterator::from_iter(iter)))
    }
}

impl<'x, C: Config> IntoIterator for &'x BackwardEdge<C> {
    type Item = RefMulti<'x, QueryID>;
    type IntoIter = dashmap::iter_set::Iter<
        'x,
        QueryID,
        C::BuildHasher,
        DashMap<QueryID, (), C::BuildHasher>,
    >;

    fn into_iter(self) -> Self::IntoIter { self.0.iter() }
}

impl<C: Config> RemoveElementFromSet for BackwardEdge<C> {
    type Element = QueryID;

    fn remove_element<Q: Hash + Eq + ?Sized>(&mut self, element: &Q) -> bool
    where
        Self::Element: Borrow<Q>,
    {
        self.0.remove(element).is_some()
    }
}

#[derive(Identifiable)]
pub struct DirtySetColumn;

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Encode, Decode,
)]
pub struct Edge {
    from: QueryID,
    to: QueryID,
}

impl WideColumn for DirtySetColumn {
    type Key = Edge;
    type Discriminant = ();

    fn discriminant_encoding() -> DiscriminantEncoding {
        DiscriminantEncoding::Prefixed
    }
}

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    StableHash,
    Encode,
    Decode,
)]
pub struct Unit;

impl WideColumnValue<DirtySetColumn> for Unit {
    fn discriminant() {}
}

#[derive(Debug, Clone, PartialEq, Eq, Encode, Decode, Identifiable)]
pub struct NodeInfo {
    fingerprint: Compact128,
    transitive_firewall_callees_fingerprint: Compact128,
    transitive_firewall_callees: Option<Interned<TransitiveFirewallCallees>>,
}

impl NodeInfo {
    pub const fn new(
        fingerprint: Compact128,
        transitive_firewall_callees_fingerprint: Compact128,
        transitive_firewall_callees: Option<
            Interned<TransitiveFirewallCallees>,
        >,
    ) -> Self {
        Self {
            fingerprint,
            transitive_firewall_callees_fingerprint,
            transitive_firewall_callees,
        }
    }

    pub const fn value_fingerprint(&self) -> Compact128 { self.fingerprint }

    pub const fn transitive_firewall_callees_fingerprint(&self) -> Compact128 {
        self.transitive_firewall_callees_fingerprint
    }

    pub const fn transitive_firewall_callees(
        &self,
    ) -> Option<&Interned<TransitiveFirewallCallees>> {
        self.transitive_firewall_callees.as_ref()
    }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Identifiable,
)]
pub struct BackwardEdgeColumn<C: Config>(PhantomData<C>);

impl<C: Config> KeyOfSetColumn for BackwardEdgeColumn<C> {
    type Key = QueryID;

    type Element = QueryID;
}

impl<C: Config> KeyOfSetContainer for BackwardEdgeColumn<C> {
    type Container = BackwardEdge<C>;
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Identifiable,
)]
pub struct QueryNodeColumn;

impl WideColumn for QueryNodeColumn {
    type Key = QueryID;
    type Discriminant = QueryNodeDiscriminant;

    fn discriminant_encoding() -> DiscriminantEncoding {
        DiscriminantEncoding::Suffixed
    }
}

macro_rules! implements_wide_column_value_new_type {
    (
        $name:ident,
        $ty:ty,
        $discriminant:expr
    ) => {
        #[derive(Debug, Clone, Encode, Decode)]
        pub struct $name(pub $ty);

        impl WideColumnValue<QueryNodeColumn> for $name {
            fn discriminant() -> QueryNodeDiscriminant { $discriminant }
        }
    };
}

macro_rules! implements_wide_column_value {
    (
        $ty:ty,
        $discriminant:expr
    ) => {
        impl WideColumnValue<QueryNodeColumn> for $ty {
            fn discriminant() -> QueryNodeDiscriminant { $discriminant }
        }
    };
}

implements_wide_column_value_new_type!(
    LastVerified,
    Timestamp,
    QueryNodeDiscriminant::LastVerified
);

implements_wide_column_value_new_type!(
    ForwardEdgeOrder,
    Arc<[QueryID]>,
    QueryNodeDiscriminant::ForwardEdgeOrder
);

#[derive(Debug, Clone)]
pub struct ForwardEdgeObservation<C: Config>(
    Arc<HashMap<QueryID, Observation, C::BuildHasher>>,
);

impl<C: Config> WideColumnValue<QueryNodeColumn> for ForwardEdgeObservation<C> {
    fn discriminant() -> QueryNodeDiscriminant {
        QueryNodeDiscriminant::ForwardEdgeObservation
    }
}

impl<C: Config> Encode for ForwardEdgeObservation<C> {
    fn encode<E: qbice_serialize::Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        plugin: &qbice_serialize::Plugin,
        session: &mut qbice_serialize::session::Session,
    ) -> std::io::Result<()> {
        (*self.0).encode(encoder, plugin, session)
    }
}

impl<C: Config> Decode for ForwardEdgeObservation<C> {
    fn decode<D: qbice_serialize::Decoder + ?Sized>(
        decoder: &mut D,
        plugin: &qbice_serialize::Plugin,
        session: &mut qbice_serialize::session::Session,
    ) -> std::io::Result<Self> {
        let map = HashMap::decode(decoder, plugin, session)?;
        Ok(Self(Arc::new(map)))
    }
}

implements_wide_column_value!(QueryKind, QueryNodeDiscriminant::QueryKind);

implements_wide_column_value!(NodeInfo, QueryNodeDiscriminant::NodeInfo);

implements_wide_column_value_new_type!(
    PendingBackwardProjection,
    Timestamp,
    QueryNodeDiscriminant::PendingBackwardProjection
);

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Identifiable,
)]
pub struct QueryStoreColumn;

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Encode, Decode,
)]
pub enum QueryStoreDiscriminant {
    Input,
    Result,
}

impl WideColumn for QueryStoreColumn {
    type Key = Compact128;
    type Discriminant = (StableTypeID, QueryStoreDiscriminant);

    fn discriminant_encoding() -> DiscriminantEncoding {
        DiscriminantEncoding::Prefixed
    }
}

#[derive(Debug, Clone, Encode, Decode)]
pub struct QueryInput<Q: Query>(pub Q);

impl<Q: Query> WideColumnValue<QueryStoreColumn> for QueryInput<Q> {
    fn discriminant() -> (StableTypeID, QueryStoreDiscriminant) {
        (Q::STABLE_TYPE_ID, QueryStoreDiscriminant::Input)
    }
}

#[derive(Debug, Clone, Encode, Decode)]
pub struct QueryResult<Q: Query>(pub Q::Value);

impl<Q: Query> WideColumnValue<QueryStoreColumn> for QueryResult<Q> {
    fn discriminant() -> (StableTypeID, QueryStoreDiscriminant) {
        (Q::STABLE_TYPE_ID, QueryStoreDiscriminant::Result)
    }
}

pub struct Persist<C: Config> {
    query_node:
        Arc<WideColumnSieve<QueryNodeColumn, C::Database, C::BuildHasher>>,
    dirty_edge_set:
        Arc<WideColumnSieve<DirtySetColumn, C::Database, C::BuildHasher>>,
    query_store:
        Arc<WideColumnSieve<QueryStoreColumn, C::Database, C::BuildHasher>>,
    backward_edges:
        Arc<KeyOfSetSieve<BackwardEdgeColumn<C>, C::Database, C::BuildHasher>>,

    // need to be declared because of the the write buffer interface, we'll
    // fix this later
    timestamp_sieve:
        Arc<WideColumnSieve<TimestampColumn, C::Database, C::BuildHasher>>,

    background_writer: BackgroundWriter<C::Database, C::BuildHasher>,
}

impl<C: Config> Persist<C> {
    pub fn new(db: Arc<C::Database>, shard_amount: usize) -> Self {
        let background_writer =
            BackgroundWriter::<C::Database, C::BuildHasher>::new(8, db.clone());

        let timestamp_sieve =
            Arc::new(WideColumnSieve::<
                TimestampColumn,
                C::Database,
                C::BuildHasher,
            >::new(
                1, 1, db.clone(), C::BuildHasher::default()
            ));

        if timestamp_sieve.get_normal::<Timestamp>(()).is_none() {
            let mut tx = background_writer.new_write_buffer();
            timestamp_sieve.put((), Some(Timestamp(0)), &mut tx);
            background_writer.submit_write_buffer(tx);
        }

        Self {
            query_node: Arc::new(Sieve::<_, C>::new(
                C::cache_entry_capacity(),
                shard_amount,
                db.clone(),
                C::BuildHasher::default(),
            )),
            query_store: Arc::new(Sieve::<_, C>::new(
                C::cache_entry_capacity(),
                shard_amount,
                db.clone(),
                C::BuildHasher::default(),
            )),
            dirty_edge_set: Arc::new(Sieve::<_, C>::new(
                C::cache_entry_capacity(),
                shard_amount,
                db.clone(),
                C::BuildHasher::default(),
            )),
            backward_edges: Arc::new(Sieve::<_, C>::new(
                C::cache_entry_capacity(),
                shard_amount,
                db,
                C::BuildHasher::default(),
            )),
            timestamp_sieve,
            background_writer,
        }
    }
}

#[derive(Debug, Encode, Decode)]
pub struct Observation {
    pub seen_value_fingerprint: Compact128,
    pub seen_transitive_firewall_callees_fingerprint: Compact128,
}

impl<C: Config> ComputationGraph<C> {
    pub fn get_forward_edges_order(
        &self,
        query_id: QueryID,
    ) -> Option<Arc<[QueryID]>> {
        self.persist
            .query_node
            .get_normal::<ForwardEdgeOrder>(query_id)
            .map(|x| x.0.clone())
    }

    pub fn get_forward_edge_observations(
        &self,
        query_id: QueryID,
    ) -> Option<Arc<HashMap<QueryID, Observation, C::BuildHasher>>> {
        self.persist
            .query_node
            .get_normal::<ForwardEdgeObservation<C>>(query_id)
            .map(|x| x.0.clone())
    }

    pub fn get_node_info(&self, query_id: QueryID) -> Option<NodeInfo> {
        self.persist
            .query_node
            .get_normal::<NodeInfo>(query_id)
            .map(|x| x.clone())
    }

    pub fn get_query_kind(&self, query_id: QueryID) -> Option<QueryKind> {
        self.persist.query_node.get_normal::<QueryKind>(query_id).map(|x| *x)
    }

    pub fn get_last_verified(&self, query_id: QueryID) -> Option<Timestamp> {
        self.persist
            .query_node
            .get_normal::<LastVerified>(query_id)
            .map(|x| x.0)
    }

    pub fn get_backward_edges(&self, query_id: QueryID) -> BackwardEdge<C> {
        self.persist.backward_edges.get_set(&query_id).clone()
    }

    pub fn get_query_result<Q: Query>(
        &self,
        query_input_hash_128: Compact128,
    ) -> Option<Q::Value> {
        self.persist
            .query_store
            .get_normal::<QueryResult<Q>>(query_input_hash_128)
            .map(|x| x.0.clone())
    }

    pub fn get_pending_backward_projection(
        &self,
        query_id: QueryID,
    ) -> Option<Timestamp> {
        self.persist
            .query_node
            .get_normal::<PendingBackwardProjection>(query_id)
            .map(|x| x.0)
    }
}

impl<C: Config> Engine<C> {
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub(super) fn set_computed<Q: Query>(
        &self,
        query_id: &QueryWithID<'_, Q>,
        query_value: Q::Value,
        query_value_fingerprint: Option<Compact128>,
        query_kind: QueryKind,
        forward_edge_order: Arc<[QueryID]>,
        forward_edge_observations: HashMap<
            QueryID,
            Observation,
            C::BuildHasher,
        >,
        tfc_achetype: Option<Interned<TransitiveFirewallCallees>>,
        has_pending_backward_projection: bool,
        continuting_tx: Option<WriteBuffer<C::Database, C::BuildHasher>>,
    ) {
        let query_value_fingerprint =
            query_value_fingerprint.unwrap_or_else(|| self.hash(&query_value));
        let transitive_firewall_callees_fingerprint = self.hash(&tfc_achetype);

        let forward_edge_order = ForwardEdgeOrder(forward_edge_order);
        let forward_edge_observations =
            ForwardEdgeObservation::<C>(Arc::new(forward_edge_observations));

        let node_info = NodeInfo::new(
            query_value_fingerprint,
            transitive_firewall_callees_fingerprint,
            tfc_achetype,
        );
        let current_timestamp = self.get_current_timestamp();

        let existing_forward_edges =
            self.computation_graph.get_forward_edges_order(query_id.id);

        let query_input = QueryInput::<Q>(query_id.query.clone());
        let query_result = QueryResult::<Q>(query_value);

        let mut tx = continuting_tx.unwrap_or_else(|| {
            self.computation_graph.persist.background_writer.new_write_buffer()
        });

        {
            // remove prior backward edges
            if let Some(forward_edges) = &existing_forward_edges {
                for edge in forward_edges.iter() {
                    self.computation_graph
                        .persist
                        .backward_edges
                        .remove_set_element(edge, &query_id.id, &mut tx);
                }
            }

            // set pending backward projection if needed
            if has_pending_backward_projection {
                self.computation_graph.persist.query_node.put(
                    query_id.id,
                    Some(PendingBackwardProjection(current_timestamp)),
                    &mut tx,
                );
            }

            self.computation_graph.persist.query_node.put(
                query_id.id,
                Some(node_info),
                &mut tx,
            );

            self.computation_graph.persist.query_node.put(
                query_id.id,
                Some(query_kind),
                &mut tx,
            );

            self.computation_graph.persist.query_node.put(
                query_id.id,
                Some(LastVerified(current_timestamp)),
                &mut tx,
            );

            for edge in forward_edge_order.0.iter() {
                self.computation_graph
                    .persist
                    .backward_edges
                    .insert_set_element(edge, query_id.id, &mut tx);
            }

            self.computation_graph.persist.query_node.put(
                query_id.id,
                Some(forward_edge_order),
                &mut tx,
            );

            self.computation_graph.persist.query_node.put(
                query_id.id,
                Some(forward_edge_observations),
                &mut tx,
            );

            self.computation_graph.persist.query_store.put(
                query_id.id.compact_hash_128(),
                Some(query_input),
                &mut tx,
            );

            self.computation_graph.persist.query_store.put(
                query_id.id.compact_hash_128(),
                Some(query_result),
                &mut tx,
            );

            self.computation_graph
                .persist
                .background_writer
                .submit_write_buffer(tx);
        }
    }

    pub(super) fn set_computed_input<Q: Query>(
        &self,
        query: Q,
        query_hash_128: Compact128,
        query_value: Q::Value,
        query_value_fingerprint: Compact128,
        tx: &mut WriteBuffer<C::Database, C::BuildHasher>,
    ) {
        let query_id = QueryID::new::<Q>(query_hash_128);

        // if have an existing forward edges, unwire the backward edges
        let existing_forward_edges =
            self.computation_graph.get_forward_edges_order(query_id);

        let empty_forward_edges = ForwardEdgeOrder(Arc::from([]));
        let empty_forward_edge_observations = ForwardEdgeObservation::<C>(
            Arc::new(HashMap::with_hasher(C::BuildHasher::default())),
        );

        let transitive_firewall_callees = None;
        let transitive_firewall_callees_fingerprint =
            self.hash(&transitive_firewall_callees);

        let node_info = NodeInfo::new(
            query_value_fingerprint,
            transitive_firewall_callees_fingerprint,
            transitive_firewall_callees,
        );

        let timestamp = self.get_current_timestamp();

        let query_input = QueryInput::<Q>(query);
        let query_result = QueryResult::<Q>(query_value);

        {
            if let Some(forward_edges) = existing_forward_edges {
                for edge in forward_edges.iter() {
                    self.computation_graph
                        .persist
                        .backward_edges
                        .remove_set_element(edge, &query_id, tx);
                }
            }

            self.computation_graph.persist.query_node.put(
                query_id,
                Some(QueryKind::Input),
                tx,
            );

            self.computation_graph.persist.query_node.put(
                query_id,
                Some(LastVerified(timestamp)),
                tx,
            );

            self.computation_graph.persist.query_node.put(
                query_id,
                Some(empty_forward_edges),
                tx,
            );

            self.computation_graph.persist.query_node.put(
                query_id,
                Some(empty_forward_edge_observations),
                tx,
            );

            self.computation_graph.persist.query_node.put(
                query_id,
                Some(node_info),
                tx,
            );

            self.computation_graph.persist.query_store.put(
                query_id.compact_hash_128(),
                Some(query_input),
                tx,
            );

            self.computation_graph.persist.query_store.put(
                query_id.compact_hash_128(),
                Some(query_result),
                tx,
            );
        }
    }

    pub(super) fn mark_dirty_forward_edge(
        &self,
        from: QueryID,
        to: QueryID,
        tx: &mut WriteBuffer<C::Database, C::BuildHasher>,
    ) {
        let edge = Edge { from, to };

        self.computation_graph.persist.dirty_edge_set.put(edge, Some(Unit), tx);

        self.computation_graph.add_dirtied_edge_count();
    }

    pub(super) fn new_write_buffer(
        &self,
    ) -> WriteBuffer<C::Database, C::BuildHasher> {
        self.computation_graph.persist.background_writer.new_write_buffer()
    }

    #[allow(clippy::option_option)]
    pub(super) fn clean_query(
        &self,
        query_id: QueryID,
        clean_edges: &[QueryID],
        new_tfc: Option<Option<Interned<TransitiveFirewallCallees>>>,
    ) {
        let current_timestamp = self.get_current_timestamp();

        let new_node_info = new_tfc.map(|x| {
            let mut current_node_info =
                self.computation_graph.get_node_info(query_id).unwrap();

            current_node_info.transitive_firewall_callees = x;
            current_node_info.transitive_firewall_callees_fingerprint =
                self.hash(&current_node_info.transitive_firewall_callees);

            current_node_info
        });

        let mut tx =
            self.computation_graph.persist.background_writer.new_write_buffer();

        {
            for callee in clean_edges.iter().copied() {
                let edge = Edge { from: query_id, to: callee };

                self.computation_graph
                    .persist
                    .dirty_edge_set
                    .put::<Unit>(edge, None, &mut tx);
            }

            if let Some(node_info) = new_node_info {
                self.computation_graph.persist.query_node.put(
                    query_id,
                    Some(node_info),
                    &mut tx,
                );
            }

            self.computation_graph.persist.query_node.put(
                query_id,
                Some(LastVerified(current_timestamp)),
                &mut tx,
            );

            self.computation_graph
                .persist
                .background_writer
                .submit_write_buffer(tx);
        }
    }

    pub(super) fn is_edge_dirty(&self, from: QueryID, to: QueryID) -> bool {
        self.computation_graph
            .persist
            .dirty_edge_set
            .get_normal::<Unit>(Edge { from, to })
            .is_some()
    }

    pub(super) fn get_query_input<Q: Query>(
        &self,
        query_id: Compact128,
    ) -> Option<Q> {
        self.computation_graph
            .persist
            .query_store
            .get_normal::<QueryInput<Q>>(query_id)
            .map(|x| x.0.clone())
    }

    pub(super) fn done_backward_projection(
        &self,
        query_id: &QueryID,
        backward_projection_lock_guard: BackwardProjectionLockGuard<'_, C>,
    ) {
        let mut tx =
            self.computation_graph.persist.background_writer.new_write_buffer();

        self.computation_graph
            .persist
            .query_node
            .put::<PendingBackwardProjection>(*query_id, None, &mut tx);

        backward_projection_lock_guard.done();

        self.computation_graph
            .persist
            .background_writer
            .submit_write_buffer(tx);
    }
}

pub(crate) struct QueryDebug {
    pub type_name: &'static str,
    pub input: String,
    pub output: String,
}

impl<C: Config> Engine<C> {
    pub(crate) fn get_query_debug<Q: Query>(
        &self,
        query_id: Compact128,
    ) -> Option<QueryDebug> {
        let (Some(query_input), Some(query_value)) = (
            self.computation_graph
                .persist
                .query_store
                .get_normal::<QueryInput<Q>>(query_id),
            self.computation_graph
                .persist
                .query_store
                .get_normal::<QueryResult<Q>>(query_id),
        ) else {
            return None;
        };

        Some(QueryDebug {
            type_name: std::any::type_name::<Q>(),
            input: format!("{query_input:?}"),
            output: format!("{query_value:?}"),
        })
    }

    pub(super) fn get_current_timestamp(&self) -> Timestamp {
        *self
            .computation_graph
            .persist
            .timestamp_sieve
            .get_normal::<Timestamp>(())
            .expect("Timestamp should always be initialized")
    }

    pub(super) fn submit_write_buffer(
        &self,
        tx: WriteBuffer<C::Database, C::BuildHasher>,
    ) {
        self.computation_graph
            .persist
            .background_writer
            .submit_write_buffer(tx);
    }

    pub(super) fn increment_timestamp(
        &self,
        tx: &mut WriteBuffer<C::Database, C::BuildHasher>,
    ) -> Timestamp {
        let current = self.get_current_timestamp();
        let new_timestamp = Timestamp(current.0 + 1);

        self.computation_graph.persist.timestamp_sieve.put(
            (),
            Some(new_timestamp),
            tx,
        );

        new_timestamp
    }
}
