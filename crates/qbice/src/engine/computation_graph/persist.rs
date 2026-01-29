//! A module for persisting the computation graph state.
//!
//! # Atomic Units
//!
//! All the updates to the computation graph are done in atomic units called
//! `WriteBuffer`s.
//!
//! # Cancellation and Timestamps
//!
//! The current policy is that when a new session starts (when updating inputs),
//! the global timestamp is incremented, the computation graph goes to a new
//! generation. Any in-progress computations from prior generations are
//! cancelled.
//!
//! We achieve this by associating all running computations with its timestamp.
//! When that particular computation tries to read-from/write-to the computation
//! graph, it checks whether its timestamp matches the current timestamp of
//! the computation graph. If not, it aborts the computation early.
//!
//! The abort can happen in two ways:
//!
//! - Make the computation stuck in the `Pending` state forever. This requires
//!   user cooperation to drop the computation eventually.
//! - Panic the computation. This is a more aggressive approach, but it ensures
//!   that resources are freed up quickly.

use std::{
    borrow::Borrow, collections::HashMap, hash::Hash, marker::PhantomData,
    pin::Pin, sync::Arc,
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
        ConcurrentSet, KeyOfSetContainer, KeyOfSetSieve, WideColumnSieve,
        WriteBuffer,
    },
};
use rayon::iter::IntoParallelRefIterator;
pub(in crate::engine::computation_graph) use sync::WriterBufferWithLock;

use crate::{
    Engine, ExecutionStyle, Query,
    config::Config,
    engine::computation_graph::{
        CallerInformation, QueryKind, QueryWithID, Sieve,
        lock::BackwardProjectionLockGuard,
        tfc_achetype::TransitiveFirewallCallees,
    },
    query::QueryID,
};

mod sync;

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

impl<C: Config> ConcurrentSet for BackwardEdge<C> {
    type Element = QueryID;

    fn insert_element(&self, element: Self::Element) { self.0.insert(element); }

    fn remove_element<Q: Hash + Eq + ?Sized>(&self, element: &Q) -> bool
    where
        Self::Element: Borrow<Q>,
    {
        self.0.remove(element).is_some()
    }
}

/// Container for storing query hashes (Compact128) that were computed
/// with [`ExecutionStyle::ExternalInput`] for a specific query type.
///
/// This enables refreshing all external input queries of a given type
/// during an input session.
pub struct ExternalInputSet<C: Config>(
    Arc<DashSet<Compact128, C::BuildHasher>>,
);

impl<C: Config> ExternalInputSet<C> {
    /// Returns an iterator over all query hashes in this set.
    pub fn iter(
        &self,
    ) -> dashmap::iter_set::Iter<
        '_,
        Compact128,
        C::BuildHasher,
        DashMap<Compact128, (), C::BuildHasher>,
    > {
        self.0.iter()
    }
}

impl<C: Config> std::fmt::Debug for ExternalInputSet<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExternalInputSet")
            .field("queries", &self.0)
            .finish_non_exhaustive()
    }
}

impl<C: Config> Clone for ExternalInputSet<C> {
    fn clone(&self) -> Self { Self(self.0.clone()) }
}

impl<C: Config> Encode for ExternalInputSet<C> {
    fn encode<E: qbice_serialize::Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        plugin: &qbice_serialize::Plugin,
        session: &mut qbice_serialize::session::Session,
    ) -> std::io::Result<()> {
        (*self.0).encode(encoder, plugin, session)
    }
}

impl<C: Config> Decode for ExternalInputSet<C> {
    fn decode<D: qbice_serialize::Decoder + ?Sized>(
        decoder: &mut D,
        plugin: &qbice_serialize::Plugin,
        session: &mut qbice_serialize::session::Session,
    ) -> std::io::Result<Self> {
        let set = DashSet::decode(decoder, plugin, session)?;
        Ok(Self(Arc::new(set)))
    }
}

impl<C: Config> Default for ExternalInputSet<C> {
    fn default() -> Self { Self(Arc::new(DashSet::default())) }
}

impl<C: Config> Extend<Compact128> for ExternalInputSet<C> {
    fn extend<T: IntoIterator<Item = Compact128>>(&mut self, iter: T) {
        for hash in iter {
            self.0.insert(hash);
        }
    }
}

impl<C: Config> FromIterator<Compact128> for ExternalInputSet<C> {
    fn from_iter<T: IntoIterator<Item = Compact128>>(iter: T) -> Self {
        Self(Arc::new(FromIterator::from_iter(iter)))
    }
}

impl<'x, C: Config> IntoIterator for &'x ExternalInputSet<C> {
    type Item = RefMulti<'x, Compact128>;
    type IntoIter = dashmap::iter_set::Iter<
        'x,
        Compact128,
        C::BuildHasher,
        DashMap<Compact128, (), C::BuildHasher>,
    >;

    fn into_iter(self) -> Self::IntoIter { self.0.iter() }
}

impl<C: Config> ConcurrentSet for ExternalInputSet<C> {
    type Element = Compact128;

    fn insert_element(&self, element: Self::Element) { self.0.insert(element); }

    fn remove_element<Q: Hash + Eq + ?Sized>(&self, element: &Q) -> bool
    where
        Self::Element: Borrow<Q>,
    {
        self.0.remove(element).is_some()
    }
}

/// Column marker for storing external input queries grouped by their type.
///
/// Maps [`StableTypeID`] (query type) → `HashSet<Compact128>` (query hashes).
/// This allows retrieving all external input queries of a specific type
/// for refreshing during input sessions.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Identifiable,
)]
pub struct ExternalInputColumn<C: Config>(PhantomData<C>);

impl<C: Config> KeyOfSetColumn for ExternalInputColumn<C> {
    type Key = StableTypeID;
    type Element = Compact128;
}

impl<C: Config> KeyOfSetContainer for ExternalInputColumn<C> {
    type Container = ExternalInputSet<C>;
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
    /// Stores external input queries grouped by their type.
    ///
    /// Maps [`StableTypeID`] → `HashSet<Compact128>` for all queries computed
    /// with [`ExecutionStyle::ExternalInput`]. Used by `InputSession::refresh`
    /// to find and re-invoke all external input queries of a specific type.
    external_input_queries:
        Arc<KeyOfSetSieve<ExternalInputColumn<C>, C::Database, C::BuildHasher>>,

    sync: sync::Sync<C>,
}

impl<C: Config> Persist<C> {
    pub fn new(db: Arc<C::Database>, shard_amount: usize) -> Self {
        Self {
            sync: sync::Sync::new(&db),
            query_node: Arc::new(Sieve::<_, C>::new(
                C::cache_entry_capacity() * 2 * 2 * 2,
                shard_amount * 2 * 2,
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
                db.clone(),
                C::BuildHasher::default(),
            )),
            external_input_queries: Arc::new(Sieve::<_, C>::new(
                C::cache_entry_capacity(),
                shard_amount,
                db,
                C::BuildHasher::default(),
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, Encode, Decode)]
pub struct Observation {
    pub seen_value_fingerprint: Compact128,
    pub seen_transitive_firewall_callees_fingerprint: Compact128,
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
        caller_information: &CallerInformation,
        existing_forward_edges: Option<&[QueryID]>,
        mut tx: WriterBufferWithLock<C>,
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

        let query_input = QueryInput::<Q>(query_id.query.clone());
        let query_result = QueryResult::<Q>(query_value);

        {
            // remove prior backward edges
            if let Some(forward_edges) = existing_forward_edges {
                for edge in forward_edges {
                    self.computation_graph
                        .persist
                        .backward_edges
                        .remove_set_element(
                            edge,
                            &query_id.id,
                            tx.writer_buffer(),
                        );
                }
            }

            // set pending backward projection if needed
            if has_pending_backward_projection {
                self.computation_graph.persist.query_node.put(
                    query_id.id,
                    Some(PendingBackwardProjection(
                        caller_information.timestamp(),
                    )),
                    tx.writer_buffer(),
                );
            }

            self.computation_graph.persist.query_node.put(
                query_id.id,
                Some(node_info),
                tx.writer_buffer(),
            );

            self.computation_graph.persist.query_node.put(
                query_id.id,
                Some(query_kind),
                tx.writer_buffer(),
            );

            self.computation_graph.persist.query_node.put(
                query_id.id,
                Some(LastVerified(caller_information.timestamp())),
                tx.writer_buffer(),
            );

            for edge in forward_edge_order.0.iter() {
                self.computation_graph
                    .persist
                    .backward_edges
                    .insert_set_element(edge, query_id.id, tx.writer_buffer());
            }

            self.computation_graph.persist.query_node.put(
                query_id.id,
                Some(forward_edge_order),
                tx.writer_buffer(),
            );

            self.computation_graph.persist.query_node.put(
                query_id.id,
                Some(forward_edge_observations),
                tx.writer_buffer(),
            );

            self.computation_graph.persist.query_store.put(
                query_id.id.compact_hash_128(),
                Some(query_input),
                tx.writer_buffer(),
            );

            self.computation_graph.persist.query_store.put(
                query_id.id.compact_hash_128(),
                Some(query_result),
                tx.writer_buffer(),
            );

            // Track external input queries by type for refresh support
            if query_kind.is_external_input() {
                self.computation_graph
                    .persist
                    .external_input_queries
                    .insert_set_element(
                        &query_id.id.stable_type_id(),
                        query_id.id.compact_hash_128(),
                        tx.writer_buffer(),
                    );
            }

            self.submit_write_buffer(tx);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) async fn set_computed_input<Q: Query>(
        &self,
        query: Q,
        query_hash_128: Compact128,
        query_value: Q::Value,
        query_value_fingerprint: Compact128,
        tx: &mut WriterBufferWithLock<C>,
        set_input: bool,
        timestamp: Timestamp,
    ) {
        let query_id = QueryID::new::<Q>(query_hash_128);

        // if have an existing forward edges, unwire the backward edges
        let existing_forward_edges =
            unsafe { self.get_forward_edges_order_unchecked(query_id).await };

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

        let query_input = QueryInput::<Q>(query);
        let query_result = QueryResult::<Q>(query_value);

        // NOTE: No more async points below here, to ensure atomicity

        {
            if let Some(forward_edges) = existing_forward_edges {
                for edge in forward_edges.iter() {
                    self.computation_graph
                        .persist
                        .backward_edges
                        .remove_set_element(
                            edge,
                            &query_id,
                            tx.writer_buffer(),
                        );
                }
            }

            if set_input {
                self.computation_graph.persist.query_node.put(
                    query_id,
                    Some(QueryKind::Input),
                    tx.writer_buffer(),
                );
            }

            self.computation_graph.persist.query_node.put(
                query_id,
                Some(LastVerified(timestamp)),
                tx.writer_buffer(),
            );

            self.computation_graph.persist.query_node.put(
                query_id,
                Some(empty_forward_edges),
                tx.writer_buffer(),
            );

            self.computation_graph.persist.query_node.put(
                query_id,
                Some(empty_forward_edge_observations),
                tx.writer_buffer(),
            );

            self.computation_graph.persist.query_node.put(
                query_id,
                Some(node_info),
                tx.writer_buffer(),
            );

            self.computation_graph.persist.query_store.put(
                query_id.compact_hash_128(),
                Some(query_input),
                tx.writer_buffer(),
            );

            self.computation_graph.persist.query_store.put(
                query_id.compact_hash_128(),
                Some(query_result),
                tx.writer_buffer(),
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

    #[allow(clippy::option_option)]
    pub(super) async fn clean_query(
        &self,
        query_id: QueryID,
        clean_edges: &[QueryID],
        new_tfc: Option<Option<Interned<TransitiveFirewallCallees>>>,
        caller_information: &CallerInformation,
    ) {
        let new_node_info = if let Some(x) = new_tfc {
            let mut current_node_info =
                self.get_node_info(query_id, caller_information).await.unwrap();

            current_node_info.transitive_firewall_callees = x;
            current_node_info.transitive_firewall_callees_fingerprint =
                self.hash(&current_node_info.transitive_firewall_callees);

            Some(current_node_info)
        } else {
            None
        };

        let mut tx = self.new_write_buffer(caller_information).await;

        {
            for callee in clean_edges.iter().copied() {
                let edge = Edge { from: query_id, to: callee };

                self.computation_graph.persist.dirty_edge_set.put::<Unit>(
                    edge,
                    None,
                    tx.writer_buffer(),
                );
            }

            if let Some(node_info) = new_node_info {
                self.computation_graph.persist.query_node.put(
                    query_id,
                    Some(node_info),
                    tx.writer_buffer(),
                );
            }

            self.computation_graph.persist.query_node.put(
                query_id,
                Some(LastVerified(caller_information.timestamp())),
                tx.writer_buffer(),
            );

            self.submit_write_buffer(tx);
        }
    }

    pub(super) async fn is_edge_dirty(
        &self,
        from: QueryID,
        to: QueryID,
    ) -> bool {
        self.computation_graph
            .persist
            .dirty_edge_set
            .get_normal::<Unit>(Edge { from, to })
            .await
            .is_some()
    }

    pub(super) async fn done_backward_projection(
        &self,
        query_id: &QueryID,
        caller_information: &CallerInformation,
        backward_projection_lock_guard: BackwardProjectionLockGuard<'_, C>,
    ) {
        let mut tx = self.new_write_buffer(caller_information).await;

        self.computation_graph
            .persist
            .query_node
            .put::<PendingBackwardProjection>(
                *query_id,
                None,
                tx.writer_buffer(),
            );

        backward_projection_lock_guard.done();

        self.submit_write_buffer(tx);
    }
}

pub(crate) struct QueryDebug {
    pub type_name: &'static str,
    pub input: String,
    pub output: String,
}

impl<C: Config> Engine<C> {
    pub(crate) async fn get_query_debug<Q: Query>(
        &self,
        query_id: Compact128,
    ) -> Option<QueryDebug> {
        let (Some(query_input), Some(query_value)) = (
            self.computation_graph
                .persist
                .query_store
                .get_normal::<QueryInput<Q>>(query_id)
                .await,
            self.computation_graph
                .persist
                .query_store
                .get_normal::<QueryResult<Q>>(query_id)
                .await,
        ) else {
            return None;
        };

        Some(QueryDebug {
            type_name: std::any::type_name::<Q>(),
            input: format!("{query_input:?}"),
            output: format!("{query_value:?}"),
        })
    }

    pub(crate) fn get_query_debug_future<'s, Q: Query>(
        &'s self,
        query_id: Compact128,
    ) -> Pin<Box<dyn std::future::Future<Output = Option<QueryDebug>> + 's>>
    {
        Box::pin(async move { self.get_query_debug::<Q>(query_id).await })
    }
}
