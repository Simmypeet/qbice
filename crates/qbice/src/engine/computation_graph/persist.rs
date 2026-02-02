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
    collections::HashMap,
    hash::{BuildHasher, Hash},
    marker::PhantomData,
    pin::Pin,
    sync::Arc,
};

use dashmap::DashSet;
use parking_lot::RwLock;
use qbice_serialize::{Decode, Encode};
use qbice_stable_hash::{Compact128, StableHash};
use qbice_stable_type_id::{Identifiable, StableTypeID};
use qbice_storage::{
    dynamic_map::DynamicMap as _,
    intern::Interned,
    key_of_set_map::{ConcurrentSet, KeyOfSetMap as _},
    kv_database::{
        DiscriminantEncoding, KeyOfSetColumn, WideColumn, WideColumnValue,
    },
    single_map::SingleMap as _,
    storage_engine::StorageEngine,
};
pub(super) use sync::ActiveComputationGuard;
pub(crate) use sync::ActiveInputSessionGuard;

use crate::{
    Engine, ExecutionStyle, Query,
    config::Config,
    engine::computation_graph::{
        CallerInformation, QueryKind, lock::BackwardProjectionLockGuard,
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
    transitive_firewall_callees: Interned<TransitiveFirewallCallees>,
}

impl NodeInfo {
    pub const fn new(
        fingerprint: Compact128,
        transitive_firewall_callees_fingerprint: Compact128,
        transitive_firewall_callees: Interned<TransitiveFirewallCallees>,
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
    ) -> &Interned<TransitiveFirewallCallees> {
        &self.transitive_firewall_callees
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

pub enum Either<L, R> {
    Left(L),
    Right(R),
}

impl<L: Iterator, R: Iterator<Item = L::Item>> Iterator for Either<L, R> {
    type Item = L::Item;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Left(l) => l.next(),
            Self::Right(r) => r.next(),
        }
    }
}

pub enum TieredStorage<S: BuildHasher + Clone> {
    /// 0-32 queries, stored as a small vec.
    Small(RwLock<Vec<QueryID>>),

    /// More than 32 queries, stored as an Arc to a slice.
    Large(DashSet<QueryID, S>),
}

impl<S: BuildHasher + Clone> TieredStorage<S> {
    #[must_use]
    pub const fn new() -> Self { Self::Small(RwLock::new(Vec::new())) }
}

impl<S: BuildHasher + Clone> Default for TieredStorage<S> {
    fn default() -> Self { Self::new() }
}

pub struct GuardedVecIterator<'a, T> {
    guard: parking_lot::RwLockReadGuard<'a, Vec<T>>,
    index: usize,
}

impl<T> Iterator for GuardedVecIterator<'_, T>
where
    T: Clone,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.guard.len() {
            None
        } else {
            let item = self.guard[self.index].clone();
            self.index += 1;
            Some(item)
        }
    }
}

#[allow(clippy::type_complexity)]
#[derive(Default, Clone)]
pub struct CompressedBackwardEdgeSet<S: BuildHasher + Clone>(
    pub Arc<RwLock<TieredStorage<S>>>,
);

impl<S: BuildHasher + Default + Clone + Send + Sync> ConcurrentSet
    for CompressedBackwardEdgeSet<S>
{
    type Element = QueryID;

    fn insert_element(&self, element: Self::Element) -> bool {
        let read = self.0.read();
        match &*read {
            TieredStorage::Small(vec_lock) => {
                let mut vec = vec_lock.write();

                // Upgrade to large storage if exceed threshold
                if vec.len() == 32 {
                    let large_set = DashSet::with_hasher(S::default());

                    for item in vec.drain(..) {
                        large_set.insert(item);
                    }

                    let result = large_set.insert(element);

                    drop(vec);
                    drop(read);

                    *self.0.write() = TieredStorage::Large(large_set);

                    result
                } else {
                    if vec.contains(&element) {
                        return false;
                    }

                    vec.push(element);

                    true
                }
            }

            TieredStorage::Large(set) => set.insert(element),
        }
    }

    fn remove_element(&self, element: &Self::Element) -> bool {
        match &*self.0.read() {
            TieredStorage::Small(vec_lock) => {
                let mut vec = vec_lock.write();

                vec.iter().position(|x| x == element).is_some_and(|pos| {
                    vec.swap_remove(pos);
                    true
                })
            }

            TieredStorage::Large(set) => set.remove(element).is_some(),
        }
    }
}

impl<S: BuildHasher + Clone> TieredStorage<S> {
    pub fn iter(&self) -> impl Iterator<Item = QueryID> + '_ {
        match self {
            Self::Small(vec_lock) => Either::Left(GuardedVecIterator {
                guard: vec_lock.read(),
                index: 0,
            }),

            Self::Large(set) => {
                Either::Right(set.iter().map(|entry| *entry.key()))
            }
        }
    }
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

type SingleMap<C, K, V> =
    <<C as Config>::StorageEngine as StorageEngine>::SingleMap<K, V>;

type DynamicMap<C, K> =
    <<C as Config>::StorageEngine as StorageEngine>::DynamicMap<K>;

type KeyOfSetMap<C, K, Con> =
    <<C as Config>::StorageEngine as StorageEngine>::KeyOfSetMap<K, Con>;

type WriteTransaction<C> =
    <<C as Config>::StorageEngine as StorageEngine>::WriteTransaction;

#[allow(clippy::type_complexity)]
pub struct Persist<C: Config> {
    last_verified: SingleMap<C, QueryNodeColumn, LastVerified>,
    forward_edge_order: SingleMap<C, QueryNodeColumn, ForwardEdgeOrder>,
    forward_edge_observation:
        SingleMap<C, QueryNodeColumn, ForwardEdgeObservation<C>>,
    query_kind: SingleMap<C, QueryNodeColumn, QueryKind>,
    node_info: SingleMap<C, QueryNodeColumn, NodeInfo>,
    pending_backward_projection:
        SingleMap<C, QueryNodeColumn, PendingBackwardProjection>,

    dirty_edge_set: SingleMap<C, DirtySetColumn, Unit>,

    query_store: DynamicMap<C, QueryStoreColumn>,

    backward_edges: KeyOfSetMap<
        C,
        BackwardEdgeColumn<C>,
        CompressedBackwardEdgeSet<C::BuildHasher>,
    >,

    external_input_queries: KeyOfSetMap<
        C,
        ExternalInputColumn<C>,
        Arc<DashSet<Compact128, C::BuildHasher>>,
    >,

    sync: sync::Sync<C>,
}

impl<C: Config> Persist<C> {
    pub async fn new(db: &C::StorageEngine) -> Self {
        Self {
        last_verified: db.new_single_map::<QueryNodeColumn, LastVerified>(),
        forward_edge_order:
            db.new_single_map::<QueryNodeColumn, ForwardEdgeOrder>(),
        forward_edge_observation: db
            .new_single_map::<QueryNodeColumn, ForwardEdgeObservation<C>>(),
        query_kind: db.new_single_map::<QueryNodeColumn, QueryKind>(),
        node_info: db.new_single_map::<QueryNodeColumn, NodeInfo>(),
        pending_backward_projection: db
            .new_single_map::<QueryNodeColumn, PendingBackwardProjection>(),

        dirty_edge_set: db.new_single_map::<DirtySetColumn, Unit>(),

        query_store: db.new_dynamic_map::<QueryStoreColumn>(),

        backward_edges: db.new_key_of_set_map::<
            BackwardEdgeColumn<C>,
            CompressedBackwardEdgeSet<C::BuildHasher>,
        >(),

        external_input_queries: db.new_key_of_set_map::<
            ExternalInputColumn<C>,
            Arc<DashSet<Compact128, C::BuildHasher>>,
        >(),

        sync: sync::Sync::new(db).await,
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
    #[inline(never)]
    pub(super) fn set_computed<Q: Query>(
        &self,
        query: Q,
        query_id: QueryID,
        query_value: Q::Value,
        query_value_fingerprint: Option<Compact128>,
        query_kind: QueryKind,
        forward_edge_order: Arc<[QueryID]>,
        forward_edge_observations: HashMap<
            QueryID,
            Observation,
            C::BuildHasher,
        >,
        tfc_achetype: Interned<TransitiveFirewallCallees>,
        has_pending_backward_projection: bool,
        current_timestamp: Timestamp,
        existing_forward_edges: Option<&[QueryID]>,
        mut tx: WriteTransaction<C>,
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

        let query_input = QueryInput::<Q>(query);
        let query_result = QueryResult::<Q>(query_value);

        {
            // remove prior backward edges
            if let Some(forward_edges) = existing_forward_edges {
                for edge in forward_edges {
                    self.computation_graph
                        .persist
                        .backward_edges
                        .remove(edge, &query_id, &mut tx);
                }
            }

            // set pending backward projection if needed
            if has_pending_backward_projection {
                self.computation_graph
                    .persist
                    .pending_backward_projection
                    .insert(
                        query_id,
                        PendingBackwardProjection(current_timestamp),
                        &mut tx,
                    );
            }

            self.computation_graph
                .persist
                .node_info
                .insert(query_id, node_info, &mut tx);

            self.computation_graph
                .persist
                .query_kind
                .insert(query_id, query_kind, &mut tx);

            self.computation_graph.persist.last_verified.insert(
                query_id,
                LastVerified(current_timestamp),
                &mut tx,
            );

            for edge in forward_edge_order.0.iter() {
                self.computation_graph
                    .persist
                    .backward_edges
                    .insert(*edge, query_id, &mut tx);
            }

            self.computation_graph.persist.forward_edge_order.insert(
                query_id,
                forward_edge_order,
                &mut tx,
            );

            self.computation_graph.persist.forward_edge_observation.insert(
                query_id,
                forward_edge_observations,
                &mut tx,
            );

            self.computation_graph.persist.query_store.insert(
                query_id.compact_hash_128(),
                query_input,
                &mut tx,
            );

            self.computation_graph.persist.query_store.insert(
                query_id.compact_hash_128(),
                query_result,
                &mut tx,
            );

            // Track external input queries by type for refresh support
            if query_kind.is_external_input() {
                self.computation_graph.persist.external_input_queries.insert(
                    query_id.stable_type_id(),
                    query_id.compact_hash_128(),
                    &mut tx,
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
        tx: &mut WriteTransaction<C>,
        set_input: bool,
        timestamp: Timestamp,
    ) {
        let query_id = QueryID::new::<Q>(query_hash_128);

        // if have an existing forward edges, unwire the backward edges
        let existing_forward_edges =
            unsafe { self.get_forward_edges_order_unchecked(&query_id).await };

        let empty_forward_edges = ForwardEdgeOrder(Arc::from([]));
        let empty_forward_edge_observations = ForwardEdgeObservation::<C>(
            Arc::new(HashMap::with_hasher(C::BuildHasher::default())),
        );

        let transitive_firewall_callees =
            self.create_tfc_from_iter(std::iter::empty());
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
                        .insert(*edge, query_id, tx);
                }
            }

            if set_input {
                self.computation_graph.persist.query_kind.insert(
                    query_id,
                    QueryKind::Input,
                    tx,
                );
            }

            self.computation_graph.persist.last_verified.insert(
                query_id,
                LastVerified(timestamp),
                tx,
            );

            self.computation_graph.persist.forward_edge_order.insert(
                query_id,
                empty_forward_edges,
                tx,
            );

            self.computation_graph.persist.forward_edge_observation.insert(
                query_id,
                empty_forward_edge_observations,
                tx,
            );

            self.computation_graph
                .persist
                .node_info
                .insert(query_id, node_info, tx);

            self.computation_graph.persist.query_store.insert(
                query_id.compact_hash_128(),
                query_input,
                tx,
            );

            self.computation_graph.persist.query_store.insert(
                query_id.compact_hash_128(),
                query_result,
                tx,
            );
        }
    }

    pub(super) fn mark_dirty_forward_edge(
        &self,
        from: QueryID,
        to: QueryID,
        tx: &mut WriteTransaction<C>,
    ) {
        let edge = Edge { from, to };

        self.computation_graph.persist.dirty_edge_set.insert(edge, Unit, tx);

        self.computation_graph.add_dirtied_edge_count();
    }

    #[allow(clippy::option_option)]
    pub(super) async fn clean_query(
        &self,
        query_id: &QueryID,
        clean_edges: &[QueryID],
        new_tfc: Option<Interned<TransitiveFirewallCallees>>,
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

        let mut tx = self.new_write_transaction(caller_information).await;

        {
            for callee in clean_edges.iter().copied() {
                let edge = Edge { from: *query_id, to: callee };

                self.computation_graph
                    .persist
                    .dirty_edge_set
                    .remove(&edge, &mut tx);
            }

            if let Some(node_info) = new_node_info {
                self.computation_graph
                    .persist
                    .node_info
                    .insert(*query_id, node_info, &mut tx);
            }

            self.computation_graph.persist.last_verified.insert(
                *query_id,
                LastVerified(caller_information.timestamp()),
                &mut tx,
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
            .get(&Edge { from, to })
            .await
            .is_some()
    }

    pub(super) async fn done_backward_projection(
        &self,
        query_id: &QueryID,
        caller_information: &CallerInformation,
        backward_projection_lock_guard: BackwardProjectionLockGuard<C>,
    ) {
        let mut tx = self.new_write_transaction(caller_information).await;

        self.computation_graph
            .persist
            .pending_backward_projection
            .remove(query_id, &mut tx);

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
                .get::<QueryInput<Q>>(&query_id)
                .await,
            self.computation_graph
                .persist
                .query_store
                .get::<QueryResult<Q>>(&query_id)
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
