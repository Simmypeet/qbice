use std::{
    collections::HashMap,
    hash::{BuildHasher, Hash},
    marker::PhantomData,
    mem::ManuallyDrop,
    pin::Pin,
    sync::Arc,
};

use dashmap::DashSet;
use enum_as_inner::EnumAsInner;
use ouroboros::self_referencing;
use parking_lot::RwLock;
use qbice_serialize::{Decode, Encode};
use qbice_stable_hash::{Compact128, StableHash};
use qbice_stable_type_id::{Identifiable, StableTypeID};
use qbice_storage::{
    dynamic_map::DynamicMap as _,
    intern::Interned,
    key_of_set_map::{ClonedDashSetIterator, ConcurrentSet, KeyOfSetMap as _},
    kv_database::{
        DiscriminantEncoding, KeyOfSetColumn, WideColumn, WideColumnValue,
    },
    single_map::SingleMap as _,
    storage_engine::StorageEngine,
};
pub use snapshot::Snapshot;
pub(super) use sync::ActiveComputationGuard;
pub(crate) use sync::ActiveInputSessionGuard;

use crate::{
    Engine, ExecutionStyle, Query,
    config::Config,
    engine::{
        computation_graph::{
            QueryKind, computing::BackwardProjectionLockGuard,
            tfc_achetype::TransitiveFirewallCallees,
        },
        guard::GuardExt,
    },
    query::QueryID,
};

mod snapshot;
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

impl Edge {
    pub const fn new(from: QueryID, to: QueryID) -> Self { Self { from, to } }
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

impl<S: BuildHasher + Default + Clone + Send + Sync + 'static> ConcurrentSet
    for CompressedBackwardEdgeSet<S>
{
    type Element = QueryID;

    type Iterator<'x>
        = GuardedIterator<'x, S>
    where
        Self: 'x;

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

    fn len(&self) -> usize {
        match &*self.0.read() {
            TieredStorage::Small(vec_lock) => vec_lock.read().len(),
            TieredStorage::Large(set) => set.len(),
        }
    }

    fn iter(&self) -> Self::Iterator<'_> {
        GuardedIterator::new(self.0.read(), |x| match &**x {
            TieredStorage::Small(vec_lock) => {
                Either::Left(GuardedVecIterator {
                    guard: vec_lock.read(),
                    index: 0,
                })
            }

            TieredStorage::Large(set) => {
                Either::Right(ClonedDashSetIterator::new(set.iter()))
            }
        })
    }
}

#[self_referencing]
pub struct GuardedIterator<'s, S: BuildHasher + Clone + Send + Sync> {
    gaurd: parking_lot::RwLockReadGuard<'s, TieredStorage<S>>,

    #[borrows(mut gaurd)]
    #[not_covariant]
    iter: Either<
        GuardedVecIterator<'this, QueryID>,
        ClonedDashSetIterator<'this, QueryID, S>,
    >,
}

impl<S: BuildHasher + Clone + Send + Sync> Iterator for GuardedIterator<'_, S> {
    type Item = QueryID;

    fn next(&mut self) -> Option<Self::Item> {
        self.with_iter_mut(|x| x.next())
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

#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Encode, Decode, EnumAsInner,
)]
pub enum NodeDependency {
    Single(QueryID),
    Unordered(Vec<QueryID>),
}

implements_wide_column_value_new_type!(
    ForwardEdgeOrder,
    Arc<[NodeDependency]>,
    QueryNodeDiscriminant::ForwardEdgeOrder
);

impl ForwardEdgeOrder {
    pub fn iter_all_callees(&self) -> impl Iterator<Item = QueryID> + '_ {
        self.0.iter().flat_map(|dep| match dep {
            NodeDependency::Single(qid) => Either::Left(std::iter::once(*qid)),
            NodeDependency::Unordered(vec) => {
                Either::Right(vec.clone().into_iter())
            }
        })
    }
}

#[derive(Clone)]
pub struct ForwardEdgeObservation<C: Config>(
    pub Arc<HashMap<QueryID, Observation, C::BuildHasher>>,
);

impl<C: Config> std::fmt::Debug for ForwardEdgeObservation<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("ForwardEdgeObservation").field(&self.0).finish()
    }
}

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
pub struct Database<C: Config> {
    sync: ManuallyDrop<sync::Sync<C>>,

    last_verified: ManuallyDrop<SingleMap<C, QueryNodeColumn, LastVerified>>,
    forward_edge_order:
        ManuallyDrop<SingleMap<C, QueryNodeColumn, ForwardEdgeOrder>>,
    forward_edge_observation:
        ManuallyDrop<SingleMap<C, QueryNodeColumn, ForwardEdgeObservation<C>>>,

    query_kind: ManuallyDrop<SingleMap<C, QueryNodeColumn, QueryKind>>,
    node_info: ManuallyDrop<SingleMap<C, QueryNodeColumn, NodeInfo>>,
    pending_backward_projection:
        ManuallyDrop<SingleMap<C, QueryNodeColumn, PendingBackwardProjection>>,

    dirty_edge_set: ManuallyDrop<SingleMap<C, DirtySetColumn, Unit>>,

    query_store: ManuallyDrop<DynamicMap<C, QueryStoreColumn>>,

    backward_edges: ManuallyDrop<
        KeyOfSetMap<
            C,
            BackwardEdgeColumn<C>,
            CompressedBackwardEdgeSet<C::BuildHasher>,
        >,
    >,

    external_input_queries: ManuallyDrop<
        KeyOfSetMap<
            C,
            ExternalInputColumn<C>,
            Arc<DashSet<Compact128, C::BuildHasher>>,
        >,
    >,
}

impl<C: Config> Database<C> {
    pub async fn new(db: &C::StorageEngine) -> Self {
        Self {
            last_verified: ManuallyDrop::new(db.new_single_map::<QueryNodeColumn, LastVerified>()),
            forward_edge_order:
                ManuallyDrop::new(db.new_single_map::<QueryNodeColumn, ForwardEdgeOrder>()),
            forward_edge_observation: ManuallyDrop::new(db
                .new_single_map::<QueryNodeColumn, ForwardEdgeObservation<C>>()),
            query_kind: ManuallyDrop::new(db.new_single_map::<QueryNodeColumn, QueryKind>()),
            node_info: ManuallyDrop::new(db.new_single_map::<QueryNodeColumn, NodeInfo>()),
            pending_backward_projection: ManuallyDrop::new(db
                .new_single_map::<QueryNodeColumn, PendingBackwardProjection>()),

            dirty_edge_set: ManuallyDrop::new(db.new_single_map::<DirtySetColumn, Unit>()),

            query_store: ManuallyDrop::new(db.new_dynamic_map::<QueryStoreColumn>()),

            backward_edges: ManuallyDrop::new(db.new_key_of_set_map::<
                BackwardEdgeColumn<C>,
                CompressedBackwardEdgeSet<C::BuildHasher>,
            >()),

            external_input_queries: ManuallyDrop::new(db.new_key_of_set_map::<
                ExternalInputColumn<C>,
                Arc<DashSet<Compact128, C::BuildHasher>>,
            >()),

            sync: ManuallyDrop::new(sync::Sync::new(db).await),
        }
    }
}

impl<C: Config> Drop for Database<C> {
    fn drop(&mut self) {
        unsafe {
            // These are the heavy data structures that take time to drop.
            let sync = ManuallyDrop::take(&mut self.sync);
            let last_verified = ManuallyDrop::take(&mut self.last_verified);
            let forward_edge_order =
                ManuallyDrop::take(&mut self.forward_edge_order);
            let forward_edge_observation =
                ManuallyDrop::take(&mut self.forward_edge_observation);
            let query_kind = ManuallyDrop::take(&mut self.query_kind);
            let node_info = ManuallyDrop::take(&mut self.node_info);
            let pending_backward_projection =
                ManuallyDrop::take(&mut self.pending_backward_projection);
            let dirty_edge_set = ManuallyDrop::take(&mut self.dirty_edge_set);
            let query_store = ManuallyDrop::take(&mut self.query_store);
            let backward_edges = ManuallyDrop::take(&mut self.backward_edges);
            let external_input_queries =
                ManuallyDrop::take(&mut self.external_input_queries);

            rayon::scope(|scope| {
                scope.spawn(|_| drop(sync));
                scope.spawn(|_| drop(last_verified));
                scope.spawn(|_| drop(forward_edge_order));
                scope.spawn(|_| drop(forward_edge_observation));
                scope.spawn(|_| drop(query_kind));
                scope.spawn(|_| drop(node_info));
                scope.spawn(|_| drop(pending_backward_projection));
                scope.spawn(|_| drop(dirty_edge_set));
                scope.spawn(|_| drop(query_store));
                scope.spawn(|_| drop(backward_edges));
                scope.spawn(|_| drop(external_input_queries));
            });
        }
    }
}

#[derive(Debug, Clone, Copy, Encode, Decode)]
pub struct Observation {
    pub seen_value_fingerprint: Compact128,
    pub seen_transitive_firewall_callees_fingerprint: Compact128,
}

impl<C: Config> Engine<C> {
    pub(super) async fn is_edge_dirty(
        &self,
        from: QueryID,
        to: QueryID,
    ) -> bool {
        self.computation_graph
            .database
            .dirty_edge_set
            .get(&Edge { from, to })
            .await
            .is_some()
    }

    /// Retrieving the query kind from a global database is safe since the
    /// query kind is immutable once created.
    pub(super) async fn get_query_kind(&self, query_id: &QueryID) -> QueryKind {
        self.computation_graph.database.query_kind.get(query_id).await.unwrap()
    }

    /// Retrieving the query input from a global database is safe since the
    /// query input is immutable once created.
    pub(super) async fn get_query_input<Q: Query>(
        &self,
        hash128: &Compact128,
    ) -> Q {
        self.computation_graph
            .database
            .query_store
            .get::<QueryInput<Q>>(hash128)
            .await
            .map(|x| x.0)
            .unwrap()
    }

    /// Directly access the node info without any lock.
    ///
    /// This is only acceptable if you've made sure that the node has no
    /// other concurrent mutations.
    pub(super) async unsafe fn get_node_info_unchecked(
        &self,
        query_id: &QueryID,
    ) -> NodeInfo {
        self.computation_graph.database.node_info.get(query_id).await.unwrap()
    }

    pub(super) async fn get_external_input_queries(
        &self,
        stable_type_id: &StableTypeID,
    ) -> impl Iterator<Item = Compact128> + Send {
        self.computation_graph
            .database
            .external_input_queries
            .get(stable_type_id)
            .await
    }

    pub(crate) async fn get_node_snapshot_for_graph(
        &self,
        query_id: &QueryID,
        include_tfc: bool,
    ) -> (
        Option<QueryKind>,
        Option<ForwardEdgeOrder>,
        Option<Interned<TransitiveFirewallCallees>>,
    ) {
        let node_info =
            self.computation_graph.database.query_kind.get(query_id).await;
        let forward_edge_order = self
            .computation_graph
            .database
            .forward_edge_order
            .get(query_id)
            .await;

        let transitive_firewall_callees = if include_tfc
            || node_info.is_some_and(|x| {
                x == QueryKind::Executable(ExecutionStyle::Firewall)
            }) {
            self.computation_graph
                .database
                .node_info
                .get(query_id)
                .await
                .map(|x| x.transitive_firewall_callees)
        } else {
            None
        };

        (node_info, forward_edge_order, transitive_firewall_callees)
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
                .database
                .query_store
                .get::<QueryInput<Q>>(&query_id)
                .await
                .map(|x| x.0),
            self.computation_graph
                .database
                .query_store
                .get::<QueryResult<Q>>(&query_id)
                .await
                .map(|x| x.0),
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
    ) -> Pin<
        Box<dyn std::future::Future<Output = Option<QueryDebug>> + Send + 's>,
    > {
        Box::pin(async move { self.get_query_debug::<Q>(query_id).await })
    }

    /// Directly access the backward edges without any lock.
    pub(super) async unsafe fn get_backward_edges_unchecked(
        &self,
        query_id: &QueryID,
    ) -> impl Iterator<Item = QueryID> + Send {
        self.computation_graph.database.backward_edges.get(query_id).await
    }
}

impl<C: Config, Q: Query> Snapshot<C, Q> {
    #[allow(clippy::option_option)]
    pub async fn clean_query(
        &mut self,
        clean_edges: Vec<QueryID>,
        new_tfc: Option<Interned<TransitiveFirewallCallees>>,
        timestamp: Timestamp,
    ) {
        let mut tx = self.engine().new_write_transaction();

        let new_node_info = if let Some(x) = new_tfc {
            let mut current_node_info = self.node_info().await.unwrap();

            current_node_info.transitive_firewall_callees = x;
            current_node_info.transitive_firewall_callees_fingerprint = self
                .engine()
                .hash(&current_node_info.transitive_firewall_callees);

            Some(current_node_info)
        } else {
            None
        };

        for callee in clean_edges.iter().copied() {
            let edge = Edge { from: *self.query_id(), to: callee };

            self.engine()
                .computation_graph
                .database
                .dirty_edge_set
                .remove(&edge, &mut tx)
                .await;
        }

        if let Some(node_info) = new_node_info {
            self.engine()
                .computation_graph
                .database
                .node_info
                .insert(*self.query_id(), node_info, &mut tx)
                .await;
        }

        self.engine()
            .computation_graph
            .database
            .last_verified
            .insert(*self.query_id(), LastVerified(timestamp), &mut tx)
            .await;

        self.engine().submit_write_buffer(tx);
    }

    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    #[inline(never)]
    pub(super) async fn set_computed(
        mut self,
        query: Q,
        query_value: Q::Value,
        query_value_fingerprint: Option<Compact128>,
        query_kind: QueryKind,
        forward_edge_order: Arc<[NodeDependency]>,
        forward_edge_observations: HashMap<
            QueryID,
            Observation,
            C::BuildHasher,
        >,
        tfc_achetype: Interned<TransitiveFirewallCallees>,
        has_pending_backward_projection: bool,
        current_timestamp: Timestamp,
        existing_forward_edges: Option<&[NodeDependency]>,
        mut tx: WriteTransaction<C>,
    ) {
        self.upgrade_to_exclusive().await;

        let query_value_fingerprint = query_value_fingerprint
            .unwrap_or_else(|| self.engine().hash(&query_value));
        let transitive_firewall_callees_fingerprint =
            self.engine().hash(&tfc_achetype);

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
                for dep in forward_edges {
                    match dep {
                        NodeDependency::Single(edge) => {
                            self.engine()
                                .computation_graph
                                .database
                                .backward_edges
                                .remove(edge, self.query_id(), &mut tx)
                                .await;
                        }
                        NodeDependency::Unordered(unordered) => {
                            for edge in unordered {
                                self.engine()
                                    .computation_graph
                                    .database
                                    .backward_edges
                                    .remove(edge, self.query_id(), &mut tx)
                                    .await;
                            }
                        }
                    }
                }
            }

            // set pending backward projection if needed
            if has_pending_backward_projection {
                self.engine()
                    .computation_graph
                    .database
                    .pending_backward_projection
                    .insert(
                        *self.query_id(),
                        PendingBackwardProjection(current_timestamp),
                        &mut tx,
                    )
                    .await;
            }

            self.engine()
                .computation_graph
                .database
                .node_info
                .insert(*self.query_id(), node_info, &mut tx)
                .await;

            self.engine()
                .computation_graph
                .database
                .query_kind
                .insert(*self.query_id(), query_kind, &mut tx)
                .await;

            self.engine()
                .computation_graph
                .database
                .last_verified
                .insert(
                    *self.query_id(),
                    LastVerified(current_timestamp),
                    &mut tx,
                )
                .await;

            for edge in forward_edge_order.0.iter() {
                match edge {
                    NodeDependency::Single(query_id) => {
                        self.engine()
                            .computation_graph
                            .database
                            .backward_edges
                            .insert(*query_id, *self.query_id(), &mut tx)
                            .await;
                    }
                    NodeDependency::Unordered(query_ids) => {
                        for edge in query_ids {
                            self.engine()
                                .computation_graph
                                .database
                                .backward_edges
                                .insert(*edge, *self.query_id(), &mut tx)
                                .await;
                        }
                    }
                }
            }

            self.engine()
                .computation_graph
                .database
                .forward_edge_order
                .insert(*self.query_id(), forward_edge_order, &mut tx)
                .await;

            self.engine()
                .computation_graph
                .database
                .forward_edge_observation
                .insert(*self.query_id(), forward_edge_observations, &mut tx)
                .await;

            self.engine()
                .computation_graph
                .database
                .query_store
                .insert(
                    self.query_id().compact_hash_128(),
                    query_input,
                    &mut tx,
                )
                .await;

            self.engine()
                .computation_graph
                .database
                .query_store
                .insert(
                    self.query_id().compact_hash_128(),
                    query_result,
                    &mut tx,
                )
                .await;

            // Track external input queries by type for refresh support
            if query_kind.is_external_input() {
                self.engine()
                    .computation_graph
                    .database
                    .external_input_queries
                    .insert(
                        self.query_id().stable_type_id(),
                        self.query_id().compact_hash_128(),
                        &mut tx,
                    )
                    .await;
            }

            self.engine().submit_write_buffer(tx);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) async fn set_computed_input(
        mut self,
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
        let existing_forward_edges = self.forward_edge_order().await;

        let empty_forward_edges = ForwardEdgeOrder(Arc::from([]));
        let empty_forward_edge_observations = ForwardEdgeObservation::<C>(
            Arc::new(HashMap::with_hasher(C::BuildHasher::default())),
        );

        let transitive_firewall_callees =
            self.engine().create_tfc_from_iter(std::iter::empty());
        let transitive_firewall_callees_fingerprint =
            self.engine().hash(&transitive_firewall_callees);

        let node_info = NodeInfo::new(
            query_value_fingerprint,
            transitive_firewall_callees_fingerprint,
            transitive_firewall_callees,
        );

        let query_input = QueryInput::<Q>(query);
        let query_result = QueryResult::<Q>(query_value);

        {
            if let Some(forward_edges) = existing_forward_edges {
                for edge in forward_edges.0.iter() {
                    match edge {
                        NodeDependency::Single(edge) => {
                            self.engine()
                                .computation_graph
                                .database
                                .backward_edges
                                .remove(edge, &query_id, tx)
                                .await;
                        }
                        NodeDependency::Unordered(query_ids) => {
                            for edge in query_ids {
                                self.engine()
                                    .computation_graph
                                    .database
                                    .backward_edges
                                    .remove(edge, &query_id, tx)
                                    .await;
                            }
                        }
                    }
                }
            }

            if set_input {
                self.engine()
                    .computation_graph
                    .database
                    .query_kind
                    .insert(query_id, QueryKind::Input, tx)
                    .await;
            }

            self.engine()
                .computation_graph
                .database
                .last_verified
                .insert(query_id, LastVerified(timestamp), tx)
                .await;

            self.engine()
                .computation_graph
                .database
                .forward_edge_order
                .insert(query_id, empty_forward_edges, tx)
                .await;

            self.engine()
                .computation_graph
                .database
                .forward_edge_observation
                .insert(query_id, empty_forward_edge_observations, tx)
                .await;

            self.engine()
                .computation_graph
                .database
                .node_info
                .insert(query_id, node_info, tx)
                .await;

            self.engine()
                .computation_graph
                .database
                .query_store
                .insert(query_id.compact_hash_128(), query_input, tx)
                .await;

            self.engine()
                .computation_graph
                .database
                .query_store
                .insert(query_id.compact_hash_128(), query_result, tx)
                .await;
        }
    }

    pub(super) async fn done_backward_projection(
        mut self,
        mut backward_projection_lock_guard: BackwardProjectionLockGuard<C>,
    ) {
        let mut tx = self.engine().new_write_transaction();
        let engine = self.engine().clone();
        let query_id = *self.query_id();

        self.upgrade_to_exclusive().await;

        async move {
            engine
                .computation_graph
                .database
                .pending_backward_projection
                .remove(&query_id, &mut tx)
                .await;

            engine.submit_write_buffer(tx);

            backward_projection_lock_guard.done();
        }
        .guarded()
        .await;
    }
}

impl<C: Config> Database<C> {
    pub(super) async fn mark_dirty_forward_edge_from(
        &self,
        edge: Edge,
        tx: &mut WriteTransaction<C>,
    ) {
        self.dirty_edge_set.insert(edge, Unit, tx).await;
    }

    pub(super) async fn mark_dirty_forward_edge(
        &self,
        from: QueryID,
        to: QueryID,
        tx: &mut WriteTransaction<C>,
    ) {
        let edge = Edge { from, to };

        self.dirty_edge_set.insert(edge, Unit, tx).await;
    }

    /// Directly access the backward edges without any lock.
    pub(super) async unsafe fn get_backward_edges_unchecked(
        &self,
        query_id: &QueryID,
    ) -> impl Iterator<Item = QueryID> + Send {
        self.backward_edges.get(query_id).await
    }

    /// Retrieving the query kind from a global database is safe since the
    /// query kind is immutable once created.
    pub(super) async fn get_query_kind(&self, query_id: &QueryID) -> QueryKind {
        self.query_kind.get(query_id).await.unwrap()
    }
}
