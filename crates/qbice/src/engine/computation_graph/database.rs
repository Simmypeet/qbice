use std::{
    any::Any,
    collections::{HashMap, HashSet},
    hash::{BuildHasher, Hash},
    marker::PhantomData,
    mem::ManuallyDrop,
    sync::Arc,
};

use crossbeam::sync::WaitGroup;
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
    write_manager::WriteManager,
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
    executor::{BoxFuture, DynQueryInput, DynValue},
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
    ForwardEdges,
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
    /// The kind of the query. It is part of the node info because whoever
    /// reads the one mostly reads the other, and both are written whenever
    /// the query is computed.
    kind: QueryKind,
    fingerprint: Compact128,
    transitive_firewall_callees_fingerprint: Compact128,
    transitive_firewall_callees: Interned<TransitiveFirewallCallees>,
}

impl NodeInfo {
    pub const fn new(
        kind: QueryKind,
        fingerprint: Compact128,
        transitive_firewall_callees_fingerprint: Compact128,
        transitive_firewall_callees: Interned<TransitiveFirewallCallees>,
    ) -> Self {
        Self {
            kind,
            fingerprint,
            transitive_firewall_callees_fingerprint,
            transitive_firewall_callees,
        }
    }

    pub const fn kind(&self) -> QueryKind { self.kind }

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

/// The most queries that [`TieredStorage::Small`] stores, beyond which they
/// are stored in [`TieredStorage::Large`].
const SMALL_STORAGE_CAPACITY: usize = 32;

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
        // insert under the shared lock, unless the small storage is full
        {
            let read = self.0.read();
            match &*read {
                TieredStorage::Small(vec_lock) => {
                    let mut vec = vec_lock.write();

                    if vec.len() < SMALL_STORAGE_CAPACITY {
                        if vec.contains(&element) {
                            return false;
                        }

                        vec.push(element);

                        return true;
                    }
                }

                TieredStorage::Large(set) => return set.insert(element),
            }
        }

        // Upgrade to the large storage under the exclusive lock. Draining the
        // small storage under the shared lock and replacing it afterwards
        // would lose what another thread inserts into the drained storage in
        // between. Another thread may have upgraded it first.
        let mut write = self.0.write();
        if let TieredStorage::Small(vec_lock) = &mut *write {
            let large_set = DashSet::with_hasher(S::default());
            for item in vec_lock.get_mut().drain(..) {
                large_set.insert(item);
            }

            *write = TieredStorage::Large(large_set);
        }

        match &*write {
            TieredStorage::Large(set) => set.insert(element),
            TieredStorage::Small(_) => {
                unreachable!("the storage was upgraded to the large one")
            }
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

/// The queries that a query read when it was last computed, in the order it
/// read them.
#[derive(Debug, Clone)]
pub struct ForwardEdgeOrder(pub Arc<[NodeDependency]>);

/// The queries that the dependencies name, in the order they are listed.
fn callees(dependencies: &[NodeDependency]) -> impl Iterator<Item = &QueryID> {
    dependencies.iter().flat_map(|dependency| match dependency {
        NodeDependency::Single(callee) => std::slice::from_ref(callee),
        NodeDependency::Unordered(callees) => callees.as_slice(),
    })
}

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

/// What a query read when it was last computed: its dependencies in the
/// order it read them, and what it saw of each of them.
///
/// The two are stored as one value. They are written together whenever the
/// query is computed and read together when it is repaired, and stored apart
/// every dependency would be written down twice, once for its place in the
/// order and once for what was seen of it.
#[derive(Clone)]
pub struct ForwardEdges<C: Config> {
    pub order: ForwardEdgeOrder,
    pub observation: ForwardEdgeObservation<C>,
}

impl<C: Config> std::fmt::Debug for ForwardEdges<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ForwardEdges")
            .field("order", &self.order)
            .field("observation", &self.observation)
            .finish()
    }
}

impl<C: Config> WideColumnValue<QueryNodeColumn> for ForwardEdges<C> {
    fn discriminant() -> QueryNodeDiscriminant {
        QueryNodeDiscriminant::ForwardEdges
    }
}

/// Marks a dependency that was read on its own.
const SINGLE_DEPENDENCY: u8 = 0;

/// Marks a group of dependencies that were read in no particular order.
const UNORDERED_DEPENDENCIES: u8 = 1;

impl<C: Config> ForwardEdges<C> {
    /// Encodes a dependency together with what was seen of it, which is
    /// nothing for a dependency that was read without its value.
    fn encode_callee<E: qbice_serialize::Encoder + ?Sized>(
        &self,
        callee: &QueryID,
        encoder: &mut E,
        plugin: &qbice_serialize::Plugin,
        session: &mut qbice_serialize::session::Session,
    ) -> std::io::Result<()> {
        callee.encode(encoder, plugin, session)?;

        self.observation.0.get(callee).copied().encode(encoder, plugin, session)
    }

    /// Decodes a dependency and records what was seen of it.
    fn decode_callee<D: qbice_serialize::Decoder + ?Sized>(
        observation: &mut HashMap<QueryID, Observation, C::BuildHasher>,
        decoder: &mut D,
        plugin: &qbice_serialize::Plugin,
        session: &mut qbice_serialize::session::Session,
    ) -> std::io::Result<QueryID> {
        let callee = QueryID::decode(decoder, plugin, session)?;

        if let Some(seen) =
            Option::<Observation>::decode(decoder, plugin, session)?
        {
            observation.insert(callee, seen);
        }

        Ok(callee)
    }
}

impl<C: Config> Encode for ForwardEdges<C> {
    fn encode<E: qbice_serialize::Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        plugin: &qbice_serialize::Plugin,
        session: &mut qbice_serialize::session::Session,
    ) -> std::io::Result<()> {
        encoder.emit_usize(self.order.0.len())?;

        for dependency in self.order.0.iter() {
            match dependency {
                NodeDependency::Single(callee) => {
                    encoder.emit_u8(SINGLE_DEPENDENCY)?;
                    self.encode_callee(callee, encoder, plugin, session)?;
                }

                NodeDependency::Unordered(callees) => {
                    encoder.emit_u8(UNORDERED_DEPENDENCIES)?;
                    encoder.emit_usize(callees.len())?;

                    for callee in callees {
                        self.encode_callee(callee, encoder, plugin, session)?;
                    }
                }
            }
        }

        Ok(())
    }
}

impl<C: Config> Decode for ForwardEdges<C> {
    fn decode<D: qbice_serialize::Decoder + ?Sized>(
        decoder: &mut D,
        plugin: &qbice_serialize::Plugin,
        session: &mut qbice_serialize::session::Session,
    ) -> std::io::Result<Self> {
        let len = usize::decode(decoder, plugin, session)?;

        let mut order = Vec::with_capacity(len);
        let mut observation = HashMap::with_hasher(C::BuildHasher::default());

        for _ in 0..len {
            match u8::decode(decoder, plugin, session)? {
                SINGLE_DEPENDENCY => {
                    order.push(NodeDependency::Single(Self::decode_callee(
                        &mut observation,
                        decoder,
                        plugin,
                        session,
                    )?));
                }

                UNORDERED_DEPENDENCIES => {
                    let count = usize::decode(decoder, plugin, session)?;
                    let mut callees = Vec::with_capacity(count);

                    for _ in 0..count {
                        callees.push(Self::decode_callee(
                            &mut observation,
                            decoder,
                            plugin,
                            session,
                        )?);
                    }

                    order.push(NodeDependency::Unordered(callees));
                }

                tag => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("unknown kind of dependency: {tag}"),
                    ));
                }
            }
        }

        Ok(Self {
            order: ForwardEdgeOrder(order.into()),
            observation: ForwardEdgeObservation(Arc::new(observation)),
        })
    }
}

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

type ReadQueryResultFn<C> = for<'a> fn(
    engine: &'a Engine<C>,
    query_input_hash_128: &'a Compact128,
    slot: &'a mut (dyn Any + Send),
) -> BoxFuture<'a, bool>;

/// A place for the value of a query, owned by the caller of the query.
///
/// The engine runs the query without knowing its type, so the caller, which
/// does know it, hands over both the place for the value, an
/// `Option<Q::Value>`, and the function that reads the value into it.
pub(crate) struct ValueSlot<'a, C: Config> {
    slot: &'a mut (dyn Any + Send),
    read: ReadQueryResultFn<C>,
}

impl<'a, C: Config> ValueSlot<'a, C> {
    pub fn new<Q: Query>(slot: &'a mut Option<Q::Value>) -> Self {
        Self { slot, read: Engine::<C>::read_dyn_query_result::<Q> }
    }

    /// Reads the stored value of the query with the given input hash into the
    /// slot. Returns whether the value was found.
    pub async fn read(
        &mut self,
        engine: &Engine<C>,
        query_input_hash_128: &Compact128,
    ) -> bool {
        (self.read)(engine, query_input_hash_128, &mut *self.slot).await
    }
}

/// The write manager of the storage engine, which creates the maps.
type WriteManagerOf<C> =
    <<C as Config>::StorageEngine as StorageEngine>::WriteManager;

type SingleMap<C, K, V> = <WriteManagerOf<C> as WriteManager>::SingleMap<K, V>;

type DynamicMap<C, K> = <WriteManagerOf<C> as WriteManager>::DynamicMap<K>;

type KeyOfSetMap<C, K, Con> =
    <WriteManagerOf<C> as WriteManager>::KeyOfSetMap<K, Con>;

type WriteTransaction<C> =
    <<C as Config>::StorageEngine as StorageEngine>::WriteTransaction;

#[allow(clippy::type_complexity)]
pub struct Database<C: Config> {
    sync: ManuallyDrop<sync::Sync<C>>,

    last_verified: ManuallyDrop<SingleMap<C, QueryNodeColumn, LastVerified>>,
    forward_edges: ManuallyDrop<SingleMap<C, QueryNodeColumn, ForwardEdges<C>>>,

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
        // every map is created by the write manager that its writes go
        // through
        let write_manager = db.new_write_manager();

        Self {
            last_verified: ManuallyDrop::new(write_manager.new_single_map::<QueryNodeColumn, LastVerified>()),
            forward_edges:
                ManuallyDrop::new(write_manager.new_single_map::<QueryNodeColumn, ForwardEdges<C>>()),
            node_info: ManuallyDrop::new(write_manager.new_single_map::<QueryNodeColumn, NodeInfo>()),
            pending_backward_projection: ManuallyDrop::new(write_manager
                .new_single_map::<QueryNodeColumn, PendingBackwardProjection>()),

            dirty_edge_set: ManuallyDrop::new(write_manager.new_single_map::<DirtySetColumn, Unit>()),

            query_store: ManuallyDrop::new(write_manager.new_dynamic_map::<QueryStoreColumn>()),

            backward_edges: ManuallyDrop::new(write_manager.new_key_of_set_map::<
                BackwardEdgeColumn<C>,
                CompressedBackwardEdgeSet<C::BuildHasher>,
            >()),

            external_input_queries: ManuallyDrop::new(write_manager.new_key_of_set_map::<
                ExternalInputColumn<C>,
                Arc<DashSet<Compact128, C::BuildHasher>>,
            >()),

            sync: ManuallyDrop::new(sync::Sync::new(write_manager).await),
        }
    }
}

impl<C: Config> Drop for Database<C> {
    fn drop(&mut self) {
        unsafe {
            macro_rules! drop_async {
                ($wait_group:ident, $drop:ident) => {{
                    let wait_group = $wait_group.clone();

                    tokio::task::spawn_blocking(move || {
                        drop($drop);
                        drop(wait_group);
                    });
                }};
            }
            let wait_group = WaitGroup::new();

            // These are the heavy data structures that take time to drop.
            let sync = ManuallyDrop::take(&mut self.sync);
            let last_verified = ManuallyDrop::take(&mut self.last_verified);
            let forward_edges = ManuallyDrop::take(&mut self.forward_edges);
            let node_info = ManuallyDrop::take(&mut self.node_info);
            let pending_backward_projection =
                ManuallyDrop::take(&mut self.pending_backward_projection);
            let dirty_edge_set = ManuallyDrop::take(&mut self.dirty_edge_set);
            let query_store = ManuallyDrop::take(&mut self.query_store);
            let backward_edges = ManuallyDrop::take(&mut self.backward_edges);
            let external_input_queries =
                ManuallyDrop::take(&mut self.external_input_queries);

            drop_async!(wait_group, sync);
            drop_async!(wait_group, last_verified);
            drop_async!(wait_group, forward_edges);
            drop_async!(wait_group, node_info);
            drop_async!(wait_group, pending_backward_projection);
            drop_async!(wait_group, dirty_edge_set);
            drop_async!(wait_group, query_store);
            drop_async!(wait_group, backward_edges);
            drop_async!(wait_group, external_input_queries);

            // blocks until all the above data structures are dropped in their
            // respective threads
            wait_group.wait();
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
        self.computation_graph.database.get_query_kind(query_id).await
    }

    /// The stored transitive firewall callees of `query_id`, if computed.
    pub(super) async fn get_transitive_firewall_callees(
        &self,
        query_id: &QueryID,
    ) -> Option<Interned<TransitiveFirewallCallees>> {
        self.computation_graph
            .database
            .node_info
            .get(query_id)
            .await
            .map(|x| x.transitive_firewall_callees)
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

    pub(super) async fn get_query_result<Q: Query>(
        &self,
        hash128: &Compact128,
    ) -> Option<Q::Value> {
        self.computation_graph
            .database
            .query_store
            .get::<QueryResult<Q>>(hash128)
            .await
            .map(|x| x.0)
    }

    /// Stores the input of a query of type `Q` and its value.
    pub(super) async fn set_query<Q: Query>(
        &self,
        hash128: Compact128,
        query: Q,
        value: Q::Value,
        tx: &mut WriteTransaction<C>,
    ) {
        let query_store = &self.computation_graph.database.query_store;

        query_store.insert(hash128, QueryInput::<Q>(query), tx).await;
        query_store.insert(hash128, QueryResult::<Q>(value), tx).await;
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
            self.computation_graph.database.node_info.get(query_id).await;
        let forward_edge_order = self
            .computation_graph
            .database
            .forward_edges
            .get(query_id)
            .await
            .map(|x| x.order);

        let query_kind = node_info.as_ref().map(NodeInfo::kind);

        let transitive_firewall_callees = if include_tfc
            || query_kind.is_some_and(|x| {
                x == QueryKind::Executable(ExecutionStyle::Firewall)
            }) {
            node_info.map(|x| x.transitive_firewall_callees)
        } else {
            None
        };

        (query_kind, forward_edge_order, transitive_firewall_callees)
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

    pub(crate) fn get_query_debug_future<Q: Query>(
        &self,
        query_id: Compact128,
    ) -> BoxFuture<'_, Option<QueryDebug>> {
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

/// The functions behind the operations of a query type that the engine calls
/// without knowing the type: those of [`crate::executor::Entry`] and
/// [`ValueSlot`].
impl<C: Config> Engine<C> {
    pub(crate) fn hash_dyn_value<Q: Query>(
        &self,
        value: &(dyn Any + Send + Sync),
    ) -> Compact128 {
        self.hash(
            value.downcast_ref::<Q::Value>().expect("value type mismatch"),
        )
    }

    pub(crate) fn clone_dyn_query<Q: Query>(
        query: &(dyn Any + Send + Sync),
    ) -> DynQueryInput {
        DynQueryInput::new(
            query.downcast_ref::<Q>().expect("query type mismatch").clone(),
        )
    }

    pub(crate) fn get_dyn_query_input<'a, Q: Query>(
        &'a self,
        hash128: &'a Compact128,
    ) -> BoxFuture<'a, DynQueryInput> {
        Box::pin(async move {
            DynQueryInput::new(self.get_query_input::<Q>(hash128).await)
        })
    }

    pub(crate) fn set_dyn_query<'a, Q: Query>(
        &'a self,
        hash128: Compact128,
        query: DynQueryInput,
        value: DynValue,
        tx: &'a mut WriteTransaction<C>,
    ) -> BoxFuture<'a, ()> {
        let query = query.downcast::<Q>();
        let value = *value.downcast::<Q::Value>().expect("value type mismatch");

        Box::pin(self.set_query::<Q>(hash128, query, value, tx))
    }

    fn read_dyn_query_result<'a, Q: Query>(
        &'a self,
        hash128: &'a Compact128,
        slot: &'a mut (dyn Any + Send),
    ) -> BoxFuture<'a, bool> {
        let slot = slot
            .downcast_mut::<Option<Q::Value>>()
            .expect("value slot type mismatch");

        Box::pin(async move {
            *slot = self.get_query_result::<Q>(hash128).await;
            slot.is_some()
        })
    }
}

impl<C: Config> Snapshot<C> {
    #[allow(clippy::option_option)]
    pub async fn clean_query(
        &mut self,
        clean_edges: Vec<QueryID>,
        new_tfc: Option<Interned<TransitiveFirewallCallees>>,
        timestamp: Timestamp,
    ) {
        let mut tx = self.engine().new_write_transaction();

        let new_node_info = if let Some(x) = new_tfc {
            let mut current_node_info = self.node_info().await.unwrap().clone();

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

    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        clippy::cognitive_complexity
    )]
    #[inline(never)]
    pub(super) async fn set_computed(
        mut self,
        query: DynQueryInput,
        query_value: DynValue,
        query_value_fingerprint: Compact128,
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
        clean_existing_forward_edges: bool,
        mut tx: WriteTransaction<C>,
    ) {
        self.upgrade_to_exclusive().await;

        let engine = self.engine().clone();
        let entry = engine
            .executor_registry
            .get_executor_entry_by_type_id(&self.query_id().stable_type_id());

        let transitive_firewall_callees_fingerprint =
            self.engine().hash(&tfc_achetype);

        let new_forward_edges = ForwardEdges::<C> {
            order: ForwardEdgeOrder(forward_edge_order),
            observation: ForwardEdgeObservation(Arc::new(
                forward_edge_observations,
            )),
        };

        let node_info = NodeInfo::new(
            query_kind,
            query_value_fingerprint,
            transitive_firewall_callees_fingerprint,
            tfc_achetype,
        );

        {
            // The callees that the query read when it was last computed, and
            // the ones it has read now. A callee that is in both has its
            // backward edge to this query in place already: taking the edge
            // out and putting it back in would be two writes that change
            // nothing, to a set that every other caller of that callee
            // writes to as well.
            let previous_callees: HashSet<QueryID, C::BuildHasher> =
                existing_forward_edges
                    .map(|edges| callees(edges).copied().collect())
                    .unwrap_or_default();

            // remove prior backward edges
            if let Some(forward_edges) = existing_forward_edges {
                // set of callees to maintain backward edges for
                let maintaining_callees: HashSet<QueryID, C::BuildHasher> =
                    // if there's no previous callees, then there are no backward 
                    // edges to remove
                    if previous_callees.is_empty() {
                        HashSet::default()
                    } else {
                        callees(&new_forward_edges.order.0).copied().collect()
                    };

                for callee in callees(forward_edges) {
                    if !maintaining_callees.contains(callee) {
                        self.engine()
                            .computation_graph
                            .database
                            .backward_edges
                            .remove(callee, self.query_id(), &mut tx)
                            .await;
                    }

                    if clean_existing_forward_edges {
                        let edge = Edge { from: *self.query_id(), to: *callee };

                        self.engine()
                            .computation_graph
                            .database
                            .dirty_edge_set
                            .remove(&edge, &mut tx)
                            .await;
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
                .last_verified
                .insert(
                    *self.query_id(),
                    LastVerified(current_timestamp),
                    &mut tx,
                )
                .await;

            for callee in callees(&new_forward_edges.order.0) {
                if !previous_callees.contains(callee) {
                    self.engine()
                        .computation_graph
                        .database
                        .backward_edges
                        .insert(*callee, *self.query_id(), &mut tx)
                        .await;
                }
            }

            self.engine()
                .computation_graph
                .database
                .forward_edges
                .insert(*self.query_id(), new_forward_edges, &mut tx)
                .await;

            entry
                .set_query(
                    &engine,
                    self.query_id().compact_hash_128(),
                    query,
                    query_value,
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

    /// Records the query as an input with the given value fingerprint. The
    /// caller stores the query input and its value afterwards, with
    /// [`Engine::set_query`], before it lets go of the snapshot.
    pub(super) async fn set_computed_input(
        &mut self,
        query_value_fingerprint: Compact128,
        tx: &mut WriteTransaction<C>,
        set_input: bool,
        timestamp: Timestamp,
    ) {
        let query_id = *self.query_id();

        // if have an existing forward edges, unwire the backward edges
        let existing_forward_edges = self.forward_edge_order().await.cloned();

        let empty_forward_edges = ForwardEdges::<C> {
            order: ForwardEdgeOrder(Arc::from([])),
            observation: ForwardEdgeObservation(Arc::new(
                HashMap::with_hasher(C::BuildHasher::default()),
            )),
        };

        // An input that is set is an input from then on. A query that is
        // refreshed keeps the kind it was computed with.
        let query_kind = if set_input {
            QueryKind::Input
        } else {
            self.query_kind()
                .await
                .expect("a query that is refreshed has been computed before")
        };

        let transitive_firewall_callees =
            self.engine().create_tfc(HashSet::default());
        let transitive_firewall_callees_fingerprint =
            self.engine().hash(&transitive_firewall_callees);

        let node_info = NodeInfo::new(
            query_kind,
            query_value_fingerprint,
            transitive_firewall_callees_fingerprint,
            transitive_firewall_callees,
        );

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

            self.engine()
                .computation_graph
                .database
                .last_verified
                .insert(query_id, LastVerified(timestamp), tx)
                .await;

            self.engine()
                .computation_graph
                .database
                .forward_edges
                .insert(query_id, empty_forward_edges, tx)
                .await;

            self.engine()
                .computation_graph
                .database
                .node_info
                .insert(query_id, node_info, tx)
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
        self.node_info.get(query_id).await.unwrap().kind
    }
}

#[cfg(test)]
mod test {
    use std::{sync::Barrier, thread};

    use fxhash::FxBuildHasher;
    use qbice_stable_hash::Compact128;

    use super::{
        CompressedBackwardEdgeSet, ConcurrentSet, SMALL_STORAGE_CAPACITY,
    };
    use crate::query::QueryID;

    /// A query ID told apart by `n`.
    fn query_id(n: usize) -> QueryID {
        QueryID::from_parts(Compact128::from(0), Compact128::from(n as u128))
    }

    // input: threads that insert distinct queries into one set at once
    // premise: the inserts make the set upgrade from its small storage to its
    // large one while the other threads keep inserting
    // output: every query is in the set
    #[test]
    fn concurrent_inserts_survive_the_upgrade() {
        const THREADS: usize = 8;

        // the window of the race is small, so try many sets
        for _ in 0..500 {
            let set = CompressedBackwardEdgeSet::<FxBuildHasher>::default();
            let barrier = Barrier::new(THREADS);

            thread::scope(|scope| {
                for thread in 0..THREADS {
                    let (set, barrier) = (&set, &barrier);
                    scope.spawn(move || {
                        barrier.wait();
                        for i in 0..SMALL_STORAGE_CAPACITY {
                            set.insert_element(query_id(
                                thread * SMALL_STORAGE_CAPACITY + i,
                            ));
                        }
                    });
                }
            });

            assert_eq!(set.len(), THREADS * SMALL_STORAGE_CAPACITY);
        }
    }
}
