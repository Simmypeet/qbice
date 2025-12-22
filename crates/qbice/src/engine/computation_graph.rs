use std::{collections::HashSet, sync::Arc};

use qbice_serialize::{Decode, Encode};
use qbice_stable_hash::Compact128;
use qbice_stable_type_id::Identifiable;
use qbice_storage::kv_database::{Column, KeyOfSet, Normal};

use crate::{
    Engine, ExecutionStyle, Query,
    config::{Config, DefaultConfig},
    engine::computation_graph::{
        caller::CallerInformation, computing_lock::ComputingLock,
        query_store::QueryStore,
    },
    executor::CyclicError,
    query::{DynValue, DynValueBox, QueryID},
};

type Sieve<Col, Con> = qbice_storage::sieve::Sieve<
    Col,
    <Con as Config>::Database,
    <Con as Config>::BuildHasher,
>;

mod caller;
mod computing_lock;
mod fast_path;
mod query_store;
mod register_callee;

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Encode, Decode,
)]
struct Timestamp(u64);

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Encode, Decode,
)]
enum QueryKind {
    Input,
    Executable(ExecutionStyle),
}

impl QueryKind {
    pub fn is_projection(&self) -> bool {
        matches!(self, QueryKind::Executable(ExecutionStyle::Projection))
    }
}

type ForwardEdgeColumn = (QueryID, Vec<QueryID>);
type NodeInfoColumn = (QueryID, NodeInfo);

#[derive(Identifiable)]
struct DirtySetColumn;

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Encode, Decode,
)]
struct Edge {
    from: QueryID,
    to: QueryID,
}

impl Column for DirtySetColumn {
    type Key = Edge;
    type Value = ();
    type Mode = Normal;
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
struct NodeInfo {
    last_verified: Timestamp,
    query_kind: QueryKind,
    fingerprint: Compact128,
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Identifiable,
)]
struct BackwardEdgeColumn;

impl Column for BackwardEdgeColumn {
    type Key = QueryID;

    type Value = HashSet<QueryID>;

    type Mode = KeyOfSet<QueryID>;
}

pub struct ComputationGraph<C: Config> {
    forward_edges: Sieve<(QueryID, Arc<[QueryID]>), C>,
    node_info: Sieve<(QueryID, NodeInfo), C>,
    dirty_edge_set: Sieve<DirtySetColumn, C>,
    backward_edges: Sieve<BackwardEdgeColumn, C>,

    query_store: QueryStore<C>,
    computing_lock: ComputingLock,

    database: Arc<C::Database>,
    timestamp: Timestamp,
}

impl<C: Config> ComputationGraph<C> {
    pub fn new(
        db: Arc<<C as Config>::Database>,
        shard_amount: usize,
        build_hasher: C::BuildHasher,
    ) -> Self {
        const CAPACITY: usize = 10_000;

        Self {
            forward_edges: Sieve::<_, C>::new(
                CAPACITY,
                shard_amount,
                db.clone(),
                build_hasher.clone(),
            ),
            node_info: Sieve::<_, C>::new(
                CAPACITY,
                shard_amount,
                db.clone(),
                build_hasher.clone(),
            ),
            dirty_edge_set: Sieve::<_, C>::new(
                CAPACITY,
                shard_amount,
                db.clone(),
                build_hasher.clone(),
            ),
            backward_edges: Sieve::<_, C>::new(
                CAPACITY,
                shard_amount,
                db.clone(),
                build_hasher.clone(),
            ),

            query_store: QueryStore::new(
                CAPACITY,
                shard_amount,
                db.clone(),
                build_hasher,
            ),
            computing_lock: ComputingLock::new(),

            database: db,
            timestamp: Timestamp(0),
        }
    }
}

/// A wrapper around [`Arc<Engine>`] that enables query execution.
///
/// `TrackedEngine` is the primary interface for executing queries. It wraps
/// an `Arc<Engine>` and provides dependency tracking during query execution.
///
/// # Creating a `TrackedEngine`
///
/// Create a `TrackedEngine` from an `Arc<Engine>`:
///
/// ```rust
/// use std::sync::Arc;
///
/// use qbice::{config::DefaultConfig, engine::Engine};
///
/// let engine = Arc::new(Engine::<DefaultConfig>::new());
/// let tracked = engine.tracked();
/// ```
///
/// # Querying
///
/// Use the [`query`][Self::query] method to execute queries:
///
/// ```rust
/// use std::sync::Arc;
///
/// use qbice::{
///     Identifiable, StableHash,
///     config::DefaultConfig,
///     engine::{Engine, TrackedEngine},
///     executor::{CyclicError, Executor},
///     query::Query,
/// };
///
/// #[derive(Debug, Clone, PartialEq, Eq, Hash, StableHash, Identifiable)]
/// struct MyQuery(u64);
/// impl Query for MyQuery {
///     type Value = i64;
/// }
///
/// struct MyExecutor;
/// impl<C: qbice::config::Config> Executor<MyQuery, C> for MyExecutor {
///     async fn execute(
///         &self,
///         q: &MyQuery,
///         _: &TrackedEngine<C>,
///     ) -> Result<i64, CyclicError> {
///         Ok(q.0 as i64)
///     }
/// }
///
/// # #[tokio::main]
/// # async fn main() {
/// let mut engine = Engine::<DefaultConfig>::new();
/// engine.register_executor::<MyQuery, _>(Arc::new(MyExecutor));
///
/// let engine = Arc::new(engine);
/// let tracked = engine.tracked();
///
/// let result = tracked.query(&MyQuery(42)).await;
/// assert_eq!(result, Ok(42));
///
/// # Thread Safety
///
/// `TrackedEngine` is `Clone`, `Send`, and `Sync`. It can be cheaply cloned
/// and sent to other threads for concurrent query execution.
///
/// # Local Caching
///
/// Each `TrackedEngine` maintains a local cache of query results. This cache
/// is specific to the `TrackedEngine` instance and its clones (they share the
/// same cache). The local cache provides fast repeated access to the same
/// query within a single "session".
pub struct TrackedEngine<C: Config> {
    engine: Arc<Engine<C>>,
    caller: CallerInformation,
}

impl<C: Config> Clone for TrackedEngine<C> {
    fn clone(&self) -> Self {
        Self { engine: Arc::clone(&self.engine), caller: self.caller }
    }
}

static_assertions::assert_impl_all!(&TrackedEngine<DefaultConfig>: Send, Sync);

impl<C: Config> std::fmt::Debug for TrackedEngine<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrackedEngine")
            .field("engine", &self.engine)
            .field("caller", &self.caller)
            .finish_non_exhaustive()
    }
}

impl<C: Config> Engine<C> {
    /// Creates a tracked engine wrapper for querying.
    ///
    /// The returned [`TrackedEngine`] provides the
    /// [`query`][TrackedEngine::query] method for executing queries.
    /// Multiple `TrackedEngine` instances can be created from the same
    /// `Arc<Engine>` for concurrent querying.
    ///
    /// # Thread Safety
    ///
    /// `TrackedEngine` is cheap to clone and can be safely sent to other
    /// threads. Each clone shares the same underlying engine but has its
    /// own local cache.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::sync::Arc;
    ///
    /// use qbice::{config::DefaultConfig, engine::Engine};
    ///
    /// let engine = Arc::new(Engine::<DefaultConfig>::new());
    /// let tracked = engine.clone().tracked();
    ///
    /// // Can create multiple tracked engines
    /// let tracked2 = engine.clone().tracked();
    /// ```
    #[must_use]
    pub fn tracked(self: Arc<Self>) -> TrackedEngine<C> {
        TrackedEngine { engine: self, caller: CallerInformation::User }
    }
}

#[derive(Debug, Clone)]
pub struct QueryWithID<'c, Q: Query> {
    id: QueryID,
    query: &'c Q,
}

impl<Q: Query> QueryWithID<'_, Q> {
    pub const fn id(&self) -> &QueryID { &self.id }
    pub const fn query(&self) -> &Q { self.query }
}

/// Specifies whether the query is has been repaired or is up-to-date.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum QueryRepairation {
    /// The query verification timestamp was older than the current timestamp.
    /// The query has been repaired to be up-to-date (this doesn't imply that
    /// the query has been recomputed)
    Repaired,

    /// The verification timestamp was already up-to-date.
    UpToDate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryResult<V> {
    Return(V, QueryRepairation),
    Checked(QueryRepairation),
}

impl<V> QueryResult<V> {
    pub fn unwrap_return(self) -> V {
        match self {
            Self::Return(v, _) => v,
            Self::Checked(_) => {
                panic!("called `unwrap_return` on a `UpToDate` value")
            }
        }
    }

    /// Type erases the type parameter `V` into a `DynValueBox<C>`.
    pub fn into_dyn_query_result<C: Config>(self) -> QueryResult<DynValueBox<C>>
    where
        V: DynValue<C>,
    {
        match self {
            Self::Return(v, c) => {
                QueryResult::Return(smallbox::smallbox!(v), c)
            }
            Self::Checked(c) => QueryResult::Checked(c),
        }
    }
}

impl<C: Config> Engine<C> {
    async fn query_for<Q: Query>(
        self: &Arc<Self>,
        query: QueryWithID<'_, Q>,
        caller: &CallerInformation,
    ) -> Result<QueryResult<Q::Value>, CyclicError> {
    }
}
