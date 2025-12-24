use std::sync::Arc;

use dashmap::DashMap;
use qbice_serialize::{Decode, Encode};
use qbice_stable_hash::{BuildStableHasher, StableHasher};

use crate::{
    Engine, ExecutionStyle, Query,
    config::{Config, DefaultConfig},
    engine::computation_graph::{
        caller::CallerInformation, computed::Computed,
        computing_lock::ComputingLock, fast_path::FastPathResult,
        query_store::QueryStore, timestamp::TimestampManager,
    },
    executor::CyclicError,
    query::{DynValue, DynValueBox, QueryID},
};

mod caller;
mod computed;
mod computing_lock;
mod fast_path;
mod input_session;
mod query_store;
mod register_callee;
mod slow_path;
mod tfc_achetype;
mod timestamp;

type Sieve<Col, Con> = qbice_storage::sieve::Sieve<
    Col,
    <Con as Config>::Database,
    <Con as Config>::BuildHasher,
>;

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
    Default,
)]
pub enum QueryKind {
    #[default]
    Input,
    Executable(ExecutionStyle),
}

pub struct ComputationGraph<C: Config> {
    computed: Computed<C>,
    query_store: QueryStore<C>,
    computing_lock: ComputingLock,

    database: Arc<C::Database>,
    timestamp_manager: TimestampManager,
}

impl<C: Config> ComputationGraph<C> {
    pub fn new(
        db: Arc<<C as Config>::Database>,
        shard_amount: usize,
        build_hasher: C::BuildHasher,
    ) -> Self {
        const CAPACITY: usize = 10_000;
        Self {
            computed: Computed::new(
                db.clone(),
                shard_amount,
                build_hasher.clone(),
            ),
            query_store: QueryStore::new(
                CAPACITY,
                shard_amount,
                db.clone(),
                build_hasher,
            ),
            computing_lock: ComputingLock::new(),

            timestamp_manager: TimestampManager::new(&*db),
            database: db,
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
    cache: Arc<DashMap<QueryID, DynValueBox<C>>>,
    caller: CallerInformation,
}

impl<C: Config> TrackedEngine<C> {
    /// Executes a query and returns its value.
    ///
    /// This is the primary method for retrieving computed values from the
    /// engine. The engine will:
    ///
    /// 1. Check the local cache for a cached result
    /// 2. Check if the query has a valid cached result in the database
    /// 3. If not valid, execute the query's registered executor
    /// 4. Track dependencies if called from within another executor
    ///
    /// # Incremental Behavior
    ///
    /// Results are cached and reused when possible. A query is recomputed
    /// only if:
    /// - It has never been computed before
    /// - Any of its dependencies have changed since the last computation
    ///
    /// # Errors
    ///
    /// Returns [`CyclicError`] if a cyclic dependency is detected (the query
    /// directly or indirectly depends on itself).
    ///
    /// # Example
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
    /// struct Double(i64);
    /// impl Query for Double {
    ///     type Value = i64;
    /// }
    ///
    /// struct DoubleExecutor;
    /// impl<C: qbice::config::Config> Executor<Double, C> for DoubleExecutor {
    ///     async fn execute(
    ///         &self,
    ///         q: &Double,
    ///         _: &TrackedEngine<C>,
    ///     ) -> Result<i64, CyclicError> {
    ///         Ok(q.0 * 2)
    ///     }
    /// }
    ///
    /// # #[tokio::main]
    /// # async fn main() {
    /// let mut engine = Engine::<DefaultConfig>::new();
    /// engine.register_executor::<Double, _>(Arc::new(DoubleExecutor));
    ///
    /// let engine = Arc::new(engine);
    /// let tracked = engine.tracked();
    ///
    /// // Execute the query
    /// let result = tracked.query(&Double(21)).await;
    /// assert_eq!(result, Ok(42));
    ///
    /// // Subsequent queries return cached result
    /// let result2 = tracked.query(&Double(21)).await;
    /// assert_eq!(result2, Ok(42));
    /// # }
    /// ```
    ///
    /// [`CyclicError`]: crate::executor::CyclicError
    pub async fn query<Q: Query>(
        &self,
        query: &Q,
    ) -> Result<Q::Value, CyclicError> {
        // YIELD POINT: query function will be called very often, this is a
        // good point for yielding to allow cancelation.
        tokio::task::yield_now().await;

        let query_with_id = self.engine.new_query_with_id(query);

        // check local cache
        if let Some(value) = self.cache.get(&query_with_id.id) {
            // cache hit! don't have to go through central database
            let value: &dyn DynValue<C> = &**value;

            return Ok(value
                .downcast_value::<Q::Value>()
                .expect("should've been a correct type")
                .clone());
        }

        // run the main process
        self.engine
            .query_for(&query_with_id, &self.caller)
            .await
            .map(QueryResult::unwrap_return)
    }
}

impl<C: Config> Clone for TrackedEngine<C> {
    fn clone(&self) -> Self {
        Self {
            engine: Arc::clone(&self.engine),
            cache: Arc::clone(&self.cache),
            caller: self.caller,
        }
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
        TrackedEngine {
            engine: self,
            cache: Arc::new(DashMap::new()),
            caller: CallerInformation::User,
        }
    }
}

#[derive(Debug, Clone)]
pub struct QueryWithID<'c, Q: Query> {
    id: QueryID,
    query: &'c Q,
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
}

impl<C: Config> Engine<C> {
    async fn query_for<Q: Query>(
        self: &Arc<Self>,
        query: &QueryWithID<'_, Q>,
        caller: &CallerInformation,
    ) -> Result<QueryResult<Q::Value>, CyclicError> {
        // register the dependency for the sake of detecting cycles
        let undo_register = self.register_callee(caller.get_caller(), query.id);

        let mut checked = QueryRepairation::UpToDate;

        // pulling the value
        let value = loop {
            match self.fast_path::<Q>(&query.id, caller).await {
                // try again
                Ok(FastPathResult::TryAgain) => continue,

                // go to slow path
                Ok(FastPathResult::ToSlowPath) => {}

                // hit
                Ok(FastPathResult::Hit(value)) => {
                    // defuse the undo `register_callee` since we have obtained
                    // the value, record the dependency successfully
                    if let Some(undo_register) = undo_register {
                        undo_register.defuse();
                    }

                    break value.map_or_else(
                        || QueryResult::Checked(checked),
                        |v| QueryResult::Return(v, checked),
                    );
                }

                Err(e) => {
                    // defuse the undo `register_callee` keep cyclic dependency
                    // detection correct
                    if let Some(undo) = undo_register {
                        undo.defuse();
                    }

                    return Err(e);
                }
            }

            // now the `query` state is held in computing state.
            // if `lock_computing` is dropped without defusing, the state will
            // be restored to previous state (either computed or absent)
            let Some(lock_computing) = self.computing_lock_guard(&query.id)
            else {
                // try the fast path again
                continue;
            };

            self.continuation(query, caller, lock_computing).await;

            checked = QueryRepairation::Repaired;

            // retry to the fast path and obtain value.
        };

        // if cyclic dependency is detected, return error
        self.is_query_running_in_scc(caller.get_caller())?;

        Ok(value)
    }

    /// Create a new query with its associated unique identifier.
    pub(super) fn new_query_with_id<'c, Q: Query>(
        &'c self,
        query: &'c Q,
    ) -> QueryWithID<'c, Q> {
        let mut hash = self.build_stable_hasher.build_stable_hasher();
        query.stable_hash(&mut hash);

        QueryWithID { id: QueryID::new::<Q>(hash.finish().into()), query }
    }
}
