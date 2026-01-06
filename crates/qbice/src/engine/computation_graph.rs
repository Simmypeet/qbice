use std::sync::Arc;

// re-export
pub(crate) use caller::CallerInformation;
use dashmap::{DashMap, DashSet};
pub(crate) use persist::QueryDebug;
use qbice_serialize::{Decode, Encode};
use qbice_stable_hash::{BuildStableHasher, StableHasher};

use crate::{
    Engine, ExecutionStyle, Query,
    config::{Config, DefaultConfig},
    engine::computation_graph::{
        caller::CallerKind, fast_path::FastPathResult, lock::Lock,
        persist::Persist, statistic::Statistic,
    },
    executor::{CyclicError, CyclicPanicPayload},
    query::{DynValue, DynValueBox, QueryID},
};

mod backward_projection;
mod caller;
mod dirty_propagation;
mod fast_path;
mod input_session;
mod lock;
mod persist;
mod register_callee;
mod repair;
mod slow_path;
mod statistic;
mod tfc_achetype;
mod visualization;

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

impl QueryKind {
    #[must_use]
    pub const fn is_input(self) -> bool { matches!(self, Self::Input) }

    #[must_use]
    pub const fn is_firewall(self) -> bool {
        matches!(self, Self::Executable(ExecutionStyle::Firewall))
    }
    pub const fn is_external_input(self) -> bool {
        matches!(self, Self::Executable(ExecutionStyle::ExternalInput))
    }
}

pub struct ComputationGraph<C: Config> {
    persist: Persist<C>,
    lock: Lock<C>,
    dirtied_queries: DashSet<QueryID, C::BuildHasher>,
    statistic: Statistic,
}

impl<C: Config> ComputationGraph<C> {
    pub fn new(db: Arc<<C as Config>::Database>, shard_amount: usize) -> Self {
        Self {
            persist: Persist::new(db, shard_amount),
            dirtied_queries: DashSet::default(),
            statistic: Statistic::default(),
            lock: Lock::new(),
        }
    }
}

/// A wrapper around [`Arc<Engine>`] that enables query execution.
///
/// `TrackedEngine` is the primary interface for executing queries in QBICE.
/// It wraps an `Arc<Engine>` and provides dependency tracking during query
/// execution, which is essential for the incremental computation system.
///
/// # Purpose
///
/// The `TrackedEngine` serves two key purposes:
///
/// 1. **Dependency Tracking**: Records which queries depend on which other
///    queries during execution
/// 2. **Local Caching**: Maintains a fast local cache for frequently accessed
///    query results
///
/// # Creating a `TrackedEngine`
///
/// Create from an `Arc<Engine>` using the [`tracked`](Engine::tracked)
/// method:
///
/// ```rust,ignore
/// use std::sync::Arc;
/// use qbice::Engine;
///
/// let engine: Engine<_> = /* ... */;
/// let engine = Arc::new(engine);
/// let tracked = engine.tracked();
/// ```
///
/// # Executing Queries
///
/// Use the [`query`](TrackedEngine::query) method to execute queries:
///
/// ```rust,ignore
/// let result = tracked.query(&my_query).await?;
/// ```
///
/// # Thread Safety
///
/// `TrackedEngine` implements `Clone`, `Send`, and `Sync`:
///
/// - **Clone**: Cheap to clone, shares the underlying engine and local cache
/// - **Send**: Can be moved to other threads
/// - **Sync**: Can be shared across threads via `Arc` or references
///
/// This enables concurrent query execution:
///
/// ```rust,ignore
/// let tracked = engine.tracked();
///
/// // Clone for concurrent execution
/// let tracked1 = tracked.clone();
/// let tracked2 = tracked.clone();
///
/// tokio::spawn(async move {
///     tracked1.query(&query1).await
/// });
///
/// tokio::spawn(async move {
///     tracked2.query(&query2).await
/// });
/// ```
///
/// # Local Caching
///
/// Each `TrackedEngine` maintains a local cache of query results. Clones
/// share this cache:
///
/// ```text
/// TrackedEngine
///   ├─ Arc<Engine>           (shared)
///   └─ Arc<LocalCache>       (shared among clones)
/// ```
///
/// Benefits:
/// - Fast repeated access to the same query within a "session"
/// - Reduces contention on the central database
///
/// # Lifecycle Management
///
/// The typical pattern for using `TrackedEngine`:
///
/// ```rust,ignore
/// // 1. Create and use
/// let mut engine_arc = Arc::new(engine);
/// let tracked = engine_arc.clone().tracked();
/// let result = tracked.query(&query).await?;
///
/// // 2. Drop to release Arc reference
/// drop(tracked);
///
/// // 3. Get mutable access to update inputs
/// let engine = Arc::get_mut(&mut engine_arc)
///     .expect("no other references exist");
/// let mut session = engine.input_session();
/// session.set_input(input_query, new_value);
/// drop(session);
///
/// // 4. Create new TrackedEngine for next round
/// let tracked = engine_arc.clone().tracked();
/// ```
///
/// # Relationship to Engine
///
/// `TrackedEngine` doesn't own the `Engine`; it holds an `Arc` reference.
/// This design allows:
///
/// - Multiple concurrent query executors
/// - Proper cleanup of local state between input updates
/// - Clear separation between querying and modification phases
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
    pub async fn query<Q: Query>(&self, query: &Q) -> Q::Value {
        let query_with_id = self.engine.new_query_with_id(query);

        // check local cache
        if let Some(value) = self.cache.get(&query_with_id.id) {
            // cache hit! don't have to go through central database
            let value: &dyn DynValue<C> = &**value;

            return value
                .downcast_value::<Q::Value>()
                .expect("should've been a correct type")
                .clone();
        }

        // run the main process
        let result = self
            .engine
            .query_for(&query_with_id, &self.caller)
            .await
            .map(QueryResult::unwrap_return);

        // cache the result locally
        if let Ok(ref value) = result {
            let dyn_value: DynValueBox<C> = smallbox::smallbox!(value.clone());

            self.cache.insert(query_with_id.id, dyn_value);
        }

        // panic! with CyclicPanicPayload if cyclic error detected
        result.unwrap_or_else(|_| CyclicPanicPayload::unwind())
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
    /// Creates a tracked engine wrapper for executing queries.
    ///
    /// The [`TrackedEngine`] provides the primary interface for query
    /// execution via the [`query`](TrackedEngine::query) method. It wraps the
    /// engine with dependency tracking capabilities needed during query
    /// execution.
    ///
    /// # Ownership
    ///
    /// This method consumes `Arc<Self>`, taking ownership of the Arc. The
    /// returned `TrackedEngine` maintains a reference to the underlying
    /// engine.
    ///
    /// # Multiple Instances
    ///
    /// You can create multiple `TrackedEngine` instances from the same
    /// `Arc<Engine>` for concurrent querying:
    ///
    /// ```rust,ignore
    /// let engine = Arc::new(engine);
    /// let tracked1 = engine.clone().tracked();
    /// let tracked2 = engine.clone().tracked();
    ///
    /// // Both can execute queries concurrently
    /// let (result1, result2) = tokio::join!(
    ///     tracked1.query(&query1),
    ///     tracked2.query(&query2),
    /// );
    /// ```
    ///
    /// # Local Caching
    ///
    /// Each `TrackedEngine` has its own local cache for query results. Cloning
    /// a `TrackedEngine` shares this cache:
    ///
    /// ```rust,ignore
    /// let tracked1 = engine.clone().tracked();
    /// let tracked2 = tracked1.clone(); // Shares cache with tracked1
    /// ```
    ///
    /// # Lifecycle
    ///
    /// Typical usage pattern:
    ///
    /// ```rust,ignore
    /// // 1. Wrap engine in Arc and create TrackedEngine
    /// let engine = Arc::new(engine);
    /// let tracked = engine.clone().tracked();
    ///
    /// // 2. Execute queries
    /// let result = tracked.query(&my_query).await?;
    ///
    /// // 3. Drop TrackedEngine to release Arc reference
    /// drop(tracked);
    ///
    /// // 4. Modify inputs (requires exclusive access)
    /// let engine_mut = Arc::try_unwrap(engine)
    ///     .unwrap_or_else(|arc| panic!("other references exist"));
    /// // or use Arc::get_mut if keeping the Arc
    /// ```
    #[must_use]
    pub fn tracked(self: Arc<Self>) -> TrackedEngine<C> {
        TrackedEngine {
            caller: CallerInformation::new(CallerKind::User, unsafe {
                self.get_current_timestamp()
            }),
            cache: Arc::new(DashMap::new()),
            engine: self,
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
            let slow_path = match self.fast_path::<Q>(query.id, caller).await {
                // try again
                Ok(FastPathResult::TryAgain) => {
                    continue;
                }

                // go to slow path
                Ok(FastPathResult::ToSlowPath(slow_path)) => slow_path,

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
            };

            // now the `query` state is held in computing state.
            // if `lock_computing` is dropped without defusing, the state will
            // be restored to previous state (either computed or absent)
            let Some(guard) = self.get_lock_guard(query.id, slow_path) else {
                // try the fast path again
                continue;
            };

            self.continuation(query, caller, guard).await;

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
