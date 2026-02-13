use std::{any::Any, cell::RefCell, collections::HashMap, sync::Arc};

// re-export
pub(crate) use caller::CallerInformation;
use dashmap::DashSet;
pub(crate) use database::{ActiveInputSessionGuard, QueryDebug};
pub use input_session::{InputSession, SetInputResult};
use qbice_serialize::{Decode, Encode};
use qbice_stable_hash::{BuildStableHasher, StableHash, StableHasher};
use qbice_stable_type_id::Identifiable;
pub(crate) use slow_path::GuardedTrackedEngine;
use thread_local::ThreadLocal;

use crate::{
    Engine, ExecutionStyle, Query,
    config::Config,
    engine::computation_graph::{
        caller::CallerKind,
        computing::Computing,
        database::{ActiveComputationGuard, Database},
        dirty_worker::DirtyWorker,
        fast_path::FastPathResult,
        query_lock_manager::QueryLockManager,
        slow_path::SlowPath,
        statistic::Statistic,
    },
    executor::{CyclicError, CyclicPanicPayload},
    query::QueryID,
};

mod backward_projection;
mod caller;
mod computing;
mod database;
mod dirty_worker;
mod fast_path;
mod input_session;
mod query_lock_manager;
mod register_callee;
mod repair;
mod slow_path;
mod statistic;
mod tfc_achetype;
mod visualization;

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
    // NOTE: we drop the dirty worker first as it holds Weak references to
    // `database`, `statistic`, and `dirtied_queries`.
    dirty_worker: DirtyWorker<C>,

    dirtied_queries: Arc<DashSet<QueryID, C::BuildHasher>>,

    database: Arc<Database<C>>,
    statistic: Arc<Statistic>,

    computing: Computing<C>,
    lock_manager: QueryLockManager,
}

impl<C: Config> ComputationGraph<C> {
    pub async fn new(db: &C::StorageEngine) -> Self {
        let database = Arc::new(Database::new(db).await);
        let statistic = Arc::new(Statistic::default());
        let dirtied_queries =
            Arc::new(DashSet::with_hasher(C::BuildHasher::default()));

        Self {
            dirty_worker: DirtyWorker::new(
                &database,
                &statistic,
                &dirtied_queries,
            ),

            database,
            dirtied_queries,
            statistic,

            computing: Computing::new(),
            lock_manager: QueryLockManager::new(2u64.pow(14)),
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
/// method.
///
/// # Executing Queries
///
/// Use the [`query`](TrackedEngine::query) method to execute queries.
///
/// # Thread Safety
///
/// `TrackedEngine` implements `Clone`, `Send`, and `Sync`:
///
/// - **Clone**: Cheap to clone, shares the underlying engine and local cache
/// - **Send**: Can be moved to other threads
/// - **Sync**: Can be shared across threads via `Arc` or references
///
/// This enables concurrent query execution from multiple threads.
///
/// # Local Caching
///
/// Each `TrackedEngine` maintains a local cache of query results. Clones
/// share this cache:
///
/// - Fast repeated access to the same query within a "session"
/// - Reduces contention on the central database
///
/// # Lifecycle Management
///
/// The typical pattern for using `TrackedEngine`:
///
/// 1. Wrap engine in Arc and create `TrackedEngine` via `tracked()`
/// 2. Execute queries via `query()`
/// 3. Drop `TrackedEngine` to release Arc reference
/// 4. Modify inputs via `InputSession`
/// 5. Create new `TrackedEngine` for next round
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
    cache: ThreadLocal<RefCell<HashMap<QueryID, Box<dyn Any + Send + Sync>>>>,
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
        if let Some(val) =
            self.cache.get_or_default().borrow().get(&query_with_id.id)
        {
            // directly access repository to avoid double wrapping
            return val
                .downcast_ref::<Q::Value>()
                .expect("cached value has incorrect type")
                .clone();
        }

        // run the main process
        let result = self
            .engine
            .query_for(&query_with_id, &self.caller)
            .await
            .map(QueryResult::unwrap_return);

        // cache the result locally
        if let Ok(value) = &result {
            self.cache
                .get_or_default()
                .borrow_mut()
                .insert(query_with_id.id, Box::new(value.clone()));
        }

        // panic! with CyclicPanicPayload if cyclic error detected
        result.unwrap_or_else(|_| CyclicPanicPayload::unwind())
    }

    /// Repairs all transitive firewall callees of the given query.
    pub async fn repair_transitive_firewall_callees<Q: Query>(
        &self,
        query: &Q,
    ) {
        let query_with_id = self.engine.new_query_with_id(query);

        self.engine
            .repair_transitive_firewall_callees_for(
                &query_with_id,
                &self.caller,
            )
            .await;
    }

    /// Interns a value, returning a reference-counted handle to the shared
    /// allocation.
    ///
    /// This is a delegation to [`Interner::intern`]. See its documentation for
    /// more details.
    ///
    /// [`Interner::intern`]: qbice_storage::intern::Interner::intern
    pub fn intern<T: StableHash + Identifiable + Send + Sync + 'static>(
        &self,
        value: T,
    ) -> qbice_storage::intern::Interned<T> {
        self.engine.intern(value)
    }

    /// Interns an unsized value, returning a reference-counted handle to the
    /// shared allocation.
    ///
    /// This is a delegation to [`Interner::intern_unsized`]. See its
    /// documentation for more details.
    ///
    /// [`Interner::intern_unsized`]: qbice_storage::intern::Interner::intern_unsized
    pub fn intern_unsized<
        T: StableHash + Identifiable + Send + Sync + 'static + ?Sized,
        Q: std::borrow::Borrow<T> + Send + Sync + 'static,
    >(
        &self,
        value: Q,
    ) -> qbice_storage::intern::Interned<T>
    where
        Arc<T>: From<Q>,
    {
        self.engine.intern_unsized(value)
    }
}

impl<C: Config> TrackedEngine<C> {
    /// Creates a new `TrackedEngine` with the given engine, cache, and caller
    /// information.
    ///
    /// This is an internal constructor used for refresh operations on external
    /// input queries.
    pub(crate) const fn new(
        engine: Arc<Engine<C>>,
        caller: CallerInformation,
    ) -> Self {
        Self { engine, cache: ThreadLocal::new(), caller }
    }
}

impl<C: Config> Clone for TrackedEngine<C> {
    fn clone(&self) -> Self {
        Self {
            engine: self.engine.clone(),
            cache: ThreadLocal::new(),
            caller: self.caller.clone(),
        }
    }
}

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
    /// `Arc<Engine>` by cloning the Arc before calling `tracked()`.
    /// This enables concurrent querying.
    ///
    /// # Local Caching
    ///
    /// Each `TrackedEngine` has its own local cache for query results. Cloning
    /// a `TrackedEngine` creates a new cache instance.
    ///
    /// # Lifecycle
    ///
    /// Typical usage pattern:
    ///
    /// 1. Wrap engine in Arc and create `TrackedEngine`
    /// 2. Execute queries
    /// 3. Drop `TrackedEngine` to release Arc reference
    /// 4. Modify inputs via `InputSession`
    #[must_use]
    #[allow(clippy::unused_async)]
    pub async fn tracked(self: Arc<Self>) -> TrackedEngine<C> {
        TrackedEngine {
            caller: CallerInformation::new(
                CallerKind::User,
                unsafe { self.get_current_timestamp_from_engine().await },
                Some(self.acquire_active_computation_guard().await),
            ),
            cache: ThreadLocal::new(),
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryStatus {
    Repaired,
    UpToDate,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryResult<V> {
    pub return_value: Option<V>,
    pub status: QueryStatus,
}

impl<V> QueryResult<V> {
    pub fn unwrap_return(self) -> V {
        self.return_value.expect("Query did not return a value")
    }
}

impl<C: Config> Engine<C> {
    async fn repair_transitive_firewall_callees_for<Q: Query>(
        self: &Arc<Self>,
        query: &QueryWithID<'_, Q>,
        caller: &CallerInformation,
    ) {
        let mut snapshot =
            self.get_read_snapshot::<Q>(query.id.compact_hash_128()).await;

        // can't repair if haven't computed before
        if snapshot.last_verified().await.is_none() {
            return;
        }

        snapshot.repair_transitive_firewall_callees(caller).await;
    }

    async fn query_for<Q: Query>(
        self: &Arc<Self>,
        query: &QueryWithID<'_, Q>,
        caller: &CallerInformation,
    ) -> Result<QueryResult<Q::Value>, CyclicError> {
        // register the dependency for the sake of detecting cycles
        let undo_register = self.register_callee(caller, &query.id);

        let mut status = QueryStatus::UpToDate;

        // pulling the value
        let value = loop {
            // exit SCC if any, otherwise deadlock may happen
            match self.exit_scc(&query.id, caller).await {
                // continue to process
                Ok(_) => {}

                Err(err) => {
                    // defuse the undo `register_callee` keep cyclic dependency
                    // detection correct
                    if let Some(undo) = undo_register {
                        undo.defuse();
                    }

                    return Err(err);
                }
            }

            // acquire read snapshot
            let mut snapshot =
                self.get_read_snapshot::<Q>(query.id.compact_hash_128()).await;

            let slow_path = match snapshot.fast_path(caller).await {
                // go to slow path
                FastPathResult::ToSlowPath(slow_path) => slow_path,

                // hit
                FastPathResult::Hit(value) => {
                    // defuse the undo `register_callee` since we have obtained
                    // the value, record the dependency successfully
                    if let Some(undo_register) = undo_register {
                        undo_register.defuse();
                    }

                    break QueryResult { return_value: value, status };
                }
            };

            // if the caller is responsible for repairing TFC queries, do it
            // but release the lock first to avoid deadlock when backward
            // projection is needed.
            //
            // we repair the TFC outside the query lock which means there can
            // be duplicated "requests" to repair the same TFC query, but
            // the work will "not be duplicated" since to repair a TFC query, we
            // need to acquire its query lock first.
            if matches!(
                caller.kind(),
                CallerKind::User | CallerKind::RepairFirewall
            ) && slow_path == SlowPath::Repair
            {
                snapshot.repair_transitive_firewall_callees(caller).await;

                // restore the snapshot after repair
                snapshot = self
                    .get_read_snapshot::<Q>(query.id.compact_hash_128())
                    .await;
            }

            // now the `query` state is held in computing state.
            // if `guard` is dropped without defusing, the state will
            // be restored to previous state (either computed or absent)
            let Some((snapshot, guard)) =
                snapshot.get_write_guard(slow_path, caller).await
            else {
                // try the fast path again
                continue;
            };

            snapshot.process_query(query.query, caller, guard).await;

            status = QueryStatus::Repaired;

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
