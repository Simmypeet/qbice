use std::{any::Any, cell::RefCell, collections::HashMap, fmt, sync::Arc};

// re-export
pub(crate) use caller::CallerInformation;
use dashmap::DashSet;
pub(crate) use database::{ActiveInputSessionGuard, QueryDebug};
pub use input_session::{InputSession, SetInputResult};
use qbice_serialize::{Decode, Encode};
use qbice_stable_hash::{BuildStableHasher, StableHash, StableHasher};
use qbice_stable_type_id::{Identifiable, StableTypeID};
pub use statistics::{
    Observability, QueryStatistics, Statistics, StatisticsLevel,
    StatisticsReset,
};
use thread_local::ThreadLocal;

use crate::{
    Engine, ExecutionStyle, Query,
    config::Config,
    engine::computation_graph::{
        caller::CallerKind,
        computing::Computing,
        database::{ActiveComputationGuard, Database, ValueSlot},
        dirty_worker::DirtyWorker,
        fast_path::FastPathResult,
        query_lock_manager::QueryLockManager,
        slow_path::SlowPath,
        statistics::StatisticsRecorder,
    },
    executor::{BoxFuture, CyclicError, CyclicPanicPayload},
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
mod statistics;
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
    // `database`, `statistics_recorder`, and `dirtied_queries`.
    dirty_worker: DirtyWorker<C>,

    dirtied_queries: Arc<DashSet<QueryID, C::BuildHasher>>,

    database: Arc<Database<C>>,
    statistics_recorder: Arc<StatisticsRecorder>,

    computing: Computing<C>,
    lock_manager: QueryLockManager,

    entry_points: EntryPoints<C>,
}

type QueryForFn<C> =
    for<'a> fn(
        engine: &'a Arc<Engine<C>>,
        query: DynQuery<'a>,
        caller: &'a CallerInformation<C>,
        value_slot: Option<ValueSlot<'a, C>>,
    ) -> BoxFuture<'a, Result<QueryStatus, CyclicError>>;

type RepairTransitiveFirewallCalleesFn<C> = for<'a> fn(
    engine: &'a Arc<Engine<C>>,
    query_id: QueryID,
    caller: &'a CallerInformation<C>,
) -> BoxFuture<'a, ()>;

/// The engine functions that the methods generic over a query type call.
///
/// This primarily exists to avoid code bloat from monomorphization.
///
/// Each crate compiles its own copy of every generic function that it uses,
/// so a crate that called the engine directly from [`TrackedEngine::query`]
/// would compile all the code that runs a query, once more for that crate.
/// Calling through these pointers instead, which are taken where the engine
/// is created, compiles that code there only.
struct EntryPoints<C: Config> {
    query_for: QueryForFn<C>,
    repair_transitive_firewall_callees: RepairTransitiveFirewallCalleesFn<C>,
}

impl<C: Config> EntryPoints<C> {
    fn new() -> Self {
        Self {
            query_for: |engine, query, caller, value_slot| {
                Box::pin(engine.query_for(query, caller, value_slot))
            },
            repair_transitive_firewall_callees: |engine, query_id, caller| {
                Box::pin(
                    engine.repair_transitive_firewall_callees_for(
                        query_id, caller,
                    ),
                )
            },
        }
    }
}

impl<C: Config> ComputationGraph<C> {
    pub async fn new(
        db: &C::StorageEngine,
        observability: Observability,
    ) -> Self {
        let database = Arc::new(Database::new(db).await);
        let statistics_recorder =
            Arc::new(StatisticsRecorder::new(observability));
        let dirtied_queries =
            Arc::new(DashSet::with_hasher(C::BuildHasher::default()));

        Self {
            dirty_worker: DirtyWorker::new(
                &database,
                &statistics_recorder,
                &dirtied_queries,
            ),

            database,
            dirtied_queries,
            statistics_recorder,

            computing: Computing::new(),
            lock_manager: QueryLockManager::new(2u64.pow(14)),

            entry_points: EntryPoints::new(),
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
    caller: CallerInformation<C>,
}

impl<C: Config> TrackedEngine<C> {
    /// The name of the query type `type_id`, or `None` if it has no
    /// registered executor: see [`Engine::get_type_name_from_stable_type_id`].
    #[must_use]
    pub fn get_type_name_from_stable_type_id(
        &self,
        type_id: StableTypeID,
    ) -> Option<&'static str> {
        self.engine.get_type_name_from_stable_type_id(type_id)
    }

    /// The execution style of the query type `type_id`, or `None` if it has
    /// no registered executor: see
    /// [`Engine::get_execution_style_from_stable_type_id`].
    #[must_use]
    pub fn get_execution_style_from_stable_type_id(
        &self,
        type_id: StableTypeID,
    ) -> Option<ExecutionStyle> {
        self.engine.get_execution_style_from_stable_type_id(type_id)
    }

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
        // this is a good yield point for possible cancellation
        self.engine.yielder.tick().await;

        let query_id = self.engine.new_query_id(query);

        // check local cache
        if let Some(val) = self.cache.get_or_default().borrow().get(&query_id) {
            // directly access repository to avoid double wrapping
            return val
                .downcast_ref::<Q::Value>()
                .expect("cached value has incorrect type")
                .clone();
        }

        // run the main process
        let mut value = None;
        let result = (self.engine.computation_graph.entry_points.query_for)(
            &self.engine,
            DynQuery::new(query_id, query),
            &self.caller,
            Some(ValueSlot::new::<Q>(&mut value)),
        )
        .await;

        // panic! with CyclicPanicPayload if cyclic error detected
        if result.is_err() {
            CyclicPanicPayload::unwind();
        }

        let value = value.expect("Query did not return a value");

        // cache the result locally
        self.cache
            .get_or_default()
            .borrow_mut()
            .insert(query_id, Box::new(value.clone()));

        value
    }

    /// Repairs all transitive firewall callees of the given query.
    pub async fn repair_transitive_firewall_callees<Q: Query>(
        &self,
        query: &Q,
    ) {
        let query_id = self.engine.new_query_id(query);

        (self
            .engine
            .computation_graph
            .entry_points
            .repair_transitive_firewall_callees)(
            &self.engine,
            query_id,
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
        caller: CallerInformation<C>,
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
        let (active_computation_guard, timestamp) =
            self.acquire_active_computation_guard().await;

        TrackedEngine {
            caller: CallerInformation::new(
                CallerKind::User,
                timestamp,
                Some(active_computation_guard),
            ),
            cache: ThreadLocal::new(),
            engine: self,
        }
    }
}

pub(crate) type FmtQueryFn = fn(
    query: &(dyn Any + Send + Sync),
    f: &mut fmt::Formatter<'_>,
) -> fmt::Result;

/// A query and its ID, for the code that runs the query without knowing its
/// type.
#[derive(Clone, Copy)]
pub(crate) struct DynQuery<'a> {
    id: QueryID,
    query: &'a (dyn Any + Send + Sync),
    type_name: &'static str,
    fmt: FmtQueryFn,
}

impl<'a> DynQuery<'a> {
    pub fn new<Q: Query>(id: QueryID, query: &'a Q) -> Self {
        Self::from_parts(
            id,
            query,
            std::any::type_name::<Q>(),
            fmt_dyn_query::<Q>,
        )
    }

    pub(crate) const fn from_parts(
        id: QueryID,
        query: &'a (dyn Any + Send + Sync),
        type_name: &'static str,
        fmt: FmtQueryFn,
    ) -> Self {
        Self { id, query, type_name, fmt }
    }

    pub const fn id(&self) -> &QueryID { &self.id }

    pub const fn query(&self) -> &'a (dyn Any + Send + Sync) { self.query }

    pub const fn type_name(&self) -> &'static str { self.type_name }
}

/// Formats the query the way that its own [`fmt::Debug`] does, so that spans
/// that record the query read the same as before it lost its type.
impl fmt::Debug for DynQuery<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (self.fmt)(self.query, f)
    }
}

pub(crate) fn fmt_dyn_query<Q: Query>(
    query: &(dyn Any + Send + Sync),
    f: &mut fmt::Formatter<'_>,
) -> fmt::Result {
    fmt::Debug::fmt(query.downcast_ref::<Q>().expect("query type mismatch"), f)
}

/// Specifies whether the query is has been repaired or is up-to-date.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryStatus {
    Repaired,
    UpToDate,
}

impl<C: Config> Engine<C> {
    async fn repair_transitive_firewall_callees_for(
        self: &Arc<Self>,
        query_id: QueryID,
        caller: &CallerInformation<C>,
    ) {
        let mut snapshot = self.get_read_snapshot(query_id).await;

        // can't repair if haven't computed before
        if snapshot.last_verified().await.is_none() {
            return;
        }

        snapshot.repair_transitive_firewall_callees(caller).await;
    }

    /// Brings the query up to date, and reads its value into `value_slot`
    /// if the caller requires the value.
    async fn query_for(
        self: &Arc<Self>,
        query: DynQuery<'_>,
        caller: &CallerInformation<C>,
        mut value_slot: Option<ValueSlot<'_, C>>,
    ) -> Result<QueryStatus, CyclicError> {
        // this is a good yield point for possible cancellation
        self.yielder.tick().await;

        // register the dependency for the sake of detecting cycles
        let undo_register = self.register_callee(caller, query.id());

        let mut status = QueryStatus::UpToDate;

        // pulling the value
        loop {
            // exit SCC if any, otherwise deadlock may happen
            match self.exit_scc(query.id(), caller).await {
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
            let mut snapshot = self.get_read_snapshot(*query.id()).await;

            let slow_path =
                match snapshot.fast_path(caller, value_slot.as_mut()).await {
                    // go to slow path
                    FastPathResult::ToSlowPath(slow_path) => slow_path,

                    // hit
                    FastPathResult::Hit => {
                        // defuse the undo `register_callee` since we have
                        // obtained the value, record the dependency
                        // successfully
                        if let Some(undo_register) = undo_register {
                            undo_register.defuse();
                        }

                        break;
                    }
                };

            // The futures of the slow path below are boxed. Every call of a
            // query allocates the future of this function, and without the
            // boxes it would be as large as the largest of them, several
            // kilobytes, while a call that the fast path answers needs a
            // fraction of that. A call that gets here computes or repairs the
            // query, next to which one more allocation is nothing.
            // `process_query` boxes only the step that it runs, see there.

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
                Box::pin(snapshot.repair_transitive_firewall_callees(caller))
                    .await;

                // restore the snapshot after repair
                snapshot = self.get_read_snapshot(*query.id()).await;
            } else if let Some(query_caller) = caller.get_query_caller()
                && !query_caller.is_in_the_previous_forward_edges(*query.id())
                && query_caller.require_value()
                && slow_path == SlowPath::Repair
            {
                // See the test case in
                // `crates/integration_test/tests/firewall_with_dynamic_edge`
                // to see the edge case that this branch is needed.
                snapshot = match Box::pin(
                    snapshot.repair_uncovered_transitive_firewall_callees(
                        query_caller,
                        caller,
                    ),
                )
                .await
                {
                    Some(snapshot) => snapshot,
                    None => self.get_read_snapshot(*query.id()).await,
                };
            }

            // now the `query` state is held in computing state.
            // if `guard` is dropped without defusing, the state will
            // be restored to previous state (either computed or absent)
            let Some((snapshot, guard)) =
                Box::pin(snapshot.get_write_guard(query, slow_path, caller))
                    .await
            else {
                // try the fast path again
                continue;
            };

            snapshot.process_query(query, caller, guard).await;

            status = QueryStatus::Repaired;

            // retry to the fast path and obtain value.
        }

        // if cyclic dependency is detected, return error
        Self::is_query_running_in_scc(caller)?;

        Ok(status)
    }

    /// Computes the unique identifier of a query.
    pub(super) fn new_query_id<Q: Query>(&self, query: &Q) -> QueryID {
        let mut hash = self.build_stable_hasher.build_stable_hasher();
        query.stable_hash(&mut hash);

        QueryID::new::<Q>(hash.finish().into())
    }
}
