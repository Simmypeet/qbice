use std::sync::Arc;

// re-export
pub(crate) use caller::CallerInformation;
use dashmap::{DashMap, DashSet};
pub(crate) use persist::query_store::QueryDebug;
use qbice_serialize::{Decode, Encode};
use qbice_stable_hash::{BuildStableHasher, StableHasher};

use crate::{
    Engine, ExecutionStyle, Query,
    config::{Config, DefaultConfig},
    engine::computation_graph::{
        fast_path::FastPathResult, lock::Lock, persist::Persist,
        statistic::Statistic, timestamp::TimestampManager,
    },
    executor::CyclicError,
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
mod timestamp;
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
    timestamp_manager: TimestampManager,
}

impl<C: Config> ComputationGraph<C> {
    pub fn new(db: Arc<<C as Config>::Database>, shard_amount: usize) -> Self {
        Self {
            timestamp_manager: TimestampManager::new(&*db),
            persist: Persist::new(db, shard_amount),
            dirtied_queries: DashSet::default(),
            statistic: Statistic::default(),
            lock: Lock::new(),
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
            let slow_path = match self.fast_path::<Q>(&query.id, caller).await {
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
            let Some(guard) = self.get_lock_guard(&query.id, slow_path) else {
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
