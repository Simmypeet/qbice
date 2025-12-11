use std::{any::Any, collections::VecDeque, pin::Pin, sync::Arc};

use crate::{
    config::Config,
    engine::{
        Engine, TrackedEngine,
        database::storage::{SetInputResult, Storage},
        meta::{self, QueryWithID},
        tfc_archetype::{TfcArchetype, TfcArchetypeID, TfcSet},
    },
    executor::CyclicError,
    query::{DynValue, Query, QueryID},
};

pub mod storage;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Timtestamp(u64);

impl Timtestamp {
    pub const fn increment(&mut self) { self.0 += 1; }
}

pub struct Database<C: Config> {
    storage: Storage<C>,

    tfc_archetype: TfcArchetype,
}

impl<C: Config> Default for Database<C> {
    fn default() -> Self {
        Self { storage: Storage::default(), tfc_archetype: TfcArchetype::new() }
    }
}

/// A drop guard for undoing the registration of a callee query.
///
/// This aims to ensure cancelation safety in case of the task being yielded and
/// canceled mid query.
pub struct UndoRegisterCallee<'d, 'c, C: Config> {
    database: &'d Database<C>,
    caller: Option<&'c Caller>,
    callee: QueryID,
    defused: bool,
}

impl<'d, 'c, C: Config> UndoRegisterCallee<'d, 'c, C> {
    /// Creates a new [`UndoRegisterCallee`] instance.
    pub const fn new(
        database: &'d Database<C>,
        caller_source: Option<&'c Caller>,
        callee: QueryID,
    ) -> Self {
        Self { database, caller: caller_source, callee, defused: false }
    }

    /// Don't undo the registration when dropped.
    pub fn defuse(mut self) { self.defused = true; }
}

impl<C: Config> Drop for UndoRegisterCallee<'_, '_, C> {
    fn drop(&mut self) {
        if self.defused {
            return;
        }

        if let Some(caller) = self.caller {
            let caller_meta = self.database.get_computing_caller(caller);

            caller_meta.remove_callee(&self.callee);
        }
    }
}

impl<C: Config> Engine<C> {
    fn register_callee<'c>(
        &self,
        caller: Option<&'c Caller>,
        calee: QueryID,
    ) -> Option<UndoRegisterCallee<'_, 'c, C>> {
        // record the dependency first, don't necessary need to figure out
        // the observed value fingerprint yet
        caller.map_or_else(
            || None,
            |caller| {
                let caller_meta = self.database.get_computing_caller(caller);

                caller_meta.add_callee(calee);

                Some(UndoRegisterCallee::new(
                    &self.database,
                    Some(caller),
                    calee,
                ))
            },
        )
    }

    fn is_in_scc(
        &self,
        called_from: Option<&Caller>,
    ) -> Result<(), CyclicError> {
        let Some(called_from) = called_from else {
            return Ok(());
        };

        if self.database.is_query_running_in_scc(&called_from.0) {
            return Err(CyclicError);
        }

        Ok(())
    }

    async fn query_for<Q: Query>(
        self: &Arc<Self>,
        query: &QueryWithID<'_, Q>,
        required_value: bool,
        caller: Option<&Caller>,
    ) -> Result<Option<Q::Value>, CyclicError> {
        // register the dependency for the sake of detecting cycles
        let undo_register = self.register_callee(caller, *query.id());

        // pulling the value
        let value = loop {
            match self
                .database
                .fast_path::<Q::Value>(query.id(), required_value, caller)
                .await
            {
                // try again
                Ok(meta::FastPathResult::TryAgain) => continue,

                // will continue down there
                Ok(meta::FastPathResult::ToSlowPath) => {}

                Ok(meta::FastPathResult::Hit(value)) => {
                    // defuse the undo `register_callee` since we have obtained
                    // the value, record the dependency successfully
                    if let Some(undo_register) = undo_register {
                        undo_register.defuse();
                    }

                    break value;
                }

                Err(e) => {
                    // defuse the undo `register_callee` keep cyclic depedency
                    // detection correct
                    if let Some(undo_register) = undo_register {
                        undo_register.defuse();
                    }

                    return Err(e);
                }
            }

            // now the `query` state is held in computing state.
            // if `lock_computing` is dropped without defusing, the state will
            // be restored to previous state (either computed or absent)
            let Some(lock_computing) =
                self.database.lock_computing(query, &self.executor_registry)
            else {
                // try the fast path again
                continue;
            };

            // retry to the fast path and obtain value.
            self.continuation(query, lock_computing).await;
        };

        // check before returning the value
        self.is_in_scc(caller)?;

        Ok(value)
    }

    pub(crate) fn recursively_repair_query<'a, Q: Query>(
        engine: &'a Arc<Self>,
        key: &'a dyn Any,
        caller: &'a QueryID,
    ) -> Pin<
        Box<
            dyn std::future::Future<Output = Result<(), CyclicError>>
                + Send
                + 'a,
        >,
    > {
        let key = key
            .downcast_ref::<Q>()
            .expect("should be of the correct query type");

        Box::pin(async move {
            let query_with_id = engine.database.new_query_with_id(key);

            let caller = Caller::new(*caller);

            engine.query_for(&query_with_id, false, Some(&caller)).await?;

            Ok(())
        })
    }
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

        let query_with_id = self.engine.database.new_query_with_id(query);

        // check local cache
        if let Some(value) = self.cache.get(query_with_id.id()) {
            // cache hit! don't have to go through central database
            let value: &dyn DynValue<C> = &**value;

            return Ok(value
                .downcast_value::<Q::Value>()
                .expect("should've been a correct type")
                .clone());
        }

        // run the main process
        self.engine
            .query_for(&query_with_id, true, self.caller.as_ref())
            .await
            .map(|x| x.unwrap())
    }
}

/// A session for setting input values in the engine.
///
/// Input sessions provide a way to set initial values or update existing
/// input queries. When the session is dropped, it triggers dirty propagation
/// to mark affected queries for recomputation.
///
/// # Creating an Input Session
///
/// Create an input session from a mutable reference to the engine:
///
/// ```rust
/// use qbice::{config::DefaultConfig, engine::Engine};
///
/// let mut engine = Engine::<DefaultConfig>::new();
/// let mut session = engine.input_session();
/// // Set inputs here
/// drop(session); // Or let it go out of scope
/// ```
///
/// # Input Queries
///
/// Input queries are leaf nodes in the dependency graph. They don't depend
/// on other queries and their values are set directly via this session.
///
/// ```rust
/// use qbice::{
///     Identifiable, StableHash, config::DefaultConfig, engine::Engine,
///     query::Query,
/// };
///
/// // Define an input query
/// #[derive(Debug, Clone, PartialEq, Eq, Hash, StableHash, Identifiable)]
/// struct ConfigValue(String);
/// impl Query for ConfigValue {
///     type Value = i64;
/// }
///
/// let mut engine = Engine::<DefaultConfig>::new();
///
/// // Set input values
/// {
///     let mut session = engine.input_session();
///     session.set_input(ConfigValue("max_connections".into()), 100);
///     session.set_input(ConfigValue("timeout_ms".into()), 5000);
/// } // Dirty propagation happens here
/// ```
///
/// # Batching
///
/// Multiple `set_input` calls within the same session are batched together.
/// Dirty propagation only occurs once when the session is dropped, making
/// bulk updates efficient.
#[derive(Debug)]
pub struct InputSession<'x, C: Config> {
    engine: &'x mut Engine<C>,
    incremented: bool,
    dirty_batch: VecDeque<QueryID>,
}

impl<C: Config> Engine<C> {
    /// Creates an input session for setting query input values.
    ///
    /// The returned session allows you to set values for input queries. When
    /// the session is dropped, dirty propagation is triggered for any changed
    /// inputs.
    ///
    /// # Example
    ///
    /// ```rust
    /// use qbice::{
    ///     Identifiable, StableHash, config::DefaultConfig, engine::Engine,
    ///     query::Query,
    /// };
    ///
    /// #[derive(Debug, Clone, PartialEq, Eq, Hash, StableHash, Identifiable)]
    /// struct Input(u64);
    /// impl Query for Input {
    ///     type Value = i64;
    /// }
    ///
    /// let mut engine = Engine::<DefaultConfig>::new();
    ///
    /// // Create a session and set inputs
    /// {
    ///     let mut session = engine.input_session();
    ///     session.set_input(Input(0), 42);
    ///     session.set_input(Input(1), 100);
    /// }
    /// // Dirty propagation happens when session is dropped
    /// ```
    #[must_use]
    pub const fn input_session(&mut self) -> InputSession<'_, C> {
        InputSession {
            engine: self,
            incremented: false,
            dirty_batch: VecDeque::new(),
        }
    }
}

impl<C: Config> InputSession<'_, C> {
    /// Sets an input value for a query.
    ///
    /// This method stores the value for the given query key. If the value
    /// differs from the previously stored value (based on fingerprint
    /// comparison), the query will be marked for dirty propagation when
    /// the session is dropped.
    ///
    /// # Type Parameters
    ///
    /// - `Q`: The query type (must implement [`Query`])
    ///
    /// # Arguments
    ///
    /// - `query_key`: The query key to set the value for
    /// - `value`: The value to store
    ///
    /// # Example
    ///
    /// ```rust
    /// use qbice::{
    ///     Identifiable, StableHash, config::DefaultConfig, engine::Engine,
    ///     query::Query,
    /// };
    ///
    /// #[derive(Debug, Clone, PartialEq, Eq, Hash, StableHash, Identifiable)]
    /// struct Counter(u64);
    /// impl Query for Counter {
    ///     type Value = i64;
    /// }
    ///
    /// let mut engine = Engine::<DefaultConfig>::new();
    ///
    /// {
    ///     let mut session = engine.input_session();
    ///
    ///     // Set initial value
    ///     session.set_input(Counter(0), 0);
    ///
    ///     // Setting the same value again won't trigger recomputation
    ///     session.set_input(Counter(0), 0);
    ///
    ///     // Setting a different value will trigger dirty propagation
    ///     session.set_input(Counter(0), 1);
    /// }
    /// ```
    ///
    /// [`Query`]: crate::query::Query
    pub fn set_input<Q: Query>(&mut self, query_key: Q, value: Q::Value) {
        let query_id = self.engine.database.query_id(&query_key);

        let SetInputResult { incremented, fingerprint_diff } = self
            .engine
            .database
            .set_input(query_key, query_id, value, self.incremented);

        self.incremented |= incremented;

        if fingerprint_diff {
            // insert into dirty batch
            self.dirty_batch.push_back(query_id);
        }
    }
}

impl<C: Config> Drop for InputSession<'_, C> {
    fn drop(&mut self) {
        // mark all dirty queries as dirty
        self.engine
            .database
            .dirty_queries(std::mem::take(&mut self.dirty_batch));
    }
}

/// A type representing the caller of a query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Caller(QueryID);

impl Caller {
    /// Creates a new [`Caller`] instance.
    pub const fn new(query_id: QueryID) -> Self { Self(query_id) }

    /// Returns the query ID of the caller.
    pub const fn query_id(&self) -> &QueryID { &self.0 }
}

impl<C: Config> Database<C> {
    pub(super) fn new_singleton_tfc(
        &self,
        query_id: QueryID,
    ) -> TfcArchetypeID {
        self.tfc_archetype.new_singleton_tfc(query_id, self.initial_seed())
    }

    pub(super) fn union_tfcs(
        &self,
        observees: impl IntoIterator<Item = TfcArchetypeID>,
    ) -> Option<TfcArchetypeID> {
        self.tfc_archetype.observes_other_tfc(observees, self.initial_seed())
    }

    pub(super) fn get_tfc_set_by_id(
        &self,
        tfc_id: &TfcArchetypeID,
    ) -> Arc<TfcSet> {
        self.tfc_archetype.get_by_id(tfc_id).unwrap()
    }
}
