use std::{collections::VecDeque, sync::Arc};

use crossbeam::sync::WaitGroup;
use dashmap::DashSet;
use qbice_stable_hash::{Compact128, StableHash};
use tokio::task::JoinSet;

use crate::{
    Engine, Query, TrackedEngine,
    config::Config,
    engine::computation_graph::{
        caller::{CallerInformation, CallerKind, CallerReason, QueryCaller},
        persist::{NodeInfo, WriterBufferWithLock},
        slow_path::GuardedTrackedEngine,
    },
    query::QueryID,
};

/// A transactional session for setting and updating input query values.
///
/// `InputSession` provides a batch interface for modifying input queries with
/// automatic dirty propagation. All changes are buffered during the session
/// lifetime and committed atomically when the session is dropped.
///
/// # Lifecycle
///
/// 1. **Creation**: Obtained via [`Engine::input_session()`]
///    - Acquires a write lock on the computation graph
///    - Increments the global timestamp to invalidate running queries
/// 2. **Modification**: Use [`set_input`](Self::set_input) or
///    [`refresh`](Self::refresh)
///    - Changes are buffered in memory
///    - Dirty queries are tracked for later propagation
/// 3. **Commit**: Automatically triggered on drop
///    - Dirty propagation executes in parallel via the Rayon thread pool
///    - Write buffer is submitted to persistent storage
///    - Statistics are reset and internal state is cleaned up
///
/// # Dirty Propagation
///
/// When the session ends, the engine:
/// - Compares new values with stored values via fingerprint hashing
/// - Marks queries whose values changed as dirty
/// - Recursively marks all dependent queries as dirty
/// - Ensures downstream queries will recompute on next access
///
/// # Concurrency
///
/// **Only one input session can be active at a time.** Creating a new session
/// while another exists will deadlock, as the write lock cannot be acquired.
///
/// The session increments the timestamp on creation, causing any in-flight
/// queries to become stale and await cancellation. This ensures consistency
/// between the input changes and query results.
///
/// # Example
///
/// ```rust,ignore
/// use qbice::Engine;
///
/// // Initial setup
/// {
///     let mut session = engine.input_session();
///     session.set_input(UserId, 42);
///     session.set_input(UserName, "Alice");
/// } // Changes committed and dirty propagation occurs here
///
/// // Query dependent values
/// let profile = engine.query(UserProfile(42)).await;
///
/// // Update an input
/// {
///     let mut session = engine.input_session();
///     session.set_input(UserName, "Alice Smith"); // Only this query marked dirty
/// } // Dirty propagation recomputes affected queries
/// ```
pub struct InputSession<C: Config> {
    engine: Arc<Engine<C>>,
    dirty_batch: VecDeque<QueryID>,
    transaction: Option<WriterBufferWithLock<C>>,
}

impl<C: Config> Drop for InputSession<C> {
    // NOTE: the commit needs to always happen, even if the user forgets to call
    // `commit()`. Thus, we perform the commit in the `Drop` impl.
    fn drop(&mut self) {
        let Some(transaction) = self.transaction.take() else {
            // the transaction has already been committed
            return;
        };

        let engine = self.engine.clone();
        let dirty_batch = std::mem::take(&mut self.dirty_batch);

        tokio::spawn(async move {
            Self::commit_internal(engine, dirty_batch, transaction).await;
        });
    }
}

impl<C: Config> InputSession<C> {
    /// Commits the input session, performing dirty propagation and submitting
    /// the write buffer.
    pub async fn commit(mut self) {
        let engine = self.engine.clone();
        let dirty_batch = std::mem::take(&mut self.dirty_batch);
        let transaction = self.transaction.take().unwrap();

        tokio::spawn(async move {
            Self::commit_internal(engine, dirty_batch, transaction).await;
        })
        .await
        .unwrap();
    }

    async fn commit_internal(
        engine: Arc<Engine<C>>,
        dirty_batch: VecDeque<QueryID>,
        mut transaction: WriterBufferWithLock<C>,
    ) {
        engine.computation_graph.reset_statistic();
        engine.clear_dirtied_queries();

        let dirty_list = engine
            .get_dirty_propagate_list_from_batch(dirty_batch.into_iter())
            .await;

        for query_id in dirty_list {
            engine.mark_dirty_forward_edge(
                query_id.caller,
                query_id.callee,
                transaction.writer_buffer(),
            );
        }

        engine.submit_write_buffer(transaction);
    }
}

impl<C: Config> std::fmt::Debug for InputSession<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InputSession")
            .field("engine", &self.engine)
            .field("dirty_batch", &self.dirty_batch)
            .finish_non_exhaustive()
    }
}

impl<C: Config> Engine<C> {
    /// Creates an input session for setting or updating input query values.
    ///
    /// An input session provides a transactional interface for modifying
    /// input values. All changes are batched and committed atomically when
    /// the session is dropped.
    ///
    /// # Lifecycle
    ///
    /// 1. **Create**: Call this method to begin a session
    /// 2. **Modify**: Use [`set_input`](InputSession::set_input) to update
    ///    values
    /// 3. **Commit**: Drop the session to apply changes and propagate dirtiness
    ///
    /// # Dirty Propagation
    ///
    /// When the session is dropped, the engine:
    /// - Compares new values with existing ones via fingerprints
    /// - Marks changed queries and their dependents as dirty
    /// - Increments the global timestamp (if any changes occurred)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use qbice::Engine;
    ///
    /// // Set initial inputs
    /// {
    ///     let mut session = engine.input_session();
    ///     session.set_input(InputA, 10);
    ///     session.set_input(InputB, 20);
    /// } // Changes committed here
    ///
    /// // Later, update an input
    /// {
    ///     let mut session = engine.input_session();
    ///     session.set_input(InputA, 15); // Triggers dirty propagation
    /// }
    /// ```
    ///
    /// # Cancellation and Timestamp Management
    ///
    /// The input session increments the global timestamp when it's created. If
    /// there're any other running queries, they will **stuck** in the pending
    /// loop forever. This signifies that the old running queries are no longer
    /// valid and must be dropped.
    ///
    /// # Deadlocks
    ///
    /// There can only be one active input session at a time. Attempting to
    /// create a new session while another is active will result in a
    /// deadlock.
    #[must_use]
    pub async fn input_session(self: &Arc<Self>) -> InputSession<C> {
        // acquire a write lock on the write buffer and increment timestamp
        let mut write_buffer_with_lock =
            self.new_write_buffer_with_write_lock().await;

        unsafe {
            self.increment_timestamp(&mut write_buffer_with_lock);
        }

        InputSession {
            dirty_batch: VecDeque::new(),
            transaction: Some(write_buffer_with_lock),
            engine: self.clone(),
        }
    }
}

/// The result of setting an input query value.
///
/// This enum indicates whether an input query was newly created, had its value
/// updated, or remained unchanged after a
/// [`set_input`](InputSession::set_input) operation.
///
/// The variant determines whether dirty propagation will occur when the input
/// session is dropped.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, StableHash,
)]
pub enum SetInputResult {
    /// The input query was set for the first time.
    ///
    /// This indicates that no prior value existed for this query. A new entry
    /// was created in the computation graph. No dirty propagation is needed
    /// since there are no dependents yet.
    Fresh,

    /// The input query's value was updated.
    ///
    /// The new value's fingerprint differs from the previously stored value,
    /// indicating a semantic change. This query and all its dependents will be
    /// marked dirty for recomputation when the input session is dropped.
    Updated,

    /// The input query's value remained unchanged.
    ///
    /// The new value's fingerprint matches the previously stored value,
    /// indicating no semantic change occurred. No dirty propagation will be
    /// performed, and dependents remain valid.
    Unchanged,
}

impl<C: Config> InputSession<C> {
    /// Sets the value for an input query.
    ///
    /// This method updates the value associated with the given query. If the
    /// new value differs from the existing value (based on fingerprint
    /// comparison), the query and its dependents will be marked as dirty for
    /// recomputation.
    ///
    /// # Behavior
    ///
    /// - **New queries**: If the query has never been set before, a new entry
    ///   is created
    /// - **Changed values**: If the value fingerprint differs from the stored
    ///   value, dirty propagation is scheduled
    /// - **Timestamp management**: The first change in a session increments the
    ///   global timestamp
    ///
    /// All dirty propagation happens when the `InputSession` is dropped, not
    /// when this method is called.
    ///
    /// # Type Parameters
    ///
    /// - `Q`: The query type, must implement [`Query`]
    ///
    /// # Arguments
    ///
    /// - `query`: The input query key
    /// - `new_value`: The new value to associate with this query
    pub async fn set_input<Q: Query>(
        &mut self,
        query: Q,
        new_value: Q::Value,
    ) -> SetInputResult {
        let query_hash = self.engine.hash(&query);
        let query_id = QueryID::new::<Q>(query_hash);

        let query_value_fingerprint = self.engine.hash(&new_value);

        // has prior node infos, check for fingerprint diff
        // also, unwire the backward edges (if any)
        let set_input_result = if let Some(node_info) =
            unsafe { self.engine.get_node_info_unchecked(query_id).await }
        {
            let fingerprint_diff =
                node_info.value_fingerprint() != query_value_fingerprint;

            // only dirty propagate if the fingerprint differs
            if fingerprint_diff {
                self.dirty_batch.push_back(query_id);
                SetInputResult::Updated
            } else {
                SetInputResult::Unchanged
            }
        } else {
            SetInputResult::Fresh
        };

        // should be safe since we're holding input session phase guard
        let ts = unsafe { self.engine.get_current_timestamp_unchecked() };

        self.engine
            .set_computed_input(
                query,
                query_hash,
                new_value,
                query_value_fingerprint,
                self.transaction.as_mut().unwrap(),
                true,
                ts,
            )
            .await;

        set_input_result
    }

    /// Refreshes all external input queries of type `Q`.
    ///
    /// This method re-executes all queries of type `Q` that were previously
    /// computed with [`ExecutionStyle::ExternalInput`]. For each query:
    ///
    /// 1. The executor is re-invoked to fetch the latest external data
    /// 2. The new result is compared with the old result via fingerprints
    /// 3. Only if the result changed, the query is marked dirty for propagation
    ///
    /// # Use Case
    ///
    /// External input queries represent data from the outside world (files,
    /// network, databases, etc.). When you know the external data has changed,
    /// call this method to refresh and update all queries of that type.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Assume ConfigFileQuery reads configuration from disk
    /// // and was executed with ExecutionStyle::ExternalInput
    ///
    /// // When you know the config file has changed:
    /// {
    ///     let mut session = engine.input_session();
    ///     session.refresh::<ConfigFileQuery>().await;
    /// } // Only changed ConfigFileQuery instances will trigger dirty
    ///   // propagation
    /// ```
    ///
    /// # Type Parameters
    ///
    /// - `Q`: The query type to refresh. Must implement [`Query`].
    ///
    /// # Cancellation
    ///
    /// This method can be cancelled. However, if cancelled midway, not all
    /// queries of type `Q` may be refreshed, potentially leaving some stale
    /// data in the computation graph. Nevertheless, it's guaranteed that the
    /// engine's internal state will remain consistent.
    #[allow(clippy::too_many_lines)]
    pub async fn refresh<Q: Query>(&mut self) {
        struct RefreshResult<K, V> {
            query_input_hash: Compact128,
            query_result_hash: Compact128,
            old_node_info: NodeInfo,
            query_input: K,
            new_value: V,
        }

        let type_id = Q::STABLE_TYPE_ID;

        // Get all query hashes for this type that were computed as
        // ExternalInput
        let external_input_set =
            self.engine.get_external_input_queries(&type_id).await;

        // should be safe since we're holding input session phase guard
        let timestamp =
            unsafe { self.engine.get_current_timestamp_unchecked() };

        let hashes = external_input_set.iter().map(|x| *x).collect::<Vec<_>>();

        let expected_parallelism = std::thread::available_parallelism()
            .map_or_else(|_| 1, std::num::NonZero::get)
            * 4;
        let chunk_size =
            std::cmp::max((hashes.len()) / expected_parallelism, 1);

        let mut join_set = JoinSet::new();

        for chunk in hashes.chunks(chunk_size) {
            let engine = self.engine.clone();
            let chunk = chunk.to_owned();

            join_set.spawn(async move {
                let mut results = Vec::with_capacity(chunk.len());

                for query_hash in chunk.iter().copied() {
                    let query_id = QueryID::new::<Q>(query_hash);

                    let (query, old_node_info) = unsafe {
                        (
                            engine
                                .get_query_input_unchecked::<Q>(query_hash)
                                .await
                                .unwrap(),
                            engine
                                .get_node_info_unchecked(query_id)
                                .await
                                .unwrap(),
                        )
                    };

                    // Create a tracked engine to execute the query
                    let cache = Arc::new(DashSet::default());
                    let wait_group = WaitGroup::new();

                    let tracked_engine = TrackedEngine::new(
                        engine.clone(),
                        cache,
                        CallerInformation::new(
                            CallerKind::Query(QueryCaller::new(
                                query_id,
                                CallerReason::RequireValue(Some(wait_group)),
                            )),
                            timestamp,
                        ),
                    );
                    let guarded_tracked_engine =
                        GuardedTrackedEngine::new(tracked_engine);

                    // Invoke the executor to get the new value
                    let entry =
                        engine.executor_registry.get_executor_entry::<Q>();
                    let result = entry
                        .invoke_executor::<Q>(&query, &guarded_tracked_engine)
                        .await;

                    // Wait for any spawned tasks to complete
                    drop(guarded_tracked_engine);

                    let new_value = match result {
                        Ok(value) => value,
                        Err(panic) => panic.resume_unwind(),
                    };

                    let new_fingerprint = engine.hash(&new_value);

                    results.push(RefreshResult {
                        query_input: query,
                        query_input_hash: query_hash,
                        query_result_hash: new_fingerprint,
                        old_node_info,
                        new_value,
                    });
                }

                results
            });
        }

        while let Some(res) = join_set.join_next().await {
            match res {
                Ok(results) => {
                    for refresh_result in results {
                        let fingerprint_diff =
                            refresh_result.old_node_info.value_fingerprint()
                                != refresh_result.query_result_hash;

                        if fingerprint_diff {
                            let query_id = QueryID::new::<Q>(
                                refresh_result.query_input_hash,
                            );

                            self.dirty_batch.push_back(query_id);
                        }

                        self.engine
                            .set_computed_input(
                                refresh_result.query_input,
                                refresh_result.query_input_hash,
                                refresh_result.new_value,
                                refresh_result.query_result_hash,
                                self.transaction.as_mut().unwrap(),
                                false,
                                timestamp,
                            )
                            .await;
                    }
                }

                Err(er) => match er.try_into_panic() {
                    Ok(panic_reason) => {
                        std::panic::resume_unwind(panic_reason);
                    }
                    Err(er) => {
                        panic!("Failed to refresh external input query: {er}");
                    }
                },
            }
        }
    }

    /// Interns a value, returning a reference-counted handle to the shared
    /// allocation.
    ///
    /// This is a delegation to [`Engine::intern`]. See its documentation for
    /// more details.
    pub fn intern<
        T: StableHash + crate::Identifiable + Send + Sync + 'static,
    >(
        &self,
        value: T,
    ) -> qbice_storage::intern::Interned<T> {
        self.engine.intern(value)
    }

    /// Interns an unsized value, returning a reference-counted handle to the
    /// shared allocation.
    ///
    /// This is a delegation to [`Engine::intern_unsized`]. See its
    /// documentation for more details.
    pub fn intern_unsized<
        T: StableHash + crate::Identifiable + Send + Sync + 'static + ?Sized,
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
