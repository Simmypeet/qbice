use std::{collections::VecDeque, sync::Arc};

use crossbeam::sync::WaitGroup;
use dashmap::DashSet;
use parking_lot::RwLock;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    Engine, Query, TrackedEngine,
    config::Config,
    engine::computation_graph::{
        caller::{CallerInformation, CallerKind, CallerReason, QueryCaller},
        persist::WriterBufferWithLock,
        slow_path::GuardedTrackedEngine,
    },
    query::QueryID,
};

pub struct InputSession<'x, C: Config> {
    engine: &'x Arc<Engine<C>>,
    dirty_batch: VecDeque<QueryID>,
    transaction: Option<WriterBufferWithLock<'x, C>>,
}

impl<C: Config> Drop for InputSession<'_, C> {
    fn drop(&mut self) {
        self.engine.computation_graph.reset_statistic();
        self.engine.clear_dirtied_queries();

        let mut tx = self.transaction.take().unwrap();
        let tx_rwlock = RwLock::new(tx.writer_buffer());

        self.engine.rayon_thread_pool.install(|| {
            self.dirty_batch.par_iter().for_each(|x| {
                self.engine.dirty_propagate(*x, &tx_rwlock);
            });
        });

        self.engine.submit_write_buffer(tx);
    }
}

impl<C: Config> std::fmt::Debug for InputSession<'_, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InputSession")
            .field("engine", self.engine)
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
    pub fn input_session(self: &Arc<Self>) -> InputSession<'_, C> {
        // acquire a write lock on the write buffer and increment timestamp
        let mut write_buffer_with_lock =
            self.new_write_buffer_with_write_lock();

        unsafe {
            self.increment_timestamp(&mut write_buffer_with_lock);
        }

        InputSession {
            dirty_batch: VecDeque::new(),
            transaction: Some(write_buffer_with_lock),
            engine: self,
        }
    }
}

impl<C: Config> InputSession<'_, C> {
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
    pub fn set_input<Q: Query>(&mut self, query: Q, new_value: Q::Value) {
        let query_hash = self.engine.hash(&query);
        let query_id = QueryID::new::<Q>(query_hash);

        let query_value_fingerprint = self.engine.hash(&new_value);

        // has prior node infos, check for fingerprint diff
        // also, unwire the backward edges (if any)
        if let Some(node_info) =
            unsafe { self.engine.get_node_info_unchecked(query_id) }
        {
            let fingerprint_diff =
                node_info.value_fingerprint() != query_value_fingerprint;

            // only dirty propagate if the fingerprint differs
            if fingerprint_diff {
                self.dirty_batch.push_back(query_id);
            }
        }

        let ts = self.transaction.as_ref().unwrap().timestamp();

        self.engine.set_computed_input(
            query,
            query_hash,
            new_value,
            query_value_fingerprint,
            self.transaction.as_mut().unwrap(),
            true,
            ts,
        );
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
    pub async fn refresh<Q: Query>(&mut self) {
        let type_id = Q::STABLE_TYPE_ID;

        // Get all query hashes for this type that were computed as
        // ExternalInput
        let external_input_set =
            self.engine.get_external_input_queries(&type_id);

        let timestamp = self.transaction.as_ref().unwrap().timestamp();

        // Re-execute each external input query and check for changes
        for hash_ref in &external_input_set {
            let query_hash = *hash_ref;
            let query_id = QueryID::new::<Q>(query_hash);

            let (query, old_node_info) = unsafe {
                (
                    self.engine
                        .get_query_input_unchecked::<Q>(query_hash)
                        .unwrap(),
                    self.engine.get_node_info_unchecked(query_id).unwrap(),
                )
            };

            // Create a tracked engine to execute the query
            let cache = Arc::new(DashSet::default());
            let wait_group = WaitGroup::new();

            let tracked_engine = TrackedEngine::new(
                Arc::clone(self.engine),
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
            let entry = self.engine.executor_registry.get_executor_entry::<Q>();
            let result = entry
                .invoke_executor::<Q>(&query, &guarded_tracked_engine)
                .await;

            // Wait for any spawned tasks to complete
            drop(guarded_tracked_engine);

            let new_value = match result {
                Ok(value) => value,
                Err(panic) => panic.resume_unwind(),
            };

            let new_fingerprint = self.engine.hash(&new_value);

            // Check if the value has changed
            let value_changed =
                old_node_info.value_fingerprint() != new_fingerprint;

            if value_changed {
                // Add to dirty batch for propagation
                self.dirty_batch.push_back(query_id);
            }

            // Store the new result
            self.engine.set_computed_input(
                query,
                query_hash,
                new_value,
                new_fingerprint,
                self.transaction.as_mut().unwrap(),
                false,
                timestamp,
            );
        }
    }
}
