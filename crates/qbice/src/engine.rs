//! Core engine types for query execution and storage.
//!
//! This module provides the central database engine ([`Engine`]) and the
//! tracked engine wrapper ([`crate::TrackedEngine`]) for executing queries.
//!
//! # Architecture Overview
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                        Engine<C>                                │
//! │  ┌─────────────────────┐    ┌─────────────────────────────┐     │
//! │  │   Query Database    │    │    Executor Registry        │     │
//! │  │  - Cached values    │    │  - Query type → Executor   │     │
//! │  │  - Dependencies     │    └─────────────────────────────┘     │
//! │  │  - Dirty flags      │                                        │
//! │  └─────────────────────┘                                        │
//! └─────────────────────────────────────────────────────────────────┘
//!                              │
//!                              │ Arc::new(engine).tracked()
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    TrackedEngine<C>                             │
//! │  - Reference to Engine                                          │
//! │  - Local query cache                                            │
//! │  - Caller tracking for dependencies                             │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Lifecycle
//!
//! 1. **Create**: Instantiate an `Engine` with your configuration
//! 2. **Register**: Add executors for each query type
//! 3. **Wrap**: Convert to `Arc<Engine>` and create `TrackedEngine`
//! 4. **Input**: Set initial input values via `InputSession`
//! 5. **Query**: Execute queries through `TrackedEngine`
//! 6. **Update**: Drop `TrackedEngine`, modify inputs, repeat from step 4

use std::{
    sync::{Arc, OnceLock},
    time::Duration,
};

use qbice_serialize::Plugin;
use qbice_stable_hash::{
    BuildStableHasher, Compact128, StableHash, StableHasher,
};
use qbice_stable_type_id::Identifiable;
use qbice_storage::{intern::Interner, storage_engine::StorageEngineFactory};

use crate::{
    config::Config,
    engine::computation_graph::ComputationGraph,
    executor::{Executor, Registry},
    query::Query,
};

pub(super) mod computation_graph;
pub(super) mod guard;

/// The central query database engine.
///
/// `Engine` is the core component of QBICE, serving as the orchestrator for
/// incremental computation. It manages:
///
/// - **Query result caching**: Stores computed values in memory and on disk
/// - **Dependency tracking**: Records which queries depend on which other
///   queries
/// - **Change propagation**: Marks affected queries as dirty when inputs change
/// - **Executor coordination**: Routes query execution to registered executors
/// - **Persistence**: Saves and loads computation state across program runs
///
/// # Architecture
///
/// The engine consists of several key components:
///
/// - **Computation Graph**: Tracks query dependencies and verification
///   timestamps
/// - **Value Cache**: Stores computed results with fingerprints for change
///   detection
/// - **Executor Registry**: Maps query types to their computation logic
/// - **Database Backend**: Persists query results and metadata
///
/// # Usage Pattern
///
/// The typical lifecycle of an engine involves these steps:
///
/// 1. **Creation**: Instantiate with [`new_with`](Engine::new_with)
/// 2. **Registration**: Add executors via
///    [`register_executor`](Engine::register_executor)
/// 3. **Input Setup**: Set initial values with
///    [`input_session`](Engine::input_session)
/// 4. **Wrapping**: Convert to `Arc<Engine>` for shared ownership
/// 5. **Querying**: Create [`TrackedEngine`](crate::TrackedEngine) via
///    [`tracked`](Engine::tracked) and execute queries
/// 6. **Updates**: Drop `TrackedEngine`, modify inputs, repeat from step 5
///
/// # Example
///
/// See the [crate-level documentation](crate) for a complete example of
/// creating and using an engine.
///
/// # Thread Safety
///
/// - **`&Engine`**: Safe to share across threads for read-only operations
/// - **`&mut Engine`**: Required for executor registration and input
///   modification
/// - **`Arc<Engine>`**: The standard way to share an engine for querying across
///   threads
///
/// The engine uses internal synchronization (locks, atomic operations) to
/// allow concurrent query execution from multiple threads.
///
/// # Persistence
///
/// Query results and metadata are automatically persisted to the database
/// backend. When you create a new `Engine` instance pointing to the same
/// database, previously computed results are reloaded, enabling:
///
/// - **Restart recovery**: Resume computation after program restart
/// - **Cross-process sharing**: Multiple processes can share computation state
///   (with appropriate database backend support)
///
/// # Memory Management
///
/// The engine maintains both in-memory and on-disk caches:
///
/// - **In-memory**: Fast access with cache eviction (size controlled by
///   [`Config::cache_entry_capacity`])
/// - **On-disk**: Persistent storage in the database backend
///
/// Large result sets should use shared ownership (`Arc`, `Arc<[T]>`) to avoid
/// expensive cloning.
pub struct Engine<C: Config> {
    interner: Interner,
    computation_graph: ComputationGraph<C>,
    executor_registry: Registry<C>,
    #[allow(unused)]
    rayon_thread_pool: rayon::ThreadPool,
    build_stable_hasher: C::BuildStableHasher,
}

impl<C: Config> Engine<C> {
    /// Registers an executor for a specific query type.
    ///
    /// Each query type should have exactly one executor registered. The
    /// executor defines how to compute the value for queries of that type.
    /// If an executor is already registered for the type, it will be silently
    /// replaced.
    ///
    /// # Type Parameters
    ///
    /// - `Q`: The query type this executor handles (must implement [`Query`])
    /// - `E`: The executor implementation (must implement [`Executor<Q, C>`])
    ///
    /// # Arguments
    ///
    /// * `executor` - The executor instance, wrapped in `Arc` for shared
    ///   ownership
    ///
    /// # Panics
    ///
    /// The executor will panic during query execution if:
    /// - No executor is registered when a query of that type is executed
    /// - The executor violates purity requirements (non-deterministic behavior)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use std::sync::Arc;
    /// use qbice::{Engine, Executor, Query, TrackedEngine, CyclicError};
    ///
    /// struct AddQuery { a: u64, b: u64 }
    /// impl Query for AddQuery {
    ///     type Value = u64;
    /// }
    ///
    /// struct AddExecutor;
    /// impl<C: qbice::Config> Executor<AddQuery, C> for AddExecutor {
    ///     async fn execute(
    ///         &self,
    ///         query: &AddQuery,
    ///         _: &TrackedEngine<C>,
    ///     ) -> Result<u64, CyclicError> {
    ///         Ok(query.a + query.b)
    ///     }
    /// }
    ///
    /// // Register the executor
    /// engine.register_executor::<AddQuery, _>(Arc::new(AddExecutor));
    /// ```
    ///
    /// # Notes
    ///
    /// - Executors must be registered **before** queries of that type are
    ///   executed
    /// - Executors must be re-registered each time a new `Engine` instance is
    ///   created, even when reusing the same database
    /// - The executor is stored behind `Arc`, so it can be shared with multiple
    ///   engines or cloned cheaply
    pub fn register_executor<Q: Query, E: Executor<Q, C>>(
        &mut self,
        executor: Arc<E>,
    ) {
        self.executor_registry.register(executor);
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
        self.interner.intern(value)
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
        self.interner.intern_unsized(value)
    }
}

fn default_shard_amount() -> usize {
    static SHARD_AMOUNT: OnceLock<usize> = OnceLock::new();
    *SHARD_AMOUNT.get_or_init(|| {
        (std::thread::available_parallelism().map_or(32, usize::from) * 32)
            .next_power_of_two()
    })
}

impl<C: Config> Engine<C> {
    /// Creates a new engine instance with the specified configuration.
    ///
    /// This is the primary constructor for creating an engine. It initializes
    /// all internal components including the computation graph, database
    /// connection, and thread pools.
    ///
    /// # Arguments
    ///
    /// * `serialization_plugin` - A [`Plugin`] for serializing and
    ///   deserializing query keys and values. Use [`Plugin::default()`] for
    ///   standard types.
    /// * `database_factory` - A factory that creates the database backend.
    ///   Common choice: `RocksDB::factory(path)`
    /// * `stable_hasher` - A hasher builder for computing stable query IDs. Use
    ///   `SeededStableHasherBuilder::new(seed)` with a fixed seed.
    ///
    /// # Returns
    ///
    /// Returns `Ok(Engine)` on success, or `Err` if the database cannot be
    /// opened.
    ///
    /// # Errors
    ///
    /// Returns the database factory's error type if:
    /// - The database path is invalid or inaccessible
    /// - The database is corrupted
    /// - Insufficient permissions to open the database
    /// - The database is already locked by another process
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use qbice::{DefaultConfig, Engine, serialize::Plugin};
    /// use qbice::stable_hash::{SeededStableHasherBuilder, Sip128Hasher};
    /// use qbice::storage::kv_database::rocksdb::RocksDB;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let temp_dir = tempfile::tempdir()?;
    ///
    /// let engine = Engine::<DefaultConfig>::new_with(
    ///     Plugin::default(),                           // Serialization
    ///     RocksDB::factory(temp_dir.path()),          // Database
    ///     SeededStableHasherBuilder::<Sip128Hasher>::new(0),  // Hasher
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Seed Stability
    ///
    /// **Important**: The seed passed to the stable hasher must be the same
    /// across runs if you want to reuse cached results. Using different seeds
    /// will cause query IDs to differ, making cached results inaccessible.
    ///
    /// # Thread Pool Creation
    ///
    /// This method creates the Rayon thread pool specified by
    /// [`Config::rayon_thread_pool_builder()`]. If thread pool creation fails,
    /// this method panics.
    pub async fn new_with<
        F: StorageEngineFactory<StorageEngine = C::StorageEngine>,
    >(
        mut serialization_plugin: Plugin,
        database_factory: F,
        stable_hasher: C::BuildStableHasher,
    ) -> Result<Self, F::Error> {
        let shared_interner = Interner::new_with_vacuum(
            default_shard_amount(),
            stable_hasher.clone(),
            Duration::from_secs(2),
        );

        assert!(
            serialization_plugin.insert(shared_interner.clone()).is_none(),
            "should have no existing interning pluging installed"
        );

        let storage_engine = database_factory.open(serialization_plugin)?;

        Ok(Self {
            computation_graph: ComputationGraph::new(&storage_engine).await,
            interner: shared_interner,
            executor_registry: Registry::default(),
            build_stable_hasher: stable_hasher,
            rayon_thread_pool: C::rayon_thread_pool_builder()
                .build()
                .expect("failed to build rayon thread pool"),
        })
    }

    fn hash<V: StableHash>(&self, value: &V) -> Compact128 {
        let mut hasher = self.build_stable_hasher.build_stable_hasher();
        value.stable_hash(&mut hasher);
        hasher.finish().into()
    }
}

impl<C: Config> std::fmt::Debug for Engine<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine").finish_non_exhaustive()
    }
}

/// A new type wrapper over a u64 representing the initial seed for all
/// fingerprinting operations.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Default,
    StableHash,
)]
pub struct InitialSeed(u64);
