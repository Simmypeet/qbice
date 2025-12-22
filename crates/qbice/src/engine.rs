//! Core engine types for query execution and storage.
//!
//! This module provides the central database engine ([`Engine`]) and the
//! tracked engine wrapper ([`TrackedEngine`]) for executing queries.
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
//! 3. **Input**: Set initial input values via `InputSession`
//! 4. **Wrap**: Convert to `Arc<Engine>` and create `TrackedEngine`
//! 5. **Query**: Execute queries through `TrackedEngine`
//! 6. **Update**: Drop `TrackedEngine`, modify inputs, repeat from step 4
//!
//! # Example
//!
//! ```rust
//! use std::sync::Arc;
//!
//! use qbice::{
//!     Identifiable, StableHash,
//!     config::DefaultConfig,
//!     engine::{Engine, TrackedEngine},
//!     executor::{CyclicError, Executor},
//!     query::Query,
//! };
//!
//! // Define queries
//! #[derive(Debug, Clone, PartialEq, Eq, Hash, StableHash, Identifiable)]
//! struct Input(u64);
//! impl Query for Input {
//!     type Value = i64;
//! }
//!
//! #[derive(Debug, Clone, PartialEq, Eq, Hash, StableHash, Identifiable)]
//! struct Squared(u64);
//! impl Query for Squared {
//!     type Value = i64;
//! }
//!
//! // Define executor
//! struct SquaredExecutor;
//! impl<C: qbice::config::Config> Executor<Squared, C> for SquaredExecutor {
//!     async fn execute(
//!         &self,
//!         query: &Squared,
//!         engine: &TrackedEngine<C>,
//!     ) -> Result<i64, CyclicError> {
//!         let value = engine.query(&Input(query.0)).await?;
//!         Ok(value * value)
//!     }
//! }
//!
//! #[tokio::main]
//! async fn main() {
//!     // 1. Create engine
//!     let mut engine = Engine::<DefaultConfig>::new();
//!
//!     // 2. Register executors
//!     engine.register_executor::<Squared, _>(Arc::new(SquaredExecutor));
//!
//!     // 3. Set inputs
//!     {
//!         let mut session = engine.input_session();
//!         session.set_input(Input(0), 5);
//!     }
//!
//!     // 4. Wrap in Arc
//!     let engine = Arc::new(engine);
//!
//!     // 5. Query
//!     let tracked = engine.tracked();
//!     assert_eq!(tracked.query(&Squared(0)).await, Ok(25));
//! }
//! ```
//!
//! # Visualization
//!
//! The engine can generate interactive HTML visualizations of the dependency
//! graph. See [`Engine::visualize_html`] and [`write_html_visualization`].

use std::sync::{Arc, OnceLock};

use dashmap::DashMap;
use qbice_serialize::Plugin;
use qbice_stable_hash::StableHash;
use qbice_storage::{intern::SharedInterner, kv_database::KvDatabaseFactory};

use crate::{
    config::{Config, DefaultConfig},
    engine::computation_graph::ComputationGraph,
    executor::{Executor, Registry},
    query::{DynValueBox, Query, QueryID},
};

mod computation_graph;
// mod database;
// mod fingerprint;
// mod visualization;

// pub(super) mod meta;

// pub use visualization::{
//     EdgeInfo, GraphSnapshot, NodeInfo, write_html_visualization,
// };

/// The central query database engine.
///
/// The `Engine` is the core component of QBICE, responsible for:
/// - Storing cached query results
/// - Tracking dependencies between queries
/// - Managing dirty propagation when inputs change
/// - Coordinating query execution through registered executors
///
/// # Usage Pattern
///
/// The engine is typically used through an ownership cycle:
///
/// 1. Create and configure the engine (register executors, set inputs)
/// 2. Wrap in `Arc` and create a [`TrackedEngine`] for querying
/// 3. Drop the `TrackedEngine` to release the Arc reference
/// 4. Use `Arc::get_mut` to modify inputs
/// 5. Repeat from step 2
///
/// # Thread Safety
///
/// - `&Engine`: Safe to share across threads (read-only access)
/// - `&mut Engine`: Required for registration and input modification
/// - `Arc<Engine>`: The typical way to share the engine for querying
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
///         Ok(q.0 as i64 * 2)
///     }
/// }
///
/// # #[tokio::main]
/// # async fn main() {
/// // Create and configure
/// let mut engine = Engine::<DefaultConfig>::new();
/// engine.register_executor::<MyQuery, _>(Arc::new(MyExecutor));
///
/// // Set inputs
/// {
///     let mut session = engine.input_session();
///     session.set_input(MyQuery(1), 100);
/// }
///
/// // Query
/// let engine = Arc::new(engine);
/// let tracked = engine.tracked();
/// let result = tracked.query(&MyQuery(0)).await;
/// # }
/// ```
pub struct Engine<C: Config> {
    database: Arc<C::Database>,
    interner: SharedInterner,
    computation_graph: ComputationGraph<C>,
    executor_registry: Registry<C>,
}

impl<C: Config> Engine<C> {
    /// Registers an executor for the given query type.
    ///
    /// Each query type should have exactly one executor registered. If an
    /// executor is already registered for the type, it will be replaced.
    ///
    /// # Type Parameters
    ///
    /// - `Q`: The query type this executor handles
    /// - `E`: The executor implementation type
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
    /// struct Add {
    ///     a: i64,
    ///     b: i64,
    /// }
    /// impl Query for Add {
    ///     type Value = i64;
    /// }
    ///
    /// struct AddExecutor;
    /// impl<C: qbice::config::Config> Executor<Add, C> for AddExecutor {
    ///     async fn execute(
    ///         &self,
    ///         q: &Add,
    ///         _: &TrackedEngine<C>,
    ///     ) -> Result<i64, CyclicError> {
    ///         Ok(q.a + q.b)
    ///     }
    /// }
    ///
    /// let mut engine = Engine::<DefaultConfig>::new();
    /// engine.register_executor::<Add, _>(Arc::new(AddExecutor));
    /// ```
    pub fn register_executor<Q: Query, E: Executor<Q, C>>(
        &mut self,
        executor: Arc<E>,
    ) {
        self.executor_registry.register(executor);
    }
}

static_assertions::assert_impl_all!(&Engine<DefaultConfig>: Send, Sync);

fn default_shard_amount() -> usize {
    static SHARD_AMOUNT: OnceLock<usize> = OnceLock::new();
    *SHARD_AMOUNT.get_or_init(|| {
        (std::thread::available_parallelism().map_or(1, usize::from) * 4)
            .next_power_of_two()
    })
}

impl<C: Config> Engine<C> {
    /// Creates a new engine instance.
    ///
    /// # Arguments
    ///
    /// * `serialization_plugin` - A plugin for serializing and deserializing
    ///   data.
    /// * `database_factory` - A factory for creating a database.
    /// * `stable_hasher` - A stable hasher for generating stable hashes.
    ///
    /// # Returns
    ///
    /// A new `Engine` instance.
    pub fn new_with<F: KvDatabaseFactory<KvDatabase = C::Database>>(
        mut serialization_plugin: Plugin,
        database_factory: F,
        stable_hasher: C::BuildStableHasher,
        hasher: C::BuildHasher,
    ) -> Result<Self, F::Error> {
        let shared_interner =
            SharedInterner::new(default_shard_amount(), stable_hasher);

        assert!(
            serialization_plugin.insert(shared_interner.clone()).is_none(),
            "should have no existing interning pluging installed"
        );

        let database = Arc::new(database_factory.open(serialization_plugin)?);

        Ok(Self {
            computation_graph: ComputationGraph::new(
                database.clone(),
                default_shard_amount(),
                hasher,
            ),
            database,
            interner: shared_interner,
            executor_registry: Registry::default(),
        })
    }

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
            // caller: CallerInformation::User,
        }
    }
}

impl<C: Config> std::fmt::Debug for Engine<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine").finish_non_exhaustive()
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
/// # }
/// ```
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
    // caller: CallerInformation,
}

impl<C: Config> Clone for TrackedEngine<C> {
    fn clone(&self) -> Self {
        Self {
            engine: Arc::clone(&self.engine),
            cache: Arc::clone(&self.cache),
            // caller: self.caller,
        }
    }
}

static_assertions::assert_impl_all!(&TrackedEngine<DefaultConfig>: Send, Sync);

impl<C: Config> std::fmt::Debug for TrackedEngine<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrackedEngine")
            .field("engine", &self.engine)
            // .field("caller", &self.caller)
            .finish_non_exhaustive()
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
