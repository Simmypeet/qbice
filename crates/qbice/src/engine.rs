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
//! 3. **Input**: Set initial input values via `InputSession`
//! 4. **Wrap**: Convert to `Arc<Engine>` and create `TrackedEngine`
//! 5. **Query**: Execute queries through `TrackedEngine`
//! 6. **Update**: Drop `TrackedEngine`, modify inputs, repeat from step 4

use std::sync::{Arc, OnceLock};

use qbice_serialize::Plugin;
use qbice_stable_hash::{
    BuildStableHasher, Compact128, StableHash, StableHasher,
};
use qbice_storage::{intern::SharedInterner, kv_database::KvDatabaseFactory};

use crate::{
    config::{Config, DefaultConfig},
    engine::computation_graph::ComputationGraph,
    executor::{Executor, Registry},
    query::Query,
};

pub(super) mod computation_graph;

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
/// 2. Wrap in `Arc` and create a [`crate::TrackedEngine`] for querying
/// 3. Drop the `TrackedEngine` to release the Arc reference
/// 4. Use `Arc::get_mut` to modify inputs
/// 5. Repeat from step 2
///
/// # Thread Safety
///
/// - `&Engine`: Safe to share across threads (read-only access)
/// - `&mut Engine`: Required for registration and input modification
/// - `Arc<Engine>`: The typical way to share the engine for querying
pub struct Engine<C: Config> {
    interner: SharedInterner,
    computation_graph: ComputationGraph<C>,
    executor_registry: Registry<C>,
    rayon_thread_pool: rayon::ThreadPool,
    build_stable_hasher: C::BuildStableHasher,
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
    ) -> Result<Self, F::Error> {
        let shared_interner =
            SharedInterner::new(default_shard_amount(), stable_hasher.clone());

        assert!(
            serialization_plugin.insert(shared_interner.clone()).is_none(),
            "should have no existing interning pluging installed"
        );

        let database = Arc::new(database_factory.open(serialization_plugin)?);

        Ok(Self {
            computation_graph: ComputationGraph::new(
                database,
                default_shard_amount(),
            ),
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
