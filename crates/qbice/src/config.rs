//! Configuration module for customizing QBICE engine behavior.
//!
//! This module provides the [`Config`] trait for customizing engine parameters
//! and the [`DefaultConfig`] implementation for typical use cases.

use std::{
    fmt::Debug,
    hash::{BuildHasher, Hash},
};

use fxhash::FxBuildHasher;
use qbice_stable_hash::{
    BuildStableHasher, SeededStableHasherBuilder, Sip128Hasher,
};
use qbice_stable_type_id::Identifiable;
use qbice_storage::kv_database::{KvDatabase, rocksdb::RocksDB};

/// Configuration trait for QBICE engine, allowing customization of various
/// parameters.
pub trait Config:
    Identifiable
    + Default
    + Debug
    + Clone
    + Copy
    + PartialEq
    + Eq
    + PartialOrd
    + Ord
    + Hash
    + Send
    + Sync
    + 'static
{
    /// The size of static storage allocated for query keys and values.
    ///
    /// This determines how much data can be stored inline (on the stack)
    /// before falling back to heap allocation. Larger values reduce
    /// allocations for queries with bigger keys/values but increase
    /// memory overhead per query.
    ///
    /// Common choices:
    /// - `[u8; 16]` - Good for small keys (e.g., simple integers, small tuples)
    /// - `[u8; 32]` - Balanced for moderate-sized keys
    /// - `[u8; 64]` - For larger query keys with multiple fields
    type Storage: Send + Sync + 'static;

    /// The key-value database backend used by the engine.
    type Database: KvDatabase;

    /// The stable hasher builder used by the engine.
    type BuildStableHasher: BuildStableHasher<Hash = u128>
        + Clone
        + Send
        + Sync
        + 'static;

    /// The standard hasher builder used by the engine.
    type BuildHasher: BuildHasher + Default + Clone + Send + Sync + 'static;

    /// The number of query entries the engine's cache can hold before dropping
    /// them for memory cache.
    #[must_use]
    fn cache_entry_capacity() -> usize { 2usize.pow(16) }

    /// Creates a Rayon thread pool builder for parallel query execution.
    #[must_use]
    fn rayon_thread_pool_builder() -> rayon::ThreadPoolBuilder {
        rayon::ThreadPoolBuilder::new()
    }
}

/// The default configuration for QBICE engine, suitable for most use cases.
///
/// It uses [`RocksDB`] as the database backend and use [`Sip128Hasher`] for
/// stable hashing.
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
    Identifiable,
)]
pub struct DefaultConfig;

impl Config for DefaultConfig {
    type Storage = [u8; 16];

    type Database = RocksDB;

    type BuildStableHasher = SeededStableHasherBuilder<Sip128Hasher>;

    type BuildHasher = FxBuildHasher;
}
