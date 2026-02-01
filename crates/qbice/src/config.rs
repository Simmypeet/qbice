//! Configuration module for customizing QBICE engine behavior.
//!
//! This module provides the [`Config`] trait for customizing various aspects
//! of the QBICE engine, including storage, database backend, hashing, and
//! caching parameters. The [`DefaultConfig`] implementation provides sensible
//! defaults suitable for most use cases.
//!
//! # Overview
//!
//! The `Config` trait allows you to customize:
//!
//! - **Storage allocation**: Size of inline storage for query keys and values
//! - **Database backend**: Choice of key-value database (e.g., `RocksDB`)
//! - **Hashing**: Stable and standard hasher implementations
//! - **Cache sizing**: Memory limits for query result caching
//! - **Concurrency**: Background thread counts and parallelism settings
//!
//! # Creating Custom Configurations
//!
//! To create a custom configuration, implement the `Config` trait:
//!
//! ```rust
//! use fxhash::FxBuildHasher;
//! use qbice::{
//!     Config,
//!     stable_hash::{SeededStableHasherBuilder, Sip128Hasher},
//!     storage::kv_database::rocksdb::RocksDB,
//! };
//!
//! #[derive(
//!     Debug,
//!     Clone,
//!     Copy,
//!     PartialEq,
//!     Eq,
//!     PartialOrd,
//!     Ord,
//!     Hash,
//!     Default,
//!     qbice::Identifiable,
//! )]
//! struct MyConfig;
//!
//! impl Config for MyConfig {
//!     // Use larger inline storage for bigger query keys
//!     type Database = RocksDB;
//!     type BuildStableHasher = SeededStableHasherBuilder<Sip128Hasher>;
//!     type BuildHasher = FxBuildHasher;
//!
//!     // Increase cache capacity for larger workloads
//!     fn cache_entry_capacity() -> usize {
//!         2usize.pow(20) // 1M entries
//!     }
//!
//!     // Use more background writers for write-heavy workloads
//!     fn background_writer_thread_count() -> usize { 4 }
//! }
//! ```
//!
//! # Using `DefaultConfig`
//!
//! For most applications, [`DefaultConfig`] provides a good starting point:
//!
//! ```rust,ignore
//! use qbice::{DefaultConfig, Engine, serialize::Plugin};
//! use qbice::stable_hash::{SeededStableHasherBuilder, Sip128Hasher};
//! use qbice::storage::kv_database::rocksdb::RocksDB;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let temp_dir = tempfile::tempdir()?;
//! let engine = Engine::<DefaultConfig>::new_with(
//!     Plugin::default(),
//!     RocksDB::factory(temp_dir.path()),
//!     SeededStableHasherBuilder::<Sip128Hasher>::new(0),
//! )?;
//! # Ok(())
//! # }
//! ```

use std::{
    fmt::Debug,
    hash::{BuildHasher, Hash},
};

use qbice_stable_hash::BuildStableHasher;
use qbice_stable_type_id::Identifiable;
use qbice_storage::storage_engine::StorageEngine;

/// Configuration trait for the QBICE engine.
///
/// This trait defines all configurable aspects of the engine's behavior,
/// including storage, database backend, hashing algorithms, and performance
/// tuning parameters. Implement this trait to customize the engine for your
/// specific use case.
///
/// # Required Associated Types
///
/// ## `Storage`
///
/// The inline storage buffer size for query keys and values. This determines
/// how much data can be stored on the stack before heap allocation is needed.
/// Larger values reduce allocations but increase per-query memory overhead.
///
/// **Common choices:**
/// - `[u8; 16]` - Small keys (integers, small tuples)
/// - `[u8; 32]` - Medium keys (default, good balance)
/// - `[u8; 64]` - Large keys (structs with many fields)
///
/// ## `Database`
///
/// The key-value database backend for persistent storage. The engine uses
/// this to cache query results across runs.
///
/// **Available backends:**
/// - [`RocksDB`] - Default, high-performance embedded database
///
/// ## `BuildStableHasher`
///
/// The hasher builder for stable hashing across program runs. Used to
/// compute query IDs and fingerprints.
///
/// **Requirements:**
/// - Must produce 128-bit hashes
/// - Must be deterministic across program runs
///
/// ## `BuildHasher`
///
/// The standard hasher for internal hash maps and sets. Does not need to
/// be stable across runs.
///
/// # Configurable Methods
///
/// The trait provides several methods with default implementations that can
/// be overridden:
///
/// - [`cache_entry_capacity()`](Config::cache_entry_capacity) - In-memory cache
///   size
/// - [`background_writer_thread_count()`](Config::background_writer_thread_count)
///   \- Database write parallelism
/// - [`rayon_thread_pool_builder()`](Config::rayon_thread_pool_builder) - Query
///   execution parallelism
///
/// # Thread Safety
///
/// All Config implementations must be `Send + Sync` since the engine may be
/// accessed from multiple threads.
///
/// # Example
///
/// See the [module-level documentation](crate::config) for a complete example
/// of implementing a custom configuration.
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
    /// The key-value database backend used by the engine.
    type StorageEngine: StorageEngine;

    /// The stable hasher builder used by the engine.
    type BuildStableHasher: BuildStableHasher<Hash = u128>
        + Clone
        + Send
        + Sync
        + 'static;

    /// The standard hasher builder used by the engine.
    type BuildHasher: BuildHasher + Default + Clone + Send + Sync + 'static;

    /// The maximum number of query entries to keep in the in-memory cache.
    ///
    /// This controls the size of the cache that stores recently computed
    /// query results in memory. When the cache exceeds this capacity, the
    /// entries are evicted.
    ///
    /// # Default Value
    ///
    /// The default is 262,144 (2^18) entries, which provides a good balance
    /// between memory usage and cache hit rate for typical workloads.
    #[must_use]
    fn cache_entry_capacity() -> usize { 2usize.pow(18) }

    /// The number of background threads for writing data to the database.
    ///
    /// These threads handle asynchronous persistence of query results and
    /// metadata to the database backend. Increasing this count can improve
    /// write throughput for write-heavy workloads.
    ///
    /// # Default Value
    ///
    /// The default is 2 threads, which provides adequate performance for most
    /// use cases without excessive thread overhead.
    ///
    /// # Note
    ///
    /// The effectiveness of additional threads depends on your database
    /// backend's write concurrency capabilities. Some backends may not benefit
    /// from many writers.
    #[must_use]
    fn background_writer_thread_count() -> usize { 2 }

    /// Creates a Rayon thread pool builder for parallel query execution.
    ///
    /// QBICE uses Rayon for parallelizing certain operations, such as dirty
    /// propagation during input updates. This method allows you to customize
    /// the thread pool configuration.
    ///
    /// # Default Configuration
    ///
    /// The default uses Rayon's default thread pool configuration, which
    /// typically creates one thread per logical CPU core.
    ///
    /// # Customization Examples
    ///
    /// ## Limit Thread Count
    ///
    /// ```rust
    /// use qbice::Config;
    ///
    /// # #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, qbice::Identifiable)]
    /// # struct MyConfig;
    /// # impl Config for MyConfig {
    /// #     type Database = qbice::storage::kv_database::rocksdb::RocksDB;
    /// #     type BuildStableHasher = qbice::stable_hash::SeededStableHasherBuilder<qbice::stable_hash::Sip128Hasher>;
    /// #     type BuildHasher = fxhash::FxBuildHasher;
    /// fn rayon_thread_pool_builder() -> rayon::ThreadPoolBuilder {
    ///     rayon::ThreadPoolBuilder::new().num_threads(4)
    /// }
    /// # }
    /// ```
    ///
    /// ## Set Thread Names
    ///
    /// ```rust
    /// use qbice::Config;
    ///
    /// # #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, qbice::Identifiable)]
    /// # struct MyConfig;
    /// # impl Config for MyConfig {
    /// #     type Database = qbice::storage::kv_database::rocksdb::RocksDB;
    /// #     type BuildStableHasher = qbice::stable_hash::SeededStableHasherBuilder<qbice::stable_hash::Sip128Hasher>;
    /// #     type BuildHasher = fxhash::FxBuildHasher;
    /// fn rayon_thread_pool_builder() -> rayon::ThreadPoolBuilder {
    ///     rayon::ThreadPoolBuilder::new()
    ///         .thread_name(|i| format!("qbice-worker-{}", i))
    /// }
    /// # }
    /// ```
    #[must_use]
    fn rayon_thread_pool_builder() -> rayon::ThreadPoolBuilder {
        rayon::ThreadPoolBuilder::new()
    }
}

// The default configuration for the QBICE engine.
//
// This configuration provides sensible defaults suitable for most
// applications. It uses:
//
// - **Storage**: 16-byte inline storage (suitable for small to medium keys)
// - **Database**: `RocksDB` for persistent storage
// - **Stable Hasher**: SipHash-128 for deterministic query identification
// - **Standard Hasher**: `FxHash` for fast internal hash maps
// - **Cache Capacity**: 262,144 entries (2^18)
// - **Writer Threads**: 2 background writers
//
// # When to Use
//
// `DefaultConfig` is appropriate for:
// - Quick prototyping and development
// - Applications with typical query sizes and workload patterns
// - Projects that don't require specialized performance tuning
//
// # When to Customize
//
// Consider implementing a custom [`Config`] if you need:
// - Different inline storage sizes for larger/smaller query keys
// - Alternative database backends
// - Tuned cache capacities for your workload
// - Custom thread pool configurations
//
// # Example
//
// ```rust,ignore
// use std::sync::Arc;
// use qbice::{DefaultConfig, Engine, serialize::Plugin};
// use qbice::stable_hash::{SeededStableHasherBuilder, Sip128Hasher};
// use qbice::storage::kv_database::rocksdb::RocksDB;
//
// # fn main() -> Result<(), Box<dyn std::error::Error>> {
// let temp_dir = tempfile::tempdir()?;
//
// // Create an engine with default configuration
// let engine = Engine::<DefaultConfig>::new_with(
//     Plugin::default(),
//     RocksDB::factory(temp_dir.path()),
//     SeededStableHasherBuilder::<Sip128Hasher>::new(0),
// )?;
//
// // Ready to use!
// # Ok(())
// # }
// ```

// #[derive(
//     Debug,
//     Clone,
//     Copy,
//     PartialEq,
//     Eq,
//     PartialOrd,
//     Ord,
//     Hash,
//     Default,
//     Identifiable,
// )]
// #[cfg(feature = "default_config")]
// pub struct DefaultConfig;

// #[cfg(feature = "default_config")]
// impl Config for DefaultConfig {
//     type Database = RocksDB;

//     type BuildStableHasher = SeededStableHasherBuilder<Sip128Hasher>;

//     type BuildHasher = FxBuildHasher;
// }

/// Type alias for a single map in the storage engine.
pub type SingleMap<C, K, V> =
    <<C as Config>::StorageEngine as StorageEngine>::SingleMap<K, V>;

/// Type alias for a dynamic map in the storage engine.
pub type DynamicMap<C, K> =
    <<C as Config>::StorageEngine as StorageEngine>::DynamicMap<K>;

/// Type alias for a key-of-set map in the storage engine.
pub type KeyOfSetMap<C, K, Con> =
    <<C as Config>::StorageEngine as StorageEngine>::KeyOfSetMap<K, Con>;

/// Type alias for a write transaction in the storage engine.
pub type WriteTransaction<C> =
    <<C as Config>::StorageEngine as StorageEngine>::WriteTransaction;
