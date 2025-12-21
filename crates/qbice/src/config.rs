//! Configuration module for customizing QBICE engine behavior.
//!
//! This module provides the [`Config`] trait for customizing engine parameters
//! and the [`DefaultConfig`] implementation for typical use cases.
//!
//! # Custom Configuration
//!
//! You can create custom configurations to tune memory allocation behavior:
//!
//! ```rust
//! use std::fmt::Debug;
//!
//! use qbice::config::Config;
//!
//! /// A custom configuration with larger inline storage.
//! #[derive(Debug, Default)]
//! struct LargeStorageConfig;
//!
//! impl Config for LargeStorageConfig {
//!     // 64 bytes of inline storage for query keys and values
//!     type Storage = [u8; 64];
//! }
//! ```
//!
//! The `Storage` type determines how much data can be stored inline before
//! heap allocation is required. Larger storage can reduce allocations for
//! queries with bigger keys or values, but increases memory usage per query.

use std::fmt::Debug;

use qbice_stable_hash::{
    BuildStableHasher, SeededStableHasherBuilder, Sip128Hasher,
};
use qbice_storage::kv_database::{KvDatabase, rocksdb::RocksDB};

/// Configuration trait for QBICE engine, allowing customization of various
/// parameters.
///
/// Implement this trait to customize the engine's behavior. The main
/// configuration point is the `Storage` associated type, which controls
/// inline storage size for query keys and values.
///
/// # Example
///
/// ```rust
/// use std::fmt::Debug;
///
/// use qbice::config::Config;
///
/// #[derive(Debug, Default)]
/// struct MyConfig;
///
/// impl Config for MyConfig {
///     type Storage = [u8; 32]; // 32 bytes of inline storage
/// }
/// ```
pub trait Config: Default + Debug + Send + Sync + 'static {
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
        + Send
        + Sync
        + 'static;
}

/// The default configuration for QBICE.
///
/// Uses 16 bytes of inline storage, which is suitable for most common
/// query types like integers, small structs, or tuple-based keys.
///
/// # Example
///
/// ```rust
/// use qbice::{config::DefaultConfig, engine::Engine};
///
/// // Create an engine with default configuration
/// let engine = Engine::<DefaultConfig>::new();
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct DefaultConfig;

impl Config for DefaultConfig {
    type Storage = [u8; 16];

    type Database = RocksDB;

    type BuildStableHasher = SeededStableHasherBuilder<Sip128Hasher>;
}
