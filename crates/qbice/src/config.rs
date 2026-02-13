//! Configuration module for customizing QBICE engine behavior.
//!
//! This module provides the [`Config`] trait for customizing various aspects
//! of the QBICE engine, including storage backend, hashing, and parameters.
//!
//! # Overview
//!
//! The `Config` trait allows you to customize:
//!
//! - **Storage Engine**: Choice of key-value database backend (e.g., `RocksDB`
//!   with `DbBacked`) or in-memory storage
//! - **Hashing**: Stable hasher for query identification and content hashing
//! - **Hash Functions**: Standard hasher for internal collections

use std::{
    fmt::Debug,
    hash::{BuildHasher, Hash},
};

use qbice_stable_hash::BuildStableHasher;
use qbice_stable_type_id::Identifiable;
use qbice_storage::storage_engine::StorageEngine;

/// Configuration trait for customizing engine behavior.
///
/// The `Config` trait defines the key type-level customization points for a
/// QBICE engine. It specifies which storage engine, hashers, and other
/// components the engine should use.
///
/// # Associated Types
///
/// - [`StorageEngine`](Self::StorageEngine): The key-value database backend
/// - [`BuildStableHasher`](Self::BuildStableHasher): Hasher for stable query
///   identification
/// - [`BuildHasher`](Self::BuildHasher): Standard hasher for internal use
///
/// # Default Implementation
///
/// Use [`DefaultConfig`] for most applications, which comes configured with:
/// - RocksDB wrapped in `DbBacked` for storage
/// - `Sip128Hasher` for stable hashing
/// - `FxBuildHasher` for fast hashing
///
/// # Custom Configurations
///
/// Create a custom config struct and implement this trait if you need different
/// storage backends or hashing strategies. See the module documentation for an
/// example.
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
    /// The storage engine implementation.
    ///
    /// This should typically be `DbBacked<T>` where T is your database choice
    /// (e.g., `RocksDB`, `Fjall`). The `DbBacked` wrapper provides the
    /// necessary abstractions for QBICE's storage needs.
    type StorageEngine: StorageEngine;

    /// The hasher builder for stable hashing.
    ///
    /// Used for computing stable, deterministic hashes of query keys across
    /// program invocations. This is critical for correctly identifying queries
    /// and tracking cache validity.
    ///
    /// Common choices:
    /// - `SeededStableHasherBuilder<Sip128Hasher>`: Cryptographically strong,
    ///   recommended
    /// - Must use the same seed across engine instances to reuse cached results
    type BuildStableHasher: BuildStableHasher<Hash = u128>
        + Clone
        + Send
        + Sync
        + 'static;

    /// The hasher builder for standard hashing.
    ///
    /// Used for internal hash collections. Does not need to be stable across
    /// runs. Faster variants like `FxBuildHasher` are preferred here.
    type BuildHasher: BuildHasher + Default + Clone + Send + Sync + 'static;
}
#[cfg(feature = "default-config")]
mod default_config {
    use qbice_stable_type_id::Identifiable;
    use qbice_storage::{
        kv_database::rocksdb::RocksDB, storage_engine::db_backed::DbBacked,
    };

    use crate::Config;

    /// A sensible default configuration suitable for most applications.
    ///
    /// `DefaultConfig` uses:
    /// - RocksDB (`DbBacked<RocksDB>`) for persistent storage
    /// - `Sip128Hasher` with a seeded stable hasher builder
    /// - `FxBuildHasher` for internal collections
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
        type StorageEngine = DbBacked<RocksDB>;
        type BuildStableHasher = qbice_stable_hash::SeededStableHasherBuilder<
            qbice_stable_hash::Sip128Hasher,
        >;
        type BuildHasher = fxhash::FxBuildHasher;
    }
}

#[cfg(feature = "default-config")]
pub use default_config::DefaultConfig;

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
