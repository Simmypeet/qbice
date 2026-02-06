//! The main storage engine abstraction.
//!
//! This module provides the [`StorageEngine`] trait that combines all map types
//! and write management into a unified storage interface.

use qbice_serialize::Plugin;

use crate::{
    dynamic_map::DynamicMap,
    key_of_set_map::{ConcurrentSet, KeyOfSetMap},
    kv_database::{KeyOfSetColumn, WideColumn, WideColumnValue},
    single_map::SingleMap,
    write_manager::WriteManager,
};

/// Database-backed storage engine implementation.
///
/// This module provides [`DbBacked`](db_backed::DbBacked), a storage engine
/// implementation that uses a key-value database backend with caching and
/// write-behind support.
pub mod db_backed;

/// In-memory storage engine implementation.
///
/// This module provides
/// [`InMemoryStorageEngine`](in_memory::InMemoryStorageEngine),
/// a storage engine implementation that stores all data in memory without
/// persistence.
pub mod in_memory;

/// A trait defining a complete storage engine with support for multiple map
/// types.
///
/// This trait combines all storage abstractions ([`SingleMap`], [`DynamicMap`],
/// [`KeyOfSetMap`]) with write management into a unified interface. It serves
/// as the main entry point for creating storage components.
///
/// # Associated Types
///
/// All map types created by this engine share the same `WriteTransaction` type,
/// allowing coordinated atomic writes across different map instances.
pub trait StorageEngine {
    /// The write batch type used to group operations across all map types.
    type WriteTransaction: Send + Sync;

    /// The write manager type for handling write transactions.
    type WriteManager: WriteManager<WriteBatch = Self::WriteTransaction>
        + Send
        + Sync;

    /// The single map type created by this engine.
    type SingleMap<K: WideColumn, V: WideColumnValue<K>>: SingleMap<K, V, WriteTransaction = Self::WriteTransaction>
        + Send
        + Sync;

    /// The dynamic map type created by this engine.
    type DynamicMap<K: WideColumn>: DynamicMap<K, WriteTransaction = Self::WriteTransaction>
        + Send
        + Sync;

    /// The key-of-set map type created by this engine.
    type KeyOfSetMap<K: KeyOfSetColumn, C: ConcurrentSet<Element = K::Element>>: KeyOfSetMap<K, C, WriteBatch = Self::WriteTransaction> + Send + Sync;

    /// Creates a new write manager for handling write transactions.
    ///
    /// # Returns
    ///
    /// A new write manager instance.
    fn new_write_manager(&self) -> Self::WriteManager;

    /// Creates a new single map for storing key-value pairs with a fixed value
    /// type.
    ///
    /// # Type Parameters
    ///
    /// - `K`: The wide column type that defines the key type.
    /// - `V`: The value type to store.
    ///
    /// # Returns
    ///
    /// A new single map instance.
    fn new_single_map<K: WideColumn, V: WideColumnValue<K>>(
        &self,
    ) -> Self::SingleMap<K, V>;

    /// Creates a new dynamic map for storing key-value pairs with dynamic
    /// value types.
    ///
    /// # Type Parameters
    ///
    /// - `K`: The wide column type that defines the key type and discriminant.
    ///
    /// # Returns
    ///
    /// A new dynamic map instance.
    fn new_dynamic_map<K: WideColumn>(&self) -> Self::DynamicMap<K>;

    /// Creates a new key-of-set map for storing key-to-set relationships.
    ///
    /// # Type Parameters
    ///
    /// - `K`: The key-of-set column type.
    /// - `C`: The concurrent set type for storing elements.
    ///
    /// # Returns
    ///
    /// A new key-of-set map instance.
    fn new_key_of_set_map<
        K: KeyOfSetColumn,
        C: ConcurrentSet<Element = K::Element>,
    >(
        &self,
    ) -> Self::KeyOfSetMap<K, C>;
}

/// A factory trait for creating instances of a storage engine.
pub trait StorageEngineFactory {
    /// The storage engine type created by this factory.
    type StorageEngine;

    /// The error type returned if opening the storage engine fails.
    type Error;

    /// Opens a new storage engine instance with the specified serialization
    /// plugin.
    fn open(
        self,
        serialization_plugin: Plugin,
    ) -> Result<Self::StorageEngine, Self::Error>;
}
