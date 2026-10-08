//! The main storage engine abstraction.
//!
//! This module provides the [`StorageEngine`] trait, which is where the write
//! manager that creates the maps and handles their writes comes from.

use qbice_serialize::Plugin;

use crate::write_manager::WriteManager;

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

/// A trait defining a complete storage engine.
///
/// A storage engine is where a [`WriteManager`] comes from. The write manager
/// is the entry point for everything else: it creates the maps
/// ([`SingleMap`](crate::single_map::SingleMap),
/// [`DynamicMap`](crate::dynamic_map::DynamicMap),
/// [`KeyOfSetMap`](crate::key_of_set_map::KeyOfSetMap)) and the write
/// transactions that group their writes.
///
/// # Associated Types
///
/// All map types created by the write manager share the same
/// `WriteTransaction` type, allowing coordinated atomic writes across
/// different map instances.
pub trait StorageEngine {
    /// The write batch type used to group operations across all map types.
    type WriteTransaction: Send + Sync;

    /// The write manager type for creating the maps and handling their write
    /// transactions.
    type WriteManager: WriteManager<WriteBatch = Self::WriteTransaction>
        + Send
        + Sync;

    /// Creates a new write manager for creating maps and handling write
    /// transactions.
    ///
    /// The maps of one write manager know nothing about the writes of
    /// another, so the maps that share a storage backend are meant to come
    /// from one write manager.
    ///
    /// # Returns
    ///
    /// A new write manager instance.
    fn new_write_manager(&self) -> Self::WriteManager;
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
