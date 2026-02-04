//! In-memory implementation of the storage engine.
//!
//! This module provides `InMemoryStorageEngine`, a storage engine that
//! keeps all data in memory without persistence. Useful for testing,
//! development, or scenarios where data persistence is not required.

use std::convert::Infallible;

use crate::{
    dynamic_map::in_memory::InMemoryDynamicMap,
    key_of_set_map::{ConcurrentSet, in_memory::InMemoryKeyOfSetMap},
    kv_database::{KeyOfSetColumn, WideColumn, WideColumnValue},
    single_map::in_memory::InMemorySingleMap,
    storage_engine::{StorageEngine, StorageEngineFactory},
    write_manager, write_transaction,
};

/// An in-memory storage engine implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct InMemoryStorageEngine;

impl StorageEngine for InMemoryStorageEngine {
    type WriteTransaction = write_transaction::FauxWriteTransaction;

    type WriteManager = write_manager::FauxWriteManager;

    type SingleMap<K: WideColumn, V: WideColumnValue<K>> =
        InMemorySingleMap<K, V>;

    type DynamicMap<K: WideColumn> = InMemoryDynamicMap<K>;

    type KeyOfSetMap<
        K: KeyOfSetColumn,
        C: ConcurrentSet<Element = K::Element>,
    > = InMemoryKeyOfSetMap<K, C>;

    fn new_write_manager(&self) -> Self::WriteManager {
        write_manager::FauxWriteManager
    }

    fn new_single_map<K: WideColumn, V: WideColumnValue<K>>(
        &self,
    ) -> Self::SingleMap<K, V> {
        InMemorySingleMap::new()
    }

    fn new_dynamic_map<K: WideColumn>(&self) -> Self::DynamicMap<K> {
        InMemoryDynamicMap::new()
    }

    fn new_key_of_set_map<
        K: KeyOfSetColumn,
        C: ConcurrentSet<Element = K::Element>,
    >(
        &self,
    ) -> Self::KeyOfSetMap<K, C> {
        InMemoryKeyOfSetMap::new()
    }
}

/// Factory for creating in-memory storage engine instances.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct InMemoryStorageEngineFactory;

impl StorageEngineFactory for InMemoryStorageEngineFactory {
    type StorageEngine = InMemoryStorageEngine;

    type Error = Infallible;

    fn open(
        self,
        _serialization_plugin: qbice_serialize::Plugin,
    ) -> Result<Self::StorageEngine, Self::Error> {
        Ok(InMemoryStorageEngine)
    }
}
