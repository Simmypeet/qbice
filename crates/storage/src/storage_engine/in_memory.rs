//! In-memory implementation of the storage engine.
//!
//! This module provides `InMemoryStorageEngine`, a storage engine that
//! keeps all data in memory without persistence. Useful for testing,
//! development, or scenarios where data persistence is not required.

use std::convert::Infallible;

use crate::{
    storage_engine::{StorageEngine, StorageEngineFactory},
    write_batch, write_manager,
};

/// An in-memory storage engine implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct InMemoryStorageEngine;

impl StorageEngine for InMemoryStorageEngine {
    type WriteTransaction = write_batch::FauxWriteBatch;

    type WriteManager = write_manager::FauxWriteManager;

    fn new_write_manager(&self) -> Self::WriteManager {
        write_manager::FauxWriteManager
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
