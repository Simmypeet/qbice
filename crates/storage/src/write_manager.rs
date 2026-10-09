//! Write transaction management.
//!
//! This module provides the [`WriteManager`] trait for managing write
//! transactions and ensuring atomicity of write operations.

use crate::{
    dynamic_map::{DynamicMap, in_memory::InMemoryDynamicMap},
    key_of_set_map::{
        ConcurrentSet, KeyOfSetMap, in_memory::InMemoryKeyOfSetMap,
    },
    kv_database::{KeyOfSetColumn, WideColumn, WideColumnValue},
    single_map::{SingleMap, in_memory::InMemorySingleMap},
    write_batch::FauxWriteBatch,
};

pub mod write_behind;

/// A trait for managing write transactions.
///
/// The write manager is responsible for creating new write transactions and
/// submitting them to be applied to the storage backend. This provides a
/// layer of abstraction for coordinating writes across multiple storage
/// components.
///
/// The maps are created by the write manager too. A map writes through the
/// write transactions of the manager that created it, and a map that keeps
/// what it is given in memory until the storage backend has it needs to know
/// how far its manager has got with applying those transactions.
pub trait WriteManager {
    /// The type representing a write transaction.
    ///
    /// A write transaction collects write batches from various storage
    /// components and applies them atomically when submitted.
    type WriteBatch;

    /// The single map type created by this write manager.
    type SingleMap<K: WideColumn, V: WideColumnValue<K>>: SingleMap<K, V, WriteTransaction = Self::WriteBatch>
        + Send
        + Sync;

    /// The dynamic map type created by this write manager.
    type DynamicMap<K: WideColumn>: DynamicMap<K, WriteTransaction = Self::WriteBatch>
        + Send
        + Sync;

    /// The key-of-set map type created by this write manager.
    type KeyOfSetMap<K: KeyOfSetColumn, C: ConcurrentSet<Element = K::Element>>: KeyOfSetMap<K, C, WriteBatch = Self::WriteBatch> + Send + Sync;

    /// Creates a new write transaction.
    ///
    /// # Returns
    ///
    /// A new write transaction that can be used to collect write batches.
    fn new_write_batch(&self) -> Self::WriteBatch;

    /// Submits a write transaction to be applied to the storage backend.
    ///
    /// Once submitted, all operations collected in the write transaction are
    /// applied atomically.
    ///
    /// # Parameters
    ///
    /// - `write_transaction`: The write transaction to submit.
    fn submit_write_batch(&self, write_transaction: Self::WriteBatch);

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

/// A faux write manager for storage engines that do not require actual
/// write management, such as in-memory databases.
#[derive(Debug, Clone, Copy)]
pub struct FauxWriteManager;

impl WriteManager for FauxWriteManager {
    type WriteBatch = FauxWriteBatch;

    type SingleMap<K: WideColumn, V: WideColumnValue<K>> =
        InMemorySingleMap<K, V>;

    type DynamicMap<K: WideColumn> = InMemoryDynamicMap<K>;

    type KeyOfSetMap<
        K: KeyOfSetColumn,
        C: ConcurrentSet<Element = K::Element>,
    > = InMemoryKeyOfSetMap<K, C>;

    fn new_write_batch(&self) -> Self::WriteBatch { FauxWriteBatch }

    fn submit_write_batch(&self, _write_transaction: Self::WriteBatch) {
        // No-op for faux write manager
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
