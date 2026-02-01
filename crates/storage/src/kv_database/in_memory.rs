//! A simple in-memory key-value database implementation.

use std::{hash::BuildHasher, marker::PhantomData};

use crate::{
    dynamic_map::in_memory::InMemoryDynamicMap,
    key_of_set_map::in_memory::InMemoryKeyOfSetMap,
    kv_database::{self, WideColumn, WideColumnValue},
    single_map::in_memory::InMemorySingleMap,
    storage_engine::{self, StorageEngine},
    write_manager::FauxWriteManager,
    write_transaction::FauxWriteTransaction,
};

/// An in-memory key-value database implementation.
#[derive(Debug)]
pub struct InMemory<S>(pub PhantomData<S>);

/// A factory for creating in-memory key-value databases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InMemoryFactory<S>(pub PhantomData<S>);

impl<S> InMemoryFactory<S> {
    /// Creates a new in-memory key-value database factory.
    #[must_use]
    pub const fn new() -> Self { Self(PhantomData) }
}

impl<S> Default for InMemoryFactory<S> {
    fn default() -> Self { Self::new() }
}

impl<S> storage_engine::StorageEngineFactory for InMemoryFactory<S> {
    type StorageEngine = InMemory<S>;

    type Error = std::convert::Infallible;

    fn open(
        self,
        _: qbice_serialize::Plugin,
    ) -> Result<Self::StorageEngine, Self::Error> {
        Ok(InMemory(PhantomData))
    }
}

impl<S: BuildHasher + Clone + Send + Sync + Default> StorageEngine
    for InMemory<S>
{
    type WriteTransaction = FauxWriteTransaction;

    type WriteManager = FauxWriteManager;

    type SingleMap<K: WideColumn, V: WideColumnValue<K>> =
        InMemorySingleMap<K, V, S>;

    type DynamicMap<K: WideColumn> = InMemoryDynamicMap<K, S>;

    type KeyOfSetMap<
        K: kv_database::KeyOfSetColumn,
        C: crate::key_of_set_map::ConcurrentSet<Element = K::Element>,
    > = InMemoryKeyOfSetMap<K, C, S>;

    fn new_write_manager(&self) -> Self::WriteManager { FauxWriteManager }

    fn new_single_map<K: WideColumn, V: WideColumnValue<K>>(
        &self,
    ) -> Self::SingleMap<K, V> {
        InMemorySingleMap::new(S::default())
    }

    fn new_dynamic_map<K: WideColumn>(&self) -> Self::DynamicMap<K> {
        InMemoryDynamicMap::new(S::default())
    }

    fn new_key_of_set_map<
        K: kv_database::KeyOfSetColumn,
        C: crate::key_of_set_map::ConcurrentSet<Element = K::Element>,
    >(
        &self,
    ) -> Self::KeyOfSetMap<K, C> {
        InMemoryKeyOfSetMap::new(S::default())
    }
}
