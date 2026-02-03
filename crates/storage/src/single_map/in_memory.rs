//! In-memory implementation of [`SingleMap`].

use std::hash::BuildHasher;

use dashmap::DashMap;

use crate::{
    kv_database::{WideColumn, WideColumnValue},
    sharded::default_shard_amount,
    single_map::SingleMap,
    write_transaction::FauxWriteTransaction,
};

/// An in-memory implementation of [`SingleMap`] backed by a concurrent
/// [`DashMap`].
///
/// This implementation stores all key-value pairs in memory and is suitable
/// for testing, caching, or scenarios where persistence is not required.
///
/// # Type Parameters
///
/// - `K`: The wide column type that defines the key type.
/// - `V`: The value type to store.
/// - `S`: The hash builder type for the underlying [`DashMap`].
#[derive(Debug)]
pub struct InMemorySingleMap<K: WideColumn, V, S: BuildHasher + Clone> {
    map: DashMap<K::Key, V, S>,
}

impl<K: WideColumn, V, S: BuildHasher + Clone> InMemorySingleMap<K, V, S> {
    /// Creates a new in-memory single map with the specified hash builder.
    ///
    /// # Parameters
    ///
    /// - `hash_builder`: The hash builder to use for the underlying map.
    ///
    /// # Returns
    ///
    /// A new instance of `InMemorySingleMap`.
    pub fn new(hash_builder: S) -> Self {
        Self {
            map: DashMap::with_capacity_and_hasher(
                default_shard_amount(),
                hash_builder,
            ),
        }
    }
}

impl<K: WideColumn, V: WideColumnValue<K>, S: BuildHasher + Clone + Send + Sync>
    SingleMap<K, V> for InMemorySingleMap<K, V, S>
{
    type WriteTransaction = FauxWriteTransaction;

    async fn get(&self, key: &K::Key) -> Option<V> {
        self.map.get(key).map(|v| v.value().clone())
    }

    async fn insert(
        &self,
        key: K::Key,
        value: V,
        _write_transaction: &mut Self::WriteTransaction,
    ) {
        self.map.insert(key, value);
    }

    async fn remove(
        &self,
        key: &K::Key,
        _write_transaction: &mut Self::WriteTransaction,
    ) {
        self.map.remove(key);
    }
}
