use std::{any::Any, hash::BuildHasher};

use dashmap::DashMap;

use crate::{
    dynamic_map::DynamicMap,
    kv_database::{WideColumn, WideColumnValue},
    sharded::default_shard_amount,
    write_transaction::FauxWriteTransaction,
};

/// An in-memory implementation of [`DynamicMap`] backed by a concurrent
/// [`DashMap`].
///
/// This implementation stores all key-value pairs in memory, using type
/// erasure to support dynamic value types. Values are stored as boxed trait
/// objects and downcasted on retrieval.
///
/// # Type Parameters
///
/// - `K`: The wide column type that defines the key type and discriminant.
/// - `S`: The hash builder type for the underlying [`DashMap`].
#[derive(Debug)]
#[allow(clippy::type_complexity)]
pub struct InMemoryDynamicMap<K: WideColumn, S: BuildHasher + Clone> {
    map: DashMap<(K::Key, K::Discriminant), Box<dyn Any + Send + Sync>, S>,
}

impl<K: WideColumn, S: BuildHasher + Clone + Send + Sync>
    InMemoryDynamicMap<K, S>
{
    /// Creates a new in-memory dynamic map with the specified hash builder.
    ///
    /// # Parameters
    ///
    /// - `hash_builder`: The hash builder to use for the underlying map.
    ///
    /// # Returns
    ///
    /// A new instance of `InMemoryDynamicMap`.
    pub fn new(hash_builder: S) -> Self {
        Self {
            map: DashMap::with_capacity_and_hasher(
                default_shard_amount() * 4,
                hash_builder,
            ),
        }
    }
}

impl<K: WideColumn, S: BuildHasher + Clone + Send + Sync> DynamicMap<K>
    for InMemoryDynamicMap<K, S>
{
    type WriteTransaction = FauxWriteTransaction;

    async fn get<V: WideColumnValue<K>>(&self, key: &K::Key) -> Option<V> {
        let discriminant = V::discriminant();

        self.map
            .get(&(key.clone(), discriminant))
            .and_then(|v| v.downcast_ref::<V>().cloned())
    }

    fn insert<V: WideColumnValue<K>>(
        &self,
        key: <K as WideColumn>::Key,
        value: V,
        _write_transaction: &mut Self::WriteTransaction,
    ) {
        let discriminant = V::discriminant();
        self.map.insert((key, discriminant), Box::new(value));
    }

    fn remove<V: WideColumnValue<K>>(
        &self,
        key: &<K as WideColumn>::Key,
        _write_transaction: &mut Self::WriteTransaction,
    ) {
        let discriminant = V::discriminant();
        self.map.remove(&(key.clone(), discriminant));
    }
}
