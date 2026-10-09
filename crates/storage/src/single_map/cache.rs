//! Cached implementation of [`SingleMap`].
//!
//! This module provides [`CacheSingleMap`], which wraps a database backend
//! with a Moka-based cache for improved read performance.

use crate::{
    kv_database::{KvDatabase, WideColumn, WideColumnValue},
    single_map::SingleMap,
    wide_column_cache::WideColumnCache,
    write_manager::write_behind::{self, CommittedEpochs},
};

/// A cached implementation of [`SingleMap`] backed by a
/// database.
///
/// This implementation combines a Moka cache for fast reads with a database
/// backend for persistence. Cache misses are transparently loaded from the
/// database, and writes are staged for asynchronous persistence.
///
/// # Type Parameters
///
/// - `K`: The wide column type defining the key.
/// - `V`: The value type to store.
/// - `Db`: The database backend implementing [`KvDatabase`].
#[derive(Debug)]
pub struct CacheSingleMap<K: WideColumn, V: WideColumnValue<K>, Db: KvDatabase>
{
    cache: WideColumnCache<K::Key, V>,
    db: Db,
}

impl<K: WideColumn, V: WideColumnValue<K>, Db: KvDatabase>
    CacheSingleMap<K, V, Db>
{
    /// Creates a new cached single map with the specified capacity, for the
    /// write manager whose committed epochs are `committed`.
    pub(crate) fn new(cap: u64, db: Db, committed: CommittedEpochs) -> Self {
        Self { cache: WideColumnCache::new(cap, committed), db }
    }
}

impl<K: WideColumn, V: WideColumnValue<K>, Db: KvDatabase> SingleMap<K, V>
    for CacheSingleMap<K, V, Db>
{
    type WriteTransaction = write_behind::WriteBatch<Db>;

    async fn get(&self, key: &K::Key) -> Option<V> {
        self.cache
            .get(key, std::clone::Clone::clone, || {
                self.db.get_wide_column::<K, V>(key)
            })
            .await
    }

    async fn insert(
        &self,
        key: K::Key,
        value: V,
        write_transaction: &mut Self::WriteTransaction,
    ) {
        write_transaction.put_wide_column::<K, V>(&key, Some(&value));

        self.cache.insert(key, value, write_transaction.epoch());
    }

    async fn remove(
        &self,
        key: &K::Key,
        write_transaction: &mut Self::WriteTransaction,
    ) {
        write_transaction.put_wide_column::<K, V>(key, None);

        self.cache.remove(key, write_transaction.epoch());
    }
}
