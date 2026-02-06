//! Cached implementation of [`SingleMap`].
//!
//! This module provides [`CacheSingleMap`], which wraps a database backend
//! with a Moka-based cache for improved read performance.

use std::sync::Arc;

use crate::{
    kv_database::{KvDatabase, WideColumn, WideColumnValue},
    single_map::SingleMap,
    wide_column_cache::{SingleMapTag, WideColumnCache},
    write_manager::write_behind,
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
    cache: Arc<WideColumnCache<K::Key, V, SingleMapTag>>,
    db: Db,
}

impl<K: WideColumn, V: WideColumnValue<K>, Db: KvDatabase>
    CacheSingleMap<K, V, Db>
{
    /// Creates a new cached single map with the specified capacity.
    ///
    /// # Parameters
    ///
    /// - `cap`: The maximum number of entries to cache.
    /// - `db`: The database backend for persistence.
    ///
    /// # Returns
    ///
    /// A new `CacheSingleMap` instance.
    #[must_use]
    pub fn new(cap: u64, db: Db) -> Self {
        Self { cache: Arc::new(WideColumnCache::new(cap)), db }
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
        let updated = write_transaction.put_wide_column::<K, V>(
            key.clone(),
            Some(value.clone()),
            self.cache.clone(),
        );

        self.cache.insert(key, value, updated);
    }

    async fn remove(
        &self,
        key: &K::Key,
        write_transaction: &mut Self::WriteTransaction,
    ) {
        let updated = write_transaction.put_wide_column::<K, V>(
            key.clone(),
            None,
            self.cache.clone(),
        );

        self.cache.remove(key, updated);
    }
}
