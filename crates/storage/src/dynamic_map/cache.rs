//! Cached implementation of [`DynamicMap`].
//!
//! This module provides [`CacheDynamicMap`], which wraps a database backend
//! with a Moka-based cache for improved read performance.

use std::{
    any::{Any, TypeId},
    fmt::Debug,
    sync::Arc,
};

use crate::{
    dynamic_map::DynamicMap,
    kv_database::{KvDatabase, WideColumn, WideColumnValue},
    wide_column_cache::{DynamicMapTag, WideColumnCache},
    write_manager::write_behind,
};

/// A cached implementation of [`DynamicMap`] backed by a
/// database.
///
/// This implementation combines a Moka cache for fast reads with a database
/// backend for persistence. Values of different types can be stored under
/// the same key using different discriminants.
///
/// # Type Parameters
///
/// - `K`: The wide column type defining the key and discriminant.
/// - `Db`: The database backend implementing [`KvDatabase`].
#[derive(Debug)]
#[allow(clippy::type_complexity)]
pub struct CacheDynamicMap<K: WideColumn, Db: KvDatabase> {
    cache: Arc<
        WideColumnCache<
            (K::Key, TypeId),
            Box<dyn Any + Send + Sync>,
            DynamicMapTag,
        >,
    >,
    db: Db,
}

impl<K: WideColumn, Db: KvDatabase> CacheDynamicMap<K, Db> {
    /// Creates a new cached dynamic map with the specified capacity.
    ///
    ///
    /// # Returns
    ///
    /// A new `CacheDynamicMap` instance.
    #[must_use]
    pub fn new(cap: u64, shard_amount: usize, db: Db) -> Self {
        Self { cache: Arc::new(WideColumnCache::new(cap, shard_amount)), db }
    }
}

impl<K: WideColumn, Db: KvDatabase> DynamicMap<K> for CacheDynamicMap<K, Db> {
    type WriteTransaction = write_behind::WriteBatch<Db>;

    async fn get<V: WideColumnValue<K>>(&self, key: &K::Key) -> Option<V> {
        let key = (key.clone(), std::any::TypeId::of::<V>());

        self.cache
            .get(
                &key,
                |x| {
                    x.downcast_ref::<V>()
                        .expect("incorrect type detected")
                        .clone()
                },
                || {
                    self.db
                        .get_wide_column::<K, V>(&key.0)
                        .map(|v| Box::new(v) as Box<dyn Any + Send + Sync>)
                },
            )
            .await
    }

    async fn insert<V: WideColumnValue<K>>(
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

        let cache_key = (key, std::any::TypeId::of::<V>());
        self.cache.insert(
            cache_key,
            Box::new(value) as Box<dyn Any + Send + Sync>,
            updated,
        );
    }

    async fn remove<V: WideColumnValue<K>>(
        &self,
        key: &K::Key,
        write_transaction: &mut Self::WriteTransaction,
    ) {
        let updated = write_transaction.put_wide_column::<K, V>(
            key.clone(),
            None,
            self.cache.clone(),
        );

        let cache_key = (key.clone(), std::any::TypeId::of::<V>());
        self.cache.remove(&cache_key, updated);
    }
}
