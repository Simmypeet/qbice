use std::{any::TypeId, hash::Hash};

use dashmap::DashMap;

use crate::{
    kv_database::{KvDatabase, WideColumn, WideColumnValue},
    write_manager::write_behind::{self, Epoch},
};

#[derive(Debug)]
pub struct VersionedValue<V> {
    pub value: V,
    pub version: Epoch,
}

#[derive(Debug)]
pub struct WideColumnCache<
    K: Eq + Hash + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
    T,
> {
    staging: DashMap<K, VersionedValue<Option<V>>>,
    moka: moka::future::Cache<K, V>,
    _phantom: std::marker::PhantomData<T>,
}

impl<K: Eq + Hash + Send + Sync + 'static, V: Clone + Send + Sync + 'static, T>
    WideColumnCache<K, V, T>
{
    pub fn new(capacity: u64) -> Self {
        Self {
            staging: DashMap::new(),
            moka: moka::future::Cache::builder().max_capacity(capacity).build(),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<
    K: Eq + Hash + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
    T,
> WideColumnCache<K, V, T>
{
    pub async fn get(
        &self,
        key: &K,
        init: impl Future<Output = Option<V>>,
    ) -> Option<V> {
        if let Some(staged) = self.staging.get(key) {
            return staged.value.clone();
        }

        self.moka.optionally_get_with(key.clone(), init).await
    }

    pub fn insert(&self, key: K, value: V, epoch: Epoch) {
        let old = self
            .staging
            .insert(key, VersionedValue { value: Some(value), version: epoch });

        if let Some(old) = old {
            assert!(
                old.version <= epoch,
                "out-of-order write detected in cache"
            );
        }
    }

    pub fn remove(&self, key: &K, epoch: Epoch) {
        let old = self.staging.insert(key.clone(), VersionedValue {
            value: None,
            version: epoch,
        });

        if let Some(old) = old {
            assert!(
                old.version <= epoch,
                "out-of-order write detected in CacheSingleMap"
            );
        }
    }
}

impl<
    K: Eq + Hash + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
    T,
> WideColumnCache<K, V, T>
{
    pub(crate) async fn flush_staging(
        &self,
        epoch: Epoch,
        keys: impl IntoIterator<Item = K>,
    ) {
        for key in keys {
            // 1. READ: Check if the staging entry is ready to be flushed.
            // We clone the data we need to minimize locking time on the DashMap
            // shard.
            let should_promote = if let Some(entry) = self.staging.get(&key) {
                // CRITICAL: Only flush if the staged version is <= the
                // committed epoch. If entry.version > epoch, a
                // NEWER write happened. We must leave it alone.
                if entry.version <= epoch {
                    Some(entry.value.clone())
                } else {
                    None
                }
            } else {
                None
            };

            // 2. PROMOTE: Update Moka (The Clean Cache)
            // We do this BEFORE removing from staging to ensure there is no
            // "gap" where a reader sees the key missing from both
            // Staging and Moka.
            if let Some(staged_value) = should_promote {
                match staged_value {
                    Some(val) => {
                        // It's an Upsert: Put it in Moka
                        self.moka.insert(key.clone(), val).await;
                    }
                    None => {
                        // It's a Delete (Tombstone): Remove from Moka
                        self.moka.invalidate(&key).await;
                    }
                }

                // 3. CLEANUP: Safe Remove from Staging
                // We use remove_if to handle the "ABA" race condition.
                // If a new write came in (bumping version > epoch) while we
                // were awaiting the moka insert, this closure
                // will return false, and we will CORRECTLY
                // leave the new dirty value in staging.
                self.staging.remove_if(&key, |_, current_val| {
                    current_val.version <= epoch
                });
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct SingleMapTag;

impl<K: WideColumn, V: WideColumnValue<K>, Db: KvDatabase>
    write_behind::WideColumnCache<K, V, Db>
    for WideColumnCache<K::Key, V, SingleMapTag>
{
    fn flush<'s: 'x, 'i: 'x, 'x>(
        &'s self,
        epoch: Epoch,
        keys: &'i mut (dyn Iterator<Item = <K as WideColumn>::Key> + Send),
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + 'x>>
    {
        Box::pin(self.flush_staging(epoch, keys))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct DynamicMapTag;

impl<
    K: WideColumn,
    V: WideColumnValue<K>,
    X: Clone + Send + Sync + 'static,
    Db: KvDatabase,
> write_behind::WideColumnCache<K, V, Db>
    for WideColumnCache<(K::Key, TypeId), X, DynamicMapTag>
{
    fn flush<'s: 'x, 'i: 'x, 'x>(
        &'s self,
        epoch: Epoch,
        keys: &'i mut (dyn Iterator<Item = <K as WideColumn>::Key> + Send),
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + 'x>>
    {
        Box::pin(self.flush_staging(
            epoch,
            keys.map(|x| (x, std::any::TypeId::of::<V>())),
        ))
    }
}
