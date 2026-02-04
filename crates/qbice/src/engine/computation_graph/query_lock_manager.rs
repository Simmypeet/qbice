use std::sync::{Arc, Weak};

use dashmap::DashMap;
use tokio::sync::RwLock;

use crate::query::QueryID;

#[derive(Debug)]
pub enum QueryLock {
    Exclusive(#[allow(unused)] tokio::sync::OwnedRwLockWriteGuard<()>),
    Shared(#[allow(unused)] tokio::sync::OwnedRwLockReadGuard<()>),
}

#[derive(Debug, Clone)]
pub struct OwnedLock(Arc<RwLock<()>>);

#[derive(Debug, Clone)]
pub struct WeakLock(Weak<RwLock<()>>);

/// Manages query-level locks
///
/// Each lock has it's associated lock instance. There are two kinds of locks
/// similar to how RwLock works.
///
/// - Exclusive locks: required for writing to query data (e.g. computing a
///   query value, repairing a query, etc.)
/// - Shared locks: required for reading query data (e.g. reading a query value
///   for use in computing another query)
pub struct QueryLockManager {
    cold: Arc<DashMap<QueryID, WeakLock>>,
    hot: moka::sync::Cache<QueryID, OwnedLock>,
}

impl QueryLockManager {
    /// Create a new LockManager with the given capacity for the hot cache.
    pub fn new(capacity: u64) -> Self {
        let cold = Arc::new(DashMap::<QueryID, WeakLock>::new());
        let hot = moka::sync::Cache::<QueryID, OwnedLock>::builder()
            .max_capacity(capacity)
            .eviction_listener({
                let cold = cold.clone();
                move |key, value, _| {
                    // Drop the strong reference from the hot cache
                    drop(value);

                    cold.remove_if(&*key, |_, v| v.0.strong_count() == 0);
                }
            })
            .build();

        Self { cold, hot }
    }

    pub fn get_lock_instance(&self, query_id: &QueryID) -> OwnedLock {
        // FAST PATH: Check hot cache first, no memory allocation needed, just
        // an atomic count bump.
        if let Some(lock) = self.hot.get(query_id) {
            return lock;
        }

        // SLOW PATH: Not in hot cache, check cold cache.
        let result = match self.cold.entry(*query_id) {
            dashmap::Entry::Occupied(mut occupied_entry) => {
                occupied_entry.get().0.upgrade().map_or_else(
                    || {
                        // The weak reference is dead, create a new one.
                        let new_lock = OwnedLock(Arc::new(RwLock::new(())));

                        occupied_entry
                            .insert(WeakLock(Arc::downgrade(&new_lock.0)));

                        new_lock
                    },
                    OwnedLock,
                )
            }
            dashmap::Entry::Vacant(vacant_entry) => {
                let new_lock = OwnedLock(Arc::new(RwLock::new(())));

                vacant_entry.insert(WeakLock(Arc::downgrade(&new_lock.0)));

                new_lock
            }
        };

        // Insert into hot cache for faster access next time.
        self.hot.insert(*query_id, result.clone());

        result
    }

    pub async fn acquire_exclusive_lock(
        &self,
        query_id: &QueryID,
    ) -> QueryLock {
        let lock_instance = self.get_lock_instance(query_id);
        let guard = lock_instance.0.clone().write_owned().await;

        QueryLock::Exclusive(guard)
    }

    pub async fn acquire_shared_lock(&self, query_id: &QueryID) -> QueryLock {
        let lock_instance = self.get_lock_instance(query_id);
        let guard = lock_instance.0.clone().read_owned().await;

        QueryLock::Shared(guard)
    }
}
