use std::sync::Arc;

use qbice_storage::tiny_lfu::{LifecycleListener, TinyLFU};
use tokio::sync::RwLock;

use crate::{engine::default_shard_amount, query::QueryID};

#[derive(Debug)]
pub enum QueryLock {
    Exclusive(#[allow(unused)] tokio::sync::OwnedRwLockWriteGuard<()>),
    Shared(#[allow(unused)] tokio::sync::OwnedRwLockReadGuard<()>),
}

#[derive(Debug, Clone)]
pub struct OwnedLock(Arc<RwLock<()>>);

#[derive(Debug, Clone, Default)]
pub struct ActiveLockLifecycleListener;

impl LifecycleListener<QueryID, OwnedLock> for ActiveLockLifecycleListener {
    fn is_pinned(&self, _key: &QueryID, value: &OwnedLock) -> bool {
        // Keep locks pinned while they are active (have at least one strong
        // reference).
        Arc::strong_count(&value.0) > 1
    }
}

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
    hot: TinyLFU<QueryID, OwnedLock, ActiveLockLifecycleListener>,
}

impl QueryLockManager {
    /// Create a new LockManager with the given capacity for the hot cache.
    #[allow(clippy::cast_possible_truncation)]
    pub fn new(capacity: u64) -> Self {
        let cache = TinyLFU::new(
            capacity as usize,
            default_shard_amount(),
            qbice_storage::tiny_lfu::UnpinStrategy::Poll,
        );

        Self { hot: cache }
    }

    pub fn get_lock_instance(&self, query_id: &QueryID) -> OwnedLock {
        // FAST PATH: Check hot cache first, no memory allocation needed,
        // just an atomic count bump.
        if let Some(lock) = self.hot.get(query_id) {
            return lock;
        }

        let lock_instance = OwnedLock(Arc::new(RwLock::new(())));

        self.hot.entry(*query_id, |x| match x {
            qbice_storage::tiny_lfu::Entry::Vacant(vacant_entry) => {
                vacant_entry.insert(lock_instance.clone());
                lock_instance
            }
            qbice_storage::tiny_lfu::Entry::Occupied(occupied_entry) => {
                occupied_entry.get().clone()
            }
        })
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
