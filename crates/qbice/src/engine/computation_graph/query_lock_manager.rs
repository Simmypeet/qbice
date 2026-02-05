use std::sync::Arc;

use fxhash::FxBuildHasher;
use quick_cache::{Lifecycle, UnitWeighter};
use tokio::sync::RwLock;

use crate::query::QueryID;

#[derive(Debug)]
pub enum QueryLock {
    Exclusive(#[allow(unused)] tokio::sync::OwnedRwLockWriteGuard<()>),
    Shared(#[allow(unused)] tokio::sync::OwnedRwLockReadGuard<()>),
}

#[derive(Debug, Clone)]
pub struct OwnedLock(Arc<RwLock<()>>);

#[derive(Debug, Clone, Default)]
pub struct Pinner;

impl Lifecycle<QueryID, OwnedLock> for Pinner {
    type RequestState = ();

    fn is_pinned(&self, _key: &QueryID, val: &OwnedLock) -> bool {
        // the lock is used elsewhere, can't evict
        Arc::strong_count(&val.0) > 1
    }

    fn begin_request(&self) -> Self::RequestState {}

    fn on_evict(
        &self,
        _state: &mut Self::RequestState,
        _key: QueryID,
        _val: OwnedLock,
    ) {
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
    hot: quick_cache::sync::Cache<
        QueryID,
        OwnedLock,
        UnitWeighter,
        FxBuildHasher,
        Pinner,
    >,
}

impl QueryLockManager {
    /// Create a new LockManager with the given capacity for the hot cache.
    #[allow(clippy::cast_possible_truncation)]
    pub fn new(capacity: u64) -> Self {
        let cache = quick_cache::sync::Cache::<
            QueryID,
            OwnedLock,
            UnitWeighter,
            FxBuildHasher,
            Pinner,
        >::with(
            capacity as usize,
            capacity,
            UnitWeighter,
            FxBuildHasher::default(),
            Pinner,
        );

        Self { hot: cache }
    }

    pub fn get_lock_instance(&self, query_id: &QueryID) -> OwnedLock {
        loop {
            // FAST PATH: Check hot cache first, no memory allocation needed,
            // just an atomic count bump.
            if let Some(lock) = self.hot.get(query_id) {
                return lock;
            }

            match self.hot.get_value_or_guard(query_id, None) {
                quick_cache::sync::GuardResult::Value(value) => return value,
                quick_cache::sync::GuardResult::Guard(placeholder_guard) => {
                    let owned_lock = OwnedLock(Arc::new(RwLock::new(())));

                    if matches!(
                        placeholder_guard.insert(owned_lock.clone()),
                        Ok(())
                    ) {
                        return owned_lock;
                    }
                }

                quick_cache::sync::GuardResult::Timeout => {
                    unreachable!("we didn't request a timeout")
                }
            }
        }
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
