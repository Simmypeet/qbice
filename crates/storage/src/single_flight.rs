//! A single-flight implementation to suppress duplicate work.

use std::{
    collections::hash_map::Entry,
    hash::{BuildHasher, Hash},
    sync::Arc,
};

use fxhash::{FxBuildHasher, FxHashMap};
use tokio::sync::Notify;

use crate::sharded::{self, Sharded};

/// A single-flight mechanism to ensure that only one concurrent operation
/// is performed for a given key.
///
/// Other concurrent requests for the same key will wait for the first operation
/// to complete.
pub struct SingleFlight<K> {
    map: Sharded<FxHashMap<K, Arc<Notify>>>,
    build_hasher: FxBuildHasher,
}

impl<K> std::fmt::Debug for SingleFlight<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SingleFlight").finish_non_exhaustive()
    }
}

impl<K> SingleFlight<K> {
    pub fn new(shard_count: usize) -> Self {
        Self {
            map: sharded::Sharded::new(shard_count, |_| FxHashMap::default()),
            build_hasher: FxBuildHasher::default(),
        }
    }
}

impl<K: Eq + Hash + Clone> SingleFlight<K> {
    /// Waits for an ongoing operation for the given key to complete, or
    /// performs the work if no operation is ongoing.
    pub async fn wait_or_work<T>(
        &self,
        key: &K,
        work: impl FnOnce() -> T,
    ) -> Option<T> {
        let hash = self.build_hasher.hash_one(key);
        let shard_index = self.map.shard_index(hash);

        let notified = {
            let mut shard = self.map.write_shard(shard_index);

            match shard.entry(key.clone()) {
                Entry::Occupied(occupied_entry) => {
                    Ok(occupied_entry.get().clone().notified_owned())
                }
                Entry::Vacant(vacant_entry) => {
                    let notify = Arc::new(Notify::new());
                    vacant_entry.insert(notify.clone());

                    Err(notify)
                }
            }
        };

        match notified {
            Ok(notified) => {
                notified.await;

                // we were a waiter, so no result
                None
            }
            Err(notify) => {
                let result = work();

                let mut shard = self.map.write_shard(shard_index);
                shard.remove(key);

                notify.notify_waiters();

                // we were the worker, so return the result
                Some(result)
            }
        }
    }
}
