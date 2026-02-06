use std::{collections::hash_map::Entry, hash::Hash, sync::Arc};

use fxhash::FxHashMap;
use tokio::sync::Notify;

use crate::sharded::{self, Sharded};

pub struct SingleFlight<K> {
    map: Sharded<FxHashMap<K, Arc<Notify>>>,
}

impl<K> SingleFlight<K> {
    pub fn new(shard_count: usize) -> Self {
        Self {
            map: sharded::Sharded::new(shard_count, |_| FxHashMap::default()),
        }
    }
}

impl<K: Eq + Hash + Clone> SingleFlight<K> {
    pub async fn wait_or_work<T>(
        &self,
        key: &K,
        shard_index: usize,
        work: impl FnOnce() -> T,
    ) -> Option<T> {
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
