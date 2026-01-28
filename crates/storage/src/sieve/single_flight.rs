use std::{hash::Hash, sync::Arc};

use dashmap::DashMap;
use tokio::sync::Notify;

pub struct SingleFlight<K> {
    in_flight: DashMap<K, Arc<Notify>>,
}

impl<K: Eq + Hash> SingleFlight<K> {
    pub fn new(shard_amount: usize) -> Self {
        Self { in_flight: DashMap::with_shard_amount(shard_amount) }
    }

    pub async fn work_or_wait<F, R>(&self, key: K, work: F) -> Option<R>
    where
        K: std::hash::Hash + Eq + Clone,
        F: FnOnce() -> R,
    {
        let entry = self.in_flight.entry(key);
        match entry {
            dashmap::mapref::entry::Entry::Occupied(occ) => {
                // Another thread is already working on this key; wait.
                let barrier = occ.get().clone();
                let notified = barrier.notified();
                drop(occ);

                notified.await;

                None
            }
            dashmap::mapref::entry::Entry::Vacant(vac) => {
                // We are the first to request this key; do the work.
                let notify = Arc::new(Notify::new());
                let key = vac.insert(notify.clone()).key().clone();

                let result = work();

                self.in_flight.remove(&key);

                notify.notify_waiters();

                Some(result)
            }
        }
    }
}
