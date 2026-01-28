use std::{hash::Hash, sync::Arc};

use dashmap::DashMap;
use parking_lot::{Condvar, Mutex};

/// Synchronization primitive for single-flight requests.
struct Barrier {
    /// Indicates if the operation is complete.
    finished: Mutex<bool>,
    /// Waiters sleep on this.
    cvar: Condvar,
}

impl Barrier {
    const fn new() -> Self {
        Self { finished: Mutex::new(false), cvar: Condvar::new() }
    }

    fn wait(&self) {
        let mut finished = self.finished.lock();
        while !*finished {
            self.cvar.wait(&mut finished);
        }
    }

    fn notify(&self) {
        let mut finished = self.finished.lock();
        *finished = true;
        self.cvar.notify_all();
    }
}

pub struct SingleFlight<K> {
    in_flight: DashMap<K, Arc<Barrier>>,
}

impl<K: Eq + Hash> SingleFlight<K> {
    pub fn new(mut shard_amount: usize) -> Self {
        if shard_amount == 1 {
            shard_amount = 2;
        }

        Self { in_flight: DashMap::with_shard_amount(shard_amount) }
    }

    pub fn work_or_wait<F, R>(&self, key: K, work: F) -> Option<R>
    where
        K: std::hash::Hash + Eq + Clone,
        F: FnOnce() -> R,
    {
        let barrier = Arc::new(Barrier::new());
        let entry = self.in_flight.entry(key);
        match entry {
            dashmap::mapref::entry::Entry::Occupied(occ) => {
                // Another thread is already working on this key; wait.
                let barrier = occ.get().clone();
                drop(occ);
                barrier.wait();
                None
            }
            dashmap::mapref::entry::Entry::Vacant(vac) => {
                // We are the first to request this key; do the work.
                let key = vac.insert(barrier.clone()).key().clone();

                let result = work();
                barrier.notify();

                self.in_flight.remove(&key);

                Some(result)
            }
        }
    }
}
