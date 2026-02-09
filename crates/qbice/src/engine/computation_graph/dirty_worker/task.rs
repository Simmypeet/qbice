use std::sync::{Arc, atomic::AtomicUsize};

use tokio::sync::{Mutex, Notify, futures::OwnedNotified};

use crate::{Config, config::WriteTransaction, query::QueryID};

pub struct WorkTracker {
    active_task_count: AtomicUsize,
    notify: Arc<Notify>,
}

impl WorkTracker {
    pub fn done(&self) {
        let count = self
            .active_task_count
            .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);

        if count == 1 {
            self.notify.notify_waiters();
        }
    }

    pub fn new_task(&self) {
        self.active_task_count
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
}

pub struct DirtyTask<C: Config> {
    query_id: QueryID,

    write_tx: Arc<Mutex<WriteTransaction<C>>>,
    work_tracker: Arc<WorkTracker>,
}

impl<C: Config> DirtyTask<C> {
    #[allow(clippy::cast_possible_truncation)]
    pub const fn determine_shard(&self, shard_count: usize) -> usize {
        self.query_id.compact_hash_128().low() as usize % shard_count
    }

    pub const fn query_id(&self) -> &QueryID { &self.query_id }

    pub fn propagate_to(&self, query_id: QueryID) -> Self {
        self.work_tracker.new_task();

        Self {
            query_id,
            write_tx: self.write_tx.clone(),
            work_tracker: self.work_tracker.clone(),
        }
    }

    pub async fn write_tx_lock(
        &self,
    ) -> tokio::sync::MutexGuard<'_, WriteTransaction<C>> {
        self.write_tx.lock().await
    }
}

pub struct Batch<C: Config> {
    work_traker: Arc<WorkTracker>,
    write_tx: Arc<Mutex<WriteTransaction<C>>>,
}

impl<C: Config> Batch<C> {
    pub fn new(write_tx: Arc<Mutex<WriteTransaction<C>>>) -> Self {
        Self {
            work_traker: Arc::new(WorkTracker {
                active_task_count: AtomicUsize::new(1),

                notify: Arc::new(Notify::new()),
            }),
            write_tx,
        }
    }

    pub fn notified_owned(&self) -> OwnedNotified {
        self.work_traker.notify.clone().notified_owned()
    }

    pub fn new_task(&self, query_id: QueryID) -> DirtyTask<C> {
        self.work_traker.new_task();

        DirtyTask {
            query_id,
            write_tx: self.write_tx.clone(),
            work_tracker: self.work_traker.clone(),
        }
    }
}

impl<C: Config> Drop for Batch<C> {
    fn drop(&mut self) { self.work_traker.done(); }
}

impl<C: Config> Drop for DirtyTask<C> {
    fn drop(&mut self) { self.work_tracker.done(); }
}

impl<C: Config> Clone for DirtyTask<C> {
    fn clone(&self) -> Self {
        self.work_tracker.new_task();

        Self {
            query_id: self.query_id,
            write_tx: self.write_tx.clone(),
            work_tracker: self.work_tracker.clone(),
        }
    }
}
