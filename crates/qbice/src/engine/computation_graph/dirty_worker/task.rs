use std::{
    hash::BuildHasher,
    mem::ManuallyDrop,
    sync::{Arc, atomic::AtomicUsize},
};

use crossbeam::queue::SegQueue;
use fxhash::FxBuildHasher;
use tokio::sync::{Mutex, Notify, futures::OwnedNotified};

use crate::{
    Config, config::WriteTransaction,
    engine::computation_graph::database::Edge, query::QueryID,
};

pub struct StrippedBuffer {
    buffers: Box<[SegQueue<Edge>]>,
    mask: usize,
}

impl StrippedBuffer {
    pub fn new() -> Self {
        let parallelism = std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(8);

        let size = parallelism.next_power_of_two();
        let buffers = (0..size)
            .map(|_| SegQueue::new())
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Self { buffers, mask: size - 1 }
    }

    #[allow(clippy::cast_possible_truncation)]
    pub fn push(&self, query_id: Edge) {
        let index = FxBuildHasher::default()
            .hash_one(std::thread::current().id()) as usize;

        self.buffers[index & self.mask].push(query_id);
    }

    pub fn drain_all(&self) -> impl Iterator<Item = Edge> {
        self.buffers
            .iter()
            .flat_map(|queue| std::iter::from_fn(move || queue.pop()))
    }

    pub fn drain_limited(&self) -> impl Iterator<Item = Edge> {
        self.buffers.iter().flat_map(|queue| {
            std::iter::from_fn({
                // prevent new tasks keep piling up in the queue
                let mut count = queue.len();
                move || {
                    if count == 0 {
                        None
                    } else {
                        count -= 1;
                        queue.pop()
                    }
                }
            })
        })
    }
}

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

    // ManuallyDrop ensures we can drop write_tx before calling done()
    // to avoid race condition where waiter wakes before Arc is released
    write_tx: ManuallyDrop<Arc<Mutex<WriteTransaction<C>>>>,
    work_tracker: Arc<WorkTracker>,
    stripped_buffer: Arc<StrippedBuffer>,
}

impl<C: Config> DirtyTask<C> {
    pub const fn query_id(&self) -> &QueryID { &self.query_id }

    pub fn propagate_to(&self, query_id: QueryID) -> Self {
        self.work_tracker.new_task();

        Self {
            query_id,
            write_tx: ManuallyDrop::new((*self.write_tx).clone()),
            work_tracker: self.work_tracker.clone(),
            stripped_buffer: self.stripped_buffer.clone(),
        }
    }

    pub fn drain_limited(&self) -> impl Iterator<Item = Edge> + '_ {
        self.stripped_buffer.drain_limited()
    }

    pub fn push_to_buffer(&self, edge: Edge) {
        self.stripped_buffer.push(edge);
    }

    pub fn try_load_write_tx(
        &self,
    ) -> Option<tokio::sync::MutexGuard<'_, WriteTransaction<C>>> {
        self.write_tx.try_lock().ok()
    }
}

pub struct Batch<C: Config> {
    work_traker: Arc<WorkTracker>,
    stripped_buffer: Arc<StrippedBuffer>,
    // ManuallyDrop ensures we can drop write_tx before calling done()
    // to avoid race condition where waiter wakes before Arc is released
    write_tx: ManuallyDrop<Arc<Mutex<WriteTransaction<C>>>>,
}

impl<C: Config> Batch<C> {
    pub fn new(
        write_tx: Arc<Mutex<WriteTransaction<C>>>,
        stripped_buffer: Arc<StrippedBuffer>,
    ) -> Self {
        Self {
            work_traker: Arc::new(WorkTracker {
                active_task_count: AtomicUsize::new(1),

                notify: Arc::new(Notify::new()),
            }),
            stripped_buffer,
            write_tx: ManuallyDrop::new(write_tx),
        }
    }

    pub fn notified_owned(&self) -> OwnedNotified {
        self.work_traker.notify.clone().notified_owned()
    }

    pub fn new_task(&self, query_id: QueryID) -> DirtyTask<C> {
        self.work_traker.new_task();

        DirtyTask {
            query_id,
            write_tx: ManuallyDrop::new((*self.write_tx).clone()),
            stripped_buffer: self.stripped_buffer.clone(),
            work_tracker: self.work_traker.clone(),
        }
    }
}

impl<C: Config> Drop for Batch<C> {
    fn drop(&mut self) {
        // SAFETY: Drop write_tx before calling done() to ensure the Arc
        // is released before any waiters are notified. This prevents the
        // race where a waiter wakes up and tries Arc::try_unwrap while
        // we still hold a reference.
        unsafe { ManuallyDrop::drop(&mut self.write_tx) };
        self.work_traker.done();
    }
}

impl<C: Config> Drop for DirtyTask<C> {
    fn drop(&mut self) {
        // SAFETY: Drop write_tx before calling done() to ensure the Arc
        // is released before any waiters are notified. This prevents the
        // race where a waiter wakes up and tries Arc::try_unwrap while
        // we still hold a reference.
        unsafe { ManuallyDrop::drop(&mut self.write_tx) };
        self.work_tracker.done();
    }
}

impl<C: Config> Clone for DirtyTask<C> {
    fn clone(&self) -> Self {
        self.work_tracker.new_task();

        Self {
            query_id: self.query_id,
            write_tx: ManuallyDrop::new((*self.write_tx).clone()),
            work_tracker: self.work_tracker.clone(),
            stripped_buffer: self.stripped_buffer.clone(),
        }
    }
}
