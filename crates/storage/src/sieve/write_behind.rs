use std::{
    any::{Any, TypeId},
    collections::{BinaryHeap, HashMap},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread,
};

use crossbeam_deque::{Injector, Stealer};
use crossbeam_utils::Backoff;
use parking_lot::{Condvar, Mutex};

use crate::{
    kv_database::{KvDatabase, WideColumn, WideColumnValue, WriteBatch},
    sieve::{KeyOfSetContainer, KeyOfSetSieve, Operation, WideColumnSieve},
};

pub trait BuildHasher:
    std::hash::BuildHasher + Default + Send + Sync + 'static
{
}

impl<T> BuildHasher for T where
    T: std::hash::BuildHasher + Default + Send + Sync + 'static
{
}

struct TypedWideColumnWrites<
    C: WideColumn,
    V: WideColumnValue<C>,
    Db: KvDatabase,
    S: BuildHasher,
> {
    /// Map of keys to their corresponding values to write.
    ///
    /// `None` indicates a deletion for that key. `Some(value)` indicates
    /// an insertion or update.
    writes: HashMap<C::Key, Option<V>, S>,
    original_sieve: Arc<WideColumnSieve<C, Db, S>>,
}

pub trait WriteEntry<Db: KvDatabase>: Any + Send + Sync + 'static {
    fn write_to_db(&self, tx: &Db::WriteBatch);
    fn after_commit(&mut self);
    fn as_any_mut(&mut self) -> &mut (dyn Any + Send + Sync);
}

impl<C: WideColumn, V: WideColumnValue<C>, Db: KvDatabase, S: BuildHasher>
    WriteEntry<Db> for TypedWideColumnWrites<C, V, Db, S>
{
    fn write_to_db(&self, tx: &<Db as KvDatabase>::WriteBatch) {
        for (key, value_opt) in &self.writes {
            match value_opt {
                Some(value) => tx.put(key, value),
                None => {
                    tx.delete::<C, V>(key);
                }
            }
        }
    }

    fn after_commit(&mut self) {
        for (key, _) in self.writes.drain() {
            self.original_sieve
                .decrement_pending_write(&(key, V::discriminant()));
        }
    }

    fn as_any_mut(&mut self) -> &mut (dyn Any + Send + Sync) { self }
}

impl<C: WideColumn, V: WideColumnValue<C>, Db: KvDatabase, S: BuildHasher>
    TypedWideColumnWrites<C, V, Db, S>
{
    fn insert(&mut self, key: C::Key, value: Option<V>) -> bool {
        self.writes.insert(key, value).is_none()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct WideColumnWritesID {
    column_id: TypeId,
    value_id: TypeId,
}

impl WideColumnWritesID {
    fn of<C: WideColumn, V: WideColumnValue<C>>() -> Self {
        Self { column_id: TypeId::of::<C>(), value_id: TypeId::of::<V>() }
    }
}

#[allow(clippy::type_complexity)]
pub(super) struct WideColumnWrites<Db: KvDatabase, S: BuildHasher> {
    writes: HashMap<WideColumnWritesID, Box<dyn WriteEntry<Db>>, S>,
}

impl<Db: KvDatabase, S: BuildHasher> WideColumnWrites<Db, S> {
    fn new() -> Self { Self { writes: HashMap::default() } }

    pub(super) fn put<C: WideColumn, V: WideColumnValue<C>>(
        &mut self,
        key: C::Key,
        value: Option<V>,
        original_sieve: &Arc<WideColumnSieve<C, Db, S>>,
    ) -> bool {
        let id = WideColumnWritesID::of::<C, V>();

        match self.writes.entry(id) {
            std::collections::hash_map::Entry::Occupied(mut occupied_entry) => {
                let typed_writes = occupied_entry
                    .get_mut()
                    .as_any_mut()
                    .downcast_mut::<TypedWideColumnWrites<C, V, Db, S>>()
                    .expect("type mismatch in WideColumnWrites map");

                assert!(
                    Arc::ptr_eq(&typed_writes.original_sieve, original_sieve),
                    "original_sieve mismatch for existing WideColumnWrites \
                     entry"
                );

                typed_writes.insert(key, value)
            }

            std::collections::hash_map::Entry::Vacant(vacant_entry) => {
                let mut typed_writes = TypedWideColumnWrites::<C, V, Db, S> {
                    writes: HashMap::with_hasher(S::default()),
                    original_sieve: original_sieve.clone(),
                };

                let result = typed_writes.insert(key, value);

                vacant_entry.insert(Box::new(typed_writes));

                result
            }
        }
    }

    pub(super) fn write_to_db(&self, tx: &<Db as KvDatabase>::WriteBatch) {
        for write_entry in self.writes.values() {
            write_entry.write_to_db(tx);
        }
    }

    pub(super) fn after_commit(&mut self) {
        for write_entry in self.writes.values_mut() {
            write_entry.after_commit();
        }
    }
}

struct TypedKeyOfSetWrites<C: KeyOfSetContainer, Db: KvDatabase, S: BuildHasher>
{
    writes: HashMap<C::Key, HashMap<C::Element, Operation, S>, S>,
    original_sieve: Arc<KeyOfSetSieve<C, Db, S>>,
}

impl<C: KeyOfSetContainer, Db: KvDatabase, S: BuildHasher> WriteEntry<Db>
    for TypedKeyOfSetWrites<C, Db, S>
{
    fn write_to_db(&self, tx: &<Db as KvDatabase>::WriteBatch) {
        for (key, element_map) in &self.writes {
            for (element, op) in element_map {
                match op {
                    Operation::Insert => {
                        tx.insert_member::<C>(key, element);
                    }
                    Operation::Delete => {
                        tx.delete_member::<C>(key, element);
                    }
                }
            }
        }
    }

    fn after_commit(&mut self) {
        for (key, element_map) in self.writes.drain() {
            for (_, _) in element_map {
                self.original_sieve.decrement_pending_write(&key);
            }
        }
    }

    fn as_any_mut(&mut self) -> &mut (dyn Any + Send + Sync) { self }
}

impl<C: KeyOfSetContainer, Db: KvDatabase, S: BuildHasher>
    TypedKeyOfSetWrites<C, Db, S>
{
    fn insert(
        &mut self,
        key: C::Key,
        element: C::Element,
        op: Operation,
    ) -> bool {
        match self.writes.entry(key) {
            std::collections::hash_map::Entry::Occupied(mut occupied_entry) => {
                occupied_entry.get_mut().insert(element, op);
                false
            }

            std::collections::hash_map::Entry::Vacant(vacant_entry) => {
                let mut element_map = HashMap::with_hasher(S::default());
                element_map.insert(element, op);
                vacant_entry.insert(element_map);
                true
            }
        }
    }
}

#[allow(clippy::type_complexity)]
pub(super) struct KeyOfSetWrites<Db: KvDatabase, S: BuildHasher> {
    writes: HashMap<TypeId, Box<dyn WriteEntry<Db>>, S>,
}

impl<Db: KvDatabase, S: BuildHasher> KeyOfSetWrites<Db, S> {
    fn new() -> Self { Self { writes: HashMap::default() } }

    pub(super) fn put<C: KeyOfSetContainer>(
        &mut self,
        key: C::Key,
        element: C::Element,
        op: Operation,
        original_sieve: &Arc<KeyOfSetSieve<C, Db, S>>,
    ) -> bool {
        match self.writes.entry(TypeId::of::<C>()) {
            std::collections::hash_map::Entry::Occupied(mut occupied_entry) => {
                let typed_writes = occupied_entry
                    .get_mut()
                    .as_any_mut()
                    .downcast_mut::<TypedKeyOfSetWrites<C, Db, S>>()
                    .expect("type mismatch in KeyOfSetWrites map");

                assert!(
                    Arc::ptr_eq(&typed_writes.original_sieve, original_sieve),
                    "original_sieve mismatch for existing KeyOfSetWrites entry"
                );

                typed_writes.insert(key, element, op)
            }

            std::collections::hash_map::Entry::Vacant(vacant_entry) => {
                let mut typed_writes = TypedKeyOfSetWrites::<C, Db, S> {
                    writes: HashMap::with_hasher(S::default()),
                    original_sieve: original_sieve.clone(),
                };

                let result = typed_writes.insert(key, element, op);

                vacant_entry.insert(Box::new(typed_writes));

                result
            }
        }
    }

    pub(super) fn write_to_db(&self, tx: &<Db as KvDatabase>::WriteBatch) {
        for write_entry in self.writes.values() {
            write_entry.write_to_db(tx);
        }
    }

    pub(super) fn after_commit(&mut self) {
        for write_entry in self.writes.values_mut() {
            write_entry.after_commit();
        }
    }
}

/// Buffer that accumulates write operations before flushing them to the
/// database.
pub struct WriteBuffer<Db: KvDatabase, S: BuildHasher> {
    pub(super) wide_column_writes: WideColumnWrites<Db, S>,
    pub(super) key_of_set_writes: KeyOfSetWrites<Db, S>,
    epoch: usize,
    active: bool,
}

impl<Db: KvDatabase, S: BuildHasher> Drop for WriteBuffer<Db, S> {
    fn drop(&mut self) {
        assert!(!self.active, "WriteBuffer dropped while still active");
    }
}

impl<Db: KvDatabase, S: BuildHasher> std::fmt::Debug for WriteBuffer<Db, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WriteBuffer").finish_non_exhaustive()
    }
}

impl<Db: KvDatabase, S: BuildHasher> WriteBuffer<Db, S> {
    fn write_to_db(&self, tx: &<Db as KvDatabase>::WriteBatch) {
        self.wide_column_writes.write_to_db(tx);
        self.key_of_set_writes.write_to_db(tx);
    }

    fn after_commit(&mut self) {
        self.wide_column_writes.after_commit();
        self.key_of_set_writes.after_commit();
    }
}

impl<Db: KvDatabase, S: BuildHasher> WriteBuffer<Db, S> {
    fn new(epoch: usize, active: bool) -> Self {
        Self {
            wide_column_writes: WideColumnWrites::new(),
            key_of_set_writes: KeyOfSetWrites::new(),
            active,
            epoch,
        }
    }
}

/// An asynchronous background writer that enables write-back strategy, allowing
/// the system to batch and defer write operations to the database.
pub struct BackgroundWriter<Db: KvDatabase, S: BuildHasher> {
    registry: Arc<Registry<Db, S>>,
    write_handles: Vec<thread::JoinHandle<()>>,
    commit_handle: Option<thread::JoinHandle<()>>,
    pool: Arc<Mutex<WriteBufferPool<Db, S>>>,

    db: Arc<Db>,
}

impl<Db: KvDatabase, S: BuildHasher> std::fmt::Debug
    for BackgroundWriter<Db, S>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BackgroundWriter").finish_non_exhaustive()
    }
}

impl<Db: KvDatabase, S: BuildHasher> BackgroundWriter<Db, S> {
    /// Creates a new `BackgroundWriter` with the specified number of worker
    /// threads.
    pub fn new(num_threads: usize, db: Arc<Db>) -> Self {
        let injector = Injector::new();
        let mut workers = Vec::new();
        let mut stealers = Vec::new();

        let (commit_sender, commit_receiver) =
            crossbeam_channel::unbounded::<CommitTask<Db, S>>();

        // Create Local Queues
        for _ in 0..num_threads {
            let w = crossbeam_deque::Worker::new_lifo();
            stealers.push(w.stealer());
            workers.push(w);
        }

        let registry = Arc::new(Registry {
            injector,
            stealers,
            jobs_event_counter: AtomicUsize::new(0),
            sleeping_threads: AtomicUsize::new(0),
            lock: Mutex::new(()),
            condvar: Condvar::new(),
            shutdown: AtomicBool::new(false),
        });

        // Spawn Threads
        let handles = workers
            .into_iter()
            .map(|local| {
                let registry = registry.clone();
                let commit_sender = commit_sender.clone();

                thread::spawn(move || {
                    let worker =
                        WorkerThread { local, registry, sender: commit_sender };
                    worker.run();
                })
            })
            .collect();

        let pool = Arc::new(Mutex::new(WriteBufferPool::new()));

        Self {
            registry,
            write_handles: handles,
            commit_handle: Some({
                let commit_receiver = commit_receiver;
                let pool = pool.clone();

                thread::spawn(move || {
                    Self::commit_worker(&commit_receiver, &pool);
                })
            }),
            pool,
            db,
        }
    }

    /// Creates a new write buffer for accumulating write operations.
    #[must_use]
    pub fn new_write_buffer(&self) -> WriteBuffer<Db, S> {
        let mut pool = self.pool.lock();
        pool.get_buffer()
    }

    /// Submits a write buffer to be processed by the background writer.
    pub fn submit_write_buffer(&self, write_buffer: WriteBuffer<Db, S>) {
        let write_task = WriteTask { write_buffer, db: self.db.clone() };

        // Push to global queue
        self.registry.injector.push(write_task);

        // Increment JEC
        self.registry.jobs_event_counter.fetch_add(1, Ordering::SeqCst);

        // Notify sleeping threads
        if self.registry.sleeping_threads.load(Ordering::SeqCst) > 0 {
            self.registry.condvar.notify_one();
        }
    }

    fn commit_worker(
        receiver: &crossbeam_channel::Receiver<CommitTask<Db, S>>,
        pool: &Mutex<WriteBufferPool<Db, S>>,
    ) {
        let mut expected_epoch = 0;
        let mut pending_commits = BinaryHeap::new();

        while let Ok(task) = receiver.recv() {
            pending_commits.push(task);

            Self::process_pending_commits(
                &mut pending_commits,
                &mut expected_epoch,
                pool,
            );
        }

        // Process remaining commits
        Self::process_pending_commits(
            &mut pending_commits,
            &mut expected_epoch,
            pool,
        );

        // should be empty now
        assert!(pending_commits.is_empty());
    }

    fn process_pending_commits(
        pending_commits: &mut BinaryHeap<CommitTask<Db, S>>,
        expected_epoch: &mut usize,
        pool: &Mutex<WriteBufferPool<Db, S>>,
    ) {
        while let Some(top) = pending_commits.peek() {
            if top.write_task.write_buffer.epoch == *expected_epoch {
                let mut task = pending_commits.pop().unwrap();
                task.write_batch.commit();
                task.write_task.write_buffer.after_commit();
                *expected_epoch += 1;

                // return write buffer to pool
                let mut pool_guard = pool.lock();
                pool_guard.return_buffer(task.write_task.write_buffer);
            } else {
                break;
            }
        }
    }
}

impl<Db: KvDatabase, S: BuildHasher> Drop for BackgroundWriter<Db, S> {
    fn drop(&mut self) {
        // Signal shutdown
        self.registry.shutdown.store(true, Ordering::Relaxed);

        self.registry.condvar.notify_all();

        // Join all threads
        for handle in self.write_handles.drain(..) {
            let _ = handle.join();
        }

        // Join commit thread
        let _ = self.commit_handle.take().unwrap().join();
    }
}

struct WriteTask<Db: KvDatabase, S: BuildHasher> {
    write_buffer: WriteBuffer<Db, S>,
    db: Arc<Db>,
}

struct CommitTask<Db: KvDatabase, S: BuildHasher> {
    write_task: WriteTask<Db, S>,
    write_batch: Db::WriteBatch,
}

impl<Db: KvDatabase, S: BuildHasher> PartialEq for CommitTask<Db, S> {
    fn eq(&self, other: &Self) -> bool {
        self.write_task.write_buffer.epoch
            == other.write_task.write_buffer.epoch
    }
}

impl<Db: KvDatabase, S: BuildHasher> Eq for CommitTask<Db, S> {}

impl<Db: KvDatabase, S: BuildHasher> PartialOrd for CommitTask<Db, S> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<Db: KvDatabase, S: BuildHasher> Ord for CommitTask<Db, S> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse order for min-heap behavior
        other
            .write_task
            .write_buffer
            .epoch
            .cmp(&self.write_task.write_buffer.epoch)
    }
}

pub struct Registry<Db: KvDatabase, S: BuildHasher> {
    injector: Injector<WriteTask<Db, S>>,
    stealers: Vec<Stealer<WriteTask<Db, S>>>,

    jobs_event_counter: AtomicUsize,
    sleeping_threads: AtomicUsize,

    lock: Mutex<()>,
    condvar: Condvar,

    shutdown: AtomicBool,
}

struct WorkerThread<Db: KvDatabase, S: BuildHasher> {
    local: crossbeam_deque::Worker<WriteTask<Db, S>>,
    registry: Arc<Registry<Db, S>>,
    sender: crossbeam_channel::Sender<CommitTask<Db, S>>,
}

impl<Db: KvDatabase, S: BuildHasher> WorkerThread<Db, S> {
    fn run(&self) {
        // Init the backoff strategy (Rayon uses this for spinning)
        let backoff = Backoff::new();

        loop {
            // A. CHECK SHUTDOWN
            if self.registry.shutdown.load(Ordering::Relaxed) {
                // Drain all queues: local, global, and steal from others
                while let Some(task) = self.find_work() {
                    self.process_task(task);
                }
                break;
            }

            // B. SNAPSHOT THE JEC
            // We remember "what time it was" before we started looking for
            // work.
            let last_jec =
                self.registry.jobs_event_counter.load(Ordering::SeqCst);

            // C. SEARCH FOR WORK (Local -> Global -> Steal)
            if let Some(task) = self.find_work() {
                self.process_task(task);
                // Reset backoff because we were useful
                backoff.reset();
                continue;
            }

            // D. THE "SLEEPY" PHASE (Spinning)
            // If we found no work, we don't park immediately. We spin a bit.
            if backoff.is_completed() {
                // E. THE "PARKING" PHASE (Sleeping)
                self.wait_until_work_appears(last_jec);
                backoff.reset();
            } else {
                // Snooze efficiently (CPU instruction 'pause')
                backoff.snooze();
            }
        }
    }

    fn process_task(&self, task: WriteTask<Db, S>) {
        let write_batch = task.db.write_transaction();
        task.write_buffer.write_to_db(&write_batch);

        // send to commit thread
        self.sender
            .send(CommitTask { write_task: task, write_batch })
            .expect("failed to send CommitTask to commit channel");
    }

    fn find_work(&self) -> Option<WriteTask<Db, S>> {
        // 1. Pop Local
        self.local
            .pop()
            .or_else(|| {
                // 2. Pop Global
                std::iter::repeat_with(|| {
                    self.registry.injector.steal_batch_and_pop(&self.local)
                })
                .find(|s| !s.is_retry())
                .and_then(crossbeam_deque::Steal::success)
            })
            .or_else(|| {
                // 3. Steal from others
                self.registry
                    .stealers
                    .iter()
                    .map(crossbeam_deque::Stealer::steal)
                    .find(crossbeam_deque::Steal::is_success)
                    .and_then(crossbeam_deque::Steal::success)
            })
    }

    // This is the core synchronization logic similar to Rayon
    fn wait_until_work_appears(&self, last_jec: usize) {
        // 1. Lock the mutex (required for Condvar)
        let mut guard = self.registry.lock.lock();

        // 2. THE CRITICAL CHECK
        // Check the JEC again *while holding the lock*.
        // If the counter changed between the time we started searching
        // (last_jec) and now, it means work was pushed while we were
        // failing to find it. We should NOT sleep.
        let current_jec =
            self.registry.jobs_event_counter.load(Ordering::SeqCst);

        if current_jec != last_jec {
            // Work arrived! Return immediately to search loop.
            return;
        }

        // 3. Mark ourselves as sleeping
        self.registry.sleeping_threads.fetch_add(1, Ordering::SeqCst);

        // 4. Wait
        // Check shutdown again to avoid hanging forever if shutdown happened
        // during lock
        if !self.registry.shutdown.load(Ordering::Relaxed) {
            self.registry.condvar.wait(&mut guard);
        }

        // 5. We woke up! Mark ourselves as active.
        self.registry.sleeping_threads.fetch_sub(1, Ordering::SeqCst);
    }
}

struct WriteBufferPool<Db: KvDatabase, S: BuildHasher> {
    pool: Vec<WriteBuffer<Db, S>>,
    epoch: usize,
}

impl<Db: KvDatabase, S: BuildHasher> WriteBufferPool<Db, S> {
    #[must_use]
    pub const fn new() -> Self { Self { pool: Vec::new(), epoch: 0 } }

    pub fn get_buffer(&mut self) -> WriteBuffer<Db, S> {
        if let Some(mut buffer) = self.pool.pop() {
            // reset epoch
            buffer.epoch = self.epoch;
            buffer.active = true;

            self.epoch += 1;

            buffer
        } else {
            let epoch = self.epoch;
            self.epoch += 1;

            WriteBuffer::new(epoch, true)
        }
    }

    pub fn return_buffer(&mut self, mut buffer: WriteBuffer<Db, S>) {
        buffer.active = false;
        self.pool.push(buffer);
    }
}
