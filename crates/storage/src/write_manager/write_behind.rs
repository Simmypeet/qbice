//! Write-behind caching implementation.
//!
//! This module provides [`WriteBehind`], an asynchronous background writer
//! that enables write-back caching. Cache updates are fast (in-memory only)
//! while persistence happens asynchronously in background worker threads.
//!
//! # Key Components
//!
//! - [`WriteBehind`]: The main background writer managing worker threads
//! - [`WriteTransaction`]: A buffer for accumulating write operations
//! - [`Epoch`]: Monotonic ordering identifier for write transactions
//!
//! # Write Pipeline
//!
//! 1. Application creates a [`WriteTransaction`] via
//!    [`WriteBehind::new_write_transaction()`]
//! 2. Writes accumulate in the transaction (fast, in-memory)
//! 3. Transaction is submitted via [`WriteBehind::submit_write_transaction()`]
//! 4. Worker threads process transactions asynchronously
//! 5. Commit thread ensures epoch ordering and applies writes to database

use std::{
    any::{Any, TypeId},
    cell::RefCell,
    collections::{BinaryHeap, HashMap},
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    thread,
};

use crossbeam_deque::{Injector, Stealer};
use crossbeam_utils::Backoff;
use fxhash::FxBuildHasher;
use parking_lot::{Condvar, Mutex};
use thread_local::ThreadLocal;

use crate::{
    kv_database::{
        KeyOfSetColumn, KvDatabase, WideColumn, WideColumnValue,
        WriteBatch as _,
    },
    write_manager, write_transaction,
};

pub(crate) trait WideColumnCache<
    K: WideColumn,
    W: WideColumnValue<K>,
    Db: KvDatabase,
>: Send + Sync
{
    fn flush<'s: 'x, 'i: 'x, 'x>(
        &'s self,
        epoch: Epoch,
        keys: &'i mut (dyn Iterator<Item = K::Key> + Send),
    ) -> Pin<Box<dyn std::future::Future<Output = ()> + Send + 'x>>;
}

pub(crate) trait KeyOfSetCache<K: KeyOfSetColumn, Db: KvDatabase>:
    Send + Sync
{
    fn flush(
        &self,
        epoch: Epoch,
        keys: &mut (dyn Iterator<Item = K::Key> + Send),
    );
}

/// A monotonically increasing identifier for write transactions.
///
/// Epochs ensure that write transactions are committed in the order they
/// were submitted. Each new transaction receives the next epoch number,
/// and the commit thread processes transactions strictly in epoch order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Epoch(pub u64);

struct TypedWideColumnWrites<
    C: WideColumn,
    V: WideColumnValue<C>,
    Db: KvDatabase,
> {
    /// Map of keys to their corresponding values to write.
    ///
    /// `None` indicates a deletion for that key. `Some(value)` indicates
    /// an insertion or update.
    writes: HashMap<C::Key, Option<V>, FxBuildHasher>,
    original_cache: Arc<dyn WideColumnCache<C, V, Db>>,
}

enum Task<'s> {
    Sync,
    Async(Pin<Box<dyn std::future::Future<Output = ()> + Send + 's>>),
}

trait WriteEntry<Db: KvDatabase>: Any + Send + Sync + 'static {
    fn write_to_db(&self, tx: &Db::WriteBatch);
    fn after_commit(&mut self, epoch: Epoch) -> Task<'_>;
    fn as_any_mut(&mut self) -> &mut (dyn Any + Send + Sync);
}

impl<C: WideColumn, V: WideColumnValue<C>, Db: KvDatabase> WriteEntry<Db>
    for TypedWideColumnWrites<C, V, Db>
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

    fn after_commit(&mut self, epoch: Epoch) -> Task<'_> {
        Task::Async(Box::pin(async move {
            let mut drained = self.writes.drain().map(|x| x.0);
            let original_cache = self.original_cache.clone();

            original_cache.flush(epoch, &mut drained).await;
        }))
    }

    fn as_any_mut(&mut self) -> &mut (dyn Any + Send + Sync) { self }
}

impl<C: WideColumn, V: WideColumnValue<C>, Db: KvDatabase>
    TypedWideColumnWrites<C, V, Db>
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
pub(super) struct WideColumnWrites<Db: KvDatabase> {
    writes: HashMap<WideColumnWritesID, Box<dyn WriteEntry<Db>>, FxBuildHasher>,
}

impl<Db: KvDatabase> WideColumnWrites<Db> {
    fn new() -> Self { Self { writes: HashMap::default() } }

    fn put<C: WideColumn, V: WideColumnValue<C>>(
        &mut self,
        key: C::Key,
        value: Option<V>,
        original_cache: Arc<dyn WideColumnCache<C, V, Db>>,
    ) -> bool {
        let id = WideColumnWritesID::of::<C, V>();

        match self.writes.entry(id) {
            std::collections::hash_map::Entry::Occupied(mut occupied_entry) => {
                let typed_writes = occupied_entry
                    .get_mut()
                    .as_any_mut()
                    .downcast_mut::<TypedWideColumnWrites<C, V, Db>>()
                    .expect("type mismatch in WideColumnWrites map");

                assert!(
                    Arc::ptr_eq(&typed_writes.original_cache, &original_cache),
                    "original_cache mismatch for existing WideColumnWrites \
                     entry"
                );

                typed_writes.insert(key, value)
            }

            std::collections::hash_map::Entry::Vacant(vacant_entry) => {
                let mut typed_writes = TypedWideColumnWrites::<C, V, Db> {
                    writes: HashMap::default(),
                    original_cache,
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

    pub(super) async fn after_commit(&mut self, epoch: Epoch) {
        for write_entry in self.writes.values_mut() {
            match write_entry.after_commit(epoch) {
                Task::Sync => {}
                Task::Async(fut) => {
                    fut.await;
                }
            }
        }
    }
}

/// The type of operation to perform on a key-of-set element.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Operation {
    /// Insert an element into the set.
    Insert,
    /// Delete an element from the set.
    Delete,
}

struct TypedKeyOfSetWrites<C: KeyOfSetColumn, Db: KvDatabase> {
    writes: HashMap<C::Key, HashMap<C::Element, Operation>, FxBuildHasher>,
    original_cache: Arc<dyn KeyOfSetCache<C, Db>>,
}

impl<C: KeyOfSetColumn, Db: KvDatabase> WriteEntry<Db>
    for TypedKeyOfSetWrites<C, Db>
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

    fn after_commit(&mut self, epoch: Epoch) -> Task<'_> {
        let mut keys = self.writes.drain().map(|x| x.0);
        self.original_cache.flush(epoch, &mut keys);

        Task::Sync
    }

    fn as_any_mut(&mut self) -> &mut (dyn Any + Send + Sync) { self }
}

impl<C: KeyOfSetColumn, Db: KvDatabase> TypedKeyOfSetWrites<C, Db> {
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
                let mut element_map = HashMap::default();
                element_map.insert(element, op);
                vacant_entry.insert(element_map);
                true
            }
        }
    }
}

#[allow(clippy::type_complexity)]
pub(super) struct KeyOfSetWrites<Db: KvDatabase> {
    writes: HashMap<TypeId, Box<dyn WriteEntry<Db>>, FxBuildHasher>,
}

impl<Db: KvDatabase> KeyOfSetWrites<Db> {
    fn new() -> Self { Self { writes: HashMap::default() } }

    fn put<C: KeyOfSetColumn>(
        &mut self,
        key: C::Key,
        element: C::Element,
        op: Operation,
        original_cache: Arc<dyn KeyOfSetCache<C, Db>>,
    ) -> bool {
        match self.writes.entry(TypeId::of::<C>()) {
            std::collections::hash_map::Entry::Occupied(mut occupied_entry) => {
                let typed_writes = occupied_entry
                    .get_mut()
                    .as_any_mut()
                    .downcast_mut::<TypedKeyOfSetWrites<C, Db>>()
                    .expect("type mismatch in KeyOfSetWrites map");

                assert!(
                    Arc::ptr_eq(&typed_writes.original_cache, &original_cache),
                    "original_cache mismatch for existing KeyOfSetWrites entry"
                );

                typed_writes.insert(key, element, op)
            }

            std::collections::hash_map::Entry::Vacant(vacant_entry) => {
                let mut typed_writes = TypedKeyOfSetWrites::<C, Db> {
                    writes: HashMap::default(),
                    original_cache,
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

    pub(super) async fn after_commit(&mut self, epoch: Epoch) {
        for write_entry in self.writes.values_mut() {
            match write_entry.after_commit(epoch) {
                Task::Sync => {}
                Task::Async(fut) => {
                    fut.await;
                }
            }
        }
    }
}

/// A buffer that accumulates write operations for batch processing.
///
/// `WriteBuffer` collects multiple cache write operations (both wide column
/// and key-of-set writes) in memory before they are asynchronously flushed to
/// the database. This enables write-back caching with significantly reduced
/// database write amplification.
///
/// # Architecture
///
/// The buffer maintains two internal collections:
/// - **Wide column writes**: Map of `(column, discriminant, key)` to values
/// - **Key-of-set writes**: Map of `(column, key, element)` to operations
///
/// # Lifecycle
///
/// 1. **Creation**: Obtained from [`WriteBehind::new_write_transaction()`]
/// 2. **Accumulation**: Writes are added via cache `put`, `insert_set_element`,
///    `remove_set_element` methods
/// 3. **Submission**: Buffer is submitted to [`WriteBehind`] for async flushing
/// 4. **Processing**: Background thread writes to database and commits
/// 5. **Recycling**: Buffer is returned to pool for reuse
///
/// # Epoch Ordering
///
/// Each buffer has an epoch number ensuring writes are committed in order:
/// - Buffers are processed in epoch order
/// - Later epochs wait for earlier epochs to commit
/// - Ensures write consistency and prevents reordering
///
/// # Durability
///
/// **Important**: Writes in the buffer are **not durable** until:
/// 1. Buffer is submitted via `submit_write_transaction()`
/// 2. Background worker flushes to database
/// 3. Database commit succeeds
///
/// **Data Loss Risk**: If the process crashes before flush, all buffered
/// writes are lost.
///
/// # Active State
///
/// The buffer has an "active" flag:
/// - Set when created from pool
/// - Cleared when returned to pool
/// - **Panics** if dropped while still active (indicates programming error)
///
/// # Example
///
/// ```ignore
/// let writer = WriteBehind::new(4, db.clone());
/// let cache = engine.new_single_map::<Column, Value>();
///
/// // Create buffer
/// let mut buffer = writer.new_write_transaction();
///
/// // Accumulate writes
/// cache.insert(key1, value1, &mut buffer).await;
/// cache.insert(key2, value2, &mut buffer).await;
///
/// // Submit for async persistence
/// writer.submit_write_transaction(buffer);
/// // Writes will be flushed to database in background
/// ```
pub struct WriteTransaction<Db: KvDatabase> {
    pub(super) wide_column_writes: WideColumnWrites<Db>,
    pub(super) key_of_set_writes: KeyOfSetWrites<Db>,
    write_batch: Option<Db::WriteBatch>,
    epoch: Epoch,
    active: bool,
}

impl<Db: KvDatabase> write_transaction::WriteTransaction
    for WriteTransaction<Db>
{
}

impl<Db: KvDatabase> Drop for WriteTransaction<Db> {
    fn drop(&mut self) {
        assert!(!self.active, "WriteBuffer dropped while still active");
    }
}

impl<Db: KvDatabase> std::fmt::Debug for WriteTransaction<Db> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WriteBuffer").finish_non_exhaustive()
    }
}

impl<Db: KvDatabase> WriteTransaction<Db> {
    fn write_to_db(&self, tx: &<Db as KvDatabase>::WriteBatch) {
        self.wide_column_writes.write_to_db(tx);
        self.key_of_set_writes.write_to_db(tx);
    }

    async fn after_commit(&mut self, epoch: Epoch) {
        self.wide_column_writes.after_commit(epoch).await;
        self.key_of_set_writes.after_commit(epoch).await;
    }

    pub(crate) fn put_wide_column<C: WideColumn, V: WideColumnValue<C>>(
        &mut self,
        key: C::Key,
        value: Option<V>,
        original_cache: Arc<dyn WideColumnCache<C, V, Db>>,
    ) -> bool {
        self.wide_column_writes.put::<C, V>(key, value, original_cache)
    }

    pub(crate) fn put_set<C: KeyOfSetColumn>(
        &mut self,
        key: C::Key,
        element: C::Element,
        op: Operation,
        original_cache: Arc<dyn KeyOfSetCache<C, Db>>,
    ) -> bool {
        self.key_of_set_writes.put::<C>(key, element, op, original_cache)
    }

    #[must_use]
    pub(crate) const fn epoch(&self) -> Epoch { self.epoch }
}

impl<Db: KvDatabase> WriteTransaction<Db> {
    fn new(epoch: Epoch, active: bool, write_batch: Db::WriteBatch) -> Self {
        Self {
            wide_column_writes: WideColumnWrites::new(),
            key_of_set_writes: KeyOfSetWrites::new(),
            write_batch: Some(write_batch),
            active,
            epoch,
        }
    }
}

/// An asynchronous background writer enabling write-back caching strategy.
///
/// `WriteBehind` manages a pool of worker threads that process write
/// transactions asynchronously. This enables high-performance write-back
/// caching where cache updates are fast (in-memory only) and persistence
/// happens in the background.
///
/// # Architecture
///
/// ```text
///                  WriteTransaction
///                          |
///                          v
///                    Work Stealing
///                       Queue
///                         |
///         +---------------+---------------+
///         |               |               |
///      Worker 1        Worker 2      Worker N
///         |               |               |
///         +-------+-------+-------+-------+
///                 |               |
///           WriteBatch      WriteBatch
///                 |               |
///                 v               v
///                   Commit Thread
///                         |
///                         v
///                      Database
/// ```
///
///
/// # Write Processing Pipeline
///
/// 1. **Buffer Creation**: Application obtains buffer from pool
/// 2. **Accumulation**: Writes accumulate in buffer (fast, in-memory)
/// 3. **Submission**: Buffer submitted to global work queue
/// 4. **Worker Processing**: Worker picks up buffer, creates write batch
/// 5. **Commit Ordering**: Commit thread applies batches in epoch order
/// 6. **Notification**: Cache is notified to decrement pending write counts
/// 7. **Buffer Recycling**: Buffer returned to pool
///
/// # Epoch-Based Ordering
///
/// Buffers are assigned monotonically increasing epoch numbers:
/// - Ensures writes are committed in submission order
/// - Later epochs wait for earlier epochs to commit
/// - Critical for maintaining cache coherency
///
/// # Example
///
/// ```ignore
/// use qbice_storage::{
///     storage_engine::{StorageEngine, db_backed::DbBacked},
///     write_manager::WriteManager,
/// };
///
/// // Create storage engine with database backend
/// let engine = DbBacked::new(database, config);
/// let write_manager = engine.new_write_manager();
/// let map = engine.new_single_map::<MyColumn, MyValue>();
///
/// // Application loop
/// loop {
///     let mut tx = write_manager.new_write_transaction();
///
///     // Accumulate many writes (fast, in-memory)
///     for (key, value) in updates {
///         map.insert(key, value, &mut tx).await;
///     }
///
///     // Submit entire batch for async persistence
///     write_manager.submit_write_transaction(tx);
///     // Control returns immediately; writes happen in background
/// }
///
/// // On shutdown, drop writer to gracefully drain queues
/// drop(write_manager);
/// ```
pub struct WriteBehind<Db: KvDatabase> {
    registry: Arc<Registry<Db>>,
    write_handles: Vec<thread::JoinHandle<()>>,
    commit_handle: Option<thread::JoinHandle<()>>,
    pool: Arc<WriteBufferPool<Db>>,
}

impl<Db: KvDatabase> write_manager::WriteManager for WriteBehind<Db> {
    type WriteTransaction = WriteTransaction<Db>;

    fn new_write_transaction(&self) -> Self::WriteTransaction {
        Self::new_write_transaction(self)
    }

    fn submit_write_transaction(
        &self,
        write_transaction: Self::WriteTransaction,
    ) {
        Self::submit_write_transaction(self, write_transaction);
    }
}

impl<Db: KvDatabase> std::fmt::Debug for WriteBehind<Db> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BackgroundWriter").finish_non_exhaustive()
    }
}

impl<Db: KvDatabase> WriteBehind<Db> {
    /// Creates a new background writer with the specified number of worker
    /// threads.
    ///
    /// # Parameters
    ///
    /// * `num_threads` - The number of worker threads for processing write
    ///   buffers. More threads increase throughput but also increase overhead.
    ///
    ///   **Note**: Returns diminish beyond database I/O capacity. If database
    ///   is the bottleneck, more threads won't help.
    ///
    /// * `db` - The database instance to write to. Must be wrapped in `Arc` for
    ///   sharing across worker threads.
    ///
    /// # Thread Spawning
    ///
    /// This method spawns `num_threads + 1` threads:
    /// - `num_threads` worker threads (named "`bg_writer_0`", "`bg_writer_1`",
    ///   ...)
    /// - 1 commit thread (named "`bg_writer_commit`")
    ///
    /// All threads start immediately and begin waiting for work.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use std::sync::Arc;
    ///
    /// // Create writer with 4 workers
    /// let writer = BackgroundWriter::new(4, Arc::new(database));
    ///
    /// // Use writer...
    ///
    /// // Shutdown gracefully on drop
    /// drop(writer);
    /// ```
    pub fn new(num_threads: usize, db: Db) -> Self {
        let injector = Injector::new();
        let mut workers = Vec::new();
        let mut stealers = Vec::new();

        let (commit_sender, commit_receiver) =
            crossbeam_channel::unbounded::<CommitTask<Db>>();

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
            shutdown: Mutex::new(false),
            condvar: Condvar::new(),
        });

        // Spawn Threads
        let handles = workers
            .into_iter()
            .enumerate()
            .map(|(i, local)| {
                let registry = registry.clone();
                let commit_sender = commit_sender.clone();

                thread::Builder::new()
                    .name(format!("bg_writer_{i}"))
                    .spawn(move || {
                        let worker = WorkerThread {
                            local,
                            registry,
                            sender: commit_sender,
                        };
                        worker.run();
                    })
                    .unwrap()
            })
            .collect();

        let pool = Arc::new(WriteBufferPool::new(db));

        Self {
            registry,
            write_handles: handles,
            commit_handle: Some({
                let commit_receiver = commit_receiver;
                let pool = pool.clone();

                thread::Builder::new()
                    .name("bg_writer_commit".to_string())
                    .spawn(move || {
                        // NOTE: we create a dedicated single-threaded runtime
                        // here because we don't want comit tasks to be
                        // interfered with by other async tasks that might be
                        // spawned in the same thread.
                        let runtime =
                            tokio::runtime::Builder::new_current_thread()
                                .enable_all()
                                .build()
                                .unwrap();

                        runtime.block_on(async {
                            Self::commit_worker(&commit_receiver, &pool).await;
                        });
                    })
                    .unwrap()
            }),
            pool,
        }
    }

    /// Creates a new write buffer for accumulating write operations.
    #[must_use]
    pub fn new_write_transaction(&self) -> WriteTransaction<Db> {
        self.pool.get_buffer()
    }

    /// Submits a write buffer to be processed by the background writer.
    pub fn submit_write_transaction(&self, write_buffer: WriteTransaction<Db>) {
        let write_task = WriteTask { write_buffer };

        // Push to global queue
        self.registry.injector.push(write_task);

        // Increment JEC
        self.registry.jobs_event_counter.fetch_add(1, Ordering::SeqCst);

        // Notify sleeping threads
        if self.registry.sleeping_threads.load(Ordering::SeqCst) > 0 {
            self.registry.condvar.notify_one();
        }
    }

    async fn commit_worker(
        receiver: &crossbeam_channel::Receiver<CommitTask<Db>>,
        pool: &WriteBufferPool<Db>,
    ) {
        let mut expected_epoch = Epoch(0);
        let mut pending_commits = BinaryHeap::new();

        while let Ok(task) = receiver.recv() {
            pending_commits.push(task);

            Self::process_pending_commits(
                &mut pending_commits,
                &mut expected_epoch,
                pool,
            )
            .await;
        }

        // Process remaining commits
        Self::process_pending_commits(
            &mut pending_commits,
            &mut expected_epoch,
            pool,
        )
        .await;

        // should be empty now
        assert!(pending_commits.is_empty());
    }

    async fn process_pending_commits(
        pending_commits: &mut BinaryHeap<CommitTask<Db>>,
        expected_epoch: &mut Epoch,
        pool: &WriteBufferPool<Db>,
    ) {
        while let Some(top) = pending_commits.peek() {
            if top.write_task.write_buffer.epoch == *expected_epoch {
                let mut task = pending_commits.pop().unwrap();

                task.write_batch.commit();
                task.write_task
                    .write_buffer
                    .after_commit(task.write_task.write_buffer.epoch)
                    .await;

                expected_epoch.0 += 1;

                // return write buffer to pool
                pool.return_buffer(task.write_task.write_buffer);
            } else {
                break;
            }
        }
    }
}

impl<Db: KvDatabase> Drop for WriteBehind<Db> {
    fn drop(&mut self) {
        // Signal shutdown
        *self.registry.shutdown.lock() = true;

        self.registry.condvar.notify_all();

        // Join all threads
        for handle in self.write_handles.drain(..) {
            let _ = handle.join();
        }

        // Join commit thread
        let _ = self.commit_handle.take().unwrap().join();
    }
}

struct WriteTask<Db: KvDatabase> {
    write_buffer: WriteTransaction<Db>,
}

struct CommitTask<Db: KvDatabase> {
    write_task: WriteTask<Db>,
    write_batch: Db::WriteBatch,
}

impl<Db: KvDatabase> PartialEq for CommitTask<Db> {
    fn eq(&self, other: &Self) -> bool {
        self.write_task.write_buffer.epoch
            == other.write_task.write_buffer.epoch
    }
}

impl<Db: KvDatabase> Eq for CommitTask<Db> {}

impl<Db: KvDatabase> PartialOrd for CommitTask<Db> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<Db: KvDatabase> Ord for CommitTask<Db> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse order for min-heap behavior
        other
            .write_task
            .write_buffer
            .epoch
            .cmp(&self.write_task.write_buffer.epoch)
    }
}

#[derive(Debug)]
struct Registry<Db: KvDatabase> {
    injector: Injector<WriteTask<Db>>,
    stealers: Vec<Stealer<WriteTask<Db>>>,

    jobs_event_counter: AtomicUsize,
    sleeping_threads: AtomicUsize,

    shutdown: Mutex<bool>,
    condvar: Condvar,
}

struct WorkerThread<Db: KvDatabase> {
    local: crossbeam_deque::Worker<WriteTask<Db>>,
    registry: Arc<Registry<Db>>,
    sender: crossbeam_channel::Sender<CommitTask<Db>>,
}

impl<Db: KvDatabase> WorkerThread<Db> {
    fn run(&self) {
        // Init the backoff strategy (Rayon uses this for spinning)
        let backoff = Backoff::new();

        loop {
            // A. CHECK SHUTDOWN
            if *self.registry.shutdown.lock() {
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

    fn process_task(&self, mut task: WriteTask<Db>) {
        let write_batch =
            task.write_buffer.write_batch.take().expect(
                "Write batch should've been set when retrieved from pool",
            );

        task.write_buffer.write_to_db(&write_batch);

        // send to commit thread
        self.sender
            .send(CommitTask { write_task: task, write_batch })
            .expect("failed to send CommitTask to commit channel");
    }

    fn find_work(&self) -> Option<WriteTask<Db>> {
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
        let mut shutdown = self.registry.shutdown.lock();

        if *shutdown {
            // Shutdown signaled while acquiring lock
            return;
        }

        self.registry.sleeping_threads.fetch_add(1, Ordering::SeqCst);

        // 4. Wait
        // Check shutdown again to avoid hanging forever if shutdown happened
        // during lock
        self.registry.condvar.wait(&mut shutdown);

        // 5. We woke up! Mark ourselves as active.
        self.registry.sleeping_threads.fetch_sub(1, Ordering::SeqCst);
    }
}

struct WriteBufferPool<Db: KvDatabase> {
    pool: ThreadLocal<RefCell<Vec<WriteTransaction<Db>>>>,
    db: Db,
    epoch: AtomicU64,
}

impl<Db: KvDatabase> WriteBufferPool<Db> {
    #[must_use]
    pub const fn new(db: Db) -> Self {
        Self { pool: ThreadLocal::new(), db, epoch: AtomicU64::new(0) }
    }

    pub fn get_buffer(&self) -> WriteTransaction<Db> {
        let curr_epoch = self.epoch.fetch_add(1, Ordering::SeqCst);
        let curr_pool = self.pool.get_or(|| RefCell::new(Vec::new()));

        let mut pool = curr_pool.borrow_mut();

        pool.pop().map_or_else(
            || {
                WriteTransaction::new(
                    Epoch(curr_epoch),
                    true,
                    self.db.write_batch(),
                )
            },
            |mut buffer| {
                buffer.epoch = Epoch(curr_epoch);
                buffer.active = true;
                buffer.write_batch = Some(self.db.write_batch());

                buffer
            },
        )
    }

    pub fn return_buffer(&self, mut buffer: WriteTransaction<Db>) {
        buffer.active = false;

        self.pool.get_or(|| RefCell::new(Vec::new())).borrow_mut().push(buffer);
    }
}
