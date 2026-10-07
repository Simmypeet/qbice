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
//! 4. Worker threads serialize transactions asynchronously
//! 5. Batch thread puts the transactions back in epoch order and groups them
//!    into database write batches
//! 6. Commit thread applies the write batches to the database, one at a time

use std::{
    any::{Any, TypeId},
    collections::{BinaryHeap, HashMap},
    ops::Not,
    sync::{
        Arc, Weak,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread,
};

use fxhash::FxBuildHasher;

use crate::{
    kv_database::{
        KeyOfSetColumn, KvDatabase, SerializationBuffer as _, WideColumn,
        WideColumnValue, WriteBatch as _,
    },
    write_batch, write_manager,
};

pub(crate) trait WideColumnCache<
    K: WideColumn,
    W: WideColumnValue<K>,
    Db: KvDatabase,
>: Send + Sync
{
    fn flush(
        &self,
        epoch: Epoch,
        keys: &mut (dyn Iterator<Item = K::Key> + Send),
    );
}

/// What a committed write batch has done to the sets of a key-of-set column:
/// for every key it has written to, the last operation on each element.
pub(crate) type CommittedSetWrites<'a, K> = dyn Iterator<
        Item = (
            <K as KeyOfSetColumn>::Key,
            HashMap<<K as KeyOfSetColumn>::Element, Operation>,
        ),
    > + Send
    + 'a;

pub(crate) trait KeyOfSetCache<K: KeyOfSetColumn, Db: KvDatabase>:
    Send + Sync
{
    /// Called once the write batch of `epoch` has been committed to the
    /// database.
    fn flush(&self, epoch: Epoch, writes: &mut CommittedSetWrites<'_, K>);
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
    original_cache: Weak<dyn WideColumnCache<C, V, Db>>,
}

trait WriteEntry<Db: KvDatabase>: Any + Send + Sync + 'static {
    fn write_to_db(&self, tx: &mut Db::SerializationBuffer);
    fn after_commit(&mut self, epoch: Epoch);
    fn as_any_mut(&mut self) -> &mut (dyn Any + Send + Sync);
}

impl<C: WideColumn, V: WideColumnValue<C>, Db: KvDatabase> WriteEntry<Db>
    for TypedWideColumnWrites<C, V, Db>
{
    fn write_to_db(&self, tx: &mut <Db as KvDatabase>::SerializationBuffer) {
        for (key, value_opt) in &self.writes {
            match value_opt {
                Some(value) => tx.put(key, value),
                None => {
                    tx.delete::<C, V>(key);
                }
            }
        }
    }

    fn after_commit(&mut self, epoch: Epoch) {
        let mut drained = self.writes.drain().map(|x| x.0);

        let Some(original_cache) = self.original_cache.clone().upgrade() else {
            return;
        };

        original_cache.flush(epoch, &mut drained);
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
        original_cache: Weak<dyn WideColumnCache<C, V, Db>>,
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
                    Weak::ptr_eq(&typed_writes.original_cache, &original_cache),
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

    pub(super) fn write_to_db(
        &self,
        tx: &mut <Db as KvDatabase>::SerializationBuffer,
    ) {
        for write_entry in self.writes.values() {
            write_entry.write_to_db(tx);
        }
    }

    pub(super) fn after_commit(&mut self, epoch: Epoch) {
        for write_entry in self.writes.values_mut() {
            write_entry.after_commit(epoch);
        }
    }
}

/// The type of operation to perform on a key-of-set element.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Operation {
    /// Insert an element into the set.
    Insert,
    /// Delete an element from the set.
    Remove,
}

struct TypedKeyOfSetWrites<C: KeyOfSetColumn, Db: KvDatabase> {
    writes: HashMap<C::Key, HashMap<C::Element, Operation>, FxBuildHasher>,
    original_cache: Weak<dyn KeyOfSetCache<C, Db>>,
}

impl<C: KeyOfSetColumn, Db: KvDatabase> WriteEntry<Db>
    for TypedKeyOfSetWrites<C, Db>
{
    fn write_to_db(&self, tx: &mut <Db as KvDatabase>::SerializationBuffer) {
        for (key, element_map) in &self.writes {
            for (element, op) in element_map {
                match op {
                    Operation::Insert => {
                        tx.insert_member::<C>(key, element);
                    }
                    Operation::Remove => {
                        tx.delete_member::<C>(key, element);
                    }
                }
            }
        }
    }

    fn after_commit(&mut self, epoch: Epoch) {
        let mut writes = self.writes.drain();

        let Some(original_cache) = self.original_cache.clone().upgrade() else {
            return;
        };

        original_cache.flush(epoch, &mut writes);
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
        original_cache: Weak<dyn KeyOfSetCache<C, Db>>,
    ) -> bool {
        match self.writes.entry(TypeId::of::<C>()) {
            std::collections::hash_map::Entry::Occupied(mut occupied_entry) => {
                let typed_writes = occupied_entry
                    .get_mut()
                    .as_any_mut()
                    .downcast_mut::<TypedKeyOfSetWrites<C, Db>>()
                    .expect("type mismatch in KeyOfSetWrites map");

                assert!(
                    Weak::ptr_eq(&typed_writes.original_cache, &original_cache),
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

    pub(super) fn write_to_db(
        &self,
        tx: &mut <Db as KvDatabase>::SerializationBuffer,
    ) {
        for write_entry in self.writes.values() {
            write_entry.write_to_db(tx);
        }
    }

    pub(super) fn after_commit(&mut self, epoch: Epoch) {
        for write_entry in self.writes.values_mut() {
            write_entry.after_commit(epoch);
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
/// 5. **Release**: Buffer is dropped once the caches have been notified
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
/// - Set when created
/// - Cleared once the background writer has finished with it
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
pub struct WriteBatch<Db: KvDatabase> {
    pub(super) wide_column_writes: WideColumnWrites<Db>,
    pub(super) key_of_set_writes: KeyOfSetWrites<Db>,
    epoch: Epoch,
    active: bool,
}

impl<Db: KvDatabase> write_batch::WriteBatch for WriteBatch<Db> {}

impl<Db: KvDatabase> Drop for WriteBatch<Db> {
    fn drop(&mut self) {
        assert!(!self.active, "WriteBuffer dropped while still active");
    }
}

impl<Db: KvDatabase> std::fmt::Debug for WriteBatch<Db> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WriteBuffer").finish_non_exhaustive()
    }
}

impl<Db: KvDatabase> WriteBatch<Db> {
    fn write_to_db(&self, tx: &mut <Db as KvDatabase>::SerializationBuffer) {
        self.wide_column_writes.write_to_db(tx);
        self.key_of_set_writes.write_to_db(tx);
    }

    fn after_commit(&mut self, epoch: Epoch) {
        self.wide_column_writes.after_commit(epoch);
        self.key_of_set_writes.after_commit(epoch);
    }

    pub(crate) fn put_wide_column<C: WideColumn, V: WideColumnValue<C>>(
        &mut self,
        key: C::Key,
        value: Option<V>,
        original_cache: Weak<dyn WideColumnCache<C, V, Db>>,
    ) -> bool {
        self.wide_column_writes.put::<C, V>(key, value, original_cache)
    }

    pub(crate) fn put_set<C: KeyOfSetColumn>(
        &mut self,
        key: C::Key,
        element: C::Element,
        op: Operation,
        original_cache: Weak<dyn KeyOfSetCache<C, Db>>,
    ) -> bool {
        self.key_of_set_writes.put::<C>(key, element, op, original_cache)
    }

    #[must_use]
    pub(crate) const fn epoch(&self) -> Epoch { self.epoch }
}

impl<Db: KvDatabase> WriteBatch<Db> {
    fn new(epoch: Epoch, active: bool) -> Self {
        Self {
            wide_column_writes: WideColumnWrites::new(),
            key_of_set_writes: KeyOfSetWrites::new(),
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
///                   Batch Thread
///                         |
///                         v
///                   Commit Thread
///                         |
///                         v
///                      Database
/// ```
///
///
/// # Write Processing Pipeline
///
/// 1. **Buffer Creation**: Application creates a buffer
/// 2. **Accumulation**: Writes accumulate in buffer (fast, in-memory)
/// 3. **Submission**: Buffer submitted to global work queue
/// 4. **Worker Processing**: Worker picks up buffer, serializes its writes
/// 5. **Batching**: Batch thread collects the buffers in epoch order into a
///    database write batch and prepares it
/// 6. **Commit**: Commit thread applies the write batches in that order, while
///    the batch thread is already preparing the next one
/// 7. **Notification**: Cache is notified to decrement pending write counts
/// 8. **Buffer Release**: Buffer is dropped
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
    batch_handle: Option<thread::JoinHandle<()>>,
    commit_handle: Option<thread::JoinHandle<()>>,

    serialize_sender: Option<crossbeam_channel::Sender<SerializeTask<Db>>>,
    serialize_handles: Vec<thread::JoinHandle<()>>,

    after_commit_handle: Option<thread::JoinHandle<()>>,
    shutting_down: Arc<AtomicBool>,

    epoch: AtomicU64,
}

impl<Db: KvDatabase> write_manager::WriteManager for WriteBehind<Db> {
    type WriteBatch = WriteBatch<Db>;

    fn new_write_batch(&self) -> Self::WriteBatch {
        Self::new_write_batch(self)
    }

    fn submit_write_batch(&self, write_transaction: Self::WriteBatch) {
        Self::submit_write_batch(self, write_transaction);
    }
}

impl<Db: KvDatabase> std::fmt::Debug for WriteBehind<Db> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BackgroundWriter").finish_non_exhaustive()
    }
}

struct CurrentBatch<Db: KvDatabase> {
    processed_logical_batch: Vec<WriteBatch<Db>>,
    db_write_batch: Db::WriteBatch,
    expected_epoch: Epoch,
}

impl<Db: KvDatabase> CurrentBatch<Db> {
    /// Hands the physical batch over to the commit thread and starts a new
    /// one.
    pub fn flush(
        &mut self,
        db: &Db,
        commit_sender: &crossbeam_channel::Sender<CommitTask<Db>>,
    ) {
        let mut db_write_batch =
            std::mem::replace(&mut self.db_write_batch, db.write_batch());
        let logical_batches = std::mem::take(&mut self.processed_logical_batch);

        // done on this thread so that the commit thread only has to write
        db_write_batch.prepare();

        commit_sender
            .send(CommitTask { db_write_batch, logical_batches })
            .unwrap();
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
    /// This method spawns `num_threads + 3` threads:
    /// - `num_threads` worker threads (named "`bg_writer_ser_0`",
    ///   "`bg_writer_ser_1`", ...)
    /// - 1 batch thread (named "`bg_writer_batch`")
    /// - 1 commit thread (named "`bg_writer_commit`")
    /// - 1 after-commit thread (named "`bg_writer_after_commit`")
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
    pub fn new(db: &Db, serialize_worker_count: usize) -> Self {
        let (batch_sender, batch_receiver) =
            crossbeam_channel::unbounded::<WriteTask<Db>>();
        // The batch thread prepares one batch ahead of the commit thread and
        // then waits for it.
        let (commit_sender, commit_receiver) =
            crossbeam_channel::bounded::<CommitTask<Db>>(1);
        let (serialize_sender, serialize_receiver) =
            crossbeam_channel::unbounded::<SerializeTask<Db>>();
        let (after_commit_sender, after_commit_receiver) =
            crossbeam_channel::unbounded::<AfterCommitTask<Db>>();

        let shutting_down = Arc::new(AtomicBool::new(false));

        Self {
            batch_handle: Some({
                let db = db.clone();

                thread::Builder::new()
                    .name("bg_writer_batch".to_string())
                    .spawn(move || {
                        Self::batch_worker(&batch_receiver, commit_sender, &db);
                    })
                    .unwrap()
            }),

            commit_handle: Some({
                let shutting_down = shutting_down.clone();

                thread::Builder::new()
                    .name("bg_writer_commit".to_string())
                    .spawn(move || {
                        Self::commit_worker(
                            &commit_receiver,
                            after_commit_sender,
                            &shutting_down,
                        );
                    })
                    .unwrap()
            }),

            serialize_sender: Some(serialize_sender),
            serialize_handles: (0..serialize_worker_count)
                .map(|i| {
                    let serialize_receiver = serialize_receiver.clone();
                    let batch_sender = batch_sender.clone();
                    let db = db.clone();

                    thread::Builder::new()
                        .name(format!("bg_writer_ser_{i}"))
                        .spawn(move || {
                            Self::serialize_worker(
                                &serialize_receiver,
                                &batch_sender,
                                &db,
                            );
                        })
                        .unwrap()
                })
                .collect(),

            after_commit_handle: Some({
                let after_commit_receiver = after_commit_receiver;
                let shutting_down = shutting_down.clone();

                thread::Builder::new()
                    .name("bg_writer_after_commit".to_string())
                    .spawn(move || {
                        Self::after_commit_worker(
                            &after_commit_receiver,
                            &shutting_down,
                        );
                    })
                    .unwrap()
            }),

            shutting_down,
            epoch: AtomicU64::new(0),
        }
    }

    /// Creates a new write buffer for accumulating write operations.
    #[must_use]
    pub fn new_write_batch(&self) -> WriteBatch<Db> {
        WriteBatch::new(Epoch(self.epoch.fetch_add(1, Ordering::SeqCst)), true)
    }

    /// Submits a write buffer to be processed by the background writer.
    pub fn submit_write_batch(&self, write_buffer: WriteBatch<Db>) {
        let write_task = SerializeTask { write_buffer };

        self.serialize_sender.as_ref().unwrap().send(write_task).unwrap();
    }

    fn after_commit_worker(
        receiver: &crossbeam_channel::Receiver<AfterCommitTask<Db>>,
        shutting_down: &Arc<AtomicBool>,
    ) {
        while let Ok(mut task) = receiver.recv() {
            let epoch = task.write_buffer.epoch();

            if shutting_down.load(Ordering::SeqCst) {
                task.write_buffer.active = false;
                continue;
            }

            task.write_buffer.after_commit(epoch);
            task.write_buffer.active = false;
        }
    }

    fn serialize_worker(
        receiver: &crossbeam_channel::Receiver<SerializeTask<Db>>,
        sender: &crossbeam_channel::Sender<WriteTask<Db>>,
        db: &Db,
    ) {
        while let Ok(task) = receiver.recv() {
            let mut serialization_buffer = db.serialization_buffer();
            task.write_buffer.write_to_db(&mut serialization_buffer);

            sender
                .send(WriteTask {
                    write_buffer: task.write_buffer,
                    serialize_buffer: serialization_buffer,
                })
                .unwrap();
        }
    }

    /// Puts the serialized write buffers back in epoch order and collects
    /// them into the physical batches that the commit thread writes.
    fn batch_worker(
        receiver: &crossbeam_channel::Receiver<WriteTask<Db>>,
        commit_sender: crossbeam_channel::Sender<CommitTask<Db>>,
        db: &Db,
    ) {
        let mut holdback_queues = BinaryHeap::new();

        let mut current_batch = CurrentBatch {
            processed_logical_batch: Vec::new(),
            db_write_batch: db.write_batch(),
            expected_epoch: Epoch(0),
        };

        while let Ok(task) = receiver.recv() {
            holdback_queues.push(task);

            Self::process_pending_commits(
                &mut holdback_queues,
                &mut current_batch,
                &commit_sender,
                db,
            );
        }

        // Process remaining commits
        Self::process_pending_commits(
            &mut holdback_queues,
            &mut current_batch,
            &commit_sender,
            db,
        );

        // flush any remaining in current batch
        current_batch.flush(db, &commit_sender);

        // should be empty now
        assert!(holdback_queues.is_empty());

        // close commit sender
        drop(commit_sender);
    }

    /// Writes the physical batches to the database in the order the batch
    /// thread made them.
    fn commit_worker(
        receiver: &crossbeam_channel::Receiver<CommitTask<Db>>,
        after_commit_sender: crossbeam_channel::Sender<AfterCommitTask<Db>>,
        shutting_down: &Arc<AtomicBool>,
    ) {
        while let Ok(task) = receiver.recv() {
            // commit physical batch
            task.db_write_batch.commit();

            // after commit actions
            for mut logical_batch in task.logical_batches {
                if shutting_down.load(Ordering::SeqCst).not() {
                    after_commit_sender
                        .send(AfterCommitTask { write_buffer: logical_batch })
                        .unwrap();
                } else {
                    logical_batch.active = false;
                }
            }
        }

        // close after commit sender
        drop(after_commit_sender);
    }

    fn process_pending_commits(
        pending_commits: &mut BinaryHeap<WriteTask<Db>>,
        current_batch: &mut CurrentBatch<Db>,
        commit_sender: &crossbeam_channel::Sender<CommitTask<Db>>,
        db: &Db,
    ) {
        while let Some(top) = pending_commits.peek() {
            if top.write_buffer.epoch == current_batch.expected_epoch {
                let task = pending_commits.pop().unwrap();

                current_batch
                    .db_write_batch
                    .consume_serialization_buffer(task.serialize_buffer);

                // push into current batch
                current_batch.processed_logical_batch.push(task.write_buffer);

                current_batch.expected_epoch.0 += 1;

                // commit if the physical batch is "big enough"
                if current_batch.db_write_batch.should_write_more().not() {
                    current_batch.flush(db, commit_sender);
                }
            } else {
                break;
            }
        }
    }
}

impl<Db: KvDatabase> Drop for WriteBehind<Db> {
    fn drop(&mut self) {
        self.shutting_down.store(true, Ordering::SeqCst);

        // close serialize sender
        drop(self.serialize_sender.take());

        // serialization workers should exit, close all batch senders
        for handle in self.serialize_handles.drain(..) {
            let _ = handle.join();
        }

        // batch sender should be closed now, wait for batch thread to exit
        // this will also close commit sender
        let _ = self.batch_handle.take().unwrap().join();

        // commit sender should be closed now, wait for commit thread to exit
        // this will also close after commit sender
        let _ = self.commit_handle.take().unwrap().join();

        // after commit thread should exit now
        let _ = self.after_commit_handle.take().unwrap().join();
    }
}

struct AfterCommitTask<Db: KvDatabase> {
    write_buffer: WriteBatch<Db>,
}

struct SerializeTask<Db: KvDatabase> {
    write_buffer: WriteBatch<Db>,
}

struct WriteTask<Db: KvDatabase> {
    write_buffer: WriteBatch<Db>,
    serialize_buffer: Db::SerializationBuffer,
}

/// A physical batch that is ready to be committed, and the write buffers
/// whose writes it holds.
struct CommitTask<Db: KvDatabase> {
    db_write_batch: Db::WriteBatch,
    logical_batches: Vec<WriteBatch<Db>>,
}

impl<Db: KvDatabase> PartialEq for WriteTask<Db> {
    fn eq(&self, other: &Self) -> bool {
        self.write_buffer.epoch == other.write_buffer.epoch
    }
}

impl<Db: KvDatabase> Eq for WriteTask<Db> {}

impl<Db: KvDatabase> PartialOrd for WriteTask<Db> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<Db: KvDatabase> Ord for WriteTask<Db> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse order for min-heap behavior
        other.write_buffer.epoch.cmp(&self.write_buffer.epoch)
    }
}
