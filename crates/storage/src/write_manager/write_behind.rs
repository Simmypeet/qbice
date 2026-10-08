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
//! 7. Commit thread publishes which epochs the database now holds. That is all
//!    the caches are told: an entry that holds a write of an epoch stays in its
//!    cache until that epoch has been published

use std::{
    any::{Any, TypeId},
    collections::{BinaryHeap, HashMap},
    ops::Not,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    thread,
};

use fxhash::FxBuildHasher;

use crate::{
    dynamic_map::cache::CacheDynamicMap,
    key_of_set_map::{ConcurrentSet, cache::CacheKeyOfSetMap},
    kv_database::{
        KeyOfSetColumn, KvDatabase, SerializationBuffer as _, WideColumn,
        WideColumnValue, WriteBatch as _,
    },
    single_map::cache::CacheSingleMap,
    write_batch, write_manager,
};

/// A monotonically increasing identifier for write transactions.
///
/// Epochs ensure that write transactions are committed in the order they
/// were submitted. Each new transaction receives the next epoch number,
/// and the commit thread processes transactions strictly in epoch order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Epoch(pub u64);

/// The epochs whose write batches have been committed to the database.
///
/// Write batches are committed in the order of their epochs, so one number
/// says which of them are in the database: every epoch below it. A cache
/// compares the epoch of a write it holds with this to tell whether the
/// database has that write, instead of being told about every key of every
/// committed batch.
///
/// A [`WriteBehind`] gives its committed epochs to every map it creates.
#[derive(Debug, Clone, Default)]
pub(crate) struct CommittedEpochs(Arc<AtomicU64>);

impl CommittedEpochs {
    /// Returns whether the write batch of `epoch` has been committed.
    pub(crate) fn contains(&self, epoch: Epoch) -> bool {
        self.0.load(Ordering::Acquire) > epoch.0
    }

    /// Records that every write batch with an epoch below `end` has been
    /// committed.
    pub(crate) fn advance_to(&self, end: Epoch) {
        self.0.fetch_max(end.0, Ordering::AcqRel);
    }
}

struct TypedWideColumnWrites<C: WideColumn, V: WideColumnValue<C>> {
    /// Map of keys to their corresponding values to write.
    ///
    /// `None` indicates a deletion for that key. `Some(value)` indicates
    /// an insertion or update.
    writes: HashMap<C::Key, Option<V>, FxBuildHasher>,
}

trait WriteEntry<Db: KvDatabase>: Any + Send + Sync + 'static {
    fn write_to_buffer(&self, tx: &mut Db::SerializationBuffer);
    fn as_any_mut(&mut self) -> &mut (dyn Any + Send + Sync);
}

impl<C: WideColumn, V: WideColumnValue<C>, Db: KvDatabase> WriteEntry<Db>
    for TypedWideColumnWrites<C, V>
{
    fn write_to_buffer(
        &self,
        tx: &mut <Db as KvDatabase>::SerializationBuffer,
    ) {
        for (key, value_opt) in &self.writes {
            match value_opt {
                Some(value) => tx.put(key, value),
                None => {
                    tx.delete::<C, V>(key);
                }
            }
        }
    }

    fn as_any_mut(&mut self) -> &mut (dyn Any + Send + Sync) { self }
}

impl<C: WideColumn, V: WideColumnValue<C>> TypedWideColumnWrites<C, V> {
    fn insert(&mut self, key: C::Key, value: Option<V>) {
        self.writes.insert(key, value);
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
    ) {
        let id = WideColumnWritesID::of::<C, V>();

        match self.writes.entry(id) {
            std::collections::hash_map::Entry::Occupied(mut occupied_entry) => {
                let typed_writes = occupied_entry
                    .get_mut()
                    .as_any_mut()
                    .downcast_mut::<TypedWideColumnWrites<C, V>>()
                    .expect("type mismatch in WideColumnWrites map");

                typed_writes.insert(key, value);
            }

            std::collections::hash_map::Entry::Vacant(vacant_entry) => {
                let mut typed_writes = TypedWideColumnWrites::<C, V> {
                    writes: HashMap::default(),
                };

                typed_writes.insert(key, value);

                vacant_entry.insert(Box::new(typed_writes));
            }
        }
    }

    pub(super) fn write_to_buffer(
        &self,
        tx: &mut <Db as KvDatabase>::SerializationBuffer,
    ) {
        for write_entry in self.writes.values() {
            write_entry.write_to_buffer(tx);
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

struct TypedKeyOfSetWrites<C: KeyOfSetColumn> {
    writes: HashMap<C::Key, HashMap<C::Element, Operation>, FxBuildHasher>,
}

impl<C: KeyOfSetColumn, Db: KvDatabase> WriteEntry<Db>
    for TypedKeyOfSetWrites<C>
{
    fn write_to_buffer(
        &self,
        tx: &mut <Db as KvDatabase>::SerializationBuffer,
    ) {
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

    fn as_any_mut(&mut self) -> &mut (dyn Any + Send + Sync) { self }
}

impl<C: KeyOfSetColumn> TypedKeyOfSetWrites<C> {
    fn insert(&mut self, key: C::Key, element: C::Element, op: Operation) {
        self.writes.entry(key).or_default().insert(element, op);
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
    ) {
        match self.writes.entry(TypeId::of::<C>()) {
            std::collections::hash_map::Entry::Occupied(mut occupied_entry) => {
                let typed_writes = occupied_entry
                    .get_mut()
                    .as_any_mut()
                    .downcast_mut::<TypedKeyOfSetWrites<C>>()
                    .expect("type mismatch in KeyOfSetWrites map");

                typed_writes.insert(key, element, op);
            }

            std::collections::hash_map::Entry::Vacant(vacant_entry) => {
                let mut typed_writes =
                    TypedKeyOfSetWrites::<C> { writes: HashMap::default() };

                typed_writes.insert(key, element, op);

                vacant_entry.insert(Box::new(typed_writes));
            }
        }
    }

    pub(super) fn write_to_db(
        &self,
        tx: &mut <Db as KvDatabase>::SerializationBuffer,
    ) {
        for write_entry in self.writes.values() {
            write_entry.write_to_buffer(tx);
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
/// 5. **Release**: Buffer is dropped once its writes have been serialized. The
///    caches hold what it wrote until the database holds it too
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
/// - Cleared once the background writer has serialized it
/// - **Panics** if dropped while still active (indicates programming error)
///
/// # Example
///
/// ```ignore
/// let writer = engine.new_write_manager();
/// let cache = writer.new_single_map::<Column, Value>();
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
}

impl<Db: KvDatabase> write_batch::WriteBatch for WriteBatch<Db> {}

impl<Db: KvDatabase> std::fmt::Debug for WriteBatch<Db> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WriteBuffer").finish_non_exhaustive()
    }
}

impl<Db: KvDatabase> WriteBatch<Db> {
    fn write_to_db(&self, tx: &mut <Db as KvDatabase>::SerializationBuffer) {
        self.wide_column_writes.write_to_buffer(tx);
        self.key_of_set_writes.write_to_db(tx);
    }

    pub(crate) fn put_wide_column<C: WideColumn, V: WideColumnValue<C>>(
        &mut self,
        key: C::Key,
        value: Option<V>,
    ) {
        self.wide_column_writes.put::<C, V>(key, value);
    }

    pub(crate) fn put_set<C: KeyOfSetColumn>(
        &mut self,
        key: C::Key,
        element: C::Element,
        op: Operation,
    ) {
        self.key_of_set_writes.put::<C>(key, element, op);
    }

    #[must_use]
    pub(crate) const fn epoch(&self) -> Epoch { self.epoch }
}

impl<Db: KvDatabase> WriteBatch<Db> {
    fn new(epoch: Epoch) -> Self {
        Self {
            wide_column_writes: WideColumnWrites::new(),
            key_of_set_writes: KeyOfSetWrites::new(),
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
/// 4. **Worker Processing**: Worker picks up buffer, serializes its writes and
///    drops it
/// 5. **Batching**: Batch thread collects the buffers in epoch order into a
///    database write batch and prepares it
/// 6. **Commit**: Commit thread applies the write batches in that order, while
///    the batch thread is already preparing the next one
/// 7. **Publication**: Commit thread publishes that the epochs of the batch are
///    in the database, which lets the caches evict what those epochs wrote
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
/// let map = write_manager.new_single_map::<MyColumn, MyValue>();
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

    epoch: AtomicU64,
    committed: CommittedEpochs,

    /// The database the maps of this writer read from.
    db: Db,

    /// The number of entries each map of this writer keeps in memory.
    cache_capacity: u64,
}

impl<Db: KvDatabase> write_manager::WriteManager for WriteBehind<Db> {
    type WriteBatch = WriteBatch<Db>;

    type SingleMap<K: WideColumn, V: WideColumnValue<K>> =
        CacheSingleMap<K, V, Db>;

    type DynamicMap<K: WideColumn> = CacheDynamicMap<K, Db>;

    type KeyOfSetMap<
        K: KeyOfSetColumn,
        C: ConcurrentSet<Element = K::Element>,
    > = CacheKeyOfSetMap<K, C, Db>;

    fn new_write_batch(&self) -> Self::WriteBatch {
        Self::new_write_batch(self)
    }

    fn submit_write_batch(&self, write_transaction: Self::WriteBatch) {
        Self::submit_write_batch(self, write_transaction);
    }

    fn new_single_map<K: WideColumn, V: WideColumnValue<K>>(
        &self,
    ) -> Self::SingleMap<K, V> {
        CacheSingleMap::new(
            self.cache_capacity,
            self.db.clone(),
            self.committed.clone(),
        )
    }

    fn new_dynamic_map<K: WideColumn>(&self) -> Self::DynamicMap<K> {
        CacheDynamicMap::new(
            self.cache_capacity,
            self.db.clone(),
            self.committed.clone(),
        )
    }

    fn new_key_of_set_map<
        K: KeyOfSetColumn,
        C: ConcurrentSet<Element = K::Element>,
    >(
        &self,
    ) -> Self::KeyOfSetMap<K, C> {
        CacheKeyOfSetMap::new(
            self.cache_capacity,
            self.db.clone(),
            self.committed.clone(),
        )
    }
}

impl<Db: KvDatabase> std::fmt::Debug for WriteBehind<Db> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BackgroundWriter").finish_non_exhaustive()
    }
}

struct CurrentBatch<Db: KvDatabase> {
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

        // done on this thread so that the commit thread only has to write
        db_write_batch.prepare();

        commit_sender
            .send(CommitTask { db_write_batch, end: self.expected_epoch })
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
    /// * `cache_capacity` - The number of entries that each map created by this
    ///   writer keeps in memory.
    ///
    /// # Thread Spawning
    ///
    /// This method spawns `num_threads + 2` threads:
    /// - `num_threads` worker threads (named "`bg_writer_ser_0`",
    ///   "`bg_writer_ser_1`", ...)
    /// - 1 batch thread (named "`bg_writer_batch`")
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
    pub fn new(
        db: &Db,
        serialize_worker_count: usize,
        cache_capacity: u64,
    ) -> Self {
        let (batch_sender, batch_receiver) =
            crossbeam_channel::unbounded::<WriteTask<Db>>();
        // The batch thread prepares one batch ahead of the commit thread and
        // then waits for it.
        let (commit_sender, commit_receiver) =
            crossbeam_channel::bounded::<CommitTask<Db>>(1);
        let (serialize_sender, serialize_receiver) =
            crossbeam_channel::unbounded::<SerializeTask<Db>>();

        let committed = CommittedEpochs::default();

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
                let committed = committed.clone();

                thread::Builder::new()
                    .name("bg_writer_commit".to_string())
                    .spawn(move || {
                        Self::commit_worker(&commit_receiver, &committed);
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

            epoch: AtomicU64::new(0),
            committed,

            db: db.clone(),
            cache_capacity,
        }
    }

    /// Creates a new write buffer for accumulating write operations.
    #[must_use]
    pub fn new_write_batch(&self) -> WriteBatch<Db> {
        WriteBatch::new(Epoch(self.epoch.fetch_add(1, Ordering::SeqCst)))
    }

    /// Submits a write buffer to be processed by the background writer.
    pub fn submit_write_batch(&self, write_buffer: WriteBatch<Db>) {
        let write_task = SerializeTask { write_buffer };

        self.serialize_sender.as_ref().unwrap().send(write_task).unwrap();
    }

    fn serialize_worker(
        receiver: &crossbeam_channel::Receiver<SerializeTask<Db>>,
        sender: &crossbeam_channel::Sender<WriteTask<Db>>,
        db: &Db,
    ) {
        while let Ok(SerializeTask { write_buffer }) = receiver.recv() {
            let mut serialize_buffer = db.serialization_buffer();
            write_buffer.write_to_db(&mut serialize_buffer);
            let epoch = write_buffer.epoch;

            sender.send(WriteTask { epoch, serialize_buffer }).unwrap();
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
            db_write_batch: db.write_batch(),
            expected_epoch: Epoch(0),
        };

        while let Ok(task) = receiver.recv() {
            holdback_queues.push(task);

            Self::process_pending_writes(
                &mut holdback_queues,
                &mut current_batch,
                &commit_sender,
                db,
            );
        }

        // Process remaining commits
        Self::process_pending_writes(
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
        committed: &CommittedEpochs,
    ) {
        while let Ok(task) = receiver.recv() {
            // commit physical batch
            task.db_write_batch.commit();

            // Not before the commit: a cache may evict an entry as soon as
            // the epoch that wrote it is published, and the next read of
            // that entry then has to find the write in the database.
            committed.advance_to(task.end);
        }
    }

    fn process_pending_writes(
        pending_writes: &mut BinaryHeap<WriteTask<Db>>,
        current_batch: &mut CurrentBatch<Db>,
        commit_sender: &crossbeam_channel::Sender<CommitTask<Db>>,
        db: &Db,
    ) {
        while let Some(top) = pending_writes.peek() {
            if top.epoch == current_batch.expected_epoch {
                let task = pending_writes.pop().unwrap();

                current_batch
                    .db_write_batch
                    .consume_serialization_buffer(task.serialize_buffer);

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
        let _ = self.commit_handle.take().unwrap().join();
    }
}

struct SerializeTask<Db: KvDatabase> {
    write_buffer: WriteBatch<Db>,
}

/// The serialized writes of the write buffer of `epoch`.
struct WriteTask<Db: KvDatabase> {
    epoch: Epoch,
    serialize_buffer: Db::SerializationBuffer,
}

/// A physical batch that is ready to be committed. It completes the writes
/// of every epoch below `end`.
struct CommitTask<Db: KvDatabase> {
    db_write_batch: Db::WriteBatch,
    end: Epoch,
}

impl<Db: KvDatabase> PartialEq for WriteTask<Db> {
    fn eq(&self, other: &Self) -> bool { self.epoch == other.epoch }
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
        other.epoch.cmp(&self.epoch)
    }
}

#[cfg(test)]
mod test;
