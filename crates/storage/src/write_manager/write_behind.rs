//! Write-behind caching implementation.
//!
//! This module provides [`WriteBehind`], an asynchronous background writer
//! that enables write-back caching. Cache updates are fast (in-memory only)
//! while persistence happens asynchronously in background worker threads.
//!
//! # Key Components
//!
//! - [`WriteBehind`]: The main background writer managing worker threads
//! - [`WriteBatch`]: A buffer for accumulating write operations
//! - [`Epoch`]: Monotonic ordering identifier for write batches
//!
//! # Write Pipeline
//!
//! 1. Application creates a [`WriteBatch`] via
//!    [`WriteBehind::new_write_batch()`]
//! 2. Writes are serialized into the write batch as they are added, on the
//!    thread that adds them
//! 3. Write batch is submitted via [`WriteBehind::submit_write_batch()`]
//! 4. Batch thread puts the write batches back in epoch order and groups them
//!    into database write batches
//! 5. Prepare thread gets each database write batch ready to be written
//! 6. Commit thread applies the database write batches to the database, one at
//!    a time
//! 7. Commit thread publishes which epochs the database now holds. That is all
//!    the caches are told: an entry that holds a write of an epoch stays in its
//!    cache until that epoch has been published
//!
//! # Falling behind
//!
//! Whoever submits a write batch waits while the database is more than
//! `MAX_UNCOMMITTED_BYTES` (1 GiB) behind. Without that, writes that are
//! produced faster than the database takes them pile up in memory for as long
//! as that lasts, and with them every cache entry they pin.
//!
//! # Write batches that are not submitted
//!
//! Every [`WriteBatch`] has to be submitted, also one that nothing was written
//! to. The write batches are committed in the order of their epochs, so one
//! that is never submitted keeps every write batch after it from being
//! committed. A [`WriteBatch`] that is dropped instead panics.

use std::{
    collections::BinaryHeap,
    ops::Not,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
    thread,
};

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

/// A monotonically increasing identifier for write batches.
///
/// Epochs ensure that write batches are committed in the order they were
/// created, in whatever order they are submitted. Each new write batch
/// receives the next epoch number, and the database is given the write
/// batches strictly in epoch order.
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

/// The type of operation to perform on a key-of-set element.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Operation {
    /// Insert an element into the set.
    Insert,
    /// Delete an element from the set.
    Remove,
}

/// A buffer that accumulates write operations for batch processing.
///
/// `WriteBatch` collects multiple cache write operations (both wide column
/// and key-of-set writes) in memory before they are asynchronously flushed to
/// the database. This enables write-back caching with significantly reduced
/// database write amplification.
///
/// # Architecture
///
/// The write batch holds its writes the way the database takes them. A write
/// is serialized when it is added, on the thread that adds it, and nothing
/// but the bytes is kept: the values themselves are in the caches.
///
/// The writes are kept in the order they were added, and a write to a key
/// that was written before does not replace the earlier one. The database
/// applies them in that order, so the last write to a key still decides what
/// the key holds.
///
/// # Lifecycle
///
/// 1. **Creation**: Obtained from [`WriteBehind::new_write_batch()`]
/// 2. **Accumulation**: Writes are added through the maps that the
///    [`WriteBehind`] creates
/// 3. **Submission**: Handed over with [`WriteBehind::submit_write_batch()`]
/// 4. **Processing**: Background threads write it to the database
/// 5. **Release**: The caches hold what the write batch wrote until the
///    database holds it too
///
/// # Epoch Ordering
///
/// Each write batch has an epoch number ensuring writes are committed in
/// order:
/// - Write batches are given to the database in epoch order
/// - A write batch is not committed before every write batch with an earlier
///   epoch has been submitted
/// - Ensures write consistency and prevents reordering
///
/// # Durability
///
/// **Important**: Writes in the write batch are **not durable** until:
/// 1. It is submitted via [`WriteBehind::submit_write_batch()`]
/// 2. The background threads have given it to the database
/// 3. The database has flushed it to disk
///
/// **Data Loss Risk**: If the process crashes before that, the writes are
/// lost.
///
/// # Panics
///
/// A write batch **panics when it is dropped** without having been submitted.
/// Its epoch would never arrive, and no write batch with a later epoch could
/// be committed anymore. A write batch that nothing was written to has to be
/// submitted all the same.
///
/// A write batch that is dropped because its thread is already panicking
/// does not panic again.
///
/// # Example
///
/// ```ignore
/// let writer = engine.new_write_manager();
/// let cache = writer.new_single_map::<Column, Value>();
///
/// // Create write batch
/// let mut batch = writer.new_write_batch();
///
/// // Accumulate writes
/// cache.insert(key1, value1, &mut batch).await;
/// cache.insert(key2, value2, &mut batch).await;
///
/// // Submit for async persistence
/// writer.submit_write_batch(batch);
/// // Writes will be flushed to database in background
/// ```
pub struct WriteBatch<Db: KvDatabase> {
    /// The writes of the batch, serialized in the order they were added.
    buffer: Db::SerializationBuffer,
    epoch: Epoch,
    unsubmitted: Unsubmitted,
}

/// Panics when it is dropped. A [`WriteBatch`] holds one until it is
/// submitted, which is what makes dropping the write batch instead panic.
///
/// This is a field of its own so that submitting can still take the write
/// batch apart, which a `Drop` on the write batch itself would not allow.
struct Unsubmitted(Epoch);

impl Drop for Unsubmitted {
    fn drop(&mut self) {
        // a second panic while the first one unwinds aborts the process
        if thread::panicking() {
            return;
        }

        panic!(
            "the write batch of epoch {} was dropped without being submitted, \
             which keeps every write batch after it from being committed",
            self.0.0
        );
    }
}

impl<Db: KvDatabase> write_batch::WriteBatch for WriteBatch<Db> {}

impl<Db: KvDatabase> std::fmt::Debug for WriteBatch<Db> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WriteBatch").finish_non_exhaustive()
    }
}

impl<Db: KvDatabase> WriteBatch<Db> {
    /// Adds the write that stores `value` under `key`, or that deletes what
    /// is stored there if `value` is `None`.
    pub(crate) fn put_wide_column<C: WideColumn, V: WideColumnValue<C>>(
        &mut self,
        key: &C::Key,
        value: Option<&V>,
    ) {
        match value {
            Some(value) => self.buffer.put::<C, V>(key, value),
            None => self.buffer.delete::<C, V>(key),
        }
    }

    /// Adds the write that makes `element` a member of the set of `key`, or
    /// that makes it not a member anymore.
    pub(crate) fn put_set<C: KeyOfSetColumn>(
        &mut self,
        key: &C::Key,
        element: &C::Element,
        op: Operation,
    ) {
        match op {
            Operation::Insert => self.buffer.insert_member::<C>(key, element),
            Operation::Remove => self.buffer.delete_member::<C>(key, element),
        }
    }

    #[must_use]
    pub(crate) const fn epoch(&self) -> Epoch { self.epoch }
}

/// The most bytes of serialized writes that wait to be committed before
/// whoever submits more has to wait for the database.
///
/// A run that the database keeps up with most of the time is not meant to
/// get here. The limit is for the run that the database does not keep up
/// with, where what is waiting would otherwise grow for as long as the run
/// lasts.
///
/// A lower limit makes submitters wait for what is over quickly anyway. A
/// single write buffer can be several hundred megabytes, like the one of an
/// input session that marks most of a large graph dirty, and whoever submits
/// a buffer that is larger than the limit waits until it has been committed.
const MAX_UNCOMMITTED_BYTES: usize = 1024 * 1024 * 1024;

/// How far the database is behind what has been submitted.
///
/// A write buffer is counted from the moment it is submitted until it has
/// been committed, except for as long as the batch thread holds it back
/// because a write buffer with an earlier epoch has not been submitted yet.
/// What is counted is on its way to the database without anybody else having
/// to do anything, so whoever waits for the count to go down waits for
/// something that is going to happen.
///
/// A write buffer that is held back is different. It cannot be committed
/// before the one it waits for has been submitted, and whoever is going to
/// submit that one may need one of the threads that would be waiting here in
/// order to get that far.
///
/// # When a background thread dies
///
/// The count only goes down as long as the background threads are running.
/// One of them that panics, like the commit thread on a write that the
/// database refuses, takes the others with it, and nothing that is counted
/// is ever committed. Whoever waits for that is told instead of being left
/// waiting: the backlog is marked as failed, and every wait panics from then
/// on.
struct Backlog {
    /// The bytes of the write buffers that have been submitted and not
    /// committed yet, without the ones that are held back.
    uncommitted: AtomicUsize,

    /// The number of uncommitted bytes above which a submitter waits.
    limit: usize,

    /// Whether a background thread has panicked, so that nothing more is
    /// going to be committed.
    failed: AtomicBool,

    lock: parking_lot::Mutex<()>,
    committed: parking_lot::Condvar,
}

/// Marks a [`Backlog`] as failed if it is dropped while its thread is
/// panicking. Every background thread holds one for as long as it runs.
struct FailOnPanic<'a>(&'a Backlog);

impl Drop for FailOnPanic<'_> {
    fn drop(&mut self) {
        if thread::panicking() {
            self.0.fail();
        }
    }
}

impl Backlog {
    const fn new(limit: usize) -> Self {
        Self {
            uncommitted: AtomicUsize::new(0),
            limit,
            failed: AtomicBool::new(false),
            lock: parking_lot::Mutex::new(()),
            committed: parking_lot::Condvar::new(),
        }
    }

    /// Counts a write buffer that is on its way to the database.
    fn add(&self, bytes: usize) {
        self.uncommitted.fetch_add(bytes, Ordering::SeqCst);
    }

    /// Stops counting a write buffer, because it has been committed or
    /// because it is held back, and wakes whoever is waiting for the count
    /// to go down.
    fn remove(&self, bytes: usize) {
        self.uncommitted.fetch_sub(bytes, Ordering::SeqCst);

        // Whoever is about to wait holds the lock from the moment it looks
        // at the count until it sleeps. Taking the lock here puts this
        // either before the look or after the sleep has begun, so the
        // notification is not missed.
        drop(self.lock.lock());

        self.committed.notify_all();
    }

    /// Returns what marks the backlog as failed if the calling thread
    /// panics before it is dropped.
    const fn fail_on_panic(&self) -> FailOnPanic<'_> { FailOnPanic(self) }

    /// Records that nothing more is going to be committed, and wakes whoever
    /// is waiting for the count to go down.
    fn fail(&self) {
        self.failed.store(true, Ordering::SeqCst);

        // as in `remove`, so that the notification is not missed
        drop(self.lock.lock());

        self.committed.notify_all();
    }

    /// Waits until the database is no further behind than the limit.
    ///
    /// # Panics
    ///
    /// Panics if a background thread has panicked: what this would wait for
    /// is not going to happen.
    fn wait(&self) {
        if self.uncommitted.load(Ordering::SeqCst) <= self.limit {
            return;
        }

        let mut guard = self.lock.lock();

        while self.uncommitted.load(Ordering::SeqCst) > self.limit {
            assert!(
                !self.failed.load(Ordering::SeqCst),
                "a background thread of the write-behind has panicked, so the \
                 writes that were submitted are not going to be committed"
            );

            self.committed.wait(&mut guard);
        }
    }
}

/// An asynchronous background writer enabling write-back caching strategy.
///
/// `WriteBehind` manages the threads that write the submitted write batches
/// to the database. This enables high-performance write-back caching where
/// cache updates are fast (in-memory only) and persistence happens in the
/// background.
///
/// # Architecture
///
/// ```text
///                     WriteBatch
///          (serialized by whoever writes to it)
///                          |
///                          v
///                    Batch Thread
///                          |
///                 database write batch
///                          |
///                          v
///                   Prepare Thread
///                          |
///                          v
///                    Commit Thread
///                          |
///                          v
///                       Database
/// ```
///
///
/// # Write Processing Pipeline
///
/// 1. **Creation**: Application creates a [`WriteBatch`]
/// 2. **Accumulation**: Writes are serialized into the write batch as they are
///    added
/// 3. **Submission**: Write batch is handed to the batch thread. Whoever
///    submits waits here if the database is too far behind
/// 4. **Batching**: Batch thread collects the write batches in epoch order into
///    database write batches
/// 5. **Preparation**: Prepare thread gets a database write batch ready to be
///    written
/// 6. **Commit**: Commit thread applies the database write batches in that
///    order, while the prepare thread is already preparing the next one
/// 7. **Publication**: Commit thread publishes that the epochs of the batch are
///    in the database, which lets the caches evict what those epochs wrote
///
/// # Epoch-Based Ordering
///
/// Write batches are assigned monotonically increasing epoch numbers when
/// they are created:
/// - Ensures writes are committed in the order of their epochs, in whatever
///   order the write batches are submitted
/// - A write batch waits until every earlier epoch has been submitted, so every
///   write batch has to be submitted (see [`WriteBatch`])
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
///     let mut batch = write_manager.new_write_batch();
///
///     // Accumulate many writes (fast, in-memory)
///     for (key, value) in updates {
///         map.insert(key, value, &mut batch).await;
///     }
///
///     // Submit entire batch for async persistence
///     write_manager.submit_write_batch(batch);
///     // Writes happen in background. This only waits if the database is
///     // too far behind
/// }
///
/// // On shutdown, drop writer to gracefully drain queues
/// drop(write_manager);
/// ```
pub struct WriteBehind<Db: KvDatabase> {
    batch_handle: Option<thread::JoinHandle<()>>,
    prepare_handle: Option<thread::JoinHandle<()>>,
    commit_handle: Option<thread::JoinHandle<()>>,

    batch_sender: Option<crossbeam_channel::Sender<WriteTask<Db>>>,

    epoch: AtomicU64,
    committed: CommittedEpochs,
    backlog: Arc<Backlog>,

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
        f.debug_struct("WriteBehind").finish_non_exhaustive()
    }
}

struct CurrentBatch<Db: KvDatabase> {
    db_write_batch: Db::WriteBatch,
    expected_epoch: Epoch,

    /// The bytes of the write buffers that the write batch holds.
    bytes: usize,
}

impl<Db: KvDatabase> CurrentBatch<Db> {
    /// Hands the write batch over to the prepare thread and starts a new one.
    pub fn flush(
        &mut self,
        db: &Db,
        prepare_sender: &crossbeam_channel::Sender<CommitTask<Db>>,
    ) {
        let db_write_batch =
            std::mem::replace(&mut self.db_write_batch, db.write_batch());

        prepare_sender
            .send(CommitTask {
                db_write_batch,
                end: self.expected_epoch,
                // NOTE: std::mem::take also resets the count to 0
                bytes: std::mem::take(&mut self.bytes),
            })
            .unwrap();
    }
}

impl<Db: KvDatabase> WriteBehind<Db> {
    /// Creates a new background writer.
    ///
    /// # Parameters
    ///
    /// * `db` - The database instance to write to.
    ///
    /// * `cache_capacity` - The number of entries that each map created by this
    ///   writer keeps in memory.
    ///
    /// # Thread Spawning
    ///
    /// This method spawns three threads:
    /// - the batch thread (named "`bg_writer_batch`")
    /// - the prepare thread (named "`bg_writer_prepare`")
    /// - the commit thread (named "`bg_writer_commit`")
    ///
    /// All threads start immediately and begin waiting for work.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let writer = WriteBehind::new(&database, 1 << 18);
    ///
    /// // Use writer...
    ///
    /// // Shutdown gracefully on drop
    /// drop(writer);
    /// ```
    pub fn new(db: &Db, cache_capacity: u64) -> Self {
        Self::with_backlog_limit(db, cache_capacity, MAX_UNCOMMITTED_BYTES)
    }

    /// Creates a new background writer whose submitters wait while more than
    /// `backlog_limit` bytes are waiting to be committed.
    fn with_backlog_limit(
        db: &Db,
        cache_capacity: u64,
        backlog_limit: usize,
    ) -> Self {
        let (batch_sender, batch_receiver) =
            crossbeam_channel::unbounded::<WriteTask<Db>>();
        // What is in here has been counted as waiting to be committed, so
        // the submitters are made to wait long before this gets long.
        let (prepare_sender, prepare_receiver) =
            crossbeam_channel::unbounded::<CommitTask<Db>>();
        // The prepare thread prepares one batch ahead of the commit thread
        // and then waits for it.
        let (commit_sender, commit_receiver) =
            crossbeam_channel::bounded::<CommitTask<Db>>(1);

        let committed = CommittedEpochs::default();
        let backlog = Arc::new(Backlog::new(backlog_limit));

        Self {
            batch_handle: Some({
                let db = db.clone();
                let backlog = backlog.clone();

                thread::Builder::new()
                    .name("bg_writer_batch".to_string())
                    .spawn(move || {
                        let _fail_on_panic = backlog.fail_on_panic();

                        Self::batch_worker(
                            &batch_receiver,
                            prepare_sender,
                            &db,
                            &backlog,
                        );
                    })
                    .unwrap()
            }),

            prepare_handle: Some({
                let backlog = backlog.clone();

                thread::Builder::new()
                    .name("bg_writer_prepare".to_string())
                    .spawn(move || {
                        let _fail_on_panic = backlog.fail_on_panic();

                        Self::prepare_worker(&prepare_receiver, &commit_sender);
                    })
                    .unwrap()
            }),

            commit_handle: Some({
                let committed = committed.clone();
                let backlog = backlog.clone();

                thread::Builder::new()
                    .name("bg_writer_commit".to_string())
                    .spawn(move || {
                        let _fail_on_panic = backlog.fail_on_panic();

                        Self::commit_worker(
                            &commit_receiver,
                            &committed,
                            &backlog,
                        );
                    })
                    .unwrap()
            }),

            batch_sender: Some(batch_sender),

            epoch: AtomicU64::new(0),
            committed,
            backlog,

            db: db.clone(),
            cache_capacity,
        }
    }

    /// Creates a new write batch for accumulating write operations, with the
    /// next epoch.
    ///
    /// The write batch has to be given to [`Self::submit_write_batch`], also
    /// if nothing is written to it: it panics when it is dropped instead.
    #[must_use]
    pub fn new_write_batch(&self) -> WriteBatch<Db> {
        let epoch = Epoch(self.epoch.fetch_add(1, Ordering::SeqCst));

        WriteBatch {
            buffer: self.db.serialization_buffer(),
            epoch,
            unsubmitted: Unsubmitted(epoch),
        }
    }

    /// Submits a write batch to be processed by the background writer.
    ///
    /// This blocks while the database is too far behind what has been
    /// submitted. What it waits for is writes that are ready to be committed
    /// being committed, which the background threads do on their own, so the
    /// wait ends no matter what the caller or anybody else holds on to.
    ///
    /// # Panics
    ///
    /// Panics if a background thread has panicked, which is what a write
    /// that the database refuses makes the commit thread do. Nothing that is
    /// submitted is committed anymore after that.
    pub fn submit_write_batch(&self, write_buffer: WriteBatch<Db>) {
        let WriteBatch { buffer, epoch, unsubmitted } = write_buffer;

        // the write batch is being submitted, so there is nothing to panic
        // about anymore
        std::mem::forget(unsubmitted);

        let bytes = buffer.size();

        // Counted before the buffer is handed over, so that the count is
        // never behind what has been submitted.
        self.backlog.add(bytes);

        self.batch_sender
            .as_ref()
            .unwrap()
            .send(WriteTask {
                epoch,
                serialize_buffer: buffer,
                bytes,
                held_back: false,
            })
            .unwrap();

        // Not before the buffer has been handed over: nothing that comes
        // after it can be committed without it.
        self.backlog.wait();
    }

    /// Puts the submitted write buffers back in epoch order and collects
    /// them into the write batches that the database is given.
    fn batch_worker(
        receiver: &crossbeam_channel::Receiver<WriteTask<Db>>,
        prepare_sender: crossbeam_channel::Sender<CommitTask<Db>>,
        db: &Db,
        backlog: &Backlog,
    ) {
        let mut holdback_queues = BinaryHeap::new();

        let mut current_batch = CurrentBatch {
            db_write_batch: db.write_batch(),
            expected_epoch: Epoch(0),
            bytes: 0,
        };

        while let Ok(mut task) = receiver.recv() {
            if task.epoch != current_batch.expected_epoch {
                // The buffer cannot be committed before every buffer with an
                // earlier epoch has been submitted. Nobody is to wait for
                // it until then.
                backlog.remove(task.bytes);
                task.held_back = true;
            }

            holdback_queues.push(task);

            Self::process_pending_writes(
                &mut holdback_queues,
                &mut current_batch,
                &prepare_sender,
                db,
                backlog,
            );
        }

        // Process remaining commits
        Self::process_pending_writes(
            &mut holdback_queues,
            &mut current_batch,
            &prepare_sender,
            db,
            backlog,
        );

        // flush any remaining in current batch
        current_batch.flush(db, &prepare_sender);

        // should be empty now
        assert!(holdback_queues.is_empty());

        // close prepare sender
        drop(prepare_sender);
    }

    /// Gets the write batches ready to be written, in the order the batch
    /// thread made them, so that the commit thread only has to write.
    fn prepare_worker(
        receiver: &crossbeam_channel::Receiver<CommitTask<Db>>,
        commit_sender: &crossbeam_channel::Sender<CommitTask<Db>>,
    ) {
        while let Ok(mut task) = receiver.recv() {
            task.db_write_batch.prepare();

            commit_sender.send(task).unwrap();
        }
    }

    /// Writes the write batches to the database in the order the batch
    /// thread made them.
    fn commit_worker(
        receiver: &crossbeam_channel::Receiver<CommitTask<Db>>,
        committed: &CommittedEpochs,
        backlog: &Backlog,
    ) {
        while let Ok(task) = receiver.recv() {
            // commit physical batch
            task.db_write_batch.commit();

            // Not before the commit: a cache may evict an entry as soon as
            // the epoch that wrote it is published, and the next read of
            // that entry then has to find the write in the database.
            committed.advance_to(task.end);

            backlog.remove(task.bytes);
        }
    }

    fn process_pending_writes(
        pending_writes: &mut BinaryHeap<WriteTask<Db>>,
        current_batch: &mut CurrentBatch<Db>,
        prepare_sender: &crossbeam_channel::Sender<CommitTask<Db>>,
        db: &Db,
        backlog: &Backlog,
    ) {
        while let Some(top) = pending_writes.peek() {
            if top.epoch == current_batch.expected_epoch {
                let task = pending_writes.pop().unwrap();

                // From here on the buffer gets committed without anybody
                // having to submit anything first.
                if task.held_back {
                    backlog.add(task.bytes);
                }

                current_batch.bytes += task.bytes;

                current_batch
                    .db_write_batch
                    .consume_serialization_buffer(task.serialize_buffer);

                current_batch.expected_epoch.0 += 1;

                // Commit if the write batch is "big enough". It is not left
                // to the database alone to say when that is: the submitters
                // wait for what has been counted to be committed, and a
                // write batch that is never handed over is never committed.
                if current_batch.db_write_batch.should_write_more().not()
                    || current_batch.bytes >= backlog.limit / 4
                {
                    current_batch.flush(db, prepare_sender);
                }
            } else {
                break;
            }
        }
    }
}

impl<Db: KvDatabase> Drop for WriteBehind<Db> {
    fn drop(&mut self) {
        // close batch sender
        drop(self.batch_sender.take());

        // the batch thread should exit, this will also close prepare sender
        let _ = self.batch_handle.take().unwrap().join();

        // prepare sender should be closed now, wait for prepare thread to
        // exit, this will also close commit sender
        let _ = self.prepare_handle.take().unwrap().join();

        // commit sender should be closed now, wait for commit thread to exit
        let _ = self.commit_handle.take().unwrap().join();
    }
}

/// The serialized writes of the write buffer of `epoch`.
struct WriteTask<Db: KvDatabase> {
    epoch: Epoch,
    serialize_buffer: Db::SerializationBuffer,

    /// The bytes that the writes take up.
    bytes: usize,

    /// Whether the batch thread has taken the buffer out of the count of the
    /// [`Backlog`] while it waits for the buffers before it.
    held_back: bool,
}

/// A write batch on its way to the database. It completes the writes of
/// every epoch below `end`.
struct CommitTask<Db: KvDatabase> {
    db_write_batch: Db::WriteBatch,
    end: Epoch,

    /// The bytes of the write buffers that the write batch holds.
    bytes: usize,
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
