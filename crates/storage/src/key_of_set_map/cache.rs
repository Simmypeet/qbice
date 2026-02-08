//! Cached implementation of [`KeyOfSetMap`].
//!
//! This module provides [`CacheKeyOfSetMap`], which wraps a database backend
//! with caching for improved read performance on key-to-set relationships.

use std::{
    collections::{BinaryHeap, HashSet},
    hash::Hash,
    ops::Not,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use crossbeam::queue::SegQueue;
use fxhash::FxBuildHasher;
use parking_lot::RwLock;

use crate::{
    key_of_set_map::{ConcurrentSet, KeyOfSetMap, OwnedIterator},
    kv_database::{KeyOfSetColumn, KvDatabase},
    single_flight,
    tiny_lfu::{self, LifecycleListener, TinyLFU},
    write_manager::write_behind::{self, Epoch, KeyOfSetCache},
};

/// A cached implementation of [`KeyOfSetMap`] backed by a
/// database.
///
/// This implementation combines a Moka cache for fast set access with a
/// database backend for persistence. For large sets (>1024 elements), it
/// falls back to streaming from the database to avoid memory exhaustion.
///
/// # Type Parameters
///
/// - `K`: The key-of-set column type.
/// - `C`: The concurrent set type for storing elements.
/// - `Db`: The database backend implementing [`KvDatabase`].
#[derive(Debug)]
pub struct CacheKeyOfSetMap<
    K: KeyOfSetColumn,
    C: ConcurrentSet<Element = K::Element> + 'static,
    Db: KvDatabase,
> {
    repr: Arc<Repr<K, C>>,
    db: Db,
}

impl<
    K: KeyOfSetColumn,
    C: ConcurrentSet<Element = K::Element> + 'static,
    Db: KvDatabase,
> CacheKeyOfSetMap<K, C, Db>
{
    /// Creates a new cached key-of-set map with the specified capacity.
    ///
    /// # Returns
    ///
    /// A new `CacheKeyOfSetMap` instance.
    #[must_use]
    pub fn new(cap: u64, shard_amount: usize, db: Db) -> Self {
        Self { repr: Arc::new(Repr::new(cap, shard_amount)), db }
    }
}

#[derive(Debug, Clone)]
enum Operation<V> {
    Insert(V),
    Remove(V),
}

#[derive(Debug, Clone)]
enum Entry<C> {
    /// For < 1024 elements, store in-memory set
    InMemory(C),

    /// For >= 1024 elements, mark as too large, and rely on streaming from DB
    TooLarge,
}

/// A versioned operation for tracking staged writes.
#[derive(Debug)]
pub struct VersionedOperation<V> {
    op: Operation<V>,
    epoch: Epoch,
}

impl<V> Eq for VersionedOperation<V> {}

impl<V> PartialEq for VersionedOperation<V> {
    fn eq(&self, other: &Self) -> bool { self.epoch.eq(&other.epoch) }
}

impl<V> PartialOrd for VersionedOperation<V> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<V> Ord for VersionedOperation<V> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.epoch.cmp(&other.epoch)
    }
}

#[derive(Debug)]
struct TrackedConcurrentLog<V> {
    log: Arc<ConcurrentLog<V>>,
    dirty: AtomicUsize,
}

enum ConcurrentLogMessage<V> {
    FlushUpTo(Epoch),
    AppendOperation(VersionedOperation<V>),
}

#[derive(Debug)]
struct ConcurrentLog<V> {
    log: RwLock<BinaryHeap<VersionedOperation<V>>>,
    deferred_messages: SegQueue<ConcurrentLogMessage<V>>,
}

impl<V: Eq + Hash + Clone> ConcurrentLog<V> {
    const fn new() -> Self {
        Self {
            log: RwLock::new(BinaryHeap::new()),
            deferred_messages: SegQueue::new(),
        }
    }

    fn apply_message(&self, op: ConcurrentLogMessage<V>) {
        let Some(mut lock) = self.log.try_write() else {
            self.deferred_messages.push(op);

            return;
        };

        Self::fix(&mut lock, &self.deferred_messages);

        Self::apply_message_to_heap(&mut lock, op);
    }

    fn apply_message_to_heap(
        heap_lock: &mut BinaryHeap<VersionedOperation<V>>,
        op: ConcurrentLogMessage<V>,
    ) {
        match op {
            ConcurrentLogMessage::FlushUpTo(epoch) => {
                while let Some(peek) = heap_lock.peek() {
                    if peek.epoch <= epoch {
                        heap_lock.pop();
                    } else {
                        break;
                    }
                }
            }
            ConcurrentLogMessage::AppendOperation(op) => {
                heap_lock.push(op);
            }
        }
    }

    fn fix(
        heap_lock: &mut BinaryHeap<VersionedOperation<V>>,
        message_queue: &SegQueue<ConcurrentLogMessage<V>>,
    ) {
        while let Some(message) = message_queue.pop() {
            Self::apply_message_to_heap(heap_lock, message);
        }
    }

    fn get_snapshot(&self) -> StagingShapshot<V> {
        let mut log = self.log.write();

        // fix any deferred messages
        Self::fix(&mut log, &self.deferred_messages);

        let mut added = HashSet::with_hasher(FxBuildHasher::default());
        let mut removed = HashSet::with_hasher(FxBuildHasher::default());

        for op in log.iter() {
            match &op.op {
                Operation::Insert(v) => {
                    if removed.remove(v).not() {
                        added.insert(v.clone());
                    }
                }
                Operation::Remove(v) => {
                    if added.remove(v).not() {
                        removed.insert(v.clone());
                    }
                }
            }
        }

        StagingShapshot { added, removed }
    }
}

#[derive(Default)]
struct PinnedLogLifecycleListener;

impl<K: Hash + Eq, V: Eq + Hash + Clone>
    LifecycleListener<K, TrackedConcurrentLog<V>>
    for PinnedLogLifecycleListener
{
    fn is_pinned(&self, _key: &K, value: &TrackedConcurrentLog<V>) -> bool {
        value.dirty.load(std::sync::atomic::Ordering::SeqCst) != 0
    }
}

/// Internal representation of the cache state.
#[derive(Debug)]
pub struct Repr<
    K: KeyOfSetColumn,
    C: ConcurrentSet<Element = K::Element> + 'static,
> {
    staging: TinyLFU<
        K::Key,
        TrackedConcurrentLog<K::Element>,
        PinnedLogLifecycleListener,
    >,

    cache: TinyLFU<K::Key, Arc<RwLock<Entry<C>>>>,
    single_flight: single_flight::SingleFlight<K::Key>,
}

impl<K: KeyOfSetColumn, C: ConcurrentSet<Element = K::Element> + 'static>
    Repr<K, C>
{
    /// Creates a new representation with the specified cache capacity.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn new(cap: u64, shard_amount: usize) -> Self {
        Self {
            staging: TinyLFU::new(
                2048,
                shard_amount,
                tiny_lfu::UnpinStrategy::Notify,
            ),
            cache: TinyLFU::new(
                cap as usize,
                shard_amount,
                tiny_lfu::UnpinStrategy::Poll,
            ),
            single_flight: single_flight::SingleFlight::new(shard_amount),
        }
    }

    pub(crate) fn flush_staging(
        &self,
        epoch: Epoch,
        keys: impl IntoIterator<Item = K::Key>,
    ) {
        for key in keys {
            let log = self.staging.get_map(&key, |x| {
                x.dirty.fetch_sub(1, Ordering::SeqCst);

                x.log.clone()
            });

            if let Some(log) = log {
                log.apply_message(ConcurrentLogMessage::FlushUpTo(epoch));
            }
        }
    }
}

/// A partially constructed set that exceeded the in-memory threshold.
///
/// This struct holds the elements that were loaded before the threshold
/// was exceeded, along with an iterator for the remaining database elements.
#[derive(Debug)]
pub struct Spilled<C, I> {
    half_constructed: C,
    rest_iterator: I,
}

impl<
    K: KeyOfSetColumn,
    C: ConcurrentSet<Element = K::Element> + Send + Sync + 'static,
    Db: KvDatabase,
> KeyOfSetMap<K, C> for CacheKeyOfSetMap<K, C, Db>
{
    type WriteBatch = write_behind::WriteBatch<Db>;

    async fn get(
        &self,
        key: &<K as KeyOfSetColumn>::Key,
    ) -> impl Iterator<Item = <K as KeyOfSetColumn>::Element> {
        let (entry, snapshot, spilled) = self.get_entry(key).await;

        if let Some(spilled) = spilled {
            return MergeIterator::Spilled(
                spilled,
                snapshot.into_iter_snapshot(),
            );
        }

        match entry.read().clone() {
            Entry::InMemory(set) => {
                MergeIterator::OwnedIterator(OwnedIterator::new(set, |set| {
                    set.iter()
                }))
            }
            Entry::TooLarge => {
                let db_iter = self.db.scan_members::<K>(key);
                MergeIterator::Streaming(db_iter, snapshot.into_iter_snapshot())
            }
        }
    }

    async fn insert(
        &self,
        key: <K as KeyOfSetColumn>::Key,
        element: <K as KeyOfSetColumn>::Element,
        write_batch: &mut Self::WriteBatch,
    ) {
        let updated = write_batch.put_set::<K>(
            key.clone(),
            element.clone(),
            write_behind::Operation::Insert,
            Arc::downgrade(&(self.repr.clone() as _)),
        );

        self.apply_op(
            &key,
            Operation::Insert(element),
            write_batch.epoch(),
            updated,
        );
    }

    async fn remove(
        &self,
        key: &<K as KeyOfSetColumn>::Key,
        element: &<K as KeyOfSetColumn>::Element,
        write_batch: &mut Self::WriteBatch,
    ) {
        let updated = write_batch.put_set::<K>(
            key.clone(),
            element.clone(),
            write_behind::Operation::Remove,
            Arc::downgrade(&(self.repr.clone() as _)),
        );

        self.apply_op(
            key,
            Operation::Remove(element.clone()),
            write_batch.epoch(),
            updated,
        );
    }
}

impl<
    K: KeyOfSetColumn,
    C: ConcurrentSet<Element = K::Element> + Send + Sync + 'static,
    Db: KvDatabase,
> CacheKeyOfSetMap<K, C, Db>
{
    async fn get_entry(
        &self,
        key: &K::Key,
    ) -> (
        Arc<RwLock<Entry<C>>>,
        StagingShapshot<K::Element>,
        Option<Spilled<C, Db::ScanMemberIterator<K>>>,
    ) {
        loop {
            let staging_snapshot = self.get_staging_snapshot(key);
            let mut spilled = None;

            if let Some(entry) = self.repr.cache.get(key) {
                return (entry, staging_snapshot, spilled);
            }

            let entry = self
                .repr
                .single_flight
                .wait_or_work(key, || {
                    let entry =
                        self.fetch_entry(key, &staging_snapshot, &mut spilled);

                    self.repr.cache.entry(key.clone(), |e| match e {
                        tiny_lfu::Entry::Vacant(vaccant_entry) => {
                            vaccant_entry.insert(entry.clone());
                        }
                        tiny_lfu::Entry::Occupied(_) => {
                            // Do nothing as another thread inserted an explicit
                            // value
                        }
                    });

                    entry
                })
                .await;

            if let Some(entry) = entry {
                return (entry, staging_snapshot, spilled);
            }
        }
    }

    fn fetch_entry(
        &self,
        key: &K::Key,
        snapshot: &StagingShapshot<K::Element>,
        spilled: &mut Option<Spilled<C, Db::ScanMemberIterator<K>>>,
    ) -> Arc<RwLock<Entry<C>>> {
        let new_set = C::default();
        let mut count = 0;
        let mut iter = self.db.scan_members::<K>(key);

        while let Some(element) = iter.next() {
            new_set.insert_element(element);
            count += 1;

            if count > 1024 {
                *spilled = Some(Spilled {
                    half_constructed: new_set,
                    rest_iterator: iter,
                });

                return Arc::new(RwLock::new(Entry::TooLarge));
            }
        }

        for element in &snapshot.added {
            new_set.insert_element(element.clone());
        }
        for element in &snapshot.removed {
            new_set.remove_element(element);
        }

        Arc::new(RwLock::new(Entry::InMemory(new_set)))
    }
}

/// A snapshot of staged (uncommitted) set operations.
///
/// Captures the added and removed elements that are pending commit,
/// allowing reads to see uncommitted changes.
#[derive(Debug)]
pub struct StagingShapshot<T> {
    added: HashSet<T, FxBuildHasher>,
    removed: HashSet<T, FxBuildHasher>,
}

impl<T> StagingShapshot<T> {
    /// Converts this snapshot into an iterator-based representation.
    #[must_use]
    pub fn into_iter_snapshot(self) -> StagingShapshotIntoIter<T> {
        StagingShapshotIntoIter {
            added: self.added.into_iter(),
            removed: self.removed,
        }
    }
}

/// An iterator over a staging snapshot's added elements.
///
/// Filters out elements that were subsequently removed from the staging area.
#[derive(Debug)]
pub struct StagingShapshotIntoIter<T> {
    added: std::collections::hash_set::IntoIter<T>,
    removed: HashSet<T, FxBuildHasher>,
}

impl<
    K: KeyOfSetColumn,
    C: ConcurrentSet<Element = K::Element> + Send + Sync + 'static,
    Db: KvDatabase,
> CacheKeyOfSetMap<K, C, Db>
{
    fn apply_op(
        &self,
        key: &K::Key,
        op: Operation<K::Element>,
        epoch: Epoch,
        updated: bool,
    ) {
        // Step 1: Write to Staging (The Anchor)
        // We ensure the log exists and push the op.
        let log = {
            self.repr
                .staging
                .get_map(key, |x| {
                    if updated {
                        x.dirty.fetch_add(1, Ordering::SeqCst);
                    }

                    x.log.clone()
                })
                .unwrap_or_else(|| {
                    let tracked_log = TrackedConcurrentLog {
                        log: Arc::new(ConcurrentLog::new()),
                        dirty: AtomicUsize::new(usize::from(updated)),
                    };

                    self.repr.staging.entry(key.clone(), |entry| match entry {
                        tiny_lfu::Entry::Vacant(vacant_entry) => {
                            let log = tracked_log.log.clone();
                            vacant_entry.insert(tracked_log);

                            log
                        }

                        tiny_lfu::Entry::Occupied(occupied_entry) => {
                            occupied_entry
                                .get()
                                .dirty
                                .fetch_add(1, Ordering::SeqCst);

                            occupied_entry.get().log.clone()
                        }
                    })
                })
        };

        // apply the operation to the log
        {
            log.apply_message(ConcurrentLogMessage::AppendOperation(
                VersionedOperation { op: op.clone(), epoch },
            ));
        }

        // Step 2: Update Cache (Optimization)
        // We DO NOT load from DB if missing. We only update if present.
        let Some(entry) = self.repr.cache.get(key) else {
            return;
        };

        let read_entry = entry.read();
        match &*read_entry {
            Entry::InMemory(set) => {
                let new_set = set;

                match op {
                    Operation::Insert(v) => {
                        new_set.insert_element(v);
                    }
                    Operation::Remove(v) => {
                        new_set.remove_element(&v);
                    }
                }

                // Step 3: Threshold Check
                // If it grew too big, downgrade to TooLarge
                if new_set.len() > 1024 {
                    drop(read_entry);

                    let mut write_entry = entry.write();
                    *write_entry = Entry::TooLarge;
                }
            }

            Entry::TooLarge => {}
        }
    }

    fn get_staging_snapshot(
        &self,
        key: &K::Key,
    ) -> StagingShapshot<K::Element> {
        let log = self.repr.staging.get_map(key, |x| x.log.clone());

        log.map_or_else(
            || StagingShapshot {
                added: HashSet::with_hasher(FxBuildHasher::default()),
                removed: HashSet::with_hasher(FxBuildHasher::default()),
            },
            |log| log.get_snapshot(),
        )
    }
}

/// An iterator that merges database results with staged operations.
///
/// This iterator handles three scenarios:
/// - `Spilled`: Set exceeded in-memory threshold; streams from partially loaded
///   data
/// - `OwnedIterator`: Full set is cached in memory
/// - `Streaming`: Set is too large; streams directly from database
pub enum MergeIterator<C, I, E, J> {
    /// Set data was partially loaded before exceeding the size threshold.
    Spilled(Spilled<C, I>, StagingShapshotIntoIter<E>),
    /// Full set data is available in memory.
    OwnedIterator(J),
    /// Set data is streamed directly from the database.
    Streaming(I, StagingShapshotIntoIter<E>),
}

impl<C, I, E, J> std::fmt::Debug for MergeIterator<C, I, E, J> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Spilled(_, _) => f.debug_tuple("Spilled").finish(),
            Self::OwnedIterator(_) => f.debug_tuple("OwnedIterator").finish(),
            Self::Streaming(_, _) => f.debug_tuple("Streaming").finish(),
        }
    }
}

impl<
    C: ConcurrentSet<Element = E>,
    I: Iterator<Item = E>,
    E: Eq + Hash + Send + Sync + 'static,
    J: Iterator<Item = E>,
> Iterator for MergeIterator<C, I, E, J>
{
    type Item = E;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Spilled(spilled, snapshot) => {
                // First drain from half_constructed
                if let Some(item) = spilled.half_constructed.iter().next() {
                    let item = item;

                    if snapshot.removed.contains(&item).not() {
                        return Some(item);
                    }
                }

                // Then drain from rest_iterator
                for item in spilled.rest_iterator.by_ref() {
                    if snapshot.removed.remove(&item).not() {
                        return Some(item);
                    }
                }

                // Finally drain from snapshot.added
                snapshot.added.next()
            }

            Self::OwnedIterator(iter) => iter.next(),

            Self::Streaming(db_iter, snapshot) => {
                // First drain from db_iter
                for item in db_iter.by_ref() {
                    if snapshot.removed.remove(&item).not() {
                        return Some(item);
                    }
                }

                // Finally drain from snapshot.added
                snapshot.added.next()
            }
        }
    }
}

impl<
    K: KeyOfSetColumn,
    C: ConcurrentSet<Element = K::Element> + 'static,
    Db: KvDatabase,
> KeyOfSetCache<K, Db> for Repr<K, C>
{
    fn flush(
        &self,
        epoch: Epoch,
        keys: &mut (dyn Iterator<Item = <K as KeyOfSetColumn>::Key> + Send),
    ) {
        self.flush_staging(epoch, keys);
    }
}
