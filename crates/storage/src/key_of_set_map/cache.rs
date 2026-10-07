//! Cached implementation of [`KeyOfSetMap`].
//!
//! This module provides [`CacheKeyOfSetMap`], which wraps a database backend
//! with caching for improved read performance on key-to-set relationships.

use std::{
    collections::{BTreeMap, HashSet},
    hash::{BuildHasher, Hash},
    ops::Not,
    sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
};

use crossbeam::queue::SegQueue;
use fxhash::FxBuildHasher;
use parking_lot::RwLock;

use crate::{
    key_of_set_map::{ConcurrentSet, KeyOfSetMap, OwnedIterator},
    kv_database::{KeyOfSetColumn, KvDatabase},
    sharded::default_shard_amount,
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
    pub fn new(cap: u64, db: Db) -> Self {
        Self { repr: Arc::new(Repr::new(cap)), db }
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

#[derive(Debug)]
struct TrackedConcurrentLog<V> {
    log: Arc<ConcurrentLog<V>>,
    dirty: AtomicUsize,
}

enum ConcurrentLogMessage<V> {
    FlushUpTo(Epoch),
    AppendOperation(VersionedOperation<V>),
}

/// The staged operations of one set, kept in the order in which the database
/// applies them: by the epoch of their write batch first, then by the order in
/// which they were appended.
#[derive(Debug)]
struct Log<V> {
    operations: BTreeMap<(Epoch, u64), Operation<V>>,

    /// The position given to the next appended operation.
    next_sequence: u64,
}

impl<V> Log<V> {
    fn apply_message(&mut self, message: ConcurrentLogMessage<V>) {
        match message {
            ConcurrentLogMessage::FlushUpTo(epoch) => {
                while self
                    .operations
                    .first_key_value()
                    .is_some_and(|((oldest, _), _)| *oldest <= epoch)
                {
                    self.operations.pop_first();
                }
            }
            ConcurrentLogMessage::AppendOperation(op) => {
                self.operations.insert((op.epoch, self.next_sequence), op.op);
                self.next_sequence += 1;
            }
        }
    }
}

#[derive(Debug)]
struct ConcurrentLog<V> {
    log: RwLock<Log<V>>,
    deferred_messages: SegQueue<ConcurrentLogMessage<V>>,
}

impl<V: Eq + Hash + Clone> ConcurrentLog<V> {
    const fn new() -> Self {
        Self {
            log: RwLock::new(Log {
                operations: BTreeMap::new(),
                next_sequence: 0,
            }),
            deferred_messages: SegQueue::new(),
        }
    }

    fn apply_message(&self, op: ConcurrentLogMessage<V>) {
        let Some(mut lock) = self.log.try_write() else {
            self.deferred_messages.push(op);

            return;
        };

        Self::fix(&mut lock, &self.deferred_messages);

        lock.apply_message(op);
    }

    fn fix(
        log: &mut Log<V>,
        message_queue: &SegQueue<ConcurrentLogMessage<V>>,
    ) {
        while let Some(message) = message_queue.pop() {
            log.apply_message(message);
        }
    }

    /// Summarizes the staged operations as the elements they add to and
    /// remove from the set stored in the database.
    ///
    /// The last operation staged for an element decides which of the two it
    /// ends up in. That makes the snapshot safe to merge with any state of the
    /// database that the write-behind can produce: replaying an operation the
    /// database has already applied gives the same membership again.
    fn get_snapshot(&self) -> StagingShapshot<V> {
        let mut log = self.log.write();

        // fix any deferred messages
        Self::fix(&mut log, &self.deferred_messages);

        let mut added = HashSet::with_hasher(FxBuildHasher::default());
        let mut removed = HashSet::with_hasher(FxBuildHasher::default());

        // NOTE: the `.values()` iteratees the values in the order of their
        // keys. this means the earliest operations are applied first,
        // in chronological order.
        for op in log.operations.values() {
            match op {
                Operation::Insert(v) => {
                    removed.remove(v);
                    added.insert(v.clone());
                }
                Operation::Remove(v) => {
                    added.remove(v);
                    removed.insert(v.clone());
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

    /// Versions of the staging area, striped by key. A stripe's version
    /// changes whenever an operation is staged for one of its keys, which is
    /// how a load notices a write that overlapped it.
    staging_versions: Box<[AtomicU64]>,
}

/// `2^STAGING_VERSION_BITS` versions are kept per map.
const STAGING_VERSION_BITS: u32 = 12;

impl<K: KeyOfSetColumn, C: ConcurrentSet<Element = K::Element> + 'static>
    Repr<K, C>
{
    /// Creates a new representation with the specified cache capacity.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn new(cap: u64) -> Self {
        Self {
            staging: TinyLFU::new(
                2048,
                tiny_lfu::UnpinStrategy::Notify,
                tiny_lfu::MaintenanceMode::Piggyback,
            ),
            cache: TinyLFU::new(
                cap as usize,
                tiny_lfu::UnpinStrategy::Poll,
                tiny_lfu::MaintenanceMode::Piggyback,
            ),
            single_flight: single_flight::SingleFlight::new(
                default_shard_amount(),
            ),
            staging_versions: (0..1usize << STAGING_VERSION_BITS)
                .map(|_| AtomicU64::new(0))
                .collect(),
        }
    }

    /// Returns the staging version shared by `key` and the other keys of its
    /// stripe.
    #[allow(clippy::cast_possible_truncation)]
    fn staging_version(&self, key: &K::Key) -> &AtomicU64 {
        let hash = FxBuildHasher::default().hash_one(key);

        // the top bits, which are the best mixed ones of this hasher
        let stripe = (hash >> (u64::BITS - STAGING_VERSION_BITS)) as usize;

        &self.staging_versions[stripe]
    }

    pub(crate) fn flush_staging(
        &self,
        epoch: Epoch,
        keys: impl IntoIterator<Item = K::Key>,
    ) {
        for key in keys {
            let result = self.staging.get_map(&key, |x| {
                let count = x.dirty.fetch_sub(1, Ordering::SeqCst);

                (x.log.clone(), count == 1)
            });

            if let Some((log, unpinned)) = result {
                log.apply_message(ConcurrentLogMessage::FlushUpTo(epoch));

                if unpinned {
                    self.staging.unpin(key);
                }
            }
        }
    }
}

/// A partially constructed set that exceeded the in-memory threshold.
///
/// This struct holds the elements that were loaded before the threshold
/// was exceeded, along with an iterator for the remaining database elements.
pub struct Spilled<C: ConcurrentSet + 'static, I> {
    half_constructed: OwnedIterator<C>,
    rest_iterator: I,
}

impl<C: ConcurrentSet + 'static, I> std::fmt::Debug for Spilled<C, I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Spilled").finish_non_exhaustive()
    }
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
            // Read before the snapshot is taken, so that an operation the
            // snapshot misses is certain to change it.
            let staging_version =
                self.repr.staging_version(key).load(Ordering::SeqCst);

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

                    // An operation staged while the set was being loaded is
                    // in neither the snapshot nor the database, and its writer
                    // may have found no cached set to apply it to. The set
                    // that was just cached could be missing it for good, so it
                    // must not stay cached.
                    if self.repr.staging_version(key).load(Ordering::SeqCst)
                        != staging_version
                    {
                        self.uncache_loaded_set(key, &entry);
                    }

                    entry
                })
                .await;

            if let Some(entry) = entry {
                return (entry, staging_snapshot, spilled);
            }
        }
    }

    /// Removes `entry` from the cache if it is still the cached entry of `key`
    /// and holds a loaded set.
    ///
    /// A [`Entry::TooLarge`] entry is left alone: it holds no elements that
    /// could be out of date, and every read of it merges the staged operations
    /// anyway.
    fn uncache_loaded_set(&self, key: &K::Key, entry: &Arc<RwLock<Entry<C>>>) {
        let removed = self.repr.cache.entry(key.clone(), |e| match e {
            tiny_lfu::Entry::Occupied(occupied)
                if Arc::ptr_eq(occupied.get(), entry)
                    && matches!(&*entry.read(), Entry::InMemory(_)) =>
            {
                Some(occupied.remove())
            }

            tiny_lfu::Entry::Occupied(_) | tiny_lfu::Entry::Vacant(_) => None,
        });

        // drop the set outside of the cache's lock
        drop(removed);
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
                    half_constructed: OwnedIterator::new(new_set, |x| x.iter()),
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
            added: self.added,
            removed: self.removed,
            remaining_added: None,
        }
    }
}

/// A staging snapshot that is being merged into the elements read from the
/// database.
///
/// It filters the database's elements first and yields the staged insertions
/// once the database is exhausted.
#[derive(Debug)]
pub struct StagingShapshotIntoIter<T> {
    added: HashSet<T, FxBuildHasher>,
    removed: HashSet<T, FxBuildHasher>,

    /// Drains `added` once the database's elements are exhausted.
    remaining_added: Option<std::collections::hash_set::IntoIter<T>>,
}

impl<T: Eq + Hash> StagingShapshotIntoIter<T> {
    /// Returns whether an element read from the database should be yielded.
    ///
    /// Elements staged for removal are dropped. Elements staged for insertion
    /// are dropped as well, since [`Self::next_added`] yields them; the
    /// database can already contain them when their write batch has been
    /// committed but not yet flushed from the staging area.
    fn keeps(&self, element: &T) -> bool {
        self.removed.contains(element).not()
            && self.added.contains(element).not()
    }

    /// Yields the staged insertions, after the database's elements.
    fn next_added(&mut self) -> Option<T> {
        self.remaining_added
            .get_or_insert_with(|| std::mem::take(&mut self.added).into_iter())
            .next()
    }
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
                            // REVIEW: Don't we need to check if the `unpdated`
                            // flag is true first?
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

        // Tell the loads that are in flight that the log has moved. This must
        // come after the append and before the cache lookup below: a load
        // whose snapshot misses the operation then sees the new version, and
        // a load that publishes its set later is seen by the lookup.
        self.repr.staging_version(key).fetch_add(1, Ordering::SeqCst);

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
pub enum MergeIterator<C: ConcurrentSet + 'static, I, E, J> {
    /// Set data was partially loaded before exceeding the size threshold.
    Spilled(Spilled<C, I>, StagingShapshotIntoIter<E>),
    /// Full set data is available in memory.
    OwnedIterator(J),
    /// Set data is streamed directly from the database.
    Streaming(I, StagingShapshotIntoIter<E>),
}

impl<C: ConcurrentSet + 'static, I, E, J> std::fmt::Debug
    for MergeIterator<C, I, E, J>
{
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
                for item in spilled.half_constructed.by_ref() {
                    if snapshot.keeps(&item) {
                        return Some(item);
                    }
                }

                // Then drain from rest_iterator
                for item in spilled.rest_iterator.by_ref() {
                    if snapshot.keeps(&item) {
                        return Some(item);
                    }
                }

                // Finally drain from snapshot.added
                snapshot.next_added()
            }

            Self::OwnedIterator(iter) => iter.next(),

            Self::Streaming(db_iter, snapshot) => {
                // First drain from db_iter
                for item in db_iter.by_ref() {
                    if snapshot.keeps(&item) {
                        return Some(item);
                    }
                }

                // Finally drain from snapshot.added
                snapshot.next_added()
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

#[cfg(test)]
mod test;
