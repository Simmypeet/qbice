//! Cached implementation of [`KeyOfSetMap`].
//!
//! This module provides [`CacheKeyOfSetMap`], which wraps a database backend
//! with caching for improved read performance on key-to-set relationships.

use std::{
    collections::{BinaryHeap, HashSet},
    hash::Hash,
    ops::Not,
    sync::Arc,
};

use dashmap::DashMap;
use fxhash::FxBuildHasher;
use moka::ops::compute;
use parking_lot::RwLock;

use crate::{
    key_of_set_map::{ConcurrentSet, KeyOfSetMap, OwnedIterator},
    kv_database::{KeyOfSetColumn, KvDatabase},
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
    /// # Parameters
    ///
    /// - `cap`: The maximum number of key entries to cache.
    /// - `db`: The database backend for persistence.
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

type ConcurrentLog<V> = Arc<RwLock<BinaryHeap<VersionedOperation<V>>>>;

/// Internal representation of the cache state.
#[derive(Debug)]
pub struct Repr<
    K: KeyOfSetColumn,
    C: ConcurrentSet<Element = K::Element> + 'static,
> {
    staging: Arc<DashMap<K::Key, ConcurrentLog<K::Element>, FxBuildHasher>>,
    moka: moka::future::Cache<K::Key, Entry<C>>,
}

impl<K: KeyOfSetColumn, C: ConcurrentSet<Element = K::Element> + 'static>
    Repr<K, C>
{
    /// Creates a new representation with the specified cache capacity.
    #[must_use]
    pub fn new(cap: u64) -> Self {
        Self {
            staging: Arc::new(DashMap::default()),
            moka: moka::future::Cache::builder().max_capacity(cap).build(),
        }
    }

    pub(crate) fn flush_staging(
        &self,
        epoch: Epoch,
        keys: impl IntoIterator<Item = K::Key>,
    ) {
        for key in keys {
            let Some(log) = self.staging.get(&key).map(|x| x.clone()) else {
                continue;
            };

            // drain ops up to epoch
            let mut log = log.write();
            while let Some(op) = log.peek() {
                if op.epoch <= epoch {
                    log.pop();
                } else {
                    break;
                }
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
        let mut spilled = None;
        let snapshot = self.get_staging_snapshot(key);

        let result = self
            .repr
            .moka
            .get_with(key.clone(), async {
                let new_set = C::default();
                let mut count = 0;
                let mut iter = self.db.scan_members::<K>(key);

                while let Some(element) = iter.next() {
                    new_set.insert_element(element);
                    count += 1;

                    if count > 1024 {
                        spilled = Some(Spilled {
                            half_constructed: new_set,
                            rest_iterator: iter,
                        });

                        return Entry::TooLarge;
                    }
                }

                for element in &snapshot.added {
                    new_set.insert_element(element.clone());
                }
                for element in &snapshot.removed {
                    new_set.remove_element(element);
                }

                Entry::InMemory(new_set)
            })
            .await;

        if let Some(spilled) = spilled {
            return MergeIterator::Spilled(
                spilled,
                snapshot.into_iter_snapshot(),
            );
        }

        match result {
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
        write_batch.put_set::<K>(
            key.clone(),
            element.clone(),
            write_behind::Operation::Insert,
            self.repr.clone(),
        );

        self.apply_op(key, Operation::Insert(element), write_batch.epoch())
            .await;
    }

    async fn remove(
        &self,
        key: &<K as KeyOfSetColumn>::Key,
        element: &<K as KeyOfSetColumn>::Element,
        write_batch: &mut Self::WriteBatch,
    ) {
        self.apply_op(
            key.clone(),
            Operation::Remove(element.clone()),
            write_batch.epoch(),
        )
        .await;
    }
}

impl<
    K: KeyOfSetColumn,
    C: ConcurrentSet<Element = K::Element> + Send + Sync + 'static,
    Db: KvDatabase,
> CacheKeyOfSetMap<K, C, Db>
{
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
    async fn apply_op(
        &self,
        key: K::Key,
        op: Operation<K::Element>,
        epoch: Epoch,
    ) {
        // Step 1: Write to Staging (The Anchor)
        // We ensure the log exists and push the op.
        let entry = {
            self.repr.staging.get(&key).map_or_else(
                || {
                    let new_log: ConcurrentLog<K::Element> =
                        Arc::new(RwLock::new(BinaryHeap::new()));

                    match self.repr.staging.entry(key.clone()) {
                        dashmap::mapref::entry::Entry::Occupied(occ) => {
                            occ.get().clone()
                        }
                        dashmap::mapref::entry::Entry::Vacant(vac) => {
                            vac.insert(new_log.clone());

                            new_log
                        }
                    }
                },
                |x| x.clone(),
            )
        };

        {
            let mut log = entry.write();
            log.push(VersionedOperation { op: op.clone(), epoch });
        }

        // Step 2: Update Cache (Optimization)
        // We DO NOT load from DB if missing. We only update if present.
        self.repr
            .moka
            .entry(key)
            .and_compute_with(async move |entry| {
                let Some(entry) = entry else {
                    // Not present in cache; do nothing.
                    return compute::Op::Nop;
                };

                match entry.into_value() {
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
                            compute::Op::Put(Entry::TooLarge)
                        } else {
                            compute::Op::Put(Entry::InMemory(new_set))
                        }
                    }
                    Entry::TooLarge => {
                        // Do nothing. It stays "TooLarge".
                        // Reads will force a stream from DB + Staging.

                        compute::Op::Nop
                    }
                }
            })
            .await;
    }

    fn get_staging_snapshot(
        &self,
        key: &K::Key,
    ) -> StagingShapshot<K::Element> {
        let mut added = HashSet::with_hasher(FxBuildHasher::default());
        let mut removed = HashSet::with_hasher(FxBuildHasher::default());

        if let Some(log) = self.repr.staging.get(key).map(|x| x.clone()) {
            let log = log.read();

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
        }

        StagingShapshot { added, removed }
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
