//! Cached implementation of [`KeyOfSetMap`].
//!
//! This module provides [`CacheKeyOfSetMap`], which wraps a database backend
//! with caching for improved read performance on key-to-set relationships.
//!
//! # How a set is represented
//!
//! The members of a set are whatever the database holds, with the writes that
//! the database has not confirmed yet laid over it. Every key that is in
//! memory has one [`KeySet`] that holds both parts:
//!
//! - `pending` records, for each element with an unconfirmed write, whether the
//!   element will be a member once every write batch is committed.
//! - `stored` is a copy of the members in the database, if it has been loaded
//!   and is small enough to keep.
//!
//! A write only touches `pending`. `stored` only changes when the write-behind
//! reports that a write batch has been committed, which is also when the
//! entries of that batch leave `pending`. A read lays `pending` over `stored`.
//!
//! Both parts live in one cache entry and are only changed under that entry's
//! exclusive lock, so they cannot disagree with each other.

use std::{
    collections::{HashMap, HashSet, hash_map},
    hash::Hash,
    ops::Not,
    sync::Arc,
};

use fxhash::FxBuildHasher;

use crate::{
    key_of_set_map::{ConcurrentSet, KeyOfSetMap},
    kv_database::{KeyOfSetColumn, KvDatabase},
    sharded::default_shard_amount,
    single_flight,
    tiny_lfu::{self, LifecycleListener, TinyLFU},
    write_manager::write_behind::{
        self, CommittedSetWrites, Epoch, KeyOfSetCache, Operation,
    },
};

/// A cached implementation of [`KeyOfSetMap`] backed by a
/// database.
///
/// This implementation combines an in-memory cache for fast set access with a
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
        Self { repr: Arc::new(Repr::new(cap, LOADED_SET_LIMIT)), db }
    }

    /// Creates a map that only keeps the members of sets with at most
    /// `loaded_set_limit` of them in memory.
    #[cfg(test)]
    fn with_loaded_set_limit(
        cap: u64,
        db: Db,
        loaded_set_limit: usize,
    ) -> Self {
        Self { repr: Arc::new(Repr::new(cap, loaded_set_limit)), db }
    }
}

/// The largest set whose members are kept in memory by default. The members of
/// a larger set are streamed from the database on every read.
const LOADED_SET_LIMIT: usize = 1024;

/// The last operation staged for an element and the write batch that staged
/// it.
#[derive(Debug, Clone, Copy)]
struct Pending {
    operation: Operation,
    epoch: Epoch,
}

/// The copy of a set's members as the database holds them.
enum Stored<E, C> {
    /// The members have not been read from the database.
    Absent,

    /// The members are being read from the database.
    ///
    /// The scan does not see a write batch that is committed after it has
    /// started. The operations of the batches that are flushed in the meantime
    /// are therefore recorded here, in order, and replayed onto the scanned
    /// members.
    Loading(Vec<(E, Operation)>),

    /// The members, kept up to date with every flushed write batch.
    Loaded(C),

    /// The set has too many members for them to be kept in memory.
    TooLarge,
}

/// What a read of a [`KeySet`] produces.
enum Read<E> {
    /// The members of the set.
    Members(Vec<E>),

    /// The stored members have to be streamed from the database and combined
    /// with this.
    TooLarge(Overlay<E>),
}

/// What the pending writes change about the members stored in the database.
struct Overlay<E> {
    /// The elements with a pending write. Whether the database holds one of
    /// them says nothing about its membership anymore.
    overridden: HashSet<E, FxBuildHasher>,

    /// The overridden elements that are members.
    members: Vec<E>,
}

/// Everything that is known in memory about the set of one key.
struct KeySet<E, C> {
    /// The last operation staged for each element that has an operation the
    /// database has not confirmed.
    pending: HashMap<E, Pending, FxBuildHasher>,

    stored: Stored<E, C>,

    /// The number of write batches that have staged an operation on this set
    /// and have not been flushed. The entry is not evicted while there is
    /// one, since `pending` cannot be recovered from the database.
    unflushed_batches: usize,
}

fn apply<C: ConcurrentSet>(
    members: &C,
    element: C::Element,
    operation: Operation,
) {
    match operation {
        Operation::Insert => {
            members.insert_element(element);
        }
        Operation::Remove => {
            members.remove_element(&element);
        }
    }
}

impl<E: Eq + Hash + Clone, C: ConcurrentSet<Element = E>> KeySet<E, C> {
    fn new() -> Self {
        Self {
            pending: HashMap::default(),
            stored: Stored::Absent,
            unflushed_batches: 0,
        }
    }

    /// Stages an operation of the write batch of `epoch`.
    ///
    /// `first_in_batch` tells whether this is the first operation the batch
    /// stages on this set.
    fn stage(
        &mut self,
        element: E,
        operation: Operation,
        epoch: Epoch,
        first_in_batch: bool,
    ) {
        if first_in_batch {
            self.unflushed_batches += 1;
        }

        match self.pending.entry(element) {
            hash_map::Entry::Vacant(vacant) => {
                vacant.insert(Pending { operation, epoch });
            }

            hash_map::Entry::Occupied(mut occupied) => {
                // The database applies the write batches in the order of
                // their epochs, so an operation of an older batch that
                // arrives late does not decide the membership.
                if epoch >= occupied.get().epoch {
                    occupied.insert(Pending { operation, epoch });
                }
            }
        }
    }

    /// Takes in the operations of the write batch of `epoch`, which has been
    /// committed to the database.
    ///
    /// The stored members are dropped from memory if there are more than
    /// `loaded_set_limit` of them afterwards.
    fn flush(
        &mut self,
        epoch: Epoch,
        operations: impl IntoIterator<Item = (E, Operation)>,
        loaded_set_limit: usize,
    ) {
        for (element, operation) in operations {
            // a newer write batch may have staged the element again, in which
            // case the element stays pending until that batch is flushed
            let confirmed = self
                .pending
                .get(&element)
                .is_some_and(|pending| pending.epoch <= epoch);

            if confirmed {
                self.pending.remove(&element);
            }

            match &mut self.stored {
                Stored::Loaded(members) => apply(members, element, operation),
                Stored::Loading(flushed) => flushed.push((element, operation)),
                Stored::Absent | Stored::TooLarge => {}
            }
        }

        let outgrown = matches!(
            &self.stored,
            Stored::Loaded(members) if members.len() > loaded_set_limit
        );

        if outgrown {
            self.stored = Stored::TooLarge;
        }

        debug_assert!(self.unflushed_batches > 0);
        self.unflushed_batches = self.unflushed_batches.saturating_sub(1);
    }

    /// Returns whether there is nothing in this entry that is worth keeping
    /// in the cache.
    fn is_empty(&self) -> bool {
        self.unflushed_batches == 0
            && self.pending.is_empty()
            && matches!(self.stored, Stored::Absent)
    }

    /// The elements that the pending writes make members.
    fn pending_members(&self) -> impl Iterator<Item = &E> {
        self.pending
            .iter()
            .filter(|(_, pending)| pending.operation == Operation::Insert)
            .map(|(element, _)| element)
    }

    /// Lays the pending writes over `stored`, which are the members that the
    /// database holds.
    fn members(&self, stored: &C) -> Vec<E> {
        let mut members = Vec::with_capacity(stored.len() + self.pending.len());

        members.extend(
            stored
                .iter()
                .filter(|element| self.pending.contains_key(element).not()),
        );
        members.extend(self.pending_members().cloned());

        members
    }

    fn overlay(&self) -> Overlay<E> {
        Overlay {
            overridden: self.pending.keys().cloned().collect(),
            members: self.pending_members().cloned().collect(),
        }
    }

    /// Reads the set, or returns `None` if its stored members have to be
    /// loaded from the database first.
    fn read(&self) -> Option<Read<E>> {
        match &self.stored {
            Stored::Loaded(stored) => Some(Read::Members(self.members(stored))),
            Stored::TooLarge => Some(Read::TooLarge(self.overlay())),
            Stored::Absent | Stored::Loading(_) => None,
        }
    }
}

/// Keeps the sets that hold something the database cannot give back in the
/// cache: writes that have not been flushed, and the flushes recorded by a
/// load that is still running.
#[derive(Debug, Default)]
struct Irreplaceable;

impl<K, E, C> LifecycleListener<K, KeySet<E, C>> for Irreplaceable {
    fn is_pinned(&self, _key: &K, set: &KeySet<E, C>) -> bool {
        set.unflushed_batches != 0 || matches!(set.stored, Stored::Loading(_))
    }
}

/// Internal representation of the cache state.
#[derive(Debug)]
struct Repr<K: KeyOfSetColumn, C: ConcurrentSet<Element = K::Element> + 'static>
{
    /// A set is read under the shared lock of its entry and changed under
    /// the exclusive one.
    sets: TinyLFU<K::Key, KeySet<K::Element, C>, Irreplaceable>,

    /// Makes sure that a set is loaded by one task at a time.
    single_flight: single_flight::SingleFlight<K::Key>,

    /// The largest set whose members are kept in memory.
    loaded_set_limit: usize,
}

impl<K: KeyOfSetColumn, C: ConcurrentSet<Element = K::Element> + 'static>
    Repr<K, C>
{
    #[allow(clippy::cast_possible_truncation)]
    fn new(cap: u64, loaded_set_limit: usize) -> Self {
        Self {
            sets: TinyLFU::new(
                cap as usize,
                tiny_lfu::UnpinStrategy::Poll,
                tiny_lfu::MaintenanceMode::Piggyback,
            ),
            single_flight: single_flight::SingleFlight::new(
                default_shard_amount(),
            ),
            loaded_set_limit,
        }
    }

    /// Stages an operation of the write batch of `epoch` on the set of `key`.
    ///
    /// The database is not read: the set does not have to be loaded for an
    /// operation to be staged on it.
    fn stage(
        &self,
        key: K::Key,
        element: K::Element,
        operation: Operation,
        epoch: Epoch,
        first_in_batch: bool,
    ) {
        self.sets.entry(key, |entry| match entry {
            tiny_lfu::Entry::Vacant(vacant) => {
                let mut set = KeySet::new();
                set.stage(element, operation, epoch, first_in_batch);

                vacant.insert(set);
            }

            tiny_lfu::Entry::Occupied(mut occupied) => {
                occupied.get_mut().stage(
                    element,
                    operation,
                    epoch,
                    first_in_batch,
                );
            }
        });
    }
}

impl<
    K: KeyOfSetColumn,
    C: ConcurrentSet<Element = K::Element> + 'static,
    Db: KvDatabase,
> KeyOfSetCache<K, Db> for Repr<K, C>
{
    fn flush(&self, epoch: Epoch, writes: &mut CommittedSetWrites<'_, K>) {
        for (key, operations) in writes {
            self.sets.entry(key, |entry| {
                // the entry cannot have been evicted: it has had an unflushed
                // write batch since that batch's first operation on it
                let tiny_lfu::Entry::Occupied(mut occupied) = entry else {
                    panic!("entry for flushed write batch was evicted");
                };

                let set = occupied.get_mut();
                set.flush(epoch, operations, self.loaded_set_limit);

                // an entry with nothing pending and nothing loaded would only
                // take up room in the cache
                if set.is_empty() {
                    drop(occupied.remove());
                }
            });
        }
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
        let read = loop {
            let read = self.repr.sets.get_map(key, KeySet::read).flatten();

            if let Some(read) = read {
                break read;
            }

            // Whoever gets to load the set reads it as part of the load. The
            // others wait for the load to finish and then look again.
            let loaded = self
                .repr
                .single_flight
                .wait_or_work(key, || self.load(key))
                .await;

            if let Some(read) = loaded {
                break read;
            }
        };

        match read {
            Read::Members(members) => Members::Loaded(members.into_iter()),

            // The overlay was taken before this scan starts. A write batch
            // that is flushed in between is then in both, which is harmless,
            // and never in neither.
            Read::TooLarge(overlay) => Members::Streaming {
                stored: self.db.scan_members::<K>(key),
                overridden: overlay.overridden,
                pending_members: overlay.members.into_iter(),
            },
        }
    }

    async fn insert(
        &self,
        key: <K as KeyOfSetColumn>::Key,
        element: <K as KeyOfSetColumn>::Element,
        write_batch: &mut Self::WriteBatch,
    ) {
        let first_in_batch = write_batch.put_set::<K>(
            key.clone(),
            element.clone(),
            Operation::Insert,
            Arc::downgrade(&(self.repr.clone() as _)),
        );

        self.repr.stage(
            key,
            element,
            Operation::Insert,
            write_batch.epoch(),
            first_in_batch,
        );
    }

    async fn remove(
        &self,
        key: &<K as KeyOfSetColumn>::Key,
        element: &<K as KeyOfSetColumn>::Element,
        write_batch: &mut Self::WriteBatch,
    ) {
        let first_in_batch = write_batch.put_set::<K>(
            key.clone(),
            element.clone(),
            Operation::Remove,
            Arc::downgrade(&(self.repr.clone() as _)),
        );

        self.repr.stage(
            key.clone(),
            element.clone(),
            Operation::Remove,
            write_batch.epoch(),
            first_in_batch,
        );
    }
}

impl<
    K: KeyOfSetColumn,
    C: ConcurrentSet<Element = K::Element> + Send + Sync + 'static,
    Db: KvDatabase,
> CacheKeyOfSetMap<K, C, Db>
{
    /// Loads the stored members of the set of `key` from the database and
    /// reads the set.
    ///
    /// Only one load of a key may run at a time.
    fn load(&self, key: &K::Key) -> Read<K::Element> {
        loop {
            if let Some(read) = self.start_load(key) {
                return read;
            }

            let scanned = C::default();
            let mut too_large = false;

            for (count, element) in self.db.scan_members::<K>(key).enumerate() {
                if count == self.repr.loaded_set_limit {
                    too_large = true;
                    break;
                }

                scanned.insert_element(element);
            }

            if let Some(read) = self.finish_load(key, scanned, too_large) {
                return read;
            }

            // The entry that was recording the flushes of this load is gone.
            // The scanned members cannot be trusted without them, so the load
            // starts over.
        }
    }

    /// Marks the set of `key` as being loaded, so that every write batch the
    /// scan can miss is recorded when it is flushed. This has to happen
    /// before the scan starts.
    ///
    /// Returns what the set reads as if it turns out not to need loading.
    fn start_load(&self, key: &K::Key) -> Option<Read<K::Element>> {
        self.repr.sets.entry(key.clone(), |entry| match entry {
            tiny_lfu::Entry::Vacant(vacant) => {
                let mut set = KeySet::new();
                set.stored = Stored::Loading(Vec::new());

                vacant.insert(set);

                None
            }

            tiny_lfu::Entry::Occupied(mut occupied) => {
                let set = occupied.get_mut();
                let read = set.read();

                // `Loading` can only be what a load that never finished has
                // left behind, since the loads of a key do not overlap.
                if read.is_none() {
                    set.stored = Stored::Loading(Vec::new());
                }

                read
            }
        })
    }

    /// Completes a load with the members that were scanned from the database
    /// and reads the set.
    ///
    /// Returns `None` if the entry is no longer the one [`Self::start_load`]
    /// marked. That entry is not evicted, so this is not expected to happen.
    fn finish_load(
        &self,
        key: &K::Key,
        scanned: C,
        too_large: bool,
    ) -> Option<Read<K::Element>> {
        let limit = self.repr.loaded_set_limit;

        self.repr.sets.entry(key.clone(), |entry| {
            let tiny_lfu::Entry::Occupied(mut occupied) = entry else {
                return None;
            };

            let set = occupied.get_mut();

            match std::mem::replace(&mut set.stored, Stored::Absent) {
                Stored::Loading(flushed) => {
                    if too_large.not() {
                        for (element, operation) in flushed {
                            apply(&scanned, element, operation);
                        }
                    }

                    if too_large || scanned.len() > limit {
                        set.stored = Stored::TooLarge;

                        Some(Read::TooLarge(set.overlay()))
                    } else {
                        let members = set.members(&scanned);
                        set.stored = Stored::Loaded(scanned);

                        Some(Read::Members(members))
                    }
                }

                stored => {
                    set.stored = stored;

                    None
                }
            }
        })
    }
}

/// The members of a set.
enum Members<E, I> {
    /// The stored members were in memory.
    Loaded(std::vec::IntoIter<E>),

    /// The stored members are streamed from the database.
    Streaming {
        stored: I,
        overridden: HashSet<E, FxBuildHasher>,
        pending_members: std::vec::IntoIter<E>,
    },
}

impl<E: Eq + Hash, I: Iterator<Item = E>> Iterator for Members<E, I> {
    type Item = E;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Loaded(members) => members.next(),

            Self::Streaming { stored, overridden, pending_members } => stored
                .find(|element| overridden.contains(element).not())
                .or_else(|| pending_members.next()),
        }
    }
}

#[cfg(test)]
mod test;
