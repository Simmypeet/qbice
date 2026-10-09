//! Cached implementation of [`KeyOfSetMap`].
//!
//! This module provides [`CacheKeyOfSetMap`], which wraps a database backend
//! with caching for improved read performance on key-to-set relationships.
//!
//! # How a set is represented
//!
//! The members of a set are whatever the database holds, with the writes that
//! the database does not hold yet laid over it. Every key that is in memory
//! has one [`KeySet`] that holds both parts:
//!
//! - `pending` records, for each element that has been written, the last
//!   operation on it and the epoch of the write batch that staged the
//!   operation.
//! - `stored` is a copy of the members in the database, if it has been loaded
//!   and is small enough to keep.
//!
//! A write only touches `pending`. A read lays `pending` over `stored`.
//!
//! Both parts live in one cache entry and are only changed under that entry's
//! exclusive lock, so they cannot disagree with each other.
//!
//! # How a set finds out about a commit
//!
//! It is not told. The write manager only publishes which epochs have been
//! committed, and a pending operation is in the database once its epoch is
//! one of them.
//!
//! A read does not have to know whether that has happened. An element with a
//! pending operation reads as what the operation makes it, and that is also
//! what the database holds for the element once the operation has been
//! committed: write batches are committed in the order of their epochs, and
//! `pending` keeps the operation of the newest epoch.
//!
//! What a committed operation still takes up is memory. A set is therefore
//! *settled* from time to time: the pending operations that have been
//! committed are folded into `stored`, if that is a copy of the members, and
//! dropped. Without a copy there is nothing to fold them into and nothing is
//! lost by dropping them, because the next scan of the database sees them.
//! The one scan that may not see them is a scan that is already running, so a
//! set is not settled while it is being loaded.
//!
//! A set whose operations have all been committed holds nothing that the
//! database cannot give back, and the cache may evict it.

use std::{
    collections::{HashMap, HashSet, hash_map},
    hash::Hash,
    ops::Not,
};

use fxhash::FxBuildHasher;

use crate::{
    key_of_set_map::{ConcurrentSet, KeyOfSetMap},
    kv_database::{KeyOfSetColumn, KvDatabase},
    s3_fifo::{self, LifecycleListener, S3Fifo},
    sharded::default_shard_amount,
    single_flight,
    write_manager::write_behind::{self, CommittedEpochs, Epoch, Operation},
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
    repr: Repr<K, C>,
    db: Db,
}

impl<
    K: KeyOfSetColumn,
    C: ConcurrentSet<Element = K::Element> + 'static,
    Db: KvDatabase,
> CacheKeyOfSetMap<K, C, Db>
{
    /// Creates a new cached key-of-set map with the specified capacity, for
    /// the write manager whose committed epochs are `committed`.
    pub(crate) fn new(cap: u64, db: Db, committed: CommittedEpochs) -> Self {
        Self { repr: Repr::new(cap, LOADED_SET_LIMIT, committed), db }
    }

    /// Creates a map that only keeps the members of sets with at most
    /// `loaded_set_limit` of them in memory.
    #[cfg(test)]
    fn with_loaded_set_limit(
        cap: u64,
        db: Db,
        loaded_set_limit: usize,
        committed: CommittedEpochs,
    ) -> Self {
        Self { repr: Repr::new(cap, loaded_set_limit, committed), db }
    }
}

/// The largest set whose members are kept in memory by default. The members of
/// a larger set are streamed from the database on every read.
const LOADED_SET_LIMIT: usize = 1024;

/// The fewest pending operations a set has when their number makes it settle.
const SETTLE_AT_LEAST: usize = 4;

/// The last operation staged for an element and the write batch that staged
/// it.
#[derive(Debug, Clone, Copy)]
struct Pending {
    operation: Operation,
    epoch: Epoch,
}

/// The copy of a set's members as the database holds them.
enum Stored<C> {
    /// The members have not been read from the database.
    NotLoaded,

    /// The members are being read from the database.
    ///
    /// The scan may or may not see a write batch that is committed after it
    /// has started. Every pending operation is therefore kept until the scan
    /// is done, when all of them are laid over the scanned members.
    Loading,

    /// The members as the scan that loaded them found them, with the pending
    /// operations folded in that have been settled since.
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
    /// The last operation staged for each element, until the operation is
    /// settled.
    pending: HashMap<E, Pending, FxBuildHasher>,

    stored: Stored<C>,

    /// The newest epoch that has staged an operation on this set. Until it
    /// has been committed, `pending` holds operations that the database
    /// cannot give back, and the entry is not evicted.
    last_staged: Option<Epoch>,

    /// The number of pending operations at which the set is settled next.
    settle_at: usize,
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
            stored: Stored::NotLoaded,
            last_staged: None,
            settle_at: SETTLE_AT_LEAST,
        }
    }

    /// Stages an operation of the write batch of `epoch`.
    fn stage(&mut self, element: E, operation: Operation, epoch: Epoch) {
        self.last_staged = self.last_staged.max(Some(epoch));

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

    /// Returns whether enough operations are pending for it to be worth
    /// looking for the ones that have been committed.
    fn is_due(&self) -> bool { self.pending.len() >= self.settle_at }

    /// Folds the pending operations that have been committed into the stored
    /// members and drops them.
    ///
    /// The stored members are dropped from memory if there are more than
    /// `loaded_set_limit` of them afterwards.
    fn settle(&mut self, committed: &CommittedEpochs, loaded_set_limit: usize) {
        // the scan may have missed what has been committed since it started
        if matches!(self.stored, Stored::Loading) {
            return;
        }

        let stored = &self.stored;

        self.pending.retain(|element, pending| {
            if committed.contains(pending.epoch).not() {
                return true;
            }

            // Without a copy of the members, the operation is just dropped:
            // it is in the database, where the next scan finds it.
            if let Stored::Loaded(members) = stored {
                apply(members, element.clone(), pending.operation);
            }

            false
        });

        let outgrown = matches!(
            &self.stored,
            Stored::Loaded(members) if members.len() > loaded_set_limit
        );

        if outgrown {
            self.stored = Stored::TooLarge;
        }

        // Looking through the pending operations takes as long as there are
        // of them, so the next look waits until there are twice as many. A
        // set that is written to over and over then spends a constant amount
        // of time on this per write.
        self.settle_at = (self.pending.len() * 2).max(SETTLE_AT_LEAST);

        if self.pending.capacity() > self.settle_at * 4 {
            self.pending.shrink_to(self.settle_at);
        }
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
            Stored::NotLoaded | Stored::Loading => None,
        }
    }
}

/// Keeps the sets that hold something the database cannot give back in the
/// cache: operations that have not been committed, and the pending
/// operations of a set that is being loaded.
#[derive(Debug)]
struct Irreplaceable {
    /// The committed epochs of the write manager that writes to the cache.
    committed: CommittedEpochs,
}

impl<K, E, C> LifecycleListener<K, KeySet<E, C>> for Irreplaceable {
    fn is_pinned(&self, _key: &K, set: &KeySet<E, C>) -> bool {
        // write batches are committed in the order of their epochs, so the
        // newest one is the last whose operations reach the database
        let uncommitted = set
            .last_staged
            .is_some_and(|epoch| self.committed.contains(epoch).not());

        uncommitted || matches!(set.stored, Stored::Loading)
    }
}

/// Internal representation of the cache state.
#[derive(Debug)]
struct Repr<K: KeyOfSetColumn, C: ConcurrentSet<Element = K::Element> + 'static>
{
    /// A set is read under the shared lock of its entry and changed under
    /// the exclusive one.
    sets: S3Fifo<K::Key, KeySet<K::Element, C>, Irreplaceable>,

    /// Makes sure that a set is loaded by one task at a time.
    single_flight: single_flight::SingleFlight<K::Key>,

    /// The largest set whose members are kept in memory.
    loaded_set_limit: usize,
}

impl<K: KeyOfSetColumn, C: ConcurrentSet<Element = K::Element> + 'static>
    Repr<K, C>
{
    #[allow(clippy::cast_possible_truncation)]
    fn new(
        cap: u64,
        loaded_set_limit: usize,
        committed: CommittedEpochs,
    ) -> Self {
        Self {
            sets: S3Fifo::with_lifecycle_listener(
                cap as usize,
                Irreplaceable { committed },
            ),
            single_flight: single_flight::SingleFlight::new(
                default_shard_amount(),
            ),
            loaded_set_limit,
        }
    }

    /// The committed epochs of the write manager that writes to the cache.
    const fn committed(&self) -> &CommittedEpochs {
        &self.sets.lifecycle_listener().committed
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
    ) {
        self.sets.entry(key, |entry| match entry {
            s3_fifo::Entry::Vacant(vacant) => {
                let mut set = KeySet::new();
                set.stage(element, operation, epoch);

                vacant.insert(set);
            }

            s3_fifo::Entry::Occupied(mut occupied) => {
                let set = occupied.get_mut();
                set.stage(element, operation, epoch);

                if set.is_due() {
                    set.settle(self.committed(), self.loaded_set_limit);
                }
            }
        });
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
            // that is committed in between is then in both, which is
            // harmless, and never in neither: an operation only leaves the
            // pending ones after it has been committed.
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
        write_batch.put_set::<K>(&key, &element, Operation::Insert);

        self.repr.stage(key, element, Operation::Insert, write_batch.epoch());
    }

    async fn remove(
        &self,
        key: &<K as KeyOfSetColumn>::Key,
        element: &<K as KeyOfSetColumn>::Element,
        write_batch: &mut Self::WriteBatch,
    ) {
        write_batch.put_set::<K>(key, element, Operation::Remove);

        self.repr.stage(
            key.clone(),
            element.clone(),
            Operation::Remove,
            write_batch.epoch(),
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

        self.finish_load(key, scanned, too_large)
    }

    /// Marks the set of `key` as being loaded, so that it keeps every
    /// operation the scan can miss. This has to happen before the scan
    /// starts.
    ///
    /// Returns what the set reads as if it turns out not to need loading.
    fn start_load(&self, key: &K::Key) -> Option<Read<K::Element>> {
        self.repr.sets.entry(key.clone(), |entry| match entry {
            s3_fifo::Entry::Vacant(vacant) => {
                let mut set = KeySet::new();
                set.stored = Stored::Loading;

                vacant.insert(set);

                None
            }

            s3_fifo::Entry::Occupied(mut occupied) => {
                let set = occupied.get_mut();
                let read = set.read();

                match set.stored {
                    Stored::NotLoaded => {
                        // The scan has not started, so it sees whatever has
                        // been committed by now.
                        set.settle(
                            self.repr.committed(),
                            self.repr.loaded_set_limit,
                        );

                        set.stored = Stored::Loading;
                    }

                    Stored::Loaded(_) | Stored::TooLarge => assert!(
                        read.is_some(),
                        "the set is loaded, so it must read as something"
                    ),

                    Stored::Loading => unreachable!(
                        "there should be only one 'load' of a key at a time"
                    ),
                }

                read
            }
        })
    }

    /// Completes a load with the members that were scanned from the database
    /// and reads the set.
    fn finish_load(
        &self,
        key: &K::Key,
        scanned: C,
        too_large: bool,
    ) -> Read<K::Element> {
        self.repr.sets.entry(key.clone(), |entry| {
            let s3_fifo::Entry::Occupied(mut occupied) = entry else {
                panic!("the entry should've been pinned and not evicted");
            };

            let set = occupied.get_mut();

            assert!(
                matches!(set.stored, Stored::Loading),
                "the set should be loading while the scan is running"
            );

            // Every operation that the scan can have missed is still
            // pending, since the set has not been settled during the scan.
            // The scan having seen one of them changes nothing: the element
            // reads as what its last operation makes it either way.
            set.stored = if too_large {
                Stored::TooLarge
            } else {
                Stored::Loaded(scanned)
            };

            set.settle(self.repr.committed(), self.repr.loaded_set_limit);

            set.read().expect("should've ready to have a read")
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
