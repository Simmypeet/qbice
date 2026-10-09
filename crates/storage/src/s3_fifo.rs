//! Contains a cache that evicts with the S3-FIFO policy, with support for
//! entries that are pinned.
//!
//! # The policy
//!
//! The keys of the cache are kept in two first-in first-out queues, a small
//! one that every new key enters and a main one. Every entry counts how often
//! it has been read, up to three times.
//!
//! - A key that reaches the end of the small queue moves to the main queue if
//!   its entry has been read more than once, and is evicted otherwise. An entry
//!   that is hardly read after it was inserted leaves through the small queue
//!   without ever taking up room in the main one.
//! - A key that reaches the end of the main queue is evicted if its entry has
//!   not been read since the key was last looked at. Otherwise it goes around
//!   the queue again, with one read taken off its count.
//! - The keys that were evicted are remembered for a while, as hashes. A key
//!   that is inserted again while it is remembered was evicted too early, and
//!   enters the main queue directly.
//!
//! This is the policy of "FIFO queues are all you need for cache eviction"
//! (Yang et al., SOSP 2023).
//!
//! # What a read costs
//!
//! A read finds the entry in the hash table and adds one to its count. Nothing
//! is recorded for the policy and no lock of the policy is taken, so reads of
//! different entries never wait for one another. Finding an entry with
//! [`S3Fifo::entry`] counts as a read of it as well.
//!
//! The queues are only changed when entries are inserted. The keys of new
//! entries are buffered, and whoever fills the buffer brings the queues up to
//! date for everyone, unless somebody else is already doing that.
//!
//! One thread at a time brings the queues up to date, and it takes in no
//! more than the keys that were buffered when it started. If keys are
//! inserted faster than that, the buffer grows, and with it the number of
//! entries above the capacity. Once the buffer holds more than a quarter of
//! what the cache has room for, whoever inserts waits for its turn to bring
//! the queues up to date instead of leaving it to somebody else. That slows
//! the insertions down to the pace at which entries are evicted.
//!
//! Nothing happens while nothing is inserted: the last few keys stay
//! buffered, and an entry that is not pinned anymore stays cached.
//! [`S3Fifo::run_maintenance`] is for whoever knows that such a moment has
//! come.
//!
//! # Pinned entries
//!
//! An entry that its [`LifecycleListener`] reports as pinned is never
//! evicted. A key whose turn to be evicted comes while its entry is pinned
//! waits in a queue of its own instead. Once the entry is not pinned anymore
//! it is evicted, unless it was read while its key waited: then the key goes
//! to the main queue. Nobody reports that an entry is not pinned anymore: the
//! cache asks again whenever it brings the queues up to date.
//!
//! # Removed entries
//!
//! [`OccupiedEntry::remove`] takes the entry out of the hash table and leaves
//! its key in the queue it is in. The cache remembers that the key has lost
//! an entry. When such a key has its turn, it is dropped and forgotten, and
//! until then it does not count towards the capacity.
//!
//! If an entry is inserted for the key again in the meantime, the key is in
//! the queues twice. The turn of the key of the removed entry must not be
//! taken for the turn of the new entry. The cache cannot tell the two keys
//! apart, and does not have to: it drops the one that has its turn first, and
//! the other one stands for the entry from then on.
//!
//! # Using the cache from inside the cache
//!
//! The closure that is given to [`S3Fifo::entry`] runs while the entry is
//! locked, and [`LifecycleListener::is_pinned`] runs while the entry and the
//! queues are locked. Neither may call into the cache that called it: such a
//! call can wait forever for a lock that its own caller holds.

use std::{
    collections::{HashMap, VecDeque},
    hash::{BuildHasher, Hash},
    sync::atomic::{AtomicU8, AtomicU64, AtomicUsize, Ordering},
};

use crossbeam::queue::SegQueue;
use crossbeam_utils::CachePadded;
use fxhash::FxBuildHasher;

/// A listener trait for cache entry lifecycle events.
pub trait LifecycleListener<K, V> {
    /// Determines if the given entry is currently pinned.
    ///
    /// If "pinned", the entry will not be evicted from the cache.
    ///
    /// This is called while the entry and the queues of the cache are
    /// locked, so it must not call into the cache.
    fn is_pinned(&self, key: &K, value: &V) -> bool;
}

/// The default lifecycle listener which does not pin any entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct DefaultLifecycleListener;

impl<K, V> LifecycleListener<K, V> for DefaultLifecycleListener {
    fn is_pinned(&self, _key: &K, _value: &V) -> bool { false }
}

/// The highest number of reads that an entry counts.
const MAX_FREQUENCY: u8 = 3;

/// The number of reads above which a key moves from the small queue to the
/// main queue instead of being evicted.
const PROMOTION_THRESHOLD: u8 = 1;

/// The part of the capacity that the small queue takes up, as a divisor.
const SMALL_QUEUE_RATIO: usize = 10;

/// The number of buffered insertions at which the queues are brought up to
/// date.
const MAINTENANCE_BATCH_SIZE: usize = 32;

/// The part of the capacity that the buffer of insertions may reach before
/// whoever inserts waits for its turn to bring the queues up to date, as a
/// divisor.
const BUFFER_RATIO: usize = 4;

/// The fewest buffered insertions that make whoever inserts wait.
const MIN_BUFFER_LIMIT: usize = 256;

/// The number of hashes in one bucket of the [`Ghost`].
const GHOST_BUCKET_SIZE: usize = 8;

/// An entry of the cache.
struct Slot<V> {
    value: V,

    /// How often the entry has been read since the policy last looked at it,
    /// up to [`MAX_FREQUENCY`].
    frequency: AtomicU8,
}

impl<V> Slot<V> {
    const fn new(value: V) -> Self {
        Self { value, frequency: AtomicU8::new(0) }
    }

    /// Counts a read of the entry.
    ///
    /// Two reads at the same time may count as one. The count only says
    /// whether the entry is in use, so nothing depends on it being exact.
    fn touch(&self) {
        let frequency = self.frequency.load(Ordering::Relaxed);

        // an entry that is read all the time is not written to all the time
        if frequency < MAX_FREQUENCY {
            self.frequency.store(frequency + 1, Ordering::Relaxed);
        }
    }

    fn frequency(&self) -> u8 { self.frequency.load(Ordering::Relaxed) }

    fn set_frequency(&self, frequency: u8) {
        self.frequency.store(frequency, Ordering::Relaxed);
    }
}

/// One remembered key of the [`Ghost`].
#[derive(Clone, Copy, Default)]
struct GhostSlot {
    /// A part of the hash of the key, or zero if the slot is empty.
    fingerprint: u32,

    /// The [`Ghost::tick`] at which the key was remembered.
    tick: u32,
}

/// Remembers the keys that were evicted most recently.
///
/// Only a part of the hash of a key is kept, in a table that the rest of the
/// hash indexes, so that looking a key up or remembering one touches a single
/// cache line. Two keys can be mistaken for one another, which makes a key
/// enter the main queue that should have entered the small one, and a key can
/// be forgotten early when its bucket is full. Neither is more than a wrong
/// guess about which entries are worth keeping.
struct Ghost {
    slots: Box<[GhostSlot]>,
    bucket_mask: usize,

    /// The number of keys that have been remembered so far.
    tick: u32,

    /// The number of most recently remembered keys that still count.
    capacity: u32,
}

impl Ghost {
    #[allow(clippy::cast_possible_truncation)]
    fn new(capacity: usize) -> Self {
        let buckets =
            capacity.div_ceil(GHOST_BUCKET_SIZE).max(1).next_power_of_two();

        Self {
            slots: vec![GhostSlot::default(); buckets * GHOST_BUCKET_SIZE]
                .into_boxed_slice(),
            bucket_mask: buckets - 1,
            tick: 0,
            capacity: capacity.min(u32::MAX as usize) as u32,
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn bucket(&mut self, hash: u64) -> (&mut [GhostSlot], u32) {
        // The high bits of the hash pick the bucket and the low bits tell
        // the keys of a bucket apart.
        let bucket = (hash >> 32) as usize & self.bucket_mask;
        let fingerprint = hash as u32 | 1;

        let start = bucket * GHOST_BUCKET_SIZE;

        (&mut self.slots[start..start + GHOST_BUCKET_SIZE], fingerprint)
    }

    /// Remembers the key with the given hash.
    fn insert(&mut self, hash: u64) {
        self.tick = self.tick.wrapping_add(1);
        let tick = self.tick;

        let (bucket, fingerprint) = self.bucket(hash);

        // the slot of the key itself, or else the slot that was written the
        // longest ago
        let mut replaced = 0;
        let mut replaced_age = 0;

        for (index, slot) in bucket.iter().enumerate() {
            if slot.fingerprint == fingerprint {
                replaced = index;
                break;
            }

            let age = if slot.fingerprint == 0 {
                u32::MAX
            } else {
                tick.wrapping_sub(slot.tick)
            };

            if age > replaced_age {
                replaced = index;
                replaced_age = age;
            }
        }

        bucket[replaced] = GhostSlot { fingerprint, tick };
    }

    /// Forgets the key with the given hash and returns whether it was
    /// remembered.
    fn take(&mut self, hash: u64) -> bool {
        let tick = self.tick;
        let capacity = self.capacity;

        let (bucket, fingerprint) = self.bucket(hash);

        for slot in bucket {
            if slot.fingerprint == fingerprint {
                let remembered = tick.wrapping_sub(slot.tick) < capacity;
                slot.fingerprint = 0;

                return remembered;
            }
        }

        false
    }
}

/// The queues that decide which entry is evicted next.
struct Policy<K> {
    /// The keys that were inserted most recently, oldest first.
    small: VecDeque<K>,

    /// The keys whose entries have proven to be read more than once, in the
    /// order in which they are looked at next.
    main: VecDeque<K>,

    /// The keys whose turn to be evicted came while their entries were
    /// pinned, in the order in which their turn came.
    pinned: VecDeque<K>,

    ghost: Ghost,

    /// The number of keys that the small and the main queue hold together
    /// before a key is evicted.
    capacity: usize,

    /// The number of keys in the small queue above which a key is evicted
    /// from the small queue instead of the main one.
    small_capacity: usize,
}

impl<K> Policy<K> {
    fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        let small_capacity = (capacity / SMALL_QUEUE_RATIO).max(1);

        Self {
            small: VecDeque::new(),
            main: VecDeque::new(),
            pinned: VecDeque::new(),
            ghost: Ghost::new(capacity - small_capacity),
            capacity,
            small_capacity,
        }
    }
}

/// The keys in the buffer and in the queues that have lost an entry to
/// [`OccupiedEntry::remove`].
struct Removed<K> {
    /// How many of them there are. Nothing but this is looked at as long as
    /// no entry has been removed.
    count: AtomicUsize,

    /// How many entries each key has lost.
    keys: parking_lot::Mutex<HashMap<K, usize, FxBuildHasher>>,
}

impl<K: Eq + Hash> Removed<K> {
    fn new() -> Self {
        Self {
            count: AtomicUsize::new(0),
            keys: parking_lot::Mutex::new(HashMap::default()),
        }
    }

    /// Remembers that the key has lost an entry.
    fn remember(&self, key: K) {
        *self.keys.lock().entry(key).or_default() += 1;

        self.count.fetch_add(1, Ordering::Relaxed);
    }

    /// Forgets one of the entries that the key has lost and returns whether
    /// it had lost any.
    fn forget(&self, key: &K) -> bool {
        if self.count.load(Ordering::Relaxed) == 0 {
            return false;
        }

        let mut keys = self.keys.lock();

        let Some(lost) = keys.get_mut(key) else {
            return false;
        };

        *lost -= 1;

        if *lost == 0 {
            keys.remove(key);
        }

        self.count.fetch_sub(1, Ordering::Relaxed);

        true
    }
}

/// A cache that evicts with the S3-FIFO policy, with support for pinned
/// entries.
///
/// See the [module documentation](self) for how it decides what to evict.
pub struct S3Fifo<
    K: Clone + Eq + Hash + Send + Sync + 'static,
    V: Send + Sync + 'static,
    L: LifecycleListener<K, V> + Send + Sync + 'static = DefaultLifecycleListener,
> {
    storage: CachePadded<scc::HashMap<K, Slot<V>, FxBuildHasher>>,

    /// The keys that have been inserted since the queues were last brought
    /// up to date.
    inserted: CachePadded<SegQueue<K>>,

    /// The number of entries that have been inserted so far.
    inserted_count: CachePadded<AtomicU64>,

    /// The number of inserted keys that the queues have taken in so far.
    /// Whoever holds the lock of the policy is the only one to change it.
    drained_count: CachePadded<AtomicU64>,

    policy: CachePadded<parking_lot::Mutex<Policy<K>>>,

    removed: Removed<K>,

    /// The number of buffered insertions above which whoever inserts waits
    /// for its turn to bring the queues up to date.
    buffer_limit: usize,

    lifecycle_listener: L,
    build_hasher: FxBuildHasher,

    #[cfg(feature = "tracing_resource")]
    resource_span: tracing::Span,
}

/// Reports to the span of a cache that its number of entries has changed by
/// one.
#[cfg(feature = "tracing_resource")]
fn trace_len_change(resource_span: &tracing::Span, op: &'static str) {
    resource_span.in_scope(|| {
        tracing::trace!(
            target: "runtime::resource::state_update",
            len = 1,
            len.op = op,
        );
    });
}

impl<
    K: Clone + Eq + Hash + Send + Sync + 'static,
    V: Send + Sync + 'static,
    L: LifecycleListener<K, V> + Send + Sync + 'static,
> std::fmt::Debug for S3Fifo<K, V, L>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("S3Fifo").finish_non_exhaustive()
    }
}

/// Represents an entry in the cache that is currently occupied.
pub struct OccupiedEntry<'a, K, V> {
    entry: scc::hash_map::OccupiedEntry<'a, K, Slot<V>, FxBuildHasher>,

    removed: &'a Removed<K>,

    #[cfg(feature = "tracing_resource")]
    resource_span: &'a tracing::Span,
}

impl<K, V> std::fmt::Debug for OccupiedEntry<'_, K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OccupiedEntry").finish_non_exhaustive()
    }
}

impl<K: Eq + Hash, V> OccupiedEntry<'_, K, V> {
    /// Returns a reference to the value in the entry.
    #[must_use]
    pub fn get(&self) -> &V { &self.entry.get().value }

    /// Returns a mutable reference to the value in the entry.
    #[must_use]
    pub fn get_mut(&mut self) -> &mut V { &mut self.entry.get_mut().value }

    /// Removes the entry from the cache and returns the value.
    ///
    /// The key stays in the queue it is in until its turn comes, where it is
    /// found to have lost its entry and dropped.
    #[must_use]
    pub fn remove(self) -> V
    where
        K: Clone,
    {
        // Remembered while the entry is still there: the key is not found
        // to have no entry before it is known to have lost one.
        self.removed.remember(self.entry.key().clone());

        #[cfg(feature = "tracing_resource")]
        trace_len_change(self.resource_span, "sub");

        self.entry.remove().value
    }
}

/// Represents an entry in the cache that is currently vacant.
pub struct VacantEntry<'a, K, V> {
    entry: scc::hash_map::VacantEntry<'a, K, Slot<V>, FxBuildHasher>,

    inserted: &'a SegQueue<K>,
    inserted_count: &'a AtomicU64,

    #[cfg(feature = "tracing_resource")]
    resource_span: &'a tracing::Span,
}

impl<K, V> std::fmt::Debug for VacantEntry<'_, K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VacantEntry").finish_non_exhaustive()
    }
}

impl<K: Clone + Eq + Hash, V> VacantEntry<'_, K, V> {
    /// Inserts a value into the entry.
    pub fn insert(self, value: V) {
        // Counted before the key is announced, so that no more keys are
        // ever taken in than have been counted.
        self.inserted_count.fetch_add(1, Ordering::Relaxed);

        // The key is announced while the entry is still locked. Whoever
        // picks the key up has to wait for the lock before it can look at
        // the entry, and then finds it.
        self.inserted.push(self.entry.key().clone());

        self.entry.insert_entry(Slot::new(value));

        #[cfg(feature = "tracing_resource")]
        trace_len_change(self.resource_span, "add");
    }
}

/// The API that provides access to the cache entries having similar interface
/// to the `HashMap`'s `Entry` API.
#[allow(missing_docs)]
pub enum Entry<'a, K, V> {
    Vacant(VacantEntry<'a, K, V>),
    Occupied(OccupiedEntry<'a, K, V>),
}

impl<K, V> std::fmt::Debug for Entry<'_, K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Entry").finish_non_exhaustive()
    }
}

impl<
    K: Clone + Eq + Hash + Send + Sync + 'static,
    V: Send + Sync + 'static,
    L: LifecycleListener<K, V> + Send + Sync + 'static,
> S3Fifo<K, V, L>
{
    /// Creates a new cache that holds about `capacity` entries, not counting
    /// the entries that it has to keep because they are pinned.
    #[must_use]
    pub fn new(capacity: usize) -> Self
    where
        L: Default,
    {
        Self::with_lifecycle_listener(capacity, L::default())
    }

    /// Creates a new cache that asks `lifecycle_listener` whether an entry is
    /// pinned.
    #[must_use]
    pub fn with_lifecycle_listener(
        capacity: usize,
        lifecycle_listener: L,
    ) -> Self {
        #[cfg(feature = "tracing_resource")]
        let resource_span = {
            let location = std::panic::Location::caller();

            let resource_span = tracing::trace_span!(
                parent: None,
                "runtime.resource",
                concrete_type = "S3Fifo",
                kind = "Cache",
                loc.file = location.file(),
                loc.line = location.line(),
                loc.col = location.column(),
            );

            resource_span.in_scope(|| {
                tracing::trace!(
                    target: "runtime::resource::state_update",
                    len = 0
                );
            });

            resource_span
        };

        Self {
            storage: CachePadded::new(scc::HashMap::with_hasher(
                FxBuildHasher::default(),
            )),
            inserted: CachePadded::new(SegQueue::new()),
            inserted_count: CachePadded::new(AtomicU64::new(0)),
            drained_count: CachePadded::new(AtomicU64::new(0)),
            policy: CachePadded::new(parking_lot::Mutex::new(Policy::new(
                capacity,
            ))),
            removed: Removed::new(),
            buffer_limit: (capacity / BUFFER_RATIO).max(MIN_BUFFER_LIMIT),
            lifecycle_listener,
            build_hasher: FxBuildHasher::default(),

            #[cfg(feature = "tracing_resource")]
            resource_span,
        }
    }

    /// Retrieves a value from the cache by key.
    #[inline]
    pub fn get(&self, key: &K) -> Option<V>
    where
        V: Clone,
    {
        self.get_map(key, std::clone::Clone::clone)
    }

    /// Retrieves a mapped value from the cache by key using the provided
    /// function.
    #[inline]
    pub fn get_map<T>(&self, key: &K, f: impl FnOnce(&V) -> T) -> Option<T> {
        self.storage.read_sync(key, |_, slot| {
            slot.touch();

            f(&slot.value)
        })
    }

    /// Provides access to a cache entry by key.
    ///
    /// The entry is locked for as long as `f` runs, so `f` must not call
    /// into this cache. Finding the entry occupied counts as a read of it.
    pub fn entry<T>(&self, key: K, f: impl FnOnce(Entry<'_, K, V>) -> T) -> T {
        let result = match self.storage.entry_sync(key) {
            scc::hash_map::Entry::Vacant(entry) => {
                f(Entry::Vacant(VacantEntry {
                    entry,
                    inserted: &self.inserted,
                    inserted_count: &self.inserted_count,

                    #[cfg(feature = "tracing_resource")]
                    resource_span: &self.resource_span,
                }))
            }

            scc::hash_map::Entry::Occupied(entry) => {
                entry.get().touch();

                f(Entry::Occupied(OccupiedEntry {
                    entry,
                    removed: &self.removed,

                    #[cfg(feature = "tracing_resource")]
                    resource_span: &self.resource_span,
                }))
            }
        };

        // not before the entry has been unlocked: the queues are brought up
        // to date by locking entries
        self.try_maintenance();

        result
    }

    /// Returns the lifecycle listener of the cache.
    #[must_use]
    pub const fn lifecycle_listener(&self) -> &L { &self.lifecycle_listener }

    /// Brings the queues up to date now, however little has been inserted
    /// since they last were.
    ///
    /// The cache does this by itself while entries are inserted, and only
    /// then. This is for whoever knows that the cache has something to let
    /// go of while nothing is inserted, for example because the entries
    /// that were pinned are not pinned anymore.
    pub fn run_maintenance(&self) {
        let mut policy = self.policy.lock();

        self.maintain(&mut policy);
    }

    /// The number of keys that have been inserted and that the queues have
    /// not taken in yet.
    fn buffered(&self) -> usize {
        let inserted = self.inserted_count.load(Ordering::Relaxed);
        let drained = self.drained_count.load(Ordering::Relaxed);

        // the two counts are not read at the same instant
        usize::try_from(inserted.saturating_sub(drained)).unwrap_or(usize::MAX)
    }

    /// The number of entries that the small and the main queue stand for.
    ///
    /// A removed key that is still buffered, or that waits for an entry
    /// that was pinned, is taken off here as well. The cache holds that
    /// many entries more until the key is dropped.
    fn queued(&self, policy: &Policy<K>) -> usize {
        (policy.small.len() + policy.main.len())
            .saturating_sub(self.removed.count.load(Ordering::Relaxed))
    }

    /// Brings the queues up to date if enough has been inserted and nobody
    /// else is doing it.
    fn try_maintenance(&self) {
        let buffered = self.buffered();

        if buffered <= MAINTENANCE_BATCH_SIZE {
            return;
        }

        let mut policy = if buffered > self.buffer_limit {
            // More is inserted than the one thread that brings the queues
            // up to date gets to evict. Waiting for a turn is what keeps
            // the cache from growing for as long as that lasts.
            self.policy.lock()
        } else {
            let Some(policy) = self.policy.try_lock() else {
                return;
            };

            policy
        };

        self.maintain(&mut policy);
    }

    fn maintain(&self, policy: &mut Policy<K>) {
        // Only the keys that are buffered by now are taken in. The ones
        // that are inserted in the meantime are left to whoever comes next,
        // so that this ends however fast keys are inserted.
        let mut taken_in = 0;

        for _ in 0..self.buffered() {
            let Some(key) = self.inserted.pop() else {
                break;
            };

            taken_in += 1;

            if policy.ghost.take(self.build_hasher.hash_one(&key)) {
                policy.main.push_back(key);
            } else {
                policy.small.push_back(key);
            }
        }

        self.drained_count.fetch_add(taken_in, Ordering::Relaxed);

        self.evict_unpinned(policy);

        // the number of keys that went around the main queue again
        let mut second_chances = 0;

        while self.queued(policy) > policy.capacity {
            if policy.small.len() > policy.small_capacity {
                self.evict_from_small(policy);
            } else {
                self.evict_from_main(policy, &mut second_chances);
            }
        }
    }

    /// Returns the entry that a key from a queue stands for, or `None` if
    /// the key is one that has lost its entry.
    fn entry_of(
        &self,
        key: &K,
    ) -> Option<scc::hash_map::OccupiedEntry<'_, K, Slot<V>, FxBuildHasher>>
    {
        // If the key has lost an entry and has an entry again, this key and
        // another one in the queues are the two of them. Whichever has its
        // turn first is taken for the one that lost its entry.
        if self.removed.forget(key) {
            return None;
        }

        let entry = self.storage.get_sync(key);

        if entry.is_none() {
            // The entry has been removed since the question above. Only
            // `OccupiedEntry::remove` takes an entry away from a key that
            // is in a queue, and it remembers the key before it does.
            let forgotten = self.removed.forget(key);

            debug_assert!(forgotten, "a key lost its entry unnoticed");
        }

        entry
    }

    /// Looks at the key at the end of the small queue.
    fn evict_from_small(&self, policy: &mut Policy<K>) {
        let Some(key) = policy.small.pop_front() else {
            return;
        };

        let Some(entry) = self.entry_of(&key) else {
            return;
        };

        if entry.get().frequency() > PROMOTION_THRESHOLD {
            // what the entry is worth in the main queue is decided by the
            // reads it gets there
            entry.get().set_frequency(0);
            drop(entry);

            policy.main.push_back(key);

            return;
        }

        self.evict(key, entry, policy);
    }

    /// Looks at the key at the end of the main queue.
    ///
    /// `second_chances` counts the keys that went around the queue again
    /// since the queues were locked.
    fn evict_from_main(
        &self,
        policy: &mut Policy<K>,
        second_chances: &mut usize,
    ) {
        let Some(key) = policy.main.pop_front() else {
            return;
        };

        let Some(entry) = self.entry_of(&key) else {
            return;
        };

        let frequency = entry.get().frequency();

        // If nothing is read in the meantime, every entry of the queue is
        // found unread after it has gone around as often as an entry counts
        // reads. More second chances than that are only asked for by
        // entries that are read as fast as their counts are taken down,
        // which can go on for as long as they are read, with the queues
        // locked. Then the oldest of them goes.
        let second_chances_left = *second_chances
            < usize::from(MAX_FREQUENCY) * (policy.main.len() + 1);

        if frequency > 0 && second_chances_left {
            *second_chances += 1;

            entry.get().set_frequency(frequency - 1);
            drop(entry);

            policy.main.push_back(key);

            return;
        }

        self.evict(key, entry, policy);
    }

    /// Evicts the entry of a key whose turn has come, or makes the key wait
    /// if the entry is pinned.
    fn evict(
        &self,
        key: K,
        entry: scc::hash_map::OccupiedEntry<'_, K, Slot<V>, FxBuildHasher>,
        policy: &mut Policy<K>,
    ) {
        // IMPORTANT: this is asked while the entry is locked, so that
        // nothing can pin the entry between the answer and the removal
        if self.lifecycle_listener.is_pinned(&key, &entry.get().value) {
            // from here on the count tells whether the entry is read while
            // its key waits
            entry.get().set_frequency(0);
            drop(entry);

            policy.pinned.push_back(key);

            return;
        }

        self.discard(&key, entry, policy);
    }

    /// Removes an entry that is not pinned from the cache.
    fn discard(
        &self,
        key: &K,
        entry: scc::hash_map::OccupiedEntry<'_, K, Slot<V>, FxBuildHasher>,
        policy: &mut Policy<K>,
    ) {
        // IMPORTANT: the value is dropped after the entry has been unlocked,
        // because dropping a large value takes time
        let removed = entry.remove_entry();

        #[cfg(feature = "tracing_resource")]
        trace_len_change(&self.resource_span, "sub");

        // If the key comes back soon, evicting it was a mistake. That holds
        // whichever queue the key was in.
        policy.ghost.insert(self.build_hasher.hash_one(key));

        drop(removed);
    }

    /// Looks at the keys that wait for their entries to be unpinned.
    ///
    /// An entry that is not pinned anymore is evicted, unless it was read
    /// while its key waited. Such an entry is in use, and its key goes to
    /// the main queue, which decides from then on how long it stays.
    ///
    /// Entries are mostly unpinned in the order in which they were pinned,
    /// so the search stops at the first entry that is still pinned. That
    /// entry goes to the back of the queue, so that an entry that stays
    /// pinned for long does not keep the ones behind it from being looked
    /// at.
    fn evict_unpinned(&self, policy: &mut Policy<K>) {
        while let Some(key) = policy.pinned.pop_front() {
            let Some(entry) = self.entry_of(&key) else {
                continue;
            };

            // IMPORTANT: this is asked while the entry is locked, as in
            // `evict`
            if self.lifecycle_listener.is_pinned(&key, &entry.get().value) {
                drop(entry);

                policy.pinned.push_back(key);

                return;
            }

            if entry.get().frequency() > 0 {
                entry.get().set_frequency(0);
                drop(entry);

                policy.main.push_back(key);

                continue;
            }

            self.discard(&key, entry, policy);
        }
    }
}

#[cfg(test)]
mod test;
