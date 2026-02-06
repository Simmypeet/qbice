//! Contains the TinyLFU cache implementation designed specifically for
//! write-behind caching layers.

use std::{collections::hash_map, hash::BuildHasher};

use fxhash::{FxBuildHasher, FxHashMap};

use crate::{
    sharded::Sharded,
    tiny_lfu::policy::{Policy, PolicyMessage},
};

mod lru;
mod policy;
mod single_flight;
mod sketch;

const MAINTENANCE_BATCH_SIZE: usize = 32;

/// A listener trait for cache entry lifecycle events.
pub trait LifecycleListener<K, V>: Default {
    /// Determines if the given entry is currently pinned.
    ///
    /// If "pinned", the entry will not be evicted from the cache.
    fn is_pinned(&self, key: &K, value: &mut V) -> bool;
}

/// The default lifecycle listener which does not pin any entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct DefaultLifecycleListener;

impl<K, V> LifecycleListener<K, V> for DefaultLifecycleListener {
    fn is_pinned(&self, _key: &K, _value: &mut V) -> bool { false }
}

/// A TinyLFU (Tiny Least Frequently Used) cache implementation with
/// support for pinned entries and negative caching.
///
/// This designed specifically for caching layers with write-behind semantics,
/// where entries may be temporarily pinned to avoid eviction during ongoing
/// write operations, and negative caching is used to remember absent entries
/// without storing actual values.
pub struct TinyLFU<K, V, L = DefaultLifecycleListener> {
    storage: Sharded<FxHashMap<K, V>>,

    message_queue: crossbeam::queue::SegQueue<PolicyMessage<K>>,
    policy: parking_lot::Mutex<Policy<K>>,

    lifecycle_listener: L,
    build_hasher: FxBuildHasher,
}

impl<K, V, L> std::fmt::Debug for TinyLFU<K, V, L> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TinyLFU").finish_non_exhaustive()
    }
}

impl<K, V, L: Default> TinyLFU<K, V, L> {
    /// Creates a new TinyLFU cache with the specified capacity and shard count.
    #[must_use]
    pub fn new(capacity: usize, shard_count: usize) -> Self {
        Self {
            storage: Sharded::new(shard_count, |_| FxHashMap::default()),
            message_queue: crossbeam::queue::SegQueue::new(),
            policy: parking_lot::Mutex::new(Policy::new(capacity)),
            lifecycle_listener: L::default(),
            build_hasher: FxBuildHasher::default(),
        }
    }
}

/// Represents an entry in the TinyLFU cache that is currently occupied.
pub struct OccupiedEntry<'a, 'x, K, V> {
    entry: hash_map::OccupiedEntry<'a, K, V>,

    message: &'x mut Option<PolicyMessage<K>>,
}

impl<K, V> std::fmt::Debug for OccupiedEntry<'_, '_, K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OccupiedEntry").finish_non_exhaustive()
    }
}

impl<K: Clone + std::hash::Hash + Eq, V> OccupiedEntry<'_, '_, K, V> {
    /// Returns a reference to the value in the entry.
    #[must_use]
    pub fn get(&self) -> &V { self.entry.get() }

    /// Returns a mutable reference to the value in the entry.
    #[must_use]
    pub fn get_mut(&mut self) -> &mut V { self.entry.get_mut() }

    /// Removes the entry from the cache and returns the value.
    #[must_use]
    pub fn remove(self) -> V {
        let key = self.entry.key().clone();
        let v = self.entry.remove();
        *self.message = Some(PolicyMessage::Removed(key));

        v
    }
}

/// Represents an entry in the TinyLFU cache that is currently vacant.
pub struct VacantEntry<'a, 'x, K, V> {
    entry: hash_map::VacantEntry<'a, K, V>,

    message: &'x mut Option<PolicyMessage<K>>,
}

impl<K, V> std::fmt::Debug for VacantEntry<'_, '_, K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VacantEntry").finish_non_exhaustive()
    }
}

impl<K: Clone + std::hash::Hash + Eq, V> VacantEntry<'_, '_, K, V> {
    /// Inserts a value into the entry.
    pub fn insert(self, value: V) {
        let key = self.entry.key().clone();

        self.entry.insert(value);

        *self.message = Some(PolicyMessage::Write(key));
    }
}

/// The API that provides access to the cache entries having similar interface
/// to the `HashMap`'s `Entry` API.
#[allow(missing_docs)]
pub enum Entry<'a, 'x, K, V> {
    Vacant(VacantEntry<'a, 'x, K, V>),
    Occupied(OccupiedEntry<'a, 'x, K, V>),
}

impl<K, V> std::fmt::Debug for Entry<'_, '_, K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Entry").finish_non_exhaustive()
    }
}

impl<K: std::hash::Hash + Eq + Clone, V, L: LifecycleListener<K, V>>
    TinyLFU<K, V, L>
{
    /// Retrieves a value from the cache by key.
    pub fn get(&self, key: &K) -> Option<V>
    where
        V: Clone,
    {
        self.get_map(key, std::clone::Clone::clone)
    }

    /// Retrieves a mapped value from the cache by key using the provided
    /// function.
    pub fn get_map<T>(&self, key: &K, f: impl Fn(&V) -> T) -> Option<T> {
        let hash = self.hash(key);
        let shard = self.storage.read_shard(self.storage.shard_index(hash));

        let entry = shard.get(key).map(f);

        // immediately drop the shard lock to avoid lock contention
        drop(shard);

        self.try_maintenance(Some(PolicyMessage::ReadHit(key.clone())));

        entry
    }

    /// Provides access to a cache entry by key.
    ///
    /// This acquires a write lock on the relevant shard of the cache.
    pub fn entry<T>(
        &self,
        key: K,
        f: impl FnOnce(Entry<'_, '_, K, V>) -> T,
    ) -> T {
        let hash = self.hash(&key);
        let mut shard =
            self.storage.write_shard(self.storage.shard_index(hash));

        let mut message = None;

        let t = match shard.entry(key) {
            hash_map::Entry::Vacant(entry) => {
                f(Entry::Vacant(VacantEntry { entry, message: &mut message }))
            }

            hash_map::Entry::Occupied(occupied_entry) => {
                f(Entry::Occupied(OccupiedEntry {
                    entry: occupied_entry,
                    message: &mut message,
                }))
            }
        };

        drop(shard);

        self.try_maintenance(message);

        t
    }

    fn try_maintenance(&self, policy_message: Option<PolicyMessage<K>>) {
        match (policy_message, self.policy.try_lock()) {
            // no message and can't lock, can't do anything
            (None, None) => {}

            (None, Some(mut lock)) => {
                // got the lock, process messages
                self.process_policy_message(&mut lock);
            }

            // have a message but can't lock, will enqueue
            (Some(value), None) => {
                // will enqueue later
                self.message_queue.push(value);
            }

            (Some(value), Some(mut lock)) => {
                // got the lock, process the message first
                self.process_message(value, &mut lock);

                // then process any queued messages
                self.process_policy_message(&mut lock);
            }
        }
    }

    fn process_policy_message(&self, lock: &mut Policy<K>) {
        let mut processed = 0;

        while processed <= MAINTENANCE_BATCH_SIZE {
            processed += 1;

            // Pop a message
            let Some(message) = self.message_queue.pop() else {
                break;
            };

            self.process_message(message, lock);
        }
    }

    fn process_message(&self, message: PolicyMessage<K>, lock: &mut Policy<K>) {
        match message {
            PolicyMessage::ReadHit(key) => {
                let hash = self.hash(&key);
                lock.on_read_hit(&key, hash, true);
            }

            PolicyMessage::Write(key) => {
                let hash = self.hash(&key);
                lock.on_write(
                    &key,
                    hash,
                    &self.build_hasher,
                    self.remove_closure(),
                );
            }

            PolicyMessage::Removed(key) => {
                lock.on_removed(&key);
            }
        }

        // before dropping the lock, we can try to overflow trimming
        lock.attempt_to_trim_overflowing_cache(self.remove_closure());
    }

    fn remove_closure(&self) -> impl Fn(&K) -> bool + '_ {
        // atomically determine if we can remove the key
        |evicted_key| {
            // IMPORTANT: must obtain exclusive access to
            // also avoid eviction race
            let mut shard = self
                .storage
                .write_shard(self.storage.shard_index(self.hash(evicted_key)));

            match shard.entry(evicted_key.clone()) {
                hash_map::Entry::Occupied(mut occupied_entry) => {
                    let can_remove = {
                        let value = occupied_entry.get_mut();

                        !self.lifecycle_listener.is_pinned(evicted_key, value)
                    };

                    // IMPORTANT: we drop the value outside
                    // of the lock to avoid potential large
                    // value drop times blocking other
                    // operations
                    let value = if can_remove {
                        Some(occupied_entry.remove_entry())
                    } else {
                        None
                    };

                    drop(shard);
                    drop(value);

                    true
                }

                hash_map::Entry::Vacant(_) => {
                    // already evicted
                    true
                }
            }
        }
    }

    fn hash<T: std::hash::Hash>(&self, t: &T) -> u64 {
        self.build_hasher.hash_one(t)
    }
}
