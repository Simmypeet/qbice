//! Contains the TinyLFU cache implementation designed specifically for
//! write-behind caching layers.

use std::{
    collections::hash_map, hash::BuildHasher, sync::atomic::AtomicUsize,
};

use fxhash::{FxBuildHasher, FxHashMap};

use crate::{
    sharded::Sharded,
    tiny_lfu::policy::{Policy, PolicyMessage},
};

mod lru;
mod policy;
mod single_flight;
mod sketch;

const MAINTENANCE_THRESHOLD: usize = 64;
const MAINTENANCE_BATCH_SIZE: usize = 32;

struct Entry<V> {
    value: Option<V>,
    pin_count: AtomicUsize,
}

/// A TinyLFU (Tiny Least Frequently Used) cache implementation with
/// support for pinned entries and negative caching.
///
/// This designed specifically for caching layers with write-behind semantics,
/// where entries may be temporarily pinned to avoid eviction during ongoing
/// write operations, and negative caching is used to remember absent entries
/// without storing actual values.
pub struct TinyLFU<K, V> {
    storage: Sharded<FxHashMap<K, Entry<V>>>,

    message_queue: crossbeam::queue::ArrayQueue<PolicyMessage<K>>,
    policy: parking_lot::Mutex<Policy<K>>,

    single_flight: single_flight::SingleFlight<K>,

    build_hasher: FxBuildHasher,
}

impl<K, V> std::fmt::Debug for TinyLFU<K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TinyLFU").finish_non_exhaustive()
    }
}

impl<K, V> TinyLFU<K, V> {
    /// Creates a new TinyLFU cache with the specified capacity and shard count.
    #[must_use]
    pub fn new(capacity: usize, shard_count: usize) -> Self {
        Self {
            storage: Sharded::new(shard_count, |_| FxHashMap::default()),
            message_queue: crossbeam::queue::ArrayQueue::new(2048),
            policy: parking_lot::Mutex::new(Policy::new(capacity)),
            single_flight: single_flight::SingleFlight::new(shard_count),
            build_hasher: FxBuildHasher::default(),
        }
    }
}

impl<K: std::hash::Hash + Eq + Clone, V> TinyLFU<K, V> {
    /// Retrieves a value from the cache by key.
    pub fn get(&self, key: &K) -> Option<V>
    where
        V: Clone,
    {
        let hash = self.hash(key);
        let shard = self.storage.read_shard(self.storage.shard_index(hash));

        let entry = shard.get(key).map(|x| x.value.clone())?;

        // immediately drop the shard lock to avoid lock contention
        drop(shard);

        // if we found the negative cache entry, don't send a read hit
        // we assume negative cache entries as "easy" to fetch again
        let message = if entry.is_some() {
            Some(PolicyMessage::ReadHit(key.clone()))
        } else {
            None
        };

        self.try_maintenance(message.as_ref());

        entry
    }

    /// Retrieves a value from the cache by key, initializing it if absent.
    ///
    /// The `init` is called to produce the value if it is not present in the
    /// cache. The `init` is guaranteed to be called at most once per key at
    /// any given time, even under concurrent access.
    ///
    /// When the value is initialized using `init`, if it is `None`, a negative
    /// cache entry is created to remember the absence of the value. However,
    /// negative cache entries do not trigger "read hit" notifications to
    /// the eviction policy, making it more likely for them to be evicted
    /// sooner as we assume "not found" entries are easier to fetch again.
    ///
    /// When a "negative cache" entry is found (can be created via `remove` with
    /// pinning, or via `get_or_init` returning None), the `init` function is
    /// not called again until the entry is evicted.
    pub async fn get_or_init(
        &self,
        key: &K,
        init: impl Fn() -> Option<V>,
    ) -> Option<V>
    where
        V: Clone + Send + Sync + 'static,
        K: Send + Sync + 'static,
    {
        let hash = self.hash(key);
        let shard_index = self.storage.shard_index(hash);

        loop {
            // Try to get the value first
            {
                let shard = self.storage.read_shard(shard_index);
                let entry = shard.get(key).map(|x| x.value.clone());

                // immediately drop the shard lock to avoid lock contention
                drop(shard);

                if let Some(value) = entry {
                    // if we found the negative cache entry, don't send a read
                    // hit
                    let message = if value.is_some() {
                        Some(PolicyMessage::ReadHit(key.clone()))
                    } else {
                        None
                    };

                    self.try_maintenance(message.as_ref());

                    return value;
                }
            }

            // Otherwise, use single flight to initialize
            self.single_flight
                .wait_or_work(key, shard_index, || {
                    let value = init();

                    // CRITICAL: while we fetching the value, another thread
                    // might have inserted it, if the entry isn't present, we
                    // insert it ourselves

                    let mut shard = self.storage.write_shard(shard_index);

                    match shard.entry(key.clone()) {
                        hash_map::Entry::Occupied(_) => {
                            // already inserted by another thread
                            // go back to the top of the loop and read again
                        }

                        hash_map::Entry::Vacant(vacant_entry) => {
                            let entry =
                                Entry { value, pin_count: AtomicUsize::new(0) };
                            vacant_entry.insert(entry);

                            // drop the shard lock to avoid contention
                            drop(shard);

                            self.try_maintenance(Some(&PolicyMessage::Write(
                                key.clone(),
                            )));
                        }
                    }
                })
                .await;
        }
    }

    /// Inserts a value into the cache with the specified key.
    ///
    /// If `increment_pin` is true, the entry's pin count is incremented,
    /// preventing it from being evicted until `decrement_pin` is called.
    pub fn insert(&self, key: K, value: V, increment_pin: bool) {
        let hash = self.hash(&key);
        let mut shard =
            self.storage.write_shard(self.storage.shard_index(hash));

        match shard.entry(key.clone()) {
            hash_map::Entry::Occupied(mut occupied_entry) => {
                occupied_entry.get_mut().value = Some(value);

                if increment_pin {
                    *occupied_entry.get_mut().pin_count.get_mut() += 1;
                }
            }

            hash_map::Entry::Vacant(vacant_entry) => {
                let entry = Entry {
                    value: Some(value),
                    pin_count: AtomicUsize::new(usize::from(increment_pin)),
                };
                vacant_entry.insert(entry);
            }
        }

        // immediately drop the shard lock to avoid lock contention
        drop(shard);

        self.try_maintenance(Some(&PolicyMessage::Write(key)));
    }

    /// Removes a value from the cache by key.
    ///
    /// If `increment_pin` is true, the entry's pin count is incremented,
    /// preventing it from being evicted until `decrement_pin` is called.
    ///
    /// If the entry is pinned (pin count > 0 or `increment_pin` is true), the
    /// cache entry is converted to a negative cache entry (value set to None)
    /// instead of being removed.
    pub fn remove(&self, key: &K, increment_pin: bool) {
        let hash = self.hash(key);
        let mut shard =
            self.storage.write_shard(self.storage.shard_index(hash));

        let message = match shard.entry(key.clone()) {
            hash_map::Entry::Occupied(mut occupied_entry) => {
                let entry = occupied_entry.get_mut();

                // required to keep track of pin count even for negative
                // cache entries
                if increment_pin {
                    *entry.pin_count.get_mut() += 1;
                    entry.value = None;

                    // no message to send yet
                    None
                } else {
                    // if the entry is pinned, we convert it to a negative
                    // cache entry instead of removing it
                    if *entry.pin_count.get_mut() > 0 {
                        entry.value = None;

                        // no message to send yet
                        None
                    }
                    // otherwise, we can remove it
                    else {
                        occupied_entry.remove_entry();

                        // the entry has been removed, notify the policy
                        Some(PolicyMessage::Removed(key.clone()))
                    }
                }
            }

            hash_map::Entry::Vacant(_) => {
                // if not present and requested with a pin, insert a negative
                // cache entry
                if increment_pin {
                    let entry =
                        Entry { value: None, pin_count: AtomicUsize::new(1) };

                    shard.insert(key.clone(), entry);
                }

                // no message to send
                None
            }
        };

        drop(shard);

        // do notify the policy of removal
        self.try_maintenance(message.as_ref());
    }

    /// Decrements the pin count of the cache entry by key.
    pub fn decrement_pin(&self, key: &K) {
        let hash = self.hash(key);
        let shard = self.storage.read_shard(self.storage.shard_index(hash));

        let should_drop = shard.get(key).is_some_and(|entry| {
            let count = entry
                .pin_count
                .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);

            // if count has reached 0 and it's a negative cache entry, we should
            // drop it
            count == 1 && entry.value.is_none()
        });

        // immediately drop the shard lock to avoid lock contention
        drop(shard);

        // the count has reached 1 before decrement, meaning it is now 0. take
        // this opportunity to remove negative cache entries
        let message = if should_drop {
            match self
                .storage
                .write_shard(self.storage.shard_index(hash))
                .entry(key.clone())
            {
                hash_map::Entry::Occupied(mut occupied_entry) => {
                    // double check: another thread might have incremented
                    if *occupied_entry.get_mut().pin_count.get_mut() == 0
                        && occupied_entry.get_mut().value.is_none()
                    {
                        occupied_entry.remove_entry();
                    }
                }

                hash_map::Entry::Vacant(_) => {
                    // already removed
                }
            }

            Some(PolicyMessage::Removed(key.clone()))
        } else {
            None
        };

        // try to perform maintenance
        self.try_maintenance(message.as_ref());
    }

    fn try_maintenance(&self, policy_message: Option<&PolicyMessage<K>>) {
        let Some(policy_message) = policy_message else {
            // opportunisitically try to lock and process messages
            let Some(mut try_lock) = self.policy.try_lock() else {
                return;
            };

            self.process_policy_message(&mut try_lock);

            return;
        };

        loop {
            // add to the message queue
            //
            // There are two cases:
            //
            // 1. Queue has space: we push the message directly and optionally
            //    process it if have no other threads doing so.
            // 2. Queue is full: force process one message to make space, then
            //    push our message.
            if self.message_queue.push(policy_message.clone()).is_ok() {
                // Try to process some messages if the queue is above the
                // maintenance threshold.
                if self.message_queue.len() < MAINTENANCE_THRESHOLD {
                    break;
                }

                // opportunistically try to lock and process messages
                let Some(mut try_lock) = self.policy.try_lock() else {
                    break;
                };

                self.process_policy_message(&mut try_lock);
            }

            // Queue was full, so we force process one message to make space.
            let mut lock = self.policy.lock();
            self.process_policy_message(&mut lock);
        }
    }

    fn process_policy_message(&self, policy: &mut Policy<K>) {
        let mut processed = 0;

        while processed <= MAINTENANCE_BATCH_SIZE {
            processed += 1;

            // Pop a message
            let Some(message) = self.message_queue.pop() else {
                break;
            };

            match message {
                PolicyMessage::ReadHit(key) => {
                    let hash = self.hash(&key);
                    policy.on_read_hit(&key, hash, true);
                }

                PolicyMessage::Write(key) => {
                    let hash = self.hash(&key);
                    policy.on_write(
                        &key,
                        hash,
                        &self.build_hasher,
                        // atomically determine if we can remove the key
                        |evicted_key| {
                            // IMPORTANT: must obtain exclusive access to
                            // also avoid eviction race
                            let mut shard = self.storage.write_shard(
                                self.storage
                                    .shard_index(self.hash(evicted_key)),
                            );

                            match shard.entry(evicted_key.clone()) {
                                hash_map::Entry::Occupied(
                                    mut occupied_entry,
                                ) => {
                                    if *occupied_entry
                                        .get_mut()
                                        .pin_count
                                        .get_mut()
                                        > 0
                                    {
                                        return false;
                                    }

                                    occupied_entry.remove_entry();
                                    true
                                }

                                hash_map::Entry::Vacant(_) => {
                                    // already evicted
                                    true
                                }
                            }
                        },
                    );
                }

                PolicyMessage::Removed(key) => {
                    policy.on_removed(&key);
                }
            }
        }
    }

    fn hash<T: std::hash::Hash>(&self, t: &T) -> u64 {
        self.build_hasher.hash_one(t)
    }
}
