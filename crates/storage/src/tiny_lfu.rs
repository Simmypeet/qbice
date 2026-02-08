//! Contains the TinyLFU cache implementation designed specifically for
//! write-behind caching layers.

use std::{
    collections::hash_map,
    hash::{BuildHasher, Hash},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize},
    },
};

use crossbeam_utils::CachePadded;
use fxhash::{FxBuildHasher, FxHashMap};

use crate::{
    sharded::Sharded,
    tiny_lfu::policy::{Policy, PolicyMessage, WriteMessage},
};

mod lru;
mod policy;
mod read_buffer;
mod single_flight;
mod sketch;
mod write_buffer;

/// Specifies the strategy for unpining entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum UnpinStrategy {
    /// Periodically asks the lifecycle listener if the entry has already
    /// unpinned by calling [`LifecycleListener::is_pinned`]
    Poll,

    /// The client is responsible for unpining entries by calling
    /// [`TinyLFU::unpin`]
    Notify,
}

const MAINTENANCE_BATCH_SIZE: usize = 32;

/// A listener trait for cache entry lifecycle events.
pub trait LifecycleListener<K, V>: Default {
    /// Determines if the given entry is currently pinned.
    ///
    /// If "pinned", the entry will not be evicted from the cache.
    fn is_pinned(&self, key: &K, value: &V) -> bool;
}

/// The default lifecycle listener which does not pin any entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct DefaultLifecycleListener;

impl<K, V> LifecycleListener<K, V> for DefaultLifecycleListener {
    fn is_pinned(&self, _key: &K, _value: &V) -> bool { false }
}

/// A TinyLFU (Tiny Least Frequently Used) cache implementation with
/// support for pinned entries and negative caching.
///
/// This designed specifically for caching layers with write-behind semantics,
/// where entries may be temporarily pinned to avoid eviction during ongoing
/// write operations, and negative caching is used to remember absent entries
/// without storing actual values.
pub struct TinyLFU<
    K: Clone + Eq + Hash + Send + Sync + 'static,
    V: Send + Sync + 'static,
    L: LifecycleListener<K, V> + Send + Sync + 'static = DefaultLifecycleListener,
> {
    inner: Arc<TinyLFUInner<K, V, L>>,

    sender: Option<crossbeam::channel::Sender<()>>,
    join_handle: Option<std::thread::JoinHandle<()>>,
}

impl<
    K: Clone + Eq + Hash + Send + Sync + 'static,
    V: Send + Sync + 'static,
    L: LifecycleListener<K, V> + Send + Sync + 'static,
> std::fmt::Debug for TinyLFU<K, V, L>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TinyLFU").finish_non_exhaustive()
    }
}

impl<
    K: Clone + Eq + Hash + Send + Sync + 'static,
    V: Send + Sync + 'static,
    L: LifecycleListener<K, V> + Send + Sync + 'static,
> TinyLFU<K, V, L>
{
    /// Creates a new TinyLFU cache with the specified capacity and shard
    /// count.
    #[must_use]
    pub fn new(
        capacity: usize,
        shard_count: usize,
        unpin_strategy: UnpinStrategy,
    ) -> Self {
        let (sender, receiver) = crossbeam::channel::bounded(1);
        let inner =
            Arc::new(TinyLFUInner::new(capacity, shard_count, unpin_strategy));

        Self {
            inner: inner.clone(),
            sender: Some(sender),
            join_handle: Some(Self::maintenance_loop(inner, receiver)),
        }
    }
}

impl<
    K: Clone + Eq + Hash + Send + Sync + 'static,
    V: Send + Sync + 'static,
    L: LifecycleListener<K, V> + Send + Sync + 'static,
> Drop for TinyLFU<K, V, L>
{
    fn drop(&mut self) {
        // signal the maintenance thread to exit
        drop(self.sender.take());
        self.join_handle.take().unwrap().join().unwrap();
    }
}

impl<
    K: Clone + Eq + Hash + Send + Sync + 'static,
    V: Send + Sync + 'static,
    L: LifecycleListener<K, V> + Send + Sync + 'static,
> TinyLFU<K, V, L>
{
    fn maintenance_loop(
        inner: Arc<TinyLFUInner<K, V, L>>,
        receiver: crossbeam::channel::Receiver<()>,
    ) -> std::thread::JoinHandle<()> {
        std::thread::Builder::new()
            .name("tiny_lfu_maintenance".to_string())
            .spawn(move || {
                while receiver.recv() == Ok(()) {
                    let mut lock = inner.policy.lock();
                    inner.process_policy_message(&mut lock);

                    inner
                        .maintenance_flag
                        .store(false, std::sync::atomic::Ordering::SeqCst);
                }
            })
            .expect("Failed to spawn TinyLFU maintenance thread")
    }

    fn try_maintenance(&self, policy_message: Option<PolicyMessage<K>>) {
        match policy_message {
            Some(PolicyMessage::ReadHit(hit)) => {
                self.inner.read_buffer.push(hit);
            }
            Some(PolicyMessage::Write(write_message)) => {
                self.inner.write_buffer.push(write_message);
            }
            None => {}
        }

        // not yet reached the maintenance threshold
        if self.inner.write_buffer.len() <= MAINTENANCE_BATCH_SIZE
            && self.inner.read_buffer.len() <= MAINTENANCE_BATCH_SIZE
        {
            return;
        }

        // attempt to acquire the maintenance flag
        if self.inner.maintenance_flag.compare_exchange(
            false,
            true,
            std::sync::atomic::Ordering::SeqCst,
            std::sync::atomic::Ordering::SeqCst,
        ) == Ok(false)
        {
            self.sender
                .as_ref()
                .expect("TinyLFU sender missing")
                .try_send(())
                .expect("Failed to send TinyLFU maintenance signal");
        }
    }
}

/// A TinyLFU (Tiny Least Frequently Used) cache implementation with
/// support for pinned entries and negative caching.
///
/// This designed specifically for caching layers with write-behind semantics,
/// where entries may be temporarily pinned to avoid eviction during ongoing
/// write operations, and negative caching is used to remember absent entries
/// without storing actual values.
struct TinyLFUInner<K, V, L = DefaultLifecycleListener> {
    storage: CachePadded<Sharded<FxHashMap<K, V>>>,

    read_buffer: CachePadded<read_buffer::ReadBuffer<K>>,
    write_buffer: CachePadded<write_buffer::UnboundedBuffer<WriteMessage<K>>>,

    policy: CachePadded<parking_lot::Mutex<Policy<K>>>,

    // true=running, false=idle
    maintenance_flag: AtomicBool,

    evicted_count: AtomicUsize,
    unpin_strategy: UnpinStrategy,

    lifecycle_listener: L,
    build_hasher: FxBuildHasher,
}

impl<K, V, L> std::fmt::Debug for TinyLFUInner<K, V, L> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TinyLFUInner").finish_non_exhaustive()
    }
}

impl<K, V, L: Default> TinyLFUInner<K, V, L> {
    /// Creates a new TinyLFUInner cache with the specified capacity and shard
    /// count.
    #[must_use]
    pub fn new(
        capacity: usize,
        shard_count: usize,
        unpin_strategy: UnpinStrategy,
    ) -> Self {
        Self {
            storage: CachePadded::new(Sharded::new(shard_count, |_| {
                FxHashMap::default()
            })),
            read_buffer: CachePadded::new(read_buffer::ReadBuffer::new(
                std::thread::available_parallelism()
                    .map(std::num::NonZero::get)
                    .unwrap_or(4),
                64,
            )),
            write_buffer: CachePadded::new(write_buffer::UnboundedBuffer::new()),

            policy: CachePadded::new(parking_lot::Mutex::new(Policy::new(
                capacity,
            ))),
            lifecycle_listener: L::default(),
            unpin_strategy,
            build_hasher: FxBuildHasher::default(),
            evicted_count: AtomicUsize::new(0),
            maintenance_flag: AtomicBool::new(false),
        }
    }
}

impl<K, V, L> Drop for TinyLFUInner<K, V, L> {
    fn drop(&mut self) {
        println!(
            "TinyLFU<{}, {}> evicted {} items during its lifetime, {} entries \
             left in cache, and {} message leeft in the cache.\n",
            std::any::type_name::<K>(),
            std::any::type_name::<V>(),
            self.evicted_count.load(std::sync::atomic::Ordering::Relaxed),
            self.storage
                .iter_read_shards()
                .map(|shard| shard.len())
                .sum::<usize>(),
            self.write_buffer.len(),
        );
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
        *self.message = Some(PolicyMessage::Write(WriteMessage::Removed(key)));

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

        *self.message = Some(PolicyMessage::Write(WriteMessage::Insert(key)));
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

impl<
    K: std::hash::Hash + Eq + Clone + Send + Sync + 'static,
    V: Send + Sync + 'static,
    L: LifecycleListener<K, V> + Send + Sync + 'static,
> TinyLFU<K, V, L>
{
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
    pub fn get_map<T>(&self, key: &K, f: impl Fn(&V) -> T) -> Option<T> {
        let hash = self.inner.hash(key);
        let shard =
            self.inner.storage.read_shard(self.inner.storage.shard_index(hash));

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
        let hash = self.inner.hash(&key);
        let mut shard = self
            .inner
            .storage
            .write_shard(self.inner.storage.shard_index(hash));

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

    /// Notifies the cache that a key has been unpinned and should be removed or
    /// reinserted into the cache.
    pub fn unpin(&self, key: K) {
        self.try_maintenance(Some(PolicyMessage::Write(
            WriteMessage::Unpinned(key),
        )));
    }
}

impl<
    K: std::hash::Hash + Eq + Clone + Send + Sync + 'static,
    V: Send + Sync + 'static,
    L: LifecycleListener<K, V> + Send + Sync + 'static,
> TinyLFUInner<K, V, L>
{
    fn process_policy_message(&self, lock: &mut Policy<K>) {
        while let Some(message) = self.write_buffer.pop() {
            self.process_write(message, lock);
        }

        for read_k in self.read_buffer.drain() {
            self.process_message(PolicyMessage::ReadHit(read_k), lock);
        }

        if self.unpin_strategy == UnpinStrategy::Poll {
            // before dropping the lock, we can try to overflow trimming
            lock.attempt_to_trim_overflowing_pinned(self.remove_closure());
        }
    }

    fn process_write(&self, message: WriteMessage<K>, lock: &mut Policy<K>) {
        match message {
            WriteMessage::Insert(key) => {
                let hash = self.hash(&key);
                lock.on_write(
                    &key,
                    hash,
                    &self.build_hasher,
                    self.remove_closure(),
                );
            }

            WriteMessage::Unpinned(key) => {
                lock.unpin(&key, &self.build_hasher, self.remove_closure());
            }

            WriteMessage::Removed(key) => {
                lock.on_removed(&key);
            }
        }
    }

    fn process_message(&self, message: PolicyMessage<K>, lock: &mut Policy<K>) {
        match message {
            PolicyMessage::ReadHit(key) => {
                let hash = self.hash(&key);
                lock.on_read_hit(&key, hash);
            }

            PolicyMessage::Write(WriteMessage::Unpinned(key)) => {
                lock.unpin(&key, &self.build_hasher, self.remove_closure());
            }

            PolicyMessage::Write(WriteMessage::Insert(key)) => {
                let hash = self.hash(&key);
                lock.on_write(
                    &key,
                    hash,
                    &self.build_hasher,
                    self.remove_closure(),
                );
            }

            PolicyMessage::Write(WriteMessage::Removed(key)) => {
                lock.on_removed(&key);
            }
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn remove_closure(&self) -> impl Fn(&K) -> bool + '_ {
        // atomically determine if we can remove the key
        |evicted_key| {
            let shard_index = self.storage.shard_index(self.hash(evicted_key));
            let shard = self.storage.upgradable_read_shard(shard_index);

            if let Some(value) = shard.get(evicted_key) {
                // the element still exists, check if it is pinned
                if self.lifecycle_listener.is_pinned(evicted_key, value) {
                    return false;
                }
            } else {
                // has already been removed
                return true;
            }

            let mut shard =
                parking_lot::RwLockUpgradableReadGuard::upgrade(shard);

            match shard.entry(evicted_key.clone()) {
                hash_map::Entry::Occupied(mut occupied_entry) => {
                    let can_remove = {
                        let value = occupied_entry.get_mut();

                        // IMPORTANT: we must ask again under the write lock
                        !self.lifecycle_listener.is_pinned(evicted_key, value)
                    };

                    // IMPORTANT: we drop the value outside
                    // of the lock to avoid potential large
                    // value drop times blocking other
                    // operations
                    let value = if can_remove {
                        self.evicted_count
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                        Some(occupied_entry.remove_entry())
                    } else {
                        None
                    };

                    let ratio = shard.len() as f64 / shard.capacity() as f64;
                    if ratio < 0.25 {
                        let halved = shard.capacity() / 2;
                        shard.shrink_to(halved);
                    }

                    drop(shard);

                    drop(value);

                    can_remove
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
