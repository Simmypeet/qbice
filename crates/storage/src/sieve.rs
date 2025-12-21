//! A sharded cache implementation using the SIEVE eviction algorithm.

use core::fmt;
use std::{
    collections::HashMap,
    fmt::Debug,
    hash::{BuildHasher, BuildHasherDefault, Hash},
    option::Option,
    sync::{Arc, atomic::AtomicBool},
};

use parking_lot::{MappedRwLockReadGuard, Mutex, RwLockReadGuard};
use qbice_serialize::{Decode, Encode};
use tokio::sync::Notify;

use crate::{
    kv_database::{Column, KeyOfSet, KvDatabase, Normal},
    sharded::Sharded,
};

/// A marker trait representing the storage mode of a column.
pub trait BackingStorage<V> {
    /// The type of value stored in the backing storage for this mode.
    type Value;
}

impl<V> BackingStorage<V> for Normal {
    type Value = Option<V>;
}

impl<ColumnValue, V> BackingStorage<ColumnValue> for KeyOfSet<V> {
    type Value = ColumnValue;
}

/// A sharded cache implementation using the SIEVE eviction algorithm.
///
/// This cache provides concurrent access to cached values with lazy loading
/// from a backing database. It uses the SIEVE (Simple and Efficient Cache)
/// algorithm for cache eviction, which provides good hit rates with minimal
/// overhead.
///
/// # Type Parameters
///
/// * `C` - The column type that defines the key-value schema for the cache.
/// * `DB` - The backing database type used to fetch values on cache misses.
/// * `S` - The hasher builder type used for hashing keys (defaults to
///   `FxHasher`).
///
/// # Sharding
///
/// The cache is divided into multiple shards to reduce lock contention during
/// concurrent access. Each shard maintains its own SIEVE state and can be
/// accessed independently.
#[allow(clippy::type_complexity)] // shards field
pub struct Sieve<C: Column, DB, S = BuildHasherDefault<fxhash::FxHasher>>
where
    C::Mode: BackingStorage<C::Value>,
{
    shards: Sharded<SieveShard<C, C::Mode, S>>,
    hasher_builder: S,

    backing_db: Arc<DB>,

    in_flight: Mutex<HashMap<C::Key, Arc<Notify>>>,
}

impl<C: Column, DB, S> Debug for Sieve<C, DB, S>
where
    C::Mode: BackingStorage<C::Value>,
    C::Key: Debug,
    C::Value: Debug,
    <C::Mode as BackingStorage<C::Value>>::Value: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut map = f.debug_map();

        for shard in self.shards.iter_read_shards() {
            for (key, &index) in &shard.map {
                let value = &shard.nodes[index].as_ref().unwrap().value;

                map.entry(key, value);
            }
        }

        map.finish_non_exhaustive()
    }
}

struct FetchingGuard<'k, 'x, K>
where
    K: Eq + Hash,
{
    in_flight: &'x Mutex<HashMap<K, Arc<Notify>>>,
    key: &'k K,
    defused: bool,
}

impl<K: Eq + Hash> Drop for FetchingGuard<'_, '_, K> {
    fn drop(&mut self) {
        if self.defused {
            return;
        }

        let mut in_flight = self.in_flight.lock();
        let notify = in_flight.remove(self.key).unwrap();
        notify.notify_waiters();
    }
}

impl<K: Eq + Hash> FetchingGuard<'_, '_, K> {
    pub fn done(mut self) {
        self.defused = true;

        let mut in_flight = self.in_flight.lock();
        let notify = in_flight.remove(self.key).unwrap();
        notify.notify_waiters();
    }
}

impl<C: Column, DB, S: Clone> Sieve<C, DB, S>
where
    C::Mode: BackingStorage<C::Value>,
{
    /// Creates a new [`Sieve`] cache with the specified configuration.
    ///
    /// # Arguments
    ///
    /// * `total_capacity` - The total number of entries the cache can hold
    ///   across all shards. The capacity is distributed evenly among shards.
    /// * `shard_amount` - The number of shards to divide the cache into. Using
    ///   more shards reduces lock contention but increases memory overhead.
    ///   This should typically be a power of two for efficient shard selection.
    /// * `backing_db` - The database to fetch values from on cache misses.
    /// * `hasher_builder` - The hasher builder used to hash keys for shard
    ///   selection and internal hash maps.
    ///
    /// # Returns
    ///
    /// A new `Sieve` cache instance ready to use.
    pub fn new(
        total_capacity: usize,
        shard_amount: usize,
        backing_db: Arc<DB>,
        hasher_builder: S,
    ) -> Self {
        let per_shard = total_capacity.div_ceil(shard_amount);

        Self {
            shards: Sharded::new(shard_amount, |_| {
                SieveShard::new(per_shard, hasher_builder.clone())
            }),
            hasher_builder,
            backing_db,

            in_flight: Mutex::new(HashMap::new()),
        }
    }
}

impl<C: Column, DB: KvDatabase, S: BuildHasher> Sieve<C, DB, S>
where
    C::Mode: BackingStorage<C::Value>,
{
    fn shard_index(&self, key: &C::Key) -> usize {
        self.shards.shard_index(self.hasher_builder.hash_one(key))
    }

    /// Retrieves a value from the cache, fetching from the backing database if
    /// not present.
    ///
    /// This method first checks if the value exists in the cache. On a cache
    /// hit, it returns immediately with a read guard to the cached value.
    /// On a cache miss, it fetches the value from the backing database,
    /// inserts it into the cache, and returns the result.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to look up in the cache.
    ///
    /// # Returns
    ///
    /// * `Some(ReadGuard)` - If the key exists (either in cache or backing
    ///   database), returns a guard providing read access to the value.
    /// * `None` - If the key does not exist in the backing database.
    ///
    /// # Concurrency
    ///
    /// Multiple concurrent calls with the same key will coordinate to avoid
    /// duplicate database fetches. Only one fetch will be performed, and other
    /// callers will wait for the result.
    ///
    /// # Cancellation Safety
    ///
    /// The function is designed to be cancellation-safe. When the database
    /// fetch is in progress, if cancellation occurs, the internal state remains
    /// consistent. However, other possible other callers waiting for the same
    /// key may be woken up and will retry fetching the value.
    ///
    /// # Deadlocks
    ///
    /// This function **can deadlock** if held another [`ReadGuard`] while
    /// calling this function. Even though, the function returns read lock,
    /// however, its internal state mutates the cache by inserting new
    /// values on cache misses.
    #[allow(clippy::cast_possible_truncation)] // from u64 to usize
    pub async fn get_normal(
        &self,
        key: &C::Key,
    ) -> Option<MappedRwLockReadGuard<'_, C::Value>>
    where
        C: Column<Mode = Normal>,
        C::Mode: BackingStorage<C::Value, Value = Option<C::Value>>,
    {
        let shard_index = self.shard_index(key);

        loop {
            let lock = self.shards.read_shard(shard_index);

            if let Ok(lock) =
                RwLockReadGuard::try_map(lock, |x| match x.get(key) {
                    Retrieve::Hit(entry) => Some(entry),
                    Retrieve::Miss => None,
                })
            {
                return MappedRwLockReadGuard::try_map(lock, Option::as_ref)
                    .ok();
            }

            let Some(fetching_guard) = self.fetching_lock(key).await else {
                // some thread has already fetched the value, retry
                continue;
            };

            let value = self.backing_db.get::<C>(key).await;

            {
                let mut lock = self.shards.write_shard(shard_index);
                lock.insert(key.clone(), value.clone());
            }

            fetching_guard.done();

            // retry again to get the read lock
        }
    }

    /// Retrieves a set value from the cache, fetching from the backing database
    /// if not present.
    ///
    /// This method is designed for columns using the [`KeyOfSet`] storage mode,
    /// where each key maps to a collection of values. On a cache hit, it
    /// returns immediately with a read guard to the cached collection. On a
    /// cache miss, it fetches all members of the set from the backing
    /// database using [`KvDatabase::collect_key_of_set`], inserts the
    /// result into the cache, and returns the collection.
    ///
    /// # Type Parameters
    ///
    /// * `V` - The element type stored in the set. Must implement [`Encode`],
    ///   [`Decode`], [`Clone`], [`Send`], [`Sync`], and have a `'static`
    ///   lifetime.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to look up in the cache.
    ///
    /// # Returns
    ///
    /// A read guard providing access to the cached set value. Unlike
    /// [`get_normal`](Self::get_normal), this always returns a value (possibly
    /// an empty collection) since the set storage mode always has a valid
    /// default state.
    ///
    /// # Concurrency
    ///
    /// Multiple concurrent calls with the same key will coordinate to avoid
    /// duplicate database fetches. Only one fetch will be performed, and other
    /// callers will wait for the result.
    ///
    /// # Cancellation Safety
    ///
    /// The function is designed to be cancellation-safe. When the database
    /// fetch is in progress, if cancellation occurs, the internal state remains
    /// consistent. However, other possible callers waiting for the same
    /// key may be woken up and will retry fetching the value.
    ///
    /// # Deadlocks
    ///
    /// This function **can deadlock** if held another [`ReadGuard`] while
    /// calling this function. Even though the function returns a read lock,
    /// its internal state mutates the cache by inserting new values on cache
    /// misses.
    pub async fn get_set<V: Encode + Decode + Clone + Send + Sync + 'static>(
        &self,
        key: &C::Key,
    ) -> MappedRwLockReadGuard<'_, C::Value>
    where
        C: Column<Mode = KeyOfSet<V>>,
        C::Value: Extend<V> + Default,
        C::Mode: BackingStorage<C::Value, Value = C::Value>,
    {
        let shard_index = self.shard_index(key);

        loop {
            let lock = self.shards.read_shard(shard_index);

            if let Ok(lock) =
                RwLockReadGuard::try_map(lock, |x| match x.get(key) {
                    Retrieve::Hit(entry) => Some(entry),
                    Retrieve::Miss => None,
                })
            {
                return lock;
            }

            let Some(fetching_guard) = self.fetching_lock(key).await else {
                // some thread has already fetched the value, retry
                continue;
            };

            let value = self.backing_db.collect_key_of_set::<V, C>(key).await;

            {
                let mut lock = self.shards.write_shard(shard_index);
                lock.insert(key.clone(), value.clone());
            }

            fetching_guard.done();

            // retry again to get the read lock
        }
    }

    #[allow(clippy::await_holding_lock)] // FALSE POSITIVE
    async fn fetching_lock<'k>(
        &self,
        key: &'k C::Key,
    ) -> Option<FetchingGuard<'k, '_, C::Key>> {
        let mut in_flight = self.in_flight.lock();

        if let Some(notify) = in_flight.get(key) {
            let notified = notify.clone().notified_owned();
            drop(in_flight);

            notified.await;

            None
        } else {
            let notify = Arc::new(Notify::new());
            in_flight.insert(key.clone(), notify);

            Some(FetchingGuard {
                key,
                defused: false,
                in_flight: &self.in_flight,
            })
        }
    }
}

struct Node<C: Column, M: BackingStorage<C::Value>> {
    key: C::Key,
    value: M::Value,
    visited: AtomicBool,
}

type Index = usize;

struct SieveShard<
    C: Column,
    M: BackingStorage<C::Value>,
    S = BuildHasherDefault<fxhash::FxHasher>,
> {
    map: HashMap<C::Key, Index, S>,
    nodes: Vec<Option<Node<C, M>>>,
    hand: Index,

    active: usize,
}

impl<C: Column, M: BackingStorage<C::Value>, S> SieveShard<C, M, S> {
    fn new(capacity: usize, hasher: S) -> Self {
        let mut nodes = Vec::with_capacity(capacity);
        nodes.resize_with(capacity, || None);

        Self { nodes, hand: 0, active: 0, map: HashMap::with_hasher(hasher) }
    }
}

enum Retrieve<'x, C: Column, M: BackingStorage<C::Value>> {
    Hit(&'x M::Value),
    Miss,
}

impl<C: Column, M: BackingStorage<C::Value>, S: BuildHasher>
    SieveShard<C, M, S>
{
    fn insert(&mut self, key: C::Key, value: M::Value) {
        // scan for an empty slot
        loop {
            let index = self.hand;
            self.hand = (self.hand + 1) % self.nodes.len();

            if let Some(node) = &mut self.nodes[index] {
                if node
                    .visited
                    .swap(false, std::sync::atomic::Ordering::Relaxed)
                {
                    // give a second chance
                    continue;
                }

                // evict this node
                self.map.remove(&node.key);
                *node = Node {
                    key: key.clone(),
                    value,
                    visited: AtomicBool::new(true),
                };
                self.map.insert(key, index);
                return;
            }

            // empty slot found
            self.nodes[index] = Some(Node {
                key: key.clone(),
                value,
                visited: AtomicBool::new(true),
            });
            self.map.insert(key, index);
            self.active += 1;
            return;
        }
    }

    fn get(&self, key: &C::Key) -> Retrieve<'_, C, M> {
        if let Some(&index) = self.map.get(key) {
            let node = self.nodes[index].as_ref().unwrap();
            node.visited.store(true, std::sync::atomic::Ordering::Relaxed);
            return Retrieve::Hit(&node.value);
        }

        Retrieve::Miss
    }
}
