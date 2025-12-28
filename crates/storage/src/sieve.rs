//! A sharded, concurrent cache implementation using the SIEVE eviction
//! algorithm.
//!
//! This module provides [`Sieve`], a high-performance cache that integrates
//! with key-value databases to provide transparent caching with lazy loading.
//!
//! # SIEVE Algorithm
//!
//! SIEVE (Simple and Efficient Cache) is a modern eviction algorithm that
//! provides excellent hit rates comparable to LRU (Least Recently Used) with
//! significantly lower overhead. Unlike LRU, SIEVE uses a single "visited" bit
//! per entry and a hand pointer that sweeps through entries, giving unvisited
//! entries a "second chance" before eviction.
//!
//! Key benefits of SIEVE:
//! - Lower overhead than LRU (no list reordering on access)
//! - Better hit rates than simple FIFO
//! - Scan-resistant (one-time accesses don't pollute the cache)
//!
//! # Sharding
//!
//! The cache is divided into multiple shards to reduce lock contention during
//! concurrent access. Each shard maintains its own SIEVE state and capacity,
//! allowing parallel operations on different keys to proceed without blocking
//! each other.

use core::fmt;
use std::{
    borrow::Borrow,
    collections::{HashMap, HashSet, hash_map::Entry},
    fmt::Debug,
    hash::{BuildHasher, BuildHasherDefault, Hash},
    option::Option,
    sync::{Arc, atomic::AtomicBool},
};

use dashmap::DashSet;
use parking_lot::{
    Condvar, MappedRwLockReadGuard, Mutex, RwLockReadGuard, RwLockWriteGuard,
};

use crate::{
    kv_database::{Column, KeyOfSet, KeyOfSetMode, KvDatabase, Normal},
    sharded::Sharded,
};

/// A marker trait representing the storage mode of a column.
///
/// This trait connects a column's storage mode ([`Normal`] or [`KeyOfSet`]) to
/// the type of value stored in the cache backing storage. Different storage
/// modes may wrap values differently for caching purposes.
///
/// # Implementations
///
/// - For [`Normal`] mode: Values are wrapped in `Option<V>` to distinguish
///   between "not in cache" and "known to not exist in database".
/// - For [`KeyOfSet<T>`] mode: Values are stored directly as the collection
///   type (e.g., `HashSet<T>`) since sets always have a valid empty state.
pub trait BackingStorage<V> {
    /// The type of value stored in the backing storage for this mode.
    ///
    /// This may differ from the column's value type to accommodate caching
    /// semantics (e.g., wrapping in `Option`).
    type Value;
}

impl<V> BackingStorage<V> for Normal {
    type Value = Option<V>;
}

impl<ColumnValue, V> BackingStorage<ColumnValue> for KeyOfSet<V> {
    type Value = ColumnValue;
}

/// A trait for removing an element from a set-like collection.
///
/// This trait provides a uniform interface for removing elements from various
/// set implementations (e.g., `HashSet`, `DashSet`) used in the SIEVE cache.
/// It enables the cache to work with different set types while providing
/// consistent element removal semantics.
///
/// # Type Parameters
///
/// - `Element`: The type of elements stored in the set.
pub trait RemoveElementFromSet {
    /// The type of elements stored in the set.
    type Element;

    /// Removes an element from the set.
    ///
    /// # Returns
    ///
    /// - `true` if the element was present and removed
    /// - `false` if the element was not found in the set
    fn remove_element<Q: Hash + Eq + ?Sized>(&mut self, element: &Q) -> bool
    where
        Self::Element: Borrow<Q>;
}

impl<T: Eq + Hash, S: BuildHasher> RemoveElementFromSet for HashSet<T, S> {
    type Element = T;

    fn remove_element<Q: Hash + Eq + ?Sized>(&mut self, element: &Q) -> bool
    where
        Self::Element: Borrow<Q>,
    {
        self.remove(element)
    }
}

impl<T: Eq + Hash, S: BuildHasher + Clone> RemoveElementFromSet
    for DashSet<T, S>
{
    type Element = T;

    fn remove_element<Q: Hash + Eq + ?Sized>(&mut self, element: &Q) -> bool
    where
        Self::Element: Borrow<Q>,
    {
        self.remove(element).is_some()
    }
}

#[derive(Clone)]
struct Notify(Arc<(Condvar, Mutex<bool>)>);

impl Notify {
    pub fn new() -> Self { Self(Arc::new((Condvar::new(), Mutex::new(false)))) }

    pub fn notify_waiters(&self) {
        let (condvar, lock) = &*self.0;

        let mut notified = lock.lock();
        *notified = true;
        condvar.notify_all();
    }

    pub fn wait(&self) {
        let (condvar, lock) = &*self.0;

        let mut notified = lock.lock();

        while !*notified {
            condvar.wait(&mut notified);
        }
    }
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

        let Some(notify) = in_flight.remove(self.key) else {
            return;
        };

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
    /// This function is designed to be cancellation-safe. If the function is
    /// dropped while a database fetch is in progress, the internal state
    /// remains consistent. However, other threads waiting for the same key
    /// may be woken up and will retry fetching the value.
    ///
    /// # Deadlocks
    ///
    /// This function **can deadlock** if you hold another read guard from this
    /// cache while calling this function. Even though this function returns a
    /// read lock, its internal implementation may acquire write locks to insert
    /// new values on cache misses.
    #[allow(clippy::cast_possible_truncation)] // from u64 to usize
    pub fn get_normal(
        &self,
        key: &C::Key,
    ) -> Option<MappedRwLockReadGuard<'_, C::Value>>
    where
        C: Column<Mode = Normal>,
        C::Mode: BackingStorage<C::Value, Value = Option<C::Value>>,
    {
        self.get_loop(
            key,
            |shard_index| {
                let lock = self.shards.read_shard(shard_index);

                RwLockReadGuard::try_map(lock, |x| match x.get(key) {
                    Retrieve::Hit(entry) => Some(entry),
                    Retrieve::Miss => None,
                })
                .map_or_else(
                    |_| None,
                    |lock| {
                        Some(
                            MappedRwLockReadGuard::try_map(
                                lock,
                                Option::as_ref,
                            )
                            .ok(),
                        )
                    },
                )
            },
            |_| self.backing_db.get::<C>(key),
        )
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
    /// This function is designed to be cancellation-safe. If the function is
    /// dropped while a database fetch is in progress, the internal state
    /// remains consistent. However, other threads waiting for the same key
    /// may be woken up and will retry fetching the value.
    ///
    /// # Deadlocks
    ///
    /// This function **can deadlock** if you hold another read guard from this
    /// cache while calling this function. Even though this function returns a
    /// read lock, its internal implementation may acquire write locks to insert
    /// new values on cache misses.
    pub fn get_set(&self, key: &C::Key) -> MappedRwLockReadGuard<'_, C::Value>
    where
        C: Column,
        C::Mode: KeyOfSetMode + BackingStorage<C::Value, Value = C::Value>,
        C::Value: FromIterator<<<C as Column>::Mode as KeyOfSetMode>::Value>,
    {
        self.get_loop(
            key,
            |index| {
                let lock = self.shards.read_shard(index);

                RwLockReadGuard::try_map(lock, |x| match x.get(key) {
                    Retrieve::Hit(entry) => Some(entry),
                    Retrieve::Miss => None,
                })
                .ok()
            },
            |_| self.backing_db.collect_key_of_set::<C>(key),
        )
    }

    /// Inserts or updates a value in the cache for the given key.
    ///
    /// This method allows direct insertion or update of a value in the cache,
    /// bypassing the backing database. If there is an ongoing fetch for the
    /// same key (i.e., another thread is currently fetching the value from
    /// the database), this method will take over the fetch lock, perform
    /// the insertion, and notify any waiters. If there is no ongoing fetch,
    /// it simply inserts or updates the value in the appropriate shard.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to insert or update in the cache.
    /// * `value` - The value to associate with the key. `None` can be used to
    ///   represent deletion or absence of a value, depending on the column
    ///   semantics.
    ///
    /// # Durability
    ///
    /// This method only updates the in-memory cache and does not persist
    /// changes to the backing database. It is the caller's responsibility to
    /// ensure that the backing database is updated accordingly before or after
    /// calling this method, if persistence is required.
    ///
    /// # Concurrency
    ///
    /// If another thread is currently fetching the value for the same key, this
    /// method will take over the fetch operation, perform the insertion, and
    /// notify all waiting threads. Otherwise, it performs the insertion
    /// directly.
    ///
    /// # Eviction
    ///
    /// If the cache shard is at capacity, this insertion may trigger eviction
    /// of another entry according to the SIEVE algorithm.
    pub fn put(&self, key: C::Key, value: Option<C::Value>)
    where
        C: Column<Mode = Normal>,
        C::Mode: BackingStorage<C::Value, Value = Option<C::Value>>,
    {
        let mut in_flight = self.in_flight.lock();

        match in_flight.entry(key.clone()) {
            // steal the fetching lock to perform the insertion
            Entry::Occupied(occupied_entry) => {
                let notify = occupied_entry.remove();

                {
                    let shard_index = self.shard_index(&key);
                    let mut lock = self.shards.write_shard(shard_index);
                    lock.insert(key, value);
                }

                notify.notify_waiters();
            }

            // do nothing with the inflight, but has to hold the lock
            Entry::Vacant(_) => {
                let shard_index = self.shard_index(&key);
                let mut lock = self.shards.write_shard(shard_index);
                lock.insert(key, value);
            }
        }
    }

    /// Inserts elements into a set stored in the cache for the given key.
    ///
    /// This method is designed for columns using the [`KeyOfSet`] storage mode.
    /// It retrieves the set associated with the key (loading it from the
    /// backing database if necessary) and adds the provided elements to it.
    ///
    /// # Type Requirements
    ///
    /// This method requires that:
    /// - `C` implements [`Column`] with a [`KeyOfSetMode`] storage mode
    /// - `C::Value` implements `Extend` to allow adding elements to the set
    /// - `C::Value` implements `FromIterator` for loading from the database
    ///
    /// # Arguments
    ///
    /// - `key` - A reference to the key identifying the set in the cache
    /// - `element` - An iterable of elements to insert into the set
    ///
    /// # Behavior
    ///
    /// 1. If the key is already cached, the elements are added to the existing
    ///    set
    /// 2. If the key is not cached, the set is loaded from the backing database
    ///    using [`KvDatabase::collect_key_of_set`], then the elements are added
    /// 3. The operation holds a write lock on the shard during insertion to
    ///    ensure thread safety
    ///
    /// # Durability
    ///
    /// This method only updates the in-memory cache. The caller is responsible
    /// for persisting changes to the backing database if needed.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Insert values into a set associated with a key
    /// cache.insert_set_element(&my_key, vec![elem1, elem2]);
    /// ```
    ///
    /// # Concurrency
    ///
    /// This method coordinates with other cache operations to ensure only one
    /// thread fetches data from the backing database if multiple threads access
    /// the same key simultaneously.
    pub fn insert_set_element(
        &self,
        key: &C::Key,
        element: impl IntoIterator<Item = <C::Mode as KeyOfSetMode>::Value>,
    ) where
        C: Column,
        C::Mode: KeyOfSetMode + BackingStorage<C::Value, Value = C::Value>,
        C::Value: FromIterator<<<C as Column>::Mode as KeyOfSetMode>::Value>
            + Extend<<<C as Column>::Mode as KeyOfSetMode>::Value>
            + Default,
    {
        let mut write_lock = self.get_loop(
            key,
            |index| {
                let lock = self.shards.write_shard(index);

                RwLockWriteGuard::try_map(lock, |x| match x.get_mut(key) {
                    RetrieveMut::Hit(entry) => Some(entry),
                    RetrieveMut::Miss => None,
                })
                .ok()
            },
            |_| self.backing_db.collect_key_of_set::<C>(key),
        );

        write_lock.extend(element);
    }

    /// Removes an element from a set stored in the cache for the given key.
    ///
    /// This method is designed for columns using the [`KeyOfSet`] storage mode.
    /// It retrieves the set associated with the key (loading it from the
    /// backing database if necessary) and removes the specified element
    /// from it.
    ///
    /// # Type Requirements
    ///
    /// This method requires that:
    /// - `C` implements [`Column`] with a [`KeyOfSetMode`] storage mode
    /// - `C::Value` implements [`RemoveElementFromSet`] for element removal
    /// - `C::Value` implements `FromIterator` for loading from the database
    ///
    /// # Arguments
    ///
    /// - `key` - A reference to the key identifying the set in the cache
    /// - `element` - A reference to the element to remove from the set
    ///
    /// # Behavior
    ///
    /// 1. If the key is already cached, attempts to remove the element from the
    ///    existing set
    /// 2. If the key is not cached, loads the set from the backing database
    ///    using [`KvDatabase::collect_key_of_set`], then attempts to remove the
    ///    element
    /// 3. If the element is not in the set, the operation has no effect
    /// 4. The operation holds a write lock on the shard during removal to
    ///    ensure thread safety
    ///
    /// # Durability
    ///
    /// This method only updates the in-memory cache. The caller is responsible
    /// for persisting changes to the backing database if needed.
    ///
    /// # Concurrency
    ///
    /// This method coordinates with other cache operations to ensure only one
    /// thread fetches data from the backing database if multiple threads access
    /// the same key simultaneously.
    pub fn remove_set_element<Q: ?Sized + Eq + Hash>(
        &self,
        key: &C::Key,
        element: &Q,
    ) where
        C: Column,
        C::Mode: KeyOfSetMode + BackingStorage<C::Value, Value = C::Value>,
        C::Value: RemoveElementFromSet<Element = <C::Mode as KeyOfSetMode>::Value>
            + FromIterator<<<C as Column>::Mode as KeyOfSetMode>::Value>,
        <C::Mode as KeyOfSetMode>::Value: Borrow<Q>,
    {
        let mut write_lock = self.get_loop(
            key,
            |index| {
                let lock = self.shards.write_shard(index);

                RwLockWriteGuard::try_map(lock, |x| match x.get_mut(key) {
                    RetrieveMut::Hit(entry) => Some(entry),
                    RetrieveMut::Miss => None,
                })
                .ok()
            },
            |_| self.backing_db.collect_key_of_set::<C>(key),
        );

        write_lock.remove_element(element);
    }

    fn get_loop<T>(
        &self,
        key: &C::Key,
        mut fast_path: impl FnMut(usize) -> Option<T>,
        mut slow_path: impl FnMut(
            usize,
        )
            -> <C::Mode as BackingStorage<C::Value>>::Value,
    ) -> T {
        let shard_index = self.shard_index(key);

        loop {
            if let Some(fast_path_result) = fast_path(shard_index) {
                return fast_path_result;
            }

            let Some(mut fetching_guard) = self.fetching_lock(key) else {
                // some thread has already fetched the value, retry
                continue;
            };

            let value = slow_path(shard_index);

            // hold the lock while inserting to avoid races
            {
                let mut in_flight_lock = self.in_flight.lock();

                // defuse the guard
                fetching_guard.defused = true;

                let Some(notify) = in_flight_lock.remove(key) else {
                    // the `put` operation has overwritten this entry, discard
                    continue;
                };

                // insert into the cache
                let mut lock = self.shards.write_shard(shard_index);
                lock.insert(key.clone(), value);

                notify.notify_waiters();
            }

            // retry again to get the read lock
        }
    }

    #[allow(clippy::await_holding_lock)] // FALSE POSITIVE
    fn fetching_lock<'k>(
        &self,
        key: &'k C::Key,
    ) -> Option<FetchingGuard<'k, '_, C::Key>> {
        let mut in_flight = self.in_flight.lock();

        if let Some(notify) = in_flight.get(key) {
            let notified = notify.clone();
            drop(in_flight);

            notified.wait();

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

enum RetrieveMut<'x, C: Column, M: BackingStorage<C::Value>> {
    Hit(&'x mut M::Value),
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

    fn get_mut(&mut self, key: &C::Key) -> RetrieveMut<'_, C, M> {
        if let Some(&index) = self.map.get(key) {
            let node = self.nodes[index].as_mut().unwrap();
            node.visited.store(true, std::sync::atomic::Ordering::Relaxed);

            return RetrieveMut::Hit(&mut node.value);
        }

        RetrieveMut::Miss
    }
}
