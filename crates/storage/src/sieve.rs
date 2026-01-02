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
    any::Any,
    borrow::Borrow,
    collections::{HashMap, HashSet},
    fmt::Debug,
    hash::{BuildHasher, BuildHasherDefault, Hash},
    marker::PhantomData,
    option::Option,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize},
    },
};

use dashmap::DashSet;
use enum_as_inner::EnumAsInner;
use parking_lot::{MappedRwLockReadGuard, RwLockReadGuard};

use crate::{
    kv_database::{KeyOfSetColumn, KvDatabase, WideColumn, WideColumnValue},
    sharded::Sharded,
};

mod write_behind;

pub use write_behind::{BackgroundWriter, WriteBuffer};

/// An internal trait abstracting the key and value types used for backing
/// storage in the SIEVE cache.
pub trait BackingStorage {
    /// The type of keys used in the backing storage.
    type Key: Eq + Hash + Clone;

    /// The type of values stored in the backing storage.
    type Value;
}

/// An adaptor for using wide columns as backing storage in the SIEVE cache.
///
/// See [`WideColumnSieve`] for details.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WideColumnAdaptor<C>(pub PhantomData<C>);

impl<C: WideColumn> BackingStorage for WideColumnAdaptor<C> {
    type Key = (C::Key, C::Discriminant);
    type Value = Option<Box<dyn Any + Send + Sync>>;
}

/// An adaptor for using key-of-set columns as backing storage in the SIEVE
/// cache.
///
/// See [`KeyOfSetSieve`] for details.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KeyOfSetAdaptor<C>(pub PhantomData<(C,)>);

/// An enumeration representing set operations for key-of-set columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[allow(missing_docs)]
pub enum Operation {
    Insert,
    Delete,
}

/// An entry in the key-of-set cache, representing either the full set of
/// elements or a partial set of operations to apply.
#[derive(Debug, EnumAsInner)]
#[allow(missing_docs)]
pub enum KeyOfSetEntry<C: KeyOfSetContainer> {
    /// The complete set was fetched from the backing storage. Contains the full
    /// container.
    Full(C::Container),

    /// The complete set is not stored; instead, a list of operations to apply
    /// is kept.
    Partial(HashMap<C::Element, Operation>),
}

impl<C: KeyOfSetContainer> BackingStorage for KeyOfSetAdaptor<C> {
    type Key = C::Key;
    type Value = KeyOfSetEntry<C>;
}

/// A sharded sieve cache operating on wide columns data schema.
pub type WideColumnSieve<C, DB, S = BuildHasherDefault<fxhash::FxHasher>> =
    Sieve<WideColumnAdaptor<C>, DB, S>;

/// A sharded sieve cache operating on key-of-set columns data schema.
pub type KeyOfSetSieve<C, DB, S = BuildHasherDefault<fxhash::FxHasher>> =
    Sieve<KeyOfSetAdaptor<C>, DB, S>;

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
pub struct Sieve<
    B: BackingStorage,
    DB,
    S = BuildHasherDefault<fxhash::FxHasher>,
> {
    shards: Sharded<SieveShard<B, S>>,
    hasher_builder: S,

    backing_db: Arc<DB>,
}

impl<B: BackingStorage, DB, S> Debug for Sieve<B, DB, S>
where
    B::Key: Debug,
    B::Value: Debug,
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

impl<B: BackingStorage, DB, S: Clone> Sieve<B, DB, S> {
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
        }
    }
}

impl<B: BackingStorage, DB: KvDatabase, S: BuildHasher> Sieve<B, DB, S> {
    fn decrement_pending_write(&self, key: &B::Key) {
        let shard_index = self.shard_index(key);

        let read = self.shards.read_shard(shard_index);

        if let Retrieve::Hit(node) = read.get(key, false) {
            node.pending_writes
                .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
        }
    }
}

impl<B: BackingStorage, DB: KvDatabase, S: BuildHasher> Sieve<B, DB, S> {
    fn shard_index(&self, key: &B::Key) -> usize {
        self.shards.shard_index(self.hasher_builder.hash_one(key))
    }
}

impl<C: WideColumn, DB: KvDatabase, S: BuildHasher>
    Sieve<WideColumnAdaptor<C>, DB, S>
{
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
    pub fn get_normal<W: WideColumnValue<C> + Send + Sync + 'static>(
        &self,
        key: C::Key,
    ) -> Option<MappedRwLockReadGuard<'_, W>> {
        let combined_key = (key, W::discriminant());
        let shard_index = self.shard_index(&combined_key);

        loop {
            let read_lock = self.shards.read_shard(shard_index);

            if let Ok(guard) = RwLockReadGuard::try_map(read_lock, |x| match x
                .get(&combined_key, true)
            {
                Retrieve::Hit(entry) => Some(entry),
                Retrieve::Miss => None,
            }) {
                return MappedRwLockReadGuard::try_map(guard, |entry| {
                    entry.value.as_ref().map(|node| {
                        let any_ref: &dyn Any = node.as_ref();
                        any_ref.downcast_ref::<W>().unwrap()
                    })
                })
                .ok();
            }

            let mut write_lock = self.shards.write_shard(shard_index);

            if let Retrieve::Hit(_) = write_lock.get(&combined_key, false) {
                // another thread already filled the cache while we were
                // acquiring the write lock, retry read
                continue;
            }

            let db_value =
                self.backing_db.get_wide_column::<C, W>(&combined_key.0);

            write_lock.insert(
                combined_key.clone(),
                db_value.map(|x| Box::new(x) as Box<dyn Any + Send + Sync>),
                0,
            );
        }
    }
}

impl<C: WideColumn, DB: KvDatabase, S: write_behind::BuildHasher>
    Sieve<WideColumnAdaptor<C>, DB, S>
{
    /// Inserts or updates a value in the cache for the given key.
    ///
    /// This method allows direct insertion or update of a value in the cache,
    /// bypassing the backing database.
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
    /// # Eviction
    ///
    /// If the cache shard is at capacity, this insertion may trigger eviction
    /// of another entry according to the SIEVE algorithm.
    pub fn put<W: WideColumnValue<C> + Send + Sync + 'static>(
        self: &Arc<Self>,
        key: C::Key,
        value: Option<W>,
        write_buffer: &mut write_behind::WriteBuffer<DB, S>,
    ) {
        let combined_key = (key, W::discriminant());
        let shard_index = self.shard_index(&combined_key);

        let mut write_lock = self.shards.write_shard(shard_index);

        let updated = write_buffer.wide_column_writes.put(
            combined_key.0.clone(),
            value.clone(),
            self,
        );

        if let RetrieveMut::Hit(node) = write_lock.get_mut(&combined_key, true)
        {
            node.value =
                value.map(|v| Box::new(v) as Box<dyn Any + Send + Sync>);
            // Relaxed is fine since we've already acquired the write lock
            *node.visited.get_mut() = true;

            if updated {
                *node.pending_writes.get_mut() += 1;
            }
        } else {
            write_lock.insert(
                combined_key,
                value.map(|v| Box::new(v) as Box<dyn Any + Send + Sync>),
                usize::from(updated),
            );
        }
    }
}

/// A trait for determining the in-memory container type used for representing
/// the set data strcture in key-of-set columns.
pub trait KeyOfSetContainer: KeyOfSetColumn {
    /// The set container type used to store the collection of elements for a
    /// given key.
    type Container: FromIterator<Self::Element>
        + RemoveElementFromSet<Element = Self::Element>
        + Extend<Self::Element>
        + Send
        + Sync
        + 'static;
}

impl<C: KeyOfSetContainer, DB: KvDatabase, S: BuildHasher>
    Sieve<KeyOfSetAdaptor<C>, DB, S>
{
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
    pub fn get_set(
        &self,
        key: &C::Key,
    ) -> MappedRwLockReadGuard<'_, C::Container> {
        let shard_index = self.shard_index(key);
        loop {
            let read_lock = self.shards.read_shard(shard_index);

            if let Ok(guard) = RwLockReadGuard::try_map(read_lock, |x| match x
                .get(key, true)
            {
                Retrieve::Hit(entry) => entry.value.as_full(),
                Retrieve::Miss => None,
            }) {
                return guard;
            }

            let mut write_lock = self.shards.write_shard(shard_index);

            // the deltas that have to be applied to the full set
            let entry = write_lock.get_mut(key, false);

            let partial = if let RetrieveMut::Hit(en) = &entry {
                match &en.value {
                    KeyOfSetEntry::Full(_) => {
                        // another thread already filled the cache while we were
                        // acquiring the write lock, retry read
                        continue;
                    }

                    KeyOfSetEntry::Partial(hash_map) => Some(hash_map),
                }
            } else {
                None
            };

            let mut db_value: C::Container =
                self.backing_db.scan_members::<C>(key).collect();

            if let Some(partial_ops) = partial {
                for (element, operation) in partial_ops {
                    match operation {
                        Operation::Insert => {
                            db_value.extend(std::iter::once(element.clone()));
                        }
                        Operation::Delete => {
                            db_value.remove_element(element);
                        }
                    }
                }
            }

            match entry {
                RetrieveMut::Hit(node) => {
                    node.value = KeyOfSetEntry::Full(db_value);
                }
                RetrieveMut::Miss => {
                    write_lock.insert(
                        key.clone(),
                        KeyOfSetEntry::Full(db_value),
                        0,
                    );
                }
            }
        }
    }
}

impl<C: KeyOfSetContainer, DB: KvDatabase, S: write_behind::BuildHasher>
    Sieve<KeyOfSetAdaptor<C>, DB, S>
{
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
        self: &Arc<Self>,
        key: &C::Key,
        element: C::Element,
        write_buffer: &mut write_behind::WriteBuffer<DB, S>,
    ) {
        let shard_index = self.shard_index(key);

        let mut write_lock = self.shards.write_shard(shard_index);

        write_buffer.key_of_set_writes.put(
            key.clone(),
            element.clone(),
            Operation::Insert,
            self,
        );

        match write_lock.get_mut(key, true) {
            RetrieveMut::Hit(entry) => match &mut entry.value {
                KeyOfSetEntry::Full(container) => {
                    container.extend(std::iter::once(element));
                }
                KeyOfSetEntry::Partial(partial) => {
                    partial.insert(element, Operation::Insert);
                }
            },

            RetrieveMut::Miss => {
                let mut partial = HashMap::new();
                partial.insert(element, Operation::Insert);

                write_lock.insert(
                    key.clone(),
                    KeyOfSetEntry::Partial(partial),
                    0,
                );
            }
        }
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
    pub fn remove_set_element(
        self: &Arc<Self>,
        key: &C::Key,
        element: &C::Element,
        write_buffer: &mut write_behind::WriteBuffer<DB, S>,
    ) {
        let shard_index = self.shard_index(key);

        let mut write_lock = self.shards.write_shard(shard_index);

        write_buffer.key_of_set_writes.put(
            key.clone(),
            element.clone(),
            Operation::Delete,
            self,
        );

        match write_lock.get_mut(key, true) {
            RetrieveMut::Hit(entry) => match &mut entry.value {
                KeyOfSetEntry::Full(container) => {
                    container.remove_element(element);
                }
                KeyOfSetEntry::Partial(partial) => {
                    partial.insert(element.clone(), Operation::Delete);
                }
            },

            RetrieveMut::Miss => {
                let mut partial = HashMap::new();
                partial.insert(element.clone(), Operation::Delete);

                write_lock.insert(
                    key.clone(),
                    KeyOfSetEntry::Partial(partial),
                    0,
                );
            }
        }
    }
}

struct Node<B: BackingStorage> {
    key: B::Key,
    value: B::Value,
    pending_writes: AtomicUsize,
    visited: AtomicBool,
}

type Index = usize;

struct SieveShard<B: BackingStorage, S = BuildHasherDefault<fxhash::FxHasher>> {
    map: HashMap<B::Key, Index, S>,
    nodes: Vec<Option<Node<B>>>,
    hand: Index,

    active: usize,
}

impl<B: BackingStorage, S> SieveShard<B, S> {
    fn new(capacity: usize, hasher: S) -> Self {
        let mut nodes = Vec::with_capacity(capacity);
        nodes.resize_with(capacity, || None);

        Self { nodes, hand: 0, active: 0, map: HashMap::with_hasher(hasher) }
    }
}

enum Retrieve<'x, B: BackingStorage> {
    Hit(&'x Node<B>),
    Miss,
}

enum RetrieveMut<'x, B: BackingStorage> {
    Hit(&'x mut Node<B>),
    Miss,
}

impl<B: BackingStorage, S: BuildHasher> SieveShard<B, S> {
    fn insert(&mut self, key: B::Key, value: B::Value, pending_writes: usize) {
        // scan for an empty slot
        let mut scanned = 0;
        let max_scan = self.nodes.len() * 2;

        loop {
            if scanned > max_scan {
                // cache is full break from the loop to evict
                break;
            }

            let index = self.hand;
            self.hand = (self.hand + 1) % self.nodes.len();

            if let Some(node) = &mut self.nodes[index] {
                if node
                    .visited
                    .swap(false, std::sync::atomic::Ordering::Relaxed)
                    && node
                        .pending_writes
                        .load(std::sync::atomic::Ordering::SeqCst)
                        == 0
                {
                    // give a second chance
                    scanned += 1;
                    continue;
                }

                // evict this node
                self.map.remove(&node.key);
                *node = Node {
                    key: key.clone(),
                    value,
                    pending_writes: AtomicUsize::new(pending_writes),
                    visited: AtomicBool::new(true),
                };
                self.map.insert(key, index);
                return;
            }

            // empty slot found
            self.nodes[index] = Some(Node {
                key: key.clone(),
                value,
                pending_writes: AtomicUsize::new(pending_writes),
                visited: AtomicBool::new(true),
            });
            self.map.insert(key, index);
            self.active += 1;
            return;
        }

        // unconditionally increases the size by appending to the end
        self.nodes.push(Some(Node {
            key: key.clone(),
            value,
            pending_writes: AtomicUsize::new(pending_writes),
            visited: AtomicBool::new(true),
        }));
        self.map.insert(key, self.nodes.len() - 1);
        self.active += 1;
        self.hand = 0; // reset hand to start scanning from beginning next time
    }

    fn get(&self, key: &B::Key, visit: bool) -> Retrieve<'_, B> {
        if let Some(&index) = self.map.get(key) {
            let node = self.nodes[index].as_ref().unwrap();

            if visit {
                node.visited.store(true, std::sync::atomic::Ordering::Relaxed);
            }

            return Retrieve::Hit(node);
        }

        Retrieve::Miss
    }

    fn get_mut(&mut self, key: &B::Key, visit: bool) -> RetrieveMut<'_, B> {
        if let Some(&index) = self.map.get(key) {
            let node = self.nodes[index].as_mut().unwrap();

            if visit {
                node.visited.store(true, std::sync::atomic::Ordering::Relaxed);
            }

            return RetrieveMut::Hit(node);
        }

        RetrieveMut::Miss
    }
}
