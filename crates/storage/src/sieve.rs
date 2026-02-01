//! A sharded, concurrent cache implementation using the SIEVE eviction
//! algorithm with write-back support.
//!
//! This module provides [`Sieve`], a high-performance cache that integrates
//! with key-value databases to provide transparent caching with lazy loading
//! and optional asynchronous write buffering.
//!
//! # SIEVE Algorithm
//!
//! SIEVE (Simpler Is Efficient Eviction) is a modern cache eviction algorithm
//! that achieves excellent hit rates comparable to LRU with significantly
//! lower overhead. Unlike LRU, SIEVE:
//!
//! - Uses a single "visited" bit per entry instead of maintaining access order
//! - Employs a hand pointer that sweeps through entries circularly
//! - Gives unvisited entries a "second chance" before eviction
//! - Naturally resists scan pollution from one-time accesses
//!
//! ## Performance Benefits
//!
//! - **Lower CPU overhead**: No list reordering on every access (O(1) marking)
//! - **Better memory locality**: Circular sweep pattern vs. linked list
//!   traversal
//! - **Scan resistance**: One-time accesses don't displace useful cached data
//! - **Comparable hit rates**: Matches or exceeds LRU in most workloads
//!
//! # Sharding
//!
//! The cache divides entries across multiple shards to minimize lock
//! contention. Each shard:
//! - Maintains its own SIEVE state (hand pointer, visited bits)
//! - Has independent capacity and eviction policy
//! - Can be accessed in parallel without blocking other shards
//!
//! More shards reduce contention but increase memory overhead. Recommended:
//! 16-32 shards for most multi-threaded workloads.
//!
//! # Write-Back Caching
//!
//! The module supports optional write-back caching via [`BackgroundWriter`],
//! which:
//! - Buffers writes in memory using [`WriteBuffer`]
//! - Asynchronously flushes to database in batches
//! - Maintains write ordering for consistency
//! - Reduces database write amplification
//!
//! # Usage Patterns
//!
//! ## Read-Through Cache
//!
//! ```ignore
//! let cache = WideColumnSieve::new(capacity, shards, db, hasher);
//!
//! // Automatically loads from DB on miss
//! let value = cache.get_normal::<ValueType>(key);
//! ```
//!
//! ## Write-Back Cache
//!
//! ```ignore
//! let writer = BackgroundWriter::new(4, db.clone());
//! let mut buffer = writer.new_write_buffer();
//!
//! // Write to cache and buffer
//! cache.put::<ValueType>(key, Some(value), &mut buffer);
//!
//! // Async flush to database
//! writer.submit_write_buffer(buffer);
//! ```

use core::fmt;
use std::{
    any::{Any, TypeId},
    borrow::Borrow,
    collections::{HashMap, hash_map},
    fmt::Debug,
    hash::{BuildHasher, BuildHasherDefault, Hash},
    marker::PhantomData,
    option::Option,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use dashmap::{DashMap, DashSet};
use enum_as_inner::EnumAsInner;
use parking_lot::RwLockReadGuard;

use crate::{
    kv_database::{KeyOfSetColumn, KvDatabase, WideColumn, WideColumnValue},
    sharded::Sharded,
    sieve::single_flight::SingleFlight,
};

mod single_flight;
mod write_behind;

pub use write_behind::{BackgroundWriter, WriteBuffer};

/// An internal trait abstracting the key and value types used for backing
/// storage in the SIEVE cache.
pub trait BackingStorage {
    /// The type of keys used in the backing storage.
    type Key: Debug + Eq + Hash + Clone;

    /// The type of values stored in the backing storage.
    type Value;

    /// The type used for single-flight deduplication of concurrent fetches.
    type SingleFlightKey: Eq + Hash;
}

/// An adaptor for using wide columns as backing storage in the SIEVE cache.
///
/// See [`WideColumnSieve`] for details.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WideColumnAdaptor<C>(pub PhantomData<C>);

impl<C: WideColumn> BackingStorage for WideColumnAdaptor<C> {
    type Key = (C::Key, C::Discriminant);
    type Value = Option<Box<dyn Any + Send + Sync>>;

    // a tuple of key and stable type id of the wide column ty
    type SingleFlightKey = (C::Key, TypeId);
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
    Partial(DashMap<C::Element, Operation>),
}

impl<C: KeyOfSetContainer> BackingStorage for KeyOfSetAdaptor<C> {
    type Key = C::Key;
    type Value = KeyOfSetEntry<C>;

    // simply the key type
    type SingleFlightKey = C::Key;
}

/// A sharded sieve cache operating on wide columns data schema.
pub type WideColumnSieve<C, DB, S = BuildHasherDefault<fxhash::FxHasher>> =
    Sieve<WideColumnAdaptor<C>, DB, S>;

/// A sharded sieve cache operating on key-of-set columns data schema.
pub type KeyOfSetSieve<C, DB, S = BuildHasherDefault<fxhash::FxHasher>> =
    Sieve<KeyOfSetAdaptor<C>, DB, S>;

/// A trait for removing elements from set-like collections.
///
/// This trait provides a uniform interface for element removal across different
/// set implementations used in the SIEVE cache. It enables the cache to work
/// with various set types (`HashSet`, `DashSet`, custom implementations) while
/// maintaining consistent removal semantics.
///
/// # Generic Element Lookup
///
/// The `remove_element` method accepts any borrowed form `Q` of the element
/// type through the `Borrow` trait. This allows efficient removal without
/// requiring owned values, matching the flexibility of standard library
/// collections.
///
/// # Implementations
///
/// The trait is implemented for:
/// - `HashSet<T, S>`: Standard single-threaded set
/// - `DashSet<T, S>`: Concurrent sharded set from `dashmap`
///
/// # Example
///
/// ```ignore
/// let mut set = HashSet::new();
/// set.insert("key".to_string());
///
/// // Remove using a borrowed form
/// let removed = set.remove_element("key");
/// assert!(removed);
/// ```
pub trait ConcurrentSet {
    /// The type of elements stored in the set.
    type Element;

    /// Inserts an element into the set.
    fn insert_element(&self, element: Self::Element);

    /// Removes an element from the set.
    ///
    /// # Returns
    ///
    /// - `true` if the element was present and removed
    /// - `false` if the element was not found in the set
    fn remove_element<Q: Hash + Eq + ?Sized>(&self, element: &Q) -> bool
    where
        Self::Element: Borrow<Q>;
}

impl<T: Eq + Hash, S: BuildHasher + Clone> ConcurrentSet for DashSet<T, S> {
    type Element = T;

    fn insert_element(&self, element: Self::Element) { self.insert(element); }

    fn remove_element<Q: Hash + Eq + ?Sized>(&self, element: &Q) -> bool
    where
        Self::Element: Borrow<Q>,
    {
        self.remove(element).is_some()
    }
}

struct StorageEntry<V: ?Sized> {
    visited: AtomicBool,
    pending_writes: AtomicUsize,
    value: V,
}

struct Policy<K> {
    keys: Vec<Option<K>>,
    hand: usize,
    capacity: usize,
}

impl<K> Policy<K> {
    /// Inserts a key into the policy structure. Returns an evicted key if
    /// necessary.
    fn insert<V, S: BuildHasher>(
        &mut self,
        key: K,
        sharded_storages: &Sharded<HashMap<K, StorageEntry<V>, S>>,
        shard_index: usize,
    ) where
        K: Debug + Eq + Hash,
    {
        const MAX_HAND_SWEEPS: usize = 64;

        // fast path: still have capacity
        if self.keys.len() < self.capacity {
            self.keys.push(Some(key));
            return;
        }

        let mut read_lock = sharded_storages.read_shard(shard_index);

        // normal path: evict using SIEVE
        for _ in 0..MAX_HAND_SWEEPS {
            let index = self.hand % self.capacity;
            self.hand = (self.hand + 1) % self.capacity;

            if let Some(existing_key) = &self.keys[index] {
                let (visited, pending_writes) = {
                    let entry = read_lock
                        .get(existing_key)
                        .expect("should exist in storage");

                    (
                        entry.visited.swap(false, Ordering::SeqCst),
                        entry.pending_writes.load(Ordering::SeqCst),
                    )
                };

                // if is visited or has pending writes, give a second chance
                if visited || pending_writes != 0 {
                    continue;
                }

                let Some(new_read_lock) = Self::try_evict(
                    read_lock,
                    existing_key,
                    sharded_storages,
                    shard_index,
                ) else {
                    // successfully evicted
                    self.keys[index].replace(key);

                    return;
                };

                read_lock = new_read_lock;
            } else {
                // has no key, can use this slot
                self.keys[index] = Some(key);

                return;
            }
        }

        // reached max sweeps, evict at the nearest available slot that has no
        // pending writes
        {
            let max_scan = self.keys.len();
            let mut scanned = 0;

            // scan until we find a slot to evict
            while scanned < max_scan {
                let index = self.hand % self.capacity;
                self.hand = (self.hand + 1) % self.capacity;
                scanned += 1;

                if let Some(existing_key) = &self.keys[index] {
                    let pending_writes = {
                        let read_lock =
                            sharded_storages.read_shard(shard_index);

                        let entry = read_lock
                            .get(existing_key)
                            .expect("should exist in storage");

                        entry.pending_writes.load(Ordering::SeqCst)
                    };

                    if pending_writes != 0 {
                        continue;
                    }

                    let Some(new_read_lock) = Self::try_evict(
                        read_lock,
                        existing_key,
                        sharded_storages,
                        shard_index,
                    ) else {
                        // successfully evicted
                        self.keys[index].replace(key);

                        return;
                    };

                    read_lock = new_read_lock;
                } else {
                    // has no key, can use this slot
                    self.keys[index] = Some(key);

                    return;
                }
            }
        }

        // last resort: no evictable entry found, expand capacity
        self.keys.push(Some(key));
        self.capacity += 1;
        self.hand = 0;
    }

    fn try_evict<'v, V, S: BuildHasher>(
        read_guard: RwLockReadGuard<'v, HashMap<K, StorageEntry<V>, S>>,
        key: &K,
        sharded_storages: &'v Sharded<HashMap<K, StorageEntry<V>, S>>,
        shard_index: usize,
    ) -> Option<RwLockReadGuard<'v, HashMap<K, StorageEntry<V>, S>>>
    where
        K: Debug + Eq + Hash,
    {
        drop(read_guard);

        let mut write_lock = sharded_storages.write_shard(shard_index);

        let Some(entry) = write_lock.get(key) else {
            panic!("key {key:?} should exist in storage");
        };

        // double check no pending writes
        if entry.pending_writes.load(Ordering::SeqCst) != 0 {
            // while we're about to obtain write lock and evict, pending writes
            // appeared; cannot evict
            return Some({
                drop(write_lock);

                sharded_storages.read_shard(shard_index)
            });
        }

        // now we can safely evict without worrying about pending writes
        let old_value = write_lock.remove(key);

        // we drop the shard lock first since dropping the entry might take long
        // time.
        drop(write_lock);

        assert!(old_value.is_some(), "key {key:?} should exist in storage");

        None
    }
}

/// A sharded cache using the SIEVE eviction algorithm with lazy loading.
///
/// `Sieve` provides a high-performance, thread-safe cache that transparently
/// loads values from a backing database on cache misses. It uses the SIEVE
/// (Simpler Is Efficient Eviction) algorithm for excellent hit rates with
/// minimal overhead.
///
/// # Type Parameters
///
/// * `B` - The backing storage adaptor ([`WideColumnAdaptor`] or
///   [`KeyOfSetAdaptor`]) that defines the data schema
/// * `DB` - The backing database implementation (must implement [`KvDatabase`])
/// * `S` - The hash builder for key hashing (defaults to `FxHasher`)
///
/// # Architecture
///
/// The cache is divided into multiple independent shards:
/// - Each shard has its own lock, capacity, and SIEVE state
/// - Keys are distributed across shards using consistent hashing
/// - Parallel operations on different shards don't block each other
/// - More shards = less contention but higher memory overhead
///
/// # Cache Semantics
///
/// ## Read Operations
///
/// - **Cache hit**: Returns value immediately from cache
/// - **Cache miss**: Fetches from backing database, caches, then returns
/// - **Concurrent misses**: Only one fetch per key; other threads wait
///
/// ## Write Operations
///
/// When using write-back mode:
/// - Updates cache immediately
/// - Buffers writes in [`WriteBuffer`]
/// - Background writer flushes to database asynchronously
/// - Tracks pending writes to prevent premature eviction
///
/// # Deadlock Warning
///
/// **Do not** call cache methods while holding a read guard from the same
/// cache. Although reads return read guards, internal implementation may
/// acquire write locks on cache misses, leading to deadlock.
///
/// # Example
///
/// ```ignore
/// use qbice_storage::sieve::WideColumnSieve;
///
/// let cache = WideColumnSieve::<MyColumn, _, _>::new(
///     10_000,              // total capacity
///     16,                  // shard count (power of 2)
///     Arc::new(database),  // backing DB
///     Default::default(),  // hasher
/// );
///
/// // Read-through: fetches from DB on miss
/// if let Some(value) = cache.get_normal::<MyValue>(key) {
///     // Use value
/// }
/// ```
///
/// # Isolations of Write Operations
///
/// The Sieve cache doesn't guarantee isolation for write operations buffered
/// in a `WriteBuffer`. If multiple write buffers are used concurrently,
/// lost updates may occur. The user must ensure that write buffers are not
/// used in overlapping contexts to maintain data consistency.
#[allow(clippy::type_complexity)] // shards field
pub struct Sieve<
    B: BackingStorage,
    DB,
    S = BuildHasherDefault<fxhash::FxHasher>,
> {
    storages: Sharded<HashMap<B::Key, StorageEntry<B::Value>, S>>,
    policies: Sharded<Policy<B::Key>>,

    hasher_builder: S,
    single_flight: SingleFlight<B::SingleFlightKey>,

    backing_db: Arc<DB>,
}

impl<B: BackingStorage, DB, S> Debug for Sieve<B, DB, S>
where
    B::Key: Debug,
    B::Value: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut map = f.debug_map();

        for shard in self.storages.iter_read_shards() {
            for (key, value) in shard.iter() {
                map.entry(key, &value.value);
            }
        }

        map.finish_non_exhaustive()
    }
}

impl<B: BackingStorage, DB, S: Clone> Sieve<B, DB, S> {
    /// Creates a new [`Sieve`] cache with the specified configuration.
    ///
    /// # Parameters
    ///
    /// * `total_capacity` - The total number of entries the cache can hold
    ///   across all shards. The actual capacity is distributed evenly among
    ///   shards (using ceiling division), so the real capacity may be slightly
    ///   higher than specified.
    ///
    /// * `shard_amount` - The number of shards to divide the cache into. **Must
    ///   be a power of two** for efficient shard selection via bit masking.
    ///   More shards reduce lock contention but increase memory overhead.
    ///   Recommended values:
    ///   - Single-threaded: 1-4 shards
    ///   - Multi-threaded: 16-32 shards
    ///   - High contention: 64+ shards
    ///
    /// * `backing_db` - The database to fetch values from on cache misses. Must
    ///   be wrapped in `Arc` for shared ownership across cache operations.
    ///
    /// * `hasher_builder` - The hash builder used to hash keys for shard
    ///   selection and internal hash maps. Use `FxHasher` for speed or
    ///   cryptographic hashers for security-sensitive applications.
    ///
    /// # Panics
    ///
    /// Panics if `shard_amount` is not a power of two.
    ///
    /// # Returns
    ///
    /// A new `Sieve` cache instance ready to use.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use std::sync::Arc;
    /// use fxhash::FxBuildHasher;
    ///
    /// let cache = Sieve::new(
    ///     10_000,                    // 10k entries total
    ///     16,                        // 16 shards (~625 entries each)
    ///     Arc::new(my_database),
    ///     FxBuildHasher::default(),
    /// );
    /// ```
    pub fn new(
        total_capacity: usize,
        shard_amount: usize,
        backing_db: Arc<DB>,
        hasher_builder: S,
    ) -> Self {
        let per_shard = total_capacity.div_ceil(shard_amount);

        Self {
            storages: Sharded::new(shard_amount, |_| {
                HashMap::with_hasher(hasher_builder.clone())
            }),
            policies: Sharded::new(shard_amount, |_| Policy {
                keys: Vec::new(),
                hand: 0,
                capacity: per_shard,
            }),
            single_flight: SingleFlight::new(shard_amount),
            hasher_builder,
            backing_db,
        }
    }
}

impl<B: BackingStorage, DB: KvDatabase, S: BuildHasher> Sieve<B, DB, S> {
    fn decrement_pending_write(&self, key: &B::Key) {
        let shard_index = self.shard_index(key);

        let read = self.storages.read_shard(shard_index);

        let node = read
            .get(key)
            .unwrap_or_else(|| panic!("key {key:?} should exist in storage"));

        node.pending_writes.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }
}

impl<B: BackingStorage, DB: KvDatabase, S: BuildHasher> Sieve<B, DB, S> {
    fn shard_index(&self, key: &B::Key) -> usize {
        self.storages.shard_index(self.hasher_builder.hash_one(key))
    }
}

impl<C: WideColumn, DB: KvDatabase, S: BuildHasher>
    Sieve<WideColumnAdaptor<C>, DB, S>
{
    /// Retrieves a value from the cache for wide column storage, fetching from
    /// the database if not present.
    ///
    /// This method implements read-through caching: it first checks the cache,
    /// and on a miss, fetches from the backing database, caches the result, and
    /// returns it. The returned guard provides read access to the cached value.
    ///
    /// # Type Parameter
    ///
    /// * `W` - The specific value type implementing [`WideColumnValue<C>`].
    ///   This determines which discriminant to use when querying the database.
    ///
    /// # Parameters
    ///
    /// * `key` - The key to look up. Combined with `W::discriminant()` to form
    ///   the complete lookup key.
    ///
    /// # Returns
    ///
    /// * `Some(ReadGuard<W>)` - If the value exists in cache or database. The
    ///   guard provides read-only access and prevents eviction while held.
    /// * `None` - If the key does not exist in the backing database.
    ///
    /// # Concurrency
    ///
    /// When multiple threads request the same key simultaneously:
    /// - Only one thread fetches from the database
    /// - Other threads wait for the fetch to complete
    /// - All threads receive the same cached result
    ///
    /// This prevents "cache stampede" where many threads redundantly fetch the
    /// same data.
    ///
    /// # Cancellation Safety
    ///
    /// This function is cancellation-safe. If dropped during a database fetch:
    /// - Internal state remains consistent
    /// - Other waiting threads will retry the fetch
    /// - No partial updates or corruption occur
    ///
    /// # Deadlock Warning
    ///
    /// **Never call this method while holding another read guard from this
    /// cache.** Although it returns a read lock, the implementation may acquire
    /// write locks internally when inserting new entries on cache misses.
    ///
    /// ```ignore
    /// // DEADLOCK - DO NOT DO THIS:
    /// let guard1 = cache.get_normal::<T>(key1)?;
    /// let guard2 = cache.get_normal::<T>(key2)?; // May deadlock!
    ///
    /// // CORRECT - Drop guards before next call:
    /// let value1 = cache.get_normal::<T>(key1)?.clone();
    /// drop(guard1);  // Explicitly drop or let it go out of scope
    /// let value2 = cache.get_normal::<T>(key2)?;
    /// ```
    ///
    /// # Example
    ///
    /// ```ignore
    /// if let Some(user_name) = cache.get_normal::<UserName>(user_id) {
    ///     println!("User: {}", user_name.0);
    /// } else {
    ///     println!("User not found");
    /// }
    /// ```
    #[allow(clippy::cast_possible_truncation)] // from u64 to usize
    pub async fn get_normal<W: WideColumnValue<C> + Send + Sync + 'static>(
        &self,
        key: C::Key,
    ) -> Option<W> {
        let combined_key = (key, W::discriminant());
        let shard_index = self.shard_index(&combined_key);

        loop {
            // first try read lock
            {
                let read_lock = self.storages.read_shard(shard_index);

                if let Some(value) = read_lock.get(&combined_key) {
                    value
                        .visited
                        .store(true, std::sync::atomic::Ordering::SeqCst);

                    return value.value.as_ref().map(|x| {
                        x.as_ref()
                            .downcast_ref::<W>()
                            .expect("stored type should match requested type")
                            .clone()
                    });
                }
            }

            self.single_flight
                .work_or_wait(
                    (combined_key.0.clone(), std::any::TypeId::of::<W>()),
                    || {
                        let db_value = self
                            .backing_db
                            .get_wide_column::<C, W>(&combined_key.0);

                        // insert into cache
                        {
                            let mut storage_lock =
                                self.storages.write_shard(shard_index);

                            let hash_map::Entry::Vacant(entry) =
                                storage_lock.entry(combined_key.clone())
                            else {
                                // another thread already filled the cache while
                                // we were acquiring the write lock, retry read
                                return;
                            };

                            entry.insert(StorageEntry {
                                value: db_value.map(|x| {
                                    Box::new(x) as Box<dyn Any + Send + Sync>
                                }),
                                visited: AtomicBool::new(true),
                                pending_writes: AtomicUsize::new(0),
                            });
                        }

                        // update policy
                        {
                            let mut policy_lock =
                                self.policies.write_shard(shard_index);

                            policy_lock.insert(
                                combined_key.clone(),
                                &self.storages,
                                shard_index,
                            );
                        };
                    },
                )
                .await;
        }
    }
}

impl<C: WideColumn, DB: KvDatabase, S: write_behind::BuildHasher>
    Sieve<WideColumnAdaptor<C>, DB, S>
{
    /// Inserts or updates a value in the cache with write-back buffering.
    ///
    /// This method updates the in-memory cache immediately and buffers the
    /// write operation for asynchronous flushing to the database via
    /// [`BackgroundWriter`]. The cache entry is marked with a pending write
    /// count to prevent eviction before the write is persisted.
    ///
    /// # Type Parameter
    ///
    /// * `W` - The specific value type implementing [`WideColumnValue<C>`].
    ///   Determines the discriminant for the database write.
    ///
    /// # Parameters
    ///
    /// * `key` - The key to insert or update.
    /// * `value` - The new value to cache. `None` represents deletion (caches
    ///   the absence of the value).
    /// * `write_buffer` - Buffer to accumulate this write operation. Must be
    ///   eventually submitted to a [`BackgroundWriter`] for persistence.
    ///
    /// # Durability
    ///
    /// This method only updates the in-memory cache. The write is **not
    /// durable** until:
    /// 1. The `write_buffer` is submitted to [`BackgroundWriter`]
    /// 2. The background writer flushes and commits the batch
    ///
    /// **Data Loss Risk**: If the process crashes before the write is flushed,
    /// the update will be lost.
    ///
    /// # Eviction Protection
    ///
    /// The cache tracks pending writes and prevents eviction of entries with
    /// unflushed writes. The pending write count is decremented after the
    /// background writer commits the batch to the database.
    ///
    /// # Write Ordering
    ///
    /// Multiple `put` calls to the same write buffer are applied in the order
    /// they appear in the buffer. Write buffers are committed in epoch order
    /// to ensure consistency.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let cache = Arc::new(WideColumnSieve::new(...));
    /// let writer = BackgroundWriter::new(4, db.clone());
    /// let mut buffer = writer.new_write_buffer();
    ///
    /// // Update cache
    /// cache.put::<UserName>(user_id, Some(name), &mut buffer);
    /// cache.put::<UserEmail>(user_id, Some(email), &mut buffer);
    ///
    /// // Submit for async persistence
    /// writer.submit_write_buffer(buffer);
    /// ```
    pub fn put<W: WideColumnValue<C> + Send + Sync + 'static>(
        self: &Arc<Self>,
        key: C::Key,
        value: Option<W>,
        write_buffer: &mut write_behind::WriteBuffer<DB, S>,
    ) {
        let combined_key = (key, W::discriminant());
        let shard_index = self.shard_index(&combined_key);

        let updated = write_buffer.wide_column_writes.put(
            combined_key.0.clone(),
            value.clone(),
            self,
        );
        let boxed = value.map(|v| Box::new(v) as Box<dyn Any + Send + Sync>);

        let mut write_lock = self.storages.write_shard(shard_index);

        if let Some(entry) = write_lock.get_mut(&combined_key) {
            let old_node = std::mem::replace(&mut entry.value, boxed);

            if updated {
                *entry.pending_writes.get_mut() += 1;
            }

            drop(write_lock);

            // NOTE: sometimes the value maybe very large, so dropping them
            // outside the lock scope to reduce the time we hold the lock
            drop(old_node);
        } else {
            let old_value =
                write_lock.insert(combined_key.clone(), StorageEntry {
                    value: boxed,
                    visited: AtomicBool::new(true),
                    pending_writes: AtomicUsize::new(usize::from(updated)),
                });

            drop(write_lock);
            drop(old_value);

            let mut policy_lock = self.policies.write_shard(shard_index);

            // update policy
            policy_lock.insert(combined_key, &self.storages, shard_index);
        }
    }
}

/// A trait for specifying the in-memory container type for key-of-set caching.
///
/// This trait extends [`KeyOfSetColumn`] to specify which collection type
/// should be used to represent sets in memory when caching key-of-set data.
/// Different container types offer different trade-offs between performance,
/// memory usage, and concurrency.
///
/// # Container Requirements
///
/// The container type must support:
/// - `FromIterator<Element>`: Construction from database query results
/// - `RemoveElementFromSet`: Efficient element removal
/// - `Extend<Element>`: Adding new elements
/// - `Send + Sync + 'static`: Thread-safe caching
///
/// # Common Implementations
///
/// - `HashSet<T>`: Standard single-threaded set with fast lookups
/// - `DashSet<T>`: Concurrent sharded set from `dashmap` crate
/// - `BTreeSet<T>`: Ordered set with range query support
///
/// # Example
///
/// ```ignore
/// use std::collections::HashSet;
///
/// impl KeyOfSetContainer for UserTagsColumn {
///     type Container = HashSet<String>;
/// }
/// ```
pub trait KeyOfSetContainer: KeyOfSetColumn {
    /// The in-memory container type used to represent the set.
    type Container: FromIterator<Self::Element>
        + ConcurrentSet<Element = Self::Element>
        + Clone
        + Send
        + Sync
        + 'static;
}

impl<C: KeyOfSetContainer, DB: KvDatabase, S: BuildHasher>
    Sieve<KeyOfSetAdaptor<C>, DB, S>
{
    /// Retrieves a set from the cache for key-of-set storage, fetching from the
    /// database if not present.
    ///
    /// This method implements read-through caching for set-based storage.
    /// Unlike [`get_normal`](Self::get_normal), this always returns a valid
    /// container (possibly empty) since key-of-set storage has a natural
    /// default state.
    ///
    /// # Parameters
    ///
    /// * `key` - The key identifying which set to retrieve.
    ///
    /// # Returns
    ///
    /// A read guard providing access to the cached set container. The container
    /// contains all members of the set, either loaded from the database or
    /// previously cached.
    ///
    /// # Caching Strategies
    ///
    /// The cache uses two internal representations:
    ///
    /// 1. **Full**: Complete set loaded from database
    ///    - Used after initial database fetch
    ///    - All members are materialized in memory
    ///    - Efficient for reads
    ///
    /// 2. **Partial**: Delta operations only
    ///    - Used when only insert/delete operations have been performed
    ///    - Database fetch is deferred until this method is called
    ///    - Operations are applied to database results before caching
    ///
    /// # Concurrency
    ///
    /// Multiple concurrent calls with the same key coordinate to avoid
    /// duplicate database fetches. Only one thread performs the fetch while
    /// others wait.
    ///
    /// # Cancellation Safety
    ///
    /// This function is cancellation-safe. If dropped during a database fetch,
    /// internal state remains consistent and other threads will retry.
    ///
    /// # Deadlock Warning
    ///
    /// **Never call this method while holding another read guard from this
    /// cache.** The implementation may acquire write locks internally when
    /// materializing partial entries.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let tags = cache.get_set(&user_id);
    ///
    /// for tag in tags.iter() {
    ///     println!("Tag: {}", tag);
    /// }
    ///
    /// println!("Total tags: {}", tags.len());
    /// ```
    #[allow(clippy::unused_async)]
    pub async fn get_set(&self, key: &C::Key) -> C::Container {
        let shard_index = self.shard_index(key);
        loop {
            // first try read lock
            {
                let read_lock = self.storages.read_shard(shard_index);

                if let Some(guard) =
                    read_lock.get(key).and_then(|x| x.value.as_full())
                {
                    return guard.clone();
                }
            }

            self.single_flight
                .work_or_wait(key.clone(), || {
                    let db_value = self
                        .backing_db
                        .scan_members::<C>(key)
                        .collect::<C::Container>();

                    {
                        let mut write_lock =
                            self.storages.write_shard(shard_index);

                        // the deltas that have to be applied to the full set
                        let entry = write_lock.entry(key.clone());

                        let partial = if let hash_map::Entry::Occupied(en) =
                            &entry
                        {
                            match &en.get().value {
                                KeyOfSetEntry::Full(_) => {
                                    // another thread already filled the cache
                                    // while
                                    // we were acquiring the write lock, retry
                                    // read
                                    return;
                                }

                                KeyOfSetEntry::Partial(hash_map) => {
                                    Some(hash_map)
                                }
                            }
                        } else {
                            None
                        };

                        if let Some(partial_ops) = partial {
                            for entry in partial_ops {
                                match entry.value() {
                                    Operation::Insert => {
                                        db_value.insert_element(
                                            entry.key().clone(),
                                        );
                                    }
                                    Operation::Delete => {
                                        db_value.remove_element(entry.key());
                                    }
                                }
                            }
                        }

                        match entry {
                            hash_map::Entry::Occupied(mut node) => {
                                let old_value = std::mem::replace(
                                    &mut node.get_mut().value,
                                    KeyOfSetEntry::Full(db_value),
                                );

                                drop(write_lock);

                                // NOTE: sometimes the value maybe very large,
                                // so dropping them outside the lock scope to
                                // reduce the time we hold the lock
                                drop(old_value);
                            }

                            hash_map::Entry::Vacant(entry) => {
                                entry.insert(StorageEntry {
                                    value: KeyOfSetEntry::Full(db_value),
                                    visited: AtomicBool::new(true),
                                    pending_writes: AtomicUsize::new(0),
                                });

                                drop(write_lock);

                                let mut policy_lock =
                                    self.policies.write_shard(shard_index);

                                policy_lock.insert(
                                    key.clone(),
                                    &self.storages,
                                    shard_index,
                                );
                            }
                        }
                    }
                })
                .await;
        }
    }
}

impl<C: KeyOfSetContainer, DB: KvDatabase, S: write_behind::BuildHasher>
    Sieve<KeyOfSetAdaptor<C>, DB, S>
{
    /// Inserts an element into a set with write-back buffering.
    ///
    /// This method updates the cached set immediately (creating a partial entry
    /// if needed) and buffers the insert operation for asynchronous
    /// persistence. The element is added to the set in cache, but the write
    /// to the database is deferred until the buffer is flushed.
    ///
    /// # Parameters
    ///
    /// * `key` - The key identifying which set to modify.
    /// * `element` - The element to insert into the set.
    /// * `write_buffer` - Buffer to accumulate this write operation.
    ///
    /// # Behavior
    ///
    /// - **If cache has full set**: Element is added to the materialized set
    /// - **If cache has partial operations**: Insert operation is recorded
    /// - **If cache miss**: A partial entry is created with this insert
    ///
    /// Partial entries defer database fetching until [`get_set`](Self::get_set)
    /// is called, at which point all recorded operations are applied to the
    /// database results.
    ///
    /// # Durability
    ///
    /// The insert is **not durable** until the write buffer is submitted to
    /// [`BackgroundWriter`] and successfully flushed to the database.
    ///
    /// # Duplicate Elements
    ///
    /// If the element already exists in the set (either in cache or database),
    /// this is a no-op from the perspective of the final state, though it still
    /// generates a database write operation.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let cache = Arc::new(KeyOfSetSieve::new(...));
    /// let writer = BackgroundWriter::new(4, db.clone());
    /// let mut buffer = writer.new_write_buffer();
    ///
    /// // Add tags
    /// cache.insert_set_element(&user_id, "rust".to_string(), &mut buffer);
    /// cache.insert_set_element(&user_id, "async".to_string(), &mut buffer);
    ///
    /// // Submit for persistence
    /// writer.submit_write_buffer(buffer);
    /// ```
    pub fn insert_set_element(
        self: &Arc<Self>,
        key: &C::Key,
        element: C::Element,
        write_buffer: &mut write_behind::WriteBuffer<DB, S>,
    ) {
        let shard_index = self.shard_index(key);

        let updated = write_buffer.key_of_set_writes.put(
            key.clone(),
            element.clone(),
            Operation::Insert,
            self,
        );

        {
            let read_shard = self.storages.read_shard(shard_index);

            if let Some(entry) = read_shard.get(key) {
                match &entry.value {
                    KeyOfSetEntry::Full(container) => {
                        container.insert_element(element);
                    }
                    KeyOfSetEntry::Partial(partial) => {
                        partial.insert(element, Operation::Insert);
                    }
                }

                if updated {
                    entry
                        .pending_writes
                        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                }

                return;
            }
        }

        // has to acquire write lock

        let mut write_lock = self.storages.write_shard(shard_index);

        match write_lock.entry(key.clone()) {
            hash_map::Entry::Occupied(mut entry) => {
                match &mut entry.get_mut().value {
                    KeyOfSetEntry::Full(container) => {
                        container.insert_element(element);
                    }
                    KeyOfSetEntry::Partial(partial) => {
                        partial.insert(element, Operation::Insert);
                    }
                }

                if updated {
                    *entry.get_mut().pending_writes.get_mut() += 1;
                }
            }

            hash_map::Entry::Vacant(entry) => {
                let partial = DashMap::new();
                partial.insert(element, Operation::Insert);

                entry.insert(StorageEntry {
                    value: KeyOfSetEntry::Partial(partial),
                    visited: AtomicBool::new(false),
                    pending_writes: AtomicUsize::new(usize::from(updated)),
                });

                drop(write_lock);

                let mut policy_lock = self.policies.write_shard(shard_index);

                policy_lock.insert(key.clone(), &self.storages, shard_index);

                drop(policy_lock);
            }
        }
    }

    /// Removes an element from a set with write-back buffering.
    ///
    /// This method updates the cached set immediately (creating a partial entry
    /// if needed) and buffers the delete operation for asynchronous
    /// persistence. The element is removed from the set in cache, but the
    /// write to the database is deferred until the buffer is flushed.
    ///
    /// # Parameters
    ///
    /// * `key` - The key identifying which set to modify.
    /// * `element` - The element to remove from the set.
    /// * `write_buffer` - Buffer to accumulate this write operation.
    ///
    /// # Behavior
    ///
    /// - **If cache has full set**: Element is removed from the materialized
    ///   set
    /// - **If cache has partial operations**: Delete operation is recorded
    /// - **If cache miss**: A partial entry is created with this delete
    ///
    /// Partial entries defer database fetching until [`get_set`](Self::get_set)
    /// is called, at which point all recorded operations are applied to the
    /// database results.
    ///
    /// # Durability
    ///
    /// The removal is **not durable** until the write buffer is submitted to
    /// [`BackgroundWriter`] and successfully flushed to the database.
    ///
    /// # Non-Existent Elements
    ///
    /// If the element doesn't exist in the set (either in cache or database),
    /// this is a no-op from the perspective of the final state, though it still
    /// generates a database write operation.
    ///
    /// # Operation Ordering
    ///
    /// Operations within a write buffer maintain order. For example:
    /// ```ignore
    /// cache.insert_set_element(&key, "tag1", &mut buffer);
    /// cache.remove_set_element(&key, &"tag1", &mut buffer);
    /// // Final result: tag1 is not in the set
    /// ```
    ///
    /// # Example
    ///
    /// ```ignore
    /// let cache = Arc::new(KeyOfSetSieve::new(...));
    /// let writer = BackgroundWriter::new(4, db.clone());
    /// let mut buffer = writer.new_write_buffer();
    ///
    /// // Remove tags
    /// cache.remove_set_element(&user_id, &"deprecated".to_string(), &mut buffer);
    ///
    /// // Submit for persistence
    /// writer.submit_write_buffer(buffer);
    /// ```
    pub fn remove_set_element(
        self: &Arc<Self>,
        key: &C::Key,
        element: &C::Element,
        write_buffer: &mut write_behind::WriteBuffer<DB, S>,
    ) {
        let shard_index = self.shard_index(key);

        let updated = write_buffer.key_of_set_writes.put(
            key.clone(),
            element.clone(),
            Operation::Delete,
            self,
        );

        {
            let read_shard = self.storages.read_shard(shard_index);

            if let Some(entry) = read_shard.get(key) {
                match &entry.value {
                    KeyOfSetEntry::Full(container) => {
                        container.remove_element(element);
                    }
                    KeyOfSetEntry::Partial(partial) => {
                        partial.insert(element.clone(), Operation::Delete);
                    }
                }

                if updated {
                    entry
                        .pending_writes
                        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                }

                return;
            }
        }

        // has to acquire write lock
        let mut write_lock = self.storages.write_shard(shard_index);

        match write_lock.entry(key.clone()) {
            hash_map::Entry::Occupied(mut entry) => {
                match &mut entry.get_mut().value {
                    KeyOfSetEntry::Full(container) => {
                        container.remove_element(element);
                    }
                    KeyOfSetEntry::Partial(partial) => {
                        partial.insert(element.clone(), Operation::Delete);
                    }
                }

                if updated {
                    *entry.get_mut().pending_writes.get_mut() += 1;
                }
            }

            hash_map::Entry::Vacant(entry) => {
                let partial = DashMap::new();
                partial.insert(element.clone(), Operation::Delete);

                entry.insert(StorageEntry {
                    value: KeyOfSetEntry::Partial(partial),
                    visited: AtomicBool::new(false),
                    pending_writes: AtomicUsize::new(usize::from(updated)),
                });

                drop(write_lock);

                let mut policy_lock = self.policies.write_shard(shard_index);

                policy_lock.insert(key.clone(), &self.storages, shard_index);
            }
        }
    }
}
