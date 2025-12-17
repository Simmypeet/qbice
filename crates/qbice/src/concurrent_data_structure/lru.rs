//! A sharded, async-aware LRU (Least Recently Used) cache backed by a
//! key-value database.
//!
//! This module provides [`Lru`], a concurrent cache that automatically evicts
//! the least recently used entries when capacity is reached. It uses sharding
//! to reduce lock contention and integrates with an async runtime (Tokio) for
//! non-blocking operations.

use core::panic;
use std::{
    collections::HashMap,
    hash::{BuildHasher, BuildHasherDefault, DefaultHasher},
    sync::Arc,
};

use crossbeam_utils::CachePadded;
use enum_as_inner::EnumAsInner;
use tokio::sync::{Mutex, Notify, RwLock, RwLockReadGuard};

use crate::{
    concurrent_data_structure::default_shard_amount,
    kv_database::{Column, KvDatabase},
};

/// Type alias for the shards of usage tracking.
type UsageShards<K> = Arc<[CachePadded<Mutex<UsageShard<K>>>]>;

type KvMap<T, S> = HashMap<<T as Column>::Key, Entry<<T as Column>::Value>, S>;

/// Type alias for the shards of where the actual data is stored.
type StorageShards<T, S> = Arc<[CachePadded<RwLock<KvMap<T, S>>>]>;

/// A node in the doubly-linked list used for LRU ordering.
///
/// Each node stores a key and maintains links to the previous and next nodes
/// in the list. The actual values are stored separately in the storage shards.
/// The head of the list represents the most recently used entry, while the
/// tail represents the least recently used.
struct Node<K> {
    key: K,
    prev: Option<Index>,
    next: Option<Index>,
}

struct Ready<V> {
    index: Index,
    value: Option<V>,
}

/// Represents the state of a cache entry.
///
/// This enum implements a "single-flight" pattern where concurrent requests
/// for the same key will wait on a single database fetch rather than
/// triggering multiple fetches.
#[derive(EnumAsInner)]
enum Entry<K> {
    /// A fetch operation is in progress. Waiters can subscribe to the
    /// [`Notify`] to be notified when the fetch completes.
    Working(Arc<Notify>),
    /// The entry is ready and stored at the given index in the node list.
    Ready(Ready<K>),
}

type Index = usize;

/// A single shard for tracking LRU ordering.
///
/// Each shard maintains a doubly-linked list for LRU ordering. The actual
/// key-value data is stored separately in the corresponding storage shard.
/// By distributing entries across multiple shards, concurrent access to
/// different keys can proceed in parallel without contention.
struct UsageShard<K> {
    nodes: Vec<Option<Node<K>>>,
    free_list: Vec<Index>,
    head: Option<Index>,
    tail: Option<Index>,

    length: usize,
    capacity: usize,
}

impl<K> UsageShard<K> {
    /// Moves the node at the given index to the head of the LRU list.
    ///
    /// This operation marks the entry as "most recently used". The algorithm:
    /// 1. Unlinks the node from its current position by updating neighbors
    /// 2. Updates the tail pointer if the node was at the tail
    /// 3. Links the node at the head of the list
    ///
    /// This is an O(1) operation due to the doubly-linked list structure.
    fn move_to_head(&mut self, index: Index) {
        if Some(index) == self.head {
            return;
        }

        let (prev, next) = {
            let node = self.nodes[index].as_mut().unwrap();

            (node.prev, node.next)
        };

        // unlink the node
        if let Some(prev_index) = prev {
            self.nodes[prev_index].as_mut().unwrap().next = next;
        }

        if let Some(next_index) = next {
            self.nodes[next_index].as_mut().unwrap().prev = prev;
        } else {
            // update tail if needed
            self.tail = prev;
        }

        // link to head
        if let Some(old_head) = self.head {
            self.nodes[old_head].as_mut().unwrap().prev = Some(index);
        }

        {
            let node = self.nodes[index].as_mut().unwrap();
            node.prev = None;
            node.next = self.head;
        }

        self.head = Some(index);

        if self.tail.is_none() {
            self.tail = Some(index);
        }
    }

    fn evict_if_needed(&mut self) -> Option<K> {
        if self.length < self.capacity {
            return None;
        }

        // evict the tail node
        let tail_index = self.tail.expect("Tail must exist when evicting");

        let (prev, key) = {
            let tail_node =
                self.nodes[tail_index].take().expect("should've existed");

            (tail_node.prev, tail_node.key)
        };

        // unlink the tail node
        if let Some(prev_index) = prev {
            self.nodes[prev_index].as_mut().unwrap().next = None;
            self.tail = Some(prev_index);
        } else {
            // list is now empty
            self.head = None;
            self.tail = None;
        }

        self.free_list.push(tail_index);
        self.length -= 1;

        Some(key)
    }

    pub fn allocate_node(&mut self, key: K) -> Index {
        let index = if let Some(free_index) = self.free_list.pop() {
            self.nodes[free_index] = Some(Node { key, prev: None, next: None });

            free_index
        } else {
            let index = self.nodes.len();
            self.nodes.push(Some(Node { key, prev: None, next: None }));

            index
        };

        self.length += 1;

        index
    }
}

/// The result of attempting the fast path for a cache lookup.
///
/// The fast path succeeds when the entry is already cached or known to be
/// absent. It fails when a database fetch is needed or when another task
/// is currently fetching the same key.
enum FastPathDecision<'x, T: Column> {
    /// Cache hit (or cached miss). Contains the index in the LRU list and the
    /// acquired shard lock for immediate access.
    Hit(Index, RwLockReadGuard<'x, Option<T::Value>>),
    /// No cached entry exists; must proceed to the slow path (database fetch).
    ToSlowPath,
    /// Another task is fetching this key; caller should retry after waiting.
    TryAgain,
}

/// A sharded, async-aware LRU cache backed by a key-value database.
///
/// `Lru` provides a concurrent cache with automatic eviction of least recently
/// used entries. It is designed for high-concurrency scenarios with the
/// following features:
///
/// - **Sharding**: Entries are distributed across multiple shards (determined
///   by hashing the key), allowing concurrent access to different keys without
///   lock contention.
/// - **Single-flight fetches**: When multiple tasks request the same uncached
///   key simultaneously, only one database fetch is performed. Other tasks wait
///   and share the result.
/// - **Async-native**: All operations are async and integrate with Tokio.
///
/// # Type Parameters
///
/// - `T`: The column type that defines the key-value schema (must implement
///   [`Column`]).
/// - `DB`: The backing database type (must implement [`KvDatabase`]).
/// - `S`: The hasher builder for determining shard assignment (defaults to
///   [`DefaultHasher`]).
///
/// # Example
///
/// ```ignore
/// let db = Arc::new(MyDatabase::new());
/// let cache = Lru::<MyColumn, _, _>::new(db, 1000, Default::default());
///
/// // Get a value (fetches from DB on cache miss)
/// if let Some(value) = cache.get(&key).await {
///     // use value
/// }
/// ```
pub struct Lru<
    T: Column,
    DB,
    S: BuildHasher = BuildHasherDefault<DefaultHasher>,
> {
    usage_shards: UsageShards<T::Key>,
    storage_shards: StorageShards<T, S>,

    hasher_builder: S,
    db: Arc<DB>,
    shard_mask: usize,
}

impl<T: Column, DB: std::fmt::Debug, S: BuildHasher + Send + 'static>
    std::fmt::Debug for Lru<T, DB, S>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Lru").finish_non_exhaustive()
    }
}

/// A RAII guard that ensures proper cleanup when fetching from the database.
///
/// This guard implements an important safety mechanism: if a database fetch
/// is interrupted (e.g., by a panic or task cancellation), the guard's `Drop`
/// implementation will:
///
/// 1. Remove the `Working` entry from the cache
/// 2. Notify all waiters so they can retry or handle the failure
///
/// When a fetch completes successfully, the guard should be "defused" by
/// setting `defused = true`, which prevents the cleanup logic from running.
///
/// This pattern ensures the cache never gets stuck with a `Working` entry
/// that will never complete.
struct FetchDbGuard<'k, 's, T: Column, S: BuildHasher + Send + Sync + 'static> {
    key: &'k T::Key,
    shard_index: usize,
    storage_shards: &'s StorageShards<T, S>,
    defused: bool,
}

impl<T: Column, S: BuildHasher + Send + Sync + 'static> Drop
    for FetchDbGuard<'_, '_, T, S>
{
    fn drop(&mut self) {
        if self.defused {
            return;
        }

        let key = self.key.clone();
        let shard_index = self.shard_index;
        let shards = self.storage_shards.clone();

        // spawn the task that will unlock the working entry and notify waiters
        tokio::spawn(async move {
            let mut shard_lock = shards[shard_index].write().await;

            let Some(Entry::Working(notify)) = shard_lock.remove(&key) else {
                panic!("FetchDbGuard dropped but entry is not Working");
            };

            drop(shard_lock);

            notify.notify_waiters();
        });
    }
}

impl<T: Column, DB: KvDatabase, S: BuildHasher + Send + Sync + 'static>
    Lru<T, DB, S>
{
    /// Creates a new LRU cache with the specified capacity.
    ///
    /// The cache will use the default number of shards (based on available
    /// parallelism). The capacity is distributed evenly across all shards.
    ///
    /// # Arguments
    ///
    /// * `db` - The backing key-value database.
    /// * `capacity` - The maximum number of entries to cache (across all
    ///   shards).
    /// * `hasher_builder` - The hasher builder for shard assignment.
    ///
    /// # Panics
    ///
    /// Panics if `capacity` is zero.
    pub fn new(db: Arc<DB>, capacity: usize, hasher_builder: S) -> Self
    where
        S: Clone,
    {
        let shard_amount = default_shard_amount();

        Self::new_with_shard_amount(db, capacity, shard_amount, hasher_builder)
    }

    /// Creates a new LRU cache with a custom number of shards.
    ///
    /// This constructor allows fine-tuning the number of shards for specific
    /// workloads. More shards reduce contention but increase memory overhead.
    ///
    /// # Arguments
    ///
    /// * `db` - The backing key-value database.
    /// * `capacity` - The maximum number of entries to cache (across all
    ///   shards).
    /// * `shard_amount` - The number of shards. Must be a power of two.
    /// * `hasher_builder` - The hasher builder for shard assignment.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `capacity` is zero
    /// - `shard_amount` is not a power of two
    pub fn new_with_shard_amount(
        db: Arc<DB>,
        capacity: usize,
        shard_amount: usize,
        hasher_builder: S,
    ) -> Self
    where
        S: Clone,
    {
        assert!(capacity > 0, "Capacity must be greater than zero");
        assert!(
            shard_amount.is_power_of_two(),
            "Shard amount must be a power of two"
        );

        let per_shard_capacity = capacity.div_ceil(shard_amount);

        let mut usage_shards = Vec::with_capacity(shard_amount);
        let mut storage_shards = Vec::with_capacity(shard_amount);

        for _ in 0..shard_amount {
            usage_shards.push(CachePadded::new(Mutex::new(UsageShard::<
                T::Key,
            > {
                capacity: per_shard_capacity,
                nodes: Vec::new(),
                free_list: Vec::new(),
                length: 0,
                head: None,
                tail: None,
            })));
            storage_shards.push(CachePadded::new(RwLock::new(
                HashMap::with_hasher(hasher_builder.clone()),
            )));
        }

        Self {
            usage_shards: Arc::from(usage_shards),
            storage_shards: Arc::from(storage_shards),
            hasher_builder,
            db,
            shard_mask: shard_amount - 1,
        }
    }

    /// Determines which shard a key belongs to.
    ///
    /// Uses the hash of the key and a bitmask (since shard count is a power
    /// of two) for efficient shard selection.
    #[allow(clippy::cast_possible_truncation)]
    fn determine_shard_index(&self, key: &T::Key) -> usize {
        (self.hasher_builder.hash_one(key) as usize) & self.shard_mask
    }

    /// Attempts the fast path for a cache lookup.
    ///
    /// The fast path checks if the entry is already in the cache (hit or
    /// known miss). If another task is currently fetching the same key,
    /// this method waits for that fetch to complete and signals the caller
    /// to retry.
    ///
    /// Returns a [`FastPathDecision`] indicating the result.
    async fn fast_path(
        &self,
        key: &T::Key,
        shard_index: usize,
    ) -> FastPathDecision<'_, T> {
        let shard_lock = RwLockReadGuard::try_map(
            self.storage_shards[shard_index].read().await,
            |x| x.get(key),
        );

        match shard_lock {
            Ok(shard) => match &*shard {
                Entry::Working(notify) => {
                    let notify = notify.clone();
                    let notified = notify.notified();

                    drop(shard);

                    notified.await;

                    FastPathDecision::TryAgain
                }
                Entry::Ready(ready) => {
                    let index = ready.index;

                    let mapped = RwLockReadGuard::map(shard, |x| {
                        &x.as_ready().expect("should've been ready").value
                    });

                    FastPathDecision::Hit(index, mapped)
                }
            },

            // no entry found return ToSlowPath
            Err(_) => FastPathDecision::ToSlowPath,
        }
    }

    /// Retrieves a value from the cache, fetching from the database on miss.
    ///
    /// This method implements the complete lookup flow:
    /// 1. **Fast path**: Check if the value is already cached
    /// 2. **Slow path**: If not cached, fetch from the database
    ///
    /// The returned guard holds the shard lock, ensuring the value remains
    /// valid while in use. The entry is moved to the head of the LRU list
    /// (marking it as most recently used).
    ///
    /// # Returns
    ///
    /// - `Some(guard)` - A read guard providing access to the cached value
    /// - `None` - The key does not exist in the database
    ///
    /// # Concurrency
    ///
    /// If multiple tasks call `get` with the same uncached key simultaneously,
    /// only one will fetch from the database. Others will wait and receive
    /// the cached result.
    ///
    /// # Deadlocks
    ///
    /// This method might **deadlock** if an existing shard lock is held. Even
    /// though the function returns read locks, however, the internal logic may
    /// evict entries and acquire write locks during the fetch process. To avoid
    /// deadlocks, ensure that no shard locks are held when calling this method.
    pub async fn get(
        &self,
        key: &T::Key,
    ) -> Option<RwLockReadGuard<'_, T::Value>> {
        let shard_index = self.determine_shard_index(key);

        loop {
            match self.fast_path(key, shard_index).await {
                FastPathDecision::Hit(opt_index, shard) => {
                    // move to head
                    let mut usage_shard_lock =
                        self.usage_shards[shard_index].lock().await;
                    usage_shard_lock.move_to_head(opt_index);
                    drop(usage_shard_lock);

                    return RwLockReadGuard::try_map(shard, Option::as_ref)
                        .ok();
                }

                FastPathDecision::ToSlowPath => {}

                FastPathDecision::TryAgain => continue,
            }

            // slow path: need to fetch from DB
            let Some(fetch_db_guard) =
                self.acquire_fetch_db_guard(key, shard_index).await
            else {
                continue;
            };

            self.fetch_db(fetch_db_guard).await;

            // after fetching from DB, retry the fast path, which should now hit
        }
    }

    async fn fetch_db(&self, mut guard: FetchDbGuard<'_, '_, T, S>) {
        // NOTE: during the fetching, we drop the lock to allows other tasks to
        // possibly access the cache. They will either hit the cache or wait on
        // the `Notify` associated with the `Working` entry.

        // hit the database
        let value_opt = self.db.get::<T>(guard.key).await;

        let mut usage_shard_lock =
            self.usage_shards[guard.shard_index].lock().await;

        // determine if we need to evict
        let removed_key = usage_shard_lock.evict_if_needed();

        let new_node_index = usage_shard_lock.allocate_node(guard.key.clone());
        usage_shard_lock.move_to_head(new_node_index);

        // insert into storage shard
        let mut storage_shard_lock =
            self.storage_shards[guard.shard_index].write().await;

        let Some(Entry::Working(notify)) = storage_shard_lock.remove(guard.key)
        else {
            panic!("fetch_db should be called on a Working entry");
        };

        // remove evicted entry from storage shard
        if let Some(evicted_key) = removed_key {
            storage_shard_lock.remove(&evicted_key);
        }

        // insert the fetched value (or None) into the cache
        storage_shard_lock.insert(
            guard.key.clone(),
            Entry::Ready(Ready { index: new_node_index, value: value_opt }),
        );

        guard.defused = true;

        drop(usage_shard_lock);
        drop(storage_shard_lock);

        notify.notify_waiters();
    }

    /// Attempts to acquire a guard for fetching a key from the database.
    ///
    /// This method atomically checks if an entry exists and, if not, inserts
    /// a `Working` entry to signal that a fetch is in progress. This prevents
    /// duplicate fetches for the same key (single-flight pattern).
    ///
    /// # Returns
    ///
    /// - `Some(guard)` - Successfully acquired; caller should fetch from DB
    /// - `None` - Another task is already fetching or entry exists; caller
    ///   should retry
    async fn acquire_fetch_db_guard<'s>(
        &self,
        key: &'s T::Key,
        shard_index: usize,
    ) -> Option<FetchDbGuard<'s, '_, T, S>> {
        let mut shard_lock = self.storage_shards[shard_index].write().await;

        // no entry exists, we can insert a Working entry and proceed to fetch
        // from DB
        if shard_lock.contains_key(key) {
            None
        } else {
            let notify = Arc::new(Notify::new());
            shard_lock.insert(key.clone(), Entry::Working(notify));

            Some(FetchDbGuard {
                key,
                shard_index,
                storage_shards: &self.storage_shards,
                defused: false,
            })
        }
    }
}

#[cfg(test)]
mod test;
