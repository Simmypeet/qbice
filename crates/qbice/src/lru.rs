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
    hash::{BuildHasher, BuildHasherDefault, DefaultHasher, Hash},
    sync::{Arc, LazyLock},
};

use crossbeam_utils::CachePadded;
use tokio::sync::{MappedMutexGuard, Mutex, MutexGuard, Notify};

use crate::kv_database::{Column, KvDatabase};

/// Type alias for the array of cache shards.
type Shards<T, S> = Arc<[CachePadded<Mutex<LruShard<T, S>>>]>;

/// A node in the doubly-linked list used for LRU ordering.
///
/// Each node stores a key-value pair and maintains links to the previous
/// and next nodes in the list. The head of the list represents the most
/// recently used entry, while the tail represents the least recently used.
struct Node<T: Column> {
    key: T,
    value: T::Value,
    prev: Option<Index>,
    next: Option<Index>,
}

/// Represents the state of a cache entry.
///
/// This enum implements a "single-flight" pattern where concurrent requests
/// for the same key will wait on a single database fetch rather than
/// triggering multiple fetches.
enum Entry {
    /// A fetch operation is in progress. Waiters can subscribe to the
    /// [`Notify`] to be notified when the fetch completes.
    Working(Arc<Notify>),
    /// The entry is ready and stored at the given index in the node list.
    Ready(Index),
    /// The key was looked up but no value exists in the database.
    None,
}

type Index = usize;

/// A single shard of the LRU cache.
///
/// Each shard maintains its own hash map and doubly-linked list for LRU
/// ordering. By distributing entries across multiple shards, concurrent
/// access to different keys can proceed in parallel without contention.
struct LruShard<T: Column, S: BuildHasher> {
    map: HashMap<T, Entry, S>,

    nodes: Vec<Option<Node<T>>>,
    free_list: Vec<Index>,
    head: Option<Index>,
    tail: Option<Index>,

    length: usize,
    capacity: usize,
}

impl<T: Column, S: BuildHasher> LruShard<T, S> {
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

    fn evict_if_needed(&mut self) {
        if self.length <= self.capacity {
            return;
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
        self.map.remove(&key);
        self.length -= 1;
    }

    pub fn allocate_node(&mut self, key: T, value: T::Value) -> Index {
        let index = if let Some(free_index) = self.free_list.pop() {
            self.nodes[free_index] =
                Some(Node { key, value, prev: None, next: None });

            free_index
        } else {
            let index = self.nodes.len();
            self.nodes.push(Some(Node { key, value, prev: None, next: None }));

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
enum FastPathDecision<'x, T: Column, S: BuildHasher> {
    /// Cache hit (or cached miss). Contains the optional index and the
    /// acquired shard lock for immediate access.
    Hit(Option<Index>, MutexGuard<'x, LruShard<T, S>>),
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
    shards: Shards<T, S>,
    hasher_builder: S,
    db: Arc<DB>,
    shard_mask: usize,
}

impl<T: Column, DB: std::fmt::Debug, S: BuildHasher + Send + 'static>
    std::fmt::Debug for Lru<T, DB, S>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Lru")
            .field("shard_count", &self.shards.len())
            .field("db", &self.db)
            .finish_non_exhaustive()
    }
}

/// Returns the default number of shards to use.
///
/// The default is calculated as `4 * available_parallelism`, rounded up to
/// the next power of two. This provides good concurrency while keeping memory
/// overhead reasonable. The value is computed once and cached.
fn default_shard_amount() -> usize {
    static DEFAULT_SHARD_AMOUNT: LazyLock<usize> = LazyLock::new(|| {
        (std::thread::available_parallelism().map_or(1, usize::from) * 4)
            .next_power_of_two()
    });

    *DEFAULT_SHARD_AMOUNT
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
struct FetchDbGuard<'k, 's, T: Column, S: BuildHasher + Send + 'static> {
    key: &'k T,
    shard_index: usize,
    shards: &'s Shards<T, S>,
    defused: bool,
}

impl<T: Column, S: BuildHasher + Send + 'static> Drop
    for FetchDbGuard<'_, '_, T, S>
{
    fn drop(&mut self) {
        if self.defused {
            return;
        }

        let key = self.key.clone();
        let shard_index = self.shard_index;
        let shards = self.shards.clone();

        // spawn the task that will unlock the working entry and notify waiters
        tokio::spawn(async move {
            let mut shard_lock = shards[shard_index].lock().await;

            let Some(Entry::Working(notify)) = shard_lock.map.remove(&key)
            else {
                panic!("FetchDbGuard dropped but entry is not Working");
            };

            drop(shard_lock);

            notify.notify_waiters();
        });
    }
}

impl<T: Column, DB: KvDatabase, S: BuildHasher + Send + 'static> Lru<T, DB, S> {
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

        let mut shards = Vec::with_capacity(shard_amount);

        for _ in 0..shard_amount {
            shards.push(CachePadded::new(Mutex::new(LruShard::<T, S> {
                map: HashMap::with_hasher(hasher_builder.clone()),
                capacity: per_shard_capacity,
                nodes: Vec::new(),
                free_list: Vec::new(),
                length: 0,
                head: None,
                tail: None,
            })));
        }

        Self {
            shards: Arc::from(shards),
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
    fn determine_shard_index(&self, key: &T) -> usize
    where
        T: Hash,
    {
        (self.hasher_builder.hash_one(key) as usize) & self.shard_mask
    }

    async fn lock_shard(&self, key: &T) -> MutexGuard<'_, LruShard<T, S>> {
        let shard_index = self.determine_shard_index(key);
        self.shards[shard_index].lock().await
    }

    /// Attempts the fast path for a cache lookup.
    ///
    /// The fast path checks if the entry is already in the cache (hit or
    /// known miss). If another task is currently fetching the same key,
    /// this method waits for that fetch to complete and signals the caller
    /// to retry.
    ///
    /// Returns a [`FastPathDecision`] indicating the result.
    async fn fast_path(&self, key: &T) -> FastPathDecision<'_, T, S> {
        let shard = self.lock_shard(key).await;

        match shard.map.get(key) {
            Some(Entry::None) => FastPathDecision::Hit(None, shard),
            Some(Entry::Ready(index)) => {
                FastPathDecision::Hit(Some(*index), shard)
            }

            // there're some process fetching the value from the database.
            Some(Entry::Working(notify)) => {
                let notify = notify.clone();
                let notified = notify.notified();

                drop(shard);

                notified.await;

                FastPathDecision::TryAgain
            }

            None => FastPathDecision::ToSlowPath,
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
    /// - `Some(guard)` - A guard providing mutable access to the cached value
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
    /// This method might **deadlock** if an existing shard lock is held.
    pub async fn get(&self, key: &T) -> Option<MappedMutexGuard<'_, T::Value>>
    where
        T: Hash + Eq,
    {
        loop {
            match self.fast_path(key).await {
                FastPathDecision::Hit(opt_index, mut shard) => {
                    match opt_index {
                        Some(index) => {
                            shard.move_to_head(index);

                            return Some(MutexGuard::map(shard, |shard| {
                                &mut shard.nodes[index].as_mut().unwrap().value
                            }));
                        }

                        None => return None,
                    }
                }

                FastPathDecision::ToSlowPath => {}

                FastPathDecision::TryAgain => continue,
            }

            // slow path: need to fetch from DB
            let Some(fetch_db_guard) = self.acquire_fetch_db_guard(key).await
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

        let mut shard_lock = self.shards[guard.shard_index].lock().await;
        shard_lock.evict_if_needed();

        let Some(Entry::Working(notify)) = shard_lock.map.remove(guard.key)
        else {
            panic!("fetch_db should be called on a Working entry");
        };

        match value_opt {
            Some(value) => {
                let index = shard_lock.allocate_node(guard.key.clone(), value);

                shard_lock.map.insert(guard.key.clone(), Entry::Ready(index));
                shard_lock.move_to_head(index);
            }

            None => {
                shard_lock.map.insert(guard.key.clone(), Entry::None);
            }
        }

        guard.defused = true;
        drop(shard_lock);

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
        key: &'s T,
    ) -> Option<FetchDbGuard<'s, '_, T, S>> {
        let shard_index = self.determine_shard_index(key);
        let mut shard_lock = self.shards[shard_index].lock().await;

        // no entry exists, we can insert a Working entry and proceed to fetch
        // from DB
        if shard_lock.map.contains_key(key) {
            None
        } else {
            let notify = Arc::new(Notify::new());
            shard_lock.map.insert(key.clone(), Entry::Working(notify));

            Some(FetchDbGuard {
                key,
                shard_index,
                shards: &self.shards,
                defused: false,
            })
        }
    }
}
