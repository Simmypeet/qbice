use std::{
    any::{Any, TypeId},
    collections::HashMap,
    hash::{BuildHasherDefault, DefaultHasher},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use tokio::sync::RwLock;

use crate::{
    kv_database::{Column, KvDatabase, WriteTransaction},
    lru::Lru,
};

/// Test column type: i32 keys to String values
struct TestColumn;

impl Column for TestColumn {
    type Key = i32;
    type Value = String;
}

/// A mock key-value database using type-erased storage.
///
/// Each column type `C: Column` gets its own `HashMap<C::Key, C::Value>` stored
/// as a type-erased `Box<dyn Any + Send + Sync>` keyed by `TypeId::of::<C>()`.
#[derive(Default)]
struct MockDatabase {
    /// Maps `TypeId` of column -> `HashMap<C::Key, C::Value>` (type-erased)
    data: RwLock<HashMap<TypeId, Box<dyn Any + Send + Sync>>>,
    get_count: AtomicUsize,
}

impl std::fmt::Debug for MockDatabase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MockDatabase").finish_non_exhaustive()
    }
}

impl MockDatabase {
    fn new() -> Self {
        Self {
            data: RwLock::new(HashMap::new()),
            get_count: AtomicUsize::new(0),
        }
    }

    /// Insert a value for a specific column type
    async fn insert<C: Column>(&self, key: C::Key, value: C::Value)
    where
        C::Key: Send + Sync + 'static,
        C::Value: Send + Sync + 'static,
    {
        let type_id = TypeId::of::<C>();
        let mut data = self.data.write().await;

        let column_map = data
            .entry(type_id)
            .or_insert_with(|| Box::new(HashMap::<C::Key, C::Value>::new()));

        // Downcast to the actual HashMap type and insert
        column_map
            .downcast_mut::<HashMap<C::Key, C::Value>>()
            .expect("Type mismatch in MockDatabase")
            .insert(key, value);
    }

    fn get_count(&self) -> usize { self.get_count.load(Ordering::SeqCst) }
}

/// A no-op write transaction for testing
struct MockWriteTransaction;

impl WriteTransaction for MockWriteTransaction {
    async fn put<'s, C: Column>(
        &'s self,
        _key: &'s C,
        _value: &'s <C as Column>::Value,
    ) {
        // no-op
    }

    async fn commit(self) -> Result<(), std::io::Error> { Ok(()) }
}

impl KvDatabase for MockDatabase {
    type WriteTransaction<'a> = MockWriteTransaction;

    async fn get<'s, C: Column>(
        &'s self,
        key: &'s C::Key,
    ) -> Option<<C as Column>::Value> {
        self.get_count.fetch_add(1, Ordering::SeqCst);

        let type_id = TypeId::of::<C>();
        let data = self.data.read().await;

        let column_map = data.get(&type_id)?;
        let map = column_map.downcast_ref::<HashMap<C::Key, C::Value>>()?;

        map.get(key).cloned()
    }

    async fn write_transaction(&self) -> Self::WriteTransaction<'_> {
        MockWriteTransaction
    }
}

/// Helper function to create the standard hasher for tests
fn test_hasher() -> BuildHasherDefault<DefaultHasher> {
    BuildHasherDefault::default()
}

// ============================================================================
// Basic Functionality Tests
// ============================================================================

#[tokio::test]
async fn cache_miss_returns_none() {
    let db = Arc::new(MockDatabase::new());
    let cache = Lru::<TestColumn, _, _>::new_with_shard_amount(
        db,
        10,
        1,
        test_hasher(),
    );

    let result = cache.get(&42).await;
    assert!(result.is_none());
}

#[tokio::test]
async fn cache_hit_returns_value() {
    let db = Arc::new(MockDatabase::new());
    db.insert::<TestColumn>(1, "one".to_string()).await;

    let cache = Lru::<TestColumn, _, _>::new_with_shard_amount(
        db,
        10,
        1,
        test_hasher(),
    );

    let result = cache.get(&1).await;
    assert_eq!(result.as_deref().map(AsRef::as_ref), Some("one"));
}

#[tokio::test]
async fn cache_hit_after_first_get() {
    let db = Arc::new(MockDatabase::new());
    db.insert::<TestColumn>(1, "one".to_string()).await;
    db.insert::<TestColumn>(2, "two".to_string()).await;

    let cache = Lru::<TestColumn, _, _>::new_with_shard_amount(
        db.clone(),
        10,
        1,
        test_hasher(),
    );

    // First access - should hit DB
    let _ = cache.get(&1).await;
    assert_eq!(db.get_count(), 1);

    // Second access - should be cached (no additional DB hit)
    let _ = cache.get(&1).await;
    assert_eq!(db.get_count(), 1);

    // Access different key - should hit DB
    let _ = cache.get(&2).await;
    assert_eq!(db.get_count(), 2);

    // Access same key again - should be cached
    let _ = cache.get(&2).await;
    assert_eq!(db.get_count(), 2);
}

#[tokio::test]
async fn negative_cache() {
    // Test that cache remembers "not found" results
    let db = Arc::new(MockDatabase::new());
    let cache = Lru::<TestColumn, _, _>::new_with_shard_amount(
        db.clone(),
        10,
        1,
        test_hasher(),
    );

    // First access - key doesn't exist, should hit DB
    let result = cache.get(&999).await;
    assert!(result.is_none());
    assert_eq!(db.get_count(), 1);

    // Second access - should be cached as "not found" (no additional DB hit)
    let result = cache.get(&999).await;
    assert!(result.is_none());
    assert_eq!(db.get_count(), 1);
}

// ============================================================================
// LRU Eviction Tests (Single Shard for Deterministic Behavior)
//
// Note: Eviction happens when length >= capacity. With capacity N, the cache
// can hold exactly N entries. Adding the (N+1)th unique entry triggers
// eviction.
// ============================================================================

#[tokio::test]
async fn eviction_removes_least_recently_used() {
    // Test eviction behavior by observing DB access counts
    // We use a small capacity to trigger evictions
    let db = Arc::new(MockDatabase::new());
    db.insert::<TestColumn>(1, "one".to_string()).await;
    db.insert::<TestColumn>(2, "two".to_string()).await;
    db.insert::<TestColumn>(3, "three".to_string()).await;
    db.insert::<TestColumn>(4, "four".to_string()).await;
    db.insert::<TestColumn>(5, "five".to_string()).await;
    db.insert::<TestColumn>(6, "six".to_string()).await;

    // Capacity 2, 1 shard: per_shard_capacity = 2
    // Eviction triggers when length >= 2, i.e., when adding 3rd entry
    let cache = Lru::<TestColumn, _, _>::new_with_shard_amount(
        db.clone(),
        2,
        1,
        test_hasher(),
    );

    // Access keys 1, 2 - fills capacity
    let _ = cache.get(&1).await;
    let _ = cache.get(&2).await;
    let count_after_2 = db.get_count();
    assert_eq!(
        count_after_2, 2,
        "Should have 2 DB accesses after getting 2 keys"
    );

    // Both should be cached now
    let _ = cache.get(&1).await;
    let _ = cache.get(&2).await;
    assert_eq!(db.get_count(), 2, "Keys 1,2 should all be cached");

    // Access key 3 - triggers eviction of key 1 (the oldest/tail)
    let _ = cache.get(&3).await;
    let count_after_3 = db.get_count();
    assert_eq!(
        count_after_3, 3,
        "Should have 3 DB accesses after getting key 3"
    );

    // Check if key 1 was evicted by accessing it
    let _ = cache.get(&1).await;
    let count_after_1_again = db.get_count();

    // If key 1 was evicted, count should be 4
    // If key 1 is still cached, count should be 3
    // Based on the eviction logic: when we add key 3 with length=2,
    // evict_if_needed sees length(2) >= capacity(2), so it evicts the tail
    // (key 1)
    assert_eq!(
        count_after_1_again, 4,
        "Key 1 should have been evicted when adding key 3, requiring a DB \
         fetch"
    );
}

#[tokio::test]
async fn access_updates_lru_order() {
    // With capacity 3, can hold exactly 3 entries
    let db = Arc::new(MockDatabase::new());
    db.insert::<TestColumn>(1, "one".to_string()).await;
    db.insert::<TestColumn>(2, "two".to_string()).await;
    db.insert::<TestColumn>(3, "three".to_string()).await;
    db.insert::<TestColumn>(4, "four".to_string()).await;

    let cache = Lru::<TestColumn, _, _>::new_with_shard_amount(
        db.clone(),
        3,
        1,
        test_hasher(),
    );

    // Access keys 1, 2, 3 - fills capacity
    // LRU order: 3 (head) -> 2 -> 1 (tail)
    let _ = cache.get(&1).await;
    let _ = cache.get(&2).await;
    let _ = cache.get(&3).await;

    // Access key 1 again - moves to head
    // LRU order: 1 (head) -> 3 -> 2 (tail)
    let _ = cache.get(&1).await;
    assert_eq!(db.get_count(), 3); // No additional DB hit

    // Access key 4 - should evict key 2 (now tail/LRU)
    // LRU order: 4 (head) -> 1 -> 3 (tail)
    let _ = cache.get(&4).await;
    assert_eq!(db.get_count(), 4);

    // Key 1 should still be cached (was refreshed)
    let _ = cache.get(&1).await;
    assert_eq!(db.get_count(), 4); // No additional DB hit

    // Key 3 should still be cached
    let _ = cache.get(&3).await;
    assert_eq!(db.get_count(), 4); // No additional DB hit

    // Key 2 was evicted
    let _ = cache.get(&2).await;
    assert_eq!(db.get_count(), 5); // DB hit for evicted key 2
}

#[tokio::test]
async fn capacity_one() {
    // With capacity 1, can hold exactly 1 entry
    let db = Arc::new(MockDatabase::new());
    db.insert::<TestColumn>(1, "one".to_string()).await;
    db.insert::<TestColumn>(2, "two".to_string()).await;
    db.insert::<TestColumn>(3, "three".to_string()).await;

    let cache = Lru::<TestColumn, _, _>::new_with_shard_amount(
        db.clone(),
        1,
        1,
        test_hasher(),
    );

    // Access key 1
    let _ = cache.get(&1).await;
    assert_eq!(db.get_count(), 1);

    // Access key 1 again - should be cached
    let _ = cache.get(&1).await;
    assert_eq!(db.get_count(), 1);

    // Access key 2 - should evict key 1 (capacity is 1)
    let _ = cache.get(&2).await;
    assert_eq!(db.get_count(), 2);

    // Access key 1 - should hit DB (was evicted)
    let _ = cache.get(&1).await;
    assert_eq!(db.get_count(), 3);

    // Access key 3 - should evict key 1
    let _ = cache.get(&3).await;
    assert_eq!(db.get_count(), 4);
}

#[tokio::test]
async fn eviction_with_negative_cache() {
    // Test that negative cache entries (None values) are also subject to
    // eviction
    let db = Arc::new(MockDatabase::new());
    db.insert::<TestColumn>(3, "three".to_string()).await;

    // With capacity 2: eviction happens when length >= 2
    let cache = Lru::<TestColumn, _, _>::new_with_shard_amount(
        db.clone(),
        2,
        1,
        test_hasher(),
    );

    // Access non-existent keys 1, 2 (cached as None) - fills capacity
    let _ = cache.get(&1).await;
    let _ = cache.get(&2).await;
    assert_eq!(db.get_count(), 2);

    // Verify both negative entries are cached
    let _ = cache.get(&1).await;
    let _ = cache.get(&2).await;
    assert_eq!(db.get_count(), 2, "Keys 1,2 should all be cached as None");

    // Access existing key 3 - should evict key 1 (tail)
    let _ = cache.get(&3).await;
    assert_eq!(db.get_count(), 3);

    // Key 1 should be evicted
    let _ = cache.get(&1).await;
    assert_eq!(db.get_count(), 4, "Key 1 should have been evicted");
}

// ============================================================================
// Concurrent Access Tests
// ============================================================================

#[tokio::test]
async fn concurrent_access_same_key() {
    // Test single-flight behavior: multiple concurrent requests for the same
    // key should result in only one DB fetch
    let db = Arc::new(MockDatabase::new());
    db.insert::<TestColumn>(1, "one".to_string()).await;

    let cache = Arc::new(Lru::<TestColumn, _, _>::new_with_shard_amount(
        db.clone(),
        10,
        1,
        test_hasher(),
    ));

    // Spawn multiple tasks that all request the same key
    let mut handles = Vec::new();
    for _ in 0..10 {
        let cache = cache.clone();
        handles.push(tokio::spawn(async move {
            let result = cache.get(&1).await;
            assert_eq!(&*result.unwrap(), "one");
        }));
    }

    for handle in handles {
        handle.await.unwrap();
    }

    // Should only have hit the DB once due to single-flight
    // Note: there might be race conditions where multiple tasks hit DB
    // before the first one completes, so we allow for some tolerance
    assert!(
        db.get_count() <= 3,
        "Expected at most 3 DB hits due to single-flight, got {}",
        db.get_count()
    );
}

#[tokio::test]
async fn concurrent_access_different_keys() {
    let db = Arc::new(MockDatabase::new());
    for i in 1..=5 {
        db.insert::<TestColumn>(i, format!("value_{i}")).await;
    }

    // Use multiple shards to test concurrent access
    let cache = Arc::new(Lru::<TestColumn, _, _>::new_with_shard_amount(
        db.clone(),
        10,
        4,
        test_hasher(),
    ));

    let mut handles = Vec::new();
    for i in 1..=5 {
        let cache = cache.clone();
        handles.push(tokio::spawn(async move {
            let result = cache.get(&i).await;
            assert!(result.is_some());
        }));
    }

    for handle in handles {
        handle.await.unwrap();
    }

    // Each unique key should hit the DB exactly once
    assert_eq!(db.get_count(), 5);
}

// ============================================================================
// Stress Tests
// ============================================================================

#[tokio::test]
async fn many_inserts_and_evictions() {
    let db = Arc::new(MockDatabase::new());
    // Insert 15 keys
    for i in 0..15 {
        db.insert::<TestColumn>(i, format!("value_{i}")).await;
    }

    // With capacity 10, eviction happens when length >= 10
    // So cache holds exactly 10 entries
    let cache = Lru::<TestColumn, _, _>::new_with_shard_amount(
        db.clone(),
        10,
        1,
        test_hasher(),
    );

    // Access keys 0-9 (10 keys total) - fills capacity
    for i in 0..10 {
        let _ = cache.get(&i).await;
    }
    assert_eq!(db.get_count(), 10, "First 10 keys should hit DB");

    // All 10 keys should be cached (exactly at capacity)
    let count = db.get_count();
    for i in 0..10 {
        let _ = cache.get(&i).await;
    }
    assert_eq!(db.get_count(), count, "All 10 keys should still be cached");

    // Now access key 10 - this triggers eviction of key 0
    let _ = cache.get(&10).await;
    assert_eq!(db.get_count(), count + 1, "Key 10 should hit DB");

    // Key 0 should now be evicted
    let count2 = db.get_count();
    let _ = cache.get(&0).await;
    assert_eq!(
        db.get_count(),
        count2 + 1,
        "Key 0 should have been evicted and require DB fetch"
    );
}

#[tokio::test]
async fn repeated_access_pattern() {
    // Test working set that fits in cache
    let db = Arc::new(MockDatabase::new());
    for i in 0..5 {
        db.insert::<TestColumn>(i, format!("value_{i}")).await;
    }

    // With capacity 5, can hold exactly 5 entries
    let cache = Lru::<TestColumn, _, _>::new_with_shard_amount(
        db.clone(),
        5,
        1,
        test_hasher(),
    );

    // First pass: load all 5 keys
    for i in 0..5 {
        let _ = cache.get(&i).await;
    }
    assert_eq!(db.get_count(), 5);

    // Second pass: all should be cached
    for i in 0..5 {
        let _ = cache.get(&i).await;
    }
    assert_eq!(db.get_count(), 5);

    // Third pass with different order: all should still be cached
    for i in [4, 2, 0, 3, 1] {
        let _ = cache.get(&i).await;
    }
    assert_eq!(db.get_count(), 5);
}

// ============================================================================
// Edge Cases
// ============================================================================

#[tokio::test]
#[should_panic(expected = "Capacity must be greater than zero")]
async fn zero_capacity_panics() {
    let db = Arc::new(MockDatabase::new());
    let _cache =
        Lru::<TestColumn, _, _>::new_with_shard_amount(db, 0, 1, test_hasher());
}

#[tokio::test]
#[should_panic(expected = "Shard amount must be a power of two")]
async fn non_power_of_two_shards_panics() {
    let db = Arc::new(MockDatabase::new());
    let _cache = Lru::<TestColumn, _, _>::new_with_shard_amount(
        db,
        10,
        3,
        test_hasher(),
    );
}

#[tokio::test]
async fn large_capacity_single_shard() {
    let db = Arc::new(MockDatabase::new());
    db.insert::<TestColumn>(1, "one".to_string()).await;

    let cache = Lru::<TestColumn, _, _>::new_with_shard_amount(
        db.clone(),
        1_000_000,
        1,
        test_hasher(),
    );

    let result = cache.get(&1).await;
    assert_eq!(&*result.unwrap(), "one");
}

// ============================================================================
// Multi-Shard Tests
// ============================================================================

#[tokio::test]
async fn multi_shard_distribution() {
    // Test that entries are distributed across shards and caching works
    let db = Arc::new(MockDatabase::new());
    for i in 0..100 {
        db.insert::<TestColumn>(i, format!("value_{i}")).await;
    }

    // Use 4 shards with total capacity 100
    // Per-shard capacity = ceil(100/4) = 25
    // Note: Due to hash distribution, some shards may get more keys than
    // others, potentially causing evictions even when total keys <= total
    // capacity
    let cache = Lru::<TestColumn, _, _>::new_with_shard_amount(
        db.clone(),
        100,
        4,
        test_hasher(),
    );

    // Access all keys - first pass
    for i in 0..100 {
        let result = cache.get(&i).await;
        assert!(result.is_some());
    }

    let first_pass_count = db.get_count();
    // First pass should hit DB at least 100 times (once per key)
    // May be more if uneven distribution causes evictions and re-fetches
    assert!(
        first_pass_count >= 100,
        "First pass should hit DB at least 100 times"
    );

    // Now do a second pass - recently accessed keys should be cached
    // (at minimum, the most recent key in each shard should be cached)
    let count_before_second_pass = db.get_count();
    for i in 0..100 {
        let _ = cache.get(&i).await;
    }
    let second_pass_hits = db.get_count() - count_before_second_pass;

    // Second pass should have fewer DB hits than first pass
    // (unless all keys were evicted, which shouldn't happen with this capacity)
    assert!(
        second_pass_hits < 100,
        "Second pass should have some cache hits (got {second_pass_hits} DB \
         hits)"
    );
}

#[tokio::test]
async fn per_shard_eviction() {
    // Each shard has independent eviction
    // With 4 shards and capacity 4, each shard gets capacity 1
    // Per-shard capacity = 1 (exactly 1 entry per shard)
    let db = Arc::new(MockDatabase::new());
    for i in 0..10i32 {
        db.insert::<TestColumn>(i, format!("value_{i}")).await;
    }

    let cache = Lru::<TestColumn, _, _>::new_with_shard_amount(
        db.clone(),
        4,
        4,
        test_hasher(),
    );

    // Access keys - each shard can hold 1 entry (exact capacity)
    // The exact eviction behavior depends on hash distribution
    for i in 0..10 {
        let _ = cache.get(&i).await;
    }

    // All 10 keys hit the DB
    assert_eq!(db.get_count(), 10);
}

// ============================================================================
// Sequential Eviction Order Test
// ============================================================================

#[tokio::test]
async fn sequential_eviction_order() {
    // Test eviction order step by step
    let db = Arc::new(MockDatabase::new());
    for i in 1..=7 {
        db.insert::<TestColumn>(i, format!("value_{i}")).await;
    }

    // With capacity 3: eviction happens when length >= 3
    let cache = Lru::<TestColumn, _, _>::new_with_shard_amount(
        db.clone(),
        3,
        1,
        test_hasher(),
    );

    // Access keys 1, 2, 3 in order - fills capacity
    for i in 1..=3 {
        let _ = cache.get(&i).await;
    }
    assert_eq!(db.get_count(), 3);

    // Verify all 3 are cached (this also updates LRU order)
    for i in 1..=3 {
        let _ = cache.get(&i).await;
    }
    assert_eq!(db.get_count(), 3, "Keys 1-3 should all be cached");

    // Access key 4: evicts key 1 (tail)
    let _ = cache.get(&4).await;
    assert_eq!(db.get_count(), 4);

    // Key 1 should have been evicted
    let _ = cache.get(&1).await;
    assert_eq!(db.get_count(), 5, "Key 1 should have been evicted");
}

// ============================================================================
// Value Correctness Test
// ============================================================================

#[tokio::test]
async fn values_remain_correct_after_eviction_and_reload() {
    let db = Arc::new(MockDatabase::new());
    for i in 1..=10 {
        db.insert::<TestColumn>(i, format!("original_value_{i}")).await;
    }

    let cache = Lru::<TestColumn, _, _>::new_with_shard_amount(
        db.clone(),
        3, // capacity: 3 entries
        1,
        test_hasher(),
    );

    // Access and verify values
    for i in 1..=10 {
        let result = cache.get(&i).await;
        assert_eq!(&*result.unwrap(), &format!("original_value_{i}"));
    }

    // Access keys again in different order - values should still be correct
    for i in (1..=10).rev() {
        let result = cache.get(&i).await;
        assert_eq!(&*result.unwrap(), &format!("original_value_{i}"));
    }
}

// ============================================================================
// Tests for exact caching behavior
// ============================================================================

#[tokio::test]
async fn exact_capacity() {
    // Verify the exact capacity behavior
    // With capacity N, we can hold exactly N entries
    let db = Arc::new(MockDatabase::new());
    for i in 1..=5 {
        db.insert::<TestColumn>(i, format!("value_{i}")).await;
    }

    let cache = Lru::<TestColumn, _, _>::new_with_shard_amount(
        db.clone(),
        3, // capacity: 3 entries
        1,
        test_hasher(),
    );

    // Add 3 entries - fills capacity
    for i in 1..=3 {
        let _ = cache.get(&i).await;
    }
    assert_eq!(db.get_count(), 3);

    // Verify all 3 are cached
    for i in 1..=3 {
        let _ = cache.get(&i).await;
    }
    assert_eq!(db.get_count(), 3); // No new DB hits

    // Add 4th entry - should evict key 1
    let _ = cache.get(&4).await;
    assert_eq!(db.get_count(), 4);

    // Key 1 should be evicted
    let _ = cache.get(&1).await;
    assert_eq!(db.get_count(), 5); // DB hit for evicted key
}

#[tokio::test]
async fn lru_refresh_on_hit() {
    // Verify that accessing a cached item moves it to head (most recently used)
    let db = Arc::new(MockDatabase::new());
    for i in 1..=5 {
        db.insert::<TestColumn>(i, format!("value_{i}")).await;
    }

    let cache = Lru::<TestColumn, _, _>::new_with_shard_amount(
        db.clone(),
        3, // capacity: 3 entries
        1,
        test_hasher(),
    );

    // Access keys 1, 2, 3
    // Order: 3 (head) -> 2 -> 1 (tail)
    for i in 1..=3 {
        let _ = cache.get(&i).await;
    }

    // Access key 1 to refresh it
    // Order: 1 (head) -> 3 -> 2 (tail)
    let _ = cache.get(&1).await;
    assert_eq!(db.get_count(), 3); // No DB hit, just refresh

    // Add key 4 - should evict key 2 (now the tail)
    // Order: 4 (head) -> 1 -> 3 (tail)
    let _ = cache.get(&4).await;
    assert_eq!(db.get_count(), 4);

    // Key 1 should still be cached (was refreshed)
    let _ = cache.get(&1).await;
    assert_eq!(db.get_count(), 4);

    // Key 2 should be evicted
    let _ = cache.get(&2).await;
    assert_eq!(db.get_count(), 5);
}
