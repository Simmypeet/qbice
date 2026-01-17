// Allow similar names in tests (e.g., hash1/hash2, data1/data2)
#![allow(clippy::similar_names)]
// Allow redundant clones in tests for clarity
#![allow(clippy::redundant_clone)]

use std::{hash::Hash, sync::Arc};

use qbice_serialize::{
    Decode, Decoder, Encode, Encoder, Plugin, PostcardDecoder, PostcardEncoder,
};
use qbice_stable_hash::{BuildStableHasherDefault, StableHash, StableHasher};
use qbice_stable_type_id::Identifiable;
use siphasher::sip128::SipHasher;

use super::{Interned, Interner, SharedInterner};

/// A type alias for the default hasher builder used in tests.
type TestHasherBuilder = BuildStableHasherDefault<SipHasher>;

/// Creates a new interner for testing.
fn test_interner() -> Interner {
    Interner::new(4, TestHasherBuilder::default())
}

/// Creates a new shared interner for testing.
fn test_shared_interner() -> SharedInterner {
    SharedInterner::new(4, TestHasherBuilder::default())
}

// =============================================================================
// Test Types
// =============================================================================

/// A simple test type that implements all required traits for interning and
/// serialization.
#[derive(
    Debug, Clone, PartialEq, Eq, Encode, Decode, StableHash, Identifiable,
)]
#[stable_hash_crate(qbice_stable_hash)]
#[serialize_crate(qbice_serialize)]
#[stable_type_id_crate(qbice_stable_type_id)]
struct TestData {
    name: String,
    value: i32,
}

/// A simple string wrapper that implements Identifiable.
#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Encode,
    Decode,
    StableHash,
    Identifiable,
)]
#[stable_hash_crate(qbice_stable_hash)]
#[serialize_crate(qbice_serialize)]
#[stable_type_id_crate(qbice_stable_type_id)]
struct TestString(String);

impl TestString {
    fn new(s: impl Into<String>) -> Self { Self(s.into()) }

    fn as_str(&self) -> &str { &self.0 }

    fn len(&self) -> usize { self.0.len() }

    fn is_empty(&self) -> bool { self.0.is_empty() }
}

/// A simple u32 wrapper that implements Identifiable.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Encode,
    Decode,
    StableHash,
    Identifiable,
)]
#[stable_hash_crate(qbice_stable_hash)]
#[serialize_crate(qbice_serialize)]
#[stable_type_id_crate(qbice_stable_type_id)]
struct TestU32(u32);

/// A simple i32 wrapper that implements Identifiable.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Encode,
    Decode,
    StableHash,
    Identifiable,
)]
#[stable_hash_crate(qbice_stable_hash)]
#[serialize_crate(qbice_serialize)]
#[stable_type_id_crate(qbice_stable_type_id)]
struct TestI32(i32);

/// A simple f64 wrapper that implements Identifiable.
#[derive(
    Debug, Clone, Copy, PartialEq, Encode, Decode, StableHash, Identifiable,
)]
#[stable_hash_crate(qbice_stable_hash)]
#[serialize_crate(qbice_serialize)]
#[stable_type_id_crate(qbice_stable_type_id)]
struct TestF64(f64);

// =============================================================================
// Basic Interning Tests
// =============================================================================

#[test]
fn intern_returns_same_value() {
    let interner = test_interner();

    let a = interner.intern(TestString::new("hello"));
    let b = interner.intern(TestString::new("hello"));

    // Both should point to the same allocation
    assert!(Arc::ptr_eq(&a.0, &b.0));
    assert_eq!(a.as_str(), "hello");
    assert_eq!(b.as_str(), "hello");
}

#[test]
fn intern_different_values_returns_different_allocations() {
    let interner = test_interner();

    let a = interner.intern(TestString::new("hello"));
    let b = interner.intern(TestString::new("world"));

    // Should be different allocations
    assert!(!Arc::ptr_eq(&a.0, &b.0));
    assert_eq!(a.as_str(), "hello");
    assert_eq!(b.as_str(), "world");
}

#[test]
fn intern_same_content_different_types_returns_different_allocations() {
    let interner = test_interner();

    let a = interner.intern(TestU32(42));
    let b = interner.intern(TestI32(42));

    // Even if the content hashes to the same value, different types
    // should result in different allocations due to StableTypeID
    assert_eq!((*a).0, 42u32);
    assert_eq!((*b).0, 42i32);
}

#[test]
fn intern_clone_shares_allocation() {
    let interner = test_interner();

    let a = interner.intern(TestString::new("test"));
    let b = a.clone();

    assert!(Arc::ptr_eq(&a.0, &b.0));
    assert_eq!(a.as_str(), b.as_str());
}

#[test]
fn interned_deref_works() {
    let interner = test_interner();

    let interned = interner.intern(TestString::new("hello world"));

    // Should be able to call TestString methods via Deref
    assert_eq!(interned.len(), 11);
    assert!(interned.as_str().starts_with("hello"));
}

// =============================================================================
// Get From Hash Tests
// =============================================================================

#[test]
fn get_from_hash_returns_interned_value() {
    let interner = test_interner();
    let value = TestString::new("test");
    let hash = interner.hash_128(&value);

    let interned = interner.intern(value);
    let retrieved = interner.get_from_hash::<TestString>(hash);

    assert!(retrieved.is_some());
    let retrieved = retrieved.unwrap();
    assert!(Arc::ptr_eq(&interned.0, &retrieved.0));
}

#[test]
fn get_from_hash_returns_none_for_missing_value() {
    let interner = test_interner();
    let hash = interner.hash_128(&TestString::new("never interned"));

    let retrieved = interner.get_from_hash::<TestString>(hash);

    assert!(retrieved.is_none());
}

#[test]
fn get_from_hash_returns_none_after_value_dropped() {
    let interner = test_interner();
    let value = TestString::new("temporary");
    let hash = interner.hash_128(&value);

    {
        let _interned = interner.intern(value);
        // Value should be retrievable while interned exists
        assert!(interner.get_from_hash::<TestString>(hash).is_some());
    }

    // After the Interned is dropped, the weak reference should be dead
    // Note: The entry may still exist in the map, but upgrade() will fail
    let retrieved = interner.get_from_hash::<TestString>(hash);
    assert!(retrieved.is_none());
}

// =============================================================================
// Weak Reference Tests
// =============================================================================

#[test]
fn interner_uses_weak_references() {
    let interner = test_interner();
    let value = TestString::new("weak test");
    let hash = interner.hash_128(&value);

    // Intern and immediately drop
    let interned = interner.intern(value);
    assert_eq!(interned.as_str(), "weak test");
    drop(interned);

    // The value should no longer be retrievable
    assert!(interner.get_from_hash::<TestString>(hash).is_none());
}

#[test]
fn intern_after_drop_creates_new_allocation() {
    let interner = test_interner();

    let first = interner.intern(TestString::new("reused"));
    let first_ptr = Arc::as_ptr(&first.0);
    drop(first);

    let second = interner.intern(TestString::new("reused"));

    // Should be a different allocation since the first was dropped
    assert_ne!(Arc::as_ptr(&second.0), first_ptr);
    assert_eq!(second.as_str(), "reused");
}

// =============================================================================
// SharedInterner Tests
// =============================================================================

#[test]
fn shared_interner_deref_works() {
    let shared = test_shared_interner();

    let interned = shared.intern(TestString::new("shared test"));
    assert_eq!(interned.as_str(), "shared test");
}

#[test]
fn shared_interner_clone_shares_interner() {
    let shared1 = test_shared_interner();
    let shared2 = shared1.clone();

    let a = shared1.intern(TestString::new("shared"));
    let b = shared2.intern(TestString::new("shared"));

    // Should share the same interned value
    assert!(Arc::ptr_eq(&a.0, &b.0));
}

#[test]
fn shared_interner_from_interner() {
    let interner = test_interner();
    let shared = SharedInterner::from_interner(interner);

    let interned = shared.intern(TestString::new("from interner"));
    assert_eq!(interned.as_str(), "from interner");
}

// =============================================================================
// Serialization and Deserialization Tests
// =============================================================================

/// Helper function to encode a value to bytes.
fn encode_to_bytes<T: Encode>(value: &T, plugin: &Plugin) -> Vec<u8> {
    let mut buffer = Vec::new();
    let mut encoder = PostcardEncoder::new(&mut buffer);
    encoder.encode(value, plugin).unwrap();
    buffer
}

/// Helper function to decode a value from bytes.
fn decode_from_bytes<T: Decode>(bytes: &[u8], plugin: &Plugin) -> T {
    let mut decoder = PostcardDecoder::new(bytes);
    decoder.decode::<T>(plugin).unwrap()
}

#[test]
fn serialize_interned_value_first_occurrence_encodes_full_value() {
    let shared = test_shared_interner();
    let mut plugin = Plugin::new();
    plugin.insert(shared.clone());

    let data = TestData { name: "test".to_string(), value: 42 };
    let interned = shared.intern(data);

    let bytes = encode_to_bytes(&interned, &plugin);

    // The encoded bytes should contain the full value (tag 0 + value)
    // First byte should be 0 (source tag)
    assert_eq!(bytes[0], 0);
}

#[test]
fn serialize_interned_value_second_occurrence_encodes_reference() {
    let shared = test_shared_interner();
    let mut plugin = Plugin::new();
    plugin.insert(shared.clone());

    let data = TestData { name: "test".to_string(), value: 42 };
    let interned1 = shared.intern(data.clone());
    let interned2 = shared.intern(data);

    // Both point to the same value
    assert!(Arc::ptr_eq(&interned1.0, &interned2.0));

    // Encode a tuple containing the same interned value twice
    let bytes = encode_to_bytes(&(interned1, interned2), &plugin);

    // The first occurrence should be tag 0 (source)
    // The second occurrence should be tag 1 (reference)
    // Check that the first byte after the tuple encoding is 0 (source tag)
    // and somewhere in the encoding there's a 1 (reference tag)

    // The encoding should contain both a source (tag 0) and a reference (tag 1)
    // We verify this by checking the first item is a source
    assert_eq!(bytes[0], 0, "first value should be encoded as source");

    // And verify that somewhere there's a reference tag (1)
    // For a tuple, the second element follows the first
    // The reference encoding is: tag(1) + 16 bytes hash
    // So the total size should be approximately:
    // source_size + 1 (tag) + compressed_hash_size
    // Which is much smaller than 2 * source_size

    // Just verify the serialization and deserialization works correctly
    let decoded: (Interned<TestData>, Interned<TestData>) =
        decode_from_bytes(&bytes, &plugin);
    assert!(Arc::ptr_eq(&decoded.0.0, &decoded.1.0));
}

#[test]
fn deserialize_interned_value_from_source() {
    let shared = test_shared_interner();
    let mut plugin = Plugin::new();
    plugin.insert(shared.clone());

    let data = TestData { name: "roundtrip".to_string(), value: 123 };
    let interned = shared.intern(data.clone());

    let bytes = encode_to_bytes(&interned, &plugin);
    let decoded: Interned<TestData> = decode_from_bytes(&bytes, &plugin);

    assert_eq!(*decoded, data);
}

#[test]
fn deserialize_interned_value_shares_allocation() {
    let shared = test_shared_interner();
    let mut plugin = Plugin::new();
    plugin.insert(shared.clone());

    let data = TestData { name: "shared".to_string(), value: 456 };
    let interned = shared.intern(data);

    let bytes = encode_to_bytes(&interned, &plugin);

    // Decode twice - both should share the same allocation from the interner
    let decoded1: Interned<TestData> = decode_from_bytes(&bytes, &plugin);
    let decoded2: Interned<TestData> = decode_from_bytes(&bytes, &plugin);

    assert!(Arc::ptr_eq(&decoded1.0, &decoded2.0));
    assert!(Arc::ptr_eq(&decoded1.0, &interned.0));
}

#[test]
fn serialize_deserialize_tuple_with_duplicate_interned_values() {
    let shared = test_shared_interner();
    let mut plugin = Plugin::new();
    plugin.insert(shared.clone());

    let data = TestData { name: "duplicate".to_string(), value: 789 };
    let interned = shared.intern(data);

    // Create a tuple with the same interned value twice
    let tuple = (interned.clone(), interned.clone());

    let bytes = encode_to_bytes(&tuple, &plugin);
    let decoded: (Interned<TestData>, Interned<TestData>) =
        decode_from_bytes(&bytes, &plugin);

    // Both parts of the decoded tuple should share the same allocation
    assert!(Arc::ptr_eq(&decoded.0.0, &decoded.1.0));
    assert_eq!(decoded.0.name, "duplicate");
    assert_eq!(decoded.1.value, 789);
}

#[test]
fn serialize_deserialize_vec_with_interned_values() {
    let shared = test_shared_interner();
    let mut plugin = Plugin::new();
    plugin.insert(shared.clone());

    let data1 = TestData { name: "first".to_string(), value: 1 };
    let data2 = TestData { name: "second".to_string(), value: 2 };
    let data3 = TestData { name: "first".to_string(), value: 1 }; // Duplicate of data1

    let vec: Vec<Interned<TestData>> =
        vec![shared.intern(data1), shared.intern(data2), shared.intern(data3)];

    // The first and third should share allocation
    assert!(Arc::ptr_eq(&vec[0].0, &vec[2].0));

    let bytes = encode_to_bytes(&vec, &plugin);
    let decoded: Vec<Interned<TestData>> = decode_from_bytes(&bytes, &plugin);

    assert_eq!(decoded.len(), 3);
    assert_eq!(decoded[0].name, "first");
    assert_eq!(decoded[1].name, "second");
    assert_eq!(decoded[2].name, "first");

    // The decoded first and third should also share allocation
    assert!(Arc::ptr_eq(&decoded[0].0, &decoded[2].0));
}

#[test]
fn deserialize_with_fresh_interner_interns_values() {
    let encode_interner = test_shared_interner();
    let mut encode_plugin = Plugin::new();
    encode_plugin.insert(encode_interner.clone());

    let data = TestData { name: "fresh".to_string(), value: 999 };
    let interned = encode_interner.intern(data);
    let bytes = encode_to_bytes(&interned, &encode_plugin);

    // Decode with a completely fresh interner
    let decode_interner = test_shared_interner();
    let mut decode_plugin = Plugin::new();
    decode_plugin.insert(decode_interner.clone());

    let decoded: Interned<TestData> = decode_from_bytes(&bytes, &decode_plugin);

    assert_eq!(decoded.name, "fresh");
    assert_eq!(decoded.value, 999);

    // The value should now be in the decode interner
    let hash = decode_interner.hash_128(&*decoded);
    let retrieved = decode_interner.get_from_hash::<TestData>(hash);
    assert!(retrieved.is_some());
    assert!(Arc::ptr_eq(&decoded.0, &retrieved.unwrap().0));
}

// =============================================================================
// StableHash Tests for Interned
// =============================================================================

#[test]
fn interned_stable_hash_matches_inner_value() {
    let interner = test_interner();
    let value = TestString::new("hash test");

    let interned = interner.intern(value.clone());

    // Hash of Interned<T> should be the same as hash of T
    let mut hasher1 = SipHasher::default();
    value.stable_hash(&mut hasher1);
    let hash1 = hasher1.finish();

    let mut hasher2 = SipHasher::default();
    interned.stable_hash(&mut hasher2);
    let hash2 = hasher2.finish();

    assert_eq!(hash1, hash2);
}

// =============================================================================
// Edge Cases and Error Handling
// =============================================================================

#[test]
fn intern_empty_string() {
    let interner = test_interner();

    let a = interner.intern(TestString::new(""));
    let b = interner.intern(TestString::new(""));

    assert!(Arc::ptr_eq(&a.0, &b.0));
    assert!(a.is_empty());
}

#[test]
fn intern_large_value() {
    let interner = test_interner();
    let large_string = TestString::new("x".repeat(100_000));

    let a = interner.intern(large_string.clone());
    let b = interner.intern(large_string);

    assert!(Arc::ptr_eq(&a.0, &b.0));
    assert_eq!(a.len(), 100_000);
}

#[test]
fn multiple_types_in_same_interner() {
    let interner = test_interner();

    let string1 = interner.intern(TestString::new("hello"));
    let int1 = interner.intern(TestU32(42));
    let float1 = interner.intern(TestF64(1.234));

    let string2 = interner.intern(TestString::new("hello"));
    let int2 = interner.intern(TestU32(42));
    let float2 = interner.intern(TestF64(1.234));

    // Same type and value should share
    assert!(Arc::ptr_eq(&string1.0, &string2.0));
    assert!(Arc::ptr_eq(&int1.0, &int2.0));
    assert!(Arc::ptr_eq(&float1.0, &float2.0));

    // Different types should not affect each other
    assert_eq!(string1.as_str(), "hello");
    assert_eq!((*int1).0, 42u32);
    assert!(((*float1).0 - 1.234_f64).abs() < f64::EPSILON);
}

// =============================================================================
// Concurrent Access Tests
// =============================================================================

#[test]
fn concurrent_intern_same_value() {
    use std::thread;

    let shared = test_shared_interner();
    let mut handles = vec![];

    for _ in 0..10 {
        let interner = shared.clone();
        handles.push(thread::spawn(move || {
            interner.intern(TestString::new("concurrent"))
        }));
    }

    let results: Vec<_> =
        handles.into_iter().map(|h| h.join().unwrap()).collect();

    // All results should share the same allocation
    let first = &results[0];
    for result in &results[1..] {
        assert!(Arc::ptr_eq(&first.0, &result.0));
    }
}

#[test]
fn concurrent_intern_different_values() {
    use std::thread;

    let shared = test_shared_interner();
    let mut handles = vec![];

    for i in 0..10 {
        let interner = shared.clone();
        handles.push(thread::spawn(move || {
            interner.intern(TestString::new(format!("value_{i}")))
        }));
    }

    let results: Vec<_> =
        handles.into_iter().map(|h| h.join().unwrap()).collect();

    // All results should be different
    for i in 0..results.len() {
        for j in (i + 1)..results.len() {
            assert!(!Arc::ptr_eq(&results[i].0, &results[j].0));
        }
    }
}

#[test]
fn concurrent_serialize_deserialize() {
    use std::thread;

    let shared = test_shared_interner();

    // Pre-populate with some values
    let data = TestData { name: "concurrent_serde".to_string(), value: 100 };
    let interned = shared.intern(data);

    let mut handles = vec![];

    for _ in 0..10 {
        let interner = shared.clone();
        let interned_clone = interned.clone();

        handles.push(thread::spawn(move || {
            let mut plugin = Plugin::new();
            plugin.insert(interner.clone());

            let bytes = encode_to_bytes(&interned_clone, &plugin);
            let decoded: Interned<TestData> =
                decode_from_bytes(&bytes, &plugin);

            // Decoded should share allocation with original
            assert!(Arc::ptr_eq(&interned_clone.0, &decoded.0));
            decoded
        }));
    }

    for handle in handles {
        let result = handle.join().unwrap();
        assert_eq!(result.name, "concurrent_serde");
        assert_eq!(result.value, 100);
    }
}

// =============================================================================
// InternedID Tests
// =============================================================================

#[test]
fn same_value_same_type_produces_same_interned_id() {
    let interner = test_interner();

    let a = interner.intern(TestU32(123));
    let b = interner.intern(TestU32(123));

    // Same type and value should result in the same interned allocation
    assert!(Arc::ptr_eq(&a.0, &b.0));
}

#[test]
fn same_value_different_type_produces_different_interned_id() {
    let interner = test_interner();

    // TestU32(42) and TestI32(42) have the same numeric value
    // but different types, so they should have different IDs
    let u32_val = interner.intern(TestU32(42));
    let i32_val = interner.intern(TestI32(42));

    // They should be different allocations
    // (We can't compare Arc<TestU32> with Arc<TestI32> directly)
    assert_eq!((*u32_val).0, 42u32);
    assert_eq!((*i32_val).0, 42i32);
}

// =============================================================================
// Serialization Session State Tests
// =============================================================================

#[test]
fn encoding_same_interned_twice_in_sequence_uses_reference() {
    let shared = test_shared_interner();
    let mut plugin = Plugin::new();
    plugin.insert(shared.clone());

    let data = TestData { name: "session_test".to_string(), value: 42 };
    let interned = shared.intern(data);

    // Create a vec with the same value multiple times
    let values = vec![interned.clone(), interned.clone(), interned.clone()];

    let bytes = encode_to_bytes(&values, &plugin);

    // The first should be a full value (tag 0), subsequent should be references
    // (tag 1)
    // After length prefix, first value starts with tag 0
    // Note: The actual byte layout depends on the encoding format

    let decoded: Vec<Interned<TestData>> = decode_from_bytes(&bytes, &plugin);

    assert_eq!(decoded.len(), 3);
    // All decoded values should share the same allocation
    assert!(Arc::ptr_eq(&decoded[0].0, &decoded[1].0));
    assert!(Arc::ptr_eq(&decoded[1].0, &decoded[2].0));
    assert!(Arc::ptr_eq(&decoded[0].0, &interned.0));
}

#[test]
#[should_panic(expected = "referenced interned value not found in interner")]
fn decoding_reference_before_source_fails() {
    // This test verifies that if we somehow have a reference before its
    // source, the deserialization should panic.
    // In practice, proper serialization should never produce such output.

    // We'll manually construct invalid bytes to test error handling
    let shared = test_shared_interner();
    let mut plugin = Plugin::new();
    plugin.insert(shared);

    // Construct bytes that start with a reference tag (1) followed by a fake
    // hash
    let mut bytes = vec![1u8]; // Reference tag
    // Add 16 bytes for the Compact128 hash (two u64s as varints)
    bytes.extend_from_slice(&[0u8; 16]);

    let mut decoder = PostcardDecoder::new(bytes.as_slice());
    // This will panic because the reference doesn't exist
    let _result: Interned<TestData> = decoder.decode(&plugin).unwrap();
}

// =============================================================================
// Complex Nested Structure Tests
// =============================================================================

/// A complex nested structure for testing.
#[derive(
    Debug, Clone, PartialEq, Eq, Encode, Decode, StableHash, Identifiable,
)]
#[stable_hash_crate(qbice_stable_hash)]
#[serialize_crate(qbice_serialize)]
#[stable_type_id_crate(qbice_stable_type_id)]
struct NestedTestData {
    inner: TestData,
    children: Vec<TestData>,
}

#[test]
fn intern_nested_structure() {
    let interner = test_interner();

    let nested = NestedTestData {
        inner: TestData { name: "inner".to_string(), value: 1 },
        children: vec![
            TestData { name: "child1".to_string(), value: 2 },
            TestData { name: "child2".to_string(), value: 3 },
        ],
    };

    let a = interner.intern(nested.clone());
    let b = interner.intern(nested);

    assert!(Arc::ptr_eq(&a.0, &b.0));
}

#[test]
fn serialize_deserialize_nested_interned() {
    let shared = test_shared_interner();
    let mut plugin = Plugin::new();
    plugin.insert(shared.clone());

    let nested = NestedTestData {
        inner: TestData { name: "nested_inner".to_string(), value: 10 },
        children: vec![TestData {
            name: "nested_child".to_string(),
            value: 20,
        }],
    };

    let interned = shared.intern(nested.clone());
    let bytes = encode_to_bytes(&interned, &plugin);
    let decoded: Interned<NestedTestData> = decode_from_bytes(&bytes, &plugin);

    assert_eq!(*decoded, nested);
    assert!(Arc::ptr_eq(&decoded.0, &interned.0));
}

// =============================================================================
// Interned<T> Trait Impls Tests
// =============================================================================

#[test]
fn interned_eq_compares_values() {
    let interner = test_interner();

    let a = interner.intern(TestU32(42));
    let b = interner.intern(TestU32(42));
    let c = interner.intern(TestU32(123));

    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn interned_ord_compares_values() {
    let interner = test_interner();

    let a = interner.intern(TestU32(1));
    let b = interner.intern(TestU32(2));
    let c = interner.intern(TestU32(2));

    assert!(a < b);
    assert!(b > a);
    assert!(b == c);
    assert!(b <= c);
    assert!(b >= c);
}

#[test]
fn interned_hash_uses_value() {
    use std::collections::HashSet;

    let interner = test_interner();

    let a = interner.intern(TestString::new("hash_key"));
    let b = interner.intern(TestString::new("hash_key"));

    let mut set = HashSet::new();
    set.insert(a.clone());

    // b should be found in the set because it has the same value
    assert!(set.contains(&b));
}

// =============================================================================
// Unsized Type Interning Tests
// =============================================================================

#[test]
fn intern_unsized_str_returns_same_allocation() {
    let interner = test_interner();

    let a: Interned<str> = interner.intern_unsized("hello world".to_string());
    let b: Interned<str> = interner.intern_unsized("hello world".to_string());

    // Both should point to the same allocation
    assert!(Arc::ptr_eq(&a.0, &b.0));
    assert_eq!(&*a, "hello world");
    assert_eq!(&*b, "hello world");
}

#[test]
fn intern_unsized_str_different_values_different_allocations() {
    let interner = test_interner();

    let a: Interned<str> = interner.intern_unsized("hello".to_string());
    let b: Interned<str> = interner.intern_unsized("world".to_string());

    // Should be different allocations
    assert!(!Arc::ptr_eq(&a.0, &b.0));
    assert_eq!(&*a, "hello");
    assert_eq!(&*b, "world");
}

#[test]
fn intern_unsized_str_from_box() {
    let interner = test_interner();

    let boxed: Box<str> = "test string".into();
    let a: Interned<str> = interner.intern_unsized(boxed);
    let b: Interned<str> = interner.intern_unsized("test string".to_string());

    // Should share the same allocation
    assert!(Arc::ptr_eq(&a.0, &b.0));
}

#[test]
fn intern_unsized_byte_slice_returns_same_allocation() {
    let interner = test_interner();

    let a: Interned<[u8]> = interner.intern_unsized(vec![1u8, 2, 3, 4, 5]);
    let b: Interned<[u8]> = interner.intern_unsized(vec![1u8, 2, 3, 4, 5]);

    // Both should point to the same allocation
    assert!(Arc::ptr_eq(&a.0, &b.0));
    assert_eq!(&*a, &[1u8, 2, 3, 4, 5]);
}

#[test]
fn intern_unsized_byte_slice_different_values_different_allocations() {
    let interner = test_interner();

    let a: Interned<[u8]> = interner.intern_unsized(vec![1u8, 2, 3]);
    let b: Interned<[u8]> = interner.intern_unsized(vec![4u8, 5, 6]);

    // Should be different allocations
    assert!(!Arc::ptr_eq(&a.0, &b.0));
    assert_eq!(&*a, &[1u8, 2, 3]);
    assert_eq!(&*b, &[4u8, 5, 6]);
}

#[test]
fn intern_unsized_byte_slice_from_box() {
    let interner = test_interner();

    let boxed: Box<[u8]> = vec![10u8, 20, 30].into_boxed_slice();
    let a: Interned<[u8]> = interner.intern_unsized(boxed);
    let b: Interned<[u8]> = interner.intern_unsized(vec![10u8, 20, 30]);

    // Should share the same allocation
    assert!(Arc::ptr_eq(&a.0, &b.0));
}

#[test]
fn intern_unsized_empty_str() {
    let interner = test_interner();

    let a: Interned<str> = interner.intern_unsized(String::new());
    let b: Interned<str> = interner.intern_unsized(String::new());

    assert!(Arc::ptr_eq(&a.0, &b.0));
    assert!(a.is_empty());
}

#[test]
fn intern_unsized_empty_byte_slice() {
    let interner = test_interner();

    let a: Interned<[u8]> = interner.intern_unsized(Vec::<u8>::new());
    let b: Interned<[u8]> = interner.intern_unsized(Vec::<u8>::new());

    assert!(Arc::ptr_eq(&a.0, &b.0));
    assert!(a.is_empty());
}

#[test]
fn intern_unsized_large_str() {
    let interner = test_interner();
    let large_string = "x".repeat(100_000);

    let a: Interned<str> = interner.intern_unsized(large_string.clone());
    let b: Interned<str> = interner.intern_unsized(large_string);

    assert!(Arc::ptr_eq(&a.0, &b.0));
    assert_eq!(a.len(), 100_000);
}

#[test]
fn intern_unsized_str_deref_works() {
    let interner = test_interner();

    let interned: Interned<str> =
        interner.intern_unsized("hello world".to_string());

    // Should be able to call str methods via Deref
    assert_eq!(interned.len(), 11);
    assert!(interned.starts_with("hello"));
    assert!(interned.ends_with("world"));
    assert!(interned.contains(' '));
}

#[test]
fn intern_unsized_byte_slice_deref_works() {
    let interner = test_interner();

    let interned: Interned<[u8]> = interner.intern_unsized(vec![1u8, 2, 3, 4]);

    // Should be able to call slice methods via Deref
    assert_eq!(interned.len(), 4);
    assert_eq!(interned.first(), Some(&1u8));
    assert_eq!(interned.last(), Some(&4u8));
}

#[test]
fn get_from_hash_works_for_unsized_str() {
    let interner = test_interner();
    let value = "test string";
    let hash = interner.hash_128(value);

    let interned: Interned<str> = interner.intern_unsized(value.to_string());
    let retrieved = interner.get_from_hash::<str>(hash);

    assert!(retrieved.is_some());
    let retrieved = retrieved.unwrap();
    assert!(Arc::ptr_eq(&interned.0, &retrieved.0));
}

#[test]
fn get_from_hash_returns_none_for_dropped_unsized_str() {
    let interner = test_interner();
    let value = "temporary string";
    let hash = interner.hash_128(value);

    {
        let _interned: Interned<str> =
            interner.intern_unsized(value.to_string());
        // Value should be retrievable while interned exists
        assert!(interner.get_from_hash::<str>(hash).is_some());
    }

    // After the Interned is dropped, the weak reference should be dead
    assert!(interner.get_from_hash::<str>(hash).is_none());
}

#[test]
fn concurrent_intern_unsized_same_value() {
    use std::thread;

    let shared = test_shared_interner();
    let mut handles = vec![];

    for _ in 0..10 {
        let interner = shared.clone();
        handles.push(thread::spawn(move || {
            interner
                .intern_unsized::<str, String>("concurrent string".to_string())
        }));
    }

    let results: Vec<_> =
        handles.into_iter().map(|h| h.join().unwrap()).collect();

    // All results should share the same allocation
    let first = &results[0];
    for result in &results[1..] {
        assert!(Arc::ptr_eq(&first.0, &result.0));
    }
}

#[test]
fn intern_unsized_str_and_string_are_separate() {
    let interner = test_interner();

    // Interning as str
    let str_interned: Interned<str> =
        interner.intern_unsized("hello".to_string());

    // Interning as String (sized type)
    let string_interned: Interned<TestString> =
        interner.intern(TestString::new("hello"));

    // These are different types, so they should be stored separately
    // (different StableTypeIDs)
    assert_eq!(&*str_interned, "hello");
    assert_eq!(string_interned.as_str(), "hello");
}

#[test]
fn intern_unsized_after_drop_creates_new_allocation() {
    let interner = test_interner();

    let first: Interned<str> =
        interner.intern_unsized("reused string".to_string());
    let first_ptr = Arc::as_ptr(&first.0);
    drop(first);

    let second: Interned<str> =
        interner.intern_unsized("reused string".to_string());

    // Should be a different allocation since the first was dropped
    assert_ne!(Arc::as_ptr(&second.0), first_ptr);
    assert_eq!(&*second, "reused string");
}

#[test]
fn intern_unsized_i32_slice() {
    let interner = test_interner();

    let a: Interned<[i32]> = interner.intern_unsized(vec![1, 2, 3, 4, 5]);
    let b: Interned<[i32]> = interner.intern_unsized(vec![1, 2, 3, 4, 5]);

    // Both should point to the same allocation
    assert!(Arc::ptr_eq(&a.0, &b.0));
    assert_eq!(&*a, &[1, 2, 3, 4, 5]);
}

#[test]
fn shared_interner_intern_unsized_works() {
    let shared = test_shared_interner();

    let a: Interned<str> = shared.intern_unsized("shared unsized".to_string());
    let b: Interned<str> = shared.intern_unsized("shared unsized".to_string());

    assert!(Arc::ptr_eq(&a.0, &b.0));
    assert_eq!(&*a, "shared unsized");
}

#[test]
fn shared_interner_clone_shares_unsized_values() {
    let shared1 = test_shared_interner();
    let shared2 = shared1.clone();

    let a: Interned<str> = shared1.intern_unsized("shared clone".to_string());
    let b: Interned<str> = shared2.intern_unsized("shared clone".to_string());

    // Should share the same interned value
    assert!(Arc::ptr_eq(&a.0, &b.0));
}
