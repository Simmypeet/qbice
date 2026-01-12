//! Test for the derive_for_query_id attribute macro.

use std::collections::HashSet;

use qbice::Query;
use qbice_derive::derive_for_query_id;

/// Test query using derive_for_query_id.
#[derive_for_query_id]
#[value(String)]
pub struct SimpleQuery {
    /// The ID field.
    pub id: u64,
}

/// Test query with extend attribute.
#[derive_for_query_id(no_copy)]
#[value(Vec<u8>)]
#[extend(name = get_data)]
pub struct ExtendQuery {
    /// The key.
    pub key: String,
}

const fn requires_query<Q: Query>(_q: &Q) {}

#[test]
fn test_derive_for_query_id_basic() {
    // Verify all traits are implemented
    let query = SimpleQuery { id: 42 };

    // Debug
    let _ = format!("{query:?}");

    // PartialEq, Eq
    assert_eq!(query, query);

    // Hash (just verify it compiles)
    let mut set = HashSet::new();
    set.insert(query);
    assert!(set.contains(&query));

    // Query trait
    requires_query(&query);
}

const fn requires_get_data<T: get_data>(_: &T) {}

struct MockEngine;
impl get_data for MockEngine {
    async fn get_data(&self, _q: &ExtendQuery) -> Vec<u8> { vec![1, 2, 3] }
}

#[test]
fn test_derive_for_query_id_with_extend() {
    // Verify the extend attribute works with derive_for_query_id(no_copy)
    let query = ExtendQuery { key: "test".to_string() };

    // Verify Query trait is implemented
    requires_query(&query);

    // Verify the extension trait is generated
    let engine = MockEngine;
    requires_get_data(&engine);
}

/// Tuple struct with derive_for_query_id.
#[derive_for_query_id]
#[value(i32)]
pub struct TupleQuery(pub u64);

#[test]
fn test_derive_for_query_id_tuple_struct() {
    let query = TupleQuery(100);

    // Query trait
    requires_query(&query);
}
