//! Test for the Query derive macro.

use qbice::{DeriveQuery, Query};
use qbice_serialize::{Decode, Encode};
use qbice_stable_hash::StableHash;
use qbice_stable_type_id::Identifiable;

/// A test query with multiple fields.
#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    Hash,
    StableHash,
    Identifiable,
    Encode,
    Decode,
    DeriveQuery,
)]
#[value(String)]
pub struct TestQuery {
    /// The ID field.
    pub id: u64,
    /// The name field.
    pub name: String,
}

#[test]
fn test_query_derive_compiles() {
    // This test just verifies that the derive macro works
    let query = TestQuery {
        id: 1,
        name: "test".to_string(),
    };

    // Ensure we can clone it
    let _cloned = query.clone();

    // Ensure the Query trait is implemented
    fn requires_query<Q: Query>(_q: &Q) {}
    requires_query(&query);
}

/// A simple query as a tuple struct.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    StableHash,
    Identifiable,
    Encode,
    Decode,
    DeriveQuery,
)]
#[value(Vec<u8>)]
pub struct SimpleQuery(pub u64);

#[test]
fn test_tuple_struct_query() {
    let query = SimpleQuery(42);
    let _cloned = query;

    fn requires_query<Q: Query>(_q: &Q) {}
    requires_query(&query);
}
