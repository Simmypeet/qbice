//! Test for the Query extend attribute.

use qbice::{Config, TrackedEngine};
use qbice_serialize::{Decode, Encode};
use qbice_stable_hash::StableHash;
use qbice_stable_type_id::Identifiable;

/// Query with extension trait by reference.
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
    qbice::Query,
)]
#[value(u64)]
#[extend(name = get_sum)]
pub struct SumQuery {
    /// First number.
    pub a: u32,
    /// Second number.
    pub b: u32,
}

/// Query with extension trait by value.
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
    qbice::Query,
)]
#[value(String)]
#[extend(name = format_number, by_val)]
pub struct FormatQuery {
    /// The number to format.
    pub num: u32,
}

#[test]
fn test_extend_trait_by_ref_generated() {
    // Verify the extension trait was generated
    fn requires_trait<T: get_sum>(_: &T) {}

    // Create a mock engine type that implements the trait
    // (In real usage, TrackedEngine<C> implements it)
    struct MockEngine;
    impl get_sum for MockEngine {
        async fn get_sum(&self, _q: &SumQuery) -> u64 { 42 }
    }

    let engine = MockEngine;
    requires_trait(&engine);
}

#[test]
fn test_extend_trait_by_val_generated() {
    // Verify the extension trait was generated
    fn requires_trait<T: format_number>(_: &T) {}

    // Create a mock engine type that implements the trait
    struct MockEngine;
    impl format_number for MockEngine {
        async fn format_number(&self, _q: FormatQuery) -> String {
            String::from("42")
        }
    }

    let engine = MockEngine;
    requires_trait(&engine);
}

#[test]
fn test_tracked_engine_implements_extension_trait() {
    // This test verifies that TrackedEngine implements the extension traits
    #[allow(dead_code)]
    fn requires_get_sum<T: get_sum>(_: &T) {}
    #[allow(dead_code)]
    fn requires_format_number<T: format_number>(_: &T) {}

    // Note: We can't actually create a TrackedEngine here without a full setup,
    // but we can verify the trait bounds compile
    #[allow(dead_code)]
    fn verify_impls<C: Config>(engine: &TrackedEngine<C>) {
        requires_get_sum(engine);
        requires_format_number(engine);
    }
}
