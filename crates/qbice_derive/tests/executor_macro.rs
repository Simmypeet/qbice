//! Test for the executor attribute macro.

use qbice::{Config, Executor, Query, TrackedEngine, executor};
use qbice_serialize::{Decode, Encode};
use qbice_stable_hash::StableHash;
use qbice_stable_type_id::Identifiable;

/// A simple test query.
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
    Query,
)]
#[value(u64)]
pub struct AddQuery {
    /// First number.
    pub a: u32,
    /// Second number.
    pub b: u32,
}

/// Another test query.
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
    Query,
)]
#[value(String)]
pub struct GreetQuery {
    /// The name.
    pub name: String,
}

// Test with generic C: Config
#[executor]
#[allow(clippy::unused_async)]
async fn add_executor<C: qbice::config::Config>(
    query: &AddQuery,
    _engine: &TrackedEngine<C>,
) -> u64 {
    u64::from(query.a) + u64::from(query.b)
}

#[executor]
#[allow(clippy::unused_async)]
async fn greet_executor<C: Config>(
    query: &GreetQuery,
    _engine: &TrackedEngine<C>,
) -> String {
    format!("Hello, {}!", query.name)
}

/// Query for testing explicit config type.
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
    Query,
)]
#[value(u64)]
pub struct MultiplyQuery {
    /// First number.
    pub x: u32,
    /// Second number.
    pub y: u32,
}

// Test with explicit config type
#[executor(config = qbice::DefaultConfig)]
#[allow(clippy::unused_async)]
async fn multiply_executor(
    query: &MultiplyQuery,
    _engine: &TrackedEngine<qbice::DefaultConfig>,
) -> u64 {
    u64::from(query.x) * u64::from(query.y)
}

#[test]
fn test_executor_trait_implementation() {
    // This test just verifies compilation - the executor trait is implemented
    fn requires_executor<Q: qbice::Query, C: Config, E: Executor<Q, C>>(_: E) {}

    requires_executor::<AddQuery, qbice::DefaultConfig, _>(AddExecutor);
    requires_executor::<GreetQuery, qbice::DefaultConfig, _>(GreetExecutor);
    requires_executor::<MultiplyQuery, qbice::DefaultConfig, _>(
        MultiplyExecutor,
    );
}
