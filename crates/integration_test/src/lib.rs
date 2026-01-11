//! Common test utilities and shared query/executor definitions for QBICE
//! integration tests.
//!
//! This crate provides shared types used across integration tests.

#![allow(missing_docs)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_const_for_fn)]

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use qbice::{
    Decode, Encode, Engine, Identifiable, StableHash, TrackedEngine,
    config::Config,
    executor::Executor,
    query::Query,
    serialize::Plugin,
    stable_hash::{SeededStableHasherBuilder, Sip128Hasher},
    storage::kv_database::rocksdb::RocksDB,
};
use tempfile::TempDir;

// ============================================================================
// Basic Variable Query
// ============================================================================

/// A simple input variable query.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    StableHash,
    Encode,
    Decode,
    Identifiable,
)]
pub struct Variable(pub u64);

impl Query for Variable {
    type Value = i64;
}

// ============================================================================
// Division Queries
// ============================================================================

/// A query that divides two variables.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    StableHash,
    Encode,
    Decode,
    Identifiable,
)]
pub struct Division {
    pub dividend: Variable,
    pub divisor: Variable,
}

impl Query for Division {
    type Value = i64;
}

impl Division {
    pub fn new(dividend: Variable, divisor: Variable) -> Self {
        Self { dividend, divisor }
    }
}

/// Executor for [`Division`] queries.
#[derive(Debug, Default)]
pub struct DivisionExecutor(pub AtomicUsize);

impl<C: Config> Executor<Division, C> for DivisionExecutor {
    async fn execute(
        &self,
        query: &Division,
        engine: &TrackedEngine<C>,
    ) -> i64 {
        self.0.fetch_add(1, Ordering::Relaxed);

        let dividend = engine.query(&query.dividend).await;
        let divisor = engine.query(&query.divisor).await;

        assert!(divisor != 0, "division by zero");

        dividend / divisor
    }
}

/// A safe division query that returns None for division by zero.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Identifiable,
    Encode,
    Decode,
    StableHash,
)]
pub struct SafeDivision {
    pub dividend: Variable,
    pub divisor: Variable,
}

impl Query for SafeDivision {
    type Value = Option<i64>;
}

impl SafeDivision {
    pub fn new(dividend: Variable, divisor: Variable) -> Self {
        Self { dividend, divisor }
    }
}

/// Executor for [`SafeDivision`] queries.
#[derive(Debug, Default)]
pub struct SafeDivisionExecutor(pub AtomicUsize);

impl<C: Config> Executor<SafeDivision, C> for SafeDivisionExecutor {
    async fn execute(
        &self,
        query: &SafeDivision,
        engine: &TrackedEngine<C>,
    ) -> Option<i64> {
        self.0.fetch_add(1, Ordering::Relaxed);

        let divisor = engine.query(&query.divisor).await;

        if divisor == 0 {
            None
        } else {
            Some(
                engine
                    .query(&Division::new(query.dividend, query.divisor))
                    .await,
            )
        }
    }
}

// ============================================================================
// Absolute Value Queries
// ============================================================================

/// A query that computes the absolute value of a variable.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    StableHash,
    Encode,
    Decode,
    Identifiable,
)]
pub struct Absolute {
    pub variable: Variable,
}

impl Query for Absolute {
    type Value = i64;
}

impl Absolute {
    pub fn new(variable: Variable) -> Self { Self { variable } }
}

/// Executor for [`Absolute`] queries.
#[derive(Debug, Default)]
pub struct AbsoluteExecutor(pub AtomicUsize);

impl<C: Config> Executor<Absolute, C> for AbsoluteExecutor {
    async fn execute(
        &self,
        query: &Absolute,
        engine: &TrackedEngine<C>,
    ) -> i64 {
        self.0.fetch_add(1, Ordering::Relaxed);

        let value = engine.query(&query.variable).await;

        value.abs()
    }
}

/// A query that adds the absolute values of two variables.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    StableHash,
    Encode,
    Decode,
    Identifiable,
)]
pub struct AddTwoAbsolutes {
    pub var_a: Variable,
    pub var_b: Variable,
}

impl Query for AddTwoAbsolutes {
    type Value = i64;
}

impl AddTwoAbsolutes {
    pub fn new(var_a: Variable, var_b: Variable) -> Self {
        Self { var_a, var_b }
    }
}

/// Executor for [`AddTwoAbsolutes`] queries.
#[derive(Debug, Default)]
pub struct AddTwoAbsolutesExecutor(pub AtomicUsize);

impl<C: Config> Executor<AddTwoAbsolutes, C> for AddTwoAbsolutesExecutor {
    async fn execute(
        &self,
        query: &AddTwoAbsolutes,
        engine: &TrackedEngine<C>,
    ) -> i64 {
        self.0.fetch_add(1, Ordering::Relaxed);

        let abs_a = engine.query(&Absolute::new(query.var_a)).await;
        let abs_b = engine.query(&Absolute::new(query.var_b)).await;

        abs_a + abs_b
    }
}

// ============================================================================
// Slow/Blocking Query (for cancellation tests)
// ============================================================================

/// A query that can be made to hang indefinitely for cancellation tests.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Identifiable,
    Encode,
    Decode,
    StableHash,
)]
pub struct SlowQuery(pub u64);

impl Query for SlowQuery {
    type Value = i64;
}

/// Executor for [`SlowQuery`] that can be configured to hang.
#[derive(Debug, Default)]
pub struct SlowExecutor {
    pub make_it_stuck: AtomicBool,
}

impl<C: Config> Executor<SlowQuery, C> for SlowExecutor {
    async fn execute(
        &self,
        query: &SlowQuery,
        engine: &TrackedEngine<C>,
    ) -> i64 {
        if self.make_it_stuck.load(Ordering::Relaxed) {
            loop {
                tokio::task::yield_now().await;
            }
        } else {
            engine.query(&Variable(query.0)).await
        }
    }
}

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Default,
    Identifiable,
)]
pub struct TestingConfig;

impl Config for TestingConfig {
    type Database = RocksDB;

    type BuildStableHasher = SeededStableHasherBuilder<Sip128Hasher>;

    type BuildHasher = std::collections::hash_map::RandomState;

    // set the cache entry small for stress testing the cache eviction
    fn cache_entry_capacity() -> usize { 4 }

    fn rayon_thread_pool_builder() -> rayon::ThreadPoolBuilder {
        rayon::ThreadPoolBuilder::new()
    }
}

pub fn create_test_engine(tempdir: &TempDir) -> Engine<TestingConfig> {
    Engine::<TestingConfig>::new_with(
        Plugin::default(),
        RocksDB::factory(tempdir.path()),
        SeededStableHasherBuilder::<Sip128Hasher>::new(0),
    )
    .unwrap()
}
