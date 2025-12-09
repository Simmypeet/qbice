//! Cancellation safety tests.

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Duration,
};

use qbice::{
    config::{Config, DefaultConfig},
    engine::{Engine, TrackedEngine},
    executor::{CyclicError, Executor},
    query::Query,
};
use qbice_stable_hash::StableHash;
use qbice_stable_type_id::Identifiable;
use tokio::task::yield_now;

use crate::common::{SlowExecutor, SlowQuery, Variable};

#[tokio::test]
async fn cancellation_safety() {
    let mut engine = Engine::<DefaultConfig>::default();

    let slow_executor = Arc::new(SlowExecutor::default());

    engine.register_executor(slow_executor.clone());

    let mut input_session = engine.input_session();

    input_session.set_input(Variable(0), 123);

    drop(input_session);

    let engine = Arc::new(engine);
    let tracked_engine = engine.tracked();

    // Now, set the executor to make it stuck
    slow_executor.make_it_stuck.store(true, Ordering::Relaxed);

    let tracked_engine_clone = tracked_engine.clone();

    // Spawn a task to run the slow query
    tokio::select! {
        () = tokio::time::sleep(Duration::from_millis(100)) => {
            // After a short delay, we cancel the task
        }
        result = tracked_engine_clone.query(&SlowQuery(0)) => {
            panic!("The slow query task should have been aborted, but it completed with result: {result:?}");
        }
    };

    // Finally, reset the executor to not be stuck and ensure we can run the
    // query again
    slow_executor.make_it_stuck.store(false, Ordering::Relaxed);

    let result = tracked_engine.query(&SlowQuery(0)).await.unwrap();

    assert_eq!(result, 123);
}

// ============================================================================
// Cancellable Chain Query Types
// ============================================================================

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
    StableHash,
)]
pub struct CancellableChainA(pub u64);

impl Query for CancellableChainA {
    type Value = i64;
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
    Identifiable,
    StableHash,
)]
pub struct CancellableChainB(pub u64);

impl Query for CancellableChainB {
    type Value = i64;
}

#[derive(Debug, Default)]
pub struct CancellableChainAExecutor {
    pub call_count: AtomicUsize,
    pub should_cancel: AtomicBool,
}

impl<C: Config> Executor<CancellableChainA, C> for CancellableChainAExecutor {
    async fn execute(
        &self,
        query: &CancellableChainA,
        engine: &TrackedEngine<C>,
    ) -> Result<i64, CyclicError> {
        self.call_count.fetch_add(1, Ordering::SeqCst);

        // Query the dependency
        let b_value = engine.query(&CancellableChainB(query.0)).await?;

        // Simulate work after dependency query
        if self.should_cancel.load(Ordering::Relaxed) {
            loop {
                yield_now().await;
            }
        }

        Ok(b_value + 10)
    }
}

#[derive(Debug, Default)]
pub struct CancellableChainBExecutor {
    pub call_count: AtomicUsize,
}

impl<C: Config> Executor<CancellableChainB, C> for CancellableChainBExecutor {
    async fn execute(
        &self,
        query: &CancellableChainB,
        engine: &TrackedEngine<C>,
    ) -> Result<i64, CyclicError> {
        self.call_count.fetch_add(1, Ordering::SeqCst);

        let var_value = engine.query(&Variable(query.0)).await?;

        Ok(var_value * 2)
    }
}

#[tokio::test]
async fn cancellation_with_dependency_chain() {
    let mut engine = Engine::<DefaultConfig>::default();

    let executor_a = Arc::new(CancellableChainAExecutor::default());
    let executor_b = Arc::new(CancellableChainBExecutor::default());

    engine.register_executor(executor_a.clone());
    engine.register_executor(executor_b.clone());

    let mut input_session = engine.input_session();
    input_session.set_input(Variable(0), 50);
    drop(input_session);

    let engine = Arc::new(engine);
    let tracked_engine = engine.tracked();

    // Set executor A to get stuck after querying B
    executor_a.should_cancel.store(true, Ordering::Relaxed);

    let tracked_engine_clone = tracked_engine.clone();

    // Try to query A, which will query B first, then get stuck
    tokio::select! {
        () = tokio::time::sleep(Duration::from_millis(100)) => {
            // Cancel after delay
        }
        result = tracked_engine_clone.query(&CancellableChainA(0)) => {
            panic!("Should have been cancelled, got: {result:?}");
        }
    };

    // B should have been called, but A should not have completed
    assert!(executor_b.call_count.load(Ordering::SeqCst) >= 1);

    // Now allow A to complete
    executor_a.should_cancel.store(false, Ordering::Relaxed);

    // Query A again - it should work and B should be cached
    let a_call_count_before = executor_a.call_count.load(Ordering::SeqCst);
    let b_call_count_before = executor_b.call_count.load(Ordering::SeqCst);

    let result = tracked_engine.query(&CancellableChainA(0)).await.unwrap();
    assert_eq!(result, 110); // (50 * 2) + 10

    // A should have been called again, but B should be reused from cache
    assert_eq!(
        executor_a.call_count.load(Ordering::SeqCst),
        a_call_count_before + 1
    );
    assert_eq!(
        executor_b.call_count.load(Ordering::SeqCst),
        b_call_count_before
    );
}

// ============================================================================
// Parallel Cancellable Query Types
// ============================================================================

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
    StableHash,
)]
pub struct ParallelCancellableQuery(pub u64);

impl Query for ParallelCancellableQuery {
    type Value = i64;
}

#[derive(Debug, Default)]
pub struct ParallelCancellableExecutor {
    pub call_count: AtomicUsize,
    pub delay_ms: AtomicUsize,
}

impl<C: Config> Executor<ParallelCancellableQuery, C>
    for ParallelCancellableExecutor
{
    async fn execute(
        &self,
        query: &ParallelCancellableQuery,
        engine: &TrackedEngine<C>,
    ) -> Result<i64, CyclicError> {
        self.call_count.fetch_add(1, Ordering::SeqCst);

        let delay = self.delay_ms.load(Ordering::Relaxed);
        if delay > 0 {
            tokio::time::sleep(Duration::from_millis(delay as u64)).await;
        }

        engine.query(&Variable(query.0)).await
    }
}

#[tokio::test]
async fn parallel_queries_with_cancellation() {
    let mut engine = Engine::<DefaultConfig>::default();

    let executor = Arc::new(ParallelCancellableExecutor::default());
    executor.delay_ms.store(200, Ordering::Relaxed);

    engine.register_executor(executor.clone());

    let mut input_session = engine.input_session();
    input_session.set_input(Variable(0), 100);
    input_session.set_input(Variable(1), 200);
    drop(input_session);

    let engine = Arc::new(engine);

    // Spawn multiple queries in parallel, but cancel some of them
    let tracked_engine_1 = engine.clone().tracked();
    let tracked_engine_2 = engine.clone().tracked();
    let tracked_engine_3 = engine.clone().tracked();

    let handle1 = tokio::spawn(async move {
        tokio::select! {
            () = tokio::time::sleep(Duration::from_millis(50)) => {
                None // Cancelled
            }
            result = tracked_engine_1.query(&ParallelCancellableQuery(0)) => {
                Some(result)
            }
        }
    });

    let handle2 = tokio::spawn(async move {
        tracked_engine_2.query(&ParallelCancellableQuery(1)).await
    });

    let handle3 = tokio::spawn(async move {
        tokio::select! {
            () = tokio::time::sleep(Duration::from_millis(50)) => {
                None // Cancelled
            }
            result = tracked_engine_3.query(&ParallelCancellableQuery(0)) => {
                Some(result)
            }
        }
    });

    let result1 = handle1.await.unwrap();
    let result2 = handle2.await.unwrap();
    let result3 = handle3.await.unwrap();

    // First and third queries should be cancelled
    assert_eq!(result1, None);
    assert_eq!(result3, None);

    // Second query should succeed
    assert_eq!(result2, Ok(200));

    // Now query again with no delay - should use cached values
    executor.delay_ms.store(0, Ordering::Relaxed);

    let tracked_engine = engine.clone().tracked();
    let result =
        tracked_engine.query(&ParallelCancellableQuery(0)).await.unwrap();
    assert_eq!(result, 100);

    // Query for Variable(0) should have completed despite cancellations
    // The exact call count depends on timing, but should be at least 1
    assert!(executor.call_count.load(Ordering::SeqCst) >= 1);
}

// ============================================================================
// Multi-Dependency Cancellable Query Types
// ============================================================================

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
    StableHash,
)]
pub struct MultiDependencyQuery {
    pub deps: [Variable; 3],
}

impl Query for MultiDependencyQuery {
    type Value = i64;
}

#[derive(Debug, Default)]
pub struct MultiDependencyExecutor {
    pub call_count: AtomicUsize,
    pub cancel_after_deps: AtomicUsize, // Cancel after N dependencies queried
}

impl<C: Config> Executor<MultiDependencyQuery, C> for MultiDependencyExecutor {
    async fn execute(
        &self,
        query: &MultiDependencyQuery,
        engine: &TrackedEngine<C>,
    ) -> Result<i64, CyclicError> {
        self.call_count.fetch_add(1, Ordering::SeqCst);

        let mut sum = 0;
        let cancel_after = self.cancel_after_deps.load(Ordering::Relaxed);

        for (i, &var) in query.deps.iter().enumerate() {
            let value = engine.query(&var).await?;
            sum += value;

            // If cancel_after is set and we've reached that point, hang
            if cancel_after > 0 && i + 1 >= cancel_after {
                loop {
                    yield_now().await;
                }
            }
        }

        Ok(sum)
    }
}

#[tokio::test]
async fn cancellation_with_partial_dependencies() {
    let mut engine = Engine::<DefaultConfig>::default();

    let executor = Arc::new(MultiDependencyExecutor::default());
    engine.register_executor(executor.clone());

    let mut input_session = engine.input_session();
    input_session.set_input(Variable(0), 10);
    input_session.set_input(Variable(1), 20);
    input_session.set_input(Variable(2), 30);
    drop(input_session);

    let mut engine = Arc::new(engine);
    let tracked_engine = engine.clone().tracked();

    // Cancel after querying 2 dependencies
    executor.cancel_after_deps.store(2, Ordering::Relaxed);

    let query =
        MultiDependencyQuery { deps: [Variable(0), Variable(1), Variable(2)] };

    // This should get cancelled partway through
    tokio::select! {
        () = tokio::time::sleep(Duration::from_secs(2)) => {
            // Cancelled
        }
        result = tracked_engine.query(&query) => {
            panic!("Should have been cancelled, got: {result:?}");
        }
    };

    drop(tracked_engine);

    // change the Variable(0) which should have been queried
    {
        let mut input_session =
            Arc::get_mut(&mut engine).unwrap().input_session();
        input_session.set_input(Variable(0), 15);
    }

    let tracked_engine = engine.clone().tracked();

    // Now allow full execution
    executor.cancel_after_deps.store(0, Ordering::Relaxed);

    let result = tracked_engine.query(&query).await.unwrap();

    assert_eq!(result, 15 + 20 + 30);

    // Executor should have been called twice
    assert_eq!(executor.call_count.load(Ordering::SeqCst), 2);
}

// ============================================================================
// Repairable Cancellable Query Types
// ============================================================================

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
    StableHash,
)]
pub struct RepairableCancellableQuery(pub Variable);

impl Query for RepairableCancellableQuery {
    type Value = i64;
}

#[derive(Debug, Default)]
pub struct RepairableCancellableExecutor {
    pub call_count: AtomicUsize,
    pub should_hang: AtomicBool,
}

impl<C: Config> Executor<RepairableCancellableQuery, C>
    for RepairableCancellableExecutor
{
    async fn execute(
        &self,
        query: &RepairableCancellableQuery,
        engine: &TrackedEngine<C>,
    ) -> Result<i64, CyclicError> {
        self.call_count.fetch_add(1, Ordering::SeqCst);

        let value = engine.query(&query.0).await?;

        if self.should_hang.load(Ordering::Relaxed) {
            loop {
                yield_now().await;
            }
        }

        Ok(value * 3)
    }
}

#[tokio::test]
async fn cancellation_during_repair() {
    let mut engine = Engine::<DefaultConfig>::default();

    let executor = Arc::new(RepairableCancellableExecutor::default());
    engine.register_executor(executor.clone());

    let mut input_session = engine.input_session();
    input_session.set_input(Variable(0), 100);
    drop(input_session);

    let mut engine = Arc::new(engine);
    let tracked_engine = engine.clone().tracked();

    // First, compute the query successfully
    let result = tracked_engine
        .query(&RepairableCancellableQuery(Variable(0)))
        .await
        .unwrap();
    assert_eq!(result, 300);
    assert_eq!(executor.call_count.load(Ordering::SeqCst), 1);

    drop(tracked_engine);

    // Change the input to trigger repair
    {
        let mut input_session =
            Arc::get_mut(&mut engine).unwrap().input_session();
        input_session.set_input(Variable(0), 200);
    }

    let tracked_engine = engine.clone().tracked();

    // Set executor to hang during repair
    executor.should_hang.store(true, Ordering::Relaxed);

    let tracked_engine_clone = tracked_engine.clone();

    // Try to query, which will trigger repair, but cancel it
    tokio::select! {
        () = tokio::time::sleep(Duration::from_millis(100)) => {
            // Cancelled during repair
        }
        result = tracked_engine_clone.query(&RepairableCancellableQuery(Variable(0))) => {
            panic!("Should have been cancelled during repair, got: {result:?}");
        }
    };

    // Now allow it to complete
    executor.should_hang.store(false, Ordering::Relaxed);

    // Query again - repair should succeed this time
    let call_count_before = executor.call_count.load(Ordering::SeqCst);

    let result = tracked_engine
        .query(&RepairableCancellableQuery(Variable(0)))
        .await
        .unwrap();
    assert_eq!(result, 600); // 200 * 3

    // Should have been called again during repair
    assert!(executor.call_count.load(Ordering::SeqCst) > call_count_before);
}

// ============================================================================
// Nested Cancellable Query Types
// ============================================================================

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
    StableHash,
)]
pub struct NestedCancellableQuery(pub u64);

impl Query for NestedCancellableQuery {
    type Value = i64;
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
    Identifiable,
    StableHash,
)]
pub struct NestedCancellableInner(pub u64);

impl Query for NestedCancellableInner {
    type Value = i64;
}

#[derive(Debug, Default)]
pub struct NestedCancellableOuterExecutor {
    pub call_count: AtomicUsize,
    pub hang_before_inner: AtomicBool,
    pub hang_after_inner: AtomicBool,
}

impl<C: Config> Executor<NestedCancellableQuery, C>
    for NestedCancellableOuterExecutor
{
    async fn execute(
        &self,
        query: &NestedCancellableQuery,
        engine: &TrackedEngine<C>,
    ) -> Result<i64, CyclicError> {
        self.call_count.fetch_add(1, Ordering::SeqCst);

        if self.hang_before_inner.load(Ordering::Relaxed) {
            loop {
                yield_now().await;
            }
        }

        let inner_value =
            engine.query(&NestedCancellableInner(query.0)).await?;

        if self.hang_after_inner.load(Ordering::Relaxed) {
            loop {
                yield_now().await;
            }
        }

        Ok(inner_value + 100)
    }
}

#[derive(Debug, Default)]
pub struct NestedCancellableInnerExecutor {
    pub call_count: AtomicUsize,
}

impl<C: Config> Executor<NestedCancellableInner, C>
    for NestedCancellableInnerExecutor
{
    async fn execute(
        &self,
        query: &NestedCancellableInner,
        engine: &TrackedEngine<C>,
    ) -> Result<i64, CyclicError> {
        self.call_count.fetch_add(1, Ordering::SeqCst);

        engine.query(&Variable(query.0)).await
    }
}

#[tokio::test]
async fn cancellation_at_different_nesting_levels() {
    let mut engine = Engine::<DefaultConfig>::default();

    let outer_executor = Arc::new(NestedCancellableOuterExecutor::default());
    let inner_executor = Arc::new(NestedCancellableInnerExecutor::default());

    engine.register_executor(outer_executor.clone());
    engine.register_executor(inner_executor.clone());

    let mut input_session = engine.input_session();
    input_session.set_input(Variable(0), 50);
    drop(input_session);

    let engine = Arc::new(engine);

    // Test 1: Cancel before inner query
    {
        let tracked_engine = engine.clone().tracked();
        outer_executor.hang_before_inner.store(true, Ordering::Relaxed);

        let tracked_engine_clone = tracked_engine.clone();

        tokio::select! {
            () = tokio::time::sleep(Duration::from_millis(100)) => {}
            result = tracked_engine_clone.query(&NestedCancellableQuery(0)) => {
                panic!("Should have been cancelled, got: {result:?}");
            }
        };

        // Inner should not have been called
        assert_eq!(inner_executor.call_count.load(Ordering::SeqCst), 0);

        outer_executor.hang_before_inner.store(false, Ordering::Relaxed);
    }

    // Test 2: Cancel after inner query
    {
        let tracked_engine = engine.clone().tracked();
        outer_executor.hang_after_inner.store(true, Ordering::Relaxed);

        let tracked_engine_clone = tracked_engine.clone();

        tokio::select! {
            () = tokio::time::sleep(Duration::from_millis(100)) => {}
            result = tracked_engine_clone.query(&NestedCancellableQuery(0)) => {
                panic!("Should have been cancelled, got: {result:?}");
            }
        };

        // Inner should have been called and completed
        assert!(inner_executor.call_count.load(Ordering::SeqCst) >= 1);

        outer_executor.hang_after_inner.store(false, Ordering::Relaxed);
    }

    // Test 3: Complete successfully
    {
        let tracked_engine = engine.clone().tracked();
        let inner_calls_before =
            inner_executor.call_count.load(Ordering::SeqCst);

        let result =
            tracked_engine.query(&NestedCancellableQuery(0)).await.unwrap();
        assert_eq!(result, 150); // 50 + 100

        // Inner should be cached from previous attempt
        assert_eq!(
            inner_executor.call_count.load(Ordering::SeqCst),
            inner_calls_before
        );
    }
}
