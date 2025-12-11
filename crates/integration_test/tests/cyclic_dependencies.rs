//! Tests for cyclic dependency detection and handling.
#![allow(missing_docs)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_const_for_fn)]

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use qbice::{
    config::{Config, DefaultConfig},
    engine::{Engine, TrackedEngine},
    executor::{CyclicError, Executor},
    query::Query,
};
use qbice_stable_hash::StableHash;
use qbice_stable_type_id::Identifiable;

// ============================================================================
// Basic Cyclic Query Types
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
pub struct CyclicQueryA;

impl Query for CyclicQueryA {
    type Value = i32;
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
pub struct CyclicQueryB;

impl Query for CyclicQueryB {
    type Value = i32;
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
pub struct DependentQuery;

impl Query for DependentQuery {
    type Value = i32;
}

#[derive(Debug, Default)]
pub struct CyclicExecutorA {
    pub call_count: AtomicUsize,
}

impl CyclicExecutorA {
    pub fn get_call_count(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }
}

impl<C: Config> Executor<CyclicQueryA, C> for CyclicExecutorA {
    async fn execute(
        &self,
        _key: &CyclicQueryA,
        engine: &TrackedEngine<C>,
    ) -> Result<i32, CyclicError> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        // This creates a cycle: A depends on B, B depends on A
        let b_value = engine.query(&CyclicQueryB).await?;

        Ok(b_value + 10)
    }

    fn scc_value() -> i32 {
        42 // default value to use in case of cycle
    }
}

#[derive(Debug, Default)]
pub struct CyclicExecutorB {
    pub call_count: AtomicUsize,
}

impl CyclicExecutorB {
    pub fn get_call_count(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }
}

impl<C: Config> Executor<CyclicQueryB, C> for CyclicExecutorB {
    async fn execute(
        &self,
        _key: &CyclicQueryB,
        engine: &TrackedEngine<C>,
    ) -> Result<i32, CyclicError> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        // This completes the cycle: B depends on A, A depends on B
        let a_value = engine.query(&CyclicQueryA).await?;
        Ok(a_value + 20)
    }

    fn scc_value() -> i32 {
        84 // default value to use in case of cycle
    }
}

#[derive(Debug, Default)]
pub struct DependentExecutor {
    pub call_count: AtomicUsize,
}

impl DependentExecutor {
    pub fn get_call_count(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }
}

impl<C: Config> Executor<DependentQuery, C> for DependentExecutor {
    async fn execute(
        &self,
        _key: &DependentQuery,
        engine: &TrackedEngine<C>,
    ) -> Result<i32, CyclicError> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        // This query depends on the cyclic queries
        let a_value = engine.query(&CyclicQueryA).await?;
        let b_value = engine.query(&CyclicQueryB).await?;

        Ok(a_value + b_value + 100)
    }
}

#[tokio::test]
async fn cyclic_dependency_returns_default_values() {
    let mut engine = Engine::<DefaultConfig>::default();

    let executor_a = Arc::new(CyclicExecutorA::default());
    let executor_b = Arc::new(CyclicExecutorB::default());

    engine.register_executor(executor_a.clone());
    engine.register_executor(executor_b.clone());

    let engine = Arc::new(engine);
    let tracked_engine = engine.tracked();

    // When we query CyclicQueryA, it should detect the cycle A -> B -> A
    // and return default values (0 for i32) without calling the executors
    let result_a = tracked_engine.query(&CyclicQueryA).await.unwrap();
    let result_b = tracked_engine.query(&CyclicQueryB).await.unwrap();

    // Both should return default values (0 for i32)
    assert_eq!(result_a, 42);
    assert_eq!(result_b, 84);

    // The executors will be called and increment their call counts,
    // but they will receive CyclicError when trying to query their dependencies
    // and return early without completing their computation

    // Both executors should be called exactly once during cycle detection:
    // A is called first, then B is called, then when B tries to call A again,
    // the cycle is detected and CyclicError is returned without calling A again
    assert_eq!(executor_a.get_call_count(), 1);
    assert_eq!(executor_b.get_call_count(), 1);
}

#[tokio::test]
async fn dependent_query_uses_cyclic_default_values() {
    let mut engine = Engine::<DefaultConfig>::default();

    let executor_a = Arc::new(CyclicExecutorA::default());
    let executor_b = Arc::new(CyclicExecutorB::default());
    let executor_dependent = Arc::new(DependentExecutor::default());

    engine.register_executor(executor_a.clone());
    engine.register_executor(executor_b.clone());
    engine.register_executor(executor_dependent.clone());

    // Query the dependent query, which depends on the cyclic queries
    let engine = Arc::new(engine).tracked();

    let result = engine.query(&DependentQuery).await.unwrap();

    // DependentQuery should execute and use the default values from the cyclic
    // queries Default values: CyclicQueryA = 0, CyclicQueryB = 0
    // So result = 84 + 42 + 100 = 226
    assert_eq!(result, 226);

    // When DependentQuery queries CyclicQueryA, the executor for CyclicQueryA
    // will be called and try to query CyclicQueryB, which triggers cycle
    // detection. Both cyclic executors will be called exactly once during
    // cycle detection.
    assert_eq!(executor_a.get_call_count(), 1);
    assert_eq!(executor_b.get_call_count(), 1);

    // The dependent executor should have been called once
    assert_eq!(executor_dependent.get_call_count(), 1);

    // Try calling the dependent query again
    let result_again = engine.query(&DependentQuery).await.unwrap();

    // It should return the same result without calling the executors again
    assert_eq!(result_again, 226);

    // The call counts should remain the same since the result is cached
    assert_eq!(executor_a.get_call_count(), 1);
    assert_eq!(executor_b.get_call_count(), 1);
    assert_eq!(executor_dependent.get_call_count(), 1);
}

// ============================================================================
// Conditional Cyclic Query Types
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
pub struct ConditionalCyclicQueryA;

impl Query for ConditionalCyclicQueryA {
    type Value = i32;
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
pub struct ConditionalCyclicQueryB;

impl Query for ConditionalCyclicQueryB {
    type Value = i32;
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
pub struct CycleControlVariable;

impl Query for CycleControlVariable {
    type Value = i32;
}

#[derive(Debug, Default)]
pub struct ConditionalCyclicExecutorA {
    pub call_count: AtomicUsize,
}

impl ConditionalCyclicExecutorA {
    pub fn get_call_count(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }

    pub fn reset_call_count(&self) {
        self.call_count.store(0, Ordering::SeqCst);
    }
}

impl<C: Config> Executor<ConditionalCyclicQueryA, C>
    for ConditionalCyclicExecutorA
{
    async fn execute(
        &self,
        _key: &ConditionalCyclicQueryA,
        engine: &TrackedEngine<C>,
    ) -> Result<i32, CyclicError> {
        self.call_count.fetch_add(1, Ordering::SeqCst);

        // Read the control variable to determine whether to create a cycle
        let control_value = engine.query(&CycleControlVariable).await?;

        Ok(if control_value == 1 {
            // When control_value is 1, create a cycle by querying B
            let b_value = engine.query(&ConditionalCyclicQueryB).await?;

            b_value + 10
        } else {
            // When control_value is not 1, no cycle - just return a computed
            // value
            control_value * 100
        })
    }

    fn scc_value() -> i32 { 0 }
}

#[derive(Debug, Default)]
pub struct ConditionalCyclicExecutorB {
    pub call_count: AtomicUsize,
}

impl ConditionalCyclicExecutorB {
    pub fn get_call_count(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }

    pub fn reset_call_count(&self) {
        self.call_count.store(0, Ordering::SeqCst);
    }
}

impl<C: Config> Executor<ConditionalCyclicQueryB, C>
    for ConditionalCyclicExecutorB
{
    async fn execute(
        &self,
        _key: &ConditionalCyclicQueryB,
        engine: &TrackedEngine<C>,
    ) -> Result<i32, CyclicError> {
        self.call_count.fetch_add(1, Ordering::SeqCst);

        // Read the control variable to determine whether to create a cycle
        let control_value = engine.query(&CycleControlVariable).await?;

        Ok(if control_value == 1 {
            // When control_value is 1, complete the cycle by querying A
            let a_value = engine.query(&ConditionalCyclicQueryA).await?;

            a_value + 20
        } else {
            // When control_value is not 1, no cycle - just return a computed
            // value
            control_value * 200
        })
    }

    fn scc_value() -> i32 { 0 }
}

// Executor for DependentQuery that depends on conditional cyclic queries
#[derive(Debug, Default)]
pub struct ConditionalDependentExecutor {
    pub call_count: AtomicUsize,
}

impl ConditionalDependentExecutor {
    pub fn get_call_count(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }
}

impl<C: Config> Executor<DependentQuery, C> for ConditionalDependentExecutor {
    async fn execute(
        &self,
        _key: &DependentQuery,
        engine: &TrackedEngine<C>,
    ) -> Result<i32, CyclicError> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        // This query depends on the conditional cyclic queries
        let a_value = engine.query(&ConditionalCyclicQueryA).await?;
        let b_value = engine.query(&ConditionalCyclicQueryB).await?;

        Ok(a_value + b_value + 100)
    }
}

#[tokio::test]
#[allow(clippy::similar_names)]
async fn conditional_cyclic_dependency() {
    let mut engine = Engine::<DefaultConfig>::default();

    let executor_a = Arc::new(ConditionalCyclicExecutorA::default());
    let executor_b = Arc::new(ConditionalCyclicExecutorB::default());

    engine.register_executor(Arc::clone(&executor_a));
    engine.register_executor(Arc::clone(&executor_b));

    // Phase 1: Set control value to create NO cycle (control_value != 1)
    {
        let mut input_session = engine.input_session();
        input_session.set_input(CycleControlVariable, 5);
    }

    let mut engine = Arc::new(engine);
    let tracked_engine = engine.clone().tracked();

    // Query both A and B - they should compute normal values without cycles
    let result_a =
        tracked_engine.query(&ConditionalCyclicQueryA).await.unwrap();
    let result_b =
        tracked_engine.query(&ConditionalCyclicQueryB).await.unwrap();

    // Expected values: A = 5 * 100 = 500, B = 5 * 200 = 1000
    assert_eq!(result_a, 500);
    assert_eq!(result_b, 1000);

    // Both executors should have been called once each
    assert_eq!(executor_a.get_call_count(), 1);
    assert_eq!(executor_b.get_call_count(), 1);

    drop(tracked_engine);

    // Phase 2: Change control value to CREATE a cycle (control_value == 1)
    {
        let mut input_session =
            Arc::get_mut(&mut engine).unwrap().input_session();

        input_session.set_input(CycleControlVariable, 1);
    }

    executor_a.reset_call_count();
    executor_b.reset_call_count();

    let tracked_engine = engine.clone().tracked();

    // Query A - this should trigger cycle detection and return default values
    let result_a_cyclic =
        tracked_engine.query(&ConditionalCyclicQueryA).await.unwrap();
    let result_b_cyclic =
        tracked_engine.query(&ConditionalCyclicQueryB).await.unwrap();

    // Both should return default values (0 for i32) due to cycle detection
    assert_eq!(result_a_cyclic, 0);
    assert_eq!(result_b_cyclic, 0);

    // Both executors should be called exactly once during cycle detection
    assert_eq!(executor_a.get_call_count(), 1);
    assert_eq!(executor_b.get_call_count(), 1);

    drop(tracked_engine);

    // Phase 3: Change control value back to break the cycle (control_value !=
    // 1)
    {
        let mut input_session =
            Arc::get_mut(&mut engine).unwrap().input_session();
        input_session.set_input(CycleControlVariable, 3);
    }

    executor_a.reset_call_count();
    executor_b.reset_call_count();

    let tracked_engine = engine.clone().tracked();

    // Query both A and B - they should recompute and return normal values again
    let result_a_normal =
        tracked_engine.query(&ConditionalCyclicQueryA).await.unwrap();
    let result_b_normal =
        tracked_engine.query(&ConditionalCyclicQueryB).await.unwrap();

    // Expected values: A = 3 * 100 = 300, B = 3 * 200 = 600
    assert_eq!(result_a_normal, 300);
    assert_eq!(result_b_normal, 600);

    // Both executors should have been called once each for recomputation
    assert_eq!(executor_a.get_call_count(), 1);
    assert_eq!(executor_b.get_call_count(), 1);

    drop(tracked_engine);

    // Phase 4: Create cycle again with a different control value
    // (control_value == 1)

    {
        let mut input_session =
            Arc::get_mut(&mut engine).unwrap().input_session();
        input_session.set_input(CycleControlVariable, 1);
    }

    executor_a.reset_call_count();
    executor_b.reset_call_count();

    let tracked_engine = engine.clone().tracked();

    // Query A - cycle should be detected again and default values returned
    let result_a_cyclic2 =
        tracked_engine.query(&ConditionalCyclicQueryA).await.unwrap();
    let result_b_cyclic2 =
        tracked_engine.query(&ConditionalCyclicQueryB).await.unwrap();

    // Both should return default values (0 for i32) due to cycle detection
    assert_eq!(result_a_cyclic2, 0);
    assert_eq!(result_b_cyclic2, 0);

    // Both executors should be called exactly once during cycle detection
    assert_eq!(executor_a.get_call_count(), 1);
    assert_eq!(executor_b.get_call_count(), 1);
}

#[tokio::test]
async fn conditional_cyclic_with_dependent_query() {
    let mut engine = Engine::<DefaultConfig>::default();

    let executor_a = Arc::new(ConditionalCyclicExecutorA::default());
    let executor_b = Arc::new(ConditionalCyclicExecutorB::default());
    let executor_dependent = Arc::new(ConditionalDependentExecutor::default());

    engine.register_executor(Arc::clone(&executor_a));
    engine.register_executor(Arc::clone(&executor_b));
    engine.register_executor(Arc::clone(&executor_dependent));

    // Phase 1: No cycle - dependent query should use computed values
    {
        let mut input_session = engine.input_session();
        input_session.set_input(CycleControlVariable, 2);
    }

    let mut engine = Arc::new(engine);
    let tracked_engine = engine.clone().tracked();

    let result_dependent = tracked_engine.query(&DependentQuery).await.unwrap();

    // DependentQuery = A + B + 100 = (2*100) + (2*200) + 100 = 200 + 400 + 100
    // = 700
    assert_eq!(result_dependent, 700);

    // All executors should have been called
    assert_eq!(executor_a.get_call_count(), 1);
    assert_eq!(executor_b.get_call_count(), 1);
    assert_eq!(executor_dependent.get_call_count(), 1);

    drop(tracked_engine);

    // Phase 2: Create cycle - dependent query should use default values
    {
        let mut input_session =
            Arc::get_mut(&mut engine).unwrap().input_session();
        input_session.set_input(CycleControlVariable, 1);
    }

    let tracked_engine = engine.clone().tracked();

    executor_a.reset_call_count();
    executor_b.reset_call_count();

    // Reset dependent executor call count to track new computation
    executor_dependent.call_count.store(0, Ordering::SeqCst);

    let result_dependent_cyclic =
        tracked_engine.query(&DependentQuery).await.unwrap();

    // DependentQuery should use default values: 0 + 0 + 100 = 100
    assert_eq!(result_dependent_cyclic, 100);

    // Cyclic executors called once each during cycle detection
    assert_eq!(executor_a.get_call_count(), 1);
    assert_eq!(executor_b.get_call_count(), 1);
    // Dependent executor called once to compute with new (default) values
    assert_eq!(executor_dependent.get_call_count(), 1);

    drop(tracked_engine);

    // Phase 3: Break cycle again - dependent query should use computed values
    {
        let mut input_session =
            Arc::get_mut(&mut engine).unwrap().input_session();
        input_session.set_input(CycleControlVariable, 4);
    }

    let tracked_engine = engine.clone().tracked();

    executor_a.reset_call_count();
    executor_b.reset_call_count();
    executor_dependent.call_count.store(0, Ordering::SeqCst);

    // Query A and B to ensure they're computed
    let _debug_a = tracked_engine.query(&ConditionalCyclicQueryA).await;
    let _debug_b = tracked_engine.query(&ConditionalCyclicQueryB).await;

    let result_dependent_normal =
        tracked_engine.query(&DependentQuery).await.unwrap();

    // DependentQuery = A + B + 100 = (4*100) + (4*200) + 100 = 400 + 800 + 100
    // = 1300
    assert_eq!(result_dependent_normal, 1300);

    // All executors should have been called for recomputation
    assert_eq!(executor_a.get_call_count(), 1);
    assert_eq!(executor_b.get_call_count(), 1);
    assert_eq!(executor_dependent.get_call_count(), 1);
}
