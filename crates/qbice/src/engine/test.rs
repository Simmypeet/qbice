use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize},
    },
    time::Duration,
};

use qbice_stable_hash::StableHash;
use qbice_stable_type_id::Identifiable;
use tokio::task::yield_now;

use super::TrackedEngine;
use crate::{
    config::{Config, DefaultConfig},
    engine::Engine,
    executor::{CyclicError, Executor},
    query::Query,
};

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
    Identifiable,
)]
pub struct Variable(u64);

impl Query for Variable {
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
    StableHash,
    Identifiable,
)]
pub struct Division {
    dividend: Variable,
    divisor: Variable,
}

impl Query for Division {
    type Value = i64;
}

#[derive(Debug, Default)]
pub struct DivisionExecutor(pub AtomicUsize);

impl<C: Config> Executor<Division, C> for DivisionExecutor {
    async fn execute(
        &self,
        query: &Division,
        engine: &TrackedEngine<C>,
    ) -> Result<i64, CyclicError> {
        // track usage
        self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let dividend = engine.query(&query.dividend).await?;
        let divisor = engine.query(&query.divisor).await?;

        assert!(divisor != 0, "division by zero");

        Ok(dividend / divisor)
    }
}

impl Division {
    pub fn new(dividend: Variable, divisor: Variable) -> Self {
        Self { dividend, divisor }
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
    Identifiable,
    StableHash,
)]
pub struct SafeDivision {
    dividend: Variable,
    divisor: Variable,
}

impl Query for SafeDivision {
    type Value = Option<i64>;
}

#[derive(Debug, Default)]
pub struct SafeDivisionExecutor(pub AtomicUsize);

impl<C: Config> Executor<SafeDivision, C> for SafeDivisionExecutor {
    async fn execute(
        &self,
        query: &SafeDivision,
        engine: &TrackedEngine<C>,
    ) -> Result<Option<i64>, CyclicError> {
        // track usage
        self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let divisor = engine.query(&query.divisor).await?;

        if divisor == 0 {
            Ok(None)
        } else {
            Ok(Some(
                engine
                    .query(&Division::new(query.dividend, query.divisor))
                    .await?,
            ))
        }
    }
}

impl SafeDivision {
    pub fn new(dividend: Variable, divisor: Variable) -> Self {
        Self { dividend, divisor }
    }
}

#[tokio::test]
pub async fn safe_division_basic() {
    let mut engine = Engine::<DefaultConfig>::default();

    let division_ex = Arc::new(DivisionExecutor::default());
    let safe_division_ex = Arc::new(SafeDivisionExecutor::default());

    engine.executor_registry.register(division_ex.clone());
    engine.executor_registry.register(safe_division_ex.clone());

    let mut input_session = engine.input_session();

    input_session.set_input(Variable(0), 42);
    input_session.set_input(Variable(1), 2);

    drop(input_session);

    let engine = Arc::new(engine);
    let tracked_engine = engine.tracked();

    assert_eq!(
        tracked_engine
            .query(&SafeDivision::new(Variable(0), Variable(1)))
            .await,
        Ok(Some(21))
    );

    assert_eq!(division_ex.0.load(std::sync::atomic::Ordering::Relaxed), 1);
    assert_eq!(
        safe_division_ex.0.load(std::sync::atomic::Ordering::Relaxed),
        1
    );
}

#[tokio::test]
async fn division_input_change() {
    let mut engine = Engine::<DefaultConfig>::default();

    let division_ex = Arc::new(DivisionExecutor::default());

    engine.executor_registry.register(division_ex.clone());

    let mut input_session = engine.input_session();

    input_session.set_input(Variable(0), 40);
    input_session.set_input(Variable(1), 4);

    drop(input_session);

    let mut engine = Arc::new(engine);
    let tracked_engine = engine.clone().tracked();

    assert_eq!(
        tracked_engine.query(&Division::new(Variable(0), Variable(1))).await,
        Ok(10)
    );

    assert_eq!(division_ex.0.load(std::sync::atomic::Ordering::Relaxed), 1);

    // drop tracked engine to release references
    drop(tracked_engine);

    {
        let mut input_session =
            Arc::get_mut(&mut engine).unwrap().input_session();

        input_session.set_input(Variable(1), 2);
    }

    let tracked_engine = engine.tracked();

    assert_eq!(
        tracked_engine.query(&Division::new(Variable(0), Variable(1))).await,
        Ok(20)
    );

    assert_eq!(division_ex.0.load(std::sync::atomic::Ordering::Relaxed), 2);
}

#[tokio::test]
async fn safe_division_input_changes() {
    let mut engine = Engine::<DefaultConfig>::default();

    let division_ex = Arc::new(DivisionExecutor::default());
    let safe_division_ex = Arc::new(SafeDivisionExecutor::default());

    engine.executor_registry.register(division_ex.clone());
    engine.executor_registry.register(safe_division_ex.clone());

    let mut input_session = engine.input_session();

    input_session.set_input(Variable(0), 42);
    input_session.set_input(Variable(1), 2);

    drop(input_session);

    let mut engine = Arc::new(engine);
    let tracked_engine = engine.clone().tracked();

    assert_eq!(
        tracked_engine
            .query(&SafeDivision::new(Variable(0), Variable(1)))
            .await,
        Ok(Some(21))
    );

    assert_eq!(division_ex.0.load(std::sync::atomic::Ordering::Relaxed), 1);
    assert_eq!(
        safe_division_ex.0.load(std::sync::atomic::Ordering::Relaxed),
        1
    );

    // drop tracked engine to release references
    drop(tracked_engine);

    // now make it divide by zero
    {
        let mut input_session =
            Arc::get_mut(&mut engine).unwrap().input_session();

        input_session.set_input(Variable(1), 0);
    }

    let tracked_engine = engine.clone().tracked();

    assert_eq!(
        tracked_engine
            .query(&SafeDivision::new(Variable(0), Variable(1)))
            .await,
        Ok(None)
    );

    // division executor should not have been called again, but safe division
    // should have
    assert_eq!(division_ex.0.load(std::sync::atomic::Ordering::Relaxed), 1);
    assert_eq!(
        safe_division_ex.0.load(std::sync::atomic::Ordering::Relaxed),
        2
    );

    // now restore to original value
    drop(tracked_engine);
    {
        let mut input_session =
            Arc::get_mut(&mut engine).unwrap().input_session();

        input_session.set_input(Variable(1), 2);
    }

    let tracked_engine = engine.tracked();
    assert_eq!(
        tracked_engine
            .query(&SafeDivision::new(Variable(0), Variable(1)))
            .await,
        Ok(Some(21))
    );

    // this time divisor executor reused the cached value but safe division
    // had to run again
    assert_eq!(division_ex.0.load(std::sync::atomic::Ordering::Relaxed), 1);
    assert_eq!(
        safe_division_ex.0.load(std::sync::atomic::Ordering::Relaxed),
        3
    );
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
    StableHash,
    Identifiable,
)]
pub struct Absolute {
    variable: Variable,
}

impl Query for Absolute {
    type Value = i64;
}

#[derive(Debug, Default)]
pub struct AbsoluteExecutor(pub AtomicUsize);

impl<C: Config> Executor<Absolute, C> for AbsoluteExecutor {
    async fn execute(
        &self,
        query: &Absolute,
        engine: &TrackedEngine<C>,
    ) -> Result<i64, CyclicError> {
        // track usage
        self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let value = engine.query(&query.variable).await?;

        Ok(value.abs())
    }
}

impl Absolute {
    pub fn new(variable: Variable) -> Self { Self { variable } }
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
    StableHash,
    Identifiable,
)]
pub struct AddTwoAbsolutes {
    var_a: Variable,
    var_b: Variable,
}

impl Query for AddTwoAbsolutes {
    type Value = i64;
}

#[derive(Debug, Default)]
pub struct AddTwoAbsolutesExecutor(pub AtomicUsize);

impl<C: Config> Executor<AddTwoAbsolutes, C> for AddTwoAbsolutesExecutor {
    async fn execute(
        &self,
        query: &AddTwoAbsolutes,
        engine: &TrackedEngine<C>,
    ) -> Result<i64, CyclicError> {
        // track usage
        self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let abs_a = engine.query(&Absolute::new(query.var_a)).await?;
        let abs_b = engine.query(&Absolute::new(query.var_b)).await?;

        Ok(abs_a + abs_b)
    }
}

impl AddTwoAbsolutes {
    pub fn new(var_a: Variable, var_b: Variable) -> Self {
        Self { var_a, var_b }
    }
}

#[tokio::test]
async fn add_two_absolutes_sign_change() {
    let mut engine = Engine::<DefaultConfig>::default();

    let absolute_ex = Arc::new(AbsoluteExecutor::default());
    let add_two_absolutes_ex = Arc::new(AddTwoAbsolutesExecutor::default());

    engine.executor_registry.register(absolute_ex.clone());
    engine.executor_registry.register(add_two_absolutes_ex.clone());

    let mut input_session = engine.input_session();

    input_session.set_input(Variable(0), 200);
    input_session.set_input(Variable(1), 150);

    drop(input_session);

    let mut engine = Arc::new(engine);
    let tracked_engine = engine.clone().tracked();

    // Initial query: abs(200) + abs(150) = 350
    assert_eq!(
        tracked_engine
            .query(&AddTwoAbsolutes::new(Variable(0), Variable(1)))
            .await,
        Ok(350)
    );

    // Both absolute executors should run once, and add_two_absolutes once
    assert_eq!(absolute_ex.0.load(std::sync::atomic::Ordering::Relaxed), 2);
    assert_eq!(
        add_two_absolutes_ex.0.load(std::sync::atomic::Ordering::Relaxed),
        1
    );

    // Drop tracked engine to release references
    drop(tracked_engine);

    // Change Variable(0) from 200 to -200
    {
        let mut input_session =
            Arc::get_mut(&mut engine).unwrap().input_session();

        input_session.set_input(Variable(0), -200);
    }

    let tracked_engine = engine.clone().tracked();

    // Result should still be 350: abs(-200) + abs(150) = 200 + 150 = 350
    assert_eq!(
        tracked_engine
            .query(&AddTwoAbsolutes::new(Variable(0), Variable(1)))
            .await,
        Ok(350)
    );

    // Only the Absolute query for Variable(0) should re-execute (3 total)
    // The result is the same (200), so AddTwoAbsolutes should NOT re-execute
    assert_eq!(absolute_ex.0.load(std::sync::atomic::Ordering::Relaxed), 3);
    assert_eq!(
        add_two_absolutes_ex.0.load(std::sync::atomic::Ordering::Relaxed),
        1
    );

    // Drop tracked engine
    drop(tracked_engine);

    // Change Variable(1) from 150 to -150
    {
        let mut input_session =
            Arc::get_mut(&mut engine).unwrap().input_session();

        input_session.set_input(Variable(1), -150);
    }

    let tracked_engine = engine.tracked();

    // Result should still be 350: abs(-200) + abs(-150) = 200 + 150 = 350
    assert_eq!(
        tracked_engine
            .query(&AddTwoAbsolutes::new(Variable(0), Variable(1)))
            .await,
        Ok(350)
    );

    // Only the Absolute query for Variable(1) should re-execute (4 total)
    // The result is still the same (150), so AddTwoAbsolutes should still NOT
    // re-execute
    assert_eq!(absolute_ex.0.load(std::sync::atomic::Ordering::Relaxed), 4);
    assert_eq!(
        add_two_absolutes_ex.0.load(std::sync::atomic::Ordering::Relaxed),
        1
    );
}

// Test cases for cyclic dependency handling
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

    fn scc_value() -> Self::Value {
        42 // default value to use in case of cycle
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
    Identifiable,
    StableHash,
)]
pub struct CyclicQueryB;

impl Query for CyclicQueryB {
    type Value = i32;

    fn scc_value() -> Self::Value {
        84 // default value to use in case of cycle
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
        self.call_count.load(std::sync::atomic::Ordering::SeqCst)
    }
}

impl<C: Config> Executor<CyclicQueryA, C> for CyclicExecutorA {
    async fn execute(
        &self,
        _key: &CyclicQueryA,
        engine: &TrackedEngine<C>,
    ) -> Result<i32, CyclicError> {
        self.call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        // This creates a cycle: A depends on B, B depends on A
        let b_value = engine.query(&CyclicQueryB).await?;

        Ok(b_value + 10)
    }
}

#[derive(Debug, Default)]
pub struct CyclicExecutorB {
    pub call_count: AtomicUsize,
}

impl CyclicExecutorB {
    pub fn get_call_count(&self) -> usize {
        self.call_count.load(std::sync::atomic::Ordering::SeqCst)
    }
}

impl<C: Config> Executor<CyclicQueryB, C> for CyclicExecutorB {
    async fn execute(
        &self,
        _key: &CyclicQueryB,
        engine: &TrackedEngine<C>,
    ) -> Result<i32, CyclicError> {
        self.call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        // This completes the cycle: B depends on A, A depends on B
        let a_value = engine.query(&CyclicQueryA).await?;
        Ok(a_value + 20)
    }
}

#[derive(Debug, Default)]
pub struct DependentExecutor {
    pub call_count: AtomicUsize,
}

impl DependentExecutor {
    pub fn get_call_count(&self) -> usize {
        self.call_count.load(std::sync::atomic::Ordering::SeqCst)
    }
}

impl<C: Config> Executor<DependentQuery, C> for DependentExecutor {
    async fn execute(
        &self,
        _key: &DependentQuery,
        engine: &TrackedEngine<C>,
    ) -> Result<i32, CyclicError> {
        self.call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
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

#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Identifiable, StableHash,
)]
pub struct CollectVariables {
    pub vars: Arc<[Variable]>,
}

impl Query for CollectVariables {
    type Value = HashMap<Variable, i64>;
}

#[derive(Debug, Default)]
pub struct CollectVariablesExecutor(pub AtomicUsize);

impl<C: Config> Executor<CollectVariables, C> for CollectVariablesExecutor {
    async fn execute(
        &self,
        query: &CollectVariables,
        engine: &TrackedEngine<C>,
    ) -> Result<HashMap<Variable, i64>, CyclicError> {
        // track usage
        self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let mut result = HashMap::new();

        for &var in query.vars.iter() {
            // simulate some async work
            tokio::time::sleep(Duration::from_millis(16)).await;

            let value = engine.query(&var).await?;

            result.insert(var, value);
        }

        Ok(result)
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
    Identifiable,
    StableHash,
)]
pub struct ReadVariableMap(pub Variable);

impl Query for ReadVariableMap {
    type Value = i64;
}

#[derive(Debug, Default)]
pub struct ReadVariableMapExecutor(pub AtomicUsize);

impl<C: Config> Executor<ReadVariableMap, C> for ReadVariableMapExecutor {
    async fn execute(
        &self,
        query: &ReadVariableMap,
        engine: &TrackedEngine<C>,
    ) -> Result<i64, CyclicError> {
        // track usage
        self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let var_map = engine
            .query(&CollectVariables {
                vars: Arc::new([Variable(0), Variable(1), Variable(2)]),
            })
            .await?;

        Ok(*var_map.get(&query.0).unwrap())
    }
}

impl ReadVariableMap {
    pub fn new(var: Variable) -> Self { Self(var) }
}

#[tokio::test]
#[allow(clippy::cast_possible_wrap)]
async fn parallel_read_variable_map() {
    let mut engine = Engine::<DefaultConfig>::default();

    let collect_ex = Arc::new(CollectVariablesExecutor::default());
    let read_map_ex = Arc::new(ReadVariableMapExecutor::default());

    engine.executor_registry.register(collect_ex.clone());
    engine.executor_registry.register(read_map_ex.clone());

    let mut input_session = engine.input_session();

    input_session.set_input(Variable(0), 10);
    input_session.set_input(Variable(1), 20);
    input_session.set_input(Variable(2), 30);

    drop(input_session);

    let engine = Arc::new(engine);
    let tracked_engine = engine.tracked();

    let handles: Vec<_> = (0..3)
        .map(|i| {
            let tracked_engine = tracked_engine.clone();
            tokio::spawn(async move {
                tracked_engine.query(&ReadVariableMap::new(Variable(i))).await
            })
        })
        .collect();

    for (i, handle) in handles.into_iter().enumerate() {
        let result = handle.await.unwrap().unwrap();
        assert_eq!(result, (i as i64 + 1) * 10);
    }

    // CollectVariablesExecutor should have been called only once
    assert_eq!(collect_ex.0.load(std::sync::atomic::Ordering::Relaxed), 1);
    // ReadVariableMapExecutor should have been called three times
    assert_eq!(read_map_ex.0.load(std::sync::atomic::Ordering::Relaxed), 3);
}

// Cancellation safety tests

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
pub struct SlowQuery(pub u64);

impl Query for SlowQuery {
    type Value = i64;
}

#[derive(Debug, Default)]
pub struct SlowExecutor {
    pub make_it_stuck: AtomicBool,
}

impl<C: Config> Executor<SlowQuery, C> for SlowExecutor {
    async fn execute(
        &self,
        query: &SlowQuery,
        engine: &TrackedEngine<C>,
    ) -> Result<i64, CyclicError> {
        if self.make_it_stuck.load(std::sync::atomic::Ordering::Relaxed) {
            loop {
                yield_now().await;
            }
        } else {
            engine.query(&Variable(query.0)).await
        }
    }
}

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
    slow_executor
        .make_it_stuck
        .store(true, std::sync::atomic::Ordering::Relaxed);

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
    slow_executor
        .make_it_stuck
        .store(false, std::sync::atomic::Ordering::Relaxed);

    let result = tracked_engine.query(&SlowQuery(0)).await.unwrap();

    assert_eq!(result, 123);
}

// Additional cancellation safety tests

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
        self.call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        // Query the dependency
        let b_value = engine.query(&CancellableChainB(query.0)).await?;

        // Simulate work after dependency query
        if self.should_cancel.load(std::sync::atomic::Ordering::Relaxed) {
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
        self.call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

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
    executor_a.should_cancel.store(true, std::sync::atomic::Ordering::Relaxed);

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
    assert!(
        executor_b.call_count.load(std::sync::atomic::Ordering::SeqCst) >= 1
    );

    // Now allow A to complete
    executor_a.should_cancel.store(false, std::sync::atomic::Ordering::Relaxed);

    // Query A again - it should work and B should be cached
    let a_call_count_before =
        executor_a.call_count.load(std::sync::atomic::Ordering::SeqCst);
    let b_call_count_before =
        executor_b.call_count.load(std::sync::atomic::Ordering::SeqCst);

    let result = tracked_engine.query(&CancellableChainA(0)).await.unwrap();
    assert_eq!(result, 110); // (50 * 2) + 10

    // A should have been called again, but B should be reused from cache
    assert_eq!(
        executor_a.call_count.load(std::sync::atomic::Ordering::SeqCst),
        a_call_count_before + 1
    );
    assert_eq!(
        executor_b.call_count.load(std::sync::atomic::Ordering::SeqCst),
        b_call_count_before
    );
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
        self.call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        let delay = self.delay_ms.load(std::sync::atomic::Ordering::Relaxed);
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
    executor.delay_ms.store(200, std::sync::atomic::Ordering::Relaxed);

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
    executor.delay_ms.store(0, std::sync::atomic::Ordering::Relaxed);

    let tracked_engine = engine.clone().tracked();
    let result =
        tracked_engine.query(&ParallelCancellableQuery(0)).await.unwrap();
    assert_eq!(result, 100);

    // Query for Variable(0) should have completed despite cancellations
    // The exact call count depends on timing, but should be at least 1
    assert!(executor.call_count.load(std::sync::atomic::Ordering::SeqCst) >= 1);
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
        self.call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        let mut sum = 0;
        let cancel_after =
            self.cancel_after_deps.load(std::sync::atomic::Ordering::Relaxed);

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
    executor.cancel_after_deps.store(2, std::sync::atomic::Ordering::Relaxed);

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
    executor.cancel_after_deps.store(0, std::sync::atomic::Ordering::Relaxed);

    let result = tracked_engine.query(&query).await.unwrap();

    assert_eq!(result, 15 + 20 + 30);

    // Executor should have been called twice
    assert_eq!(
        executor.call_count.load(std::sync::atomic::Ordering::SeqCst),
        2
    );
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
        self.call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        let value = engine.query(&query.0).await?;

        if self.should_hang.load(std::sync::atomic::Ordering::Relaxed) {
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
    assert_eq!(
        executor.call_count.load(std::sync::atomic::Ordering::SeqCst),
        1
    );

    drop(tracked_engine);

    // Change the input to trigger repair
    {
        let mut input_session =
            Arc::get_mut(&mut engine).unwrap().input_session();
        input_session.set_input(Variable(0), 200);
    }

    let tracked_engine = engine.clone().tracked();

    // Set executor to hang during repair
    executor.should_hang.store(true, std::sync::atomic::Ordering::Relaxed);

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
    executor.should_hang.store(false, std::sync::atomic::Ordering::Relaxed);

    // Query again - repair should succeed this time
    let call_count_before =
        executor.call_count.load(std::sync::atomic::Ordering::SeqCst);

    let result = tracked_engine
        .query(&RepairableCancellableQuery(Variable(0)))
        .await
        .unwrap();
    assert_eq!(result, 600); // 200 * 3

    // Should have been called again during repair
    assert!(
        executor.call_count.load(std::sync::atomic::Ordering::SeqCst)
            > call_count_before
    );
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
        self.call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        if self.hang_before_inner.load(std::sync::atomic::Ordering::Relaxed) {
            loop {
                yield_now().await;
            }
        }

        let inner_value =
            engine.query(&NestedCancellableInner(query.0)).await?;

        if self.hang_after_inner.load(std::sync::atomic::Ordering::Relaxed) {
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
        self.call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

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
        outer_executor
            .hang_before_inner
            .store(true, std::sync::atomic::Ordering::Relaxed);

        let tracked_engine_clone = tracked_engine.clone();

        tokio::select! {
            () = tokio::time::sleep(Duration::from_millis(100)) => {}
            result = tracked_engine_clone.query(&NestedCancellableQuery(0)) => {
                panic!("Should have been cancelled, got: {result:?}");
            }
        };

        // Inner should not have been called
        assert_eq!(
            inner_executor.call_count.load(std::sync::atomic::Ordering::SeqCst),
            0
        );

        outer_executor
            .hang_before_inner
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }

    // Test 2: Cancel after inner query
    {
        let tracked_engine = engine.clone().tracked();
        outer_executor
            .hang_after_inner
            .store(true, std::sync::atomic::Ordering::Relaxed);

        let tracked_engine_clone = tracked_engine.clone();

        tokio::select! {
            () = tokio::time::sleep(Duration::from_millis(100)) => {}
            result = tracked_engine_clone.query(&NestedCancellableQuery(0)) => {
                panic!("Should have been cancelled, got: {result:?}");
            }
        };

        // Inner should have been called and completed
        assert!(
            inner_executor.call_count.load(std::sync::atomic::Ordering::SeqCst)
                >= 1
        );

        outer_executor
            .hang_after_inner
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }

    // Test 3: Complete successfully
    {
        let tracked_engine = engine.clone().tracked();
        let inner_calls_before =
            inner_executor.call_count.load(std::sync::atomic::Ordering::SeqCst);

        let result =
            tracked_engine.query(&NestedCancellableQuery(0)).await.unwrap();
        assert_eq!(result, 150); // 50 + 100

        // Inner should be cached from previous attempt
        assert_eq!(
            inner_executor.call_count.load(std::sync::atomic::Ordering::SeqCst),
            inner_calls_before
        );
    }
}
