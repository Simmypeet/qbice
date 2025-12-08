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
            tokio::time::sleep(Duration::from_secs(1)).await;

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
