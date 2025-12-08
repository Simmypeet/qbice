use std::sync::{Arc, atomic::AtomicUsize};

use qbice_stable_hash::StableHash;
use qbice_stable_type_id::Identifiable;

use super::TrackedEngine;
use crate::{
    config::{Config, DefaultConfig},
    engine::Engine,
    executor::{CyclicQuery, Executor},
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
    ) -> Result<i64, CyclicQuery> {
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
    ) -> Result<Option<i64>, CyclicQuery> {
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
    ) -> Result<i64, CyclicQuery> {
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
    ) -> Result<i64, CyclicQuery> {
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
