use std::sync::Arc;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct DivisionExecutor;

impl<C: Config> Executor<Division, C> for DivisionExecutor {
    async fn execute(
        &self,
        query: &Division,
        engine: &TrackedEngine<C>,
    ) -> Result<i64, CyclicQuery> {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct SafeDivisionExecutor;

impl<C: Config> Executor<SafeDivision, C> for SafeDivisionExecutor {
    async fn execute(
        &self,
        query: &SafeDivision,
        engine: &TrackedEngine<C>,
    ) -> Result<Option<i64>, CyclicQuery> {
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

    engine.executor_registry.register(Arc::new(DivisionExecutor));
    engine.executor_registry.register(Arc::new(SafeDivisionExecutor));

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
}
