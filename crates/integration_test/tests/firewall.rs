#![allow(missing_docs)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_const_for_fn)]

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use flexstr::SharedStr;
use qbice::{
    Engine, Identifiable, StableHash, TrackedEngine,
    config::{Config, DefaultConfig},
    executor::{CyclicError, Executor},
    query::Query,
};
use qbice_integration_test::Variable;
use tracing_test::traced_test;

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
pub struct Square(pub Variable);

impl Query for Square {
    type Value = i64;
}

#[derive(Debug, Default)]
pub struct SquareExecutor(pub AtomicUsize);

impl<C: Config> Executor<Square, C> for SquareExecutor {
    async fn execute(
        &self,
        query: &Square,
        engine: &TrackedEngine<C>,
    ) -> Result<i64, CyclicError> {
        self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let value = engine.query(&query.0).await?;

        Ok(value * value)
    }

    fn execution_style() -> qbice::ExecutionStyle {
        qbice::ExecutionStyle::Firewall
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
pub struct NegativeSquare(pub Variable);

impl Query for NegativeSquare {
    type Value = i64;
}

#[derive(Debug, Default)]
pub struct NegativeSquareExecutor(pub AtomicUsize);

impl<C: Config> Executor<NegativeSquare, C> for NegativeSquareExecutor {
    async fn execute(
        &self,
        query: &NegativeSquare,
        engine: &TrackedEngine<C>,
    ) -> Result<i64, CyclicError> {
        self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let value = engine.query(&Square(query.0)).await?;

        Ok(-value)
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
pub struct NegativeSquareToString(pub Variable);

impl Query for NegativeSquareToString {
    type Value = SharedStr;
}

#[derive(Debug, Default)]
pub struct NegativeSquareToStringExecutor(pub AtomicUsize);

impl<C: Config> Executor<NegativeSquareToString, C>
    for NegativeSquareToStringExecutor
{
    async fn execute(
        &self,
        query: &NegativeSquareToString,
        engine: &TrackedEngine<C>,
    ) -> Result<SharedStr, CyclicError> {
        self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let value = engine.query(&NegativeSquare(query.0)).await?;

        Ok(SharedStr::from(value.to_string()))
    }
}

#[tokio::test]
#[traced_test]
async fn firewall() {
    let mut engine = Engine::<DefaultConfig>::default();

    let square_ex = Arc::new(SquareExecutor::default());
    let negative_square_ex = Arc::new(NegativeSquareExecutor::default());
    let negative_square_to_string_ex =
        Arc::new(NegativeSquareToStringExecutor::default());

    engine.register_executor(square_ex.clone());
    engine.register_executor(negative_square_ex.clone());
    engine.register_executor(negative_square_to_string_ex.clone());

    // Step 1: initial to 3
    {
        let mut input_session = engine.input_session();
        input_session.set_input(Variable(0), 3);
    }

    let mut engine = Arc::new(engine);
    let tracked_engine = engine.clone().tracked();

    assert_eq!(
        tracked_engine.query(&NegativeSquareToString(Variable(0))).await,
        Ok("-9".into())
    );

    assert_eq!(square_ex.0.load(Ordering::Relaxed), 1);
    assert_eq!(negative_square_ex.0.load(Ordering::Relaxed), 1);
    assert_eq!(negative_square_to_string_ex.0.load(Ordering::Relaxed), 1);

    // Step 2: change to -3, diry flag should only propagate upto Square only
    // NegativeSquare is not dirty because its firewall blocks the change
    drop(tracked_engine);

    {
        let mut input_session =
            Arc::get_mut(&mut engine).unwrap().input_session();

        input_session.set_input(Variable(0), -3);
    }

    let tracked_engine = engine.clone().tracked();

    // only Square -> Variable is dirtied
    assert_eq!(tracked_engine.get_dirtied_edges_count(), 1);

    assert_eq!(
        tracked_engine.query(&NegativeSquareToString(Variable(0))).await,
        Ok("-9".into())
    );

    // square_ex should be called again, others should not
    assert_eq!(square_ex.0.load(Ordering::Relaxed), 2);
    assert_eq!(negative_square_ex.0.load(Ordering::Relaxed), 1);
    assert_eq!(negative_square_to_string_ex.0.load(Ordering::Relaxed), 1);

    // only 1 edge should be dirtied in total since firewall hasn't changed
    assert_eq!(tracked_engine.get_dirtied_edges_count(), 1);

    // Step 3: change to 4, now the whole query chain should be re-executed
    drop(tracked_engine);

    {
        let mut input_session =
            Arc::get_mut(&mut engine).unwrap().input_session();

        input_session.set_input(Variable(0), 4);
    }

    let tracked_engine = engine.tracked();

    // for now dirty edge only reach to firewall, but will pass through all
    // the rest queries later
    assert_eq!(tracked_engine.get_dirtied_edges_count(), 1);

    assert_eq!(
        tracked_engine.query(&NegativeSquareToString(Variable(0))).await,
        Ok("-16".into())
    );

    // all executors should be called again
    assert_eq!(square_ex.0.load(Ordering::Relaxed), 3);
    assert_eq!(negative_square_ex.0.load(Ordering::Relaxed), 2);
    assert_eq!(negative_square_to_string_ex.0.load(Ordering::Relaxed), 2);

    // all edges should be dirtied now
    assert_eq!(tracked_engine.get_dirtied_edges_count(), 3);
}
