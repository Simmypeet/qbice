#![allow(missing_docs)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_const_for_fn)]

use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use qbice::{
    Engine, Executor, Identifiable, Query, StableHash, TrackedEngine,
    config::{Config, DefaultConfig},
};
use qbice_integration_test::Variable;

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
pub struct SlowSquare(pub Variable);

impl Query for SlowSquare {
    type Value = i64;
}

#[derive(Debug, Default)]
pub struct SlowSquareExecutor(pub AtomicUsize);

impl<C: Config> Executor<SlowSquare, C> for SlowSquareExecutor {
    async fn execute(
        &self,
        query: &SlowSquare,
        engine: &TrackedEngine<C>,
    ) -> Result<i64, qbice::executor::CyclicError> {
        // increment computation count
        self.0.fetch_add(1, Ordering::SeqCst);

        let var_value = engine.query(&query.0).await?;

        // Introduce an artificial delay to simulate a slow computation.
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;

        Ok(var_value * var_value)
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
    StableHash,
    Identifiable,
)]
pub struct VariableTarget;

impl Query for VariableTarget {
    type Value = Arc<[Variable]>;
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
pub struct CollectDoubledSquareVariables;

impl Query for CollectDoubledSquareVariables {
    type Value = Arc<HashMap<Variable, i64>>;
}

#[derive(Debug, Default)]
pub struct CollectDoubledSquareVariablesExecutor(pub AtomicUsize);

impl<C: Config> Executor<CollectDoubledSquareVariables, C>
    for CollectDoubledSquareVariablesExecutor
{
    async fn execute(
        &self,
        _query: &CollectDoubledSquareVariables,
        engine: &TrackedEngine<C>,
    ) -> Result<Arc<HashMap<Variable, i64>>, qbice::executor::CyclicError> {
        // increment computation count
        self.0.fetch_add(1, Ordering::SeqCst);

        let target_vars = engine.query(&VariableTarget).await?;

        let mut join_handles = Vec::new();
        for var in target_vars.iter().copied() {
            let engine = engine.clone();

            join_handles.push(tokio::spawn(async move {
                let square = engine.query(&SlowSquare(var)).await?;

                // simulating long computation
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;

                Ok((var, square * 2))
            }));
        }

        let mut result = HashMap::new();
        for handle in join_handles {
            let (var, doubled_square) = handle.await.unwrap()?;
            result.insert(var, doubled_square);
        }

        Ok(Arc::new(result))
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
    StableHash,
    Identifiable,
)]
pub struct DoubleSquare(pub Variable);

impl Query for DoubleSquare {
    type Value = Option<i64>;
}

#[derive(Debug, Default)]
pub struct DoubleSquareExecutor(pub AtomicUsize);

impl<C: Config> Executor<DoubleSquare, C> for DoubleSquareExecutor {
    async fn execute(
        &self,
        query: &DoubleSquare,
        engine: &TrackedEngine<C>,
    ) -> Result<Option<i64>, qbice::executor::CyclicError> {
        // increment computation count
        self.0.fetch_add(1, Ordering::SeqCst);

        let map = engine.query(&CollectDoubledSquareVariables).await?;

        Ok(map.get(&query.0).copied())
    }

    fn execution_style() -> qbice::ExecutionStyle {
        qbice::ExecutionStyle::Projection
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
    StableHash,
    Identifiable,
)]
pub struct SumAllDoubleSquares;

impl Query for SumAllDoubleSquares {
    type Value = i64;
}

#[derive(Debug, Default)]
pub struct SumAllDoubleSquaresExecutor(pub AtomicUsize);

impl<C: Config> Executor<SumAllDoubleSquares, C>
    for SumAllDoubleSquaresExecutor
{
    async fn execute(
        &self,
        _query: &SumAllDoubleSquares,
        engine: &TrackedEngine<C>,
    ) -> Result<i64, qbice::executor::CyclicError> {
        // increment computation count
        self.0.fetch_add(1, Ordering::SeqCst);

        let target_vars = engine.query(&VariableTarget).await?;

        let mut handles = Vec::new();

        for var in target_vars.iter().copied() {
            let engine = engine.clone();

            handles.push(tokio::spawn(async move {
                let double_square = engine.query(&DoubleSquare(var)).await?;
                Ok(double_square.unwrap_or(0))
            }));
        }

        let mut sum = 0;
        for handle in handles {
            sum += handle.await.unwrap()?;
        }

        Ok(sum)
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn double_square_summing() {
    let mut engine = Engine::<DefaultConfig>::new();

    let slow_square_ex = Arc::new(SlowSquareExecutor::default());
    let collect_doubled_ex =
        Arc::new(CollectDoubledSquareVariablesExecutor::default());
    let double_square_ex = Arc::new(DoubleSquareExecutor::default());
    let sum_all_ex = Arc::new(SumAllDoubleSquaresExecutor::default());

    engine.register_executor(slow_square_ex.clone());
    engine.register_executor(collect_doubled_ex.clone());
    engine.register_executor(double_square_ex.clone());
    engine.register_executor(sum_all_ex.clone());

    // initialze variables
    {
        let mut input_session = engine.input_session();
        let mut target_vars = Vec::new();

        for i in 0..8 {
            input_session.set_input(Variable(i), i.cast_signed() + 1);
            target_vars.push(Variable(i));
        }

        input_session.set_input(VariableTarget, Arc::from(target_vars));
    }

    let mut engine = Arc::new(engine);
    let tracked_engine = engine.clone().tracked();

    // 1^2 = 1
    // 2^2 = 4
    // 3^2 = 9
    // 4^2 = 16
    // 5^2 = 25
    // 6^2 = 36
    // 7^2 = 49
    // 8^2 = 64
    // total * 2 = 408
    let sum_result = tracked_engine
        .query(&SumAllDoubleSquares)
        .await
        .expect("SumAllDoubleSquares query failed");

    assert_eq!(sum_result, 408);

    drop(tracked_engine);

    // Verify executor call counts
    assert_eq!(slow_square_ex.0.load(Ordering::SeqCst), 8);
    assert_eq!(collect_doubled_ex.0.load(Ordering::SeqCst), 1);
    assert_eq!(double_square_ex.0.load(Ordering::SeqCst), 8);
    assert_eq!(sum_all_ex.0.load(Ordering::SeqCst), 1);

    // Change the `Variable(0)` to 10
    {
        let mut input_session =
            Arc::get_mut(&mut engine).unwrap().input_session();
        input_session.set_input(Variable(0), 10);
    }

    let tracked_engine = engine.clone().tracked();

    // there shouldd be only two dirtied edges:
    // CollectDoubledSquareVariables -> SlowSquare -> Variable(0)
    assert_eq!(tracked_engine.get_dirtied_edges_count(), 2);

    drop(tracked_engine);

    // 10^2 = 100
    // 2^2 = 4
    // 3^2 = 9
    // 4^2 = 16
    // 5^2 = 25
    // 6^2 = 36
    // 7^2 = 49
    // 8^2 = 64
    // total * 2 = 408 - 2 + 200 = 606
    let sum_result = engine
        .clone()
        .tracked()
        .query(&SumAllDoubleSquares)
        .await
        .expect("SumAllDoubleSquares query failed after input change");

    assert_eq!(sum_result, 606);

    // should have 11 dirtied edges:
    // 2 from the previous assertion +
    // each of the 8 DoubleSquare -> CollectDoubledSquareVariables +
    // DoubleSquare(Variable(0)) -> SumAllDoubleSquares
    assert_eq!(engine.tracked().get_dirtied_edges_count(), 11);
}
