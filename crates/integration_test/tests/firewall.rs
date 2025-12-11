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

// ============================================================================
// Test: Multiple Firewalls in Chain
// ============================================================================

/// First firewall that computes absolute value
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
pub struct AbsoluteFirewall(pub Variable);

impl Query for AbsoluteFirewall {
    type Value = i64;
}

#[derive(Debug, Default)]
pub struct AbsoluteFirewallExecutor(pub AtomicUsize);

impl<C: Config> Executor<AbsoluteFirewall, C> for AbsoluteFirewallExecutor {
    async fn execute(
        &self,
        query: &AbsoluteFirewall,
        engine: &TrackedEngine<C>,
    ) -> Result<i64, CyclicError> {
        self.0.fetch_add(1, Ordering::Relaxed);
        let value = engine.query(&query.0).await?;
        Ok(value.abs())
    }

    fn execution_style() -> qbice::ExecutionStyle {
        qbice::ExecutionStyle::Firewall
    }
}

/// Second firewall that clamps value to max 100
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
pub struct ClampFirewall(pub Variable);

impl Query for ClampFirewall {
    type Value = i64;
}

#[derive(Debug, Default)]
pub struct ClampFirewallExecutor(pub AtomicUsize);

impl<C: Config> Executor<ClampFirewall, C> for ClampFirewallExecutor {
    async fn execute(
        &self,
        query: &ClampFirewall,
        engine: &TrackedEngine<C>,
    ) -> Result<i64, CyclicError> {
        self.0.fetch_add(1, Ordering::Relaxed);
        let value = engine.query(&AbsoluteFirewall(query.0)).await?;
        Ok(value.min(100))
    }

    fn execution_style() -> qbice::ExecutionStyle {
        qbice::ExecutionStyle::Firewall
    }
}

/// Final consumer that doubles the clamped value
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
pub struct DoubleClampedValue(pub Variable);

impl Query for DoubleClampedValue {
    type Value = i64;
}

#[derive(Debug, Default)]
pub struct DoubleClampedValueExecutor(pub AtomicUsize);

impl<C: Config> Executor<DoubleClampedValue, C> for DoubleClampedValueExecutor {
    async fn execute(
        &self,
        query: &DoubleClampedValue,
        engine: &TrackedEngine<C>,
    ) -> Result<i64, CyclicError> {
        self.0.fetch_add(1, Ordering::Relaxed);
        let value = engine.query(&ClampFirewall(query.0)).await?;
        Ok(value * 2)
    }
}

/// Tests chained firewalls: `Input` -> `AbsoluteFirewall` -> `ClampFirewall` ->
/// `DoubleClampedValue`
///
/// This tests the transitive firewall callees (TFC) mechanism where the final
/// query needs to repair transitive firewall dependencies.
#[tokio::test]
#[traced_test]
async fn chained_firewalls() {
    let mut engine = Engine::<DefaultConfig>::default();

    let abs_ex = Arc::new(AbsoluteFirewallExecutor::default());
    let clamp_ex = Arc::new(ClampFirewallExecutor::default());
    let double_ex = Arc::new(DoubleClampedValueExecutor::default());

    engine.register_executor(abs_ex.clone());
    engine.register_executor(clamp_ex.clone());
    engine.register_executor(double_ex.clone());

    // Step 1: initial value 50
    {
        let mut input_session = engine.input_session();
        input_session.set_input(Variable(0), 50);
    }

    let mut engine = Arc::new(engine);
    let tracked_engine = engine.clone().tracked();

    assert_eq!(
        tracked_engine.query(&DoubleClampedValue(Variable(0))).await,
        Ok(100) // abs(50) = 50, min(50, 100) = 50, 50 * 2 = 100
    );

    assert_eq!(abs_ex.0.load(Ordering::Relaxed), 1);
    assert_eq!(clamp_ex.0.load(Ordering::Relaxed), 1);
    assert_eq!(double_ex.0.load(Ordering::Relaxed), 1);

    // Step 2: change to -50, first firewall blocks (abs(-50) = 50 = abs(50))
    drop(tracked_engine);

    {
        let mut input_session =
            Arc::get_mut(&mut engine).unwrap().input_session();
        input_session.set_input(Variable(0), -50);
    }

    let tracked_engine = engine.clone().tracked();

    // Only edge to first firewall should be dirty
    assert_eq!(tracked_engine.get_dirtied_edges_count(), 1);

    assert_eq!(
        tracked_engine.query(&DoubleClampedValue(Variable(0))).await,
        Ok(100) // same result
    );

    // Only first firewall re-executed, second firewall and final not touched
    assert_eq!(abs_ex.0.load(Ordering::Relaxed), 2);
    assert_eq!(clamp_ex.0.load(Ordering::Relaxed), 1);
    assert_eq!(double_ex.0.load(Ordering::Relaxed), 1);

    // Step 2 dirtied edges should still be 1
    assert_eq!(tracked_engine.get_dirtied_edges_count(), 1);

    // Step 3: change to 80, first firewall changes, second blocks
    // (min(80,100)=80 vs min(50,100)=50)
    drop(tracked_engine);

    {
        let mut input_session =
            Arc::get_mut(&mut engine).unwrap().input_session();
        input_session.set_input(Variable(0), 80);
    }

    let tracked_engine = engine.clone().tracked();

    // Only edge to first firewall should be dirty
    assert_eq!(tracked_engine.get_dirtied_edges_count(), 1);

    assert_eq!(
        tracked_engine.query(&DoubleClampedValue(Variable(0))).await,
        Ok(160) // abs(80) = 80, min(80, 100) = 80, 80 * 2 = 160
    );

    // All executors called again since values changed
    assert_eq!(abs_ex.0.load(Ordering::Relaxed), 3);
    assert_eq!(clamp_ex.0.load(Ordering::Relaxed), 2);
    assert_eq!(double_ex.0.load(Ordering::Relaxed), 2);

    // All of the edges should be dirtied now
    assert_eq!(tracked_engine.get_dirtied_edges_count(), 3);

    // Step 4: change to 150, first firewall changes, second blocks
    // (min(150,100)=100 = previous 80? No, previous was 80)
    // So second firewall: min(150, 100) = 100 vs previous min(80, 100) = 80, so
    // it changes
    drop(tracked_engine);

    {
        let mut input_session =
            Arc::get_mut(&mut engine).unwrap().input_session();
        input_session.set_input(Variable(0), 150);
    }

    let tracked_engine = engine.clone().tracked();

    // Only edge to first firewall should be dirty
    assert_eq!(tracked_engine.get_dirtied_edges_count(), 1);

    assert_eq!(
        tracked_engine.query(&DoubleClampedValue(Variable(0))).await,
        Ok(200) // abs(150) = 150, min(150, 100) = 100, 100 * 2 = 200
    );

    assert_eq!(abs_ex.0.load(Ordering::Relaxed), 4);
    assert_eq!(clamp_ex.0.load(Ordering::Relaxed), 3);
    assert_eq!(double_ex.0.load(Ordering::Relaxed), 3);

    // All edges should be dirtied now
    assert_eq!(tracked_engine.get_dirtied_edges_count(), 3);

    // Step 5: change to 200, first firewall changes (150->200), second blocks
    // (min(200,100)=100 = min(150,100)=100)
    drop(tracked_engine);

    {
        let mut input_session =
            Arc::get_mut(&mut engine).unwrap().input_session();
        input_session.set_input(Variable(0), 200);
    }

    let tracked_engine = engine.tracked();

    // Only edge to first firewall should be dirty
    assert_eq!(tracked_engine.get_dirtied_edges_count(), 1);

    assert_eq!(
        tracked_engine.query(&DoubleClampedValue(Variable(0))).await,
        Ok(200) // abs(200) = 200, min(200, 100) = 100, 100 * 2 = 200 (same!)
    );

    // First firewall changes, second doesn't propagate further
    assert_eq!(abs_ex.0.load(Ordering::Relaxed), 5);
    assert_eq!(clamp_ex.0.load(Ordering::Relaxed), 4);
    assert_eq!(double_ex.0.load(Ordering::Relaxed), 3); // NOT called again!

    // This time only 2 edges dirtied since final firewall blocked propagation
    assert_eq!(tracked_engine.get_dirtied_edges_count(), 2);
}

// ============================================================================
// Test: Diamond Dependency with Firewall
// ============================================================================

/// Query that sums two firewall outputs
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
pub struct SumOfSquares {
    pub a: Variable,
    pub b: Variable,
}

impl Query for SumOfSquares {
    type Value = i64;
}

#[derive(Debug, Default)]
pub struct SumOfSquaresExecutor(pub AtomicUsize);

impl<C: Config> Executor<SumOfSquares, C> for SumOfSquaresExecutor {
    async fn execute(
        &self,
        query: &SumOfSquares,
        engine: &TrackedEngine<C>,
    ) -> Result<i64, CyclicError> {
        self.0.fetch_add(1, Ordering::Relaxed);
        let a_squared = engine.query(&Square(query.a)).await?;
        let b_squared = engine.query(&Square(query.b)).await?;
        Ok(a_squared + b_squared)
    }
}

/// Tests diamond dependency pattern with firewalls
///
/// ```text
///     Var(0)    Var(1)
///       |         |
///       v         v
///   Square(0)  Square(1)  <- both are firewalls
///         \     /
///          \   /
///           v v
///      SumOfSquares
/// ```
#[tokio::test]
#[traced_test]
async fn diamond_dependency_with_firewalls() {
    let mut engine = Engine::<DefaultConfig>::default();

    let square_ex = Arc::new(SquareExecutor::default());
    let sum_ex = Arc::new(SumOfSquaresExecutor::default());

    engine.register_executor(square_ex.clone());
    engine.register_executor(sum_ex.clone());

    // Initial: Variable(0) = 3, Variable(1) = 4
    {
        let mut input_session = engine.input_session();
        input_session.set_input(Variable(0), 3);
        input_session.set_input(Variable(1), 4);
    }

    let mut engine = Arc::new(engine);
    let tracked_engine = engine.clone().tracked();

    assert_eq!(
        tracked_engine
            .query(&SumOfSquares { a: Variable(0), b: Variable(1) })
            .await,
        Ok(25) // 9 + 16 = 25
    );

    assert_eq!(square_ex.0.load(Ordering::Relaxed), 2); // Both squares computed
    assert_eq!(sum_ex.0.load(Ordering::Relaxed), 1);

    // Change Variable(0) to -3, Square(0) output unchanged (9 = 9)
    drop(tracked_engine);

    {
        let mut input_session =
            Arc::get_mut(&mut engine).unwrap().input_session();
        input_session.set_input(Variable(0), -3);
    }

    let tracked_engine = engine.clone().tracked();

    assert_eq!(
        tracked_engine
            .query(&SumOfSquares { a: Variable(0), b: Variable(1) })
            .await,
        Ok(25) // Still 25
    );

    // Square(0) re-executed but SumOfSquares not (firewall blocked)
    assert_eq!(square_ex.0.load(Ordering::Relaxed), 3);
    assert_eq!(sum_ex.0.load(Ordering::Relaxed), 1);

    // Change Variable(1) to -4, Square(1) output unchanged (16 = 16)
    drop(tracked_engine);

    {
        let mut input_session =
            Arc::get_mut(&mut engine).unwrap().input_session();
        input_session.set_input(Variable(1), -4);
    }

    let tracked_engine = engine.clone().tracked();

    assert_eq!(
        tracked_engine
            .query(&SumOfSquares { a: Variable(0), b: Variable(1) })
            .await,
        Ok(25) // Still 25
    );

    // Square(1) re-executed but SumOfSquares not (firewall blocked)
    assert_eq!(square_ex.0.load(Ordering::Relaxed), 4);
    assert_eq!(sum_ex.0.load(Ordering::Relaxed), 1);

    // Change Variable(0) to 5, Square(0) changes (9 -> 25)
    drop(tracked_engine);

    {
        let mut input_session =
            Arc::get_mut(&mut engine).unwrap().input_session();
        input_session.set_input(Variable(0), 5);
    }

    let tracked_engine = engine.tracked();

    assert_eq!(
        tracked_engine
            .query(&SumOfSquares { a: Variable(0), b: Variable(1) })
            .await,
        Ok(41) // 25 + 16 = 41
    );

    // Both firewalls repaired, SumOfSquares recomputed
    assert_eq!(square_ex.0.load(Ordering::Relaxed), 5);
    assert_eq!(sum_ex.0.load(Ordering::Relaxed), 2);
}

// ============================================================================
// Test: Firewall with Multiple Callers
// ============================================================================

/// Another consumer of Square
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
pub struct SquareTimesTwo(pub Variable);

impl Query for SquareTimesTwo {
    type Value = i64;
}

#[derive(Debug, Default)]
pub struct SquareTimesTwoExecutor(pub AtomicUsize);

impl<C: Config> Executor<SquareTimesTwo, C> for SquareTimesTwoExecutor {
    async fn execute(
        &self,
        query: &SquareTimesTwo,
        engine: &TrackedEngine<C>,
    ) -> Result<i64, CyclicError> {
        self.0.fetch_add(1, Ordering::Relaxed);
        let squared = engine.query(&Square(query.0)).await?;
        Ok(squared * 2)
    }
}

/// Another consumer of Square
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
pub struct SquarePlusOne(pub Variable);

impl Query for SquarePlusOne {
    type Value = i64;
}

#[derive(Debug, Default)]
pub struct SquarePlusOneExecutor(pub AtomicUsize);

impl<C: Config> Executor<SquarePlusOne, C> for SquarePlusOneExecutor {
    async fn execute(
        &self,
        query: &SquarePlusOne,
        engine: &TrackedEngine<C>,
    ) -> Result<i64, CyclicError> {
        self.0.fetch_add(1, Ordering::Relaxed);
        let squared = engine.query(&Square(query.0)).await?;
        Ok(squared + 1)
    }
}

/// Tests that firewall properly blocks propagation to multiple callers
///
/// ```text
///        Variable(0)
///             |
///             v
///         Square(0)  <- firewall
///          /    \
///         /      \
///        v        v
/// SquareTimesTwo  SquarePlusOne
/// ```
#[tokio::test]
#[traced_test]
async fn firewall_multiple_callers() {
    let mut engine = Engine::<DefaultConfig>::default();

    let square_ex = Arc::new(SquareExecutor::default());
    let times_two_ex = Arc::new(SquareTimesTwoExecutor::default());
    let plus_one_ex = Arc::new(SquarePlusOneExecutor::default());

    engine.register_executor(square_ex.clone());
    engine.register_executor(times_two_ex.clone());
    engine.register_executor(plus_one_ex.clone());

    // Initial: Variable(0) = 3
    {
        let mut input_session = engine.input_session();
        input_session.set_input(Variable(0), 3);
    }

    let mut engine = Arc::new(engine);
    let tracked_engine = engine.clone().tracked();

    // Query both consumers
    assert_eq!(
        tracked_engine.query(&SquareTimesTwo(Variable(0))).await,
        Ok(18) // 9 * 2 = 18
    );
    assert_eq!(
        tracked_engine.query(&SquarePlusOne(Variable(0))).await,
        Ok(10) // 9 + 1 = 10
    );

    assert_eq!(square_ex.0.load(Ordering::Relaxed), 1);
    assert_eq!(times_two_ex.0.load(Ordering::Relaxed), 1);
    assert_eq!(plus_one_ex.0.load(Ordering::Relaxed), 1);

    // Change to -3, Square output unchanged
    drop(tracked_engine);

    {
        let mut input_session =
            Arc::get_mut(&mut engine).unwrap().input_session();
        input_session.set_input(Variable(0), -3);
    }

    let tracked_engine = engine.clone().tracked();

    // Both queries should return same values without re-executing
    assert_eq!(
        tracked_engine.query(&SquareTimesTwo(Variable(0))).await,
        Ok(18)
    );
    assert_eq!(tracked_engine.query(&SquarePlusOne(Variable(0))).await, Ok(10));

    // Square re-executed, but neither consumer re-executed
    assert_eq!(square_ex.0.load(Ordering::Relaxed), 2);
    assert_eq!(times_two_ex.0.load(Ordering::Relaxed), 1);
    assert_eq!(plus_one_ex.0.load(Ordering::Relaxed), 1);

    // Change to 5, Square output changes
    drop(tracked_engine);

    {
        let mut input_session =
            Arc::get_mut(&mut engine).unwrap().input_session();
        input_session.set_input(Variable(0), 5);
    }

    let tracked_engine = engine.tracked();

    assert_eq!(
        tracked_engine.query(&SquareTimesTwo(Variable(0))).await,
        Ok(50) // 25 * 2 = 50
    );
    assert_eq!(
        tracked_engine.query(&SquarePlusOne(Variable(0))).await,
        Ok(26) // 25 + 1 = 26
    );

    // All re-executed since firewall changed
    assert_eq!(square_ex.0.load(Ordering::Relaxed), 3);
    assert_eq!(times_two_ex.0.load(Ordering::Relaxed), 2);
    assert_eq!(plus_one_ex.0.load(Ordering::Relaxed), 2);
}

// ============================================================================
// Test: Firewall with Conditional Dependency
// ============================================================================

/// A query that conditionally depends on a firewall
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
pub struct ConditionalSquare {
    pub condition: Variable,
    pub value: Variable,
}

impl Query for ConditionalSquare {
    type Value = i64;
}

#[derive(Debug, Default)]
pub struct ConditionalSquareExecutor(pub AtomicUsize);

impl<C: Config> Executor<ConditionalSquare, C> for ConditionalSquareExecutor {
    async fn execute(
        &self,
        query: &ConditionalSquare,
        engine: &TrackedEngine<C>,
    ) -> Result<i64, CyclicError> {
        self.0.fetch_add(1, Ordering::Relaxed);
        let condition = engine.query(&query.condition).await?;

        if condition > 0 {
            // Only depend on Square when condition > 0
            engine.query(&Square(query.value)).await
        } else {
            // Return -1 without querying Square
            Ok(-1)
        }
    }
}

/// Tests conditional dependency on firewall
///
/// When condition changes behavior, dependencies change.
#[tokio::test]
#[traced_test]
async fn firewall_conditional_dependency() {
    let mut engine = Engine::<DefaultConfig>::default();

    let square_ex = Arc::new(SquareExecutor::default());
    let conditional_ex = Arc::new(ConditionalSquareExecutor::default());

    engine.register_executor(square_ex.clone());
    engine.register_executor(conditional_ex.clone());

    // Initial: condition = 1 (true), value = 3
    {
        let mut input_session = engine.input_session();
        input_session.set_input(Variable(0), 1); // condition
        input_session.set_input(Variable(1), 3); // value
    }

    let mut engine = Arc::new(engine);
    let tracked_engine = engine.clone().tracked();

    assert_eq!(
        tracked_engine
            .query(&ConditionalSquare {
                condition: Variable(0),
                value: Variable(1)
            })
            .await,
        Ok(9) // Square(3) = 9
    );

    assert_eq!(square_ex.0.load(Ordering::Relaxed), 1);
    assert_eq!(conditional_ex.0.load(Ordering::Relaxed), 1);

    // Change value to -3, Square output unchanged, ConditionalSquare unchanged
    drop(tracked_engine);

    {
        let mut input_session =
            Arc::get_mut(&mut engine).unwrap().input_session();
        input_session.set_input(Variable(1), -3);
    }

    let tracked_engine = engine.clone().tracked();

    assert_eq!(
        tracked_engine
            .query(&ConditionalSquare {
                condition: Variable(0),
                value: Variable(1)
            })
            .await,
        Ok(9) // Still 9
    );

    // Square re-executed but ConditionalSquare not (firewall blocked)
    assert_eq!(square_ex.0.load(Ordering::Relaxed), 2);
    assert_eq!(conditional_ex.0.load(Ordering::Relaxed), 1);

    // Change condition to 0 (false)
    drop(tracked_engine);

    {
        let mut input_session =
            Arc::get_mut(&mut engine).unwrap().input_session();
        input_session.set_input(Variable(0), 0);
        input_session.set_input(Variable(1), 4);
    }

    let tracked_engine = engine.clone().tracked();

    // total of 2 edges are dirtied: condition and value
    assert_eq!(tracked_engine.get_dirtied_edges_count(), 2);

    assert_eq!(
        tracked_engine
            .query(&ConditionalSquare {
                condition: Variable(0),
                value: Variable(1)
            })
            .await,
        Ok(-1) // Condition is false, return -1
    );

    // the square_ex is still re-executed because the ConditionalSquare has
    // TFC to it and thus have always be repaired
    assert_eq!(square_ex.0.load(Ordering::Relaxed), 3);
    // ConditionalSquare re-executed because condition changed
    assert_eq!(conditional_ex.0.load(Ordering::Relaxed), 2);

    // all of the edges should be dirtied now
    assert_eq!(tracked_engine.get_dirtied_edges_count(), 3);

    // Change value while condition is false
    drop(tracked_engine);

    {
        let mut input_session =
            Arc::get_mut(&mut engine).unwrap().input_session();
        input_session.set_input(Variable(1), 100);
    }

    let tracked_engine = engine.tracked();

    // only edge Square -> Variable(1) is dirtied
    assert_eq!(tracked_engine.get_dirtied_edges_count(), 1);

    assert_eq!(
        tracked_engine
            .query(&ConditionalSquare {
                condition: Variable(0),
                value: Variable(1)
            })
            .await,
        Ok(-1) // Still -1, condition is still false
    );

    // no new edges dirtied
    assert_eq!(tracked_engine.get_dirtied_edges_count(), 1);

    // Square not called,
    assert_eq!(square_ex.0.load(Ordering::Relaxed), 3);
}

// ============================================================================
// Test: Firewall propagation timing
// ============================================================================

/// Tests that dirty propagation happens correctly when firewall value changes
#[tokio::test]
#[traced_test]
async fn firewall_dirty_propagation_on_change() {
    let mut engine = Engine::<DefaultConfig>::default();

    let square_ex = Arc::new(SquareExecutor::default());
    let negative_square_ex = Arc::new(NegativeSquareExecutor::default());

    engine.register_executor(square_ex.clone());
    engine.register_executor(negative_square_ex.clone());

    // Initial: 2
    {
        let mut input_session = engine.input_session();
        input_session.set_input(Variable(0), 2);
    }

    let mut engine = Arc::new(engine);
    let tracked_engine = engine.clone().tracked();

    assert_eq!(
        tracked_engine.query(&NegativeSquare(Variable(0))).await,
        Ok(-4) // -(2^2) = -4
    );

    assert_eq!(square_ex.0.load(Ordering::Relaxed), 1);
    assert_eq!(negative_square_ex.0.load(Ordering::Relaxed), 1);

    // Initially only 0 edges dirtied
    assert_eq!(tracked_engine.get_dirtied_edges_count(), 0);

    drop(tracked_engine);

    // Change to 3 - firewall output changes
    {
        let mut input_session =
            Arc::get_mut(&mut engine).unwrap().input_session();
        input_session.set_input(Variable(0), 3);
    }

    let tracked_engine = engine.clone().tracked();

    // Before querying, only edge to firewall is dirty
    assert_eq!(tracked_engine.get_dirtied_edges_count(), 1);

    assert_eq!(
        tracked_engine.query(&NegativeSquare(Variable(0))).await,
        Ok(-9) // -(3^2) = -9
    );

    // After querying, firewall changed so propagation happened
    assert_eq!(tracked_engine.get_dirtied_edges_count(), 2);

    assert_eq!(square_ex.0.load(Ordering::Relaxed), 2);
    assert_eq!(negative_square_ex.0.load(Ordering::Relaxed), 2);

    drop(tracked_engine);

    // Change to -3 - firewall output unchanged
    {
        let mut input_session =
            Arc::get_mut(&mut engine).unwrap().input_session();
        input_session.set_input(Variable(0), -3);
    }

    let tracked_engine = engine.tracked();

    assert_eq!(
        tracked_engine.query(&NegativeSquare(Variable(0))).await,
        Ok(-9) // Still -9
    );

    // Only 1 edge dirtied because firewall blocked propagation
    assert_eq!(tracked_engine.get_dirtied_edges_count(), 1);

    assert_eq!(square_ex.0.load(Ordering::Relaxed), 3);
    assert_eq!(negative_square_ex.0.load(Ordering::Relaxed), 2);
}
