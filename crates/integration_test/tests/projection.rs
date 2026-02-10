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
    Decode, Encode, Executor, Identifiable, Query, StableHash, TrackedEngine,
    config::Config,
};
use qbice_integration_test::{Variable, create_test_engine};
use tempfile::tempdir;

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
    Encode,
    Decode,
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
    ) -> i64 {
        // increment computation count
        self.0.fetch_add(1, Ordering::SeqCst);

        let var_value = engine.query(&query.0).await;

        // Introduce an artificial delay to simulate a slow computation.
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;

        var_value * var_value
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
    Encode,
    Decode,
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
    Encode,
    Decode,
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
    ) -> Arc<HashMap<Variable, i64>> {
        // increment computation count
        self.0.fetch_add(1, Ordering::SeqCst);

        let target_vars = engine.query(&VariableTarget).await;

        let mut join_handles = Vec::new();
        for var in target_vars.iter().copied() {
            let engine = engine.clone();

            join_handles.push(tokio::spawn(async move {
                let square = engine.query(&SlowSquare(var)).await;

                // simulating long computation
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;

                (var, square * 2)
            }));
        }

        let mut result = HashMap::new();
        for handle in join_handles {
            let (var, doubled_square) = handle.await.unwrap();
            result.insert(var, doubled_square);
        }

        Arc::new(result)
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
    Encode,
    Decode,
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
    ) -> Option<i64> {
        // increment computation count
        self.0.fetch_add(1, Ordering::SeqCst);

        let map = engine.query(&CollectDoubledSquareVariables).await;

        map.get(&query.0).copied()
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
    Encode,
    Decode,
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
    ) -> i64 {
        // increment computation count
        self.0.fetch_add(1, Ordering::SeqCst);

        let target_vars = engine.query(&VariableTarget).await;

        let mut handles = Vec::new();

        for var in target_vars.iter().copied() {
            let engine = engine.clone();

            handles.push(tokio::spawn(async move {
                let double_square = engine.query(&DoubleSquare(var)).await;
                double_square.unwrap_or(0)
            }));
        }

        let mut sum = 0;
        for handle in handles {
            sum += handle.await.unwrap();
        }

        sum
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn double_square_summing() {
    let tempdir = tempdir().unwrap();
    let mut engine = create_test_engine(&tempdir).await;

    let slow_square_ex = Arc::new(SlowSquareExecutor::default());
    let collect_doubled_ex =
        Arc::new(CollectDoubledSquareVariablesExecutor::default());
    let double_square_ex = Arc::new(DoubleSquareExecutor::default());
    let sum_all_ex = Arc::new(SumAllDoubleSquaresExecutor::default());

    engine.register_executor(slow_square_ex.clone());
    engine.register_executor(collect_doubled_ex.clone());
    engine.register_executor(double_square_ex.clone());
    engine.register_executor(sum_all_ex.clone());

    let engine = Arc::new(engine);

    // initialze variables
    {
        let mut input_session = engine.input_session().await;
        let mut target_vars = Vec::new();

        for i in 0..8 {
            input_session.set_input(Variable(i), i.cast_signed() + 1).await;
            target_vars.push(Variable(i));
        }

        input_session.set_input(VariableTarget, Arc::from(target_vars)).await;
    }

    let tracked_engine = engine.clone().tracked().await;

    // 1^2 = 1
    // 2^2 = 4
    // 3^2 = 9
    // 4^2 = 16
    // 5^2 = 25
    // 6^2 = 36
    // 7^2 = 49
    // 8^2 = 64
    // total * 2 = 408
    let sum_result = tracked_engine.query(&SumAllDoubleSquares).await;

    assert_eq!(sum_result, 408);

    drop(tracked_engine);

    // Verify executor call counts
    assert_eq!(slow_square_ex.0.load(Ordering::SeqCst), 8);
    assert_eq!(collect_doubled_ex.0.load(Ordering::SeqCst), 1);
    assert_eq!(double_square_ex.0.load(Ordering::SeqCst), 8);
    assert_eq!(sum_all_ex.0.load(Ordering::SeqCst), 1);

    // Change the `Variable(0)` to 10
    {
        let mut input_session = engine.input_session().await;
        input_session.set_input(Variable(0), 10).await;
    }

    let tracked_engine = engine.clone().tracked().await;

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
    let sum_result =
        engine.clone().tracked().await.query(&SumAllDoubleSquares).await;

    assert_eq!(sum_result, 606);

    // should have 11 dirtied edges:
    // 2 from the previous assertion +
    // each of the 8 DoubleSquare -> CollectDoubledSquareVariables +
    // DoubleSquare(Variable(0)) -> SumAllDoubleSquares
    assert_eq!(engine.tracked().await.get_dirtied_edges_count(), 11);
}

// ============================================================================
// Test: Multiple Projections from Single Firewall
// ============================================================================
//
// Graph structure:
//   Variable(0..3) -> SlowSquare -> CollectDoubledSquareVariables (Firewall)
//                                          |
//                    +---------------------+---------------------+
//                    |                     |                     |
//                    v                     v                     v
//            DoubleSquare(0)       DoubleSquare(1)       DoubleSquare(2)
//            (Projection)         (Projection)          (Projection)
//
// When Variable(0) changes, only the edge to SlowSquare(0) and firewall
// should be dirtied. The projections are NOT dirtied during initial
// propagation - they only get invoked during backward propagation when
// the firewall recomputes.

#[tokio::test(flavor = "multi_thread")]
async fn multiple_projections_single_firewall() {
    let tempdir = tempdir().unwrap();
    let mut engine = create_test_engine(&tempdir).await;

    let slow_square_ex = Arc::new(SlowSquareExecutor::default());
    let collect_doubled_ex =
        Arc::new(CollectDoubledSquareVariablesExecutor::default());
    let double_square_ex = Arc::new(DoubleSquareExecutor::default());

    engine.register_executor(slow_square_ex.clone());
    engine.register_executor(collect_doubled_ex.clone());
    engine.register_executor(double_square_ex.clone());

    let engine = Arc::new(engine);

    // Initialize 4 variables
    {
        let mut input_session = engine.input_session().await;
        let mut target_vars = Vec::new();

        for i in 0..4 {
            input_session.set_input(Variable(i), i.cast_signed() + 1).await;
            target_vars.push(Variable(i));
        }

        input_session.set_input(VariableTarget, Arc::from(target_vars)).await;
    }

    // Query all three projections
    {
        let tracked = engine.clone().tracked().await;
        let v0 = tracked.query(&DoubleSquare(Variable(0))).await;
        let v1 = tracked.query(&DoubleSquare(Variable(1))).await;
        let v2 = tracked.query(&DoubleSquare(Variable(2))).await;

        // 1^2 * 2 = 2, 2^2 * 2 = 8, 3^2 * 2 = 18
        assert_eq!(v0, Some(2));
        assert_eq!(v1, Some(8));
        assert_eq!(v2, Some(18));
    }

    // Verify initial execution counts
    assert_eq!(slow_square_ex.0.load(Ordering::SeqCst), 4);
    assert_eq!(collect_doubled_ex.0.load(Ordering::SeqCst), 1);
    assert_eq!(double_square_ex.0.load(Ordering::SeqCst), 3);

    // Change Variable(0) only
    {
        let mut input_session = engine.input_session().await;
        input_session.set_input(Variable(0), 10).await;
    }

    // Check dirtied edges - should only be 2:
    // CollectDoubledSquareVariables -> SlowSquare(0) -> Variable(0)
    let tracked = engine.clone().tracked().await;
    assert_eq!(tracked.get_dirtied_edges_count(), 2);
    drop(tracked);

    // Re-query projection for Variable(1) - this triggers firewall repair
    // since the firewall is dirty (due to SlowSquare(0) -> Variable(0) being
    // dirty)
    {
        let tracked = engine.clone().tracked().await;
        let v1 = tracked.query(&DoubleSquare(Variable(1))).await;
        assert_eq!(v1, Some(8));
    }

    // Firewall was recomputed (since it's dirty)
    // SlowSquare(0) recomputed because Variable(0) changed
    // Note: Other SlowSquare queries may also be re-queried during firewall
    // recomputation but should hit cache if unchanged
    assert_eq!(collect_doubled_ex.0.load(Ordering::SeqCst), 2); // +1 for firewall

    // DoubleSquare(1) was the original caller (queried above)
    assert_eq!(double_square_ex.0.load(Ordering::SeqCst), 4); // +1

    // in total there should be 5 dirtied edges now:
    // - Original 2 from previous assertion
    // - Each of the 3 DoubleSquare -> CollectDoubledSquareVariables edges
    let tracked = engine.clone().tracked().await;
    assert_eq!(tracked.get_dirtied_edges_count(), 5);
    drop(tracked);

    // Query the changed projection
    {
        let tracked = engine.clone().tracked().await;
        let v0 = tracked.query(&DoubleSquare(Variable(0))).await;
        // 10^2 * 2 = 200
        assert_eq!(v0, Some(200));
    }

    // The firewall was recomputed since Variable(0) changed, the
    // DoubleSquare(0) depends on it, so +1 recomputation
    assert_eq!(double_square_ex.0.load(Ordering::SeqCst), 5);
}

// ============================================================================
// Test: Chained Projections
// ============================================================================
//
// Graph structure:
//   Variable(0) -> Firewall (produces HashMap)
//                      |
//                      v
//              Projection1 (extracts and transforms)
//                      |
//                      v
//              Projection2 (further transforms)
//                      |
//                      v
//               NormalQuery (consumes)
//
// This tests that dirty propagation stops at projection boundaries and
// the chain of projections correctly filters changes.

/// A firewall query that produces a simple map from Variable to its value.
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
    Encode,
    Decode,
)]
pub struct SimpleFirewall;

impl Query for SimpleFirewall {
    type Value = Arc<HashMap<u64, i64>>;
}

#[derive(Debug, Default)]
pub struct SimpleFirewallExecutor(pub AtomicUsize);

impl<C: Config> Executor<SimpleFirewall, C> for SimpleFirewallExecutor {
    async fn execute(
        &self,
        _query: &SimpleFirewall,
        engine: &TrackedEngine<C>,
    ) -> Arc<HashMap<u64, i64>> {
        self.0.fetch_add(1, Ordering::SeqCst);

        let mut map = HashMap::new();

        for i in 0..100 {
            let val = engine.query(&Variable(i)).await;
            map.insert(i, val);
        }

        Arc::new(map)
    }

    fn execution_style() -> qbice::ExecutionStyle {
        qbice::ExecutionStyle::Firewall
    }
}

/// First level projection - extracts a single key and doubles it.
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
    Encode,
    Decode,
)]
pub struct ProjectionLevel1(pub u64);

impl Query for ProjectionLevel1 {
    type Value = i64;
}

#[derive(Debug, Default)]
pub struct ProjectionLevel1Executor(pub AtomicUsize);

impl<C: Config> Executor<ProjectionLevel1, C> for ProjectionLevel1Executor {
    async fn execute(
        &self,
        query: &ProjectionLevel1,
        engine: &TrackedEngine<C>,
    ) -> i64 {
        self.0.fetch_add(1, Ordering::SeqCst);

        let map = engine.query(&SimpleFirewall).await;
        let val = map.get(&query.0).copied().unwrap_or(0);

        val * 2
    }

    fn execution_style() -> qbice::ExecutionStyle {
        qbice::ExecutionStyle::Projection
    }
}

// ============================================================================
// Test: Diamond Pattern with Projections
// ============================================================================
//
// Graph structure:
//              SimpleFirewall
//              /           \
//             v             v
//     ProjectionLevel1(0)  ProjectionLevel1(1)
//              \           /
//               v         v
//              DiamondCombiner
//
// Tests that both projections correctly filter changes and the combiner
// only recomputes when necessary.

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
    Encode,
    Decode,
)]
pub struct DiamondCombiner;

impl Query for DiamondCombiner {
    type Value = i64;
}

#[derive(Debug, Default)]
pub struct DiamondCombinerExecutor(pub AtomicUsize);

impl<C: Config> Executor<DiamondCombiner, C> for DiamondCombinerExecutor {
    async fn execute(
        &self,
        _query: &DiamondCombiner,
        engine: &TrackedEngine<C>,
    ) -> i64 {
        self.0.fetch_add(1, Ordering::SeqCst);

        let p0 = engine.query(&ProjectionLevel1(0)).await;
        let p1 = engine.query(&ProjectionLevel1(1)).await;

        p0 + p1
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn diamond_projection_pattern() {
    let tempdir = tempdir().unwrap();
    let mut engine = create_test_engine(&tempdir).await;

    let firewall_ex = Arc::new(SimpleFirewallExecutor::default());
    let proj1_ex = Arc::new(ProjectionLevel1Executor::default());
    let combiner_ex = Arc::new(DiamondCombinerExecutor::default());

    engine.register_executor(firewall_ex.clone());
    engine.register_executor(proj1_ex.clone());
    engine.register_executor(combiner_ex.clone());

    let engine = Arc::new(engine);

    // Initialize variables
    {
        let mut input_session = engine.input_session().await;
        for i in 0..100 {
            input_session.set_input(Variable(i), i.cast_signed() + 1).await;
        }
    }

    // Query the combiner
    {
        let tracked = engine.clone().tracked().await;
        // Variable(0)=1 -> proj1=2, Variable(1)=2 -> proj1=4, combiner=6
        let result = tracked.query(&DiamondCombiner).await;
        assert_eq!(result, 6);
    }

    assert_eq!(firewall_ex.0.load(Ordering::SeqCst), 1);
    assert_eq!(proj1_ex.0.load(Ordering::SeqCst), 2);
    assert_eq!(combiner_ex.0.load(Ordering::SeqCst), 1);

    // Change Variable(2) - neither projection depends on this directly
    // but firewall does
    {
        let mut input_session = engine.input_session().await;
        input_session.set_input(Variable(2), 100).await;
    }

    // Only 1 dirtied edge: Firewall -> Variable(2)
    let tracked = engine.clone().tracked().await;
    assert_eq!(tracked.get_dirtied_edges_count(), 1);
    drop(tracked);

    // Re-query combiner
    {
        let tracked = engine.clone().tracked().await;
        let result = tracked.query(&DiamondCombiner).await;
        // Same result since proj1(0) and proj1(1) didn't change
        assert_eq!(result, 6);
    }

    // Firewall recomputed, and backward prop invokes all projections that
    // depend on the firewall
    assert_eq!(firewall_ex.0.load(Ordering::SeqCst), 2); // +1
    // Both projections are invoked via backward prop when firewall recomputes
    assert_eq!(proj1_ex.0.load(Ordering::SeqCst), 4); // +2 (backward prop)
    // Combiner is NOT recomputed because projections filtered the change -
    // Variable(2) changed but proj1(0) and proj1(1) don't depend on it!
    assert_eq!(combiner_ex.0.load(Ordering::SeqCst), 1); // unchanged - filtered!

    // Check dirtied edges - should be 3 now:
    // - Original 1 from previous assertion
    // - ProjectionLevel1(0) -> SimpleFirewall
    // - ProjectionLevel1(1) -> SimpleFirewall
    let tracked = engine.clone().tracked().await;
    assert_eq!(tracked.get_dirtied_edges_count(), 3);
    drop(tracked);

    // Now change Variable(0) which DOES affect proj1(0)
    {
        let mut input_session = engine.input_session().await;
        input_session.set_input(Variable(0), 50).await;
    }

    // Only 1 dirtied edge: Firewall -> Variable(0)
    let tracked = engine.clone().tracked().await;
    assert_eq!(tracked.get_dirtied_edges_count(), 1);
    drop(tracked);

    // Re-query combiner
    {
        let tracked = engine.clone().tracked().await;
        // Variable(0)=50 -> proj1=100, Variable(1)=2 -> proj1=4, combiner=104
        let result = tracked.query(&DiamondCombiner).await;
        assert_eq!(result, 104);
    }

    assert_eq!(firewall_ex.0.load(Ordering::SeqCst), 3); // +1
    assert_eq!(proj1_ex.0.load(Ordering::SeqCst), 6); // +2 (backward prop)
    assert_eq!(combiner_ex.0.load(Ordering::SeqCst), 2); // +1 (value changed)

    // Check dirtied edges - should be 4 now:
    // - Original 1 from previous assertion
    // - ProjectionLevel1(0) -> SimpleFirewall
    // - ProjectionLevel1(1) -> SimpleFirewall
    // - DiamondCombiner -> ProjectionLevel1(0)
    let tracked = engine.clone().tracked().await;
    assert_eq!(tracked.get_dirtied_edges_count(), 4);
    drop(tracked);
}

// ============================================================================
// Test: Diamond Pattern with All Projections
// ============================================================================
//
// Graph structure:
//              SimpleFirewall
//              /           \
//             v             v
//     ProjectionLevel1(0)  ProjectionLevel1(1)
//        (Projection)        (Projection)
//              \           /
//               v         v
//        DiamondCombinerProjection (PROJECTION)
//                   |
//                   v
//           DiamondFinalConsumer
//
// Tests a more complex projection chain where the combiner itself is a
// projection, forming a multi-level projection chain. When an unrelated
// variable changes, all projections should recompute but produce same
// outputs, stopping propagation at the combiner.

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
    Encode,
    Decode,
)]
pub struct DiamondCombinerProjection;

impl Query for DiamondCombinerProjection {
    type Value = i64;
}

#[derive(Debug, Default)]
pub struct DiamondCombinerProjectionExecutor(pub AtomicUsize);

impl<C: Config> Executor<DiamondCombinerProjection, C>
    for DiamondCombinerProjectionExecutor
{
    async fn execute(
        &self,
        _query: &DiamondCombinerProjection,
        engine: &TrackedEngine<C>,
    ) -> i64 {
        self.0.fetch_add(1, Ordering::SeqCst);

        let p0 = engine.query(&ProjectionLevel1(0)).await;
        let p1 = engine.query(&ProjectionLevel1(1)).await;

        p0 + p1
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
    Encode,
    Decode,
)]
pub struct DiamondFinalConsumer;

impl Query for DiamondFinalConsumer {
    type Value = i64;
}

#[derive(Debug, Default)]
pub struct DiamondFinalConsumerExecutor(pub AtomicUsize);

impl<C: Config> Executor<DiamondFinalConsumer, C>
    for DiamondFinalConsumerExecutor
{
    async fn execute(
        &self,
        _query: &DiamondFinalConsumer,
        engine: &TrackedEngine<C>,
    ) -> i64 {
        self.0.fetch_add(1, Ordering::SeqCst);

        let combined = engine.query(&DiamondCombinerProjection).await;
        combined * 100
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn diamond_all_projections_pattern() {
    let tempdir = tempdir().unwrap();
    let mut engine = create_test_engine(&tempdir).await;

    let firewall_ex = Arc::new(SimpleFirewallExecutor::default());
    let proj1_ex = Arc::new(ProjectionLevel1Executor::default());
    let combiner_proj_ex =
        Arc::new(DiamondCombinerProjectionExecutor::default());
    let consumer_ex = Arc::new(DiamondFinalConsumerExecutor::default());

    engine.register_executor(firewall_ex.clone());
    engine.register_executor(proj1_ex.clone());
    engine.register_executor(combiner_proj_ex.clone());
    engine.register_executor(consumer_ex.clone());

    let engine = Arc::new(engine);

    // Initialize variables
    {
        let mut input_session = engine.input_session().await;
        for i in 0..100 {
            input_session.set_input(Variable(i), i.cast_signed() + 1).await;
        }
    }

    // =========================================================================
    // Test Case 1: Initial query through the diamond
    // Variable(0)=1 -> proj1(0)=2, Variable(1)=2 -> proj1(1)=4
    // CombinerProjection=6, FinalConsumer=600
    // =========================================================================
    {
        let tracked = engine.clone().tracked().await;
        let result = tracked.query(&DiamondFinalConsumer).await;
        assert_eq!(result, 600);
    }

    assert_eq!(firewall_ex.0.load(Ordering::SeqCst), 1);
    assert_eq!(proj1_ex.0.load(Ordering::SeqCst), 2); // proj1(0) and proj1(1)
    assert_eq!(combiner_proj_ex.0.load(Ordering::SeqCst), 1);
    assert_eq!(consumer_ex.0.load(Ordering::SeqCst), 1);

    // =========================================================================
    // Test Case 2: Change Variable(2) - neither projection depends on this
    // Firewall and level-1 projections recompute, but outputs unchanged
    // CombinerProjection should recompute (backward prop) but output unchanged
    // FinalConsumer should NOT recompute
    // =========================================================================
    {
        let mut input_session = engine.input_session().await;
        input_session.set_input(Variable(2), 100).await;
    }

    // Only 1 dirtied edge: Firewall -> Variable(2)
    let tracked = engine.clone().tracked().await;
    assert_eq!(tracked.get_dirtied_edges_count(), 1);
    drop(tracked);

    {
        let tracked = engine.clone().tracked().await;
        let result = tracked.query(&DiamondFinalConsumer).await;
        // Same result since proj1(0) and proj1(1) didn't change
        assert_eq!(result, 600);
    }

    // Firewall recomputed
    assert_eq!(firewall_ex.0.load(Ordering::SeqCst), 2); // +1
    // Both level-1 projections invoked via backward prop
    assert_eq!(proj1_ex.0.load(Ordering::SeqCst), 4); // +2
    // CombinerProjection NOT invoked - level-1 projections outputs unchanged!
    assert_eq!(combiner_proj_ex.0.load(Ordering::SeqCst), 1); // unchanged!
    // FinalConsumer NOT recomputed - CombinerProjection not invoked
    assert_eq!(consumer_ex.0.load(Ordering::SeqCst), 1); // unchanged!

    // =========================================================================
    // Test Case 3: Change Variable(0) which affects proj1(0)
    // Variable(0)=50 -> proj1(0)=100, proj1(1)=4, Combiner=104, Consumer=10400
    // =========================================================================
    {
        let mut input_session = engine.input_session().await;
        input_session.set_input(Variable(0), 50).await;
    }

    {
        let tracked = engine.clone().tracked().await;
        let result = tracked.query(&DiamondFinalConsumer).await;
        assert_eq!(result, 10400);
    }

    assert_eq!(firewall_ex.0.load(Ordering::SeqCst), 3); // +1
    assert_eq!(proj1_ex.0.load(Ordering::SeqCst), 6); // +2
    assert_eq!(combiner_proj_ex.0.load(Ordering::SeqCst), 2); // +1 (proj1(0) changed)
    assert_eq!(consumer_ex.0.load(Ordering::SeqCst), 2); // +1 (combiner changed)

    // =========================================================================
    // Test Case 4: Change both Variable(0) and Variable(1) but preserve sum
    // Variable(0)=52 -> proj1(0)=104, Variable(1)=0 -> proj1(1)=0
    // Combiner=104 (same!), Consumer should NOT recompute
    // =========================================================================
    {
        let mut input_session = engine.input_session().await;
        input_session.set_input(Variable(0), 52).await;
        input_session.set_input(Variable(1), 0).await;
    }

    {
        let tracked = engine.clone().tracked().await;
        let result = tracked.query(&DiamondFinalConsumer).await;
        // 52*2 + 0*2 = 104, 104 * 100 = 10400 (same!)
        assert_eq!(result, 10400);
    }

    // Firewall recomputed
    assert_eq!(firewall_ex.0.load(Ordering::SeqCst), 4); // +1
    // Both level-1 projections recomputed (their inputs changed)
    assert_eq!(proj1_ex.0.load(Ordering::SeqCst), 8); // +2
    // CombinerProjection recomputed (proj1(0) changed: 100->104, proj1(1)
    // changed: 4->0)
    assert_eq!(combiner_proj_ex.0.load(Ordering::SeqCst), 3); // +1
    // FinalConsumer NOT recomputed - CombinerProjection output unchanged (104)!
    assert_eq!(consumer_ex.0.load(Ordering::SeqCst), 2); // unchanged!

    // =========================================================================
    // Test Case 5: Query intermediate projections independently
    // =========================================================================
    {
        let tracked = engine.clone().tracked().await;

        let p0 = tracked.query(&ProjectionLevel1(0)).await;
        assert_eq!(p0, 104); // 52 * 2

        let p1 = tracked.query(&ProjectionLevel1(1)).await;
        assert_eq!(p1, 0); // 0 * 2

        let combiner = tracked.query(&DiamondCombinerProjection).await;
        assert_eq!(combiner, 104); // 104 + 0
    }

    // No additional executions - everything cached
    assert_eq!(firewall_ex.0.load(Ordering::SeqCst), 4); // unchanged
    assert_eq!(proj1_ex.0.load(Ordering::SeqCst), 8); // unchanged
    assert_eq!(combiner_proj_ex.0.load(Ordering::SeqCst), 3); // unchanged
    assert_eq!(consumer_ex.0.load(Ordering::SeqCst), 2); // unchanged
}

// ============================================================================
// Test: Asymmetric Diamond Pattern with Uneven Projection Heights
// ============================================================================
//
// Graph structure:
//              SimpleFirewall
//              /           \
//             v             \
//     ProjectionLevel1(0)    \
//        (Projection)         \
//             |                \
//             v                 \
//     DeepProjectionL2          \
//        (Projection)            \
//             |                   \
//             v                    v
//     DeepProjectionL3      ProjectionLevel1(1)
//        (Projection)         (Projection)
//              \                  /
//               \                /
//                v              v
//           AsymmetricCombinerProjection (PROJECTION)
//                       |
//                       v
//              AsymmetricFinalConsumer
//
// Left branch: Firewall -> L1 -> L2 -> L3 (3 levels of projections)
// Right branch: Firewall -> L1 (1 level of projection)
//
// This tests that dirty propagation correctly handles asymmetric projection
// chains where one branch is deeper than the other.

/// Second level projection in the deep branch - adds 100.
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
    Encode,
    Decode,
)]
pub struct DeepProjectionL2;

impl Query for DeepProjectionL2 {
    type Value = i64;
}

#[derive(Debug, Default)]
pub struct DeepProjectionL2Executor(pub AtomicUsize);

impl<C: Config> Executor<DeepProjectionL2, C> for DeepProjectionL2Executor {
    async fn execute(
        &self,
        _query: &DeepProjectionL2,
        engine: &TrackedEngine<C>,
    ) -> i64 {
        self.0.fetch_add(1, Ordering::SeqCst);

        let val = engine.query(&ProjectionLevel1(0)).await;
        val + 100
    }

    fn execution_style() -> qbice::ExecutionStyle {
        qbice::ExecutionStyle::Projection
    }
}

/// Third level projection in the deep branch - squares the value.
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
    Encode,
    Decode,
)]
pub struct DeepProjectionL3;

impl Query for DeepProjectionL3 {
    type Value = i64;
}

#[derive(Debug, Default)]
pub struct DeepProjectionL3Executor(pub AtomicUsize);

impl<C: Config> Executor<DeepProjectionL3, C> for DeepProjectionL3Executor {
    async fn execute(
        &self,
        _query: &DeepProjectionL3,
        engine: &TrackedEngine<C>,
    ) -> i64 {
        self.0.fetch_add(1, Ordering::SeqCst);

        let val = engine.query(&DeepProjectionL2).await;
        // Take absolute value to enable unchanged output tests
        val.abs()
    }

    fn execution_style() -> qbice::ExecutionStyle {
        qbice::ExecutionStyle::Projection
    }
}

/// Asymmetric combiner - combines deep branch (L3) with shallow branch (L1).
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
    Encode,
    Decode,
)]
pub struct AsymmetricCombinerProjection;

impl Query for AsymmetricCombinerProjection {
    type Value = i64;
}

#[derive(Debug, Default)]
pub struct AsymmetricCombinerProjectionExecutor(pub AtomicUsize);

impl<C: Config> Executor<AsymmetricCombinerProjection, C>
    for AsymmetricCombinerProjectionExecutor
{
    async fn execute(
        &self,
        _query: &AsymmetricCombinerProjection,
        engine: &TrackedEngine<C>,
    ) -> i64 {
        self.0.fetch_add(1, Ordering::SeqCst);

        // Deep branch (3 levels)
        let deep = engine.query(&DeepProjectionL3).await;
        // Shallow branch (1 level)
        let shallow = engine.query(&ProjectionLevel1(1)).await;

        deep + shallow
    }

    fn execution_style() -> qbice::ExecutionStyle {
        qbice::ExecutionStyle::Projection
    }
}

/// Final consumer for the asymmetric pattern.
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
    Encode,
    Decode,
)]
pub struct AsymmetricFinalConsumer;

impl Query for AsymmetricFinalConsumer {
    type Value = i64;
}

#[derive(Debug, Default)]
pub struct AsymmetricFinalConsumerExecutor(pub AtomicUsize);

impl<C: Config> Executor<AsymmetricFinalConsumer, C>
    for AsymmetricFinalConsumerExecutor
{
    async fn execute(
        &self,
        _query: &AsymmetricFinalConsumer,
        engine: &TrackedEngine<C>,
    ) -> i64 {
        self.0.fetch_add(1, Ordering::SeqCst);

        let combined = engine.query(&AsymmetricCombinerProjection).await;
        combined * 10
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn asymmetric_diamond_projection_pattern() {
    let tempdir = tempdir().unwrap();
    let mut engine = create_test_engine(&tempdir).await;

    let firewall_ex = Arc::new(SimpleFirewallExecutor::default());
    let proj1_ex = Arc::new(ProjectionLevel1Executor::default());
    let l2_ex = Arc::new(DeepProjectionL2Executor::default());
    let l3_ex = Arc::new(DeepProjectionL3Executor::default());
    let combiner_ex = Arc::new(AsymmetricCombinerProjectionExecutor::default());
    let consumer_ex = Arc::new(AsymmetricFinalConsumerExecutor::default());

    engine.register_executor(firewall_ex.clone());
    engine.register_executor(proj1_ex.clone());
    engine.register_executor(l2_ex.clone());
    engine.register_executor(l3_ex.clone());
    engine.register_executor(combiner_ex.clone());
    engine.register_executor(consumer_ex.clone());

    let engine = Arc::new(engine);

    // Initialize variables
    {
        let mut input_session = engine.input_session().await;
        for i in 0..100 {
            input_session.set_input(Variable(i), i.cast_signed() + 1).await;
        }
    }

    // =========================================================================
    // Test Case 1: Initial query through the asymmetric diamond
    // Variable(0)=1 -> L1(0)=2 -> L2=102 -> L3=102 (abs)
    // Variable(1)=2 -> L1(1)=4
    // Combiner=102+4=106, Consumer=1060
    // =========================================================================
    {
        let tracked = engine.clone().tracked().await;
        let result = tracked.query(&AsymmetricFinalConsumer).await;
        assert_eq!(result, 1060);
    }

    assert_eq!(firewall_ex.0.load(Ordering::SeqCst), 1);
    assert_eq!(proj1_ex.0.load(Ordering::SeqCst), 2); // L1(0) and L1(1)
    assert_eq!(l2_ex.0.load(Ordering::SeqCst), 1);
    assert_eq!(l3_ex.0.load(Ordering::SeqCst), 1);
    assert_eq!(combiner_ex.0.load(Ordering::SeqCst), 1);
    assert_eq!(consumer_ex.0.load(Ordering::SeqCst), 1);

    // =========================================================================
    // Test Case 2: Change unrelated Variable(2)
    // Firewall recomputes, L1(0) and L1(1) recompute via backward prop
    // Outputs unchanged -> L2, L3, Combiner, Consumer NOT invoked
    // =========================================================================
    {
        let mut input_session = engine.input_session().await;
        input_session.set_input(Variable(2), 100).await;
    }

    {
        let tracked = engine.clone().tracked().await;
        let result = tracked.query(&AsymmetricFinalConsumer).await;
        assert_eq!(result, 1060); // Same
    }

    assert_eq!(firewall_ex.0.load(Ordering::SeqCst), 2); // +1
    assert_eq!(proj1_ex.0.load(Ordering::SeqCst), 4); // +2 (backward prop)
    // Deep branch NOT invoked - L1(0) output unchanged
    assert_eq!(l2_ex.0.load(Ordering::SeqCst), 1); // unchanged
    assert_eq!(l3_ex.0.load(Ordering::SeqCst), 1); // unchanged
    assert_eq!(combiner_ex.0.load(Ordering::SeqCst), 1); // unchanged
    assert_eq!(consumer_ex.0.load(Ordering::SeqCst), 1); // unchanged

    // =========================================================================
    // Test Case 3: Change Variable(0) - affects deep branch only
    // Variable(0)=5 -> L1(0)=10 -> L2=110 -> L3=110
    // Variable(1)=2 -> L1(1)=4 (unchanged)
    // Combiner=110+4=114, Consumer=1140
    // =========================================================================
    {
        let mut input_session = engine.input_session().await;
        input_session.set_input(Variable(0), 5).await;
    }

    {
        let tracked = engine.clone().tracked().await;
        let result = tracked.query(&AsymmetricFinalConsumer).await;
        assert_eq!(result, 1140);
    }

    assert_eq!(firewall_ex.0.load(Ordering::SeqCst), 3); // +1
    assert_eq!(proj1_ex.0.load(Ordering::SeqCst), 6); // +2 (backward prop)
    assert_eq!(l2_ex.0.load(Ordering::SeqCst), 2); // +1 (L1(0) changed)
    assert_eq!(l3_ex.0.load(Ordering::SeqCst), 2); // +1 (L2 changed)
    assert_eq!(combiner_ex.0.load(Ordering::SeqCst), 2); // +1 (L3 changed)
    assert_eq!(consumer_ex.0.load(Ordering::SeqCst), 2); // +1 (combiner changed)

    // =========================================================================
    // Test Case 4: Change Variable(1) - affects shallow branch only
    // IMPORTANT: Since L1(1) is only queried by Combiner (a projection),
    // and L3 (combiner's other dependency) output didn't change,
    // the Combiner won't be invoked via backward prop!
    //
    // This is the key asymmetric behavior: changes in the shallow branch
    // don't automatically propagate because the combiner projection
    // wasn't re-executed.
    //
    // Variable(0)=5 -> L1(0)=10 -> L2=110 -> L3=110 (same)
    // Variable(1)=10 but L1(1) won't be queried again!
    // Result stays: Combiner=114, Consumer=1140
    // =========================================================================
    {
        let mut input_session = engine.input_session().await;
        input_session.set_input(Variable(1), 10).await;
    }

    {
        let tracked = engine.clone().tracked().await;
        let result = tracked.query(&AsymmetricFinalConsumer).await;
        // L1(1) is NOT re-queried because Combiner wasn't invoked
        // (its other dependency L3 didn't change)
        // The firewall also doesn't recompute because the query path
        // (through L1(0)) doesn't touch Variable(1)
        assert_eq!(result, 1140); // SAME as before!
    }

    // Firewall NOT recomputed - the query path through L3->L2->L1(0) doesn't
    // trigger recomputation because Variable(0) hasn't changed
    assert_eq!(firewall_ex.0.load(Ordering::SeqCst), 3); // unchanged!
    // proj1 unchanged - L1(0) was not re-invoked
    assert_eq!(proj1_ex.0.load(Ordering::SeqCst), 6); // unchanged!
    // Deep branch unchanged - nothing triggered recomputation
    assert_eq!(l2_ex.0.load(Ordering::SeqCst), 2); // unchanged
    assert_eq!(l3_ex.0.load(Ordering::SeqCst), 2); // unchanged
    // Combiner NOT invoked - L3 output unchanged!
    assert_eq!(combiner_ex.0.load(Ordering::SeqCst), 2); // unchanged!
    assert_eq!(consumer_ex.0.load(Ordering::SeqCst), 2); // unchanged!

    // =========================================================================
    // Test Case 5: Now change Variable(0) so L3 changes - this will trigger
    // Combiner to re-execute and pick up the NEW L1(1) value
    // Variable(0)=10 -> L1(0)=20 -> L2=120 -> L3=120
    // L1(1) will be re-queried now: 10*2=20
    // Combiner=120+20=140, Consumer=1400
    // =========================================================================
    {
        let mut input_session = engine.input_session().await;
        input_session.set_input(Variable(0), 10).await;
    }

    {
        let tracked = engine.clone().tracked().await;
        let result = tracked.query(&AsymmetricFinalConsumer).await;
        assert_eq!(result, 1400);
    }

    assert_eq!(firewall_ex.0.load(Ordering::SeqCst), 4); // +1
    // L1(0) invoked via backward prop, L1(1) when Combiner executes
    assert_eq!(proj1_ex.0.load(Ordering::SeqCst), 8); // +2
    assert_eq!(l2_ex.0.load(Ordering::SeqCst), 3); // +1 (L1(0) changed)
    assert_eq!(l3_ex.0.load(Ordering::SeqCst), 3); // +1 (L2 changed)
    assert_eq!(combiner_ex.0.load(Ordering::SeqCst), 3); // +1 (L3 changed)
    assert_eq!(consumer_ex.0.load(Ordering::SeqCst), 3); // +1 (combiner changed)

    // =========================================================================
    // Test Case 6: Change Variable(0) to negative - L3 uses abs() so output
    // stays same
    // Current: L1(0)=20 -> L2=120 -> L3=120
    // Need |L2|=120, so L2=120 or L2=-120
    // L2 = L1(0)+100, so L1(0)=20 or L1(0)=-220
    // L1(0) = Variable(0)*2, so Variable(0)=10 or Variable(0)=-110
    // If Variable(0)=-110: L1(0)=-220 -> L2=-120 -> L3=|-120|=120 (same!)
    // =========================================================================
    {
        let mut input_session = engine.input_session().await;
        input_session.set_input(Variable(0), -110).await;
    }

    {
        let tracked = engine.clone().tracked().await;
        let result = tracked.query(&AsymmetricFinalConsumer).await;
        // L1(0)=-220 -> L2=-120 -> L3=120 (same as before!)
        // L1(1)=20 (not re-queried since combiner not invoked)
        // Combiner NOT invoked because L3 unchanged
        // Result stays: 140*10=1400
        assert_eq!(result, 1400);
    }

    assert_eq!(firewall_ex.0.load(Ordering::SeqCst), 5); // +1
    // Both L1(0) and L1(1) invoked via backward prop from firewall
    assert_eq!(proj1_ex.0.load(Ordering::SeqCst), 10); // +2
    // Deep branch recomputes through L2, but L3 output unchanged
    assert_eq!(l2_ex.0.load(Ordering::SeqCst), 4); // +1 (L1(0) changed)
    assert_eq!(l3_ex.0.load(Ordering::SeqCst), 4); // +1 (L2 changed)
    // Combiner NOT invoked - L3 output unchanged, L1(1) unchanged
    assert_eq!(combiner_ex.0.load(Ordering::SeqCst), 3); // unchanged!
    assert_eq!(consumer_ex.0.load(Ordering::SeqCst), 3); // unchanged!

    // =========================================================================
    // Test Case 7: Query intermediate projections to verify values
    // =========================================================================
    {
        let tracked = engine.clone().tracked().await;

        let l1_0 = tracked.query(&ProjectionLevel1(0)).await;
        assert_eq!(l1_0, -220); // -110 * 2

        let l1_1 = tracked.query(&ProjectionLevel1(1)).await;
        assert_eq!(l1_1, 20); // 10 * 2 (from test case 5)

        let l2 = tracked.query(&DeepProjectionL2).await;
        assert_eq!(l2, -120); // -220 + 100

        let l3 = tracked.query(&DeepProjectionL3).await;
        assert_eq!(l3, 120); // |-120|

        let combiner = tracked.query(&AsymmetricCombinerProjection).await;
        assert_eq!(combiner, 140); // 120 + 20
    }

    // No additional executions - everything cached
    assert_eq!(firewall_ex.0.load(Ordering::SeqCst), 5); // unchanged
    assert_eq!(proj1_ex.0.load(Ordering::SeqCst), 10); // unchanged
    assert_eq!(l2_ex.0.load(Ordering::SeqCst), 4); // unchanged
    assert_eq!(l3_ex.0.load(Ordering::SeqCst), 4); // unchanged
    assert_eq!(combiner_ex.0.load(Ordering::SeqCst), 3); // unchanged
    assert_eq!(consumer_ex.0.load(Ordering::SeqCst), 3); // unchanged
}

// ============================================================================
// Test: No-Change Firewall Propagation
// ============================================================================
//
// Tests that when a firewall recomputes but produces the same output,
// no downstream propagation occurs.

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
    Encode,
    Decode,
)]
pub struct SumFirewall;

impl Query for SumFirewall {
    type Value = i64;
}

#[derive(Debug, Default)]
pub struct SumFirewallExecutor(pub AtomicUsize);

impl<C: Config> Executor<SumFirewall, C> for SumFirewallExecutor {
    async fn execute(
        &self,
        _query: &SumFirewall,
        engine: &TrackedEngine<C>,
    ) -> i64 {
        self.0.fetch_add(1, Ordering::SeqCst);

        let a = engine.query(&Variable(0)).await;
        let b = engine.query(&Variable(1)).await;

        a + b
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
    Encode,
    Decode,
)]
pub struct SumProjection;

impl Query for SumProjection {
    type Value = i64;
}

#[derive(Debug, Default)]
pub struct SumProjectionExecutor(pub AtomicUsize);

impl<C: Config> Executor<SumProjection, C> for SumProjectionExecutor {
    async fn execute(
        &self,
        _query: &SumProjection,
        engine: &TrackedEngine<C>,
    ) -> i64 {
        self.0.fetch_add(1, Ordering::SeqCst);

        engine.query(&SumFirewall).await
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
    Encode,
    Decode,
)]
pub struct SumConsumer;

impl Query for SumConsumer {
    type Value = i64;
}

#[derive(Debug, Default)]
pub struct SumConsumerExecutor(pub AtomicUsize);

impl<C: Config> Executor<SumConsumer, C> for SumConsumerExecutor {
    async fn execute(
        &self,
        _query: &SumConsumer,
        engine: &TrackedEngine<C>,
    ) -> i64 {
        self.0.fetch_add(1, Ordering::SeqCst);

        let proj = engine.query(&SumProjection).await;
        proj * 10
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn firewall_same_output_no_propagation() {
    let tempdir = tempdir().unwrap();
    let mut engine = create_test_engine(&tempdir).await;

    let firewall_ex = Arc::new(SumFirewallExecutor::default());
    let proj_ex = Arc::new(SumProjectionExecutor::default());
    let consumer_ex = Arc::new(SumConsumerExecutor::default());

    engine.register_executor(firewall_ex.clone());
    engine.register_executor(proj_ex.clone());
    engine.register_executor(consumer_ex.clone());

    let engine = Arc::new(engine);

    // Initialize: Variable(0)=5, Variable(1)=5, sum=10
    {
        let mut input_session = engine.input_session().await;
        input_session.set_input(Variable(0), 5_i64).await;
        input_session.set_input(Variable(1), 5_i64).await;
    }

    // Initial query
    {
        let tracked = engine.clone().tracked().await;
        let result = tracked.query(&SumConsumer).await;
        assert_eq!(result, 100); // sum=10, consumer=100
    }

    assert_eq!(firewall_ex.0.load(Ordering::SeqCst), 1);
    assert_eq!(proj_ex.0.load(Ordering::SeqCst), 1);
    assert_eq!(consumer_ex.0.load(Ordering::SeqCst), 1);

    // Change Variable(0)=3, Variable(1)=7 - sum still equals 10!
    {
        let mut input_session = engine.input_session().await;
        input_session.set_input(Variable(0), 3_i64).await;
        input_session.set_input(Variable(1), 7_i64).await;
    }

    // 2 dirtied edges: Firewall -> Variable(0), Firewall -> Variable(1)
    let tracked = engine.clone().tracked().await;
    assert_eq!(tracked.get_dirtied_edges_count(), 2);
    drop(tracked);

    // Re-query
    {
        let tracked = engine.clone().tracked().await;
        let result = tracked.query(&SumConsumer).await;
        assert_eq!(result, 100); // Still 100
    }

    // Firewall recomputed (to verify inputs)
    assert_eq!(firewall_ex.0.load(Ordering::SeqCst), 2); // +1

    // Since the firewall output is unchanged, backward propagation is NOT
    // triggered, so projection does NOT get invoked
    assert_eq!(proj_ex.0.load(Ordering::SeqCst), 1); // unchanged - no backward prop

    // Consumer verifies its dependencies - projection value unchanged so
    // consumer doesn't need to recompute
    assert_eq!(consumer_ex.0.load(Ordering::SeqCst), 1); // unchanged - projection filtered the change

    // After query, check dirtied edges - should include backward prop edges
    // but be minimal
    let tracked = engine.clone().tracked().await;
    let dirtied = tracked.get_dirtied_edges_count();
    // Should be: 2 original + Projection -> Firewall from backward prop
    assert!(
        dirtied <= 4,
        "Dirtied edges should be minimal since output unchanged, got {dirtied}"
    );
}

// ============================================================================
// Test: Concurrent Projection Access
// ============================================================================
//
// Tests that multiple concurrent queries to different projections work
// correctly without race conditions.

#[tokio::test(flavor = "multi_thread")]
#[allow(clippy::cast_possible_wrap)]
async fn concurrent_projection_access() {
    for _ in 0..500 {
        let tempdir = tempdir().unwrap();
        let mut engine = create_test_engine(&tempdir).await;

        let firewall_ex = Arc::new(SimpleFirewallExecutor::default());
        let proj1_ex = Arc::new(ProjectionLevel1Executor::default());

        engine.register_executor(firewall_ex.clone());
        engine.register_executor(proj1_ex.clone());

        let engine = Arc::new(engine);

        // Initialize variables
        {
            let mut input_session = engine.input_session().await;
            for i in 0..100 {
                input_session.set_input(Variable(i), i.cast_signed() + 1).await;
            }
        }

        // Spawn concurrent queries to all 100 projections
        let mut handles = Vec::new();
        for i in 0..100 {
            let engine = engine.clone();
            handles.push(tokio::spawn(async move {
                let tracked = engine.tracked().await;
                tracked.query(&ProjectionLevel1(i)).await
            }));
        }

        // Collect results
        let mut results = Vec::new();
        for handle in handles {
            results.push(handle.await.unwrap());
        }

        // Verify results: (i+1) * 2 for each
        for (i, result) in results.iter().copied().enumerate() {
            assert_eq!(result, (i as i64 + 1) * 2);
        }

        // Firewall should only execute once despite concurrent access
        assert_eq!(firewall_ex.0.load(Ordering::SeqCst), 1);

        // Each projection executed once
        assert_eq!(proj1_ex.0.load(Ordering::SeqCst), 100);
    }
}

// ============================================================================
// Test: Nested Firewall with Projection
// ============================================================================
//
// Graph structure:
//   Variable(0) -> OuterFirewall -> InnerFirewall -> Projection -> Consumer
//
// Tests the TFC (transitive firewall callees) mechanism ensures proper
// repair order.

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
    Encode,
    Decode,
)]
pub struct OuterFirewall;

impl Query for OuterFirewall {
    type Value = i64;
}

#[derive(Debug, Default)]
pub struct OuterFirewallExecutor(pub AtomicUsize);

impl<C: Config> Executor<OuterFirewall, C> for OuterFirewallExecutor {
    async fn execute(
        &self,
        _query: &OuterFirewall,
        engine: &TrackedEngine<C>,
    ) -> i64 {
        self.0.fetch_add(1, Ordering::SeqCst);

        let val = engine.query(&Variable(0)).await;
        val * 2
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
    Encode,
    Decode,
)]
pub struct InnerFirewall;

impl Query for InnerFirewall {
    type Value = i64;
}

#[derive(Debug, Default)]
pub struct InnerFirewallExecutor(pub AtomicUsize);

impl<C: Config> Executor<InnerFirewall, C> for InnerFirewallExecutor {
    async fn execute(
        &self,
        _query: &InnerFirewall,
        engine: &TrackedEngine<C>,
    ) -> i64 {
        self.0.fetch_add(1, Ordering::SeqCst);

        let val = engine.query(&OuterFirewall).await;

        val + 10
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
    Encode,
    Decode,
)]
pub struct NestedProjection;

impl Query for NestedProjection {
    type Value = i64;
}

#[derive(Debug, Default)]
pub struct NestedProjectionExecutor(pub AtomicUsize);

impl<C: Config> Executor<NestedProjection, C> for NestedProjectionExecutor {
    async fn execute(
        &self,
        _query: &NestedProjection,
        engine: &TrackedEngine<C>,
    ) -> i64 {
        self.0.fetch_add(1, Ordering::SeqCst);

        let val = engine.query(&InnerFirewall).await;
        val * 3
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
    Encode,
    Decode,
)]
pub struct NestedConsumer;

impl Query for NestedConsumer {
    type Value = i64;
}

#[derive(Debug, Default)]
pub struct NestedConsumerExecutor(pub AtomicUsize);

impl<C: Config> Executor<NestedConsumer, C> for NestedConsumerExecutor {
    async fn execute(
        &self,
        _query: &NestedConsumer,
        engine: &TrackedEngine<C>,
    ) -> i64 {
        self.0.fetch_add(1, Ordering::SeqCst);

        let val = engine.query(&NestedProjection).await;

        val + 1000
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn nested_firewall_with_projection() {
    let tempdir = tempdir().unwrap();
    let mut engine = create_test_engine(&tempdir).await;

    let outer_ex = Arc::new(OuterFirewallExecutor::default());
    let inner_ex = Arc::new(InnerFirewallExecutor::default());
    let proj_ex = Arc::new(NestedProjectionExecutor::default());
    let consumer_ex = Arc::new(NestedConsumerExecutor::default());

    engine.register_executor(outer_ex.clone());
    engine.register_executor(inner_ex.clone());
    engine.register_executor(proj_ex.clone());
    engine.register_executor(consumer_ex.clone());

    let engine = Arc::new(engine);

    // Initialize
    {
        let mut input_session = engine.input_session().await;
        input_session.set_input(Variable(0), 5_i64).await;
    }

    // Initial query
    // Variable(0)=5 -> Outer=10 -> Inner=20 -> Proj=60 -> Consumer=1060
    {
        let tracked = engine.clone().tracked().await;
        let result = tracked.query(&NestedConsumer).await;
        assert_eq!(result, 1060);
    }

    assert_eq!(outer_ex.0.load(Ordering::SeqCst), 1);
    assert_eq!(inner_ex.0.load(Ordering::SeqCst), 1);
    assert_eq!(proj_ex.0.load(Ordering::SeqCst), 1);
    assert_eq!(consumer_ex.0.load(Ordering::SeqCst), 1);

    // Change input
    {
        let mut input_session = engine.input_session().await;
        input_session.set_input(Variable(0), 10_i64).await;
    }

    // Only 1 dirtied edge: OuterFirewall -> Variable(0)
    // InnerFirewall and beyond are NOT dirtied due to firewall boundary
    let tracked = engine.clone().tracked().await;
    assert_eq!(tracked.get_dirtied_edges_count(), 1);
    drop(tracked);

    // Re-query
    // Variable(0)=10 -> Outer=20 -> Inner=30 -> Proj=90 -> Consumer=1090
    {
        let tracked = engine.clone().tracked().await;
        let result = tracked.query(&NestedConsumer).await;
        assert_eq!(result, 1090);
    }

    // All executors should have run again due to cascading changes
    assert_eq!(outer_ex.0.load(Ordering::SeqCst), 2);
    assert_eq!(inner_ex.0.load(Ordering::SeqCst), 2);
    assert_eq!(proj_ex.0.load(Ordering::SeqCst), 2);
    assert_eq!(consumer_ex.0.load(Ordering::SeqCst), 2);

    // Final dirtied edge count check
    let tracked = engine.clone().tracked().await;
    let dirtied = tracked.get_dirtied_edges_count();
    // Should include edges from backward propagation through firewalls
    // but be controlled by projection boundaries
    assert!(
        dirtied <= 5,
        "Dirtied edges should be controlled by firewall/projection \
         boundaries, got {dirtied}"
    );
}

// ============================================================================
// Test: Projection Depending on Two Firewalls
// ============================================================================
//
// Graph structure:
//   Variable(0) -> FirewallA (doubles value)
//                      \
//                       \
//   Variable(1) -> FirewallB (triples value)
//                       /
//                      /
//              DualFirewallProjection (sums both firewall outputs)
//                      |
//                      v
//               DualFirewallConsumer
//
// This tests that a projection can correctly depend on multiple firewalls
// and that dirty propagation works correctly when either or both firewalls
// change.

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
    Encode,
    Decode,
)]
pub struct FirewallA;

impl Query for FirewallA {
    type Value = i64;
}

#[derive(Debug, Default)]
pub struct FirewallAExecutor(pub AtomicUsize);

impl<C: Config> Executor<FirewallA, C> for FirewallAExecutor {
    async fn execute(
        &self,
        _query: &FirewallA,
        engine: &TrackedEngine<C>,
    ) -> i64 {
        self.0.fetch_add(1, Ordering::SeqCst);

        let val = engine.query(&Variable(0)).await;

        val * 2 // doubles
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
    Encode,
    Decode,
)]
pub struct FirewallB;

impl Query for FirewallB {
    type Value = i64;
}

#[derive(Debug, Default)]
pub struct FirewallBExecutor(pub AtomicUsize);

impl<C: Config> Executor<FirewallB, C> for FirewallBExecutor {
    async fn execute(
        &self,
        _query: &FirewallB,
        engine: &TrackedEngine<C>,
    ) -> i64 {
        self.0.fetch_add(1, Ordering::SeqCst);

        let val = engine.query(&Variable(1)).await;

        val * 3 // triples
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
    Encode,
    Decode,
)]
pub struct DualFirewallProjection;

impl Query for DualFirewallProjection {
    type Value = i64;
}

#[derive(Debug, Default)]
pub struct DualFirewallProjectionExecutor(pub AtomicUsize);

impl<C: Config> Executor<DualFirewallProjection, C>
    for DualFirewallProjectionExecutor
{
    async fn execute(
        &self,
        _query: &DualFirewallProjection,
        engine: &TrackedEngine<C>,
    ) -> i64 {
        self.0.fetch_add(1, Ordering::SeqCst);

        let a = engine.query(&FirewallA).await;
        let b = engine.query(&FirewallB).await;

        a + b
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
    Encode,
    Decode,
)]
pub struct DualFirewallConsumer;

impl Query for DualFirewallConsumer {
    type Value = i64;
}

#[derive(Debug, Default)]
pub struct DualFirewallConsumerExecutor(pub AtomicUsize);

impl<C: Config> Executor<DualFirewallConsumer, C>
    for DualFirewallConsumerExecutor
{
    async fn execute(
        &self,
        _query: &DualFirewallConsumer,
        engine: &TrackedEngine<C>,
    ) -> i64 {
        self.0.fetch_add(1, Ordering::SeqCst);

        let proj = engine.query(&DualFirewallProjection).await;

        proj * 10
    }
}

#[tokio::test(flavor = "multi_thread")]
#[allow(clippy::similar_names)]
async fn projection_with_two_firewalls() {
    let tempdir = tempdir().unwrap();
    let mut engine = create_test_engine(&tempdir).await;

    let firewall_a_ex = Arc::new(FirewallAExecutor::default());
    let firewall_b_ex = Arc::new(FirewallBExecutor::default());
    let proj_ex = Arc::new(DualFirewallProjectionExecutor::default());
    let consumer_ex = Arc::new(DualFirewallConsumerExecutor::default());

    engine.register_executor(firewall_a_ex.clone());
    engine.register_executor(firewall_b_ex.clone());
    engine.register_executor(proj_ex.clone());
    engine.register_executor(consumer_ex.clone());

    let engine = Arc::new(engine);

    // Initialize: Variable(0)=5, Variable(1)=10
    {
        let mut input_session = engine.input_session().await;
        input_session.set_input(Variable(0), 5_i64).await;
        input_session.set_input(Variable(1), 10_i64).await;
    }

    // Initial query
    // FirewallA: 5*2=10, FirewallB: 10*3=30, Proj: 10+30=40, Consumer:
    // 40*10=400
    {
        let tracked = engine.clone().tracked().await;
        let result = tracked.query(&DualFirewallConsumer).await;
        assert_eq!(result, 400);
    }

    assert_eq!(firewall_a_ex.0.load(Ordering::SeqCst), 1);
    assert_eq!(firewall_b_ex.0.load(Ordering::SeqCst), 1);
    assert_eq!(proj_ex.0.load(Ordering::SeqCst), 1);
    assert_eq!(consumer_ex.0.load(Ordering::SeqCst), 1);

    // =========================================================================
    // Test Case 1: Change only Variable(0) - only FirewallA should recompute
    // =========================================================================
    {
        let mut input_session = engine.input_session().await;
        input_session.set_input(Variable(0), 10_i64).await;
    }

    // Only 1 dirtied edge: FirewallA -> Variable(0)
    let tracked = engine.clone().tracked().await;
    assert_eq!(tracked.get_dirtied_edges_count(), 1);
    drop(tracked);

    // Re-query
    // FirewallA: 10*2=20, FirewallB: 10*3=30 (unchanged), Proj: 20+30=50,
    // Consumer: 50*10=500
    {
        let tracked = engine.clone().tracked().await;
        let result = tracked.query(&DualFirewallConsumer).await;
        assert_eq!(result, 500);
    }

    // FirewallA recomputed, FirewallB should NOT recompute (not dirty)
    assert_eq!(firewall_a_ex.0.load(Ordering::SeqCst), 2); // +1
    assert_eq!(firewall_b_ex.0.load(Ordering::SeqCst), 1); // unchanged
    // Projection invoked via backward prop from FirewallA
    assert_eq!(proj_ex.0.load(Ordering::SeqCst), 2); // +1
    // Consumer recomputed since projection changed
    assert_eq!(consumer_ex.0.load(Ordering::SeqCst), 2); // +1

    // =========================================================================
    // Test Case 2: Change only Variable(1) - only FirewallB should recompute
    // =========================================================================
    {
        let mut input_session = engine.input_session().await;
        input_session.set_input(Variable(1), 20_i64).await;
    }

    // Only 1 dirtied edge: FirewallB -> Variable(1)
    let tracked = engine.clone().tracked().await;
    assert_eq!(tracked.get_dirtied_edges_count(), 1);
    drop(tracked);

    // Re-query
    // FirewallA: 10*2=20 (unchanged), FirewallB: 20*3=60, Proj: 20+60=80,
    // Consumer: 80*10=800
    {
        let tracked = engine.clone().tracked().await;
        let result = tracked.query(&DualFirewallConsumer).await;
        assert_eq!(result, 800);
    }

    // FirewallA should NOT recompute (not dirty)
    assert_eq!(firewall_a_ex.0.load(Ordering::SeqCst), 2); // unchanged
    // FirewallB recomputed
    assert_eq!(firewall_b_ex.0.load(Ordering::SeqCst), 2); // +1
    // Projection invoked via backward prop from FirewallB
    assert_eq!(proj_ex.0.load(Ordering::SeqCst), 3); // +1
    // Consumer recomputed since projection changed
    assert_eq!(consumer_ex.0.load(Ordering::SeqCst), 3); // +1

    // =========================================================================
    // Test Case 3: Change BOTH Variable(0) and Variable(1)
    // =========================================================================
    {
        let mut input_session = engine.input_session().await;
        input_session.set_input(Variable(0), 15_i64).await;
        input_session.set_input(Variable(1), 25_i64).await;
    }

    // 2 dirtied edges: FirewallA -> Variable(0), FirewallB -> Variable(1)
    let tracked = engine.clone().tracked().await;
    assert_eq!(tracked.get_dirtied_edges_count(), 2);
    drop(tracked);

    // Re-query
    // FirewallA: 15*2=30, FirewallB: 25*3=75, Proj: 30+75=105,
    // Consumer: 105*10=1050
    {
        let tracked = engine.clone().tracked().await;
        let result = tracked.query(&DualFirewallConsumer).await;
        assert_eq!(result, 1050);
    }

    // Both firewalls recomputed
    assert_eq!(firewall_a_ex.0.load(Ordering::SeqCst), 3); // +1
    assert_eq!(firewall_b_ex.0.load(Ordering::SeqCst), 3); // +1
    // Projection invoked via backward prop (from whichever firewall completes
    // first, but only once since it's already computing when the second
    // tries to invoke it)
    assert_eq!(proj_ex.0.load(Ordering::SeqCst), 4); // +1
    // Consumer recomputed since projection changed
    assert_eq!(consumer_ex.0.load(Ordering::SeqCst), 4); // +1

    // =========================================================================
    // Test Case 4: Change Variable(0) but result in same FirewallA output
    // FirewallA doubles, so 15*2=30 and 30/2*2=30 would be same
    // Let's change to value that produces same output: not possible with
    // doubling Let's instead verify that if we change to same value, no
    // recomputation
    // =========================================================================
    {
        let mut input_session = engine.input_session().await;
        // Set to same value - should not trigger any recomputation
        input_session.set_input(Variable(0), 15_i64).await;
    }

    // 0 dirtied edges since value didn't change
    let tracked = engine.clone().tracked().await;
    assert_eq!(tracked.get_dirtied_edges_count(), 0);
    drop(tracked);

    // Re-query - everything should be cached
    {
        let tracked = engine.clone().tracked().await;
        let result = tracked.query(&DualFirewallConsumer).await;
        assert_eq!(result, 1050); // Same as before
    }

    // No executors should have run
    assert_eq!(firewall_a_ex.0.load(Ordering::SeqCst), 3); // unchanged
    assert_eq!(firewall_b_ex.0.load(Ordering::SeqCst), 3); // unchanged
    assert_eq!(proj_ex.0.load(Ordering::SeqCst), 4); // unchanged
    assert_eq!(consumer_ex.0.load(Ordering::SeqCst), 4); // unchanged
}

// ============================================================================
// Test: Chained Projections
// ============================================================================
//
// Graph structure:
//   Variable(0) -> FirewallA (produces value * 3)
//                      |
//                      v
//               ProjectionInner (extracts and adds 10)
//                      |
//                      v
//               ProjectionOuter (multiplies by 2)
//                      |
//                      v
//               ProjectionConsumer (divides by 5)
//
// This tests that a chain of projections correctly filters changes and
// that dirty propagation works properly through multiple projection layers.

/// A firewall query that triples the input value.
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
    Encode,
    Decode,
)]
pub struct ChainedFirewallA;

impl Query for ChainedFirewallA {
    type Value = i64;
}

#[derive(Debug, Default)]
pub struct ChainedFirewallAExecutor(pub AtomicUsize);

impl<C: Config> Executor<ChainedFirewallA, C> for ChainedFirewallAExecutor {
    async fn execute(
        &self,
        _query: &ChainedFirewallA,
        engine: &TrackedEngine<C>,
    ) -> i64 {
        self.0.fetch_add(1, Ordering::SeqCst);

        let val = engine.query(&Variable(0)).await;
        val * 3
    }

    fn execution_style() -> qbice::ExecutionStyle {
        qbice::ExecutionStyle::Firewall
    }
}

/// First projection in the chain - adds 10 to the firewall result.
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
    Encode,
    Decode,
)]
pub struct ChainedProjectionInner;

impl Query for ChainedProjectionInner {
    type Value = i64;
}

#[derive(Debug, Default)]
pub struct ChainedProjectionInnerExecutor(pub AtomicUsize);

impl<C: Config> Executor<ChainedProjectionInner, C>
    for ChainedProjectionInnerExecutor
{
    async fn execute(
        &self,
        _query: &ChainedProjectionInner,
        engine: &TrackedEngine<C>,
    ) -> i64 {
        self.0.fetch_add(1, Ordering::SeqCst);

        let firewall_result = engine.query(&ChainedFirewallA).await;
        firewall_result + 10
    }

    fn execution_style() -> qbice::ExecutionStyle {
        qbice::ExecutionStyle::Projection
    }
}

/// Second projection in the chain - multiplies the inner projection result by
/// 2.
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
    Encode,
    Decode,
)]
pub struct ChainedProjectionOuter;

impl Query for ChainedProjectionOuter {
    type Value = i64;
}

#[derive(Debug, Default)]
pub struct ChainedProjectionOuterExecutor(pub AtomicUsize);

impl<C: Config> Executor<ChainedProjectionOuter, C>
    for ChainedProjectionOuterExecutor
{
    async fn execute(
        &self,
        _query: &ChainedProjectionOuter,
        engine: &TrackedEngine<C>,
    ) -> i64 {
        self.0.fetch_add(1, Ordering::SeqCst);

        let inner_result = engine.query(&ChainedProjectionInner).await;
        inner_result * 2
    }

    fn execution_style() -> qbice::ExecutionStyle {
        qbice::ExecutionStyle::Projection
    }
}

/// Final consumer in the chain - divides the outer projection result by 5.
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
    Encode,
    Decode,
)]
pub struct ChainedProjectionConsumer;

impl Query for ChainedProjectionConsumer {
    type Value = i64;
}

#[derive(Debug, Default)]
pub struct ChainedProjectionConsumerExecutor(pub AtomicUsize);

impl<C: Config> Executor<ChainedProjectionConsumer, C>
    for ChainedProjectionConsumerExecutor
{
    async fn execute(
        &self,
        _query: &ChainedProjectionConsumer,
        engine: &TrackedEngine<C>,
    ) -> i64 {
        self.0.fetch_add(1, Ordering::SeqCst);

        let outer_result = engine.query(&ChainedProjectionOuter).await;
        outer_result / 5
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn chained_projections() {
    let tempdir = tempdir().unwrap();
    let mut engine = create_test_engine(&tempdir).await;

    let firewall_ex = Arc::new(ChainedFirewallAExecutor::default());
    let inner_proj_ex = Arc::new(ChainedProjectionInnerExecutor::default());
    let outer_proj_ex = Arc::new(ChainedProjectionOuterExecutor::default());
    let consumer_ex = Arc::new(ChainedProjectionConsumerExecutor::default());

    engine.register_executor(firewall_ex.clone());
    engine.register_executor(inner_proj_ex.clone());
    engine.register_executor(outer_proj_ex.clone());
    engine.register_executor(consumer_ex.clone());

    let engine = Arc::new(engine);

    // =========================================================================
    // Test Case 1: Initial query through the chain
    // Variable(0)=5 -> FirewallA=15 -> ProjectionInner=25 -> ProjectionOuter=50
    // -> Consumer=10
    // =========================================================================
    {
        let mut input_session = engine.input_session().await;
        input_session.set_input(Variable(0), 5_i64).await;
    }

    {
        let tracked = engine.clone().tracked().await;
        let result = tracked.query(&ChainedProjectionConsumer).await;
        // 5 * 3 = 15, 15 + 10 = 25, 25 * 2 = 50, 50 / 5 = 10
        assert_eq!(result, 10);
    }

    // Verify initial execution counts - each executor should run once
    assert_eq!(firewall_ex.0.load(Ordering::SeqCst), 1);
    assert_eq!(inner_proj_ex.0.load(Ordering::SeqCst), 1);
    assert_eq!(outer_proj_ex.0.load(Ordering::SeqCst), 1);
    assert_eq!(consumer_ex.0.load(Ordering::SeqCst), 1);

    // =========================================================================
    // Test Case 2: Change Variable(0) - should propagate through entire chain
    // Variable(0)=10 -> FirewallA=30 -> ProjectionInner=40 ->
    // ProjectionOuter=80 -> Consumer=16
    // =========================================================================
    {
        let mut input_session = engine.input_session().await;
        input_session.set_input(Variable(0), 10_i64).await;
    }

    // Should have 1 dirtied edge: FirewallA -> Variable(0)
    let tracked = engine.clone().tracked().await;
    assert_eq!(tracked.get_dirtied_edges_count(), 1);
    drop(tracked);

    {
        let tracked = engine.clone().tracked().await;
        let result = tracked.query(&ChainedProjectionConsumer).await;
        // 10 * 3 = 30, 30 + 10 = 40, 40 * 2 = 80, 80 / 5 = 16
        assert_eq!(result, 16);
    }

    // All executors should have run once more due to the change propagating
    assert_eq!(firewall_ex.0.load(Ordering::SeqCst), 2); // +1
    assert_eq!(inner_proj_ex.0.load(Ordering::SeqCst), 2); // +1 (backward prop)
    assert_eq!(outer_proj_ex.0.load(Ordering::SeqCst), 2); // +1 (backward prop)
    assert_eq!(consumer_ex.0.load(Ordering::SeqCst), 2); // +1

    // Should have 4 dirtied edges now:
    // - Original 1 from previous assertion
    // - ProjectionInner -> FirewallA
    // - ProjectionOuter -> ProjectionInner
    // - Consumer -> ProjectionOuter
    let tracked = engine.clone().tracked().await;
    assert_eq!(tracked.get_dirtied_edges_count(), 4);
    drop(tracked);

    // =========================================================================
    // Test Case 3: Change Variable(0) to same value - no recomputation should
    // occur
    // =========================================================================
    {
        let mut input_session = engine.input_session().await;
        input_session.set_input(Variable(0), 10_i64).await; // Same value
    }

    // 0 dirtied edges since value didn't change
    let tracked = engine.clone().tracked().await;
    assert_eq!(tracked.get_dirtied_edges_count(), 0);
    drop(tracked);

    // Re-query - everything should be cached
    {
        let tracked = engine.clone().tracked().await;
        let result = tracked.query(&ChainedProjectionConsumer).await;
        assert_eq!(result, 16); // Same as before
    }

    // No executors should have run additional times
    assert_eq!(firewall_ex.0.load(Ordering::SeqCst), 2); // unchanged
    assert_eq!(inner_proj_ex.0.load(Ordering::SeqCst), 2); // unchanged
    assert_eq!(outer_proj_ex.0.load(Ordering::SeqCst), 2); // unchanged
    assert_eq!(consumer_ex.0.load(Ordering::SeqCst), 2); // unchanged

    // =========================================================================
    // Test Case 4: Query intermediate projections independently
    // =========================================================================
    {
        let tracked = engine.clone().tracked().await;

        let firewall_result = tracked.query(&ChainedFirewallA).await;
        assert_eq!(firewall_result, 30); // 10 * 3

        let inner_result = tracked.query(&ChainedProjectionInner).await;
        assert_eq!(inner_result, 40); // 30 + 10

        let outer_result = tracked.query(&ChainedProjectionOuter).await;
        assert_eq!(outer_result, 80); // 40 * 2
    }

    // No additional executions should occur since everything is cached
    assert_eq!(firewall_ex.0.load(Ordering::SeqCst), 2); // unchanged
    assert_eq!(inner_proj_ex.0.load(Ordering::SeqCst), 2); // unchanged
    assert_eq!(outer_proj_ex.0.load(Ordering::SeqCst), 2); // unchanged
    assert_eq!(consumer_ex.0.load(Ordering::SeqCst), 2); // unchanged
}

// ============================================================================
// Test: Chained Projections with Unchanged Intermediate Result
// ============================================================================
//
// Graph structure:
//   Variable(0) -> IdentityFirewall (passes value through)
//                      |
//                      v
//               AbsoluteValueProjection (takes absolute value) [PROJECTION]
//                      |
//                      v
//               DoubleProjection (doubles the value) [PROJECTION]
//                      |
//                      v
//               AbsChainConsumer (adds 100)
//
// This tests that when an intermediate projection's output doesn't change
// (e.g., |-5| = |5| = 5), the dirty propagation stops early and downstream
// projections and consumers are NOT recomputed.

/// A firewall query that passes the input value through unchanged.
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
    Encode,
    Decode,
)]
pub struct IdentityFirewall;

impl Query for IdentityFirewall {
    type Value = i64;
}

#[derive(Debug, Default)]
pub struct IdentityFirewallExecutor(pub AtomicUsize);

impl<C: Config> Executor<IdentityFirewall, C> for IdentityFirewallExecutor {
    async fn execute(
        &self,
        _query: &IdentityFirewall,
        engine: &TrackedEngine<C>,
    ) -> i64 {
        self.0.fetch_add(1, Ordering::SeqCst);

        engine.query(&Variable(0)).await
    }

    fn execution_style() -> qbice::ExecutionStyle {
        qbice::ExecutionStyle::Firewall
    }
}

/// Projection that computes absolute value - key for testing unchanged output.
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
    Encode,
    Decode,
)]
pub struct AbsoluteValueProjection;

impl Query for AbsoluteValueProjection {
    type Value = i64;
}

#[derive(Debug, Default)]
pub struct AbsoluteValueProjectionExecutor(pub AtomicUsize);

impl<C: Config> Executor<AbsoluteValueProjection, C>
    for AbsoluteValueProjectionExecutor
{
    async fn execute(
        &self,
        _query: &AbsoluteValueProjection,
        engine: &TrackedEngine<C>,
    ) -> i64 {
        self.0.fetch_add(1, Ordering::SeqCst);

        let val = engine.query(&IdentityFirewall).await;
        val.abs()
    }

    fn execution_style() -> qbice::ExecutionStyle {
        qbice::ExecutionStyle::Projection
    }
}

/// Second projection in chain - doubles the absolute value.
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
    Encode,
    Decode,
)]
pub struct DoubleProjection;

impl Query for DoubleProjection {
    type Value = i64;
}

#[derive(Debug, Default)]
pub struct DoubleProjectionExecutor(pub AtomicUsize);

impl<C: Config> Executor<DoubleProjection, C> for DoubleProjectionExecutor {
    async fn execute(
        &self,
        _query: &DoubleProjection,
        engine: &TrackedEngine<C>,
    ) -> i64 {
        self.0.fetch_add(1, Ordering::SeqCst);

        let abs_val = engine.query(&AbsoluteValueProjection).await;
        abs_val * 2
    }

    fn execution_style() -> qbice::ExecutionStyle {
        qbice::ExecutionStyle::Projection
    }
}

/// Consumer that adds 100 to the doubled absolute value.
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
    Encode,
    Decode,
)]
pub struct AbsChainConsumer;

impl Query for AbsChainConsumer {
    type Value = i64;
}

#[derive(Debug, Default)]
pub struct AbsChainConsumerExecutor(pub AtomicUsize);

impl<C: Config> Executor<AbsChainConsumer, C> for AbsChainConsumerExecutor {
    async fn execute(
        &self,
        _query: &AbsChainConsumer,
        engine: &TrackedEngine<C>,
    ) -> i64 {
        self.0.fetch_add(1, Ordering::SeqCst);

        let doubled = engine.query(&DoubleProjection).await;
        doubled + 100
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn chained_projections_unchanged_intermediate() {
    let tempdir = tempdir().unwrap();
    let mut engine = create_test_engine(&tempdir).await;

    let firewall_ex = Arc::new(IdentityFirewallExecutor::default());
    let abs_proj_ex = Arc::new(AbsoluteValueProjectionExecutor::default());
    let double_proj_ex = Arc::new(DoubleProjectionExecutor::default());
    let consumer_ex = Arc::new(AbsChainConsumerExecutor::default());

    engine.register_executor(firewall_ex.clone());
    engine.register_executor(abs_proj_ex.clone());
    engine.register_executor(double_proj_ex.clone());
    engine.register_executor(consumer_ex.clone());

    let engine = Arc::new(engine);

    // =========================================================================
    // Test Case 1: Initial query with negative value
    // Variable(0)=-5 -> Firewall=-5 -> Abs=5 -> Double=10 -> Consumer=110
    // =========================================================================
    {
        let mut input_session = engine.input_session().await;
        input_session.set_input(Variable(0), -5_i64).await;
    }

    {
        let tracked = engine.clone().tracked().await;
        let result = tracked.query(&AbsChainConsumer).await;
        // |-5| = 5, 5 * 2 = 10, 10 + 100 = 110
        assert_eq!(result, 110);
    }

    // Verify initial execution counts - each executor should run once
    assert_eq!(firewall_ex.0.load(Ordering::SeqCst), 1);
    assert_eq!(abs_proj_ex.0.load(Ordering::SeqCst), 1);
    assert_eq!(double_proj_ex.0.load(Ordering::SeqCst), 1);
    assert_eq!(consumer_ex.0.load(Ordering::SeqCst), 1);

    // =========================================================================
    // Test Case 2: Change Variable(0) from -5 to 5
    // The firewall output changes (-5 -> 5), but the absolute value stays the
    // same (5), so DoubleProjection and Consumer should NOT be recomputed
    // =========================================================================
    {
        let mut input_session = engine.input_session().await;
        input_session.set_input(Variable(0), 5_i64).await;
    }

    // Should have 1 dirtied edge: IdentityFirewall -> Variable(0)
    let tracked = engine.clone().tracked().await;
    assert_eq!(tracked.get_dirtied_edges_count(), 1);
    drop(tracked);

    {
        let tracked = engine.clone().tracked().await;
        let result = tracked.query(&AbsChainConsumer).await;
        // |5| = 5, 5 * 2 = 10, 10 + 100 = 110 (same result)
        assert_eq!(result, 110);

        // Should have 2 dirtied edges now:
        // - Original 1 from previous assertion
        // - AbsoluteValueProjection -> IdentityFirewall
        assert_eq!(tracked.get_dirtied_edges_count(), 2);
    }

    // Firewall should recompute (input changed)
    assert_eq!(firewall_ex.0.load(Ordering::SeqCst), 2); // +1
    // AbsoluteValue projection should recompute (its input changed)
    assert_eq!(abs_proj_ex.0.load(Ordering::SeqCst), 2); // +1
    // DoubleProjection should NOT recompute - AbsoluteValue output didn't
    // change
    assert_eq!(double_proj_ex.0.load(Ordering::SeqCst), 1); // unchanged!
    // Consumer should NOT recompute - DoubleProjection output didn't change
    assert_eq!(consumer_ex.0.load(Ordering::SeqCst), 1); // unchanged!

    // =========================================================================
    // Test Case 3: Change Variable(0) to a different absolute value
    // Variable(0)=10 -> Firewall=10 -> Abs=10 -> Double=20 -> Consumer=120
    // =========================================================================
    {
        let mut input_session = engine.input_session().await;
        input_session.set_input(Variable(0), 10_i64).await;
    }

    {
        let tracked = engine.clone().tracked().await;
        let result = tracked.query(&AbsChainConsumer).await;
        // |10| = 10, 10 * 2 = 20, 20 + 100 = 120
        assert_eq!(result, 120);
    }

    // All should recompute since the absolute value changed (5 -> 10)
    assert_eq!(firewall_ex.0.load(Ordering::SeqCst), 3); // +1
    assert_eq!(abs_proj_ex.0.load(Ordering::SeqCst), 3); // +1
    assert_eq!(double_proj_ex.0.load(Ordering::SeqCst), 2); // +1
    assert_eq!(consumer_ex.0.load(Ordering::SeqCst), 2); // +1

    // =========================================================================
    // Test Case 4: Change Variable(0) from 10 to -10
    // The firewall output changes (10 -> -10), but |10| = |-10| = 10
    // DoubleProjection and Consumer should NOT be recomputed
    // =========================================================================
    {
        let mut input_session = engine.input_session().await;
        input_session.set_input(Variable(0), -10_i64).await;
    }

    {
        let tracked = engine.clone().tracked().await;
        let result = tracked.query(&AbsChainConsumer).await;
        // |-10| = 10, 10 * 2 = 20, 20 + 100 = 120 (same result)
        assert_eq!(result, 120);
    }

    // Firewall and AbsoluteValue should recompute
    assert_eq!(firewall_ex.0.load(Ordering::SeqCst), 4); // +1
    assert_eq!(abs_proj_ex.0.load(Ordering::SeqCst), 4); // +1
    // DoubleProjection should NOT recompute - AbsoluteValue output didn't
    // change
    assert_eq!(double_proj_ex.0.load(Ordering::SeqCst), 2); // unchanged!
    // Consumer should NOT recompute - DoubleProjection output didn't change
    assert_eq!(consumer_ex.0.load(Ordering::SeqCst), 2); // unchanged!

    // =========================================================================
    // Test Case 5: Verify intermediate queries work correctly
    // =========================================================================
    {
        let tracked = engine.clone().tracked().await;

        let firewall_result = tracked.query(&IdentityFirewall).await;
        assert_eq!(firewall_result, -10);

        let abs_result = tracked.query(&AbsoluteValueProjection).await;
        assert_eq!(abs_result, 10);

        let double_result = tracked.query(&DoubleProjection).await;
        assert_eq!(double_result, 20);
    }

    // No additional executions should occur since everything is cached
    assert_eq!(firewall_ex.0.load(Ordering::SeqCst), 4); // unchanged
    assert_eq!(abs_proj_ex.0.load(Ordering::SeqCst), 4); // unchanged
    assert_eq!(double_proj_ex.0.load(Ordering::SeqCst), 2); // unchanged
    assert_eq!(consumer_ex.0.load(Ordering::SeqCst), 2); // unchanged
}
