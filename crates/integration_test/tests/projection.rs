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
    let mut engine = Engine::<DefaultConfig>::new();

    let slow_square_ex = Arc::new(SlowSquareExecutor::default());
    let collect_doubled_ex =
        Arc::new(CollectDoubledSquareVariablesExecutor::default());
    let double_square_ex = Arc::new(DoubleSquareExecutor::default());

    engine.register_executor(slow_square_ex.clone());
    engine.register_executor(collect_doubled_ex.clone());
    engine.register_executor(double_square_ex.clone());

    // Initialize 4 variables
    {
        let mut input_session = engine.input_session();
        let mut target_vars = Vec::new();

        for i in 0..4 {
            input_session.set_input(Variable(i), i.cast_signed() + 1);
            target_vars.push(Variable(i));
        }

        input_session.set_input(VariableTarget, Arc::from(target_vars));
    }

    let mut engine = Arc::new(engine);

    // Query all three projections
    {
        let tracked = engine.clone().tracked();
        let v0 = tracked.query(&DoubleSquare(Variable(0))).await.unwrap();
        let v1 = tracked.query(&DoubleSquare(Variable(1))).await.unwrap();
        let v2 = tracked.query(&DoubleSquare(Variable(2))).await.unwrap();

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
        let input_session =
            &mut Arc::get_mut(&mut engine).unwrap().input_session();
        input_session.set_input(Variable(0), 10);
    }

    // Check dirtied edges - should only be 2:
    // CollectDoubledSquareVariables -> SlowSquare(0) -> Variable(0)
    let tracked = engine.clone().tracked();
    assert_eq!(tracked.get_dirtied_edges_count(), 2);
    drop(tracked);

    // Re-query projection for Variable(1) - this triggers firewall repair
    // since the firewall is dirty (due to SlowSquare(0) -> Variable(0) being
    // dirty)
    {
        let tracked = engine.clone().tracked();
        let v1 = tracked.query(&DoubleSquare(Variable(1))).await.unwrap();
        assert_eq!(v1, Some(8));
    }

    // Firewall was recomputed (since it's dirty)
    // SlowSquare(0) recomputed because Variable(0) changed
    // Note: Other SlowSquare queries may also be re-queried during firewall
    // recomputation but should hit cache if unchanged
    assert_eq!(collect_doubled_ex.0.load(Ordering::SeqCst), 2); // +1 for firewall

    // DoubleSquare(1) was the original caller (queried), so it's skipped in
    // backward prop but it still recomputes (+1). DoubleSquare(0) and
    // DoubleSquare(2) are invoked via backward propagation (+2). Total: 3 +
    // 1 + 2 = 6
    assert_eq!(double_square_ex.0.load(Ordering::SeqCst), 6); // +3

    // in total there should be 5 dirtied edges now:
    // - Original 2 from previous assertion
    // - Each of the 3 DoubleSquare -> CollectDoubledSquareVariables edges
    let tracked = engine.clone().tracked();
    assert_eq!(tracked.get_dirtied_edges_count(), 5);
    drop(tracked);

    // Query the changed projection
    {
        let tracked = engine.clone().tracked();
        let v0 = tracked.query(&DoubleSquare(Variable(0))).await.unwrap();
        // 10^2 * 2 = 200
        assert_eq!(v0, Some(200));
    }

    // No additional recomputation needed - already computed during backward
    // propagation
    assert_eq!(double_square_ex.0.load(Ordering::SeqCst), 6);
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
    ) -> Result<Arc<HashMap<u64, i64>>, qbice::executor::CyclicError> {
        self.0.fetch_add(1, Ordering::SeqCst);

        let mut map = HashMap::new();
        // Query variables 0-3
        for i in 0..4 {
            let val = engine.query(&Variable(i)).await?;
            map.insert(i, val);
        }

        Ok(Arc::new(map))
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
    ) -> Result<i64, qbice::executor::CyclicError> {
        self.0.fetch_add(1, Ordering::SeqCst);

        let map = engine.query(&SimpleFirewall).await?;
        let val = map.get(&query.0).copied().unwrap_or(0);

        Ok(val * 2)
    }

    fn execution_style() -> qbice::ExecutionStyle {
        qbice::ExecutionStyle::Projection
    }
}

/// Second level projection - depends on first level and adds 100.
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
pub struct ProjectionLevel2(pub u64);

impl Query for ProjectionLevel2 {
    type Value = i64;
}

#[derive(Debug, Default)]
pub struct ProjectionLevel2Executor(pub AtomicUsize);

impl<C: Config> Executor<ProjectionLevel2, C> for ProjectionLevel2Executor {
    async fn execute(
        &self,
        query: &ProjectionLevel2,
        engine: &TrackedEngine<C>,
    ) -> Result<i64, qbice::executor::CyclicError> {
        self.0.fetch_add(1, Ordering::SeqCst);

        let doubled = engine.query(&ProjectionLevel1(query.0)).await?;

        Ok(doubled + 100)
    }

    fn execution_style() -> qbice::ExecutionStyle {
        qbice::ExecutionStyle::Projection
    }
}

/// Normal query that consumes the projection chain.
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
pub struct ConsumeProjectionChain(pub u64);

impl Query for ConsumeProjectionChain {
    type Value = i64;
}

#[derive(Debug, Default)]
pub struct ConsumeProjectionChainExecutor(pub AtomicUsize);

impl<C: Config> Executor<ConsumeProjectionChain, C>
    for ConsumeProjectionChainExecutor
{
    async fn execute(
        &self,
        query: &ConsumeProjectionChain,
        engine: &TrackedEngine<C>,
    ) -> Result<i64, qbice::executor::CyclicError> {
        self.0.fetch_add(1, Ordering::SeqCst);

        let proj2_val = engine.query(&ProjectionLevel2(query.0)).await?;

        // Add another transformation
        Ok(proj2_val * 3)
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn chained_projections() {
    let mut engine = Engine::<DefaultConfig>::new();

    let firewall_ex = Arc::new(SimpleFirewallExecutor::default());
    let proj1_ex = Arc::new(ProjectionLevel1Executor::default());
    let proj2_ex = Arc::new(ProjectionLevel2Executor::default());
    let consumer_ex = Arc::new(ConsumeProjectionChainExecutor::default());

    engine.register_executor(firewall_ex.clone());
    engine.register_executor(proj1_ex.clone());
    engine.register_executor(proj2_ex.clone());
    engine.register_executor(consumer_ex.clone());

    // Initialize variables
    {
        let mut input_session = engine.input_session();
        for i in 0..4 {
            input_session.set_input(Variable(i), i.cast_signed() + 1);
        }
    }

    let mut engine = Arc::new(engine);

    // Query through the chain for Variable(0) and Variable(1)
    {
        let tracked = engine.clone().tracked();

        // Variable(0): value=1, proj1=2, proj2=102, consumer=306
        let c0 = tracked.query(&ConsumeProjectionChain(0)).await.unwrap();
        assert_eq!(c0, 306);

        // Variable(1): value=2, proj1=4, proj2=104, consumer=312
        let c1 = tracked.query(&ConsumeProjectionChain(1)).await.unwrap();
        assert_eq!(c1, 312);
    }

    // Initial execution counts
    assert_eq!(firewall_ex.0.load(Ordering::SeqCst), 1);
    assert_eq!(proj1_ex.0.load(Ordering::SeqCst), 2);
    assert_eq!(proj2_ex.0.load(Ordering::SeqCst), 2);
    assert_eq!(consumer_ex.0.load(Ordering::SeqCst), 2);

    // Change Variable(0) - this should dirty only:
    // SimpleFirewall -> Variable(0)
    {
        let input_session =
            &mut Arc::get_mut(&mut engine).unwrap().input_session();
        input_session.set_input(Variable(0), 10);
    }

    // Dirtied edges should be minimal: only 1 edge (Firewall -> Variable(0))
    let tracked = engine.clone().tracked();
    assert_eq!(tracked.get_dirtied_edges_count(), 1);
    drop(tracked);

    // Query ConsumeProjectionChain(1) - Variable(1) didn't change
    {
        let tracked = engine.clone().tracked();
        let c1 = tracked.query(&ConsumeProjectionChain(1)).await.unwrap();
        assert_eq!(c1, 312); // Same value
    }

    // Firewall recomputed, ProjectionLevel1(0) and (1) invoked via backward
    // propagation when the firewall output changes.
    assert_eq!(firewall_ex.0.load(Ordering::SeqCst), 2); // +1
    // Both proj1(0) and proj1(1) are invoked via backward prop since they
    // both depend on the firewall that changed
    assert_eq!(proj1_ex.0.load(Ordering::SeqCst), 4); // +2 (backward prop for both)
    // proj2(0) invoked via backward prop from proj1(0) which changed,
    // proj2(1) queried directly via consumer(1)
    assert_eq!(proj2_ex.0.load(Ordering::SeqCst), 3); // +1 (only queried path)
    // consumer(1) is queried and verifies deps, but projection filtered
    // the Variable(1) path so consumer(1) doesn't recompute
    assert_eq!(consumer_ex.0.load(Ordering::SeqCst), 2); // unchanged - proj2(1) unchanged

    // intotal there should be 5 dirtied edges now:
    // - Original 1 from previous assertion
    // - ProjectionLevel1(0) -> SimpleFirewall
    // - ProjectionLevel1(1) -> SimpleFirewall
    // - ProjectionLevel2(0) -> ProjectionLevel1(0)
    // - ConsumeProjectionChain(0) -> ProjectionLevel2(0)
    let tracked = engine.clone().tracked();
    assert_eq!(tracked.get_dirtied_edges_count(), 5);
    drop(tracked);

    // Query the changed path
    {
        let tracked = engine.clone().tracked();
        // Variable(0): value=10, proj1=20, proj2=120, consumer=360
        let c0 = tracked.query(&ConsumeProjectionChain(0)).await.unwrap();
        assert_eq!(c0, 360);
    }

    // consumer(0) now executed
    assert_eq!(consumer_ex.0.load(Ordering::SeqCst), 3); // +1

    // Verify dirtied edges after all queries - should include:
    // - Original: Firewall -> Variable(0)
    // - Backward prop from firewall: Proj1(0), Proj1(1) -> Firewall
    // - Backward prop from Proj1(0): Proj2(0) -> Proj1(0)
    // - Backward prop from Proj2(0): Consumer(0) -> Proj2(0)
    let tracked = engine.clone().tracked();

    let dirtied = tracked.get_dirtied_edges_count();

    // there should be at 6 dirtied edges
    // 4 from above +
    // Consumer(1) -> Proj2(1) which was not dirtied before but is now
    assert_eq!(dirtied, 5);
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
    ) -> Result<i64, qbice::executor::CyclicError> {
        self.0.fetch_add(1, Ordering::SeqCst);

        let p0 = engine.query(&ProjectionLevel1(0)).await?;
        let p1 = engine.query(&ProjectionLevel1(1)).await?;

        Ok(p0 + p1)
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn diamond_projection_pattern() {
    let mut engine = Engine::<DefaultConfig>::new();

    let firewall_ex = Arc::new(SimpleFirewallExecutor::default());
    let proj1_ex = Arc::new(ProjectionLevel1Executor::default());
    let combiner_ex = Arc::new(DiamondCombinerExecutor::default());

    engine.register_executor(firewall_ex.clone());
    engine.register_executor(proj1_ex.clone());
    engine.register_executor(combiner_ex.clone());

    // Initialize variables
    {
        let mut input_session = engine.input_session();
        for i in 0..4 {
            input_session.set_input(Variable(i), i.cast_signed() + 1);
        }
    }

    let mut engine = Arc::new(engine);

    // Query the combiner
    {
        let tracked = engine.clone().tracked();
        // Variable(0)=1 -> proj1=2, Variable(1)=2 -> proj1=4, combiner=6
        let result = tracked.query(&DiamondCombiner).await.unwrap();
        assert_eq!(result, 6);
    }

    assert_eq!(firewall_ex.0.load(Ordering::SeqCst), 1);
    assert_eq!(proj1_ex.0.load(Ordering::SeqCst), 2);
    assert_eq!(combiner_ex.0.load(Ordering::SeqCst), 1);

    // Change Variable(2) - neither projection depends on this directly
    // but firewall does
    {
        let input_session =
            &mut Arc::get_mut(&mut engine).unwrap().input_session();
        input_session.set_input(Variable(2), 100);
    }

    // Only 1 dirtied edge: Firewall -> Variable(2)
    let tracked = engine.clone().tracked();
    assert_eq!(tracked.get_dirtied_edges_count(), 1);
    drop(tracked);

    // Re-query combiner
    {
        let tracked = engine.clone().tracked();
        let result = tracked.query(&DiamondCombiner).await.unwrap();
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
    let tracked = engine.clone().tracked();
    assert_eq!(tracked.get_dirtied_edges_count(), 3);
    drop(tracked);

    // Now change Variable(0) which DOES affect proj1(0)
    {
        let input_session =
            &mut Arc::get_mut(&mut engine).unwrap().input_session();
        input_session.set_input(Variable(0), 50);
    }

    // Only 1 dirtied edge: Firewall -> Variable(0)
    let tracked = engine.clone().tracked();
    assert_eq!(tracked.get_dirtied_edges_count(), 1);
    drop(tracked);

    // Re-query combiner
    {
        let tracked = engine.clone().tracked();
        // Variable(0)=50 -> proj1=100, Variable(1)=2 -> proj1=4, combiner=104
        let result = tracked.query(&DiamondCombiner).await.unwrap();
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
    let tracked = engine.clone().tracked();
    assert_eq!(tracked.get_dirtied_edges_count(), 4);
    drop(tracked);
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
    ) -> Result<i64, qbice::executor::CyclicError> {
        self.0.fetch_add(1, Ordering::SeqCst);

        let a = engine.query(&Variable(0)).await?;
        let b = engine.query(&Variable(1)).await?;

        Ok(a + b)
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
    ) -> Result<i64, qbice::executor::CyclicError> {
        self.0.fetch_add(1, Ordering::SeqCst);

        let sum = engine.query(&SumFirewall).await?;
        Ok(sum)
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
    ) -> Result<i64, qbice::executor::CyclicError> {
        self.0.fetch_add(1, Ordering::SeqCst);

        let proj = engine.query(&SumProjection).await?;
        Ok(proj * 10)
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn firewall_same_output_no_propagation() {
    let mut engine = Engine::<DefaultConfig>::new();

    let firewall_ex = Arc::new(SumFirewallExecutor::default());
    let proj_ex = Arc::new(SumProjectionExecutor::default());
    let consumer_ex = Arc::new(SumConsumerExecutor::default());

    engine.register_executor(firewall_ex.clone());
    engine.register_executor(proj_ex.clone());
    engine.register_executor(consumer_ex.clone());

    // Initialize: Variable(0)=5, Variable(1)=5, sum=10
    {
        let mut input_session = engine.input_session();
        input_session.set_input(Variable(0), 5_i64);
        input_session.set_input(Variable(1), 5_i64);
    }

    let mut engine = Arc::new(engine);

    // Initial query
    {
        let tracked = engine.clone().tracked();
        let result = tracked.query(&SumConsumer).await.unwrap();
        assert_eq!(result, 100); // sum=10, consumer=100
    }

    assert_eq!(firewall_ex.0.load(Ordering::SeqCst), 1);
    assert_eq!(proj_ex.0.load(Ordering::SeqCst), 1);
    assert_eq!(consumer_ex.0.load(Ordering::SeqCst), 1);

    // Change Variable(0)=3, Variable(1)=7 - sum still equals 10!
    {
        let input_session =
            &mut Arc::get_mut(&mut engine).unwrap().input_session();
        input_session.set_input(Variable(0), 3_i64);
        input_session.set_input(Variable(1), 7_i64);
    }

    // 2 dirtied edges: Firewall -> Variable(0), Firewall -> Variable(1)
    let tracked = engine.clone().tracked();
    assert_eq!(tracked.get_dirtied_edges_count(), 2);
    drop(tracked);

    // Re-query
    {
        let tracked = engine.clone().tracked();
        let result = tracked.query(&SumConsumer).await.unwrap();
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
    let tracked = engine.clone().tracked();
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
async fn concurrent_projection_access() {
    let mut engine = Engine::<DefaultConfig>::new();

    let firewall_ex = Arc::new(SimpleFirewallExecutor::default());
    let proj1_ex = Arc::new(ProjectionLevel1Executor::default());

    engine.register_executor(firewall_ex.clone());
    engine.register_executor(proj1_ex.clone());

    // Initialize variables
    {
        let mut input_session = engine.input_session();
        for i in 0..4 {
            input_session.set_input(Variable(i), i.cast_signed() + 1);
        }
    }

    let engine = Arc::new(engine);

    // Spawn concurrent queries to all 4 projections
    let mut handles = Vec::new();
    for i in 0..4 {
        let engine = engine.clone();
        handles.push(tokio::spawn(async move {
            let tracked = engine.tracked();
            tracked.query(&ProjectionLevel1(i)).await
        }));
    }

    // Collect results
    let mut results = Vec::new();
    for handle in handles {
        results.push(handle.await.unwrap().unwrap());
    }

    // Verify results: (i+1) * 2 for each
    assert_eq!(results[0], 2); // 1 * 2
    assert_eq!(results[1], 4); // 2 * 2
    assert_eq!(results[2], 6); // 3 * 2
    assert_eq!(results[3], 8); // 4 * 2

    // Firewall should only execute once despite concurrent access
    assert_eq!(firewall_ex.0.load(Ordering::SeqCst), 1);

    // Each projection executed once
    assert_eq!(proj1_ex.0.load(Ordering::SeqCst), 4);
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
    ) -> Result<i64, qbice::executor::CyclicError> {
        self.0.fetch_add(1, Ordering::SeqCst);

        let val = engine.query(&Variable(0)).await?;
        Ok(val * 2)
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
    ) -> Result<i64, qbice::executor::CyclicError> {
        self.0.fetch_add(1, Ordering::SeqCst);

        let val = engine.query(&OuterFirewall).await?;
        Ok(val + 10)
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
    ) -> Result<i64, qbice::executor::CyclicError> {
        self.0.fetch_add(1, Ordering::SeqCst);

        let val = engine.query(&InnerFirewall).await?;
        Ok(val * 3)
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
    ) -> Result<i64, qbice::executor::CyclicError> {
        self.0.fetch_add(1, Ordering::SeqCst);

        let val = engine.query(&NestedProjection).await?;
        Ok(val + 1000)
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn nested_firewall_with_projection() {
    let mut engine = Engine::<DefaultConfig>::new();

    let outer_ex = Arc::new(OuterFirewallExecutor::default());
    let inner_ex = Arc::new(InnerFirewallExecutor::default());
    let proj_ex = Arc::new(NestedProjectionExecutor::default());
    let consumer_ex = Arc::new(NestedConsumerExecutor::default());

    engine.register_executor(outer_ex.clone());
    engine.register_executor(inner_ex.clone());
    engine.register_executor(proj_ex.clone());
    engine.register_executor(consumer_ex.clone());

    // Initialize
    {
        let mut input_session = engine.input_session();
        input_session.set_input(Variable(0), 5_i64);
    }

    let mut engine = Arc::new(engine);

    // Initial query
    // Variable(0)=5 -> Outer=10 -> Inner=20 -> Proj=60 -> Consumer=1060
    {
        let tracked = engine.clone().tracked();
        let result = tracked.query(&NestedConsumer).await.unwrap();
        assert_eq!(result, 1060);
    }

    assert_eq!(outer_ex.0.load(Ordering::SeqCst), 1);
    assert_eq!(inner_ex.0.load(Ordering::SeqCst), 1);
    assert_eq!(proj_ex.0.load(Ordering::SeqCst), 1);
    assert_eq!(consumer_ex.0.load(Ordering::SeqCst), 1);

    // Change input
    {
        let input_session =
            &mut Arc::get_mut(&mut engine).unwrap().input_session();
        input_session.set_input(Variable(0), 10_i64);
    }

    // Only 1 dirtied edge: OuterFirewall -> Variable(0)
    // InnerFirewall and beyond are NOT dirtied due to firewall boundary
    let tracked = engine.clone().tracked();
    assert_eq!(tracked.get_dirtied_edges_count(), 1);
    drop(tracked);

    // Re-query
    // Variable(0)=10 -> Outer=20 -> Inner=30 -> Proj=90 -> Consumer=1090
    {
        let tracked = engine.clone().tracked();
        let result = tracked.query(&NestedConsumer).await.unwrap();
        assert_eq!(result, 1090);
    }

    // All executors should have run again due to cascading changes
    assert_eq!(outer_ex.0.load(Ordering::SeqCst), 2);
    assert_eq!(inner_ex.0.load(Ordering::SeqCst), 2);
    assert_eq!(proj_ex.0.load(Ordering::SeqCst), 2);
    assert_eq!(consumer_ex.0.load(Ordering::SeqCst), 2);

    // Final dirtied edge count check
    let tracked = engine.clone().tracked();
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
    ) -> Result<i64, qbice::executor::CyclicError> {
        self.0.fetch_add(1, Ordering::SeqCst);

        let val = engine.query(&Variable(0)).await?;
        Ok(val * 2) // doubles
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
    ) -> Result<i64, qbice::executor::CyclicError> {
        self.0.fetch_add(1, Ordering::SeqCst);

        let val = engine.query(&Variable(1)).await?;
        Ok(val * 3) // triples
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
    ) -> Result<i64, qbice::executor::CyclicError> {
        self.0.fetch_add(1, Ordering::SeqCst);

        let a = engine.query(&FirewallA).await?;
        let b = engine.query(&FirewallB).await?;

        Ok(a + b)
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
    ) -> Result<i64, qbice::executor::CyclicError> {
        self.0.fetch_add(1, Ordering::SeqCst);

        let proj = engine.query(&DualFirewallProjection).await?;
        Ok(proj * 10)
    }
}

#[tokio::test(flavor = "multi_thread")]
#[allow(clippy::similar_names)]
async fn projection_with_two_firewalls() {
    let mut engine = Engine::<DefaultConfig>::new();

    let firewall_a_ex = Arc::new(FirewallAExecutor::default());
    let firewall_b_ex = Arc::new(FirewallBExecutor::default());
    let proj_ex = Arc::new(DualFirewallProjectionExecutor::default());
    let consumer_ex = Arc::new(DualFirewallConsumerExecutor::default());

    engine.register_executor(firewall_a_ex.clone());
    engine.register_executor(firewall_b_ex.clone());
    engine.register_executor(proj_ex.clone());
    engine.register_executor(consumer_ex.clone());

    // Initialize: Variable(0)=5, Variable(1)=10
    {
        let mut input_session = engine.input_session();
        input_session.set_input(Variable(0), 5_i64);
        input_session.set_input(Variable(1), 10_i64);
    }

    let mut engine = Arc::new(engine);

    // Initial query
    // FirewallA: 5*2=10, FirewallB: 10*3=30, Proj: 10+30=40, Consumer:
    // 40*10=400
    {
        let tracked = engine.clone().tracked();
        let result = tracked.query(&DualFirewallConsumer).await.unwrap();
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
        let input_session =
            &mut Arc::get_mut(&mut engine).unwrap().input_session();
        input_session.set_input(Variable(0), 10_i64);
    }

    // Only 1 dirtied edge: FirewallA -> Variable(0)
    let tracked = engine.clone().tracked();
    assert_eq!(tracked.get_dirtied_edges_count(), 1);
    drop(tracked);

    // Re-query
    // FirewallA: 10*2=20, FirewallB: 10*3=30 (unchanged), Proj: 20+30=50,
    // Consumer: 50*10=500
    {
        let tracked = engine.clone().tracked();
        let result = tracked.query(&DualFirewallConsumer).await.unwrap();
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
        let input_session =
            &mut Arc::get_mut(&mut engine).unwrap().input_session();
        input_session.set_input(Variable(1), 20_i64);
    }

    // Only 1 dirtied edge: FirewallB -> Variable(1)
    let tracked = engine.clone().tracked();
    assert_eq!(tracked.get_dirtied_edges_count(), 1);
    drop(tracked);

    // Re-query
    // FirewallA: 10*2=20 (unchanged), FirewallB: 20*3=60, Proj: 20+60=80,
    // Consumer: 80*10=800
    {
        let tracked = engine.clone().tracked();
        let result = tracked.query(&DualFirewallConsumer).await.unwrap();
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
        let input_session =
            &mut Arc::get_mut(&mut engine).unwrap().input_session();
        input_session.set_input(Variable(0), 15_i64);
        input_session.set_input(Variable(1), 25_i64);
    }

    // 2 dirtied edges: FirewallA -> Variable(0), FirewallB -> Variable(1)
    let tracked = engine.clone().tracked();
    assert_eq!(tracked.get_dirtied_edges_count(), 2);
    drop(tracked);

    // Re-query
    // FirewallA: 15*2=30, FirewallB: 25*3=75, Proj: 30+75=105,
    // Consumer: 105*10=1050
    {
        let tracked = engine.clone().tracked();
        let result = tracked.query(&DualFirewallConsumer).await.unwrap();
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
        let input_session =
            &mut Arc::get_mut(&mut engine).unwrap().input_session();
        // Set to same value - should not trigger any recomputation
        input_session.set_input(Variable(0), 15_i64);
    }

    // 0 dirtied edges since value didn't change
    let tracked = engine.clone().tracked();
    assert_eq!(tracked.get_dirtied_edges_count(), 0);
    drop(tracked);

    // Re-query - everything should be cached
    {
        let tracked = engine.clone().tracked();
        let result = tracked.query(&DualFirewallConsumer).await.unwrap();
        assert_eq!(result, 1050); // Same as before
    }

    // No executors should have run
    assert_eq!(firewall_a_ex.0.load(Ordering::SeqCst), 3); // unchanged
    assert_eq!(firewall_b_ex.0.load(Ordering::SeqCst), 3); // unchanged
    assert_eq!(proj_ex.0.load(Ordering::SeqCst), 4); // unchanged
    assert_eq!(consumer_ex.0.load(Ordering::SeqCst), 4); // unchanged
}
