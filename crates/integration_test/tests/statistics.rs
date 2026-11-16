#![allow(missing_docs)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_const_for_fn)]

use std::{any::type_name, sync::Arc};

use qbice::{
    Decode, Encode, Engine, Identifiable, Observability, QueryStatistics,
    StableHash, StatisticsLevel, StatisticsReset, TrackedEngine,
    config::Config, executor::Executor, query::Query,
};
use qbice_integration_test::{
    Variable, create_test_engine, create_test_engine_with,
};
use tempfile::tempdir;

/// The square of a variable, a firewall.
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
    Encode,
    Decode,
)]
pub struct Square(pub Variable);

impl Query for Square {
    type Value = i64;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct SquareExecutor;

impl<C: Config> Executor<Square, C> for SquareExecutor {
    async fn execute(&self, query: &Square, engine: &TrackedEngine<C>) -> i64 {
        let value = engine.query(&query.0).await;
        value * value
    }

    fn execution_style() -> qbice::ExecutionStyle {
        qbice::ExecutionStyle::Firewall
    }
}

/// The negation of the square of a variable.
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
    Encode,
    Decode,
)]
pub struct NegativeSquare(pub Variable);

impl Query for NegativeSquare {
    type Value = i64;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct NegativeSquareExecutor;

impl<C: Config> Executor<NegativeSquare, C> for NegativeSquareExecutor {
    async fn execute(
        &self,
        query: &NegativeSquare,
        engine: &TrackedEngine<C>,
    ) -> i64 {
        -engine.query(&Square(query.0)).await
    }
}

/// Whether a variable is odd.
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
    Encode,
    Decode,
)]
pub struct IsOdd(pub Variable);

impl Query for IsOdd {
    type Value = bool;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct IsOddExecutor;

impl<C: Config> Executor<IsOdd, C> for IsOddExecutor {
    async fn execute(&self, query: &IsOdd, engine: &TrackedEngine<C>) -> bool {
        engine.query(&query.0).await % 2 != 0
    }
}

/// Describes whether a variable is odd.
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
    Encode,
    Decode,
)]
pub struct Parity(pub Variable);

impl Query for Parity {
    type Value = String;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ParityExecutor;

impl<C: Config> Executor<Parity, C> for ParityExecutor {
    async fn execute(
        &self,
        query: &Parity,
        engine: &TrackedEngine<C>,
    ) -> String {
        if engine.query(&IsOdd(query.0)).await { "odd" } else { "even" }
            .to_owned()
    }
}

/// The statistics of the query type `Q`, or all zero if it had no work.
fn of<Q: Identifiable, C: Config>(engine: &Engine<C>) -> QueryStatistics {
    engine.statistics().get(Q::STABLE_TYPE_ID).copied().unwrap_or_default()
}

async fn set(engine: &Arc<Engine<impl Config>>, value: i64) {
    let mut input_session = engine.input_session().await;
    input_session.set_input(Variable(0), value).await;
}

// input: Variable -> Square (firewall) -> NegativeSquare
// output: fresh executions first, then a firewall that stops an unchanged
// value, then one that passes a changed value on
#[tokio::test]
async fn counts_executions_through_a_firewall() {
    let tempdir = tempdir().unwrap();
    let mut engine = create_test_engine(&tempdir).await;
    engine.register_executor(Arc::new(SquareExecutor));
    engine.register_executor(Arc::new(NegativeSquareExecutor));
    let engine = Arc::new(engine);

    // every query is computed for the first time
    set(&engine, 3).await;
    let tracked = engine.clone().tracked().await;
    assert_eq!(tracked.query(&NegativeSquare(Variable(0))).await, -9);
    drop(tracked);

    assert_eq!(of::<Square, _>(&engine).fresh(), 1);
    assert_eq!(of::<NegativeSquare, _>(&engine).fresh(), 1);
    assert_eq!(
        engine.get_execution_style_from_stable_type_id(Square::STABLE_TYPE_ID),
        Some(qbice::ExecutionStyle::Firewall)
    );
    assert_eq!(
        engine.get_execution_style_from_stable_type_id(
            NegativeSquare::STABLE_TYPE_ID
        ),
        Some(qbice::ExecutionStyle::Normal)
    );
    assert_eq!(engine.statistics().total().executed(), 2);

    // the square stays 9, so the firewall stops the change
    set(&engine, -3).await;
    let tracked = engine.clone().tracked().await;
    assert_eq!(tracked.query(&NegativeSquare(Variable(0))).await, -9);
    drop(tracked);

    let square = of::<Square, _>(&engine);
    assert_eq!(square.dirtied_edges(), 1);
    assert_eq!(square.recomputed_unchanged(), 1);
    assert_eq!(square.recomputed_changed(), 0);
    assert_eq!(of::<NegativeSquare, _>(&engine).dirtied_edges(), 0);
    assert_eq!(of::<NegativeSquare, _>(&engine).executed(), 0);
    assert_eq!(engine.statistics().total().executed(), 1);

    // the square becomes 16, which the firewall passes on
    set(&engine, 4).await;
    let tracked = engine.clone().tracked().await;
    assert_eq!(tracked.query(&NegativeSquare(Variable(0))).await, -16);
    drop(tracked);

    assert_eq!(of::<Square, _>(&engine).recomputed_changed(), 1);
    assert_eq!(of::<NegativeSquare, _>(&engine).dirtied_edges(), 1);
    assert_eq!(of::<NegativeSquare, _>(&engine).recomputed_changed(), 1);
    assert_eq!(engine.statistics().total().fresh(), 0);
}

// input: Variable -> IsOdd -> Parity, all normal queries
// premise: the variable changes from 3 to 5, which IsOdd maps to the same
// value
// output: both queries dirtied, IsOdd recomputed unchanged, Parity verified
// clean without executing
#[tokio::test]
async fn counts_a_recompute_that_changes_nothing() {
    let tempdir = tempdir().unwrap();
    let mut engine = create_test_engine(&tempdir).await;
    engine.register_executor(Arc::new(IsOddExecutor));
    engine.register_executor(Arc::new(ParityExecutor));
    let engine = Arc::new(engine);

    set(&engine, 3).await;
    let tracked = engine.clone().tracked().await;
    assert_eq!(tracked.query(&Parity(Variable(0))).await, "odd");
    drop(tracked);

    set(&engine, 5).await;
    let tracked = engine.clone().tracked().await;
    assert_eq!(tracked.query(&Parity(Variable(0))).await, "odd");
    drop(tracked);

    let is_odd = of::<IsOdd, _>(&engine);
    assert_eq!(is_odd.dirtied_edges(), 1);
    assert_eq!(is_odd.recomputed_unchanged(), 1);
    assert_eq!(is_odd.edges_checked(), 1);

    let parity = of::<Parity, _>(&engine);
    assert_eq!(parity.dirtied_edges(), 1);
    assert_eq!(parity.executed(), 0);
    assert_eq!(parity.verified_clean(), 1);
    assert_eq!(parity.edges_checked(), 1);
}

// output: no counters while the statistics are off
#[tokio::test]
async fn counts_nothing_while_disabled() {
    let tempdir = tempdir().unwrap();
    let mut engine =
        create_test_engine_with(&tempdir, Observability::default()).await;
    engine.register_executor(Arc::new(IsOddExecutor));
    let engine = Arc::new(engine);

    set(&engine, 3).await;
    let tracked = engine.clone().tracked().await;
    assert!(tracked.query(&IsOdd(Variable(0))).await);
    drop(tracked);

    assert_eq!(engine.statistics().total(), QueryStatistics::default());
}

// input: Variable -> Square (firewall) -> NegativeSquare, counted by edge
// premise: the variable changes from 3 to -3, which Square maps to the same
// value
// output: the edge from the input to the firewall dirtied, where the
// propagation stops
#[tokio::test]
async fn counts_dirtied_edges_by_callee_and_caller() {
    let tempdir = tempdir().unwrap();
    let mut engine = create_test_engine(&tempdir).await;
    engine.register_executor(Arc::new(SquareExecutor));
    engine.register_executor(Arc::new(NegativeSquareExecutor));
    let engine = Arc::new(engine);

    set(&engine, 3).await;
    let tracked = engine.clone().tracked().await;
    assert_eq!(tracked.query(&NegativeSquare(Variable(0))).await, -9);
    drop(tracked);

    set(&engine, -3).await;
    let tracked = engine.clone().tracked().await;
    assert_eq!(tracked.query(&NegativeSquare(Variable(0))).await, -9);
    drop(tracked);

    let statistics = engine.statistics();
    assert_eq!(statistics.dirtied_edges().collect::<Vec<_>>(), [(
        Variable::STABLE_TYPE_ID,
        Square::STABLE_TYPE_ID,
        1
    )]);
    assert_eq!(
        statistics.dirtied_edges_between(
            Variable::STABLE_TYPE_ID,
            Square::STABLE_TYPE_ID
        ),
        1
    );
    assert_eq!(of::<Square, _>(&engine).propagation_stops(), 1);
    assert_eq!(of::<NegativeSquare, _>(&engine).propagation_stops(), 0);

    // the input type has no executor to name it, so it shows as its ID
    let display = statistics
        .display(|query_type| {
            engine.get_type_name_from_stable_type_id(query_type)
        })
        .to_string();
    assert!(
        display.contains(&format!(
            "1 dirtied edges {:?} -> {}",
            Variable::STABLE_TYPE_ID,
            type_name::<Square>()
        )),
        "{display}"
    );
}

// output: the name and execution style of a query type with an executor,
// and none for an input type, from the engine and a tracked engine
#[tokio::test]
async fn names_query_types_by_stable_type_id() {
    let tempdir = tempdir().unwrap();
    let mut engine = create_test_engine(&tempdir).await;
    engine.register_executor(Arc::new(IsOddExecutor));
    let engine = Arc::new(engine);

    assert_eq!(
        engine.get_type_name_from_stable_type_id(IsOdd::STABLE_TYPE_ID),
        Some(type_name::<IsOdd>())
    );
    assert_eq!(
        engine.get_execution_style_from_stable_type_id(IsOdd::STABLE_TYPE_ID),
        Some(qbice::ExecutionStyle::Normal)
    );

    set(&engine, 3).await;
    let tracked = engine.clone().tracked().await;
    assert_eq!(
        tracked.get_type_name_from_stable_type_id(IsOdd::STABLE_TYPE_ID),
        Some(type_name::<IsOdd>())
    );
    assert_eq!(
        tracked.get_type_name_from_stable_type_id(Variable::STABLE_TYPE_ID),
        None
    );
    assert_eq!(
        tracked
            .get_execution_style_from_stable_type_id(Variable::STABLE_TYPE_ID),
        None
    );
}

// input: Variable -> IsOdd, counted by query type only
// output: the dirtied edge counted for IsOdd, but no edge by callee and
// caller
#[tokio::test]
async fn counts_no_edges_by_query_type() {
    let tempdir = tempdir().unwrap();
    let mut engine = create_test_engine_with(
        &tempdir,
        Observability::builder()
            .statistics(StatisticsLevel::ByQueryType)
            .build(),
    )
    .await;
    engine.register_executor(Arc::new(IsOddExecutor));
    let engine = Arc::new(engine);

    for value in [3, 4] {
        set(&engine, value).await;
        let tracked = engine.clone().tracked().await;
        assert_eq!(tracked.query(&IsOdd(Variable(0))).await, value % 2 != 0);
    }

    assert_eq!(of::<IsOdd, _>(&engine).dirtied_edges(), 1);
    assert_eq!(engine.statistics().dirtied_edges().count(), 0);
}

// input: Variable -> Square (firewall) -> NegativeSquare, counted for the
// engine's lifetime
// premise: the variable changes from 3 to 4
// output: the counters keep the first computation after the commit, and the
// work of the edit alone is the difference of two reads
#[tokio::test]
async fn counts_for_the_lifetime_without_reset() {
    let tempdir = tempdir().unwrap();
    let mut engine = create_test_engine_with(
        &tempdir,
        Observability::builder()
            .statistics(StatisticsLevel::ByEdge)
            .reset(StatisticsReset::Never)
            .build(),
    )
    .await;
    engine.register_executor(Arc::new(SquareExecutor));
    engine.register_executor(Arc::new(NegativeSquareExecutor));
    let engine = Arc::new(engine);

    set(&engine, 3).await;
    let tracked = engine.clone().tracked().await;
    assert_eq!(tracked.query(&NegativeSquare(Variable(0))).await, -9);
    drop(tracked);
    let before = engine.statistics();

    set(&engine, 4).await;
    let tracked = engine.clone().tracked().await;
    assert_eq!(tracked.query(&NegativeSquare(Variable(0))).await, -16);
    drop(tracked);
    let after = engine.statistics();

    // the fresh computations survive the commit
    assert_eq!(after.total().fresh(), 2);
    assert_eq!(after.total().recomputed_changed(), 2);

    let edit = after.since(&before);
    assert_eq!(edit.total().fresh(), 0);
    assert_eq!(
        edit.get(Square::STABLE_TYPE_ID).unwrap().recomputed_changed(),
        1
    );
    assert_eq!(
        edit.get(NegativeSquare::STABLE_TYPE_ID).unwrap().recomputed_changed(),
        1
    );
    assert_eq!(
        edit.dirtied_edges_between(
            Square::STABLE_TYPE_ID,
            NegativeSquare::STABLE_TYPE_ID
        ),
        1
    );
}
