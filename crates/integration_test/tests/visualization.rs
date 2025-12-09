//! Graph visualization snapshot tests.

#![allow(missing_docs)]

use std::sync::Arc;

use qbice::{config::DefaultConfig, engine::Engine};
use qbice_integration_test::{
    Division, DivisionExecutor, SafeDivision, SafeDivisionExecutor, Variable,
};

#[tokio::test]
async fn visualization_snapshot() {
    let mut engine = Engine::<DefaultConfig>::default();

    let division_ex = Arc::new(DivisionExecutor::default());
    let safe_division_ex = Arc::new(SafeDivisionExecutor::default());

    engine.register_executor(division_ex.clone());
    engine.register_executor(safe_division_ex.clone());

    let mut input_session = engine.input_session();

    input_session.set_input(Variable(0), 100);
    input_session.set_input(Variable(1), 10);
    input_session.set_input(Variable(2), 5);

    drop(input_session);

    let mut engine = Arc::new(engine);
    let tracked_engine = engine.clone().tracked();

    // Execute some queries to build the dependency graph
    let safe_div_query = SafeDivision::new(Variable(0), Variable(1));
    let div_query = Division::new(Variable(0), Variable(2));

    let _ = tracked_engine.query(&safe_div_query).await;
    let _ = tracked_engine.query(&div_query).await;

    // Drop tracked_engine to get mutable access
    drop(tracked_engine);

    // Get the graph snapshot starting from SafeDivision query
    let snapshot =
        Arc::get_mut(&mut engine).unwrap().snapshot_graph_from(&safe_div_query);

    // SafeDivision depends on:
    // - Variable(1) (divisor check)
    // - Division (if divisor != 0)
    //   - Variable(0) (dividend)
    //   - Variable(1) (divisor)
    // So we should have: SafeDivision, Division, Variable(0), Variable(1)
    assert!(
        snapshot.nodes.len() >= 3,
        "Expected at least 3 nodes (SafeDivision, Division, Variable(0), \
         Variable(1)), got {}",
        snapshot.nodes.len()
    );

    // Should have edges representing forward dependencies
    assert!(!snapshot.edges.is_empty(), "Expected some edges in the graph");

    // Verify we have both input and computed nodes
    let input_count = snapshot.nodes.iter().filter(|n| n.is_input).count();
    let computed_count = snapshot.nodes.iter().filter(|n| !n.is_input).count();

    assert!(input_count >= 2, "Expected at least 2 input nodes");
    assert!(computed_count >= 1, "Expected at least 1 computed node");

    // Get the graph snapshot starting from Division query (different root)
    let div_snapshot =
        Arc::get_mut(&mut engine).unwrap().snapshot_graph_from(&div_query);

    // Division depends on Variable(0) and Variable(2)
    assert!(
        div_snapshot.nodes.len() >= 3,
        "Expected at least 3 nodes (Division, Variable(0), Variable(2)), got \
         {}",
        div_snapshot.nodes.len()
    );
}
