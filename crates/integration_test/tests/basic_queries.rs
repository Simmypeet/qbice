//! Basic query execution tests.

use std::sync::Arc;

use qbice_integration_test::{
    DivisionExecutor, SafeDivision, SafeDivisionExecutor, Variable,
    create_test_engine,
};
use tempfile::tempdir;

#[tokio::test]
async fn safe_division_basic() {
    let tempdir = tempdir().unwrap();

    let mut engine = create_test_engine(&tempdir);

    let division_ex = Arc::new(DivisionExecutor::default());
    let safe_division_ex = Arc::new(SafeDivisionExecutor::default());

    engine.register_executor(division_ex.clone());
    engine.register_executor(safe_division_ex.clone());

    let engine = Arc::new(engine);

    {
        let mut input_session = engine.input_session().await;

        input_session.set_input(Variable(0), 42).await;
        input_session.set_input(Variable(1), 2).await;
    }

    let tracked_engine = engine.tracked().await;

    assert_eq!(
        tracked_engine
            .query(&SafeDivision::new(Variable(0), Variable(1)))
            .await,
        Some(21)
    );

    assert_eq!(division_ex.0.load(std::sync::atomic::Ordering::Relaxed), 1);
    assert_eq!(
        safe_division_ex.0.load(std::sync::atomic::Ordering::Relaxed),
        1
    );
}
