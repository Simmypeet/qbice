//! Basic query execution tests.

use std::sync::Arc;

use qbice::{config::DefaultConfig, engine::Engine};
use qbice_integration_test::{
    DivisionExecutor, SafeDivision, SafeDivisionExecutor, Variable,
};

#[tokio::test]
async fn safe_division_basic() {
    let mut engine = Engine::<DefaultConfig>::default();

    let division_ex = Arc::new(DivisionExecutor::default());
    let safe_division_ex = Arc::new(SafeDivisionExecutor::default());

    engine.register_executor(division_ex.clone());
    engine.register_executor(safe_division_ex.clone());

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
