//! Tests for input changes and cache invalidation.

use std::sync::Arc;

use qbice_integration_test::{
    AbsoluteExecutor, AddTwoAbsolutes, AddTwoAbsolutesExecutor, Division,
    DivisionExecutor, SafeDivision, SafeDivisionExecutor, Variable,
    create_test_engine,
};
use tempfile::tempdir;

#[tokio::test]
async fn division_input_change() {
    let tempdir = tempdir().unwrap();
    let division_ex = Arc::new(DivisionExecutor::default());

    // session 1: initial inputs and query
    {
        let mut engine = create_test_engine(&tempdir).await;

        engine.register_executor(division_ex.clone());

        let engine = Arc::new(engine);

        {
            let mut input_session = engine.input_session().await;

            input_session.set_input(Variable(0), 40).await;
            input_session.set_input(Variable(1), 4).await;
        }

        let tracked_engine = engine.clone().tracked().await;

        assert_eq!(
            tracked_engine
                .query(&Division::new(Variable(0), Variable(1)))
                .await,
            10
        );

        assert_eq!(division_ex.0.load(std::sync::atomic::Ordering::Relaxed), 1);

        drop(tracked_engine);

        // session 2: change input and re-query

        {
            let mut input_session = engine.input_session().await;
            input_session.set_input(Variable(1), 2).await;

            input_session.commit().await;
        }

        let tracked_engine = engine.tracked().await;

        assert_eq!(
            tracked_engine
                .query(&Division::new(Variable(0), Variable(1)))
                .await,
            20
        );

        assert_eq!(division_ex.0.load(std::sync::atomic::Ordering::Relaxed), 2);
    }
}

#[tokio::test]
async fn safe_division_input_changes() {
    let tempdir = tempdir().unwrap();

    let division_ex = Arc::new(DivisionExecutor::default());
    let safe_division_ex = Arc::new(SafeDivisionExecutor::default());

    // session 1: initial inputs and query
    {
        let mut engine = create_test_engine(&tempdir).await;

        engine.register_executor(division_ex.clone());
        engine.register_executor(safe_division_ex.clone());

        let engine = Arc::new(engine);

        {
            let mut input_session = engine.input_session().await;

            input_session.set_input(Variable(0), 42).await;
            input_session.set_input(Variable(1), 2).await;
            input_session.commit().await;
        }

        let tracked_engine = engine.clone().tracked().await;

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

        // session 2: divide by zero
        drop(tracked_engine);

        {
            let mut input_session = engine.input_session().await;
            input_session.set_input(Variable(1), 0).await;
            input_session.commit().await;
        }

        let tracked_engine = engine.clone().tracked().await;

        assert_eq!(
            tracked_engine
                .query(&SafeDivision::new(Variable(0), Variable(1)))
                .await,
            None
        );

        // division executor should not have been called again, but safe
        // division should have
        assert_eq!(division_ex.0.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert_eq!(
            safe_division_ex.0.load(std::sync::atomic::Ordering::Relaxed),
            2
        );

        // session 3: restore to original inputs
        drop(tracked_engine);

        {
            let mut input_session = engine.input_session().await;
            input_session.set_input(Variable(1), 2).await;
            input_session.commit().await;
        }

        let tracked_engine = engine.tracked().await;

        assert_eq!(
            tracked_engine
                .query(&SafeDivision::new(Variable(0), Variable(1)))
                .await,
            Some(21)
        );

        // this time divisor executor reused the cached value but safe division
        // had to run again
        assert_eq!(division_ex.0.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert_eq!(
            safe_division_ex.0.load(std::sync::atomic::Ordering::Relaxed),
            3
        );
    }
}

#[tokio::test]
async fn add_two_absolutes_sign_change() {
    let tempdir = tempdir().unwrap();

    let absolute_ex = Arc::new(AbsoluteExecutor::default());
    let add_two_absolutes_ex = Arc::new(AddTwoAbsolutesExecutor::default());

    // session 1: initial inputs and query
    {
        let mut engine = create_test_engine(&tempdir).await;

        engine.register_executor(absolute_ex.clone());
        engine.register_executor(add_two_absolutes_ex.clone());

        let engine = Arc::new(engine);

        {
            let mut input_session = engine.input_session().await;

            input_session.set_input(Variable(0), 200).await;
            input_session.set_input(Variable(1), 150).await;
        }

        let tracked_engine = engine.clone().tracked().await;

        // Initial query: abs(200) + abs(150) = 350
        assert_eq!(
            tracked_engine
                .query(&AddTwoAbsolutes::new(Variable(0), Variable(1)))
                .await,
            350
        );

        // Both absolute executors should run once, and add_two_absolutes once
        assert_eq!(absolute_ex.0.load(std::sync::atomic::Ordering::Relaxed), 2);
        assert_eq!(
            add_two_absolutes_ex.0.load(std::sync::atomic::Ordering::Relaxed),
            1
        );

        // session 2: change sign of inputs
        drop(tracked_engine);

        // Change Variable(0) from 200 to -200
        {
            let mut input_session = engine.input_session().await;
            input_session.set_input(Variable(0), -200).await;
        }

        let tracked_engine = engine.clone().tracked().await;

        // Result should still be 350: abs(-200) + abs(150) = 200 + 150 = 350
        assert_eq!(
            tracked_engine
                .query(&AddTwoAbsolutes::new(Variable(0), Variable(1)))
                .await,
            350
        );

        // Only the Absolute query for Variable(0) should re-execute (3 total)
        // The result is the same (200), so AddTwoAbsolutes should NOT
        // re-execute
        assert_eq!(absolute_ex.0.load(std::sync::atomic::Ordering::Relaxed), 3);
        assert_eq!(
            add_two_absolutes_ex.0.load(std::sync::atomic::Ordering::Relaxed),
            1
        );

        // session 3: change sign of other input
        drop(tracked_engine);

        // Change Variable(1) from 150 to -150
        {
            let mut input_session = engine.input_session().await;
            input_session.set_input(Variable(1), -150).await;
        }

        let tracked_engine = engine.tracked().await;

        // Result should still be 350: abs(-200) + abs(-150) = 200 + 150 = 350
        assert_eq!(
            tracked_engine
                .query(&AddTwoAbsolutes::new(Variable(0), Variable(1)))
                .await,
            350
        );

        // Only the Absolute query for Variable(1) should re-execute (4 total)
        // The result is still the same (150), so AddTwoAbsolutes should still
        // NOT re-execute
        assert_eq!(absolute_ex.0.load(std::sync::atomic::Ordering::Relaxed), 4);
        assert_eq!(
            add_two_absolutes_ex.0.load(std::sync::atomic::Ordering::Relaxed),
            1
        );
    }
}
