//! Minimal qbice reproduction: a stale firewall reached through a dependency
//! path that the user query did not have in the previous revision.

use std::sync::Arc;

use qbice::{Config, Decode, Encode, Executor, Query, StableHash};
use qbice_integration_test::create_test_engine;
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
    Encode,
    Decode,
    Query,
)]
#[value(i32)]
struct Value;

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
    Encode,
    Decode,
    Query,
)]
#[value(bool)]
struct Use;

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
    Encode,
    Decode,
    Query,
)]
#[value(i32)]
struct Fw;

struct FwExecutor;

impl<C: Config> Executor<Fw, C> for FwExecutor {
    async fn execute(&self, _: &Fw, engine: &qbice::TrackedEngine<C>) -> i32 {
        engine.query(&Value).await
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
    Encode,
    Decode,
    Query,
)]
#[value(i32)]
struct Reader;

struct ReaderExecutor;

impl<C: Config> Executor<Reader, C> for ReaderExecutor {
    async fn execute(
        &self,
        _: &Reader,
        engine: &qbice::TrackedEngine<C>,
    ) -> i32 {
        engine.query(&Fw).await
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
    Encode,
    Decode,
    Query,
)]
#[value(Option<i32>)]
struct Top;

struct TopExecutor;

impl<C: Config> Executor<Top, C> for TopExecutor {
    async fn execute(
        &self,
        _: &Top,
        engine: &qbice::TrackedEngine<C>,
    ) -> Option<i32> {
        if engine.query(&Use).await {
            Some(engine.query(&Reader).await)
        } else {
            None
        }
    }
}

#[tokio::test]
async fn firewall_with_dynamic_edge() {
    let tempdir = tempdir().unwrap();
    let mut engine = create_test_engine(&tempdir).await;

    engine.register_executor(Arc::new(FwExecutor));
    engine.register_executor(Arc::new(ReaderExecutor));
    engine.register_executor(Arc::new(TopExecutor));
    let engine = Arc::new(engine);

    // Use = true, Value = 1, Expected Top = Some(1)
    {
        let mut session = engine.input_session().await;
        session.set_input(Use, true).await;
        session.set_input(Value, 1).await;
        session.commit().await;

        let engine = engine.clone().tracked().await;
        assert_eq!(engine.query(&Top).await, Some(1));
    }

    // Use = false, Value = 1, Expected Top = None
    {
        let mut session = engine.input_session().await;
        session.set_input(Use, false).await;
        session.set_input(Value, 1).await;
        session.commit().await;

        let engine = engine.clone().tracked().await;
        assert_eq!(engine.query(&Top).await, None);
    }

    // Use = true, Value = 2, Expected Top = Some(2)
    {
        let mut session = engine.input_session().await;
        session.set_input(Use, true).await;
        session.set_input(Value, 2).await;
        session.commit().await;

        let engine = engine.clone().tracked().await;
        assert_eq!(engine.query(&Top).await, Some(2));
    }
}
