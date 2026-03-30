#![allow(missing_docs)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_const_for_fn)]

use core::panic;
use std::{collections::HashMap, sync::Arc};

use qbice::{
    Config, Decode, Encode, Query, StableHash, TrackedEngine, executor,
};
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
    Encode,
    Decode,
    StableHash,
    Query,
)]
#[value(Arc<HashMap<char, i32>>)]
pub struct InputMap;

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Encode,
    Decode,
    StableHash,
    Query,
)]
#[value(Option<i32>)]
pub struct DoubleFirewall(char);

#[executor(style = qbice::ExecutionStyle::Firewall)]
pub async fn double_firewall_executor<C: Config>(
    &DoubleFirewall(key): &DoubleFirewall,
    engine: &TrackedEngine<C>,
) -> Option<i32> {
    let map = engine.query(&InputMap).await;
    map.get(&key).copied()
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
    Encode,
    Decode,
    StableHash,
    Query,
)]
#[value(Option<i32>)]
pub struct DoubleProjection(char);

#[executor(style = qbice::ExecutionStyle::Projection)]
pub async fn double_projection_executor<C: Config>(
    &DoubleProjection(key): &DoubleProjection,
    engine: &TrackedEngine<C>,
) -> Option<i32> {
    let value = engine.query(&DoubleFirewall(key)).await?;
    Some(value * 2)
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
    Encode,
    Decode,
    StableHash,
    Query,
)]
#[value(i32)]
pub struct Double(char);

#[executor]
pub async fn double_executor<C: Config>(
    &Double(key): &Double,
    engine: &TrackedEngine<C>,
) -> i32 {
    engine
        .query(&DoubleProjection(key))
        .await
        .unwrap_or_else(|| panic!("{key}"))
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
    Encode,
    Decode,
    StableHash,
    Query,
)]
#[value(i32)]
pub struct SumDouble;

#[executor]
pub async fn sum_double_executor<C: Config>(
    _: &SumDouble,
    engine: &TrackedEngine<C>,
) -> i32 {
    let inputs = engine.query(&InputMap).await;
    let mut sum = 0;

    for var in inputs.keys() {
        sum += engine.query(&Double(*var)).await;
    }

    sum
}

#[tokio::test]
async fn dynamic_firewall_and_projection() {
    let tempdir = tempdir().unwrap();
    let mut engine = create_test_engine(&tempdir).await;

    engine.register_executor(Arc::new(DoubleProjectionExecutor));
    engine.register_executor(Arc::new(DoubleFirewallExecutor));
    engine.register_executor(Arc::new(DoubleExecutor));
    engine.register_executor(Arc::new(SumDoubleExecutor));

    let engine = Arc::new(engine);

    // first session
    {
        let mut input_session = engine.input_session().await;
        input_session
            .set_input(
                InputMap,
                Arc::new(HashMap::from([('a', 1), ('b', 2), ('c', 3)])),
            )
            .await;
        input_session.commit().await;
    }

    // query sum double
    {
        let tracked = engine.clone().tracked().await;
        let result = tracked.query(&SumDouble).await;
        assert_eq!(result, 12);
    }

    // next session, remove 'b'
    {
        let mut input_session = engine.input_session().await;
        input_session
            .set_input(InputMap, Arc::new(HashMap::from([('a', 1), ('c', 3)])))
            .await;
        input_session.commit().await;
    }

    // query sum double again, should be 8 now
    {
        let tracked = engine.clone().tracked().await;
        let result = tracked.query(&SumDouble).await;
        assert_eq!(result, 8);
    }

    // next bring back but change its value
    {
        let mut input_session = engine.input_session().await;
        input_session
            .set_input(
                InputMap,
                Arc::new(HashMap::from([('a', 1), ('b', 20), ('c', 3)])),
            )
            .await;
        input_session.commit().await;
    }

    // query sum double again, should be 48 now
    {
        let tracked = engine.tracked().await;
        let result = tracked.query(&SumDouble).await;
        assert_eq!(result, 48);
    }
}
