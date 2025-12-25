//! Tests for parallel query execution.

#![allow(missing_docs)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_const_for_fn)]

use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use qbice::{
    Decode, Encode, TrackedEngine,
    config::Config,
    executor::{CyclicError, Executor},
    query::Query,
};
use qbice_integration_test::{Variable, create_test_engine};
use qbice_stable_hash::StableHash;
use qbice_stable_type_id::Identifiable;
use tempfile::tempdir;

// ============================================================================
// Collect Variables Query Types
// ============================================================================

#[derive(
    Debug,
    Clone,
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
pub struct CollectVariables {
    pub vars: Arc<[Variable]>,
}

impl Query for CollectVariables {
    type Value = HashMap<Variable, i64>;
}

#[derive(Debug, Default)]
pub struct CollectVariablesExecutor(pub AtomicUsize);

impl<C: Config> Executor<CollectVariables, C> for CollectVariablesExecutor {
    async fn execute(
        &self,
        query: &CollectVariables,
        engine: &TrackedEngine<C>,
    ) -> Result<HashMap<Variable, i64>, CyclicError> {
        // track usage
        self.0.fetch_add(1, Ordering::Relaxed);

        let mut result = HashMap::new();

        for &var in query.vars.iter() {
            // simulate some async work
            tokio::time::sleep(Duration::from_millis(16)).await;

            let value = engine.query(&var).await?;

            result.insert(var, value);
        }

        Ok(result)
    }
}

// ============================================================================
// Read Variable Map Query Types
// ============================================================================

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
pub struct ReadVariableMap(pub Variable);

impl Query for ReadVariableMap {
    type Value = i64;
}

impl ReadVariableMap {
    pub fn new(var: Variable) -> Self { Self(var) }
}

#[derive(Debug, Default)]
pub struct ReadVariableMapExecutor(pub AtomicUsize);

impl<C: Config> Executor<ReadVariableMap, C> for ReadVariableMapExecutor {
    async fn execute(
        &self,
        query: &ReadVariableMap,
        engine: &TrackedEngine<C>,
    ) -> Result<i64, CyclicError> {
        // track usage
        self.0.fetch_add(1, Ordering::Relaxed);

        let var_map = engine
            .query(&CollectVariables {
                vars: Arc::new([Variable(0), Variable(1), Variable(2)]),
            })
            .await?;

        Ok(*var_map.get(&query.0).unwrap())
    }
}

#[tokio::test(flavor = "multi_thread")]
#[allow(clippy::cast_possible_wrap)]
async fn parallel_read_variable_map() {
    let tempdir = tempdir().unwrap();
    let mut engine = create_test_engine(&tempdir);

    let collect_ex = Arc::new(CollectVariablesExecutor::default());
    let read_map_ex = Arc::new(ReadVariableMapExecutor::default());

    engine.register_executor(collect_ex.clone());
    engine.register_executor(read_map_ex.clone());

    let mut input_session = engine.input_session();

    input_session.set_input(Variable(0), 10);
    input_session.set_input(Variable(1), 20);
    input_session.set_input(Variable(2), 30);

    drop(input_session);

    let engine = Arc::new(engine);
    let tracked_engine = engine.tracked();

    let handles: Vec<_> = (0..3)
        .map(|i| {
            let tracked_engine = tracked_engine.clone();
            tokio::spawn(async move {
                tracked_engine.query(&ReadVariableMap::new(Variable(i))).await
            })
        })
        .collect();

    for (i, handle) in handles.into_iter().enumerate() {
        let result = handle.await.unwrap().unwrap();
        assert_eq!(result, (i as i64 + 1) * 10);
    }

    // CollectVariablesExecutor should have been called only once
    assert_eq!(collect_ex.0.load(Ordering::Relaxed), 1);
    // ReadVariableMapExecutor should have been called three times
    assert_eq!(read_map_ex.0.load(Ordering::Relaxed), 3);
}
