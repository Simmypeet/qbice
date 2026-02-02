#![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
#![allow(missing_docs)]

use std::{self, ops::Range, sync::Arc};

use qbice::{
    Decode, Encode, Identifiable, Query, StableHash, TrackedEngine,
    config::Config,
};
use qbice_benchmark::create_test_engine;

#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    StableHash,
    Identifiable,
    Encode,
    Decode,
)]
pub struct VariableRange;

impl Query for VariableRange {
    type Value = Range<u64>;
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
pub struct Variable(pub u64);

impl Query for Variable {
    type Value = i64;
}

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    StableHash,
    Identifiable,
    Encode,
    Decode,
)]
pub struct Mean;

impl Mean {
    async fn algo<C: Config>(&self, tracked_engine: &TrackedEngine<C>) -> i64 {
        let range = tracked_engine.query(&VariableRange).await;
        let var_count = (range.end - range.start) as usize;

        let mut sum = 0;

        for i in range.clone() {
            let var = Variable(i);
            let value = tracked_engine.query(&var).await;
            sum += value;
        }

        sum / (var_count as i64)
    }
}

impl Query for Mean {
    type Value = i64;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MeanNormalExecutor;

impl<C: Config> qbice::executor::Executor<Mean, C> for MeanNormalExecutor {
    async fn execute(&self, query: &Mean, engine: &TrackedEngine<C>) -> i64 {
        query.algo(engine).await
    }

    fn execution_style() -> qbice::ExecutionStyle {
        qbice::ExecutionStyle::Normal
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MeanFirewallExecutor;

impl<C: Config> qbice::executor::Executor<Mean, C> for MeanFirewallExecutor {
    async fn execute(&self, query: &Mean, engine: &TrackedEngine<C>) -> i64 {
        query.algo(engine).await
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
    Hash,
    StableHash,
    Identifiable,
    Encode,
    Decode,
)]
pub struct Diff(Variable);

impl Diff {
    async fn algo<C: Config>(&self, tracked_engine: &TrackedEngine<C>) -> i64 {
        let value = tracked_engine.query(&self.0).await;
        let mean = tracked_engine.query(&Mean).await;

        value - mean
    }
}

impl Query for Diff {
    type Value = i64;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DiffExecutor;

impl<C: Config> qbice::executor::Executor<Diff, C> for DiffExecutor {
    async fn execute(&self, query: &Diff, engine: &TrackedEngine<C>) -> i64 {
        query.algo(engine).await
    }
}

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    StableHash,
    Identifiable,
    Encode,
    Decode,
)]
pub struct DiffSquared(Variable);

impl Query for DiffSquared {
    type Value = i64;
}

impl DiffSquared {
    async fn algo<C: Config>(&self, tracked_engine: &TrackedEngine<C>) -> i64 {
        let diff = tracked_engine.query(&Diff(self.0)).await;

        diff * diff
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DiffSquaredExecutor;

impl<C: Config> qbice::executor::Executor<DiffSquared, C>
    for DiffSquaredExecutor
{
    async fn execute(
        &self,
        query: &DiffSquared,
        engine: &TrackedEngine<C>,
    ) -> i64 {
        query.algo(engine).await
    }
}

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    StableHash,
    Identifiable,
    Encode,
    Decode,
)]
pub struct DiffSquaredChain1(Variable);

impl Query for DiffSquaredChain1 {
    type Value = i64;
}

impl DiffSquaredChain1 {
    async fn algo<C: Config>(&self, tracked_engine: &TrackedEngine<C>) -> i64 {
        tracked_engine.query(&DiffSquared(self.0)).await
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DiffSquaredChain1Executor;

impl<C: Config> qbice::executor::Executor<DiffSquaredChain1, C>
    for DiffSquaredChain1Executor
{
    async fn execute(
        &self,
        query: &DiffSquaredChain1,
        engine: &TrackedEngine<C>,
    ) -> i64 {
        query.algo(engine).await
    }
}

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    StableHash,
    Identifiable,
    Encode,
    Decode,
)]
pub struct DiffSquaredChain2(Variable);

impl Query for DiffSquaredChain2 {
    type Value = i64;
}

impl DiffSquaredChain2 {
    async fn algo<C: Config>(&self, tracked_engine: &TrackedEngine<C>) -> i64 {
        tracked_engine.query(&DiffSquaredChain1(self.0)).await
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DiffSquaredChain2Executor;

impl<C: Config> qbice::executor::Executor<DiffSquaredChain2, C>
    for DiffSquaredChain2Executor
{
    async fn execute(
        &self,
        query: &DiffSquaredChain2,
        engine: &TrackedEngine<C>,
    ) -> i64 {
        query.algo(engine).await
    }
}

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    StableHash,
    Identifiable,
    Encode,
    Decode,
)]
pub struct Variance;

impl Query for Variance {
    type Value = i64;
}

impl Variance {
    async fn algo<C: Config>(&self, tracked_engine: &TrackedEngine<C>) -> i64 {
        let range = tracked_engine.query(&VariableRange).await;

        let mut sum_squared_diff = 0i64;
        let mut count = 0i64;

        for i in range {
            let var = Variable(i);
            let diff_squared =
                tracked_engine.query(&DiffSquaredChain2(var)).await;

            sum_squared_diff += diff_squared;
            count += 1;
        }

        sum_squared_diff / count
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VarianceExecutor;

impl<C: Config> qbice::executor::Executor<Variance, C> for VarianceExecutor {
    async fn execute(
        &self,
        query: &Variance,
        engine: &TrackedEngine<C>,
    ) -> i64 {
        query.algo(engine).await
    }
}

async fn run(firewall: bool) {
    let mut engine = create_test_engine().await;

    // register executors
    if firewall {
        engine.register_executor::<Mean, _>(Arc::new(MeanFirewallExecutor));
    } else {
        engine.register_executor::<Mean, _>(Arc::new(MeanNormalExecutor));
    }

    engine.register_executor::<Diff, _>(Arc::new(DiffExecutor));
    engine.register_executor::<DiffSquared, _>(Arc::new(DiffSquaredExecutor));
    engine.register_executor::<DiffSquaredChain1, _>(Arc::new(
        DiffSquaredChain1Executor,
    ));
    engine.register_executor::<DiffSquaredChain2, _>(Arc::new(
        DiffSquaredChain2Executor,
    ));
    engine.register_executor::<Variance, _>(Arc::new(VarianceExecutor));

    // first session: run as normal
    let engine = Arc::new(engine);

    let mut input_session = engine.input_session().await;
    let var_count = 5_000u64;
    input_session.set_input(VariableRange, 0..var_count).await;
    for i in 0..var_count {
        input_session.set_input(Variable(i), (i + 1) as i64).await;
    }
    drop(input_session);

    let tracked_engine = engine.clone().tracked().await;

    tracked_engine.query(&Variance).await;

    drop(tracked_engine);

    // second session: switch 2 variables value (mean should remain the same)
    // avoid dirtying the `Diff` and `DiffSquared` nodes
    {
        let mut input_session = engine.input_session().await;

        input_session.set_input(Variable(0), var_count as i64).await;
        input_session.set_input(Variable(var_count - 1), 1).await;
        drop(input_session);
    }

    let tracked_engine = engine.clone().tracked().await;

    tracked_engine.query(&Variance).await;

    drop(tracked_engine);

    // final session: change one variable to dirty the mean
    {
        let mut input_session = engine.input_session().await;

        input_session.set_input(Variable(var_count / 2), 1).await;
        drop(input_session);
    }

    let tracked_engine = engine.clone().tracked().await;

    tracked_engine.query(&Variance).await;
}

fn run_with_tokio(firewall: bool) {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    rt.block_on(run(firewall));
}

fn bench_compare_firewall(c: &mut criterion::Criterion) {
    let mut group = c.benchmark_group("firewall_vs_normal");

    group.bench_function("firewall_execution", |b| {
        b.iter(|| run_with_tokio(true));
    });
    group.bench_function("normal_execution", |b| {
        b.iter(|| run_with_tokio(false));
    });

    group.finish();
}

criterion::criterion_group!(benches, bench_compare_firewall);
criterion::criterion_main!(benches);
