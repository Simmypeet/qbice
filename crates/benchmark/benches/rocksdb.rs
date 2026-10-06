//! Benchmarks for the RocksDB-backed storage engine.
//!
//! The workload is a layered graph: a row of inputs at the bottom and `LAYERS`
//! layers of derived queries on top. Every derived query reads `FAN_IN`
//! neighbouring queries from the layer below, so changing one input
//! invalidates a narrow cone of queries above it and leaves the rest of the
//! graph untouched.
//!
//! Every layer holds 10,000 queries by default. Set `QBICE_BENCH_WIDTH` to
//! benchmark a larger graph; the size is part of each benchmark's name, so
//! results for different sizes are never compared with each other.

#![allow(missing_docs)]

use std::{
    hint::black_box,
    ops::Range,
    path::Path,
    sync::{Arc, LazyLock},
    time::Duration,
};

use criterion::{BatchSize, BenchmarkId, Criterion, SamplingMode};
use qbice::{
    Decode, Encode, Engine, Identifiable, Query, StableHash, TrackedEngine,
    config::Config, executor::Executor,
};
use qbice_benchmark::{RocksDbConfig, create_rocksdb_engine};
use tempfile::TempDir;

#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Number of layers of derived queries stacked on top of the inputs.
const LAYERS: u32 = 10;

/// Number of queries from the layer below that a derived query reads.
const FAN_IN: u64 = 3;

/// Size in bytes of a derived query's value.
const PAYLOAD_LEN: usize = 128;

/// One update changes every `CHANGED_EVERY`-th input, which is far enough
/// apart that the cones of queries they invalidate do not overlap.
const CHANGED_EVERY: usize = 100;

/// The storage engine's default in-memory cache capacity. It holds the whole
/// graph at the default size.
const DEFAULT_CACHE_CAPACITY: u64 = 1 << 18;

/// An in-memory cache capacity that holds only a few percent of the graph at
/// the default size, so that most reads have to be served by RocksDB.
const SMALL_CACHE_CAPACITY: u64 = 1 << 12;

/// Number of queries in every layer.
fn width() -> u64 {
    static WIDTH: LazyLock<u64> = LazyLock::new(|| {
        std::env::var("QBICE_BENCH_WIDTH").map_or(10_000, |width| {
            width.parse().expect("QBICE_BENCH_WIDTH must be an integer")
        })
    });

    *WIDTH
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
pub struct Input(pub u64);

impl Query for Input {
    type Value = u64;
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
pub struct Derived {
    pub layer: u32,
    pub index: u64,
}

impl Query for Derived {
    type Value = Arc<[u8]>;
}

/// Expands a seed into `PAYLOAD_LEN` pseudo-random bytes (splitmix64).
fn payload(mut state: u64) -> Arc<[u8]> {
    let mut bytes = Vec::with_capacity(PAYLOAD_LEN + 8);

    while bytes.len() < PAYLOAD_LEN {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);

        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);

        bytes.extend_from_slice(&(z ^ (z >> 31)).to_le_bytes());
    }

    bytes.truncate(PAYLOAD_LEN);
    bytes.into()
}

/// Folds a payload back into the seed used by the layer above.
fn seed_of(payload: &[u8]) -> u64 {
    u64::from_le_bytes(payload[..8].try_into().unwrap())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DerivedExecutor;

impl<C: Config> Executor<Derived, C> for DerivedExecutor {
    async fn execute(
        &self,
        query: &Derived,
        engine: &TrackedEngine<C>,
    ) -> Arc<[u8]> {
        let mut seed = (u64::from(query.layer) << 48) ^ query.index;

        for offset in 0..FAN_IN {
            let index = (query.index + offset) % width();

            let read = if query.layer == 0 {
                engine.query(&Input(index)).await
            } else {
                let below = Derived { layer: query.layer - 1, index };

                seed_of(&engine.query(&below).await)
            };

            seed = (seed ^ read).wrapping_mul(0x0000_0100_0000_01B3);
        }

        payload(seed)
    }
}

type BenchEngine = Engine<RocksDbConfig>;

async fn open(path: &Path, cache_capacity: u64) -> Arc<BenchEngine> {
    let mut engine = create_rocksdb_engine(path, cache_capacity).await;

    engine.register_executor::<Derived, _>(Arc::new(DerivedExecutor));

    Arc::new(engine)
}

/// Drops the engine, which drains the write-behind queue and closes RocksDB.
fn close(engine: Arc<BenchEngine>) {
    drop(Arc::try_unwrap(engine).expect("the engine is still shared"));
}

/// Sets the given inputs. A different `revision` gives every input a
/// different value.
async fn set_inputs(
    engine: &Arc<BenchEngine>,
    inputs: impl IntoIterator<Item = u64>,
    revision: u64,
) {
    let mut session = engine.input_session().await;

    for index in inputs {
        let value = index.wrapping_mul(31).wrapping_add(revision);

        session.set_input(Input(index), value).await;
    }

    session.commit().await;
}

fn changed_inputs() -> impl Iterator<Item = u64> {
    (0..width()).step_by(CHANGED_EVERY)
}

/// Queries every query in the given layers from one task per core.
async fn query_layers(engine: &Arc<BenchEngine>, layers: Range<u32>) {
    let tracked = engine.clone().tracked().await;
    let tasks = std::thread::available_parallelism().map_or(4, usize::from);

    let handles = (0..tasks as u64)
        .map(|task| {
            let tracked = tracked.clone();
            let layers = layers.clone();

            tokio::spawn(async move {
                for layer in layers {
                    for index in (task..width()).step_by(tasks) {
                        black_box(
                            tracked.query(&Derived { layer, index }).await,
                        );
                    }
                }
            })
        })
        .collect::<Vec<_>>();

    for handle in handles {
        handle.await.unwrap();
    }
}

/// Computes the whole graph into the database at `path` and persists it.
async fn build(path: &Path) {
    let engine = open(path, DEFAULT_CACHE_CAPACITY).await;

    set_inputs(&engine, 0..width(), 0).await;
    query_layers(&engine, LAYERS - 1..LAYERS).await;

    close(engine);
}

fn bench_rocksdb(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("rocksdb");

    // the number of derived queries in the graph
    let size = width() * u64::from(LAYERS);

    group
        .sample_size(10)
        .sampling_mode(SamplingMode::Flat)
        .measurement_time(Duration::from_secs(20));

    // Computes every query into an empty database, then waits for the
    // write-behind queue to drain and RocksDB to close.
    group.bench_function(BenchmarkId::new("build", size), |b| {
        b.iter_batched(
            || TempDir::new().unwrap(),
            |dir| {
                runtime.block_on(build(dir.path()));

                // returned so that deleting the directory is not measured
                dir
            },
            BatchSize::PerIteration,
        );
    });

    // Opens an existing database with empty in-memory caches and reads every
    // query back. Nothing is recomputed, so this is RocksDB's read path.
    group.bench_function(BenchmarkId::new("reopen_read_all", size), |b| {
        let dir = TempDir::new().unwrap();
        runtime.block_on(build(dir.path()));

        b.iter(|| {
            runtime.block_on(async {
                let engine = open(dir.path(), DEFAULT_CACHE_CAPACITY).await;

                query_layers(&engine, 0..LAYERS).await;

                close(engine);
            });
        });
    });

    // Opens an existing database, changes a few inputs and brings the top
    // layer up to date: dirty propagation, verification and recomputation,
    // all starting from empty in-memory caches.
    group.bench_function(BenchmarkId::new("reopen_update", size), |b| {
        let dir = TempDir::new().unwrap();
        runtime.block_on(build(dir.path()));

        let mut revision = 0;

        b.iter(|| {
            revision += 1;

            runtime.block_on(async {
                let engine = open(dir.path(), DEFAULT_CACHE_CAPACITY).await;

                set_inputs(&engine, changed_inputs(), revision).await;
                query_layers(&engine, LAYERS - 1..LAYERS).await;

                close(engine);
            });
        });
    });

    // The same update against a long-lived engine whose in-memory caches are
    // far smaller than the graph, so RocksDB serves reads continuously.
    group.bench_function(BenchmarkId::new("update_small_cache", size), |b| {
        let dir = TempDir::new().unwrap();
        runtime.block_on(build(dir.path()));

        let engine = runtime.block_on(open(dir.path(), SMALL_CACHE_CAPACITY));
        let mut revision = 0;

        b.iter(|| {
            revision += 1;

            runtime.block_on(async {
                set_inputs(&engine, changed_inputs(), revision).await;
                query_layers(&engine, LAYERS - 1..LAYERS).await;
            });
        });

        runtime.block_on(async { close(engine) });
    });

    group.finish();
}

criterion::criterion_group!(benches, bench_rocksdb);
criterion::criterion_main!(benches);
