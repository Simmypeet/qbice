//! Compares a clean build on the in-memory storage engine with the same build
//! on the RocksDB-backed one.
//!
//! Without a database every map of the engine is a concurrent hash map. With
//! one, every read goes through a bounded cache that falls back to the
//! database, and every write is staged for a write-behind queue that persists
//! it in the background. A clean build is where the second mode pays the most
//! for that: no query has been computed before, so every read of the database
//! comes back empty and every query is written. An application that has
//! finished computing also has to wait for the queue to drain before it can
//! exit, so the time to shut down is part of what the build costs.
//!
//! This benchmark computes the same graph from scratch on both storage
//! engines and prints the phases next to each other.
//!
//! The workload is the layered graph of the `shutdown` benchmark: a row of
//! inputs at the bottom and layers of derived queries on top, where every
//! derived query reads a few queries from the layer below. The rows are cut
//! into blocks and a query only reads from its own block, so the tasks that
//! compute different blocks never wait for one another. Unlike in the other
//! benchmarks, an executor spends some time on a computation of its own, the
//! way the executors of an application do: the engine then shares the cores
//! with the work it is tracking, and what the storage engine costs is
//! measured against that work instead of against nothing.
//!
//! The defaults follow a compiler front end checking a large program, which
//! is the workload this benchmark was written to reproduce: three
//! dependencies per query, about 1.5KB written to the database per query
//! (the value and what the engine records about the query), and an executor
//! that is cheap next to the engine's own bookkeeping. That program has about
//! two million queries; `QBICE_BUILD_WIDTH=180000` builds a graph of that
//! size, which takes a few gigabytes of memory. The default is a quarter of
//! it. The shape is set through the environment:
//!
//! - `QBICE_BUILD_WIDTH`: queries per layer (default 50,000)
//! - `QBICE_BUILD_LAYERS`: layers of derived queries (default 10)
//! - `QBICE_BUILD_FAN_IN`: queries read by a derived query (default 3)
//! - `QBICE_BUILD_BLOCK`: queries of a layer in one block (default 100)
//! - `QBICE_BUILD_PAYLOAD`: bytes in a derived query's value (default 512)
//! - `QBICE_BUILD_WORK`: rounds of mixing an executor does on top of reading
//!   its dependencies, a few nanoseconds each (default 4,000)
//! - `QBICE_BUILD_CACHE`: entries each in-memory cache of the RocksDB-backed
//!   engine holds (default 262,144, the storage engine's own default)
//! - `QBICE_BUILD_THREADS`: threads that compute queries (default: one per
//!   core)
//! - `QBICE_BUILD_RUNS`: number of times each build is repeated (default 3)
//! - `QBICE_BUILD_BACKENDS`: comma separated list of the storage engines to
//!   run, out of `in-memory` and `rocksdb` (default: both)
//!
//! This is not a criterion benchmark: one run takes seconds and the
//! interesting numbers are the split between the phases and the ratio between
//! the storage engines, so the phases are timed directly and printed as a
//! table.

#![allow(missing_docs)]

use std::{
    hint::black_box,
    sync::{
        Arc, LazyLock,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use qbice::{
    Decode, Encode, Engine, Identifiable, Query, StableHash, TrackedEngine,
    config::Config, executor::Executor,
};
use qbice_benchmark::{create_rocksdb_engine, create_test_engine};
use tempfile::TempDir;

#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn env_or(name: &str, default: u64) -> u64 {
    std::env::var(name).map_or(default, |value| {
        value.parse().unwrap_or_else(|_| panic!("{name} must be an integer"))
    })
}

/// The shape of the graph that is built.
#[derive(Debug, Clone, Copy)]
struct Shape {
    width: u64,
    layers: u32,
    fan_in: u64,
    block: u64,
    payload_len: usize,
    work: u64,
    cache_capacity: u64,
}

fn shape() -> Shape {
    static SHAPE: LazyLock<Shape> = LazyLock::new(|| Shape {
        width: env_or("QBICE_BUILD_WIDTH", 50_000),
        layers: u32::try_from(env_or("QBICE_BUILD_LAYERS", 10)).unwrap(),
        fan_in: env_or("QBICE_BUILD_FAN_IN", 3),
        block: env_or("QBICE_BUILD_BLOCK", 100).max(1),
        payload_len: usize::try_from(env_or("QBICE_BUILD_PAYLOAD", 512))
            .unwrap(),
        work: env_or("QBICE_BUILD_WORK", 4_000),
        cache_capacity: env_or("QBICE_BUILD_CACHE", 1 << 18),
    });

    *SHAPE
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

/// One round of splitmix64.
const fn mix(state: u64) -> (u64, u64) {
    let state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);

    let mut z = state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);

    (state, z ^ (z >> 31))
}

/// Stands in for the computation an executor does with what it has read.
fn work(mut seed: u64) -> u64 {
    for _ in 0..shape().work {
        seed = mix(seed).1;
    }

    seed
}

/// Expands a seed into pseudo-random bytes.
fn payload(mut state: u64) -> Arc<[u8]> {
    let len = shape().payload_len.max(8);
    let mut bytes = Vec::with_capacity(len + 8);

    while bytes.len() < len {
        let (next, output) = mix(state);

        state = next;
        bytes.extend_from_slice(&output.to_le_bytes());
    }

    bytes.truncate(len);
    bytes.into()
}

/// Folds a payload back into the seed used by the layer above.
fn seed_of(payload: &[u8]) -> u64 {
    u64::from_le_bytes(payload[..8].try_into().unwrap())
}

/// Number of times a derived query has been executed.
static EXECUTIONS: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DerivedExecutor;

impl<C: Config> Executor<Derived, C> for DerivedExecutor {
    async fn execute(
        &self,
        query: &Derived,
        engine: &TrackedEngine<C>,
    ) -> Arc<[u8]> {
        EXECUTIONS.fetch_add(1, Ordering::Relaxed);

        let shape = shape();
        let mut seed = (u64::from(query.layer) << 48) ^ query.index;

        // the block this query belongs to, which is the only one it reads
        let start = query.index - query.index % shape.block;
        let len = shape.block.min(shape.width - start);

        for offset in 0..shape.fan_in {
            let index = start + (query.index - start + offset) % len;

            let read = if query.layer == 0 {
                engine.query(&Input(index)).await
            } else {
                let below = Derived { layer: query.layer - 1, index };

                seed_of(&engine.query(&below).await)
            };

            seed = (seed ^ read).wrapping_mul(0x0000_0100_0000_01B3);
        }

        payload(work(seed))
    }
}

/// Number of threads that compute queries.
fn threads() -> usize {
    static THREADS: LazyLock<usize> = LazyLock::new(|| {
        let cores = std::thread::available_parallelism().map_or(4, usize::from);

        usize::try_from(env_or("QBICE_BUILD_THREADS", cores as u64))
            .unwrap()
            .max(1)
    });

    *THREADS
}

/// Sets every input and queries the top layer, which computes every query
/// below it. Every thread runs a task that takes the next block nobody has
/// started.
async fn compute<C: Config>(engine: &Arc<Engine<C>>) {
    let shape = shape();

    {
        let mut session = engine.input_session().await;

        for index in 0..shape.width {
            session.set_input(Input(index), index.wrapping_mul(31)).await;
        }

        session.commit().await;
    }

    let tracked = engine.clone().tracked().await;
    let next_block = Arc::new(AtomicU64::new(0));
    let layer = shape.layers - 1;

    let handles = (0..threads())
        .map(|_| {
            let tracked = tracked.clone();
            let next_block = next_block.clone();

            tokio::spawn(async move {
                loop {
                    let start =
                        next_block.fetch_add(shape.block, Ordering::Relaxed);

                    if start >= shape.width {
                        break;
                    }

                    let end = (start + shape.block).min(shape.width);

                    for index in start..end {
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

/// How long each phase of one clean build took.
#[derive(Debug, Clone, Copy)]
struct Timing {
    /// Until every query has been answered, which is when an application
    /// would be done with its own work.
    compute: Duration,

    /// Dropping the engine. With a database this drains the write-behind
    /// queue and closes the database.
    shutdown: Duration,
}

impl Timing {
    fn total(self) -> Duration { self.compute + self.shutdown }
}

/// Builds the graph on a newly opened engine and shuts the engine down.
async fn build<C: Config>(mut engine: Engine<C>) -> Timing {
    engine.register_executor::<Derived, _>(Arc::new(DerivedExecutor));

    let engine = Arc::new(engine);
    let executed = EXECUTIONS.load(Ordering::Relaxed);

    let start = Instant::now();
    compute(&engine).await;
    let compute = start.elapsed();

    assert_eq!(
        EXECUTIONS.load(Ordering::Relaxed) - executed,
        shape().width * u64::from(shape().layers),
        "every derived query has to be executed exactly once"
    );

    let start = Instant::now();
    drop(Arc::try_unwrap(engine).expect("the engine is still shared"));
    let shutdown = start.elapsed();

    Timing { compute, shutdown }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Backend {
    InMemory,
    RocksDb,
}

impl Backend {
    const fn name(self) -> &'static str {
        match self {
            Self::InMemory => "in-memory",
            Self::RocksDb => "rocksdb",
        }
    }

    fn run(self, runtime: &tokio::runtime::Runtime) -> Timing {
        match self {
            Self::InMemory => runtime
                .block_on(async { build(create_test_engine().await).await }),

            Self::RocksDb => {
                // dropped, and with that deleted, after the timing is taken
                let dir = TempDir::new().unwrap();

                runtime.block_on(async {
                    let engine = create_rocksdb_engine(
                        dir.path(),
                        shape().cache_capacity,
                    )
                    .await;

                    build(engine).await
                })
            }
        }
    }
}

fn backends() -> Vec<Backend> {
    let Ok(list) = std::env::var("QBICE_BUILD_BACKENDS") else {
        return vec![Backend::InMemory, Backend::RocksDb];
    };

    list.split(',')
        .map(|name| match name.trim() {
            "in-memory" => Backend::InMemory,
            "rocksdb" => Backend::RocksDb,
            other => {
                panic!("QBICE_BUILD_BACKENDS: unknown storage engine `{other}`")
            }
        })
        .collect()
}

fn median(mut durations: Vec<Duration>) -> Duration {
    durations.sort_unstable();
    durations[durations.len() / 2]
}

#[allow(clippy::cast_precision_loss)]
fn main() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(threads())
        .enable_all()
        .build()
        .unwrap();

    let shape = shape();
    let runs = env_or("QBICE_BUILD_RUNS", 3).max(1);

    println!(
        "clean_build: {} threads computing {} queries ({} inputs + {} layers \
         of {}), fan-in {}, blocks of {}, {} byte values, {} rounds of work \
         per query, caches of {} entries",
        threads(),
        shape.width * (u64::from(shape.layers) + 1),
        shape.width,
        shape.layers,
        shape.width,
        shape.fan_in,
        shape.block,
        shape.payload_len,
        shape.work,
        shape.cache_capacity,
    );
    println!(
        "{:>10}  {:>6}  {:>10}  {:>10}  {:>10}",
        "backend", "run", "compute", "shutdown", "total"
    );

    let mut medians = Vec::new();

    for backend in backends() {
        let timings =
            (0..runs).map(|_| backend.run(&runtime)).collect::<Vec<_>>();

        for (index, timing) in timings.iter().enumerate() {
            println!(
                "{:>10}  {:>6}  {:>9.2}s  {:>9.2}s  {:>9.2}s",
                backend.name(),
                index + 1,
                timing.compute.as_secs_f64(),
                timing.shutdown.as_secs_f64(),
                timing.total().as_secs_f64(),
            );
        }

        let median = Timing {
            compute: median(timings.iter().map(|x| x.compute).collect()),
            shutdown: median(timings.iter().map(|x| x.shutdown).collect()),
        };

        println!(
            "{:>10}  {:>6}  {:>9.2}s  {:>9.2}s  {:>9.2}s",
            backend.name(),
            "median",
            median.compute.as_secs_f64(),
            median.shutdown.as_secs_f64(),
            median.total().as_secs_f64(),
        );

        medians.push((backend, median));
    }

    let in_memory = medians.iter().find(|x| x.0 == Backend::InMemory);
    let rocksdb = medians.iter().find(|x| x.0 == Backend::RocksDb);

    if let (Some((_, in_memory)), Some((_, rocksdb))) = (in_memory, rocksdb) {
        let ratio =
            |a: Duration, b: Duration| a.as_secs_f64() / b.as_secs_f64();

        println!(
            "rocksdb / in-memory (medians): compute {:.1}x, shutdown {:.1}x, \
             total {:.1}x",
            ratio(rocksdb.compute, in_memory.compute),
            ratio(rocksdb.shutdown, in_memory.shutdown),
            ratio(rocksdb.total(), in_memory.total()),
        );
    }
}
