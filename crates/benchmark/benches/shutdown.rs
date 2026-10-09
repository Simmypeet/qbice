//! Measures how long a RocksDB-backed engine takes to shut down after a cold
//! build.
//!
//! Computing a query only stages its writes; a write-behind queue persists
//! them in the background. An application that has finished computing still
//! has to wait for that queue to drain and for RocksDB to close before it can
//! exit, and this benchmark reports that wait next to the time the computation
//! itself took.
//!
//! The workload is a layered graph computed from scratch into an empty
//! database: a row of inputs at the bottom and layers of derived queries on
//! top, where every derived query reads a few queries from the layer below.
//! Unlike the graph of the `rocksdb` benchmark, the rows are cut into blocks
//! and a query only reads from its own block, the way the queries of one
//! source file mostly read each other. Blocks are computed by different tasks
//! that never wait for one another, so the computation uses every core and
//! can outrun the write-behind queue.
//!
//! The shape of the graph is set through the environment:
//!
//! - `QBICE_SHUTDOWN_WIDTH`: queries per layer (default 100,000)
//! - `QBICE_SHUTDOWN_LAYERS`: layers of derived queries (default 10)
//! - `QBICE_SHUTDOWN_FAN_IN`: queries read by a derived query (default 3)
//! - `QBICE_SHUTDOWN_BLOCK`: queries of a layer in one block (default 100)
//! - `QBICE_SHUTDOWN_PAYLOAD`: bytes in a derived query's value (default 128)
//! - `QBICE_SHUTDOWN_CACHE`: entries each in-memory cache holds (default
//!   4,194,304, which holds the whole default graph, so the computation never
//!   has to read a value back from RocksDB)
//! - `QBICE_SHUTDOWN_THREADS`: threads that compute queries (default: one per
//!   core)
//! - `QBICE_SHUTDOWN_RUNS`: number of times the build is repeated (default 3)
//!
//! After the engine has shut down, the database is opened again and every
//! query is read back. This must not execute a single query: it checks that
//! everything the queue held made it to disk, however the writes were
//! scheduled.
//!
//! This is not a criterion benchmark: one run takes many seconds and the
//! interesting number is the split between the phases, so the phases are timed
//! directly and printed as a table.

#![allow(missing_docs)]

use std::{
    hint::black_box,
    ops::Range,
    path::Path,
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
use qbice_benchmark::{RocksDbConfig, create_rocksdb_engine};
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
    cache_capacity: u64,
}

fn shape() -> Shape {
    static SHAPE: LazyLock<Shape> = LazyLock::new(|| Shape {
        width: env_or("QBICE_SHUTDOWN_WIDTH", 100_000),
        layers: u32::try_from(env_or("QBICE_SHUTDOWN_LAYERS", 10)).unwrap(),
        fan_in: env_or("QBICE_SHUTDOWN_FAN_IN", 3),
        block: env_or("QBICE_SHUTDOWN_BLOCK", 100).max(1),
        payload_len: usize::try_from(env_or("QBICE_SHUTDOWN_PAYLOAD", 128))
            .unwrap(),
        cache_capacity: env_or("QBICE_SHUTDOWN_CACHE", 1 << 22),
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

/// Expands a seed into pseudo-random bytes (splitmix64).
fn payload(mut state: u64) -> Arc<[u8]> {
    let len = shape().payload_len.max(8);
    let mut bytes = Vec::with_capacity(len + 8);

    while bytes.len() < len {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);

        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);

        bytes.extend_from_slice(&(z ^ (z >> 31)).to_le_bytes());
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

        payload(seed)
    }
}

type BenchEngine = Engine<RocksDbConfig>;

async fn open(path: &Path) -> Arc<BenchEngine> {
    let mut engine = create_rocksdb_engine(path, shape().cache_capacity).await;

    engine.register_executor::<Derived, _>(Arc::new(DerivedExecutor));

    Arc::new(engine)
}

/// Number of threads that compute queries.
fn threads() -> usize {
    static THREADS: LazyLock<usize> = LazyLock::new(|| {
        let cores = std::thread::available_parallelism().map_or(4, usize::from);

        usize::try_from(env_or("QBICE_SHUTDOWN_THREADS", cores as u64))
            .unwrap()
            .max(1)
    });

    *THREADS
}

/// Queries every query in the given layers. Every thread runs a task that
/// takes the next block nobody has started.
async fn query_layers(engine: &Arc<BenchEngine>, layers: Range<u32>) {
    let shape = shape();

    let tracked = engine.clone().tracked().await;
    let next_block = Arc::new(AtomicU64::new(0));

    let handles = (0..threads())
        .map(|_| {
            let tracked = tracked.clone();
            let next_block = next_block.clone();
            let layers = layers.clone();

            tokio::spawn(async move {
                loop {
                    let start =
                        next_block.fetch_add(shape.block, Ordering::Relaxed);

                    if start >= shape.width {
                        break;
                    }

                    let end = (start + shape.block).min(shape.width);

                    for layer in layers.clone() {
                        for index in start..end {
                            black_box(
                                tracked.query(&Derived { layer, index }).await,
                            );
                        }
                    }
                }
            })
        })
        .collect::<Vec<_>>();

    for handle in handles {
        handle.await.unwrap();
    }
}

/// Sets every input and queries the top layer, which computes every query
/// below it.
async fn compute(engine: &Arc<BenchEngine>) {
    {
        let mut session = engine.input_session().await;

        for index in 0..shape().width {
            session.set_input(Input(index), index.wrapping_mul(31)).await;
        }

        session.commit().await;
    }

    query_layers(engine, shape().layers - 1..shape().layers).await;
}

/// Drops the engine, which drains the write-behind queue and closes RocksDB.
fn close(engine: Arc<BenchEngine>) {
    drop(Arc::try_unwrap(engine).expect("the engine is still shared"));
}

/// How long each phase of one cold build took.
#[derive(Debug, Clone, Copy)]
struct Timing {
    /// Until every query has been answered, which is when an application
    /// would be done with its own work.
    compute: Duration,

    /// Dropping the engine: draining the write-behind queue and closing
    /// RocksDB.
    shutdown: Duration,

    /// Opening the database again and reading every query back.
    reopen: Duration,
}

async fn run(path: &Path) -> Timing {
    let engine = open(path).await;

    let start = Instant::now();
    compute(&engine).await;
    let compute = start.elapsed();

    let start = Instant::now();
    close(engine);
    let shutdown = start.elapsed();

    let executed = EXECUTIONS.load(Ordering::Relaxed);

    let start = Instant::now();
    let engine = open(path).await;
    query_layers(&engine, 0..shape().layers).await;
    close(engine);
    let reopen = start.elapsed();

    assert_eq!(
        EXECUTIONS.load(Ordering::Relaxed),
        executed,
        "queries were executed again after reopening: the shutdown lost writes"
    );

    Timing { compute, shutdown, reopen }
}

fn directory_size(path: &Path) -> u64 {
    std::fs::read_dir(path).map_or(0, |entries| {
        entries
            .filter_map(Result::ok)
            .map(|entry| {
                let path = entry.path();

                if path.is_dir() {
                    directory_size(&path)
                } else {
                    entry.metadata().map_or(0, |metadata| metadata.len())
                }
            })
            .sum()
    })
}

#[allow(clippy::cast_precision_loss)]
fn main() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(threads())
        .enable_all()
        .build()
        .unwrap();

    let shape = shape();
    let runs = env_or("QBICE_SHUTDOWN_RUNS", 3);

    println!(
        "shutdown: {} threads computing {} queries ({} inputs + {} layers of \
         {}), fan-in {}, blocks of {}, {} byte values, caches of {} entries",
        threads(),
        shape.width * (u64::from(shape.layers) + 1),
        shape.width,
        shape.layers,
        shape.width,
        shape.fan_in,
        shape.block,
        shape.payload_len,
        shape.cache_capacity,
    );
    println!(
        "{:>4}  {:>10}  {:>10}  {:>8}  {:>10}  {:>10}",
        "run", "compute", "shutdown", "ratio", "reopen", "on disk"
    );

    for index in 0..runs {
        let dir = TempDir::new().unwrap();
        let timing = runtime.block_on(run(dir.path()));

        println!(
            "{:>4}  {:>9.2}s  {:>9.2}s  {:>7.1}x  {:>9.2}s  {:>8.1}MB",
            index + 1,
            timing.compute.as_secs_f64(),
            timing.shutdown.as_secs_f64(),
            timing.shutdown.as_secs_f64() / timing.compute.as_secs_f64(),
            timing.reopen.as_secs_f64(),
            directory_size(dir.path()) as f64 / (1024.0 * 1024.0),
        );
    }
}
