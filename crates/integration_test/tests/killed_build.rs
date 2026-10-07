//! Tests that an engine that is killed in the middle of a cold build leaves a
//! database behind that the next run can trust.
//!
//! The build runs in a child process, which is this test binary started
//! again. The test kills it (`SIGKILL` on Unix) once RocksDB has started to
//! write files, so the database holds whatever had been flushed by then:
//! some of the queries, without the ones whose writes were still in memory.
//!
//! That database is then checked twice, on two copies of it:
//!
//! - with **unchanged inputs**: every query must read back with the value it
//!   has to have. The queries that the database still knows are not executed
//!   again, which tells how much of the build survived the kill.
//! - with **changed inputs**: every input gets a new value first. A query that
//!   was written without the edges that lead to it would not learn that its
//!   inputs have changed and would keep its old value.

use std::{
    ops::Range,
    path::Path,
    process::{Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use qbice::{
    Decode, Encode, Engine, Identifiable, Query, StableHash, TrackedEngine,
    config::Config,
    executor::Executor,
    serialize::Plugin,
    stable_hash::{SeededStableHasherBuilder, Sip128Hasher},
    storage::{
        kv_database::rocksdb::RocksDB,
        storage_engine::db_backed::{Configuration, DbBackedFactory},
    },
};
use qbice_integration_test::TestingConfig;
use tempfile::tempdir;

/// Set in the child process, to the directory of the database it builds.
const CHILD_DIRECTORY: &str = "QBICE_KILLED_BUILD_DIRECTORY";

/// Number of queries in every layer.
const WIDTH: u64 = 1_200;

/// Number of layers of derived queries stacked on top of the inputs.
const LAYERS: u32 = 10;

/// Number of queries from the layer below that a derived query reads.
const FAN_IN: u64 = 3;

/// Number of queries of a layer in one block. A query only reads queries of
/// its own block, so the blocks are computed in parallel.
const BLOCK: u64 = 50;

/// Size in bytes of a derived query's value. The values are large so that a
/// build of few queries writes enough for RocksDB to flush several times.
const PAYLOAD_LEN: usize = 16 * 1024;

/// Holds the whole graph.
const CACHE_CAPACITY: u64 = 1 << 16;

/// Number of derived queries in the graph.
const QUERIES: u64 = WIDTH * LAYERS as u64;

/// The build is killed once RocksDB has written this many table files, which
/// is more than its first flush writes.
const KILL_AFTER_TABLE_FILES: usize = 6;

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
struct Input(u64);

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
struct Derived {
    layer: u32,
    index: u64,
}

impl Query for Derived {
    type Value = Arc<[u8]>;
}

/// The value of an input. A different `revision` gives every input a
/// different value.
const fn input_value(index: u64, revision: u64) -> u64 {
    index.wrapping_mul(31).wrapping_add(revision)
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

/// The indices of the queries in the layer below that a query reads.
fn reads(index: u64) -> impl Iterator<Item = u64> {
    let start = index - index % BLOCK;
    let len = BLOCK.min(WIDTH - start);

    (0..FAN_IN).map(move |offset| start + (index - start + offset) % len)
}

/// The seed of a derived query's value, given what the queries it reads
/// return, in the order of [`reads`].
fn seed(query: Derived, read: impl IntoIterator<Item = u64>) -> u64 {
    read.into_iter()
        .fold((u64::from(query.layer) << 48) ^ query.index, |seed, read| {
            (seed ^ read).wrapping_mul(0x0000_0100_0000_01B3)
        })
}

/// The seeds of the values that every derived query must have when the inputs
/// are at `revision`, layer by layer. Computed without the engine.
fn expected_seeds(revision: u64) -> Vec<Vec<u64>> {
    let mut layers = Vec::<Vec<u64>>::new();

    for layer in 0..LAYERS {
        let seeds = (0..WIDTH)
            .map(|index| {
                seed(
                    Derived { layer, index },
                    reads(index).map(|read| {
                        layers.last().map_or_else(
                            || input_value(read, revision),
                            |below| {
                                seed_of(&payload(
                                    below[usize::try_from(read).unwrap()],
                                ))
                            },
                        )
                    }),
                )
            })
            .collect();

        layers.push(seeds);
    }

    layers
}

/// Counts how many times it has executed a query.
#[derive(Debug, Default)]
struct DerivedExecutor(AtomicU64);

impl<C: Config> Executor<Derived, C> for DerivedExecutor {
    async fn execute(
        &self,
        query: &Derived,
        engine: &TrackedEngine<C>,
    ) -> Arc<[u8]> {
        self.0.fetch_add(1, Ordering::Relaxed);

        let mut read = Vec::new();

        for index in reads(query.index) {
            read.push(if query.layer == 0 {
                engine.query(&Input(index)).await
            } else {
                let below = Derived { layer: query.layer - 1, index };

                seed_of(&engine.query(&below).await)
            });
        }

        payload(seed(*query, read))
    }
}

type TestEngine = Engine<TestingConfig>;

async fn open(path: &Path, executor: Arc<DerivedExecutor>) -> Arc<TestEngine> {
    let mut engine = Engine::<TestingConfig>::new_with(
        Plugin::default(),
        DbBackedFactory::builder()
            .configuration(
                Configuration::builder().cache_capacity(CACHE_CAPACITY).build(),
            )
            .db_factory(RocksDB::factory(path))
            .build(),
        SeededStableHasherBuilder::<Sip128Hasher>::new(0),
    )
    .await
    .unwrap();

    engine.register_executor::<Derived, _>(executor);

    Arc::new(engine)
}

/// Drops the engine, which drains the write-behind queue and closes RocksDB.
fn close(engine: Arc<TestEngine>) {
    drop(Arc::try_unwrap(engine).expect("the engine is still shared"));
}

async fn set_inputs(engine: &Arc<TestEngine>, revision: u64) {
    let mut session = engine.input_session().await;

    for index in 0..WIDTH {
        session.set_input(Input(index), input_value(index, revision)).await;
    }

    session.commit().await;
}

/// Queries every query in the given layers from one task per core and
/// returns how many of them do not have the value that `expected` says they
/// must have.
async fn query_layers(
    engine: &Arc<TestEngine>,
    layers: Range<u32>,
    expected: Arc<Vec<Vec<u64>>>,
) -> u64 {
    let tracked = engine.clone().tracked().await;
    let tasks = std::thread::available_parallelism().map_or(4, usize::from);
    let next_block = Arc::new(AtomicU64::new(0));

    let handles = (0..tasks)
        .map(|_| {
            let tracked = tracked.clone();
            let next_block = next_block.clone();
            let layers = layers.clone();
            let expected = expected.clone();

            tokio::spawn(async move {
                let mut wrong = 0;

                loop {
                    let start = next_block.fetch_add(BLOCK, Ordering::Relaxed);

                    if start >= WIDTH {
                        break wrong;
                    }

                    for layer in layers.clone() {
                        for index in start..(start + BLOCK).min(WIDTH) {
                            let value =
                                tracked.query(&Derived { layer, index }).await;

                            let seed = expected[layer as usize]
                                [usize::try_from(index).unwrap()];

                            wrong += u64::from(value != payload(seed));
                        }
                    }
                }
            })
        })
        .collect::<Vec<_>>();

    let mut wrong = 0;

    for handle in handles {
        wrong += handle.await.unwrap();
    }

    wrong
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread().enable_all().build().unwrap()
}

/// What the child process does until it is killed: a cold build and a clean
/// shutdown. Does nothing when the tests are run the usual way.
#[test]
#[ignore = "the tests of this file run this in a child process"]
fn build_until_killed() {
    let Some(directory) = std::env::var_os(CHILD_DIRECTORY) else {
        return;
    };

    runtime().block_on(async {
        let engine = open(Path::new(&directory), Arc::default()).await;

        set_inputs(&engine, 0).await;
        query_layers(&engine, LAYERS - 1..LAYERS, Arc::new(expected_seeds(0)))
            .await;

        close(engine);
    });
}

fn table_files(path: &Path) -> usize {
    std::fs::read_dir(path).map_or(0, |entries| {
        entries
            .filter_map(Result::ok)
            .filter(|entry| {
                entry.path().extension().is_some_and(|x| x == "sst")
            })
            .count()
    })
}

fn copy_directory(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();

    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());

        if entry.file_type().unwrap().is_dir() {
            copy_directory(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

/// Builds the graph into `path` in a child process and kills that process
/// while it is still building. Returns whether it was killed; `false` means
/// that the build has finished first.
fn build_and_kill(path: &Path, linger: Duration) -> bool {
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "build_until_killed", "--ignored", "--nocapture"])
        .env(CHILD_DIRECTORY, path)
        .stdout(Stdio::null())
        .spawn()
        .unwrap();

    let killed = loop {
        if child.try_wait().unwrap().is_some() {
            break false;
        }

        if table_files(path) >= KILL_AFTER_TABLE_FILES {
            std::thread::sleep(linger);

            break child.try_wait().unwrap().is_none();
        }

        std::thread::sleep(Duration::from_millis(2));
    };

    // the child gets no chance to clean up
    let _ = child.kill();
    child.wait().unwrap();

    killed
}

/// What a check of a database has found.
#[derive(Debug, Clone, Copy)]
struct Found {
    /// Queries that read back with a wrong value.
    wrong: u64,

    /// Queries that had to be executed.
    executed: u64,
}

/// Opens the database at `path`, sets the inputs to `revision` and reads
/// every query back.
async fn check(path: &Path, revision: u64) -> Found {
    let executor = Arc::new(DerivedExecutor::default());
    let engine = open(path, executor.clone()).await;

    set_inputs(&engine, revision).await;

    let wrong =
        query_layers(&engine, 0..LAYERS, Arc::new(expected_seeds(revision)))
            .await;

    close(engine);

    Found { wrong, executed: executor.0.load(Ordering::Relaxed) }
}

#[test]
fn killed_build_leaves_a_consistent_database() {
    let runtime = runtime();
    let start = Instant::now();

    // Nothing can go wrong with a database that holds none of the queries or
    // all of them, so a kill that came too early or too late is repeated. How
    // long the build is left running after the first files appear differs
    // from one attempt to the next.
    for attempt in 0..8 {
        let killed = tempdir().unwrap();
        let copy = tempdir().unwrap();

        let was_killed =
            build_and_kill(killed.path(), Duration::from_millis(40 * attempt));

        copy_directory(killed.path(), copy.path());

        let unchanged = runtime.block_on(check(killed.path(), 0));
        let changed = runtime.block_on(check(copy.path(), 1));

        assert_eq!(
            unchanged.wrong, 0,
            "queries read back with a wrong value after the kill"
        );
        assert_eq!(
            changed.wrong, 0,
            "queries kept their old value after the kill and a change of the \
             inputs"
        );

        // a query that survived the kill is not executed again
        let survived = QUERIES.saturating_sub(unchanged.executed);

        println!(
            "attempt {attempt}: killed = {was_killed}, {survived} of \
             {QUERIES} queries survived ({:.1}s)",
            start.elapsed().as_secs_f64()
        );

        if was_killed && survived > 0 && survived < QUERIES {
            return;
        }
    }

    panic!("the build was never killed with only a part of it written");
}
