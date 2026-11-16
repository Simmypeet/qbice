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
//!
//! # When the build is killed
//!
//! The build takes a fraction of a second, and from the moment RocksDB has
//! written a few files to the moment the process is gone it is a few dozen
//! milliseconds. A kill that is meant to come in between comes too early or
//! too late whenever the machine is busy with something else. Too early, and
//! nothing has reached the database; too late, and all of it has.
//!
//! The test therefore does not depend on a single kill coming at the right
//! moment:
//!
//! - It first kills a child that keeps the engine open once the build is done
//!   instead of shutting it down ([`AfterBuild::Waits`]). What that engine has
//!   not written is never written, so no kill comes too late. A kill that came
//!   too early is repeated with the build left running for longer, until one of
//!   them leaves a part of the build in the database. Without one that does,
//!   the test has not looked at what it is about, and fails.
//! - It then kills at a few other points, among them while the first files are
//!   written and while the engine shuts down. Whatever these leave behind, be
//!   it nothing, a part or everything, has to pass the two checks.
//!
//! Set `QBICE_KILLED_BUILD_SOAK` to a number to kill that many more builds,
//! each at a different point. The points are the same on every run.

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

/// Set in the child process if it is to shut the engine down once the build
/// is done. See [`AfterBuild`].
const CHILD_SHUTS_DOWN: &str = "QBICE_KILLED_BUILD_SHUTS_DOWN";

/// The number of additional builds to kill, each at a different point.
const SOAK: &str = "QBICE_KILLED_BUILD_SOAK";

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

/// The number of table files RocksDB has started to write when the build is
/// killed in order to leave a part of it in the database. It is more than the
/// first flush writes, and as many as the build writes before it shuts the
/// engine down.
const KILL_AFTER_TABLE_FILES: usize = 6;

/// More table files than the build writes before it shuts the engine down. A
/// kill that waits for this many comes while the engine shuts down, or not at
/// all.
const SHUTDOWN_TABLE_FILES: usize = 8;

/// The most table files that a kill waits for when it is picked at random.
const MOST_TABLE_FILES: usize = 10;

/// The longest that a build is left running, after the table files it waited
/// for, when the kill is picked at random. The whole build takes a few times
/// as long.
const LONGEST_RANDOM_LINGER: Duration = Duration::from_millis(40);

/// How often a kill that left nothing in the database is repeated with the
/// build left running for longer.
const PARTIAL_ATTEMPTS: u32 = 8;

/// How long the table files that a kill waits for may take to appear before
/// the test gives up. A child that keeps the engine open would otherwise be
/// waited for forever.
const TABLE_FILES_TIMEOUT: Duration = Duration::from_secs(60);

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

/// What the child process does until it is killed: a cold build, and after
/// it what [`AfterBuild`] says. Does nothing when the tests are run the usual
/// way.
#[test]
#[ignore = "the tests of this file run this in a child process"]
fn build_until_killed() {
    let Some(directory) = std::env::var_os(CHILD_DIRECTORY) else {
        return;
    };

    let shuts_down = std::env::var_os(CHILD_SHUTS_DOWN).is_some();

    runtime().block_on(async {
        let engine = open(Path::new(&directory), Arc::default()).await;

        set_inputs(&engine, 0).await;
        query_layers(&engine, LAYERS - 1..LAYERS, Arc::new(expected_seeds(0)))
            .await;

        if shuts_down {
            close(engine);
        } else {
            // The engine stays open, with whatever it has not written yet
            // still in memory, until the process is killed.
            loop {
                std::thread::park();
            }
        }
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

/// What the child process does once it has built the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AfterBuild {
    /// Shuts the engine down and exits, the way an application does. A kill
    /// that comes late finds the engine shutting down, or the process gone
    /// and all of the build in the database.
    ShutsDown,

    /// Keeps the engine open and waits to be killed. What the engine has not
    /// written by then is never written, so the kill leaves the build
    /// incomplete however late it comes.
    Waits,
}

/// When a build is killed.
#[derive(Debug, Clone, Copy)]
struct KillPoint {
    /// The number of table files that RocksDB has to have started to write.
    table_files: usize,

    /// How long the build is left running after that.
    linger: Duration,

    after_build: AfterBuild,
}

/// Builds the graph into `path` in a child process and kills that process at
/// the given point. Returns whether it was killed; `false` means that the
/// child has shut the engine down and exited first.
///
/// # Panics
///
/// Panics if the child process ends in any other way than by being killed or
/// by exiting after a clean shutdown, and if the table files that the kill
/// waits for do not appear.
fn build_and_kill(path: &Path, kill_point: KillPoint) -> bool {
    let mut command = Command::new(std::env::current_exe().unwrap());

    command
        .args(["--exact", "build_until_killed", "--ignored", "--nocapture"])
        .env(CHILD_DIRECTORY, path)
        .stdout(Stdio::null());

    match kill_point.after_build {
        AfterBuild::ShutsDown => command.env(CHILD_SHUTS_DOWN, "1"),
        AfterBuild::Waits => command.env_remove(CHILD_SHUTS_DOWN),
    };

    let mut child = command.spawn().unwrap();
    let started = Instant::now();

    // Whether the child has ended on its own, which only a child that shuts
    // the engine down does, and only successfully.
    let has_exited = |child: &mut std::process::Child| {
        let Some(status) = child.try_wait().unwrap() else {
            return false;
        };

        assert!(
            kill_point.after_build == AfterBuild::ShutsDown && status.success(),
            "the build ended on its own with {status} before it was killed at \
             {kill_point:?}"
        );

        true
    };

    let killed = loop {
        if has_exited(&mut child) {
            break false;
        }

        if table_files(path) >= kill_point.table_files {
            std::thread::sleep(kill_point.linger);

            break !has_exited(&mut child);
        }

        if started.elapsed() > TABLE_FILES_TIMEOUT {
            let _ = child.kill();

            panic!(
                "the build has written {} table files after {:?}, which is \
                 fewer than {kill_point:?} waits for",
                table_files(path),
                started.elapsed(),
            );
        }

        std::thread::sleep(Duration::from_millis(1));
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

/// What a kill has left behind.
#[derive(Debug, Clone, Copy)]
struct Outcome {
    #[allow(dead_code)] // read when the outcomes are printed
    kill_point: KillPoint,

    /// Whether the child was killed, rather than gone already.
    #[allow(dead_code)] // read when the outcomes are printed
    killed: bool,

    /// The number of queries that the database still knew.
    survived: u64,
}

impl Outcome {
    /// Whether the database held a part of the build: more than nothing, and
    /// not all of it.
    const fn is_partial(&self) -> bool {
        self.survived > 0 && self.survived < QUERIES
    }
}

/// Kills a build at the given point and checks that the database it leaves
/// behind is consistent, with unchanged inputs and with changed ones.
fn kill_and_check(
    runtime: &tokio::runtime::Runtime,
    kill_point: KillPoint,
) -> Outcome {
    let killed_directory = tempdir().unwrap();
    let copy = tempdir().unwrap();

    let killed = build_and_kill(killed_directory.path(), kill_point);

    copy_directory(killed_directory.path(), copy.path());

    let unchanged = runtime.block_on(check(killed_directory.path(), 0));
    let changed = runtime.block_on(check(copy.path(), 1));

    assert_eq!(
        unchanged.wrong, 0,
        "queries read back with a wrong value after the kill at \
         {kill_point:?} (killed = {killed})"
    );
    assert_eq!(
        changed.wrong, 0,
        "queries kept their old value after the kill at {kill_point:?} \
         (killed = {killed}) and a change of the inputs"
    );

    // a query that survived the kill is not executed again
    let outcome = Outcome {
        kill_point,
        killed,
        survived: QUERIES.saturating_sub(unchanged.executed),
    };

    println!(
        "{kill_point:?}: killed = {killed}, {} of {QUERIES} queries survived",
        outcome.survived
    );

    outcome
}

/// One step of splitmix64.
const fn next_random(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);

    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);

    z ^ (z >> 31)
}

/// The kill points of a soak: as many as [`SOAK`] says, the same ones on
/// every run.
fn soak_kill_points() -> impl Iterator<Item = KillPoint> {
    let count = std::env::var(SOAK).map_or(0, |count| {
        count.parse::<u64>().unwrap_or_else(|_| {
            panic!("{SOAK} must be a number of builds to kill")
        })
    });

    let mut state = 0;

    (0..count).map(move |index| {
        // A child that keeps the engine open never writes more table files
        // than the build itself does, so the kill must not wait for more.
        let (after_build, most_table_files) = if index % 2 == 0 {
            (AfterBuild::ShutsDown, MOST_TABLE_FILES)
        } else {
            (AfterBuild::Waits, KILL_AFTER_TABLE_FILES)
        };

        let table_files = 1 + usize::try_from(
            next_random(&mut state) % most_table_files as u64,
        )
        .unwrap();

        let linger = Duration::from_micros(
            next_random(&mut state)
                % u64::try_from(LONGEST_RANDOM_LINGER.as_micros()).unwrap(),
        );

        KillPoint { table_files, linger, after_build }
    })
}

#[test]
fn killed_build_leaves_a_consistent_database() {
    let runtime = runtime();

    // A kill that leaves a part of the build in the database. The child
    // keeps the engine open, so the kill cannot come too late, and a kill
    // that came before anything had reached the database is repeated with
    // the build left running for longer.
    let mut outcomes = Vec::new();

    for attempt in 0..PARTIAL_ATTEMPTS {
        let linger = if attempt == 0 {
            Duration::ZERO
        } else {
            Duration::from_millis(25 << attempt)
        };

        outcomes.push(kill_and_check(&runtime, KillPoint {
            table_files: KILL_AFTER_TABLE_FILES,
            linger,
            after_build: AfterBuild::Waits,
        }));

        if outcomes.last().is_some_and(Outcome::is_partial) {
            break;
        }
    }

    assert!(
        outcomes.last().is_some_and(Outcome::is_partial),
        "no kill has left a part of the build in the database, so what such a \
         database looks like has not been checked: {outcomes:#?}"
    );

    // Kills at other points. Whatever they leave behind has to be
    // consistent.
    let other_kill_points = [
        // while the first table files are written
        KillPoint {
            table_files: 1,
            linger: Duration::ZERO,
            after_build: AfterBuild::ShutsDown,
        },
        // while the engine shuts down
        KillPoint {
            table_files: SHUTDOWN_TABLE_FILES,
            linger: Duration::ZERO,
            after_build: AfterBuild::ShutsDown,
        },
    ];

    for kill_point in other_kill_points.into_iter().chain(soak_kill_points()) {
        kill_and_check(&runtime, kill_point);
    }
}
