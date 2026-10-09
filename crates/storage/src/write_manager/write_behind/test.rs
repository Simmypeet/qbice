use std::{
    any::Any,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc,
    },
    time::Duration,
};

use parking_lot::{Condvar, Mutex};
use qbice_stable_type_id::Identifiable;

use super::{Epoch, WriteBehind};
use crate::kv_database::{
    DiscriminantEncoding, KeyOfSetColumn, KvDatabase, SerializationBuffer,
    WideColumn, WideColumnValue, WriteBatch,
};

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Identifiable,
)]
#[stable_type_id_crate(qbice_stable_type_id)]
struct Column;

impl WideColumn for Column {
    type Discriminant = ();

    type Key = u64;

    fn discriminant_encoding() -> DiscriminantEncoding {
        DiscriminantEncoding::Prefixed
    }
}

impl WideColumnValue<Column> for u64 {
    fn discriminant() {}
}

/// A database that only remembers the keys of [`Column`] it has been given,
/// and that takes its time to commit them.
#[derive(Clone, Default)]
struct SlowDb {
    keys: Arc<Mutex<Vec<u64>>>,
    gate: Arc<Gate>,

    /// Whether the database refuses what it is asked to commit.
    broken: Arc<AtomicBool>,
}

/// Lets the commits of a [`SlowDb`] through, unless it is closed.
#[derive(Default)]
struct Gate {
    closed: Mutex<bool>,
    opened: Condvar,
}

impl SlowDb {
    /// Makes the database commit nothing until it is released.
    fn hold(&self) { *self.gate.closed.lock() = true; }

    fn release(&self) {
        *self.gate.closed.lock() = false;
        self.gate.opened.notify_all();
    }
}

/// The keys of the values that were put. Every one of them counts as one
/// byte.
struct Keys(Vec<u64>);

impl SerializationBuffer for Keys {
    fn put<W: WideColumn, C: WideColumnValue<W>>(
        &mut self,
        key: &W::Key,
        _: &C,
    ) {
        let key = (key as &dyn Any)
            .downcast_ref::<u64>()
            .expect("only `Column` is written");

        self.0.push(*key);
    }

    fn delete<W: WideColumn, C: WideColumnValue<W>>(&mut self, _: &W::Key) {
        unreachable!()
    }

    fn insert_member<C: KeyOfSetColumn>(&mut self, _: &C::Key, _: &C::Element) {
        unreachable!()
    }

    fn delete_member<C: KeyOfSetColumn>(&mut self, _: &C::Key, _: &C::Element) {
        unreachable!()
    }

    fn size(&self) -> usize { self.0.len() }
}

struct SlowBatch {
    db: SlowDb,
    keys: Vec<u64>,
}

impl WriteBatch for SlowBatch {
    type SerializationBuffer = Keys;

    fn put<W: WideColumn, C: WideColumnValue<W>>(&mut self, _: &W::Key, _: &C) {
        unreachable!()
    }

    fn delete<W: WideColumn, C: WideColumnValue<W>>(&mut self, _: &W::Key) {
        unreachable!()
    }

    fn insert_member<C: KeyOfSetColumn>(&mut self, _: &C::Key, _: &C::Element) {
        unreachable!()
    }

    fn delete_member<C: KeyOfSetColumn>(&mut self, _: &C::Key, _: &C::Element) {
        unreachable!()
    }

    fn consume_serialization_buffer(&mut self, mut buffer: Keys) {
        self.keys.append(&mut buffer.0);
    }

    fn commit(self) {
        {
            let mut closed = self.db.gate.closed.lock();

            while *closed {
                self.db.gate.opened.wait(&mut closed);
            }
        }

        assert!(
            !self.db.broken.load(Ordering::SeqCst),
            "the database refuses the write"
        );

        std::thread::sleep(Duration::from_millis(2));

        self.db.keys.lock().extend(self.keys);
    }
}

impl KvDatabase for SlowDb {
    type WriteBatch = SlowBatch;

    type SerializationBuffer = Keys;

    type ScanMemberIterator<C: KeyOfSetColumn> = std::iter::Empty<C::Element>;

    fn get_wide_column<W: WideColumn, C: WideColumnValue<W>>(
        &self,
        _: &W::Key,
    ) -> Option<C> {
        None
    }

    fn scan_members<C: KeyOfSetColumn>(
        &self,
        _: &C::Key,
    ) -> Self::ScanMemberIterator<C> {
        std::iter::empty()
    }

    fn write_batch(&self) -> SlowBatch {
        SlowBatch { db: self.clone(), keys: Vec::new() }
    }

    fn serialization_buffer(&self) -> Keys { Keys(Vec::new()) }
}

/// A cache evicts an entry as soon as the epoch that wrote it is published,
/// and the next read of the entry goes to the database. An epoch must
/// therefore not be published before its write batch and every write batch
/// before it have been committed, in whatever order they were submitted.
#[test]
fn epoch_is_published_once_it_is_in_the_database() {
    const BATCHES: u64 = 64;

    let db = SlowDb::default();
    let writer = WriteBehind::new(&db, 16);
    let committed = writer.committed.clone();

    let batches =
        (0..BATCHES).map(|_| writer.new_write_batch()).collect::<Vec<_>>();

    // the newest first, so that nothing can be committed before the last
    // submission
    for mut batch in batches.into_iter().rev() {
        let epoch = batch.epoch();

        batch.put_wide_column::<Column, u64>(&epoch.0, Some(&epoch.0));
        writer.submit_write_batch(batch);

        if epoch != Epoch(0) {
            assert!(!committed.contains(epoch));
        }
    }

    for epoch in 0..BATCHES {
        while !committed.contains(Epoch(epoch)) {
            std::thread::yield_now();
        }

        // every key up to this one, in the order of the epochs
        let keys = db.keys.lock();

        assert!(
            keys.iter().copied().take_while(|key| *key <= epoch).eq(0..=epoch),
            "epoch {epoch} is published, but the database holds {keys:?}"
        );
    }

    drop(writer);

    assert!(db.keys.lock().iter().copied().eq(0..BATCHES));
}

/// Writes that are submitted faster than the database takes them would
/// otherwise pile up in memory for as long as that lasts. Whoever submits
/// has to wait instead, and gets to go on once the database has caught up.
#[test]
fn submitter_waits_while_the_database_is_behind() {
    const LIMIT: usize = 8;

    let db = SlowDb::default();
    db.hold();

    let writer = WriteBehind::with_backlog_limit(&db, 16, LIMIT);

    let submitted = AtomicU64::new(0);
    let stop = AtomicBool::new(false);

    std::thread::scope(|scope| {
        scope.spawn(|| {
            while !stop.load(Ordering::SeqCst) {
                let mut batch = writer.new_write_batch();
                let epoch = batch.epoch();

                batch.put_wide_column::<Column, u64>(&epoch.0, Some(&epoch.0));
                writer.submit_write_batch(batch);

                submitted.fetch_add(1, Ordering::SeqCst);
            }
        });

        // the database commits nothing, so this is only a matter of time
        while writer.backlog.uncommitted.load(Ordering::SeqCst) <= LIMIT {
            std::thread::yield_now();
        }

        // whichever submission the submitter was in the middle of is done
        // by now, and the next one cannot be
        std::thread::sleep(Duration::from_millis(100));
        let waiting_at = submitted.load(Ordering::SeqCst);

        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(
            submitted.load(Ordering::SeqCst),
            waiting_at,
            "the submitter went on while the database was behind"
        );

        stop.store(true, Ordering::SeqCst);
        db.release();
    });

    let submitted = submitted.load(Ordering::SeqCst);

    drop(writer);

    assert!(db.keys.lock().iter().copied().eq(0..submitted));
}

/// A write batch that has not been submitted keeps every batch after it from
/// being committed. Whoever submits those must not be left waiting for the
/// database: the database cannot do anything about them, and the batch that
/// is missing may be up to one of the waiting submitters to submit.
#[test]
fn write_batches_held_back_by_a_missing_one_keep_nobody_waiting() {
    const BATCHES: u64 = 64;
    const LIMIT: usize = 8;

    let db = SlowDb::default();
    let writer = WriteBehind::with_backlog_limit(&db, 16, LIMIT);
    let committed = writer.committed.clone();

    let mut batches = (0..BATCHES)
        .map(|_| {
            let mut batch = writer.new_write_batch();
            let epoch = batch.epoch();

            batch.put_wide_column::<Column, u64>(&epoch.0, Some(&epoch.0));
            batch
        })
        .collect::<Vec<_>>();

    let first = batches.remove(0);

    let (done_sender, done_receiver) = mpsc::channel();

    let waited = std::thread::scope(|scope| {
        let writer = &writer;

        // far more than the limit, and all of it behind the first batch
        scope.spawn(move || {
            for batch in batches {
                writer.submit_write_batch(batch);
            }

            done_sender.send(()).unwrap();
        });

        let waited =
            done_receiver.recv_timeout(Duration::from_secs(5)).is_err();

        // this is what lets a submitter that did wait go on, so that the
        // test fails instead of hanging
        writer.submit_write_batch(first);

        waited
    });

    assert!(!waited, "a submitter waits for what cannot be committed");

    while !committed.contains(Epoch(BATCHES - 1)) {
        std::thread::yield_now();
    }

    drop(writer);

    assert!(db.keys.lock().iter().copied().eq(0..BATCHES));
}

/// A write batch that is never submitted keeps every write batch after it
/// from being committed, without anything saying so. Dropping one is
/// therefore a panic instead.
#[test]
#[should_panic(expected = "dropped without being submitted")]
fn write_batch_dropped_without_being_submitted_panics() {
    let db = SlowDb::default();
    let writer = WriteBehind::new(&db, 16);

    drop(writer.new_write_batch());
}

/// A write that the database refuses kills the commit thread, and nothing is
/// committed after that. A submitter that waits for the database to catch up
/// must be told, instead of waiting for what is never going to happen.
#[test]
fn submitter_that_waits_panics_once_the_commit_thread_has_died() {
    const LIMIT: usize = 8;

    let db = SlowDb::default();
    db.hold();

    let writer = WriteBehind::with_backlog_limit(&db, 16, LIMIT);
    let (done_sender, done_receiver) = mpsc::channel();

    std::thread::scope(|scope| {
        let writer = &writer;

        scope.spawn(move || {
            let outcome =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    loop {
                        let mut batch = writer.new_write_batch();
                        let epoch = batch.epoch();

                        batch.put_wide_column::<Column, u64>(
                            &epoch.0,
                            Some(&epoch.0),
                        );
                        writer.submit_write_batch(batch);
                    }
                }));

            done_sender.send(outcome.is_err()).unwrap();
        });

        // the database commits nothing, so the submitter ends up waiting
        while writer.backlog.uncommitted.load(Ordering::SeqCst) <= LIMIT {
            std::thread::yield_now();
        }
        std::thread::sleep(Duration::from_millis(100));

        db.broken.store(true, Ordering::SeqCst);
        db.release();

        let panicked = done_receiver.recv_timeout(Duration::from_secs(5));

        if panicked.is_err() {
            // lets the submitter out, so that the test fails instead of
            // hanging
            writer.backlog.fail();
        }

        assert_eq!(panicked, Ok(true), "the submitter is still waiting");
    });
}
