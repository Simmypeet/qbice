use std::sync::Arc;

use dashmap::DashSet;

use super::{
    ConcurrentLog, ConcurrentLogMessage, MergeIterator, Operation, Spilled,
    VersionedOperation,
};
use crate::{
    key_of_set_map::{ConcurrentSet, OwnedIterator},
    write_manager::write_behind::Epoch,
};

type Merge = MergeIterator<
    Arc<DashSet<i32>>,
    std::vec::IntoIter<i32>,
    i32,
    std::iter::Empty<i32>,
>;

fn append(log: &ConcurrentLog<i32>, op: Operation<i32>, epoch: u64) {
    log.apply_message(ConcurrentLogMessage::AppendOperation(
        VersionedOperation { op, epoch: Epoch(epoch) },
    ));
}

fn flush_up_to(log: &ConcurrentLog<i32>, epoch: u64) {
    log.apply_message(ConcurrentLogMessage::FlushUpTo(Epoch(epoch)));
}

fn staged_epochs(log: &ConcurrentLog<i32>) -> Vec<u64> {
    log.log.read().operations.keys().map(|(epoch, _)| epoch.0).collect()
}

/// The elements the staged operations add and remove, each sorted.
fn snapshot(log: &ConcurrentLog<i32>) -> (Vec<i32>, Vec<i32>) {
    let snapshot = log.get_snapshot();

    let mut added = snapshot.added.into_iter().collect::<Vec<_>>();
    let mut removed = snapshot.removed.into_iter().collect::<Vec<_>>();

    added.sort_unstable();
    removed.sort_unstable();

    (added, removed)
}

fn sorted(merge: Merge) -> Vec<i32> {
    let mut members = merge.collect::<Vec<_>>();

    members.sort_unstable();
    members
}

#[test]
fn flush_removes_every_operation_up_to_the_flushed_epoch() {
    let log = ConcurrentLog::new();

    // batches do not append in epoch order
    append(&log, Operation::Insert(3), 3);
    append(&log, Operation::Insert(0), 0);
    append(&log, Operation::Insert(2), 2);
    append(&log, Operation::Insert(1), 1);

    flush_up_to(&log, 1);
    assert_eq!(staged_epochs(&log), [2, 3]);

    flush_up_to(&log, 3);
    assert_eq!(staged_epochs(&log), [0; 0]);
}

/// This is the log the engine builds for a backward edge whose caller is
/// computed once and then recomputed twice: every recompute removes the edge
/// and inserts it again. The edge must still be there.
#[test]
fn snapshot_keeps_an_element_that_is_removed_and_reinserted() {
    let log = ConcurrentLog::new();

    append(&log, Operation::Insert(7), 2);
    append(&log, Operation::Remove(7), 109);
    append(&log, Operation::Insert(7), 109);
    append(&log, Operation::Remove(7), 214);
    append(&log, Operation::Insert(7), 214);

    assert_eq!(snapshot(&log), (vec![7], vec![]));
}

/// A committed write batch stays in the log until the after-commit thread
/// flushes it, so the database may already hold the insert. The remove has to
/// be reported anyway instead of cancelling out against the insert.
#[test]
fn snapshot_reports_a_remove_that_follows_an_insert() {
    let log = ConcurrentLog::new();

    append(&log, Operation::Insert(7), 0);
    append(&log, Operation::Remove(7), 1);

    assert_eq!(snapshot(&log), (vec![], vec![7]));
}

#[test]
fn snapshot_orders_operations_by_epoch_then_by_arrival() {
    // the database applies epoch 1 before epoch 5, whichever arrives first
    let log = ConcurrentLog::new();

    append(&log, Operation::Remove(7), 5);
    append(&log, Operation::Insert(7), 1);

    assert_eq!(snapshot(&log), (vec![], vec![7]));

    // within one write batch the later operation wins
    let log = ConcurrentLog::new();

    append(&log, Operation::Remove(7), 1);
    append(&log, Operation::Insert(7), 1);
    append(&log, Operation::Insert(8), 1);
    append(&log, Operation::Remove(8), 1);

    assert_eq!(snapshot(&log), (vec![7], vec![8]));
}

#[test]
fn staged_remove_survives_the_flush_of_an_older_insert() {
    let log = ConcurrentLog::new();

    append(&log, Operation::Insert(7), 0);
    append(&log, Operation::Remove(7), 1);

    flush_up_to(&log, 0);

    assert_eq!(snapshot(&log), (vec![], vec![7]));
}

#[test]
fn staged_insert_survives_the_flush_of_an_older_remove() {
    let log = ConcurrentLog::new();

    append(&log, Operation::Remove(7), 0);
    append(&log, Operation::Insert(7), 1);

    flush_up_to(&log, 0);

    assert_eq!(snapshot(&log), (vec![7], vec![]));
}

/// The database already holds a staged insert when its write batch has been
/// committed but not yet flushed from staging. The element must not be
/// yielded twice.
#[test]
fn streaming_merge_yields_every_member_once() {
    let log = ConcurrentLog::new();

    append(&log, Operation::Remove(1), 0);
    append(&log, Operation::Insert(3), 0);
    append(&log, Operation::Insert(4), 0);

    let merge = Merge::Streaming(
        vec![1, 2, 3].into_iter(),
        log.get_snapshot().into_iter_snapshot(),
    );

    assert_eq!(sorted(merge), [2, 3, 4]);
}

/// Skipping a removed element in the partially loaded set must not end the
/// iteration while there are members left.
#[test]
fn spilled_merge_yields_every_member_that_was_not_removed() {
    let log = ConcurrentLog::new();

    for element in 0..50 {
        append(&log, Operation::Remove(element), 0);
    }

    append(&log, Operation::Insert(200), 0);

    let half_constructed = Arc::new((0..100).collect::<DashSet<i32>>());

    let merge = Merge::Spilled(
        Spilled {
            half_constructed: OwnedIterator::new(half_constructed, |set| {
                set.iter()
            }),
            rest_iterator: vec![100, 101].into_iter(),
        },
        log.get_snapshot().into_iter_snapshot(),
    );

    let mut expected = (50..102).collect::<Vec<_>>();
    expected.push(200);

    assert_eq!(sorted(merge), expected);
}

/// These replay, without the background threads, the calls the write-behind
/// makes on the map.
#[cfg(feature = "rocksdb")]
mod rocksdb {
    use std::sync::Arc;

    use dashmap::DashSet;
    use qbice_serialize::Plugin;
    use qbice_stable_type_id::Identifiable;

    use super::super::{CacheKeyOfSetMap, Operation};
    use crate::{
        key_of_set_map::KeyOfSetMap,
        kv_database::{
            KeyOfSetColumn, KvDatabase, WriteBatch, rocksdb::RocksDB,
        },
        write_manager::write_behind::Epoch,
    };

    #[derive(
        Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Identifiable,
    )]
    #[stable_type_id_crate(qbice_stable_type_id)]
    pub struct Column;

    impl KeyOfSetColumn for Column {
        type Key = i32;

        type Element = i32;
    }

    type Map = CacheKeyOfSetMap<Column, Arc<DashSet<i32>>, RocksDB>;

    fn open(tempdir: &tempfile::TempDir) -> (RocksDB, Map) {
        let db = RocksDB::open(tempdir.path(), Plugin::default()).unwrap();
        let map = Map::new(16, db.clone());

        (db, map)
    }

    /// Nothing has been committed yet and the set is not cached, so its
    /// members come from the staged operations alone.
    #[tokio::test]
    async fn uncached_read_keeps_an_element_that_is_removed_and_reinserted() {
        let tempdir = tempfile::tempdir().unwrap();
        let (_db, map) = open(&tempdir);

        map.apply_op(&0, Operation::Insert(7), Epoch(2), true);

        for epoch in [109, 214] {
            map.apply_op(&0, Operation::Remove(7), Epoch(epoch), true);
            map.apply_op(&0, Operation::Insert(7), Epoch(epoch), false);
        }

        assert_eq!(map.get(&0).await.collect::<Vec<_>>(), [7]);
    }

    /// An older write batch is committed and flushed from staging while a
    /// newer one that touches the same set is still open.
    #[tokio::test]
    async fn uncached_read_sees_a_remove_staged_after_a_flushed_insert() {
        let tempdir = tempfile::tempdir().unwrap();
        let (db, map) = open(&tempdir);

        map.apply_op(&0, Operation::Insert(7), Epoch(0), true);
        map.apply_op(&0, Operation::Remove(7), Epoch(1), true);

        // only epoch 0 reaches the database
        let mut batch = db.write_batch();
        batch.insert_member::<Column>(&0, &7);
        batch.commit();

        map.repr.flush_staging(Epoch(0), [0]);

        assert_eq!(map.get(&0).await.collect::<Vec<_>>(), [0; 0]);
    }
}

/// These run a write in the middle of a load, which needs a database that
/// can be interrupted while it is being scanned.
mod load {
    use std::sync::Arc;

    use dashmap::DashSet;
    use parking_lot::Mutex;
    use qbice_stable_type_id::Identifiable;

    use super::super::{CacheKeyOfSetMap, Operation};
    use crate::{
        key_of_set_map::KeyOfSetMap,
        kv_database::{
            KeyOfSetColumn, KvDatabase, SerializationBuffer, WideColumn,
            WideColumnValue, WriteBatch,
        },
        write_manager::write_behind::Epoch,
    };

    #[derive(
        Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Identifiable,
    )]
    #[stable_type_id_crate(qbice_stable_type_id)]
    pub struct Column;

    impl KeyOfSetColumn for Column {
        type Key = i32;

        type Element = i32;
    }

    type Hook = Box<dyn FnOnce() + Send + Sync>;

    /// An empty database that runs a hook while it is being scanned.
    #[derive(Clone, Default)]
    struct InterruptibleDb {
        during_scan: Arc<Mutex<Option<Hook>>>,
    }

    /// Nothing is ever written to [`InterruptibleDb`].
    struct Unused;

    impl SerializationBuffer for Unused {
        fn put<W: WideColumn, C: WideColumnValue<W>>(
            &mut self,
            _: &W::Key,
            _: &C,
        ) {
            unreachable!()
        }

        fn delete<W: WideColumn, C: WideColumnValue<W>>(&mut self, _: &W::Key) {
            unreachable!()
        }

        fn insert_member<C: KeyOfSetColumn>(
            &mut self,
            _: &C::Key,
            _: &C::Element,
        ) {
            unreachable!()
        }

        fn delete_member<C: KeyOfSetColumn>(
            &mut self,
            _: &C::Key,
            _: &C::Element,
        ) {
            unreachable!()
        }
    }

    impl WriteBatch for Unused {
        type SerializationBuffer = Self;

        fn put<W: WideColumn, C: WideColumnValue<W>>(
            &mut self,
            _: &W::Key,
            _: &C,
        ) {
            unreachable!()
        }

        fn delete<W: WideColumn, C: WideColumnValue<W>>(&mut self, _: &W::Key) {
            unreachable!()
        }

        fn insert_member<C: KeyOfSetColumn>(
            &mut self,
            _: &C::Key,
            _: &C::Element,
        ) {
            unreachable!()
        }

        fn delete_member<C: KeyOfSetColumn>(
            &mut self,
            _: &C::Key,
            _: &C::Element,
        ) {
            unreachable!()
        }

        fn consume_serialization_buffer(&mut self, _: Self) { unreachable!() }

        fn commit(self) { unreachable!() }
    }

    impl KvDatabase for InterruptibleDb {
        type WriteBatch = Unused;

        type SerializationBuffer = Unused;

        type ScanMemberIterator<C: KeyOfSetColumn> =
            std::iter::Empty<C::Element>;

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
            let hook = self.during_scan.lock().take();

            if let Some(hook) = hook {
                hook();
            }

            std::iter::empty()
        }

        fn write_batch(&self) -> Unused { Unused }

        fn serialization_buffer(&self) -> Unused { Unused }
    }

    type Map = CacheKeyOfSetMap<Column, Arc<DashSet<i32>>, InterruptibleDb>;

    /// The operation is in neither the load's snapshot nor the database, and
    /// its writer finds no cached set to apply it to. The loaded set must not
    /// go on being served without it.
    #[tokio::test]
    async fn operation_staged_during_a_load_is_not_lost() {
        let db = InterruptibleDb::default();
        let map = Arc::new(Map::new(16, db.clone()));

        *db.during_scan.lock() = Some(Box::new({
            let map = map.clone();

            move || map.apply_op(&0, Operation::Insert(7), Epoch(0), true)
        }));

        // the read that overlaps the write may or may not see it
        let _ = map.get(&0).await.count();

        // every later read must
        assert_eq!(map.get(&0).await.collect::<Vec<_>>(), [7]);
    }

    /// The check for overlapping writes must not stop sets from being cached.
    #[tokio::test]
    async fn load_without_an_overlapping_write_stays_cached() {
        let map = Map::new(16, InterruptibleDb::default());

        map.apply_op(&0, Operation::Insert(7), Epoch(0), true);

        assert_eq!(map.get(&0).await.collect::<Vec<_>>(), [7]);
        assert!(map.repr.cache.get(&0).is_some());
    }
}
