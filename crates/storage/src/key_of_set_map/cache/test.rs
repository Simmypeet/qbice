use std::{
    any::Any,
    collections::{BTreeSet, HashMap, VecDeque},
    sync::Arc,
};

use dashmap::DashSet;
use parking_lot::Mutex;
use qbice_stable_type_id::Identifiable;

use super::{CacheKeyOfSetMap, LOADED_SET_LIMIT, Stored};
use crate::{
    key_of_set_map::KeyOfSetMap,
    kv_database::{
        KeyOfSetColumn, KvDatabase, SerializationBuffer, WideColumn,
        WideColumnValue, WriteBatch,
    },
    write_manager::write_behind::{Epoch, KeyOfSetCache, Operation},
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

/// A database that keeps the sets of [`Column`] in memory.
///
/// A scan sees the members as they are when it starts. `during_scan` runs
/// right after that, before the scan hands anything out, which is how a test
/// makes something happen in the middle of a load.
#[derive(Clone, Default)]
struct MemoryDb {
    sets: Arc<Mutex<HashMap<i32, BTreeSet<i32>>>>,
    during_scan: Arc<Mutex<Option<Hook>>>,
}

impl MemoryDb {
    fn members(&self, key: i32) -> Vec<i32> {
        self.sets
            .lock()
            .get(&key)
            .map(|set| set.iter().copied().collect())
            .unwrap_or_default()
    }

    fn set_members(&self, key: i32, members: impl IntoIterator<Item = i32>) {
        self.sets.lock().insert(key, members.into_iter().collect());
    }
}

/// The tests write to [`MemoryDb`] directly.
struct Unused;

impl SerializationBuffer for Unused {
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
}

impl WriteBatch for Unused {
    type SerializationBuffer = Self;

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

    fn consume_serialization_buffer(&mut self, _: Self) { unreachable!() }

    fn commit(self) { unreachable!() }
}

impl KvDatabase for MemoryDb {
    type WriteBatch = Unused;

    type SerializationBuffer = Unused;

    type ScanMemberIterator<C: KeyOfSetColumn> = std::vec::IntoIter<C::Element>;

    fn get_wide_column<W: WideColumn, C: WideColumnValue<W>>(
        &self,
        _: &W::Key,
    ) -> Option<C> {
        None
    }

    fn scan_members<C: KeyOfSetColumn>(
        &self,
        key: &C::Key,
    ) -> Self::ScanMemberIterator<C> {
        let key = *(key as &dyn Any)
            .downcast_ref::<i32>()
            .expect("only `Column` is stored");

        let members = self.members(key);

        let hook = self.during_scan.lock().take();

        if let Some(hook) = hook {
            hook();
        }

        let members: Box<dyn Any> = Box::new(members);
        let members = *members
            .downcast::<Vec<C::Element>>()
            .expect("only `Column` is stored");

        members.into_iter()
    }

    fn write_batch(&self) -> Unused { Unused }

    fn serialization_buffer(&self) -> Unused { Unused }
}

type Map = CacheKeyOfSetMap<Column, Arc<DashSet<i32>>, MemoryDb>;

/// Opens a map over an empty database. The map keeps up to `capacity` sets
/// and only loads the members of sets that have at most `limit` of them.
fn open(capacity: u64, limit: usize) -> (MemoryDb, Arc<Map>) {
    let db = MemoryDb::default();
    let map = Arc::new(Map::with_loaded_set_limit(capacity, db.clone(), limit));

    (db, map)
}

async fn members(map: &Map, key: i32) -> Vec<i32> {
    let mut members = map.get(&key).await.collect::<Vec<_>>();

    members.sort_unstable();
    members
}

/// What a write batch records for the write-behind.
struct Batch {
    epoch: u64,
    writes: HashMap<i32, HashMap<i32, Operation>>,
}

impl Batch {
    fn new(epoch: u64) -> Self { Self { epoch, writes: HashMap::new() } }

    fn stage(
        &mut self,
        map: &Map,
        key: i32,
        element: i32,
        operation: Operation,
    ) {
        let first_in_batch = !self.writes.contains_key(&key);

        self.writes.entry(key).or_default().insert(element, operation);

        map.repr.stage(
            key,
            element,
            operation,
            Epoch(self.epoch),
            first_in_batch,
        );
    }

    fn insert(&mut self, map: &Map, key: i32, element: i32) {
        self.stage(map, key, element, Operation::Insert);
    }

    fn remove(&mut self, map: &Map, key: i32, element: i32) {
        self.stage(map, key, element, Operation::Remove);
    }

    /// Does what the write-behind does with a batch: writes it to the
    /// database and then tells the cache about it.
    fn commit(self, db: &MemoryDb, map: &Map) {
        {
            let mut sets = db.sets.lock();

            for (key, operations) in &self.writes {
                let set = sets.entry(*key).or_default();

                for (element, operation) in operations {
                    match operation {
                        Operation::Insert => {
                            set.insert(*element);
                        }
                        Operation::Remove => {
                            set.remove(element);
                        }
                    }
                }
            }
        }

        KeyOfSetCache::<Column, MemoryDb>::flush(
            &*map.repr,
            Epoch(self.epoch),
            &mut self.writes.into_iter(),
        );
    }
}

fn is_loaded(map: &Map, key: i32) -> bool {
    map.repr
        .sets
        .get_map(&key, |set| matches!(set.stored, Stored::Loaded(_)))
        .unwrap_or(false)
}

fn is_too_large(map: &Map, key: i32) -> bool {
    map.repr
        .sets
        .get_map(&key, |set| matches!(set.stored, Stored::TooLarge))
        .unwrap_or(false)
}

fn is_cached(map: &Map, key: i32) -> bool {
    map.repr.sets.get_map(&key, |_| ()).is_some()
}

fn pending_count(map: &Map, key: i32) -> usize {
    map.repr.sets.get_map(&key, |set| set.pending.len()).unwrap_or(0)
}

#[tokio::test]
async fn pending_writes_are_laid_over_the_stored_members() {
    let (db, map) = open(16, LOADED_SET_LIMIT);
    db.set_members(0, [1, 2, 3]);

    let mut batch = Batch::new(0);
    batch.remove(&map, 0, 1);
    batch.insert(&map, 0, 3); // already stored: must not show up twice
    batch.insert(&map, 0, 4);

    // the first read loads the set, the second finds it in memory
    assert_eq!(members(&map, 0).await, [2, 3, 4]);
    assert_eq!(members(&map, 0).await, [2, 3, 4]);

    batch.commit(&db, &map);

    assert_eq!(members(&map, 0).await, [2, 3, 4]);
    assert_eq!(db.members(0), [2, 3, 4]);
}

/// This is what the engine does to a backward edge whose caller is computed
/// once and then recomputed twice: every recompute removes the edge and
/// inserts it again.
#[tokio::test]
async fn element_that_is_removed_and_reinserted_stays_a_member() {
    let (db, map) = open(16, LOADED_SET_LIMIT);

    let mut first = Batch::new(2);
    first.insert(&map, 0, 7);

    let mut second = Batch::new(109);
    second.remove(&map, 0, 7);
    second.insert(&map, 0, 7);

    let mut third = Batch::new(214);
    third.remove(&map, 0, 7);
    third.insert(&map, 0, 7);

    assert_eq!(members(&map, 0).await, [7]);

    for batch in [first, second, third] {
        batch.commit(&db, &map);

        assert_eq!(members(&map, 0).await, [7]);
    }

    assert_eq!(db.members(0), [7]);
}

#[tokio::test]
async fn later_write_batch_decides_whatever_the_arrival_order() {
    let (db, map) = open(16, LOADED_SET_LIMIT);

    let mut older = Batch::new(1);
    let mut newer = Batch::new(5);

    // the database applies the older batch first, although it arrives last
    newer.remove(&map, 0, 7);
    older.insert(&map, 0, 7);

    assert_eq!(members(&map, 0).await, [0; 0]);

    older.commit(&db, &map);
    assert_eq!(members(&map, 0).await, [0; 0]);

    newer.commit(&db, &map);
    assert_eq!(members(&map, 0).await, [0; 0]);
    assert_eq!(db.members(0), [0; 0]);
}

#[tokio::test]
async fn later_operation_of_a_write_batch_decides() {
    let (db, map) = open(16, LOADED_SET_LIMIT);

    let mut batch = Batch::new(0);
    batch.remove(&map, 0, 7);
    batch.insert(&map, 0, 7);
    batch.insert(&map, 0, 8);
    batch.remove(&map, 0, 8);

    assert_eq!(members(&map, 0).await, [7]);

    batch.commit(&db, &map);

    assert_eq!(members(&map, 0).await, [7]);
}

/// An insert that has reached the database must not hide a later remove that
/// is still pending.
#[tokio::test]
async fn pending_remove_survives_the_flush_of_an_older_insert() {
    let (db, map) = open(16, LOADED_SET_LIMIT);

    let mut older = Batch::new(0);
    older.insert(&map, 0, 7);

    let mut newer = Batch::new(1);
    newer.remove(&map, 0, 7);

    older.commit(&db, &map);

    assert_eq!(db.members(0), [7]);
    assert_eq!(members(&map, 0).await, [0; 0]);

    newer.commit(&db, &map);

    assert_eq!(members(&map, 0).await, [0; 0]);
}

/// The mirror image: a remove that has reached the database must not hide a
/// later insert that is still pending.
#[tokio::test]
async fn pending_insert_survives_the_flush_of_an_older_remove() {
    let (db, map) = open(16, LOADED_SET_LIMIT);
    db.set_members(0, [7]);

    let mut older = Batch::new(0);
    older.remove(&map, 0, 7);

    let mut newer = Batch::new(1);
    newer.insert(&map, 0, 7);

    older.commit(&db, &map);

    assert_eq!(db.members(0), [0; 0]);
    assert_eq!(members(&map, 0).await, [7]);

    newer.commit(&db, &map);

    assert_eq!(members(&map, 0).await, [7]);
}

#[tokio::test]
async fn flush_keeps_a_loaded_set_up_to_date() {
    let (db, map) = open(16, LOADED_SET_LIMIT);
    db.set_members(0, [1]);

    assert_eq!(members(&map, 0).await, [1]);
    assert!(is_loaded(&map, 0));

    let mut batch = Batch::new(0);
    batch.remove(&map, 0, 1);
    batch.insert(&map, 0, 2);
    batch.commit(&db, &map);

    // nothing is pending anymore, so this is the loaded copy on its own
    assert_eq!(pending_count(&map, 0), 0);
    assert!(is_loaded(&map, 0));
    assert_eq!(members(&map, 0).await, [2]);
}

#[tokio::test]
async fn set_that_is_only_written_is_dropped_once_flushed() {
    let (db, map) = open(16, LOADED_SET_LIMIT);

    let mut batch = Batch::new(0);
    batch.insert(&map, 0, 7);

    assert!(is_cached(&map, 0));

    batch.commit(&db, &map);

    assert!(!is_cached(&map, 0));
    assert_eq!(members(&map, 0).await, [7]);
}

#[tokio::test]
async fn set_at_the_limit_is_loaded_and_one_above_is_not() {
    let (db, map) = open(16, 8);

    db.set_members(0, 0..8);
    db.set_members(1, 0..9);

    assert_eq!(members(&map, 0).await, (0..8).collect::<Vec<_>>());
    assert_eq!(members(&map, 1).await, (0..9).collect::<Vec<_>>());

    assert!(is_loaded(&map, 0));
    assert!(is_too_large(&map, 1));
}

#[tokio::test]
async fn large_set_is_streamed_with_the_pending_writes_laid_over() {
    let (db, map) = open(16, LOADED_SET_LIMIT);
    db.set_members(0, 0..2000);

    let mut batch = Batch::new(0);

    for element in 0..50 {
        batch.remove(&map, 0, element);
    }

    batch.insert(&map, 0, 100); // already stored: must not show up twice
    batch.insert(&map, 0, 5000);

    let mut expected = (50..2000).collect::<Vec<_>>();
    expected.push(5000);

    // the first read finds out that the set is too large to load
    assert_eq!(members(&map, 0).await, expected);
    assert!(is_too_large(&map, 0));
    assert_eq!(members(&map, 0).await, expected);

    batch.commit(&db, &map);

    assert!(is_too_large(&map, 0));
    assert_eq!(members(&map, 0).await, expected);
}

#[tokio::test]
async fn loaded_set_that_outgrows_the_limit_is_streamed() {
    let (db, map) = open(16, 8);
    db.set_members(0, 0..6);

    assert_eq!(members(&map, 0).await, (0..6).collect::<Vec<_>>());
    assert!(is_loaded(&map, 0));

    let mut batch = Batch::new(0);

    for element in 6..12 {
        batch.insert(&map, 0, element);
    }

    // pending writes do not count towards the limit
    assert_eq!(members(&map, 0).await, (0..12).collect::<Vec<_>>());
    assert!(is_loaded(&map, 0));

    batch.commit(&db, &map);

    assert!(is_too_large(&map, 0));
    assert_eq!(members(&map, 0).await, (0..12).collect::<Vec<_>>());
}

/// The write is in neither the scan nor, when the scan started, the pending
/// writes that the load could have looked at.
#[tokio::test]
async fn write_staged_during_a_load_is_not_lost() {
    let (db, map) = open(16, LOADED_SET_LIMIT);
    db.set_members(0, [1]);

    *db.during_scan.lock() = Some(Box::new({
        let map = map.clone();

        move || map.repr.stage(0, 7, Operation::Insert, Epoch(0), true)
    }));

    // the read that overlaps the write may or may not see it
    let _ = members(&map, 0).await;

    // every later read must
    assert_eq!(members(&map, 0).await, [1, 7]);
}

/// The scan started before the write batch was committed, so it does not have
/// the batch's writes, and the flush takes them out of the pending writes
/// before the load is done.
#[tokio::test]
async fn write_batch_flushed_during_a_load_is_not_lost() {
    let (db, map) = open(16, LOADED_SET_LIMIT);
    db.set_members(0, [1, 2]);

    let mut batch = Batch::new(0);
    batch.insert(&map, 0, 7);
    batch.remove(&map, 0, 2);

    *db.during_scan.lock() = Some(Box::new({
        let (db, map) = (db.clone(), map.clone());

        move || batch.commit(&db, &map)
    }));

    let _ = members(&map, 0).await;

    // nothing is pending, so this is what the load has left in memory
    assert_eq!(pending_count(&map, 0), 0);
    assert!(is_loaded(&map, 0));
    assert_eq!(members(&map, 0).await, [1, 7]);
}

/// The same, with a write batch that is staged and flushed entirely within
/// the load.
#[tokio::test]
async fn write_batch_staged_and_flushed_during_a_load_is_not_lost() {
    let (db, map) = open(16, LOADED_SET_LIMIT);
    db.set_members(0, [1, 2]);

    *db.during_scan.lock() = Some(Box::new({
        let (db, map) = (db.clone(), map.clone());

        move || {
            let mut batch = Batch::new(0);
            batch.insert(&map, 0, 7);
            batch.remove(&map, 0, 2);
            batch.commit(&db, &map);
        }
    }));

    let _ = members(&map, 0).await;

    assert_eq!(pending_count(&map, 0), 0);
    assert_eq!(members(&map, 0).await, [1, 7]);
}

#[tokio::test]
async fn load_without_an_overlapping_write_stays_loaded() {
    let (db, map) = open(16, LOADED_SET_LIMIT);
    db.set_members(0, [1]);

    assert_eq!(members(&map, 0).await, [1]);
    assert!(is_loaded(&map, 0));
}

/// The pending writes of a set exist nowhere else until they are committed,
/// however full the cache is.
#[tokio::test]
async fn set_with_unflushed_writes_is_not_evicted() {
    let (db, map) = open(2, LOADED_SET_LIMIT);

    let mut batch = Batch::new(0);

    for key in 0..200 {
        batch.insert(&map, key, key + 1000);
    }

    for key in 0..200 {
        assert_eq!(members(&map, key).await, [key + 1000]);
    }

    batch.commit(&db, &map);

    for key in 0..200 {
        assert_eq!(members(&map, key).await, [key + 1000]);
    }
}

/// A small deterministic generator, so that a failure can be replayed from
/// its seed.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);

        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);

        z ^ (z >> 31)
    }

    fn below(&mut self, bound: u32) -> u32 {
        u32::try_from(self.next() % u64::from(bound)).unwrap()
    }

    fn index(&mut self, len: usize) -> usize {
        usize::try_from(self.next() % u64::try_from(len).unwrap()).unwrap()
    }

    fn element(&mut self, bound: u32) -> i32 {
        i32::try_from(self.below(bound)).unwrap()
    }
}

/// The map together with everything the write-behind would be holding.
struct World {
    db: MemoryDb,
    map: Arc<Map>,

    /// The write batches that have not been committed, oldest first.
    open: VecDeque<Batch>,
    next_epoch: u64,
}

impl World {
    const KEYS: u32 = 24;
    const ELEMENTS: u32 = 12;

    /// What the set of `key` has to read as: the members in the database
    /// with the open write batches applied in the order of their epochs.
    fn expected(&self, key: i32) -> Vec<i32> {
        let mut members =
            self.db.sets.lock().get(&key).cloned().unwrap_or_default();

        for batch in &self.open {
            let Some(operations) = batch.writes.get(&key) else {
                continue;
            };

            for (element, operation) in operations {
                match operation {
                    Operation::Insert => {
                        members.insert(*element);
                    }
                    Operation::Remove => {
                        members.remove(element);
                    }
                }
            }
        }

        members.into_iter().collect()
    }

    fn open_batch(&mut self) {
        self.open.push_back(Batch::new(self.next_epoch));
        self.next_epoch += 1;
    }

    /// Takes a random step that a writer or the write-behind could take.
    fn write_step(&mut self, rng: &mut Rng) {
        match rng.below(10) {
            // the write-behind commits the oldest batch
            0 | 1 => {
                if let Some(batch) = self.open.pop_front() {
                    batch.commit(&self.db, &self.map);
                }
            }

            2 => self.open_batch(),

            // any of the open batches stages an operation
            _ => {
                if self.open.is_empty() {
                    self.open_batch();
                }

                let batch = rng.index(self.open.len());
                let key = rng.element(Self::KEYS);
                let element = rng.element(Self::ELEMENTS);
                let operation = if rng.below(2) == 0 {
                    Operation::Insert
                } else {
                    Operation::Remove
                };

                self.open[batch].stage(&self.map, key, element, operation);
            }
        }
    }
}

/// Runs random writes, commits and reads against a map that is small enough
/// to keep evicting sets and to keep moving them across the size limit, and
/// checks every read against a model. Every so often a write or a commit is
/// made to happen in the middle of a load.
#[tokio::test]
async fn reads_match_a_model_under_random_interleavings() {
    for seed in 0..300 {
        let mut rng = Rng(seed);
        let (db, map) = open(2, 6);

        let world = Arc::new(Mutex::new(World {
            db: db.clone(),
            map: map.clone(),
            open: VecDeque::new(),
            next_epoch: 0,
        }));

        for step in 0..600 {
            if rng.below(3) != 0 {
                world.lock().write_step(&mut rng);
                continue;
            }

            let key = rng.element(World::KEYS);
            let before = world.lock().expected(key);

            let interrupt = rng.below(4) == 0;

            if interrupt {
                let world = world.clone();
                let mut rng = Rng(rng.next());

                *db.during_scan.lock() =
                    Some(Box::new(move || world.lock().write_step(&mut rng)));
            }

            let read = members(&map, key).await;

            // the hook is still there if the read did not have to load
            let interrupted =
                interrupt && db.during_scan.lock().take().is_none();
            let after = world.lock().expected(key);

            if interrupted {
                assert!(
                    read == before || read == after,
                    "seed {seed}, step {step}, key {key}: read {read:?}, \
                     expected {before:?} or {after:?}"
                );
            } else {
                assert_eq!(read, after, "seed {seed}, step {step}, key {key}");
            }

            // once things are quiet, there is only one right answer
            assert_eq!(
                members(&map, key).await,
                after,
                "seed {seed}, step {step}, key {key}, read again"
            );
        }
    }
}

/// The same paths against the real database.
#[cfg(feature = "rocksdb")]
mod rocksdb {
    use std::{collections::HashMap, sync::Arc};

    use dashmap::DashSet;
    use qbice_serialize::Plugin;

    use super::Column;
    use crate::{
        key_of_set_map::{KeyOfSetMap, cache::CacheKeyOfSetMap},
        kv_database::{KvDatabase, WriteBatch, rocksdb::RocksDB},
        write_manager::write_behind::{Epoch, KeyOfSetCache, Operation},
    };

    type Map = CacheKeyOfSetMap<Column, Arc<DashSet<i32>>, RocksDB>;

    fn open(tempdir: &tempfile::TempDir) -> (RocksDB, Map) {
        let db = RocksDB::open(tempdir.path(), Plugin::default()).unwrap();
        let map = Map::new(16, db.clone());

        (db, map)
    }

    async fn members(map: &Map, key: i32) -> Vec<i32> {
        let mut members = map.get(&key).await.collect::<Vec<_>>();

        members.sort_unstable();
        members
    }

    /// Nothing has been committed yet, so the members come from the pending
    /// writes alone.
    #[tokio::test]
    async fn element_that_is_removed_and_reinserted_stays_a_member() {
        let tempdir = tempfile::tempdir().unwrap();
        let (_db, map) = open(&tempdir);

        map.repr.stage(0, 7, Operation::Insert, Epoch(2), true);

        for epoch in [109, 214] {
            map.repr.stage(0, 7, Operation::Remove, Epoch(epoch), true);
            map.repr.stage(0, 7, Operation::Insert, Epoch(epoch), false);
        }

        assert_eq!(members(&map, 0).await, [7]);
    }

    /// An older write batch is committed and flushed while a newer one that
    /// writes to the same set is still open.
    #[tokio::test]
    async fn pending_remove_survives_the_flush_of_an_older_insert() {
        let tempdir = tempfile::tempdir().unwrap();
        let (db, map) = open(&tempdir);

        map.repr.stage(0, 1, Operation::Insert, Epoch(0), true);
        map.repr.stage(0, 7, Operation::Insert, Epoch(0), false);
        map.repr.stage(0, 7, Operation::Remove, Epoch(1), true);

        // only epoch 0 reaches the database
        let mut batch = db.write_batch();
        batch.insert_member::<Column>(&0, &1);
        batch.insert_member::<Column>(&0, &7);
        batch.commit();

        let committed =
            HashMap::from([(1, Operation::Insert), (7, Operation::Insert)]);

        KeyOfSetCache::<Column, RocksDB>::flush(
            &*map.repr,
            Epoch(0),
            &mut std::iter::once((0, committed)),
        );

        assert_eq!(db.scan_members::<Column>(&0).count(), 2);
        assert_eq!(members(&map, 0).await, [1]);
    }
}
