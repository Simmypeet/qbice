use std::{any::Any, sync::Arc, time::Duration};

use parking_lot::Mutex;
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
}

/// The keys of the values that were put.
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
    let writer = WriteBehind::new(&db, 2, 16);
    let committed = writer.committed.clone();

    let batches =
        (0..BATCHES).map(|_| writer.new_write_batch()).collect::<Vec<_>>();

    // the newest first, so that nothing can be committed before the last
    // submission
    for mut batch in batches.into_iter().rev() {
        let epoch = batch.epoch();

        batch.put_wide_column::<Column, u64>(epoch.0, Some(epoch.0));
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
