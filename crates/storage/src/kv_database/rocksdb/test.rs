use std::collections::{HashMap, HashSet};

use qbice_serialize::Plugin;
use qbice_stable_type_id::Identifiable;

use crate::kv_database::{
    DiscriminantEncoding, KeyOfSetColumn, KvDatabase, SerializationBuffer,
    WideColumn, WideColumnValue, WriteBatch, rocksdb::RocksDB,
};

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Identifiable,
)]
#[stable_type_id_crate(qbice_stable_type_id)]
pub struct KeyOfSetTest;

impl KeyOfSetColumn for KeyOfSetTest {
    type Key = i32;

    type Element = i32;
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Identifiable,
)]
#[stable_type_id_crate(qbice_stable_type_id)]
pub struct WideColumnTest;

impl WideColumn for WideColumnTest {
    type Key = u32;

    type Discriminant = ();

    fn discriminant_encoding() -> DiscriminantEncoding {
        DiscriminantEncoding::Prefixed
    }
}

impl WideColumnValue<WideColumnTest> for u64 {
    fn discriminant() {}
}

/// A stream of pseudo-random numbers (splitmix64).
fn random(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);

    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);

    z ^ (z >> 31)
}

/// The values and the sets that a sequence of operations leaves behind.
#[derive(Debug, Default, PartialEq, Eq)]
struct Model {
    values: HashMap<u32, u64>,
    members: HashMap<i32, HashSet<i32>>,
}

const MODEL_KEYS: u32 = 512;
const MODEL_SETS: i32 = 16;
const MODEL_ELEMENTS: i32 = 64;

impl Model {
    /// Adds one pseudo-random operation to `writer` and applies it to the
    /// model. The keys are few, so most of them are written many times.
    fn step<W>(
        &mut self,
        state: &mut u64,
        writer: &mut W,
        put: impl Fn(&mut W, u32, u64),
        delete: impl Fn(&mut W, u32),
        insert_member: impl Fn(&mut W, i32, i32),
        delete_member: impl Fn(&mut W, i32, i32),
    ) {
        let choice = random(state);
        let key = u32::try_from(random(state) % u64::from(MODEL_KEYS)).unwrap();
        let set = i32::try_from(random(state) % 16).unwrap() % MODEL_SETS;
        let element =
            i32::try_from(random(state) % 64).unwrap() % MODEL_ELEMENTS;

        match choice % 4 {
            0 => {
                put(writer, key, choice);
                self.values.insert(key, choice);
            }
            1 => {
                delete(writer, key);
                self.values.remove(&key);
            }
            2 => {
                insert_member(writer, set, element);
                self.members.entry(set).or_default().insert(element);
            }
            _ => {
                delete_member(writer, set, element);
                self.members.entry(set).or_default().remove(&element);
            }
        }
    }

    /// Reads what the database holds for every key the model can write.
    fn read(db: &RocksDB) -> Self {
        Self {
            values: (0..MODEL_KEYS)
                .filter_map(|key| {
                    db.get_wide_column::<WideColumnTest, u64>(&key)
                        .map(|value| (key, value))
                })
                .collect(),
            members: (0..MODEL_SETS)
                .map(|set| {
                    (set, db.scan_members::<KeyOfSetTest>(&set).collect())
                })
                .collect(),
        }
    }

    /// Gives every set an entry, the way [`Model::read`] does.
    fn with_every_set(mut self) -> Self {
        for set in 0..MODEL_SETS {
            self.members.entry(set).or_default();
        }

        self
    }
}

fn step_batch(model: &mut Model, state: &mut u64, batch: &mut impl WriteBatch) {
    model.step(
        state,
        batch,
        |batch, key, value| batch.put::<WideColumnTest, u64>(&key, &value),
        |batch, key| batch.delete::<WideColumnTest, u64>(&key),
        |batch, set, element| {
            batch.insert_member::<KeyOfSetTest>(&set, &element);
        },
        |batch, set, element| {
            batch.delete_member::<KeyOfSetTest>(&set, &element);
        },
    );
}

fn step_buffer(
    model: &mut Model,
    state: &mut u64,
    buffer: &mut impl SerializationBuffer,
) {
    model.step(
        state,
        buffer,
        |buffer, key, value| buffer.put::<WideColumnTest, u64>(&key, &value),
        |buffer, key| buffer.delete::<WideColumnTest, u64>(&key),
        |buffer, set, element| {
            buffer.insert_member::<KeyOfSetTest>(&set, &element);
        },
        |buffer, set, element| {
            buffer.delete_member::<KeyOfSetTest>(&set, &element);
        },
    );
}

/// The operations of a batch are reordered before they are written. The last
/// operation on a key must still be the one that decides what the key holds.
fn last_operation_on_a_key_wins(operations: usize) {
    let tempdir = tempfile::tempdir().unwrap();
    let db = RocksDB::open(tempdir.path(), Plugin::default()).unwrap();

    let mut model = Model::default();
    let mut state = 1;
    let mut batch = db.write_batch();

    // through serialization buffers, the way the write-behind adds them
    for _ in 0..operations / 20 {
        let mut buffer = db.serialization_buffer();

        for _ in 0..10 {
            step_buffer(&mut model, &mut state, &mut buffer);
        }

        batch.consume_serialization_buffer(buffer);
    }

    // directly
    for _ in 0..operations / 2 {
        step_batch(&mut model, &mut state, &mut batch);
    }

    batch.commit();

    assert_eq!(Model::read(&db), model.with_every_set());
}

#[test]
fn last_operation_on_a_key_wins_in_a_small_batch() {
    last_operation_on_a_key_wins(1_000);
}

#[test]
fn last_operation_on_a_key_wins_in_a_large_batch() {
    last_operation_on_a_key_wins(100_000);
}

#[test]
fn operations_added_after_prepare_are_applied_last() {
    let tempdir = tempfile::tempdir().unwrap();
    let db = RocksDB::open(tempdir.path(), Plugin::default()).unwrap();

    let mut model = Model::default();
    let mut state = 2;
    let mut batch = db.write_batch();

    for _ in 0..3 {
        for _ in 0..2_000 {
            step_batch(&mut model, &mut state, &mut batch);
        }

        batch.prepare();
    }

    // nothing is written before the commit
    assert_eq!(Model::read(&db), Model::default().with_every_set());

    batch.commit();

    assert_eq!(Model::read(&db), model.with_every_set());
}

#[tokio::test]
async fn scan_iterator_isolation() {
    let tempdir = tempfile::tempdir().unwrap();
    let db = RocksDB::open(tempdir.path(), Plugin::default()).unwrap();

    let mut a = db.write_batch();

    a.insert_member::<KeyOfSetTest>(&0, &1);
    a.insert_member::<KeyOfSetTest>(&0, &2);
    a.insert_member::<KeyOfSetTest>(&0, &3);

    a.commit();

    let scanned = db.scan_members::<KeyOfSetTest>(&0).collect::<HashSet<_>>();
    let expected = HashSet::from([1, 2, 3]);

    assert_eq!(scanned, expected);

    let mut b = db.write_batch();
    b.insert_member::<KeyOfSetTest>(&0, &4);

    // Iterator should not see uncommitted data
    let iter_after =
        db.scan_members::<KeyOfSetTest>(&0).collect::<HashSet<_>>();

    assert_eq!(iter_after, expected);

    b.commit();

    let iter_final =
        db.scan_members::<KeyOfSetTest>(&0).collect::<HashSet<_>>();

    let mut c = db.write_batch();
    c.insert_member::<KeyOfSetTest>(&0, &5);
    c.commit();

    let expected_final = HashSet::from([1, 2, 3, 4]);

    // Should not see 5 as it was added after the iterator was created
    assert_eq!(iter_final, expected_final);
}
