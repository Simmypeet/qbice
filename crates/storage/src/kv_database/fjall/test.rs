use qbice_serialize::Plugin;
use qbice_stable_type_id::Identifiable;

use crate::kv_database::{
    DiscriminantEncoding, KeyOfSetColumn, KvDatabase, SerializationBuffer,
    WideColumn, WideColumnValue, WriteBatch, fjall::Fjall,
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

/// A write buffer holds every write that was made to it, in the order they
/// were made, also when a key was written more than once. The last write to
/// a key must be the one that decides what the key holds, within a buffer and
/// from one buffer of a batch to the next.
#[test]
fn last_operation_on_a_key_wins() {
    let tempdir = tempfile::tempdir().unwrap();
    let db = Fjall::open(tempdir.path(), Plugin::default()).unwrap();

    let mut first = db.serialization_buffer();

    first.put::<WideColumnTest, u64>(&1, &10);
    first.put::<WideColumnTest, u64>(&1, &11);

    first.put::<WideColumnTest, u64>(&2, &20);
    first.delete::<WideColumnTest, u64>(&2);

    first.delete::<WideColumnTest, u64>(&3);
    first.put::<WideColumnTest, u64>(&3, &30);

    first.put::<WideColumnTest, u64>(&4, &40);
    first.put::<WideColumnTest, u64>(&5, &50);

    first.insert_member::<KeyOfSetTest>(&0, &1);
    first.delete_member::<KeyOfSetTest>(&0, &1);

    first.delete_member::<KeyOfSetTest>(&0, &2);
    first.insert_member::<KeyOfSetTest>(&0, &2);

    first.insert_member::<KeyOfSetTest>(&0, &3);
    first.insert_member::<KeyOfSetTest>(&0, &4);

    let mut second = db.serialization_buffer();

    second.put::<WideColumnTest, u64>(&4, &41);
    second.delete::<WideColumnTest, u64>(&5);

    second.delete_member::<KeyOfSetTest>(&0, &3);

    let mut batch = db.write_batch();

    batch.consume_serialization_buffer(first);
    batch.consume_serialization_buffer(second);
    batch.commit();

    assert_eq!(db.get_wide_column::<WideColumnTest, u64>(&1), Some(11));
    assert_eq!(db.get_wide_column::<WideColumnTest, u64>(&2), None);
    assert_eq!(db.get_wide_column::<WideColumnTest, u64>(&3), Some(30));
    assert_eq!(db.get_wide_column::<WideColumnTest, u64>(&4), Some(41));
    assert_eq!(db.get_wide_column::<WideColumnTest, u64>(&5), None);

    let mut members = db.scan_members::<KeyOfSetTest>(&0).collect::<Vec<_>>();
    members.sort_unstable();

    assert_eq!(members, vec![2, 4]);
}
