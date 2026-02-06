use std::collections::HashSet;

use qbice_serialize::Plugin;
use qbice_stable_type_id::Identifiable;

use crate::kv_database::{
    KeyOfSetColumn, KvDatabase, WriteBatch, rocksdb::RocksDB,
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

#[tokio::test]
async fn scan_iterator_isolation() {
    let tempdir = tempfile::tempdir().unwrap();
    let db = RocksDB::open(tempdir.path(), Plugin::default()).unwrap();

    let mut a = db.write_batch();

    a.insert_member::<KeyOfSetTest>(&0, &1);
    a.insert_member::<KeyOfSetTest>(&0, &2);
    a.insert_member::<KeyOfSetTest>(&0, &3);

    a.commit();

    let iter = db.scan_members::<KeyOfSetTest>(&0).collect::<HashSet<_>>();

    assert_eq!(iter.len(), 3);
    assert!(iter.contains(&0));
    assert!(iter.contains(&1));
    assert!(iter.contains(&2));

    let mut b = db.write_batch();
    b.insert_member::<KeyOfSetTest>(&0, &4);

    // Iterator should not see uncommitted data
    let iter_after =
        db.scan_members::<KeyOfSetTest>(&0).collect::<HashSet<_>>();

    assert_eq!(iter_after.len(), 3);
    assert!(iter_after.contains(&0));
    assert!(iter_after.contains(&1));
    assert!(iter_after.contains(&2));

    b.commit();

    let iter_final =
        db.scan_members::<KeyOfSetTest>(&0).collect::<HashSet<_>>();

    let mut c = db.write_batch();
    c.insert_member::<KeyOfSetTest>(&0, &5);
    c.commit();

    // Should not see 5 as it was added after the iterator was created
    assert_eq!(iter_final.len(), 4);
    assert!(iter_final.contains(&0));
    assert!(iter_final.contains(&1));
    assert!(iter_final.contains(&2));
    assert!(iter_final.contains(&4));
}
