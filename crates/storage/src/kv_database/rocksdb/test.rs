use std::collections::HashSet;

use qbice_stable_type_id::Identifiable;

use super::*;
use crate::kv_database::{KeyOfSet, Normal};

/// Test column for normal key-value storage.
#[derive(Debug, Clone, Identifiable)]
struct TestColumn;

impl Column for TestColumn {
    type Key = String;
    type Value = u64;
    type Mode = Normal;
}

/// Test column for set storage.
#[derive(Debug, Clone, Identifiable)]
struct TestSetColumn;

impl Column for TestSetColumn {
    type Key = String;
    type Value = HashSet<u32>;
    type Mode = KeyOfSet<u32>;
}

#[tokio::test]
async fn basic_put_get() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db = RocksDB::open(temp_dir.path(), Plugin::new()).unwrap();

    let tx = db.write_transaction();
    tx.put::<TestColumn>(&"key1".to_string(), &42);
    tx.put::<TestColumn>(&"key2".to_string(), &100);
    tx.commit();

    assert_eq!(db.get::<TestColumn>(&"key1".to_string()).await, Some(42));
    assert_eq!(db.get::<TestColumn>(&"key2".to_string()).await, Some(100));
    assert_eq!(db.get::<TestColumn>(&"key3".to_string()).await, None);
}

#[tokio::test]
async fn set_operations() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db = RocksDB::open(temp_dir.path(), Plugin::new()).unwrap();

    let tx = db.write_transaction();
    tx.insert_member::<u32, TestSetColumn>(&"set1".to_string(), &1);
    tx.insert_member::<u32, TestSetColumn>(&"set1".to_string(), &2);
    tx.insert_member::<u32, TestSetColumn>(&"set1".to_string(), &3);
    tx.insert_member::<u32, TestSetColumn>(&"set2".to_string(), &10);
    tx.commit();

    let members =
        db.collect_key_of_set::<u32, TestSetColumn>(&"set1".to_string()).await;

    assert_eq!(members.len(), 3);
    assert!(members.contains(&1));
    assert!(members.contains(&2));
    assert!(members.contains(&3));

    let members2 =
        db.collect_key_of_set::<u32, TestSetColumn>(&"set2".to_string()).await;

    assert_eq!(members2.len(), 1);
    assert!(members2.contains(&10));
}

#[tokio::test]
async fn persistence() {
    let temp_dir = tempfile::tempdir().unwrap();

    // Write data
    {
        let db = RocksDB::open(temp_dir.path(), Plugin::new()).unwrap();
        let tx = db.write_transaction();
        tx.put::<TestColumn>(&"persistent".to_string(), &999);
        tx.commit();
    }

    // Read data after reopening
    {
        let db = RocksDB::open(temp_dir.path(), Plugin::new()).unwrap();
        assert_eq!(
            db.get::<TestColumn>(&"persistent".to_string()).await,
            Some(999)
        );
    }
}

#[tokio::test]
async fn read_committed() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db = RocksDB::open(temp_dir.path(), Plugin::new()).unwrap();

    {
        let tx = db.write_transaction();
        tx.put::<TestColumn>(&"key".to_string(), &123);
        tx.commit();
    }

    assert_eq!(db.get::<TestColumn>(&"key".to_string()).await, Some(123));

    {
        let tx = db.write_transaction();
        tx.put::<TestColumn>(&"key".to_string(), &456);

        // Before commit, the old value should still be visible
        assert_eq!(db.get::<TestColumn>(&"key".to_string()).await, Some(123));

        tx.commit();
    }

    // After commit, the new value should be visible
    assert_eq!(db.get::<TestColumn>(&"key".to_string()).await, Some(456));
}

#[tokio::test]
async fn last_commit_wins() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db = RocksDB::open(temp_dir.path(), Plugin::new()).unwrap();

    {
        let tx1 = db.write_transaction();
        tx1.put::<TestColumn>(&"conflict".to_string(), &1);

        let tx2 = db.write_transaction();
        tx2.put::<TestColumn>(&"conflict".to_string(), &2);

        tx1.commit();
        tx2.commit();
    }

    // The last committed value should be visible
    assert_eq!(db.get::<TestColumn>(&"conflict".to_string()).await, Some(2));
}
