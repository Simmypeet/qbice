//! `RocksDB` backend implementation for the KV database trait.
//!
//! This module provides a [`RocksDB`] struct that implements the [`KvDatabase`]
//! trait using `RocksDB` as the underlying storage engine.

use std::{path::Path, sync::Arc};

use dashmap::DashMap;
use futures::{Stream, stream};
use parking_lot::Mutex;
use qbice_serialize::{
    Decoder, Encode, Encoder, Plugin, PostcardDecoder, PostcardEncoder,
};
use qbice_stable_type_id::StableTypeID;
use rust_rocksdb::{
    BoundColumnFamily, ColumnFamilyDescriptor, DBWithThreadMode, MultiThreaded,
    Options, WriteBatch,
};

use super::{Column, KeyOfSet, KvDatabase, Normal, WriteTransaction};
use crate::kv_database::KvDatabaseFactory;

#[derive(Debug)]
struct BufferPool {
    pool: Mutex<Vec<Vec<u8>>>,
}

impl BufferPool {
    const fn new() -> Self { Self { pool: Mutex::new(Vec::new()) } }

    fn get_buffer(&self) -> Vec<u8> {
        self.pool.lock().pop().unwrap_or_else(|| Vec::with_capacity(1024))
    }

    fn return_buffer(&self, mut buffer: Vec<u8>) {
        // clear the buffer content but keep the allocated capacity
        buffer.clear();

        self.pool.lock().push(buffer);
    }
}

/// A `RocksDB`-backed key-value database implementation.
///
/// This struct wraps a `RocksDB` instance and provides the [`KvDatabase`] trait
/// implementation. It manages column families dynamically based on the
/// [`Column`] types used.
///
/// # Column Family Management
///
/// Column families are created lazily when first accessed. Each column type
/// (identified by its [`StableTypeID`]) gets its own column family. The column
/// family name is derived from the stable type ID to ensure consistency across
/// restarts.
///
/// # Thread Safety
///
/// This implementation is fully thread-safe and can be shared across multiple
/// threads. `RocksDB`'s `DBWithThreadMode<MultiThreaded>` is used internally.
#[derive(Debug)]
pub struct RocksDB {
    /// The underlying `RocksDB` instance.
    db: DBWithThreadMode<MultiThreaded>,

    /// Serialization plugin used for encoding/decoding keys and values.
    plugin: Plugin,

    /// Cache mapping stable type IDs to column family names.
    ///
    /// This is used to avoid repeated lookups for the same column family.
    column_families: DashMap<StableTypeID, String>,

    /// Buffer pool for reusing encoding/decoding buffers.
    buffer_pool: BufferPool,
}

/// A factory for creating `RocksDB` instances.
#[derive(Debug)]
pub struct RocksDBFactory<P> {
    path: P,
}

impl<P: AsRef<Path>> KvDatabaseFactory for RocksDBFactory<P> {
    type KvDatabase = RocksDB;

    type Error = rust_rocksdb::Error;

    fn open(
        self,
        serialization_plugin: Plugin,
    ) -> Result<Self::KvDatabase, Self::Error> {
        RocksDB::open(self.path, serialization_plugin)
    }
}

impl RocksDB {
    /// Opens or creates a `RocksDB` database at the specified path.
    ///
    /// # Arguments
    ///
    /// * `path` - The filesystem path where the database will be stored.
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be opened or created.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use qbice_storage::kv_database::rocksdb::RocksDB;
    ///
    /// let db = RocksDB::open("/tmp/my_database").unwrap();
    /// ```
    pub fn open<P: AsRef<Path>>(
        path: P,
        plugin: Plugin,
    ) -> Result<Self, rust_rocksdb::Error> {
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);

        // List existing column families
        let existing_cfs =
            DBWithThreadMode::<MultiThreaded>::list_cf(&opts, &path)
                .unwrap_or_default();

        // Create column family descriptors for existing families
        let cf_descriptors: Vec<_> = existing_cfs
            .iter()
            .map(|name| ColumnFamilyDescriptor::new(name, Options::default()))
            .collect();

        let db = if cf_descriptors.is_empty() {
            DBWithThreadMode::<MultiThreaded>::open(&opts, path)?
        } else {
            DBWithThreadMode::<MultiThreaded>::open_cf_descriptors(
                &opts,
                path,
                cf_descriptors,
            )?
        };

        Ok(Self {
            db,
            plugin,
            column_families: DashMap::new(),
            buffer_pool: BufferPool::new(),
        })
    }

    /// Generates a column family name from a stable type ID.
    fn cf_name_from_id(id: StableTypeID) -> String {
        format!("cf_{:032x}", id.as_u128())
    }

    /// Gets or creates a column family for the given column type.
    fn get_or_create_cf<C: Column>(&self) -> Arc<BoundColumnFamily<'_>> {
        let id = C::STABLE_TYPE_ID;

        // Check if we already have this column family cached
        if let Some(cf_name) = self.column_families.get(&id)
            && let Some(cf) = self.db.cf_handle(&cf_name)
        {
            return cf;
        }

        // Create the column family if it doesn't exist
        let cf_name = Self::cf_name_from_id(id);

        // Try to get existing CF first
        if let Some(cf) = self.db.cf_handle(&cf_name) {
            self.column_families.insert(id, cf_name);
            return cf;
        }

        // Create new column family
        self.db
            .create_cf(&cf_name, &Options::default())
            .expect("failed to create column family");

        self.column_families.insert(id, cf_name.clone());
        self.db
            .cf_handle(&cf_name)
            .expect("column family should exist after creation")
    }

    /// Encodes a key using the postcard format.
    fn encode_value<K: Encode>(&self, key: &K, buffer: &mut Vec<u8>) {
        let mut encoder = PostcardEncoder::new(buffer);

        encoder.encode(key, &self.plugin).expect("encoding should not fail");
    }

    fn encode_value_length_prefixed<K: Encode>(
        &self,
        key: &K,
        buffer: &mut Vec<u8>,
    ) {
        let start_len = buffer.len();

        // reserve space for length prefix
        buffer.extend_from_slice(&0u64.to_le_bytes());

        let mut encoder = PostcardEncoder::new(&mut *buffer);
        encoder.encode(key, &self.plugin).expect("encoding should not fail");

        let end_len = buffer.len();
        // minus the length prefix size
        let value_len = (end_len - start_len - 8) as u64;

        // write the length prefix
        buffer[start_len..start_len + 8]
            .copy_from_slice(&value_len.to_le_bytes());
    }
}

/// A write transaction for the `RocksDB` backend.
///
/// This struct batches multiple write operations and commits them atomically
/// when [`WriteTransaction::commit`] is called.
pub struct RocksDBWriteTransaction<'a> {
    /// Reference to the parent database.
    db: &'a RocksDB,

    /// The write batch accumulating all operations.
    batch: Mutex<WriteBatch>,
}

impl std::fmt::Debug for RocksDBWriteTransaction<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RocksDBWriteTransaction")
            .field("db", &self.db)
            .field("batch", &"<WriteBatch>")
            .finish()
    }
}

impl<'a> RocksDBWriteTransaction<'a> {
    /// Creates a new write transaction.
    fn new(db: &'a RocksDB) -> Self {
        Self { db, batch: Mutex::new(WriteBatch::default()) }
    }
}

impl WriteTransaction for RocksDBWriteTransaction<'_> {
    fn put<C: Column<Mode = Normal>>(
        &self,
        key: &<C as Column>::Key,
        value: &<C as Column>::Value,
    ) {
        let cf = self.db.get_or_create_cf::<C>();

        let mut key_buffer = self.db.buffer_pool.get_buffer();
        let mut value_buffer = self.db.buffer_pool.get_buffer();

        self.db.encode_value(key, &mut key_buffer);
        self.db.encode_value(value, &mut value_buffer);

        self.batch.lock().put_cf(
            &cf,
            key_buffer.as_slice(),
            value_buffer.as_slice(),
        );

        self.db.buffer_pool.return_buffer(key_buffer);
        self.db.buffer_pool.return_buffer(value_buffer);
    }

    fn insert_member<C: Column<Mode = KeyOfSet>>(
        &self,
        key: &<C as Column>::Key,
        value: &<C as Column>::Value,
    ) {
        let cf = self.db.get_or_create_cf::<C>();
        let mut buffer = self.db.buffer_pool.get_buffer();

        self.db.encode_value_length_prefixed(key, &mut buffer);
        self.db.encode_value(value, &mut buffer);

        // For set membership, the value is empty (presence indicates
        // membership)
        self.batch.lock().put_cf(&cf, buffer.as_slice(), []);

        self.db.buffer_pool.return_buffer(buffer);
    }

    fn commit(self) {
        let batch = self.batch.into_inner();
        self.db.db.write(&batch).expect("write should not fail");
    }
}

impl KvDatabase for RocksDB {
    type WriteTransaction<'a> = RocksDBWriteTransaction<'a>;

    async fn get<C: Column<Mode = Normal>>(
        &self,
        key: &C::Key,
    ) -> Option<<C as Column>::Value> {
        let cf = self.get_or_create_cf::<C>();

        let mut buffer = self.buffer_pool.get_buffer();
        self.encode_value(key, &mut buffer);

        match self.db.get_cf(&cf, &buffer) {
            Ok(Some(value_bytes)) => {
                let mut decoder =
                    PostcardDecoder::new(std::io::Cursor::new(value_bytes));

                let value = decoder
                    .decode::<C::Value>(&self.plugin)
                    .expect("decoding should not fail");

                self.buffer_pool.return_buffer(buffer);
                Some(value)
            }
            Ok(None) => {
                self.buffer_pool.return_buffer(buffer);
                None
            }
            Err(err) => {
                panic!("RocksDB get error: {err}");
            }
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn scan_members<'s, C: Column<Mode = KeyOfSet>>(
        &'s self,
        key: &'s C::Key,
    ) -> impl Stream<Item = <C as Column>::Value> + Send {
        let cf = self.get_or_create_cf::<C>();
        let mut prefix_buffer = self.buffer_pool.get_buffer();
        self.encode_value_length_prefixed(key, &mut prefix_buffer);

        // Use an iterator with prefix seek
        let iter = self.db.prefix_iterator_cf(&cf, &prefix_buffer);
        let iter = iter.map(|x| {
            let (key, _) = x.expect("RocksDB iteration error");
            let length = u64::from_le_bytes(
                key[0..8].try_into().expect("length prefix should be 8 bytes"),
            ) as usize;

            let mut decoder =
                PostcardDecoder::new(std::io::Cursor::new(&key[8 + length..]));

            decoder
                .decode::<C::Value>(&self.plugin)
                .expect("decoding should not fail")
        });

        stream::iter(iter)
    }

    fn write_transaction(&self) -> Self::WriteTransaction<'_> {
        RocksDBWriteTransaction::new(self)
    }
}

#[cfg(test)]
mod tests {
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
        type Value = u32;
        type Mode = KeyOfSet;
    }

    #[tokio::test]
    async fn test_basic_put_get() {
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
    async fn test_set_operations() {
        use futures::StreamExt;

        let temp_dir = tempfile::tempdir().unwrap();
        let db = RocksDB::open(temp_dir.path(), Plugin::new()).unwrap();

        let tx = db.write_transaction();
        tx.insert_member::<TestSetColumn>(&"set1".to_string(), &1);
        tx.insert_member::<TestSetColumn>(&"set1".to_string(), &2);
        tx.insert_member::<TestSetColumn>(&"set1".to_string(), &3);
        tx.insert_member::<TestSetColumn>(&"set2".to_string(), &10);
        tx.commit();

        let members: Vec<u32> = db
            .scan_members::<TestSetColumn>(&"set1".to_string())
            .collect()
            .await;

        assert_eq!(members.len(), 3);
        assert!(members.contains(&1));
        assert!(members.contains(&2));
        assert!(members.contains(&3));

        let members2: Vec<u32> = db
            .scan_members::<TestSetColumn>(&"set2".to_string())
            .collect()
            .await;

        assert_eq!(members2, vec![10]);
    }

    #[tokio::test]
    async fn test_persistence() {
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
}
