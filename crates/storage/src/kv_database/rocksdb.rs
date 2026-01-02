//! [`RocksDB`] backend implementation for the key-value database abstraction.
//!
//! This module provides a [`RocksDB`] struct that implements the [`KvDatabase`]
//! trait using [`RocksDB`] as the underlying storage engine. It supports
//! dynamic column families, efficient buffer reuse, and full thread safety.

use std::{path::Path, sync::Arc};

use dashmap::DashMap;
use parking_lot::Mutex;
use qbice_serialize::{
    Decoder, Encode, Encoder, Plugin, PostcardDecoder, PostcardEncoder,
};
use qbice_stable_type_id::{Identifiable, StableTypeID};
use rust_rocksdb::{
    BlockBasedOptions, BoundColumnFamily, ColumnFamilyDescriptor,
    DBCompactionStyle, DBCompressionType, DBWithThreadMode, DataBlockIndexType,
    IteratorMode, MultiThreaded, Options, SliceTransform,
};

use crate::kv_database::{
    DiscriminantEncoding, KeyOfSetColumn, KvDatabase, KvDatabaseFactory,
    WideColumn, WideColumnValue, WriteBatch,
};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum ColumnKind {
    WideColumn,
    KeyOfSet,
}

/// A RocksDB-backed key-value database implementation.
///
/// This struct wraps a [`RocksDB`] instance and provides the [`KvDatabase`]
/// trait implementation.
///
/// # Column Family Management
///
/// - Column families are created lazily when first accessed.
/// - Each column type (identified by its [`StableTypeID`]) gets its own column
///   family.
/// - Column family names are derived from the stable type ID for consistency
///   across restarts.
///
/// # Thread Safety
///
/// This implementation is fully thread-safe and can be shared across threads.
/// Internally uses `DBWithThreadMode<MultiThreaded>`.
#[derive(Debug)]
pub struct RocksDB(Arc<Impl>);

#[derive(Debug)]
struct Impl {
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

/// Factory for creating [`RocksDB`] instances.
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
fn configure_rocksdb_for_small_kv_high_writes() -> Options {
    let mut opts = Options::default();

    opts.create_if_missing(true);
    opts.create_missing_column_families(true);
    opts.set_atomic_flush(true);

    // Shared Memtable Budget (e.g., 512MB total for all CFs)
    opts.set_db_write_buffer_size(512 * 1024 * 1024);
    opts.set_max_background_jobs(6);
    // Use LZ4 for speed
    opts.set_compression_type(DBCompressionType::Lz4);

    opts
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
        let opts = configure_rocksdb_for_small_kv_high_writes();

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

        Ok(Self(Arc::new(Impl {
            db,
            plugin,
            column_families: DashMap::new(),
            buffer_pool: BufferPool::new(),
        })))
    }

    /// Creates a factory for opening or creating a `RocksDB` database.
    #[must_use]
    pub const fn factory<P: AsRef<Path>>(path: P) -> RocksDBFactory<P> {
        RocksDBFactory { path }
    }
}

impl Impl {
    /// Generates a column family name from a stable type ID.
    fn cf_name_from_id(id: StableTypeID, kind: ColumnKind) -> String {
        format!(
            "cf_{}_{:#X}",
            match kind {
                ColumnKind::WideColumn => "wide_column",
                ColumnKind::KeyOfSet => "key_of_set",
            },
            id.as_u128()
        )
    }

    fn get_cf_options(kind: ColumnKind) -> Options {
        match kind {
            ColumnKind::WideColumn => Self::get_point_lookup_options(),
            ColumnKind::KeyOfSet => Self::get_key_of_set_options(),
        }
    }

    /// Gets or creates a column family for the given column type.
    fn get_or_create_cf<C: Identifiable>(
        &self,
        kind: ColumnKind,
    ) -> Arc<BoundColumnFamily<'_>> {
        let id = C::STABLE_TYPE_ID;

        // Check if we already have this column family cached
        if let Some(cf_name) = self.column_families.get(&id)
            && let Some(cf) = self.db.cf_handle(&cf_name)
        {
            return cf;
        }

        // Create the column family if it doesn't exist
        let cf_name = Self::cf_name_from_id(id, kind);

        // Try to get existing CF first
        if let Some(cf) = self.db.cf_handle(&cf_name) {
            self.column_families.insert(id, cf_name);
            return cf;
        }

        match self.column_families.entry(id) {
            dashmap::Entry::Occupied(occupied_entry) => {
                self.db.cf_handle(occupied_entry.get()).unwrap_or_else(|| {
                    self.db
                        .create_cf(&cf_name, &Self::get_cf_options(kind))
                        .expect("failed to create column family");

                    self.db
                        .cf_handle(&cf_name)
                        .expect("column family should exist after creation")
                })
            }

            dashmap::Entry::Vacant(vacant_entry) => {
                if let Some(cf) = self.db.cf_handle(&cf_name) {
                    vacant_entry.insert(cf_name);
                    cf
                } else {
                    // proceed to create new column family
                    self.db
                        .create_cf(&cf_name, &Self::get_cf_options(kind))
                        .expect("failed to create column family");

                    vacant_entry.insert(cf_name.clone());
                    self.db
                        .cf_handle(&cf_name)
                        .expect("column family should exist after creation")
                }
            }
        }
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

    fn prefix_upper_bound(prefix: &[u8]) -> Vec<u8> {
        let mut upper_bound = prefix.to_vec();

        for i in (0..upper_bound.len()).rev() {
            if upper_bound[i] < 0xFF {
                upper_bound[i] += 1;
                upper_bound.truncate(i + 1);
                return upper_bound;
            }
        }

        // If all bytes are 0xFF, return an empty vector which indicates no
        // upper bound
        Vec::new()
    }

    #[allow(clippy::cast_possible_truncation)]
    fn transform_key(key: &[u8]) -> &[u8] {
        // our length prefix is u64 (8 bytes)
        if key.len() < 8 {
            return key;
        }

        let length = u64::from_le_bytes(
            key[0..8].try_into().expect("length prefix should be 8 bytes"),
        ) as usize;

        if key.len() < 8 + length {
            return key;
        }

        &key[0..8 + length]
    }

    fn create_key_of_set_prefix_extractor() -> SliceTransform {
        let name = "rocksdb.length_prefixed_slice_extractor";

        let in_domain = |key: &[u8]| -> bool { key.len() >= 8 };

        SliceTransform::create(name, Self::transform_key, Some(in_domain))
    }

    fn encode_wide_column_key<W: WideColumn, C: WideColumnValue<W>>(
        &self,
        key: &W::Key,
        buffer: &mut Vec<u8>,
    ) {
        if W::discriminant_encoding() == DiscriminantEncoding::Prefixed {
            self.encode_value(&C::discriminant(), buffer);
        }

        self.encode_value(key, buffer);

        if W::discriminant_encoding() == DiscriminantEncoding::Suffixed {
            self.encode_value(&C::discriminant(), buffer);
        }
    }

    fn get_point_lookup_options() -> Options {
        let mut opts = Options::default();
        opts.set_compaction_style(DBCompactionStyle::Universal);

        // Low latency block settings
        let mut table_opts = BlockBasedOptions::default();
        table_opts.set_block_size(4 * 1024); // 4KB: Minimize unnecessary data load
        table_opts.set_data_block_index_type(DataBlockIndexType::BinaryAndHash); // O(1) search inside block
        table_opts.set_data_block_hash_ratio(0.75);

        // Standard Bloom Filter
        table_opts.set_bloom_filter(10.0, false);
        table_opts.set_whole_key_filtering(true);

        // Cache & Index
        table_opts.set_cache_index_and_filter_blocks(true);
        table_opts.set_pin_l0_filter_and_index_blocks_in_cache(true);
        table_opts.set_format_version(5);

        opts.set_block_based_table_factory(&table_opts);
        opts
    }

    fn get_key_of_set_options() -> Options {
        let mut opts = Options::default();
        opts.set_compaction_style(DBCompactionStyle::Universal);

        let mut table_opts = BlockBasedOptions::default();

        // 8KB: Better compression for repetitive prefixes (the <K> part)
        // Since values are 0-byte, 8KB still loads very fast.
        table_opts.set_block_size(8 * 1024);

        // Increase restart interval to compress prefixes more aggressively
        table_opts.set_block_restart_interval(32);

        // Hash index is STILL vital for membership test speed
        table_opts.set_data_block_index_type(DataBlockIndexType::BinaryAndHash);
        table_opts.set_data_block_hash_ratio(0.75);

        // Standard Bloom Filter
        table_opts.set_bloom_filter(10.0, false);
        table_opts.set_whole_key_filtering(true);

        table_opts.set_cache_index_and_filter_blocks(true);
        table_opts.set_pin_l0_filter_and_index_blocks_in_cache(true);
        table_opts.set_format_version(5);

        opts.set_block_based_table_factory(&table_opts);

        opts.set_prefix_extractor(Self::create_key_of_set_prefix_extractor());

        opts
    }
}

/// Write transaction for the [`RocksDB`] backend.
///
/// Batches multiple write operations and commits them atomically when
/// [`WriteTransaction::commit`] is called.
///
/// [`WriteTransaction::commit`]: crate::kv_database::WriteBatch::commit
pub struct RocksDBWriteTransaction {
    /// Reference to the parent database.
    db: Arc<Impl>,

    /// The write batch accumulating all operations.
    batch: Mutex<rust_rocksdb::WriteBatch>,
}

impl std::fmt::Debug for RocksDBWriteTransaction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RocksDBWriteTransaction")
            .field("db", &self.db)
            .field("batch", &"<WriteBatch>")
            .finish()
    }
}

impl RocksDBWriteTransaction {
    /// Creates a new write transaction.
    fn new(db: Arc<Impl>) -> Self {
        Self { db, batch: Mutex::new(rust_rocksdb::WriteBatch::default()) }
    }
}

impl WriteBatch for RocksDBWriteTransaction {
    fn put<W: WideColumn, C: WideColumnValue<W>>(
        &self,
        key: &W::Key,
        value: &C,
    ) {
        let cf = self.db.get_or_create_cf::<W>(ColumnKind::WideColumn);

        let mut key_buffer = self.db.buffer_pool.get_buffer();
        let mut value_buffer = self.db.buffer_pool.get_buffer();

        self.db.encode_wide_column_key::<W, C>(key, &mut key_buffer);
        self.db.encode_value(value, &mut value_buffer);

        self.batch.lock().put_cf(
            &cf,
            key_buffer.as_slice(),
            value_buffer.as_slice(),
        );

        self.db.buffer_pool.return_buffer(key_buffer);
        self.db.buffer_pool.return_buffer(value_buffer);
    }

    fn delete<W: WideColumn, C: WideColumnValue<W>>(&self, key: &W::Key) {
        let cf = self.db.get_or_create_cf::<W>(ColumnKind::WideColumn);

        let mut key_buffer = self.db.buffer_pool.get_buffer();

        self.db.encode_wide_column_key::<W, C>(key, &mut key_buffer);

        self.batch.lock().delete_cf(&cf, key_buffer.as_slice());

        self.db.buffer_pool.return_buffer(key_buffer);
    }

    fn insert_member<C: KeyOfSetColumn>(
        &self,
        key: &C::Key,
        value: &C::Element,
    ) {
        let cf = self.db.get_or_create_cf::<C>(ColumnKind::KeyOfSet);
        let mut buffer = self.db.buffer_pool.get_buffer();

        self.db.encode_value_length_prefixed(key, &mut buffer);
        self.db.encode_value(value, &mut buffer);

        // For set membership, the value is empty (presence indicates
        // membership)
        self.batch.lock().put_cf(&cf, buffer.as_slice(), []);

        self.db.buffer_pool.return_buffer(buffer);
    }

    fn delete_member<C: KeyOfSetColumn>(
        &self,
        key: &C::Key,
        value: &C::Element,
    ) {
        let cf = self.db.get_or_create_cf::<C>(ColumnKind::KeyOfSet);
        let mut buffer = self.db.buffer_pool.get_buffer();

        self.db.encode_value_length_prefixed(key, &mut buffer);
        self.db.encode_value(value, &mut buffer);

        self.batch.lock().delete_cf(&cf, buffer.as_slice());

        self.db.buffer_pool.return_buffer(buffer);
    }

    fn commit(self) {
        let batch = self.batch.into_inner();

        let mut write_opts = rust_rocksdb::WriteOptions::default();
        write_opts.disable_wal(true);

        self.db
            .db
            .write_opt(&batch, &write_opts)
            .expect("write should not fail");
    }
}

impl KvDatabase for RocksDB {
    type WriteBatch = RocksDBWriteTransaction;

    fn get_wide_column<W: WideColumn, C: WideColumnValue<W>>(
        &self,
        key: &W::Key,
    ) -> Option<C> {
        let cf = self.0.get_or_create_cf::<W>(ColumnKind::WideColumn);

        let mut buffer = self.0.buffer_pool.get_buffer();
        self.0.encode_wide_column_key::<W, C>(key, &mut buffer);

        match self.0.db.get_cf(&cf, &buffer) {
            Ok(Some(value_bytes)) => {
                let mut decoder =
                    PostcardDecoder::new(std::io::Cursor::new(value_bytes));

                let value = decoder
                    .decode::<C>(&self.0.plugin)
                    .expect("decoding should not fail");

                self.0.buffer_pool.return_buffer(buffer);
                Some(value)
            }
            Ok(None) => {
                self.0.buffer_pool.return_buffer(buffer);
                None
            }
            Err(err) => {
                panic!("RocksDB get error: {err}");
            }
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn scan_members<'s, C: KeyOfSetColumn>(
        &'s self,
        key: &'s C::Key,
    ) -> impl Iterator<Item = C::Element> {
        let cf = self.0.get_or_create_cf::<C>(ColumnKind::KeyOfSet);
        let mut prefix_buffer = self.0.buffer_pool.get_buffer();
        self.0.encode_value_length_prefixed(key, &mut prefix_buffer);

        let prefix_upper_bound =
            Impl::prefix_upper_bound(prefix_buffer.as_slice());

        // Use an iterator with prefix seek
        let mut read_opts = rust_rocksdb::ReadOptions::default();
        read_opts.set_iterate_upper_bound(prefix_upper_bound);

        let iter = self.0.db.iterator_cf_opt(
            &cf,
            read_opts,
            IteratorMode::From(
                prefix_buffer.as_slice(),
                rust_rocksdb::Direction::Forward,
            ),
        );

        // Filter to only include keys that actually start with our prefix.
        // RocksDB's prefix_iterator doesn't guarantee this - it just starts
        // at the prefix position and continues iterating.
        iter.map(|x| x.expect("RocksDB iteration error")).map(|(key, _)| {
            let length = u64::from_le_bytes(
                key[0..8].try_into().expect("length prefix should be 8 bytes"),
            ) as usize;

            let mut decoder =
                PostcardDecoder::new(std::io::Cursor::new(&key[8 + length..]));

            decoder
                .decode::<C::Element>(&self.0.plugin)
                .expect("decoding should not fail")
        })
    }

    fn write_batch(&self) -> Self::WriteBatch {
        RocksDBWriteTransaction::new(self.0.clone())
    }
}
