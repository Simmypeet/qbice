//! [`RocksDB`] backend implementation for the key-value database abstraction.
//!
//! This module provides a [`RocksDB`] struct that implements the [`KvDatabase`]
//! trait using [`RocksDB`] as the underlying storage engine. It supports
//! dynamic column families, efficient buffer reuse, and full thread safety.

use std::{ops::Range, path::Path, sync::Arc};

use dashmap::DashMap;
use ouroboros::self_referencing;
use qbice_serialize::{
    Decoder, Encode, Encoder, Plugin, PostcardDecoder, PostcardEncoder,
};
use qbice_stable_type_id::{Identifiable, StableTypeID};
use rust_rocksdb::{
    BlockBasedOptions, BoundColumnFamily, Cache, ColumnFamilyDescriptor,
    DBCompactionStyle, DBCompressionType, DBWithThreadMode, DataBlockIndexType,
    IteratorMode, MultiThreaded, Options, SliceTransform,
};

use crate::{
    kv_database::{
        DiscriminantEncoding, KeyOfSetColumn, KvDatabase, KvDatabaseFactory,
        SerializationBuffer, WideColumn, WideColumnValue, WriteBatch,
    },
    sharded::default_shard_amount,
};

/// Alias for RocksDB error type.
pub type RocksDBError = rust_rocksdb::Error;

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
#[derive(Debug, Clone)]
pub struct RocksDB(Arc<Impl>);

struct Impl {
    /// The underlying `RocksDB` instance.
    db: DBWithThreadMode<MultiThreaded>,

    /// Serialization plugin used for encoding/decoding keys and values.
    plugin: Arc<Plugin>,

    /// Cache mapping stable type IDs to column family names.
    ///
    /// This is used to avoid repeated lookups for the same column family.
    column_families: DashMap<StableTypeID, String>,

    /// The block cache shared by every column family.
    block_cache: Cache,
}

impl std::fmt::Debug for Impl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Impl")
            .field("db", &"<DBWithThreadMode>")
            .field("plugin", &self.plugin)
            .field("column_families", &self.column_families)
            .finish_non_exhaustive()
    }
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

/// Capacity of the block cache shared by all column families.
///
/// Without a shared cache every column family gets a private 32MB cache, so
/// this is roughly the same memory budget, but pooled.
const BLOCK_CACHE_CAPACITY: usize = 256 * 1024 * 1024;

/// Number of background threads `RocksDB` may use for flushes and compactions.
fn background_jobs() -> i32 {
    let cores = std::thread::available_parallelism().map_or(4, usize::from);

    i32::try_from(cores.clamp(2, 8)).unwrap_or(8)
}

fn configure_rocksdb_for_small_kv_high_writes() -> Options {
    let mut opts = Options::default();

    opts.create_if_missing(true);
    opts.create_missing_column_families(true);
    opts.set_atomic_flush(true);
    opts.set_compression_type(DBCompressionType::Lz4);

    // Shared Memtable Budget (e.g., 512MB total for all CFs)
    opts.set_db_write_buffer_size(512 * 1024 * 1024);

    // Flushes and compactions otherwise share two background threads.
    opts.increase_parallelism(background_jobs());

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
        let block_cache = Cache::new_hyper_clock_cache(BLOCK_CACHE_CAPACITY, 0);

        // List existing column families
        let existing_cfs =
            DBWithThreadMode::<MultiThreaded>::list_cf(&opts, &path)
                .unwrap_or_default();

        // Create column family descriptors for existing families
        let cf_descriptors: Vec<_> = existing_cfs
            .iter()
            .map(|name| {
                let options = if name.contains("wide_column") {
                    Impl::get_point_lookup_options(&block_cache)
                } else if name.contains("key_of_set") {
                    Impl::get_key_of_set_options(&block_cache)
                } else {
                    configure_rocksdb_for_small_kv_high_writes()
                };

                ColumnFamilyDescriptor::new(name, options)
            })
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
            plugin: Arc::new(plugin),
            column_families: DashMap::with_shard_amount(default_shard_amount()),
            block_cache,
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

    fn get_cf_options(&self, kind: ColumnKind) -> Options {
        match kind {
            ColumnKind::WideColumn => {
                Self::get_point_lookup_options(&self.block_cache)
            }
            ColumnKind::KeyOfSet => {
                Self::get_key_of_set_options(&self.block_cache)
            }
        }
    }

    fn get_or_create_cf_from_cf_identifier(
        &self,
        stable_type_id: StableTypeID,
        kind: ColumnKind,
    ) -> Arc<BoundColumnFamily<'_>> {
        // Check if we already have this column family cached
        if let Some(cf_name) = self.column_families.get(&stable_type_id)
            && let Some(cf) = self.db.cf_handle(&cf_name)
        {
            return cf;
        }

        // Create the column family if it doesn't exist
        let cf_name = Self::cf_name_from_id(stable_type_id, kind);

        // Try to get existing CF first
        if let Some(cf) = self.db.cf_handle(&cf_name) {
            self.column_families.insert(stable_type_id, cf_name);
            return cf;
        }

        match self.column_families.entry(stable_type_id) {
            dashmap::Entry::Occupied(occupied_entry) => {
                self.db.cf_handle(occupied_entry.get()).unwrap_or_else(|| {
                    self.db
                        .create_cf(&cf_name, &self.get_cf_options(kind))
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
                    let Ok(()) =
                        self.db.create_cf(&cf_name, &self.get_cf_options(kind))
                    else {
                        panic!("failed to create column family");
                    };

                    vacant_entry.insert(cf_name.clone());
                    self.db
                        .cf_handle(&cf_name)
                        .expect("column family should exist after creation")
                }
            }
        }
    }

    /// Gets or creates a column family for the given column type.
    fn get_or_create_cf<C: Identifiable>(
        &self,
        kind: ColumnKind,
    ) -> Arc<BoundColumnFamily<'_>> {
        let stable_type_id = C::STABLE_TYPE_ID;
        self.get_or_create_cf_from_cf_identifier(stable_type_id, kind)
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

    fn apply_optimized_opts(opts: &mut Options) {
        // 1. Switch to LZ4 (Faster decompression)
        opts.set_compression_type(DBCompressionType::Lz4);

        // 2. Disable compression for L0 and L1 to save CPU on hot data Levels
        //    2+ use LZ4.
        opts.set_compression_per_level(&[
            DBCompressionType::None, // L0
            DBCompressionType::None, // L1
            DBCompressionType::Lz4,  // L2
            DBCompressionType::Lz4,  // L3
            DBCompressionType::Lz4,  // L4
            DBCompressionType::Lz4,  // L5
            DBCompressionType::Lz4,  // L6
        ]);

        // 3. Bloom filter over the memtable, so that looking up a key that was
        //    never written does not have to search the skiplist.
        opts.set_memtable_whole_key_filtering(true);
        opts.set_memtable_prefix_bloom_ratio(0.02);
    }

    fn get_point_lookup_options(block_cache: &Cache) -> Options {
        let mut opts = Options::default();
        opts.set_compaction_style(DBCompactionStyle::Level);

        Self::apply_optimized_opts(&mut opts);

        // Low latency block settings
        let mut table_opts = BlockBasedOptions::default();
        table_opts.set_block_size(4 * 1024); // 4KB: Minimize unnecessary data load
        table_opts.set_data_block_index_type(DataBlockIndexType::BinaryAndHash); // O(1) search inside block
        table_opts.set_data_block_hash_ratio(0.75);

        // Standard Bloom Filter
        table_opts.set_bloom_filter(10.0, false);
        table_opts.set_whole_key_filtering(true);

        // Cache & Index
        table_opts.set_block_cache(block_cache);
        table_opts.set_cache_index_and_filter_blocks(true);
        table_opts.set_pin_l0_filter_and_index_blocks_in_cache(true);
        table_opts.set_format_version(5);

        opts.set_block_based_table_factory(&table_opts);
        opts
    }

    fn get_key_of_set_options(block_cache: &Cache) -> Options {
        let mut opts = Options::default();
        opts.set_compaction_style(DBCompactionStyle::Level);

        Self::apply_optimized_opts(&mut opts);

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

        table_opts.set_block_cache(block_cache);
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
pub struct RocksDBWriteBatch {
    /// Reference to the parent database.
    db: Arc<Impl>,

    /// The operations added since the batch was last prepared, in the order
    /// they were added: the chunks one after another, and the operations of
    /// a chunk one after another.
    chunks: Vec<Chunk>,

    /// The operations that have been prepared, the way `RocksDB` takes them.
    batch: rust_rocksdb::WriteBatch,

    /// Estimated size of the write batch.
    estimated_size: usize,
}

unsafe impl Sync for RocksDBWriteBatch {}

impl std::fmt::Debug for RocksDBWriteBatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RocksDBWriteTransaction")
            .field("db", &self.db)
            .field("batch", &"<WriteBatch>")
            .field("estimated_size", &self.estimated_size)
            .finish_non_exhaustive()
    }
}

impl RocksDBWriteBatch {
    /// Creates a new write transaction.
    fn new(db: Arc<Impl>) -> Self {
        Self {
            db,
            chunks: Vec::new(),
            batch: rust_rocksdb::WriteBatch::default(),
            estimated_size: 0,
        }
    }

    /// Adds operations to the batch directly. They go to the end of the last
    /// chunk, which is after every operation that the batch already holds.
    fn write(&mut self, write: impl FnOnce(&mut Chunk, &Impl)) {
        if self.chunks.is_empty() {
            self.chunks.push(Chunk::default());
        }

        let chunk = self.chunks.last_mut().expect("has just been checked");
        let size_before = chunk.bytes.len();

        write(chunk, &self.db);

        self.estimated_size += chunk.bytes.len() - size_before;
    }
}

const PREFERRED_WRITE_BATCH_SIZE: usize = 16 * 1024 * 1024; // 16MB

/// Where an operation goes in the order `RocksDB` is given the operations of a
/// write batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Position<'a> {
    /// The index of the column family in the list that comes with the
    /// positions.
    column_family: usize,

    key: &'a [u8],

    /// The index of the operation in the order the operations were added.
    index: usize,

    /// The value to store under the key, or `None` to delete the key. No two
    /// operations have the same index, so this never decides the order.
    value: Option<&'a [u8]>,
}

/// Returns the order in which `RocksDB` is given the operations of a write
/// batch: column family by column family, with ascending keys.
///
/// `RocksDB` inserts the keys of a write batch into a skip list one after
/// another and has to find the place of each of them. A key that directly
/// follows the one before it is put there without a search, and the search
/// for a key that is close to the one before it visits the nodes that the
/// last search has just visited. The keys of the queries are hashes, so in the
/// order they were added every one of them is a search through a part of the
/// list that has not been touched in a while.
///
/// Operations on the same key keep the order they were added in. That is the
/// order `RocksDB` applies them in, so the last one still decides what the key
/// holds.
///
/// Returns the column families that are written to, and the operations with
/// the index of their column family in that list.
fn sorted_positions(
    chunks: &[Chunk],
) -> (Vec<CfIdentifier>, Vec<Position<'_>>) {
    let mut column_families = Vec::new();

    let mut positions = chunks
        .iter()
        .flat_map(|chunk| {
            chunk.operations.iter().map(move |operation| (chunk, operation))
        })
        .enumerate()
        .map(|(index, (chunk, operation))| {
            let column_family = column_families
                .iter()
                .position(|cf| *cf == operation.cf)
                .unwrap_or_else(|| {
                    column_families.push(operation.cf);
                    column_families.len() - 1
                });

            Position {
                column_family,
                key: &chunk.bytes[operation.key.clone()],
                index,
                value: operation.value.clone().map(|value| &chunk.bytes[value]),
            }
        })
        .collect::<Vec<_>>();

    // no two positions are equal, so an unstable sort has one result
    positions.sort_unstable();

    (column_families, positions)
}

impl WriteBatch for RocksDBWriteBatch {
    type SerializationBuffer = RocksDBSerializationBuffer;

    fn put<W: WideColumn, C: WideColumnValue<W>>(
        &mut self,
        key: &W::Key,
        value: &C,
    ) {
        self.write(|chunk, db| chunk.put::<W, C>(db, key, value));
    }

    fn delete<W: WideColumn, C: WideColumnValue<W>>(&mut self, key: &W::Key) {
        self.write(|chunk, db| chunk.delete::<W, C>(db, key));
    }

    fn insert_member<C: KeyOfSetColumn>(
        &mut self,
        key: &C::Key,
        value: &C::Element,
    ) {
        self.write(|chunk, db| chunk.set_member::<C>(db, key, value, true));
    }

    fn delete_member<C: KeyOfSetColumn>(
        &mut self,
        key: &C::Key,
        value: &C::Element,
    ) {
        self.write(|chunk, db| chunk.set_member::<C>(db, key, value, false));
    }

    fn consume_serialization_buffer(
        &mut self,
        buffer: Self::SerializationBuffer,
    ) {
        if buffer.chunk.operations.is_empty() {
            return;
        }

        self.estimated_size += buffer.chunk.bytes.len();
        self.chunks.push(buffer.chunk);
    }

    fn prepare(&mut self) {
        let chunks = std::mem::take(&mut self.chunks);
        let (column_families, positions) = sorted_positions(&chunks);

        for column_family in
            positions.chunk_by(|a, b| a.column_family == b.column_family)
        {
            let cf = column_families[column_family[0].column_family];
            let cf_handle = self.db.get_or_create_cf_from_cf_identifier(
                cf.stable_type_id,
                cf.kind,
            );

            for (index, position) in column_family.iter().enumerate() {
                // An operation that is followed by another one on the same
                // key decides nothing about what the key holds, so only the
                // last of them is written. A query that is computed again
                // removes what it wrote before and writes most of it anew.
                if column_family
                    .get(index + 1)
                    .is_some_and(|next| next.key == position.key)
                {
                    continue;
                }

                match position.value {
                    Some(value) => {
                        self.batch.put_cf(&cf_handle, position.key, value);
                    }
                    None => self.batch.delete_cf(&cf_handle, position.key),
                }
            }
        }
    }

    fn commit(mut self) {
        self.prepare();

        let mut write_opts = rust_rocksdb::WriteOptions::default();
        write_opts.disable_wal(true);

        self.db
            .db
            .write_opt(&self.batch, &write_opts)
            .expect("write should not fail");
    }

    fn should_write_more(&self) -> bool {
        self.estimated_size < PREFERRED_WRITE_BATCH_SIZE
    }
}

impl KvDatabase for RocksDB {
    type WriteBatch = RocksDBWriteBatch;

    type SerializationBuffer = RocksDBSerializationBuffer;

    type ScanMemberIterator<C: KeyOfSetColumn> = ScanMembersIterator<C>;

    fn get_wide_column<W: WideColumn, C: WideColumnValue<W>>(
        &self,
        key: &W::Key,
    ) -> Option<C> {
        let cf = self.0.get_or_create_cf::<W>(ColumnKind::WideColumn);

        let mut buffer = Vec::new();
        self.0.encode_wide_column_key::<W, C>(key, &mut buffer);

        match self.0.db.get_cf(&cf, &buffer) {
            Ok(Some(value_bytes)) => {
                let mut decoder =
                    PostcardDecoder::new(std::io::Cursor::new(value_bytes));

                let value = decoder
                    .decode::<C>(&self.0.plugin)
                    .expect("decoding should not fail");

                Some(value)
            }
            Ok(None) => None,
            Err(err) => {
                panic!("RocksDB get error: {err}");
            }
        }
    }

    fn scan_members<C: KeyOfSetColumn>(
        &self,
        key: &C::Key,
    ) -> ScanMembersIterator<C> {
        let ither =
            OwnedScannedMembersIterator::new(self.0.clone(), move |x| {
                let mut prefix_buffer = Vec::new();
                x.encode_value_length_prefixed(key, &mut prefix_buffer);

                let prefix_upper_bound =
                    Impl::prefix_upper_bound(prefix_buffer.as_slice());

                // Use an iterator with prefix seek
                let mut read_opts = rust_rocksdb::ReadOptions::default();
                read_opts.set_iterate_upper_bound(prefix_upper_bound);
                read_opts.set_verify_checksums(false);

                x.db.iterator_cf_opt(
                    &x.get_or_create_cf::<C>(ColumnKind::KeyOfSet),
                    read_opts,
                    IteratorMode::From(
                        prefix_buffer.as_slice(),
                        rust_rocksdb::Direction::Forward,
                    ),
                )
            });

        let inner = self.0.clone();

        // Filter to only include keys that actually start with our prefix.
        // RocksDB's prefix_iterator doesn't guarantee this - it just starts
        // at the prefix position and continues iterating.
        ScanMembersIterator::<C> {
            inner: ither,
            inner_db: inner,
            _marker: std::marker::PhantomData,
        }
    }

    fn write_batch(&self) -> Self::WriteBatch {
        RocksDBWriteBatch::new(self.0.clone())
    }

    fn serialization_buffer(&self) -> Self::SerializationBuffer {
        RocksDBSerializationBuffer {
            chunk: Chunk::default(),
            db: self.0.clone(),
        }
    }
}

#[self_referencing]
struct OwnedScannedMembersIterator {
    db: Arc<Impl>,

    #[borrows(db)]
    #[not_covariant]
    iterator: rust_rocksdb::DBIteratorWithThreadMode<
        'this,
        DBWithThreadMode<MultiThreaded>,
    >,
}

impl Iterator for OwnedScannedMembersIterator {
    type Item = Result<(Box<[u8]>, Box<[u8]>), rust_rocksdb::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        self.with_iterator_mut(|iter| iter.next())
    }
}

/// Iterator over members of a key-of-set column in RocksDB.
pub struct ScanMembersIterator<C: KeyOfSetColumn> {
    inner: OwnedScannedMembersIterator,
    inner_db: Arc<Impl>,
    _marker: std::marker::PhantomData<C>,
}

impl<C: KeyOfSetColumn> std::fmt::Debug for ScanMembersIterator<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScanMembersIterator")
            .field("inner", &"<OwnedScannedMembersIterator>")
            .field("inner_db", &self.inner_db)
            .finish()
    }
}

impl<C: KeyOfSetColumn> Iterator for ScanMembersIterator<C> {
    type Item = C::Element;

    #[allow(clippy::cast_possible_truncation)]
    fn next(&mut self) -> Option<Self::Item> {
        let (key, _value) = self.inner.next()?.expect("RocksDB iterator error");

        let length = u64::from_le_bytes(
            key[0..8].try_into().expect("length prefix should be 8 bytes"),
        ) as usize;

        let mut decoder =
            PostcardDecoder::new(std::io::Cursor::new(&key[8 + length..]));

        Some(
            decoder
                .decode::<C::Element>(&self.inner_db.plugin)
                .expect("decoding should not fail"),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CfIdentifier {
    stable_type_id: StableTypeID,
    kind: ColumnKind,
}

/// A write to one key of a column family. The key and the value are in the
/// bytes of the [`Chunk`] that the operation belongs to.
#[derive(Debug)]
struct Operation {
    cf: CfIdentifier,

    /// Where the key is.
    key: Range<usize>,

    /// Where the value to store under the key is, or `None` to delete the
    /// key.
    value: Option<Range<usize>>,
}

/// Operations that were serialized one after another, in the order they were
/// added.
///
/// The keys and the values of all of them are in one buffer. An operation
/// with a buffer of its own for its key and another for its value costs two
/// allocations, and more for each time one of them outgrows what it was
/// given, and that for every one of the ten or so operations that a computed
/// query writes.
#[derive(Debug, Default)]
struct Chunk {
    /// The keys and the values of the operations, one after another.
    bytes: Vec<u8>,

    operations: Vec<Operation>,
}

impl Chunk {
    /// Adds the operation that stores a value of a wide column.
    fn put<W: WideColumn, C: WideColumnValue<W>>(
        &mut self,
        db: &Impl,
        key: &W::Key,
        value: &C,
    ) {
        let start = self.bytes.len();
        db.encode_wide_column_key::<W, C>(key, &mut self.bytes);

        let key_end = self.bytes.len();
        db.encode_value(value, &mut self.bytes);

        self.operations.push(Operation {
            cf: CfIdentifier {
                stable_type_id: W::STABLE_TYPE_ID,
                kind: ColumnKind::WideColumn,
            },
            key: start..key_end,
            value: Some(key_end..self.bytes.len()),
        });
    }

    /// Adds the operation that deletes a value of a wide column.
    fn delete<W: WideColumn, C: WideColumnValue<W>>(
        &mut self,
        db: &Impl,
        key: &W::Key,
    ) {
        let start = self.bytes.len();
        db.encode_wide_column_key::<W, C>(key, &mut self.bytes);

        self.operations.push(Operation {
            cf: CfIdentifier {
                stable_type_id: W::STABLE_TYPE_ID,
                kind: ColumnKind::WideColumn,
            },
            key: start..self.bytes.len(),
            value: None,
        });
    }

    /// Adds the operation that makes `element` a member of the set of `key`,
    /// or that makes it not a member anymore.
    fn set_member<C: KeyOfSetColumn>(
        &mut self,
        db: &Impl,
        key: &C::Key,
        element: &C::Element,
        member: bool,
    ) {
        let start = self.bytes.len();
        db.encode_value_length_prefixed(key, &mut self.bytes);
        db.encode_value(element, &mut self.bytes);

        let end = self.bytes.len();

        self.operations.push(Operation {
            cf: CfIdentifier {
                stable_type_id: C::STABLE_TYPE_ID,
                kind: ColumnKind::KeyOfSet,
            },
            key: start..end,
            // For set membership, the value is empty (presence indicates
            // membership)
            value: member.then_some(end..end),
        });
    }
}

/// Serialization buffer for batching RocksDB operations.
#[derive(Debug)]
pub struct RocksDBSerializationBuffer {
    chunk: Chunk,

    db: Arc<Impl>,
}

impl SerializationBuffer for RocksDBSerializationBuffer {
    fn put<W: WideColumn, C: WideColumnValue<W>>(
        &mut self,
        key: &W::Key,
        value: &C,
    ) {
        self.chunk.put::<W, C>(&self.db, key, value);
    }

    fn delete<W: WideColumn, C: WideColumnValue<W>>(&mut self, key: &W::Key) {
        self.chunk.delete::<W, C>(&self.db, key);
    }

    fn insert_member<C: KeyOfSetColumn>(
        &mut self,
        key: &C::Key,
        value: &C::Element,
    ) {
        self.chunk.set_member::<C>(&self.db, key, value, true);
    }

    fn delete_member<C: KeyOfSetColumn>(
        &mut self,
        key: &C::Key,
        value: &C::Element,
    ) {
        self.chunk.set_member::<C>(&self.db, key, value, false);
    }

    fn size(&self) -> usize { self.chunk.bytes.len() }
}

#[cfg(test)]
mod test;
