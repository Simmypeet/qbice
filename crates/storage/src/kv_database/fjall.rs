//! [`Fjall`] backend implementation for the key-value database abstraction.
//!
//! This module provides a [`Fjall`] struct that implements the [`KvDatabase`]
//! trait using [`Fjall`] as the underlying storage engine.

use std::{path::Path, sync::Arc};

use dashmap::{DashMap, Entry};
use fjall::Keyspace;
use qbice_serialize::{
    Decoder, Encode, Encoder, Plugin, PostcardDecoder, PostcardEncoder,
};
use qbice_stable_type_id::{Identifiable, StableTypeID};

use crate::kv_database::{
    DiscriminantEncoding, KeyOfSetColumn, KvDatabase, KvDatabaseFactory,
    SerializationBuffer, WideColumn, WideColumnValue, WriteBatch,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum ColumnKind {
    WideColumn,
    KeyOfSet,
}

/// A Fjall-backed key-value database implementation.
///
/// This struct wraps a [`fjall::Database`] instance and provides the
/// [`KvDatabase`] trait implementation.
///
/// # Keyspace Management
///
/// - Keyspaces (partitions) are created lazily when first accessed.
/// - Each column type (identified by its [`StableTypeID`]) gets its own
///   keyspace.
/// - Keyspace names are derived from the stable type ID for consistency across
///   restarts.
///
/// # Thread Safety
///
/// This implementation is fully thread-safe and can be shared across threads.
#[derive(Debug, Clone)]
pub struct Fjall(Arc<Impl>);

struct Impl {
    /// The underlying `Fjall` database instance.
    db: fjall::Database,

    /// Serialization plugin used for encoding/decoding keys and values.
    plugin: Plugin,

    /// Cache mapping stable type IDs to keyspace names.
    ///
    /// This is used to avoid repeated lookups for the same keyspace.
    keyspaces: DashMap<StableTypeID, Keyspace>,
}

impl std::fmt::Debug for Impl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Impl")
            .field("db", &"<Database>")
            .field("plugin", &self.plugin)
            .finish_non_exhaustive()
    }
}

/// Factory for creating [`Fjall`] instances.
#[derive(Debug)]
pub struct FjallFactory<P> {
    path: P,
}

/// Error type for Fjall operations.
pub type FjallError = fjall::Error;

impl<P: AsRef<Path>> KvDatabaseFactory for FjallFactory<P> {
    type KvDatabase = Fjall;

    type Error = fjall::Error;

    fn open(
        self,
        serialization_plugin: Plugin,
    ) -> Result<Self::KvDatabase, Self::Error> {
        Fjall::open(self.path, serialization_plugin)
    }
}

fn configure_fjall_for_small_kv_high_writes<P: AsRef<Path>>(
    path: P,
) -> fjall::Config {
    fjall::Config::new(path.as_ref())
}

impl Fjall {
    /// Opens or creates a `Fjall` database at the specified path.
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
    /// use qbice_storage::kv_database::fjall::Fjall;
    ///
    /// let db = Fjall::open("/tmp/my_database").unwrap();
    /// ```
    pub fn open<P: AsRef<Path>>(
        path: P,
        plugin: Plugin,
    ) -> Result<Self, fjall::Error> {
        let config = configure_fjall_for_small_kv_high_writes(&path);

        let db = fjall::Database::open(config)?;

        Ok(Self(Arc::new(Impl { db, plugin, keyspaces: DashMap::new() })))
    }

    /// Creates a factory for opening or creating a `Fjall` database.
    #[must_use]
    pub const fn factory<P: AsRef<Path>>(path: P) -> FjallFactory<P> {
        FjallFactory { path }
    }
}

impl Impl {
    /// Generates a keyspace name from a stable type ID.
    fn keyspace_name_from_id(id: StableTypeID, kind: ColumnKind) -> String {
        format!(
            "ks_{}_{:#X}",
            match kind {
                ColumnKind::WideColumn => "wide_column",
                ColumnKind::KeyOfSet => "key_of_set",
            },
            id.as_u128()
        )
    }

    fn get_keyspace_config(kind: ColumnKind) -> fjall::KeyspaceCreateOptions {
        match kind {
            ColumnKind::WideColumn => Self::get_point_lookup_config(),
            ColumnKind::KeyOfSet => Self::get_key_of_set_config(),
        }
    }

    /// Gets or creates a keyspace for the given column type.
    fn get_or_create_keyspace<C: Identifiable>(
        &self,
        kind: ColumnKind,
    ) -> fjall::Keyspace {
        loop {
            let id = C::STABLE_TYPE_ID;

            if let Some(keyspace) = self.keyspaces.get(&id) {
                return keyspace.clone();
            }

            let keyspace_name = Self::keyspace_name_from_id(id, kind);

            if let Entry::Vacant(entry) = self.keyspaces.entry(id) {
                let keyspace = self
                    .db
                    .keyspace(&keyspace_name, || {
                        Self::get_keyspace_config(kind)
                    })
                    .expect("keyspace creation should not fail");

                entry.insert(keyspace.clone());

                return keyspace;
            }
        }
    }

    /// Encodes a value using the postcard format.
    fn encode_value<K: Encode>(
        &self,
        key: &K,
        buffer: &mut Vec<u8>,
        non_empty: bool,
    ) {
        let starting_len = buffer.len();
        let mut encoder = PostcardEncoder::new(&mut *buffer);

        encoder.encode(key, &self.plugin).expect("encoding should not fail");

        if non_empty && starting_len == buffer.len() {
            // Ensure non-empty encoding
            buffer.push(0);
        }
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

    fn encode_wide_column_key<W: WideColumn, C: WideColumnValue<W>>(
        &self,
        key: &W::Key,
        buffer: &mut Vec<u8>,
    ) {
        if W::discriminant_encoding() == DiscriminantEncoding::Prefixed {
            self.encode_value(&C::discriminant(), buffer, false);
        }

        self.encode_value(key, buffer, true);

        if W::discriminant_encoding() == DiscriminantEncoding::Suffixed {
            self.encode_value(&C::discriminant(), buffer, false);
        }
    }

    fn get_point_lookup_config() -> fjall::KeyspaceCreateOptions {
        fjall::KeyspaceCreateOptions::default()
    }

    fn get_key_of_set_config() -> fjall::KeyspaceCreateOptions {
        fjall::KeyspaceCreateOptions::default()
    }
}

/// Write transaction for the [`Fjall`] backend.
///
/// Batches multiple write operations and commits them atomically when
/// [`WriteTransaction::commit`] is called.
///
/// [`WriteTransaction::commit`]: crate::kv_database::WriteBatch::commit
pub struct FjallWriteBatch {
    /// Reference to the parent database.
    db: Arc<Impl>,

    /// The write batch accumulating all operations.
    batch: fjall::OwnedWriteBatch,

    bytes_written: usize,
}

const BATCH_SIZE: usize = 4 * 1024 * 1024; // 4 MB

impl std::fmt::Debug for FjallWriteBatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FjallWriteTransaction")
            .field("db", &self.db)
            .field("batch", &"<Batch>")
            .finish_non_exhaustive()
    }
}

impl FjallWriteBatch {
    /// Creates a new write transaction.
    fn new(db: &Arc<Impl>) -> Self {
        let batch = db.db.batch().durability(None);

        Self { db: db.clone(), batch, bytes_written: 0 }
    }
}

impl WriteBatch for FjallWriteBatch {
    type SerializationBuffer = FjallSerializationBuffer;

    fn put<W: WideColumn, C: WideColumnValue<W>>(
        &mut self,
        key: &W::Key,
        value: &C,
    ) {
        let keyspace =
            self.db.get_or_create_keyspace::<W>(ColumnKind::WideColumn);

        let mut key_buffer = Vec::new();
        let mut value_buffer = Vec::new();

        self.db.encode_wide_column_key::<W, C>(key, &mut key_buffer);
        self.db.encode_value(value, &mut value_buffer, false);

        self.batch.insert(&keyspace, &key_buffer, &value_buffer);

        self.bytes_written += key_buffer.len() + value_buffer.len();
    }

    fn delete<W: WideColumn, C: WideColumnValue<W>>(&mut self, key: &W::Key) {
        let keyspace =
            self.db.get_or_create_keyspace::<W>(ColumnKind::WideColumn);

        let mut key_buffer = Vec::new();

        self.db.encode_wide_column_key::<W, C>(key, &mut key_buffer);

        self.batch.remove(&keyspace, &key_buffer);

        self.bytes_written += key_buffer.len();
    }

    fn insert_member<C: KeyOfSetColumn>(
        &mut self,
        key: &C::Key,
        value: &C::Element,
    ) {
        let keyspace =
            self.db.get_or_create_keyspace::<C>(ColumnKind::KeyOfSet);
        let mut buffer = Vec::new();

        self.db.encode_value_length_prefixed(key, &mut buffer);
        self.db.encode_value(value, &mut buffer, false);

        // For set membership, the value is empty (presence indicates
        // membership)
        self.batch.insert(&keyspace, &buffer, []);

        self.bytes_written += buffer.len();
    }

    fn delete_member<C: KeyOfSetColumn>(
        &mut self,
        key: &C::Key,
        value: &C::Element,
    ) {
        let keyspace =
            self.db.get_or_create_keyspace::<C>(ColumnKind::KeyOfSet);
        let mut buffer = Vec::new();

        self.db.encode_value_length_prefixed(key, &mut buffer);
        self.db.encode_value(value, &mut buffer, false);

        self.batch.remove(&keyspace, &buffer);

        self.bytes_written += buffer.len();
    }

    fn commit(self) {
        let batch = self.batch.durability(None);

        batch.commit().expect("write should not fail");
    }

    fn consume_serialization_buffer(
        &mut self,
        buffer: Self::SerializationBuffer,
    ) {
        for operation in buffer.operations {
            match operation {
                Operation::WideColumnPut { cf, key, value } => {
                    self.batch.insert(&cf, &key, &value);
                    self.bytes_written += key.len() + value.len();
                }

                Operation::WideColumnDelete { cf, key }
                | Operation::DeleteMember { cf, key } => {
                    self.batch.remove(&cf, &key);
                    self.bytes_written += key.len();
                }

                Operation::InsertMember { cf, key } => {
                    self.batch.insert(&cf, &key, []);
                    self.bytes_written += key.len();
                }
            }
        }
    }

    fn should_write_more(&self) -> bool { self.bytes_written < BATCH_SIZE }
}

impl KvDatabase for Fjall {
    type WriteBatch = FjallWriteBatch;

    type SerializationBuffer = FjallSerializationBuffer;

    type ScanMemberIterator<C: KeyOfSetColumn> = ScanMemberIterator<C>;

    fn get_wide_column<W: WideColumn, C: WideColumnValue<W>>(
        &self,
        key: &W::Key,
    ) -> Option<C> {
        let keyspace =
            self.0.get_or_create_keyspace::<W>(ColumnKind::WideColumn);

        let mut buffer = Vec::new();
        self.0.encode_wide_column_key::<W, C>(key, &mut buffer);

        match keyspace.get(&buffer) {
            Ok(Some(value_bytes)) => {
                let mut decoder = PostcardDecoder::new(std::io::Cursor::new(
                    value_bytes.as_ref(),
                ));

                let value = decoder
                    .decode::<C>(&self.0.plugin)
                    .expect("decoding should not fail");

                Some(value)
            }
            Ok(None) => None,

            Err(err) => {
                panic!("Fjall get error: {err}");
            }
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn scan_members<'s, C: KeyOfSetColumn>(
        &'s self,
        key: &'s C::Key,
    ) -> Self::ScanMemberIterator<C> {
        let keyspace = self.0.get_or_create_keyspace::<C>(ColumnKind::KeyOfSet);
        let mut prefix_buffer = Vec::new();
        self.0.encode_value_length_prefixed(key, &mut prefix_buffer);

        // Use prefix iterator
        let iter = keyspace.prefix(prefix_buffer.as_slice());

        ScanMemberIterator {
            iter,
            db: self.0.clone(),
            _marker: std::marker::PhantomData,
        }
    }

    fn write_batch(&self) -> Self::WriteBatch { FjallWriteBatch::new(&self.0) }

    fn serialization_buffer(&self) -> Self::SerializationBuffer {
        FjallSerializationBuffer { operations: Vec::new(), db: self.0.clone() }
    }
}

enum Operation {
    WideColumnPut { cf: Keyspace, key: Vec<u8>, value: Vec<u8> },
    WideColumnDelete { cf: Keyspace, key: Vec<u8> },
    InsertMember { cf: Keyspace, key: Vec<u8> },
    DeleteMember { cf: Keyspace, key: Vec<u8> },
}

/// Serialization buffer for batching Fjall operations.
pub struct FjallSerializationBuffer {
    operations: Vec<Operation>,
    db: Arc<Impl>,
}

impl std::fmt::Debug for FjallSerializationBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FjallSerializationBuffer").finish_non_exhaustive()
    }
}

impl SerializationBuffer for FjallSerializationBuffer {
    fn put<W: WideColumn, C: WideColumnValue<W>>(
        &mut self,
        key: &W::Key,
        value: &C,
    ) {
        let mut key_buffer = Vec::new();
        let mut value_buffer = Vec::new();

        self.db.encode_wide_column_key::<W, C>(key, &mut key_buffer);
        self.db.encode_value(value, &mut value_buffer, false);

        self.operations.push(Operation::WideColumnPut {
            cf: self.db.get_or_create_keyspace::<W>(ColumnKind::WideColumn),
            key: key_buffer,
            value: value_buffer,
        });
    }

    fn delete<W: WideColumn, C: WideColumnValue<W>>(&mut self, key: &W::Key) {
        let mut key_buffer = Vec::new();

        self.db.encode_wide_column_key::<W, C>(key, &mut key_buffer);

        self.operations.push(Operation::WideColumnDelete {
            cf: self.db.get_or_create_keyspace::<W>(ColumnKind::WideColumn),
            key: key_buffer,
        });
    }

    fn insert_member<C: KeyOfSetColumn>(
        &mut self,
        key: &C::Key,
        value: &C::Element,
    ) {
        let mut buffer = Vec::new();

        self.db.encode_value_length_prefixed(key, &mut buffer);
        self.db.encode_value(value, &mut buffer, false);

        self.operations.push(Operation::InsertMember {
            cf: self.db.get_or_create_keyspace::<C>(ColumnKind::KeyOfSet),
            key: buffer,
        });
    }

    fn delete_member<C: KeyOfSetColumn>(
        &mut self,
        key: &C::Key,
        value: &C::Element,
    ) {
        let mut buffer = Vec::new();

        self.db.encode_value_length_prefixed(key, &mut buffer);
        self.db.encode_value(value, &mut buffer, false);

        self.operations.push(Operation::DeleteMember {
            cf: self.db.get_or_create_keyspace::<C>(ColumnKind::KeyOfSet),
            key: buffer,
        });
    }
}

/// Iterator created [`KvDatabase::scan_members`] for the Fjall backend.
pub struct ScanMemberIterator<C> {
    iter: fjall::Iter,
    db: Arc<Impl>,
    _marker: std::marker::PhantomData<C>,
}

impl<C> std::fmt::Debug for ScanMemberIterator<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScanMemberIterator").finish_non_exhaustive()
    }
}

impl<C: KeyOfSetColumn> Iterator for ScanMemberIterator<C> {
    type Item = C::Element;

    #[allow(clippy::cast_possible_truncation)]
    fn next(&mut self) -> Option<Self::Item> {
        let guard = self.iter.next()?;
        let key = guard.key().ok()?;

        let key_bytes = key.as_ref();

        let length = u64::from_le_bytes(
            key_bytes[0..8]
                .try_into()
                .expect("length prefix should be 8 bytes"),
        ) as usize;

        let mut decoder = PostcardDecoder::new(std::io::Cursor::new(
            &key_bytes[8 + length..],
        ));

        Some(
            decoder
                .decode::<C::Element>(&self.db.plugin)
                .expect("decoding should not fail"),
        )
    }
}
