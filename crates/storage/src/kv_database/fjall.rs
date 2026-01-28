//! [`Fjall`] backend implementation for the key-value database abstraction.
//!
//! This module provides a [`Fjall`] struct that implements the [`KvDatabase`]
//! trait using [`Fjall`] as the underlying storage engine.

use std::{path::Path, sync::Arc};

use dashmap::{DashMap, Entry};
use fjall::Keyspace;
use parking_lot::Mutex;
use qbice_serialize::{
    Decoder, Encode, Encoder, Plugin, PostcardDecoder, PostcardEncoder,
};
use qbice_stable_type_id::{Identifiable, StableTypeID};

use crate::kv_database::{
    DiscriminantEncoding, KeyOfSetColumn, KvDatabase, KvDatabaseFactory,
    WideColumn, WideColumnValue, WriteBatch, buffer_pool::BufferPool,
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
#[derive(Debug)]
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

    /// Buffer pool for reusing encoding/decoding buffers.
    buffer_pool: BufferPool,
}

impl std::fmt::Debug for Impl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Impl")
            .field("db", &"<Database>")
            .field("plugin", &self.plugin)
            .field("buffer_pool", &self.buffer_pool)
            .finish_non_exhaustive()
    }
}

/// Factory for creating [`Fjall`] instances.
#[derive(Debug)]
pub struct FjallFactory<P> {
    path: P,
}

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

        Ok(Self(Arc::new(Impl {
            db,
            plugin,
            keyspaces: DashMap::new(),
            buffer_pool: BufferPool::new(),
        })))
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
    batch: Mutex<fjall::OwnedWriteBatch>,
}

impl std::fmt::Debug for FjallWriteBatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FjallWriteTransaction")
            .field("db", &self.db)
            .field("batch", &"<Batch>")
            .finish()
    }
}

impl FjallWriteBatch {
    /// Creates a new write transaction.
    fn new(db: &Arc<Impl>) -> Self {
        let batch = db.db.batch().durability(None);

        Self { db: db.clone(), batch: Mutex::new(batch) }
    }
}

impl WriteBatch for FjallWriteBatch {
    fn put<W: WideColumn, C: WideColumnValue<W>>(
        &self,
        key: &W::Key,
        value: &C,
    ) {
        let keyspace =
            self.db.get_or_create_keyspace::<W>(ColumnKind::WideColumn);

        let mut key_buffer = self.db.buffer_pool.get_buffer();
        let mut value_buffer = self.db.buffer_pool.get_buffer();

        self.db.encode_wide_column_key::<W, C>(key, &mut key_buffer);
        self.db.encode_value(value, &mut value_buffer, false);

        self.batch.lock().insert(&keyspace, &key_buffer, &value_buffer);

        self.db.buffer_pool.return_buffer(key_buffer);
        self.db.buffer_pool.return_buffer(value_buffer);
    }

    fn delete<W: WideColumn, C: WideColumnValue<W>>(&self, key: &W::Key) {
        let keyspace =
            self.db.get_or_create_keyspace::<W>(ColumnKind::WideColumn);

        let mut key_buffer = self.db.buffer_pool.get_buffer();

        self.db.encode_wide_column_key::<W, C>(key, &mut key_buffer);

        self.batch.lock().remove(&keyspace, &key_buffer);

        self.db.buffer_pool.return_buffer(key_buffer);
    }

    fn insert_member<C: KeyOfSetColumn>(
        &self,
        key: &C::Key,
        value: &C::Element,
    ) {
        let keyspace =
            self.db.get_or_create_keyspace::<C>(ColumnKind::KeyOfSet);
        let mut buffer = self.db.buffer_pool.get_buffer();

        self.db.encode_value_length_prefixed(key, &mut buffer);
        self.db.encode_value(value, &mut buffer, false);

        // For set membership, the value is empty (presence indicates
        // membership)
        self.batch.lock().insert(&keyspace, &buffer, []);

        self.db.buffer_pool.return_buffer(buffer);
    }

    fn delete_member<C: KeyOfSetColumn>(
        &self,
        key: &C::Key,
        value: &C::Element,
    ) {
        let keyspace =
            self.db.get_or_create_keyspace::<C>(ColumnKind::KeyOfSet);
        let mut buffer = self.db.buffer_pool.get_buffer();

        self.db.encode_value_length_prefixed(key, &mut buffer);
        self.db.encode_value(value, &mut buffer, false);

        self.batch.lock().remove(&keyspace, &buffer);

        self.db.buffer_pool.return_buffer(buffer);
    }

    fn commit(self) {
        let batch = self.batch.into_inner().durability(None);

        batch.commit().expect("write should not fail");
    }
}

impl KvDatabase for Fjall {
    type WriteBatch = FjallWriteBatch;

    fn get_wide_column<W: WideColumn, C: WideColumnValue<W>>(
        &self,
        key: &W::Key,
    ) -> Option<C> {
        let keyspace =
            self.0.get_or_create_keyspace::<W>(ColumnKind::WideColumn);

        let mut buffer = self.0.buffer_pool.get_buffer();
        self.0.encode_wide_column_key::<W, C>(key, &mut buffer);

        match keyspace.get(&buffer) {
            Ok(Some(value_bytes)) => {
                let mut decoder = PostcardDecoder::new(std::io::Cursor::new(
                    value_bytes.as_ref(),
                ));

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
                panic!("Fjall get error: {err}");
            }
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn scan_members<'s, C: KeyOfSetColumn>(
        &'s self,
        key: &'s C::Key,
    ) -> impl Iterator<Item = C::Element> {
        let keyspace = self.0.get_or_create_keyspace::<C>(ColumnKind::KeyOfSet);
        let mut prefix_buffer = self.0.buffer_pool.get_buffer();
        self.0.encode_value_length_prefixed(key, &mut prefix_buffer);

        // Use prefix iterator
        let iter = keyspace.prefix(prefix_buffer.as_slice());

        iter.filter_map(move |entry| {
            entry.key().ok().map(|key| {
                let key_bytes = key.as_ref();

                let length = u64::from_le_bytes(
                    key_bytes[0..8]
                        .try_into()
                        .expect("length prefix should be 8 bytes"),
                ) as usize;

                let mut decoder = PostcardDecoder::new(std::io::Cursor::new(
                    &key_bytes[8 + length..],
                ));

                decoder
                    .decode::<C::Element>(&self.0.plugin)
                    .expect("decoding should not fail")
            })
        })
    }

    fn write_batch(&self) -> Self::WriteBatch { FjallWriteBatch::new(&self.0) }
}
