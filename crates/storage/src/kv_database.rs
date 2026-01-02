//! Key-Value Database Abstraction Layer
//!
//! This module defines the core traits and types for interacting with key-value
//! databases in a backend-agnostic way. It provides abstractions for logical
//! data structures, column families, transactions, and async access patterns.

use std::hash::Hash;

use qbice_serialize::{Decode, Encode, Plugin};
use qbice_stable_type_id::Identifiable;

pub mod rocksdb;

/// A write transaction for batching multiple write operations atomically.
///
/// Write transactions provide atomicity: either all operations succeed, or none
/// are applied. This is useful for ensuring consistency when multiple related
/// changes must be made together.
///
/// # Visibility and Isolation
///
/// - Changes made within a transaction are **not visible** to other readers
///   until [`commit`](Self::commit) is called.
/// - If the transaction is dropped without calling `commit`, all changes **must
///   be rolled back** automatically.
/// - Transactions provide snapshot isolation: reads within a transaction see a
///   consistent view of the database from when the transaction started.
///
/// # Error Handling
///
/// Individual operations (`put`, `delete`, `insert_member`, `delete_member`)
/// are assumed to always succeed. If they cannot (e.g., due to serialization
/// errors), they should panic. The `commit` operation is also assumed to
/// succeed; if it cannot, the implementation should panic.
///
/// # Thread Safety
///
/// Transactions are `Send + Sync` and can be shared across threads, but
/// implementations should document any specific concurrency guarantees or
/// limitations.
pub trait WriteBatch {
    /// Inserts or updates a value in a wide column.
    fn put<W: WideColumn, C: WideColumnValue<W>>(
        &self,
        key: &W::Key,
        value: &C,
    );

    /// Deletes a value from a wide column.
    fn delete<W: WideColumn, C: WideColumnValue<W>>(&self, key: &W::Key);

    /// Inserts a member into the set associated with the given key in a
    /// [`KeyOfSet`] mode column.
    fn insert_member<C: KeyOfSetColumn>(
        &self,
        key: &C::Key,
        value: &C::Element,
    );

    /// Deletes a member from the set associated with the given key in a
    /// [`KeyOfSet`] mode column.
    fn delete_member<C: KeyOfSetColumn>(
        &self,
        key: &C::Key,
        value: &C::Element,
    );

    /// Commits all pending write operations to the database.
    ///
    /// After commit, all changes become visible to other readers.
    ///
    /// Implementations should assume commit always succeeds; rollback on
    /// failure is not supported.
    fn commit(self);
}

/// Factory trait for creating instances of a key-value database backend.
pub trait KvDatabaseFactory {
    /// The type of key-value database produced by this factory.
    type KvDatabase;

    /// The error type returned if opening the database fails.
    type Error;

    /// Opens a new instance of the key-value database with the given
    /// serialization plugin.
    fn open(
        self,
        serialization_plugin: Plugin,
    ) -> Result<Self::KvDatabase, Self::Error>;
}

/// The main interface for a key-value database backend.
///
/// This trait abstracts over different key-value storage implementations,
/// allowing the system to work with various backends (e.g., `RocksDB`, `LMDB`,
/// in-memory, etc.).
///
/// # Thread Safety
///
/// All implementations of `KvDatabase` must be thread-safe (`Send + Sync`).
/// Multiple threads can safely read from and write to the database
/// concurrently. Reads never block other reads, and write transactions provide
/// isolation guarantees.
///
/// # Storage Modes
///
/// The database supports two storage modes via the [`Column::Mode`] associated
/// type:
///
/// - **[`Normal`] mode**: Use [`get`](Self::get) to retrieve single values.
///   Returns `None` if the key doesn't exist.
///
/// - **[`KeyOfSet`] mode**: Use [`scan_members`](Self::scan_members) or
///   [`collect_key_of_set`](Self::collect_key_of_set) to retrieve all members
///   of the set. Returns an empty iterator/collection if the key has no
///   members.
///
/// # Lifetime Considerations
///
/// Methods like `get` and `scan_members` borrow `self` with lifetime `'s`.
/// The returned iterators and values are tied to this lifetime, ensuring they
/// don't outlive the database reference.
pub trait KvDatabase: 'static + Send + Sync {
    /// The type of write transaction provided by this database implementation.
    type WriteBatch: WriteBatch + Send + Sync;

    /// Retrieves a value from a wide column based on its key.
    fn get_wide_column<W: WideColumn, C: WideColumnValue<W>>(
        &self,
        key: &W::Key,
    ) -> Option<C>;

    /// Scans all members of the set associated with the given key in a
    /// [`KeyOfSet`] mode column.
    ///
    /// This method returns an iterator over all values that are members of the
    /// set for the given key. If the key has no members, the iterator will be
    /// empty.
    ///
    /// # Returns
    ///
    /// An iterator yielding each member of the set. The iteration order is
    /// implementation-defined and may not be stable across calls.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Scan all tags for a user
    /// for tag in db.scan_members::<UserTagsColumn>(&user_id) {
    ///     println!("Tag: {}", tag);
    /// }
    /// ```
    fn scan_members<'s, C: KeyOfSetColumn>(
        &'s self,
        key: &'s C::Key,
    ) -> impl Iterator<Item = C::Element> + use<'s, Self, C>;

    /// Creates a new write transaction for batching multiple write operations.
    ///
    /// The returned transaction must be explicitly committed via
    /// [`WriteTransaction::commit`] for changes to be persisted.
    fn write_transaction(&self) -> Self::WriteBatch;
}

/// Specifies how the discriminant is encoded in a wide column.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DiscriminantEncoding {
    /// The discriminant is stored as a prefix to the key.
    Prefixed,

    /// The discriminant is stored as a suffix to the key.
    Suffixed,
}

/// A trait representing a wide column storage structure.
///
/// Wide columns allow storing multiple related values under a single key,
/// distinguished by a discriminant (e.g., a category or type identifier).
/// Some databases support wide columns natively, while others may emulate them
/// using composite keys.
///
/// For example, [`rocksdb::RocksDB`] doesn't natively support wide columns, but
/// they can be emulated by encoding the discriminant into the key itself.
pub trait WideColumn: Identifiable + Send + Sync + 'static {
    /// The type of the discriminant used to distinguish different values.
    type Discriminant: Encode
        + Decode
        + Hash
        + Eq
        + Clone
        + 'static
        + Send
        + Sync;

    /// The type of keys used in this wide column.
    type Key: Encode + Decode + Hash + Eq + Clone + 'static + Send + Sync;

    /// Specifies how the discriminant is encoded in the key.
    fn discriminant_encoding() -> DiscriminantEncoding;
}

/// A trait for values associated with a [`WideColumn`].
pub trait WideColumnValue<W: WideColumn>:
    Encode + Decode + Clone + Send + Sync + 'static
{
    /// Retrieves the discriminant for this value.
    ///
    /// **IMPORTANT:** This method should return a unique discriminant for each
    /// variant of the wide column. Failing to do so may lead to data corruption
    /// or loss, as multiple values could map to the same storage location.
    fn discriminant() -> W::Discriminant;
}

/// A trait for columns that logically represent `HashMap<K, HashSet<V>>`.
///
/// This allows fast:
/// - insertions
/// - deletions
/// - membership tests
///
/// of elements in the set associated with a key.
pub trait KeyOfSetColumn: Identifiable + Send + Sync + 'static {
    /// The type of keys used in this column.
    type Key: Encode + Hash + Eq + Clone + 'static + Send + Sync;

    /// The type of elements in the set.
    type Element: Encode + Decode + Hash + Eq + Clone + 'static + Send + Sync;
}
