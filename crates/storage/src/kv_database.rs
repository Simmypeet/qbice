//! Key-Value Database Abstraction Layer
//!
//! This module defines the core traits and types for interacting with key-value
//! databases in a backend-agnostic way. It provides abstractions for logical
//! data structures, column families, transactions, and async access patterns.

use std::{hash::Hash, marker::PhantomData};

use qbice_serialize::{Decode, Encode, Plugin};
use qbice_stable_type_id::Identifiable;

pub mod rocksdb;

/// Marker trait for logical data structure representation in a column family.
///
/// Key-value databases can efficiently represent different logical data
/// structures using different physical representations. This trait is used to
/// indicate which logical structure is being represented in a column family.
///
/// # Implementations
///
/// - Use [`Normal`] for standard key-value pairs where each key maps to a
///   single value (`HashMap<K, V>`).
/// - Use [`KeyOfSet<T>`] for set membership where each key maps to a collection
///   of values (`HashMap<K, HashSet<T>>`).
///
/// # Usage
///
/// This trait is typically used as an associated type in the [`Column`] trait
/// to specify how the column's data should be physically stored.
pub trait StorageMode {}

/// Marker trait for set membership columns.
///
/// This trait extends [`StorageMode`] and is automatically implemented for
/// [`KeyOfSet<T>`] types. It provides access to the element type stored in
/// the set.
///
/// See [`KeyOfSet`] for more details on set-based storage.
pub trait KeyOfSetMode: StorageMode {
    /// The type of each element in the set.
    ///
    /// This type must support serialization, hashing, equality comparison,
    /// and thread-safe sharing.
    type Value: Encode + Decode + Hash + Eq + Clone + 'static + Send + Sync;
}

/// Marker type for normal key-value pairs in a column family.
///
/// Use this storage mode when each key maps to a single value, representing
/// a standard `HashMap<K, V>` logical structure. This is the default and most
/// common storage mode.
///
/// # Example
///
/// ```ignore
/// struct UserColumn;
/// impl Column for UserColumn {
///     type Key = u64;
///     type Value = User;
///     type Mode = Normal;  // Each user ID maps to one User
/// }
/// ```
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Encode, Decode,
)]
#[serialize_crate(qbice_serialize)]
pub struct Normal;

impl StorageMode for Normal {}

/// Marker type for set membership columns (`HashMap<K, HashSet<V>>`).
///
/// ## Motivation
///
/// Naively, adding a value to a set in a key-value store requires reading,
/// modifying, and writing the entire set (a read-modify-write cycle). This is
/// inefficient for large sets and doesn't scale well with concurrent updates.
///
/// Instead, this marker type signals that the set should be stored as
/// individual key-value pairs, where the presence of a key-value pair indicates
/// membership. This allows efficient single-element insertions and deletions
/// without reading the entire set.
///
/// ## When to Use
///
/// Use `KeyOfSet<T>` when:
/// - You frequently add/remove individual elements from sets
/// - Sets can grow large (hundreds or thousands of elements)
/// - Multiple concurrent operations may modify the same set
///
/// Use [`Normal`] mode with a `HashSet<T>` value when:
/// - Sets are small and rarely modified
/// - You typically replace entire sets rather than modifying them
/// - You need to read the entire set frequently
///
/// ## Physical Representation
///
/// For a logical set:
///
/// ```json
/// {
///   "keyA": ["value1", "value2"],
///   "keyB": ["value3"]
/// }
/// ```
///
/// The physical storage is:
///
/// ```txt
/// '<keyA>|<value1>' -> (empty)
/// '<keyA>|<value2>' -> (empty)
/// '<keyB>|<value3>' -> (empty)
/// ```
///
/// The database value is empty; only the composite key's presence indicates
/// set membership.
///
/// **Note:** A length-prefixed encoding is used to avoid prefix collisions
/// (e.g., between "user" and "username").
///
/// ## Iteration
///
/// To iterate all values in a set for a key, use [`KvDatabase::scan_members`]
/// or [`KvDatabase::collect_key_of_set`].
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Encode, Decode,
)]
#[serialize_crate(qbice_serialize)]
pub struct KeyOfSet<T>(PhantomData<T>);

impl<T> StorageMode for KeyOfSet<T> {}

impl<T: Encode + Decode + Hash + Eq + Clone + Send + Sync + 'static>
    KeyOfSetMode for KeyOfSet<T>
{
    type Value = T;
}

/// Represents a column (or table) in the key-value database.
///
/// A column defines the schema for a logical grouping of data, including:
/// - The key type used to identify entries
/// - The value type stored for each key
/// - The storage mode that determines the physical representation
///
/// # Type Requirements
///
/// - Keys must implement [`Encode`], [`Hash`], [`Eq`], and [`Clone`] for
///   serialization and efficient lookups.
/// - Values must implement [`Encode`], [`Decode`], and [`Clone`] for
///   serialization and retrieval.
/// - The column type itself must implement [`Identifiable`] to provide a unique
///   stable identifier.
///
/// # Example
///
/// ```ignore
/// use qbice_storage::kv_database::{Column, Normal};
/// use qbice_stable_type_id::Identifiable;
///
/// // Define a column for storing user data
/// #[derive(Identifiable)]
/// struct UserColumn;
///
/// impl Column for UserColumn {
///     type Key = u64;          // User ID
///     type Value = User;       // User data
///     type Mode = Normal;      // Standard key-value storage
/// }
/// ```
pub trait Column: 'static + Send + Sync + Identifiable {
    /// The type of keys used in this column.
    type Key: Encode + Hash + Eq + Clone + 'static + Send + Sync;

    /// The type of values stored in this column.
    type Value: Encode + Decode + Clone + 'static + Send + Sync;

    /// The tag type representing the storage mode for this column.
    type Mode: StorageMode;
}

impl<
    K: Encode + Decode + Hash + Eq + Clone + 'static + Send + Sync + Identifiable,
    V: Encode + Decode + Clone + 'static + Send + Sync + Identifiable,
> Column for (K, V)
{
    type Key = K;
    type Value = V;
    type Mode = Normal;
}

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
pub trait WriteTransaction {
    /// Insert or overwrite a key-value pair in a [`Normal`] column.
    fn put<C: Column<Mode = Normal>>(
        &self,
        key: &<C as Column>::Key,
        value: &<C as Column>::Value,
    );

    /// Delete a key-value pair from a [`Normal`] column.
    fn delete<C: Column<Mode = Normal>>(&self, key: &<C as Column>::Key);

    /// Insert a value into the set associated with the given key in a
    /// [`KeyOfSet`] column.
    fn insert_member<C: Column>(
        &self,
        key: &<C as Column>::Key,
        value: &<<C as Column>::Mode as KeyOfSetMode>::Value,
    ) where
        <C as Column>::Mode: KeyOfSetMode;

    /// Delete a value from the set associated with the given key in a
    /// [`KeyOfSet`] column.
    fn delete_member<C: Column>(
        &self,
        key: &<C as Column>::Key,
        value: &<<C as Column>::Mode as KeyOfSetMode>::Value,
    ) where
        <C as Column>::Mode: KeyOfSetMode;

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
    type WriteTransaction<'a>: WriteTransaction + Send + Sync
    where
        Self: 'a;

    /// Retrieves the value associated with the given key from a [`Normal`] mode
    /// column.
    ///
    /// # Returns
    ///
    /// - `Some(value)` if the key exists in the database
    /// - `None` if the key does not exist
    ///
    /// # Note
    ///
    /// This method is only available for columns with [`Normal`] storage mode.
    /// For [`KeyOfSet`] columns, use [`scan_members`](Self::scan_members) or
    /// [`collect_key_of_set`](Self::collect_key_of_set) instead.
    fn get<'s, C: Column<Mode = Normal>>(
        &'s self,
        key: &'s C::Key,
    ) -> Option<<C as Column>::Value>;

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
    fn scan_members<'s, C: Column>(
        &'s self,
        key: &'s C::Key,
    ) -> impl Iterator<Item = <C::Mode as KeyOfSetMode>::Value>
    + Send
    + use<'s, Self, C>
    where
        C::Mode: KeyOfSetMode;

    /// Collects all members of the set associated with the given key into a
    /// collection.
    ///
    /// This is a convenience method that calls
    /// [`scan_members`](Self::scan_members) and collects the results into
    /// the column's value type. The value type must implement
    /// `FromIterator` to support collection.
    ///
    /// # Returns
    ///
    /// A collection (e.g., `HashSet`, `Vec`) containing all members of the set.
    /// If the key has no members, returns an empty collection.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Collect all tags for a user into a HashSet
    /// let tags: HashSet<String> = db.collect_key_of_set::<UserTagsColumn>(&user_id);
    /// ```
    fn collect_key_of_set<'s, C: Column>(&'s self, key: &'s C::Key) -> C::Value
    where
        C::Mode: KeyOfSetMode,
        C::Value: FromIterator<<C::Mode as KeyOfSetMode>::Value>,
    {
        self.scan_members::<C>(key).collect()
    }

    /// Creates a new write transaction for batching multiple write operations.
    ///
    /// The returned transaction must be explicitly committed via
    /// [`WriteTransaction::commit`] for changes to be persisted.
    fn write_transaction(&self) -> Self::WriteTransaction<'_>;
}
