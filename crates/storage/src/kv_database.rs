//! Key-Value Database Abstraction Layer.
//!
//! This module provides traits for interacting with key-value databases.

use std::hash::Hash;

use futures::Stream;
use qbice_serialize::{Decode, Encode, Plugin};
use qbice_stable_type_id::Identifiable;

pub mod rocksdb;

/// A trait representing what logical data structure a column is storing.
///
/// KV-databases can efficiently represent different logical data structures
/// using different physical representations. This trait is used to indicate
/// which logical structure is being represented.
///
/// - Representing normal `HashMap<K, V>` key-value pairs: [`Normal`].
/// - Representing `HashMap<K, HashSet<V>>` keys mapping to sets of values :
///   [`KeyOfSet`].
pub trait StorageMode {}

/// A marker type indicating that this column family is used to represent
/// normal key-value pairs.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Encode, Decode,
)]
pub struct Normal;

impl StorageMode for Normal {}

/// A marker type indicating that this column family is used to represent
/// a key mapping to a set of values
/// (something equivalent to `HashMap<K  HashSet<V>>`).
///
/// ## Motivation
///
/// When inserting a value into a set associated with a key, in naive
/// implementation, we need to deserialize the entire set, modify it in memory,
/// and then serialize it back to the database. This can be inefficient for
/// large sets. To optimize this, we store the data in physical representation
/// in different way that allows us to add or remove individual values from the
/// set without needing to read or write the entire set.
///
/// ## Physical Representation
///
/// The physical representation in the database will be different from a
/// normal key-value pair. This is how it's represented at the physical level.
///
/// ```json
/// {
///     "keyA": ["value1", "value2", "value3"],
///     "keyB": ["value4", "value5"]
/// }
/// ```
///
/// Internally, this will be represented as:
///
/// ```txt
/// - '<keyA>|<value1>' -> (empty)
/// - '<keyA>|<value2>' -> (empty)
/// - '<keyA>|<value3>' -> (empty)
/// - '<keyB>|<value4>' -> (empty)
/// - '<keyB>|<value5>' -> (empty)
/// ```
///
/// We use the `KeyOfSet` marker type to indicate that the value type is not
/// actually stored, and the presence of the key-value pair indicates membership
/// in the set.
///
/// **NOTE**: Additionally, it should append a length-prefixed encoding to the
/// key to avoid prefix collisions. For example, keys "user" and "username"
/// share the same prefix "user", which could lead to incorrect set
///
/// # Iterating Over Sets
///
/// To iterate over all values in the set of a given key, use the `prefix_scan`
/// functionality of the underlying `KvDatabase`. For example, to get all values
/// in the set for `keyA`, you would perform a prefix scan with the prefix
/// `"<keyA>|"`. This will return all entries that belong to the set associated
/// with `keyA`.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Encode, Decode,
)]
pub struct KeyOfSet;

impl StorageMode for KeyOfSet {}

/// Represents a column (or table) in the key-value database.
///
/// Each column has associated key and value types, which must implement the
/// `Encodable` and `Decodable` traits for serialization and deserialization.
pub trait Column: 'static + Send + Sync + Identifiable {
    /// The type of keys used in this column.
    type Key: Encode + Decode + Hash + Eq + Clone + 'static + Send + Sync;

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

/// A write transaction that allows batching multiple write operations.
///
/// Write transactions provide atomicity guarantees - either all operations in
/// the transaction succeed, or none of them are applied to the database.
///
/// The write transaction doesn't provide "read your own writes" semantics.
/// But it does guarantee that the database will only see the committed writes
/// after `commit` is called.
///
/// The implementation must rollback any changes if `commit` is not called
/// (dropped without commit).
pub trait WriteTransaction {
    /// If the key already exists, its value will be overwritten.
    fn put<C: Column<Mode = Normal>>(
        &self,
        key: &<C as Column>::Key,
        value: &<C as Column>::Value,
    );

    /// Inserts a member into the set associated with the given key in the
    /// specified column.
    fn insert_member<C: Column<Mode = KeyOfSet>>(
        &self,
        key: &<C as Column>::Key,
        value: &<C as Column>::Value,
    );

    /// Commits all pending write operations to the database.
    ///
    /// We assume that commit always succeeds as it's very difficult to restore
    /// the invariant that a transaction.
    fn commit(self);
}

/// A factory trait for creating instances of a key-value database.
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
/// in-memory storage, etc.).
///
/// All operations are async to allow for non-blocking I/O with the underlying
/// storage.
pub trait KvDatabase: 'static + Send + Sync {
    /// The type of write transaction provided by this database implementation.
    type WriteTransaction<'a>: WriteTransaction + Send
    where
        Self: 'a;

    /// Retrieves the value associated with the given key from the specified
    /// column.
    ///
    /// Returns `None` if the key does not exist in the database.
    fn get<'s, C: Column<Mode = Normal>>(
        &'s self,
        key: &'s C::Key,
    ) -> impl std::future::Future<Output = Option<<C as Column>::Value>>
    + Send
    + use<'s, Self, C>;

    /// Scans all members of the set associated with the given key in the
    /// specified column.
    ///
    /// Returns a stream of values in the set.
    fn scan_members<'s, C: Column<Mode = KeyOfSet>>(
        &'s self,
        key: &'s C::Key,
    ) -> impl Stream<Item = <C as Column>::Value> + Send + use<'s, Self, C>;

    /// Creates a new write transaction for batching multiple write operations.
    ///
    /// The returned transaction must be explicitly committed via
    /// [`WriteTransaction::commit`] for changes to be persisted.
    fn write_transaction(&self) -> Self::WriteTransaction<'_>;
}
