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
/// - Use [`Normal`] for standard key-value pairs (`HashMap<K, V>`).
/// - Use [`KeyOfSet`] for set membership (`HashMap<K, HashSet<V>>`).
pub trait StorageMode {}

/// Marker trait for set membership columns (`HashMap<K, HashSet<V>>`).
///
/// See [`KeyOfSet`] for more details.
pub trait KeyOfSetMode: StorageMode {
    /// The type of each element in the set.
    type Value: Encode + Decode + Hash + Eq + Clone + 'static + Send + Sync;
}

/// Marker type for normal key-value pairs in a column family.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Encode, Decode,
)]
pub struct Normal;

impl StorageMode for Normal {}

/// Marker type for set membership columns (`HashMap<K, HashSet<V>>`).
///
/// ## Motivation
///
/// Naively, adding a value to a set in a key-value store requires reading,
/// modifying, and writing the entire set. This is inefficient for large sets.
/// Instead, this marker type signals that the set is stored as individual
/// key-value pairs, where the presence of a key-value pair indicates
/// membership.
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
/// The value is not stored; only the key's presence matters.
///
/// **Note:** A length-prefixed encoding is used to avoid prefix collisions
/// (e.g., between "user" and "username").
///
/// # Iteration
///
/// To iterate all values in a set for a key, use the `scan_members` method of
/// [`KvDatabase`].
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Encode, Decode,
)]
pub struct KeyOfSet<T>(PhantomData<T>);

impl<T> StorageMode for KeyOfSet<T> {}

impl<T: Encode + Decode + Hash + Eq + Clone + Send + Sync + 'static>
    KeyOfSetMode for KeyOfSet<T>
{
    type Value = T;
}

/// Represents a column (or table) in the key-value database.
///
/// Each column defines its key and value types, and a storage mode (`Normal` or
/// `KeyOfSet`). Key and value types must implement [`Encode`] and [`Decode`]
/// for serialization.
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

/// A write transaction for batching multiple write operations atomically.
///
/// Write transactions provide atomicity: either all operations succeed, or none
/// are applied.
///
/// - No "read your own writes" guarantee: reads in the same transaction may not
///   see uncommitted writes.
/// - Changes are only visible after `commit` is called.
/// - If dropped without `commit`, all changes must be rolled back.
pub trait WriteTransaction {
    /// Insert or overwrite a key-value pair in a [`Normal`] column.
    fn put<C: Column<Mode = Normal>>(
        &self,
        key: &<C as Column>::Key,
        value: &<C as Column>::Value,
    );

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
/// All operations are async to support non-blocking I/O.
pub trait KvDatabase: 'static + Send + Sync {
    /// The type of write transaction provided by this database implementation.
    type WriteTransaction<'a>: WriteTransaction + Send + Sync
    where
        Self: 'a;

    /// Retrieves the value associated with the given key from the specified
    /// column.
    ///
    /// Returns `None` if the key does not exist.
    fn get<'s, C: Column<Mode = Normal>>(
        &'s self,
        key: &'s C::Key,
    ) -> Option<<C as Column>::Value>;

    /// Scans all members of the set associated with the given key in the
    /// specified column.
    ///
    /// Returns a stream of values in the set.
    fn scan_members<'s, C: Column>(
        &'s self,
        key: &'s C::Key,
    ) -> impl Iterator<Item = <C::Mode as KeyOfSetMode>::Value>
    + Send
    + use<'s, Self, C>
    where
        C::Mode: KeyOfSetMode;

    /// Collects all members of the set associated with the given key in the
    /// specified column.
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
