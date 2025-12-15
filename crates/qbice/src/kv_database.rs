//! Key-Value Database Abstraction Layer.
//!
//! This module provides traits for interacting with key-value databases.

use std::hash::Hash;

/// A trait for types that can be encoded into a binary format for storage in
/// the key-value database.
///
/// Implementations should define how to serialize the type into bytes that can
/// be stored as keys or values in the database.
pub trait Encodable {} // placeholder we'll fill in later

/// A trait for types that can be decoded from a binary format retrieved from
/// the key-value database.
///
/// Implementations should define how to deserialize bytes back into the
/// original type.
pub trait Decodable {} // placeholder we'll fill in later

/// Represents a column (or table) in the key-value database.
///
/// Each column has associated key and value types, which must implement the
/// `Encodable` and `Decodable` traits for serialization and deserialization.
pub trait Column: 'static + Send + Sync {
    /// The type of keys used in this column.
    type Key: Encodable + Decodable + Hash + Eq + Clone + 'static + Send + Sync;

    /// The type of values stored in this column.
    type Value: Encodable + Decodable + 'static + Send + Sync;
}

impl<
    K: Encodable + Decodable + Hash + Eq + Clone + 'static + Send + Sync,
    V: Encodable + Decodable + 'static + Send + Sync,
> Column for (K, V)
{
    type Key = K;
    type Value = V;
}

/// A write transaction that allows batching multiple write operations.
///
/// Write transactions provide atomicity guarantees - either all operations in
/// the transaction succeed, or none of them are applied to the database.
pub trait WriteTransaction {
    /// Inserts or updates a key-value pair in the specified column.
    ///
    /// If the key already exists, its value will be overwritten.
    fn put<'s, C: Column>(
        &'s self,
        key: &'s C,
        value: &'s <C as Column>::Value,
    ) -> impl std::future::Future<Output = ()> + Send + use<'s, Self, C>;

    /// Commits all pending write operations to the database.
    ///
    /// # Errors
    ///
    /// Returns an error if the transaction could not be committed (e.g., due to
    /// I/O failures or database corruption).
    fn commit(
        self,
    ) -> impl std::future::Future<Output = Result<(), std::io::Error>>
    + Send
    + use<Self>;
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
    fn get<'s, C: Column>(
        &'s self,
        key: &'s C::Key,
    ) -> impl std::future::Future<Output = Option<<C as Column>::Value>>
    + Send
    + use<'s, Self, C>;

    /// Creates a new write transaction for batching multiple write operations.
    ///
    /// The returned transaction must be explicitly committed via
    /// [`WriteTransaction::commit`] for changes to be persisted.
    fn write_transaction(
        &self,
    ) -> impl std::future::Future<Output = Self::WriteTransaction<'_>>
    + Send
    + use<'_, Self>;
}
