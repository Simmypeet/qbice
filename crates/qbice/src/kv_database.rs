//! Key-Value Database Abstraction Layer.
//!
//! This module provides traits for interacting with key-value databases.

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
/// Each column defines a mapping from keys of the implementing type to values
/// of the associated [`Column::Value`] type. Both the key and value must be
/// encodable and decodable for storage and retrieval.
pub trait Column: Encodable + Decodable {
    /// The type of values stored in this column.
    type Value: Encodable + Decodable;
}

/// A write transaction that allows batching multiple write operations.
///
/// Write transactions provide atomicity guarantees - either all operations in
/// the transaction succeed, or none of them are applied to the database.
pub trait WriteTransaction {
    /// Inserts or updates a key-value pair in the specified column.
    ///
    /// If the key already exists, its value will be overwritten.
    fn put<C: Column>(&self, key: &C, value: &<C as Column>::Value);

    /// Commits all pending write operations to the database.
    ///
    /// # Errors
    ///
    /// Returns an error if the transaction could not be committed (e.g., due to
    /// I/O failures or database corruption).
    fn commit(self) -> Result<(), std::io::Error>;
}

/// The main interface for a key-value database backend.
///
/// This trait abstracts over different key-value storage implementations,
/// allowing the system to work with various backends (e.g., `RocksDB`, `LMDB`,
/// in-memory storage, etc.).
///
/// All operations are async to allow for non-blocking I/O with the underlying
/// storage.
pub trait KvDatabase: Send + Sync {
    /// The type of write transaction provided by this database implementation.
    type WriteTransaction<'a>: WriteTransaction + Send
    where
        Self: 'a;

    /// Retrieves the value associated with the given key from the specified
    /// column.
    ///
    /// Returns `None` if the key does not exist in the database.
    fn get<C: Column>(
        &self,
        key: &C,
    ) -> impl std::future::Future<Output = Option<<C as Column>::Value>> + Send;

    /// Creates a new write transaction for batching multiple write operations.
    ///
    /// The returned transaction must be explicitly committed via
    /// [`WriteTransaction::commit`] for changes to be persisted.
    fn write_transaction(
        &self,
    ) -> impl std::future::Future<Output = Self::WriteTransaction<'_>> + Send;
}
