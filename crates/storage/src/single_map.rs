//! A map storage abstraction for single value types per key.
//!
//! This module provides the [`SingleMap`] trait for key-value storage with a
//! fixed value type.

use crate::{
    kv_database::{WideColumn, WideColumnValue},
    write_batch::WriteBatch,
};

pub mod cache;
pub mod in_memory;

/// A trait for key-value map storage with a fixed value type.
///
/// This trait provides a simple interface for storing and retrieving values
/// associated with keys, where all values have the same type `V`.
///
/// # Type Parameters
///
/// - `K`: The wide column type that defines the key type and discriminant.
/// - `V`: The value type to store, which must implement [`WideColumnValue<K>`].
pub trait SingleMap<K: WideColumn, V: WideColumnValue<K>> {
    /// The write batch type used to group write operations atomically.
    type WriteTransaction: WriteBatch;

    /// Retrieves a value by its key.
    ///
    /// # Parameters
    ///
    /// - `key`: The key to look up.
    ///
    /// # Returns
    ///
    /// Returns `Some(value)` if the key exists, or `None` if not found.
    fn get<'s, 'k>(
        &'s self,
        key: &'k K::Key,
    ) -> impl std::future::Future<Output = Option<V>> + use<'s, 'k, Self, K, V> + Send;

    /// Inserts a key-value pair into the map.
    ///
    /// If the key already exists, the value is updated.
    ///
    /// # Parameters
    ///
    /// - `key`: The key to insert.
    /// - `value`: The value to associate with the key.
    /// - `write_transaction`: The write transaction to record this operation
    ///   in.
    fn insert<'s, 't>(
        &'s self,
        key: K::Key,
        value: V,
        write_transaction: &'t mut Self::WriteTransaction,
    ) -> impl std::future::Future<Output = ()> + use<'s, 't, Self, K, V> + Send;

    /// Removes a key-value pair from the map.
    ///
    /// # Parameters
    ///
    /// - `key`: The key to remove.
    /// - `write_transaction`: The write transaction to record this operation
    ///   in.
    fn remove<'s, 'k, 't>(
        &'s self,
        key: &'k K::Key,
        write_transaction: &'t mut Self::WriteTransaction,
    ) -> impl std::future::Future<Output = ()> + use<'s, 'k, 't, Self, K, V> + Send;
}
