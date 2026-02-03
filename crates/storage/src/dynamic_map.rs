//! A map storage abstraction that supports dynamic value types per key.
//!
//! This module provides the [`DynamicMap`] trait for key-value storage where
//! the value type can vary dynamically based on a discriminant.

use crate::{
    kv_database::{WideColumn, WideColumnValue},
    write_transaction::WriteTransaction,
};

pub mod cache;
pub mod in_memory;

/// A trait for key-value map storage that supports dynamic value types.
///
/// Unlike [`SingleMap`](crate::single_map::SingleMap), this trait allows
/// storing multiple value types under the same key, distinguished by their
/// discriminants. This enables heterogeneous storage where different value
/// types can be associated with the same key.
///
/// # Type Parameters
///
/// - `K`: The wide column type that defines the key type and discriminant.
pub trait DynamicMap<K: WideColumn> {
    /// The write batch type used to group write operations atomically.
    type WriteTransaction: WriteTransaction;

    /// Retrieves a value of type `V` by its key.
    ///
    /// The discriminant is determined by the value type `V`.
    ///
    /// # Type Parameters
    ///
    /// - `V`: The value type to retrieve, which determines the discriminant.
    ///
    /// # Parameters
    ///
    /// - `key`: The key to look up.
    ///
    /// # Returns
    ///
    /// Returns `Some(value)` if a value of type `V` exists for the key,
    /// or `None` if not found.
    fn get<'s, 'k, V: WideColumnValue<K>>(
        &'s self,
        key: &'k K::Key,
    ) -> impl std::future::Future<Output = Option<V>> + use<'s, 'k, Self, K, V> + Send;

    /// Inserts a key-value pair into the map.
    ///
    /// The discriminant is determined by the value type `V`. If a value with
    /// the same key and discriminant already exists, it is updated.
    ///
    /// # Type Parameters
    ///
    /// - `V`: The value type to insert.
    ///
    /// # Parameters
    ///
    /// - `key`: The key to insert.
    /// - `value`: The value to associate with the key.
    /// - `write_transaction`: The write transaction to record this operation
    fn insert<'s, 't, V: WideColumnValue<K>>(
        &'s self,
        key: K::Key,
        value: V,
        write_transaction: &'t mut Self::WriteTransaction,
    ) -> impl std::future::Future<Output = ()> + use<'s, 't, Self, K, V> + Send;

    /// Removes a value of type `V` from the map.
    ///
    /// Only removes the value with the discriminant corresponding to type `V`.
    ///
    /// # Type Parameters
    ///
    /// - `V`: The value type to remove, which determines the discriminant.
    ///
    /// # Parameters
    ///
    /// - `key`: The key to remove.
    /// - `write_transaction`: The write transaction to record this operation
    fn remove<'s, 'k, 't, V: WideColumnValue<K>>(
        &'s self,
        key: &'k K::Key,
        write_transaction: &'t mut Self::WriteTransaction,
    ) -> impl std::future::Future<Output = ()> + use<'s, 'k, 't, Self, K, V> + Send;
}
