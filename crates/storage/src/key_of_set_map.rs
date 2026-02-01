use std::{
    hash::{BuildHasher, Hash},
    sync::Arc,
};

use dashmap::DashSet;

use crate::kv_database::KeyOfSetColumn;

/// In-memory implementation of [`KeyOfSetMap`].
pub mod in_memory;

/// A trait for containers used in key-of-set storage.
///
/// This implemented type must be able to clone itself in cheap manner. For
/// example, `Arc<DashSet<T>>` is a good candidate.
pub trait ConcurrentSet: Clone + Default + Send + Sync {
    /// The type of elements stored in the set.
    type Element;

    /// Inserts an element into the set.
    ///
    /// # Returns
    ///
    /// - `true` if the element was not present and inserted
    /// - `false` if the element was already present
    fn insert_element(&self, element: Self::Element) -> bool;

    /// Removes an element from the set.
    ///
    /// # Returns
    ///
    /// - `true` if the element was present and removed
    /// - `false` if the element was not found in the set
    fn remove_element(&self, element: &Self::Element) -> bool;
}

#[allow(clippy::type_repetition_in_bounds)]
impl<T: Send + Sync, S: Default + BuildHasher + Clone + Send + Sync>
    ConcurrentSet for Arc<DashSet<T, S>>
where
    T: Eq + Hash + Clone,
{
    type Element = T;

    fn insert_element(&self, element: Self::Element) -> bool {
        self.insert(element)
    }

    fn remove_element(&self, element: &Self::Element) -> bool {
        self.remove(element).is_some()
    }
}

/// A trait for key-to-set map storage.
///
/// This trait provides an efficient interface for storing and managing
/// `HashMap<K, HashSet<V>>` relationships, where each key maps to a set of
/// elements.
///
/// # Type Parameters
///
/// - `K`: The key-of-set column type that defines the key and element types.
/// - `C`: The concurrent set type used to store elements.
pub trait KeyOfSetMap<K: KeyOfSetColumn, C: ConcurrentSet<Element = K::Element>>
{
    /// The write batch type used to group write operations.
    type WriteBatch;

    /// Retrieves the set associated with a key.
    ///
    /// If the key does not exist, returns an empty set and may create the
    /// entry for future operations.
    ///
    /// # Parameters
    ///
    /// - `key`: The key to look up.
    ///
    /// # Returns
    ///
    /// Returns the set associated with the key.
    fn get<'s, 'k>(
        &'s self,
        key: &'k K::Key,
    ) -> impl std::future::Future<Output = C> + use<'s, 'k, Self, K, C> + Send;

    /// Inserts an element into the set associated with a key.
    ///
    /// If the key does not exist, a new set is created.
    ///
    /// # Parameters
    ///
    /// - `key`: The key whose set to insert into.
    /// - `element`: The element to insert.
    /// - `write_batch`: The write batch to record this operation in.
    ///
    /// # Returns
    ///
    /// Returns `true` if the element was inserted (not already present),
    /// `false` if it already existed.
    fn insert(
        &self,
        key: K::Key,
        element: K::Element,
        write_batch: &mut Self::WriteBatch,
    ) -> bool;

    /// Removes an element from the set associated with a key.
    ///
    /// # Parameters
    ///
    /// - `key`: The key whose set to remove from.
    /// - `element`: The element to remove.
    /// - `write_batch`: The write batch to record this operation in.
    ///
    /// # Returns
    ///
    /// Returns `true` if the element was removed, `false` if it was not found.
    fn remove(
        &self,
        key: &K::Key,
        element: &K::Element,
        write_batch: &mut Self::WriteBatch,
    ) -> bool;
}
