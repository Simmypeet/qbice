//! A map storage abstraction for key-to-set relationships.
//!
//! This module provides the [`KeyOfSetMap`] trait for efficiently storing and
//! managing `HashMap<K, HashSet<V>>` relationships.

use std::{
    hash::{BuildHasher, Hash},
    sync::Arc,
};

use dashmap::DashSet;
use ouroboros::self_referencing;

use crate::kv_database::KeyOfSetColumn;

pub mod cache;
pub mod in_memory;

/// A trait for containers used in key-of-set storage.
///
/// This implemented type must be able to clone itself in cheap manner. For
/// example, `Arc<DashSet<T>>` is a good candidate.
pub trait ConcurrentSet: Clone + Default + Send + Sync + 'static {
    /// The type of elements stored in the set.
    type Element: Eq + Hash + Send + Sync + 'static;

    /// The iterator type over the elements in the set.
    type Iterator<'x>: Iterator<Item = Self::Element> + Send
    where
        Self: 'x;

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

    /// Returns the number of elements in the set.
    fn len(&self) -> usize;

    /// Checks if the set is empty.
    fn is_empty(&self) -> bool { self.len() == 0 }

    /// Returns an iterator over the elements in the set.
    fn iter(&self) -> Self::Iterator<'_>;
}

/// An iterator over a DashSet that yields cloned elements.
pub struct ClonedDashSetIterator<
    's,
    T: Eq + Hash,
    S: BuildHasher + Clone + Send + Sync,
> {
    inner: dashmap::iter_set::Iter<'s, T, S, dashmap::DashMap<T, (), S>>,
}

impl<T: Clone + Eq + Hash, S: BuildHasher + Clone + Send + Sync> std::fmt::Debug
    for ClonedDashSetIterator<'_, T, S>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClonedDashSetIterator").finish_non_exhaustive()
    }
}

impl<'s, T: Clone + Eq + Hash, S: BuildHasher + Clone + Send + Sync>
    ClonedDashSetIterator<'s, T, S>
{
    /// Creates a new `ClonedDashSetIterator` from a DashSet iterator.
    #[must_use]
    pub const fn new(
        inner: dashmap::iter_set::Iter<'s, T, S, dashmap::DashMap<T, (), S>>,
    ) -> Self {
        Self { inner }
    }
}

impl<T: Clone + Eq + Hash, S: BuildHasher + Clone + Send + Sync> Iterator
    for ClonedDashSetIterator<'_, T, S>
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|x| x.key().clone())
    }
}

#[allow(clippy::type_repetition_in_bounds)]
impl<
    T: Send + Sync + 'static,
    S: Default + BuildHasher + Clone + Send + Sync + 'static,
> ConcurrentSet for Arc<DashSet<T, S>>
where
    T: Eq + Hash + Clone,
{
    type Iterator<'x> = ClonedDashSetIterator<'x, T, S>;

    type Element = T;

    fn insert_element(&self, element: Self::Element) -> bool {
        self.insert(element)
    }

    fn remove_element(&self, element: &Self::Element) -> bool {
        self.remove(element).is_some()
    }

    /// Returns an iterator over the elements in the set.
    fn iter(&self) -> Self::Iterator<'_> {
        ClonedDashSetIterator { inner: DashSet::iter(self) }
    }

    fn len(&self) -> usize { DashSet::len(self) }
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
    ) -> impl std::future::Future<
        Output = impl Iterator<Item = K::Element> + use<'s, 'k, Self, K, C> + Send,
    > + use<'s, 'k, Self, K, C>
    + Send;

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
    fn insert<'s, 't>(
        &'s self,
        key: K::Key,
        element: K::Element,
        write_batch: &'t mut Self::WriteBatch,
    ) -> impl std::future::Future<Output = ()> + use<'s, 't, Self, K, C> + Send;

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
    fn remove<'s, 'k, 'e, 't>(
        &'s self,
        key: &'k K::Key,
        element: &'e K::Element,
        write_batch: &'t mut Self::WriteBatch,
    ) -> impl std::future::Future<Output = ()> + use<'s, 'k, 'e, 't, Self, K, C> + Send;
}

#[self_referencing]
pub struct OwnedIterator<C: ConcurrentSet + 'static> {
    owned_iter: C,

    #[borrows(mut owned_iter)]
    #[not_covariant]
    iter: C::Iterator<'this>,
}

impl<C: ConcurrentSet + 'static> Iterator for OwnedIterator<C> {
    type Item = C::Element;

    fn next(&mut self) -> Option<Self::Item> {
        self.with_iter_mut(|iter| iter.next())
    }
}
