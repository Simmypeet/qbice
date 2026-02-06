//! In-memory implementation of [`KeyOfSetMap`].

use dashmap::DashMap;
use fxhash::FxBuildHasher;

use crate::{
    key_of_set_map::{ConcurrentSet, KeyOfSetMap, OwnedIterator},
    kv_database::KeyOfSetColumn,
    sharded::default_shard_amount,
    write_batch::FauxWriteBatch,
};

/// An in-memory implementation of [`KeyOfSetMap`] backed by a concurrent
/// [`DashMap`].
///
/// This implementation stores all key-to-set mappings in memory and is
/// suitable for testing, caching, or scenarios where persistence is not
/// required.
///
/// # Type Parameters
///
/// - `K`: The key-of-set column type that defines the key and element types.
/// - `C`: The concurrent set type used to store elements.
#[derive(Debug)]
pub struct InMemoryKeyOfSetMap<
    K: KeyOfSetColumn,
    C: ConcurrentSet<Element = K::Element>,
> {
    map: DashMap<K::Key, C, FxBuildHasher>,
}

impl<K: KeyOfSetColumn, C: ConcurrentSet<Element = K::Element>>
    InMemoryKeyOfSetMap<K, C>
{
    /// Creates a new in-memory key-of-set map.
    ///
    /// # Returns
    ///
    /// A new instance of `InMemoryKeyOfSetMap`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            map: DashMap::with_capacity_and_hasher(
                default_shard_amount(),
                FxBuildHasher::default(),
            ),
        }
    }
}

impl<K: KeyOfSetColumn, C: ConcurrentSet<Element = K::Element>> Default
    for InMemoryKeyOfSetMap<K, C>
{
    fn default() -> Self { Self::new() }
}

impl<K: KeyOfSetColumn, C: ConcurrentSet<Element = K::Element> + Default>
    KeyOfSetMap<K, C> for InMemoryKeyOfSetMap<K, C>
{
    type WriteBatch = FauxWriteBatch;

    async fn get(&self, key: &K::Key) -> impl Iterator<Item = C::Element> {
        if let Some(set) = self.map.get(key) {
            let cloned_set = set.clone();
            drop(set);

            return OwnedIterator::new(cloned_set, |x| x.iter());
        }

        match self.map.entry(key.clone()) {
            dashmap::Entry::Occupied(occupied_entry) => {
                let cloned_set = occupied_entry.get().clone();
                drop(occupied_entry);

                OwnedIterator::new(cloned_set, |x| x.iter())
            }

            dashmap::Entry::Vacant(vacant_entry) => {
                let new_set = C::default();
                vacant_entry.insert(new_set.clone());

                OwnedIterator::new(new_set, |x| x.iter())
            }
        }
    }

    async fn insert(
        &self,
        key: <K as KeyOfSetColumn>::Key,
        element: <K as KeyOfSetColumn>::Element,
        _write_batch: &mut Self::WriteBatch,
    ) {
        let set = self.map.get(&key).map(|set| set.clone());

        // If the set does not exist, create a new one and insert it.
        if let Some(set) = set {
            set.insert_element(element);
            return;
        }

        let new_set = C::default();

        match self.map.entry(key) {
            dashmap::Entry::Occupied(occupied_entry) => {
                let set = occupied_entry.get();

                set.insert_element(element);
            }

            dashmap::Entry::Vacant(vacant_entry) => {
                vacant_entry.insert(new_set.clone());
                new_set.insert_element(element);
            }
        }
    }

    async fn remove(
        &self,
        key: &<K as KeyOfSetColumn>::Key,
        element: &<K as KeyOfSetColumn>::Element,
        _write_batch: &mut Self::WriteBatch,
    ) {
        let set = self.map.get(key).map(|set| set.clone());

        if let Some(set) = set {
            set.remove_element(element);
        }
    }
}
