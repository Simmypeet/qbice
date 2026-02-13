//! In-memory implementation of [`SingleMap`].

use fxhash::FxBuildHasher;

use crate::{
    kv_database::{WideColumn, WideColumnValue},
    single_map::SingleMap,
    write_batch::FauxWriteBatch,
};

/// An in-memory implementation of [`SingleMap`] backed by a concurrent
/// [`DashMap`].
///
/// This implementation stores all key-value pairs in memory and is suitable
/// for testing, caching, or scenarios where persistence is not required.
///
/// # Type Parameters
///
/// - `K`: The wide column type that defines the key type.
/// - `V`: The value type to store.
/// - `S`: The hash builder type for the underlying [`DashMap`].
#[derive(Debug)]
pub struct InMemorySingleMap<K: WideColumn, V> {
    map: scc::HashMap<K::Key, V, FxBuildHasher>,
}

impl<K: WideColumn, V> InMemorySingleMap<K, V> {
    /// Creates a new in-memory single map with the specified hash builder.
    ///
    /// # Returns
    ///
    /// A new instance of `InMemorySingleMap`.
    #[must_use]
    pub fn new() -> Self {
        Self { map: scc::HashMap::with_hasher(FxBuildHasher::default()) }
    }
}

impl<K: WideColumn, V> Default for InMemorySingleMap<K, V> {
    fn default() -> Self { Self::new() }
}

impl<K: WideColumn, V: WideColumnValue<K>> SingleMap<K, V>
    for InMemorySingleMap<K, V>
{
    type WriteTransaction = FauxWriteBatch;

    async fn get(&self, key: &K::Key) -> Option<V> {
        self.map.read_sync(key, |_, value| value.clone())
    }

    async fn insert(
        &self,
        key: K::Key,
        value: V,
        _write_transaction: &mut Self::WriteTransaction,
    ) {
        self.map.upsert_sync(key, value);
    }

    async fn remove(
        &self,
        key: &K::Key,
        _write_transaction: &mut Self::WriteTransaction,
    ) {
        self.map.remove_sync(key);
    }
}
