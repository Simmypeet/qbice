//! In-memory implementation of [`DynamicMap`].

use std::any::Any;

use fxhash::FxBuildHasher;

use crate::{
    dynamic_map::DynamicMap,
    kv_database::{WideColumn, WideColumnValue},
    write_batch::FauxWriteBatch,
};

/// An in-memory implementation of [`DynamicMap`] backed by a concurrent
/// [`scc::HashMap`].
///
/// This implementation stores all key-value pairs in memory, using type
/// erasure to support dynamic value types. Values are stored as boxed trait
/// objects and downcasted on retrieval.
#[derive(Debug)]
#[allow(clippy::type_complexity)]
pub struct InMemoryDynamicMap<K: WideColumn> {
    map: scc::HashMap<
        (K::Key, K::Discriminant),
        Box<dyn Any + Send + Sync>,
        FxBuildHasher,
    >,
}

impl<K: WideColumn> InMemoryDynamicMap<K> {
    /// Creates a new in-memory dynamic map with the specified hash builder.
    ///
    /// # Parameters
    ///
    /// - `hash_builder`: The hash builder to use for the underlying map.
    ///
    /// # Returns
    ///
    /// A new instance of `InMemoryDynamicMap`.
    #[must_use]
    pub fn new() -> Self {
        Self { map: scc::HashMap::with_hasher(FxBuildHasher::default()) }
    }
}

impl<K: WideColumn> Default for InMemoryDynamicMap<K> {
    fn default() -> Self { Self::new() }
}

impl<K: WideColumn> DynamicMap<K> for InMemoryDynamicMap<K> {
    type WriteTransaction = FauxWriteBatch;

    async fn get<V: WideColumnValue<K>>(&self, key: &K::Key) -> Option<V> {
        let discriminant = V::discriminant();

        self.map.read_sync(&(key.clone(), discriminant), |_, v| {
            v.downcast_ref::<V>().expect("should be the correct type").clone()
        })
    }

    async fn insert<V: WideColumnValue<K>>(
        &self,
        key: <K as WideColumn>::Key,
        value: V,
        _write_transaction: &mut Self::WriteTransaction,
    ) {
        let discriminant = V::discriminant();

        self.map.upsert_sync((key, discriminant), Box::new(value));
    }

    async fn remove<V: WideColumnValue<K>>(
        &self,
        key: &<K as WideColumn>::Key,
        _write_transaction: &mut Self::WriteTransaction,
    ) {
        let discriminant = V::discriminant();

        self.map.remove_sync(&(key.clone(), discriminant));
    }
}
