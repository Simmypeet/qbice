use bon::Builder;

use crate::{
    dynamic_map::cache::CacheDynamicMap,
    key_of_set_map::{ConcurrentSet, cache::CacheKeyOfSetMap},
    kv_database::{
        KeyOfSetColumn, KvDatabase, KvDatabaseFactory, WideColumn,
        WideColumnValue,
    },
    sharded::default_shard_amount,
    single_map::cache::CacheSingleMap,
    storage_engine::{StorageEngine, StorageEngineFactory},
    write_manager::write_behind,
};

/// Configuration options for a database-backed storage engine.
///
/// This struct holds the configuration parameters used when creating
/// storage components like caches and write managers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Builder)]
pub struct Configuration {
    /// The maximum capacity of the cache in number of entries.
    ///
    /// Higher values allow more data to be cached in memory, reducing
    /// database reads but increasing memory usage.
    #[builder(default = 2u64.pow(18))]
    pub cache_capacity: u64,

    /// The number of worker threads for serialization and write processing.
    ///
    /// More workers can increase throughput for write-heavy workloads,
    /// but returns diminish beyond the database's I/O capacity.
    #[builder(default = 2)]
    pub serialization_workers: usize,

    /// The default number of shards to use for caches.
    ///
    /// More shards can improve concurrency but increase memory overhead.
    #[builder(default = default_shard_amount())]
    pub default_shard_amount: usize,
}

/// A database-backed storage engine with caching and write-behind support.
///
/// This storage engine wraps a key-value database backend and provides:
/// - Caching via Moka for frequently accessed data
/// - Write-behind buffering for improved write performance
/// - Automatic cache invalidation on writes
///
/// # Type Parameters
///
/// - `Db`: The key-value database backend implementing [`KvDatabase`].
///
/// # Example
///
/// ```ignore
/// use qbice_storage::storage_engine::db_backed::{DbBacked, Configuration};
///
/// let config = Configuration::builder()
///     .cache_capacity(10_000)
///     .serialization_workers(4)
///     .build();
///
/// // Create storage engine with a RocksDB backend
/// let engine = DbBacked::new(rocksdb_instance, config);
/// let map = engine.new_single_map::<MyColumn, MyValue>();
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DbBacked<Db> {
    backing_db: Db,
    configuration: Configuration,
}

impl<Db: KvDatabase> StorageEngine for DbBacked<Db> {
    type WriteTransaction = write_behind::WriteBatch<Db>;

    type WriteManager = write_behind::WriteBehind<Db>;

    type SingleMap<K: WideColumn, V: WideColumnValue<K>> =
        CacheSingleMap<K, V, Db>;

    type DynamicMap<K: WideColumn> = CacheDynamicMap<K, Db>;

    type KeyOfSetMap<
        K: KeyOfSetColumn,
        C: ConcurrentSet<Element = K::Element>,
    > = CacheKeyOfSetMap<K, C, Db>;

    fn new_write_manager(&self) -> Self::WriteManager {
        write_behind::WriteBehind::new(
            &self.backing_db,
            self.configuration.serialization_workers,
        )
    }

    fn new_single_map<K: WideColumn, V: WideColumnValue<K>>(
        &self,
    ) -> Self::SingleMap<K, V> {
        CacheSingleMap::new(
            self.configuration.cache_capacity,
            self.configuration.default_shard_amount,
            self.backing_db.clone(),
        )
    }

    fn new_dynamic_map<K: WideColumn>(&self) -> Self::DynamicMap<K> {
        CacheDynamicMap::new(
            self.configuration.cache_capacity,
            self.configuration.default_shard_amount,
            self.backing_db.clone(),
        )
    }

    fn new_key_of_set_map<
        K: KeyOfSetColumn,
        C: ConcurrentSet<Element = K::Element>,
    >(
        &self,
    ) -> Self::KeyOfSetMap<K, C> {
        CacheKeyOfSetMap::new(
            self.configuration.cache_capacity,
            self.configuration.default_shard_amount,
            self.backing_db.clone(),
        )
    }
}

/// A factory for creating `DbBacked` storage engines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Builder)]
pub struct DbBackedFactory<F> {
    /// The configuration for the storage engine.
    pub coniguration: Configuration,

    /// The database factory for creating the backing database.
    pub db_factory: F,
}

impl<F: KvDatabaseFactory> StorageEngineFactory for DbBackedFactory<F> {
    type StorageEngine = DbBacked<F::KvDatabase>;

    type Error = F::Error;

    fn open(
        self,
        serialization_plugin: qbice_serialize::Plugin,
    ) -> Result<Self::StorageEngine, Self::Error> {
        let db = self.db_factory.open(serialization_plugin)?;

        Ok(DbBacked { backing_db: db, configuration: self.coniguration })
    }
}
