use bon::Builder;

use crate::{
    kv_database::{KvDatabase, KvDatabaseFactory},
    sharded::default_shard_amount,
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
/// let config = Configuration::builder().cache_capacity(10_000).build();
///
/// // Create storage engine with a RocksDB backend
/// let engine = DbBacked::new(rocksdb_instance, config);
/// let write_manager = engine.new_write_manager();
/// let map = write_manager.new_single_map::<MyColumn, MyValue>();
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DbBacked<Db> {
    backing_db: Db,
    configuration: Configuration,
}

impl<Db> DbBacked<Db> {
    /// Creates a new database-backed storage engine.
    ///
    /// # Parameters
    ///
    /// - `backing_db`: The initialized key-value database backend.
    /// - `configuration`: The configuration for the storage engine.
    #[must_use]
    pub const fn new(backing_db: Db, configuration: Configuration) -> Self {
        Self { backing_db, configuration }
    }
}

impl<Db: KvDatabase> StorageEngine for DbBacked<Db> {
    type WriteTransaction = write_behind::WriteBatch<Db>;

    type WriteManager = write_behind::WriteBehind<Db>;

    fn new_write_manager(&self) -> Self::WriteManager {
        write_behind::WriteBehind::new(
            &self.backing_db,
            self.configuration.cache_capacity,
        )
    }
}

/// A factory for creating `DbBacked` storage engines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Builder)]
pub struct DbBackedFactory<F> {
    /// The configuration for the storage engine.
    pub configuration: Configuration,

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

        Ok(DbBacked { backing_db: db, configuration: self.configuration })
    }
}
