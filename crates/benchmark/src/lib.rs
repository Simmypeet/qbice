#![allow(missing_docs)]

use std::{hash::BuildHasherDefault, path::Path};

use fxhash::FxHasher;
use qbice::{
    Engine, Identifiable, config,
    serialize::Plugin,
    stable_hash::{SeededStableHasherBuilder, Sip128Hasher},
    storage::{
        kv_database::rocksdb::RocksDB,
        storage_engine::{
            db_backed::{Configuration, DbBacked, DbBackedFactory},
            in_memory::{InMemoryStorageEngine, InMemoryStorageEngineFactory},
        },
    },
};

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Default,
    Identifiable,
)]
pub struct Config;

impl config::Config for Config {
    type StorageEngine = InMemoryStorageEngine;

    type BuildStableHasher = SeededStableHasherBuilder<Sip128Hasher>;

    type BuildHasher = BuildHasherDefault<FxHasher>;
}

#[must_use]
pub async fn create_test_engine() -> Engine<Config> {
    Engine::<Config>::new_with(
        Plugin::default(),
        InMemoryStorageEngineFactory,
        SeededStableHasherBuilder::<Sip128Hasher>::new(0),
    )
    .await
    .unwrap()
}

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Default,
    Identifiable,
)]
pub struct RocksDbConfig;

impl config::Config for RocksDbConfig {
    type StorageEngine = DbBacked<RocksDB>;

    type BuildStableHasher = SeededStableHasherBuilder<Sip128Hasher>;

    type BuildHasher = BuildHasherDefault<FxHasher>;
}

/// Opens (or creates) a RocksDB-backed engine at `path` whose in-memory
/// caches hold at most `cache_capacity` entries each.
#[must_use]
pub async fn create_rocksdb_engine(
    path: &Path,
    cache_capacity: u64,
) -> Engine<RocksDbConfig> {
    Engine::<RocksDbConfig>::new_with(
        Plugin::default(),
        DbBackedFactory::builder()
            .configuration(
                Configuration::builder().cache_capacity(cache_capacity).build(),
            )
            .db_factory(RocksDB::factory(path))
            .build(),
        SeededStableHasherBuilder::<Sip128Hasher>::new(0),
    )
    .await
    .unwrap()
}
