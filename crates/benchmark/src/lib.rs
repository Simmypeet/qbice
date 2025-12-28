#![allow(missing_docs)]

use qbice::{
    Engine,
    config::DefaultConfig,
    serialize::Plugin,
    stable_hash::{SeededStableHasherBuilder, Sip128Hasher},
    storage::kv_database::rocksdb::RocksDB,
};
use tempfile::TempDir;

#[must_use]
pub fn create_test_engine(tempdir: &TempDir) -> Engine<DefaultConfig> {
    Engine::<DefaultConfig>::new_with(
        Plugin::default(),
        RocksDB::factory(tempdir.path()),
        SeededStableHasherBuilder::<Sip128Hasher>::new(0),
    )
    .unwrap()
}
