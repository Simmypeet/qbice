#![allow(missing_docs)]

use qbice::{Engine, config::DefaultConfig};
use qbice_serialize::Plugin;
use qbice_stable_hash::{SeededStableHasherBuilder, Sip128Hasher};
use qbice_storage::kv_database::rocksdb::RocksDB;
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
