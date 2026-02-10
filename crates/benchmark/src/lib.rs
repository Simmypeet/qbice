#![allow(missing_docs)]

use std::hash::BuildHasherDefault;

use fxhash::FxHasher;
use qbice::{
    Engine, Identifiable, config,
    serialize::Plugin,
    stable_hash::{SeededStableHasherBuilder, Sip128Hasher},
    storage::storage_engine::in_memory::{
        InMemoryStorageEngine, InMemoryStorageEngineFactory,
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
