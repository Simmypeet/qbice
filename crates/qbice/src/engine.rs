//! QBICE Engine module, responsible for managing query execution and storage.

use std::sync::Arc;

use dashmap::DashMap;

use crate::{
    config::{Config, DefaultConfig},
    engine::database::{Caller, Database},
    executor::{Executor, Registry},
    query::{DynValueBox, Query, QueryID},
};

mod database;
mod meta;

/// The main central database engine for managing query execution and storage.
///
/// Mainly you would use [`Arc<Engine<C>>`] to create an instance of the engine
/// and [`TrackedEngine`] if you want to start querying for values.
///
/// With the [`&mut Engine`] you can register executors and mutate query inputs
/// but not query the database.
pub struct Engine<C: Config> {
    database: Database<C>,
    executor_registry: Registry<C>,
}

impl<C: Config> Engine<C> {
    /// Registers a new executor for the given query type.
    pub fn register_executor<Q: Query, E: Executor<Q, C>>(
        &mut self,
        executor: Arc<E>,
    ) {
        self.executor_registry.register(executor);
    }
}

static_assertions::assert_impl_all!(Engine<DefaultConfig>: Send, Sync);

impl<C: Config> Engine<C> {
    /// Creates a new instance of the engine.
    #[must_use]
    pub fn new() -> Self {
        Self {
            database: Database::default(),
            executor_registry: Registry::default(),
        }
    }

    /// Creates a tracked engine wrapper for querying the database.
    #[must_use]
    pub fn tracked(self: Arc<Self>) -> TrackedEngine<C> {
        TrackedEngine {
            engine: self,
            cache: Arc::new(DashMap::new()),
            caller: None,
        }
    }
}

impl<C: Config> Default for Engine<C> {
    fn default() -> Self { Self::new() }
}

impl<C: Config> std::fmt::Debug for Engine<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine").finish_non_exhaustive()
    }
}

/// A wrapper around [`Arc<Engine>`] allowing query for the database.
///
/// The struct is very cheap to clone and can be sent to multiple thread for
/// concurrent query execution.
pub struct TrackedEngine<C: Config> {
    engine: Arc<Engine<C>>,
    cache: Arc<DashMap<QueryID, DynValueBox<C>>>,
    caller: Option<Caller>,
}

impl<C: Config> std::fmt::Debug for TrackedEngine<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrackedEngine")
            .field("engine", &self.engine)
            .field("caller", &self.caller)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod test;
