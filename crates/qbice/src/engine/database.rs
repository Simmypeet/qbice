use std::{collections::VecDeque, sync::Arc};

use dashmap::DashMap;

use crate::{
    config::Config,
    engine::{
        Engine, TrackedEngine,
        database::meta::{QueryMeta, QueryWithID, SetInputResult},
    },
    executor::CyclicQuery,
    query::{DynValue, Query, QueryID},
};

mod meta;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Timtestamp(u64);

impl Timtestamp {
    pub const fn increment(&mut self) { self.0 += 1; }
}

pub struct Database<C: Config> {
    query_metas: DashMap<QueryID, QueryMeta<C>>,

    initial_seed: u64,
    current_timestamp: Timtestamp,
}

impl<C: Config> Default for Database<C> {
    fn default() -> Self {
        Self {
            query_metas: DashMap::default(),
            initial_seed: 0,
            current_timestamp: Timtestamp(0),
        }
    }
}

impl<C: Config> Engine<C> {
    fn register_callee(&self, caller: Option<&QueryID>) {
        // record the dependency first, don't necessary need to figure out
        // the observed value fingerprint yet
        if let Some(caller) = caller {
            let caller_meta = self
                .database
                .query_metas
                .get(caller)
                .expect("caller query meta must exist");

            caller_meta.add_callee(*caller);
        }
    }

    fn is_in_scc(
        &self,
        called_from: Option<&QueryID>,
    ) -> Result<(), CyclicQuery> {
        let Some(called_from) = called_from else {
            return Ok(());
        };

        if self
            .database
            .query_metas
            .get(called_from)
            .expect("should be present")
            .is_running_in_scc()
        {
            return Err(CyclicQuery);
        }

        Ok(())
    }

    async fn query_for<Q: Query>(
        self: &Arc<Self>,
        query: &QueryWithID<'_, Q>,
        required_value: bool,
        caller: Option<&QueryID>,
    ) -> Result<Option<Q::Value>, CyclicQuery> {
        // register the dependency for the sake of detecting cycles
        self.register_callee(caller);

        // pulling the value
        let value = loop {
            match self
                .database
                .fast_path::<Q::Value>(query.id(), required_value, caller)
                .await?
            {
                // try again
                meta::FastPathResult::TryAgain => continue,

                // will continue down there
                meta::FastPathResult::ToSlowPath => {}

                meta::FastPathResult::Hit(value) => break value,
            }

            let (notify, continuation) = match self.database.slow_path(query) {
                // try the fast path again
                meta::SlowPathResult::TryAgain => continue,

                // proceed with the continuation
                meta::SlowPathResult::Continuation(notify, continuation) => {
                    (notify, continuation)
                }
            };

            // retry to the fast path and obtain value.
            self.continuation(query, &notify, continuation).await;
        };

        // check before returning the value
        self.is_in_scc(caller)?;

        Ok(value)
    }
}

impl<C: Config> TrackedEngine<C> {
    /// Query the database for a value.
    pub async fn query<Q: Query>(
        &self,
        query: &Q,
    ) -> Result<Q::Value, CyclicQuery> {
        let query_with_id = self.engine.database.new_query_with_id(query);

        // check local cache
        if let Some(value) = self.cache.get(query_with_id.id()) {
            // cache hit! don't have to go through central database
            let value: &dyn DynValue<C> = &**value;

            return Ok(value
                .downcast_value::<Q::Value>()
                .expect("should've been a correct type")
                .clone());
        }

        // run the main process
        self.engine
            .query_for(&query_with_id, true, self.caller.as_ref())
            .await
            .map(|x| x.unwrap())
    }
}

/// A struct allowing to set input values for queries.
#[derive(Debug)]
pub struct InputSession<'x, C: Config> {
    engine: &'x mut Engine<C>,
    incremented: bool,
    dirty_batch: VecDeque<QueryID>,
}

impl<C: Config> Engine<C> {
    /// Create a [`InputSession`] allowing to set input values for queries.
    #[must_use]
    pub const fn input_session(&mut self) -> InputSession<'_, C> {
        InputSession {
            engine: self,
            incremented: false,
            dirty_batch: VecDeque::new(),
        }
    }
}

impl<C: Config> InputSession<'_, C> {
    /// Set an input value for a query.
    pub fn set_input<Q: Query>(&mut self, query_key: Q, value: Q::Value) {
        let query_id = self.engine.database.query_id(&query_key);

        let SetInputResult { incremented, fingerprint_diff } = self
            .engine
            .database
            .set_input(query_key, query_id, value, self.incremented);

        self.incremented |= incremented;

        if fingerprint_diff {
            // insert into dirty batch
            self.dirty_batch.push_back(query_id);
        }
    }
}

impl<C: Config> Drop for InputSession<'_, C> {
    fn drop(&mut self) {
        // mark all dirty queries as dirty
        self.engine
            .database
            .dirty_queries(std::mem::take(&mut self.dirty_batch));
    }
}
