use std::collections::VecDeque;

use qbice_storage::kv_database::{KvDatabase, WriteTransaction};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{Engine, Query, config::Config, query::QueryID};

pub struct InputSession<'x, C: Config> {
    engine: &'x Engine<C>,
    incremented: bool,
    dirty_batch: VecDeque<QueryID>,
    transaction: Option<<C::Database as KvDatabase>::WriteTransaction<'x>>,
}

impl<C: Config> Drop for InputSession<'_, C> {
    fn drop(&mut self) {
        let tx = self.transaction.take().unwrap();

        self.engine.rayon_thread_pool.install(|| {
            self.dirty_batch.par_iter().for_each(|x| {
                self.engine.dirty_propagate(*x, &tx);
            });
        });

        tx.commit();
    }
}

impl<C: Config> std::fmt::Debug for InputSession<'_, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InputSession")
            .field("engine", self.engine)
            .field("incremented", &self.incremented)
            .field("dirty_batch", &self.dirty_batch)
            .finish_non_exhaustive()
    }
}

impl<C: Config> Engine<C> {
    /// Creates an input session for setting query input values.
    ///
    /// The returned session allows you to set values for input queries. When
    /// the session is dropped, dirty propagation is triggered for any changed
    /// inputs.
    ///
    /// # Example
    ///
    /// ```rust
    /// use qbice::{
    ///     Identifiable, StableHash, config::DefaultConfig, engine::Engine,
    ///     query::Query,
    /// };
    ///
    /// #[derive(Debug, Clone, PartialEq, Eq, Hash, StableHash, Identifiable)]
    /// struct Input(u64);
    /// impl Query for Input {
    ///     type Value = i64;
    /// }
    ///
    /// let mut engine = Engine::<DefaultConfig>::new();
    ///
    /// // Create a session and set inputs /// { ///     let mut session = engine.input_session(); ///     session.set_input(Input(0), 42);
    ///     session.set_input(Input(1), 100);
    /// ```
    #[must_use]
    pub fn input_session(&mut self) -> InputSession<'_, C> {
        InputSession {
            incremented: false,
            dirty_batch: VecDeque::new(),
            transaction: Some(self.database.write_transaction()),
            engine: self,
        }
    }
}

impl<C: Config> InputSession<'_, C> {
    pub fn set_input<Q: Query>(&mut self, query: Q, new_value: Q::Value) {
        let query_hash = self.engine.hash(&query);
        let query_id = QueryID::new::<Q>(query_hash);

        let query_value_fingerprint = self.engine.hash(&new_value);

        // has prior node infos, check for fingerprint diff
        // also, unwire the backward edges (if any)
        if let Some(node_info) =
            self.engine.computation_graph.get_node_info(&query_id)
        {
            let fingerprint_diff =
                node_info.value_fingerprint() != query_value_fingerprint;

            if fingerprint_diff && !self.incremented {
                self.engine
                    .computation_graph
                    .timestamp_manager
                    .increment(self.transaction.as_ref().unwrap());

                self.incremented = true;
            }

            if fingerprint_diff {
                self.dirty_batch.push_back(query_id);
            }
        } else {
            // if the query does not exist yet, we need to create a new node
            // info
            if !self.incremented {
                self.engine
                    .computation_graph
                    .timestamp_manager
                    .increment(self.transaction.as_ref().unwrap());

                self.incremented = true;
            }
        }

        self.engine.set_computed_input(
            query,
            query_hash,
            new_value,
            query_value_fingerprint,
            self.transaction.as_ref().unwrap(),
        );
    }
}
