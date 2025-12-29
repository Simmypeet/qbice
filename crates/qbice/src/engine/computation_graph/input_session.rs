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
        self.engine.computation_graph.reset_statistic();
        self.engine.clear_dirtied_queries();

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
    /// Sets the value for an input query.
    ///
    /// This method updates the value associated with the given query. If the
    /// new value differs from the existing value (based on fingerprint
    /// comparison), the query and its dependents will be marked as dirty for
    /// recomputation.
    ///
    /// # Behavior
    ///
    /// - **New queries**: If the query has never been set before, a new entry
    ///   is created
    /// - **Changed values**: If the value fingerprint differs from the stored
    ///   value, dirty propagation is scheduled
    /// - **Timestamp management**: The first change in a session increments the
    ///   global timestamp
    ///
    /// All dirty propagation happens when the `InputSession` is dropped, not
    /// when this method is called.
    ///
    /// # Type Parameters
    ///
    /// - `Q`: The query type, must implement [`Query`]
    ///
    /// # Arguments
    ///
    /// - `query`: The input query key
    /// - `new_value`: The new value to associate with this query
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
