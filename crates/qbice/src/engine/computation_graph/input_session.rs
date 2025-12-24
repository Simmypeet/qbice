use std::{collections::VecDeque, sync::Arc};

use qbice_storage::kv_database::{KvDatabase, WriteTransaction};

use crate::{
    Engine, Query,
    config::Config,
    engine::computation_graph::{
        QueryKind,
        computed::{
            ForwardEdgeColumn, LastVerifiedColumn, NodeInfo, NodeInfoColumn,
        },
        query_store::{QueryColumn, QueryEntry},
    },
    query::QueryID,
};

pub struct InputSession<'x, C: Config> {
    engine: &'x Engine<C>,
    incremented: bool,
    dirty_batch: VecDeque<QueryID>,
    transaction: <C::Database as KvDatabase>::WriteTransaction<'x>,
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
            transaction: self.database.write_transaction(),
            engine: self,
        }
    }
}

impl<C: Config> InputSession<'_, C> {
    pub fn set_input<Q: Query>(&mut self, query: Q, new_value: Q::Value) {
        let query_hash = self.engine.hash(&query);
        let query_id = QueryID::new::<Q>(query_hash);

        let query_value_fingerprint = self.engine.hash(&new_value);
        let transitive_firewall_callees = None;
        let transitive_firewall_callees_fingerprint =
            self.engine.hash(&transitive_firewall_callees);

        let node_info = NodeInfo::new(
            QueryKind::Input,
            query_value_fingerprint,
            transitive_firewall_callees_fingerprint,
            transitive_firewall_callees,
        );

        let empty_edges: Arc<[QueryID]> = Arc::from([]);
        let query_entry = QueryEntry::new(query, new_value);

        if let Some(node_info) =
            self.engine.computation_graph.node_info().get_normal(&query_id)
        {
            let fingerprint_diff =
                node_info.value_fingerprint() != query_value_fingerprint;

            if fingerprint_diff && !self.incremented {
                self.engine
                    .computation_graph
                    .timestamp_manager
                    .increment(&self.transaction);

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
                    .increment(&self.transaction);

                self.incremented = true;
            }
        }

        let timestamp =
            self.engine.computation_graph.timestamp_manager.get_current();

        // durable, write to db first
        {
            self.transaction.put::<LastVerifiedColumn>(&query_id, &timestamp);

            self.transaction.put::<ForwardEdgeColumn>(&query_id, &empty_edges);

            self.transaction.put::<NodeInfoColumn>(&query_id, &node_info);

            self.transaction.put::<QueryColumn<Q>>(
                &query_id.hash_128().into(),
                &query_entry,
            );
        }

        // write to in-memory structures
        {
            self.engine
                .computation_graph
                .last_verifieds()
                .put(query_id, Some(timestamp));

            self.engine
                .computation_graph
                .forward_edges()
                .put(query_id, Some(empty_edges));

            self.engine
                .computation_graph
                .node_info()
                .put(query_id, Some(node_info));

            self.engine
                .computation_graph
                .query_store
                .insert(query_id.hash_128().into(), query_entry);
        }
    }
}
