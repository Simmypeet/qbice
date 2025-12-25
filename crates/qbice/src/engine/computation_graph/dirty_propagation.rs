use qbice_storage::kv_database::KvDatabase;
use rayon::prelude::ParallelIterator;

use crate::{
    Engine, ExecutionStyle,
    config::{self, Config},
    engine::computation_graph::QueryKind,
    query::QueryID,
};

impl<C: Config> Engine<C> {
    pub(super) fn dirty_propagate(
        &self,
        query_id: QueryID,
        tx: &<C::Database as KvDatabase>::WriteTransaction<'_>,
    ) {
        // has already been marked dirty
        if !self.insert_dirty_query(query_id) {
            return;
        }

        let backward_edges =
            self.computation_graph.get_backward_edges(&query_id);

        self.rayon_thread_pool.install(|| {
            backward_edges.par_iter().for_each(|id| {
                let caller_query_id = *id;

                self.mark_dirty_forward_edge(caller_query_id, query_id, tx);

                let node_info = self
                    .computation_graph
                    .get_node_info(&caller_query_id)
                    .unwrap();

                // if this is a firewall node or projection node, then we stop
                // propagation here.
                if matches!(
                    node_info.query_kind(),
                    QueryKind::Executable(
                        ExecutionStyle::Firewall | ExecutionStyle::Projection
                    )
                ) {
                    return;
                }

                self.dirty_propagate(caller_query_id, tx);
            });
        });
    }

    pub(super) fn insert_dirty_query(&self, query_id: QueryID) -> bool {
        self.computation_graph.dirtied_queries.insert(query_id)
    }
}
