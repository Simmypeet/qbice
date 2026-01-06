use parking_lot::RwLock;
use qbice_storage::sieve::WriteBuffer;
use rayon::prelude::ParallelIterator;

use crate::{
    Engine, ExecutionStyle, config::Config,
    engine::computation_graph::QueryKind, query::QueryID,
};

impl<C: Config> Engine<C> {
    pub(super) fn dirty_propagate(
        &self,
        query_id: QueryID,
        tx: &RwLock<&mut WriteBuffer<C::Database, C::BuildHasher>>,
    ) {
        // has already been marked dirty
        if !self.insert_dirty_query(query_id) {
            return;
        }

        let backward_edges =
            self.computation_graph.get_backward_edges(query_id);

        self.rayon_thread_pool.install(|| {
            backward_edges.par_iter().for_each(|id| {
                let caller_query_id = *id;

                self.mark_dirty_forward_edge(
                    caller_query_id,
                    query_id,
                    *tx.write(),
                );

                let query_kind = self
                    .computation_graph
                    .get_query_kind(caller_query_id)
                    .unwrap();

                // if this is a firewall node or projection node, then we stop
                // propagation here.
                if matches!(
                    query_kind,
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

    pub(super) fn clear_dirtied_queries(&self) {
        self.computation_graph.dirtied_queries.clear();
    }
}
