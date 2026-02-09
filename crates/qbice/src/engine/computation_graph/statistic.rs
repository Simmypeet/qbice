use std::sync::atomic::AtomicUsize;

use crate::{
    TrackedEngine, config::Config, engine::computation_graph::ComputationGraph,
};

#[derive(Debug, Default)]
pub struct Statistic {
    dirtied_edge_count: AtomicUsize,
}

impl Statistic {
    pub(super) fn add_dirtied_edge_count(&self) {
        self.dirtied_edge_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

impl<C: Config> ComputationGraph<C> {
    pub(super) fn reset_statistic(&self) {
        self.statistic
            .dirtied_edge_count
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }
}

impl<C: Config> TrackedEngine<C> {
    /// Gets the number of dirtied edges in the current timestamp.
    #[must_use]
    pub fn get_dirtied_edges_count(&self) -> usize {
        self.engine
            .computation_graph
            .statistic
            .dirtied_edge_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}
