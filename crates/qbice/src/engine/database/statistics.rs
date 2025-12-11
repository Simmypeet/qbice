use std::sync::atomic::AtomicUsize;

use crate::{TrackedEngine, config::Config, engine::database::Database};

#[derive(Default)]
pub struct Statistics {
    dirtied_edges: AtomicUsize,
}

impl<C: Config> Database<C> {
    pub fn clear_statistics(&mut self) {
        self.statistics = Statistics::default();
    }

    pub fn increment_dirtied_edges(&self) {
        self.statistics
            .dirtied_edges
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
}

impl<C: Config> TrackedEngine<C> {
    /// Returns the number of dirtied edges recorded in the database statistics.
    #[must_use]
    pub fn get_dirtied_edges_count(&self) -> usize {
        self.engine
            .database
            .statistics
            .dirtied_edges
            .load(std::sync::atomic::Ordering::SeqCst)
    }
}
