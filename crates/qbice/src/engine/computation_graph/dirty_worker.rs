use std::sync::{Arc, Weak};

use crossbeam::utils::CachePadded;
use dashmap::DashSet;
use tokio::sync::{
    Mutex,
    mpsc::{UnboundedReceiver, UnboundedSender},
};

use crate::{
    Engine, ExecutionStyle,
    config::{Config, WriteTransaction},
    engine::computation_graph::{
        QueryKind,
        database::Database,
        dirty_worker::task::{Batch, DirtyTask},
        statistic::Statistic,
    },
    query::QueryID,
};

mod task;

pub enum Message<C: Config> {
    DirtyTask(DirtyTask<C>),
    Shutdown,
}

/// Distributed dirty propagation worker pool.
pub struct DirtyWorker<C: Config> {
    shards: Arc<[CachePadded<UnboundedSender<Message<C>>>]>,
    mask: usize,
}

impl<C: Config> DirtyWorker<C> {
    pub fn new(
        database: &Arc<Database<C>>,
        stats: &Arc<Statistic>,
        dirtied_queries: &Arc<DashSet<QueryID, C::BuildHasher>>,
    ) -> Self {
        let parallelism = std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(8)
            .next_power_of_two();

        let mask = parallelism - 1;
        let mut shards = Vec::with_capacity(parallelism);
        let mut receivers = Vec::with_capacity(parallelism);

        // create worker tasks
        for _ in 0..parallelism {
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
            shards.push(CachePadded::new(tx));
            receivers.push(rx);
        }

        let shards: Arc<[_]> = Arc::from(shards);

        // spawn worker tasks
        for receiver in receivers {
            let database = Arc::downgrade(database);
            let stats = Arc::downgrade(stats);
            let dirtied_queries = Arc::downgrade(dirtied_queries);
            let shards = shards.clone();

            tokio::spawn(async move {
                Self::worker_loop(
                    &database,
                    &stats,
                    &dirtied_queries,
                    &shards,
                    mask,
                    receiver,
                )
                .await;
            });
        }

        Self { shards, mask }
    }

    #[allow(clippy::cast_possible_truncation)]
    pub fn submit_task(&self, dirty_task: DirtyTask<C>) {
        let shard_idx = dirty_task.determine_shard(self.mask);

        // submit to the appropriate shard
        self.shards[shard_idx].send(Message::DirtyTask(dirty_task)).unwrap();
    }

    #[allow(clippy::cast_possible_truncation)]
    fn submit_task_from(
        shards: &[CachePadded<UnboundedSender<Message<C>>>],
        mask: usize,
        dirty_task: DirtyTask<C>,
    ) {
        let shard_idx = dirty_task.determine_shard(mask);

        // submit to the appropriate shard
        shards[shard_idx].send(Message::DirtyTask(dirty_task)).unwrap();
    }

    async fn worker_loop(
        database: &Weak<Database<C>>,
        statistic: &Weak<Statistic>,
        dirtied_queries: &Weak<DashSet<QueryID, C::BuildHasher>>,
        shards: &[CachePadded<UnboundedSender<Message<C>>>],
        mask: usize,
        mut receiver: UnboundedReceiver<Message<C>>,
    ) {
        while let Some(task) = receiver.recv().await {
            let task = match task {
                Message::DirtyTask(task) => task,
                Message::Shutdown => break,
            };

            let dirtied_queries = dirtied_queries.upgrade().unwrap();
            let statistic = statistic.upgrade().unwrap();
            let database = database.upgrade().unwrap();

            if !dirtied_queries.insert(*task.query_id()) {
                continue;
            }

            for caller in unsafe {
                database.get_backward_edges_unchecked(task.query_id()).await
            } {
                database
                    .mark_dirty_forward_edge(
                        caller,
                        *task.query_id(),
                        &mut *task.write_tx_lock().await,
                    )
                    .await;
                statistic.add_dirtied_edge_count();

                let query_kind = database.get_query_kind(&caller).await;

                if matches!(
                    query_kind,
                    QueryKind::Executable(
                        ExecutionStyle::Projection | ExecutionStyle::Firewall
                    )
                ) {
                    // dont continue propagation through firewall or projection
                    // nodes
                    continue;
                }

                Self::submit_task_from(shards, mask, task.propagate_to(caller));
            }
        }
    }
}

impl<C: Config> Drop for DirtyWorker<C> {
    fn drop(&mut self) {
        for shard in self.shards.iter() {
            let _ = shard.send(Message::Shutdown);
        }
    }
}

impl<C: Config> Engine<C> {
    pub(super) async fn dirty_propagate_from_batch(
        self: &Arc<Self>,
        query_id: impl IntoIterator<Item = QueryID>,
        trasnaction: WriteTransaction<C>,
    ) -> WriteTransaction<C> {
        let write_tx = Arc::new(Mutex::new(trasnaction));

        let batch = Batch::new(write_tx.clone());
        let notified = batch.notified_owned();

        for query_id in query_id {
            self.computation_graph
                .dirty_worker
                .submit_task(batch.new_task(query_id));
        }

        drop(batch);

        // wait for all tasks to complete
        notified.await;

        Arc::try_unwrap(write_tx)
            .unwrap_or_else(|_| {
                panic!("should be unique, notified system is broken")
            })
            .into_inner()
    }

    pub(super) fn clear_dirtied_queries(&self) {
        self.computation_graph.dirtied_queries.clear();
    }
}
