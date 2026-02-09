use std::sync::{Arc, atomic::AtomicUsize};

use crossbeam::utils::CachePadded;
use dashmap::DashSet;
use tokio::sync::{
    Mutex, Notify,
    mpsc::{Receiver, Sender},
};

use crate::{
    Engine, ExecutionStyle,
    config::{Config, WriteTransaction},
    engine::computation_graph::{
        QueryKind, database::Database, statistic::Statistic,
    },
    query::QueryID,
};

pub struct WorkTracker {
    active_task_count: AtomicUsize,
    notify: Arc<Notify>,
}

impl WorkTracker {
    pub fn done(&self) {
        let count = self
            .active_task_count
            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);

        if count == 1 {
            self.notify.notify_waiters();
        }
    }

    pub fn new_task(&self) {
        self.active_task_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

pub enum Message<C: Config> {
    DirtyTask(DirtyTask<C>),
    Shutdown,
}

pub struct DirtyTask<C: Config> {
    to: QueryID,
    from: Option<QueryID>,

    write_tx: Arc<Mutex<WriteTransaction<C>>>,
    work_tracker: Arc<WorkTracker>,
}

/// Distributed dirty propagation worker pool.
pub struct DirtyWorker<C: Config> {
    shards: Arc<[CachePadded<Sender<Message<C>>>]>,
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
            let (tx, rx) = tokio::sync::mpsc::channel(1024);
            shards.push(CachePadded::new(tx));
            receivers.push(rx);
        }

        let shards: Arc<[_]> = Arc::from(shards);

        // spawn worker tasks
        for receiver in receivers {
            let database = database.clone();
            let stats = stats.clone();
            let dirtied_queries = dirtied_queries.clone();
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
    pub async fn submit_task(&self, dirty_task: DirtyTask<C>) {
        let shard_idx =
            dirty_task.to.compact_hash_128().low() as usize & self.mask;

        // submit to the appropriate shard
        self.shards[shard_idx]
            .send(Message::DirtyTask(dirty_task))
            .await
            .unwrap();
    }

    #[allow(clippy::cast_possible_truncation)]
    async fn submit_task_from(
        shards: &[CachePadded<Sender<Message<C>>>],
        mask: usize,
        dirty_task: DirtyTask<C>,
    ) {
        let shard_idx = dirty_task.to.compact_hash_128().low() as usize & mask;

        // submit to the appropriate shard
        shards[shard_idx].send(Message::DirtyTask(dirty_task)).await.unwrap();
    }

    async fn worker_loop(
        database: &Database<C>,
        statistic: &Statistic,
        dirtied_queries: &DashSet<QueryID, C::BuildHasher>,
        shards: &[CachePadded<Sender<Message<C>>>],
        mask: usize,
        mut receiver: Receiver<Message<C>>,
    ) {
        while let Some(task) = receiver.recv().await {
            let task = match task {
                Message::DirtyTask(task) => task,
                Message::Shutdown => break,
            };

            // insert into the dirtied set to prevent duplicate work
            if !dirtied_queries.insert(task.to) {
                task.work_tracker.done();
                continue;
            }

            // request to mark the edge as dirty
            if let Some(from) = task.from {
                statistic.add_dirtied_edge_count();
                database
                    .mark_dirty_forward_edge(
                        from,
                        task.to,
                        &mut *task.write_tx.lock().await,
                    )
                    .await;
            }

            let backward_edges = unsafe {
                database.get_backward_edges_unchecked(&task.to).await
            };

            // if this firewall or projection node, then we stop
            let query_kind = database.get_query_kind(&task.to).await;

            // don't propagate further
            if matches!(
                query_kind,
                QueryKind::Executable(
                    ExecutionStyle::Firewall | ExecutionStyle::Projection
                )
            ) {
                task.work_tracker.done();
                continue;
            }

            // propagate to callers
            for caller in backward_edges {
                let work_tracker = task.work_tracker.clone();
                let write_tx = task.write_tx.clone();

                work_tracker.new_task();

                Self::submit_task_from(shards, mask, DirtyTask {
                    to: caller,
                    from: Some(task.to),
                    write_tx: write_tx.clone(),
                    work_tracker: work_tracker.clone(),
                })
                .await;
            }

            // done with this task
            task.work_tracker.done();
        }
    }
}

impl<C: Config> Drop for DirtyWorker<C> {
    fn drop(&mut self) {
        for shard in self.shards.iter() {
            let _ = shard.blocking_send(Message::Shutdown);
        }
    }
}

impl<C: Config> Engine<C> {
    pub(super) async fn dirty_propagate_from_batch(
        self: &Arc<Self>,
        query_id: impl IntoIterator<Item = QueryID>,
        trasnaction: WriteTransaction<C>,
    ) -> WriteTransaction<C> {
        let work_tracker = Arc::new(WorkTracker {
            active_task_count: AtomicUsize::new(0),
            notify: Arc::new(Notify::new()),
        });
        let write_tx = Arc::new(Mutex::new(trasnaction));
        let notified = work_tracker.notify.clone().notified_owned();

        for query_id in query_id {
            let work_tracker = work_tracker.clone();
            let write_tx = write_tx.clone();

            work_tracker.new_task();

            self.computation_graph
                .dirty_worker
                .submit_task(DirtyTask {
                    to: query_id,
                    from: None,
                    write_tx: write_tx.clone(),
                    work_tracker: work_tracker.clone(),
                })
                .await;
        }

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
