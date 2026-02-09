use std::{
    sync::{
        Arc, Weak,
        atomic::{AtomicBool, Ordering},
    },
    thread,
};

use crossbeam::{
    deque::{Injector, Stealer, Worker},
    utils::Backoff,
};
use dashmap::DashSet;
use tokio::sync::{Mutex, Notify};

use crate::{
    Engine, ExecutionStyle,
    config::{Config, WriteTransaction},
    engine::computation_graph::{
        QueryKind,
        database::{Database, Edge},
        dirty_worker::task::{Batch, DirtyTask, StrippedBuffer},
        statistic::Statistic,
    },
    query::QueryID,
};

mod task;

/// Work-stealing dirty propagation worker pool.
///
/// Uses a global [`Injector`] queue paired with per-worker local
/// deques. Workers batch-steal from the global queue to amortize
/// synchronization overhead, and can steal from sibling workers' local
/// deques when both the local deque and global queue are empty.
pub struct DirtyWorker<C: Config> {
    injector: Arc<Injector<DirtyTask<C>>>,
    notify: Arc<Notify>,
    shutdown: Arc<AtomicBool>,
}

impl<C: Config> DirtyWorker<C> {
    pub fn new(
        database: &Arc<Database<C>>,
        stats: &Arc<Statistic>,
        dirtied_queries: &Arc<DashSet<QueryID, C::BuildHasher>>,
    ) -> Self {
        let parallelism = thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(8);

        let injector = Arc::new(Injector::new());
        let notify = Arc::new(Notify::new());
        let shutdown = Arc::new(AtomicBool::new(false));

        // Create per-worker local deques and their stealers.
        let mut workers = Vec::with_capacity(parallelism);
        let mut stealers = Vec::with_capacity(parallelism);

        for _ in 0..parallelism {
            let w = Worker::new_fifo();
            stealers.push(w.stealer());
            workers.push(w);
        }

        let stealers: Arc<[Stealer<DirtyTask<C>>]> = Arc::from(stealers);

        // Spawn worker tasks.
        for worker in workers {
            let injector = injector.clone();
            let stealers = stealers.clone();
            let notify = notify.clone();
            let shutdown = shutdown.clone();
            let database = Arc::downgrade(database);
            let stats = Arc::downgrade(stats);
            let dirtied_queries = Arc::downgrade(dirtied_queries);

            tokio::spawn(async move {
                Self::worker_loop(
                    worker,
                    &injector,
                    &stealers,
                    &notify,
                    &shutdown,
                    &database,
                    &stats,
                    &dirtied_queries,
                )
                .await;
            });
        }

        Self { injector, notify, shutdown }
    }

    /// Submit a dirty-propagation task to the global injector queue.
    pub fn submit_task(&self, dirty_task: DirtyTask<C>) {
        self.injector.push(dirty_task);
        self.notify.notify_one();
    }

    /// Attempt to find a task using the three-tier strategy:
    /// 1. Pop from the thread-local deque (zero contention).
    /// 2. Batch-steal from the global [`Injector`].
    /// 3. Steal from a sibling worker's deque.
    fn find_task(
        local: &Worker<DirtyTask<C>>,
        injector: &Injector<DirtyTask<C>>,
        stealers: &[Stealer<DirtyTask<C>>],
    ) -> Option<DirtyTask<C>> {
        // Pop a task from the local queue, if not empty.
        local.pop().or_else(|| {
            // Otherwise, we need to look for a task elsewhere.
            std::iter::repeat_with(|| {
                // Try stealing a batch of tasks from the global queue.
                injector
                    .steal_batch_and_pop(local)
                    // Or try stealing a task from one of the other threads.
                    .or_else(|| {
                        stealers
                            .iter()
                            .map(crossbeam::deque::Stealer::steal)
                            .collect()
                    })
            })
            // Loop while no task was stolen and any steal operation needs to be
            // retried.
            .find(|s| !s.is_retry())
            // Extract the stolen task, if there is one.
            .and_then(crossbeam::deque::Steal::success)
        })
    }

    #[allow(clippy::too_many_arguments)]
    async fn worker_loop(
        local: Worker<DirtyTask<C>>,
        injector: &Injector<DirtyTask<C>>,
        stealers: &[Stealer<DirtyTask<C>>],
        notify: &Notify,
        shutdown: &AtomicBool,
        database: &Weak<Database<C>>,
        statistic: &Weak<Statistic>,
        dirtied_queries: &Weak<DashSet<QueryID, C::BuildHasher>>,
    ) {
        let backoff = Backoff::new();
        loop {
            // Drain all available work before parking.
            let mut count = 0;
            while let Some(task) = Self::find_task(&local, injector, stealers) {
                Self::process_task(
                    task,
                    injector,
                    notify,
                    database,
                    statistic,
                    dirtied_queries,
                )
                .await;

                // work was found, reset backoff
                backoff.reset();

                // every 32 tasks, yield to allow other tasks to run
                count += 1;
                if count >= 32 {
                    count = 0;
                    tokio::task::yield_now().await;
                }
            }

            if shutdown.load(Ordering::Acquire) {
                break;
            }

            // Prepare to park – register *before* the final emptiness
            // check so that a concurrent push + notify is never lost.
            let notified = if backoff.is_completed() {
                Some(notify.notified())
            } else {
                None
            };

            // Double-check: work may have arrived between the while-
            // loop exit and the enable() call above.
            if let Some(task) = Self::find_task(&local, injector, stealers) {
                Self::process_task(
                    task,
                    injector,
                    notify,
                    database,
                    statistic,
                    dirtied_queries,
                )
                .await;

                // work was found, reset backoff
                continue;
            }

            if shutdown.load(Ordering::Acquire) {
                break;
            }

            match notified {
                Some(notified) => {
                    // Backoff is complete, park until notified.
                    notified.await;
                    backoff.reset();
                }
                None => {
                    // Backoff is not complete, spin-wait.
                    backoff.snooze();
                }
            }
        }
    }

    async fn process_task(
        task: DirtyTask<C>,
        injector: &Injector<DirtyTask<C>>,
        notify: &Notify,
        database: &Weak<Database<C>>,
        statistic: &Weak<Statistic>,
        dirtied_queries: &Weak<DashSet<QueryID, C::BuildHasher>>,
    ) {
        let dirtied_queries = dirtied_queries.upgrade().unwrap();
        let statistic = statistic.upgrade().unwrap();
        let database = database.upgrade().unwrap();

        if !dirtied_queries.insert(*task.query_id()) {
            return;
        }

        let query_id = *task.query_id();

        for caller in
            unsafe { database.get_backward_edges_unchecked(&query_id).await }
        {
            {
                // opportunisitically try to use the write transaction
                if let Some(mut write_tx) = task.try_load_write_tx() {
                    database
                        .mark_dirty_forward_edge(
                            caller,
                            *task.query_id(),
                            &mut *write_tx,
                        )
                        .await;

                    // maintenance the remaining buffer edges
                    for edge in task.drain_limited() {
                        database
                            .mark_dirty_forward_edge_from(edge, &mut *write_tx)
                            .await;
                    }
                } else {
                    // couldn't get the write transaction, push to the buffer
                    task.push_to_buffer(Edge::new(caller, *task.query_id()));
                }

                statistic.add_dirtied_edge_count();
            }

            let query_kind = database.get_query_kind(&caller).await;

            if matches!(
                query_kind,
                QueryKind::Executable(
                    ExecutionStyle::Projection | ExecutionStyle::Firewall
                )
            ) {
                // don't continue propagation through firewall or
                // projection nodes
                continue;
            }

            injector.push(task.propagate_to(caller));
            notify.notify_one();
        }
    }
}

impl<C: Config> Drop for DirtyWorker<C> {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        self.notify.notify_waiters();
    }
}

impl<C: Config> Engine<C> {
    pub(super) async fn dirty_propagate_from_batch(
        self: &Arc<Self>,
        query_id: impl IntoIterator<Item = QueryID>,
        trasnaction: WriteTransaction<C>,
    ) -> WriteTransaction<C> {
        let write_tx = Arc::new(Mutex::new(trasnaction));
        let stripped_buffer = Arc::new(StrippedBuffer::new());

        let batch = Batch::new(write_tx.clone(), stripped_buffer.clone());
        let notified = batch.notified_owned();

        for query_id in query_id {
            self.computation_graph
                .dirty_worker
                .submit_task(batch.new_task(query_id));
        }

        drop(batch);

        // wait for all tasks to complete
        notified.await;

        let mut write_tx = Arc::try_unwrap(write_tx)
            .unwrap_or_else(|_| {
                panic!("should be unique, notified system is broken")
            })
            .into_inner();

        for remaining_edge in stripped_buffer.drain_all() {
            self.computation_graph
                .database
                .mark_dirty_forward_edge_from(remaining_edge, &mut write_tx)
                .await;
        }

        write_tx
    }

    pub(super) fn clear_dirtied_queries(&self) {
        self.computation_graph.dirtied_queries.clear();
    }
}
