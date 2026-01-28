use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use qbice_stable_hash::Compact128;
use qbice_stable_type_id::{Identifiable, StableTypeID};
use qbice_storage::{
    kv_database::{
        DiscriminantEncoding, KvDatabase, WideColumn, WideColumnValue,
    },
    sieve::{BackgroundWriter, WriteBuffer},
};

use crate::{
    Config, Engine, Query,
    engine::computation_graph::{
        CallerInformation, QueryKind,
        persist::{
            BackwardEdge, ExternalInputSet, ForwardEdgeObservation,
            ForwardEdgeOrder, LastVerified, NodeInfo, Observation,
            PendingBackwardProjection, QueryInput, QueryResult, Timestamp,
        },
    },
    query::QueryID,
};

pub struct PhaseMutex {
    /// The current phase of the mutex.
    ///
    /// [1 bit: InputSessionActive | 31 bits: WaitingInputSessions | 32 bits:
    /// ActiveComputations]
    state: AtomicU64,

    notify: Arc<tokio::sync::Notify>,
}

const INPUT_SESSION_ACTIVE_BIT: u64 = 1 << (u64::BITS - 1);
const ACTIVE_COMPUTATION_BITS: u64 = (u64::BITS as u64) / 2;

const ACTIVE_COMPUTATION_MASK: u64 = (1 << ACTIVE_COMPUTATION_BITS) - 1;
const INPUT_SESSION_WAITING_MASK: u64 =
    !ACTIVE_COMPUTATION_MASK & !INPUT_SESSION_ACTIVE_BIT;
const INPUT_SESSION_WAITING_ONE: u64 = 1 << ACTIVE_COMPUTATION_BITS;

impl PhaseMutex {
    pub async fn active_computation_phase_guard(
        self: &Arc<Self>,
    ) -> ActiveComputationPhaseGuard {
        loop {
            let state = self.state.load(Ordering::SeqCst);

            if (state & INPUT_SESSION_ACTIVE_BIT != 0)
                || (state & INPUT_SESSION_WAITING_MASK != 0)
            {
                self.wait_for_state_change().await;
                continue;
            }

            // Try to acquire Read lock
            // We only succeed if state hasn't changed (no new writers arrived)
            if self
                .state
                .compare_exchange_weak(
                    state,
                    state + 1, // Increment reader count
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                )
                .is_ok()
            {
                return ActiveComputationPhaseGuard {
                    phase_mutex: self.clone(),
                };
            }
        }
    }

    pub async fn input_session_pphase_guard(
        self: &Arc<Self>,
    ) -> InputSessionPhaseGuard {
        // Indicate that an input session is waiting
        self.state.fetch_add(INPUT_SESSION_WAITING_ONE, Ordering::SeqCst);

        loop {
            let state = self.state.load(Ordering::SeqCst);

            let active_input_session = state & INPUT_SESSION_ACTIVE_BIT != 0;
            let active_computations = state & ACTIVE_COMPUTATION_MASK != 0;

            if active_input_session || active_computations {
                self.wait_for_state_change().await;
                continue;
            }

            let next_state =
                (state - INPUT_SESSION_WAITING_ONE) | INPUT_SESSION_ACTIVE_BIT;

            // Try to acquire Write lock
            // We only succeed if state hasn't changed (no new readers/writers
            // arrived)
            if self
                .state
                .compare_exchange_weak(
                    state,
                    next_state,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                )
                .is_ok()
            {
                return InputSessionPhaseGuard { phase_mutex: self.clone() };
            }
        }
    }

    async fn wait_for_state_change(&self) {
        let notified = self.notify.notified();
        notified.await;
    }
}

pub struct ActiveComputationPhaseGuard {
    phase_mutex: Arc<PhaseMutex>,
}

impl Drop for ActiveComputationPhaseGuard {
    fn drop(&mut self) {
        let prev_state = self.phase_mutex.state.fetch_sub(1, Ordering::SeqCst);

        // If there are no more active computations, notify waiters
        if (prev_state & ACTIVE_COMPUTATION_MASK) == 1 {
            self.phase_mutex.notify.notify_waiters();
        }
    }
}

pub struct InputSessionPhaseGuard {
    phase_mutex: Arc<PhaseMutex>,
}

impl Drop for InputSessionPhaseGuard {
    fn drop(&mut self) {
        self.phase_mutex
            .state
            .fetch_and(!INPUT_SESSION_ACTIVE_BIT, Ordering::SeqCst);

        self.phase_mutex.notify.notify_waiters();
    }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Identifiable,
)]
pub struct TimestampColumn;

impl WideColumn for TimestampColumn {
    type Key = ();

    type Discriminant = ();

    fn discriminant_encoding() -> DiscriminantEncoding {
        DiscriminantEncoding::Prefixed
    }
}

impl WideColumnValue<TimestampColumn> for Timestamp {
    fn discriminant() {}
}

pub struct Sync<C: Config> {
    timestamp: AtomicU64,
    phase_mutex: Arc<PhaseMutex>,

    background: BackgroundWriter<C::Database, C::BuildHasher>,
}

pub enum Guard {
    ActiveComputation(#[allow(unused)] ActiveComputationPhaseGuard),
    InputSession(#[allow(unused)] InputSessionPhaseGuard),
}

pub struct WriterBufferWithLock<C: Config> {
    guard: Guard,
    writer_buffer: WriteBuffer<C::Database, C::BuildHasher>,
}

impl<C: Config> WriterBufferWithLock<C> {
    pub const fn writer_buffer(
        &mut self,
    ) -> &mut WriteBuffer<C::Database, C::BuildHasher> {
        &mut self.writer_buffer
    }
}

impl<C: Config> Sync<C> {
    pub fn new(db: &Arc<C::Database>) -> Self {
        let background_writer =
            BackgroundWriter::<C::Database, C::BuildHasher>::new(
                C::background_writer_thread_count(),
                db.clone(),
            );

        let timestamp = db
            .get_wide_column::<TimestampColumn, Timestamp>(&())
            .unwrap_or_else(|| {
                let tx = background_writer.new_write_buffer();
                tx.put::<TimestampColumn, Timestamp>(&(), &Timestamp(0));
                background_writer.submit_write_buffer(tx);

                Timestamp(0)
            });

        Self {
            timestamp: AtomicU64::new(timestamp.0),
            phase_mutex: Arc::new(PhaseMutex {
                state: AtomicU64::new(0),
                notify: Arc::new(tokio::sync::Notify::new()),
            }),
            background: background_writer,
        }
    }
}

impl<C: Config> Engine<C> {
    pub(in crate::engine::computation_graph) async fn new_write_buffer(
        &'_ self,
        caller_information: &CallerInformation,
    ) -> WriterBufferWithLock<C> {
        // the guard must be dropped here to make the future Send
        {
            let guard = self
                .computation_graph
                .persist
                .sync
                .phase_mutex
                .active_computation_phase_guard()
                .await;

            if self
                .computation_graph
                .persist
                .sync
                .timestamp
                .load(Ordering::SeqCst)
                == caller_information.timestamp().0
            {
                return WriterBufferWithLock {
                    guard: Guard::ActiveComputation(guard),
                    writer_buffer: self
                        .computation_graph
                        .persist
                        .sync
                        .background
                        .new_write_buffer(),
                };
            }
        }

        self.cancel().await
    }

    pub(in crate::engine::computation_graph) async fn new_write_buffer_with_write_lock(
        &self,
    ) -> WriterBufferWithLock<C> {
        let writer_buffer =
            self.computation_graph.persist.sync.background.new_write_buffer();

        let guard = Guard::InputSession(
            self.computation_graph
                .persist
                .sync
                .phase_mutex
                .input_session_pphase_guard()
                .await,
        );

        WriterBufferWithLock { guard, writer_buffer }
    }

    pub(in crate::engine::computation_graph) fn submit_write_buffer(
        &self,
        write_buffer: WriterBufferWithLock<C>,
    ) {
        self.computation_graph
            .persist
            .sync
            .background
            .submit_write_buffer(write_buffer.writer_buffer);
    }

    pub(in crate::engine::computation_graph) unsafe fn get_current_timestamp_unchecked(
        &self,
    ) -> Timestamp {
        Timestamp(
            self.computation_graph
                .persist
                .sync
                .timestamp
                .load(Ordering::SeqCst),
        )
    }

    pub(in crate::engine::computation_graph) async unsafe fn get_current_timestamp_from_engine(
        &self,
    ) -> Timestamp {
        let _active_computation_phase_guard = self
            .computation_graph
            .persist
            .sync
            .phase_mutex
            .active_computation_phase_guard()
            .await;

        Timestamp(
            self.computation_graph
                .persist
                .sync
                .timestamp
                .load(Ordering::SeqCst),
        )
    }

    pub(in crate::engine::computation_graph) unsafe fn increment_timestamp(
        &self,
        tx: &mut WriterBufferWithLock<C>,
    ) -> Timestamp {
        assert!(
            matches!(tx.guard, Guard::InputSession(_)),
            "Timestamp can only be incremented under input session phase"
        );

        let prev = self
            .computation_graph
            .persist
            .sync
            .timestamp
            .fetch_add(1, Ordering::SeqCst);

        tx.writer_buffer
            .put::<TimestampColumn, Timestamp>(&(), &Timestamp(prev + 1));

        Timestamp(prev + 1)
    }
}

impl<C: Config> Engine<C> {
    pub(in crate::engine::computation_graph) async fn get_forward_edges_order(
        &self,
        query_id: QueryID,
        caller_information: &CallerInformation,
    ) -> Option<Arc<[QueryID]>> {
        {
            let _guard = self
                .computation_graph
                .persist
                .sync
                .phase_mutex
                .active_computation_phase_guard()
                .await;

            if self
                .computation_graph
                .persist
                .sync
                .timestamp
                .load(Ordering::SeqCst)
                == caller_information.timestamp().0
            {
                return self
                    .computation_graph
                    .persist
                    .query_node
                    .get_normal::<ForwardEdgeOrder>(query_id)
                    .await
                    .map(|x| x.0.clone());
            }
        }

        self.cancel().await
    }

    pub(in crate::engine::computation_graph) async fn get_forward_edge_observations(
        &self,
        query_id: QueryID,
        caller_information: &CallerInformation,
    ) -> Option<Arc<HashMap<QueryID, Observation, C::BuildHasher>>> {
        {
            let _guard = self
                .computation_graph
                .persist
                .sync
                .phase_mutex
                .active_computation_phase_guard()
                .await;

            if self
                .computation_graph
                .persist
                .sync
                .timestamp
                .load(Ordering::SeqCst)
                == caller_information.timestamp().0
            {
                return self
                    .computation_graph
                    .persist
                    .query_node
                    .get_normal::<ForwardEdgeObservation<C>>(query_id)
                    .await
                    .map(|x| x.0.clone());
            }
        }

        self.cancel().await
    }

    pub(in crate::engine::computation_graph) async fn get_node_info(
        &self,
        query_id: QueryID,
        caller_information: &CallerInformation,
    ) -> Option<NodeInfo> {
        {
            let _guard = self
                .computation_graph
                .persist
                .sync
                .phase_mutex
                .active_computation_phase_guard()
                .await;

            if self
                .computation_graph
                .persist
                .sync
                .timestamp
                .load(Ordering::SeqCst)
                == caller_information.timestamp().0
            {
                return self
                    .computation_graph
                    .persist
                    .query_node
                    .get_normal::<NodeInfo>(query_id)
                    .await
                    .map(|x| x.clone());
            }
        }

        self.cancel().await
    }

    pub(in crate::engine::computation_graph) async fn get_query_kind(
        &self,
        query_id: QueryID,
        caller_information: &CallerInformation,
    ) -> Option<QueryKind> {
        {
            let _guard = self
                .computation_graph
                .persist
                .sync
                .phase_mutex
                .active_computation_phase_guard()
                .await;

            if self
                .computation_graph
                .persist
                .sync
                .timestamp
                .load(Ordering::SeqCst)
                == caller_information.timestamp().0
            {
                return self
                    .computation_graph
                    .persist
                    .query_node
                    .get_normal::<QueryKind>(query_id)
                    .await
                    .map(|x| *x);
            }
        }

        self.cancel().await
    }

    pub(in crate::engine::computation_graph) async fn get_last_verified(
        &self,
        query_id: QueryID,
        caller_information: &CallerInformation,
    ) -> Option<Timestamp> {
        {
            let _guard = self
                .computation_graph
                .persist
                .sync
                .phase_mutex
                .active_computation_phase_guard()
                .await;

            if self
                .computation_graph
                .persist
                .sync
                .timestamp
                .load(Ordering::SeqCst)
                == caller_information.timestamp().0
            {
                return self
                    .computation_graph
                    .persist
                    .query_node
                    .get_normal::<LastVerified>(query_id)
                    .await
                    .map(|x| x.0);
            }
        }

        self.cancel().await
    }

    pub(in crate::engine::computation_graph) async fn get_backward_edges(
        &self,
        query_id: QueryID,
        caller_information: &CallerInformation,
    ) -> BackwardEdge<C> {
        {
            let _guard = self
                .computation_graph
                .persist
                .sync
                .phase_mutex
                .active_computation_phase_guard()
                .await;

            if self
                .computation_graph
                .persist
                .sync
                .timestamp
                .load(Ordering::SeqCst)
                == caller_information.timestamp().0
            {
                return self
                    .computation_graph
                    .persist
                    .backward_edges
                    .get_set(&query_id)
                    .await
                    .clone();
            }
        }

        self.cancel().await
    }

    pub(in crate::engine::computation_graph) async fn get_query_result<
        Q: Query,
    >(
        &self,
        query_input_hash_128: Compact128,
        caller_information: &CallerInformation,
    ) -> Option<Q::Value> {
        {
            let _guard = self
                .computation_graph
                .persist
                .sync
                .phase_mutex
                .active_computation_phase_guard()
                .await;

            if self
                .computation_graph
                .persist
                .sync
                .timestamp
                .load(Ordering::SeqCst)
                == caller_information.timestamp().0
            {
                return self
                    .computation_graph
                    .persist
                    .query_store
                    .get_normal::<QueryResult<Q>>(query_input_hash_128)
                    .await
                    .map(|x| x.0.clone());
            }
        }

        self.cancel().await
    }

    pub(in crate::engine::computation_graph) async fn get_pending_backward_projection(
        &self,
        query_id: QueryID,
        caller_information: &CallerInformation,
    ) -> Option<Timestamp> {
        {
            let _guard = self
                .computation_graph
                .persist
                .sync
                .phase_mutex
                .active_computation_phase_guard()
                .await;

            if self
                .computation_graph
                .persist
                .sync
                .timestamp
                .load(Ordering::SeqCst)
                == caller_information.timestamp().0
            {
                return self
                    .computation_graph
                    .persist
                    .query_node
                    .get_normal::<PendingBackwardProjection>(query_id)
                    .await
                    .map(|x| x.0);
            }
        }

        self.cancel().await
    }

    pub(in crate::engine::computation_graph) async fn get_query_input<
        Q: Query,
    >(
        &self,
        query_id: Compact128,
        caller_information: &CallerInformation,
    ) -> Option<Q> {
        {
            let _guard = self
                .computation_graph
                .persist
                .sync
                .phase_mutex
                .active_computation_phase_guard()
                .await;

            if self
                .computation_graph
                .persist
                .sync
                .timestamp
                .load(Ordering::SeqCst)
                == caller_information.timestamp().0
            {
                return self
                    .computation_graph
                    .persist
                    .query_store
                    .get_normal::<QueryInput<Q>>(query_id)
                    .await
                    .map(|x| x.0.clone());
            }
        }

        self.cancel().await
    }

    pub(in crate::engine::computation_graph) async unsafe fn get_forward_edges_order_unchecked(
        &self,
        query_id: QueryID,
    ) -> Option<Arc<[QueryID]>> {
        self.computation_graph
            .persist
            .query_node
            .get_normal::<ForwardEdgeOrder>(query_id)
            .await
            .map(|x| x.0.clone())
    }

    pub(in crate::engine::computation_graph) async unsafe fn get_backward_edges_unchecked(
        &self,
        query_id: QueryID,
    ) -> BackwardEdge<C> {
        self.computation_graph
            .persist
            .backward_edges
            .get_set(&query_id)
            .await
            .clone()
    }

    pub(in crate::engine::computation_graph) async unsafe fn get_query_kind_unchecked(
        &self,
        query_id: QueryID,
    ) -> Option<QueryKind> {
        self.computation_graph
            .persist
            .query_node
            .get_normal::<QueryKind>(query_id)
            .await
            .map(|x| *x)
    }

    pub(in crate::engine::computation_graph) async unsafe fn get_node_info_unchecked(
        &self,
        query_id: QueryID,
    ) -> Option<NodeInfo> {
        self.computation_graph
            .persist
            .query_node
            .get_normal::<NodeInfo>(query_id)
            .await
            .map(|x| x.clone())
    }

    pub(in crate::engine::computation_graph) async unsafe fn get_query_input_unchecked<
        Q: Query,
    >(
        &self,
        query_id: Compact128,
    ) -> Option<Q> {
        self.computation_graph
            .persist
            .query_store
            .get_normal::<QueryInput<Q>>(query_id)
            .await
            .map(|x| x.0.clone())
    }

    pub(in crate::engine::computation_graph) async fn get_external_input_queries(
        &self,
        type_id: &StableTypeID,
    ) -> ExternalInputSet<C> {
        self.computation_graph
            .persist
            .external_input_queries
            .get_set(type_id)
            .await
            .clone()
    }
}
