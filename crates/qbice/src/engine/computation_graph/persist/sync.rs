use std::{collections::HashMap, sync::Arc};

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
    timestamp_lock: Arc<tokio::sync::RwLock<Timestamp>>,

    background: BackgroundWriter<C::Database, C::BuildHasher>,
}

pub enum Guard<'x> {
    Read(tokio::sync::RwLockReadGuard<'x, Timestamp>),
    Write(tokio::sync::OwnedRwLockWriteGuard<Timestamp>),
}

pub struct WriterBufferWithLock<'x, C: Config> {
    guard: Guard<'x>,
    writer_buffer: WriteBuffer<C::Database, C::BuildHasher>,
}

impl<C: Config> WriterBufferWithLock<'_, C> {
    pub const fn writer_buffer(
        &mut self,
    ) -> &mut WriteBuffer<C::Database, C::BuildHasher> {
        &mut self.writer_buffer
    }

    pub fn timestamp(&self) -> Timestamp {
        match &self.guard {
            Guard::Read(guard) => **guard,
            Guard::Write(guard) => **guard,
        }
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
            timestamp_lock: Arc::new(tokio::sync::RwLock::new(timestamp)),
            background: background_writer,
        }
    }
}

impl<C: Config> Engine<C> {
    pub(in crate::engine::computation_graph) async fn new_write_buffer(
        &'_ self,
        caller_information: &CallerInformation,
    ) -> WriterBufferWithLock<'_, C> {
        // the guard must be dropped here to make the future Send
        {
            let guard =
                self.computation_graph.persist.sync.timestamp_lock.read().await;

            if *guard == caller_information.timestamp() {
                return WriterBufferWithLock {
                    guard: Guard::Read(guard),
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
        &'_ self,
    ) -> WriterBufferWithLock<'static, C> {
        let writer_buffer =
            self.computation_graph.persist.sync.background.new_write_buffer();

        let guard = Guard::Write(
            self.computation_graph
                .persist
                .sync
                .timestamp_lock
                .clone()
                .write_owned()
                .await,
        );

        WriterBufferWithLock { guard, writer_buffer }
    }

    pub(in crate::engine::computation_graph) fn submit_write_buffer(
        &self,
        write_buffer: WriterBufferWithLock<'_, C>,
    ) {
        self.computation_graph
            .persist
            .sync
            .background
            .submit_write_buffer(write_buffer.writer_buffer);
    }

    pub(in crate::engine::computation_graph) async unsafe fn get_current_timestamp_unchecked(
        &self,
    ) -> Timestamp {
        *self.computation_graph.persist.sync.timestamp_lock.read().await
    }

    pub(in crate::engine::computation_graph) unsafe fn increment_timestamp(
        tx: &mut WriterBufferWithLock<C>,
    ) -> Timestamp {
        let guard = match &mut tx.guard {
            Guard::Read(_) => {
                panic!("Cannot increment timestamp with a read guard");
            }
            Guard::Write(guard) => guard,
        };

        let current_timestamp = Timestamp(guard.0 + 1);
        **guard = current_timestamp;

        tx.writer_buffer
            .put::<TimestampColumn, Timestamp>(&(), &current_timestamp);

        current_timestamp
    }
}

impl<C: Config> Engine<C> {
    pub(in crate::engine::computation_graph) async fn get_forward_edges_order(
        &self,
        query_id: QueryID,
        caller_information: &CallerInformation,
    ) -> Option<Arc<[QueryID]>> {
        {
            let guard =
                self.computation_graph.persist.sync.timestamp_lock.read().await;

            if *guard == caller_information.timestamp() {
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
            let guard =
                self.computation_graph.persist.sync.timestamp_lock.read().await;

            if *guard == caller_information.timestamp() {
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
            let guard =
                self.computation_graph.persist.sync.timestamp_lock.read().await;

            if *guard == caller_information.timestamp() {
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
            let guard =
                self.computation_graph.persist.sync.timestamp_lock.read().await;

            if *guard == caller_information.timestamp() {
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
            let guard =
                self.computation_graph.persist.sync.timestamp_lock.read().await;

            if *guard == caller_information.timestamp() {
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
            let guard =
                self.computation_graph.persist.sync.timestamp_lock.read().await;

            if *guard == caller_information.timestamp() {
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
            let guard =
                self.computation_graph.persist.sync.timestamp_lock.read().await;

            if *guard == caller_information.timestamp() {
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
            let guard =
                self.computation_graph.persist.sync.timestamp_lock.read().await;

            if *guard == caller_information.timestamp() {
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
            let guard =
                self.computation_graph.persist.sync.timestamp_lock.read().await;

            if *guard == caller_information.timestamp() {
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
