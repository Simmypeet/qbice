use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use dashmap::DashSet;
use qbice_stable_hash::Compact128;
use qbice_stable_type_id::{Identifiable, StableTypeID};
use qbice_storage::{
    dynamic_map::DynamicMap as _,
    key_of_set_map::KeyOfSetMap as _,
    kv_database::{DiscriminantEncoding, WideColumn, WideColumnValue},
    single_map::SingleMap as _,
    storage_engine::StorageEngine,
    write_manager::WriteManager,
};
use tokio::sync::{OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock};

use crate::{
    Config, Engine, Query,
    engine::computation_graph::{
        QueryKind,
        persist::{
            CompressedBackwardEdgeSet, NodeInfo, Observation, QueryInput,
            QueryResult, SingleMap, Timestamp, WriteTransaction,
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
    timestamp: AtomicU64,
    phase_mutex: Arc<RwLock<()>>,
    timestamp_map: SingleMap<C, TimestampColumn, Timestamp>,
    write_manager: <C::StorageEngine as StorageEngine>::WriteManager,
}

#[derive(Debug, Clone)]
pub struct ActiveComputationGuard(
    #[allow(unused)] Arc<OwnedRwLockReadGuard<()>>,
);

#[derive(Debug)]
pub struct ActiveInputSessionGuard(
    #[allow(unused)] Arc<OwnedRwLockWriteGuard<()>>,
);

impl<C: Config> Sync<C> {
    pub async fn new(db: &C::StorageEngine) -> Self {
        let write_manager = db.new_write_manager();
        let timestamp_map = db.new_single_map::<TimestampColumn, Timestamp>();

        let timestamp = timestamp_map.get(&()).await;

        let timestamp = if let Some(timestamp) = timestamp {
            timestamp
        } else {
            let mut tx = write_manager.new_write_transaction();

            timestamp_map.insert((), Timestamp(0), &mut tx).await;

            write_manager.submit_write_transaction(tx);

            Timestamp(0)
        };

        Self {
            timestamp: AtomicU64::new(timestamp.0),
            phase_mutex: Arc::new(RwLock::new(())),
            timestamp_map,
            write_manager,
        }
    }
}

impl<C: Config> Engine<C> {
    pub(in crate::engine::computation_graph) fn new_write_transaction(
        &'_ self,
    ) -> WriteTransaction<C> {
        // the guard must be dropped here to make the future Send
        self.computation_graph
            .persist
            .sync
            .write_manager
            .new_write_transaction()
    }

    pub(in crate::engine::computation_graph) async fn acquire_active_computation_guard(
        &self,
    ) -> ActiveComputationGuard {
        let guard = self
            .computation_graph
            .persist
            .sync
            .phase_mutex
            .clone()
            .read_owned()
            .await;

        ActiveComputationGuard(Arc::new(guard))
    }

    pub(in crate::engine::computation_graph) async fn acquire_active_input_session_guard(
        &self,
    ) -> (WriteTransaction<C>, ActiveInputSessionGuard) {
        let mut write_buffer = self
            .computation_graph
            .persist
            .sync
            .write_manager
            .new_write_transaction();

        let prev = self
            .computation_graph
            .persist
            .sync
            .timestamp
            .fetch_add(1, Ordering::SeqCst);
        let new_timestamp = prev + 1;

        self.computation_graph
            .persist
            .sync
            .timestamp_map
            .insert((), Timestamp(new_timestamp), &mut write_buffer)
            .await;

        let guard = self
            .computation_graph
            .persist
            .sync
            .phase_mutex
            .clone()
            .write_owned()
            .await;

        (write_buffer, ActiveInputSessionGuard(Arc::new(guard)))
    }

    pub(in crate::engine::computation_graph) fn submit_write_buffer(
        &self,
        write_buffer: WriteTransaction<C>,
    ) {
        self.computation_graph
            .persist
            .sync
            .write_manager
            .submit_write_transaction(write_buffer);
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
        let _active_computation_phase_guard =
            self.computation_graph.persist.sync.phase_mutex.read().await;

        Timestamp(
            self.computation_graph
                .persist
                .sync
                .timestamp
                .load(Ordering::SeqCst),
        )
    }
}

impl<C: Config> Engine<C> {
    pub(in crate::engine::computation_graph) async fn get_forward_edges_order(
        &self,
        query_id: &QueryID,
    ) -> Option<Arc<[QueryID]>> {
        self.computation_graph
            .persist
            .forward_edge_order
            .get(query_id)
            .await
            .map(|x| x.0)
    }

    pub(in crate::engine::computation_graph) async fn get_forward_edge_observations(
        &self,
        query_id: &QueryID,
    ) -> Option<Arc<HashMap<QueryID, Observation, C::BuildHasher>>> {
        self.computation_graph
            .persist
            .forward_edge_observation
            .get(query_id)
            .await
            .map(|x| x.0)
    }

    pub(in crate::engine::computation_graph) async fn get_node_info(
        &self,
        query_id: &QueryID,
    ) -> Option<NodeInfo> {
        self.computation_graph.persist.node_info.get(query_id).await
    }

    pub(in crate::engine::computation_graph) async fn get_query_kind(
        &self,
        query_id: &QueryID,
    ) -> Option<QueryKind> {
        self.computation_graph.persist.query_kind.get(query_id).await
    }

    pub(in crate::engine::computation_graph) async fn get_last_verified(
        &self,
        query_id: &QueryID,
    ) -> Option<Timestamp> {
        self.computation_graph
            .persist
            .last_verified
            .get(query_id)
            .await
            .map(|x| x.0)
    }

    pub(in crate::engine::computation_graph) async fn get_backward_edges(
        &self,
        query_id: &QueryID,
    ) -> CompressedBackwardEdgeSet<C::BuildHasher> {
        self.computation_graph.persist.backward_edges.get(query_id).await
    }

    pub(in crate::engine::computation_graph) async fn get_query_result<
        Q: Query,
    >(
        &self,
        query_input_hash_128: &Compact128,
    ) -> Option<Q::Value> {
        self.computation_graph
            .persist
            .query_store
            .get::<QueryResult<Q>>(query_input_hash_128)
            .await
            .map(|x| x.0)
    }

    pub(in crate::engine::computation_graph) async fn get_pending_backward_projection(
        &self,
        query_id: &QueryID,
    ) -> Option<Timestamp> {
        return self
            .computation_graph
            .persist
            .pending_backward_projection
            .get(query_id)
            .await
            .map(|x| x.0);
    }

    pub(in crate::engine::computation_graph) async fn get_query_input<
        Q: Query,
    >(
        &self,
        query_id: &Compact128,
    ) -> Option<Q> {
        self.computation_graph
            .persist
            .query_store
            .get::<QueryInput<Q>>(query_id)
            .await
            .map(|x| x.0)
    }

    pub(in crate::engine::computation_graph) async fn get_external_input_queries(
        &self,
        stable_type_id: &StableTypeID,
    ) -> Arc<DashSet<Compact128, C::BuildHasher>> {
        self.computation_graph
            .persist
            .external_input_queries
            .get(stable_type_id)
            .await
    }
}
