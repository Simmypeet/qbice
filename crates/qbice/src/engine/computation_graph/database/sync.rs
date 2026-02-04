use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use qbice_stable_type_id::Identifiable;
use qbice_storage::{
    kv_database::{DiscriminantEncoding, WideColumn, WideColumnValue},
    single_map::SingleMap as _,
    storage_engine::StorageEngine,
    write_manager::WriteManager,
};
use tokio::sync::{OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock};

use crate::{
    Config, Engine,
    engine::computation_graph::database::{
        SingleMap, Timestamp, WriteTransaction,
    },
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
            .database
            .sync
            .write_manager
            .new_write_transaction()
    }

    pub(in crate::engine::computation_graph) async fn acquire_active_computation_guard(
        &self,
    ) -> ActiveComputationGuard {
        let guard = self
            .computation_graph
            .database
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
            .database
            .sync
            .write_manager
            .new_write_transaction();

        let prev = self
            .computation_graph
            .database
            .sync
            .timestamp
            .fetch_add(1, Ordering::SeqCst);
        let new_timestamp = prev + 1;

        self.computation_graph
            .database
            .sync
            .timestamp_map
            .insert((), Timestamp(new_timestamp), &mut write_buffer)
            .await;

        let guard = self
            .computation_graph
            .database
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
            .database
            .sync
            .write_manager
            .submit_write_transaction(write_buffer);
    }

    pub(in crate::engine::computation_graph) unsafe fn get_current_timestamp_unchecked(
        &self,
    ) -> Timestamp {
        Timestamp(
            self.computation_graph
                .database
                .sync
                .timestamp
                .load(Ordering::SeqCst),
        )
    }

    pub(in crate::engine::computation_graph) async unsafe fn get_current_timestamp_from_engine(
        &self,
    ) -> Timestamp {
        let _active_computation_phase_guard =
            self.computation_graph.database.sync.phase_mutex.read().await;

        Timestamp(
            self.computation_graph
                .database
                .sync
                .timestamp
                .load(Ordering::SeqCst),
        )
    }
}
