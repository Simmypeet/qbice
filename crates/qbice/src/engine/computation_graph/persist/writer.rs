use std::sync::{Arc, atomic::AtomicU64};

use parking_lot::{RwLock, RwLockReadGuard};
use qbice_stable_type_id::Identifiable;
use qbice_storage::{
    kv_database::{DiscriminantEncoding, WideColumn, WideColumnValue},
    sieve::{BackgroundWriter, WideColumnSieve, WriteBuffer},
};

use crate::{Config, Engine, engine::computation_graph::persist::Timestamp};

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

pub struct Writer<C: Config> {
    timestamp_lock: RwLock<()>,

    // need to be declared because of the the write buffer interface, we'll
    // fix this later
    timestamp_sieve:
        Arc<WideColumnSieve<TimestampColumn, C::Database, C::BuildHasher>>,

    current_timestamp: AtomicU64,

    background: BackgroundWriter<C::Database, C::BuildHasher>,
}

pub struct WriterBufferWithLock<'x, C: Config> {
    _guard: RwLockReadGuard<'x, ()>,
    writer_buffer: WriteBuffer<C::Database, C::BuildHasher>,
}

impl<C: Config> WriterBufferWithLock<'_, C> {
    pub const fn writer_buffer(
        &mut self,
    ) -> &mut WriteBuffer<C::Database, C::BuildHasher> {
        &mut self.writer_buffer
    }
}

impl<C: Config> Writer<C> {
    pub fn new(db: Arc<C::Database>) -> Self {
        let background_writer =
            BackgroundWriter::<C::Database, C::BuildHasher>::new(
                C::background_writer_thread_count(),
                db.clone(),
            );

        let timestamp_sieve =
            Arc::new(WideColumnSieve::<
                TimestampColumn,
                C::Database,
                C::BuildHasher,
            >::new(1, 1, db, C::BuildHasher::default()));

        let timstamp = timestamp_sieve.get_normal::<Timestamp>(()).map_or_else(
            || {
                let mut tx = background_writer.new_write_buffer();
                timestamp_sieve.put((), Some(Timestamp(0)), &mut tx);
                background_writer.submit_write_buffer(tx);

                AtomicU64::new(0)
            },
            |timestamp| AtomicU64::new(timestamp.0),
        );

        Self {
            timestamp_lock: RwLock::new(()),
            timestamp_sieve,
            current_timestamp: timstamp,
            background: background_writer,
        }
    }
}

impl<C: Config> Engine<C> {
    pub(in crate::engine::computation_graph) fn new_write_buffer(
        &'_ self,
    ) -> WriterBufferWithLock<'_, C> {
        let writer_buffer = self
            .computation_graph
            .persist
            .writer_lock
            .background
            .new_write_buffer();

        let guard =
            self.computation_graph.persist.writer_lock.timestamp_lock.read();

        WriterBufferWithLock { _guard: guard, writer_buffer }
    }

    pub(in crate::engine::computation_graph) fn submit_write_buffer(
        &self,
        write_buffer: WriterBufferWithLock<'_, C>,
    ) {
        self.computation_graph
            .persist
            .writer_lock
            .background
            .submit_write_buffer(write_buffer.writer_buffer);
    }

    pub(in crate::engine::computation_graph) unsafe fn get_current_timestamp_unchecked(
        &self,
    ) -> Timestamp {
        let lock =
            self.computation_graph.persist.writer_lock.timestamp_lock.read();

        let result = Timestamp(
            self.computation_graph
                .persist
                .writer_lock
                .current_timestamp
                .load(std::sync::atomic::Ordering::SeqCst),
        );

        drop(lock);

        result
    }

    pub(in crate::engine::computation_graph) unsafe fn increment_timestamp(
        &self,
        tx: &mut WriterBufferWithLock<C>,
    ) -> Timestamp {
        let current = self
            .computation_graph
            .persist
            .writer_lock
            .current_timestamp
            .load(std::sync::atomic::Ordering::SeqCst);

        let new_timestamp = Timestamp(current + 1);

        self.computation_graph.persist.writer_lock.timestamp_sieve.put(
            (),
            Some(new_timestamp),
            tx.writer_buffer(),
        );
        self.computation_graph
            .persist
            .writer_lock
            .current_timestamp
            .store(new_timestamp.0, std::sync::atomic::Ordering::SeqCst);

        new_timestamp
    }
}
