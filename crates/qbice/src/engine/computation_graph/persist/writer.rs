use parking_lot::{RwLock, RwLockReadGuard};
use qbice_storage::sieve::{BackgroundWriter, WriteBuffer};

use crate::{Config, Engine};

pub struct Writer<C: Config> {
    timestamp_lock: RwLock<()>,
    background_writer: BackgroundWriter<C::Database, C::BuildHasher>,
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
    pub const fn new(
        background_writer: BackgroundWriter<C::Database, C::BuildHasher>,
    ) -> Self {
        Self { timestamp_lock: RwLock::new(()), background_writer }
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
            .background_writer
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
            .background_writer
            .submit_write_buffer(write_buffer.writer_buffer);
    }
}
