use std::marker::PhantomData;

use parking_lot::{Condvar, Mutex, RwLock};
use qbice_storage::sieve::{BackgroundWriter, WriteBuffer};

use crate::{Config, Engine};

#[derive(Debug)]
struct Control {
    current_timestamp_count: usize,
    new_timestamp_active: bool,
    waiting_new_timestamp_count: usize,
}

struct GenerationLockGuard<'x> {
    generation_lock: &'x GenerationLock,
    is_new_generation: bool,
}

/// A synchronization primitive for coordinating between:
///
/// - A single "generation update" writer that wants to create a new generation
///  for the computation graph.
/// - Multiple concurrent "ongoing computation" that wants to update the
///   computation graph within their own generation.
#[derive(Debug)]
pub struct GenerationLock {
    control: Mutex<Control>,
    current_timestamp_cv: Condvar,
    new_timestamp_cv: Condvar,
}

pub struct Writer<C: Config> {
    background_writer: BackgroundWriter<C::Database, C::BuildHasher>,
}

pub struct WriterBufferWithLock<'x, C: Config> {
    _lock: PhantomData<&'x ()>,
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

        WriterBufferWithLock { _lock: PhantomData, writer_buffer }
    }

    pub(in crate::engine::computation_graph) fn submit_write_buffer(
        &self,
        write_buffer: WriterBufferWithLock<'_, C>,
    ) {
        use std::sync::RwLock;

        self.computation_graph
            .persist
            .writer_lock
            .background_writer
            .submit_write_buffer(write_buffer.writer_buffer);
    }
}
