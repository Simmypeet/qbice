use std::cell::RefCell;

use thread_local::ThreadLocal;

#[derive(Debug)]
pub struct BufferPool {
    pool: ThreadLocal<RefCell<Vec<Vec<u8>>>>,
}

impl BufferPool {
    pub const fn new() -> Self { Self { pool: ThreadLocal::new() } }

    pub fn get_buffer(&self) -> Vec<u8> {
        let cell = self.pool.get_or(|| RefCell::new(Vec::new()));

        cell.borrow_mut().pop().unwrap_or_default()
    }

    pub fn return_buffer(&self, mut buffer: Vec<u8>) {
        // clear the buffer content but keep the allocated capacity
        buffer.clear();

        let cell = self.pool.get_or(|| RefCell::new(Vec::new()));
        cell.borrow_mut().push(buffer);
    }
}
