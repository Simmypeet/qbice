use crossbeam::queue::SegQueue;

use crate::tiny_lfu::policy::WriteMessage;

pub struct WriteBuffer<T> {
    queue: SegQueue<WriteMessage<T>>,
    len: std::sync::atomic::AtomicUsize,
}

impl<T> WriteBuffer<T> {
    pub const fn new() -> Self {
        Self {
            queue: SegQueue::new(),
            len: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    pub fn push(&self, item: WriteMessage<T>) {
        self.queue.push(item);
        self.len.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn pop(&self) -> Option<WriteMessage<T>> {
        let result = self.queue.pop();
        if result.is_some() {
            self.len.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        }
        result
    }

    pub fn len(&self) -> usize {
        self.len.load(std::sync::atomic::Ordering::Relaxed)
    }
}
