use std::{hash::BuildHasher, sync::atomic::AtomicUsize};

use crossbeam::{queue::ArrayQueue, utils::CachePadded};
use fxhash::FxBuildHasher;

pub struct ReadBuffer<T> {
    // Box<[]> avoids the indirection of a Vec.
    // CachePadded ensures each queue is on its own CPU cache line.
    shards: Box<[CachePadded<ArrayQueue<T>>]>,
    mask: usize,
    capacity_per_shard: usize,
    count: AtomicUsize,
}

impl<T> ReadBuffer<T> {
    pub fn new(num_shards: usize, capacity_per_shard: usize) -> Self {
        // Ensure power of 2 for fast masking
        let shards_count = num_shards.max(1).next_power_of_two();

        // Initialize the shards
        let mut shards = Vec::with_capacity(shards_count);
        for _ in 0..shards_count {
            shards.push(CachePadded::new(ArrayQueue::new(capacity_per_shard)));
        }

        Self {
            shards: shards.into_boxed_slice(),
            mask: shards_count - 1,
            capacity_per_shard,
            count: AtomicUsize::new(0),
        }
    }

    pub fn len(&self) -> usize {
        self.count.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The "Fire and Forget" push.
    /// Returns: Nothing. If it fails, we don't care.
    #[allow(clippy::cast_possible_truncation)]
    pub fn push(&self, item: T) {
        let id = FxBuildHasher::new().hash_one(std::thread::current().id());
        let index = (id as usize) & self.mask;

        if self.shards[index].push(item).is_ok() {
            self.count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Called by the Maintenance Thread to process events.
    /// Returns an iterator that drains all buffers.
    pub fn drain(&self) -> impl Iterator<Item = T> + '_ {
        self.shards.iter().flat_map(move |q| {
            let mut pop_count = 0;

            std::iter::from_fn(move || {
                if pop_count >= self.capacity_per_shard {
                    return None;
                }

                q.pop().map_or_else(
                    || None,
                    |item| {
                        pop_count += 1;

                        self.count
                            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);

                        Some(item)
                    },
                )
            })
        })
    }
}
