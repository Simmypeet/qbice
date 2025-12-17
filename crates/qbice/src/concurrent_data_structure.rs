//! Module for collections of concurrent data structures.

use std::sync::LazyLock;

pub mod lru;

/// Returns the default number of shards to use.
///
/// The default is calculated as `4 * available_parallelism`, rounded up to
/// the next power of two. This provides good concurrency while keeping memory
/// overhead reasonable. The value is computed once and cached.
fn default_shard_amount() -> usize {
    static DEFAULT_SHARD_AMOUNT: LazyLock<usize> = LazyLock::new(|| {
        (std::thread::available_parallelism().map_or(1, usize::from) * 4)
            .next_power_of_two()
    });

    *DEFAULT_SHARD_AMOUNT
}
