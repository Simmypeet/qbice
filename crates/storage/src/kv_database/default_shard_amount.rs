use std::sync::OnceLock;

pub fn default_shard_amount() -> usize {
    static SHARD_AMOUNT: OnceLock<usize> = OnceLock::new();
    *SHARD_AMOUNT.get_or_init(|| {
        (std::thread::available_parallelism().map_or(1, usize::from) * 16)
            .next_power_of_two()
    })
}
