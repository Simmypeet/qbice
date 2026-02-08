use std::sync::OnceLock;

use crossbeam_utils::CachePadded;
use parking_lot::RwLock;

pub fn default_shard_amount() -> usize {
    static DEFAULT_SHARD_AMOUNT: OnceLock<usize> = OnceLock::new();

    *DEFAULT_SHARD_AMOUNT.get_or_init(|| {
        (std::thread::available_parallelism().map_or(1, usize::from) * 8)
            .next_power_of_two()
    })
}

pub(crate) struct Sharded<T> {
    shards: Box<[CachePadded<RwLock<T>>]>,
    shift: usize,
}

impl<T> Sharded<T> {}

const fn ptr_size_bits() -> usize { std::mem::size_of::<usize>() * 8 }

const fn ncb(shard_amount: usize) -> usize {
    shard_amount.trailing_zeros() as usize
}

impl<T> Sharded<T> {
    pub fn new(
        shard_amount: usize,
        mut make_shard: impl FnMut(usize) -> T,
    ) -> Self {
        assert!(shard_amount > 1, "shard_amount must be greater than 1");
        assert!(
            shard_amount.is_power_of_two(),
            "shard_amount must be a power of two"
        );

        let shift = ptr_size_bits() - ncb(shard_amount);

        let mut shards = Vec::with_capacity(shard_amount);
        for i in 0..shard_amount {
            shards.push(CachePadded::new(RwLock::new(make_shard(i))));
        }

        Self { shards: shards.into_boxed_slice(), shift }
    }

    pub fn shard_amount(&self) -> usize { self.shards.len() }

    pub fn read_shard(
        &self,
        shard_index: usize,
    ) -> parking_lot::RwLockReadGuard<'_, T> {
        self.shards[shard_index].read_recursive()
    }

    pub fn upgradable_read_shard(
        &self,
        shard_index: usize,
    ) -> parking_lot::RwLockUpgradableReadGuard<'_, T> {
        self.shards[shard_index].upgradable_read()
    }

    pub fn write_shard(
        &self,
        shard_index: usize,
    ) -> parking_lot::RwLockWriteGuard<'_, T> {
        self.shards[shard_index].write()
    }

    pub fn iter_read_shards(
        &self,
    ) -> impl Iterator<Item = parking_lot::RwLockReadGuard<'_, T>> {
        self.shards.iter().map(|shard| shard.read())
    }

    pub fn try_iter_read_shards(
        &self,
    ) -> impl Iterator<Item = parking_lot::RwLockReadGuard<'_, T>> {
        self.shards.iter().filter_map(|shard| shard.try_read())
    }

    pub fn try_iter_write_shards(
        &self,
    ) -> impl Iterator<Item = parking_lot::RwLockWriteGuard<'_, T>> {
        self.shards.iter().filter_map(|shard| shard.try_write())
    }

    #[allow(clippy::cast_possible_truncation)]
    pub const fn shard_index(&self, hash: u64) -> usize {
        let hash = hash as usize;

        // Leave the high 7 bits for the HashBrown SIMD tag.
        (hash << 7) >> self.shift
    }
}
