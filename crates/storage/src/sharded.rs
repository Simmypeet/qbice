use crossbeam_utils::CachePadded;
use parking_lot::RwLock;

pub(crate) struct Sharded<T> {
    shards: Box<[CachePadded<RwLock<T>>]>,
    mask: usize,
}

impl<T> Sharded<T> {}

impl<T> Sharded<T> {
    pub fn new(
        shard_amount: usize,
        mut make_shard: impl FnMut(usize) -> T,
    ) -> Self {
        assert!(
            shard_amount.is_power_of_two(),
            "shard_amount must be a power of two"
        );

        let mut shards = Vec::with_capacity(shard_amount);
        for i in 0..shard_amount {
            shards.push(CachePadded::new(RwLock::new(make_shard(i))));
        }

        Self { shards: shards.into_boxed_slice(), mask: shard_amount - 1 }
    }

    pub fn shard_amount(&self) -> usize { self.shards.len() }

    pub fn read_shard(
        &self,
        shard_index: usize,
    ) -> parking_lot::RwLockReadGuard<'_, T> {
        self.shards[shard_index].read()
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

    pub fn iter_write_shards(
        &self,
    ) -> impl Iterator<Item = parking_lot::RwLockWriteGuard<'_, T>> {
        self.shards.iter().map(|shard| shard.write())
    }

    #[allow(clippy::cast_possible_truncation)]
    pub const fn shard_index(&self, hash: u64) -> usize {
        (hash as usize) & self.mask
    }
}
