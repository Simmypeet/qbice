use std::{any::TypeId, hash::Hash, sync::atomic::AtomicI32};

use crate::{
    kv_database::{KvDatabase, WideColumn, WideColumnValue},
    tiny_lfu::{self, LifecycleListener, TinyLFU},
    write_manager::write_behind::{self, Epoch},
};

mod single_flight;

#[derive(Debug, Default)]
struct PinnedLifecycleListener;

impl<K, V> LifecycleListener<K, Entry<V>> for PinnedLifecycleListener {
    fn is_pinned(&self, _key: &K, value: &mut Entry<V>) -> bool {
        *value.pin_count.get_mut() > 0
    }
}

struct Entry<V> {
    value: Option<V>,
    pin_count: AtomicI32,
}

#[derive(Debug)]
pub struct WideColumnCache<
    K: Eq + Hash + Send + Sync + 'static,
    V: Send + Sync + 'static,
    T,
> {
    tiny_lfu: TinyLFU<K, Entry<V>, PinnedLifecycleListener>,
    single_flight: single_flight::SingleFlight<K>,

    _phantom: std::marker::PhantomData<T>,
}

impl<K: Eq + Hash + Send + Sync + 'static, V: Send + Sync + 'static, T>
    WideColumnCache<K, V, T>
{
    #[allow(clippy::cast_possible_truncation)]
    pub fn new(capacity: u64) -> Self {
        let shard_count = std::thread::available_parallelism()
            .map_or_else(|_| 4, |x| x.get().next_power_of_two());

        Self {
            tiny_lfu: TinyLFU::new(capacity as usize, shard_count),
            single_flight: single_flight::SingleFlight::new(shard_count),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<K: Eq + Hash + Clone + Send + Sync + 'static, V: Send + Sync + 'static, T>
    WideColumnCache<K, V, T>
{
    pub async fn get<U>(
        &self,
        key: &K,
        map: impl Fn(&V) -> U,
        init: impl Fn() -> Option<V>,
    ) -> Option<U> {
        loop {
            // FAST PATH: Check if the value is already cached, return it  if
            // found.
            if let Some(entry) =
                self.tiny_lfu.get_map(key, |e| e.value.as_ref().map(&map))
            {
                return entry;
            }

            // obtain the single-flight for fetching the value
            self.single_flight
                .wait_or_work(key, || {
                    let value = init();

                    self.tiny_lfu.entry(key.clone(), |entry| match entry {
                        tiny_lfu::Entry::Vacant(vaccant_entry) => {
                            vaccant_entry.insert(Entry {
                                value,
                                pin_count: AtomicI32::new(0),
                            });
                        }

                        tiny_lfu::Entry::Occupied(_) => {
                            // Do nothing as there's an another thread inserted
                            // an explicit value
                        }
                    });
                })
                .await;
        }
    }

    pub fn insert(&self, key: K, value: V, updated: bool) {
        let old_value = self.tiny_lfu.entry(key, |e| {
            match e {
                tiny_lfu::Entry::Vacant(vaccant_entry) => {
                    vaccant_entry.insert(Entry {
                        value: Some(value),
                        pin_count: AtomicI32::new(i32::from(updated)),
                    });

                    None
                }

                tiny_lfu::Entry::Occupied(mut entry) => {
                    // update the existing value and take the value to drop
                    // outside

                    let old_value = entry.get_mut().value.replace(value);

                    if updated {
                        *entry.get_mut().pin_count.get_mut() += 1;
                    }

                    old_value
                }
            }
        });

        // drop the value outside entry lock
        drop(old_value);
    }

    pub fn remove(&self, key: &K, updated: bool) {
        let old_value = self.tiny_lfu.entry(key.clone(), |x| match x {
            tiny_lfu::Entry::Vacant(vaccant_entry) => {
                // if ran with updated=true, with must create a negative
                // cache entry that will use to prevent future `get_init`
                // from database
                if updated {
                    vaccant_entry.insert(Entry {
                        value: None,
                        pin_count: AtomicI32::new(1),
                    });
                }

                None
            }
            tiny_lfu::Entry::Occupied(mut occupied_entry) => {
                if updated {
                    let old_value = occupied_entry.get_mut().value.take();

                    *occupied_entry.get_mut().pin_count.get_mut() += 1;

                    old_value
                } else {
                    // if no pin, we can safely remove the entry
                    if *occupied_entry.get_mut().pin_count.get_mut() == 0 {
                        occupied_entry.remove().value
                    }
                    // this entry is pinned, we'll just mark it as negative
                    else {
                        occupied_entry.get_mut().value.take()
                    }
                }
            }
        });

        drop(old_value);
    }
}

impl<K: Eq + Hash + Clone + Send + Sync + 'static, V: Send + Sync + 'static, T>
    WideColumnCache<K, V, T>
{
    pub(crate) fn flush_staging(
        &self,
        _epoch: Epoch,
        keys: impl IntoIterator<Item = K>,
    ) {
        for key in keys {
            self.tiny_lfu.get_map(&key, |x| {
                x.pin_count.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
            });
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct SingleMapTag;

impl<K: WideColumn, V: WideColumnValue<K>, Db: KvDatabase>
    write_behind::WideColumnCache<K, V, Db>
    for WideColumnCache<K::Key, V, SingleMapTag>
{
    fn flush<'s: 'x, 'i: 'x, 'x>(
        &'s self,
        epoch: Epoch,
        keys: &'i mut (dyn Iterator<Item = <K as WideColumn>::Key> + Send),
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + 'x>>
    {
        Box::pin(async move { self.flush_staging(epoch, keys) })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct DynamicMapTag;

impl<
    K: WideColumn,
    V: WideColumnValue<K>,
    X: Send + Sync + 'static,
    Db: KvDatabase,
> write_behind::WideColumnCache<K, V, Db>
    for WideColumnCache<(K::Key, TypeId), X, DynamicMapTag>
{
    fn flush<'s: 'x, 'i: 'x, 'x>(
        &'s self,
        epoch: Epoch,
        keys: &'i mut (dyn Iterator<Item = <K as WideColumn>::Key> + Send),
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + 'x>>
    {
        Box::pin(async move {
            self.flush_staging(
                epoch,
                keys.map(|x| (x, std::any::TypeId::of::<V>())),
            );
        })
    }
}

#[cfg(test)]
mod test;
