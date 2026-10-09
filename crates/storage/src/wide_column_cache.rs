use std::hash::Hash;

use crate::{
    s3_fifo::{self, LifecycleListener, S3Fifo},
    sharded::default_shard_amount,
    single_flight,
    write_manager::write_behind::{CommittedEpochs, Epoch},
};

/// Keeps an entry in the cache until the database holds its last write.
/// Before that, a read that does not find the entry would fall back to a
/// database that does not have the value yet.
///
/// Also keeps an entry for as long as its key is being read from the
/// database. That read caches what it finds only if the key has no entry,
/// so an entry that goes missing meanwhile takes the writes it stood for
/// with it.
#[derive(Debug)]
struct PinnedLifecycleListener<K> {
    /// The committed epochs of the write manager that writes to the cache.
    committed: CommittedEpochs,

    /// The reads of the database that are under way, by the key they read.
    reads: single_flight::SingleFlight<K>,
}

impl<K: Eq + Hash + Clone, V> LifecycleListener<K, Entry<V>>
    for PinnedLifecycleListener<K>
{
    fn is_pinned(&self, key: &K, value: &Entry<V>) -> bool {
        if value.last_write.is_some_and(|epoch| !self.committed.contains(epoch))
        {
            return true;
        }

        // A read of the database that is under way may have looked before
        // the write of this entry was committed. The entry is then all that
        // tells the read, when it comes back, that what it found is out of
        // date.
        //
        // This has to be asked after the epoch. A read that starts after
        // the write was seen to be committed finds the write in the
        // database. Asked first, a read could start and the write be
        // committed between the two questions, and the entry would be let
        // go while that read holds what the database held before the write.
        self.reads.is_in_flight(key)
    }
}

#[derive(Debug)]
struct Entry<V> {
    value: Option<V>,

    /// The epoch of the last write batch that wrote the entry, or `None` if
    /// the entry is what was read from the database.
    last_write: Option<Epoch>,
}

#[derive(Debug)]
pub struct WideColumnCache<
    K: Clone + Eq + Hash + Send + Sync + 'static,
    V: Send + Sync + 'static,
> {
    entries: S3Fifo<K, Entry<V>, PinnedLifecycleListener<K>>,
}

impl<K: Clone + Eq + Hash + Send + Sync + 'static, V: Send + Sync + 'static>
    WideColumnCache<K, V>
{
    /// Creates a cache for what the write batches of one write manager
    /// write. `committed` are the committed epochs of that write manager.
    #[allow(clippy::cast_possible_truncation)]
    pub fn new(capacity: u64, committed: CommittedEpochs) -> Self {
        Self {
            entries: S3Fifo::with_lifecycle_listener(
                capacity as usize,
                PinnedLifecycleListener {
                    committed,
                    reads: single_flight::SingleFlight::new(
                        default_shard_amount(),
                    ),
                },
            ),
        }
    }
}

impl<K: Eq + Hash + Clone + Send + Sync + 'static, V: Send + Sync + 'static>
    WideColumnCache<K, V>
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
                self.entries.get_map(key, |e| e.value.as_ref().map(&map))
            {
                return entry;
            }

            // obtain the single-flight for fetching the value
            self.entries
                .lifecycle_listener()
                .reads
                .wait_or_work(key, || {
                    let value = init();

                    self.entries.entry(key.clone(), |entry| match entry {
                        s3_fifo::Entry::Vacant(vaccant_entry) => {
                            vaccant_entry
                                .insert(Entry { value, last_write: None });
                        }

                        s3_fifo::Entry::Occupied(_) => {
                            // Do nothing as there's an another thread inserted
                            // an explicit value
                        }
                    });
                })
                .await;
        }
    }

    /// Stores a value that the write batch of `epoch` writes.
    pub fn insert(&self, key: K, value: V, epoch: Epoch) {
        let old_value = self.entries.entry(key, |e| {
            match e {
                s3_fifo::Entry::Vacant(vaccant_entry) => {
                    vaccant_entry.insert(Entry {
                        value: Some(value),
                        last_write: Some(epoch),
                    });

                    None
                }

                s3_fifo::Entry::Occupied(mut entry) => {
                    // update the existing value and take the value to drop
                    // outside
                    let entry = entry.get_mut();

                    entry.last_write = entry.last_write.max(Some(epoch));
                    entry.value.replace(value)
                }
            }
        });

        // drop the value outside entry lock
        drop(old_value);
    }

    /// Removes the value, which is what the write batch of `epoch` does.
    ///
    /// The entry stays as a negative one: until the removal has been
    /// committed the database still holds the value, and a read must not
    /// fall back to it.
    pub fn remove(&self, key: &K, epoch: Epoch) {
        let old_value = self.entries.entry(key.clone(), |x| match x {
            s3_fifo::Entry::Vacant(vaccant_entry) => {
                vaccant_entry
                    .insert(Entry { value: None, last_write: Some(epoch) });

                None
            }
            s3_fifo::Entry::Occupied(mut occupied_entry) => {
                let entry = occupied_entry.get_mut();

                entry.last_write = entry.last_write.max(Some(epoch));
                entry.value.take()
            }
        });

        drop(old_value);
    }
}

#[cfg(test)]
mod test;
