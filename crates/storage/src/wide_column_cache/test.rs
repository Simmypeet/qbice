use std::sync::Arc;

#[tokio::test(flavor = "multi_thread")]
pub async fn insert_overwrite_pending_get_init() {
    let cache = Arc::new(moka::future::Cache::<i32, i32>::builder().build());

    let (done_outer_insert_tx, done_outer_insert_rx) =
        tokio::sync::oneshot::channel::<()>();

    let (start_get_init_tx, start_get_init_rx) =
        tokio::sync::oneshot::channel::<()>();

    tokio::spawn({
        let cache = cache.clone();

        async move {
            cache
                .optionally_get_with(1, async {
                    start_get_init_tx.send(()).unwrap();

                    // Wait until we are notified to proceed.
                    let _ = done_outer_insert_rx.await;
                    None
                })
                .await;
        }
    });

    // wait until get_with has started its init future
    let _ = start_get_init_rx.await;

    // we can now insert while the init future is still pending
    cache.insert(1, 7).await;
    assert_eq!(cache.get(&1).await, Some(7));

    // now allow the get_with init future to complete
    let _ = done_outer_insert_tx.send(());

    assert_eq!(cache.get(&1).await, Some(7));
}

mod pinning {
    use std::sync::atomic::{AtomicBool, Ordering};

    use crate::{
        tiny_lfu::LifecycleListener,
        wide_column_cache::WideColumnCache,
        write_manager::write_behind::{CommittedEpochs, Epoch},
    };

    type Cache = WideColumnCache<i32, i32>;

    /// Reads `key` from a cache whose database holds `stored` for it.
    /// Returns what the cache answers and whether it asked the database.
    async fn read(
        cache: &Cache,
        key: i32,
        stored: Option<i32>,
    ) -> (Option<i32>, bool) {
        let asked = AtomicBool::new(false);

        let value = cache
            .get(
                &key,
                |value| *value,
                || {
                    asked.store(true, Ordering::Relaxed);
                    stored
                },
            )
            .await;

        (value, asked.load(Ordering::Relaxed))
    }

    fn is_cached(cache: &Cache, key: i32) -> bool {
        cache.tiny_lfu.get_map(&key, |_| ()).is_some()
    }

    /// Whether the cache has to keep the entry of `key`.
    fn is_pinned(cache: &Cache, key: i32) -> bool {
        cache
            .tiny_lfu
            .get_map(&key, |entry| {
                cache.tiny_lfu.lifecycle_listener().is_pinned(&key, entry)
            })
            .unwrap_or(false)
    }

    #[test]
    fn written_entry_is_pinned_until_its_epoch_is_committed() {
        let committed = CommittedEpochs::default();
        let cache = Cache::new(16, committed.clone());

        cache.insert(1, 10, Epoch(0));
        cache.insert(2, 20, Epoch(3));

        assert!(is_pinned(&cache, 1));
        assert!(is_pinned(&cache, 2));

        committed.advance_to(Epoch(1));

        assert!(!is_pinned(&cache, 1));
        assert!(is_pinned(&cache, 2));

        committed.advance_to(Epoch(4));

        assert!(!is_pinned(&cache, 2));
    }

    /// The write batches that write to an entry do not have to arrive in the
    /// order of their epochs. The entry is pinned until the last of them has
    /// been committed.
    #[test]
    fn entry_is_pinned_for_the_newest_epoch_that_wrote_it() {
        let committed = CommittedEpochs::default();
        let cache = Cache::new(16, committed.clone());

        cache.insert(1, 10, Epoch(5));
        cache.insert(1, 11, Epoch(2));

        committed.advance_to(Epoch(3));
        assert!(is_pinned(&cache, 1));

        committed.advance_to(Epoch(6));
        assert!(!is_pinned(&cache, 1));
    }

    #[tokio::test]
    async fn entry_read_from_the_database_is_not_pinned() {
        let cache = Cache::new(16, CommittedEpochs::default());

        assert_eq!(read(&cache, 1, Some(10)).await, (Some(10), true));
        assert_eq!(read(&cache, 1, Some(10)).await, (Some(10), false));

        assert!(!is_pinned(&cache, 1));
    }

    /// Until the removal has been committed, the database still holds the
    /// value. The cache has to answer for it.
    #[tokio::test]
    async fn removed_value_is_not_read_back_before_the_removal_is_committed() {
        let committed = CommittedEpochs::default();
        let cache = Cache::new(16, committed.clone());

        cache.remove(&1, Epoch(0));

        assert_eq!(read(&cache, 1, Some(10)).await, (None, false));
        assert!(is_pinned(&cache, 1));

        committed.advance_to(Epoch(1));

        assert_eq!(read(&cache, 1, None).await, (None, false));
        assert!(!is_pinned(&cache, 1));
    }

    #[tokio::test]
    async fn removing_a_cached_value_leaves_a_pinned_negative_entry() {
        let committed = CommittedEpochs::default();
        let cache = Cache::new(16, committed.clone());

        assert_eq!(read(&cache, 1, Some(10)).await, (Some(10), true));

        cache.remove(&1, Epoch(0));

        assert_eq!(read(&cache, 1, Some(10)).await, (None, false));
        assert!(is_pinned(&cache, 1));
    }

    /// A value that has not been committed exists nowhere but in the cache,
    /// however full the cache is. Once it has been committed, a full cache
    /// lets go of it without being told about the commit.
    #[tokio::test]
    async fn full_cache_evicts_what_is_committed_and_nothing_else() {
        let committed = CommittedEpochs::default();
        let cache = Cache::new(2, committed.clone());

        for key in 0..200 {
            cache.insert(key, key + 1000, Epoch(0));
        }

        for key in 0..200 {
            assert_eq!(
                read(&cache, key, None).await,
                (Some(key + 1000), false)
            );
        }

        committed.advance_to(Epoch(1));

        // the cache only looks at what it holds when it is used
        for key in 200..400 {
            cache.insert(key, key + 1000, Epoch(1));
        }

        let cached = (0..200).filter(|key| is_cached(&cache, *key)).count();

        assert!(cached <= 8, "{cached} committed entries are cached");

        for key in 200..400 {
            assert_eq!(
                read(&cache, key, None).await,
                (Some(key + 1000), false)
            );
        }
    }

    /// Presses a full cache for room by writing `others` to it, in a write
    /// batch that has been committed already.
    fn press(cache: &Cache, others: std::ops::Range<i32>) {
        for other in others {
            cache.insert(other, 0, Epoch(0));
        }
    }

    /// A read of the database takes time, and nothing stops the key from
    /// being written in the meantime. If that write were also committed and
    /// its entry evicted before the read comes back, the cache would have no
    /// trace of the write left, and would cache what the read found: a value
    /// older than the one the database holds.
    #[tokio::test]
    async fn entry_is_kept_while_its_key_is_being_read_from_the_database() {
        let committed = CommittedEpochs::default();
        let cache = Cache::new(2, committed.clone());

        let overtaken = AtomicBool::new(false);

        cache
            .get(
                &1,
                |value| *value,
                || {
                    if overtaken.swap(true, Ordering::Relaxed) {
                        // a read that starts after the write finds it
                        return Some(10);
                    }

                    // The read finds nothing for the key. Before it comes
                    // back with that, the key is written, ...
                    cache.insert(1, 10, Epoch(0));

                    // ... the write is committed, ...
                    committed.advance_to(Epoch(1));

                    // ... and the cache needs the room.
                    press(&cache, 1_000..2_000);

                    None
                },
            )
            .await;

        // What the overtaken read answers is its own business: the write
        // happened while it was running. Every read after it has to find
        // the write.
        assert_eq!(read(&cache, 1, Some(10)).await.0, Some(10));

        // Nothing is reading the key anymore, so the cache does not have to
        // keep the entry any longer.
        press(&cache, 2_000..3_000);

        assert!(!is_cached(&cache, 1));
        assert_eq!(read(&cache, 1, Some(10)).await, (Some(10), true));
    }
}
