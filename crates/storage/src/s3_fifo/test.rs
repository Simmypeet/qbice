use std::{
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
    time::{Duration, Instant},
};

use super::{
    Entry, Ghost, LifecycleListener, MAINTENANCE_BATCH_SIZE, MAX_FREQUENCY,
    S3Fifo,
};

type Cache = S3Fifo<u32, u32>;

fn insert<L: LifecycleListener<u32, u32> + Send + Sync + 'static>(
    cache: &S3Fifo<u32, u32, L>,
    key: u32,
) {
    cache.entry(key, |entry| match entry {
        Entry::Vacant(vacant) => vacant.insert(key),
        Entry::Occupied(_) => panic!("{key} has been inserted before"),
    });
}

fn remove(cache: &Cache, key: u32) {
    let removed = cache.entry(key, |entry| match entry {
        Entry::Occupied(occupied) => occupied.remove(),
        Entry::Vacant(_) => panic!("{key} has been inserted"),
    });

    assert_eq!(removed, key);
}

/// Whether the cache holds `key`, without counting as a read of it.
fn is_cached<L: LifecycleListener<u32, u32> + Send + Sync + 'static>(
    cache: &S3Fifo<u32, u32, L>,
    key: u32,
) -> bool {
    cache.storage.contains_sync(&key)
}

/// Inserts enough keys that are never read to push every key that was
/// inserted before through the small queue.
fn press<L: LifecycleListener<u32, u32> + Send + Sync + 'static>(
    cache: &S3Fifo<u32, u32, L>,
    capacity: u32,
    keys_from: u32,
) {
    let count = capacity + u32::try_from(4 * MAINTENANCE_BATCH_SIZE).unwrap();

    for key in keys_from..keys_from + count {
        insert(cache, key);
    }
}

#[test]
fn reads_and_writes_what_was_inserted() {
    let cache = Cache::new(16);

    insert(&cache, 1);

    assert_eq!(cache.get(&1), Some(1));
    assert_eq!(cache.get_map(&1, |value| value + 1), Some(2));
    assert_eq!(cache.get(&2), None);

    cache.entry(1, |entry| match entry {
        Entry::Occupied(mut occupied) => *occupied.get_mut() = 7,
        Entry::Vacant(_) => panic!("1 has been inserted"),
    });

    assert_eq!(cache.get(&1), Some(7));
}

#[test]
fn holds_about_as_many_entries_as_its_capacity() {
    let cache = Cache::new(1_000);

    for key in 0..100_000 {
        insert(&cache, key);
    }

    let cached = cache.storage.len();

    // what has been inserted since the queues were last brought up to date
    // has not been counted yet
    assert!(cached <= 1_000 + 2 * MAINTENANCE_BATCH_SIZE, "{cached} cached");
    assert!(cached >= 900, "{cached} cached");
}

/// An entry that is read at most once leaves through the small queue. An
/// entry that is read more than once moves to the main queue, where the
/// entries that pass through the small queue do not push it out.
#[test]
fn entry_read_more_than_once_outlives_the_ones_that_are_not() {
    let cache = Cache::new(1_000);

    for key in 0..50 {
        insert(&cache, key);
    }

    // read once: not enough
    for key in 0..25 {
        assert_eq!(cache.get(&key), Some(key));
    }

    // read twice
    for key in 25..50 {
        assert_eq!(cache.get(&key), Some(key));
        assert_eq!(cache.get(&key), Some(key));
    }

    press(&cache, 1_000, 1_000);

    for key in 0..25 {
        assert!(!is_cached(&cache, key), "{key} should have been evicted");
    }

    for key in 25..50 {
        assert!(is_cached(&cache, key), "{key} should have been kept");
    }
}

/// Finding an entry with the entry API is a use of the entry like any other.
#[test]
fn finding_an_entry_occupied_counts_as_a_read_of_it() {
    let cache = Cache::new(1_000);

    for key in 0..50 {
        insert(&cache, key);
    }

    for key in 25..50 {
        for _ in 0..2 {
            cache.entry(key, |entry| {
                assert!(matches!(entry, Entry::Occupied(_)));
            });
        }
    }

    press(&cache, 1_000, 1_000);

    for key in 0..25 {
        assert!(!is_cached(&cache, key), "{key} should have been evicted");
    }

    for key in 25..50 {
        assert!(is_cached(&cache, key), "{key} should have been kept");
    }
}

/// The main queue is not a place to stay forever: an entry in it that is not
/// read anymore makes room for the entries that are.
#[test]
fn entry_in_the_main_queue_is_evicted_once_it_is_not_read_anymore() {
    let cache = Cache::new(100);

    insert(&cache, 0);
    assert_eq!(cache.get(&0), Some(0));
    assert_eq!(cache.get(&0), Some(0));

    // moves 0 to the main queue
    press(&cache, 100, 1_000);
    assert!(is_cached(&cache, 0));

    // fills the main queue with entries that are read over and over
    for round in 0..20 {
        for key in 10_000..10_200 {
            if round == 0 {
                insert(&cache, key);
            }

            cache.get(&key);
            cache.get(&key);
        }

        press(&cache, 100, 100_000 + round * 1_000);
    }

    assert!(!is_cached(&cache, 0));
}

/// An entry of the main queue that has been read goes around the queue
/// again. If every entry is read again before its next turn comes, no turn
/// ever finds an entry to evict. The search must end all the same, because
/// the queues are locked while it lasts.
#[test]
fn main_queue_gives_an_entry_up_even_if_all_of_them_are_read_all_the_time() {
    let cache = Cache::new(100);

    for key in 0..50 {
        insert(&cache, key);

        cache.get(&key);
        cache.get(&key);
    }

    // moves the keys to the main queue
    press(&cache, 100, 1_000);

    let mut policy = cache.policy.lock();
    let in_main = policy.main.len();

    assert_eq!(in_main, 50);

    let cached = cache.storage.len();

    let mut second_chances = 0;
    let mut turns = 0;

    while cache.storage.len() == cached {
        // whoever reads the entries is as fast as whoever looks for one to
        // evict
        for key in 0..50 {
            cache.get(&key);
        }

        cache.evict_from_main(&mut policy, &mut second_chances);
        turns += 1;

        assert!(
            turns <= usize::from(MAX_FREQUENCY) * in_main + 1,
            "nothing has been evicted after {turns} turns"
        );
    }
}

/// A key that is inserted again shortly after it was evicted enters the main
/// queue, so it is not evicted again by the entries that are only passing
/// through.
#[test]
fn key_that_comes_back_after_being_evicted_is_kept() {
    let cache = Cache::new(1_000);

    insert(&cache, 0);
    press(&cache, 1_000, 1_000);

    assert!(!is_cached(&cache, 0));

    // comes back, and is not read a single time
    insert(&cache, 0);
    press(&cache, 1_000, 10_000);

    assert!(is_cached(&cache, 0));
}

/// Pins the keys below a bound for as long as the flag is set.
struct PinBelow {
    bound: u32,
    pinned: AtomicBool,
}

impl LifecycleListener<u32, u32> for PinBelow {
    fn is_pinned(&self, key: &u32, _value: &u32) -> bool {
        *key < self.bound && self.pinned.load(Ordering::Relaxed)
    }
}

#[test]
fn pinned_entry_is_kept_until_it_is_unpinned() {
    let cache = S3Fifo::with_lifecycle_listener(100, PinBelow {
        bound: 500,
        pinned: AtomicBool::new(true),
    });

    // five times as many pinned entries as the cache has room for
    for key in 0..500 {
        insert(&cache, key);
    }

    press(&cache, 100, 1_000);

    for key in 0..500 {
        assert!(is_cached(&cache, key), "{key} is pinned");
    }

    cache.lifecycle_listener().pinned.store(false, Ordering::Relaxed);

    // the cache finds out when it is used
    press(&cache, 100, 10_000);

    for key in 0..500 {
        assert!(!is_cached(&cache, key), "{key} is not pinned anymore");
    }
}

/// An entry that is read while it cannot be evicted is in use. The end of
/// what pinned it is no reason to evict it.
#[test]
fn entry_that_is_read_while_it_is_pinned_is_kept_once_it_is_unpinned() {
    let cache = S3Fifo::with_lifecycle_listener(1_000, PinBelow {
        bound: 100,
        pinned: AtomicBool::new(true),
    });

    for key in 0..100 {
        insert(&cache, key);
    }

    // the turn of every one of them comes, and finds them pinned
    press(&cache, 1_000, 1_000);

    for key in 0..50 {
        assert_eq!(cache.get(&key), Some(key));
    }

    cache.lifecycle_listener().pinned.store(false, Ordering::Relaxed);

    press(&cache, 1_000, 10_000);

    for key in 0..50 {
        assert!(is_cached(&cache, key), "{key} was read while pinned");
    }

    for key in 50..100 {
        assert!(!is_cached(&cache, key), "{key} was not read");
    }
}

/// The cache only looks at what it holds when something is inserted. Whoever
/// knows that entries are not pinned anymore does not have to wait for that.
#[test]
fn maintenance_can_be_run_without_inserting() {
    let cache = S3Fifo::with_lifecycle_listener(100, PinBelow {
        bound: 500,
        pinned: AtomicBool::new(true),
    });

    for key in 0..500 {
        insert(&cache, key);
    }

    press(&cache, 100, 1_000);

    cache.lifecycle_listener().pinned.store(false, Ordering::Relaxed);

    // one more key, which is too little for the cache to do anything
    insert(&cache, 2_000);

    for key in 0..500 {
        assert!(is_cached(&cache, key), "nothing has been inserted since");
    }

    cache.run_maintenance();

    for key in 0..500 {
        assert!(!is_cached(&cache, key), "{key} is not pinned anymore");
    }

    assert_eq!(cache.buffered(), 0);

    let cached = cache.storage.len();
    assert!(cached <= 100, "{cached} cached");
}

/// Pins a single key.
struct PinOne(u32);

impl LifecycleListener<u32, u32> for PinOne {
    fn is_pinned(&self, key: &u32, _value: &u32) -> bool { *key == self.0 }
}

/// An entry that never gets unpinned must not keep the entries whose turn
/// came after it from being evicted.
#[test]
fn entry_that_stays_pinned_does_not_hold_up_the_others() {
    let cache = S3Fifo::with_lifecycle_listener(100, PinOne(0));

    for key in 0..50_000 {
        insert(&cache, key);
    }

    assert!(is_cached(&cache, 0));

    let cached = cache.storage.len();
    assert!(cached <= 100 + 2 * MAINTENANCE_BATCH_SIZE, "{cached} cached");
}

/// A key whose entry was removed is still in a queue. Its turn must neither
/// fail nor take the entry of the same key that was inserted afterwards for
/// more than what it is.
#[test]
fn removed_entry_leaves_nothing_behind() {
    let cache = Cache::new(100);

    for key in 0..50 {
        insert(&cache, key);
    }

    for key in 0..50 {
        remove(&cache, key);

        assert_eq!(cache.get(&key), None);
    }

    for key in 0..50_000 {
        if !is_cached(&cache, key) {
            insert(&cache, key);
        }
    }

    let cached = cache.storage.len();
    assert!(cached <= 100 + 2 * MAINTENANCE_BATCH_SIZE, "{cached} cached");

    // every key that lost its entry has had its turn
    assert_eq!(cache.removed.count.load(Ordering::Relaxed), 0);
    assert!(cache.removed.keys.lock().is_empty());
}

/// A key that is removed and inserted again is in the queues twice: for the
/// entry that is gone, and for the one that took its place. The turn of the
/// first is not the turn of the second.
#[test]
fn turn_of_a_removed_entry_is_not_the_turn_of_the_one_inserted_after_it() {
    let cache = Cache::new(1_000);

    // the first key to have its turn
    insert(&cache, 0);
    remove(&cache, 0);

    for key in 1..=900 {
        insert(&cache, key);
    }

    // a long way from having its turn
    insert(&cache, 0);

    // makes the cache evict the hundred keys whose turn it is
    for key in 901..=1_100 {
        insert(&cache, key);
    }

    assert!(is_cached(&cache, 0));

    // its own turn comes like that of any other entry
    press(&cache, 1_000, 10_000);

    assert!(!is_cached(&cache, 0));
    assert_eq!(cache.removed.count.load(Ordering::Relaxed), 0);
    assert!(cache.removed.keys.lock().is_empty());
}

/// The key of the removed entry can also be the one whose turn comes last,
/// if it is in the main queue. Then the key of the new entry is dropped, and
/// the entry is not left without a key in the queues: it would stay cached
/// forever.
#[test]
fn entry_inserted_after_a_removed_one_keeps_one_key_in_the_queues() {
    let cache = Cache::new(1_000);

    insert(&cache, 0);
    assert_eq!(cache.get(&0), Some(0));
    assert_eq!(cache.get(&0), Some(0));

    // moves 0 to the main queue
    press(&cache, 1_000, 1_000);
    assert!(is_cached(&cache, 0));

    remove(&cache, 0);
    insert(&cache, 0);

    // the key of the new entry passes through the small queue
    press(&cache, 1_000, 10_000);

    assert!(is_cached(&cache, 0));
    assert_eq!(cache.removed.count.load(Ordering::Relaxed), 0);

    {
        let policy = cache.policy.lock();

        assert!(policy.main.contains(&0));
        assert!(!policy.small.contains(&0));
    }

    // fills the main queue with entries that are read over and over, until
    // the turn of 0 comes there
    for round in 0..20 {
        for key in 100_000..102_000 {
            if round == 0 {
                insert(&cache, key);
            }

            cache.get(&key);
            cache.get(&key);
        }

        press(&cache, 1_000, 200_000 + round * 2_000);
    }

    assert!(!is_cached(&cache, 0));
}

/// A removed entry leaves its key in a queue. The cache must not evict the
/// entries it holds to make room for an entry it does not hold.
#[test]
fn removed_entries_do_not_count_towards_the_capacity() {
    let cache = Cache::new(1_000);

    for key in 0..1_000 {
        insert(&cache, key);
    }

    // the keys that are the furthest from having their turn
    for key in 500..1_000 {
        remove(&cache, key);
    }

    for key in 1_000..1_400 {
        insert(&cache, key);
    }

    // nine hundred entries, which the cache has room for
    for key in (0..500).chain(1_000..1_400) {
        assert!(is_cached(&cache, key), "{key} should have been kept");
    }
}

/// Takes its time to answer, which makes evicting an entry much slower than
/// inserting one.
struct Slow;

impl LifecycleListener<u32, u32> for Slow {
    fn is_pinned(&self, _key: &u32, _value: &u32) -> bool {
        let start = Instant::now();

        while start.elapsed() < Duration::from_micros(20) {
            std::hint::spin_loop();
        }

        false
    }
}

/// The queues are brought up to date by one thread at a time. Threads that
/// insert more than that thread gets to evict must not grow the cache for as
/// long as they keep at it.
#[test]
fn threads_that_insert_faster_than_entries_are_evicted_have_to_wait() {
    const THREADS: u32 = 8;
    const CAPACITY: usize = 100;

    let cache = S3Fifo::with_lifecycle_listener(CAPACITY, Slow);

    let done = AtomicBool::new(false);
    let most_buffered = AtomicUsize::new(0);
    let most_cached = AtomicUsize::new(0);

    std::thread::scope(|scope| {
        let cache = &cache;
        let done = &done;
        let most_buffered = &most_buffered;
        let most_cached = &most_cached;

        scope.spawn(move || {
            while !done.load(Ordering::Relaxed) {
                most_cached.fetch_max(cache.storage.len(), Ordering::Relaxed);
            }
        });

        let inserters = (0..THREADS)
            .map(|thread| {
                scope.spawn(move || {
                    for index in 0..2_000 {
                        insert(cache, thread * 1_000_000 + index);

                        most_buffered
                            .fetch_max(cache.buffered(), Ordering::Relaxed);
                    }
                })
            })
            .collect::<Vec<_>>();

        for inserter in inserters {
            inserter.join().unwrap();
        }

        done.store(true, Ordering::Relaxed);
    });

    // every thread inserts one key before it finds out that it has to wait
    let most_buffered = most_buffered.load(Ordering::Relaxed);
    assert!(
        most_buffered <= cache.buffer_limit + THREADS as usize,
        "{most_buffered} buffered"
    );

    // what one thread has just taken in and not evicted for yet, and what
    // the others have inserted in the meantime
    let most_cached = most_cached.load(Ordering::Relaxed);
    assert!(
        most_cached <= CAPACITY + 2 * (cache.buffer_limit + THREADS as usize),
        "{most_cached} cached"
    );
}

#[test]
fn concurrent_use_keeps_to_the_capacity() {
    let cache = Cache::new(1_000);

    std::thread::scope(|scope| {
        for thread in 0..8u32 {
            let cache = &cache;

            scope.spawn(move || {
                for index in 0..50_000u32 {
                    let key = thread * 1_000_000 + index;

                    insert(cache, key);
                    assert!(cache.get(&key).is_none_or(|value| value == key));

                    // a key that every thread reads
                    cache.get(&(index % 64));
                }
            });
        }
    });

    // one more round on a single thread, so that nothing is left buffered
    press(&cache, 1_000, 900_000_000);

    let cached = cache.storage.len();
    assert!(cached <= 1_000 + 2 * MAINTENANCE_BATCH_SIZE, "{cached} cached");
}

#[test]
fn ghost_remembers_a_key_once() {
    let mut ghost = Ghost::new(1_000);

    ghost.insert(42);

    assert!(ghost.take(42));
    assert!(!ghost.take(42));
    assert!(!ghost.take(43));
}

#[test]
fn ghost_forgets_the_keys_that_were_evicted_long_ago() {
    let mut ghost = Ghost::new(1_000);

    // a hash that is spread over the whole table, like the ones of real
    // keys
    let hash = |key: u64| key.wrapping_mul(0x9E37_79B9_7F4A_7C15);

    ghost.insert(hash(0));

    for key in 1..=999 {
        ghost.insert(hash(key));
    }

    // the most recent thousand still count, and 0 is the oldest of them
    assert!(ghost.take(hash(0)));

    ghost.insert(hash(0));

    for key in 1_000..=2_000 {
        ghost.insert(hash(key));
    }

    assert!(!ghost.take(hash(0)));

    // what was remembered recently is still there, give or take the keys
    // that shared a full bucket
    let remembered =
        (1_500..=2_000).filter(|key| ghost.take(hash(*key))).count();

    assert!(remembered >= 450, "{remembered} of 501 remembered");
}
