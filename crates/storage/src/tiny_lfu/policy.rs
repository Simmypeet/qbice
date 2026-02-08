use std::hash::BuildHasher;

use fxhash::FxBuildHasher;

use crate::tiny_lfu::{lru, sketch::Sketch};

#[derive(Clone)]
pub enum WriteMessage<K> {
    Insert(K),
    Removed(K),
    Unpinned(K),
}

#[derive(Clone)]
pub enum PolicyMessage<K> {
    ReadHit(K),
    Write(WriteMessage<K>),
}

pub struct Policy<K> {
    lru: lru::Lru<K>,

    window_capacity: usize,
    protected_capacity: usize,
    max_capacity: usize,

    sketch: Sketch,
}

const WINDOW_RATIO: f64 = 0.01;

impl<K> Policy<K> {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    pub fn new(capacity: usize) -> Self {
        let window_capacity = (capacity as f64 * WINDOW_RATIO).ceil() as usize;
        let mut main_capacity = capacity - window_capacity;

        // protected take 80% of main cache
        let protected_capacity = (main_capacity as f64 * 0.8).ceil() as usize;
        // probation takes the remaining 20%
        let probation_capacity = (main_capacity - protected_capacity).max(1);

        main_capacity = protected_capacity + probation_capacity;

        Self {
            lru: lru::Lru::new(),
            window_capacity,
            protected_capacity,
            max_capacity: window_capacity + main_capacity,

            // prepare for the burst traffic by allocating a larger sketch
            sketch: Sketch::new(capacity * 16),
        }
    }

    pub fn on_read_hit(&mut self, key: &K, hash: u64) -> bool
    where
        K: std::hash::Hash + Eq + Clone,
    {
        self.sketch.record_access(hash);

        self.lru.hit(key, self.protected_capacity)
    }

    #[allow(clippy::collapsible_else_if)]
    pub fn on_write(
        &mut self,
        key: &K,
        hash: u64,
        hasher: &FxBuildHasher,
        remove: impl Fn(&K) -> bool,
    ) where
        K: std::hash::Hash + Eq + Clone,
    {
        // first try read hit
        if self.on_read_hit(key, hash) {
            // was already in cache, done
            return;
        }

        // insert into window
        self.lru.new_entry(key.clone(), lru::Region::Window);

        // evict from window if over capacity
        if self.lru.window_len() <= self.window_capacity {
            return;
        }

        // decide whether to add to main cache or evict
        let main_usage = self.lru.probation_len() + self.lru.protected_len();
        let main_limit = self.max_capacity - self.window_capacity;

        if main_usage < main_limit {
            // add to main cache
            self.lru.move_least_recent_of_to_new_region(
                lru::Region::Window,
                lru::Region::Probation,
            );
            return;
        }

        // The DUEL: LRU probation vs LRU window

        let candidate_key =
            self.lru.peek_least_recent(lru::Region::Window).unwrap();
        let candidate_hash = hasher.hash_one(candidate_key);

        let victim_key =
            self.lru.peek_least_recent(lru::Region::Probation).unwrap();
        let victim_hash = hasher.hash_one(victim_key);

        let candidate_freq = self.sketch.estimate_frequency(candidate_hash);
        let victim_freq = self.sketch.estimate_frequency(victim_hash);

        // CANDIDATE wins, VICTIM has to go
        if candidate_freq > victim_freq {
            // can't evict victim, add it to the pinned region
            if remove(victim_key) {
                // the main storage has confirmed removal of the victim,
                // we can evict it safely
                self.lru.pop_least_recent(lru::Region::Probation);
            } else {
                // If the victim is pinned, we move it to pinned region to keep
                // tracking it.
                self.lru.move_least_recent_of_to_new_region(
                    lru::Region::Probation,
                    lru::Region::Pinned,
                );
            }

            // promote candidate to probation region
            self.lru.move_least_recent_of_to_new_region(
                lru::Region::Window,
                lru::Region::Probation,
            );
        }
        // CANDIDATE loses, EVICT it
        else {
            if remove(candidate_key) {
                // the main storage has confirmed removal of the candidate,
                // we can evict it safely
                self.lru.pop_least_recent(lru::Region::Window);
            } else {
                // If the candidate is pinned, we move it to pinned region to
                // keep tracking it.
                self.lru.move_least_recent_of_to_new_region(
                    lru::Region::Window,
                    lru::Region::Pinned,
                );
            }
        }
    }

    pub fn unpin(
        &mut self,
        unpin: &K,
        build_hash: &FxBuildHasher,
        remove: impl Fn(&K) -> bool,
    ) where
        K: std::hash::Hash + Eq + Clone,
    {
        // If the candidate is not pinned, we do nothing.
        if !self.lru.check_is_in_region(unpin, lru::Region::Pinned) {
            return;
        }

        let victim =
            self.lru.peek_least_recent(lru::Region::Probation).unwrap();

        let (pinned_frequency, victim_frequency) = {
            let pinned_hash = build_hash.hash_one(unpin);
            let victim_hash = build_hash.hash_one(victim);

            let pinned_frequency = self.sketch.estimate_frequency(pinned_hash);
            let victim_frequency = self.sketch.estimate_frequency(victim_hash);

            (pinned_frequency, victim_frequency)
        };

        if pinned_frequency > victim_frequency {
            // the pinned frequency wins, we move it back to probation
            if remove(victim) {
                // the storage has confirmed removal of the victim
                self.lru.pop_least_recent(lru::Region::Probation).unwrap();
            } else {
                // the storage has not confirmed removal of the victim
                // we need to move victim to pinned
                self.lru.move_least_recent_of_to_new_region(
                    lru::Region::Probation,
                    lru::Region::Pinned,
                );
            }

            // move the pinned key to the head of the probation region
            self.lru.move_key_to_head_of_region(unpin, lru::Region::Probation);
        } else {
            // the pinned item lost, we evict it
            if remove(unpin) {
                self.lru.remove(unpin);
            }

            // otherwise, keep it
        }
    }

    #[allow(clippy::unused_self)]
    pub fn attempt_to_trim_overflowing_pinned(
        &mut self,
        remove: impl Fn(&K) -> bool,
    ) where
        K: std::hash::Hash + Eq + Clone,
    {
        while self.lru.pinned_len() > 0 {
            let key = self.lru.peek_least_recent(lru::Region::Pinned).unwrap();

            if remove(key) {
                self.lru.pop_least_recent(lru::Region::Pinned);
            } else {
                self.lru.shuffle_tail_to_head(lru::Region::Pinned);
                break;
            }
        }
    }

    pub fn on_removed(&mut self, key: &K)
    where
        K: std::hash::Hash + Eq + Clone,
    {
        self.lru.remove(key);
    }
}
