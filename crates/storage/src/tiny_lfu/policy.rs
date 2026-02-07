use std::hash::BuildHasher;

use fxhash::{FxBuildHasher, FxHashMap};

use crate::tiny_lfu::{
    lru::{self, MoveTo},
    sketch::Sketch,
};

#[derive(Clone)]
pub enum PolicyMessage<K> {
    ReadHit(K),
    Write(K),
    Removed(K),
}

enum Location {
    Window,
    Probation,
    Protected,
}

pub struct Policy<K> {
    windows: lru::Lru<K>,
    probation: lru::Lru<K>,
    protected: lru::Lru<K>,

    location_map: FxHashMap<K, Location>,

    window_capacity: usize,
    protected_capacity: usize,
    max_capacity: usize,

    sketch: Sketch,
}

const MAX_ATTEMPTS: usize = 16;
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
            windows: lru::Lru::new(),
            probation: lru::Lru::new(),
            protected: lru::Lru::new(),
            location_map: FxHashMap::default(),
            window_capacity,
            protected_capacity,
            max_capacity: window_capacity + main_capacity,
            sketch: Sketch::new(capacity),
        }
    }

    pub fn on_read_hit(&mut self, key: &K, hash: u64, increment: bool)
    where
        K: std::hash::Hash + Eq + Clone,
    {
        if increment {
            self.sketch.record_access(hash);
        }

        match self.location_map.get(key) {
            Some(Location::Window) => {
                self.windows.hit(key);
            }

            Some(Location::Probation) => {
                // promote to protected
                self.probation.remove(key);

                self.protected.hit(key);
                self.location_map.insert(key.clone(), Location::Protected);

                // demote LRU of protected to probation if over capacity
                if self.protected.len() > self.protected_capacity
                    && let Some(lru_key) = self.protected.pop_least_recent()
                {
                    self.location_map
                        .insert(lru_key.clone(), Location::Probation);
                    self.probation.hit(&lru_key);
                }
            }

            Some(Location::Protected) => {
                self.protected.hit(key);
            }

            None => {}
        }
    }

    pub fn on_write(
        &mut self,
        key: &K,
        hash: u64,
        hasher: &FxBuildHasher,
        remove: impl Fn(&K) -> bool,
    ) where
        K: std::hash::Hash + Eq + Clone,
    {
        self.sketch.record_access(hash);

        if self.location_map.contains_key(key) {
            self.on_read_hit(key, hash, false);
            return;
        }

        // insert into window
        self.windows.hit(key);
        self.location_map.insert(key.clone(), Location::Window);

        // evict from window if over capacity
        if self.windows.len() <= self.window_capacity {
            return;
        }

        let candidate_key = self.windows.pop_least_recent().unwrap();
        let candidate_hash = hasher.hash_one(&candidate_key);

        // decide whether to add to main cache or evict
        let main_usage = self.probation.len() + self.protected.len();
        let main_limit = self.max_capacity - self.window_capacity;

        if main_usage < main_limit {
            // add to main cache
            self.location_map
                .insert(candidate_key.clone(), Location::Probation);
            self.probation.hit(&candidate_key);

            return;
        }

        // main cache is full, have to evict an item

        // in case of pinned items (ref_count > 0), we can't evict them, so
        // we have to resuffle them to the back of the probation LRU.
        //
        // In the worst case, we may have to try multiple times to find a
        // non-pinned item to evict. We limit the number of attempts to avoid
        // getting stuck at the cost of possibly evicting a better candidate
        // or growing beyond capacity temporarily.
        let mut attempt = 0;
        let mut cursor = self.probation.least_recent_cursor();

        while attempt <= MAX_ATTEMPTS {
            attempt += 1;

            let Some(victim_key) = cursor.get() else {
                // no more victims to try
                break;
            };

            let victim_hash = hasher.hash_one(victim_key);

            let candidate_freq = self.sketch.estimate_frequency(candidate_hash);
            let victim_freq = self.sketch.estimate_frequency(victim_hash);

            if candidate_freq > victim_freq {
                // can't evict victim, try next
                if !remove(victim_key) {
                    cursor.move_to(MoveTo::MoreRecent);

                    continue;
                }

                // the main storage has confirmed removal of the victim,
                // we can evict it safely
                let evicted_key = cursor.remove(MoveTo::MoreRecent).unwrap();
                self.location_map.remove(&evicted_key);

                self.location_map
                    .insert(candidate_key.clone(), Location::Probation);
                self.probation.hit(&candidate_key);

                return;
            }

            // candidate loses, evict it
            if !remove(&candidate_key) {
                // If the candidate is pinned, we can't evict it.
                // We force-promote it to probation to keep tracking it.
                self.location_map
                    .insert(candidate_key.clone(), Location::Probation);
                self.probation.hit(&candidate_key);
                return;
            }

            self.location_map.remove(&candidate_key);
            return;
        }

        // if we reach here, it means we have exceeded MAX_ATTEMPTS and couldn't
        // find a victim to evict (likely all pinned). So we have to evict the
        // candidate itself.
        if !remove(&candidate_key) {
            // Cannot evict candidate either, so we keep it in probation.
            self.location_map
                .insert(candidate_key.clone(), Location::Probation);
            self.probation.hit(&candidate_key);
            return;
        }

        self.location_map.remove(&candidate_key);
    }

    pub fn attempt_to_trim_overflowing_cache(
        &mut self,
        remove: impl Fn(&K) -> bool,
    ) where
        K: std::hash::Hash + Eq + Clone,
    {
        let probation_capacity =
            self.max_capacity - self.window_capacity - self.protected_capacity;

        while self.probation.len() > probation_capacity {
            let Some(victim_key) = self.probation.peek_least_recent() else {
                // no more victims to try
                break;
            };

            if !remove(victim_key) {
                break;
            }

            // the main storage has confirmed removal of the victim,
            // we can evict it safely
            let evicted_key = self.probation.pop_least_recent().unwrap();
            self.location_map.remove(&evicted_key);
        }
    }

    pub fn on_removed(&mut self, key: &K)
    where
        K: std::hash::Hash + Eq + Clone,
    {
        match self.location_map.remove(key) {
            Some(Location::Window) => {
                self.windows.remove(key);
            }

            Some(Location::Probation) => {
                self.probation.remove(key);
            }

            Some(Location::Protected) => {
                self.protected.remove(key);
            }

            None => {}
        }
    }
}
