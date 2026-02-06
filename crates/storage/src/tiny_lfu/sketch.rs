struct BloomFilter {
    bitmap: Vec<u64>,
    size_mask: usize,
}

impl BloomFilter {
    /// Capacity should be roughly the number of items expected in the window.
    pub fn new(capacity: usize) -> Self {
        // Round up to next power of 2 for fast masking
        let bits = capacity.max(64).next_power_of_two();
        let u64_count = bits / 64;

        Self { bitmap: vec![0; u64_count], size_mask: bits - 1 }
    }

    #[allow(clippy::cast_possible_truncation)]
    pub fn contains_or_add(&mut self, hash: u64) -> bool {
        let bit_index = (hash as usize) & self.size_mask;
        let array_index = bit_index / 64;
        let bit_in_word = bit_index % 64;
        let mask = 1u64 << bit_in_word;

        let already_present = (self.bitmap[array_index] & mask) != 0;
        self.bitmap[array_index] |= mask;

        already_present
    }

    #[allow(clippy::cast_possible_truncation)]
    pub fn read_access(&self, hash: u64) -> bool {
        let bit_index = (hash as usize) & self.size_mask;
        let array_index = bit_index / 64;
        let bit_in_word = bit_index % 64;
        let mask = 1u64 << bit_in_word;

        (self.bitmap[array_index] & mask) != 0
    }

    pub fn clear(&mut self) {
        for word in &mut self.bitmap {
            *word = 0;
        }
    }
}

struct CountMinSketch {
    table: Vec<u8>,
    width: usize,
    depth: usize,
}

impl CountMinSketch {
    fn new(mut width: usize) -> Self {
        width = width.max(1).next_power_of_two();
        let depth = 4; // 4 Hash functions is standard

        Self { table: vec![0; width * depth], width, depth }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn increment(&mut self, hash: u64) {
        let mut h = hash;
        for r in 0..self.depth {
            let idx = (r * self.width) + (h as usize % self.width);
            self.table[idx] = self.table[idx].saturating_add(1);
            h = h.wrapping_add(hash); // Simple rehashing
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn estimate(&self, hash: u64) -> u8 {
        let mut min = u8::MAX;
        let mut h = hash;
        for r in 0..self.depth {
            let idx = (r * self.width) + (h as usize % self.width);
            min = min.min(self.table[idx]);
            h = h.wrapping_add(hash);
        }
        min
    }

    // "Aging" process: Divide all counters by 2 to favor recent history
    fn reset(&mut self) {
        for c in &mut self.table {
            *c /= 2;
        }
    }
}

pub struct Sketch {
    bloom_filter: BloomFilter,
    cms: CountMinSketch,

    additions: usize,
    reset_threshold: usize,
}

impl Sketch {
    pub fn new(capacity: usize) -> Self {
        Self {
            bloom_filter: BloomFilter::new(capacity),
            cms: CountMinSketch::new(capacity),
            additions: 0,
            reset_threshold: capacity,
        }
    }

    pub fn record_access(&mut self, hash: u64) {
        if self.bloom_filter.contains_or_add(hash) {
            self.cms.increment(hash);
        }

        self.additions += 1;
        if self.additions >= self.reset_threshold {
            self.bloom_filter.clear();
            self.cms.reset();
            self.additions = 0;
        }
    }

    pub fn estimate_frequency(&self, hash: u64) -> u8 {
        let mut estimate = self.cms.estimate(hash);
        if self.bloom_filter.read_access(hash) {
            estimate = estimate.saturating_add(1);
        }

        estimate
    }
}
