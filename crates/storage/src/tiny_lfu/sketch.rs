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

    // In BloomFilter
    #[allow(clippy::cast_possible_truncation)]
    pub fn contains_or_add(&mut self, hash: u64) -> bool {
        let bit_index = (hash as usize) & self.size_mask;
        let array_index = bit_index / 64;
        let mask = 1u64 << (bit_index % 64);

        let old_val = self.bitmap[array_index];
        if (old_val & mask) != 0 {
            return true;
        }

        // Only write if necessary to avoid cache line dirtying (minor
        // optimization)
        self.bitmap[array_index] = old_val | mask;
        false
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
    table: Vec<u64>, // Stores packed 4-bit counters (16 per u64)
    mask: usize,     // Used instead of modulo
}

impl CountMinSketch {
    fn new(capacity: usize) -> Self {
        // We need 4 rows (depth=4)
        // Capacity determines the width.
        // We pack 16 counters into one u64.

        let width = capacity.max(1).next_power_of_two();

        // Calculate how many u64s we need to hold (width * 4 rows) counters
        // Dividing by 16 because each u64 holds 16 counters
        let len = (width * 4).div_ceil(16);

        Self { table: vec![0; len], mask: width - 1 }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn increment(&mut self, hash: u64) {
        // Double Hashing Strategy
        // h1 is the starting point, h2 is the stride
        let h1 = hash;
        let h2 = hash.rotate_left(32); // Simple remix for stride

        let mut h = h1;

        for r in 0..4 {
            // 1. Calculate the logical index (0..width)
            let idx = (h as usize) & self.mask;

            // 2. Map logical index to physical location
            // "Global" index across the 4 rows conceptually laid out linearly
            let global_idx = (r * (self.mask + 1)) + idx;

            let array_idx = global_idx / 16; // Which u64?
            let bit_offset = (global_idx % 16) * 4; // Which 4-bit block?

            // 3. Read-Modify-Write
            let mut word = self.table[array_idx];
            let counter = (word >> bit_offset) & 0xF;

            if counter < 15 {
                // Increment only if not saturated
                word += 1 << bit_offset;
                self.table[array_idx] = word;
            }

            // Stride for next hash function
            h = h.wrapping_add(h2);
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn estimate(&self, hash: u64) -> u8 {
        let mut min = 15; // Max possible value

        let h1 = hash;
        let h2 = hash.rotate_left(32);
        let mut h = h1;

        for r in 0..4 {
            let idx = (h as usize) & self.mask;
            let global_idx = (r * (self.mask + 1)) + idx;

            let array_idx = global_idx / 16;
            let bit_offset = (global_idx % 16) * 4;

            let count = (self.table[array_idx] >> bit_offset) & 0xF;

            if count == 0 {
                return 0;
            } // Optimization: can't go lower
            if (count as u8) < min {
                min = count as u8;
            }

            h = h.wrapping_add(h2);
        }
        min
    }

    // The "SWAR" Reset (SIMD Within A Register)
    // This is where the 4-bit packing shines.
    fn reset(&mut self) {
        // Divide every counter by 2 in parallel
        for word in &mut self.table {
            // Mask 0xF -> 0x7 to keep counters from spilling into neighbors
            // during shift 0x777... preserves the bottom 3 bits of
            // every nibble Shifting right effectively divides by 2
            *word = (*word >> 1) & 0x7777_7777_7777_7777;
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
