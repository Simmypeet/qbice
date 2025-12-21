use qbice_serialize::{Decode, Encode};
use qbice_stable_hash::{Sip128Hasher, StableHash, StableHasher};

use crate::engine::InitialSeed;

/// Calculates a 128-bit fingerprint for a value implementing `StableHash`,
/// using an initial seed for variability.
pub fn calculate_fingerprint<T: StableHash>(
    value: &T,
    initial_seed: InitialSeed,
) -> u128 {
    let mut hasher = Sip128Hasher::new();
    initial_seed.stable_hash(&mut hasher);
    value.stable_hash(&mut hasher);
    hasher.finish()
}

/// A less-aligned representation of a 128-bit hash value.
///
/// In 64-bit arch, this u128 normally has an alignment of 16 bytes,
/// which can cause unnecessary padding in structs. This struct
/// represents the same data with two u64s, reducing alignment to 8 bytes.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Default,
    StableHash,
    Encode,
    Decode,
)]
pub struct Compact128(u64, u64);

impl Compact128 {
    pub const fn from_u128(n: u128) -> Self {
        let hi: u64 = (n >> 64) as u64; // upper 64 bits
        let lo: u64 = (n & 0xFFFF_FFFF_FFFF_FFFF) as u64; // lower 64 bits

        Self(hi, lo)
    }
}
