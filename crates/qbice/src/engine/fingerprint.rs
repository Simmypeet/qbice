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
