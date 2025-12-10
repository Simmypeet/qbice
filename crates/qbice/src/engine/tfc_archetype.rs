use std::{collections::BTreeSet, ops::Deref, sync::Arc};

use bimap::BiMap;
use parking_lot::RwLock;

use crate::{
    engine::{InitialSeed, fingerprint, meta::Compact128},
    query::QueryID,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TfcArchetypeID(Compact128);

pub type TfcSet = BTreeSet<QueryID>;

/// Manages Transitive Firewall Callees (TFC) archetypes.
///
/// This is used to efficiently share sets of QueryIDs representing TFCs
/// across multiple entities, avoiding duplication of identical sets.
///
/// Most of the time TFC sets are small and identical across query nodes,
/// so this structure helps reduce memory usage and improve performance
/// when checking TFC relationships.
pub struct TfcArchetype {
    pub bimap: RwLock<BiMap<TfcArchetypeID, Arc<TfcSet>>>,
}

impl TfcArchetype {
    /// Creates a new, empty TfcArchetype manager.
    pub fn new() -> Self { Self { bimap: RwLock::new(BiMap::new()) } }

    /// Given a TfcArchetypeID, returns the corresponding TfcSet if it exists.  
    pub fn get_by_id(&self, id: &TfcArchetypeID) -> Option<Arc<TfcSet>> {
        let bimap = self.bimap.read();
        bimap.get_by_left(id).cloned()
    }

    /// Given a starting TfcArchetypeID and an iterator of other
    /// TfcArchetypeIDs, returns a TfcArchetypeID that represents the union
    /// of all the sets.
    pub fn observes_other_tfc(
        &self,
        mut current: TfcArchetypeID,
        others: impl IntoIterator<Item = TfcArchetypeID>,
        initial_seed: InitialSeed,
    ) -> TfcArchetypeID {
        let mut new_archetype: Option<TfcSet> = None;

        for other in others {
            match &mut new_archetype {
                Some(existing_set) => {
                    // has already created a new archetype set, just extend it
                    let bimap = self.bimap.read();
                    let other_set = bimap
                        .get_by_left(&other)
                        .expect("TfcArchetypeID not found")
                        .deref();

                    existing_set.extend(other_set.iter().cloned());
                }

                // happy path: no new archetype set created yet
                None => {
                    if other == current {
                        continue;
                    }

                    let bimap = self.bimap.read();

                    let current_set = bimap
                        .get_by_left(&current)
                        .expect("TfcArchetypeID not found")
                        .deref();

                    let other_set = bimap
                        .get_by_left(&other)
                        .expect("TfcArchetypeID not found")
                        .deref();

                    // if one of these two is a superset of the other, we can
                    // skip creating a new archetype
                    if other_set.is_superset(current_set) {
                        current = other;
                    } else if current_set.is_superset(other_set) {
                        continue;
                    } else {
                        // create a new archetype set that is the union of both
                        let mut union_set = current_set.clone();
                        union_set.extend(other_set.iter().cloned());

                        new_archetype = match &new_archetype {
                            Some(existing_set) => {
                                if existing_set == &union_set {
                                    None
                                } else {
                                    Some(union_set)
                                }
                            }
                            None => Some(union_set),
                        };
                    }
                }
            }
        }

        match new_archetype {
            // has to create a new archetype
            Some(set) => {
                let tfc_hash_128 = Compact128::from_u128(
                    fingerprint::calculate_fingerprint(&set, initial_seed),
                );

                let mut bimap = self.bimap.write();

                bimap.insert_no_overwrite(
                    TfcArchetypeID(tfc_hash_128),
                    Arc::new(set),
                );

                TfcArchetypeID(tfc_hash_128)
            }

            None => current,
        }
    }
}
