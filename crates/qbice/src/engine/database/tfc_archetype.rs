use std::{collections::BTreeSet, ops::Deref, sync::Arc};

use bimap::BiMap;
use parking_lot::RwLock;
use qbice_stable_hash::StableHash;

use crate::{
    config::Config,
    engine::{
        database::Database,
        fingerprint::{self, Compact128},
    },
    query::QueryID,
};

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, StableHash,
)]
pub struct TfcArchetypeID(Compact128);

pub type TfcSet = BTreeSet<QueryID>;

/// Manages Transitive Firewall Callees (TFC) archetypes.
///
/// This is used to efficiently share sets of `QueryIDs` representing TFCs
/// across multiple entities, avoiding duplication of identical sets.
///
/// Most of the time TFC sets are small and identical across query nodes,
/// so this structure helps reduce memory usage and improve performance
/// when checking TFC relationships.
#[derive(Default)]
pub struct TfcArchetype {
    pub bimap: RwLock<BiMap<TfcArchetypeID, Arc<TfcSet>>>,
}

impl<C: Config> Database<C> {
    /// Given a `TfcArchetypeID`, returns the corresponding `TfcSet` if it
    /// exists.
    pub fn get_tfc_set_by_id(&self, id: &TfcArchetypeID) -> Arc<TfcSet> {
        let bimap = self.tfc_archetype.bimap.read();
        bimap.get_by_left(id).cloned().unwrap()
    }

    /// Creates a new `TfcArchetypeID` for a singleton set containing the given
    /// `QueryID`.
    pub fn new_singleton_tfc(&self, query_id: QueryID) -> TfcArchetypeID {
        let mut set = TfcSet::new();
        set.insert(query_id);
        let tfc_hash_128 = Compact128::from_u128(
            fingerprint::calculate_fingerprint(&set, self.initial_seed()),
        );
        let mut bimap = self.tfc_archetype.bimap.write();
        let _ = bimap
            .insert_no_overwrite(TfcArchetypeID(tfc_hash_128), Arc::new(set));
        TfcArchetypeID(tfc_hash_128)
    }

    /// Continuously unions multiple TFC archetypes into a single archetype.
    ///
    /// Returns `None` if no archetypes were provided.
    pub fn union_tfcs(
        &self,
        others: impl IntoIterator<Item = TfcArchetypeID>,
    ) -> Option<TfcArchetypeID> {
        let mut current_tfc: Option<TfcArchetypeID> = None;
        let mut new_archetype: Option<TfcSet> = None;

        for other in others {
            match (&mut current_tfc, &mut new_archetype) {
                // extract new tfc
                (None, None) => {
                    current_tfc = Some(other);
                }

                (None, Some(_)) => {
                    unreachable!("should've extracted current tfc first")
                }

                (Some(current), None) => {
                    if other == *current {
                        continue;
                    }

                    let bimap = self.tfc_archetype.bimap.read();

                    let current_set = bimap
                        .get_by_left(current)
                        .expect("TfcArchetypeID not found")
                        .deref();

                    let other_set = bimap
                        .get_by_left(&other)
                        .expect("TfcArchetypeID not found")
                        .deref();

                    // if one of these two is a superset of the other, we can
                    // skip creating a new archetype
                    if other_set.is_superset(current_set) {
                        *current = other;
                    } else if current_set.is_superset(other_set) {
                    } else {
                        // create a new archetype set that is the union of both
                        let mut union_set = current_set.clone();
                        union_set.extend(other_set.iter().copied());

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
                (Some(_), Some(existing_set)) => {
                    // has already created a new archetype set, just extend it
                    let bimap = self.tfc_archetype.bimap.read();
                    let other_set = bimap
                        .get_by_left(&other)
                        .expect("TfcArchetypeID not found")
                        .deref();

                    existing_set.extend(other_set.iter().copied());
                }
            }
        }

        match (current_tfc, new_archetype) {
            (None, Some(_)) => {
                unreachable!("should've extracted current tfc")
            }

            (Some(current_tfc), None) => Some(current_tfc),

            (Some(_), Some(set)) => {
                let tfc_hash_128 =
                    Compact128::from_u128(fingerprint::calculate_fingerprint(
                        &set,
                        self.initial_seed(),
                    ));

                let mut bimap = self.tfc_archetype.bimap.write();

                let _ = bimap.insert_no_overwrite(
                    TfcArchetypeID(tfc_hash_128),
                    Arc::new(set),
                );

                Some(TfcArchetypeID(tfc_hash_128))
            }

            (None, None) => None,
        }
    }
}
