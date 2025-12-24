use std::ops::{Deref, DerefMut};

use fxhash::FxHashSet;
use qbice_serialize::{Decode, Encode};
use qbice_stable_hash::StableHash;
use qbice_stable_type_id::Identifiable;
use qbice_storage::intern::Interned;

use crate::{Engine, config::Config, query::QueryID};

#[derive(
    Debug, Clone, PartialEq, Eq, StableHash, Encode, Decode, Identifiable,
)]
pub struct TransitiveFirewallCallees(FxHashSet<QueryID>);

impl Deref for TransitiveFirewallCallees {
    type Target = FxHashSet<QueryID>;

    fn deref(&self) -> &Self::Target { &self.0 }
}

impl DerefMut for TransitiveFirewallCallees {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
}

impl<C: Config> Engine<C> {
    pub(super) fn new_singleton_tfc(
        &self,
        query_id: QueryID,
    ) -> Interned<TransitiveFirewallCallees> {
        let mut set = FxHashSet::default();
        set.insert(query_id);

        let tfc = TransitiveFirewallCallees(set);
        self.interner.intern(tfc)
    }

    pub(super) fn union_tfcs<'a>(
        &self,
        others: impl IntoIterator<Item = &'a Interned<TransitiveFirewallCallees>>,
    ) -> Option<Interned<TransitiveFirewallCallees>> {
        let mut current_tfc: Option<Interned<TransitiveFirewallCallees>> = None;
        let mut new_archetype: Option<FxHashSet<QueryID>> = None;

        for other in others {
            match (&mut current_tfc, &mut new_archetype) {
                // extract new tfc
                (None, None) => {
                    current_tfc = Some(other.clone());
                }

                (None, Some(_)) => {
                    unreachable!("should've extracted current tfc first")
                }

                (Some(current), None) => {
                    // if one of these two is a superset of the other, we can
                    // skip creating a new archetype
                    if other.0.is_superset(&current.0) {
                        *current = other.clone();
                    } else if current.0.is_superset(&other.0) {
                    } else {
                        // create a new archetype set that is the union of both
                        let mut union_set = current.inner_owned();
                        union_set.0.extend(other.0.iter().copied());

                        new_archetype = Some(union_set.0);
                    }
                }
                (Some(_), Some(existing_set)) => {
                    // has already created a new archetype set, just extend it
                    existing_set.extend(other.0.iter().copied());
                }
            }
        }

        match (current_tfc, new_archetype) {
            (None, Some(_)) => {
                unreachable!("should've extracted current tfc")
            }

            (Some(current_tfc), None) => Some(current_tfc),

            (Some(_), Some(set)) => {
                let new_tfc =
                    TransitiveFirewallCallees(set.into_iter().collect());

                Some(self.interner.intern(new_tfc))
            }

            (None, None) => None,
        }
    }
}
