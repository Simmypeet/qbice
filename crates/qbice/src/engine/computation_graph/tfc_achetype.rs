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
    pub(super) fn create_tfc(
        &self,
        hash_set: FxHashSet<QueryID>,
    ) -> Interned<TransitiveFirewallCallees> {
        let tfc = TransitiveFirewallCallees(hash_set);
        self.interner.intern(tfc)
    }

    pub(super) fn create_tfc_from_iter(
        &self,
        query_ids: impl IntoIterator<Item = QueryID>,
    ) -> Interned<TransitiveFirewallCallees> {
        let set: FxHashSet<QueryID> = query_ids.into_iter().collect();

        let tfc = TransitiveFirewallCallees(set);
        self.interner.intern(tfc)
    }
}
