use std::{collections::HashMap, sync::Arc};

use dashmap::{DashMap, mapref::one::Ref};
use fxhash::FxHashSet;
use qbice_serialize::{Decode, Encode};
use qbice_stable_hash::Compact128;
use qbice_stable_type_id::Identifiable;
use qbice_storage::intern::Interned;
use tokio::sync::{Notify, futures::OwnedNotified};

use crate::query::QueryID;

#[derive(Debug, Clone, PartialEq, Eq, Encode, Decode, Identifiable)]
pub struct TransitiveFirewallCallees(FxHashSet<QueryID>);

#[derive(Debug)]
pub struct Observation {
    seen_value_fingerprint: Compact128,
    seen_transitive_firewall_callees_fingerprint:
        Interned<TransitiveFirewallCallees>,
}

pub struct CalleeInfo {
    callee_queries: HashMap<QueryID, Option<Observation>>,
}

pub struct Computing {
    notify: Arc<Notify>,
    callee_info: CalleeInfo,
}

impl Computing {
    pub fn notified_owned(&self) -> OwnedNotified {
        self.notify.clone().notified_owned()
    }
}

pub struct ComputingLock {
    lock: DashMap<QueryID, Computing>,
}

impl ComputingLock {
    pub fn new() -> Self { Self { lock: DashMap::new() } }
}

impl ComputingLock {
    pub fn try_get_lcok(
        &self,
        query_id: &QueryID,
    ) -> Option<Ref<'_, QueryID, Computing>> {
        self.lock.get(query_id)
    }
}
