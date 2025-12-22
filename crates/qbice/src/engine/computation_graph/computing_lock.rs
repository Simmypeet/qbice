use std::{
    collections::{HashMap, hash_map::Entry},
    sync::Arc,
};

use dashmap::{
    DashMap,
    mapref::one::{Ref, RefMut},
};
use fxhash::FxHashSet;
use qbice_serialize::{Decode, Encode};
use qbice_stable_hash::Compact128;
use qbice_stable_type_id::Identifiable;
use qbice_storage::intern::Interned;
use tokio::sync::{Notify, futures::OwnedNotified};

use crate::{engine::computation_graph::QueryKind, query::QueryID};

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
    callee_order: Vec<QueryID>,
    query_kind: QueryKind,
}

impl Computing {
    pub fn register_calee(&mut self, callee: QueryID) {
        match self.callee_info.callee_queries.entry(callee) {
            Entry::Occupied(_) => {}

            Entry::Vacant(vacant_entry) => {
                vacant_entry.insert(None);

                // if haven't inserted, add to dependency order
                self.callee_info.callee_order.push(callee);
            }
        }
    }

    pub fn abort_callee(&mut self, callee: &QueryID) {
        assert!(self.callee_info.callee_queries.remove(callee).is_some());
        if let Some(pos) =
            self.callee_info.callee_order.iter().position(|&id| id == *callee)
        {
            self.callee_info.callee_order.remove(pos);
        }
    }

    pub fn query_kind(&self) -> QueryKind {
        self.callee_info.query_kind.clone()
    }
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

    pub fn get_lock_mut(
        &self,
        query_id: &QueryID,
    ) -> RefMut<'_, QueryID, Computing> {
        self.lock.get_mut(query_id).unwrap()
    }
}
