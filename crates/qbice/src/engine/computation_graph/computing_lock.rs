use std::{
    collections::{HashMap, hash_map::Entry},
    sync::{Arc, atomic::AtomicBool},
};

use dashmap::{
    DashMap,
    mapref::one::{Ref, RefMut},
};
use qbice_stable_hash::Compact128;
use tokio::sync::{Notify, futures::OwnedNotified};

use crate::{engine::computation_graph::QueryKind, query::QueryID};

#[derive(Debug)]
pub struct Observation {
    seen_value_fingerprint: Compact128,
    seen_transitive_firewall_callees_fingerprint: Compact128,
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

    pub const fn query_kind(&self) -> QueryKind { self.callee_info.query_kind }

    pub fn mark_scc(&self) {
        self.is_in_scc.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    pub fn contains_query(&self, query_id: &QueryID) -> bool {
        self.callee_info.callee_queries.contains_key(query_id)
    }

    pub fn registered_callees(&self) -> impl Iterator<Item = &QueryID> {
        self.callee_info.callee_queries.keys()
    }

    pub fn observe_callee(
        &mut self,
        callee_target_id: &QueryID,
        seen_value_fingerprint: Compact128,
        seen_transitive_firewall_callees_fingerprint: Compact128,
    ) {
        let callee_observation = self
            .callee_info
            .callee_queries
            .get_mut(callee_target_id)
            .expect("callee should have been registered");

        *callee_observation = Some(Observation {
            seen_value_fingerprint,
            seen_transitive_firewall_callees_fingerprint,
        });
    }
}

pub struct Computing {
    notify: Arc<Notify>,
    callee_info: CalleeInfo,
    is_in_scc: AtomicBool,
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

    pub fn get_lock(&self, query_id: &QueryID) -> Ref<'_, QueryID, Computing> {
        self.lock.get(query_id).unwrap()
    }
}
