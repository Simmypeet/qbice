use std::{
    collections::{HashMap, hash_map::Entry},
    sync::{Arc, atomic::AtomicBool},
};

use dashmap::{
    DashMap,
    mapref::one::{Ref, RefMut},
};
use qbice_stable_hash::Compact128;
use qbice_storage::{
    intern::Interned,
    kv_database::{KvDatabase, WriteTransaction},
};
use tokio::sync::{Notify, futures::OwnedNotified};

use crate::{
    Engine, ExecutionStyle, Query,
    config::Config,
    engine::computation_graph::{
        QueryKind, QueryWithID,
        computed::{
            BackwardEdgeColumn, ForwardEdgeColumn, LastVerifiedColumn,
            NodeInfo, NodeInfoColumn,
        },
        query_store::{QueryColumn, QueryEntry},
        tfc_achetype::TransitiveFirewallCallees,
    },
    executor::CyclicError,
    query::QueryID,
};

#[derive(Debug)]
pub struct Observation {
    seen_value_fingerprint: Compact128,
    seen_transitive_firewall_callees_fingerprint: Compact128,
}

#[derive(Debug, Default)]
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

    pub fn is_in_scc(&self) -> bool {
        self.is_in_scc.load(std::sync::atomic::Ordering::SeqCst)
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

impl<C: Config> Engine<C> {
    pub(super) fn caller_observe_tfc_callees(
        &self,
        computing_caller: &mut Computing,
        callee_info: &NodeInfo,
        callee_id: &QueryID,
    ) {
        match callee_info.query_kind() {
            QueryKind::Input
            | QueryKind::Executable(
                ExecutionStyle::Normal | ExecutionStyle::Projection,
            ) => {
                computing_caller.tfc_archetype = self.union_tfcs(
                    computing_caller
                        .tfc_archetype
                        .as_ref()
                        .into_iter()
                        .chain(callee_info.transitive_firewall_callees()),
                );
            }
            QueryKind::Executable(ExecutionStyle::Firewall) => {
                let singleton_tfc = self.new_singleton_tfc(*callee_id);

                computing_caller.tfc_archetype = self.union_tfcs(
                    computing_caller
                        .tfc_archetype
                        .as_ref()
                        .into_iter()
                        .chain(std::iter::once(&singleton_tfc)),
                );
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ComputingMode {
    Execute,
    Repair,
}

pub struct Computing {
    notify: Arc<Notify>,
    callee_info: CalleeInfo,
    is_in_scc: AtomicBool,
    tfc_archetype: Option<Interned<TransitiveFirewallCallees>>,
    mode: ComputingMode,
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

pub struct ComputingLockGuard<'a> {
    computing_lock: &'a ComputingLock,
    query_id: QueryID,
    defused: bool,
    computing_mode: ComputingMode,
}

impl ComputingLockGuard<'_> {
    pub const fn computing_mode(&self) -> ComputingMode { self.computing_mode }

    /// Don't undo the computing lock when dropped.
    pub fn defuse(mut self) { self.defused = true; }
}

impl Drop for ComputingLockGuard<'_> {
    fn drop(&mut self) {
        if self.defused {
            return;
        }

        let removed = self.computing_lock.lock.remove(&self.query_id).expect(
            "the computing lock guard has dropped and tried to remove \
             existing computing lock, but no entry found",
        );

        // wake up all threads
        removed.1.notify.notify_waiters();
    }
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

impl<C: Config> Engine<C> {
    pub(super) fn computing_lock_guard(
        &self,
        query_id: &QueryID,
    ) -> Option<ComputingLockGuard<'_>> {
        match self.computation_graph.computing_lock.lock.entry(*query_id) {
            dashmap::Entry::Occupied(_) => {
                // there's some computing state already try again
                None
            }

            dashmap::Entry::Vacant(vacant_entry) => {
                // second check, if the value has already been up-to-date
                let last_verified = self
                    .computation_graph
                    .last_verifieds()
                    .get_normal(query_id);

                if last_verified.as_ref().is_some_and(|x| {
                    **x == self
                        .computation_graph
                        .timestamp_manager
                        .get_current()
                }) {
                    return None;
                }

                let executor = self
                    .executor_registry
                    .get_executor_entry_by_type_id(&query_id.stable_type_id());

                let style = executor.obtain_execution_style();
                let mode = if last_verified.is_some() {
                    ComputingMode::Repair
                } else {
                    ComputingMode::Execute
                };

                vacant_entry.insert(Computing {
                    notify: Arc::new(Notify::new()),
                    callee_info: CalleeInfo {
                        callee_queries: HashMap::new(),
                        callee_order: Vec::new(),
                        query_kind: QueryKind::Executable(style),
                    },

                    // if has some prior node info, then we are repairing the
                    // node.
                    mode,
                    is_in_scc: AtomicBool::new(false),
                    tfc_archetype: None,
                });

                Some(ComputingLockGuard {
                    computing_lock: &self.computation_graph.computing_lock,
                    query_id: *query_id,
                    defused: false,
                    computing_mode: mode,
                })
            }
        }
    }

    pub(super) fn computing_lock_to_computed<Q: Query>(
        &self,
        query_id: &QueryWithID<'_, Q>,
        value: Q::Value,
        lock_guard: ComputingLockGuard<'_>,
    ) {
        let dashmap::Entry::Occupied(mut entry_lock) =
            self.computation_graph.computing_lock.lock.entry(query_id.id)
        else {
            panic!("computing lock should exist when transferring to computed");
        };

        let (notify, query_kind, callee_info, tfc_archetype) = {
            let computing = entry_lock.get_mut();

            (
                computing.notify.clone(),
                computing.callee_info.query_kind,
                std::mem::take(&mut computing.callee_info),
                computing.tfc_archetype.take(),
            )
        };

        let forward_edges: Arc<[QueryID]> = Arc::from(callee_info.callee_order);

        let input_hash_128 = self.hash(query_id.query);

        let node_info = NodeInfo::new(
            query_kind,
            self.hash(&value),
            self.hash(&tfc_archetype),
            tfc_archetype,
        );

        let query_entry = QueryEntry::new(query_id.query.clone(), value);
        let current_timestamp =
            self.computation_graph.timestamp_manager.get_current();

        // durable, write to db first
        {
            let tx = self.database.write_transaction();

            tx.put::<LastVerifiedColumn>(&query_id.id, &current_timestamp);

            for edge in forward_edges.iter() {
                tx.insert_member::<BackwardEdgeColumn>(edge, &query_id.id);
            }

            tx.put::<ForwardEdgeColumn>(&query_id.id, &forward_edges);

            tx.put::<NodeInfoColumn>(&query_id.id, &node_info);

            tx.put::<QueryColumn<Q>>(&input_hash_128, &query_entry);

            tx.commit();
        }

        {
            self.computation_graph
                .last_verifieds()
                .put(query_id.id, Some(current_timestamp));

            for edge in forward_edges.iter() {
                self.computation_graph
                    .backward_edges()
                    .insert_set(edge, std::iter::once(query_id.id));
            }

            self.computation_graph
                .forward_edges()
                .put(query_id.id, Some(forward_edges));

            self.computation_graph
                .node_info()
                .put(query_id.id, Some(node_info));

            self.computation_graph
                .query_store
                .insert(input_hash_128, query_entry);
        }

        lock_guard.defuse();

        entry_lock.remove();

        notify.notify_waiters();
    }

    pub(super) fn is_query_running_in_scc(
        &self,
        caller: Option<&QueryID>,
    ) -> Result<(), CyclicError> {
        let Some(called_from) = caller else {
            return Ok(());
        };

        let computing_lock =
            self.computation_graph.computing_lock.get_lock(called_from);

        if computing_lock.is_in_scc() {
            return Err(CyclicError);
        }

        Ok(())
    }
}
