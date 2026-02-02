use std::{
    self,
    sync::{Arc, atomic::AtomicBool},
};

use dashmap::{DashMap, DashSet, Entry, mapref::one::Ref};
use parking_lot::RwLock;
use qbice_stable_hash::Compact128;
use qbice_storage::intern::Interned;
use tokio::sync::{Notify, futures::OwnedNotified};

use crate::{
    Engine, ExecutionStyle, Query,
    config::{Config, WriteTransaction},
    engine::{
        computation_graph::{
            CallerInformation, QueryKind,
            persist::{NodeInfo, Observation, Timestamp},
            slow_path::SlowPath,
            tfc_achetype::TransitiveFirewallCallees,
        },
        default_shard_amount,
    },
    executor::CyclicError,
    query::QueryID,
};

#[derive(Debug, Default)]
pub struct ComputingForwardEdges {
    pub callee_queries: DashMap<QueryID, Option<Observation>>,
    pub callee_order: RwLock<Vec<QueryID>>,
}

impl Computing {
    pub fn register_calee(&self, callee: &QueryID) {
        match self.callee_info.callee_queries.entry(*callee) {
            Entry::Occupied(_) => {}

            Entry::Vacant(vacant_entry) => {
                vacant_entry.insert(None);

                // if haven't inserted, add to dependency order
                self.callee_info.callee_order.write().push(*callee);
            }
        }
    }

    pub fn abort_callee(&self, callee: &QueryID) {
        assert!(self.callee_info.callee_queries.remove(callee).is_some());
        let mut callee_order = self.callee_info.callee_order.write();

        if let Some(pos) = callee_order.iter().position(|&id| id == *callee) {
            callee_order.remove(pos);
        }
    }

    pub fn mark_scc(&self) {
        self.is_in_scc.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    pub fn is_in_scc(&self) -> bool {
        self.is_in_scc.load(std::sync::atomic::Ordering::SeqCst)
    }

    pub fn observe_callee(
        &self,
        callee_target_id: &QueryID,
        seen_value_fingerprint: Compact128,
        seen_transitive_firewall_callees_fingerprint: Compact128,
    ) {
        let mut callee_observation = self
            .callee_info
            .callee_queries
            .get_mut(callee_target_id)
            .expect("callee should have been registered");

        *callee_observation = Some(Observation {
            seen_value_fingerprint,
            seen_transitive_firewall_callees_fingerprint,
        });
    }

    pub const fn query_kind(&self) -> QueryKind { self.query_kind }
}

impl<C: Config> Engine<C> {
    pub(super) fn caller_observe_tfc_callees(
        computing_caller: &Computing,
        callee_info: &NodeInfo,
        kind: QueryKind,
        callee_id: QueryID,
    ) {
        match kind {
            QueryKind::Input
            | QueryKind::Executable(ExecutionStyle::ExternalInput) => {
                // input queries do not contribute to tfc archetype
            }

            QueryKind::Executable(
                ExecutionStyle::Normal | ExecutionStyle::Projection,
            ) => {
                for q in
                    callee_info.transitive_firewall_callees().iter().copied()
                {
                    computing_caller.tfc.insert(q);
                }
            }
            QueryKind::Executable(ExecutionStyle::Firewall) => {
                computing_caller.tfc.insert(callee_id);
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
    callee_info: ComputingForwardEdges,
    is_in_scc: Arc<AtomicBool>,
    tfc: DashSet<QueryID>,
    query_kind: QueryKind,
}

impl Computing {
    pub fn notified_owned(&self) -> OwnedNotified {
        self.notify.clone().notified_owned()
    }
}

pub struct PendingBackwardProjection {
    notify: Arc<Notify>,
}

impl PendingBackwardProjection {
    pub fn notified_owned(&self) -> OwnedNotified {
        self.notify.clone().notified_owned()
    }
}

pub struct BackwardProjectionLockGuard<C: Config> {
    computing_lock: Arc<Lock<C>>,
    query_id: QueryID,
    defused: bool,
}

impl<C: Config> BackwardProjectionLockGuard<C> {
    pub fn done(mut self) {
        self.defused = true;

        let removed = self
            .computing_lock
            .backward_projection_lock
            .remove(&self.query_id)
            .expect(
                "the pending backward projection lock guard has dropped and \
                 tried to remove existing lock, but no entry found",
            );

        // wake up all threads
        removed.1.notify.notify_waiters();
    }
}

impl<C: Config> Drop for BackwardProjectionLockGuard<C> {
    fn drop(&mut self) {
        if self.defused {
            return;
        }

        let removed = self
            .computing_lock
            .backward_projection_lock
            .remove(&self.query_id)
            .expect(
                "the pending backward projection lock guard has dropped and \
                 tried to remove existing lock, but no entry found",
            );

        // wake up all threads
        removed.1.notify.notify_waiters();
    }
}

pub struct Lock<C: Config> {
    normal_lock: DashMap<QueryID, Arc<Computing>, C::BuildHasher>,
    backward_projection_lock:
        DashMap<QueryID, PendingBackwardProjection, C::BuildHasher>,
}

impl<C: Config> Lock<C> {
    pub fn new() -> Self {
        Self {
            normal_lock: DashMap::with_hasher_and_shard_amount(
                C::BuildHasher::default(),
                default_shard_amount(),
            ),
            backward_projection_lock: DashMap::with_hasher_and_shard_amount(
                C::BuildHasher::default(),
                default_shard_amount(),
            ),
        }
    }
}

pub struct ComputingLockGuard<C: Config> {
    computing_lock: Arc<Lock<C>>,
    query_id: QueryID,
    defused: bool,
    computing_mode: ComputingMode,
}

impl<C: Config> ComputingLockGuard<C> {
    pub const fn computing_mode(&self) -> ComputingMode { self.computing_mode }

    /// Don't undo the computing lock when dropped.
    pub fn defuse(mut self) { self.defused = true; }
}

impl<C: Config> Drop for ComputingLockGuard<C> {
    fn drop(&mut self) {
        if self.defused {
            return;
        }

        let removed =
            self.computing_lock.normal_lock.remove(&self.query_id).expect(
                "the computing lock guard has dropped and tried to remove \
                 existing computing lock, but no entry found",
            );

        // wake up all threads
        removed.1.notify.notify_waiters();
    }
}

impl<C: Config> Lock<C> {
    pub fn try_get_lock_for_fast_path(
        &self,
        query_id: &QueryID,
    ) -> Option<(OwnedNotified, Arc<Computing>)> {
        let guard = self.normal_lock.get(query_id)?;
        let notified_owned = guard.notified_owned();

        Some((notified_owned, guard.clone()))
    }

    pub fn try_get_lock(&self, query_id: &QueryID) -> Option<Arc<Computing>> {
        let guard = self.normal_lock.get(query_id)?;
        Some(guard.clone())
    }

    pub fn try_get_pending_backward_projection_lock(
        &self,
        query_id: &QueryID,
    ) -> Option<Ref<'_, QueryID, PendingBackwardProjection>> {
        self.backward_projection_lock.get(query_id)
    }

    pub fn get_lock(&self, query_id: &QueryID) -> Arc<Computing> {
        self.normal_lock
            .get(query_id)
            .expect("computing lock should exist")
            .clone()
    }
}
pub enum LockGuard<C: Config> {
    ComputingLockGuard(ComputingLockGuard<C>),
    BackwardProjectionLockGuard(BackwardProjectionLockGuard<C>),
}

impl<C: Config> Engine<C> {
    async fn computing_lock_guard(
        &self,
        query_id: &QueryID,
        caller_information: &CallerInformation,
        type_name: &'static str,
    ) -> Option<ComputingLockGuard<C>> {
        // IMPORTANT: here we move the retrival logic outside the lock guard
        // to avoid holding the lock across await points

        let last_verified =
            self.get_last_verified(query_id, caller_information).await;

        let (mode, query_kind) = if last_verified.is_some() {
            let kind = self
                .get_query_kind(query_id, caller_information)
                .await
                .unwrap();

            (ComputingMode::Repair, kind)
        } else {
            let executor = self
                .executor_registry
                .get_executor_entry_by_type_id_with_type_name(
                    &query_id.stable_type_id(),
                    type_name,
                );

            let style = executor.obtain_execution_style();

            (ComputingMode::Execute, QueryKind::Executable(style))
        };

        let entry = Arc::new(Computing {
            notify: Arc::new(Notify::new()),
            callee_info: ComputingForwardEdges {
                callee_queries: DashMap::new(),
                callee_order: RwLock::new(Vec::new()),
            },

            query_kind,
            is_in_scc: Arc::new(AtomicBool::new(false)),
            tfc: DashSet::new(),
        });

        match self.computation_graph.lock.normal_lock.entry(*query_id) {
            dashmap::Entry::Occupied(_) => {
                // there's some computing state already try again
                None
            }

            dashmap::Entry::Vacant(vacant_entry) => {
                vacant_entry.insert(entry);

                Some(ComputingLockGuard {
                    computing_lock: self.computation_graph.lock.clone(),
                    query_id: *query_id,
                    defused: false,
                    computing_mode: mode,
                })
            }
        }
    }

    pub(super) fn get_backward_projection_lock_guard(
        &self,
        query_id: &QueryID,
    ) -> Option<BackwardProjectionLockGuard<C>> {
        match self
            .computation_graph
            .lock
            .backward_projection_lock
            .entry(*query_id)
        {
            dashmap::Entry::Occupied(_) => {
                // there's some computing state already try again
                None
            }

            dashmap::Entry::Vacant(vacant_entry) => {
                vacant_entry.insert(PendingBackwardProjection {
                    notify: Arc::new(Notify::new()),
                });

                Some(BackwardProjectionLockGuard {
                    computing_lock: self.computation_graph.lock.clone(),
                    query_id: *query_id,
                    defused: false,
                })
            }
        }
    }

    pub(super) async fn get_lock_guard(
        &self,
        query_id: &QueryID,
        slow_path: SlowPath,
        caller_information: &CallerInformation,
        type_name: &'static str,
    ) -> Option<LockGuard<C>> {
        match slow_path {
            SlowPath::Computing => self
                .computing_lock_guard(query_id, caller_information, type_name)
                .await
                .map(LockGuard::ComputingLockGuard),

            SlowPath::BaackwardProjection => self
                .get_backward_projection_lock_guard(query_id)
                .map(LockGuard::BackwardProjectionLockGuard),
        }
    }

    #[allow(clippy::option_option)]
    pub(super) async fn computing_lock_to_clean_query(
        &self,
        query_id: &QueryID,
        clean_edges: &[QueryID],
        new_tfc: Option<Interned<TransitiveFirewallCallees>>,
        caller_information: &CallerInformation,
        lock_guard: ComputingLockGuard<C>,
    ) {
        self.clean_query(query_id, clean_edges, new_tfc, caller_information)
            .await;

        let dashmap::Entry::Occupied(entry_lock) =
            self.computation_graph.lock.normal_lock.entry(*query_id)
        else {
            panic!("computing lock should exist when transferring to computed");
        };

        let notify = entry_lock.get().notify.clone();

        lock_guard.defuse();

        entry_lock.remove();

        notify.notify_waiters();
    }

    /// Checks whether the stack of computing queries contains a cycle
    #[allow(clippy::needless_pass_by_value)]
    fn check_cyclic_internal(
        &self,
        computing: &Computing,
        target: &QueryID,
    ) -> bool {
        if computing.callee_info.callee_queries.contains_key(target) {
            computing
                .is_in_scc
                .store(true, std::sync::atomic::Ordering::SeqCst);

            return true;
        }

        let mut found = false;

        // OPTIMIZE: this can be parallelized
        for dep in computing.callee_info.callee_queries.iter().map(|x| *x.key())
        {
            let Some(state) = self.computation_graph.lock.try_get_lock(&dep)
            else {
                continue;
            };

            found |= self.check_cyclic_internal(&state, target);
        }

        if found {
            computing
                .is_in_scc
                .store(true, std::sync::atomic::Ordering::SeqCst);
        }

        found
    }

    /// Checks whether the stack of computing queries contains a cycle
    #[allow(clippy::needless_pass_by_value)]
    pub(super) fn check_cyclic(
        &self,
        running_state: &Computing,
        target: &QueryID,
    ) -> bool {
        self.check_cyclic_internal(running_state, target)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn computing_lock_to_computed<Q: Query>(
        &self,
        query: Q,
        query_id: QueryID,
        value: Q::Value,
        query_value_fingerprint: Option<Compact128>,
        lock_guard: ComputingLockGuard<C>,
        has_pending_backward_projection: bool,
        current_timestamp: Timestamp,
        existing_forward_edges: Option<&[QueryID]>,
        continuing_tx: WriteTransaction<C>,
    ) {
        let (
            notify,
            query_kind,
            forward_edge_order,
            forward_edges,
            tfc_archetype,
        ) = {
            let computing = self.computation_graph.lock.get_lock(&query_id);

            (
                computing.notify.clone(),
                computing.query_kind,
                std::mem::take(
                    &mut *computing.callee_info.callee_order.write(),
                ),
                computing
                    .callee_info
                    .callee_queries
                    .iter()
                    // in case of cyclic dependencies, some callees may have
                    // been aborted
                    .filter_map(|x| x.value().map(|v| (*x.key(), v)))
                    .collect(),
                self.create_tfc_from_iter(
                    computing.tfc.iter().map(|x| *x.key()),
                ),
            )
        };

        self.set_computed(
            query,
            query_id,
            value,
            query_value_fingerprint,
            query_kind,
            forward_edge_order.into(),
            forward_edges,
            tfc_archetype,
            has_pending_backward_projection,
            current_timestamp,
            existing_forward_edges,
            continuing_tx,
        );

        lock_guard.defuse();

        // done, remove the computing lock
        self.computation_graph.lock.normal_lock.remove(&query_id);

        notify.notify_waiters();
    }

    pub(super) fn is_query_running_in_scc(
        &self,
        caller: Option<&QueryID>,
    ) -> Result<(), CyclicError> {
        let Some(called_from) = caller else {
            return Ok(());
        };

        let computing_lock = self.computation_graph.lock.get_lock(called_from);

        if computing_lock.is_in_scc() {
            return Err(CyclicError);
        }

        Ok(())
    }
}
