use std::{
    self,
    sync::{Arc, atomic::AtomicBool},
};

use dashmap::{DashMap, DashSet, Entry};
use parking_lot::RwLock;
use qbice_stable_hash::Compact128;
use qbice_storage::intern::Interned;

use crate::{
    Engine, ExecutionStyle, Query,
    config::{Config, WriteTransaction},
    engine::{
        computation_graph::{
            CallerInformation, QueryKind,
            database::{NodeInfo, Observation, Snapshot, Timestamp},
            slow_path::SlowPath,
            tfc_achetype::TransitiveFirewallCallees,
        },
        default_shard_amount,
        guard::GuardExt,
    },
    executor::CyclicError,
    query::QueryID,
};

#[derive(Debug, Default)]
pub struct ComputingForwardEdges {
    pub callee_queries: DashMap<QueryID, Option<Observation>>,
    pub callee_order: RwLock<Vec<QueryID>>,
}

impl QueryComputing {
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

    pub fn caller_observe_tfc_callees(
        &self,
        callee_info: &NodeInfo,
        callee_kind: QueryKind,
        callee_id: QueryID,
    ) {
        match callee_kind {
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
                    self.tfc.insert(q);
                }
            }
            QueryKind::Executable(ExecutionStyle::Firewall) => {
                self.tfc.insert(callee_id);
            }
        }
    }
}

impl<C: Config> Engine<C> {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ComputingMode {
    Execute,
    Repair,
}

#[derive(Debug)]
pub struct QueryComputing {
    callee_info: ComputingForwardEdges,
    is_in_scc: Arc<AtomicBool>,
    tfc: DashSet<QueryID>,
    query_kind: QueryKind,
}

#[derive(Debug)]
pub struct PendingBackwardProjection;

pub struct BackwardProjectionLockGuard<C: Config> {
    engine: Arc<Engine<C>>,
    query_id: QueryID,
    defused: bool,
}

impl<C: Config> BackwardProjectionLockGuard<C> {
    pub fn done(mut self) {
        self.defused = true;

        self.engine
            .computation_graph
            .computing
            .backward_projection_lock
            .remove(&self.query_id)
            .expect(
                "the pending backward projection lock guard has dropped and \
                 tried to remove existing lock, but no entry found",
            );
    }
}

impl<C: Config> Drop for BackwardProjectionLockGuard<C> {
    fn drop(&mut self) {
        if self.defused {
            return;
        }

        self.engine
            .computation_graph
            .computing
            .backward_projection_lock
            .remove(&self.query_id)
            .expect(
                "the pending backward projection lock guard has dropped and \
                 tried to remove existing lock, but no entry found",
            );
    }
}

pub struct Computing<C: Config> {
    computing_lock: DashMap<QueryID, Arc<QueryComputing>, C::BuildHasher>,
    backward_projection_lock:
        DashMap<QueryID, PendingBackwardProjection, C::BuildHasher>,
}

impl<C: Config> Computing<C> {
    pub fn new() -> Self {
        Self {
            computing_lock: DashMap::with_hasher_and_shard_amount(
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
    engine: Arc<Engine<C>>,
    this_computing: Arc<QueryComputing>,
    query_id: QueryID,
    defused: bool,
    computing_mode: ComputingMode,
}

impl<C: Config> ComputingLockGuard<C> {
    pub const fn computing_mode(&self) -> ComputingMode { self.computing_mode }

    pub const fn query_computing(&self) -> &Arc<QueryComputing> {
        &self.this_computing
    }

    /// Don't undo the computing lock when dropped.
    pub fn defuse(mut self) { self.defused = true; }
}

impl<C: Config> Drop for ComputingLockGuard<C> {
    fn drop(&mut self) {
        if self.defused {
            return;
        }

        self.engine
            .computation_graph
            .computing
            .computing_lock
            .remove(&self.query_id)
            .expect(
                "the computing lock guard has dropped and tried to remove \
                 existing computing lock, but no entry found",
            );
    }
}

impl<C: Config> Computing<C> {
    pub fn try_get_query_computing(
        &self,
        query_id: &QueryID,
    ) -> Option<Arc<QueryComputing>> {
        let guard = self.computing_lock.get(query_id)?;
        Some(guard.clone())
    }
}

pub enum WriteGuard<C: Config> {
    ComputingLockGuard(ComputingLockGuard<C>),
    BackwardProjectionLockGuard(BackwardProjectionLockGuard<C>),
}

impl<C: Config> Engine<C> {
    /// Exit early if a cyclic dependency is detected.
    pub fn exit_scc(
        &self,
        callee: &QueryID,
        caller_information: &CallerInformation,
    ) -> Result<(), CyclicError> {
        let Some(running_state) =
            self.computation_graph.computing.try_get_query_computing(callee)
        else {
            return Ok(());
        };

        // if there is no caller, we are at the root.
        let Some(query_caller) = caller_information.get_query_caller() else {
            return Ok(());
        };

        let is_in_scc =
            self.check_cyclic(&running_state, &query_caller.query_id());

        // mark the caller as being in scc
        if is_in_scc {
            let computing = query_caller.computing();
            computing.mark_scc();

            return Err(CyclicError);
        }

        Ok(())
    }

    /// Checks whether the stack of computing queries contains a cycle
    #[allow(clippy::needless_pass_by_value)]
    fn check_cyclic_internal(
        &self,
        computing: &QueryComputing,
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
            let Some(state) =
                self.computation_graph.computing.try_get_query_computing(&dep)
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
        running_state: &QueryComputing,
        target: &QueryID,
    ) -> bool {
        self.check_cyclic_internal(running_state, target)
    }

    pub(super) fn is_query_running_in_scc(
        &self,
        caller: Option<&QueryID>,
    ) -> Result<(), CyclicError> {
        let Some(called_from) = caller else {
            return Ok(());
        };

        let computing_lock = self
            .computation_graph
            .computing
            .computing_lock
            .get(called_from)
            .expect("computing lock should exist for caller")
            .clone();

        if computing_lock.is_in_scc() {
            return Err(CyclicError);
        }

        Ok(())
    }
}

impl<C: Config, Q: Query> Snapshot<C, Q> {
    async fn computing_lock_guard(
        &mut self,
        caller_information: &CallerInformation,
    ) -> Option<ComputingLockGuard<C>> {
        // IMPORTANT: here we move the retrival logic outside the lock guard
        // to avoid holding the lock across await points

        let last_verified = self.last_verified().await;

        let (mode, query_kind) = if let Some(last_verified) = last_verified {
            let kind = self.query_kind().await.unwrap();

            assert!(last_verified.0 != caller_information.timestamp());

            (ComputingMode::Repair, kind)
        } else {
            let executor =
                self.engine().executor_registry.get_executor_entry::<Q>();

            let style = executor.obtain_execution_style();

            (ComputingMode::Execute, QueryKind::Executable(style))
        };

        let entry = Arc::new(QueryComputing {
            callee_info: ComputingForwardEdges {
                callee_queries: DashMap::new(),
                callee_order: RwLock::new(Vec::new()),
            },

            query_kind,
            is_in_scc: Arc::new(AtomicBool::new(false)),
            tfc: DashSet::new(),
        });

        let result = match self
            .engine()
            .computation_graph
            .computing
            .computing_lock
            .entry(*self.query_id())
        {
            dashmap::Entry::Occupied(_) => {
                // there's some computing state already try again

                None
            }

            dashmap::Entry::Vacant(vacant_entry) => {
                vacant_entry.insert(entry.clone());

                Some(ComputingLockGuard {
                    engine: self.engine().clone(),
                    query_id: *self.query_id(),
                    defused: false,
                    computing_mode: mode,
                    this_computing: entry,
                })
            }
        };

        if result.is_some() {
            self.upgrade_to_exclusive().await;
        }

        result
    }

    pub(super) async fn get_backward_projection_lock_guard(
        &mut self,
    ) -> Option<BackwardProjectionLockGuard<C>> {
        let pending_backward_projection = PendingBackwardProjection;

        let result = match self
            .engine()
            .computation_graph
            .computing
            .backward_projection_lock
            .entry(*self.query_id())
        {
            dashmap::Entry::Occupied(_) => {
                // there's some computing state already try again
                None
            }

            dashmap::Entry::Vacant(vacant_entry) => {
                vacant_entry.insert(pending_backward_projection);

                Some(BackwardProjectionLockGuard {
                    engine: self.engine().clone(),
                    query_id: *self.query_id(),
                    defused: false,
                })
            }
        };

        if result.is_some() {
            self.upgrade_to_exclusive().await;
        }

        result
    }

    pub(super) async fn get_write_guard(
        &mut self,
        slow_path: SlowPath,
        caller_information: &CallerInformation,
    ) -> Option<WriteGuard<C>> {
        match slow_path {
            SlowPath::Computing => self
                .computing_lock_guard(caller_information)
                .await
                .map(WriteGuard::ComputingLockGuard),

            SlowPath::BaackwardProjection => self
                .get_backward_projection_lock_guard()
                .await
                .map(WriteGuard::BackwardProjectionLockGuard),
        }
    }
}

impl<C: Config, Q: Query> Snapshot<C, Q> {
    #[allow(clippy::option_option)]
    pub(super) async fn computing_lock_to_clean_query(
        mut self,
        clean_edges: Vec<QueryID>,
        new_tfc: Option<Interned<TransitiveFirewallCallees>>,
        caller_information: &CallerInformation,
        lock_guard: ComputingLockGuard<C>,
    ) {
        let timsestamp = caller_information.timestamp();

        async move {
            self.clean_query(clean_edges, new_tfc, timsestamp).await;

            self.engine()
                .computation_graph
                .computing
                .computing_lock
                .remove(self.query_id())
                .expect("should've existed");

            lock_guard.defuse();
        }
        .guarded()
        .await;
    }

    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn computing_lock_to_computed(
        &mut self,
        query: Q,
        value: Q::Value,
        query_value_fingerprint: Option<Compact128>,
        lock_guard: ComputingLockGuard<C>,
        has_pending_backward_projection: bool,
        current_timestamp: Timestamp,
        existing_forward_edges: Option<&[QueryID]>,
        continuing_tx: WriteTransaction<C>,
    ) {
        let (query_kind, forward_edge_order, forward_edges, tfc_archetype) = {
            (
                lock_guard.this_computing.query_kind,
                std::mem::take(
                    &mut *lock_guard
                        .this_computing
                        .callee_info
                        .callee_order
                        .write(),
                ),
                lock_guard
                    .this_computing
                    .callee_info
                    .callee_queries
                    .iter()
                    // in case of cyclic dependencies, some callees may have
                    // been aborted
                    .filter_map(|x| x.value().map(|v| (*x.key(), v)))
                    .collect(),
                self.engine().create_tfc_from_iter(
                    lock_guard.this_computing.tfc.iter().map(|x| *x.key()),
                ),
            )
        };

        self.set_computed(
            query,
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
        )
        .await;

        lock_guard.defuse();

        // done, remove the computing lock
        self.engine()
            .computation_graph
            .computing
            .computing_lock
            .remove(self.query_id());
    }
}
