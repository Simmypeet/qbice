use std::{
    self,
    collections::HashMap,
    sync::{Arc, atomic::AtomicBool},
};

use fxhash::FxBuildHasher;
use parking_lot::RwLock;
use qbice_stable_hash::Compact128;
use qbice_storage::intern::Interned;
use scc::hash_map::Entry;
use tokio::sync::{Notify, futures::OwnedNotified};

use crate::{
    Engine, ExecutionStyle, Query, TrackedEngine,
    config::{Config, WriteTransaction},
    engine::{
        computation_graph::{
            CallerInformation, QueryKind,
            database::{
                NodeDependency, NodeInfo, Observation, Snapshot, Timestamp,
            },
            slow_path::SlowPath,
            tfc_achetype::TransitiveFirewallCallees,
        },
        guard::GuardExt,
    },
    executor::CyclicError,
    query::QueryID,
};

#[derive(Debug, Default)]
pub struct CalleeOrder {
    pub order: Vec<NodeDependency>,
    pub mode: Mode,
}

impl CalleeOrder {
    pub fn push(&mut self, query_id: QueryID) {
        match self.mode {
            Mode::Single => {
                self.order.push(NodeDependency::Single(query_id));
            }

            Mode::Unordered => {
                self.order
                    .last_mut()
                    .unwrap()
                    .as_unordered_mut()
                    .unwrap()
                    .push(query_id);
            }
        }
    }

    pub fn clear(&mut self) {
        self.order.clear();
        self.mode = Mode::Single;
    }

    pub fn start_unordered_group(&mut self) {
        assert!(
            self.mode == Mode::Single,
            "Cannot start unordered callee group when already in unordered \
             mode"
        );
        self.mode = Mode::Unordered;
        self.order.push(NodeDependency::Unordered(Vec::new()));
    }

    pub fn end_unordered_group(&mut self) {
        assert!(
            self.mode == Mode::Unordered,
            "Cannot end unordered callee group when not in unordered mode"
        );
        self.mode = Mode::Single;
    }

    pub fn abort_callee(&mut self, callee: &QueryID) {
        for (i, dep) in self.order.iter_mut().enumerate() {
            match dep {
                NodeDependency::Single(qid) => {
                    if qid == callee {
                        self.order.remove(i);
                        return;
                    }
                }

                NodeDependency::Unordered(qids) => {
                    if let Some(pos) = qids.iter().position(|x| x == callee) {
                        qids.swap_remove(pos);
                        return;
                    }
                }
            }
        }
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum Mode {
    #[default]
    Single,

    Unordered,
}

#[derive(Debug, Default)]
pub struct ComputingForwardEdges {
    pub callee_queries:
        scc::HashMap<QueryID, Option<Observation>, FxBuildHasher>,
    pub callee_order: RwLock<CalleeOrder>,
}

impl QueryComputing {
    pub fn register_calee(&self, callee: &QueryID) {
        if self.callee_info.callee_queries.contains_sync(callee) {
            return;
        }

        match self.callee_info.callee_queries.entry_sync(*callee) {
            Entry::Occupied(_) => {}

            Entry::Vacant(vacant_entry) => {
                vacant_entry.insert_entry(None);

                self.callee_info.callee_order.write().push(*callee);
            }
        }
    }

    pub fn start_unordered_callee_group(&self) {
        self.callee_info.callee_order.write().start_unordered_group();
    }

    pub fn end_unordered_callee_group(&self) {
        self.callee_info.callee_order.write().end_unordered_group();
    }

    pub fn abort_callee(&self, callee: &QueryID) {
        assert!(self.callee_info.callee_queries.remove_sync(callee).is_some());

        let mut callee_order = self.callee_info.callee_order.write();

        callee_order.abort_callee(callee);
    }

    pub fn clear_dependencies(&self) {
        self.callee_info.callee_queries.clear_sync();
        self.callee_info.callee_order.write().clear();
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
            .get_sync(callee_target_id)
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
                    let _ = self.tfc.insert_sync(q);
                }
            }
            QueryKind::Executable(ExecutionStyle::Firewall) => {
                let _ = self.tfc.insert_sync(callee_id);
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
    notify: Arc<Notify>,
    callee_info: ComputingForwardEdges,
    is_in_scc: Arc<AtomicBool>,
    tfc: scc::HashSet<QueryID, FxBuildHasher>,
    query_kind: QueryKind,
}

impl QueryComputing {
    pub fn notified_owned(&self) -> OwnedNotified {
        self.notify.clone().notified_owned()
    }
}

#[derive(Debug)]
pub struct PendingBackwardProjection {
    notify: Arc<Notify>,
}

impl PendingBackwardProjection {
    pub fn notified_owned(&self) -> OwnedNotified {
        self.notify.clone().notified_owned()
    }
}

pub struct BackwardProjectionLockGuard<C: Config> {
    engine: Arc<Engine<C>>,
    query_id: QueryID,
    defused: bool,
}

impl<C: Config> BackwardProjectionLockGuard<C> {
    pub fn done(&mut self) {
        if self.defused {
            return;
        }

        self.defused = true;

        let entry = self
            .engine
            .computation_graph
            .computing
            .backward_projection_lock
            .remove_sync(&self.query_id)
            .expect(
                "the pending backward projection lock guard has dropped and \
                 tried to remove existing lock, but no entry found",
            );

        entry.1.notify.notify_waiters();
    }
}

impl<C: Config> Drop for BackwardProjectionLockGuard<C> {
    fn drop(&mut self) { self.done(); }
}

pub struct Computing<C: Config> {
    computing_lock: scc::HashMap<QueryID, Arc<QueryComputing>, C::BuildHasher>,
    backward_projection_lock:
        scc::HashMap<QueryID, PendingBackwardProjection, C::BuildHasher>,
}

impl<C: Config> Computing<C> {
    pub fn new() -> Self {
        Self {
            computing_lock: scc::HashMap::with_hasher(C::BuildHasher::default()),
            backward_projection_lock: scc::HashMap::with_hasher(
                C::BuildHasher::default(),
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
}

impl<C: Config> ComputingLockGuard<C> {
    pub fn done(&mut self) {
        if self.defused {
            return;
        }

        self.defused = true;

        let entry = self
            .engine
            .computation_graph
            .computing
            .computing_lock
            .remove_sync(&self.query_id)
            .expect(
                "the computing lock guard has dropped and tried to remove \
                 existing computing lock, but no entry found",
            );

        entry.1.notify.notify_waiters();
    }
}

impl<C: Config> Drop for ComputingLockGuard<C> {
    fn drop(&mut self) { self.done(); }
}

impl<C: Config> Computing<C> {
    pub fn try_get_query_computing(
        &self,
        query_id: &QueryID,
    ) -> Option<Arc<QueryComputing>> {
        self.computing_lock.read_sync(query_id, |_, v| v.clone())
    }

    pub fn try_get_notified_computing_lock(
        &self,
        query_id: &QueryID,
    ) -> Option<(OwnedNotified, Arc<QueryComputing>)> {
        self.computing_lock
            .read_sync(query_id, |_, v| (v.notified_owned(), v.clone()))
    }
}

pub enum WriteGuard<C: Config> {
    ComputingLockGuard(ComputingLockGuard<C>),
    BackwardProjectionLockGuard(BackwardProjectionLockGuard<C>),
}

impl<C: Config> Engine<C> {
    /// Exit early if a cyclic dependency is detected.
    pub(crate) async fn exit_scc(
        &self,
        callee: &QueryID,
        caller_information: &CallerInformation,
    ) -> Result<bool, CyclicError> {
        let Some((notified, running_state)) = self
            .computation_graph
            .computing
            .try_get_notified_computing_lock(callee)
        else {
            return Ok(true);
        };

        // if there is no caller, we are at the root.
        let Some(query_caller) = caller_information.get_query_caller() else {
            return Ok(true);
        };

        let is_in_scc =
            self.check_cyclic(&running_state, &query_caller.query_id());

        // mark the caller as being in scc
        if is_in_scc {
            let computing = query_caller.computing();
            computing.mark_scc();

            return Err(CyclicError);
        }

        notified.await;

        Ok(false)
    }

    /// Checks whether the stack of computing queries contains a cycle
    #[allow(clippy::needless_pass_by_value)]
    fn check_cyclic_internal(
        &self,
        computing: &QueryComputing,
        target: &QueryID,
    ) -> bool {
        if computing.callee_info.callee_queries.contains_sync(target) {
            computing
                .is_in_scc
                .store(true, std::sync::atomic::Ordering::SeqCst);

            return true;
        }

        let mut found = false;

        // OPTIMIZE: this can be parallelized
        computing.callee_info.callee_queries.iter_sync(|k, _| {
            let Some(state) =
                self.computation_graph.computing.try_get_query_computing(k)
            else {
                return true;
            };

            found |= self.check_cyclic_internal(&state, target);

            true
        });

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

        let is_in_scc = self
            .computation_graph
            .computing
            .computing_lock
            .read_sync(called_from, |_, v| v.is_in_scc())
            .expect("computing lock should exist for caller");

        if is_in_scc {
            return Err(CyclicError);
        }

        Ok(())
    }
}

impl<C: Config, Q: Query> Snapshot<C, Q> {
    async fn computing_lock_guard(
        mut self,
        caller_information: &CallerInformation,
    ) -> Option<(Self, ComputingLockGuard<C>)> {
        // we have to double check if we still really need to compute

        let last_verified = self.last_verified().await;

        let (mode, query_kind) = if let Some(last_verified) = last_verified {
            let kind = self.query_kind().await.unwrap();

            if last_verified.0 == caller_information.timestamp() {
                // no need to repair
                return None;
            }

            (ComputingMode::Repair, kind)
        } else {
            let executor =
                self.engine().executor_registry.get_executor_entry::<Q>();

            let style = executor.obtain_execution_style();

            (ComputingMode::Execute, QueryKind::Executable(style))
        };

        let entry = Arc::new(QueryComputing {
            callee_info: ComputingForwardEdges {
                callee_queries: scc::HashMap::with_hasher(
                    FxBuildHasher::default(),
                ),
                callee_order: RwLock::new(CalleeOrder::default()),
            },

            query_kind,
            notify: Arc::new(Notify::new()),
            is_in_scc: Arc::new(AtomicBool::new(false)),
            tfc: scc::HashSet::with_hasher(FxBuildHasher::default()),
        });

        let engine = self.engine().clone();
        let result = match engine
            .computation_graph
            .computing
            .computing_lock
            .entry_sync(*self.query_id())
        {
            Entry::Occupied(entry) => {
                // there's some computing state already try again
                let notified_owned = entry.get().notified_owned();

                drop(entry);
                drop(self);

                // wait for the existing computing to finish
                notified_owned.await;

                return None;
            }

            Entry::Vacant(vacant_entry) => {
                vacant_entry.insert_entry(entry.clone());

                Some(ComputingLockGuard {
                    engine: self.engine().clone(),
                    query_id: *self.query_id(),
                    defused: false,
                    computing_mode: mode,
                    this_computing: entry,
                })
            }
        };

        result.map(|x| (self, x))
    }

    pub(super) async fn get_backward_projection_lock_guard(
        mut self,
        caller_information: &CallerInformation,
    ) -> Option<(Self, BackwardProjectionLockGuard<C>)> {
        let pending_backward_projection =
            PendingBackwardProjection { notify: Arc::new(Notify::new()) };

        let engine = self.engine().clone();

        // double check if we really need to get the lock
        if self
            .pending_backward_projection()
            .await
            .is_none_or(|x| x.0 != caller_information.timestamp())
        {
            return None;
        }

        let result = match engine
            .computation_graph
            .computing
            .backward_projection_lock
            .entry_sync(*self.query_id())
        {
            Entry::Occupied(entry) => {
                let notified = entry.get().notified_owned();

                drop(entry);
                drop(self);

                // wait for the existing backward projection to finish
                notified.await;

                return None;
            }

            Entry::Vacant(vacant_entry) => {
                vacant_entry.insert_entry(pending_backward_projection);

                Some(BackwardProjectionLockGuard {
                    engine: self.engine().clone(),
                    query_id: *self.query_id(),
                    defused: false,
                })
            }
        };

        result.map(|x| (self, x))
    }

    pub(super) async fn get_write_guard(
        self,
        slow_path: SlowPath,
        caller_information: &CallerInformation,
    ) -> Option<(Self, WriteGuard<C>)> {
        match slow_path {
            SlowPath::Compute | SlowPath::Repair => self
                .computing_lock_guard(caller_information)
                .await
                .map(|x| (x.0, WriteGuard::ComputingLockGuard(x.1))),

            SlowPath::BaackwardProjection => self
                .get_backward_projection_lock_guard(caller_information)
                .await
                .map(|x| (x.0, WriteGuard::BackwardProjectionLockGuard(x.1))),
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
        mut lock_guard: ComputingLockGuard<C>,
    ) {
        self.upgrade_to_exclusive().await;
        let timsestamp = caller_information.timestamp();

        async move {
            self.clean_query(clean_edges, new_tfc, timsestamp).await;

            lock_guard.done();
        }
        .guarded()
        .await;
    }

    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn computing_lock_to_computed(
        self,
        query: Q,
        value: Q::Value,
        query_value_fingerprint: Option<Compact128>,
        mut lock_guard: ComputingLockGuard<C>,
        has_pending_backward_projection: bool,
        current_timestamp: Timestamp,
        existing_forward_edges: Option<&[NodeDependency]>,
        continuing_tx: WriteTransaction<C>,
    ) {
        let (query_kind, forward_edge_order, forward_edges, tfc_archetype) = {
            (
                lock_guard.this_computing.query_kind,
                std::mem::take(
                    &mut lock_guard
                        .this_computing
                        .callee_info
                        .callee_order
                        .write()
                        .order,
                ),
                {
                    let mut hash_map = HashMap::with_capacity_and_hasher(
                        lock_guard
                            .this_computing
                            .callee_info
                            .callee_queries
                            .len(),
                        C::BuildHasher::default(),
                    );

                    lock_guard
                        .this_computing
                        .callee_info
                        .callee_queries
                        .iter_sync(|k, v| {
                            if let Some(obs) = v {
                                hash_map.insert(*k, *obs);
                            }

                            true
                        });

                    hash_map
                },
                self.engine().create_tfc_from_scc_hash_set(
                    &lock_guard.this_computing.tfc,
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

        lock_guard.done();
    }
}

impl<C: Config> TrackedEngine<C> {
    /// Starts recording the dependency in an "unordered mode".
    ///
    /// ## Safety
    ///
    /// This method is purely for performance optimization. This allows the
    /// engine to concurrently repair the callees during the query repair phase.
    ///
    /// Normally, the engine needs to repair the callees in the order they were
    /// registered to ensure that the casual dependencies are respected (e.g.,
    /// if query `A` says `true`, don't call query `B`).
    ///
    /// However, in some scenarios, those casual dependencies are not necessary,
    /// and the engine can safely repair those queries in any order. For
    /// example, a query that aggregates the results of multiple independent
    /// queries can safely repair those independent queries in any order.
    ///
    /// ## Caution
    ///
    /// This should be used sparingly and only when you are certain that there
    /// are no causal dependencies between the queries being called.
    ///
    /// For example, let's say we have a query ("Divide") that divides two
    /// numbers, and it calls two other input queries ("Numerator" and
    /// "Denominator"). This query panics if the denominator is zero.
    ///
    /// From that, we build an another query ("SafeDivide") that calls "Divide"
    /// but returns `None` if the denominator is zero. In order for "SafeDivide"
    /// to work it checks the value of "Denominator" first, and if it's zero, it
    /// does not call "Divide" because it would panic.
    ///
    /// The above scenario is an example of a causal dependency: the value of
    /// "SafeDivide" depends on the value of "Denominator" to decide whether it
    /// can call "Divide" safely or not. By default, the engine would repair
    /// "Denominator" first, and then "Divide" if "Denominator" result does not
    /// differ.
    ///
    /// ## Panics
    ///
    /// This method will panic if called outside the context of an `Executor`
    /// computing a query or it's already in unordered mode.
    pub unsafe fn start_unordered_callee_group(&self) {
        self.caller
            .get_query_caller()
            .unwrap_or_else(|| {
                panic!(
                    "This method is only available in the context of \
                     `Executor` computing the query"
                )
            })
            .computing()
            .start_unordered_callee_group();
    }

    /// Ends recording the dependency in an "unordered mode".
    ///
    /// ## Safety
    ///
    /// See [`Self::start_unordered_callee_group`] for details.
    ///
    /// ## Panics
    ///
    /// This method will panic if called outside the context of an `Executor`
    /// computing a query or it's not in unordered mode.
    pub unsafe fn end_unordered_callee_group(&self) {
        self.caller
            .get_query_caller()
            .unwrap_or_else(|| {
                panic!(
                    "This method is only available in the context of \
                     `Executor` computing the query"
                )
            })
            .computing()
            .end_unordered_callee_group();
    }
}
