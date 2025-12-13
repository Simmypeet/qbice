use std::{
    any::Any,
    collections::{HashMap, VecDeque, hash_map::Entry},
    ops::Not,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use dashmap::{DashMap, DashSet, mapref::one::Ref};
use enum_as_inner::EnumAsInner;
use parking_lot::{RwLock, RwLockReadGuard};
use tokio::sync::Notify;

use crate::{
    config::Config,
    engine::{
        Engine, InitialSeed, TrackedEngine,
        database::{
            Database, Timtestamp,
            storage::{ComputingLockGuard, SetInputResult},
            tfc_archetype::{TfcArchetypeID, TfcSet},
        },
        fingerprint::Compact128,
    },
    executor::CyclicError,
    query::{
        DynQuery, DynQueryBox, DynValue, DynValueBox, ExecutionStyle, Query,
        QueryID,
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallerReason {
    RequireValue,
    Repair,

    /// This occurs when a projection got invoked with
    /// `BackwardProjectionPropagation` and now the projection itself has to
    /// recompute and in the process it has to call another query with this
    /// flag.
    ///
    /// ```txt
    ///       Firewall1                Firewall2
    ///          ^                        ^
    ///          |                        |
    ///          +------- Projection  ----+
    ///                    ^^^^^^^
    ///                    Got backward propagated and now have to recompute
    ///                    itself and call Firewall1 and Firewall2 with
    ///                    this flag.
    /// ```
    ProjectionRecomputingDueToBackwardPropagation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueryCaller {
    query_id: QueryID,
    reason: CallerReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallerInformation {
    User,
    Query(QueryCaller),

    /// The caller is either a firewall or projection query. The caller calls
    /// the query when the caller itself has changed its value and has to
    /// invoke all of its backward projections (projections that use the caller)
    /// in order to propagate the dirtiness.
    ///
    /// ```txt
    ///         Firewall <-- caller has changed its value
    ///          ^    ^
    ///         /      \
    ///   Projection1  Projection2 <-- both got called with
    ///                                 `BackwardProjectionPropagation`
    /// ```
    BackwardProjectionPropagation,

    RepairFirewall,
}

impl CallerInformation {
    pub const fn get_caller(&self) -> Option<&QueryID> {
        match self {
            Self::RepairFirewall
            | Self::BackwardProjectionPropagation
            | Self::User => None,

            Self::Query(q) => Some(&q.query_id),
        }
    }

    pub const fn require_value(&self) -> bool {
        match self {
            Self::RepairFirewall | Self::BackwardProjectionPropagation => false,

            Self::User => true,
            Self::Query(q) => matches!(q.reason, CallerReason::RequireValue
                | CallerReason::ProjectionRecomputingDueToBackwardPropagation
            ),
        }
    }

    pub fn has_a_caller_requiring_value(&self) -> Option<&QueryID> {
        match self {
            // it does require value, but the caller is not another query
            Self::RepairFirewall
            | Self::User
            | Self::BackwardProjectionPropagation => None,

            Self::Query(q) => {
                matches!(q.reason,
                    CallerReason::RequireValue
                    | CallerReason::ProjectionRecomputingDueToBackwardPropagation
                ).then_some(&q.query_id)
            }
        }
    }
}

/// Specifies whether the query is has been repaired or is up-to-date.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum QueryRepairation {
    /// The query verification timestamp was older than the current timestamp.
    /// The query has been repaired to be up-to-date (this doesn't imply that
    /// the query has been recomputed)
    Repaired,

    /// The verification timestamp was already up-to-date.
    UpToDate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryResult<V> {
    Return(V, QueryRepairation),
    Checked(QueryRepairation),
}

impl<V> QueryResult<V> {
    pub fn unwrap_return(self) -> V {
        match self {
            Self::Return(v, _) => v,
            Self::Checked(_) => {
                panic!("called `unwrap_return` on a `UpToDate` value")
            }
        }
    }

    /// Type erases the type parameter `V` into a `DynValueBox<C>`.
    pub fn into_dyn_query_result<C: Config>(self) -> QueryResult<DynValueBox<C>>
    where
        V: DynValue<C>,
    {
        match self {
            Self::Return(v, c) => {
                QueryResult::Return(smallbox::smallbox!(v), c)
            }
            Self::Checked(c) => QueryResult::Checked(c),
        }
    }
}

#[derive(Debug)]
pub struct ReadyComputed<C: Config> {
    pub value_fingerprint: Compact128,
    pub tfc_archetype: Option<TfcArchetypeID>,
    pub value: DynValueBox<C>,
    pub next_version: Timtestamp,
}

#[derive(Debug, EnumAsInner)]
pub enum ComputingState<C: Config> {
    Fresh,
    Repair,
    ForBackwardPropagation(ReadyComputed<C>),
}

#[derive(Debug)]
pub struct Computing<C: Config> {
    notification: Arc<Notify>,

    // We use a RwLock here since we want to avoid acquiring a write lock
    // when observing callees.
    callee_info: RwLock<CalleeInfo>,

    computed_placeholder_for_firewall_dirty_propagation: Option<Computed<C>>,
    state: ComputingState<C>,

    is_in_scc: AtomicBool,
}

impl<C: Config> Computing<C> {
    pub fn notification_owned(&self) -> Arc<Notify> {
        self.notification.clone()
    }

    pub fn callee_info(&self) -> RwLockReadGuard<'_, CalleeInfo> {
        self.callee_info.read()
    }

    pub const fn take_computed_placeholder_for_firewall_dirty_propagation(
        &mut self,
    ) -> Option<Computed<C>> {
        self.computed_placeholder_for_firewall_dirty_propagation.take()
    }

    pub fn new(notification: Arc<Notify>, state: ComputingState<C>) -> Self {
        Self {
            notification,
            computed_placeholder_for_firewall_dirty_propagation: None,
            state,
            callee_info: RwLock::new(CalleeInfo {
                callee_queries: HashMap::new(),
                callee_order: Vec::new(),
                tfc_archetype: None,
            }),
            is_in_scc: AtomicBool::new(false),
        }
    }

    pub fn add_callee(&self, calee: QueryID) {
        let mut callee_info = self.callee_info.write();

        match callee_info.callee_queries.entry(calee) {
            Entry::Occupied(_) => {}
            Entry::Vacant(v) => {
                v.insert(None);

                // if haven't inserted, add to dependency order
                callee_info.callee_order.push(calee);
            }
        }
    }

    pub fn remove_callee(&self, callee: &QueryID) {
        let mut callee_info = self.callee_info.write();

        assert!(callee_info.callee_queries.remove(callee).is_some());
        let pos =
            callee_info.callee_order.iter().position(|x| x == callee).unwrap();

        assert!(callee_info.callee_order.remove(pos) == *callee);
    }
}

#[derive(Debug)]
pub struct Computed<C: Config> {
    result: DynValueBox<C>,
    callee_info: CalleeInfo,
    verified_at: Timtestamp,

    value_fingerprint: Compact128,
}

impl<C: Config> Computed<C> {
    pub const fn verified_at(&self) -> Timtestamp { self.verified_at }

    /// Returns the callee info containing forward dependencies.
    pub const fn callee_info(&self) -> &CalleeInfo { &self.callee_info }

    /// Returns the computed result value.
    pub fn result(&self) -> &dyn crate::query::DynValue<C> { &*self.result }
}

#[derive(Debug, EnumAsInner)]
pub enum State<C: Config> {
    Computing(Computing<C>),
    Computed(Computed<C>),
}

#[derive(Debug)]
pub struct Observation {
    seen_value_fingerprint: Compact128,
    seen_transitive_firewall_callees_fingerprint: Option<TfcArchetypeID>,

    // Dirty flag is used as an atomic because we want to avoid acquiring
    // write lock when marking as dirty when propagating dirtiness.
    dirty: AtomicBool,
}

#[derive(Debug, Default)]
pub struct CalleeInfo {
    callee_queries: HashMap<QueryID, Option<Observation>>,
    callee_order: Vec<QueryID>,
    tfc_archetype: Option<TfcArchetypeID>,
}

impl CalleeInfo {
    /// Returns an iterator over callee query IDs (forward dependencies) along
    /// with their dirty flag status.
    ///
    /// Each item is a tuple of `(QueryID, Option<bool>)` where:
    /// - The `QueryID` is the callee's identifier
    /// - The `Option<bool>` indicates the dirty status:
    ///   - `Some(true)` if the dependency is dirty (needs revalidation)
    ///   - `Some(false)` if the dependency is clean (validated)
    ///   - `None` if the observation status is unknown
    pub fn callee_order(
        &self,
    ) -> impl Iterator<Item = (QueryID, Option<bool>)> + '_ {
        self.callee_order.iter().map(|callee_id| {
            let is_dirty = self
                .callee_queries
                .get(callee_id)
                .and_then(|obs| obs.as_ref())
                .map(|obs| {
                    obs.dirty.load(std::sync::atomic::Ordering::Relaxed)
                });
            (*callee_id, is_dirty)
        })
    }
}

#[derive(Debug, Default)]
pub struct CallerInfo {
    caller_queries: DashSet<QueryID>,
}

impl CallerInfo {
    /// Returns an iterator over all caller query IDs.
    pub fn iter(&self) -> impl Iterator<Item = QueryID> + '_ {
        self.caller_queries.iter().map(|r| *r.key())
    }

    /// Inserts a caller query ID.
    pub(super) fn insert(&self, query_id: QueryID) {
        self.caller_queries.insert(query_id);
    }

    /// Removes a caller query ID.
    pub(super) fn remove(&self, query_id: &QueryID) {
        self.caller_queries.remove(query_id);
    }
}

pub struct QueryMeta<C: Config> {
    original_key: DynQueryBox<C>,
    caller_info: CallerInfo,

    query_kind: QueryKind,
    state: State<C>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum QueryKind {
    Input,
    Execute(ExecutionStyle),
}

impl<C: Config> QueryMeta<C> {
    pub fn new<Q>(
        original_key: Q,
        state: State<C>,
        execution_style: ExecutionStyle,
    ) -> Self
    where
        Q: Query,
    {
        Self {
            original_key: smallbox::smallbox!(original_key),
            caller_info: CallerInfo::default(),
            query_kind: QueryKind::Execute(execution_style),
            state,
        }
    }

    pub fn new_input<Q: Query>(
        query_key: Q,
        query_value: Q::Value,
        timestamp: &Timtestamp,
        initial_seed: InitialSeed,
    ) -> Self {
        let hash_128 =
            DynValue::<C>::hash_128_value(&query_value, initial_seed);

        Self {
            original_key: smallbox::smallbox!(query_key),
            caller_info: CallerInfo::default(),
            query_kind: QueryKind::Input,
            state: State::Computed(Computed {
                result: smallbox::smallbox!(query_value),
                callee_info: CalleeInfo::default(),
                verified_at: *timestamp,
                value_fingerprint: Compact128::from_u128(hash_128),
            }),
        }
    }

    pub fn set_input<Q: Query>(
        &mut self,
        query_value: Q::Value,
        has_already_incremented_timestamp: bool,
        initial_seed: InitialSeed,
        timestamp: &mut Timtestamp,
    ) -> SetInputResult {
        let mut this_has_incremented = false;

        self.query_kind = QueryKind::Input;

        let computed =
            self.state.as_computed_mut().expect("should've been computed");

        let hash = Compact128::from_u128(DynValue::<C>::hash_128_value(
            &query_value,
            initial_seed,
        ));

        let fingerprint_diff = hash != computed.value_fingerprint;

        // if haven't incremented and the hash is different,
        // update the timestamp
        if !has_already_incremented_timestamp && fingerprint_diff {
            timestamp.increment();
            computed.verified_at = *timestamp;

            this_has_incremented = true;
        }

        // delete all callees
        computed.callee_info = CalleeInfo::default();
        computed.result = smallbox::smallbox!(query_value);
        computed.value_fingerprint = hash;

        SetInputResult { incremented: this_has_incremented, fingerprint_diff }
    }

    pub fn get_computing(&self) -> &Computing<C> {
        self.state.as_computing().expect("should be computing")
    }

    pub fn get_computing_mut(&mut self) -> &mut Computing<C> {
        self.state.as_computing_mut().expect("should be computing")
    }

    pub fn get_computed_mut(&mut self) -> &mut Computed<C> {
        self.state.as_computed_mut().expect("should be computed")
    }

    pub fn get_computed(&self) -> &Computed<C> {
        self.state.as_computed().expect("should be computed")
    }

    pub const fn state(&self) -> &State<C> { &self.state }

    /// Returns a reference to the original query key.
    pub fn original_key(&self) -> &dyn crate::query::DynQuery<C> {
        &*self.original_key
    }

    pub fn is_projection(&self) -> bool {
        self.query_kind == QueryKind::Execute(ExecutionStyle::Projection)
    }

    /// Returns whether this query is an input query.
    pub fn is_input(&self) -> bool { self.query_kind == QueryKind::Input }

    pub fn take_state_mut(&mut self, f: impl FnOnce(State<C>) -> State<C>) {
        take_mut::take(&mut self.state, f);
    }

    pub const fn replace_state(&mut self, state: State<C>) -> State<C> {
        std::mem::replace(&mut self.state, state)
    }
}

impl<C: Config> QueryMeta<C> {
    pub fn is_running_in_scc(&self) -> bool {
        let computing = self
            .state
            .as_computing()
            .expect("can only check scc while computing");

        computing.is_in_scc.load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[derive(Debug, Clone)]
pub struct QueryWithID<'c, Q: Query> {
    id: QueryID,
    query: &'c Q,
}

impl<Q: Query> QueryWithID<'_, Q> {
    pub const fn id(&self) -> &QueryID { &self.id }
    pub const fn query(&self) -> &Q { self.query }
}

impl<C: Config> Database<C> {
    /// Create a new query with its associated unique identifier.
    pub(super) fn new_query_with_id<'c, Q: Query>(
        &'c self,
        query: &'c Q,
    ) -> QueryWithID<'c, Q> {
        QueryWithID {
            id: DynQuery::<C>::query_identifier(query, self.initial_seed()),
            query,
        }
    }

    pub(super) fn query_id<Q: Query>(&self, query: &Q) -> QueryID {
        DynQuery::<C>::query_identifier(query, self.initial_seed())
    }
}

enum CheckProjectionQueryGotBackwardPropagate<V> {
    GoToWait,
    Done(Option<V>),
}

/// The result of attempting a fast-path query execution.
pub enum FastPathResult<V> {
    TryAgain,
    ToSlowPath,
    Hit(Option<V>),
}

impl<C: Config> Database<C> {
    /// Add both forward and backward dependencies for both caller and callee.
    fn observe_callee_fingerprint(
        &self,
        computed_fingerprint: Compact128,
        computed_tfc_archetype: Option<TfcArchetypeID>,
        callee_meta: &QueryMeta<C>,
        callee_target: &QueryID,
        caller_source: &QueryID,
        callee_kind: QueryKind,
    ) {
        // add dependency for the caller
        let caller_computing = self.get_computing(caller_source);
        let mut caller_callee_info = caller_computing.callee_info.write();
        {
            assert!(
                caller_callee_info
                    .callee_queries
                    .insert(
                        *callee_target,
                        Some(Observation {
                            seen_value_fingerprint: computed_fingerprint,
                            dirty: AtomicBool::new(false),
                            seen_transitive_firewall_callees_fingerprint:
                                computed_tfc_archetype,
                        }),
                    )
                    .is_some(),
                "should've been pre-inserted in `query_for`"
            );

            match callee_kind {
                QueryKind::Input
                | QueryKind::Execute(
                    ExecutionStyle::Normal | ExecutionStyle::Projection,
                ) => {
                    caller_callee_info.tfc_archetype = self.union_tfcs(
                        caller_callee_info
                            .tfc_archetype
                            .into_iter()
                            .chain(computed_tfc_archetype),
                    );
                }
                QueryKind::Execute(ExecutionStyle::Firewall) => {
                    let singleton_tfc = self.new_singleton_tfc(*callee_target);
                    caller_callee_info.tfc_archetype = self.union_tfcs(
                        caller_callee_info
                            .tfc_archetype
                            .into_iter()
                            .chain(std::iter::once(singleton_tfc)),
                    );
                }
            }
        }

        // add dependency for the callee
        callee_meta.caller_info.insert(*caller_source);
    }

    /// Checks whether the stack of computing queries contains a cycle
    fn check_cyclic(&self, computing: &Computing<C>, target: QueryID) -> bool {
        let caller_callee_info = computing.callee_info();
        if caller_callee_info.callee_queries.contains_key(&target) {
            computing
                .is_in_scc
                .store(true, std::sync::atomic::Ordering::Relaxed);

            return true;
        }

        let mut found = false;

        // OPTIMIZE: this can be parallelized
        for dep in caller_callee_info.callee_queries.keys() {
            let Some(state) = self.try_get_read_meta(dep) else {
                continue;
            };

            let State::Computing(state) = &state.state else {
                continue;
            };

            found |= self.check_cyclic(state, target);
        }

        if found {
            computing
                .is_in_scc
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }

        found
    }

    /// Exit early if a cyclic dependency is detected.
    fn exit_scc(
        &self,
        called_from: Option<&QueryID>,
        running_state: &Computing<C>,
    ) -> Result<(), CyclicError> {
        // if there is no caller, we are at the root.
        let Some(called_from) = called_from else {
            return Ok(());
        };

        let is_in_scc = self.check_cyclic(running_state, *called_from);

        // mark the caller as being in scc
        if is_in_scc {
            let meta = self.get_computing(called_from);

            meta.is_in_scc.store(true, std::sync::atomic::Ordering::SeqCst);

            return Err(CyclicError);
        }

        Ok(())
    }

    fn check_projection_query_got_backward_propagated<
        V: Any + Send + Sync + Clone,
    >(
        &self,
        callee_computing: &Computing<C>,
        callee_meta: &QueryMeta<C>,
        callee_target_id: &QueryID,
        caller: &CallerInformation,
        required_value: bool,
    ) -> CheckProjectionQueryGotBackwardPropagate<V> {
        // this check only applies to caller being a projection query
        let CallerInformation::Query(caller_query) = caller else {
            return CheckProjectionQueryGotBackwardPropagate::GoToWait;
        };

        // must be only a projection query that called with backward propagation
        // flag to be able for accessing the ready computed value.
        if caller_query.reason
            != CallerReason::ProjectionRecomputingDueToBackwardPropagation
        {
            return CheckProjectionQueryGotBackwardPropagate::GoToWait;
        }

        match &callee_computing.state {
            ComputingState::Fresh | ComputingState::Repair => {
                CheckProjectionQueryGotBackwardPropagate::GoToWait
            }

            // the value is ready for backward propagation, use this query
            // and return right away.
            ComputingState::ForBackwardPropagation(mini_copmuted) => {
                self.observe_callee_fingerprint(
                    mini_copmuted.value_fingerprint,
                    mini_copmuted.tfc_archetype,
                    callee_meta,
                    callee_target_id,
                    &caller_query.query_id,
                    callee_meta.query_kind,
                );

                CheckProjectionQueryGotBackwardPropagate::Done(
                    if required_value {
                        Some(
                            mini_copmuted
                                .value
                                .downcast_value::<V>()
                                .unwrap()
                                .clone(),
                        )
                    } else {
                        None
                    },
                )
            }
        }
    }

    /// Attempt to fast-path the query execution. Should acquire only a read
    /// lock, no write lock involved.
    pub(super) async fn fast_path<V: 'static + Send + Sync + Clone>(
        &self,
        query_id: &QueryID,
        caller: &CallerInformation,
    ) -> Result<FastPathResult<V>, CyclicError> {
        // checks if the query result is already computed
        let Some(callee_target_meta) = self.try_get_read_meta(query_id) else {
            return Ok(FastPathResult::ToSlowPath);
        };

        match &callee_target_meta.state {
            State::Computing(computing) => {
                self.exit_scc(caller.get_caller(), computing)?;

                // Special case handling for projection queries that can be
                // backward propagated.
                match self.check_projection_query_got_backward_propagated::<V>(
                    computing,
                    &callee_target_meta,
                    query_id,
                    caller,
                    caller.require_value(),
                ) {
                    // proceed to waiting
                    CheckProjectionQueryGotBackwardPropagate::GoToWait => {}
                    CheckProjectionQueryGotBackwardPropagate::Done(res) => {
                        return Ok(FastPathResult::Hit(res));
                    }
                }

                let notify = computing.notification.clone();

                // IMPORTANT: add the current thread to the waiter list
                // first before dropping the state read lock to avoid the
                // notification being sent before the thread is added to the
                // waiters list.
                let notified = notify.notified();

                // drop the read lock to allow the thread that is computing
                // the query to access the state and notify the waiters.
                drop(callee_target_meta);

                notified.await;

                // try again after being notified
                Ok(FastPathResult::TryAgain)
            }

            State::Computed(computed) => {
                // the query isn't verified at the current timestamp
                if computed.verified_at != self.current_timestamp() {
                    return Ok(FastPathResult::ToSlowPath);
                }

                if let Some(caller) = caller.has_a_caller_requiring_value() {
                    // register dependency
                    self.observe_callee_fingerprint(
                        computed.value_fingerprint,
                        computed.callee_info.tfc_archetype,
                        &callee_target_meta,
                        query_id,
                        caller,
                        callee_target_meta.query_kind,
                    );
                }

                Ok(FastPathResult::Hit(if caller.require_value() {
                    Some(computed.result.downcast_value::<V>().unwrap().clone())
                } else {
                    None
                }))
            }
        }
    }

    // Done computing the value, defuse the computing lock and set the computed
    // state
    fn done_compute(
        &self,
        calling_query: &QueryID,
        ready_computed: ReadyComputed<C>,
        lock_computing: ComputingLockGuard<'_, C>,
    ) {
        self.set_computed_from_computing_and_defuse(
            calling_query,
            |computing| Computed {
                result: ready_computed.value,
                callee_info: computing.callee_info.into_inner(),
                verified_at: ready_computed.next_version,
                value_fingerprint: ready_computed.value_fingerprint,
            },
            lock_computing,
        );
    }

    fn done_compute_from_backward_propagation(
        &self,
        calling_query: &QueryID,
        lock_computing: ComputingLockGuard<'_, C>,
    ) {
        self.set_computed_from_computing_and_defuse(
            calling_query,
            |computing| {
                let ready_computed =
                    computing.state.into_for_backward_propagation().unwrap();

                Computed {
                    result: ready_computed.value,
                    callee_info: computing.callee_info.into_inner(),
                    verified_at: ready_computed.next_version,
                    value_fingerprint: ready_computed.value_fingerprint,
                }
            },
            lock_computing,
        );
    }

    pub(super) fn dirty_queries(&mut self, mut queries: VecDeque<QueryID>) {
        // clean up the dirtied queries set
        self.clear_dirty_queries();
        self.clear_statistics();

        // OPTIMIZE: we could potentially have worker threads to process dirty
        // queries
        while let Some(query_id) = queries.pop_front() {
            if !self.insert_dirty_query(query_id) {
                // already dirtied, skip
                continue;
            }

            let meta = self.get_read_meta(&query_id);

            // iterate through all callers and mark their
            // observations as dirty
            for caller in meta.caller_info.iter() {
                let caller_meta = self.get_read_meta(&caller);
                let caller_computed = caller_meta.get_computed();

                self.propagate_dirty_from_computed_caller(
                    caller,
                    caller_computed,
                    caller_meta.query_kind,
                    &query_id,
                    &mut queries,
                );
            }
        }
    }

    pub fn unwire_backward_dependency(
        &self,
        caller_source: &QueryID,
        callee_target: &QueryID,
    ) {
        let callee_meta = self.get_read_meta(callee_target);
        callee_meta.caller_info.remove(caller_source);
    }

    pub fn unwire_backward_dependencies_from_callee(
        &self,
        query_id: &QueryID,
        callee_info: &CalleeInfo,
    ) {
        // OPTIMIZE: this can be parallelized
        for callee in &callee_info.callee_order {
            self.unwire_backward_dependency(query_id, callee);
        }
    }

    fn is_query_input(&self, query_id: &QueryID) -> bool {
        let meta = self.get_read_meta(query_id);
        meta.is_input()
    }

    fn propagate_dirtiness_from_firewall(&self, query_id: &QueryID) {
        let mut work_queue = VecDeque::new();
        work_queue.push_back(*query_id);

        while let Some(current_query_id) = work_queue.pop_front() {
            if !self.insert_dirty_query(current_query_id) {
                // has already dirtied, skip
                continue;
            }

            let current_meta = self.get_read_meta(&current_query_id);
            for caller in current_meta.caller_info.iter() {
                let caller_meta = self.get_read_meta(&caller);

                match &caller_meta.state {
                    // if it's computing, it must have been waiting for this
                    // firewall to complete repairation, so we directly access
                    // the placeholder computed value for dirty propagation.
                    State::Computing(computing) => {
                        let caller_computed = computing
                            .computed_placeholder_for_firewall_dirty_propagation
                            .as_ref()
                            .expect("should be present");

                        self.propagate_dirty_from_computed_caller(
                            caller,
                            caller_computed,
                            caller_meta.query_kind,
                            &current_query_id,
                            &mut work_queue,
                        );
                    }

                    State::Computed(computed) => {
                        self.propagate_dirty_from_computed_caller(
                            caller,
                            computed,
                            caller_meta.query_kind,
                            &current_query_id,
                            &mut work_queue,
                        );
                    }
                }
            }
        }
    }

    fn propagate_dirty_from_computed_caller(
        &self,
        target: QueryID,
        computed: &Computed<C>,
        kind: QueryKind,
        from: &QueryID,
        work_queue: &mut VecDeque<QueryID>,
    ) {
        let obs = computed
            .callee_info
            .callee_queries
            .get(from)
            .expect("should be present");

        let obs = obs.as_ref().expect("should be present");
        let old = obs.dirty.swap(true, std::sync::atomic::Ordering::SeqCst);

        // from clean to dirty
        if !old {
            self.increment_dirtied_edges();
        }

        // if this is a firewall or projection, stop propagating further
        if matches!(
            kind,
            QueryKind::Execute(
                ExecutionStyle::Firewall | ExecutionStyle::Projection
            )
        ) {
            return;
        }

        work_queue.push_back(target);
    }
}

struct ComputedMeta {
    pub value_fingerprint: Compact128,
    pub tfc_archetype: Option<TfcArchetypeID>,
}

impl<C: Config> Database<C> {
    fn get_computed_meta(&self, current_id: QueryID) -> ComputedMeta {
        let meta = self.get_read_meta(&current_id);
        match &meta.state {
            State::Computing(computing) => match &computing.state {
                ComputingState::Fresh | ComputingState::Repair => {
                    unreachable!("should at least be ready to computed")
                }
                ComputingState::ForBackwardPropagation(ready_to_computed) => {
                    assert!(
                        matches!(
                            meta.query_kind,
                            QueryKind::Execute(
                                ExecutionStyle::Firewall
                                    | ExecutionStyle::Projection
                            )
                        ),
                        "only projection and firewall queries are allowed to \
                         peak into ReadyComputed prematurely"
                    );

                    ComputedMeta {
                        value_fingerprint: ready_to_computed.value_fingerprint,
                        tfc_archetype: Some(self.new_singleton_tfc(current_id)),
                    }
                }
            },

            State::Computed(computed) => ComputedMeta {
                value_fingerprint: computed.value_fingerprint,
                tfc_archetype: if meta.query_kind
                    == QueryKind::Execute(ExecutionStyle::Firewall)
                {
                    Some(self.new_singleton_tfc(current_id))
                } else {
                    computed.callee_info.tfc_archetype
                },
            },
        }
    }
}

#[derive(Debug)]
pub enum RepairDecision {
    Recompute,
    Clean { repair_transitive_firewall_callees: bool },
}

pub enum ExecuteQueryFor {
    FreshQuery,
    RecomputeQuery,
}

impl<C: Config> Engine<C> {
    async fn dynamic_repair_query(
        self: &Arc<Self>,
        callee_target_query_id: &QueryID,
        caller_information: &CallerInformation,
    ) -> Result<QueryResult<DynValueBox<C>>, CyclicError> {
        // NOTE: we clone the original key here since we cannot hold the
        // read lock on the query metas while the repair is happening.
        let original_callee_value = self
            .database
            .get_read_meta(callee_target_query_id)
            .original_key
            .dyn_clone();

        // NOTE: we must repair the callee through executor registry
        // since it's impossible to statically know the
        // type of the query here.
        let entry = self.executor_registry.get_executor_entry_by_type_id(
            &callee_target_query_id.stable_type_id(),
        );

        let callee_value: &dyn DynQuery<C> = &*original_callee_value;
        let callee_value_any = callee_value as &(dyn Any + Send + Sync);

        entry.invoke_query_for(self, callee_value_any, caller_information).await
    }

    #[tracing::instrument(skip(self, computed_callee), ret, level = "info")]
    pub(super) async fn should_recompute_query_based_on_callee<Q: Query>(
        self: &Arc<Self>,
        query_id: &QueryWithID<'_, Q>,
        computed_callee: &Computed<C>,
    ) -> RepairDecision {
        let mut repair_transitive_firewall_callees = false;

        // search for queries that are dirty
        for callee in &computed_callee.callee_info.callee_order {
            // skip if not dirty
            {
                let obs = computed_callee
                    .callee_info
                    .callee_queries
                    .get(callee)
                    .expect("should be present");

                let obs =
                    obs.as_ref().expect("should've been set in the fast path");

                if obs.dirty.load(Ordering::SeqCst).not() {
                    continue;
                }
            }

            // NOTE: if the callee is an input (explicitly set), it's impossible
            // to try to repair it, so we'll skip repairing and directly
            // compare the fingerprint.
            if !self.database.is_query_input(callee) {
                // recursively repair the callee.
                let _ = self
                    .dynamic_repair_query(
                        callee,
                        &CallerInformation::Query(QueryCaller {
                            query_id: query_id.id,
                            reason: CallerReason::Repair,
                        }),
                    )
                    .await;
            }

            let callee_computed = self.database.get_computed_meta(*callee);

            // depend on whether the observed fingerprint has changed
            {
                let obs = computed_callee
                    .callee_info
                    .callee_queries
                    .get(callee)
                    .expect("should be present");

                let obs = obs.as_ref().expect("should be present");

                let new_fingerprint = callee_computed.value_fingerprint;
                let new_tfc_fingerprint = callee_computed.tfc_archetype;

                if obs.seen_transitive_firewall_callees_fingerprint
                    != new_tfc_fingerprint
                {
                    repair_transitive_firewall_callees = true;
                }

                // fingerprint is the same, mark as clean
                if obs.seen_value_fingerprint != new_fingerprint {
                    // fingerprint changed, need to recompute this node, stop
                    // repairing further callees
                    return RepairDecision::Recompute;
                }
            }
        }

        // all callees are clean, no need to recompute
        RepairDecision::Clean { repair_transitive_firewall_callees }
    }

    async fn repair_transitive_firewall_callees_tfc_set(
        self: &Arc<Self>,
        tfc_set: &TfcSet,
    ) {
        // OPTIMIZE: this can be parallelized
        for callee in tfc_set {
            // has to directly repair since the chain of firewall might happen
            // and the dirty flag might have not yet propagated
            //
            //               Input
            //                 | *
            //                 V
            //             Firewall
            //                 |
            //                 V
            //             Firewall2
            //                 |
            //                 V
            //               .....
            //                 |
            //                 V
            //           CurrentQuery
            //
            // NOTE: The asterisked on the edge indicates dirty edge.
            //
            // Here the `CurrentQuery`` have a transitive dependency on
            // `Firewall2`. However, due to the `Firewall` node, the dirty
            // flag doesn't propagate to `Firewall2` and thus the `CurrentQuery`
            // cannot observe the dirty flag on `Firewall2`. Therefore, we have
            // to immediately repair the `Firewall2` node to ensure that the
            // dirty flag from the `Firewall` node is properly propagated.
            let _ = self
                .dynamic_repair_query(
                    callee,
                    &CallerInformation::RepairFirewall,
                )
                .await;
        }
    }

    /// Repair the transitive firewall callees first in order for the dirty
    /// propagation to be invoked for the firewall callees.
    async fn repair_transitive_firewall_callees(
        self: &Arc<Self>,
        caller_id: &QueryID,
        lock_computing: &mut ComputingLockGuard<'_, C>,
    ) {
        let Some(tfc_achetype_id) =
            lock_computing.prior_computed().callee_info.tfc_archetype
        else {
            // no transitive firewall callees to repair, skip
            return;
        };

        // Move the prior computed from the lock computing into the computing
        // state itself. This is because when repairing the firewalls, the
        // dirty propagation might need to access the computed state of the
        // affected queries in order to propagate the dirty flags.
        {
            let prior_computed = lock_computing
                .take_prior_computed()
                .expect("should've been set");
            let mut computing_mut = self.database.get_computing_mut(caller_id);

            // should've not set
            assert!(
                computing_mut
                    .computed_placeholder_for_firewall_dirty_propagation
                    .is_none()
            );

            computing_mut.computed_placeholder_for_firewall_dirty_propagation =
                Some(prior_computed);
        }

        let tfc_set = self.database.get_tfc_set_by_id(&tfc_achetype_id);
        self.repair_transitive_firewall_callees_tfc_set(&tfc_set).await;

        // Finish repairing the transitive firewall callees, move back the
        // computed placeholder into the lock computing.
        {
            let mut computing = self.database.get_computing_mut(caller_id);
            let placeholder_computed = computing
                .computed_placeholder_for_firewall_dirty_propagation
                .take()
                .expect("should've been set");

            lock_computing.set_prior_computed(placeholder_computed);
        }
    }

    async fn should_recompute_query<'x, Q: Query>(
        self: &Arc<Self>,
        query_id: &QueryWithID<'_, Q>,
        caller_information: &CallerInformation,
        mut lock_computing: ComputingLockGuard<'x, C>,
    ) -> Option<ComputingLockGuard<'x, C>> {
        // if the caller is backward projection propagation, we always
        // recompute since the projection query have already told us
        // that the value is required to be recomputed.
        if *caller_information
            == CallerInformation::BackwardProjectionPropagation
        {
            return Some(lock_computing);
        }

        // continue normal path ...

        // repair transitive firewall callees first before deciding whether to
        // recompute, since the transitive firewall callees might affect the
        // decision by propagating dirtiness.
        self.repair_transitive_firewall_callees(
            &query_id.id,
            &mut lock_computing,
        )
        .await;

        let computed_callee = lock_computing.prior_computed_mut();

        let recompute = self
            .should_recompute_query_based_on_callee(query_id, computed_callee)
            .await;

        match recompute {
            RepairDecision::Recompute => {
                // continue to recompute
                Some(lock_computing)
            }

            // the query is clean, update the verified timestamp
            RepairDecision::Clean {
                repair_transitive_firewall_callees: false,
            } => {
                computed_callee.verified_at = self.database.current_timestamp();

                self.database
                    .set_computed_from_existing_lock_computing_and_defuse(
                        &query_id.id,
                        lock_computing,
                    );
                None
            }

            // the query is clean, but need to repair transitive firewall
            // callees
            RepairDecision::Clean {
                repair_transitive_firewall_callees: true,
            } => {
                let mut new_tfc_callees = Vec::new();
                for callee in &computed_callee.callee_info.callee_order {
                    // SAFETY: the previous call to `should_recompute_query`
                    // has already ensured that all callees are clean

                    // use spin get computed since some of the callees are not
                    // dirty, so they might be called by a different thread and
                    // turn into computing state.
                    let callee_meta =
                        self.database.spin_get_computed(callee).await;

                    new_tfc_callees.extend(
                        callee_meta.callee_info.tfc_archetype.into_iter(),
                    );
                }

                computed_callee.callee_info.tfc_archetype =
                    self.database.union_tfcs(new_tfc_callees.into_iter());
                computed_callee.verified_at = self.database.current_timestamp();

                self.database
                    .set_computed_from_existing_lock_computing_and_defuse(
                        &query_id.id,
                        lock_computing,
                    );

                None
            }
        }
    }

    #[tracing::instrument(skip(self, lock_computing), level = "info")]
    pub(super) async fn repair_query<Q: Query>(
        self: &Arc<Self>,
        query_id: &QueryWithID<'_, Q>,
        caller_information: &CallerInformation,
        lock_computing: ComputingLockGuard<'_, C>,
    ) {
        // return Some(lock_computing) if should recompute
        let Some(lock_computing) = self
            .should_recompute_query(
                query_id,
                caller_information,
                lock_computing,
            )
            .await
        else {
            return;
        };

        let computed_callee = lock_computing.prior_computed();

        // before recomputing the query, unwire all previous dependencies
        // to avoid dangling dependencies.
        //
        // however, the callers should still be kept since they are still valid.
        // if it has to be unwired, it should be done by the upper chain.
        //
        // we must clear all the current callees first before recomputing since
        // the previous repairing add dependencies that are no longer valid.

        {
            self.database.unwire_backward_dependencies_from_callee(
                &query_id.id,
                &computed_callee.callee_info,
            );

            // clear all callees, that might have been added during repairing
            *self.database.get_computing(&query_id.id).callee_info.write() =
                CalleeInfo::default();
        }

        // now, execute the query again
        self.execute_query(
            query_id,
            lock_computing,
            caller_information,
            ExecuteQueryFor::RecomputeQuery,
        )
        .await;
    }

    // ABOUT CANCELLATIONS:
    // this function is async, meaning that it can be cancelled at any await
    // point. However, cancelling this function means that the dirty propagation
    // might not be fully completed, however, this is not a problem since the
    // currently repairing node itself will be reverted back as not up-to-date,
    // and thus will eventually trigger another repairing that will complete the
    // dirty propagation again.
    async fn on_firewall_or_projection_recompute(
        self: &Arc<Self>,
        fmt: &(dyn std::fmt::Debug + Sync),
        query_id: &QueryID,
        prior_computed: &Computed<C>,
        new_fingerprint: Compact128,
        notify: &Notify,
    ) {
        let current_meta = self.database.get_read_meta(query_id);

        // if the fingerprint is different, we need to propagate dirtiness
        if new_fingerprint == prior_computed.value_fingerprint {
            tracing::info!(
                "{:?} firewall recompute did not change fingerprint {:?}, not \
                 propagating dirtiness",
                fmt,
                new_fingerprint
            );
        } else {
            tracing::info!(
                "{:?} firewall recompute changed fingerprint from {:?} to \
                 {:?}, propagating dirtiness",
                fmt,
                prior_computed.value_fingerprint,
                new_fingerprint
            );
            self.database.propagate_dirtiness_from_firewall(query_id);

            // invoke all backward projections to keep propagation going
            self.invoke_backward_projections(current_meta, notify).await;
        }
    }

    async fn invoke_backward_projections(
        self: &Arc<Self>,
        current_meta: Ref<'_, QueryID, QueryMeta<C>>,
        notify: &Notify,
    ) {
        // OPTIMIZE: this can be parallelized
        let backward_projections = current_meta
            .caller_info
            .iter()
            .filter(|x| {
                let caller_meta = self.database.get_read_meta(x);

                // not a projection, skip
                caller_meta.is_projection()
            })
            .collect::<Vec<_>>();

        // IMPORTANT: release the read lock before invoking dynamic repairs
        // which will eventually require write lock
        drop(current_meta);

        // NOTE: notify the potentially waiting projection that arrives earlier
        // and is waiting for the value to be ready for backward propagation.
        notify.notify_waiters();

        // OPTIMIZE: this can be parallelized
        for query_id in backward_projections {
            // dynamically invoke the projection query to update its value
            // NOTE: backward propagation of projections must not take
            // this query as the caller query since it will create
            // a cyclic dependency immediately.
            let _ = self
                .dynamic_repair_query(
                    &query_id,
                    &CallerInformation::BackwardProjectionPropagation,
                )
                .await;
        }
    }

    pub(super) async fn execute_query<Q: Query>(
        self: &Arc<Self>,
        query: &QueryWithID<'_, Q>,
        lock_computing: ComputingLockGuard<'_, C>,
        original_caller: &CallerInformation,
        execut_query_for: ExecuteQueryFor,
    ) {
        // create a new tracked engine
        let cache = Arc::new(DashMap::default());

        let reason = if *original_caller
            == CallerInformation::BackwardProjectionPropagation
        {
            CallerReason::ProjectionRecomputingDueToBackwardPropagation
        } else {
            CallerReason::RequireValue
        };

        let mut tracked_engine = TrackedEngine {
            engine: self.clone(),
            cache: cache.clone(),
            caller: CallerInformation::Query(QueryCaller {
                query_id: query.id,
                reason,
            }),
        };

        let entry = self.executor_registry.get_executor_entry::<Q>();

        let result =
            entry.invoke_executor(query.query, &mut tracked_engine).await;

        // use the `cache`'s strong count to determine if the tracked
        // engine is still held elsewhere other than the
        // current call stack.
        //
        // if there're still references to the `TrackedEngine`, it means
        // that there's some dangling references to the
        // `TrackedEngine` on some other threads that
        // the implementation of the query is not aware of.
        //
        // in this case, we'll panic to avoid silent bugs in the query
        // implementation.
        assert!(
            // 2 one for aliving tracked engine, and one for cache
            Arc::strong_count(&tracked_engine.cache) == 2,
            "`TrackedEngine` is still held elsewhere, this is a bug in the \
             query implementation which violates the query system's contract. \
             It's possible that the `TrackedEngine` is being sent to a \
             different thread and the query implementation hasn't properly \
             joined the thread before returning the value. Key: `{}`",
            std::any::type_name::<Q>()
        );

        let is_in_scc = self
            .database
            .get_computing(&query.id)
            .is_in_scc
            .load(std::sync::atomic::Ordering::Relaxed);

        // if `is_in_scc` is `true`, it means that the query is part of
        // a strongly connected component (SCC) and the
        // value should be an error, otherwise, it
        // should be a valid value.

        assert_eq!(
            is_in_scc,
            result.is_err(),
            "Cyclic dependency state mismatch: expected {}, got {}",
            result.is_err(),
            is_in_scc
        );

        let value = result.unwrap_or_else(|_| {
            let smallbox: DynValueBox<C> =
                smallbox::smallbox!(entry.obtain_scc_value::<Q>());

            smallbox
        });

        let new_fingerprint = Compact128::from_u128(
            value.hash_128_value(self.database.initial_seed()),
        );

        // mark the query as ready to change to computed
        let ready_computed = {
            let mut computing_mut = self.database.get_computing_mut(&query.id);

            let computing_mut: &mut Computing<_> = &mut *computing_mut;

            ReadyComputed {
                value_fingerprint: new_fingerprint,
                tfc_archetype: computing_mut.callee_info.read().tfc_archetype,
                value,
                next_version: self.database.current_timestamp(),
            }
        };

        self.handle_ready_computed_after_execute_query(
            query,
            lock_computing,
            ready_computed,
            execut_query_for,
        )
        .await;
    }

    async fn handle_ready_computed_after_execute_query(
        self: &Arc<Self>,
        query: &QueryWithID<'_, impl Query>,
        mut lock_computing: ComputingLockGuard<'_, C>,
        ready_computed: ReadyComputed<C>,
        execut_query_for: ExecuteQueryFor,
    ) {
        match execut_query_for {
            // recompute case, make sure that if it's a firewall or projection,
            // propagate dirtiness if needed.
            //
            // as well as drop the prior computed placeholder.
            ExecuteQueryFor::RecomputeQuery => {
                let mut current_meta_mut =
                    self.database.get_meta_mut(&query.id);

                // if isn't firewall or projection, no-op
                if matches!(
                    current_meta_mut.query_kind,
                    QueryKind::Execute(
                        ExecutionStyle::Firewall | ExecutionStyle::Projection
                    )
                ) {
                    let computed_callee = lock_computing.prior_computed();
                    let new_fingerprint = ready_computed.value_fingerprint;
                    current_meta_mut.get_computing_mut().state =
                        ComputingState::ForBackwardPropagation(ready_computed);

                    drop(current_meta_mut);

                    // after recomputation, we need to check if the firewall or
                    // projection fingerprint has changed to
                    // propagate dirtiness
                    self.on_firewall_or_projection_recompute(
                        &query.query,
                        &query.id,
                        computed_callee,
                        new_fingerprint,
                        lock_computing.notification(),
                    )
                    .await;

                    // remove the prior computed placeholder to prevent it
                    // restoring the old computed state.
                    assert!(lock_computing.take_prior_computed().is_some());

                    // set the computing state from the Computing::State
                    self.database.done_compute_from_backward_propagation(
                        &query.id,
                        lock_computing,
                    );
                } else {
                    drop(current_meta_mut);

                    // remove the prior computed placeholder to prevent it
                    // restoring the old computed state.
                    assert!(lock_computing.take_prior_computed().is_some());

                    // directly set to computed state
                    self.database.done_compute(
                        &query.id,
                        ready_computed,
                        lock_computing,
                    );
                }
            }

            ExecuteQueryFor::FreshQuery => {
                self.database.done_compute(
                    &query.id,
                    ready_computed,
                    lock_computing,
                );
            }
        }
    }

    pub(super) async fn continuation<Q: Query>(
        self: &Arc<Self>,
        query: &QueryWithID<'_, Q>,
        caller_information: &CallerInformation,
        lock_computing: ComputingLockGuard<'_, C>,
    ) {
        if lock_computing.has_prior_computed_placeholder() {
            self.repair_query(query, caller_information, lock_computing).await;
        } else {
            self.execute_query(
                query,
                lock_computing,
                caller_information,
                ExecuteQueryFor::FreshQuery,
            )
            .await;
        }
    }
}
