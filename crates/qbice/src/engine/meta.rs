use std::{
    any::Any,
    collections::{HashMap, HashSet, VecDeque, hash_map::Entry},
    ops::Not,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use dashmap::{DashMap, DashSet};
use enum_as_inner::EnumAsInner;
use parking_lot::{RwLock, RwLockReadGuard};
use tokio::sync::Notify;

use crate::{
    config::Config,
    engine::{
        Engine, TrackedEngine,
        database::{Caller, ComputingLockGuard, Database, Timtestamp},
    },
    executor::CyclicError,
    query::{
        DynQuery, DynQueryBox, DynValue, DynValueBox, Query, QueryID, QueryKind,
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Compact128(u64, u64);

impl Compact128 {
    pub const fn from_u128(n: u128) -> Self {
        let hi: u64 = (n >> 64) as u64; // upper 64 bits
        let lo: u64 = (n & 0xFFFF_FFFF_FFFF_FFFF) as u64; // lower 64 bits

        Self(hi, lo)
    }
}

#[derive(Debug)]
pub struct Computing {
    notification: Arc<Notify>,

    // We use a RwLock here since we want to avoid acquiring a write lock
    // when observing callees.
    callee_info: RwLock<CalleeInfo>,

    is_in_scc: AtomicBool,
}

impl Computing {
    pub fn callee_info(&self) -> RwLockReadGuard<'_, CalleeInfo> {
        self.callee_info.read()
    }

    pub fn new(notification: Arc<Notify>) -> Self {
        Self {
            notification,
            callee_info: RwLock::new(CalleeInfo {
                callee_queries: HashMap::new(),
                callee_order: Vec::new(),
                transitive_firewall_callees: HashSet::new(),
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

    fingerprint: Compact128,
}

impl<C: Config> Computed<C> {
    pub const fn verified_at(&self) -> Timtestamp { self.verified_at }
}

#[derive(Debug, EnumAsInner)]
pub enum State<C: Config> {
    Computing(Computing),
    Computed(Computed<C>),
}

#[derive(Debug)]
pub struct Observation {
    seen_fingerprint: Compact128,

    // Dirty flag is used as an atomic because we want to avoid acquiring
    // write lock when marking as dirty when propagating dirtiness.
    dirty: AtomicBool,
}

#[derive(Debug, Default)]
pub struct CalleeInfo {
    callee_queries: HashMap<QueryID, Option<Observation>>,
    callee_order: Vec<QueryID>,
    transitive_firewall_callees: HashSet<QueryID>,
}

#[derive(Debug, Default)]
pub struct CallerInfo {
    caller_queries: DashSet<QueryID>,
}

pub struct QueryMeta<C: Config> {
    original_key: DynQueryBox<C>,
    caller_info: CallerInfo,

    is_input: bool,
    query_kind: QueryKind,
    state: State<C>,
}

impl<C: Config> QueryMeta<C> {
    pub fn new<Q>(original_key: Q, state: State<C>) -> Self
    where
        Q: Query,
    {
        Self {
            original_key: smallbox::smallbox!(original_key),
            caller_info: CallerInfo::default(),
            is_input: false,
            query_kind: Q::query_kind(),
            state,
        }
    }

    pub fn new_input<Q: Query>(
        query_key: Q,
        query_value: Q::Value,
        timestamp: &Timtestamp,
        hash_seed: u64,
    ) -> Self {
        let hash_128 = DynValue::<C>::hash_128_value(&query_value, hash_seed);

        Self {
            original_key: smallbox::smallbox!(query_key),
            caller_info: CallerInfo::default(),
            is_input: true,
            query_kind: Q::query_kind(),
            state: State::Computed(Computed {
                result: smallbox::smallbox!(query_value),
                callee_info: CalleeInfo::default(),
                verified_at: *timestamp,
                fingerprint: Compact128::from_u128(hash_128),
            }),
        }
    }

    pub fn set_input<Q: Query>(
        &mut self,
        query_value: Q::Value,
        has_already_incremented_timestamp: bool,
        hash_seed: u64,
        timestamp: &mut Timtestamp,
    ) -> SetInputResult {
        let mut this_has_incremented = false;

        self.is_input = true;

        let computed =
            self.state.as_computed_mut().expect("should've been computed");

        let hash = Compact128::from_u128(DynValue::<C>::hash_128_value(
            &query_value,
            hash_seed,
        ));

        let fingerprint_diff = hash != computed.fingerprint;

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
        computed.fingerprint = hash;

        SetInputResult { incremented: this_has_incremented, fingerprint_diff }
    }

    #[allow(unused)]
    pub fn get_computing(&self) -> &Computing {
        self.state.as_computing().expect("should be computing")
    }

    #[allow(unused)]
    pub fn get_computing_mut(&mut self) -> &mut Computing {
        self.state.as_computing_mut().expect("should be computing")
    }

    pub fn get_computed_mut(&mut self) -> &mut Computed<C> {
        self.state.as_computed_mut().expect("should be computed")
    }

    pub fn get_computed(&self) -> &Computed<C> {
        self.state.as_computed().expect("should be computed")
    }

    pub const fn state(&self) -> &State<C> { &self.state }

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

/// The result of attempting a fast-path query execution.
pub enum FastPathResult<V> {
    TryAgain,
    ToSlowPath,
    Hit(Option<V>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SetInputResult {
    pub incremented: bool,
    pub fingerprint_diff: bool,
}

impl<C: Config> Database<C> {
    /// Add both forward and backward dependencies for both caller and callee.
    fn observe_callee_fingerprint(
        &self,
        callee_meta: &QueryMeta<C>,
        computed_callee: &Computed<C>,
        callee_target: &QueryID,
        caller_source: &Caller,
        callee_kind: QueryKind,
    ) {
        // add dependency for the caller
        let caller_computing = self.get_computing_caller(caller_source);
        let mut caller_callee_info = caller_computing.callee_info.write();
        {
            assert!(
                caller_callee_info
                    .callee_queries
                    .insert(
                        *callee_target,
                        Some(Observation {
                            seen_fingerprint: computed_callee.fingerprint,
                            dirty: AtomicBool::new(false),
                        }),
                    )
                    .is_some(),
                "should've been pre-inserted in `query_for`"
            );

            match callee_kind {
                QueryKind::Normal | QueryKind::Projection => {
                    for dep in
                        &computed_callee.callee_info.transitive_firewall_callees
                    {
                        caller_callee_info
                            .transitive_firewall_callees
                            .insert(*dep);
                    }
                }
                QueryKind::Firewall => {
                    caller_callee_info
                        .transitive_firewall_callees
                        .insert(*callee_target);
                }
            }
        }

        // add dependency for the callee
        callee_meta
            .caller_info
            .caller_queries
            .insert(*caller_source.query_id());
    }

    pub(super) fn unwire_backward_dependencies_from_callee(
        &self,
        query_id: &QueryID,
        callee_info: &CalleeInfo,
    ) {
        // OPTIMIZE: this can be parallelized
        for callee in &callee_info.callee_order {
            self.unwire_backward_dependency(query_id, callee);
        }
    }

    /// Checks whether the stack of computing queries contains a cycle
    fn check_cyclic(&self, computing: &Computing, target: QueryID) -> bool {
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
        called_from: Option<&Caller>,
        running_state: &Computing,
    ) -> Result<(), CyclicError> {
        // if there is no caller, we are at the root.
        let Some(called_from) = called_from else {
            return Ok(());
        };

        let is_in_scc =
            self.check_cyclic(running_state, *called_from.query_id());

        // mark the caller as being in scc
        if is_in_scc {
            let meta = self.get_computing_caller(called_from);

            meta.is_in_scc.store(true, std::sync::atomic::Ordering::SeqCst);

            return Err(CyclicError);
        }

        Ok(())
    }

    /// Attempt to fast-path the query execution. Should acquire only a read
    /// lock, no write lock involved.
    pub(super) async fn fast_path<V: 'static + Send + Sync + Clone>(
        &self,
        query_id: &QueryID,
        required_value: bool,
        caller: Option<&Caller>,
    ) -> Result<FastPathResult<V>, CyclicError> {
        // checks if the query result is already computed
        let Some(callee_target_meta) = self.try_get_read_meta(query_id) else {
            return Ok(FastPathResult::ToSlowPath);
        };

        match &callee_target_meta.state {
            State::Computing(computing) => {
                self.exit_scc(caller, computing)?;

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

                if let Some(caller) = caller
                    && required_value
                {
                    // register dependency
                    self.observe_callee_fingerprint(
                        &callee_target_meta,
                        computed,
                        query_id,
                        caller,
                        callee_target_meta.query_kind,
                    );
                }

                Ok(FastPathResult::Hit(required_value.then(|| {
                    computed.result.downcast_value::<V>().unwrap().clone()
                })))
            }
        }
    }

    // Done computing the value, defuse the computing lock and set the computed
    // state
    fn done_compute(
        &self,
        calling_query: &Caller,
        value: DynValueBox<C>,
        lock_computing: ComputingLockGuard<'_, C>,
    ) {
        let hash_128 = value.hash_128_value(self.initial_seed());

        self.set_computed_from_computing_and_defuse(
            calling_query,
            |computing| Computed {
                result: value,
                callee_info: computing.callee_info.into_inner(),
                verified_at: self.current_timestamp(),
                fingerprint: Compact128::from_u128(hash_128),
            },
            lock_computing,
        );
    }

    pub(super) fn dirty_queries(&mut self, mut queries: VecDeque<QueryID>) {
        // OPTIMIZE: we could potentially have worker threads to process dirty
        // queries
        let mut dirtied = HashSet::<QueryID>::default();

        while let Some(query_id) = queries.pop_front() {
            let meta = self.get_read_meta(&query_id);

            // iterate through all callers and mark their
            // observations as dirty
            for caller in meta.caller_info.caller_queries.iter() {
                let caller_meta = self.get_computed(caller.key());

                let obs = caller_meta
                    .callee_info
                    .callee_queries
                    .get(&query_id)
                    .expect("should be present");

                let obs = obs.as_ref().expect("should be present");

                obs.dirty.store(true, Ordering::SeqCst);

                // if hasn't already dirty, add to dirty batch
                if dirtied.insert(*caller.key()) {
                    queries.push_back(*caller.key());
                }
            }
        }
    }

    pub(crate) fn unwire_backward_dependency(
        &self,
        caller_source: &QueryID,
        callee_target: &QueryID,
    ) {
        let callee_meta = self.get_read_meta(callee_target);
        callee_meta.caller_info.caller_queries.remove(caller_source);
    }

    fn is_query_input(&self, query_id: &QueryID) -> bool {
        let meta = self.get_read_meta(query_id);
        meta.is_input
    }
}

impl<C: Config> Engine<C> {
    #[tracing::instrument(skip(self, computed_callee), ret, level = "info")]
    pub(super) async fn should_recompute_query<Q: Query>(
        self: &Arc<Self>,
        query_id: &QueryWithID<'_, Q>,
        computed_callee: &mut Computed<C>,
    ) -> bool {
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
                // NOTE: we clone the original key here since we cannot hold the
                // read lock on the query metas while the repair is happening.
                let original_callee_value = self
                    .database
                    .get_read_meta(callee)
                    .original_key
                    .dyn_clone();

                // NOTE: we must repair the callee through executor registry
                // since it's impossible to statically know the
                // type of the query here.
                let entry = self
                    .executor_registry
                    .get_executor_entry_by_type_id(&callee.stable_type_id());

                let callee_value: &dyn DynQuery<C> = &*original_callee_value;
                let callee_value_any = callee_value as &(dyn Any + Send + Sync);

                tracing::info!(
                    "{:?} is repairing {:?}",
                    query_id.query,
                    callee_value
                );
                let _ = entry
                    .recursively_repair_query(
                        self,
                        callee_value_any,
                        &query_id.id,
                    )
                    .await;
            }

            // SAFETY: after the repair, the callee's state should be at
            // computed at the current timestamp.
            let callee_meta = self.database.get_read_meta(callee);

            let callee_computed =
                callee_meta.state.as_computed().expect("should be computed");

            // depend on whether the observed fingerprint has changed
            {
                let obs = computed_callee
                    .callee_info
                    .callee_queries
                    .get_mut(callee)
                    .expect("should be present");

                let obs = obs.as_mut().expect("should be present");

                let new_fingerprint = callee_computed.fingerprint;

                // fingerprint is the same, mark as clean
                if obs.seen_fingerprint == new_fingerprint {
                    tracing::info!(
                        "{:?} observed {:?} is clean",
                        query_id.query,
                        callee_meta.original_key
                    );
                    obs.dirty = false.into();
                } else {
                    // fingerprint changed, need to recompute this node, stop
                    // repairing further callees
                    return true;
                }
            }
        }

        // all callees are clean, no need to recompute
        false
    }

    #[tracing::instrument(skip(self, lock_computing), level = "info")]
    pub(super) async fn repair_query<Q: Query>(
        self: &Arc<Self>,
        query_id: &QueryWithID<'_, Q>,
        mut lock_computing: ComputingLockGuard<'_, C>,
    ) {
        let caller_id = Caller::new(query_id.id);

        // has already been verified at the current timestamp
        let computed_callee = lock_computing
            .get_prior_computed_mut()
            .expect("should've had prior computed");

        if computed_callee.verified_at == self.database.current_timestamp() {
            return;
        }

        let recompute =
            self.should_recompute_query(query_id, computed_callee).await;

        // the query is clean, update the verified timestamp
        if !recompute {
            computed_callee.verified_at = self.database.current_timestamp();
            self.database.set_computed_from_existing_lock_computing_and_defuse(
                &caller_id,
                lock_computing,
            );
            return;
        }

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
            *self
                .database
                .get_computing_caller(&caller_id)
                .callee_info
                .write() = CalleeInfo::default();
        }

        // take the lock_computing's computed state in case execute_query
        // cancels as we have already unwired the dependencies.
        lock_computing.take_prior_computed();

        // now, execute the query again
        self.execute_query(query_id, lock_computing).await;
    }

    pub(super) async fn execute_query<Q: Query>(
        self: &Arc<Self>,
        query: &QueryWithID<'_, Q>,
        lock_computing: ComputingLockGuard<'_, C>,
    ) {
        // create a new tracked engine
        let cache = Arc::new(DashMap::default());
        let caller_query = Caller::new(query.id);

        let mut tracked_engine = TrackedEngine {
            engine: self.clone(),
            cache: cache.clone(),
            caller: Some(caller_query),
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
            .get_computing_caller(&caller_query)
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
            let smallbox: DynValueBox<C> = smallbox::smallbox!(Q::scc_value());

            smallbox
        });

        self.database.done_compute(&caller_query, value, lock_computing);
    }

    pub(super) async fn continuation<Q: Query>(
        self: &Arc<Self>,
        query: &QueryWithID<'_, Q>,
        lock_computing: ComputingLockGuard<'_, C>,
    ) {
        if lock_computing.has_prior_computed() {
            self.repair_query(query, lock_computing).await;
        } else {
            self.execute_query(query, lock_computing).await;
        }
    }
}
