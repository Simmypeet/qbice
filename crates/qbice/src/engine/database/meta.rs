use std::{
    any::Any,
    collections::{HashSet, VecDeque},
    ops::Not,
    pin::Pin,
    sync::{Arc, atomic::AtomicBool},
};

use dashmap::{DashMap, DashSet};
use enum_as_inner::EnumAsInner;
use tokio::sync::{Notify, RwLock};

use crate::{
    config::Config,
    engine::{
        Engine, TrackedEngine,
        database::{Database, Timtestamp},
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
    callee_info: CalleeInfo,
    is_in_scc: AtomicBool,
}

impl Computing {
    pub fn new(notification: Arc<Notify>) -> Self {
        Self {
            notification,
            callee_info: CalleeInfo {
                callee_queries: DashMap::new(),
                callee_order: RwLock::new(Vec::new()),
                transitive_firewall_callees: DashSet::new(),
            },
            is_in_scc: AtomicBool::new(false),
        }
    }
}

#[derive(Debug)]
pub struct Computed<C: Config> {
    result: DynValueBox<C>,
    caller_info: CallerInfo,
    callee_info: CalleeInfo,
    verified_at: Timtestamp,
    is_input: bool,

    fingerprint: Compact128,
}

#[derive(Debug, EnumAsInner)]
pub enum State<C: Config> {
    Computing(Computing),
    Computed(Computed<C>),
}

#[derive(Debug)]
pub struct Observation {
    pub seen_fingerprint: Compact128,
    pub dirty: bool,
}

#[derive(Debug, Default)]
pub struct CalleeInfo {
    callee_queries: DashMap<QueryID, Option<Observation>>,
    callee_order: RwLock<Vec<QueryID>>,
    transitive_firewall_callees: DashSet<QueryID>,
}

#[derive(Debug, Default)]
pub struct CallerInfo {
    caller_queries: DashSet<QueryID>,
}

pub struct QueryMeta<C: Config> {
    original_key: DynQueryBox<C>,

    query_kind: QueryKind,
    state: State<C>,
}

impl<C: Config> QueryMeta<C> {
    pub fn add_callee(&self, callee: QueryID) {
        let computing = self
            .state
            .as_computing()
            .expect("can only add dependencies while computing");

        match computing.callee_info.callee_queries.entry(callee) {
            dashmap::mapref::entry::Entry::Occupied(_) => {}
            dashmap::mapref::entry::Entry::Vacant(v) => {
                v.insert(None);
            }
        }
    }

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
}

impl<C: Config> Database<C> {
    /// Create a new query with its associated unique identifier.
    pub(super) fn new_query_with_id<'c, Q: Query>(
        &'c self,
        query: &'c Q,
    ) -> QueryWithID<'c, Q> {
        QueryWithID {
            id: DynQuery::<C>::query_identifier(query, self.initial_seed),
            query,
        }
    }

    pub(super) fn query_id<Q: Query>(&self, query: &Q) -> QueryID {
        DynQuery::<C>::query_identifier(query, self.initial_seed)
    }
}

pub enum FastPathResult<V> {
    TryAgain,
    ToSlowPath,
    Hit(Option<V>),
}

#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum Continuation<C: Config> {
    Fresh,
    Reverify(Computed<C>),
}

#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum SlowPathResult<C: Config> {
    TryAgain,
    Continuation(Arc<Notify>, Continuation<C>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SetInputResult {
    pub incremented: bool,
    pub fingerprint_diff: bool,
}

impl<C: Config> Database<C> {
    /// Add both forward and backward dependencies for both caller and callee.
    async fn observe_callee_fingerprint(
        caller_meta: &QueryMeta<C>,
        computed_callee: &Computed<C>,
        callee_target: &QueryID,
        caller_source: &QueryID,
        callee_kind: QueryKind,
    ) {
        // add dependency for the caller
        let caller_computing = caller_meta
            .state
            .as_computing()
            .expect("caller should've been computing");

        {
            let inserted = caller_computing
                .callee_info
                .callee_queries
                .insert(
                    *callee_target,
                    Some(Observation {
                        seen_fingerprint: computed_callee.fingerprint,
                        dirty: false,
                    }),
                )
                .is_some();

            // if haven't inserted, add to dependency order
            if inserted.not() {
                caller_computing
                    .callee_info
                    .callee_order
                    .write()
                    .await
                    .push(*callee_target);
            }

            match callee_kind {
                QueryKind::Normal | QueryKind::Projection => {
                    for dep in computed_callee
                        .callee_info
                        .transitive_firewall_callees
                        .iter()
                    {
                        caller_computing
                            .callee_info
                            .transitive_firewall_callees
                            .insert(*dep);
                    }
                }
                QueryKind::Firewall => {
                    caller_computing
                        .callee_info
                        .transitive_firewall_callees
                        .insert(*callee_target);
                }
            }
        }

        // add dependency for the callee
        computed_callee.caller_info.caller_queries.insert(*caller_source);
    }

    /// Checks whether the stack of computing queries contains a cycle
    fn check_cyclic(&self, computing: &Computing, target: QueryID) -> bool {
        if computing.callee_info.callee_queries.contains_key(&target) {
            computing
                .is_in_scc
                .store(true, std::sync::atomic::Ordering::Relaxed);

            return true;
        }

        let mut found = false;

        for dep in &computing.callee_info.callee_queries {
            let Some(state) = self.query_metas.get(dep.key()) else {
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
        running_state: &Computing,
    ) -> Result<(), CyclicError> {
        // if there is no caller, we are at the root.
        let Some(called_from) = called_from else {
            return Ok(());
        };

        let is_in_scc = self.check_cyclic(running_state, *called_from);

        // mark the caller as being in scc
        if is_in_scc {
            let meta = self
                .query_metas
                .get(called_from)
                .expect("called_from query state must exist");

            let called_from_state =
                meta.state.as_computing().expect("should be computing");

            called_from_state
                .is_in_scc
                .store(true, std::sync::atomic::Ordering::SeqCst);

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
        caller: Option<&QueryID>,
    ) -> Result<FastPathResult<V>, CyclicError> {
        // checks if the query result is already computed
        let Some(callee_target_meta) = self.query_metas.get(query_id) else {
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
                if computed.verified_at != self.current_timestamp {
                    return Ok(FastPathResult::ToSlowPath);
                }

                if let Some(caller) = caller
                    && required_value
                {
                    // register dependency
                    let caller_source_meta =
                        self.query_metas.get(caller).unwrap();

                    tracing::info!(
                        "{:?} observes {:?}",
                        caller_source_meta.original_key,
                        callee_target_meta.original_key
                    );

                    Self::observe_callee_fingerprint(
                        &caller_source_meta,
                        computed,
                        query_id,
                        caller,
                        callee_target_meta.query_kind,
                    )
                    .await;
                }

                Ok(FastPathResult::Hit(required_value.then(|| {
                    computed.result.downcast_value::<V>().unwrap().clone()
                })))
            }
        }
    }

    pub(super) fn slow_path<Q: Query>(
        &self,
        query_id: &QueryWithID<'_, Q>,
    ) -> SlowPathResult<C> {
        // obtain a write lock on the query meta
        match self.query_metas.entry(query_id.id) {
            dashmap::Entry::Occupied(mut occupied_entry) => {
                match &occupied_entry.get().state {
                    State::Computing(_) => {
                        // another thread is computing the query, try again
                        SlowPathResult::TryAgain
                    }

                    State::Computed(computed) => {
                        if computed.verified_at == self.current_timestamp {
                            // already computed and verified, try fast path
                            // and retrieve value again
                            return SlowPathResult::TryAgain;
                        }

                        let noti = Arc::new(Notify::new());

                        let computed = std::mem::replace(
                            &mut occupied_entry.get_mut().state,
                            State::Computing(Computing::new(noti.clone())),
                        )
                        .into_computed()
                        .expect("should've been computed");

                        SlowPathResult::Continuation(
                            noti,
                            Continuation::Reverify(computed),
                        )
                    }
                }
            }

            dashmap::Entry::Vacant(vacant_entry) => {
                let noti = Arc::new(Notify::new());
                let fresh_query_meta = QueryMeta {
                    original_key: smallbox::smallbox!(query_id.query.clone()),
                    query_kind: Q::query_kind(),
                    state: State::Computing(Computing::new(noti.clone())),
                };

                vacant_entry.insert(fresh_query_meta);

                SlowPathResult::Continuation(noti, Continuation::Fresh)
            }
        }
    }

    fn set_computed(
        &self,
        query_id: &QueryID,
        computed_callee: Computed<C>,
        notify: &Notify,
    ) {
        let mut meta = self
            .query_metas
            .get_mut(query_id)
            .expect("should've been a computing query");

        assert!(meta.state.is_computing(), "should be computing");

        meta.state = State::Computed(computed_callee);

        notify.notify_waiters();
    }

    fn done_compute(
        &self,
        query_id: &QueryID,
        value: DynValueBox<C>,
        caller_info: Option<CallerInfo>,
        noti: &Notify,
    ) {
        let mut meta = self
            .query_metas
            .get_mut(query_id)
            .expect("should've been a computing query");

        take_mut::take(&mut *meta, |mut meta| {
            let computing = meta.state.into_computing().unwrap();
            let hash_128 = value.hash_128_value(self.initial_seed);

            meta.state = State::Computed(Computed {
                result: value,
                caller_info: caller_info.unwrap_or_default(),
                callee_info: computing.callee_info,
                verified_at: self.current_timestamp,
                fingerprint: Compact128::from_u128(hash_128),
                is_input: false,
            });

            meta
        });

        noti.notify_waiters();
    }

    pub(super) fn set_input<Q: Query>(
        &mut self,
        query_key: Q,
        query_id: QueryID,
        value: Q::Value,
        incremented: bool,
    ) -> SetInputResult {
        match self.query_metas.entry(query_id) {
            dashmap::Entry::Occupied(mut occupied_entry) => {
                let mut has_incremented = false;
                let meta = occupied_entry.get_mut();

                match &mut meta.state {
                    State::Computing(_) => unreachable!(
                        "shouldn't exist since we've obtained exclusive \
                         mutable reference"
                    ),

                    State::Computed(computed) => {
                        let hash = Compact128::from_u128(
                            DynValue::<C>::hash_128_value(
                                &value,
                                self.initial_seed,
                            ),
                        );

                        let fingerprint_diff = hash != computed.fingerprint;

                        // if haven't incremented and the hash is different,
                        // update the timestamp
                        if !incremented && fingerprint_diff {
                            self.current_timestamp.increment();
                            computed.verified_at = self.current_timestamp;

                            has_incremented = true;
                        }

                        // delete all callees
                        computed.callee_info = CalleeInfo::default();
                        computed.result = smallbox::smallbox!(value);
                        computed.fingerprint = hash;

                        SetInputResult {
                            incremented: has_incremented,
                            fingerprint_diff,
                        }
                    }
                }
            }

            // new vaccant input
            dashmap::Entry::Vacant(vacant_entry) => {
                vacant_entry.insert(QueryMeta {
                    original_key: smallbox::smallbox!(query_key),
                    query_kind: Q::query_kind(),
                    state: State::Computed(Computed {
                        caller_info: CallerInfo::default(),
                        callee_info: CalleeInfo::default(),
                        verified_at: self.current_timestamp,
                        is_input: true,
                        fingerprint: Compact128::from_u128(
                            DynValue::<C>::hash_128_value(
                                &value,
                                self.initial_seed,
                            ),
                        ),
                        result: smallbox::smallbox!(value),
                    }),
                });

                SetInputResult { incremented, fingerprint_diff: false }
            }
        }
    }

    pub(super) fn dirty_queries(&mut self, mut queries: VecDeque<QueryID>) {
        // OPTIMIZE: we could potentially have worker threads to process dirty
        // queries
        let mut dirtied = HashSet::<QueryID>::default();

        while let Some(query_id) = queries.pop_front() {
            if let Some(mut meta) = self.query_metas.get_mut(&query_id) {
                tracing::info!("Dirtying query {:?}", meta.original_key);

                match &mut meta.state {
                    State::Computing(_) => unreachable!(
                        "shouldn't exist since we've obtained exclusive \
                         mutable reference"
                    ),

                    State::Computed(computed) => {
                        // iterate through all callers and mark their
                        // observations as dirty
                        for caller in computed.caller_info.caller_queries.iter()
                        {
                            let mut caller_meta = self
                                .query_metas
                                .get_mut(caller.key())
                                .expect("should be present");

                            let mut obs = caller_meta
                                .state
                                .as_computed_mut()
                                .expect("should've been computed")
                                .callee_info
                                .callee_queries
                                .get_mut(&query_id)
                                .expect("should be present");

                            let obs = obs.as_mut().expect("should be present");

                            obs.dirty = true;

                            // if hasn't already dirty, add to dirty batch
                            if dirtied.insert(*caller.key()) {
                                queries.push_back(*caller.key());
                            }
                        }
                    }
                }
            }
        }
    }

    pub(crate) async fn unwire_backward_dependencies(
        &self,
        caller_source: &QueryID,
        callee_target: &QueryID,
    ) {
        loop {
            let callee_meta =
                self.query_metas.get(callee_target).expect("should've existed");

            match &callee_meta.state {
                State::Computing(computing) => {
                    // still computing, wait and try again
                    let noti = computing.notification.clone();
                    let notified = noti.notified();
                    drop(callee_meta);

                    notified.await;
                }

                State::Computed(computed) => {
                    assert!(
                        computed
                            .caller_info
                            .caller_queries
                            .remove(caller_source)
                            .is_some()
                    );

                    break;
                }
            }
        }
    }

    async fn is_query_input(&self, query_id: &QueryID) -> bool {
        self.try_get_computed(query_id, |computed| computed.is_input).await
    }

    async fn try_get_computed<T>(
        &self,
        query_id: &QueryID,
        f: impl FnOnce(&Computed<C>) -> T,
    ) -> T {
        loop {
            let meta =
                self.query_metas.get(query_id).expect("should be present");

            match &meta.state {
                State::Computing(computing) => {
                    // still computing, wait and try again
                    let noti = computing.notification.clone();
                    let notified = noti.notified();
                    drop(meta);

                    notified.await;
                }

                State::Computed(computed) => {
                    return f(computed);
                }
            }
        }
    }
}

impl<C: Config> Engine<C> {
    #[tracing::instrument(
        skip(self, query_id, computed_callee),
        level = "info"
    )]
    pub(super) async fn should_recompute_query<Q: Query>(
        self: &Arc<Self>,
        query_id: &QueryWithID<'_, Q>,
        computed_callee: &mut Computed<C>,
    ) -> bool {
        let callee_orders =
            computed_callee.callee_info.callee_order.read().await;

        // search for queries that are dirty
        for callee in callee_orders.iter() {
            // skip if not dirty
            {
                let obs = computed_callee
                    .callee_info
                    .callee_queries
                    .get(callee)
                    .expect("should be present");

                let obs =
                    obs.as_ref().expect("should've been set in the fast path");

                if obs.dirty.not() {
                    continue;
                }
            }

            // NOTE: if the callee is an input (explicitly set), it's impossible
            // to try to repair it, so we'll skip repairing and directly
            // compare the fingerprint.
            if !self.database.is_query_input(callee).await {
                // recursively repair the callee.
                // NOTE: we clone the original key here since we cannot hold the
                // read lock on the query metas while the repair is happening.
                let original_callee_value = self
                    .database
                    .query_metas
                    .get(callee)
                    .expect("should be present")
                    .original_key
                    .dyn_clone();

                // NOTE: we must repair the callee through executor registry
                // since it's impossible to statically know the
                // type of the query here.
                let entry = self
                    .executor_registry
                    .get_executor_entry_by_type_id(&callee.stable_type_id());

                let callee_value: &dyn DynQuery<C> = &*original_callee_value;
                let callee_value_any = callee_value as &dyn Any;

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
            let callee_meta = self
                .database
                .query_metas
                .get(callee)
                .expect("should be present");

            let callee_computed =
                callee_meta.state.as_computed().expect("should be computed");

            // depend on whether the observed fingerprint has changed
            {
                let mut obs = computed_callee
                    .callee_info
                    .callee_queries
                    .get_mut(callee)
                    .expect("should be present");

                let obs = obs.as_mut().expect("should be present");

                let new_fingerprint = callee_computed.fingerprint;

                // fingerprint is the same, mark as clean
                if obs.seen_fingerprint == new_fingerprint {
                    obs.dirty = false;
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

    pub(super) async fn repair_query<Q: Query>(
        self: &Arc<Self>,
        query_id: &QueryWithID<'_, Q>,
        mut computed_callee: Computed<C>,
        notify: &Notify,
    ) {
        // has already been verified at the current timestamp
        if computed_callee.verified_at == self.database.current_timestamp {
            return;
        }

        let recompute =
            self.should_recompute_query(query_id, &mut computed_callee).await;

        // the query is clean, update the verified timestamp
        if !recompute {
            computed_callee.verified_at = self.database.current_timestamp;
            self.database.set_computed(&query_id.id, computed_callee, notify);
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
            let callees = computed_callee.callee_info.callee_order.read().await;

            // loop through each callee and remove their backward dependencies
            // OPTIMIZE: this can be parallelized
            for callee in callees.iter() {
                self.database
                    .unwire_backward_dependencies(&query_id.id, callee)
                    .await;
            }

            // clear all callees, that might have been added during repairing
            self.database
                .query_metas
                .get_mut(&query_id.id)
                .expect("should be present")
                .state
                .as_computing_mut()
                .expect("should've been runninng")
                .callee_info = CalleeInfo::default();
        }

        // now, execute the query again
        self.execute_query(query_id, Some(computed_callee.caller_info), notify)
            .await;
    }

    pub(crate) fn recursively_repair_query<'a, Q: Query>(
        engine: &'a Arc<Self>,
        key: &'a dyn Any,
        called_from: &'a QueryID,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<(), CyclicError>> + 'a>>
    {
        let key = key
            .downcast_ref::<Q>()
            .expect("should be of the correct query type");

        Box::pin(async move {
            let query_with_id = engine.database.new_query_with_id(key);

            engine.query_for(&query_with_id, false, Some(called_from)).await?;

            Ok(())
        })
    }

    pub(super) async fn execute_query<Q: Query>(
        self: &Arc<Self>,
        query: &QueryWithID<'_, Q>,
        caller_info: Option<CallerInfo>,
        notify: &Notify,
    ) {
        // create a new tracked engine
        let cache = Arc::new(DashMap::default());

        let mut tracked_engine = TrackedEngine {
            engine: self.clone(),
            cache: cache.clone(),
            caller: Some(query.id),
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
            .query_metas
            .get(&query.id)
            .expect("should be present with running state")
            .state
            .as_computing()
            .expect("should'be been computing")
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

        self.database.done_compute(&query.id, value, caller_info, notify);
    }

    pub(super) async fn continuation<Q: Query>(
        self: &Arc<Self>,
        query: &QueryWithID<'_, Q>,
        notify: &Notify,
        continuation: Continuation<C>,
    ) {
        match continuation {
            Continuation::Fresh => {
                self.execute_query(query, None, notify).await;
            }

            Continuation::Reverify(computed) => {
                self.repair_query(query, computed, notify).await;
            }
        }
    }
}
