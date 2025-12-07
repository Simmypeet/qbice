use std::{
    collections::VecDeque,
    sync::{Arc, RwLock, atomic::AtomicBool},
};

use dashmap::{DashMap, DashSet};
use enum_as_inner::EnumAsInner;
use tokio::sync::Notify;

use crate::{
    config::Config,
    engine::{
        Engine, TrackedEngine,
        database::{Database, Timtestamp},
    },
    executor::CyclicQuery,
    query::{DynQuery, DynValue, DynValueBox, Query, QueryID, QueryKind},
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
                transitive_firewall_callees: Arc::new(DashSet::new()),
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
    dirty: bool,

    fingerprint: Compact128,
}

#[derive(Debug, EnumAsInner)]
pub enum State<C: Config> {
    Computing(Computing),
    Computed(Computed<C>),
}

#[derive(Debug, Default)]
pub struct CalleeInfo {
    callee_queries: DashMap<QueryID, Option<Compact128>>,
    callee_order: RwLock<Vec<QueryID>>,
    transitive_firewall_callees: Arc<DashSet<QueryID>>,
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
        computing.callee_info.callee_order.write().unwrap().push(callee);
    }

    pub fn is_running_in_scc(&self) -> bool {
        let computing = self
            .state
            .as_computing()
            .expect("can only check scc while computing");

        computing.is_in_scc.load(std::sync::atomic::Ordering::Relaxed)
    }
}

pub type DynQueryBox<C> =
    smallbox::SmallBox<dyn DynQuery, <C as Config>::Storage>;

#[derive(Debug, Clone)]
pub struct QueryWithID<'c, Q: Query> {
    id: QueryID,
    query: &'c Q,
}

impl<'c, Q: Query> QueryWithID<'c, Q> {
    pub const fn id(&self) -> &QueryID { &self.id }

    pub const fn query(&self) -> &'c Q { self.query }
}

impl<C: Config> Database<C> {
    /// Create a new query with its associated unique identifier.
    pub(super) fn new_query_with_id<'c, Q: Query>(
        &'c self,
        query: &'c Q,
    ) -> QueryWithID<'c, Q> {
        QueryWithID { id: query.query_identifier(self.initial_seed), query }
    }

    pub(super) fn query_id<Q: Query>(&self, query: &Q) -> QueryID {
        query.query_identifier(self.initial_seed)
    }
}

pub enum FastPathResult<V> {
    TryAgain,
    ToSlowPath,
    Hit(Option<V>),
}

#[derive(Debug)]
pub enum Continuation<C: Config> {
    Fresh,
    Reverify(Computed<C>),
}

#[derive(Debug)]
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
    fn observe_callee_fingerprint(
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
                .insert(*callee_target, Some(computed_callee.fingerprint))
                .is_some();

            // if haven't inserted, add to dependency order
            if inserted {
                caller_computing
                    .callee_info
                    .callee_order
                    .write()
                    .unwrap()
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
    ) -> Result<(), CyclicQuery> {
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

            return Err(CyclicQuery);
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
    ) -> Result<FastPathResult<V>, CyclicQuery> {
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

                if let Some(caller) = caller {
                    // register dependency
                    let caller_source_meta =
                        self.query_metas.get(caller).unwrap();

                    Self::observe_callee_fingerprint(
                        &caller_source_meta,
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

    fn done_compute(
        &self,
        query_id: &QueryID,
        value: DynValueBox<C>,
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
                caller_info: CallerInfo::default(),
                callee_info: computing.callee_info,
                verified_at: self.current_timestamp,
                dirty: false,
                fingerprint: Compact128::from_u128(hash_128),
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
                        computed.dirty = false;
                        computed.result = smallbox::smallbox!(value);

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
                        dirty: false,
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
        // TOOD: we could potentially have worker threads to process dirty
        // queries
        while let Some(query_id) = queries.pop_front() {
            if let Some(mut meta) = self.query_metas.get_mut(&query_id) {
                match &mut meta.state {
                    State::Computing(_) => unreachable!(
                        "shouldn't exist since we've obtained exclusive \
                         mutable reference"
                    ),

                    State::Computed(computed) => {
                        // short-circuit if already dirty
                        if computed.dirty {
                            continue;
                        }

                        computed.dirty = true;

                        // propagate to callers
                        for caller in computed.caller_info.caller_queries.iter()
                        {
                            queries.push_back(*caller.key());
                        }
                    }
                }
            }
        }
    }
}

impl<C: Config> Engine<C> {
    pub(super) async fn continuation<Q: Query>(
        self: &Arc<Self>,
        caller: Option<&QueryID>,
        query: &QueryWithID<'_, Q>,
        notify: &Notify,
        continuation: Continuation<C>,
    ) {
        match continuation {
            Continuation::Fresh => {
                // create a new tracked engine
                let cache = Arc::new(DashMap::default());

                let mut tracked_engine = TrackedEngine {
                    engine: self.clone(),
                    cache: cache.clone(),
                    caller: Some(query.id),
                };

                let entry = self
                    .executor_registry
                    .get_executor_entry::<Q>()
                    .unwrap_or_else(|| {
                        panic!(
                            "Executor for query type {:?} not found",
                            std::any::type_name::<Q>()
                        )
                    });

                let result = entry
                    .invoke_executor(query.query, &mut tracked_engine)
                    .await;

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
                    "`TrackedEngine` is still held elsewhere, this is a bug \
                     in the query implementation which violates the query \
                     system's contract. It's possible that the \
                     `TrackedEngine` is being sent to a different thread and \
                     the query implementation hasn't properly joined the \
                     thread before returning the value. Key: `{}`",
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
                    let smallbox: DynValueBox<C> =
                        smallbox::smallbox!(Q::scc_value());

                    smallbox
                });

                self.database.done_compute(&query.id, value, notify);
            }

            Continuation::Reverify(computed) => todo!(),
        }
    }
}
