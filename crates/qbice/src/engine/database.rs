use std::{any::Any, collections::VecDeque, pin::Pin, sync::Arc};

use dashmap::{
    DashMap,
    mapref::one::{MappedRef, MappedRefMut, Ref, RefMut},
};
use tokio::sync::Notify;

use crate::{
    config::Config,
    engine::{
        Engine, TrackedEngine,
        meta::{
            self, Computed, Computing, QueryMeta, QueryWithID, SetInputResult,
            State,
        },
    },
    executor::CyclicError,
    query::{DynValue, Query, QueryID},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Timtestamp(u64);

impl Timtestamp {
    pub const fn increment(&mut self) { self.0 += 1; }
}

pub struct Database<C: Config> {
    query_metas: DashMap<QueryID, QueryMeta<C>>,

    initial_seed: u64,
    current_timestamp: Timtestamp,
}

impl<C: Config> Database<C> {
    pub const fn current_timestamp(&self) -> Timtestamp {
        self.current_timestamp
    }

    pub const fn initial_seed(&self) -> u64 { self.initial_seed }
}

impl<C: Config> Default for Database<C> {
    fn default() -> Self {
        Self {
            query_metas: DashMap::default(),
            initial_seed: 0,
            current_timestamp: Timtestamp(0),
        }
    }
}

/// A drop guard for undoing the registration of a callee query.
///
/// This aims to ensure cancelation safety in case of the task being yielded and
/// canceled mid query.
pub struct UndoRegisterCallee<'d, 'c, C: Config> {
    database: &'d Database<C>,
    caller: Option<&'c Caller>,
    callee: QueryID,
    defused: bool,
}

impl<'d, 'c, C: Config> UndoRegisterCallee<'d, 'c, C> {
    /// Creates a new [`UndoRegisterCallee`] instance.
    pub const fn new(
        database: &'d Database<C>,
        caller_source: Option<&'c Caller>,
        callee: QueryID,
    ) -> Self {
        Self { database, caller: caller_source, callee, defused: false }
    }

    /// Don't undo the registration when dropped.
    pub fn defuse(mut self) { self.defused = true; }
}

impl<C: Config> Drop for UndoRegisterCallee<'_, '_, C> {
    fn drop(&mut self) {
        if self.defused {
            return;
        }

        if let Some(caller) = self.caller {
            let caller_meta = self.database.get_computing_caller(caller);

            caller_meta.remove_callee(&self.callee);
        }
    }
}

impl<C: Config> Engine<C> {
    fn register_callee<'c>(
        &self,
        caller: Option<&'c Caller>,
        calee: QueryID,
    ) -> Option<UndoRegisterCallee<'_, 'c, C>> {
        // record the dependency first, don't necessary need to figure out
        // the observed value fingerprint yet
        caller.map_or_else(
            || None,
            |caller| {
                let caller_meta = self.database.get_computing_caller(caller);

                caller_meta.add_callee(calee);

                Some(UndoRegisterCallee::new(
                    &self.database,
                    Some(caller),
                    calee,
                ))
            },
        )
    }

    fn is_in_scc(
        &self,
        called_from: Option<&Caller>,
    ) -> Result<(), CyclicError> {
        let Some(called_from) = called_from else {
            return Ok(());
        };

        if self
            .database
            .query_metas
            .get(&called_from.0)
            .expect("should be present")
            .is_running_in_scc()
        {
            return Err(CyclicError);
        }

        Ok(())
    }

    async fn query_for<Q: Query>(
        self: &Arc<Self>,
        query: &QueryWithID<'_, Q>,
        required_value: bool,
        caller: Option<&Caller>,
    ) -> Result<Option<Q::Value>, CyclicError> {
        // register the dependency for the sake of detecting cycles
        let undo_register = self.register_callee(caller, *query.id());

        // pulling the value
        let value = loop {
            match self
                .database
                .fast_path::<Q::Value>(query.id(), required_value, caller)
                .await
            {
                // try again
                Ok(meta::FastPathResult::TryAgain) => continue,

                // will continue down there
                Ok(meta::FastPathResult::ToSlowPath) => {}

                Ok(meta::FastPathResult::Hit(value)) => {
                    // defuse the undo `register_callee` since we have obtained
                    // the value, record the dependency successfully
                    if let Some(undo_register) = undo_register {
                        undo_register.defuse();
                    }

                    break value;
                }

                Err(e) => {
                    // defuse the undo `register_callee` keep cyclic depedency
                    // detection correct
                    if let Some(undo_register) = undo_register {
                        undo_register.defuse();
                    }

                    return Err(e);
                }
            }

            // now the `query` state is held in computing state.
            // if `lock_computing` is dropped without defusing, the state will
            // be restored to previous state (either computed or absent)
            let Some(lock_computing) = self.database.lock_computing(query)
            else {
                // try the fast path again
                continue;
            };

            // retry to the fast path and obtain value.
            self.continuation(query, lock_computing).await;
        };

        // check before returning the value
        self.is_in_scc(caller)?;

        Ok(value)
    }

    pub(crate) fn recursively_repair_query<'a, Q: Query>(
        engine: &'a Arc<Self>,
        key: &'a dyn Any,
        caller: &'a QueryID,
    ) -> Pin<
        Box<
            dyn std::future::Future<Output = Result<(), CyclicError>>
                + Send
                + 'a,
        >,
    > {
        let key = key
            .downcast_ref::<Q>()
            .expect("should be of the correct query type");

        Box::pin(async move {
            let query_with_id = engine.database.new_query_with_id(key);

            let caller = Caller::new(*caller);

            engine.query_for(&query_with_id, false, Some(&caller)).await?;

            Ok(())
        })
    }
}

impl<C: Config> TrackedEngine<C> {
    /// Query the database for a value.
    pub async fn query<Q: Query>(
        &self,
        query: &Q,
    ) -> Result<Q::Value, CyclicError> {
        // YIELD POINT: query function will be called very often, this is a
        // good point for yielding to allow cancelation.
        tokio::task::yield_now().await;

        let query_with_id = self.engine.database.new_query_with_id(query);

        // check local cache
        if let Some(value) = self.cache.get(query_with_id.id()) {
            // cache hit! don't have to go through central database
            let value: &dyn DynValue<C> = &**value;

            return Ok(value
                .downcast_value::<Q::Value>()
                .expect("should've been a correct type")
                .clone());
        }

        // run the main process
        self.engine
            .query_for(&query_with_id, true, self.caller.as_ref())
            .await
            .map(|x| x.unwrap())
    }
}

/// A struct allowing to set input values for queries.
#[derive(Debug)]
pub struct InputSession<'x, C: Config> {
    engine: &'x mut Engine<C>,
    incremented: bool,
    dirty_batch: VecDeque<QueryID>,
}

impl<C: Config> Engine<C> {
    /// Create a [`InputSession`] allowing to set input values for queries.
    #[must_use]
    pub const fn input_session(&mut self) -> InputSession<'_, C> {
        InputSession {
            engine: self,
            incremented: false,
            dirty_batch: VecDeque::new(),
        }
    }
}

impl<C: Config> InputSession<'_, C> {
    /// Set an input value for a query.
    pub fn set_input<Q: Query>(&mut self, query_key: Q, value: Q::Value) {
        let query_id = self.engine.database.query_id(&query_key);

        let SetInputResult { incremented, fingerprint_diff } = self
            .engine
            .database
            .set_input(query_key, query_id, value, self.incremented);

        self.incremented |= incremented;

        if fingerprint_diff {
            // insert into dirty batch
            self.dirty_batch.push_back(query_id);
        }
    }
}

impl<C: Config> Drop for InputSession<'_, C> {
    fn drop(&mut self) {
        // mark all dirty queries as dirty
        self.engine
            .database
            .dirty_queries(std::mem::take(&mut self.dirty_batch));
    }
}

/// A type representing the caller of a query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Caller(QueryID);

impl Caller {
    /// Creates a new [`Caller`] instance.
    pub const fn new(query_id: QueryID) -> Self { Self(query_id) }

    /// Returns the query ID of the caller.
    pub const fn query_id(&self) -> &QueryID { &self.0 }
}

/// A drop guard for the computing lock of a query.
pub struct ComputingLockGuard<'a, C: Config> {
    database: &'a Database<C>,
    query_id: QueryID,
    existing_computed: Option<Computed<C>>,
    notification: Arc<Notify>,
    defused: bool,
}

impl<C: Config> Drop for ComputingLockGuard<'_, C> {
    fn drop(&mut self) {
        if self.defused {
            return;
        }

        // held read lock first
        let query_meta = self
            .database
            .query_metas
            .get(&self.query_id)
            .expect("query ID should be present");

        // unwire backward dendencies
        self.database.unwire_backward_dependencies_from_callee(
            &self.query_id,
            &query_meta.get_computing().callee_info(),
        );

        drop(query_meta);

        // then mutably restore the state
        let mut query_meta = self
            .database
            .query_metas
            .get_mut(&self.query_id)
            .expect("query ID should be present");

        if let Some(computed) = self.existing_computed.take() {
            // restore to previous computed state
            query_meta.replace_state(State::Computed(computed));
        } else {
            // drop query meta lock
            drop(query_meta);

            // remove the query meta entirely
            self.database.query_metas.remove(&self.query_id);
        }

        self.notification.notify_waiters();
    }
}

impl<C: Config> ComputingLockGuard<'_, C> {
    /// Don't restore the previous state when dropped.
    fn defuse_and_notify(mut self) {
        self.defused = true;
        self.notification.notify_waiters();
    }

    pub const fn has_prior_computed(&self) -> bool {
        self.existing_computed.is_some()
    }

    pub const fn take_prior_computed(&mut self) -> Option<Computed<C>> {
        self.existing_computed.take()
    }

    pub const fn get_prior_computed_mut(&mut self) -> Option<&mut Computed<C>> {
        self.existing_computed.as_mut()
    }
}

impl<C: Config> Database<C> {
    /// Gets a mutable reference to the computing state of the running caller.
    ///
    /// We assume that the caller token is valid and corresponds to that the
    /// query is currently in computing state.
    pub(super) fn get_computing_caller(
        &self,
        caller: &Caller,
    ) -> MappedRef<'_, QueryID, QueryMeta<C>, Computing> {
        self.query_metas
            .get(&caller.0)
            .unwrap_or_else(|| panic!("query ID {:?} is not found", caller.0))
            .map(|x| x.get_computing())
    }

    pub(super) fn get_read_meta(
        &self,
        query_id: &QueryID,
    ) -> Ref<'_, QueryID, QueryMeta<C>> {
        self.query_metas
            .get(query_id)
            .unwrap_or_else(|| panic!("query ID {query_id:?} is not found"))
    }

    pub(super) fn try_get_read_meta(
        &self,
        query_id: &QueryID,
    ) -> Option<Ref<'_, QueryID, QueryMeta<C>>> {
        self.query_metas.get(query_id)
    }

    #[allow(unused)]
    pub(super) fn get_meta_mut(
        &self,
        query_id: &QueryID,
    ) -> RefMut<'_, QueryID, QueryMeta<C>> {
        self.query_metas
            .get_mut(query_id)
            .unwrap_or_else(|| panic!("query ID {query_id:?} is not found"))
    }

    #[allow(unused)]
    pub(super) fn try_get_meta_mut(
        &self,
        query_id: &QueryID,
    ) -> Option<RefMut<'_, QueryID, QueryMeta<C>>> {
        self.query_metas.get_mut(query_id)
    }

    #[allow(unused)]
    pub(super) fn try_get_computed_mut(
        &self,
        query_id: &QueryID,
    ) -> Option<MappedRefMut<'_, QueryID, QueryMeta<C>, Computed<C>>> {
        self.query_metas
            .get_mut(query_id)
            .map(|meta| meta.map(|x| x.get_computed_mut()))
    }

    pub(super) fn get_computed_mut(
        &self,
        query_id: &QueryID,
    ) -> MappedRefMut<'_, QueryID, QueryMeta<C>, Computed<C>> {
        self.query_metas
            .get_mut(query_id)
            .unwrap_or_else(|| panic!("query ID {query_id:?} is not found"))
            .map(|x| x.get_computed_mut())
    }

    pub(super) fn lock_computing<Q: Query>(
        &self,
        query_id: &QueryWithID<'_, Q>,
    ) -> Option<ComputingLockGuard<'_, C>> {
        match self.query_metas.entry(*query_id.id()) {
            dashmap::Entry::Occupied(mut occupied_entry) => {
                match occupied_entry.get().state() {
                    State::Computing(_) => {
                        // another thread is computing the query, try again
                        None
                    }

                    State::Computed(computed) => {
                        if computed.verified_at() == self.current_timestamp {
                            // already computed and verified, try fast path
                            // and retrieve value again
                            return None;
                        }

                        let noti = Arc::new(Notify::new());

                        let computed = occupied_entry
                            .get_mut()
                            .replace_state(State::Computing(Computing::new(
                                noti.clone(),
                            )))
                            .into_computed()
                            .expect("should've been computed");

                        Some(ComputingLockGuard {
                            database: self,
                            query_id: *query_id.id(),
                            existing_computed: Some(computed),
                            defused: false,
                            notification: noti,
                        })
                    }
                }
            }

            dashmap::Entry::Vacant(vacant_entry) => {
                let noti = Arc::new(Notify::new());
                let fresh_query_meta = QueryMeta::new(
                    query_id.query().clone(),
                    State::Computing(Computing::new(noti.clone())),
                );

                vacant_entry.insert(fresh_query_meta);

                Some(ComputingLockGuard {
                    database: self,
                    query_id: *query_id.id(),
                    existing_computed: None,
                    defused: false,
                    notification: noti,
                })
            }
        }
    }

    pub(super) fn set_computed_from_existing_lock_computing_and_defuse(
        &self,
        query_id: &Caller,
        mut lock_computing: ComputingLockGuard<'_, C>,
    ) {
        let mut query_meta = self
            .query_metas
            .get_mut(&query_id.0)
            .expect("query ID should be present");

        let computed = lock_computing
            .take_prior_computed()
            .expect("should have prior computing state");

        query_meta.replace_state(State::Computed(computed));

        // defuse the undoing of computing lock
        lock_computing.defuse_and_notify();
    }

    pub(super) fn set_computed_from_computing_and_defuse(
        &self,
        query_id: &Caller,
        computed: impl FnOnce(Computing) -> Computed<C>,
        lock_computing: ComputingLockGuard<'_, C>,
    ) {
        let mut query_meta = self
            .query_metas
            .get_mut(&query_id.0)
            .expect("query ID should be present");

        query_meta.take_state_mut(|x| {
            State::Computed(computed(x.into_computing().unwrap()))
        });

        // defuse the undoing of computing lock
        lock_computing.defuse_and_notify();
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
                let meta = occupied_entry.get_mut();

                meta.set_input::<Q>(
                    value,
                    incremented,
                    self.initial_seed,
                    &mut self.current_timestamp,
                )
            }

            // new vaccant input
            dashmap::Entry::Vacant(vacant_entry) => {
                vacant_entry.insert(QueryMeta::new_input(
                    query_key,
                    value,
                    &self.current_timestamp,
                    self.initial_seed,
                ));

                SetInputResult { incremented, fingerprint_diff: false }
            }
        }
    }
}
