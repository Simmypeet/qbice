use std::sync::Arc;

use dashmap::{
    DashMap, DashSet,
    mapref::one::{MappedRef, MappedRefMut, Ref, RefMut},
};
use tokio::sync::Notify;

use crate::{
    Query,
    config::Config,
    engine::{
        InitialSeed,
        database::{Database, Timtestamp},
        meta::{
            Computed, Computing, ComputingState, QueryMeta, QueryWithID, State,
        },
    },
    executor::Registry,
    query::QueryID,
};

pub struct Storage<C: Config> {
    query_metas: DashMap<QueryID, QueryMeta<C>>,

    dirtied_queries: DashSet<QueryID>,

    initial_seed: InitialSeed,
    current_timestamp: Timtestamp,
}

impl<C: Config> Default for Storage<C> {
    fn default() -> Self {
        Self {
            query_metas: DashMap::new(),
            dirtied_queries: DashSet::new(),
            initial_seed: InitialSeed::default(),
            current_timestamp: Timtestamp::default(),
        }
    }
}

/// A drop guard for the computing lock of a query.
pub struct ComputingLockGuard<'a, C: Config> {
    database: &'a Database<C>,
    prior_computed: Option<Computed<C>>,
    query_id: QueryID,
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
            .storage
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
            .storage
            .query_metas
            .get_mut(&self.query_id)
            .expect("query ID should be present");

        let existed_computed_from_computing = query_meta
            .get_computing_mut()
            .take_computed_placeholder_for_firewall_dirty_propagation();
        let prior_existed_computed = self.prior_computed.take();

        let existed_computed =
            existed_computed_from_computing.or(prior_existed_computed);

        if let Some(computed) = existed_computed {
            // restore to previous computed state
            query_meta.replace_state(State::Computed(computed));
        } else {
            // drop query meta lock
            drop(query_meta);

            // remove the query meta entirely
            self.database.storage.query_metas.remove(&self.query_id);
        }

        self.notification.notify_waiters();
    }
}

impl<C: Config> ComputingLockGuard<'_, C> {
    /// Don't restore the previous state when dropped.
    pub fn defuse_and_notify(mut self) {
        self.defused = true;
        self.notification.notify_waiters();
    }

    pub fn notification(&self) -> &Notify { &self.notification }

    pub const fn take_prior_computed(&mut self) -> Option<Computed<C>> {
        self.prior_computed.take()
    }

    pub const fn prior_computed(&self) -> &Computed<C> {
        self.prior_computed.as_ref().unwrap()
    }

    pub const fn prior_computed_mut(&mut self) -> &mut Computed<C> {
        self.prior_computed.as_mut().unwrap()
    }

    pub fn set_prior_computed(&mut self, computed: Computed<C>) {
        // shouldn't have prior computed already
        assert!(
            self.prior_computed.is_none(),
            "there's some previous computed"
        );

        self.prior_computed = Some(computed);
    }

    pub const fn has_prior_computed_placeholder(&self) -> bool {
        self.prior_computed.is_some()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SetInputResult {
    pub incremented: bool,
    pub fingerprint_diff: bool,
}

impl<C: Config> Database<C> {
    pub const fn current_timestamp(&self) -> Timtestamp {
        self.storage.current_timestamp
    }

    pub const fn initial_seed(&self) -> InitialSeed {
        self.storage.initial_seed
    }

    pub fn get_query_meta(
        &self,
        query_id: &QueryID,
    ) -> Option<Ref<'_, QueryID, QueryMeta<C>>> {
        self.storage.query_metas.get(query_id)
    }

    pub fn is_query_running_in_scc(&self, query_id: &QueryID) -> bool {
        self.storage.query_metas.get(query_id).unwrap().is_running_in_scc()
    }

    pub fn get_computing(
        &self,
        query_id: &QueryID,
    ) -> MappedRef<'_, QueryID, QueryMeta<C>, Computing<C>> {
        self.storage
            .query_metas
            .get(query_id)
            .unwrap_or_else(|| panic!("query ID {query_id:?} is not found"))
            .map(|x| x.get_computing())
    }

    pub fn get_computing_mut(
        &self,
        query_id: &QueryID,
    ) -> MappedRefMut<'_, QueryID, QueryMeta<C>, Computing<C>> {
        self.storage
            .query_metas
            .get_mut(query_id)
            .unwrap_or_else(|| panic!("query ID {query_id:?} is not found"))
            .map(|x| x.get_computing_mut())
    }

    pub fn get_read_meta(
        &self,
        query_id: &QueryID,
    ) -> Ref<'_, QueryID, QueryMeta<C>> {
        self.storage
            .query_metas
            .get(query_id)
            .unwrap_or_else(|| panic!("query ID {query_id:?} is not found"))
    }

    pub fn try_get_read_meta(
        &self,
        query_id: &QueryID,
    ) -> Option<Ref<'_, QueryID, QueryMeta<C>>> {
        self.storage.query_metas.get(query_id)
    }

    #[allow(unused)]
    pub fn get_meta_mut(
        &self,
        query_id: &QueryID,
    ) -> RefMut<'_, QueryID, QueryMeta<C>> {
        self.storage
            .query_metas
            .get_mut(query_id)
            .unwrap_or_else(|| panic!("query ID {query_id:?} is not found"))
    }

    #[allow(unused)]
    pub fn try_get_meta_mut(
        &self,
        query_id: &QueryID,
    ) -> Option<RefMut<'_, QueryID, QueryMeta<C>>> {
        self.storage.query_metas.get_mut(query_id)
    }

    #[allow(unused)]
    pub fn try_get_computed_mut(
        &self,
        query_id: &QueryID,
    ) -> Option<MappedRefMut<'_, QueryID, QueryMeta<C>, Computed<C>>> {
        self.storage
            .query_metas
            .get_mut(query_id)
            .map(|meta| meta.map(|x| x.get_computed_mut()))
    }

    #[allow(unused)]
    pub fn get_computed_mut(
        &self,
        query_id: &QueryID,
    ) -> MappedRefMut<'_, QueryID, QueryMeta<C>, Computed<C>> {
        self.storage
            .query_metas
            .get_mut(query_id)
            .unwrap_or_else(|| panic!("query ID {query_id:?} is not found"))
            .map(|x| x.get_computed_mut())
    }

    pub fn lock_computing<Q: Query>(
        &self,
        query_id: &QueryWithID<'_, Q>,
        executor_registry: &Registry<C>,
    ) -> Option<ComputingLockGuard<'_, C>> {
        match self.storage.query_metas.entry(*query_id.id()) {
            dashmap::Entry::Occupied(mut occupied_entry) => {
                match occupied_entry.get().state() {
                    State::Computing(_) => {
                        // another thread is computing the query, try again
                        None
                    }

                    State::Computed(computed) => {
                        if computed.verified_at()
                            == self.storage.current_timestamp
                        {
                            // already computed and verified, try fast path
                            // and retrieve value again
                            return None;
                        }

                        let noti = Arc::new(Notify::new());

                        let computed = occupied_entry
                            .get_mut()
                            .replace_state(State::Computing(Computing::new(
                                noti.clone(),
                                ComputingState::Repair,
                            )))
                            .into_computed()
                            .expect("should've been computed");

                        Some(ComputingLockGuard {
                            database: self,
                            prior_computed: Some(computed),
                            query_id: *query_id.id(),
                            defused: false,
                            notification: noti,
                        })
                    }
                }
            }

            dashmap::Entry::Vacant(vacant_entry) => {
                let noti = Arc::new(Notify::new());
                let executor = executor_registry.get_executor_entry::<Q>();
                let fresh_query_meta = QueryMeta::new(
                    query_id.query().clone(),
                    State::Computing(Computing::new(
                        noti.clone(),
                        ComputingState::Fresh,
                    )),
                    executor.obtain_execution_style(),
                );

                vacant_entry.insert(fresh_query_meta);

                // no prior computed state

                Some(ComputingLockGuard {
                    database: self,
                    prior_computed: None,
                    query_id: *query_id.id(),
                    defused: false,
                    notification: noti,
                })
            }
        }
    }

    pub fn clear_dirty_queries(&self) { self.storage.dirtied_queries.clear(); }

    pub fn insert_dirty_query(&self, query_id: QueryID) -> bool {
        self.storage.dirtied_queries.insert(query_id)
    }

    pub fn set_computed_from_existing_lock_computing_and_defuse(
        &self,
        query_id: &QueryID,
        mut lock_computing: ComputingLockGuard<'_, C>,
    ) {
        let mut query_meta = self
            .storage
            .query_metas
            .get_mut(query_id)
            .expect("query ID should be present");

        query_meta.replace_state(State::Computed(
            lock_computing
                .prior_computed
                .take()
                .expect("prior computed should be present"),
        ));

        // defuse the undoing of computing lock
        lock_computing.defuse_and_notify();
    }

    pub fn set_computed_from_computing_and_defuse(
        &self,
        query_id: &QueryID,
        computed: impl FnOnce(Computing<C>) -> Computed<C>,
        lock_computing: ComputingLockGuard<'_, C>,
    ) {
        let mut query_meta = self
            .storage
            .query_metas
            .get_mut(query_id)
            .expect("query ID should be present");

        query_meta.take_state_mut(|x| {
            State::Computed(computed(x.into_computing().unwrap()))
        });

        // defuse the undoing of computing lock
        lock_computing.defuse_and_notify();
    }

    pub async fn spin_get_computed(
        &self,
        query_id: &QueryID,
    ) -> MappedRef<'_, QueryID, QueryMeta<C>, Computed<C>> {
        loop {
            let meta = self.storage.query_metas.get(query_id).unwrap();
            match meta.state() {
                State::Computing(computing) => {
                    // wait for notification
                    let noti = computing.notification_owned();
                    drop(meta);

                    noti.notified().await;
                }
                State::Computed(_) => {
                    return meta.map(|x| x.get_computed());
                }
            }
        }
    }

    pub(super) fn set_input<Q: Query>(
        &mut self,
        query_key: Q,
        query_id: QueryID,
        value: Q::Value,
        incremented: bool,
    ) -> SetInputResult {
        match self.storage.query_metas.entry(query_id) {
            dashmap::Entry::Occupied(mut occupied_entry) => {
                let meta = occupied_entry.get_mut();

                meta.set_input::<Q>(
                    value,
                    incremented,
                    self.storage.initial_seed,
                    &mut self.storage.current_timestamp,
                )
            }

            // new vaccant input
            dashmap::Entry::Vacant(vacant_entry) => {
                vacant_entry.insert(QueryMeta::new_input(
                    query_key,
                    value,
                    &self.storage.current_timestamp,
                    self.storage.initial_seed,
                ));

                SetInputResult { incremented, fingerprint_diff: false }
            }
        }
    }
}
