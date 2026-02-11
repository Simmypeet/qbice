use std::{
    ops::Not,
    pin::Pin,
    sync::{Arc, atomic::AtomicBool},
};

use fxhash::FxHashSet;
use qbice_stable_hash::Compact128;
use tokio::task::JoinSet;
use tracing::instrument;

use super::database::{ForwardEdgeObservation, NodeDependency};
use crate::{
    Engine, Query,
    config::Config,
    engine::computation_graph::{
        ActiveComputationGuard, QueryStatus, QueryWithID,
        caller::{CallerInformation, CallerKind, CallerReason, QueryCaller},
        computing::{ComputingLockGuard, QueryComputing},
        database::{Snapshot, Timestamp},
    },
    executor::CyclicError,
    query::QueryID,
};

#[derive(Debug)]
pub enum RepairDecision {
    Recompute,
    Clean {
        repair_transitive_firewall_callees: bool,
        cleaned_edges: Vec<QueryID>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CalleeCheckDecision {
    /// The edge was dirty, result differs, need to recompute the caller.
    Recompute,

    /// The edge was not dirty, no further action done.
    NoNeed,

    /// The edge was seen dirty, but result stays the same after repair.
    Cleaned {
        repair_transitive_firewall_callees: bool,
        add_to_clean_list: bool,
    },
}

pub enum ChunkedCalleeCheckDecision {
    Recompute,
    Cancelled,
    Cleaned {
        repair_transitive_firewall_callees: bool,
        cleaned_edges: Vec<QueryID>,
    },
}

impl<C: Config, Q: Query> Snapshot<C, Q> {
    #[instrument(
        skip(self, caller_information, lock_guard),
        level = "info",
        name = "repair_query",
        target = "qbice"
    )]
    pub(super) async fn repair_query(
        self,
        query: &Q,
        caller_information: &CallerInformation,
        lock_guard: ComputingLockGuard<C>,
    ) {
        let Some((lock_guard, snapshot)) =
            self.should_recompute_query(caller_information, lock_guard).await
        else {
            return;
        };

        // recompute the query
        snapshot
            .execute_query(
                query,
                super::slow_path::ExecuteQueryFor::RecomputeQuery,
                caller_information,
                lock_guard,
            )
            .await;
    }

    #[allow(clippy::too_many_lines)]
    async fn should_recompute_query(
        mut self,
        caller_information: &CallerInformation,
        lock_guard: ComputingLockGuard<C>,
    ) -> Option<(ComputingLockGuard<C>, Self)> {
        // if the caller is backward projection propagation, we always
        // recompute since the projection query have already told us
        // that the value is required to be recomputed.
        if matches!(
            caller_information.kind(),
            CallerKind::BackwardProjectionPropagation
        ) {
            return Some((lock_guard, self));
        }

        // continue normal path ...

        // repair transitive firewall callees first before deciding whether to
        // recompute, since the transitive firewall callees might affect the
        // decision by propagating dirtiness.
        if matches!(
            caller_information.kind(),
            CallerKind::User | CallerKind::RepairFirewall { .. }
        ) {
            self.repair_transitive_firewall_callees(caller_information).await;
        }

        let recompute = self
            .recompute_decision_based_on_forward_edges(
                caller_information,
                &lock_guard,
            )
            .await;

        let (repair_transitive_firewall_callees, cleaned_edges) =
            match recompute {
                RepairDecision::Recompute => return Some((lock_guard, self)),
                RepairDecision::Clean {
                    repair_transitive_firewall_callees,
                    cleaned_edges,
                } => (repair_transitive_firewall_callees, cleaned_edges),
            };

        if repair_transitive_firewall_callees.not() {
            self.computing_lock_to_clean_query(
                cleaned_edges,
                None,
                caller_information,
                lock_guard,
            )
            .await;
        } else {
            let forward_edges = self.forward_edge_order().await.unwrap();

            let expected_parallelism = std::thread::available_parallelism()
                .map_or_else(|_| 4, |x| x.get() * 4);
            let chunk_size =
                std::cmp::max(cleaned_edges.len() / expected_parallelism, 1);

            let mut handles = JoinSet::new();

            for chunk in cleaned_edges.chunks(chunk_size).map(<[_]>::to_vec) {
                let engine = self.engine().clone();
                let timestamp = caller_information.timestamp();
                let query_computing = lock_guard.query_computing().clone();
                let active_computation_guard =
                    caller_information.clone_active_computation_guard();

                handles.spawn(async move {
                    for callee in chunk {
                        let entry = engine
                            .executor_registry
                            .get_executor_entry_by_type_id(
                                &callee.stable_type_id(),
                            );

                        let _ = entry
                            .repair_query_from_query_id(
                                &engine,
                                &callee.compact_hash_128(),
                                &CallerInformation::new(
                                    CallerKind::Query(QueryCaller::new(
                                        callee,
                                        CallerReason::Repair,
                                        query_computing.clone(),
                                    )),
                                    timestamp,
                                    active_computation_guard.clone(),
                                ),
                            )
                            .await;
                    }
                });
            }

            // join all handles
            while let Some(handle) = handles.join_next().await {
                handle.unwrap();
            }

            let mut new_tfcs = FxHashSet::default();

            for x in forward_edges.iter_all_callees() {
                let kind = self.engine().get_query_kind(&x).await;

                if kind.is_firewall() {
                    new_tfcs.insert(x);
                } else {
                    let callee_info = unsafe {
                        self.engine().get_node_info_unchecked(&x).await
                    };

                    new_tfcs.extend(
                        callee_info
                            .transitive_firewall_callees()
                            .iter()
                            .copied(),
                    );
                }
            }

            let new_tfc = self.engine().create_tfc(new_tfcs);

            self.computing_lock_to_clean_query(
                cleaned_edges,
                Some(new_tfc),
                caller_information,
                lock_guard,
            )
            .await;
        }

        None
    }

    pub(super) async fn repair_transitive_firewall_callees(
        &mut self,
        caller_information: &CallerInformation,
    ) {
        let node_info = self.node_info().await.unwrap();
        let is_current_query_projection =
            self.query_kind().await.unwrap().is_projection();

        let tfcs = node_info.transitive_firewall_callees();
        let tfcs = tfcs.iter().copied().collect::<Vec<_>>();

        let chunk_size = std::cmp::max(
            tfcs.len()
                / std::thread::available_parallelism()
                    .map_or_else(|_| 4, |x| x.get() * 4),
            1,
        );

        let mut join_set = JoinSet::new();

        // run all repairs in parallel
        let timestamp = caller_information.timestamp();

        for tfc_chunk in tfcs.chunks(chunk_size).map(<[_]>::to_vec) {
            let engine = self.engine().clone();
            let active_computation_guard =
                caller_information.clone_active_computation_guard();

            join_set.spawn(async move {
                for tfc in tfc_chunk {
                    let executor = engine
                        .executor_registry
                        .get_executor_entry_by_type_id(&tfc.stable_type_id());

                    let _ = executor
                        .repair_query_from_query_id(
                            &engine,
                            &tfc.compact_hash_128(),
                            &CallerInformation::new(
                                CallerKind::RepairFirewall {
                                    // if current query is projection, then
                                    // repairing the firewalls should not
                                    // invoke
                                    // backward projection, since it will
                                    // immediately request this query again,
                                    // causing
                                    // deadlock.
                                    invoke_backward_projection:
                                        !is_current_query_projection,
                                },
                                timestamp,
                                active_computation_guard.clone(),
                            ),
                        )
                        .await;
                }
            });
        }

        // join all handles
        while let Some(handle) = join_set.join_next().await {
            handle.unwrap();
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn check_callee(
        engine: &Arc<Engine<C>>,
        query_id: &QueryID,
        callee: &QueryID,
        forward_edge_observation: &ForwardEdgeObservation<C>,
        current_timestamp: Timestamp,
        active_computation_guard: Option<&ActiveComputationGuard>,
        query_computing: &Arc<QueryComputing>,
        pedantic_repair: bool,
    ) -> CalleeCheckDecision {
        // skip if not dirty
        // however, we can't skip if pedantic_repair is true
        let edge_is_dirty = engine.is_edge_dirty(*query_id, *callee).await;

        if !edge_is_dirty && !pedantic_repair {
            return CalleeCheckDecision::NoNeed;
        }

        let kind = engine.get_query_kind(callee).await;

        // NOTE: if the callee is an input (explicitly set), it's impossible
        // to try to repair it, so we'll skip repairing and directly
        // compare the fingerprint.
        if !kind.is_input() {
            // recursively repair the callee first
            let entry = engine
                .executor_registry
                .get_executor_entry_by_type_id(&callee.stable_type_id());

            let _ = entry
                .repair_query_from_query_id(
                    engine,
                    &callee.compact_hash_128(),
                    &CallerInformation::new(
                        CallerKind::Query(QueryCaller::new(
                            *query_id,
                            CallerReason::Repair,
                            query_computing.clone(),
                        )),
                        current_timestamp,
                        active_computation_guard.cloned(),
                    ),
                )
                .await;
        }

        let mut repair_transitive_firewall_callees = false;

        // after repairing, compare the fingerprints to see if we need to
        // recompute
        {
            // SAFETY: we have just repaired the callee, so the node info
            // must exist and immutable now.
            let callee_node_info =
                unsafe { engine.get_node_info_unchecked(callee).await };

            let value_fingerprint_diff = callee_node_info.value_fingerprint()
                != forward_edge_observation
                    .0
                    .get(callee)
                    .unwrap()
                    .seen_value_fingerprint;

            // if any of the callee's value fingerprint differs, we need to
            // recompute
            if value_fingerprint_diff {
                return CalleeCheckDecision::Recompute;
            }

            // check wherther the transitive firewall callee needs repair
            if !kind.is_firewall() {
                let tfc_fingerprint_diff = callee_node_info
                    .transitive_firewall_callees_fingerprint()
                    != forward_edge_observation
                        .0
                        .get(callee)
                        .unwrap()
                        .seen_transitive_firewall_callees_fingerprint;

                if tfc_fingerprint_diff {
                    repair_transitive_firewall_callees = true;
                }
            }

            CalleeCheckDecision::Cleaned {
                repair_transitive_firewall_callees,
                add_to_clean_list: edge_is_dirty,
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn check_callee_chunked(
        engine: &Arc<Engine<C>>,
        query_id: &QueryID,
        callees: &[QueryID],
        forward_edge_observation: &ForwardEdgeObservation<C>,
        current_timestamp: Timestamp,
        active_computation_guard: Option<&ActiveComputationGuard>,
        computing_lock_guard: &Arc<QueryComputing>,
        pedantic_repair: bool,
        cancelled: Arc<AtomicBool>,
    ) -> ChunkedCalleeCheckDecision {
        let mut cleaned_edges = Vec::new();
        let mut repair_transitive_firewall_callees = false;

        for callee in callees {
            // other threads have found dirty edge that differs, cancel
            // further checking.
            if cancelled.load(std::sync::atomic::Ordering::Relaxed) {
                return ChunkedCalleeCheckDecision::Cancelled;
            }

            let decision = Self::check_callee(
                engine,
                query_id,
                callee,
                forward_edge_observation,
                current_timestamp,
                active_computation_guard,
                computing_lock_guard,
                pedantic_repair,
            )
            .await;

            match decision {
                CalleeCheckDecision::Recompute => {
                    // notify other potentially running tasks that we've found
                    // one dirty edge that requires recompute. no need to
                    // continue checking other edges.
                    cancelled.store(true, std::sync::atomic::Ordering::Relaxed);

                    return ChunkedCalleeCheckDecision::Recompute;
                }

                CalleeCheckDecision::NoNeed => {}

                CalleeCheckDecision::Cleaned {
                    repair_transitive_firewall_callees: repair_tfc_needed,
                    add_to_clean_list,
                } => {
                    if add_to_clean_list {
                        cleaned_edges.push(*callee);
                    }

                    if repair_tfc_needed {
                        repair_transitive_firewall_callees = true;
                    }
                }
            }
        }

        ChunkedCalleeCheckDecision::Cleaned {
            repair_transitive_firewall_callees,
            cleaned_edges,
        }
    }

    #[allow(clippy::too_many_lines)]
    async fn recompute_decision_based_on_forward_edges(
        &mut self,
        caller_information: &CallerInformation,
        computing_lock_guard: &ComputingLockGuard<C>,
    ) -> RepairDecision {
        let mut repair_transitive_firewall_callees = false;
        let mut cleaned_edges = Vec::new();

        let forward_edges = self.forward_edge_order().await.unwrap();
        let forward_edge_observation =
            self.forward_edge_observation().await.unwrap();

        for dep in forward_edges.0.iter() {
            match dep {
                NodeDependency::Single(callee) => {
                    let decision = Self::check_callee(
                        self.engine(),
                        self.query_id(),
                        callee,
                        &forward_edge_observation,
                        caller_information.timestamp(),
                        caller_information.active_computation_guard(),
                        computing_lock_guard.query_computing(),
                        caller_information.get_query_caller().is_some_and(
                            super::caller::QueryCaller::pedantic_repair,
                        ),
                    )
                    .await;

                    match decision {
                        CalleeCheckDecision::Recompute => {
                            return RepairDecision::Recompute;
                        }

                        CalleeCheckDecision::NoNeed => {}

                        CalleeCheckDecision::Cleaned {
                            repair_transitive_firewall_callees:
                                repair_tfc_needed,
                            add_to_clean_list,
                        } => {
                            if add_to_clean_list {
                                cleaned_edges.push(*callee);
                            }

                            if repair_tfc_needed {
                                repair_transitive_firewall_callees = true;
                            }
                        }
                    }
                }

                NodeDependency::Unordered(query_ids) => {
                    let expected_parallelism =
                        std::thread::available_parallelism()
                            .map_or_else(|_| 4, |x| x.get() * 4);

                    let chunk_size = std::cmp::max(
                        query_ids.len() / expected_parallelism,
                        1,
                    );

                    let cancelled = Arc::new(AtomicBool::new(false));
                    let mut chunk_handles = Vec::new();

                    for chunk in query_ids.chunks(chunk_size).map(<[_]>::to_vec)
                    {
                        let engine = self.engine().clone();
                        let query_id = *self.query_id();
                        let forward_edge_observation =
                            forward_edge_observation.clone();
                        let cancelled = cancelled.clone();
                        let timestamp = caller_information.timestamp();
                        let active_computation_guard =
                            caller_information.clone_active_computation_guard();
                        let computing_lock_guard =
                            computing_lock_guard.query_computing().clone();
                        let pedantic_repair =
                            caller_information.get_query_caller().is_some_and(
                                super::caller::QueryCaller::pedantic_repair,
                            );

                        chunk_handles.push(tokio::spawn(async move {
                            Self::check_callee_chunked(
                                &engine,
                                &query_id,
                                &chunk,
                                &forward_edge_observation,
                                timestamp,
                                active_computation_guard.as_ref(),
                                &computing_lock_guard,
                                pedantic_repair,
                                cancelled,
                            )
                            .await
                        }));
                    }

                    let mut found_recompute = false;
                    for handle in chunk_handles {
                        if found_recompute {
                            // tell the other tasks to cancel the checking
                            // as we've found one dirty edge that requires
                            // recompute.
                            handle.abort();
                        }

                        match handle.await.unwrap() {
                            ChunkedCalleeCheckDecision::Cancelled
                            | ChunkedCalleeCheckDecision::Recompute => {
                                found_recompute = true;
                            }

                            ChunkedCalleeCheckDecision::Cleaned {
                                repair_transitive_firewall_callees:
                                    repair_tfc_needed,
                                cleaned_edges: mut edges,
                            } => {
                                // already found one edge that requires
                                // recompute, skip
                                if found_recompute {
                                    continue;
                                }

                                cleaned_edges.append(&mut edges);

                                if repair_tfc_needed {
                                    repair_transitive_firewall_callees = true;
                                }
                            }
                        }
                    }

                    if found_recompute {
                        return RepairDecision::Recompute;
                    }
                }
            }
        }

        RepairDecision::Clean {
            repair_transitive_firewall_callees,
            cleaned_edges,
        }
    }
}

impl<C: Config> Engine<C> {
    pub(crate) fn repair_query_from_query_id<'x, Q: Query>(
        self: &'x Arc<Self>,
        query_id: &'x Compact128,
        called_from: &'x CallerInformation,
    ) -> Pin<
        Box<dyn Future<Output = Result<QueryStatus, CyclicError>> + Send + 'x>,
    > {
        Box::pin(async move {
            let query_input = self.get_query_input::<Q>(query_id).await;

            let query_for = QueryWithID {
                id: QueryID::new::<Q>(*query_id),
                query: &query_input,
            };

            self.query_for(&query_for, called_from).await.map(|x| x.status)
        })
    }
}
