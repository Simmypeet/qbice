use std::{ops::Not, pin::Pin, sync::Arc};

use fxhash::FxHashSet;
use qbice_stable_hash::Compact128;

use crate::{
    Engine, Query,
    config::Config,
    engine::computation_graph::{
        QueryStatus, QueryWithID,
        caller::{CallerInformation, CallerKind, CallerReason, QueryCaller},
        computing::ComputingLockGuard,
        database::Snapshot,
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

impl<C: Config, Q: Query> Snapshot<C, Q> {
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

            // repair all callees
            for callee in forward_edges.0.iter() {
                let entry = self
                    .engine()
                    .executor_registry
                    .get_executor_entry_by_type_id(&callee.stable_type_id());

                let _ = entry
                    .repair_query_from_query_id(
                        self.engine(),
                        &callee.compact_hash_128(),
                        &CallerInformation::new(
                            CallerKind::Query(QueryCaller::new(
                                *self.query_id(),
                                CallerReason::Repair,
                                lock_guard.query_computing().clone(),
                            )),
                            caller_information.timestamp(),
                            caller_information.clone_active_computation_guard(),
                        ),
                    )
                    .await;
            }

            let mut new_tfcs = FxHashSet::default();

            for x in forward_edges.0.iter() {
                let kind = self.engine().get_query_kind(x).await;

                if kind.is_firewall() {
                    new_tfcs.insert(*x);
                } else {
                    let callee_info = unsafe {
                        self.engine().get_node_info_unchecked(x).await
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

    async fn repair_transitive_firewall_callees(
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
                    .map(std::num::NonZero::get)
                    .unwrap_or(1),
            1,
        );

        let mut handles = Vec::new();

        // run all repairs in parallel
        let timestamp = caller_information.timestamp();

        for tfc_chunk in tfcs.chunks(chunk_size).map(<[_]>::to_vec) {
            let engine = self.engine().clone();
            let active_computation_guard =
                caller_information.clone_active_computation_guard();

            handles.push(tokio::spawn(async move {
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
                                    // repairing the firewalls should not invoke
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
            }));
        }

        // join all handles
        for handle in handles {
            handle.await.unwrap();
        }
    }

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

        for callee in forward_edges.0.iter() {
            // skip if not dirty
            if !self.is_edge_dirty(*callee).await {
                continue;
            }

            let kind = self.engine().get_query_kind(callee).await;

            // NOTE: if the callee is an input (explicitly set), it's impossible
            // to try to repair it, so we'll skip repairing and directly
            // compare the fingerprint.
            if !kind.is_input() {
                // recursively repair the callee first
                let entry = self
                    .engine()
                    .executor_registry
                    .get_executor_entry_by_type_id(&callee.stable_type_id());

                let _ = entry
                    .repair_query_from_query_id(
                        self.engine(),
                        &callee.compact_hash_128(),
                        &CallerInformation::new(
                            CallerKind::Query(QueryCaller::new(
                                *self.query_id(),
                                CallerReason::Repair,
                                computing_lock_guard.query_computing().clone(),
                            )),
                            caller_information.timestamp(),
                            caller_information.clone_active_computation_guard(),
                        ),
                    )
                    .await;
            }

            // after repairing, compare the fingerprints to see if we need to
            // recompute
            {
                // SAFETY: we have just repaired the callee, so the node info
                // must exist and immutable now.
                let callee_node_info = unsafe {
                    self.engine().get_node_info_unchecked(callee).await
                };

                let value_fingerprint_diff = callee_node_info
                    .value_fingerprint()
                    != forward_edge_observation
                        .0
                        .get(callee)
                        .unwrap()
                        .seen_value_fingerprint;

                // if any of the callee's value fingerprint differs, we need to
                // recompute
                if value_fingerprint_diff {
                    return RepairDecision::Recompute;
                }

                cleaned_edges.push(*callee);

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
