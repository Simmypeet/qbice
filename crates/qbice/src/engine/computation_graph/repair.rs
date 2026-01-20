use std::{borrow::Cow, ops::Not, pin::Pin, sync::Arc};

use qbice_stable_hash::Compact128;

use crate::{
    Engine, Query,
    config::Config,
    engine::computation_graph::{
        QueryWithID,
        caller::{CallerInformation, CallerKind, CallerReason, QueryCaller},
        lock::ComputingLockGuard,
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

impl<C: Config> Engine<C> {
    pub(super) async fn recompute_decision_based_on_forward_edges<Q: Query>(
        self: &Arc<Self>,
        query: &QueryWithID<'_, Q>,
        caller_information: &CallerInformation,
    ) -> RepairDecision {
        let mut repair_transitive_firewall_callees = false;
        let mut cleaned_edges = Vec::new();

        let forward_edges = self
            .get_forward_edges_order(query.id, caller_information)
            .await
            .unwrap();
        let forward_edge_observations = self
            .get_forward_edge_observations(query.id, caller_information)
            .await
            .unwrap();

        for callee in forward_edges.iter() {
            // skip if not dirty
            if !self.is_edge_dirty(query.id, *callee) {
                continue;
            }

            let kind =
                self.get_query_kind(*callee, caller_information).await.unwrap();

            // NOTE: if the callee is an input (explicitly set), it's impossible
            // to try to repair it, so we'll skip repairing and directly
            // compare the fingerprint.
            if !kind.is_input() {
                // recursively repair the callee first
                let entry = self
                    .executor_registry
                    .get_executor_entry_by_type_id(&callee.stable_type_id());

                let _ = entry
                    .repair_query_from_query_id(
                        self,
                        callee.compact_hash_128(),
                        CallerInformation::new(
                            CallerKind::Query(QueryCaller::new(
                                query.id,
                                CallerReason::Repair,
                            )),
                            caller_information.timestamp(),
                        ),
                    )
                    .await;
            }

            // after repairing, compare the fingerprints to see if we need to
            // recompute
            {
                let callee_node_info = self
                    .get_node_info(*callee, caller_information)
                    .await
                    .expect(
                        "callee node info should exist when forward edge \
                         exists",
                    );

                let value_fingerprint_diff = callee_node_info
                    .value_fingerprint()
                    != forward_edge_observations
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
                        != forward_edge_observations
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

    pub(super) async fn repair_transitive_firewall_callees<Q: Query>(
        self: &Arc<Self>,
        query: &QueryWithID<'_, Q>,
        caller_information: &CallerInformation,
    ) {
        let node_info =
            self.get_node_info(query.id, caller_information).await.unwrap();

        let is_current_query_projection = self
            .get_query_kind(query.id, caller_information)
            .await
            .unwrap()
            .is_projection();

        let tfcs = node_info.transitive_firewall_callees();

        let tfcs = tfcs
            .into_iter()
            .flat_map(|x| x.iter())
            .copied()
            .collect::<Vec<_>>();

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
            let engine = Arc::clone(self);

            handles.push(tokio::spawn(async move {
                for tfc in tfc_chunk {
                    let executor = engine
                        .executor_registry
                        .get_executor_entry_by_type_id(&tfc.stable_type_id());

                    let _ = executor
                        .repair_query_from_query_id(
                            &engine,
                            tfc.compact_hash_128(),
                            CallerInformation::new(
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

    pub(super) async fn should_recompute_query<'x, Q: Query>(
        self: &Arc<Self>,
        query: &QueryWithID<'_, Q>,
        caller_information: &CallerInformation,
        lock_guard: ComputingLockGuard<'x, C>,
    ) -> Option<ComputingLockGuard<'x, C>> {
        // if the caller is backward projection propagation, we always
        // recompute since the projection query have already told us
        // that the value is required to be recomputed.
        if matches!(
            caller_information.kind(),
            CallerKind::BackwardProjectionPropagation
        ) {
            return Some(lock_guard);
        }

        // continue normal path ...

        // repair transitive firewall callees first before deciding whether to
        // recompute, since the transitive firewall callees might affect the
        // decision by propagating dirtiness.
        if matches!(
            caller_information.kind(),
            CallerKind::User | CallerKind::RepairFirewall { .. }
        ) {
            self.repair_transitive_firewall_callees(query, caller_information)
                .await;
        }

        let recompute = self
            .recompute_decision_based_on_forward_edges(
                query,
                caller_information,
            )
            .await;

        let (repair_transitive_firewall_callees, cleaned_edges) =
            match recompute {
                RepairDecision::Recompute => return Some(lock_guard),
                RepairDecision::Clean {
                    repair_transitive_firewall_callees,
                    cleaned_edges,
                } => (repair_transitive_firewall_callees, cleaned_edges),
            };

        if repair_transitive_firewall_callees.not() {
            self.computing_lock_to_clean_query(
                query.id,
                &cleaned_edges,
                None,
                caller_information,
                lock_guard,
            )
            .await;
        } else {
            let forward_edges = self
                .get_forward_edges_order(query.id, caller_information)
                .await
                .unwrap();

            // repair all callees
            for callee in forward_edges.iter() {
                let entry = self
                    .executor_registry
                    .get_executor_entry_by_type_id(&callee.stable_type_id());

                let _ = entry
                    .repair_query_from_query_id(
                        self,
                        callee.compact_hash_128(),
                        CallerInformation::new(
                            CallerKind::Query(QueryCaller::new(
                                query.id,
                                CallerReason::Repair,
                            )),
                            caller_information.timestamp(),
                        ),
                    )
                    .await;
            }

            let mut unioning_tfcs = Vec::new();

            for x in forward_edges.iter() {
                let kind =
                    self.get_query_kind(*x, caller_information).await.unwrap();

                if kind.is_firewall() {
                    unioning_tfcs.push(Cow::Owned(self.new_singleton_tfc(*x)));
                } else {
                    let callee_info = self
                        .get_node_info(*x, caller_information)
                        .await
                        .unwrap();
                    if let Some(tfc) = callee_info.transitive_firewall_callees()
                    {
                        unioning_tfcs.push(Cow::Owned(tfc.clone()));
                    }
                }
            }

            let new_tfc = self.union_tfcs(unioning_tfcs);

            self.computing_lock_to_clean_query(
                query.id,
                &cleaned_edges,
                Some(new_tfc),
                caller_information,
                lock_guard,
            )
            .await;
        }

        None
    }

    pub(super) async fn repair_query<Q: Query>(
        self: &Arc<Self>,
        query: &QueryWithID<'_, Q>,
        caller_information: &CallerInformation,
        lock_guard: ComputingLockGuard<'_, C>,
    ) {
        let Some(lock_guard) = self
            .should_recompute_query(query, caller_information, lock_guard)
            .await
        else {
            return;
        };

        // recompute the query
        self.execute_query(
            query,
            caller_information,
            super::slow_path::ExecuteQueryFor::RecomputeQuery,
            lock_guard,
        )
        .await;
    }

    pub(crate) fn repair_query_from_query_id<'x, Q: Query>(
        self: &'x Arc<Self>,
        query_id: Compact128,
        called_from: CallerInformation,
    ) -> Pin<Box<dyn Future<Output = Result<(), CyclicError>> + Send + 'x>>
    {
        Box::pin(async move {
            let query_input = self
                .get_query_input::<Q>(query_id, &called_from)
                .await
                .unwrap();

            let query_for = QueryWithID {
                id: QueryID::new::<Q>(query_id),
                query: &query_input,
            };

            self.query_for(&query_for, &called_from).await.map(|_| ())
        })
    }
}
