use std::{borrow::Cow, ops::Not, pin::Pin, sync::Arc};

use qbice_stable_hash::Compact128;

use crate::{
    Engine, Query,
    config::Config,
    engine::computation_graph::{
        QueryWithID,
        caller::{CallerInformation, CallerReason, QueryCaller},
        computing_lock::ComputingLockGuard,
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
    ) -> RepairDecision {
        let mut repair_transitive_firewall_callees = false;
        let mut cleaned_edges = Vec::new();

        let forward_edges =
            self.computation_graph.get_forward_edges(&query.id).unwrap();

        for callee in &forward_edges.callee_order {
            // skip if not dirty
            if !self.is_edge_dirty(query.id, *callee) {
                continue;
            }

            let callee_node_info =
                self.computation_graph.get_node_info(callee).expect(
                    "callee node info should exist when forward edge exists",
                );

            // NOTE: if the callee is an input (explicitly set), it's impossible
            // to try to repair it, so we'll skip repairing and directly
            // compare the fingerprint.
            if !callee_node_info.query_kind().is_input() {
                // recursively repair the callee first
                let entry = self
                    .executor_registry
                    .get_executor_entry_by_type_id(&callee.stable_type_id());

                let _ = entry
                    .repair_query_from_query_id(
                        self,
                        callee.compact_hash_128(),
                        query.id,
                    )
                    .await;
            }

            // after repairing, compare the fingerprints to see if we need to
            // recompute
            {
                let callee_node_info =
                    self.computation_graph.get_node_info(callee).expect(
                        "callee node info should exist when forward edge \
                         exists",
                    );

                let value_fingerprint_diff = callee_node_info
                    .value_fingerprint()
                    != forward_edges
                        .callee_observations
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
                if !callee_node_info.query_kind().is_firewall() {
                    let tfc_fingerprint_diff = callee_node_info
                        .transitive_firewall_callees_fingerprint()
                        != forward_edges
                            .callee_observations
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

    pub(super) async fn should_recompute_query<'x, Q: Query>(
        self: &Arc<Self>,
        query: &QueryWithID<'_, Q>,
        caller_information: &CallerInformation,
        lock_guard: ComputingLockGuard<'x>,
    ) -> Option<ComputingLockGuard<'x>> {
        // if the caller is backward projection propagation, we always
        // recompute since the projection query have already told us
        // that the value is required to be recomputed.
        if *caller_information
            == CallerInformation::BackwardProjectionPropagation
        {
            return Some(lock_guard);
        }

        // continue normal path ...

        // repair transitive firewall callees first before deciding whether to
        // recompute, since the transitive firewall callees might affect the
        // decision by propagating dirtiness.
        // self.repair_transitive_firewall_callees(
        //     &query_id.id,
        //     &mut lock_computing,
        // )
        // .await;

        let recompute =
            self.recompute_decision_based_on_forward_edges(query).await;

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
                &query.id,
                &cleaned_edges,
                None,
                lock_guard,
            );
        } else {
            let forward_edges =
                self.computation_graph.get_forward_edges(&query.id).unwrap();

            // repair all callees
            for callee in &forward_edges.callee_order {
                let entry = self
                    .executor_registry
                    .get_executor_entry_by_type_id(&callee.stable_type_id());

                let _ = entry
                    .repair_query_from_query_id(
                        self,
                        callee.compact_hash_128(),
                        query.id,
                    )
                    .await;
            }

            let new_tfc = self.union_tfcs(
                forward_edges.callee_order.iter().filter_map(|x| {
                    let callee_info =
                        self.computation_graph.get_node_info(x).unwrap();

                    if callee_info.query_kind().is_firewall() {
                        Some(Cow::Owned(self.new_singleton_tfc(*x)))
                    } else {
                        callee_info
                            .transitive_firewall_callees()
                            .map(|x| Cow::Owned(x.clone()))
                    }
                }),
            );

            self.computing_lock_to_clean_query(
                &query.id,
                &cleaned_edges,
                Some(new_tfc),
                lock_guard,
            );
        }

        None
    }

    pub(super) async fn repair_query<Q: Query>(
        self: &Arc<Self>,
        query: &QueryWithID<'_, Q>,
        caller_information: &CallerInformation,
        lock_guard: ComputingLockGuard<'_>,
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
        called_from: QueryID,
    ) -> Pin<Box<dyn Future<Output = Result<(), CyclicError>> + Send + 'x>>
    {
        Box::pin(async move {
            let query_input = self.get_query_input::<Q>(&query_id).unwrap();
            let query_for = QueryWithID {
                id: QueryID::new::<Q>(query_id),
                query: &query_input,
            };

            self.query_for(
                &query_for,
                &CallerInformation::Query(QueryCaller::new(
                    called_from,
                    CallerReason::Repair,
                )),
            )
            .await
            .map(|_| ())
        })
    }
}
