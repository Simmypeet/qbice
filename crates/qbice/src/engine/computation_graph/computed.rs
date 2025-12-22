use std::{collections::HashSet, sync::Arc};

use qbice_serialize::{Decode, Encode};
use qbice_stable_hash::Compact128;
use qbice_stable_type_id::Identifiable;
use qbice_storage::{
    intern::Interned,
    kv_database::{Column, KeyOfSet, Normal},
};

use crate::{
    ExecutionStyle,
    config::Config,
    engine::computation_graph::{
        ComputationGraph, QueryKind, Sieve, Timestamp,
        TransitiveFirewallCallees,
    },
    query::QueryID,
};

impl QueryKind {
    pub fn is_projection(&self) -> bool {
        matches!(self, QueryKind::Executable(ExecutionStyle::Projection))
    }
}

type ForwardEdgeColumn = (QueryID, Vec<QueryID>);
type NodeInfoColumn = (QueryID, NodeInfo);

#[derive(Identifiable)]
struct DirtySetColumn;

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Encode, Decode,
)]
struct Edge {
    from: QueryID,
    to: QueryID,
}

impl Column for DirtySetColumn {
    type Key = Edge;
    type Value = ();
    type Mode = Normal;
}

#[derive(Debug, Clone, PartialEq, Eq, Encode, Decode, Identifiable)]
pub struct NodeInfo {
    last_verified: Timestamp,
    query_kind: QueryKind,
    fingerprint: Compact128,
    transitive_firewall_callees_fingerprint: Compact128,
    transitive_firewall_callees: Interned<TransitiveFirewallCallees>,
}

impl NodeInfo {
    pub fn last_verified(&self) -> Timestamp { self.last_verified }

    pub fn value_fingerprint(&self) -> Compact128 { self.fingerprint }

    pub fn transitive_firewall_callees_fingerprint(&self) -> Compact128 {
        self.transitive_firewall_callees_fingerprint
    }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Identifiable,
)]
struct BackwardEdgeColumn;

impl Column for BackwardEdgeColumn {
    type Key = QueryID;

    type Value = HashSet<QueryID>;

    type Mode = KeyOfSet<QueryID>;
}

pub struct Computed<C: Config> {
    forward_edges: Sieve<(QueryID, Arc<[QueryID]>), C>,
    node_info: Sieve<(QueryID, NodeInfo), C>,
    dirty_edge_set: Sieve<DirtySetColumn, C>,
    backward_edges: Sieve<BackwardEdgeColumn, C>,
}

impl<C: Config> Computed<C> {
    pub fn new(
        db: Arc<C::Database>,
        shard_amount: usize,
        build_hasher: C::BuildHasher,
    ) -> Self {
        const CAPACITY: usize = 10_000;

        Self {
            forward_edges: Sieve::<_, C>::new(
                CAPACITY,
                shard_amount,
                db.clone(),
                build_hasher.clone(),
            ),
            node_info: Sieve::<_, C>::new(
                CAPACITY,
                shard_amount,
                db.clone(),
                build_hasher.clone(),
            ),
            dirty_edge_set: Sieve::<_, C>::new(
                CAPACITY,
                shard_amount,
                db.clone(),
                build_hasher.clone(),
            ),
            backward_edges: Sieve::<_, C>::new(
                CAPACITY,
                shard_amount,
                db.clone(),
                build_hasher.clone(),
            ),
        }
    }
}

impl<C: Config> ComputationGraph<C> {
    pub fn forward_edges(&self) -> &Sieve<(QueryID, Arc<[QueryID]>), C> {
        &self.computed.forward_edges
    }

    pub fn node_info(&self) -> &Sieve<(QueryID, NodeInfo), C> {
        &self.computed.node_info
    }

    pub fn backward_edges(&self) -> &Sieve<BackwardEdgeColumn, C> {
        &self.computed.backward_edges
    }

    pub fn dirty_edge_set(&self) -> &Sieve<DirtySetColumn, C> {
        &self.computed.dirty_edge_set
    }
}
