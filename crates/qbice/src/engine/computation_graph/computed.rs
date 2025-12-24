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
        ComputationGraph, QueryKind, Sieve,
        tfc_achetype::TransitiveFirewallCallees, timestamp::Timestamp,
    },
    query::QueryID,
};

impl QueryKind {
    pub const fn is_projection(self) -> bool {
        matches!(self, Self::Executable(ExecutionStyle::Projection))
    }
}

pub type LastVerifiedColumn = (QueryID, Timestamp);
pub type ForwardEdgeColumn = (QueryID, Arc<[QueryID]>);
pub type NodeInfoColumn = (QueryID, NodeInfo);

#[derive(Identifiable)]
pub struct DirtySetColumn;

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Encode, Decode,
)]
pub struct Edge {
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
    query_kind: QueryKind,
    fingerprint: Compact128,
    transitive_firewall_callees_fingerprint: Compact128,
    transitive_firewall_callees: Option<Interned<TransitiveFirewallCallees>>,
}

impl NodeInfo {
    pub const fn new(
        query_kind: QueryKind,
        fingerprint: Compact128,
        transitive_firewall_callees_fingerprint: Compact128,
        transitive_firewall_callees: Option<
            Interned<TransitiveFirewallCallees>,
        >,
    ) -> Self {
        Self {
            query_kind,
            fingerprint,
            transitive_firewall_callees_fingerprint,
            transitive_firewall_callees,
        }
    }

    pub const fn value_fingerprint(&self) -> Compact128 { self.fingerprint }

    pub const fn transitive_firewall_callees_fingerprint(&self) -> Compact128 {
        self.transitive_firewall_callees_fingerprint
    }

    pub const fn transitive_firewall_callees(
        &self,
    ) -> Option<&Interned<TransitiveFirewallCallees>> {
        self.transitive_firewall_callees.as_ref()
    }

    pub const fn query_kind(&self) -> QueryKind { self.query_kind }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Identifiable,
)]
pub struct BackwardEdgeColumn;

impl Column for BackwardEdgeColumn {
    type Key = QueryID;

    type Value = HashSet<QueryID>;

    type Mode = KeyOfSet<QueryID>;
}

pub struct Computed<C: Config> {
    last_verifieds: Sieve<LastVerifiedColumn, C>,
    forward_edges: Sieve<ForwardEdgeColumn, C>,
    node_info: Sieve<NodeInfoColumn, C>,
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
            last_verifieds: Sieve::<_, C>::new(
                CAPACITY,
                shard_amount,
                db.clone(),
                build_hasher.clone(),
            ),
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
                db,
                build_hasher,
            ),
        }
    }
}

impl<C: Config> ComputationGraph<C> {
    pub const fn forward_edges(&self) -> &Sieve<(QueryID, Arc<[QueryID]>), C> {
        &self.computed.forward_edges
    }

    pub const fn node_info(&self) -> &Sieve<(QueryID, NodeInfo), C> {
        &self.computed.node_info
    }

    pub const fn last_verifieds(&self) -> &Sieve<(QueryID, Timestamp), C> {
        &self.computed.last_verifieds
    }

    pub const fn backward_edges(&self) -> &Sieve<BackwardEdgeColumn, C> {
        &self.computed.backward_edges
    }

    pub const fn dirty_edge_set(&self) -> &Sieve<DirtySetColumn, C> {
        &self.computed.dirty_edge_set
    }
}
