use std::{collections::HashSet, sync::Arc};

use qbice_serialize::{Decode, Encode};
use qbice_stable_hash::Compact128;
use qbice_stable_type_id::Identifiable;
use qbice_storage::kv_database::{Column, KeyOfSet, Normal};

use crate::{ExecutionStyle, config::Config, query::QueryID};

type Sieve<Col, Con> = qbice_storage::sieve::Sieve<
    Col,
    <Con as Config>::Database,
    <Con as Config>::BuildStableHasher,
>;

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Encode, Decode,
)]
struct Timestamp(u64);

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Encode, Decode,
)]
enum QueryKind {
    Input,
    Executable(ExecutionStyle),
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

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Encode,
    Decode,
    Identifiable,
)]
struct NodeInfo {
    last_verified: Timestamp,
    query_kind: QueryKind,
    fingerprint: Compact128,
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

struct ComputationGraph<C: Config> {
    forward_edges: Sieve<(QueryID, Arc<[QueryID]>), C>,
    node_info: Sieve<(QueryID, NodeInfo), C>,
    dirty_edge_set: Sieve<DirtySetColumn, C>,
    backward_edges: Sieve<BackwardEdgeColumn, C>,

    database: Arc<C::Database>,
    timestamp: Timestamp,
}

impl<C: Config> ComputationGraph<C> {
    pub fn new_with(
        db: Arc<<C as Config>::Database>,
        shard_amount: usize,
        builder_stable_hasher: C::BuildStableHasher,
    ) -> Self {
        const CAPACITY: usize = 10_000;

        Self {
            forward_edges: Sieve::<_, C>::new(
                CAPACITY,
                shard_amount,
                db.clone(),
                builder_stable_hasher.clone(),
            ),
            node_info: Sieve::<_, C>::new(
                CAPACITY,
                shard_amount,
                db.clone(),
                builder_stable_hasher.clone(),
            ),
            dirty_edge_set: Sieve::<_, C>::new(
                CAPACITY,
                shard_amount,
                db.clone(),
                builder_stable_hasher.clone(),
            ),
            backward_edges: Sieve::<_, C>::new(
                CAPACITY,
                shard_amount,
                db.clone(),
                builder_stable_hasher,
            ),

            database: db,
            timestamp: Timestamp(0),
        }
    }
}
