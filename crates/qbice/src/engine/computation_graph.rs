use std::sync::Arc;

use qbice_serialize::{Decode, Encode};
use qbice_stable_hash::Compact128;
use qbice_stable_type_id::Identifiable;
use qbice_storage::{
    kv_database::{Column, Normal},
    sieve::Sieve,
};

use crate::{ExecutionStyle, config::Config, query::QueryID};

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

struct ComputationGraph<C: Config> {
    forward_edges:
        Sieve<(QueryID, Arc<[QueryID]>), C::Database, C::BuildStableHasher>,
    node_info: Sieve<(QueryID, NodeInfo), C::Database, C::BuildStableHasher>,
    dirty_edge_set: Sieve<DirtySetColumn, C::Database, C::BuildStableHasher>,

    timestamp: Timestamp,
}
