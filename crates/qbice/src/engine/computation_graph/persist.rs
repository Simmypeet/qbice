use std::{
    borrow::Borrow, collections::HashMap, hash::Hash, marker::PhantomData,
    sync::Arc,
};

use dashmap::{DashMap, DashSet, setref::multiple::RefMulti};
use qbice_serialize::{Decode, Encode};
use qbice_stable_hash::Compact128;
use qbice_stable_type_id::Identifiable;
use qbice_storage::{
    intern::Interned,
    kv_database::{Column, KeyOfSet, KvDatabase, Normal, WriteTransaction},
    sieve::RemoveElementFromSet,
};
use rayon::iter::IntoParallelRefIterator;

pub(super) mod query_store;

use crate::{
    Engine, ExecutionStyle, Query,
    config::Config,
    engine::computation_graph::{
        ComputationGraph, QueryKind, QueryWithID, Sieve,
        lock::BackwardProjectionLockGuard,
        persist::query_store::{QueryColumn, QueryStore},
        tfc_achetype::TransitiveFirewallCallees,
        timestamp::Timestamp,
    },
    query::QueryID,
};

impl QueryKind {
    pub const fn is_projection(self) -> bool {
        matches!(self, Self::Executable(ExecutionStyle::Projection))
    }
}

pub struct BackwardEdge<C: Config>(Arc<DashSet<QueryID, C::BuildHasher>>);

impl<C: Config> BackwardEdge<C> {
    pub fn par_iter(&self) -> dashmap::rayon::set::Iter<'_, QueryID> {
        self.0.par_iter()
    }

    pub fn iter(
        &self,
    ) -> dashmap::iter_set::Iter<
        '_,
        QueryID,
        C::BuildHasher,
        DashMap<QueryID, (), C::BuildHasher>,
    > {
        self.0.iter()
    }
}

impl<C: Config> std::fmt::Debug for BackwardEdge<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BackwardEdge")
            .field("edges", &self.0)
            .finish_non_exhaustive()
    }
}

impl<C: Config> Clone for BackwardEdge<C> {
    fn clone(&self) -> Self { Self(self.0.clone()) }
}

impl<C: Config> Encode for BackwardEdge<C> {
    fn encode<E: qbice_serialize::Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        plugin: &qbice_serialize::Plugin,
        session: &mut qbice_serialize::session::Session,
    ) -> std::io::Result<()> {
        (*self.0).encode(encoder, plugin, session)
    }
}

impl<C: Config> Decode for BackwardEdge<C> {
    fn decode<D: qbice_serialize::Decoder + ?Sized>(
        decoder: &mut D,
        plugin: &qbice_serialize::Plugin,
        session: &mut qbice_serialize::session::Session,
    ) -> std::io::Result<Self> {
        let set = DashSet::decode(decoder, plugin, session)?;
        Ok(Self(Arc::new(set)))
    }
}

impl<C: Config> Default for BackwardEdge<C> {
    fn default() -> Self { Self(Arc::new(DashSet::default())) }
}

impl<C: Config> Extend<QueryID> for BackwardEdge<C> {
    fn extend<T: IntoIterator<Item = QueryID>>(&mut self, iter: T) {
        for query_id in iter {
            self.0.insert(query_id);
        }
    }
}

impl<C: Config> FromIterator<QueryID> for BackwardEdge<C> {
    fn from_iter<T: IntoIterator<Item = QueryID>>(iter: T) -> Self {
        Self(Arc::new(FromIterator::from_iter(iter)))
    }
}

impl<'x, C: Config> IntoIterator for &'x BackwardEdge<C> {
    type Item = RefMulti<'x, QueryID>;
    type IntoIter = dashmap::iter_set::Iter<
        'x,
        QueryID,
        C::BuildHasher,
        DashMap<QueryID, (), C::BuildHasher>,
    >;

    fn into_iter(self) -> Self::IntoIter { self.0.iter() }
}

impl<C: Config> RemoveElementFromSet for BackwardEdge<C> {
    type Element = QueryID;

    fn remove_element<Q: Hash + Eq + ?Sized>(&mut self, element: &Q) -> bool
    where
        Self::Element: Borrow<Q>,
    {
        self.0.remove(element).is_some()
    }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Identifiable,
)]
struct LastVerifiedColumn;

impl Column for LastVerifiedColumn {
    type Key = QueryID;

    type Value = Timestamp;

    type Mode = Normal;
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Identifiable,
)]
pub struct ForwardEdgeOrderColumn;

impl Column for ForwardEdgeOrderColumn {
    type Key = QueryID;

    type Value = Arc<[QueryID]>;

    type Mode = Normal;
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Identifiable,
)]
pub struct ForwardEdgeObservationColumn<C>(PhantomData<C>);

impl<C: Config> Column for ForwardEdgeObservationColumn<C> {
    type Key = QueryID;

    type Value = Arc<HashMap<QueryID, Observation, C::BuildHasher>>;

    type Mode = Normal;
}

pub type NodeInfoColumn = (QueryID, NodeInfo);

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Identifiable,
)]
struct PendingBackwardProjectionColumn;

impl Column for PendingBackwardProjectionColumn {
    type Key = QueryID;

    type Value = Timestamp;

    type Mode = Normal;
}

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
    fingerprint: Compact128,
    transitive_firewall_callees_fingerprint: Compact128,
    transitive_firewall_callees: Option<Interned<TransitiveFirewallCallees>>,
}

impl NodeInfo {
    pub const fn new(
        fingerprint: Compact128,
        transitive_firewall_callees_fingerprint: Compact128,
        transitive_firewall_callees: Option<
            Interned<TransitiveFirewallCallees>,
        >,
    ) -> Self {
        Self {
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
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Identifiable,
)]
pub struct BackwardEdgeColumn<C: Config>(PhantomData<C>);

impl<C: Config> Column for BackwardEdgeColumn<C> {
    type Key = QueryID;

    type Value = BackwardEdge<C>;

    type Mode = KeyOfSet<QueryID>;
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Identifiable,
)]
pub struct QueryKindColumn;

impl Column for QueryKindColumn {
    type Key = QueryID;

    type Value = QueryKind;

    type Mode = Normal;
}

pub struct Persist<C: Config> {
    last_verifieds: Sieve<LastVerifiedColumn, C>,
    forward_edge_orders: Sieve<ForwardEdgeOrderColumn, C>,
    forward_edge_observations: Sieve<ForwardEdgeObservationColumn<C>, C>,
    query_kinds: Sieve<QueryKindColumn, C>,
    node_info: Sieve<NodeInfoColumn, C>,
    dirty_edge_set: Sieve<DirtySetColumn, C>,
    backward_edges: Sieve<BackwardEdgeColumn<C>, C>,
    pending_backward_projections: Sieve<PendingBackwardProjectionColumn, C>,

    query_store: query_store::QueryStore<C>,
}

impl<C: Config> Persist<C> {
    pub fn new(db: Arc<C::Database>, shard_amount: usize) -> Self {
        Self {
            last_verifieds: Sieve::<_, C>::new(
                C::cache_entry_capacity(),
                shard_amount,
                db.clone(),
                C::BuildHasher::default(),
            ),
            forward_edge_orders: Sieve::<_, C>::new(
                C::cache_entry_capacity(),
                shard_amount,
                db.clone(),
                C::BuildHasher::default(),
            ),
            forward_edge_observations: Sieve::<_, C>::new(
                C::cache_entry_capacity(),
                shard_amount,
                db.clone(),
                C::BuildHasher::default(),
            ),
            node_info: Sieve::<_, C>::new(
                C::cache_entry_capacity(),
                shard_amount,
                db.clone(),
                C::BuildHasher::default(),
            ),
            query_kinds: Sieve::<_, C>::new(
                C::cache_entry_capacity(),
                shard_amount,
                db.clone(),
                C::BuildHasher::default(),
            ),
            dirty_edge_set: Sieve::<_, C>::new(
                C::cache_entry_capacity(),
                shard_amount,
                db.clone(),
                C::BuildHasher::default(),
            ),
            backward_edges: Sieve::<_, C>::new(
                C::cache_entry_capacity(),
                shard_amount,
                db.clone(),
                C::BuildHasher::default(),
            ),
            pending_backward_projections: Sieve::<_, C>::new(
                C::cache_entry_capacity(),
                shard_amount,
                db.clone(),
                C::BuildHasher::default(),
            ),
            query_store: QueryStore::new(
                C::cache_entry_capacity(),
                shard_amount,
                db,
                C::BuildHasher::default(),
            ),
        }
    }
}

#[derive(Debug, Encode, Decode)]
pub struct Observation {
    pub seen_value_fingerprint: Compact128,
    pub seen_transitive_firewall_callees_fingerprint: Compact128,
}

impl<C: Config> ComputationGraph<C> {
    pub fn get_forward_edges_order(
        &self,
        query_id: &QueryID,
    ) -> Option<Arc<[QueryID]>> {
        self.persist.forward_edge_orders.get_normal(query_id).map(|x| x.clone())
    }

    pub fn get_forward_edge_observations(
        &self,
        query_id: &QueryID,
    ) -> Option<Arc<HashMap<QueryID, Observation, C::BuildHasher>>> {
        self.persist
            .forward_edge_observations
            .get_normal(query_id)
            .map(|x| x.clone())
    }

    pub fn get_node_info(&self, query_id: &QueryID) -> Option<NodeInfo> {
        self.persist.node_info.get_normal(query_id).map(|x| x.clone())
    }

    pub fn get_query_kind(&self, query_id: &QueryID) -> Option<QueryKind> {
        self.persist.query_kinds.get_normal(query_id).map(|x| *x)
    }

    pub fn get_last_verified(&self, query_id: &QueryID) -> Option<Timestamp> {
        self.persist.last_verifieds.get_normal(query_id).map(|x| *x)
    }

    pub fn get_backward_edges(&self, query_id: &QueryID) -> BackwardEdge<C> {
        self.persist.backward_edges.get_set(query_id).clone()
    }

    pub fn get_value<Q: Query>(
        &self,
        query_input_hash_128: &Compact128,
    ) -> Option<Q::Value> {
        self.persist.query_store.get_value::<Q>(query_input_hash_128)
    }

    pub fn get_pending_backward_projection(
        &self,
        query_id: &QueryID,
    ) -> Option<Timestamp> {
        self.persist
            .pending_backward_projections
            .get_normal(query_id)
            .map(|x| *x)
    }
}

impl<C: Config> Engine<C> {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn set_computed<'s, Q: Query>(
        &'s self,
        query_id: &QueryWithID<'_, Q>,
        query_value: Q::Value,
        query_value_fingerprint: Option<Compact128>,
        query_kind: QueryKind,
        forward_edge_order: Arc<[QueryID]>,
        forward_edge_observations: HashMap<
            QueryID,
            Observation,
            C::BuildHasher,
        >,
        tfc_achetype: Option<Interned<TransitiveFirewallCallees>>,
        has_pending_backward_projection: bool,
        continuting_tx: Option<
            <C::Database as KvDatabase>::WriteTransaction<'s>,
        >,
    ) {
        let query_value_fingerprint =
            query_value_fingerprint.unwrap_or_else(|| self.hash(&query_value));
        let transitive_firewall_callees_fingerprint = self.hash(&tfc_achetype);

        let forward_edge_observations = Arc::new(forward_edge_observations);

        let query_entry =
            query_store::QueryEntry::new(query_id.query.clone(), query_value);
        let node_info = NodeInfo::new(
            query_value_fingerprint,
            transitive_firewall_callees_fingerprint,
            tfc_achetype,
        );
        let current_timestamp =
            self.computation_graph.timestamp_manager.get_current();

        let existing_forward_edges =
            self.computation_graph.get_forward_edges_order(&query_id.id);

        {
            let tx = continuting_tx
                .unwrap_or_else(|| self.database.write_transaction());

            // mark pending backward projection if needed
            if has_pending_backward_projection {
                tx.put::<PendingBackwardProjectionColumn>(
                    &query_id.id,
                    &current_timestamp,
                );
            }

            tx.put::<QueryKindColumn>(&query_id.id, &query_kind);

            // remove prior backward edges
            if let Some(forward_edges) = &existing_forward_edges {
                for edge in forward_edges.iter() {
                    tx.delete_member::<BackwardEdgeColumn<C>>(
                        edge,
                        &query_id.id,
                    );
                }
            }

            tx.put::<LastVerifiedColumn>(&query_id.id, &current_timestamp);

            for edge in forward_edge_order.iter() {
                tx.insert_member::<BackwardEdgeColumn<C>>(edge, &query_id.id);
            }

            tx.put::<ForwardEdgeOrderColumn>(&query_id.id, &forward_edge_order);

            tx.put::<ForwardEdgeObservationColumn<C>>(
                &query_id.id,
                &forward_edge_observations,
            );

            tx.put::<NodeInfoColumn>(&query_id.id, &node_info);

            tx.put::<QueryColumn<Q>>(
                &query_id.id.hash_128().into(),
                &query_entry,
            );

            tx.commit();
        }

        {
            // remove prior backward edges
            if let Some(forward_edges) = &existing_forward_edges {
                for edge in forward_edges.iter() {
                    self.computation_graph
                        .persist
                        .backward_edges
                        .remove_set_element(edge, &query_id.id);
                }
            }

            // set pending backward projection if needed
            if has_pending_backward_projection {
                self.computation_graph
                    .persist
                    .pending_backward_projections
                    .put(query_id.id, Some(current_timestamp));
            }

            self.computation_graph
                .persist
                .query_kinds
                .put(query_id.id, Some(query_kind));

            self.computation_graph
                .persist
                .last_verifieds
                .put(query_id.id, Some(current_timestamp));

            for edge in forward_edge_order.iter() {
                self.computation_graph
                    .persist
                    .backward_edges
                    .insert_set_element(edge, std::iter::once(query_id.id));
            }

            self.computation_graph
                .persist
                .forward_edge_orders
                .put(query_id.id, Some(forward_edge_order));

            self.computation_graph
                .persist
                .forward_edge_observations
                .put(query_id.id, Some(forward_edge_observations));

            self.computation_graph
                .persist
                .node_info
                .put(query_id.id, Some(node_info));

            self.computation_graph
                .persist
                .query_store
                .insert::<Q>(query_id.id.hash_128().into(), query_entry);
        }
    }

    pub(super) fn set_computed_input<Q: Query>(
        &self,
        query: Q,
        query_hash_128: Compact128,
        query_value: Q::Value,
        query_value_fingerprint: Compact128,
        tx: &<C::Database as KvDatabase>::WriteTransaction<'_>,
    ) {
        let query_id = QueryID::new::<Q>(query_hash_128);

        // if have an existing forward edges, unwire the backward edges
        let existing_forward_edges =
            self.computation_graph.get_forward_edges_order(&query_id);

        let empty_forward_edges = Arc::from([]);
        let empty_forward_edge_observations =
            Arc::new(HashMap::with_hasher(C::BuildHasher::default()));

        let transitive_firewall_callees = None;
        let transitive_firewall_callees_fingerprint =
            self.hash(&transitive_firewall_callees);

        let node_info = NodeInfo::new(
            query_value_fingerprint,
            transitive_firewall_callees_fingerprint,
            transitive_firewall_callees,
        );

        let timestamp = self.computation_graph.timestamp_manager.get_current();

        let query_entry = query_store::QueryEntry::new(query, query_value);

        {
            // remove prior backward edges
            if let Some(forward_edges) = &existing_forward_edges {
                for edge in forward_edges.iter() {
                    tx.delete_member::<BackwardEdgeColumn<C>>(edge, &query_id);
                }
            }

            tx.put::<QueryKindColumn>(&query_id, &QueryKind::Input);

            tx.put::<LastVerifiedColumn>(&query_id, &timestamp);

            tx.put::<ForwardEdgeOrderColumn>(&query_id, &empty_forward_edges);

            tx.put::<ForwardEdgeObservationColumn<C>>(
                &query_id,
                &empty_forward_edge_observations,
            );

            tx.put::<NodeInfoColumn>(&query_id, &node_info);

            tx.put::<QueryColumn<Q>>(&query_id.hash_128().into(), &query_entry);
        }

        {
            if let Some(forward_edges) = existing_forward_edges {
                for edge in forward_edges.iter() {
                    self.computation_graph
                        .persist
                        .backward_edges
                        .remove_set_element(edge, &query_id);
                }
            }

            self.computation_graph
                .persist
                .query_kinds
                .put(query_id, Some(QueryKind::Input));

            self.computation_graph
                .persist
                .last_verifieds
                .put(query_id, Some(timestamp));

            self.computation_graph
                .persist
                .forward_edge_orders
                .put(query_id, Some(empty_forward_edges));

            self.computation_graph
                .persist
                .forward_edge_observations
                .put(query_id, Some(empty_forward_edge_observations));

            self.computation_graph
                .persist
                .node_info
                .put(query_id, Some(node_info));

            self.computation_graph
                .persist
                .query_store
                .insert::<Q>(query_id.hash_128().into(), query_entry);
        }
    }

    pub(super) fn mark_dirty_forward_edge(
        &self,
        from: QueryID,
        to: QueryID,
        tx: &<C::Database as KvDatabase>::WriteTransaction<'_>,
    ) {
        let edge = Edge { from, to };

        tx.put::<DirtySetColumn>(&edge, &());

        self.computation_graph.persist.dirty_edge_set.put(edge, Some(()));
        self.computation_graph.add_dirtied_edge_count();
    }

    #[allow(clippy::option_option)]
    pub(super) fn clean_query(
        &self,
        query_id: &QueryID,
        clean_edges: &[QueryID],
        new_tfc: Option<Option<Interned<TransitiveFirewallCallees>>>,
    ) {
        let current_timestamp =
            self.computation_graph.timestamp_manager.get_current();

        let new_node_info = new_tfc.map(|x| {
            let mut current_node_info =
                self.computation_graph.get_node_info(query_id).unwrap();

            current_node_info.transitive_firewall_callees = x;
            current_node_info.transitive_firewall_callees_fingerprint =
                self.hash(&current_node_info.transitive_firewall_callees);

            current_node_info
        });

        {
            let tx = self.database.write_transaction();

            for callee in clean_edges.iter().copied() {
                let edge = Edge { from: *query_id, to: callee };

                tx.delete::<DirtySetColumn>(&edge);
            }

            tx.put::<LastVerifiedColumn>(query_id, &current_timestamp);

            if let Some(node_info) = &new_node_info {
                tx.put::<NodeInfoColumn>(query_id, node_info);
            }

            tx.commit();
        }

        {
            for callee in clean_edges.iter().copied() {
                let edge = Edge { from: *query_id, to: callee };

                self.computation_graph.persist.dirty_edge_set.put(edge, None);
            }

            if let Some(node_info) = new_node_info {
                self.computation_graph
                    .persist
                    .node_info
                    .put(*query_id, Some(node_info));
            }

            self.computation_graph
                .persist
                .last_verifieds
                .put(*query_id, Some(current_timestamp));
        }
    }

    pub(super) fn is_edge_dirty(&self, from: QueryID, to: QueryID) -> bool {
        self.computation_graph
            .persist
            .dirty_edge_set
            .get_normal(&Edge { from, to })
            .is_some()
    }

    pub(super) fn get_query_input<Q: Query>(
        &self,
        query_id: &Compact128,
    ) -> Option<Q> {
        self.computation_graph.persist.query_store.get_input::<Q>(query_id)
    }

    pub(super) fn done_backward_projection(
        &self,
        query_id: &QueryID,
        backward_projection_lock_guard: BackwardProjectionLockGuard<'_, C>,
    ) {
        let tx = self.database.write_transaction();

        tx.delete::<PendingBackwardProjectionColumn>(query_id);

        tx.commit();

        self.computation_graph
            .persist
            .pending_backward_projections
            .put(*query_id, None);

        backward_projection_lock_guard.done();
    }
}
