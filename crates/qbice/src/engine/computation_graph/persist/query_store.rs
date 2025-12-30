use std::{
    any::{Any, TypeId},
    marker::PhantomData,
    sync::Arc,
};

use dashmap::{DashMap, Entry, mapref::one::MappedRef};
use qbice_serialize::{Decode, Encode, Plugin, session::Session};
use qbice_stable_hash::Compact128;
use qbice_stable_type_id::Identifiable;
use qbice_storage::kv_database::{Column, Normal};

use crate::{Engine, Query, config::Config, engine::computation_graph::Sieve};

pub struct QueryStore<C: Config> {
    map: DashMap<TypeId, Box<dyn Any + Send + Sync>>,

    total_capacity: usize,
    shard_amount: usize,
    backing_db: Arc<C::Database>,
    hasher_builder: C::BuildHasher,

    _marker: PhantomData<C>,
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Identifiable,
)]
pub struct QueryColumn<Q>(PhantomData<Q>);

impl<Q: Query> Column for QueryColumn<Q>
where
    QueryEntry<Q>: Encode + Decode,
{
    type Key = Compact128;

    type Value = QueryEntry<Q>;

    type Mode = Normal;
}

impl<Q: Query> Encode for QueryEntry<Q> {
    fn encode<E: qbice_serialize::Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        plugin: &Plugin,
        session: &mut Session,
    ) -> std::io::Result<()> {
        self.original_query_input.encode(encoder, plugin, session)?;
        self.query_result.encode(encoder, plugin, session)?;

        Ok(())
    }
}

impl<Q: Query> Decode for QueryEntry<Q> {
    fn decode<D: qbice_serialize::Decoder + ?Sized>(
        decoder: &mut D,
        plugin: &Plugin,
        session: &mut Session,
    ) -> std::io::Result<Self> {
        Ok(Self {
            original_query_input: Q::decode(decoder, plugin, session)?,
            query_result: Q::Value::decode(decoder, plugin, session)?,
        })
    }
}

#[derive(Debug, Clone)]
pub struct QueryEntry<Q: Query> {
    original_query_input: Q,
    query_result: Q::Value,
}

impl<Q: Query> QueryEntry<Q> {
    pub const fn new(original_query_input: Q, query_result: Q::Value) -> Self {
        Self { original_query_input, query_result }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct QueryDebug {
    pub r#type: &'static str,
    pub input: String,
    pub output: String,
}

impl<C: Config> QueryStore<C> {
    pub(super) fn new(
        total_capacity: usize,
        shard_amount: usize,
        backing_db: Arc<C::Database>,
        hasher_builder: C::BuildHasher,
    ) -> Self {
        Self {
            map: DashMap::new(),
            total_capacity,
            shard_amount,
            backing_db,
            hasher_builder,
            _marker: PhantomData,
        }
    }

    fn get_sieve<Q: Query>(
        &self,
    ) -> MappedRef<
        '_,
        TypeId,
        Box<dyn Any + Send + Sync>,
        Sieve<QueryColumn<Q>, C>,
    > {
        loop {
            if let Some(entry) = self.map.get(&TypeId::of::<Q>()) {
                return entry.map(|e| {
                    e.downcast_ref::<Sieve<QueryColumn<Q>, C>>().unwrap()
                });
            }

            if let Entry::Vacant(entry) = self.map.entry(TypeId::of::<Q>()) {
                entry.insert(Box::new(Sieve::<QueryColumn<Q>, C>::new(
                    self.total_capacity,
                    self.shard_amount,
                    self.backing_db.clone(),
                    self.hasher_builder.clone(),
                )));
            }
        }
    }

    pub(super) fn insert<Q: Query>(
        &self,
        query_input_hash_128: Compact128,
        query_entry: QueryEntry<Q>,
    ) {
        let entry = self.get_sieve::<Q>();

        entry.put(query_input_hash_128, Some(query_entry));
    }

    pub(super) fn get_input<Q: Query>(
        &self,
        query_input_hash_128: &Compact128,
    ) -> Option<Q> {
        let entry = self.get_sieve::<Q>();

        entry
            .get_normal(query_input_hash_128)
            .map(|x| x.original_query_input.clone())
    }

    pub(super) fn get_value<Q: Query>(
        &self,
        query_input_hash_128: &Compact128,
    ) -> Option<Q::Value> {
        let entry = self.get_sieve::<Q>();

        entry.get_normal(query_input_hash_128).map(|x| x.query_result.clone())
    }
}

impl<C: Config> Engine<C> {
    pub(crate) fn get_query_debug<Q: Query>(
        &self,
        query_input_hash_128: &Compact128,
    ) -> Option<QueryDebug> {
        let entry = self.computation_graph.persist.query_store.get_sieve::<Q>();

        entry.get_normal(query_input_hash_128).map(|x| QueryDebug {
            r#type: std::any::type_name::<Q>(),
            input: format!("{:?}", x.original_query_input),
            output: format!("{:?}", x.query_result),
        })
    }
}
