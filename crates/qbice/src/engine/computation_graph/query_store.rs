use std::{
    any::{Any, TypeId},
    marker::PhantomData,
    sync::Arc,
};

use dashmap::DashMap;
use qbice_serialize::{Decode, Encode, Plugin, session::Session};
use qbice_stable_hash::Compact128;
use qbice_stable_type_id::Identifiable;
use qbice_storage::kv_database::{Column, Normal};

use crate::{Query, config::Config, engine::computation_graph::Sieve};

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

impl<C: Config> QueryStore<C> {
    pub fn new(
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

    pub fn insert<Q: Query>(
        &mut self,
        query_input_hash_128: Compact128,
        original_query_input: Q,
        query_result: Q::Value,
    ) {
        let mut entry =
            self.map.entry(TypeId::of::<Q>()).or_insert_with(|| {
                Box::new(Sieve::<QueryColumn<Q>, C>::new(
                    self.total_capacity,
                    self.shard_amount,
                    self.backing_db.clone(),
                    self.hasher_builder.clone(),
                ))
            });

        let entry =
            (**entry).downcast_mut::<Sieve<QueryColumn<Q>, C>>().unwrap();

        entry.put(
            query_input_hash_128,
            Some(QueryEntry { original_query_input, query_result }),
        );
    }

    pub async fn get_value<Q: Query>(
        &self,
        query_input_hash_128: &Compact128,
    ) -> Option<Q::Value> {
        let entry = self.map.get(&TypeId::of::<Q>())?;

        let sieve =
            (**entry).downcast_ref::<Sieve<QueryColumn<Q>, C>>().unwrap();

        sieve
            .get_normal(query_input_hash_128)
            .await
            .map(|x| x.query_result.clone())
    }
}
