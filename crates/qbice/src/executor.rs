//! Defines the [`Executor`] trait for executing queries.

use std::{any::Any, collections::HashMap, pin::Pin, sync::Arc};

use qbice_stable_type_id::StableTypeID;

use crate::{
    config::Config,
    engine::{Engine, TrackedEngine},
    query::{DynValueBox, Query, QueryID},
};

/// Error indicating that a cyclic query dependency was detected.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Default,
    thiserror::Error,
)]
#[error("cyclic query detected")]
pub struct CyclicQuery;

/// Representing the executor of a [`Query`].
///
/// The executor defines the computation logic for a specific query type.
pub trait Executor<Q: Query, C: Config>: 'static + Send + Sync {
    /// Execute the given query using the provided tracked engine.
    fn execute<'s, 'q, 'e>(
        &'s self,
        query: &'q Q,
        engine: &'e TrackedEngine<C>,
    ) -> impl Future<Output = Result<Q::Value, CyclicQuery>>
    + use<'s, 'q, 'e, Self, Q, C>;
}

fn invoke_executor<
    'a,
    C: Config,
    E: Executor<K, C> + 'static,
    K: Query + 'static,
>(
    key: &'a dyn Any,
    executor: &'a dyn Any,
    engine: &'a mut TrackedEngine<C>,
) -> Pin<Box<dyn Future<Output = Result<DynValueBox<C>, CyclicQuery>> + 'a>> {
    let key = key.downcast_ref::<K>().expect("Key type mismatch");
    let executor =
        executor.downcast_ref::<E>().expect("Executor type mismatch");

    Box::pin(async {
        executor.execute(key, engine).await.map(|x| {
            let boxed: DynValueBox<C> = smallbox::smallbox!(x);
            boxed
        })
    })
}

type InvokeExecutorFn<C> = for<'a> fn(
    key: &'a dyn Any,
    executor: &'a dyn Any,
    engine: &'a mut TrackedEngine<C>,
) -> Pin<
    Box<dyn Future<Output = Result<DynValueBox<C>, CyclicQuery>> + 'a>,
>;
type RecursivelyRepairQueryFn<C> = for<'a> fn(
    engine: &'a Arc<Engine<C>>,
    key: &'a dyn Any,
    called_from: &'a QueryID,
) -> Pin<
    Box<dyn std::future::Future<Output = Result<(), CyclicQuery>> + 'a>,
>;

#[derive(Debug, Clone)]
pub(crate) struct Entry<C: Config> {
    executor: Arc<dyn Any + Send + Sync>,
    invoke_executor: InvokeExecutorFn<C>,
    recursively_repair_query: RecursivelyRepairQueryFn<C>,
}

impl<C: Config> Entry<C> {
    pub fn new<Q: Query, E: Executor<Q, C> + 'static>(
        executor: Arc<E>,
    ) -> Self {
        Self {
            executor,
            invoke_executor: invoke_executor::<C, E, Q>,
            recursively_repair_query: Engine::recursively_repair_query::<Q>,
        }
    }

    pub async fn invoke_executor(
        &self,
        query_key: &dyn Any,
        engine: &mut TrackedEngine<C>,
    ) -> Result<DynValueBox<C>, CyclicQuery> {
        (self.invoke_executor)(query_key, self.executor.as_ref(), engine).await
    }

    pub async fn recursively_repair_query(
        &self,
        engine: &Arc<Engine<C>>,
        query_key: &dyn Any,
        called_from: &QueryID,
    ) -> Result<(), CyclicQuery> {
        (self.recursively_repair_query)(engine, query_key, called_from).await
    }
}

/// Contains the [`Executor`] objects for each key type. This struct allows
/// registering and retrieving executors for different query key types.
#[derive(Debug, Default)]
pub struct Registry<C: Config> {
    executors_by_key_type_id: HashMap<StableTypeID, Entry<C>>,
}

impl<C: Config> Registry<C> {
    /// Register a new executor for the given query type.
    pub fn register<Q: Query, E: Executor<Q, C> + 'static>(
        &mut self,
        executor: Arc<E>,
    ) {
        let entry = Entry::new::<Q, E>(executor);
        self.executors_by_key_type_id.insert(Q::STABLE_TYPE_ID, entry);
    }

    /// Retrieve the executor entry for the given query type id.
    #[must_use]
    pub(crate) fn get_executor_entry_by_type_id(
        &self,
        type_id: &StableTypeID,
    ) -> &Entry<C> {
        self.executors_by_key_type_id.get(type_id).unwrap_or_else(|| {
            panic!("Failed to find executor for query type id: {type_id:?}")
        })
    }

    /// Retrieve the executor entry for the given query type.
    #[must_use]
    pub(crate) fn get_executor_entry<Q: Query>(&self) -> &Entry<C> {
        self.executors_by_key_type_id.get(&Q::STABLE_TYPE_ID).unwrap_or_else(
            || {
                panic!(
                    "Failed to find executor for query name: {}",
                    std::any::type_name::<Q>()
                )
            },
        )
    }
}
