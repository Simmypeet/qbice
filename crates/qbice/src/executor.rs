//! Query executor definitions and registry.
//!
//! This module provides the [`Executor`] trait for defining query computation
//! logic, the [`CyclicError`] type for reporting cyclic dependencies, and the
//! [`Registry`] for managing executors.
//!
//! # Defining Executors
//!
//! An executor defines how to compute the value for a specific query type.
//! Executors are async and can depend on other queries through the
//! [`TrackedEngine`].

use std::{
    any::Any, collections::HashMap, mem::MaybeUninit, pin::Pin, sync::Arc,
};

use qbice_stable_hash::Compact128;
use qbice_stable_type_id::StableTypeID;

use crate::{
    Engine, TrackedEngine,
    config::Config,
    engine::computation_graph::CallerInformation,
    query::{ExecutionStyle, Query},
};

/// Error indicating that a cyclic query dependency was detected.
///
/// This error is returned when a query directly or indirectly depends on
/// itself, creating an infinite loop in the dependency graph.
///
/// # Example
///
/// A cycle occurs when Query A depends on Query B, which depends on Query A:
///
/// ```text
/// Query A ──depends on──> Query B ──depends on──> Query A (cycle!)
/// ```
///
/// # Handling Cycles
///
/// There are several strategies for handling cyclic dependencies:
///
/// 1. **Restructure queries**: Break the cycle by introducing intermediate
///    queries or restructuring the computation
///
/// 2. **Use SCC values**: For queries that intentionally form cycles (e.g.,
///    fixed-point computations), implement [`Executor::scc_value`] to provide a
///    default value when a cycle is detected
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
pub struct CyclicError;

/// Defines the computation logic for a specific query type.
///
/// An executor is responsible for computing the value associated with a
/// query key. Executors are registered with the engine and invoked
/// automatically when a query is requested.
///
/// # Type Parameters
///
/// - `Q`: The query type this executor handles
/// - `C`: The engine configuration type
///
/// # Pure Function Semantics
///
/// **Executors must behave as pure functions** - given the same query key
/// and the same values from dependent queries, they must return the same
/// result. This is critical for correctness of the incremental computation
/// system.
///
/// The engine relies on this property to:
/// - Cache and reuse computed values safely
/// - Skip recomputation when dependencies haven't changed
/// - Detect changes via fingerprint comparison
///
/// ## What "Pure" Means Here
///
/// - **Deterministic**: Same inputs → same output, always
/// - **No hidden state**: Don't read from global mutable state, files, network,
///   or system time without modeling them as query dependencies
/// - **No side effects**: Don't write to files, databases, or external systems
///
/// If you need external data, model it as an input query that you update
/// via an input session when the external state changes.
///
/// # Thread Safety
///
/// Executors must be `Send + Sync` since they may be called from multiple
/// threads concurrently. Use interior mutability (e.g., `Mutex`, `RwLock`,
/// atomics) if the executor needs mutable state.
pub trait Executor<Q: Query, C: Config>: 'static + Send + Sync {
    /// Executes the query and returns its computed value.
    ///
    /// # Arguments
    ///
    /// - `query`: The query key to compute
    /// - `engine`: The tracked engine for querying dependencies
    ///
    /// # Returns
    ///
    /// Returns `Ok(value)` on success, or `Err(CyclicError)` if a cyclic
    /// dependency is detected.
    ///
    /// # Cancellation Safety
    ///
    /// This method should be cancellation-safe. If the future is dropped
    /// before completion, the engine will properly clean up any partial
    /// state.
    fn execute<'s, 'q, 'e>(
        &'s self,
        query: &'q Q,
        engine: &'e TrackedEngine<C>,
    ) -> impl Future<Output = Result<Q::Value, CyclicError>>
    + Send
    + use<'s, 'q, 'e, Self, Q, C>;

    /// Returns the execution style for this query type.
    ///
    /// Override this to specify non-default execution behavior:
    ///
    /// - [`ExecutionStyle::Normal`]: Standard dependency tracking (default)
    /// - [`ExecutionStyle::Projection`]: Fast field extraction
    /// - [`ExecutionStyle::Firewall`]: Change boundary
    ///
    /// # Default
    ///
    /// Returns [`ExecutionStyle::Normal`].
    #[must_use]
    fn execution_style() -> ExecutionStyle { ExecutionStyle::Normal }

    /// Returns the default value for strongly-connected component (SCC)
    /// resolution.
    ///
    /// When a query is part of a cycle and another query outside the SCC
    /// tries to access it, this value is returned instead of causing a
    /// deadlock or infinite recursion.
    ///
    /// # Panics
    ///
    /// The default implementation panics. Override this method if your
    /// query intentionally participates in cycles (e.g., fixed-point
    /// computations).
    #[must_use]
    fn scc_value() -> Q::Value { panic!("SCC value is not specified") }
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
    result: &'a mut (dyn Any + Send + Sync),
) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
    let key = key.downcast_ref::<K>().expect("Key type mismatch");
    let executor =
        executor.downcast_ref::<E>().expect("Executor type mismatch");

    Box::pin(async {
        let result_buffer: &mut MaybeUninit<Result<K::Value, CyclicError>> =
            result.downcast_mut().expect("Result type mismatch");

        result_buffer.write(executor.execute(key, engine).await);
    })
}

type InvokeExecutorFn<C> =
    for<'a> fn(
        key: &'a dyn Any,
        executor: &'a dyn Any,
        engine: &'a mut TrackedEngine<C>,
        result: &'a mut (dyn Any + Send + Sync),
    ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>>;

type RepairQueryFn<C> = for<'a> fn(
    engine: &'a Arc<Engine<C>>,
    query_id: Compact128,
    called_from: CallerInformation,
) -> Pin<
    Box<dyn Future<Output = Result<(), CyclicError>> + Send + 'a>,
>;

type ObtainSccValueFn = for<'a> fn(buffer: &'a mut dyn Any);

type ObtainExecutionStyleFn = fn() -> ExecutionStyle;

fn obtain_scc_value<
    C: Config,
    E: Executor<K, C> + 'static,
    K: Query + 'static,
>(
    buffer: &mut dyn Any,
) {
    let buffer = buffer
        .downcast_mut::<MaybeUninit<K::Value>>()
        .expect("SCC value buffer type mismatch");

    let scc_value = E::scc_value();
    buffer.write(scc_value);
}

fn obtain_execution_style<
    C: Config,
    E: Executor<K, C> + 'static,
    K: Query + 'static,
>() -> ExecutionStyle {
    E::execution_style()
}

#[derive(Debug, Clone)]
pub(crate) struct Entry<C: Config> {
    executor: Arc<dyn Any + Send + Sync>,
    invoke_executor: InvokeExecutorFn<C>,
    repair_query: RepairQueryFn<C>,
    obtain_scc_value: ObtainSccValueFn,
    obtain_execution_style: ObtainExecutionStyleFn,
}

impl<C: Config> Entry<C> {
    pub fn new<Q: Query, E: Executor<Q, C> + 'static>(
        executor: Arc<E>,
    ) -> Self {
        Self {
            executor,
            invoke_executor: invoke_executor::<C, E, Q>,
            repair_query: Engine::<C>::repair_query_from_query_id::<Q>,
            obtain_scc_value: obtain_scc_value::<C, E, Q>,
            obtain_execution_style: obtain_execution_style::<C, E, Q>,
        }
    }

    pub async fn invoke_executor<Q: Query>(
        &self,
        query_key: &Q,
        engine: &mut TrackedEngine<C>,
    ) -> Result<Q::Value, CyclicError> {
        let mut result_buffer =
            MaybeUninit::<Result<Q::Value, CyclicError>>::uninit();

        (self.invoke_executor)(
            query_key,
            self.executor.as_ref(),
            engine,
            &mut result_buffer,
        )
        .await;

        // SAFETY: the previous call should've initialized the buffer
        unsafe { result_buffer.assume_init() }
    }

    pub async fn repair_query_from_query_id(
        &self,
        engine: &Arc<Engine<C>>,
        query_id: Compact128,
        caller_information: CallerInformation,
    ) -> Result<(), CyclicError> {
        (self.repair_query)(engine, query_id, caller_information).await
    }

    pub fn obtain_scc_value<Q: Query>(&self) -> Q::Value {
        let mut buffer = MaybeUninit::<Q::Value>::uninit();
        (self.obtain_scc_value)(&mut buffer);

        unsafe { buffer.assume_init() }
    }

    pub fn obtain_execution_style(&self) -> ExecutionStyle {
        (self.obtain_execution_style)()
    }
}

/// Registry for managing query executors.
///
/// The registry stores executors for different query types and provides
/// lookup functionality for the engine. Each query type can have exactly
/// one executor registered.
///
/// # Thread Safety
///
/// The registry itself is not thread-safe for mutation. Executors should
/// be registered during engine setup before any queries are executed.
#[derive(Debug, Default)]
pub struct Registry<C: Config> {
    executors_by_key_type_id: HashMap<StableTypeID, Entry<C>>,
}

impl<C: Config> Registry<C> {
    /// Registers an executor for the given query type.
    ///
    /// # Panics
    ///
    /// If an executor is already registered for this query type, the
    /// previous registration is silently replaced.
    pub fn register<Q: Query, E: Executor<Q, C> + 'static>(
        &mut self,
        executor: Arc<E>,
    ) {
        let entry = Entry::new::<Q, E>(executor);
        self.executors_by_key_type_id.insert(Q::STABLE_TYPE_ID, entry);
    }

    /// Retrieves the executor entry for the given query type ID.
    ///
    /// # Panics
    ///
    /// Panics if no executor is registered for the query type.
    #[must_use]
    pub(crate) fn get_executor_entry_by_type_id(
        &self,
        type_id: &StableTypeID,
    ) -> &Entry<C> {
        self.executors_by_key_type_id.get(type_id).unwrap_or_else(|| {
            panic!("Failed to find executor for query type id: {type_id:?}")
        })
    }

    /// Retrieves the executor entry for the given query type.
    ///
    /// # Panics
    ///
    /// Panics if no executor is registered for the query type.
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
