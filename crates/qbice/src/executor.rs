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
    any::Any, collections::HashMap, mem::MaybeUninit, panic::AssertUnwindSafe,
    pin::Pin, sync::Arc,
};

use futures::FutureExt;
use qbice_stable_hash::Compact128;
use qbice_stable_type_id::StableTypeID;

use crate::{
    Engine, TrackedEngine,
    config::Config,
    engine::computation_graph::{
        CallerInformation, GuardedTrackedEngine, QueryDebug,
    },
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

/// An error type indicating that a query executor panicked during execution.
pub(crate) struct Panicked(Box<dyn Any + Send + 'static>);

impl Panicked {
    /// Resumes unwinding the panic with the original payload.
    pub fn resume_unwind(self) -> ! { std::panic::resume_unwind(self.0) }
}

/// Payload used to indicate a cyclic panic during executor invocation.
pub(crate) struct CyclicPanicPayload;

impl CyclicPanicPayload {
    /// Unwinds the current thread with a `CyclicPanicPayload`.
    pub fn unwind() -> ! { std::panic::panic_any(Self) }
}

/// Defines the computation logic for a specific query type.
///
/// An executor is the "implementation" side of the query system - it defines
/// **how** to compute a value for a given query key. Each query type should
/// have exactly one executor registered with the engine.
///
/// # Core Responsibilities
///
/// Executors must:
///
/// 1. **Compute values**: Implement the [`execute`](Executor::execute) method
///    to derive a result from the query key
/// 2. **Maintain purity**: Behave as pure functions for correctness
/// 3. **Handle dependencies**: Query other values via [`TrackedEngine`] as
///    needed
///
/// # Type Parameters
///
/// - `Q`: The query type this executor handles (must implement [`Query`])
/// - `C`: The engine configuration type (must implement [`Config`])
///
/// # Pure Function Semantics
///
/// **Critical: Executors must behave as pure functions** - given the same
/// query key and the same values from dependent queries, they must return the
/// same result. This is fundamental to the correctness of incremental
/// computation.
///
/// ## What "Pure" Means
///
/// ✅ **Allowed:**
/// - Reading query parameters
/// - Querying other values via `engine.query(...)`
/// - Deterministic computation
/// - Allocating and returning new values
///
/// ❌ **Not Allowed:**
/// - Reading files, network, or system time (use
///   [`ExternalInput`](crate::ExecutionStyle::ExternalInput) style instead)
/// - Accessing global mutable state
/// - Random number generation with non-deterministic seeds
/// - Side effects (writing files, printing, etc.)
///
/// ## Why Purity Matters
///
/// The engine relies on purity to:
/// - **Cache safely**: Reuse results without re-execution
/// - **Detect changes**: Compare fingerprints to skip recomputation
/// - **Track dependencies**: Build correct dependency graphs
///
/// Violating purity leads to:
/// - Stale values being returned
/// - Missing updates when dependencies change
/// - Undefined behavior in the incremental computation system
///
/// # Thread Safety
///
/// Executors must be `Send + Sync` because:
/// - They may be called from multiple threads concurrently
/// - The engine is designed for parallel query execution
///
/// # Example: Basic Executor
///
/// ```rust,ignore
/// use qbice::{Executor, Query, TrackedEngine, CyclicError, Config};
///
/// // The query type
/// #[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// struct Add { a: u64, b: u64 }
///
/// impl Query for Add {
///     type Value = u64;
/// }
///
/// // The executor implementation
/// struct AddExecutor;
///
/// impl<C: Config> Executor<Add, C> for AddExecutor {
///     async fn execute(
///         &self,
///         query: &Add,
///         _engine: &TrackedEngine<C>,
///     ) -> Result<u64, CyclicError> {
///         // Simple, pure computation
///         Ok(query.a + query.b)
///     }
/// }
/// ```
///
/// # Example: Executor with Dependencies
///
/// ```rust,ignore
/// struct MultiplyExecutor;
///
/// impl<C: Config> Executor<Multiply, C> for MultiplyExecutor {
///     async fn execute(
///         &self,
///         query: &Multiply,
///         engine: &TrackedEngine<C>,
///     ) -> Result<u64, CyclicError> {
///         // Query dependencies
///         let a = engine.query(&query.a).await?;
///         let b = engine.query(&query.b).await?;
///
///         // Compute based on dependencies
///         Ok(a * b)
///     }
/// }
/// ```
///
/// # Example: Executor with State (Instrumentation)
///
/// ```rust,ignore
/// use std::sync::atomic::{AtomicUsize, Ordering};
///
/// struct InstrumentedExecutor {
///     call_count: AtomicUsize,
/// }
///
/// impl<C: Config> Executor<MyQuery, C> for InstrumentedExecutor {
///     async fn execute(
///         &self,
///         query: &MyQuery,
///         engine: &TrackedEngine<C>,
///     ) -> Result<Value, CyclicError> {
///         // Track calls (doesn't affect purity of result)
///         self.call_count.fetch_add(1, Ordering::Relaxed);
///
///         // Pure computation
///         // ...
///     }
/// }
/// ```
pub trait Executor<Q: Query, C: Config>: 'static + Send + Sync {
    /// Executes the query and returns its computed value.
    ///
    /// This is the core method that defines your computation logic. It takes
    /// a query key and an engine reference, and returns the computed value.
    ///
    /// # Arguments
    ///
    /// * `query` - The query key containing the inputs to the computation
    /// * `engine` - The tracked engine for querying dependencies
    ///
    /// # Returns
    ///
    /// Returns `Ok(value)` on success, or `Err(CyclicError)` if a cyclic
    /// dependency is detected during execution.
    ///
    /// # Dependency Queries
    ///
    /// Use `engine.query(&dep_query)` to access dependent values. Each such
    /// call:
    /// - Records a dependency edge in the computation graph
    /// - May trigger recursive execution if the dependency is stale
    /// - May detect cycles and return `CyclicError`
    ///
    /// # Async Execution
    ///
    /// This method is async, allowing you to:
    /// - Query multiple dependencies concurrently using `tokio::join!` or
    ///   `futures::join!`
    /// - Perform async computations
    /// - Cooperate with the runtime for cancellation
    ///
    /// # Error Handling
    ///
    /// Currently, only `CyclicError` can be returned. If your computation can
    /// fail in other ways, encode the error in the value type:
    ///
    /// ```rust,ignore
    /// impl Query for Fallible {
    ///     type Value = Result<Success, MyError>;
    /// }
    ///
    /// impl<C: Config> Executor<Fallible, C> for FallibleExecutor {
    ///     async fn execute(...) -> Result<Result<Success, MyError>, CyclicError> {
    ///         let result = try_computation();
    ///         Ok(result) // Wrap your Result in Ok
    ///     }
    /// }
    /// ```
    ///
    /// # Example: Concurrent Dependency Queries
    ///
    /// ```rust,ignore
    /// async fn execute(
    ///     &self,
    ///     query: &ComplexQuery,
    ///     engine: &TrackedEngine<C>,
    /// ) -> Result<Value, CyclicError> {
    ///     // Query multiple dependencies in parallel
    ///     let (a, b, c) = tokio::join!(
    ///         engine.query(&query.dep_a),
    ///         engine.query(&query.dep_b),
    ///         engine.query(&query.dep_c),
    ///     );
    ///
    ///     // All three must succeed
    ///     let a = a?;
    ///     let b = b?;
    ///     let c = c?;
    ///
    ///     // Compute based on dependencies
    ///     Ok(combine(a, b, c))
    /// }
    /// ```
    fn execute<'s, 'q, 'e>(
        &'s self,
        query: &'q Q,
        engine: &'e TrackedEngine<C>,
    ) -> impl Future<Output = Q::Value> + Send + use<'s, 'q, 'e, Self, Q, C>;

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
    /// When a query is part of a cycle (a strongly-connected component in the
    /// dependency graph), and another query outside the SCC tries to access
    /// it, this value is returned instead of deadlocking or causing infinite
    /// recursion.
    ///
    /// # What are SCCs?
    ///
    /// A strongly-connected component is a set of queries that form a cycle:
    ///
    /// ```text
    /// Query A → Query B → Query C → Query A  (forms an SCC)
    /// ```
    ///
    /// # When This is Used
    ///
    /// The SCC value is used when:
    /// 1. A cycle is detected during execution
    /// 2. A query outside the SCC queries a member of the SCC
    /// 3. The SCC hasn't been fully resolved yet
    ///
    /// # Default Behavior
    ///
    /// The default implementation **panics**. Override this method if your
    /// query intentionally participates in cycles.
    ///
    /// # Use Cases
    ///
    /// ## Fixed-Point Computation
    ///
    /// Some algorithms require iterative refinement until a fixed point:
    ///
    /// ```rust,ignore
    /// impl<C: Config> Executor<IterativeQuery, C> for IterativeExecutor {
    ///     fn scc_value() -> Value {
    ///         // Start with a conservative initial value
    ///         Value::default()
    ///     }
    ///
    ///     async fn execute(...) -> Result<Value, CyclicError> {
    ///         // Compute based on previous iteration
    ///         let prev = engine.query(&self).await?;  // May get SCC value
    ///         let next = refine(prev);
    ///         Ok(next)
    ///     }
    /// }
    /// ```
    ///
    /// ## Graph Analysis
    ///
    /// Queries representing graph nodes that reference each other:
    ///
    /// ```rust,ignore
    /// impl<C: Config> Executor<NodeQuery, C> for NodeExecutor {
    ///     fn scc_value() -> NodeInfo {
    ///         // Provide a default for cyclic references
    ///         NodeInfo::unresolved()
    ///     }
    /// }
    /// ```
    ///
    /// # Best Practices
    ///
    /// - **Most queries should NOT implement this** - cycles usually indicate
    ///   design issues
    /// - If you do implement it, ensure the SCC value is "safe" (won't cause
    ///   incorrect results)
    /// - Document why cycles are intentional in your executor
    /// - Consider alternative designs that avoid cycles
    ///
    /// # Panics
    ///
    /// The default implementation panics with message "SCC value is not
    /// specified". Override this method to provide a value if your query
    /// type participates in cycles.
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
    engine: &'a GuardedTrackedEngine<C>,
    result: &'a mut (dyn Any + Send),
) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
    let key = key.downcast_ref::<K>().expect("Key type mismatch");
    let executor =
        executor.downcast_ref::<E>().expect("Executor type mismatch");

    Box::pin(async {
        let result_buffer: &mut MaybeUninit<Result<K::Value, Panicked>> =
            result.downcast_mut().expect("Result type mismatch");

        let result =
            AssertUnwindSafe(executor.execute(key, engine.tracked_engine()))
                .catch_unwind()
                .await
                .map_err(Panicked);

        // SAFETY: we're initializing the buffer here
        result_buffer.write(result);
    })
}

type InvokeExecutorFn<C> =
    for<'a> fn(
        key: &'a dyn Any,
        executor: &'a dyn Any,
        engine: &'a GuardedTrackedEngine<C>,
        result: &'a mut (dyn Any + Send),
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

type DebugQueryFn<C> = for<'a> fn(
    engine: &'a Engine<C>,
    query_input_hash_128: Compact128,
) -> Option<QueryDebug>;

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
    query_debug: DebugQueryFn<C>,
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
            query_debug: Engine::<C>::get_query_debug::<Q>,
            repair_query: Engine::<C>::repair_query_from_query_id::<Q>,
            obtain_scc_value: obtain_scc_value::<C, E, Q>,
            obtain_execution_style: obtain_execution_style::<C, E, Q>,
        }
    }

    pub async fn invoke_executor<Q: Query>(
        &self,
        query_key: &Q,
        engine: &GuardedTrackedEngine<C>,
    ) -> Result<Q::Value, Panicked> {
        let mut result_buffer =
            MaybeUninit::<Result<Q::Value, Panicked>>::uninit();

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

    pub fn get_query_debug(
        &self,
        engine: &Engine<C>,
        query_input_hash_128: Compact128,
    ) -> Option<QueryDebug> {
        (self.query_debug)(engine, query_input_hash_128)
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

    #[must_use]
    pub(crate) fn get_executor_entry_by_type_id_with_type_name(
        &self,
        type_id: &StableTypeID,
        type_name: &'static str,
    ) -> &Entry<C> {
        self.executors_by_key_type_id.get(type_id).unwrap_or_else(|| {
            panic!("Failed to find executor for query type: {type_name}")
        })
    }

    /// Retrieves the executor entry for the given query type ID.
    ///
    /// Returns `None` if no executor is registered for the query type.
    #[must_use]
    pub(crate) fn try_get_executor_entry_by_type_id(
        &self,
        type_id: &StableTypeID,
    ) -> Option<&Entry<C>> {
        self.executors_by_key_type_id.get(type_id)
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
