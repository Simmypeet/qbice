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
//!
//! ## Basic Executor
//!
//! ```rust
//! use std::sync::Arc;
//! use qbice::{
//!     Identifiable, StableHash,
//!     config::{Config, DefaultConfig},
//!     engine::{Engine, TrackedEngine},
//!     executor::{CyclicError, Executor},
//!     query::Query,
//! };
//!
//! // Define a query
//! #[derive(Debug, Clone, PartialEq, Eq, Hash, StableHash, Identifiable)]
//! struct Factorial(u64);
//!
//! impl Query for Factorial {
//!     type Value = u64;
//! }
//!
//! // Define the executor
//! struct FactorialExecutor;
//!
//! impl<C: Config> Executor<Factorial, C> for FactorialExecutor {
//!     async fn execute(
//!         &self,
//!         query: &Factorial,
//!         engine: &TrackedEngine<C>,
//!     ) -> Result<u64, CyclicError> {
//!         let n = query.0;
//!         if n <= 1 {
//!             Ok(1)
//!         } else {
//!             // Query for factorial of n-1
//!             let prev = engine.query(&Factorial(n - 1)).await?;
//!             Ok(n * prev)
//!         }
//!     }
//! }
//!
//! # #[tokio::main]
//! # async fn main() {
//! // Register and use
//! let mut engine = Engine::<DefaultConfig>::new();
//! engine.register_executor::<Factorial, _>(Arc::new(FactorialExecutor));
//!
//! // Set base case as input
//! {
//!     let mut session = engine.input_session();
//!     session.set_input(Factorial(0), 1);
//!     session.set_input(Factorial(1), 1);
//! }
//!
//! let engine = Arc::new(engine);
//! let tracked = engine.tracked();
//! assert_eq!(tracked.query(&Factorial(5)).await, Ok(120));
//! # }
//! ```
//!
//! ## Executor with State
//!
//! Executors can hold state (e.g., caches, configuration):
//!
//! ```rust
//! use std::sync::atomic::{AtomicUsize, Ordering};
//! use qbice::{
//!     Identifiable, StableHash,
//!     config::Config,
//!     engine::TrackedEngine,
//!     executor::{CyclicError, Executor},
//!     query::Query,
//! };
//!
//! #[derive(Debug, Clone, PartialEq, Eq, Hash, StableHash, Identifiable)]
//! struct CountedQuery(u64);
//!
//! impl Query for CountedQuery {
//!     type Value = u64;
//! }
//!
//! /// An executor that tracks how many times it has been called.
//! struct CountingExecutor {
//!     call_count: AtomicUsize,
//! }
//!
//! impl<C: Config> Executor<CountedQuery, C> for CountingExecutor {
//!     async fn execute(
//!         &self,
//!         query: &CountedQuery,
//!         _engine: &TrackedEngine<C>,
//!     ) -> Result<u64, CyclicError> {
//!         self.call_count.fetch_add(1, Ordering::Relaxed);
//!         Ok(query.0 * 2)
//!     }
//! }
//! ```
//!
//! [`TrackedEngine`]: crate::engine::TrackedEngine

use std::{
    any::Any, collections::HashMap, mem::MaybeUninit, pin::Pin, sync::Arc,
};

use qbice_stable_type_id::StableTypeID;

use crate::{
    config::Config,
    engine::{Engine, TrackedEngine},
    query::{DynValueBox, ExecutionStyle, Query, QueryID},
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
///    fixed-point computations), implement [`Executor::scc_value`] to provide
///    a default value when a cycle is detected
///
/// # Example of SCC Handling
///
/// ```rust
/// use qbice::{
///     Identifiable, StableHash,
///     config::Config,
///     engine::TrackedEngine,
///     executor::{CyclicError, Executor},
///     query::{Query, ExecutionStyle},
/// };
///
/// #[derive(Debug, Clone, PartialEq, Eq, Hash, StableHash, Identifiable)]
/// struct FixedPoint(u64);
///
/// impl Query for FixedPoint {
///     type Value = i64;
/// }
///
/// struct FixedPointExecutor;
///
/// impl<C: Config> Executor<FixedPoint, C> for FixedPointExecutor {
///     async fn execute(
///         &self,
///         query: &FixedPoint,
///         engine: &TrackedEngine<C>,
///     ) -> Result<i64, CyclicError> {
///         // Computation that may form cycles
///         Ok(0)
///     }
///
///     fn scc_value() -> i64 {
///         // Return a default value when a cycle is detected
///         0
///     }
/// }
/// ```
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
/// # Thread Safety
///
/// Executors must be `Send + Sync` since they may be called from multiple
/// threads concurrently. Use interior mutability (e.g., `Mutex`, `RwLock`,
/// atomics) if the executor needs mutable state.
///
/// # Querying Dependencies
///
/// Use the [`TrackedEngine`] parameter to query for dependent values:
///
/// ```rust
/// use qbice::{
///     Identifiable, StableHash,
///     config::Config,
///     engine::TrackedEngine,
///     executor::{CyclicError, Executor},
///     query::Query,
/// };
///
/// #[derive(Debug, Clone, PartialEq, Eq, Hash, StableHash, Identifiable)]
/// struct Input(u64);
/// impl Query for Input { type Value = i64; }
///
/// #[derive(Debug, Clone, PartialEq, Eq, Hash, StableHash, Identifiable)]
/// struct Doubled(u64);
/// impl Query for Doubled { type Value = i64; }
///
/// struct DoubledExecutor;
///
/// impl<C: Config> Executor<Doubled, C> for DoubledExecutor {
///     async fn execute(
///         &self,
///         query: &Doubled,
///         engine: &TrackedEngine<C>,
///     ) -> Result<i64, CyclicError> {
///         // Query the input - this creates a dependency
///         let input = engine.query(&Input(query.0)).await?;
///         Ok(input * 2)
///     }
/// }
/// ```
///
/// [`TrackedEngine`]: crate::engine::TrackedEngine
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
) -> Pin<
    Box<dyn Future<Output = Result<DynValueBox<C>, CyclicError>> + Send + 'a>,
> {
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
    Box<dyn Future<Output = Result<DynValueBox<C>, CyclicError>> + Send + 'a>,
>;
type RecursivelyRepairQueryFn<C> = for<'a> fn(
    engine: &'a Arc<Engine<C>>,
    key: &'a dyn Any,
    called_from: &'a QueryID,
) -> Pin<
    Box<dyn std::future::Future<Output = Result<(), CyclicError>> + Send + 'a>,
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
    recursively_repair_query: RecursivelyRepairQueryFn<C>,
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
            recursively_repair_query: Engine::recursively_repair_query::<Q>,
            obtain_scc_value: obtain_scc_value::<C, E, Q>,
            obtain_execution_style: obtain_execution_style::<C, E, Q>,
        }
    }

    pub async fn invoke_executor(
        &self,
        query_key: &(dyn Any + Send + Sync),
        engine: &mut TrackedEngine<C>,
    ) -> Result<DynValueBox<C>, CyclicError> {
        (self.invoke_executor)(query_key, self.executor.as_ref(), engine).await
    }

    pub async fn recursively_repair_query(
        &self,
        engine: &Arc<Engine<C>>,
        query_key: &(dyn Any + Send + Sync),
        called_from: &QueryID,
    ) -> Result<(), CyclicError> {
        (self.recursively_repair_query)(engine, query_key, called_from).await
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
///
/// # Example
///
/// ```rust
/// use std::sync::Arc;
/// use qbice::{
///     Identifiable, StableHash,
///     config::{Config, DefaultConfig},
///     engine::{Engine, TrackedEngine},
///     executor::{CyclicError, Executor, Registry},
///     query::Query,
/// };
///
/// #[derive(Debug, Clone, PartialEq, Eq, Hash, StableHash, Identifiable)]
/// struct MyQuery(u64);
/// impl Query for MyQuery { type Value = i64; }
///
/// struct MyExecutor;
/// impl<C: Config> Executor<MyQuery, C> for MyExecutor {
///     async fn execute(&self, q: &MyQuery, _: &TrackedEngine<C>) -> Result<i64, CyclicError> {
///         Ok(q.0 as i64)
///     }
/// }
///
/// let mut engine = Engine::<DefaultConfig>::new();
/// engine.register_executor::<MyQuery, _>(Arc::new(MyExecutor));
/// ```
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
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::sync::Arc;
    /// use qbice::{
    ///     Identifiable, StableHash,
    ///     config::{Config, DefaultConfig},
    ///     engine::TrackedEngine,
    ///     executor::{CyclicError, Executor, Registry},
    ///     query::Query,
    /// };
    ///
    /// #[derive(Debug, Clone, PartialEq, Eq, Hash, StableHash, Identifiable)]
    /// struct MyQuery;
    /// impl Query for MyQuery { type Value = (); }
    ///
    /// struct MyExecutor;
    /// impl<C: Config> Executor<MyQuery, C> for MyExecutor {
    ///     async fn execute(&self, _: &MyQuery, _: &TrackedEngine<C>) -> Result<(), CyclicError> {
    ///         Ok(())
    ///     }
    /// }
    ///
    /// let mut registry = Registry::<DefaultConfig>::default();
    /// registry.register::<MyQuery, _>(Arc::new(MyExecutor));
    /// ```
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
