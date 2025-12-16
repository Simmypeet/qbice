//! # QBICE - Query-Based Incremental Computation Engine
//!
//! QBICE is a high-performance, asynchronous incremental computation framework
//! for Rust. It enables you to define computation as a graph of queries, where
//! each query can depend on other queries. When inputs change, QBICE
//! automatically determines which computations need to be re-executed,
//! minimizing redundant work through intelligent caching and dependency
//! tracking.
//!
//! ## Key Features
//!
//! - **Incremental Computation**: Only recomputes what's necessary when inputs
//!   change
//! - **Async-First Design**: Built on top of Tokio for efficient concurrent
//!   execution
//! - **Cycle Detection**: Automatically detects and handles cyclic dependencies
//! - **Type-Safe Queries**: Strongly-typed query definitions with associated
//!   value types
//! - **Thread-Safe**: Safely share the engine across multiple threads
//! - **Dependency Visualization**: Generate interactive HTML visualizations of
//!   the dependency graph
//!
//! ## Core Concepts
//!
//! ### Queries
//!
//! A **query** represents a unit of computation with an associated input (the
//! query key) and output (the query value). Queries implement the [`Query`]
//! trait and are identified by their type and a stable hash of their contents.
//!
//! ### Executors
//!
//! An **executor** defines how to compute the value for a specific query type.
//! Executors implement the [`Executor`] trait and can depend on other queries
//! through the [`TrackedEngine`].
//!
//! ### Engine
//!
//! The [`Engine`] is the central database that stores computed values and
//! manages the dependency graph. It tracks which queries depend on which other
//! queries and handles cache invalidation when inputs change.
//!
//! ## Quick Start
//!
//! Here's a simple example demonstrating the core concepts:
//!
//! ```rust
//! use std::sync::Arc;
//!
//! use qbice::{
//!     Identifiable, StableHash,
//!     config::DefaultConfig,
//!     engine::{Engine, TrackedEngine},
//!     executor::{CyclicError, Executor},
//!     query::Query,
//! };
//!
//! // Define an input query representing a variable
//! #[derive(Debug, Clone, PartialEq, Eq, Hash, StableHash, Identifiable)]
//! struct Variable(u64);
//!
//! impl Query for Variable {
//!     type Value = i64;
//! }
//!
//! // Define a computation query
//! #[derive(Debug, Clone, PartialEq, Eq, Hash, StableHash, Identifiable)]
//! struct Sum {
//!     a: Variable,
//!     b: Variable,
//! }
//!
//! impl Query for Sum {
//!     type Value = i64;
//! }
//!
//! // Define the executor for Sum queries
//! struct SumExecutor;
//!
//! impl<C: qbice::config::Config> Executor<Sum, C> for SumExecutor {
//!     async fn execute(
//!         &self,
//!         query: &Sum,
//!         engine: &TrackedEngine<C>,
//!     ) -> Result<i64, CyclicError> {
//!         let a = engine.query(&query.a).await?;
//!         let b = engine.query(&query.b).await?;
//!         Ok(a + b)
//!     }
//! }
//!
//! #[tokio::main]
//! async fn main() {
//!     // Create and configure the engine
//!     let mut engine = Engine::<DefaultConfig>::new();
//!     engine.register_executor::<Sum, _>(Arc::new(SumExecutor));
//!
//!     // Set input values
//!     {
//!         let mut session = engine.input_session();
//!         session.set_input(Variable(0), 10);
//!         session.set_input(Variable(1), 20);
//!     }
//!
//!     // Query the engine
//!     let engine = Arc::new(engine);
//!     let tracked = engine.clone().tracked();
//!     let result =
//!         tracked.query(&Sum { a: Variable(0), b: Variable(1) }).await;
//!
//!     assert_eq!(result, Ok(30));
//! }
//! ```
//!
//! ## Incremental Updates
//!
//! QBICE shines when inputs change. After modifying inputs, only the affected
//! queries are recomputed:
//!
//! ```rust
//! # use std::sync::Arc;
//! # use qbice::{
//! #     Identifiable, StableHash,
//! #     config::DefaultConfig,
//! #     engine::{Engine, TrackedEngine},
//! #     executor::{CyclicError, Executor},
//! #     query::Query,
//! # };
//! #
//! # #[derive(Debug, Clone, PartialEq, Eq, Hash, StableHash, Identifiable)]
//! # struct Variable(u64);
//! # impl Query for Variable { type Value = i64; }
//! #
//! # #[derive(Debug, Clone, PartialEq, Eq, Hash, StableHash, Identifiable)]
//! # struct Sum { a: Variable, b: Variable }
//! # impl Query for Sum { type Value = i64; }
//! #
//! # struct SumExecutor;
//! # impl<C: qbice::config::Config> Executor<Sum, C> for SumExecutor {
//! #     async fn execute(&self, query: &Sum, engine: &TrackedEngine<C>) -> Result<i64, CyclicError> {
//! #         Ok(engine.query(&query.a).await? + engine.query(&query.b).await?)
//! #     }
//! # }
//! #
//! # #[tokio::main]
//! # async fn main() {
//! let mut engine = Engine::<DefaultConfig>::new();
//! engine.register_executor::<Sum, _>(Arc::new(SumExecutor));
//!
//! // Initial inputs
//! {
//!     let mut session = engine.input_session();
//!     session.set_input(Variable(0), 100);
//!     session.set_input(Variable(1), 200);
//! }
//!
//! let mut engine = Arc::new(engine);
//! let tracked = engine.clone().tracked();
//! let query = Sum { a: Variable(0), b: Variable(1) };
//!
//! assert_eq!(tracked.query(&query).await, Ok(300));
//! drop(tracked); // Release the tracked engine
//!
//! // Update an input - only affected queries will recompute
//! {
//!     let engine_mut = Arc::get_mut(&mut engine).unwrap();
//!     let mut session = engine_mut.input_session();
//!     session.set_input(Variable(0), 150); // Changed from 100 to 150
//! }
//!
//! let tracked = engine.tracked();
//! assert_eq!(tracked.query(&query).await, Ok(350)); // Sum is recomputed
//! # }
//! ```
//!
//! ## Handling Cycles
//!
//! QBICE automatically detects cyclic dependencies and returns a
//! [`CyclicError`]. Cycles occur when Query A depends on Query B, which
//! depends on Query A (directly or transitively).
//!
//! For queries that intentionally form cycles (e.g., fixed-point computations),
//! you can implement [`Executor::scc_value`] to provide a default value:
//!
//! ```rust
//! # use std::sync::Arc;
//! # use qbice::{
//! #     Identifiable, StableHash,
//! #     config::DefaultConfig,
//! #     engine::{Engine, TrackedEngine},
//! #     executor::{CyclicError, Executor},
//! #     query::Query,
//! # };
//! #
//! // A query that may form cycles in a fixed-point computation
//! #[derive(Debug, Clone, PartialEq, Eq, Hash, StableHash, Identifiable)]
//! struct Reachable { from: u64, to: u64 }
//!
//! impl Query for Reachable {
//!     type Value = bool;
//! }
//!
//! struct ReachableExecutor;
//!
//! impl<C: qbice::config::Config> Executor<Reachable, C> for ReachableExecutor {
//!     async fn execute(
//!         &self,
//!         query: &Reachable,
//!         engine: &TrackedEngine<C>,
//!     ) -> Result<bool, CyclicError> {
//!         // Base case: same node
//!         if query.from == query.to {
//!             return Ok(true);
//!         }
//!         // Would check edges here...
//!         Ok(false)
//!     }
//!
//!     // Provide a default value for cyclic queries
//!     fn scc_value() -> bool {
//!         false // Not reachable by default
//!     }
//! }
//!
//! #[tokio::main]
//! async fn main() {
//!     let mut engine = Engine::<DefaultConfig>::new();
//!     engine.register_executor::<Reachable, _>(Arc::new(ReachableExecutor));
//!
//!     let engine = Arc::new(engine);
//!     let tracked = engine.tracked();
//!
//!     let result = tracked.query(&Reachable { from: 1, to: 1 }).await;
//!     assert_eq!(result, Ok(true));
//! }
//! ```
//!
//! [`Executor::scc_value`]: executor::Executor::scc_value
//!
//! ## Modules
//!
//! - [`config`]: Configuration traits and default implementations
//! - [`engine`]: The core engine and tracked engine types
//! - [`executor`]: Executor trait and registry for query execution
//! - [`query`]: Query trait and related types
//!
//! ## Re-exports
//!
//! - [`StableHash`]: Derive macro for stable hashing (required for [`Query`])
//! - [`Identifiable`]: Derive macro for stable type identification (required
//!   for [`Query`])
//!
//! [`Query`]: query::Query
//! [`Executor`]: executor::Executor
//! [`Engine`]: engine::Engine
//! [`TrackedEngine`]: engine::TrackedEngine
//! [`CyclicError`]: executor::CyclicError

pub mod config;
pub mod engine;
pub mod executor;
pub mod kv_database;
pub mod lru;
pub mod query;

pub use engine::{Engine, TrackedEngine};
pub use executor::Executor;
pub use qbice_serialize::{Decode, Encode};
pub use qbice_stable_hash::StableHash;
pub use qbice_stable_type_id::Identifiable;
pub use query::{ExecutionStyle, Query};
