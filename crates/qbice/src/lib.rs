//! # QBICE - Query-Based Incremental Computation Engine
//!
//! QBICE is a high-performance, asynchronous incremental computation framework
//! for Rust. It enables you to define computation as a graph of queries, where
//! each query can depend on other queries. When inputs change, QBICE
//! automatically determines which computations need to be re-executed,
//! minimizing redundant work through intelligent caching and dependency
//! tracking.
//!
//! Typical use cases include:
//! - Compilers and language toolchains that need to recompile only affected
//!   code
//! - Build systems with automatic incremental rebuilding
//! - Data processing pipelines with complex dependencies
//! - View derivation systems (e.g., UI updates from model changes)
//! - Cache invalidation and recomputation in distributed systems
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
//! - **Persistent Storage**: Supports pluggable key-value database backends for
//!   caching query results
//! - **Dependency Visualization**: Generate interactive HTML dependency graphs
//!
//! ## Engine Lifecycle
//!
//! A typical QBICE workflow follows this pattern:
//!
//! 1. **Create**: Instantiate an [`Engine`] with your configuration
//! 2. **Register**: Add executors for each query type via
//!    [`register_executor`](Engine::register_executor)
//! 3. **Wrap**: Convert to `Arc<Engine>` for shared ownership
//! 4. **Session**: Create an [`InputSession`] to set initial inputs
//! 5. **Track**: Create a [`TrackedEngine`] for querying
//! 6. **Query**: Execute queries asynchronously
//! 7. **Update**: Drop the `TrackedEngine`, modify inputs, and repeat from step
//!    4
//!
//! ## Thread Safety
//!
//! - `&Engine`: Safe to share across threads for reading and querying
//! - `&mut Engine`: Required for executor registration and input modification
//! - `Arc<Engine>`: The standard pattern for shared engine ownership
//! - `TrackedEngine`: Lightweight, thread-local query context
//!
//! See the [`Engine`] documentation for detailed thread safety guarantees.
//!
//! ## Core Concepts
//!
//! ### Queries
//!
//! A **query** represents a unit of computation with an associated input (the
//! query key) and output (the query value). Queries implement the [`Query`]
//! trait and are identified by their type and a stable hash of their contents.
//! Query keys should be cheaply cloneable (preferably small or use `Arc` for
//! large data).
//!
//! ### Executors
//!
//! An **executor** defines how to compute the value for a specific query type.
//! Executors implement the [`Executor`] trait and can depend on other queries
//! through the [`TrackedEngine`]. Executors must be **pure functions** that
//! always return the same output for the same query input and dependent values.
//!
//! ### Engine
//!
//! The [`Engine`] is the central database that stores computed values and
//! manages the dependency graph. It tracks which queries depend on which other
//! queries and handles cache invalidation when inputs change.
//!
//! For full usage examples, see the `integration_test` crate.

extern crate self as qbice;

pub mod config;
pub mod engine;
pub mod executor;
pub mod program;
pub mod query;

pub use config::Config;
pub use engine::{
    Engine,
    computation_graph::{InputSession, SetInputResult, TrackedEngine},
};
pub use executor::{CyclicError, Executor};
// re-export companion crates
pub use qbice_derive::{Query, derive_for_query_id, executor};
pub use qbice_serialize as serialize;
pub use qbice_serialize::{Decode, Encode};
pub use qbice_stable_hash as stable_hash;
pub use qbice_stable_hash::StableHash;
pub use qbice_stable_type_id as stable_type_id;
pub use qbice_stable_type_id::Identifiable;
pub use qbice_storage as storage;
pub use query::{ExecutionStyle, Query};
