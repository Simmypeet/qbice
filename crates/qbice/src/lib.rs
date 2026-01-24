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
//! - **Persistent Storage**: Supports pluggable key-value database backends
//!   caching query results
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
//! Let's write a simple query called `SafeDivide` that performs division but
//! returns `None` if dividing by zero. This is classic example presented from
//! **adapton** library.
//!
//! This shows how to define queries, implement executors, set up the engine,
//! and execute queries.
//!
//! ```rust
//! use std::sync::{
//!     Arc,
//!     atomic::{AtomicUsize, Ordering},
//! };
//!
//! use qbice::{
//!     Config, CyclicError, Decode, DefaultConfig, Encode, Engine, Executor,
//!     Identifiable, Query, StableHash, TrackedEngine,
//!     serialize::Plugin,
//!     stable_hash::{SeededStableHasherBuilder, Sip128Hasher},
//!     storage::kv_database::rocksdb::RocksDB,
//! };
//!
//! // ===== Define the Query Type ===== (The Interface)
//!
//! #[derive(
//!     Debug,
//!     Clone,
//!     Copy,
//!     PartialEq,
//!     Eq,
//!     PartialOrd,
//!     Ord,
//!     Hash,
//!     StableHash,
//!     Identifiable,
//!     Encode,
//!     Decode,
//! )]
//! pub enum Variable {
//!     A,
//!     B,
//! }
//!
//! // implements `Query` trait; the `Variable` becomes the query key/input to
//! // the computation
//! impl Query for Variable {
//!     // the `Value` associated type defines the output type of the query
//!     type Value = i32;
//! }
//!
//! #[derive(
//!     Debug,
//!     Clone,
//!     PartialEq,
//!     Eq,
//!     PartialOrd,
//!     Ord,
//!     Hash,
//!     StableHash,
//!     Identifiable,
//!     Encode,
//!     Decode,
//! )]
//! pub struct Divide {
//!     pub numerator: Variable,
//!     pub denominator: Variable,
//! }
//!
//! // implements `Query` trait; the `Divide` takes two `Variable`s as input
//! // and produces an `i32` as output
//! impl Query for Divide {
//!     type Value = i32;
//! }
//!
//! #[derive(
//!     Debug,
//!     Clone,
//!     PartialEq,
//!     Eq,
//!     PartialOrd,
//!     Ord,
//!     Hash,
//!     StableHash,
//!     Identifiable,
//!     Encode,
//!     Decode,
//! )]
//! pub struct SafeDivide {
//!     pub numerator: Variable,
//!     pub denominator: Variable,
//! }
//!
//! // implements `Query` trait; the `SafeDivide` takes two `Variable`s as input
//! // but produces an `Option<i32>` as output to handle division by zero
//! impl Query for SafeDivide {
//!     type Value = Option<i32>;
//! }
//!
//! // ===== Define Executors ===== (The Implementation)
//!
//! struct DivideExecutor(AtomicUsize);
//!
//! impl<C: Config> Executor<Divide, C> for DivideExecutor {
//!     async fn execute(
//!         &self,
//!         query: &Divide,
//!         engine: &TrackedEngine<C>,
//!     ) -> i32 {
//!         // increment the call count
//!         self.0.fetch_add(1, Ordering::SeqCst);
//!
//!         let num = engine.query(&query.numerator).await;
//!         let denom = engine.query(&query.denominator).await;
//!
//!         assert!(denom != 0, "denominator should not be zero");
//!
//!         num / denom
//!     }
//! }
//!
//! struct SafeDivideExecutor(AtomicUsize);
//!
//! impl<C: Config> Executor<SafeDivide, C> for SafeDivideExecutor {
//!     async fn execute(
//!         &self,
//!         query: &SafeDivide,
//!         engine: &TrackedEngine<C>,
//!     ) -> Option<i32> {
//!         // increment the call count
//!         self.0.fetch_add(1, Ordering::SeqCst);
//!
//!         let denom = engine.query(&query.denominator).await;
//!         if denom == 0 {
//!             return None;
//!         }
//!
//!         Some(
//!             engine
//!                 .query(&Divide {
//!                     numerator: query.numerator,
//!                     denominator: query.denominator,
//!                 })
//!                 .await,
//!         )
//!     }
//! }
//!
//! // putting it all together
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // create the temporary directory for the database
//!     let temp_dir = tempfile::tempdir()?;
//!
//!     let divide_executor = Arc::new(DivideExecutor(AtomicUsize::new(0)));
//!     let safe_divide_executor =
//!         Arc::new(SafeDivideExecutor(AtomicUsize::new(0)));
//!
//!     {
//!         // create the engine
//!         let mut engine = Engine::<DefaultConfig>::new_with(
//!             Plugin::default(),
//!             RocksDB::factory(temp_dir.path()),
//!             SeededStableHasherBuilder::<Sip128Hasher>::new(0),
//!         )?;
//!
//!         // register executors
//!         engine.register_executor(divide_executor.clone());
//!         engine.register_executor(safe_divide_executor.clone());
//!
//!         // wrap in Arc for shared ownership
//!         let engine = Arc::new(engine);
//!
//!         // create an input session to set input values
//!         {
//!             let mut input_session = engine.input_session();
//!             input_session.set_input(Variable::A, 42);
//!             input_session.set_input(Variable::B, 2);
//!         } // once the input session is dropped, the values are set
//!
//!         // create a tracked engine for querying
//!         let tracked_engine = engine.tracked();
//!
//!         // perform a safe division
//!         let result = tracked_engine
//!             .query(&SafeDivide {
//!                 numerator: Variable::A,
//!                 denominator: Variable::B,
//!             })
//!             .await;
//!
//!         assert_eq!(result, Some(21));
//!
//!         // both executors should have been called exactly once
//!         assert_eq!(divide_executor.0.load(Ordering::SeqCst), 1);
//!         assert_eq!(safe_divide_executor.0.load(Ordering::SeqCst), 1);
//!     }
//!
//!     // the engine is dropped here, but the database persists
//!
//!     {
//!         // create a new engine instance pointing to the same database
//!         let mut engine = Engine::<DefaultConfig>::new_with(
//!             Plugin::default(),
//!             RocksDB::factory(temp_dir.path()),
//!             SeededStableHasherBuilder::<Sip128Hasher>::new(0),
//!         )?;
//!
//!         // everytime the engine is created, executors must be re-registered
//!         engine.register_executor(divide_executor.clone());
//!         engine.register_executor(safe_divide_executor.clone());
//!
//!         // wrap in Arc for shared ownership
//!         let engine = Arc::new(engine);
//!
//!         // create a tracked engine for querying
//!         let tracked_engine = engine.clone().tracked();
//!
//!         // perform a safe division again; this time the data is loaded from
//!         // persistent storage
//!         let result = tracked_engine
//!             .query(&SafeDivide {
//!                 numerator: Variable::A,
//!                 denominator: Variable::B,
//!             })
//!             .await;
//!
//!         assert_eq!(result, Some(21));
//!
//!         // no additional executor calls should have been made
//!         assert_eq!(divide_executor.0.load(Ordering::SeqCst), 1);
//!         assert_eq!(safe_divide_executor.0.load(Ordering::SeqCst), 1);
//!
//!         // drop the tracked engine to release the Arc reference
//!         drop(tracked_engine);
//!
//!         // let's test division by zero
//!         {
//!             let mut input_session = engine.input_session();
//!
//!             input_session.set_input(Variable::B, 0);
//!         } // once the input session is dropped, the value is set
//!
//!         // create a new tracked engine for querying
//!         let tracked_engine = engine.clone().tracked();
//!
//!         let result = tracked_engine
//!             .query(&SafeDivide {
//!                 numerator: Variable::A,
//!                 denominator: Variable::B,
//!             })
//!             .await;
//!
//!         assert_eq!(result, None);
//!
//!         // the divide executor should not have been called again
//!         assert_eq!(divide_executor.0.load(Ordering::SeqCst), 1);
//!         assert_eq!(safe_divide_executor.0.load(Ordering::SeqCst), 2);
//!     }
//!
//!     // again, the engine is dropped here, but the database persists
//!
//!     {
//!         // create a new engine instance pointing to the same database
//!         let mut engine = Engine::<DefaultConfig>::new_with(
//!             Plugin::default(),
//!             RocksDB::factory(temp_dir.path()),
//!             SeededStableHasherBuilder::<Sip128Hasher>::new(0),
//!         )?;
//!
//!         // everytime the engine is created, executors must be re-registered
//!         engine.register_executor(divide_executor.clone());
//!         engine.register_executor(safe_divide_executor.clone());
//!
//!         // wrap in Arc for shared ownership
//!         let engine = Arc::new(engine);
//!
//!         // let's restore the denominator to 2
//!         {
//!             let mut input_session = engine.input_session();
//!             input_session.set_input(Variable::B, 2);
//!         } // once the input session is dropped, the value is set
//!
//!         // create tracked engine for querying
//!         let tracked_engine = engine.tracked();
//!
//!         let result = tracked_engine
//!             .query(&SafeDivide {
//!                 numerator: Variable::A,
//!                 denominator: Variable::B,
//!             })
//!             .await;
//!
//!         assert_eq!(result, Some(21));
//!
//!         // the divide executor should not have been called again
//!         assert_eq!(divide_executor.0.load(Ordering::SeqCst), 1);
//!         assert_eq!(safe_divide_executor.0.load(Ordering::SeqCst), 3);
//!     }
//!
//!     Ok(())
//! }
//! ```

extern crate self as qbice;

pub mod config;
pub mod engine;
pub mod executor;
pub mod program;
pub mod query;

pub use config::{Config, DefaultConfig};
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
