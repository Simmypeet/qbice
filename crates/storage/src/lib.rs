//! Storage abstractions and caching utilities for key-value databases.
//!
//! This crate provides a foundation for building persistent storage systems
//! with efficient caching. It includes:
//!
//! - **Key-Value Database Abstraction** ([`kv_database`]): A trait-based
//!   abstraction layer for key-value databases, allowing interchangeable
//!   backends (e.g., `RocksDB`, LMDB, in-memory storage).
//!
//! - **SIEVE Cache** ([`sieve`]): A high-performance, sharded cache
//!   implementation using the SIEVE eviction algorithm, which provides
//!   excellent hit rates with minimal overhead compared to LRU.
//!
//! # Architecture
//!
//! The crate is designed around the [`kv_database::Column`] trait, which
//! defines the schema for key-value pairs including their types and
//! serialization requirements. This allows for type-safe database operations
//! with compile-time guarantees.
//!
//! The [`sieve::Sieve`] cache integrates seamlessly with any
//! [`kv_database::KvDatabase`] implementation, providing transparent caching
//! with lazy loading from the backing store.
//!
//! # Example
//!
//! ```ignore
//! use storage::{kv_database::{Column, KvDatabase}, sieve::Sieve};
//!
//! // Define a column schema
//! struct UserColumn;
//! impl Column for UserColumn {
//!     type Key = u64;
//!     type Value = String;
//! }
//!
//! // Create a cache backed by your database
//! let cache = Sieve::<UserColumn, MyDatabase, _>::new(
//!     1000,           // total capacity
//!     16,             // number of shards
//!     db,             // backing database
//!     Default::default(),
//! );
//!
//! // Retrieve values (automatically fetched from DB on cache miss)
//! let value = cache.get(&user_id).await;
//! ```

pub mod intern;
pub mod kv_database;
pub mod sieve;

pub(crate) mod sharded;
