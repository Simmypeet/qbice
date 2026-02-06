//! Storage abstractions and caching utilities for key-value databases.
//!
//! This crate provides a comprehensive foundation for building high-performance
//! persistent storage systems with efficient caching, data deduplication, and
//! flexible storage modes. It includes:
//!
//! - **Key-Value Database Abstraction** ([`kv_database`]): A trait-based
//!   abstraction layer supporting multiple storage backends (e.g., `RocksDB`,
//!   LMDB, in-memory). Features include wide columns for multi-value storage
//!   and key-of-set mode for efficient set operations.
//!
//! - **Storage Engine** ([`storage_engine`]): A unified interface for creating
//!   storage components including single maps, dynamic maps, and key-of-set
//!   maps with integrated write management.
//!
//! - **Write-Behind Caching** ([`write_manager`]): High-performance write-back
//!   caching with background persistence, supporting asynchronous database
//!   writes with epoch-based ordering for consistency.
//!
//! - **Interning System** ([`intern`]): A thread-safe value interning system
//!   that deduplicates immutable data using stable hashing. Features
//!   cross-execution stability and seamless serialization support for reducing
//!   memory usage and serialized data size.
//!
//! # Core Concepts
//!
//! ## Storage Modes
//!
//! The crate supports two primary storage patterns:
//!
//! - **Wide Columns**: Store multiple related values under a single key,
//!   distinguished by discriminants. Each discriminant maps to a specific value
//!   type, enabling type-safe heterogeneous storage.
//!
//! - **Key-of-Set**: Represent `HashMap<K, HashSet<V>>` relationships with
//!   efficient member insertion, deletion, and scanning operations.
//!
//! ## Caching Strategy
//!
//! The storage system uses Moka-based caching with:
//! - Transparent lazy loading from the backing database on cache misses
//! - Write-behind buffering via [`write_manager::write_behind::WriteBehind`]
//! - Epoch-based write ordering for consistency
//! - Staging layer for uncommitted writes
//!
//! ## Value Interning
//!
//! The [`intern::Interner`] enables:
//! - Automatic deduplication of equal values based on stable hash
//! - Reference-counted memory management with automatic cleanup
//! - Serialization optimization through hash-based value references
//!
//! # Example
//!
//! ```ignore
//! use qbice_storage::{
//!     kv_database::{WideColumn, WideColumnValue, DiscriminantEncoding},
//!     storage_engine::{StorageEngine, in_memory::InMemoryStorageEngine},
//! };
//! use qbice_stable_type_id::Identifiable;
//!
//! // Define a wide column
//! #[derive(Identifiable)]
//! struct UserDataColumn;
//!
//! impl WideColumn for UserDataColumn {
//!     type Discriminant = u8;
//!     type Key = u64;
//!     fn discriminant_encoding() -> DiscriminantEncoding {
//!         DiscriminantEncoding::Prefixed
//!     }
//! }
//!
//! // Define value types
//! #[derive(Clone, Encode, Decode)]
//! struct UserName(String);
//!
//! impl WideColumnValue<UserDataColumn> for UserName {
//!     fn discriminant() -> u8 { 0 }
//! }
//!
//! // Create a storage engine and maps
//! let engine = InMemoryStorageEngine;
//! let user_map = engine.new_single_map::<UserDataColumn, UserName>();
//!
//! // Retrieve values (automatically fetched from DB on miss)
//! let name = user_map.get(&user_id).await;
//! if let Some(name) = name {
//!     println!("User: {}", name.0);
//! }
//! ```
//!
//! # Design Considerations
//!
//! - **Atomicity**: Ensures that batches of operations are applied to the
//!   database atomically.
//! - **Durability Tolerance**: In case of crashes, some recent writes may be
//!   lost, at worst, there may be some recomputation in the next run.
//! - **Write Efficiency**: Optimized for high-throughput write workloads with
//!   batching and background writing.

pub mod dynamic_map;
pub mod intern;
pub mod key_of_set_map;
pub mod kv_database;
pub mod single_map;
pub mod storage_engine;
pub mod tiny_lfu;
pub mod write_batch;
pub mod write_manager;

pub(crate) mod sharded;
pub(crate) mod single_flight;
pub(crate) mod wide_column_cache;
