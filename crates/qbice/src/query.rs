//! Query definitions and related types for the QBICE engine.
//!
//! This module defines the core [`Query`] trait that all queries must
//! implement, along with supporting types for query identification and
//! execution behavior.
//!
//! # Defining Queries
//!
//! A query represents a computation with an input key and an output value.
//! To define a query:
//!
//! 1. Create a struct representing the query key
//! 2. Derive the required traits: `StableHash`, `Identifiable`, `Debug`,
//!    `Clone`, `PartialEq`, `Eq`, and `Hash`
//! 3. Implement the [`Query`] trait, specifying the output `Value` type
//!
//! # Query Identification
//!
//! Each query instance is uniquely identified by a [`QueryID`], which combines:
//! - A stable type identifier (from the [`Identifiable`] trait)
//! - A 128-bit hash of the query's contents (from the [`StableHash`] trait)
//!
//! This ensures that different query types never collide, and different
//! instances of the same query type are distinguished by their content.
//!
//! # Execution Styles
//!
//! The [`ExecutionStyle`] enum controls how queries interact with the
//! dependency tracking system:
//!
//! - [`ExecutionStyle::Normal`]: Standard queries with full dependency tracking
//! - [`ExecutionStyle::Projection`]: Fast extractors for parts of other queries
//! - [`ExecutionStyle::Firewall`]: Boundary queries that limit dirty
//!   propagation
//!
//! Most queries should use `Normal`. Advanced users can leverage `Projection`
//! and `Firewall` for performance optimization in complex dependency graphs.

use std::{any::Any, fmt::Debug, hash::Hash};

use qbice_serialize::{Decode, Encode};
use qbice_stable_hash::{Compact128, StableHash};
use qbice_stable_type_id::{Identifiable, StableTypeID};

/// The query interface of the QBICE engine.
///
/// A type implementing [`Query`] represents a query input (key) that is
/// associated with a specific output value type. The query itself only defines
/// the *what* - the actual computation is provided by an [`crate::Executor`].
///
/// # Required Traits
///
/// Query types must implement several traits:
///
/// - [`StableHash`]: For consistent hashing across program runs
/// - [`Identifiable`]: For stable type identification
/// - [`Eq`] + [`Hash`]: For use in hash maps
/// - [`Clone`]: For storing query keys
/// - [`Debug`]: For error messages and debugging
/// - [`Send`] + [`Sync`]: For thread-safe access
///
/// Most of these can be derived automatically:
///
/// ```rust
/// use qbice::{Decode, Encode, Identifiable, Query, StableHash};
///
/// #[derive(
///     Debug,
///     Clone,
///     PartialEq,
///     Eq,
///     Hash,
///     StableHash,
///     Identifiable,
///     Encode,
///     Decode,
/// )]
/// struct MyQuery {
///     id: u64,
///     name: String,
/// }
///
/// impl Query for MyQuery {
///     type Value = Vec<u8>;
/// }
/// ```
///
/// # Value Type Requirements
///
/// The associated `Value` type must also satisfy certain bounds:
///
/// - `'static`: No borrowed data
/// - [`Send`] + [`Sync`]: For thread-safe caching
/// - [`Clone`]: For returning cached values
/// - [`Debug`]: For debugging and visualization
/// - [`StableHash`]: For change detection (fingerprinting)
///
/// # Performance: Cheap Cloning
///
/// **Both the query type and its value type should be cheaply cloneable.**
///
/// The engine frequently clones queries and values internally for:
/// - Storing query keys in the dependency graph
/// - Caching computed results
/// - Returning values to callers
/// - Tracking dependencies across async boundaries
///
/// For types containing heap-allocated data, consider using shared ownership:
///
/// | Instead of | Use |
/// |------------|-----|
/// | `String` | `Arc<str>` or a shared string type |
/// | `Vec<T>` | `Arc<[T]>` |
/// | `HashMap<K, V>` | `Arc<HashMap<K, V>>` |
/// | Large structs | `Arc<T>` |
pub trait Query:
    StableHash
    + Identifiable
    + Any
    + Eq
    + Hash
    + Clone
    + Debug
    + Encode
    + Decode
    + Send
    + Sync
    + 'static
{
    /// The output value type associated with this query.
    ///
    /// This is the type returned when fuerying the engine for this query key.
    type Value: 'static
        + Send
        + Sync
        + Clone
        + Debug
        + StableHash
        + Encode
        + Decode;
}

/// Specifies the execution style of a query.
///
/// The execution style determines how a query participates in the dependency
/// tracking and dirty propagation system. Different styles offer trade-offs
/// between fine-grained reactivity and computational efficiency.
///
/// # Choosing an Execution Style
///
/// - **Most queries**: Use [`Normal`](ExecutionStyle::Normal) - it provides
///   standard incremental computation semantics
/// - **External data**: Use [`ExternalInput`](ExecutionStyle::ExternalInput)
///   for queries that read from files, network, or other external sources
/// - **Dirty Popagation Optimization**: Use
///   [`Firewall`](ExecutionStyle::Firewall) and
///   [`Projection`](ExecutionStyle::Projection) to optimize dirty propagation
///   in large dependency graphs
///
/// # Variants
///
/// ## Normal
///
/// Standard queries with full dependency tracking. This is the default and
/// appropriate for most use cases.
///
/// **Behavior:**
/// - Dependencies are tracked automatically
/// - Changes to dependencies cause immediate dirty propagation
/// - Results are cached and reused when dependencies haven't changed
///
/// **When to use:**
/// - Most computation queries
/// - Queries with stable, well-defined dependencies
/// - When you want automatic reactivity to changes
///
/// ## Firewall
///
/// Boundary queries that limit dirty propagation based on output changes
/// rather than input changes. This is a key optimization for reducing
/// unnecessary dirty propagation in large computation graphs.
///
/// **Behavior:**
/// - When dependencies change, the firewall is recomputed
/// - Downstream queries are marked dirty **only if** the firewall's output
///   value actually changed (based on fingerprint)
/// - Acts as an incremental **dirty propagation boundary** in the large
///   computation graph system
///
/// **When to use:**
/// - A computation that has a large graph, the dirty propagation of which
///   becomes expensive
/// - When many inputs change frequently, but the overall output is stable
///
/// **Note:** You almost always want to visualize your computation graph first
/// to identify where firewalls will be most effective. Moreover, firewalls
/// often work best when paired with projections downstream to minimize dirty
/// propagation.
///
/// ## Projection
///
/// Lightweight queries that extract data from firewall queries. Projections
/// are designed to be very fast (essentially field access) and work in
/// conjunction with firewalls to provide fine-grained access to coarse
/// computations.
///
/// **Behavior:**
/// - Must be extremely fast to execute (no expensive computation)
/// - Used internally to extract specific fields from firewall results
/// - Prevents unnecessary full recomputation when only a small part is needed
///
/// **Example pattern:**
/// ```text
/// Firewall Query: ComputeAllMetrics
///   ├─ Projection: ExtractMetricA
///   ├─ Projection: ExtractMetricB
///   └─ Projection: ExtractMetricC
/// ```
///
/// ## `ExternalInput`
///
/// Queries that interact with the outside world (files, network, system time,
/// etc.). These are leaf queries that don't depend on other queries but can
/// be explicitly refreshed during input sessions.
///
/// **Behavior:**
/// - Cannot depend on other queries (must be leaf nodes)
/// - Not automatically refreshed when other inputs change
/// - Can be explicitly refreshed via input session methods
/// - Executor may perform I/O or other impure operations
///
/// **When to use:**
/// - Reading configuration files
/// - Fetching data from network APIs
/// - Querying databases
/// - Reading system time or environment variables
/// - Any external state that should be explicitly controlled
///
/// **Example:**
/// ```rust,ignore
/// #[derive(Query)]
/// struct ConfigFileQuery;
///
/// impl Executor<ConfigFileQuery> for ConfigFileExecutor {
///     fn execution_style() -> ExecutionStyle {
///         ExecutionStyle::ExternalInput
///     }
///
///     async fn execute(&self, ...) -> Result<Config, CyclicError> {
///         // Read from file system
///         let config = read_config_from_disk()?;
///         Ok(config)
///     }
/// }
/// ```
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Encode, Decode,
)]
pub enum ExecutionStyle {
    /// A normal query with standard dependency tracking.
    ///
    /// This is the default and appropriate for most queries.
    Normal,

    /// A projection query for extracting parts of firewall query outputs.
    ///
    /// Projections should be very fast to compute (essentially field access)
    /// and are used to prevent unnecessary dirty propagation across firewall
    /// boundaries.
    Projection,

    /// A firewall query that acts as a change boundary.
    ///
    /// Dirty propagation stops at firewall queries - downstream queries are
    /// only marked dirty if the firewall's output value actually changes,
    /// not just because its inputs changed.
    Firewall,

    /// An external input query that can interact with the outside world.
    ///
    /// This is similar to input queries that are explicitly set with input
    /// sessions. However, external input queries are modeled by executors
    /// invoking impure operations, such as reading from files or network.
    ///
    /// The queries executed as [`Self::ExternalInput`] are assumed to be the
    /// leaf nodes in the computation graph (cannot depend on other queries).
    ///
    /// During the input session, the input session can ask the engine to
    /// refresh the values of external input queries, causing them to be
    /// re-executed and possibly dirtying dependent queries if their values
    /// change.
    ExternalInput,
}

/// A unique identifier for a query instance.
///
/// A `QueryID` uniquely identifies a specific query within the engine by
/// combining:
///
/// - A [`StableTypeID`] that identifies the query type
/// - A 128-bit hash of the query's contents
///
/// This combination ensures that:
/// - Different query types never collide (due to type ID)
/// - Different instances of the same type are distinguished (due to content
///   hash)
/// - Identification is stable across program runs (due to stable hashing)
///
/// # Hash Collision
///
/// While the probability of hash collision is astronomically low (2^-128),
/// QBICE assumes no collisions occur. In practice, this is safe for all
/// reasonable use cases.
///
/// # Internal Use
///
/// You typically don't need to work with `QueryID` directly - the engine
/// handles identification automatically. However, it's useful for:
/// - Debugging dependency graphs
/// - Custom visualization
/// - Advanced introspection
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    StableHash,
    Encode,
    Decode,
    Identifiable,
)]
pub struct QueryID {
    stable_type_id: Compact128,
    hash_128: Compact128,
}

impl QueryID {
    /// Creates a new `QueryID` for the given query type and content hash.
    #[must_use]
    pub fn new<Q: Query>(hash_128: Compact128) -> Self {
        let stable_type_id = Q::STABLE_TYPE_ID.as_u128();

        Self { stable_type_id: stable_type_id.into(), hash_128 }
    }

    /// Returns the stable type ID of this query.
    ///
    /// The type ID uniquely identifies the query's Rust type.
    #[must_use]
    pub const fn stable_type_id(&self) -> StableTypeID {
        unsafe {
            StableTypeID::from_raw_parts(
                self.stable_type_id.high(),
                self.stable_type_id.low(),
            )
        }
    }

    /// Returns the 128-bit content hash of this query.
    ///
    /// This hash is computed from the query's fields using stable hashing,
    /// ensuring consistency across program runs.
    #[must_use]
    pub fn hash_128(&self) -> u128 { self.hash_128.to_u128() }

    /// Returns the compact representation of the 128-bit content hash.
    #[must_use]
    pub const fn compact_hash_128(&self) -> Compact128 { self.hash_128 }
}
