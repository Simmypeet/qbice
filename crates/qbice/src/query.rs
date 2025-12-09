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
//! ## Example
//!
//! ```rust
//! use qbice::{Identifiable, StableHash, query::Query};
//!
//! /// A query that computes the sum of two values.
//! #[derive(Debug, Clone, PartialEq, Eq, Hash, StableHash, Identifiable)]
//! struct Add {
//!     left: i64,
//!     right: i64,
//! }
//!
//! impl Query for Add {
//!     type Value = i64;
//! }
//! ```
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
//! - [`ExecutionStyle::Firewall`]: Boundary queries that limit dirty propagation
//!
//! Most queries should use `Normal`. Advanced users can leverage `Projection`
//! and `Firewall` for performance optimization in complex dependency graphs.
//!
//! [`Identifiable`]: crate::Identifiable
//! [`StableHash`]: crate::StableHash

use std::{any::Any, fmt::Debug, hash::Hash};

use qbice_stable_hash::{Sip128Hasher, StableHash, StableHasher};
use qbice_stable_type_id::{Identifiable, StableTypeID};

use crate::config::Config;

/// The query interface of the QBICE engine.
///
/// A type implementing [`Query`] represents a query input (key) that is
/// associated with a specific output value type. The query itself only defines
/// the *what* - the actual computation is provided by an [`Executor`].
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
/// use qbice::{Identifiable, StableHash, query::Query};
///
/// #[derive(Debug, Clone, PartialEq, Eq, Hash, StableHash, Identifiable)]
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
/// # Example: Input Query
///
/// Input queries are simple keys whose values are set directly:
///
/// ```rust
/// use qbice::{Identifiable, StableHash, query::Query};
///
/// /// Represents a configuration variable by name.
/// #[derive(Debug, Clone, PartialEq, Eq, Hash, StableHash, Identifiable)]
/// struct ConfigVar(String);
///
/// impl Query for ConfigVar {
///     type Value = String;
/// }
/// ```
///
/// # Example: Computed Query
///
/// Computed queries derive their values from other queries:
///
/// ```rust
/// use qbice::{Identifiable, StableHash, query::Query};
///
/// /// Computes the length of a file's contents.
/// #[derive(Debug, Clone, PartialEq, Eq, Hash, StableHash, Identifiable)]
/// struct FileLength {
///     path: String,
/// }
///
/// impl Query for FileLength {
///     type Value = usize;
/// }
/// ```
///
/// [`Executor`]: crate::executor::Executor
/// [`StableHash`]: crate::StableHash
/// [`Identifiable`]: crate::Identifiable
pub trait Query:
    StableHash
    + Identifiable
    + Any
    + Eq
    + Hash
    + Clone
    + Debug
    + Send
    + Sync
    + 'static
{
    /// The output value type associated with this query.
    ///
    /// This is the type returned when querying the engine for this query key.
    type Value: 'static + Send + Sync + Clone + Debug + StableHash;
}

/// Specifies the execution style of a query.
///
/// The execution style determines how a query participates in the dependency
/// tracking and dirty propagation system. Most queries should use
/// [`ExecutionStyle::Normal`].
///
/// # Variants
///
/// ## Normal
///
/// Standard queries with full dependency tracking. Changes to dependencies
/// cause the query to be marked dirty and recomputed on the next request.
///
/// ## Projection
///
/// Lightweight queries that extract data from firewall queries. Projections
/// are designed to be very fast (essentially field access) and are used
/// internally to optimize dependency tracking.
///
/// Use projections when you need to extract a small piece of a larger
/// computed value without creating a full dependency on that value.
///
/// ## Firewall
///
/// Boundary queries that limit dirty propagation. When a firewall query's
/// dependencies change, the dirty flag doesn't automatically propagate to
/// queries that depend on the firewall - instead, the firewall is
/// recomputed, and propagation only continues if the firewall's *output*
/// actually changes.
///
/// Firewalls are useful for:
/// - Isolating volatile inputs from stable computations
/// - Creating natural boundaries in the dependency graph
/// - Optimizing rebuild times in large systems
///
/// # Example
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
/// struct ExpensiveQuery(u64);
///
/// impl Query for ExpensiveQuery {
///     type Value = Vec<u8>;
/// }
///
/// struct ExpensiveExecutor;
///
/// impl<C: Config> Executor<ExpensiveQuery, C> for ExpensiveExecutor {
///     async fn execute(
///         &self,
///         query: &ExpensiveQuery,
///         engine: &TrackedEngine<C>,
///     ) -> Result<Vec<u8>, CyclicError> {
///         // Expensive computation here
///         Ok(vec![query.0 as u8])
///     }
///
///     fn execution_style() -> ExecutionStyle {
///         // Mark this as a firewall to prevent unnecessary recomputation
///         // of downstream queries when the output hasn't changed
///         ExecutionStyle::Firewall
///     }
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
}

/// Type-erased interface for queries.
///
/// This trait enables storing and manipulating query keys without knowing
/// their concrete types at compile time. It's used internally by the engine
/// for dynamic dispatch and dependency tracking.
///
/// You typically don't need to implement or use this trait directly - it's
/// automatically implemented for all types that implement [`Query`].
pub trait DynQuery<C: Config>: 'static + Send + Sync + Any {
    /// Returns the stable type ID of the query.
    fn stable_type_id(&self) -> StableTypeID;

    /// Computes a 128-bit hash of the query, seeded with the given seed.
    fn hash_128(&self, initial_seed: u64) -> u128;

    /// Compares this query with another type-erased query for equality.
    fn eq_dyn(&self, other: &dyn DynQuery<C>) -> bool;

    /// Hashes this query into the given hasher.
    fn hash_dyn(&self, state: &mut dyn std::hash::Hasher);

    /// Generates a unique identifier for this query instance.
    fn query_identifier(&self, initial_seed: u64) -> QueryID {
        QueryID {
            stable_type_id: self.stable_type_id(),
            hash_128: {
                let hash_128 = self.hash_128(initial_seed);
                (
                    (hash_128 >> 64) as u64,
                    (hash_128 & 0xFFFF_FFFF_FFFF_FFFF) as u64,
                )
            },
        }
    }

    /// Clones this query into a type-erased box.
    fn dyn_clone(&self) -> DynQueryBox<C>;

    /// Formats this query for debugging.
    fn dbg_dyn(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
}

impl<Q: Query, C: Config> DynQuery<C> for Q {
    fn stable_type_id(&self) -> StableTypeID { Q::STABLE_TYPE_ID }

    fn hash_128(&self, initial_seed: u64) -> u128 {
        let mut hasher = qbice_stable_hash::Sip128Hasher::new();

        hasher.write_u64(initial_seed);
        self.stable_hash(&mut hasher);

        hasher.finish()
    }

    fn eq_dyn(&self, other: &dyn DynQuery<C>) -> bool {
        let Some(other_q) = other.downcast_query::<Q>() else {
            return false;
        };

        self == other_q
    }

    fn hash_dyn(&self, mut state: &mut dyn std::hash::Hasher) {
        std::hash::Hash::hash(self, &mut state);
    }

    fn dyn_clone(&self) -> DynQueryBox<C> {
        let boxed: DynQueryBox<C> = smallbox::smallbox!(self.clone());
        boxed
    }

    fn dbg_dyn(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

impl<C: Config> dyn DynQuery<C> + '_ {
    /// Attempt to downcast the query to a specific query type.
    pub fn downcast_query<Q: Query>(&self) -> Option<&Q> {
        let as_any = self as &dyn Any;
        as_any.downcast_ref::<Q>()
    }
}

impl<C: Config> PartialEq for dyn DynQuery<C> + '_ {
    fn eq(&self, other: &Self) -> bool { self.eq_dyn(other) }
}

impl<C: Config> Eq for dyn DynQuery<C> + '_ {}

impl<C: Config> Hash for dyn DynQuery<C> + '_ {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.hash_dyn(state);
    }
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
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, StableHash,
)]
pub struct QueryID {
    stable_type_id: StableTypeID,
    hash_128: (u64, u64),
}

impl QueryID {
    /// Returns the stable type ID of this query.
    ///
    /// The type ID uniquely identifies the query's Rust type.
    #[must_use]
    pub const fn stable_type_id(&self) -> StableTypeID { self.stable_type_id }

    /// Returns the 128-bit content hash of this query.
    ///
    /// This hash is computed from the query's fields using stable hashing,
    /// ensuring consistency across program runs.
    #[must_use]
    pub const fn hash_128(&self) -> u128 {
        ((self.hash_128.0 as u128) << 64) | (self.hash_128.1 as u128)
    }
}

/// Type-erased boxed query value.
///
/// This is an internal type used for storing query results with inline
/// optimization. The storage size is determined by [`Config::Storage`].
///
/// [`Config::Storage`]: crate::config::Config::Storage
pub type DynValueBox<C> =
    smallbox::SmallBox<dyn DynValue<C>, <C as Config>::Storage>;

/// Type-erased boxed query key.
///
/// This is an internal type used for storing query keys with inline
/// optimization. The storage size is determined by [`Config::Storage`].
///
/// [`Config::Storage`]: crate::config::Config::Storage
pub type DynQueryBox<C> =
    smallbox::SmallBox<dyn DynQuery<C>, <C as Config>::Storage>;

/// Type-erased interface for query values.
///
/// This trait enables storing and manipulating query values without knowing
/// their concrete types at compile time. It's used internally by the engine
/// for the value cache.
///
/// You typically don't need to implement or use this trait directly.
pub trait DynValue<C: Config>: 'static + Send + Sync + Any {
    /// Clones the value into a type-erased box.
    fn dyn_clone(&self) -> DynValueBox<C>;

    /// Formats the value for debugging.
    fn dyn_dbg(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;

    /// Computes a 128-bit hash of the value for fingerprinting.
    fn hash_128_value(&self, initial_seed: u64) -> u128;
}

impl<T: 'static + Send + Sync + Clone + Debug + StableHash, C: Config>
    DynValue<C> for T
{
    fn dyn_clone(&self) -> DynValueBox<C> { smallbox::smallbox!(self.clone()) }

    fn dyn_dbg(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }

    fn hash_128_value(&self, initial_seed: u64) -> u128 {
        let mut hasher = Sip128Hasher::new();
        initial_seed.stable_hash(&mut hasher);
        self.stable_hash(&mut hasher);
        hasher.finish()
    }
}

impl<C: Config> dyn DynValue<C> {
    /// Attempt to downcast the value to a specific type.
    pub fn downcast_value<T: 'static + Send + Sync>(&self) -> Option<&T> {
        let as_any = self as &dyn Any;
        as_any.downcast_ref::<T>()
    }
}

impl<C: Config> Debug for dyn DynValue<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.dyn_dbg(f)
    }
}

impl<C: Config> Debug for dyn DynQuery<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.dbg_dyn(f)
    }
}
