//! Defines the query interface for the QBICE engine.

use std::{any::Any, fmt::Debug, hash::Hash};

use qbice_stable_hash::{Sip128Hasher, StableHash, StableHasher};
use qbice_stable_type_id::{Identifiable, StableTypeID};

use crate::config::Config;

/// The query interface of QBICE engine.
///
/// The type implements the [`Query`] trait represents a query input type,
/// which is associated with a specific output value type. This is merely an
/// interface definition; query executors define the actual behavior of queries.
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
    type Value: 'static + Send + Sync + Clone + Debug + StableHash;

    /// The kind of the query.
    ///
    /// See [`QueryKind`] for details.
    #[must_use]
    fn query_kind() -> QueryKind { QueryKind::Normal }

    /// The default value for the SCC (Strongly Connected Component) value.
    ///
    /// This is the value to returned when another query that is not a part of a
    /// SCC try to query for it.
    #[must_use]
    fn scc_value() -> Self::Value { panic!("SCC value is not specified") }
}

/// Specifies the kind of a query.
///
/// The query kinds other than `Normal` are mainly used for optimization
/// purposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum QueryKind {
    /// A normal query without any special properties.
    Normal,

    /// A projection query whose sole purpose is to extract a part of another
    /// query of kind [`QueryKind::Firewall`]  or [`QueryKind::Projection`].
    ///
    /// This kind of query should be very-fast to compute, as it is expected
    /// to be used frequently internally as a part of dependency tracking.
    ///
    /// It's sole purpose is to prevent the dirty propagation from crossing
    /// certain boundaries defined by firewall queries.
    Projection,

    /// A firewall query that acts as a boundary between a group of queries
    Firewall,
}

/// A type erased query interface.
///
/// Used internally for dynamic dispatch of queries.
pub trait DynQuery<C: Config>: 'static + Send + Sync + Any {
    /// Returns the stable type ID of the query.
    fn stable_type_id(&self) -> StableTypeID;

    /// Returns a 128-bit hash of the query, seeded with the given initial seed.
    fn hash_128(&self, initial_seed: u64) -> u128;

    /// Compares this query with another dynamically typed query for equality.
    fn eq_dyn(&self, other: &dyn DynQuery<C>) -> bool;

    /// Hashes this query into the std hash state.
    fn hash_dyn(&self, state: &mut dyn std::hash::Hasher);

    /// Generates a unique identifier for this query.
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

    /// Clones this query into a type erased box.
    fn dyn_clone(&self) -> DynQueryBox<C>;

    /// Formats this query for debugging purposes.
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

/// A unique identifier for a query key.
///
/// We assume that the combination of `StableTypeID` and `hash_128` uniquely
/// identifies a query key.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, StableHash,
)]
pub struct QueryID {
    stable_type_id: StableTypeID,
    hash_128: (u64, u64),
}

impl QueryID {
    /// Returns the stable type ID of the query.
    #[must_use]
    pub const fn stable_type_id(&self) -> StableTypeID { self.stable_type_id }

    /// Returns the 128-bit hash of the query.
    #[must_use]
    pub const fn hash_128(&self) -> u128 {
        ((self.hash_128.0 as u128) << 64) | (self.hash_128.1 as u128)
    }
}

/// A type aliased for boxed-type-erased query value.
pub type DynValueBox<C> =
    smallbox::SmallBox<dyn DynValue<C>, <C as Config>::Storage>;

/// A type aliased for boxed-type-erased query key.
pub type DynQueryBox<C> =
    smallbox::SmallBox<dyn DynQuery<C>, <C as Config>::Storage>;

/// A type erased value interface for [`Query::Value`].
pub trait DynValue<C: Config>: 'static + Send + Sync + Any {
    /// Clone the value into a type erased box.
    fn dyn_clone(&self) -> DynValueBox<C>;

    /// Format the value for debugging purposes.
    fn dyn_dbg(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;

    /// Hash the value into a 128-bit hash.
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
