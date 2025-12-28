//! Session management for serialization and deserialization operations.
//!
//! This module provides a type-safe session storage mechanism that allows
//! serializers and deserializers to maintain state across multiple operations.
//! Sessions are particularly useful for:
//!
//! - Tracking visited objects to handle circular references
//! - Caching intermediate results during serialization
//! - Storing context-specific configuration
//! - Managing shared resources during a serialization pass
//!
//! # Type Safety
//!
//! The session uses a type-keyed storage system where each [`SessionKey`]
//! implementation maps to exactly one value type. This provides compile-time
//! guarantees about the types stored and retrieved from the session.
//!
//! # Example
//!
//! ```ignore
//! // Define a session key for tracking serialization depth
//! struct DepthKey;
//!
//! impl SessionKey for DepthKey {
//!     type Value = usize;
//! }
//!
//! // Use the session to track depth
//! let mut session = Session::new();
//! let depth = session.get_mut_or_default::<DepthKey>();
//! *depth += 1;
//! ```

use std::{any::TypeId, collections::HashMap};

/// A trait for defining type-safe keys into a [`Session`].
///
/// Each implementation of `SessionKey` defines a unique key type that maps
/// to a specific value type. This allows the session to store heterogeneous
/// data while maintaining type safety.
///
/// # Requirements
///
/// Both the key type and its associated value type must be:
/// - `'static` - No borrowed data
/// - `Send` - Safe to transfer between threads
/// - `Sync` - Safe to share between threads
///
/// # Example
///
/// ```ignore
/// /// Key for storing visited object IDs during serialization.
/// struct VisitedObjectsKey;
///
/// impl SessionKey for VisitedObjectsKey {
///     type Value = HashSet<usize>;
/// }
/// ```
pub trait SessionKey: 'static + Send + Sync {
    /// The type of value associated with this session key.
    type Value: 'static + Send + Sync;
}

/// A type-safe, heterogeneous storage for session state.
///
/// `Session` provides a way to store and retrieve arbitrary state during
/// serialization and deserialization operations. It uses [`TypeId`] internally
/// to distinguish between different types of stored values.
///
/// # Thread Safety
///
/// While the stored values must be `Send + Sync`, the `Session` itself is not
/// designed for concurrent access. It should be used within a single thread
/// or protected by external synchronization.
#[derive(Debug)]
pub struct Session {
    /// Internal storage mapping type IDs to boxed values.
    states: HashMap<TypeId, Box<dyn std::any::Any + Send + Sync>>,
}

impl Session {
    /// Creates a new, empty session.
    pub(crate) fn new() -> Self { Self { states: HashMap::new() } }

    /// Returns a mutable reference to the value associated with the given key,
    /// inserting a default value if none exists.
    ///
    /// This method provides get-or-insert semantics, ensuring that a value
    /// is always available for the given key type.
    ///
    /// # Type Parameters
    ///
    /// - `K`: The session key type that identifies the value to retrieve. The
    ///   key's associated `Value` type must implement [`Default`].
    ///
    /// # Returns
    ///
    /// A mutable reference to the value associated with `K`. If no value
    /// existed, a new default value is created and stored.
    ///
    /// # Example
    ///
    /// ```ignore
    /// struct CounterKey;
    /// impl SessionKey for CounterKey {
    ///     type Value = u32;
    /// }
    ///
    /// let mut session = Session::new();
    ///
    /// // First access creates the default value (0 for u32)
    /// let counter = session.get_mut_or_default::<CounterKey>();
    /// assert_eq!(*counter, 0);
    ///
    /// // Modify the value
    /// *counter = 42;
    ///
    /// // Subsequent access returns the same value
    /// let counter = session.get_mut_or_default::<CounterKey>();
    /// assert_eq!(*counter, 42);
    /// ```
    pub fn get_mut_or_default<K: SessionKey>(&mut self) -> &mut K::Value
    where
        K::Value: Default,
    {
        self.states
            .entry(TypeId::of::<K>())
            .or_insert_with(|| Box::new(K::Value::default()))
            .downcast_mut::<K::Value>()
            .expect("Session state has incorrect type")
    }
}
