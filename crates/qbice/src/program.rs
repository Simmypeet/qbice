//! Distributed executor registration using [`linkme`].
//!
//! This module provides a mechanism for registering executors at their
//! declaration site rather than centralizing all registrations in a single
//! location (e.g., the `main` function).
//!
//! # Motivation
//!
//! In large applications with many executors, the traditional approach of
//! registering all executors in one place leads to a long, hard-to-maintain
//! list of registration calls. This causes several issues:
//!
//! - **Scalability**: Adding a new query type requires modifying a central file
//! - **Coupling**: Query declarations become tightly coupled to a single
//!   registration point
//! - **Maintainability**: Long registration functions are error-prone
//!
//! This module allows you to declare executor registrations alongside the
//! executor definitions themselves, using the [`linkme`] crate's distributed
//! slice feature (similar to the [`inventory`](https://crates.io/crates/inventory) crate).
//!
//! # Usage
//!
//! 1. **Define a Distributed Slice**: Create a module or library that defines
//!    your program's registrations using `#[distributed_slice]`
//!
//! 2. **Register Executors at Declaration Site**: In each module where you
//!    define executors, add a distributed slice entry using
//!    `#[distributed_slice(REGISTRATIONS)]`
//!
//! 3. **Register All Executors at Once**: In your `main` or initialization
//!    function, call [`Engine::register_program`] with your distributed slice
//!
//! # Benefits
//!
//! - **Modularity**: Executors are registered near their definitions
//! - **Scalability**: Adding new queries doesn't require modifying a central
//!   file
//! - **Maintainability**: Each executor registration is self-contained
//! - **Decoupling**: Query modules don't depend on a central registry
//!
//! # Comparison with Manual Registration
//!
//! | Aspect | Manual | Distributed |
//! |--------|--------|-------------|
//! | New executor | Modify main.rs | Add line near executor |
//! | Central point of failure | Yes | No |
//! | Scalability | Poor | Good |
//! | Code organization | Centralized | Decentralized |

use std::sync::Arc;

use crate::{Config, Engine, Executor, Query};

/// A registration entry for an executor in a distributed slice.
///
/// This struct captures the information needed to register an executor with an
/// [`Engine`] at runtime. It is designed to be used with [`linkme`]'s
/// distributed slices, allowing executor registrations to be declared
/// alongside their implementations rather than in a centralized location.
///
/// # Type Parameters
///
/// * `C` - The configuration type that implements [`Config`].
#[derive(Debug, Clone, Copy, Hash)]
pub struct Registration<C: Config> {
    register_executor_fn: fn(&mut Engine<C>),
}

impl<C: Config> Registration<C> {
    /// Creates a new registration for an executor.
    ///
    /// This function creates a [`Registration`] that, when applied to an
    /// [`Engine`], will register the specified executor type `E` to handle
    /// queries of type `Q`.
    ///
    /// The executor is instantiated using its `Default` implementation. If you
    /// need custom executor initialization, use manual registration instead.
    ///
    /// # Type Parameters
    ///
    /// * `Q` - The query type that the executor handles. Must implement
    ///   [`Query`].
    /// * `E` - The executor type. Must implement [`Executor<Q, C>`] and
    ///   [`Default`]. The [`Default`] implementation is used to create the
    ///   executor instance during registration.
    ///
    /// # Best Practices
    ///
    /// - Place the registration next to the executor definition for clarity
    /// - Use the `_` name for the registration (it's never directly used)
    /// - Make sure the executor implements `Default` or create a manual
    ///   registration
    /// - Test that all expected executors are registered by the distributed
    ///   slice
    #[must_use]
    pub const fn new<Q: Query, E: Executor<Q, C> + Default>() -> Self {
        Self { register_executor_fn: Self::register_executor::<Q, E> }
    }

    fn register_executor<Q: Query, E: Executor<Q, C> + Default>(
        engine: &mut Engine<C>,
    ) {
        let executor = Arc::new(E::default());
        engine.register_executor::<Q, E>(executor);
    }
}

impl<C: Config> Engine<C> {
    /// Registers all executors from a distributed slice of registrations.
    ///
    /// This method iterates through all [`Registration`] entries in the
    /// provided distributed slice and registers each executor with this
    /// engine. This enables a decentralized registration pattern where
    /// executors can be registered at their declaration site.
    ///
    /// # Arguments
    ///
    /// * `registrations` - A [`DistributedSlice`] containing all
    ///   [`Registration`] entries to process. This is typically a static slice
    ///   populated using `#[distributed_slice]` attributes throughout the
    ///   codebase.
    pub fn register_program<'x>(
        &mut self,
        registrations: impl IntoIterator<Item = &'x Registration<C>>,
    ) {
        for reg in registrations {
            (reg.register_executor_fn)(self);
        }
    }
}
