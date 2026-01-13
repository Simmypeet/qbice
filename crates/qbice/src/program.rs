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
//! list of registration calls:
//!
//! ```ignore
//! fn main() {
//!     let mut engine = Engine::new();
//!     engine.register_executor::<QueryA, ExecutorA>(Arc::new(ExecutorA::default()));
//!     engine.register_executor::<QueryB, ExecutorB>(Arc::new(ExecutorB::default()));
//!     engine.register_executor::<QueryC, ExecutorC>(Arc::new(ExecutorC::default()));
//!     // ... hundreds more lines ...
//! }
//! ```
//!
//! This module allows you to declare executor registrations alongside the
//! executor definitions themselves, using the [`linkme`] crate's distributed
//! slice feature (similar to the [`inventory`](https://crates.io/crates/inventory) crate).
//!
//! # Usage
//!
//! 1. Define a distributed slice for your program's registrations:
//!
//! ```ignore
//! use linkme::distributed_slice;
//! use qbice::{program::Registration, MyConfig};
//!
//! #[distributed_slice]
//! pub static REGISTRATIONS: [Registration<MyConfig>];
//! ```
//!
//! 2. Register executors at their declaration site:
//!
//! ```ignore
//! use linkme::distributed_slice;
//! use qbice::{program::Registration, Executor, Query};
//!
//! struct MyQuery;
//! impl Query for MyQuery { /* ... */ }
//!
//! #[derive(Default)]
//! struct MyExecutor;
//! impl Executor<MyQuery, MyConfig> for MyExecutor { /* ... */ }
//!
//! #[distributed_slice(REGISTRATIONS)]
//! static _: Registration<MyConfig> = Registration::new::<MyQuery, MyExecutor>();
//! ```
//!
//! 3. Register all executors at once during engine setup:
//!
//! ```ignore
//! fn main() {
//!     let mut engine = Engine::new();
//!     engine.register_program(REGISTRATIONS);
//!     // All executors are now registered!
//! }
//! ```

use std::sync::Arc;

use linkme::DistributedSlice;

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
///
/// # Example
///
/// ```ignore
/// use linkme::distributed_slice;
/// use qbice::{program::Registration, MyConfig};
///
/// // Define the distributed slice in a central location
/// #[distributed_slice]
/// pub static MY_PROGRAM: [Registration<MyConfig>];
///
/// // In each executor's module, add a registration
/// #[distributed_slice(MY_PROGRAM)]
/// static _: Registration<MyConfig> = Registration::new::<MyQuery, MyExecutor>();
/// ```
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
    /// # Type Parameters
    ///
    /// * `Q` - The query type that the executor handles. Must implement
    ///   [`Query`].
    /// * `E` - The executor type. Must implement [`Executor<Q, C>`] and
    ///   [`Default`]. The [`Default`] implementation is used to create the
    ///   executor instance during registration.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use linkme::distributed_slice;
    /// use qbice::program::Registration;
    ///
    /// #[distributed_slice(MY_PROGRAM)]
    /// static _: Registration<MyConfig> = Registration::new::<MyQuery, MyExecutor>();
    /// ```
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
    ///
    /// # Example
    ///
    /// ```ignore
    /// use linkme::distributed_slice;
    /// use qbice::{Engine, program::Registration, MyConfig};
    ///
    /// // Define the program's registration slice
    /// #[distributed_slice]
    /// pub static MY_PROGRAM: [Registration<MyConfig>];
    ///
    /// fn main() {
    ///     let mut engine: Engine<MyConfig> = Engine::new();
    ///     
    ///     // Register all executors that were added to MY_PROGRAM
    ///     // throughout the codebase
    ///     engine.register_program(MY_PROGRAM);
    /// }
    /// ```
    pub fn register_program(
        &mut self,
        registrations: DistributedSlice<[Registration<C>]>,
    ) {
        for reg in registrations {
            (reg.register_executor_fn)(self);
        }
    }
}
