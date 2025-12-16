//! Plugin system for extensible serialization.
//!
//! This module provides a type-safe plugin system that allows users to pass
//! additional context and functionality to serialization and deserialization
//! operations. Plugins are keyed by type, allowing different subsystems to
//! store and retrieve their own context without conflicts.
//!
//! # Example
//!
//! ```ignore
//! use qbice_serialize::plugin::Plugin;
//!
//! struct MyContext {
//!     compression_level: u32,
//! }
//!
//! let mut plugin = Plugin::new();
//! plugin.insert(MyContext { compression_level: 9 });
//!
//! if let Some(ctx) = plugin.get::<MyContext>() {
//!     println!("Compression level: {}", ctx.compression_level);
//! }
//! ```

use std::{
    any::{Any, TypeId},
    collections::HashMap,
};

/// A type-safe container for plugin data.
///
/// [`Plugin`] allows passing arbitrary context to serialization and
/// deserialization operations. Each plugin is identified by its type, ensuring
/// type-safe access without runtime string keys.
///
/// This is useful for:
/// - Passing configuration options to custom serializers
/// - Providing access to shared resources (e.g., string interning tables)
/// - Implementing custom serialization strategies for specific types
#[derive(Default)]
pub struct Plugin {
    plugins: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
}

impl Plugin {
    /// Creates a new empty plugin container.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use qbice_serialize::plugin::Plugin;
    ///
    /// let plugin = Plugin::new();
    /// ```
    #[must_use]
    pub fn new() -> Self { Self { plugins: HashMap::new() } }

    /// Inserts a plugin value into the container.
    ///
    /// If a plugin of the same type already exists, it will be replaced and
    /// the old value will be returned.
    ///
    /// # Arguments
    ///
    /// * `value` - The plugin value to insert
    ///
    /// # Returns
    ///
    /// The previous value if one existed, or `None` otherwise.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use qbice_serialize::plugin::Plugin;
    ///
    /// struct Config { level: u32 }
    ///
    /// let mut plugin = Plugin::new();
    /// assert!(plugin.insert(Config { level: 5 }).is_none());
    /// assert!(plugin.insert(Config { level: 10 }).is_some());
    /// ```
    pub fn insert<T: Any + Send + Sync>(&mut self, value: T) -> Option<T> {
        self.plugins
            .insert(TypeId::of::<T>(), Box::new(value))
            .and_then(|boxed| boxed.downcast().ok().map(|b| *b))
    }

    /// Returns a reference to a plugin value if it exists.
    ///
    /// # Type Parameters
    ///
    /// * `T` - The type of plugin to retrieve
    ///
    /// # Returns
    ///
    /// A reference to the plugin value if it exists, or `None` otherwise.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use qbice_serialize::plugin::Plugin;
    ///
    /// struct Config { level: u32 }
    ///
    /// let mut plugin = Plugin::new();
    /// plugin.insert(Config { level: 5 });
    ///
    /// assert_eq!(plugin.get::<Config>().map(|c| c.level), Some(5));
    /// ```
    #[must_use]
    pub fn get<T: Any + Send + Sync>(&self) -> Option<&T> {
        self.plugins
            .get(&TypeId::of::<T>())
            .and_then(|boxed| boxed.downcast_ref())
    }

    /// Returns a mutable reference to a plugin value if it exists.
    ///
    /// # Type Parameters
    ///
    /// * `T` - The type of plugin to retrieve
    ///
    /// # Returns
    ///
    /// A mutable reference to the plugin value if it exists, or `None`
    /// otherwise.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use qbice_serialize::plugin::Plugin;
    ///
    /// struct Config { level: u32 }
    ///
    /// let mut plugin = Plugin::new();
    /// plugin.insert(Config { level: 5 });
    ///
    /// if let Some(config) = plugin.get_mut::<Config>() {
    ///     config.level = 10;
    /// }
    /// ```
    pub fn get_mut<T: Any + Send + Sync>(&mut self) -> Option<&mut T> {
        self.plugins
            .get_mut(&TypeId::of::<T>())
            .and_then(|boxed| boxed.downcast_mut())
    }

    /// Removes a plugin value from the container.
    ///
    /// # Type Parameters
    ///
    /// * `T` - The type of plugin to remove
    ///
    /// # Returns
    ///
    /// The removed plugin value if it existed, or `None` otherwise.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use qbice_serialize::plugin::Plugin;
    ///
    /// struct Config { level: u32 }
    ///
    /// let mut plugin = Plugin::new();
    /// plugin.insert(Config { level: 5 });
    ///
    /// let config = plugin.remove::<Config>();
    /// assert!(config.is_some());
    /// assert!(plugin.get::<Config>().is_none());
    /// ```
    pub fn remove<T: Any + Send + Sync>(&mut self) -> Option<T> {
        self.plugins
            .remove(&TypeId::of::<T>())
            .and_then(|boxed| boxed.downcast().ok().map(|b| *b))
    }

    /// Returns `true` if the container contains a plugin of the given type.
    ///
    /// # Type Parameters
    ///
    /// * `T` - The type of plugin to check for
    ///
    /// # Example
    ///
    /// ```ignore
    /// use qbice_serialize::plugin::Plugin;
    ///
    /// struct Config { level: u32 }
    ///
    /// let mut plugin = Plugin::new();
    /// assert!(!plugin.contains::<Config>());
    /// plugin.insert(Config { level: 5 });
    /// assert!(plugin.contains::<Config>());
    /// ```
    #[must_use]
    pub fn contains<T: Any + Send + Sync>(&self) -> bool {
        self.plugins.contains_key(&TypeId::of::<T>())
    }

    /// Returns the number of plugins in the container.
    #[must_use]
    pub fn len(&self) -> usize { self.plugins.len() }

    /// Returns `true` if the container has no plugins.
    #[must_use]
    pub fn is_empty(&self) -> bool { self.plugins.is_empty() }

    /// Removes all plugins from the container.
    pub fn clear(&mut self) { self.plugins.clear(); }
}

impl std::fmt::Debug for Plugin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Plugin")
            .field("count", &self.plugins.len())
            .finish_non_exhaustive()
    }
}
