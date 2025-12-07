//! Configuration module for QBICE engine.
use std::fmt::Debug;

/// Configuration trait for QBICE engine, allowing customization of various
/// parameters.
pub trait Config: Default + Debug + Send + Sync + 'static {
    /// The size of static storage allocated for query keys.
    ///
    /// This can avoid heap allocations for small query keys.
    type Storage;
}

/// A default configuration implementation for QBICE.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct DefaultConfig;

impl Config for DefaultConfig {
    type Storage = [u8; 16];
}
