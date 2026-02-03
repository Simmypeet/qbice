//! Write transaction abstractions for atomic operations.
//!
//! This module provides the [`WriteTransaction`] trait for grouping
//! multiple write operations into atomic batches.

/// A trait for grouping write operations into atomic batches.
///
/// Write batches allow multiple write operations to be collected and then
/// applied atomically to the storage backend. This ensures consistency and
/// can improve performance by reducing the number of individual write
/// operations.
pub trait WriteTransaction {}

/// A no-op write batch implementation for in-memory storage.
///
/// This is a placeholder implementation used by in-memory storage backends
/// that don't require actual batch semantics, as all operations take effect
/// immediately.
#[derive(Debug, Clone, Copy)]
pub struct FauxWriteTransaction;

impl WriteTransaction for FauxWriteTransaction {}
