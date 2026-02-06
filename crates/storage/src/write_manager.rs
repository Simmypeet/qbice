//! Write transaction management.
//!
//! This module provides the [`WriteManager`] trait for managing write
//! transactions and ensuring atomicity of write operations.

use crate::write_batch::FauxWriteBatch;

pub mod write_behind;

/// A trait for managing write transactions.
///
/// The write manager is responsible for creating new write transactions and
/// submitting them to be applied to the storage backend. This provides a
/// layer of abstraction for coordinating writes across multiple storage
/// components.
pub trait WriteManager {
    /// The type representing a write transaction.
    ///
    /// A write transaction collects write batches from various storage
    /// components and applies them atomically when submitted.
    type WriteBatch;

    /// Creates a new write transaction.
    ///
    /// # Returns
    ///
    /// A new write transaction that can be used to collect write batches.
    fn new_write_batch(&self) -> Self::WriteBatch;

    /// Submits a write transaction to be applied to the storage backend.
    ///
    /// Once submitted, all operations collected in the write transaction are
    /// applied atomically.
    ///
    /// # Parameters
    ///
    /// - `write_transaction`: The write transaction to submit.
    fn submit_write_batch(&self, write_transaction: Self::WriteBatch);
}

/// A faux write manager for storage engines that do not require actual
/// write management, such as in-memory databases.
#[derive(Debug, Clone, Copy)]
pub struct FauxWriteManager;

impl WriteManager for FauxWriteManager {
    type WriteBatch = FauxWriteBatch;

    fn new_write_batch(&self) -> Self::WriteBatch { FauxWriteBatch }

    fn submit_write_batch(&self, _write_transaction: Self::WriteBatch) {
        // No-op for faux write manager
    }
}
