//! Write transaction management.
//!
//! This module provides the [`WriteManager`] trait for managing write
//! transactions and ensuring atomicity of write operations.

use crate::write_transaction::FauxWriteTransaction;

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
    type WriteTransaction;

    /// Creates a new write transaction.
    ///
    /// # Returns
    ///
    /// A new write transaction that can be used to collect write batches.
    fn new_write_transaction(&self) -> Self::WriteTransaction;

    /// Submits a write transaction to be applied to the storage backend.
    ///
    /// Once submitted, all operations collected in the write transaction are
    /// applied atomically.
    ///
    /// # Parameters
    ///
    /// - `write_transaction`: The write transaction to submit.
    fn submit_write_transaction(
        &self,
        write_transaction: Self::WriteTransaction,
    );
}

/// A faux write manager for storage engines that do not require actual
/// write management, such as in-memory databases.
#[derive(Debug, Clone, Copy)]
pub struct FauxWriteManager;

impl WriteManager for FauxWriteManager {
    type WriteTransaction = FauxWriteTransaction;

    fn new_write_transaction(&self) -> Self::WriteTransaction {
        FauxWriteTransaction
    }

    fn submit_write_transaction(
        &self,
        _write_transaction: Self::WriteTransaction,
    ) {
        // No-op for faux write manager
    }
}
