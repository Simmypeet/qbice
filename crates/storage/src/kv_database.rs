//! Key-Value Database Abstraction Layer
//!
//! This module defines the core traits and types for interacting with key-value
//! databases in a backend-agnostic way. It provides abstractions for logical
//! data structures, column families, transactions, and async access patterns.

use std::hash::Hash;

use qbice_serialize::{Decode, Encode, Plugin};
use qbice_stable_type_id::Identifiable;

mod buffer_pool;
mod default_shard_amount;

#[cfg(feature = "fjall")]
pub mod fjall;

#[cfg(feature = "rocksdb")]
pub mod rocksdb;

/// A write batch for accumulating multiple write operations that are committed
/// atomically.
///
/// Write batches provide atomicity guarantees: either all operations in the
/// batch are applied to the database, or none are. This is essential for
/// maintaining data consistency when multiple related changes must be made
/// as a single logical unit.
///
/// # Visibility and Isolation
///
/// - Changes made within a batch are **not visible** to other readers until
///   [`commit`](Self::commit) is called.
/// - If a batch is dropped without calling `commit`, all pending changes **must
///   be rolled back** automatically by the implementation.
/// - Implementations should provide snapshot isolation: reads see a consistent
///   database state from when the batch was created.
///
/// # Usage Pattern
///
/// ```ignore
/// let batch = db.write_transaction();
///
/// // Accumulate operations
/// batch.put::<MyColumn, MyValue>(&key1, &value1);
/// batch.delete::<MyColumn, MyValue>(&key2);
/// batch.insert_member::<SetColumn>(&key3, &member);
///
/// // Commit atomically
/// batch.commit();
/// ```
pub trait WriteBatch {
    /// Inserts or updates a value in a wide column.
    fn put<W: WideColumn, C: WideColumnValue<W>>(
        &self,
        key: &W::Key,
        value: &C,
    );

    /// Deletes a value from a wide column.
    fn delete<W: WideColumn, C: WideColumnValue<W>>(&self, key: &W::Key);

    /// Inserts a member into the set associated with the given key in a
    /// [`KeyOfSetColumn`] mode.
    fn insert_member<C: KeyOfSetColumn>(
        &self,
        key: &C::Key,
        value: &C::Element,
    );

    /// Deletes a member from the set associated with the given key in a
    /// [`KeyOfSetColumn`] mode.
    fn delete_member<C: KeyOfSetColumn>(
        &self,
        key: &C::Key,
        value: &C::Element,
    );

    /// Commits all pending write operations to the database.
    ///
    /// After commit, all changes become visible to other readers.
    ///
    /// The commit should always succeed, panic if it fails.
    fn commit(self);
}

/// Factory trait for creating instances of a key-value database backend.
pub trait KvDatabaseFactory {
    /// The type of key-value database produced by this factory.
    type KvDatabase;

    /// The error type returned if opening the database fails.
    type Error;

    /// Opens a new instance of the key-value database with the given
    /// serialization plugin.
    fn open(
        self,
        serialization_plugin: Plugin,
    ) -> Result<Self::KvDatabase, Self::Error>;
}

/// The primary interface for key-value database backends with support for
/// advanced storage patterns.
///
/// This trait provides a backend-agnostic abstraction over various key-value
/// storage implementations (e.g., `RocksDB`, LMDB, in-memory stores), enabling
/// portable code that can work with different storage engines.
///
/// # Thread Safety and Concurrency
///
/// All `KvDatabase` implementations must be fully thread-safe (`Send + Sync`):
/// - Multiple threads can safely perform concurrent reads without blocking
/// - Write batches provide isolation, preventing visibility of partial updates
/// - Implementations should support high-concurrency read workloads efficiently
///
/// # Storage Patterns
///
/// The database supports two distinct storage patterns:
///
/// ## Wide Columns
///
/// Store multiple heterogeneous values under a single key, distinguished by
/// discriminants. Use [`get_wide_column`](Self::get_wide_column) to retrieve
/// specific value types. Returns `None` if the key-discriminant pair doesn't
/// exist.
///
/// ## Key-of-Set
///
/// Represent `HashMap<K, HashSet<V>>` relationships where each key maps to a
/// set of elements. Use [`scan_members`](Self::scan_members) to iterate over
/// all members. Returns an empty iterator if the key has no members.
///
/// # Write Operations
///
/// All writes must go through write batches obtained via
/// [`write_batch`](Self::write_batch). This ensures:
/// - Atomic application of multiple operations
/// - Consistent isolation semantics
/// - Efficient batching for better performance
pub trait KvDatabase: 'static + Send + Sync {
    /// The type of write transaction provided by this database implementation.
    type WriteBatch: WriteBatch + Send + Sync;

    /// Retrieves a value from a wide column based on its key.
    fn get_wide_column<W: WideColumn, C: WideColumnValue<W>>(
        &self,
        key: &W::Key,
    ) -> Option<C>;

    /// Scans all members of the set associated with the given key in a
    /// [`KeyOfSetColumn`] mode.
    ///
    /// This method returns an iterator over all values that are members of the
    /// set for the given key. If the key has no members, the iterator will be
    /// empty.
    ///
    /// # Returns
    ///
    /// An iterator yielding each member of the set. The iteration order is
    /// implementation-defined and may not be stable across calls.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Scan all tags for a user
    /// for tag in db.scan_members::<UserTagsColumn>(&user_id) {
    ///     println!("Tag: {}", tag);
    /// }
    /// ```
    fn scan_members<'s, C: KeyOfSetColumn>(
        &'s self,
        key: &'s C::Key,
    ) -> impl Iterator<Item = C::Element> + use<'s, Self, C>;

    /// Creates a new write transaction for batching multiple write operations.
    ///
    /// The returned transaction must be explicitly committed via
    /// [`WriteTransaction::commit`] for changes to be persisted.
    ///
    /// [`WriteTransaction::commit`]: crate::kv_database::WriteBatch::commit
    fn write_batch(&self) -> Self::WriteBatch;
}

/// Specifies how discriminants are encoded in composite keys for wide columns.
///
/// When databases don't natively support wide columns, they are emulated by
/// encoding the discriminant into the key itself. This enum controls whether
/// the discriminant appears before or after the logical key.
///
/// # Variants
///
/// - **Prefixed**: Discriminant is prepended to the key
///   - Format: `[discriminant][key]`
///   - Groups values by discriminant in key-space
///   - Better for range queries over same value type
///
/// - **Suffixed**: Discriminant is appended to the key
///   - Format: `[key][discriminant]`
///   - Groups all values for same key together
///   - Better for accessing multiple value types for one key
///
/// # Choosing an Encoding
///
/// Choose **Prefixed** when:
/// - You often scan all values of the same type
/// - Value types are queried independently
///
/// Choose **Suffixed** when:
/// - You often access multiple value types for the same key
/// - Logical key locality is more important
///
/// # Example
///
/// For key=`42` and discriminant=`1`:
/// - Prefixed: stored as `[0x01, 0x2A]`
/// - Suffixed: stored as `[0x2A, 0x01]`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DiscriminantEncoding {
    /// Discriminant is stored before the key.
    Prefixed,
    /// Discriminant is stored after the key.
    Suffixed,
}

/// A trait representing a wide column storage structure for heterogeneous
/// multi-value storage.
///
/// Wide columns enable storing multiple related but differently-typed values
/// under a single logical key, distinguished by discriminants. Each
/// discriminant identifies a specific value type, providing type-safe access to
/// heterogeneous data within a single column family.
///
/// This pattern is particularly useful when you need to:
/// - Store multiple attributes of an entity with different types
/// - Maintain type safety while avoiding separate column families
/// - Enable efficient partial retrieval of entity data
///
/// # Storage Implementation
///
/// While some databases support wide columns natively, others emulate them
/// by encoding the discriminant into the key. For example, `RocksDB` uses
/// composite keys combining the logical key with the discriminant (either as
/// prefix or suffix, controlled by
/// [`discriminant_encoding`](Self::discriminant_encoding)).
///
/// # Example
///
/// ```ignore
/// #[derive(Identifiable)]
/// struct UserColumn;
///
/// impl WideColumn for UserColumn {
///     type Discriminant = u8;
///     type Key = u64;
///
///     fn discriminant_encoding() -> DiscriminantEncoding {
///         DiscriminantEncoding::Prefixed
///     }
/// }
///
/// // Different value types with unique discriminants
/// struct UserName(String);
/// impl WideColumnValue<UserColumn> for UserName {
///     fn discriminant() -> u8 { 0 }
/// }
///
/// struct UserAge(u32);
/// impl WideColumnValue<UserColumn> for UserAge {
///     fn discriminant() -> u8 { 1 }
/// }
/// ```
pub trait WideColumn: Identifiable + Send + Sync + 'static {
    /// The type of the discriminant used to distinguish different values.
    type Discriminant: Encode
        + Decode
        + Hash
        + Eq
        + Clone
        + 'static
        + Send
        + Sync;

    /// The type of keys used in this wide column.
    type Key: Encode + Decode + Hash + Eq + Clone + 'static + Send + Sync;

    /// Specifies how the discriminant is encoded in the key.
    fn discriminant_encoding() -> DiscriminantEncoding;
}

/// A trait for values stored in a [`WideColumn`] with a specific discriminant.
///
/// This trait associates a value type with a [`WideColumn`] and provides its
/// unique discriminant. Each implementation represents one possible value type
/// that can be stored in the wide column.
///
/// # Discriminant Uniqueness
///
/// **CRITICAL**: Each value type for a given wide column **must** have a unique
/// discriminant. Violating this invariant will cause:
/// - Data corruption (values overwriting each other)
/// - Type confusion (wrong type returned from database)
/// - Silent data loss
///
/// The discriminant is used by the storage layer to distinguish different value
/// types stored under the same key. Reusing discriminants is undefined
/// behavior.
///
/// # Type Requirements
///
/// Value types must be:
/// - `Encode + Decode`: For serialization to/from the database
/// - `Clone`: To allow caching and copying
/// - `Send + Sync + 'static`: For thread-safe storage and caching
///
/// # Example
///
/// ```ignore
/// // Good: Each value has a unique discriminant
/// struct UserName(String);
/// impl WideColumnValue<UserColumn> for UserName {
///     fn discriminant() -> u8 { 0 }
/// }
///
/// struct UserEmail(String);
/// impl WideColumnValue<UserColumn> for UserEmail {
///     fn discriminant() -> u8 { 1 }  // Different from UserName
/// }
///
/// // BAD: DO NOT reuse discriminants!
/// struct UserPhone(String);
/// impl WideColumnValue<UserColumn> for UserPhone {
///     fn discriminant() -> u8 { 0 }  // WRONG: conflicts with UserName!
/// }
/// ```
pub trait WideColumnValue<C: WideColumn>:
    Encode + Decode + Clone + 'static + Send + Sync
{
    /// Returns the unique discriminant for this value type within the
    /// associated wide column.
    fn discriminant() -> C::Discriminant;
}

/// A trait for columns representing `HashMap<Key, HashSet<Element>>`
/// relationships.
///
/// This storage pattern is optimized for managing set memberships, where each
/// key is associated with a collection of unique elements. It provides
/// efficient:
/// - Element insertion into sets
/// - Element deletion from sets
/// - Membership testing
/// - Full set scanning
///
/// # Use Cases
///
/// Key-of-set columns are ideal for:
/// - Tags or labels (e.g., user tags, article categories)
/// - Relationships (e.g., user followers, group members)
/// - Access control lists
/// - Dependency tracking
///
/// # Type Requirements
///
/// - **Key**: Must be `Encode + Hash + Eq` for storage and lookup
/// - **Element**: Must be `Encode + Decode + Hash + Eq` for storage, retrieval,
///   and uniqueness checking
///
/// Both types must also be `Clone`, `Send`, `Sync`, and `'static` for
/// thread-safe caching.
///
/// # Storage Implementation
///
/// Implementations may store set members using various strategies:
/// - Individual key-value pairs per member (e.g., `RocksDB` with composite
///   keys)
/// - Serialized sets (compact but less efficient for single-member updates)
/// - Native set types (if supported by the database)
///
/// # Example
///
/// ```ignore
/// #[derive(Identifiable)]
/// struct UserTagsColumn;
///
/// impl KeyOfSetColumn for UserTagsColumn {
///     type Key = u64;           // User ID
///     type Element = String;     // Tag name
/// }
///
/// // Operations:
/// batch.insert_member::<UserTagsColumn>(&user_id, &"rust".to_string());
/// batch.delete_member::<UserTagsColumn>(&user_id, &"python".to_string());
///
/// for tag in db.scan_members::<UserTagsColumn>(&user_id) {
///     println!("Tag: {}", tag);
/// }
/// ```
pub trait KeyOfSetColumn: Identifiable + Send + Sync + 'static {
    /// The type of keys used in this key-of-set column.
    type Key: Encode + Hash + Eq + Clone + 'static + Send + Sync;

    /// The type of elements stored in the sets associated with each key.
    type Element: Encode + Decode + Hash + Eq + Clone + 'static + Send + Sync;
}
