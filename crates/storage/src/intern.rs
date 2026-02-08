//! A thread-safe interning system for deduplicating values based on stable
//! hashing.
//!
//! Interning is a memory optimization technique where equal values are stored
//! only once in memory, with all references pointing to the same allocation.
//! This module provides a concurrent interner using stable hashing to identify
//! equal values deterministically across program executions.
//!
//! # Key Components
//!
//! - [`InternedID`]: A unique identifier combining type ID and content hash
//! - [`Interned<T>`]: A reference-counted handle providing transparent value
//!   access
//! - [`Interner`]: The core deduplication engine with sharded concurrent access
//!
//! # Benefits of Interning
//!
//! ## Memory Savings
//!
//! When you have many duplicate values, interning can dramatically reduce
//! memory usage:
//! - 1000 copies of "hello" → 1 allocation + 1000 small handles
//! - Large duplicate structures → single allocation shared by all references
//!
//! ## Unsized Type Support
//!
//! The interner supports interning unsized types like `str` and `[T]` through
//! the [`Interner::intern_unsized`] method. This is especially useful for
//! string interning where you want to store `Interned<str>` rather than
//! `Interned<String>`, saving the extra indirection of the `String` wrapper:
//!
//! ```ignore
//! // Intern a str slice directly
//! let s1: Interned<str> = interner.intern_unsized("hello world".to_string());
//! let s2: Interned<str> = interner.intern_unsized("hello world".to_string());
//!
//! // Same allocation
//! assert!(Arc::ptr_eq(&s1.0, &s2.0));
//!
//! // Can also use Box<str> or any type that converts to Arc<str>
//! let s3: Interned<str> = interner.intern_unsized(Box::<str>::from("hello world"));
//! ```
//!
//! ## Serialization Optimization
//!
//! The interning system integrates with the serialization framework:
//! - First occurrence: full value serialized
//! - Subsequent occurrences: only hash reference serialized
//! - Automatic deduplication during deserialization
//!
//! ## Cross-Execution Stability
//!
//! Using stable hashing ensures:
//! - Same values produce same hashes across program runs
//! - Serialized data can reference values by hash
//! - Deterministic behavior for testing and debugging
//!
//! # When to Use Interning
//!
//! **Use interning when:**
//! - Many duplicate immutable values exist (strings, configs, AST nodes)
//! - Values are compared frequently by identity
//! - Serialization size matters (network protocols, caching)
//! - Memory usage is more important than allocation speed
//!
//! **Use regular `Arc`/`Rc` when:**
//! - Values are rarely duplicated
//! - Cross-execution stability isn't needed
//! - Allocation performance is critical (hashing overhead matters)
//! - Values are mutable
//!
//! # Architecture
//!
//! The interner uses:
//! - **Sharding**: Multiple independent hash tables to reduce lock contention
//! - **Weak references**: Automatic cleanup when values are no longer used
//! - **Stable hashing**: Deterministic 128-bit hashes for cross-run stability
//! - **Type safety**: Type IDs prevent hash collisions between different types
//!
//! # Example
//!
//! ```ignore
//! use qbice_storage::intern::Interner;
//! use qbice_stable_hash::BuildStableHasher128;
//!
//! // Create an interner
//! let interner = Interner::new(16, BuildStableHasher128::default());
//!
//! // Intern some values
//! let s1 = interner.intern("hello world".to_string());
//! let s2 = interner.intern("hello world".to_string());
//!
//! // Same allocation: s1 and s2 point to the exact same memory
//! assert!(Arc::ptr_eq(&s1.0, &s2.0));
//!
//! // Transparent access via Deref
//! assert_eq!(&*s1, "hello world");
//!
//! // When all Interned handles are dropped, the value is freed
//! drop(s1);
//! drop(s2);
//! // Memory is now reclaimed
//! ```
//!
//! # Serialization Integration
//!
//! ```ignore
//! use qbice_serialize::Plugin;
//!
//! // Add interner to serialization plugin
//! let mut plugin = Plugin::new();
//! plugin.insert(interner.clone());
//!
//! // Encode data with automatic deduplication
//! let data = vec![s1.clone(), s1.clone(), s2.clone()];
//! // s1's value is serialized once, then referenced by hash
//! encoder.encode(&data, &plugin, &mut session)?;
//! ```

use std::{
    any::Any,
    borrow::Borrow,
    collections::{HashMap, hash_map::Entry},
    hash::Hash,
    ops::Deref,
    path::Path,
    sync::{Arc, Weak},
    thread::JoinHandle,
    time::Duration,
};

use crossbeam_channel::{Receiver, Sender, unbounded};
use fxhash::{FxBuildHasher, FxHashSet};
use parking_lot::RwLockReadGuard;
use qbice_serialize::{
    Decode, Decoder, Encode, Encoder, Plugin,
    session::{Session, SessionKey},
};
use qbice_stable_hash::{
    BuildStableHasher, Compact128, StableHash, StableHasher,
};
use qbice_stable_type_id::{Identifiable, StableTypeID};

use crate::sharded::Sharded;

/// A globally unique identifier for an interned value.
///
/// `InternedID` combines two components to create a collision-resistant
/// identifier:
/// 1. **Type ID** ([`StableTypeID`]): Uniquely identifies the Rust type
/// 2. **Content Hash** ([`Compact128`]): 128-bit hash of the value's content
///
/// This dual-component design prevents type confusion: values of different
/// types with the same content hash will still have different `InternedID`s.
///
/// # Structure
///
/// ```text
/// InternedID {
///     stable_type_id: 0x1234_5678_9abc_def0,  // Type's stable ID
///     hash_128: 0xfedC_BA98_7654_3210_...,    // 128-bit content hash
/// }
/// ```
///
/// # Example Scenario
///
/// ```ignore
/// // These would have the same content hash but different type IDs
/// let string_42 = interner.intern("42".to_string());
/// let int_42 = interner.intern(42_i32);
///
/// // Different InternedIDs prevent type confusion:
/// // - string_42's ID: (String type ID, hash("42"))
/// // - int_42's ID: (i32 type ID, hash(42))
/// ```
///
/// # Hash Collisions
///
/// While 128-bit hashes make collisions astronomically unlikely, the
/// combination with type IDs provides an additional layer of protection.
/// Different types cannot collide even if their content hashes match.
///
/// # Serialization
///
/// When serializing [`Interned<T>`], subsequent occurrences of the same value
/// are encoded as references using this ID, significantly reducing serialized
/// size for duplicate data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InternedID {
    stable_type_id: StableTypeID,
    hash_128: Compact128,
}

/// A reference-counted handle to an interned value with transparent access.
///
/// `Interned<T>` wraps an `Arc<T>` and provides seamless access to the
/// underlying value through the [`Deref`] trait. When multiple `Interned<T>`
/// instances contain equal values (as determined by stable hash), they share
/// the same underlying allocation when created through the same [`Interner`].
///
/// # Unsized Type Support
///
/// `Interned<T>` supports unsized types like `str` and `[T]`. Use
/// [`Interner::intern_unsized`] to create `Interned<str>` or `Interned<[T]>`
/// values. This avoids the extra layer of indirection from wrapper types like
/// `String` or `Vec<T>`.
///
/// # Memory Management
///
/// - **Automatic deallocation**: The value is freed when all `Interned<T>`
///   handles are dropped
/// - **Weak references**: The interner only holds weak references, allowing
///   garbage collection of unused values
/// - **No memory leaks**: Values without active handles are automatically
///   reclaimed
///
/// # Cloning
///
/// Cloning is extremely cheap (O(1)) as it only increments the reference count:
/// ```ignore
/// let s1 = interner.intern("data".to_string());
/// let s2 = s1.clone();  // Fast: just increments refcount
/// ```
///
/// # Comparison with `Arc<T>`
///
/// | Aspect | `Arc<T>` | `Interned<T>` |
/// |--------|----------|---------------|
/// | Deduplication | No (each `Arc::new` creates new allocation) | Yes (equal values share allocation) |
/// | Overhead | Reference count only | Hash computation + lookup |
/// | Memory usage | Higher for duplicates | Lower for duplicates |
/// | Creation speed | Faster | Slower (requires hashing) |
/// | Cross-execution stability | No | Yes (with stable hashing) |
///
/// # Serialization
///
/// When using [`Encode`]/[`Decode`] traits, `Interned<T>` automatically
/// deduplicates values:
/// - **First occurrence**: Serialized in full
/// - **Subsequent occurrences**: Serialized as hash reference only
///
/// This dramatically reduces serialized size when many duplicate values exist.
///
/// # Example
///
/// ```ignore
/// // Create interned values
/// let s1 = interner.intern("hello".to_string());
/// let s2 = interner.intern("hello".to_string());
///
/// // They share the same allocation
/// assert!(Arc::ptr_eq(&s1.0, &s2.0));
///
/// // Transparent access
/// assert_eq!(&*s1, "hello");
/// assert_eq!(s1.len(), 5);
///
/// // Get owned copy if needed
/// let owned: String = s1.clone_inner();
/// ```
#[derive(
    Debug, PartialEq, Eq, PartialOrd, Ord, Hash, StableHash, Identifiable,
)]
#[stable_hash_crate(qbice_stable_hash)]
#[stable_type_id_crate(qbice_stable_type_id)]
pub struct Interned<T: ?Sized>(Arc<T>);

impl<T: ?Sized> Interned<T> {
    /// Creates a new `Interned<T>` by wrapping the given value in an `Arc`.
    ///
    /// This method doesn't guarantee deduplication. To intern values with
    /// deduplication, use the [`Interner::intern`] or
    /// [`Interner::intern_unsized`]
    pub fn new_duplicating(value: T) -> Self
    where
        T: Sized,
    {
        Self(Arc::new(value))
    }

    /// Creates a new `Interned<T>` for an unsized type by wrapping the given
    /// value in an `Arc`.
    ///
    /// This method doesn't guarantee deduplication. To intern values with
    /// deduplication, use the [`Interner::intern_unsized`]
    pub fn new_duplicating_unsized<U>(value: U) -> Self
    where
        U: Into<Arc<T>>,
    {
        Self(value.into())
    }
}

impl<T: ?Sized> Clone for Interned<T> {
    fn clone(&self) -> Self { Self(self.0.clone()) }
}

impl<T> Interned<T> {
    /// Returns an owned clone of the interned value.
    #[must_use]
    pub fn clone_inner(&self) -> T
    where
        T: Clone,
    {
        self.0.as_ref().clone()
    }
}

impl<T: ?Sized> Deref for Interned<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target { &self.0 }
}

impl<T: ?Sized> AsRef<T> for Interned<T> {
    fn as_ref(&self) -> &T { &self.0 }
}

impl<T: ?Sized> Borrow<T> for Interned<T> {
    fn borrow(&self) -> &T { &self.0 }
}

enum WiredInterned<T> {
    Source(T),
    Reference(Compact128),
}

impl<T: Encode> Encode for WiredInterned<T> {
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        plugin: &Plugin,
        session: &mut Session,
    ) -> std::io::Result<()> {
        match self {
            Self::Source(value) => {
                encoder.emit_u8(0)?;
                value.encode(encoder, plugin, session)
            }
            Self::Reference(hash) => {
                encoder.emit_u8(1)?;
                hash.encode(encoder, plugin, session)
            }
        }
    }
}

impl<T: Decode> Decode for WiredInterned<T> {
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        plugin: &Plugin,
        session: &mut Session,
    ) -> std::io::Result<Self> {
        let tag = decoder.read_u8()?;
        match tag {
            0 => {
                let value = T::decode(decoder, plugin, session)?;
                Ok(Self::Source(value))
            }
            1 => {
                let hash = Compact128::decode(decoder, plugin, session)?;
                Ok(Self::Reference(hash))
            }

            _ => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid tag for WiredInterned",
            )),
        }
    }
}

impl Decode for Interned<str> {
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        plugin: &Plugin,
        session: &mut Session,
    ) -> std::io::Result<Self> {
        let wired =
            WiredInterned::<Box<str>>::decode(decoder, plugin, session)?;

        let interner = plugin.get::<Interner>().expect(
            "`SharedInterner` plugin missing for decoding `Interned<str>`",
        );

        let value = match wired {
            WiredInterned::Source(source) => interner.intern_unsized(source),

            WiredInterned::Reference(compact128) => interner
                .get_from_hash::<str>(compact128)
                .expect("referenced interned value not found in interner"),
        };

        Ok(value)
    }
}

impl<T: Decode + StableHash + Identifiable + Send + Sync + 'static> Decode
    for Interned<[T]>
{
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        plugin: &Plugin,
        session: &mut Session,
    ) -> std::io::Result<Self> {
        let wired =
            WiredInterned::<Box<[T]>>::decode(decoder, plugin, session)?;

        let interner = plugin.get::<Interner>().expect(
            "`SharedInterner` plugin missing for decoding `Interned<[T]>`",
        );

        let value = match wired {
            WiredInterned::Source(source) => interner.intern_unsized(source),

            WiredInterned::Reference(compact128) => interner
                .get_from_hash::<[T]>(compact128)
                .expect("referenced interned value not found in interner"),
        };

        Ok(value)
    }
}

impl Decode for Interned<Path> {
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        plugin: &Plugin,
        session: &mut Session,
    ) -> std::io::Result<Self> {
        let wired =
            WiredInterned::<Box<Path>>::decode(decoder, plugin, session)?;

        let interner = plugin.get::<Interner>().expect(
            "`SharedInterner` plugin missing for decoding `Interned<Path>`",
        );

        let value = match wired {
            WiredInterned::Source(source) => interner.intern_unsized(source),

            WiredInterned::Reference(compact128) => interner
                .get_from_hash::<Path>(compact128)
                .expect("referenced interned value not found in interner"),
        };

        Ok(value)
    }
}

/// A session key for tracking seen interned IDs during encoding.
struct SeenInterned;

impl SessionKey for SeenInterned {
    type Value = FxHashSet<InternedID>;
}

impl<T: Identifiable + StableHash + Encode + Send + Sync + 'static + ?Sized>
    Encode for Interned<T>
{
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        plugin: &Plugin,
        session: &mut Session,
    ) -> std::io::Result<()> {
        let interner = plugin.get::<Interner>().expect(
            "`SharedInterner` plugin missing for encoding `Interned<T>`",
        );

        let value: &T = &self.0;
        let compact_128 = interner.hash_128(value);

        let seen_interned = session.get_mut_or_default::<SeenInterned>();
        let first = seen_interned.insert(InternedID {
            stable_type_id: T::STABLE_TYPE_ID,
            hash_128: compact_128,
        });

        if first {
            // serialize the full value
            encoder.emit_u8(0)?;
            value.encode(encoder, plugin, session)
        } else {
            // serialize only the reference
            encoder.emit_u8(1)?;
            compact_128.encode(encoder, plugin, session)
        }
    }
}

impl<T: Identifiable + StableHash + Decode + Send + Sync + 'static> Decode
    for Interned<T>
{
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        plugin: &Plugin,
        session: &mut Session,
    ) -> std::io::Result<Self> {
        let wired = WiredInterned::<T>::decode(decoder, plugin, session)?;
        let interner = plugin.get::<Interner>().expect(
            "`SharedInterner` plugin missing for decoding `Interned<T>`",
        );

        let value = match wired {
            WiredInterned::Source(source) => interner.intern(source),

            WiredInterned::Reference(compact128) => interner
                .get_from_hash::<T>(compact128)
                .expect("referenced interned value not found in interner"),
        };

        Ok(value)
    }
}

fn stable_hash<H: BuildStableHasher + 'static>(
    build_hasher: &dyn Any,
    value: &dyn DynStableHash,
) -> u128
where
    <H as BuildStableHasher>::Hasher: StableHasher<Hash = u128>,
{
    let build_hasher =
        build_hasher.downcast_ref::<H>().expect("invalid hasher type");

    let mut hasher = build_hasher.build_stable_hasher();
    value.stable_hash_dyn(&mut hasher);
    hasher.finish()
}

trait DynStableHash {
    fn stable_hash_dyn(&self, hasher: &mut dyn StableHasher<Hash = u128>);
}

impl<T: StableHash + ?Sized> DynStableHash for T {
    fn stable_hash_dyn(&self, hasher: &mut dyn StableHasher<Hash = u128>) {
        StableHash::stable_hash(self, hasher);
    }
}

struct Shard {
    typed_shard: Box<dyn Any + Send + Sync>,
    vacuum_fn: fn(&dyn Any),
}

type TypedShard<T> = Sharded<HashMap<Compact128, Weak<T>, FxBuildHasher>>;
type WholeShard = Sharded<HashMap<StableTypeID, Arc<Shard>, FxBuildHasher>>;

struct Repr {
    shards: WholeShard,

    hasher_builder_erased: Box<dyn Any + Send + Sync>,
    stable_hash_fn: fn(&dyn Any, &dyn DynStableHash) -> u128,

    /// Sender to signal the vacuum thread.
    vacuum_command: Option<Sender<VacuumCommand>>,
}

/// A thread-safe interner for deduplicating values based on stable hashing.
///
/// The `Interner` uses sharding to enable high-concurrency access from multiple
/// threads with minimal contention. Values are identified by their 128-bit
/// stable hash combined with type ID, ensuring safe and deterministic
/// deduplication.
///
/// # How It Works
///
/// 1. **Hash**: When interning a value, compute its stable 128-bit hash
/// 2. **Lookup**: Check if a value with that hash already exists in the
///    appropriate shard
/// 3. **Reuse or Store**:
///    - If found and alive: return handle to existing value
///    - Otherwise: store new value and return handle to it
///
/// # Sharding Architecture
///
/// ```text
/// Interner
///   ├─ Shard 0: HashMap<InternedID, Weak<T>>
///   ├─ Shard 1: HashMap<InternedID, Weak<T>>
///   ├─ ...
///   └─ Shard N: HashMap<InternedID, Weak<T>>
/// ```
///
/// Each shard:
/// - Has its own lock (reducing contention)
/// - Handles a subset of hash values
/// - Can be accessed in parallel with other shards
///
/// # Weak References and Garbage Collection
///
/// The interner stores [`Weak`] references, not strong references:
/// - Values are kept alive only by [`Interned<T>`] handles
/// - When all handles are dropped, the value is deallocated
/// - Subsequent intern calls for the same value create a new allocation
/// - Dead weak references can be cleaned up via [`vacuum`](Self::vacuum) or
///   automatically by a background vacuum thread
///
/// This prevents memory leaks from long-lived interners accumulating values.
///
/// # Background Vacuum Thread
///
/// The interner can optionally run a background thread that periodically
/// removes dead weak references from the shards. Use
/// [`new_with_vacuum`](Self::new_with_vacuum) to create an interner with
/// automatic cleanup:
///
/// ```ignore
/// use std::time::Duration;
///
/// // Vacuum every 60 seconds
/// let interner = Interner::new_with_vacuum(
///     16,
///     BuildStableHasher128::default(),
///     Duration::from_secs(60),
/// );
///
/// // Or trigger vacuum manually
/// interner.vacuum();
/// ```
///
/// # Thread Safety
///
/// Fully thread-safe and can be shared using `.clone()` directly. Multiple
/// threads can safely call [`intern`](Self::intern) concurrently. Sharding
/// ensures that operations on different hash buckets don't block each other.
///
/// # Performance Characteristics
///
/// - **Intern**: O(1) expected, requires hashing entire value
/// - **Lookup**: O(1) expected, requires hashing entire value
/// - **Memory**: O(unique values + shard overhead)
/// - **Concurrency**: Excellent when shards ≥ typical concurrent threads
///
/// # Example
///
/// ```ignore
/// use qbice_stable_hash::BuildStableHasher128;
///
/// let interner = Interner::new(16, BuildStableHasher128::default());
///
/// // Intern values
/// let v1 = interner.intern("data".to_string());
/// let v2 = interner.intern("data".to_string());
///
/// // Same allocation
/// assert!(Arc::ptr_eq(&v1.0, &v2.0));
///
/// // When all handles drop, value is freed
/// drop(v1);
/// drop(v2);
/// // "data" is now deallocated
/// ```
#[derive(Debug, Clone)]
pub struct Interner {
    inner: Arc<Repr>,
    /// Handle to the vacuum thread (if any). Kept for lifetime management.
    _vacuum_thread: Option<Arc<VacuumThreadHandle>>,
}

/// Command sent to the vacuum thread.
enum VacuumCommand {
    /// Trigger an immediate vacuum cycle.
    Trigger,
    /// Stop the vacuum thread.
    Stop,
}

/// Handle to manage the vacuum thread's lifecycle.
struct VacuumThreadHandle {
    command_sender: Sender<VacuumCommand>,
    join_handle: Option<JoinHandle<()>>,
}

impl std::fmt::Debug for VacuumThreadHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VacuumThreadHandle").finish_non_exhaustive()
    }
}

impl Drop for VacuumThreadHandle {
    fn drop(&mut self) {
        // Signal the thread to stop
        let _ = self.command_sender.send(VacuumCommand::Stop);

        // Wait for the thread to finish
        if let Some(handle) = self.join_handle.take() {
            let _ = handle.join();
        }
    }
}

impl Interner {
    /// Creates a new shared interner with the specified sharding and hasher.
    ///
    /// This creates an interner without a background vacuum thread. Dead weak
    /// references will accumulate until you call [`vacuum`](Self::vacuum)
    /// manually or until they are replaced by new intern calls.
    ///
    /// For automatic cleanup, use [`new_with_vacuum`](Self::new_with_vacuum).
    ///
    /// # Parameters
    ///
    /// - `shard_amount`: The number of shards for concurrent access. More
    ///   shards reduce contention but increase memory overhead. Recommended:
    ///   16-32 for multi-threaded use, or the number of CPU cores.
    /// - `hasher_builder`: The builder for creating stable hashers used to hash
    ///   interned values. Must produce deterministic 128-bit hashes.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use qbice_stable_hash::BuildStableHasher128;
    ///
    /// let interner = Interner::new(16, BuildStableHasher128::default());
    /// ```
    pub fn new<S: BuildStableHasher<Hash = u128> + Send + Sync + 'static>(
        shard_amount: usize,
        hasher_builder: S,
    ) -> Self {
        Self {
            inner: Arc::new(Repr::new(shard_amount, hasher_builder)),
            _vacuum_thread: None,
        }
    }

    /// Creates a new shared interner with a background vacuum thread.
    ///
    /// The vacuum thread periodically removes dead weak references from the
    /// interner's shards, freeing up memory from entries that are no longer
    /// referenced.
    ///
    /// # Parameters
    ///
    /// - `shard_amount`: The number of shards for concurrent access. More
    ///   shards reduce contention but increase memory overhead. Recommended:
    ///   16-32 for multi-threaded use, or the number of CPU cores.
    /// - `hasher_builder`: The builder for creating stable hashers used to hash
    ///   interned values. Must produce deterministic 128-bit hashes.
    /// - `vacuum_interval`: The duration between automatic vacuum runs. The
    ///   vacuum thread will sleep for this duration between cleanup cycles.
    ///
    /// # Thread Behavior
    ///
    /// - The vacuum thread is automatically stopped when the interner is
    ///   dropped
    /// - You can still call [`vacuum`](Self::vacuum) manually for immediate
    ///   cleanup
    /// - The vacuum process acquires write locks on shards, so very frequent
    ///   vacuuming may impact performance
    ///
    /// # Recommended Intervals
    ///
    /// - Low memory pressure: 5-10 minutes
    /// - Moderate usage: 30-60 seconds
    /// - High churn (many short-lived values): 5-15 seconds
    ///
    /// # Example
    ///
    /// ```ignore
    /// use std::time::Duration;
    /// use qbice_stable_hash::BuildStableHasher128;
    ///
    /// // Create interner with vacuum every 30 seconds
    /// let interner = Interner::new_with_vacuum(
    ///     16,
    ///     BuildStableHasher128::default(),
    ///     Duration::from_secs(30),
    /// );
    /// ```
    pub fn new_with_vacuum<
        S: BuildStableHasher<Hash = u128> + Send + Sync + 'static,
    >(
        shard_amount: usize,
        hasher_builder: S,
        vacuum_interval: Duration,
    ) -> Self {
        let (command_tx, command_rx) = unbounded::<VacuumCommand>();

        let repr = Arc::new(Repr::new_with_vacuum(
            shard_amount,
            hasher_builder,
            command_tx.clone(),
        ));

        let repr_weak = Arc::downgrade(&repr);

        let join_handle = std::thread::Builder::new()
            .name("interner-vacuum".to_string())
            .spawn(move || {
                vacuum_thread_loop(&repr_weak, &command_rx, vacuum_interval);
            })
            .expect("failed to spawn vacuum thread");

        Self {
            inner: repr,
            _vacuum_thread: Some(Arc::new(VacuumThreadHandle {
                command_sender: command_tx,
                join_handle: Some(join_handle),
            })),
        }
    }

    /// Manually triggers a vacuum operation to clean up dead weak references.
    ///
    /// This method removes all dead [`Weak`] references from the interner's
    /// internal hash maps. Dead references occur when all [`Interned<T>`]
    /// handles for a value have been dropped.
    ///
    /// # When to Use
    ///
    /// - After a large batch of interned values has been dropped
    /// - Before serializing the interner state (to reduce size)
    /// - In memory-constrained environments
    /// - When not using a background vacuum thread
    ///
    /// # Performance
    ///
    /// - Acquires write locks on all shards sequentially
    /// - Time complexity: O(total entries across all shards)
    /// - May briefly block other intern/lookup operations
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Intern many temporary values
    /// for i in 0..10000 {
    ///     let _ = interner.intern(format!("temp_{}", i));
    /// }
    /// // All Interned handles dropped, but weak refs remain
    ///
    /// // Clean up dead references
    /// interner.vacuum();
    /// ```
    pub fn vacuum(&self) { vacuum_shards(&self.inner.shards); }

    /// Requests the background vacuum thread to run immediately.
    ///
    /// If the interner was created with
    /// [`new_with_vacuum`](Self::new_with_vacuum), this signals the
    /// background thread to perform a vacuum cycle as soon as
    /// possible, rather than waiting for the next scheduled interval.
    ///
    /// If no background vacuum thread exists (interner created with
    /// [`new`](Self::new)), this method does nothing.
    ///
    /// # Non-blocking
    ///
    /// This method returns immediately after signaling the thread. It does not
    /// wait for the vacuum operation to complete. If you need synchronous
    /// cleanup, use [`vacuum`](Self::vacuum) instead.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Request early vacuum (non-blocking)
    /// interner.request_vacuum();
    ///
    /// // Or use vacuum() for synchronous cleanup
    /// interner.vacuum();
    /// ```
    pub fn request_vacuum(&self) {
        if let Some(sender) = &self.inner.vacuum_command {
            // Send trigger command; unbounded channel won't block
            let _ = sender.send(VacuumCommand::Trigger);
        }
    }

    /// Computes the 128-bit stable hash of a value.
    ///
    /// This method uses the interner's configured hasher to compute a
    /// deterministic hash. The same value will always produce the same hash
    /// within this interner instance (and across instances using the same
    /// hasher configuration).
    ///
    /// # Parameters
    ///
    /// * `value` - A reference to the value to hash. Must implement
    ///   [`StableHash`].
    ///
    /// # Returns
    ///
    /// A [`Compact128`] representing the 128-bit stable hash of the value.
    ///
    /// # Use Cases
    ///
    /// This method is primarily used internally but can be useful for:
    /// - Computing hashes for caching or indexing
    /// - Debugging hash collisions
    /// - Implementing custom interning strategies
    ///
    /// # Example
    ///
    /// ```ignore
    /// let hash1 = interner.hash_128(&"hello".to_string());
    /// let hash2 = interner.hash_128(&"hello".to_string());
    /// assert_eq!(hash1, hash2);  // Same value, same hash
    ///
    /// let hash3 = interner.hash_128(&"world".to_string());
    /// assert_ne!(hash1, hash3);  // Different value, different hash
    /// ```
    pub fn hash_128<T: StableHash + ?Sized>(&self, value: &T) -> Compact128 {
        let hash_u128 = (self.inner.stable_hash_fn)(
            &*self.inner.hasher_builder_erased,
            &value,
        );

        hash_u128.into()
    }

    fn obtain_read_shard<T: Identifiable + Send + Sync + 'static + ?Sized>(
        &self,
    ) -> Arc<Shard> {
        let shard_amount = self.inner.shards.shard_amount();
        let stable_type_id = T::STABLE_TYPE_ID;

        loop {
            let shard_index =
                self.inner.shards.shard_index(stable_type_id.low());
            let shard = self.inner.shards.read_shard(shard_index);

            // fast path, obtain read lock if shard exists
            if let Ok(exist) =
                RwLockReadGuard::try_map(shard, |x| x.get(&stable_type_id))
            {
                return exist.clone();
            }

            // slow path, need to create the shard, the `shard` has already
            // dropped here
            let mut write_shard = self.inner.shards.write_shard(shard_index);

            if let Entry::Vacant(entry) = write_shard.entry(stable_type_id) {
                entry.insert(Arc::new(Shard {
                    typed_shard: {
                        Box::new(TypedShard::<T>::new(shard_amount, |_| {
                            HashMap::default()
                        }))
                    },
                    vacuum_fn: vacuum_shard::<T>,
                }));
            }
        }
    }

    /// Retrieves an interned value by its 128-bit hash, if it exists and is
    /// alive.
    ///
    /// This method looks up a value that was previously interned and still has
    /// at least one active [`Interned<T>`] handle. It's primarily used during
    /// deserialization to resolve hash references back to actual values.
    ///
    /// # Parameters
    ///
    /// * `hash_128` - The 128-bit stable hash of the value to retrieve.
    ///
    /// # Type Parameter
    ///
    /// * `T` - The expected type of the value. Must match the type that was
    ///   originally interned.
    ///
    /// # Returns
    ///
    /// * `Some(Interned<T>)` - If a value with this hash exists, is still alive
    ///   (has active handles), and has the correct type
    /// * `None` - If:
    ///   - No value with this hash exists
    ///   - The value has been deallocated (no active handles)
    ///   - Type mismatch (requested type doesn't match interned type)
    ///
    /// # Type Safety
    ///
    /// The method uses [`StableTypeID`] to ensure type safety. Attempting to
    /// retrieve a value with the wrong type returns `None`, preventing type
    /// confusion even if hash values match.
    ///
    /// # Serialization Context
    ///
    /// During serialization, [`Interned<T>`] values are encoded as:
    /// 1. **First occurrence**: Full value + hash
    /// 2. **Subsequent occurrences**: Hash reference only
    ///
    /// During deserialization, hash references are resolved using this method
    /// to reconstruct the original value structure.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Intern a value
    /// let original = interner.intern("data".to_string());
    /// let hash = interner.hash_128(&*original);
    ///
    /// // Retrieve by hash
    /// let retrieved = interner.get_from_hash::<String>(hash)
    ///     .expect("value should still be alive");
    ///
    /// assert!(Arc::ptr_eq(&original.0, &retrieved.0));
    ///
    /// // After all handles drop, lookup returns None
    /// drop(original);
    /// drop(retrieved);
    /// assert!(interner.get_from_hash::<String>(hash).is_none());
    /// ```
    #[must_use]
    pub fn get_from_hash<T: Identifiable + Send + Sync + 'static + ?Sized>(
        &self,
        hash_128: Compact128,
    ) -> Option<Interned<T>> {
        let read_shard = self.obtain_read_shard::<T>();
        let typed_shard = read_shard
            .typed_shard
            .downcast_ref::<TypedShard<T>>()
            .expect("should be correct type");

        let shard_index = typed_shard.shard_index(hash_128.low());
        let read_lock = typed_shard.read_shard(shard_index);

        read_lock
            .get(&hash_128)
            .and_then(std::sync::Weak::upgrade)
            .map(|arc| Interned(arc))
    }

    /// Interns a value, returning a reference-counted handle to the shared
    /// allocation.
    ///
    /// If an equal value (as determined by stable hash) has already been
    /// interned and is still alive, this method returns a handle to the
    /// existing allocation. Otherwise, it stores the new value and returns a
    /// handle to it.
    ///
    /// # Equality Semantics
    ///
    /// Values are considered equal if:
    /// 1. They have the same type (same [`StableTypeID`])
    /// 2. They have the same stable hash (128-bit)
    ///
    /// **Important**: This uses **hash equality**, not structural equality
    /// (`PartialEq`). While 128-bit hash collisions are astronomically
    /// unlikely, they are theoretically possible. In practice, this is not a
    /// concern for most applications.
    ///
    /// # Type Parameters
    ///
    /// * `T` - The type of value to intern. Must implement:
    ///   - [`StableHash`]: For computing the deterministic content hash
    ///   - [`Identifiable`]: For the stable type ID
    ///   - `Send + Sync + 'static`: For safe sharing across threads
    ///
    /// # Parameters
    ///
    /// * `value` - The value to intern. Takes ownership.
    ///
    /// # Returns
    ///
    /// An [`Interned<T>`] handle to the interned value. If an equal value was
    /// already interned, the returned handle points to the existing allocation.
    ///
    /// # Thread Safety
    ///
    /// This method is fully thread-safe and can be called concurrently:
    /// - If two threads intern the same value simultaneously, only one
    ///   allocation is created
    /// - Both threads receive handles to the same allocation
    /// - The losing thread's value is dropped
    ///
    /// # Performance
    ///
    /// * **Time complexity**: O(1) expected, O(n) worst case (hash table)
    /// * **Hashing cost**: O(size of value) - entire value is hashed
    /// * **Lock contention**: Only affects the specific shard for this hash
    /// * **Memory**: Reuses existing allocation if value already interned
    ///
    /// # Trade-offs
    ///
    /// **Slower than `Arc::new`** because:
    /// - Must hash the entire value
    /// - Requires hash table lookup
    /// - May need to acquire locks
    ///
    /// **Saves memory when**:
    /// - Many duplicate values exist
    /// - Values are large
    /// - Long-lived values with many references
    ///
    /// # Example
    ///
    /// ```ignore
    /// // First intern - creates new allocation
    /// let s1 = interner.intern("hello".to_string());
    ///
    /// // Second intern - reuses existing allocation
    /// let s2 = interner.intern("hello".to_string());
    ///
    /// // Same pointer, same allocation
    /// assert!(Arc::ptr_eq(&s1.0, &s2.0));
    ///
    /// // Different value - new allocation
    /// let s3 = interner.intern("world".to_string());
    /// assert!(!Arc::ptr_eq(&s1.0, &s3.0));
    /// ```
    pub fn intern<T: StableHash + Identifiable + Send + Sync + 'static>(
        &self,
        value: T,
    ) -> Interned<T> {
        let hash_128 = self.hash_128(&value);
        let read_shard = self.obtain_read_shard::<T>();
        let typed_shard = read_shard
            .typed_shard
            .downcast_ref::<TypedShard<T>>()
            .expect("should be correct type");

        let shard_index = typed_shard.shard_index(hash_128.low());

        // First, try to find an existing interned value
        {
            let read_shard = typed_shard.read_shard(shard_index);

            if let Some(arc) =
                read_shard.get(&hash_128).and_then(std::sync::Weak::upgrade)
            {
                return Interned(arc);
            }
        }

        // Not found, so insert a new one
        {
            let mut write_shard = typed_shard.write_shard(shard_index);

            match write_shard.entry(hash_128) {
                // double check in case another thread inserted it
                std::collections::hash_map::Entry::Occupied(mut entry) => {
                    if let Some(arc) = entry.get().upgrade() {
                        return Interned(arc);
                    }

                    // The weak reference is dead, we can replace it
                    let arc = Arc::new(value);
                    let weak = Arc::downgrade(&arc);

                    entry.insert(weak);

                    Interned(arc)
                }

                std::collections::hash_map::Entry::Vacant(entry) => {
                    let arc = Arc::new(value);
                    let weak = Arc::downgrade(&arc);

                    entry.insert(weak);

                    Interned(arc)
                }
            }
        }
    }

    /// Interns an unsized value, returning a reference-counted handle to the
    /// shared allocation.
    ///
    /// This method is similar to [`intern`](Self::intern) but supports unsized
    /// types like `str` and `[T]`. The value is provided as a sized type `Q`
    /// that can be borrowed as `T` and converted into `Arc<T>`.
    ///
    /// # Type Parameters
    ///
    /// * `T` - The unsized type to intern (e.g., `str`, `[u8]`). Must
    ///   implement:
    ///   - [`StableHash`]: For computing the deterministic content hash
    ///   - [`Identifiable`]: For the stable type ID
    ///   - `Send + Sync + 'static`: For safe sharing across threads
    ///
    /// * `Q` - The sized type that owns the value. Must implement:
    ///   - [`Borrow<T>`]: To access the unsized value for hashing
    ///   - `Send + Sync + 'static`: For safe sharing across threads
    ///   - `Arc<T>: From<Q>`: For efficient conversion to `Arc<T>`
    ///
    /// # Common Type Combinations
    ///
    /// | Unsized `T` | Sized `Q` options |
    /// |-------------|-------------------|
    /// | `str` | `String`, `Box<str>`, `Cow<'static, str>` |
    /// | `[u8]` | `Vec<u8>`, `Box<[u8]>` |
    /// | `[T]` | `Vec<T>`, `Box<[T]>` |
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Intern strings as str slices
    /// let s1: Interned<str> = interner.intern_unsized("hello".to_string());
    /// let s2: Interned<str> = interner.intern_unsized("hello".to_string());
    ///
    /// // Same allocation
    /// assert!(Arc::ptr_eq(&s1.0, &s2.0));
    ///
    /// // Transparent access via Deref
    /// assert_eq!(&*s1, "hello");
    /// assert_eq!(s1.len(), 5);
    ///
    /// // Works with byte slices too
    /// let bytes: Interned<[u8]> = interner.intern_unsized(vec![1, 2, 3]);
    /// assert_eq!(&*bytes, &[1, 2, 3]);
    /// ```
    ///
    /// # Performance
    ///
    /// Same as [`intern`](Self::intern) - O(1) expected time with hashing
    /// overhead proportional to value size.
    pub fn intern_unsized<
        T: StableHash + Identifiable + Send + Sync + 'static + ?Sized,
        Q: Borrow<T> + Send + Sync + 'static,
    >(
        &self,
        value: Q,
    ) -> Interned<T>
    where
        Arc<T>: From<Q>,
    {
        let hash_128 = self.hash_128(value.borrow());
        let read_shard = self.obtain_read_shard::<T>();
        let typed_shard = read_shard
            .typed_shard
            .downcast_ref::<TypedShard<T>>()
            .expect("should be correct type");

        let shard_index = typed_shard.shard_index(hash_128.low());

        // First, try to find an existing interned value
        {
            let read_shard = typed_shard.read_shard(shard_index);

            if let Some(arc) =
                read_shard.get(&hash_128).and_then(std::sync::Weak::upgrade)
            {
                return Interned(arc);
            }
        }

        // Not found, so insert a new one
        {
            let mut write_shard = typed_shard.write_shard(shard_index);

            match write_shard.entry(hash_128) {
                // double check in case another thread inserted it
                std::collections::hash_map::Entry::Occupied(mut entry) => {
                    if let Some(arc) = entry.get().upgrade() {
                        return Interned(arc);
                    }

                    // The weak reference is dead, we can replace it
                    let arc = Arc::from(value);
                    let weak = Arc::downgrade(&arc);

                    entry.insert(weak);

                    Interned(arc)
                }

                std::collections::hash_map::Entry::Vacant(entry) => {
                    let arc = Arc::from(value);
                    let weak = Arc::downgrade(&arc);

                    entry.insert(weak);

                    Interned(arc)
                }
            }
        }
    }
}

impl std::fmt::Debug for Repr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Interner").finish_non_exhaustive()
    }
}

#[allow(clippy::cast_precision_loss)]
fn vacuum_shard<T: ?Sized + 'static>(shard: &dyn Any) {
    let typed_shard = shard
        .downcast_ref::<TypedShard<T>>()
        .expect("invalid shard type for vacuum");

    for mut write_shard in typed_shard.iter_write_shards() {
        write_shard.retain(|_, weak_value| weak_value.upgrade().is_some());

        let ratio = write_shard.len() as f64 / write_shard.capacity() as f64;
        if ratio < 0.25 {
            let halved = write_shard.capacity() / 2;
            write_shard.shrink_to(halved);
        }
    }
}

/// Vacuum all shards in the interner, removing dead weak references.
fn vacuum_shards(shards: &WholeShard) {
    for read_shard in shards.iter_read_shards() {
        for shard in read_shard.values() {
            (shard.vacuum_fn)(&*shard.typed_shard);
        }
    }
}

/// The main loop for the background vacuum thread.
fn vacuum_thread_loop(
    repr_weak: &Weak<Repr>,
    command_rx: &Receiver<VacuumCommand>,
    vacuum_interval: Duration,
) {
    loop {
        // Wait for a command with timeout as the vacuum interval
        match command_rx.recv_timeout(vacuum_interval) {
            Ok(VacuumCommand::Stop)
            | Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                // Stop signal received or channel disconnected, exit the loop
                return;
            }
            Ok(VacuumCommand::Trigger)
            | Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                // Trigger signal received, run vacuum immediately
            }
        }

        // Try to upgrade the weak reference to run vacuum
        if let Some(repr) = repr_weak.upgrade() {
            vacuum_shards(&repr.shards);
        } else {
            // The interner has been dropped, exit the loop
            return;
        }
    }
}

impl Repr {
    /// Creates a new interner with specified sharding and stable hasher.
    ///
    /// # Parameters
    ///
    /// * `shard_amount` - The number of shards for concurrent access. Each
    ///   shard has its own lock, so more shards reduce contention at the cost
    ///   of memory overhead.
    ///
    ///   **Recommended values:**
    ///   - Single-threaded: 1-4 shards
    ///   - Multi-threaded (typical): 16-32 shards
    ///   - High contention: 64-128 shards
    ///   - Powers of 2 work best (efficient bit masking)
    ///
    /// * `hasher_builder` - Builder for creating stable hashers. Must produce
    ///   deterministic 128-bit hashes. The same input must always produce the
    ///   same hash across all program executions for serialization to work
    ///   correctly.
    ///
    /// # Panics
    ///
    /// Panics if `shard_amount` is not a power of two.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use qbice_stable_hash::BuildStableHasher128;
    ///
    /// // Create interner with 16 shards
    /// let interner = Interner::new(16, BuildStableHasher128::default());
    /// ```
    pub fn new<S: BuildStableHasher + Send + Sync + 'static>(
        shard_amount: usize,
        hasher_builder: S,
    ) -> Self
    where
        <S as BuildStableHasher>::Hasher: StableHasher<Hash = u128>,
    {
        Self {
            shards: Sharded::new(shard_amount, |_| HashMap::default()),
            hasher_builder_erased: Box::new(hasher_builder),
            stable_hash_fn: stable_hash::<S>,
            vacuum_command: None,
        }
    }

    /// Creates a new interner with vacuum thread support.
    pub fn new_with_vacuum<S: BuildStableHasher + Send + Sync + 'static>(
        shard_amount: usize,
        hasher_builder: S,
        vacuum_command: Sender<VacuumCommand>,
    ) -> Self
    where
        <S as BuildStableHasher>::Hasher: StableHasher<Hash = u128>,
    {
        Self {
            shards: Sharded::new(shard_amount, |_| HashMap::default()),
            hasher_builder_erased: Box::new(hasher_builder),
            stable_hash_fn: stable_hash::<S>,
            vacuum_command: Some(vacuum_command),
        }
    }
}

#[cfg(test)]
mod test;
