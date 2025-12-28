//! A thread-safe interning system for deduplicating values based on their
//! stable hash.
//!
//! Interning is a technique where equal values are stored only once in memory,
//! and all references to that value point to the same allocation. This module
//! provides a concurrent interner that uses stable hashing to identify equal
//! values across different program executions.
//!
//! # Overview
//!
//! The interning system consists of three main layers:
//!
//! - [`InternedID`]: A unique identifier combining a type's stable ID and
//!   content hash.
//! - [`Interned<T>`]: A reference-counted handle to an interned value that
//!   provides transparent access via [`Deref`].
//! - [`Interner`]: The core interner managing deduplicated values using sharded
//!   concurrent access.
//! - [`SharedInterner`]: A convenience wrapper around `Arc<Interner>` for easy
//!   sharing across threads.
//!
//! # When to Use Interning
//!
//! Interning is beneficial when:
//! - You have many duplicate values that are expensive to store
//! - Values are immutable and compared frequently
//! - You need deterministic serialization with deduplication
//! - Memory usage is more important than allocation speed
//!
//! Use regular `Arc`/`Rc` when:
//! - Values are rarely duplicated
//! - You don't need cross-execution stability
//! - Allocation performance is critical
//!
//! # Example
//!
//! ```ignore
//! use qbice_storage::intern::{Interner, SharedInterner};
//! use qbice_stable_hash::BuildStableHasher128;
//!
//! // Create a shared interner
//! let interner = SharedInterner::new(16, BuildStableHasher128::default());
//!
//! // Intern some values
//! let s1 = interner.intern("hello".to_string());
//! let s2 = interner.intern("hello".to_string());
//!
//! // s1 and s2 point to the same allocation
//! assert!(Arc::ptr_eq(&s1.0, &s2.0));
//! ```

use std::{
    any::Any,
    collections::HashMap,
    hash::Hash,
    ops::Deref,
    sync::{Arc, Weak},
};

use fxhash::FxHashSet;
use qbice_serialize::{
    Decode, Decoder, Encode, Encoder, Plugin,
    session::{Session, SessionKey},
};
use qbice_stable_hash::{
    BuildStableHasher, Compact128, StableHash, StableHasher,
};
use qbice_stable_type_id::{Identifiable, StableTypeID};

use crate::sharded::Sharded;

/// A unique identifier for an interned value.
///
/// This ID combines two components to uniquely identify an interned value:
/// - A [`StableTypeID`] that identifies the type of the value
/// - A 128-bit hash ([`Compact128`]) of the value's content
///
/// The combination of type ID and content hash ensures that values of
/// different types with the same hash are distinguishable, preventing type
/// confusion. For example, a string "42" and an integer `42` would have
/// different `InternedID`s even if their content hashes collided.
///
/// # Structure
///
/// ```text
/// InternedID {
///     stable_type_id: 0x1234_5678_...,  // Type's unique ID
///     hash_128: 0xabcd_ef01_...,        // 128-bit content hash  
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InternedID {
    stable_type_id: StableTypeID,
    hash_128: Compact128,
}

/// A reference-counted handle to an interned value.
///
/// `Interned<T>` wraps an `Arc<T>` and provides transparent access to the
/// underlying value through [`Deref`]. Multiple `Interned<T>` instances
/// containing equal values (as determined by their stable hash) will share
/// the same underlying allocation when created through the same [`Interner`].
///
/// # Memory Management
///
/// The interned value is automatically deallocated when all `Interned<T>`
/// handles to it are dropped. The interner only holds weak references,
/// allowing unused values to be garbage collected.
///
/// # Cloning
///
/// Cloning an `Interned<T>` is cheap (O(1)) as it only increments the
/// reference count of the underlying `Arc`. No data is copied.
///
/// # Comparison with Arc
///
/// Unlike `Arc<T>`, which creates a new allocation for each `Arc::new()` call,
/// `Interned<T>` ensures that equal values (by stable hash) share a single
/// allocation. This saves memory but requires hashing on creation.
///
/// # Serialization
///
/// When using the `Encode`/`Decode` traits, `Interned<T>` automatically
/// deduplicates values during serialization. The first occurrence of a value
/// is serialized in full; subsequent occurrences are serialized as references
/// to the hash, significantly reducing serialized size for duplicate data.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, StableHash)]
#[stable_hash_crate(qbice_stable_hash)]
pub struct Interned<T>(Arc<T>);

impl<T> Interned<T> {
    /// Returns an owned clone of the interned value.
    #[must_use]
    pub fn inner_owned(&self) -> T
    where
        T: Clone,
    {
        self.0.as_ref().clone()
    }
}

impl<T> Deref for Interned<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target { &self.0 }
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

/// A session key for tracking seen interned IDs during encoding.
struct SeenInterned;

impl SessionKey for SeenInterned {
    type Value = FxHashSet<InternedID>;
}

impl<T: Identifiable + StableHash + Encode + Send + Sync + 'static> Encode
    for Interned<T>
{
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        plugin: &Plugin,
        session: &mut Session,
    ) -> std::io::Result<()> {
        let interner = plugin.get::<SharedInterner>().expect(
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
        let interner = plugin.get::<SharedInterner>().expect(
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

impl<T: StableHash> DynStableHash for T {
    fn stable_hash_dyn(&self, hasher: &mut dyn StableHasher<Hash = u128>) {
        StableHash::stable_hash(self, hasher);
    }
}

/// A thread-safe interner for deduplicating values based on their stable hash.
///
/// The `Interner` uses sharding to allow concurrent access from multiple
/// threads with minimal contention. Values are identified by their 128-bit
/// stable hash combined with their type ID, ensuring that equal values of
/// the same type share a single allocation.
///
/// # How It Works
///
/// 1. When you intern a value, the interner computes its stable hash
/// 2. It checks if a value with that hash already exists
/// 3. If found and still alive, it returns a handle to the existing value
/// 4. Otherwise, it stores the new value and returns a handle to it
///
/// # Weak References
///
/// The interner stores weak references to interned values. When all
/// [`Interned<T>`] handles to a value are dropped, the value is automatically
/// deallocated. Subsequent intern calls for the same value will create a new
/// allocation. This prevents memory leaks for values no longer in use.
///
/// # Thread Safety
///
/// The interner is fully thread-safe and can be shared across threads using
/// `Arc<Interner>` (or the convenience wrapper [`SharedInterner`]). Multiple
/// threads can safely call [`intern`](Self::intern) concurrently. Sharding
/// ensures that operations on different hash buckets don't block each other.
///
/// # Choosing Shard Count
///
/// More shards reduce lock contention but increase memory overhead. Good
/// starting points:
/// - Single-threaded: 1-4 shards
/// - Multi-threaded: 16-32 shards (power of 2 recommended)
/// - High contention: Number of CPU cores or more
pub struct Interner {
    shards: Sharded<HashMap<InternedID, Weak<dyn Any + Send + Sync>>>,

    hasher_builder_erased: Box<dyn Any + Send + Sync>,
    stable_hash_fn: fn(&dyn Any, &dyn DynStableHash) -> u128,
}

/// A shared, reference-counted interner.
///
/// This is a convenience wrapper around `Arc<Interner>` that simplifies
/// sharing an interner across threads and components. It implements [`Deref`]
/// to provide transparent access to the underlying [`Interner`].
///
/// # When to Use
///
/// Use `SharedInterner` when:
/// - You need to share an interner across multiple components or threads
/// - You want a simpler API than manually wrapping in `Arc`
/// - You're using the interner as a plugin in the serialization system
///
/// Use `Interner` directly when:
/// - You're managing ownership manually
/// - The interner doesn't need to be shared
///
/// # Example
///
/// ```ignore
/// let interner = SharedInterner::new(16, hasher_builder);
///
/// // Clone is cheap - just increments reference count
/// let interner2 = interner.clone();
///
/// // Both refer to the same underlying interner
/// let v1 = interner.intern("test".to_string());
/// let v2 = interner2.intern("test".to_string());
/// assert!(Arc::ptr_eq(&v1.0, &v2.0));
/// ```
#[derive(Debug, Clone)]
pub struct SharedInterner(Arc<Interner>);

impl SharedInterner {
    /// Creates a new shared interner with the specified sharding and hasher.
    ///
    /// This is a convenience method equivalent to creating an [`Interner`] and
    /// wrapping it in an `Arc`.
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
    /// let interner = SharedInterner::new(16, BuildStableHasher128::default());
    /// ```
    pub fn new<S: BuildStableHasher<Hash = u128> + Send + Sync + 'static>(
        shard_amount: usize,
        hasher_builder: S,
    ) -> Self {
        Self(Arc::new(Interner::new(shard_amount, hasher_builder)))
    }

    /// Creates a shared interner from an existing interner.
    ///
    /// This wraps the provided [`Interner`] in an `Arc` to make it shareable.
    ///
    /// # Parameters
    ///
    /// - `interner`: The interner to wrap.
    #[must_use]
    pub fn from_interner(interner: Interner) -> Self {
        Self(Arc::new(interner))
    }
}

impl Deref for SharedInterner {
    type Target = Interner;

    fn deref(&self) -> &Self::Target { &self.0 }
}

impl std::fmt::Debug for Interner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Interner").finish_non_exhaustive()
    }
}

impl Interner {
    /// Creates a new interner with the specified sharding and hasher.
    ///
    /// # Parameters
    ///
    /// - `shard_amount`: The number of shards for concurrent access. Each shard
    ///   has its own lock, so more shards reduce contention. Good starting
    ///   values:
    ///   - Single-threaded: 1-4 shards
    ///   - Multi-threaded: 16-32 shards (powers of 2 work well)
    ///   - High contention: Number of CPU cores or 2× that
    ///
    /// - `hasher_builder`: The builder for creating stable hashers. Must
    ///   produce deterministic 128-bit hashes for the same input across all
    ///   program executions.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use qbice_stable_hash::BuildStableHasher128;
    ///
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
            shards: Sharded::new(shard_amount, |_| HashMap::new()),
            hasher_builder_erased: Box::new(hasher_builder),
            stable_hash_fn: stable_hash::<S>,
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
    /// - `value`: A reference to the value to hash.
    ///
    /// # Returns
    ///
    /// A [`Compact128`] representing the 128-bit hash of the value.
    pub fn hash_128<T: StableHash>(&self, value: &T) -> Compact128 {
        let hash_u128 =
            (self.stable_hash_fn)(&*self.hasher_builder_erased, value);

        hash_u128.into()
    }

    /// Retrieves an interned value by its 128-bit hash, if it exists.
    ///
    /// This method looks up a value that was previously interned and is still
    /// alive (has at least one [`Interned<T>`] handle). It's primarily used
    /// during deserialization when references to interned values need to be
    /// resolved.
    ///
    /// # Parameters
    ///
    /// - `hash_128`: The 128-bit hash of the value to retrieve.
    ///
    /// # Returns
    ///
    /// - `Some(Interned<T>)` if a value with this hash exists and is still
    ///   alive
    /// - `None` if no matching value is found or the value has been deallocated
    ///
    /// # Relationship to Serialization
    ///
    /// During serialization, `Interned<T>` values are encoded either as:
    /// 1. Full value (first occurrence)
    /// 2. Hash reference (subsequent occurrences)
    ///
    /// During deserialization, hash references are resolved using this method
    /// to point to the previously-deserialized instance.
    ///
    /// # Type Safety
    ///
    /// The method uses the type's [`StableTypeID`] to ensure type safety.
    /// Attempting to retrieve a value with the wrong type will return `None`,
    /// even if a value with that hash exists for a different type.
    pub fn get_from_hash<T: Identifiable + Send + Sync + 'static>(
        &self,
        hash_128: Compact128,
    ) -> Option<Interned<T>> {
        let shard_index = self.shards.shard_index(hash_128.low());

        let interned_id =
            InternedID { stable_type_id: T::STABLE_TYPE_ID, hash_128 };

        let read_shard = self.shards.read_shard(shard_index);

        if let Some(arc) = read_shard
            .get(&interned_id)
            .and_then(std::sync::Weak::upgrade)
            .map(|x| x.downcast::<T>().expect("should've been a correct type"))
        {
            return Some(Interned(arc));
        }

        None
    }

    /// Interns a value, returning a reference-counted handle to it.
    ///
    /// If an equal value (as determined by stable hash) has already been
    /// interned and is still alive, this method returns a handle to the
    /// existing value. Otherwise, it stores the new value and returns a handle
    /// to it.
    ///
    /// # Equality Semantics
    ///
    /// Values are considered equal if:
    /// 1. They have the same type (same `StableTypeID`)
    /// 2. They have the same stable hash (128-bit)
    ///
    /// Note that this uses **hash equality**, not value equality. Hash
    /// collisions are extremely rare with 128-bit hashes but theoretically
    /// possible.
    ///
    /// # Parameters
    ///
    /// - `value`: The value to intern. Must implement:
    ///   - [`StableHash`]: For computing the content hash
    ///   - [`Identifiable`]: For the stable type ID
    ///   - `Send + Sync + 'static`: For safe sharing across threads
    ///
    /// # Returns
    ///
    /// An [`Interned<T>`] handle to the interned value. If an equal value was
    /// already interned, the returned handle points to the existing allocation.
    ///
    /// # Thread Safety
    ///
    /// This method is thread-safe and can be called concurrently. If two
    /// threads intern the same value simultaneously, only one allocation
    /// will be created and both threads will receive handles to it.
    ///
    /// # Performance
    ///
    /// - **Time complexity**: O(1) expected, O(n) worst case due to hash table
    /// - **Hashing cost**: Requires computing a stable hash of the entire value
    /// - **Lock contention**: Only affects the specific shard for this hash
    pub fn intern<T: StableHash + Identifiable + Send + Sync + 'static>(
        &self,
        value: T,
    ) -> Interned<T> {
        let hash_128 = self.hash_128(&value);
        let shard_index = self.shards.shard_index(hash_128.low());

        let interned_id =
            InternedID { stable_type_id: T::STABLE_TYPE_ID, hash_128 };

        // First, try to find an existing interned value
        {
            let read_shard = self.shards.read_shard(shard_index);

            if let Some(arc) = read_shard
                .get(&interned_id)
                .and_then(std::sync::Weak::upgrade)
                .map(|x| {
                    x.downcast::<T>().expect("should've been a correct type")
                })
            {
                return Interned(arc);
            }
        }

        // Not found, so insert a new one
        {
            let mut write_shard = self.shards.write_shard(shard_index);

            match write_shard.entry(interned_id) {
                // double check in case another thread inserted it
                std::collections::hash_map::Entry::Occupied(mut entry) => {
                    if let Some(arc) = entry
                        .get()
                        .upgrade()
                        .and_then(|x| x.downcast::<T>().ok())
                    {
                        return Interned(arc);
                    }

                    // The weak reference is dead, we can replace it
                    let arc = Arc::new(value);
                    let weak = Arc::downgrade(&arc);
                    let weak = weak as Weak<dyn Any + Send + Sync>;

                    entry.insert(weak);

                    Interned(arc)
                }

                std::collections::hash_map::Entry::Vacant(entry) => {
                    let arc = Arc::new(value);
                    let weak = Arc::downgrade(&arc);
                    let weak = weak as Weak<dyn Any + Send + Sync>;

                    entry.insert(weak);

                    Interned(arc)
                }
            }
        }
    }
}

#[cfg(test)]
mod test;
