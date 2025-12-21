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
//! The interning system consists of three main components:
//!
//! - [`InternedID`]: A unique identifier for an interned value, combining the
//!   type's stable ID and its 128-bit hash.
//! - [`Interned<T>`]: A reference-counted handle to an interned value.
//! - [`Interner<S>`]: The main interner that manages interned values using
//!   sharding for concurrent access.
//!
//! # Example
//!
//! ```ignore
//! use qbice_storage::intern::Interner;
//!
//! let interner = Interner::new(16, SipHasher128Builder::default());
//!
//! let a = interner.intern("hello".to_string());
//! let b = interner.intern("hello".to_string());
//!
//! // Both `a` and `b` point to the same allocation
//! assert!(Arc::ptr_eq(&a.0, &b.0));
//! ```

use std::{
    any::Any,
    collections::HashMap,
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
/// different types with the same hash are still distinguishable, preventing
/// type confusion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InternedID {
    stable_type_id: StableTypeID,
    hash_128: Compact128,
}

/// A reference-counted handle to an interned value.
///
/// `Interned<T>` wraps an `Arc<T>` and provides transparent access to the
/// underlying value through [`Deref`]. Multiple `Interned<T>` instances
/// containing equal values will share the same underlying allocation when
/// created through the same [`Interner`].
///
/// # Memory Management
///
/// The interned value is automatically deallocated when all `Interned<T>`
/// handles to it are dropped. The interner only holds weak references to
/// interned values, so it does not prevent deallocation.
///
/// # Cloning
///
/// Cloning an `Interned<T>` is cheap as it only increments the reference
/// count of the underlying `Arc`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, StableHash)]
pub struct Interned<T>(Arc<T>);

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
/// # Type Parameters
///
/// - `S`: A [`BuildStableHasher`] implementation that produces 128-bit hashes.
///   This hasher must be deterministic to ensure consistent interning across
///   the interner's lifetime.
///
/// # Weak References
///
/// The interner stores weak references to interned values. This means that
/// when all [`Interned<T>`] handles to a value are dropped, the value is
/// deallocated and removed from the interner on the next lookup. This
/// prevents memory leaks for values that are no longer in use.
///
/// # Thread Safety
///
/// The interner is fully thread-safe and can be shared across threads using
/// `Arc<Interner<S>>`. Concurrent intern operations may block briefly on
/// the same shard but will not deadlock.
pub struct Interner {
    shards: Sharded<HashMap<InternedID, Weak<dyn Any + Send + Sync>>>,

    hasher_builder_erased: Box<dyn Any + Send + Sync>,
    stable_hash_fn: fn(&dyn Any, &dyn DynStableHash) -> u128,
}

/// A shared, reference-counted [`Interner<S>`].
///
/// See [`Interner<S>`] for details.
#[derive(Debug, Clone)]
pub struct SharedInterner(Arc<Interner>);

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
    /// Creates a new interner with the specified number of shards and hasher
    /// builder.
    ///
    /// # Parameters
    ///
    /// - `shard_amount`: The number of shards to use for concurrent access.
    ///   More shards reduce contention but increase memory overhead. A good
    ///   starting point is the number of CPU cores or a power of 2 like 16 or
    ///   32.
    /// - `hasher_builder`: The hasher builder used to create stable hashers for
    ///   computing value hashes.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let interner = Interner::new(16, SipHasher128Builder::default());
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
    /// This method uses the interner's hasher builder to compute a
    /// deterministic hash of the given value. The same value will always
    /// produce the same hash within the same interner instance.
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

    /// Attempts to retrieve an interned value by its 128-bit hash.
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
    /// If an equal value has already been interned and is still alive, this
    /// method returns a handle to the existing value. Otherwise, it stores
    /// the new value and returns a handle to it.
    ///
    /// # Parameters
    ///
    /// - `value`: The value to intern. Must implement [`StableHash`],
    ///   [`Identifiable`], [`Send`], [`Sync`], and have a `'static` lifetime.
    ///
    /// # Returns
    ///
    /// An [`Interned<T>`] handle to the interned value. If an equal value
    /// was already interned, this will point to the existing allocation.
    ///
    /// # Thread Safety
    ///
    /// This method is thread-safe and can be called concurrently from multiple
    /// threads. If two threads attempt to intern the same value simultaneously,
    /// only one allocation will be created and both threads will receive
    /// handles to it.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let interner = Interner::new(16, hasher_builder);
    ///
    /// let a = interner.intern("hello".to_string());
    /// let b = interner.intern("hello".to_string());
    ///
    /// // Both handles point to the same allocation
    /// assert!(std::ptr::eq(&*a, &*b));
    /// ```
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
