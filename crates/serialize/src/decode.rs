//! Decoding traits and implementations for deserialization.
//!
//! This module provides the [`Decoder`] trait for implementing custom decoders,
//! and the [`Decode`] trait for types that can be deserialized.

use std::{
    borrow::Cow,
    collections::{BTreeMap, BTreeSet, HashMap, HashSet, LinkedList, VecDeque},
    hash::{BuildHasher, Hash},
    io,
    rc::Rc,
    sync::Arc,
};

use dashmap::DashMap;

use crate::{plugin::Plugin, session::Session};

/// A trait for types that can deserialize primitive values from a binary
/// format.
///
/// Implementors of this trait provide the low-level deserialization of
/// primitive types. The default implementations delegate to other methods where
/// possible, but can be overridden for performance or format-specific reasons.
///
/// All methods return `Result<T, std::io::Error>` to allow for fallible I/O
/// operations.
///
/// # Example
///
/// ```ignore
/// struct SliceDecoder<'a> {
///     data: &'a [u8],
///     position: usize,
/// }
///
/// impl<'a> Decoder for SliceDecoder<'a> {
///     fn read_u8(&mut self) -> io::Result<u8> {
///         if self.position >= self.data.len() {
///             return Err(io::Error::new(
///                 io::ErrorKind::UnexpectedEof,
///                 "unexpected end of data",
///             ));
///         }
///         let byte = self.data[self.position];
///         self.position += 1;
///         Ok(byte)
///     }
///
///     fn read_raw_bytes(&mut self, len: usize) -> io::Result<Vec<u8>> {
///         if self.position + len > self.data.len() {
///             return Err(io::Error::new(
///                 io::ErrorKind::UnexpectedEof,
///                 "unexpected end of data",
///             ));
///         }
///         let bytes = self.data[self.position..self.position + len].to_vec();
///         self.position += len;
///         Ok(bytes)
///     }
///
///     // ... implement other required methods
/// }
/// ```
pub trait Decoder {
    // =========================================================================
    // Required methods - these must be implemented by all decoders
    // =========================================================================

    /// Reads a single unsigned byte.
    fn read_u8(&mut self) -> io::Result<u8>;

    /// Reads a 16-bit unsigned integer in little-endian format.
    fn read_u16(&mut self) -> io::Result<u16>;

    /// Reads a 32-bit unsigned integer in little-endian format.
    fn read_u32(&mut self) -> io::Result<u32>;

    /// Reads a 64-bit unsigned integer in little-endian format.
    fn read_u64(&mut self) -> io::Result<u64>;

    /// Reads a 128-bit unsigned integer in little-endian format.
    fn read_u128(&mut self) -> io::Result<u128>;

    /// Reads a platform-sized unsigned integer.
    ///
    /// This is deserialized from a 64-bit value for portability.
    fn read_usize(&mut self) -> io::Result<usize>;

    /// Reads a single signed byte.
    fn read_i8(&mut self) -> io::Result<i8>;

    /// Reads a 16-bit signed integer in little-endian format.
    fn read_i16(&mut self) -> io::Result<i16>;

    /// Reads a 32-bit signed integer in little-endian format.
    fn read_i32(&mut self) -> io::Result<i32>;

    /// Reads a 64-bit signed integer in little-endian format.
    fn read_i64(&mut self) -> io::Result<i64>;

    /// Reads a 128-bit signed integer in little-endian format.
    fn read_i128(&mut self) -> io::Result<i128>;

    /// Reads a platform-sized signed integer.
    ///
    /// This is deserialized from a 64-bit value for portability.
    fn read_isize(&mut self) -> io::Result<isize>;

    /// Reads raw bytes directly from the input.
    ///
    /// Returns an owned vector containing the read bytes.
    fn read_raw_bytes(&mut self, len: usize) -> io::Result<Vec<u8>>;

    // =========================================================================
    // Default implementations - can be overridden for optimization
    // =========================================================================

    /// Reads a boolean value.
    ///
    /// Default implementation decodes `0u8` as `false` and any non-zero value
    /// as `true`.
    fn read_bool(&mut self) -> io::Result<bool> { Ok(self.read_u8()? != 0) }

    /// Reads a Unicode character.
    ///
    /// Default implementation decodes from a 32-bit Unicode scalar value.
    ///
    /// # Errors
    ///
    /// Returns an error if the value is not a valid Unicode scalar value.
    fn read_char(&mut self) -> io::Result<char> {
        let code = self.read_u32()?;
        char::from_u32(code).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid Unicode scalar value: {code}"),
            )
        })
    }

    /// Reads a 32-bit floating-point number.
    ///
    /// Default implementation uses IEEE 754 binary representation.
    fn read_f32(&mut self) -> io::Result<f32> {
        Ok(f32::from_bits(self.read_u32()?))
    }

    /// Reads a 64-bit floating-point number.
    ///
    /// Default implementation uses IEEE 754 binary representation.
    fn read_f64(&mut self) -> io::Result<f64> {
        Ok(f64::from_bits(self.read_u64()?))
    }

    /// Reads a string.
    ///
    /// Default implementation reads the length as `usize` followed by UTF-8
    /// bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the bytes are not valid UTF-8.
    fn read_str(&mut self) -> io::Result<String> {
        let len = self.read_usize()?;
        let bytes = self.read_raw_bytes(len)?;
        String::from_utf8(bytes).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid UTF-8: {e}"),
            )
        })
    }

    /// Reads a byte vector.
    ///
    /// Default implementation reads the length as `usize` followed by raw
    /// bytes.
    fn read_bytes(&mut self) -> io::Result<Vec<u8>> {
        let len = self.read_usize()?;
        self.read_raw_bytes(len)
    }

    /// Decodes a value of type `D` from this decoder.
    ///
    /// This is the primary entry point for decoding values. It creates a new
    /// [`Session`] and invokes [`Decode::decode`] on the target type, managing
    /// session state automatically.
    ///
    /// # Type Parameters
    ///
    /// - `D`: The type to decode. Must implement [`Decode`].
    ///
    /// # Arguments
    ///
    /// * `plugin` - Plugin context for custom deserialization strategies.
    ///
    /// # Returns
    ///
    /// The decoded value of type `D`, or an I/O error if decoding fails.
    fn decode<D: Decode>(&mut self, plugin: &Plugin) -> io::Result<D> {
        let mut session = Session::new();

        D::decode(self, plugin, &mut session)
    }
}

/// A trait for types that can be deserialized.
///
/// Types implementing [`Decode`] can deserialize themselves using a
/// [`Decoder`]. The [`Plugin`] parameter allows passing additional context for
/// custom deserialization strategies.
///
/// # Example
///
/// ```ignore
/// struct Point {
///     x: i32,
///     y: i32,
/// }
///
/// impl Decode for Point {
///     fn decode<D: Decoder + ?Sized>(
///         decoder: &mut D,
///         plugin: &Plugin,
///     ) -> io::Result<Self> {
///         let x = i32::decode(decoder, plugin)?;
///         let y = i32::decode(decoder, plugin)?;
///         Ok(Point { x, y })
///     }
/// }
/// ```
pub trait Decode: Sized {
    /// Decodes a value using the provided decoder.
    ///
    /// # Arguments
    ///
    /// * `decoder` - The decoder to read from
    /// * `plugin` - Plugin context for custom deserialization strategies
    /// * `session` - Session state for managing deserialization context
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying I/O operation fails or if the data
    /// is invalid.
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        plugin: &Plugin,
        session: &mut Session,
    ) -> io::Result<Self>;
}

// =============================================================================
// Implementations for primitive types
// =============================================================================

impl Decode for u8 {
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        _plugin: &Plugin,
        _session: &mut Session,
    ) -> io::Result<Self> {
        decoder.read_u8()
    }
}

impl Decode for u16 {
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        _plugin: &Plugin,
        _session: &mut Session,
    ) -> io::Result<Self> {
        decoder.read_u16()
    }
}

impl Decode for u32 {
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        _plugin: &Plugin,
        _session: &mut Session,
    ) -> io::Result<Self> {
        decoder.read_u32()
    }
}

impl Decode for u64 {
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        _plugin: &Plugin,
        _session: &mut Session,
    ) -> io::Result<Self> {
        decoder.read_u64()
    }
}

impl Decode for u128 {
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        _plugin: &Plugin,
        _session: &mut Session,
    ) -> io::Result<Self> {
        decoder.read_u128()
    }
}

impl Decode for usize {
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        _plugin: &Plugin,
        _session: &mut Session,
    ) -> io::Result<Self> {
        decoder.read_usize()
    }
}

impl Decode for i8 {
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        _plugin: &Plugin,
        _session: &mut Session,
    ) -> io::Result<Self> {
        decoder.read_i8()
    }
}

impl Decode for i16 {
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        _plugin: &Plugin,
        _session: &mut Session,
    ) -> io::Result<Self> {
        decoder.read_i16()
    }
}

impl Decode for i32 {
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        _plugin: &Plugin,
        _session: &mut Session,
    ) -> io::Result<Self> {
        decoder.read_i32()
    }
}

impl Decode for i64 {
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        _plugin: &Plugin,
        _session: &mut Session,
    ) -> io::Result<Self> {
        decoder.read_i64()
    }
}

impl Decode for i128 {
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        _plugin: &Plugin,
        _session: &mut Session,
    ) -> io::Result<Self> {
        decoder.read_i128()
    }
}

impl Decode for isize {
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        _plugin: &Plugin,
        _session: &mut Session,
    ) -> io::Result<Self> {
        decoder.read_isize()
    }
}

impl Decode for bool {
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        _plugin: &Plugin,
        _session: &mut Session,
    ) -> io::Result<Self> {
        decoder.read_bool()
    }
}

impl Decode for char {
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        _plugin: &Plugin,
        _session: &mut Session,
    ) -> io::Result<Self> {
        decoder.read_char()
    }
}

impl Decode for f32 {
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        _plugin: &Plugin,
        _session: &mut Session,
    ) -> io::Result<Self> {
        decoder.read_f32()
    }
}

impl Decode for f64 {
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        _plugin: &Plugin,
        _session: &mut Session,
    ) -> io::Result<Self> {
        decoder.read_f64()
    }
}

impl Decode for String {
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        _plugin: &Plugin,
        _session: &mut Session,
    ) -> io::Result<Self> {
        decoder.read_str()
    }
}

// =============================================================================
// Implementations for smart pointers
// =============================================================================

impl<T: Decode> Decode for Box<T> {
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        plugin: &Plugin,
        session: &mut Session,
    ) -> io::Result<Self> {
        Ok(Self::new(T::decode(decoder, plugin, session)?))
    }
}

impl<T: Decode> Decode for Rc<T> {
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        plugin: &Plugin,
        session: &mut Session,
    ) -> io::Result<Self> {
        Ok(Self::new(T::decode(decoder, plugin, session)?))
    }
}

impl<T: Decode> Decode for Arc<T> {
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        plugin: &Plugin,
        session: &mut Session,
    ) -> io::Result<Self> {
        Ok(Self::new(T::decode(decoder, plugin, session)?))
    }
}

impl<T: Decode> Decode for Box<[T]> {
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        plugin: &Plugin,
        session: &mut Session,
    ) -> io::Result<Self> {
        let len = decoder.read_usize()?;
        let mut vec = Vec::with_capacity(len);
        for _ in 0..len {
            vec.push(T::decode(decoder, plugin, session)?);
        }
        Ok(vec.into_boxed_slice())
    }
}

impl<T: Decode> Decode for Arc<[T]> {
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        plugin: &Plugin,
        session: &mut Session,
    ) -> io::Result<Self> {
        let len = decoder.read_usize()?;
        let mut vec = Vec::with_capacity(len);
        for _ in 0..len {
            vec.push(T::decode(decoder, plugin, session)?);
        }
        Ok(Self::from(vec))
    }
}

impl<T: Decode> Decode for Rc<[T]> {
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        plugin: &Plugin,
        session: &mut Session,
    ) -> io::Result<Self> {
        let len = decoder.read_usize()?;
        let mut vec = Vec::with_capacity(len);
        for _ in 0..len {
            vec.push(T::decode(decoder, plugin, session)?);
        }
        Ok(Self::from(vec))
    }
}

impl Decode for Box<str> {
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        _plugin: &Plugin,
        _session: &mut Session,
    ) -> io::Result<Self> {
        let s = decoder.read_str()?;
        Ok(s.into_boxed_str())
    }
}

impl Decode for Rc<str> {
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        _plugin: &Plugin,
        _session: &mut Session,
    ) -> io::Result<Self> {
        let s = decoder.read_str()?;
        Ok(Self::from(s))
    }
}

impl Decode for Arc<str> {
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        _plugin: &Plugin,
        _session: &mut Session,
    ) -> io::Result<Self> {
        let s = decoder.read_str()?;
        Ok(Self::from(s))
    }
}

impl<T: ToOwned + ?Sized> Decode for Cow<'_, T>
where
    T::Owned: Decode,
{
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        plugin: &Plugin,
        session: &mut Session,
    ) -> io::Result<Self> {
        Ok(Cow::Owned(T::Owned::decode(decoder, plugin, session)?))
    }
}

// =============================================================================
// Implementations for Option and Result
// =============================================================================

impl<T: Decode> Decode for Option<T> {
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        plugin: &Plugin,
        session: &mut Session,
    ) -> io::Result<Self> {
        let is_some = decoder.read_bool()?;
        if is_some {
            Ok(Some(T::decode(decoder, plugin, session)?))
        } else {
            Ok(None)
        }
    }
}

impl<T: Decode, E: Decode> Decode for Result<T, E> {
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        plugin: &Plugin,
        session: &mut Session,
    ) -> io::Result<Self> {
        let is_ok = decoder.read_bool()?;
        if is_ok {
            Ok(Ok(T::decode(decoder, plugin, session)?))
        } else {
            Ok(Err(E::decode(decoder, plugin, session)?))
        }
    }
}

// =============================================================================
// Implementations for collections
// =============================================================================

impl<T: Decode> Decode for Vec<T> {
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        plugin: &Plugin,
        session: &mut Session,
    ) -> io::Result<Self> {
        let len = decoder.read_usize()?;
        let mut vec = Self::with_capacity(len);
        for _ in 0..len {
            vec.push(T::decode(decoder, plugin, session)?);
        }
        Ok(vec)
    }
}

impl<T: Decode> Decode for VecDeque<T> {
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        plugin: &Plugin,
        session: &mut Session,
    ) -> io::Result<Self> {
        let len = decoder.read_usize()?;
        let mut deque = Self::with_capacity(len);
        for _ in 0..len {
            deque.push_back(T::decode(decoder, plugin, session)?);
        }
        Ok(deque)
    }
}

impl<T: Decode> Decode for LinkedList<T> {
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        plugin: &Plugin,
        session: &mut Session,
    ) -> io::Result<Self> {
        let len = decoder.read_usize()?;
        let mut list = Self::new();
        for _ in 0..len {
            list.push_back(T::decode(decoder, plugin, session)?);
        }
        Ok(list)
    }
}

impl<T: Decode, const N: usize> Decode for [T; N] {
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        plugin: &Plugin,
        session: &mut Session,
    ) -> io::Result<Self> {
        // Use array::try_from_fn when it's stable, for now use manual
        // initialization
        let mut array: [std::mem::MaybeUninit<T>; N] =
            unsafe { std::mem::MaybeUninit::uninit().assume_init() };

        for (i, slot) in array.iter_mut().enumerate() {
            match T::decode(decoder, plugin, session) {
                Ok(value) => {
                    slot.write(value);
                }
                Err(e) => {
                    // Drop already-initialized elements on error
                    for initialized in array.iter_mut().take(i) {
                        unsafe {
                            initialized.assume_init_drop();
                        }
                    }
                    return Err(e);
                }
            }
        }

        // Safety: All elements have been initialized
        Ok(array.map(|x| unsafe { x.assume_init() }))
    }
}

impl<K, V, S> Decode for HashMap<K, V, S>
where
    K: Decode + Eq + Hash,
    V: Decode,
    S: BuildHasher + Default,
{
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        plugin: &Plugin,
        session: &mut Session,
    ) -> io::Result<Self> {
        let len = decoder.read_usize()?;
        let mut map = Self::with_capacity_and_hasher(len, S::default());
        for _ in 0..len {
            let key = K::decode(decoder, plugin, session)?;
            let value = V::decode(decoder, plugin, session)?;
            map.insert(key, value);
        }
        Ok(map)
    }
}

impl<T, S> Decode for HashSet<T, S>
where
    T: Decode + Eq + Hash,
    S: BuildHasher + Default,
{
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        plugin: &Plugin,
        session: &mut Session,
    ) -> io::Result<Self> {
        let len = decoder.read_usize()?;
        let mut set = Self::with_capacity_and_hasher(len, S::default());
        for _ in 0..len {
            set.insert(T::decode(decoder, plugin, session)?);
        }
        Ok(set)
    }
}

impl<K: Decode + Ord, V: Decode> Decode for BTreeMap<K, V> {
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        plugin: &Plugin,
        session: &mut Session,
    ) -> io::Result<Self> {
        let len = decoder.read_usize()?;
        let mut map = Self::new();
        for _ in 0..len {
            let key = K::decode(decoder, plugin, session)?;
            let value = V::decode(decoder, plugin, session)?;
            map.insert(key, value);
        }
        Ok(map)
    }
}

impl<T: Decode + Ord> Decode for BTreeSet<T> {
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        plugin: &Plugin,
        session: &mut Session,
    ) -> io::Result<Self> {
        let len = decoder.read_usize()?;
        let mut set = Self::new();
        for _ in 0..len {
            set.insert(T::decode(decoder, plugin, session)?);
        }
        Ok(set)
    }
}

// =============================================================================
// Implementations for tuples
// =============================================================================

impl Decode for () {
    fn decode<D: Decoder + ?Sized>(
        _decoder: &mut D,
        _plugin: &Plugin,
        _session: &mut Session,
    ) -> io::Result<Self> {
        Ok(())
    }
}

macro_rules! impl_decode_tuple {
    ($($name:ident),+) => {
        impl<$($name: Decode),+> Decode for ($($name,)+) {
            fn decode<D: Decoder + ?Sized>(
                decoder: &mut D,
                plugin: &Plugin,
                session: &mut Session,
            ) -> io::Result<Self> {
                Ok(($(
                    $name::decode(decoder, plugin, session)?,
                )+))
            }
        }
    };
}

impl_decode_tuple!(A);
impl_decode_tuple!(A, B);
impl_decode_tuple!(A, B, C);
impl_decode_tuple!(A, B, C, D_);
impl_decode_tuple!(A, B, C, D_, E);
impl_decode_tuple!(A, B, C, D_, E, F);
impl_decode_tuple!(A, B, C, D_, E, F, G);
impl_decode_tuple!(A, B, C, D_, E, F, G, H);
impl_decode_tuple!(A, B, C, D_, E, F, G, H, I);
impl_decode_tuple!(A, B, C, D_, E, F, G, H, I, J);
impl_decode_tuple!(A, B, C, D_, E, F, G, H, I, J, K);
impl_decode_tuple!(A, B, C, D_, E, F, G, H, I, J, K, L);

// =============================================================================
// Implementations for special types
// =============================================================================

impl<T> Decode for std::marker::PhantomData<T> {
    fn decode<D: Decoder + ?Sized>(
        _decoder: &mut D,
        _plugin: &Plugin,
        _session: &mut Session,
    ) -> io::Result<Self> {
        Ok(Self)
    }
}

impl Decode for std::time::Duration {
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        plugin: &Plugin,
        session: &mut Session,
    ) -> io::Result<Self> {
        let secs = u64::decode(decoder, plugin, session)?;
        let nanos = u32::decode(decoder, plugin, session)?;
        Ok(Self::new(secs, nanos))
    }
}

impl<T: Decode + Copy> Decode for std::cell::Cell<T> {
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        plugin: &Plugin,
        session: &mut Session,
    ) -> io::Result<Self> {
        Ok(Self::new(T::decode(decoder, plugin, session)?))
    }
}

impl<T: Decode> Decode for std::cell::RefCell<T> {
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        plugin: &Plugin,
        session: &mut Session,
    ) -> io::Result<Self> {
        Ok(Self::new(T::decode(decoder, plugin, session)?))
    }
}

impl<T: Decode> Decode for std::num::Wrapping<T> {
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        plugin: &Plugin,
        session: &mut Session,
    ) -> io::Result<Self> {
        Ok(Self(T::decode(decoder, plugin, session)?))
    }
}

impl<T: Decode> Decode for std::cmp::Reverse<T> {
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        plugin: &Plugin,
        session: &mut Session,
    ) -> io::Result<Self> {
        Ok(Self(T::decode(decoder, plugin, session)?))
    }
}

// Non-zero integer types
macro_rules! impl_decode_nonzero {
    ($ty:ty, $inner:ty) => {
        impl Decode for $ty {
            fn decode<D: Decoder + ?Sized>(
                decoder: &mut D,
                plugin: &Plugin,
                session: &mut Session,
            ) -> io::Result<Self> {
                let value = <$inner>::decode(decoder, plugin, session)?;
                <$ty>::new(value).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        concat!(
                            "invalid ",
                            stringify!($ty),
                            ": value cannot be zero"
                        ),
                    )
                })
            }
        }
    };
}

impl_decode_nonzero!(std::num::NonZeroU8, u8);
impl_decode_nonzero!(std::num::NonZeroU16, u16);
impl_decode_nonzero!(std::num::NonZeroU32, u32);
impl_decode_nonzero!(std::num::NonZeroU64, u64);
impl_decode_nonzero!(std::num::NonZeroU128, u128);
impl_decode_nonzero!(std::num::NonZeroUsize, usize);
impl_decode_nonzero!(std::num::NonZeroI8, i8);
impl_decode_nonzero!(std::num::NonZeroI16, i16);
impl_decode_nonzero!(std::num::NonZeroI32, i32);
impl_decode_nonzero!(std::num::NonZeroI64, i64);
impl_decode_nonzero!(std::num::NonZeroI128, i128);
impl_decode_nonzero!(std::num::NonZeroIsize, isize);

// Atomic types
macro_rules! impl_decode_atomic {
    ($atomic:ty, $inner:ty) => {
        impl Decode for $atomic {
            fn decode<D: Decoder + ?Sized>(
                decoder: &mut D,
                plugin: &Plugin,
                session: &mut Session,
            ) -> io::Result<Self> {
                Ok(<$atomic>::new(<$inner>::decode(decoder, plugin, session)?))
            }
        }
    };
}

impl_decode_atomic!(std::sync::atomic::AtomicBool, bool);
impl_decode_atomic!(std::sync::atomic::AtomicI8, i8);
impl_decode_atomic!(std::sync::atomic::AtomicI16, i16);
impl_decode_atomic!(std::sync::atomic::AtomicI32, i32);
impl_decode_atomic!(std::sync::atomic::AtomicI64, i64);
impl_decode_atomic!(std::sync::atomic::AtomicIsize, isize);
impl_decode_atomic!(std::sync::atomic::AtomicU8, u8);
impl_decode_atomic!(std::sync::atomic::AtomicU16, u16);
impl_decode_atomic!(std::sync::atomic::AtomicU32, u32);
impl_decode_atomic!(std::sync::atomic::AtomicU64, u64);
impl_decode_atomic!(std::sync::atomic::AtomicUsize, usize);

// Range types
impl<T: Decode> Decode for std::ops::Range<T> {
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        plugin: &Plugin,
        session: &mut Session,
    ) -> io::Result<Self> {
        let start = T::decode(decoder, plugin, session)?;
        let end = T::decode(decoder, plugin, session)?;
        Ok(start..end)
    }
}

impl<T: Decode> Decode for std::ops::RangeInclusive<T> {
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        plugin: &Plugin,
        session: &mut Session,
    ) -> io::Result<Self> {
        let start = T::decode(decoder, plugin, session)?;
        let end = T::decode(decoder, plugin, session)?;
        Ok(start..=end)
    }
}

impl<T: Decode> Decode for std::ops::RangeFrom<T> {
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        plugin: &Plugin,
        session: &mut Session,
    ) -> io::Result<Self> {
        Ok(T::decode(decoder, plugin, session)?..)
    }
}

impl<T: Decode> Decode for std::ops::RangeTo<T> {
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        plugin: &Plugin,
        session: &mut Session,
    ) -> io::Result<Self> {
        Ok(..T::decode(decoder, plugin, session)?)
    }
}

impl<T: Decode> Decode for std::ops::RangeToInclusive<T> {
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        plugin: &Plugin,
        session: &mut Session,
    ) -> io::Result<Self> {
        Ok(..=T::decode(decoder, plugin, session)?)
    }
}

impl Decode for std::ops::RangeFull {
    fn decode<D: Decoder + ?Sized>(
        _decoder: &mut D,
        _plugin: &Plugin,
        _session: &mut Session,
    ) -> io::Result<Self> {
        Ok(..)
    }
}

impl<T: Decode> Decode for std::ops::Bound<T> {
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        plugin: &Plugin,
        session: &mut Session,
    ) -> io::Result<Self> {
        let tag = decoder.read_u8()?;
        match tag {
            0 => Ok(Self::Unbounded),
            1 => Ok(Self::Included(T::decode(decoder, plugin, session)?)),
            2 => Ok(Self::Excluded(T::decode(decoder, plugin, session)?)),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid Bound tag: {tag}"),
            )),
        }
    }
}

// =============================================================================

impl<K: Decode + Eq + Hash, V: Decode, S: BuildHasher + Default + Clone> Decode
    for DashMap<K, V, S>
{
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        plugin: &Plugin,
        session: &mut Session,
    ) -> io::Result<Self> {
        let len = decoder.read_usize()?;
        let map = Self::with_capacity_and_hasher(len, S::default());

        for _ in 0..len {
            let key = K::decode(decoder, plugin, session)?;
            let value = V::decode(decoder, plugin, session)?;
            map.insert(key, value);
        }

        Ok(map)
    }
}

impl<T: Decode + Eq + Hash, S: BuildHasher + Default + Clone> Decode
    for dashmap::DashSet<T, S>
{
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        plugin: &Plugin,
        session: &mut Session,
    ) -> io::Result<Self> {
        let len = decoder.read_usize()?;
        let set = Self::with_capacity_and_hasher(len, S::default());

        for _ in 0..len {
            let value = T::decode(decoder, plugin, session)?;
            set.insert(value);
        }

        Ok(set)
    }
}
