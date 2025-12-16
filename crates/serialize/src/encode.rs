//! Encoding traits and implementations for serialization.
//!
//! This module provides the [`Encoder`] trait for implementing custom encoders,
//! and the [`Encode`] trait for types that can be serialized.

use std::{
    borrow::Cow,
    collections::{BTreeMap, BTreeSet, HashMap, HashSet, LinkedList, VecDeque},
    hash::BuildHasher,
    io,
    rc::Rc,
    sync::Arc,
};

use crate::plugin::Plugin;

/// A trait for types that can serialize primitive values to a binary format.
///
/// Implementors of this trait provide the low-level serialization of primitive
/// types. The default implementations delegate to other methods where possible,
/// but can be overridden for performance or format-specific reasons.
///
/// All methods return `Result<(), std::io::Error>` to allow for fallible I/O
/// operations.
///
/// # Example
///
/// ```ignore
/// use qbice_serialize::encode::Encoder;
/// use std::io::{self, Write};
///
/// struct VecEncoder {
///     buffer: Vec<u8>,
/// }
///
/// impl Encoder for VecEncoder {
///     fn emit_u8(&mut self, v: u8) -> io::Result<()> {
///         self.buffer.push(v);
///         Ok(())
///     }
///
///     fn emit_raw_bytes(&mut self, s: &[u8]) -> io::Result<()> {
///         self.buffer.extend_from_slice(s);
///         Ok(())
///     }
///
///     // ... implement other required methods
/// }
/// ```
pub trait Encoder {
    // =========================================================================
    // Required methods - these must be implemented by all encoders
    // =========================================================================

    /// Emits a single unsigned byte.
    fn emit_u8(&mut self, v: u8) -> io::Result<()>;

    /// Emits a 16-bit unsigned integer in little-endian format.
    fn emit_u16(&mut self, v: u16) -> io::Result<()>;

    /// Emits a 32-bit unsigned integer in little-endian format.
    fn emit_u32(&mut self, v: u32) -> io::Result<()>;

    /// Emits a 64-bit unsigned integer in little-endian format.
    fn emit_u64(&mut self, v: u64) -> io::Result<()>;

    /// Emits a 128-bit unsigned integer in little-endian format.
    fn emit_u128(&mut self, v: u128) -> io::Result<()>;

    /// Emits a platform-sized unsigned integer.
    ///
    /// This is serialized as a 64-bit value for portability.
    fn emit_usize(&mut self, v: usize) -> io::Result<()>;

    /// Emits a single signed byte.
    fn emit_i8(&mut self, v: i8) -> io::Result<()>;

    /// Emits a 16-bit signed integer in little-endian format.
    fn emit_i16(&mut self, v: i16) -> io::Result<()>;

    /// Emits a 32-bit signed integer in little-endian format.
    fn emit_i32(&mut self, v: i32) -> io::Result<()>;

    /// Emits a 64-bit signed integer in little-endian format.
    fn emit_i64(&mut self, v: i64) -> io::Result<()>;

    /// Emits a 128-bit signed integer in little-endian format.
    fn emit_i128(&mut self, v: i128) -> io::Result<()>;

    /// Emits a platform-sized signed integer.
    ///
    /// This is serialized as a 64-bit value for portability.
    fn emit_isize(&mut self, v: isize) -> io::Result<()>;

    /// Emits raw bytes directly to the output.
    fn emit_raw_bytes(&mut self, s: &[u8]) -> io::Result<()>;

    // =========================================================================
    // Default implementations - can be overridden for optimization
    // =========================================================================

    /// Emits a boolean value.
    ///
    /// Default implementation encodes `true` as `1u8` and `false` as `0u8`.
    fn emit_bool(&mut self, v: bool) -> io::Result<()> {
        self.emit_u8(u8::from(v))
    }

    /// Emits a Unicode character.
    ///
    /// Default implementation encodes the character as its 32-bit Unicode
    /// scalar value.
    fn emit_char(&mut self, v: char) -> io::Result<()> {
        self.emit_u32(v as u32)
    }

    /// Emits a 32-bit floating-point number.
    ///
    /// Default implementation uses IEEE 754 binary representation.
    fn emit_f32(&mut self, v: f32) -> io::Result<()> {
        self.emit_u32(v.to_bits())
    }

    /// Emits a 64-bit floating-point number.
    ///
    /// Default implementation uses IEEE 754 binary representation.
    fn emit_f64(&mut self, v: f64) -> io::Result<()> {
        self.emit_u64(v.to_bits())
    }

    /// Emits a string slice.
    ///
    /// Default implementation encodes the length as `usize` followed by the
    /// UTF-8 bytes.
    fn emit_str(&mut self, v: &str) -> io::Result<()> {
        self.emit_usize(v.len())?;
        self.emit_raw_bytes(v.as_bytes())
    }

    /// Emits a byte slice.
    ///
    /// Default implementation encodes the length as `usize` followed by the
    /// raw bytes.
    fn emit_bytes(&mut self, v: &[u8]) -> io::Result<()> {
        self.emit_usize(v.len())?;
        self.emit_raw_bytes(v)
    }
}

/// A trait for types that can be serialized.
///
/// Types implementing [`Encode`] can serialize themselves using an [`Encoder`].
/// The [`Plugin`] parameter allows passing additional context for custom
/// serialization strategies.
///
/// # Example
///
/// ```ignore
/// use qbice_serialize::{encode::{Encode, Encoder}, plugin::Plugin};
/// use std::io;
///
/// struct Point {
///     x: i32,
///     y: i32,
/// }
///
/// impl Encode for Point {
///     fn encode<E: Encoder + ?Sized>(
///         &self,
///         encoder: &mut E,
///         plugin: &Plugin,
///     ) -> io::Result<()> {
///         self.x.encode(encoder, plugin)?;
///         self.y.encode(encoder, plugin)?;
///         Ok(())
///     }
/// }
/// ```
pub trait Encode {
    /// Encodes this value using the provided encoder.
    ///
    /// # Arguments
    ///
    /// * `encoder` - The encoder to write to
    /// * `plugin` - Plugin context for custom serialization strategies
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying I/O operation fails.
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        plugin: &Plugin,
    ) -> io::Result<()>;
}

// =============================================================================
// Implementations for primitive types
// =============================================================================

impl Encode for u8 {
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        _plugin: &Plugin,
    ) -> io::Result<()> {
        encoder.emit_u8(*self)
    }
}

impl Encode for u16 {
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        _plugin: &Plugin,
    ) -> io::Result<()> {
        encoder.emit_u16(*self)
    }
}

impl Encode for u32 {
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        _plugin: &Plugin,
    ) -> io::Result<()> {
        encoder.emit_u32(*self)
    }
}

impl Encode for u64 {
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        _plugin: &Plugin,
    ) -> io::Result<()> {
        encoder.emit_u64(*self)
    }
}

impl Encode for u128 {
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        _plugin: &Plugin,
    ) -> io::Result<()> {
        encoder.emit_u128(*self)
    }
}

impl Encode for usize {
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        _plugin: &Plugin,
    ) -> io::Result<()> {
        encoder.emit_usize(*self)
    }
}

impl Encode for i8 {
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        _plugin: &Plugin,
    ) -> io::Result<()> {
        encoder.emit_i8(*self)
    }
}

impl Encode for i16 {
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        _plugin: &Plugin,
    ) -> io::Result<()> {
        encoder.emit_i16(*self)
    }
}

impl Encode for i32 {
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        _plugin: &Plugin,
    ) -> io::Result<()> {
        encoder.emit_i32(*self)
    }
}

impl Encode for i64 {
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        _plugin: &Plugin,
    ) -> io::Result<()> {
        encoder.emit_i64(*self)
    }
}

impl Encode for i128 {
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        _plugin: &Plugin,
    ) -> io::Result<()> {
        encoder.emit_i128(*self)
    }
}

impl Encode for isize {
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        _plugin: &Plugin,
    ) -> io::Result<()> {
        encoder.emit_isize(*self)
    }
}

impl Encode for bool {
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        _plugin: &Plugin,
    ) -> io::Result<()> {
        encoder.emit_bool(*self)
    }
}

impl Encode for char {
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        _plugin: &Plugin,
    ) -> io::Result<()> {
        encoder.emit_char(*self)
    }
}

impl Encode for f32 {
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        _plugin: &Plugin,
    ) -> io::Result<()> {
        encoder.emit_f32(*self)
    }
}

impl Encode for f64 {
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        _plugin: &Plugin,
    ) -> io::Result<()> {
        encoder.emit_f64(*self)
    }
}

impl Encode for str {
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        _plugin: &Plugin,
    ) -> io::Result<()> {
        encoder.emit_str(self)
    }
}

impl Encode for String {
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        _plugin: &Plugin,
    ) -> io::Result<()> {
        encoder.emit_str(self)
    }
}

// =============================================================================
// Implementations for references and smart pointers
// =============================================================================

impl<T: Encode + ?Sized> Encode for &T {
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        plugin: &Plugin,
    ) -> io::Result<()> {
        (**self).encode(encoder, plugin)
    }
}

impl<T: Encode + ?Sized> Encode for &mut T {
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        plugin: &Plugin,
    ) -> io::Result<()> {
        (**self).encode(encoder, plugin)
    }
}

impl<T: Encode + ?Sized> Encode for Box<T> {
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        plugin: &Plugin,
    ) -> io::Result<()> {
        (**self).encode(encoder, plugin)
    }
}

impl<T: Encode + ?Sized> Encode for Rc<T> {
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        plugin: &Plugin,
    ) -> io::Result<()> {
        (**self).encode(encoder, plugin)
    }
}

impl<T: Encode + ?Sized> Encode for Arc<T> {
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        plugin: &Plugin,
    ) -> io::Result<()> {
        (**self).encode(encoder, plugin)
    }
}

impl<T: Encode + ToOwned + ?Sized> Encode for Cow<'_, T> {
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        plugin: &Plugin,
    ) -> io::Result<()> {
        (**self).encode(encoder, plugin)
    }
}

// =============================================================================
// Implementations for Option and Result
// =============================================================================

impl<T: Encode> Encode for Option<T> {
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        plugin: &Plugin,
    ) -> io::Result<()> {
        match self {
            Some(v) => {
                encoder.emit_bool(true)?;
                v.encode(encoder, plugin)
            }
            None => encoder.emit_bool(false),
        }
    }
}

impl<T: Encode, U: Encode> Encode for Result<T, U> {
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        plugin: &Plugin,
    ) -> io::Result<()> {
        match self {
            Ok(v) => {
                encoder.emit_bool(true)?;
                v.encode(encoder, plugin)
            }
            Err(e) => {
                encoder.emit_bool(false)?;
                e.encode(encoder, plugin)
            }
        }
    }
}

// =============================================================================
// Implementations for collections
// =============================================================================

impl<T: Encode> Encode for Vec<T> {
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        plugin: &Plugin,
    ) -> io::Result<()> {
        encoder.emit_usize(self.len())?;
        for item in self {
            item.encode(encoder, plugin)?;
        }
        Ok(())
    }
}

impl<T: Encode> Encode for VecDeque<T> {
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        plugin: &Plugin,
    ) -> io::Result<()> {
        encoder.emit_usize(self.len())?;
        for item in self {
            item.encode(encoder, plugin)?;
        }
        Ok(())
    }
}

impl<T: Encode> Encode for LinkedList<T> {
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        plugin: &Plugin,
    ) -> io::Result<()> {
        encoder.emit_usize(self.len())?;
        for item in self {
            item.encode(encoder, plugin)?;
        }
        Ok(())
    }
}

impl<T: Encode, const N: usize> Encode for [T; N] {
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        plugin: &Plugin,
    ) -> io::Result<()> {
        for item in self {
            item.encode(encoder, plugin)?;
        }
        Ok(())
    }
}

impl<T: Encode> Encode for [T] {
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        plugin: &Plugin,
    ) -> io::Result<()> {
        encoder.emit_usize(self.len())?;
        for item in self {
            item.encode(encoder, plugin)?;
        }
        Ok(())
    }
}

impl<K: Encode, V: Encode, S: BuildHasher> Encode for HashMap<K, V, S> {
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        plugin: &Plugin,
    ) -> io::Result<()> {
        encoder.emit_usize(self.len())?;
        for (key, value) in self {
            key.encode(encoder, plugin)?;
            value.encode(encoder, plugin)?;
        }
        Ok(())
    }
}

impl<T: Encode, S: BuildHasher> Encode for HashSet<T, S> {
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        plugin: &Plugin,
    ) -> io::Result<()> {
        encoder.emit_usize(self.len())?;
        for item in self {
            item.encode(encoder, plugin)?;
        }
        Ok(())
    }
}

impl<K: Encode, V: Encode> Encode for BTreeMap<K, V> {
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        plugin: &Plugin,
    ) -> io::Result<()> {
        encoder.emit_usize(self.len())?;
        for (key, value) in self {
            key.encode(encoder, plugin)?;
            value.encode(encoder, plugin)?;
        }
        Ok(())
    }
}

impl<T: Encode> Encode for BTreeSet<T> {
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        plugin: &Plugin,
    ) -> io::Result<()> {
        encoder.emit_usize(self.len())?;
        for item in self {
            item.encode(encoder, plugin)?;
        }
        Ok(())
    }
}

// =============================================================================
// Implementations for tuples
// =============================================================================

impl Encode for () {
    fn encode<E: Encoder + ?Sized>(
        &self,
        _encoder: &mut E,
        _plugin: &Plugin,
    ) -> io::Result<()> {
        Ok(())
    }
}

macro_rules! impl_encode_tuple {
    ($($name:ident),+) => {
        impl<$($name: Encode),+> Encode for ($($name,)+) {
            #[allow(non_snake_case)]
            fn encode<E: Encoder + ?Sized>(
                &self,
                encoder: &mut E,
                plugin: &Plugin,
            ) -> io::Result<()> {
                let ($($name,)+) = self;
                $(
                    $name.encode(encoder, plugin)?;
                )+
                Ok(())
            }
        }
    };
}

impl_encode_tuple!(A);
impl_encode_tuple!(A, B);
impl_encode_tuple!(A, B, C);
impl_encode_tuple!(A, B, C, D);
impl_encode_tuple!(A, B, C, D, E_);
impl_encode_tuple!(A, B, C, D, E_, F);
impl_encode_tuple!(A, B, C, D, E_, F, G);
impl_encode_tuple!(A, B, C, D, E_, F, G, H);
impl_encode_tuple!(A, B, C, D, E_, F, G, H, I);
impl_encode_tuple!(A, B, C, D, E_, F, G, H, I, J);
impl_encode_tuple!(A, B, C, D, E_, F, G, H, I, J, K);
impl_encode_tuple!(A, B, C, D, E_, F, G, H, I, J, K, L);

// =============================================================================
// Implementations for special types
// =============================================================================

impl<T: Encode> Encode for std::marker::PhantomData<T> {
    fn encode<E: Encoder + ?Sized>(
        &self,
        _encoder: &mut E,
        _plugin: &Plugin,
    ) -> io::Result<()> {
        Ok(())
    }
}

impl Encode for std::time::Duration {
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        plugin: &Plugin,
    ) -> io::Result<()> {
        self.as_secs().encode(encoder, plugin)?;
        self.subsec_nanos().encode(encoder, plugin)?;
        Ok(())
    }
}

impl<T: Encode + Copy> Encode for std::cell::Cell<T> {
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        plugin: &Plugin,
    ) -> io::Result<()> {
        self.get().encode(encoder, plugin)
    }
}

impl<T: Encode + ?Sized> Encode for std::cell::RefCell<T> {
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        plugin: &Plugin,
    ) -> io::Result<()> {
        self.borrow().encode(encoder, plugin)
    }
}

impl<T: Encode> Encode for std::num::Wrapping<T> {
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        plugin: &Plugin,
    ) -> io::Result<()> {
        self.0.encode(encoder, plugin)
    }
}

impl<T: Encode> Encode for std::cmp::Reverse<T> {
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        plugin: &Plugin,
    ) -> io::Result<()> {
        self.0.encode(encoder, plugin)
    }
}

// Non-zero integer types
macro_rules! impl_encode_nonzero {
    ($($ty:ty),+) => {
        $(
            impl Encode for $ty {
                fn encode<E: Encoder + ?Sized>(
                    &self,
                    encoder: &mut E,
                    plugin: &Plugin,
                ) -> io::Result<()> {
                    self.get().encode(encoder, plugin)
                }
            }
        )+
    };
}

impl_encode_nonzero!(
    std::num::NonZeroU8,
    std::num::NonZeroU16,
    std::num::NonZeroU32,
    std::num::NonZeroU64,
    std::num::NonZeroU128,
    std::num::NonZeroUsize,
    std::num::NonZeroI8,
    std::num::NonZeroI16,
    std::num::NonZeroI32,
    std::num::NonZeroI64,
    std::num::NonZeroI128,
    std::num::NonZeroIsize
);

// Atomic types (encode the current value)
macro_rules! impl_encode_atomic {
    ($atomic:ty, $inner:ty) => {
        impl Encode for $atomic {
            fn encode<E: Encoder + ?Sized>(
                &self,
                encoder: &mut E,
                plugin: &Plugin,
            ) -> io::Result<()> {
                self.load(std::sync::atomic::Ordering::Relaxed)
                    .encode(encoder, plugin)
            }
        }
    };
}

impl_encode_atomic!(std::sync::atomic::AtomicBool, bool);
impl_encode_atomic!(std::sync::atomic::AtomicI8, i8);
impl_encode_atomic!(std::sync::atomic::AtomicI16, i16);
impl_encode_atomic!(std::sync::atomic::AtomicI32, i32);
impl_encode_atomic!(std::sync::atomic::AtomicI64, i64);
impl_encode_atomic!(std::sync::atomic::AtomicIsize, isize);
impl_encode_atomic!(std::sync::atomic::AtomicU8, u8);
impl_encode_atomic!(std::sync::atomic::AtomicU16, u16);
impl_encode_atomic!(std::sync::atomic::AtomicU32, u32);
impl_encode_atomic!(std::sync::atomic::AtomicU64, u64);
impl_encode_atomic!(std::sync::atomic::AtomicUsize, usize);

// Range types
impl<T: Encode> Encode for std::ops::Range<T> {
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        plugin: &Plugin,
    ) -> io::Result<()> {
        self.start.encode(encoder, plugin)?;
        self.end.encode(encoder, plugin)?;
        Ok(())
    }
}

impl<T: Encode> Encode for std::ops::RangeInclusive<T> {
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        plugin: &Plugin,
    ) -> io::Result<()> {
        self.start().encode(encoder, plugin)?;
        self.end().encode(encoder, plugin)?;
        Ok(())
    }
}

impl<T: Encode> Encode for std::ops::RangeFrom<T> {
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        plugin: &Plugin,
    ) -> io::Result<()> {
        self.start.encode(encoder, plugin)
    }
}

impl<T: Encode> Encode for std::ops::RangeTo<T> {
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        plugin: &Plugin,
    ) -> io::Result<()> {
        self.end.encode(encoder, plugin)
    }
}

impl<T: Encode> Encode for std::ops::RangeToInclusive<T> {
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        plugin: &Plugin,
    ) -> io::Result<()> {
        self.end.encode(encoder, plugin)
    }
}

impl Encode for std::ops::RangeFull {
    fn encode<E: Encoder + ?Sized>(
        &self,
        _encoder: &mut E,
        _plugin: &Plugin,
    ) -> io::Result<()> {
        Ok(())
    }
}

impl<T: Encode> Encode for std::ops::Bound<T> {
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        plugin: &Plugin,
    ) -> io::Result<()> {
        match self {
            Self::Unbounded => encoder.emit_u8(0),
            Self::Included(v) => {
                encoder.emit_u8(1)?;
                v.encode(encoder, plugin)
            }
            Self::Excluded(v) => {
                encoder.emit_u8(2)?;
                v.encode(encoder, plugin)
            }
        }
    }
}
