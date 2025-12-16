//! Postcard-style binary encoding format.
//!
//! This module provides [`PostcardEncoder`] and [`PostcardDecoder`] which
//! implement the [`Encoder`] and [`Decoder`] traits using a format similar to
//! the [`postcard`](https://docs.rs/postcard) crate.
//!
//! # Format Overview
//!
//! - **Variable-length integers**: Uses LEB128-style varint encoding for
//!   integers, making small values compact.
//! - **Fixed-size primitives**: `u8`, `i8`, `bool` use single bytes.
//! - **Floating-point**: `f32` and `f64` use little-endian IEEE 754.
//! - **Strings/bytes**: Length-prefixed with varint length.
//!
//! # Varint Encoding
//!
//! Unsigned integers use standard LEB128 encoding where each byte stores 7 bits
//! of data with the MSB indicating continuation. Signed integers use zigzag
//! encoding before LEB128 to efficiently encode small negative numbers.
//!
//! # Example
//!
//! ```ignore
//! use qbice_serialize::{Encode, Decode, Plugin};
//! use qbice_serialize::postcard::{PostcardEncoder, PostcardDecoder};
//!
//! let plugin = Plugin::new();
//!
//! // Encoding to a Vec<u8>
//! let mut encoder = PostcardEncoder::new(Vec::new());
//! 42u32.encode(&mut encoder, &plugin).unwrap();
//! let bytes = encoder.into_inner();
//!
//! // Decoding from a slice
//! let mut decoder = PostcardDecoder::new(&bytes[..]);
//! let value = u32::decode(&mut decoder, &plugin).unwrap();
//! assert_eq!(value, 42);
//! ```

use std::io::{self, Read, Write};

use crate::{Decoder, Encoder};

// =============================================================================
// Varint helper functions
// =============================================================================

/// Maximum number of bytes for various varint sizes.
const MAX_VARINT_U16_BYTES: usize = 3;
const MAX_VARINT_U32_BYTES: usize = 5;
const MAX_VARINT_U64_BYTES: usize = 10;
const MAX_VARINT_U128_BYTES: usize = 19;

/// Encodes an unsigned 16-bit integer as a varint into the buffer.
/// Returns the number of bytes written.
#[inline]
#[allow(clippy::cast_possible_truncation)]
const fn encode_varint_u16(
    mut value: u16,
    buf: &mut [u8; MAX_VARINT_U16_BYTES],
) -> usize {
    let mut i = 0;
    while value >= 0x80 {
        buf[i] = (value as u8) | 0x80;
        value >>= 7;
        i += 1;
    }
    buf[i] = value as u8;
    i + 1
}

/// Encodes an unsigned 32-bit integer as a varint into the buffer.
/// Returns the number of bytes written.
#[inline]
#[allow(clippy::cast_possible_truncation)]
const fn encode_varint_u32(
    mut value: u32,
    buf: &mut [u8; MAX_VARINT_U32_BYTES],
) -> usize {
    let mut i = 0;
    while value >= 0x80 {
        buf[i] = (value as u8) | 0x80;
        value >>= 7;
        i += 1;
    }
    buf[i] = value as u8;
    i + 1
}

/// Encodes an unsigned 64-bit integer as a varint into the buffer.
/// Returns the number of bytes written.
#[inline]
#[allow(clippy::cast_possible_truncation)]
const fn encode_varint_u64(
    mut value: u64,
    buf: &mut [u8; MAX_VARINT_U64_BYTES],
) -> usize {
    let mut i = 0;
    while value >= 0x80 {
        buf[i] = (value as u8) | 0x80;
        value >>= 7;
        i += 1;
    }
    buf[i] = value as u8;
    i + 1
}

/// Encodes an unsigned 128-bit integer as a varint into the buffer.
/// Returns the number of bytes written.
#[inline]
#[allow(clippy::cast_possible_truncation)]
const fn encode_varint_u128(
    mut value: u128,
    buf: &mut [u8; MAX_VARINT_U128_BYTES],
) -> usize {
    let mut i = 0;
    while value >= 0x80 {
        buf[i] = (value as u8) | 0x80;
        value >>= 7;
        i += 1;
    }
    buf[i] = value as u8;
    i + 1
}

/// Zigzag encodes a signed 16-bit integer to unsigned.
#[inline]
#[allow(clippy::cast_sign_loss)]
const fn zigzag_encode_i16(value: i16) -> u16 {
    ((value << 1) ^ (value >> 15)) as u16
}

/// Zigzag encodes a signed 32-bit integer to unsigned.
#[inline]
#[allow(clippy::cast_sign_loss)]
const fn zigzag_encode_i32(value: i32) -> u32 {
    ((value << 1) ^ (value >> 31)) as u32
}

/// Zigzag encodes a signed 64-bit integer to unsigned.
#[inline]
#[allow(clippy::cast_sign_loss)]
const fn zigzag_encode_i64(value: i64) -> u64 {
    ((value << 1) ^ (value >> 63)) as u64
}

/// Zigzag encodes a signed 128-bit integer to unsigned.
#[inline]
#[allow(clippy::cast_sign_loss)]
const fn zigzag_encode_i128(value: i128) -> u128 {
    ((value << 1) ^ (value >> 127)) as u128
}

/// Zigzag decodes an unsigned 16-bit integer to signed.
#[inline]
#[allow(clippy::cast_possible_wrap)]
const fn zigzag_decode_i16(value: u16) -> i16 {
    ((value >> 1) as i16) ^ (-((value & 1) as i16))
}

/// Zigzag decodes an unsigned 32-bit integer to signed.
#[inline]
#[allow(clippy::cast_possible_wrap)]
const fn zigzag_decode_i32(value: u32) -> i32 {
    ((value >> 1) as i32) ^ (-((value & 1) as i32))
}

/// Zigzag decodes an unsigned 64-bit integer to signed.
#[inline]
#[allow(clippy::cast_possible_wrap)]
const fn zigzag_decode_i64(value: u64) -> i64 {
    ((value >> 1) as i64) ^ (-((value & 1) as i64))
}

/// Zigzag decodes an unsigned 128-bit integer to signed.
#[inline]
#[allow(clippy::cast_possible_wrap)]
const fn zigzag_decode_i128(value: u128) -> i128 {
    ((value >> 1) as i128) ^ (-((value & 1) as i128))
}

// =============================================================================
// PostcardEncoder
// =============================================================================

/// A postcard-style encoder that writes to any [`Write`] implementation.
///
/// Uses varint encoding for integers to achieve compact output.
/// The encoder is generic over the writer type, allowing it to write to
/// files, network streams, or in-memory buffers.
///
/// # Type Parameters
///
/// * `W` - The writer type that implements [`std::io::Write`].
///
/// # Example
///
/// ```ignore
/// use qbice_serialize::postcard::PostcardEncoder;
/// use std::io::Cursor;
///
/// // Write to a Vec<u8>
/// let encoder = PostcardEncoder::new(Vec::new());
///
/// // Write to a file
/// let file = std::fs::File::create("output.bin").unwrap();
/// let encoder = PostcardEncoder::new(file);
///
/// // Write to a cursor
/// let cursor = Cursor::new(Vec::new());
/// let encoder = PostcardEncoder::new(cursor);
/// ```
#[derive(Debug)]
pub struct PostcardEncoder<W> {
    writer: W,
}

impl<W> PostcardEncoder<W> {
    /// Creates a new encoder wrapping the given writer.
    #[must_use]
    pub const fn new(writer: W) -> Self { Self { writer } }

    /// Returns a reference to the underlying writer.
    #[must_use]
    pub const fn get_ref(&self) -> &W { &self.writer }

    /// Returns a mutable reference to the underlying writer.
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn get_mut(&mut self) -> &mut W { &mut self.writer }

    /// Consumes the encoder and returns the underlying writer.
    #[must_use]
    pub fn into_inner(self) -> W { self.writer }
}

impl<W: Write> Encoder for PostcardEncoder<W> {
    fn emit_u8(&mut self, v: u8) -> io::Result<()> {
        self.writer.write_all(&[v])
    }

    fn emit_u16(&mut self, v: u16) -> io::Result<()> {
        let mut buf = [0u8; MAX_VARINT_U16_BYTES];
        let len = encode_varint_u16(v, &mut buf);
        self.writer.write_all(&buf[..len])
    }

    fn emit_u32(&mut self, v: u32) -> io::Result<()> {
        let mut buf = [0u8; MAX_VARINT_U32_BYTES];
        let len = encode_varint_u32(v, &mut buf);
        self.writer.write_all(&buf[..len])
    }

    fn emit_u64(&mut self, v: u64) -> io::Result<()> {
        let mut buf = [0u8; MAX_VARINT_U64_BYTES];
        let len = encode_varint_u64(v, &mut buf);
        self.writer.write_all(&buf[..len])
    }

    fn emit_u128(&mut self, v: u128) -> io::Result<()> {
        let mut buf = [0u8; MAX_VARINT_U128_BYTES];
        let len = encode_varint_u128(v, &mut buf);
        self.writer.write_all(&buf[..len])
    }

    #[allow(clippy::cast_possible_truncation)]
    fn emit_usize(&mut self, v: usize) -> io::Result<()> {
        // Encode as u64 for portability
        self.emit_u64(v as u64)
    }

    #[allow(clippy::cast_sign_loss)]
    fn emit_i8(&mut self, v: i8) -> io::Result<()> {
        self.writer.write_all(&[v as u8])
    }

    fn emit_i16(&mut self, v: i16) -> io::Result<()> {
        self.emit_u16(zigzag_encode_i16(v))
    }

    fn emit_i32(&mut self, v: i32) -> io::Result<()> {
        self.emit_u32(zigzag_encode_i32(v))
    }

    fn emit_i64(&mut self, v: i64) -> io::Result<()> {
        self.emit_u64(zigzag_encode_i64(v))
    }

    fn emit_i128(&mut self, v: i128) -> io::Result<()> {
        self.emit_u128(zigzag_encode_i128(v))
    }

    fn emit_isize(&mut self, v: isize) -> io::Result<()> {
        // Encode as i64 for portability
        self.emit_i64(v as i64)
    }

    fn emit_raw_bytes(&mut self, s: &[u8]) -> io::Result<()> {
        self.writer.write_all(s)
    }

    // Override for efficiency - use varint for char
    fn emit_char(&mut self, v: char) -> io::Result<()> {
        self.emit_u32(v as u32)
    }

    // Override to use fixed little-endian for floats
    fn emit_f32(&mut self, v: f32) -> io::Result<()> {
        self.writer.write_all(&v.to_le_bytes())
    }

    fn emit_f64(&mut self, v: f64) -> io::Result<()> {
        self.writer.write_all(&v.to_le_bytes())
    }
}

// =============================================================================
// PostcardDecoder
// =============================================================================

/// A postcard-style decoder that reads from any [`Read`] implementation.
///
/// Uses varint decoding for integers. The decoder is generic over the reader
/// type, allowing it to read from files, network streams, or in-memory buffers.
///
/// # Type Parameters
///
/// * `R` - The reader type that implements [`std::io::Read`].
///
/// # Example
///
/// ```ignore
/// use qbice_serialize::postcard::PostcardDecoder;
/// use std::io::Cursor;
///
/// // Read from a slice
/// let data = [42u8];
/// let decoder = PostcardDecoder::new(&data[..]);
///
/// // Read from a file
/// let file = std::fs::File::open("input.bin").unwrap();
/// let decoder = PostcardDecoder::new(file);
///
/// // Read from a cursor
/// let cursor = Cursor::new(vec![42u8]);
/// let decoder = PostcardDecoder::new(cursor);
/// ```
#[derive(Debug)]
pub struct PostcardDecoder<R> {
    reader: R,
}

impl<R> PostcardDecoder<R> {
    /// Creates a new decoder wrapping the given reader.
    #[must_use]
    pub const fn new(reader: R) -> Self { Self { reader } }

    /// Returns a reference to the underlying reader.
    #[must_use]
    pub const fn get_ref(&self) -> &R { &self.reader }

    /// Returns a mutable reference to the underlying reader.
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn get_mut(&mut self) -> &mut R { &mut self.reader }

    /// Consumes the decoder and returns the underlying reader.
    #[must_use]
    pub fn into_inner(self) -> R { self.reader }
}

impl<R: Read> PostcardDecoder<R> {
    /// Reads a single byte from the reader.
    fn read_byte(&mut self) -> io::Result<u8> {
        let mut buf = [0u8; 1];
        self.reader.read_exact(&mut buf)?;
        Ok(buf[0])
    }

    /// Reads a varint-encoded u16.
    fn read_varint_u16(&mut self) -> io::Result<u16> {
        let mut result: u16 = 0;
        let mut shift = 0;

        loop {
            let byte = self.read_byte()?;

            if shift >= 16 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "varint too long for u16",
                ));
            }

            result |= u16::from(byte & 0x7F) << shift;

            if byte & 0x80 == 0 {
                return Ok(result);
            }

            shift += 7;
        }
    }

    /// Reads a varint-encoded u32.
    fn read_varint_u32(&mut self) -> io::Result<u32> {
        let mut result: u32 = 0;
        let mut shift = 0;

        loop {
            let byte = self.read_byte()?;

            if shift >= 32 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "varint too long for u32",
                ));
            }

            result |= u32::from(byte & 0x7F) << shift;

            if byte & 0x80 == 0 {
                return Ok(result);
            }

            shift += 7;
        }
    }

    /// Reads a varint-encoded u64.
    fn read_varint_u64(&mut self) -> io::Result<u64> {
        let mut result: u64 = 0;
        let mut shift = 0;

        loop {
            let byte = self.read_byte()?;

            if shift >= 64 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "varint too long for u64",
                ));
            }

            result |= u64::from(byte & 0x7F) << shift;

            if byte & 0x80 == 0 {
                return Ok(result);
            }

            shift += 7;
        }
    }

    /// Reads a varint-encoded u128.
    fn read_varint_u128(&mut self) -> io::Result<u128> {
        let mut result: u128 = 0;
        let mut shift = 0;

        loop {
            let byte = self.read_byte()?;

            if shift >= 128 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "varint too long for u128",
                ));
            }

            result |= u128::from(byte & 0x7F) << shift;

            if byte & 0x80 == 0 {
                return Ok(result);
            }

            shift += 7;
        }
    }
}

impl<R: Read> Decoder for PostcardDecoder<R> {
    fn read_u8(&mut self) -> io::Result<u8> { self.read_byte() }

    fn read_u16(&mut self) -> io::Result<u16> { self.read_varint_u16() }

    fn read_u32(&mut self) -> io::Result<u32> { self.read_varint_u32() }

    fn read_u64(&mut self) -> io::Result<u64> { self.read_varint_u64() }

    fn read_u128(&mut self) -> io::Result<u128> { self.read_varint_u128() }

    fn read_usize(&mut self) -> io::Result<usize> {
        // Decode from u64 for portability
        let value = self.read_u64()?;
        usize::try_from(value).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "usize value out of range for this platform",
            )
        })
    }

    #[allow(clippy::cast_possible_wrap)]
    fn read_i8(&mut self) -> io::Result<i8> { Ok(self.read_u8()? as i8) }

    fn read_i16(&mut self) -> io::Result<i16> {
        Ok(zigzag_decode_i16(self.read_varint_u16()?))
    }

    fn read_i32(&mut self) -> io::Result<i32> {
        Ok(zigzag_decode_i32(self.read_varint_u32()?))
    }

    fn read_i64(&mut self) -> io::Result<i64> {
        Ok(zigzag_decode_i64(self.read_varint_u64()?))
    }

    fn read_i128(&mut self) -> io::Result<i128> {
        Ok(zigzag_decode_i128(self.read_varint_u128()?))
    }

    fn read_isize(&mut self) -> io::Result<isize> {
        // Decode from i64 for portability
        let value = self.read_i64()?;
        isize::try_from(value).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "isize value out of range for this platform",
            )
        })
    }

    fn read_raw_bytes(&mut self, len: usize) -> io::Result<Vec<u8>> {
        let mut buf = vec![0u8; len];
        self.reader.read_exact(&mut buf)?;
        Ok(buf)
    }

    // Override for efficiency - use varint for char
    fn read_char(&mut self) -> io::Result<char> {
        let code = self.read_u32()?;
        char::from_u32(code).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid Unicode scalar value: {code}"),
            )
        })
    }

    // Override to use fixed little-endian for floats
    fn read_f32(&mut self) -> io::Result<f32> {
        let mut buf = [0u8; 4];
        self.reader.read_exact(&mut buf)?;
        Ok(f32::from_le_bytes(buf))
    }

    fn read_f64(&mut self) -> io::Result<f64> {
        let mut buf = [0u8; 8];
        self.reader.read_exact(&mut buf)?;
        Ok(f64::from_le_bytes(buf))
    }
}

// =============================================================================
// Convenience functions
// =============================================================================

/// Encodes a value using the postcard format.
///
/// This is a convenience function that creates a [`PostcardEncoder`], encodes
/// the value, and returns the resulting bytes.
///
/// # Example
///
/// ```ignore
/// use qbice_serialize::{Encode, Plugin};
/// use qbice_serialize::postcard::encode;
///
/// let plugin = Plugin::new();
/// let bytes = encode(&42u32, &plugin).unwrap();
/// ```
pub fn encode<T: crate::Encode>(
    value: &T,
    plugin: &crate::Plugin,
) -> io::Result<Vec<u8>> {
    let mut encoder = PostcardEncoder::new(Vec::new());
    value.encode(&mut encoder, plugin)?;
    Ok(encoder.into_inner())
}

/// Decodes a value from bytes using the postcard format.
///
/// This is a convenience function that creates a [`PostcardDecoder`] and
/// decodes the value.
///
/// # Example
///
/// ```ignore
/// use qbice_serialize::{Decode, Plugin};
/// use qbice_serialize::postcard::decode;
///
/// let plugin = Plugin::new();
/// let bytes = [42]; // varint-encoded 42
/// let value: u32 = decode(&bytes, &plugin).unwrap();
/// ```
pub fn decode<T: crate::Decode>(
    bytes: &[u8],
    plugin: &crate::Plugin,
) -> io::Result<T> {
    let mut decoder = PostcardDecoder::new(bytes);
    T::decode(&mut decoder, plugin)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Plugin;

    #[test]
    fn test_varint_u32_roundtrip() {
        let plugin = Plugin::new();
        let test_values: &[u32] =
            &[0, 1, 127, 128, 255, 256, 16383, 16384, u32::MAX / 2, u32::MAX];

        for &value in test_values {
            let bytes = encode(&value, &plugin).unwrap();
            let decoded: u32 = decode(&bytes, &plugin).unwrap();
            assert_eq!(value, decoded, "roundtrip failed for {value}");
        }
    }

    #[test]
    fn test_zigzag_i32_roundtrip() {
        let plugin = Plugin::new();
        let test_values: &[i32] =
            &[0, 1, -1, 63, -64, 64, -65, 127, -128, i32::MAX, i32::MIN];

        for &value in test_values {
            let bytes = encode(&value, &plugin).unwrap();
            let decoded: i32 = decode(&bytes, &plugin).unwrap();
            assert_eq!(value, decoded, "roundtrip failed for {value}");
        }
    }

    #[test]
    fn test_varint_compactness() {
        let plugin = Plugin::new();

        // Small values should be compact
        let bytes = encode(&0u32, &plugin).unwrap();
        assert_eq!(bytes.len(), 1);

        let bytes = encode(&127u32, &plugin).unwrap();
        assert_eq!(bytes.len(), 1);

        let bytes = encode(&128u32, &plugin).unwrap();
        assert_eq!(bytes.len(), 2);
    }

    #[test]
    fn test_string_roundtrip() {
        let plugin = Plugin::new();
        let value = "Hello, World!".to_string();

        let bytes = encode(&value, &plugin).unwrap();
        let decoded: String = decode(&bytes, &plugin).unwrap();
        assert_eq!(value, decoded);
    }

    #[test]
    fn test_float_roundtrip() {
        let plugin = Plugin::new();

        let f32_value: f32 = 3.14159;
        let bytes = encode(&f32_value, &plugin).unwrap();
        assert_eq!(bytes.len(), 4); // f32 is always 4 bytes
        let decoded: f32 = decode(&bytes, &plugin).unwrap();
        assert_eq!(f32_value, decoded);

        let f64_value: f64 = std::f64::consts::PI;
        let bytes = encode(&f64_value, &plugin).unwrap();
        assert_eq!(bytes.len(), 8); // f64 is always 8 bytes
        let decoded: f64 = decode(&bytes, &plugin).unwrap();
        assert_eq!(f64_value, decoded);
    }
}
