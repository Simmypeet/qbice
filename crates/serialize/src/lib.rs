//! QBICE Serialization Library
//!
//! This crate provides traits and implementations for compact binary
//! serialization and deserialization. The design is inspired by
//! `rustc_serialize` with extensions for plugin-based extensibility.
//!
//! # Overview
//!
//! The crate provides four core traits:
//!
//! - [`Encoder`]: Low-level trait for emitting primitive values
//! - [`Encode`]: High-level trait for types that can be serialized
//! - [`Decoder`]: Low-level trait for reading primitive values
//! - [`Decode`]: High-level trait for types that can be deserialized
//!
//! The [`Plugin`] system allows passing additional context to serialization
//! operations, enabling custom serialization strategies without modifying the
//! core traits.
//!
//! # Example
//!
//! ```ignore
//! use qbice_serialize::{Encode, Decode, Encoder, Decoder, Plugin};
//!
//! // Implement Encode/Decode for custom types
//! struct Point { x: i32, y: i32 }
//!
//! impl Encode for Point {
//!     fn encode<E: Encoder + ?Sized>(
//!         &self,
//!         encoder: &mut E,
//!         plugin: &Plugin,
//!     ) -> std::io::Result<()> {
//!         self.x.encode(encoder, plugin)?;
//!         self.y.encode(encoder, plugin)?;
//!         Ok(())
//!     }
//! }
//!
//! impl Decode for Point {
//!     fn decode<D: Decoder + ?Sized>(
//!         decoder: &mut D,
//!         plugin: &Plugin,
//!     ) -> std::io::Result<Self> {
//!         Ok(Point {
//!             x: i32::decode(decoder, plugin)?,
//!             y: i32::decode(decoder, plugin)?,
//!         })
//!     }
//! }
//! ```

pub mod decode;
pub mod encode;
pub mod plugin;

// Re-export main traits and types at the crate root for convenience
pub use decode::{Decode, Decoder};
pub use encode::{Encode, Encoder};
pub use plugin::Plugin;
