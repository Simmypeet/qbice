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
//! # Derive Macros
//!
//! This crate re-exports derive macros for automatically implementing
//! `Encode` and `Decode`:
//!
//! ```ignore
//! #[derive(Encode, Decode)]
//! struct Point { x: i32, y: i32 }
//!
//! #[derive(Encode, Decode)]
//! struct Id(u64);
//!
//! #[derive(Encode, Decode)]
//! struct Marker;
//!
//! #[derive(Encode, Decode)]
//! enum Message {
//!     Quit,
//!     Move { x: i32, y: i32 },
//!     Write(String),
//! }
//! ```
//!
//! ## Field Attributes
//!
//! Use `#[serialize(skip)]` to skip a field during serialization. The field
//! must implement `Default` for deserialization:
//!
//! ```ignore
//! #[derive(Encode, Decode)]
//! struct Config {
//!     name: String,
//!     #[serialize(skip)]
//!     cache: Vec<u8>, // Uses Default::default() when decoding
//! }
//! ```
//!
//! # Example
//!
//! ```ignore
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

// Allow derive macros to reference this crate as `qbice_serialize` internally
extern crate self as qbice_serialize;

pub mod decode;
pub mod encode;
pub mod plugin;
pub mod postcard;
pub mod session;

// Re-export main traits and types at the crate root for convenience
pub use decode::{Decode, Decoder};
pub use encode::{Encode, Encoder};
pub use plugin::Plugin;
pub use postcard::{PostcardDecoder, PostcardEncoder};
// Re-export derive macros
pub use qbice_serialize_derive::{Decode, Encode};
