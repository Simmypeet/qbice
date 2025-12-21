//! Derive macros for the `Encode` and `Decode` traits.
//!
//! This crate provides derive macros for automatically implementing the
//! [`Encode`] and [`Decode`] traits from the `qbice_serialize` crate.
//!
//! # Supported Types
//!
//! The derive macros support:
//! - Structs with named fields
//! - Tuple structs
//! - Unit structs
//! - Enums with any combination of unit, tuple, and struct variants
//!
//! # Examples
//!
//! ## Struct with Named Fields
//!
//! ```ignore
//! use qbice_serialize_derive::{Encode, Decode};
//!
//! #[derive(Encode, Decode)]
//! struct Point {
//!     x: i32,
//!     y: i32,
//! }
//! ```
//!
//! ## Tuple Struct
//!
//! ```ignore
//! use qbice_serialize_derive::{Encode, Decode};
//!
//! #[derive(Encode, Decode)]
//! struct Id(u64);
//! ```
//!
//! ## Unit Struct
//!
//! ```ignore
//! use qbice_serialize_derive::{Encode, Decode};
//!
//! #[derive(Encode, Decode)]
//! struct Marker;
//! ```
//!
//! ## Enum
//!
//! ```ignore
//! use qbice_serialize_derive::{Encode, Decode};
//!
//! #[derive(Encode, Decode)]
//! enum Message {
//!     Quit,
//!     Move { x: i32, y: i32 },
//!     Write(String),
//! }
//! ```
//!
//! # Field Attributes
//!
//! ## `#[serialize(skip)]`
//!
//! Skip a field during serialization and use `Default::default()` during
//! deserialization.
//!
//! ```ignore
//! use qbice_serialize_derive::{Encode, Decode};
//!
//! #[derive(Encode, Decode)]
//! struct Config {
//!     name: String,
//!     #[serialize(skip)]
//!     cache: Vec<u8>, // Uses Default::default() when decoding
//! }
//! ```

use proc_macro::TokenStream;
use quote::quote;
use syn::{
    Data, DataEnum, DataStruct, DeriveInput, Field, Fields, Index,
    parse_macro_input,
};

/// Checks if a field has the `#[serialize(skip)]` attribute.
fn should_skip(field: &Field) -> bool {
    field.attrs.iter().any(|attr| {
        if !attr.path().is_ident("serialize") {
            return false;
        }
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("skip") {
                Ok(())
            } else {
                Err(meta.error("unknown serialize attribute"))
            }
        })
        .is_ok()
    })
}

/// Derive macro for `Encode`.
///
/// This macro automatically implements the `Encode` trait for structs and
/// enums. The implementation ensures that:
///
/// - For structs: all non-skipped fields are encoded in declaration order
/// - For enums: the variant index is encoded first (as `usize`), followed by
///   any variant data
///
/// # Struct Example
///
/// ```ignore
/// use qbice_serialize_derive::Encode;
///
/// #[derive(Encode)]
/// struct Person {
///     name: String,
///     age: u32,
/// }
/// ```
///
/// # Enum Example
///
/// ```ignore
/// use qbice_serialize_derive::Encode;
///
/// #[derive(Encode)]
/// enum Color {
///     Red,
///     Green,
///     Blue,
///     Rgb(u8, u8, u8),
///     Named { name: String },
/// }
/// ```
#[proc_macro_derive(Encode, attributes(serialize))]
pub fn derive_encode(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) =
        input.generics.split_for_impl();

    // Build where clause for Encode bounds
    let mut where_clause =
        where_clause.cloned().unwrap_or_else(|| syn::parse_quote!(where));

    // Add Encode bounds for all generic type parameters
    for param in &input.generics.params {
        if let syn::GenericParam::Type(type_param) = param {
            let ident = &type_param.ident;
            where_clause
                .predicates
                .push(syn::parse_quote!(#ident: ::qbice_serialize::Encode));
        }
    }

    let encode_impl = match &input.data {
        Data::Struct(data_struct) => impl_encode_struct(data_struct),
        Data::Enum(data_enum) => impl_encode_enum(data_enum),
        Data::Union(_) => {
            return syn::Error::new_spanned(
                &input,
                "Encode cannot be derived for unions due to memory safety \
                 concerns",
            )
            .to_compile_error()
            .into();
        }
    };

    let expanded = quote! {
        #[allow(clippy::trait_duplication_in_bounds)]
        impl #impl_generics ::qbice_serialize::Encode for #name #ty_generics #where_clause {
            fn encode<__E: ::qbice_serialize::Encoder + ?Sized>(
                &self,
                encoder: &mut __E,
                plugin: &::qbice_serialize::Plugin,
                session: &mut ::qbice_serialize::session::Session,
            ) -> ::std::io::Result<()> {
                #encode_impl
            }
        }
    };

    TokenStream::from(expanded)
}

fn impl_encode_struct(data_struct: &DataStruct) -> proc_macro2::TokenStream {
    match &data_struct.fields {
        Fields::Named(fields) => {
            let field_encodes = fields
                .named
                .iter()
                .filter(|field| !should_skip(field))
                .map(|field| {
                    let field_name = &field.ident;
                    quote! {
                        ::qbice_serialize::Encode::encode(&self.#field_name, encoder, plugin, session)?;
                    }
                });

            quote! {
                #(#field_encodes)*
                Ok(())
            }
        }
        Fields::Unnamed(fields) => {
            let field_encodes = fields
                .unnamed
                .iter()
                .enumerate()
                .filter(|(_, field)| !should_skip(field))
                .map(|(i, _)| {
                    let index = Index::from(i);
                    quote! {
                        ::qbice_serialize::Encode::encode(&self.#index, encoder, plugin, session)?;
                    }
                });

            quote! {
                #(#field_encodes)*
                Ok(())
            }
        }
        Fields::Unit => {
            quote! {
                // Unit struct has no fields to encode
                Ok(())
            }
        }
    }
}

fn impl_encode_enum(data_enum: &DataEnum) -> proc_macro2::TokenStream {
    let variant_matches =
        data_enum.variants.iter().enumerate().map(|(idx, variant)| {
            let variant_name = &variant.ident;

            match &variant.fields {
                Fields::Named(fields) => {
                    let field_names: Vec<_> = fields
                        .named
                        .iter()
                        .map(|f| (&f.ident, should_skip(f)))
                        .collect();

                    let pattern_bindings = field_names.iter().map(|(name, skip)| {
                        if *skip {
                            quote! { #name: _ }
                        } else {
                            quote! { #name }
                        }
                    });

                    let field_encodes =
                        field_names.iter().filter(|(_, skip)| !skip).map(
                            |(field_name, _)| {
                                quote! {
                                    ::qbice_serialize::Encode::encode(#field_name, encoder, plugin, session)?;
                                }
                            },
                        );

                    quote! {
                        Self::#variant_name { #(#pattern_bindings),* } => {
                            encoder.emit_usize(#idx)?;
                            #(#field_encodes)*
                        }
                    }
                }
                Fields::Unnamed(fields) => {
                    let field_data: Vec<_> = fields
                        .unnamed
                        .iter()
                        .enumerate()
                        .map(|(i, f)| {
                            let binding = syn::Ident::new(
                                &format!("field_{i}"),
                                proc_macro2::Span::call_site(),
                            );
                            (binding, should_skip(f))
                        })
                        .collect();

                    let pattern_bindings = field_data.iter().map(|(binding, skip)| {
                        if *skip {
                            quote! { _ }
                        } else {
                            quote! { #binding }
                        }
                    });

                    let field_encodes =
                        field_data.iter().filter(|(_, skip)| !skip).map(
                            |(binding, _)| {
                                quote! {
                                    ::qbice_serialize::Encode::encode(#binding, encoder, plugin, session)?;
                                }
                            },
                        );

                    quote! {
                        Self::#variant_name(#(#pattern_bindings),*) => {
                            encoder.emit_usize(#idx)?;
                            #(#field_encodes)*
                        }
                    }
                }
                Fields::Unit => {
                    quote! {
                        Self::#variant_name => {
                            encoder.emit_usize(#idx)?;
                        }
                    }
                }
            }
        });

    quote! {
        match self {
            #(#variant_matches)*
        }
        Ok(())
    }
}

/// Derive macro for `Decode`.
///
/// This macro automatically implements the `Decode` trait for structs and
/// enums. The implementation ensures that:
///
/// - For structs: all fields are decoded in declaration order (skipped fields
///   use `Default::default()`)
/// - For enums: the variant index is decoded first (as `usize`), then the
///   variant data is decoded
///
/// # Struct Example
///
/// ```ignore
/// use qbice_serialize_derive::Decode;
///
/// #[derive(Decode)]
/// struct Person {
///     name: String,
///     age: u32,
/// }
/// ```
///
/// # Enum Example
///
/// ```ignore
/// use qbice_serialize_derive::Decode;
///
/// #[derive(Decode)]
/// enum Color {
///     Red,
///     Green,
///     Blue,
///     Rgb(u8, u8, u8),
///     Named { name: String },
/// }
/// ```
#[proc_macro_derive(Decode, attributes(serialize))]
pub fn derive_decode(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) =
        input.generics.split_for_impl();

    // Build where clause for Decode bounds
    let mut where_clause =
        where_clause.cloned().unwrap_or_else(|| syn::parse_quote!(where));

    // Add Decode bounds for all generic type parameters
    for param in &input.generics.params {
        if let syn::GenericParam::Type(type_param) = param {
            let ident = &type_param.ident;
            where_clause
                .predicates
                .push(syn::parse_quote!(#ident: ::qbice_serialize::Decode));
        }
    }

    let decode_impl = match &input.data {
        Data::Struct(data_struct) => impl_decode_struct(data_struct),
        Data::Enum(data_enum) => impl_decode_enum(name, data_enum),
        Data::Union(_) => {
            return syn::Error::new_spanned(
                &input,
                "Decode cannot be derived for unions due to memory safety \
                 concerns",
            )
            .to_compile_error()
            .into();
        }
    };

    let expanded = quote! {
        #[allow(clippy::trait_duplication_in_bounds)]
        impl #impl_generics ::qbice_serialize::Decode for #name #ty_generics #where_clause {
            fn decode<__D: ::qbice_serialize::Decoder + ?Sized>(
                decoder: &mut __D,
                plugin: &::qbice_serialize::Plugin,
                session: &mut ::qbice_serialize::session::Session,
            ) -> ::std::io::Result<Self> {
                #decode_impl
            }
        }
    };

    TokenStream::from(expanded)
}

fn impl_decode_struct(data_struct: &DataStruct) -> proc_macro2::TokenStream {
    match &data_struct.fields {
        Fields::Named(fields) => {
            let field_decodes = fields.named.iter().map(|field| {
                let field_name = &field.ident;
                let field_type = &field.ty;

                if should_skip(field) {
                    quote! {
                        #field_name: <#field_type as ::std::default::Default>::default(),
                    }
                } else {
                    quote! {
                        #field_name: <#field_type as ::qbice_serialize::Decode>::decode(decoder, plugin, session)?,
                    }
                }
            });

            quote! {
                Ok(Self {
                    #(#field_decodes)*
                })
            }
        }
        Fields::Unnamed(fields) => {
            let field_decodes = fields.unnamed.iter().map(|field| {
                let field_type = &field.ty;

                if should_skip(field) {
                    quote! {
                        <#field_type as ::std::default::Default>::default(),
                    }
                } else {
                    quote! {
                        <#field_type as ::qbice_serialize::Decode>::decode(decoder, plugin, session)?,
                    }
                }
            });

            quote! {
                Ok(Self(#(#field_decodes)*))
            }
        }
        Fields::Unit => {
            quote! {
                Ok(Self)
            }
        }
    }
}

fn impl_decode_enum(
    name: &syn::Ident,
    data_enum: &DataEnum,
) -> proc_macro2::TokenStream {
    let variant_count = data_enum.variants.len();
    let variant_matches =
        data_enum.variants.iter().enumerate().map(|(idx, variant)| {
            let variant_name = &variant.ident;

            match &variant.fields {
                Fields::Named(fields) => {
                    let field_decodes = fields.named.iter().map(|field| {
                        let field_name = &field.ident;
                        let field_type = &field.ty;

                        if should_skip(field) {
                            quote! {
                                #field_name: <#field_type as ::std::default::Default>::default(),
                            }
                        } else {
                            quote! {
                                #field_name: <#field_type as ::qbice_serialize::Decode>::decode(decoder, plugin, session)?,
                            }
                        }
                    });

                    quote! {
                        #idx => Ok(Self::#variant_name { #(#field_decodes)* }),
                    }
                }
                Fields::Unnamed(fields) => {
                    let field_decodes = fields.unnamed.iter().map(|field| {
                        let field_type = &field.ty;

                        if should_skip(field) {
                            quote! {
                                <#field_type as ::std::default::Default>::default(),
                            }
                        } else {
                            quote! {
                                <#field_type as ::qbice_serialize::Decode>::decode(decoder, plugin, session)?,
                            }
                        }
                    });

                    quote! {
                        #idx => Ok(Self::#variant_name(#(#field_decodes)*)),
                    }
                }
                Fields::Unit => {
                    quote! {
                        #idx => Ok(Self::#variant_name),
                    }
                }
            }
        });

    let name_str = name.to_string();

    quote! {
        let variant_idx = decoder.read_usize()?;
        match variant_idx {
            #(#variant_matches)*
            _ => Err(::std::io::Error::new(
                ::std::io::ErrorKind::InvalidData,
                ::std::format!(
                    "invalid variant index {} for enum {} (expected 0..{})",
                    variant_idx,
                    #name_str,
                    #variant_count
                ),
            )),
        }
    }
}
