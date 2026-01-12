//! Derive and attribute macros for the QBICE engine.
//!
//! This crate provides:
//! - `#[derive(Query)]`: Automatically implements the `Query` trait
//! - `#[derive_for_query_id]`: Convenience macro that adds all essential
//!   derives
//! - `#[executor]`: Generates an executor implementation from an async function
//!
//! # Query Derive
//!
//! To derive the `Query` trait, use the `#[derive(Query)]` attribute along
//! with a `#[value(...)]` attribute to specify the associated `Value` type:
//!
//! ```ignore
//! use qbice_derive::Query;
//!
//! #[derive(Query)]
//! #[value(Vec<String>)]
//! struct MyQuery {
//!     id: u64,
//!     name: String,
//! }
//! ```
//!
//! The type must also implement the required super-traits:
//! - `StableHash`
//! - `Identifiable`
//! - `Eq`, `Hash`, `Clone`, `Debug`
//! - `Encode`, `Decode`
//! - `Send`, `Sync`
//!
//! # Query ID Derive Helper
//!
//! For convenience, use `#[derive_for_query_id]` to automatically add all
//! essential trait derives:
//!
//! ```ignore
//! use qbice_derive::derive_for_query_id;
//!
//! #[derive_for_query_id]
//! #[value(Vec<String>)]
//! struct MyQuery {
//!     id: u64,
//! }
//! ```
//!
//! # Executor Attribute
//!
//! The `#[executor]` attribute macro generates an executor implementation from
//! an async function:
//!
//! ```ignore
//! use qbice_derive::executor;
//!
//! #[executor]
//! async fn my_query_executor<C: Config>(
//!     query: &MyQuery,
//!     engine: &TrackedEngine<C>,
//! ) -> Result<String, CyclicError> {
//!     // Implementation
//! }
//! ```

use proc_macro::TokenStream;
use quote::{ToTokens, format_ident, quote};
use syn::{
    DeriveInput, Error, FnArg, GenericParam, Ident, ItemFn, Meta, Pat, PatType,
    ReturnType, Type, parse_macro_input,
};

/// Derive macro for the `Query` trait.
///
/// This macro implements the `Query` trait for a type, using the
/// `#[value(...)]` attribute to specify the associated `Value` type.
///
/// **Note:** This macro automatically implements the `Identifiable` trait for
/// the type, so you don't need to derive it separately.
///
/// # Required Attribute
///
/// - `#[value(Type)]`: Specifies the associated `Value` type for the query
///
/// # Optional Attribute
///
/// - `#[extend(name = trait_name)]`: Generates an extension trait that provides
///   a convenience method for querying
/// - `#[extend(name = trait_name, by_val)]`: Generates an extension trait where
///   each field becomes a function parameter (only works with named fields or
///   unit structs)
///
/// # Example
///
/// ```ignore
/// #[derive(Query)]
/// #[value(String)]
/// struct NameQuery {
///     id: u64,
/// }
/// ```
///
/// This will generate:
///
/// ```ignore
/// impl Query for NameQuery {
///     type Value = String;
/// }
/// ```
///
/// # Extension Trait Examples
///
/// ## By reference (default):
///
/// ```ignore
/// #[derive(Query)]
/// #[value(String)]
/// #[extend(name = get_name)]
/// struct NameQuery {
///     id: u64,
/// }
/// ```
///
/// This will generate the Query impl plus:
///
/// ```ignore
/// trait get_name {
///     async fn get_name<C: Config>(&self, q: &NameQuery) -> String;
/// }
///
/// impl<C: Config> get_name for TrackedEngine<C> {
///     async fn get_name(&self, q: &NameQuery) -> String {
///         self.query(q).await
///     }
/// }
/// ```
///
/// ## By value (with field parameters):
///
/// ```ignore
/// #[derive(Query)]
/// #[value(String)]
/// #[extend(name = get_name, by_val)]
/// struct NameQuery {
///     id: u64,
///     extra: String,
/// }
/// ```
///
/// This will generate:
///
/// ```ignore
/// trait get_name {
///     async fn get_name<C: Config>(&self, id: u64, extra: String) -> String;
/// }
///
/// impl<C: Config> get_name for TrackedEngine<C> {
///     async fn get_name(&self, id: u64, extra: String) -> String {
///         let q = NameQuery { id, extra };
///         self.query(&q).await
///     }
/// }
/// ```
#[proc_macro_derive(Query, attributes(value, extend))]
pub fn derive_query(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    match derive_query_impl(&input) {
        Ok(tokens) => tokens,
        Err(err) => err.to_compile_error().into(),
    }
}

#[allow(clippy::too_many_lines)]
fn derive_query_impl(input: &DeriveInput) -> Result<TokenStream, Error> {
    let name = &input.ident;
    let generics = &input.generics;
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    // Find the #[value(...)] attribute
    let value_type = input
        .attrs
        .iter()
        .find_map(|attr| {
            // Check if this is a #[value(...)] attribute
            if !attr.path().is_ident("value") {
                return None;
            }

            // Parse the attribute value
            match &attr.meta {
                Meta::List(meta_list) => {
                    // Parse the tokens inside #[value(...)]
                    match syn::parse2::<syn::Type>(meta_list.tokens.clone()) {
                        Ok(ty) => Some(Ok(ty)),
                        Err(e) => Some(Err(e)),
                    }
                }
                _ => Some(Err(Error::new_spanned(
                    attr,
                    "expected #[value(Type)] format",
                ))),
            }
        })
        .transpose()?
        .ok_or_else(|| {
            Error::new_spanned(
                input,
                "missing #[value(Type)] attribute - Query derive requires \
                 specifying the Value type",
            )
        })?;

    // Find the optional #[extend(...)] attribute
    let extend_trait = input
        .attrs
        .iter()
        .find_map(|attr| {
            if !attr.path().is_ident("extend") {
                return None;
            }

            match &attr.meta {
                Meta::List(meta_list) => {
                    Some(parse_extend_attribute(&meta_list.tokens))
                }
                _ => Some(Err(Error::new_spanned(
                    attr,
                    "expected #[extend(name = ...)] format",
                ))),
            }
        })
        .transpose()?;

    let query_impl = quote! {
        impl #impl_generics ::qbice::Query for #name #ty_generics #where_clause {
            type Value = #value_type;
        }
    };

    let extension_trait = if let Some((trait_name, by_val)) = extend_trait {
        let vis = &input.vis;

        // Copy doc comments from the struct to the trait and method
        let doc_attrs: Vec<_> = input
            .attrs
            .iter()
            .filter(|attr| attr.path().is_ident("doc"))
            .collect();

        if by_val {
            // Extract fields for by_val mode
            let fields = match &input.data {
                syn::Data::Struct(data) => match &data.fields {
                    syn::Fields::Named(fields) => &fields.named,
                    syn::Fields::Unit => {
                        // Unit structs are allowed (no parameters)
                        &syn::punctuated::Punctuated::new()
                    }
                    syn::Fields::Unnamed(_) => {
                        return Err(Error::new_spanned(
                            input,
                            "by_val extension only works with named fields or \
                             unit structs",
                        ));
                    }
                },
                _ => {
                    return Err(Error::new_spanned(
                        input,
                        "extend attribute can only be used on structs",
                    ));
                }
            };

            // Generate function parameters from struct fields
            let params: Vec<_> = fields
                .iter()
                .map(|f| {
                    let name = &f.ident;
                    let ty = &f.ty;
                    quote! { #name: #ty }
                })
                .collect();

            // Generate struct construction from parameters
            let field_names: Vec<_> = fields.iter().map(|f| &f.ident).collect();
            let struct_construction = quote! {
                let q = #name #ty_generics { #(#field_names),* };
            };

            quote! {
                #(#doc_attrs)*
                #[allow(non_camel_case_types)]
                #vis trait #trait_name {
                    #(#doc_attrs)*
                    async fn #trait_name(
                        &self,
                        #(#params),*
                    ) -> <#name #ty_generics as ::qbice::Query>::Value;
                }

                impl<C: ::qbice::Config> #trait_name for ::qbice::TrackedEngine<C> {
                    async fn #trait_name(
                        &self,
                        #(#params),*
                    ) -> <#name #ty_generics as ::qbice::Query>::Value {
                        #struct_construction
                        self.query(&q).await
                    }
                }
            }
        } else {
            // by_ref mode (default)
            let param = quote! { q: &#name #ty_generics };

            quote! {
                #(#doc_attrs)*
                #[allow(non_camel_case_types)]
                #vis trait #trait_name {
                    #(#doc_attrs)*
                    async fn #trait_name(
                        &self,
                        #param,
                    ) -> <#name #ty_generics as ::qbice::Query>::Value;
                }

                impl<C: ::qbice::Config> #trait_name for ::qbice::TrackedEngine<C> {
                    async fn #trait_name(
                        &self,
                        #param,
                    ) -> <#name #ty_generics as ::qbice::Query>::Value {
                        self.query(q).await
                    }
                }
            }
        }
    } else {
        quote! {}
    };

    // Automatically generate Identifiable implementation
    let identifiable_path: syn::Path =
        syn::parse_quote!(::qbice::stable_type_id::Identifiable);
    let stable_type_id_path: syn::Path =
        syn::parse_quote!(::qbice::stable_type_id::StableTypeID);

    let identifiable_impl =
        qbice_identifiable_derive_lib::implements_identifiable(
            name,
            generics.clone(),
            Some(&identifiable_path),
            Some(&stable_type_id_path),
        );

    let expanded = quote! {
        #query_impl
        #extension_trait
        #identifiable_impl
    };

    Ok(expanded.into())
}

/// Parse the #[extend(...)] attribute
fn parse_extend_attribute(
    tokens: &proc_macro2::TokenStream,
) -> Result<(Ident, bool), Error> {
    let mut name: Option<Ident> = None;
    let mut by_val = false;

    let parser = syn::meta::parser(|meta| {
        if meta.path.is_ident("name") {
            name = Some(meta.value()?.parse()?);
            Ok(())
        } else if meta.path.is_ident("by_val") {
            by_val = true;
            Ok(())
        } else {
            Err(meta.error("expected `name` or `by_val`"))
        }
    });

    syn::parse::Parser::parse2(parser, tokens.clone())?;

    let name = name.ok_or_else(|| {
        Error::new_spanned(
            tokens,
            "missing `name` argument in extend attribute",
        )
    })?;

    Ok((name, by_val))
}

/// Attribute macro that automatically adds all essential derive macros for a
/// Query type.
///
/// This is a convenience macro that expands a struct to include all the
/// necessary trait derives for use as a Query type in the qbice engine.
///
/// By default, `Copy` is included in the derives. Use
/// `#[derive_for_query_id(no_copy)]` to exclude it for types that contain
/// non-Copy fields.
///
/// # Example
///
/// ```ignore
/// use qbice_derive::derive_for_query_id;
///
/// #[derive_for_query_id]
/// #[value(String)]
/// pub struct MyQuery {
///     id: u64,
/// }
/// ```
///
/// Expands to:
///
/// ```ignore
/// #[derive(
///     Debug,
///     Clone,
///     Copy,
///     PartialEq,
///     Eq,
///     Hash,
///     StableHash,
///     Identifiable,
///     Encode,
///     Decode,
///     Query,
/// )]
/// #[value(String)]
/// pub struct MyQuery {
///     id: u64,
/// }
/// ```
///
/// For types with non-Copy fields:
///
/// ```ignore
/// #[derive_for_query_id(no_copy)]
/// #[value(Vec<String>)]
/// pub struct MyQuery {
///     data: String,
/// }
/// ```
#[proc_macro_attribute]
pub fn derive_for_query_id(
    attr: TokenStream,
    item: TokenStream,
) -> TokenStream {
    let mut input = parse_macro_input!(item as DeriveInput);

    // Parse the attribute to check for no_copy
    let no_copy = if attr.is_empty() {
        false
    } else {
        let attr_input = parse_macro_input!(attr as Ident);
        attr_input == "no_copy"
    };

    // Add the derive attribute with all necessary traits using fully qualified
    // paths
    let derive_attr: syn::Attribute = if no_copy {
        syn::parse_quote! {
            #[derive(
                Debug,
                Clone,
                PartialEq,
                Eq,
                PartialOrd,
                Ord,
                Hash,
                ::qbice::stable_hash::StableHash,
                ::qbice_serialize::Encode,
                ::qbice_serialize::Decode,
                ::qbice::Query,
            )]
        }
    } else {
        syn::parse_quote! {
            #[derive(
                Debug,
                Clone,
                Copy,
                PartialEq,
                Eq,
                PartialOrd,
                Ord,
                Hash,
                ::qbice::stable_hash::StableHash,
                ::qbice_serialize::Encode,
                ::qbice_serialize::Decode,
                ::qbice::Query,
            )]
        }
    };

    // Insert the derive attribute at the beginning
    input.attrs.insert(0, derive_attr);

    // Return the modified struct
    quote! {
        #input
    }
    .into()
}

/// Attribute macro for generating an executor implementation from an async
/// function.
///
/// This macro generates a unit struct with a PascalCase name derived from the
/// function name, and implements the `Executor` trait for it.
///
/// # Basic Usage
///
/// ```ignore
/// #[executor]
/// async fn my_query_executor<C: Config>(
///     query: &MyQuery,
///     engine: &TrackedEngine<C>,
/// ) -> Result<String, CyclicError> {
///     // Implementation
/// }
/// ```
///
/// This expands to:
///
/// ```ignore
/// #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
/// pub struct MyQueryExecutor;
///
/// const _: () = {
///     async fn my_query_executor<C: Config>(
///         query: &MyQuery,
///         engine: &TrackedEngine<C>,
///     ) -> Result<String, CyclicError> {
///         // Implementation
///     }
///
///     impl<C: Config> Executor<MyQuery, C> for MyQueryExecutor {
///         async fn execute(
///             &self,
///             query: &MyQuery,
///             engine: &TrackedEngine<C>,
///         ) -> Result<String, CyclicError> {
///             my_query_executor(query, engine).await
///         }
///     }
/// };
/// ```
///
/// # Custom Config Type
///
/// You can specify a concrete config type instead of a generic parameter:
///
/// ```ignore
/// #[executor(config = MyConfig)]
/// async fn my_query_executor(
///     query: &MyQuery,
///     engine: &TrackedEngine<MyConfig>,
/// ) -> Result<String, CyclicError> {
///     // Implementation
/// }
/// ```
///
/// # Requirements
///
/// - Function must be async
/// - Must have exactly one generic parameter `C: Config` (or use `config =
///   Type`)
/// - First parameter must be `&QueryType` (reference to a Query type)
/// - Second parameter must be `&TrackedEngine<C>` (or `&TrackedEngine<Type>`)
#[proc_macro_attribute]
#[allow(clippy::too_many_lines)]
pub fn executor(attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut input = parse_macro_input!(item as ItemFn);

    match executor_impl(attr.into(), &mut input) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

#[allow(clippy::too_many_lines)]
fn executor_impl(
    attr: proc_macro2::TokenStream,
    input: &mut ItemFn,
) -> Result<proc_macro2::TokenStream, Error> {
    // Parse attribute arguments
    let custom_config = if attr.is_empty() {
        None
    } else {
        // Parse config = Type
        let meta: Meta = syn::parse2(attr)?;
        match meta {
            Meta::NameValue(nv) if nv.path.is_ident("config") => {
                let config_type: Type =
                    syn::parse2(nv.value.to_token_stream())?;
                Some(config_type)
            }
            _ => {
                return Err(Error::new_spanned(
                    meta,
                    "expected #[executor(config = Type)] format",
                ));
            }
        }
    };

    // Verify function is async
    if input.sig.asyncness.is_none() {
        return Err(Error::new_spanned(
            &input.sig,
            "executor function must be async",
        ));
    }

    // Get function name and convert to PascalCase struct name
    let fn_name = &input.sig.ident;
    let struct_name = snake_to_pascal(&fn_name.to_string());
    let struct_ident = format_ident!("{}", struct_name);

    // Extract and validate generic parameters
    let (config_generic, config_bound) = if let Some(config_type) =
        &custom_config
    {
        (quote! {}, quote! { #config_type })
    } else {
        // Must have exactly one generic parameter C: Config
        let generics = &input.sig.generics;
        if generics.params.len() != 1 {
            return Err(Error::new_spanned(
                generics,
                "executor function must have exactly one generic parameter \
                 `C: Config`",
            ));
        }

        let generic_param = generics.params.first().unwrap();
        match generic_param {
            GenericParam::Type(type_param) => {
                let ident = &type_param.ident;

                // Check if it has Config bound
                let has_config_bound = type_param.bounds.iter().any(|bound| {
                    matches!(bound, syn::TypeParamBound::Trait(..))
                });

                if !has_config_bound {
                    return Err(Error::new_spanned(
                        type_param,
                        "generic parameter must have `Config` bound",
                    ));
                }

                (quote! { <#ident: ::qbice::Config> }, quote! { #ident })
            }
            _ => {
                return Err(Error::new_spanned(
                    generic_param,
                    "executor function must have a type parameter `C: Config`",
                ));
            }
        }
    };

    // Validate function parameters
    if input.sig.inputs.len() != 2 {
        return Err(Error::new_spanned(
            &input.sig.inputs,
            "executor function must have exactly 2 parameters: (query: \
             &QueryType, engine: &TrackedEngine<C>)",
        ));
    }

    // Extract query type from first parameter
    let query_type = match &input.sig.inputs[0] {
        FnArg::Typed(PatType { ty, .. }) => {
            if let Type::Reference(ref_ty) = &**ty {
                &*ref_ty.elem
            } else {
                return Err(Error::new_spanned(
                    ty,
                    "first parameter must be a reference (&QueryType)",
                ));
            }
        }
        FnArg::Receiver(_) => {
            return Err(Error::new_spanned(
                &input.sig.inputs[0],
                "first parameter must be a reference to a query type",
            ));
        }
    };

    // Validate second parameter is &TrackedEngine<C>
    match &input.sig.inputs[1] {
        FnArg::Typed(PatType { ty, .. }) => {
            if let Type::Reference(ref_ty) = &**ty {
                // Just check it's a reference, detailed type checking is left
                // to compiler
                if let Type::Path(_) = &*ref_ty.elem {
                    // OK
                } else {
                    return Err(Error::new_spanned(
                        ty,
                        "second parameter must be &TrackedEngine<C>",
                    ));
                }
            } else {
                return Err(Error::new_spanned(
                    ty,
                    "second parameter must be a reference (&TrackedEngine<C>)",
                ));
            }
        }
        FnArg::Receiver(_) => {
            return Err(Error::new_spanned(
                &input.sig.inputs[1],
                "second parameter must be &TrackedEngine<C>",
            ));
        }
    }

    // Extract return type
    let return_type = match &input.sig.output {
        ReturnType::Type(_, ty) => ty,
        ReturnType::Default => {
            return Err(Error::new_spanned(
                &input.sig,
                "executor function must have a return type",
            ));
        }
    };

    // Extract parameter names
    let param_names: Vec<_> = input
        .sig
        .inputs
        .iter()
        .filter_map(|arg| {
            if let FnArg::Typed(PatType { pat, .. }) = arg
                && let Pat::Ident(ident) = &**pat
            {
                return Some(&ident.ident);
            }
            None
        })
        .collect();

    if param_names.len() != 2 {
        return Err(Error::new_spanned(
            &input.sig.inputs,
            "could not extract parameter names",
        ));
    }

    // Get visibility
    let vis = &input.vis;

    // Clone the function for the const block
    let original_fn = input.clone();

    let expanded = quote! {
        #[derive(
            ::std::fmt::Debug,
            ::std::clone::Clone,
            ::std::marker::Copy,
            ::std::cmp::PartialEq,
            ::std::cmp::Eq,
            ::std::cmp::PartialOrd,
            ::std::cmp::Ord,
            ::std::hash::Hash,
            ::std::default::Default,
        )]
        #vis struct #struct_ident;

        const _: () = {
            #original_fn

            impl #config_generic ::qbice::Executor<#query_type, #config_bound>
                for #struct_ident
            {
                async fn execute(
                    &self,
                    a: &#query_type,
                    b: &::qbice::TrackedEngine<#config_bound>,
                ) -> #return_type {
                    #fn_name(a, b).await
                }
            }
        };
    };

    Ok(expanded)
}

/// Converts snake_case to PascalCase
fn snake_to_pascal(s: &str) -> String {
    s.split('_')
        .map(|word| {
            let mut chars = word.chars();
            chars.next().map_or_else(String::new, |first| {
                first.to_uppercase().collect::<String>() + chars.as_str()
            })
        })
        .collect()
}
