//! # ManifoldExpr Derive Macro
//!
//! Generates implementations for the `ManifoldExpr` marker trait,
//! which gates access to `ManifoldExt` methods.
//!
//! ## Usage
//!
//! ```ignore
//! #[derive(ManifoldExpr)]
//! pub struct Sqrt<M>(pub M);
//! ```
//!
//! ## Generated Code
//!
//! For `Sqrt<M>`:
//! ```ignore
//! impl<M> ::pixelflow_core::ManifoldExpr for Sqrt<M> {}
//! ```
//!
//! ## Future: Chained Ops
//!
//! This macro could be extended to also generate operator impls
//! (Add, Sub, Mul, Div) currently handled by `impl_chained_ops!`.

use proc_macro2::TokenStream;
use quote::quote;
use syn::{DeriveInput, GenericParam, Generics, parse_quote};

/// Generate the `ManifoldExpr` impl for a type.
pub fn derive_manifold_expr(input: DeriveInput) -> TokenStream {
    let name = &input.ident;
    let generics = &input.generics;

    // Extract generic parameters for the impl
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    quote! {
        impl #impl_generics ::pixelflow_core::ManifoldExpr for #name #ty_generics #where_clause {}
    }
}
