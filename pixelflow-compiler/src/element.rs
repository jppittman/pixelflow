//! # Element Derive Macro
//!
//! Generates operator implementations (`Add`, `Sub`, `BitAnd`, etc.) for combinator types.
//! This transforms a struct into an "Element" of the algebra, allowing it to be
//! composed naturally.
//!
//! # Mapping
//!
//! | Trait    | Result Type |
//! |----------|-------------|
//! | `Add`    | `Add<Self, Rhs>` |
//! | `Sub`    | `Sub<Self, Rhs>` |
//! | `Mul`    | `Mul<Self, Rhs>` |
//! | `Div`    | `Div<Self, Rhs>` |
//! | `Neg`    | `Neg<Self>` |
//! | `BitAnd` | `And<Self, Rhs>` |
//! | `BitOr`  | `Or<Self, Rhs>` |
//! | `Not`    | `BNot<Self>` |
//!
//! # Crate Resolution
//!
//! The macro detects if it's running inside `pixelflow-core` or an external crate
//! to generate the correct paths (`crate::ops::...` vs `::pixelflow_core::ops::...`).

use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::{DeriveInput, Ident};

pub fn derive_element(input: DeriveInput) -> TokenStream {
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    // Create generics with 'Rhs' for binary operators
    let mut generics_with_rhs = input.generics.clone();
    generics_with_rhs.params.push(syn::parse_quote!(Rhs));
    let (impl_generics_with_rhs, _, _) = generics_with_rhs.split_for_impl();

    // Detect if we are compiling pixelflow-core itself
    // If so, we use `crate::` paths. If not, `::pixelflow_core::`.
    // Note: This environment variable is set by Cargo for the crate being compiled.
    let pkg_name = std::env::var("CARGO_PKG_NAME").unwrap_or_default();
    let is_core = pkg_name == "pixelflow-core";

    let root = if is_core {
        quote! { crate }
    } else {
        quote! { ::pixelflow_core }
    };

    let ops_mod = quote! { #root::ops };

    // Helper to generate binary op impl
    let binary_op = |trait_name: &str, method: &str, node_name: &str, node_mod: &TokenStream| {
        let trait_ident = Ident::new(trait_name, Span::call_site());
        let method_ident = Ident::new(method, Span::call_site());
        let node_ident = Ident::new(node_name, Span::call_site());

        // Use impl_generics_with_rhs to introduce <..., Rhs>
        quote! {
            impl #impl_generics_with_rhs core::ops::#trait_ident<Rhs> for #name #ty_generics
            where
                #name #ty_generics: #root::ManifoldExpr,
                Rhs: #root::ManifoldExpr,
                #where_clause
            {
                type Output = #node_mod::#node_ident<Self, Rhs>;
                #[inline(always)]
                fn #method_ident(self, rhs: Rhs) -> Self::Output {
                    #node_mod::#node_ident(self, rhs)
                }
            }
        }
    };

    // Helper to generate unary op impl
    let unary_op = |trait_name: &str, method: &str, node_name: &str, node_mod: &TokenStream| {
        let trait_ident = Ident::new(trait_name, Span::call_site());
        let method_ident = Ident::new(method, Span::call_site());
        let node_ident = Ident::new(node_name, Span::call_site());

        // Use original impl_generics (no Rhs)
        quote! {
            impl #impl_generics core::ops::#trait_ident for #name #ty_generics
            where
                #name #ty_generics: #root::ManifoldExpr,
                #where_clause
            {
                type Output = #node_mod::#node_ident<Self>;
                #[inline(always)]
                fn #method_ident(self) -> Self::Output {
                    #node_mod::#node_ident(self)
                }
            }
        }
    };

    // Generate impls
    let mut impls = Vec::new();

    // Marker trait (ManifoldExpr)
    // This allows the type to be used as an operand in other operations
    impls.push(quote! {
        impl #impl_generics #root::ManifoldExpr for #name #ty_generics #where_clause {}
    });

    // Arithmetic
    impls.push(binary_op("Add", "add", "Add", &ops_mod));
    impls.push(binary_op("Sub", "sub", "Sub", &ops_mod));
    impls.push(binary_op("Mul", "mul", "Mul", &ops_mod));
    impls.push(binary_op("Div", "div", "Div", &ops_mod));
    impls.push(unary_op("Neg", "neg", "Neg", &ops_mod));

    // Logic
    // BitAnd -> And
    impls.push(binary_op("BitAnd", "bitand", "And", &ops_mod));
    // BitOr -> Or
    impls.push(binary_op("BitOr", "bitor", "Or", &ops_mod));
    // Not -> BNot
    impls.push(unary_op("Not", "not", "BNot", &ops_mod));

    quote! {
        #(#impls)*
    }
}
