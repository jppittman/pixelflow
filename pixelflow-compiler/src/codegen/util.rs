//! Shared utility functions for code generation.

use proc_macro2::TokenStream;
use quote::quote;

/// Standard imports used in all generated eval functions.
///
/// This avoids duplicating the import list in every code path.
pub fn standard_imports() -> TokenStream {
    quote! {
        use ::pixelflow_core::{
            X, Y, Z, W,
            ManifoldExt, ManifoldCompat, Manifold,
            Let, Var, WithContext, CtxVar, ContextFree, Computed, ManifoldBind,
            A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15,
            GradientMag2D, GradientMag3D,
            Antialias2D, Antialias3D,
            Curvature2D, Normalized2D, Normalized3D,
            V, DX, DY, DZ, DXX, DXY, DYY,
        };
    }
}

/// Sort indexed values by index and extract just the values.
///
/// This is a common pattern: collect (index, value) pairs, sort by index, keep values.
pub fn sort_by_index(indexed: impl IntoIterator<Item = (usize, TokenStream)>) -> Vec<TokenStream> {
    let mut pairs: Vec<_> = indexed.into_iter().collect();
    pairs.sort_by_key(|(idx, _)| *idx);
    pairs.into_iter().map(|(_, val)| val).collect()
}

/// Build a tuple expression from values, handling the single-element case.
///
/// Rust requires a trailing comma for single-element tuples: `(x,)` not `(x)`.
// Only referenced from codegen/binding.rs, which is not currently wired into
// the module tree.
#[allow(dead_code)]
pub fn build_tuple(values: &[TokenStream]) -> TokenStream {
    match values.len() {
        0 => quote! { () },
        1 => {
            let val = &values[0];
            quote! { (#val,) }
        }
        _ => quote! { (#(#values),*) },
    }
}

/// Build an array expression wrapped in a single-element tuple.
///
/// Produces `([val0, val1, ...],)` - the format expected by WithContext
/// for the array-based context system.
pub fn build_array(values: &[TokenStream]) -> TokenStream {
    if values.is_empty() {
        quote! { () }
    } else {
        quote! { ([#(#values),*],) }
    }
}
