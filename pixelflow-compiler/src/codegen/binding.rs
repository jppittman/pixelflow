//! Binding strategies for parameter and literal values.

use proc_macro2::TokenStream;
use quote::quote;

use super::util::build_tuple;

/// Strategy for binding parameter and literal values to the expression.
///
/// ## Categorical Perspective
///
/// Binding extends the domain of a manifold with captured values.
/// This is the "contramap" half of the profunctor: we're modifying
/// how coordinates are supplied, not what values are produced.
///
/// The three strategies represent different ways to encode this extension:
/// - **FlatTuple**: Single WithContext with a flat tuple (efficient for many params)
/// - **NestedLet**: Nested Let bindings (original approach, exponential trait solving)
/// - **Mixed**: Combine both for params + literals
#[derive(Debug)]
pub enum BindingStrategy {
    /// No bindings needed - evaluate expression directly.
    None,

    /// Flat tuple binding: `WithContext::new((v0, v1, ...), expr)`
    ///
    /// Most efficient for multiple bindings. Avoids trait solver explosion.
    FlatTuple {
        /// Values in tuple order (already sorted by index).
        values: Vec<TokenStream>,
    },

    /// Nested Let binding: `Let::new(v0, Let::new(v1, expr))`
    ///
    /// Legacy approach. Still used for edge cases.
    NestedLet {
        /// Values in binding order (outermost to innermost).
        values: Vec<TokenStream>,
    },

    /// Mixed strategy: FlatTuple for params, NestedLet for literals.
    ///
    /// Used in Jet mode where literals need Let wrapping but params use WithContext.
    Mixed {
        /// Params for WithContext tuple.
        param_tuple: Vec<TokenStream>,
        /// Literals for nested Let.
        literal_lets: Vec<TokenStream>,
    },
}

impl BindingStrategy {
    /// Emit the binding wrapper around an expression.
    ///
    /// Returns token stream that evaluates to the bound expression's result.
    pub fn emit(self, expr: TokenStream) -> TokenStream {
        match self {
            BindingStrategy::None => {
                quote! { #expr.eval(__p) }
            }

            BindingStrategy::FlatTuple { values } => {
                let tuple = build_tuple(&values);
                quote! { WithContext::new(#tuple, #expr).eval(__p) }
            }

            BindingStrategy::NestedLet { values } => {
                // Fold right: innermost binding is last value
                values.into_iter().rev().fold(
                    quote! { #expr.eval(__p) },
                    |inner, val| quote! { Let::new(#val, #inner).eval(__p) },
                )
            }

            BindingStrategy::Mixed { param_tuple, literal_lets } => {
                let tuple = build_tuple(&param_tuple);
                let with_ctx = quote! { WithContext::new(#tuple, #expr) };

                // Wrap with Let bindings for literals
                literal_lets.into_iter().rev().fold(
                    quote! { #with_ctx.eval(__p) },
                    |inner, val| quote! { Let::new(#val, #inner).eval(__p) },
                )
            }
        }
    }
}
