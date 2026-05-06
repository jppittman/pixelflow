//! Builder pattern for generating struct definitions with Manifold impls.

use proc_macro2::TokenStream;
use quote::quote;

/// Builder for generating struct definitions with Manifold impls.
///
/// ## Purpose
///
/// Consolidates the 8+ code paths in `emit_named_kernel` into a single
/// builder that handles all configuration combinations:
/// - With/without generic parameters (manifold params)
/// - With/without Copy derive (single scalar vs multiple)
/// - Fixed domain type vs generic domain P
///
/// ## Usage
///
/// ```ignore
/// StructEmitter::new(visibility, name)
///     .with_generics(generic_names)
///     .with_derives(Derives::CloneCopy)
///     .with_fields(fields)
///     .with_fixed_domain(domain_type, scalar_type, output_type)
///     .with_eval_body(body, imports)
///     .build()
/// ```
pub struct StructEmitter {
    visibility: syn::Visibility,
    name: syn::Ident,
    generic_names: Vec<syn::Ident>,
    fields: Vec<TokenStream>,
    field_names: Vec<syn::Ident>,
    constructor_params: Vec<TokenStream>,
    derives: Derives,
    domain_config: DomainConfig,
    eval_body: Option<EvalBody>,
}

/// What traits to derive on the struct.
#[derive(Clone, Copy)]
pub enum Derives {
    /// Clone only (default for multi-field or manifold params).
    Clone,
    /// Clone + Copy (for unit struct or single scalar param).
    CloneCopy,
}

/// Domain type configuration.
pub enum DomainConfig {
    /// Fixed domain: `impl Manifold<__Domain> for Struct`
    Fixed {
        domain_type: TokenStream,
        output_type: TokenStream,
        trait_bounds: Vec<TokenStream>,
    },
    /// Generic domain: `impl<__P: Spatial> Manifold<__P> for Struct`
    Generic {
        output_type: TokenStream,
    },
}

/// The eval function body.
pub struct EvalBody {
    pub imports: TokenStream,
    pub peano_imports: TokenStream,
    pub pre_eval_stmts: TokenStream,
    pub expr: TokenStream,
    pub binding: TokenStream,
}

impl StructEmitter {
    pub fn new(visibility: syn::Visibility, name: syn::Ident) -> Self {
        Self {
            visibility,
            name,
            generic_names: Vec::new(),
            fields: Vec::new(),
            field_names: Vec::new(),
            constructor_params: Vec::new(),
            derives: Derives::Clone,
            domain_config: DomainConfig::Generic {
                output_type: quote! { ::pixelflow_core::Field },
            },
            eval_body: None,
        }
    }

    pub fn with_generics(mut self, names: Vec<syn::Ident>) -> Self {
        self.generic_names = names;
        self
    }

    pub fn with_derives(mut self, derives: Derives) -> Self {
        self.derives = derives;
        self
    }

    pub fn with_fields(
        mut self,
        fields: Vec<TokenStream>,
        field_names: Vec<syn::Ident>,
        constructor_params: Vec<TokenStream>,
    ) -> Self {
        self.fields = fields;
        self.field_names = field_names;
        self.constructor_params = constructor_params;
        self
    }

    pub fn with_fixed_domain(
        mut self,
        domain_type: TokenStream,
        output_type: TokenStream,
        trait_bounds: Vec<TokenStream>,
    ) -> Self {
        self.domain_config = DomainConfig::Fixed {
            domain_type,
            output_type,
            trait_bounds,
        };
        self
    }

    pub fn with_eval_body(
        mut self,
        imports: TokenStream,
        peano_imports: TokenStream,
        pre_eval_stmts: TokenStream,
        expr: TokenStream,
        binding: TokenStream,
    ) -> Self {
        self.eval_body = Some(EvalBody {
            imports,
            peano_imports,
            pre_eval_stmts,
            expr,
            binding,
        });
        self
    }

    pub fn build(self) -> TokenStream {
        let vis = &self.visibility;
        let name = &self.name;
        let generics = &self.generic_names;
        let fields = &self.fields;
        let field_names = &self.field_names;
        let ctor_params = &self.constructor_params;

        // Build struct definition
        // All structs derive ManifoldExpr for composability
        let struct_def = if self.fields.is_empty() {
            // Unit struct
            quote! { #[derive(Clone, Copy, ::pixelflow_compiler::ManifoldExpr)] #vis struct #name; }
        } else if generics.is_empty() {
            // Non-generic struct
            match self.derives {
                Derives::CloneCopy => quote! {
                    #[derive(Clone, Copy, ::pixelflow_compiler::ManifoldExpr)]
                    #vis struct #name { #(#fields),* }
                },
                Derives::Clone => quote! {
                    #[derive(Clone, ::pixelflow_compiler::ManifoldExpr)]
                    #vis struct #name { #(#fields),* }
                },
            }
        } else {
            // Generic struct - manual Clone/Copy impls, derive ManifoldExpr
            quote! {
                #[derive(::pixelflow_compiler::ManifoldExpr)]
                #vis struct #name<#(#generics),*> { #(#fields),* }

                impl<#(#generics: Clone),*> Clone for #name<#(#generics),*> {
                    fn clone(&self) -> Self {
                        Self { #(#field_names: self.#field_names.clone()),* }
                    }
                }

                impl<#(#generics: Copy),*> Copy for #name<#(#generics),*> {}
            }
        };

        // Build constructor
        let constructor = if self.fields.is_empty() {
            quote! {
                impl #name {
                    pub fn new() -> Self { Self }
                }
                impl Default for #name {
                    fn default() -> Self { Self::new() }
                }
            }
        } else if generics.is_empty() {
            quote! {
                impl #name {
                    pub fn new(#(#ctor_params),*) -> Self {
                        Self { #(#field_names),* }
                    }
                }
            }
        } else {
            quote! {
                impl<#(#generics),*> #name<#(#generics),*> {
                    pub fn new(#(#ctor_params),*) -> Self {
                        Self { #(#field_names),* }
                    }
                }
            }
        };

        // Build Manifold impl
        let eval_body = self.eval_body.expect("eval_body required");
        let imports = &eval_body.imports;
        let peano_imports = &eval_body.peano_imports;
        let pre_eval = &eval_body.pre_eval_stmts;
        let expr = &eval_body.expr;
        let binding = &eval_body.binding;

        let manifold_impl = match &self.domain_config {
            DomainConfig::Fixed { domain_type, output_type, trait_bounds } => {
                if generics.is_empty() {
                    quote! {
                        impl ::pixelflow_core::Manifold<#domain_type> for #name {
                            type Output = #output_type;

                            #[inline(always)]
                            fn eval(&self, __p: #domain_type) -> #output_type {
                                #imports
                                #peano_imports
                                #pre_eval
                                let __expr = { #expr };
                                #binding
                            }
                        }
                    }
                } else {
                    quote! {
                        impl<#(#generics),*> ::pixelflow_core::Manifold<#domain_type> for #name<#(#generics),*>
                        where
                            #(#trait_bounds),*
                        {
                            type Output = #output_type;

                            #[inline(always)]
                            fn eval(&self, __p: #domain_type) -> #output_type {
                                #imports
                                #peano_imports
                                #pre_eval
                                let __expr = { #expr };
                                #binding
                            }
                        }
                    }
                }
            }

            DomainConfig::Generic { output_type } => {
                if generics.is_empty() {
                    quote! {
                        impl<__P> ::pixelflow_core::Manifold<__P> for #name
                        where
                            __P: Copy + Send + Sync + ::pixelflow_core::Spatial,
                        {
                            type Output = #output_type;

                            #[inline(always)]
                            fn eval(&self, __p: __P) -> #output_type {
                                #imports
                                #peano_imports
                                #pre_eval
                                let __expr = { #expr };
                                #binding
                            }
                        }
                    }
                } else {
                    quote! {
                        impl<#(#generics),*, __P> ::pixelflow_core::Manifold<__P> for #name<#(#generics),*>
                        where
                            __P: Copy + Send + Sync + ::pixelflow_core::Spatial,
                            #(#generics: ::pixelflow_core::Manifold<__P, Output = #output_type>),*,
                        {
                            type Output = #output_type;

                            #[inline(always)]
                            fn eval(&self, __p: __P) -> #output_type {
                                #imports
                                #peano_imports
                                #pre_eval
                                let __expr = { #expr };
                                #binding
                            }
                        }
                    }
                }
            }
        };

        quote! {
            #struct_def
            #constructor
            #manifold_impl
        }
    }
}
