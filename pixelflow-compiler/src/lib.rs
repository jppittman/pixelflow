//! # PixelFlow Kernel Compiler Frontend
//!
//! A compiler frontend for the PixelFlow DSL, implemented as Rust proc-macros.
//!
//! ## Architecture
//!
//! The live frontend pipeline is:
//!
//! ```text
//! Source (macro input)
//!     │
//!     ▼ Parser (parser.rs)
//! AST (ast.rs)
//!     │
//!     ▼ Semantic Analysis (sema.rs)
//! Analyzed AST + Symbol Table
//!     │
//!     ▼ E-graph optimization (optimize.rs)
//! Optimized AST
//!     │
//!     ▼ Code generation / arena lowering
//! Rust TokenStream (output)
//! ```
//!
//! The compiler still has some older utility modules, but the hot path is
//! parse -> analyze -> optimize -> emit.

mod annotate;
mod ast;
mod codegen;
mod element;
mod ir_bridge;
mod manifold_expr;
mod optimize;
mod parser;
mod sema;
mod symbol;

use proc_macro::TokenStream;
use quote::format_ident;
use syn::parse::{Parse, ParseStream};
use syn::{LitInt, parse_macro_input};

/// Derive macro for the `Element` trait.
///
/// This macro generates the "Applicative" structure for a type, making it behave
/// like a first-class value in the DSL. It automatically implements:
///
/// - `ManifoldExpr` marker trait
/// - Arithmetic operators: `Add`, `Sub`, `Mul`, `Div`, `Neg`
/// - Logic operators: `BitAnd`, `BitOr`, `Not`
///
/// # Usage
///
/// ```ignore
/// #[derive(Element)]
/// pub struct MyCombinator<A, B>(pub A, pub B);
/// ```
#[proc_macro_derive(Element)]
pub fn derive_element(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as syn::DeriveInput);
    element::derive_element(input).into()
}

/// The `kernel!` macro: closure-like syntax for SIMD manifold kernels.
///
/// # Syntax
///
/// ```ignore
/// kernel!(|param1: Type1, param2: Type2, ...| expression)
/// ```
///
/// # Example
///
/// ```ignore
/// use pixelflow_compiler::kernel;
/// use pixelflow_core::{X, Y, Manifold, ManifoldExt};
///
/// let circle = kernel!(|cx: f32, cy: f32, r: f32| {
///     let dx = X - cx;
///     let dy = Y - cy;
///     (dx * dx + dy * dy).sqrt() - r
/// });
///
/// let unit_circle = circle(0.0, 0.0, 1.0);
/// ```
///
/// # Compiler Pipeline
///
/// 1. **Parser**: Builds AST from closure syntax
/// 2. **Semantic Analysis**: Resolves symbols, validates types
/// 3. **Optimization**: E-graph saturation + learned extraction
/// 4. **Code Generation**: Emits struct + Manifold impl
#[proc_macro]
pub fn kernel(input: TokenStream) -> TokenStream {
    // Phase 1: Lex (syn handles this)
    let tokens = proc_macro2::TokenStream::from(input);

    // Phase 2: Parse
    let kernel_ast = match parser::parse(tokens) {
        Ok(ast) => ast,
        Err(e) => return e.to_compile_error().into(),
    };

    // Phase 3: Semantic analysis
    let analyzed = match sema::analyze(kernel_ast) {
        Ok(a) => a,
        Err(e) => return e.to_compile_error().into(),
    };

    // Phase 4: Optimization
    let optimized = optimize::optimize(analyzed);

    // Phase 5: Code generation
    codegen::emit(optimized).into()
}

/// The `kernel_raw!` macro: like `kernel!` but **without optimization**.
///
/// This macro skips the AST optimization phase (constant folding, FMA fusion,
/// algebraic simplification). Use this when you need to benchmark the exact
/// expression form without the compiler transforming it.
///
/// # Use Cases
///
/// - Training data generation: benchmark `X * Y + Z` vs `mul_add(X, Y, Z)` separately
/// - Debugging: see what code is generated without optimization
/// - Testing: verify optimization actually improves things
///
/// # Example
///
/// ```ignore
/// // These will benchmark DIFFERENT code with kernel_raw!
/// let unoptimized = kernel_raw!(|| X * Y + Z);  // Stays as mul + add
/// let explicit_fma = kernel_raw!(|| (X).mul_add(Y, Z));  // Uses FMA
///
/// // With kernel!, both might compile to the same FMA instruction
/// ```
#[proc_macro]
pub fn kernel_raw(input: TokenStream) -> TokenStream {
    // Phase 1: Lex (syn handles this)
    let tokens = proc_macro2::TokenStream::from(input);

    // Phase 2: Parse
    let kernel_ast = match parser::parse(tokens) {
        Ok(ast) => ast,
        Err(e) => return e.to_compile_error().into(),
    };

    // Phase 3: Semantic analysis
    let analyzed = match sema::analyze(kernel_ast) {
        Ok(a) => a,
        Err(e) => return e.to_compile_error().into(),
    };

    // Phase 4: SKIP optimization - go directly to codegen
    // This preserves the exact expression structure for benchmarking

    // Phase 5: Code generation
    codegen::emit(analyzed).into()
}

/// The `kernel_jit!` macro: JIT-compiled kernels that bypass LLVM.
///
/// Has identical semantics to `kernel!`:
/// - With parameters: returns a builder closure that JITs on each call
/// - Without parameters: returns a `JitManifold` directly
///
/// Parameters are constant-folded into the JIT'd kernel. Different parameter
/// values produce different kernels — no cache, caller owns the result.
///
/// # Example
///
/// ```ignore
/// use pixelflow_compiler::kernel_jit;
///
/// // With parameters — builder pattern
/// let builder = kernel_jit!(|cx: f32, r: f32| (X - cx) * r);
/// let manifold = builder(1.0, 2.0);  // JITs immediately
/// manifold.eval((x, y, z, w));
///
/// // Without parameters — direct JitManifold
/// let manifold = kernel_jit!(|| X + Y);
/// manifold.eval((x, y, z, w));
/// ```
#[proc_macro]
pub fn kernel_jit(input: TokenStream) -> TokenStream {
    use quote::quote;

    // Phase 1: Parse
    let tokens = proc_macro2::TokenStream::from(input);
    let kernel_ast = match parser::parse(tokens) {
        Ok(ast) => ast,
        Err(e) => return e.to_compile_error().into(),
    };

    // Phase 2: Semantic analysis
    let analyzed = match sema::analyze(kernel_ast) {
        Ok(a) => a,
        Err(e) => return e.to_compile_error().into(),
    };

    // Phase 3: Optimization (e-graph saturation + NNUE extraction at compile time)
    // Same optimization pipeline as kernel! — the only difference between
    // kernel! and kernel_jit! should be the backend (LLVM vs JIT), not the
    // optimization. This gives us FMA fusion, algebraic simplification,
    // CSE, rsqrt, etc. before the IR is emitted for runtime JIT compilation.
    let analyzed = optimize::optimize(analyzed);

    // Phase 4: Collect scalar params in declaration order
    let scalar_params: Vec<_> = analyzed
        .def
        .params
        .iter()
        .filter(|p| matches!(p.kind, ast::ParamKind::Scalar(_)))
        .collect();

    // Phase 5: Convert optimized body to arena IR (params become Param(i) nodes)
    let param_map = ir_bridge::scalar_param_indices(&analyzed);
    let arena_code = match ir_bridge::ast_to_runtime_arena(&analyzed.def.body, &param_map) {
        Ok(code) => code,
        Err(e) => {
            return syn::Error::new(proc_macro2::Span::call_site(), e)
                .to_compile_error()
                .into();
        }
    };

    if scalar_params.is_empty() {
        // Zero-param: compile immediately, return JitManifold
        let output = quote! {
            {
                let (__arena, __root) = #arena_code;
                let __code = ::pixelflow_ir::backend::emit::compile_arena_dag(&__arena, __root).map(|r| r.code)
                    .expect("kernel_jit! JIT compilation failed");
                let __jit = ::pixelflow_ir::JitManifold::new(__code);
                // Wrap in a struct that implements Manifold in the user's crate
                struct __JitWrapper(::pixelflow_ir::JitManifold);
                impl ::pixelflow_core::Manifold<(
                    ::pixelflow_core::Field,
                    ::pixelflow_core::Field,
                    ::pixelflow_core::Field,
                    ::pixelflow_core::Field,
                )> for __JitWrapper {
                    type Output = ::pixelflow_core::Field;
                    #[inline(always)]
                    fn eval(&self, (x, y, z, w): (
                        ::pixelflow_core::Field,
                        ::pixelflow_core::Field,
                        ::pixelflow_core::Field,
                        ::pixelflow_core::Field,
                    )) -> ::pixelflow_core::Field {
                        unsafe {
                            ::core::mem::transmute(
                                self.0.call(
                                    ::core::mem::transmute(x),
                                    ::core::mem::transmute(y),
                                    ::core::mem::transmute(z),
                                    ::core::mem::transmute(w),
                                )
                            )
                        }
                    }
                }
                unsafe impl Send for __JitWrapper {}
                unsafe impl Sync for __JitWrapper {}
                __JitWrapper(__jit)
            }
        };
        output.into()
    } else {
        // N-param: emit a builder closure
        let param_names: Vec<proc_macro2::Ident> =
            scalar_params.iter().map(|p| p.name.clone()).collect();
        let param_types: Vec<proc_macro2::TokenStream> =
            scalar_params.iter().map(|_| quote! { f32 }).collect();
        // params slice in declaration order: first param = index 0
        let param_slice = quote! { &[ #( #param_names as f32 ),* ] };

        let output = quote! {
            move | #( #param_names : #param_types ),* | {
                let (mut __arena, __root) = #arena_code;
                let __root = __arena.substitute_params(__root, #param_slice);
                let __code = ::pixelflow_ir::backend::emit::compile_arena_dag(&__arena, __root).map(|r| r.code)
                    .expect("kernel_jit! JIT compilation failed");
                let __jit = ::pixelflow_ir::JitManifold::new(__code);
                struct __JitWrapper(::pixelflow_ir::JitManifold);
                impl ::pixelflow_core::Manifold<(
                    ::pixelflow_core::Field,
                    ::pixelflow_core::Field,
                    ::pixelflow_core::Field,
                    ::pixelflow_core::Field,
                )> for __JitWrapper {
                    type Output = ::pixelflow_core::Field;
                    #[inline(always)]
                    fn eval(&self, (x, y, z, w): (
                        ::pixelflow_core::Field,
                        ::pixelflow_core::Field,
                        ::pixelflow_core::Field,
                        ::pixelflow_core::Field,
                    )) -> ::pixelflow_core::Field {
                        unsafe {
                            ::core::mem::transmute(
                                self.0.call(
                                    ::core::mem::transmute(x),
                                    ::core::mem::transmute(y),
                                    ::core::mem::transmute(z),
                                    ::core::mem::transmute(w),
                                )
                            )
                        }
                    }
                }
                unsafe impl Send for __JitWrapper {}
                unsafe impl Sync for __JitWrapper {}
                __JitWrapper(__jit)
            }
        };
        output.into()
    }
}

/// Derive macro for the `ManifoldExpr` marker trait.
///
/// This trait gates access to `ManifoldExt` methods, preventing them from
/// polluting the method namespace of non-manifold types like iterators.
///
/// # Example
///
/// ```ignore
/// use pixelflow_compiler::ManifoldExpr;
///
/// #[derive(ManifoldExpr)]
/// pub struct MyCustomCombinator<M>(pub M);
/// ```
///
/// # Generated Code
///
/// ```ignore
/// impl<M> ::pixelflow_core::ManifoldExpr for MyCustomCombinator<M> {}
/// ```
#[proc_macro_derive(ManifoldExpr)]
pub fn derive_manifold_expr(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as syn::DeriveInput);
    manifold_expr::derive_manifold_expr(input).into()
}

/// Configuration for `generate_peano_types!` macro.
struct PeanoConfig {
    count: usize,
}

impl Parse for PeanoConfig {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let lit: LitInt = input.parse()?;
        let count = lit.base10_parse()?;
        Ok(PeanoConfig { count })
    }
}

/// Generate binary-encoded type aliases N0, N1, ..., N{count-1}.
///
/// Uses binary encoding with UTerm/UInt/B0/B1 for logarithmic depth:
/// - N0 = UTerm
/// - N1 = UInt<UTerm, B1>  (0b1)
/// - N2 = UInt<UInt<UTerm, B1>, B0>  (0b10)
/// - N3 = UInt<UInt<UTerm, B1>, B1>  (0b11)
/// - etc.
///
/// This reduces type nesting from O(n) to O(log n).
///
/// # Example
///
/// ```ignore
/// generate_binary_types!(256);
/// // N30 = UInt<UInt<UInt<UInt<UInt<UTerm, B1>, B1>, B1>, B1>, B0>  (0b11110)
/// // Instead of Succ<Succ<Succ<...30 times...>>>
/// ```
#[proc_macro]
pub fn generate_binary_types(input: TokenStream) -> TokenStream {
    let config = parse_macro_input!(input as PeanoConfig);
    let count = config.count;

    let mut types = Vec::new();

    for i in 0..count {
        let name = format_ident!("N{}", i);
        let doc = format!("Index {} (0b{:b})", i, i);
        let ty = to_binary_type(i);

        types.push(quote::quote! {
            #[doc = #doc]
            pub type #name = #ty;
        });
    }

    TokenStream::from(quote::quote! {
        #(#types)*
    })
}

/// Convert a number to its binary type representation.
fn to_binary_type(n: usize) -> proc_macro2::TokenStream {
    if n == 0 {
        return quote::quote! { UTerm };
    }

    let bit = if n % 2 == 0 {
        quote::quote! { B0 }
    } else {
        quote::quote! { B1 }
    };

    let rest = to_binary_type(n >> 1);

    quote::quote! { UInt<#rest, #bit> }
}
