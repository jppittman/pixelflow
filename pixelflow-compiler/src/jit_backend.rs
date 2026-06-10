//! Shared JIT backend codegen for `kernel!` and `kernel_jit!`.
//!
//! Both macros optimize an [`AnalyzedKernel`] identically (e-graph saturation +
//! NNUE extraction); the only difference is the backend that consumes the
//! optimized AST. This module is the JIT backend: it lowers the optimized body
//! to an [`ExprArena`](pixelflow_ir::arena::ExprArena) and emits code that JIT-
//! compiles the DAG at runtime, wrapping the result in a `Manifold` impl.
//!
//! `kernel_jit!` always routes here. `kernel!` routes here for the subset of
//! kernels the JIT can express today (see [`is_jit_eligible`]) and falls back
//! to the monomorphizing combinator backend (`codegen`) otherwise.

use proc_macro2::TokenStream;
use quote::quote;

use crate::ast::ParamKind;
use crate::ir_bridge;
use crate::sema::AnalyzedKernel;

/// Whether the JIT backend can express this kernel.
///
/// The runtime JIT currently evaluates scalar `Field` kernels only: 4 coordinate
/// lanes in, one `Field` out. It has no representation for autodiff jets, color
/// (`Discrete`), manifold-typed parameters (`.at()` composition), or named ZST
/// kernel structs (which would need a construction API to own JIT'd memory).
///
/// Returns `true` only when every gate is satisfied. A `true` result does not
/// guarantee lowering succeeds — the body may still contain an op the IR bridge
/// rejects — so callers should treat [`emit_jit`] errors as a fall-back signal.
///
/// NOTE: signature eligibility is necessary but not *sufficient* to swap the JIT
/// in for `kernel!` transparently — see the note in `kernel`. It gates the
/// future dual-backend wrapper and is asserted by the parity tests.
#[allow(dead_code)]
pub fn is_jit_eligible(analyzed: &AnalyzedKernel) -> bool {
    // Named structs would need to own runtime-compiled memory; `Name {}` ZST
    // construction can't carry a JitManifold. Defer until there's an API for it.
    if analyzed.def.struct_decl.is_some() {
        return false;
    }

    // Manifold-typed params imply `.at()` composition, which the arena can't
    // splice yet.
    if analyzed
        .def
        .params
        .iter()
        .any(|p| matches!(p.kind, ParamKind::Manifold))
    {
        return false;
    }

    // Domain and return must be the default scalar `Field`. Any explicit
    // annotation (`-> Jet3`, `Field -> Discrete`, ...) leaves the JIT's lane.
    is_field_ty(&analyzed.def.domain_ty) && is_field_ty(&analyzed.def.return_ty)
}

/// `None` (the default `Field` domain/return) or an explicit `Field` annotation.
/// Anything else (`Jet2`, `Jet3`, `Discrete`, tuples) is not the JIT's lane.
fn is_field_ty(ty: &Option<syn::Type>) -> bool {
    match ty {
        None => true,
        // Compare on the type's final path segment so both `Field` and
        // `pixelflow_core::Field` match, while `Jet3`/`Discrete` do not.
        Some(syn::Type::Path(p)) => p
            .path
            .segments
            .last()
            .is_some_and(|seg| seg.ident == "Field"),
        Some(_) => false,
    }
}

/// How the zero-parameter case is surfaced — the two macros disagree.
///
/// `kernel!` emits a closure for *every* anonymous kernel (zero-param results
/// are consumed as `k()`), so the JIT backend must match that to stay drop-in.
/// `kernel_jit!` historically returns the manifold value directly for zero
/// params. N-param kernels are a builder closure under both conventions.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ZeroParam {
    /// Wrap the zero-param result in a `move || { ... }` closure (`kernel!`).
    #[allow(dead_code)]
    Closure,
    /// Return the zero-param manifold value directly (`kernel_jit!`).
    Value,
}

/// Emit JIT-backend code for an (already optimized) kernel.
///
/// On success, returns a token stream with the same surface contract as the
/// combinator backend:
/// - zero scalar params → per `conv`: a `move || -> impl Manifold` closure
///   ([`ZeroParam::Closure`]) or the manifold value directly
///   ([`ZeroParam::Value`]).
/// - N scalar params → a builder closure `move |p0: f32, ...| -> impl Manifold`
///   that JIT-compiles when called (compile-once, e.g. on window resize — not
///   per frame).
///
/// The result implements `Manifold<(Field, Field, Field, Field), Output =
/// Field>` and is `Clone` (combinator kernels are `Copy` ZSTs; a JIT kernel
/// owns executable memory, so the wrapper shares it via `Arc`).
///
/// Returns `Err` if the body contains an operation the IR bridge cannot lower;
/// callers should fall back to the combinator backend in that case.
pub fn emit_jit(analyzed: &AnalyzedKernel, conv: ZeroParam) -> Result<TokenStream, String> {
    // Scalar params, in declaration order, become `Param(i)` arena nodes that
    // are constant-folded at build time.
    let scalar_params: Vec<_> = analyzed
        .def
        .params
        .iter()
        .filter(|p| matches!(p.kind, ParamKind::Scalar(_)))
        .collect();

    let param_map = ir_bridge::scalar_param_indices(analyzed);
    let arena_code = ir_bridge::ast_to_runtime_arena(&analyzed.def.body, &param_map)?;
    let wrapper = jit_wrapper_tokens();

    if scalar_params.is_empty() {
        // Zero-param: compile and wrap. Combinators always hand `kernel!` a
        // closure, so match that under `ZeroParam::Closure`.
        let build = quote! {
            {
                let (__arena, __root) = #arena_code;
                let __code = ::pixelflow_ir::backend::emit::compile_arena_dag(&__arena, __root)
                    .map(|r| r.code)
                    .expect("kernel JIT compilation failed");
                let __jit = ::pixelflow_ir::JitManifold::new(__code);
                #wrapper
                __JitWrapper(::std::sync::Arc::new(__jit))
            }
        };
        Ok(match conv {
            ZeroParam::Closure => quote! { move || #build },
            ZeroParam::Value => build,
        })
    } else {
        // N-param: emit a builder closure that JITs on call.
        let param_names: Vec<proc_macro2::Ident> =
            scalar_params.iter().map(|p| p.name.clone()).collect();
        let param_types: Vec<TokenStream> = scalar_params.iter().map(|_| quote! { f32 }).collect();
        // Params slice in declaration order: first param = index 0.
        let param_slice = quote! { &[ #( #param_names as f32 ),* ] };

        Ok(quote! {
            move | #( #param_names : #param_types ),* | {
                let (mut __arena, __root) = #arena_code;
                let __root = __arena.substitute_params(__root, #param_slice);
                let __code = ::pixelflow_ir::backend::emit::compile_arena_dag(&__arena, __root)
                    .map(|r| r.code)
                    .expect("kernel JIT compilation failed");
                let __jit = ::pixelflow_ir::JitManifold::new(__code);
                #wrapper
                __JitWrapper(::std::sync::Arc::new(__jit))
            }
        })
    }
}

/// The `Manifold` wrapper around a `JitManifold`, emitted into the user's crate
/// (the IR crate can't depend on `pixelflow_core`). Identical in both arms, so
/// it is defined once here and interpolated.
fn jit_wrapper_tokens() -> TokenStream {
    quote! {
        // `Arc` so the wrapper is `Clone` (combinator kernels are `Copy` ZSTs;
        // a JIT kernel owns executable memory). `Send`/`Sync` are auto-derived
        // because `JitManifold` is `Send + Sync`.
        #[derive(Clone)]
        struct __JitWrapper(::std::sync::Arc<::pixelflow_ir::JitManifold>);
        // The JIT emits and is called at `pixelflow_ir::JIT_VECTOR_BYTES` width;
        // `eval` transmutes `Field` to that ABI. If this build selected a `Field`
        // whose width the JIT does not emit (e.g. an AVX-512 `Field` while the
        // JIT still emits 128-bit), fail at compile time with a clear message
        // rather than a raw transmute size error or a silent miscompile.
        const _: () = assert!(
            ::core::mem::size_of::<::pixelflow_core::Field>() == ::pixelflow_ir::JIT_VECTOR_BYTES,
            "kernel_jit!: pixelflow-core Field width does not match the JIT's emitted \
             vector width (pixelflow_ir::JIT_VECTOR_BYTES) — the JIT does not yet emit \
             this SIMD width",
        );
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
    }
}
