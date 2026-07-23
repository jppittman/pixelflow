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

    // Domain and return must be the default scalar `Field`. Any explicit
    // annotation (`-> Jet3`, `Field -> Discrete`, ...) leaves the JIT's lane.
    is_field_ty(&analyzed.def.domain_ty) && is_field_ty(&analyzed.def.return_ty)
}

/// `None` (the default `Field` domain/return) or an explicit `Field` annotation.
/// Anything else (`Jet2`, `Jet3`, `Discrete`, tuples) is not the JIT's lane.
pub(crate) fn is_field_ty(ty: &Option<syn::Type>) -> bool {
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

/// Whether `kernel!` may transparently route this kernel to the arena backend
/// (the P3 `arena-backend` feature) without changing observable behavior.
///
/// Stricter than [`is_jit_eligible`]: bodies containing derivative
/// projections (`DX`/`DY`/`DZ`/`DXX`/`DXY`/`DYY`) are excluded even though
/// the JIT compiles them, because the two backends *intentionally* disagree
/// about them on a `Field` domain. The combinator backend evaluates the body
/// in the caller's domain — projections yield 0 over `Field` (the fonts'
/// "hard step" degenerate case) and true derivatives only over jets — while
/// the arena backend computes symbolic derivatives unconditionally. Kernels
/// that want the symbolic semantics ask for them via `kernel_jit!`.
///
/// `V` is not excluded: it is the identity in both backends' `Field` lanes.
/// Coordinate-free (constant) bodies are also excluded: they exist to be
/// evaluated in *any* domain (e.g. a scalar-param expression used over
/// `Jet3`), which the combinator's domain polymorphism provides and the
/// monomorphic JIT wrapper cannot — and JIT-compiling a constant buys
/// nothing per pixel anyway.
///
/// Manifold-typed params are excluded from *transparent* routing even though
/// `kernel_jit!` composes them via `HasIr` splicing: a routed builder would
/// require its kernel arguments to implement `HasIr`, which combinator
/// kernels at the call sites do not.
#[allow(dead_code)] // referenced by `kernel!` only under feature = "arena-backend"
pub fn is_transparent_routing_safe(analyzed: &AnalyzedKernel) -> bool {
    is_jit_eligible(analyzed)
        && !analyzed
            .def
            .params
            .iter()
            .any(|p| matches!(p.kind, ParamKind::Manifold))
        && !uses_derivative_projections(&analyzed.def.body)
        && references_coordinates(&analyzed.def.body)
}

/// Whether the body reads any coordinate variable (`X`/`Y`/`Z`/`W`). Sema
/// rejects shadowing of the coordinate intrinsics, so an ident match is exact.
fn references_coordinates(expr: &crate::ast::Expr) -> bool {
    use crate::ast::{Expr, Stmt};
    match expr {
        Expr::Ident(id) => matches!(id.name.to_string().as_str(), "X" | "Y" | "Z" | "W"),
        Expr::Literal(_) | Expr::Verbatim(_) => false,
        Expr::Binary(b) => references_coordinates(&b.lhs) || references_coordinates(&b.rhs),
        Expr::Unary(u) => references_coordinates(&u.operand),
        Expr::MethodCall(m) => {
            references_coordinates(&m.receiver) || m.args.iter().any(references_coordinates)
        }
        Expr::Call(c) => c.args.iter().any(references_coordinates),
        Expr::Block(block) => {
            block.stmts.iter().any(|s| match s {
                Stmt::Let(l) => references_coordinates(&l.init),
                Stmt::Expr(e) => references_coordinates(e),
            }) || block.expr.as_deref().is_some_and(references_coordinates)
        }
        Expr::Tuple(t) => t.elems.iter().any(references_coordinates),
        Expr::Paren(inner) => references_coordinates(inner),
    }
}

/// AST scan for the derivative projections whose semantics are
/// domain-dependent under the combinator backend.
fn uses_derivative_projections(expr: &crate::ast::Expr) -> bool {
    use crate::ast::{Expr, Stmt};
    match expr {
        Expr::Call(call) => {
            matches!(
                call.func.to_string().as_str(),
                "DX" | "DY" | "DZ" | "DXX" | "DXY" | "DYY"
            ) || call.args.iter().any(uses_derivative_projections)
        }
        Expr::Binary(b) => {
            uses_derivative_projections(&b.lhs) || uses_derivative_projections(&b.rhs)
        }
        Expr::Unary(u) => uses_derivative_projections(&u.operand),
        Expr::MethodCall(m) => {
            uses_derivative_projections(&m.receiver)
                || m.args.iter().any(uses_derivative_projections)
        }
        Expr::Block(block) => {
            block.stmts.iter().any(|s| match s {
                Stmt::Let(l) => uses_derivative_projections(&l.init),
                Stmt::Expr(e) => uses_derivative_projections(e),
            }) || block
                .expr
                .as_deref()
                .is_some_and(uses_derivative_projections)
        }
        Expr::Tuple(t) => t.elems.iter().any(uses_derivative_projections),
        Expr::Paren(inner) => uses_derivative_projections(inner),
        Expr::Ident(_) | Expr::Literal(_) | Expr::Verbatim(_) => false,
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
    // Scalar params (dense over scalars, declaration order) become `Param(i)`
    // arena nodes constant-folded at build time; manifold params become
    // reserved slot variables (`Var(8+k)`) substituted with the argument
    // kernels' spliced fragments at build time.
    let scalar_params: Vec<_> = analyzed
        .def
        .params
        .iter()
        .filter(|p| matches!(p.kind, ParamKind::Scalar(_)))
        .collect();
    let manifold_params: Vec<_> = analyzed
        .def
        .params
        .iter()
        .filter(|p| matches!(p.kind, ParamKind::Manifold))
        .collect();
    if manifold_params.len() > ir_bridge::MAX_MANIFOLD_PARAMS {
        return Err(format!(
            "kernel has {} manifold params; the arena backend supports at most {}",
            manifold_params.len(),
            ir_bridge::MAX_MANIFOLD_PARAMS
        ));
    }

    let param_map = ir_bridge::scalar_param_indices(analyzed);
    let manifold_map = ir_bridge::manifold_param_indices(analyzed);
    let (arena_code, plan) =
        ir_bridge::ast_to_runtime_arena(&analyzed.def.body, &param_map, &manifold_map)?;
    let wrapper = jit_wrapper_tokens();

    if scalar_params.is_empty() && manifold_params.is_empty() {
        // Zero-param: compile and wrap. Combinators always hand `kernel!` a
        // closure, so match that under `ZeroParam::Closure`.
        let build = quote! {
            {
                let (__arena, __root) = #arena_code;
                let __jit = ::pixelflow_core::__ir::jit_cache::compile_cached(&__arena, __root)
                    .expect("kernel JIT compilation failed");
                #wrapper
                __JitWrapper {
                    jit: __jit,
                    ir: ::std::sync::Arc::new((__arena, __root)),
                }
            }
        };
        Ok(match conv {
            ZeroParam::Closure => quote! { move || #build },
            ZeroParam::Value => build,
        })
    } else {
        // Builder closure that composes and JITs on call. Args appear in
        // declaration order; scalar params are typed `f32`, manifold params
        // are untyped (closures cannot be generic — the single call site's
        // inference binds each to any `HasIr` kernel).
        let arg_tokens: Vec<TokenStream> = analyzed
            .def
            .params
            .iter()
            .map(|p| {
                let name = &p.name;
                match p.kind {
                    ParamKind::Scalar(_) => quote! { #name: f32 },
                    ParamKind::Manifold => quote! { #name },
                }
            })
            .collect();

        // Composition: splice bare fragments and per-`.at()`-site warped
        // fragments, then substitute every slot (shared logic with named
        // kernel structs — see `ir_bridge::composition_stmts`).
        let manifold_accessors: Vec<TokenStream> = manifold_params
            .iter()
            .map(|p| {
                let name = &p.name;
                quote! { #name }
            })
            .collect();
        let compose = ir_bridge::composition_stmts(&plan, &manifold_accessors);

        // Scalar values, dense in scalar declaration order (matches the
        // `Param(i)` numbering from `scalar_param_indices`).
        let scalar_names: Vec<proc_macro2::Ident> =
            scalar_params.iter().map(|p| p.name.clone()).collect();
        let param_slice = quote! { &[ #( #scalar_names as f32 ),* ] };

        Ok(quote! {
            move | #( #arg_tokens ),* | {
                let (mut __arena, mut __root) = #arena_code;
                #compose
                __root = __arena.substitute_params(__root, #param_slice);
                let __jit = ::pixelflow_core::__ir::jit_cache::compile_cached(&__arena, __root)
                    .expect("kernel JIT compilation failed");
                #wrapper
                __JitWrapper {
                    jit: __jit,
                    ir: ::std::sync::Arc::new((__arena, __root)),
                }
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
        // because `JitManifold` is `Send + Sync`. Alongside the compiled code
        // the wrapper keeps the (composed, pre-lowering) arena it was built
        // from, so the kernel is IR-carrying: `HasIr::splice_into` lets a host
        // kernel absorb it as a fragment (kernel-unification P4).
        #[derive(Clone)]
        struct __JitWrapper {
            jit: ::std::sync::Arc<::pixelflow_core::__ir::JitManifold>,
            ir: ::std::sync::Arc<(
                ::pixelflow_core::__ir::ExprArena,
                ::pixelflow_core::__ir::ExprId,
            )>,
        }
        impl ::pixelflow_core::__ir::HasIr for __JitWrapper {
            fn splice_into(
                &self,
                arena: &mut ::pixelflow_core::__ir::ExprArena,
            ) -> ::pixelflow_core::__ir::ExprId {
                arena.splice(&self.ir.0, self.ir.1)
            }
        }
        // The JIT emits and is called at `pixelflow_ir::JIT_VECTOR_BYTES` width;
        // `eval` transmutes `Field` to that ABI. If this build selected a `Field`
        // whose width the JIT does not emit (e.g. an AVX-512 `Field` while the
        // JIT still emits 128-bit), fail at compile time with a clear message
        // rather than a raw transmute size error or a silent miscompile.
        const _: () = assert!(
            ::core::mem::size_of::<::pixelflow_core::Field>() == ::pixelflow_core::__ir::JIT_VECTOR_BYTES,
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
                        self.jit.call(
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
