#![allow(clippy::all)]
#![allow(warnings)]
#![allow(unused)]
#![allow(improper_ctypes_definitions)]
//! # PixelFlow IR
//!
//! The shared Intermediate Representation (IR) and Backend abstraction.
//!
//! - **Traits**: `Op` trait defines behavior, `EmitStyle` for codegen.
//! - **Ops**: Unit structs (`Add`, `Mul`) implement `Op`.
//! - **ALL_OPS**: The single source of truth for all operations.
//! - **Backend**: SIMD execution traits.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod backend;
pub mod expr;
pub mod kind;
pub mod ops;
pub mod traits;
#[cfg(feature = "alloc")]
pub mod jit_manifold;
#[cfg(feature = "alloc")]
pub use jit_manifold::JitManifold;

// Primary API - Op traits and EmitStyle
pub use traits::{EmitStyle, Op, OpMeta};
pub use ops::{ALL_OPS, OP_COUNT, op_by_name, op_by_index, known_method_names};

// Legacy - OpKind enum (being phased out)
pub use kind::OpKind;

#[cfg(feature = "alloc")]
pub use expr::Expr;

/// Replaces every [`Expr::Param(i)`] node with [`Expr::Const(params[i])`].
///
/// Call this before passing an expression to the JIT emitter. [`Expr::Param`]
/// is an ephemeral node — the emitter panics if it encounters one.
///
/// # Panics
///
/// Panics if any `Param(i)` has `i >= params.len()`. This is always a bug in
/// the calling macro — the number of params in the expression must match the
/// slice provided.
#[cfg(feature = "alloc")]
pub fn substitute_params(expr: &Expr, params: &[f32]) -> Expr {
    match expr {
        Expr::Param(i) => {
            let i = *i as usize;
            assert!(
                i < params.len(),
                "substitute_params: param index {} out of range (have {} params)",
                i,
                params.len()
            );
            Expr::Const(params[i])
        }
        Expr::Var(i) => Expr::Var(*i),
        Expr::Const(v) => Expr::Const(*v),
        Expr::Unary(op, child) => {
            Expr::Unary(*op, alloc::boxed::Box::new(substitute_params(child, params)))
        }
        Expr::Binary(op, left, right) => Expr::Binary(
            *op,
            alloc::boxed::Box::new(substitute_params(left, params)),
            alloc::boxed::Box::new(substitute_params(right, params)),
        ),
        Expr::Ternary(op, a, b, c) => Expr::Ternary(
            *op,
            alloc::boxed::Box::new(substitute_params(a, params)),
            alloc::boxed::Box::new(substitute_params(b, params)),
            alloc::boxed::Box::new(substitute_params(c, params)),
        ),
        Expr::Nary(op, children) => Expr::Nary(
            *op,
            children.iter().map(|c| substitute_params(c, params)).collect(),
        ),
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;

    #[test]
    fn substitute_single_param() {
        let expr = Expr::Param(0);
        let result = substitute_params(&expr, &[3.14_f32]);
        assert!(matches!(result, Expr::Const(v) if (v - 3.14).abs() < 1e-6));
    }

    #[test]
    fn substitute_params_in_binary() {
        let expr = Expr::Binary(
            OpKind::Sub,
            alloc::boxed::Box::new(Expr::Var(0)),
            alloc::boxed::Box::new(Expr::Param(0)),
        );
        let result = substitute_params(&expr, &[1.0_f32]);
        match result {
            Expr::Binary(OpKind::Sub, left, right) => {
                assert!(matches!(*left, Expr::Var(0)));
                assert!(matches!(*right, Expr::Const(v) if (v - 1.0).abs() < 1e-6));
            }
            _ => panic!("expected Binary(Sub, ...)"),
        }
    }

    #[test]
    fn substitute_multiple_params() {
        let expr = Expr::Binary(
            OpKind::Add,
            alloc::boxed::Box::new(Expr::Param(0)),
            alloc::boxed::Box::new(Expr::Param(1)),
        );
        let result = substitute_params(&expr, &[10.0_f32, 32.0_f32]);
        match result {
            Expr::Binary(OpKind::Add, left, right) => {
                assert!(matches!(*left, Expr::Const(v) if (v - 10.0).abs() < 1e-6));
                assert!(matches!(*right, Expr::Const(v) if (v - 32.0).abs() < 1e-6));
            }
            _ => panic!("expected Binary(Add, ...)"),
        }
    }



    // =========================================================================
    // Emitter integration: substitute_params → compile → execute
    // These tests verify the full JIT pipeline: an Expr tree with Param nodes
    // gets substituted, compiled to machine code, and produces correct results.
    // =========================================================================

    /// Compile an Expr (no Param nodes allowed) and run it, returning the
    /// first f32 lane. Params must have been substituted before calling this.
    #[cfg(target_arch = "aarch64")]
    fn jit_eval(expr: &Expr, x: f32, y: f32, z: f32, w: f32) -> f32 {
        use backend::emit::{compile, executable::KernelFn};
        use core::arch::aarch64::*;
        let code = compile(expr).expect("JIT compile failed");
        unsafe {
            let func: KernelFn = code.as_fn();
            let result = func(
                vdupq_n_f32(x),
                vdupq_n_f32(y),
                vdupq_n_f32(z),
                vdupq_n_f32(w),
            );
            vgetq_lane_f32(result, 0)
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn jit_eval(expr: &Expr, x: f32, y: f32, z: f32, w: f32) -> f32 {
        use backend::emit::{compile, executable::KernelFn};
        use core::arch::x86_64::*;
        let code = compile(expr).expect("JIT compile failed");
        unsafe {
            let func: KernelFn = code.as_fn();
            let result = func(
                _mm_set1_ps(x),
                _mm_set1_ps(y),
                _mm_set1_ps(z),
                _mm_set1_ps(w),
            );
            _mm_cvtss_f32(result)
        }
    }

    /// `X + param[0]` — substituting offset=32.0 then evaluating at X=10 gives 42.
    #[test]
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    fn emitter_substitute_then_compile_add_const() {
        let expr = Expr::Binary(
            OpKind::Add,
            alloc::boxed::Box::new(Expr::Var(0)), // X
            alloc::boxed::Box::new(Expr::Param(0)),
        );
        let subst = substitute_params(&expr, &[32.0_f32]);
        let result = jit_eval(&subst, 10.0, 0.0, 0.0, 0.0);
        assert!((result - 42.0).abs() < 1e-5, "expected 42.0, got {result}");
    }

    /// `(X - param[0]) * param[1]` — two params substituted in one pass.
    /// cx=1.0, r=2.0, X=5.0 → (5-1)*2 = 8.0
    #[test]
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    fn emitter_substitute_two_params_then_compile() {
        let expr = Expr::Binary(
            OpKind::Mul,
            alloc::boxed::Box::new(Expr::Binary(
                OpKind::Sub,
                alloc::boxed::Box::new(Expr::Var(0)), // X
                alloc::boxed::Box::new(Expr::Param(0)), // cx
            )),
            alloc::boxed::Box::new(Expr::Param(1)), // r
        );
        let subst = substitute_params(&expr, &[1.0_f32, 2.0_f32]);
        let result = jit_eval(&subst, 5.0, 0.0, 0.0, 0.0);
        assert!((result - 8.0).abs() < 1e-5, "expected 8.0, got {result}");
    }

    /// Same IR template, two different param values → two different kernels,
    /// two different results. Verifies no implicit caching.
    #[test]
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    fn emitter_different_params_produce_different_results() {
        let template = Expr::Binary(
            OpKind::Sub,
            alloc::boxed::Box::new(Expr::Var(0)), // X
            alloc::boxed::Box::new(Expr::Param(0)),
        );

        let subst_a = substitute_params(&template, &[3.0_f32]);
        let subst_b = substitute_params(&template, &[10.0_f32]);

        // X=15: 15-3=12, 15-10=5
        let result_a = jit_eval(&subst_a, 15.0, 0.0, 0.0, 0.0);
        let result_b = jit_eval(&subst_b, 15.0, 0.0, 0.0, 0.0);

        assert!((result_a - 12.0).abs() < 1e-5, "expected 12.0, got {result_a}");
        assert!((result_b - 5.0).abs() < 1e-5, "expected 5.0, got {result_b}");
    }
}
