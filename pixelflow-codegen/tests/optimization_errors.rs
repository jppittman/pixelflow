//! Optimization is part of compilation, so refusing it is observable as a
//! compile error rather than as a different, unoptimized program.

#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use pixelflow_codegen::CompileError;
use pixelflow_codegen::jit_cache;
use pixelflow_ir::{ExprArena, Kernel, KernelStore, LatticeShape, OpKind};
use pixelflow_search::egraph::Declined;
use pixelflow_search::runtime::OptimizationError;

#[test]
fn an_optimizer_admission_failure_is_a_compile_error() {
    let mut arena = ExprArena::new();
    let root = arena.push_param(7);
    let kernel = Kernel::from_parts(arena, root);

    let error = jit_cache::compile(&kernel, LatticeShape::POINT)
        .err()
        .expect("an unbound parameter is not an optimizable program");

    assert_eq!(
        error,
        CompileError::Optimization(OptimizationError::Admission {
            unit: None,
            reason: Declined::Param(7),
        })
    );
}

#[test]
fn a_refused_unit_is_named_by_the_compile_error() {
    let mut body = ExprArena::new();
    let value = body.push_const(3.0);
    let body_root = body.push_nary(OpKind::Tuple, &[value]);
    let key = KernelStore::intern(&Kernel::from_parts(body, body_root));

    let mut arena = ExprArena::new();
    let unit = arena.push_ref(key);
    let x = arena.push_var(0);
    let root = arena.push_binary(OpKind::Add, unit, x);
    let kernel = Kernel::from_parts(arena, root);

    let error = jit_cache::compile(&kernel, LatticeShape::POINT)
        .err()
        .expect("the unit's unbound parameter must be reported");

    assert_eq!(
        error,
        CompileError::Optimization(OptimizationError::Admission {
            unit: Some(key),
            reason: Declined::Op(OpKind::Tuple),
        })
    );
}
