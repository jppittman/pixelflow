//! `EmitTraffic` reports what was emitted, on whichever pipeline compiled it.
//!
//! The selection pipeline hands out whole register files and loads every
//! constant it uses, so a zero in `pool` or `remats` there would be a silent
//! one: both are journaled as cost-model features. Run on the selection
//! pipeline (AVX2's own, or `PIXELFLOW_CODEGEN=selection`), these fail when
//! `EmitTraffic::of` reports either as zero.

#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use pixelflow_codegen::emit::{CompileResult, ScopeTraffic, compile};
use pixelflow_ir::{ExprArena, LatticeShape, OpKind};

include!("support/knob.rs");

/// `(c − x) + y` over a small lattice, where `c` is a constant or, for the
/// control, another variable. A constant that is neither zero nor all-ones is
/// a load.
fn compiled(constant: bool) -> CompileResult {
    let mut a = ExprArena::new();
    let [x, y] = [0, 1].map(|v| a.push_var(v));
    let c = if constant { a.push_const(3.5) } else { y };
    let flipped = a.push_binary(OpKind::Sub, c, x);
    let root = a.push_binary(OpKind::Add, flipped, y);
    compile(&a, root, LatticeShape::new([64, 4])).expect("the kernel compiles")
}

#[test]
fn a_kernel_reports_the_registers_its_allocator_hands_out() {
    assert!(
        compiled(true).traffic.pool > 0,
        "no register was handed out"
    );
}

/// Legacy counts the constant once, where it is defined; selection counts it
/// at each read, and not as an instruction.
#[test]
fn a_selected_constant_is_counted_as_brought_in() {
    if !selection() {
        return;
    }
    let sum = |kernel: &CompileResult, count: fn(&ScopeTraffic) -> u64| -> u64 {
        kernel.traffic.scopes.iter().map(count).sum()
    };
    let (with, without) = (compiled(true), compiled(false));
    assert!(
        sum(&with, |s| s.remats) > sum(&without, |s| s.remats),
        "the constant 3.5 was not counted as brought in"
    );
    assert_eq!(
        sum(&with, |s| s.instructions),
        sum(&without, |s| s.instructions),
        "the constant 3.5 was counted as an instruction as well"
    );
}

/// `max(x, k) + y`, where `k` is zero or, for the control, `y`.
fn clamped(zero: bool) -> CompileResult {
    let mut a = ExprArena::new();
    let [x, y] = [0, 1].map(|v| a.push_var(v));
    let k = if zero { a.push_const(0.0) } else { y };
    let floor = a.push_binary(OpKind::Max, x, k);
    let root = a.push_binary(OpKind::Add, floor, y);
    compile(&a, root, LatticeShape::new([64, 4])).expect("the kernel compiles")
}

/// A zero is made in a register without reading memory, so it is not a pool
/// read, whichever pipeline brings it in.
#[test]
fn a_zero_is_not_counted_as_a_pool_read() {
    let remats =
        |kernel: &CompileResult| -> u64 { kernel.traffic.scopes.iter().map(|s| s.remats).sum() };
    assert_eq!(
        remats(&clamped(true)),
        remats(&clamped(false)),
        "the constant 0.0 was counted as a read of the pool"
    );
}
