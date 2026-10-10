//! `EmitTraffic` reports what was emitted, on whichever pipeline compiled it.
//!
//! The selection pipeline hands out whole register files and loads every
//! constant it uses, so a zero in `pool` or `remats` there would be a silent
//! one: both are journaled as cost-model features. Run under
//! `PIXELFLOW_CODEGEN=selection`, these fail when `EmitTraffic::of` reports
//! either as zero.

#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use pixelflow_codegen::emit::{CompileResult, compile};
use pixelflow_ir::{ExprArena, LatticeShape, OpKind};

/// `(3.5 − x) + y` over a small lattice: the constant is neither zero nor
/// all-ones, so it is a load.
fn compiled() -> CompileResult {
    let mut a = ExprArena::new();
    let [x, y] = [0, 1].map(|v| a.push_var(v));
    let c = a.push_const(3.5);
    let flipped = a.push_binary(OpKind::Sub, c, x);
    let root = a.push_binary(OpKind::Add, flipped, y);
    compile(&a, root, LatticeShape::new([64, 4])).expect("the kernel compiles")
}

#[test]
fn a_kernel_reports_the_registers_its_allocator_hands_out() {
    assert!(compiled().traffic.pool > 0, "no register was handed out");
}

/// The legacy pipeline parks a constant in a register for the whole kernel, so
/// it reports no remat for this one; the knob goes with the legacy pipeline.
#[test]
fn a_selected_constant_is_counted_as_brought_in() {
    let selection = std::env::var("PIXELFLOW_CODEGEN")
        .is_ok_and(|name| name.trim().eq_ignore_ascii_case("selection"));
    if !selection {
        return;
    }
    let remats: u64 = compiled().traffic.scopes.iter().map(|s| s.remats).sum();
    assert!(remats > 0, "the constant 3.5 was not counted as brought in");
}
