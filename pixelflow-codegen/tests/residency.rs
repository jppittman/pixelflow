//! A value stays in its register until something needs the register.
//!
//! Run under `PIXELFLOW_CODEGEN=selection`, these fail on an allocator that
//! stores every value to the frame when it is defined and reloads it at every
//! read. They are written against what a kernel emits, not against the
//! allocator: its traffic, its frame, and the numbers it computes.

#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use pixelflow_codegen::emit::{CompileResult, CompiledKernel, compile};
use pixelflow_ir::{ExprArena, ExprId, LatticeShape, OpKind};

/// `build`'s kernel over the lattice coordinates `x` and `y`, compiled at one
/// point.
fn compiled(build: impl FnOnce(&mut ExprArena, ExprId, ExprId) -> ExprId) -> CompileResult {
    let mut a = ExprArena::new();
    let [x, y] = [0, 1].map(|v| a.push_var(v));
    let root = build(&mut a, x, y);
    compile(&a, root, LatticeShape::POINT).expect("the kernel compiles")
}

/// The one sample of a kernel compiled at one point, at `(x, y)`.
fn eval_point(kernel: &CompiledKernel, x: f32, y: f32) -> f32 {
    let mut out = [0.0f32; 1];
    let origin = [x, y];
    let no_block: [f32; 0] = [];
    // SAFETY: `ctx[0]` is the uniform block, which these kernels declare
    // nothing for, `ctx[1]` is the origin block, and `out` holds the one
    // sample a single-point lattice writes.
    let ctx: [*const f32; 2] = [no_block.as_ptr(), origin.as_ptr()];
    unsafe {
        kernel.call(ctx.as_ptr(), out.as_mut_ptr(), 1);
    }
    out[0]
}

/// Arithmetic on the coordinates alone is defined and consumed inside the
/// innermost loop's body, so none of it lives across a loop head: adding to it
/// adds no store, no reload and no slot. An allocator that homes every value
/// in the frame pays for each.
#[test]
fn values_that_live_within_a_loop_body_cost_no_memory() {
    let small = compiled(|a, x, y| a.push_binary(OpKind::Add, x, y));
    let large = compiled(|a, x, y| {
        let sum = a.push_binary(OpKind::Add, x, y);
        let difference = a.push_binary(OpKind::Sub, x, y);
        let product = a.push_binary(OpKind::Mul, sum, difference);
        a.push_binary(OpKind::Add, product, sum)
    });
    assert_eq!(
        large.traffic.dynamic_memory_ops(),
        small.traffic.dynamic_memory_ops(),
        "a longer chain of values that die inside the loop body moved memory"
    );
    assert_eq!(large.spill_count, small.spill_count);
    assert_eq!(large.frame_bytes, small.frame_bytes);
}

/// `MulAdd` accumulates into its addend. When the addend is read again after
/// the `MulAdd`, the allocator copies it first, so the copy is what is
/// overwritten.
#[test]
fn an_operand_overwritten_in_place_survives_when_it_is_read_again() {
    let kernel = compiled(|a, x, y| {
        let fused = a.push_ternary(OpKind::MulAdd, x, y, x);
        a.push_binary(OpKind::Add, fused, x)
    });
    let (x, y) = (3.0f32, 5.0f32);
    assert_eq!(
        eval_point(&kernel.code, x, y).to_bits(),
        (x.mul_add(y, x) + x).to_bits()
    );
}

/// More values live at once than the vector file has registers. The allocator
/// gives up the register of the value read farthest out, so no term past the
/// file costs more than a store where it is defined and a reload where it is
/// read again; giving up the nearest instead reloads terms the second sum
/// reads at once. The loop nest's own traffic cancels against a wall that fits.
#[test]
fn the_value_read_farthest_out_is_the_one_given_up() {
    /// Terms in the wall that fits every tier's vector file.
    const FITS: u64 = 4;
    /// Terms in the wall past the vector file of every tier but AVX-512's.
    const TERMS: u64 = 24;
    let wall = |terms: u64| {
        compiled(|a, x, y| {
            let mut sums = Vec::new();
            let mut term = a.push_binary(OpKind::Mul, x, y);
            for _ in 0..terms {
                term = a.push_ternary(OpKind::MulAdd, term, x, y);
                sums.push(term);
            }
            // Summed in opposite orders: the same order would be the same nodes.
            let sum = |a: &mut ExprArena, terms: &[ExprId]| {
                let first = terms[0];
                terms[1..]
                    .iter()
                    .fold(first, |acc, &t| a.push_binary(OpKind::Add, acc, t))
            };
            let up = sum(a, &sums);
            sums.reverse();
            let down = sum(a, &sums);
            a.push_binary(OpKind::Add, up, down)
        })
        .traffic
        .dynamic_memory_ops()
    };
    let (fits, wide) = (wall(FITS), wall(TERMS));
    assert!(
        wide - fits <= 2 * (TERMS - FITS),
        "{wide} memory operations for {TERMS} terms against {fits} for {FITS}"
    );
}
