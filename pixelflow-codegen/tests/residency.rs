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
/// gives up the register of the value read farthest out, so each term past the
/// file costs at most a handful of memory operations, where giving up the
/// nearest reloads terms the second sum reads at once. The loop nest's own
/// traffic cancels against a wall that fits.
#[test]
fn the_value_read_farthest_out_is_the_one_given_up() {
    /// Terms in the wall that fits every tier's vector file.
    const FITS: u64 = 4;
    /// Terms past the vector file, so that this many must go to memory.
    const EXCESS: u64 = 8;
    /// What each costs at most: a store and a reload in each of the two sums
    /// that read it.
    const PER_TERM: u64 = 4;
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
    };
    let fits = wall(FITS);
    let wide = wall(u64::from(fits.pool) + EXCESS);
    assert!(
        wide.dynamic_memory_ops() <= fits.dynamic_memory_ops() + PER_TERM * EXCESS,
        "{} memory operations for {EXCESS} terms past a pool of {}, against {} for {FITS}",
        wide.dynamic_memory_ops(),
        fits.pool,
        fits.dynamic_memory_ops()
    );
}

/// A constant given up is defined again where it is read: it has no slot, so
/// there is nothing to store and nothing to reload.
///
/// More constants than any vector file holds, summed twice over in one chain:
/// the first pass's reads leave every constant wanted again, so a register must
/// be given up for each one past the file. Few enough constants to stay in
/// registers is the kernel's own traffic, which the many add nothing to but, at
/// most, the chain's result parked for the loops that store it: a store and a
/// load, however many constants there are.
#[test]
fn constants_are_rematerialized_rather_than_spilled() {
    /// Constants that fit every tier's vector file with room to spare.
    const FEW: usize = 4;
    /// Constants past the vector file.
    const EXCESS: usize = 24;
    const PARKED_RESULT: u64 = 2;
    let constants = |count: usize| {
        let values: Vec<f32> = (0..count).map(|i| 1.0 + 0.25 * i as f32).collect();
        let kernel = compiled(|a, _, _| {
            let leaves: Vec<_> = values.iter().map(|&c| a.push_const(c)).collect();
            let once = leaves[1..]
                .iter()
                .fold(leaves[0], |acc, &c| a.push_binary(OpKind::Add, acc, c));
            leaves
                .iter()
                .fold(once, |acc, &c| a.push_binary(OpKind::Add, acc, c))
        });
        let want = 2.0 * values.iter().sum::<f32>();
        let got = eval_point(&kernel.code, 0.0, 0.0);
        assert!(
            (got - want).abs() <= want.abs() * 1e-5,
            "{count} constants: {got} against {want}"
        );
        kernel.traffic
    };
    let few = constants(FEW);
    let many = constants(usize::from(few.pool) + EXCESS);
    assert!(
        many.dynamic_memory_ops() <= few.dynamic_memory_ops() + PARKED_RESULT,
        "{} memory operations for {} constants against {} for {FEW}",
        many.dynamic_memory_ops(),
        usize::from(few.pool) + EXCESS,
        few.dynamic_memory_ops()
    );
}
