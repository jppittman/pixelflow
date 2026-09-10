//! Execute the raw collapse ABI once, the way the benchmarks do.
//!
//! `collapse_overhead` drives `call_collapse` directly — its own context
//! pointer, its own `Point4`, its own tile — and **no CI job runs it**:
//! `cargo nextest` does not execute benchmark targets, and the bench declares
//! `harness = false`, so a `#[test]` inside it would not run either. Clippy
//! compiles it and nothing more.
//!
//! Two bugs shipped green through that hole in a single change: a bench
//! expression left reading the retired Z axis, and a fix for it that read a
//! uniform through the null context the bench passes, which segfaults. Both
//! are execution failures in code that type-checks perfectly.
//!
//! So this is the same ABI usage as a test, at one call of each granularity.
//! It is not a measurement and takes no timings — it exists so that "the
//! benchmark still runs" is something a normal `cargo test` can answer.

#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use pixelflow_codegen::JIT_VECTOR_BYTES;
use pixelflow_codegen::emit::compile;
use pixelflow_codegen::emit::executable::{Point4, TileSlice};
use pixelflow_ir::OpKind;
use pixelflow_ir::expr::{Environment, ExprBuilder, ExprData};
use pixelflow_ir::{Rooted, Term};

const LANES: usize = JIT_VECTOR_BYTES / core::mem::size_of::<f32>();
const GROUPS: usize = 3;
const ROWS: usize = 2;
const BIAS: f32 = 1.75;

/// `X * Y + BIAS` — reads both axes, so every lane and row must differ, and
/// a scaffold that failed to step X or Y would land on the wrong answer
/// rather than merely a slow one.
fn kernel() -> (Rooted<ExprData>, Environment) {
    let mut a = ExprBuilder::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let xy = a.push_binary(OpKind::Mul, x, y);
    let bias = a.push_const(BIAS);
    let root = a.push_binary(OpKind::Add, xy, bias);
    a.finish(&[root])
}

fn lane_x0() -> [f32; LANES] {
    let mut x0 = [0.0f32; LANES];
    for (i, lane) in x0.iter_mut().enumerate() {
        *lane = 0.5 + i as f32;
    }
    x0
}

#[test]
fn one_call_per_group_computes_the_kernel() {
    let built = kernel();
    let code =
        compile(Term::new(built.0.entry(), &built.1)).expect("the smoke kernel must compile");
    let x0 = lane_x0();

    for row in 0..ROWS {
        let y = row as f32 + 0.5;
        for g in 0..GROUPS {
            let mut out = [0.0f32; LANES];
            let mut xs = [0.0f32; LANES];
            for (i, lane) in xs.iter_mut().enumerate() {
                *lane = x0[i] + (g * LANES) as f32;
            }
            // The context is null on purpose: this kernel declares neither a
            // buffer nor an argument, which is the precondition that makes a
            // null context sound. A kernel that declared one would fault
            // here, which is the point.
            unsafe {
                code.code.call_collapse(
                    core::ptr::null(),
                    TileSlice::single(out.as_mut_ptr()),
                    Point4::new(xs, [y; LANES], [0.0; LANES], [0.0; LANES]),
                );
            }
            for lane in 0..LANES {
                let want = xs[lane] * y + BIAS;
                assert!(
                    (out[lane] - want).abs() <= 1e-4,
                    "row {row}, group {g}, lane {lane}: want {want}, got {}",
                    out[lane]
                );
            }
        }
    }
}

#[test]
fn one_call_for_the_whole_frame_computes_the_kernel() {
    let built = kernel();
    let code =
        compile(Term::new(built.0.entry(), &built.1)).expect("the smoke kernel must compile");
    let mut out = vec![0.0f32; GROUPS * LANES * ROWS];

    unsafe {
        code.code.call_collapse(
            core::ptr::null(),
            TileSlice::contiguous(out.as_mut_ptr(), GROUPS, ROWS),
            Point4::new(lane_x0(), [0.5; LANES], [0.0; LANES], [0.0; LANES]),
        );
    }

    for row in 0..ROWS {
        let y = row as f32 + 0.5;
        for g in 0..GROUPS {
            for lane in 0..LANES {
                let x = 0.5 + (g * LANES + lane) as f32;
                let want = x * y + BIAS;
                let got = out[row * GROUPS * LANES + g * LANES + lane];
                assert!(
                    (got - want).abs() <= 1e-4,
                    "row {row}, group {g}, lane {lane}: want {want}, got {got}"
                );
            }
        }
    }
}
