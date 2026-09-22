//! Every path a collapse can take through the emitter, executed: a uniform
//! and a gather read through the context, a surviving `Reduce` nested in the
//! lattice's folds, a `Select` whose guard may branch, and a shape narrower
//! than a batch — each at a width with a remainder, each checked against the
//! kernel's own definition computed in scalar `f32`.

#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use pixelflow_codegen::JIT_VECTOR_BYTES;
use pixelflow_codegen::emit::compile;
use pixelflow_ir::arena::{
    BufferDecl, BufferIdentity, ExprArena, ExprId, UniformDecl, UniformIdentity,
};
use pixelflow_ir::fold::{Binder, Fold, Monoid};
use pixelflow_ir::{LatticeShape, OpKind};

const LANES: usize = JIT_VECTOR_BYTES / core::mem::size_of::<f32>();
const WIDTH: usize = 2 * LANES + 3;
const ROWS: usize = 3;
const PITCH: usize = WIDTH + 2;
const ORIGIN: [f32; 2] = [2.0, 5.0];

/// Run `root` over the plane and hand back `(x, y) -> sample`, with the
/// buffers and uniforms the arena declares supplied from `buffers` and
/// `uniforms`.
fn collapse(arena: &ExprArena, root: ExprId, buffers: &[&[f32]], uniforms: &[f32]) -> Vec<f32> {
    let shape = LatticeShape::new([WIDTH as u32, ROWS as u32]);
    let code = compile(arena, root, shape).expect("compile");
    let mut out = vec![f32::NAN; ROWS * PITCH];
    let origin = ORIGIN;
    let mut ctx: Vec<*const f32> = buffers.iter().map(|b| b.as_ptr()).collect();
    ctx.push(uniforms.as_ptr());
    ctx.push(origin.as_ptr());
    unsafe {
        code.code.call(ctx.as_ptr(), out.as_mut_ptr(), PITCH);
    }
    out
}

fn check(out: &[f32], want: impl Fn(f32, f32) -> f32) {
    for row in 0..ROWS {
        for col in 0..WIDTH {
            let (x, y) = (ORIGIN[0] + col as f32, ORIGIN[1] + row as f32);
            let got = out[row * PITCH + col];
            let want = want(x, y);
            assert!(
                (got - want).abs() <= 1e-3 * want.abs().max(1.0),
                "row {row} col {col} (x={x}, y={y}): got {got}, want {want}"
            );
        }
        for col in WIDTH..PITCH {
            assert!(
                out[row * PITCH + col].is_nan(),
                "row {row} col {col} past the width was written"
            );
        }
    }
}

/// `x * u0 + y * u1`: two uniforms, read through the block after the
/// (empty) buffer table.
#[test]
fn uniforms_are_read_from_the_context() {
    let mut a = ExprArena::new();
    let u0 = a.declare_uniform(UniformDecl {
        id: UniformIdentity::mint(),
        default: 0.0,
    });
    let u1 = a.declare_uniform(UniformDecl {
        id: UniformIdentity::mint(),
        default: 0.0,
    });
    let x = a.push_var(0);
    let y = a.push_var(1);
    let u0 = a.push_uniform(u0);
    let u1 = a.push_uniform(u1);
    let xu = a.push_binary(OpKind::Mul, x, u0);
    let yu = a.push_binary(OpKind::Mul, y, u1);
    let root = a.push_binary(OpKind::Add, xu, yu);

    let out = collapse(&a, root, &[], &[3.0, 7.0]);
    check(&out, |x, y| x * 3.0 + y * 7.0);
}

/// `table[x] + y`: a gather whose address is the column, through the first
/// context slot.
#[test]
fn a_gather_reads_the_bound_buffer() {
    let table: Vec<f32> = (0..64).map(|i| (i * i) as f32 * 0.25).collect();
    let mut a = ExprArena::new();
    let buffer = a.declare_buffer(BufferDecl {
        id: BufferIdentity::mint(),
        width: table.len() as u32,
        height: 1,
    });
    let x = a.push_var(0);
    let y = a.push_var(1);
    let buf = a.push_buffer(buffer);
    let gathered = a.push_binary(OpKind::RawGather, buf, x);
    let root = a.push_binary(OpKind::Add, gathered, y);

    let out = collapse(&a, root, &[&table], &[]);
    check(&out, |x, y| table[x as usize] + y);
}

/// `sum_{k in [0, 5)} (x + k) * y`: a surviving fold nested inside the
/// lattice's own, whose body reads both coordinates.
#[test]
fn a_reduce_runs_inside_the_lattices_folds() {
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let binder = Binder::from_slot(0).expect("slot 0");
    let k = a.push_var(binder.var());
    let xk = a.push_binary(OpKind::Add, x, k);
    let body = a.push_binary(OpKind::Mul, xk, y);
    let root = a.push_reduce(Fold::new(Monoid::SUM, binder, 0..5), body);

    let out = collapse(&a, root, &[], &[]);
    check(&out, |x, y| (0..5).map(|k| (x + k as f32) * y).sum());
}

/// `if x < y { x * x } else { y * y }`: a select whose mask varies by lane
/// in some batches and is uniform in others, so both the blend and the
/// guarded branch execute.
#[test]
fn a_select_blends_and_branches() {
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let lt = a.push_binary(OpKind::Lt, x, y);
    let xx = a.push_binary(OpKind::Mul, x, x);
    let yy = a.push_binary(OpKind::Mul, y, y);
    let root = a.push_ternary(OpKind::Select, lt, xx, yy);

    let out = collapse(&a, root, &[], &[]);
    check(&out, |x, y| if x < y { x * x } else { y * y });
}

/// A plane narrower than one batch: the main fold is empty and the whole
/// column is the remainder.
#[test]
fn a_shape_narrower_than_a_batch_is_all_remainder() {
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let root = a.push_binary(OpKind::Sub, x, y);
    let shape = LatticeShape::new([3, 2]);
    let code = compile(&a, root, shape).expect("compile");
    let mut out = vec![f32::NAN; 2 * 4];
    let origin = [1.0f32, 10.0];
    let ctx: [*const f32; 2] = [core::ptr::null(), origin.as_ptr()];
    unsafe {
        code.code.call(ctx.as_ptr(), out.as_mut_ptr(), 4);
    }
    for row in 0..2 {
        for col in 0..3 {
            let want = (1.0 + col as f32) - (10.0 + row as f32);
            assert_eq!(out[row * 4 + col], want, "row {row} col {col}");
        }
        assert!(
            out[row * 4 + 3].is_nan(),
            "row {row}: the pitch gap was written"
        );
    }
}
