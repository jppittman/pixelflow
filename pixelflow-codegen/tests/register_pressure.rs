//! A kernel with more of everything live than any register file holds, run.
//!
//! Every class of register gives some up: the vectors (a wall of products, each
//! read by two sums in opposite orders), and the general registers (one bound
//! buffer per table, each an address read inside the column loop, beside the
//! counters and the ABI's three). A guard's mask is a general register too, so
//! the `If` at the end branches on whichever one is left. A value is wrong the
//! moment any of them is addressed through the wrong register, so the check is
//! the kernel's values, against the same arithmetic in `f64`.
//!
//! The predicates are the opmask file's turn (AVX-512 holds a comparison in a
//! `k` register, seven of them): a wall of masks combined by `&` and `|`, each
//! read by two sums in opposite orders, then selected by.
//!
//! The sizes outnumber the largest file of each class on any tier, so one
//! kernel of each is the pressure kernel of AVX2, AVX-512 and NEON; the host's
//! tier is the one that runs it.

#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use pixelflow_codegen::emit::{CompileResult, compile};
use pixelflow_codegen::jit_vector_bytes;
use pixelflow_ir::arena::{BufferDecl, BufferIdentity, ExprArena, ExprId};
use pixelflow_ir::{LatticeShape, OpKind};

/// Bound buffers, each an address live across the column loop: more than the
/// general file of any tier.
const TABLES: usize = 24;
/// Products live at once: more than the vector file of any tier.
const TERMS: usize = 48;
/// Predicates live at once: more than the opmask file, and than the vector
/// file of any tier that holds a mask in one.
const PREDICATES: usize = 40;
const TABLE_LEN: usize = 64;
const ROWS: usize = 3;
const ORIGIN: [f32; 2] = [2.0, 5.0];

fn lanes() -> usize {
    jit_vector_bytes() / core::mem::size_of::<f32>()
}

fn width() -> usize {
    2 * lanes() + 3
}

/// Where the guarded `If` turns: halfway through the second batch, on every
/// tier, so one batch is wholly before it and one is across it.
fn edge() -> f32 {
    ORIGIN[0] + (lanes() + lanes() / 2) as f32
}

fn pitch() -> usize {
    width() + 2
}

fn table(t: usize) -> Vec<f32> {
    (0..TABLE_LEN)
        .map(|i| (i + 1) as f32 * 0.5 + t as f32 * 0.25)
        .collect()
}

fn offset(k: usize) -> f32 {
    0.1 * k as f32
}

/// `if x < edge() { sin(x)·y + up } else { exp(y/8) + down }`, where `up` and
/// `down` are `Σ_k table_{k mod T}[x] · (y + c_k)` summed in opposite orders.
fn kernel() -> (ExprArena, ExprId, Vec<Vec<f32>>) {
    let mut a = ExprArena::new();
    let (x, y) = (a.push_var(0), a.push_var(1));
    let tables: Vec<Vec<f32>> = (0..TABLES).map(table).collect();
    let gathered: Vec<ExprId> = tables
        .iter()
        .map(|t| {
            let buffer = a.declare_buffer(BufferDecl {
                id: BufferIdentity::mint(),
                width: t.len() as u32,
                height: 1,
            });
            let base = a.push_buffer(buffer);
            a.push_binary(OpKind::RawGather, base, x)
        })
        .collect();
    let mut products: Vec<ExprId> = (0..TERMS)
        .map(|k| {
            let c = a.push_const(offset(k));
            let shifted = a.push_binary(OpKind::Add, y, c);
            a.push_binary(OpKind::Mul, gathered[k % TABLES], shifted)
        })
        .collect();
    let sum = |a: &mut ExprArena, terms: &[ExprId]| {
        terms[1..]
            .iter()
            .fold(terms[0], |acc, &t| a.push_binary(OpKind::Add, acc, t))
    };
    let up = sum(&mut a, &products);
    products.reverse();
    let down = sum(&mut a, &products);

    let edge = a.push_const(edge());
    let before = a.push_binary(OpKind::Lt, x, edge);
    let wave = a.push_unary(OpKind::Sin, x);
    let waved = a.push_binary(OpKind::Mul, wave, y);
    let hot = a.push_binary(OpKind::Add, waved, up);
    let eighth = a.push_const(0.125);
    let scaled = a.push_binary(OpKind::Mul, y, eighth);
    let grown = a.push_unary(OpKind::Exp, scaled);
    let cold = a.push_binary(OpKind::Add, grown, down);
    let root = a.push_ternary(OpKind::If, before, hot, cold);
    (a, root, tables)
}

fn reference(tables: &[Vec<f32>], x: f32, y: f32) -> f64 {
    let (x64, y64) = (f64::from(x), f64::from(y));
    let sum: f64 = (0..TERMS)
        .map(|k| {
            let entry = f64::from(tables[k % TABLES][x as usize]);
            entry * (y64 + f64::from(offset(k)))
        })
        .sum();
    match x < edge() {
        true => x64.sin() * y64 + sum,
        false => (y64 * 0.125).exp() + sum,
    }
}

/// What the compile reports of the pressure: values outnumber the files, so
/// something is spilled and read back from the frame. A kernel the optimizer
/// shrinks below that checks nothing about allocation.
fn assert_pressured(result: &CompileResult) {
    let frame_reads: u64 = result.traffic.scopes.iter().map(|s| s.loads).sum();
    assert!(
        result.spill_count > 0 && frame_reads > 0,
        "the kernel no longer outnumbers the register files: {} slots, {frame_reads} frame reads",
        result.spill_count
    );
}

/// `result` called over `ROWS` rows of `width()` samples, `tables` bound.
fn run(result: &CompileResult, tables: &[Vec<f32>]) -> Vec<f32> {
    let mut out = vec![f32::NAN; ROWS * pitch()];
    let origin = ORIGIN;
    let uniforms: [f32; 0] = [];
    let mut ctx: Vec<*const f32> = tables.iter().map(|t| t.as_ptr()).collect();
    ctx.push(uniforms.as_ptr());
    ctx.push(origin.as_ptr());
    // SAFETY: `ctx` holds one base per buffer the arena declares, in
    // declaration order, then the (empty) uniform block and the origin;
    // `out` holds `ROWS` rows of `pitch()` samples, which the call fills.
    unsafe {
        result.code.call(ctx.as_ptr(), out.as_mut_ptr(), pitch());
    }
    out
}

/// Every sample of `out` against `reference(x, y)`.
fn assert_values(out: &[f32], reference: impl Fn(f32, f32) -> f64) {
    for row in 0..ROWS {
        for col in 0..width() {
            let (x, y) = (ORIGIN[0] + col as f32, ORIGIN[1] + row as f32);
            let (got, want) = (f64::from(out[row * pitch() + col]), reference(x, y));
            assert!(
                (got - want).abs() <= 1e-3 * want.abs().max(1.0),
                "row {row} col {col} (x={x}, y={y}): got {got}, want {want}"
            );
        }
    }
}

#[test]
fn a_kernel_wider_than_every_register_file_computes_its_values() {
    let (arena, root, tables) = kernel();
    let shape = LatticeShape::new([width() as u32, ROWS as u32]);
    let result = compile(&arena, root, shape).expect("the kernel compiles");
    assert_pressured(&result);
    let out = run(&result, &tables);
    assert_values(&out, |x, y| reference(&tables, x, y));
}

fn threshold(k: usize) -> f32 {
    ORIGIN[0] + 0.5 + (k * width() / PREDICATES) as f32
}

fn is_below(x: f32, k: usize) -> bool {
    x < threshold(k % PREDICATES)
}

fn joint(x: f32, k: usize) -> bool {
    (is_below(x, k) && is_below(x, k + 1)) || is_below(x, k + 2)
}

/// `Σ_k up_k + Σ_{k reversed} down_k`, where `up_k` and `down_k` each select
/// by `joint_k = (x < c_k & x < c_{k+1}) | x < c_{k+2}`, so every joint mask is
/// live from the first sum to the second.
fn predicate_kernel() -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let (x, y) = (a.push_var(0), a.push_var(1));
    let below: Vec<ExprId> = (0..PREDICATES)
        .map(|k| {
            let c = a.push_const(threshold(k));
            a.push_binary(OpKind::Lt, x, c)
        })
        .collect();
    let joints: Vec<ExprId> = (0..PREDICATES)
        .map(|k| {
            let both = a.push_binary(OpKind::BitAnd, below[k], below[(k + 1) % PREDICATES]);
            a.push_binary(OpKind::BitOr, both, below[(k + 2) % PREDICATES])
        })
        .collect();
    let mut up: Vec<ExprId> = Vec::new();
    let mut down: Vec<ExprId> = Vec::new();
    for (k, &mask) in joints.iter().enumerate() {
        let c = a.push_const(k as f32);
        let (raised, lowered) = (
            a.push_binary(OpKind::Add, y, c),
            a.push_binary(OpKind::Sub, y, c),
        );
        let (half, quarter) = (a.push_const(0.5), a.push_const(0.25));
        let (halved, quartered) = (
            a.push_binary(OpKind::Mul, y, half),
            a.push_binary(OpKind::Mul, y, quarter),
        );
        up.push(a.push_ternary(OpKind::If, mask, raised, halved));
        down.push(a.push_ternary(OpKind::If, mask, lowered, quartered));
    }
    let sum = |a: &mut ExprArena, terms: &[ExprId]| {
        terms[1..]
            .iter()
            .fold(terms[0], |acc, &t| a.push_binary(OpKind::Add, acc, t))
    };
    let ascending = sum(&mut a, &up);
    down.reverse();
    let descending = sum(&mut a, &down);
    let root = a.push_binary(OpKind::Add, ascending, descending);
    (a, root)
}

fn predicate_reference(x: f32, y: f32) -> f64 {
    let y = f64::from(y);
    (0..PREDICATES)
        .map(|k| match joint(x, k) {
            true => (y + k as f64) + (y - k as f64),
            false => y * 0.5 + y * 0.25,
        })
        .sum()
}

#[test]
fn more_predicates_than_the_mask_file_holds_compute_their_values() {
    let (arena, root) = predicate_kernel();
    let shape = LatticeShape::new([width() as u32, ROWS as u32]);
    let result = compile(&arena, root, shape).expect("the kernel compiles");
    assert_pressured(&result);
    let out = run(&result, &[]);
    assert_values(&out, predicate_reference);
}
