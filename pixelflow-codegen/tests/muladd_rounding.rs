//! `MulAdd`'s rounding form, asserted through compiled code.
//!
//! CLAUDE.md's platform-divergence table has a `MulAdd` row: one rounding
//! where the hardware has an FMA, two where it does not, and the two disagree
//! on inputs like `mul_add(1.0000001, 4097.0, 4097.0)`. That is a precision
//! difference the language puts on the table, not a divergence — the folder
//! and the oracle round once and never refuse an input over it. It is still
//! the entire reason the emitter carries two shapes for one op —
//! `ResolvedOp::FusedMulAdd` and `ResolvedOp::DecomposedMulAdd` — and which
//! one a node gets is decided by register pressure alone.
//!
//! Every other JIT-vs-interpreter test in this crate compares within a
//! tolerance (`spill_pressure`'s 4 ULP, `oracle_reference`'s per-op
//! `Tolerance`), and one-rounding vs. two is a last-bit difference: it fits
//! inside all of them. So a backend that silently emitted the wrong shape —
//! decomposing where it has an FMA, or fusing a decomposition — would keep
//! every one of those tests green. These assert the *bits*, on inputs chosen
//! so the two forms cannot agree.
//!
//! x86-64 only, because it executes: the encodings themselves are pinned for
//! all four backends from any host by `emit::tests::muladd_encoding`.
#![cfg(target_arch = "x86_64")]

use pixelflow_codegen::emit::executable::{Point4, TileSlice};
use pixelflow_codegen::emit::{EmitCtx, compile_dag};
use pixelflow_codegen::{CompiledKernel, JIT_VECTOR_BYTES};
use pixelflow_ir::ExprBuilder;
use pixelflow_ir::OpKind;

/// Lanes in one emitted batch.
const LANES: usize = JIT_VECTOR_BYTES / core::mem::size_of::<f32>();

/// One point of a compiled kernel: a single-batch collapse call, lane 0 read
/// back. `call_collapse` is the collapse driver's entry; a test that wants one
/// number owns this loop rather than the crate growing a point API for it.
///
/// `block` holds the kernel's arguments in link order — the addend and the
/// wall's multiplier, which used to be the Z and W coordinates.
fn eval_point(jit: &CompiledKernel, x: f32, y: f32, block: &[f32]) -> f32 {
    let mut out = [0.0f32; LANES];
    let ctx: [*const f32; 1] = [block.as_ptr()];
    // SAFETY: `out` holds exactly one whole batch; these arenas declare no
    // buffers, so `ctx[0]` is the block entry and holds one `f32` per
    // declared argument, alive for the call.
    unsafe {
        jit.call_collapse(
            ctx.as_ptr(),
            TileSlice::single(out.as_mut_ptr()),
            Point4::new([x; LANES], [y; LANES], [0.0; LANES], [0.0; LANES]),
        );
    }
    out[0]
}

/// Declare an argument in `a` and return its leaf.
fn arg_leaf(a: &mut ExprBuilder) -> pixelflow_ir::ExprHandle {
    a.uniform(pixelflow_ir::Uniform::new(0.0).decl())
}

// ── The two rounding forms, as scalar references ─────────────────────────────

/// `a*b + c` with **one** rounding — what an FMA instruction computes.
fn fused(a: f32, b: f32, c: f32) -> f32 {
    a.mul_add(b, c)
}

/// `a*b + c` with **two** roundings — a multiply, rounded, then an add.
///
/// `black_box` is load-bearing: under `+fma` LLVM contracts a plain `a*b + c`
/// into a single `fma` instruction (which is exactly why `eval_scalar`'s
/// oracle agrees with the fused form on those builds), and this function's
/// whole job is to be the answer that contraction destroys.
fn decomposed(a: f32, b: f32, c: f32) -> f32 {
    core::hint::black_box(a * b) + c
}

/// An input where the two forms differ, so an assertion against one of them
/// genuinely rejects the other. `1.0000001 * 4097.0` needs more mantissa bits
/// than an `f32` has, and rounding it before the add loses the bit that the
/// add would otherwise have kept.
const A: f32 = 1.000_000_1;
const B: f32 = 4097.0;
const C: f32 = 4097.0;

/// `A`/`B` halved: `X + X` and `Y + Y` reconstruct them exactly (doubling only
/// touches the exponent), which is how the spilled scenario below gets a
/// divergent product out of operands that are computed values rather than
/// coordinates — a coordinate is precolored into an input register and never
/// spills.
const HALF_A: f32 = 0.500_000_06;
const HALF_B: f32 = 2048.5;

// ── JIT invocation at this build's width ─────────────────────────────────────

fn assert_bits(tag: &str, got: f32, want: f32) {
    assert_eq!(
        got.to_bits(),
        want.to_bits(),
        "{tag}: JIT {got} ({:#010x}) is not {want} ({:#010x})",
        got.to_bits(),
        want.to_bits()
    );
}

/// The two forms must actually disagree on `A`/`B`/`C`, or every assertion
/// below is vacuous.
#[test]
fn the_reference_forms_disagree_on_these_inputs() {
    assert_ne!(
        fused(A, B, C).to_bits(),
        decomposed(A, B, C).to_bits(),
        "the chosen inputs no longer separate one rounding from two, so the \
         rest of this file proves nothing"
    );
    assert_eq!(
        OpKind::MulAdd.eval_ternary(A, B, C).map(f32::to_bits),
        Some(fused(A, B, C).to_bits()),
        "the folder rounds once, and does so whatever profile built it"
    );
    // The halved constants must double back exactly.
    assert_eq!((HALF_A + HALF_A).to_bits(), A.to_bits());
    assert_eq!((HALF_B + HALF_B).to_bits(), B.to_bits());
}

/// An unspilled `MulAdd(X, Y, Z)` reaches the backend as `FusedMulAdd`, and
/// what that compiles to is exactly what the target's hardware offers: one
/// rounding wherever there is an FMA, two on the SSE2 baseline, whose
/// `FusedMulAdd` arm is a `movaps`/`mulps`/`addps` stand-in because that is
/// all SSE2 has.
#[test]
fn an_unspilled_muladd_rounds_the_way_this_target_does() {
    let mut a = ExprBuilder::new();
    let x = a.var(0);
    let y = a.var(1);
    let z = arg_leaf(&mut a);
    let root = a.ternary(OpKind::MulAdd, x, y, z);
    let graph = a.finish_one(root);

    let result = compile_dag(graph.root(), graph.environment()).expect("compile MulAdd(X, Y, U)");
    assert_eq!(result.spill_count, 0, "this scenario must not spill");
    let jit = CompiledKernel::new(result.code, pixelflow_ir::LatticeShape::POINT);
    let got = eval_point(&jit, A, B, &[C]);

    #[cfg(target_feature = "fma")]
    assert_bits("fused MulAdd on an FMA target", got, fused(A, B, C));
    #[cfg(not(target_feature = "fma"))]
    assert_bits(
        "fused MulAdd on the SSE2 baseline",
        got,
        decomposed(A, B, C),
    );
}

/// Under enough register pressure that `a` and `b` cannot both stay in
/// registers, the same node reaches the backend as `DecomposedMulAdd` — a
/// multiply and an add, two roundings, on *every* target including the ones
/// with an FMA.
///
/// This is the arm AVX-512 had no test for at all: `spill_pressure.rs`'s
/// scenarios are sized for the six-register SSE2 pool and stop spilling
/// against AVX-512's nineteen, so that whole file is `cfg`'d off there.
/// Shrinking the pool explicitly — `EmitCtx::with_max_regs`, which its own
/// doc calls "how a caller forces spilling deliberately" — reaches it at
/// every width instead of at whichever one the scenario happened to suit.
///
/// The multiplicands have to be *computed* values: a coordinate is precolored
/// into an input register and never spills, so `MulAdd(X, Y, Z)` cannot
/// decompose no matter how small the pool is. Doubling is the cheapest
/// computation that is also exact, which is why the inputs are `A`/`B` halved.
///
/// The `MulAdd` also has to come *last*. The allocator evicts by Belady — the
/// value needed furthest in the future — so what decides the multiplicands'
/// fate is not how small the pool is but where their last use falls relative
/// to everything competing for a register. With the `MulAdd` in the middle and
/// the wall summed after it, the wall outlives the multiplicands and the wall
/// is what spills, however tight the budget; the scenario only ever reached
/// the decomposed arm because a one-register pool spills whatever it is
/// holding. That is no longer a budget a caller can ask for
/// (`RegisterFile::MIN_SCRATCH` — a temp cannot spill), so the pressure has to
/// come from the live ranges rather than from the pool being a single
/// register.
#[test]
fn a_spilled_muladd_rounds_twice_on_every_target() {
    let mut a = ExprBuilder::new();
    let x = a.var(0);
    let y = a.var(1);
    let z = arg_leaf(&mut a);
    // The multiplicands, defined first and consumed last, so they hold the
    // longest live ranges in the schedule — which is what makes them Belady's
    // first two eviction choices once the wall fills the pool.
    let ma = a.binary(OpKind::Add, x, x);
    let mb = a.binary(OpKind::Add, y, y);

    // The wall. One spilled multiplicand is not enough — `resolve_operands`
    // only decomposes when *both* are out of registers — so something has to
    // hold the pool across the whole schedule.
    //
    // Every term is `(X + i) · U`, which is exactly +0.0 at U = 0 and so
    // perturbs no bit of the sum it lands in. It has to be built from
    // something the folder cannot see through: the wall here was
    // `(l − r) − (l − r)`, also worth zero, and `legalize` folded all six of
    // those to one constant — the scenario had no wall at all, and only ever
    // reached the decomposed arm because the pool was smaller than three. A
    // uniform is never folded — its value is unknown until the call — so
    // multiplying by it keeps the term worth zero without saying so in a form
    // the folder can see. Each term still depends on X, so none is
    // loop-invariant and hoistable out of a collapse body.
    let w = arg_leaf(&mut a);
    let wall = (1..=10u32)
        .map(|i| {
            let c = a.constant(i as f32);
            let xi = a.binary(OpKind::Add, x, c);
            a.binary(OpKind::Mul, xi, w)
        })
        .collect();

    // Sum the wall first, then feed it to the `MulAdd` as part of the addend.
    // Each term is exactly +0.0, so the addend is bit-for-bit `z` and the
    // rounding under test is the `MulAdd`'s alone.
    let wall_sum = wall
        .iter()
        .skip(1)
        .fold(wall[0], |acc, &w| a.binary(OpKind::Add, acc, w));
    let addend = a.binary(OpKind::Add, z, wall_sum);
    let root = a.ternary(OpKind::MulAdd, ma, mb, addend);
    let graph = a.finish_one(root);

    let result = EmitCtx::with_max_regs(1)
        .compile_dag(graph.root(), graph.environment())
        .expect("compile spilled MulAdd");
    assert!(
        result.spill_count > 0,
        "scenario failed to create register pressure"
    );
    let jit = CompiledKernel::new(result.code, pixelflow_ir::LatticeShape::POINT);
    let got = eval_point(&jit, HALF_A, HALF_B, &[C, 0.0]);
    assert_bits("decomposed MulAdd", got, decomposed(A, B, C));
}
