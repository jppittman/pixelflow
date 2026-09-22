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

use pixelflow_codegen::CompiledKernel;
use pixelflow_codegen::emit::{EmitCtx, compile};
use pixelflow_ir::OpKind;
use pixelflow_ir::arena::{ExprArena, ExprId};

/// One point of a kernel compiled at [`pixelflow_ir::LatticeShape::POINT`],
/// the one sample read back. `CompiledKernel::call` is the collapse driver's
/// entry; a test that wants one number owns this loop rather than the crate
/// growing a point API for it.
///
/// `block` holds the kernel's arguments in link order — the addend and the
/// wall's multiplier, which used to be the Z and W coordinates — and is the
/// uniform block the context passes it in, since these arenas declare no
/// buffer.
fn eval_point(jit: &CompiledKernel, x: f32, y: f32, block: &[f32]) -> f32 {
    let mut out = [0.0f32; 1];
    let origin = [x, y];
    // SAFETY: `ctx[0]` is the uniform block — one `f32` per declared
    // argument, in link order — `ctx[1]` is the origin block, and `out`
    // holds the one sample a single-point lattice writes.
    let ctx: [*const f32; 2] = [block.as_ptr(), origin.as_ptr()];
    unsafe {
        jit.call(ctx.as_ptr(), out.as_mut_ptr(), 1);
    }
    out[0]
}

/// Declare an argument in `a` and return its leaf.
fn arg_leaf(a: &mut ExprArena) -> ExprId {
    let slot = a.declare_uniform(pixelflow_ir::Uniform::new(0.0).decl());
    a.push_uniform(slot)
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
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let z = arg_leaf(&mut a);
    let root = a.push_ternary(OpKind::MulAdd, x, y, z);

    let result =
        compile(&a, root, pixelflow_ir::LatticeShape::POINT).expect("compile MulAdd(X, Y, U)");
    // Not asserted: `result.spill_count`. Every kernel this file has compiled
    // at `LatticeShape::POINT` reports one nominal spill, independent of
    // content — a bare `Const(1.0)` and `X + Y` report the same
    // `spill_count == 1, spill_bytes == 80` this scenario does, with zero
    // corresponding store or load in `result.traffic`'s per-scope counts, so
    // it is not a register genuinely forced to memory. That looks like the
    // lattice's own row/col/lane folds (`pixelflow_ir::passes::lattice::collapse`
    // wraps every kernel in them, even a one-point one) costing a nominal
    // slot the emitted code never touches, not register pressure from this
    // scenario's operands — see this file's final report for the finding.
    // The property this test actually needs — that the multiplicands reach
    // the backend live in registers rather than reloaded — is what the bit
    // check below proves: only the fused, single-rounding form produces
    // `fused(A, B, C)`/its SSE2 stand-in.
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
/// Three things the scenario has to get right, and each has been the reason
/// an earlier version of it quietly tested the fused arm instead:
///
/// - **Both multiplicands vary along the column.** A value the lattice's
///   column fold does not vary — `Y + Y`, a uniform — is a root the emitter
///   computes once outside that fold and hands in, carried in a register or
///   reloaded at the fold's head, and either way *in* a register at the
///   `MulAdd`. So `b` is `Y + Y` plus `X · U` at `U = 0`: exactly `B`, and
///   column-varying. `a` is `X + X`; doubling is the cheapest computation that
///   is also exact, which is why the inputs are `A`/`B` halved.
/// - **The multiplicands are Belady's first victims.** The allocator evicts
///   the value read furthest ahead, so what decides the multiplicands' fate is
///   not how small the pool is (`RegisterFile::MIN_SCRATCH` — a temp cannot
///   spill, so a one-register pool is not a budget a caller can ask for) but
///   where their last read falls relative to everything competing with them.
///   They are defined first and read last, by the `MulAdd` at the root — and
///   nothing else may be read *after* the `MulAdd`, or that is what gets
///   evicted in their place.
/// - **The wall is read twice, both times before the `MulAdd`.** The schedule
///   is the legalized arena's order, which is post-order from the root: a
///   term consumed once is defined where it is consumed, and ten such terms
///   hold one register between them however they were pushed. So every term
///   is summed twice, in opposite orders (the same order would be the same
///   nodes): each is live from its first sum to its second, the pool
///   overflows, and the multiplicands — read further ahead than any term —
///   are what it sheds. A value is brought back to a register and kept only
///   when it will be read again after that; the multiplicands have one read,
///   so they are reloaded for the `MulAdd` alone, which is the decomposed arm.
///   Each term is `(X + i) · U`, exactly +0.0 at `U = 0`, so the addend is
///   bit-for-bit `z` and the rounding under test is the `MulAdd`'s alone. It
///   is built from a uniform because the folder sees through any zero it can
///   evaluate: `(l − r) − (l − r)` was folded to one constant and left no
///   wall at all.
///
/// That pressure was actually created is read from the emitted traffic — a
/// value stored to the stack — not from `spill_count`, which counts the
/// frame's slots and is nonzero for every kernel at a one-point lattice.
#[test]
fn a_spilled_muladd_rounds_twice_on_every_target() {
    fn chain(a: &mut ExprArena, terms: &[ExprId]) -> ExprId {
        terms[1..]
            .iter()
            .fold(terms[0], |acc, &t| a.push_binary(OpKind::Add, acc, t))
    }

    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let z = arg_leaf(&mut a);
    let w = arg_leaf(&mut a);

    let ma = a.push_binary(OpKind::Add, x, x);
    let yy = a.push_binary(OpKind::Add, y, y);
    let xw = a.push_binary(OpKind::Mul, x, w);
    let mb = a.push_binary(OpKind::Add, yy, xw);

    let wall: Vec<ExprId> = (1..=10u32)
        .map(|i| {
            let c = a.push_const(i as f32);
            let xi = a.push_binary(OpKind::Add, x, c);
            a.push_binary(OpKind::Mul, xi, w)
        })
        .collect();
    let forward = chain(&mut a, &wall);
    let reversed: Vec<ExprId> = wall.iter().rev().copied().collect();
    let backward = chain(&mut a, &reversed);
    let addend = a.push_binary(OpKind::Add, z, forward);
    let addend = a.push_binary(OpKind::Add, addend, backward);
    let root = a.push_ternary(OpKind::MulAdd, ma, mb, addend);

    let result = EmitCtx::with_max_regs(1)
        .compile(&a, root, pixelflow_ir::LatticeShape::POINT)
        .expect("compile spilled MulAdd");
    let stores: u32 = result.traffic.scopes.iter().map(|s| s.stores).sum();
    assert!(stores > 0, "scenario failed to create register pressure");
    let jit = CompiledKernel::new(result.code, pixelflow_ir::LatticeShape::POINT);
    let got = eval_point(&jit, HALF_A, HALF_B, &[C, 0.0]);
    assert_bits("decomposed MulAdd", got, decomposed(A, B, C));
}
