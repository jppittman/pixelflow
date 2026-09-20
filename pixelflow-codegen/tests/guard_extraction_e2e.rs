//! G3 end-to-end: extraction *chooses* `Guard` over `Select` for a kernel
//! whose arms are expensive enough to clear the branch's fixed overhead, and
//! the resulting `ExprNode::Guard` compiles and executes correctly.
//!
//! docs/plans/2026-09-12-emit-should-just-emit.md — this is the stage that
//! lets the search G2 made emittable actually get chosen. Pipeline exercised:
//!
//!   Kernel (mask.select(on, off))
//!     -> pixelflow_search::runtime::optimize_runtime_arena (the real
//!        runtime-tier saturate+extract entry point `jit_cache::compile`
//!        itself calls — `RuleSet::runtime()`, which is where G3's
//!        `Select -> Guard` rewrite lives)
//!     -> assert the optimized arena contains `ExprNode::Guard`
//!     -> pixelflow_codegen::emit::compile (G2's already-built emitter)
//!     -> executed as native code, both arms checked.
//!
//! The arms are **closed**: no coordinate `Var`, no `Uniform`, no `Buffer` —
//! only `Const` and `Round`. That is what lets extraction's finalization
//! actually name them as a `Guard`'s arms rather than falling back to
//! `Select` (`choices_to_arena`'s own doc): a coordinate-dependent arm can't
//! be split off because `passes::lattice::collapse`'s warp doesn't reach a
//! named kernel (G2's stated non-goal, plan §8), and a `Uniform`/`Buffer`-
//! dependent arm can't either — its own re-materialized arena renumbers
//! slots from scratch, with nothing reconciling that against the outer
//! kernel's calling convention, so a mismatched slot would read the wrong
//! per-call value rather than merely fail to compile.
//!
//! Each arm is 8 chained `Round`s over one `Const` (32 latency-prior cycles),
//! chosen instead of an equally "expensive" pure-arithmetic chain because
//! `ConstantFold` would otherwise collapse Add/Mul/Sub of literals to one
//! `Const` during saturation — cheap, not expensive. `Round` of a literal
//! that is not itself a `Round`'s output only ever matches `ConstantFold` at
//! the outermost application, so nesting it does not fold away; using tie
//! inputs (`2.5`, `-2.5`) additionally keeps that one application declined
//! (CLAUDE.md, "Floating point at the edges") without changing the point —
//! the chain still executes as a chain, and the platform's own tie rule
//! (nearest-even on x86, ties-away on aarch64) is what the JIT will compute,
//! which is what `expected_tie` reproduces.
use pixelflow_codegen::emit;
use pixelflow_ir::arena::ExprNode;
use pixelflow_ir::kernel::Uniform;
use pixelflow_ir::{Kernel, LatticeShape};

const ROUND_CHAIN_DEPTH: usize = 8;

fn round_chain(start: f32) -> Kernel {
    let mut k = Kernel::constant(start);
    for _ in 0..ROUND_CHAIN_DEPTH {
        k = k.round();
    }
    k
}

/// The value `round_chain(x)` executes to on this target: `x`'s tie rounded
/// once (nearest-even on x86, ties-away on aarch64 — CLAUDE.md's own table),
/// then re-rounded `ROUND_CHAIN_DEPTH - 1` more times, which is a no-op on
/// an already-integral value on both targets.
fn expected_tie(x: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        // Nearest-even.
        let floor = x.floor();
        let frac = x - floor;
        if frac == 0.5 {
            if (floor as i64) % 2 == 0 {
                floor
            } else {
                floor + 1.0
            }
        } else {
            x.round()
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // Ties away from zero — Rust's `f32::round` already does this.
        x.round()
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        x.round()
    }
}

/// `mask.select(on, off)`: `mask` reads a `Uniform` (a real, always-evaluated
/// child, unrestricted), `on`/`off` are each an 8-deep `Round` chain over a
/// tie constant — closed, and expensive enough (32 cycles each, `S = 64`) to
/// clear `GUARD_TEST_BRANCH_CYCLES + MISPREDICT_PENALTY_CYCLES` by a wide
/// margin: `hard = 18 + 0.5*64 = 50` beats `soft = 64 + 4 = 68` outright, no
/// fixture tuning needed.
fn build_guard_favoring_kernel() -> (Kernel, Uniform) {
    let flag = Uniform::new(0.0);
    let mask = flag.kernel().gt(&Kernel::constant(0.5));
    let on = round_chain(2.5);
    let off = round_chain(-2.5);
    let out = mask.select(&on, &off);
    (out, flag)
}

/// Run `kernel` through the real runtime-tier saturate+extract entry point
/// (`jit_cache::compile`'s own first step), returning the optimized arena.
fn optimize_via_production_pipeline(
    kernel: &Kernel,
    shape: LatticeShape,
) -> (pixelflow_ir::ExprArena, pixelflow_ir::ExprId) {
    let (arena, root) = kernel.parts();
    let optimized = pixelflow_search::runtime::optimize_runtime_arena(arena, root, shape)
        .expect("the e-graph must not decline this kernel");
    (optimized.0.clone(), optimized.1)
}

#[test]
fn extraction_chooses_guard_for_expensive_closed_arms() {
    let (kernel, _flag) = build_guard_favoring_kernel();
    let (optimized, _root) = optimize_via_production_pipeline(&kernel, LatticeShape::POINT);

    let has_guard = optimized
        .nodes()
        .any(|(_, node)| matches!(node, ExprNode::Guard { .. }));
    assert!(
        has_guard,
        "extraction must have chosen Guard for arms this expensive ({} nodes)",
        optimized.len()
    );
}

#[test]
fn the_extracted_guard_compiles_and_both_arms_execute_correctly() {
    let (kernel, flag) = build_guard_favoring_kernel();
    let shape = LatticeShape::POINT;
    let (optimized, root) = optimize_via_production_pipeline(&kernel, shape);

    assert!(
        optimized
            .nodes()
            .any(|(_, node)| matches!(node, ExprNode::Guard { .. })),
        "premise: this test's whole point is exercising a real Guard node"
    );

    // G2's already-built emitter: unchanged by G3, this is the path that
    // must now actually receive a Guard from a real extraction run.
    let compiled = emit::compile(&optimized, root, shape).expect("a chosen Guard must compile");

    // The optimized arena declares exactly one uniform (`flag` — the arms
    // are closed, by construction and now by assertion above), so its slot
    // is `optimized.uniforms()[0]` regardless of traversal order.
    assert_eq!(
        optimized.uniforms().len(),
        1,
        "the mask's flag must be the only declared uniform"
    );
    let _ = flag;

    let run = |flag_v: f32| -> f32 {
        let uniforms = [flag_v];
        let origin_vals = [0.0f32, 0.0f32];
        let mut out = [f32::NAN; 1];
        let ctx: [*const f32; 2] = [uniforms.as_ptr(), origin_vals.as_ptr()];
        // SAFETY: the arena declares one uniform and no buffers, so
        // `ctx[0]` is a one-`f32` uniform block and `ctx[1]` the origin
        // block; `out` holds the one sample a `POINT`-shaped lattice writes.
        unsafe {
            compiled.code.call(ctx.as_ptr(), out.as_mut_ptr(), 1);
        }
        out[0]
    };

    assert_eq!(
        run(1.0),
        expected_tie(2.5),
        "flag > 0.5 must take the on arm"
    );
    assert_eq!(
        run(0.0),
        expected_tie(-2.5),
        "flag <= 0.5 must take the off arm"
    );
}
