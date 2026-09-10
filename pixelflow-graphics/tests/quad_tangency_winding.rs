//! A tangential touch must contribute zero winding — and must keep doing so
//! when the two segments that form it round their discriminants differently.
//!
//! Where an outline reaches a local Y-extremum at an on-curve point, the two
//! quadratics meeting there each have that point as an endpoint *and* as their
//! own extremum (`t_vertex` exactly 0 for the outgoing one, exactly 1 for the
//! incoming one — 12 of `'8'`'s 32 quadratics are this shape). A horizontal
//! ray through that row grazes the outline: it enters and leaves at the same
//! point, so the two segments contribute equal coverage with opposite winding
//! signs and the sum is zero.
//!
//! That cancellation is load-bearing and it is not robust. Each segment
//! decides whether it reaches the row with its own `disc >= 0`, and on the
//! extremum row `disc = Y*4ay + (by^2 - 4ay*cy)` sits within an ulp of zero,
//! so the two decisions are made by two different roundings of two different
//! expressions. When they disagree, one segment contributes its single valid
//! root and the other contributes nothing, and a whole crossing survives.
//!
//! This test evaluates the lowered arena directly, with no *runtime* e-graph
//! in the path, so it pins the numerics rather than a bake-time extraction
//! choice. It is not free of the e-graph altogether, and an earlier revision
//! of this comment claimed it was: `AnalyticalQuad::kernel()` is built by
//! `kernel!`, so the macro tier's optimizer has already chosen an association
//! before the first line of this file runs.
//!
//! That is not a defect in the test so much as a limit on what a pin like
//! this can mean, and it showed itself when the macro tier moved from the AST
//! to the arena (docs/plans/2026-09-08-macro-tier-is-arena-native.md): the
//! residual went from 0.745 to 0.688 without anything about the numerics
//! changing. Which is precisely the trap "approach 1" below records — a
//! number that moves when the optimizer is perturbed is measuring the
//! extraction, not the bug. So read the pin as an order of magnitude with a
//! tight leash, not as a constant: what it is really asserting is that a
//! grazing ray still picks up *most of a crossing* where it must pick up
//! zero.

//! # Five approaches that do not work
//!
//! Recorded here because the next person to try this will otherwise re-derive
//! them, and four of the five look obviously right until measured.
//!
//! 1. **A tight Y-extent gate.** Excise the rows where `t_vertex` falls
//!    outside `[0, 1]`. Refuted: an `EXTENT_SLOP` of `1e6` — a gate true on
//!    every row — removes the divergence just as well. It was perturbing the
//!    e-graph into a different extraction, not fixing anything. Variants with
//!    an exact extent, a half-open one, and endpoint-exact bounds all leave
//!    the grazing residual at -0.426.
//! 2. **Dropping `disc >= 0`** so the clamped root pair cancels. Refuted: the
//!    pair does not cancel when `t_vertex` is exactly 0 or 1, which is the
//!    common TrueType shape — one root passes `t in [0, 1]` and the other does
//!    not. Regresses `'8'` at 7/13/17/19/41 px.
//! 3. **A scale-relative `MIN_DISC`**, bounding the fabricated pair's
//!    separation. Refuted: the non-cancellation is driven by root *validity*,
//!    not separation, and the residual does not move.
//! 4. **Splitting each quadratic at its vertex** into monotone pieces, so
//!    existence is decided by `Y` against two exact control-point coordinates
//!    rather than by a rounded discriminant. This one is half right — it makes
//!    the discrete decision exact and the optimized-vs-raw sweep goes green —
//!    but it moves 616 corpus texels, up to 0.83, and the cause is not the
//!    root clamp (removing the clamp gives identical numbers).
//! 5. **Ramping the contribution to zero across a band around `disc == 0`.**
//!    The closest, and refuted most decisively. It works only when *both*
//!    segments of a near-tangency are inside the band; where one is inside and
//!    the other comfortably positive, the ramp halves one signed contribution
//!    and the pair stops cancelling — so **the ramp creates the imbalance it
//!    exists to remove**. That is corpus-dependent by construction: every time
//!    the glyph set widened, the largest usable band fell (1e4, then 3e4, then
//!    2400, then ~125, then ~0.3 on a second font), while the smallest useful
//!    one stayed at 876. The window is empty, and no constant closes it.
//!
//! The instrument that refuted all five is `freetype_oracle.rs`: an external
//! rasterizer, because every check that compares this code to itself agrees
//! with the bug.
//!
use pixelflow_graphics::fonts::ttf_curve_analytical::AnalyticalQuad;
use pixelflow_ir::expr::Term;
use pixelflow_ir::{eval_scalar, passes::lower_dwrt, BindingTable};

/// The defect's measured size, pinned. **These are not tolerances — they are
/// the bug.** A grazing ray must pick up zero winding and picks up most of a
/// crossing instead; pinning it means the number cannot drift, a fix has to
/// come here and say so, and the two cases stay distinguishable.
///
/// Windows are wide enough for nothing-in-particular and tight enough that
/// half a crossing either way fails. `eval_scalar` on a lowered arena is plain
/// Rust `f32` arithmetic, so these are reproducible bit-for-bit on any target;
/// if a platform disagrees, that is a finding in itself.
/// Was 0.744_913_4 while `kernel!` optimized on the AST. The arena-native
/// macro tier associates the same expression differently, so the same defect
/// now measures 0.688 — see the note above on what this number can and
/// cannot mean.
const KNOWN_SHARED_EXTREMUM_WINDING: f32 = 0.687_676_4;
const KNOWN_ORIGIN_WINDING: f32 = 1.0;
/// How far the measured defect may stray from the pinned value.
const PIN_TOLERANCE: f32 = 0.02;

/// Found by search over shared-extremum segment pairs: coefficients whose
/// discriminants straddle zero differently at the extremum row.
/// How far either side of the extremum row to walk, in ulps. The band the fix
/// installs is ~2200 ulps wide and the residual peaks around 100 ulps out, so
/// a narrow window reports a number from the wrong part of the curve: at ±8
/// ulps this test measured 0.0076 where the true saturated peak is 8.76/K.
const WALK_ULPS: i32 = 512;

/// Worst grazing winding within [`WALK_ULPS`] of `extremum`, for a pair of
/// quadratics meeting there, sampled at `x`.
fn worst_grazing(
    incoming: AnalyticalQuad,
    outgoing: AnalyticalQuad,
    extremum: f32,
    x: f32,
) -> (f32, f32) {
    let sum = incoming.kernel().add(&outgoing.kernel());
    let term = sum.term();
    let lowered = lower_dwrt(term).expect("lower");
    let lowered_term = Term::new(lowered.entry(), term.env());

    // `next_down`/`next_up`, not bit arithmetic: the extremum here can be
    // exactly 0.0, where `to_bits() - 1` underflows (a debug panic, and in
    // release a wrap to a NaN bit pattern that silently walks nowhere near the
    // row under test).
    let mut y = extremum;
    for _ in 0..WALK_ULPS {
        y = y.next_down();
    }
    let (mut worst, mut worst_y) = (0.0f32, y);
    for _ in 0..=(2 * WALK_ULPS) {
        let v = eval_scalar(lowered_term, &[x, y], &BindingTable::empty());
        assert!(
            v.is_finite(),
            "winding is {v} at y = {y:?} ({:#x}) — a non-finite coverage is a \
             division by a vanished quantity, not an inaccurate one",
            y.to_bits()
        );
        if v.abs() > worst.abs() {
            worst = v;
            worst_y = y;
        }
        y = y.next_up();
    }
    (worst, worst_y)
}

/// Found by search over shared-extremum segment pairs: coefficients whose
/// discriminants straddle zero differently at the extremum row.
#[test]
fn grazing_winding_at_a_shared_extremum_is_still_wrong() {
    let shared = [-0.966_354_37f32, 8.683_796];
    let incoming = AnalyticalQuad::new(
        [-4.499_054, 0.079_550_94],
        [-1.554_296_3, 8.683_796],
        shared,
    );
    let outgoing =
        AnalyticalQuad::new(shared, [1.835_096_6, 8.683_796], [4.617_066_4, 3.184_099_4]);
    let (worst, worst_y) = worst_grazing(incoming, outgoing, shared[1], 39.033_646);
    assert!(
        (worst.abs() - KNOWN_SHARED_EXTREMUM_WINDING).abs() < PIN_TOLERANCE,
        "grazing winding is {worst} at y = {worst_y:?} ({:#x}), pinned at \
         {KNOWN_SHARED_EXTREMUM_WINDING}. Smaller means someone has fixed \
         this — lower the pin, or delete it and assert zero. Larger means it \
         has got worse.",
        worst_y.to_bits()
    );
}

/// The same pair, translated so the shared extremum sits at **exactly zero**.
///
/// This is not a contrived coordinate. `Font::compile` normalizes every
/// outline with the bbox y-minimum at exactly 0.0, so any glyph whose lowest
/// point is an on-curve extremum — `'8'`, `'O'`, `'e'`, most round letters —
/// puts this pair at `y == 0.0` in the frame the kernel is built in.
///
/// It is also where several candidate fixes have died: anything scaled by the
/// discriminant's distance from the coordinate origin vanishes here.
#[test]
fn grazing_winding_at_an_extremum_on_zero_is_still_wrong() {
    let dy = -8.683_796f32; // put the shared extremum on y == 0.0
    let shared = [-0.966_354_37f32, 8.683_796 + dy];
    assert_eq!(shared[1], 0.0, "the extremum must land exactly on zero");
    let incoming = AnalyticalQuad::new(
        [-4.499_054, 0.079_550_94 + dy],
        [-1.554_296_3, 8.683_796 + dy],
        shared,
    );
    let outgoing = AnalyticalQuad::new(
        shared,
        [1.835_096_6, 8.683_796 + dy],
        [4.617_066_4, 3.184_099_4 + dy],
    );
    let (worst, worst_y) = worst_grazing(incoming, outgoing, shared[1], 39.033_646);
    assert!(
        (worst.abs() - KNOWN_ORIGIN_WINDING).abs() < PIN_TOLERANCE,
        "grazing winding is {worst} at y = {worst_y:?} with the extremum on \
         zero, pinned at {KNOWN_ORIGIN_WINDING} — a whole crossing. See the \
         pin on the general case."
    );
}
