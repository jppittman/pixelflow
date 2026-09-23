//! A kernel fold (`Reduce`) inside a `Select` arm, when something outside the
//! arm depends on it — or it depends on something outside the arm — through
//! the fold's *body* rather than through a register operand.
//!
//! `pixelflow-codegen/src/emit/guards.rs` decides which schedule entries a
//! select's arm owns (`select_arms`), and reorders a scope so an arm is one
//! contiguous run (`cluster_select_arms`). Both used to read dependencies off
//! `regalloc::operands` alone, which treats a `Reduce` def as a leaf: once
//! `extract_folds` has carved a fold's body into its own scope, nothing in the
//! enclosing scope's schedule recorded what that body reads. They now read it
//! from `guards::FoldReads` as well, which makes the def a consumer of what its
//! fold reads. Two miscompiles came of the missing edges, each pinned below
//! against a plain-`f64` reference (never a pixelflow evaluator):
//!
//! 1. **A guard skipped a fold that a sibling fold still read.**
//!    `out = select(X < T, W·sin(X/10), 0) + D`, with `W = Σ_j |X − j/2|` and
//!    `D = Σ_k |W − 40k|`. `D`'s body reads `W`'s accumulator through a
//!    placeholder, so `W`'s only consumer in the batch scope was the arm's
//!    `Mul`; `W`'s `Reduce` is never a scope root (`stays_put`), so the
//!    `OUTSIDE` pin did not cover it either. `sin`'s expansion alone prices
//!    the arm past the 16-cycle bound, and it was guarded — `W`'s whole loop
//!    inside the skipped range. On a batch whose mask is
//!    uniformly false, `D` read `W`'s slot as the last batch that ran the arm
//!    left it (or as the frame's initial garbage).
//!
//! 2. **Clustering moved a fold's input past the fold.**
//!    `out = select(X < T, W·sin(X/10), 0)` with `W = Σ_j |X + Y − j/2|`.
//!    `X + Y` is a batch-scope root read only by `W`'s body, so to the select
//!    it was a stranger — outside its cone, since `W`'s `Reduce` had no
//!    operands — and `partition_around` sank it past the select, *after* `W`'s
//!    loop. `W` then read the previous batch's `X + Y` on every batch, whether
//!    or not any guard fired; `is_topological` walked the same leaf-`Reduce`
//!    operands and did not notice. No sibling fold is involved.
//!
//! Section 3 is the same two hazards reached another way — through a fold
//! nested in the reader, from the mask or the other arm, in a fold's own
//! body rather than the batch scope, and in a row's remainder batch — each of
//! which failed before `FoldReads` as well: they pin the claims its doc makes
//! (the edges are transitive, and every scope has them), not new mechanisms.
//!
//! The controls at the bottom compile the same folds without the select, with
//! an arm that owns nothing to guard or cluster, and with the sibling reading
//! `W` through an ordinary batch-scope value — all correct.
//!
//! The errors were stale accumulators — a whole batch's `W` in place of
//! another — not rounding; the tolerance is sized for `f32` accumulation over
//! ~100 terms and `sin`'s polynomial, and nothing more.
//!
//! `PIXELFLOW_GUARD_TELEMETRY=1 cargo test -p pixelflow-core --test
//! guard_sibling_fold -- --nocapture --test-threads=1` prints, per scope,
//! which selects were guarded and how many entries each arm skips.

#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use pixelflow_core::Lattice;
use pixelflow_ir::Kernel;

/// Wide enough to span several SIMD batches at every supported width
/// (4, 8 or 16 lanes), so uniform-true and uniform-false batches both exist.
const WIDTH: usize = 64;
/// More than one row, so a stale slot can also be the *previous row's*.
const HEIGHT: usize = 4;
/// Trip count of every fold. Even, so the optimizer halves the fold without
/// peeling a term out of it: `W` stays a bare `Reduce`, and a sibling reads
/// its accumulator directly. (With an odd count the peeled term makes `W` an
/// `Add` in the batch scope, which is an ordinary root, and pins the fold.)
const TRIPS: u32 = 100;
/// `W`'s per-trip offset step.
const W_STEP: f64 = 0.5;
/// `D`'s per-trip offset step: `40·k` sweeps `W`'s range, so `D` is not affine
/// in `W` and the e-graph cannot factor `W` out of `D`'s body.
const D_STEP: f64 = 40.0;
/// The phase scale inside `sin` — enough exclusive work in the arm to clear
/// the guard's 16-cycle bound.
const PHASE: f64 = 0.1;
/// The mask's threshold on X: at any lane width, most batches are uniform on
/// one side of it or the other.
const THRESHOLD: f64 = 20.0;
/// Relative tolerance: `f32` summation of ~100 terms, and `sin`'s polynomial.
const REL_TOL: f64 = 1e-4;

// ─────────────────────────── kernels ───────────────────────────

fn x() -> Kernel {
    Kernel::x()
}

fn c(v: f64) -> Kernel {
    Kernel::constant(v as f32)
}

/// `W = Σ_{j<N} |X − W_STEP·j|` — reads only `X`, which the mask reads too.
fn w_of_x() -> Kernel {
    Kernel::sum_over(TRIPS, |j| x().sub(&j.mul(&c(W_STEP))).abs())
}

/// `W = Σ_{j<N} |X + Y − W_STEP·j|` — reads `X + Y`, which nothing outside
/// the fold reads.
fn w_of_x_plus_y() -> Kernel {
    let s = x().add(&Kernel::y());
    Kernel::sum_over(TRIPS, |j| s.sub(&j.mul(&c(W_STEP))).abs())
}

/// `D = Σ_{k<N} |W − D_STEP·k|` — a sibling fold whose body reads `W`.
fn sibling_of(w: &Kernel) -> Kernel {
    Kernel::sum_over(TRIPS, |k| w.sub(&k.mul(&c(D_STEP))).abs())
}

/// `W·sin(PHASE·X)` — the arm's work.
fn heavy_of(w: &Kernel) -> Kernel {
    w.mul(&x().mul(&c(PHASE)).sin())
}

// ────────────────────────── reference ──────────────────────────

fn w_ref(s: f64) -> f64 {
    (0..TRIPS).map(|j| (s - W_STEP * f64::from(j)).abs()).sum()
}

fn sibling_ref(w: f64) -> f64 {
    (0..TRIPS).map(|k| (w - D_STEP * f64::from(k)).abs()).sum()
}

fn heavy_ref(w: f64, x: f64) -> f64 {
    w * (x * PHASE).sin()
}

/// Bake `kernel` and compare every texel with `want(x, y)`; report every
/// mismatch at once, so a failure shows the pattern (which batches, which
/// rows) and not one texel.
fn check(name: &str, kernel: &Kernel, want: impl Fn(f64, f64) -> f64) {
    check_at(name, WIDTH, kernel, want);
}

/// [`check`] on a lattice `width` samples wide.
fn check_at(name: &str, width: usize, kernel: &Kernel, want: impl Fn(f64, f64) -> f64) {
    let baked = Lattice::frame(width, HEIGHT).bake(kernel);
    let buf = baked.buffer();
    assert_eq!(buf.len(), width * HEIGHT);

    let mut bad = Vec::new();
    for row in 0..HEIGHT {
        for col in 0..width {
            let got = f64::from(buf[row * width + col]);
            let want = want(col as f64, row as f64);
            let within = (got - want).abs() <= REL_TOL * want.abs().max(1.0);
            if !within {
                bad.push((col, row, got, want));
            }
        }
    }
    assert!(
        bad.is_empty(),
        "{name}: {} of {} texels wrong; (x, y, got, want): {:?}",
        bad.len(),
        width * HEIGHT,
        bad
    );
}

// ─────── 1. a guard may not skip a fold that a sibling fold reads ───────

/// `select(X < T, W·sin, 0) + D`: batches right of `T` are uniformly false
/// and skip the arm. When that took `W`'s loop with it, `D` read the last
/// left batch's `W`.
#[test]
fn a_guard_keeps_a_fold_a_sibling_fold_reads_mask_true_on_the_left() {
    let w = w_of_x();
    let k = x()
        .lt(&c(THRESHOLD))
        .select(&heavy_of(&w), &c(0.0))
        .add(&sibling_of(&w));
    check("mask true on the left", &k, |x, _| {
        let w = w_ref(x);
        let arm = if x < THRESHOLD { heavy_ref(w, x) } else { 0.0 };
        arm + sibling_ref(w)
    });
}

/// `select(X >= T, W·sin, 0) + D`: the first batches of every row skip the
/// arm. When that took `W`'s loop with it, `D` read the previous row's last
/// `W` — or, on the first row, a slot nothing had written yet.
#[test]
fn a_guard_keeps_a_fold_a_sibling_fold_reads_mask_true_on_the_right() {
    let w = w_of_x();
    let k = x()
        .ge(&c(THRESHOLD))
        .select(&heavy_of(&w), &c(0.0))
        .add(&sibling_of(&w));
    check("mask true on the right", &k, |x, _| {
        let w = w_ref(x);
        let arm = if x >= THRESHOLD { heavy_ref(w, x) } else { 0.0 };
        arm + sibling_ref(w)
    });
}

/// `select(X < T, 0, W·sin) + D`: the expensive arm is the false one, skipped
/// where the mask is uniformly true.
#[test]
fn a_guard_keeps_a_fold_a_sibling_fold_reads_heavy_false_arm() {
    let w = w_of_x();
    let k = x()
        .lt(&c(THRESHOLD))
        .select(&c(0.0), &heavy_of(&w))
        .add(&sibling_of(&w));
    check("heavy false arm", &k, |x, _| {
        let w = w_ref(x);
        let arm = if x < THRESHOLD { 0.0 } else { heavy_ref(w, x) };
        arm + sibling_ref(w)
    });
}

/// `D + select(X < T, W·sin, 0)`: the sibling first in source order.
#[test]
fn a_guard_keeps_a_fold_a_sibling_fold_reads_sibling_first() {
    let w = w_of_x();
    let k = sibling_of(&w).add(&x().lt(&c(THRESHOLD)).select(&heavy_of(&w), &c(0.0)));
    check("sibling first", &k, |x, _| {
        let w = w_ref(x);
        let arm = if x < THRESHOLD { heavy_ref(w, x) } else { 0.0 };
        sibling_ref(w) + arm
    });
}

// ──── 2. clustering may not sink a fold's input past the fold ────

/// `select(X < T, W·sin, 0)` with `W` reading `X + Y`. No sibling. When
/// clustering sank `X + Y` past the select, this was wrong on the *true*
/// side — the batches that do run the arm — because `W`'s loop ran before
/// this batch's `X + Y` was computed.
#[test]
fn clustering_keeps_a_folds_input_ahead_of_the_fold() {
    let w = w_of_x_plus_y();
    let k = x().lt(&c(THRESHOLD)).select(&heavy_of(&w), &c(0.0));
    check("fold input moved after the fold", &k, |x, y| {
        if x < THRESHOLD {
            heavy_ref(w_ref(x + y), x)
        } else {
            0.0
        }
    });
}

// ─────── 3. the edges reach every read, from every scope ───────

/// A width no lane count divides: every row ends in a remainder batch, a
/// second column fold sharing the first's closure — `W` and `D` open in both.
const RAGGED_WIDTH: usize = 61;

/// Bug 1 with a remainder batch: the same guard forms in both column folds.
#[test]
fn a_guard_keeps_a_fold_a_sibling_fold_reads_in_a_remainder_batch() {
    let w = w_of_x();
    let k = x()
        .lt(&c(THRESHOLD))
        .select(&heavy_of(&w), &c(0.0))
        .add(&sibling_of(&w));
    check_at("remainder batch", RAGGED_WIDTH, &k, |x, _| {
        let w = w_ref(x);
        let arm = if x < THRESHOLD { heavy_ref(w, x) } else { 0.0 };
        arm + sibling_ref(w)
    });
}

/// Trip count of the two-level folds below, whose cost is the product.
const NESTED_TRIPS: u32 = 12;
/// The inner fold's per-trip step, distinct from the outer's so neither
/// factors out of the other.
const INNER_STEP: f64 = 3.0;

/// `select(X < T, W·sin, 0) + D` with `D = Σ_k |E_k − 40k|` and
/// `E_k = Σ_m |W − 3m − k|`: only `E`, nested in `D`, reads `W`. `D`'s
/// edge to `W` is there because `E`'s schedule is carved out of `D`'s.
#[test]
fn a_guard_keeps_a_fold_a_nested_fold_reads() {
    let w = w_of_x();
    let d = Kernel::sum_over(NESTED_TRIPS, |k| {
        let e = Kernel::sum_over(NESTED_TRIPS, |m| w.sub(&m.mul(&c(INNER_STEP))).sub(k).abs());
        e.sub(&k.mul(&c(D_STEP))).abs()
    });
    let k = x().lt(&c(THRESHOLD)).select(&heavy_of(&w), &c(0.0)).add(&d);
    check("nested reader", &k, |x, _| {
        let w = w_ref(x);
        let d: f64 = (0..NESTED_TRIPS)
            .map(f64::from)
            .map(|k| {
                let e: f64 = (0..NESTED_TRIPS)
                    .map(|m| (w - INNER_STEP * f64::from(m) - k).abs())
                    .sum();
                (e - D_STEP * k).abs()
            })
            .sum();
        let arm = if x < THRESHOLD { heavy_ref(w, x) } else { 0.0 };
        arm + d
    });
}

/// `select(X < T, W·sin, 0)` with `W = Σ_j |E_j − j/2|` and
/// `E_j = Σ_m |X + Y − m − 3j|`: bug 2 with `X + Y` read only by the fold
/// nested in `W`.
#[test]
fn clustering_keeps_a_nested_folds_input_ahead_of_the_fold() {
    let s = x().add(&Kernel::y());
    let w = Kernel::sum_over(NESTED_TRIPS, |j| {
        let e = Kernel::sum_over(NESTED_TRIPS, |m| s.sub(m).sub(&j.mul(&c(INNER_STEP))).abs());
        e.sub(&j.mul(&c(W_STEP))).abs()
    });
    let k = x().lt(&c(THRESHOLD)).select(&heavy_of(&w), &c(0.0));
    check("nested input", &k, |x, y| {
        let w: f64 = (0..NESTED_TRIPS)
            .map(f64::from)
            .map(|j| {
                let e: f64 = (0..NESTED_TRIPS)
                    .map(|m| (x + y - f64::from(m) - INNER_STEP * j).abs())
                    .sum();
                (e - W_STEP * j).abs()
            })
            .sum();
        if x < THRESHOLD { heavy_ref(w, x) } else { 0.0 }
    });
}

/// Trip count of the mask fold below.
const MASK_TRIPS: u32 = 8;
/// The mask fold's lowest threshold on `W`: `W` exceeds it for `X < 6` and
/// `X > 44`, so uniformly true, uniformly false and mixed batches all occur.
const MASK_BASE: f64 = 2000.0;
/// The mask fold's threshold step.
const MASK_STEP: f64 = 150.0;

/// `select(∃j. W > MASK_BASE + MASK_STEP·j, W·sin, 0)`: the mask is a fold
/// whose body reads `W`, so `W` is the mask's, not the arm's — or the
/// partition moved it after the mask's loop.
#[test]
fn a_guard_keeps_a_fold_the_mask_fold_reads() {
    let w = w_of_x();
    let mask = Kernel::any_over(MASK_TRIPS, |j| {
        w.gt(&j.mul(&c(MASK_STEP)).add(&c(MASK_BASE)))
    });
    let k = mask.select(&heavy_of(&w), &c(0.0));
    check("mask fold", &k, |x, _| {
        let w = w_ref(x);
        let any = (0..MASK_TRIPS).any(|j| w > MASK_STEP * f64::from(j) + MASK_BASE);
        if any { heavy_ref(w, x) } else { 0.0 }
    });
}

/// `select(X < T, W·sin, D·cos)`: `W` is read by the true arm and, through
/// `D`'s body, by the false one — so by neither alone.
#[test]
fn a_guard_keeps_a_fold_the_other_arm_reads() {
    let w = w_of_x();
    let cos = x().mul(&c(PHASE)).cos();
    let k = x()
        .lt(&c(THRESHOLD))
        .select(&heavy_of(&w), &sibling_of(&w).mul(&cos));
    check("other arm", &k, |x, _| {
        let w = w_ref(x);
        if x < THRESHOLD {
            heavy_ref(w, x)
        } else {
            sibling_ref(w) * (x * PHASE).cos()
        }
    });
}

/// Trip count of the outer fold below.
const OUTER_TRIPS: u32 = 6;
/// Trip count of the folds nested in it.
const INNER_TRIPS: u32 = 40;

/// Bug 1 inside a fold's own body: `Σ_j select(X + j < T, V_j·sin, 0) + U_j`
/// with `V_j = Σ_m |X + j − m/2|` and `U_j = Σ_k |V_j − 40k|`. The select, `V`
/// and `U` are all in the outer fold's scope, whose edges `allocate_nest`
/// and `cluster_pending` build from that fold's children.
#[test]
fn a_guard_in_a_folds_body_keeps_a_fold_a_sibling_fold_reads() {
    let k = Kernel::sum_over(OUTER_TRIPS, |j| {
        let xj = x().add(j);
        let v = Kernel::sum_over(INNER_TRIPS, |m| xj.sub(&m.mul(&c(W_STEP))).abs());
        let u = Kernel::sum_over(INNER_TRIPS, |k| v.sub(&k.mul(&c(D_STEP))).abs());
        xj.lt(&c(THRESHOLD))
            .select(&v.mul(&xj.mul(&c(PHASE)).sin()), &c(0.0))
            .add(&u)
    });
    check("guard in a fold's body", &k, |x, _| {
        (0..OUTER_TRIPS)
            .map(|j| {
                let xj = x + f64::from(j);
                let v: f64 = (0..INNER_TRIPS)
                    .map(|m| (xj - W_STEP * f64::from(m)).abs())
                    .sum();
                let u: f64 = (0..INNER_TRIPS)
                    .map(|k| (v - D_STEP * f64::from(k)).abs())
                    .sum();
                let arm = if xj < THRESHOLD {
                    v * (xj * PHASE).sin()
                } else {
                    0.0
                };
                arm + u
            })
            .sum()
    });
}

// ────────────────────────── controls ──────────────────────────

/// `W` alone.
#[test]
fn control_the_fold_alone() {
    check("fold alone", &w_of_x(), |x, _| w_ref(x));
}

/// `D` alone: a fold reading a fold, no select.
#[test]
fn control_the_sibling_alone() {
    check("sibling alone", &sibling_of(&w_of_x()), |x, _| {
        sibling_ref(w_ref(x))
    });
}

/// `W·sin + D`: the same folds and the same arm work, no select.
#[test]
fn control_both_folds_without_the_select() {
    let w = w_of_x_plus_y();
    let k = heavy_of(&w).add(&sibling_of(&w));
    check("no select", &k, |x, y| {
        let w = w_ref(x + y);
        heavy_ref(w, x) + sibling_ref(w)
    });
}

/// `select(X < T, W, 0) + D`: the arm is the fold and nothing else, and `D`
/// reads the fold too, so the arm owns nothing — never guarded, never
/// clustered, however its loop is priced — and nothing moves or is skipped.
#[test]
fn control_an_arm_that_owns_nothing() {
    let w = w_of_x_plus_y();
    let k = x()
        .lt(&c(THRESHOLD))
        .select(&w, &c(0.0))
        .add(&sibling_of(&w));
    check("arm owning nothing", &k, |x, y| {
        let w = w_ref(x + y);
        let arm = if x < THRESHOLD { w } else { 0.0 };
        arm + sibling_ref(w)
    });
}

/// `select(X < T, W·sin, 0) + D'` with `D' = Σ_k |W + Y − 40k|`: the sibling
/// reads `W + Y`, which the batch scope computes and parks — an ordinary
/// root, pinned outside every arm, and `W` with it. The guard fired and the
/// answer was right even before the fix, which is what made a *direct* read
/// the hazard.
#[test]
fn control_a_sibling_reading_through_a_parked_value() {
    let w = w_of_x();
    let d = Kernel::sum_over(TRIPS, |k| w.add(&Kernel::y()).sub(&k.mul(&c(D_STEP))).abs());
    let k = x().lt(&c(THRESHOLD)).select(&heavy_of(&w), &c(0.0)).add(&d);
    check("sibling through a parked value", &k, |x, y| {
        let w = w_ref(x);
        let arm = if x < THRESHOLD { heavy_ref(w, x) } else { 0.0 };
        let d: f64 = (0..TRIPS)
            .map(|k| (w + y - D_STEP * f64::from(k)).abs())
            .sum();
        arm + d
    });
}
