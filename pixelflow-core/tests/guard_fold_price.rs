//! A `Select` arm that owns a fold (`Reduce`) is worth a branch by what the
//! fold's loop costs to run: its trip count times its body, plus the combine
//! per trip.
//!
//! `pixelflow-codegen/src/emit/guards.rs` guards an arm only when its price
//! clears `MISPREDICT_PENALTY_CYCLES` (16): an arm cheaper than a mispredict
//! cannot pay for its own branch. The price is the latency prior summed over
//! the entries the arm owns, and it used to take a fold's entry at
//! `cost(Reduce) · len`, where the table's `Reduce` is 0 — so an arm that is
//! nothing but a 64-trip loop was priced 0 and never guarded, while the
//! extractor that chose the loop had priced it at `len · body`. One fact, two
//! answers. The guard analysis now prices the loop with the extractor's own
//! formula (`CostModel::fold_cost`), its body priced over the fold's own
//! schedule, nested folds recursively.
//!
//! Every kernel in the first section has an arm that is a fold and nothing
//! else it owns, so it is guarded now and was not before: on a batch whose
//! mask is uniform the whole loop is jumped over. Each is checked texel by
//! texel against a plain-`f64` reference, never a pixelflow evaluator; a
//! skipped loop that should have run, or a run loop whose result was then
//! dropped, shows as a wrong texel. The second section is guarded now too,
//! around more than one loop: both arms loops, an arm skipping a fold and the
//! fold only it reads (at one and at two scopes' distance), a mask that is
//! itself a loop, and the glyph's shape of a fold read across two selects.
//! The third section is two arms the price now clears but ownership still
//! refuses — correctness only, and what the demand-regions work (D1) is for.
//!
//! That the guard forms is not visible from here — nothing outside
//! `pixelflow-codegen` can name a guard — and is pinned there
//! (`emit::guards`' unit tests price the arm; `emit`'s test
//! `a_fold_owned_by_an_arm_is_guarded` finds the branch in the allocated
//! nest). `PIXELFLOW_GUARD_TELEMETRY=1 cargo test -p pixelflow-core --test
//! guard_fold_price -- --nocapture --test-threads=1` prints, per scope, the
//! entries each arm skips.

#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use pixelflow_core::Lattice;
use pixelflow_ir::Kernel;

/// Wide enough to span several SIMD batches at every supported width
/// (4, 8 or 16 lanes), so uniform-true and uniform-false batches both exist.
const WIDTH: usize = 64;
/// A width no lane count divides: every row ends in a remainder batch, whose
/// column fold opens the same folds again.
const RAGGED_WIDTH: usize = 61;
/// More than one row, so a stale accumulator can also be the previous row's.
const HEIGHT: usize = 4;
/// The arm fold's trip count: `64 · (sub + abs) + 63 · add`, far past the
/// guard's 16-cycle bound, where the arm used to be priced 0.
const TRIPS: u32 = 64;
/// The mask's threshold on `X`: at any lane width, most batches are uniform
/// on one side of it or the other.
const THRESHOLD: f64 = 20.0;
/// Trip count of the outer fold in the two nested kernels below.
const OUTER_TRIPS: u32 = 4;
/// The inner fold's per-outer-trip offset, so the two binders do not merge.
const INNER_STEP: f64 = 8.0;
/// Trip count of `G`, a second fold over `X`: distinct from `F`'s, as is
/// [`SECOND_STEP`], so the e-graph cannot merge the two.
const SECOND_TRIPS: u32 = 48;
/// `G`'s per-trip offset.
const SECOND_STEP: f64 = 2.0;
/// Trip count of a fold whose body reads `F`'s accumulator.
const READER_TRIPS: u32 = 16;
/// That reader's per-trip offset: `100·k` sweeps `F`'s range, so the reader
/// is not affine in `F` and `F` cannot be factored out of it.
const READER_STEP: f64 = 100.0;
/// Trip count of the mask fold `∃j. X > MASK_BASE + MASK_STEP·j`.
const MASK_TRIPS: u32 = 8;
/// The mask fold's lowest threshold on `X`.
const MASK_BASE: f64 = 10.0;
/// The mask fold's threshold step.
const MASK_STEP: f64 = 5.0;
/// The second select's threshold in the two-select kernel: right of it, far
/// from [`THRESHOLD`], so each select has uniform batches of its own.
const FAR_THRESHOLD: f64 = 40.0;
/// Relative tolerance: `f32` summation of 64 terms, each an integer here.
const REL_TOL: f64 = 1e-5;

// ─────────────────────────── kernels ───────────────────────────

fn x() -> Kernel {
    Kernel::x()
}

fn y() -> Kernel {
    Kernel::y()
}

fn c(v: f64) -> Kernel {
    Kernel::constant(v as f32)
}

/// `F(s) = Σ_{j<TRIPS} |s − j|`.
fn fold_of(s: &Kernel) -> Kernel {
    Kernel::sum_over(TRIPS, |j| s.sub(j).abs())
}

/// `G(s) = Σ_{j<SECOND_TRIPS} |s − SECOND_STEP·j|`.
fn second_fold_of(s: &Kernel) -> Kernel {
    Kernel::sum_over(SECOND_TRIPS, |j| s.sub(&j.mul(&c(SECOND_STEP))).abs())
}

/// `R(a) = Σ_{k<READER_TRIPS} |a − READER_STEP·k|`: a fold whose body reads
/// `a` — for `a` a fold that does not vary with `k`, its accumulator, through
/// a placeholder.
fn reader_of(a: &Kernel) -> Kernel {
    Kernel::sum_over(READER_TRIPS, |k| a.sub(&k.mul(&c(READER_STEP))).abs())
}

// ────────────────────────── reference ──────────────────────────

fn fold_ref(s: f64) -> f64 {
    (0..TRIPS).map(|j| (s - f64::from(j)).abs()).sum()
}

fn second_fold_ref(s: f64) -> f64 {
    (0..SECOND_TRIPS)
        .map(|j| (s - SECOND_STEP * f64::from(j)).abs())
        .sum()
}

fn reader_ref(a: f64) -> f64 {
    (0..READER_TRIPS)
        .map(|k| (a - READER_STEP * f64::from(k)).abs())
        .sum()
}

/// Bake `kernel` on a `width × HEIGHT` frame and compare every texel with
/// `want(x, y)`; report every mismatch at once, so a failure shows the
/// pattern (which batches, which rows) and not one texel.
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

/// [`check_at`] on the [`WIDTH`]-wide frame.
fn check(name: &str, kernel: &Kernel, want: impl Fn(f64, f64) -> f64) {
    check_at(name, WIDTH, kernel, want);
}

// ─────────────── an arm that is a loop is guarded ───────────────

/// `select(X < T, F(X), 0) + Y`: batches right of `T` skip the loop.
#[test]
fn an_arm_that_is_a_fold_mask_true_on_the_left() {
    let k = x()
        .lt(&c(THRESHOLD))
        .select(&fold_of(&x()), &c(0.0))
        .add(&y());
    check("mask true on the left", &k, |x, y| {
        let arm = if x < THRESHOLD { fold_ref(x) } else { 0.0 };
        arm + y
    });
}

/// `select(X >= T, F(X), 0) + Y`: the first batches of every row skip the
/// loop, so a skipped loop's accumulator is also the previous row's.
#[test]
fn an_arm_that_is_a_fold_mask_true_on_the_right() {
    let k = x()
        .ge(&c(THRESHOLD))
        .select(&fold_of(&x()), &c(0.0))
        .add(&y());
    check("mask true on the right", &k, |x, y| {
        let arm = if x >= THRESHOLD { fold_ref(x) } else { 0.0 };
        arm + y
    });
}

/// `select(X < T, 0, F(X)) + Y`: the loop is the false arm, skipped where the
/// mask is uniformly true.
#[test]
fn an_arm_that_is_a_fold_heavy_false_arm() {
    let k = x()
        .lt(&c(THRESHOLD))
        .select(&c(0.0), &fold_of(&x()))
        .add(&y());
    check("heavy false arm", &k, |x, y| {
        let arm = if x < THRESHOLD { 0.0 } else { fold_ref(x) };
        arm + y
    });
}

/// The first kernel on a width no lane count divides.
#[test]
fn an_arm_that_is_a_fold_in_a_remainder_batch() {
    let k = x()
        .lt(&c(THRESHOLD))
        .select(&fold_of(&x()), &c(0.0))
        .add(&y());
    check_at("remainder batch", RAGGED_WIDTH, &k, |x, y| {
        let arm = if x < THRESHOLD { fold_ref(x) } else { 0.0 };
        arm + y
    });
}

/// `select(X < T, Σ_{j<4} |F(X − 8j) − j|, 0) + Y`: the arm is a fold whose
/// body is a fold, priced as the product of the two trip counts.
#[test]
fn an_arm_that_is_a_nested_fold() {
    let arm = Kernel::sum_over(OUTER_TRIPS, |j| {
        fold_of(&x().sub(&j.mul(&c(INNER_STEP)))).sub(j).abs()
    });
    let k = x().lt(&c(THRESHOLD)).select(&arm, &c(0.0)).add(&y());
    check("nested fold", &k, |x, y| {
        let arm: f64 = (0..OUTER_TRIPS)
            .map(f64::from)
            .map(|j| (fold_ref(x - INNER_STEP * j) - j).abs())
            .sum();
        (if x < THRESHOLD { arm } else { 0.0 }) + y
    });
}

// ───── an arm that owns more than one loop, and the loops around it ─────

/// `select(X < T, F(X), G(X)) + Y`: both arms are loops, and each is skipped
/// where the mask is uniformly the other way.
#[test]
fn both_arms_are_folds() {
    let k = x()
        .lt(&c(THRESHOLD))
        .select(&fold_of(&x()), &second_fold_of(&x()))
        .add(&y());
    check("both arms folds", &k, |x, y| {
        let arm = if x < THRESHOLD {
            fold_ref(x)
        } else {
            second_fold_ref(x)
        };
        arm + y
    });
}

/// `select(X < T, R(F(X)), 0) + Y`: `F` is invariant in `R`, so it opens
/// beside `R` rather than inside it, and only `R`'s body reads it — the arm
/// owns both loops, and the range the branch skips holds two, `F`'s before
/// `R`'s. Skipped together they are right; `R` run without this batch's `F`
/// would read a stale accumulator.
#[test]
fn an_arm_skips_a_fold_and_the_fold_it_reads() {
    let k = x()
        .lt(&c(THRESHOLD))
        .select(&reader_of(&fold_of(&x())), &c(0.0))
        .add(&y());
    let want = |x: f64, y: f64| {
        let arm = if x < THRESHOLD {
            reader_ref(fold_ref(x))
        } else {
            0.0
        };
        arm + y
    };
    check("two loops in one arm", &k, want);
    check_at("two loops in one arm", RAGGED_WIDTH, &k, want);
}

/// `select(X ≥ T, Σ_{j<4} |Σ_{m<16} |F(X) − 8j − m| − 100j|, 0) + Y`: as
/// above, with `F`'s only reader the fold nested in the arm's fold, two
/// scopes below the one `F` opens in.
#[test]
fn an_arm_skips_a_nested_fold_and_the_fold_it_reads() {
    let f = fold_of(&x());
    let arm = Kernel::sum_over(OUTER_TRIPS, |j| {
        Kernel::sum_over(READER_TRIPS, |m| f.sub(&j.mul(&c(INNER_STEP))).sub(m).abs())
            .sub(&j.mul(&c(READER_STEP)))
            .abs()
    });
    let k = x().ge(&c(THRESHOLD)).select(&arm, &c(0.0)).add(&y());
    check("nested reader in the arm", &k, |x, y| {
        let f = fold_ref(x);
        let arm: f64 = (0..OUTER_TRIPS)
            .map(f64::from)
            .map(|j| {
                let inner: f64 = (0..READER_TRIPS)
                    .map(|m| (f - INNER_STEP * j - f64::from(m)).abs())
                    .sum();
                (inner - READER_STEP * j).abs()
            })
            .sum();
        (if x >= THRESHOLD { arm } else { 0.0 }) + y
    });
}

/// `select(∃j. X > 10 + 5j, F(X), 0) + Y`: the mask is a loop too, which
/// must run on every batch — the branch reads its result.
#[test]
fn an_arm_that_is_a_fold_under_a_mask_that_is_a_fold() {
    let mask = Kernel::any_over(MASK_TRIPS, |j| {
        x().gt(&j.mul(&c(MASK_STEP)).add(&c(MASK_BASE)))
    });
    let k = mask.select(&fold_of(&x()), &c(0.0)).add(&y());
    check("mask fold", &k, |x, y| {
        let any = (0..MASK_TRIPS).any(|j| x > MASK_BASE + MASK_STEP * f64::from(j));
        (if any { fold_ref(x) } else { 0.0 }) + y
    });
}

/// `select(X < T, F, 0) + select(X > 40, min_k |F − 100k|, 0)` — the glyph's
/// shape, where a winding fold is read both by its own select's arm and by
/// the distance fold in the other's. `F` is owned by neither arm and runs on
/// every batch; the second arm's loop is skipped where its mask is uniformly
/// false, and still reads this batch's `F` where it runs.
#[test]
fn a_fold_read_across_two_selects() {
    let f = fold_of(&x());
    let nearest = Kernel::min_over(READER_TRIPS, |k| f.sub(&k.mul(&c(READER_STEP))).abs());
    let k = x()
        .lt(&c(THRESHOLD))
        .select(&f, &c(0.0))
        .add(&x().gt(&c(FAR_THRESHOLD)).select(&nearest, &c(0.0)));
    let want = |x: f64, _: f64| {
        let f = fold_ref(x);
        let nearest = (0..READER_TRIPS)
            .map(|k| (f - READER_STEP * f64::from(k)).abs())
            .fold(f64::INFINITY, f64::min);
        (if x < THRESHOLD { f } else { 0.0 }) + (if x > FAR_THRESHOLD { nearest } else { 0.0 })
    };
    check("fold read across two selects", &k, want);
    check_at("fold read across two selects", RAGGED_WIDTH, &k, want);
}

// ───── priced past the bound, and still refused by ownership ─────

/// `select(X < T, 0, F(X + Y)) + Y`: the loop reads `X + Y`, which nothing
/// outside it reads. Clustering gathers `X + Y` into the arm, but the batch
/// scope parks it for the loop, and a parked value is pinned outside every
/// arm (no arm may own a root), so it splits the run.
#[test]
fn an_arm_that_is_a_fold_reading_a_value_only_it_reads() {
    let s = x().add(&y());
    let k = x()
        .lt(&c(THRESHOLD))
        .select(&c(0.0), &fold_of(&s))
        .add(&y());
    check("fold reading a parked value", &k, |x, y| {
        let arm = if x < THRESHOLD { 0.0 } else { fold_ref(x + y) };
        arm + y
    });
}

/// `Σ_{j<4} select(X + j < T, F(X + j), 0) + Y`: the select is in the outer
/// fold's own body. The inner loop's schedule still carries the lane and
/// coordinate leaves `X + j` was built from before it was parked, and a leaf
/// the loop's schedule shares is a read of it (`guards::FoldReads`), so the
/// arm owns those leaves too — scheduled ahead of the mask, where no branch
/// after the mask can span them.
#[test]
fn an_arm_that_is_a_fold_in_a_folds_body() {
    let k = Kernel::sum_over(OUTER_TRIPS, |j| {
        let xj = x().add(j);
        xj.lt(&c(THRESHOLD)).select(&fold_of(&xj), &c(0.0))
    })
    .add(&y());
    check("guard in a fold's body", &k, |x, y| {
        let sum: f64 = (0..OUTER_TRIPS)
            .map(|j| x + f64::from(j))
            .map(|xj| if xj < THRESHOLD { fold_ref(xj) } else { 0.0 })
            .sum();
        sum + y
    });
}

// ────────────────────────── controls ──────────────────────────

/// `F(X) + Y`: the same loop, no select.
#[test]
fn control_the_fold_without_the_select() {
    let k = fold_of(&x()).add(&y());
    check("no select", &k, |x, y| fold_ref(x) + y);
}
