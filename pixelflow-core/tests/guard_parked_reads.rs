//! `guard_sibling_fold.rs`'s two hazards again, with the fold's body reading a
//! value an enclosing scope *parks* rather than one it computes in place: a
//! constant (computed once per call and read from its park), a table read
//! whose address the lane binder does not reach (one scalar load broadcast,
//! through a context pointer the per-call scope loads once), and a uniform.
//! Each is a shared `ValueId` in the fold's schedule — a `Const(0.0)`
//! placeholder for a vector, the pointer's own `Context` op for a pointer —
//! so `guards::FoldReads` must count it as a read of the fold's `Reduce` def,
//! and the guard analysis must count a pointer operand as a read at all.
//!
//! The selects sit at every scope a select can: the batch (a mask on `X`),
//! the row (a mask on `Y` alone, uniform across the row) and the call (a mask
//! on a uniform, one arm for the whole call). Sections 1–3 are the sibling
//! and clustering hazards at each; section 4 is an arm that is a table fold
//! and nothing else, which its trips price past the branch bound, so the loop
//! and the broadcasts in it are what a uniform mask skips.
//!
//! Both fixes are load-bearing here. With the guard analysis reading only
//! vector operands, four of the per-call kernels fail to compile (a pointer
//! sunk past its reader panics the allocator); with a fold's reads dropped
//! from `FoldReads`, the sibling kernels at every scope come back wrong and
//! the per-call one faults.
//!
//! Every kernel is checked texel by texel against a plain-`f64` reference,
//! never a pixelflow evaluator. `PIXELFLOW_GUARD_TELEMETRY=1` prints, per
//! scope, which selects were guarded.

#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use pixelflow_core::{DiscreteManifold, Kernel, Lattice, Manifold, Uniform};

/// Wide enough to span several SIMD batches at every supported width.
const WIDTH: usize = 64;
/// A width no lane count divides: every row ends in a remainder batch.
const RAGGED_WIDTH: usize = 61;
/// More than one row, and an even split for the per-row mask below.
const HEIGHT: usize = 4;
/// Trip count of every fold. Even, so the optimizer halves the fold rather
/// than peeling a term out of it, and the fold stays a bare `Reduce`.
const TRIPS: u32 = 48;
/// `D`'s per-trip offset step: sweeps the folds' range so `D` is not affine
/// in what it reads.
const D_STEP: f64 = 40.0;
/// The phase scale inside the arms' `sin`: exclusive work past the 16-cycle
/// bound even before the fold's own price.
const PHASE: f64 = 0.1;
/// The batch mask's threshold on `X`.
const THRESHOLD: f64 = 20.0;
/// The row mask's threshold on `Y`: rows 0 and 1 take the arm, 2 and 3 skip.
const ROW_THRESHOLD: f64 = 2.0;
/// Relative tolerance: `f32` summation of ~50 terms, and `sin`'s polynomial.
const REL_TOL: f64 = 1e-4;

// ─────────────────────────── tables ───────────────────────────

/// `t[j]`, one row of [`TRIPS`] samples, exact in `f32` and in no order a
/// wrong index could hide behind.
fn t_value(j: u32) -> f64 {
    0.5 * f64::from(j) + f64::from(j % 3)
}

/// `r[y]`, one sample per row, exact in `f32`.
fn r_value(y: usize) -> f64 {
    [3.0, -1.5, 7.25, 0.5][y]
}

/// The table `t` as a kernel reading `t[x]` at row 0, its data carried.
/// Build it once per kernel and share it: each call mints a buffer of its
/// own, and a buffer is one context pointer.
fn t_table() -> Kernel {
    t_table_of(TRIPS)
}

/// [`t_table`] with `len` samples.
fn t_table_of(len: u32) -> Kernel {
    let data = (0..len).map(|j| t_value(j) as f32).collect();
    DiscreteManifold::new(data, len as usize, 1).kernel()
}

/// The table `r` as a kernel reading `r[x]` at row 0, its data carried.
fn r_table() -> Kernel {
    let data = (0..HEIGHT).map(|y| r_value(y) as f32).collect();
    DiscreteManifold::new(data, HEIGHT, 1).kernel()
}

/// `table[i]` at row 0.
fn read(table: &Kernel, i: &Kernel) -> Kernel {
    table.at(i, &c(0.0))
}

/// `t[X/2]`, per lane: a gather through the same pointer a fold's
/// broadcasts use.
fn t_at_half_x(t: &Kernel) -> Kernel {
    read(t, &x().mul(&c(0.5)))
}

fn t_at_half_x_ref(x: f64) -> f64 {
    t_value(((x * 0.5).floor() as u32).min(TRIPS - 1))
}

// ─────────────────────────── kernels ───────────────────────────

fn x() -> Kernel {
    Kernel::x()
}

fn c(v: f64) -> Kernel {
    Kernel::constant(v as f32)
}

/// `Σ_{j<N} |s − t[j]|`, reading `s` and the table through its pointer:
/// the lane binder does not reach `t[j]`'s address, so it is one scalar load
/// broadcast per trip.
fn w_over_table(t: &Kernel, s: &Kernel) -> Kernel {
    w_over_table_of(TRIPS, t, s)
}

/// [`w_over_table`] over `trips` trips.
fn w_over_table_of(trips: u32, t: &Kernel, s: &Kernel) -> Kernel {
    Kernel::sum_over(trips, |j| s.sub(&read(t, j)).abs())
}

/// `D = Σ_{k<N} |w − 40k|` — a sibling fold whose body reads `w`.
fn sibling_of(w: &Kernel) -> Kernel {
    Kernel::sum_over(TRIPS, |k| w.sub(&k.mul(&c(D_STEP))).abs())
}

/// `w·sin(PHASE·s)` — the arm's work.
fn heavy_of(w: &Kernel, s: &Kernel) -> Kernel {
    w.mul(&s.mul(&c(PHASE)).sin())
}

// ────────────────────────── reference ──────────────────────────

fn w_over_table_ref(s: f64) -> f64 {
    w_over_table_ref_of(TRIPS, s)
}

fn w_over_table_ref_of(trips: u32, s: f64) -> f64 {
    (0..trips).map(|j| (s - t_value(j)).abs()).sum()
}

fn sibling_ref(w: f64) -> f64 {
    (0..TRIPS).map(|k| (w - D_STEP * f64::from(k)).abs()).sum()
}

fn heavy_ref(w: f64, s: f64) -> f64 {
    w * (s * PHASE).sin()
}

/// Compare every texel of `got` (row-major, `width` wide) with `want(x, y)`,
/// reporting every mismatch at once.
fn compare(name: &str, width: usize, got: &[f32], want: impl Fn(f64, f64) -> f64) {
    assert_eq!(got.len(), width * HEIGHT);
    let mut bad = Vec::new();
    for row in 0..HEIGHT {
        for col in 0..width {
            let got = f64::from(got[row * width + col]);
            let want = want(col as f64, row as f64);
            if (got - want).abs() > REL_TOL * want.abs().max(1.0) {
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

/// Bake `kernel` over `width × HEIGHT` and compare it with `want`.
fn check_at(name: &str, width: usize, kernel: &Kernel, want: impl Fn(f64, f64) -> f64) {
    let baked = Lattice::frame(width, HEIGHT).bake(kernel);
    compare(name, width, baked.buffer(), want);
}

fn check(name: &str, kernel: &Kernel, want: impl Fn(f64, f64) -> f64) {
    check_at(name, WIDTH, kernel, want);
}

/// Compile `kernel` once and collapse it once per value of `u`, in order, so
/// a call that skips an arm follows one that ran it: a park or accumulator
/// the skipped call failed to write still holds the previous call's value.
fn check_per_call(
    name: &str,
    kernel: &Kernel,
    u: Uniform,
    values: &[f64],
    want: impl Fn(f64, f64, f64) -> f64,
) {
    let program = Manifold::compile(kernel, [WIDTH as u32, HEIGHT as u32]);
    let mut block = program.block();
    for &v in values {
        block.set(u, v as f32).expect("u is this kernel's argument");
        let bound = program.bind(&[]).with_uniforms(&block);
        let baked = Lattice::frame(WIDTH, HEIGHT).collapse(&bound);
        compare(
            &format!("{name}, u = {v}"),
            WIDTH,
            baked.buffer(),
            |x, y| want(x, y, v),
        );
    }
}

// ─── 1. the batch scope: a fold body reads through a parked pointer ───

/// `select(X < T, W·sin, 0) + D` with `W = Σ_j |X − t[j]|`: `W`'s body reads
/// the table through a pointer the per-call scope parks, `D`'s body reads
/// `W`'s accumulator. A skipped arm may not take `W`'s loop with it.
#[test]
fn a_guard_keeps_a_table_fold_a_sibling_reads() {
    let w = w_over_table(&t_table(), &x());
    let k = x()
        .lt(&c(THRESHOLD))
        .select(&heavy_of(&w, &x()), &c(0.0))
        .add(&sibling_of(&w));
    check("table fold, sibling", &k, |x, _| {
        let w = w_over_table_ref(x);
        let arm = if x < THRESHOLD { heavy_ref(w, x) } else { 0.0 };
        arm + sibling_ref(w)
    });
}

/// The same, with the heavy arm the false one, a remainder batch, and the
/// table read per lane outside the select through the same pointer.
#[test]
fn a_guard_keeps_a_table_fold_a_sibling_reads_heavy_false_arm_ragged() {
    let t = t_table();
    let w = w_over_table(&t, &x());
    let k = x()
        .lt(&c(THRESHOLD))
        .select(&c(0.0), &heavy_of(&w, &x()))
        .add(&sibling_of(&w))
        .add(&t_at_half_x(&t));
    check_at("table fold, false arm, ragged", RAGGED_WIDTH, &k, |x, _| {
        let w = w_over_table_ref(x);
        let arm = if x < THRESHOLD { 0.0 } else { heavy_ref(w, x) };
        arm + sibling_ref(w) + t_at_half_x_ref(x)
    });
}

/// `select(X < T, W·sin, 0)` with `W = Σ_j |X + r[Y] − t[j]|`: `X + r[Y]` is
/// read only by `W`'s body, and `r[Y]` is a per-row broadcast the row scope
/// parks. Clustering may not sink either past the loop.
#[test]
fn clustering_keeps_a_broadcast_input_ahead_of_the_fold() {
    let s = x().add(&read(&r_table(), &Kernel::y()));
    let w = w_over_table(&t_table(), &s);
    let k = x().lt(&c(THRESHOLD)).select(&heavy_of(&w, &x()), &c(0.0));
    check("broadcast input", &k, |x, y| {
        if x < THRESHOLD {
            heavy_ref(w_over_table_ref(x + r_value(y as usize)), x)
        } else {
            0.0
        }
    });
}

/// `select(X < T, W·sin, 0) + D` with `W = Σ_j |X − r[Y]·j|`: the per-row
/// broadcast `r[Y]` is a placeholder in `W`'s body, read every trip, and
/// nothing in the batch scope reads it but that body.
#[test]
fn a_guard_keeps_a_fold_reading_a_per_row_park() {
    let r = read(&r_table(), &Kernel::y());
    let w = Kernel::sum_over(TRIPS, |j| x().sub(&r.mul(j)).abs());
    let k = x()
        .lt(&c(THRESHOLD))
        .select(&heavy_of(&w, &x()), &c(0.0))
        .add(&sibling_of(&w));
    check("per-row park", &k, |x, y| {
        let r = r_value(y as usize);
        let w: f64 = (0..TRIPS).map(|j| (x - r * f64::from(j)).abs()).sum();
        let arm = if x < THRESHOLD { heavy_ref(w, x) } else { 0.0 };
        arm + sibling_ref(w)
    });
}

/// How many distinct constants [`poly_of`] reads: more than the register
/// file carries alongside the scope's own work, so some are parked in a
/// slot and every trip, and the arm, reload them.
const COEFFS: usize = 14;

/// A constant per coefficient, distinct and exact in `f32`.
fn coeff(i: usize) -> f64 {
    (i as f64 + 1.0) * 0.0625 - if i.is_multiple_of(2) { 0.5 } else { 0.0 }
}

/// `Σ_i coeff(i) · z^i`, Horner — [`COEFFS`] constants read per evaluation.
fn poly_of(z: &Kernel) -> Kernel {
    (0..COEFFS)
        .rev()
        .fold(c(0.0), |acc, i| acc.mul(z).add(&c(coeff(i))))
}

fn poly_ref(z: f64) -> f64 {
    (0..COEFFS).rev().fold(0.0, |acc, i| acc * z + coeff(i))
}

/// Trip count of the constant-pressure fold: the polynomial per trip is the
/// cost, so fewer trips keep the reference's rounding small.
const POLY_TRIPS: u32 = 8;
/// The polynomials' argument scales, keeping every argument inside `[0, 1)`
/// and each polynomial a value of its own.
const POLY_SCALE: f64 = 1.0 / 128.0;
const ARM_SCALE: f64 = 1.0 / 256.0;
const TAIL_SCALE: f64 = 1.0 / 512.0;

/// `select(X < T, W·p(X/256), 0) + D + p(X/512)` with
/// `W = Σ_j p(|X − j|/128)`: fourteen constants, parked by the per-call
/// scope, read by `W`'s body every trip, by the arm, and after the select —
/// some from a slot, reloaded inside the skipped range.
#[test]
fn a_guard_keeps_a_fold_reading_many_parked_constants() {
    let w = Kernel::sum_over(POLY_TRIPS, |j| {
        poly_of(&x().sub(j).abs().mul(&c(POLY_SCALE)))
    });
    let arm = w.mul(&poly_of(&x().mul(&c(ARM_SCALE))));
    let k = x()
        .lt(&c(THRESHOLD))
        .select(&arm, &c(0.0))
        .add(&sibling_of(&w))
        .add(&poly_of(&x().mul(&c(TAIL_SCALE))));
    for (name, width) in [("many parked constants", WIDTH), ("ragged", RAGGED_WIDTH)] {
        check_at(name, width, &k, |x, _| {
            let w: f64 = (0..POLY_TRIPS)
                .map(|j| poly_ref((x - f64::from(j)).abs() * POLY_SCALE))
                .sum();
            let arm = if x < THRESHOLD {
                w * poly_ref(x * ARM_SCALE)
            } else {
                0.0
            };
            arm + sibling_ref(w) + poly_ref(x * TAIL_SCALE)
        });
    }
}

// ─── 2. the row scope: a select on Y alone ───

/// `sin`'s argument for the row arm, `(Y + 10)·PHASE`: `Y` alone, so the
/// row scope computes it.
fn row_phase() -> Kernel {
    Kernel::y().add(&c(1.0 / PHASE))
}

fn row_phase_ref(y: f64) -> f64 {
    y + 1.0 / PHASE
}

/// `select(Y < 2, R·sin, 0) + Σ_k |R − 40k| + X` with
/// `R = Σ_j |r[Y] − t[j]|`: `R` is a per-row fold over two broadcasts, and
/// the select and the sibling are per-row values the row scope computes and
/// parks for the batches. On rows 2 and 3 the row's arm is skipped whole.
#[test]
fn a_row_guard_keeps_a_per_row_fold_a_sibling_reads() {
    let r = w_over_table(&t_table(), &read(&r_table(), &Kernel::y()));
    let k = Kernel::y()
        .lt(&c(ROW_THRESHOLD))
        .select(&heavy_of(&r, &row_phase()), &c(0.0))
        .add(&sibling_of(&r))
        .add(&x());
    check("row guard", &k, |x, y| {
        let r = w_over_table_ref(r_value(y as usize));
        let arm = if y < ROW_THRESHOLD {
            heavy_ref(r, row_phase_ref(y))
        } else {
            0.0
        };
        arm + sibling_ref(r) + x
    });
}

/// `select(Y < 2, R·sin, 0) + t[X/2]` with `R` as above and nothing else
/// reading it: the arm owns the fold, and the fold reads two pointers the
/// call parks — one of them read per lane by the batches too. The loop and
/// its broadcasts are skipped on rows 2 and 3; the pointers may not be.
#[test]
fn a_row_guard_skips_a_fold_over_parked_pointers() {
    let t = t_table();
    let r = w_over_table(&t, &read(&r_table(), &Kernel::y()));
    let k = Kernel::y()
        .lt(&c(ROW_THRESHOLD))
        .select(&heavy_of(&r, &row_phase()), &c(0.0))
        .add(&t_at_half_x(&t));
    for (name, width) in [("row guard over pointers", WIDTH), ("ragged", RAGGED_WIDTH)] {
        check_at(name, width, &k, |x, y| {
            let arm = if y < ROW_THRESHOLD {
                heavy_ref(w_over_table_ref(r_value(y as usize)), row_phase_ref(y))
            } else {
                0.0
            };
            arm + t_at_half_x_ref(x)
        });
    }
}

// ─── 3. the call scope: a select on a uniform ───

/// The values `u` takes, in order: the arm runs, is skipped, runs again,
/// is skipped again — a skipped call always follows one that ran.
const U_VALUES: [f64; 4] = [3.0, -2.0, 5.5, -0.25];

/// `select(u > 0, F·sin(u), 0) + G + X` with `F = Σ_j |u − t[j]|` and
/// `G = Σ_k |F − 40k|`: `F` and `G` are per-call folds, `F`'s body reads the
/// table through the call's pointer, and `G` reads `F`'s accumulator. A call
/// with `u ≤ 0` may not skip `F`'s loop.
#[test]
fn a_call_guard_keeps_a_per_call_fold_a_sibling_reads() {
    let u = Uniform::new(1.0);
    let f = w_over_table(&t_table(), &u.kernel());
    let k = u
        .kernel()
        .gt(&c(0.0))
        .select(&heavy_of(&f, &u.kernel()), &c(0.0))
        .add(&sibling_of(&f))
        .add(&x());
    check_per_call("call guard, sibling", &k, u, &U_VALUES, |x, _, u| {
        let f = w_over_table_ref(u);
        let arm = if u > 0.0 { heavy_ref(f, u) } else { 0.0 };
        arm + sibling_ref(f) + x
    });
}

/// The per-call fold's scale and step below.
const U_SCALE: f64 = 0.37;
const U_STEP: f64 = 1.75;

/// `select(u > 0, F·sin(u), 0) + t[X/2] + 1.75·X` with
/// `F = Σ_j |0.37·u − 1.75·j + t[j]|`: the arm owns `F`, and `F` reads a
/// constant and a pointer the batches read too — both roots of the call
/// scope, which a skipped arm may not take with it.
#[test]
fn a_call_guard_skips_a_fold_over_roots_the_batches_read() {
    let u = Uniform::new(1.0);
    let t = t_table();
    let f = Kernel::sum_over(TRIPS, |j| {
        u.kernel()
            .mul(&c(U_SCALE))
            .sub(&j.mul(&c(U_STEP)))
            .add(&read(&t, j))
            .abs()
    });
    let k = u
        .kernel()
        .gt(&c(0.0))
        .select(&heavy_of(&f, &u.kernel()), &c(0.0))
        .add(&t_at_half_x(&t))
        .add(&x().mul(&c(U_STEP)));
    check_per_call("call guard over roots", &k, u, &U_VALUES, |x, _, u| {
        let s = u * U_SCALE;
        let f: f64 = (0..TRIPS)
            .map(|j| (s - U_STEP * f64::from(j) + t_value(j)).abs())
            .sum();
        let arm = if u > 0.0 { heavy_ref(f, u) } else { 0.0 };
        arm + t_at_half_x_ref(x) + U_STEP * x
    });
}

/// `select(u > 0, F·sin(u), 0) + (u + 1) + X` with `F = Σ_j |u − t[j]|` and
/// nothing else reading the table: the pointer is read only by the arm's
/// loop, and `u + 1` — read by the root, not the select — makes the arm
/// non-contiguous until clustering gathers it.
#[test]
fn a_call_guard_over_a_fold_that_alone_reads_its_pointer() {
    let u = Uniform::new(1.0);
    let f = w_over_table(&t_table(), &u.kernel());
    let k = u
        .kernel()
        .gt(&c(0.0))
        .select(&heavy_of(&f, &u.kernel()), &c(0.0))
        .add(&u.kernel().add(&c(1.0)))
        .add(&x());
    check_per_call("call guard, lone pointer", &k, u, &U_VALUES, |x, _, u| {
        let arm = if u > 0.0 {
            heavy_ref(w_over_table_ref(u), u)
        } else {
            0.0
        };
        arm + (u + 1.0) + x
    });
}

/// `select(u > 0, F·sin(u), 0) + Σ_k |F + X − t[k]|`: a per-call select
/// whose arm owns a per-call fold, and a batch fold whose body reads that
/// fold's accumulator — two scopes from where `F`'s loop ran — and the
/// table through the same pointer.
#[test]
fn a_call_guard_keeps_a_fold_a_batch_fold_reads() {
    let u = Uniform::new(1.0);
    let t = t_table();
    let f = w_over_table(&t, &u.kernel());
    let reader = w_over_table(&t, &f.add(&x()));
    let k = u
        .kernel()
        .gt(&c(0.0))
        .select(&heavy_of(&f, &u.kernel()), &c(0.0))
        .add(&reader);
    check_per_call("call guard, batch reader", &k, u, &U_VALUES, |x, _, u| {
        let f = w_over_table_ref(u);
        let arm = if u > 0.0 { heavy_ref(f, u) } else { 0.0 };
        arm + w_over_table_ref(f + x)
    });
}

// ─── 4. an arm that is a fold over a parked pointer, at every scope ───
//
// At [`TRIPS`] a per-row or per-call fold may be unrolled into its scope,
// which leaves the arm its broadcasts; [`LONG_TRIPS`] keeps it a loop.

/// `select(X < T, Σ_j |X − t[j]|, 0) + t[X/2]`: the arm is the loop and
/// nothing else, priced by its trips, and guarded; the pointer its
/// broadcasts read is read per lane after the select as well.
#[test]
fn a_batch_arm_that_is_a_table_fold() {
    let t = t_table();
    let k = x()
        .lt(&c(THRESHOLD))
        .select(&w_over_table(&t, &x()), &c(0.0))
        .add(&t_at_half_x(&t));
    for (name, width) in [("batch arm fold", WIDTH), ("ragged", RAGGED_WIDTH)] {
        check_at(name, width, &k, |x, _| {
            let arm = if x < THRESHOLD {
                w_over_table_ref(x)
            } else {
                0.0
            };
            arm + t_at_half_x_ref(x)
        });
    }
}

/// `select(Y < 2, Σ_j |r[Y] − t[j]|, 0) + t[X/2]`: the per-row arm is the
/// loop, over two pointers the call parks.
#[test]
fn a_row_arm_that_is_a_table_fold() {
    let t = t_table();
    let k = Kernel::y()
        .lt(&c(ROW_THRESHOLD))
        .select(&w_over_table(&t, &read(&r_table(), &Kernel::y())), &c(0.0))
        .add(&t_at_half_x(&t));
    check("row arm fold", &k, |x, y| {
        let arm = if y < ROW_THRESHOLD {
            w_over_table_ref(r_value(y as usize))
        } else {
            0.0
        };
        arm + t_at_half_x_ref(x)
    });
}

/// `select(u > 0, Σ_j |u − t[j]|, 0) + t[X/2]`: the per-call arm is the
/// loop; the pointer is a root of the call scope the batches read.
#[test]
fn a_call_arm_that_is_a_table_fold() {
    let u = Uniform::new(1.0);
    let t = t_table();
    let k = u
        .kernel()
        .gt(&c(0.0))
        .select(&w_over_table(&t, &u.kernel()), &c(0.0))
        .add(&t_at_half_x(&t));
    check_per_call("call arm fold", &k, u, &U_VALUES, |x, _, u| {
        let arm = if u > 0.0 { w_over_table_ref(u) } else { 0.0 };
        arm + t_at_half_x_ref(x)
    });
}

/// Trips for a per-row or per-call fold long enough that the optimizer keeps
/// it a loop rather than unrolling it into the scope.
const LONG_TRIPS: u32 = 512;

/// `select(Y < 2, Σ_{j<512} |r[Y] − t[j]|, 0) + t[X/2]`, the long form.
#[test]
fn a_row_arm_that_is_a_long_table_fold() {
    let t = t_table_of(LONG_TRIPS);
    let arm = w_over_table_of(LONG_TRIPS, &t, &read(&r_table(), &Kernel::y()));
    let k = Kernel::y()
        .lt(&c(ROW_THRESHOLD))
        .select(&arm, &c(0.0))
        .add(&t_at_half_x(&t));
    check("row arm long fold", &k, |x, y| {
        let arm = if y < ROW_THRESHOLD {
            w_over_table_ref_of(LONG_TRIPS, r_value(y as usize))
        } else {
            0.0
        };
        arm + t_value((x * 0.5).floor() as u32)
    });
}

/// `select(u > 0, Σ_{j<512} |u − t[j]|, 0) + t[X/2]`, the long form.
#[test]
fn a_call_arm_that_is_a_long_table_fold() {
    let u = Uniform::new(1.0);
    let t = t_table_of(LONG_TRIPS);
    let k = u
        .kernel()
        .gt(&c(0.0))
        .select(&w_over_table_of(LONG_TRIPS, &t, &u.kernel()), &c(0.0))
        .add(&t_at_half_x(&t));
    check_per_call("call arm long fold", &k, u, &U_VALUES, |x, _, u| {
        let arm = if u > 0.0 {
            w_over_table_ref_of(LONG_TRIPS, u)
        } else {
            0.0
        };
        arm + t_value((x * 0.5).floor() as u32)
    });
}
