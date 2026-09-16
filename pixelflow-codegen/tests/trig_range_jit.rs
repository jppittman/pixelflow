//! `sin`/`cos`/`tan` range and domain, asserted on emitted machine code.
//!
//! Backlog **C1** (`docs/BACKLOG.md`, "Correctness and CI") and CLAUDE.md's
//! "Precision is on the table; range is not": `pixelflow-ir/tests/trig_range.rs`
//! asserted this property through the scalar interpreter (`eval_scalar`) and
//! was deleted with it (commit `077c1641`, "a glyph is two folds over one
//! table"). The property did not change — an out-of-range `sin` still ships
//! green with nothing to catch it — so this rebuilds the claim against the
//! thing that actually ships: a JIT-compiled kernel, called through
//! `pixelflow_codegen::CompiledKernel`, the same route
//! `transcendental_jit.rs` uses.
//!
//! The property, exactly as CLAUDE.md states it: **range is a hard property,
//! asserted with no tolerance, while accuracy is a tunable.**
//! `sin`/`cos` return values in `[-1, 1]` for every input in the documented
//! domain `|x| < TRIG_DOMAIN` (2²⁰, `pixelflow_ir::passes::TRIG_DOMAIN`);
//! outside it they return NaN, never a clamp. `tan` has no bounded range, so
//! its claim is only the NaN-outside-domain half, plus (like `sin`/`cos`)
//! never NaN *inside* the domain. This file does not touch accuracy,
//! periodicity, `atan`/`atan2`/`asin`/`acos`, `sin(-0.0)`'s sign, or the
//! log/pow family — those are separate backlog rows and separate claims of
//! the deleted file; the three claims rebuilt here are
//! `sin_and_cos_never_leave_unit_range`, `sin_propagates_nonfinite_arguments`
//! and `sin_is_nan_outside_the_domain`, extended to `tan`.
//!
//! # This is a bound, not a same-form check
//!
//! Every assertion below compares a JIT result against `in_domain`, a
//! two-line predicate over the *input* — finite and `|x| < TRIG_DOMAIN` —
//! matching the `Lt`/`Select` guard `expand_sin_phase` (`pixelflow-ir/src/
//! passes.rs`) builds into the expansion itself. It does not evaluate
//! `Sin`/`Cos`/`Tan`/`Select`/`Lt` a second time through any oracle, so a bug
//! shared by every tier (the failure mode that let the original defect ship,
//! per CLAUDE.md) cannot cancel out here the way a same-form differential
//! check would let it.
//!
//! # Sampling
//!
//! Dense and adversarial, roughly 100,000 points total, all evaluated against
//! `sin`, `cos` and `tan`:
//!
//! - a uniform linear sweep across the whole domain (tens of thousands of
//!   points);
//! - every f32 binade from the smallest subnormal (2⁻¹⁴⁹) to `f32::MAX`
//!   (2¹²⁷), both inside and outside the domain — the linear sweep alone
//!   leaves the domain's lower magnitudes essentially unsampled;
//! - points at and one ULP either side of every ~133rd multiple of π/2
//!   across the *entire* valid range of `k` (not just small `k`), because
//!   reduction error is sharpest at quadrant boundaries;
//! - the domain boundary `±TRIG_DOMAIN`, approached from both inside and
//!   outside, at several distances;
//! - `±0.0` and explicit subnormals;
//! - `f32::MAX`, `f32::MIN`, and the specific magnitudes the original defect
//!   report and measurement used (`1.4e7`, `1.72e7`, `8.64e8`, `2.61e13`,
//!   `1e30`);
//! - `±inf` and `NaN`;
//! - a log-uniform random sweep strictly outside the domain, up to
//!   `f32::MAX`.

use pixelflow_codegen::jit_cache;
use pixelflow_ir::Kernel;
use pixelflow_ir::passes::TRIG_DOMAIN;

const LANES: usize = pixelflow_codegen::JIT_VECTOR_BYTES / 4;

fn eval_points_1d(jit: &pixelflow_codegen::CompiledKernel, inputs: &[f32]) -> Vec<f32> {
    let mut outputs = Vec::with_capacity(inputs.len());
    for chunk in inputs.chunks(LANES) {
        let mut xs = [0.0f32; LANES];
        for (i, &x) in chunk.iter().enumerate() {
            xs[i] = x;
        }
        let res = unsafe {
            jit.call(pixelflow_codegen::Point4::new(
                xs,
                [0.0; LANES],
                [0.0; LANES],
                [0.0; LANES],
            ))
        };
        outputs.extend_from_slice(&res[..chunk.len()]);
    }
    outputs
}

/// Reproducible LCG — a fixed seed keeps a failure reproducible from the
/// printed argument alone.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 11
    }
    fn unit(&mut self) -> f64 {
        (self.next_u64() % (1 << 52)) as f64 / (1u64 << 52) as f64
    }
    /// A signed magnitude, log-uniform over `[lo, hi]` — linear-uniform would
    /// put essentially no samples below the last decade of `hi`.
    fn log_uniform(&mut self, lo: f64, hi: f64) -> f32 {
        let (log_lo, log_hi) = (lo.log2(), hi.log2());
        let mag = (2.0f64).powf(log_lo + self.unit() * (log_hi - log_lo));
        let signed = if self.next_u64() & 1 == 0 { mag } else { -mag };
        signed as f32
    }
}

/// One f32 ULP toward `+inf` (`+0.0`/`-0.0` both step to the smallest
/// positive subnormal). Not a general-purpose `nextafter` — it only needs to
/// place an adversarial point next to a quadrant boundary, and NaN/`+inf`
/// inputs never reach it here.
fn next_up(x: f32) -> f32 {
    if x == 0.0 {
        return f32::from_bits(1);
    }
    let bits = x.to_bits();
    if x > 0.0 {
        f32::from_bits(bits + 1)
    } else {
        f32::from_bits(bits - 1)
    }
}

fn next_down(x: f32) -> f32 {
    -next_up(-x)
}

/// The domain contract itself: finite and strictly inside `TRIG_DOMAIN`.
///
/// This mirrors the *predicate* `expand_sin_phase` guards on (`abs_x < limit`
/// over `Lt`, which is false for NaN and for `±inf`), not the expansion that
/// predicate feeds — the property under test, stated once, independent of the
/// code being checked.
fn in_domain(x: f32) -> bool {
    x.is_finite() && x.abs() < TRIG_DOMAIN
}

fn describe(x: f32) -> String {
    format!("{x:e} (bits {:#010x})", x.to_bits())
}

/// Every point this file checks `sin`, `cos` and `tan` against.
fn build_samples() -> Vec<f32> {
    let mut xs = Vec::new();
    let mut rng = Rng(0x005e_ed12_34c0_ffee);

    // 1. Uniform linear sweep across the whole domain.
    const N_SWEEP: usize = 50_000;
    for i in 0..N_SWEEP {
        let t = (i as f64 + 0.5) / N_SWEEP as f64; // strictly inside (0, 1)
        let x = -(TRIG_DOMAIN as f64) + t * 2.0 * TRIG_DOMAIN as f64;
        xs.push(x as f32);
    }

    // 2. Every f32 binade, subnormal through f32::MAX, both inside and
    //    outside the domain — the linear sweep above has essentially zero
    //    density below ~1.0, since it is uniform over a range spanning 2^41.
    for e in -149i32..=127 {
        for _ in 0..20 {
            let m = (2.0f64).powi(e) * (1.0 + rng.unit());
            xs.push(m as f32);
            xs.push(-(m as f32));
        }
        let edge = (2.0f64).powi(e) as f32;
        xs.push(edge);
        xs.push(-edge);
    }

    // 3. Near every ~133rd multiple of π/2, spanning the *entire* valid `k`
    //    range (not just small k) — quadrant boundaries are where reduction
    //    error is sharpest. Literally every multiple is ~1.3M points; this
    //    stride still covers every magnitude decade of k.
    let half_pi = core::f64::consts::FRAC_PI_2;
    let k_max = (TRIG_DOMAIN as f64 / half_pi).floor() as i64;
    const K_SAMPLES: i64 = 10_000;
    let stride = (2 * k_max / K_SAMPLES).max(1);
    let mut k = -k_max;
    while k <= k_max {
        let center = (k as f64 * half_pi) as f32;
        xs.push(center);
        xs.push(next_up(center));
        xs.push(next_down(center));
        k += stride;
    }
    xs.push(0.0);
    xs.push((k_max as f64 * half_pi) as f32);
    xs.push(-((k_max as f64) * half_pi) as f32);

    // 4. The domain boundary, approached from both sides, both signs.
    for &frac in &[1e-7_f64, 1e-6, 1e-4, 1e-2, 0.1, 0.5] {
        let inside = (TRIG_DOMAIN as f64 * (1.0 - frac)) as f32;
        let outside = (TRIG_DOMAIN as f64 * (1.0 + frac)) as f32;
        xs.extend_from_slice(&[inside, -inside, outside, -outside]);
    }
    xs.push(TRIG_DOMAIN);
    xs.push(-TRIG_DOMAIN);
    xs.push(next_down(TRIG_DOMAIN)); // just inside
    xs.push(next_up(TRIG_DOMAIN)); // just outside
    xs.push(next_up(-TRIG_DOMAIN)); // just inside (negative side)
    xs.push(next_down(-TRIG_DOMAIN)); // just outside (negative side)

    // 5. Signed zero and explicit subnormals.
    xs.extend_from_slice(&[
        0.0,
        -0.0,
        f32::from_bits(1), // smallest positive subnormal
        -f32::from_bits(1),
        f32::from_bits(0x007f_ffff), // largest subnormal
        -f32::from_bits(0x007f_ffff),
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
    ]);

    // 6. Largest finite floats and the original defect's own reported and
    //    measured magnitudes (all well outside the domain).
    xs.extend_from_slice(&[
        f32::MAX,
        f32::MIN,
        1.4e7, // measured onset of |sin| > 1
        -1.4e7,
        1.72e7, // first reported failure
        -1.72e7,
        8.64e8, // worst reported value
        -8.64e8,
        2.61e13, // measured inf
        -2.61e13,
        1e30,
        -1e30,
    ]);

    // 7. Non-finite inputs.
    xs.extend_from_slice(&[f32::INFINITY, f32::NEG_INFINITY, f32::NAN]);

    // 8. Log-uniform random sweep strictly outside the domain.
    for _ in 0..10_000 {
        xs.push(rng.log_uniform(TRIG_DOMAIN as f64, f32::MAX as f64));
    }

    xs
}

/// A function's own range claim: `sin`/`cos` are bounded to `[-1, 1]`, `tan`
/// has no such bound (CLAUDE.md: "`tan` is unbounded, so its range claim is
/// only the NaN-outside-domain half").
#[derive(Clone, Copy)]
enum Bound {
    Unit,
    Unbounded,
}

/// Assert the domain contract for one function's `(input, output)` pairs:
/// NaN outside the domain, never NaN inside it, and (for `Bound::Unit`)
/// `|output| <= 1.0` inside it — asserted with no tolerance at all, per
/// CLAUDE.md's "range is a hard property, asserted with no tolerance".
fn check_domain_contract(name: &str, bound: Bound, samples: &[(f32, f32)]) {
    for &(x, got) in samples {
        if in_domain(x) {
            assert!(
                !got.is_nan(),
                "{name}({}) = NaN — every input with |x| < TRIG_DOMAIN must \
                 produce a value, never NaN",
                describe(x),
            );
            if let Bound::Unit = bound {
                assert!(
                    got.abs() <= 1.0,
                    "{name}({}) = {got:e} (bits {:#010x}) — outside [-1, 1], \
                     asserted with no tolerance",
                    describe(x),
                    got.to_bits(),
                );
            }
        } else {
            assert!(
                got.is_nan(),
                "{name}({}) = {got:e} (bits {:#010x}) — should be NaN \
                 outside the documented domain |x| < TRIG_DOMAIN",
                describe(x),
                got.to_bits(),
            );
        }
    }
}

/// The headline property, through emitted machine code: `sin`/`cos` never
/// leave `[-1, 1]` inside the domain, `tan` is never NaN inside it, and all
/// three are NaN outside it (or for non-finite/NaN input) — asserted with no
/// tolerance, against ~100,000 points per function.
#[test]
fn sin_cos_tan_domain_contract_on_jit() {
    let samples = build_samples();
    assert!(
        samples.len() >= 90_000,
        "sample set shrank to {} — this test's claim is only as strong as \
         its coverage",
        samples.len(),
    );

    for (name, bound, k) in [
        ("sin", Bound::Unit, Kernel::x().sin()),
        ("cos", Bound::Unit, Kernel::x().cos()),
        ("tan", Bound::Unbounded, Kernel::x().tan()),
    ] {
        let jit = jit_cache::compile(&k, pixelflow_ir::LatticeShape::POINT)
            .unwrap_or_else(|e| panic!("{name}: kernel failed to compile on this backend: {e}"))
            .kernel;
        let outputs = eval_points_1d(&jit, &samples);
        assert_eq!(outputs.len(), samples.len());
        let pairs: Vec<(f32, f32)> = samples.iter().copied().zip(outputs).collect();
        check_domain_contract(name, bound, &pairs);
    }
}
