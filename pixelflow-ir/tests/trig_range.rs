//! Range and accuracy of the trig expansions, through the scalar oracle.
//!
//! `eval_scalar` runs the same `expand_transcendentals` lowering the JIT
//! emitters do, so a defect here is a defect in every tier at once — which is
//! exactly why the argument-reduction bug these tests pin survived so long.
//! Both tiers agreed bit-for-bit on the garbage, so every *same-form*
//! equivalence test passed, and outputs in the 1e2-1e6 range sailed past the
//! `>1e30` "ill-conditioned" filter the corpus quarantine gates use and were
//! admitted as valid training labels.
//!
//! So these compare against an *external* reference (`f64`), not against
//! another run of the same expansion, and they check a property the expansion
//! cannot define away: `|sin| ≤ 1`.

use pixelflow_ir::passes::TRIG_DOMAIN;
use pixelflow_ir::{BindingTable, ExprArena, ExprId, OpKind, eval_scalar};

fn eval1(build: impl FnOnce(&mut ExprArena, ExprId) -> ExprId, x: f32) -> f32 {
    let mut a = ExprArena::new();
    let v = a.push_var(0);
    let root = build(&mut a, v);
    eval_scalar(&a, root, &[x, 0.0], &BindingTable::empty())
}

fn sin_at(x: f32) -> f32 {
    eval1(|a, v| a.push_unary(OpKind::Sin, v), x)
}
fn cos_at(x: f32) -> f32 {
    eval1(|a, v| a.push_unary(OpKind::Cos, v), x)
}
fn tan_at(x: f32) -> f32 {
    eval1(|a, v| a.push_unary(OpKind::Tan, v), x)
}
fn atan_at(x: f32) -> f32 {
    eval1(|a, v| a.push_unary(OpKind::Atan, v), x)
}
fn asin_at(x: f32) -> f32 {
    eval1(|a, v| a.push_unary(OpKind::Asin, v), x)
}
fn acos_at(x: f32) -> f32 {
    eval1(|a, v| a.push_unary(OpKind::Acos, v), x)
}
fn log2_at(x: f32) -> f32 {
    eval1(|a, v| a.push_unary(OpKind::Log2, v), x)
}
fn ln_at(x: f32) -> f32 {
    eval1(|a, v| a.push_unary(OpKind::Ln, v), x)
}
fn log10_at(x: f32) -> f32 {
    eval1(|a, v| a.push_unary(OpKind::Log10, v), x)
}
fn exp_at(x: f32) -> f32 {
    eval1(|a, v| a.push_unary(OpKind::Exp, v), x)
}
fn exp2_at(x: f32) -> f32 {
    eval1(|a, v| a.push_unary(OpKind::Exp2, v), x)
}
fn atan2_at(y: f32, x: f32) -> f32 {
    let mut a = ExprArena::new();
    let vy = a.push_var(0);
    let vx = a.push_var(1);
    let root = a.push_binary(OpKind::Atan2, vy, vx);
    eval_scalar(&a, root, &[y, x], &BindingTable::empty())
}
fn pow_at(a_val: f32, b_val: f32) -> f32 {
    let mut a = ExprArena::new();
    let va = a.push_var(0);
    let vb = a.push_var(1);
    let root = a.push_binary(OpKind::Pow, va, vb);
    eval_scalar(&a, root, &[a_val, b_val], &BindingTable::empty())
}

/// Reproducible LCG — a fixed seed keeps a failure reproducible from the
/// printed argument alone.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 11
    }
    fn unit(&mut self) -> f64 {
        (self.next() % (1 << 52)) as f64 / (1u64 << 52) as f64
    }
    /// A signed argument log-uniform over `[1, hi]`, so every decade of
    /// magnitude gets equal weight.
    fn log_uniform(&mut self, hi: f64) -> f32 {
        let m = (2.0f64).powf(self.unit() * hi.log2()) * (1.0 + self.unit());
        let m = if self.next() & 1 == 0 { m } else { -m };
        m as f32
    }
}

// ===========================================================================
// The headline property: sin and cos are bounded, everywhere, always.
// ===========================================================================

/// `|sin(x)| ≤ 1` and `|cos(x)| ≤ 1` across the whole finite f32 range.
///
/// This is the regression test for the reported defect: `|sin|` first exceeded
/// 1 near `x ≈ 1.4e7` and reached `inf` by `x ≈ 2.6e13`. A value outside
/// `[-1, 1]` is not a precision trade, it is a wrong answer, so this is
/// asserted with no tolerance at all — either the result is in range, or it is
/// NaN because the argument is outside the documented domain.
#[test]
fn sin_and_cos_never_leave_unit_range() {
    let mut rng = Rng(0x5eed_1234);
    // Every binade from 2^-20 up to the top of f32, plus the magnitudes where
    // a one-term reduction first leaves the interval (~1e7 upward).
    let mut args: Vec<f32> = Vec::new();
    for e in -20i32..=127 {
        for _ in 0..2_000 {
            args.push((2.0f64).powi(e) as f32 * (1.0 + rng.unit() as f32));
        }
        args.push((2.0f64).powi(e) as f32);
        args.push(-((2.0f64).powi(e) as f32));
    }
    args.extend_from_slice(&[
        0.0,
        -0.0,
        f32::MIN_POSITIVE,
        f32::MAX,
        f32::MIN,
        1.72e7,  // first reported failure
        8.64e8,  // worst reported value
        1.4e7,   // measured onset
        2.61e13, // measured inf
        1e30,
    ]);

    for &x in &args {
        for (name, got) in [("sin", sin_at(x)), ("cos", cos_at(x))] {
            assert!(
                got.is_nan() || got.abs() <= 1.0,
                "{name}({x:e}) = {got:e} — outside [-1, 1]",
            );
        }
    }
}

/// NaN and infinite arguments produce NaN, not a fabricated value.
#[test]
fn sin_propagates_nonfinite_arguments() {
    for &x in &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(sin_at(x).is_nan(), "sin({x}) should be NaN");
        assert!(cos_at(x).is_nan(), "cos({x}) should be NaN");
    }
}

// ===========================================================================
// The domain contract: accurate inside, loudly absent outside.
// ===========================================================================

/// Inside `TRIG_DOMAIN` the answer is not merely bounded, it is correct.
///
/// Without this, `sin(x) = 0` would pass every other test in this file.
#[test]
fn sin_and_cos_are_accurate_inside_the_domain() {
    // Reduction drift at the domain edge plus the degree-11 polynomial's own
    // 6e-7. Measured worst case over this sweep is ~1.5e-6.
    const TOL: f32 = 4e-6;

    let mut rng = Rng(0xc0ffee);
    let mut worst = 0.0f32;
    let mut worst_at = 0.0f32;
    for i in 0..200_000 {
        // A quarter of the samples sit in the top decade, where reduction is
        // hardest; the rest spread over all magnitudes.
        let x = if i % 4 == 0 {
            (TRIG_DOMAIN as f64 * (0.9 + 0.0999 * rng.unit())) as f32
        } else {
            rng.log_uniform(TRIG_DOMAIN as f64)
        };
        if x.abs() >= TRIG_DOMAIN {
            continue;
        }
        for (name, got, want) in [
            ("sin", sin_at(x), (x as f64).sin() as f32),
            ("cos", cos_at(x), (x as f64).cos() as f32),
        ] {
            let err = (got - want).abs();
            if err > worst {
                worst = err;
                worst_at = x;
            }
            assert!(
                err <= TOL,
                "{name}({x:e}) = {got}, want {want} (err {err:e})"
            );
        }
    }
    assert!(worst < TOL, "worst error {worst:e} at x={worst_at:e}");
}

/// Outside the domain the answer is NaN — loudly absent, not silently wrong.
///
/// A clamp into `[-1, 1]` would satisfy the range test above while still
/// handing the corpus gates a plausible-looking wrong number, which is the
/// failure mode being fixed.
#[test]
fn sin_is_nan_outside_the_domain() {
    let mut rng = Rng(0xd15ea5e);
    for _ in 0..20_000 {
        let x = rng.log_uniform(f32::MAX as f64);
        if x.abs() < TRIG_DOMAIN {
            continue;
        }
        assert!(
            sin_at(x).is_nan(),
            "sin({x:e}) should be NaN outside the domain"
        );
        assert!(
            cos_at(x).is_nan(),
            "cos({x:e}) should be NaN outside the domain"
        );
        assert!(
            tan_at(x).is_nan(),
            "tan({x:e}) should be NaN outside the domain"
        );
    }
    // The boundary itself is excluded, and just inside it is not.
    assert!(sin_at(TRIG_DOMAIN).is_nan());
    assert!(!sin_at(TRIG_DOMAIN * 0.999).is_nan());
}

/// Periodicity across many windings.
///
/// Walking whole periods away from a principal-range angle must not change the
/// answer. This is what fails first as reduction error grows — well before the
/// result leaves `[-1, 1]`, so it is the sharper of the two properties.
#[test]
fn sin_is_periodic_across_many_windings() {
    use core::f64::consts::TAU;
    for &base_x in &[0.3f32, 1.7, -2.1, 0.0, 1.5707964] {
        let base = sin_at(base_x);
        for &k in &[1i32, 2, 37, 1_000, 50_000, 160_000] {
            // Build the shifted argument in f64 so the *test* is not the thing
            // losing precision, then round once into f32.
            let shifted = (base_x as f64 + k as f64 * TAU) as f32;
            if shifted.abs() >= TRIG_DOMAIN {
                continue;
            }
            let got = sin_at(shifted);
            // Rounding `base_x + k·2π` into f32 moves the angle by up to
            // ulp(shifted)/2, and sin moves with it — that, not the reduction,
            // is what sets this tolerance.
            let quantization = (shifted as f64 - (base_x as f64 + k as f64 * TAU)).abs() as f32;
            let tol = 4e-6 + quantization;
            assert!(
                (got - base).abs() <= tol,
                "sin({base_x} + {k}·2π) = {got}, want {base} (tol {tol:e})",
            );
        }
    }
}

// ===========================================================================
// The same defect, checked for in the other trig functions.
// ===========================================================================

/// `tan` has no bounded range, so it gets the property it does have: it must
/// agree with `sin/cos` computed by this same tier, and be NaN where they are.
#[test]
fn tan_agrees_with_sin_over_cos() {
    let mut rng = Rng(0xbadc0de);
    for _ in 0..20_000 {
        let x = rng.log_uniform(TRIG_DOMAIN as f64);
        if x.abs() >= TRIG_DOMAIN {
            continue;
        }
        let (s, c) = (sin_at(x), cos_at(x));
        let got = tan_at(x);
        let want = s / c;
        assert!(
            (got - want).abs() <= 1e-5 * want.abs().max(1.0),
            "tan({x:e}) = {got}, but sin/cos = {want}",
        );
    }
}

/// `atan2` must land in `[-π, π]` — its own range bound — for every finite
/// pair, including the quadrant fix-up branches.
#[test]
fn atan2_stays_within_pi() {
    use core::f32::consts::PI;
    // The quadrant fix-up adds ±π to a polynomial value, so the bound carries
    // the atan approximation error (~2e-4 including the Recip estimate).
    const SLACK: f32 = 1e-3;

    let mut rng = Rng(0x1234_5678);
    let mut args: Vec<(f32, f32)> = vec![
        (0.0, 1.0),
        (1.0, 0.0),
        (0.0, -1.0),
        (-1.0, 0.0),
        (1.0, 1.0),
        (1.0, -1.0),
        (-1.0, -1.0),
        (-1.0, 1.0),
        (0.0, 0.0),
        (1e30, 1e-30),
        (1e-30, 1e30),
        (f32::MAX, f32::MAX),
    ];
    for _ in 0..40_000 {
        args.push((rng.log_uniform(1e20), rng.log_uniform(1e20)));
    }
    for &(y, x) in &args {
        let got = atan2_at(y, x);
        assert!(
            got.is_nan() || got.abs() <= PI + SLACK,
            "atan2({y:e}, {x:e}) = {got:e} — outside [-π, π]",
        );
    }
}

/// `atan` is bounded by π/2, and accurate.
///
/// The accuracy half is the point: a Taylor fit of the same degree is 6.2e-2 at
/// `|t| = 1` and would pass any bound-only check, so this pins the minimax fit
/// that costs exactly as much.
#[test]
fn atan_is_bounded_and_accurate() {
    use core::f32::consts::FRAC_PI_2;
    // Dominated by the `Recip` estimate in the ratio (~1.2e-4), not by the
    // polynomial (8.7e-5).
    const TOL: f32 = 5e-4;

    let mut rng = Rng(0xfeed_face);
    for i in 0..40_000 {
        let x = if i % 3 == 0 {
            (rng.unit() * 2.0 - 1.0) as f32
        } else {
            rng.log_uniform(1e18)
        };
        let got = atan_at(x);
        assert!(
            got.abs() <= FRAC_PI_2 + TOL,
            "atan({x:e}) = {got} — outside [-π/2, π/2]",
        );
        let want = (x as f64).atan() as f32;
        assert!(
            (got - want).abs() <= TOL,
            "atan({x:e}) = {got}, want {want}",
        );
    }
}

/// `asin`/`acos` are bounded on `[-1, 1]` and NaN outside it — the domain
/// error is loud (√ of a negative), not a clamp.
#[test]
fn asin_acos_bounded_in_domain_and_nan_outside() {
    use core::f32::consts::{FRAC_PI_2, PI};
    const TOL: f32 = 1e-3;

    let mut rng = Rng(0x0bad_beef);
    for _ in 0..20_000 {
        let x = (rng.unit() * 2.0 - 1.0) as f32;
        let (s, c) = (asin_at(x), acos_at(x));
        assert!(
            s.abs() <= FRAC_PI_2 + TOL,
            "asin({x}) = {s} — outside [-π/2, π/2]",
        );
        assert!(
            (-TOL..=PI + TOL).contains(&c),
            "acos({x}) = {c} — outside [0, π]",
        );
        assert!(
            (s - (x as f64).asin() as f32).abs() <= TOL,
            "asin({x}) = {s}, want {}",
            (x as f64).asin(),
        );
    }
    for &x in &[1.0001f32, -1.0001, 2.0, -5.0, 1e9, -1e9] {
        assert!(
            asin_at(x).is_nan(),
            "asin({x}) should be NaN outside [-1, 1]"
        );
        assert!(
            acos_at(x).is_nan(),
            "acos({x}) should be NaN outside [-1, 1]"
        );
    }
}

// ===========================================================================
// The expansion differentiates.
// ===========================================================================

/// `sin` differentiates even though its domain guard is a `Select`.
///
/// A `Select` in an expansion looks like it should break the derivative path,
/// and would if the passes ran the other way round. It does not, because
/// `legalize` runs `lower_dwrt` *before* `expand_transcendentals`: derivatives
/// are taken symbolically against `Sin`/`Cos`, and only the result is expanded.
/// That ordering is the whole reason the guard is allowed to exist, so this
/// pins it — if it ever inverts, `d/dx sin` breaks here rather than silently in
/// a shader.
#[test]
fn sin_differentiates_through_the_domain_guard() {
    use pixelflow_ir::passes::lower_dwrt_owned;

    for &x in &[0.0f32, 0.7, 1.3, -2.5, 100.0] {
        let mut a = ExprArena::new();
        let v = a.push_var(0);
        let s = a.push_unary(OpKind::Sin, v);
        let zero = a.push_const(0.0);
        let root = a.push_binary(OpKind::Dwrt, s, zero);
        let (out, out_root) = lower_dwrt_owned(&a, root).expect("d/dx sin must lower");

        let got = eval_scalar(&out, out_root, &[x, 0.0], &BindingTable::empty());
        let want = cos_at(x);
        assert!(
            (got - want).abs() <= 1e-5,
            "d/dx sin({x}) = {got}, want cos({x}) = {want}",
        );
    }
}

/// `sin(-0.0)` is `+0.0`, not `-0.0` — a documented, deliberate divergence.
///
/// The Cody-Waite reduction ends in `Sub(-0.0, -0.0)` at `k = 0` because
/// `TAU_LO` is negative. Fixing it would require an all-positive split whose
/// coarser remainder multiplies the reduction's drift by 15 across the whole
/// domain, so it is accepted rather than paid for (see `expand_sin_phase`).
/// This test exists so the choice stays visible: if the reduction is ever
/// restructured, this either keeps passing or is consciously updated — it does
/// not drift silently in either direction.
#[test]
fn sin_of_negative_zero_is_positive_zero() {
    let got = sin_at(-0.0);
    assert_eq!(got, 0.0, "sin(-0.0) should still be a zero");
    assert!(
        got.is_sign_positive(),
        "sin(-0.0) = {got} with sign bit set — the reduction's signed-zero \
         behaviour changed; see expand_sin_phase before updating this",
    );
    // The value at +0.0 is unaffected either way.
    assert_eq!(sin_at(0.0), 0.0);
    assert!(sin_at(0.0).is_sign_positive());
}

// ===========================================================================
// Log and Pow family domain contracts.
// ===========================================================================

/// `log2`, `ln`, `log10` return NaN for x <= 0 and non-finite arguments, and
/// are accurate for x > 0.
#[test]
fn log_family_nan_outside_domain_and_accurate_inside() {
    let out_of_domain = [0.0, -0.0, -1.0, -100.0, f32::NEG_INFINITY, f32::NAN];
    for &x in &out_of_domain {
        let l2 = log2_at(x);
        let ln = ln_at(x);
        let l10 = log10_at(x);
        assert!(l2.is_nan(), "log2({x}) should be NaN, got {l2}");
        assert!(ln.is_nan(), "ln({x}) should be NaN, got {ln}");
        assert!(l10.is_nan(), "log10({x}) should be NaN, got {l10}");
    }

    let in_domain = [0.1f32, 0.5, 1.0, 2.0, 10.0, 1000.0];
    for &x in &in_domain {
        let got_log2 = log2_at(x);
        let want_log2 = (x as f64).log2() as f32;
        assert!(
            (got_log2 - want_log2).abs() <= 1e-4,
            "log2({x}) = {got_log2}, want {want_log2}"
        );

        let got_ln = ln_at(x);
        let want_ln = (x as f64).ln() as f32;
        assert!(
            (got_ln - want_ln).abs() <= 1e-4,
            "ln({x}) = {got_ln}, want {want_ln}"
        );

        let got_log10 = log10_at(x);
        let want_log10 = (x as f64).log10() as f32;
        assert!(
            (got_log10 - want_log10).abs() <= 1e-4,
            "log10({x}) = {got_log10}, want {want_log10}"
        );
    }
}

/// `pow` returns NaN for non-positive bases (a <= 0) or NaN arguments.
#[test]
fn pow_nan_for_nonpositive_base_or_nan() {
    let bad_bases = [0.0, -0.0, -0.5, -1.0, -100.0, f32::NAN];
    for &a in &bad_bases {
        for &b in &[0.5f32, 1.0, 2.0, -1.0, f32::NAN] {
            let res = pow_at(a, b);
            assert!(res.is_nan(), "pow({a}, {b}) should be NaN, got {res}");
        }
    }
    // Base positive but exponent NaN
    assert!(pow_at(2.0, f32::NAN).is_nan());

    // Valid positive base
    let got = pow_at(2.0, 3.0);
    assert!((got - 8.0).abs() <= 1e-3, "pow(2.0, 3.0) = {got}, want 8.0");
    let got_sqrt = pow_at(4.0, 0.5);
    assert!(
        (got_sqrt - 2.0).abs() <= 1e-3,
        "pow(4.0, 0.5) = {got_sqrt}, want 2.0"
    );
}

/// `exp` and `exp2` propagate NaN arguments.
#[test]
fn exp_family_propagates_nan() {
    let e_nan = exp_at(f32::NAN);
    let e2_nan = exp2_at(f32::NAN);
    assert!(e_nan.is_nan(), "exp(NAN) should be NaN, got {e_nan}");
    assert!(e2_nan.is_nan(), "exp2(NAN) should be NaN, got {e2_nan}");

    let got_exp = exp_at(0.0);
    assert!((got_exp - 1.0).abs() <= 1e-4);

    let got_exp2 = exp2_at(3.0);
    assert!((got_exp2 - 8.0).abs() <= 1e-4);
}
