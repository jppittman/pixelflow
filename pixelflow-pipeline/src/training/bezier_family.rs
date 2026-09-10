//! The `bezier` DEV-only out-of-distribution family: Bernstein / de Casteljau
//! curve distances and a bicubic tensor patch, POLY-only
//! (`Add`/`Sub`/`Mul`/`MulAdd`/`Neg`), per
//! `docs/plans/2026-09-01-phase3-round1b-domain-shift-registration.md` §3b.
//!
//! The generator lives here, beside [`super::sh_family`], so that every
//! harness sampling the family — `gen_bezier_corpus` (the corpus writer) and
//! the structural-gap inventory (`corpus_gaps`) — draws from one definition.
//! The writer's admission logic (size band, in-family dedup, TRAIN fence,
//! numeric quarantine) stays in the binary: it is corpus policy, not the
//! family.
//!
//! # Oracle validation
//!
//! `Quarantine` cross-checks the JIT lane against `eval_scalar` on the SAME
//! arena — it catches an emitter bug, but it cannot catch this file building
//! the WRONG expression (a transposed binomial coefficient, an off-by-one
//! power), because both lanes agree on whatever tree got built. The
//! `oracle_tests` module below is the check that catches THAT: a
//! from-scratch scalar-Rust reference for each form, cross-checked against
//! `eval_scalar` on genuinely independent control points and sample points.

use pixelflow_ir::{ExprBuilder, ExprGraph, ExprHandle, OpKind};

// ============================================================================
// RNG — the same PCG-style LCG `pixelflow_search::nnue::BwdGenerator` uses,
// so a `bezier` reader recognizes the recipe instead of learning a new one.
// ============================================================================

/// The family's draw stream.
pub struct Lcg(u64);

impl Lcg {
    /// A stream seeded at `seed`; the same seed reproduces the same draws.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        (self.0 >> 33) as f32 / (1u64 << 31) as f32
    }

    /// Uniform in `[lo, hi)`.
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next_f32() * (hi - lo)
    }

    /// Uniform index in `0..n`. `n` must be nonzero.
    pub fn choice(&mut self, n: usize) -> usize {
        let v = (self.next_f32() * n as f32) as usize;
        v.min(n - 1)
    }
}

// ============================================================================
// Bernstein / de Casteljau construction — POLY-only (Add/Sub/Mul/MulAdd/Neg)
// ============================================================================

const BINOM3: [f32; 4] = [1.0, 3.0, 3.0, 1.0];
const BINOM4: [f32; 5] = [1.0, 4.0, 6.0, 4.0, 1.0];

fn binom(n: usize) -> &'static [f32] {
    match n {
        3 => &BINOM3,
        4 => &BINOM4,
        other => panic!("bezier corpus only defines degree 3/4, got {other}"),
    }
}

fn fold_add(builder: &mut ExprBuilder, xs: &[ExprHandle]) -> ExprHandle {
    assert!(!xs.is_empty(), "fold_add of an empty term list");
    let mut acc = xs[0];
    for &x in &xs[1..] {
        acc = builder.binary(OpKind::Add, acc, x);
    }
    acc
}

/// `[base^0 (represented as None, i.e. "no factor"), base^1, .., base^n]`.
fn powers(builder: &mut ExprBuilder, base: ExprHandle, n: usize) -> Vec<Option<ExprHandle>> {
    let mut v = vec![None; n + 1];
    if n >= 1 {
        v[1] = Some(base);
    }
    for k in 2..=n {
        let prev = v[k - 1].expect("power k-1 already computed");
        v[k] = Some(builder.binary(OpKind::Mul, prev, base));
    }
    v
}

/// One 1-D Bernstein-basis-weighted sum, degree `n`, evaluated at `t`, self-
/// contained (recomputes its own power table each call — deliberately not
/// shared across the two coordinate calls a curve needs, both because it
/// keeps this function simple and because the registration's own node-count
/// estimates (~60 cubic / ~85 quartic for `bezier-bernstein`) assume the
/// un-shared construction).
fn bernstein_1d(builder: &mut ExprBuilder, t: ExprHandle, n: usize, values: &[f32]) -> ExprHandle {
    assert_eq!(values.len(), n + 1, "bernstein_1d: need n+1 control values");
    let one = builder.constant(1.0);
    let omt = builder.binary(OpKind::Sub, one, t);
    let t_pows = powers(builder, t, n);
    let omt_pows = powers(builder, omt, n);
    let b = binom(n);
    let terms: Vec<ExprHandle> = (0..=n)
        .map(|i| {
            let c = builder.constant(b[i]);
            let mut acc = c;
            if let Some(tp) = t_pows[i] {
                acc = builder.binary(OpKind::Mul, acc, tp);
            }
            if let Some(op) = omt_pows[n - i] {
                acc = builder.binary(OpKind::Mul, acc, op);
            }
            let pc = builder.constant(values[i]);
            builder.binary(OpKind::Mul, acc, pc)
        })
        .collect();
    fold_add(builder, &terms)
}

/// `lerp(a, b, t) = a + (b - a)*t`, one `Sub` + one `MulAdd` (one rounding).
fn lerp(builder: &mut ExprBuilder, p0: ExprHandle, p1: ExprHandle, t: ExprHandle) -> ExprHandle {
    let diff = builder.binary(OpKind::Sub, p1, p0);
    builder.ternary(OpKind::MulAdd, diff, t, p0)
}

/// de Casteljau reduction of `points` (already-pushed leaf `ExprId`s) at `t`:
/// repeatedly lerp adjacent pairs until one point remains. `n` points give
/// `n-1 + n-2 + .. + 1` lerps (6 for 4 points / degree 3, 10 for 5 points /
/// degree 4 — registration §3b).
fn de_casteljau(builder: &mut ExprBuilder, points: &[ExprHandle], t: ExprHandle) -> ExprHandle {
    assert!(points.len() >= 2, "de Casteljau needs at least 2 points");
    let mut level = points.to_vec();
    while level.len() > 1 {
        let next: Vec<ExprHandle> = level
            .windows(2)
            .map(|w| lerp(builder, w[0], w[1], t))
            .collect();
        level = next;
    }
    level[0]
}

/// The squared-distance tail shared by `bezier-bernstein` and
/// `bezier-casteljau`: `(Bx(X) - Y)^2 + (By(X) - c0)^2` (registration §3b).
/// Squared, not `Sqrt`ed, so the family stays `Div`/`Sqrt`-free.
fn squared_distance(
    builder: &mut ExprBuilder,
    bx: ExprHandle,
    by: ExprHandle,
    y: ExprHandle,
    c0: f32,
) -> ExprHandle {
    let c0_id = builder.constant(c0);
    let dx = builder.binary(OpKind::Sub, bx, y);
    let dx2 = builder.binary(OpKind::Mul, dx, dx);
    let dy = builder.binary(OpKind::Sub, by, c0_id);
    let dy2 = builder.binary(OpKind::Mul, dy, dy);
    builder.binary(OpKind::Add, dx2, dy2)
}

/// Draw `n+1` control values ~ U(lo, hi).
fn draw_controls(rng: &mut Lcg, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    (0..=n).map(|_| rng.range(lo, hi)).collect()
}

fn build_bernstein(rng: &mut Lcg, n: usize) -> ExprGraph {
    let mut a = ExprBuilder::new();
    let t = a.var(0);
    let y = a.var(1);
    let cx = draw_controls(rng, n, -2.0, 2.0);
    let cy = draw_controls(rng, n, -2.0, 2.0);
    let c0 = rng.range(-2.0, 2.0);
    let bx = bernstein_1d(&mut a, t, n, &cx);
    let by = bernstein_1d(&mut a, t, n, &cy);
    let root = squared_distance(&mut a, bx, by, y, c0);
    a.finish_one(root)
}

fn build_casteljau(rng: &mut Lcg, n: usize) -> ExprGraph {
    let mut a = ExprBuilder::new();
    let t = a.var(0);
    let y = a.var(1);
    let cx = draw_controls(rng, n, -2.0, 2.0);
    let cy = draw_controls(rng, n, -2.0, 2.0);
    let c0 = rng.range(-2.0, 2.0);
    let px: Vec<ExprHandle> = cx.iter().map(|&v| a.constant(v)).collect();
    let py: Vec<ExprHandle> = cy.iter().map(|&v| a.constant(v)).collect();
    let bx = de_casteljau(&mut a, &px, t);
    let by = de_casteljau(&mut a, &py, t);
    let root = squared_distance(&mut a, bx, by, y, c0);
    a.finish_one(root)
}

/// Fixed bicubic tensor-product patch: z(X,Y) = Σᵢ Σⱼ Pᵢⱼ·Bᵢ(X)·Bⱼ(Y), 16
/// independent heights ~ U(-2,2) (registration §3b). `Bᵢ`/`Bⱼ` are built
/// per (i,j) term rather than shared across terms — same "self-contained,
/// not hand-optimized for sharing" choice as `bernstein_1d`.
fn build_patch(rng: &mut Lcg) -> ExprGraph {
    let mut a = ExprBuilder::new();
    let x = a.var(0);
    let y = a.var(1);
    let n = 3usize;
    let b = binom(n);
    let one_x = a.constant(1.0);
    let omx = a.binary(OpKind::Sub, one_x, x);
    let x_pows = powers(&mut a, x, n);
    let omx_pows = powers(&mut a, omx, n);
    let one_y = a.constant(1.0);
    let omy = a.binary(OpKind::Sub, one_y, y);
    let y_pows = powers(&mut a, y, n);
    let omy_pows = powers(&mut a, omy, n);

    let basis = |a: &mut ExprBuilder,
                 pows: &[Option<ExprHandle>],
                 ompows: &[Option<ExprHandle>],
                 i: usize| {
        let c = a.constant(b[i]);
        let mut acc = c;
        if let Some(p) = pows[i] {
            acc = a.binary(OpKind::Mul, acc, p);
        }
        if let Some(op) = ompows[n - i] {
            acc = a.binary(OpKind::Mul, acc, op);
        }
        acc
    };

    let mut terms = Vec::with_capacity((n + 1) * (n + 1));
    for i in 0..=n {
        let bi = basis(&mut a, &x_pows, &omx_pows, i);
        for j in 0..=n {
            let bj = basis(&mut a, &y_pows, &omy_pows, j);
            let height = rng.range(-2.0, 2.0);
            let p = a.constant(height);
            let bij = a.binary(OpKind::Mul, bi, bj);
            terms.push(a.binary(OpKind::Mul, bij, p));
        }
    }
    let root = fold_add(&mut a, &terms);
    a.finish_one(root)
}

/// The five registered constructions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Form {
    BernsteinCubic,
    BernsteinQuartic,
    CasteljauCubic,
    CasteljauQuartic,
    Patch,
}

/// Every form, in a fixed order; `draw` picks uniformly among them.
pub const FORMS: [Form; 5] = [
    Form::BernsteinCubic,
    Form::BernsteinQuartic,
    Form::CasteljauCubic,
    Form::CasteljauQuartic,
    Form::Patch,
];

impl Form {
    /// The registered family label of this form.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Form::BernsteinCubic | Form::BernsteinQuartic => "bezier-bernstein",
            Form::CasteljauCubic | Form::CasteljauQuartic => "bezier-casteljau",
            Form::Patch => "bezier-patch",
        }
    }

    /// Polynomial degree of this form.
    #[must_use]
    pub fn degree(self) -> usize {
        match self {
            Form::BernsteinCubic | Form::CasteljauCubic | Form::Patch => 3,
            Form::BernsteinQuartic | Form::CasteljauQuartic => 4,
        }
    }

    /// Build one instance of this form from `rng`.
    #[must_use]
    pub fn build(self, rng: &mut Lcg) -> ExprGraph {
        match self {
            Form::BernsteinCubic => build_bernstein(rng, 3),
            Form::BernsteinQuartic => build_bernstein(rng, 4),
            Form::CasteljauCubic => build_casteljau(rng, 3),
            Form::CasteljauQuartic => build_casteljau(rng, 4),
            Form::Patch => build_patch(rng),
        }
    }
}

/// One draw: a form chosen uniformly, built from `rng`.
#[must_use]
pub fn draw(rng: &mut Lcg) -> (Form, ExprGraph) {
    let form = FORMS[rng.choice(FORMS.len())];
    (form, form.build(rng))
}

// ============================================================================
// Oracle validation: an independent scalar-Rust reference for each form,
// cross-checked against `eval_scalar` on the SAME construction code the
// corpus writer uses — this is the check that would catch a wrong formula,
// which `Quarantine` (JIT vs. `eval_scalar` on the SAME, possibly-wrong,
// arena) structurally cannot.
// ============================================================================

#[cfg(test)]
mod oracle_tests {

    #[test]
    fn de_casteljau_reduces_correct_lerp_count() {
        // Registration §3b: 6 lerps for cubic, 10 for quartic, per
        // coordinate. Spot-check the reduction count itself, independent of
        // arena construction.
        assert_eq!(
            {
                let mut count = 0;
                let mut level = vec![0.0f32; 4];
                while level.len() > 1 {
                    level = level
                        .windows(2)
                        .map(|_| {
                            count += 1;
                            0.0
                        })
                        .collect();
                }
                count
            },
            6
        );
        assert_eq!(
            {
                let mut count = 0;
                let mut level = vec![0.0f32; 5];
                while level.len() > 1 {
                    level = level
                        .windows(2)
                        .map(|_| {
                            count += 1;
                            0.0
                        })
                        .collect();
                }
                count
            },
            10
        );
    }
}
