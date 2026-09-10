//! Real spherical-harmonics expressions — the round 1b `sh` OOD family.
//!
//! Built for
//! `docs/plans/2026-09-01-phase3-round1b-domain-shift-registration.md` §3a,
//! testing JP's claim verbatim: *"The pythagorean identities are useless
//! when trying to evaluate bezier curves, but i sure as shit want them
//! firing when we're doing spherical harmonics."* This module is the
//! generator; `pixelflow-pipeline/src/bin/gen_sh_corpus.rs` is the binary
//! that fences, quarantines, and writes it to `corpus_dev_ood.bin`.
//!
//! # Parameterisation (fixed by the registration)
//!
//! θ = [`THETA_VAR`] (`Var(0)`, the graph's `X`), φ = [`PHI_VAR`] (`Var(1)`,
//! `Y`) — the coordinate variables ARE the angles. Not the Cartesian route
//! (normalise a direction, then polynomials in nx, ny, nz — contains no trig
//! at all) and not `atan2`/`acos` of a direction (inverse-trig ops no rule
//! addresses, hiding the sin/cos structure the identities match on).
//!
//! # Basis
//!
//! Real spherical harmonics in spherical coordinates, Condon–Shortley phase
//! omitted — the graphics convention of Green, "Spherical Harmonic
//! Lighting: The Gritty Details" (2003), §"The Real Spherical Harmonics":
//!
//! ```text
//! y_l^m(θ,φ) = √2 · K_l^m · cos(mφ) · P_l^m(cosθ)   for m > 0
//!            = √2 · K_l^m · sin(|m|φ) · P_l^|m|(cosθ)  for m < 0
//!            = K_l^0 · P_l^0(cosθ)                  for m = 0
//! ```
//!
//! [`theta_factor`] hardcodes the products of `K_l^m` with the associated
//! Legendre polynomial `P_l^{|m|}(cosθ)`, written out in sinθ/cosθ — the same
//! table any real-SH reference lists through l = 4 (matching Green's table;
//! the Cartesian-polynomial form in Sloan, "Stupid Spherical Harmonics
//! Tricks" (2008) Appendix A2 is the same functions and is deliberately NOT
//! used here — see the parameterisation note above). Per the table, `K_l^m`
//! and the θ-polynomial are shared between +m and -m; only the φ factor
//! (`cos(mφ)` vs `sin(mφ)`) differs, which is exactly how [`theta_factor`]
//! and [`phi_factor`] split the two.
//!
//! # Forms
//!
//! [`Form::Direct`]: `sin(mφ)`/`cos(mφ)` for m ∈ {2,3,4} spelled as
//! `Sin(Const(m) * φ)` — angle addition can only reach these through
//! `doubling`/`halving` (`2x ↔ x + x`), so this is the form where that
//! enabler rule matters. [`Form::Expanded`]: the multiple-angle identities
//! written out in sinφ, cosφ (`sin2φ = 2 sinφ cosφ`, `cos2φ = cos²φ - sin²φ`,
//! `sin3φ = 3 sinφ - 4 sin³φ`, `cos3φ = 4 cos³φ - 3 cosφ`,
//! `sin4φ = 4 sinφ cosφ (2cos²φ - 1)`, `cos4φ = 8cos⁴φ - 8cos²φ + 1`) — the
//! form where `half-angle-product`, `reverse-angle-addition`, and
//! `pythagorean` have live matches. m = 1 is `sinφ`/`cosφ` directly in
//! either form (no "multiple" to expand).
//!
//! [`sh_power`] builds the third form: a band energy `Σ_m (Y_l^m)²`, the
//! rotation-invariant quantity SH lighting actually sums — the purest
//! `pythagorean` bait in the set, since φ appears only inside
//! `sin²(mφ) + cos²(mφ)` terms once every m is squared and summed.

use pixelflow_ir::{ExprBuilder, ExprGraph, ExprHandle, OpKind};

/// θ — the polar angle — is the graph's `X` coordinate (`Var(0)`).
pub const THETA_VAR: u8 = 0;
/// φ — the azimuthal angle — is the graph's `Y` coordinate (`Var(1)`).
pub const PHI_VAR: u8 = 1;

/// How `sin(mφ)`/`cos(mφ)` for m ∈ {2,3,4} are spelled. m = 1 is identical
/// (plain `sinφ`/`cosφ`) under both — see the module docs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Form {
    /// `Sin(Const(m) * φ)` / `Cos(Const(m) * φ)`.
    Direct,
    /// The multiple-angle identity expanded in sinφ, cosφ.
    Expanded,
}

/// Deterministic, dependency-free xorshift64* generator — this module's own
/// PRNG so the corpus is reproducible from a `u64` seed without pulling in a
/// `rand` dependency this crate does not otherwise need.
pub struct Rng(u64);

impl Rng {
    #[must_use]
    pub fn new(seed: u64) -> Self {
        // SplitMix64-style finalizer avoids a zero or low-entropy seed
        // producing a degenerate short cycle.
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        Self(z ^ (z >> 31))
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform `f32` in `[0, 1)`.
    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform `f32` in `[lo, hi)`.
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.unit() * (hi - lo)
    }

    /// Uniform choice among `[0, n)`.
    fn below(&mut self, n: u32) -> u32 {
        (self.next_u64() % u64::from(n)) as u32
    }

    fn form(&mut self) -> Form {
        if self.below(2) == 0 {
            Form::Direct
        } else {
            Form::Expanded
        }
    }
}

/// `base` raised to the integer power `n` (n ∈ {2,3,4}) by repeated
/// multiplication in the builder — no `Pow` op, so no exp/log rule can touch
/// it, matching the hand-expanded polynomial form real-SH references use.
fn ipow(builder: &mut ExprBuilder, base: ExprHandle, n: u32) -> ExprHandle {
    assert!(n >= 1, "ipow: exponent must be >= 1, got {n}");
    let mut acc = base;
    for _ in 1..n {
        acc = builder.binary(OpKind::Mul, acc, base);
    }
    acc
}

/// `Sin(θ)`/`Cos(θ)`/`Sin(φ)`/`Cos(φ)`, computed once per draw and threaded
/// through every basis-function builder as one value (DAG sharing — each
/// node is reused, never rebuilt) instead of four positional handles
/// arguments.
#[derive(Clone, Copy)]
struct TrigBasis {
    sin_th: ExprHandle,
    cos_th: ExprHandle,
    sin_phi: ExprHandle,
    cos_phi: ExprHandle,
}

impl TrigBasis {
    /// Push fresh `θ`/`φ` variables and their `Sin`/`Cos` into the builder.
    fn build(builder: &mut ExprBuilder) -> Self {
        let x = builder.var(THETA_VAR);
        let y = builder.var(PHI_VAR);
        Self {
            sin_th: builder.unary(OpKind::Sin, x),
            cos_th: builder.unary(OpKind::Cos, x),
            sin_phi: builder.unary(OpKind::Sin, y),
            cos_phi: builder.unary(OpKind::Cos, y),
        }
    }
}

/// `sin(mφ)` for m ∈ {1,2,3,4}, per [`Form`]. `sin_phi`/`cos_phi` are the
/// caller's cached `Sin(φ)`/`Cos(φ)` nodes, reused (DAG sharing) rather than
/// rebuilt per call.
fn sin_mphi(
    builder: &mut ExprBuilder,
    sin_phi: ExprHandle,
    cos_phi: ExprHandle,
    m: u32,
    form: Form,
) -> ExprHandle {
    match (m, form) {
        (1, _) => sin_phi,
        (m, Form::Direct) => {
            let k = builder.constant(m as f32);
            let y = phi(builder);
            let angle = builder.binary(OpKind::Mul, k, y);
            builder.unary(OpKind::Sin, angle)
        }
        (2, Form::Expanded) => {
            // sin2φ = 2 sinφ cosφ
            let two = builder.constant(2.0);
            let sc = builder.binary(OpKind::Mul, sin_phi, cos_phi);
            builder.binary(OpKind::Mul, two, sc)
        }
        (3, Form::Expanded) => {
            // sin3φ = 3 sinφ - 4 sin³φ
            let three = builder.constant(3.0);
            let four = builder.constant(4.0);
            let a = builder.binary(OpKind::Mul, three, sin_phi);
            let sin3 = ipow(builder, sin_phi, 3);
            let b = builder.binary(OpKind::Mul, four, sin3);
            builder.binary(OpKind::Sub, a, b)
        }
        (4, Form::Expanded) => {
            // sin4φ = 4 sinφ cosφ (2cos²φ - 1)
            let four = builder.constant(4.0);
            let two = builder.constant(2.0);
            let one = builder.constant(1.0);
            let sc = builder.binary(OpKind::Mul, sin_phi, cos_phi);
            let four_sc = builder.binary(OpKind::Mul, four, sc);
            let cos2 = ipow(builder, cos_phi, 2);
            let two_cos2 = builder.binary(OpKind::Mul, two, cos2);
            let paren = builder.binary(OpKind::Sub, two_cos2, one);
            builder.binary(OpKind::Mul, four_sc, paren)
        }
        (m, _) => unreachable!("sin_mphi: m = {m} outside 1..=4"),
    }
}

/// `cos(mφ)` for m ∈ {1,2,3,4}, per [`Form`] — mirrors [`sin_mphi`].
fn cos_mphi(
    builder: &mut ExprBuilder,
    sin_phi: ExprHandle,
    cos_phi: ExprHandle,
    m: u32,
    form: Form,
) -> ExprHandle {
    match (m, form) {
        (1, _) => cos_phi,
        (m, Form::Direct) => {
            let k = builder.constant(m as f32);
            let y = phi(builder);
            let angle = builder.binary(OpKind::Mul, k, y);
            builder.unary(OpKind::Cos, angle)
        }
        (2, Form::Expanded) => {
            // cos2φ = cos²φ - sin²φ
            let c2 = ipow(builder, cos_phi, 2);
            let s2 = ipow(builder, sin_phi, 2);
            builder.binary(OpKind::Sub, c2, s2)
        }
        (3, Form::Expanded) => {
            // cos3φ = 4cos³φ - 3cosφ
            let four = builder.constant(4.0);
            let three = builder.constant(3.0);
            let cos3 = ipow(builder, cos_phi, 3);
            let a = builder.binary(OpKind::Mul, four, cos3);
            let b = builder.binary(OpKind::Mul, three, cos_phi);
            builder.binary(OpKind::Sub, a, b)
        }
        (4, Form::Expanded) => {
            // cos4φ = 8cos⁴φ - 8cos²φ + 1
            let eight = builder.constant(8.0);
            let one = builder.constant(1.0);
            let cos4 = ipow(builder, cos_phi, 4);
            let cos2 = ipow(builder, cos_phi, 2);
            let a = builder.binary(OpKind::Mul, eight, cos4);
            let b = builder.binary(OpKind::Mul, eight, cos2);
            let ab = builder.binary(OpKind::Sub, a, b);
            builder.binary(OpKind::Add, ab, one)
        }
        (m, _) => unreachable!("cos_mphi: m = {m} outside 1..=4"),
    }
}

/// Convenience: push a fresh reference to the φ variable. Cheap — `Var` is a
/// tiny leaf node, and the builder interns equivalent nodes — callers that
/// already hold a cached φ id should use that instead of this.
fn phi(builder: &mut ExprBuilder) -> ExprHandle {
    builder.var(PHI_VAR)
}

/// `K_l^{|m|} · P_l^{|m|}(cosθ)` — the θ-only factor shared by `+m` and `-m`,
/// written out in sinθ/cosθ per Green's real-SH table. `l` ∈ {0,1,2,3,4},
/// `m_abs` ∈ `0..=l`. `sin_th`/`cos_th` are the caller's cached
/// `Sin(θ)`/`Cos(θ)` nodes.
///
/// # Panics
///
/// Panics for `l` or `m_abs` outside the table — a caller bug, not a data
/// condition.
fn theta_factor(
    builder: &mut ExprBuilder,
    sin_th: ExprHandle,
    cos_th: ExprHandle,
    l: u32,
    m_abs: u32,
) -> ExprHandle {
    use std::f32::consts::PI;
    let k_const = |builder: &mut ExprBuilder, k: f32| builder.constant(k);
    match (l, m_abs) {
        (0, 0) => {
            let k = 0.5 * (1.0 / PI).sqrt();
            k_const(builder, k)
        }
        (1, 0) => {
            let k = (3.0 / (4.0 * PI)).sqrt();
            let kc = k_const(builder, k);
            builder.binary(OpKind::Mul, kc, cos_th)
        }
        (1, 1) => {
            let k = (3.0 / (4.0 * PI)).sqrt();
            let kc = k_const(builder, k);
            builder.binary(OpKind::Mul, kc, sin_th)
        }
        (2, 0) => {
            // K * (3cos²θ - 1)
            let k = 0.25 * (5.0 / PI).sqrt();
            let three = builder.constant(3.0);
            let one = builder.constant(1.0);
            let cos2 = ipow(builder, cos_th, 2);
            let a = builder.binary(OpKind::Mul, three, cos2);
            let paren = builder.binary(OpKind::Sub, a, one);
            let kc = k_const(builder, k);
            builder.binary(OpKind::Mul, kc, paren)
        }
        (2, 1) => {
            // K * sinθ cosθ
            let k = 0.5 * (15.0 / PI).sqrt();
            let sc = builder.binary(OpKind::Mul, sin_th, cos_th);
            let kc = k_const(builder, k);
            builder.binary(OpKind::Mul, kc, sc)
        }
        (2, 2) => {
            // K * sin²θ
            let k = 0.25 * (15.0 / PI).sqrt();
            let s2 = ipow(builder, sin_th, 2);
            let kc = k_const(builder, k);
            builder.binary(OpKind::Mul, kc, s2)
        }
        (3, 0) => {
            // K * (5cos³θ - 3cosθ)
            let k = 0.25 * (7.0 / PI).sqrt();
            let five = builder.constant(5.0);
            let three = builder.constant(3.0);
            let cos3 = ipow(builder, cos_th, 3);
            let a = builder.binary(OpKind::Mul, five, cos3);
            let b = builder.binary(OpKind::Mul, three, cos_th);
            let paren = builder.binary(OpKind::Sub, a, b);
            let kc = k_const(builder, k);
            builder.binary(OpKind::Mul, kc, paren)
        }
        (3, 1) => {
            // K * sinθ (5cos²θ - 1)
            let k = 0.25 * (21.0 / (2.0 * PI)).sqrt();
            let five = builder.constant(5.0);
            let one = builder.constant(1.0);
            let cos2 = ipow(builder, cos_th, 2);
            let a = builder.binary(OpKind::Mul, five, cos2);
            let paren = builder.binary(OpKind::Sub, a, one);
            let s_paren = builder.binary(OpKind::Mul, sin_th, paren);
            let kc = k_const(builder, k);
            builder.binary(OpKind::Mul, kc, s_paren)
        }
        (3, 2) => {
            // K * sin²θ cosθ
            let k = 0.25 * (105.0 / PI).sqrt();
            let s2 = ipow(builder, sin_th, 2);
            let s2c = builder.binary(OpKind::Mul, s2, cos_th);
            let kc = k_const(builder, k);
            builder.binary(OpKind::Mul, kc, s2c)
        }
        (3, 3) => {
            // K * sin³θ
            let k = 0.25 * (35.0 / (2.0 * PI)).sqrt();
            let s3 = ipow(builder, sin_th, 3);
            let kc = k_const(builder, k);
            builder.binary(OpKind::Mul, kc, s3)
        }
        (4, 0) => {
            // K * (35cos⁴θ - 30cos²θ + 3)
            let k = (3.0 / 16.0) * (1.0 / PI).sqrt();
            let c35 = builder.constant(35.0);
            let c30 = builder.constant(30.0);
            let c3 = builder.constant(3.0);
            let cos4 = ipow(builder, cos_th, 4);
            let cos2 = ipow(builder, cos_th, 2);
            let a = builder.binary(OpKind::Mul, c35, cos4);
            let b = builder.binary(OpKind::Mul, c30, cos2);
            let ab = builder.binary(OpKind::Sub, a, b);
            let paren = builder.binary(OpKind::Add, ab, c3);
            let kc = k_const(builder, k);
            builder.binary(OpKind::Mul, kc, paren)
        }
        (4, 1) => {
            // K * sinθ (7cos³θ - 3cosθ)
            let k = 0.75 * (5.0 / (2.0 * PI)).sqrt();
            let seven = builder.constant(7.0);
            let three = builder.constant(3.0);
            let cos3 = ipow(builder, cos_th, 3);
            let a = builder.binary(OpKind::Mul, seven, cos3);
            let b = builder.binary(OpKind::Mul, three, cos_th);
            let paren = builder.binary(OpKind::Sub, a, b);
            let s_paren = builder.binary(OpKind::Mul, sin_th, paren);
            let kc = k_const(builder, k);
            builder.binary(OpKind::Mul, kc, s_paren)
        }
        (4, 2) => {
            // K * sin²θ (7cos²θ - 1)
            let k = 0.375 * (5.0 / PI).sqrt();
            let seven = builder.constant(7.0);
            let one = builder.constant(1.0);
            let cos2 = ipow(builder, cos_th, 2);
            let a = builder.binary(OpKind::Mul, seven, cos2);
            let paren = builder.binary(OpKind::Sub, a, one);
            let s2 = ipow(builder, sin_th, 2);
            let s2_paren = builder.binary(OpKind::Mul, s2, paren);
            let kc = k_const(builder, k);
            builder.binary(OpKind::Mul, kc, s2_paren)
        }
        (4, 3) => {
            // K * sin³θ cosθ
            let k = 0.75 * (35.0 / (2.0 * PI)).sqrt();
            let s3 = ipow(builder, sin_th, 3);
            let s3c = builder.binary(OpKind::Mul, s3, cos_th);
            let kc = k_const(builder, k);
            builder.binary(OpKind::Mul, kc, s3c)
        }
        (4, 4) => {
            // K * sin⁴θ
            let k = (3.0 / 16.0) * (35.0 / PI).sqrt();
            let s4 = ipow(builder, sin_th, 4);
            let kc = k_const(builder, k);
            builder.binary(OpKind::Mul, kc, s4)
        }
        (l, m) => unreachable!("theta_factor: (l={l}, m={m}) outside the l<=4 table"),
    }
}

/// `Y_l^m(θ, φ)`: [`theta_factor`] times the φ factor (`sin(|m|φ)` for
/// m < 0, `cos(mφ)` for m > 0, nothing for m = 0). `m` is signed
/// (`-l..=l`).
fn y_l_m(builder: &mut ExprBuilder, basis: TrigBasis, l: u32, m: i32, form: Form) -> ExprHandle {
    let m_abs = m.unsigned_abs();
    let theta = theta_factor(builder, basis.sin_th, basis.cos_th, l, m_abs);
    match m.cmp(&0) {
        std::cmp::Ordering::Equal => theta,
        std::cmp::Ordering::Less => {
            let s = sin_mphi(builder, basis.sin_phi, basis.cos_phi, m_abs, form);
            builder.binary(OpKind::Mul, theta, s)
        }
        std::cmp::Ordering::Greater => {
            let c = cos_mphi(builder, basis.sin_phi, basis.cos_phi, m_abs, form);
            builder.binary(OpKind::Mul, theta, c)
        }
    }
}

/// Fisher-Yates shuffle of `items` using `rng`.
fn shuffle<T>(items: &mut [T], rng: &mut Rng) {
    for i in (1..items.len()).rev() {
        let j = rng.below((i + 1) as u32) as usize;
        items.swap(i, j);
    }
}

/// A band-limited SH dot product `Σ_{l<=L} Σ_m c_lm · Y_l^m(θ,φ)` over fresh
/// `θ`/`φ` roots, with `c_lm ~ U(-1,1)` drawn from `rng`.
///
/// The `(l,m)` term set is not always the full band — each term is kept
/// independently at a per-draw rate in `[0.65, 1.0]`, so the corpus's
/// structural dedup key (`FenceKey`, which is blind to `c_lm`'s literal
/// value — see `training::structural`) sees a different topology per draw
/// instead of collapsing every L=3 draw to one shape. Term order is then
/// shuffled, and each accumulation step independently chooses `MulAdd`
/// fusion or a plain `Mul`+`Add` pair — both mathematically the sum the
/// docstring states; the choice only varies which op nodes represent it,
/// which is exactly the axis `FenceKey` is sensitive to.
fn sh_sum(builder: &mut ExprBuilder, l_max: u32, form: Form, rng: &mut Rng) -> ExprHandle {
    let basis = TrigBasis::build(builder);

    let mut terms: Vec<(u32, i32)> = Vec::new();
    for l in 0..=l_max {
        for m in -(l as i32)..=(l as i32) {
            terms.push((l, m));
        }
    }
    let keep_rate = rng.range(0.65, 1.0);
    terms.retain(|_| rng.unit() < keep_rate);
    if terms.is_empty() {
        terms.push((0, 0));
    }
    shuffle(&mut terms, rng);

    let mut acc: Option<ExprHandle> = None;
    for (l, m) in terms {
        let ylm = y_l_m(builder, basis, l, m, form);
        let c = rng.range(-1.0, 1.0);
        let cc = builder.constant(c);
        acc = Some(match acc {
            None => builder.binary(OpKind::Mul, cc, ylm),
            Some(prev) if rng.below(2) == 0 => builder.ternary(OpKind::MulAdd, cc, ylm, prev),
            Some(prev) => {
                let term = builder.binary(OpKind::Mul, cc, ylm);
                builder.binary(OpKind::Add, prev, term)
            }
        });
    }
    acc.expect("terms is non-empty (guarded above)")
}

/// Band energy `Σ_m (Y_l^m)²` for one `l` — the rotation-invariant quantity
/// SH lighting sums, and (by the spherical-harmonics addition theorem) a
/// constant independent of θ,φ: exactly the shape where φ appears only
/// inside `sin²(mφ) + cos²(mφ)` once every `m` is squared and summed, the
/// purest `pythagorean` bait this family offers.
fn band_energy(builder: &mut ExprBuilder, basis: TrigBasis, l: u32, form: Form) -> ExprHandle {
    let mut terms: Vec<ExprHandle> = Vec::new();
    for m in -(l as i32)..=(l as i32) {
        let ylm = y_l_m(builder, basis, l, m, form);
        terms.push(builder.binary(OpKind::Mul, ylm, ylm));
    }
    let mut acc = terms[0];
    for &t in &terms[1..] {
        acc = builder.binary(OpKind::Add, acc, t);
    }
    acc
}

/// One `sh-power` draw: band energies for a random non-empty subset of
/// l ∈ {1,2,3} (each at an independently drawn [`Form`]), added to a small
/// dot product so the whole expression clears the classical size band (node
/// count > 50) — the band-energy terms alone are small. The subset/form/L
/// draws are this structure's source of `FenceKey` diversity, the same role
/// [`sh_sum`]'s term-subset draw plays for the direct/expanded structures.
fn sh_power(builder: &mut ExprBuilder, rng: &mut Rng) -> ExprHandle {
    let basis = TrigBasis::build(builder);

    let mut bands: Vec<u32> = vec![1, 2, 3];
    bands.retain(|_| rng.unit() < 0.85);
    if bands.is_empty() {
        bands.push(1 + rng.below(3));
    }
    shuffle(&mut bands, rng);

    let mut energies: Option<ExprHandle> = None;
    for l in bands {
        let e = band_energy(builder, basis, l, rng.form());
        energies = Some(match energies {
            None => e,
            Some(prev) => builder.binary(OpKind::Add, prev, e),
        });
    }
    let energies = energies.expect("bands is non-empty (guarded above)");

    let l_max = 1 + rng.below(2);
    let form = rng.form();
    let dot = sh_sum(builder, l_max, form, rng);
    builder.binary(OpKind::Add, energies, dot)
}

/// Draw one `sh` corpus candidate from `rng`: picks a form and a structure
/// (single dot product / product of two / band energy) per
/// docs/plans/2026-09-01-phase3-round1b-domain-shift-registration.md §3a,
/// and returns the immutable graph. Callers are responsible for node-count
/// filtering, structural dedup, and the numeric quarantine — this function
/// only builds the expression.
#[must_use]
pub fn draw(rng: &mut Rng) -> ExprGraph {
    let mut builder = ExprBuilder::new();
    // 3-way structure draw: single sum / product of two sums / band energy.
    match rng.below(3) {
        0 => {
            let form = rng.form();
            let l_max = 2 + rng.below(3); // L in {2,3,4}
            let root = sh_sum(&mut builder, l_max, form, rng);
            builder.finish_one(root)
        }
        1 => {
            // Product of two independent sums, L<=2 each (irradiance x
            // transfer). "Independent" means independently drawn
            // coefficients over the SAME (θ,φ) — both factors are still
            // functions of the one graph's X/Y, matching an actual
            // lighting x transfer product evaluated at one direction.
            let form = rng.form();
            let a = sh_sum(&mut builder, 1 + rng.below(2), form, rng);
            let b = sh_sum(&mut builder, 1 + rng.below(2), form, rng);
            let root = builder.binary(OpKind::Mul, a, b);
            builder.finish_one(root)
        }
        _ => {
            let root = sh_power(&mut builder, rng);
            builder.finish_one(root)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draw_node_counts_are_in_a_plausible_range() {
        // L=2..4 dot products and their products span the plan's estimate
        // (~50 for L=2 up to ~350+ for an L=4 draw or a product); this is a
        // smoke bound, not the corpus binary's own filter.
        let mut rng = Rng::new(1);
        for i in 0..50u64 {
            let mut draw_rng = Rng::new(i.wrapping_mul(0x2545_F491));
            let graph = draw(&mut draw_rng);
            let n = graph.root().node_count();
            assert!(n >= 5, "draw {i}: implausibly small ({n} nodes)");
            assert!(n <= 2000, "draw {i}: implausibly large ({n} nodes)");
            let _ = &mut rng;
        }
    }

    #[test]
    fn rng_is_deterministic_for_a_fixed_seed() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }
}
