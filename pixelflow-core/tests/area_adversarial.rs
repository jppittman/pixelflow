//! The integration rules' closed forms, attacked where their floating point
//! is thinnest, and judged by a reference that shares nothing with them.
//!
//! `area_oracle` judges the chord by clipping the pixel's polygon. This file
//! judges by a second, unrelated method — integrating the chord's clamp
//! profile piecewise-exactly in `f64`, split at its kinks — and aims at the
//! cases `area_oracle` samples sparsely or not at all:
//!
//! - **A literal slope.** `k` a constant, so the e-graph sees `K = k·h`:
//!   `0`, `-0`, far below and just above `DEGENERATE_SPAN`, and `±1e8`. The
//!   `0` found a miscompile — a provably zero sweep divided by, and the area
//!   collapsed to a constant (`mean_of_clamp`, "The divisor").
//! - **Spellings.** Coefficients that are not `±1` or `±2` (`3`, `−0.1`,
//!   `−3`), the `c = −1` root path, one-sided cuts with a rest, and
//!   redundant bounds, which the cut combines with `max`/`min`.
//! - **Empty and saturated, exactly.** A cut that misses the pixel is `0`
//!   bit for bit, and a crossing wholly beside it `0` or `σ·h` bit for bit —
//!   at `|X| ≈ 1000` and with slopes to `1e8`, never a quotient that rounded
//!   near the answer.
//! - **A clamp into any band.** `[P, Q]` other than `[0, 1]` — negative, both
//!   negative, wide — in both nestings, where `mean_of_clamp`'s `P` term and
//!   its `Q` scaling are live.
//! - **Mixed lanes.** A collapse of whole rows whose slope varies by lane,
//!   so one batch holds the saturated, degenerate and quotient arms at once,
//!   one lane's slope exactly `0`.
//! - **An interval far from zero**, where the degenerate arm's centre is the
//!   interval's midpoint and not its offset.

#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use pixelflow_core::{Kernel, Manifold, PlaneRegion, Uniform, UniformBlock};
use pixelflow_ir::{Binder, ExprArena, Fold, LatticeShape};
use pixelflow_search::runtime::unclosed_integrals;

/// `2⁻²⁴`.
const EPS: f64 = 1.0 / 16_777_216.0;
/// How many roundoffs of the terms a value may lose.
const ROUNDOFFS: f64 = 16.0;
/// Chords per family.
const SAMPLES: usize = 400;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        let unit = (self.next() >> 11) as f64 / (1u64 << 53) as f64;
        lo + (hi - lo) * unit
    }

    fn pick<T: Copy>(&mut self, from: &[T]) -> T {
        from[(self.next() % from.len() as u64) as usize]
    }
}

/// `∫_lo^hi clamp(a·t + b, p, q) dt` in `f64`, the line given as `[a, b]`:
/// cut at the two kinks, and each linear piece is its trapezoid.
fn clamp_integral([a, b]: [f64; 2], [p, q]: [f64; 2], [lo, hi]: [f64; 2]) -> f64 {
    if hi <= lo {
        return 0.0;
    }
    let value = |t: f64| (a * t + b).clamp(p, q);
    let mut cuts = vec![lo, hi];
    if a != 0.0 {
        for level in [p, q] {
            let t = (level - b) / a;
            if t > lo && t < hi {
                cuts.push(t);
            }
        }
    }
    cuts.sort_by(f64::total_cmp);
    cuts.windows(2)
        .map(|w| (w[1] - w[0]) * (value(w[0]) + value(w[1])) / 2.0)
        .sum()
}

fn indicator(mask: &Kernel) -> Kernel {
    mask.select(&Kernel::constant(1.0), &Kernel::constant(0.0))
}

fn c(v: f32) -> Kernel {
    Kernel::constant(v)
}

/// The spellings of `σ·[band]·[x < x_p(y)]` this file integrates.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Spelling {
    /// `σ·[y ≥ y₀]·[y < y₁]·[x < x_p]`.
    AsWritten,
    /// `σ·[y₀ − y ≤ 0]·[y − y₁ < 0]·[x_p − x ≥ 0]`: every root through the
    /// `c = ±1` paths, the `x` one through `c = −1`.
    UnitDifferences,
    /// `σ·[3y ≥ 3y₀]·[−0.1·y > −0.1·y₁]·[−3x + 3x_p > 0]`: literal
    /// reciprocals that round.
    OddCoefficients,
    /// `σ·[y < y₁]·[x < x_p]`: one-sided, with a rest.
    UpperOnly,
    /// `σ·[y ≥ y₀]·[x < x_p]`: one-sided, with a rest.
    LowerOnly,
    /// `σ·[y ≥ y₀]·[y ≥ y₀′]·[y < y₁]·[y < y₁′]·[x < x_p]`.
    Redundant,
}

impl Spelling {
    /// Whether every root is the bound itself, unrounded by a reciprocal —
    /// so an empty cut is exactly empty and a full one exactly full.
    fn exact_roots(self) -> bool {
        self != Self::OddCoefficients
    }
}

/// A chord's parameters as the kernel reads them.
#[derive(Clone, Copy, Debug)]
struct Chord {
    sigma: f32,
    band: [f32; 2],
    /// The redundant bounds `y₀′`, `y₁′`, read only by [`Spelling::Redundant`].
    extra: [f32; 2],
    anchor: [f32; 2],
    slope: f32,
}

impl Chord {
    /// `[lower, upper)`, the band the spelling keeps, in `f64`.
    fn rows(self, spelling: Spelling) -> [f64; 2] {
        let f = f64::from;
        let (y0, y1) = (f(self.band[0]), f(self.band[1]));
        match spelling {
            Spelling::UpperOnly => [f64::NEG_INFINITY, y1],
            Spelling::LowerOnly => [y0, f64::INFINITY],
            Spelling::Redundant => [y0.max(f(self.extra[0])), y1.min(f(self.extra[1]))],
            _ => [y0, y1],
        }
    }

    /// The band's rows inside the pixel about `y`.
    fn clipped(self, spelling: Spelling, y: f64) -> [f64; 2] {
        let [lo, hi] = self.rows(spelling);
        [lo.max(y - 0.5), hi.min(y + 0.5)]
    }

    /// The exact area over the pixel about `(x, y)`: along `y`, the covered
    /// width of the row is `clamp(x_p(y) − (x − ½), 0, 1)`, and `x_p` is
    /// affine, so it integrates piecewise-exactly in `t = y − a_y`.
    fn area(self, spelling: Spelling, [x, y]: [f64; 2]) -> f64 {
        let f = f64::from;
        let [lo, hi] = self.clipped(spelling, y);
        let (ax, ay, k) = (f(self.anchor[0]), f(self.anchor[1]), f(self.slope));
        f(self.sigma) * clamp_integral([k, ax - x + 0.5], [0.0, 1.0], [lo - ay, hi - ay])
    }

    /// `f32`'s rounding of the terms the closed form is computed from: the
    /// band's ends against the pixel, and the crossing, whose shift moves the
    /// area by at most the rows where it is not saturated.
    fn tolerance(self, spelling: Spelling, [x, y]: [f64; 2]) -> f64 {
        let f = |v: f32| f64::from(v).abs();
        let bounds = f(self.band[0]) + f(self.band[1]) + f(self.extra[0]) + f(self.extra[1]);
        let band = bounds + 2.0 * y.abs() + 2.0;
        let k = f(self.slope);
        let crossing = k * (y.abs() + f(self.anchor[1]) + 1.0) + f(self.anchor[0]) + x.abs() + 1.0;
        let [lo, hi] = self.clipped(spelling, y);
        let height = (hi - lo).max(0.0);
        let unsaturated = if k > 0.0 { height.min(1.0 / k) } else { height };
        f(self.sigma) * ROUNDOFFS * EPS * (band + crossing * unsaturated) + 8.0 * EPS
    }

    /// Whether the crossing is wholly beside the pixel across the clipped
    /// band, by a margin its rounding cannot cross: `Some(true)` right of it
    /// (every row fully covered), `Some(false)` left of it.
    fn saturated(self, spelling: Spelling, [x, y]: [f64; 2]) -> Option<bool> {
        let f = f64::from;
        let [lo, hi] = self.clipped(spelling, y);
        if hi <= lo {
            return None;
        }
        let (ax, ay, k) = (f(self.anchor[0]), f(self.anchor[1]), f(self.slope));
        let at = |row: f64| ax + (row - ay) * k;
        let (least, most) = (at(lo).min(at(hi)), at(lo).max(at(hi)));
        let margin =
            64.0 * EPS * (k.abs() * (y.abs() + ay.abs() + 1.0) + ax.abs() + x.abs() + 2.0) + 1e-3;
        if least >= x + 0.5 + margin {
            return Some(true);
        }
        if most <= x - 0.5 - margin {
            return Some(false);
        }
        None
    }

    /// `σ·h` as `f32` computes the band's height: each end clamped into the
    /// pixel after one subtraction, then one more. What a fully covered
    /// pixel must be, bit for bit.
    fn full(self, spelling: Spelling, y: f32) -> f32 {
        let (lo, hi) = match spelling {
            Spelling::UpperOnly => (None, Some(self.band[1])),
            Spelling::LowerOnly => (Some(self.band[0]), None),
            Spelling::Redundant => (
                Some(self.band[0].max(self.extra[0])),
                Some(self.band[1].min(self.extra[1])),
            ),
            _ => (Some(self.band[0]), Some(self.band[1])),
        };
        let clip = |end: Option<f32>, open: f32| end.map_or(open, |e| (e - y).clamp(-0.5, 0.5));
        let height = (clip(hi, 0.5) - clip(lo, -0.5)).max(0.0);
        self.sigma * height
    }
}

/// The chord term over uniforms, with the slope a literal or a uniform.
struct ChordTerm {
    sigma: Uniform,
    band: [Uniform; 2],
    extra: [Uniform; 2],
    anchor: [Uniform; 2],
    slope: Result<f32, Uniform>,
}

impl ChordTerm {
    fn new(literal_slope: Option<f32>) -> Self {
        let u = || Uniform::new(0.0);
        Self {
            sigma: u(),
            band: [u(), u()],
            extra: [u(), u()],
            anchor: [u(), u()],
            slope: literal_slope.ok_or_else(u),
        }
    }

    fn slope(&self) -> Kernel {
        match self.slope {
            Ok(k) => c(k),
            Err(u) => u.kernel(),
        }
    }

    fn crossing(&self) -> Kernel {
        Kernel::y()
            .sub(&self.anchor[1].kernel())
            .mul(&self.slope())
            .add(&self.anchor[0].kernel())
    }

    fn kernel(&self, spelling: Spelling) -> Kernel {
        let (x, y) = (Kernel::x(), Kernel::y());
        let (y0, y1) = (self.band[0].kernel(), self.band[1].kernel());
        let sigma = self.sigma.kernel();
        let right_of = indicator(&x.lt(&self.crossing()));
        let factors = match spelling {
            Spelling::AsWritten => vec![indicator(&y.ge(&y0)), indicator(&y.lt(&y1)), right_of],
            Spelling::UnitDifferences => vec![
                indicator(&y0.sub(&y).le(&c(0.0))),
                indicator(&y.sub(&y1).lt(&c(0.0))),
                indicator(&self.crossing().sub(&x).ge(&c(0.0))),
            ],
            Spelling::OddCoefficients => vec![
                indicator(&y.mul(&c(3.0)).ge(&y0.mul(&c(3.0)))),
                indicator(&y.mul(&c(-0.1)).gt(&y1.mul(&c(-0.1)))),
                indicator(
                    &x.mul(&c(-3.0))
                        .add(&self.crossing().mul(&c(3.0)))
                        .gt(&c(0.0)),
                ),
            ],
            Spelling::UpperOnly => vec![indicator(&y.lt(&y1)), right_of],
            Spelling::LowerOnly => vec![indicator(&y.ge(&y0)), right_of],
            Spelling::Redundant => vec![
                indicator(&y.ge(&y0)),
                indicator(&y.ge(&self.extra[0].kernel())),
                indicator(&y.lt(&y1)),
                indicator(&y.lt(&self.extra[1].kernel())),
                right_of,
            ],
        };
        factors.iter().fold(sigma, |product, f| product.mul(f))
    }

    /// Set the uniforms `spelling` reads — and only those, since a uniform
    /// the kernel never reads is not in its block.
    fn set(&self, block: &mut UniformBlock, chord: Chord, spelling: Spelling) {
        let mut values = vec![
            (self.sigma, chord.sigma),
            (self.anchor[0], chord.anchor[0]),
            (self.anchor[1], chord.anchor[1]),
        ];
        if spelling != Spelling::UpperOnly {
            values.push((self.band[0], chord.band[0]));
        }
        if spelling != Spelling::LowerOnly {
            values.push((self.band[1], chord.band[1]));
        }
        if spelling == Spelling::Redundant {
            values.push((self.extra[0], chord.extra[0]));
            values.push((self.extra[1], chord.extra[1]));
        }
        if let Err(u) = self.slope {
            values.push((u, chord.slope));
        }
        for (uniform, value) in values {
            block
                .set(uniform, value)
                .expect("a uniform the spelling reads");
        }
    }
}

/// A band about the pixel's rows `[y − ½, y + ½)`: below, above, straddling,
/// inside, reversed, touching an edge, or a single row.
fn band_about(rng: &mut Rng, y: f64) -> [f64; 2] {
    let (lo, hi) = (y - 0.5, y + 0.5);
    let mut r = |a: f64, b: f64| rng.range(a, b);
    match r(0.0, 9.0) as u32 {
        0 => [r(lo - 3.0, lo - 1.0), lo],
        1 => [hi, r(hi + 1.0, hi + 3.0)],
        2 => [r(lo - 3.0, lo), r(lo, hi)],
        3 => [r(lo, hi), r(hi, hi + 3.0)],
        4 => {
            let (a, b) = (r(lo, hi), r(lo, hi));
            [a.min(b), a.max(b)]
        }
        5 => {
            let (a, b) = (r(lo, hi), r(lo, hi));
            [a.max(b), a.min(b)]
        }
        6 => {
            let a = r(lo, hi);
            [a, a]
        }
        _ => [r(lo - 3.0, lo), r(hi, hi + 3.0)],
    }
}

fn chord_about(rng: &mut Rng, [x, y]: [f64; 2], slope: f32) -> Chord {
    let band = band_about(rng, y);
    let extra = [rng.range(y - 1.0, y + 0.5), rng.range(y - 0.5, y + 1.0)];
    // The crossing passes somewhere near the pixel's centre row, or far to
    // one side; with a steep slope, the anchor row is what places it.
    let anchor_x = x + rng.pick(&[-40.0, 40.0, 0.0, 0.0, 0.0]) + rng.range(-1.5, 1.5);
    let anchor_y = y + rng.range(-1.0, 1.0) * f64::from(slope.abs().max(1.0)).recip().max(1e-6);
    Chord {
        sigma: rng.pick(&[1.0, -1.0]),
        band: [band[0] as f32, band[1] as f32],
        extra: [extra[0] as f32, extra[1] as f32],
        anchor: [anchor_x as f32, anchor_y as f32],
        slope,
    }
}

struct Program {
    manifold: Manifold,
    block: UniformBlock,
}

impl Program {
    fn new(kernel: &Kernel) -> Self {
        let manifold = Manifold::compile(kernel, [1, 1]);
        let block = manifold.block();
        Self { manifold, block }
    }

    fn eval(&self, x: f32, y: f32) -> f32 {
        self.manifold
            .bind(&[])
            .with_uniforms(&self.block)
            .eval_at(x, y)
    }
}

fn unclosed(kernel: &Kernel) -> Option<usize> {
    let (arena, root) = kernel.parts();
    unclosed_integrals(arena, root, LatticeShape::new([1, 1]))
}

/// One evaluation against its `f64` reference.
struct Sample {
    got: f32,
    want: f64,
    tolerance: f64,
}

/// The worst error one run saw.
#[derive(Default)]
struct Worst {
    error: f64,
    ratio: f64,
    exact_pins: usize,
}

impl Worst {
    fn check(&mut self, case: &str, sample: Sample) {
        let Sample {
            got,
            want,
            tolerance,
        } = sample;
        assert!(got.is_finite(), "{case}: {got}");
        let error = (f64::from(got) - want).abs();
        assert!(
            error <= tolerance,
            "{case}: got {got:e}, f64 {want:e}, error {error:e} > tolerance {tolerance:e}"
        );
        self.error = self.error.max(error);
        self.ratio = self.ratio.max(error / tolerance);
    }
}

/// Pixel centres near the origin and near `|X|, |Y| ≈ 1000`.
fn centre(rng: &mut Rng, index: usize) -> [f32; 2] {
    let reach = if index.is_multiple_of(2) {
        16.0
    } else {
        1000.0
    };
    [
        rng.range(-reach, reach) as f32,
        rng.range(-reach, reach) as f32,
    ]
}

/// A run of chords: what it is called, which spelling it integrates, the
/// slope when it is a literal (a uniform otherwise), and its seed.
struct Family {
    label: String,
    spelling: Spelling,
    literal: Option<f32>,
    seed: u64,
}

/// `SAMPLES` chords of one family through the pixel integral, each judged
/// against its `f64` area — and, where the spelling's roots are the bounds
/// themselves, an empty cut, a crossing wholly left of the pixel and one
/// wholly right of it judged bit for bit.
fn run_chords(family: Family) {
    let Family {
        label,
        spelling,
        literal,
        seed,
    } = family;
    let term = ChordTerm::new(literal);
    let area = term.kernel(spelling).area();
    assert_eq!(
        unclosed(&area),
        Some(0),
        "{label}: an integral was left for quadrature"
    );
    let mut program = Program::new(&area);
    let mut rng = Rng(seed);
    let mut worst = Worst::default();
    for index in 0..SAMPLES {
        let [x, y] = centre(&mut rng, index);
        let at = [f64::from(x), f64::from(y)];
        let slope = literal.unwrap_or_else(|| {
            let magnitude = rng.pick(&[
                0.0, 1e-30, 1e-7, 4e-7, 1.1e-6, 3e-6, 0.3, 1.0, 4.0, 1e3, 1e6, 1e8,
            ]);
            (magnitude * rng.pick(&[1.0, -1.0])) as f32
        });
        let chord = chord_about(&mut rng, at, slope);
        term.set(&mut program.block, chord, spelling);
        let got = program.eval(x, y);
        let case = format!("{label} #{index} at ({x}, {y}) {chord:?}");
        if spelling.exact_roots() {
            let [lo, hi] = chord.clipped(spelling, at[1]);
            if hi <= lo {
                assert_eq!(got, 0.0, "{case}: an empty cut is exactly 0");
                worst.exact_pins += 1;
            }
            match chord.saturated(spelling, at) {
                Some(true) => {
                    let full = chord.full(spelling, y);
                    assert_eq!(
                        got.to_bits(),
                        full.to_bits(),
                        "{case}: full is exactly σ·h = {full}"
                    );
                    worst.exact_pins += 1;
                }
                Some(false) => {
                    assert_eq!(
                        got, 0.0,
                        "{case}: a crossing left of the pixel is exactly 0"
                    );
                    worst.exact_pins += 1;
                }
                None => {}
            }
        }
        let sample = Sample {
            got,
            want: chord.area(spelling, at),
            tolerance: chord.tolerance(spelling, at),
        };
        worst.check(&case, sample);
    }
    eprintln!(
        "area_adversarial {label}: max |error| {:.3e}, error/tolerance {:.3}, exact pins {}",
        worst.error, worst.ratio, worst.exact_pins
    );
}

/// **Every spelling, slope a uniform** spanning `0` to `1e8`.
#[test]
fn every_spelling_is_its_exact_area() {
    let spellings = [
        Spelling::AsWritten,
        Spelling::UnitDifferences,
        Spelling::OddCoefficients,
        Spelling::UpperOnly,
        Spelling::LowerOnly,
        Spelling::Redundant,
    ];
    for (i, spelling) in spellings.into_iter().enumerate() {
        run_chords(Family {
            label: format!("{spelling:?}"),
            spelling,
            literal: None,
            seed: 0xad00 + i as u64,
        });
    }
}

/// **A literal slope**, so the closed form constant-folds around `K = k·h`.
///
/// `k = 0` is the regression: a provably zero slope made the clamp moment's
/// sweep provably zero, a quotient by it let the algebra's `x·recip(x) = 1`
/// and `(x·a)/a = x` merge it with arbitrary classes, and the chord's whole
/// area extracted as the constant `σ·½·max(1 − 1, 0) = 0` — with no integral
/// left, so a closure count could not see it (`mean_of_clamp`, "The
/// divisor").
#[test]
fn a_literal_slope_is_its_exact_area() {
    let slopes: [f32; 12] = [
        0.0,
        -0.0,
        1e-30,
        -1e-7,
        // 2⁻¹⁹ and 3·2⁻¹⁸: a full-height sweep just past DEGENERATE_SPAN.
        1.907_348_6e-6,
        -1.144_409_2e-5,
        0.4,
        -3.0,
        1e6,
        1e8,
        -1e8,
        3.0e-3,
    ];
    for (i, k) in slopes.into_iter().enumerate() {
        run_chords(Family {
            label: format!("literal k = {k:e}"),
            spelling: Spelling::AsWritten,
            literal: Some(k),
            seed: 0xbe00 + i as u64,
        });
    }
}

/// `min(max(z, P), Q)` or `max(min(z, Q), P)`.
#[derive(Clone, Copy, Debug)]
enum Nesting {
    FloorFirst,
    CeilingFirst,
}

fn clamped(z: &Kernel, [p, q]: [f32; 2], nesting: Nesting) -> Kernel {
    match nesting {
        Nesting::FloorFirst => z.clamp(&c(p), &c(q)),
        Nesting::CeilingFirst => z.min(&c(q)).max(&c(p)),
    }
}

/// **A clamp into any band.** `area(clamp(a·x + b, P, Q))` for bands with a
/// live `P` term, a `Q` other than 1, both ends negative, and a wide band.
/// The inner integral is the clamp moment's; the outer integrates a value
/// its variable does not reach, which the one-point quadrature is exact on.
#[test]
fn a_clamp_into_any_band_is_its_exact_integral() {
    let bands: [[f32; 2]; 6] = [
        [-2.0, 3.0],
        [0.25, 0.75],
        [-3.0, -1.0],
        [0.0, 1000.0],
        [-0.5, 0.5],
        [-1e4, 1e-3],
    ];
    for (i, band) in bands.into_iter().enumerate() {
        for nesting in [Nesting::FloorFirst, Nesting::CeilingFirst] {
            let (slope, offset) = (Uniform::new(0.0), Uniform::new(0.0));
            let z = Kernel::x().mul(&slope.kernel()).add(&offset.kernel());
            let area = clamped(&z, band, nesting).area();
            let label = format!("clamp {band:?} {nesting:?}");
            assert_eq!(
                unclosed(&area),
                Some(1),
                "{label}: the inner integral did not close"
            );
            let mut program = Program::new(&area);
            let mut rng = Rng(0xc1a0 + i as u64);
            let mut worst = Worst::default();
            let [p, q] = [f64::from(band[0]), f64::from(band[1])];
            for index in 0..300 {
                let [x, y] = centre(&mut rng, index);
                let xf = f64::from(x);
                let magnitude = rng.pick(&[0.0, 1e-30, 1e-7, 1e-6, 3e-6, 0.3, 2.0, 40.0, 1e6, 1e8]);
                let a = (magnitude * rng.pick(&[1.0, -1.0]) * (q - p).max(1.0)) as f32;
                // The ramp crosses the band somewhere near the pixel, or
                // sits wholly above or below it there.
                let beyond = [p - 1.0 - (q - p) / 2.0, q + 1.0 + (q - p) / 2.0, p, q];
                let level = if rng.next().is_multiple_of(2) {
                    rng.range(p, q)
                } else {
                    rng.pick(&beyond)
                };
                let b = (level - f64::from(a) * (xf + rng.range(-1.0, 1.0))) as f32;
                program.block.set(slope, a).expect("slope");
                program.block.set(offset, b).expect("offset");
                let got = program.eval(x, y);
                let (a64, b64) = (f64::from(a), f64::from(b));
                let want = clamp_integral([a64, a64 * xf + b64], [p, q], [-0.5, 0.5]);
                let reach = a64.abs() * (xf.abs() + 1.0) + b64.abs() + p.abs() + q.abs() + 1.0;
                let spread = if a64.abs() > q - p {
                    (q - p) / a64.abs()
                } else {
                    1.0
                };
                let tolerance = ROUNDOFFS * EPS * (reach * spread + p.abs().max(q.abs()))
                    + (q - p) * EPS
                    + 8.0 * EPS;
                let case = format!("{label} #{index} at {x}: a = {a:e}, b = {b:e}");
                // Wholly above or below the band across the pixel: exactly
                // Q or P, decided before any quotient.
                let ends = [a64 * (xf - 0.5) + b64, a64 * (xf + 0.5) + b64];
                let margin = 64.0 * EPS * reach + 1e-6;
                if ends.iter().all(|&e| e >= q + margin) {
                    assert_eq!(got.to_bits(), band[1].to_bits(), "{case}: saturated high");
                    worst.exact_pins += 1;
                }
                if ends.iter().all(|&e| e <= p - margin) {
                    assert_eq!(got.to_bits(), band[0].to_bits(), "{case}: saturated low");
                    worst.exact_pins += 1;
                }
                let sample = Sample {
                    got,
                    want,
                    tolerance,
                };
                worst.check(&case, sample);
            }
            eprintln!(
                "area_adversarial {label}: max |error| {:.3e}, error/tolerance {:.3}, exact pins {}",
                worst.error, worst.ratio, worst.exact_pins
            );
        }
    }
}

/// The `Var` kernel binder `slot` is read through.
fn binder(slot: u8) -> Kernel {
    let mut a = ExprArena::new();
    let v = a.push_var(Binder::from_slot(slot).expect("a live slot").var());
    Kernel::from_parts(a, v)
}

/// `∫_lo^hi body(u) du` binding slot 0, built through `Fold::from_bits`'s
/// documented layout.
fn integral_over([lo, hi]: [f32; 2], body: impl FnOnce(&Kernel) -> Kernel) -> Kernel {
    let integrand = body(&binder(0));
    let (arena, root) = integrand.parts();
    let bits = 1u128 << 112 | u128::from(lo.to_bits()) << 32 | u128::from(hi.to_bits());
    let fold = Fold::from_bits(bits).expect("a finite, nonempty interval");
    let mut a = arena.clone();
    let root = a.push_reduce(fold, root);
    Kernel::from_parts(a, root)
}

/// **Mixed lanes, and an interval far from zero.**
/// `∫_lo^hi clamp(s·(X − 3.5)·u + b, P, Q) du`, collapsed over whole rows:
/// the slope varies by lane and is exactly `0` in lane 3, so one batch holds
/// saturated lanes, lanes in the degenerate arm and lanes in the quotient
/// arm. Lane 3's zero is the one that once collapsed a whole kernel
/// (`a_literal_slope_is_its_exact_area`), but lane-varying, so no rule can
/// prove it: this pins the arms, that test the e-graph. Over `[1000, 1002)`
/// the degenerate arm must take the clamp at the interval's midpoint
/// `u = 1001` — at `u = 0` it would be off by `1001·|slope|`, far outside
/// the tolerance.
#[test]
fn mixed_lanes_and_a_far_interval_are_exact_integrals() {
    const WIDTH: usize = 16;
    let intervals: [[f32; 2]; 3] = [[-0.5, 0.5], [1000.0, 1002.0], [-7.25, -3.0]];
    let bands: [[f32; 2]; 2] = [[0.0, 1.0], [-2.0, 3.0]];
    let scales: [f32; 7] = [0.0, 2.4e-7, 1.0 / 4_194_304.0, 1e-6, 0.37, 5.0, 1e3];
    for interval in intervals {
        for band in bands {
            let (scale, offset) = (Uniform::new(0.0), Uniform::new(0.0));
            let slope = Kernel::x().sub(&c(3.5)).mul(&scale.kernel());
            let kernel = integral_over(interval, |u| {
                clamped(
                    &slope.mul(u).add(&offset.kernel()),
                    band,
                    Nesting::FloorFirst,
                )
            });
            let label = format!("lanes over {interval:?} into {band:?}");
            assert_eq!(unclosed(&kernel), Some(0), "{label}: not closed");
            let manifold = Manifold::compile(&kernel, [WIDTH as u32, 1]);
            let mut block = manifold.block();
            let (lo, hi) = (f64::from(interval[0]), f64::from(interval[1]));
            let mid = (lo + hi) / 2.0;
            let [p, q] = [f64::from(band[0]), f64::from(band[1])];
            let mut worst = Worst::default();
            for s in scales {
                for level in [p - 0.5, p, (p + q) / 2.0, q - 1e-4, q, q + 7.0] {
                    // Centre the ramp's midpoint value at `level`, lane by
                    // lane only through the slope.
                    let b = level as f32;
                    block.set(scale, s).expect("scale");
                    block.set(offset, b).expect("offset");
                    let mut out = [f32::NAN; WIDTH];
                    manifold.bind(&[]).with_uniforms(&block).collapse_rows(
                        PlaneRegion::rows(WIDTH, 0, 1),
                        &mut out,
                        WIDTH,
                    );
                    for (lane, &got) in out.iter().enumerate() {
                        let x = lane as f32 + 0.5;
                        let a = (x - 3.5) * s;
                        let (a64, b64) = (f64::from(a), f64::from(b));
                        let want = clamp_integral([a64, b64], [p, q], [lo, hi]);
                        // The slope's own terms, not the slope: the e-graph
                        // is free to distribute `s·(X − 3.5)·u` into
                        // `s·X·u − 3.5·s·u` (it hoists the lane-invariant
                        // half), so a lane whose slope is exactly 0 still
                        // rounds at `|s|·(|X| + 3.5)`.
                        let terms = f64::from(s).abs() * (f64::from(x) + 3.5);
                        let reach = terms * (lo.abs().max(hi.abs()) + 1.0) + b64.abs() + 1.0;
                        let spread = if a64.abs() * (hi - lo) > q - p {
                            (q - p) / (a64.abs() * (hi - lo))
                        } else {
                            1.0
                        };
                        let tolerance = (hi - lo)
                            * (ROUNDOFFS * EPS * (reach * spread + p.abs().max(q.abs()))
                                + (q - p) * EPS)
                            + 8.0 * EPS;
                        let case = format!(
                            "{label}: s = {s:e}, b = {b}, lane {lane} (slope {a:e}, mid {mid})"
                        );
                        let sample = Sample {
                            got,
                            want,
                            tolerance,
                        };
                        worst.check(&case, sample);
                    }
                }
            }
            eprintln!(
                "area_adversarial {label}: max |error| {:.3e}, error/tolerance {:.3}",
                worst.error, worst.ratio
            );
        }
    }
}

/// **Rows of chords, collapsed whole.** The chord's area over a `16 × 4`
/// band of pixels translated to `|X|, |Y| ≈ 1000`, so each batch mixes
/// lanes left of, on and right of the crossing — and rows whose band misses
/// them entirely.
#[test]
fn collapsed_rows_of_chords_are_exact_areas() {
    const WIDTH: usize = 16;
    const ROWS: usize = 4;
    for (i, &origin) in [[0.0f32, 0.0], [1000.0, -777.0], [-996.0, 1003.0]]
        .iter()
        .enumerate()
    {
        let term = ChordTerm::new(None);
        let area = term.kernel(Spelling::AsWritten).area().at(
            &Kernel::x().add(&c(origin[0])),
            &Kernel::y().add(&c(origin[1])),
        );
        assert_eq!(
            unclosed(&area),
            Some(0),
            "rows about {origin:?}: not closed"
        );
        let manifold = Manifold::compile(&area, [WIDTH as u32, ROWS as u32]);
        let mut block = manifold.block();
        let mut rng = Rng(0xd000 + i as u64);
        let mut worst = Worst::default();
        for _ in 0..60 {
            let slope = (rng.pick(&[0.0, 1e-7, 2e-6, 0.25, 1.0, 3.0, 40.0, 1e8])
                * rng.pick(&[1.0, -1.0])) as f32;
            let centre_y = f64::from(origin[1]) + rng.range(0.0, ROWS as f64);
            let x_mid = f64::from(origin[0]) + rng.range(0.0, WIDTH as f64);
            let band = [
                (centre_y - rng.range(0.0, 3.0)) as f32,
                (centre_y + rng.range(-1.0, 3.0)) as f32,
            ];
            let chord = Chord {
                sigma: rng.pick(&[1.0, -1.0]),
                band,
                extra: [0.0, 0.0],
                anchor: [x_mid as f32, centre_y as f32],
                slope,
            };
            term.set(&mut block, chord, Spelling::AsWritten);
            let mut out = [f32::NAN; WIDTH * ROWS];
            manifold.bind(&[]).with_uniforms(&block).collapse_rows(
                PlaneRegion::rows(WIDTH, 0, ROWS),
                &mut out,
                WIDTH,
            );
            for row in 0..ROWS {
                for col in 0..WIDTH {
                    let got = out[row * WIDTH + col];
                    let at = [
                        f64::from(origin[0]) + col as f64 + 0.5,
                        f64::from(origin[1]) + row as f64 + 0.5,
                    ];
                    let case = format!("rows about {origin:?}: ({col}, {row}) {chord:?}");
                    let want = chord.area(Spelling::AsWritten, at);
                    let sample = Sample {
                        got,
                        want,
                        tolerance: chord.tolerance(Spelling::AsWritten, at),
                    };
                    worst.check(&case, sample);
                }
            }
        }
        eprintln!(
            "area_adversarial rows about {origin:?}: max |error| {:.3e}, error/tolerance {:.3}",
            worst.error, worst.ratio
        );
    }
}
