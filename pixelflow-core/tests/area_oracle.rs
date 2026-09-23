//! Integrals the e-graph closes, judged by exact areas computed in `f64` by
//! a method that shares nothing with pixelflow.
//!
//! Step 4 of docs/plans/2026-09-23-an-integral-is-a-fold.md §8: the chord a
//! glyph piece contributes, `σ·[y ≥ y₀]·[y < y₁]·[x < a_x + (y − a_y)·k]`,
//! integrated over a pixel, is closed by the integration rules —
//! `FactorFold`, `NarrowInterval` twice and `ClampMoment` — to an exact
//! formula, and a clamp `clamp(a·x + b, 0, 1)` by `ClampMoment` alone. This
//! file checks three things about that:
//!
//! - **The value.** Every kernel is compiled once, through the production
//!   route (`Manifold::compile` → the jit cache → saturation, extraction,
//!   `legalize`), over *uniforms*, and evaluated at seeded random pixels and
//!   parameters. The reference is the area of the region intersected with
//!   the pixel, computed by Sutherland–Hodgman clipping of the pixel's
//!   rectangle by the region's half-planes and the shoelace formula — a
//!   polygon clip, not an antiderivative, so a shared-definition bug in the
//!   closed form cannot also be in the judge (CLAUDE.md, "a same-form check
//!   cannot see a shared-definition bug"). The clamp's reference integrates
//!   its linear pieces between their kinks. Masks never become numbers:
//!   every indicator is `select(mask, 1, 0)`.
//! - **That the closed form is what was judged.** `unclosed_integrals`
//!   counts the interval folds the extraction kept; `0` means the texels
//!   came from the rules, not from the one-point quadrature that
//!   legalizes a survivor — which would also pass the pins that sample a
//!   pixel's centre.
//! - **That no integral reaches the emitter.** Each kernel's arena, raw and
//!   optimized, holds no reachable interval fold after `legalize`.
//!
//! **The tolerance** is `f32`'s rounding of the terms the closed form is
//! computed from, not of its result (critique of the step-4 design, A5): at
//! `x ≈ 1000` the argument of the clamp has already lost everything below
//! `2⁻²⁴·1000`. For a region clipped to band height `h` in a rectangle of
//! width `L`,
//!
//! ```text
//! tol = 8·2⁻²⁴·|σ|·(L·M_h + h·M_z·min(1, L/(|k|·h)))
//! ```
//!
//! `M_h` the band's terms (`|y₀| + |y₁| + 2|Y|` and the rectangle), `M_z`
//! the clamp argument's (`(|Y| + |a_y| + H)·|k| + |a_x| + |X| + L`). The last
//! factor is the clamp's own: when the argument sweeps a span wider than the
//! band, a shift of every sample by `δ` moves the mean by at most
//! `δ·L/(|k|·h)`. The bound is capped by the trivial one, since the mean of
//! a clamp into `[0, L]` lies in it.

#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use pixelflow_core::{Kernel, Manifold, Uniform, UniformBlock};
use pixelflow_ir::passes::lattice::{Collapse, Domain};
use pixelflow_ir::{Binder, ExprArena, ExprId, ExprNode, Fold, LatticeShape, OpKind};
use pixelflow_search::runtime::{optimize_runtime_arena, unclosed_integrals};

/// `2⁻²⁴`, `f32`'s unit roundoff.
const ROUNDOFF: f64 = 1.0 / 16_777_216.0;
/// How many roundoffs of the terms the closed form may lose.
const ROUNDOFFS: f64 = 8.0;
/// Samples per category.
const SAMPLES: usize = 400;
/// How far from the origin a pixel may be.
const FAR: f64 = 1000.0;

/// splitmix64: a seeded, dependency-free generator, so a failure names its
/// case by seed and index.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `[lo, hi)`.
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        let unit = (self.next() >> 11) as f64 / (1u64 << 53) as f64;
        lo + (hi - lo) * unit
    }

    fn pick<T: Copy>(&mut self, from: &[T]) -> T {
        from[(self.next() % from.len() as u64) as usize]
    }

    fn sign(&mut self) -> f64 {
        self.pick(&[-1.0, 1.0])
    }
}

/// A point.
type Point = [f64; 2];

/// Clip a convex polygon to `{p : inside(p) ≤ 0}`, `inside` affine.
fn clip(polygon: &[Point], inside: impl Fn(Point) -> f64) -> Vec<Point> {
    let mut out = Vec::new();
    for (i, &p) in polygon.iter().enumerate() {
        let q = polygon[(i + 1) % polygon.len()];
        let (dp, dq) = (inside(p), inside(q));
        if dp <= 0.0 {
            out.push(p);
        }
        if (dp < 0.0 && dq > 0.0) || (dp > 0.0 && dq < 0.0) {
            let t = dp / (dp - dq);
            out.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]);
        }
    }
    out
}

/// The shoelace area of a simple polygon, counter-clockwise positive.
fn shoelace(polygon: &[Point]) -> f64 {
    let n = polygon.len();
    (0..n)
        .map(|i| {
            let (p, q) = (polygon[i], polygon[(i + 1) % n]);
            p[0] * q[1] - q[0] * p[1]
        })
        .sum::<f64>()
        / 2.0
}

/// An axis-aligned rectangle: `[x_lo, x_hi) × [y_lo, y_hi)`.
#[derive(Clone, Copy, Debug)]
struct Rect {
    x: [f64; 2],
    y: [f64; 2],
}

impl Rect {
    /// The pixel `Kernel::area` integrates about `(x, y)`.
    fn pixel(x: f64, y: f64) -> Self {
        Self::about(x, y, Offsets::PIXEL)
    }

    fn about(x: f64, y: f64, offsets: Offsets) -> Self {
        Self {
            x: [x + offsets.x[0], x + offsets.x[1]],
            y: [y + offsets.y[0], y + offsets.y[1]],
        }
    }

    fn corners(self) -> Vec<Point> {
        vec![
            [self.x[0], self.y[0]],
            [self.x[1], self.y[0]],
            [self.x[1], self.y[1]],
            [self.x[0], self.y[1]],
        ]
    }

    fn width(self) -> f64 {
        self.x[1] - self.x[0]
    }

    fn height(self) -> f64 {
        self.y[1] - self.y[0]
    }
}

/// An integral's intervals, as offsets from the sample: `x` for the inner
/// variable, `y` for the outer.
#[derive(Clone, Copy, Debug)]
struct Offsets {
    x: [f64; 2],
    y: [f64; 2],
}

impl Offsets {
    const PIXEL: Self = Self {
        x: [-0.5, 0.5],
        y: [-0.5, 0.5],
    };
}

/// A chord's parameters exactly as the kernel reads them — `f32`, so the
/// reference integrates the region the kernel was given, not the one the
/// generator meant.
#[derive(Clone, Copy, Debug)]
struct Chord {
    sigma: f32,
    band: [f32; 2],
    anchor: [f32; 2],
    slope: f32,
}

impl Chord {
    fn f64(v: f32) -> f64 {
        f64::from(v)
    }

    /// `x_p(y) = a_x + (y − a_y)·k`.
    fn crossing(self, y: f64) -> f64 {
        Self::f64(self.anchor[0]) + (y - Self::f64(self.anchor[1])) * Self::f64(self.slope)
    }

    /// The exact signed area of `σ·[y₀ ≤ y < y₁]·[x < x_p(y)]` over `rect`.
    fn area(self, rect: Rect) -> f64 {
        let (y0, y1) = (Self::f64(self.band[0]), Self::f64(self.band[1]));
        let region = clip(&rect.corners(), |p| y0 - p[1]);
        let region = clip(&region, |p| p[1] - y1);
        let region = clip(&region, |p| p[0] - self.crossing(p[1]));
        Self::f64(self.sigma) * shoelace(&region).abs()
    }

    /// The band's height inside `rect`.
    fn height_in(self, rect: Rect) -> f64 {
        let (y0, y1) = (Self::f64(self.band[0]), Self::f64(self.band[1]));
        (y1.min(rect.y[1]) - y0.max(rect.y[0])).max(0.0)
    }

    /// See the module doc.
    fn tolerance(self, rect: Rect, centre: Point) -> f64 {
        let f = Self::f64;
        let (x, y) = (centre[0].abs(), centre[1].abs());
        let (width, height) = (rect.width(), rect.height());
        let m_h = f(self.band[0]).abs() + f(self.band[1]).abs() + 2.0 * y + height + 1.0;
        let m_z = (y + f(self.anchor[1]).abs() + height) * f(self.slope).abs()
            + f(self.anchor[0]).abs()
            + x
            + width
            + 1.0;
        let h = self.height_in(rect);
        let sweep = f(self.slope).abs() * h;
        let spread = if sweep > width { width / sweep } else { 1.0 };
        let clamp_term = (h * ROUNDOFFS * ROUNDOFF * m_z * spread).min(h * width);
        f(self.sigma).abs() * (ROUNDOFFS * ROUNDOFF * width * m_h + clamp_term)
    }
}

/// The chord term over uniforms: one structure, compiled once, every chord
/// a call.
struct ChordTerm {
    sigma: Uniform,
    band: [Uniform; 2],
    anchor: [Uniform; 2],
    slope: Uniform,
}

impl ChordTerm {
    fn new() -> Self {
        let u = || Uniform::new(0.0);
        Self {
            sigma: u(),
            band: [u(), u()],
            anchor: [u(), u()],
            slope: u(),
        }
    }

    /// `a_x + (y − a_y)·k`.
    fn crossing(&self) -> Kernel {
        Kernel::y()
            .sub(&self.anchor[1].kernel())
            .mul(&self.slope.kernel())
            .add(&self.anchor[0].kernel())
    }

    /// `σ·[y ≥ y₀]·[y < y₁]·[x < a_x + (y − a_y)·k]`.
    fn kernel(&self) -> Kernel {
        let (x, y) = (Kernel::x(), Kernel::y());
        self.sigma
            .kernel()
            .mul(&indicator(&y.ge(&self.band[0].kernel())))
            .mul(&indicator(&y.lt(&self.band[1].kernel())))
            .mul(&indicator(&x.lt(&self.crossing())))
    }

    /// The same region spelled with every comparison's difference sloping
    /// *down* in its variable: `σ·[y₀ ≤ y]·[y₁ > y]·[2·x_p(y) > 2·x]`. Each
    /// bound is now read through a negative slope — `−1` twice, `−2` once —
    /// so the side it keeps is the one the sign flips it to.
    fn flipped(&self) -> Kernel {
        let (x, y, two) = (Kernel::x(), Kernel::y(), Kernel::constant(2.0));
        self.sigma
            .kernel()
            .mul(&indicator(&self.band[0].kernel().le(&y)))
            .mul(&indicator(&self.band[1].kernel().gt(&y)))
            .mul(&indicator(&self.crossing().mul(&two).gt(&x.mul(&two))))
    }

    fn set(&self, block: &mut UniformBlock, chord: Chord) {
        let values = [
            (self.sigma, chord.sigma),
            (self.band[0], chord.band[0]),
            (self.band[1], chord.band[1]),
            (self.anchor[0], chord.anchor[0]),
            (self.anchor[1], chord.anchor[1]),
            (self.slope, chord.slope),
        ];
        for (uniform, value) in values {
            block.set(uniform, value).expect("an argument of the chord");
        }
    }
}

/// `[mask]`: the one place a mask becomes a number.
fn indicator(mask: &Kernel) -> Kernel {
    mask.select(&Kernel::constant(1.0), &Kernel::constant(0.0))
}

/// The `Var` kernel binder `slot` is read through.
fn binder(slot: u8) -> Kernel {
    let mut a = ExprArena::new();
    let v = a.push_var(Binder::from_slot(slot).expect("a live slot").var());
    Kernel::from_parts(a, v)
}

/// `∫_lo^hi` binding `slot`, through `Fold::from_bits`'s documented layout —
/// `Kernel::area` builds only the centred pixel.
fn interval(slot: u8, [lo, hi]: [f64; 2]) -> Fold {
    let bits = 1u128 << 112
        | u128::from(slot) << 64
        | u128::from((lo as f32).to_bits()) << 32
        | u128::from((hi as f32).to_bits());
    Fold::from_bits(bits).expect("a finite, nonempty interval")
}

/// `∫_{v ∈ y} ∫_{u ∈ x} k(X + u, Y + v) du dv`: `Kernel::area` over any
/// rectangle about the sample.
fn integrated(k: &Kernel, offsets: Offsets) -> Kernel {
    let body = k.at(&Kernel::x().add(&binder(0)), &Kernel::y().add(&binder(1)));
    let (arena, root) = body.parts();
    let mut a = arena.clone();
    let inner = a.push_reduce(interval(0, offsets.x), root);
    let outer = a.push_reduce(interval(1, offsets.y), inner);
    Kernel::from_parts(a, outer)
}

/// A kernel compiled once, through the production route, and evaluated at
/// single points under a uniform block.
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

    fn eval(&self, x: f32, y: f32) -> f64 {
        f64::from(
            self.manifold
                .bind(&[])
                .with_uniforms(&self.block)
                .eval_at(x, y),
        )
    }
}

/// One evaluation against its reference.
struct Sample {
    got: f64,
    want: f64,
    tolerance: f64,
    /// The pixel it was taken at.
    centre: [f32; 2],
}

/// The largest error a category saw, absolute and against its tolerance,
/// for pixels near the origin and far from it.
#[derive(Default, Debug)]
struct Worst {
    near: [f64; 2],
    far: [f64; 2],
}

impl Worst {
    fn record(&mut self, name: &str, sample: Sample, case: &dyn core::fmt::Debug) {
        let Sample {
            got,
            want,
            tolerance,
            centre,
        } = sample;
        let error = (got - want).abs();
        assert!(
            error <= tolerance,
            "{name}: {case:?}: got {got}, f64 area {want}, error {error:e} > tolerance {tolerance:e}"
        );
        let reach = centre[0].abs().max(centre[1].abs());
        let slot = if reach <= NEAR {
            &mut self.near
        } else {
            &mut self.far
        };
        slot[0] = slot[0].max(error);
        slot[1] = slot[1].max(error / tolerance);
    }

    fn report(&self, name: &str) {
        eprintln!(
            "area_oracle {name}: |X|,|Y| ≤ {NEAR}: max |error| {:.3e} (error/tolerance {:.3}); \
             farther: max |error| {:.3e} (error/tolerance {:.3})",
            self.near[0], self.near[1], self.far[0], self.far[1]
        );
    }
}

/// Where "near the origin" ends, for [`Worst`]'s report and [`centre`].
const NEAR: f32 = 16.0;

/// Where a band lies against the rectangle's `[lo, hi)` rows.
fn band_about(rng: &mut Rng, rows: [f64; 2]) -> [f64; 2] {
    let [lo, hi] = rows;
    let (below, inside, above) = (
        rng.range(lo - 3.0, lo),
        rng.range(lo, hi),
        rng.range(hi, hi + 3.0),
    );
    let inside_too = rng.range(lo, hi);
    match rng.next() % 6 {
        0 => [below, rng.range(below, lo)],
        1 => [above, rng.range(above, hi + 4.0)],
        2 => [below, inside],
        3 => [inside, above],
        4 => [inside.min(inside_too), inside.max(inside_too)],
        _ => [below, above],
    }
}

/// A random chord about the centre of `rect`, with slope from `slopes`.
fn chord_about(rng: &mut Rng, rect: Rect, slope: f64) -> Chord {
    let centre = [(rect.x[0] + rect.x[1]) / 2.0, (rect.y[0] + rect.y[1]) / 2.0];
    let band = band_about(rng, rect.y);
    Chord {
        sigma: rng.sign() as f32,
        band: [band[0] as f32, band[1] as f32],
        anchor: [
            (centre[0] + rng.range(-3.0, 3.0)) as f32,
            (centre[1] + rng.range(-3.0, 3.0)) as f32,
        ],
        slope: slope as f32,
    }
}

/// A pixel centre, near the origin for half the samples (tight tolerances)
/// and anywhere within [`FAR`] of it for the rest.
fn centre(rng: &mut Rng, index: usize) -> [f32; 2] {
    let reach = if index.is_multiple_of(2) {
        f64::from(NEAR)
    } else {
        FAR
    };
    [
        rng.range(-reach, reach) as f32,
        rng.range(-reach, reach) as f32,
    ]
}

/// Whether a node `is` names is reachable from `root`.
fn reaches(arena: &ExprArena, root: ExprId, is: impl Fn(ExprNode) -> bool) -> bool {
    let mut seen = vec![false; arena.len()];
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if std::mem::replace(&mut seen[id.0 as usize], true) {
            continue;
        }
        if is(arena.node(id)) {
            return true;
        }
        stack.extend(arena.children(id));
    }
    false
}

/// Whether an interval fold is reachable from `root`.
fn reaches_interval(arena: &ExprArena, root: ExprId) -> bool {
    reaches(arena, root, |node| {
        matches!(
            node,
            ExprNode::Reduce {
                fold: Fold::Interval(_),
                ..
            }
        )
    })
}

/// **(c) No integral reaches the emitter.** `legalize` — what
/// `emit::compile` runs before it schedules — leaves no interval fold in
/// `kernel`'s arena, as built or as the runtime tier optimized it; and
/// `unclosed` is how many the optimized extraction kept for it to replace.
fn assert_legal(name: &str, kernel: &Kernel, unclosed: usize) {
    let (arena, root) = kernel.parts();
    let shape = LatticeShape::new([1, 1]);
    assert_eq!(
        unclosed_integrals(arena, root, shape),
        Some(unclosed),
        "{name}: integrals the extraction left unclosed"
    );
    let optimized =
        optimize_runtime_arena(arena, root, shape).expect("the runtime tier optimizes an integral");
    // A closed form divides by literals and by one span, and an estimate
    // there costs 2⁻¹² of the area: none may be extracted.
    assert!(
        !reaches(&optimized.0, optimized.1, |node| matches!(
            node,
            ExprNode::Unary(OpKind::Recip | OpKind::Rsqrt, _)
        )),
        "{name}: the optimized term holds a reciprocal estimate"
    );
    let collapse = Collapse {
        domain: Domain {
            shape,
            origin: [Uniform::new(0.0).decl(), Uniform::new(0.0).decl()],
        },
        lanes: 4,
    };
    for (route, (arena, root)) in [
        ("as built", (arena, root)),
        ("optimized", (&optimized.0, optimized.1)),
    ] {
        let (legal, legal_root) =
            pixelflow_ir::passes::legalize(arena, root, &collapse).expect("legalize");
        assert!(
            !reaches_interval(&legal, legal_root),
            "{name} ({route}): an integral survived legalize"
        );
    }
}

/// Which spelling of the chord a category integrates.
#[derive(Clone, Copy)]
enum Spelling {
    /// [`ChordTerm::kernel`].
    AsWritten,
    /// [`ChordTerm::flipped`].
    Flipped,
}

/// A category of chords: its name in reports and failures, its seed, and
/// which spelling of the chord it integrates.
struct Family {
    name: &'static str,
    seed: u64,
    spelling: Spelling,
}

/// Run `SAMPLES` chords of one slope family, as written, through the pixel
/// integral.
fn chords_over_the_pixel(name: &'static str, seed: u64, slope: impl Fn(&mut Rng) -> f64) {
    chords(
        Family {
            name,
            seed,
            spelling: Spelling::AsWritten,
        },
        slope,
    );
}

/// Run `SAMPLES` chords of `family` through the pixel integral.
fn chords(family: Family, slope: impl Fn(&mut Rng) -> f64) {
    let Family {
        name,
        seed,
        spelling,
    } = family;
    let term = ChordTerm::new();
    let area = match spelling {
        Spelling::AsWritten => term.kernel(),
        Spelling::Flipped => term.flipped(),
    }
    .area();
    assert_legal(name, &area, 0);
    let mut program = Program::new(&area);
    let mut rng = Rng(seed);
    let mut worst = Worst::default();
    for index in 0..SAMPLES {
        let [x, y] = centre(&mut rng, index);
        let rect = Rect::pixel(f64::from(x), f64::from(y));
        let k = slope(&mut rng);
        let chord = chord_about(&mut rng, rect, k);
        term.set(&mut program.block, chord);
        let got = program.eval(x, y);
        let centre = [f64::from(x), f64::from(y)];
        let sample = Sample {
            got,
            want: chord.area(rect),
            tolerance: chord.tolerance(rect, centre),
            centre: [x, y],
        };
        worst.record(name, sample, &(index, x, y, chord));
    }
    worst.report(name);
}

/// **(a) A slanted chord**, `k` uniform in `[−4, 4]`.
#[test]
fn a_slanted_chord_is_its_exact_area() {
    chords_over_the_pixel("slanted", 0x5eed_0001, |rng| rng.range(-4.0, 4.0));
}

/// **(a) A vertical chord**, `k = 0`: the clamp moment's degenerate arm.
#[test]
fn a_vertical_chord_is_its_exact_area() {
    chords_over_the_pixel("vertical", 0x5eed_0002, |_| 0.0);
}

/// **(a) A nearly vertical chord**, `k ∈ {±1e-7, ±1e-30}`: a sweep far
/// below `DEGENERATE_SPAN`.
#[test]
fn a_nearly_vertical_chord_is_its_exact_area() {
    chords_over_the_pixel("near-vertical", 0x5eed_0003, |rng| {
        rng.sign() * rng.pick(&[1.0e-7, 1.0e-30])
    });
}

/// **(a) A nearly horizontal chord**, `k ∈ {±1e6, ±1e8}`: a sweep so wide
/// the clamp saturates almost everywhere.
#[test]
fn a_nearly_horizontal_chord_is_its_exact_area() {
    chords_over_the_pixel("near-horizontal", 0x5eed_0004, |rng| {
        rng.sign() * rng.pick(&[1.0e6, 1.0e8])
    });
}

/// **(a) A chord whose sweep straddles `DEGENERATE_SPAN`**: `|k|` log-uniform
/// in `[2⁻²⁴, 2⁻⁸]`, so `h·|k|` lands on both sides of the degenerate arm's
/// threshold — where the difference quotient divides the least.
#[test]
fn a_chord_near_the_degenerate_span_is_its_exact_area() {
    chords_over_the_pixel("near-degenerate", 0x5eed_0005, |rng| {
        rng.sign() * rng.range(-24.0, -8.0).exp2()
    });
}

/// **(a) Every comparison flipped.** The chord spelled so that each
/// indicator's difference slopes down in its variable (`−1`, `−1`, `−2`):
/// the same area, reached only if a negative slope flips the bound's side.
#[test]
fn a_chord_spelled_with_falling_differences_is_its_exact_area() {
    let flipped = Family {
        name: "flipped",
        seed: 0x5eed_000a,
        spelling: Spelling::Flipped,
    };
    chords(flipped, |rng| rng.range(-4.0, 4.0));
}

/// **(a) Off the pixel.** The same chord integrated over a rectangle that is
/// neither centred nor of unit size, where `narrowing`'s Jacobian divides
/// by the length and its point is shifted by the midpoint — the pixel's
/// identities (length 1, midpoint 0) would hide either being dropped.
#[test]
fn a_chord_over_an_asymmetric_rectangle_is_its_exact_area() {
    let offsets = Offsets {
        x: [-2.0, 0.5],
        y: [1.0, 3.0],
    };
    let term = ChordTerm::new();
    let integral = integrated(&term.kernel(), offsets);
    assert_legal("asymmetric", &integral, 0);
    let mut program = Program::new(&integral);
    let mut rng = Rng(0x5eed_0006);
    let mut worst = Worst::default();
    for index in 0..SAMPLES {
        let [x, y] = centre(&mut rng, index);
        let rect = Rect::about(f64::from(x), f64::from(y), offsets);
        let k = rng.range(-4.0, 4.0);
        let chord = chord_about(&mut rng, rect, k);
        term.set(&mut program.block, chord);
        let got = program.eval(x, y);
        let centre = [f64::from(x), f64::from(y)];
        let sample = Sample {
            got,
            want: chord.area(rect),
            tolerance: chord.tolerance(rect, centre),
            centre: [x, y],
        };
        worst.record("asymmetric", sample, &(index, x, y, chord));
    }
    worst.report("asymmetric");
}

/// `∫_lo^hi clamp(a·s + b, 0, 1) ds`, exactly: split at the kinks, and
/// integrate each linear piece by its trapezoid.
fn clamp_integral(a: f64, b: f64, [lo, hi]: [f64; 2]) -> f64 {
    let value = |s: f64| (a * s + b).clamp(0.0, 1.0);
    let mut cuts = vec![lo, hi];
    if a != 0.0 {
        for level in [0.0, 1.0] {
            let s = (level - b) / a;
            if s > lo && s < hi {
                cuts.push(s);
            }
        }
    }
    cuts.sort_by(f64::total_cmp);
    cuts.windows(2)
        .map(|w| (w[1] - w[0]) * (value(w[0]) + value(w[1])) / 2.0)
        .sum()
}

/// **(a) A clamp integrand.** `area(clamp(a·x + b, 0, 1))`: the inner
/// integral is `ClampMoment`'s, and the outer integrates a value its
/// variable does not reach — which the one-point quadrature computes
/// exactly, so it is left to legalization (no rule in this change closes a
/// constant integrand). `a` spans slanted, vertical-ramp and steep.
#[test]
fn a_clamp_integrand_is_its_exact_integral() {
    let (slope, offset) = (Uniform::new(0.0), Uniform::new(0.0));
    let ramp = Kernel::x()
        .mul(&slope.kernel())
        .add(&offset.kernel())
        .clamp(&Kernel::constant(0.0), &Kernel::constant(1.0));
    let area = ramp.area();
    assert_legal("clamp", &area, 1);
    let mut program = Program::new(&area);
    let mut rng = Rng(0x5eed_0007);
    let mut worst = Worst::default();
    for index in 0..SAMPLES {
        let [x, y] = centre(&mut rng, index);
        let a = match index % 4 {
            0 => 0.0,
            1 => rng.sign() * 1.0e-7,
            2 => rng.sign() * 1.0e6,
            _ => rng.range(-4.0, 4.0),
        } as f32;
        // The ramp crosses somewhere near the pixel, or nowhere near it.
        let crossing = f64::from(x) + rng.range(-2.0, 2.0);
        let b = (-f64::from(a) * crossing + rng.range(-1.0, 2.0)) as f32;
        program.block.set(slope, a).expect("an argument");
        program.block.set(offset, b).expect("an argument");
        let got = program.eval(x, y);
        let (a64, b64, x64) = (f64::from(a), f64::from(b), f64::from(x));
        let want = clamp_integral(a64, b64, [x64 - 0.5, x64 + 0.5]);
        let m_z = a64.abs() * (x64.abs() + 1.0) + b64.abs() + 1.0;
        let spread = if a64.abs() > 1.0 {
            1.0 / a64.abs()
        } else {
            1.0
        };
        let tolerance = ROUNDOFFS * ROUNDOFF * (1.0 + (m_z * spread).min(1.0 / ROUNDOFF));
        let sample = Sample {
            got,
            want,
            tolerance,
            centre: [x, y],
        };
        worst.record("clamp", sample, &(index, x, a, b));
    }
    worst.report("clamp");
}

/// **(a) Pins, exact.** A band covering the pixel with the chord far to its
/// right is exactly `1`; a band wholly above it, or a chord wholly to its
/// left, exactly `0`; a band of height `h` with the chord far right exactly
/// `±h`. No tolerance: saturated sweeps are decided by comparison, never by
/// a quotient that happens to round to 1.
#[test]
fn full_and_empty_chords_are_exact() {
    let term = ChordTerm::new();
    let mut program = Program::new(&term.kernel().area());
    let (x, y) = (37.0f32, -12.0f32);
    let cases = [
        ("full", 1.0, [y - 5.0, y + 5.0], x + 10.0, 1.0f32),
        ("band above", 1.0, [y + 2.0, y + 3.0], x + 10.0, 0.0),
        ("band below", 1.0, [y - 3.0, y - 0.5], x + 10.0, 0.0),
        ("chord left", 1.0, [y - 5.0, y + 5.0], x - 10.0, 0.0),
        (
            "height, rising",
            1.0,
            [y - 0.25, y + 0.125],
            x + 10.0,
            0.375,
        ),
        (
            "height, falling",
            -1.0,
            [y - 0.25, y + 0.125],
            x + 10.0,
            -0.375,
        ),
    ];
    for (name, sigma, band, anchor_x, want) in cases {
        let chord = Chord {
            sigma,
            band,
            anchor: [anchor_x, y],
            slope: 0.3,
        };
        term.set(&mut program.block, chord);
        let got = program.eval(x, y);
        assert_eq!(got.to_bits(), f64::from(want).to_bits(), "{name}: {got}");
    }
}

/// **(a) Two chords sharing a vertex.** A triangle with one vertex inside
/// the pixel and the others a few pixels out: the sum of its three chords'
/// areas is the triangle's area inside the pixel — judged against the
/// triangle clipped to the pixel in `f64`, never against pixelflow's own
/// sum.
#[test]
fn chords_sharing_a_vertex_sum_to_the_polygon() {
    let terms = [ChordTerm::new(), ChordTerm::new(), ChordTerm::new()];
    let sum = terms
        .iter()
        .map(|term| term.kernel().area())
        .reduce(|a, b| a.add(&b))
        .expect("three chords");
    assert_legal("vertex", &sum, 0);
    let mut program = Program::new(&sum);
    let mut rng = Rng(0x5eed_0008);
    let mut worst = Worst::default();
    for index in 0..SAMPLES {
        let [x, y] = centre(&mut rng, index);
        let (cx, cy) = (f64::from(x), f64::from(y));
        let mut triangle = [
            [cx + rng.range(-0.45, 0.45), cy + rng.range(-0.45, 0.45)],
            [cx + rng.range(-3.0, 3.0), cy + rng.range(-3.0, 3.0)],
            [cx + rng.range(-3.0, 3.0), cy + rng.range(-3.0, 3.0)],
        ];
        // The chords' own `f32` vertices define the triangle both sides see.
        for p in &mut triangle {
            *p = [f64::from(p[0] as f32), f64::from(p[1] as f32)];
        }
        if shoelace(&triangle) < 0.0 {
            triangle.swap(1, 2);
        }
        let mut tolerance = 0.0;
        for (i, term) in terms.iter().enumerate() {
            let (p, q) = (triangle[i], triangle[(i + 1) % 3]);
            let dy = q[1] - p[1];
            // Counter-clockwise, so an edge crossing the ray to a point's
            // right counts +1 going up and −1 going down.
            let chord = Chord {
                sigma: dy.signum() as f32,
                band: [p[1].min(q[1]) as f32, p[1].max(q[1]) as f32],
                anchor: [p[0] as f32, p[1] as f32],
                slope: ((q[0] - p[0]) / dy) as f32,
            };
            tolerance += chord.tolerance(Rect::pixel(cx, cy), [cx, cy]);
            term.set(&mut program.block, chord);
        }
        // Inside the triangle is left of every counter-clockwise edge: clip
        // the pixel by each edge's half-plane in turn.
        let mut region = Rect::pixel(cx, cy).corners();
        for i in 0..3 {
            let (a, b) = (triangle[i], triangle[(i + 1) % 3]);
            region = clip(&region, |p| {
                (b[1] - a[1]) * (p[0] - a[0]) - (b[0] - a[0]) * (p[1] - a[1])
            });
        }
        let want = shoelace(&region);
        let got = program.eval(x, y);
        let sample = Sample {
            got,
            want,
            tolerance,
            centre: [x, y],
        };
        worst.record("vertex", sample, &(index, x, y, triangle));
    }
    worst.report("vertex");
}

/// `∫_{u ∈ [-½, ½)} body(X + u) du`: one integral, across the pixel's
/// width, binding slot 0.
fn along_x(body: impl FnOnce(&Kernel) -> Kernel) -> Kernel {
    let integrand = body(&Kernel::x().add(&binder(0)));
    let (arena, root) = integrand.parts();
    let mut a = arena.clone();
    let root = a.push_reduce(interval(0, Offsets::PIXEL.x), root);
    Kernel::from_parts(a, root)
}

/// **(b) Each rule closes its own integrand**, and what it closes to is the
/// integral:
/// - narrowing alone: `∫ [x < 3.2] = clamp(3.7 − X, 0, 1)`;
/// - factoring, then narrowing: `∫ Y·[x ≥ 1.7] = Y·clamp(X − 1.2, 0, 1)`;
/// - the clamp moment alone: `∫ clamp(0.3·x − 1, 0, 1)`, integrated
///   piecewise.
#[test]
fn each_rule_closes_its_own_integrand_to_its_integral() {
    let c = Kernel::constant;
    type Reference = fn(f64, f64) -> f64;
    let cases: [(&str, Kernel, Reference); 3] = [
        ("narrow", along_x(|x| indicator(&x.lt(&c(3.2)))), |x, _| {
            (f64::from(3.2f32) - (x - 0.5)).clamp(0.0, 1.0)
        }),
        (
            "factor, narrow",
            along_x(|x| Kernel::y().mul(&indicator(&x.ge(&c(1.7))))),
            |x, y| y * ((x + 0.5) - f64::from(1.7f32)).clamp(0.0, 1.0),
        ),
        (
            "clamp moment",
            along_x(|x| x.mul(&c(0.3)).sub(&c(1.0)).clamp(&c(0.0), &c(1.0))),
            |x, _| clamp_integral(f64::from(0.3f32), -1.0, [x - 0.5, x + 0.5]),
        ),
    ];
    for (name, kernel, reference) in cases {
        assert_legal(name, &kernel, 0);
        let program = Program::new(&kernel);
        for step in -12..=12 {
            let (x, y) = (step as f32 * 0.37, 2.0 - step as f32 * 0.5);
            let (got, want) = (program.eval(x, y), reference(f64::from(x), f64::from(y)));
            let tolerance = ROUNDOFFS * ROUNDOFF * (4.0 + f64::from(x.abs() + y.abs()));
            assert!(
                (got - want).abs() <= tolerance,
                "{name} at ({x}, {y}): got {got}, f64 {want}"
            );
        }
    }
}

/// **(b) What no rule closes is legalized, not emitted.** `∫∫ sin(x)` keeps
/// both integrals through extraction; `resolve` replaces them by the
/// one-point quadrature, so the texel is `sin` at the pixel's centre —
/// judged by `f64` `sin`.
#[test]
fn an_unclosed_integral_is_its_centre_sample() {
    let area = Kernel::x().sin().area();
    assert_legal("sin", &area, 2);
    let program = Program::new(&area);
    let mut rng = Rng(0x5eed_0009);
    for _ in 0..64 {
        let (x, y) = (rng.range(-64.0, 64.0) as f32, rng.range(-64.0, 64.0) as f32);
        let got = program.eval(x, y);
        let want = f64::from(x).sin();
        assert!(
            (got - want).abs() <= 1.0e-5,
            "sin at ({x}, {y}): got {got}, f64 {want}"
        );
    }
}
