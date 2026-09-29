//! **The closed form, attacked where the e-graph can see the most and the
//! floating point is thinnest**, and judged piece by piece by a reference
//! that shares nothing with it.
//!
//! The cases are `pixelflow-core/tests/arc_adversarial.rs`'s and
//! `area_adversarial.rs`'s, which attacked the integral a piece was written
//! as and the rules that closed it. The integral is gone from the glyph
//! (the module docs), so the same arcs and chords are pointed at what
//! replaced it, [`piece_term`] — cut, reflected and signed as the fold sums
//! it:
//!
//! - **Every direction, anywhere, any length.** Monotone quadratics in the
//!   four diagonal directions, frames to `±1000`, spans to `600` pixels, a
//!   fifth flat on an axis, control points anywhere or hooked into a corner
//!   of their box — collapsed with denormals kept and flushed, as the
//!   renderer's workers flush them.
//! - **Literal zeros.** Every column a constant, so the algebra folds the
//!   certificates and sees the zero steps: horizontal and vertical lines on
//!   pixel edges and corners, an extremum at an end, a point, a hook.
//! - **A bend the algebra proves zero over a step it cannot fold.** One
//!   uniform read as both steps of a line. The miscompile it found in the
//!   integral's closing — each root's denominator stopped varying, and
//!   `δ/d` extracted as a hoisted `δ·recip(d)`, `3.9e-2` of coverage wrong
//!   at AVX2 — is the one the closed form's quotients are spelled against
//!   (`integral::monotone_root`, "Emitted as `δ·(1/d)`").
//! - **A glyph is one fold over a table**, of closed contours whose pieces
//!   are hundreds of pixels long, and of lines read through one step column.
//! - **Chords.** Lines of slope `0` to `10⁸` either way, through bands
//!   below, above, straddling, inside, touching and missing the pixel,
//!   about `|X|, |Y| ≈ 1000`; with the slope a uniform, a literal the
//!   e-graph folds around (`0` found a miscompile in the integral's closing:
//!   a provably zero sweep divided by, and a chord's whole area extracted as
//!   `0`), and in whole rows, so one batch holds lanes left of, on and right
//!   of the crossing. A cut the pixel misses and a crossing wholly left of
//!   it are exactly `0`.
//!
//! Every program here is asserted to optimize to a term with no reciprocal
//! estimate.
//!
//! **The reference** is Green's theorem on the arc the row describes, in
//! `f64`: a piece's share of the pixel's winding integral is
//! `σ·∫₀¹ clamp(x(t) − X + ½, 0, 1)·[|y(t) − S·Y| < ½]·y′(t) dt` along the
//! reflected arc. Cut at every parameter where `x` or `y` meets an edge of
//! the pixel, the integrand is a polynomial of degree at most three on each
//! piece, which three-point Gauss–Legendre integrates exactly. No closed
//! form, and no polygon. The row is what the kernel is asked to integrate,
//! so the reference reads the row: the host's rounding of an outline into
//! rows is `tests/glyph_exact_area.rs`'s to judge, not this file's.
//!
//! Dropped with their subjects: the spellings (an author's way of writing
//! the integrand, which the rule read or declined — a closed form has one
//! spelling), floors other than `ROOT_FLOOR` (`RootFloor`'s admission, an
//! IR type pinned in its own crate), clamps into bands other than
//! `[0, 1]` and integrals over intervals other than the pixel (the
//! integration rules' generality). A crossing wholly right of the pixel
//! was pinned bit for bit to `σ·h` there, a property of the clamp
//! moment's comparison-first arms; the closed form climbs the band's
//! height through its roots, and is held to the tolerance instead.

use std::sync::Arc;

use super::*;
use pixelflow_core::{FastMathGuard, PlaneRegion, Uniform, UniformBlock};
use pixelflow_ir::{ExprNode, LatticeShape};
use pixelflow_search::runtime::optimize_runtime_arena;

/// Texels per frame row.
const WIDTH: usize = 16;
/// Frame rows.
const ROWS: usize = 8;
/// `2⁻²⁴`.
const EPS: f64 = 1.0 / 16_777_216.0;
/// Rows a contour's table holds; a contour with fewer pads with zeros,
/// which contribute exactly nothing.
const TABLE_ROWS: usize = 12;
/// A texel whose share is within this of an integer is not counted as cut.
const UNCUT: f64 = 1.0e-3;

// ───────────────────────────── the reference ─────────────────────────────

type Point = [f64; 2];

/// A quadratic Bézier, `p₀ → p₁ → p₂`.
#[derive(Clone, Copy, Debug)]
struct Quad([Point; 3]);

impl Quad {
    /// The straight piece `p₀ → p₂`, its control point the midpoint.
    fn line(p0: Point, p2: Point) -> Self {
        Self([p0, [0.5 * (p0[0] + p2[0]), 0.5 * (p0[1] + p2[1])], p2])
    }

    fn coordinate(self, axis: usize, t: f64) -> f64 {
        let [a, b, c] = self.0.map(|p| p[axis]);
        let s = 1.0 - t;
        s * s * a + 2.0 * s * t * b + t * t * c
    }

    fn velocity(self, axis: usize, t: f64) -> f64 {
        let [a, b, c] = self.0.map(|p| p[axis]);
        2.0 * ((1.0 - t) * (b - a) + t * (c - b))
    }

    /// Every `t` in `(0, 1)` where the coordinate is `level`: the stable
    /// quadratic roots, polished by Newton's method.
    fn meets(self, axis: usize, level: f64) -> Vec<f64> {
        let [a, b, c] = self.0.map(|p| p[axis]);
        let (qa, qb, qc) = (a - 2.0 * b + c, 2.0 * (b - a), a - level);
        let mut roots = Vec::new();
        if qa == 0.0 {
            if qb != 0.0 {
                roots.push(-qc / qb);
            }
        } else {
            let disc = qb * qb - 4.0 * qa * qc;
            if disc >= 0.0 {
                let q = -0.5 * (qb + disc.sqrt().copysign(qb));
                roots.push(q / qa);
                if q != 0.0 {
                    roots.push(qc / q);
                }
            }
        }
        roots
            .into_iter()
            .map(|mut t| {
                for _ in 0..3 {
                    let v = self.velocity(axis, t);
                    if v == 0.0 {
                        break;
                    }
                    t -= (self.coordinate(axis, t) - level) / v;
                }
                t
            })
            .filter(|t| *t > 0.0 && *t < 1.0)
            .collect()
    }

    /// `∫₀¹ clamp(x(t) − X + ½, 0, 1)·[|y(t) − Y| < ½]·y′(t) dt` for the
    /// pixel about `centre = (X, Y)`: this arc's share of the pixel's
    /// winding integral, read on its left. Three-point Gauss–Legendre
    /// between the parameters where either coordinate meets an edge.
    fn share(self, [x, y]: Point) -> f64 {
        const NODE: f64 = 0.774_596_669_241_483_4; // √(3/5)
        const RULE: [(f64, f64); 3] = [(-NODE, 5.0 / 9.0), (0.0, 8.0 / 9.0), (NODE, 5.0 / 9.0)];
        let mut cuts = vec![0.0, 1.0];
        for (axis, c) in [(0, x), (1, y)] {
            cuts.extend(self.meets(axis, c - 0.5));
            cuts.extend(self.meets(axis, c + 0.5));
        }
        cuts.sort_by(f64::total_cmp);
        let integrand = |p: Point| {
            if p[1] < y - 0.5 || p[1] >= y + 0.5 {
                return 0.0;
            }
            (p[0] - (x - 0.5)).clamp(0.0, 1.0)
        };
        cuts.windows(2)
            .map(|w| {
                let (mid, half) = (0.5 * (w[0] + w[1]), 0.5 * (w[1] - w[0]));
                let sum: f64 = RULE
                    .iter()
                    .map(|&(node, weight)| {
                        let t = mid + half * node;
                        let p = [self.coordinate(0, t), self.coordinate(1, t)];
                        weight * integrand(p) * self.velocity(1, t)
                    })
                    .sum();
                sum * half
            })
            .sum()
    }

    /// The larger of the piece's two extents.
    fn extent(self) -> f64 {
        let span = |axis: usize| {
            let v = self.0.map(|p| p[axis]);
            let high = v.iter().copied().fold(f64::MIN, f64::max);
            high - v.iter().copied().fold(f64::MAX, f64::min)
        };
        span(0).max(span(1))
    }
}

// ───────────────────────────── the rows ─────────────────────────────

/// A piece's row, as the kernel reads it.
#[derive(Clone, Copy, Debug)]
struct Row([f32; PIECE_ROW_COLS]);

impl Row {
    /// A raw quadratic on the screen as a row: reversed so `x` rises, then
    /// reflected so `y` rises, the steps and the start rounded to `f32` —
    /// this file's own orientation, not [`Piece::oriented`], so the
    /// reference does not share the host's. The band is the arc's rows
    /// widened by half a pixel, rounded outward.
    ///
    /// # Panics
    ///
    /// If the piece is not monotone.
    fn of(quad: Quad) -> Self {
        let [mut p0, p1, mut p2] = quad.0;
        let reversed = p2[0] < p0[0];
        if reversed {
            core::mem::swap(&mut p0, &mut p2);
        }
        let s = if p2[1] < p0[1] { -1.0 } else { 1.0 };
        let mut row = [0.0f32; PIECE_ROW_COLS];
        row[COL_X0] = p0[0] as f32;
        row[COL_Y0] = (s * p0[1]) as f32;
        row[COL_E0X] = (p1[0] - p0[0]) as f32;
        row[COL_E0Y] = (s * (p1[1] - p0[1])) as f32;
        row[COL_E1X] = (p2[0] - p1[0]) as f32;
        row[COL_E1Y] = (s * (p2[1] - p1[1])) as f32;
        row[COL_SIGMA] = if reversed { -s } else { s } as f32;
        row[COL_S] = s as f32;
        let half = f64::from(PIXEL_HALF);
        row[COL_ROWS_LO] = f32_down(p0[1].min(p2[1]) - half);
        row[COL_ROWS_HI] = f32_up(p0[1].max(p2[1]) + half);
        for step in [COL_E0X, COL_E0Y, COL_E1X, COL_E1Y] {
            assert!(row[step] >= 0.0, "{quad:?} is not monotone: {row:?}");
        }
        Self(row)
    }

    /// The arc the row describes, in its reflected frame, where it rises
    /// in both coordinates: what the kernel integrates.
    fn arc(self) -> Quad {
        let f = |k: usize| f64::from(self.0[k]);
        let p0 = [f(COL_X0), f(COL_Y0)];
        let p1 = [p0[0] + f(COL_E0X), p0[1] + f(COL_E0Y)];
        let p2 = [p1[0] + f(COL_E1X), p1[1] + f(COL_E1Y)];
        Quad([p0, p1, p2])
    }

    /// `σ·share` of the pixel about the screen's `centre`, read in the
    /// reflected frame, or `0` outside the rows the row's cut keeps.
    fn share(self, [x, y]: Point) -> f64 {
        let f = |k: usize| f64::from(self.0[k]);
        if y <= f(COL_ROWS_LO) || y >= f(COL_ROWS_HI) {
            return 0.0;
        }
        f(COL_SIGMA) * self.arc().share([x, f(COL_S) * y])
    }
}

/// Which column each column is read through: a host that stores a line's
/// one step and reads it as both, or every column as itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Steps {
    Two,
    OneReadTwice,
}

impl Steps {
    fn read_as(self, col: usize) -> usize {
        match (self, col) {
            (Self::OneReadTwice, COL_E1X) => COL_E0X,
            (Self::OneReadTwice, COL_E1Y) => COL_E0Y,
            _ => col,
        }
    }
}

// ───────────────────────────── the kernel ─────────────────────────────

fn c(v: f32) -> Kernel {
    Kernel::constant(v)
}

/// `k` translated so the frame's first texel is `origin`'s pixel.
fn placed(k: &Kernel, [ox, oy]: [&Kernel; 2]) -> Kernel {
    k.at(&Kernel::x().add(ox), &Kernel::y().add(oy))
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

/// Asserts the runtime tier optimizes `kernel` at the frame's shape to a
/// term that holds no reciprocal estimate.
fn assert_no_estimate(name: &str, kernel: &Kernel) {
    let (arena, root) = kernel.linked_parts();
    let shape = LatticeShape::new([WIDTH as u32, ROWS as u32]);
    let optimized = optimize_runtime_arena(&arena, root, shape)
        .unwrap_or_else(|| panic!("{name}: the runtime tier declined the term"));
    let estimate = reaches(&optimized.0, optimized.1, |node| {
        matches!(node, ExprNode::Unary(OpKind::Recip | OpKind::Rsqrt, _))
    });
    assert!(
        !estimate,
        "{name}: the optimized term holds a reciprocal estimate"
    );
}

/// Whether a collapse runs with denormals flushed, as the renderer's
/// workers run (`FastMathGuard`), or with the IEEE default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Denormals {
    Kept,
    Flushed,
}

/// A collapsed frame, and where its first texel's pixel is.
struct Frame {
    values: Vec<f32>,
    origin: [f32; 2],
}

impl Frame {
    /// Each texel's centre on the screen, and what the kernel wrote there.
    fn texels(&self) -> impl Iterator<Item = (Point, f32)> + '_ {
        (0..ROWS).flat_map(move |row| {
            (0..WIDTH).map(move |col| {
                let centre = [
                    f64::from(self.origin[0]) + col as f64 + 0.5,
                    f64::from(self.origin[1]) + row as f64 + 0.5,
                ];
                (centre, self.values[row * WIDTH + col])
            })
        })
    }
}

/// A compiled kernel and the uniforms it reads.
struct Program {
    manifold: Manifold,
    block: UniformBlock,
}

impl Program {
    fn new(kernel: &Kernel) -> Self {
        let manifold = Manifold::compile(kernel, [WIDTH as u32, ROWS as u32]);
        let block = manifold.block();
        Self { manifold, block }
    }

    fn set(&mut self, uniform: Uniform, value: f32) {
        self.block
            .set(uniform, value)
            .expect("a uniform the kernel reads");
    }

    /// The frame at `origin`, its tables bound to `tables`.
    fn collapse(
        &self,
        origin: [f32; 2],
        tables: &[(BufferIdentity, Arc<Vec<f32>>)],
        denormals: Denormals,
    ) -> Frame {
        let mut values = vec![f32::NAN; WIDTH * ROWS];
        let region = PlaneRegion::rows(WIDTH, 0, ROWS);
        let bound = self.manifold.bind(tables);
        let bound = bound.with_uniforms(&self.block);
        match denormals {
            Denormals::Kept => bound.collapse_rows(region, &mut values, WIDTH),
            Denormals::Flushed => {
                // SAFETY: the guard restores this thread's floating-point
                // mode when it drops at the end of this arm, and nothing
                // here needs denormals preserved.
                let _flushed = unsafe { FastMathGuard::new() };
                bound.collapse_rows(region, &mut values, WIDTH);
            }
        }
        Frame { values, origin }
    }
}

/// [`piece_term`] over one row of uniforms, placed by an origin uniform:
/// compiled once, collapsed per row.
struct Piecewise {
    columns: [Uniform; PIECE_ROW_COLS],
    steps: Steps,
    origin: [Uniform; 2],
    program: Program,
}

impl Piecewise {
    fn new(name: &str, steps: Steps) -> Self {
        let columns: [Uniform; PIECE_ROW_COLS] = core::array::from_fn(|_| Uniform::new(0.0));
        let origin = [Uniform::new(0.0), Uniform::new(0.0)];
        let term = piece_term(&|k| columns[steps.read_as(k)].kernel());
        let kernel = placed(&term, [&origin[0].kernel(), &origin[1].kernel()]);
        assert_no_estimate(name, &kernel);
        Self {
            columns,
            steps,
            origin,
            program: Program::new(&kernel),
        }
    }

    fn frame(&mut self, row: Row, origin: [f32; 2], denormals: Denormals) -> Frame {
        for k in 0..PIECE_ROW_COLS {
            if self.steps.read_as(k) == k {
                self.program.set(self.columns[k], row.0[k]);
            }
        }
        for (uniform, value) in self.origin.into_iter().zip(origin) {
            self.program.set(uniform, value);
        }
        self.program.collapse(origin, &[], denormals)
    }
}

/// [`piece_term`] over one row of literals at `origin`, every column a
/// constant the algebra folds: compiled once.
fn literal_program(name: &str, row: Row, origin: [f32; 2]) -> Program {
    let term = piece_term(&|k| c(row.0[k]));
    let kernel = placed(&term, [&c(origin[0]), &c(origin[1])]);
    assert_no_estimate(name, &kernel);
    Program::new(&kernel)
}

// ───────────────────────────── the judge ─────────────────────────────

/// What a texel's error is held to.
#[derive(Clone, Copy, Debug)]
enum Tolerance {
    /// `16·2⁻²⁴·(1 + extent)`: the parameter an arc is read at resolves to
    /// `2⁻²⁴`, which its extent multiplies, a few times over — and nothing
    /// in where the pixel is, since with exact columns `X − x₀` and
    /// `S·Y − y₀` are differences of exact values.
    Extent,
    /// [`Tolerance::Extent`] plus `4·2⁻²⁴·(|X| + |Y|)`: the rounding of a
    /// coordinate as large as the pixel's, which a literal column costs
    /// where the algebra distributes it over `(Y + origin) − y₀` and folds
    /// the constant part, so the difference is no longer taken first.
    Offset,
}

impl Tolerance {
    fn at(self, extent: f64, centre: Point) -> f64 {
        let own = 16.0 * EPS * (1.0 + extent);
        match self {
            Self::Extent => own,
            Self::Offset => own + 4.0 * EPS * (centre[0].abs() + centre[1].abs()),
        }
    }
}

/// The largest error a category saw, against its tolerance, and how many
/// of its texels a piece cut — covered neither wholly nor not at all, the
/// ones a closed form can get wrong while every pin on `0` and `1` holds.
struct Worst {
    tolerance: Tolerance,
    error: f64,
    ratio: f64,
    samples: usize,
    cut: usize,
}

impl Worst {
    fn new(tolerance: Tolerance) -> Self {
        Self {
            tolerance,
            error: 0.0,
            ratio: 0.0,
            samples: 0,
            cut: 0,
        }
    }

    /// Every texel of `frame` against the rows' summed shares.
    fn frame(&mut self, case: &dyn core::fmt::Debug, frame: &Frame, rows: &[Row]) {
        let extent: f64 = rows.iter().map(|r| r.arc().extent()).sum();
        for (centre, got) in frame.texels() {
            let want: f64 = rows.iter().map(|r| r.share(centre)).sum();
            let tolerance = self.tolerance.at(extent, centre);
            self.judge(case, centre, (got, want), tolerance);
        }
    }

    fn judge(
        &mut self,
        case: &dyn core::fmt::Debug,
        centre: Point,
        (got, want): (f32, f64),
        tolerance: f64,
    ) {
        let error = (f64::from(got) - want).abs();
        assert!(
            error <= tolerance,
            "{case:?} at {centre:?}: got {got}, want {want}, error {error:e} > {tolerance:e}"
        );
        self.error = self.error.max(error);
        self.ratio = self.ratio.max(error / tolerance);
        self.samples += 1;
        let fraction = want.abs().fract();
        if fraction > UNCUT && fraction < 1.0 - UNCUT {
            self.cut += 1;
        }
    }

    /// Report the category, and require that its pieces cut some texels.
    fn report(&self, name: &str) {
        eprintln!(
            "adversarial {name}: {} texels ({} cut), max |error| {:.3e}, error/tolerance {:.3}",
            self.samples, self.cut, self.error, self.ratio
        );
        assert!(self.cut > 0, "{name}: no texel was cut");
    }
}

// ───────────────────────────── the pieces ─────────────────────────────

/// splitmix64: seeded and dependency-free, so a failure names its case.
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

    /// A pixel corner within `reach` of the origin, on each axis.
    fn origin(&mut self, reach: f64) -> [f32; 2] {
        [
            self.range(-reach, reach).round() as f32,
            self.range(-reach, reach).round() as f32,
        ]
    }

    fn pick<T: Copy>(&mut self, from: &[T]) -> T {
        from[(self.next() % from.len() as u64) as usize]
    }
}

/// Where a piece's control point sits in the box of its ends, per axis:
/// `0` at the start's side, `1` at the end's.
#[derive(Clone, Copy, Debug)]
enum Bulge {
    /// Anywhere in the box.
    Anywhere,
    /// At or within `10⁻⁴` of a corner of the box: a step exactly or nearly
    /// zero, so the arc turns hard or has its extremum at an end.
    Hooked,
    /// At the midpoint: a line.
    Straight,
}

impl Bulge {
    fn draw(self, rng: &mut Rng) -> [f64; 2] {
        match self {
            Self::Anywhere => [rng.range(0.0, 1.0), rng.range(0.0, 1.0)],
            Self::Hooked => {
                let near = [0.0, 1.0e-4, 1.0 - 1.0e-4, 1.0];
                [rng.pick(&near), rng.pick(&near)]
            }
            Self::Straight => [0.5, 0.5],
        }
    }
}

/// How to draw a monotone piece through a frame.
#[derive(Clone, Copy, Debug)]
struct Draw {
    /// The frame's first texel's pixel.
    origin: [f32; 2],
    /// The ends' separation per axis, both `≥ 0`.
    span: [f64; 2],
    /// `±1` per axis: which way the piece runs.
    direction: [f64; 2],
    bulge: Bulge,
    /// Every coordinate on a grid of `1/grid`: `2` puts ends, control
    /// points and extrema on pixel edges and centres.
    grid: f64,
}

impl Draw {
    /// A monotone quadratic through the frame. For a straight piece the
    /// control point is the ends' midpoint, exact on a grid twice as fine.
    fn piece(self, rng: &mut Rng) -> Quad {
        let on = |v: f64| (v * self.grid).round() / self.grid;
        let through = [
            f64::from(self.origin[0]) + rng.range(0.0, WIDTH as f64),
            f64::from(self.origin[1]) + rng.range(0.0, ROWS as f64),
        ];
        let along = rng.range(0.0, 1.0);
        let [dx, dy] = self.direction;
        let p0 = [
            on(through[0] - dx * along * self.span[0]),
            on(through[1] - dy * along * self.span[1]),
        ];
        let p2 = [on(p0[0] + dx * self.span[0]), on(p0[1] + dy * self.span[1])];
        let bulge = self.bulge.draw(rng);
        let p1 = [0, 1].map(|axis| {
            let p1 = p0[axis] + bulge[axis] * (p2[axis] - p0[axis]);
            match self.bulge {
                Bulge::Straight => p1,
                _ => on(p1),
            }
        });
        Quad([p0, p1, p2])
    }
}

// ───────────────────────────── the arcs ─────────────────────────────

/// **Every direction, anywhere, any length.** Monotone quadratics running
/// in each of the four diagonal directions — reversed, reflected, both,
/// neither — frames from the origin to `±1000`, spans from a pixel to
/// `600`, a fifth of them flat on an axis (a uniform step of exactly zero,
/// whose root divides by the floor at run time), control points anywhere
/// or hooked into a corner of their box, on a fine grid or on pixel edges
/// and centres, collapsed with denormals kept and flushed.
#[test]
fn every_direction_is_the_arcs_share() {
    let mut piecewise = Piecewise::new("every direction", Steps::Two);
    let mut rng = Rng(0xad0e_0001);
    let mut worst = Worst::new(Tolerance::Extent);
    for bulge in [Bulge::Anywhere, Bulge::Hooked] {
        for denormals in [Denormals::Kept, Denormals::Flushed] {
            for index in 0..200 {
                let reach = rng.pick(&[0.0, 30.0, 1000.0]);
                let origin = rng.origin(reach);
                let long = rng.pick(&[1.0, 8.0, 100.0, 600.0]);
                let mut span = [rng.range(0.0, long), rng.range(0.0, long)];
                if rng.next().is_multiple_of(5) {
                    span[(rng.next() % 2) as usize] = 0.0;
                }
                let draw = Draw {
                    origin,
                    span,
                    direction: [rng.pick(&[-1.0, 1.0]), rng.pick(&[-1.0, 1.0])],
                    bulge,
                    grid: rng.pick(&[64.0, 2.0]),
                };
                let row = Row::of(draw.piece(&mut rng));
                let frame = piecewise.frame(row, origin, denormals);
                worst.frame(&(index, draw, row), &frame, &[row]);
            }
        }
    }
    worst.report("every direction");
}

/// **Literal zeros.** Every column a constant, so the e-graph folds the
/// certificates and sees the zero steps: horizontal and vertical lines
/// (on pixel edges, corner to corner, running down), a zero first or second
/// step on either axis — an extremum at an end — a point, and a hook to an
/// edge, about `base`, in every direction; with denormals kept, and
/// flushed for the pieces that start on a pixel edge, where a flat rise's
/// root is `δ` times the reciprocal of the floor at a height `δ` that is
/// exactly zero.
///
/// Every case is a program of its own, which is a saturation of its own:
/// one test per base, so the three run in parallel.
fn literal_zero_steps_about(base: Point) {
    let mut worst = Worst::new(Tolerance::Offset);
    let o = |dx: f64, dy: f64| [base[0] + dx, base[1] + dy];
    let cases = [
        ("horizontal", Quad::line(o(1.25, 3.5), o(9.75, 3.5))),
        (
            "horizontal on an edge",
            Quad::line(o(1.0, 3.0), o(9.0, 3.0)),
        ),
        ("vertical", Quad::line(o(4.25, 0.5), o(4.25, 6.75))),
        ("vertical on an edge", Quad::line(o(4.0, 0.5), o(4.0, 6.75))),
        (
            "vertical from an edge",
            Quad::line(o(4.0, 1.0), o(4.0, 6.5)),
        ),
        (
            "vertical corner to corner",
            Quad::line(o(7.0, 1.0), o(7.0, 6.0)),
        ),
        ("vertical down", Quad::line(o(4.25, 6.75), o(4.25, 0.5))),
        ("line", Quad::line(o(1.5, 0.25), o(13.25, 7.5))),
        (
            "line corner to corner",
            Quad::line(o(2.0, 1.0), o(14.0, 7.0)),
        ),
        ("line down-left", Quad::line(o(13.25, 7.5), o(1.5, 0.25))),
        (
            "flat start",
            Quad([o(1.0, 1.25), o(7.0, 1.25), o(13.0, 7.5)]),
        ),
        (
            "flat start on an edge",
            Quad([o(1.0, 2.0), o(6.0, 2.0), o(12.0, 7.0)]),
        ),
        ("flat end", Quad([o(1.0, 1.25), o(1.0, 7.5), o(13.0, 7.5)])),
        (
            "steep start",
            Quad([o(2.5, 0.5), o(2.5, 5.0), o(12.0, 7.0)]),
        ),
        (
            "steep start on a corner",
            Quad([o(3.0, 1.0), o(3.0, 5.0), o(11.0, 7.0)]),
        ),
        ("steep end", Quad([o(2.5, 0.5), o(12.0, 0.5), o(12.0, 7.0)])),
        (
            "flat start, falling",
            Quad([o(1.0, 7.25), o(7.0, 7.25), o(13.0, 0.5)]),
        ),
        (
            "steep end, reversed",
            Quad([o(12.0, 7.0), o(12.0, 0.5), o(2.5, 0.5)]),
        ),
        ("point", Quad([o(5.5, 3.5), o(5.5, 3.5), o(5.5, 3.5)])),
        (
            "hook to an edge",
            Quad([o(3.0, 2.0), o(3.0, 5.0), o(9.0, 5.0)]),
        ),
    ];
    let origin = [base[0] as f32, base[1] as f32];
    for (name, quad) in cases {
        let row = Row::of(quad);
        let program = literal_program(name, row, origin);
        for denormals in [Denormals::Kept, Denormals::Flushed] {
            let frame = program.collapse(origin, &[], denormals);
            worst.frame(&(name, base, denormals, quad), &frame, &[row]);
        }
    }
    worst.report(&format!("literal zeros about {base:?}"));
}

#[test]
fn literal_zero_steps_near_the_origin_are_the_arcs_share() {
    literal_zero_steps_about([0.0, 0.0]);
}

#[test]
fn literal_zero_steps_at_minus_1000_1000_are_the_arcs_share() {
    literal_zero_steps_about([-1000.0, 1000.0]);
}

#[test]
fn literal_zero_steps_at_997_minus_1003_are_the_arcs_share() {
    literal_zero_steps_about([997.0, -1003.0]);
}

/// **A line whose bend the algebra proves zero, over a step that is
/// neither literal nor varying.** A host that stores a line's one step and
/// reads it as both — the same uniform for `p₁ − p₀` and `p₂ − p₁` — gives
/// `a = max(s, 0) − max(s, 0)`, which the algebra proves zero, and each
/// root's denominator stops varying. Spelled `δ/d`, a root was extracted as
/// `δ·recip(d)` with the estimate hoisted: `3.9e-2` of coverage wrong at
/// AVX2 and `3.5e-3` at AVX-512 on the longest lines here, when the
/// integral's closing built the roots. The closed form builds them from the
/// same definition, and this holds it to it.
#[test]
fn a_line_read_through_one_step_column() {
    let mut piecewise = Piecewise::new("one step column", Steps::OneReadTwice);
    let mut rng = Rng(0xad0e_0002);
    let mut worst = Worst::new(Tolerance::Extent);
    for denormals in [Denormals::Kept, Denormals::Flushed] {
        for index in 0..100 {
            let reach = rng.pick(&[0.0, 1000.0]);
            let origin = rng.origin(reach);
            let long = rng.pick(&[1.0, 8.0, 100.0, 600.0]);
            let draw = Draw {
                origin,
                span: [rng.range(0.0, long), rng.range(0.0, long)],
                direction: [rng.pick(&[-1.0, 1.0]), rng.pick(&[-1.0, 1.0])],
                bulge: Bulge::Straight,
                grid: 32.0,
            };
            let row = Row::of(draw.piece(&mut rng));
            let frame = piecewise.frame(row, origin, denormals);
            worst.frame(&(index, draw, row), &frame, &[row]);
        }
    }
    worst.report("one step column");
}

/// A closed contour about `centre`: a star-shaped polygon of up to
/// [`TABLE_ROWS`] vertices at radius up to `radius`, each edge a monotone
/// quadratic drawn as `bulge` says, every coordinate on a grid of `1/64`
/// (a straight edge's control point on `1/128`), so consecutive pieces meet
/// exactly.
struct Contour {
    centre: Point,
    radius: f64,
    bulge: Bulge,
}

impl Contour {
    fn pieces(&self, rng: &mut Rng) -> Vec<Quad> {
        let on = |v: f64| (v * 64.0).round() / 64.0;
        let sides = 3 + (rng.next() % (TABLE_ROWS as u64 - 2)) as usize;
        let mut angles: Vec<f64> = (0..sides)
            .map(|_| rng.range(0.0, core::f64::consts::TAU))
            .collect();
        angles.sort_by(f64::total_cmp);
        let vertices: Vec<Point> = angles
            .iter()
            .map(|&angle| {
                let r = rng.range(0.3, 1.0) * self.radius;
                [
                    on(self.centre[0] + r * angle.cos()),
                    on(self.centre[1] + r * angle.sin()),
                ]
            })
            .collect();
        (0..sides)
            .map(|k| {
                let (p0, p2) = (vertices[k], vertices[(k + 1) % sides]);
                let bulge = self.bulge.draw(rng);
                let p1 = [0, 1].map(|axis| {
                    let p1 = p0[axis] + bulge[axis] * (p2[axis] - p0[axis]);
                    match self.bulge {
                        Bulge::Straight => p1,
                        _ => on(p1),
                    }
                });
                Quad([p0, p1, p2])
            })
            .collect()
    }
}

/// **A glyph is one fold over a table**, at `±1000` and with pieces
/// hundreds of pixels long, judged by summing the rows' shares. With
/// [`Steps::OneReadTwice`] every row is a line whose one step column is
/// read as both — the bend the algebra can prove zero, per row.
fn table_contours(name: &str, seed: u64, steps: Steps) {
    let id = BufferIdentity::mint();
    let table = DiscreteManifold::kernel_for(id, PIECE_ROW_COLS as u32, TABLE_ROWS as u32);
    let glyph = Kernel::sum_over(TABLE_ROWS as u32, |p| {
        piece_term(&|k| table.at(&c(steps.read_as(k) as f32), p))
    });
    let origin = [Uniform::new(0.0), Uniform::new(0.0)];
    let kernel = placed(&glyph, [&origin[0].kernel(), &origin[1].kernel()]);
    assert_no_estimate(name, &kernel);
    let mut program = Program::new(&kernel);
    let mut rng = Rng(seed);
    let mut worst = Worst::new(Tolerance::Extent);
    let bulge = match steps {
        Steps::Two => Bulge::Anywhere,
        Steps::OneReadTwice => Bulge::Straight,
    };
    for index in 0..60 {
        let reach = rng.pick(&[0.0, 1000.0]);
        let at = rng.origin(reach);
        for (uniform, value) in origin.into_iter().zip(at) {
            program.set(uniform, value);
        }
        let radius = rng.pick(&[4.0, 40.0, 400.0]);
        let contour = Contour {
            centre: [
                f64::from(at[0]) + WIDTH as f64 / 2.0 + rng.range(-radius, radius) * 0.5,
                f64::from(at[1]) + ROWS as f64 / 2.0 + rng.range(-radius, radius) * 0.5,
            ],
            radius,
            bulge,
        };
        let rows: Vec<Row> = contour.pieces(&mut rng).into_iter().map(Row::of).collect();
        let mut data = vec![0.0f32; PIECE_ROW_COLS * TABLE_ROWS];
        for (k, row) in rows.iter().enumerate() {
            data[k * PIECE_ROW_COLS..(k + 1) * PIECE_ROW_COLS].copy_from_slice(&row.0);
        }
        let frame = program.collapse(at, &[(id, Arc::new(data))], Denormals::Kept);
        worst.frame(&(name, index, &rows), &frame, &rows);
    }
    worst.report(name);
}

#[test]
fn a_contour_table_is_the_rows_shares() {
    table_contours("contour table", 0xad0e_0004, Steps::Two);
}

#[test]
fn a_line_table_read_through_one_step_column() {
    table_contours(
        "line table, one step column",
        0xad0e_0005,
        Steps::OneReadTwice,
    );
}

// ───────────────────────────── the chords ─────────────────────────────

/// A chord: the line `x = anchor_x + (y − anchor_y)·slope` over the band of
/// rows `[lo, hi]`, run toward `+Y` (`σ = +1`) or `−Y` (`σ = −1`).
#[derive(Clone, Copy, Debug)]
struct Chord {
    sigma: f32,
    band: [f32; 2],
    anchor: [f32; 2],
    slope: f32,
}

impl Chord {
    /// The chord as a row, run the way `σ` says — rounded to `f32` from
    /// `f64`, so the row's arc is the chord up to that rounding, and the
    /// reference reads the row's.
    fn row(self) -> Row {
        let f = f64::from;
        let [lo, hi] = [f(self.band[0]), f(self.band[1])];
        let at = |y: f64| f(self.anchor[0]) + (y - f(self.anchor[1])) * f(self.slope);
        let (bottom, top) = ([at(lo), lo], [at(hi), hi]);
        Row::of(match self.sigma < 0.0 {
            true => Quad::line(top, bottom),
            false => Quad::line(bottom, top),
        })
    }

    /// `f32`'s rounding of the terms the closed form is computed from: the
    /// band's ends against the pixel, and the crossing, whose shift moves
    /// the area by at most the rows where it is not saturated.
    fn tolerance(self, [x, y]: Point) -> f64 {
        const ROUNDOFFS: f64 = 16.0;
        let f = |v: f32| f64::from(v).abs();
        let band = f(self.band[0]) + f(self.band[1]) + 2.0 * y.abs() + 2.0;
        let k = f(self.slope);
        let crossing = k * (y.abs() + f(self.anchor[1]) + 1.0) + f(self.anchor[0]) + x.abs() + 1.0;
        let [lo, hi] = self.clipped(y);
        let height = (hi - lo).max(0.0);
        let unsaturated = if k > 0.0 { height.min(1.0 / k) } else { height };
        ROUNDOFFS * EPS * (band + crossing * unsaturated) + 8.0 * EPS
    }

    /// The band's rows inside the pixel about `y`.
    fn clipped(self, y: f64) -> [f64; 2] {
        let [lo, hi] = self.band.map(f64::from);
        [lo.max(y - 0.5), hi.min(y + 0.5)]
    }

    /// Whether the crossing is wholly left of the pixel across the clipped
    /// band, by a margin its rounding cannot cross.
    fn wholly_left(self, [x, y]: Point) -> bool {
        let f = f64::from;
        let [lo, hi] = self.clipped(y);
        let (ax, ay, k) = (f(self.anchor[0]), f(self.anchor[1]), f(self.slope));
        let at = |row: f64| ax + (row - ay) * k;
        let margin =
            64.0 * EPS * (k.abs() * (y.abs() + ay.abs() + 1.0) + ax.abs() + x.abs() + 2.0) + 1e-3;
        hi > lo && at(lo).max(at(hi)) <= x - 0.5 - margin
    }
}

/// A band about the pixel's rows `[y − ½, y + ½)`: below, above, straddling,
/// inside, touching an edge, or a single row.
fn band_about(rng: &mut Rng, y: f64) -> [f64; 2] {
    let (lo, hi) = (y - 0.5, y + 0.5);
    let mut r = |a: f64, b: f64| rng.range(a, b);
    match r(0.0, 8.0) as u32 {
        0 => [r(lo - 3.0, lo - 1.0), lo],
        1 => [hi, r(hi + 1.0, hi + 3.0)],
        2 => [r(lo - 3.0, lo), r(lo, hi)],
        3 => [r(lo, hi), r(hi, hi + 3.0)],
        4 => {
            let (a, b) = (r(lo, hi), r(lo, hi));
            [a.min(b), a.max(b)]
        }
        5 => {
            let a = r(lo, hi);
            [a, a]
        }
        _ => [r(lo - 3.0, lo), r(hi, hi + 3.0)],
    }
}

fn chord_about(rng: &mut Rng, [x, y]: Point, slope: f32) -> Chord {
    let band = band_about(rng, y);
    // The crossing passes somewhere near the pixel's centre row, or far to
    // one side; with a steep slope, the anchor row is what places it.
    let anchor_x = x + rng.pick(&[-40.0, 40.0, 0.0, 0.0, 0.0]) + rng.range(-1.5, 1.5);
    let anchor_y = y + rng.range(-1.0, 1.0) * f64::from(slope.abs().max(1.0)).recip().max(1e-6);
    Chord {
        sigma: rng.pick(&[1.0, -1.0]),
        band: [band[0] as f32, band[1] as f32],
        anchor: [anchor_x as f32, anchor_y as f32],
        slope,
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

/// A chord's term over uniforms, evaluated at one pixel; `slope` a literal
/// when given — its `x` steps then `|slope|` times the uniform `y` step, so
/// the e-graph sees the slope as a constant — else a uniform row.
struct ChordTerm {
    columns: [Uniform; PIECE_ROW_COLS],
    literal: Option<f32>,
    manifold: Manifold,
    block: UniformBlock,
}

impl ChordTerm {
    fn new(name: &str, literal: Option<f32>) -> Self {
        let columns: [Uniform; PIECE_ROW_COLS] = core::array::from_fn(|_| Uniform::new(0.0));
        let read = |k: usize| match (literal, k) {
            (Some(slope), COL_E0X | COL_E1X) => c(slope.abs()).mul(&columns[COL_E0Y].kernel()),
            (Some(_), COL_E1Y) => columns[COL_E0Y].kernel(),
            _ => columns[k].kernel(),
        };
        let kernel = piece_term(&read);
        let (arena, root) = kernel.linked_parts();
        let optimized = optimize_runtime_arena(&arena, root, LatticeShape::new([1, 1]))
            .unwrap_or_else(|| panic!("{name}: the runtime tier declined the term"));
        let estimate = reaches(&optimized.0, optimized.1, |node| {
            matches!(node, ExprNode::Unary(OpKind::Recip | OpKind::Rsqrt, _))
        });
        assert!(
            !estimate,
            "{name}: the optimized term holds a reciprocal estimate"
        );
        let manifold = Manifold::compile(&kernel, [1, 1]);
        let block = manifold.block();
        Self {
            columns,
            literal,
            manifold,
            block,
        }
    }

    /// The term for `row` at the pixel about `(x, y)` — the row as the
    /// kernel reads it, with a literal slope's `x` steps computed there.
    fn eval(&mut self, row: Row, [x, y]: [f32; 2]) -> (f32, Row) {
        let mut read = row;
        if let Some(slope) = self.literal {
            let step = slope.abs() * row.0[COL_E0Y];
            read.0[COL_E0X] = step;
            read.0[COL_E1X] = step;
            read.0[COL_E1Y] = row.0[COL_E0Y];
        }
        for k in 0..PIECE_ROW_COLS {
            let computed = self.literal.is_some() && matches!(k, COL_E0X | COL_E1X | COL_E1Y);
            if !computed {
                self.block
                    .set(self.columns[k], read.0[k])
                    .expect("a column the term reads");
            }
        }
        let got = self
            .manifold
            .bind(&[])
            .with_uniforms(&self.block)
            .eval_at(x, y);
        (got, read)
    }
}

/// Chords of one family through the closed form, each judged against the
/// row's `f64` share — and a cut the pixel misses, or a crossing wholly left
/// of it, judged bit for bit: exactly `0`.
fn run_chords(label: &str, literal: Option<f32>, seed: u64) {
    const SAMPLES: usize = 400;
    let mut term = ChordTerm::new(label, literal);
    let mut rng = Rng(seed);
    let mut worst = Worst::new(Tolerance::Extent);
    let mut exact_zeros = 0;
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
        let (got, read) = term.eval(chord.row(), [x, y]);
        let case = format!("{label} #{index} at ({x}, {y}) {chord:?}");
        let [lo, hi] = chord.clipped(at[1]);
        if hi <= lo || chord.wholly_left(at) {
            assert_eq!(got, 0.0, "{case}: exactly 0");
            exact_zeros += 1;
        }
        worst.judge(&case, at, (got, read.share(at)), chord.tolerance(at));
    }
    eprintln!("adversarial {label}: {exact_zeros} exact zeros");
    worst.report(label);
}

/// **Chords, slope a uniform** spanning `0` to `10⁸` either way.
#[test]
fn every_chord_is_its_rows_share() {
    run_chords("uniform slope", None, 0xad00);
}

/// **A literal slope**, so the closed form constant-folds around it. `0`
/// was a regression of the integral's closing: a provably zero slope made
/// a sweep provably zero, a quotient by it let the algebra merge it with
/// arbitrary classes, and a chord's whole area extracted as the constant
/// `0` (`mean_of_clamp`, "The divisor"). The closed form divides only by a
/// root's floored denominator.
#[test]
fn a_literal_slope_is_its_rows_share() {
    let slopes: [f32; 12] = [
        0.0,
        -0.0,
        1e-30,
        -1e-7,
        // 2⁻¹⁹ and 3·2⁻¹⁸.
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
        run_chords(&format!("literal k = {k:e}"), Some(k), 0xbe00 + i as u64);
    }
}

/// **Rows of chords, collapsed whole.** A chord's term over a `16 × 8`
/// frame placed at `|X|, |Y| ≈ 1000`, so each batch mixes lanes left of, on
/// and right of the crossing — and rows whose band misses them entirely.
#[test]
fn collapsed_rows_of_chords_are_their_rows_shares() {
    let mut piecewise = Piecewise::new("rows of chords", Steps::Two);
    let mut worst = Worst::new(Tolerance::Extent);
    for (i, origin) in [[0.0f32, 0.0], [1000.0, -777.0], [-996.0, 1003.0]]
        .into_iter()
        .enumerate()
    {
        let mut rng = Rng(0xd000 + i as u64);
        for index in 0..60 {
            let slope = (rng.pick(&[0.0, 1e-7, 2e-6, 0.25, 1.0, 3.0, 40.0, 1e8])
                * rng.pick(&[1.0, -1.0])) as f32;
            let centre_y = f64::from(origin[1]) + rng.range(0.0, ROWS as f64);
            let x_mid = f64::from(origin[0]) + rng.range(0.0, WIDTH as f64);
            let band = [
                (centre_y - rng.range(0.0, 3.0)) as f32,
                (centre_y + rng.range(0.0, 3.0)) as f32,
            ];
            let chord = Chord {
                sigma: rng.pick(&[1.0, -1.0]),
                band,
                anchor: [x_mid as f32, centre_y as f32],
                slope,
            };
            let row = chord.row();
            let frame = piecewise.frame(row, origin, Denormals::Kept);
            for (at, got) in frame.texels() {
                let case = (origin, index, chord);
                worst.judge(&case, at, (got, row.share(at)), chord.tolerance(at));
            }
        }
    }
    worst.report("rows of chords");
}
