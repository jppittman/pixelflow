//! `ArcMoment`'s closed form, attacked where the e-graph can see the most
//! and the floating point is thinnest, and judged by a reference that shares
//! nothing with it or with `arc_oracle`.
//!
//! Step 5 of docs/plans/2026-09-23-an-integral-is-a-fold.md §8, adversarial
//! review. `arc_oracle` judges random pieces over uniforms by clipping
//! polygons; this file aims at what the e-graph can prove about a piece and
//! the spellings an author can write:
//!
//! - **Literal zeros.** Every column a constant — horizontal and vertical
//!   lines, a zero step at either end on either axis, pieces on pixel edges
//!   and corners — so the algebra folds the certificates and sees the zeros.
//! - **A bend the algebra proves zero over a step it cannot fold.** One
//!   uniform read as both steps of a line. Found a miscompile: each root's
//!   denominator stopped varying, and `δ/d` extracted as `δ·recip(d)`, a
//!   hoisted 12–14-bit estimate (`monotone_root`, "Emitted as `δ·(1/d)`").
//! - **Every floor `RootFloor` admits**, with denormals kept and flushed as
//!   the renderer's workers flush them. Found a miscompile: a subnormal floor
//!   was admitted, and read `1` for `0` (`RootFloor`, "Normal").
//! - **Spellings.** The crossing with literal coefficients (`3`, `−0.1`),
//!   moved across the comparison, in a frame scaled in `x` (so the rule's
//!   scaled rise is live), the band's other strictness; and ones that must
//!   decline — a step floored below zero, a raw step, the region right of
//!   the arc.
//! - **Orientation and reach.** Pieces running in all four diagonal
//!   directions, frames to `±1000`, spans to `600` pixels, control points
//!   hooked into the corners of their box, steps exactly zero over
//!   uniforms, and a table of closed contours as one `Σ_p` fold.
//!
//! **The reference** is Green's theorem on the raw quadratic, in the
//! direction its contour runs, in `f64`: a piece's share of the pixel's
//! winding integral is
//! `∫₀¹ clamp(x(t) − X + ½, 0, 1)·[|y(t) − Y| < ½]·y′(t) dt`. Cut at every
//! parameter where `x` or `y` meets an edge of the pixel, the integrand is a
//! polynomial of degree at most three on each piece, which three-point
//! Gauss–Legendre integrates exactly. No closed form and no polygon; the
//! host's reversal and reflection are this file's own, and the kernel is
//! judged against the raw piece. [`the_reference_is_greens_theorem`] checks
//! the reference against a rectangle's overlap, computed by hand.

#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use std::sync::Arc;

use pixelflow_core::{
    DiscreteManifold, FastMathGuard, Kernel, Manifold, PlaneRegion, Uniform, UniformBlock,
};
use pixelflow_ir::arena::BufferIdentity;
use pixelflow_ir::integral::ROOT_FLOOR;
use pixelflow_ir::{ExprArena, ExprId, ExprNode, LatticeShape, OpKind};
use pixelflow_search::runtime::{optimize_runtime_arena, unclosed_integrals};

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

/// Which side of a piece its share is read on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Side {
    /// `clamp(x − X + ½, 0, 1)`: the pixel's row left of the piece.
    Left,
    /// `1 − clamp(x − X + ½, 0, 1)`: the row right of it.
    Right,
}

/// A quadratic Bézier on the screen, running `p₀ → p₂` as its contour does.
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

    /// `∫₀¹ g(x(t), y(t))·y′(t) dt`, `g` a polynomial between the
    /// parameters in `cuts`: three-point Gauss–Legendre on each piece.
    fn integrate(self, mut cuts: Vec<f64>, g: impl Fn(Point) -> f64) -> f64 {
        const NODE: f64 = 0.774_596_669_241_483_4; // √(3/5)
        const RULE: [(f64, f64); 3] = [(-NODE, 5.0 / 9.0), (0.0, 8.0 / 9.0), (NODE, 5.0 / 9.0)];
        cuts.extend([0.0, 1.0]);
        cuts.sort_by(f64::total_cmp);
        cuts.windows(2)
            .map(|w| {
                let (mid, half) = (0.5 * (w[0] + w[1]), 0.5 * (w[1] - w[0]));
                let sum: f64 = RULE
                    .iter()
                    .map(|&(node, weight)| {
                        let t = mid + half * node;
                        let p = [self.coordinate(0, t), self.coordinate(1, t)];
                        weight * g(p) * self.velocity(1, t)
                    })
                    .sum();
                sum * half
            })
            .sum()
    }

    /// The piece's share of the winding number integrated over the pixel
    /// at `centre`, read on `side` of it.
    fn share_on(self, side: Side, centre: Point) -> f64 {
        let [x, y] = centre;
        let mut cuts = Vec::new();
        for (axis, c) in [(0, x), (1, y)] {
            cuts.extend(self.meets(axis, c - 0.5));
            cuts.extend(self.meets(axis, c + 0.5));
        }
        self.integrate(cuts, |p| {
            if p[1] < y - 0.5 || p[1] >= y + 0.5 {
                return 0.0;
            }
            let left = (p[0] - (x - 0.5)).clamp(0.0, 1.0);
            match side {
                Side::Left => left,
                Side::Right => 1.0 - left,
            }
        })
    }

    /// [`Quad::share_on`] the left: what the author's crossing integrates.
    fn share(self, centre: Point) -> f64 {
        self.share_on(Side::Left, centre)
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

// ───────────────────────────── the host ─────────────────────────────

/// A piece's columns, in the order a table row holds them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Column {
    X0,
    Y0,
    FirstX,
    FirstY,
    SecondX,
    SecondY,
    Sigma,
    Reflect,
}

impl Column {
    const ALL: [Self; 8] = [
        Self::X0,
        Self::Y0,
        Self::FirstX,
        Self::FirstY,
        Self::SecondX,
        Self::SecondY,
        Self::Sigma,
        Self::Reflect,
    ];
}

/// A table row: the piece reversed so `x` rises, then `y` reflected so it
/// rises too, every value exact in `f32`.
#[derive(Clone, Copy, Debug)]
struct Row {
    x0: f32,
    y0: f32,
    first: [f32; 2],
    second: [f32; 2],
    sigma: f32,
    reflect: f32,
}

impl Row {
    /// # Panics
    ///
    /// If a column is not exact in `f32`, or the piece is not monotone.
    fn of(quad: Quad) -> Self {
        let [mut p0, p1, mut p2] = quad.0;
        let reversed = p2[0] < p0[0];
        if reversed {
            core::mem::swap(&mut p0, &mut p2);
        }
        let s = if p2[1] < p0[1] { -1.0 } else { 1.0 };
        let exact = |v: f64| {
            let f = v as f32;
            assert_eq!(f64::from(f), v, "{quad:?}: {v} is not an f32");
            f
        };
        let row = Self {
            x0: exact(p0[0]),
            y0: exact(s * p0[1]),
            first: [exact(p1[0] - p0[0]), exact(s * (p1[1] - p0[1]))],
            second: [exact(p2[0] - p1[0]), exact(s * (p2[1] - p1[1]))],
            sigma: exact(if reversed { -s } else { s }),
            reflect: exact(s),
        };
        for v in [row.first, row.second].concat() {
            assert!(v >= 0.0, "{quad:?} is not monotone: {row:?}");
        }
        row
    }

    fn column(self, c: Column) -> f32 {
        match c {
            Column::X0 => self.x0,
            Column::Y0 => self.y0,
            Column::FirstX => self.first[0],
            Column::FirstY => self.first[1],
            Column::SecondX => self.second[0],
            Column::SecondY => self.second[1],
            Column::Sigma => self.sigma,
            Column::Reflect => self.reflect,
        }
    }
}

// ───────────────────────────── the author ─────────────────────────────

fn c(v: f32) -> Kernel {
    Kernel::constant(v)
}

/// `[mask]`: the one place a mask becomes a number.
fn indicator(mask: &Kernel) -> Kernel {
    mask.select(&c(1.0), &c(0.0))
}

/// How the author spells the band and the crossing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Spelling {
    /// `[0 ≤ T]·[T < 1]·[x < x₀ + R]`: the plan's spelling.
    Plain,
    /// `[2x < x₀ + R]`: the arc drawn in a frame stretched twice in `x`, so
    /// narrowing's clamp reads `R` at the slope `½` — the rule's scaled
    /// rise, in an author's spelling.
    Squeezed,
    /// `[0.5·x < x₀ + R]`: the slope `2`.
    Stretched,
    /// `[3x < 3x₀ + 3R]`: narrowing's root a product by `fl(1/3)`.
    Thrice,
    /// `[−0.1·x > −0.1·(x₀ + R)]`: a negative coefficient that rounds.
    Tenth,
    /// `[x₀ + R − x > 0]`.
    Difference,
    /// `[x − x₀ < R]`.
    Moved,
    /// `[0 < T]·[T ≤ 1]` — the band's other strictness — and `[R + x₀ > x]`.
    OtherStrictness,
    /// Every step floored at `−2⁻¹⁰⁰`, not `0`: no certificate.
    NegativeFloor,
    /// The first `x` step raw: no certificate on the clamp's rise.
    RawX,
    /// `[x > x₀ + R]`: the region right of the arc, whose clamp falls as
    /// `R` rises. The rule reads only a positive slope, so this declines
    /// today; should it ever close, it must close to the right-hand share.
    RightOf,
}

impl Spelling {
    const ALL: [Self; 11] = [
        Self::Plain,
        Self::Squeezed,
        Self::Stretched,
        Self::Thrice,
        Self::Tenth,
        Self::Difference,
        Self::Moved,
        Self::OtherStrictness,
        Self::NegativeFloor,
        Self::RawX,
        Self::RightOf,
    ];

    /// How many integrals the rules leave open: none for an integrand the
    /// rule promises to read, one — the outer, for quadrature — for one it
    /// cannot certify, and either for one it may learn to read.
    fn open(self) -> Option<usize> {
        match self {
            Self::NegativeFloor | Self::RawX => Some(1),
            Self::RightOf => None,
            _ => Some(0),
        }
    }

    /// The author's arc for a piece on the screen: the same piece in the
    /// frame this spelling scales in `x` — by a power of two, so exactly.
    fn arc_of(self, screen: Quad) -> Quad {
        let scale = match self {
            Self::Squeezed => 2.0,
            Self::Stretched => 0.5,
            _ => 1.0,
        };
        Quad(screen.0.map(|[x, y]| [scale * x, y]))
    }

    /// Which side of the piece this spelling integrates.
    fn side(self) -> Side {
        match self {
            Self::RightOf => Side::Right,
            _ => Side::Left,
        }
    }
}

/// What the author writes: a spelling, and the floor under each root.
#[derive(Clone, Copy, Debug)]
struct Author {
    spelling: Spelling,
    floor: f32,
}

impl Author {
    /// The plan's spelling under `floor`.
    fn plain(floor: f32) -> Self {
        Self {
            spelling: Spelling::Plain,
            floor,
        }
    }

    /// `σ·area(band·[crossing]).at(X, S·Y)`, `T` the arc's parameter at the
    /// height `y − y₀` and `R = x(T)` its other coordinate from `x₀`:
    /// `T = d/max(b + √max(b² + a·d, 0), floor)`, `R = T·(β + β + α·T)`,
    /// every step floored at `0` unless the spelling says otherwise — each
    /// column read through `read`.
    fn term(self, read: &dyn Fn(Column) -> Kernel) -> Kernel {
        let spelling = self.spelling;
        let zero = c(0.0);
        let step_floor = match spelling {
            Spelling::NegativeFloor => c(-ROOT_FLOOR),
            _ => zero.clone(),
        };
        let step = |col| read(col).max(&step_floor);
        let b = step(Column::FirstY);
        let beta = match spelling {
            Spelling::RawX => read(Column::FirstX),
            _ => step(Column::FirstX),
        };
        let a = step(Column::SecondY).sub(&b);
        let alpha = step(Column::SecondX).sub(&beta);
        let d = Kernel::y().sub(&read(Column::Y0));
        let radicand = b.mul(&b).add(&a.mul(&d)).max(&zero);
        let t = d.div(&b.add(&radicand.sqrt()).max(&c(self.floor)));
        let r = t.mul(&beta.add(&beta).add(&alpha.mul(&t)));
        let (x, x0) = (Kernel::x(), read(Column::X0));
        let band = match spelling {
            Spelling::OtherStrictness => indicator(&zero.lt(&t)).mul(&indicator(&t.le(&c(1.0)))),
            _ => indicator(&zero.le(&t)).mul(&indicator(&t.lt(&c(1.0)))),
        };
        let crosses = match spelling {
            Spelling::Squeezed => c(2.0).mul(&x).lt(&x0.add(&r)),
            Spelling::Stretched => c(0.5).mul(&x).lt(&x0.add(&r)),
            Spelling::Thrice => c(3.0).mul(&x).lt(&c(3.0).mul(&x0).add(&c(3.0).mul(&r))),
            Spelling::Tenth => c(-0.1).mul(&x).gt(&c(-0.1).mul(&x0.add(&r))),
            Spelling::Difference => x0.add(&r).sub(&x).gt(&zero),
            Spelling::Moved => x.sub(&x0).lt(&r),
            Spelling::OtherStrictness => r.add(&x0).gt(&x),
            Spelling::RightOf => x.gt(&x0.add(&r)),
            Spelling::Plain | Spelling::NegativeFloor | Spelling::RawX => x.lt(&x0.add(&r)),
        };
        let chi = band.mul(&indicator(&crosses));
        let screen_y = read(Column::Reflect).mul(&Kernel::y());
        read(Column::Sigma).mul(&chi.area().at(&Kernel::x(), &screen_y))
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
    fn read_as(self, col: Column) -> Column {
        match (self, col) {
            (Self::OneReadTwice, Column::SecondX) => Column::FirstX,
            (Self::OneReadTwice, Column::SecondY) => Column::FirstY,
            _ => col,
        }
    }
}

// ───────────────────────────── the kernel ─────────────────────────────

fn shape() -> LatticeShape {
    LatticeShape::new([WIDTH as u32, ROWS as u32])
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

/// How many integrals the runtime tier leaves open in `kernel`, and whether
/// its optimized term holds a reciprocal estimate.
fn closure(kernel: &Kernel) -> (usize, bool) {
    let (arena, root) = kernel.parts();
    let open = unclosed_integrals(arena, root, shape()).expect("saturates");
    let optimized = optimize_runtime_arena(arena, root, shape()).expect("optimizes");
    let estimate = reaches(&optimized.0, optimized.1, |node| {
        matches!(node, ExprNode::Unary(OpKind::Recip | OpKind::Rsqrt, _))
    });
    (open, estimate)
}

/// `k` translated so the frame's first texel is `origin`'s pixel.
fn placed(k: &Kernel, origin: [&Kernel; 2]) -> Kernel {
    k.at(&Kernel::x().add(origin[0]), &Kernel::y().add(origin[1]))
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
    fn collapse(&self, origin: [f32; 2], tables: &[(BufferIdentity, Arc<Vec<f32>>)]) -> Frame {
        self.collapse_under(origin, tables, Denormals::Kept)
    }

    fn collapse_under(
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

/// One author's crossing over uniforms, placed by an origin uniform:
/// compiled once, collapsed per piece.
struct Piecewise {
    columns: [Uniform; 8],
    steps: Steps,
    origin: [Uniform; 2],
    program: Program,
    closure: (usize, bool),
}

impl Piecewise {
    fn new(author: Author, steps: Steps) -> Self {
        let columns = Column::ALL.map(|_| Uniform::new(0.0));
        let origin = [Uniform::new(0.0), Uniform::new(0.0)];
        let term = author.term(&|col| columns[steps.read_as(col) as usize].kernel());
        let kernel = placed(&term, [&origin[0].kernel(), &origin[1].kernel()]);
        Self {
            columns,
            steps,
            origin,
            program: Program::new(&kernel),
            closure: closure(&kernel),
        }
    }

    fn frame(&mut self, row: Row, origin: [f32; 2], denormals: Denormals) -> Frame {
        for col in Column::ALL {
            if self.steps.read_as(col) == col {
                self.program
                    .set(self.columns[col as usize], row.column(col));
            }
        }
        for (uniform, value) in self.origin.into_iter().zip(origin) {
            self.program.set(uniform, value);
        }
        self.program.collapse_under(origin, &[], denormals)
    }
}

// ───────────────────────────── the judge ─────────────────────────────

/// What a texel's error is held to.
#[derive(Clone, Copy, Debug)]
enum Tolerance {
    /// `16·2⁻²⁴·(1 + extent)`: the parameter an arc is read at resolves to
    /// `2⁻²⁴`, which its extent multiplies, a few times over — and nothing
    /// in where the pixel is, since with exact columns `x₀ − X` and
    /// `S·Y − y₀` are differences of exact values.
    Extent,
    /// [`Tolerance::Extent`] plus `4·2⁻²⁴·(|X| + |Y|)`: the rounding of a
    /// coordinate as large as the pixel's, which a literal column or
    /// coefficient costs. The algebra distributes a literal over
    /// `(Y + origin) − y₀` and folds the constant part, so the difference is
    /// no longer taken first — measured `2.2e-5` at `|Y| ≈ 1000` on a
    /// vertical line of literal columns, which over uniforms is exact.
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

/// What a frame is judged against: the pieces whose shares it sums, and
/// on which side.
struct Reference<'a> {
    pieces: &'a [Quad],
    side: Side,
}

impl<'a> Reference<'a> {
    fn left(pieces: &'a [Quad]) -> Self {
        Self {
            pieces,
            side: Side::Left,
        }
    }

    fn share(&self, centre: Point) -> f64 {
        let shares = self.pieces.iter().map(|q| q.share_on(self.side, centre));
        shares.sum()
    }

    fn extent(&self) -> f64 {
        self.pieces.iter().map(|q| q.extent()).sum()
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

    /// Every texel of `frame` against `reference`.
    fn frame(&mut self, case: &dyn core::fmt::Debug, frame: &Frame, reference: &Reference) {
        let extent = reference.extent();
        for (centre, got) in frame.texels() {
            let want = reference.share(centre);
            let tolerance = self.tolerance.at(extent, centre);
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
    }

    /// Report the category, and require that its pieces cut some texels.
    fn report(&self, name: &str) {
        eprintln!(
            "arc_adversarial {name}: {} texels ({} cut), max |error| {:.3e}, error/tolerance {:.3}",
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

// ───────────────────────────── the tests ─────────────────────────────

/// **The reference is Green's theorem.** The four edges of a rectangle,
/// counter-clockwise, sum over a pixel to the rectangle's overlap with it
/// — the product of two interval overlaps, by hand — and clockwise to its
/// negation; each edge read on the right sums to the pixel's rows the
/// rectangle spans, less that overlap.
#[test]
fn the_reference_is_greens_theorem() {
    let overlap = |[a, b]: [f64; 2], [c, d]: [f64; 2]| (b.min(d) - a.max(c)).max(0.0);
    let (x, y) = ([0.3, 4.7], [-1.25, 2.6]);
    let corners = [[x[0], y[0]], [x[1], y[0]], [x[1], y[1]], [x[0], y[1]]];
    let edge = |k: usize| Quad::line(corners[k], corners[(k + 1) % 4]);
    let reversed = |k: usize| Quad::line(corners[(k + 1) % 4], corners[k]);
    for i in -2..7 {
        for j in -3..4 {
            let centre = [f64::from(i) + 0.25, f64::from(j) + 0.5];
            let pixel = |c: f64| [c - 0.5, c + 0.5];
            let want = overlap(x, pixel(centre[0])) * overlap(y, pixel(centre[1]));
            let ccw: f64 = (0..4).map(|k| edge(k).share(centre)).sum();
            let cw: f64 = (0..4).map(|k| reversed(k).share(centre)).sum();
            let right: f64 = (0..4).map(|k| edge(k).share_on(Side::Right, centre)).sum();
            assert!((ccw - want).abs() < 1e-14, "{centre:?}: {ccw} vs {want}");
            assert!((cw + want).abs() < 1e-14, "{centre:?}: {cw} vs {want}");
            // A closed contour's right-hand reading is its rows' full width
            // times zero net crossings, less the area: `−want`.
            assert!(
                (right + want).abs() < 1e-14,
                "{centre:?}: {right} vs {want}"
            );
        }
    }
}

/// **Every direction, anywhere, any length.** Monotone quadratics running
/// in each of the four diagonal directions — reversed, reflected, both,
/// neither — frames from the origin to `±1000`, spans from a pixel to
/// `600`, a fifth of them flat on an axis (a uniform step of exactly zero,
/// whose root divides by the floor at run time), control points anywhere
/// or hooked into a corner of their box, on a fine grid or on pixel edges
/// and centres, collapsed with denormals kept and flushed.
#[test]
fn every_direction_is_the_raw_pieces_share() {
    let mut piecewise = Piecewise::new(Author::plain(ROOT_FLOOR), Steps::Two);
    assert_eq!(piecewise.closure, (0, false), "(unclosed, estimate)");
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
                let quad = draw.piece(&mut rng);
                let frame = piecewise.frame(Row::of(quad), origin, denormals);
                worst.frame(&(index, draw, quad), &frame, &Reference::left(&[quad]));
            }
        }
    }
    worst.report("every direction");
}

/// **Literal zeros.** Every column a constant, so the e-graph folds the
/// certificates and sees the zero steps: horizontal and vertical lines
/// (on pixel edges, corner to corner, running down), a zero first or second
/// step on either axis — an extremum at an end — a point, and a hook to an
/// edge, near the origin and at `±1000`, in every direction. All close,
/// with no reciprocal estimate, to the raw share within the offset
/// tolerance ([`Tolerance::Offset`]: literal columns are where the algebra
/// takes the coordinates' difference apart).
#[test]
fn literal_zero_steps_close_to_the_raw_share() {
    let mut worst = Worst::new(Tolerance::Offset);
    for base in [[0.0, 0.0], [-1000.0, 1000.0], [997.0, -1003.0]] {
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
            ("flat end", Quad([o(1.0, 1.25), o(1.0, 7.5), o(13.0, 7.5)])),
            (
                "steep start",
                Quad([o(2.5, 0.5), o(2.5, 5.0), o(12.0, 7.0)]),
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
        for (name, quad) in cases {
            let row = Row::of(quad);
            let origin = [base[0] as f32, base[1] as f32];
            let term = Author::plain(ROOT_FLOOR).term(&|col| c(row.column(col)));
            let kernel = placed(&term, [&c(origin[0]), &c(origin[1])]);
            assert_eq!(closure(&kernel), (0, false), "{name} at {base:?}");
            let frame = Program::new(&kernel).collapse(origin, &[]);
            worst.frame(&(name, quad), &frame, &Reference::left(&[quad]));
        }
    }
    worst.report("literal zeros");
}

/// **A line whose bend the algebra proves zero, over a step that is
/// neither literal nor varying.** A host that stores a line's one step and
/// reads it as both — the same uniform for `p₁ − p₀` and `p₂ − p₁` — gives
/// `a = max(s, 0) − max(s, 0)`. The rule closes the arc while that is still
/// a difference; the algebra proves it zero afterwards, and each root's
/// denominator stops varying. Spelled `δ/d`, the root was then extracted as
/// `δ·recip(d)` with the estimate hoisted: `3.9e-2` of coverage wrong at
/// AVX2 and `3.5e-3` at AVX-512 on the longest lines here.
#[test]
fn a_line_read_through_one_step_column() {
    let mut piecewise = Piecewise::new(Author::plain(ROOT_FLOOR), Steps::OneReadTwice);
    assert_eq!(piecewise.closure, (0, false), "(unclosed, estimate)");
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
            let quad = draw.piece(&mut rng);
            let frame = piecewise.frame(Row::of(quad), origin, denormals);
            worst.frame(&(index, draw, quad), &frame, &Reference::left(&[quad]));
        }
    }
    worst.report("one step column");
}

/// **Every floor the rule admits, and none it must not.** `RootFloor`
/// takes a normal literal no larger than `2⁻¹⁰⁰`: the largest, a smaller
/// one and the smallest normal each close, and are exact with denormals
/// kept and flushed. Over literal columns a flat rise's root is `δ` times
/// `1/floor` — the product that saturates the parameter — at a height `δ`
/// that is exactly zero where the piece starts on a pixel edge. A
/// subnormal floor is refused: under denormals-are-zero it is zero, and
/// its reciprocal overflows. Admitted, `2⁻¹³⁶` read `1` for `0` along the
/// edge its vertical line starts on.
#[test]
fn every_admitted_floor_closes_to_the_raw_share() {
    let pieces = [
        ("vertical from an edge", Quad::line([4.0, 1.0], [4.0, 6.5])),
        ("vertical", Quad::line([4.25, 1.0], [4.25, 6.5])),
        ("horizontal on an edge", Quad::line([1.0, 3.0], [9.0, 3.0])),
        (
            "flat start on an edge",
            Quad([[1.0, 2.0], [6.0, 2.0], [12.0, 7.0]]),
        ),
        (
            "steep start on a corner",
            Quad([[3.0, 1.0], [3.0, 5.0], [11.0, 7.0]]),
        ),
        ("line from a corner", Quad::line([2.0, 1.0], [14.0, 7.0])),
    ];
    let literal = |floor: f32, quad: Quad| {
        let row = Row::of(quad);
        Author::plain(floor).term(&|col| c(row.column(col)))
    };
    let mut worst = Worst::new(Tolerance::Extent);
    for floor in [ROOT_FLOOR, ROOT_FLOOR / 1024.0, f32::MIN_POSITIVE] {
        for (name, quad) in pieces {
            let kernel = literal(floor, quad);
            assert_eq!(closure(&kernel), (0, false), "floor {floor:e}, {name}");
            let program = Program::new(&kernel);
            for denormals in [Denormals::Kept, Denormals::Flushed] {
                let frame = program.collapse_under([0.0, 0.0], &[], denormals);
                let case = (floor, name, denormals, quad);
                worst.frame(&case, &frame, &Reference::left(&[quad]));
            }
        }
    }
    worst.report("floors");
    let subnormal = [
        f32::MIN_POSITIVE / 2.0,
        f32::MIN_POSITIVE / 1024.0,
        f32::from_bits(1),
    ];
    for floor in subnormal {
        for (name, quad) in pieces {
            let (open, _) = closure(&literal(floor, quad));
            assert_eq!(open, 1, "subnormal floor {floor:e}, {name}: declines");
        }
    }
}

/// **Every spelling closes to the same area, or declines.** A spelling the
/// rule promises to read closes, with no reciprocal estimate, to the raw
/// piece's share; one it cannot certify stays open, for quadrature. Pieces
/// to `300` pixels, frames to `±1000`, held to the offset tolerance: a
/// literal coefficient is what takes `x₀ − X` apart.
#[test]
fn every_spelling_closes_exactly_or_declines() {
    let mut rng = Rng(0xad0e_0003);
    for spelling in Spelling::ALL {
        let author = Author {
            spelling,
            floor: ROOT_FLOOR,
        };
        let mut piecewise = Piecewise::new(author, Steps::Two);
        let (open, estimate) = piecewise.closure;
        assert!(!estimate, "{spelling:?}: a reciprocal estimate");
        if let Some(expected) = spelling.open() {
            assert_eq!(open, expected, "{spelling:?}: integrals left open");
        }
        if open != 0 {
            continue;
        }
        let mut worst = Worst::new(Tolerance::Offset);
        for index in 0..100 {
            let reach = rng.pick(&[0.0, 1000.0]);
            let origin = rng.origin(reach);
            let long = rng.pick(&[1.0, 12.0, 300.0]);
            let draw = Draw {
                origin,
                span: [rng.range(0.0, long), rng.range(0.0, long)],
                direction: [rng.pick(&[-1.0, 1.0]), rng.pick(&[-1.0, 1.0])],
                bulge: Bulge::Anywhere,
                grid: 64.0,
            };
            let quad = draw.piece(&mut rng);
            let row = Row::of(spelling.arc_of(quad));
            let frame = piecewise.frame(row, origin, Denormals::Kept);
            let reference = Reference {
                pieces: &[quad],
                side: spelling.side(),
            };
            worst.frame(&(spelling, index, quad), &frame, &reference);
        }
        worst.report(&format!("{spelling:?}"));
    }
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
/// hundreds of pixels long, judged by summing the raw pieces' shares. With
/// [`Steps::OneReadTwice`] every row is a line whose one step column is
/// read as both — the bend the algebra can prove zero, per row.
fn table_contours(name: &str, seed: u64, steps: Steps) {
    let id = BufferIdentity::mint();
    let table = DiscreteManifold::kernel_for(id, Column::ALL.len() as u32, TABLE_ROWS as u32);
    let glyph = Kernel::sum_over(TABLE_ROWS as u32, |p| {
        let read = |col| table.at(&c(steps.read_as(col) as usize as f32), p);
        Author::plain(ROOT_FLOOR).term(&read)
    });
    let origin = [Uniform::new(0.0), Uniform::new(0.0)];
    let kernel = placed(&glyph, [&origin[0].kernel(), &origin[1].kernel()]);
    assert_eq!(closure(&kernel), (0, false), "{name}: (unclosed, estimate)");
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
        let quads = contour.pieces(&mut rng);
        let mut data = vec![0.0f32; Column::ALL.len() * TABLE_ROWS];
        for (k, quad) in quads.iter().enumerate() {
            let row = Row::of(*quad);
            for col in Column::ALL {
                data[k * Column::ALL.len() + col as usize] = row.column(col);
            }
        }
        let frame = program.collapse(at, &[(id, Arc::new(data))]);
        worst.frame(&(name, index, &quads), &frame, &Reference::left(&quads));
    }
    worst.report(name);
}

#[test]
fn a_contour_table_is_the_raw_pieces_shares() {
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
