//! The area left of a monotone quadratic arc, closed by `ArcMoment`, judged
//! by exact areas computed in `f64` by clipping polygons — a method that
//! shares nothing with pixelflow.
//!
//! Step 5 of docs/plans/2026-09-23-an-integral-is-a-fold.md §8. The author
//! writes a glyph piece's crossing as a graph over `y` through the arc's own
//! parameter,
//!
//! ```text
//! σ·area([0 ≤ T]·[T < 1]·[x < x₀ + T·(2β + α·T)]).at(X, S·Y)
//! T = (y − y₀) / max(b + √max(b² + a·(y − y₀), 0), 2⁻¹⁰⁰)
//! ```
//!
//! with every control-polygon step floored at `0` — the certificate that
//! makes both coordinates rise for any value a table holds — and the rules
//! close it: `FactorFold` takes the band out of the inner integral,
//! `NarrowInterval` closes that to a clamp, and `ArcMoment` closes the
//! outer one. A line is the arc whose bend is zero, so one integrand serves
//! every piece. This file checks:
//!
//! - **(a) The value, piece by piece.** Random monotone quadratics — built
//!   monotone, their control point inside the box of their ends — curved,
//!   straight (`a = 0`), nearly horizontal, nearly vertical, nearly
//!   collinear, tiny, hundreds of pixels long, placed up to `1000` from the
//!   origin, and with ends on pixel corners and extrema (a zero step) on
//!   pixel edges. Each is compiled once over uniforms, collapsed over a
//!   `16 × 8` frame, and every texel compared.
//! - **(b) A closed contour** of lines and arcs, as **one fold with one
//!   body** over a table of rows — the shape a glyph takes — summing to the
//!   polygon-clip coverage of the whole contour.
//! - **(c) That the closed form is what was judged**: `unclosed_integrals`
//!   is `0` for the single piece, for a line written with literal columns,
//!   and for the contour's fold.
//! - **(d) What no rule closes is still right.** With one step's
//!   certificate removed the outer integral stays open, `resolve` legalizes
//!   it by one-point quadrature, and each texel is the exact coverage of the
//!   pixel's centre line.
//! - **(e) No integral reaches the emitter**, and no reciprocal estimate is
//!   extracted, for any kernel above.
//!
//! **The reference** clips a polygon to the pixel (Sutherland–Hodgman) and
//! takes its shoelace area. The region is the arc — flattened to at least
//! `256` chords, and every chord whose box meets the pixel cut down to
//! `REFINE` — closed by a vertical far to its left, so the polygon is the
//! region left of the arc within its band. A monotone arc stays inside the
//! box of its ends, so a chord whose box misses the pixel changes nothing
//! the pixel sees. Flattening errs by the area between the chords and the
//! arc, which halves quarter-fold when every chord is halved; the reference
//! is the Richardson extrapolation `(4·A(2n) − A(n))/3` of two flattenings.
//!
//! **The tolerance** is `2⁻²²·(1 + |X| + |Y| + 2·extent)` per texel: the
//! parameter an arc is read at resolves to `2⁻²⁴`, which the arc's extent
//! multiplies, and the coordinates carry their own rounding into the
//! clamp (critique of the step-5 design, §2).

#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use std::sync::Arc;

use pixelflow_core::{DiscreteManifold, Kernel, Manifold, PlaneRegion, Uniform, UniformBlock};
use pixelflow_ir::arena::BufferIdentity;
use pixelflow_ir::integral::ROOT_FLOOR;
use pixelflow_ir::passes::lattice::{Collapse, Domain};
use pixelflow_ir::{ExprArena, ExprId, ExprNode, Fold, LatticeShape, OpKind};
use pixelflow_search::runtime::{optimize_runtime_arena, unclosed_integrals};

/// `2⁻²²`: the tolerance's unit.
const TOLERANCE_UNIT: f64 = 1.0 / 4_194_304.0;
/// Texels per frame row.
const WIDTH: usize = 16;
/// Frame rows.
const ROWS: usize = 8;
/// Pieces per category.
const CASES: usize = 48;
/// Chords the reference flattens every arc into before refining.
const CHORDS: usize = 256;
/// The largest chord, in pixels, the reference keeps where it meets the
/// pixel.
const REFINE: f64 = 1.0 / 1024.0;
/// How deep the reference halves a chord, at most.
const MAX_DEPTH: u32 = 40;
/// Rows the contour's table holds; a contour with fewer pads with zeros,
/// which contribute exactly nothing.
const TABLE_ROWS: usize = 10;

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

    /// An integer in `[lo, hi]`.
    fn int(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next() % (hi - lo + 1) as u64) as i64
    }

    fn pick<T: Copy>(&mut self, from: &[T]) -> T {
        from[(self.next() % from.len() as u64) as usize]
    }

    fn sign(&mut self) -> f64 {
        self.pick(&[-1.0, 1.0])
    }
}

// ───────────────────────────── the reference ─────────────────────────────

/// A point.
type Point = [f64; 2];

/// Clip a polygon to `{p : inside(p) ≤ 0}`, `inside` affine. The clip is
/// convex, so a concave polygon comes out with its area — and its winding —
/// intact, whatever degenerate edges it gains along the clip line.
fn clip(polygon: &[Point], inside: impl Fn(Point) -> f64) -> Vec<Point> {
    let mut out = Vec::with_capacity(polygon.len() + 4);
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

/// The signed shoelace area, counter-clockwise positive.
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

/// A pixel, `[x₀, x₁) × [y₀, y₁)`.
#[derive(Clone, Copy, Debug)]
struct Rect {
    x: [f64; 2],
    y: [f64; 2],
}

impl Rect {
    /// The pixel `Kernel::area` integrates about `centre`.
    fn pixel(centre: Point) -> Self {
        Self {
            x: [centre[0] - 0.5, centre[0] + 0.5],
            y: [centre[1] - 0.5, centre[1] + 0.5],
        }
    }

    /// The signed area of `polygon` inside this rectangle: the integral of
    /// its winding number over the pixel.
    fn signed_area_of(self, polygon: &[Point]) -> f64 {
        let polygon = clip(polygon, |p| self.x[0] - p[0]);
        let polygon = clip(&polygon, |p| p[0] - self.x[1]);
        let polygon = clip(&polygon, |p| self.y[0] - p[1]);
        let polygon = clip(&polygon, |p| p[1] - self.y[1]);
        if polygon.len() < 3 {
            return 0.0;
        }
        shoelace(&polygon)
    }

    /// Whether the box spanned by `p` and `q` meets this rectangle.
    fn meets_box_of(self, p: Point, q: Point) -> bool {
        p[0].min(q[0]) <= self.x[1]
            && p[0].max(q[0]) >= self.x[0]
            && p[1].min(q[1]) <= self.y[1]
            && p[1].max(q[1]) >= self.y[0]
    }
}

/// A monotone quadratic piece exactly as the kernel reads it — `f32`
/// columns, so the reference integrates the arc the kernel was given — in
/// the frame the host orients it to, where both coordinates rise.
#[derive(Clone, Copy, Debug)]
struct Piece {
    /// `(x₀, S·y₀)`: the start, `y` reflected.
    start: [f32; 2],
    /// `p₁ − p₀`, oriented: both components `≥ 0`.
    first: [f32; 2],
    /// `p₂ − p₁`, oriented: both components `≥ 0`.
    second: [f32; 2],
    /// `+1` for a piece that rises on the screen, `−1` for one that falls.
    sigma: f32,
    /// `S`: `−1` when the host reflected `y` to make it rise.
    reflect: f32,
    /// Whether the host reversed the piece to make `x` rise — so the
    /// contour runs along it from `t = 1` to `t = 0`.
    reversed: bool,
}

impl Piece {
    /// A row of the table, in [`Column`] order.
    fn row(self) -> [f32; Column::COUNT] {
        [
            self.start[0],
            self.start[1],
            self.first[0],
            self.first[1],
            self.second[0],
            self.second[1],
            self.sigma,
            self.reflect,
        ]
    }

    /// The arc at `t`, on the screen: the Bézier through the columns'
    /// control points, in `f64`.
    fn at(self, t: f64) -> Point {
        let f = f64::from;
        let rise = |i: usize| {
            let (s, e0, e1) = (f(self.start[i]), f(self.first[i]), f(self.second[i]));
            s + t * (2.0 * e0 + (e1 - e0) * t)
        };
        [rise(0), f(self.reflect) * rise(1)]
    }

    /// The larger of the piece's two spans.
    fn extent(self) -> f64 {
        let f = f64::from;
        (f(self.first[0]) + f(self.second[0])).max(f(self.first[1]) + f(self.second[1]))
    }

    /// Where the reference cuts the arc for `rect`: [`CHORDS`] chords, each
    /// whose box meets the pixel halved down to `chord` pixels across.
    fn cuts(self, rect: Rect, chord: f64) -> Vec<f64> {
        let mut cuts = vec![0.0];
        let n = CHORDS as f64;
        for i in 0..CHORDS {
            // Depth first, the earlier half on top, so the cuts come out in
            // order along the arc.
            let mut stack = vec![(i as f64 / n, (i + 1) as f64 / n, 0)];
            while let Some((t0, t1, depth)) = stack.pop() {
                let (p, q) = (self.at(t0), self.at(t1));
                let size = (p[0] - q[0]).abs().max((p[1] - q[1]).abs());
                if depth < MAX_DEPTH && size > chord && rect.meets_box_of(p, q) {
                    let mid = 0.5 * (t0 + t1);
                    stack.push((mid, t1, depth + 1));
                    stack.push((t0, mid, depth + 1));
                    continue;
                }
                cuts.push(t1);
            }
        }
        cuts
    }

    /// The region left of the arc within its band, as a polygon over the
    /// arc cut at `cuts`, closed by a vertical left of `rect`.
    fn left_region(self, cuts: &[f64], rect: Rect) -> Vec<Point> {
        let mut polygon: Vec<Point> = cuts.iter().map(|&t| self.at(t)).collect();
        let (start, end) = (self.at(0.0), self.at(1.0));
        let far = rect.x[0].min(start[0]) - 1.0;
        polygon.push([far, end[1]]);
        polygon.push([far, start[1]]);
        polygon
    }

    /// `σ·|pixel ∩ region left of the arc|`, exact to the reference's own
    /// flattening.
    fn coverage(self, centre: Point) -> f64 {
        self.coverage_to(centre, REFINE)
    }

    /// [`Piece::coverage`], flattened to chords `chord` pixels across where
    /// they meet the pixel.
    fn coverage_to(self, centre: Point, chord: f64) -> f64 {
        let rect = Rect::pixel(centre);
        let coarse = self.cuts(rect, chord);
        let area = |cuts: &[f64]| rect.signed_area_of(&self.left_region(cuts, rect)).abs();
        f64::from(self.sigma) * richardson(area(&coarse), area(&halved(&coarse)))
    }

    /// The exact coverage of the pixel's centre line: where the line
    /// `y = Y` crosses the arc, found by bisection on the parameter, `None`
    /// within `margin` of the band's ends, where the kernel's `f32`
    /// parameter and this one may land on opposite sides of an indicator.
    fn centre_line(self, centre: Point, margin: f64) -> Option<f64> {
        let f = f64::from;
        let level = f(self.reflect) * centre[1];
        let (low, high) = (
            f(self.start[1]),
            f(self.start[1]) + f(self.first[1]) + f(self.second[1]),
        );
        if (level - low).abs() < margin || (level - high).abs() < margin {
            return None;
        }
        if level < low || level >= high {
            return Some(0.0);
        }
        let height = |t: f64| f(self.reflect) * self.at(t)[1];
        let [mut t0, mut t1] = [0.0f64, 1.0];
        for _ in 0..200 {
            let mid = 0.5 * (t0 + t1);
            if height(mid) < level {
                t0 = mid;
            } else {
                t1 = mid;
            }
        }
        let crossing = self.at(0.5 * (t0 + t1))[0];
        Some(f(self.sigma) * (crossing - (centre[0] - 0.5)).clamp(0.0, 1.0))
    }

    /// The tolerance for `centre`, see the module doc.
    fn tolerance(self, centre: Point) -> f64 {
        TOLERANCE_UNIT * (1.0 + centre[0].abs() + centre[1].abs() + 2.0 * self.extent())
    }
}

/// `cuts` with the midpoint of every pair inserted: every chord halved.
fn halved(cuts: &[f64]) -> Vec<f64> {
    let mut fine = Vec::with_capacity(2 * cuts.len());
    for pair in cuts.windows(2) {
        fine.push(pair[0]);
        fine.push(0.5 * (pair[0] + pair[1]));
    }
    fine.extend(cuts.last());
    fine
}

/// `(4·A(2n) − A(n))/3`: the flattening's leading error, cancelled.
fn richardson(coarse: f64, fine: f64) -> f64 {
    (4.0 * fine - coarse) / 3.0
}

// ───────────────────────────── the kernel ─────────────────────────────

/// A piece's columns, in the order a row of the table holds them.
#[derive(Clone, Copy)]
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
    const COUNT: usize = 8;
    const ALL: [Self; Self::COUNT] = [
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

/// Which steps the integrand certifies.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Certificate {
    /// Every step, as an author writes it.
    Every,
    /// Not the first `y` step: the rule's side condition fails, on purpose.
    NotTheFirstY,
}

/// `[mask]`: the one place a mask becomes a number.
fn indicator(mask: &Kernel) -> Kernel {
    mask.select(&Kernel::constant(1.0), &Kernel::constant(0.0))
}

/// The author's crossing term, each column read through `column`.
fn crossing(column: &dyn Fn(Column) -> Kernel, certificate: Certificate) -> Kernel {
    let c = Kernel::constant;
    let (zero, one) = (c(0.0), c(1.0));
    let certify = |step| column(step).max(&zero);
    let b = match certificate {
        Certificate::Every => certify(Column::FirstY),
        Certificate::NotTheFirstY => column(Column::FirstY),
    };
    let bx = certify(Column::FirstX);
    let a = certify(Column::SecondY).sub(&b);
    let ax = certify(Column::SecondX).sub(&bx);
    let d = Kernel::y().sub(&column(Column::Y0));
    let root = b.mul(&b).add(&a.mul(&d)).max(&zero).sqrt();
    let t = d.div(&b.add(&root).max(&c(ROOT_FLOOR)));
    let xt = column(Column::X0).add(&t.mul(&bx.add(&bx).add(&ax.mul(&t))));
    let chi = indicator(&zero.le(&t))
        .mul(&indicator(&t.lt(&one)))
        .mul(&indicator(&Kernel::x().lt(&xt)));
    let screen_y = column(Column::Reflect).mul(&Kernel::y());
    column(Column::Sigma).mul(&chi.area().at(&Kernel::x(), &screen_y))
}

/// `k` translated so the frame's first texel is `origin`'s pixel.
fn placed(k: &Kernel, origin: &[Uniform; 2]) -> Kernel {
    k.at(
        &Kernel::x().add(&origin[0].kernel()),
        &Kernel::y().add(&origin[1].kernel()),
    )
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

/// **(c) and (e).** `unclosed` integrals survive the runtime tier's
/// extraction of `kernel`; the optimized term holds no reciprocal estimate
/// (the closed form's quotients are exact divides, and an estimate would
/// cost `2⁻¹²` of the area); and `legalize` — what `emit::compile` runs
/// before it schedules — leaves no interval fold in the arena, as built or
/// as optimized.
fn assert_legal(name: &str, kernel: &Kernel, unclosed: usize) {
    let (arena, root) = kernel.parts();
    let shape = LatticeShape::new([WIDTH as u32, ROWS as u32]);
    assert_eq!(
        unclosed_integrals(arena, root, shape),
        Some(unclosed),
        "{name}: integrals the extraction left unclosed"
    );
    let optimized =
        optimize_runtime_arena(arena, root, shape).expect("the runtime tier optimizes an integral");
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
            !reaches(&legal, legal_root, |node| matches!(
                node,
                ExprNode::Reduce {
                    fold: Fold::Interval(_),
                    ..
                }
            )),
            "{name} ({route}): an integral survived legalize"
        );
    }
}

/// One piece's term over uniforms, placed by an origin uniform, compiled
/// once at the frame and collapsed per piece.
struct PieceProgram {
    columns: [Uniform; Column::COUNT],
    origin: [Uniform; 2],
    manifold: Manifold,
    block: UniformBlock,
}

impl PieceProgram {
    fn new(name: &str, certificate: Certificate, unclosed: usize) -> Self {
        let columns = Column::ALL.map(|_| Uniform::new(0.0));
        let origin = [Uniform::new(0.0), Uniform::new(0.0)];
        let term = crossing(&|c: Column| columns[c as usize].kernel(), certificate);
        let kernel = placed(&term, &origin);
        assert_legal(name, &kernel, unclosed);
        let manifold = Manifold::compile(&kernel, [WIDTH as u32, ROWS as u32]);
        let block = manifold.block();
        Self {
            columns,
            origin,
            manifold,
            block,
        }
    }

    /// The frame for `piece`, its first texel's pixel at `origin`.
    fn collapse(&mut self, piece: Piece, origin: [f32; 2]) -> Vec<f32> {
        for (uniform, value) in self.columns.iter().zip(piece.row()) {
            self.block.set(*uniform, value).expect("a column");
        }
        for (uniform, value) in self.origin.iter().zip(origin) {
            self.block.set(*uniform, value).expect("the origin");
        }
        let mut out = vec![f32::NAN; WIDTH * ROWS];
        self.manifold
            .bind(&[])
            .with_uniforms(&self.block)
            .collapse_rows(PlaneRegion::rows(WIDTH, 0, ROWS), &mut out, WIDTH);
        out
    }
}

/// The screen coordinate of texel `(col, row)`'s centre in a frame whose
/// first texel's pixel is at `origin`.
fn texel_centre(origin: [f32; 2], col: usize, row: usize) -> Point {
    [
        f64::from(origin[0]) + col as f64 + 0.5,
        f64::from(origin[1]) + row as f64 + 0.5,
    ]
}

/// The largest error a category saw, absolute and against its tolerance,
/// and how many of its texels the arc cut — covered neither wholly nor not
/// at all, the ones a closed form can get wrong without a pin noticing.
#[derive(Default)]
struct Worst {
    error: f64,
    ratio: f64,
    samples: usize,
    cut: usize,
}

/// A texel covered by less than this, or by more than `1 −` this, is not
/// counted as cut.
const UNCUT: f64 = 1.0e-3;

impl Worst {
    fn check(&mut self, case: &dyn core::fmt::Debug, got: f32, want: f64, tolerance: f64) {
        let error = (f64::from(got) - want).abs();
        assert!(
            error <= tolerance,
            "{case:?}: got {got}, f64 {want}, error {error:e} > tolerance {tolerance:e}"
        );
        self.error = self.error.max(error);
        self.ratio = self.ratio.max(error / tolerance);
        self.samples += 1;
        let covered = want.abs().fract();
        if covered > UNCUT && covered < 1.0 - UNCUT {
            self.cut += 1;
        }
    }

    /// Report the category, and require that the arcs cut some of its
    /// texels: a category of whole and empty texels would pass a closed
    /// form that only got the pins right.
    fn report(&self, name: &str) {
        eprintln!(
            "arc_oracle {name}: {} texels ({} cut), max |error| {:.3e}, error/tolerance {:.3}",
            self.samples, self.cut, self.error, self.ratio
        );
        assert!(self.cut > 0, "{name}: no texel was cut");
    }
}

// ───────────────────────────── the pieces ─────────────────────────────

/// A piece's shape before it is placed: its spans (both `≥ 0`) and where
/// its control point sits in the box of its ends, per axis (`0` at the
/// start's side, `1` at the end's).
#[derive(Clone, Copy, Debug)]
struct Shape {
    span: [f64; 2],
    bulge: [f64; 2],
}

impl Shape {
    /// The piece with this shape starting at `start` on the screen, going
    /// up (`reflect = 1`) or down (`−1`), and reversed or not by the host.
    fn at(self, start: Point, reflect: f32, reversed: bool) -> Piece {
        let step = |axis: usize| {
            let first = self.bulge[axis] * self.span[axis];
            [first as f32, (self.span[axis] - first) as f32]
        };
        let (x, y) = (step(0), step(1));
        let rho = if reversed { -1.0 } else { 1.0 };
        Piece {
            start: [start[0] as f32, reflect * start[1] as f32],
            first: [x[0], y[0]],
            second: [x[1], y[1]],
            sigma: rho * reflect,
            reflect,
            reversed,
        }
    }
}

/// How a category draws its shapes and where it puts them.
struct Category {
    name: &'static str,
    seed: u64,
    /// The shape of a piece.
    shape: fn(&mut Rng) -> Shape,
    /// How far from the origin the frame may be.
    reach: i64,
    /// Whether the piece's end sits on a pixel corner.
    on_the_grid: bool,
}

/// A bulge anywhere in the box.
fn any_bulge(rng: &mut Rng) -> [f64; 2] {
    [rng.range(0.0, 1.0), rng.range(0.0, 1.0)]
}

/// Run `CASES` pieces of `category` through the exact integrand.
fn pieces(category: Category) {
    let Category {
        name,
        seed,
        shape,
        reach,
        on_the_grid,
    } = category;
    let mut program = PieceProgram::new(name, Certificate::Every, 0);
    let mut rng = Rng(seed);
    let mut worst = Worst::default();
    for index in 0..CASES {
        let origin = [rng.int(-reach, reach) as f32, rng.int(-reach, reach) as f32];
        let shape = shape(&mut rng);
        let reflect = rng.sign() as f32;
        let reversed = rng.next().is_multiple_of(2);
        // A point of the frame the piece passes through, at a random
        // fraction of its length — or, on the grid, its start or end on a
        // pixel corner.
        let (anchor, fraction) = if on_the_grid {
            let corner = [
                f64::from(origin[0]) + rng.int(2, WIDTH as i64 - 2) as f64,
                f64::from(origin[1]) + rng.int(1, ROWS as i64 - 1) as f64,
            ];
            (corner, rng.pick(&[0.0, 1.0]))
        } else {
            let inside = [
                f64::from(origin[0]) + rng.range(0.0, WIDTH as f64),
                f64::from(origin[1]) + rng.range(0.0, ROWS as f64),
            ];
            (inside, rng.range(0.0, 1.0))
        };
        let start = [
            anchor[0] - fraction * shape.span[0],
            anchor[1] - f64::from(reflect) * fraction * shape.span[1],
        ];
        let piece = shape.at(start, reflect, reversed);
        let frame = program.collapse(piece, origin);
        for row in 0..ROWS {
            for col in 0..WIDTH {
                let centre = texel_centre(origin, col, row);
                let case = (name, index, col, row, piece, shape);
                worst.check(
                    &case,
                    frame[row * WIDTH + col],
                    piece.coverage(centre),
                    piece.tolerance(centre),
                );
            }
        }
    }
    worst.report(name);
}

/// **(a) Curves**, a few pixels long, near the origin.
#[test]
fn a_curved_piece_is_its_exact_area() {
    pieces(Category {
        name: "curves",
        seed: 0xa4c0_0001,
        shape: |rng| Shape {
            span: [rng.range(0.5, 12.0), rng.range(0.5, 7.0)],
            bulge: any_bulge(rng),
        },
        reach: 16,
        on_the_grid: false,
    });
}

/// **(a) Lines**: the control point on the chord's midpoint, so the bend is
/// exactly zero on both axes — the same integrand, the same rule.
#[test]
fn a_straight_piece_is_its_exact_area() {
    pieces(Category {
        name: "lines",
        seed: 0xa4c0_0002,
        shape: |rng| Shape {
            span: [rng.range(0.0, 12.0), rng.range(0.25, 7.0)],
            bulge: [0.5, 0.5],
        },
        reach: 16,
        on_the_grid: false,
    });
}

/// **(a) Far from the origin**: frames anywhere within `1000`, where the
/// coordinates' own rounding is `2⁻¹⁴`.
#[test]
fn a_piece_far_from_the_origin_is_its_exact_area() {
    pieces(Category {
        name: "far",
        seed: 0xa4c0_0003,
        shape: |rng| Shape {
            span: [rng.range(0.5, 12.0), rng.range(0.5, 7.0)],
            bulge: any_bulge(rng),
        },
        reach: 1000,
        on_the_grid: false,
    });
}

/// **(a) Nearly flat on one axis**: spans from `10⁻⁷` to `10⁻³` in `y`
/// (nearly horizontal), or in `x` (nearly vertical), and exactly zero —
/// a horizontal piece, whose band is empty, and a vertical one, whose
/// roots in `x` divide by the floor.
#[test]
fn a_nearly_flat_piece_is_its_exact_area() {
    pieces(Category {
        name: "near-flat",
        seed: 0xa4c0_0004,
        shape: |rng| {
            let flat = rng.pick(&[0.0, 1.0e-7, 1.0e-5, 1.0e-3]);
            let long = rng.range(0.5, 7.0);
            let span = match rng.next() % 2 {
                0 => [long, flat],
                _ => [flat, long],
            };
            Shape {
                span,
                bulge: any_bulge(rng),
            }
        },
        reach: 64,
        on_the_grid: false,
    });
}

/// **(a) Nearly collinear, thin, and hooked**: the control point within
/// `10⁻⁶`–`10⁻³` of the chord, or within `10⁻⁴` of a corner of the box —
/// where one step is nearly zero and the arc turns sharply.
#[test]
fn a_nearly_degenerate_piece_is_its_exact_area() {
    pieces(Category {
        name: "near-degenerate",
        seed: 0xa4c0_0005,
        shape: |rng| {
            let along = rng.range(0.0, 1.0);
            let bulge = match rng.next() % 3 {
                0 => {
                    let off = rng.sign() * rng.pick(&[1.0e-6, 1.0e-4, 1.0e-3]);
                    [along, (along + off).clamp(0.0, 1.0)]
                }
                1 => [rng.pick(&[1.0e-4, 1.0 - 1.0e-4]), along],
                _ => [
                    rng.pick(&[1.0e-4, 1.0 - 1.0e-4]),
                    rng.pick(&[1.0e-4, 1.0 - 1.0e-4]),
                ],
            };
            Shape {
                span: [rng.range(0.5, 9.0), rng.range(0.5, 7.0)],
                bulge,
            }
        },
        reach: 64,
        on_the_grid: false,
    });
}

/// **(a) Tiny pieces**, `10⁻³` to `0.3` of a pixel across.
#[test]
fn a_tiny_piece_is_its_exact_area() {
    pieces(Category {
        name: "tiny",
        seed: 0xa4c0_0006,
        shape: |rng| {
            let scale = rng.pick(&[1.0e-3, 1.0e-2, 0.3]);
            Shape {
                span: [rng.range(0.0, scale), rng.range(0.1 * scale, scale)],
                bulge: any_bulge(rng),
            }
        },
        reach: 64,
        on_the_grid: false,
    });
}

/// **(a) Long pieces**, `100` to `600` pixels, of which the frame sees a
/// part: the parameter's `2⁻²⁴` is multiplied by the whole extent.
#[test]
fn a_long_piece_is_its_exact_area() {
    pieces(Category {
        name: "long",
        seed: 0xa4c0_0007,
        shape: |rng| Shape {
            span: [rng.range(100.0, 600.0), rng.range(100.0, 600.0)],
            bulge: any_bulge(rng),
        },
        reach: 400,
        on_the_grid: false,
    });
}

/// **(a) On the grid**: integer spans, a start or an end on a pixel corner,
/// and the control point at a corner of the box or its middle — so a step
/// is zero and the arc's extremum (a horizontal or vertical tangent) lies
/// on a pixel edge, where the host's split would put it.
#[test]
fn a_piece_on_the_pixel_grid_is_its_exact_area() {
    pieces(Category {
        name: "grid",
        seed: 0xa4c0_0008,
        shape: |rng| Shape {
            span: [rng.int(0, 8) as f64, rng.int(1, 6) as f64],
            bulge: [rng.pick(&[0.0, 0.5, 1.0]), rng.pick(&[0.0, 0.5, 1.0])],
        },
        reach: 64,
        on_the_grid: true,
    });
}

/// **The judge's own error.** The reference, flattened four times finer,
/// agrees with itself far below any tolerance above — on curves, hooks and
/// long arcs, at pixels the arc crosses — so what the categories measure
/// is the kernel's error and not the reference's.
#[test]
fn the_reference_converges() {
    let mut rng = Rng(0xa4c0_000b);
    let mut worst: f64 = 0.0;
    for _ in 0..200 {
        let span = rng.pick(&[4.0, 40.0, 400.0]);
        let bulge = match rng.next() % 2 {
            0 => any_bulge(&mut rng),
            _ => [
                rng.pick(&[1.0e-4, 1.0 - 1.0e-4]),
                rng.pick(&[1.0e-4, 1.0 - 1.0e-4]),
            ],
        };
        let shape = Shape {
            span: [rng.range(0.1, 1.0) * span, rng.range(0.1, 1.0) * span],
            bulge,
        };
        let piece = shape.at([rng.range(-8.0, 8.0), rng.range(-8.0, 8.0)], 1.0, false);
        let on = piece.at(rng.range(0.0, 1.0));
        let centre = [on[0].round(), on[1].round()];
        let (coarse, fine) = (
            piece.coverage_to(centre, REFINE),
            piece.coverage_to(centre, REFINE / 4.0),
        );
        worst = worst.max((coarse - fine).abs());
    }
    eprintln!("arc_oracle reference: max |A(REFINE) − A(REFINE/4)| {worst:.3e}");
    assert!(worst <= 1.0e-9, "the reference moves by {worst:e}");
}

/// **(c) A line written with literal columns closes**, and is its exact
/// area: every column a constant, the bend folded to a literal zero.
#[test]
fn a_literal_line_closes_to_its_exact_area() {
    let piece = Shape {
        span: [5.0, 3.0],
        bulge: [0.5, 0.5],
    }
    .at([3.25, 2.5], 1.0, false);
    let row = piece.row();
    let term = crossing(
        &|c: Column| Kernel::constant(row[c as usize]),
        Certificate::Every,
    );
    assert_legal("literal line", &term, 0);
    let manifold = Manifold::compile(&term, [WIDTH as u32, ROWS as u32]);
    let mut out = vec![f32::NAN; WIDTH * ROWS];
    manifold
        .bind(&[])
        .collapse_rows(PlaneRegion::rows(WIDTH, 0, ROWS), &mut out, WIDTH);
    let mut worst = Worst::default();
    for row in 0..ROWS {
        for col in 0..WIDTH {
            let centre = texel_centre([0.0, 0.0], col, row);
            worst.check(
                &(col, row),
                out[row * WIDTH + col],
                piece.coverage(centre),
                piece.tolerance(centre),
            );
        }
    }
    worst.report("literal line");
}

/// **(d) What no rule closes is legalized, and right.** The first `y` step
/// read raw: the rule cannot see that the arc rises, declines, and leaves
/// the outer integral — the inner one still narrows to its clamp, which
/// extraction keeps. `resolve` replaces the outer integral by its one-point
/// quadrature, so each texel is the exact coverage of the pixel's centre
/// line — judged by bisection on the `f64` arc, away from the band's ends.
#[test]
fn an_uncertified_piece_is_its_centre_line_coverage() {
    let mut program = PieceProgram::new("uncertified", Certificate::NotTheFirstY, 1);
    let mut rng = Rng(0xa4c0_0009);
    let mut worst = Worst::default();
    for index in 0..CASES {
        let origin = [rng.int(-500, 500) as f32, rng.int(-500, 500) as f32];
        let shape = Shape {
            span: [rng.range(0.0, 12.0), rng.range(0.5, 7.0)],
            bulge: any_bulge(&mut rng),
        };
        let reflect = rng.sign() as f32;
        let start = [
            f64::from(origin[0]) + rng.range(0.0, WIDTH as f64) - 0.5 * shape.span[0],
            f64::from(origin[1]) + rng.range(0.0, ROWS as f64)
                - f64::from(reflect) * 0.5 * shape.span[1],
        ];
        let piece = shape.at(start, reflect, rng.next().is_multiple_of(2));
        let frame = program.collapse(piece, origin);
        for row in 0..ROWS {
            for col in 0..WIDTH {
                let centre = texel_centre(origin, col, row);
                let margin = 1.0 / 65_536.0 * (1.0 + centre[1].abs() + piece.extent());
                let Some(want) = piece.centre_line(centre, margin) else {
                    continue;
                };
                let case = (index, col, row, piece);
                let tolerance = piece.tolerance(centre);
                worst.check(&case, frame[row * WIDTH + col], want, tolerance);
            }
        }
    }
    worst.report("uncertified (centre line)");
}

// ───────────────────────────── the contour ─────────────────────────────

/// A closed contour's pieces, as the host would orient them: a star-shaped
/// polygon about `centre` whose edges are quadratics with their control
/// point in the box of their ends — some exactly straight — on a grid of
/// `1/1024`, so every column is exact in `f32` and consecutive pieces meet
/// exactly.
fn contour(rng: &mut Rng, centre: Point) -> Vec<Piece> {
    let on_grid = |v: f64, per: f64| (v * per).round() / per;
    let count = rng.int(4, TABLE_ROWS as i64) as usize;
    let mut angles: Vec<f64> = (0..count)
        .map(|_| rng.range(0.0, core::f64::consts::TAU))
        .collect();
    angles.sort_by(f64::total_cmp);
    let vertices: Vec<Point> = angles
        .iter()
        .map(|&angle| {
            let radius = rng.range(1.0, 3.5);
            [
                on_grid(centre[0] + radius * angle.cos(), 64.0),
                on_grid(centre[1] + radius * angle.sin(), 64.0),
            ]
        })
        .collect();
    (0..count)
        .map(|k| {
            let (p0, p2) = (vertices[k], vertices[(k + 1) % count]);
            let straight = rng.next().is_multiple_of(3);
            let bulge = |rng: &mut Rng| {
                if straight {
                    0.5
                } else {
                    on_grid(rng.range(0.0, 1.0), 16.0)
                }
            };
            let bx = bulge(rng);
            let by = if straight { 0.5 } else { bulge(rng) };
            let p1 = [p0[0] + bx * (p2[0] - p0[0]), p0[1] + by * (p2[1] - p0[1])];
            orient([p0, p1, p2])
        })
        .collect()
}

/// A quadratic `p₀ → p₂` as the host orients it: reversed so `x` rises,
/// then `y` reflected so it rises too.
fn orient([p0, p1, p2]: [Point; 3]) -> Piece {
    let reversed = p2[0] < p0[0];
    let (p0, p2) = if reversed { (p2, p0) } else { (p0, p2) };
    let reflect: f32 = if p2[1] < p0[1] { -1.0 } else { 1.0 };
    let s = f64::from(reflect);
    let rho = if reversed { -1.0 } else { 1.0 };
    Piece {
        start: [p0[0] as f32, (s * p0[1]) as f32],
        first: [(p1[0] - p0[0]) as f32, (s * (p1[1] - p0[1])) as f32],
        second: [(p2[0] - p1[0]) as f32, (s * (p2[1] - p1[1])) as f32],
        sigma: rho * reflect,
        reflect,
        reversed,
    }
}

/// The contour's winding number integrated over the pixel at `centre`:
/// every piece flattened along the contour's own direction, the closed
/// polygon clipped to the pixel.
fn contour_coverage(pieces: &[Piece], centre: Point) -> f64 {
    let rect = Rect::pixel(centre);
    let polygon = |refine: fn(&[f64]) -> Vec<f64>| -> Vec<Point> {
        pieces
            .iter()
            .flat_map(|piece| {
                let mut cuts = refine(&piece.cuts(rect, REFINE));
                if piece.reversed {
                    cuts.reverse();
                }
                cuts.into_iter().map(|t| piece.at(t)).collect::<Vec<_>>()
            })
            .collect()
    };
    let area = |refine| rect.signed_area_of(&polygon(refine));
    richardson(area(|cuts| cuts.to_vec()), area(halved))
}

/// **(b) A closed contour is one fold with one body.** Every piece of a
/// contour — lines and arcs alike — is a row of one table, and the glyph's
/// coverage is `Σ_p σ_p·area(χ_p).at(X, S_p·Y)`, one range fold whose body
/// reads its columns at the fold's index. **(c)** No integral survives in
/// it. The frame's texels sum the pieces to the contour's winding-number
/// coverage, judged by clipping the whole flattened contour to each pixel.
#[test]
fn a_contour_of_lines_and_arcs_is_its_exact_coverage() {
    let id = BufferIdentity::mint();
    let table = DiscreteManifold::kernel_for(id, Column::COUNT as u32, TABLE_ROWS as u32);
    let origin = [Uniform::new(0.0), Uniform::new(0.0)];
    let glyph = Kernel::sum_over(TABLE_ROWS as u32, |p| {
        crossing(
            &|c: Column| table.at(&Kernel::constant(c as usize as f32), p),
            Certificate::Every,
        )
    });
    let kernel = placed(&glyph, &origin);
    assert_legal("contour", &kernel, 0);
    let manifold = Manifold::compile(&kernel, [WIDTH as u32, ROWS as u32]);
    let mut block = manifold.block();
    let mut rng = Rng(0xa4c0_000a);
    let mut worst = Worst::default();
    for index in 0..CASES {
        let reach = rng.pick(&[0, 1000]);
        let at = [rng.int(-reach, reach) as f32, rng.int(-reach, reach) as f32];
        for (uniform, value) in origin.iter().zip(at) {
            block.set(*uniform, value).expect("the origin");
        }
        let centre = [
            f64::from(at[0]) + WIDTH as f64 / 2.0 + rng.range(-2.0, 2.0),
            f64::from(at[1]) + ROWS as f64 / 2.0 + rng.range(-0.5, 0.5),
        ];
        let pieces = contour(&mut rng, centre);
        let mut data = vec![0.0f32; Column::COUNT * TABLE_ROWS];
        for (row, piece) in pieces.iter().enumerate() {
            data[row * Column::COUNT..(row + 1) * Column::COUNT].copy_from_slice(&piece.row());
        }
        let mut out = vec![f32::NAN; WIDTH * ROWS];
        manifold
            .bind(&[(id, Arc::new(data))])
            .with_uniforms(&block)
            .collapse_rows(PlaneRegion::rows(WIDTH, 0, ROWS), &mut out, WIDTH);
        for row in 0..ROWS {
            for col in 0..WIDTH {
                let texel = texel_centre(at, col, row);
                let tolerance: f64 = pieces.iter().map(|piece| piece.tolerance(texel)).sum();
                let case = (index, col, row, &pieces);
                worst.check(
                    &case,
                    out[row * WIDTH + col],
                    contour_coverage(&pieces, texel),
                    tolerance,
                );
            }
        }
    }
    worst.report("contour");
}
