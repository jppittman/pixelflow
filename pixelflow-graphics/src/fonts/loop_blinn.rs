//! Coverage of an outline as a [`Kernel`]: **the exact area of the pixel
//! under ink**, written in closed form.
//!
//! ## The formula
//!
//! A glyph is a finite set of pieces, each a quadratic arc that turns back
//! on neither axis. Its coverage at the pixel about the sample `s` is
//!
//! ```text
//! F(s)     = Σ_p σ_p · A_p(s)          the pixel's signed area under ink
//! coverage = min(|F|, 1)
//! ```
//!
//! `A_p` is the area of the pixel left of piece `p` within the piece's own
//! band of rows, and `σ_p` the direction it runs, `+1` toward `+Y`. The sum
//! is the winding number integrated over the pixel — Green's theorem, one
//! piece at a time — so a pixel deep inside reads `1`, one outside `0`, and
//! one on an edge the fraction of it the ink covers: the area, not a ramp on
//! a distance. Where contours overlap, `|F|` reaches 2 and the clamp folds
//! it, which is FreeType's approximation and accepted as such: a pixel where
//! two edges of two overlapping contours cross reads the overlap of two
//! fractions as one.
//!
//! ## Each term is written as its closed form
//!
//! `A_p` is a polynomial in the arc's own parameter, read between four
//! square roots: where the arc enters and leaves the pixel's rows, and
//! where it reaches the pixel's left and right edges
//! (`RisingArc::pixel_area`, which carries the derivation). It is the
//! trapezoid-and-cover accumulation of FreeType and font-rs, per pixel and
//! exact on a curve. A line is the arc whose bend is zero, so every piece
//! has the same term, and a glyph is **one fold with one body** over a table
//! of rows.
//!
//! The term is written, not derived. It used to be written as the integral
//! it is — `σ·area(χ).at(X, S·Y)`, `χ` the region left of the arc — and
//! closed by e-graph rules, which made a glyph's correctness a property of
//! the saturation budget. Under the flat class cap the rules stopped
//! closing past a few dozen integrals in one graph, and the one-point
//! quadrature legalized in their place point-sampled every edge it
//! reached, with coverage still in range and ink still where it was: a
//! 94-glyph run read up to 0.43 off before its folds shared one body
//! (docs/results/2026-09-23-glyph-is-a-formula.md, §3). A closed form has
//! no such dependence: whatever saturation does or does not reach, it
//! rewrites an area into an equal area. No interval fold is in a glyph's
//! arena (`tests/glyph_is_closed.rs`).
//!
//! `τ` is [`pixelflow_ir::integral::monotone_root`], the one definition.
//! Each control-polygon step is floored at `0` in the kernel — the
//! certificate that makes the arc rise for *any* number a table holds, so
//! the closed form's case analysis holds whatever the row. The host's split
//! (`MonotoneQuad`, in `fonts/monotone.rs`) is what makes the certified arc
//! the glyph's arc.
//!
//! ## The host orients every piece
//!
//! A quadratic is cut at its interior extrema, and each monotone arc is
//! turned so both coordinates rise: reversed if `x` falls (`ρ = −1`), then
//! reflected in `y` if `y` falls (`S = −1`), with `σ = ρ·S`. The kernel reads
//! the reflected arc at `S·Y` — the pixel is symmetric, so the reflected
//! pixel is the same pixel. A horizontal piece crosses no row and bounds no
//! area; it contributes exactly `0` and is dropped.
//!
//! ## Where a term is zero
//!
//! A piece's term is `0` wherever the pixel misses the piece's band, so the
//! body is cut to it — the band dilated by the pixel's half-height, in
//! screen `Y` — by an `If` whose mask depends on the row and the piece
//! alone. That is an identity of the formula, and its mask is uniform over
//! a batch: the shape a guard can jump, for every row the piece does not
//! reach. The same reasoning one level up: `F` is `0` outside the outline's
//! box dilated by half a pixel, so [`glyph`] cuts its kernel to the box
//! dilated by [`RAMP_REACH`] and reports that box as its [`Support`].
//!
//! ## Exactly 0, exactly 1
//!
//! The pieces' bands telescope only to an `f32` ulp — a piece's end is
//! computed, the next piece's start is stored — so a pixel wholly inside
//! reads `1` only to a few ulps. Coverage within [`COVERAGE_SNAP`] of `0`
//! or `1` is snapped to it: a quarter of an 8-bit step, far above that
//! rounding and far below anything a frame can show.
//!
//! The module keeps its name from the Loop–Blinn glyph it replaced
//! (docs/plans/2026-09-08-loop-blinn-glyph.md): an exact winding number and
//! a ramp on the distance to the nearest edge, two folds where this is one.

use super::monotone::MonotoneQuad;
use super::outline::{Outline, Point, Segment};
use pixelflow_core::{BoundManifold, DiscreteManifold, Kernel, Lattice, Manifold, Monoid, Uniform};
use pixelflow_ir::arena::BufferIdentity;
use pixelflow_ir::integral::{self, RootFloor, ROOT_FLOOR};
use pixelflow_ir::{library, ExprArena, ExprId, OpKind};
use std::collections::BTreeMap;
use std::sync::Arc;

/// How far coverage can reach past the outline, in the frame the kernel is
/// built in. The pixel is a unit square about the sample, so half a unit is
/// the analytic reach; a whole unit keeps the bound a whole pixel when that
/// frame is the screen, at no cost — the mask sits where every term is
/// already exactly zero.
pub const RAMP_REACH: f32 = 1.0;

/// Coverage this close to `0` or `1` is `0` or `1`: `2⁻¹⁰`, a quarter of an
/// 8-bit step.
///
/// The ends have to be exact. A pixel deep inside a glyph is `1`, not
/// `1 − 2⁻²⁰` — the frame's pack truncates, so the second is byte 254 — and
/// the pieces' bands telescope only to an ulp of the coordinates they
/// share. Measured before the snap: interior texels off by up to `1.9e-6`
/// (the step-5 design's gate prediction). The snap is an `If` on the
/// finished coverage, which no rewrite rule touches.
pub const COVERAGE_SNAP: f32 = 1.0 / 1024.0;

/// A segment whose ends are closer than this is dropped: a line that short
/// covers nothing, and a quadratic whose ends coincide retraces itself and
/// encloses no area — cut at its turn, its two halves would cancel only to
/// rounding.
const ZERO_LENGTH: f64 = 1e-6;

/// Half the pixel's side: the pixel about `(X, Y)` is
/// `[X − ½, X + ½) × [Y − ½, Y + ½)`. A term is zero wherever the pixel's
/// rows miss the piece's band, so the band is dilated by this before it is
/// stored.
const PIXEL_HALF: f32 = 0.5;

/// `⅓`, rounded once: the closed form's cubic coefficient is a product by
/// it, never a quotient by `3` (see [`RisingArc::pixel_area`], "Floating
/// point").
const ONE_THIRD: f32 = 1.0 / 3.0;

// ═══════════════════════════════════════════════════════════════════════════
// The glyph, and the box outside which it is exactly zero
// ═══════════════════════════════════════════════════════════════════════════

/// A coverage kernel's signed area together with the box outside which it
/// is exactly zero. They travel together because they are derived together
/// — a support restated separately from the composition it describes is a
/// future divergence.
///
/// **The signed area, not the coverage.** Coverage is not additive —
/// summing two glyphs' coverages reaches 2 where their ink overlaps, and 2
/// is not a coverage — so a glyph that could only offer its finished
/// coverage could not compose. The signed area *is* additive, and zero
/// outside every closed contour, so keeping it until the last step is what
/// makes a run a glyph (docs/plans/2026-09-09-a-run-is-a-glyph.md).
///
/// The area reads one piece table at its fold's binder ([`glyph`]); the
/// table's data travels with the kernel itself (`Kernel::with_buffer_data`),
/// so there is no separate binding a caller must keep paired with it.
#[derive(Clone)]
pub struct Glyph {
    /// Zero outside every contour — which is what lets [`Self::over`] mask
    /// it to a box and lose nothing.
    area: SignedArea,
    /// Where it can be nonzero.
    pub support: Support,
}

impl Glyph {
    /// The identity of [`Self::over`]: no ink anywhere. `text("")` is this.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            area: SignedArea::zero(),
            support: Support::EMPTY,
        }
    }

    /// **The one exit.** Coverage, cut to the support so the box's claim
    /// holds under any later coordinate warp rather than only where the
    /// terms happen to vanish.
    ///
    /// A method rather than a field because it is derived: two glyphs
    /// compose through [`Self::over`] on the signed area, and a stored
    /// coverage would be a second representation to keep in step.
    #[must_use]
    pub fn kernel(&self) -> Kernel {
        self.support
            .contains()
            .select(&coverage(&self.area), &constant(0.0))
    }

    /// **Glyphs form a monoid, and this is its operation.** Signed areas
    /// sum and supports union — each componentwise, so the whole is
    /// associative and commutative and a run's coverage does not depend on
    /// the order its characters were laid out in.
    ///
    /// Each contributor is masked to its own support first, and that is
    /// *exact*, not an approximation: a closed contour's signed area is
    /// **0** at every pixel its box does not reach (`Contour::new` refusing
    /// an open contour is what licenses this), and [`Support::around`]
    /// dilates the box past the pixel's own reach.
    ///
    /// The mask is also the *binning*: structurally every character is still
    /// in the graph — `If` is dispatch and both arms stay live — but a
    /// SIMD batch is adjacent pixels almost always inside one character's
    /// box, so the emitter's guard can skip the rest.
    #[must_use]
    pub fn over(glyphs: &[Glyph]) -> Self {
        let mut areas = Vec::with_capacity(glyphs.len());
        let mut support = Support::EMPTY;
        for g in glyphs {
            areas.push(g.area.masked(&g.support.contains()));
            support = support.union(g.support);
        }
        Self {
            area: SignedArea::sum(&areas),
            support,
        }
    }

    /// `kernel` — [`Self::kernel`] itself, or a `Kernel::at` contramap of
    /// it (a placement, a pixel-center shift) — compiled at `extent`. The
    /// piece table the fold reads travels with `kernel` itself (see
    /// [`Self`]'s docs), so there is nothing further to bind here — a bare
    /// `Lattice::bake` still refuses `kernel` because it *declares* a
    /// buffer, so this goes through `Manifold::compile`/`bind` directly,
    /// with an empty binding list.
    #[must_use]
    pub fn bound(&self, kernel: &Kernel, extent: [u32; 2]) -> BoundManifold {
        Manifold::compile(kernel, extent).bind(&[])
    }

    /// Tabulate `kernel` over `lattice`: compile at its extent, bind
    /// (trivially — see [`Self::bound`]), collapse.
    #[must_use]
    pub fn bake(&self, kernel: &Kernel, lattice: Lattice) -> DiscreteManifold {
        lattice.collapse(&self.bound(kernel, lattice.extent))
    }
}

/// **The box outside which a coverage [`Kernel`] is exactly zero.**
///
/// The outline's bounding box dilated by [`RAMP_REACH`], in the frame the
/// kernel was built in; [`glyph`] cuts its kernel to this box, so the claim
/// holds under any coordinate warp and not only where the terms' own
/// vanishing makes it true. Coordinates are `[x0, y0, x1, y1]`; a box with
/// no area is [`Support::EMPTY`] and meets nothing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Support([f32; 4]);

impl Support {
    /// No support at all: a glyph with no outline is the constant 0 everywhere.
    pub const EMPTY: Self = Self([0.0, 0.0, 0.0, 0.0]);

    /// The bounding box `[x0, y0, x1, y1]` dilated by [`RAMP_REACH`].
    fn around([x0, y0, x1, y1]: [f32; 4]) -> Self {
        Self([
            x0 - RAMP_REACH,
            y0 - RAMP_REACH,
            x1 + RAMP_REACH,
            y1 + RAMP_REACH,
        ])
    }

    /// `[x0, y0, x1, y1]`.
    #[must_use]
    pub fn bounds(self) -> [f32; 4] {
        self.0
    }

    /// Whether the box encloses no samples.
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.0[2] <= self.0[0] || self.0[3] <= self.0[1]
    }

    /// The mask that is set exactly inside this box — the binning test, and
    /// the cut [`Glyph::kernel`] applies so the box's claim survives a warp.
    ///
    /// **The four edges are arguments, not constants**, and that is the whole
    /// reason a font needs so few programs. A box spelled as four `Const`
    /// leaves is *data living in the program*: the key `canonical` builds
    /// digests a `Const`'s bits, so two glyphs alike in every other way
    /// compiled to two different regions, and 95 ASCII glyphs minted 90
    /// distinct programs. A `Uniform` keys by its **dense slot** — "the
    /// default is the block's business, not the code's" — so the same glyphs
    /// now share one region per shape and differ only in the block bound
    /// beside it.
    ///
    /// Minting here rather than storing handles on `Self` keeps [`Support`] a
    /// plain geometric value (`Copy`, comparable, `const`-constructible), and
    /// puts the mint at the one point where the box stops being host data and
    /// becomes program. Two calls on one box are therefore two argument sets,
    /// exactly as [`Uniform`]'s own contract says; each call site below makes
    /// one call per box, which is the arity that means.
    ///
    /// The defaults carry the box, so nothing downstream binds a block: a
    /// `Manifold` reads its own `defaults` off the link it was compiled
    /// against, and `bind` binds every argument at its default.
    fn contains(self) -> Kernel {
        let [x0, y0, x1, y1] = self.0;
        let edge = |v: f32| Uniform::new(v).kernel();
        Kernel::x()
            .ge(&edge(x0))
            .and(&Kernel::x().le(&edge(x1)))
            .and(&Kernel::y().ge(&edge(y0)))
            .and(&Kernel::y().le(&edge(y1)))
    }

    /// The smallest box containing both — the support of two glyphs laid
    /// side by side. Empty is the identity, which is what makes
    /// [`Glyph::over`] a monoid on this component too.
    #[must_use]
    fn union(self, other: Self) -> Self {
        match (self.is_empty(), other.is_empty()) {
            (true, _) => other,
            (_, true) => self,
            _ => Self([
                self.0[0].min(other.0[0]),
                self.0[1].min(other.0[1]),
                self.0[2].max(other.0[2]),
                self.0[3].max(other.0[3]),
            ]),
        }
    }
}

/// The coverage of `outline` as ONE kernel over the whole plane, in the
/// outline's own frame, together with its [`Support`].
///
/// Every piece contributes at every sample whose row its band reaches, so
/// the cost is linear in the outline's piece count. The kernel is cut to its
/// support by a mask whose false arm is the literal 0, so it composes into a
/// larger scene as a guardable arm.
#[must_use]
pub fn glyph(outline: &Outline) -> Glyph {
    run(core::slice::from_ref(outline))
}

/// The coverage of several outlines as ONE [`Glyph`], each keeping its own
/// bounding box so a sample pays only for the outlines whose box contains it.
///
/// **One table, not one per outline.** A piece table is a tabulation of
/// *pieces*; which outline a piece came from is a row range, not a separate
/// buffer. Giving each character its own table is what a frame cannot afford
/// — `Manifold` binds at most `MAX_BOUND_BUFFERS` slots without allocating,
/// so a five-character run would not compile
/// (docs/plans/2026-09-09-composition-is-linking.md §4 named this limit as
/// the cost of naming memory a slot at a time). Concatenating the rows and
/// folding each outline over its own range of them binds one slot however
/// long the run is.
///
/// **One body, not one per outline.** Where an outline's rows start is its
/// fold's range, not an offset inside the body, so every outline's fold
/// reads `table[i]` through the same body, and the e-graph holds that body
/// once however long the run is. With the offset in the body each outline's
/// term was a body of its own — and while a term was an integral the rules
/// closed, past about thirty characters the closing phase ran into the
/// saturation's class cap and left the rest to one-point quadrature:
/// point-sampled, aliased glyphs.
///
/// [`glyph`] is this at one outline.
#[must_use]
pub fn run(outlines: &[Outline]) -> Glyph {
    // Host side: every outline's rows appended into one table, and the row
    // range plus bounding box each will fold over.
    let mut rows: Vec<f32> = Vec::new();
    let mut spans: Vec<(u32, u32, [f32; 4])> = Vec::new();
    for outline in outlines {
        let pieces = pieces(outline);
        let (Some(bounds), false) = (outline.bounds(), pieces.is_empty()) else {
            continue;
        };
        let start = u32::try_from(rows.len() / PIECE_ROW_COLS)
            .expect("a run has far fewer pieces than u32::MAX");
        let count =
            u32::try_from(pieces.len()).expect("an outline has far fewer pieces than u32::MAX");
        // The fold's trip count is part of the JIT cache's key
        // (`docs/plans/2026-09-09-glyph-as-a-fold-execution.md` §S3), so a
        // glyph's own piece count is rounded up to a shared bucket *before*
        // it becomes that trip count — otherwise every distinct piece count
        // in the font mints its own program. The gap between `count` and
        // the bucket is filled with `padding_row`s, each an exact identity
        // of the fold (see its own docs), so this changes which program a
        // glyph shares, never the glyph.
        let padded = bucketed_trip_count(count);
        rows.extend(pieces.iter().flat_map(|p| piece_row(*p)));
        rows.extend((count..padded).flat_map(|_| padding_row()));
        spans.push((start, padded, bounds));
    }
    if spans.is_empty() {
        return Glyph::empty();
    }

    // `DiscreteManifold::new` mints the table's own identity and `.kernel()`
    // seeds the fragment with the piece data itself
    // (`Kernel::with_buffer_data`), so the data travels with every fold below
    // rather than riding beside them in a field a caller has to keep paired.
    // Every span reads the same identity, so they bind one slot between them.
    let height = rows.len() / PIECE_ROW_COLS;
    let table = DiscreteManifold::new(rows, PIECE_ROW_COLS, height).kernel();

    let placed: Vec<Glyph> = spans
        .iter()
        .map(|&(start, count, bounds)| {
            let area = SignedArea(Kernel::over(Monoid::SUM, start..start + count, |i| {
                piece_term(&row_at(&table, i))
            }));
            Glyph {
                area,
                support: Support::around(bounds),
            }
        })
        .collect();
    Glyph::over(&placed)
}

// ═══════════════════════════════════════════════════════════════════════════
// Host geometry: the outline as oriented monotone arcs, in f64
// ═══════════════════════════════════════════════════════════════════════════

type P = [f64; 2];

fn wide([x, y]: Point) -> P {
    [f64::from(x), f64::from(y)]
}

fn distance(a: P, b: P) -> f64 {
    (b[0] - a[0]).hypot(b[1] - a[1])
}

/// One monotone arc of the outline, turned so both of its coordinates rise:
/// the numbers one row of the table holds.
#[derive(Clone, Copy, Debug)]
struct Piece {
    /// `p₀`, its `y` reflected by [`Self::reflect`].
    start: P,
    /// `p₁ − p₀`, oriented: neither component negative.
    first: P,
    /// `p₂ − p₁`, oriented: neither component negative.
    second: P,
    /// `σ = ρ·S`: `+1` where the outline runs toward `+Y` along this piece,
    /// `−1` where it runs toward `−Y` — the winding a ray to `+X` picks up
    /// crossing it.
    sigma: f64,
    /// `S`: `−1` where `y` was reflected to make it rise, else `+1`.
    reflect: f64,
    /// `[min, max]` of its `y` on the screen, unreflected.
    rows: [f64; 2],
}

impl Piece {
    /// The monotone arc `arc`, turned so `x` and `y` rise — or `None` for a
    /// horizontal one, which crosses no row and bounds no area.
    ///
    /// Reversed if `x` falls (`ρ = −1`): the arc is the same curve run the
    /// other way, so its steps swap and change sign. Then reflected in `y` if
    /// `y` falls (`S = −1`). Both keep a monotone arc monotone, and after
    /// both every step is non-negative — which is exactly the certificate
    /// the kernel floors each step with, so the floor never moves a real
    /// row.
    fn oriented(arc: MonotoneQuad) -> Option<Self> {
        let [p0, p1, p2] = arc.points();
        // Monotone in `y`, so the control point's `y` lies between the ends'.
        if p0[1] == p2[1] {
            return None;
        }
        let (rho, [p0, p2]) = match p2[0] < p0[0] {
            true => (-1.0, [p2, p0]),
            false => (1.0, [p0, p2]),
        };
        let s = match p2[1] < p0[1] {
            true => -1.0,
            false => 1.0,
        };
        Some(Self {
            start: [p0[0], s * p0[1]],
            first: [p1[0] - p0[0], s * (p1[1] - p0[1])],
            second: [p2[0] - p1[0], s * (p2[1] - p1[1])],
            sigma: rho * s,
            reflect: s,
            rows: [p0[1].min(p2[1]), p0[1].max(p2[1])],
        })
    }
}

/// `outline` as oriented monotone arcs, every contour's in order.
///
/// A line is the quadratic whose control point is its midpoint — monotone
/// already, and cut by nothing. A quadratic is cut at its interior extrema
/// ([`MonotoneQuad::split`]) — after dropping it if its ends coincide, since
/// such a curve retraces itself, encloses nothing, and has a cusp on both
/// axes that the cut would turn into two halves cancelling only to rounding.
fn pieces(outline: &Outline) -> Vec<Piece> {
    let mut out = Vec::new();
    for segment in outline.segments() {
        let [p0, p1, p2] = match segment {
            Segment::Line { from, to } => {
                let (a, b) = (wide(from), wide(to));
                [a, [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])], b]
            }
            Segment::Quad { from, control, to } => [wide(from), wide(control), wide(to)],
        };
        if distance(p0, p2) < ZERO_LENGTH {
            continue;
        }
        out.extend(
            MonotoneQuad::split(p0, p1, p2)
                .into_iter()
                .filter_map(Piece::oriented),
        );
    }
    out
}

// ═══════════════════════════════════════════════════════════════════════════
// The kernel
// ═══════════════════════════════════════════════════════════════════════════

fn constant(v: f32) -> Kernel {
    Kernel::constant(v)
}

/// **The signed area of the pixel under ink**: the winding number
/// integrated over the pixel, summed from every piece's term.
///
/// Additive, and zero outside every closed contour — the two facts that
/// make a run of glyphs one glyph ([`Glyph::over`]).
#[derive(Clone)]
struct SignedArea(Kernel);

impl SignedArea {
    /// No ink, and the identity of the sum.
    fn zero() -> Self {
        Self(constant(0.0))
    }

    fn sum(terms: &[SignedArea]) -> Self {
        let raw: Vec<Kernel> = terms.iter().map(|a| a.0.clone()).collect();
        Self(Kernel::fold(Monoid::SUM, &raw))
    }

    /// This area where `inside` holds, [`SignedArea::zero`] elsewhere —
    /// exact wherever the area is already zero outside `inside`.
    fn masked(&self, inside: &Kernel) -> Self {
        Self(inside.select(&self.0, &constant(0.0)))
    }
}

/// Coverage, from the signed area: `min(|F|, 1)`, with the ends snapped
/// ([`COVERAGE_SNAP`]).
fn coverage(area: &SignedArea) -> Kernel {
    let (zero, one) = (constant(0.0), constant(1.0));
    let c = area.0.abs().min(&one);
    let full = c.ge(&constant(1.0 - COVERAGE_SNAP));
    let empty = c.le(&constant(COVERAGE_SNAP));
    full.select(&one, &empty.select(&zero, &c))
}

// ─────────────────────────────────────────────────────────────────────────
// The piece row: one layout, one body
// ─────────────────────────────────────────────────────────────────────────
//
// Every term below reads a piece's numbers **by column**, from a bound table
// at the fold's binder, so a glyph is one fold with a fixed body rather than
// one arena fragment per piece: [`glyph`]'s `Kernel::over` is the signed
// area, and row `i` of the table is piece `i` — or, past the outline's own
// piece count and up to its bucketed trip count ([`bucketed_trip_count`]), a
// [`padding_row`]. The [`Coeff`] indirection is what keeps the read one
// definition regardless of which kind of row is asking.
//
// **A row always evaluates**, and every distinction a piece could carry is a
// number: a line is the arc whose two steps are equal (bend `0`), and a
// padding row is the arc of no rows at all.

/// `x₀`, the oriented arc's start.
const COL_X0: usize = 0;
/// `p₁.x − p₀.x`, the first step in `x`: never negative.
const COL_E0X: usize = 1;
/// `p₂.x − p₁.x`, the second step in `x`: never negative.
const COL_E1X: usize = 2;
/// `S·y₀`, the oriented arc's start, reflected.
const COL_Y0: usize = 3;
/// `S·(p₁.y − p₀.y)`, the first step in `y`: never negative.
const COL_E0Y: usize = 4;
/// `S·(p₂.y − p₁.y)`, the second step in `y`: never negative.
const COL_E1Y: usize = 5;
/// `σ`, the direction the outline runs along the piece: `±1`.
const COL_SIGMA: usize = 6;
/// `S`, the reflection the arc is read under: `±1`.
const COL_S: usize = 7;
/// The lowest screen `Y` a pixel reaching the piece can be sampled at: the
/// piece's band less half a pixel, rounded down.
const COL_ROWS_LO: usize = 8;
/// The highest: the band plus half a pixel, rounded up.
const COL_ROWS_HI: usize = 9;
/// Columns in one piece's row: `x₀, e₀ₓ, e₁ₓ, S·y₀, e₀ᵧ, e₁ᵧ, σ, S,
/// rows_lo, rows_hi`.
const PIECE_ROW_COLS: usize = 10;

/// One piece's row, column `k`, as a [`Kernel`] — a bound-table read at the
/// fold's binder. See the module section above.
type Coeff<'a> = &'a dyn Fn(usize) -> Kernel;

/// Piece `i`'s row, column by column: the one [`Coeff`] the fold reads
/// through, `i` being the fold's own reduce binder.
fn row_at<'a>(table: &'a Kernel, i: &'a Kernel) -> impl Fn(usize) -> Kernel + 'a {
    move |k| table.at(&Kernel::constant(k as f32), i)
}

/// A term built node by node in one arena, from kernels grafted into it,
/// carrying the tabulations those kernels carry — so the finished kernel
/// binds what they bound, as a composition of kernel values would.
///
/// A piece's term is built here and not through [`Kernel`]'s methods
/// because each of those copies its operands' whole graph, so a term built
/// a method at a time costs the square of its size. The closed form is
/// about 150 nodes: built by methods, the printable ASCII glyphs at 16 px
/// took 260–320 ms to build (release, before compiling any), and 50–55 ms
/// built here — against 70–110 ms for the integral they replace.
#[derive(Default)]
struct Grafted {
    arena: ExprArena,
    tables: BTreeMap<BufferIdentity, Arc<[f32]>>,
}

impl Grafted {
    /// `kernel` spliced into the arena.
    fn graft(&mut self, kernel: &Kernel) -> ExprId {
        for (id, data) in kernel.buffer_data() {
            self.tables.insert(id, Arc::clone(data));
        }
        let (from, root) = kernel.parts();
        self.arena.splice(from, root)
    }

    /// The kernel rooted at `root`.
    fn finish(self, root: ExprId) -> Kernel {
        let kernel = Kernel::from_parts(self.arena, root);
        self.tables.into_iter().fold(kernel, |kernel, (id, data)| {
            kernel.with_buffer_data(id, data)
        })
    }
}

/// One coordinate of a piece's arc, measured from its start:
/// `q(t) = t·(2·step + bend·t)` for `t ∈ [0, 1]`, with `step = p₁ − p₀` and
/// `bend = (p₂ − p₁) − step` along that axis.
///
/// **Certified to rise.** Both control-polygon steps are floored at `0`
/// ([`Rise::certified`]), so `q′(t) = 2·((1 − t)·step + t·(step + bend))`
/// is never negative, whatever number a row holds. The closed form's case
/// analysis stands on that — the arc crosses each row and each column of
/// the pixel at most once — and the host's orientation
/// ([`Piece::oriented`]) is what makes the certified arc the glyph's arc:
/// for a real row the floor moves nothing.
#[derive(Clone, Copy)]
struct Rise {
    step: ExprId,
    bend: ExprId,
}

impl Rise {
    /// The rise whose control-polygon steps are `first` and `second`, each
    /// floored at `0`.
    fn certified(arena: &mut ExprArena, [first, second]: [ExprId; 2]) -> Self {
        let zero = arena.push_const(0.0);
        let step = arena.push_binary(OpKind::Max, first, zero);
        let second = arena.push_binary(OpKind::Max, second, zero);
        let bend = arena.push_binary(OpKind::Sub, second, step);
        Self { step, bend }
    }

    /// `2·step + bend·s`: the factor [`Rise::at`] and [`Rise::climb`] share.
    fn slope_through(self, arena: &mut ExprArena, s: ExprId) -> ExprId {
        let twice = arena.push_binary(OpKind::Add, self.step, self.step);
        let bent = arena.push_binary(OpKind::Mul, self.bend, s);
        arena.push_binary(OpKind::Add, twice, bent)
    }

    /// `q(t)`.
    fn at(self, arena: &mut ExprArena, t: ExprId) -> ExprId {
        let slope = self.slope_through(arena, t);
        arena.push_binary(OpKind::Mul, t, slope)
    }

    /// `Δ(s, t) = q(t) − q(s) = (t − s)·(2·step + bend·(s + t))`: how far
    /// the rise climbs from `s` to `t`, as one product, so an empty span is
    /// exactly `0`.
    fn climb(self, arena: &mut ExprArena, [s, t]: [ExprId; 2]) -> ExprId {
        let width = arena.push_binary(OpKind::Sub, t, s);
        let span = arena.push_binary(OpKind::Add, s, t);
        let slope = self.slope_through(arena, span);
        arena.push_binary(OpKind::Mul, width, slope)
    }

    /// `clamp(τ(δ), lo, hi)`: where the rise reaches `δ`, held to
    /// `[lo, hi]` — `τ` being [`integral::monotone_root`], the one
    /// definition.
    fn reaches(self, arena: &mut ExprArena, delta: ExprId, bounds: [ExprId; 2]) -> ExprId {
        let rise = integral::Rise {
            step: self.step,
            bend: self.bend,
        };
        let floor =
            RootFloor::new(ROOT_FLOOR).expect("ROOT_FLOOR is the largest floor RootFloor admits");
        let root = integral::monotone_root(arena, delta, rise, floor);
        library::clamp(arena, root, bounds)
    }
}

/// One piece's arc as the kernel reads it: from `(x₀, y₀)`, rising in both
/// coordinates.
struct RisingArc {
    x0: ExprId,
    y0: ExprId,
    x: Rise,
    y: Rise,
}

impl RisingArc {
    /// **The area of the pixel about `(x, y)` left of the arc, within the
    /// arc's band of rows**, in closed form.
    ///
    /// ## Derivation
    ///
    /// The arc is `x(t) = x₀ + q_x(t)`, `y(t) = y₀ + q_y(t)` for
    /// `t ∈ [0, 1]`, both rising ([`Rise`]). The pixel is
    /// `[L, L + 1) × [y − ½, y + ½)`, `L = x − ½`. Each of its rows the arc
    /// crosses is ink from `L` to the arc, `clamp(x_arc − L, 0, 1)` of the
    /// row, and a row the arc does not cross holds none of this piece's
    /// ink. So the area is an integral over the rows the pixel and the band
    /// share, and with `v = y(t)`, `dv = y′(t)·dt` it runs along the arc:
    ///
    /// ```text
    /// A = ∫_{t₀}^{t₁} clamp(x(t) − L, 0, 1)·y′(t) dt
    ///
    /// t₀ = clamp(τ_y(y − ½ − y₀), 0, 1)      t₁ = clamp(τ_y(y + ½ − y₀), 0, 1)
    /// ```
    ///
    /// `τ` is a rise's inverse ([`integral::monotone_root`]), so `t₀` and
    /// `t₁` are where the arc enters and leaves the pixel's rows, held to
    /// the arc. `x` rises, so the clamp is `0` until the arc reaches the
    /// pixel's left edge, `1` once it has passed the right one, and
    /// `x(t) − L` between — three pieces:
    ///
    /// ```text
    /// t_L = clamp(τ_x(L − x₀), t₀, t₁)      t_R = clamp(τ_x(L + 1 − x₀), t₀, t₁)
    ///
    /// A = ∫_{t_R}^{t₁} y′(t) dt + ∫_{t_L}^{t_R} (x(t) − L)·y′(t) dt
    ///   = Δ_y(t_R, t₁)          + F(t_R) − F(t_L)
    /// ```
    ///
    /// `Δ_y(s, t) = q_y(t) − q_y(s)` is the rows the arc climbs from `s` to
    /// `t`: the first term is the rows where the arc is right of the pixel,
    /// wholly inked. `F` is an antiderivative of `(x(t) − L)·y′(t)`, a
    /// quartic, and `F(t_R) − F(t_L)` is not evaluated as two values and
    /// their difference — each is the size of the arc's whole moment about
    /// `L`, which its extent sets, while the difference is at most a
    /// pixel — but as what it is, the area left of the sub-arc from `t_L`
    /// to `t_R` over the rows it climbs: its chord's trapezoid, plus the
    /// region between the sub-arc and its chord, which for a quadratic
    /// depends on nothing but the span `w = t_R − t_L`, its second
    /// derivative being the same everywhere:
    ///
    /// ```text
    /// F(t_R) − F(t_L) = ½·(x̂(t_L) + x̂(t_R))·Δ_y(t_L, t_R) + K·w³
    ///
    /// x̂(t) = x(t) − L = q_x(t) − (L − x₀)      K = (β·a − b·α)/3
    /// ```
    ///
    /// with `b`, `a` the `y` rise's step and bend and `β`, `α` the `x`
    /// rise's. Expanded in `w` about `t_L`, the two sides agree in `w`, `w²`
    /// and `w⁴`; in `w³` they differ by `(x′(t_L)·a − α·y′(t_L))/6`, which
    /// is `2·(β·a − b·α)/6` wherever `t_L` is.
    ///
    /// No case needs an arm of its own. An arc left of the pixel has
    /// `t_L = t_R = t₁`, one right of it `t_L = t_R = t₀`, and a band that
    /// misses the pixel's rows `t₀ = t₁`; a horizontal arc (`b = a = 0`)
    /// climbs nothing, and a vertical one (`β = α = 0`) divides by the
    /// root's floor and saturates to `t₀` or `t₁`. Each lands on the
    /// formula through a clamp, and where its area is `0` it is exactly `0`:
    /// every term is then a product with an empty span's `t − s`.
    ///
    /// ## Floating point
    ///
    /// - `x − x₀` and `y − y₀` are taken before the pixel's `±½`, so where
    ///   the pixel is relative to the arc is a difference of the two
    ///   coordinates, exact when they are close, rather than a rounded edge
    ///   less a coordinate.
    /// - The only quotients are the roots', `δ·(1/d)` with `d` floored at a
    ///   positive literal ([`integral::monotone_root`], "Emitted as
    ///   `δ·(1/d)`"): never a `Recip` estimate, whatever the algebra proves
    ///   about `d`. `⅓` and `½` are products by literals.
    /// - The parameters resolve to about `2⁻²⁴`, and a parameter moved by
    ///   `ε` moves a height by at most `ε·y′`: the error grows with the
    ///   arc's extent, and a pixel wholly inside a region sums to `1` only
    ///   to a few ulps ([`COVERAGE_SNAP`]). Measured within
    ///   `2⁻²²·(1 + |X| + |Y| + 2·extent)` per texel of a glyph
    ///   (`tests/glyph_exact_area.rs`, `tests/glyph_area_edge_cases.rs`),
    ///   and within `2⁻²⁰·(1 + extent)` arc by arc (this module's
    ///   `adversarial` tests, which use a quarter of it).
    fn pixel_area(&self, arena: &mut ExprArena, [x, y]: [ExprId; 2]) -> ExprId {
        let zero = arena.push_const(0.0);
        let one = arena.push_const(1.0);
        let half = arena.push_const(PIXEL_HALF);
        let across = arena.push_binary(OpKind::Sub, x, self.x0);
        let up = arena.push_binary(OpKind::Sub, y, self.y0);
        // `L − x₀` and `L + 1 − x₀`: the pixel's left and right edges, and
        // its lowest and highest rows, measured from the arc's start.
        let left = arena.push_binary(OpKind::Sub, across, half);
        let right = arena.push_binary(OpKind::Add, across, half);
        let lowest = arena.push_binary(OpKind::Sub, up, half);
        let highest = arena.push_binary(OpKind::Add, up, half);

        let t0 = self.y.reaches(arena, lowest, [zero, one]);
        let t1 = self.y.reaches(arena, highest, [zero, one]);
        let t_left = self.x.reaches(arena, left, [t0, t1]);
        let t_right = self.x.reaches(arena, right, [t0, t1]);

        let right_of_the_pixel = self.y.climb(arena, [t_right, t1]);

        let x_left = self.x.at(arena, t_left);
        let x_left = arena.push_binary(OpKind::Sub, x_left, left);
        let x_right = self.x.at(arena, t_right);
        let x_right = arena.push_binary(OpKind::Sub, x_right, left);
        let ends = arena.push_binary(OpKind::Add, x_left, x_right);
        let mean = arena.push_binary(OpKind::Mul, half, ends);
        let rise = self.y.climb(arena, [t_left, t_right]);
        let trapezoid = arena.push_binary(OpKind::Mul, mean, rise);

        let x_over_y = arena.push_binary(OpKind::Mul, self.x.step, self.y.bend);
        let y_over_x = arena.push_binary(OpKind::Mul, self.y.step, self.x.bend);
        let twist = arena.push_binary(OpKind::Sub, x_over_y, y_over_x);
        let third = arena.push_const(ONE_THIRD);
        let k = arena.push_binary(OpKind::Mul, third, twist);
        let w = arena.push_binary(OpKind::Sub, t_right, t_left);
        let square = arena.push_binary(OpKind::Mul, w, w);
        let cube = arena.push_binary(OpKind::Mul, square, w);
        let bow = arena.push_binary(OpKind::Mul, k, cube);

        let between = arena.push_binary(OpKind::Add, trapezoid, bow);
        arena.push_binary(OpKind::Add, right_of_the_pixel, between)
    }
}

/// **One piece's term**: `σ·A(X, S·Y)` where the pixel's rows reach the
/// piece, `0` where they do not — `A` the area of the pixel left of the arc
/// within its band ([`RisingArc::pixel_area`]).
///
/// The arc is read reflected, at `S·Y`; the pixel is symmetric, so the
/// pixel about `(X, S·Y)` in the reflected frame is the pixel about
/// `(X, Y)` on the screen.
///
/// The cut to the rows is an identity — outside them the term is exactly
/// `0` — and its mask depends on the row and the piece alone, uniform over
/// a batch: an `If` a guard may lower to a jump over the whole body.
fn piece_term(c: Coeff) -> Kernel {
    let mut term = Grafted::default();
    let row: [ExprId; PIECE_ROW_COLS] = core::array::from_fn(|k| term.graft(&c(k)));
    let x = term.graft(&Kernel::x());
    let y = term.graft(&Kernel::y());
    let arena = &mut term.arena;

    let arc = RisingArc {
        x0: row[COL_X0],
        y0: row[COL_Y0],
        x: Rise::certified(arena, [row[COL_E0X], row[COL_E1X]]),
        y: Rise::certified(arena, [row[COL_E0Y], row[COL_E1Y]]),
    };
    let reflected = arena.push_binary(OpKind::Mul, row[COL_S], y);
    let area = arc.pixel_area(arena, [x, reflected]);
    let signed = arena.push_binary(OpKind::Mul, row[COL_SIGMA], area);

    let above = arena.push_binary(OpKind::Gt, y, row[COL_ROWS_LO]);
    let below = arena.push_binary(OpKind::Lt, y, row[COL_ROWS_HI]);
    let reaches = arena.push_binary(OpKind::BitAnd, above, below);
    let zero = arena.push_const(0.0);
    let cut = arena.push_ternary(OpKind::If, reaches, signed, zero);
    term.finish(cut)
}

/// `value` rounded to `f32` toward `−∞`.
fn f32_down(value: f64) -> f32 {
    let rounded = value as f32;
    match f64::from(rounded) > value {
        true => rounded.next_down(),
        false => rounded,
    }
}

/// `value` rounded to `f32` toward `+∞`.
fn f32_up(value: f64) -> f32 {
    let rounded = value as f32;
    match f64::from(rounded) < value {
        true => rounded.next_up(),
        false => rounded,
    }
}

/// One piece's row in the coefficient table (host-side, `f32`) — see the
/// column layout above.
///
/// The band's rows are rounded outward, so the cut in [`piece_term`] can
/// only be wider than the band, never narrower — it stays an identity.
fn piece_row(piece: Piece) -> [f32; PIECE_ROW_COLS] {
    let mut row = [0.0f32; PIECE_ROW_COLS];
    row[COL_X0] = piece.start[0] as f32;
    row[COL_E0X] = piece.first[0] as f32;
    row[COL_E1X] = piece.second[0] as f32;
    row[COL_Y0] = piece.start[1] as f32;
    row[COL_E0Y] = piece.first[1] as f32;
    row[COL_E1Y] = piece.second[1] as f32;
    row[COL_SIGMA] = piece.sigma as f32;
    row[COL_S] = piece.reflect as f32;
    row[COL_ROWS_LO] = f32_down(piece.rows[0] - f64::from(PIXEL_HALF));
    row[COL_ROWS_HI] = f32_up(piece.rows[1] + f64::from(PIXEL_HALF));
    debug_assert!(
        row.iter().all(|v| v.is_finite()),
        "a piece row holds a non-finite column: {row:?} from {piece:?}"
    );
    debug_assert!(
        [COL_E0X, COL_E1X, COL_E0Y, COL_E1Y]
            .iter()
            .all(|&step| row[step] >= 0.0),
        "an oriented step is negative, so the kernel's certificate would move \
         the arc: {row:?} from {piece:?}"
    );
    row
}

/// The number of rows [`run`] folds over for a piece count of `pieces`: the
/// count itself, rounded up to a shared bucket.
///
/// The JIT cache keys a compiled program by canonical arena *and shape*
/// (`docs/plans/2026-09-09-glyph-as-a-fold-execution.md` §S3), and a fold's
/// shape is its trip count, so two glyphs otherwise alike but for their
/// piece count compile to two programs. Bucketing collapses the font-wide
/// *set* of trip counts a piece count can land on, so unrelated glyphs whose
/// counts round to the same bucket share one program; [`padding_row`] is
/// what fills the gap between a glyph's own count and its bucket without
/// changing what the glyph draws.
///
/// Powers of two, per
/// `docs/plans/2026-09-09-glyph-as-a-fold-execution.md` §S3's own framing
/// — see that section, and this crate's measurements in
/// `docs/BACKLOG.md`, for the trade against a coarser bucket.
fn bucketed_trip_count(pieces: u32) -> u32 {
    pieces.next_power_of_two()
}

/// A row that folds to the identity of [`glyph`]'s sum at every sample:
/// **all zeros**.
///
/// Its band is the empty interval `(0, 0)`, so [`piece_term`]'s cut —
/// `Y > 0 ∧ Y < 0` — holds nowhere, and the term selects its literal `0.0`
/// outright; not a mask folded against a coefficient, so summing this row
/// changes no bit of a real glyph's area. (Read uncut it would be an arc of
/// no rise at all — every step zero — whose band has measure zero.)
///
/// The one producer of a padding row. [`piece_row`] takes a [`Piece`], and
/// [`Piece::oriented`] never builds a horizontal one, so there is no way to
/// reach this row's shape by constructing a "fake" piece. [`run`] calls this
/// function directly wherever [`bucketed_trip_count`] asks for more rows than
/// the outline has pieces.
fn padding_row() -> [f32; PIECE_ROW_COLS] {
    [0.0f32; PIECE_ROW_COLS]
}

#[cfg(test)]
mod adversarial;

#[cfg(test)]
mod kernel_copy;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fonts::outline::Contour;

    fn outline_of(segments: Vec<Segment>) -> Outline {
        Outline {
            contours: vec![Contour::new(segments).expect("the test's contour closes")],
        }
    }

    /// A quadratic whose ends coincide encloses nothing, and is dropped
    /// before it is split — so a closed loop of one such curve is no pieces
    /// at all, rather than two halves cancelling to rounding.
    #[test]
    fn a_quadratic_whose_ends_coincide_is_dropped() {
        let loop_ = outline_of(vec![Segment::Quad {
            from: [0.0, 0.0],
            control: [5.0, 7.0],
            to: [0.0, 0.0],
        }]);
        assert!(pieces(&loop_).is_empty());
    }

    /// A horizontal piece crosses no row: dropped. A square is its two
    /// vertical sides, each oriented to rise, running opposite ways.
    #[test]
    fn a_square_is_its_two_vertical_sides() {
        let corners = [[1.0f32, 2.0], [5.0, 2.0], [5.0, 9.0], [1.0, 9.0]];
        let square = outline_of(
            (0..4)
                .map(|k| Segment::Line {
                    from: corners[k],
                    to: corners[(k + 1) % 4],
                })
                .collect(),
        );
        let sides = pieces(&square);
        assert_eq!(sides.len(), 2, "{sides:?}");
        let sigmas: Vec<f64> = sides.iter().map(|p| p.sigma).collect();
        assert_eq!(sigmas, vec![1.0, -1.0], "up the right side, down the left");
        for side in sides {
            assert_eq!(side.rows, [2.0, 9.0]);
            assert_eq!(side.first, side.second, "a line's bend is zero");
            assert!(side.first[1] > 0.0 && side.first[0] == 0.0);
        }
    }

    /// Every piece rises in both coordinates once oriented, whatever way the
    /// outline ran — and a hook, which turns back in both, is cut into
    /// three arcs.
    #[test]
    fn a_hook_is_cut_and_every_arc_rises() {
        let hook = outline_of(vec![
            Segment::Quad {
                from: [0.0, 0.0],
                control: [-100.0, 5.0],
                to: [1.0, 0.0],
            },
            Segment::Line {
                from: [1.0, 0.0],
                to: [0.0, 0.0],
            },
        ]);
        let arcs = pieces(&hook);
        assert_eq!(arcs.len(), 3, "the closing line is horizontal: {arcs:?}");
        for arc in &arcs {
            let row = piece_row(*arc);
            for step in [COL_E0X, COL_E1X, COL_E0Y, COL_E1Y] {
                assert!(row[step] >= 0.0, "{row:?}");
            }
        }
    }

    #[test]
    fn bucketed_trip_count_rounds_up_to_a_power_of_two() {
        assert_eq!(bucketed_trip_count(1), 1);
        assert_eq!(bucketed_trip_count(2), 2);
        assert_eq!(bucketed_trip_count(3), 4);
        assert_eq!(bucketed_trip_count(4), 4);
        assert_eq!(bucketed_trip_count(5), 8);
        assert_eq!(bucketed_trip_count(34), 64);
    }

    /// A [`padding_row`] changes no bit of the fold. Baking the signed area
    /// with extra padding rows appended — over a lattice wide enough to
    /// sample both sides of a real piece — must reproduce exactly the same
    /// buffer as baking with the real row alone, at every padding count a
    /// glyph's own pieces ever land between (`bucketed_trip_count` never pads
    /// past the next power of two, so 1–7 extra rows covers every case up to
    /// an 8-piece bucket).
    #[test]
    fn a_padding_row_is_an_exact_identity_of_the_fold() {
        let arc = MonotoneQuad::split([0.0, 0.0], [2.0, 2.0], [4.0, 4.0])[0];
        let piece = Piece::oriented(arc).expect("a diagonal is not horizontal");
        let real = piece_row(piece);

        let bake = |rows: &[[f32; PIECE_ROW_COLS]]| -> Vec<f32> {
            let flat: Vec<f32> = rows.iter().flatten().copied().collect();
            let table = DiscreteManifold::new(flat, PIECE_ROW_COLS, rows.len()).kernel();
            let count = u32::try_from(rows.len()).expect("test row counts fit u32");
            let area = Kernel::sum_over(count, |i| piece_term(&row_at(&table, i)));
            let bound = Manifold::compile(&area, [4, 4]).bind(&[]);
            Lattice::frame(4, 4).collapse(&bound).into_buffer()
        };

        let alone = bake(&[real]);
        assert!(
            alone.iter().any(|&v| v != 0.0),
            "the real row draws nothing"
        );
        for padding in 1..=7 {
            let mut rows = vec![real];
            rows.extend(core::iter::repeat_with(padding_row).take(padding));
            assert_eq!(
                alone,
                bake(&rows),
                "the area changed with {padding} padding row(s) appended"
            );
        }
    }
}
