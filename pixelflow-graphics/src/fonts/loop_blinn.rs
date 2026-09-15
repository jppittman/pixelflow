//! Coverage of an outline as a [`Kernel`]: an exact winding number decides
//! inside from outside, a distance decides how much of the pixel, and
//! Loop–Blinn's implicit is what makes a *curved* edge as cheap to ask
//! about as a straight one.
//!
//! ## A glyph is a circle with a sum where the sign used to be
//!
//! A circle is easy because it arrives as a formula whose sign is the
//! answer: `x² + y² − 1`, negative inside. Sign says inside, magnitude says
//! how far to the edge, and both come from one polynomial.
//!
//! A glyph needs the same two things and cannot get them from one
//! polynomial, because its edge is many pieces rather than one:
//!
//! ```text
//! inside   = (Σ crossings) ≠ 0          // the sign
//! distance = min over boundary pieces   // the magnitude
//! coverage = inside ? ½ + distance : ½ − distance
//! ```
//!
//! A **crossing** is the schoolbook inside test: fire a ray from the sample
//! to `+X`, count the edges it meets on the way out — `+1` for an edge
//! running upward, `−1` downward. Nonzero means inside. That sum is the
//! winding number, and the non-zero rule is what TrueType specifies.
//!
//! For a straight edge a crossing is two comparisons and a multiply-add,
//! and all but the final comparison is a function of `Y` alone, so it
//! hoists into the row prologue. That is the *good* kind of Y-only work.
//! What made the scanline formulation slow was hoisting a **root solve** —
//! discriminant, square root, two roots, per segment per row — and there is
//! none here, because every crossing is against a straight chord.
//!
//! ## What Loop–Blinn contributes, and it is exactly one thing
//!
//! Replace each curve by its straight chord and the crossing count above is
//! wrong by precisely one region: the crescent between the chord and the
//! real curve. Loop and Blinn give the membership test for that crescent
//! without ever solving for where the curve is.
//!
//! The trick is that all parabolas are the same shape, the way all circles
//! are. Skew the plane — an affine map, so straight lines stay straight —
//! until your particular arc lands on the one parabola `v = u²`. Then
//! "which side of the curve" is the sign of `u² − v`, where `u` and `v` are
//! each just `a·X + b·Y + c`. A formula whose sign is the answer, exactly
//! like the circle. Root-finding becomes arithmetic, and the `'8'` waist
//! defect — two segments disagreeing about their own discriminant within an
//! ulp at a tangency — is not expressible.
//!
//! The skew is a **contramap**, six numbers per curve computed on the host.
//! It only lands *your arc* on the parabola; keep going and you are on the
//! rest of an infinite parabola that is not your curve, where `u² − v` is
//! not slow but **wrong**. So the test is fenced — and the fence is one
//! half-plane, the chord's: `crescent = {v ≥ u²} ∩ {v ≤ u}`, which is
//! already bounded, because `u² ≤ v ≤ u` holds nowhere but `u ∈ [0, 1]`.
//! A GPU spends a whole rasterized triangle on that fence; two comparisons
//! do it here.
//!
//! ## Only a boundary softens a pixel
//!
//! `distance` is a minimum over the pieces — but over the pieces that are
//! actually *boundaries*, not over every piece the font draws. A stroke
//! built from two overlapping contours, or two overlapping components of a
//! compound glyph, puts drawn edges inside the ink, where coverage should
//! stay saturated. Minimising over those too dims a line straight through
//! solid fill: measured at exactly ½ along the shared edges of two
//! overlapping squares, a seam one pixel wide.
//!
//! Whether a piece is a boundary is a question the winding already answers.
//! With `w₋` the winding minus that piece's own contribution, its two sides
//! are `w₋` and `w₋ + dir` — they differ by one, so never both zero — and
//! the piece separates ink from no-ink exactly when one of them is zero.
//!
//! That reads the *crossing* term, which is 0 wherever the sample's row lies
//! outside the chord's own Y band, so for a horizontal or near-horizontal
//! chord the pair it compares is not the piece's two sides at all. It is
//! therefore asked only of the pieces another contour's ink may cover, where
//! it is the answer; a piece nothing covers always separates and says so
//! with a mask ORed into the test ([`COL_ALWAYS_BOUNDARY`]).
//!
//! ## Two folds over one table
//!
//! Both halves are `Kernel::over` a coefficient table whose row `i` is piece
//! `i`: the winding is a `sum_over`, the distance a `min_over`, and each
//! reads its numbers by column at its own reduce binder. So the arena a
//! glyph builds is two bodies and a few hundred nodes rather than one
//! fragment per piece, whatever the outline's size, and every per-piece
//! distinction is a *number* — a sliver sign of 0 for a line, a direction of
//! 0 for a horizontal chord, a boundary threshold nothing can reach for a
//! piece no other contour covers. Unrolling is the compiler's
//! (`passes::expand_reduce`), and the min's body names the sum rather than
//! copying it, so the sum is unrolled once.
//!
//! ## The winding is exact; only the distance is soft
//!
//! Every winding term is a hard mask selecting a signed constant, so the sum
//! is an integer. Keeping that separate from the ramp is the point: an
//! earlier draft made a crossing's *existence* and its *coverage* one
//! number, so a comparison landing on the wrong side of an edge moved
//! coverage by half a unit instead of by a rounding. That coupling is the
//! `'8'` defect's whole family.
//!
//! Distances are in **pixels of the lattice being collapsed**, not of the
//! frame the outline lives in: each is divided by the gradient magnitude of
//! its own edge function, with `DX`/`DY` resolved symbolically at bake, so
//! the chain rule carries the scale through every enclosing `Kernel::at`.
//! The coefficients are table reads, so that normalisation is a `sqrt` per
//! term at run time rather than the compile-time constant it was when they
//! were literals — the price of one body, and the alternative (a gradient
//! magnitude precomputed on the host) puts the distance back in the
//! outline's units and is wrong under a magnifying `at`.
//!
//! Per piece the distance is the larger of two estimates, because neither
//! is usable alone: the **capsule distance to the chord** (exact for a
//! line; for a curve a lower bound once the curve's deviation is
//! subtracted) and the **implicit's own** `|f|/‖∇f‖` (accurate near the arc
//! and unsound away from it — it underestimates where the curve is sharp,
//! and an underestimate past the ramp is a saturation failure, not a
//! rounding). A lower bound never exceeds the truth, so the larger of the
//! two is the closer. That is also why a segment straying more than half a
//! pixel (`MAX_DEVIATION`) from its chord is halved until it does not.
//!
//! ## What is exactly zero, and where
//!
//! Beyond half a pixel from every boundary the ramp is saturated and the
//! winding is exact, so coverage is exactly 0 or exactly 1 — and outside
//! the outline it is 0. [`glyph`] cuts its kernel to the outline's bounding
//! box dilated by [`RAMP_REACH`] and reports that box as its [`Support`]:
//! exact, whatever the glyph's shape.

use super::outline::{Outline, Point, Segment};
use pixelflow_core::{BoundManifold, DiscreteManifold, Kernel, Lattice, Manifold, Monoid};
use pixelflow_ir::OpKind;

/// How far coverage can reach past the outline, in the frame the kernel is
/// built in. A ramp is one unit wide and centred on its edge, so half a unit
/// is the analytic reach; a whole unit keeps the bound a whole pixel when
/// that frame is the screen, at no cost — the mask sits where every ramp is
/// already saturated.
pub const RAMP_REACH: f32 = 1.0;

/// Gradient floor for every ramp, so a degenerate edge divides by something.
const MIN_GRADIENT: f32 = 1e-3;

/// A chord shorter than this crosses nothing and is dropped.
const ZERO_LENGTH: f64 = 1e-6;

/// How far a segment may stray from its chord, in the kernel's own units,
/// before it is split in two.
///
/// It is an **accuracy** bound on the antialiasing ramp and nothing else.
/// The ramp's distance is the implicit's, taken against a lower bound built
/// from the chord — and that bound is the chord's own distance less this
/// deviation, so a fatter curve makes the bound looser, and past half a
/// pixel it stops bounding anything useful (measured: a ring whose control
/// points sat at its bounding square's corners, deviation 5 px, softened
/// every sample within 5 px of a chord to exactly ½). Splitting is exact —
/// de Casteljau halves a quadratic into two quadratics — so the *winding*
/// never sees this constant, and no split is ever needed for correctness.
const MAX_DEVIATION: f64 = 0.5;

/// Splits a segment will take before it is accepted however fat it is. Each
/// halving quarters the deviation, so six cover a curve straying 2000× the
/// bound; past that the ramp is simply the implicit's, unbounded below.
const MAX_SPLITS: u32 = 6;

/// A curve straying less than this from its chord is drawn as its chord.
/// Dropping a sliver moves coverage by at most the sliver's own width, so
/// this is an order of magnitude below one 8-bit step — and fonts are full
/// of quadratics that are straight to within it, each of which would
/// otherwise carry a sliver and an implicit for nothing.
const FLAT_ENOUGH: f64 = 1.0 / 256.0;

// ═══════════════════════════════════════════════════════════════════════════
// The glyph, and the box outside which it is exactly zero
// ═══════════════════════════════════════════════════════════════════════════

/// A coverage kernel together with the box outside which it is exactly
/// zero. They travel together because they are derived together — a support
/// restated separately from the composition it describes is a future
/// divergence.
/// **The two folds, not the coverage.** Coverage is not additive — summing
/// two glyphs' coverages reaches 2 where their ink overlaps, and 2 is not a
/// coverage — so a glyph that could only offer its finished coverage could
/// not compose, and `text()` merged every character's contours into one
/// outline to get around it. Winding *is* additive and distance combines
/// under `min`, so keeping the two halves apart until the last step is what
/// makes a run a glyph (docs/plans/2026-09-09-a-run-is-a-glyph.md).
///
/// Both read one piece coefficient table at their own reduce binder
/// ([`glyph`]); the table's data travels with the kernels themselves
/// (`Kernel::with_buffer_data`), so there is no separate binding a caller
/// must keep paired with them.
#[derive(Clone)]
pub struct Glyph {
    /// Exact, integral, and zero outside every contour — which is what lets
    /// [`Self::over`] mask it to a box and lose nothing.
    winding: Winding,
    /// In lattice pixels, and at least [`RAMP_REACH`] outside [`Self::support`].
    distance: Distance,
    /// Where it can be nonzero.
    pub support: Support,
}

impl Glyph {
    /// The identity of [`Self::over`]: no ink anywhere. `text("")` is this.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            winding: Winding::zero(),
            distance: Distance::unreachable(),
            support: Support::EMPTY,
        }
    }

    /// **The one exit.** Coverage, cut to the support so the box's claim
    /// holds under any later coordinate warp rather than only where the
    /// ramps happen to saturate.
    ///
    /// A method rather than a field because it is derived: two glyphs
    /// compose through [`Self::over`] on the folds, and a stored coverage
    /// would be a second representation to keep in step.
    #[must_use]
    pub fn kernel(&self) -> Kernel {
        self.support
            .contains()
            .select(&coverage(&self.winding, &self.distance), &constant(0.0))
    }

    /// **Glyphs form a monoid, and this is its operation.** Winding sums,
    /// distance takes the minimum, support unions — each componentwise, so
    /// the whole is associative and commutative and a run's coverage does
    /// not depend on the order its characters were laid out in.
    ///
    /// Each contributor is masked to its own support first, and that is
    /// *exact*, not an approximation, for two reasons:
    ///
    /// - a closed contour's winding is **0** at every exterior point, so
    ///   masking a glyph's winding to its box discards only zeros
    ///   (`Contour::new` refusing an open contour is what licenses this);
    /// - [`Support::around`] dilates by exactly [`RAMP_REACH`], and a piece
    ///   further away than that contributes nothing to the ramp, so masking
    ///   the distance to `RAMP_REACH` outside the box discards only losers
    ///   of the `min`.
    ///
    /// The mask is also the *binning*: structurally every character is still
    /// in the graph — `Select` is dispatch and both arms stay live — but a
    /// SIMD batch is adjacent pixels almost always inside one character's
    /// box, so the emitter's guard can skip the rest. That is a
    /// data-dependent win, and about as coherent as such a mask ever gets.
    #[must_use]
    pub fn over(glyphs: &[Glyph]) -> Self {
        let mut windings = Vec::with_capacity(glyphs.len());
        // Seeded with the ceiling, so the `min` is never empty and a run
        // with no ink is `RAMP_REACH` away from a boundary rather than an
        // infinity the table's H1 rule would rather not see.
        let mut distances = vec![Distance::unreachable()];
        let mut support = Support::EMPTY;
        for g in glyphs {
            let inside = g.support.contains();
            windings.push(g.winding.masked(&inside));
            distances.push(g.distance.masked(&inside));
            support = support.union(g.support);
        }
        Self {
            winding: Winding::sum(&windings),
            distance: Distance::min_of(&distances),
            support,
        }
    }

    /// `kernel` — [`Self::kernel`] itself, or a `Kernel::at` contramap of
    /// it (a placement, a pixel-center shift) — compiled at `extent`. The
    /// piece table both folds read travels with `kernel` itself (see
    /// [`Self`]'s docs), so there is nothing further to bind here — a bare
    /// `Lattice::bake` still refuses `kernel` because it *declares* a
    /// buffer, so this goes through `Manifold::compile`/`bind` directly,
    /// same as before, just with an empty binding list.
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
/// holds under any coordinate warp and not only where the ramps' own
/// saturation makes it true. Coordinates are `[x0, y0, x1, y1]`; a box with
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
    fn contains(self) -> Kernel {
        let [x0, y0, x1, y1] = self.0;
        Kernel::x()
            .ge(&constant(x0))
            .and(&Kernel::x().le(&constant(x1)))
            .and(&Kernel::y().ge(&constant(y0)))
            .and(&Kernel::y().le(&constant(y1)))
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
/// Every chord and every sliver contributes at every sample, so the cost is
/// linear in the outline's segment count. The kernel is cut to its support
/// by a mask whose false arm is the literal 0, so it composes into a larger
/// scene as a guardable arm.
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
/// giving each outline's folds an offset costs one add per fold and binds
/// one slot however long the run is.
///
/// [`glyph`] is this at one outline.
#[must_use]
pub fn run(outlines: &[Outline]) -> Glyph {
    // Host side: every outline's rows appended into one table, and the row
    // range plus bounding box each will fold over.
    let mut rows: Vec<f32> = Vec::new();
    let mut spans: Vec<(u32, u32, [f32; 4])> = Vec::new();
    for outline in outlines {
        let pieces = Pieces::of(outline);
        let (Some(bounds), false) = (outline.bounds(), pieces.is_empty()) else {
            continue;
        };
        let start = u32::try_from(rows.len() / PIECE_ROW_COLS)
            .expect("a run has far fewer pieces than u32::MAX");
        let count = u32::try_from(pieces.pieces.len())
            .expect("an outline has far fewer pieces than u32::MAX");
        rows.extend(pieces.pieces.iter().flat_map(|p| piece_row(*p)));
        spans.push((start, count, bounds));
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
            let offset = constant(start as f32);
            let winding = Winding(Kernel::sum_over(count, |i| {
                let row = i.add(&offset);
                let c = row_at(&table, &row);
                crossing_term(&c).add(&sliver_term(&c)).0
            }));
            // By name, not by value: the boundary test reads the whole
            // winding once per piece, and composition splices, so the sum
            // itself would put a copy of its `Reduce` node in the min's body
            // — which `expand_reduce` then unrolls per piece, quadratic
            // (measured: 16k, 37k, 58k nodes per piece at 40, 73, 132
            // pieces). A name is one node however often it is used, and
            // expansion shares its referent (`passes::expand_refs`), so the
            // sum is unrolled once and both folds read it.
            let winding = Winding(winding.0.by_ref());
            // `RAMP_REACH` as the fold's ceiling rather than a padding row:
            // it is the min's identity, not a piece.
            let distance = Distance(Kernel::min_over(count, |i| {
                let row = i.add(&offset);
                let c = row_at(&table, &row);
                boundary_distance(&c, &winding).0
            }))
            .min(&Distance::unreachable());
            Glyph {
                winding,
                distance,
                support: Support::around(bounds),
            }
        })
        .collect();
    Glyph::over(&placed)
}

// ═══════════════════════════════════════════════════════════════════════════
// Host geometry: the outline as chords and slivers, in f64
// ═══════════════════════════════════════════════════════════════════════════

type P = [f64; 2];

fn wide([x, y]: Point) -> P {
    [f64::from(x), f64::from(y)]
}

/// The signed area of `(o, a, b)`, doubled: positive where `b` is
/// counter-clockwise from `a` around `o` (in a y-up frame; the sign is only
/// ever compared with itself).
fn cross(o: P, a: P, b: P) -> f64 {
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
}

/// Distance from `p` to the closed segment `chord`.
fn point_to_segment(p: P, chord: Chord) -> f64 {
    let (ex, ey) = (chord.b[0] - chord.a[0], chord.b[1] - chord.a[1]);
    let len2 = ex * ex + ey * ey;
    let t = if len2 == 0.0 {
        0.0
    } else {
        (((p[0] - chord.a[0]) * ex + (p[1] - chord.a[1]) * ey) / len2).clamp(0.0, 1.0)
    };
    let (qx, qy) = (chord.a[0] + t * ex, chord.a[1] + t * ey);
    ((p[0] - qx).powi(2) + (p[1] - qy).powi(2)).sqrt()
}

/// A straight chord `a → b`, never of zero length.
#[derive(Clone, Copy, Debug)]
struct Chord {
    a: P,
    b: P,
}

impl Chord {
    fn length(self) -> f64 {
        ((self.b[0] - self.a[0]).powi(2) + (self.b[1] - self.a[1]).powi(2)).sqrt()
    }

    /// `[min, max]` of the chord's Y.
    fn y_range(self) -> [f64; 2] {
        [self.a[1].min(self.b[1]), self.a[1].max(self.b[1])]
    }

    /// `+1` where the chord runs in the direction of increasing Y, `−1`
    /// where it runs the other way, and **`0` where it is horizontal** —
    /// the sign a ray crossing it picks up, and a horizontal chord is
    /// crossed by none.
    ///
    /// That third case is not a special case, it is the answer: a
    /// horizontal edge carries no winding jump at all. The jump across the
    /// top of a rectangle is carried by the *vertical* edges, whose
    /// half-open Y bands end there. Returning `−1` for it — which this did,
    /// because `b.y > a.y` is false when they are equal — fabricates a jump
    /// of one wherever a caller asks a horizontal chord which side of it
    /// you are on, and every caller had to remember to ask `is_horizontal`
    /// first. Two did; one did not.
    fn direction(self) -> f64 {
        match self.is_horizontal() {
            true => 0.0,
            false => match self.b[1] > self.a[1] {
                true => 1.0,
                false => -1.0,
            },
        }
    }

    /// Whether the chord is horizontal, and so is crossed by no horizontal
    /// ray: it spans no row, and contributes nothing to any winding.
    fn is_horizontal(self) -> bool {
        (self.b[1] - self.a[1]).abs() < ZERO_LENGTH
    }
}

/// The sliver a quadratic bulges past its chord: its control triangle, and
/// the sign the sliver's winding carries.
#[derive(Clone, Copy, Debug)]
struct Bulge {
    p0: P,
    p1: P,
    p2: P,
    /// `sign(cross(p0, p1, p2))`: the control triangle's orientation.
    sign: f64,
}

/// One outline segment, prepared: its chord, and its sliver if it curves
/// enough to have one.
#[derive(Clone, Copy, Debug)]
struct Piece {
    chord: Chord,
    bulge: Option<Bulge>,
    /// Whether this piece lies under *other* contours' ink, so that it may
    /// be inside the filled region rather than bound it — the one thing
    /// [`boundary_distance`]'s test has to be *asked* about, and it reaches
    /// the kernel as column [`COL_ALWAYS_BOUNDARY`] rather than as a branch.
    ///
    /// It is a correctness gate, not a cost one. The two-sides formula reads
    /// the crossing term, which is 0 wherever the sample's row lies outside
    /// the chord's own Y band, so for a horizontal or near-horizontal chord
    /// the pair it compares is not that piece's two sides — asking it of a
    /// piece that always separates would drop that piece's ramp. A piece
    /// nothing else covers always separates, so it is not asked.
    ///
    /// Measured on this crate's own font: of 189 printable and Latin-1
    /// glyphs, exactly two need it — `Ç` and `ç`, where the cedilla
    /// overlaps the C. Bounding boxes are far too coarse a proxy (adjacent
    /// glyphs in a string overlap constantly without their ink ever
    /// touching), so this asks the winding directly.
    may_be_interior: bool,
}

impl Piece {
    /// A straight piece, or nothing when its ends coincide: a zero-length
    /// chord crosses nothing.
    fn line(a: P, b: P) -> Option<Self> {
        let chord = Chord { a, b };
        (chord.length() >= ZERO_LENGTH).then_some(Self {
            chord,
            bulge: None,
            may_be_interior: false,
        })
    }

    /// A point on the segment, for asking what else covers it.
    fn midpoint(self) -> P {
        match self.bulge {
            // B(½) = (p0 + 2p1 + p2) / 4.
            Some(b) => [
                (b.p0[0] + 2.0 * b.p1[0] + b.p2[0]) / 4.0,
                (b.p0[1] + 2.0 * b.p1[1] + b.p2[1]) / 4.0,
            ],
            None => [
                (self.chord.a[0] + self.chord.b[0]) / 2.0,
                (self.chord.a[1] + self.chord.b[1]) / 2.0,
            ],
        }
    }

    /// The quadratic `p0 → p1 → p2` as pieces, each straying no more than
    /// [`MAX_DEVIATION`] from its own chord, appended to `out`. A curve
    /// flatter than that is one piece; a fatter one is halved at its
    /// midpoint (de Casteljau, exact) and each half asked again. A piece
    /// whose curve is flat enough to be a line is one — its sliver would be
    /// empty and its implicit degenerate.
    fn quad(p0: P, p1: P, p2: P, splits: u32, out: &mut Vec<Self>) {
        let chord = Chord { a: p0, b: p2 };
        if chord.length() < ZERO_LENGTH {
            // A closed loop: the curve leaves and returns, enclosing no
            // area on either side of a chord that does not exist.
            return;
        }
        let area2 = cross(p0, p1, p2);
        // The same measure [`Piece::deviation`] reports, because both
        // decisions below are about the quantity the ramp's bound
        // subtracts. Measuring against the chord's infinite *line* here —
        // `|area2| / length / 2`, which this used to do — is the very trap
        // that method's doc names, and it fails in both directions on the
        // hook it cites: `(0,0) → (−100, 0.1) → (1,0)` reads as 0.05 and is
        // never split though the curve strays 49.75, and flattening the
        // control point to `(−100, 0.005)` reads as 0.0025, under
        // `FLAT_ENOUGH`, so the whole hook is discarded as a straight line.
        // That one changes the *winding*, which is never approximated.
        let deviation = point_to_segment(p1, chord);
        if deviation < FLAT_ENOUGH {
            out.extend(Self::line(p0, p2));
            return;
        }
        if splits > 0 && deviation > MAX_DEVIATION {
            let mid = |a: P, b: P| [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0];
            let (l, r) = (mid(p0, p1), mid(p1, p2));
            let m = mid(l, r);
            Self::quad(p0, l, m, splits - 1, out);
            Self::quad(m, r, p2, splits - 1, out);
            return;
        }
        out.push(Self {
            chord,
            bulge: Some(Bulge {
                p0,
                p1,
                p2,
                sign: area2.signum(),
            }),
            may_be_interior: false,
        });
    }

    /// How far the segment strays from its chord *segment*: 0 for a line,
    /// and for a curve the distance from its control point to the chord.
    ///
    /// The curve lies in the convex hull of its three control points, and
    /// distance to a convex set is itself convex, so its maximum over the
    /// hull is attained at a vertex. Two of those vertices are the chord's
    /// own ends, at distance 0. So the control point's distance bounds the
    /// whole curve's, which is what the ramp's lower bound needs.
    ///
    /// Not the perpendicular distance to the chord's infinite *line*: a
    /// control point can project far outside the chord's span — the curve
    /// `(0,0) → (−100, 0.1) → (1,0)` deviates 0.05 from the line and 50
    /// from the segment — and a bound that misses that is not a bound.
    fn deviation(self) -> f64 {
        self.bulge
            .map_or(0.0, |b| point_to_segment(b.p1, self.chord))
    }
}

/// The winding of a contour's chords at `p`, by a horizontal ray. Chords
/// rather than curves: this decides only whether a piece needs the boundary
/// test, and a curve's crescent is at most [`MAX_DEVIATION`] wide, so a
/// point that close to the answer is already inside the ramp.
fn chord_winding(pieces: &[Piece], p: P) -> i32 {
    let mut w = 0;
    for piece in pieces {
        let (a, b) = (piece.chord.a, piece.chord.b);
        if (a[1] <= p[1]) == (b[1] <= p[1]) {
            continue;
        }
        let t = (p[1] - a[1]) / (b[1] - a[1]);
        if a[0] + t * (b[0] - a[0]) > p[0] {
            w += if b[1] > a[1] { 1 } else { -1 };
        }
    }
    w
}

/// An outline prepared for the kernel.
struct Pieces {
    pieces: Vec<Piece>,
}

impl Pieces {
    fn of(outline: &Outline) -> Self {
        // Per contour, so a piece can be asked whether any *other* contour
        // reaches it. Contours are what overlap in a font — a stroke drawn
        // as two shapes, a compound glyph's components — and an edge under
        // another contour's ink is not a boundary.
        let mut by_contour: Vec<Vec<Piece>> = Vec::with_capacity(outline.contours.len());
        for contour in &outline.contours {
            let mut pieces = Vec::new();
            for segment in contour.segments() {
                match *segment {
                    Segment::Line { from, to } => {
                        pieces.extend(Piece::line(wide(from), wide(to)));
                    }
                    Segment::Quad { from, control, to } => {
                        Piece::quad(wide(from), wide(control), wide(to), MAX_SPLITS, &mut pieces);
                    }
                }
            }
            by_contour.push(pieces);
        }
        // A piece is possibly-interior exactly when the *other* contours
        // already cover it: their winding at its midpoint is nonzero. Asked
        // directly rather than through bounding boxes, because boxes touch
        // whenever glyphs sit side by side and ink almost never does.
        let mut pieces = Vec::new();
        for (i, contour) in by_contour.iter().enumerate() {
            for piece in contour {
                let mid = piece.midpoint();
                let covered = by_contour
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .any(|(_, other)| chord_winding(other, mid) != 0);
                pieces.push(Piece {
                    may_be_interior: covered,
                    ..*piece
                });
            }
        }
        Self { pieces }
    }

    fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// The kernel
// ═══════════════════════════════════════════════════════════════════════════

fn constant(v: f32) -> Kernel {
    Kernel::constant(v)
}

/// **How many times the outline wraps this sample** — a signed integer,
/// exactly, because every contributing term is a hard mask selecting a
/// signed constant. Nothing here is soft.
///
/// That integrality is not decoration: it is what licenses both of the
/// constants below. `|w| < ½` is a valid test for *zero* only because the
/// nearest other value is `1`, and `|w| ≥ 1` is a valid test for *inside*
/// for the same reason. Stated once, here, instead of twice in prose.
#[derive(Clone)]
struct Winding(Kernel);

impl Winding {
    /// Outside every contour, and the identity of the sum.
    fn zero() -> Self {
        Self(constant(0.0))
    }

    fn sum(terms: &[Winding]) -> Self {
        let raw: Vec<Kernel> = terms.iter().map(|w| w.0.clone()).collect();
        Self(Kernel::fold(Monoid::SUM, &raw))
    }

    fn add(&self, other: &Self) -> Self {
        Self(self.0.add(&other.0))
    }

    fn sub(&self, other: &Self) -> Self {
        Self(self.0.sub(&other.0))
    }

    /// This winding where `inside` holds, and [`Winding::zero`] elsewhere —
    /// exact, because a closed contour winds zero about every exterior point.
    fn masked(&self, inside: &Kernel) -> Self {
        Self(inside.select(&self.0, &constant(0.0)))
    }

    /// `|w| < ½`, i.e. `w == 0`. See the type's docs for why ½ is the number.
    fn is_zero(&self) -> Kernel {
        self.0.abs().lt(&constant(WINDING_ZERO))
    }

    /// `|w| ≥ 1`, the non-zero fill rule.
    fn is_inside(&self) -> Kernel {
        self.0.abs().ge(&constant(1.0))
    }
}

/// **How far this sample is from the nearest boundary, in pixels of the
/// lattice being collapsed** — not of the frame the outline was built in.
///
/// The unit is the whole point of the type. Every construction from a raw
/// [`Kernel`] goes through [`Distance::in_pixels`], which divides by the
/// gradient of the edge function it came from, so the chain rule carries the
/// scale through every enclosing `Kernel::at`. Mixing in a length that
/// skipped that division is how a lower bound stopped being a lower bound
/// under a magnifying warp; with this type that subtraction does not
/// compile.
#[derive(Clone)]
struct Distance(Kernel);

impl Distance {
    /// Further than any ramp reaches — the identity of the `min`, and what a
    /// piece contributes to a sample outside its own support.
    fn unreachable() -> Self {
        Self(constant(RAMP_REACH))
    }

    /// `value / (‖∇scale‖ + ε)`: `value` measured in units of `scale`'s own
    /// gradient. **The only way to build one from a raw kernel.**
    ///
    /// `scale`'s coefficients are table reads, so `‖∇scale‖` is a `sqrt` at
    /// run time rather than the compile-time constant it was when they were
    /// literals. That is the price of one body per fold, and it is paid
    /// deliberately: precomputing the magnitude on the host would put the
    /// distance back in the outline's units and break under a magnifying
    /// warp. (`passes::lower_dwrt` is what makes it *compile* — a table read
    /// whose index does not move with the differentiation variable is a
    /// constant, so its derivative is 0.)
    fn in_pixels(value: &Kernel, scale: &Kernel) -> Self {
        let gradient = scale.dx().hypot(&scale.dy());
        Self(value.div(&gradient.add(&constant(MIN_GRADIENT))))
    }

    /// The capsule distance: two orthogonal distances, already both in
    /// pixels, combined by Pythagoras. This is the one place a distance is
    /// built from other distances rather than from an edge function.
    fn hypot(&self, other: &Self) -> Self {
        Self(self.0.mul(&self.0).add(&other.0.mul(&other.0)).sqrt())
    }

    fn abs(&self) -> Self {
        Self(self.0.abs())
    }

    fn min_of(terms: &[Distance]) -> Self {
        let raw: Vec<Kernel> = terms.iter().map(|d| d.0.clone()).collect();
        Self(Kernel::fold(Monoid::MIN, &raw))
    }

    fn min(&self, other: &Self) -> Self {
        Self(self.0.min(&other.0))
    }

    fn max(&self, other: &Self) -> Self {
        Self(self.0.max(&other.0))
    }

    /// Both operands are already in pixels, so the subtraction is meaningful
    /// — which is the invariant the type exists to hold.
    fn sub(&self, other: &Self) -> Self {
        Self(self.0.sub(&other.0))
    }

    /// This distance where `inside` holds, [`Distance::unreachable`]
    /// elsewhere — exact, because [`Support::around`] dilates by exactly
    /// [`RAMP_REACH`], so nothing outside the box could have won the `min`.
    fn masked(&self, inside: &Kernel) -> Self {
        Self(inside.select(&self.0, &constant(RAMP_REACH)))
    }

    /// A distance is not negative, and this is the one place that can be
    /// told: the chord bound is the chord's own distance less the curve's
    /// deviation, which goes below zero for a sample on the chord of a
    /// curve. Unclamped it makes coverage's outside arm `½ − d` exceed one —
    /// a coverage of 1.17, which is not a rounding difference but a value
    /// outside the function's range.
    fn non_negative(&self) -> Self {
        Self(self.0.max(&constant(0.0)))
    }

    fn select(mask: &Kernel, if_true: &Self, if_false: &Self) -> Self {
        Self(mask.select(&if_true.0, &if_false.0))
    }
}

// ─────────────────────────────────────────────────────────────────────────
// The piece row: one layout, one body per fold
// ─────────────────────────────────────────────────────────────────────────
//
// Every term below reads a piece's numbers **by column**, from a bound table
// at a reduce binder, so a glyph is two folds with fixed bodies rather than
// one arena fragment per piece: [`glyph`]'s `Kernel::sum_over` is the
// winding and its `Kernel::min_over` the ramp distance, and row `i` of the
// one table is piece `i` for both. The [`Coeff`] indirection is what keeps
// the read one definition regardless of which fold is asking.
//
// **A row always evaluates.** There is no per-kind branch on the host,
// because every distinction a piece could carry is a number that makes the
// term its own identity:
//
// - a line is a quadratic whose sliver sign (column 12) is 0, so
//   [`sliver_term`]'s mask is always true where the u/v columns are all 0
//   (`0 ≥ 0` and `0·0 − 0 ≤ 0`) and it selects `0.0` either way — never a
//   mask multiplied by a coefficient (see "Floating point at the edges" in
//   CLAUDE.md: a mask is a bit pattern, and `mask * 0.0` is not `0.0` when
//   the mask is all-ones-as-NaN);
// - a line's `u`/`v` are identically zero, so [`implicit_distance`] is
//   `|0| / (0 + ε) = 0`, which loses the `max` against a chord bound that
//   is never negative for a line — the identity, again;
// - a horizontal chord's `direction` (column 5) is 0, the honest winding
//   jump for an edge no horizontal ray crosses;
// - a piece nothing else covers carries a set mask in column 21, the
//   absorbing element of the `∨` [`boundary_distance`] ORs it into, so that
//   test is true without being asked.

/// `y_min` of the chord.
const COL_Y_MIN: usize = 0;
/// `y_max` of the chord.
const COL_Y_MAX: usize = 1;
/// `dx/dy` of the chord — `0.0` for a horizontal chord (H1: the crossing
/// band is empty for one regardless, so this is never read as a divisor,
/// but a finite placeholder keeps an `inf` from ever entering the table).
const COL_DX_OVER_DY: usize = 2;
/// `a.x`, the chord's start.
const COL_AX: usize = 3;
/// `a.y`, the chord's start.
const COL_AY: usize = 4;
/// The chord's `direction` (±1 running, `0.0` for a horizontal chord).
const COL_DIRECTION: usize = 5;
/// `u`'s affine coefficients `(a, b, c)` — `0.0` for a line.
const COL_U_A: usize = 6;
const COL_U_B: usize = 7;
const COL_U_C: usize = 8;
/// `v`'s affine coefficients `(a, b, c)` — `0.0` for a line.
const COL_V_A: usize = 9;
const COL_V_B: usize = 10;
const COL_V_C: usize = 11;
/// The sliver's sign (±1), `0.0` for a line — the sum's identity.
const COL_SLIVER_SIGN: usize = 12;
/// The signed distance to the chord's *line*, as `a·X + b·Y + c`: positive on
/// the left of `a → b`.
const COL_ACROSS_A: usize = 13;
const COL_ACROSS_B: usize = 14;
const COL_ACROSS_C: usize = 15;
/// The distance along the chord measured from `a`, as `a·X + b·Y + c`.
const COL_ALONG_A: usize = 16;
const COL_ALONG_B: usize = 17;
const COL_ALONG_C: usize = 18;
/// Half the chord's length — the capsule's half-extent along itself.
const COL_HALF_LENGTH: usize = 19;
/// How far the segment strays from its chord ([`Piece::deviation`]), `0.0`
/// for a line.
const COL_DEVIATION: usize = 20;
/// A **mask**, set for a piece no other contour's ink can reach: such a
/// piece always bounds what it draws, so [`boundary_distance`] ORs this in
/// and the winding test is not consulted.
///
/// A mask, not a threshold. This was `f32::MAX` compared with `lt`, which
/// makes "always" an argument about how large a winding can get — false
/// given enough overlapping contours. "Always" is the *absorbing* element of
/// `∨`, and masks are where the codebase already keeps bit patterns.
const COL_ALWAYS_BOUNDARY: usize = 21;
/// Columns in one piece's row: `y_min, y_max, dx/dy, a.x, a.y, direction,
/// u.a, u.b, u.c, v.a, v.b, v.c, sliver_sign, across.a, across.b, across.c,
/// along.a, along.b, along.c, half_length, deviation, always_boundary`.
const PIECE_ROW_COLS: usize = 22;

/// A winding this close to zero **is** zero: every winding term is a hard
/// mask selecting a signed constant, so the sum is an integer and half a unit
/// separates `0` from `±1`.
const WINDING_ZERO: f32 = 0.5;

/// One piece's row, column `k`, as a [`Kernel`] — a bound-table read at a
/// fold's binder. See the module section above.
type Coeff<'a> = &'a dyn Fn(usize) -> Kernel;

/// Piece `i`'s row, column by column: the one [`Coeff`] both of [`glyph`]'s
/// folds read through, `i` being the fold's own reduce binder.
fn row_at<'a>(table: &'a Kernel, i: &'a Kernel) -> impl Fn(usize) -> Kernel + 'a {
    move |k| table.at(&Kernel::constant(k as f32), i)
}

/// `a·X + b·Y + c` read from three of a row's columns. Loop–Blinn's `u` and
/// `v`, and the chord's across/along projections, are all this — one
/// definition, four uses.
fn affine(c: Coeff, a: usize, b: usize, constant: usize) -> Kernel {
    Kernel::x()
        .mul(&c(a))
        .add(&Kernel::y().mul(&c(b)))
        .add(&c(constant))
}

/// The affine maps taking `(X, Y)` to Loop–Blinn's canonical `(u, v)`,
/// landing `bulge`'s arc on the one parabola `v = u²` — computed once, on
/// the host, for [`piece_row`]. The one definition: [`sliver_term`] and
/// [`implicit_distance`] both read the columns it fills.
fn bulge_uv(bulge: Bulge) -> (Linear, Linear) {
    let Bulge { p0, p1, p2, .. } = bulge;
    let e1 = [p1[0] - p0[0], p1[1] - p0[1]];
    let e2 = [p2[0] - p0[0], p2[1] - p0[1]];
    let det = e1[0] * e2[1] - e1[1] * e2[0];
    // Barycentric coordinates as affine functions of the sample: λ₁ weights
    // the control point, λ₂ the end point, λ₀ = 1 − λ₁ − λ₂ the start.
    let lambda1 = Linear {
        a: e2[1] / det,
        b: -e2[0] / det,
        c: (p0[1] * e2[0] - p0[0] * e2[1]) / det,
    };
    let lambda2 = Linear {
        a: -e1[1] / det,
        b: e1[0] / det,
        c: (e1[1] * p0[0] - e1[0] * p0[1]) / det,
    };
    // (u, v) = (λ₁/2 + λ₂, λ₂), so λ₂ = v and λ₁ = 2(u − v).
    let u = Linear {
        a: lambda1.a / 2.0 + lambda2.a,
        b: lambda1.b / 2.0 + lambda2.b,
        c: lambda1.c / 2.0 + lambda2.c,
    };
    (u, lambda2)
}

/// One piece's row in the coefficient table (host-side, `f32`) — see the
/// column layout above. A line's sliver columns (6–12) stay `0.0`, the
/// sum's identity, and so does its deviation.
fn piece_row(piece: Piece) -> [f32; PIECE_ROW_COLS] {
    let chord = piece.chord;
    let [y_min, y_max] = chord.y_range();
    // H1: never store an infinite dx/dy — see the module note above. The
    // direction needs no such guard: `Chord::direction` is already `0` for
    // a horizontal chord, which is the honest winding jump rather than a
    // placeholder.
    let dx_over_dy = match chord.is_horizontal() {
        true => 0.0,
        false => (chord.b[0] - chord.a[0]) / (chord.b[1] - chord.a[1]),
    };
    let direction = chord.direction();
    let length = chord.length();
    let tangent = [
        (chord.b[0] - chord.a[0]) / length,
        (chord.b[1] - chord.a[1]) / length,
    ];
    let across = Linear::projection(chord.a, [-tangent[1], tangent[0]]);
    let along = Linear::projection(chord.a, tangent);
    let mut row = [0.0f32; PIECE_ROW_COLS];
    row[COL_Y_MIN] = y_min as f32;
    row[COL_Y_MAX] = y_max as f32;
    row[COL_DX_OVER_DY] = dx_over_dy as f32;
    row[COL_AX] = chord.a[0] as f32;
    row[COL_AY] = chord.a[1] as f32;
    row[COL_DIRECTION] = direction as f32;
    if let Some(bulge) = piece.bulge {
        let (u, v) = bulge_uv(bulge);
        row[COL_U_A] = u.a as f32;
        row[COL_U_B] = u.b as f32;
        row[COL_U_C] = u.c as f32;
        row[COL_V_A] = v.a as f32;
        row[COL_V_B] = v.b as f32;
        row[COL_V_C] = v.c as f32;
        row[COL_SLIVER_SIGN] = bulge.sign as f32;
    }
    row[COL_ACROSS_A] = across.a as f32;
    row[COL_ACROSS_B] = across.b as f32;
    row[COL_ACROSS_C] = across.c as f32;
    row[COL_ALONG_A] = along.a as f32;
    row[COL_ALONG_B] = along.b as f32;
    row[COL_ALONG_C] = along.c as f32;
    row[COL_HALF_LENGTH] = (length / 2.0) as f32;
    row[COL_DEVIATION] = piece.deviation() as f32;
    row[COL_ALWAYS_BOUNDARY] = OpKind::mask(!piece.may_be_interior);
    row
}

/// `a·X + b·Y + c`.
#[derive(Clone, Copy, Debug)]
struct Linear {
    a: f64,
    b: f64,
    c: f64,
}

impl Linear {
    /// `n · (P − a)`.
    fn projection(a: P, n: P) -> Self {
        Self {
            a: n[0],
            b: n[1],
            c: -(n[0] * a[0] + n[1] * a[1]),
        }
    }
}

/// **The ramp distance** of one piece: how far the sample is from it, in
/// pixels of the lattice being collapsed, and [`RAMP_REACH`] — a distance
/// that never wins the fold's `min` — where the piece is not a boundary.
///
/// Only a boundary softens a pixel. A font may draw an edge that is not on
/// the outside of anything — a stroke built from two overlapping contours,
/// or two components of a compound glyph that overlap — and ink continues
/// across such an edge rather than stopping at it. Taking the nearest
/// *drawn* edge would dim a line straight through solid ink (measured: two
/// overlapping squares read 0.5 along their shared interior edges, a
/// half-dark seam one pixel wide).
///
/// Whether a piece is a boundary is a local question the winding already
/// answers. Let `w₋` be the winding with this piece's own contribution
/// removed; the two sides of the piece are `w₋` and `w₋ + direction`, which
/// differ by one and so are never both zero. The piece separates ink from
/// no-ink exactly when one of them is zero.
///
/// It is only *that* piece's own contribution that is removed, so `winding`
/// is read once here — by name ([`Kernel::by_ref`]), which is why the whole
/// fold unrolls the sum once rather than once per piece.
///
/// The test is asked of every row and answered by column 21: a piece no
/// other contour's ink can reach carries a set mask, ORed into the result,
/// so it passes unconditionally. That is not
/// merely an optimization — the two-sides formula reads the *crossing*
/// term, which is 0 wherever the sample's row lies outside the chord's own
/// Y band, so for a horizontal or near-horizontal chord it is `w₋ = w` and
/// the pair is not the piece's two sides at all. Asking it of a piece that
/// always separates would drop that piece's ramp, not merely re-derive it.
fn boundary_distance(c: Coeff, winding: &Winding) -> Distance {
    let own = crossing_term(c).add(&sliver_term(c));
    let without = winding.sub(&own);
    // `dir` is itself a winding: the jump a ray takes crossing this piece.
    let other_side = without.add(&Winding(c(COL_DIRECTION)));
    let separates = without
        .is_zero()
        .or(&other_side.is_zero())
        .or(&c(COL_ALWAYS_BOUNDARY));
    Distance::select(&separates, &piece_distance(c), &Distance::unreachable())
}

/// The distance from the sample to one piece, in pixels: the larger of the
/// implicit's own estimate and a lower bound built from the chord.
///
/// The distance to the chord, less how far the segment strays from it, is a
/// lower bound on the distance to the segment — exact for a line, where
/// nothing strays. For a curve the implicit is the better estimate near the
/// arc and the bound is the sound one away from it, so take whichever is
/// larger: a bound can never exceed the truth, so the larger is the closer.
/// A line's implicit is exactly 0 and its bound never negative, so the same
/// `max` is the identity there and needs no host-side branch.
///
/// The chord's own distance is the **capsule**, not the distance to its
/// infinite line, and that is load-bearing rather than tidy: a sample
/// outside a convex corner is far from both segments but close to the
/// *line* of each, so a minimum over line distances would soften a pixel a
/// corner's width away — a dark halo on every serif.
///
/// Reads columns 13–20 (and, through [`implicit_distance`], 6–11).
fn piece_distance(c: Coeff) -> Distance {
    let across = affine(c, COL_ACROSS_A, COL_ACROSS_B, COL_ACROSS_C);
    let along = affine(c, COL_ALONG_A, COL_ALONG_B, COL_ALONG_C);
    let half = c(COL_HALF_LENGTH);
    let overshoot = along.sub(&half).abs().sub(&half).max(&constant(0.0));
    let perpendicular = Distance::in_pixels(&across, &across);
    let past_the_end = Distance::in_pixels(&overshoot, &along);
    let capsule = perpendicular.hypot(&past_the_end);
    // Both in the same units, or the subtraction is meaningless. The capsule
    // is already divided by its own gradient, so it is in pixels of whatever
    // lattice this is baked on; the deviation is a length in the outline's
    // own coordinates. Those agree only while nothing warps the kernel —
    // under a magnifying `Kernel::at` the raw column under-subtracts by the
    // magnification, and a lower bound that is too large is not a lower
    // bound. Normalising it through the same gradient makes them agree by
    // construction; today it divides by one.
    let bound = capsule.sub(&Distance::in_pixels(&c(COL_DEVIATION), &across));
    implicit_distance(c).max(&bound)
}

/// **The crossing term**: `±1` where a ray from the sample to `+X` crosses
/// this chord, `0` where it does not.
///
/// The schoolbook inside test, and the whole of the winding for a straight
/// edge: does the chord span this row, and is it to the right? The row test
/// is half-open (`y_min ≤ Y < y_max`) so a vertex shared by two chords is
/// counted once, and the sign is the direction the chord runs in.
///
/// Everything here but the final comparison is a function of `Y` alone —
/// one multiply-add per chord per row — so it hoists into the row prologue
/// and leaves one comparison per pixel. That is the *good* kind of Y-only
/// work; what made the scanline formulation slow was hoisting a **root
/// solve** per segment per row, and there is none here because a chord is a
/// line. A curve's own contribution is not a crossing at all — see
/// [`sliver_term`].
///
/// Reads columns 0–5 of `c` (`y_min, y_max, dx/dy, a.x, a.y, direction`) —
/// see the piece-row layout above [`piece_row`]. Asked of every row by both
/// of [`glyph`]'s folds, with no host-side pruning: a horizontal chord's
/// `direction` is already `0`, so the term is its own identity there.
fn crossing_term(c: Coeff) -> Winding {
    let spans = Kernel::y()
        .ge(&c(COL_Y_MIN))
        .and(&Kernel::y().lt(&c(COL_Y_MAX)));
    // x = a.x + (Y − a.y)·dx/dy, exact and linear: a chord is a line, so
    // there is no discriminant and no root to be on the wrong side of.
    let crossing_x = Kernel::y()
        .sub(&c(COL_AY))
        .mul(&c(COL_DX_OVER_DY))
        .add(&c(COL_AX));
    Winding(
        spans
            .and(&crossing_x.gt(&Kernel::x()))
            .select(&c(COL_DIRECTION), &constant(0.0)),
    )
}

/// **The sliver term**: `±1` inside the crescent between a curve and its
/// chord, `0` outside.
///
/// Replacing a curve by its chord changes the winding by exactly the closed
/// loop *chord → curve*, whose winding is `±1` on the crescent between them
/// and `0` elsewhere, with the sign of the control triangle's orientation.
/// So this is the only place a curve differs from a straight edge, and the
/// only place Loop and Blinn's construction is used.
///
/// The crescent is `{v ≥ u²} ∩ {v ≤ u}` under the affine map taking
/// `P0, P1, P2` to `(0,0), (½,0), (1,1)` — the change of coordinates that
/// lands *every* quadratic on the one parabola `v = u²`, turning "solve for
/// where the curve is" into "evaluate a formula and read its sign".
///
/// ## The fence is one half-plane, not a triangle
///
/// `u² − v` is *wrong* outside the arc, not merely slow, so the test has to
/// be fenced — but the control triangle is not the cheapest fence, because
/// two of its three edges are implied by the parabola itself. In canonical
/// coordinates the triangle is `v ≥ 0`, `v ≤ u` (the chord), and
/// `1 + v − 2u ≥ 0`; given `v ≥ u²`,
///
/// - `v ≥ u² ≥ 0` is the first edge, and
/// - `1 + v − 2u ≥ 1 + u² − 2u = (1 − u)² ≥ 0` is the third.
///
/// So only the chord's own half-plane is left, and `u² ≤ v ≤ u` is already
/// bounded on its own: it forces `u ∈ [0, 1]`, since `u² ≤ u` holds nowhere
/// else. An axis-aligned box would have been strictly worse — it is not
/// equivalent to the triangle, so it would be needed *in addition to* the
/// chord test rather than instead of it.
///
/// Affine maps preserve half-planes and intersections, so this is the
/// crescent in screen space too: two comparisons and one `and`.
///
/// Reads columns 6–12 of `c` (`u.a, u.b, u.c, v.a, v.b, v.c, sign` — see
/// [`piece_row`]/[`bulge_uv`]); the `(u, v)` affine map itself is computed
/// once on the host, not here. A line's row makes every one of those
/// columns `0.0`, which makes this term the identity unconditionally (see
/// the module note above [`COL_Y_MIN`]), so a line needs no special case.
fn sliver_term(c: Coeff) -> Winding {
    let u = affine(c, COL_U_A, COL_U_B, COL_U_C);
    let v = affine(c, COL_V_A, COL_V_B, COL_V_C);
    let under_the_chord = u.ge(&v);
    let over_the_parabola = u.mul(&u).sub(&v).le(&constant(0.0));
    Winding(
        under_the_chord
            .and(&over_the_parabola)
            .select(&c(COL_SLIVER_SIGN), &constant(0.0)),
    )
}

/// Coverage, given the winding and the distance to the nearest boundary: an
/// exact integer decides inside from outside, and a distance decides how
/// much of the pixel.
///
/// Both arguments are already resolved by the caller — [`glyph`]'s
/// `Kernel::sum_over` and `Kernel::min_over` over one bound table — and read
/// here as plain `Kernel`s; nothing below cares how either was built.
///
/// The two halves are separate on purpose. An earlier draft made each
/// winding term *soft*, so a crossing's existence and its coverage were one
/// number, and a comparison landing on the wrong side of an edge moved
/// coverage by half a unit rather than by a rounding. That coupling is what
/// made `'8'` render with a smear at its waist.
fn coverage(winding: &Winding, distance: &Distance) -> Kernel {
    // The one place the two types are spent: past here it is a plain
    // coverage kernel, and neither invariant is any longer available to
    // state. `non_negative` is why the outside arm cannot exceed one.
    let Distance(d) = distance.non_negative();
    let half = constant(0.5);
    winding.is_inside().select(
        &half.add(&d).min(&constant(1.0)),
        &half.sub(&d).max(&constant(0.0)),
    )
}

/// `|f|/‖∇f‖`, the implicit's own first-order distance to the parabola —
/// what a GPU Loop–Blinn shader antialiases with. Accurate near the curve
/// and **not sound away from it**: it underestimates where the curve is
/// sharp (measured: a texel a full pixel outside `O` at 7 px read as an
/// edge texel), and an underestimate past the ramp is not a rounding
/// difference but a saturation failure. [`piece_distance`] therefore takes
/// it against a bound built from the chord.
///
/// Reads the same `(u, v)` columns [`sliver_term`] does — [`bulge_uv`] is
/// the one place the map is computed. A line's are all `0.0`, which makes
/// `f` identically zero and this term exactly `0`: the `max`'s identity
/// against a bound that is never negative for a line.
fn implicit_distance(c: Coeff) -> Distance {
    let u = affine(c, COL_U_A, COL_U_B, COL_U_C);
    let v = affine(c, COL_V_A, COL_V_B, COL_V_C);
    let f = u.mul(&u).sub(&v);
    Distance::in_pixels(&f, &f).abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    // The three tests below exercise `Piece::quad`'s splitting/flattening
    // directly rather than through an `Outline`/`Contour` of one open arc:
    // an arc alone is exactly the shape `Contour::new` now refuses (its
    // `to` never meets its own `from`), and going through
    // `Pieces::of(&Outline{..})` for a single-segment, single-contour
    // outline is nothing more than this one call — `Pieces::of` only adds
    // `may_be_interior` bookkeeping against *other* contours, and there is
    // no other contour here (`covered` is `false` either way). So the
    // shorter path tests the identical splitting logic without needing a
    // closed boundary these cases were never testing in the first place.

    #[test]
    fn a_flat_quadratic_is_its_chord() {
        let mut pieces = Vec::new();
        Piece::quad(
            wide([0.0, 0.0]),
            wide([5.0, 0.001]), // deviation 5e-4, under FLAT_ENOUGH
            wide([10.0, 0.0]),
            MAX_SPLITS,
            &mut pieces,
        );
        assert_eq!(pieces.len(), 1);
        assert!(pieces[0].bulge.is_none());
    }

    /// A curve too fat for the ramp's chord bound is split until it is not,
    /// and every piece is a real quadratic — the winding never needs a
    /// split, so a split must not change it.
    #[test]
    fn a_fat_quadratic_is_split_until_its_chord_bounds_it() {
        let mut pieces = Vec::new();
        Piece::quad(
            wide([0.0, 0.0]),
            wide([10.0, 40.0]),
            wide([20.0, 0.0]),
            MAX_SPLITS,
            &mut pieces,
        );
        assert!(pieces.len() > 1, "a 20-px bulge was not split");
        for p in &pieces {
            assert!(
                p.deviation() <= MAX_DEVIATION,
                "a piece still strays {} from its chord",
                p.deviation()
            );
            assert!(p.bulge.is_some(), "a split piece lost its curve");
        }
    }

    /// A curve whose control point projects *outside* its chord's span is
    /// still measured against the segment, not the chord's infinite line.
    ///
    /// This is [`Piece::deviation`]'s own documented counterexample, and
    /// [`Piece::quad`] used to miss it in both directions: measuring
    /// `|area2| / length / 2` the hook below reads as 0.0025 — under
    /// [`FLAT_ENOUGH`] — so the whole curve was discarded as a straight
    /// line while it strays about 50 units from that line. Discarding a
    /// sliver changes the **winding**, which nothing is allowed to
    /// approximate, so this is a stronger requirement than the ramp's.
    ///
    /// Scaled down by 1e-3 from the doc's `(0, 0) → (−100, 0.1) → (1, 0)`
    /// only to put the deviation under `FLAT_ENOUGH` on the line measure —
    /// the failure is scale-free, and this is the half of it that is not
    /// merely a loose bound.
    #[test]
    fn a_hook_is_not_flattened_by_measuring_the_wrong_distance() {
        let mut pieces = Vec::new();
        Piece::quad(
            wide([0.0, 0.0]),
            wide([-100.0, 0.005]),
            wide([1.0, 0.0]),
            MAX_SPLITS,
            &mut pieces,
        );
        // Not "every piece is a curve": once split, a sub-arc of a hook
        // really can be straight to within `FLAT_ENOUGH`, and emitting it
        // as a line is right. What must survive is the *shape* — the
        // pieces have to go where the curve goes, so their endpoints reach
        // far from the chord the whole thing used to collapse onto.
        let chord = Chord {
            a: [0.0, 0.0],
            b: [1.0, 0.0],
        };
        let reach = pieces
            .iter()
            .map(|p| point_to_segment(p.chord.a, chord).max(point_to_segment(p.chord.b, chord)))
            .fold(0.0f64, f64::max);
        assert!(
            reach > 10.0,
            "a hook straying ~50 units collapsed onto its chord: reach {reach}"
        );
        for p in &pieces {
            assert!(
                p.deviation() <= MAX_DEVIATION,
                "a piece still strays {} from its chord",
                p.deviation()
            );
        }
    }
}
