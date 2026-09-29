//! **The exact area of a glyph under each texel**, in `f64`, from nothing
//! but the outline's control points — the external reference a coverage
//! rasterizer is judged against.
//!
//! ## What it computes
//!
//! For the texel `(i, j)`, the square `[i, i+1) × [j, j+1)`:
//!
//! ```text
//! F(i, j)   = ∫∫_texel w(x, y) dx dy          w = the outline's winding number
//! coverage  = min(|F|, 1)
//! ```
//!
//! `w` is the non-zero rule's integer, so `F` is the texel's **signed area
//! under ink**, and `min(|F|, 1)` is the coverage FreeType's rasterizer
//! computes — exact wherever contours do not overlap *inside the texel*, and
//! otherwise off by the overlap of two fractions (a texel half-covered by two
//! overlapping contours reads 1, not ½). That approximation is accepted, not
//! an error of this reference: it is what the coverage *means* here.
//!
//! ## How
//!
//! Green's theorem turns the area integral into a line integral along the
//! outline. With `w` counted by a ray to `+X` (`+1` for an edge running
//! toward `+Y`), every oriented piece of outline contributes
//!
//! ```text
//! ∫_piece clamp(x − i, 0, 1) · [j ≤ y ≤ j+1] dy
//! ```
//!
//! to texel `(i, j)`, and the contributions of a closed outline sum to `F`.
//! A piece lying inside one cell's closed square `[i, i+1] × [j, j+1]`
//! therefore contributes `∫ (x − i) dy` to that texel, its rise `Δy` to
//! every texel of the row to its left, and nothing to the right — the
//! trapezoid-and-cover accumulation of FreeType and font-rs, done in `f64`.
//!
//! - A **line** is cut at every integer `x` and `y` it crosses; each cut
//!   piece's `∫ (x − i) dy` is its trapezoid, exactly.
//! - A **quadratic** is halved (de Casteljau) until each piece's control
//!   hull lies inside one cell — where `∫ (x − i) dy` is a cubic polynomial
//!   in the parameter, integrated exactly below — or until the piece is
//!   within [`FLATNESS`] (10⁻⁹ px) of its chord, where it is flattened to
//!   that chord and cut like a line. Only the pieces straddling a cell
//!   boundary are ever flattened, so the whole error is `FLATNESS` times the
//!   length of outline near a cell edge: below 10⁻⁸ of a texel's area.
//!
//! Nothing here reads the renderer's geometry. The outline comes from the
//! TrueType parser in font units ([`Font::outline_by_id`]); the map to the
//! screen frame is restated below in `f64` from the font's own metrics, and
//! every piece of arithmetic after that is this file's.

use pixelflow_graphics::fonts::{Font, Outline, Segment};

/// A point, `[x, y]`, in the texel frame.
pub type Point = [f64; 2];

/// How far a quadratic may stray from its chord before it is flattened to
/// it — the maximum over the parameter of `|B(t) − L(t)|`, which is
/// `|p0 − 2p1 + p2| / 4`. Each halving quarters it, so it is reached in
/// `log₄(extent / FLATNESS)` levels, about twenty for a glyph.
pub const FLATNESS: f64 = 1e-9;

/// One piece of a closed outline, oriented.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Piece {
    /// A straight edge `from → to`.
    Line(Point, Point),
    /// A quadratic Bézier `from → control → to`.
    Quad(Point, Point, Point),
}

/// Every segment of `outline`, each control point pushed through `map`.
#[must_use]
pub fn pieces(outline: &Outline, map: impl Fn([f32; 2]) -> Point) -> Vec<Piece> {
    outline
        .segments()
        .map(|segment| match segment {
            Segment::Line { from, to } => Piece::Line(map(from), map(to)),
            Segment::Quad { from, control, to } => Piece::Quad(map(from), map(control), map(to)),
        })
        .collect()
}

/// The glyph for `ch` in the renderer's screen frame at `size` pixels,
/// restated rather than borrowed: the font-unit outline, scaled so the
/// ascent-to-descent height is `size`, flipped so `y` grows downward, with
/// the ascent line at `y = 0` and the descent line at `y = size`.
///
/// # Panics
///
/// Panics if the font has no glyph for `ch`.
#[must_use]
pub fn screen_pieces(font: &Font, ch: char, size: f64) -> Vec<Piece> {
    let id = font
        .cmap_lookup(ch)
        .unwrap_or_else(|| panic!("the font has no glyph for {ch:?}"));
    let outline = font
        .outline_by_id(id)
        .unwrap_or_else(|| panic!("the font has no outline for {ch:?}"));
    let ascent = f64::from(font.ascent);
    let scale = size / (ascent + f64::from(font.descent).abs());
    pieces(&outline, |[x, y]| {
        [f64::from(x) * scale, (ascent - f64::from(y)) * scale]
    })
}

/// The texels `[0, width) × [0, height)`, row-major.
#[derive(Clone, Copy, Debug)]
pub struct Grid {
    /// Texels per row.
    pub width: usize,
    /// Rows.
    pub height: usize,
}

/// `F(i, j)` for every texel of `grid`, row-major: the signed area of the
/// texel under the outline `pieces` draw. See the module docs.
///
/// `pieces` must close — every contour returning to its start — or `F` is
/// not an area. (Ink outside the grid is not an error: it simply is not in
/// any texel.)
#[must_use]
pub fn signed_area(pieces: &[Piece], grid: Grid) -> Vec<f64> {
    let mut acc = Accumulator::new(grid);
    for &piece in pieces {
        match piece {
            Piece::Line(a, b) => acc.line(a, b),
            Piece::Quad(a, c, b) => acc.quad(a, c, b),
        }
    }
    acc.finish()
}

/// The coverage a signed area means: `min(|F|, 1)`.
#[must_use]
pub fn coverage(signed: f64) -> f64 {
    signed.abs().min(1.0)
}

/// Per texel, the area the pieces inside its cell leave (`∫ (x − i) dy`),
/// and per cell the rise every texel left of it inherits.
struct Accumulator {
    grid: Grid,
    /// `width` per row.
    area: Vec<f64>,
    /// `width + 1` per row; the last column holds whatever lies right of
    /// the grid, which every texel of the row inherits.
    rise: Vec<f64>,
}

impl Accumulator {
    fn new(grid: Grid) -> Self {
        Self {
            grid,
            area: vec![0.0; grid.width * grid.height],
            rise: vec![0.0; (grid.width + 1) * grid.height],
        }
    }

    fn width(&self) -> f64 {
        self.grid.width as f64
    }

    fn height(&self) -> f64 {
        self.grid.height as f64
    }

    /// A piece of outline inside cell `(i, j)`'s closed square, rising by
    /// `rise` with `∫ (x − i) dy = moment`. Left of the grid it reaches no
    /// texel; right of it, every texel of the row inherits its rise.
    fn deposit(&mut self, [i, j]: [f64; 2], moment: f64, rise: f64) {
        if j < 0.0 || j >= self.height() || i < 0.0 {
            return;
        }
        let row = j as usize;
        let stride = self.grid.width + 1;
        if i >= self.width() {
            self.rise[row * stride + self.grid.width] += rise;
            return;
        }
        let col = i as usize;
        self.area[row * self.grid.width + col] += moment;
        self.rise[row * stride + col] += rise;
    }

    /// A straight edge, cut at every integer `x` and `y` it crosses.
    fn line(&mut self, a: Point, b: Point) {
        let mut cuts = vec![0.0, 1.0];
        for axis in 0..2 {
            let (lo, hi) = (a[axis].min(b[axis]), a[axis].max(b[axis]));
            let mut k = lo.floor() + 1.0;
            while k < hi {
                cuts.push((k - a[axis]) / (b[axis] - a[axis]));
                k += 1.0;
            }
        }
        cuts.sort_by(f64::total_cmp);
        let at = |t: f64| [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
        for pair in cuts.windows(2) {
            let (p, q) = (at(pair[0]), at(pair[1]));
            let mid = at(0.5 * (pair[0] + pair[1]));
            let cell = [mid[0].floor(), mid[1].floor()];
            let rise = q[1] - p[1];
            self.deposit(cell, (0.5 * (p[0] + q[0]) - cell[0]) * rise, rise);
        }
    }

    /// A quadratic, halved until each piece sits inside one cell (and is
    /// integrated exactly there) or is flat enough to be its chord.
    fn quad(&mut self, p0: Point, p1: Point, p2: Point) {
        let (x_lo, x_hi) = (p0[0].min(p1[0]).min(p2[0]), p0[0].max(p1[0]).max(p2[0]));
        let (y_lo, y_hi) = (p0[1].min(p1[1]).min(p2[1]), p0[1].max(p1[1]).max(p2[1]));
        // Above or below every row, or left of every texel: it reaches none.
        if y_hi <= 0.0 || y_lo >= self.height() || x_hi <= 0.0 {
            return;
        }
        // Right of every texel: each row inherits its clipped rise, whatever
        // the path between the ends.
        if x_lo >= self.width() {
            let mut j = y_lo.floor().max(0.0);
            while j < y_hi.min(self.height()) {
                let clip = |y: f64| y.clamp(j, j + 1.0);
                self.deposit([self.width(), j], 0.0, clip(p2[1]) - clip(p0[1]));
                j += 1.0;
            }
            return;
        }
        let cell = [x_lo.floor(), y_lo.floor()];
        if x_hi <= cell[0] + 1.0 && y_hi <= cell[1] + 1.0 {
            self.deposit(cell, quad_moment(p0, p1, p2, cell[0]), p2[1] - p0[1]);
            return;
        }
        let bend = [p0[0] - 2.0 * p1[0] + p2[0], p0[1] - 2.0 * p1[1] + p2[1]];
        if 0.25 * bend[0].hypot(bend[1]) <= FLATNESS {
            self.line(p0, p2);
            return;
        }
        let mid = |a: Point, b: Point| [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])];
        let (l, r) = (mid(p0, p1), mid(p1, p2));
        let m = mid(l, r);
        self.quad(p0, l, m);
        self.quad(m, r, p2);
    }

    /// `F(i, j) = area(i, j) + Σ_{i' > i} rise(i', j)`.
    fn finish(self) -> Vec<f64> {
        let Grid { width, height } = self.grid;
        let mut out = vec![0.0; width * height];
        for row in 0..height {
            let rise = &self.rise[row * (width + 1)..(row + 1) * (width + 1)];
            let mut inherited = rise[width];
            for col in (0..width).rev() {
                out[row * width + col] = self.area[row * width + col] + inherited;
                inherited += rise[col];
            }
        }
        out
    }
}

/// `∫₀¹ (x(t) − i) · y′(t) dt` for the quadratic `p0 → p1 → p2`, exactly.
///
/// With `a_k = x_k − i` and the steps `e₀ = y₁ − y₀`, `e₁ = y₂ − y₁`,
/// `y′ = 2((1−t)e₀ + t·e₁)`, and a product of Bernstein polynomials is a
/// Bernstein polynomial whose integral is `1/(n+1)`:
/// `a₀(e₀/2 + e₁/6) + a₁(e₀ + e₁)/3 + a₂(e₀/6 + e₁/2)`. A line (`p1` the
/// midpoint) reduces to its trapezoid, `(a₀ + a₂)/2 · (y₂ − y₀)`.
fn quad_moment(p0: Point, p1: Point, p2: Point, i: f64) -> f64 {
    let (a0, a1, a2) = (p0[0] - i, p1[0] - i, p2[0] - i);
    let (e0, e1) = (p1[1] - p0[1], p2[1] - p1[1]);
    a0 * (e0 / 2.0 + e1 / 6.0) + a1 * (e0 + e1) / 3.0 + a2 * (e0 / 6.0 + e1 / 2.0)
}
