//! **The formula glyph on the outlines a font rarely draws**, judged texel by
//! texel against the exact area (`common/exact_area.rs`).
//!
//! `glyph_exact_area.rs` measures the font's own glyphs; this file measures
//! the geometry its integrand is most likely to get wrong: pieces that are
//! horizontal or vertical, that start or end on a pixel's edge or corner or
//! centre, that are tiny or span many rows; contours that share a vertex or
//! an edge, overlap, or wind a hole; quadratics whose control sits on an
//! end, whose ends coincide, or that turn back on both axes; ink reaching
//! past the lattice on every side or far from the origin; and 120 random
//! outlines, half drawn on a quarter-pixel grid, where those coincidences
//! are the rule rather than the exception.
//!
//! ## What is asserted
//!
//! Coverage is `min(|F|, 1)` with its ends snapped (`loop_blinn::
//! COVERAGE_SNAP`, `2⁻¹⁰`). So, with `e` the exact coverage, `o` ours and
//! `ε` the closed form's own error bound at the texel:
//!
//! - `e` at most `2⁻¹⁰ − ε` reads **exactly** `0`, and at least
//!   `1 − 2⁻¹⁰ + ε` **exactly** `1`;
//! - between the snaps, `|o − e| ≤ ε`;
//! - within `ε` of a snap threshold, either is allowed: `|o − e| ≤ 2⁻¹⁰ + ε`.
//!
//! `ε = 2⁻²²·(1 + |X| + |Y| + 2·extent)` at the texel's centre `(X, Y)`,
//! `extent` the outline's larger side: the parameter an arc is read at
//! resolves to `2⁻²⁴`, which the arc's extent multiplies, and the
//! coordinates carry their rounding into the clamp
//! (`pixelflow_ir::IntervalFold::arc_moment`, "Floating point"; pinned per
//! arc by `pixelflow-core/tests/arc_oracle.rs`). Derived, not measured:
//! the worst texel here uses under a quarter of it (a random contour off
//! the grid; 9% for the named shapes), and the font's about a tenth
//! (`glyph_exact_area.rs`).

#[path = "common/exact_area.rs"]
mod exact_area;

use exact_area::{coverage, screen_pieces, signed_area, Grid, Piece};
use pixelflow_core::{Kernel, Lattice};
use pixelflow_graphics::fonts::{loop_blinn, Contour, Font, Outline, Segment};

const FONT_DATA: &[u8] = include_bytes!("../assets/DejaVuSansMono-Fallback.ttf");

/// `2⁻²²`: the unit of the closed form's error bound; see the module docs.
const ARC_TOLERANCE_UNIT: f64 = 1.0 / 4_194_304.0;

type P = [f32; 2];

fn line(from: P, to: P) -> Segment {
    Segment::Line { from, to }
}

fn quad(from: P, control: P, to: P) -> Segment {
    Segment::Quad { from, control, to }
}

/// A closed polygon through `points`.
fn polygon(points: &[P]) -> Vec<Segment> {
    (0..points.len())
        .map(|k| line(points[k], points[(k + 1) % points.len()]))
        .collect()
}

fn rect([x0, y0]: P, [x1, y1]: P) -> Vec<Segment> {
    polygon(&[[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
}

fn outline(contours: Vec<Vec<Segment>>) -> Outline {
    Outline {
        contours: contours
            .into_iter()
            .map(|c| Contour::new(c).expect("the test's contour closes"))
            .collect(),
    }
}

/// Ours: the glyph over a `width × height` lattice, texel `(i, j)` sampling
/// `(i + ½ + ox, j + ½ + oy)`.
fn ours(outline: &Outline, grid: Grid, [ox, oy]: P) -> Vec<f32> {
    let glyph = loop_blinn::glyph(outline);
    let at_centres = glyph.kernel().at(
        &Kernel::x().add(&Kernel::constant(0.5 + ox)),
        &Kernel::y().add(&Kernel::constant(0.5 + oy)),
    );
    glyph
        .bake(&at_centres, Lattice::frame(grid.width, grid.height))
        .into_buffer()
}

/// The reference, over the same texels.
fn exact(outline: &Outline, grid: Grid, [ox, oy]: P) -> Vec<f64> {
    let pieces = exact_area::pieces(outline, |[x, y]| {
        [f64::from(x) - f64::from(ox), f64::from(y) - f64::from(oy)]
    });
    signed_area(&pieces, grid)
        .into_iter()
        .map(coverage)
        .collect()
}

/// A tile to judge: `grid`'s texel `(i, j)` is centred at
/// `origin + (i + ½, j + ½)` in the outline's frame, and the outline's
/// larger side is `extent` (both set the bound; see the module docs).
struct Judged<'a> {
    name: &'a str,
    grid: Grid,
    origin: [f64; 2],
    extent: f64,
}

impl Judged<'_> {
    /// `ours` against `exact` over the tile, held to the module's contract.
    /// Returns the worst error between the snaps as a multiple of the bound.
    fn assert(&self, ours: &[f32], exact: &[f64]) -> f64 {
        let snap = f64::from(loop_blinn::COVERAGE_SNAP);
        let mut worst = 0.0f64;
        let mut wrong = Vec::new();
        for (k, (&o, &e)) in ours.iter().zip(exact).enumerate() {
            let (i, j) = (k % self.grid.width, k / self.grid.width);
            let centre = [
                self.origin[0] + i as f64 + 0.5,
                self.origin[1] + j as f64 + 0.5,
            ];
            let eps =
                ARC_TOLERANCE_UNIT * (1.0 + centre[0].abs() + centre[1].abs() + 2.0 * self.extent);
            let o = f64::from(o);
            let ok = match e {
                _ if o.is_nan() => false,
                e if e <= snap - eps => o == 0.0,
                e if e >= 1.0 - snap + eps => o == 1.0,
                e if (snap + eps..=1.0 - snap - eps).contains(&e) => {
                    worst = worst.max((o - e).abs() / eps);
                    (o - e).abs() <= eps
                }
                e => (o - e).abs() <= snap + eps,
            };
            if !ok {
                wrong.push(format!("({i}, {j}): ours {o}, exact {e}"));
            }
        }
        assert!(
            wrong.is_empty(),
            "{}: {} texel(s) off the exact area:\n{}",
            self.name,
            wrong.len(),
            wrong[..wrong.len().min(24)].join("\n")
        );
        worst
    }
}

/// Every texel of `outline` over `grid`, sampled at `offset` past the
/// lattice's own centres, held to the module's contract. Returns the worst
/// error between the snaps as a multiple of the bound.
fn assert_exact_at(name: &str, outline: &Outline, grid: Grid, offset: P) -> f64 {
    let [x0, y0, x1, y1] = outline.bounds().expect("the test's outline has points");
    let judged = Judged {
        name,
        grid,
        origin: offset.map(f64::from),
        extent: f64::from((x1 - x0).max(y1 - y0)),
    };
    judged.assert(&ours(outline, grid, offset), &exact(outline, grid, offset))
}

fn assert_exact(name: &str, outline: &Outline, grid: Grid) {
    assert_exact_at(name, outline, grid, [0.0, 0.0]);
}

const GRID: Grid = Grid {
    width: 12,
    height: 12,
};

// ───────────────────────── horizontal and vertical ─────────────────────────

/// Rectangles are nothing but horizontal pieces (dropped) and vertical ones
/// (a clamp in `x` over a band in `y`): edges on pixel edges, at pixel
/// centres, at thirds; thinner than a pixel; wholly inside one pixel.
#[test]
fn rectangles_on_and_off_the_pixel_grid() {
    let cases: [(&str, P, P); 7] = [
        ("on pixel edges", [2.0, 3.0], [7.0, 9.0]),
        ("at pixel centres", [2.5, 3.5], [7.5, 9.5]),
        (
            "at thirds",
            [1.0 / 3.0, 2.0 / 3.0],
            [22.0 / 3.0, 17.0 / 3.0],
        ),
        ("a hairline column", [4.3, 1.0], [4.6, 10.0]),
        ("a hairline row", [1.0, 4.3], [10.0, 4.6]),
        ("inside one pixel", [5.2, 6.1], [5.7, 6.9]),
        ("a pixel's own square", [5.0, 6.0], [6.0, 7.0]),
    ];
    for (name, lo, hi) in cases {
        assert_exact(name, &outline(vec![rect(lo, hi)]), GRID);
    }
}

// ─────────────────────── ends on edges, corners, centres ───────────────────

/// A diamond whose four vertices are pixel corners, so every edge runs
/// through a corner of every pixel it crosses.
#[test]
fn a_diamond_through_pixel_corners() {
    let d = polygon(&[[6.0, 1.0], [11.0, 6.0], [6.0, 11.0], [1.0, 6.0]]);
    assert_exact("diamond", &outline(vec![d]), GRID);
}

/// A polygon whose vertices are pixel centres, with steep and shallow edges.
#[test]
fn a_polygon_through_pixel_centres() {
    let p = polygon(&[[1.5, 1.5], [10.5, 2.5], [8.5, 10.5], [5.5, 6.5], [2.5, 9.5]]);
    assert_exact("centres", &outline(vec![p]), GRID);
}

/// Curves from pixel corner to pixel corner, bulging across several pixels
/// — each end's pixel read with the band starting on its edge.
#[test]
fn curves_from_corner_to_corner() {
    let c = vec![
        quad([2.0, 2.0], [10.0, 2.0], [10.0, 10.0]),
        quad([10.0, 10.0], [2.0, 10.0], [2.0, 2.0]),
    ];
    assert_exact("lens", &outline(vec![c]), GRID);
}

// ───────────────────────────── tiny and tall ───────────────────────────────

/// A square with one side cut into pieces from `10⁻⁵` down to under the
/// drop length (`10⁻⁶`), each ending where the next starts.
#[test]
fn tiny_pieces() {
    let mut side = Vec::new();
    let mut y = 2.0f32;
    for step in [1e-5f32, 2e-6, 5e-7, 1e-7, 3e-6] {
        side.push(line([8.25, y], [8.25, y + step]));
        y += step;
    }
    side.push(line([8.25, y], [8.25, 9.75]));
    let mut contour = vec![line([2.25, 2.0], [8.25, 2.0])];
    contour.extend(side);
    contour.push(line([8.25, 9.75], [2.25, 9.75]));
    contour.push(line([2.25, 9.75], [2.25, 2.0]));
    assert_exact("tiny pieces", &outline(vec![contour]), GRID);
}

/// A slanted stroke and a curve spanning sixty rows, so one piece's band
/// covers every row of the lattice and its cut is read at each.
#[test]
fn pieces_spanning_many_rows() {
    let grid = Grid {
        width: 8,
        height: 64,
    };
    let stroke = polygon(&[[0.3, 0.5], [3.7, 60.5], [5.0, 60.5], [1.6, 0.5]]);
    assert_exact("stroke", &outline(vec![stroke]), grid);
    let d = vec![
        quad([1.25, 1.5], [9.0, 31.0], [1.25, 62.25]),
        line([1.25, 62.25], [1.25, 1.5]),
    ];
    assert_exact("tall D", &outline(vec![d]), grid);
}

// ─────────────────────────── contours together ─────────────────────────────

/// Two triangles meeting at one vertex — a pixel corner, then a point
/// inside a pixel — and a bowtie whose two lobes wind opposite ways.
#[test]
fn contours_sharing_a_vertex() {
    for v in [[6.0, 6.0], [6.3, 5.7]] {
        let a = polygon(&[[1.0, 1.5], [v[0], v[1]], [1.5, 10.0]]);
        let b = polygon(&[[v[0], v[1]], [10.5, 1.0], [11.0, 10.5]]);
        assert_exact("touching triangles", &outline(vec![a, b]), GRID);
    }
    let bowtie = polygon(&[[1.0, 1.0], [11.0, 11.0], [11.0, 1.0], [1.0, 11.0]]);
    assert_exact("bowtie", &outline(vec![bowtie]), GRID);
}

/// Two squares sharing an edge inside a pixel column — no seam — then
/// overlapping (winding 2, clamped), then a hole wound the other way.
#[test]
fn shared_edges_overlaps_and_holes() {
    let left = rect([1.5, 2.25], [5.3, 9.5]);
    let right = rect([5.3, 2.25], [10.5, 9.5]);
    assert_exact("shared edge", &outline(vec![left, right]), GRID);
    let a = rect([1.5, 1.5], [7.3, 7.3]);
    let b = rect([4.7, 4.7], [10.5, 10.5]);
    assert_exact("overlap", &outline(vec![a, b]), GRID);
    let outer = rect([1.25, 1.25], [10.75, 10.75]);
    let hole: Vec<Segment> = polygon(&[[3.4, 3.6], [3.4, 8.2], [8.1, 8.2], [8.1, 3.6]]);
    assert_exact("hole", &outline(vec![outer, hole]), GRID);
}

// ───────────────────────────── odd quadratics ──────────────────────────────

/// A quadratic whose control is one of its ends is a straight line run at
/// a varying speed; one whose ends coincide retraces itself and encloses
/// nothing. Both must draw exactly what the straight square draws.
#[test]
fn degenerate_quadratics() {
    let square = rect([2.3, 2.6], [9.4, 8.8]);
    let want = exact(&outline(vec![square]), GRID, [0.0, 0.0]);
    let control_on_start = vec![
        quad([2.3, 2.6], [2.3, 2.6], [9.4, 2.6]),
        quad([9.4, 2.6], [9.4, 8.8], [9.4, 8.8]),
        quad([9.4, 8.8], [9.4, 8.8], [2.3, 8.8]),
        quad([2.3, 8.8], [2.3, 2.6], [2.3, 2.6]),
    ];
    let slanted = vec![
        quad([2.3, 2.6], [9.4, 2.6], [9.4, 8.8]),
        line([9.4, 8.8], [2.3, 2.6]),
    ];
    let mut with_a_loop = rect([2.3, 2.6], [9.4, 8.8]);
    with_a_loop.insert(1, quad([9.4, 2.6], [11.0, 7.0], [9.4, 2.6]));
    assert_exact(
        "control on an end",
        &outline(vec![control_on_start.clone()]),
        GRID,
    );
    assert_exact("slanted half", &outline(vec![slanted]), GRID);
    assert_exact(
        "a retracing loop",
        &outline(vec![with_a_loop.clone()]),
        GRID,
    );
    // The judge agrees with itself, to its own rounding: the three shapes
    // are one square.
    for (name, c) in [
        ("control on an end", control_on_start),
        ("loop", with_a_loop),
    ] {
        let got = exact(&outline(vec![c]), GRID, [0.0, 0.0]);
        for (k, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() < 1e-12,
                "{name}: the reference reads texel {k} as {g}, the square as {w}"
            );
        }
    }
}

/// Quadratics that turn back on one axis, on both, and nearly at one
/// parameter on both (a near-cusp) — cut by `MonotoneQuad` into pieces
/// that meet with a flat tangent.
#[test]
fn quadratics_that_turn_back() {
    let one_axis = vec![
        quad([3.0, 1.5], [14.0, 6.0], [3.0, 10.5]),
        line([3.0, 10.5], [3.0, 1.5]),
    ];
    let both_axes = vec![
        quad([2.0, 2.0], [12.0, 13.0], [3.0, 1.0]),
        line([3.0, 1.0], [2.0, 2.0]),
    ];
    let near_cusp = vec![
        quad([1.5, 1.5], [10.5, 10.5], [1.5001, 1.5002]),
        line([1.5001, 1.5002], [1.5, 1.5]),
    ];
    assert_exact("one axis", &outline(vec![one_axis]), GRID);
    assert_exact("both axes", &outline(vec![both_axes]), GRID);
    assert_exact("near cusp", &outline(vec![near_cusp]), GRID);
}

// ─────────────────────────── past the lattice ──────────────────────────────

/// Ink reaching past the lattice on every side: pieces left of it (which no
/// texel sees), right of it (whose rise every texel of the row inherits),
/// above and below it.
#[test]
fn ink_past_the_lattice() {
    let big = vec![
        quad([-3.5, -2.25], [6.0, -9.0], [15.75, -2.25]),
        line([15.75, -2.25], [15.75, 14.5]),
        quad([15.75, 14.5], [6.0, 20.0], [-3.5, 14.5]),
        line([-3.5, 14.5], [-3.5, -2.25]),
    ];
    assert_exact("past every side", &outline(vec![big]), GRID);
    let across_the_right = rect([9.3, 2.5], [20.0, 9.5]);
    assert_exact(
        "across the right edge",
        &outline(vec![across_the_right]),
        GRID,
    );
    let across_the_left = rect([-5.0, 3.2], [2.6, 8.7]);
    assert_exact(
        "across the left edge",
        &outline(vec![across_the_left]),
        GRID,
    );
    let wholly_right = rect([13.0, 2.5], [20.0, 9.5]);
    assert_exact("wholly right of it", &outline(vec![wholly_right]), GRID);
}

// ──────────────────────────────── far away ─────────────────────────────────

/// A run of text puts its pieces at pen positions far from the origin,
/// where a coordinate's ulp is `2.4e-4` at 2048 — the bound grows with
/// `|X| + |Y|` to allow for it, and the closed form, which subtracts before
/// it multiplies, uses a sliver of it.
#[test]
fn far_from_the_origin() {
    for [ox, oy] in [[1024.0f32, 0.0], [2048.0, 512.0], [16384.0, 0.0]] {
        let shape = vec![
            quad(
                [ox + 1.3, oy + 1.6],
                [ox + 13.0, oy + 5.5],
                [ox + 2.1, oy + 10.4],
            ),
            line([ox + 2.1, oy + 10.4], [ox + 1.3, oy + 1.6]),
        ];
        let worst = assert_exact_at("far", &outline(vec![shape]), GRID, [ox, oy]);
        eprintln!("offset ({ox}, {oy}): the worst texel uses {worst:.4} of the bound");
    }
}

// ─────────────────────────── the font, past ASCII ───────────────────────────

/// Font glyphs the ASCII ratchet (`glyph_exact_area.rs`, 7 / 16 / 32 px)
/// does not reach, at the sizes it does not bake: compounds (`é`, `Ç`),
/// glyphs whose quadratics turn back and are cut by `MonotoneQuad` (`ƕ`,
/// `ȡ`, `ʓ`, `ʚ` — no ASCII glyph has such a curve), and the font's largest
/// outline (`⌨`, 347 segments, a trip count of 512). Each is baked with a
/// tile of margin on every side, so ink past its advance box is judged too.
#[test]
fn font_glyphs_past_ascii_at_5_and_64_px() {
    let font = Font::parse(FONT_DATA).expect("parse font");
    for size in [5usize, 64] {
        for ch in ['é', 'Ç', 'ƕ', 'ȡ', 'ʓ', 'ʚ', '⌨'] {
            let glyph = font
                .glyph_kernel_scaled(ch, size as f32)
                .unwrap_or_else(|| panic!("the font has no glyph for {ch:?}"));
            let margin = size as f32;
            let side = 3 * size;
            let grid = Grid {
                width: side,
                height: side,
            };
            let at_centres = glyph.kernel().at(
                &Kernel::x().add(&Kernel::constant(0.5 - margin)),
                &Kernel::y().add(&Kernel::constant(0.5 - margin)),
            );
            let ours = glyph
                .bake(&at_centres, Lattice::frame(side, side))
                .into_buffer();
            let moved: Vec<Piece> = screen_pieces(&font, ch, size as f64)
                .into_iter()
                .map(|p| translated(p, f64::from(margin)))
                .collect();
            let exact: Vec<f64> = signed_area(&moved, grid)
                .into_iter()
                .map(coverage)
                .collect();
            let [x0, y0, x1, y1] = glyph.support.bounds();
            let judged = Judged {
                name: &format!("{ch:?}@{size}"),
                grid,
                origin: [-f64::from(margin); 2],
                extent: f64::from((x1 - x0).max(y1 - y0)),
            };
            judged.assert(&ours, &exact);
        }
    }
}

fn translated(p: Piece, d: f64) -> Piece {
    let m = |[x, y]: [f64; 2]| [x + d, y + d];
    match p {
        Piece::Line(a, b) => Piece::Line(m(a), m(b)),
        Piece::Quad(a, c, b) => Piece::Quad(m(a), m(c), m(b)),
    }
}

// ───────────────────────────── random contours ─────────────────────────────

/// xorshift64*: deterministic and dependency-free.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// A coordinate in `[0.5, 11.5]`: on the quarter-pixel grid for
    /// `quantized`, else anywhere.
    fn coordinate(&mut self, quantized: bool) -> f32 {
        let u = (self.next() >> 40) as f32 / (1u64 << 24) as f32;
        let v = 0.5 + 11.0 * u;
        match quantized {
            true => (v * 4.0).round() / 4.0,
            false => v,
        }
    }
}

/// Random closed contours — two per outline, three to six segments each,
/// lines and quadratics mixed — on the quarter-pixel grid, where ends land
/// on pixel edges, corners and centres, and horizontal and vertical pieces
/// are common; then off it.
#[test]
fn random_contours_on_and_off_the_quarter_grid() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for quantized in [true, false] {
        for case in 0..60 {
            let contours = (0..2)
                .map(|_| {
                    let n = 3 + (rng.next() % 4) as usize;
                    let mut p = |_| [rng.coordinate(quantized), rng.coordinate(quantized)];
                    let corners: Vec<P> = (0..n).map(&mut p).collect();
                    (0..n)
                        .map(|k| {
                            let (a, b) = (corners[k], corners[(k + 1) % n]);
                            match rng.next() % 2 {
                                0 => line(a, b),
                                _ => quad(
                                    a,
                                    [rng.coordinate(quantized), rng.coordinate(quantized)],
                                    b,
                                ),
                            }
                        })
                        .collect()
                })
                .collect();
            let name = format!("random case {case}, quantized {quantized}");
            assert_exact(&name, &outline(contours), GRID);
        }
    }
}
