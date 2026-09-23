//! **Today's glyph coverage against the exact area under each texel.**
//!
//! `tests/common/exact_area.rs` computes, in `f64` and from nothing but the
//! outline's control points, the area of each texel the glyph covers —
//! `min(|∫∫_texel w|, 1)` for the non-zero winding number `w`, FreeType's
//! coverage (see that module for why the clamp is the accepted meaning where
//! contours overlap). This file measures the shipped renderer against it,
//! through the path a frame draws from: [`GlyphAtlas`] tiles, baked at
//! texel centres.
//!
//! ## The statistic
//!
//! Per `(glyph, size)`, with `Δ = ours − exact` over the tile:
//!
//! - `E_max = max |Δ|`,
//! - `E_mean = Σ |Δ| / #{exact > 0}` — the error per texel the exact glyph
//!   inks ("> 0" meaning above the reference's resolution, [`INKED`]),
//! - `N₀.₁ = #{|Δ| > 0.1}` — texels wrong by more than a tenth of a pixel.
//!
//! `E_mean`'s denominator is the reference's, not the union of both sides'
//! ink. So every statistic can only rise when a texel gets worse. Over the
//! union, a renderer that laid faint ink around every glyph would enlarge
//! the denominator, and read as better while every tile got worse. A NaN
//! texel reads as an infinite error.
//!
//! Every printable ASCII glyph of the crate's font, at 7, 16 and 32 px.
//!
//! ## The baseline: the exact area, closed by the compiler
//!
//! Coverage is the area of the pixel under ink, written as a formula and
//! closed by the e-graph (`fonts/loop_blinn.rs`,
//! docs/plans/2026-09-23-a-glyph-is-a-formula.md). Measured 2026-09-23 on
//! the AVX-512 and AVX2 tiers, which agree to every printed digit (the
//! per-glyph table is [`BASELINE`], and
//! `docs/results/2026-09-23-glyph-is-a-formula.md`):
//!
//! | size | inked glyphs | inked texels | max `E_max` | mean `E_mean` | `Σ N₀.₁` | mean centroid shift (x, y) |
//! |---|---|---|---|---|---|---|
//! | 7 px | 94 | 1322 | 0.00089 (`G`) | 0.000007 | 0 | (−0.00003, +0.000002) px |
//! | 16 px | 94 | 4166 | 0.00095 (`9`) | 0.000003 | 0 | (+0.000002, −0.000003) px |
//! | 32 px | 94 | 12529 | 0.00095 (`Z`) | 0.000005 | 0 | (−0.000003, +0.000008) px |
//!
//! ("mean `E_mean`" averages the per-glyph means.) What is left is the snap
//! (`loop_blinn::COVERAGE_SNAP`, `2⁻¹⁰ ≈ 0.00098`): a texel whose exact
//! area is a sliver under `2⁻¹⁰` reads `0`, one within `2⁻¹⁰` of full reads
//! `1`. 105 of the 285 rows are exact to the sixth decimal.
//!
//! It was a one-sided ramp on the distance to the nearest edge
//! (`docs/results/2026-09-23-glyph-exact-area-baseline.md`), measured on the
//! SSE2 build before that tier was deleted: max `E_max` 0.357 (`t`), 0.424
//! (`h`) and 0.427 (`)`), mean `E_mean` 0.0763, 0.0288 and 0.0130, and
//! `Σ N₀.₁` 405, 308 and 370 at 7, 16 and 32 px — the *model's* error: a
//! corner, where two edges each cut the texel, and a thin stem, where both
//! sides do, read a single distance where the area needs two.
//!
//! ## The gate: a ratchet
//!
//! [`todays_renderer_is_no_worse_than_its_baseline`] fails if any
//! `(glyph, size)` gets **worse** than its row — `E_max` or `E_mean` above the
//! recorded value, or more texels past `0.1` — by more than
//! [`PLATFORM_NOISE`]. Getting better passes silently; a change that improves
//! the model re-pins the rows it moved, with its reasons, in its own commit.
//!
//! ## One pixel, on both sides
//!
//! Texel `(i, j)` integrates `[i, i+1) × [j, j+1)` in the frame the atlas
//! bakes in — the screen frame `Font::glyph_kernel_scaled` builds (ascent at
//! `y = 0`, descent at `y = size`), sampled at `(i + ½, j + ½)` by the atlas's
//! own contramap. The reference restates that frame from the font's metrics
//! rather than borrowing the renderer's map, so an offset between the two
//! would be a systematic error, not a shared one. Two checks say there is
//! none: an integer-aligned square, where both sides must read exactly 0 or
//! 1 ([`the_renderer_and_the_reference_share_a_pixel`]), and the ink-weighted
//! centroid of every glyph (the table's last column; `O`, `0`, `o`, `H` and
//! `I` are within 0.02 px at 32 px, where a half-texel slip would read 0.5).
//! No glyph's exact ink falls outside its tile at any of the three sizes.

#[path = "common/exact_area.rs"]
mod exact_area;

use exact_area::{coverage, screen_pieces, signed_area, Grid, Piece, Point};
use pixelflow_core::{Kernel, Lattice};
use pixelflow_graphics::fonts::{loop_blinn, Contour, Font, GlyphAtlas, Outline, Segment};
use std::sync::OnceLock;

const FONT_DATA: &[u8] = include_bytes!("../assets/DejaVuSansMono-Fallback.ttf");

/// The sizes, in pixels per ascent-to-descent height — the atlas's `tile_px`.
const SIZES: [usize; 3] = [7, 16, 32];

/// The threshold of `N₀.₁`: a texel wrong by more than a tenth of a pixel.
const BAD_TEXEL: f64 = 0.1;

/// Exact coverage above this is ink. The reference is good to 1e-8 of a
/// texel (`exact_area`'s module docs), and the rises of a closed row cancel
/// only to rounding, leaving up to 5e-15 in texels no contour reaches.
const INKED: f64 = 1e-8;

/// How far a statistic may move between two builds of the same renderer.
///
/// The JIT's arithmetic differs by target — `MulAdd` may round once or
/// twice, `Recip` is an estimate (CLAUDE.md's platform table) — so the
/// renderer's texels need not be the same bits everywhere. Measured for the
/// exact-area renderer: the AVX2 and AVX-512 tiers agree on every row to
/// the printed digit, and the atlas they bake is bit-identical
/// (`glyph_atlas_golden.rs`). aarch64 was not measured.
///
/// What sets the slack is the snap, not the arithmetic. A texel whose exact
/// area sits within a rounding of `COVERAGE_SNAP` (`2⁻¹⁰`) snaps on one
/// target and not on another, and moves its row's `E_max` by up to `2⁻¹⁰`
/// either way. `10⁻³` is that and no more: any change of model worth the
/// name moves a row by far more.
const PLATFORM_NOISE: f64 = 1e-3;

/// A ratchet row: `(glyph, size, E_max, E_mean, N₀.₁)`.
type Row = (char, usize, f64, f64, u32);

/// One `(glyph, size)` measurement.
#[derive(Clone, Copy, Debug, Default)]
struct Stat {
    e_max: f64,
    e_mean: f64,
    /// `#{|Δ| > 0.1}`.
    n_bad: u32,
    /// `#{|Δ| > 0.1 + PLATFORM_NOISE}` — the count the ratchet compares, so
    /// a texel sitting on the threshold cannot flip it by a rounding.
    n_bad_beyond_noise: u32,
    /// Texels the exact glyph inks: `#{exact > INKED}`, the denominator of
    /// `e_mean`.
    inked: u32,
    /// `ours − exact` of the ink-weighted centroid, `[x, y]`, in texels.
    centroid_shift: [f64; 2],
}

/// `ours` against `exact`, both row-major over a `width`-wide tile.
fn measure(ours: &[f64], exact: &[f64], width: usize) -> Stat {
    let mut stat = Stat::default();
    let mut sum = 0.0;
    let mut moments = [[0.0f64; 3]; 2];
    for (k, (&o, &e)) in ours.iter().zip(exact).enumerate() {
        let (x, y) = ((k % width) as f64 + 0.5, (k / width) as f64 + 0.5);
        for (m, v) in moments.iter_mut().zip([o, e]) {
            m[0] += v;
            m[1] += v * x;
            m[2] += v * y;
        }
        // A NaN is not a coverage. It must read as the worst error there is,
        // not vanish from a `max` and every comparison.
        let d = match (o - e).abs() {
            d if d.is_nan() => f64::INFINITY,
            d => d,
        };
        stat.inked += u32::from(e > INKED);
        sum += d;
        stat.e_max = stat.e_max.max(d);
        stat.n_bad += u32::from(d > BAD_TEXEL);
        stat.n_bad_beyond_noise += u32::from(d > BAD_TEXEL + PLATFORM_NOISE);
    }
    stat.e_mean = sum / f64::from(stat.inked.max(1));
    let [ours_m, exact_m] = moments;
    // Only the reference's ink is tested: a renderer that draws nothing, or
    // NaN, has no centroid, and must fail the check rather than skip it.
    if exact_m[0] > 0.0 {
        stat.centroid_shift = [
            ours_m[1] / ours_m[0] - exact_m[1] / exact_m[0],
            ours_m[2] / ours_m[0] - exact_m[2] / exact_m[0],
        ];
    }
    stat
}

/// Every printable ASCII glyph at every size, measured once per process.
fn measurements() -> &'static [(char, usize, Stat)] {
    static MEASURED: OnceLock<Vec<(char, usize, Stat)>> = OnceLock::new();
    MEASURED.get_or_init(|| {
        let font = Font::parse(FONT_DATA).expect("parse font");
        let mut out = Vec::new();
        for size in SIZES {
            let mut atlas = GlyphAtlas::new(size as f32, 1.0, 128);
            assert_eq!(atlas.tile_px(), size, "a tile is one texel per pixel");
            atlas.warm(&font, ' '..='~');
            let buffer = atlas.buffer();
            for ch in ' '..='~' {
                let (u, v) = atlas.uv(&font, ch);
                let (u, v) = (u as usize, v as usize);
                let ours: Vec<f64> = (0..size * size)
                    .map(|k| f64::from(buffer[(v + k / size) * atlas.width() + u + k % size]))
                    .collect();
                let pieces = screen_pieces(&font, ch, size as f64);
                let tile = Grid {
                    width: size,
                    height: size,
                };
                let exact: Vec<f64> = signed_area(&pieces, tile)
                    .into_iter()
                    .map(coverage)
                    .collect();
                assert_no_ink_outside_the_tile(&pieces, &exact, size, ch);
                out.push((ch, size, measure(&ours, &exact, size)));
            }
        }
        out
    })
}

/// The statistic is over the tile, so ink the tile clips would be error
/// nobody measures. None of this font's ASCII glyphs has any.
fn assert_no_ink_outside_the_tile(pieces: &[Piece], exact_in_tile: &[f64], size: usize, ch: char) {
    let pad = size as f64;
    let moved: Vec<Piece> = pieces.iter().map(|&p| translated(p, [pad, pad])).collect();
    let around = Grid {
        width: 3 * size,
        height: 3 * size,
    };
    let everywhere: f64 = signed_area(&moved, around).into_iter().map(coverage).sum();
    let inside: f64 = exact_in_tile.iter().sum();
    assert!(
        (everywhere - inside).abs() < 1e-9,
        "{ch:?}@{size}: {} texels of exact ink fall outside the atlas tile",
        everywhere - inside
    );
}

fn translated(p: Piece, [dx, dy]: Point) -> Piece {
    let m = |[x, y]: Point| [x + dx, y + dy];
    match p {
        Piece::Line(a, b) => Piece::Line(m(a), m(b)),
        Piece::Quad(a, c, b) => Piece::Quad(m(a), m(c), m(b)),
    }
}

fn reversed(pieces: &[Piece]) -> Vec<Piece> {
    pieces
        .iter()
        .rev()
        .map(|&p| match p {
            Piece::Line(a, b) => Piece::Line(b, a),
            Piece::Quad(a, c, b) => Piece::Quad(b, c, a),
        })
        .collect()
}

/// A closed polygon through `points`.
fn polygon(points: &[Point]) -> Vec<Piece> {
    (0..points.len())
        .map(|k| Piece::Line(points[k], points[(k + 1) % points.len()]))
        .collect()
}

// ─────────────────────────── the reference itself ───────────────────────────

/// A rectangle's texels are its overlap with each texel — computed here by
/// interval arithmetic, not by the reference's accumulation — in both
/// orientations, the second negated.
#[test]
fn a_rectangle_with_fractional_edges_covers_its_overlap() {
    let (x, y) = ([1.25, 3.5], [0.75, 2.0]);
    let grid = Grid {
        width: 5,
        height: 3,
    };
    let rect = polygon(&[[x[0], y[0]], [x[1], y[0]], [x[1], y[1]], [x[0], y[1]]]);
    let overlap = |lo: f64, hi: f64, k: f64| (hi.min(k + 1.0) - lo.max(k)).max(0.0);
    for (pieces, sign) in [(rect.clone(), 1.0), (reversed(&rect), -1.0)] {
        let f = signed_area(&pieces, grid);
        for j in 0..grid.height {
            for i in 0..grid.width {
                let want = overlap(x[0], x[1], i as f64) * overlap(y[0], y[1], j as f64);
                let got = f[j * grid.width + i];
                assert!(
                    (got.abs() - want).abs() < 1e-12 && (want == 0.0 || got.signum() == sign),
                    "texel ({i},{j}): {got}, want {}",
                    sign * want
                );
            }
        }
    }
}

/// Archimedes: the region between a parabola's arc and its chord is two
/// thirds of the control triangle. Summed over every texel, the reference
/// must hold exactly that — at one scale where every piece of arc spans a
/// few texels, and at a tenfold one where the halving goes deeper.
#[test]
fn a_parabolic_segment_holds_two_thirds_of_its_control_triangle() {
    let (p0, p1, p2) = ([0.37, 0.61], [4.83, 1.29], [1.14, 5.02]);
    for scale in [1.0, 10.0] {
        let s = |[x, y]: Point| [x * scale, y * scale];
        let segment = [Piece::Quad(s(p0), s(p1), s(p2)), Piece::Line(s(p2), s(p0))];
        let side = (6.0 * scale) as usize;
        let grid = Grid {
            width: side,
            height: side,
        };
        let total: f64 = signed_area(&segment, grid).iter().sum();
        let triangle =
            0.5 * ((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]));
        let want = 2.0 / 3.0 * triangle.abs() * scale * scale;
        assert!(
            (total.abs() - want).abs() < 1e-9 * want,
            "scale {scale}: the segment holds {total}, Archimedes says {want}"
        );
    }
}

/// The winding number at `(sx, sy)` by a ray to `+x`: `+1` for an edge
/// crossing it toward `+y`. Quadratics are intersected by solving
/// `y(t) = sy` — a different algorithm from the reference's, on purpose.
fn winding(pieces: &[Piece], [sx, sy]: Point) -> i32 {
    let mut w = 0;
    for &p in pieces {
        let [p0, p1, p2] = match p {
            Piece::Line(a, b) => [a, [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])], b],
            Piece::Quad(a, c, b) => [a, c, b],
        };
        let (a, b, c) = (
            p0[1] - 2.0 * p1[1] + p2[1],
            2.0 * (p1[1] - p0[1]),
            p0[1] - sy,
        );
        let roots: Vec<f64> = match a.abs() < 1e-12 {
            true => vec![-c / b],
            false => {
                let disc = b * b - 4.0 * a * c;
                match disc < 0.0 {
                    true => vec![],
                    false => vec![
                        (-b - disc.sqrt()) / (2.0 * a),
                        (-b + disc.sqrt()) / (2.0 * a),
                    ],
                }
            }
        };
        for t in roots.into_iter().filter(|t| (0.0..1.0).contains(t)) {
            let x = (1.0 - t) * (1.0 - t) * p0[0] + 2.0 * t * (1.0 - t) * p1[0] + t * t * p2[0];
            let dy = 2.0 * a * t + b;
            if x > sx && dy != 0.0 {
                w += if dy > 0.0 { 1 } else { -1 };
            }
        }
    }
    w
}

/// Against the winding number point-sampled on a 128 × 128 grid per texel,
/// over a curved contour with a hole wound the other way. Point sampling
/// errs by at most one sample column per boundary crossing per sample row,
/// so four crossings bound the error by `4 / 128`; the worst measured is
/// 5.4e-4, and `2 / 128` is asserted.
#[test]
fn the_reference_agrees_with_point_sampling() {
    const SAMPLES: usize = 128;
    let d = [
        Piece::Quad([1.3172, 0.4117], [6.9031, 2.6123], [1.2091, 5.6263]),
        Piece::Line([1.2091, 5.6263], [1.3172, 0.4117]),
    ];
    // A triangle inside the `D`, traversed so it winds against it.
    let hole = polygon(&[[1.7137, 1.9291], [2.3419, 3.8173], [3.1057, 2.2039]]);
    let shoelace = |points: &[Point]| -> f64 {
        (0..points.len())
            .map(|k| {
                let (a, b) = (points[k], points[(k + 1) % points.len()]);
                a[0] * b[1] - b[0] * a[1]
            })
            .sum()
    };
    let d_winds = signed_area(
        &d,
        Grid {
            width: 7,
            height: 7,
        },
    )
    .iter()
    .sum::<f64>();
    let hole_winds = shoelace(&[[1.7137, 1.9291], [2.3419, 3.8173], [3.1057, 2.2039]]);
    let hole = match d_winds.signum() == hole_winds.signum() {
        true => reversed(&hole),
        false => hole,
    };
    let shape: Vec<Piece> = d.iter().copied().chain(hole).collect();
    let grid = Grid {
        width: 5,
        height: 7,
    };
    let f = signed_area(&shape, grid);
    let mut worst = 0.0f64;
    for j in 0..grid.height {
        for i in 0..grid.width {
            let mut sum = 0i64;
            for sj in 0..SAMPLES {
                for si in 0..SAMPLES {
                    let p = [
                        i as f64 + (si as f64 + 0.5) / SAMPLES as f64,
                        j as f64 + (sj as f64 + 0.5) / SAMPLES as f64,
                    ];
                    sum += i64::from(winding(&shape, p));
                }
            }
            let sampled = sum as f64 / (SAMPLES * SAMPLES) as f64;
            worst = worst.max((f[j * grid.width + i] - sampled).abs());
        }
    }
    assert!(
        worst <= 2.0 / SAMPLES as f64,
        "the reference and point sampling differ by {worst} in some texel"
    );
}

/// Winding is odd in orientation: reversing every contour negates every
/// texel, which a sign slip anywhere in the accumulation would break.
#[test]
fn reversing_every_contour_negates_every_texel() {
    let font = Font::parse(FONT_DATA).expect("parse font");
    let pieces = screen_pieces(&font, '8', 32.0);
    let grid = Grid {
        width: 32,
        height: 32,
    };
    let forward = signed_area(&pieces, grid);
    let backward = signed_area(&reversed(&pieces), grid);
    for (k, (f, b)) in forward.iter().zip(&backward).enumerate() {
        assert!(
            (f + b).abs() < 1e-12,
            "texel {k}: {f} forward, {b} reversed"
        );
    }
    assert!(forward.iter().any(|&f| f.abs() > 0.5), "'8' drew nothing");
}

// ─────────────────────────── the renderer against it ─────────────────────────

/// An integer-aligned square: every texel is wholly in or wholly out, so
/// both sides must read exactly 0 or 1 — the renderer too, since its
/// coverage is the area and snaps its ends. A half-texel slip between the
/// two conventions would read ½ along every edge.
#[test]
fn the_renderer_and_the_reference_share_a_pixel() {
    let corners = [[2.0f32, 3.0], [6.0, 3.0], [6.0, 7.0], [2.0, 7.0]];
    let contour = Contour::new(
        (0..4)
            .map(|k| Segment::Line {
                from: corners[k],
                to: corners[(k + 1) % 4],
            })
            .collect(),
    )
    .expect("a square closes");
    let outline = Outline {
        contours: vec![contour],
    };
    let (width, height) = (9, 10);
    let glyph = loop_blinn::glyph(&outline);
    let centred = glyph.kernel().at(
        &Kernel::x().add(&Kernel::constant(0.5)),
        &Kernel::y().add(&Kernel::constant(0.5)),
    );
    let ours = glyph
        .bake(&centred, Lattice::frame(width, height))
        .into_buffer();
    let exact = signed_area(
        &exact_area::pieces(&outline, |[x, y]| [f64::from(x), f64::from(y)]),
        Grid { width, height },
    );
    for (k, (&o, &e)) in ours.iter().zip(&exact).enumerate() {
        assert!(
            e == 0.0 || e == 1.0,
            "texel {k}: the square's exact area is {e}"
        );
        assert!(
            f64::from(o) == e,
            "texel ({}, {}): renderer {o}, exact {e}",
            k % width,
            k / width
        );
    }
}

/// The ratchet. See the module docs.
#[test]
fn todays_renderer_is_no_worse_than_its_baseline() {
    let measured = measurements();
    let mut worse = Vec::new();
    let mut table = String::new();
    for &(ch, size, stat) in measured {
        table.push_str(&format!(
            "    ({ch:?}, {size}, {:.6}, {:.6}, {}),\n",
            stat.e_max, stat.e_mean, stat.n_bad
        ));
        let Some(&(_, _, e_max, e_mean, n_bad)) =
            BASELINE.iter().find(|row| row.0 == ch && row.1 == size)
        else {
            worse.push(format!("{ch:?}@{size}: no baseline row"));
            continue;
        };
        if stat.e_max > e_max + PLATFORM_NOISE {
            worse.push(format!("{ch:?}@{size}: E_max {:.6} > {e_max}", stat.e_max));
        }
        if stat.e_mean > e_mean + PLATFORM_NOISE {
            worse.push(format!(
                "{ch:?}@{size}: E_mean {:.6} > {e_mean}",
                stat.e_mean
            ));
        }
        if stat.n_bad_beyond_noise > n_bad {
            worse.push(format!(
                "{ch:?}@{size}: {} texels past {BAD_TEXEL} > {n_bad}",
                stat.n_bad_beyond_noise
            ));
        }
    }
    eprintln!("measured (glyph, size, E_max, E_mean, N₀.₁):\n{table}");
    assert!(
        worse.is_empty(),
        "coverage got worse against the exact area:\n{}",
        worse.join("\n")
    );
}

/// The ink-weighted centroid of every glyph, ours against the exact one.
/// A convention slip between the renderer's frame and the reference's —
/// the ascent line, the flip, the half-texel centre — moves every glyph the
/// same way; the model's own error does not. Measured with coverage the
/// exact area: the mean shift is at most 3e-5 px on either axis at any
/// size, and no glyph's exceeds 0.001 px (it was 0.013 and 0.26 px with the
/// distance ramp, and these bounds were 0.05 and 0.35). The bounds sit
/// thirty and ten times above the measurement: a half-texel slip reads 0.5.
#[test]
fn the_renderer_and_the_reference_agree_on_where_the_ink_is() {
    const SYSTEMATIC: f64 = 0.001;
    const ANY_GLYPH: f64 = 0.01;
    for size in SIZES {
        let inked: Vec<Stat> = measurements()
            .iter()
            .filter(|(_, s, stat)| *s == size && stat.inked > 0)
            .map(|&(_, _, stat)| stat)
            .collect();
        let mean = [0, 1].map(|axis| {
            inked.iter().map(|s| s.centroid_shift[axis]).sum::<f64>() / inked.len() as f64
        });
        let worst = inked
            .iter()
            .map(|s| s.centroid_shift[0].abs().max(s.centroid_shift[1].abs()))
            .fold(0.0, f64::max);
        eprintln!("{size} px: mean centroid shift {mean:?}, worst glyph {worst}");
        for (axis, mean) in mean.into_iter().enumerate() {
            assert!(
                mean.abs() <= SYSTEMATIC,
                "{size} px: ink sits {mean} px off the reference's on axis {axis}, on average"
            );
        }
        for (ch, s, stat) in measurements() {
            let shift = stat.centroid_shift[0]
                .abs()
                .max(stat.centroid_shift[1].abs());
            assert!(
                *s != size || shift <= ANY_GLYPH,
                "{ch:?}@{size}: ink sits {shift} px off the reference's"
            );
        }
    }
}

/// `(glyph, size, E_max, E_mean, N₀.₁)` for the exact-area renderer,
/// measured 2026-09-23 on the AVX-512 and AVX2 tiers (see the module docs).
/// A ratchet, not a pin: a row may be beaten, never exceeded.
const BASELINE: [Row; 285] = [
    (' ', 7, 0.000000, 0.000000, 0),
    ('!', 7, 0.000000, 0.000000, 0),
    ('"', 7, 0.000000, 0.000000, 0),
    ('#', 7, 0.000000, 0.000000, 0),
    ('$', 7, 0.000000, 0.000000, 0),
    ('%', 7, 0.000000, 0.000000, 0),
    ('&', 7, 0.000000, 0.000000, 0),
    ('\'', 7, 0.000000, 0.000000, 0),
    ('(', 7, 0.000073, 0.000007, 0),
    (')', 7, 0.000178, 0.000016, 0),
    ('*', 7, 0.000090, 0.000008, 0),
    ('+', 7, 0.000000, 0.000000, 0),
    (',', 7, 0.000000, 0.000000, 0),
    ('-', 7, 0.000000, 0.000000, 0),
    ('.', 7, 0.000000, 0.000000, 0),
    ('/', 7, 0.000000, 0.000000, 0),
    ('0', 7, 0.000570, 0.000030, 0),
    ('1', 7, 0.000000, 0.000000, 0),
    ('2', 7, 0.000059, 0.000004, 0),
    ('3', 7, 0.000524, 0.000031, 0),
    ('4', 7, 0.000000, 0.000000, 0),
    ('5', 7, 0.000000, 0.000000, 0),
    ('6', 7, 0.000000, 0.000000, 0),
    ('7', 7, 0.000474, 0.000042, 0),
    ('8', 7, 0.000000, 0.000000, 0),
    ('9', 7, 0.000492, 0.000026, 0),
    (':', 7, 0.000000, 0.000000, 0),
    (';', 7, 0.000000, 0.000000, 0),
    ('<', 7, 0.000767, 0.000064, 0),
    ('=', 7, 0.000000, 0.000000, 0),
    ('>', 7, 0.000000, 0.000000, 0),
    ('?', 7, 0.000005, 0.000000, 0),
    ('@', 7, 0.000000, 0.000000, 0),
    ('A', 7, 0.000000, 0.000000, 0),
    ('B', 7, 0.000000, 0.000000, 0),
    ('C', 7, 0.000000, 0.000000, 0),
    ('D', 7, 0.000000, 0.000000, 0),
    ('E', 7, 0.000462, 0.000026, 0),
    ('F', 7, 0.000874, 0.000055, 0),
    ('G', 7, 0.000886, 0.000049, 0),
    ('H', 7, 0.000000, 0.000000, 0),
    ('I', 7, 0.000000, 0.000000, 0),
    ('J', 7, 0.000000, 0.000000, 0),
    ('K', 7, 0.000000, 0.000000, 0),
    ('L', 7, 0.000000, 0.000000, 0),
    ('M', 7, 0.000000, 0.000000, 0),
    ('N', 7, 0.000000, 0.000000, 0),
    ('O', 7, 0.000000, 0.000000, 0),
    ('P', 7, 0.000000, 0.000000, 0),
    ('Q', 7, 0.000000, 0.000000, 0),
    ('R', 7, 0.000220, 0.000011, 0),
    ('S', 7, 0.000000, 0.000000, 0),
    ('T', 7, 0.000000, 0.000000, 0),
    ('U', 7, 0.000000, 0.000000, 0),
    ('V', 7, 0.000266, 0.000018, 0),
    ('W', 7, 0.000000, 0.000000, 0),
    ('X', 7, 0.000000, 0.000000, 0),
    ('Y', 7, 0.000000, 0.000000, 0),
    ('Z', 7, 0.000000, 0.000000, 0),
    ('[', 7, 0.000000, 0.000000, 0),
    ('\\', 7, 0.000000, 0.000000, 0),
    (']', 7, 0.000000, 0.000000, 0),
    ('^', 7, 0.000000, 0.000000, 0),
    ('_', 7, 0.000000, 0.000000, 0),
    ('`', 7, 0.000000, 0.000000, 0),
    ('a', 7, 0.000000, 0.000000, 0),
    ('b', 7, 0.000000, 0.000000, 0),
    ('c', 7, 0.000627, 0.000042, 0),
    ('d', 7, 0.000000, 0.000000, 0),
    ('e', 7, 0.000000, 0.000000, 0),
    ('f', 7, 0.000000, 0.000000, 0),
    ('g', 7, 0.000000, 0.000000, 0),
    ('h', 7, 0.000000, 0.000000, 0),
    ('i', 7, 0.000000, 0.000000, 0),
    ('j', 7, 0.000000, 0.000000, 0),
    ('k', 7, 0.000000, 0.000000, 0),
    ('l', 7, 0.000000, 0.000000, 0),
    ('m', 7, 0.000000, 0.000000, 0),
    ('n', 7, 0.000000, 0.000000, 0),
    ('o', 7, 0.000018, 0.000001, 0),
    ('p', 7, 0.000000, 0.000000, 0),
    ('q', 7, 0.000000, 0.000000, 0),
    ('r', 7, 0.000000, 0.000000, 0),
    ('s', 7, 0.000000, 0.000000, 0),
    ('t', 7, 0.000000, 0.000000, 0),
    ('u', 7, 0.000000, 0.000000, 0),
    ('v', 7, 0.000000, 0.000000, 0),
    ('w', 7, 0.000000, 0.000000, 0),
    ('x', 7, 0.000000, 0.000000, 0),
    ('y', 7, 0.000733, 0.000049, 0),
    ('z', 7, 0.000000, 0.000000, 0),
    ('{', 7, 0.000012, 0.000001, 0),
    ('|', 7, 0.000822, 0.000059, 0),
    ('}', 7, 0.000000, 0.000000, 0),
    ('~', 7, 0.000825, 0.000103, 0),
    (' ', 16, 0.000000, 0.000000, 0),
    ('!', 16, 0.000000, 0.000000, 0),
    ('"', 16, 0.000001, 0.000000, 0),
    ('#', 16, 0.000256, 0.000004, 0),
    ('$', 16, 0.000001, 0.000000, 0),
    ('%', 16, 0.000563, 0.000010, 0),
    ('&', 16, 0.000481, 0.000010, 0),
    ('\'', 16, 0.000000, 0.000000, 0),
    ('(', 16, 0.000250, 0.000013, 0),
    (')', 16, 0.000324, 0.000016, 0),
    ('*', 16, 0.000314, 0.000009, 0),
    ('+', 16, 0.000000, 0.000000, 0),
    (',', 16, 0.000000, 0.000000, 0),
    ('-', 16, 0.000001, 0.000000, 0),
    ('.', 16, 0.000000, 0.000000, 0),
    ('/', 16, 0.000747, 0.000021, 0),
    ('0', 16, 0.000447, 0.000012, 0),
    ('1', 16, 0.000001, 0.000000, 0),
    ('2', 16, 0.000665, 0.000014, 0),
    ('3', 16, 0.000000, 0.000000, 0),
    ('4', 16, 0.000000, 0.000000, 0),
    ('5', 16, 0.000001, 0.000000, 0),
    ('6', 16, 0.000000, 0.000000, 0),
    ('7', 16, 0.000436, 0.000011, 0),
    ('8', 16, 0.000539, 0.000008, 0),
    ('9', 16, 0.000951, 0.000016, 0),
    (':', 16, 0.000000, 0.000000, 0),
    (';', 16, 0.000000, 0.000000, 0),
    ('<', 16, 0.000152, 0.000005, 0),
    ('=', 16, 0.000001, 0.000000, 0),
    ('>', 16, 0.000000, 0.000000, 0),
    ('?', 16, 0.000001, 0.000000, 0),
    ('@', 16, 0.000000, 0.000000, 0),
    ('A', 16, 0.000204, 0.000007, 0),
    ('B', 16, 0.000818, 0.000018, 0),
    ('C', 16, 0.000313, 0.000007, 0),
    ('D', 16, 0.000001, 0.000000, 0),
    ('E', 16, 0.000001, 0.000000, 0),
    ('F', 16, 0.000001, 0.000000, 0),
    ('G', 16, 0.000001, 0.000000, 0),
    ('H', 16, 0.000001, 0.000000, 0),
    ('I', 16, 0.000001, 0.000000, 0),
    ('J', 16, 0.000072, 0.000002, 0),
    ('K', 16, 0.000693, 0.000011, 0),
    ('L', 16, 0.000000, 0.000000, 0),
    ('M', 16, 0.000001, 0.000000, 0),
    ('N', 16, 0.000001, 0.000000, 0),
    ('O', 16, 0.000884, 0.000014, 0),
    ('P', 16, 0.000001, 0.000000, 0),
    ('Q', 16, 0.000663, 0.000010, 0),
    ('R', 16, 0.000138, 0.000002, 0),
    ('S', 16, 0.000793, 0.000014, 0),
    ('T', 16, 0.000001, 0.000000, 0),
    ('U', 16, 0.000000, 0.000000, 0),
    ('V', 16, 0.000001, 0.000000, 0),
    ('W', 16, 0.000001, 0.000000, 0),
    ('X', 16, 0.000780, 0.000014, 0),
    ('Y', 16, 0.000619, 0.000023, 0),
    ('Z', 16, 0.000001, 0.000000, 0),
    ('[', 16, 0.000001, 0.000000, 0),
    ('\\', 16, 0.000553, 0.000016, 0),
    (']', 16, 0.000001, 0.000000, 0),
    ('^', 16, 0.000000, 0.000000, 0),
    ('_', 16, 0.000000, 0.000000, 0),
    ('`', 16, 0.000000, 0.000000, 0),
    ('a', 16, 0.000011, 0.000000, 0),
    ('b', 16, 0.000000, 0.000000, 0),
    ('c', 16, 0.000000, 0.000000, 0),
    ('d', 16, 0.000001, 0.000000, 0),
    ('e', 16, 0.000001, 0.000000, 0),
    ('f', 16, 0.000000, 0.000000, 0),
    ('g', 16, 0.000474, 0.000008, 0),
    ('h', 16, 0.000000, 0.000000, 0),
    ('i', 16, 0.000001, 0.000000, 0),
    ('j', 16, 0.000001, 0.000000, 0),
    ('k', 16, 0.000000, 0.000000, 0),
    ('l', 16, 0.000000, 0.000000, 0),
    ('m', 16, 0.000001, 0.000000, 0),
    ('n', 16, 0.000000, 0.000000, 0),
    ('o', 16, 0.000818, 0.000016, 0),
    ('p', 16, 0.000000, 0.000000, 0),
    ('q', 16, 0.000238, 0.000004, 0),
    ('r', 16, 0.000000, 0.000000, 0),
    ('s', 16, 0.000001, 0.000000, 0),
    ('t', 16, 0.000000, 0.000000, 0),
    ('u', 16, 0.000001, 0.000000, 0),
    ('v', 16, 0.000001, 0.000000, 0),
    ('w', 16, 0.000001, 0.000000, 0),
    ('x', 16, 0.000001, 0.000000, 0),
    ('y', 16, 0.000021, 0.000001, 0),
    ('z', 16, 0.000001, 0.000000, 0),
    ('{', 16, 0.000001, 0.000000, 0),
    ('|', 16, 0.000000, 0.000000, 0),
    ('}', 16, 0.000001, 0.000000, 0),
    ('~', 16, 0.000001, 0.000000, 0),
    (' ', 32, 0.000000, 0.000000, 0),
    ('!', 32, 0.000001, 0.000000, 0),
    ('"', 32, 0.000001, 0.000000, 0),
    ('#', 32, 0.000819, 0.000004, 0),
    ('$', 32, 0.000823, 0.000005, 0),
    ('%', 32, 0.000874, 0.000006, 0),
    ('&', 32, 0.000600, 0.000004, 0),
    ('\'', 32, 0.000001, 0.000000, 0),
    ('(', 32, 0.000784, 0.000010, 0),
    (')', 32, 0.000753, 0.000012, 0),
    ('*', 32, 0.000612, 0.000007, 0),
    ('+', 32, 0.000001, 0.000000, 0),
    (',', 32, 0.000918, 0.000024, 0),
    ('-', 32, 0.000001, 0.000000, 0),
    ('.', 32, 0.000001, 0.000000, 0),
    ('/', 32, 0.000152, 0.000002, 0),
    ('0', 32, 0.000935, 0.000024, 0),
    ('1', 32, 0.000001, 0.000000, 0),
    ('2', 32, 0.000320, 0.000003, 0),
    ('3', 32, 0.000766, 0.000015, 0),
    ('4', 32, 0.000860, 0.000007, 0),
    ('5', 32, 0.000393, 0.000003, 0),
    ('6', 32, 0.000678, 0.000011, 0),
    ('7', 32, 0.000790, 0.000008, 0),
    ('8', 32, 0.000828, 0.000009, 0),
    ('9', 32, 0.000217, 0.000003, 0),
    (':', 32, 0.000001, 0.000000, 0),
    (';', 32, 0.000918, 0.000016, 0),
    ('<', 32, 0.000608, 0.000009, 0),
    ('=', 32, 0.000001, 0.000000, 0),
    ('>', 32, 0.000007, 0.000000, 0),
    ('?', 32, 0.000001, 0.000000, 0),
    ('@', 32, 0.000196, 0.000001, 0),
    ('A', 32, 0.000847, 0.000015, 0),
    ('B', 32, 0.000319, 0.000002, 0),
    ('C', 32, 0.000210, 0.000004, 0),
    ('D', 32, 0.000695, 0.000004, 0),
    ('E', 32, 0.000001, 0.000000, 0),
    ('F', 32, 0.000001, 0.000000, 0),
    ('G', 32, 0.000173, 0.000001, 0),
    ('H', 32, 0.000001, 0.000000, 0),
    ('I', 32, 0.000002, 0.000000, 0),
    ('J', 32, 0.000288, 0.000003, 0),
    ('K', 32, 0.000645, 0.000004, 0),
    ('L', 32, 0.000001, 0.000000, 0),
    ('M', 32, 0.000946, 0.000005, 0),
    ('N', 32, 0.000934, 0.000009, 0),
    ('O', 32, 0.000205, 0.000002, 0),
    ('P', 32, 0.000002, 0.000000, 0),
    ('Q', 32, 0.000774, 0.000006, 0),
    ('R', 32, 0.000553, 0.000006, 0),
    ('S', 32, 0.000823, 0.000007, 0),
    ('T', 32, 0.000002, 0.000000, 0),
    ('U', 32, 0.000607, 0.000004, 0),
    ('V', 32, 0.000492, 0.000007, 0),
    ('W', 32, 0.000100, 0.000001, 0),
    ('X', 32, 0.000662, 0.000005, 0),
    ('Y', 32, 0.000230, 0.000004, 0),
    ('Z', 32, 0.000953, 0.000008, 0),
    ('[', 32, 0.000001, 0.000000, 0),
    ('\\', 32, 0.000450, 0.000011, 0),
    (']', 32, 0.000001, 0.000000, 0),
    ('^', 32, 0.000688, 0.000021, 0),
    ('_', 32, 0.000001, 0.000000, 0),
    ('`', 32, 0.000137, 0.000006, 0),
    ('a', 32, 0.000592, 0.000009, 0),
    ('b', 32, 0.000681, 0.000004, 0),
    ('c', 32, 0.000504, 0.000006, 0),
    ('d', 32, 0.000945, 0.000013, 0),
    ('e', 32, 0.000812, 0.000008, 0),
    ('f', 32, 0.000335, 0.000003, 0),
    ('g', 32, 0.000027, 0.000000, 0),
    ('h', 32, 0.000001, 0.000000, 0),
    ('i', 32, 0.000001, 0.000000, 0),
    ('j', 32, 0.000001, 0.000000, 0),
    ('k', 32, 0.000535, 0.000007, 0),
    ('l', 32, 0.000001, 0.000000, 0),
    ('m', 32, 0.000001, 0.000000, 0),
    ('n', 32, 0.000001, 0.000000, 0),
    ('o', 32, 0.000732, 0.000007, 0),
    ('p', 32, 0.000001, 0.000000, 0),
    ('q', 32, 0.000951, 0.000011, 0),
    ('r', 32, 0.000001, 0.000000, 0),
    ('s', 32, 0.000398, 0.000005, 0),
    ('t', 32, 0.000011, 0.000000, 0),
    ('u', 32, 0.000832, 0.000010, 0),
    ('v', 32, 0.000839, 0.000010, 0),
    ('w', 32, 0.000505, 0.000004, 0),
    ('x', 32, 0.000179, 0.000002, 0),
    ('y', 32, 0.000551, 0.000006, 0),
    ('z', 32, 0.000001, 0.000000, 0),
    ('{', 32, 0.000377, 0.000003, 0),
    ('|', 32, 0.000000, 0.000000, 0),
    ('}', 32, 0.000518, 0.000005, 0),
    ('~', 32, 0.000154, 0.000005, 0),
];
