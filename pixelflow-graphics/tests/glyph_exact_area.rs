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
//! ## The baseline: today's antialiasing model
//!
//! Measured 2026-09-23 on the SSE2 baseline build (the per-glyph table is
//! [`BASELINE`], and `docs/results/2026-09-23-glyph-exact-area-baseline.md`):
//!
//! | size | inked glyphs | inked texels | max `E_max` | mean `E_mean` | `Σ N₀.₁` | mean centroid shift (x, y) |
//! |---|---|---|---|---|---|---|
//! | 7 px | 94 | 1322 | 0.357 (`t`) | 0.0763 | 405 (30.6%) | (−0.006, +0.013) px |
//! | 16 px | 94 | 4166 | 0.424 (`h`) | 0.0288 | 308 (7.4%) | (−0.005, −0.001) px |
//! | 32 px | 94 | 12529 | 0.427 (`)`) | 0.0130 | 370 (3.0%) | (−0.000, −0.003) px |
//!
//! ("mean `E_mean`" averages the per-glyph means.) This is the *model's*
//! error, not arithmetic noise: coverage today is a one-sided ramp on the
//! distance to the nearest edge, so a corner, where two edges each cut the
//! texel, and a thin stem, where both sides do, read a single distance where
//! the area needs two. A glyph is a formula whose terms add
//! (docs/plans/2026-09-23-a-glyph-is-a-formula.md), and that is the change
//! these numbers are recorded to judge.
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
/// The JIT's arithmetic differs by ISA level — `MulAdd` rounds once with FMA
/// and twice without, `Recip` is an estimate (CLAUDE.md's platform table) —
/// so the renderer's texels are not the same bits everywhere. Measured
/// between the SSE2 baseline and a `-C target-cpu=native` (AVX-512) build:
/// `E_max` or `E_mean` moves in 20 of the 285 rows, each time by 1e-6, and
/// no `N₀.₁` moves. `glyph_atlas_golden.rs` measured single texels moving
/// 2.1e-4 between ISA levels. aarch64 was not measured. This is five times
/// the texel figure and a thousandth of a pixel, far below any change of
/// model.
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
/// both sides must read 0 or 1 — the renderer to within its ramp's gradient
/// floor (a distance of ½ reads `½ / (1 + 10⁻³)`). A half-texel slip between
/// the two conventions would read ½ along every edge.
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
            (f64::from(o) - e).abs() < 1e-3,
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
/// same way; the model's own error does not. Measured: the mean shift is at
/// most 0.013 px on either axis at any size, and no glyph's exceeds 0.26 px.
#[test]
fn the_renderer_and_the_reference_agree_on_where_the_ink_is() {
    const SYSTEMATIC: f64 = 0.05;
    const ANY_GLYPH: f64 = 0.35;
    for size in SIZES {
        let inked: Vec<Stat> = measurements()
            .iter()
            .filter(|(_, s, stat)| *s == size && stat.inked > 0)
            .map(|&(_, _, stat)| stat)
            .collect();
        for axis in 0..2 {
            let mean =
                inked.iter().map(|s| s.centroid_shift[axis]).sum::<f64>() / inked.len() as f64;
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

/// `(glyph, size, E_max, E_mean, N₀.₁)` for today's renderer, measured
/// 2026-09-23 (see the module docs). A ratchet, not a pin: a row may be
/// beaten, never exceeded.
const BASELINE: [Row; 285] = [
    (' ', 7, 0.000000, 0.000000, 0),
    ('!', 7, 0.202802, 0.040996, 1),
    ('"', 7, 0.218055, 0.102208, 2),
    ('#', 7, 0.259061, 0.125583, 11),
    ('$', 7, 0.242921, 0.073131, 7),
    ('%', 7, 0.167799, 0.092354, 9),
    ('&', 7, 0.304797, 0.104160, 7),
    ('\'', 7, 0.089078, 0.047552, 0),
    ('(', 7, 0.202576, 0.066258, 3),
    (')', 7, 0.244430, 0.067180, 2),
    ('*', 7, 0.250237, 0.082043, 4),
    ('+', 7, 0.182616, 0.077446, 4),
    (',', 7, 0.209569, 0.099592, 2),
    ('-', 7, 0.130306, 0.056867, 1),
    ('.', 7, 0.235878, 0.129484, 2),
    ('/', 7, 0.110638, 0.043357, 1),
    ('0', 7, 0.174889, 0.056696, 4),
    ('1', 7, 0.136937, 0.050267, 4),
    ('2', 7, 0.234742, 0.077119, 4),
    ('3', 7, 0.221746, 0.078639, 6),
    ('4', 7, 0.238347, 0.113936, 7),
    ('5', 7, 0.283844, 0.098667, 6),
    ('6', 7, 0.201809, 0.060062, 3),
    ('7', 7, 0.296336, 0.091364, 4),
    ('8', 7, 0.167242, 0.064073, 3),
    ('9', 7, 0.169364, 0.061195, 3),
    (':', 7, 0.235878, 0.133588, 4),
    (';', 7, 0.235056, 0.116526, 4),
    ('<', 7, 0.169929, 0.068843, 5),
    ('=', 7, 0.179314, 0.067397, 3),
    ('>', 7, 0.277043, 0.073683, 4),
    ('?', 7, 0.238328, 0.066406, 4),
    ('@', 7, 0.212802, 0.074485, 7),
    ('A', 7, 0.241376, 0.070952, 5),
    ('B', 7, 0.216682, 0.058864, 4),
    ('C', 7, 0.181067, 0.056875, 4),
    ('D', 7, 0.234112, 0.058772, 3),
    ('E', 7, 0.197789, 0.061235, 4),
    ('F', 7, 0.197789, 0.059408, 6),
    ('G', 7, 0.164449, 0.062502, 4),
    ('H', 7, 0.234930, 0.054246, 4),
    ('I', 7, 0.205342, 0.054866, 3),
    ('J', 7, 0.245319, 0.173832, 8),
    ('K', 7, 0.301674, 0.089390, 6),
    ('L', 7, 0.172634, 0.049409, 2),
    ('M', 7, 0.263971, 0.095156, 7),
    ('N', 7, 0.293627, 0.078153, 6),
    ('O', 7, 0.139923, 0.046019, 2),
    ('P', 7, 0.230178, 0.072264, 5),
    ('Q', 7, 0.180498, 0.054402, 4),
    ('R', 7, 0.294543, 0.073460, 7),
    ('S', 7, 0.197002, 0.085388, 6),
    ('T', 7, 0.266675, 0.081186, 4),
    ('U', 7, 0.136069, 0.031005, 2),
    ('V', 7, 0.251044, 0.066875, 4),
    ('W', 7, 0.221797, 0.105955, 10),
    ('X', 7, 0.286881, 0.074948, 5),
    ('Y', 7, 0.219712, 0.064178, 4),
    ('Z', 7, 0.239098, 0.102622, 6),
    ('[', 7, 0.179156, 0.104616, 6),
    ('\\', 7, 0.163109, 0.048271, 2),
    (']', 7, 0.138667, 0.026188, 2),
    ('^', 7, 0.177299, 0.081848, 3),
    ('_', 7, 0.089436, 0.022558, 0),
    ('`', 7, 0.132018, 0.078846, 1),
    ('a', 7, 0.210500, 0.082867, 5),
    ('b', 7, 0.181277, 0.050456, 5),
    ('c', 7, 0.214277, 0.063857, 3),
    ('d', 7, 0.241119, 0.051165, 3),
    ('e', 7, 0.223753, 0.076055, 5),
    ('f', 7, 0.286706, 0.108569, 3),
    ('g', 7, 0.194050, 0.071422, 6),
    ('h', 7, 0.191252, 0.047594, 4),
    ('i', 7, 0.276941, 0.089204, 6),
    ('j', 7, 0.246013, 0.071463, 3),
    ('k', 7, 0.291155, 0.079290, 7),
    ('l', 7, 0.189671, 0.070672, 5),
    ('m', 7, 0.296429, 0.113100, 9),
    ('n', 7, 0.191252, 0.062976, 5),
    ('o', 7, 0.172618, 0.055042, 3),
    ('p', 7, 0.173160, 0.051967, 4),
    ('q', 7, 0.227984, 0.060366, 4),
    ('r', 7, 0.265680, 0.138474, 3),
    ('s', 7, 0.293672, 0.121823, 7),
    ('t', 7, 0.356861, 0.168384, 7),
    ('u', 7, 0.151417, 0.041988, 3),
    ('v', 7, 0.245229, 0.071281, 4),
    ('w', 7, 0.337089, 0.125189, 8),
    ('x', 7, 0.273003, 0.100075, 5),
    ('y', 7, 0.251277, 0.085426, 5),
    ('z', 7, 0.280828, 0.096785, 6),
    ('{', 7, 0.185855, 0.057036, 2),
    ('|', 7, 0.004720, 0.000608, 0),
    ('}', 7, 0.202724, 0.072536, 2),
    ('~', 7, 0.181513, 0.084160, 3),
    (' ', 16, 0.000000, 0.000000, 0),
    ('!', 16, 0.179815, 0.042185, 5),
    ('"', 16, 0.224383, 0.027823, 3),
    ('#', 16, 0.288591, 0.033439, 8),
    ('$', 16, 0.202876, 0.037140, 8),
    ('%', 16, 0.189677, 0.032217, 2),
    ('&', 16, 0.358794, 0.036295, 4),
    ('\'', 16, 0.231570, 0.058036, 3),
    ('(', 16, 0.166927, 0.018553, 2),
    (')', 16, 0.281868, 0.031460, 4),
    ('*', 16, 0.211129, 0.069131, 10),
    ('+', 16, 0.068845, 0.009448, 0),
    (',', 16, 0.114285, 0.026448, 2),
    ('-', 16, 0.216780, 0.069254, 2),
    ('.', 16, 0.173288, 0.047266, 1),
    ('/', 16, 0.101640, 0.018345, 1),
    ('0', 16, 0.198803, 0.022365, 1),
    ('1', 16, 0.195893, 0.023116, 3),
    ('2', 16, 0.252962, 0.028138, 2),
    ('3', 16, 0.267792, 0.027690, 1),
    ('4', 16, 0.268456, 0.042355, 8),
    ('5', 16, 0.225603, 0.025188, 4),
    ('6', 16, 0.154327, 0.021859, 2),
    ('7', 16, 0.183692, 0.019061, 2),
    ('8', 16, 0.335870, 0.036760, 4),
    ('9', 16, 0.187934, 0.025375, 2),
    (':', 16, 0.185240, 0.050275, 3),
    (';', 16, 0.185240, 0.034115, 4),
    ('<', 16, 0.250942, 0.036181, 4),
    ('=', 16, 0.210119, 0.040707, 7),
    ('>', 16, 0.258239, 0.036699, 5),
    ('?', 16, 0.197233, 0.042036, 3),
    ('@', 16, 0.095852, 0.021761, 0),
    ('A', 16, 0.182825, 0.021406, 2),
    ('B', 16, 0.355011, 0.028337, 2),
    ('C', 16, 0.220181, 0.023036, 2),
    ('D', 16, 0.181479, 0.012232, 1),
    ('E', 16, 0.191782, 0.017471, 3),
    ('F', 16, 0.139423, 0.012752, 3),
    ('G', 16, 0.195198, 0.023351, 3),
    ('H', 16, 0.128630, 0.007812, 3),
    ('I', 16, 0.172070, 0.025955, 3),
    ('J', 16, 0.241017, 0.028548, 4),
    ('K', 16, 0.149347, 0.023929, 5),
    ('L', 16, 0.155524, 0.021034, 4),
    ('M', 16, 0.228847, 0.025320, 7),
    ('N', 16, 0.230440, 0.016484, 3),
    ('O', 16, 0.200562, 0.019538, 1),
    ('P', 16, 0.163599, 0.028628, 5),
    ('Q', 16, 0.226232, 0.023053, 2),
    ('R', 16, 0.277510, 0.023238, 4),
    ('S', 16, 0.127148, 0.021517, 1),
    ('T', 16, 0.132078, 0.016544, 2),
    ('U', 16, 0.126855, 0.013091, 2),
    ('V', 16, 0.146128, 0.017463, 3),
    ('W', 16, 0.238409, 0.022947, 9),
    ('X', 16, 0.197293, 0.036815, 6),
    ('Y', 16, 0.142258, 0.025646, 3),
    ('Z', 16, 0.261224, 0.026910, 2),
    ('[', 16, 0.208935, 0.020684, 2),
    ('\\', 16, 0.132376, 0.018004, 1),
    (']', 16, 0.213642, 0.019412, 3),
    ('^', 16, 0.323228, 0.052585, 4),
    ('_', 16, 0.127651, 0.014216, 1),
    ('`', 16, 0.039731, 0.018406, 0),
    ('a', 16, 0.182454, 0.025302, 2),
    ('b', 16, 0.349217, 0.041355, 5),
    ('c', 16, 0.124105, 0.022233, 2),
    ('d', 16, 0.181566, 0.018517, 2),
    ('e', 16, 0.165705, 0.022978, 3),
    ('f', 16, 0.166828, 0.033072, 5),
    ('g', 16, 0.166484, 0.023091, 4),
    ('h', 16, 0.424316, 0.031969, 5),
    ('i', 16, 0.147348, 0.035970, 5),
    ('j', 16, 0.153209, 0.022454, 2),
    ('k', 16, 0.210786, 0.034749, 5),
    ('l', 16, 0.075224, 0.012830, 0),
    ('m', 16, 0.176517, 0.035299, 6),
    ('n', 16, 0.424316, 0.032958, 4),
    ('o', 16, 0.068751, 0.015218, 0),
    ('p', 16, 0.343858, 0.040712, 5),
    ('q', 16, 0.098203, 0.013026, 0),
    ('r', 16, 0.207368, 0.052975, 7),
    ('s', 16, 0.227864, 0.041944, 4),
    ('t', 16, 0.096183, 0.018025, 0),
    ('u', 16, 0.176107, 0.015725, 2),
    ('v', 16, 0.155596, 0.025545, 4),
    ('w', 16, 0.288677, 0.037659, 7),
    ('x', 16, 0.192614, 0.039870, 3),
    ('y', 16, 0.314021, 0.039365, 5),
    ('z', 16, 0.174968, 0.031927, 5),
    ('{', 16, 0.268662, 0.039532, 2),
    ('|', 16, 0.181222, 0.010633, 2),
    ('}', 16, 0.199685, 0.041216, 4),
    ('~', 16, 0.353448, 0.055807, 2),
    (' ', 32, 0.000000, 0.000000, 0),
    ('!', 32, 0.182670, 0.007959, 2),
    ('"', 32, 0.220702, 0.010348, 2),
    ('#', 32, 0.264488, 0.016355, 11),
    ('$', 32, 0.262728, 0.020338, 13),
    ('%', 32, 0.154368, 0.015582, 2),
    ('&', 32, 0.286075, 0.018975, 5),
    ('\'', 32, 0.201555, 0.014536, 1),
    ('(', 32, 0.147594, 0.011022, 1),
    (')', 32, 0.426973, 0.018321, 3),
    ('*', 32, 0.279890, 0.027210, 7),
    ('+', 32, 0.145835, 0.005794, 3),
    (',', 32, 0.168860, 0.016190, 3),
    ('-', 32, 0.026420, 0.002191, 0),
    ('.', 32, 0.210865, 0.022787, 2),
    ('/', 32, 0.285914, 0.013372, 1),
    ('0', 32, 0.055423, 0.008420, 0),
    ('1', 32, 0.249424, 0.011027, 6),
    ('2', 32, 0.128148, 0.012065, 2),
    ('3', 32, 0.247641, 0.012607, 2),
    ('4', 32, 0.284689, 0.016712, 9),
    ('5', 32, 0.220714, 0.009758, 3),
    ('6', 32, 0.314315, 0.012454, 3),
    ('7', 32, 0.294893, 0.013159, 4),
    ('8', 32, 0.135508, 0.011223, 1),
    ('9', 32, 0.322550, 0.010990, 1),
    (':', 32, 0.210865, 0.019777, 4),
    (';', 32, 0.168860, 0.016386, 5),
    ('<', 32, 0.165805, 0.012613, 2),
    ('=', 32, 0.178501, 0.008447, 3),
    ('>', 32, 0.141921, 0.013450, 2),
    ('?', 32, 0.223810, 0.016050, 2),
    ('@', 32, 0.146717, 0.012182, 3),
    ('A', 32, 0.279049, 0.014324, 8),
    ('B', 32, 0.295302, 0.016169, 8),
    ('C', 32, 0.179767, 0.012118, 2),
    ('D', 32, 0.128823, 0.007440, 2),
    ('E', 32, 0.242397, 0.008632, 7),
    ('F', 32, 0.076287, 0.003731, 0),
    ('G', 32, 0.300187, 0.013814, 6),
    ('H', 32, 0.228344, 0.005927, 5),
    ('I', 32, 0.146137, 0.008972, 6),
    ('J', 32, 0.323678, 0.009228, 1),
    ('K', 32, 0.302119, 0.016989, 7),
    ('L', 32, 0.203750, 0.005748, 3),
    ('M', 32, 0.180336, 0.011030, 10),
    ('N', 32, 0.319891, 0.008815, 5),
    ('O', 32, 0.051811, 0.009577, 0),
    ('P', 32, 0.295302, 0.016685, 10),
    ('Q', 32, 0.097910, 0.010305, 0),
    ('R', 32, 0.298798, 0.014058, 6),
    ('S', 32, 0.155903, 0.010229, 1),
    ('T', 32, 0.180336, 0.006107, 3),
    ('U', 32, 0.228344, 0.009752, 4),
    ('V', 32, 0.188581, 0.010582, 5),
    ('W', 32, 0.373860, 0.010316, 7),
    ('X', 32, 0.222818, 0.019637, 7),
    ('Y', 32, 0.307048, 0.015926, 4),
    ('Z', 32, 0.203750, 0.014357, 6),
    ('[', 32, 0.167046, 0.005837, 1),
    ('\\', 32, 0.257283, 0.013836, 1),
    (']', 32, 0.174090, 0.006605, 3),
    ('^', 32, 0.131734, 0.020444, 2),
    ('_', 32, 0.033623, 0.001426, 0),
    ('`', 32, 0.215801, 0.038255, 3),
    ('a', 32, 0.250039, 0.012903, 4),
    ('b', 32, 0.197919, 0.009538, 2),
    ('c', 32, 0.106179, 0.012210, 1),
    ('d', 32, 0.393807, 0.014385, 4),
    ('e', 32, 0.232952, 0.011724, 3),
    ('f', 32, 0.220388, 0.014649, 8),
    ('g', 32, 0.411450, 0.016666, 6),
    ('h', 32, 0.184974, 0.007916, 3),
    ('i', 32, 0.225423, 0.020471, 12),
    ('j', 32, 0.246233, 0.011881, 5),
    ('k', 32, 0.187155, 0.016769, 7),
    ('l', 32, 0.218487, 0.009098, 1),
    ('m', 32, 0.239297, 0.017262, 9),
    ('n', 32, 0.184974, 0.009315, 3),
    ('o', 32, 0.041660, 0.008415, 0),
    ('p', 32, 0.217338, 0.009044, 2),
    ('q', 32, 0.260551, 0.011633, 5),
    ('r', 32, 0.302224, 0.018129, 5),
    ('s', 32, 0.183586, 0.012684, 2),
    ('t', 32, 0.144220, 0.011824, 6),
    ('u', 32, 0.347477, 0.011937, 4),
    ('v', 32, 0.224317, 0.013535, 4),
    ('w', 32, 0.419054, 0.014812, 7),
    ('x', 32, 0.340619, 0.022611, 6),
    ('y', 32, 0.277639, 0.019405, 7),
    ('z', 32, 0.266846, 0.014328, 3),
    ('{', 32, 0.384362, 0.018008, 4),
    ('|', 32, 0.215090, 0.003381, 1),
    ('}', 32, 0.229663, 0.014619, 3),
    ('~', 32, 0.138109, 0.017998, 2),
];
