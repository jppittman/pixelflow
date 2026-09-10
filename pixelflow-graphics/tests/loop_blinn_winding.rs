//! The Loop–Blinn glyph kernel against an independent winding-number oracle.
//!
//! `loop_blinn` computes the winding number of an outline as a sum of
//! per-segment terms relative to a reference point, with Loop–Blinn's
//! implicit for the curved slivers. The oracle here is the *other*
//! formulation — a horizontal ray cast in `f64`, intersecting each quadratic
//! by solving its `y(t)` — written from scratch and sharing no code or
//! constants with the kernel. Where both are defined and away from the
//! antialiasing ramp, they must agree exactly: coverage there is an exact
//! integer, 0 or 1, and a kernel that gets it wrong has the wrong winding,
//! not the wrong rounding.
//!
//! What this suite covers, and what it does not: it pins the *winding*
//! (interior/exterior classification, holes, self-intersections, contour
//! direction, compound placement) and the exactness of the support. The
//! antialiasing ramp's shape is pinned by `font_antialiasing.rs`, and where
//! the ink is against a second rasterizer by `freetype_oracle.rs`.

use std::ops::RangeInclusive;

use pixelflow_core::{Kernel, Lattice};
use pixelflow_graphics::fonts::{loop_blinn, Contour, Font, Outline, Segment};

const FONT_DATA: &[u8] = include_bytes!("../assets/DejaVuSansMono-Fallback.ttf");

/// The rasterizer's pixel-centre convention.
const CENTER: f32 = 0.5;

/// Samples closer than this to the outline are in the antialiasing ramp
/// (half a pixel wide on each side) and are not the oracle's to judge. A
/// little past the ramp's half-pixel, for the fast `sqrt`'s estimate error.
const RAMP_CLEARANCE: f64 = 0.75;

/// Pieces each curve is cut into when measuring distance to the outline.
const DISTANCE_PIECES: usize = 64;

// ────────────────────────────── the oracle ──────────────────────────────

fn wide([x, y]: [f32; 2]) -> [f64; 2] {
    [f64::from(x), f64::from(y)]
}

/// The winding number at `p` by a horizontal ray to `+x`, half-open in `y`
/// so a vertex on the ray counts once; quadratics are intersected by solving
/// `y(t) = p.y` exactly.
fn winding(outline: &Outline, [px, py]: [f64; 2]) -> i32 {
    let mut w = 0i32;
    for seg in outline.segments() {
        match seg {
            Segment::Line { from, to } => {
                let (a, b) = (wide(from), wide(to));
                w += crossing_sign(a, b, [px, py]);
            }
            Segment::Quad { from, control, to } => {
                let (p0, p1, p2) = (wide(from), wide(control), wide(to));
                // y(t) = ay t² + by t + cy − py = 0
                let ay = p0[1] - 2.0 * p1[1] + p2[1];
                let by = 2.0 * (p1[1] - p0[1]);
                let cy = p0[1] - py;
                let roots: Vec<f64> = if ay.abs() < 1e-12 {
                    if by.abs() < 1e-12 {
                        vec![]
                    } else {
                        vec![-cy / by]
                    }
                } else {
                    let disc = by * by - 4.0 * ay * cy;
                    if disc < 0.0 {
                        vec![]
                    } else {
                        let s = disc.sqrt();
                        vec![(-by - s) / (2.0 * ay), (-by + s) / (2.0 * ay)]
                    }
                };
                for t in roots {
                    // Half-open in t, matching the half-open y rule at the
                    // endpoints: a root at t = 1 is the next segment's t = 0.
                    if !(0.0..1.0).contains(&t) {
                        continue;
                    }
                    let x =
                        (1.0 - t) * (1.0 - t) * p0[0] + 2.0 * (1.0 - t) * t * p1[0] + t * t * p2[0];
                    let dy = 2.0 * ay * t + by;
                    if x > px && dy != 0.0 {
                        w += if dy > 0.0 { 1 } else { -1 };
                    }
                }
            }
        }
    }
    w
}

fn crossing_sign(a: [f64; 2], b: [f64; 2], [px, py]: [f64; 2]) -> i32 {
    if (a[1] <= py) == (b[1] <= py) {
        return 0;
    }
    let t = (py - a[1]) / (b[1] - a[1]);
    let x = a[0] + t * (b[0] - a[0]);
    if x > px {
        if b[1] > a[1] {
            1
        } else {
            -1
        }
    } else {
        0
    }
}

/// The outline as one polyline, flattened once. Built per outline rather
/// than per sample: rebuilding it inside the sample loop made this harness
/// quadratic enough to blow nextest's ten-minute cap on a debug build,
/// which is the build CI runs.
fn polyline(outline: &Outline) -> Vec<[[f64; 2]; 2]> {
    let mut out = Vec::new();
    for seg in outline.segments() {
        match seg {
            Segment::Line { from, to } => out.push([wide(from), wide(to)]),
            Segment::Quad { from, control, to } => {
                let (p0, p1, p2) = (wide(from), wide(control), wide(to));
                let at = |t: f64| {
                    let (a, b, c) = ((1.0 - t) * (1.0 - t), 2.0 * (1.0 - t) * t, t * t);
                    [
                        a * p0[0] + b * p1[0] + c * p2[0],
                        a * p0[1] + b * p1[1] + c * p2[1],
                    ]
                };
                for i in 0..DISTANCE_PIECES {
                    let (t0, t1) = (
                        i as f64 / DISTANCE_PIECES as f64,
                        (i + 1) as f64 / DISTANCE_PIECES as f64,
                    );
                    out.push([at(t0), at(t1)]);
                }
            }
        }
    }
    out
}

/// Distance from `p` to a flattened outline.
fn distance(polyline: &[[[f64; 2]; 2]], p: [f64; 2]) -> f64 {
    polyline
        .iter()
        .map(|[a, b]| segment_distance(*a, *b, p))
        .fold(f64::INFINITY, f64::min)
}

fn segment_distance(a: [f64; 2], b: [f64; 2], p: [f64; 2]) -> f64 {
    let (ex, ey) = (b[0] - a[0], b[1] - a[1]);
    let len2 = ex * ex + ey * ey;
    let t = if len2 == 0.0 {
        0.0
    } else {
        (((p[0] - a[0]) * ex + (p[1] - a[1]) * ey) / len2).clamp(0.0, 1.0)
    };
    let (qx, qy) = (a[0] + t * ex, a[1] + t * ey);
    ((p[0] - qx).powi(2) + (p[1] - qy).powi(2)).sqrt()
}

/// The oracle's coverage: `min(|w|, 1)`.
fn oracle(outline: &Outline, p: [f64; 2]) -> f32 {
    winding(outline, p).unsigned_abs().min(1) as f32
}

// ────────────────────────────── harness ──────────────────────────────

fn pixel_centered(k: &Kernel) -> Kernel {
    k.at(
        &Kernel::x().add(&Kernel::constant(CENTER)),
        &Kernel::y().add(&Kernel::constant(CENTER)),
    )
}

/// Bake the single-kernel form over `lattice` at pixel centres.
///
/// `glyph`'s winding sum is a `Kernel::sum_over` reading a bound piece
/// table (S1a), so this must bind it before collapsing.
fn bake_single(outline: &Outline, lattice: Lattice) -> Vec<f32> {
    let glyph = loop_blinn::glyph(outline);
    let kernel = pixel_centered(&glyph.kernel());
    glyph.bake(&kernel, lattice).into_buffer()
}

/// Compare a baked buffer against the oracle at every sample clear of the
/// ramp; returns (samples judged, samples in the ramp). Panics on the first
/// disagreement with the offending sample.
fn judge(label: &str, outline: &Outline, lattice: Lattice, baked: &[f32]) -> (usize, usize) {
    let [w, h] = lattice.extent.map(|e| e as usize);
    let (mut judged, mut in_ramp) = (0usize, 0usize);
    let flattened = polyline(outline);
    let mut failures = Vec::new();
    for j in 0..h {
        for i in 0..w {
            let p = [i as f64 + f64::from(CENTER), j as f64 + f64::from(CENTER)];
            let got = baked[j * w + i];
            assert!(
                got.is_finite(),
                "{label}: non-finite coverage {got} at ({i},{j})"
            );
            assert!(
                (0.0..=1.0).contains(&got),
                "{label}: coverage {got} out of range at ({i},{j})"
            );
            if distance(&flattened, p) < RAMP_CLEARANCE {
                in_ramp += 1;
                continue;
            }
            judged += 1;
            let want = oracle(outline, p);
            // `!=` rather than bit comparison: a saturated sum is an exact
            // integer, and the optimizer is free to hand back `-0.0` for it.
            if got != want {
                failures.push(format!("({i},{j}): kernel {got} oracle {want}"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{label}: {} of {judged} samples clear of the ramp disagree with the oracle:\n{}",
        failures.len(),
        failures.join("\n")
    );
    (judged, in_ramp)
}

fn square(x0: f32, y0: f32, x1: f32, y1: f32, clockwise: bool) -> Contour {
    let mut pts = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
    if clockwise {
        pts.reverse();
    }
    let segments = (0..4)
        .map(|i| Segment::Line {
            from: pts[i],
            to: pts[(i + 1) % 4],
        })
        .collect();
    Contour::new(segments).expect("a square's own corners close the loop")
}

/// A circle of radius `r` about `c` from four quadratics through the
/// diagonal control points — the curved case with every tangent direction.
fn ring(c: [f32; 2], r: f32, clockwise: bool) -> Contour {
    let k = r; // control points at the corners of the bounding square
    let mut on = [
        [c[0] + r, c[1]],
        [c[0], c[1] + r],
        [c[0] - r, c[1]],
        [c[0], c[1] - r],
    ];
    let mut off = [
        [c[0] + k, c[1] + k],
        [c[0] - k, c[1] + k],
        [c[0] - k, c[1] - k],
        [c[0] + k, c[1] - k],
    ];
    if clockwise {
        on.reverse();
        off.reverse();
        off.rotate_right(1);
    }
    let segments = (0..4)
        .map(|i| Segment::Quad {
            from: on[i],
            control: off[i],
            to: on[(i + 1) % 4],
        })
        .collect();
    Contour::new(segments).expect("a ring's own on-curve points close the loop")
}

fn outline_of(contours: Vec<Contour>) -> Outline {
    Outline { contours }
}

// ────────────────────────────── synthetic outlines ──────────────────────────────

#[test]
fn synthetic_outlines_wind_like_the_oracle() {
    let lattice = Lattice::frame(48, 48);
    let cases: Vec<(&str, Outline)> = vec![
        (
            "square",
            outline_of(vec![square(8.0, 8.0, 40.0, 40.0, false)]),
        ),
        (
            "square clockwise",
            outline_of(vec![square(8.0, 8.0, 40.0, 40.0, true)]),
        ),
        (
            "square with a hole",
            outline_of(vec![
                square(4.0, 4.0, 44.0, 44.0, false),
                square(16.0, 16.0, 32.0, 32.0, true),
            ]),
        ),
        (
            "nested same direction: winding 2 is still covered",
            outline_of(vec![
                square(4.0, 4.0, 44.0, 44.0, false),
                square(16.0, 16.0, 32.0, 32.0, false),
            ]),
        ),
        (
            "bow tie",
            outline_of(vec![Contour::new(vec![
                Segment::Line {
                    from: [4.0, 4.0],
                    to: [44.0, 44.0],
                },
                Segment::Line {
                    from: [44.0, 44.0],
                    to: [4.0, 44.0],
                },
                Segment::Line {
                    from: [4.0, 44.0],
                    to: [44.0, 4.0],
                },
                Segment::Line {
                    from: [44.0, 4.0],
                    to: [4.0, 4.0],
                },
            ])
            .expect("the bow tie's own corners close the loop")]),
        ),
        (
            "sharp triangle",
            outline_of(vec![Contour::new(vec![
                Segment::Line {
                    from: [4.0, 4.0],
                    to: [44.0, 22.0],
                },
                Segment::Line {
                    from: [44.0, 22.0],
                    to: [4.0, 30.0],
                },
                Segment::Line {
                    from: [4.0, 30.0],
                    to: [4.0, 4.0],
                },
            ])
            .expect("the triangle's own corners close the loop")]),
        ),
        ("ring", outline_of(vec![ring([24.0, 24.0], 18.0, false)])),
        (
            "ring clockwise",
            outline_of(vec![ring([24.0, 24.0], 18.0, true)]),
        ),
        (
            "annulus",
            outline_of(vec![
                ring([24.0, 24.0], 20.0, false),
                ring([24.0, 24.0], 9.0, true),
            ]),
        ),
        (
            "concave bulges: a square pinched by curves",
            outline_of(vec![Contour::new(vec![
                Segment::Quad {
                    from: [4.0, 4.0],
                    control: [24.0, 20.0],
                    to: [44.0, 4.0],
                },
                Segment::Quad {
                    from: [44.0, 4.0],
                    control: [28.0, 24.0],
                    to: [44.0, 44.0],
                },
                Segment::Quad {
                    from: [44.0, 44.0],
                    control: [24.0, 28.0],
                    to: [4.0, 44.0],
                },
                Segment::Quad {
                    from: [4.0, 44.0],
                    control: [20.0, 24.0],
                    to: [4.0, 4.0],
                },
            ])
            .expect("the pinched square's own corners close the loop")]),
        ),
    ];
    for (name, outline) in &cases {
        let single = bake_single(outline, lattice);
        let (judged, _) = judge(&format!("{name} (single)"), outline, lattice, &single);
        assert!(judged > 1000, "{name}: only {judged} samples judged");
    }
}

// ────────────────────────────── real glyphs ──────────────────────────────

const GLYPHS: [char; 12] = ['O', '8', 'A', 'g', 'W', '@', '%', '{', 'f', 'e', 'S', '&'];
const SIZES: [f32; 3] = [7.0, 13.0, 24.0];

/// Large enough that a support computed a pixel too tight leaves the ink
/// visibly outside the box rather than inside the ramp's own slop.
const SUPPORT_SWEEP_PX: f32 = 48.0;

/// **Why these suites come in bands.** A production bake is a full
/// runtime-tier saturation — seconds per glyph in the debug build CI runs —
/// so a sweep over the printable range is a hundred of those, and one such
/// sweep walked into nextest's ten-minute cap. That cap is **per test**, and
/// each test is its own process (`.config/nextest.toml`), so cutting a sweep
/// into bands puts every glyph through the same compile it always did while
/// no single process carries the whole font. On a multi-core runner the bands
/// also run at once, so the suite gets faster rather than merely legal.
///
/// The cuts are where ASCII's own are: `' '..='?'` is punctuation and the
/// digits, `'@'..='_'` the capitals, `'`'..='~'` the lowercase — each about a
/// third of the range, and each a third of the compiles.
fn winds_like_the_oracle(band: RangeInclusive<char>) {
    const SIZE: f32 = 20.0;
    const EXTENT: usize = 30;

    let font = Font::parse(FONT_DATA).expect("font");
    let lattice = Lattice::frame(EXTENT, EXTENT);
    for ch in band {
        let id = font.cmap_lookup(ch).expect("glyph");
        let outline = font.outline_scaled_by_id(id, SIZE).expect("outline");
        let single = bake_single(&outline, lattice);
        judge(&format!("{ch:?}@{SIZE}"), &outline, lattice, &single);
    }
}

/// The single-kernel form of real glyphs: every contour shape a font
/// produces — holes, counters, sharp joins, tangent joins, slivers.
///
/// One test per size, for the reason in [`winds_like_the_oracle`].
fn glyphs_wind_like_the_oracle_at(size: f32) {
    let font = Font::parse(FONT_DATA).expect("font");
    let n = (size * 1.5).ceil() as usize;
    let lattice = Lattice::frame(n, n);
    for ch in GLYPHS {
        let id = font.cmap_lookup(ch).expect("glyph");
        let outline = font.outline_scaled_by_id(id, size).expect("outline");
        let single = bake_single(&outline, lattice);
        let (judged, _) = judge(&format!("{ch}@{size} (single)"), &outline, lattice, &single);
        // Scale-free, and stronger than a total: a glyph whose every sample
        // fell inside the ramp was carried by this suite without the oracle
        // ever having been asked about it.
        assert!(judged > 0, "{ch}@{size}: every sample was in the ramp");
    }
}

#[test]
fn glyphs_wind_like_the_oracle_at_7px() {
    glyphs_wind_like_the_oracle_at(SIZES[0]);
}

#[test]
fn glyphs_wind_like_the_oracle_at_13px() {
    glyphs_wind_like_the_oracle_at(SIZES[1]);
}

#[test]
fn glyphs_wind_like_the_oracle_at_24px() {
    glyphs_wind_like_the_oracle_at(SIZES[2]);
}

/// Every printable glyph, one size: the whole font's parser and every
/// contour shape it produces, against the oracle.
#[test]
fn punctuation_and_digits_wind_like_the_oracle() {
    winds_like_the_oracle(' '..='?');
}

#[test]
fn uppercase_winds_like_the_oracle() {
    winds_like_the_oracle('@'..='_');
}

#[test]
fn lowercase_winds_like_the_oracle() {
    winds_like_the_oracle('`'..='~');
}

// ──────────────────────── boundaries, not edges ────────────────────────

/// **An edge a font draws is not necessarily a boundary.** A stroke built
/// from two overlapping contours, or two overlapping components of a
/// compound glyph, puts drawn edges *inside* the ink — winding nonzero on
/// both sides. Coverage must stay saturated across them.
///
/// This is the case the ramp-clearance rule in [`judge`] cannot see: it
/// skips every sample within `RAMP_CLEARANCE` of any drawn segment, and an
/// interior edge is a drawn segment, so the seam lives exactly in the
/// region the oracle declines to judge. Measured before the fix: 0.500 at
/// each interior edge, a half-dark line one pixel wide through solid ink.
#[test]
fn an_edge_inside_the_ink_is_not_a_boundary() {
    let overlap = outline_of(vec![
        square(0.0, 0.0, 24.0, 30.0, false),
        square(16.0, 0.0, 40.0, 30.0, false),
    ]);
    // Same shape wound the other way, and one contour reversed: the union
    // is identical, so the interior edges must stay invisible either way.
    let reversed = outline_of(vec![
        square(0.0, 0.0, 24.0, 30.0, true),
        square(16.0, 0.0, 40.0, 30.0, true),
    ]);
    let mixed = outline_of(vec![
        square(0.0, 0.0, 24.0, 30.0, false),
        square(16.0, 0.0, 40.0, 30.0, true),
    ]);

    for (name, outline) in [("same direction", &overlap), ("both reversed", &reversed)] {
        let lattice = Lattice::frame(44, 34);
        let baked = bake_single(outline, lattice);
        let at = |x: usize, y: usize| baked[y * 44 + x];
        for x in [16usize, 24] {
            let v = at(x, 15);
            assert!(
                v > 0.99,
                "{name}: the drawn edge at x={x} is inside the ink (winding is \
                 nonzero on both sides), but coverage there is {v} — a seam \
                 through solid fill"
            );
        }
        // The outer boundary is still a boundary.
        assert!(at(2, 15) > 0.99, "{name}: interior of the left square");
        assert!(at(42, 15) < 0.01, "{name}: outside the right square");
    }

    // The rule is "ask the winding", not "ignore every shared edge": wind the
    // two squares oppositely and the overlap becomes a hole, which has to be
    // empty rather than filled.
    let lattice = Lattice::frame(44, 34);
    let baked = bake_single(&mixed, lattice);
    let at = |x: usize, y: usize| baked[y * 44 + x];
    assert!(
        at(20, 15) < 0.01,
        "opposite directions: the overlap is a hole, got {} at its centre",
        at(20, 15)
    );
    assert!(at(8, 15) > 0.99, "opposite directions: left square is ink");
    assert!(
        at(32, 15) > 0.99,
        "opposite directions: right square is ink"
    );
}

// ────────────────────────────── support ──────────────────────────────

/// A two-pixel ring just outside the support box, and far away.
fn probes_outside(support: loop_blinn::Support) -> Vec<[f32; 2]> {
    let [x0, y0, x1, y1] = support.bounds();
    (0..12)
        .flat_map(|i| {
            let t = i as f32 / 12.0;
            let (w, h) = (x1 - x0 + 4.0, y1 - y0 + 4.0);
            [
                [x0 - 2.0 + t * w, y0 - 1.0],
                [x0 - 2.0 + t * w, y1 + 1.0],
                [x0 - 1.0, y0 - 2.0 + t * h],
                [x1 + 1.0, y0 - 2.0 + t * h],
                [x0 - 2.0 + t * w, y0 - 100.0],
                [x1 + 1000.0, y0 - 2.0 + t * h],
            ]
        })
        .collect()
}

/// Not "small enough to round to zero" — the literal `+0.0`, bit for bit.
/// The support's claim is that the kernel *is* the constant outside the box,
/// and `Support::contains().select(_, 0.0)` makes that structural, so a
/// tolerance here would accept a ramp that leaks instead.
fn assert_zero(
    label: &str,
    bounds: [f32; 4],
    probes: &[[f32; 2]],
    mut at: impl FnMut([f32; 2]) -> f32,
) {
    for &[x, y] in probes {
        let v = at([x, y]);
        assert_eq!(
            v.to_bits(),
            0f32.to_bits(),
            "{label}: {v:e} at ({x}, {y}) outside the support {bounds:?}"
        );
    }
}

/// Outside the reported support the kernel is the literal `0.0` — sampled on
/// a two-pixel ring just outside the box, and far away.
///
/// Banded like the winding sweeps above, and for the same reason: this is the
/// suite that walked into nextest's ten-minute cap, 119 production
/// saturations in one process to answer 72 probes each.
fn is_exactly_zero_outside_its_support(band: RangeInclusive<char>, size: f32) {
    let font = Font::parse(FONT_DATA).expect("font");
    for ch in band {
        let id = font.cmap_lookup(ch).expect("glyph");
        let glyph = font.glyph_scaled_by_id(id, size).expect("glyph");
        if glyph.support.is_empty() {
            continue;
        }
        // `Lattice::eval_at` binds nothing, so — unlike before S1a — it
        // cannot serve this kernel's winding table; bind it explicitly.
        let bound = glyph.bound(&glyph.kernel(), [1, 1]);
        assert_zero(
            &format!("{ch:?}@{size}"),
            glyph.support.bounds(),
            &probes_outside(glyph.support),
            |[x, y]| bound.eval_at(x, y),
        );
    }
}

/// The awkward shapes, at the two sizes a terminal actually renders.
#[test]
fn a_glyph_is_exactly_zero_outside_its_support() {
    let font = Font::parse(FONT_DATA).expect("font");
    for ch in GLYPHS {
        for size in [12.0f32, 20.0] {
            let id = font.cmap_lookup(ch).expect("glyph");
            let glyph = font.glyph_scaled_by_id(id, size).expect("glyph");
            let bound = glyph.bound(&glyph.kernel(), [1, 1]);
            assert_zero(
                &format!("{ch:?}@{size}"),
                glyph.support.bounds(),
                &probes_outside(glyph.support),
                |[x, y]| bound.eval_at(x, y),
            );
        }
    }
}

/// The same claim across the whole printable range, at the size where a
/// support computed one pixel too tight has the most room to show.
#[test]
fn punctuation_and_digits_are_zero_outside_their_support() {
    is_exactly_zero_outside_its_support(' '..='?', SUPPORT_SWEEP_PX);
}

#[test]
fn uppercase_is_zero_outside_its_support() {
    is_exactly_zero_outside_its_support('@'..='_', SUPPORT_SWEEP_PX);
}

#[test]
fn lowercase_is_zero_outside_its_support() {
    is_exactly_zero_outside_its_support('`'..='~', SUPPORT_SWEEP_PX);
}

/// The bounding box of the *ink* plus the ramp's reach: the support is not
/// the loose box the scanline kernel's trailing ramps forced (up to 21 px past
/// the ink at 48 px) — it is the outline's bounding box plus one pixel.
#[test]
fn the_support_is_the_outline_bounds_plus_the_ramp_reach() {
    let font = Font::parse(FONT_DATA).expect("font");
    let id = font.cmap_lookup('}').expect("glyph");
    let outline = font.outline_scaled_by_id(id, 48.0).expect("outline");
    let [x0, y0, x1, y1] = outline.bounds().expect("bounds");
    let support = font.glyph_scaled_by_id(id, 48.0).expect("glyph").support;
    assert_eq!(
        support.bounds(),
        [
            x0 - loop_blinn::RAMP_REACH,
            y0 - loop_blinn::RAMP_REACH,
            x1 + loop_blinn::RAMP_REACH,
            y1 + loop_blinn::RAMP_REACH
        ]
    );
}
