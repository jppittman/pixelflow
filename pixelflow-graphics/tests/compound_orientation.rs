//! A compound glyph whose component is placed by a mirror covers its ink
//! with no seam, judged by the exact area under each texel.
//!
//! A symmetric glyph is often one component drawn twice, once reflected.
//! The reflection turns that copy inside out: it winds −1 where the other
//! winds +1. `|w|` does not mind, so no winding test sees it. Coverage adds
//! signed area, though, so a pixel straddling the join of the two halves
//! sums `+a − (1 − a)`, and the join shows as a seam. `ttf.rs` reverses a
//! mirrored component's contours. This test proves that on a font built
//! here byte by byte, because the crate's own font has no mirrored
//! components.
//!
//! The judge is the exact-area reference (`common/exact_area.rs`), not the
//! renderer. Today's renderer ramps coverage at every boundary it draws and
//! softens this join either way; it is the formula glyph, which adds signed
//! area, that the orientation matters to.

#[path = "common/exact_area.rs"]
mod exact_area;

use exact_area::{coverage, screen_pieces, signed_area, Grid};
use pixelflow_graphics::fonts::{Affine, Font, Outline};

// ─────────────────────────────── the font ───────────────────────────────

/// Font units per em, and the vertical metrics: 1000 units from ascent to
/// descent, so at [`SIZE`] px one unit is 0.02 px.
const UNITS_PER_EM: u16 = 1000;
const ASCENT: i16 = 800;
const DESCENT: i16 = -200;

/// The half: a `D` whose straight side is `x = 0`, from `y = 0` to `700`,
/// bulging right through the control point `(400, 350)` to `x = 200`.
const HALF: [(i16, i16, bool); 3] = [(0, 0, true), (400, 350, false), (0, 700, true)];

/// Where both halves' straight sides sit: at [`SIZE`] px, `x = 10.4`, so
/// the join cuts texel column 10 at 0.4.
const JOIN: i16 = 520;

/// `'A'`: the half alone. `'B'`: the half at [`JOIN`] and its mirror image
/// across the join (x-scale −1). `'C'`: the half at [`JOIN`] and its
/// half-turn about `(JOIN, 350)` (both scales −1). This `D` is symmetric
/// about `y = 350`, so `'B'` and `'C'` draw the same lens. The half-turn
/// keeps orientation and so is not reversed; the mirror does not keep it.
const HALF_ID: u16 = 1;
const MIRRORED_ID: u16 = 2;
const TURNED_ID: u16 = 3;

const ON_CURVE: u8 = 0x01;
const ARGS_ARE_WORDS: u16 = 0x0001;
const ARGS_ARE_XY_VALUES: u16 = 0x0002;
const MORE_COMPONENTS: u16 = 0x0020;
const X_AND_Y_SCALE: u16 = 0x0040;
/// ±1.0 in F2Dot14.
const PLUS_ONE: i16 = 0x4000;
const MINUS_ONE: i16 = -0x4000;

fn be16(out: &mut Vec<u8>, v: i32) {
    out.extend_from_slice(&(v as u16).to_be_bytes());
}

fn be32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_be_bytes());
}

/// A simple glyph: one contour through `points`, every coordinate a word.
fn simple_glyph(points: &[(i16, i16, bool)]) -> Vec<u8> {
    let mut g = Vec::new();
    be16(&mut g, 1); // one contour
    for bound in [0, 0, 400, 700] {
        be16(&mut g, bound);
    }
    be16(&mut g, points.len() as i32 - 1); // its last point
    be16(&mut g, 0); // no instructions
    for &(_, _, on) in points {
        g.push(if on { ON_CURVE } else { 0 });
    }
    let mut previous = (0i32, 0i32);
    let mut ys = Vec::new();
    for &(x, y, _) in points {
        be16(&mut g, i32::from(x) - previous.0);
        be16(&mut ys, i32::from(y) - previous.1);
        previous = (i32::from(x), i32::from(y));
    }
    g.extend(ys);
    g
}

/// A compound glyph: [`HALF_ID`] at `(JOIN, 0)`, then [`HALF_ID`] again
/// scaled by `scale` and offset by `offset`.
fn compound_glyph(scale: (i16, i16), offset: (i16, i16)) -> Vec<u8> {
    let mut g = Vec::new();
    be16(&mut g, -1); // compound
    for bound in [0, 0, 0, 0] {
        be16(&mut g, bound);
    }
    be16(
        &mut g,
        i32::from(ARGS_ARE_WORDS | ARGS_ARE_XY_VALUES | MORE_COMPONENTS),
    );
    be16(&mut g, i32::from(HALF_ID));
    be16(&mut g, i32::from(JOIN));
    be16(&mut g, 0);
    be16(
        &mut g,
        i32::from(ARGS_ARE_WORDS | ARGS_ARE_XY_VALUES | X_AND_Y_SCALE),
    );
    be16(&mut g, i32::from(HALF_ID));
    be16(&mut g, i32::from(offset.0));
    be16(&mut g, i32::from(offset.1));
    be16(&mut g, i32::from(scale.0));
    be16(&mut g, i32::from(scale.1));
    g
}

/// A TrueType file holding exactly what `Font::parse` reads: `head`,
/// `hhea`, `hmtx`, `loca` (long), `glyf`, and a format-4 `cmap` sending
/// `'A'`, `'B'`, `'C'` to glyphs 1, 2, 3.
fn font_bytes() -> Vec<u8> {
    let glyphs = [
        Vec::new(), // .notdef: empty
        simple_glyph(&HALF),
        compound_glyph((MINUS_ONE, PLUS_ONE), (JOIN, 0)),
        compound_glyph((MINUS_ONE, MINUS_ONE), (JOIN, 700)),
    ];
    let (mut glyf, mut loca) = (Vec::new(), Vec::new());
    for g in &glyphs {
        be32(&mut loca, glyf.len() as u32);
        glyf.extend(g);
        glyf.resize(glyf.len().next_multiple_of(4), 0);
    }
    be32(&mut loca, glyf.len() as u32);

    let mut head = vec![0u8; 54];
    head[18..20].copy_from_slice(&UNITS_PER_EM.to_be_bytes());
    head[50..52].copy_from_slice(&1i16.to_be_bytes()); // long loca

    let mut hhea = vec![0u8; 36];
    hhea[4..6].copy_from_slice(&ASCENT.to_be_bytes());
    hhea[6..8].copy_from_slice(&DESCENT.to_be_bytes());
    hhea[34..36].copy_from_slice(&(glyphs.len() as u16).to_be_bytes());

    let mut hmtx = Vec::new();
    for _ in &glyphs {
        be16(&mut hmtx, 1000);
        be16(&mut hmtx, 0);
    }

    let mut cmap = Vec::new();
    be16(&mut cmap, 0); // version
    be16(&mut cmap, 1); // one encoding record
    be16(&mut cmap, 3); // Windows
    be16(&mut cmap, 1); // Unicode BMP
    be32(&mut cmap, 12); // its subtable follows
    let segments: [(u16, u16, i16); 2] = [(0x41, 0x43, 1 - 0x41), (0xFFFF, 0xFFFF, 1)];
    be16(&mut cmap, 4); // format
    be16(&mut cmap, (16 + 8 * segments.len()) as i32); // length
    be16(&mut cmap, 0); // language
    be16(&mut cmap, 2 * segments.len() as i32);
    for _ in 0..3 {
        be16(&mut cmap, 0); // search hints, unread
    }
    for &(_, end, _) in &segments {
        be16(&mut cmap, i32::from(end));
    }
    be16(&mut cmap, 0); // reserved
    for &(start, _, _) in &segments {
        be16(&mut cmap, i32::from(start));
    }
    for &(_, _, delta) in &segments {
        be16(&mut cmap, i32::from(delta));
    }
    for _ in &segments {
        be16(&mut cmap, 0); // no range offsets
    }

    let tables: [(&[u8; 4], Vec<u8>); 6] = [
        (b"cmap", cmap),
        (b"glyf", glyf),
        (b"head", head),
        (b"hhea", hhea),
        (b"hmtx", hmtx),
        (b"loca", loca),
    ];
    let mut out = Vec::new();
    be32(&mut out, 0x0001_0000);
    be16(&mut out, tables.len() as i32);
    for _ in 0..3 {
        be16(&mut out, 0); // search hints, unread
    }
    let mut offset = 12 + 16 * tables.len();
    let mut data = Vec::new();
    for (tag, table) in &tables {
        out.extend_from_slice(*tag);
        be32(&mut out, 0); // checksum, unread
        be32(&mut out, offset as u32);
        be32(&mut out, table.len() as u32);
        data.extend(table);
        data.resize(data.len().next_multiple_of(4), 0);
        offset = 12 + 16 * tables.len() + data.len();
    }
    out.extend(data);
    out
}

// ─────────────────────────────── the test ───────────────────────────────

/// Pixels from ascent to descent: one font unit is 0.02 px.
const SIZE: f64 = 20.0;
const GRID: Grid = Grid {
    width: 20,
    height: 20,
};
/// The texel column the join cuts, at 0.4 of its width.
const JOIN_COLUMN: usize = 10;
/// The rows of that column that lie wholly inside the lens: the lens is at
/// least 30 units (0.6 px) wide on each side of the join from font `y = 27`
/// to `673`, which is screen `y` 2.54 to 15.46.
const INSIDE_ROWS: std::ops::RangeInclusive<usize> = 3..=14;

fn exact_coverage(font: &Font, ch: char) -> Vec<f64> {
    signed_area(&screen_pieces(font, ch, SIZE), GRID)
        .into_iter()
        .map(coverage)
        .collect()
}

#[test]
fn the_synthetic_font_parses_into_the_glyphs_it_describes() {
    let bytes = font_bytes();
    let font = Font::parse(&bytes).expect("the synthetic font parses");
    for (ch, id, contours) in [
        ('A', HALF_ID, 1),
        ('B', MIRRORED_ID, 2),
        ('C', TURNED_ID, 2),
    ] {
        assert_eq!(font.cmap_lookup(ch), Some(id));
        let outline = font.outline_by_id(id).expect("an outline");
        assert_eq!(outline.contours.len(), contours, "{ch:?}");
    }
    let half: f64 = exact_coverage(&font, 'A').iter().sum();
    // Archimedes: two thirds of the control triangle, 0.02² px² per unit².
    let want = 2.0 / 3.0 * (0.5 * 400.0 * 700.0) * 0.02 * 0.02;
    assert!(
        (half - want).abs() < 1e-9,
        "the half covers {half}, not {want}"
    );
}

/// The lens has no seam: every texel of the join's column that the lens
/// wholly covers reads exactly 1. The mirrored lens is also, texel for
/// texel, the half-turned one, which was never reversed.
#[test]
fn a_mirrored_component_joins_its_original_without_a_seam() {
    let bytes = font_bytes();
    let font = Font::parse(&bytes).expect("the synthetic font parses");
    let mirrored = exact_coverage(&font, 'B');
    let turned = exact_coverage(&font, 'C');
    for j in INSIDE_ROWS {
        for (name, lens) in [("mirrored", &mirrored), ("half-turned", &turned)] {
            let c = lens[j * GRID.width + JOIN_COLUMN];
            assert!(
                (c - 1.0).abs() < 1e-9,
                "{name} lens: texel ({JOIN_COLUMN}, {j}) on the join reads {c}"
            );
        }
    }
    for (k, (m, t)) in mirrored.iter().zip(&turned).enumerate() {
        assert!(
            (m - t).abs() < 1e-9,
            "texel ({}, {}): mirrored {m}, half-turned {t}",
            k % GRID.width,
            k / GRID.width
        );
    }
}

/// The same two components with the mirror applied and nothing reversed
/// — what `ttf.rs` built before — straddle the join at `|0.6 − 0.4|`.
/// So the seam is real, and the test above could see one.
#[test]
fn without_the_reversal_the_join_is_a_seam() {
    let bytes = font_bytes();
    let font = Font::parse(&bytes).expect("the synthetic font parses");
    let half = font.outline_by_id(HALF_ID).expect("the half");
    let join = f32::from(JOIN);
    let mut unreversed = Outline::default();
    unreversed.append(half.transformed(Affine::translation(join, 0.0)));
    unreversed.append(half.transformed(Affine([-1.0, 0.0, 0.0, 1.0, join, 0.0])));
    let scale = SIZE / (f64::from(ASCENT) - f64::from(DESCENT));
    let pieces = exact_area::pieces(&unreversed, |[x, y]| {
        [
            f64::from(x) * scale,
            (f64::from(ASCENT) - f64::from(y)) * scale,
        ]
    });
    let seamed: Vec<f64> = signed_area(&pieces, GRID)
        .into_iter()
        .map(coverage)
        .collect();
    for j in INSIDE_ROWS {
        let c = seamed[j * GRID.width + JOIN_COLUMN];
        assert!(
            (c - 0.2).abs() < 1e-9,
            "texel ({JOIN_COLUMN}, {j}) reads {c}, not the seam's 0.2"
        );
    }
}
