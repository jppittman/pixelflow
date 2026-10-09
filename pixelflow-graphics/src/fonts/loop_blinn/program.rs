//! **A font at one tile size is one program per piece count.**
//!
//! A glyph is its pieces' terms summed under its box, written once in
//! `kernel!` below: [`one_piece`] is a piece's term over its own ten
//! uniforms, [`sum2`] the monoid the host folds a glyph's pieces with, and
//! [`glyph`] coverage under the box (docs/plans/2026-09-25-the-language-is-kernel.md
//! §1.7). Every value a glyph holds is a uniform, so two glyphs with the
//! same number of pieces are the same program over different blocks: the
//! font at a tile size is the program for each piece count its glyphs have,
//! and drawing a glyph is writing its block ([`GlyphRows`]) into its count's
//! program and calling it ([`FontPrograms::draw`]). JP, 2026-10-09: *the
//! "atlas" becomes the kernel for that number of control points, everything
//! else is a uniform.*
//!
//! Choosing a glyph's program is choosing by a structural parameter, the
//! piece count, which is what structural means: each value its own program.
//! There is no table: a glyph's rows are a block's values, written per call.
//!
//! What stays host Rust: the outline's monotone split and each piece's
//! rounding to a row ([`pieces`], [`piece_row`]), which [`GlyphRows::of`]
//! reads, as the builder's [`super::glyph`] does.

use super::{piece_row, pieces, PIECE_ROW_COLS};
use super::{COL_E0X, COL_E0Y, COL_E1X, COL_E1Y, COL_ROWS_HI, COL_ROWS_LO, COL_S, COL_SIGMA};
use super::{COL_X0, COL_Y0};
use crate::fonts::outline::Outline;
use pixelflow_compiler::kernel;
use pixelflow_core::{DiscreteManifold, Kernel, Lattice, Manifold};
use std::collections::BTreeMap;

kernel! {
    /// One oriented monotone arc piece.
    pub struct Row {
        pub x0: f32, pub e0x: f32, pub e1x: f32,
        pub y0: f32, pub e0y: f32, pub e1y: f32,
        pub sigma: f32, pub s: f32,
        pub lo: f32, pub hi: f32,
    }

    /// A glyph's box: the outline's bounding box, four uniforms. Coverage
    /// reaches half a pixel past it, which `inside` reaches too, so the host
    /// passes the outline's own box.
    pub struct Bounds { pub x0: f32, pub y0: f32, pub x1: f32, pub y1: f32 }

    const PIXEL_CENTER: f32 = 0.5;
    const COVERAGE_SNAP: f32 = 1.0 / 1024.0;
    const NEARLY_ONE: f32 = 1.0 - COVERAGE_SNAP;
    const PIXEL_HALF: f32 = 0.5;
    const ONE_THIRD: f32 = 1.0 / 3.0;
    const ROOT_FLOOR: f32 = 1.0 / 1_267_650_600_228_229_401_496_703_205_376.0;

    /// `τ(δ) = δ / max(step + √max(step² + bend·δ, 0), ROOT_FLOOR)`: the
    /// parameter at which the rise `t·(2·step + bend·t)` reaches the height
    /// `δ`, the reciprocal exact — `fonts/loop_blinn.rs`'s
    /// `Rise::monotone_root`, which carries its law.
    fn monotone_root(delta: f32, step: f32, bend: f32) -> f32 {
        delta * (1.0 / (step + (step * step + bend * delta).max(0.0).sqrt()).max(ROOT_FLOOR))
    }

    /// `2·step + bend·s`: a rise `q(t) = t·(2·step + bend·t)` is `t` times it
    /// at `s = t`, and climbs `(t − s)` times it at `s + t` from `s` to `t`.
    fn slope_through(step: f32, bend: f32, s: f32) -> f32 {
        step + step + bend * s
    }

    /// The area of the pixel about `(x, y)` left of the arc, within its
    /// band, in closed form: `fonts/loop_blinn.rs`'s
    /// `RisingArc::pixel_area`, which carries the derivation.
    fn piece_area(p: Row, x: f32, y: f32) -> f32 {
        let b = p.e0y.max(0.0);
        let bx = p.e0x.max(0.0);
        let a = p.e1y.max(0.0) - b;
        let ax = p.e1x.max(0.0) - bx;
        let across = x - p.x0;
        let up = y - p.y0;
        let left = across - PIXEL_HALF;
        let right = across + PIXEL_HALF;
        // Where the arc enters and leaves the pixel's rows, and where it
        // reaches the pixel's left and right edges.
        let t0 = monotone_root(up - PIXEL_HALF, b, a).clamp(0.0, 1.0);
        let t1 = monotone_root(up + PIXEL_HALF, b, a).clamp(0.0, 1.0);
        let t_left = monotone_root(left, bx, ax).clamp(t0, t1);
        let t_right = monotone_root(right, bx, ax).clamp(t0, t1);
        let right_of_the_pixel = (t1 - t_right) * slope_through(b, a, t_right + t1);
        let x_left = t_left * slope_through(bx, ax, t_left) - left;
        let x_right = t_right * slope_through(bx, ax, t_right) - left;
        let rise = (t_right - t_left) * slope_through(b, a, t_left + t_right);
        let trapezoid = PIXEL_HALF * (x_left + x_right) * rise;
        let w = t_right - t_left;
        let bow = ONE_THIRD * (bx * a - b * ax) * (w * w * w);
        right_of_the_pixel + (trapezoid + bow)
    }

    /// σ·A over the pixel about (x, S·y), cut to the rows the piece reaches.
    fn piece_term(p: Row, x: f32, y: f32) -> f32 {
        let term = p.sigma * piece_area(p, x, p.s * y);
        if (y > p.lo) & (y < p.hi) { term } else { 0.0 }
    }

    /// Coverage, from a signed area: `min(|f|, 1)`, its ends snapped.
    fn coverage(f: f32) -> f32 {
        let c = f.abs().min(1.0);
        if c >= NEARLY_ONE { 1.0 } else if c <= COVERAGE_SNAP { 0.0 } else { c }
    }

    /// Whether the pixel about (x, y) can meet the outline: its centre
    /// within half a pixel of the outline's box. Every piece's term is
    /// exactly 0 farther out.
    fn inside(b: Bounds, x: f32, y: f32) -> bool {
        (x >= b.x0 - PIXEL_HALF) & (x <= b.x1 + PIXEL_HALF)
            & (y >= b.y0 - PIXEL_HALF) & (y <= b.y1 + PIXEL_HALF)
    }

    /// One piece's term at the sample, over its own ten uniforms. The host
    /// composes one instance per piece.
    pub fn one_piece(p: Row) -> f32 {
        piece_term(p, X, Y)
    }

    /// Two kernels summed at the sample: the operation of the monoid a
    /// glyph's ink is, which the host folds its pieces' instances with — a
    /// balanced tree, by index.
    pub fn sum2(a: impl Fn(f32, f32) -> f32, b: impl Fn(f32, f32) -> f32) -> f32 {
        a(X, Y) + b(X, Y)
    }

    /// One glyph. Texel (i, j) holds coverage at (i+½, j+½). `ink` is the sum
    /// of the glyph's pieces, composed by the host, and `ink(x, y)` reads it
    /// at the pixel's centre: application is contramap (§1.2).
    pub fn glyph(ink: impl Fn(f32, f32) -> f32, bounds: Bounds) -> f32 {
        let (x, y) = (X + PIXEL_CENTER, Y + PIXEL_CENTER);
        if inside(bounds, x, y) { coverage(ink(x, y)) } else { 0.0 }
    }

    /// A glyph with no pieces: no ink anywhere.
    pub fn blank() -> f32 {
        0.0
    }
}

/// One glyph's block: what a call writes into its piece count's program.
///
/// The outline's box, then one row of ten per piece, in piece order — the
/// order the program declares them in: [`glyph`] declares its own parameter
/// before its argument's, and [`ink`]'s balanced tree keeps the pieces in
/// order. A glyph with no pieces has an empty block, since [`blank`] reads
/// nothing.
#[derive(Clone, Debug, PartialEq)]
pub struct GlyphRows {
    pieces: usize,
    block: Vec<f32>,
}

impl GlyphRows {
    /// `outline`'s block: its pieces split and oriented as the builder's are
    /// ([`pieces`]), each rounded to its row ([`piece_row`]), under its box.
    #[must_use]
    pub fn of(outline: &Outline) -> Self {
        let rows: Vec<[f32; PIECE_ROW_COLS]> = pieces(outline).into_iter().map(piece_row).collect();
        if rows.is_empty() {
            return Self {
                pieces: 0,
                block: Vec::new(),
            };
        }
        let bounds = outline
            .bounds()
            .expect("an outline with pieces has a box: every piece is part of it");
        Self {
            pieces: rows.len(),
            block: bounds
                .into_iter()
                .chain(rows.into_iter().flatten())
                .collect(),
        }
    }

    /// How many pieces the glyph has: which program draws it.
    #[must_use]
    pub fn pieces(&self) -> usize {
        self.pieces
    }
}

/// The programs a font's glyphs are drawn by at one tile extent: one per
/// piece count, compiled when a glyph first needs it.
pub struct FontPrograms {
    extent: [u32; 2],
    programs: BTreeMap<usize, Manifold>,
}

impl FontPrograms {
    /// No programs yet, at `extent` (texels across, texels down).
    #[must_use]
    pub fn new(extent: [u32; 2]) -> Self {
        Self {
            extent,
            programs: BTreeMap::new(),
        }
    }

    /// `glyph`'s coverage over the tile: texel `(i, j)` holds the area of the
    /// pixel about `(i + ½, j + ½)` under its ink. Its count's program,
    /// compiled on first use, called over its block.
    ///
    /// # Panics
    ///
    /// Never for a [`GlyphRows`] from [`GlyphRows::of`]: its block is its
    /// count's program's arguments by construction.
    pub fn draw(&mut self, glyph: &GlyphRows) -> DiscreteManifold {
        let extent = self.extent;
        let program = self
            .programs
            .entry(glyph.pieces)
            .or_insert_with(|| Manifold::compile(&program(glyph.pieces), extent));
        let mut block = program.block();
        block
            .set_declared(glyph.block.iter().copied())
            .expect("a glyph's block is its count's arguments: the box, then ten per piece");
        let [width, height] = extent.map(|texels| texels as usize);
        Lattice::frame(width, height).collapse(&program.bind(&[]).with_uniforms(&block))
    }
}

/// The program every glyph of `pieces` pieces is drawn by: `pieces`
/// instances of [`one_piece`] summed by [`ink`] under [`glyph`]'s box, every
/// value a uniform the block writes, or [`blank`] for none.
fn program(pieces: usize) -> Kernel {
    if pieces == 0 {
        return blank();
    }
    let instances: Vec<Kernel> = (0..pieces)
        .map(|_| one_piece(record([0.0; PIECE_ROW_COLS])))
        .collect();
    glyph(
        &ink(&instances),
        Bounds {
            x0: 0.0,
            y0: 0.0,
            x1: 0.0,
            y1: 0.0,
        },
    )
}

/// `row` as the block's record: the columns, in declaration order.
fn record(row: [f32; PIECE_ROW_COLS]) -> Row {
    Row {
        x0: row[COL_X0],
        e0x: row[COL_E0X],
        e1x: row[COL_E1X],
        y0: row[COL_Y0],
        e0y: row[COL_E0Y],
        e1y: row[COL_E1Y],
        sigma: row[COL_SIGMA],
        s: row[COL_S],
        lo: row[COL_ROWS_LO],
        hi: row[COL_ROWS_HI],
    }
}

/// The sum of `terms` by [`sum2`] alone: a balanced tree, halved by index,
/// so a glyph's pieces compose to one shape however many there are, and
/// declare their uniforms in piece order.
fn ink(terms: &[Kernel]) -> Kernel {
    match terms {
        [] => panic!("a glyph with no pieces has no ink to compose"),
        [term] => term.clone(),
        _ => {
            let (left, right) = terms.split_at(terms.len() / 2);
            sum2(&ink(left), &ink(right))
        }
    }
}

#[cfg(test)]
mod tests;
