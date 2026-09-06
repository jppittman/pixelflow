//! The two encodings of a laid-out string, and what makes them agree.
//!
//! `text` sums the placed glyphs into one kernel, so every pixel evaluates
//! every glyph; `text_union` restricts the index instead, so a glyph is only
//! asked about the columns its support reaches. Four things have to hold for
//! that to be a rewrite rather than an approximation, and the first three are
//! pinned **bit-exactly**:
//!
//! 1. **The union machinery is exact.** A union whose one range is the whole
//!    frame, carrying the same kernel, bakes the same buffer as a plain bake —
//!    down to the bit. That pins the ranges, the destination offsets, the
//!    sampling origin, and the no-overhang collapse.
//! 2. **A glyph is *exactly* zero outside its support.** Not small: the
//!    literal `0.0`. That is the whole licence for dropping a glyph from a
//!    cell, and it is a property of how `ttf::compile` composes an outline (a
//!    unit-square mask whose false arm is the constant 0), so it is checked
//!    against the font rather than assumed.
//! 3. **Every cell bakes exactly its own kernel at its own shape.** Together
//!    with (2) that *is* the correctness argument: the union computes the
//!    right function, and drops only summands that were the literal zero.
//! 4. **What is left is the compiler's, not the union's.** E-graph extraction
//!    is a function of the arena *and* of the lattice shape — the same glyph
//!    kernel compiled at 160×24 and at 9×24 differs in 6 of 216 samples by
//!    2.4e-7 — so a decomposition necessarily reschedules its pieces and
//!    rounds differently at antialiasing edges. That is not a difference the
//!    union introduces: baking the pieces separately and adding them in f32,
//!    with no union anywhere, pays the same. Test (4) is the only one with a
//!    threshold, and the threshold is the output alphabet's own step, chosen
//!    to sit between scheduling noise and the coverage-scale difference a
//!    wrongly dropped glyph would make.

use pixelflow_core::{IndexRange, Kernel, Lattice, Union};
use pixelflow_graphics::fonts::{text, text_cells, text_union, Font, TextCell};

const FONT_DATA: &[u8] = include_bytes!("../assets/DejaVuSansMono-Fallback.ttf");

const CORPUS: [&str; 10] = [
    "A",
    "AB",
    "HELLO",
    "ABCDEFGHIJ",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "The quick brown fox jumps over the lazy dog",
    "iiiiWWWWmmmm",
    "WWW",
    "{{{",
    " leading and trailing ",
];

const SIZES: [f32; 4] = [12.0, 16.0, 20.0, 32.0];

/// A frame wide enough for the string at the sample-center convention the
/// font path uses.
fn frame(text_str: &str, size: f32) -> Lattice {
    let columns = text_str.chars().count().max(1) as f32;
    Lattice {
        extent: [
            (columns * size).ceil() as u32,
            (size * 1.5).ceil() as u32,
            1,
            1,
        ],
        origin: [0.5, 0.5, 0.0, 0.0],
    }
}

/// The lattice one cell of the decomposition is collapsed over: the cell's
/// extent, sampled at the ambient frame's coordinates for those indices.
fn cell_lattice(frame: Lattice, cell: &TextCell) -> Lattice {
    Lattice {
        extent: [
            cell.range.width() as u32,
            cell.range.rows() as u32,
            frame.extent[2],
            frame.extent[3],
        ],
        origin: [
            frame.origin[0] + cell.range.x0() as f32,
            frame.origin[1] + cell.range.y0() as f32,
            frame.origin[2],
            frame.origin[3],
        ],
    }
}

/// Samples that differ, and by how much.
fn compare(a: &[f32], b: &[f32]) -> (usize, f32) {
    assert_eq!(a.len(), b.len(), "buffers of different size");
    let differing = a
        .iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    let worst = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    (differing, worst)
}

/// 1. The union machinery, with the arena held fixed: one range covering the
///    whole frame, carrying exactly the kernel a plain bake would take.
#[test]
fn a_whole_frame_union_bakes_the_plain_bake_bit_for_bit() {
    let font = Font::parse(FONT_DATA).expect("font");
    for text_str in CORPUS {
        for size in SIZES {
            let lattice = frame(text_str, size);
            let kernel = text(&font, text_str, size);
            let mut union = Union::over(lattice);
            union.place(
                IndexRange::new(0, 0, lattice.extent[0] as usize, lattice.extent[1] as usize),
                &kernel,
            );
            let (differing, worst) = compare(lattice.bake(&kernel).buffer(), union.bake().buffer());
            assert_eq!(
                differing, 0,
                "{text_str:?} at {size}: a whole-frame union moved {differing} samples \
                 (worst {worst:e}) — the union's own machinery must be exact"
            );
        }
    }
}

/// 2. The licence for dropping a glyph: outside the box `Font::glyph_scaled_by_id`
///    reports, its kernel is the literal `0.0` at every sample.
///
/// A tolerance here would defeat the purpose — "small enough to ignore" is
/// precisely the claim this is meant to refuse. It is also the assumption an
/// ink box plus a fixed apron would get *wrong*: the crossing ramp is `‖∇d‖`
/// pixels wide in X, so a nearly horizontal segment leaves a glyph nonzero
/// tens of pixels past its outline (`}` at 48 px reaches 21 px right of its
/// ink). The unit-square mask does not care, because its false arm is a
/// constant.
#[test]
fn a_glyph_is_exactly_zero_outside_its_support() {
    let font = Font::parse(FONT_DATA).expect("font");
    const PEN: f32 = 48.0;
    for size in [12.0f32, 20.0, 48.0] {
        let lattice = Lattice {
            extent: [160, (size * 2.5) as u32, 1, 1],
            origin: [0.5, 0.5, 0.0, 0.0],
        };
        let (w, h) = (lattice.extent[0] as usize, lattice.extent[1] as usize);
        for ch in ' '..='~' {
            let id = font.cmap_lookup(ch).unwrap_or(0);
            let Some(glyph) = font.glyph_scaled_by_id(id, size) else {
                continue;
            };
            let placed = glyph.kernel.at(
                &Kernel::x().sub(&Kernel::constant(PEN)),
                &Kernel::y(),
                &Kernel::z(),
                &Kernel::w(),
            );
            let [x0, y0, x1, y1] = glyph.support.shifted_x(PEN).bounds();
            let baked = lattice.bake(&placed);
            for row in 0..h {
                let cy = row as f32 + lattice.origin[1];
                for col in 0..w {
                    let cx = col as f32 + lattice.origin[0];
                    let inside = (x0..=x1).contains(&cx) && (y0..=y1).contains(&cy);
                    if inside {
                        continue;
                    }
                    let v = baked.buffer()[row * w + col];
                    assert_eq!(
                        v.to_bits(),
                        0f32.to_bits(),
                        "{ch:?} at {size}: sample ({cx}, {cy}) is {v:e} outside the \
                         support [{x0}, {y0}, {x1}, {y1}]"
                    );
                }
            }
        }
    }
}

/// 3a. **Every cell bakes exactly its own kernel at its own shape** — bit for
///     bit, no tolerance. This is the complete statement of what the union
///     promises: the range, the destination offset, the sampling origin and
///     the no-overhang collapse, with nothing left for the compiler to vary,
///     because the reference is compiled from the same arena at the same
///     extent and sampled at the same coordinates.
#[test]
fn every_cell_bakes_exactly_its_own_kernel_at_its_own_shape() {
    let font = Font::parse(FONT_DATA).expect("font");
    let mut checked = 0usize;
    for text_str in CORPUS {
        for size in SIZES {
            let lattice = frame(text_str, size);
            let width = lattice.extent[0] as usize;
            let united = text_union(&font, lattice, text_str, size).bake();
            for cell in text_cells(&font, lattice, text_str, size) {
                checked += 1;
                let alone = cell_lattice(lattice, &cell).bake(&Kernel::sum(&cell.glyphs));
                for row in 0..cell.range.rows() {
                    for col in 0..cell.range.width() {
                        let here = (cell.range.y0() + row) * width + cell.range.x0() + col;
                        let there = row * cell.range.width() + col;
                        assert_eq!(
                            united.buffer()[here].to_bits(),
                            alone.buffer()[there].to_bits(),
                            "{text_str:?} at {size}: the cell at column {} is {:?} at \
                             (row {row}, col {col}) in the union but {:?} baked on its own",
                            cell.range.x0(),
                            united.buffer()[here],
                            alone.buffer()[there],
                        );
                    }
                }
            }
        }
    }
    assert!(
        checked > 100,
        "only {checked} cells in the corpus — this gate is not covering anything"
    );
}

/// One step of the 8-bit coverage every consumer of this path quantizes to.
///
/// This is the line between the two failure modes, and it is drawn by the
/// output alphabet rather than by what makes the numbers pass. A **wrongly
/// dropped glyph** is a coverage-scale difference: a whole antialiased edge
/// pixel, tens to hundreds of steps. **Scheduling noise** — the same
/// expression extracted differently because the arena or the lattice shape
/// changed — is a rounding-scale difference. On this corpus they are more than
/// an order of magnitude apart (worst observed 1.785e-4, 4.6% of one step),
/// and a regression that dropped a glyph could not hide under this.
const COVERAGE_STEP: f32 = 1.0 / 255.0;

/// 3b. What is left over once tests 1, 2 and 3a have pinned the union exactly:
///     the union's cells are a *different arena at a different extent* from
///     the whole string's, and e-graph extraction is a function of both, so
///     the compiler is free to schedule them differently and round differently
///     at antialiasing edges.
///
/// The gate is that this stays scheduling noise. It is checked against the
/// same measurement made with **no union anywhere** — the cell's glyphs baked
/// individually and added in f32 — so the bound is shown to be the compiler's
/// rather than the union's. (Their *ordering* is not a theorem: three
/// schedules of one expression do not round in any particular order, and
/// asserting one is below the other fails on rounding coincidences that mean
/// nothing.)
#[test]
fn a_shared_cell_only_moves_a_sample_by_scheduling_noise() {
    let font = Font::parse(FONT_DATA).expect("font");
    let mut shared = 0usize;
    let (mut worst_union, mut worst_split) = (0.0f32, 0.0f32);
    for text_str in CORPUS {
        for size in SIZES {
            let lattice = frame(text_str, size);
            let width = lattice.extent[0] as usize;
            let whole = lattice.bake(&text(&font, text_str, size));
            let united = text_union(&font, lattice, text_str, size).bake();
            for cell in text_cells(&font, lattice, text_str, size) {
                if cell.glyphs.len() < 2 {
                    continue;
                }
                shared += 1;
                let piece = cell_lattice(lattice, &cell);
                let mut split = vec![0.0f32; cell.range.width() * cell.range.rows()];
                for glyph in &cell.glyphs {
                    for (dst, src) in split.iter_mut().zip(piece.bake(glyph).buffer()) {
                        *dst += src;
                    }
                }
                for row in 0..cell.range.rows() {
                    for col in 0..cell.range.width() {
                        let here = (cell.range.y0() + row) * width + cell.range.x0() + col;
                        let there = row * cell.range.width() + col;
                        let reference = whole.buffer()[here];
                        worst_union = worst_union.max((united.buffer()[here] - reference).abs());
                        worst_split = worst_split.max((split[there] - reference).abs());
                    }
                }
            }
        }
    }
    assert!(
        shared > 10,
        "only {shared} shared cells in the corpus — this gate is not covering anything"
    );
    assert!(
        worst_union < COVERAGE_STEP,
        "a shared cell moved a sample by {worst_union:e}, which is {:.1} coverage \
         steps — that is a dropped glyph, not a schedule",
        worst_union / COVERAGE_STEP
    );
    assert!(
        worst_split < COVERAGE_STEP,
        "splitting the arena with no union in sight already moves a sample by \
         {worst_split:e} ({:.1} coverage steps), so this gate is measuring something \
         other than the union",
        worst_split / COVERAGE_STEP
    );
}
/// A string whose glyphs are wide relative to their advance is the case where
/// a neighbour's support spills into a cell, so the cell takes both glyphs and
/// the union's kernel *is* the sum's. Then there is nothing left for the
/// compiler to schedule differently, and the agreement is exact.
#[test]
fn a_cell_that_takes_every_glyph_agrees_exactly() {
    let font = Font::parse(FONT_DATA).expect("font");
    for text_str in ["W", "@", "{"] {
        for size in [16.0f32, 48.0] {
            let lattice = frame(text_str, size);
            let (differing, worst) = compare(
                lattice.bake(&text(&font, text_str, size)).buffer(),
                text_union(&font, lattice, text_str, size).bake().buffer(),
            );
            assert_eq!(
                differing, 0,
                "{text_str:?} at {size}: {differing} samples differ by up to {worst:e} \
                 with only one glyph to schedule"
            );
        }
    }
}

/// The empty string places no summand, and an unclaimed frame is all zeros.
#[test]
fn text_that_reaches_nothing_collapses_to_zero() {
    let font = Font::parse(FONT_DATA).expect("font");
    let lattice = Lattice {
        extent: [32, 32, 1, 1],
        origin: [0.5, 0.5, 0.0, 0.0],
    };
    let union = text_union(&font, lattice, "", 16.0);
    assert!(union.is_empty(), "the empty string places no summand");
    assert!(union.bake().buffer().iter().all(|&v| v == 0.0));
}

/// Two summands claiming one column is not a blend and not a painter's order.
/// The union refuses it when it is built, so a scene that would have produced
/// silently-wrong pixels never reaches a collapse.
#[test]
#[should_panic(expected = "overlaps the summand")]
fn overlapping_ranges_are_refused_at_build() {
    let lattice = Lattice {
        extent: [64, 16, 1, 1],
        origin: [0.5, 0.5, 0.0, 0.0],
    };
    let mut union = Union::over(lattice);
    union.place(IndexRange::new(0, 0, 32, 16), &Kernel::constant(1.0));
    union.place(IndexRange::new(31, 0, 33, 16), &Kernel::constant(2.0));
}
