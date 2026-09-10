//! **A run of characters is a glyph, and glyphs form a monoid.**
//!
//! The laws behind docs/plans/2026-09-09-a-run-is-a-glyph.md. `text()` used
//! to merge every character's contours into one `Outline` before building a
//! kernel, on the correct premise that *coverage is not additive*. Windings
//! are, and they vanish outside a closed contour, so the run folds the
//! windings instead and each character keeps its own bounding box.
//!
//! What that buys is binning; what it risks is composition being wrong in a
//! way a single glyph never exercises. These are the properties that would
//! catch it. The pixel-for-pixel agreement with the old merge is pinned
//! separately by `font_rasterization_regression`'s text goldens.

use pixelflow_core::{Kernel, Lattice, Manifold};
use pixelflow_graphics::fonts::{loop_blinn, text, Contour, Font, Outline, Segment};

const FONT_BYTES: &[u8] = include_bytes!("../assets/DejaVuSansMono-Fallback.ttf");

/// The lattice every comparison collapses over — wide enough to hold a short
/// run at this size, so the boxes it bins by are genuinely disjoint.
const EXTENT: [u32; 2] = [96, 32];

fn font() -> Font<'static> {
    Font::parse(FONT_BYTES).expect("parse font")
}

/// Tabulate a glyph's coverage over [`EXTENT`], binding nothing: whatever
/// tables the run reads must have travelled with its kernel.
fn render(glyph: &loop_blinn::Glyph) -> Vec<f32> {
    let kernel = glyph.kernel();
    let bound = Manifold::compile(&kernel, EXTENT).bind(&[]);
    Lattice::frame(EXTENT[0] as usize, EXTENT[1] as usize)
        .collapse(&bound)
        .buffer()
        .to_vec()
}

/// An axis-aligned box as a closed contour, wound counter-clockwise.
fn box_outline(x0: f32, y0: f32, x1: f32, y1: f32) -> Outline {
    let corners = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
    let segments = (0..4)
        .map(|i| Segment::Line {
            from: corners[i],
            to: corners[(i + 1) % 4],
        })
        .collect();
    Outline {
        contours: vec![Contour::new(segments).expect("a box closes")],
    }
}

/// **The identity.** An empty run is the monoid's unit: no ink anywhere.
#[test]
fn an_empty_run_is_the_identity() {
    let empty = render(&text(&font(), "", 24.0));
    assert!(
        empty.iter().all(|&c| c == 0.0),
        "an empty run must be exactly zero everywhere"
    );
    assert!(render(&loop_blinn::run(&[])).iter().all(|&c| c == 0.0));
}

/// **Commutativity.** `over` combines under `+` and `min`, both commutative,
/// so laying the same characters out in a different *order of composition*
/// cannot move a pixel. (The pen positions are baked into each outline, so
/// this permutes which order they are folded in, not where they sit.)
#[test]
fn composition_order_does_not_move_a_pixel() {
    let a = box_outline(4.0, 4.0, 20.0, 28.0);
    let b = box_outline(30.0, 4.0, 46.0, 28.0);
    let c = box_outline(56.0, 4.0, 72.0, 28.0);

    let forward = render(&loop_blinn::run(&[a.clone(), b.clone(), c.clone()]));
    let reversed = render(&loop_blinn::run(&[c, b, a]));
    assert_eq!(
        forward, reversed,
        "the fold is a monoid, so its order cannot matter"
    );
}

/// **Associativity, and the unit.** Folding a run in two halves and
/// combining, or folding it whole, are the same glyph.
#[test]
fn a_run_folds_the_same_in_any_grouping() {
    let a = box_outline(4.0, 4.0, 20.0, 28.0);
    let b = box_outline(30.0, 4.0, 46.0, 28.0);
    let c = box_outline(56.0, 4.0, 72.0, 28.0);

    let whole = render(&loop_blinn::run(&[a.clone(), b.clone(), c.clone()]));
    let split = render(&loop_blinn::Glyph::over(&[
        loop_blinn::run(&[a.clone(), b.clone()]),
        loop_blinn::run(std::slice::from_ref(&c)),
        loop_blinn::Glyph::empty(),
    ]));
    assert_eq!(
        whole, split,
        "grouping and the empty glyph must both be invisible to the result"
    );
}

/// **Binning is exact, not an approximation.** Two disjoint boxes composed
/// as a run must render exactly what each renders alone, added — because a
/// closed contour's winding is 0 outside it and the support is dilated by
/// the ramp's own reach, so masking each contributor to its box discards
/// only zeros and only losers of the `min`.
///
/// This is the property the whole design rests on: if masking lost anything,
/// the run would differ from the parts here.
#[test]
fn disjoint_characters_do_not_disturb_each_other() {
    let a = box_outline(4.0, 4.0, 20.0, 28.0);
    let b = box_outline(60.0, 4.0, 76.0, 28.0);

    let together = render(&loop_blinn::run(&[a.clone(), b.clone()]));
    let (alone_a, alone_b) = (
        render(&loop_blinn::run(&[a])),
        render(&loop_blinn::run(&[b])),
    );

    for (i, &both) in together.iter().enumerate() {
        let apart = alone_a[i] + alone_b[i];
        assert!(
            (both - apart).abs() < 1e-6,
            "texel {i}: run gives {both}, the two boxes alone give {apart} — \
             binning to a support must be exact, not approximate"
        );
    }
    assert!(
        together.iter().any(|&c| c > 0.5),
        "the fixture must actually draw ink"
    );
}

/// **One table for the run, however long it is.** A per-character table
/// would need a bound buffer slot each, and `Manifold::compile` refuses more
/// than `MAX_BOUND_BUFFERS` — so this is not a tidiness property, it is what
/// makes a run of more than four characters compile at all. It regressed
/// exactly once, the first time `text()` folded per character.
#[test]
fn a_long_run_binds_one_buffer_slot() {
    let font = font();
    let long = text(&font, "HELLO WORLD", 16.0);
    let program = Manifold::compile(&long.kernel(), EXTENT);
    assert_eq!(
        program.buffers().len(),
        1,
        "every character folds over its own rows of ONE shared table"
    );
}

/// A run's support is the union of its characters', so its coverage really
/// is zero outside — the claim `Support` makes, at run scale.
#[test]
fn a_run_is_zero_outside_its_support() {
    let font = font();
    let run = text(&font, "Hi", 20.0);
    let [x0, y0, x1, y1] = run.support.bounds();
    let coverage = render(&run);
    let width = EXTENT[0] as usize;
    for (i, &c) in coverage.iter().enumerate() {
        let (x, y) = ((i % width) as f32, (i / width) as f32);
        let outside = x < x0 || x > x1 || y < y0 || y > y1;
        assert!(
            !outside || c == 0.0,
            "texel ({x},{y}) is outside the support [{x0},{y0},{x1},{y1}] \
             but reads {c}"
        );
    }
}

/// Sanity on the fixture the laws above are stated over: a single box is a
/// glyph like any other, and `Kernel` composition still reaches it.
#[test]
fn a_box_is_a_glyph() {
    let one = loop_blinn::run(&[box_outline(8.0, 8.0, 24.0, 24.0)]);
    let coverage = render(&one);
    assert!(coverage.iter().any(|&c| c > 0.9), "the box must be filled");
    assert!(coverage.contains(&0.0), "and bounded");
    // Composes as an ordinary kernel: doubling coverage is just arithmetic.
    let doubled = one.kernel().mul(&Kernel::constant(2.0));
    let bound = Manifold::compile(&doubled, EXTENT).bind(&[]);
    let out = Lattice::frame(EXTENT[0] as usize, EXTENT[1] as usize)
        .collapse(&bound)
        .buffer()
        .to_vec();
    for (i, &c) in coverage.iter().enumerate() {
        assert!((out[i] - 2.0 * c).abs() < 1e-6);
    }
}
