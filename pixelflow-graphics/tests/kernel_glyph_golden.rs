//! Fused-vs-combinator golden: a real font glyph baked two ways must agree.
//!
//! - **Reference** (`CachedGlyph::new`): the combinator scene graph evaluated
//!   over `Jet2` screen-space autodiff (`Antialiased`) and tabulated via
//!   `Lattice::collapse` — the production path.
//! - **JIT-first** (`Font::glyph_kernel_scaled` → `Lattice::bake`): the same
//!   glyph dissolved into a single coverage `Kernel`, JIT-compiled once, with
//!   antialiasing resolved from symbolic `Dwrt` (no `Jet2` domain).
//!
//! The two AA formulations are proven equivalent for font coverage (P2 parity),
//! so the two baked lattices match closely. This is the guard for retiring the
//! combinator glyph pipeline.

use pixelflow_core::{Kernel, Lattice};
use pixelflow_graphics::fonts::{CachedGlyph, Font};

const FONT_BYTES: &[u8] = include_bytes!("../assets/DejaVuSansMono-Fallback.ttf");

fn golden_for(ch: char, size: usize) {
    let font = Font::parse(FONT_BYTES).expect("parse font");

    // Combinator reference: Jet2 AA coverage, baked via collapse.
    let glyph = font
        .glyph_scaled(ch, size as f32)
        .unwrap_or_else(|| panic!("glyph {ch:?} not found"));
    let cached = CachedGlyph::new(&glyph, size, 1.0);
    let reference = cached.coverage().buffer();
    let (w, h) = (cached.coverage().width(), cached.coverage().height());
    assert_eq!(reference.len(), w * h);

    // JIT-first: the same glyph as a Kernel, baked over the identical grid.
    let kernel: Kernel = font
        .glyph_kernel_scaled(ch, size as f32)
        .expect("glyph kernel");
    let baked = Lattice {
        extent: [w as u32, h as u32, 1, 1],
        origin: [0.5, 0.5, 0.0, 0.0],
    }
    .bake(&kernel);
    let got = baked.buffer();
    assert_eq!(got.len(), reference.len());

    // The reference must actually render ink (guards against a blank golden).
    let ink: f32 = reference.iter().sum();
    assert!(ink > 1.0, "reference glyph {ch:?} is blank (ink={ink})");

    // Dwrt AA vs Jet2 AA agree on font coverage → the lattices match closely.
    let mut sum_abs = 0.0f32;
    let mut max_abs = 0.0f32;
    let mut disagree = 0usize; // hard inside/outside classification mismatches
    for (&a, &b) in got.iter().zip(reference.iter()) {
        assert!(a.is_finite(), "kernel bake produced non-finite coverage");
        assert!((-0.01..=1.01).contains(&a), "coverage out of range: {a}");
        let d = (a - b).abs();
        sum_abs += d;
        max_abs = max_abs.max(d);
        // Count only gross disagreements: one path calls a texel solid ink,
        // the other calls it clear background.
        if (a > 0.75 && b < 0.25) || (a < 0.25 && b > 0.75) {
            disagree += 1;
        }
    }
    let mean = sum_abs / got.len() as f32;

    assert!(
        mean < 0.02,
        "{ch}@{size}: mean |Δcoverage| {mean:.4} too high (max {max_abs:.3})"
    );
    assert_eq!(
        disagree, 0,
        "{ch}@{size}: {disagree} texels flip inside/outside between the paths"
    );
}

#[test]
fn simple_glyph_bake_matches_combinator_cache() {
    // 'A' — a simple glyph (lines + no compound children).
    golden_for('A', 32);
}

#[test]
fn round_glyph_bake_matches_combinator_cache() {
    // 'O' — exercises quadratic Bezier leaves through the whole path. This was
    // the test that exposed two JIT codegen soundness bugs (the select-guard
    // mask-before-range assumption and the unsorted Clamp decomposition — see
    // pixelflow-ir/tests/spill_pressure.rs); it guards them now.
    golden_for('O', 32);
}

#[test]
fn descender_glyph_bake_matches_combinator_cache() {
    // 'g' — descender, multiple contours, quad-heavy.
    golden_for('g', 32);
}

#[test]
fn double_counter_glyph_bake_matches_combinator_cache() {
    // '8' — two enclosed counters: the winding sum crosses zero twice per
    // scanline, stressing the sign bookkeeping of the fused contributions.
    golden_for('8', 32);
}
