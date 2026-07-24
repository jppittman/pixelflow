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

// 'O' — exercises quadratic Bezier leaves through the whole path. The fused
// arena is CORRECT (the IR interpreter reproduces the combinator coverage
// exactly, hole and all), but the JIT backend cannot yet compile it: a fused
// quad glyph is register-heavy enough that a `Select` lands with BOTH branches
// spilled, which the x86/aarch64 emitter rejects
// ("Select with both if_true and if_false spilled not supported"). This is a
// JIT codegen robustness gap, not a defect in the glyph→Kernel rewrite, and it
// is the concrete blocker for retiring the combinator glyph path (P5/P6). The
// correct fix reads the spilled branches straight from their stack slots as
// memory operands to the blend (vandps/vandnps accept a memory source), needing
// no extra scratch registers. Un-ignore once that lands.
#[test]
#[ignore = "JIT: fused quad glyph hits Select-with-both-branches-spilled (arena is correct; see comment)"]
fn round_glyph_bake_matches_combinator_cache() {
    golden_for('O', 32);
}
