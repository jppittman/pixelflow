//! JIT-vs-interpreter goldens for real font glyphs.
//!
//! A glyph is one fused coverage `Kernel`; `Lattice::bake` JIT-compiles it.
//! The reference is the IR interpreter (`eval_scalar`) evaluating the SAME
//! lowered arena at every texel — the language's semantic ground truth, used
//! here as a test oracle only (there is no interpreter fallback at runtime).
//! Any disagreement is a JIT miscompile; this suite caught the select-guard
//! mask-ordering hole and the unsorted Clamp decomposition (see
//! pixelflow-ir/tests/spill_pressure.rs for the distilled regressions).

use pixelflow_core::{Kernel, Lattice};
use pixelflow_graphics::fonts::Font;
use pixelflow_ir::backend::emit::lowering::lower_dwrt_owned;
use pixelflow_ir::binding::BindingTable;
use pixelflow_ir::eval_scalar;

const FONT_BYTES: &[u8] = include_bytes!("../assets/DejaVuSansMono-Fallback.ttf");

fn golden_for(ch: char, size: usize) {
    let font = Font::parse(FONT_BYTES).expect("parse font");
    let kernel: Kernel = font
        .glyph_kernel_scaled(ch, size as f32)
        .unwrap_or_else(|| panic!("glyph {ch:?} not found"));

    // JIT: bake over the texel-center lattice.
    let n = size as u32;
    let baked = Lattice {
        extent: [n, n, 1, 1],
        origin: [0.5, 0.5, 0.0, 0.0],
    }
    .bake(&kernel);
    let got = baked.buffer();
    assert_eq!(got.len(), size * size);

    // Reference: the interpreter on the same arena (Dwrt lowered first — the
    // interpreter evaluates the post-calculus program the JIT compiles).
    let (arena, root) = kernel.parts();
    let (lowered, lroot) = lower_dwrt_owned(arena, root).expect("dwrt lowering");

    let mut ink = 0.0f32;
    for j in 0..size {
        for i in 0..size {
            let (x, y) = (i as f32 + 0.5, j as f32 + 0.5);
            let want = eval_scalar(&lowered, lroot, &[x, y, 0.0, 0.0], &BindingTable::empty());
            let jit = got[j * size + i];
            assert!(jit.is_finite(), "{ch}@{size}: non-finite coverage at ({x},{y})");
            assert!(
                (jit - want).abs() < 1e-4,
                "{ch}@{size}: JIT {jit} != interpreter {want} at texel ({i},{j})"
            );
            ink += want;
        }
    }
    // The glyph must actually render ink (guards against a blank golden).
    assert!(ink > 1.0, "glyph {ch:?} is blank (ink={ink})");
}

#[test]
fn simple_glyph_bake_matches_interpreter() {
    // 'A' — a simple glyph (line segments).
    golden_for('A', 32);
}

#[test]
fn round_glyph_bake_matches_interpreter() {
    // 'O' — quadratic Bezier leaves; the glyph that exposed the JIT
    // soundness bugs.
    golden_for('O', 32);
}

#[test]
fn descender_glyph_bake_matches_interpreter() {
    // 'g' — descender, multiple contours, quad-heavy.
    golden_for('g', 32);
}

#[test]
fn double_counter_glyph_bake_matches_interpreter() {
    // '8' — two enclosed counters: the winding sum crosses zero twice per
    // scanline, stressing the sign bookkeeping of the fused contributions.
    golden_for('8', 32);
}
