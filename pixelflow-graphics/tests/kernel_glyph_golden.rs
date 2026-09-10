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
use pixelflow_ir::binding::BindingTable;
use pixelflow_ir::eval_scalar;
use pixelflow_ir::expr::Term;
use pixelflow_ir::passes::lower_dwrt;

const FONT_BYTES: &[u8] = include_bytes!("../assets/DejaVuSansMono-Fallback.ttf");

fn golden_for(ch: char, size: usize) {
    let font = Font::parse(FONT_BYTES).expect("parse font");
    let kernel: Kernel = font
        .glyph_kernel_scaled(ch, size as f32)
        .unwrap_or_else(|| panic!("glyph {ch:?} not found"));

    // JIT: bake over the texel-center convention, a contramap over a plain
    // index lattice.
    let n = size as u32;
    let centered = kernel.at(
        &Kernel::x().add(&Kernel::constant(0.5)),
        &Kernel::y().add(&Kernel::constant(0.5)),
    );
    let baked = Lattice { extent: [n, n] }.bake(&centered);
    let got = baked.buffer();
    assert_eq!(got.len(), size * size);

    // Reference: the interpreter on the same arena the JIT compiles —
    // `centered`, texel-center contramap included, so the two sides are
    // asked about literal texel indices and the golden pins the COMPILER
    // (JIT vs interpreter on one program), not the contramap arithmetic.
    // Bake runs the runtime e-graph tier (`optimize_runtime_arena`, which
    // lowers Dwrt itself) before compiling, so the oracle interprets that
    // same optimized arena; optimization soundness (optimized vs raw, within
    // float-reassociation tolerance) is pinned separately by
    // tests/kernel_glyph_optimize.rs.
    let term = centered.term();
    let optimized = pixelflow_search::runtime::optimize_runtime_term(
        term,
        pixelflow_ir::LatticeShape::new([size as u32, size as u32]),
    );
    let (lowered, lowered_env) = match optimized.as_deref() {
        Some((rooted, env)) => (rooted.clone(), env.clone()),
        None => (lower_dwrt(term).expect("dwrt lowering"), term.env().clone()),
    };
    let lowered_term = Term::new(lowered.entry(), &lowered_env);

    let mut ink = 0.0f32;
    for j in 0..size {
        for i in 0..size {
            let (x, y) = (i as f32, j as f32);
            let want = eval_scalar(lowered_term, &[x, y], &BindingTable::empty());
            let jit = got[j * size + i];
            assert!(
                jit.is_finite(),
                "{ch}@{size}: non-finite coverage at ({x},{y})"
            );
            // 1e-3, not 1e-4: the optimized arena contains `MulAdd` (e-graph
            // FMA fusion), whose rounding is documented platform-specific —
            // the interpreter's reference `mul_add` rounds once, a build
            // without hardware FMA rounds twice (see CLAUDE.md's FP table).
            // Winding sums amplify that last-bit divergence to ~1e-4; a real
            // miscompile shifts coverage by O(1).
            assert!(
                (jit - want).abs() < 1e-3,
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
