//! **A glyph's compile keeps the branches it earns.**
//!
//! A guarded `If` and a blended one cover the same texels, so the glyph
//! goldens cannot see a branch lost or gained; they only see pixels. A glyph
//! branches over exactly three arms — the ones that own a loop over its
//! pieces, so a texel outside the glyph skips the loop — and a lost arm is
//! that loop back on every texel, 5-16x slower per texel
//! (docs/results/2026-10-03-guard-structure-baseline.md).
//!
//! The guard tables are the emitter's and nothing a caller holds names them;
//! the code is what the compile hands back. So these pin the code itself, per
//! tier, through `jit_cache::compile` — the entry a bake compiles by — for the
//! crate's own font, at the sizes the atlas bakes. The pin moves with any
//! change to the emitted code, deliberately: a change that meant to move it
//! re-pins from the values the failure prints, after checking the glyphs'
//! branches with the emitter's census and their bake throughput.

use pixelflow_codegen::isa::Isa;
use pixelflow_codegen::{fnv1a64, isa, jit_cache};
use pixelflow_core::Kernel;
use pixelflow_graphics::fonts::Font;
use pixelflow_ir::LatticeShape;

const FONT_DATA: &[u8] = include_bytes!("../assets/DejaVuSansMono-Fallback.ttf");

/// One kernel's code on each tier, `(tier, bytes, fnv1a64)`. A tier is pinned
/// under the pipeline it compiles with by default; the other, which
/// `PIXELFLOW_CODEGEN` asks for, is a tier of its own, `+legacy` after the ISA's
/// name: the same glyphs, other code.
type TierPins = [(&'static str, usize, u64); 4];

/// `(glyph, px, pins)`: each glyph's code per tier.
const PINS: [(char, u32, TierPins); 3] = [
    (
        '@',
        16,
        [
            ("avx2", 2836, 0x30fd_ab98_f7df_3b32),
            ("avx512", 3172, 0x2d16_3a20_0ca5_5eee),
            ("neon", 1904, 0x8b99_0dca_5fa6_15da),
            ("avx2+legacy", 3012, 0x2a8c_6135_0420_6970),
        ],
    ),
    (
        '8',
        32,
        [
            ("avx2", 2832, 0x9335_b8d3_4de6_2c38),
            ("avx512", 3172, 0x7939_6944_482a_feeb),
            ("neon", 1904, 0x113e_3102_6b9b_2b3e),
            ("avx2+legacy", 3008, 0x3c1c_4b85_1887_fa8a),
        ],
    ),
    (
        'O',
        32,
        [
            ("avx2", 2836, 0x56b2_1cc7_6344_6fd3),
            ("avx512", 3172, 0xd675_74c3_f870_26e3),
            ("neon", 1888, 0xc973_f004_e574_5a98),
            ("avx2+legacy", 3012, 0x0d3a_e133_99a6_af0d),
        ],
    ),
];

/// `(bytes, fnv1a64)` of `ch` at `px`, compiled at its own tile with the
/// texel-centre warp the atlas bakes under.
fn code_of(font: &Font<'_>, ch: char, px: u32) -> (usize, u64) {
    let glyph = font
        .glyph_kernel_scaled(ch, px as f32)
        .expect("a glyph the font has");
    let half = Kernel::constant(0.5);
    let warped = glyph
        .kernel()
        .at(&Kernel::x().add(&half), &Kernel::y().add(&half));
    let linked = jit_cache::compile(&warped, LatticeShape::new([px, px])).expect("compile");
    let code = linked.kernel.as_bytes();
    (code.len(), fnv1a64(code))
}

#[test]
fn a_glyphs_code_is_pinned() {
    let font = Font::parse(FONT_DATA).expect("parse font");
    let host = isa::detect();
    let default = match host {
        Isa::Avx2 => "selection",
        Isa::Avx512 | Isa::Neon => "legacy",
    };
    let tier = match std::env::var("PIXELFLOW_CODEGEN") {
        Ok(knob) if !knob.trim().eq_ignore_ascii_case(default) => {
            format!("{}+{}", host.name(), knob.trim().to_ascii_lowercase())
        }
        _ => host.name().to_string(),
    };
    let mut moved = Vec::new();
    for (ch, px, pins) in PINS {
        let emitted = code_of(&font, ch, px);
        let pinned = pins
            .iter()
            .find(|(t, _, _)| *t == tier)
            .map(|&(_, len, fnv)| (len, fnv));
        let (len, fnv) =
            pinned.unwrap_or_else(|| panic!("{ch} at {px}px has no pin for the {tier} tier"));
        if emitted != (len, fnv) {
            moved.push(format!(
                "{ch} at {px}px on {tier}: pinned ({len}, {fnv:#018x}), emitted ({}, {:#018x})",
                emitted.0, emitted.1
            ));
        }
    }
    assert!(moved.is_empty(), "glyph code moved:\n{}", moved.join("\n"));
}
