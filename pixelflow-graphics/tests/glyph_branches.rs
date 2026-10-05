//! **A glyph's compile keeps the branches it earns.**
//!
//! A guarded `If` and a blended one cover the same texels, so the glyph
//! goldens cannot see a branch lost or gained; they only see pixels. These
//! pins read the branches the compile was emitted with
//! (`CompiledKernel::branches`), through `jit_cache::compile` — the entry a
//! bake compiles by — for the crate's own font, at the sizes the atlas
//! bakes.
//!
//! Pinned on guards and arms, not entries: an entry count moves with every
//! rewrite rule, while an arm gained or lost is the failure this exists to
//! catch.

use pixelflow_codegen::jit_cache;
use pixelflow_core::Kernel;
use pixelflow_graphics::fonts::Font;
use pixelflow_ir::LatticeShape;

const FONT_DATA: &[u8] = include_bytes!("../assets/DejaVuSansMono-Fallback.ttf");

/// `(guards, arms branched)` of `ch` at `px`, compiled at its own tile with
/// the texel-centre warp the atlas bakes under.
fn branches_of(ch: char, px: u32) -> (u64, u64) {
    let font = Font::parse(FONT_DATA).expect("parse font");
    let glyph = font
        .glyph_kernel_scaled(ch, px as f32)
        .expect("a glyph the font has");
    let half = Kernel::constant(0.5);
    let warped = glyph
        .kernel()
        .at(&Kernel::x().add(&half), &Kernel::y().add(&half));
    let linked = jit_cache::compile(&warped, LatticeShape::new([px, px])).expect("compile");
    let b = linked.kernel.branches();
    (b.guards, b.arms_branched)
}

/// A glyph branches over exactly three arms, whatever the glyph: the arms
/// that own a loop over its pieces, so a texel outside the glyph skips the loop.
///
/// Each costs thousands of cycles — far past the mispredict bound, which is
/// what keeps the cheap arms (a coverage mask's few ops, measured 3.6x
/// *slower* guarded) out — and none was a branch before the layout chose the
/// order: the old search could not make them one run, so they stayed blended
/// and every texel paid for every piece. The count is the pin: a lost arm is
/// the loop back on every texel (5-16x slower per texel), a gained one is a
/// branch on work too cheap to pay for it. The chrome sphere's pins, in
/// `render::packed`, hold the same property for a scene.
#[test]
fn a_glyph_branches_over_its_piece_loops() {
    for (ch, px) in [('@', 16), ('8', 32), ('O', 32)] {
        assert_eq!(
            branches_of(ch, px),
            (3, 3),
            "{ch} at {px}px: the loops over its pieces are not all behind a branch"
        );
    }
}
