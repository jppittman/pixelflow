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

/// A glyph's coverage mask is the control: its `If`s have arms under the
/// mispredict bound, and a branch there costs more than the blend it replaces
/// (a coverage mask measured 3.6x slower guarded). A compile that guarded one
/// would draw the same glyph, slower — the opposite failure to the chrome
/// sphere's, which `render::packed`'s pins hold.
#[test]
fn a_glyph_earns_no_branch() {
    for (ch, px) in [('@', 16), ('8', 32), ('O', 32)] {
        assert_eq!(branches_of(ch, px), (0, 0), "{ch} at {px}px took a branch");
    }
}
