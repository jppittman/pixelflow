//! Byte-identity fixture for docs/plans/2026-09-09-exprarena-on-dag.md's
//! "Demand moves to the IR" work: run on the base commit and on this
//! branch's head, then diff stdout. Any difference means a guard decision
//! or the emitted code changed, which that section of the plan forbids.
//!
//! Fixtures:
//! - A guarded `Select` kernel (mirrors
//!   `pixelflow-codegen/tests/collapse_paths.rs`'s
//!   `a_select_blends_and_branches`), at a shape wide enough for the guard
//!   to actually fire.
//! - The glyph `8` at 32px, composed with `.at(x+0.5, y+0.5)` — the exact
//!   fixture docs/plans/2026-09-09-exprarena-on-dag.md's "Demand moves to
//!   the IR" section names — compiled through the same production path
//!   (`pixelflow_codegen::jit_cache::compile`) a real bake uses.
//!
//! ```bash
//! cargo run -p pixelflow-graphics --example demand_move_byte_check
//! ```

use pixelflow_codegen::jit_cache;
use pixelflow_graphics::fonts::Font;
use pixelflow_ir::{Kernel, LatticeShape};

const FONT_DATA: &[u8] = include_bytes!("../assets/DejaVuSansMono-Fallback.ttf");

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// `Select(x < y, sqrt(x), abs(y))`: a guardable select whose arms are each
/// cheap enough to schedule but distinct in cost, so a real branch is on
/// the table for the emitter to take or refuse.
fn guarded_select_kernel() -> Kernel {
    let x = Kernel::x();
    let y = Kernel::y();
    let mask = x.lt(&y);
    let true_arm = x.sqrt();
    let false_arm = y.abs();
    mask.select(&true_arm, &false_arm)
}

fn report(label: &str, kernel: &Kernel, shape: LatticeShape) {
    let linked = jit_cache::compile(kernel, shape).expect("compile");
    let bytes = linked.kernel.code_bytes();
    println!(
        "{label}: shape={:?} len={} fnv1a64={:016x}",
        shape,
        bytes.len(),
        fnv1a64(bytes)
    );
}

fn main() {
    report(
        "guarded_select",
        &guarded_select_kernel(),
        LatticeShape::new([64, 64]),
    );

    let font = Font::parse(FONT_DATA).expect("parse font");
    let glyph = font
        .glyph_kernel_scaled('8', 32.0)
        .expect("'8' glyph at 32px");
    let half = Kernel::constant(0.5);
    let x_shifted = Kernel::x().add(&half);
    let y_shifted = Kernel::y().add(&half);
    let warped = glyph.kernel().at(&x_shifted, &y_shifted);
    report("glyph_8_at_32px", &warped, LatticeShape::new([32, 32]));
}
