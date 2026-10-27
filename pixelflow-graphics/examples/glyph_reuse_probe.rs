//! Cold / zoom / re-warm cost of warming printable ASCII into a `GlyphAtlas`,
//! the measurement behind `docs/results/2026-10-26-font-reuse-policy.md`.
//!
//! Run: `cargo run --release -p pixelflow-graphics --example glyph_reuse_probe -- [density]`

use pixelflow_graphics::fonts::{Font, GlyphAtlas};
use std::time::Instant;
fn main() {
    let font = Font::parse(include_bytes!("../assets/DejaVuSansMono-Fallback.ttf")).unwrap();
    let density: f32 = std::env::args().nth(1).map_or(1.0, |s| s.parse().unwrap());
    let t = Instant::now();
    let mut a = GlyphAtlas::new(16.0, density, 128);
    a.warm(&font, ' '..='~');
    let cold = t.elapsed();
    let t = Instant::now();
    let mut b = GlyphAtlas::new(16.0, density * 1.125, 128);
    b.warm(&font, ' '..='~');
    let zoom = t.elapsed();
    let t = Instant::now();
    let mut c = GlyphAtlas::new(16.0, density, 128);
    c.warm(&font, ' '..='~');
    let rewarm = t.elapsed();
    println!(
        "cold {:.1} ms  zoom {:.1} ms  rewarm {:.1} ms",
        cold.as_secs_f64() * 1e3,
        zoom.as_secs_f64() * 1e3,
        rewarm.as_secs_f64() * 1e3
    );
}
