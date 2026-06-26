//! Tests for parallel rasterization.

use pixelflow_graphics::render::color::{Color, NamedColor, Rgba8};
use pixelflow_graphics::render::frame::Frame;
use pixelflow_graphics::render::rasterizer::rasterize;

#[test]
fn test_parallel_rasterization_matches_sequential() {
    let width = 100;
    let height = 100;
    let color = Color::Named(NamedColor::Green);

    // Sequential rendering
    let mut seq_frame = Frame::<Rgba8>::new(width as u32, height as u32);
    rasterize(&color, &mut seq_frame, 1);

    // Parallel rendering with 4 threads
    let mut par_frame = Frame::<Rgba8>::new(width as u32, height as u32);
    rasterize(&color, &mut par_frame, 4);

    // Results should be identical
    assert_eq!(seq_frame.data, par_frame.data);
}
