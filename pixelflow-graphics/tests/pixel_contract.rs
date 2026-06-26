//! Tests for pixel types.
//!
//! These tests verify the Rgba8 and Bgra8 pixel types work correctly.

use pixelflow_graphics::render::color::{Bgra8, Rgba8};
use pixelflow_graphics::render::pixel::Pixel;

#[test]
fn test_rgba8_components() {
    let rgba = Rgba8::new(0x11, 0x22, 0x33, 0x44);
    assert_eq!(rgba.r(), 0x11);
    assert_eq!(rgba.g(), 0x22);
    assert_eq!(rgba.b(), 0x33);
    assert_eq!(rgba.a(), 0x44);
}

#[test]
fn test_bgra8_components() {
    let bgra = Bgra8::new(0x33, 0x22, 0x11, 0x44);
    assert_eq!(bgra.b(), 0x33);
    assert_eq!(bgra.g(), 0x22);
    assert_eq!(bgra.r(), 0x11);
    assert_eq!(bgra.a(), 0x44);
}

#[test]
fn test_swizzle_correctness() {
    let rgba = Rgba8::new(0x11, 0x22, 0x33, 0x44);
    let bgra = Bgra8::from(rgba);

    assert_eq!(bgra.r(), 0x11);
    assert_eq!(bgra.g(), 0x22);
    assert_eq!(bgra.b(), 0x33);
    assert_eq!(bgra.a(), 0x44);
}

#[test]
fn test_pixel_from_rgba_f32() {
    let rgba = Rgba8::from_rgba(1.0, 0.5, 0.0, 1.0);
    assert_eq!(rgba.r(), 255);
    assert!(rgba.g() > 120 && rgba.g() < 140); // ~127
    assert_eq!(rgba.b(), 0);
    assert_eq!(rgba.a(), 255);
}

#[test]
fn test_pixel_from_u32() {
    let packed: u32 = 0xFF332211; // ABGR in memory = RGBA little-endian
    let rgba = Rgba8::from_u32(packed);
    assert_eq!(rgba.to_u32(), packed);
}
