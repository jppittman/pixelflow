//! Headless demo: bake a text run through the Kernel-based glyph pipeline
//! and write it out as a PNG.
//!
//! This exercises the exact path the JIT-first font rewrite added:
//! `text()` composes one fused coverage `Kernel` for the whole string,
//! `Lattice::bake` JIT-compiles it once and tabulates coverage — no
//! combinator scene graph, no jet domain, antialiasing from symbolic `Dwrt`.
//!
//! Run: `cargo run -p pixelflow-graphics --example font_demo -- out.png`

use pixelflow_core::Lattice;
use pixelflow_graphics::fonts::{text, Font};

const FONT_BYTES: &[u8] = include_bytes!("../assets/DejaVuSansMono-Fallback.ttf");

fn main() {
    let out_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "font_demo.png".to_string());

    let font = Font::parse(FONT_BYTES).expect("parse font");
    let message = "pixelflow: fonts are Kernels now";
    let size = 32.0f32;

    let kernel = text(&font, message, size);

    let width = 620u32;
    let height = (size * 1.4) as u32;
    let baked = Lattice {
        extent: [width, height, 1, 1],
        origin: [0.5, 0.5, 0.0, 0.0],
    }
    .bake(&kernel);

    let coverage = baked.buffer();
    println!(
        "baked {}x{} texels, ink sum = {:.1}",
        width,
        height,
        coverage.iter().sum::<f32>()
    );

    // Coverage -> grayscale-on-black RGBA8: bright text on a dark background.
    let mut rgba = Vec::with_capacity(coverage.len() * 4);
    for &c in coverage {
        let v = (c.clamp(0.0, 1.0) * 255.0).round() as u8;
        rgba.extend_from_slice(&[v, v, v, 255]);
    }

    write_png(&out_path, width, height, &rgba);
    println!("wrote {out_path}");
}

/// Minimal PNG encoder: 8-bit RGBA, no external deps. Uses uncompressed
/// ("stored") DEFLATE blocks (valid per RFC 1951 — compression is optional),
/// so no zlib dependency is needed for a demo-scale image.
fn write_png(path: &str, width: u32, height: u32, rgba: &[u8]) {
    let mut png = Vec::new();
    png.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);

    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]); // 8-bit depth, RGBA, defaults
    write_chunk(&mut png, b"IHDR", &ihdr);

    // Raw scanlines: filter-type byte (0 = None) + RGBA row, per PNG spec.
    let stride = width as usize * 4;
    let mut raw = Vec::with_capacity((stride + 1) * height as usize);
    for row in rgba.chunks_exact(stride) {
        raw.push(0);
        raw.extend_from_slice(row);
    }

    write_chunk(&mut png, b"IDAT", &zlib_store(&raw));
    write_chunk(&mut png, b"IEND", &[]);

    std::fs::write(path, png).expect("write png");
}

fn write_chunk(out: &mut Vec<u8>, tag: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let start = out.len();
    out.extend_from_slice(tag);
    out.extend_from_slice(data);
    out.extend_from_slice(&crc32(&out[start..]).to_be_bytes());
}

/// Wrap `data` in a minimal zlib stream using uncompressed DEFLATE blocks
/// (max 65535 bytes each).
fn zlib_store(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0x78, 0x01]; // zlib header: CMF/FLG, no dict, default level
    for chunk in data.chunks(65535) {
        let is_final = std::ptr::eq(
            chunk.as_ptr().wrapping_add(chunk.len()),
            data.as_ptr().wrapping_add(data.len()),
        );
        out.push(u8::from(is_final)); // BFINAL bit in bit 0, BTYPE=00 (stored)
        let len = chunk.len() as u16;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes());
        out.extend_from_slice(chunk);
    }
    out.extend_from_slice(&adler32(data).to_be_bytes());
    out
}

fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in data {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}
