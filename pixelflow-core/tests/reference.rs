//! Scalar reference implementations for validating SIMD operations.
//!
//! These are intentionally naive, simple implementations that prioritize
//! correctness over performance. They serve as ground truth for testing.

/// Unpack 4-bit packed grayscale data to 8-bit (scalar reference).
///
/// Each byte contains two pixels: high nibble first, low nibble second.
/// Expansion: nibble * 17 maps [0-15] to [0-255].
#[must_use]
pub fn ref_unpack_4bit(packed: &[u8], width: usize, height: usize) -> Vec<u8> {
    let pixel_count = width * height;
    let mut result = Vec::with_capacity(pixel_count);

    for byte in packed {
        let high_nibble = (byte >> 4) & 0x0F;
        let low_nibble = byte & 0x0F;

        result.push(high_nibble * 17);
        result.push(low_nibble * 17);

        if result.len() >= pixel_count {
            break;
        }
    }

    result.truncate(pixel_count);
    result
}

/// Gather a single pixel from 4-bit packed data (scalar reference).
#[allow(dead_code)]
fn read_4bpp_pixel(packed: &[u8], x: usize, y: usize, stride: usize) -> u8 {
    ref_gather_4bit_single(packed, x, y, stride)
}

#[must_use]
pub fn ref_gather_4bit_single(packed: &[u8], x: usize, y: usize, stride: usize) -> u8 {
    let byte_idx = y * stride + (x / 2);
    let is_odd = x % 2 == 1;

    let byte = packed[byte_idx];
    let nibble = if is_odd {
        byte & 0x0F
    } else {
        (byte >> 4) & 0x0F
    };

    nibble * 17
}

/// Bilinear interpolation (scalar float reference).
///
/// Samples from a 2x2 neighborhood with fractional coordinates.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn ref_bilinear_interpolate(p00: u8, p10: u8, p01: u8, p11: u8, dx: f32, dy: f32) -> u8 {
    let top = p00 as f32 * (1.0 - dx) + p10 as f32 * dx;
    let bottom = p01 as f32 * (1.0 - dx) + p11 as f32 * dx;
    let result = top * (1.0 - dy) + bottom * dy;
    result.round().clamp(0.0, 255.0) as u8
}

/// Sample from 4-bit packed image with bilinear filtering (scalar reference).
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn ref_sample_4bit_bilinear(
    packed: &[u8],
    width: usize,
    height: usize,
    stride: usize,
    u: f32,
    v: f32,
) -> u8 {
    // Clamp coordinates to valid range
    let u = u.max(0.0).min((width - 1) as f32);
    let v = v.max(0.0).min((height - 1) as f32);

    let x0 = u.floor() as usize;
    let y0 = v.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);

    let dx = u - x0 as f32;
    let dy = v - y0 as f32;

    let p00 = ref_gather_4bit_single(packed, x0, y0, stride);
    let p10 = ref_gather_4bit_single(packed, x1, y0, stride);
    let p01 = ref_gather_4bit_single(packed, x0, y1, stride);
    let p11 = ref_gather_4bit_single(packed, x1, y1, stride);

    ref_bilinear_interpolate(p00, p10, p01, p11, dx, dy)
}

/// Alpha blend two ARGB colors (scalar reference).
///
/// Formula: (fg * alpha + bg * (256 - alpha)) >> 8
#[must_use]
pub fn ref_blend_alpha_argb(fg: u32, bg: u32, alpha: u32) -> u32 {
    let a_fg = ((fg >> 24) & 0xFF) as u16;
    let r_fg = ((fg >> 16) & 0xFF) as u16;
    let g_fg = ((fg >> 8) & 0xFF) as u16;
    let b_fg = (fg & 0xFF) as u16;

    let a_bg = ((bg >> 24) & 0xFF) as u16;
    let r_bg = ((bg >> 16) & 0xFF) as u16;
    let g_bg = ((bg >> 8) & 0xFF) as u16;
    let b_bg = (bg & 0xFF) as u16;

    let a_alpha = ((alpha >> 24) & 0xFF) as u16;
    let r_alpha = ((alpha >> 16) & 0xFF) as u16;
    let g_alpha = ((alpha >> 8) & 0xFF) as u16;
    let b_alpha = (alpha & 0xFF) as u16;

    let inv_a_alpha = 256 - a_alpha;
    let inv_r_alpha = 256 - r_alpha;
    let inv_g_alpha = 256 - g_alpha;
    let inv_b_alpha = 256 - b_alpha;

    let a = ((a_fg * a_alpha + a_bg * inv_a_alpha) >> 8) as u8;
    let r = ((r_fg * r_alpha + r_bg * inv_r_alpha) >> 8) as u8;
    let g = ((g_fg * g_alpha + g_bg * inv_g_alpha) >> 8) as u8;
    let b = ((b_fg * b_alpha + b_bg * inv_b_alpha) >> 8) as u8;

    ((a as u32) << 24) | ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
}

/// Saturating add (scalar reference).
#[must_use]
pub fn ref_saturating_add_u32(a: u32, b: u32) -> u32 {
    a.saturating_add(b)
}

/// Saturating subtract (scalar reference).
#[must_use]
pub fn ref_saturating_sub_u32(a: u32, b: u32) -> u32 {
    a.saturating_sub(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ref_unpack_4bit_works() {
        // Test data: 0x12 = high:1, low:2 -> 17, 34
        let packed = [0x12u8];
        let result = ref_unpack_4bit(&packed, 2, 1);
        assert_eq!(result, vec![17, 34]);
    }

    #[test]
    fn ref_gather_4bit_single_works() {
        // [0x12, 0x34] represents [1*17, 2*17, 3*17, 4*17]
        let packed = [0x12u8, 0x34];
        let stride = 2; // 2 bytes per row, so 4 pixels per row
        assert_eq!(ref_gather_4bit_single(&packed, 0, 0, stride), 17);
        assert_eq!(ref_gather_4bit_single(&packed, 1, 0, stride), 34);
        assert_eq!(ref_gather_4bit_single(&packed, 2, 0, stride), 51);
        assert_eq!(ref_gather_4bit_single(&packed, 3, 0, stride), 68);
    }

    #[test]
    fn ref_bilinear_interpolate_works() {
        // 50% blend horizontally and vertically
        let result = ref_bilinear_interpolate(0, 100, 100, 200, 0.5, 0.5);
        // Average of [0, 100, 100, 200] = 100
        assert!(
            (result as i32 - 100).abs() <= 1,
            "Expected ~100, got {}",
            result
        );
    }

    #[test]
    fn ref_blend_alpha_works() {
        // White fg, black bg, 50% alpha in all channels
        let fg = 0xFF_FF_FF_FF;
        let bg = 0x00_00_00_00;
        let alpha = 0x80_80_80_80;
        let result = ref_blend_alpha_argb(fg, bg, alpha);

        let r = (result & 0xFF) as u8;
        // 255 * 128 / 256 ≈ 127-128
        assert!((r as i32 - 128).abs() <= 2, "Expected ~128, got {}", r);
    }

    #[test]
    fn ref_saturating_operations_works() {
        assert_eq!(ref_saturating_add_u32(u32::MAX, 1), u32::MAX);
        assert_eq!(ref_saturating_sub_u32(0, 1), 0);
        assert_eq!(ref_saturating_add_u32(100, 50), 150);
        assert_eq!(ref_saturating_sub_u32(100, 50), 50);
    }
}
