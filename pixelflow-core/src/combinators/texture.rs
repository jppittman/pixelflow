//! # Texture Combinator
//!
//! A manifold that samples from a backing memory slice.
//! This enables query caching and texture lookups.

use crate::{Field, Manifold, PARALLELISM};
use alloc::vec::Vec;

type Field4 = (Field, Field, Field, Field);

/// Evaluate a manifold graph to Field.
#[inline(always)]
fn eval<M: Manifold<Field4, Output = Field>>(m: M) -> Field {
    let zero = Field::from(0.0);
    m.eval((zero, zero, zero, zero))
}

/// A manifold backed by a 2D texture in memory.
///
/// `Texture` wraps a slice of f32 values and serves queries by
/// indexing into that memory. Coordinates are mapped to indices
/// via floor + clamp (or wrap for tiled textures).
///
/// This is the foundational primitive for:
/// - Glyph caching (bake SDF once, sample many times)
/// - Texture mapping
/// - Lookup tables (LUTs)
///
/// # Example
///
/// ```ignore
/// // Create a 64x64 texture
/// let data: Vec<f32> = (0..64*64).map(|i| i as f32 / 4096.0).collect();
/// let tex = Texture::new(data, 64, 64);
///
/// // Sample at coordinates - returns cached values
/// let val = tex.eval((x, y, z, w));
/// ```
pub struct Texture {
    /// Row-major f32 data.
    data: Vec<f32>,
    /// Width in pixels.
    width: usize,
    /// Height in pixels.
    height: usize,
}

impl Texture {
    /// Create a new texture from data.
    ///
    /// # Panics
    /// Panics if `data.len() != width * height`.
    #[must_use]
    pub fn new(data: Vec<f32>, width: usize, height: usize) -> Self {
        assert_eq!(data.len(), width * height, "texture size mismatch");
        Self {
            data,
            width,
            height,
        }
    }

    /// Create a texture by sampling a manifold.
    ///
    /// Evaluates `source` at each pixel center `(x + 0.5, y + 0.5)`.
    pub fn from_manifold<M>(source: &M, width: usize, height: usize) -> Self
    where
        M: Manifold<Field4, Output = Field>,
    {
        let mut data = Vec::with_capacity(width * height);
        let mut buf = [0.0f32; PARALLELISM];

        for y in 0..height {
            let fy = Field::from(y as f32 + 0.5);
            for x in 0..width {
                let fx = Field::from(x as f32 + 0.5);
                let val = source.eval((fx, fy, Field::from(0.0), Field::from(0.0)));
                // All lanes have same value for splat input, extract first
                val.store(&mut buf);
                data.push(buf[0]);
            }
        }

        Self {
            data,
            width,
            height,
        }
    }

    /// Get the texture width.
    #[inline]
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Get the texture height.
    #[inline]
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Get the backing data.
    #[inline]
    #[must_use]
    pub fn data(&self) -> &[f32] {
        &self.data
    }
}

impl Manifold<Field4> for Texture {
    type Output = Field;

    #[inline(always)]
    fn eval(&self, p: Field4) -> Field {
        let (x, y, _z, _w) = p;
        // Compute linear indices: floor(y) * width + floor(x)
        let w = Field::from(self.width as f32);
        let zero = Field::from(0.0);
        let max_x = Field::from((self.width - 1) as f32);
        let max_y = Field::from((self.height - 1) as f32);

        // Optimization: x.floor() is redundant because Field::gather implicitly truncates indices.
        // For x (minor dimension), truncation is sufficient as long as we clamp first.
        let x_idx = x.max(zero).min(max_x);

        // For y (major dimension), we MUST floor explicitly because multiplying by width
        // would magnify fractional parts (e.g. 1.9 * 64 = 121.6 -> row 1, pixel 57 vs row 1, pixel 0).
        let y_idx = y.floor().max(zero).min(max_y);

        // Linear index = y * width + x
        let indices = eval(y_idx * w + x_idx);

        Field::gather(&self.data, indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PARALLELISM;

    #[test]
    fn test_texture_creation() {
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let tex = Texture::new(data, 4, 4);
        assert_eq!(tex.width(), 4);
        assert_eq!(tex.height(), 4);
    }

    #[test]
    fn test_texture_sample() {
        // Create 4x4 texture with values 0-15
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let tex = Texture::new(data, 4, 4);

        // Sample at (0.5, 0.5) should give value at index 0
        let val = tex.eval((
            Field::from(0.5),
            Field::from(0.5),
            Field::from(0.0),
            Field::from(0.0),
        ));
        let mut buf = [0.0f32; PARALLELISM];
        val.store(&mut buf);
        assert!((buf[0] - 0.0).abs() < 0.01);

        // Sample at (1.5, 0.5) should give value at index 1
        let val = tex.eval((
            Field::from(1.5),
            Field::from(0.5),
            Field::from(0.0),
            Field::from(0.0),
        ));
        val.store(&mut buf);
        assert!((buf[0] - 1.0).abs() < 0.01);

        // Sample at (0.5, 1.5) should give value at index 4 (row 1, col 0)
        let val = tex.eval((
            Field::from(0.5),
            Field::from(1.5),
            Field::from(0.0),
            Field::from(0.0),
        ));
        val.store(&mut buf);
        assert!((buf[0] - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_texture_clamping() {
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let tex = Texture::new(data, 4, 4);

        // Sample at negative coords should clamp to 0
        let val = tex.eval((
            Field::from(-1.0),
            Field::from(0.5),
            Field::from(0.0),
            Field::from(0.0),
        ));
        let mut buf = [0.0f32; PARALLELISM];
        val.store(&mut buf);
        assert!((buf[0] - 0.0).abs() < 0.01);

        // Sample at coords > width should clamp to edge
        let val = tex.eval((
            Field::from(10.0),
            Field::from(0.5),
            Field::from(0.0),
            Field::from(0.0),
        ));
        val.store(&mut buf);
        assert!((buf[0] - 3.0).abs() < 0.01); // Last column of row 0
    }
}
