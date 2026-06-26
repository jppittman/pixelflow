//! # Baked Combinator
//!
//! Caches a color manifold's results to memory.
//! Queries are served from the cache with proper SIMD sampling.
//!
//! This is the foundation for glyph caching - bake once, sample many.

use crate::render::color::Pixel;
use crate::render::frame::Frame;
use crate::render::rasterizer::execute;
use pixelflow_core::{Discrete, Field, Manifold, ManifoldCompat, Texture};

/// The standard 4D Field domain type.
type Field4 = (Field, Field, Field, Field);

/// A color manifold baked to a texture cache.
///
/// `Baked` pre-computes a color manifold at a given resolution and serves
/// subsequent queries via SIMD-friendly texture sampling. It stores
/// separate f32 textures for each RGBA channel.
///
/// # Example
///
/// ```ignore
/// use pixelflow_graphics::{Baked, Rgba8};
///
/// // Bake an expensive gradient at 256x256
/// let cached: Baked<_, Rgba8> = Baked::new(expensive_gradient, 256, 256);
///
/// // Now queries sample from the cache with SIMD
/// let color = cached.eval_raw(x, y, z, w);
/// ```
pub struct Baked<M, P: Pixel> {
    /// Red channel texture.
    r: Texture,
    /// Green channel texture.
    g: Texture,
    /// Blue channel texture.
    b: Texture,
    /// Alpha channel texture.
    a: Texture,
    /// Cache dimensions.
    width: usize,
    height: usize,
    /// The original manifold.
    #[allow(dead_code)]
    inner: M,
    /// Pixel type marker.
    _pixel: core::marker::PhantomData<P>,
}

impl<M, P> Baked<M, P>
where
    M: Manifold<Field4, Output = Discrete>,
    P: Pixel,
{
    /// Bake a color manifold to textures at the given resolution.
    ///
    /// Eagerly rasterizes the manifold, then extracts RGBA channels
    /// into separate f32 textures for SIMD-friendly sampling.
    pub fn new(source: M, width: usize, height: usize) -> Self {
        // First rasterize to a Frame
        let mut frame = Frame::<P>::new(width as u32, height as u32);
        execute(&source, &mut frame);

        // Extract channels to separate f32 buffers
        let mut r_data = Vec::with_capacity(width * height);
        let mut g_data = Vec::with_capacity(width * height);
        let mut b_data = Vec::with_capacity(width * height);
        let mut a_data = Vec::with_capacity(width * height);

        for pixel in &frame.data {
            let packed = pixel.to_u32();
            let [r, g, b, a] = packed.to_le_bytes();
            r_data.push(r as f32 / 255.0);
            g_data.push(g as f32 / 255.0);
            b_data.push(b as f32 / 255.0);
            a_data.push(a as f32 / 255.0);
        }

        Self {
            r: Texture::new(r_data, width, height),
            g: Texture::new(g_data, width, height),
            b: Texture::new(b_data, width, height),
            a: Texture::new(a_data, width, height),
            width,
            height,
            inner: source,
            _pixel: core::marker::PhantomData,
        }
    }

    /// Get the cache width.
    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Get the cache height.
    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Get the red channel texture.
    #[inline]
    pub fn red(&self) -> &Texture {
        &self.r
    }

    /// Get the green channel texture.
    #[inline]
    pub fn green(&self) -> &Texture {
        &self.g
    }

    /// Get the blue channel texture.
    #[inline]
    pub fn blue(&self) -> &Texture {
        &self.b
    }

    /// Get the alpha channel texture.
    #[inline]
    pub fn alpha(&self) -> &Texture {
        &self.a
    }
}

impl<M, P> Manifold<Field4> for Baked<M, P>
where
    M: Manifold<Field4, Output = Discrete> + Send + Sync,
    P: Pixel + Send + Sync,
{
    type Output = Discrete;

    #[inline(always)]
    fn eval(&self, p: Field4) -> Discrete {
        let (x, y, z, w) = p;
        // Sample each channel texture (proper SIMD gather)
        let r = self.r.eval_raw(x, y, z, w);
        let g = self.g.eval_raw(x, y, z, w);
        let b = self.b.eval_raw(x, y, z, w);
        let a = self.a.eval_raw(x, y, z, w);

        // Pack into Discrete
        Discrete::pack(r, g, b, a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Rgba8;
    use pixelflow_core::{materialize_discrete, Discrete, ManifoldExt, PARALLELISM};

    // A simple solid color manifold for testing
    struct SolidColor(u8, u8, u8, u8);

    impl Manifold<Field4> for SolidColor {
        type Output = Discrete;

        fn eval(&self, _p: Field4) -> Discrete {
            Discrete::pack(
                Field::from(self.0 as f32 / 255.0),
                Field::from(self.1 as f32 / 255.0),
                Field::from(self.2 as f32 / 255.0),
                Field::from(self.3 as f32 / 255.0),
            )
        }
    }

    #[test]
    fn test_baked_creation() {
        let color = SolidColor(255, 128, 64, 255);
        let baked: Baked<_, Rgba8> = Baked::new(color, 8, 8);

        assert_eq!(baked.width(), 8);
        assert_eq!(baked.height(), 8);
    }

    #[test]
    fn test_baked_eval() {
        let color = SolidColor(255, 0, 0, 255);
        let baked: Baked<_, Rgba8> = Baked::new(color, 4, 4);

        // Sample should return red
        let mut buf = [0u32; PARALLELISM];
        materialize_discrete(&baked, 0.5, 0.5, &mut buf);

        let [r, g, b, a] = buf[0].to_le_bytes();
        assert_eq!(r, 255);
        assert_eq!(g, 0);
        assert_eq!(b, 0);
        assert_eq!(a, 255);
    }

    #[test]
    fn test_baked_sequential_sampling() {
        // Create a gradient: red increases with x
        struct Gradient;
        impl Manifold<Field4> for Gradient {
            type Output = Discrete;
            fn eval(&self, p: Field4) -> Discrete {
                let (x, _y, _z, _w) = p;
                // Red channel = x / 4 (for 4 wide texture)
                let r = (x / Field::from(4.0)).constant();
                Discrete::pack(r, Field::from(0.0), Field::from(0.0), Field::from(1.0))
            }
        }

        let baked: Baked<_, Rgba8> = Baked::new(Gradient, 4, 4);

        // Sample at x=0.5 should give red ~= 0.125 (pixel 0 sampled at 0.5)
        // Sample at x=1.5 should give red ~= 0.375 (pixel 1 sampled at 1.5)
        let mut buf = [0u32; PARALLELISM];
        materialize_discrete(&baked, 0.5, 0.5, &mut buf);

        // First pixel (x=0.5) should have low red
        let [r0, _, _, _] = buf[0].to_le_bytes();
        assert!(r0 < 50, "expected low red at x=0, got {}", r0);

        // Second pixel (x=1.5) should have higher red
        if PARALLELISM > 1 {
            let [r1, _, _, _] = buf[1].to_le_bytes();
            assert!(r1 > r0, "expected red to increase with x");
        }
    }
}
