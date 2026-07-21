//! # Font Caching
//!
//! Glyph caching via lattice collapse + bilinear sampling.
//!
//! ## Categorical Semantics
//!
//! Caching is a **morphism between evaluation strategies**:
//! - `Glyph` evaluates mathematically (winding numbers, infinite resolution)
//! - `CachedGlyph` evaluates from a baked lattice (SIMD gather, fixed resolution)
//!
//! Both implement `Manifold`, preserving composition. The cache is a functor
//! that transforms evaluation strategy while maintaining the algebraic
//! interface. The bake evaluates the glyph through [`Antialiased`] (`Jet2`
//! autodiff coordinates, gradient-normalized crossing ramps) and tabulates
//! the result via `Lattice::collapse`; the read-back is `DiscreteManifold`
//! (index) smoothed by the [`Bilinear`] combinator. Texels therefore store
//! *antialiased* coverage — no post-hoc filtering of hard 0/1 samples.
//!
//! ```text
//!            cache_at(size)
//!     Glyph ──────────────► CachedGlyph
//!       │                        │
//!       │ Manifold<I>           │ Manifold<Field>
//!       ▼                        ▼
//!     coverage                 coverage
//! ```
//!
//! ## Coordinate convention (half-pixel)
//!
//! The rasterizer samples pixel *centers*: output pixel `(i, j)` is the
//! manifold evaluated at `(i + 0.5, j + 0.5)` (see `render/rasterizer`).
//! The bake follows the same convention: texel `(i, j)` of the coverage
//! lattice stores the glyph's coverage at continuous coordinate
//! `(i + 0.5, j + 0.5)` (the lattice origin is `(0.5, 0.5)`).
//! `CachedGlyph::eval` shifts incoming coordinates by −0.5 into
//! [`Bilinear`]'s integer texel grid, so a query at a pixel center returns
//! the stored texel exactly — the cached glyph reproduces the analytical
//! antialiased glyph (`Antialiased::new(glyph)`) at pixel centers with no
//! half-pixel shift and no extra blur at the baked size, while fractional
//! positions interpolate smoothly.
//!
//! ## Usage
//!
//! ```ignore
//! use pixelflow_graphics::fonts::{Font, GlyphCache};
//!
//! let font = Font::parse(data).unwrap();
//! let mut cache = GlyphCache::new();
//!
//! // Cache glyphs at specific sizes (happy path: fast)
//! let cached = cache.get(&font, 'A', 16.0);
//!
//! // Arbitrary sizes still work (uncached: infinite resolution)
//! let uncached = font.glyph_scaled('A', 17.3);
//! ```

use crate::render::aa::Antialiased;
use crate::render::bilinear::Bilinear;
use pixelflow_core::jet::Jet2;
use pixelflow_core::{
    At, DiscreteManifold, Field, Lattice, Manifold, ManifoldCompat, ManifoldExt, Select, W, X, Y, Z,
};
use std::collections::HashMap;
use std::sync::Arc;

/// The standard 4D Field domain type.
type Field4 = (Field, Field, Field, Field);

/// The 4D Jet2 domain type (2D autodiff seeded in screen space).
type Jet2x4 = (Jet2, Jet2, Jet2, Jet2);

use super::ttf::{affine, Affine, Font, Glyph, Sum};

/// Offset of a texel's sampling point from its integer index.
///
/// Texel `(i, j)` stores coverage at continuous coordinate
/// `(i + TEXEL_CENTER, j + TEXEL_CENTER)`, matching the rasterizer's
/// pixel-center convention. The bake adds this offset (lattice origin);
/// `CachedGlyph::eval` subtracts it before bilinear sampling.
const TEXEL_CENTER: f32 = 0.5;

// ═══════════════════════════════════════════════════════════════════════════
// CachedGlyph: The Morphism
// ═══════════════════════════════════════════════════════════════════════════

/// A glyph baked to a coverage lattice.
///
/// This is the output of the caching morphism: a glyph that evaluates from
/// memory rather than computing winding numbers. The lattice stores f32
/// *antialiased* coverage values (0.0 to 1.0) — baked through [`Antialiased`],
/// no u8 quantization roundtrip — sampled back via SIMD gather with bilinear
/// interpolation.
///
/// # Resolution
///
/// Unlike raw `Glyph` which has infinite resolution, `CachedGlyph` is baked
/// at a fixed size. For best quality, cache at the exact render size.
/// The cache uses size buckets (multiples of 4) to balance memory with reuse.
#[derive(Clone)]
pub struct CachedGlyph {
    /// Bilinear sampler over the baked coverage lattice.
    /// Arc so cloning a cached glyph (once per cell per frame) is O(1).
    sampler: Arc<Bilinear<DiscreteManifold>>,
    /// Baked width in pixels.
    width: usize,
    /// Baked height in pixels.
    height: usize,
}

impl CachedGlyph {
    /// Create a cached glyph by collapsing a glyph manifold over a lattice.
    ///
    /// The glyph's *antialiased* coverage is tabulated at `size × size`
    /// resolution, at pixel centers (see the module docs for the coordinate
    /// convention). The bake evaluates the glyph through [`Antialiased`], so
    /// every texel stores a gradient-normalized crossing ramp value — the
    /// bilinear read-back then interpolates already-smooth coverage.
    #[must_use]
    pub fn new<L, Q>(glyph: &Glyph<L, Q>, size: usize) -> Self
    where
        Glyph<L, Q>: Manifold<Jet2x4, Output = Field>,
    {
        // Antialiased coverage: seed Jet2 screen-space derivatives so each
        // edge crossing bakes as a ~1px gradient-normalized ramp.
        let coverage = Antialiased::new(glyph);

        // Tabulate at pixel centers: texel (i, j) = coverage(i+0.5, j+0.5).
        let lattice = Lattice {
            extent: [size as u32, size as u32, 1, 1],
            origin: [TEXEL_CENTER, TEXEL_CENTER, 0.0, 0.0],
        };
        let baked = lattice.collapse(&coverage);

        Self {
            sampler: Arc::new(Bilinear::new(baked)),
            width: size,
            height: size,
        }
    }

    /// Get the cache width.
    #[inline]
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Get the cache height.
    #[inline]
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    /// The baked coverage lattice (row-major f32 in `[0, 1]`).
    #[inline]
    #[must_use]
    pub fn coverage(&self) -> &DiscreteManifold {
        &self.sampler.tex
    }
}

impl Manifold<Field4> for CachedGlyph {
    type Output = Field;

    #[inline(always)]
    fn eval(&self, p: Field4) -> Field {
        // Contramap into the sampler's integer texel grid: texel (i, j)
        // holds coverage at pixel center (i + 0.5, j + 0.5), so shift the
        // query by -0.5 before bilinear interpolation.
        let sampled = At {
            inner: &*self.sampler,
            x: X - TEXEL_CENTER,
            y: Y - TEXEL_CENTER,
            z: Z,
            w: W,
        };
        // Bound to the baked extent: `DiscreteManifold` clamps out-of-range
        // indices to the edge texel, which would smear nonzero boundary
        // coverage (e.g. a descender reaching the em-box bottom) to infinity.
        // Outside the bake there is no data — coverage is zero.
        let in_bounds = X.ge(0.0) & X.le(self.width as f32) & Y.ge(0.0) & Y.le(self.height as f32);
        Select {
            cond: in_bounds,
            if_true: sampled,
            if_false: 0.0f32,
        }
        .eval(p)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GlyphCache: The Functor
// ═══════════════════════════════════════════════════════════════════════════

/// Size bucket for cache keys.
///
/// Quantizes sizes to reduce cache entries while maintaining quality.
/// Uses multiples of 4 pixels for SIMD-friendly dimensions.
fn size_bucket(size: f32) -> usize {
    // Round up to next multiple of 4, minimum 8
    let bucket = ((size / 4.0).ceil() as usize) * 4;
    bucket.max(8)
}

/// Key for cached glyphs.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct CacheKey {
    codepoint: u32,
    size_bucket: usize,
}

/// A cache of baked glyphs.
///
/// `GlyphCache` is a functor that transforms `Font × char × size` into
/// `CachedGlyph`. It memoizes the baking operation to avoid redundant
/// rasterization.
///
/// # Size Bucketing
///
/// To balance cache efficiency with quality, sizes are quantized to
/// multiples of 4 pixels. A 17px request uses the 20px bucket.
///
/// # Thread Safety
///
/// The cache is `Send + Sync` but requires `&mut self` for insertion.
/// For concurrent access, wrap in `RwLock` or use per-thread caches.
#[derive(Clone)]
pub struct GlyphCache {
    entries: HashMap<CacheKey, CachedGlyph>,
}

impl GlyphCache {
    /// Create an empty glyph cache.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Create a cache with pre-allocated capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: HashMap::with_capacity(capacity),
        }
    }

    /// Get or create a cached glyph.
    ///
    /// If the glyph at this size bucket is already cached, returns it.
    /// Otherwise, bakes the glyph and caches it.
    pub fn get(&mut self, font: &Font, ch: char, size: f32) -> Option<CachedGlyph> {
        let bucket = size_bucket(size);
        let key = CacheKey {
            codepoint: ch as u32,
            size_bucket: bucket,
        };

        if let Some(cached) = self.entries.get(&key) {
            return Some(cached.clone());
        }

        // Bake the glyph at the bucket size
        let glyph = font.glyph_scaled(ch, bucket as f32)?;
        let cached = CachedGlyph::new(&glyph, bucket);
        self.entries.insert(key, cached.clone());
        Some(cached)
    }

    /// Check if a glyph is cached at this size.
    #[must_use]
    pub fn contains(&self, ch: char, size: f32) -> bool {
        let bucket = size_bucket(size);
        let key = CacheKey {
            codepoint: ch as u32,
            size_bucket: bucket,
        };
        self.entries.contains_key(&key)
    }

    /// Pre-warm the cache with common characters.
    ///
    /// Call this at startup to avoid cache misses during rendering.
    pub fn warm(&mut self, font: &Font, chars: impl IntoIterator<Item = char>, size: f32) {
        for ch in chars {
            self.get(font, ch, size);
        }
    }

    /// Pre-warm with ASCII printable characters.
    pub fn warm_ascii(&mut self, font: &Font, size: f32) {
        self.warm(font, ' '..='~', size);
    }

    /// Clear all cached entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Number of cached glyphs.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Estimated memory usage in bytes.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.entries
            .values()
            .map(|g| g.width * g.height * 4) // f32 per texel
            .sum()
    }
}

impl Default for GlyphCache {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CachedText: Composition with Caching
// ═══════════════════════════════════════════════════════════════════════════

/// A text manifold backed by cached glyphs.
///
/// Like `Text`, but uses `GlyphCache` to avoid recomputing glyphs.
/// For text at cached sizes, this is significantly faster.
///
/// # Example
///
/// ```ignore
/// let mut cache = GlyphCache::new();
/// cache.warm_ascii(&font, 16.0);
///
/// let text = CachedText::new(&font, &mut cache, "Hello", 16.0);
/// execute(&Lift(text), buffer, shape);
/// ```
#[derive(Clone)]
pub struct CachedText {
    /// Composed cached glyphs.
    inner: Sum<Affine<CachedGlyph>>,
}

impl CachedText {
    /// Create cached text from a string.
    ///
    /// Glyphs are retrieved from the cache (or baked on-demand).
    pub fn new(font: &Font, cache: &mut GlyphCache, text: &str, size: f32) -> Self {
        let mut glyphs = Vec::with_capacity(text.len());
        let mut cursor_x = 0.0f32;
        let mut prev_id = None;

        let bucket = size_bucket(size);
        let scale = size / bucket as f32;
        let inv_em = size / font.units_per_em as f32;

        for ch in text.chars() {
            // Single CMAP lookup per character, reused for all operations
            let Some(id) = font.cmap_lookup(ch) else {
                continue;
            };

            // Apply kerning using pre-looked-up glyph IDs
            if let Some(prev) = prev_id {
                cursor_x += font.kern_by_ids(prev, id) * inv_em;
            }

            // Get cached glyph
            if let Some(cached) = cache.get(font, ch, size) {
                // Scale and translate to cursor position
                // The cached glyph is at bucket size, so we need to scale it
                let transform = [scale, 0.0, 0.0, scale, cursor_x, 0.0];
                glyphs.push(affine(cached, transform));
            }

            // Advance cursor using pre-looked-up glyph ID
            if let Some(adv) = font.advance_by_id(id) {
                cursor_x += adv * inv_em;
            }

            prev_id = Some(id);
        }

        Self {
            inner: Sum(glyphs.into()),
        }
    }

    /// Get the composed glyph structure.
    #[must_use]
    pub fn inner(&self) -> &Sum<Affine<CachedGlyph>> {
        &self.inner
    }
}

impl Manifold<Field4> for CachedText {
    type Output = Field;

    #[inline(always)]
    fn eval(&self, p: Field4) -> Field {
        let (x, y, z, w) = p;
        // Sum of cached glyph coverages (just like Text)
        self.inner.eval_raw(x, y, z, w)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // Use the fallback font, which is committed directly (not Git-LFS) so the
    // tests run without `git lfs pull`. NotoSansMono is an LFS pointer.
    const FONT_DATA: &[u8] = include_bytes!("../../assets/DejaVuSansMono-Fallback.ttf");

    /// Evaluate a coverage manifold at a single point via public lattice API.
    fn sample<M>(m: &M, x: f32, y: f32) -> f32
    where
        M: Manifold<Field4, Output = Field>,
    {
        Lattice::point(x, y, 0.0, 0.0).collapse(m).into_buffer()[0]
    }

    /// Center of mass of a row-major coverage grid.
    ///
    /// Panics if the grid has no ink: a blank glyph makes position
    /// comparisons meaningless, and silently passing would hide it.
    fn center_of_mass(buf: &[f32], width: usize) -> (f32, f32) {
        let mut total = 0.0f64;
        let mut mx = 0.0f64;
        let mut my = 0.0f64;
        for (idx, &v) in buf.iter().enumerate() {
            let x = (idx % width) as f64;
            let y = (idx / width) as f64;
            total += v as f64;
            mx += v as f64 * x;
            my += v as f64 * y;
        }
        assert!(total > 0.0, "center_of_mass on a blank coverage grid");
        ((mx / total) as f32, (my / total) as f32)
    }

    #[test]
    fn glyph_cache_buckets_sizes_within_4px_together() {
        let font = Font::parse(FONT_DATA).unwrap();
        let mut cache = GlyphCache::new();

        // 9.0 and 12.0 both round up into the 12px bucket, so caching one
        // makes the cache report the other as already cached.
        cache.get(&font, 'A', 9.0);
        assert_eq!(cache.len(), 1);
        assert!(
            cache.contains('A', 12.0),
            "9.0 and 12.0 should share the 12px bucket"
        );

        // 13.0 rounds up to the next bucket (16px), so it must not collide
        // with the 12px bucket.
        assert!(
            !cache.contains('A', 13.0),
            "13.0 should land in a different bucket than 12.0"
        );

        // Sizes at or below the minimum clamp to the 8px bucket.
        cache.get(&font, 'B', 1.0);
        assert!(
            cache.contains('B', 8.0),
            "sub-minimum sizes should clamp to the 8px bucket"
        );
    }

    #[test]
    fn cached_glyph_creation() {
        let font = Font::parse(FONT_DATA).unwrap();
        let glyph = font.glyph_scaled('A', 32.0).unwrap();
        let cached = CachedGlyph::new(&glyph, 32);

        assert_eq!(cached.width(), 32);
        assert_eq!(cached.height(), 32);
    }

    #[test]
    fn baked_coverage_dimensions_range_and_ink() {
        let font = Font::parse(FONT_DATA).unwrap();
        let glyph = font.glyph_scaled('A', 32.0).unwrap();
        let cached = CachedGlyph::new(&glyph, 32);

        let coverage = cached.coverage();
        assert_eq!(coverage.width(), 32);
        assert_eq!(coverage.height(), 32);

        let buf = coverage.buffer();
        assert_eq!(buf.len(), 32 * 32);

        let mut ink = 0.0f32;
        for (i, &v) in buf.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&v),
                "coverage out of [0,1] at texel {i}: {v}"
            );
            ink += v;
        }
        // 'A' at 32px must put real ink in the interior.
        assert!(
            ink > 10.0,
            "glyph 'A' baked with almost no ink: sum = {ink}"
        );
    }

    #[test]
    fn cached_glyph_matches_analytical_at_pixel_centers() {
        // At pixel centers the bilinear weights vanish, so the cached glyph
        // must reproduce the analytical *antialiased* glyph exactly (up to
        // f32 noise) — the bake stores Antialiased coverage.
        let font = Font::parse(FONT_DATA).unwrap();
        let glyph = font.glyph_scaled('A', 32.0).unwrap();
        let smooth = Antialiased::new(&glyph);
        let cached = CachedGlyph::new(&glyph, 32);

        for &(i, j) in &[(4usize, 4usize), (10, 16), (16, 8), (16, 20), (24, 28)] {
            let (x, y) = (i as f32 + 0.5, j as f32 + 0.5);
            let direct = sample(&smooth, x, y);
            let baked = sample(&cached, x, y);
            assert!(
                (direct - baked).abs() < 1e-5,
                "cached glyph diverges from analytical AA at pixel center ({x}, {y}): \
                 direct {direct}, baked {baked}"
            );
        }
    }

    #[test]
    fn no_half_pixel_shift_center_of_mass() {
        // Regression: the baked glyph must sit at the same position as the
        // analytical glyph rasterized directly at pixel centers. A half-pixel
        // convention error here shows up as a ~0.5px center-of-mass shift.
        let size = 32usize;
        let font = Font::parse(FONT_DATA).unwrap();
        let glyph = font.glyph_scaled('A', size as f32).unwrap();

        // Direct analytical AA tabulation at pixel centers (the rasterizer's
        // sampling convention, without the u8 quantization of the old path).
        // Antialiased to match what the bake stores.
        let direct = Lattice {
            extent: [size as u32, size as u32, 1, 1],
            origin: [0.5, 0.5, 0.0, 0.0],
        }
        .collapse(&Antialiased::new(&glyph));
        let (dx, dy) = center_of_mass(direct.buffer(), size);

        // Cached glyph sampled at pixel centers through the full
        // bake -> bilinear -> half-pixel-shift chain.
        let cached = CachedGlyph::new(&glyph, size);
        let resampled = Lattice {
            extent: [size as u32, size as u32, 1, 1],
            origin: [0.5, 0.5, 0.0, 0.0],
        }
        .collapse(&cached);
        let (cx, cy) = center_of_mass(resampled.buffer(), size);

        assert!(
            (dx - cx).abs() < 0.05 && (dy - cy).abs() < 0.05,
            "center of mass shifted: direct ({dx}, {dy}) vs cached ({cx}, {cy})"
        );
    }

    #[test]
    fn cached_glyph_interpolates_smoothly() {
        // Sampling at sub-texel steps must ramp, not jump: with bilinear
        // filtering a quarter-pixel step can change coverage by at most
        // ~0.25 (texels are in [0,1]); nearest-neighbor jumps by up to 1.0.
        let font = Font::parse(FONT_DATA).unwrap();
        let glyph = font.glyph_scaled('A', 32.0).unwrap();
        let cached = CachedGlyph::new(&glyph, 32);

        let step = 0.25;
        for &y in &[8.5f32, 16.5, 24.5] {
            let mut prev = sample(&cached, 1.0, y);
            let mut x = 1.0 + step;
            while x <= 31.0 {
                let v = sample(&cached, x, y);
                let jump = (v - prev).abs();
                assert!(
                    jump < 0.3,
                    "hard jump {jump} at ({x}, {y}): nearest-neighbor artifact"
                );
                prev = v;
                x += step;
            }
        }
    }

    #[test]
    fn glyph_cache_get() {
        let font = Font::parse(FONT_DATA).unwrap();
        let mut cache = GlyphCache::new();

        // First access should cache
        let cached = cache.get(&font, 'A', 16.0);
        assert!(cached.is_some());
        assert_eq!(cache.len(), 1);

        // Second access should hit cache
        let cached2 = cache.get(&font, 'A', 16.0);
        assert!(cached2.is_some());
        assert_eq!(cache.len(), 1);

        // Different size should create new entry
        let cached3 = cache.get(&font, 'A', 32.0);
        assert!(cached3.is_some());
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn glyph_cache_warm() {
        let font = Font::parse(FONT_DATA).unwrap();
        let mut cache = GlyphCache::new();

        cache.warm_ascii(&font, 16.0);

        // ASCII printable is 95 characters
        assert_eq!(cache.len(), 95);

        // All should be cached now
        assert!(cache.contains('A', 16.0));
        assert!(cache.contains('z', 16.0));
        assert!(cache.contains(' ', 16.0));
    }

    #[test]
    fn cached_glyph_eval() {
        use pixelflow_core::Field;

        let font = Font::parse(FONT_DATA).unwrap();
        let glyph = font.glyph_scaled('A', 32.0).unwrap();
        let cached = CachedGlyph::new(&glyph, 32);

        // Evaluate coverage at multiple coordinates - should not panic
        for x in [2.0, 8.0, 16.0, 24.0] {
            for y in [2.0, 8.0, 16.0, 24.0] {
                let _coverage = cached.eval_raw(
                    Field::from(x),
                    Field::from(y),
                    Field::from(0.0),
                    Field::from(0.0),
                );
            }
        }
    }

    #[test]
    fn cached_text_creation() {
        use pixelflow_core::Field;

        let font = Font::parse(FONT_DATA).unwrap();
        let mut cache = GlyphCache::new();

        let text = CachedText::new(&font, &mut cache, "Hello", 16.0);

        // Should have cached glyphs for H, e, l, o (l appears twice)
        assert_eq!(cache.len(), 4);

        // Evaluate text at multiple coordinates - should not panic
        for x in [0.0, 5.0, 10.0, 20.0] {
            for y in [5.0, 10.0, 15.0] {
                let _coverage = text.eval_raw(
                    Field::from(x),
                    Field::from(y),
                    Field::from(0.0),
                    Field::from(0.0),
                );
            }
        }
    }

    #[test]
    fn cache_memory_usage() {
        let font = Font::parse(FONT_DATA).unwrap();
        let mut cache = GlyphCache::new();

        cache.get(&font, 'A', 16.0); // 16x16 = 256 texels * 4 bytes = 1024
        cache.get(&font, 'B', 16.0); // Another 1024

        assert_eq!(cache.memory_usage(), 2048);
    }
}
