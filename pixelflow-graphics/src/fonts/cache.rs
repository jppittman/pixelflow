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
//! let cached = cache.get(&font, 'A', 16.0, 1.0);
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
    /// Point-space width (the baked lattice holds `width × density` texels).
    width: usize,
    /// Point-space height (the baked lattice holds `height × density` texels).
    height: usize,
    /// Sample density the lattice was baked at, in texels per point.
    /// Queries stay in point space; `eval` contramaps by this factor.
    density: f32,
}

/// Baked lattice extent in texels for a point-space size at a density.
fn px_extent(size: usize, density: f32) -> usize {
    (size as f32 * density).round() as usize
}

impl CachedGlyph {
    /// Create a cached glyph by collapsing a glyph manifold over a lattice.
    ///
    /// The glyph's *antialiased* coverage is tabulated at
    /// `(size × density)²` texels, at texel centers (see the module docs for
    /// the coordinate convention). The bake evaluates the glyph through
    /// [`Antialiased`], so every texel stores a gradient-normalized crossing
    /// ramp value — the bilinear read-back then interpolates already-smooth
    /// coverage.
    ///
    /// `density` is the sample density in texels per point (a display's
    /// backing scale). The caller must supply a `glyph` whose outline is
    /// scaled to `size × density` pixels, so the analytic curves are truly
    /// sampled denser — not upscaled. The resulting manifold still takes
    /// point-space coordinates and occupies `size × size` points.
    #[must_use]
    pub fn new<L, Q>(glyph: &Glyph<L, Q>, size: usize, density: f32) -> Self
    where
        Glyph<L, Q>: Manifold<Jet2x4, Output = Field>,
    {
        assert!(
            density.is_finite() && density > 0.0,
            "invalid bake density: {density}"
        );
        // Antialiased coverage: seed Jet2 screen-space derivatives so each
        // edge crossing bakes as a ~1-texel gradient-normalized ramp.
        let coverage = Antialiased::new(glyph);

        // Tabulate at texel centers: texel (i, j) = coverage(i+0.5, j+0.5).
        let px = px_extent(size, density);
        let lattice = Lattice {
            extent: [px as u32, px as u32, 1, 1],
            origin: [TEXEL_CENTER, TEXEL_CENTER, 0.0, 0.0],
        };
        let baked = lattice.collapse(&coverage);

        Self {
            sampler: Arc::new(Bilinear::new(baked)),
            width: size,
            height: size,
            density,
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
        // Contramap the point-space query into the sampler's texel grid:
        // point p lands on baked pixel p·density, and texel (i, j) holds
        // coverage at pixel center (i + 0.5, j + 0.5), so the composed
        // embedding is p·density − 0.5. At integer densities this maps a
        // density-matched sample grid exactly onto texel centers, so the
        // bilinear read degenerates to lossless lookup.
        let sampled = At {
            inner: &*self.sampler,
            x: X * self.density - TEXEL_CENTER,
            y: Y * self.density - TEXEL_CENTER,
            z: Z,
            w: W,
        };
        // Bound to the point-space extent: `DiscreteManifold` clamps
        // out-of-range indices to the edge texel, which would smear nonzero
        // boundary coverage (e.g. a descender reaching the em-box bottom) to
        // infinity. Outside the bake there is no data — coverage is zero.
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

/// Quantization granularity for density buckets: eighth-of-a-texel steps.
const DENSITY_STEPS: f32 = 8.0;

/// Density bucket for cache keys.
///
/// Quantizes texels-per-point to eighth steps (1.0, 1.125, …, 2.0, …) so a
/// bake and its key always agree, and a 16pt glyph on a 2x display never
/// collides with a 32pt glyph on a 1x display — same texel count, different
/// point-space geometry.
fn density_bucket(density: f32) -> u16 {
    assert!(
        density.is_finite() && density > 0.0,
        "invalid glyph density: {density}"
    );
    ((density * DENSITY_STEPS).round() as u16).max(1)
}

/// Key for cached glyphs.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct CacheKey {
    codepoint: u32,
    size_bucket: usize,
    density_q: u16,
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
    /// If the glyph at this (size bucket, density bucket) is already cached,
    /// returns it. Otherwise bakes the glyph at `size × density` texels and
    /// caches it. `density` is texels per point (a display's backing scale);
    /// the returned glyph takes point-space coordinates regardless.
    pub fn get(&mut self, font: &Font, ch: char, size: f32, density: f32) -> Option<CachedGlyph> {
        let bucket = size_bucket(size);
        let density_q = density_bucket(density);
        let key = CacheKey {
            codepoint: ch as u32,
            size_bucket: bucket,
            density_q,
        };

        if let Some(cached) = self.entries.get(&key) {
            return Some(cached.clone());
        }

        // Bake at the quantized density so the key and the lattice agree.
        let density = density_q as f32 / DENSITY_STEPS;
        let px = px_extent(bucket, density);
        let glyph = font.glyph_scaled(ch, px as f32)?;
        let cached = CachedGlyph::new(&glyph, bucket, density);
        self.entries.insert(key, cached.clone());
        Some(cached)
    }

    /// Check if a glyph is cached at this size and density.
    #[must_use]
    pub fn contains(&self, ch: char, size: f32, density: f32) -> bool {
        let key = CacheKey {
            codepoint: ch as u32,
            size_bucket: size_bucket(size),
            density_q: density_bucket(density),
        };
        self.entries.contains_key(&key)
    }

    /// Pre-warm the cache with common characters.
    ///
    /// Call this at startup to avoid cache misses during rendering.
    pub fn warm(
        &mut self,
        font: &Font,
        chars: impl IntoIterator<Item = char>,
        size: f32,
        density: f32,
    ) {
        for ch in chars {
            self.get(font, ch, size, density);
        }
    }

    /// Pre-warm with ASCII printable characters.
    pub fn warm_ascii(&mut self, font: &Font, size: f32, density: f32) {
        self.warm(font, ' '..='~', size, density);
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
            .map(|g| {
                let tex = g.coverage();
                tex.width() * tex.height() * 4 // f32 per texel
            })
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
/// cache.warm_ascii(&font, 16.0, 1.0);
///
/// let text = CachedText::new(&font, &mut cache, "Hello", 16.0, 1.0);
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
    /// Glyphs are retrieved from the cache (or baked on-demand) at the given
    /// sample density (texels per point); layout stays in point space.
    pub fn new(font: &Font, cache: &mut GlyphCache, text: &str, size: f32, density: f32) -> Self {
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
            if let Some(cached) = cache.get(font, ch, size, density) {
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
        cache.get(&font, 'A', 9.0, 1.0);
        assert_eq!(cache.len(), 1);
        assert!(
            cache.contains('A', 12.0, 1.0),
            "9.0 and 12.0 should share the 12px bucket"
        );

        // 13.0 rounds up to the next bucket (16px), so it must not collide
        // with the 12px bucket.
        assert!(
            !cache.contains('A', 13.0, 1.0),
            "13.0 should land in a different bucket than 12.0"
        );

        // Sizes at or below the minimum clamp to the 8px bucket.
        cache.get(&font, 'B', 1.0, 1.0);
        assert!(
            cache.contains('B', 8.0, 1.0),
            "sub-minimum sizes should clamp to the 8px bucket"
        );
    }

    #[test]
    fn cached_glyph_creation() {
        let font = Font::parse(FONT_DATA).unwrap();
        let glyph = font.glyph_scaled('A', 32.0).unwrap();
        let cached = CachedGlyph::new(&glyph, 32, 1.0);

        assert_eq!(cached.width(), 32);
        assert_eq!(cached.height(), 32);
    }

    #[test]
    fn baked_coverage_dimensions_range_and_ink() {
        let font = Font::parse(FONT_DATA).unwrap();
        let glyph = font.glyph_scaled('A', 32.0).unwrap();
        let cached = CachedGlyph::new(&glyph, 32, 1.0);

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
        let cached = CachedGlyph::new(&glyph, 32, 1.0);

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
        let cached = CachedGlyph::new(&glyph, size, 1.0);
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
        let cached = CachedGlyph::new(&glyph, 32, 1.0);

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
        let cached = cache.get(&font, 'A', 16.0, 1.0);
        assert!(cached.is_some());
        assert_eq!(cache.len(), 1);

        // Second access should hit cache
        let cached2 = cache.get(&font, 'A', 16.0, 1.0);
        assert!(cached2.is_some());
        assert_eq!(cache.len(), 1);

        // Different size should create new entry
        let cached3 = cache.get(&font, 'A', 32.0, 1.0);
        assert!(cached3.is_some());
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn glyph_cache_warm() {
        let font = Font::parse(FONT_DATA).unwrap();
        let mut cache = GlyphCache::new();

        cache.warm_ascii(&font, 16.0, 1.0);

        // ASCII printable is 95 characters
        assert_eq!(cache.len(), 95);

        // All should be cached now
        assert!(cache.contains('A', 16.0, 1.0));
        assert!(cache.contains('z', 16.0, 1.0));
        assert!(cache.contains(' ', 16.0, 1.0));
    }

    #[test]
    fn cached_glyph_eval() {
        use pixelflow_core::Field;

        let font = Font::parse(FONT_DATA).unwrap();
        let glyph = font.glyph_scaled('A', 32.0).unwrap();
        let cached = CachedGlyph::new(&glyph, 32, 1.0);

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

        let text = CachedText::new(&font, &mut cache, "Hello", 16.0, 1.0);

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

        cache.get(&font, 'A', 16.0, 1.0); // 16x16 = 256 texels * 4 bytes = 1024
        cache.get(&font, 'B', 16.0, 1.0); // Another 1024

        assert_eq!(cache.memory_usage(), 2048);
    }

    #[test]
    fn density_preserves_point_geometry() {
        // A denser bake must not change where the glyph lives in point space:
        // same reported extent, same out-of-bounds masking, and roughly the
        // same coverage at matching point coordinates.
        let font = Font::parse(FONT_DATA).unwrap();
        let mut cache = GlyphCache::new();

        let d1 = cache.get(&font, 'A', 16.0, 1.0).unwrap();
        let d2 = cache.get(&font, 'A', 16.0, 2.0).unwrap();

        assert_eq!(d1.width(), 16);
        assert_eq!(d2.width(), 16, "density must not change point-space width");
        assert_eq!(d2.height(), 16);
        assert_eq!(d2.coverage().width(), 32, "density 2 must bake 2x texels");

        // Outside the point-space extent both are transparent.
        assert_eq!(sample(&d1, 17.0, 8.0), 0.0);
        assert_eq!(sample(&d2, 17.0, 8.0), 0.0);

        // Ink sits in the same place: centers of mass over the same
        // point-space grid agree to well under a point.
        let grid = |g: &CachedGlyph| {
            Lattice {
                extent: [16, 16, 1, 1],
                origin: [0.5, 0.5, 0.0, 0.0],
            }
            .collapse(g)
        };
        let (x1, y1) = center_of_mass(grid(&d1).buffer(), 16);
        let (x2, y2) = center_of_mass(grid(&d2).buffer(), 16);
        assert!(
            (x1 - x2).abs() < 0.25 && (y1 - y2).abs() < 0.25,
            "density moved the glyph: d1 ({x1}, {y1}) vs d2 ({x2}, {y2})"
        );
    }

    #[test]
    fn density_two_resamples_sharper_edges() {
        // The whole point of density: a 2x bake must re-sample the analytic
        // outline, not upscale the 1x lattice. The AA crossing ramp is ~1
        // texel wide, so in point space it is ~1 point at density 1 but only
        // ~0.5 points at density 2 — scanning a stem edge at sub-point steps
        // must find a strictly narrower transition zone.
        let font = Font::parse(FONT_DATA).unwrap();
        let mut cache = GlyphCache::new();
        let d1 = cache.get(&font, 'l', 16.0, 1.0).unwrap();
        let d2 = cache.get(&font, 'l', 16.0, 2.0).unwrap();

        let transition_samples = |g: &CachedGlyph| {
            let mut count = 0usize;
            let mut x = 0.0f32;
            while x <= 16.0 {
                let v = sample(g, x, 8.0);
                if v > 0.1 && v < 0.9 {
                    count += 1;
                }
                x += 0.0625;
            }
            count
        };

        let t1 = transition_samples(&d1);
        let t2 = transition_samples(&d2);
        assert!(t1 > 0, "density-1 scan found no edge transition to compare");
        assert!(
            t2 < t1,
            "density 2 did not sharpen edges: {t2} transition samples vs {t1} at density 1 \
             (denser bake is interpolating, not re-sampling)"
        );
    }

    #[test]
    fn cache_key_distinguishes_density_from_size() {
        // 16pt @ 2x and 32pt @ 1x bake the same texel count but are different
        // glyphs (different point-space extent) — they must not collide.
        let font = Font::parse(FONT_DATA).unwrap();
        let mut cache = GlyphCache::new();

        let hidpi = cache.get(&font, 'A', 16.0, 2.0).unwrap();
        let large = cache.get(&font, 'A', 32.0, 1.0).unwrap();

        assert_eq!(cache.len(), 2, "16pt@2x collided with 32pt@1x");
        assert_eq!(hidpi.width(), 16);
        assert_eq!(large.width(), 32);

        // Densities quantize to eighth steps: 2.0 and 2.04 share a bucket,
        // 2.0 and 2.1 do not.
        assert!(cache.contains('A', 16.0, 2.04));
        assert!(!cache.contains('A', 16.0, 2.1));
    }
}
