//! # Font Caching
//!
//! Glyph caching via the Baked combinator.
//!
//! ## Categorical Semantics
//!
//! Caching is a **morphism between evaluation strategies**:
//! - `Glyph` evaluates mathematically (winding numbers, infinite resolution)
//! - `CachedGlyph` evaluates from texture memory (SIMD gather, fixed resolution)
//!
//! Both implement `Manifold`, preserving composition. The cache is a functor
//! that transforms evaluation strategy while maintaining the algebraic interface.
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

use crate::render::color::Pixel;
use crate::render::frame::Frame;
use crate::render::rasterizer::execute;
use crate::Rgba8;
use pixelflow_core::{Field, Manifold, ManifoldCompat, Texture};
use std::collections::HashMap;
use std::sync::Arc;

/// The standard 4D Field domain type.
type Field4 = (Field, Field, Field, Field);

use super::ttf::{affine, Affine, Font, Glyph, Sum};
use crate::Grayscale;

// ═══════════════════════════════════════════════════════════════════════════
// CachedGlyph: The Morphism
// ═══════════════════════════════════════════════════════════════════════════

/// A glyph cached to texture memory.
///
/// This is the output of the caching morphism: a glyph that evaluates from
/// memory rather than computing winding numbers. The texture stores coverage
/// values (0.0 to 1.0), enabling SIMD gather for parallel sampling.
///
/// # Resolution
///
/// Unlike raw `Glyph` which has infinite resolution, `CachedGlyph` is baked
/// at a fixed size. For best quality, cache at the exact render size.
/// The cache uses size buckets (multiples of 4) to balance memory with reuse.
#[derive(Clone)]
pub struct CachedGlyph {
    /// Coverage texture (single channel, 0.0-1.0).
    coverage: Arc<Texture>,
    /// Baked width in pixels.
    width: usize,
    /// Baked height in pixels.
    height: usize,
}

impl CachedGlyph {
    /// Create a cached glyph by baking a glyph manifold.
    ///
    /// The glyph is rasterized at `size × size` resolution.
    #[must_use]
    pub fn new<L, Q>(glyph: &Glyph<L, Q>, size: usize) -> Self
    where
        L: Manifold<Field4, Output = Field>,
        Q: Manifold<Field4, Output = Field>,
        Glyph<L, Q>: Clone,
    {
        // Rasterize glyph to RGBA frame using Grayscale (coverage → grayscale)
        let lifted = Grayscale(glyph.clone());
        let mut frame = Frame::<Rgba8>::new(size as u32, size as u32);
        execute(&lifted, &mut frame);

        // Extract coverage from red channel (Lift produces uniform gray)
        let mut data = Vec::with_capacity(size * size);
        for pixel in &frame.data {
            let packed = pixel.to_u32();
            let r = (packed & 0xFF) as f32 / 255.0;
            data.push(r);
        }

        let coverage = Texture::new(data, size, size);

        Self {
            coverage: Arc::new(coverage),
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
}

impl Manifold<Field4> for CachedGlyph {
    type Output = Field;

    #[inline(always)]
    fn eval(&self, p: Field4) -> Field {
        let (x, y, z, w) = p;
        // Sample coverage from texture (SIMD gather)
        self.coverage.eval_raw(x, y, z, w)
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
            .map(|g| g.width * g.height * 4) // f32 per pixel
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
        let mut glyphs = Vec::new();
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

    const FONT_DATA: &[u8] = include_bytes!("../../assets/NotoSansMono-Regular.ttf");

    #[test]
    fn test_size_bucket() {
        assert_eq!(size_bucket(8.0), 8);
        assert_eq!(size_bucket(9.0), 12);
        assert_eq!(size_bucket(12.0), 12);
        assert_eq!(size_bucket(13.0), 16);
        assert_eq!(size_bucket(16.0), 16);
        assert_eq!(size_bucket(17.0), 20);
    }

    #[test]
    fn test_cached_glyph_creation() {
        let font = Font::parse(FONT_DATA).unwrap();
        let glyph = font.glyph_scaled('A', 32.0).unwrap();
        let cached = CachedGlyph::new(&glyph, 32);

        assert_eq!(cached.width(), 32);
        assert_eq!(cached.height(), 32);
    }

    #[test]
    fn test_glyph_cache_get() {
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
    fn test_glyph_cache_warm() {
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
    fn test_cached_glyph_eval() {
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
    fn test_cached_text_creation() {
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
    fn test_cache_memory_usage() {
        let font = Font::parse(FONT_DATA).unwrap();
        let mut cache = GlyphCache::new();

        cache.get(&font, 'A', 16.0); // 16x16 = 256 pixels * 4 bytes = 1024
        cache.get(&font, 'B', 16.0); // Another 1024

        assert_eq!(cache.memory_usage(), 2048);
    }
}
