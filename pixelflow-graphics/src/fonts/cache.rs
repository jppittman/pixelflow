//! # Font Caching
//!
//! Glyph caching via lattice collapse + bilinear sampling.
//!
//! ## Categorical Semantics
//!
//! Caching is a **morphism between evaluation strategies**:
//! - a glyph `Kernel` evaluates mathematically (winding numbers, infinite
//!   resolution)
//! - `CachedGlyph` evaluates from a baked lattice (SIMD gather, fixed
//!   resolution)
//!
//! The bake JIT-compiles the fused glyph kernel once (`Lattice::bake`,
//! global compile cache) and tabulates it; antialiasing is intrinsic to the
//! kernel (symbolic `Dwrt` crossing ramps resolved at compile time). The
//! read-back is a [`BilinearSampler`] — a JIT'd 4-tap gather kernel bound to
//! the baked buffer. Texels therefore store *antialiased* coverage — no
//! post-hoc filtering of hard 0/1 samples.
//!
//! Both sides are `Kernel`s, and the morphism is between how they get their
//! numbers: the analytic glyph computes them, the cached one reads them.
//!
//! ```text
//!            cache_at(size)
//!     Glyph ──────────────► CachedGlyph
//!       │                        │
//!       │ Kernel                 │ Kernel over a bound buffer
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
//! `(i + 0.5, j + 0.5)` — a contramap on the kernel before it is baked
//! over a plain index lattice, since a lattice carries no coordinate frame
//! of its own to shift by.
//! [`CachedGlyph::kernel`] shifts incoming coordinates by −0.5 into the
//! sampler's integer texel grid, so a query at a pixel center returns
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

use pixelflow_core::{BilinearSampler, DiscreteManifold, Kernel, Lattice};
use std::collections::HashMap;
use std::sync::Arc;

use super::loop_blinn::Glyph;
use super::ttf::Font;
// `PIXEL_CENTER` is this crate's shared rasterizer convention (`fonts/mod.rs`)
// — see the "Coordinate convention" section above for what it means here.
use super::PIXEL_CENTER;

// ═══════════════════════════════════════════════════════════════════════════
// CachedGlyph: The Morphism
// ═══════════════════════════════════════════════════════════════════════════

/// A glyph baked to a coverage lattice.
///
/// This is the output of the caching morphism: a glyph whose kernel reads
/// coverage from memory rather than computing winding numbers. The lattice stores f32
/// *antialiased* coverage values (0.0 to 1.0) — no u8 quantization roundtrip
/// — sampled back via SIMD gather with bilinear interpolation.
///
/// # Resolution
///
/// Unlike the analytical glyph kernel, which has infinite resolution,
/// `CachedGlyph` is baked at a fixed size. For best quality, cache at the
/// exact render size. The cache uses size buckets (multiples of 4) to
/// balance memory with reuse.
#[derive(Clone)]
pub struct CachedGlyph {
    /// Bilinear sampler over the baked coverage lattice.
    /// Arc so cloning a cached glyph (once per cell per frame) is O(1).
    sampler: Arc<BilinearSampler>,
    /// Point-space width (the baked lattice holds `width × density` texels).
    width: usize,
    /// Point-space height (the baked lattice holds `height × density` texels).
    height: usize,
    /// Sample density the lattice was baked at, in texels per point.
    /// Queries stay in point space; the kernel contramaps by this factor.
    density: f32,
}

/// Baked lattice extent in texels for a point-space size at a density.
fn px_extent(size: usize, density: f32) -> usize {
    (size as f32 * density).round() as usize
}

impl CachedGlyph {
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
        self.sampler.texture()
    }

    /// Create a cached glyph by baking a glyph coverage [`Kernel`]
    /// ([`Font::glyph_kernel_scaled`] → one fused arena, compiled once
    /// through the global cache, tabulated over a [`Lattice`]).
    /// Antialiasing comes from the kernel's symbolic `Dwrt` ramps resolved
    /// at compile time. The JIT-vs-interpreter goldens
    /// (tests/kernel_glyph_golden.rs) guard this path. The kernel's outline
    /// must be scaled to `size × density` pixels; texels sample at centers,
    /// and the result takes point-space coordinates.
    ///
    /// Takes the whole [`Glyph`], not a bare [`Kernel`], to match
    /// [`Glyph::bake`]'s own shape; the winding sum's piece table travels
    /// with `glyph.kernel()` itself (`Kernel::with_buffer_data`, seeded by
    /// [`loop_blinn::glyph`](super::loop_blinn::glyph)), so — unlike
    /// before — there is no second value that must come from the same call.
    #[must_use]
    pub fn from_kernel(glyph: &Glyph, size: usize, density: f32) -> Self {
        assert!(
            density.is_finite() && density > 0.0,
            "invalid bake density: {density}"
        );
        let px = px_extent(size, density);
        // Texel-center convention: texel (i, j) holds coverage at
        // (i + PIXEL_CENTER, j + PIXEL_CENTER). Used to be the bake
        // lattice's own origin; a contramap on the kernel now that a
        // lattice is a pure index.
        let centered = glyph.kernel().at(
            &Kernel::x().add(&Kernel::constant(PIXEL_CENTER)),
            &Kernel::y().add(&Kernel::constant(PIXEL_CENTER)),
        );
        let lattice = Lattice {
            extent: [px as u32, px as u32],
        };
        // `Glyph::bake` needs no explicit binding: the winding table the
        // kernel declares (S1a) travels with it, and a glyph with no
        // outline declares no buffer at all — both bake the same way.
        let baked = glyph.bake(&centered, lattice);

        Self {
            sampler: Arc::new(baked.bilinear()),
            width: size,
            height: size,
            density,
        }
    }
}

impl CachedGlyph {
    /// This glyph's coverage in point space, as a [`Kernel`]: the baked texels
    /// read through the bilinear blend, masked to the glyph's own extent.
    ///
    /// The kernel declares the coverage buffer as a slot, but reaches
    /// numbers without a caller binding it: [`BilinearSampler::kernel`]
    /// seeds this fragment with the coverage lattice's own texels
    /// (`Kernel::with_buffer_data`), so the data travels with the value —
    /// compile at a lattice's shape, bind (trivially), collapse. It
    /// composes like any other kernel too: `.at(..)` reads it at computed
    /// coordinates, which is how [`CachedText`] places it.
    ///
    /// Two things happen here and nowhere else. The query is contramapped into
    /// the sampler's texel grid: point `p` lands on baked pixel `p·density`,
    /// and texel `(i, j)` holds coverage at pixel center `(i + ½, j + ½)`, so
    /// the composed embedding is `p·density − ½` — at integer densities a
    /// density-matched sample grid lands exactly on texel centers and the
    /// bilinear read degenerates to a lossless lookup. And the result is
    /// masked to the point-space extent, because a gather clamps out-of-range
    /// indices to the edge texel, which would smear a nonzero boundary
    /// coverage (a descender reaching the em-box bottom) out to infinity.
    /// Outside the bake there is no data — coverage is zero.
    #[must_use]
    pub fn kernel(&self) -> Kernel {
        let texel = |axis: &Kernel| {
            axis.mul(&Kernel::constant(self.density))
                .sub(&Kernel::constant(PIXEL_CENTER))
        };
        let sampled = self
            .sampler
            .kernel()
            .at(&texel(&Kernel::x()), &texel(&Kernel::y()));
        let zero = Kernel::constant(0.0);
        let in_bounds = Kernel::x()
            .ge(&zero)
            .and(&Kernel::x().le(&Kernel::constant(self.width as f32)))
            .and(&Kernel::y().ge(&zero))
            .and(&Kernel::y().le(&Kernel::constant(self.height as f32)));
        in_bounds.select(&sampled, &zero)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GlyphCache: The Functor
// ═══════════════════════════════════════════════════════════════════════════

/// Quantization granularity for size buckets: multiples of 4 pixels, for
/// SIMD-friendly dimensions.
const SIZE_BUCKET_GRANULARITY: usize = 4;

/// Smallest size bucket a glyph can round down into.
const MIN_SIZE_BUCKET: usize = 8;

/// Size bucket for cache keys.
///
/// Quantizes sizes to reduce cache entries while maintaining quality.
fn size_bucket(size: f32) -> usize {
    let granularity = SIZE_BUCKET_GRANULARITY as f32;
    let bucket = ((size / granularity).ceil() as usize) * SIZE_BUCKET_GRANULARITY;
    bucket.max(MIN_SIZE_BUCKET)
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
        // The glyph as ONE fused coverage Kernel, compiled once, tabulated
        // over a Lattice. There is no fallback path: an architecture without
        // an arena backend fails to build, loudly, rather than rendering
        // slowly.
        let glyph = font.glyph_kernel_scaled(ch, px as f32)?;
        let cached = CachedGlyph::from_kernel(&glyph, bucket, density);
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
    /// Positioned cached glyphs: each sampled at `((x - dx) * inv_scale, y * inv_scale)`.
    glyphs: Vec<PlacedGlyph>,
}

/// One laid-out glyph: a baked sampler plus its pen translation and the
/// bucket→request scale. The samplers enter the language as bound buffers
/// rather than as expressions, so placing one is a contramap of its kernel
/// and a run is their sum.
#[derive(Clone)]
struct PlacedGlyph {
    glyph: CachedGlyph,
    dx: f32,
    inv_scale: f32,
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

            // Get cached glyph; the bake is at bucket size, so queries scale
            // from request space into bucket space.
            if let Some(cached) = cache.get(font, ch, size, density) {
                glyphs.push(PlacedGlyph {
                    glyph: cached,
                    dx: cursor_x,
                    inv_scale: 1.0 / scale,
                });
            }

            // Advance cursor using pre-looked-up glyph ID
            if let Some(adv) = font.advance_by_id(id) {
                cursor_x += adv * inv_em;
            }

            prev_id = Some(id);
        }

        Self { glyphs }
    }
}

impl CachedText {
    /// The whole run's coverage as one [`Kernel`]: every glyph's kernel read
    /// at its own pen position, summed.
    ///
    /// Composition is `Kernel::at` and `Kernel::sum` — the same two moves the
    /// layout above already makes, now in the language rather than in a Rust
    /// loop over `eval`, so the run compiles as one kernel with each glyph's
    /// coverage a declared slot, its data carried along rather than gathered
    /// separately. A run that draws the same character twice places two
    /// `Kernel`s built from the same `CachedGlyph` (the cache returns a
    /// clone, sharing its `Arc<BilinearSampler>`), so `Kernel::sum`'s merge
    /// sees one `BufferIdentity` twice naming the very same `Arc` — a
    /// pointer-equal no-op, not two tabulations to compare.
    #[must_use]
    pub fn kernel(&self) -> Kernel {
        let placed: Vec<Kernel> = self
            .glyphs
            .iter()
            .map(|pg| {
                let scale = Kernel::constant(pg.inv_scale);
                pg.glyph.kernel().at(
                    &Kernel::x().sub(&Kernel::constant(pg.dx)).mul(&scale),
                    &Kernel::y().mul(&scale),
                )
            })
            .collect();
        Kernel::sum(&placed)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_core::Manifold;

    // Use the fallback font, which is committed directly (not Git-LFS) so the
    // tests run without `git lfs pull`. NotoSansMono is an LFS pointer.
    const FONT_DATA: &[u8] = include_bytes!("../../assets/DejaVuSansMono-Fallback.ttf");

    /// A coverage kernel tabulated over `lattice`: compile at its shape, bind
    /// (trivially — every buffer slot `kernel` declares carries its own data
    /// now), collapse. The whole evaluation API.
    fn collapse(kernel: &Kernel, lattice: Lattice) -> Vec<f32> {
        let bound = Manifold::compile(kernel, lattice.extent).bind(&[]);
        lattice.collapse(&bound).into_buffer()
    }

    /// One glyph's coverage over `lattice`.
    fn glyph_grid(g: &CachedGlyph, lattice: Lattice) -> Vec<f32> {
        collapse(&g.kernel(), lattice)
    }

    /// One glyph's coverage over a `size × size` point-space grid, sampled at
    /// pixel centers — the rasterizer's own convention, applied as a
    /// contramap since the lattice it bakes over is a pure index.
    fn glyph_grid_at_pixel_centers(g: &CachedGlyph, size: usize) -> Vec<f32> {
        let centered = g.kernel().at(
            &Kernel::x().add(&Kernel::constant(0.5)),
            &Kernel::y().add(&Kernel::constant(0.5)),
        );
        collapse(&centered, Lattice::frame(size, size))
    }

    /// One glyph's coverage at a single point — not a lattice at all now,
    /// since a lattice carries no coordinate; a bound manifold answers a
    /// point directly.
    fn sample(g: &CachedGlyph, x: f32, y: f32) -> f32 {
        let bound = Manifold::compile(&g.kernel(), [1, 1]).bind(&[]);
        bound.eval_at(x, y)
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
        let glyph = font.glyph_kernel_scaled('A', 32.0).unwrap();
        let cached = CachedGlyph::from_kernel(&glyph, 32, 1.0);

        assert_eq!(cached.width(), 32);
        assert_eq!(cached.height(), 32);
    }

    #[test]
    fn baked_coverage_dimensions_range_and_ink() {
        let font = Font::parse(FONT_DATA).unwrap();
        let glyph = font.glyph_kernel_scaled('A', 32.0).unwrap();
        let cached = CachedGlyph::from_kernel(&glyph, 32, 1.0);

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
    fn no_half_pixel_shift_center_of_mass() {
        // Regression: the baked glyph must sit at the same position as the
        // analytical glyph rasterized directly at pixel centers. A half-pixel
        // convention error here shows up as a ~0.5px center-of-mass shift.
        let size = 32usize;
        let font = Font::parse(FONT_DATA).unwrap();
        let glyph = font.glyph_kernel_scaled('A', size as f32).unwrap();

        // Direct analytical tabulation at pixel centers (the rasterizer's
        // sampling convention), as a contramap over a plain index lattice.
        // The winding sum reads a bound piece table (S1a), so bind it
        // rather than a bare `Lattice::bake`.
        let centered = glyph.kernel().at(
            &Kernel::x().add(&Kernel::constant(0.5)),
            &Kernel::y().add(&Kernel::constant(0.5)),
        );
        let lattice = Lattice::frame(size, size);
        let direct = glyph.bake(&centered, lattice);
        let (dx, dy) = center_of_mass(direct.buffer(), size);

        // Cached glyph sampled at pixel centers through the full
        // bake -> bilinear -> half-pixel-shift chain.
        let cached = CachedGlyph::from_kernel(&glyph, size, 1.0);
        let resampled = glyph_grid_at_pixel_centers(&cached, size);
        let (cx, cy) = center_of_mass(&resampled, size);

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
        let glyph = font.glyph_kernel_scaled('A', 32.0).unwrap();
        let cached = CachedGlyph::from_kernel(&glyph, 32, 1.0);

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
    fn a_cached_glyph_collapses_over_its_own_extent() {
        let font = Font::parse(FONT_DATA).unwrap();
        let glyph = font.glyph_kernel_scaled('A', 32.0).unwrap();
        let cached = CachedGlyph::from_kernel(&glyph, 32, 1.0);

        // One kernel, one buffer slot, one compile at the grid's shape: the
        // whole point-space extent comes back in one collapse, in [0, 1].
        let grid = glyph_grid(&cached, Lattice::frame(32, 32));
        assert_eq!(grid.len(), 32 * 32);
        assert!(
            grid.iter().all(|v| (0.0..=1.0).contains(v)),
            "sampled coverage left [0, 1]"
        );
        assert!(grid.iter().sum::<f32>() > 10.0, "glyph 'A' sampled blank");
    }

    #[test]
    fn a_run_collapses_as_one_kernel_over_its_glyphs_buffers() {
        let font = Font::parse(FONT_DATA).unwrap();
        let mut cache = GlyphCache::new();

        let text = CachedText::new(&font, &mut cache, "Hello", 16.0, 1.0);

        // Should have cached glyphs for H, e, l, o (l appears twice)
        assert_eq!(cache.len(), 4);

        // The run is ONE kernel — every glyph placed by `Kernel::at` and
        // summed — over the four distinct coverage buffers its glyphs bake,
        // repeats sharing one identity and (now) one carried tabulation:
        // `Kernel::sum`'s merge sees the repeated 'l's name the same
        // `BufferIdentity` with the very same `Arc`, so it collapses to one
        // entry rather than asserting a mismatch. Collapsing draws the
        // whole line with no binding gathered by hand.
        let lattice = Lattice::frame(48, 16);
        let line = collapse(&text.kernel(), lattice);
        assert_eq!(line.len(), 48 * 16);
        assert!(
            line.iter().sum::<f32>() > 10.0,
            "the run collapsed to no ink at all"
        );
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
        let grid = |g: &CachedGlyph| glyph_grid_at_pixel_centers(g, 16);
        let (x1, y1) = center_of_mass(&grid(&d1), 16);
        let (x2, y2) = center_of_mass(&grid(&d2), 16);
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
