//! # Font Rendering Pipeline
//!
//! Bridges vector font formats (TTF) to glyph coverage kernels
//! ([`pixelflow_core::Kernel`]).
//!
//! ## Architecture: Four Layers
//!
//! ```text
//! Text Layer (text(), CachedText)
//!      ↓
//!      │  Layout: advances/kerning; one outline, placed
//!      │
//! Cache Layer (GlyphCache, CachedGlyph)
//!      ↓
//!      │  Lattice::bake'd f32 AA coverage + bilinear read-back
//!      │
//! Coverage Layer (loop_blinn)
//!      ↓
//!      │  An outline's area under each pixel, as one Kernel
//!      │
//! Font Layer (Font, outline)
//!      ↓
//!      │  TTF parsing; a glyph is geometry, in the caller's frame
//!      │
//! Loading Layer (loader: DataSource, EmbeddedSource, MmapSource, LoadedFont)
//!      ↓
//! In-Memory Font Data
//! ```
//!
//! ## Coverage semantics: the exact area under ink
//!
//! A glyph's coverage of a pixel is `min(|F|, 1)`, where `F` is the pixel's
//! signed area under ink — the non-zero rule's winding number integrated
//! over the pixel. [`loop_blinn`] writes `F` as a formula, one term per
//! monotone arc of the outline, each the area of the pixel to the arc's
//! left within its band, and the e-graph closes every term's integral
//! exactly (docs/plans/2026-09-23-a-glyph-is-a-formula.md). A pixel on an
//! edge reads the fraction of it the ink covers; a corner and a thin stem
//! read their areas, not a ramp on one distance. Where two contours overlap
//! inside a pixel, `|F|` clamped reads their union as FreeType's rasterizer
//! does, off by the overlap of two fractions.
//!
//! The pixel is the frame's unit square about the sample, so the glyph is
//! built in the frame it is drawn in (every scale is applied to control
//! points on the host) and a caller places it with a translation. There is
//! no separate hard/AA mode — coverage is antialiased by construction.
//!
//! ## Layer 1: Font Loading (`loader` module)
//!
//! Font bytes come from a [`FontSource`]: [`DataSource`] (owned bytes),
//! [`EmbeddedSource`] (bytes baked into the binary), or [`MmapSource`]
//! (zero-copy memory-mapped file). [`LoadedFont`] owns the source and
//! lends out parsed [`Font`] views.
//!
//! ## Layer 2: Glyph Compilation (`outline`, `loop_blinn` and `ttf`)
//!
//! [`Font::parse`] reads the TTF tables (cmap, glyf, loca, hmtx, kern), and
//! `ttf` produces **geometry**: an [`Outline`] of line and quadratic
//! segments, with compound glyphs flattened through their component
//! transforms. Every affine map — the em scale, the screen flip, a
//! component's placement, a pen position — is applied to control points on
//! the host, so the kernel is built in the frame it is evaluated in.
//!
//! [`loop_blinn`] turns an outline into coverage: [`loop_blinn::glyph`] as
//! one kernel over the whole plane, cut to an exact [`Support`]. Metrics
//! come from `advance`/`kern` and their `*_by_id`/`*_scaled` variants.
//!
//! ## Layer 3: Glyph Caching (`cache` module)
//!
//! Analytical evaluation walks every curve per sample. `GlyphCache` bakes
//! glyphs once per (character, size bucket): [`CachedGlyph::from_kernel`]
//! JIT-compiles the fused kernel (global compile cache) and tabulates it
//! over a `Lattice` into f32 coverage at pixel centers. Read-back goes
//! through `pixelflow_core::BilinearSampler` — a JIT'd 4-tap gather kernel
//! bound to the baked buffer — so fractional positions interpolate the
//! baked AA coverage smoothly. See the `cache` module docs for the
//! half-pixel coordinate convention.
//!
//! ## Layer 4: Text Layout (`text` module and `CachedText`)
//!
//! [`text()`](text::text) lays out a string as one fused `Kernel` — a sum
//! of advance-translated glyph kernels. [`CachedText::new`] composes baked
//! glyph samplers instead (with kerning), and is a `Kernel` just the same:
//!
//! ```ignore
//! use pixelflow_graphics::fonts::{CachedText, Font, GlyphCache};
//!
//! let font = Font::parse(font_data).unwrap();
//! let mut cache = GlyphCache::new();
//! cache.warm_ascii(&font, 16.0, 1.0);
//!
//! let text = CachedText::new(&font, &mut cache, "Hello, World!", 16.0, 1.0);
//! ```
//!
//! Both produce **coverage** (values in `[0, 1]`), not colors. Map coverage
//! to pixels with `render::color::Grayscale`, or blend foreground/background
//! per channel the way `core-term`'s cell renderer does.
//!
//! ## Supported Formats
//!
//! - **TTF** (TrueType): quadratic Bézier outlines, cmap formats 4 and 12,
//!   horizontal kerning (kern format 0).
//!
//! [`Outline`]: outline::Outline
//! [`Support`]: loop_blinn::Support
//!
pub mod atlas;
pub mod cache;
pub mod loader;
pub mod loop_blinn;
mod monotone;
pub mod outline;
pub mod text;
pub mod ttf;

/// The rasterizer's pixel-center convention, shared by every module in this
/// crate that bakes or samples at one: texel/pixel `(i, j)` corresponds to
/// continuous coordinate `(i + PIXEL_CENTER, j + PIXEL_CENTER)`. One
/// definition, so `atlas.rs`, `cache.rs` and `text.rs` cannot drift onto
/// different halves — `pixelflow-core`'s own `SAMPLE_CENTER`
/// (`lattice/manifold.rs`) is the same value for the same reason, restated
/// there rather than imported because it is on the other side of the crate
/// boundary and predates this module.
pub(crate) const PIXEL_CENTER: f32 = 0.5;

// Re-export font types (user-facing only)
pub use loop_blinn::{Glyph, Support};
pub use outline::{Affine, Contour, ContourError, Outline, Segment};
pub use ttf::Font;

// Re-export loader types
pub use loader::{DataSource, EmbeddedSource, FontSource, LoadedFont, MmapSource};

// Re-export text
pub use text::text;

// Re-export cache
pub use atlas::GlyphAtlas;
pub use cache::{CachedGlyph, CachedText, GlyphCache};
