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
//!      │  Layout: advances/kerning; sum of translated glyph kernels
//!      │
//! Cache Layer (GlyphCache, CachedGlyph)
//!      ↓
//!      │  Lattice::bake'd f32 AA coverage + bilinear read-back
//!      │
//! Font Layer (Font)
//!      ↓
//!      │  TTF parsing; a glyph is ONE fused coverage Kernel
//!      │
//! Loading Layer (loader: DataSource, EmbeddedSource, MmapSource, LoadedFont)
//!      ↓
//! In-Memory Font Data
//! ```
//!
//! ## Coverage semantics: antialiasing is intrinsic
//!
//! Each outline segment is a leaf kernel (`AnalyticalLine`/`AnalyticalQuad`)
//! whose crossing computes a gradient-normalized ramp
//! `clamp(d / (‖∇d‖ + ε) + 0.5, 0, 1)`. The `DX`/`DY` in the ramp are
//! symbolic `Dwrt` derivatives resolved when the kernel compiles, and the
//! chain rule carries ‖∇d‖ through every coordinate warp (`Kernel::at`), so
//! the ramp is ~1 *screen* pixel wide at any glyph scale. There is no
//! separate hard/AA mode and no jet domain — coverage is antialiased by
//! construction.
//!
//! ## Layer 1: Font Loading (`loader` module)
//!
//! Font bytes come from a [`FontSource`]: [`DataSource`] (owned bytes),
//! [`EmbeddedSource`] (bytes baked into the binary), or [`MmapSource`]
//! (zero-copy memory-mapped file). [`LoadedFont`] owns the source and
//! lends out parsed [`Font`] views.
//!
//! ## Layer 2: Glyph Compilation (`ttf` module)
//!
//! [`Font::parse`] reads the TTF tables (cmap, glyf, loca, hmtx, kern).
//! [`Font::glyph_kernel_scaled`] compiles a character's outline into one
//! coverage `Kernel`: segment kernels summed under the non-zero winding
//! rule (`abs().min(1)`), bounded by a unit-square mask, and warped through
//! the restore/scale affines. Metrics come from `advance`/`kern` and their
//! `*_by_id`/`*_scaled` variants.
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
pub mod atlas;
pub mod cache;
pub mod loader;
pub mod text;
pub mod ttf;
pub mod ttf_curve_analytical;

// Re-export font types (user-facing only)
pub use ttf::{Font, Glyph, Support};

// Re-export loader types
pub use loader::{DataSource, EmbeddedSource, FontSource, LoadedFont, MmapSource};

// Re-export text
pub use text::{text, text_cells, text_union, TextCell};

// Re-export cache
pub use atlas::GlyphAtlas;
pub use cache::{CachedGlyph, CachedText, GlyphCache};
