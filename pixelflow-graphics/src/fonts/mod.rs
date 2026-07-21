//! # Font Rendering Pipeline
//!
//! Bridges vector font formats (TTF) to glyph coverage manifolds.
//!
//! ## Architecture: Four Layers
//!
//! ```text
//! Text Layer (text(), CachedText)
//!      ↓
//!      │  Layout: kerning, advances; Sum of positioned glyphs
//!      │
//! Cache Layer (GlyphCache, CachedGlyph)
//!      ↓
//!      │  Lattice-collapsed f32 AA coverage + bilinear read-back
//!      │
//! Font Layer (Font, Glyph)
//!      ↓
//!      │  TTF parsing; glyphs as analytical curve manifolds
//!      │
//! Loading Layer (loader: DataSource, EmbeddedSource, MmapSource, LoadedFont)
//!      ↓
//! In-Memory Font Data
//! ```
//!
//! ## Coverage semantics: hard over `Field`, antialiased over `Jet2`
//!
//! The whole analytical pipeline (`AnalyticalLine`/`AnalyticalQuad` crossing
//! kernels, `Geometry` winding accumulation, affine transforms, `Glyph`,
//! `Translate`, `Sum`) is generic over the evaluation domain:
//!
//! - Over **`Field`** coordinates, derivatives are zero and every edge
//!   crossing degenerates to a hard 0/1 step — the classic aliased
//!   winding-number test.
//! - Over **`Jet2`** coordinates (2D dual numbers), each crossing computes a
//!   gradient-normalized ramp `clamp(d / (‖∇d‖ + ε) + 0.5, 0, 1)` that is
//!   ~1 *screen* pixel wide at any glyph scale — the chain rule carries
//!   ‖∇d‖ through every coordinate transform.
//!
//! [`crate::render::aa::Antialiased`] (or the [`crate::render::aa::aa`]
//! function) is the bridge: it wraps a domain-generic coverage manifold and
//! seeds `Jet2` derivatives in screen space, exposing an antialiased
//! `Manifold<(Field, Field, Field, Field), Output = Field>`.
//!
//! ```ignore
//! use pixelflow_graphics::fonts::{text, Font};
//! use pixelflow_graphics::render::aa::aa;
//!
//! let font = Font::parse(font_data).unwrap();
//! let hard = text(&font, "Hello", 16.0);   // aliased coverage
//! let smooth = aa(text(&font, "Hello", 16.0)); // antialiased coverage
//! ```
//!
//! ## Layer 1: Font Loading (`loader` module)
//!
//! Font bytes come from a [`FontSource`]: [`DataSource`] (owned bytes),
//! [`EmbeddedSource`] (bytes baked into the binary), or [`MmapSource`]
//! (zero-copy memory-mapped file). [`LoadedFont`] owns the source and
//! lends out parsed [`Font`] views.
//!
//! ## Layer 2: Glyph Decomposition (`ttf` module)
//!
//! [`Font::parse`] reads the TTF tables (cmap, glyf, loca, hmtx, kern).
//! [`Font::glyph_scaled`] compiles a character's outline into a
//! [`Glyph`]: lines and quadratic Béziers as analytical crossing kernels,
//! composed with bounds checks and affine transforms. Metrics come from
//! `advance`/`kern` and their `*_by_id`/`*_scaled` variants.
//!
//! ## Layer 3: Glyph Caching (`cache` module)
//!
//! Analytical evaluation walks every curve per sample. `GlyphCache` bakes
//! glyphs once per (character, size bucket): `CachedGlyph::new` evaluates
//! the glyph through `Antialiased` and collapses it over a `Lattice` into a
//! `DiscreteManifold` of f32 coverage sampled at pixel centers. Read-back
//! goes through the [`crate::render::bilinear::Bilinear`] combinator, so
//! fractional positions interpolate the baked AA coverage smoothly. See the
//! `cache` module docs for the half-pixel coordinate convention.
//!
//! ## Layer 4: Text Layout (`text` module and `CachedText`)
//!
//! [`text()`](text::text) lays out a string as a `Sum` of translated
//! analytical glyphs (kerning-free advance layout). [`CachedText::new`]
//! does the same over cached glyphs, with kerning:
//!
//! ```ignore
//! use pixelflow_graphics::fonts::{CachedText, Font, GlyphCache};
//!
//! let font = Font::parse(font_data).unwrap();
//! let mut cache = GlyphCache::new();
//! cache.warm_ascii(&font, 16.0);
//!
//! let text = CachedText::new(&font, &mut cache, "Hello, World!", 16.0);
//! ```
//!
//! Both produce **coverage** manifolds (`Output = Field`, values in
//! `[0, 1]`), not colors. Map coverage to pixels with
//! `render::color::Grayscale`, or blend foreground/background per channel
//! the way `core-term`'s cell renderer does.
//!
//! ## Supported Formats
//!
//! - **TTF** (TrueType): quadratic Bézier outlines, cmap formats 4 and 12,
//!   horizontal kerning (kern format 0).
//!
pub mod cache;
pub mod loader;
pub mod text;
pub mod ttf;
pub mod ttf_curve_analytical;

// Re-export font types (user-facing only; internal geometry types stay in ttf module)
pub use ttf::{Font, Glyph};

// Re-export loader types
pub use loader::{DataSource, EmbeddedSource, FontSource, LoadedFont, MmapSource};

// Re-export text
pub use text::text;

// Re-export cache
pub use cache::{CachedGlyph, CachedText, GlyphCache};
