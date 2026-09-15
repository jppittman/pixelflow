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
//!      │  An outline's winding number, as one Kernel
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
//! ## Coverage semantics: an exact winding, an antialiased distance
//!
//! A glyph's coverage is `min(|w|, 1)` for the winding number `w` under the
//! non-zero rule, and [`loop_blinn`] computes `w` **exactly**: hard masks
//! selecting signed constants, relative to a reference point, with
//! Loop–Blinn's implicit `u² − v` for the sliver each quadratic bulges past
//! its chord. What is antialiased is a separate number — the distance to
//! the nearest piece of outline — ramped as
//! `inside ? min(1, ½ + d) : max(0, ½ − d)`.
//!
//! `d` is gradient-normalized: divided by `‖∇d‖` with the `DX`/`DY` as
//! symbolic `Dwrt` derivatives resolved when the kernel compiles, so the
//! chain rule carries the scale through every coordinate warp
//! (`Kernel::at`) and the ramp is ~1 *screen* pixel wide at any glyph
//! scale. There is no separate hard/AA mode and no jet domain — coverage is
//! antialiased by construction.
//!
//! Keeping the winding and the ramp apart is what makes a mis-decided
//! comparison cost a rounding rather than half a unit of coverage. The
//! formulation this replaced made a crossing's existence *be* its coverage
//! (see docs/plans/2026-09-08-loop-blinn-glyph.md).
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
