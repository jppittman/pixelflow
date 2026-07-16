
//! # PixelFlow Graphics
//!
//! Turns **algebraic manifolds into pixels** through three composable tiers: colors, fonts, and materialization.
//!
//! ## The Graphics Pipeline: Three Tiers
//!
//! ```text
//! Manifold (Algebra)        Pixels (Discrete u32)
//!      ↓                             ↓
//!      │  Tier 1: Colors            │
//!      │  ├─ ColorCube + At         │
//!      │  ├─ Color (semantic)       │
//!      │  └─ Grayscale              │
//!      ↓                             ↓
//!      │  Tier 2: Fonts             │
//!      │  ├─ Font (glyph atlas)     │
//!      │  ├─ GlyphCache             │
//!      │  └─ Text (layout)          │
//!      ↓                             ↓
//!      │  Tier 3: Materialization   │
//!      │  ├─ execute()              │
//!      │  ├─ execute_stripe()       │
//!      │  └─ Parallel rendering     │
//!      ↓                             ↓
//!    Frame (pixel buffer)
//! ```
//!
//! ## Tier 1: Colors
//!
//! **Colors ARE coordinates.** The `ColorCube` manifold interprets its input as RGBA:
//! - X = Red, Y = Green, Z = Blue, W = Alpha
//!
//! Use `At` (the universal contramap) to navigate the color cube:
//!
//! ### Solid Colors
//! ```ignore
//! use pixelflow_graphics::ColorCube;
//! use pixelflow_core::combinators::At;
//!
//! // Solid red: navigate to (1, 0, 0, 1) in color space
//! let red = At { inner: ColorCube, x: 1.0, y: 0.0, z: 0.0, w: 1.0 };
//! ```
//!
//! ### Gradients
//! ```ignore
//! use pixelflow_graphics::ColorCube;
//! use pixelflow_core::{combinators::At, X};
//!
//! // Red varies with screen X position
//! let gradient = At { inner: ColorCube, x: X / 255.0, y: 0.5, z: 0.5, w: 1.0 };
//! ```
//!
//! ### Blending
//! ```ignore
//! // Blend is just coordinate arithmetic before At
//! let blended = At {
//!     inner: ColorCube,
//!     x: t * r1 + (1.0 - t) * r2,
//!     y: t * g1 + (1.0 - t) * g2,
//!     z: t * b1 + (1.0 - t) * b2,
//!     w: t * a1 + (1.0 - t) * a2,
//! };
//! ```
//!
//! ### Semantic Colors (`Color` enum)
//! **For human thinking**: ANSI colors, indexed palette, or RGB true color.
//!
//! ```ignore
//! use pixelflow_graphics::Color;
//!
//! let red = Color::Rgb(255, 0, 0);
//! let named = Color::Named(NamedColor::Red);
//! ```
//!
//! ### Grayscale
//! ```ignore
//! use pixelflow_graphics::Grayscale;
//! use pixelflow_core::X;
//!
//! let gray_gradient = Grayscale(X / 255.0);  // R=G=B=value, A=1
//! ```
//!
//! ## Tier 2: Fonts
//!
//! Font rendering bridges vector glyphs (TTF/OTF) to pixels. Typically used in terminals for character rendering.
//!
//! ### Font Loading
//! Fonts are loaded from multiple sources (embedded, filesystem, or memory-mapped):
//!
//! ```ignore
//! use pixelflow_graphics::fonts::{Font, FontSource, LoadedFont};
//!
//! // Load from memory
//! let font = Font::from_source(FontSource::Embedded)?;
//! ```
//!
//! ### Glyph Caching
//! Glyphs are **lazily cached** per size. A `GlyphCache` stores rasterized glyphs to avoid redundant computation.
//!
//! ```ignore
//! use pixelflow_graphics::GlyphCache;
//!
//! let cache = GlyphCache::new();
//! let glyph = cache.get_or_rasterize('A', font, size)?;
//! ```
//!
//! ### Text Layout and Rendering
//! `CachedText` combines font, cache, layout, and rendering into a single manifold.
//!
//! ## Tier 3: Materialization (Manifold → Pixels)
//!
//! ### The Rasterizer
//! Bridges the gap between continuous manifolds and discrete framebuffers.
//!
//! ```ignore
//! use pixelflow_graphics::render::{execute, TensorShape};
//!
//! // Render a 800x600 color manifold to a pixel buffer
//! execute(&color_manifold, &mut framebuffer, TensorShape::new(800, 600));
//! ```
//!
//! The rasterizer:
//! 1. **Samples** the manifold at pixel center coordinates (x + 0.5, y + 0.5)
//! 2. **Uses SIMD**: Processes `PARALLELISM` pixels per iteration (16-64 depending on target)
//! 3. **Falls back to scalar**: Handles edge pixels one at a time
//! 4. **Caches across frames**: Manifold evaluation is cached between frames if size unchanged
//!
//! ### Parallel Rendering
//! For large framebuffers, rendering can be parallelized via work-stealing:
//!
//! ```ignore
//! use pixelflow_graphics::render::render_parallel;
//!
//! render_parallel(&manifold, &mut framebuffer, shape, num_threads);
//! ```
//!
//! ## Pixel Formats
//!
//! Different platforms use different byte orders:
//!
//! | Type | Layout | Usage |
//! |------|--------|-------|
//! | `Rgba8` | [R, G, B, A] | macOS Cocoa, Web |
//! | `Bgra8` | [B, G, R, A] | Linux X11 |
//! | `Discrete` | Packed u32 RGBA | IR (internal) |
//!
//! Platform aliases handle conversion automatically.
//!
//! ## Design Philosophy
//!
//! **Manifolds all the way down**: Every visual element (colors, text, backgrounds) is composed as manifolds.
//! No immediate-mode drawing, no retained-mode state machines. Just pure functions from coordinates to pixels.
//!
//! **Zero-cost abstractions**: The type system captures the computation graph.
//! The compiler monomorphizes the entire pipeline into a single fused kernel.
//!
//! **Polymorphic evaluation**: The same manifold pipeline can evaluate:
//! - With concrete SIMD (normal rendering)
//! - With automatic differentiation (antialiasing via `Jet2`)
//! - On any hardware target (via `Field` abstraction)

pub mod animation;
pub mod baked;
pub mod fonts;
pub mod image;
pub mod mesh;
pub mod patch;
pub mod render;
pub mod scene3d;
pub mod shapes;
pub mod spatial_bsp;
pub mod subdiv;
pub mod subdivision;
pub mod transform;

pub use spatial_bsp::{Positioned, SpatialBSP};

pub use baked::Baked;
pub use transform::Scale;

// Re-export fonts (user-facing types only; internal combinators like Affine/Sum stay in fonts::ttf)
pub use fonts::{CachedGlyph, CachedText, Font, Glyph, GlyphCache};

// Re-export render
pub use render::color::{
    AttrFlags, Bgra8, BgraColorCube, CocoaPixel, Color, ColorCube, Grayscale, NamedColor, Pixel,
    PlatformColorCube, Rgba8, RgbaColorCube, WebPixel, X11Pixel,
};
pub use render::frame::Frame;
pub use render::rasterizer::rasterize;

// Re-export core types for convenience
// Field/Discrete are doc(hidden) - use manifolds instead of direct field manipulation
pub use pixelflow_core::{Manifold, ManifoldExt, Map, W, X, Y, Z};
