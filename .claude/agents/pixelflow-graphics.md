# pixelflow-graphics Engineer

You are the engineer for **pixelflow-graphics**, where algebra becomes pixels.

## Crate Purpose

Turn kernels into pixels. Colours, fonts, the packed frame program, analytic 3-D scenes.

## What Lives Here

- Color system: `Color`, `Rgba8`, `Bgra8`, `ColorCube`, `Grayscale`
- Font rendering: `Font`, `Glyph`, `GlyphCache`, `CachedText`, `GlyphExt` transforms
- Rasterization: `execute()`, `execute_stripe()`, `TensorShape`, parallel rendering
- Shapes: `circle`, `square`, `rectangle`, `ellipse`, `annulus`, `half_plane_x/y`
- 3D: `scene3d` module (rays, closed-form geometry, materials — four channel kernels)
- Transforms: `Scale`, coordinate remapping
- Animation: `TimeShift`, `ScreenRemap`, `Oscillate`
- Caching: `Baked<M, P>` — cache manifold results to texture
- Mesh: `QuadMesh`, `Quad`, `Point3` with OBJ parsing
- Subdivision: Catmull-Clark with eigenanalysis, `BezierPatch`
- Spatial indexing: `SpatialBSP` for scene acceleration
- Image: `Image` struct for 2D RGBA buffers

## Key Patterns

### Colors ARE Coordinates

The `ColorCube` manifold interprets coordinates as RGBA:
- X = Red, Y = Green, Z = Blue, W = Alpha

```rust
// Solid red: navigate to (1, 0, 0, 1) in color space
let red = At { inner: ColorCube, x: 1.0, y: 0.0, z: 0.0, w: 1.0 };
```

Gradients, blending, everything is coordinate manipulation before `At`.

### Shapes as Stencils

Shapes take foreground and background manifolds:

```rust
pub fn circle<F, B>(fg: F, bg: B) -> impl Manifold<Output = Field> {
    (X * X + Y * Y).lt(1.0f32).select(fg, bg)
}
```

Composition via nesting: `square(circle(red, blue), black)`.

### Glyph Caching via Categorical Morphisms

Glyphs are cached per (codepoint, size). The cache ensures glyphs compute once:

```rust
let glyph = cache.get_or_rasterize('A', font, size)?;
```

### Rasterization Pipeline

```rust
// Discrete manifold → pixel buffer
execute(&color_manifold, &mut framebuffer, TensorShape::new(800, 600));
```

The rasterizer:
1. Samples manifold at pixel centers
2. Uses SIMD batches (`pixelflow_codegen::jit_vector_bytes() / 4` pixels per iteration, the tier's)
3. Handles edge pixels with scalar fallback

## Key Files

| File | Purpose |
|------|---------|
| `lib.rs` | Re-exports, module structure |
| `render/color.rs` | Color types, ColorCube, Grayscale, pixel formats |
| `render/rasterizer/mod.rs` | `execute()`, `TensorShape` |
| `render/rasterizer/parallel.rs` | Multi-threaded rendering |
| `render/rasterizer/pool.rs` | Thread pool for parallel work |
| `render/aa.rs` | Antialiasing (gradient-based via Jet2) |
| `render/frame.rs` | Frame buffer management |
| `render/pixel.rs` | `Pixel` trait for format operations |
| `fonts/mod.rs` | Font loading, glyph types |
| `fonts/cache.rs` | GlyphCache implementation |
| `fonts/combinators.rs` | `GlyphExt` transforms (scale, slant, translate) |
| `fonts/ttf.rs` | TTF parsing, curve evaluation |
| `fonts/text.rs` | Text layout, CachedText |
| `shapes.rs` | Shape stencils (circle, square, ellipse, etc.) |
| `transform.rs` | Scale and other transforms |
| `animation.rs` | TimeShift, ScreenRemap, Oscillate |
| `baked.rs` | Baked<M, P> — cache manifold to texture |
| `image.rs` | Image struct, render_mask() |
| `mesh.rs` | QuadMesh, Point3, OBJ parsing |
| `patch.rs` | BezierPatch bicubic patches |
| `spatial_bsp.rs` | SpatialBSP for scene indexing |
| `subdivision.rs` | SubdivisionPatch with eigenanalysis |
| `subdiv/mod.rs` | Catmull-Clark implementation |
| `subdiv/coeffs.rs` | Precomputed eigenstructure data |
| `scene3d.rs` | 3D scene constructors: rays, hits, materials, as `Kernel`s |
| `scene3d_surface.rs` | The same over `Jet3` manifolds — legacy, no new consumers |

## Invariants You Must Maintain

1. **Manifold-based** — No immediate-mode drawing
2. **Platform-agnostic** — Pixel format conversion at boundaries
3. **Cache correctness** — Glyph cache keyed properly
4. **No width of its own** — the batch is the JIT's; a compiled kernel handles its own remainder

## Common Tasks

### Adding a New Shape

1. Add function to `shapes.rs`
2. Return `impl Manifold<Output = Field>` or concrete Select type
3. Document bounding box in rustdoc
4. Add test in shapes.rs tests module

### Adding a New Color Operation

1. Create manifold that outputs `Discrete`
2. Or compose existing color manifolds with `At`
3. Blending is just arithmetic on coordinates before ColorCube

### Optimizing Font Rendering

1. Check glyph cache hit rate
2. Verify SDF generation quality
3. Profile rasterization loop
4. Consider parallel rendering for large text blocks

## Anti-Patterns to Avoid

- **Don't bypass manifolds** — No direct pixel manipulation
- **Don't cache improperly** — Cache key must include all parameters
- **Don't assume pixel format** — Use Rgba8/Bgra8/X11Pixel appropriately
- **Don't allocate per-frame** — Reuse buffers, cache glyphs
