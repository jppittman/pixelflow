# The terminal

Glyphs and fonts live in `pixelflow-graphics`, which stays terminal-agnostic.
They are grouped here because the terminal is their consumer.

### Host versus program

- **Is:** "What stays host Rust. None of this is program." The host walks
  data (layout, outlines, cells) and composes instances. What it produces is
  a program's uniforms and shape.
- **Is not:** part of the program. Not unrolling.
- **Follows:** a loop nest outside the compiler is "a second invocation path,
  a second copy of the ctx layout". It is tolerated only until scheduling
  moves into the compiler.
- **Lives:** the-language-is-kernel §1.7.

### Cell and the cell grid

- **Is:** "A cell is one call." A cell writes its glyph's block (box and
  rows), its origin, fg and bg, and calls its glyph's piece count's program
  over its tile.
- **Is not:** a fold iteration, for now. One-pipeline §1.6's fold over cells
  invoking `P_N` is superseded. Not one fused program over the device frame
  gathering ten floats per cell from a buffer (the 2026-07-29 and 2026-09-06
  `CellGridProgram`, which is today's code), because buffers are leaving the
  language. Not tree structure ("data beats structure"). Not a glyph id
  chosen by an `if id < k` tree inside one font program (superseded
  2026-10-09; see "Font programs").
- **Follows:** metric changes are uniform writes, and anything that moves an
  extent (window, column count, tile) recompiles. L4 measured this as
  "conditional, not general". Terminal-shaped kernels leave core (D13).
- **Lives:** `pixelflow_core::{CellGridProgram, CellGridMetrics,
  CellGridShape}` (`pixelflow-core/src/lattice/cell_grid.rs`), which today
  puts terminal logic in pixelflow-core against CLAUDE.md's first
  constraint; `pixelflow-graphics/src/render/cell_grid.rs`;
  the-language-is-kernel §1.7, D13; lattice-is-the-index §2, L4.

### Frame (display)

- **Is:** a sequence of per-cell calls: host schedule, until scheduling moves
  into the compiler.
- **Is not:** a single program, for now. One-pipeline §1.6's "A frame is a
  schedule" is superseded.
- **Follows:** no per-frame heap allocation.
- **Lives:** the-language-is-kernel §1.7, D9. Today a frame is one fused
  `CellGridProgram` collapse in stripes, not per-cell calls (see Cell).

### Font programs

- **Is:** a font at a zoom level is one program per piece count: "the
  'atlas' becomes the kernel for that number of control points, everything
  else is a uniform" (JP, Q1), read literally on 2026-10-09 ("Just make the
  94 programs 28 programs"). Every value a glyph holds is a uniform, so
  every glyph with `n` pieces is the same program over a different block,
  and a glyph is drawn by writing its block into its count's program.
- **Is not:** an atlas. Not a table mapping id to glyph. Not one program
  of every glyph under a balanced `if id < k` tree (superseded 2026-10-09:
  measured, the tree blended because a mask computed in an enclosing scope
  earns no block, and every call ran every glyph's uniform-only work, 1.1
  ms a call for DejaVu's ASCII at 16 px against 0.8–13 µs for a glyph
  alone).
- **Follows:** choosing the program is choosing by a structural parameter,
  the piece count, which is what structural means. No program holds more
  than one glyph, so O1's units and the class limit are not the font's
  concern. A zoom recompiles one program per count. Bucketing counts to
  share programs is a cache policy, not adopted (JP: unbucketed).
- **Lives:** `FontPrograms`, `GlyphRows`
  (`pixelflow-graphics/src/fonts/loop_blinn/program.rs`); the-language-is-kernel
  §1.7, D9, C1. The frame still draws from the atlas until C2.

### Glyph

- **Is:** "a glyph is its pieces summed under its box": `glyph(ink, bounds)`,
  where `ink` is a balanced `sum2` tree of `one_piece` instances and
  `coverage(ink(x, y))` is taken inside the box.
- **Is not:**
  - An integral (deleted).
  - A winding sum plus a distance min, "two loops, chosen at authoring
    time".
  - Its own program: its piece count's program draws it, shared with every
    glyph of that count.
  - Scale-invariant.
  - "One fold over a table": that is today's production code and CLAUDE.md's
    wording, superseded by JP's ruling.
  - Superseded: "a winding number about a reference point" (09-08: "there
    is no reference point"); "a circle that cannot read its own sign"
    (09-09).
- **Follows:** a glyph is correct under translation only. A box test that is
  uniform over a batch is a jump.
- **Lives:** the language glyph's block (`one_piece`, `sum2`, `glyph`,
  `Row`, `Bounds`) in `pixelflow-graphics/src/fonts/loop_blinn/program.rs`,
  which the font's programs are written in; the builder glyph in
  `pixelflow-graphics/src/fonts/loop_blinn.rs` (`glyph`, `Glyph`, a fold over
  a `DiscreteManifold` piece table masked to `Support`'s box, which
  contradicts this entry), still what the atlas bakes until C2 and deleted in
  Phase D; the-language-is-kernel §1.7.

### Piece, band, box

- **Is:**
  - A **piece** is one oriented monotone arc over ten uniforms (`Row`). Its
    term is σ·A over the pixel about `(x, S·y)`, cut to the rows the piece
    reaches, with the area in closed form (`piece_area`). It is exactly zero
    outside its band.
  - A **band** is `(y > lo) & (y < hi)`, batch-uniform: "that cut changes no
    bit".
  - A **box** (`Bounds`) is the outline's bounding box as four uniforms,
    reaching half a pixel past it.
- **Is not:** a table row read at a binder index. Not one body per kind of
  piece. Not `Support`'s dilated box with uniform edges (09-09), which is
  superseded.
- **Follows:** the band's arms jump, and its guard belongs at row scope (D1).
  Pruning is exact or it is not pruning: `FLAT_ENOUGH` changed an integer by
  dropping a nearly flat curve.
- **Lives:** the-language-is-kernel §1.6–§1.7; `Row`, `Bounds`, `piece_area`,
  `one_piece` in `pixelflow-graphics/src/fonts/loop_blinn/program.rs`. The
  builder in `fonts/loop_blinn.rs` still reads each piece as a table
  row (`PIECE_ROW_COLS`) at a binder index and masks with `Support`
  (`Support::around`, dilated by `RAMP_REACH`), which contradicts this
  entry.

### Coverage and the closed form

- **Is:** the exact area of the pixel under ink,
  `min(|Σ_p σ_p·A_p|, 1)`, snapped by `COVERAGE_SNAP`. "Antialiasing: is the
  area — no ramp, no distance." Each piece's area is written in closed form:
  "a closed form is a formula to write down, not a derivation to rediscover
  under a saturation budget."
- **Is not:**
  - A `Dwrt` ramp over a distance (`f/‖∇f‖`, "not sound as a distance on
    the CPU").
  - An SDF with smoothstep.
  - `clamp(½ + sign·d)`.
  - A quadrature.
  - Additive across glyphs: "summing per-glyph coverages reaches 2 where ink
    overlaps, and 2 is not a coverage". Compose the signed areas, then take
    coverage once.
- **Follows:** a scaling `at` covers a pixel of the wrong size. `Kernel::dx`
  and `dy` remain only for kernels that do ramp on a distance.
- **Lives:** `pixelflow-graphics/src/fonts/loop_blinn.rs`
  (`RisingArc::pixel_area`, `Glyph::kernel`, `COVERAGE_SNAP`),
  `fonts/monotone.rs`; CLAUDE.md "Execution Notes";
  `2026-09-23-a-glyph-is-a-formula.md` (2026-09-29 note). Today
  `.claude/agents/numerics.md` and `pixelflow-graphics.md` describe gradient
  antialiasing and Jet2, both stale.

### Contour

- **Is:** a closed chain of segments. Closure is "the precondition, not a
  convention" that makes masking a glyph to its box exact.
- **Is not:** whatever `Contour::segments` hands back. The invariant once
  lived in prose.
- **Follows:** a closed contour's winding, and now its signed area, is zero
  at every exterior point. An open contour would fill a half-plane instead of
  being rejected.
- **Lives:** `pixelflow-graphics/src/fonts/outline.rs`,
  `Contour::new(segments) -> Result<Self, ContourError>` (`Empty`,
  `NotClosed`); loop-blinn-glyph §8.

### Geometry on the host

- **Is:** "Every affine map (component placement, em scale, screen flip, pen
  position) is applied to control points before a kernel exists." Quadratics
  are closed under affine maps, so nothing is lost.
- **Is not:** a coordinate warp applied to a finished kernel built in a unit
  square. That arrangement put a `½` into the arena where the e-graph could
  reassociate it (L2). Not `Glyph::at` (retracted in a-run-is-a-glyph §9).
- **Follows:** every glyph is born placed. A zoom rescales the outlines and
  rewrites the uniforms.
- **Lives:** `pixelflow-graphics/src/fonts/{ttf,outline,text}.rs`;
  `docs/plans/2026-09-08-loop-blinn-glyph.md` §5.

### Run

- **Is:** a text run is the monoid product of placed glyphs,
  `text() = Glyph::over(layout(..).map(place))`. "A run's coverage does not
  depend on character order."
- **Is not:** a merge of outlines that destroys per-character structure. Not
  one buffer per character, which broke at five characters.
- **Follows:** `text("") == Glyph::empty()`. Order independence and
  associativity are testable laws.
- **Lives:** `loop_blinn::run`, `Glyph::over`, `Glyph::empty`
  (`pixelflow-graphics/src/fonts/loop_blinn.rs`), `fonts/text.rs`;
  `pixelflow-graphics/tests/run_is_a_glyph.rs`;
  `2026-09-09-a-run-is-a-glyph.md`.

### Atlas

- **Is:** today, a host-filled memo of one font at one tile size, read by
  bilinear resampling. JP: "The 'atlas' becomes the kernel for that number of
  control points."
- **Is not:** a plain tabulation, because it is read bilinearly. Not kept
  until D1 (one-pipeline Q3, superseded).
- **Follows:** it goes once C2 measures the alternative (O4). An atlas-free
  frame changes pixels wherever density ≠ 1, so pins are re-baselined in a
  commit of their own.
- **Lives:** `GlyphAtlas` (`pixelflow-graphics/src/fonts/atlas.rs`),
  `GlyphCache`, `CachedGlyph`, `CachedText` (`fonts/cache.rs`),
  `BilinearSampler` (`pixelflow-core/src/lattice/mod.rs`);
  the-language-is-kernel §1.7, D10.

### Zoom level

- **Is:** structural. A new tile extent is a new program, so a zoom
  recompiles the font.
- **Is not:** a scaling `at`.
- **Follows:** the host rescales the outlines and rewrites the font's
  uniforms.
- **Lives:** the-language-is-kernel §1.7.
