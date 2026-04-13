# Lattice as Representable Functor: Unified Scheduling

## The Idea (Apr 3, 2026)

The Lattice doesn't just describe dimensions — it IS the evaluator. A manifold is a function. A Lattice is a demand for evaluation. Binding a kernel to a Lattice collapses it — like observation collapsing a wavefunction. Nothing computes until a Lattice demands it.

There is no `eval(point)` vs `eval(lattice)`. There is only `Lattice::collapse(manifold)`. A single point evaluation is just `Lattice<1,1,1,1>` with fixed coordinates.

## Current State

```rust
// Today: eval takes a point, loops are in Rust
let shader = kernel!(|| { ... });
for y in 0..1080 {
    for x in (0..1920).step_by(4) {
        shader.eval((x_field, y_field, z, w));  // compiler can't see this loop
    }
}
```

The compiler can't optimize the loop structure because it can't see it. The variance analysis knows `sin(Z*0.3)` is X-invariant, but there's no loop to hoist it out of. The scheduling is invisible — buried in Rust `for` loops outside the DSL.

## Design

### The Lattice binds and collapses

```rust
let shader = kernel!(|| {
    let scale = 2.0 / 1080.0;
    let x = (X - 960.0) * scale;
    let y = (540.0 - Y) * scale;
    let time = Z + 1.3;
    sin(time * 0.3) * (x * x + y * y).sqrt()
});

// The Lattice IS the evaluator.
// Dimensions: X ∈ [0, 1920), Y ∈ [0, 1080), Z = time, W = 0
// The scheduler sees the DAG, reads variance, emits nested loops.
let pixels = Lattice::new(1920, 1080)
    .with_z(time)
    .collapse(&shader);
```

Or more concisely:

```rust
let pixels = Lattice::frame(1920, 1080, time).collapse(&shader);
```

### What `collapse` does

1. **Reads the kernel's DAG** (the optimized expression from the e-graph)
2. **Computes variance** for each node (which coordinates it depends on)
3. **Determines loop nesting** from the Lattice shape:
   - Fixed dimensions (Z, W with concrete values) → substitute as constants
   - Loop dimensions (X, Y) → nested loops, outermost first by convention
4. **Topological sorts by scope** — nodes that don't depend on inner loops go in outer setup blocks
5. **Emits JIT code** with nested loops and hoisting at each boundary
6. **Runs** the JIT kernel, fills the output buffer

### A single point is just a degenerate Lattice

```rust
// These are equivalent:
shader.eval((x, y, z, w))
Lattice::point(x, y, z, w).collapse(&shader)
```

`Lattice::point` is `Lattice<1,1,1,1>` with all four coordinates fixed. The scheduler sees zero loop dimensions — everything is constant. The JIT emits straight-line code. Same result, unified path.

### The Lattice determines which variables are loops

The kernel uses X, Y, Z, W as abstract coordinates. The Lattice says what they mean:

| Lattice constructor | X | Y | Z | W |
|---|---|---|---|---|
| `Lattice::frame(1920, 1080, t)` | pixel (loop) | scanline (loop) | t (const) | 0 (const) |
| `Lattice::scanline(1920, y, t)` | pixel (loop) | y (const) | t (const) | 0 (const) |
| `Lattice::point(x, y, z, w)` | x (const) | y (const) | z (const) | w (const) |
| `Lattice::volume(w, h, d)` | loop | loop | loop | 0 (const) |
| `Lattice::new(w, h).with_z(t)` | loop | loop | t (const) | 0 (const) |

Fixed coordinates are substituted into the DAG before compilation (partial evaluation → constant folding). Loop coordinates become nested loops with variance-driven hoisting.

### The `.at()` is INSIDE the kernel

The kernel already uses `.at()` for coordinate transforms:

```rust
let shader = kernel!(|| {
    let warped = texture.at(rotate(X, Y, angle));  // contramap inside
    warped * brightness
});
```

The `.at()` calls are part of the expression DAG. The Lattice doesn't wrap the kernel — it evaluates it. The scheduler reads the DAG (including `.at()` nodes) and emits the right code.

### No `eval_scanline`, no special paths

Today we have:
- `eval((x, y, z, w))` — single point
- `eval_scanline(&xs, y, z, w, &mut outputs)` — scanline-optimized path
- Different calling conventions, different code paths

With Lattice-as-evaluator:
- `Lattice::frame(w, h, t).collapse(&shader)` — the scheduler picks the optimal evaluation strategy
- For a 2D lattice with X inner / Y outer, it naturally produces the scanline loop
- No separate `eval_scanline` — the Lattice shape determines the strategy

### Connection to the type system

The Lattice carries its shape in the type (const generics or runtime dimensions):

```rust
// Option A: Const generics (compile-time shape)
struct Lattice<const W: usize, const H: usize>;

// Option B: Runtime dimensions (more flexible)
struct Lattice {
    dims: [LatticeAxis; 4],  // X, Y, Z, W — each is Loop(size) or Fixed(value)
}

enum LatticeAxis {
    Loop(usize),     // This coordinate varies over [0, size)
    Fixed(f32),      // This coordinate is a constant
}
```

Option B is simpler and more flexible. The JIT compiles at runtime anyway — it can read the axis types and emit the right loop structure.

### Connection to variance

The Lattice tells you which coordinates are loops. Variance tells you which coordinates each node depends on. Together:

- Node depends on `{X}`, X is a loop → node is in the X loop body
- Node depends on `{Z}`, Z is fixed → node is constant-folded (substituted before compilation)
- Node depends on `{Y}`, Y is a loop, X is inner → node is in the Y setup (hoisted out of X loop)
- Node depends on `{X, Y}` → node is in the innermost loop

The Lattice converts the variance bitset into a scope level. The topological sort emits code at the right scope. This is the two-lattice architecture from the brainstorm doc, made concrete.

## What changes

| Component | Current | With Lattice evaluator |
|---|---|---|
| `Manifold::eval` | Takes `(Field, Field, Field, Field)` | Deprecated — use Lattice |
| `ScanlineJitManifold` | Separate type with `eval_scanline` | Gone — Lattice handles it |
| Loop structure | In Rust, invisible to compiler | In Lattice, drives JIT scheduling |
| Hoisting | Manual or absent | Automatic — variance + Lattice shape |
| Partial evaluation | Not done | Fixed Lattice axes → constants before JIT |

## Lattice as Monad (Apr 3, 2026)

The Lattice is a monad, NOT a comonad.

- No `extract` (peek at one point) — you render the whole grid, period
- No `extend` (neighborhood operation) — that implies streaming/incremental, not the model
- You collapse ONCE. The result is a manifold (discrete buffer lookup). It re-enters the algebra.

### Monadic structure

- **return**: `Lattice::frame(w, h, t).bind(&shader)` — wrap a manifold in an unevaluated demand
- **collapse** (join): `Lattice<Manifold> → DiscreteManifold` — evaluate the grid, produce a buffer
- **bind**: chain lattice computations — multi-pass rendering

```rust
// Pass 1: render shader → buffer (a discrete manifold)
let buffer = Lattice::frame(1920, 1080, time).collapse(&shader);

// buffer IS a manifold. It re-enters the algebra.
// Pass 2: use buffer as input to another kernel
let post = kernel!(|buf: Manifold| some_effect(buf));
let result = Lattice::frame(1920, 1080, time).collapse(&post(buffer));
```

### Join IS tiling

`Lattice<Lattice<A>> → Lattice<A>` — a lattice of tiles collapses to a flat grid. Tiling is monadic join:

```rust
// Outer lattice: 60×34 tiles
// Inner lattice: 32×32 pixels per tile
// join: 1920×1088 pixel grid (with 8 rows of padding)
let tiled = Lattice::tiled(1920, 1080, 32, 32, time);
let result = tiled.collapse(&shader);
```

### Evaluated lattice = discrete manifold

The collapsed result is a manifold — `(x, y) → value` via buffer lookup. It composes with everything:

```rust
let buffer: DiscreteManifold = Lattice::frame(w, h, t).collapse(&shader);
buffer.eval((x, y, 0, 0))   // lookup by coordinate
other_shader.at(buffer)       // use as texture input
buffer + offset               // arithmetic on discrete manifold
```

This is how multi-pass rendering works in the algebra. First pass collapses to a buffer. The buffer IS a manifold. Second pass reads from it. No special "texture" type — textures are just collapsed lattices.

### A point IS a degenerate lattice

```rust
// These are the same thing:
Lattice::point(x, y, z, w).collapse(&shader)   // 1×1×1×1 lattice
shader.eval((x, y, z, w))                       // legacy point eval
```

The point eval path is just `Lattice<1,1,1,1>` with all coordinates fixed. The scheduler sees zero loop dimensions and emits straight-line code.

## Open questions

1. **JIT compilation caching**: For animation, Z changes per frame. Do we re-JIT every frame? Or compile once with Z as a runtime parameter? The current scanline JIT already passes Z as a register — the compiled code stays the same, only the Z value changes. The Lattice should cache the compiled kernel and only recompile when the expression changes.

2. **Parallel evaluation**: Scanlines are independent. The Lattice should drive parallelism — partition Y across worker threads, each evaluating a horizontal band. The actor scheduler already has the thread pool.

3. **Memory model**: A collapsed lattice (discrete manifold) owns a buffer. How does this interact with the ping-pong frame buffer strategy? The DiscreteManifold should be able to alias into the existing frame buffer without copying.

4. **Pull-based preservation**: The Lattice is the ONLY thing that forces evaluation. Until `collapse` is called, the manifold is just a function. This preserves pull-based: the Lattice is the demand, the manifold is the supply.

## Trait Design: Representable Functor

The Lattice is a representable functor. `tabulate` evaluates a function at every point in the domain. `index` reads back by coordinate. The isomorphism: `index . tabulate = id`.

### Core traits (pixelflow-core)

```rust
/// A finite domain that can generate coordinates.
///
/// This is the `Rep` part of a representable functor. Each index
/// maps to a coordinate tuple that a Manifold can evaluate.
pub trait LatticeDomain {
    /// Number of points in this domain.
    fn len(&self) -> usize;

    /// Which coordinate variables are loop dimensions (variance bitset).
    /// Fixed dimensions have been substituted as constants.
    fn loop_vars(&self) -> &[u8];

    /// Map an index to concrete coordinate values.
    /// Returns (x, y, z, w) for this point in the domain.
    fn coord(&self, index: usize) -> (f32, f32, f32, f32);
}

/// A Lattice collapses a Manifold over a finite domain.
///
/// `collapse` = `tabulate` in representable functor terms.
/// `collapse_with` = `foldMap` in Foldable terms.
///
/// The result of `collapse` is a `DiscreteManifold` — a buffer
/// that IS a Manifold (indexable by coordinate). It re-enters
/// the algebra.
pub trait Lattice: LatticeDomain {
    /// Collapse: evaluate the manifold at every point in the domain.
    /// Returns a discrete manifold (buffer lookup).
    ///
    /// This is `tabulate`: `(Rep -> a) -> F a`
    fn collapse<M: Manifold<Output = Field>>(&self, manifold: &M) -> DiscreteManifold;

    /// Collapse with reduction: fold a dimension away via a binary operator.
    /// The reduced dimension is consumed — the result has one fewer loop variable.
    ///
    /// This is `foldMap`: `Monoid m => (a -> m) -> F a -> m`
    fn collapse_with<M: Manifold<Output = Field>>(
        &self,
        op: ReduceOp,
        manifold: &M,
    ) -> Field;
}

/// The result of collapsing a Lattice. A buffer of values that IS a Manifold.
///
/// `index`: `F a -> Rep -> a` — read by coordinate.
/// This closes the representable functor isomorphism:
/// `index(collapse(f)) = f` (up to discretization).
pub struct DiscreteManifold {
    /// Raw pixel/value buffer.
    buffer: Vec<f32>,
    /// Width of the grid (X dimension).
    width: usize,
    /// Height of the grid (Y dimension).
    height: usize,
}

impl Manifold for DiscreteManifold {
    type Output = Field;

    /// `index`: read by coordinate. This IS the representable functor's index.
    fn eval(&self, coords: (Field, Field, Field, Field)) -> Field {
        // Convert continuous coordinates to discrete indices
        // Bilinear interpolation or nearest-neighbor
        // ...
    }
}

/// Binary operators for reduction.
pub enum ReduceOp {
    Add,    // Monoid: (0, +)
    Mul,    // Monoid: (1, *)
    Min,    // Monoid: (+inf, min)
    Max,    // Monoid: (-inf, max)
}
```

### Concrete lattices

```rust
/// A 2D frame lattice: X varies per pixel, Y per scanline.
/// Z and W are fixed (frame time and layer).
pub struct FrameLattice {
    pub width: usize,
    pub height: usize,
    pub z: f32,  // frame time
    pub w: f32,  // layer (usually 0)
}

impl LatticeDomain for FrameLattice {
    fn len(&self) -> usize { self.width * self.height }
    fn loop_vars(&self) -> &[u8] { &[0, 1] }  // X=0, Y=1 are loops
    fn coord(&self, index: usize) -> (f32, f32, f32, f32) {
        let x = (index % self.width) as f32;
        let y = (index / self.width) as f32;
        (x, y, self.z, self.w)
    }
}

impl Lattice for FrameLattice {
    fn collapse<M: Manifold<Output = Field>>(&self, manifold: &M) -> DiscreteManifold {
        // This is where the JIT scheduler lives:
        // 1. Read manifold's DAG
        // 2. Compute variance
        // 3. Substitute z/w as constants (partial evaluation)
        // 4. Topological sort by scope (Y outer, X inner)
        // 5. Hoist Y-invariant to scanline setup, X-invariant to frame setup
        // 6. JIT compile with nested loops
        // 7. Run, fill buffer
        // 8. Return DiscreteManifold
        todo!()
    }

    fn collapse_with<M: Manifold<Output = Field>>(
        &self,
        op: ReduceOp,
        manifold: &M,
    ) -> Field {
        // Reduce: evaluate at every point, fold with op
        todo!()
    }
}

/// A 1D scanline lattice: only X varies.
pub struct ScanlineLattice {
    pub width: usize,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

/// A degenerate lattice: all coordinates fixed. Single point.
pub struct PointLattice {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Lattice for PointLattice {
    fn collapse<M: Manifold<Output = Field>>(&self, manifold: &M) -> DiscreteManifold {
        // Just eval at the single point — degenerate case
        let val = manifold.eval((
            Field::from(self.x),
            Field::from(self.y),
            Field::from(self.z),
            Field::from(self.w),
        ));
        DiscreteManifold {
            buffer: vec![/* extract f32 from Field */],
            width: 1,
            height: 1,
        }
    }
}
```

### Usage in the render loop

```rust
let shader = kernel!(|| {
    let scale = 2.0 / 1080.0;
    let x = (X - 960.0) * scale;
    let y = (540.0 - Y) * scale;
    let time = Z + 1.3;
    sin(time * 0.3) * (x * x + y * y).sqrt()
});

// Frame render: Lattice collapses the shader
let lattice = FrameLattice { width: 1920, height: 1080, z: time, w: 0.0 };
let frame: DiscreteManifold = lattice.collapse(&shader);

// frame IS a manifold — use it as input to post-processing
let bloom = kernel!(|src: Manifold| /* bloom effect on src */);
let final_frame = lattice.collapse(&bloom(frame));

// Display
display.present(&final_frame.buffer);
```

### Multi-pass as monadic bind

```rust
// Pass 1: render scene
let scene = FrameLattice::new(1920, 1080, time).collapse(&scene_shader);

// Pass 2: blur (scene is a DiscreteManifold, re-enters algebra)
let blurred = FrameLattice::new(1920, 1080, time).collapse(&blur_shader(scene));

// Pass 3: composite
let final = FrameLattice::new(1920, 1080, time).collapse(&composite(blurred, ui_overlay));
```

Each `collapse` is a render pass. The result re-enters the algebra as a manifold. Chaining is monadic bind. The scheduler optimizes each pass independently.

### Reduction as collapse_with

```rust
// Dot product: reduce over Channel dimension
let channel_lattice = Lattice1D::new(3);  // 3 channels
let dot_product: Field = channel_lattice.collapse_with(
    ReduceOp::Add,
    &(normal * light),  // manifold over (X, Y, Channel)
);
// dot_product has variance {X, Y} — Channel consumed

// Average luminance: reduce over entire frame
let frame = FrameLattice::new(1920, 1080, time);
let avg: Field = frame.collapse_with(ReduceOp::Add, &luminance) / (1920.0 * 1080.0);
// avg has variance {} — all dimensions consumed
```
