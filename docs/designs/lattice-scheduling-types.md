# Lattice Scheduling via Types: Extraction as Factoring

## The Insight

E-graph extraction is expression factoring. When the extractor discovers that `sin(t * 0.3)` appears in all three color channels, it's discovering that the computation factors:

```
f(x, y, t) = g(x, y, h(t))     where h(t) = sin(t * 0.3)
```

CSE is factoring. Let-binding is factoring. Loop-invariant code motion is factoring. They're all the same operation: identifying a subexpression that depends on fewer variables than its context, and computing it at the appropriate scope.

The Halide scheduling problem — "when to materialize an intermediate" — is choosing WHERE in the factorization hierarchy to force evaluation. And in a pull-based system, this should be expressed in the types, not as a separate scheduling language.

## Everything Is Factoring (Apr 2, 2026)

Every "optimization" in a compiler is factoring — identifying independent subcomputations and assigning them to execution resources:

| Optimization | What it factors | Resource it assigns to |
|---|---|---|
| **CSE** | Shared subexpressions | Compute once, reuse via register/variable |
| **LICM** | Loop-invariant subexpressions | Outer scope (compute before loop) |
| **ILP** (instruction-level parallelism) | Independent instruction streams | Different execution units (superscalar) |
| **SIMD** vectorization | Identical ops on different data | Different data lanes |
| **Tiling** | Spatial locality groups | Cache hierarchy |
| **Register allocation** | Live values | Physical registers |

These are NOT separate optimizations. They are the same operation — **"identify what's independent, assign it to a resource"** — applied at different granularities.

**Variance analysis unifies them all.** The `depends_on` bitset tells you:
- **LICM**: `deps(e) ⊂ deps(context)` → hoist to outer scope
- **CSE**: same e-class, same deps → compute once
- **ILP**: `deps(a) ∩ deps(b) = {}` → no data dependency, can issue in parallel
- **SIMD**: `deps(e) ∩ {X} = {}` → same value across lanes, broadcast instead of compute
- **Tiling**: `deps(e) = {X, Y}` → benefits from 2D spatial locality

The variance lattice is the SINGLE analysis that drives ALL of these factoring decisions. Different backends (LLVM, JIT, interpreter) exploit the factoring at different levels, but the factoring itself is the same.

### The Solver Spectrum

Given the factoring problem (what to factor, where to place it), solvers vary in quality:

| Solver | Strategy | Speed |
|---|---|---|
| LLVM passes | Greedy, one expression form | Fast (single pass) |
| E-graph + greedy | Explores equivalences, picks first improvement | Medium |
| E-graph + NNUE | Learned approximation trained on optimal solutions | Fast (microseconds) |
| E-graph + Integer LP | Provably optimal factoring + scope + register assignment | Slow (for training data) |

The Integer Linear Programming formulation for optimal extraction:
```
minimize  Σ (evals_per_frame(scope(e)) * cost(node(e))) * x_{e}
subject to:
  one node per e-class                        (extraction)
  scope(e) ⊇ deps(e)                          (variance-legal factoring)
  Σ hoisted_to(Frame) ≤ register_limit        (register pressure)
  reduction bodies at deeper scope than result  (reduction ordering)
```

The NNUE is trained on ILP-optimal solutions (the Stockfish parallel: NNUE learns from stronger-but-slower exact search). At compile time, the NNUE approximates optimal factoring in microseconds.

## Variance as a Lattice

Every coordinate input to a manifold has a variance level:

```
Const ⊑ Uniform ⊑ Varying
```

- **Const**: Known at compile time. Folds to a literal.
- **Uniform**: Same for all pixels in a scope (frame, scanline, tile). Compute once per scope.
- **Varying**: Changes per pixel. Must be in the inner loop.

This forms a lattice (the mathematical kind). The join operation:
```
Const  ⊔ Const  = Const
Const  ⊔ Uniform = Uniform
Uniform ⊔ Varying = Varying
```

If a subexpression depends on a `Uniform` input and a `Varying` input, the subexpression is `Varying`. This propagates through composition automatically.

## Factoring Through the Variance Lattice

A manifold `f : (Varying, Varying, Uniform, Const) → Field` factors as:

```
f(x, y, t, _) = inner(x, y, setup(t))
```

Where `setup(t)` computes everything that depends only on `Uniform` inputs, and `inner(x, y, ...)` is the per-pixel kernel that receives the precomputed values.

The factorization is:
1. **Const** factor: evaluated at compile time (constant folding)
2. **Uniform** factor: evaluated once per scope (LICM)
3. **Varying** factor: evaluated per pixel (inner loop)

This is exactly what LLVM's LICM does — but expressed as a type-level property, not an optimization pass. The types DECLARE the factorization; the compiler ENFORCES it.

## Types as Schedules

```rust
// The manifold declares its variance signature
trait Manifold<P> {
    type Output;
    fn eval(&self, p: P) -> Self::Output;
}

// Variance-annotated coordinates
struct Varying(Field);   // changes per pixel
struct Uniform(Field);   // same per scope
struct Const(f32);       // compile-time known

// A shader with declared variance:
// X, Y vary per pixel; time is uniform per frame
impl Manifold<(Varying, Varying, Uniform, Const)> for MyShader {
    type Output = Field;
    fn eval(&self, (x, y, t, _): (Varying, Varying, Uniform, Const)) -> Field {
        // The type system knows:
        //   sin(t.0 * 0.3) is Uniform (depends only on Uniform input)
        //   x.0 * x.0 + y.0 * y.0 is Varying
        //   The compiler can factor accordingly
    }
}
```

## Composition Propagates Variance

```rust
// If f : (Varying, Uniform) → Field
// and g : Field → Field  (pure, no additional inputs)
// then f.map(g) : (Varying, Uniform) → Field

// If f : (Varying, Varying) → Field
// and g : (Uniform,) → Field
// then f.at(g) : (Varying, Varying, Uniform) → Field
// The contramap extends the domain with g's variance
```

The variance algebra follows the categorical structure. Contramap (`.at()`) extends the domain. Map preserves variance. Select takes the join of both branches.

## The Lattice as Materializer

A `Lattice<W, H>` is a demand for evaluation over a grid. When you evaluate a manifold on a lattice, the system:

1. Reads the manifold's variance signature
2. Factors out Const subexpressions (compile time)
3. Factors out Uniform subexpressions (compute once before the scan)
4. JIT-compiles only the Varying part as the inner loop
5. Scans the lattice, calling the inner loop per pixel with precomputed uniforms

This IS the schedule. The lattice dimensions define the scope. The variance types define what's invariant at each scope level. The factoring falls out of the types — no separate scheduling language needed.

## Multi-Level Lattices (Tiling)

```rust
// Frame-level lattice: 60fps, time varies per frame
// Scanline-level: Y varies per scanline, X varies per pixel within scanline

Lattice<Frame, 60> {
    time: Uniform,    // same for all pixels in this frame
    Lattice<Scanline, 1080> {
        y: Uniform,   // same for all pixels in this scanline
        Lattice<Pixel, 1920> {
            x: Varying, // changes per pixel
        }
    }
}
```

At each lattice level, the system factors out what's uniform at that level:
- Frame level: compute `sin(time * 0.3)` once per frame
- Scanline level: compute `y_factor = exp(y * 1.0)` once per scanline
- Pixel level: compute the remaining X-dependent part per pixel

This is tiling. The lattice hierarchy IS the tile structure. And it composes — you can nest lattices arbitrarily.

## Connection to Categories

In Conal Elliott's "compiling to categories" framework:

- A **manifold** is a morphism in a cartesian closed category
- **Variance** is a grading on the category — each morphism carries a variance annotation
- **Factoring** is the universal property of the product: `f : A × B → C` factors through `A → (B → C)` (currying), and the variance tells you which factor to curry out
- **The lattice** is a representable functor: `Lattice W H ≅ Hom(Fin W × Fin H, —)`
- **The schedule** is a choice of factorization, determined by the variance grading
- **The JIT** is a functor from the graded category to machine code

The backend (LLVM, JIT, interpreter) is just a choice of target category. The variance-graded factorization is the same regardless of backend. The types express the mathematical structure; the backend interprets it.

## What This Means for Pixelflow

1. **Variance becomes a type parameter** on manifolds and coordinates
2. **Composition propagates variance** via the join operation
3. **Lattice evaluation factors automatically** based on variance types
4. **The JIT sees the factorization** and emits code at the right scope
5. **No separate scheduling language** — the types ARE the schedule
6. **Pull-based is preserved** — nothing computes until demanded by a lattice
7. **The NNUE optimizer works within this framework** — it optimizes the Varying factor (the hot inner loop), while the type system handles the factoring

## Prior Art and Positioning (Apr 2, 2026)

### Elliott's Vertigo (2004): Frequency Analysis
Vertigo compiled functional shaders to GPU assembly with frequency analysis (constant/vertex/fragment). The GPU pipeline IS staged currying — constants applied first, then vertices, then fragments. But Elliott implemented this as a post-hoc AST pass, not type-level. No equivalence exploration, no cost model.

### Halide (2013): Schedule as Separate Language
Halide separates algorithm (pure function) from schedule (tiling, parallelism, fusion). Reinking & Bernstein (2020) proved schedule transformations preserve denotational semantics. But the schedule is a separate language with its own complexity — users must write it.

### PixelFlow's Position: Types ARE Schedules
We eliminate the schedule language entirely:
1. **Variance analysis in the e-graph** (exact, during saturation) determines WHAT factors
2. **The Lattice type** (compile-time, structural) determines WHERE each factor evaluates
3. **NNUE extraction** (learned, approximate) determines HOW each factor computes

This is stronger than both Elliott and Halide:
- Elliott had no equivalence exploration (stuck with one form)
- Halide requires manual schedule specification
- We discover all equivalent factorizations automatically (e-graph) and pick the cheapest (NNUE)

### The Monomorphization Advantage
Halide must schedule producer-consumer relationships because it materializes intermediates. PixelFlow's monomorphization fuses the entire pipeline into a single kernel — no intermediates to schedule. The only scheduling decision is which SCOPE to evaluate each subexpression at, and variance analysis determines that exactly.

## Open Questions

- How does this interact with the `kernel!` macro? The macro currently sees a flat expression. It would need to see the variance annotations. One approach: the macro emits the expression with variance holes, the e-graph fills them in during saturation, the extractor emits scoped code.
- How do we express "compute_at" boundaries? Nested lattices? Or explicit `materialize` combinators? Current thinking: nested `Lattice` types compose naturally. `Lattice<Frame, 60> { Lattice<Scanline, 1080> { Lattice<Pixel, 1920> } }` — each level binds variables and defines a scope.
- What's the zero-cost abstraction story? Variance should be ZST at runtime. The BitSet<4> is a compile-time annotation; at runtime, code is already emitted at the correct scope.
- How does autodiff (Jets) interact with variance? If a coordinate is Uniform, its derivative is zero — the Jet for that coordinate collapses. This could eliminate entire AD computation paths statically. A Uniform-variance subexpression has zero spatial gradient — no need to compute Jet2 for it.
- Can variance-aware rewrite rules discover factorizations the programmer missed? e.g., `sin(Z) * X + sin(Z) * Y` → `sin(Z) * (X + Y)` reduces the variance of the `sin(Z)` factor from `{X,Y,Z}` (in the original) to `{Z}` (factored out). The e-graph already has distributivity rules — variance analysis would tag the factored form as "cheaper scope" automatically.
