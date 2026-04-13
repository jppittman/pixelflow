# Reductions, Folds, and Dimension Collapse

## The Problem (Apr 2, 2026)

Everything in PixelFlow is a Manifold — a function from coordinates to values. Composition, warping, selection all PRESERVE or GROW the variance set (which coordinates an expression depends on). But some operations CONSUME a dimension:

- **Dot product**: `Σ_i a(i) * b(i)` — eliminates the index dimension
- **Blur/convolution**: `∫ f(x') * G(x-x', σ) dx'` — integrates over a spatial offset
- **Reduction**: `fold(+, 0, f)` over a domain — collapses one axis to a scalar
- **Matrix multiply**: double reduction over shared index

These are all the same operation: **reduction over a Lattice dimension**.

## Vectors as Manifolds

A vector isn't a special type — it's a manifold with an index coordinate:

```
a : (Index, X, Y, Z, W) → Field     // 3-component vector at each pixel
b : (Index) → Field                   // constant 3-vector (e.g., light direction)
```

The index coordinate is just another variable. It has variance like any other:
- `a(i, x, y, z, w)` — depends on `{Index, X, Y, Z, W}`
- `b(i)` — depends on `{Index}`
- `a * b` (pointwise) — depends on `{Index, X, Y, Z, W}`
- `Σ_Index (a * b)` — depends on `{X, Y, Z, W}` — Index is GONE

## The Variance Rule for Reduction

```
depends_on(Reduce(op, dim, body)) = depends_on(body) \ {dim}
```

This is the ONLY operation that shrinks the variance set:
- `Var(i)` → `{i}`
- `Const` → `{}`
- `BinaryOp(a, b)` → `deps(a) ∪ deps(b)` — grows or preserves
- `UnaryOp(a)` → `deps(a)` — preserves
- `Reduce(op, dim, body)` → `deps(body) \ {dim}` — SHRINKS

## Strictness, Not Eagerness

Pull-based says "nothing computes until coordinates arrive." A reduction is **strict in the reduced dimension** — all values in the domain must be produced before the fold yields a result. But it is still **demand-driven** along the surviving dimensions. Nothing computes until an (X, Y) coordinate arrives.

"Eager" is the wrong word. An eager system would precompute the reduction even if nobody asks. PixelFlow does not do that. The reduction waits for demand on the surviving axes, then strictly evaluates all values along the reduced axis to produce the result.

For small reductions (N=3 dot product), strictness is academic — the compiler unrolls the reduction into a single expression `a0*b0 + a1*b1 + a2*b2`. For large N (matrix multiply, wide convolution kernel), strictness has real scheduling implications: the accumulator must process all N values before yielding.

### Semantics vs Implementation

These are distinct levels that the design must not conflate:

1. **Semantics**: `Reduce(+, Channel, f)` denotes `λ(x, y). Σ_{c ∈ Channel} f(x, y, c)`. The variance rule `deps(body) \ {dim}` is a fact about this mathematical function. Implementation-independent.

2. **Implementation**: The reduced dimension requires strict evaluation. The surviving dimensions are demand-driven. The Lattice type determines the materialization schedule. Multiple equivalent implementations may exist (unrolled, loop, tree reduction).

3. **Optimization**: The e-graph explores equivalent denotations. The NNUE picks the cheapest implementation of each denotation.

### The Lattice as Materializer

```
Lattice<Y, 1080> {              // demand-driven: one scanline at a time
    Lattice<X, 1920> {          // demand-driven: one pixel at a time
        Lattice<Channel, 3>.reduce(+, normal * light)
        // strict: all 3 channels must be evaluated for this (x,y)
        // but only computed because (x,y) was demanded
    }
}
```

The reduced Lattice materializes values along one axis and collapses them. The surviving axes remain demand-driven.

## Concrete Example: Per-Pixel Dot Product

```rust
// Surface normal: 3-component vector at each pixel
// normal : (X, Y, Channel) → Field
//   normal(x, y, 0) = nx(x, y)    // X component
//   normal(x, y, 1) = ny(x, y)    // Y component
//   normal(x, y, 2) = nz(x, y)    // Z component

// Light direction: constant 3-vector
// light : (Channel) → Field
//   light(0) = lx, light(1) = ly, light(2) = lz

// Dot product:
// dot = Reduce(+, Channel, normal * light)
// Variance: {X, Y, Channel} \ {Channel} = {X, Y}
// Result: a scalar field varying per-pixel
```

Variance analysis at each stage:
| Expression | depends_on |
|---|---|
| `normal(x, y, c)` | `{X, Y, Channel}` |
| `light(c)` | `{Channel}` |
| `normal * light` | `{X, Y, Channel}` |
| `Reduce(+, Channel, ...)` | `{X, Y}` |

The Channel dimension is consumed. The result is a 2D scalar field.

## E-Graph Interaction

`Reduce` is a new e-node type. The e-graph can discover optimizations:

### Hoisting invariants out of reductions
```
Σ_i (f(i) * c) = c * Σ_i f(i)     // when deps(c) ∩ {i} = {}
```
The e-graph already has distributivity rules. Variance analysis PROVES `c` can be hoisted — its `depends_on` doesn't include the reduction variable. This is "free LICM through reductions."

### Reduction fusion
```
Σ_i a(i) + Σ_i b(i) = Σ_i (a(i) + b(i))    // fuse two reductions into one pass
```
One loop instead of two. The e-graph can discover this via linearity of summation.

### Reduction elimination
```
Σ_i c = N * c       // when deps(c) ∩ {i} = {}, reducing a constant is multiplication
```
Variance analysis proves `c` doesn't depend on `i`, so the sum of N copies is `N * c`.

### Symbolic integration (continuous reductions)
```
∫ (ax + b) * G(x, σ) dx = a * μ + b    // Gaussian integral of linear function
blur(linear_gradient) = linear_gradient   // blur of a line is itself
```
When the e-graph has calculus rules AND knows the integrand's form, it can find closed forms. The discrete reduction becomes unnecessary.

## Cost Model

For the NNUE and ILP:
```
cost(Reduce(+, dim, body, N)) = N * cost(body) + (N-1) * cost(+)
```

Reductions are **synchronization points** in the ILP — the result isn't available until all N evaluations complete. This constrains scheduling: nothing that depends on the reduction result can be parallelized with the reduction body.

For scope assignment: a `Reduce` over dimension `i` at scope `S` means the body must be evaluated at scope `S ∪ {i}` (one level deeper), and the result lives at scope `S`. The reduction is the operation that CROSSES a scope boundary.

## Connection to Continuous Calculus

Discrete reduction over `Lattice<N>` is a Riemann sum. The continuous version is integration:

```
Discrete: Σ_i f(i)           (reduction over Lattice<N>)
Continuous: ∫ f(x) dx        (integration over a domain)
```

Blur is integration: `blur(f, σ) = ∫ f(x') * G(x-x', σ) dx'`

The e-graph with symbolic calculus rules can handle both:
1. **Closed-form**: if the integrand has a known antiderivative, evaluate symbolically (no loop)
2. **Quadrature**: if no closed form, materialize via `Lattice<KernelTap, N>.reduce(...)` (discrete approximation)

The symbolic path is preferred when it exists — it's exact and O(1). But for general expressions (e.g., `blur(sin(X*X + Y*Y), sigma)`), no closed form exists and the quadrature path is required. The e-graph explores BOTH when symbolic rules apply, and the NNUE picks the cheaper one.

**Jets collapse convolution to O(1).** We don't need to re-evaluate at shifted coordinates because the Jet gives us the local Taylor expansion:

```
f(x + dx) ≈ f(x) + f'(x) * dx + f''(x) * dx²/2 + ...
```

Substituting into the Gaussian convolution integral:
```
∫ f(x + t) * G(t, σ) dt
≈ ∫ (f(x) + f'(x)*t + f''(x)*t²/2) * G(t, σ) dt
= f(x) · 1  +  f'(x) · 0  +  f''(x) · σ²/2
= f(x) + D²(f)(x) · σ²/2
```

**Blur = value + curvature correction.** The Jet2H system already computes `f''` for antialiasing. Gaussian blur is a function of derivatives we already have — no shifted evaluations needed.

E-graph rewrite rule:
```
blur(f, σ) → f + D²(f) * σ²/2              // second-order (Jet2H)
blur(f, σ) → f + D²(f) * σ²/2 + D⁴(f) * σ⁴/8   // fourth-order (higher Jet)
```

For small σ (most visual effects), second-order is visually exact. For large σ, higher-order Jets extend the range. The cost is O(1) per pixel regardless of kernel size — the "kernel" is implicit in the derivative order.

This eliminates the O(kernel_size) multi-point evaluation entirely for smooth functions. The only case requiring actual multi-point evaluation is non-smooth functions (hard edges, discontinuities) where the Taylor expansion diverges — and there, the Jet's derivative magnitude flags the discontinuity, allowing fallback to a narrow discrete kernel only where needed.

## Matrix Operations

Matrix multiply is a double reduction over a shared index:

```
C(i, j) = Σ_k A(i, k) * B(k, j)
```

In Manifold terms:
- `A : (I, K) → Field`
- `B : (K, J) → Field`
- `A * B` (pointwise, broadcast): `(I, K, J) → Field` — deps `{I, K, J}`
- `Reduce(+, K, A * B)`: `(I, J) → Field` — deps `{I, J}`

The K dimension is consumed. The I and J dimensions survive. This is a manifold from (row, column) to value — a matrix.

Variance analysis reveals: if `A` only depends on `{I, K}` and `B` only depends on `{K, J}`, the inner product over K produces a result depending on `{I, J}`. The K dimension is the "contraction" in tensor notation.

## Connection to Tensor Contraction

Einstein summation notation: `C_ij = A_ik B_kj` (implied sum over repeated index k).

In our framework:
- Repeated index = dimension that appears in BOTH operands' variance
- Contraction = `Reduce` over that dimension
- Free indices = dimensions that survive in the result's variance

The variance bitset tells you which indices are free and which are contracted:
```
free_indices = (deps(A) ∪ deps(B)) \ contracted_dims
```

This is tensor algebra expressed through variance analysis.

## The Type Signature

```rust
/// A reduction consumes a Lattice dimension, collapsing it via a binary operator.
///
/// The body manifold depends on the reduced coordinate.
/// The result manifold has one fewer coordinate — the reduced dimension is gone.
///
/// Variance rule: deps(Reduce) = deps(body) \ {reduced_dim}
///
/// Pull-based along surviving dimensions. Eager along reduced dimension.
fn reduce<const N: usize, F, Op>(
    body: F,      // Manifold that depends on the reduced coordinate
    op: Op,       // Associative binary operator: Add, Mul, Min, Max
    init: Field,  // Identity element: 0 for Add, 1 for Mul, +∞ for Min, -∞ for Max
) -> impl Manifold<...>
where
    F: Manifold<(..., Varying, ...)>,   // the Varying slot is being reduced
    Op: Fn(Field, Field) -> Field,       // must be associative for parallel reduction
```

The `Lattice<N>` that materializes the reduced dimension is implicit — it's determined by the context. When you `reduce` over Channel with `Lattice<Channel, 3>`, N=3 evaluations happen.

## Categorical Foundation: `jam` as Dual of `dup` (Apr 2, 2026)

Elliott's categorical AD framework ("The Simple Essence of Automatic Differentiation", ICFP 2018) provides the algebraic foundation for reductions:

### The Biproduct Category
- **Cartesian** (products): `dup : a → (a, a)` (fan-out), `exl`, `exr` (projections)
- **Cocartesian** (coproducts): `jam : (a, a) → a` (summation), `inl`, `inr` (injections)
- In AD's biproduct category, products and coproducts coincide. `jam` IS addition.

The indexed generalization:
```haskell
jamPF :: h a → a      -- categorical reduction: sum over functor h
replF :: a → h a      -- categorical broadcast: replicate into functor h
```

**`jamPF` is the categorical reduction.** The functor `h` is the dimension being consumed. `replF` is its dual — broadcasting a value into a dimension. These are the same operations as our Reduce and our SIMD broadcast.

### Forward vs Reverse AD as Factoring Duality

| Forward-mode AD (Jets) | Reverse-mode AD (Backprop) |
|---|---|
| `dup` (fan-out: use value in multiple places) | `jam` (sum: add gradient contributions) |
| Pushes derivatives forward | Pulls gradients backward by summing |
| Variance-PRESERVING | Variance-CONSUMING (reduction) |
| `D f : a → (a, a ⊸ b)` | `D† f : (b, b ⊸ a)` (dual direction) |

**The dual category swaps `dup` and `jam`.** Every variable use (fan-out) in the forward pass becomes a gradient summation (reduction) in the backward pass. This is why backprop sums gradients from multiple uses of a variable.

For PixelFlow:
- Forward-mode (Jets): `deps(Jet(f)) = deps(f)` — variance preserved
- Reverse-mode (if we ever need it): the backward pass introduces reductions (sums over all uses)
- The duality between "grow variance" (dup/broadcast) and "shrink variance" (jam/reduce) is categorically fundamental

### Convolution as Monoidal Reduction

Elliott's "Generalized Convolution" (2019) formalizes convolution as summation over monoid factorizations:
```
(f * g)(m) = Σ over (p, q) where p <> q = m of f(p) · g(q)
```

Image convolution: domain monoid is `(Z², +)`. The summation over factorizations IS a reduction over the kernel footprint. In our framework, this is:
```
conv(f, kernel)(x, y) = Reduce(+, Offset, f(X + offset_x, Y + offset_y) * kernel(offset_x, offset_y))
```
Where Offset is the dimension consumed by the reduction.

### What Elliott's Framework Already Encodes

The categorical composition `jam . f` gives `(X, Y) → Field` from `(X, Y, Channel) → Field`. The types encode the data dependency: Channel must be fully consumed before the result exists. This IS scheduling information, via the universal property of the coproduct. **Elliott's framework is denotationally adequate for reductions.**

### What We Add Beyond Elliott

1. **Equivalence exploration**: Elliott has one representation per program. Our e-graph discovers ALL equivalent representations — including factored forms that reduce variance. This is the key novelty.
2. **Cost model**: Elliott has no notion of "this costs cycles." Our NNUE estimates cost, enabling the system to choose among equivalent denotations.
3. **Concrete scope mapping**: Elliott's categorical structure encodes dependencies abstractly. We map them concretely to evaluation scopes (Frame, Scanline, Pixel) in a multi-level loop nest, driven by variance analysis as an e-class annotation.
4. **Convolution ↔ AD interaction**: His 2018 AD paper and 2019 convolution paper are independent. He never unified them — showing how differentiation of a convolution becomes a convolution of derivatives (which it does, and variance analysis could prove it).

### References
- Elliott, "The Simple Essence of Automatic Differentiation" (ICFP 2018)
- Elliott, "Generalized Convolution and Efficient Language Recognition" (2019)
- Fong & Spivak, "Backprop as Functor" (2019) — training loop as monoidal functor
- Elliott, "Compiling to Categories" (ICFP 2017)

## Open Questions

1. **Associativity matters**: parallel reduction requires an associative operator. `+` and `*` are associative; `-` is not (but can be expressed as `+ negate`). How do we enforce this in the type system?

2. **Partial reductions**: what about reducing over a SUBSET of a dimension? e.g., a sliding window sum (moving average). This is a reduction where the domain shifts per output point. It's still a `Reduce`, but the Lattice bounds depend on the surviving coordinates. This connects to Halide's sliding window optimization.

3. **Scan (prefix sum)**: `scan(+, [a, b, c, d]) = [a, a+b, a+b+c, a+b+c+d]`. This is NOT a reduction — it preserves the dimension. But it's related. Does it fit our framework? A scan is a reduction that emits intermediate results. The dimension survives but each point depends on all previous points. Variance: still `{Index, ...}` but with a CAUSAL dependency within Index.

4. **Commutativity and SIMD**: if the operator is commutative AND associative, the reduction can be done in any order — including SIMD tree reduction (`FADDP`). The e-graph could have rules that exploit commutativity for SIMD-friendly reduction order.

5. **When does reduction interact with the e-graph's equivalence classes?** If the e-graph discovers that `Σ_i f(i) = closed_form_g`, the reduction e-class contains both the loop and the closed form. The NNUE picks the cheaper one. This is the symbolic calculus payoff — the e-graph explores whether a reduction has a closed form.

6. **Multiple reductions**: `Σ_i Σ_j f(i, j)` — nested reductions. Order matters for performance (which loop is inner?). The e-graph could have commutativity rules for independent reductions: `Σ_i Σ_j = Σ_j Σ_i` when the reductions are over independent dimensions. The NNUE/ILP picks the cheaper order.
