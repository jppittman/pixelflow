# Brainstorm: Variance Analysis via E-Graph Saturation

## Context

Pixelflow is a pull-based functional graphics engine built on manifold algebra. We have:
- E-graph equality saturation for expression optimization
- NNUE neural cost model for extraction (picking cheapest equivalent form)
- Lattice type as a materializer (discrete grid that demands evaluation)
- The insight that CSE/LICM/scheduling are all FACTORING

## The Question

Can we do variance analysis (which coordinates an expression depends on) as an e-graph analysis — fully saturated, exact, no neural approximation?

The idea: alongside the equivalence classes (which expressions compute the same VALUE), track which VARIABLES each e-class depends on. This is a monotone dataflow analysis that the e-graph's rebuild mechanism already supports.

## Concrete Example

```
let time = Z + 1.3;
let sin_w03 = sin(time * 0.3);   // depends on: {Z}
let r_sq = X*X + Y*Y;            // depends on: {X, Y}
let radial = abs(r_sq - 0.7);    // depends on: {X, Y}
let result = sin_w03 * radial;   // depends on: {X, Y, Z}
```

Each e-class gets a `depends_on: BitSet<{X, Y, Z, W}>` annotation. The analysis is:
- Var(i) → {i}
- Const → {}
- BinaryOp(a, b) → depends(a) ∪ depends(b)
- UnaryOp(a) → depends(a)

This is a semilattice analysis — monotone, finitely bounded, converges in one pass for acyclic expressions. For e-graphs with cycles (from rewrite rules that create equivalences), it converges in O(depth) rebuild passes.

## The Factoring Connection

Once we know `depends_on` for each e-class:
- E-classes with `depends_on = {}` → Const factor (compile-time)
- E-classes with `depends_on = {Z}` → Uniform factor (per-frame)
- E-classes with `depends_on = {Y, Z}` → Row factor (per-scanline)
- E-classes with `depends_on = {X, Y, Z}` → Pixel factor (inner loop)

The factorization falls out of the analysis. No scheduling heuristics needed.

## The Ick Factor

Why NOT use the NNUE for this?
- Variance is a FACT, not a prediction. `sin(Z)` depends on Z. Period. No neural net needed.
- Saturation should be EXACT for variance. Every equivalence the e-graph discovers should propagate variance precisely.
- The NNUE's job is COST estimation (which equivalent form is cheapest), not VARIANCE (which variables matter). These are orthogonal.

But: should the e-graph's rewrite rules be variance-AWARE? e.g., should the saturation head prefer rules that REDUCE variance (factor things out)? That feels more like the NNUE's territory — it's a search guidance question, not a correctness question.

## Prior Art: What Elliott Actually Did

Research into Conal Elliott's rendering systems (Apr 2, 2026) reveals:

### Pan (2003): Functional Images
- Images as `Point -> Color`, compiled to C with an explicit pixel loop
- **LICM as an explicit AST pass** — walks expression tree, finds pixel-independent subexprs, hoists them
- CSE also as an explicit pass. Elliott calls sharing recovery "awkward and expensive"
- The categorical structure makes dependency *representable* (via `fst`/`snd` projections) but does NOT exploit it for scheduling

### Vertigo (2004): GPU Shaders — The Closest Analog
- Compiles functional shader descriptions to GPU vertex/pixel shader assembly
- **Frequency analysis**: each expression node tagged with frequency (constant / vertex / fragment)
- This IS variance analysis — but as a post-hoc pass on the expression tree, not type-level
- Key insight: **the GPU hardware IS a staged evaluation of a curried function**
  - Constants "applied" first (loaded into uniform registers)
  - Vertex data "applied" next (vertex shader)
  - Fragment interpolants "applied" last (pixel shader)
  - Currying = staging = hoisting

### What Elliott Never Had
- No equivalence exploration (no e-graph — stuck with one expression form)
- No cost model (no NNUE equivalent — structural optimization only)
- No systematic continuous→discrete bridge (Pan's pixel loop is ad-hoc)
- Never made variance type-level — always a post-hoc analysis pass
- Never addressed tiling, SIMD mapping, or fusion decisions

### What We Add Beyond Elliott
1. **E-graph discovers ALL equivalent factorizations** (not just the one the programmer wrote)
2. **Variance as an e-class annotation** (runs during saturation, not after extraction)
3. **NNUE picks cheapest form within each variance class** (cost-aware, not just structural)
4. **Lattice type = schedule** (the type system determines evaluation scope, not a separate language)
5. **Monomorphization fuses everything** (no Halide-style producer-consumer scheduling needed — never materialize intermediates)

## The Two-Lattice Architecture

There are two lattices in play:

### Lattice 1: Variance (exact, in the e-graph)
```
depends_on: PowerSet({X, Y, Z, W})
join = set union
```
This is the **powerset lattice** over coordinate variables. It's a monotone semilattice analysis that converges in O(depth) rebuild passes. Every e-class gets a `depends_on: BitSet<4>` annotation.

### Lattice 2: Evaluation Scope (determined by the Lattice type)
```
Const → Frame → Scanline → Pixel
  {}     {Z}    {Y,Z}    {X,Y,Z}
```
This is a **total order** determined by the rasterizer's Lattice shape. Different Lattice shapes give different scope assignments:
- `Lattice<1920, 1080>` → scanline order (Y outer, X inner)
- `Lattice<32, 32>` tiled → tile blocks
- Nested `Lattice<Frame, N> { Lattice<Scanline, H> { Lattice<Pixel, W> } }` → multi-level

The mapping from Lattice 1 → Lattice 2 is: given a `depends_on` bitset, find the shallowest scope that binds all those variables. This mapping is determined by the Lattice type at compile time. There's always a unique "cheapest scope" because Lattice 2 is a total order.

### The Clean Separation
- **E-graph saturation + variance analysis** = discovers ALL factorizations (exact, monotone)
- **Lattice type** = maps variance classes to evaluation scopes (structural, type-level)
- **NNUE extraction** = picks cheapest equivalent form WITHIN each scope (learned, approximate)

The neural net never decides WHAT to factor — the types do that. The neural net decides HOW to compute each factor.

## Critical Finding: Meet vs Join (Apr 2, 2026)

**BUG**: The current `DepsAnalysis` (in `pixelflow-search/src/egraph/deps.rs`) computes the JOIN (union) of dependencies across all e-nodes in an e-class. This is WRONG for variance analysis.

**Example**: After the e-graph discovers `X - X = 0`, the e-class contains both `Sub(X, X)` (deps `{X}`) and `Const(0)` (deps `{}`). The join gives `{X}`. But the correct answer is `{}` — the e-class CAN be computed with zero dependencies by choosing the `0` representative.

**Fix**: Variance should use MEET (minimum) across e-class representatives:
```
class_deps(c) = min over n in nodes(c) of node_deps(n)
where node_deps(n) = join over children of n
```

For each e-class: compute deps for EACH e-node independently (joining children deps), then take the MIN across all e-nodes in the class. The variance reflects the BEST available implementation, not the worst.

This is a soundness issue — pessimistic variance prevents hoisting that should be legal.

## Select Decomposition (Apr 2, 2026)

`select(X > 0, sin(Z), cos(Z))`:
- Condition: `{X}`, true branch: `{Z}`, false branch: `{Z}`
- Naive union: `{X, Z}` — tags whole expression as Pixel-varying

But both branches are `{Z}` — they can be precomputed at Frame scope! The inner loop becomes:
```asm
// Frame setup:
sin_z = sin(Z * 0.3)     // computed once
cos_z = cos(Z * 0.3)     // computed once
// Pixel loop:
mask = X > 0              // per-pixel
result = BSL mask, sin_z, cos_z  // bit select on precomputed values
```

**Need**: Select-specific variance decomposition. Record that select operands can be independently hoisted. The annotation must be richer than a single bitset per e-class — at minimum, track `(condition_deps, branch_deps)` for select nodes.

## Variance + Jets = Free Derivative Elimination (Apr 2, 2026)

If `depends_on = {Z}` for a subexpression, then `dF/dX = 0` and `dF/dY = 0` EXACTLY. This is not an approximation — it's a mathematical fact.

The Jet2 system computes `(value, dF/dX, dF/dY)` for EVERY subexpression. For Uniform subexpressions, the derivative components are provably zero. Variance analysis could eliminate them at compile time, nearly halving Jet computation for shaders with substantial uniform setup.

**Connection**: Variance tells you the DIMENSION of dependence. The Jet tells you the MAGNITUDE of change. Together:
- `depends_on = {}` → value is constant, all derivatives zero
- `depends_on = {Z}` → spatial derivatives zero, temporal derivative may be nonzero
- `depends_on = {X}` with small `|dF/dX|` → slowly varying, candidate for interpolation/mipmap

## Loop-Invariant Hoisting in the JIT (Apr 2, 2026)

~~SIMD Lane Broadcasting~~ — RETRACTED. Y/Z/W are already pre-broadcast as `float32x4_t` via `vdupq_n_f32()`. Computing `Y*Y` on a broadcast vector uses one NEON FMUL cycle — same throughput as scalar. No wasted work per lane.

The real optimization: expressions with `deps ∩ {X} = {}` are **loop-invariant within the X loop**. Currently `sin(Y)` is recomputed on every X iteration even though Y doesn't change. The fix is hoisting to a per-scanline setup phase — see "Missing Scope Level: Scanline" below.

## Missing Scope Level: Scanline (Apr 2, 2026)

`exp(-Y * Y * 0.5)` has variance `{Y}` — should compute once per scanline, before the X loop. But the current scanline JIT (`emit_scanline_prologue` in `aarch64.rs`) has no "per-scanline setup" phase. It just shuffles Y/Z/W into callee-saved registers and enters the X loop.

Need: a setup block between Y assignment and X iteration that computes all `{Y}`-dependent (but not `{X}`-dependent) subexpressions and stores them in callee-saved NEON registers (v8-v15 per AAPCS64, up to 8 hoisted values).

The scope lattice should be: `Const < Frame({Z},{W}) < Scanline({Y}) < Pixel({X})`

## Variance as NNUE Feature (Apr 2, 2026)

4-bit variance field per node in the NNUE feature vector. The extraction head would naturally learn that variance-reducing rewrites are valuable:
- `sin(Z)*X + sin(Z)*Y` → `sin(Z)*(X+Y)` moves `sin(Z)` from `{X,Y,Z}` context to `{Z}` (2M evals → 1 eval)
- The NNUE learns this is a 2-million-fold cost reduction without special rules

## Engineering: Duplicate Deps Types (Apr 2, 2026)

Two `Deps` enums exist:
- `pixelflow-compiler/src/codegen/leveled.rs` (macro compiler path)
- `pixelflow-search/src/egraph/deps.rs` (e-graph path)

Should unify in `pixelflow-ir` with `BitSet<4>` replacing the three-level enum.

## Symbolic Calculus in the E-Graph (Apr 2, 2026)

**Key insight from JP**: Spatial reads (stencils, blur, convolution) are a category error. They are discrete finite-difference approximations of calculus operations. In PixelFlow, calculus should be done SYMBOLICALLY in the e-graph via rewrite rules.

### Differentiation Rules
```
d/dx(sin(f))  → cos(f) * d/dx(f)          // chain rule
d/dx(f * g)   → f * d/dx(g) + g * d/dx(f) // product rule
d/dx(f + g)   → d/dx(f) + d/dx(g)         // linearity
d/dx(X)       → 1                           // identity
d/dx(Y)       → 0                           // ← variance tells you this!
d/dx(const)   → 0
```

### Integration / Convolution
```
blur(f, σ) ≈ f + D²(f) * σ²/2             // Taylor expansion of Gaussian blur
glow(f, r) = ∫∫ f(x',y') * kernel(x-x', y-y') dx'dy'  // symbolic convolution
```

### The Pipeline
User writes blur/glow/shadow as mathematical operations → e-graph has calculus rules → symbolic simplification discovers closed forms → variance analysis hoists what it can → NNUE picks cheapest remaining form. **No discrete stencils ever.**

### Variance + Calculus Interaction
- `D_x(f)` has same `depends_on` as `f` (differentiation doesn't add variable dependencies)
- `D_x(f)` where `depends_on(f) = {Y,Z}` → `D_x(f) = 0` — e-graph rewrites to zero immediately
- This collapses entire branches of the chain rule at saturation time
- Jets already compute `(value, dF/dX, dF/dY)` — the e-graph's symbolic calculus and the Jet's AD are DUAL views of the same thing. The e-graph can verify Jet results symbolically, or the Jet can provide numerical validation of symbolic rules.

## ILP for Joint Extraction + Scheduling (Apr 2, 2026)

### The Connection
Three places where Integer Linear Programming fits:

**1. Optimal E-Graph Extraction (known)**
Extracting minimum-cost tree from e-graph is NP-hard. ILP formulation:
```
minimize  Σ cost(n) * x_n
s.t.      Σ x_n for n in class(c) = 1          ∀ e-classes c
          x_n = 1 ∧ child(n, c') → Σ x_m for m in c' = 1
          x_n ∈ {0, 1}
```
Our NNUE hill-climber approximates this. ILP gives OPTIMAL for small-to-medium expressions.

**2. Scope Assignment with Register Pressure**
Given variance annotations, hoisting has tradeoffs (register pressure):
```
minimize  Σ scope_cost(e, s) * x_{e,s}
s.t.      Σ x_{e,s} for s in scopes = 1        ∀ e-classes e   (one scope)
          x_{e,s} = 0 if deps(e) ⊄ vars(s)     (can't hoist past deps)
          Σ x_{e, Frame} ≤ 8                    (AAPCS64 callee-saved v8-v15)
          Σ x_{e, Scanline} ≤ remaining_regs
```

**3. JOINT Extraction + Scope Assignment (the big one)**
Extraction choice and scope assignment are NOT independent. Choosing `sin(Z)*(X+Y)` over `sin(Z)*X + sin(Z)*Y` changes BOTH cost AND variance profile.
```
minimize  Σ (pixel_count / scope_size(s)) * cost(n) * x_{n,s}
s.t.      extraction constraints (pick one node per class)
          scope constraints (respect variance)
          register constraints (limited hoisted slots)
```

### The Stockfish Training Parallel
Stockfish's NNUE was trained on positions evaluated by the stronger-but-slower classical search (Syzygy tablebases + deep alpha-beta). Similarly:
- **ILP gives ground-truth optimal extractions** on the training set (offline, can be slow)
- **NNUE learns to approximate ILP solutions** (online, must be fast)
- The NNUE is trained on ILP-optimal (extraction, scope) pairs
- At runtime, NNUE hill-climbing approximates ILP in microseconds

This closes the loop: ILP provides the "teacher" signal, NNUE provides the "student" speed.

## What Variance Cannot Do (Apr 2, 2026)

Honest boundaries of the approach:

1. **Convolution/blur**: Jets collapse Gaussian blur to O(1): `blur(f, σ) ≈ f + D²(f) * σ²/2`. The Jet2H system already computes second derivatives for antialiasing — blur is a function of derivatives we already have. No shifted evaluations, no stencils, no re-evaluation. Falls back to narrow discrete kernel only at discontinuities (where Jet derivative magnitude flags the divergence). See REDUCTIONS_AND_FOLDS.md.
2. **Frequency content**: `sin(X*0.01)` and `fract(X*1000)` both have variance `{X}`. One is smooth, the other aliased. Variance is binary — need Jet magnitude for adaptive precision.
3. **Data-dependent branching**: Beyond select decomposition, general control flow (loops, recursion) may have variance that depends on runtime values. Our system is expression-level, which sidesteps this.

## Abstract Interpretation Foundation (Apr 2, 2026)

The mapping from variance bitsets to evaluation scopes is a **Galois connection**:

- `alpha: PowerSet({X,Y,Z,W}) -> Scope` maps a bitset to its shallowest scope
- `gamma: Scope -> PowerSet({X,Y,Z,W})` maps a scope to its maximum bitset
- `alpha(S) <= scope` iff `S <= gamma(scope)`

Variance analysis IS abstract interpretation over the powerset domain. The Lattice type defines the concrete domain (evaluation scopes). The Galois connection is determined at compile time by the Lattice shape.

## Implementation Priority (Apr 2, 2026)

### Phase 1: Foundation (immediate)
1. ~~Fix `DepsAnalysis` meet-vs-join bug (soundness)~~ — DONE (Apr 2)
2. Unify two `Deps` types into `pixelflow-ir` with `BitSet<4>`
3. Add per-scanline setup block to JIT emitter

### Phase 2: Integration (near-term)
4. Select-specific variance decomposition
5. Wire variance into NNUE features (4-bit per node)
6. Connect variance to Jet elimination
7. Hoisted uniform register ABI (v8-v15)

### Phase 3: Symbolic Calculus (medium-term)
8. Differentiation rewrite rules in e-graph (chain rule, product rule, etc.)
9. Variance-guided derivative elimination (D_x where deps ∩ {X} = {} → 0)
10. Symbolic convolution / blur as closed-form e-graph rewrites

### Phase 4: Optimal Extraction (medium-term)
11. ILP formulation for e-graph extraction (exact solver for training data)
12. Joint extraction + scope assignment ILP
13. Train NNUE on ILP-optimal solutions (Stockfish parallel)
14. Register-pressure-aware scope assignment

### Phase 5: Type-Level Scheduling (long-term)
15. Lattice type hierarchy with variance-driven scope assignment
16. Nested Lattice types for tiling
17. Type-level Galois connection (compile-time scope inference)

## Questions to Explore

1. Can e-graph analyses track variance alongside value equivalence? egg supports "e-class analyses" — is our e-graph framework extensible the same way?

2. Does variance analysis interact with rewrite rules? If rule A→B changes which variables appear in the expression, does the variance annotation change? (Answer: the annotation is on the e-CLASS, not the e-node. When A and B are merged, their variance annotations are MEET'd. If a rewrite discovers a lower-variance form, that propagates to the whole class via meet.)

3. How does this connect to Conal Elliott's "compiling to categories"? Variance is a grading on the category of manifolds. Elliott's Vertigo proved frequency analysis works but never formalized it categorically. We could.

4. Can we express the lattice hierarchy (Frame > Scanline > Pixel) as a tower of adjunctions? The "compute_at" decision is choosing where in the tower to force evaluation. (Answer: YES — the alpha/gamma Galois connection IS the adjunction. See Abstract Interpretation Foundation above.)

5. What's the relationship between variance and automatic differentiation? (Answer: variance determines which Jet components are provably zero. See Variance + Jets section above.)

6. Is there a connection to abstract interpretation? (Answer: YES — variance analysis is abstract interpretation over the powerset domain. See Abstract Interpretation Foundation above.)

7. How does this relate to Halide's bound inference? Halide computes which regions of intermediate functions are needed. Variance analysis computes which dimensions are needed. Are these dual?

8. The "higher dimension" intuition: what if variance isn't tracked per-variable but per-LATTICE-LEVEL? Each level of the lattice hierarchy induces a variance class. (Answer: this is exactly the alpha mapping from the Galois connection. The bitset is the fine-grained analysis; the scope level is the coarsened version.)

9. **Variance-aware rewrite rules**: Should the saturation engine prefer rewrites that REDUCE variance (split an {X,Y,Z} expression into {Z} * {X,Y})? This is factoring as a rewrite. The e-graph could discover these factorizations if it has the right rules. The NNUE saturation head could learn to prioritize factoring rewrites — this is NOT the ick, because the factoring itself is still exact (verified by the variance annotation), only the SEARCH GUIDANCE is learned.

10. **Scope-aware CSE**: Two expressions with `depends_on = {Z}` that compute the same value are CSE candidates at Frame scope. The e-graph already merges them. But the variance annotation tells you that the merged computation lives at Frame scope, not Pixel scope. This is "free LICM" — it falls out of CSE + variance.

## References

- Conal Elliott, "Compiling to Categories" (2017)
- Conal Elliott, "Compiling Embedded Languages" (JFP 2003) — Pan compiler
- Conal Elliott, "Programming Graphics Processors Functionally" (2004) — Vertigo, frequency analysis
- Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation (2013)
- Reinking & Bernstein, "Formal Semantics for the Halide Language" (2020) — schedule transformations preserve denotational semantics
- egg: Fast and Extensible Equality Saturation (2021)
- The PixelFlow manifold algebra (see CLAUDE.md, pixelflow-core)
