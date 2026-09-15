> **Historical (Jan 2026) — a ranked wishlist against a compiler that has since
> been replaced (annotated 2026-09-09).** Several entries have been done by other
> means and several are moot: optimization now runs on the arena rather than the
> AST (#1206), the combinator tier whose ergonomics it costs is retired, and
> `context.rs` is gone. The ranking *method* — improvements to the compiler are
> multiplicative because every kernel pays them — is still the right frame, and is
> why the e-graph and codegen work has been prioritised the way it has. Check any
> individual row against the tree before acting on it.

# Compiler Improvement Opportunities

Analysis of potential improvements to the `kernel!` macro compiler, ranked by bang-for-buck.

**Key insight**: Compiler improvements have multiplicative value. Every kernel written benefits from better optimization, better error messages, or more ergonomic APIs. High implementation cost can be justified if it improves every use site.

## Ranking Criteria

- **Impact**: How much does every kernel benefit?
- **Effort**: Implementation complexity and maintenance burden
- **Risk**: Likelihood of hitting Rust limitations (trait solver, compile times)

---

## Tier 1: High Impact, Moderate Effort

### 1. Observable Sharing / Hash-Consing

| Aspect | Rating |
|--------|--------|
| Impact | ★★★★★ |
| Effort | ★★★☆☆ |
| Risk   | ★☆☆☆☆ |

**Problem**: `(X - cx) * (X - cx)` duplicates the subtree. Complex kernels can have exponential AST blowup.

**Solution**: Track structurally identical subexpressions during AST construction. Emit actual `let` bindings for shared nodes.

**Benefits**:
- Prevents exponential expression tree growth
- Guarantees CSE at DSL level (not relying on LLVM)
- Reduces generated code size
- Faster compile times for complex kernels

**Literature**: Gill, "Type-Safe Observable Sharing in Haskell" (2009)

**Implementation sketch**:
```rust
// In annotation pass, hash each node
// If hash collision + structural equality → reuse existing node
// Emit let-binding for any node used more than once
```

---

### 2. E-Graphs for Optimization

| Aspect | Rating |
|--------|--------|
| Impact | ★★★★★ |
| Effort | ★★★★☆ |
| Risk   | ★★☆☆☆ |

**Problem**: Current `optimize.rs` is ad-hoc peephole rules. Misses optimizations that require associativity/commutativity reasoning.

**Solution**: Use equality saturation via the `egg` crate. Define rewrite rules declaratively.

**Benefits**:
- Discovers non-obvious optimizations automatically
- Proves optimality via saturation
- Declarative rules are easier to maintain than pattern-matching code
- New rules don't interfere with existing ones

**Literature**: Willsey et al., "egg: Fast and Extensible Equality Saturation" (POPL 2021)

**Implementation sketch**:
```rust
// Define rewrite rules declaratively
let rules: &[Rewrite<KernelLang, ()>] = &[
    rewrite!("commute-add"; "(+ ?a ?b)" => "(+ ?b ?a)"),
    rewrite!("assoc-add"; "(+ (+ ?a ?b) ?c)" => "(+ ?a (+ ?b ?c))"),
    rewrite!("zero-add"; "(+ ?a 0)" => "?a"),
    rewrite!("distribute"; "(* ?a (+ ?b ?c))" => "(+ (* ?a ?b) (* ?a ?c))"),
    // ... more rules
];

// Run to saturation, extract optimal
let runner = Runner::default().with_expr(&expr).run(&rules);
let best = Extractor::new(&runner.egraph, AstSize).find_best(root);
```

---

### 3. Explicit AD Functor (WithGradient combinator)

| Aspect | Rating |
|--------|--------|
| Impact | ★★★★☆ |
| Effort | ★★☆☆☆ |
| Risk   | ★☆☆☆☆ |

**Problem**: AD is implicit—users must know to use Jet domain. No separation between "compute value" and "compute value + gradient".

**Solution**: Add `WithGradient<M, DIM>` combinator that lifts Field coordinates to Jet internally.

**Benefits**:
- Makes AD a first-class, explicit operation
- Field-domain input, Jet-domain output (cleaner API)
- Composable with other combinators
- Zero runtime overhead (monomorphizes away)

**Implementation**:
```rust
pub struct WithGradient<M, const DIM: usize>(pub M);

impl<M> Manifold<Field4> for WithGradient<M, 2>
where
    M: Manifold<Jet2_4, Output = Jet2>,
{
    type Output = Jet2;

    #[inline(always)]
    fn eval(&self, p: Field4) -> Jet2 {
        let (x, y, z, w) = p;
        let jp = (
            Jet2::var(x, 0),
            Jet2::var(y, 1),
            Jet2::constant(z),
            Jet2::constant(w),
        );
        self.0.eval(jp)
    }
}

// Usage:
let sdf = kernel!(|r: f32| (X*X + Y*Y).sqrt() - r);
let with_grad = WithGradient::<_, 2>(sdf(1.0));
let result: Jet2 = with_grad.eval(field_coords);  // Field in, Jet2 out!
```

---

### 4. Autotuned Extraction Costs

| Aspect | Rating |
|--------|--------|
| Impact | ★★★★☆ |
| Effort | ★★★☆☆ |
| Risk   | ★☆☆☆☆ |

**Problem**: E-graph extraction uses hand-tuned operation costs. But real costs depend on:
- CPU microarchitecture (Intel vs AMD vs Apple Silicon)
- SIMD width (AVX-512 vs SSE2 vs NEON)
- Latency vs throughput characteristics
- Cache pressure and register allocation

**Solution**: Treat extraction costs as hyperparameters. Tune them against real benchmarks per-platform.

**Benefits**:
- Empirically optimal code generation
- Platform-specific optimization without platform-specific rules
- Discovers non-obvious tradeoffs (e.g., rsqrt+Newton vs sqrt+div)
- Can be re-tuned for new hardware

**Architecture**:
```
┌─────────────────────────────────────────────────────────┐
│                    Build Time                           │
├─────────────────────────────────────────────────────────┤
│  kernel! macro                                          │
│      │                                                  │
│      ▼                                                  │
│  E-Graph saturation (platform-independent)              │
│      │                                                  │
│      ▼                                                  │
│  Extraction with cost table ◄─── costs_avx512.rs        │
│      │                           costs_sse2.rs          │
│      │                           costs_neon.rs          │
│      ▼                                                  │
│  Optimal ZST expression tree                            │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│              Offline Tuning (one-time)                  │
├─────────────────────────────────────────────────────────┤
│  Benchmark kernels ──► Measure cycles ──► Optimize      │
│                                             costs       │
│                                               │         │
│                                               ▼         │
│                                        costs_xxx.rs     │
└─────────────────────────────────────────────────────────┘
```

**Implementation sketch**:
```rust
/// Cost vector - one weight per operation
struct CostWeights {
    add: f32, mul: f32, div: f32,
    sqrt: f32, rsqrt: f32,
    muladd: f32,      // FMA - might equal mul on modern CPUs!
    mulrsqrt: f32,
    sin: f32, cos: f32,
    exp2: f32, log2: f32,
}

/// Evaluate: measure total runtime with these extraction costs
fn evaluate(weights: &CostWeights, benchmarks: &[Benchmark]) -> f64 {
    benchmarks.iter()
        .map(|b| compile_and_measure(b, weights))
        .sum()
}

/// Gradient-free optimization (costs aren't differentiable)
fn tune(benchmarks: &[Benchmark]) -> CostWeights {
    let mut best = CostWeights::default();
    let mut best_score = evaluate(&best, benchmarks);

    // Nelder-Mead, CMA-ES, or simple hill climbing
    for _ in 0..1000 {
        let candidate = best.perturb();
        let score = evaluate(&candidate, benchmarks);
        if score < best_score {
            best = candidate;
            best_score = score;
        }
    }
    best
}
```

**User-facing workflow**:
```bash
# Ship with reasonable defaults, but allow local tuning
cargo run -p xtask --release -- tune  # Benchmark for ~30s, emit costs_local.rs
cargo build --release     # Builds with tuned costs
```

**Interesting discoveries this might make**:
- FMA is "free" (same cost as mul) on modern CPUs
- `rsqrt` + Newton-Raphson vs `1/sqrt` depends on context
- Transcendental costs vary wildly (hardware vs polynomial approx)
- Memory-bound workloads prefer smaller expression trees

**Literature**:
- ATLAS (Whaley & Dongarra) - autotuned BLAS
- Halide autoscheduler (Adams et al.)
- OpenTuner (Ansel et al.)

**Prerequisite**: E-Graphs (#2) must be implemented first.

---

## Tier 2: Medium Impact, Variable Effort

### 5. Type-Level Symbolic AD (ZST Derivatives)

| Aspect | Rating |
|--------|--------|
| Impact | ★★★★★ |
| Effort | ★★★★☆ |
| Risk   | ★★☆☆☆ |

**Problem**: Jet carries derivatives through computation at runtime. Want compile-time derivative computation that follows the same ZST pattern as expression trees.

**Solution**: Symbolic differentiation at the *type level*. Just as `Add<A, B>` is ZST if both children are ZST, `Derivative<Add<A, B>>` produces a ZST derivative expression.

**Key Insight**: This follows the same structural pattern as ZST propagation:
```rust
// Current: ZST propagates through structure
Add<X, Y>           // ZST because X and Y are ZST

// Proposed: Derivative propagates through structure
trait DiffWrt<Var> {
    type D;  // The derivative TYPE (also ZST!)
}

// Sum rule: d/dv (a + b) = da/dv + db/dv
impl<V, A: DiffWrt<V>, B: DiffWrt<V>> DiffWrt<V> for Add<A, B> {
    type D = Add<A::D, B::D>;
}

// Product rule: d/dv (a * b) = da/dv * b + a * db/dv
impl<V, A: DiffWrt<V>, B: DiffWrt<V>> DiffWrt<V> for Mul<A, B> {
    type D = Add<Mul<A::D, B>, Mul<A, B::D>>;
}

// Variable rules
impl DiffWrt<XVar> for X { type D = One; }   // dx/dx = 1
impl DiffWrt<YVar> for X { type D = Zero; }  // dx/dy = 0
impl DiffWrt<XVar> for Y { type D = Zero; }  // dy/dx = 0
impl DiffWrt<YVar> for Y { type D = One; }   // dy/dy = 1

// Constants
impl<V, T> DiffWrt<V> for Const<T> { type D = Zero; }
```

**Example**:
```rust
// x² + y²
type Circle = Add<Mul<X, X>, Mul<Y, Y>>;

// ∂/∂x (x² + y²) - computed entirely at compile time!
type DCircleDx = <Circle as DiffWrt<XVar>>::D;
// Expands to: Add<Add<Mul<One, X>, Mul<X, One>>, Add<Mul<Zero, Y>, Mul<Y, Zero>>>
// After e-graph optimization: Add<X, X>  (i.e., 2x)

// The derivative is ALSO a ZST expression tree!
// Evaluating it has zero overhead vs. hand-written code.
```

**Gradient combinator becomes trivial**:
```rust
struct Grad2D<E>(PhantomData<E>);

impl<P, E> Manifold<P> for Grad2D<E>
where
    E: Manifold<P, Output = Field> + DiffWrt<XVar> + DiffWrt<YVar>,
    <E as DiffWrt<XVar>>::D: Manifold<P, Output = Field>,
    <E as DiffWrt<YVar>>::D: Manifold<P, Output = Field>,
{
    type Output = (Field, Field, Field);  // (value, dx, dy)

    #[inline(always)]
    fn eval(&self, p: P) -> Self::Output {
        // All three are ZST expression evaluations - zero overhead!
        let value = E::default().eval(p);
        let dx = <E as DiffWrt<XVar>>::D::default().eval(p);
        let dy = <E as DiffWrt<YVar>>::D::default().eval(p);
        (value, dx, dy)
    }
}
```

**Benefits**:
- Zero runtime overhead (derivative is ZST, just like original)
- Composable: `<<E as DiffWrt<X>>::D as DiffWrt<X>>::D` = second derivative
- E-graphs can optimize the derivative expression (simplify `Add<X, X>` → `Mul<Two, X>`)
- No Jet types needed for gradients
- Higher-order derivatives fall out naturally
- Follows existing architectural patterns (ZST propagation)

**Implementation needed** - derivative rules for each combinator:
```rust
impl<V, M: DiffWrt<V>> DiffWrt<V> for Sqrt<M> {
    // d/dv sqrt(m) = dm/dv / (2 * sqrt(m))
    type D = Div<M::D, Mul<Two, Sqrt<M>>>;
}

impl<V, M: DiffWrt<V>> DiffWrt<V> for Sin<M> {
    // d/dv sin(m) = cos(m) * dm/dv
    type D = Mul<Cos<M>, M::D>;
}

impl<V, M: DiffWrt<V>> DiffWrt<V> for Cos<M> {
    // d/dv cos(m) = -sin(m) * dm/dv
    type D = Neg<Mul<Sin<M>, M::D>>;
}

// etc. for Exp, Log, Atan2, Pow, ...
```

**Risk**: Product rule causes expression tree branching. But:
1. E-graphs can simplify the result
2. Observable sharing deduplicates common subexpressions
3. It's compile-time growth, not runtime

**Literature**: Elliott, "The Simple Essence of Automatic Differentiation" (2018)

---

### 6. Recursion Schemes (beyond catamorphism)

| Aspect | Rating |
|--------|--------|
| Impact | ★★★☆☆ |
| Effort | ★★★☆☆ |
| Risk   | ★★☆☆☆ |

**Problem**: Each compiler phase reimplements tree traversal. `ExprFold` is a basic catamorphism but more patterns exist.

**Solution**: Implement full recursion scheme library for AST.

**Benefits**:
- Paramorphism: optimization rules that inspect original + result
- Histomorphism: memoized results for dynamic programming on AST
- Hylomorphism: fused parse→emit without intermediate AST
- Cleaner phase implementations

**Literature**: Meijer et al., "Functional Programming with Bananas, Lenses, Envelopes and Barbed Wire" (1991)

**Implementation sketch**:
```rust
trait Recursive {
    type Base<T>;  // The "base functor"
    fn project(&self) -> Self::Base<&Self>;
}

fn cata<F, A>(alg: impl Fn(Expr::Base<A>) -> A, expr: &Expr) -> A { ... }
fn para<F, A>(alg: impl Fn(Expr::Base<(A, &Expr)>) -> A, expr: &Expr) -> A { ... }
```

---

### 7. Macro-Generated Tuple Impls

| Aspect | Rating |
|--------|--------|
| Impact | ★★☆☆☆ |
| Effort | ★★☆☆☆ |
| Risk   | ★☆☆☆☆ |

**Problem**: `context.rs` has ~800 lines of repetitive impl blocks for tuples 1-10.

**Solution**: Generate with a declarative macro.

**Benefits**:
- Easier to add new arities
- Single source of truth for the pattern
- Reduces maintenance burden
- Catches copy-paste errors

**Note**: The flat tuple approach is correct (HList causes trait solver explosion). This just reduces boilerplate.

```rust
macro_rules! impl_with_context {
    ($($idx:tt: $V:ident -> $O:ident),*) => {
        impl<P, $($V,)* B, $($O,)* Out> Manifold<P>
            for WithContext<($($V,)*), B>
        where
            P: Copy + Send + Sync,
            $($V: Manifold<P, Output = $O>,)*
            $($O: Copy + Send + Sync,)*
            B: Manifold<(($($O,)*), P), Output = Out>,
        {
            type Output = Out;
            #[inline(always)]
            fn eval(&self, p: P) -> Self::Output {
                $(let $V = self.ctx.$idx.eval(p);)*
                self.body.eval((($($V,)*), p))
            }
        }
    };
}

impl_with_context!(0: V0 -> O0);
impl_with_context!(0: V0 -> O0, 1: V1 -> O1);
// ... etc
```

---

## Tier 3: Theoretical Interest, High Effort

### 8. Tagless Final Encoding

| Aspect | Rating |
|--------|--------|
| Impact | ★★★★☆ |
| Effort | ★★★★★ |
| Risk   | ★★★★☆ |

**Problem**: Current AST uses initial encoding (enum). Adding new operations or interpretations requires modifying existing code.

**Solution**: Tagless final—define operations as trait methods, not data constructors.

**Benefits**:
- No intermediate AST allocation
- Multiple interpretations (eval, pretty-print, optimize) without pattern matching
- Solves the expression problem
- More extensible

**Risk is high** because:
- Massive rewrite of parser and all phases
- Rust's trait system is less ergonomic than Haskell's typeclasses for this pattern
- May hit trait solver limits with complex expressions

**Literature**: Kiselyov, "Typed Tagless Final Interpreters" (JFP 2009)

---

### 9. Trees That Grow

| Aspect | Rating |
|--------|--------|
| Impact | ★★☆☆☆ |
| Effort | ★★★☆☆ |
| Risk   | ★★☆☆☆ |

**Problem**: AST nodes have fields (like `span`) unused in later phases.

**Solution**: Parameterize AST by phase, with extension points.

**Benefits**:
- Type system enforces phase correctness
- Smaller AST in later phases
- Zero-cost phase transitions

**Literature**: Najd & Peyton Jones, "Trees That Grow" (2016)

```rust
trait Phase {
    type ExprExt;
    type LiteralExt;
}

struct Parsed;
impl Phase for Parsed {
    type ExprExt = Span;
    type LiteralExt = Span;
}

struct Optimized;
impl Phase for Optimized {
    type ExprExt = ();  // No spans needed
    type LiteralExt = ();
}

enum Expr<P: Phase> {
    Literal { value: f64, ext: P::LiteralExt },
    Binary { op: BinaryOp, lhs: Box<Expr<P>>, rhs: Box<Expr<P>>, ext: P::ExprExt },
    // ...
}
```

---

## Summary Table

| Opportunity | Impact | Effort | Risk | Bang/Buck | Priority |
|-------------|--------|--------|------|-----------|----------|
| Observable Sharing | ★★★★★ | ★★★☆☆ | ★☆☆☆☆ | **High** | 1 |
| E-Graphs | ★★★★★ | ★★★★☆ | ★★☆☆☆ | **High** | 2 |
| Explicit AD Functor | ★★★★☆ | ★★☆☆☆ | ★☆☆☆☆ | **High** | 3 |
| Autotuned Costs | ★★★★☆ | ★★★☆☆ | ★☆☆☆☆ | **High** | 4 |
| Type-Level Symbolic AD | ★★★★★ | ★★★★☆ | ★★☆☆☆ | **High** | 5 |
| Recursion Schemes | ★★★☆☆ | ★★★☆☆ | ★★☆☆☆ | Medium | 6 |
| Macro-Gen Tuples | ★★☆☆☆ | ★★☆☆☆ | ★☆☆☆☆ | Medium | 7 |
| Tagless Final | ★★★★☆ | ★★★★★ | ★★★★☆ | Low | 8 |
| Trees That Grow | ★★☆☆☆ | ★★★☆☆ | ★★☆☆☆ | Low | 9 |

## Recommended Roadmap

### Phase 1: Quick Wins
1. **Explicit AD Functor** - Low effort, immediate ergonomic benefit
2. **Macro-generate tuple impls** - Reduce maintenance burden

### Phase 2: Core Improvements
3. **Observable Sharing** - Prevents pathological cases, enables efficient AD
4. **E-Graphs** - Better optimization with less code, handles FMA commutativity
5. **Autotuned Costs** - Platform-specific optimal extraction (requires E-Graphs)

### Phase 3: Zero-Overhead Gradients
6. **Type-Level Symbolic AD** - Compile-time derivatives following ZST pattern

### Phase 4: Polish (if needed)
7. **Recursion Schemes** - If compiler phases get complex

### Defer Indefinitely
- Tagless Final (too invasive for unclear benefit in Rust)

---

## References

1. Gill, A. (2009). "Type-Safe Observable Sharing in Haskell"
2. Willsey, M. et al. (2021). "egg: Fast and Extensible Equality Saturation" - POPL
3. Elliott, C. (2018). "The Simple Essence of Automatic Differentiation"
4. Kiselyov, O. (2009). "Typed Tagless Final Interpreters" - JFP
5. Meijer, E. et al. (1991). "Functional Programming with Bananas, Lenses, Envelopes and Barbed Wire"
6. Najd, S. & Peyton Jones, S. (2016). "Trees That Grow"
7. Elliott, C. (2017). "Compiling to Categories"
