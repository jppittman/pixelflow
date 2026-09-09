> **Resolved and obsolete (annotated 2026-09-09).** The defect this investigates —
> kernels with more than three parameters failing on trait-bound errors — was a
> property of the type-level combinator tier, which no longer exists. `kernel!`
> lowers to an arena and folds parameters in as constants; the tree today contains
> working five- and seven-parameter kernels. The root cause identified below is
> correct history for a compiler that was replaced, not a live constraint. See
> [`plans/2026-07-20-kernel-unification.md`](plans/2026-07-20-kernel-unification.md).

# Kernel Parameter Limit Investigation

**Date**: 2026-01-19
**Issue**: Kernels with >3 parameters fail to compile with trait bound errors
**Status**: Root cause identified

## Executive Summary

The `kernel!` macro has a practical limit of ~3 parameters before the Rust compiler's trait solver gives up. This limit persists **even with binary-encoded type-level numbers** (O(log n) depth instead of O(n)). The root cause is not type nesting depth, but rather **combinatorial explosion of trait bound obligations**.

### Key Finding

Binary encoding reduced the **depth** of type-level numbers (N255: 8 levels vs 255 levels), but the compiler must still prove trait bounds for **every combination** of:
- Each Let binding layer (params + literals)
- Each Var reference in the body
- Each operator/combinator applied to Vars
- Each recursive Pred step for indexing

With 5 parameters + a few literals, this creates thousands of trait bound obligations.

---

## How kernel! Works

### Code Generation Pipeline

When you write:
```rust
kernel!(|a: f32, b: f32, c: f32| { a + b + c })
```

The macro generates (simplified):
```rust
struct __Kernel { a: f32, b: f32, c: f32 }

impl Manifold<Field4> for __Kernel {
    fn eval(&self, p: Field4) -> Field {
        let __expr = Var::<N2>::new() + Var::<N1>::new() + Var::<N0>::new();
        Let::new(self.a,
            Let::new(self.b,
                Let::new(self.c,
                    __expr))).eval(p)
    }
}
```

### The Hidden Cost: Literal Hoisting

**CRITICAL**: The macro also hoists **all literals** from the body into Let bindings!

Example with literals:
```rust
kernel!(|a: f32, b: f32| { (a + b) * 0.5 })
```

Becomes:
```rust
// 0.5 gets hoisted!
Let::new(a,
    Let::new(b,
        Let::new(0.5,  // ← Literal becomes a binding!
            (Var::<N2> + Var::<N1>) * Var::<N0>
        )))
```

**Impact**: A 5-parameter kernel with 3 literals in the body creates **8 nested Let bindings**.

See: `pixelflow-compiler/src/codegen.rs:171-182`

```rust
// Literals get indices 0..m-1, params get indices m..m+n-1
let (annotated_body, _, collected_literals) = annotate(&self.analyzed.def.body, annotation_ctx);
let literal_count = collected_literals.len();

// Adjust param indices to account for literals
for (_, idx) in self.param_indices.iter_mut() {
    *idx += literal_count;  // Params are shifted by literal count!
}
```

---

## Trait Bound Explosion

### What the Compiler Must Prove

For each `Let::new(val, body).eval(p)`, the compiler must prove:

```rust
impl<P, Val, Body> Manifold<P> for Let<Val, Body>
where
    P: Copy + Send + Sync,
    Val: Manifold<P>,
    Val::Output: Copy + Send + Sync,
    Body: Manifold<LetExtended<Val::Output, P>>,  // ← Recursive!
    //   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    //   Body must work on EXTENDED domain
```

### Concrete Example: 3 Parameters

```rust
kernel!(|a: f32, b: f32, c: f32| { a + b + c })
```

Expands to:
```rust
Let::new(a,           // Outer
    Let::new(b,       // Middle
        Let::new(c,   // Inner
            Var::<N2> + Var::<N1> + Var::<N0>
        )))
```

Trait bounds to prove:

1. **Outer Let**: `Let<f32, Let<...>>: Manifold<P>`
   - Requires: `Let<f32, Let<...>>: Manifold<LetExtended<Field, P>>`

2. **Middle Let**: `Let<f32, Let<...>>: Manifold<LetExtended<Field, P>>`
   - Requires: `Let<f32, ...>: Manifold<LetExtended<Field, LetExtended<Field, P>>>`

3. **Inner Let**: `Let<f32, Expr>: Manifold<LetExtended<Field, LetExtended<Field, P>>>`
   - Requires: `Expr: Manifold<LetExtended<Field, LetExtended<Field, LetExtended<Field, P>>>>`

4. **For each Var in Expr**:
   - `Var<N2> + Var<N1> + Var<N0>` means proving 3 Var impls + 2 Add impls
   - Each Var must prove it can index into the 3-deep LetExtended domain

### Var Indexing Proof Chain

For `Var<N2>` (which is `Var<UInt<UInt<UTerm, B1>, B0>>` in binary):

```rust
// Step 1: Prove Var<UInt<UInt<UTerm, B1>, B0>> works
impl<U, B, P> Manifold<P> for Var<UInt<U, B>>
where
    UInt<U, B>: Pred,                           // ← Must prove Pred
    <UInt<U, B> as Pred>::Output: Send + Sync, // ← Output must be valid
    P: Tail,                                     // ← Domain must have tail
    P::Rest: Copy,                               // ← Tail must be copyable
    Var<<UInt<U, B> as Pred>::Output>: Manifold<P::Rest>,  // ← RECURSIVE!
    //  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    //  Must prove predecessor Var works on tail domain
```

For N2 (binary: `0b10`), the Pred chain:
1. `UInt<UInt<UTerm, B1>, B0>: Pred` → `UInt<UTerm, B1>` (borrow from higher bit)
2. `Var<UInt<UTerm, B1>>: Manifold<Tail::Rest>`
3. `UInt<UTerm, B1>: Pred` → `UTerm` (decrement to 0)
4. `Var<UTerm>: Manifold<Tail::Rest::Rest>` (base case)

**With binary encoding**: 2-3 recursive steps for small indices (vs 5 steps for unary N5)
**Depth improvement**: ✅ 2-3x fewer steps
**Problem**: Still recursive, and multiplies with operator count

---

## The Combinatorial Explosion

### 5 Parameters + 3 Literals = 8 Let Bindings

```rust
kernel!(|a: f32, b: f32, c: f32, d: f32, e: f32| {
    let t = (Y - cy) / by;
    let in_range = (t >= 0.0) & (t <= 1.0);
    let x_at_t = t * bx + cx;
    let crosses = X < x_at_t;
    in_range.select(crosses.select(dir, 0.0), 0.0)
})
```

**Collected literals**: `0.0`, `1.0`, possibly more from operations
**Total Let bindings**: 5 params + 3 literals = **8 layers**

**Domain nesting depth**: `LetExtended<Field, LetExtended<Field, LetExtended<..., P>>>` (8 deep)

### Proof Obligations Per Expression

For a simple `a + b`:
1. Prove `Add<Var<Na>, Var<Nb>>: Manifold<Domain>`
2. Prove `Var<Na>: Manifold<Domain>` → requires Na-1 recursive Pred steps
3. Prove `Var<Nb>: Manifold<Domain>` → requires Nb-1 recursive Pred steps
4. Prove `Field + Field → Field` (operator trait bounds)

For complex expressions with nested operations (`a + b > c`, `x.select(y, z)`):
- Each sub-expression must be proven independently
- Trait solver explores ALL possible proof paths
- With 8 Let layers, this branches exponentially

### Why Binary Encoding Helped (But Not Enough)

**Unary encoding** (Succ/Zero):
- N5 = `Succ<Succ<Succ<Succ<Succ<Zero>>>>>`
- Depth: 5 nested types
- Pred steps: 5 recursive proofs

**Binary encoding** (UInt/UTerm/B0/B1):
- N5 = `UInt<UInt<UInt<UTerm, B1>, B0>, B1>` (binary: 0b101)
- Depth: 3 nested types ✅ (improvement)
- Pred steps: 3 recursive proofs ✅ (improvement)

**But**:
- Still recursive (just fewer steps)
- Doesn't reduce the NUMBER of Vars to prove
- Doesn't reduce the NUMBER of Let layers
- Doesn't reduce combinatorial explosion from operators

---

## Why 3 Parameters Works But 5 Doesn't

### Empirical Threshold

| Parameters | Literals | Total Lets | Compiles? | Observation |
|------------|----------|------------|-----------|-------------|
| 2          | 1-2      | 3-4        | ✅        | Fast compilation |
| 3          | 1-2      | 4-5        | ✅        | Slow but succeeds |
| 4          | 1-2      | 5-6        | ⚠️        | Very slow, may timeout |
| 5          | 2-3      | 7-8        | ❌        | Trait solver gives up |
| 5          | 0        | 5          | ⚠️        | Might work without literals |

### Trait Solver Limits

Rust's trait solver has internal limits (not publicly documented):
- Recursion depth limit (can be increased with `#![recursion_limit = "N"]`)
- **Proof obligation count limit** (cannot be configured)
- **Time limit** for proof search (cannot be configured)

With 8 Let layers and complex expressions:
- Thousands of trait bounds to prove
- Each requires exploring multiple impl candidates
- Solver hits internal limits and gives up

**Increasing recursion_limit doesn't help** because the issue isn't depth—it's the sheer number of independent proofs.

---

## Attempted Solutions

### ✅ Binary Encoding (Completed)

**Implementation**: Replaced `Succ<Succ<Zero>>` with `UInt<UTerm, B1>` (binary encoding)

**Results**:
- Reduced type nesting depth from O(n) to O(log n)
- N255: 8 levels instead of 255 levels
- N5: 3 levels instead of 5 levels

**Impact**:
- Compilation slightly faster for 3-4 param kernels
- Still fails for 5+ param kernels
- **Root cause persists**: Combinatorial proof explosion

### ❌ Increasing Recursion Limit

**Attempt**: Set `#![recursion_limit = "2048"]` in pixelflow-core

**Result**: No improvement

**Reason**: Recursion limit controls *depth* of proof search, not *breadth*. The issue is too many independent proofs, not too-deep proofs.

### ⚠️ Splitting Kernels Manually

**Approach**: User manually splits 5-param kernel into nested 2-param + 3-param kernels

**Problem**: `kernel!` macro doesn't support:
- Nested closures: `|a, b| { |c, d, e| { ... } }`
- Tuple destructuring: `|params: (f32, f32, f32)| { let (a, b, c) = params; ... }`

**Workaround**: User must write separate kernels and compose manually (verbose, defeats purpose of macro)

---

## Proposed Solutions

### Option 1: Automatic Kernel Splitting in Macro ⭐

**Idea**: Modify `kernel!` macro to automatically split large parameter lists into nested kernels.

**Input**:
```rust
kernel!(|a: f32, b: f32, c: f32, d: f32, e: f32| { body })
```

**Generated code**:
```rust
kernel!(|a: f32, b: f32, c: f32| {
    kernel!(|d: f32, e: f32| { body })
})
```

**Challenges**:
1. Macro must rewrite Var indices in body to account for nesting
2. Inner kernel must capture variables from outer kernel
3. Type inference becomes more complex

**Estimated effort**: Medium (2-3 days of macro work)

**Pros**:
- Transparent to user
- No API changes
- Works with any parameter count

**Cons**:
- Complex macro implementation
- Potential for confusing error messages
- May hit limits again with 10+ parameters (nested splitting)

### Option 2: Flat Context Tuple Instead of Recursive Domain

**Idea**: Replace recursive `LetExtended<Field, LetExtended<Field, P>>` with flat tuple context `(Field, Field, Field, ..., P)`.

**Current**:
```rust
pub struct LetExtended<V, Rest>(pub V, pub Rest);  // Recursive nesting

impl<P> Head for LetExtended<V, P> { ... }
impl<P> Tail for LetExtended<V, P> { ... }
```

**Proposed**:
```rust
// Flat tuple of up to N values + base domain
type Context8<P> = (Field, Field, Field, Field, Field, Field, Field, Field, P);

// Implement indexed access without recursion
trait Get<const N: usize> {
    fn get(&self) -> Field;
}

impl<P> Get<0> for Context8<P> {
    fn get(&self) -> Field { self.0 }
}
impl<P> Get<1> for Context8<P> {
    fn get(&self) -> Field { self.1 }
}
// etc...
```

**Pros**:
- Eliminates recursive domain extension
- Direct indexing (no Pred/Tail recursion)
- Provably O(1) proof complexity

**Cons**:
- **Breaking change**: Fundamentally different domain model
- Limits context size to tuple length (max 12-16 in Rust)
- Would require rewriting all Manifold impls

**Estimated effort**: Large (1-2 weeks, high risk)

### Option 3: Reduce Literal Hoisting

**Idea**: Don't hoist literals into Let bindings; inline them directly in expressions.

**Current** (5 params + 3 literals = 8 Lets):
```rust
Let::new(a, Let::new(b, Let::new(c, Let::new(d, Let::new(e,
    Let::new(0.0, Let::new(1.0, Let::new(0.5,
        Var<N7> + Var<N6> * Var<N0>  // 0.5 is Var<N0>
    ))))))))
```

**Proposed** (5 params + 0 literals = 5 Lets):
```rust
Let::new(a, Let::new(b, Let::new(c, Let::new(d, Let::new(e,
    Var<N4> + Var<N3> * 0.5  // Inline literal directly
)))))
```

**Challenges**:
- Literals must support Manifold trait (currently they don't)
- May break Jet domain support (constants need special wrapping)
- Complicates operator trait impls (Var<N> + f32 vs Var<N> + Var<M>)

**Pros**:
- Reduces Let nesting for most kernels
- Relatively small code change

**Cons**:
- Doesn't solve the fundamental issue (5 Lets still too many)
- May introduce type system issues

**Estimated effort**: Medium (3-5 days)

### Option 4: Compiler Plugin / Extern Proc Macro

**Idea**: Use unstable Rust features to bypass trait solver limits.

**Options**:
- Custom trait solver (requires nightly + unstable features)
- Procedural macro that generates monomorphized code directly
- LLVM plugin for specialized compilation

**Pros**:
- Could handle arbitrary parameter counts
- Full control over code generation

**Cons**:
- Requires nightly Rust (project currently stable)
- Unstable features may break between Rust versions
- Significant maintenance burden

**Estimated effort**: Very large (weeks to months, ongoing maintenance)

**Recommendation**: ❌ Not viable for this project

---

## Recommendations

### Short Term: Option 1 (Automatic Kernel Splitting) ⭐

**Why**: Best balance of user experience vs implementation effort.

**Implementation plan**:
1. Modify `parser.rs` to detect param count > 3
2. Split params into chunks: `[p0, p1, p2] + [p3, p4, ...]`
3. Generate nested kernel with outer params captured
4. Rewrite Var indices in body to account for nesting depth

**Risk**: Medium (macro complexity)
**Benefit**: Transparent to users, works immediately

### Long Term: Option 2 (Flat Context)

**Why**: Provably solves the fundamental issue, but requires major refactor.

**Approach**:
1. Prototype flat context in new module
2. Benchmark against recursive domain
3. If successful, migrate incrementally
4. Deprecate old LetExtended approach

**Risk**: High (breaking change, extensive testing)
**Benefit**: Eliminates problem permanently

### Immediate Workaround: Document Limit + Provide Examples

**For now**:
1. Document 3-parameter limit in kernel! macro docs
2. Provide examples of manual kernel composition:

```rust
// Instead of:
// kernel!(|a, b, c, d, e| { body })

// Write:
let k_outer = kernel!(|a: f32, b: f32, c: f32| {
    |d: f32, e: f32| {
        // Capture a, b, c from outer scope
        // Body uses a, b, c directly and d, e as arguments
    }
});
```

---

## Technical Deep Dive: Why Trait Solver Gives Up

### Rust Trait Resolution Algorithm

Rust uses **lazy normalization** and **depth-first search** for trait resolution:

1. **Goal**: Prove `T: Trait<P>`
2. **Search space**: All `impl Trait for U where ...` blocks
3. **Recursion**: If bound requires `V: OtherTrait`, recursively prove that
4. **Caching**: Memoize successful proofs (but not failed attempts)

### The Exponential Blowup

For `Let<Val, Body>: Manifold<P>`:

```
Prove: Let<Val, Body>: Manifold<P>
├─ Prove: Val: Manifold<P>
│  └─ Prove: Field: Manifold<P>  [cached]
├─ Prove: Val::Output: Copy
├─ Prove: Body: Manifold<LetExtended<Val::Output, P>>
   ├─ If Body = Let<V2, B2>:
   │  ├─ Prove: V2: Manifold<LetExtended<...>>
   │  └─ Prove: B2: Manifold<LetExtended<V2::Output, LetExtended<...>>>
   │     └─ RECURSE (depth + 1)
   └─ If Body = Expr:
      └─ Prove: Expr: Manifold<LetExtended<...>>
         ├─ If Expr = Add<L, R>:
         │  ├─ Prove: L: Manifold<LetExtended<...>>
         │  ├─ Prove: R: Manifold<LetExtended<...>>
         │  └─ Prove: L::Output == R::Output
         └─ If L = Var<N>:
            ├─ Prove: Var<N>: Manifold<LetExtended<...>>
            ├─ Prove: N: Pred
            ├─ Prove: P: Tail
            └─ Prove: Var<Pred<N>>: Manifold<Tail::Rest>>
               └─ RECURSE (Pred chain)
```

**With 8 Let layers and 5 Var references**:
- Each Let adds 1 depth level
- Each Var adds log2(N) Pred recursions
- Each operator multiplies the proof tree

**Approximate proof count**:
- 8 Let proofs: `O(8)`
- 5 Var proofs × 3 Pred steps: `O(15)`
- 10 operators × 2 operands: `O(20)`
- **Total**: ~43+ independent proof obligations

**Actual** (with all intermediate steps): Likely **hundreds** to **thousands** of trait bound checks.

### Trait Solver Heuristics

The compiler uses heuristics to prune the search space:
- **Depth limit**: Stop after N levels of recursion (configurable via `recursion_limit`)
- **Proof cache**: Reuse proven bounds (helps)
- **Early pruning**: Give up if obviously impossible (helps)
- **Timeout**: Abort after X seconds of proving (NOT configurable)

**Problem**: With 8 Lets + complex body, the solver likely hits the timeout or proof count limit before succeeding.

---

## Conclusion

### Root Cause

The 3-parameter limit stems from **combinatorial explosion** of trait bound proofs, not type nesting depth. Binary encoding helped reduce depth but didn't solve the fundamental issue.

### Key Insights

1. **Literal hoisting amplifies the problem**: 5 params + 3 literals = 8 Let layers
2. **Recursive domain extension is expensive**: Each LetExtended layer multiplies proof obligations
3. **Binary encoding was necessary but insufficient**: Reduced depth from O(n) to O(log n), but proof count is still exponential
4. **Trait solver has hard limits**: Cannot be bypassed with rustc flags

### Recommended Path Forward

1. **Immediate**: Implement automatic kernel splitting (Option 1)
2. **Medium-term**: Investigate flat context tuples (Option 2)
3. **Long-term**: Consider DSL compilation to avoid trait solver entirely

### Status

- ✅ Binary encoding implemented and committed (commit 9eecc29)
- ✅ Literal inlining optimization implemented and committed (commit 5809192)
- ✅ pixelflow-graphics now compiles successfully!
- ⚠️ 5+ parameter kernels still fail (needs automatic splitting or flat context)
- 📊 Achieved 33-38% reduction in Let nesting depth

### Results

**Binary Encoding (Commit 9eecc29)**:
- Reduced type depth from O(n) to O(log n)
- N255: 255 levels → 8 levels
- Improved compilation speed for 3-4 param kernels
- Did not solve fundamental trait explosion issue

**Literal Inlining (Commit 5809192)**:
- Eliminated Let bindings for literals in non-Jet mode
- 5-param + 3 literals: 8 Lets → 5 Lets (38% reduction)
- 2-param + 1 literal: 3 Lets → 2 Lets (33% reduction)
- **pixelflow-graphics compiles successfully!**

**Combined Impact**:
- ✅ Kernels with ≤3 parameters work reliably
- ⚠️ Kernels with 4-5 parameters may work depending on expression complexity
- ❌ Kernels with 5+ parameters still hit trait solver limits
- 🎯 **Main goal achieved**: Project compiles and demonstrates both optimizations

---

## Appendix: Reproduction Case

```rust
// This FAILS with trait bounds error:
let k = kernel!(|a: f32, b: f32, c: f32, d: f32, e: f32| {
    let t = (Y - a) / b;
    let in_range = (t >= 0.0) & (t <= 1.0);
    let x_at_t = t * c + d;
    let crosses = X < x_at_t;
    in_range.select(crosses.select(e, 0.0), 0.0)
});

// Actual Let nesting: 5 params + 3 literals (0.0, 1.0, 0.0) = 8 layers
// Approximate proof obligations: 500+
// Compiler: "trait bounds not satisfied"
```

Error message:
```
error[E0599]: the method `eval` exists for struct `Let<f32, Let<f32, Let<f32, Let<f32, Let<f32, Let<{float}, ...>>>>>>`, but its trait bounds were not satisfied
```

Translation: "I need to prove this 8-deep Let structure implements Manifold, and I gave up trying."
