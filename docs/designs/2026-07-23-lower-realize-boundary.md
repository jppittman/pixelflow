# Lower/realize: the compilation boundary consumers never cross

**Date:** 2026-07-23
**Status:** Design for the P5 rebuild (supersedes the reverted `HasIr`-in-graphics attempt)

## The failure being corrected

The first P5 attempt had pixelflow-graphics walking the glyph scene graph and
assembling `ExprArena` nodes by hand: a `fused.rs` translator, `.ir() ->
Box<dyn HasIr>` accessors on curve kernels, a `GlyphIr` carrier passed into a
macro. That is a consumer manipulating the compiler's IR — the exact thing
Halide and LuaJIT never let happen. Halide users write `Func`s and call
`realize`; the IR exists, is even public, but no pipeline author touches it.
The missing level of abstraction: **lowering is a property of the language,
not a chore for consumers.**

## The boundary

One trait, implemented *beside* each combinator's `Manifold` impl; one verb,
which is all a consumer ever calls.

```rust
// pixelflow-ir (re-exported properly by pixelflow-core, not via __ir):
pub trait Lower {
    /// Emit this manifold's expression into `arena`, or `None` if it has no
    /// IR form (yet). Consumers never call or implement this — it rides as a
    /// bound, like `Send`.
    fn lower(&self, arena: &mut ExprArena) -> Option<ExprId>;
}

// pixelflow-core:
impl Lattice {
    /// Halide's verb: tabulate the manifold over this lattice. Lowers and
    /// JIT-compiles ONE fused kernel (through the global compile cache) when
    /// the whole tree lowers; falls back to the generic per-batch `collapse`
    /// when any node declines or a backend refuses. The consumer cannot
    /// observe which path ran except by speed.
    pub fn realize<M>(&self, m: &M) -> DiscreteManifold
    where M: Manifold<Field4, Output = Field> + Lower;
}
```

`CachedGlyph::new` changes exactly one word: `collapse` → `realize`. That is
the entire consumer-visible surface of P5.

## Who implements `Lower`

- **pixelflow-core combinators, one impl per generator, next to `Manifold`:**
  coordinates (`X`→`Var(0)`…), scalar/`Field` constants, the unary/binary op
  ZSTs, comparisons, `And`/`Or` (→`BitAnd`/`BitOr` — canonical masks in both
  tiers), `Select`, `Let`/`Var` bindings, `WithContext`/`CtxVar` (lower inner
  with `CtxVar<_, N>` → `Param(N)`, then `substitute_params` with the stored
  values — the existing machinery), the derivative projections (`ValOf` →
  identity, `DxOf`/`DyOf`/`DzOf` → `Dwrt`), and `At` (lower the coordinate
  expressions, lower the inner, `substitute_vars_with` — contramap is
  precomposition, in the language).
- **Macro backends:** the `kernel_jit!` wrapper lowers by splicing its stored
  arena (absorbing `HasIr`, which this trait replaces and renames); `kernel!`
  named structs keep the compositional impl they already have. Builder
  manifold-params bound on `Lower` instead of `HasIr` — which also opens
  composition to any lowerable combinator, not just JIT wrappers.
- **Library combinator types** (`Sum`, `Geometry`, `Glyph`, `Antialiased`,
  `AffineTransform`) implement it beside their hand-written `Manifold` impls.
  Extending the language has always meant implementing `Manifold`; it now
  means implementing both — the same locality as `Display` next to `Debug`.
  Notably `Antialiased::lower` is the **identity**: the wrapper exists to
  switch evaluation strategy (seed `Jet2`), and in the IR that strategy is
  already `Dwrt` — the "one kernel language" thesis made concrete.
- **Leaf kernels** (`AnalyticalLine`/`AnalyticalQuad`) implement it by
  **delegation**: their `eval` stamps already build a `kernel!` combinator
  tree; `lower` builds the same tree and lowers *it*. One body, no twins, and
  zero arena pushes anywhere in pixelflow-graphics.

The key consequence: because anonymous `kernel!` output is a composition of
core combinator ZSTs, **everything ever built with the macros lowers for
free** once the core generators have impls. The functor from the combinator
category to the arena category is defined once per generator, and composition
does the rest — which is also why this aligns with the phases-are-morphisms
refactor rather than fighting it.

## Fallibility policy

`lower` returns `Option`: a node with no IR form (today: `DiscreteManifold`
sampling, `Bilinear`, anything wrapping opaque Rust) declines, and `realize`
falls back to generic collapse — the plan's "combinator path remains the
fallback until goldens pass," made structural. Bound-memory nodes stop
declining when lattice sampling lowers to `Gather` (the M4 completion);
`realize` then fuses frames end to end with no consumer change.

## Backend prerequisite (found by the first attempt)

A whole-glyph fused kernel (~50 segments × ~40 nodes after derivative
lowering) exceeds the x86 per-batch emitter's spill budget: "spill frame
exceeds 128-byte red zone." The emitter must allocate a real frame
(`sub rsp, N`) when the schedule's spill footprint exceeds the red zone —
aarch64-style — before glyph-scale kernels compile on x86. This is
independent of the boundary design and needed regardless.

## Order of work

1. x86 spill frame (unblocks glyph-scale kernels).
2. `Lower` in pixelflow-ir + core generator impls + `Lattice::realize`;
   `HasIr` renamed away; macro backends and `kernel_composition` tests moved
   onto the new bound.
3. Graphics impls (`Sum`/`Geometry`/`Glyph`/`Antialiased`/leaf delegation);
   `CachedGlyph::new` switches to `realize`; fused-vs-combinator golden
   becomes the P5 acceptance again — now with zero IR code in graphics.
