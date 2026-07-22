# One Kernel Language: Retiring the Combinator Emitter

**Date:** 2026-07-20
**Status:** Plan of record
**Supersedes:** the dual-backend framing of the 2025-02-21 kernel-jit-feature-parity plan (whose param-baking scope is done).

## North star

We don't want two languages. There is one kernel language with one pipeline:

```
parse → sema → e-graph → ExprArena → backend emit
```

The proc-macro and the runtime compiler are the *same pipeline* run at different
times with different e-graph budgets:

- **AOT (`kernel!`):** parse/sema/e-graph run at macro-expansion time with a
  generous saturation budget; the macro emits tokens reconstructing the
  *pre-optimized arena*; machine code is emitted once at load time (kernels are
  constructed at load time — existing project law).
- **Runtime (fused roots, dynamic composition, param re-folding):** same
  pipeline with a bounded e-graph budget. The guided-saturation / rule-masking
  research (docs/plans/2026-07-07-guided-saturation-redesign.md) is the budget
  policy for this tier.

**The combinator emitter (`pixelflow-compiler/src/codegen/`, ZST Let/Var trees)
is the part that goes away.** It is not the long-term fallback. It keeps working
until the parity suite + font goldens prove arena coverage, then it is deleted.
`kernel!` and `kernel_jit!` converge into one macro surface.

## What replaces the combinator emitter's roles

| Role today | Replacement |
|---|---|
| Jet2/Jet3 domain evaluation (font AA, 3D normals) | `Dwrt` symbolic differentiation lowered in pixelflow-ir; derivatives are ordinary expressions compiled by the same backend. NOT the deleted emit-time jet mode (bd47aa7c) — that was an architectural trap. |
| Portable / reference semantics (SSE2, WASM, scalar — no JIT emitters) | The IR interpreter (`pixelflow-ir/src/eval.rs`). Same arena, same semantics, slow. WASM may later get its own arena emitter. |
| Composition (`.at()`, `Sum`, named structs, manifold params) | IR-carrying kernels: every kernel exposes its arena fragment; composition is arena splicing (contramap = Var substitution, Sum = chained Add). |
| Opaque Rust manifolds (DiscreteManifold, dynamic BSP, CachedGlyph) | Enter the IR as bound buffers / ctx-pointer calls (the existing Gather pattern), not as expressions. The `Manifold` trait survives as the composition substrate. |

Numerics are defined once: the JIT's near-libm behavior is canonical. The
combinator backend's inaccurate `sin` (0.5116 vs 0.8415 at x=1; see
pixelflow-compiler/tests/jit_parity.rs header) is a parity bug in the
to-be-retired backend, tracked until that backend dies.

## Phases

- **P0 — ground truth** *(in flight)*: JIT compile-cost bench
  (docs/results/2026-07-20-jit-compile-cost.md) settles load-time compile
  budgets; golden coverage-lattice fixtures for the warm-ASCII font set are the
  visual regression guard for every backend switch.
- **P0.5 — e-graph type stability** *(in flight)*: extraction must be
  type-stable across the V/DX/DY projection boundary and must not share
  non-Copy manifold-param subexpressions. Prerequisite for fonts on `kernel!`;
  the defect class itself dies with the emitter, but fonts run on the emitter
  until parity lands.
- **P2 — `lower_dwrt` in pixelflow-ir** *(landed 2026-07-22)*:
  symbolic-differentiation lowering pass
  (peer of `expand_gather`/`expand_reduce`); ir_bridge maps V/DX/DY/DZ →
  `Dwrt` nodes (plus DXX/DXY/DYY as nested `Dwrt`); the codegen panic at a
  surviving `Dwrt` becomes an unreachable
  precondition. Errors loudly on unsupported ops. Acceptance: font-shaped
  coverage expressions JIT-compile and match combinator-over-Jet2 within
  tolerance — see `jit_font_coverage_matches_*` in
  pixelflow-compiler/tests/jit_parity.rs.
- **P3 — routing scaffolding**: `kernel!` routes eligible bodies through the
  arena backend behind a cargo feature; parity suite runs both ways. This is
  transition scaffolding, not an end-state decision — default flips when the
  parity suite says so.
- **P4 — IR-carrying kernels + arena splicing**: `HasIr`-style fragment
  accessor emitted by the macro; splice/substitute APIs on `ExprArena`. This is
  what shrinks `is_jit_eligible`'s ineligible set toward zero (named structs,
  manifold params, composition).
- **P5 — JIT at the root**: glyph bake through the fused arena
  (`CachedGlyph::new`), then `Lattice::collapse` JIT fast path → frame as one
  bound kernel (M4 in docs/designs/KERNELS_AND_LATTICES.md). Interpreter/
  combinator path remains the fallback until goldens pass.
- **P6 — deprecate and delete** the combinator emitter once P2–P5 cover the
  surface (parity suite + goldens green on arena-only).

**Demoted:** native multi-domain stamping in `kernel!` (`Field | Jet2 -> Field`
syntax). It would invest in the dying emitter; the fonts' `macro_rules!`
stamping is acceptable transitional scaffolding. Revisit only if the transition
drags.

## Constraints and non-goals

- No per-leaf JIT-at-construction for the thousands of curve-kernel instances:
  compile cost is µs-scale (see P0 results), leaves are bake-time-only, and
  P4/P5 fuse at roots instead.
- A general jet-in/jet-out ABI is not needed: consumers seed screen-space at
  the root (`Antialiased`), so derivative kernels are single-output Field
  kernels containing `Dwrt`. Deferred indefinitely.
- Suckless: pixelflow-ir gets no pixelflow-search dependency; the runtime
  `lower_dwrt` pass is ir-local, while the e-graph `ChainRule` remains the
  compile-time optimizer of the same algebra.
