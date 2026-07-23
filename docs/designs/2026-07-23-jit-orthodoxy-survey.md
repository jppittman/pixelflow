# Is the PixelFlow JIT orthodox? A survey against V8, LuaJIT, HotSpot, and Halide-class compilers

**Date:** 2026-07-23
**Question:** the kernel-unification work (P2–P4) built a JIT that compiles
expression arenas at kernel-construction time, with an e-graph optimizer at
macro expansion and a symbolic-differentiation lowering tier at load. Is this
architecture orthodox, or have we gone off the rails?

**Answer: it is orthodox — but for the peer group we actually belong to.**
Measured against dynamic-language VMs (V8, LuaJIT, HotSpot) we look exotic,
because we omit their entire speculation apparatus. Measured against
data-parallel kernel DSLs (Halide, XLA, ISPC, GPU shader compilers) — which is
what PixelFlow is — we are textbook. The omissions are not gaps; they are the
removal of machinery whose only purpose is recovering, at runtime, static
knowledge those VMs lack and we have by construction.

## What the dynamic-language VMs do, and why we don't

| Mechanism | V8 / HotSpot / LuaJIT | PixelFlow | Why the difference is sound |
|---|---|---|---|
| **Tiering by heat** | Interpreter → baseline → optimizing (V8: Ignition→Sparkplug→Maglev→TurboFan; HotSpot: interp→C1→C2), promotion driven by profiling counters | Compile once at kernel construction; execute millions of times | Tiering exists because compilation must pay for itself against unknown execution counts. Our execution count is structurally known: a kernel is built at load/composition time and then evaluated per pixel per frame. The break-even question answers itself. |
| **Speculative optimization + deoptimization** | Type feedback, inline caches, guard checks, deopt bailouts to the interpreter | None. Types are static (`f32` lanes); the arena is fully typed before codegen | Speculation recovers static types a dynamic language erased. Our types were never erased — "types are the AST." There is nothing to speculate about, hence nothing to deoptimize. Our analogue of deopt is *static* and *loud*: unsupported ops refuse to compile and fall back a tier (combinator / interpreter), never silently. |
| **Trace recording** (LuaJIT) | Record hot loop paths through the interpreter, specialize on observed types/branches, guard + side-exit | The kernel *is* the trace: straight-line SIMD with `Select` instead of branches | Tracing discovers the hot straight-line path empirically. Ours is known syntactically — pull-based rendering means the pixel loop is the only loop, and branchless `Select` means there are no side exits to guard. |
| **On-stack replacement** | Swap running frames to optimized code mid-loop | N/A | Kernels are straight-line; there is no long-running loop frame to replace. The loop lives in the scanline/collapse driver, which is itself emitted code (P5). |
| **Code cache** | All of them cache and share compiled code aggressively | **Missing — a real orthodoxy gap** | Repeated builder-closure calls with identical (arena, params) re-lower and re-JIT identical code. Tracked as follow-up work; keyed on (arena bytes, root, substituted params) → shared `ExecutableCode`. Note P0's finding applies: codegen dominates compile cost, so a *result* cache is the right lever where buffer pooling was not. |

Where we *do* match the VM world is backend discipline: linear-scan register
allocation (Maglev and LuaJIT's allocators are the same family), µs-scale
compile times (P0 measured ~5µs for small kernels — LuaJIT's ballpark, orders
of magnitude below TurboFan/C2), and correct executable-memory hygiene
(W^X, `MAP_JIT` + `pthread_jit_write_protect_np`, icache invalidation).

## The actual peer group: staged kernel DSLs

The compile-once-execute-many, no-profiling architecture is *the* orthodox
shape for data-parallel kernel compilers:

- **Halide** (the project's explicit reference point): build an expression IR
  from an embedded DSL, optimize algebraically via term rewriting, compile at
  pipeline-construction time, run many times. Our e-graph is the modern form
  of that rewriting layer — equality saturation (egg lineage) is current
  state of the art in exactly this space, and "no pass ordering" is its
  defining property (which is why P2's staged-saturation detour was wrong and
  was reverted).
- **XLA / TorchInductor**: trace/capture a graph, algebraic + fusion
  optimization, one ahead-of-runtime compile per graph shape, cached
  thereafter. Our macro-expansion optimization with runtime param baking is
  the same split, moved earlier because our "graph capture" is a proc macro.
- **GPU shader/driver JITs**: shaders are compiled at pipeline-state creation
  (load time), specialized to bound state, executed per fragment. `kernel!`'s
  "construct at load time, render forever" law is this exact contract.
- **Multi-stage programming** (MetaOCaml/LMS lineage): statically-typed code
  templates specialized with runtime values, then compiled. Our builder
  closures — param substitution + `HasIr` fragment splicing into a template
  arena, then one codegen — are staging, with the arena as the code type.

Within that peer group our two-tier budget story (generous e-graph at
expansion; bounded/deterministic lowering at load; interpreter as reference
semantics) is a conventional AOT/JIT split keyed on *when code is known*
rather than *how hot it is* — appropriate because kernels are born at load
and composition time, not discovered hot mid-frame.

## The Halide comparison, specifically

Halide is the project's own stated reference ("writing it should look like
Halide"), so this is the comparison that decides orthodoxy. Concept by
concept:

| Halide | PixelFlow | Notes |
|---|---|---|
| `Func`: a pure function from coordinates to values, defined over an infinite grid | `Manifold`: pull-based, "pixels are sampled, not pushed" | Same founding idea. Halide realizes a `Func` over a requested region; we evaluate at requested coordinates. |
| `Expr`: scalar value IR, `select` instead of branches, no general control flow | `ExprArena`: identical shape (`Select`, no loops, no calls) | Both deliberately sub-Turing per point so the compiler owns all structure. |
| Staging: C++ operator overloading builds the IR at generator-run time | `kernel!` proc macro builds the arena at *Rust compile time*; params/fragments bind at construction | Same two-stage architecture, shifted one stage earlier. Our builder closures (param baking + `HasIr` splicing, then one codegen) are Halide's parameterized generators. |
| JIT mode: compile at pipeline-construction, `realize` many times | Kernels constructed at load time (project law), evaluated per pixel per frame | Same contract. Halide JIT compiles in ms–s (LLVM); ours in µs (direct emission) — see below. |
| Simplifier: a large handwritten term-rewriting system (its maintenance burden is well studied — Newcomb et al.) | E-graph equality saturation with provenance | We are on the *modern* side of this one: e-graphs are the literature's answer to exactly Halide's simplifier-maintenance problem. |
| Autoschedulers search schedule space with cost models; the learned-cost-model generation (Adams 2019) is notoriously hard to train well | E-graph extraction with a handwritten latency prior; NNUE learned model measured, lost, and shelved (opt-in) | Same architecture — search + cost model — and our 3-way bench conclusion (handwritten prior beats the learned model until proven otherwise) recapitulates the field's experience. |
| Gradient Halide (Li et al. 2018): derivatives as ordinary Halide exprs, differentiated at the IR level, optimized by the same compiler | `Dwrt` symbolic differentiation in the e-graph + `lower_dwrt`; derivatives are ordinary arena expressions | Philosophically identical (reverse-mode for training there, forward-mode for screen-space AA here — dictated by use, not architecture). |
| Bounds inference: interval analysis from consumer to producer | Declared static extents (`BufferDecl`) + clamped sampling | We sidestep general bounds inference; clamp-at-edge is Halide's `BoundaryConditions::repeat_edge` baked in as the only policy. Sound while lattices are the only bounded storage. |
| LLVM backend | Handwritten per-ISA emitters (LuaJIT-family) | The real tradeoff: Halide inherits LLVM's instruction selection and vector maturity, and pays ms–s compiles; we pay per-ISA emitter maintenance and get µs compiles, which P0 showed is the budget that makes load/composition-time specialization viable at all. |

**The one structural thing Halide has that we do not: the schedule
language.** Halide's thesis is algorithm/schedule separation — the same pure
`Func` can be tiled, vectorized, parallelized, computed-at or stored-at any
granularity, and the *scheduling* choices are where the performance lives.
PixelFlow has scheduling decisions, but they are baked policy rather than a
language:

- SIMD width = `Field` lanes (a fixed `vectorize`);
- scanline emission with variance-based hoisting = a fixed loop-invariant
  `compute_at` policy;
- the actor render pool = a fixed `parallel` over rows;
- and most importantly, **bake vs. fuse** — materialize a kernel into a
  lattice (`CachedGlyph`, collapse) or splice its IR into the consumer
  (`HasIr`) — *is* our `compute_root` vs. `inline` axis, chosen per kernel by
  the author.

For a terminal emulator — one workload class, one target family — baked
policy is defensible; Halide needed a schedule language because image
pipelines' optimal loop nests vary wildly. But the mapping is worth keeping
explicit, because if per-kernel scheduling ever becomes necessary (tile sizes
for large blurs, fusion granularity in lattice collapse), the orthodox move
is Halide's: keep the algebra pure and grow the bake/fuse choice into a small
scheduling vocabulary, rather than baking more policy into the emitters.

## Deliberate departures worth defending

1. **No jet (dual-number) runtime for derivatives.** Forward-mode AD via
   operator overloading is how the combinator backend works and how most AD
   libraries ship. The arena backend instead differentiates *symbolically*
   (e-graph `ChainRule` at expansion; `lower_dwrt` at load). This is the
   Halide/XLA-style choice: derivatives as ordinary expressions in the same
   IR, visible to the optimizer and register allocator, one scalar ABI. A jet
   ABI would triple register pressure and fork every backend. Orthodox for
   compilers, even though it is unorthodox for AD *libraries*.
2. **Fusion at roots, not per-leaf compilation.** P0 killed per-leaf JIT with
   data (~5µs floor vs ~1µs budget). Composition therefore splices IR
   fragments (`HasIr`) into one fused root before a single compile — which is
   Halide's fusion story and XLA's, not a VM inlining story. If per-leaf or
   per-frame compilation ever becomes real, the orthodox escape hatch is a
   copy-and-patch baseline tier (Xu & Kjølstad; Python 3.13's JIT), which P0
   already identified as the only path below the codegen floor.

## Verdict

Nothing here is off the rails. The one place we are genuinely unorthodox
relative to *every* mature JIT — dynamic or static — is the missing compile
cache, and it is tracked. The rest of the delta from V8/LuaJIT/HotSpot is the
principled removal of speculation machinery that exists to recover facts we
never lose, and the architecture we converged on instead is the standard one
for the Halide/XLA/shader-compiler family PixelFlow actually belongs to.
