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
