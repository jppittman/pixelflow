# Loop-aware codegen: giving the register allocator the lattice

**Date:** 2026-09-01
**Status:** design; revised 2026-09-02, twice (see below); stage 0 landed (#1092), stage 0′ in progress
**Depends on:** the one-kernel-ABI refactor (`#1077` and its in-flight
follow-up), which retires `compile_arena_dag`/`compile_arena_dag_avx2`/
`compile_arena_dag_avx512`/`compile_dag_via_backend` and the `PerBatch`
marker, collapses nine `compile_*` entries into one `compile`, and drops
`ExecutableCode`'s ABI type parameter. This plan assumes that has landed:
**the collapse kernel is the only ABI**, so "the compiler" below means the
one path from `ExprArena` to a loop-emitting kernel, not a per-batch path plus
a loop wrapper bolted onto it.
**Precedes:** step 4 of
[2026-09-01-register-allocation-escape-hatches.md](2026-09-01-register-allocation-escape-hatches.md)
("Show the allocator the loop"), which this plan expands into its own design.
Steps 1–3 of that plan (declare `fixed`, let the allocator name reload/temp
defs, add the register-class axis) are independent of this one and are not
re-litigated here.

## Revision 2026-09-02: one program, one e-graph, one allocator

**Status:** design revised (two passes on 2026-09-02); stage 0 landed as
#1092; stage 0′ in progress. Sections 4, 6 and 7 stand as written except
where marked. Sections 1, 2, 3 and 5 are superseded in the places called out
below; the original text is kept as the record of what was reconsidered.

### What the plan below got wrong, and why

It was written from where the code stands. LICM lives in codegen
(`plan_collapse_hoist`), so handing its three regions to the register
allocator looked like the next step. That continues the July direction and
contradicts two design documents this codebase already carries:

- [`docs/designs/lattice-scheduling-types.md`](../designs/lattice-scheduling-types.md)
  — *"CSE is factoring. Let-binding is factoring. Loop-invariant code motion
  is factoring."* The e-graph discovers the factorizations, variance is an
  e-class analysis, the **Lattice** maps a variance class to a scope, and
  extraction is scope-weighted. Codegen decides only *how* each factor is
  computed.
- [`docs/designs/REDUCTIONS_AND_FOLDS.md`](../designs/REDUCTIONS_AND_FOLDS.md)
  — `Reduce` is an e-node; hoisting out of a sum, fusing two sums and
  `Σ_i c = N·c` are rewrites whose side condition is the variance analysis.

What shipped instead: `plan_collapse_hoist` is a post-extraction pass over
the flat schedule — the "LICM as an explicit AST pass" that
`BRAINSTORM_VARIANCE_EGRAPH.md` attributes to Elliott's Pan and sets out to
improve on. Extraction has no scope weighting; `DepsAnalysis` fed NNUE
features and tests only. And `optimize_runtime_arena` bailed out of any
arena containing a `Reduce`, so reductions never reached the e-graph.

The plan below also made two further errors that this revision corrects:

1. **It kept the lattice's extents out of the compiler** (§1–§2) to preserve
   cache hits across a window resize. Decision: that property is not
   wanted. *Resizing is recompilation.* The extents are compile-time, part
   of every key, and the compiler is entitled to specialize on them — which
   is the whole point of "the whole program is lattice-aware."
2. **It treated reduction binders as a second loop nest, out of scope**
   (§6). They are not a second anything. `pixelflow-ir/src/variance.rs`
   already says so: *"Scopes are binders. Coordinates are bound by the
   lattice nest, reduction indices by `Kernel::over` — the same kind of
   thing, so they share the bitset."* One bitset, one kind of thing.

### The denotation: one unrolled program

A kernel together with its lattice is a **closed, finite, loop-free
expression**. Every `over` has a static extent; by decision (1) every
coordinate axis does too. Its only runtime inputs are the origin (a parameter
in effect) and the contents of bound buffers. That expression is the thing the
e-graph reasons about and the register allocator allocates. There is no
second IR and no scaffold outside the analysis.

A **binder** is the one construct that introduces repetition: it binds an
index over `0..extent` and either **folds** the body with a monoid
(`Reduce`, `Kernel::over`) or **stores** it per index (the lattice axes).
Fold-versus-store is a property of the binder; everything else about binders
is shared:

- **Scope.** `scope(e)` is the shallowest binder level enclosing `deps(e)`.
  That one rule is LICM out of the X loop, hoisting out of `Σ_i`, and
  constant folding.
- **Schedule.** N binders give N nested regions. The "three regions" of §3
  are N = 2 with Z and W degenerate — a test case, never a special case.
- **Cost.** With every extent static, extraction's objective is not an
  ordinal weight but the exact instruction count of the unrolled program:
  `Σ cost(node) · Π extent(binders enclosing node)`. The ILP in
  `lattice-scheduling-types.md` is literally computable.
- **Liveness.** Belady over the unrolled sequence *is* liveness across back
  edges; it is computed once per body with the period known.
- **Interchange.** Binder order is a scheduling choice. `Σ_i` inside or
  outside the X loop is `REDUCTIONS_AND_FOLDS.md`'s `Σ_i Σ_j = Σ_j Σ_i`,
  legal exactly when the variance side condition holds.

### Vocabulary: distribute / unroll and factor / loop

Two words for the two forms a binder can take, chosen so the algebra and the
code agree:

- **Distribute (unroll).** `⊕_{i<N} body(i) = body(0) ⊕ body(1) ⊕ … ⊕ body(N−1)`:
  the binder is pushed through to N explicit copies. This is the expanded
  polynomial. The program is loop-free and every copy is an ordinary node
  the e-graph can CSE, fold, and reassociate across.
- **Factor (loop).** The inverse: the copies are recognised as one body
  applied at N indices and written once. This is the factored polynomial;
  a loop is a run-length-encoded unroll. Every analysis is *defined* on the
  distributed form and *computed* on the factored one, and the factored form
  is what large extents must take — a 1920×1080 frame is never
  distributed.

The choice between them is a **per-binder codegen policy made on extent**:
small, distribute; large, factor. It is not a property of fold-versus-store.
`expand_reduce`, now run ahead of saturation (#1092), is this policy at the
"small" end for fold binders. Applied to the coordinate axes it is the same
policy: a `Lattice::point` kernel is fully distributed (no loop at all), a
scanline distributes Y and factors X, a frame factors both.

`unroll_reduce` already contains the factoring theorem in miniature: a body
subtree that does not depend on the index is *not copied* — the N terms
share it — which is `⊕_i (f(i)·c) = c·⊕_i f(i)` obtained by declining to
distribute `c`. Variance is the proof obligation for that step, and it is
the same obligation everywhere in this plan.

### Representing a factored binder in the e-graph

The only thing that needs care. A large binder cannot be distributed, so it
must be an e-node — `Reduce` with the body as a shared child, and the lattice
axes as the outermost such binders — and rewriting *under* it must be sound.
It is, given one discipline: **extraction places every value at the scope its
variance demands.** A body term depending on slot 4 can then never be hoisted
past the binder of slot 4, and two sibling binders sharing a slot may share
body e-classes freely because each still evaluates them inside its own
scope. That discipline is the scope rule itself. The "binder-aware rules"
the old bail-out asked for were never the missing piece; the scope rule was.

**Distribution is a rewrite, and it is the same rewrite that proves Gauss.**
A fold is defined by two equations,

```
⊕_{i<0}   body(i) = e                                   (identity)
⊕_{i<N+1} body(i) = (⊕_{i<N} body(i)) ⊕ body[i := N]    (peel)
```

and peel *is* the unroll: applied N times to a static extent it yields the
distributed form. It is also the proof of every closed form the e-graph
discovers — `Σ_{i<4} i²` came out as `Const(14)` in #1092 because four peels
and `ConstantFold` derive it. `Σ_{i<N} i = N(N−1)/2` and `Σ_i c = N·c` are
the *factored* statements of what peel-plus-fold computes, valid without N
steps: theorems the e-graph can prove for any concrete N, and lemmas it
should carry so it need not (rule provenance and hindsight labeling in
`pixelflow-search` are the machinery for noticing a derivation worth keeping).
Both the factored `Reduce` and its peeled expansion then sit in one e-class,
and **extraction chooses**: pick the peeled member and codegen emits copies;
pick the binder and codegen emits a loop. Unroll-or-not is the extractor
comparing `N·cost(body) + (N−1)·cost(⊕)` against loop overhead with N a
number it knows — not a codegen policy after all, which supersedes the
"per-binder codegen policy" wording above.

The one requirement is that `body[i := N]` be something the rewrite engine
can do. E-graphs cannot substitute natively (the classic binder problem), so
substitution is an explicit node with three rules:

```
Subst(a ⊕ b, i, k) = Subst(a, i, k) ⊕ Subst(b, i, k)     (distribute)
Subst(e, i, k)     = e            when i ∉ deps(e)        (factor)
Subst(Var i, i, k) = Const k
```

The second rule is `unroll_reduce`'s sharing of index-invariant subtrees,
stated as a rewrite with variance as its side condition. Peel on a 1920-wide
axis is 1920 saturation steps, so the rule needs an extent-gated mask (the
guided-saturation direction already on file); and a lattice axis is a fold
with `Store` as its monoid, so its peel is a tile split and a point kernel is
`Store_{i<1} body = body[i := 0]`, the same rule at N = 1. This answers
stage 1's IR question: a `Binder` node with `Fold(op) | Store`, a `Subst`
node, three rules, and `expand_reduce` becomes deletable.

### The shape type

`LoopShape` as landed in #1092 (X/Y loop mask) is the *wrong denotation*
under decision (1) and only the coordinate half under decision (2). It is
replaced by the **coordinate extents**, `[u32; 4]` — `pixelflow_core::Lattice`
with `origin` erased and nothing else erased. The compiler assembles the full
binder nest from two sources that already exist: those extents from the
caller, and each `Reduce` node's `(slot, extent)` from the arena. `Variance`
indexes the nest uniformly. Per-batch `eval` paths (the `kernel_jit!`
wrapper, `BilinearSampler`) have no lattice: one batch of caller-supplied
points, extent `[1, 1, 1, 1]`, fully distributed.

Origin stays runtime. Every production bake fixes it (texel centers), but
`Lattice::point(x, y, …)` in the antialiasing tests varies it per call, and
nothing is gained by folding it: it is an additive offset the body reads
once.

Cost of decision (1), measured on the host with `pixelflow-pipeline`'s
`bench_jit_compile_cost` (x86-64, median of 117 distinct kernels, one full
`jit_cache::compile` miss = key + lock + optimize + emit + insert):

| nodes | code bytes | miss | emit only |
|---|---|---|---|
| 8 | 382 | 1.45 ms | 13 µs |
| 32 | 883 | 6.2 ms | 25 µs |
| 128 | 2 725 | 23 ms | 115 µs |
| 512 | 10 150 | 39 ms | 1.5 ms |

So a distinct kernel costs tens of milliseconds, and **saturation is that
cost — emit is one to two orders of magnitude below it.** The terminal's
glyph bakes are per glyph size and do not change on resize; its cell-grid
programs are per frame size and do, so an interactive resize pays one such
miss per size step per program. Two things follow. First, this is the price
of decision (1) and it is accepted. Second, it is mostly avoidable without
giving the decision back: saturation does not depend on the extents — only
extraction's weights do — so the e-graph can be **saturated once per
structure and extracted once per extent**. `optimize_runtime_arena`'s cache
should key the saturated e-graph on structure and the extracted arena on
structure plus extents. That is a stage-1 detail, recorded here so the
39 ms is not mistaken for the cost of the design.

### Re-staged migration

**Stage 0 — landed (#1092).** The shape reaches `jit_cache::compile` and
`optimize_runtime_arena` as a key component; `JitManifold` carries it;
`Reduce` is distributed ahead of saturation and the bail-out is gone.

**Stage 0′ — the shape is the extents.** `LoopShape` → coordinate extents;
`Lattice::bake` passes `self.extent`; the cell-grid programs pass their
plane size; per-batch paths pass `[1, 1, 1, 1]`. `JitManifold::call_collapse`
asserts the tile lies within the extents. The resize-sharing test from
#1092 is deleted (it pinned the property decision (1) rejects). *Invariant:*
emitted code byte-identical (`emit::compile` untouched); `reduction_binder.rs`
and `kernel_bake.rs` pass.

**Stage 1 — scope-weighted extraction emits the binder-nest schedule.**
*Landed 2026-09-02 (#1097), with the measurement this plan's methodology
note demands.* Extraction multiplies each node's cost by
`LatticeShape::evals` of its chosen form's variance. Measured on a full
1920×1080 collapse, best of seven, against the same kernels extracted at
`POINT` (which weights everything by one, i.e. the old behavior):

| hoistable terms | fused | scope-weighted | |
|---|---|---|---|
| 1 | 1.522 ms | 1.499 ms | −1.5% |
| 4 | 1.481 ms | 1.471 ms | −0.7% |
| 8 | 1.767 ms | 1.568 ms | **−11.3%** |

The win grows with the number of terms that can leave the loop, which is
what the model predicts. It is bought with accuracy, deliberately and within
the contract `CLAUDE.md` sets out: the optimizer un-fuses an FMA whose
multiplier is loop-invariant, trading one rounding for two per sample in
exchange for hoisting the multiply, so a glyph bake sits ~8.5e-6 from the
interpreter where it used to sit ~1.1e-6. Range is untouched; precision is
the tunable.

Two consequences worth stating plainly. **Two lattices are now two
compilations**, and need not agree bit-for-bit — a test comparing a point
bake against a frame bake at f32 tolerance was pinning a promise the
compiler no longer makes, and now compares against the interpreter instead.
And **trip-count weighting multiplies cost-model error by the trip count**:
the FMA decision turns on the table pricing `MulAdd` at 5 cycles against
`Add`'s 4, so a one-cycle error that used to move one instruction now
decides the shape of a whole loop body. That is an argument for measuring
the table against loop bodies, not only against straight-line kernels — the
`latency_prior` table is a *latency* prior and a pixel loop is
throughput-bound. Not acted on here.

Original text follows.

Extraction cost becomes `Σ cost(node) · Π extent(enclosing binders)`, with
scopes from the node's *chosen* variance (`Extraction::chosen_variance`
exists) against the nest. Output is the N-region schedule; codegen's
`compile_via_backend` consumes it and `plan_collapse_hoist` /
`schedule_variance` are deleted. The gather policy ("memory reads stay where
the body had them") moves with the partition as a stated rule. *Invariant:*
for every kernel in `collapse_loop.rs`/`spill_pressure.rs`, the regions
extraction emits equal the regions `plan_collapse_hoist` computed on the same
extracted form (golden-partition test, run before the deletion); differential
execution across all four backends.

**Stage 2 — the allocator sees the nest.** Former stage 1: allocation over
N regions, policy still "every loop-carried value gets a slot".

**Stage 3 — loop-carried values compete for registers.** Former stage 2.

**Stage 4 — distribute-or-factor per binder.** Former stage 3, generalised:
codegen chooses per binder by extent, so a point kernel has no loop, a
scanline has one, and a small `over` inside a frame is copies while a large
one is a loop. *Invariant:* differential against the always-factored kernel
at every extent actually used.

Stage 0′ has no dependency on the escape-hatches steps. Stage 1 should land
after escape-hatches step 1 (fixed registers declared) so the
golden-partition test runs against a stable allocator; stage 3 still wants
steps 1–2 first for the reason §6 gives.

### Follow-ups this revision surfaces, not taken here

- `Dwrt(Reduce(…))`: `legalize` runs `lower_dwrt` before `expand_reduce` and
  `lower_dwrt` refuses a `Reduce`, so a derivative of a sum is a compile
  error today. Distributing first makes it a sum of derivatives — exact and
  free. The optimizer mirrors `legalize`'s order for now so the two paths
  stay uniform; reordering both is a separate, tested change.
- `push_reduce` encodes the combiner `OpKind` and the bound index as
  `Const(f32)` children (CLAUDE.md names this). A `Binder { slot, extent,
  Fold(op) | Store }` node is the type that should hold it, and is the same
  type the lattice axes need; that is stage 1's IR question.
- `Kernel::over` extents are static today. If a runtime-extent binder is
  ever wanted, it is the one binder that must always factor, and the cost
  model needs a bound for it; nothing in this plan forecloses that.

## Verifying the premise

Two claims were made about the current state of the compiler. Both check out
against the code as it stands on `main` (`abb0d74c`).

**`EmitCtx` is `{ max_regs: Option<u8> }` and nothing else**
(`pixelflow-codegen/src/emit/mod.rs:281-303`):

```rust
/// Emission context with register budget for ML training.
#[derive(Clone, Debug, Default)]
pub struct EmitCtx {
    pub max_regs: Option<u8>,
}
```

The doc comment says what it is for: training data for the cost model, not
compiling real kernels differently. `max_regs` only ever *shrinks* a backend's
existing pool (`RegisterFile::capped`) — it cannot express anything about the
program being compiled, only about how ungenerous to be to the allocator.

**Every non-default use of it is a test.** `with_max_regs` appears at three
call sites in the whole tree:

| site | context |
|---|---|
| `pixelflow-codegen/src/emit/mod.rs:2824` | `#[test] fn arena_compile_with_spills()` |
| `pixelflow-codegen/src/emit/mod.rs:4085` | `#[test] fn avx512_spills_to_real_frame()` |
| `pixelflow-codegen/tests/muladd_rounding.rs:166` | integration test, forcing a decomposed-MulAdd shape under register pressure |

Every other call anywhere in the workspace is `EmitCtx::default()` — I counted
seventeen in `pixelflow-codegen/src/emit/mod.rs` alone, all either production
entry points (`compile_collapse`, the `Native::new` calls inside
`compile_collapse_via_backend`) or tests that don't care about register
pressure. **Confirmed as stated**: the compiler's only configuration surface
today exists to make register allocation fail on purpose, for a test to
observe.

Meanwhile the collapse loop is real, already emitted, and already reasoning
about invariance it cannot act on. `emit_collapse_loop`
(`pixelflow-codegen/src/emit/mod.rs:486-556`) stores all four coordinates to
stack slots once per call (`slot_store`, lines 500-503) and reloads them from
those slots *at the top of the row loop and again at the top of the batch
loop* (lines 515-516, 524-525) — every batch iteration, unconditionally, even
though `SCAFFOLD_ACC`/`SCAFFOLD_SCRATCH` (`Reg(0)`/`Reg(1)`, lines 632-633) are
literally the X/Y input registers reused, and the comment at line 628 admits
why: *"every coordinate is reloaded from its slot at the top of each
iteration, so the scaffold is free to clobber them once the body has run."*
The reload exists to recover from a clobber the scaffold itself performs, and
nothing tracks that X (say) could instead just be incremented in place, kept
resident, because `emit_dag_body`'s allocator has no idea a loop surrounds it.

`plan_collapse_hoist` (`mod.rs:1110-1230ish`) runs twice —
`compile_collapse_via_backend` calls it once at `XY_SCOPE` (frame-invariant,
computed once per call) and once at `X_SCOPE` (row-invariant, computed once
per row) — which is real loop-invariant code motion over a two-level loop
nest whose trip counts (`rows`, `groups`) are runtime arguments the compiler
never sees. Every hoisted root is parked in a memory slot
(`hoist_slot`/`frame_map`/`row_map`, `mod.rs:1927-1958`) unconditionally: there
is no path where a frame-invariant value stays in a register across the row
and batch loops, because a register that survives a loop iteration is exactly
the fact the allocator cannot express — `emit_dag_body` allocates registers
for one straight-line body and forgets its answer between calls.

The premise holds. The rest of this plan addresses it.

## 1. What a lattice is, and whether it's *the* `Lattice`

> **Superseded 2026-09-02** in one respect: the lossy projection to a loop mask
> is withdrawn. The compile-time lattice is the coordinate extents — see "The
> shape type" in the revision above. The rest of the section (which object
> this is, where it lives) stands.

Denotationally, a lattice (the compile-time object this plan introduces, not
literally always `pixelflow_core::lattice::Lattice`) is:

```
extent  : [Bound; 4]     -- per-axis trip count, X..W, where Bound is a
                            statically-known u32 or "unknown until call time"
stride  : implicit        -- X is lane-contiguous (SIMD width), Y/Z/W step by 1
order   : X innermost, then Y, then Z, then W  -- fixed, matches
                            pixelflow_core::Lattice::coord's row-major choice
```

`pixelflow_core::lattice::Lattice` (`pixelflow-core/src/lattice/mod.rs:211+`)
is **already** exactly this shape:

```rust
pub struct Lattice {
    pub extent: [u32; 4],
    pub origin: [f32; 4],
}
```

with `loop_mask()` (`mod.rs:280-289`) computing "which axes have extent > 1" —
the bitmask a codegen-side variance/liveness pass would want directly. And
`Lattice::bake` (`mod.rs:373+`) is the one call site in the whole workspace
that holds a concrete `Lattice` *and* triggers compilation
(`pixelflow_codegen::jit_cache::compile(arena, root)`, line ~389) in the same
function. It just doesn't pass the two along together — `compile` sees only
`(arena, root)`; `bake` separately computes `full_groups`/`ey`/`tail` from
`self.extent` and hands them to `call_collapse` as **runtime** `TileSlice`
arguments, after the kernel already exists.

So: **is the compile-time lattice this plan wants the same object as
`pixelflow_core::Lattice`, or a different thing wearing the same name?**

Honest answer: **the same law, a narrower type.** `pixelflow_core::Lattice`
is a full runtime value — concrete `origin`, concrete `extent`, constructed
fresh per `bake` call (every window resize is a new `Lattice`). Feeding *that*
verbatim into codegen as a specialization key would mean every distinct
extent mints a distinct compiled kernel, which directly attacks the property
`jit_cache.rs`'s own module doc calls out as the point of caching: *"every
call of an N-param builder with the same arguments (window resizes) ... hit[s]
[the cache]."* A terminal resize sweeping through a few hundred (cols, rows)
pairs would mint a few hundred kernels for what is today one.

What the allocator actually needs is not `extent: [u32; 4]` but the much
coarser fact `pixelflow_core::Lattice::loop_mask()` already computes: **which
axes are loop axes at all**, plus one more bit codegen alone can supply —
whether the innermost (X) extent is known to be an exact multiple of the
lane width, because that changes whether a scalar-tail path needs to exist
inside the specialized kernel or can stay a separate, unspecialized fallback.
Call this the **codegen lattice**:

```rust
struct LoopShape {
    /// Which of X/Y/Z/W are loop axes (extent > 1) vs. call-time constants.
    /// Same bit order as pixelflow_core::Lattice::loop_mask().
    loop_mask: u8,
    /// Whether the caller promises the X extent is an exact multiple of the
    /// lane width (no scalar tail inside this compiled body).
    exact_lanes: bool,
}
```

This is `pixelflow_core::Lattice` with `origin` erased and `extent` collapsed
to two bits — a deliberately lossy projection of the *same* representable
object, not an unrelated concept that happens to share a word. The
`index(collapse(f)) = f` law lives on the `pixelflow_core::Lattice` side and
is untouched by any of this: `LoopShape` never claims to *be* the domain,
only to describe which of its axes the emitted machine code specializes on.
Concretely, `Lattice::bake` is the place `LoopShape::from(self)` would be
computed and threaded into `jit_cache::compile`, one line away from where it
already computes `self.loop_mask()`-equivalent facts (`full_groups`, `ey`,
`tail`) for the call-time path.

A kernel bound to no buffers and reduction-free is legal to specialize on
`loop_mask` alone; a kernel that reduces over `Kernel::over` binders (variance
bits 4-8, `pixelflow-ir/src/variance.rs:9-13`) has more loop structure than
the four coordinate axes, and this plan is scoped to the coordinate lattice —
extending it to binder loops is future work, flagged in Risks.

## 2. What moves to compile time, what stays dynamic

> **Superseded 2026-09-02:** the trip counts move to compile time as well;
> the "specialization-vs-cache-pressure trade" paragraph below argued for the
> property *resizing keeps its kernel*, which was rejected: resizing is
> recompilation. `exact_lanes` dissolves — with the extent known, the tail is
> known. Origin, `row_skip_bytes` and buffer pointers stay dynamic as stated.

**Moves to compile time (part of the cache key, part of what the allocator
sees as the loop shape):**

- `loop_mask`: which of X/Y/Z/W are genuine loop axes for this kernel. A
  kernel called only at `Lattice::scanline`/`Lattice::point` shapes never
  needs a Y-loop's coordinate bookkeeping at all — today it gets one anyway
  (`rows=1`, one full iteration of dead loop overhead) because
  `emit_collapse_loop` always emits both loop levels.
- `exact_lanes`: whether the specialized body may assume no scalar tail. Only
  meaningful together with `loop_mask & X != 0`.
- Per-axis **register residency intent**: given `loop_mask`, the allocator can
  decide "X is an induction value with N live registers across the batch
  loop's back edge" as a compile-time fact about the schedule, not a runtime
  unknown.

**Stays dynamic (part of the ABI, read from `TileSlice`/`Point4` at call
time, exactly as today):**

- The actual trip counts (`groups`, `rows`) and the actual `origin` (`x0`,
  `y0`, `z`, `w`). These are genuinely per-call — the whole reason
  `Lattice::bake` calls the same compiled kernel multiple times per bake (a
  full-groups block, then a scalar-tail block) with different `Point4`
  arguments (`pixelflow-core/src/lattice/mod.rs:428-462`).
- `row_skip_bytes` — output layout is a caller concern, unrelated to the
  kernel's own loop structure.
- The buffer base pointers passed via `ctx`.

**The specialization-vs-cache-pressure trade, stated as a number.** Today
`jit_cache::canonical_key` (`pixelflow-codegen/src/jit_cache.rs:88+`) keys
purely on arena structure. Adding `loop_mask`/`exact_lanes` to the key
multiplies the entry count by at most the number of distinct `(loop_mask,
exact_lanes)` pairs actually requested — bounded (5 useful `loop_mask` values:
point/scanline/frame-fixed-Z/tensor-index/full-frame, × 2 for
`exact_lanes`), **not** by the number of distinct extents, because extents
are deliberately excluded from the key. This is why the projection in §1 has
to be lossy: the whole value of the cache is that a terminal resizing its
window every frame keeps hitting the same compiled kernel, and that property
survives specializing on `loop_mask` (a terminal's glyph-grid kernels are
always called at `Lattice::frame`-shaped domains — `loop_mask` never changes
across a resize) but would not survive specializing on `extent` itself.

The one genuinely open question this plan does not resolve: whether
`exact_lanes` is worth the second cache slot per kernel, given that most
callers (terminal cell grids, arbitrary-width glyphs) *do* have a scalar
tail some of the time. Recommendation: ship `loop_mask`-only specialization
first (§5, stage 2) and measure before adding `exact_lanes` — it is the
smaller, second win, not a prerequisite.

## 3. How the register allocator changes

> **Revised 2026-09-02:** the *residency* half of this section stands; the
> *partition* half (which values leave the loop, and for which scope) moves to
> extraction — see the revision above. Kept as written for the record.

### The conflation this fixes

`RegisterAllocator::allocate` (`pixelflow-codegen/src/emit/regalloc.rs:410`)
takes `Vec<Def>` — one flat, loop-free, SSA schedule — and so does the
e-graph extractor. That single type currently serves two different
requirements at once, and the escape-hatches plan already names the
conflict directly (`docs/plans/2026-09-01-register-allocation-escape-hatches.md:78-85`):

> the e-graph needs acyclicity while the allocator needs liveness over back
> edges — today both are conflated in one `Vec<Def>` schedule.

The acyclicity requirement is real and load-bearing on the e-graph side —
`pixelflow-search/src/egraph/extract.rs` has an entire acyclicity-repair pass
(`repair_choices_well_founded`, `backfill_well_founded`) whose job is to
guarantee extraction produces a DAG, because saturation can otherwise merge
e-classes into cycles. `arena_to_schedule` inherits that guarantee for free:
the arena is append-only, so walking it in id order is already a topological
sort (`compile_arena_dag`'s old doc comment said as much: "the arena IS the
linearized schedule, so linearization is free").

The allocator's liveness analysis has no comparable need for acyclicity — a
value live across a loop back edge is not a cycle in the dependency sense,
it is an ordinary interval that happens to span multiple passes through the
same instructions. Belady eviction (what `LinearScan` already implements,
`regalloc.rs:442+`) is defined in terms of "next use," and next-use over a
loop with a known (or boundedly-unknown) trip count is a well-defined
question — it's `min(next use inside this iteration, next use in iteration
i+1 if the loop repeats)`.

### What changes

**The allocator's input stops being one `Vec<Def>` and becomes a schedule
with declared loop structure.** Concretely, three regions instead of one:

```
prologue : Vec<Def>   -- runs once per call (today: frame_hoist)
row      : Vec<Def>   -- runs once per row, with a back edge to itself
                          (today: row_hoist, folded into the loop by
                          emit_collapse_loop's scaffold, not the allocator)
batch    : Vec<Def>   -- runs once per batch, with a back edge to itself,
                          nested inside `row`'s back edge
```

This is not a new concept — `compile_collapse_via_backend` already computes
exactly these three regions today (`frame_prologue`/`row_prologue`/
`body_schedule` from the two `plan_collapse_hoist` calls,
`mod.rs:1867-1886`). What's missing is that the allocator never sees the
nesting: it allocates each region's `Vec<Def>` independently
(`emit_dag_body_hoisted` called three separate times,
`mod.rs:1957-1988`), and the *loop-carried* liveness — "this value computed
in the prologue must still be findable when the batch region runs, possibly
thousands of iterations later" — is handled entirely outside allocation, by
`HoistCtx::Prologue`/`HoistCtx::Body` unconditionally forcing every hoisted
value to a fixed memory slot (`hoist_slot`, `mod.rs:1938-1939`) rather than
letting it compete for a register.

With the allocator shown the loop nest, this becomes an ordinary liveness
fact: a value defined in `prologue` and used in `batch` has a live range that
crosses both back edges, exactly like a value defined before a `for` loop and
used inside it in any register-allocated language. Whether it wins a
register or gets spilled is then a pressure decision, not a foregone
conclusion — decided by the same Belady "farthest next use" rule the
allocator already uses within one region, generalized so "next use" can name
a program point in a different region across a back edge.

**X-as-induction, Y-as-loop-state, Z/W-as-invariant become allocation facts
instead of ABI positions.** Today `coord_reg` (`mod.rs:624-627`) hardcodes
"X/Y/Z/W live in registers 0-3, always, for the scaffold's own reload/store
choreography" — a fact about the *calling convention*, unrelated to whether
the compiled body's own dataflow wants X resident across the batch loop. Once
the allocator sees the batch loop's back edge, X's residency is a decision it
makes the same way it decides any other loop-carried value's residency: X is
read at the top of the batch body (probably every iteration, since it's the
per-pixel coordinate), so it wants a register with a live range spanning the
whole batch loop, refreshed by an add-immediate at the back edge instead of
reloaded from a stack slot. That is exactly what an induction variable is —
the allocator gets to *notice* X is one, rather than being told where X lives
by the ABI and reloading it every iteration regardless.

**Scope note.** This plan does not propose rewriting `LinearScan` from
scratch — Belady eviction generalizes; the change is to what a "program
point" means (a `(region, index)` pair instead of a flat index) and to
`last_use` computation walking across back edges with a bounded number of
passes (loops don't need a fixed-point liveness solver here: two passes over
each loop region — one to see what's used before the value's next
prologue-side definition, one to confirm nothing inside the loop redefines it
sooner — suffice, because the schedule is still acyclic *within* each region;
only the three regions relate to each other with back edges, and there are
only ever two of those, always in the same shape).

## 4. What this deletes

Concrete, with the current file:line as reference (subject to renumbering
after the one-kernel-ABI refactor lands, since line numbers will move):

- **The unconditional per-iteration coordinate reload.** The `slot_load` calls
  at `mod.rs:515-516` (top of row loop) and `mod.rs:524-525` (top of batch
  loop) for coordinates that stay allocator-resident. A coordinate that
  degenerates to a loop-invariant constant (Z, W, and Y whenever
  `loop_mask & Y == 0`) needs neither a slot nor a reload at all.
- **`COORD_SLOTS` / `SLOT_X` / `SLOT_Y` / `SLOT_ROW_START_X`** (`mod.rs:610-616`)
  shrink to whatever the *actual* loop-carried coordinates require for a
  given `loop_mask` — a `Lattice::scanline` kernel (`loop_mask` = X only)
  needs zero coordinate slots, since Y/Z/W are frame-hoisted constants and X
  is an allocator-resident induction register with no cross-call state to
  restore.
- **`INPUT_REGS`/`coord_reg`'s fixed mapping** (`mod.rs:624-627`,
  `RegisterFile::inputs`, `regalloc.rs:144+`) stops being "X/Y/Z/W always
  precolored to registers 0-3 for the scaffold's benefit." `inputs` remains
  the *ABI* fact (where the caller hands values in) but is no longer also
  the *loop-body* fact (where the value lives for the rest of its life) —
  those separate the moment residency is decided by liveness instead of by
  where the calling convention happened to put the argument.
- **`SCAFFOLD_ACC`/`SCAFFOLD_SCRATCH`** (`mod.rs:632-633`) as hand-picked
  registers disappear the same way class D of the escape-hatches plan
  describes: the X-increment and Y-increment become ordinary allocated
  operations the allocator places, not a hardcoded reuse of `Reg(0)`/`Reg(1)`
  justified by a comment.
- **Unconditional memory for hoisted values.** `hoist_slot`/`frame_map`/
  `row_map` (`mod.rs:1927-1958`) stop being the only outcome of hoisting — a
  frame-invariant value competes for a register across the whole call, same
  as any other loop-carried value, and only spills when pressure says so.
  `HoistCtx::Prologue`/`HoistCtx::Body`'s job narrows from "always place in
  this memory slot" to "this value's live range crosses these back edges,"
  which is the allocator's normal vocabulary, not a special case.
- **Possibly parts of `CollapseBody`'s `frame_hoist`/`row_hoist` split as
  distinct byte buffers** (`mod.rs:600-608`) — if the allocator schedules all
  three regions together, the scaffold may go back to being "emit three
  labeled regions with two back edges" rather than "splice three
  independently-emitted `Vec<u8>` at fixed points," though this is an
  implementation simplification, not a load-bearing part of the design (the
  regions themselves — prologue/row/batch — stay real either way).
- **Not deleted: `TileSlice`, `Point4`, `Counter`, `OutStep`, the branch
  machinery.** The runtime trip counts and origin are still runtime — see §2.
  `TileSlice`'s fields don't change shape; what changes is that the emitted
  loop body reads fewer of its own coordinate slots because more of its state
  lives in registers across the loop rather than being written down and read
  back every iteration.

## 5. Staged migration (superseded — see the re-staged migration in the revision above)

Each stage has a checkable invariant, following this codebase's existing
practice (byte-identity carried the ABI-unification refactor's pure-refactor
steps; a structured diff carries steps that are supposed to change codegen —
see the escape-hatches plan's methodology note, `docs/plans/2026-09-01-register-allocation-escape-hatches.md:~215`).

**Stage 0 — descriptive only, no allocator change.**
Compute `LoopShape` (§1) at the `Lattice::bake` call site and thread it
through `jit_cache::compile`/`compile_collapse` as an *extra key component*,
with the emitted code identical regardless of its value (it's consulted by
nothing yet). *Invariant:* byte-identical output to today for every existing
call site; the only observable change is that `entry_count()` can now report
more than one entry for what was previously one arena+root pair, if and only
if it's actually called at more than one `loop_mask`. A test that bakes the
same kernel at `Lattice::frame` and `Lattice::scanline` and asserts
`entry_count()` grows by exactly one (not two — dedup within a `loop_mask`
still works) pins this.

**Stage 1 — give the allocator the three-region view, still one register
outcome.**
Change `RegisterAllocator::allocate`'s input from `Vec<Def>` to the
three-region schedule (§3), but keep the *policy* identical to today: every
loop-carried value still gets a memory slot; the only thing that changes is
that `HoistCtx`'s slot assignment is now driven by the new allocator's
placement decisions instead of being computed independently by
`plan_collapse_hoist`'s caller. *Invariant:* structured diff — assert the
only bytes that move are addressing-mode encodings for what are still,
by policy, all-memory hoisted values (i.e., the emitted machine code should
be provably equivalent, differing at most in slot numbering), backed by
differential execution across all four backends on the existing
`collapse_loop.rs` (deleted)/`spill_pressure.rs` (deleted) test suites. This stage is real
plumbing work with no user-visible win — it exists so stage 2 is "change one
policy" rather than "rewrite the allocator and change the policy at once."

**Stage 2 — let loop-carried values compete for registers.**
Flip the policy: a value live across a back edge is spillable like any other
value, not forced to memory. This is where `hoist_slots` stops being
unconditional and coordinates can go resident. *Invariant:* this stage is
expected to change emitted code and improve it — so the checkable claim is
not byte-identity but (a) every existing correctness test still passes
(`collapse_loop.rs` (deleted), `spill_pressure.rs` (deleted), `muladd_rounding.rs`,
`transcendental_jit.rs`, the JIT/interpreter differential oracle), and (b) a
new benchmark on `pixelflow-codegen`'s JIT bench harness
(`jit_bench.rs`) shows a measured cycle reduction on a kernel with at least
one frame-hoisted value and a tight batch body, on every backend —
"improved" is the falsifiable claim here, and the harness that already
exists for exactly this purpose (median-of-samples, per
`CLAUDE.md`'s cost-model-training section) is what falsifies or confirms it.

**Stage 3 — specialize on `loop_mask` for real.**
Consume the `LoopShape` from stage 0 to skip emitting loop levels the
`loop_mask` says are trivial (a `Lattice::point`/`Lattice::scanline` kernel
never gets a live, checked Y-loop). *Invariant:* differential test against
the un-specialized collapse kernel at every `loop_mask` value actually used
in the codebase (point, scanline, frame, tensor index-range, index2) —
same output, fewer emitted branches, smaller `SCAFFOLD_HEADROOM` bound.

Stages 0-1 can land independently of each other's timing but not out of
order; stage 2 depends on stage 1; stage 3 depends on stage 0 and benefits
from (but does not strictly require) stage 2.

## 6. Risks, and what would make this not worth doing

- **Register pressure across a loop is a genuinely harder allocation problem
  than straight-line code**, and the codebase's own register files are
  small (4-10 allocatable registers per the escape-hatches plan's table).
  If step 2/3 of the escape-hatches plan (declare `fixed`, free the phantom
  reservations) hasn't landed first, this plan is optimizing allocation
  policy on top of a pool that's artificially small for reasons unrelated to
  loops — the wins here are likely to be dwarfed by the escape-hatches
  plan's own AVX-512 finding (6 of 32 registers allocatable today, 22
  provably available). **Sequencing recommendation: land escape-hatches
  steps 1-2 before stage 2 of this plan**, or the register-pressure work here
  is measuring the wrong bottleneck.
- **~~Cache-key inflation is a real failure mode, not a hypothetical one~~** —
  *withdrawn 2026-09-02: extents are in the key by decision; the cost is one
  kernel per distinct extent, to be measured and recorded above.* Original:
  §2 already bounds it for `loop_mask`, but if a future author is tempted to
  widen the compile-time lattice to carry more (exact extents, an unrolled
  bound, a stride), the terminal's own resize path is the regression that
  would catch it, and it should be an explicit test (`entry_count()` stays
  bounded across a synthetic resize sweep), not a hoped-for property.
  Recommend a `jit_cache` test to that effect lands alongside stage 0.
- **~~Reduction binders (`Kernel::over`) have loop structure this plan doesn't
  touch.~~** *Withdrawn 2026-09-02 — this was the miss the revision above
  corrects: coordinate axes and reduction indices are one binder nest and
  the schedule has N regions from the start.* Original text follows.
  **Reduction binders (`Kernel::over`) have loop structure this plan doesn't
  touch.** Variance bits 4-8 (`pixelflow-ir/src/variance.rs:12-13`) already
  exist for exactly this, and a kernel with both a coordinate lattice and a
  reduction binder has *two* loop nests, only one of which this plan gives
  the allocator visibility into. Extending the three-region schedule to N
  regions for N nested binders is a natural generalization but is explicitly
  out of scope here — it should be its own plan once this one has landed and
  been measured.
- **Complexity-vs-payoff**: if stage 2's benchmark (§5) doesn't show a
  measurable win — plausible if the hot kernels in practice are dominated by
  transcendental expansions rather than coordinate/hoist traffic — the
  three-region allocator plumbing (stage 1) is pure complexity with no
  corresponding benefit, and the honest move is to stop after stage 0/1 land
  the descriptive infrastructure and not chase stage 2 further. The staged
  design is deliberately structured so that's a legitimate place to stop:
  stage 1 alone is defensible as "the allocator's input type finally matches
  what the emitter has been doing since collapse loops existed," independent
  of whether stage 2 pays off.
- **This plan does not fix the e-graph/allocator conflation everywhere** —
  only the coordinate-loop nest. `ScheduledOp`/`Def`/`Allocation` still
  serve both the e-graph's acyclic-DAG needs and (after this plan) the
  allocator's three-region loop needs; that dual service is narrowed, not
  eliminated. A cleaner long-term separation (the e-graph owns `Vec<Def>`
  exclusively, the allocator owns a distinct loop-shaped IR the emitter
  lowers `Vec<Def>` regions into) is a larger refactor this plan
  deliberately does not propose, on subtract-before-add grounds: introduce
  the region split first, see whether the existing `Def`/`ScheduledOp` types
  are actually inadequate once regions exist, before reaching for a new IR.

## 7. Appendix — `Field` conflates two denotations, and that bears on this plan

Raised mid-design: should `Field` move out of `pixelflow-core`, since the
dependency graph (`ir → search → codegen → compiler → core`, verified below)
places the most primitive value type at the top of the stack instead of the
bottom. This surfaced two claims that turned out to need correction, and one
finding underneath them that is the one worth carrying forward. It bears
directly on this plan because §1 already had to decide where a
codegen-facing lattice projection lives relative to `Field`; this section
makes that reasoning explicit instead of implicit.

### The dependency graph, verified

`pixelflow-core/Cargo.toml` depends on `pixelflow-ir`, `pixelflow-codegen`,
and `pixelflow-compiler` as real dependencies. `pixelflow-compiler/Cargo.toml`
depends on `pixelflow-core` only under `[dev-dependencies]` (its benches).
`pixelflow-codegen/Cargo.toml` depends on `pixelflow-ir` and
`pixelflow-search`. No cycle: the real chain is `ir → search → codegen →
compiler → core`, and `Field` (`pixelflow-core/src/lib.rs:428`) sits at the
top of it — the most primitive value in the system is defined in the crate
every other compiler crate would need a dependency *on* to name it, which is
why none of them do, and why `pixelflow-codegen/src/lib.rs:31-60` carries a
hand-maintained, cfg-triplicated `JIT_VECTOR_BYTES: usize` const instead of
just asking `Field` its width — its own doc comment says exactly this
("The JIT has no dependency on `pixelflow-core`, so it cannot name `Field`
directly") and narrates the resulting bug: the two width definitions
diverged once (`+avx512f` on a non-AVX-512 host: codegen kept emitting
512-bit code while core's `Field` had silently narrowed to 256), because
they are two independently-maintained items that merely promise to agree,
not one item imported twice. That is `CLAUDE.md`'s own "one definition,
imported, not restated" rule, violated by the crate layout itself.

### First correction: the visibility claim

I first reasoned that moving `Field` out would turn `pub(crate)` into `pub`
across a new boundary. Checked, and that framing doesn't survive contact
with the code: **there is no method named `Field::splat`.** `Field<A>`
(`lib.rs:428`) has no inherent `splat` at all; its public constructor is
`Field::from(f32)`, which calls `FieldStorage::splat_storage` internally
(`storage.rs`). `splat` only exists as a *trait* method,
`SimdOps::splat`/`SimdU32Ops::splat` (`backend/mod.rs:74,270`), implemented
by the concrete lane types `F32x4`/`F32x8`/`F32x16`/`U32x4`/... in
`backend/x86.rs` and `backend/arm.rs`.

And that trait, those types, and the modules holding them are **already
fully `pub` today**, independent of any crate split: `pub mod backend;`
(`lib.rs:142`), `pub mod x86;` / `pub mod arm;` inside it
(`backend/mod.rs:280,283`), `pub trait SimdOps { fn splat(val: f32) -> Self;
... }`, and every concrete lane type (`pub struct F32x4(__m128);`, etc.) —
no `#[doc(hidden)]` anywhere in that tree. So the door CLAUDE.md's "do not
expose `SimdVec`s" rule describes is not closed by the current crate
boundary and would not be newly opened by moving `Field`; **it is already
open, and something is already walking through it**:

- `pixelflow-graphics/src/render/discrete.rs:3-4,88` imports
  `pixelflow_core::backend::{Backend, SimdOps, SimdU32Ops}` and
  `backend::x86::Avx512` directly and calls `NativeU32Simd::splat(val as
  u32)` — and its own comment (`// Selection logic matching pixelflow-core`)
  admits it re-derives a *second*, hand-synchronized copy of the
  target-feature dispatch `pixelflow-core` already has, which is the same
  duplication defect as `JIT_VECTOR_BYTES` one layer up, in a downstream
  crate this time.
- `pixelflow-core/tests/x86_{backend,avx2_backend,avx512_backend}_tests.rs`
  call backend-level `splat` 61 + 62 + 64 = **187 times**, and `tests/` is
  compiled as an external crate seeing only the public API — 187 confirmed,
  live, external call sites into exactly the layer `CLAUDE.md` says must
  never be named.

So the honest framing is not "the move would expose lanes," it's "lanes are
already exposed, the move is unrelated to that fact, and fixing it — sealing
`backend`/`storage` to `pub(crate)`, keeping only `Field`/`Discrete`/`Mask`
and their arithmetic trait impls public — is worth doing regardless of
whether `Field` ever changes crates." If it *does* move, `pub(crate)` becomes
a **smaller** privacy boundary (fewer other files share it), so sealing at
that point is strictly easier, not harder, than sealing it in place.

### Second, sharper correction: `Field` is two denotations wearing one type

The deeper issue isn't visibility, it's `CLAUDE.md`'s own failure mode —
"when you extend a type's meaning, extend its type" — applied to `Field`
itself. Two genuinely different things currently share one Rust type:

1. **The language primitive.** What `Manifold::eval` returns
   (`Manifold<P = (Field, Field, Field, Field)>`,
   `pixelflow-core/src/manifold.rs:123`), what `X * Y + Z` denotes in the
   algebra, what a rewrite rule in `pixelflow-search` reasons about. This
   value has no lane count and no register width — mathematically, `sin(X)`
   means the same thing whether it is ever run 1-wide, 16-wide, or (on some
   future target) once per GPU thread with the hardware doing the batching
   invisibly, the way SIMT already works.
2. **The register-sized ABI unit.** "One machine batch" — the literal bit
   pattern an `xmm`/`ymm`/`zmm`/NEON `v` register holds, the thing whose
   byte count `JIT_VECTOR_BYTES` names and `Reg`/`RegisterFile`
   (`pixelflow-codegen/src/emit/regalloc.rs`) place into physical registers.

`Field<A: FieldStorage>(pub(crate) A::Storage)` is asked to be both at once,
and the seam where that fusion is paid for is visible in the code: every
JIT call site that hands `Field` across the ABI boundary carries a **runtime**
`debug_assert_eq!(size_of::<V>(), JIT_VECTOR_BYTES)`
(`pixelflow-codegen/src/emit/executable.rs:82-89`,
`pixelflow-codegen/src/jit_manifold.rs:44`) rather than a compile-time
equality — because nothing in the type system says `Field`'s width and the
JIT's emitted width are the same thing; they merely happen to be cfg-gated on
identical predicates today, the exact "two items that promise to agree"
pattern the `JIT_VECTOR_BYTES` doc comment already narrates a bug from.
`Lattice::bake` (`pixelflow-core/src/lattice/mod.rs:389-462`) is the site
where this is most visible: it constructs `x0: Field` from `Field::sequential`
and hands it straight to `pixelflow_codegen::Point4::new(x0, ...)` /
`call_collapse::<Field>`, i.e. it uses `Field` in denotation (2) — as the raw
ABI payload transmuted into a register — one function away from every other
use of `Field` in `pixelflow-core` being denotation (1).

**Which one does the register allocator actually allocate?** Neither,
literally: `RegisterAllocator::allocate` (`regalloc.rs:410`) never mentions
`Field` — its currency is `ValueId → Placement(Reg(u8) | Spilled |
Remat(u32))`. The allocator already lives entirely at denotation (2)'s
level of abstraction (a register holds *some* fixed-width payload; the
allocator does not care what), which is exactly why `pixelflow-codegen` gets
away with never depending on `pixelflow-core` today. If a real `Batch` type
("this target's native register width, as a Rust type") is ever introduced
to replace `JIT_VECTOR_BYTES: usize` + `call_collapse<V: Copy>` +
`debug_assert_eq!`, it belongs beside `Reg`/`RegisterFile` in
`pixelflow-codegen`'s emit layer (or a leaf ABI crate below it) — **not** in
the algebra crate, and **not** as the same type as the language primitive.
That would let `size_of::<Batch>() == JIT_VECTOR_BYTES` become one constant
instead of a checked coincidence, closing the exact bug class the doc
comment already narrates.

**Where does the language primitive live, and what does it need to know?**
Wherever the algebra (`Manifold`, `X`/`Y`/`Z`/`W`, arithmetic, `Select`)
ends up, and it needs to know neither a lane count nor a register width —
keeping it width-agnostic is precisely what keeps a hypothetical
non-SIMD-register target (a GPU SIMT lane, an interpreter, a future scalar
fallback) representable at all. Defining the primitive as "64 bytes" (or
16, or 32) forecloses that by construction, the same way `Field<A>` today
can only be instantiated over CPU-SIMD storage because `FieldStorage` is
defined in terms of `SimdOps`/`SimdU32Ops`, which are themselves defined in
terms of fixed-width register operations. **Denotation (1)'s trait surface
should be lanes-free** — arithmetic, comparisons producing masks, select,
transcendentals — with the *choice* of how many values one physical
operation processes belonging entirely to the backend that instantiates it,
never to the primitive's own definition.

**Does the lattice belong with (1), (2), or neither?** Neither, and for the
same reason §1/§2 of this plan already separate `LoopShape` from
`pixelflow_core::Lattice`'s full `extent`/`origin`: batch width (2) is a
*target* property (which ISA, how many lanes fit a register); lattice extent
is a *domain* property (how many samples the caller asked for). These are
independent axes — a `Lattice::frame(1920, 1080, 0.0)` means the same 1080×1920
samples whether compiled 4-wide, 16-wide, or not vectorized at all — and
`pixelflow_core::lattice::Lattice`'s own struct
(`extent: [u32; 4], origin: [f32; 4]`, `lattice/mod.rs:211`) already has zero
dependency on `Field` in its definition; only its *evaluator* methods
(`collapse`, `bake`, `collapse_with`) need denotation-(1) `Field` for the
naive path and denotation-(2)-shaped codegen types (`Point4`, `TileSlice`,
already Field-free and already living in `pixelflow-codegen`, confirmed —
see below) for the JIT path. The language primitive should know about
neither axis; the lattice should know about the domain axis only; the batch
width belongs to the target axis alone. This plan's `LoopShape` (§1) is
already scoped to the domain axis for exactly this reason, and this section
confirms rather than revises that placement.

One correction to the original framing worth logging precisely:
`Point4`/`TileSlice`/`Extent2D` were named as *candidates to move* alongside
`Field`, but checking `pixelflow-codegen/src/emit/executable.rs:108,158,213`
shows they are **already** defined exactly once, already in
`pixelflow-codegen`, already generic (`Point4<T>`) with no `Field`
dependency — `Lattice::bake` constructs `Point4<Field>` at its call site
the same way it would construct `Point4<Batch>` under the split above.
Nothing about them needs to move; they are the existing precedent for where
a codegen-facing, Field-free ABI shape type belongs, which is the same
place this plan's `LoopShape` already lives.

### What this means for the present plan, concretely

None of this changes §1-§6 above — `LoopShape` was already designed to be
Field-free and to live in `pixelflow-codegen`, and the three-region
allocator schedule in §3 was already operating purely on `Def`/`ScheduledOp`/
`Reg`, denotation (2)'s vocabulary, with no `Field` dependency to begin
with. What it does is retroactively justify that placement rather than leave
it as an unexamined choice, and it surfaces a **separate, independently
valuable** crate-layout plan this document does not attempt to spec in full
(subtract-before-add: the region split in §5 should land and be measured
before taking on a second, orthogonal refactor):

1. Split `Field<A>`'s two denotations. The language-primitive half
   (arithmetic/comparison/select/transcendental surface, lane-count-free)
   and the register-sized-batch half (what the JIT ABI actually transmutes)
   become two types, not one generic parameterization of the same struct.
2. Seal `backend`/`storage` (or wherever the lane-level SIMD types land
   after the split) to `pub(crate)` — this is overdue regardless of (1), per
   the 187 already-external call sites and `pixelflow-graphics`'s duplicated
   dispatch logic found above — and give `pixelflow-graphics`'s
   `discrete.rs` a real shared entry point instead of its own copy of the
   target-feature `cfg` chain.
3. Independently corroborating evidence for why ABI shape wants to live in
   a type rather than a signature every caller re-declares: reported (not
   independently reproduced here) from the concurrent one-kernel-ABI branch
   — deleting the per-batch ABI made hand-written `extern "C"` test helpers
   that still assumed the old per-batch signature **segfault** against a
   collapse kernel, rather than fail to compile or return a wrong value.
   That is the failure mode a type-checked ABI boundary (`Batch`, or a
   `KernelFn`-family newtype the caller cannot mismatch) exists to convert
   into a compile error, and it is the same class of argument this plan
   already makes in §4 about `EmitCtx`/`Vec<Def>` doing too many jobs at
   once — a signature everyone re-derives by hand is a convention, and a
   convention is an invariant something else will eventually break.

This crate-layout question — split `Field`, seal the SIMD layer, give the
register-sized batch type a real home in the emit layer — is real, motivated
by concrete evidence, and out of scope for this plan to execute. It should
be written up as its own design document once §5's stages 1-2 here have
landed, at which point the register allocator's actual denotation-(2)
vocabulary (`Reg`, `RegisterFile`, the loop-aware three-region schedule) will
be settled and won't need revisiting the moment `Field` moves under it.
