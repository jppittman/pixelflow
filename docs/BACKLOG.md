# Backlog

## Metadata
- **Status**: `Plan of record`
- **Verified against**: `6f3eb619e314304149db65d71bafbe7c096cfd15`

The running list of open work. One line per item, pointing at the document
that owns the detail — this file is an **index and a status**, never the
design. If an entry here starts explaining itself, it wants a plan doc.

**Why this exists.** A session's own task list dies with the session, and the
next one re-derives it from `git log` and guesswork. Every item below was
found by someone who then had to explain it again. Edit this file in the same
CL as the work; an entry that goes stale is worse than no entry, because it
reads as current.

Ordering inside a section is rough priority, not a commitment.

---

## Where the staged work actually is

The sections below are organised by *subject*, which is right for finding
things and wrong for answering "is that done?". Several plans land in numbered
stages, and a stage that is deliberately additive — built, correct, and with
no production caller yet — reads exactly like a stage that is finished. This
table is the one place that distinguishes them. **Update it in the same CL as
the stage.**

| plan | landed | next | note |
|---|---|---|---|
| [a surviving `Reduce` is a loop](plans/2026-09-10-a-surviving-reduce-is-a-loop.md) | R0, 2a, 2b, **2c** (a fold reaches codegen as a loop), **nested fold loops** (H6 step 1: a fold inside a fold is a loop inside a loop; `ExpandNestedReduce` deleted) | [collapse-is-a-fold](plans/2026-09-16-collapse-is-a-fold.md) §5 steps 2–6 | Nothing unrolls on either compile entry. Emit −61%, atlas −22% after 2c; the nested-loop numbers are in H6. `HalveFold` (E1b) landed separately. |
| [emit should just emit](plans/2026-09-12-emit-should-just-emit.md) | G1 (`Guard` node, unchosen) | **G2** — emitter emits it, allocator reads regions off structure, the analysis deletes | G1 is additive by design: nothing constructs a `Guard`, and `arena_to_schedule` panics on one. |
| [composition is linking](plans/2026-09-09-composition-is-linking.md) | L1, L2, L3 | **L4** — `Ref(k) ⟷ body(k)` as a growth-gated rule | `expand_refs` still inlines every `Ref` unconditionally, so inline-vs-by-reference is not yet a choice. L5 (a survivor is a call) follows. |
| [one conditional, three lowerings](plans/2026-09-08-one-conditional-three-lowerings.md) | D1 (`mask_support`) | D2 — emit the split | D1 derives the range and checks it; nothing is lowered. |
| [glyph as a fold execution](plans/2026-09-09-glyph-as-a-fold-execution.md) | S0, S1, S1b | S3 — one program per font, **reframed**: bucket trip counts, 95 → 6 programs | S2's `cells` is on no production path. S3-as-written (one global program) is a worse trade than bucketing. |
| [a glyph is a formula](plans/2026-09-23-a-glyph-is-a-formula.md) | nothing — denotation only (2026-09-23) | The glyph is `area(Σ_p t_p)`: coverage as the pixel integral of the pieces' indicators, one reduction, no winding number, no distance field, no boundary test. Compiler: **`area`** (the definite integral over the pixel, an e-graph operator like `Dwrt`: the e-graph factors it — linearity, factors invariant along the measure pulled out, an indicator narrows the range, half-plane and conic over the clipped square exact, Taylor through `Dwrt` — priced so extraction never keeps one; where each factor is evaluated is demand, not a rule of the integral); **a range is a value** (sibling reductions over one range are one loop scope); **an arm may own a root when it owns every scope that reads it** (so a box select skips the loops inside it); **the piece tree as bound data**, one program per font | Baseline in [results/2026-09-23-freetype-comparison.md](results/2026-09-23-freetype-comparison.md): `8`@32 collapse 153 µs on AVX-512 vs FreeType 1.8–3.9 µs per glyph unhinted at the same size (its raster alone 1.5–4.3 µs); a 50-glyph run 510× FreeType because the per-glyph box selects own none of their folds (guard telemetry). The H6 stack removed the instruction overhead; what is left is the algorithm, and the plan puts it in the compiler rather than the font code. |

Measured, at `9643f3b`, tile 16, 95 glyphs: optimize ~1,896 ms, emit ~3,704 ms
(down 76% from the guard fixes), and **21.1 MB of emitted code, mean 227
KB/glyph**. The code-size number is what 2c exists to collapse.

**After 2c**, same harness and tile, re-baselined on one host (before: emit
4,640 ms, 21.1 MB): emit **1,431 ms** (−69%), atlas **15.8 MB**, mean **174
KB/glyph** (−25%), `'@'` alone 478,404 B (−27%), and saturation unmoved at
~2.5 s. It did not
collapse, and the reason is named: a glyph's winding fold is nested inside its
distance fold (`Kernel::by_ref` + `expand_refs`), and `extract_folds` carved
one level, so the winding was still unrolled. **Nested fold loops** (H6 step 1)
is where the rest of that 15.8 MB was; its numbers are in H6's row.

---

## The shape

**Almost everything below is one pattern.** A structure the language has is
destroyed early by an unconditional pass, and a later stage spends real work
partially reconstructing it:

| the structure | destroyed by | reconstructed by |
|---|---|---|
| a fold — a **loop** | ~~`ExpandReduce`, unconditionally~~ → ~~`ExpandNestedReduce`, only a fold inside a fold's body~~ → nothing | a fold survives to codegen as a loop (2c), and a fold inside a fold as a loop inside a loop (H6 step 1). **Closed.** What remains of **N5**/H4 is the hoist out of a fold, which is placement once the lattice's loops are folds too (H6). |
| a reference — a **call** or a block | `ExpandRefs`, unconditionally, *before* saturation | nothing; extraction never sees a boundary to keep (**N2**, N3) |
| a tabulation — a **name for memory** | `push_gather` in `DiscreteManifold::kernel_for` | a `Gather` case in every pass — `contains_gather`, `lower_dwrt`'s table rule, `MAX_BOUND_BUFFERS` (**N1**, N4) |
| a select's arms — **blocks** | flattening the DAG to a linear schedule | `cluster_select_arms`, which was a permutation *search* and 73% of a glyph bake (**H5**, now one pass) |
| a mask's region — a **domain split** | never derived at all | nothing, until `mask_support` (**D1**, landed) |

The fix is the same shape every time, and it is not "optimize the
reconstruction": **stop destroying it, and let the cost model choose.** That is
what [a-kept-structure-is-control-flow](plans/2026-09-10-a-kept-structure-is-control-flow.md)
says for control flow and [one-name-bound-later](plans/2026-09-10-one-name-bound-later.md)
says for names — the same claim about the two halves of a function, its
parameters and its control flow. **D7** is that claim about the conditional.

**Two threads, and they are not the same work.**

- **The terminal is unusable**, and the fix is **H1 (S3)** alone: 95 compiles →
  1, ~31 s → ~2 s. This is *not* an instance of the pattern above. It is
  "don't pay a cost 95 times," it is orthogonal to everything architectural,
  and it is **untouched**.
- **The compiler reconstructs what it destroyed.** Everything in Names, Demand
  and the e-graph. This is where the measurement led and where the effort has
  gone; it makes the compiler right rather than merely fast.

Keep them apart when prioritising. H5 cut a bake 36%, which is a real win and
also a constant factor on a cost H1 would make **95× smaller**.

## The hump

**A glyph bake is ~99.8% compile; the hump is one function
(`cluster_select_arms`, 73% of a bake). H1 alone — one program per font
instead of one per glyph — makes the terminal usable.** Full measurements
(cold/warm split, saturation telemetry, the emitter split, the
`cluster_select_arms` finding) are in
[2026-09-10-glyph-bake-hump](results/2026-09-10-glyph-bake-hump.md).

| | what | where |
|---|---|---|
| **H1** | ~~S3 — one program for the font.~~ **Done.** A piece table's trip count is rounded up to `u32::next_power_of_two()` before it becomes the fold's extent, so glyphs whose piece counts round to the same bucket share a program; the padding rows are exact identities of both folds (`loop_blinn::tests::a_padding_row_is_an_exact_identity_of_both_folds`, plus every coverage golden, bit-identical). Measured on this host: 95 → 6 programs at tile 16 (was 36), 7 at tile 32 (was 39); cold `GlyphAtlas::warm` 20.3 s → 3.7 s at tile 16, 25.9 s → 12.0 s at tile 32 — a net win despite the unrolled fold evaluating every padding row, because collapsing 30-odd compiles into one outweighs the per-compile growth. | [glyph-as-a-fold-execution](plans/2026-09-09-glyph-as-a-fold-execution.md) §S3 |
| **H2** | ~~Split the 331 ms between compile and collapse.~~ **Done** — see [results](results/2026-09-10-glyph-bake-hump.md). | — |
| **H3** | ~~Hash-consing in `ExprArena`.~~ **Done** (2026-09-20). `ExprArena::intern` structurally hash-conses through `Builder::intern`; every `push_*` call routes through it, and nothing in this tree needed the `push_unique` escape hatch. Built arena on a real glyph: 2,721 → 165 nodes (16.5×); extracted node count unchanged (155, both sides) — saturation already normalized the redundancy away, so the win is construction cost, not the emitted kernel. One real, verified exception: the *compiled* code for a two-contour glyph nearly halves (19,566 → 10,005 bytes) because consing lets codegen see the contours' identical fold bodies as one shared subexpression instead of two copies — full `pixelflow-graphics` suite (goldens included) and `xtask isa-matrix --clippy --smoke` (sse2/avx2/avx512) green on both sides. Lands on the compile half, so it compounds with H1 rather than competing. | [exprarena-on-dag](plans/2026-09-09-exprarena-on-dag.md) §5.3 |
| **H5** | ~~Bound the guard search by what a guard can pay.~~ **Search killed** (`3a4c4e3`): `cluster_select_arms` is one unconditional pass — partition every select worth guarding and not already contiguous, outermost first, each once. `MAX_CLUSTER_ROUNDS`, `is_improvement` and `guarded_spans` went with the hill-climbing. **31.5 s → 20.0 s** on the 95-glyph atlas *with guards kept* (the 9.8 s figure in [results](results/2026-09-10-glyph-bake-hump.md) is what the optimization is worth, not a target — it comes from discarding them). `MISPREDICT_PENALTY_CYCLES` stays: one comparison, and measured, not tuned — a glyph's coverage mask is 3.6× *slower* guarded. **What remains:** `select_arms` is recomputed once per partitioned select, O(selects²·n). A single stable sort keyed by arm ownership would be one pass. But see **D6/D7** — the decision belongs in the e-graph, and optimizing this further is polishing a reconstruction. | [one-conditional-three-lowerings](plans/2026-09-08-one-conditional-three-lowerings.md) §8 |
| **H4** | **Ask B — hoist binder-only work out of the pixel loop.** ~~On the hump.~~ **Demoted by H2**: it optimizes *collapse*, which is 0.2% of a bake, and a glyph bakes once into the atlas and is a gather forever after. Still real for per-frame kernels that are not atlas-cached; not the terminal's startup problem. **Do not patch `contains_gather`** (N1) and do not write a new hoist (N5). **Subsumed by H6** §2.2: with the lattice's loops as folds, the hoist is per-scope placement, which exists. | [a-glyph-is-a-circle](plans/2026-09-09-a-glyph-is-a-circle.md) §B |
| **H6** | **Collapse is a fold.** Disassembled, `8`@32 on the AVX-512 tier is 48,452 instructions of which the Loop–Blinn arithmetic is ~3,000: the rest is 1,610 table reads, each a per-lane gather plus its address arithmetic and constant remats, unrolled 34 times — three symptoms of one thing, the collapse loop being a scaffold outside the fold machinery. **Agreed (JP, 2026-09-17):** the lattice is three `Reduce`s over the unit monoid — rows, batches, lanes — wrapped around the kernel by two legalize passes (`collapse(extent)`, `pack(L)`), with one `pub(crate)` `Write` as the body. Lanes are a scope: the fold a `Write` names as its lane binder is executed by lanes, and lane-uniformity is that binder's variance bit. Subtracts the emitter's collapse scaffold, `plan_collapse_hoist`, the `Var(0..3)` axioms, `BandPlan`'s tail and `Point4`/`TileSlice`. Six steps, one PR each. **Step 1 done** (nested fold scopes): a glyph's winding fold is a loop inside its distance fold, and an invariant nested fold is hoisted, run once and read from its slot. `8`@32 on this host's tier: 536,960 B → **12,390 B**, body 24,738 → **612** instructions, table reads 1,610 → **35**, spills 19 → 5; `A`@32 compiles to the identical program, since the body no longer scales with the piece count. **Step 2 done** (one saturation per structure): `optimize_runtime_arena` holds the saturated e-graph keyed on structure — never identity, so table-binding kernels stop bypassing it — and extracts per shape, since extraction is priced by the lattice and saturation is not; a second shape or composition is an extraction, not a saturation. Measured: warming the 95-glyph atlas at tile 32 after tile 16 saturates **1** structure (its one new bucket) instead of 7, 0.41 s → 0.31 s; the atlas itself is 0.37 s cold at tile 16 now, after step 1. **Step 2½** (the allocator is the boss): a fold's binder and accumulator are roots `allocate_nest` places — carried in a register or parked in a slot — instead of a temp reserved across every pool inside, which had capped nesting at three levels on SSE2 before step 5 adds three more. Every carry in the nest is one ranking by reads saved per batch (a fold's roots weighted by trip count) under one constraint (no scope has more carried across it than the pool has above the floor); found and pinned two bugs on the way, a binder slot shared by a glyph's two sibling folds and a hoisted fold's result never handed to its carry. **Step 3 done** (vocabulary): `Write { row, col, lane, value }` (crate-private constructor), `OpKind::Seq`/`Monoid::SEQ`, `Variance` a `u64` with `REDUCE_BINDERS` derived from it (60), `Fold`'s ends `u32`; every consumer's exhaustive match has its arm, and the e-graph declines both new words. **Step 4 done** (the two passes): `collapse` and `pack` live in `pixelflow-ir::passes::lattice`, standalone until step 5 wires them in — the emitter refuses a `Write` until then. `collapse` leaves a one-trip lane fold so `pack` reshapes ranges only and shares the `Write`; the origin is two uniforms the caller declares; names (`Ref`, `Guard`) and derivatives are refused by the pass because substitution cannot reach through them. `Fold::strided` is the stride's second constructor. Records the refused alternatives (a post-e-graph fold pass, a `Row` node, fixing the two emitter sites in place, a `Lane` leaf). **Step 5 done** (the emitter): the two passes run inside `legalize`, the lane fold is inlined into its parent's schedule as a `Write` def, the row and column folds are loops the fold emitter already emitted, the emitter's collapse scaffold and the per-batch ABI are deleted (`fn(ctx, out, pitch)`, origin as two uniforms in a context entry of their own), every backend stores through `emit_write` with a masked or lane-wise remainder, and `pixelflow-core` collapses a band in one call at the band's own shape, compiled on first use. Goldens bit-exact; the five collapse paths execute on all three x86 tiers. **Step 5½ done** (constants are values): the allocator's three hand rules for constants — a tier below every value in the eviction rank, never re-kept, never parked — fold into `in_slot` being true at a constant's birth, so a constant is evicted by distance, re-kept on a re-read, and parked or carried for the scopes inside like any root (a root's hand-off is a read at its definition); the lane binder is no longer counted as bound by the storing scope in `place_roots`, so the iota is a body root built once per call; every x86 tier reads constants from one pool after `ret` anchored in `r8`, one `vbroadcastss` each, where it rebuilt them in two instructions per read. On this host's 128-bit tier: glyph `8`@32 10,005 → **7,288 B** with its per-pixel folds' remats 5 → 0; the cell grid's column loop 2,140 → **1,438 B** per batch with 24 → 0 remats; the packed four-channel frame program 5,773 → **3,672 B**. `constant_traffic` is the instrument. **Step 6 done** (the broadcast load): a `RawGather` whose index lacks the lane binder's bit is `ScheduledOp::Broadcast` — `cvttss2si` of lane 0, the base from the context, `vbroadcastss [base + idx*4]` (or `fcvtzs`/`ldr s, [base, w, uxtw #2]`/`dup`) — split from `Gather` in `arena_to_schedule` on the arena's variance. `8`@32 on this host's tier: `vpextrd`/`vinsertps`/`vmovss`/`vcvttps2dq` 140/105/140/35 → **0**, 10,005 → **7,912 B**, 2,444 → **2,090** instructions. ~~The 35 base-pointer loads per batch remain: one call-invariant GPR value, and nothing allocates GPRs across instructions yet.~~ **A pointer is a value** (2026-09-22): the allocator has a pointer class — `ScheduledOp::Context(k)` is a def, `Gather`/`Broadcast`/`Uniform` take the base as an operand, `Where::Ptr` beside `Where::Reg`, one forward pass per class over one schedule with per-class carry budgets, `RegisterFile::pointers` the pool (`r9`–`r11`; `x3`–`x8`, `x12`–`x15`). `8`@32 on this host's tier: base loads 45 per batch → **3** per call, 5,160 → **4,376 B**, 1,199 → **1,041** instructions. The fold carve stops at a value the enclosing scope computes, so a placeholder's operands are no longer roots parked for nobody. | [collapse-is-a-fold](plans/2026-09-16-collapse-is-a-fold.md), [a-pointer-is-a-value](plans/2026-09-22-a-pointer-is-a-value.md) |

## Names and binding time

| | what | where |
|---|---|---|
| **N1** | **One kind of name.** `Var`/`Uniform`/`Buffer`+`Gather`/`Ref` are five spellings of "bound later", differing only in *when*. Reframes L6 from "delete `Gather`" to "there is one kind of name", with `Gather`'s disappearance a consequence. Blocks a principled H4. | [one-name-bound-later](plans/2026-09-10-one-name-bound-later.md) |
| **N5** | **A kept structure is control flow.** A surviving `Ref` is a call, a surviving `Reduce` is a loop, a `Select` with a derived range is a domain split — one rule, three structures, and today all three are flattened unconditionally so the cost model never chooses. The hoist is *already* shared machinery (`partition_by_scope` is generic over binders); it is handed `[0, 1]` because no fold binder survives to schedule time. **`partition_by_scope(.., &[0, 1, 4])` is the whole of H4** once one does. | [a-kept-structure-is-control-flow](plans/2026-09-10-a-kept-structure-is-control-flow.md) |
| **N2** | **L4 — `Ref(k) ⟷ body(k)` as a growth-gated e-graph rule.** The gate exists (`EGraph::predicted_growth`, asserted against measured delta over 10,819 applications); the rule does not. Its case is a scene composing many *identical* kernels, not construction-side sharing (L3 settled that). | [composition-is-linking](plans/2026-09-09-composition-is-linking.md) §3, §5.1 |
| **N3** | **L5 — a surviving `Ref` is a call.** Second function, coordinate ABI, a register-allocation boundary the allocator does not model. Open question: whether a *tabulated* `Ref` (a leaf that emits a load) avoids all of it, which would let N1 land the cheap half first. | [composition-is-linking](plans/2026-09-09-composition-is-linking.md) §5.2 |
| **N4** | **L6 — a tabulated kernel is a `Ref` with a cached tabulation.** Subsumed by N1; kept as a row because the task list and several commit messages name it. **Not independent:** a table is a `Ref`, reading it is `Kernel::at`, and `at` expands every `Ref` at construction — so a tabulated `Ref` cannot survive a read. Needs L4 and `Apply` first (§8). | [composition-is-linking](plans/2026-09-09-composition-is-linking.md) §4 |

## Demand and the conditional

| | what | where |
|---|---|---|
| **D1** | ~~mask ⟹ index range, symbolic tier.~~ **Landed.** `pixelflow_ir::mask_support` — axis-aligned literals in either operand order, conjunctions (intersect), disjunctions (hull), everything else the full extent. No lowering; production behaviour unchanged. 13 tests: soundness by containment over the whole extent in `pixelflow-ir/tests/mask_support.rs`, usefulness against the real `Kernel` builder and `grid_range` in `pixelflow-core/tests/mask_support_of_a_built_kernel.rs`. **A bare `Var` is load-bearing**: the cell grid samples pixel *centres*, so its arena holds `Add(Var, 0.5)` and the analysis widens; teaching `axis_of` to see through that `Add` without moving the shift into the literal deletes a row of pixels, and a test pins it. | [one-conditional-three-lowerings](plans/2026-09-08-one-conditional-three-lowerings.md) §8 |
| **D2** | Lowering 1 — emit the split: select over the derived range, root specialized at `m ≡ false` over the complement. | *ibid.* |
| **D3** | Bind-time tier, and splitting `IndexRange` into a derived region and a requested band. | *ibid.* |
| **D4** | Interval evaluation, target-aware and rounding outward. Unlocks glyph supports, which the symbolic tier cannot reach (a compound glyph's affine mixes X and Y). | *ibid.* |
| **D5** | Lowering 2 on the general predicate — the superseded demand plan's §1–§2, as the third case rather than the whole subject. **Does not delete `cluster_select_arms`**: `emit/demand.rs` already computes the predicate and disproved that plan's scheduling claim (see the correction in [results](results/2026-09-10-glyph-bake-hump.md)). What it buys is *more* exclusivity than `guards` finds — the per-select `demand_exclusive` vs `exclusive` gap under `PIXELFLOW_GUARD_TELEMETRY` — and `demand.rs` says outright that this gap, not the scheduling claim, is what C1b should be justified by. Read that measurement before starting. | *ibid.* |
| **D6** | **A static demand fraction in the extraction cost** (the demand plan's C2a). Extraction is additive per node; the demand-aware cost is `cost(node) · P(demanded)`, with `P = 1` where a node is unguardable. Where the demand is a row-uniform mask with a known extent, `P` is **static and already in the program** — a segment gated on `y_lo ≤ Y < y_hi` over a 45-row glyph is demanded on `(y_hi − y_lo)/45` of rows. This is what D7 needs and what nothing today supplies. | [demand-is-a-dag-property](plans/2026-09-07-demand-is-a-dag-property.md) §4 |
| **D7** | **`Guard` in the graph, and the sink rule** (C2b) — *guarding becomes the e-graph's decision instead of codegen's.* Today it is neither: `analyze_select_guards` runs on the linearized schedule, after extraction, and the emitter places branches from it. Worse, the graph pulls the other way — `SelectHoistUnary` rewrites `Select(m, f(a), f(b)) → f(Select(m, a, b))`, hoisting shared work *out* of arms, which is right for op count and exactly wrong for guarding when `f` is expensive and `m` is coherent, and **no term in the extraction cost opposes it**. Denote `Select(m, a, b)` as `Guard(m, a) ⊕ Guard(¬m, b)`, add the inverse sink rule, and let the cost decide. Needs D6 for the term. Composes with N1: `Guard(m, Ref(a))` carries its arm as a unit, which is what makes H5's partition *unsayable* rather than merely cheap. | [demand-is-a-dag-property](plans/2026-09-07-demand-is-a-dag-property.md) §4 |

D1 → D2 unblocks **S2**: deleting `cells`, `contour_bounds`, the `Union`
plumbing, `TEXT_CELL`, `min_of`, `may_be_interior` and `chord_winding` —
roughly 800 lines to 150 — and makes H1's padding free.

## The e-graph

| | what | where |
|---|---|---|
| **X1** | ~~Split the ~1,990 ms that is not saturation.~~ **Done** — see [results](results/2026-09-10-glyph-bake-hump.md). The suspicion recorded here (linear-scan regalloc) was wrong; it is `guards::cluster_select_arms`. | — |
| **R0** | ~~Codegen cannot *say* "here".~~ **Done.** Every branch was an opaque per-backend fixup token the emitter carried by hand to a `patch_branch`, against an offset it read off `code.len()` at the one point that was correct. A program is now a sequence of items — instruction, **binding** of a name to this position, **reference** to a name — with `assemble_labeled` for a program that is a value and a streaming `Resolver` for an emitter whose verbs are `&mut self` methods. Both emitter loops (the collapse nest, the `Select` short-circuit) moved; `IsaBackend::Branch`, `emit_jump`, `patch_branch`, the two `emit_skip_if_all_*`, `Aarch64Branch`, `Cond19`, `Rel26`, `emit_jmp_rel32` and `patch_rel32` are gone. No emitted byte changed. | [a-surviving-reduce-is-a-loop](plans/2026-09-10-a-surviving-reduce-is-a-loop.md) §R0 |
| **R1–R3** | **A surviving `Reduce` is a loop.** The emitter is handed 34,993 straight-line instructions for one glyph because `ExpandReduce` unrolls unconditionally and then gather/transcendental expansion multiplies each copy ~4×; `emit` is ~O(n^1.6) in that. A loop makes the program the *body* — ~1,030 entries — and the accumulator lives in a **slot**, exactly as `emit_collapse_loop` keeps its coordinate, so nothing is live across the back edge and `LinearScan` needs no change. **This is the hump's root cause, not H1.** R0 landed; R1's shape is now read off the code rather than guessed: the region is `partition_by_scope(.., &[4, 0, 1])` (binders are innermost-first, so the fold binder goes at the *front* — the plan's earlier `[0, 1, 4]` was backwards), a fold region is shaped like a guard region, and the binder can be its own counter so no new GPR is needed. **2a and 2b have landed** (the nest is now a tree: `Scope::Fold`, a `FoldScope` holding a whole `ScopeCode`, and `within()`/`parked_by_an_enclosing_scope()` as tree walks rather than a chain suffix and prefix). What remains is 2c — delete `ExpandReduce` and emit the loop — which is gated on **E6**, not on more machinery. | [a-surviving-reduce-is-a-loop](plans/2026-09-10-a-surviving-reduce-is-a-loop.md) |
| **E1** | **Geometric `SplitFold`.** *(= R4)* `⊕_{[lo,hi)} = ⊕_{[lo,mid)} ⊕ ⊕_{[mid,hi)}`. Needs no substitution (both halves share the body e-class) and is *exactly* cost-preserving in both extraction arms, so it cannot desync the claim/price audit. Bisect at the midpoint to bound growth at 2n−1. `EmptyFold` already exists as its base case. **Still open, and not what `HalveFold` built** — see E1b. Its advantage over `HalveFold` is precisely that it substitutes nothing, so it is the cheap decomposition; its cost is that bisecting to exhaustion is still O(n) applications (n leaves plus n−1 internal), where `HalveFold` is O(log n). Different trade, both sound. | — |
| **E1b** | ~~Unrolling costs n rule applications.~~ **Done** (`HalveFold`, 2026-09-11): halve the trip count and double the body — `[lo,hi) step s` → `step 2s`, body `b ⊕ b[i := i+s]`. **Stride-2, not range-bisection**: it re-brackets the original left-to-right order, so it needs associativity alone, where the halve-and-offset form interleaves adjacent pairs and would additionally need commutativity. `Fold` gained a `stride`; an odd trip count peels once first, so `peel_back` is the epilogue. `PeelFold` now declines whenever a fold can still be halved — an e-graph fires every matching rule every round, so an ungated `PeelFold` would race it straight back to O(n). Measured: **~89 applications** for the 34,993-term glyph fold. | [a-surviving-reduce-is-a-loop](plans/2026-09-10-a-surviving-reduce-is-a-loop.md) §2a″ |
| **E2** | **A critical-path term in the cost model.** Without it E1 buys reachability and no speed: `CostModel::latency_prior()` sets `depth_threshold: 1024, depth_penalty: 0` ("effectively disabled"), so a 34-deep serial chain and a 6-deep balanced tree price identically. A *global* depth hinge is the wrong shape — what is wanted is the reduction's critical path. | [schedule-cost-model-denotation](plans/2026-09-01-schedule-cost-model-denotation.md) |
| **E3** | Extraction: `shared_dag_dp_pass` is O(L²) — make reach tracking sparse. | — |
| **E4** | ~~Saturation rescans every class with every rule every iteration — dirty tracking.~~ **Already built**, and this entry was stale the day it was written: `EGraph::class_is_dirty` with per-rule `rule_last_swept` baselines and a 2-hop forward-neighbourhood check (`DIRTY_TRACKING_MAX_DEPTH`), wired into the scan so a clean class skips the clone and every `apply`. What remains is narrower and, per X1, **not hump work**: the scan still visits every class for every rule and filters, and `class_is_dirty` is itself a neighbourhood walk rather than O(1), so the cheap path is O(classes × rules) per iteration. A dirty *worklist* would make it O(dirty). Saturation is a constant ~32 ms of a 2,154 ms compile, so this needs a reason other than the hump. | — |
| **E5** | Extraction is not monotone in graph richness: the same kernel in a superset graph can extract a worse DAG. | — |
| **E6** | **Projected graph explosion is not in the cost model, so a loop and its unrolling tie.** A `Reduce` prices as `(len−1) × cost(⊕)` and the DP multiplies its body by `len` (`extract.rs`'s `fold_body_multiple`), which is *correct* — a loop and its unrolling do the same arithmetic, and a latency prior that separated them would be lying. So the choice falls to the tie-break, and that is `Dp`'s `ties` knob, in production `Insertion`, whose `prefer()` is unconditionally `false`; the DP takes a new node only on strict `<`. **Whether a glyph emits a loop or 34,993 straight-line instructions therefore turns on which e-node the class happened to hold first.** What is missing is a second axis: not latency and not depth (that is E2), but how much *emitted program* a choice projects — `emit` is ~O(n^1.6) in instruction count, so the loop is far cheaper to compile and smaller in I-cache at identical runtime. Two pieces already exist: `TieBreak` is a type (with a `Content` research arm beside `Insertion`), and `egraph/growth.rs` already *measures* per-rule growth (`RuleGrowth::median_nodes_added`, `max_nodes_added`) — nothing consults it when costing. Blocks nothing today: 2b is additive and `ExpandReduce` still unrolls everything, so no production kernel has a surviving fold to choose about. | [a-surviving-reduce-is-a-loop](plans/2026-09-10-a-surviving-reduce-is-a-loop.md) |

## Correctness and CI

| | what | where |
|---|---|---|
| **C1** | ~~Trig range is unasserted.~~ **Done** — `pixelflow-codegen/tests/trig_range_jit.rs` asserts `\|sin\|`/`\|cos\| ≤ 1` with no tolerance and NaN outside `TRIG_DOMAIN` against ~100,000 JIT-evaluated points. | CLAUDE.md, "Precision is on the table; range is not" |
| **C2** | **The `'8'` waist bug is open on `main`.** Five fixes tried and refuted; `freetype_oracle.rs` pins it rather than fixing it. The general demand predicate (D5), not a sixth per-select patch, is the intended next attempt. | [demand-is-a-dag-property](plans/2026-09-07-demand-is-a-dag-property.md) §9 |
| **C3** | `CachedText::kernel` sums glyph coverages, so overlapping glyphs can exceed 1. Not on any production path — `core-term` renders through `GlyphAtlas`, and `CachedText` has no non-test caller. | — |
| **C4** | A corpus needs a new acceptance criterion before `gen_bench_corpus` can come back; its quarantine gate compared against the interpreter. | CLAUDE.md, "Cost Model and the Guide" |
| **C5** | A local CI runner, so a full presubmit does not cost a push. | — |
| **C6** | ~~Every CI job set `RUSTC_WRAPPER: sccache` against the GHA cache backend, making sccache a hard dependency of `rustc` rather than an accelerator — a transient DNS failure reaching the cache service killed `ISA matrix` at `rustc -vV` with `compile_requests: 0` on a docs-only commit.~~ **Done** — set `SCCACHE_SKIP_CACHE_CHECK: "1"` (skips the server's eager startup probe of the GHA backend — the incident's path) and `SCCACHE_IGNORE_SERVER_IO_ERROR: "1"` (falls back to local compilation when a running server later loses the cache) alongside every job-level `RUSTC_WRAPPER: sccache` in `rust.yaml`, `benchmark_regression.yaml`, and `postsubmit-flake-detection.yaml`. | .github/workflows/rust.yaml |

## Housekeeping

- ~~`Kernel::parts()` hands out the **unlinked** fragment, and five measurement
  consumers each learned to link first. Right division, five copies of one
  line.~~ **Done**: `Kernel::linked_parts()` is the one accessor; the five
  call sites call it instead of repeating `expand_refs_owned` by hand.
  ([composition-is-linking](plans/2026-09-09-composition-is-linking.md) §7)
- `cells` / `text_union` reach only one Criterion bench; nothing on screen has
  ever gone through them. Delete with S2, not before — they are the worked
  example of a domain-side extent.
  ([glyph-as-a-fold-execution](plans/2026-09-09-glyph-as-a-fold-execution.md) §S2)
- ~~"The hump" drifted from this file's own rule — an index and a status,
  never the design — into several screens of narrative.~~ **Done** — the
  measurements moved to
  [2026-09-10-glyph-bake-hump](results/2026-09-10-glyph-bake-hump.md); the
  section is an index and a pointer again.
