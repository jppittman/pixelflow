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
| [a surviving `Reduce` is a loop](plans/2026-09-10-a-surviving-reduce-is-a-loop.md) | R0, 2a, 2b, **2c** (a fold reaches codegen as a loop) | **nested fold loops** — `extract_folds` carves one level; the winding fold is still unrolled because it is the nested one | `ExpandReduce` is demoted, not deleted: both compile entries run `ExpandNestedReduce`, which unrolls only a `Reduce` inside another's body. Emit −61%, atlas −22%. `HalveFold` (E1b) landed separately. |
| [emit should just emit](plans/2026-09-12-emit-should-just-emit.md) | G1 (`Guard` node, unchosen) | **G2** — emitter emits it, allocator reads regions off structure, the analysis deletes | G1 is additive by design: nothing constructs a `Guard`, and `arena_to_schedule` panics on one. |
| [composition is linking](plans/2026-09-09-composition-is-linking.md) | L1, L2, L3 | **L4** — `Ref(k) ⟷ body(k)` as a growth-gated rule | `expand_refs` still inlines every `Ref` unconditionally, so inline-vs-by-reference is not yet a choice. L5 (a survivor is a call) follows. |
| [one conditional, three lowerings](plans/2026-09-08-one-conditional-three-lowerings.md) | D1 (`mask_support`) | D2 — emit the split | D1 derives the range and checks it; nothing is lowered. |
| [glyph as a fold execution](plans/2026-09-09-glyph-as-a-fold-execution.md) | S0, S1, S1b | S3 — one program per font, **reframed**: bucket trip counts, 95 → 6 programs | S2's `cells` is on no production path. S3-as-written (one global program) is a worse trade than bucketing. |

Measured, at `9643f3b`, tile 16, 95 glyphs: optimize ~1,896 ms, emit ~3,704 ms
(down 76% from the guard fixes), and **21.1 MB of emitted code, mean 227
KB/glyph**. The code-size number is what 2c exists to collapse.

**After 2c**, same harness and tile, re-baselined on one host (before: emit
4,640 ms, 21.1 MB): emit **1,431 ms** (−69%), atlas **15.8 MB**, mean **174
KB/glyph** (−25%), `'@'` alone 478,404 B (−27%), and saturation unmoved at
~2.5 s. It did not
collapse, and the reason is named: a glyph's winding fold is nested inside its
distance fold (`Kernel::by_ref` + `expand_refs`), and `extract_folds` carves
one level, so the winding is still unrolled. **Nested fold loops** is where the
rest of that 15.8 MB is.

---

## The shape

**Almost everything below is one pattern.** A structure the language has is
destroyed early by an unconditional pass, and a later stage spends real work
partially reconstructing it:

| the structure | destroyed by | reconstructed by |
|---|---|---|
| a fold — a **loop** | ~~`ExpandReduce`, unconditionally~~ → `ExpandNestedReduce`, only a fold inside a fold's body | a fold survives to codegen as a loop (2c); the *nested* one is still unrolled, so a glyph's winding fold is the remaining instance (**N5**, H4) |
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
| **H1** | **S3 — one program for the font.** Font-wide extent, table padded with monoid identities, so every glyph compiles to the same program and a glyph becomes a table write. 95 compiles → 1. With H2 measured, this is the whole hump. S3's own doc calls itself "a trade, not a win" for general use, but for the **atlas** path a glyph bakes once into texels and is a gather forever after, so the padding is a one-time bake cost, not per frame — worth re-deciding when H1 is picked up. | [glyph-as-a-fold-execution](plans/2026-09-09-glyph-as-a-fold-execution.md) §S3 |
| **H2** | ~~Split the 331 ms between compile and collapse.~~ **Done** — see [results](results/2026-09-10-glyph-bake-hump.md). | — |
| **H3** | **Hash-consing in `ExprArena`.** Prototyped and measured: arena 2,721 → 154 nodes, 2.1–2.2× on the glyph suites, extracted kernel unchanged. In flight (JP). Lands on the compile half, so it compounds with H1 rather than competing. | [exprarena-on-dag](plans/2026-09-09-exprarena-on-dag.md) §5.2 |
| **H5** | ~~Bound the guard search by what a guard can pay.~~ **Search killed** (`3a4c4e3`): `cluster_select_arms` is one unconditional pass — partition every select worth guarding and not already contiguous, outermost first, each once. `MAX_CLUSTER_ROUNDS`, `is_improvement` and `guarded_spans` went with the hill-climbing. **31.5 s → 20.0 s** on the 95-glyph atlas *with guards kept* (the 9.8 s figure in [results](results/2026-09-10-glyph-bake-hump.md) is what the optimization is worth, not a target — it comes from discarding them). `MISPREDICT_PENALTY_CYCLES` stays: one comparison, and measured, not tuned — a glyph's coverage mask is 3.6× *slower* guarded. **What remains:** `select_arms` is recomputed once per partitioned select, O(selects²·n). A single stable sort keyed by arm ownership would be one pass. But see **D6/D7** — the decision belongs in the e-graph, and optimizing this further is polishing a reconstruction. | [one-conditional-three-lowerings](plans/2026-09-08-one-conditional-three-lowerings.md) §8 |
| **H4** | **Ask B — hoist binder-only work out of the pixel loop.** ~~On the hump.~~ **Demoted by H2**: it optimizes *collapse*, which is 0.2% of a bake, and a glyph bakes once into the atlas and is a gather forever after. Still real for per-frame kernels that are not atlas-cached; not the terminal's startup problem. **Do not patch `contains_gather`** (N1) and do not write a new hoist (N5). | [a-glyph-is-a-circle](plans/2026-09-09-a-glyph-is-a-circle.md) §B |

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

- `Kernel::parts()` hands out the **unlinked** fragment, and five measurement
  consumers each learned to link first. Right division, five copies of one
  line. ([composition-is-linking](plans/2026-09-09-composition-is-linking.md) §7)
- `cells` / `text_union` reach only one Criterion bench; nothing on screen has
  ever gone through them. Delete with S2, not before — they are the worked
  example of a domain-side extent.
  ([glyph-as-a-fold-execution](plans/2026-09-09-glyph-as-a-fold-execution.md) §S2)
- ~~"The hump" drifted from this file's own rule — an index and a status,
  never the design — into several screens of narrative.~~ **Done** — the
  measurements moved to
  [2026-09-10-glyph-bake-hump](results/2026-09-10-glyph-bake-hump.md); the
  section is an index and a pointer again.
