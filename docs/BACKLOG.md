# Backlog

## Metadata
- **Status**: `Plan of record`
- **Verified against**: `3a4c4e3247c90bc1dd980e268f2edec505dd6fd1`

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

## The shape

**Almost everything below is one pattern.** A structure the language has is
destroyed early by an unconditional pass, and a later stage spends real work
partially reconstructing it:

| the structure | destroyed by | reconstructed by |
|---|---|---|
| a fold — a **loop** | `ExpandReduce`, unconditionally, every time | nothing; the loop is simply gone, so `partition_by_scope` is handed `[0,1]` and can never hoist out of a fold (**N5**, H4) |
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

**A glyph bake costs ~331 ms in release** — 31.5 s for a 95-glyph ASCII atlas,
measured at the sha above. `core-term` calls `atlas.warm(&font, ' '..='~')` at
startup (`core-term/src/terminal_app.rs`) and again on every font-size or
density change, so that is ~31 s to launch and ~31 s per resize. **The glyphs
are correct; the terminal is not usable.** Everything in this section is about
that number.

**H2 is answered (2026-09-10, release, this host): a glyph bake is 99.7–99.9%
compile.** Cold vs warm bake, using the fact that `jit_cache` keys compiles by
(canonical key, shape) so a second bake of the same glyph at the same shape
pays only collapse:

```
glyph   px    cold_ms    warm_ms compile_ms compile%
    A   16      172.5        0.2      172.3   99.9%
    O   16      561.8        0.8      561.1   99.9%
    8   16      758.0        0.9      757.1   99.9%
    m   16      531.8        0.5      531.4   99.9%
    A   32      163.0        0.4      162.6   99.7%
    O   32      672.1        1.7      670.4   99.7%
    8   32     2096.0        4.5     2091.4   99.8%
    m   32      639.4        1.6      637.8   99.8%
```

Two consequences, both of which move items in this table:

- **H1 alone is the fix.** 95 compiles → 1 takes ~31 s to ~2 s; the 95
  collapses that remain total well under a tenth of a second.
- **Compile is superlinear in piece count and depends on the shape.** `A` (11
  pieces) and `8` (34) at 32 px are 163 ms and 2,096 ms — 3.1× the pieces for
  12.9× the time — and `8` costs 758 ms at 16 px against 2,096 ms at 32 px on
  the same pieces.

**And compile is not saturation.** Measured the same day from the telemetry
feature's own `wall_clock_us`:

```
A@32: cold 164.4 ms   runtime saturation 32.2 ms   (~20%)
8@32: cold 2154.0 ms  runtime saturation 31.5 ms   (~1.5%)
```

Both saturations are *identical* — `nodes=2881, classes=5000, apps≈7054,
extracted cost 6573` — which is legalize-last working as designed: the fold
stays folded, so the e-graph sees the same program for an 11-piece glyph and
a 34-piece one. Saturation is a **constant ~32 ms**. `8`'s extra ~1,990 ms is
entirely downstream of it.

So the earlier reading here — "E3/E4/E5 are on the critical path, compile *is*
saturation plus extraction plus emit" — was **wrong about which term
dominates**, and is corrected above.

**X1 is answered, and the hump is one function.** Splitting the compile:

```
glyph  optimize_ms  emit_ms  post_opt_nodes     (optimize = saturate+extract+legalize)
A@32          34.4    130.8           1,870
O@32          33.0    655.3           4,516
8@32          33.1  2,082.0           8,548
```

Optimization is flat ~33 ms — so **extraction is inside the constant too**,
which clears E3. Splitting the emitter, by schedule length `n`:

```
glyph       n  variance_ms  partition_ms  cluster_ms  regalloc_ms  emit_ms
A@32    6,055          0.0           0.4        96.1         19.8    133.0
O@32   17,521          0.1           1.8       493.8         93.0    672.8
8@32   34,993          0.1           1.9     1,539.7        285.1  2,088.4
```

**`guards::cluster_select_arms` is 73% of an entire glyph bake.** And what it
buys is constant — the same 282 bytes of extra code for every glyph, against
a search that grows superlinearly:

```
glyph   with cluster   without   code with   code without      Δ
A@32         131.8 ms   38.4 ms    126,745       126,463   282 B (0.22%)
O@32         658.8 ms  182.8 ms    367,846       367,564   282 B (0.08%)
8@32       2,101.6 ms  549.0 ms    735,238       734,956   282 B (0.04%)
```

With clustering bypassed the whole `pixelflow-graphics` suite is **171/171
green in 25.9 s**, `glyph_atlas_golden` included — so it is a pure
optimization, not load-bearing, and `glyph_atlas_coverage_is_unchanged`
alone goes **31.5 s → 9.8 s (3.2×)**.

**The missing bound is on the search, not on the guard.**
`MISPREDICT_PENALTY_CYCLES` bounds whether a guard *pays at runtime*, and
`guards.rs` argues carefully for it ("bounding the downside by the upside is
enough to keep the analysis honest without a tuned number anywhere"). Nothing
bounds the *search that looks for guards*. Its cost is compile-time and
superlinear in schedule length; its benefit is runtime and proportional to
trips × cycles saved. For a glyph baked once into an atlas at 32×32, a guard
can save at most microseconds — against 1.5 s of searching for it.

Both terms are already computable at the call site: schedule length is `n`,
and trips come from the `LatticeShape` the kernel is compiled for. So this is
a bound to *derive*, not a threshold to tune — which matters, because this
area has already produced three constants whose stated derivations did not
survive measurement (`EXTENT_SLOP`, `CLUSTER_ROUNDS_PER_SELECT`,
`DISC_BAND_ULPS`). **Do not fix this with a size cutoff.**

See **H5**.

**Correction (same day).** This section first said "D5 deletes the question:
with a demand predicate, values of equal demand are contiguous *by
construction*, so there is nothing to cluster." That is **false**, and the
tree already knew it. `pixelflow-codegen/src/emit/demand.rs` computes the
predicate today and refutes the scheduling claim in its own module docs, with
two counterexamples pinned by tests:

- **A select arm breaks superset.** For `S = Select(m, a, b)` at the root,
  `demand(S)` is `true` and `demand(a)` is `m`, yet `a` produces `S`. Sorting
  weakest-first puts the select before the arm it consumes.
- **A value shared across both arms breaks subset.** `demand(p) = m ∨ ¬m`
  while its consumer in the true arm has demand `m`, so strongest-first puts
  the consumer before `p`. That is what CSE across arms produces — the common
  case, not a corner.

Its conclusion, verbatim: *"demand orders neither way on its own, and a
schedule keyed by it is not topological. What survives is demand as a
property … Making regions contiguous remains real work, which is what
`cluster_select_arms` is, and this module does not delete it."*

So the clustering is not an accident of a missing sort key, and D5 does not
dissolve it. What is unbounded is the **search**, not the need.

| | what | where |
|---|---|---|
| **H1** | **S3 — one program for the font.** Font-wide extent, table padded with monoid identities, so every glyph compiles to the same program and a glyph becomes a table write. 95 compiles → 1. With H2 measured, this is the whole hump. | [glyph-as-a-fold-execution](plans/2026-09-09-glyph-as-a-fold-execution.md) §S3 |
| **H2** | ~~Split the 331 ms between compile and collapse.~~ **Done** — see above. | — |
| **H3** | **Hash-consing in `ExprArena`.** Prototyped and measured: arena 2,721 → 154 nodes, 2.1–2.2× on the glyph suites, extracted kernel unchanged. In flight (JP). Lands on the compile half, so it compounds with H1 rather than competing. | [exprarena-on-dag](plans/2026-09-09-exprarena-on-dag.md) §5.2 |
| **H5** | ~~Bound the guard search by what a guard can pay.~~ **Search killed** (`3a4c4e3`): `cluster_select_arms` is one unconditional pass — partition every select worth guarding and not already contiguous, outermost first, each once. `MAX_CLUSTER_ROUNDS`, `is_improvement` and `guarded_spans` went with the hill-climbing. **31.5 s → 20.0 s** on the 95-glyph atlas *with guards kept* (the 9.8 s figure above is what the optimization is worth, not a target — it comes from discarding them). `MISPREDICT_PENALTY_CYCLES` stays: one comparison, and measured, not tuned — a glyph's coverage mask is 3.6× *slower* guarded. **What remains:** `select_arms` is recomputed once per partitioned select, O(selects²·n). A single stable sort keyed by arm ownership would be one pass. But see **D6/D7** — the decision belongs in the e-graph, and optimizing this further is polishing a reconstruction. | [one-conditional-three-lowerings](plans/2026-09-08-one-conditional-three-lowerings.md) §8 |
| **H4** | **Ask B — hoist binder-only work out of the pixel loop.** ~~On the hump.~~ **Demoted by H2**: it optimizes *collapse*, which is 0.2% of a bake, and a glyph bakes once into the atlas and is a gather forever after. Still real for per-frame kernels that are not atlas-cached; not the terminal's startup problem. **Do not patch `contains_gather`** (N1) and do not write a new hoist (N5). | [a-glyph-is-a-circle](plans/2026-09-09-a-glyph-is-a-circle.md) §B |

S3's own doc calls itself "a trade, not a win — fewer compiles against
evaluation of rows that contribute nothing." For the **atlas** path that is
too pessimistic: a glyph bakes once into texels and is a gather forever after,
so the padding is a one-time bake cost, not per frame. Worth re-deciding when
H1 is picked up.

## Names and binding time

| | what | where |
|---|---|---|
| **N1** | **One kind of name.** `Var`/`Uniform`/`Buffer`+`Gather`/`Ref` are five spellings of "bound later", differing only in *when*. Reframes L6 from "delete `Gather`" to "there is one kind of name", with `Gather`'s disappearance a consequence. Blocks a principled H4. | [one-name-bound-later](plans/2026-09-10-one-name-bound-later.md) |
| **N5** | **A kept structure is control flow.** A surviving `Ref` is a call, a surviving `Reduce` is a loop, a `Select` with a derived range is a domain split — one rule, three structures, and today all three are flattened unconditionally so the cost model never chooses. The hoist is *already* shared machinery (`partition_by_scope` is generic over binders); it is handed `[0, 1]` because no fold binder survives to schedule time. **`partition_by_scope(.., &[0, 1, 4])` is the whole of H4** once one does. | [a-kept-structure-is-control-flow](plans/2026-09-10-a-kept-structure-is-control-flow.md) |
| **N2** | **L4 — `Ref(k) ⟷ body(k)` as a growth-gated e-graph rule.** The gate exists (`EGraph::predicted_growth`, asserted against measured delta over 10,819 applications); the rule does not. Its case is a scene composing many *identical* kernels, not construction-side sharing (L3 settled that). | [composition-is-linking](plans/2026-09-09-composition-is-linking.md) §3, §5.1 |
| **N3** | **L5 — a surviving `Ref` is a call.** Second function, coordinate ABI, a register-allocation boundary the allocator does not model. Open question: whether a *tabulated* `Ref` (a leaf that emits a load) avoids all of it, which would let N1 land the cheap half first. | [composition-is-linking](plans/2026-09-09-composition-is-linking.md) §5.2 |
| **N4** | **L6 — a tabulated kernel is a `Ref` with a cached tabulation.** Subsumed by N1; kept as a row because the task list and several commit messages name it. | [composition-is-linking](plans/2026-09-09-composition-is-linking.md) §4 |

## Demand and the conditional

| | what | where |
|---|---|---|
| **D1** | ~~mask ⟹ index range, symbolic tier.~~ **Landed.** `pixelflow_ir::mask_support` — axis-aligned literals in either operand order, conjunctions (intersect), disjunctions (hull), everything else the full extent. No lowering; production behaviour unchanged. 13 tests: soundness by containment over the whole extent in `pixelflow-ir/tests/mask_support.rs`, usefulness against the real `Kernel` builder and `grid_range` in `pixelflow-core/tests/mask_support_of_a_built_kernel.rs`. **A bare `Var` is load-bearing**: the cell grid samples pixel *centres*, so its arena holds `Add(Var, 0.5)` and the analysis widens; teaching `axis_of` to see through that `Add` without moving the shift into the literal deletes a row of pixels, and a test pins it. | [one-conditional-three-lowerings](plans/2026-09-08-one-conditional-three-lowerings.md) §8 |
| **D2** | Lowering 1 — emit the split: select over the derived range, root specialized at `m ≡ false` over the complement. | *ibid.* |
| **D3** | Bind-time tier, and splitting `IndexRange` into a derived region and a requested band. | *ibid.* |
| **D4** | Interval evaluation, target-aware and rounding outward. Unlocks glyph supports, which the symbolic tier cannot reach (a compound glyph's affine mixes X and Y). | *ibid.* |
| **D5** | Lowering 2 on the general predicate — the superseded demand plan's §1–§2, as the third case rather than the whole subject. **Does not delete `cluster_select_arms`**: `emit/demand.rs` already computes the predicate and disproved that plan's scheduling claim (see the correction above). What it buys is *more* exclusivity than `guards` finds — the per-select `demand_exclusive` vs `exclusive` gap under `PIXELFLOW_GUARD_TELEMETRY` — and `demand.rs` says outright that this gap, not the scheduling claim, is what C1b should be justified by. Read that measurement before starting. | *ibid.* |

| **D6** | **A static demand fraction in the extraction cost** (the demand plan's C2a). Extraction is additive per node; the demand-aware cost is `cost(node) · P(demanded)`, with `P = 1` where a node is unguardable. Where the demand is a row-uniform mask with a known extent, `P` is **static and already in the program** — a segment gated on `y_lo ≤ Y < y_hi` over a 45-row glyph is demanded on `(y_hi − y_lo)/45` of rows. This is what D7 needs and what nothing today supplies. | [demand-is-a-dag-property](plans/2026-09-07-demand-is-a-dag-property.md) §4 |
| **D7** | **`Guard` in the graph, and the sink rule** (C2b) — *guarding becomes the e-graph's decision instead of codegen's.* Today it is neither: `analyze_select_guards` runs on the linearized schedule, after extraction, and the emitter places branches from it. Worse, the graph pulls the other way — `SelectHoistUnary` rewrites `Select(m, f(a), f(b)) → f(Select(m, a, b))`, hoisting shared work *out* of arms, which is right for op count and exactly wrong for guarding when `f` is expensive and `m` is coherent, and **no term in the extraction cost opposes it**. Denote `Select(m, a, b)` as `Guard(m, a) ⊕ Guard(¬m, b)`, add the inverse sink rule, and let the cost decide. Needs D6 for the term. Composes with N1: `Guard(m, Ref(a))` carries its arm as a unit, which is what makes H5's partition *unsayable* rather than merely cheap. | [demand-is-a-dag-property](plans/2026-09-07-demand-is-a-dag-property.md) §4 |

D1 → D2 unblocks **S2**: deleting `cells`, `contour_bounds`, the `Union`
plumbing, `TEXT_CELL`, `min_of`, `may_be_interior` and `chord_winding` —
roughly 800 lines to 150 — and makes H1's padding free.

## The e-graph

| | what | where |
|---|---|---|
| **X1** | ~~Split the ~1,990 ms that is not saturation.~~ **Done** — see below. The suspicion recorded here (linear-scan regalloc) was wrong; it is `guards::cluster_select_arms`. | — |
| **R0** | ~~Codegen cannot *say* "here".~~ **Done.** Every branch was an opaque per-backend fixup token the emitter carried by hand to a `patch_branch`, against an offset it read off `code.len()` at the one point that was correct. A program is now a sequence of items — instruction, **binding** of a name to this position, **reference** to a name — with `assemble_labeled` for a program that is a value and a streaming `Resolver` for an emitter whose verbs are `&mut self` methods. Both emitter loops (the collapse nest, the `Select` short-circuit) moved; `IsaBackend::Branch`, `emit_jump`, `patch_branch`, the two `emit_skip_if_all_*`, `Aarch64Branch`, `Cond19`, `Rel26`, `emit_jmp_rel32` and `patch_rel32` are gone. No emitted byte changed. | [a-surviving-reduce-is-a-loop](plans/2026-09-10-a-surviving-reduce-is-a-loop.md) §R0 |
| **R1–R3** | **A surviving `Reduce` is a loop.** The emitter is handed 34,993 straight-line instructions for one glyph because `ExpandReduce` unrolls unconditionally and then gather/transcendental expansion multiplies each copy ~4×; `emit` is ~O(n^1.6) in that. A loop makes the program the *body* — ~1,030 entries — and the accumulator lives in a **slot**, exactly as `emit_collapse_loop` keeps its coordinate, so nothing is live across the back edge and `LinearScan` needs no change. **This is the hump's root cause, not H1.** R0 landed; R1's shape is now read off the code rather than guessed: the region is `partition_by_scope(.., &[4, 0, 1])` (binders are innermost-first, so the fold binder goes at the *front* — the plan's earlier `[0, 1, 4]` was backwards), a fold region is shaped like a guard region, and the binder can be its own counter so no new GPR is needed. | [a-surviving-reduce-is-a-loop](plans/2026-09-10-a-surviving-reduce-is-a-loop.md) |
| **E1** | **Geometric `SplitFold`.** *(= R4; inert before R2, since split and unsplit price identically today.)* `⊕_{[lo,hi)} = ⊕_{[lo,mid)} ⊕ ⊕_{[mid,hi)}`. Needs no substitution (both halves share the body e-class) and is *exactly* cost-preserving in both extraction arms, so it cannot desync the claim/price audit. Bisect at the midpoint to bound growth at 2n−1. `EmptyFold` already exists as its base case. | — |
| **E2** | **A critical-path term in the cost model.** Without it E1 buys reachability and no speed: `CostModel::latency_prior()` sets `depth_threshold: 1024, depth_penalty: 0` ("effectively disabled"), so a 34-deep serial chain and a 6-deep balanced tree price identically. A *global* depth hinge is the wrong shape — what is wanted is the reduction's critical path. | [schedule-cost-model-denotation](plans/2026-09-01-schedule-cost-model-denotation.md) |
| **E3** | Extraction: `shared_dag_dp_pass` is O(L²) — make reach tracking sparse. | — |
| **E4** | ~~Saturation rescans every class with every rule every iteration — dirty tracking.~~ **Already built**, and this entry was stale the day it was written: `EGraph::class_is_dirty` with per-rule `rule_last_swept` baselines and a 2-hop forward-neighbourhood check (`DIRTY_TRACKING_MAX_DEPTH`), wired into the scan so a clean class skips the clone and every `apply`. What remains is narrower and, per X1, **not hump work**: the scan still visits every class for every rule and filters, and `class_is_dirty` is itself a neighbourhood walk rather than O(1), so the cheap path is O(classes × rules) per iteration. A dirty *worklist* would make it O(dirty). Saturation is a constant ~32 ms of a 2,154 ms compile, so this needs a reason other than the hump. | — |
| **E5** | Extraction is not monotone in graph richness: the same kernel in a superset graph can extract a worse DAG. | — |

## Correctness and CI

| | what | where |
|---|---|---|
| **C1** | **Trig range is unasserted.** `pixelflow-ir/tests/trig_range.rs` made its claims through the deleted interpreter and went with it. The property is unchanged; an out-of-range `sin` would now ship green. Needs rebuilding on the JIT. | CLAUDE.md, "Precision is on the table; range is not" |
| **C2** | **The `'8'` waist bug is open on `main`.** Five fixes tried and refuted; `freetype_oracle.rs` pins it rather than fixing it. The general demand predicate (D5), not a sixth per-select patch, is the intended next attempt. | [demand-is-a-dag-property](plans/2026-09-07-demand-is-a-dag-property.md) §9 |
| **C3** | `CachedText::kernel` sums glyph coverages, so overlapping glyphs can exceed 1. Not on any production path — `core-term` renders through `GlyphAtlas`, and `CachedText` has no non-test caller. | — |
| **C4** | A corpus needs a new acceptance criterion before `gen_bench_corpus` can come back; its quarantine gate compared against the interpreter. | CLAUDE.md, "Cost Model and the Guide" |
| **C5** | A local CI runner, so a full presubmit does not cost a push. | — |
| **C6** | **A transient cache-service error fails a whole job before it compiles anything.** Every job sets `RUSTC_WRAPPER: sccache` with the GitHub Actions cache backend, so sccache is a hard dependency of `rustc` rather than an accelerator: on 2026-09-10 a DNS failure reaching `productionresultssa*.blob.core.windows.net` killed `ISA matrix` at `rustc -vV` with `compile_requests: 0`, on a docs-only commit whose parent had passed the same job. sccache offers `SCCACHE_IGNORE_SERVER_IO_ERROR=1` for exactly this — degrade to no cache instead of failing. A spurious red costs a cycle and, worse, teaches everyone to re-run reds without reading them. | — |

## Housekeeping

- `Kernel::parts()` hands out the **unlinked** fragment, and five measurement
  consumers each learned to link first. Right division, five copies of one
  line. ([composition-is-linking](plans/2026-09-09-composition-is-linking.md) §7)
- **This file is drifting from its own rule.** "The hump" is now several
  screens of narrative where the preamble says index-and-status. The
  measurements belong in `docs/results/`; the section should be four rows and
  a pointer.
- `cells` / `text_union` reach only one Criterion bench; nothing on screen has
  ever gone through them. Delete with S2, not before — they are the worked
  example of a domain-side extent.
  ([glyph-as-a-fold-execution](plans/2026-09-09-glyph-as-a-fold-execution.md) §S2)
