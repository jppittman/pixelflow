# Collapse is a fold

## Metadata
- **Author**: JP (design), Claude (draft)
- **Status**: `In progress` — denotation settled 2026-09-17; implementation in
  the order of §5, one PR per row. Step 1 landed (#1277); step 2 in review.
- **Created**: 2026-09-16 (as "a uniform read is one load"; rewritten
  2026-09-17 around the decision below)
- **Verified against**: `93d48c86` (main with #1268, the re-land of the
  surviving fold, and #1271)
- **Continues**: [a-surviving-reduce-is-a-loop](2026-09-10-a-surviving-reduce-is-a-loop.md)
  (its "nested fold loops" remainder) and
  [a-kept-structure-is-control-flow](2026-09-10-a-kept-structure-is-control-flow.md)
  (a surviving `Reduce` is a loop — this plan makes the lattice's loops
  surviving `Reduce`s).

**Decision it records (JP, 2026-09-17):**

> *"I'd literally insert folds to iterate over the lattice. … I'd add a
> `pub(crate)` write instruction and use literally the existing machinery for
> everything else. Same as the register allocator. No special cases."*
>
> *"This is actually a legalize pass."*
>
> *"Why can't lanes be a scope too?"*

So: the collapse loop stops being a scaffold the emitter builds around a
kernel and becomes three folds the kernel is wrapped in. A lattice's rows, its
batches and its lanes are three binders of three ordinary `Reduce`s; the one
new instruction is a store; the loop nest, the hoisting, the broadcast-versus-
gather choice and the tail are all consequences of machinery that already
exists for a kernel's own folds. It is a subtraction.

---

## 1. Why, with the numbers

`8` at tile 32, main at `04d537ba` (before #1268), the JIT for the AVX-512
tier, disassembled with `objdump -D -b binary -m i386:x86-64` and
histogrammed by mnemonic. 48,452 instructions, 354 KB, 2.1 s to compile in
release. Where they go:

| what | instructions | per piece (34) |
|---|---|---|
| table reads (`vgatherdps` + `kmovw` + `vcvttps2dq`) | 4,830 | 47 reads |
| address arithmetic on those reads (`vrndscaleps`, `vminps`/`vmaxps`, the `vmulps`/`vaddps` of `row·22 + col`) | ~13,000 | ~380 |
| constant materialisation (`mov` + `vbroadcastss`) | ~21,000 | ~620 |
| the Loop–Blinn arithmetic itself (`vfmadd`, `vsubps`, `vcmpps`, `vpternlogd`, `vandps`) | ~3,000 | ~90 |

The geometry is 6% of the program. Same shape at the SSE2 tier (573 KB, 508
spilled values), where each read is additionally 13 instructions of scalar
loads and inserts; the mnemonic counts are internally consistent (1,610
`vpextrd`/4 = 1,610 reads = 4,830 `vinsertps`/3), so they are the reads.

Three facts, and under this plan one owner:

1. **The reads exist 34 times because the winding fold is unrolled.** It is
   nested inside the distance fold, and `extract_folds` carves one level
   (`pixelflow-codegen/src/emit/mod.rs`, the assertion naming "2c handles one
   level"), so `expand_nested_reduce` unrolls it. Nothing after extraction
   folds the substituted addresses — legalize-last is on purpose — and
   `expand_reduce`'s doc claim that "the emitter folds their addresses to
   immediates" is false and always was.
2. **A lane-uniform read is emitted as a per-lane gather.** Every table read's
   row index depends only on a fold binder, so it is the same in every lane,
   and `compute_arena_variance` already says so. The emitter has one lowering
   for `RawGather` (`emit_gather_scalar`: 13 instructions per four lanes on
   SSE2, `vgatherdps` on AVX-512, four scalar loads on aarch64) and one for a
   lane-uniform value (`emit_uniform_load`, a `mov` and a `vbroadcastss`) that
   only `Uniform` leaves reach. The variance bit is computed, handed to the
   hoist planner, and never consulted by the gather lowering.
3. **Reads never hoist.** `plan_collapse_hoist` refuses anything under a
   gather, on the comment "winding kernels are gather-free", which stopped
   being true when [glyph-as-a-fold-execution](2026-09-09-glyph-as-a-fold-execution.md)
   made the glyph two folds over a table.

Facts 2 and 3 are the same defect: the collapse loop is built by a scaffold
(`emit_collapse_loop`, `Level`, `CollapseBody`, the frame/row/body tiers,
`plan_collapse_hoist`) that is *not* the fold machinery, so every property
the fold machinery gets from the DAG — which scope a value belongs to,
which binders it depends on — the scaffold has to recover by hand, with its
own axioms (`Var(0)` is lane-varying) and its own carve-outs. Fact 1 is the
fold machinery stopping one level short. All three go when the lattice is
folds and there is one nest.

### 1a. The readings this plan refused

Recorded so the next reader of the histogram does not re-derive them.

- **A constant-fold-and-CSE stage after `legalize`**, and **a `Row(buf, i,
  col)` node** whose lowering carries no floor and no clamp. Refused
  2026-09-16: *"the advantages of not having memory reads in the language
  outweigh denoting this … I don't think it's gonna be possible for pixelflow
  to be slow off of wearing out the ALU."* Both mistook a symptom of
  unrolling for a property of reads; a second optimizer behind the e-graph is
  the shape [a-kept-structure-is-control-flow](2026-09-10-a-kept-structure-is-control-flow.md)
  argues against, and a memory-shaped node leaks what
  [one-name-bound-later](2026-09-10-one-name-bound-later.md) is removing.
- **Fixing the two emitter sites in place** (delete the `contains_gather`
  carve-out; split `ScheduledOp::Gather` on the X bit). Correct as far as it
  goes, and the first draft of this document. Refused 2026-09-17 because it
  keeps the scaffold: two nests, two notions of scope, and a "loop order"
  question that only exists because the collapse loops are not folds.
- **A `Lane` leaf.** The pack pass could rewrite the batch fold's body as
  `i := i + Lane` with `Lane` a constant `[0, 1, …, L−1]`. That is a lane
  fold with its binder inlined and its scope forgotten: lane-uniformity would
  need an analysis of its own, a horizontal reduction would be unsayable, and
  the remainder of a width would be a scalar loop re-evaluating the body per
  lane. A binder is already a leaf whose scope the machinery knows. See §2.2.

---

## 2. The denotation

### 2.1 Three folds and a store

`collapse` of a kernel `f` over a lattice of extent `w × h`, tabulated into a
plane `out` whose rows are `pitch` elements apart, is

```text
collapse(f) = fold_{j ∈ [0,h)} fold_{k ∈ [0,w−r) step L} fold_{l ∈ [0,L)}
                 Write(out, j, k, l, f(x0 + k + l, y0 + j))
            ; fold_{j} fold_{l ∈ [0,r)} Write(out, j, w−r, l, f(x0 + (w−r) + l, y0 + j))
```

with `L` the lane count of the target, `r = w mod L`, and the second line
present only when `r > 0`. Each `fold` is an ordinary `Reduce` over the
**unit monoid** `SEQ` (combine is sequencing, identity is nothing); `Write`
is the body's root; `;` is `SEQ`'s own binary op. `x0`, `y0` are the region's
origin — pixel centres plus offset, what `Field::sequential(x0)` carries
today — and are uniforms. The extent is static: it is the shape the kernel
is compiled for, and `Manifold::compile(extent)` already keys the JIT cache
on it (`jit_cache::same_kernel_at_two_extents_is_two_entries`).

This is `Kernel::at` — `f.at(x0 + k + l, y0 + j)` substitutes into `Var(0)`
and `Var(1)` exactly as every warp does — plus two folds, plus one strip-mine.
After it no `Var(0..3)` exists in the arena; only binders. The magic ranges
(`Var(0..3)` an axis, `Var(4..)` a binder) go with them.

### 2.2 Lanes are a scope

The lane fold `fold_{l ∈ [0,L)}` is a `Reduce` like any other: its binder has
a variance bit, values that do not depend on it are placed outside it by the
same per-scope placement (`partition_by_scope`, `NestAllocation`) that
places every value, and its body is the kernel. What differs is execution:

> **The one rule.** The fold whose binder a `Write` names as its `lane` is
> executed by lanes. Its binder materialises as the constant
> `[0, 1, …, len−1]`; it has no counter and no back edge; its body is emitted
> once. Every other fold is a loop.

Everything the scaffold recovered by hand falls out:

- **Lane-uniform is "does not depend on `l`".** A table read whose address
  carries the piece binder but not `l` is one load, broadcast. That is the
  variance bit the emitter already computes, read at `arena_to_schedule`
  where `Gather` becomes either a gather or a broadcast load — decided once,
  where the DAG is read, not flagged per backend.
- **The three hoist tiers are three scopes.** Outside `fold_j` is the frame
  tier; inside `fold_j` outside `fold_k` is the row tier; inside `fold_k`
  outside `fold_l` is a fourth tier the scaffold never had — per batch,
  lane-uniform — which is where a binder-indexed read's address lands.
  `plan_collapse_hoist` is placement, and placement exists.
- **The kernel's own folds nest inside the lane scope**, as they execute
  today: a glyph's piece fold runs per batch with the lane binder live, its
  terms vectors across lanes. A value inside it that does not depend on `l` is
  a broadcast, by the bit, wherever it is placed.
- **The remainder is the same fold, shorter.** `fold_{l ∈ [0,r)}` executed by
  lanes computes `L` lanes and stores `r`: a masked store where the ISA has
  one (AVX-512 `k`-mask, AVX2 `vmaskmovps`), `r` extract-and-store
  instructions where it does not (SSE2, NEON). No scratch batch, no overhang,
  no `RowTail`, and at most `L−1` lane-stores per row. Glyph tiles are lane
  multiples at every ISA level and never take it.
- **A lane fold over a value monoid is a horizontal reduction.** Not needed
  by anything; refused by the emitter until something needs it. It is named
  here because it is what the vocabulary now says, and a design that lets a
  true thing be said is the check that the vocabulary is right.

What is *not* claimed: that a value placed inside the lane scope is
lane-varying. Placement is "outermost scope binding all its binders"; a value
depending on the piece binder sits in the piece fold's scope, inside `l`,
and is uniform by its bit. The bit and the scope are two facts, both already
computed.

### 2.3 Two legalize passes

Both run in `pixelflow-ir`'s `legalize`, after extraction, so the e-graph
stays blind to the lattice exactly as it is today:

```text
expand_refs → lower_dwrt → collapse(extent) → pack(L) → expand_gather → expand_transcendentals
```

- **`collapse(extent)`**: `f ↦ fold_j fold_i Write(out, j, i, f.at(x0 + i, y0 + j))`.
  Target-agnostic; `L` does not appear.
- **`pack(L)`**: strip-mine `fold_i` into `fold_k fold_l` with `k` stepping by
  `L` (`Fold::stride`, the field HalveFold already uses) and `i := k + l`;
  when `r > 0`, sequence the remainder fold after it. `L` is the pass's one
  parameter, handed in by `pixelflow-codegen`; `pixelflow-ir` never names a
  width. HalveFold is this transform with `L = 2` and the inner fold
  unrolled; unifying them is a later subtraction, not this plan's.

`lower_dwrt` precedes `collapse` so `Dwrt` is taken with respect to `Var(0)`
before it is substituted; the substitution's chain-rule factor is 1.
`expand_gather` follows so a read's address arithmetic is built over the
binders and its variance is read off them. `expand_nested_reduce` is
deleted — §5 step 1 — because every kernel fold is now nested at depth three.

### 2.4 Post-legalize vocabulary

Three words, constructible only by these passes, refused by the e-graph and
by `kernel!`:

| word | what | denotation |
|---|---|---|
| `Write { out, row, col, lane, value }` | `ExprNode`, `pub(crate)` | store `value`'s first `len(lane)` lanes at `out + 4·(row·pitch + col + lane)` |
| `OpKind::Seq` | binary, unit-typed | evaluate the left, then the right |
| `Monoid::SEQ` | `Monoid(OpKind::Seq)` | the unit monoid |

`Write` names binders, not an address expression, so contiguity along
`lane` is by construction and needs no analysis. Its store width is the lane
fold's `len()`, so it carries none. A `SEQ` fold goes through the fold
emitter unchanged — `alu(Seq)` emits no bytes — and the accumulator slot it
would allocate is one dead vector of stack until a measurement says
otherwise; that is what "no special case" costs, and it is cheap.

`out` and `pitch` are call arguments: the collapse ABI becomes
`fn(ctx, out, pitch)`. `Point4` and `TileSlice` go.

### 2.5 Static extents, and what that costs

A region is a shape. Today `collapse_rows(region)` collapses any
sub-region of the compiled extent with bounds latched at run time; under this
plan the bounds are the fold's, and a band of a different height is a
different kernel. That is the direction CLAUDE.md already states ("a trip
count that must change is a recompile through the shape-keyed cache") and
`manifold.rs`'s own `plan` anticipates ("becomes load-bearing when the
emitted code specializes on the extents").

The cost is bounded by one fix in `pixelflow-search`'s `optimize_runtime_arena`.
Saturation does not depend on the shape; extraction does
(`Optimizer::for_lattice` prices a node by how many times the loop nest
evaluates it), so the cache holds the **saturated e-graph per structure** and
extracts per shape — which is what that function's own doc prescribed. It
also keys on structure rather than identity, so buffer- and uniform-bearing
arenas — every glyph, the cell grid — stop bypassing the cache entirely, as
they did. A second band height then costs an extraction and an emit
(milliseconds), not a saturation (seconds). The JIT cache stays keyed on
`(arena, shape)`. A frame stripes into at most two heights; a glyph is one
tile; a `Union`'s pieces are already per-piece collapses.

*(The first draft of this section said "drop the shape from the key", on a
stale comment claiming the cache's output was shape-independent. It is not:
the extraction is priced by the lattice, and the split above is the fix.)*

The alternative — a dynamic bound on `Fold` — is a type extension and an
emitter case for one caller's convenience. Refused.

---

## 3. What it subtracts

**`pixelflow-codegen`**: the collapse section of `IsaBackend` (`scaffold_anchor`/`finish`,
`latch_bounds`, `store_result`, `advance_out`, `add_scalar`, the coordinate
`slot_store`/`slot_load` dance, `emit_collapse_loop`, `emit_nest`, `Level`,
`CollapseBody`, `Counter::Row`/`Batch`, `OutStep`, `COORD_SLOTS`,
`INPUT_COORDS`, `SLOT_ROW_START_X`); `plan_collapse_hoist` and
`contains_gather`; the frame/row/body trichotomy in `EmitTraffic` and
`partition_by_scope(.., &[4, 0, 1])`'s hard-coded axis order; the
`Var(0..3)` checks in `compile`; `expand_nested_reduce`'s reason to exist.

**`pixelflow-ir`**: `Var(0)`/`Var(1)`'s status as axioms of `variance`
(they remain the pre-wrap free variables and are never set after
`collapse`); the `is_bitwise_domain`-style recovery of "which scope" from
magic index ranges.

**`pixelflow-core`**: `BandPlan`'s tail, `RowTail`, the scratch batch and its
per-row call, `Point4`, `TileSlice`; `collapse_subrect` and
`collapse_int_subrect` become `collapse_rows`/`collapse_int_rows`, since
every collapse is now exact.

Nothing is added to the surface language. `Kernel::x()` is still `Var(0)`;
a kernel still knows nothing about lanes, rows or memory.

---

## 4. What this plan does not do

- **The e-graph does not see the lattice.** Wrapping happens after
  extraction. Interchanging a kernel fold with a lattice fold — the
  pieces-outer glyph, `fold_piece fold_j fold_k fold_l Write(out, gather(out)
  ⊕ term_piece)`, a reduction into the output buffer — is an equation of the
  effect language and E6's to price. Not here.
- **No dynamic fold bounds.** §2.5.
- **No `Row`, no post-e-graph pass, no `Lane` leaf.** §1a.
- **No change to `Gather`'s meaning.** `buf[clamp(⌊y⌋)][clamp(⌊x⌋)]`; the
  floor and clamps survive under a loop as a few vector ops per trip.
- **No bucketing changes.** #1270's padding is orthogonal.

---

## 5. Order, and how each step is measured

One PR per row. Each lands green on its own; none needs the next.

| step | what | gate | number that should move |
|---|---|---|---|
| 1 | ~~**Nested fold scopes.** `extract_folds` carves a fold inside a fold's body; `expand_nested_reduce` deleted. 2c's stated remainder.~~ **Done** (#1277). A nested fold that does not depend on the enclosing binder is hoisted: it stays the enclosing scope's fold, run once, and the fold reading it keeps its def as a placeholder parked in the accumulator slot — the glyph's winding sum runs once per batch, not once per piece. | goldens; `run_is_a_glyph`; `font_rasterization_regression` at every ISA level; `traffic` | `8`@32, this host's tier: 536,960 B → **12,390 B**; body 24,738 → **612** instructions; table reads (`vcvttps2dq`) 1,610 → **35**; spills 19 → 5. `A`@32 is the identical program: the body no longer scales with the piece count. |
| 2 | **One saturation per structure, one extraction per shape.** `optimize_runtime_arena` holds the saturated e-graph, keyed on structure (never identity), and extracts per shape; the extracted term comes back in the caller's own names and slot order (`ExprArena::with_tables` + `relink`). `Optimizer::run` split into `saturate_term` and `extract`. | `one_compile_per_shape` counts saturations across shapes and compositions; a second composition keeps its own names; a second composition at a second shape reads its own table end to end | saturations per structure: 1, whatever the shape or the composition. Measured: the 95-glyph atlas at tile 32 after tile 16 saturates 1 structure (its one new bucket), not 7; warm 0.41 s → 0.31 s |
| 3 | **Vocabulary.** `Write`, `OpKind::Seq`, `Monoid::SEQ`; `Variance` widened past `u8` and `REDUCE_BINDERS` past 4 (three lattice binders plus a kernel's own — the control plane is 64-bit). | unit tests; no production path changes | — |
| 4 | **The two passes.** `collapse(extent)` and `pack(L)` in `legalize`, on an arena the emitter does not yet accept. | arena-level tests: structure, remainder for `w = qL + r`, every binder's bit, a read's uniformity | — |
| 5 | **The emitter executes the lane fold and the `Write`; the scaffold is deleted; the ABI is `fn(ctx, out, pitch)`; `pixelflow-core` collapses in one call.** | goldens at every ISA level (`isa-matrix --smoke`); `bind_allocates_nothing`; `traffic`; distinct shapes per terminal frame counted before landing | collapse time; body instruction count; `loads_kept` by scope |
| 6 | **Broadcast load** for a `Gather` whose address lacks the lane bit, split at `arena_to_schedule`. | goldens; `avx512_evex_proof` | `vgatherdps`/`vpextrd`/`vinsertps` → 0 on glyph programs |

Step 5 cannot be split without keeping two collapse paths alive, so it is
the one large PR; steps 3–4 make its diff the emitter's alone. The histogram
of §1 is the instrument for 1 and 6: compile through `compile_as_baked`,
write `code.as_bytes()`, count mnemonics.

---

## 6. Open questions

- **Do the glyph's two folds bind the same binder?** If so the e-graph already
  hash-conses their shared column reads; if not the duplication (47 reads per
  piece for 22 columns) arises before unrolling. Either way H3 folds it; the
  count after step 1 tells which.
- **Where the write's address is computed.** From the row and batch counters
  (`lea`-shaped, two GPR ops per store) or as an induction pointer stepped at
  each scope's close. The emitter's choice; whichever is fewer lines.
- **`Fold`'s `u16` ends.** A frame width fits; the control-plane rule says
  widen anyway, and step 3 is where.
