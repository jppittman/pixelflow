> **Retracted/Superseded (2026-09-07), ledger L082.** Section 9's reading that a tight Y-extent gate fixes the '8' waist is withdrawn - an always-true gate fixed it equally; the bug is open on main and belongs to the font thread, out of this program's scope. Verdict and rationale: `docs/results/2026-09-07-claims-ledger.md` (PR #1207); the corrected benchmark and re-validation order: `docs/plans/2026-09-07-benchmark-correction.md`.

# Demand is a property of the DAG

**Date:** 2026-09-07
**Status:** **Superseded** by
[2026-09-08-one-conditional-three-lowerings.md](2026-09-08-one-conditional-three-lowerings.md).
Do not start C1b from this document. Only stage C1a executed, and its
outcome inverted what this plan expected — see §9 for what actually
happened. What this plan got wrong is its framing, not its analysis: it
treated demand as a codegen concern replacing per-select guard machinery,
when it is a language concern and that machinery is one of three lowerings
of a single conditional. §1–§2 survive as the analysis the successor needs
(they are its §6); §9–§10 remain the record of the glyph investigation. §3
was corrected in place against what is actually on `main` and is kept for
that record.
**Author:** JP (direction), Claude (draft)
**Refines:** §5 of [2026-09-06-lattice-is-the-index.md](2026-09-06-lattice-is-the-index.md)
("exclusivity does not survive the partition"). This plan's premise was that
#1187 would land a small per-select `Exclusivity`/`ArmOwnership`
implementation of that gate for this plan to replace before merging. That
implementation never merged — see §9: five candidate fixes were tried and
refuted, and #1187 shipped test-only, with the glyph defect still open on
`main`. §3's "what this deletes" described machinery that was never built;
it is corrected in place to describe what `guards.rs` actually holds.
**Depends on:** the uniform leaf (#1183) for the frame prologue's shape;
index-range unions (#1184) for the loop-nest destination in §6.
**Decision it records (JP, 2026-09-07):** *"This operands work being
select-specific feels wrong to me. Why isn't the DAG over the whole
program? Why isn't that what we're topologically sorting to find out
dependencies?"* and, on whether to merge the per-select version first:
*"Deleting code is expensive once it's wrong and it's merged. Why not build
the control demand predicate on the DAG and use that?"*

---

## The shape

Every value in a kernel's DAG is observed under some condition. Call it
the value's **demand**:

```
demand(root)                 = true
demand(m)   for S = Select(m, a, b)   ⊇ demand(S)
demand(a)                              ⊇ demand(S) ∧ m
demand(b)                              ⊇ demand(S) ∧ ¬m
demand(v)   for consumers c₁ … cₙ      = ⋁ᵢ demand_edge(cᵢ → v)
```

One backward pass over the DAG in reverse topological order computes it
for every value. It is control dependence, and it is a property of the
graph — not of any select, not of any scope's slice of the schedule.

What #1187 built, `Exclusivity { selects: Vec<ArmOwnership> }`, is the
special case where `demand(v)` is a single literal: the values a select's
arm owns outright. A value read by the true arm of `S₁` and the false arm
of `S₂` has `demand = m₁ ∨ ¬m₂`, and the special case has no way to say
that, so it says "not exclusive" and the value is computed on every batch.
Every rule that follows from the general object had to be re-derived as a
repair in the special case: a value's consumers had to be re-discovered
per select, an arm's entries had to be *made* contiguous by iterative
partitioning (`cluster_select_arms`), and how many rounds of repair to
allow became a constant (`CLUSTER_ROUNDS_PER_SELECT = 2`) in a module
whose doc promised none.

**The invariant that makes demand a scheduler.** For every edge `u → v`
(`u` a producer of `v`), `demand(u) ⊇ demand(v)`: `u` has `v` among its
consumers, and demand is the union over consumers. So an order that lists
values by demand, weakest first, and by topological order within one
demand, is itself a topological order — every producer sorts no later
than its consumer. Values with equal demand are contiguous *by
construction*. There is nothing to cluster and no round count to choose.
This is the answer to "why isn't the topological sort finding this out":
it is the same sort, keyed by one more thing.

## Why now

C1's measurements, all on `O` at 32 px, AVX-512:

- Row-prologue guard telemetry went from `selects=60 guarded=0` to
  `selects=125 guarded=2801`. The mechanism works, and the per-row cost
  halved (0.69 → 0.37 µs/row; `S` 1.07 → 0.54).
- The largest single arm, 1,217 entries under the glyph's
  `unit_square_mask`, stayed unguarded in the row prologue because its
  mask is X-dependent. A Y-only *projection* of that mask would gate the
  rows outside the glyph; the per-select shape has no notion of
  projecting a predicate, so this was filed as a font follow-up.
- Values shared between selects are refused; nested selects are handled
  by nesting the special case; both are the general object's ordinary
  cases.

The adversarial review of #1187 found, in the per-select machinery:

- A skipped *false* arm parks the mask into the arm's root — all-ones,
  which read as `f32` is NaN. Every arithmetic consumer is annihilated
  by the blend, but a gather index is not arithmetic: through a commuted
  `Min`/`Max` a NaN index becomes `i32::MIN`, and the address is
  `base + i32::MIN · 4`. Unreachable with today's masks; forbidden by
  nothing.
- The clustering round cap is chosen, not derived, and the module doc
  says otherwise.
- The frame prologue is allocated against guards that `Trips::Once`
  never emits.

None of these is a bug in the idea. They are the cost of implementing a
graph property as a per-node repair.

## 1. Representation

A predicate is in disjunctive normal form: a set of clauses, each a set of
`(mask: ValueId, polarity)` literals. Disjunction is clause union with
subsumption; conjunction with a literal adds it to every clause and drops
any clause that now holds both polarities of one mask; the empty clause is
`true`.

Predicates are **capped** at a small number of clauses and **widened to
`true`** when the cap is exceeded. Widening is always sound: it can only
lose a guard, never skip a demanded value. The cap is a compile-budget
knob and is documented as one; it is not a correctness bound and must not
be described as derived.

Ordering the distinct predicates that occur (a few dozen in a glyph
kernel) for the sort: subsumption-based implication where it decides,
and the value DAG's own topological order between the two groups where it
does not. The producer-superset invariant is pinned by a property test
over random DAGs, because a representation that breaks it makes the
schedule silently non-topological.

## 2. A guard is a region of equal demand

Not an arm of a select. Within a scope, after the sort, the values with
one demand form one run; a branch over that run tests the predicate on
the batch: per lane, `⋁ᵢ ⋀ⱼ litᵢⱼ` with the mask ops the backends already
have (`and`, `andn`, `or`), then one any-true test. A single literal is
the one test the special case already emits; a conjunction costs one
`and` per extra literal; a disjunction one `or` per extra clause. The
profitability bound is unchanged: the region's latency-prior cost against
`MISPREDICT_PENALTY_CYCLES`. Nested regions are nested predicates and fall
out of the sort.

**Scope projection.** In the row prologue an X-dependent literal is not
yet computed. Drop it — weaken the predicate to its Y-only and
frame-only literals — and guard on the remainder. Weakening is sound.
This is the general form of the `unit_square_mask` follow-up: the 1,217
entries get their rows-outside-the-glyph guard with no font change.

**Parks.** C1's rule stands: a guard that skips a hoisted root's
definition writes that root's park ahead of the branch, so no inner
scope reads a slot nothing wrote. The value written is **zero**. For a
skipped true arm the mask is already zero; for a skipped false arm the
select takes the true operand and the false one is annihilated bitwise
whatever it holds. Zero is equally correct and removes NaN from the data
flow for one `xorps`.

**Allocation.** C1's liveness rule, "a guard site is a read of its
mask", becomes "a guard site is a read of every literal in its
predicate". The frame prologue is `Trips::Once`; guard analysis is gated
on the same `Trips` the emitter uses so nothing is pinned for a branch
that is never emitted.

## 3. What this deletes — corrected against what actually shipped (see §10)

**As drafted, this section assumed #1187 would land a per-select
`Exclusivity { selects: Vec<ArmOwnership> }` implementation of §5's gate for
this plan to delete before merging.** That implementation was never built:
`Exclusivity` and `ArmOwnership` name no type on `main` (verified directly
in `pixelflow-codegen/src/emit/guards.rs`), and #1187 shipped as three test
files with zero production code changes. So there is nothing of that shape
to delete, and "what this deletes" has to be restated against what
`guards.rs` actually holds — which predates *both* this plan and #1187, from
`pixelflow-codegen`'s register-allocation work (#1150) and S3b (#1177).

What is on `main` today, to be replaced by the demand predicate when C1b
lands:

- **`SelectGuard`** — one struct per guardable `Select`, holding
  `select_idx`, `mask_vid`, and a `(usize, usize)` half-open schedule range
  per arm. Not the `Option<_>`-shaped field this draft anticipated; there is
  no absent-select case to special-case away.
- **`SelectArms`** and `select_arms(schedule)`, which compute each arm's
  transitive dependency closure, its schedule-index run, and its
  latency-prior cost (`arm_cycles`) — the per-node piece this plan's demand
  predicate replaces with one backward pass over the whole DAG.
- **`analyze_select_guards`**, which turns each `SelectArms` into a
  `SelectGuard`, and the `Telemetry`/`SelectStat` machinery behind
  `PIXELFLOW_GUARD_TELEMETRY` — the counts §"Why now" cites.
- **`cluster_select_arms`** and its round cap, which is **`MAX_CLUSTER_ROUNDS
  = 8`**, not the `CLUSTER_ROUNDS_PER_SELECT = 2` this draft named (that
  number belonged to the undelivered design, not to anything merged). The
  cap's own doc already says plainly what it bounds — compile cost of the
  search, not correctness — so "a module whose doc promised none" is not
  accurate to the code that exists; it accurately describes the
  never-shipped alternative.
- **`HoistCtx::Prologue { Trips }`**, `MISPREDICT_PENALTY_CYCLES`, and the
  park-write rule referenced in §2 all predate #1187 as well (S3b, #1177);
  none of it is something #1187 "kept," since #1187 touched no production
  file.

Nothing in `ttf_curve_analytical.rs` carries a `y_extent`/`EXTENT_SLOP` gate
on `AnalyticalQuad` — that was approach 1 of #1187's five refuted attempts
(§9), and being refuted, it was never committed to production code.

The shape of the deletion this plan intends is otherwise unchanged: when
the demand predicate (§1–§2) lands, `SelectArms`/`select_arms`,
`analyze_select_guards` in its select-centric form, and `cluster_select_arms`
with its round cap are what get replaced by "compute demand, sort by it,
emit a branch per region that clears the bound." `SelectGuard` itself, or
something isomorphic to it, likely survives as the per-region output shape
that emission and allocation already consume — that decision is for C1b,
which is unbuilt.

## 4. The e-graph is currently against it

`pixelflow-search` has `SelectHoistUnary`:

```
Select(m, f(a), f(b))  →  f(Select(m, a, b))
```

It pulls shared work *out* of arms. That is right for operation count and
exactly wrong for guarding when `f` is expensive and `m` is coherent, and
there is no term in the extraction cost that could oppose it. The
observation in #1184's review that "CSE can break exclusivity" was not an
accident of CSE; the optimizer has a rule whose direction is anti-guard
and nothing weighing the other way.

Two levels, both later stages:

**Cost.** Extraction is additive per node. The demand-aware cost is
`cost(node) · P(demanded)`. `P = 1` where the node is unguardable. Where
the node's demand is a row-uniform mask with a known extent, `P` is
**static**: a segment gated on `y_lo ≤ Y < y_hi` over a 45-row glyph is
demanded on `(y_hi − y_lo)/45` of rows, and both numbers are in the
program. Coherence beyond the static fraction is a data property and is
the "first profile-dependent term" the S3b commit named for the
schedule-cost residual. The static part is available today from the
extents L3 and C1 put into kernels.

**Structure.** Make demand something the graph can hold and rewrite:
denote `Select(m, a, b)` as `Guard(m, a) ⊕ Guard(¬m, b)`. Then the sink
rule is the inverse of `SelectHoistUnary`, applied when
`cost(f) · (1 − P(m))` exceeds the duplicated work; guards with equal
masks merge; and extraction *chooses* a guarded form rather than codegen
recovering one from whatever form was extracted.

## 5. Guards and index ranges are one thing

A Y-only mask `y_lo ≤ Y < y_hi` *is* an index range. L3 put that range
in the loop nest so the pixel is never asked; C1 recovered it from a mask
so the pixel is asked and skipped when the batch is coherent. Same
denotation, two implementations, and demand predicates unify them: a
value whose demand is a Y-range should not be *guarded inside* a loop over
every row, it should be *scheduled into* a loop over those rows. That is
the one-loop-nest refinement §3 of the lattice plan names, and it is why
the row-prologue guard is a bridge and not the destination. The bridge is
worth crossing now because the loop-nest form needs the same demand
computation; nothing built for §2 is thrown away by §6.

## 6. Stages

**C1a — the gate, first and alone.** The `'8'@17` finding from #1187's
review is a live bug on `main`: the runtime e-graph's `MulAdd` fusion
flips `disc.ge(0)` at a horizontal-tangent row and moves coverage by 0.5,
and `optimized_glyph_matches_raw_within_reassociation_noise` cannot see it
because it runs at 32 px only. Extend it to the sizes the goldens use and
the one that bites; it must fail on `main` at 17 px. If it fails on the
branch, the tangent gate is ill-conditioned at `disc ≈ 0` by construction
and is replaced by the tight `y_extent` (which contains the vertex) or
compared against a scaled `−eps`. The `y_extent` commit rides here. Small
PR; lands before anything else.
**Landed as #1187, commit `4fe39607` — but inverted.** The tight-Y-extent
gate's mechanism was disproved rather than confirmed (an always-true gate
fixed the observed divergence equally well — it was perturbing extraction,
not correcting anything), and after five refuted approaches the fix was
abandoned. #1187 shipped test-only: an independent FreeType oracle, an
e-graph-free reproducer of the tangency math, and pinned baselines on the
existing goldens. **The glyph waist bug remains open on `main`.** See §9.

**C1b — demand replaces exclusivity.** §1–§3, on a fresh branch from
`main` after C1a. Gate: row-prologue telemetry on `O`@32 no worse than
2,801 guarded entries; the 1,217-entry arm partially guarded by
projection; a value shared by two selects guarded (the refused case);
nested selects yield a conjunction; the producer-superset property test;
a Y-only-demanded region containing a gather never reads outside its
buffer; differential against the oracle over all-true, all-false and
mixed batches for every polarity; identity on the packed programs; ISA
matrix.

**C2a — static demand in the extraction cost.** §4, cost half. Gate:
extraction on the glyph corpus prefers the guarded form where the static
fraction says so, and the row-prologue guard count on `O` rises again
without a font change.

**C2b — `Guard` in the graph and the sink rule.** §4, structure half.
Gate: `SelectHoistUnary` and its inverse are both present and the cost
decides between them; no golden moves.

**C2c — demand becomes loop structure.** §5. A Y-range demand schedules
into a loop over that range, and the per-row call overhead L3 measured
on narrow summands (1.46× at 15-px cells) closes. Depends on L3's union
being the loop-nest primitive.

## 7. Constraints

- **Widening is always sound.** Any predicate may be replaced by a weaker
  one, up to `true`. Nothing may ever strengthen one.
- **No tuned constant stands in for a bound.** The clause cap is a
  compile-budget knob and says so; `MISPREDICT_PENALTY_CYCLES` stays the
  one profitability bound, with its existing derivation.
- **The frame prologue is never guarded.** It runs once per call; a
  branch there buys nothing and the allocator must not be pinned for it.
- **A skipped region leaves zero in its parks.** Never a mask.
- **Select stays a blend.** Its value semantics do not change; demand
  only decides what is computed, never what is selected.

## 8. Non-goals

- Profile-guided coherence. The dynamic part of `P(demanded)` is the
  schedule-cost residual and is not this plan.
- Guarding in the frame prologue.
- Changing `Select`'s NaN or bit-pattern semantics (CLAUDE.md's
  floating-point section).

## 9. What happened to C1a, and why the glyph waist bug is still open

C1a landed as **#1187** (`test(graphics): main renders the '8' waist wrong —
an external oracle that proves it, and five fixes that don't work`,
commit `4fe39607`). It made **no production code changes** — three test
files only, `pixelflow-graphics/tests/freetype_oracle.rs` (new),
`pixelflow-graphics/tests/quad_tangency_winding.rs` (new), and an extended
`optimized_glyph_matches_raw_within_reassociation_noise` in
`kernel_glyph_optimize.rs`. `main`'s rasterizer is untouched, byte for byte.

### The mechanism

`main` renders `'8'` wrong at ordinary terminal font sizes: on the waist row
a half-covered band extends four texels past the glyph's right edge — ink
where the letter is not (`'8'@19`: our ink extent x[1,12] against FreeType's
x[1,8]; `'8'@7`: x[0,4] against x[0,2]). Where an outline reaches a local
Y-extremum at an on-curve point, the two quadratics meeting there each have
that point as an endpoint *and* as their own extremum. A ray through that
row should graze — equal coverage, opposite winding, sum zero — but each
segment decides `disc >= 0` from its own rounding of its own discriminant
expression, and on that row both discriminants sit within an ulp of zero.
They disagree, and one crossing survives uncancelled.

### The five approaches, and the measurement that killed each

1. **A tight Y-extent gate** — refuted, and refuted in the strongest
   possible way: an `EXTENT_SLOP` of `1e6` (a gate true on every row, i.e.
   no gate at all) removed the divergence just as well as a tight one. The
   fix was not correcting the geometry; it was perturbing the e-graph into
   a different extraction that happened not to exhibit the bug. This is
   the plan's own stated mechanism for C1a, and it is the one the
   measurement disproved.
2. **Dropping `disc >= 0`** so the clamped root pair cancels — refuted: the
   pair does not cancel when the tangent point sits exactly at parameter
   `t = 0` or `t = 1` (the common TrueType shape), because one root passes
   `t ∈ [0,1]` and the other does not. Regressed `'8'` at five more sizes.
3. **A scale-relative `MIN_DISC`** — refuted: the non-cancellation is
   driven by root *validity*, not by how close the two discriminants sit
   to each other, so widening or narrowing the separation threshold did
   not move the residual at all.
4. **Splitting each quadratic at its vertex into monotone pieces**, so
   existence is decided by comparing `Y` to exact control-point
   coordinates rather than by a rounded discriminant — half right: the
   discrete decision becomes exact and the optimized-vs-raw sweep goes
   green, but it moves 616 corpus texels by up to 0.83, and removing the
   root clamp entirely gives identical numbers — so whatever the actual
   cause is, it isn't the clamp, and this approach was masking rather than
   fixing it.
5. **Ramping the contribution to zero across a band around `disc == 0`** —
   the closest of the five, and refuted the most decisively. The ramp is
   only safe when *both* segments of a near-tangency fall inside the band;
   where one is inside and the other is comfortably positive, the ramp
   halves one signed contribution and the pair stops cancelling — the
   ramp creates the imbalance it exists to remove. That is corpus-dependent
   by construction: widening the glyph-and-size corpus from 94 glyphs at
   sizes 6–32 to the same set at sizes 6–64 took the count of orphan texels
   from 0 (at the chosen band width) to 140 on new glyphs (`f`, `{`, `}`),
   and the estimated safe ceiling fell every time the corpus grew — 10⁴,
   then 3×10⁴, then 2400, then ≈125, then ≈0.3 on a second font — while the
   floor needed to fix the original defect stayed at 876. **The usable
   window is empty, not narrow**, and this is the one of the five shown
   unfixable by parameter choice rather than merely unproven.

What shipped instead pins the defect rather than fixing it: an external
FreeType oracle (`freetype_oracle.rs`) bounding total ink and orphan-texel
count against an independent rasterizer; a JIT- and e-graph-free numerical
reproducer of the tangency cancellation (`quad_tangency_winding.rs`); and a
restored, actually-asserting sweep of
`optimized_glyph_matches_raw_within_reassociation_noise` across ten sizes,
pinned at 29 known divergent texels all on `'8'`. The module docs on all
three files carry the full detail (exact geometry, corpus tables, and two
more vacuous-guard findings folded into the lattice plan's CI section);
this is a summary, not a replacement for reading them.

**The glyph waist bug is open on `main` as of `d0b504c2`.** Nothing in
this session's tree fixes it. C1b (§1–§3 above) is unbuilt, and per JP's
2026-09-07 decision recorded at the top of this plan, the general demand
predicate — not a sixth per-select patch — is the intended next attempt.

### The regression-corpus lesson

`freetype_oracle.rs`'s own module doc states the lesson plainly, and it is
worth repeating because it cost a real reversal during this work: **a
regression corpus is a change-detector, not an oracle.** It silently
encodes whatever the code did on the day it was minted, and a comparison
against it cannot distinguish a wrong change from a right one — both move
the corpus. During the investigation, the ramp fix (approach 5) was
initially rejected on the grounds that it "discarded a crossing" the
existing corpus expected; the corpus was being read as ground truth. It
was not: the crossing being discarded was the very spurious one the corpus
had encoded as correct, because the corpus was minted from `main`'s
already-wrong rasterizer. Only the external oracle — a second, independent
rasterizer that had never seen this codebase's output — could tell the two
cases apart. This is the reason `freetype_oracle.rs` exists as a blocking
check rather than as one more same-form comparison, and it is the general
argument, from CLAUDE.md, that "a same-form check cannot see a
shared-definition bug; only an external bound can" — a regression corpus
is a same-form check with extra steps.
