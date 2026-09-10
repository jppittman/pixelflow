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

## The hump

**A glyph bake costs ~331 ms in release** — 31.5 s for a 95-glyph ASCII atlas,
measured at the sha above. `core-term` calls `atlas.warm(&font, ' '..='~')` at
startup (`core-term/src/terminal_app.rs`) and again on every font-size or
density change, so that is ~31 s to launch and ~31 s per resize. **The glyphs
are correct; the terminal is not usable.** Everything in this section is about
that number.

| | what | where |
|---|---|---|
| **H1** | **S3 — one program for the font.** Font-wide extent, table padded with monoid identities, so every glyph compiles to the same program and a glyph becomes a table write. 95 compiles → 1. Nothing else is the right order of magnitude. | [glyph-as-a-fold-execution](plans/2026-09-09-glyph-as-a-fold-execution.md) §S3 |
| **H2** | **Split the 331 ms** between saturation+extraction and collapse. Unmeasured, and it decides whether H1 alone is the fix or H1 must land with H4. Cheapest item here; do it first. | — |
| **H3** | **Hash-consing in `ExprArena`.** Prototyped and measured: arena 2,721 → 154 nodes, 2.1–2.2× on the glyph suites, extracted kernel unchanged. In flight (JP). | [exprarena-on-dag](plans/2026-09-09-exprarena-on-dag.md) §5.2 |
| **H4** | **Ask B — hoist binder-only work out of the pixel loop.** `‖∇scale‖` is invariant in X and Y but emitted per pixel; ~8,700 `rsqrt` per 16×16 tile where 34 would do. **Do not patch `contains_gather`** — see N1 for why. | [a-glyph-is-a-circle](plans/2026-09-09-a-glyph-is-a-circle.md) §B |

S3's own doc calls itself "a trade, not a win — fewer compiles against
evaluation of rows that contribute nothing." For the **atlas** path that is
too pessimistic: a glyph bakes once into texels and is a gather forever after,
so the padding is a one-time bake cost, not per frame. Worth re-deciding when
H1 is picked up.

## Names and binding time

| | what | where |
|---|---|---|
| **N1** | **One kind of name.** `Var`/`Uniform`/`Buffer`+`Gather`/`Ref` are five spellings of "bound later", differing only in *when*. Reframes L6 from "delete `Gather`" to "there is one kind of name", with `Gather`'s disappearance a consequence. Blocks a principled H4. | [one-name-bound-later](plans/2026-09-10-one-name-bound-later.md) |
| **N2** | **L4 — `Ref(k) ⟷ body(k)` as a growth-gated e-graph rule.** The gate exists (`EGraph::predicted_growth`, asserted against measured delta over 10,819 applications); the rule does not. Its case is a scene composing many *identical* kernels, not construction-side sharing (L3 settled that). | [composition-is-linking](plans/2026-09-09-composition-is-linking.md) §3, §5.1 |
| **N3** | **L5 — a surviving `Ref` is a call.** Second function, coordinate ABI, a register-allocation boundary the allocator does not model. Open question: whether a *tabulated* `Ref` (a leaf that emits a load) avoids all of it, which would let N1 land the cheap half first. | [composition-is-linking](plans/2026-09-09-composition-is-linking.md) §5.2 |
| **N4** | **L6 — a tabulated kernel is a `Ref` with a cached tabulation.** Subsumed by N1; kept as a row because the task list and several commit messages name it. | [composition-is-linking](plans/2026-09-09-composition-is-linking.md) §4 |

## Demand and the conditional

| | what | where |
|---|---|---|
| **D1** | **mask ⟹ index range, symbolic tier.** Axis-aligned literals and their conjunctions/disjunctions, read off the DAG. No lowering — derive the range and check it. Gate is *containment* (collapse the mask, assert every nonzero index is inside), plus a usefulness comparison against `grid_range`. Independent of everything else here. | [one-conditional-three-lowerings](plans/2026-09-08-one-conditional-three-lowerings.md) §8 |
| **D2** | Lowering 1 — emit the split: select over the derived range, root specialized at `m ≡ false` over the complement. | *ibid.* |
| **D3** | Bind-time tier, and splitting `IndexRange` into a derived region and a requested band. | *ibid.* |
| **D4** | Interval evaluation, target-aware and rounding outward. Unlocks glyph supports, which the symbolic tier cannot reach (a compound glyph's affine mixes X and Y). | *ibid.* |
| **D5** | Lowering 2 on the general predicate — the superseded demand plan's §1–§2, as the third case rather than the whole subject. | *ibid.* |

D1 → D2 unblocks **S2**: deleting `cells`, `contour_bounds`, the `Union`
plumbing, `TEXT_CELL`, `min_of`, `may_be_interior` and `chord_winding` —
roughly 800 lines to 150 — and makes H1's padding free.

## The e-graph

| | what | where |
|---|---|---|
| **E1** | **Geometric `SplitFold`.** `⊕_{[lo,hi)} = ⊕_{[lo,mid)} ⊕ ⊕_{[mid,hi)}`. Needs no substitution (both halves share the body e-class) and is *exactly* cost-preserving in both extraction arms, so it cannot desync the claim/price audit. Bisect at the midpoint to bound growth at 2n−1. `EmptyFold` already exists as its base case. | — |
| **E2** | **A critical-path term in the cost model.** Without it E1 buys reachability and no speed: `CostModel::latency_prior()` sets `depth_threshold: 1024, depth_penalty: 0` ("effectively disabled"), so a 34-deep serial chain and a 6-deep balanced tree price identically. A *global* depth hinge is the wrong shape — what is wanted is the reduction's critical path. | [schedule-cost-model-denotation](plans/2026-09-01-schedule-cost-model-denotation.md) |
| **E3** | Extraction: `shared_dag_dp_pass` is O(L²) — make reach tracking sparse. | — |
| **E4** | Saturation rescans every class with every rule every iteration — dirty tracking. | — |
| **E5** | Extraction is not monotone in graph richness: the same kernel in a superset graph can extract a worse DAG. | — |

## Correctness and CI

| | what | where |
|---|---|---|
| **C1** | **Trig range is unasserted.** `pixelflow-ir/tests/trig_range.rs` made its claims through the deleted interpreter and went with it. The property is unchanged; an out-of-range `sin` would now ship green. Needs rebuilding on the JIT. | CLAUDE.md, "Precision is on the table; range is not" |
| **C2** | **The `'8'` waist bug is open on `main`.** Five fixes tried and refuted; `freetype_oracle.rs` pins it rather than fixing it. The general demand predicate (D5), not a sixth per-select patch, is the intended next attempt. | [demand-is-a-dag-property](plans/2026-09-07-demand-is-a-dag-property.md) §9 |
| **C3** | `CachedText::kernel` sums glyph coverages, so overlapping glyphs can exceed 1. Not on any production path — `core-term` renders through `GlyphAtlas`, and `CachedText` has no non-test caller. | — |
| **C4** | A corpus needs a new acceptance criterion before `gen_bench_corpus` can come back; its quarantine gate compared against the interpreter. | CLAUDE.md, "Cost Model and the Guide" |
| **C5** | A local CI runner, so a full presubmit does not cost a push. | — |

## Housekeeping

- `Kernel::parts()` hands out the **unlinked** fragment, and five measurement
  consumers each learned to link first. Right division, five copies of one
  line. ([composition-is-linking](plans/2026-09-09-composition-is-linking.md) §7)
- `cells` / `text_union` reach only one Criterion bench; nothing on screen has
  ever gone through them. Delete with S2, not before — they are the worked
  example of a domain-side extent.
  ([glyph-as-a-fold-execution](plans/2026-09-09-glyph-as-a-fold-execution.md) §S2)
