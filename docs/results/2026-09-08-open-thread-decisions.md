# Decisions owed on #1207 and #1215 — 2026-09-08

Companion to [2026-09-08-open-pr-triage.md](2026-09-08-open-pr-triage.md).

That sweep closed fifteen of twenty-three review threads on #1207 and three
of seven on #1215 by deciding them against a source in the tree, and handed
back the remaining twelve as "yours". Handing back a problem is not
completed staff work. This document is the missing half: for each open
thread, the finding, the evidence, and **a recommendation to accept or
reject** — so the decision is one pass, not a re-derivation.

Nothing here is applied. Every row below is a proposal.

Verified against `main` at `385b5bc`.

## #1207 — the claims ledger

### 1. L081 — reclassify from NEVER TESTED ON REAL → **HELD**

`kind=real`. The row records `'O'`@32 going selects=60/guarded=0 →
selects=125/guarded=2801, 0.69 → 0.37 µs/row, and `'S'` 1.07 → 0.54. Those
are shipped glyph kernels. The ledger defines NEVER TESTED ON REAL as
"minted on a generated corpus and never taken on a shipped kernel", which
this is not.

Its `real_check` — "#1187 shipped TEST-ONLY; the per-select machinery never
merged" — is a **deployment** qualification, not an absence of measurement.

**Recommend:** HELD, keeping the unmerged-implementation caveat in `note`.
**Effect:** NEVER TESTED 14 → 13, HELD 35 → 36, headline 53 → 52 of 88.

### 2. L047 — cannot be FAILED; one question decides the replacement

FAILED is defined as "tested on a real shader and contradicted". L047's
`corpus` is synthetic `sh`/bezier and its `real_check` is `n/a`. That much
is settled by the document's own definition.

The replacement turns on a single fact I could not establish from the tree:

> **Were the 1.09–1.11× bilinear deltas priced in tree cost or DAG cost?**

- **tree** → UNITS INVALID (its `note` already says "extractor still
  tree-objective, so re-take under #1192", which points this way)
- **DAG** → NEVER TESTED ON REAL (matches `real_check = n/a` exactly)

**Recommend:** UNITS INVALID, on the strength of its own note — but the
question above is the decider and you can answer it faster than I can.
**Effect:** FAILED 7 → 6; headline 53/88 unchanged either way, since both
are non-standing.

### 3. L083 — split the row; HELD is right for one half only

The claim bundles two results: Estrin beats Horner **1.2–4.0× on the
serialized critical path**, and is **a tie to 11% slower in the scanline
(production) regime**. The cited report warns that AVX-512 latency
measurements below degree 16 are dominated by chaining overhead and should
not be read per row — and its own table has AVX-512 degree 8/9 and `log2`
at 0.84/0.84/0.80, i.e. Estrin *losing*.

**Recommend:** split into two rows. The scanline/production conclusion —
which is the one that matters and the one the repo acted on — stays HELD.
The serialized 1.2–4.0× becomes its own row, narrowed to the degrees the
report says are readable, or marked INSTRUMENT DEFECT.
**Effect:** 88 → 89 rows; HELD unchanged or +0/−1 depending on the second
row's verdict.

### 4. L072 — rephrase the chrome result as out-of-domain, keep the verdict

The chrome cap experiment measures **runtime on one scene**; the group-G
cap A/Bs measure **latency-prior extraction cost** over glyph and cell-grid
populations. A slower chrome clock shows the cost conclusion does not
transfer; it cannot contradict *every* per-kernel A/B observation. The
ledger's own corpus rule (§"corpus", the 86× → 0.97× lesson) rejects
exactly this cross-population inference.

**Recommend:** keep L072 HELD and keep the older cost rows invalidated —
but on the **independent tree-vs-DAG grounds**, with chrome recorded as an
out-of-domain counterexample rather than a contradiction.
**Effect:** no verdict or count change; wording only.

### 5. L028 — INSTRUMENT DEFECT → **synthetic estimate with sampling uncertainty**

The row is ρ = 0.35 (n=55); its `real_check` is "tightened-labeler re-draw
on a different 800: rho 0.186". Two draws, the **same computation**,
different corpora. That is sampling variance. An instrument defect is an
instrument later shown wrong — this instrument was shown *noisy*, and the
later report still supports the qualitative "moderate, not 0.02" reading
and recommends reporting a range.

**Recommend:** reclassify, and state the claim as a range (ρ ≈ 0.19–0.35,
two draws) rather than a point estimate.
**Effect:** INSTRUMENT DEFECT 18 → 17; headline 53 → 52 if the replacement
verdict stands.

### 6. L078 — supply the provenance or drop the row

Its `minted` is "kernel-with-a-lattice S4b-1 landing block". I searched
that document for `bypass`, `JitManifold`, `optimize_runtime_arena` and
`collapse_cost` near S4b-1 and could not find the text that either the row
or the review describes.

I cannot verify the claim, its retraction, or the review's counter-claim
against a source I cannot locate. **This is the one open thread where I
have no recommendation** — only the observation that a row whose provenance
cannot be resolved is itself an instance of §"provenance", the fifth of
this ledger's own named failure modes.

**Recommend:** either point `minted` at the real artifact, or drop the row
and its retraction. Both beat leaving it unresolvable.

### 7. L020 — superseded; F has now been run on real shaders

`docs/results/2026-09-07-egraph-off-vs-on-real-shaders.md` is on `main` and
completes the F measurement through the production path on the shipped-kernel
corpus: **18.6% and 20.8%** aggregate glyph32 improvement across two rounds,
90/95 glyphs faster.

L020 still reads NEVER TESTED ON REAL, and §5 still reads "F first, running
now".

**Recommend:** re-verdict L020 against that measurement and update §5 to
record F as complete. **This is a re-take, not an edit** — it moves the
headline and the re-validation order, which is why the sweep did not do it.

### 8. L053 / L056 — mark historical and add the current regime

Both quote the retired 200 ms-timeout regime: 68.4% ClassCap, median 8,446
applications. `2026-09-07-corpus-structural-gaps.md:28` runs the current
`Budget::Production` and reports **93% ClassCap, median 2 iterations**, with
a companion median of **5,422** applications.

L069 in this same ledger already says the wall clock is now only a fail-loud
ceiling, so the ledger contains both regimes without marking which is which.

**Recommend:** qualify L053/L056 as historical, add the current figures as
their own rows, and re-check the dependent "85× the registered B=100"
conclusion — 5,422 against B=100 is 54×, not 85×.

## #1215 — the benchmark correction

### 9. The 5% decision rule sits below the plan's own noise floor

Line 54: "per-kernel clock ratios under ~10% are not trusted (L076)".
Item 1: "if rules-beyond-CSE move ns/px by **< 5%** on every DEV family, the
research target is extraction and the saturation half waits".

A 5–10% movement is both *below the floor* and *decision-triggering*.

**Recommend: raise the rule to 10%**, matching the floor the document
already commits to. The alternative — keeping 5% and lowering the stated
floor — contradicts L076, which is a real-shader measurement.

**Note this may already be moot.** The off-vs-on result above (18.6%/20.8%
on glyph32, a DEV family) is far outside either threshold, and appears to
discharge item 1 in the *opposite* direction before the ordered plan starts.
Worth settling before item 1 is run as written.

### 10. The cap gate constrains chrome but not glyph startup

Its ship rule bounds compile time only by chrome's 250 ms scene threshold,
while DEV contains **380 glyph bakes**. A cap change could add milliseconds
to every glyph — seconds of startup in aggregate — with each compile still
under 250 ms and every other condition passing. Line 58 already names
per-glyph and summed warm time as the user-facing budget.

**Recommend:** add a summed-glyph-warm-time bound to the cap gate's ship
rule, at whatever regression percentage you consider shippable.

### 11. The Guide may be promoted on `dag_cost` alone

Items 5 and 6 promote the Guide when its extracted `dag_cost` is no worse per
family. This document's own verdict establishes that `dag_cost` can fall
while emitted bytes and runtime-facing schedules get worse on real scenes.

**Recommend:** require the per-family primary metric *and* an emitted-bytes
check before promotion — the same evidence standard the rest of the plan
applies to every other candidate.

### 12. The #1207 dependency — narrowed from three inputs to one

When filed, none of the three cited inputs existed in that tree. Merging
`main` brought two: corpus structural gaps (#1212) and e-graph off vs on
(#1210). Only the claims ledger (#1207) is still absent — and **all 28
retraction banners point at it**.

**Recommend:** land #1207 first, as the thread says. Nothing else blocks it;
both PRs are unconflicted and green.

## What these have in common

Nine of the twelve are the ledger or the plan disagreeing with **itself** or
with **a file already in the tree** — a verdict against its own definition, a
threshold against its own floor, a regime against its own successor
measurement. Only #6 (L078) is genuinely unresolvable, and only #2 turns on
a fact I could not read.

That is the same shape as the fifteen already fixed, and it suggests the
useful gate is not more review but the one field these documents lack:
**which tree a row was verified against.** With it, #7, #8 and #12 would have
announced themselves the moment `main` moved.

---

## Addendum (2026-09-08, later the same day): the six threads the first pass missed

The table above was written against #1215's review as it stood at 16:17. A
later round added six more threads, and the original document's claim to have
covered "every open thread" was true when written and false within the hour —
which is the fifth failure mode again, in a document *about* that failure mode.
They are assessed here on the same terms.

All six are on `docs/plans/2026-09-07-benchmark-correction.md`. None is a
research judgement in the sense the first twelve were: each is the plan
disagreeing with its own stated gap, its own §B.2, or a lesson already recorded
in CLAUDE.md. I recommend accepting all six.

| # | thread | recommendation |
|---|---|---|
| 13 | chrome consumed by item 1 | **Accept.** §E item 1 clocks "chrome, psychedelic, cell grid and `O`@32"; §B.2 says any post-freeze opening permanently promotes chrome to DEV and owes a replacement held-out member. As written, the *baseline* run spends the scene reserved for the publication run. Cheapest fix: drop chrome from item 1 and clock the three DEV members — item 3 already says "the DEV scene and `O`@32", so item 1 need only match its sibling's wording. Naming a replacement held-out scene instead is also sound but costs a new frozen artifact |
| 14 | axis 5 closes without checking op frequencies | **Accept.** The gap column is stated in frequencies (transcendental-bearing 4% real vs 88% synthetic; `MulAdd` 15.9 / `Select` 7.4 / compares 10.9 / `Dwrt` 6.5 / `Sqrt` 4.5 / `Div` 3.3 against `Neg` 16.3 / `Abs` 7.9 / `Pow` 5.0 / `Div` 0.15), and the acceptance column checks rule-firing plus one seam. A corpus can hold the entire stated skew and still close the axis. Add the frequencies themselves: transcendental-bearing rate and the six named per-op shares each inside real [p10, p90], which is the form axes 1, 3 and 7 already use |
| 15 | retractions not propagated to downstream copies | **Accept, scoped.** All three checked against `main` rather than taken from the finding, and all three stand (only the path was off — the third is in `docs/results/`, not `docs/plans/`): `plans/2026-07-07-guided-saturation-redesign.md` still presents ρ ≈ 0.35 as a live finding; `plans/2026-09-01-phase3-round2-registration-v3.md` still carries "Production quick-win, in one sentence" with 96.58% → 1.12% at B=100; `results/2026-09-02-missing-congruence.md` still opens on "68.4% of real kernels, median 8.66% / p90 13.2% truncation cost" as the premise under test — the very regime figure thread #8 above marks historical. None of the three carries a retraction banner. Full transitive propagation is unbounded, so bound it: banner those three, then give the ledger a *cited-by* column so the next retraction knows its blast radius. Better still, and in the spirit of §B.5 — make it a check: a retracted row's ID appearing in a doc with no retraction banner is grep-detectable, and CLAUDE.md's rule is that a gap in CI is a check to write rather than a caveat to attach |
| 16 | axis 1 closes without select prevalence | **Accept.** Same shape as #14 and cheaper. The gap states "kernels with a select 95% vs 0%"; the acceptance checks three medians. 51% select-heavy plus 49% select-free satisfies every median while half the corpus cannot exercise a select guard at all. Add: fraction of kernels containing at least one select ≥ 0.9 |
| 17 | item 6 never compares bilinear against linear | **Accept.** Item 6 exists to revalidate L047, which *is* the functional-form question, and it inherits item 5's guided-vs-unguided rule — so a bilinear head materially worse than the linear one passes by beating unguided. Add a head-to-head: bilinear ≤ linear `dag_cost` on every DEV family, pre-committed, on top of item 5's bar. Note this is the same underlying question as #2 above: L047's verdict cannot be settled until the units are known, and its re-take cannot be settled until the arms are compared to each other |
| 18 | shader oracle is same-form only | **Accept, and this is the strongest of the six.** The listed oracles for `shader_bench` are same-form `eval_scalar` plus a range bound. A rewrite that changes the function to a different in-range value passes both. This is not a hypothetical: CLAUDE.md records it as the reason the `sin` range-reduction bug survived — "the JIT and the `eval_scalar` oracle run the *same* expansion, so they agreed bit-for-bit on the garbage and every same-form equivalence test passed", with the out-of-range outputs slipping under the `>1e30` filter. A same-form check cannot see a shared-definition bug; only an external bound can. The plan already knows this, and applied it one table up: this very PR restricted the cell grid's byte-identity oracle to the `off`/`cse` arms and gave the FMA-bearing arms a tolerance. Do the same for shaders — compare optimized against the `off` form under a stated numerical contract, keeping the same-form check beside it |

### What the six have in common

Five of the six are an **acceptance rule that does not test the gap its own row
states** (#14, #16), a **decision rule that cannot answer the question its item
exists to ask** (#13, #17), or **an oracle the repository has already been
burned by** (#18). Only #15 is about work not done rather than a rule written
too weakly.

That is a sharper version of the pattern in the first twelve. There the trouble
was a row disagreeing with a *file elsewhere in the tree*; here it is a row
disagreeing with **the paragraph above it**. A plan that pre-commits decision
rules — which is this plan's whole method, and the right method — has to have
those rules read against the gaps they close, because a pre-committed rule that
does not bind is worse than none: it will be cited as evidence the axis was
checked.

**None of these six needs a measurement to settle.** Each is decidable by
reading the document against itself, which is why they are recommendations
rather than open questions.
