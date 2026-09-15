> **Retracted/Superseded (2026-09-07), ledger L043 / L053 / L056.** The "Production quick-win" appended below — unguided regret 96.58% → 1.12% at B=100, 40.49% → 0.44% at B=200 from the numeric-first reorder — is a **registered budget the production regime does not run at**. Production stops at a median 5,422 rule applications (`docs/results/2026-09-07-corpus-structural-gaps.md`), which is 54× the B=100 the regret was measured at, and 93% of real kernels stop on `ClassCap` rather than the budget. The measurement stands at the budget it was taken at; the *production* reading of it does not, and no reorder has shipped. Verdict and rationale: `docs/results/2026-09-07-claims-ledger.md` (PR #1207); the corrected benchmark and re-validation order (item 2 re-takes rule order on real kernels in DAG units): `docs/plans/2026-09-07-benchmark-correction.md`.

# Phase 3 Round 2 registration v3: the |R| effect and the order effect, separated

**SUPERSEDES** `docs/plans/2026-09-01-phase3-round2-registration-v2.md`'s **H1 READING — not its
data.** Every curve, fingerprint, and statistic v2 measured (§4/§5/§6 of that document) stands
exactly as run and is not rerun or revised by this document. **Reason (v2 §6b, appended
2026-09-01):** v2's H1 verdict compared each inflated point `U(p)` against the unshuffled `base`
(production-order) point. Mode (i) (`dup:*`) proves that comparison is confounded: a duplicate
rule delegates its `apply`/templates verbatim to its original (`DuplicateRule`, `inflate.rs`), so
it can never let saturation reach a node the base 62 rules could not already reach — mode (i)'s
closure is unchanged by construction. Yet v2's own table shows `U(|R|)` falling under mode (i) just
as sharply as under modes (ii)/(iii) (96.59% → 43.76% → 41.12% → 15.44% → 38.67% at B=100, ρ =
−0.900) — which cannot be a closure/rule-count effect, so it is not evidence that `|R|` itself moves
`U`. What changed between `base` and every inflated point is that `Interleave(seed)`
Fisher-Yates-shuffles the *whole* base+inflation vector together, relocating each of the 62 base
rules to a new position — and B=100 is sub-sweep at every inflated point (v2 §4.1: 1.01 sweeps at
`base`, 0.10 sweeps at `dup:248`), so which rules the budget reaches is decided mostly by where the
shuffle put them, not by how many total rules exist. v3 separates the two effects: it measures the
`|R|` effect against a reference that holds order fixed (`RuleOrder::OrderMatchedBase`, already
committed on this branch), and measures the order effect on its own, as an independent registered
finding, rather than leaving it latent inside every inflated-point comparison.

**Date:** 2026-09-01
**Status:** REGISTERED. §3–§5.2's Register run is complete (§8, Entry 1) — every constant
previously marked **TBD** is now fixed; §2.1, §3's v1-definition Δ1 test, §5.3–§5.6 (Δ1 under v1's
definition, Y from the order-matched truncation loss, Δ2(v3), the §7.1 overhead check), §6.1 (H2
UNTESTED, with its rule) and §9 (the grep proof) are filled from the same unguided data (§8,
Entry 2 — no new run). Committed before any guided run at `|R| > 62` uses this document, per the
binding rule inherited from v1/v2. **Headline:** with order held fixed the `|R|` effect is null to
adverse (§3); the order effect is the finding (§4), and a static reorder of the production rule
list is a measured, unadopted quick win.

**Authority:** carries forward v2's full authority chain (`docs/plans/2026-08-31-guide-design-revision.md`
§5 protocol; `docs/plans/2026-09-01-phase3-registration.md` — Round 1, FROZEN; `docs/plans/2026-09-01-phase3-round2-rule-scaling.md`
— Round 2 design) plus `docs/plans/2026-09-01-phase3-round2-registration-v2.md` (v2 — H1 *reading*
superseded by this document, §6b's argument and every measured number otherwise inherited
verbatim). Orchestrator finding on the v2 result (this branch's task header, 2026-09-01, restated
in v2 §6b): mode (i) cannot change closure, so the interleaved-order drop in `U(|R|)` is a
confound between rule order and rule count; a static reorder of `all_rules()` may itself capture
much of Round 1's Guide-vs-unguided win at zero runtime cost — the production quick-win candidate
this document's §4 registers a finding about, without changing `all_rules()` itself (out of this
branch's scope, per its binding rules).

## 0. Mechanism (already committed — `pixelflow-search/src/math/inflate.rs`, prior commit on this
branch)

Three new `RuleOrder` variants, all base-62-only (`spec.mode == None`; a caller that reaches
`apply_order` with one of them on an inflated, `total > 62` build gets a loud panic, never a silent
no-op — CLAUDE.md's no-silent-failures rule):

- **`RuleOrder::OrderMatchedBase(seed, total)`** — the base-62 rules, reordered to the exact
  relative order they occupy inside `Interleave(seed)` of the `total`-rule inflated set (the
  subsequence of that interleaved vector restricted to base indices). **Proven mode-independent**
  by `order_matched_base_is_the_interleaved_subsequence_and_mode_independent` (`inflate.rs`):
  `fisher_yates`'s swap sequence is a function of vector LENGTH only, never content, and
  `DUPLICATE_GRID`/`COMPOSITION_GRID` share the same total↔inflation-count mapping at every shared
  grid point (93/124/186/248), so `OrderMatchedBase(s, t)` needs no mode argument — `dup:t` and
  `comp:t` under `Interleave(s)` yield an identical base-62 subsequence. This is the reference §3
  below measures every inflated point against.
- **`RuleOrder::Shuffled(seed)`** — base-62 fully shuffled by `seed`, no inflation content at all.
  Isolates seed sensitivity of the order effect on its own (§4).
- **`RuleOrder::StaticReorder(StaticReorderKind::NumericFirst)`** — base-62 ordered by descending
  TRAIN strict-positive rate (`docs/results/2026-09-01-train-guide-report.md`'s per-rule table),
  ties broken by ascending production index; pinned as `NUMERIC_FIRST_ORDER: [usize; 62]`
  (`inflate.rs`). The production quick-win candidate (§4).

CLI spec grammar (`RuleSetSpec::parse`, already committed): `"base:matched:<seed>:<total>"`,
`"base:shuffled:<seed>"`, `"base:static:numeric-first"` — `<seed>` decimal or `0x`-prefixed hex, no
underscores. Every one of these builds a 62-rule set; `rule_set_fingerprint` distinguishes all four
base-62 orders (production, matched, shuffled, static) from each other, pinned by
`base_only_orders_fingerprint_differently_from_each_other_and_from_production`.

**No production behavior change.** `all_rules()` is untouched by this branch's every commit — these
are harness-selectable orders a research binary opts into, exactly like `Append`/`Interleave`
before them. Adopting `StaticReorder(NumericFirst)` (or any reorder) as `all_rules()`'s own order is
a separate, unmade decision for JP, on this document's future data — not something this branch does.

## 1. Environment

Inherited from v2 §1 verbatim except where stated: same corpus, same 400-expression H1 sample (188
classical), same B = 100 (primary) / 200 (secondary), same `CostModel::latency_prior()`, same
application-count work axis (sweeps/evals reported alongside, per v2 §0.1/§0.2), same 14-point
checkpoint grid, same class cap, same safety ceilings (`comp:186`/`comp:248` remain unattempted here
— §2), same source-rev discipline. **New:** the three orders above; `DEFAULT_INTERLEAVE_SEED =
0x2026_0901` is reused, unchanged, as the seed `OrderMatchedBase` matches against (it must be the
same seed v2's `Interleave(seed)` points were built with, or `ΔU` would compare against the wrong
reference). Three additional seeds are pre-committed for the order-effect measurement (§4, §5.2):
**`SEED_A = 1`, `SEED_B = 2`, `SEED_C = 3`** — chosen for nothing but being fixed, small, and stated
here before any `Shuffled(*)` curve is run, exactly as the interleave seed was in v2 §1.

## 2. Grid

Same points as v2's realized grid (v2 §4): mode (i) `{62, 93, 124, 186, 248}`, mode (ii) `{62, 93,
124}` (`comp:186`/`comp:248` did not complete under v2's safety ceiling and are **not
re-attempted by this document** — same wall, out of scope here), mode (iii) `{62, 95}`. This
document adds no new inflated points; it adds a new *reference* (§3) and a new *independent
measurement* (§4) over the same grid.

### 2.1 Realized grid and seeds (filled 2026-09-01, §8 Entry 2)

Every rule set this document's numbers touch, with its order-inclusive fingerprint
(`rule_set_fingerprint`), median one-sweep-probe `apps_per_sweep` (`aps`) on the 188 classical
expressions, and B in sweeps (`B / aps`, per v2 §0.1's binding rule). Seeds: interleave seed
`0x2026_0901` (registered in v2 §1, reused unchanged as the `OrderMatchedBase` seed); order-effect
seeds `SEED_A/B/C = 1/2/3` (§1, pre-committed in the skeleton commit before any `Shuffled(*)` curve
ran); seed-sensitivity interleave seeds `1`, `2` (§4 addendum). Source:
`docs/results/2026-09-01-round2-unguided-vs-rulecount-v3.md`, "Registration extras" table.

| rule set | `\|R\|` | fingerprint | aps | B=100 sweeps | B=200 sweeps | role |
|---|---:|---|---:|---:|---:|---|
| `base` | 62 | `e99af8402beaff5d` | 100 | 1.01 | 2.01 | production order; = v2's \|R\|=62 point, byte-identical (re-measured in the order-effect run, loader dedup-asserted) |
| `base:matched:0x20260901:93` | 62 | `ab09ee08705f96aa` | 84 | 1.18 | 2.37 | §3 reference for `dup:93` **and** `comp:93` (mode-independent, §0) |
| `base:matched:0x20260901:124` | 62 | `5c6917e6bde09de4` | 82 | 1.21 | 2.42 | §3 reference for `dup:124`, `comp:124` |
| `base:matched:0x20260901:186` | 62 | `4c185b36d078890c` | 86 | 1.16 | 2.33 | §3 reference for `dup:186` |
| `base:matched:0x20260901:248` | 62 | `2baca95771460159` | 85 | 1.18 | 2.35 | §3 reference for `dup:248` |
| `base:matched:0x20260901:95` | 62 | `51d425eb78ad5821` | 85 | 1.18 | 2.35 | §3 reference for `new:95` |
| `base:shuffled:1` | 62 | `0b5612ba8d0abf82` | 88 | 1.14 | 2.29 | §4 order effect, `SEED_A` |
| `base:shuffled:2` | 62 | `02c5ed25ec0daff4` | 85 | 1.18 | 2.35 | §4 order effect, `SEED_B` |
| `base:shuffled:3` | 62 | `185ebacfa932f651` | 85 | 1.18 | 2.35 | §4 order effect, `SEED_C` |
| `base:static:numeric-first` | 62 | `9e6d66598d997f37` | 88 | 1.14 | 2.29 | §4 quick-win candidate |
| `dup:93` / `dup:124` / `dup:186` / `dup:248` | 93/124/186/248 | `83e610e33e782a68` / `b207aa331bb625ab` / `3a00c565900b48e6` / `43c43d764ef7f76b` | 142/300/502/996 | 0.70/0.33/0.20/0.10 | 1.41/0.67/0.40/0.20 | mode (i), inherited from v2 unchanged |
| `comp:93` / `comp:124` | 93/124 | `904ceec9b110e89e` / `a7600e5942f0baa5` | 87/90 | 1.15/1.12 | 2.30/2.23 | mode (ii), inherited from v2 unchanged |
| `new:95` | 95 | `113cca49c99cc850` | 292 | 0.34 | 0.69 | mode (iii), inherited from v2 unchanged |
| `dup:124:interleave:1` / `:2` | 124 | `6333431113f09bde` / `5ca26c601faaf74c` | 266/293 | 0.38/0.34 | 0.75/0.68 | §4 addendum, seed sensitivity |
| `comp:93:interleave:1` / `:2` | 93 | `71851d821506a78b` / `437971d00006a362` | 86/90 | 1.16/1.12 | 2.31/2.23 | §4 addendum, seed sensitivity |

Two things the table makes visible that the prose above does not: (a) a reorder of the same 62
rules changes `apps_per_sweep` (100 in production order, 82–88 under every other base-62 order;
the probe is one front-to-back pass, and how many applications one pass records depends on which
rules have already fired when each rule's turn comes) — so B=100 is 1.01 sweeps in production order but
1.14–1.21 sweeps under every reordered base-62 set, and every `ΔU` in §3 compares two curves that
are both ~1.2 sweeps at B=100, not one at 1.0 and one at 0.1–0.7; (b) `dup:93` and `comp:93` have
*different* `apps_per_sweep` (142 vs 87) despite sharing one `OrderMatchedBase` reference —
duplicate rules re-record every application of their original, composition rules mostly do not
fire at this budget (§5.2's `comp:93` Δ1 of 0.00).

## 3. H1 (v3): the `|R|` effect, order held fixed

For every inflated point `p = (mode, |R|)` at the registered interleave seed `s =
DEFAULT_INTERLEAVE_SEED`:

```
ΔU(p) = U(p) − U(OrderMatchedBase(s, |p|))
```

where `U(p)` is v2's already-measured median unguided regret at B for the inflated point (v2 §4/§6
data, unchanged), and `U(OrderMatchedBase(s, |p|))` is a **new** unguided curve — same 188-expression
classical sample, same B, same reference convention (§3 below) — measured on the base-62 rules
swept in exactly the order they occupy inside `Interleave(s)` at that `|p|`. This holds sweep order
fixed between the two terms: both curves see "the base rules in interleaved position order at this
|R|," and the only thing that differs is whether the inflated rules are present at all. `ΔU`
therefore isolates the `|R|` effect from the order effect v2 §6b identified.

**H1(v3):** `ΔU(p)` grows with `|R|` — Spearman `ρ` of `ΔU` over each mode's grid, **and** `ΔU` at
the largest completed `|R|` point clears **+Δ1(v3)** (§5.2; the same derivation rule as v2 §5.3 —
95% bootstrap CI of the median `ΔU` at the smallest inflated point per mode, 10,000 resamples, seed
42 — but computed on `ΔU`, not raw `U`, since the quantity being tested for a minimum detectable
effect has changed). Both conditions are evaluated **per mode**, exactly as v2 §6 did for `U`.

**What this predicts, stated in advance so it is not rediscovered by surprise:** mode (i) is exact
duplicates — its closure is identical to `base`'s at every |R| — so if v2 §6b's confound argument is
correct, mode (i)'s `ΔU` should sit near zero at every point (no room for the added rules to help or
hurt, once order is matched), unlike v2's raw `U(|R|)`, which fell sharply. Modes (ii)/(iii) DO add
real closure (mechanical compositions, genuinely new rules) and are where a nonzero, `|R|`-growing
`ΔU` — if it exists — would be a genuine capacity/coverage effect, not an order artifact. This
document does not assert either outcome; §8 registers whichever the data show.

`ΔU` is undefined at `|R|=62` itself (that point is never reordered — §0/§2 invariant — so it has
no `OrderMatchedBase` counterpart); rows below cover the 7 realized inflated points only.

| Mode | grid `\|R\|` | B | `U(p)` (v2, unchanged) | `U(`OrderMatchedBase`)` | `ΔU(p)` | Spearman ρ | `ΔU`(max) | ≥ +Δ1(v3)? | **verdict** |
|---|---|---:|---|---|---|---:|---:|---|---|
| (i) | 93,124,186,248 | 100 | 43.76, 41.12, 15.44, 38.67 | 60.55, 43.88, 13.85, 4.33 | -16.79, -2.76, +1.59, **+34.34** | **1.000** | +34.34 | YES (Δ1=0.05) | **HOLDS** (direction + effect) |
| (i) | 93,124,186,248 | 200 | 6.69, 25.44, 9.47, 27.02 | 24.82, 16.89, 4.70, 3.36 | -18.13, +8.55, +4.76, **+23.66** | 0.800 | +23.66 | YES (Δ1=4.82) | **HOLDS** (direction + effect) |
| (ii) | 93,124 | 100 | 60.55, 41.69 | 60.55, 43.88 | 0.00, -2.19 | -1.000 (n=2, degenerate) | -2.19 | trivially, Δ1≈0 | direction FAILS; only 2 points, not a real trend read |
| (ii) | 93,124 | 200 | 25.60, 21.22 | 24.82, 16.89 | +0.78, +4.33 | +1.000 (n=2, degenerate) | +4.33 | trivially, Δ1≈0 | direction FAILS (n=2 rho uninformative); effect clears a near-zero Δ1 |
| (iii) | 95 | 100 | 33.11 | 48.25 | -15.14 | undefined (n=1) | -15.14 | NO (Δ1=11.45) | direction undefined; effect FAILS |
| (iii) | 95 | 200 | 52.06 | 27.52 | +24.54 | undefined (n=1) | +24.54 | YES (Δ1=16.74) | direction undefined; effect HOLDS |

**Reading.** Mode (i) is the only mode with enough grid points (4) for a Spearman read, and it
holds cleanly at both budgets — but the sign is *increasing* regret with `|R|`, not decreasing:
`ΔU` climbs from -16.8% at `dup:93` to +34.3% at `dup:248` (B=100). Since mode (i) cannot change
what saturation can reach (exact duplicates, closure identical to `base` at every point), this is
not a capacity effect — it is application-budget dilution: a fixed *count* of applications buys a
shrinking share of productive (non-duplicate) matches as redundant duplicate slots are added to
every sweep, even with the 62 real rules' relative sweep position held fixed via
`OrderMatchedBase`. This is the opposite of §3's stated advance prediction (mode (i) `ΔU` sitting
near zero) — the prediction assumed holding order fixed would leave nothing for `|R|` to move;
what it misses is that `|R|` still governs how much of a fixed *application* budget each sweep
pass costs, independent of order or closure. Modes (ii) (2 points) and (iii) (1 point) do not have
enough grid points to read a trend at all — mode (ii)'s two-point Spearman ρ is ±1 by construction
regardless of effect size, and both modes' `ΔU` values are small-to-moderate and inconsistent in
sign between B=100/B=200. Full numbers, per-expression pairing, and the bootstrap detail:
`docs/results/2026-09-01-round2-unguided-vs-rulecount-v3.{json,md}`.

**`ΔU` vs `|R|` with the `≥ +Δ1` test under v1's definition (appended 2026-09-01, §8 Entry 2).**
The table above tests `ΔU(max)` against `Δ1(v3)` (paired, §5.2), which is near zero in modes
(i)/(ii) because the smallest inflated point barely differs from its matched reference — a
near-zero minimum detectable effect makes the effect condition easy to clear, so it is also
evaluated against the inherited, far more conservative `Δ1` of v1's definition (§5.3: 28.23 pts at
B=100, 12.34 pts at B=200 — the bootstrap half-width of median `U(62)` itself):

| Mode | B | `ΔU`(max) | Δ1 (v1 def.) | ratio | `ΔU`(max) ≥ +Δ1? | with ρ (§3 table) |
|---|---:|---:|---:|---:|---|---|
| (i) | 100 | +34.34 | 28.23 | +1.22× | **yes** | ρ = 1.000 → H1(v3) HOLDS, adverse sign |
| (i) | 200 | +23.66 | 12.34 | +1.92× | **yes** | ρ = 0.800 → HOLDS, adverse sign |
| (ii) | 100 | −2.19 | 28.23 | −0.08× | no | n=2, no trend readable |
| (ii) | 200 | +4.33 | 12.34 | +0.35× | no | n=2, no trend readable |
| (iii) | 100 | −15.14 | 28.23 | −0.54× | no | n=1, ρ undefined |
| (iii) | 200 | +24.54 | 12.34 | +1.99× | yes (one point, no trend) | n=1, ρ undefined |

**Stated plainly.** With order held fixed, the `|R|` effect on unguided regret is **null to
adverse** on this grid. In the one mode with a readable trend (i), regret at fixed applications
*rises* with `|R|` — by 34 points from `dup:93` to `dup:248` at B=100, clearing even v1's
conservative Δ1 — and since those rules are exact duplicates with `base`'s closure, the rise is the
cost of paying for more rule slots per sweep out of a fixed application budget, not a change in
what the search can reach. Modes (ii)/(iii), which do add closure, show `ΔU` within ±5 points at
their 2 and 1 points except `new:95`'s single B-dependent swing (−15.1 / +24.5), which one point
cannot resolve into a trend. **That is the capacity finding: on this corpus, at these budgets,
adding rules does not lower unguided regret once the order confound is removed — and it is not a
reason to touch the grid.** v2's raw `U(|R|)` drop (96.6% → 15–44%) was the order effect (§4).

## 4. The order effect, as its own registered finding

Independent of `|R|` entirely — every rule set measured here is 62 rules. Two sub-measurements,
both on the 188-expression classical sample, both at B = 100/200, both against `ref_U(e, 62)`
(unchanged from v2 §3 — every set here IS the |R|=62 point, so the closure-aware reference is the
min over these curves' own checkpoints together with `base`'s, exactly as v2 §3 defines it for any
fixed |R|):

- **Seed sensitivity:** `U(Shuffled(SEED_A))`, `U(Shuffled(SEED_B))`, `U(Shuffled(SEED_C))` — three
  independently-shuffled base-62 orders, no inflation. If these three disagree substantially, v2's
  single interleave seed (`0x2026_0901`) was not a representative draw and §3's `ΔU` (which is
  anchored to that one seed) needs the caveat spelled out, not silently trusted.
- **Static reorder vs production order:** `U(StaticReorder(NumericFirst))` vs `U(base)` (production
  `all_rules()` order, v2's already-measured |R|=62 point) — the quick-win candidate, measured
  directly rather than inferred.

For every one of the four rule sets in this section, report the **count of the 188 classical
expressions whose extracted cost@B differs from `base`'s** at B=100 and B=200 — the same
"differing" convention v2 §4 used to state that inflated points were reachable at all, here used to
state how much of the sample a pure reorder (no rule added or removed) touches.

| Rule set | B=100 median U | B=200 median U | differing from `base` @100/@200 (of 188) |
|---|---:|---:|---|
| `base` (production order, = v2's \|R\|=62 point) | 96.58 | 40.49 | — (reference row) |
| `Shuffled(SEED_A=1)` | 43.74 | 23.57 | 175 / 150 |
| `Shuffled(SEED_B=2)` | 46.19 | 25.70 | 186 / 143 |
| `Shuffled(SEED_C=3)` | 26.28 | 1.49 | 186 / 151 |
| `StaticReorder(NumericFirst)` | **1.12** | **0.44** | 186 / 140 |

**Reading (the prediction above, evaluated).** `StaticReorder(NumericFirst)`'s median U sits at
1.12% (B=100) — not merely "substantially below" `base`'s 96.58%, but ~86x smaller, far past the
per-rule-control's 0.565 benchmark the orchestrator finding cited, and better than every one of
the three random shuffles too (26–46%). This is the production quick-win registered as data, not
yet as a production change (§0). The three `Shuffled` seeds do spread meaningfully (26–46% at
B=100, 1.5–25.7% at B=200) — `0x2026_0901` (v2's single interleave seed, itself equivalent to a
4th random draw once inflation content is stripped out — see the seed-sensitivity addendum below)
was not an outlier in the sense of falling outside this range, but the range itself is wide enough
that a single seed's `ΔU` (§3) should be read with that spread as its uncertainty, not as an exact
value. Every tested order — random or static — still beats `base` by a wide margin, so the
*existence and direction* of the order effect is not seed-fragile even though a specific `ΔU`
number is. `rapid`/`blitz` bands: U=0.00% at every order tested (differing counts nonzero, so the
orders do change costs somewhere in the sample — the differences just don't survive to that
budget's regret at those node-count scales); reported for completeness, no claim drawn from them.

**Production quick-win, in one sentence (appended 2026-09-01, §8 Entry 2):** the static
numeric-first reorder of the identical 62 production rules — no rule added, removed, or changed,
zero runtime cost — takes median unguided regret at B=100 from **96.58% (production order) to
1.12%**, and at B=200 from **40.49% to 0.44%**, changing extracted cost@B on **186 / 140** of the
188 classical expressions (B=100 / 200) and beating all three random shuffles (26.3–46.2% at
B=100); **adopting it as `all_rules()`'s order is JP's decision, in its own PR** — this branch
registers the measurement and leaves `all_rules()` untouched (§0).

**Seed sensitivity of an inflated point (addendum, not in the original skeleton).** `dup:124` and
`comp:93` re-run under two additional interleave seeds (`1`, `2`) alongside the registered seed —
classical band:

| Rule set | seed | U@100 | U@200 |
|---|---|---:|---:|
| `dup:124` | `0x20260901` (registered) | 41.12 | 25.44 |
| `dup:124:interleave:1` | 1 | 37.80 | 32.37 |
| `dup:124:interleave:2` | 2 | 50.33 | 22.62 |
| `comp:93` | `0x20260901` (registered) | 60.55 | 25.60 |
| `comp:93:interleave:1` | 1 | 39.95 | 18.77 |
| `comp:93:interleave:2` | 2 | 38.70 | 13.07 |

Spread (range) across the 3 seeds: `dup:124` 12.5 pts @100 / 9.8 pts @200; `comp:93` 21.9 pts @100
/ 12.5 pts @200 — real, on the same order as the base-62 spread above, another sign that a single
inflated point's `U` (and thus §3's `ΔU`, anchored to one seed) carries seed noise of this
magnitude. Full numbers: `docs/results/2026-09-01-round2-unguided-vs-rulecount-v3.{json,md}`.

## 5. Registered constants

Same binding rule inherited from v1/v2: fixed **only** from unguided data on this harness,
**committed before any guided run at `|R| > 62` uses this document's `OrderMatchedBase` reference**,
and — once filled — may not be revised except by append or a further superseding registration.

### 5.1 Inherited, not re-derived
B = 100/200; `Y`'s formula; ε = 0.005; the 0.02 Δ2 floor; composition pool seed `0x5EED2`;
bootstrap seed `42`; oracle seed `0xC0FF_EE42`; the 400-expression sample and its stratification;
the 14-point checkpoint grid; the closure-aware reference convention (v2 §3, design §1.1); the H2
statistics (design §1.3); `DEFAULT_INTERLEAVE_SEED = 0x2026_0901` (v2 §1, reused as the seed
`OrderMatchedBase` and §3's `ΔU` are anchored to).

### 5.2 New this document
- Order-effect seeds: `SEED_A = 1`, `SEED_B = 2`, `SEED_C = 3` (§1) — pre-committed here, before any
  `Shuffled(*)` curve is run.
- **Δ1(v3)**, per mode, at the smallest inflated point, computed on PAIRED per-expression
  `regret_p(e) − regret_matched(e)` (10,000 resamples, seed 42, same protocol as v2 §5.3):
  - Mode (i) at `dup:93`: B=100 half-width **0.05 pts** (median 0.00, CI [-0.09, 0.00]); B=200
    **4.82 pts** (median -4.21).
  - Mode (ii) at `comp:93`: B=100 half-width **0.00 pts** (median 0.00, CI [0.00, 0.00] — the
    composition rules barely fire within B=100 at |R|=93, so `dup:93`/`comp:93` are nearly
    identical to their `OrderMatchedBase` reference for almost every expression); B=200 **0.00
    pts** (same).
  - Mode (iii) at `new:95`: B=100 half-width **11.45 pts** (median -0.18, CI [-21.22, +1.67]);
    B=200 **16.74 pts** (median +6.12).
- Every other v3-specific number in §3/§4's tables: filled by this run (§8, Entry 1).

### 5.3 Δ1 — v1's definition, on v3 data (appended 2026-09-01, §8 Entry 2)

v1/v2's Δ1 is the 95% bootstrap CI half-width of the median unguided regret at `|R| = 62` in
production order (10,000 resamples, seed 42, order-statistic 2.5/97.5 percentiles). The `base`
curve was re-measured by this document's order-effect run and is byte-identical to v2's (the
loader asserts it, §8 Entry 1), so the CI is recomputed here rather than copied — and comes out
identical:

| B | median U(62) | CI | **Δ1 (v1 def.)** |
|---:|---:|---|---:|
| 100 | 0.9658 (96.58%) | [0.7312, 1.2958] | **0.2823 (28.23 pts)** |
| 200 | 0.4049 (40.49%) | [0.2570, 0.5039] | **0.1234 (12.34 pts)** |

Both Δ1s are registered: Δ1(v3) (§5.2, paired, the statistic §3's H1(v3) names) and Δ1 (v1 def.)
(this section, the conservative floor §3's second table applies). Where they disagree on a
verdict — mode (ii) at B=200 (+4.33 clears 0.00, not 12.34) and mode (iii) at B=100 (−15.14 vs
11.45 / 28.23, fails both) — the conservative one is the reading this document reports.

### 5.4 Y per point, from the order-matched truncation loss (appended 2026-09-01, §8 Entry 2)

`Y = 1 − (1 + L/2)/(1 + L)` with `L` the median truncation loss (cost@B vs cost@4B, Round 1's
convention), classical band. `Y(p)` for the seven inflated points is v2 §5.2's number, unchanged;
`Y(matched)` is new, from each point's `OrderMatchedBase` reference; `ΔY = Y(p) − Y(matched)` is
the order-held-fixed analogue of §3's `ΔU`.

| Mode | rule set | `\|R\|` | L(matched)@100 | Y(p)@100 | Y(matched)@100 | **ΔY@100** | L(matched)@200 | Y(p)@200 | Y(matched)@200 | **ΔY@200** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| (i) | `dup:93` | 93 | 24.384 | 8.65 | 9.80 | −1.15 | 0.160 | 0.09 | 0.08 | +0.01 |
| (i) | `dup:124` | 124 | 16.767 | 6.75 | 7.18 | −0.43 | 1.763 | 2.42 | 0.87 | +1.55 |
| (i) | `dup:186` | 186 | 1.692 | 1.25 | 0.83 | +0.42 | 0.776 | 0.52 | 0.38 | +0.13 |
| (i) | `dup:248` | 248 | 0.623 | 4.95 | 0.31 | **+4.64** | 0.477 | 3.27 | 0.24 | **+3.03** |
| (ii) | `comp:93` | 93 | 24.384 | 9.87 | 9.80 | +0.07 | 0.160 | 0.09 | 0.08 | +0.01 |
| (ii) | `comp:124` | 124 | 16.767 | 6.28 | 7.18 | −0.90 | 1.763 | 1.31 | 0.87 | +0.44 |
| (iii) | `new:95` | 95 | 17.706 | −0.00 | 7.52 | −7.52 | 16.837 | 0.00 | 7.21 | −7.21 |

For the order-effect sets (§4, all `|R|=62`): `base` L/Y@100 = 48.467 / 16.32, @200 = 21.922 /
8.99; `Shuffled(1)` 13.672 / 6.01, 14.957 / 6.51; `Shuffled(2)` 18.585 / 7.84, 7.934 / 3.68;
`Shuffled(3)` 13.448 / 5.93, 0.499 / 0.25; `StaticReorder(NumericFirst)` 0.599 / 0.30, 0.002 /
0.00. Read through the truncation-loss lens the order effect is the same size as through the
regret lens: production order leaves 16 points of Y on the table at B=100 that the static reorder
leaves 0.3 of. Mode (i)'s `ΔY` rises with `|R|` exactly as its `ΔU` does (−1.15 → +4.64 at B=100):
the duplicates' dilution cost shows up as truncation loss too, because the matched reference
quiesces sooner (L(matched)@100 falls from 24.4 to 0.6 across 93 → 248 — the reordered 62 rules
alone reach their 4B cost within B, and adding duplicate slots is what stops the inflated set from
doing the same).

### 5.5 Δ2 — H2's minimum effect, per mode, order held fixed (appended 2026-09-01, §8 Entry 2)

v2 §5.4's Δ2 (`max(0.02, Y(|R|max) − Y(62))`, against production-order `base`) is inherited
unchanged: 0.02 (floor) in every mode, every raw difference negative. Under this document's
reference the same rule reads `Δ2(v3) = max(0.02, ΔY at |R|max)` with `ΔY` from §5.4:

| Mode | `\|R\|`max | ΔY@100 | **Δ2(v3)@100** | ΔY@200 | **Δ2(v3)@200** |
|---|---:|---:|---:|---:|---:|
| (i) | 248 | +4.64 pts | **0.0464** | +3.03 pts | **0.0303** |
| (ii) | 124 (not 248 — §2) | −0.90 pts | 0.020 (floor; negative) | +0.44 pts | 0.020 (floor; 0.0044 < 0.02) |
| (iii) | 95 | −7.52 pts | 0.020 (floor; negative) | −7.21 pts | 0.020 (floor; negative) |

Mode (i) is the only mode whose Δ2(v3) lifts off the floor — and it does so for the same dilution
reason as its `ΔU` (§3), so a guided arm in mode (i) would have to beat a *worse* unguided baseline
by 4.6 points of Y to register, not a better one. Modes (ii)/(iii) stay on the floor as in v2.

### 5.6 The §7.1 overhead precondition, unguided half — order held fixed (appended 2026-09-01, §8 Entry 2)

Threshold inherited from v2 §5.6 verbatim: flat ⇔ median `evals_actual / app_actual` at B is
≤ 2× its production-order `|R| = 62` value, i.e. **≤ 62.41 at B = 100 and ≤ 79.08 at B = 200**
(v2 printed 62.40 — 2 × 31.20 rounded twice; 2 × 31.2049 is 62.41 — no rule set sits between the
two, so every v2 verdict is unchanged). Measured for every set new to this document:

| rule set | `\|R\|` | evals/app @100 | × base | flat @100? | evals/app @200 | × base | flat @200? |
|---|---:|---:|---:|---|---:|---:|---|
| `base` (production order) | 62 | 31.20 | 1.00 | — | 39.54 | 1.00 | — |
| `base:matched:0x20260901:93` | 62 | 59.36 | 1.90 | yes | 63.11 | 1.60 | yes |
| `base:matched:0x20260901:124` | 62 | 40.29 | 1.29 | yes | 49.82 | 1.26 | yes |
| `base:matched:0x20260901:186` | 62 | 79.36 | 2.54 | no | 77.94 | 1.97 | yes |
| `base:matched:0x20260901:248` | 62 | 144.96 | **4.65** | no | 121.30 | 3.07 | no |
| `base:matched:0x20260901:95` | 62 | 67.98 | 2.18 | no | 69.84 | 1.77 | yes |
| `base:shuffled:1` | 62 | 58.72 | 1.88 | yes | 59.58 | 1.51 | yes |
| `base:shuffled:2` | 62 | 22.20 | **0.71** | yes | 41.41 | 1.05 | yes |
| `base:shuffled:3` | 62 | 130.67 | 4.19 | no | 115.81 | 2.93 | no |
| `base:static:numeric-first` | 62 | 46.76 | 1.50 | yes | 57.55 | 1.46 | yes |
| `dup:124:interleave:1` / `:2` | 124 | 97.46 / 61.67 | 3.12 / 1.98 | no / yes | 89.30 / 69.46 | 2.26 / 1.76 | no / yes |
| `comp:93:interleave:1` / `:2` | 93 | 94.87 / 128.14 | 3.04 / 4.11 | no / no | 99.45 / 123.49 | 2.51 / 3.12 | no / no |

(v2's seven inflated points are in v2 §5.6 and are unchanged.)

**Result.** The overhead ratio is **order-confounded in the same way `U` was**: reordering the
identical 62 rules moves `evals/app` at B=100 from 0.71× to 4.65× the production-order value —
a wider range than v2 measured across the whole inflated grid (2.2–3.2×). So v2 §5.6's "every
completed inflated point exceeds 2×" is not evidence that `|R|` raises per-application enumeration
cost; against the order-matched reference (the two columns above, divided — `evals/app(p) /
evals/app(matched)`, straight from this table and v2 §5.6's), the inflated points sit at 1.27
(`dup:93`), 1.94 (`dup:124`), 1.16 (`dup:186`), **0.54** (`dup:248`), 1.44 (`comp:93`), 2.50
(`comp:124`), 1.02 (`new:95`) at B=100 and 1.26 / 1.53 / 1.19 / 0.62 / 1.46 / 2.06 / 0.89 at
B=200 — under 2× everywhere but `comp:124`, and *below* 1× at the largest duplicate point. The
static reorder (the quick-win candidate) is flat at both budgets (1.50× / 1.46×). The guided half
(scored-candidate count, `GuidedEpisodeStats`) remains unmeasured — unchanged from v1 §9 point 3.

## 6. Gates

Same accept/kill/honest-fallback shape as v2 §8, evaluated against v3's `ΔU` statistic (§3) in
place of v2's raw `U` statistic wherever this document supersedes v2's reading.

- **Accept gate (per mode):** H1(v3) (§3) AND H2 (design §1.3, unchanged) hold on DEV classical
  (n=334) at B=100. **Not evaluated here** — this Register run is TRAIN+DEV-sample unguided data
  only (§1, inherited from v2), matching v1/v2's own scope; no DEV-only re-split or guided run was
  performed under this document. H2 remains UNTESTED, as it was left in v2 §11 Entry 2.
- **Kill gate (per mode):** H2 part 3 failing at any `|R|` point on DEV — not triggered; H2 was not
  evaluated (above).
- **Honest fallback — this is what fired.** `ΔU` shows an `|R|`-growing effect only in mode (i)
  (§3), and it is a dilution cost (increasing regret), not a coverage gain — the opposite sign
  from what would make `|R|` itself a reason to add rules. Modes (ii)/(iii) have too few grid
  points to assert a trend either way. So: v2 §6b's confound argument is essentially the WHOLE
  explanation for v2's raw `U(|R|)` finding — order dominates, and the residual `|R|` effect
  (mode i) is small and adverse relative to it (§4's order effect: 50–95 points at B=100 from a
  reorder alone, vs mode (i)'s +34.3-point `ΔU` maximum). This is the honest-fallback outcome
  §6 anticipated, now recorded as data (§8, Entry 1).

### 6.1 H2: UNTESTED — and its rule (appended 2026-09-01, §8 Entry 2)

**H2 status: UNTESTED.** No guided run exists at any `|R| > 62` (§9 is the grep proof, run fresh
at this commit), so nothing in this document evaluates H2. Its statistics (`Q`, `G`, design §1.3),
its per-part reading rule (v1 §7: parts 1/3 testable at every point, part 2 — the Guide's
advantage growing with `|R|` — live in modes (ii)/(iii) and impossible by construction in mode (i)),
and the two-arm reference convention (v2 §3) stand unchanged and are neither weakened nor
pre-judged by §3/§4. **The rule this document adds:** any guided run that consumes this
registration at `|R| > 62` must report its guided curve against the **`OrderMatchedBase`
reference at that `|R|`** as well as against `base` — otherwise its "advantage over unguided" would
inherit the same order confound v2 §6b found in `U`, and Round 1's Guide-vs-unguided win is itself
the case in point (v2's Round 1 note: at `|R| = 62`, B = 100 is 1.01 sweeps in production order,
and §4 shows a reorder alone closes most of that gap). Headroom `1 − Y` (the window a guided arm
would have to close, v2 §7's lens) under the order-matched reference: `|R|` = 93: 90.20% / 99.92%
(B=100 / 200); 124: 92.82% / 99.13%; 186: 99.17% / 99.62%; 248: 99.69% / 99.76%; 95: 92.48% /
92.79% — versus 83.68% / 91.01% under production-order `base` and 99.70% / 100.00% under
`StaticReorder(NumericFirst)`. Every reordered base-62 set leaves a guided arm *less* to win than
production order does; whatever H2 part 2 would need to show, it must show it against that window.

## 7. Reproduction (commands actually run for Entry 1, §8)

```bash
# §3 — OrderMatchedBase references, one 62-rule curve per inflated |R| point
# on the SAME seed those points were built with (0x20260901 = DEFAULT_INTERLEAVE_SEED
# = 539363585 decimal — the CLI form takes plain hex or decimal, no underscores)
cargo run --release -p pixelflow-pipeline --bin phase3_round2_unguided_curves -- \
    --corpus-dir pixelflow-pipeline/data --samples 400 \
    --out-csv docs/results/2026-09-01-round2-order-matched-base-v3.csv \
    --out-json docs/results/2026-09-01-round2-order-matched-base-v3.json \
    --rule-sets base:matched:0x20260901:93,base:matched:0x20260901:124,base:matched:0x20260901:186,base:matched:0x20260901:248,base:matched:0x20260901:95

# §4 — order effect in isolation: 3 shuffled seeds + the static reorder, vs base
cargo run --release -p pixelflow-pipeline --bin phase3_round2_unguided_curves -- \
    --corpus-dir pixelflow-pipeline/data --samples 400 \
    --out-csv docs/results/2026-09-01-round2-order-effect-v3.csv \
    --out-json docs/results/2026-09-01-round2-order-effect-v3.json \
    --rule-sets base,base:shuffled:1,base:shuffled:2,base:shuffled:3,base:static:numeric-first

# §4 addendum — seed sensitivity of an inflated point: dup:124 and comp:93 under
# 2 additional interleave seeds (the registered-seed rows are reused from v2's CSV)
cargo run --release -p pixelflow-pipeline --bin phase3_round2_unguided_curves -- \
    --corpus-dir pixelflow-pipeline/data --samples 400 \
    --out-csv docs/results/2026-09-01-round2-seed-sensitivity-v3.csv \
    --out-json docs/results/2026-09-01-round2-seed-sensitivity-v3.json \
    --rule-sets dup:124:interleave:1,dup:124:interleave:2,comp:93:interleave:1,comp:93:interleave:2

# fingerprint + mode-independence + pinned-order guarantees for the three new RuleOrder variants
cargo test -p pixelflow-search math::inflate -- --nocapture

# raw-row union under one header (the .csv this document's numbers trace back to)
{ head -1 docs/results/2026-09-01-round2-order-matched-base-v3.csv; \
  tail -n +2 docs/results/2026-09-01-round2-order-matched-base-v3.csv; \
  tail -n +2 docs/results/2026-09-01-round2-order-effect-v3.csv; \
  tail -n +2 docs/results/2026-09-01-round2-seed-sensitivity-v3.csv; \
} > docs/results/2026-09-01-round2-unguided-vs-rulecount-v3.csv

# aggregate stats — ΔU, the order table, seed spread (round2_register_stats.py's
# per_rule_set/quartiles/bootstrap machinery, imported not re-derived; new script
# because base_rs in the shared tool is hard-wired to "base", and §3 needs an
# arbitrary per-point reference)
python3 pixelflow-pipeline/scripts/round2_register_stats_v3.py \
    --v2-csv docs/results/2026-09-01-round2-unguided-vs-rulecount-v2.csv \
    --matched-csv docs/results/2026-09-01-round2-order-matched-base-v3.csv \
    --order-csv docs/results/2026-09-01-round2-order-effect-v3.csv \
    --seed-csv docs/results/2026-09-01-round2-seed-sensitivity-v3.csv \
    --out-json docs/results/2026-09-01-round2-unguided-vs-rulecount-v3.json \
    --out-md docs/results/2026-09-01-round2-unguided-vs-rulecount-v3.md
```

## 8. Results appended against the gates

(Append-only, as in v1/v2.)

**Entry 1 (2026-09-01, this commit).** Unguided Register run under this document, `phase3_round2
_unguided_curves --release`, same 400-expression sample as v1/v2 (188 classical). v2's 8 already-
measured rule sets (`base`, `dup:93/124/186/248`, `comp:93/124`, `new:95`) are **not re-run** —
reused byte-identical from `docs/results/2026-09-01-round2-unguided-vs-rulecount-v2.csv` (the
loader in `round2_register_stats_v3.py` asserts byte-identity wherever a `(rule_set, expr,
checkpoint)` key appears in more than one input file, which it does for `base` here, and it did
not die). 3 new binary runs, 13 new rule-set curves total: 5 `OrderMatchedBase` references (§3),
5 order-effect sets (`base` re-measured + 3 `Shuffled` seeds + `StaticReorder(NumericFirst)`, §4),
4 seed-sensitivity curves (`dup:124`/`comp:93` × 2 additional seeds, §4 addendum). All at
`|R| ∈ {62, 93, 124}` — no run in this document touches `|R| ≥ 186`, so none of it revisits the
`comp:186`/`comp:248` wall (§2) or v2's safety-ceiling panics.

**H1(v3) verdict: mode (i) HOLDS (direction + effect, both budgets) — but in the dilution
direction, not the coverage direction §3 flagged as the alternative to watch for. Modes (ii)/(iii)
are underdetermined (2 and 1 grid points).** Full tables: §3/§4 above. **§6's honest fallback
fired**: order dominates and the residual `|R|` effect, where measurable, is small and adverse.

**Production quick-win, measured directly:** `StaticReorder(NumericFirst)` — descending TRAIN
strict-positive rate, ties by production index, zero runtime cost, no rule added or removed —
cuts median unguided regret at B=100 from `base`'s 96.58% to 1.12% (188-expression classical
sample), beating every random shuffle tested too. This is data, not a production change (§0) —
adopting it as `all_rules()`'s order is JP's decision on this data, out of this branch's scope.

**Files:** `docs/results/2026-09-01-round2-order-matched-base-v3.{csv,json}`, `docs/results/2026
-09-01-round2-order-effect-v3.{csv,json}`, `docs/results/2026-09-01-round2-seed-sensitivity-v3.
{csv,json}` (the three raw per-run outputs); `docs/results/2026-09-01-round2-unguided-vs-rulecount
-v3.{csv,json,md}` (the union CSV and the aggregate stats + narrative this document's numbers are
drawn from); `pixelflow-pipeline/scripts/round2_register_stats_v3.py` (the aggregation script, new
this commit).

**Entry 2 (2026-09-01, this commit).** Registration filled — **no new run, no new rule set, no
code change outside the aggregation script.** `round2_register_stats_v3.py` gained a
`registration_extras` block (same §7 command, re-run; every SS3/SS4/seed-sensitivity number it
already emitted is unchanged, diff-verified additive-only) that computes, from the same four CSVs:
per-rule-set identity + `apps_per_sweep` + B in sweeps (§2.1), Δ1 under v1's definition (§5.3 —
recomputed on the re-measured `base` copy, identical to v2: 28.23 / 12.34 pts), `L`/`Y` for every
set and paired `ΔY` (§5.4), Δ2(v3) (§5.5 — mode (i) 0.0464 / 0.0303, modes (ii)/(iii) on the 0.02
floor), the §7.1 overhead ratio and flatness verdict per set (§5.6 — threshold 62.41 / 79.08), and
`ΔU(max)` against v1's Δ1 (§3, second table). §6.1 records H2 as UNTESTED with the rule a
consuming guided run must obey; §9 is the fresh grep proof.

**H1 verdict, final form (unguided data, order held fixed):** mode (i) — `ΔU` grows with `|R|`
(ρ = 1.000 / 0.800) and `ΔU(max)` = +34.34 / +23.66 pts clears both Δ1(v3) and v1's Δ1
(1.22× / 1.92×) — H1(v3) HOLDS as a statistic, with the **adverse** sign: regret rises with rule
count at fixed applications; modes (ii)/(iii) — `ΔU(max)` fails v1's Δ1 at B=100 (−2.19, −15.14)
and has 2 / 1 grid points, no trend readable. **The `|R|` effect with order fixed is null to
adverse; the order effect (§4: 50–95 points from a reorder alone; static reorder 96.58% → 1.12%)
is the finding.** H2 UNTESTED (§6.1). Honest fallback fired (§6), unchanged from Entry 1.

## 9. Proof that no guided run at `|R| > 62` exists under this registration

Same obligation as v2 §10, inherited through §5's binding rule, run fresh at this commit on
`claude/phase3-round2` — every fingerprint from §2.1 (the 13 sets new to this document, `base`, and
v2's 7 realized + 2 never-realized inflated points), excluding only the two registration documents
themselves from the search (they quote the fingerprints as text):

```text
$ for fp in ab09ee08705f96aa 5c6917e6bde09de4 4c185b36d078890c 2baca95771460159 51d425eb78ad5821 \
           0b5612ba8d0abf82 02c5ed25ec0daff4 185ebacfa932f651 9e6d66598d997f37 \
           6333431113f09bde 5ca26c601faaf74c 71851d821506a78b 437971d00006a362 \
           e99af8402beaff5d 83e610e33e782a68 b207aa331bb625ab 3a00c565900b48e6 43c43d764ef7f76b \
           904ceec9b110e89e a7600e5942f0baa5 113cca49c99cc850 9e9bf3a4458a3045 b89d841eada63c13; do
    printf '%s: ' $fp; git grep -l "$fp" -- . ':!docs/plans/2026-09-01-phase3-round2-registration-v2.md' \
        ':!docs/plans/2026-09-01-phase3-round2-registration-v3.md' | tr '\n' ' '; echo; done
ab09ee08705f96aa: docs/results/2026-09-01-round2-order-matched-base-v3.csv docs/results/2026-09-01-round2-order-matched-base-v3.json docs/results/2026-09-01-round2-unguided-vs-rulecount-v3.csv
5c6917e6bde09de4: (same three files)
4c185b36d078890c: (same three files)
2baca95771460159: (same three files)
51d425eb78ad5821: (same three files)
0b5612ba8d0abf82: docs/results/2026-09-01-round2-order-effect-v3.csv docs/results/2026-09-01-round2-order-effect-v3.json docs/results/2026-09-01-round2-unguided-vs-rulecount-v3.csv
02c5ed25ec0daff4: (same three files)
185ebacfa932f651: (same three files)
9e6d66598d997f37: (same three files)
6333431113f09bde: docs/results/2026-09-01-round2-seed-sensitivity-v3.csv docs/results/2026-09-01-round2-seed-sensitivity-v3.json docs/results/2026-09-01-round2-unguided-vs-rulecount-v3.csv
5ca26c601faaf74c: (same three files)
71851d821506a78b: (same three files)
437971d00006a362: (same three files)
e99af8402beaff5d: docs/plans/2026-09-01-phase3-round2-registration.md docs/results/2026-09-01-phase3-round2-registration-tables.md docs/results/2026-09-01-phase3-round2-registration-v2.json docs/results/2026-09-01-phase3-round2-registration.json docs/results/2026-09-01-round2-order-effect-v3.csv docs/results/2026-09-01-round2-order-effect-v3.json docs/results/2026-09-01-round2-unguided-vs-rulecount-mode-iii.csv docs/results/2026-09-01-round2-unguided-vs-rulecount-modes-i-ii.csv docs/results/2026-09-01-round2-unguided-vs-rulecount-v2.csv docs/results/2026-09-01-round2-unguided-vs-rulecount-v2.json docs/results/2026-09-01-round2-unguided-vs-rulecount-v2.md docs/results/2026-09-01-round2-unguided-vs-rulecount-v3.csv
83e610e33e782a68: docs/results/2026-09-01-phase3-round2-registration-v2.json docs/results/2026-09-01-round2-unguided-vs-rulecount-v2.csv docs/results/2026-09-01-round2-unguided-vs-rulecount-v2.json docs/results/2026-09-01-round2-unguided-vs-rulecount-v2.md
b207aa331bb625ab, 3a00c565900b48e6, 43c43d764ef7f76b, 904ceec9b110e89e, a7600e5942f0baa5, 113cca49c99cc850: (same four v2 unguided files)
9e9bf3a4458a3045:                                   <- comp:186 interleaved: no file anywhere (never realized)
b89d841eada63c13:                                   <- comp:248 interleaved: no file anywhere (never realized)

$ git grep -l -E 'RuleOrder::(Interleave|OrderMatchedBase|Shuffled|StaticReorder)|DEFAULT_INTERLEAVE_SEED|NUMERIC_FIRST_ORDER'
docs/plans/2026-09-01-phase3-round2-registration-v2.md
docs/plans/2026-09-01-phase3-round2-registration-v3.md
docs/results/2026-09-01-round2-unguided-vs-rulecount-v3.md
docs/results/journal.jsonl                          <- the unguided runs' own journal records
pixelflow-pipeline/scripts/round2_register_stats_v3.py   <- a comment naming the seed constant
pixelflow-search/src/math/inflate.rs                <- the types' definitions and their pinning tests

$ grep -c 'nnue::guide\|GuidedSaturation' pixelflow-pipeline/src/bin/phase3_round2_unguided_curves.rs
0

$ grep -rl 'math::inflate\|inflate::' pixelflow-pipeline/src/bin/
pixelflow-pipeline/src/bin/phase3_round2_new_rules.rs
pixelflow-pipeline/src/bin/phase3_round2_unguided_curves.rs   <- the only two binaries that can build a reordered or inflated set; neither links a Guide

$ grep -n 'all_rules()' pixelflow-pipeline/src/bin/phase3_at_budget_eval.rs   # the guided harness
623:        all_rules(),
648:        all_rules(),                                <- hard-wired to |R| = 62 in production order; no rule-set or order argument exists
```

So: every fingerprint of every set this document measured occurs, as data, only in this
registration's own unguided output files (and, for `base`, in Round 1's / v1's / v2's unguided
files, where the production-order 62-rule set has always lived); the two never-realized points
occur as data nowhere. The only binary that can select a non-production order is the unguided
curves binary, which links no Guide. No Guide checkpoint, label set, or `phase3_at_budget_eval`
output on the branch carries any fingerprint but production-order `base`'s — there is no guided
run at any `|R| > 62`, and none at `|R| = 62` under any order other than production's.
