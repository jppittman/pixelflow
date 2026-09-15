# Production saturation budgets denominated in rule applications (2026-09-01)

> **Revised 2026-09-08 — the classical class cap is sized by the input, and the raise
> is pinned off.** The numbers below were calibrated at a flat classical cap of 5,000
> classes. The class-cap sweep (`docs/results/2026-09-08-class-cap-sweep.md`) showed that
> cap is smaller than what 44 of the 95 DejaVu glyphs *insert*: their first rewrite
> round hits it before it completes, so the rules never run on them. The classical cap
> is now `clamp(8 × inserted_classes, CLASSICAL_CLASS_FLOOR, CLASSICAL_CLASS_CEILING)`
> with the application budget and safety ceiling scaling with it at this document's own
> ratios — **and the ceiling is pinned at the floor**, because raising `'8'` off the
> floor re-fuses its quadratic solver and puts the waist smear the FreeType oracle found
> back on screen at 13–21 px. See "Revision: the classical cap grows with the inserted
> input" at the end; everything else here stands, with the flat 5,000 read as the
> classical **floor**.

**Calibration only. No code in this commit.** This document fixes the constants and
names; the implementation is a separate change.

## The problem, restated

Production sized its saturation budget from the input's node count
(`config_for_node_count`, `pixelflow-search/src/egraph/saturate.rs`), and one of the
three dimensions in that budget was **wall clock**: blitz 20 rounds / 500 classes /
10 ms, rapid 50 / 2,000 / 50 ms, classical 100 / 5,000 / 200 ms, the deadline checked
per iteration and passed down into `apply_rule`.

Two of those three are deterministic. The third is not, and the #1085 review named the
consequence: at proc-macro expansion time the optimizer runs at opt-level 0, where the
same saturation takes ~1.7 s untimed, so `kernel!` kernels reached the 200 ms deadline
and stopped early. **Which kernel you got depended on the build host's speed and on the
proc-macro's optimization level.** Same source, different codegen, silently.

`#1108` ("one optimizer entry point") already removed the clock from the production
path: `Optimizer::production()` resolves `Budget::Production` to `Limits { iterations,
classes, applications: None }` and calls `EGraph::saturate_budgeted`, which never looks
at a clock. What #1108 did **not** do is put anything back in the clock's place. Today
production's only bounds are the round cap and the class cap, and on the measured
corpora the round cap never binds — so a pathological kernel has no time-shaped bound at
all. This document supplies the deterministic replacement.

## The design

| dimension | status | why |
|---|---|---|
| **rule applications** | **new budget** | deterministic; the counter already exists (`EGraph::application_count`, incremented unconditionally in `apply_action_from_rule`) and is the one currency that means the same thing to every ordering policy |
| **e-class cap** | kept unchanged | deterministic; memory protection, and the bound that actually binds on real kernels |
| **iteration cap** | kept unchanged | deterministic; never observed to bind (see below) |
| **wall clock** | **no longer a budget** — a fail-loud safety ceiling | exceeding it is a bug in the budget, so it panics; it can never silently change codegen |

The ceiling is `Optimizer::hard_ceiling`, which already exists and already panics. What
is new is that production sets one.

## Calibration data

Two independent corpora, both measured 2026-09-01:

- **#1087**, `docs/results/2026-09-01-production-saturation-telemetry.csv` — 193 *real*
  core-term kernels (the 623-node packed cell grid at 3 geometries + 190 printable-ASCII
  glyph kernels at 2 bake densities), each run three ways. The column that matters here
  is `ref_apps`/`ref_stop`: the **same class cap, no clock, 4× rounds** — i.e. exactly
  the regime this change creates. `ref_stop` is `ClassCap` on 181/193 and `Quiesced` on
  12/193; nothing hit the round cap.
- **#1084**, `docs/results/2026-09-01-phase3-at-budget-eval.jsonl`, `production` record —
  394 DEV expressions run through the production saturation probe, 334 classical /
  30 rapid / 30 blitz. These still carry the clock, so their `timeout` rows are
  *lower bounds* on the deterministic stop; the application counts themselves are exact.

Every number below was recomputed from those two raw files (the CSV's `ref_*` columns and
the JSONL's `production` record), not lifted from either report's prose. Quantiles are
inclusive linear interpolation; `p100` is the observed maximum, and every `n` is the full
tier population with no filtering.

Round-1 stop *labels* (`...-round1-inferred.csv`) are not used: they were inferred, and
#1087's own §"Round 1 stop labels were inferred, and are suspect" shows 39 of 56 lifted
`Quiesced` labels were wrong. Application counts from both rounds agree to the digit
wherever the clock was not involved, so this document uses counts, never inferred labels.

### Applications at a deterministic stop, per tier

| tier | class cap | corpus | n | q1 | median | p90 | p99 | **p100** | stop at p100 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| blitz | 500 | #1084 DEV | 30 | 9 | 12 | 16 | 4,414 | **6,198** | `timeout` (415/500 classes) — a *lower* bound |
| blitz | 500 | #1084 DEV, quiesced only | 29 | — | 12 | 16 | 37 | 45 | `quiesced` |
| blitz | 500 | #1087 real | 2 | 0 | 0 | 0 | 0 | 0 | `Quiesced` (1-node space glyph) |
| rapid | 2,000 | #1084 DEV | 30 | 23 | 49 | 167 | 1,423 | **1,862** | `quiesced` (259/2,000 classes) |
| rapid | 2,000 | #1087 real | 0 | — | — | — | — | — | no core-term kernel is rapid-tier |
| classical | 5,000 | #1087 real, `ref` (no clock) | 191 | 3,065 | 8,446 | 39,040 | 45,482 | **55,242** | `ClassCap` (3,545 classes) |
| classical | 5,000 | #1084 DEV | 334 | 400 | 1,671 | 23,799 | 63,256 | **85,900** | `timeout` (4,065/5,000) — a *lower* bound |
| classical | 5,000 | #1084 DEV, quiesced only | 314 | — | — | — | 34,252 | 38,645 | `quiesced` |

The classical p100 on real kernels is `glyph16:U+005F` / `glyph32:U+005F` (the
underscore, 119 nodes): 55,242 applications to reach the class cap. That is the number
the budget must sit strictly above, because under today's clock those two rows stop at
22,600 applications with a `Timeout` — they are among the 49 rows (25.4%) whose kernel is
currently a function of machine load.

### Applications per e-class at stop — the invariant that makes one number serve three tiers

| tier | corpus | median | p90 | **max** |
|---|---|---:|---:|---:|
| blitz | #1084 DEV | 1.0 | — | **14.9** |
| rapid | #1084 DEV | 1.5 | — | **7.2** |
| classical | #1087 real (`ref`) | 1.7 | 10.7 | **15.6** |
| classical | #1084 DEV | 4.0 | — | **21.1** |

The ratio is bounded because the class cap's scan refuses an action whose estimated new
classes would cross the cap; once the cap is reached the only applications left are
unions and re-matches, and those run out. **21.1 applications per e-class is the highest
ratio ever observed at a tier's own class cap.**

This ratio is *not* a law to extrapolate across caps: at #1087's lifted 4× cap (20,000
classes) the same corpus reaches a max ratio of 35.8. It is used here only to set each
tier's budget at *its own* cap, where 21.1 is the measured worst.

## Decision: the application budgets

**40 applications per e-class of the tier's class cap** — 1.9× the highest ratio ever
observed (21.1), rounded to a round number at every tier:

| tier | class cap | **`max_applications`** | vs. observed p100 | vs. deterministic worst case |
|---|---:|---:|---|---|
| blitz | 500 | **20,000** | 3.2× (6,198) | 2.7× (7,467 = the clock-cut row extrapolated to its own cap) |
| rapid | 2,000 | **80,000** | 43× (1,862) | 2.7× (29,800 = blitz's worst ratio 14.9 applied to rapid's 2,000-class cap) |
| classical | 5,000 | **200,000** | 3.6× real (55,242), 2.3× DEV (85,900) | 1.9× (105,658 = the DEV clock-cut row extrapolated to its own cap) |

**The gate this satisfies:** on all 193 real core-term kernels, the deterministic stop
(class cap or quiescence) arrives at ≤ 55,242 applications, which is 27.6% of the
classical budget. The application budget binds strictly *after* the class cap or
quiescence on **every** observed kernel, in both corpora, at all three tiers. Its
measured cost on the production corpus is therefore exactly **zero** — no kernel changes.

That is the point. This is a *ceiling*, not an operating point: the class cap remains
what binds (68.4% of real kernels per #1087), the application budget exists so that the
tail terminates deterministically instead of terminating on a clock.

### What it costs when it does bind

It has never been observed to bind. If it did, 200,000 applications at the slowest
throughput #1087 measured (14 applications/ms, release profile, loaded host) is ~14 s of
saturation, and roughly 20× that at proc-macro opt-level 0. A kernel that reaches the
classical application budget is a kernel worth investigating; it is not a kernel worth
silently truncating.

## Decision: the iteration cap is kept as is

Kept — 20 / 50 / 100 rounds, unchanged.

The data says it never binds: 0 of 193 real kernels (#1087, `IterationCeiling` census
0/193; production max 8 rounds of 100, the no-clock reference max 10 of 100) and 0 of 394
DEV expressions (#1084: classical max 20 of 100, rapid max 7 of 50, blitz max 13 of 20).

Three reasons not to remove it anyway:

1. **It is already deterministic.** It is not the thing that was broken, and removing a
   working bound is a behavior change with no evidence behind it.
2. **It is not dominated by the application budget.** A round with zero applications *is*
   quiescence, so a run reaching round *N* has performed ≥ *N* applications — but the
   converse does not hold. Classical would have to average under 2,000 applications/round
   for 100 rounds to hit the round cap first, and the observed classical median is
   ~1,000–1,700 applications/round. The two bounds cut in different places.
3. **The one place it comes close is a place it should bind.** #1084's single blitz
   `timeout` row reached 13 of 20 rounds — 415 e-classes from a *10-node* expression. If
   removing the clock lets that row run to 20 rounds and stop there, that is a blitz-tier
   input behaving pathologically and the tier's job is to stop it.

**Follow-up measurement this leaves open:** the round cap has only ever been observed
under a clock that cut runs short first. Re-run the #1087 harness with no clock and check
whether blitz's 20-round cap binds on real kernels. If it does, the fix is to raise
blitz's round cap, not to lower its application budget — the application budget is the
bound calibrated against data, the round cap is not.

## Decision: the wall-clock safety ceiling

| tier | old `hard_timeout` (a budget) | **new `safety_ceiling`** (an assertion) | multiple |
|---|---:|---:|---:|
| blitz | 10 ms | **30 s** | 3,000× |
| rapid | 50 ms | **120 s** | 2,400× |
| classical | 200 ms | **300 s** | 1,500× |

Denominated as "the tier's application budget could not take this long on any machine we
would build on": a floor throughput of **1 application/ms**, which is 14× below the
slowest rate #1087 measured (14 applications/ms — 35,682 applications in 2,535.8 ms,
release profile, on a host whose 1-minute load never dropped below 4) and roughly 4×
below an opt-level-0 rate derived from it. 20,000 / 80,000 / 200,000 applications at
1 app/ms is 20 s / 80 s / 200 s, rounded up.

Far more than the "20× the old timeout" the brief floated, deliberately: the old numbers
were *operating* budgets tuned on an optimized build, and the whole failure being fixed is
that they did not survive a debug proc-macro. A ceiling that a healthy opt-level-0 build
can trip is the same bug wearing a louder shirt. The worst real kernel (55,242
applications) is ~3.9 s at 14 apps/ms and ~79 s at a 20× opt-level-0 penalty on a loaded
host — which is why classical's ceiling is 300 s and not 4 s.

**It is a build-failure condition, not a budget.** Exceeding it panics
(`Optimizer::run`'s existing `assert!`), with the elapsed time, the resolved `Limits`,
and the `SaturationStop`. It never truncates and it never changes what is emitted:
either the build produces the deterministic kernel or it fails loudly. A ceiling that can
fire is still host-dependent — that is unavoidable for a clock — and the design tolerates
it only because firing is a build failure and never a silent change of output.

**Corollary for anyone tempted to tighten it:** if a tier is too slow for a user, the
lever is `max_applications` (deterministic, changes codegen, reviewable) — never the
ceiling.

### The diagnostic override

**`PIXELFLOW_SATURATION_CEILING_MS`** — read once, at the same place and in the same
style as `PIXELFLOW_NNUE_WEIGHTS` (proc-macro expansion time for the macro tier, process
start for the runtime tier).

| value | effect |
|---|---|
| unset | the tier default above |
| a positive integer | that many milliseconds, at every tier |
| `0` or `off` | ceiling disabled — for profiling a deliberately slow run under a debugger |
| anything else | **panic**, quoting the offending value (no silent failures) |

The invariant to pin with a test: **this variable cannot change which kernel you get.**
It gates only whether the build fails. That is exactly the property the old
`hard_timeout` lacked, and it is why the application budget deliberately gets *no* env
override — an env-tunable budget would put the nondeterminism back, just spelled
differently.

## Decision: gating rule provenance

`Provenance::new()` is constructed in `EGraph`'s constructor and `record_provenance`
defaults to `true`, so every production compile builds an `origins: HashMap<ENodeId,
Origin>` (one entry per e-node ever created) and an append-only `ApplicationRecord` /
`UnionEvent` journal — up to 55,242 records on a real glyph kernel — that **no production
consumer reads**. `Optimizer::run` already narrows this with
`set_provenance_recording(self.observer.is_some())`, so production is off *through that
path*; the feature makes it off by construction, so a path that forgets the call cannot
silently start recording.

Feature name: **`provenance-journal`** (in `pixelflow-search/Cargo.toml`, **default
off**).

### Gated (compiled out when the feature is off)

- `Provenance::origins` and `record_origin` / `origin` / `origins` / `origin_count`
- `Provenance::applications` (the `Vec<ApplicationRecord>`), `record_application`,
  `application`, `applications`, and **`Provenance::application_count`** — the journal's
  *length*, a `usize`, which is a different quantity from the kept
  `EGraph::application_count() -> u64` despite the identical name. The two agree today
  only because the journal records every application; gating the journal is precisely
  what separates them, so the implementation should rename the surviving journal one
  (`Provenance::recorded_count`) rather than leave two same-named functions whose
  agreement is now conditional
- the union journal: `UnionEvent`, `Provenance::unions`, `union_events`, `union_count`
- `derivation_ancestors`, `EGraph::derivation_ancestors`, and `format_derivation_trace`
- `EGraph::active_application` and the `ActiveApplication` bookkeeping inside
  `apply_action_from_rule`
- `Optimizer::observe`, the `Observer` trait, and `Optimizer::run`'s drain loop
- `set_provenance_recording` / `provenance_recording` (the runtime flag has nothing left
  to gate when the journal is compiled out)
- `saturation-telemetry`'s `journal_unions` column (the journal-vs-counter cross-check —
  a research quantity; telemetry keeps `SaturationStats::total_unions`, which is counted
  unconditionally). **`saturation-telemetry` must not imply `provenance-journal`**, or the
  cheap production-shaped telemetry would drag the expensive journal back in.

### Kept unconditional

- `EGraph::applications: u64` and `EGraph::application_count()` — the budget's
  denominator. Already documented as counted "whether or not provenance is being
  recorded ... the budget must not depend on whether anyone is watching"; the feature
  must not weaken that.
- `ApplicationId` the type and its sequencing contract (`ApplicationId(n)` is the *n*-th
  application), which is derivable from the counter alone.
- `SaturationStats::applications`, `OptimizerStats::applications`, `SaturationStop`,
  `ApplyResult::scan` and its `ScanStop` variants (including `ScanStop::ApplicationBudget`, which is how the new budget reports itself), `SaturationStats::total_unions`.

### Who enables it

| crate | `provenance-journal` | why |
|---|---|---|
| `pixelflow-pipeline` | **on** (unconditional dependency feature) | the labeler, the guided-saturation harnesses, `derivation_ancestors` |
| `pixelflow-search` dev-dependency on itself | **on** | `cargo test -p pixelflow-search` covers the journal |
| `pixelflow-compiler` | **must not** | the `kernel!` expansion path |
| `pixelflow-codegen` | **must not** | JIT compile path |
| `pixelflow-runtime` | **must not** | — |
| `core-term` | **must not** | — |
| `pixelflow-core`, `pixelflow-graphics`, `pixelflow-ir` | **must not** | — |

**Honest caveat about feature unification.** "Default off in production crates" is a
property of `cargo build -p core-term`, not of `cargo build --workspace`: within one
`--workspace` invocation Cargo unifies `pixelflow-pipeline`'s `on` into every other
member's `pixelflow-search`. The guarantee that matters — that a `kernel!` expansion in a
*downstream user's* build does not journal — holds because nothing in
`pixelflow-compiler`'s dependency graph enables it. Pin it with a gate:

```
cargo tree -p pixelflow-compiler -e normal -f "{p} {f}" | grep provenance-journal   # must be empty
cargo tree -p core-term          -e normal -f "{p} {f}" | grep provenance-journal   # must be empty
```

## Decision: one saturation loop, not two

Already true, and this change must not undo it. `EGraph::saturate_bounded(max_iters,
max_classes, max_applications, timeout)` is the sole rewrite loop in the crate;
`saturate_budgeted` is its public face with `timeout: None`, and `Optimizer::run` is the
sole caller from all three production tiers (`ir_bridge.rs:734` for the macro tier,
`runtime.rs:166` for the runtime tier, `ir_bridge.rs:1155` for the `Dwrt` tier).

So there is no "anytime loop" to merge with a "production loop" — the anytime property
*is* `saturate_budgeted`'s `max_applications` argument, and production's loop becomes an
anytime loop the moment `Budget::Production` stops resolving `applications: None`. The
research arms already run `Budget::Applications(n)` through the same function.

**The Guide plugs in as ordering, never as a second loop.** The seams are
`Optimizer::rules` (which rules, in which order) and `Optimizer::rerank` (which
extraction), both of which leave the budget dimension untouched — that is the L4
guide-neutrality property `optimizer.rs`'s module docs already register. Any future
Guide that wants a per-application decision belongs inside `saturate_bounded`'s
candidate scan, behind the same `max_applications` counter, not in a parallel loop.
J11 was two loops disagreeing about when to stop; the fix holds only as long as
`saturate_bounded` stays the one place that decides.

## Implementation checklist (next change)

1. `SaturationConfig`: add `max_applications: u64`; rename `hard_timeout` →
   `safety_ceiling`. Presets get the six constants above.
2. `Budget::Production` resolves `applications: Some(preset.max_applications)`.
3. `Optimizer::production()` sets `.hard_ceiling(preset.safety_ceiling)`, honoring
   `PIXELFLOW_SATURATION_CEILING_MS`.
4. `provenance-journal` feature; `pixelflow-pipeline` and the self dev-dependency enable
   it; add the two `cargo tree` gates to CI.
5. Tests: (a) every tier's application budget exceeds that tier's observed p100 above;
   (b) `PIXELFLOW_SATURATION_CEILING_MS` changes no extracted arena; (c) a run that
   exceeds its ceiling panics rather than truncating; (d) `application_count()` is
   identical with the journal feature on and off.

## Revision: the classical cap grows with the inserted input (2026-09-08)

**Measured:** `docs/results/2026-09-08-class-cap-sweep.{md,csv,json}`, with the
harness `egraph_off_on run --class-cap / --classes-per-inserted` and `cap-sweep`.

### What was wrong with a flat 5,000

The flat cap was calibrated as memory protection and as "the bound that actually
binds on real kernels" (above). On the glyph family it binds *before the first round
completes*: a `Kernel` built by composition inserts to 600–4,000 hash-consed classes
(`glyph16:U+0038` 3,405; `U+0025` 3,994), and the production rule set's first sweep
grows every glyph to between **6× and 7×** its inserted size — at 6 classes per
inserted class the same 44 of 95 glyphs stop on `ClassCap` inside round 1 as at the flat
5,000, at 7 none do. Below that the cap is not a budget, it is a truncation of the
input's own rewrite frontier, and the numbers this document calibrated on those 44
kernels (`ref_apps` at `ClassCap`, 1 round) describe the cap, not the kernels.

### The rule

| dimension | classical, before | **classical, now** | how |
|---|---:|---:|---|
| class cap | 5,000 | **`clamp(8 × inserted_classes, 5,000, CLASSICAL_CLASS_CEILING)`**, the ceiling **pinned at 5,000** (calibrated: 50,000) | `SaturationConfig::classical_for(inserted)` via `config_for_input(InputSize { nodes, classes })`; `Optimizer::run` reads `egraph.num_classes()` on entry, which is the inserted count |
| application budget | 200,000 | **40 × the resolved cap** (200,000 at the floor, 2,000,000 at the ceiling) | `APPLICATIONS_PER_CLASS`, this document's ratio, now derived at every cap rather than copied at three |
| safety ceiling | 300 s | **1.5 ms × the application budget** (300 s at the floor, 50 min at the ceiling) | `SAFETY_CEILING_PER_APPLICATION`: the 1 application/ms floor throughput above, rounded up by half, which reproduces 30 s / 120 s / 300 s at the three presets |
| iteration cap | 100 | 100 | unchanged |
| blitz, rapid | 500 / 2,000 | unchanged | their inputs never approach their caps |
| tier key | node count | node count (tier), **inserted classes (classical cap)** | see below |

`SaturationConfig::classical()` keeps its name and its constants as the **floor** — what
every classical kernel of at most 625 inserted classes still gets, and what an offline
caller with only a node count (`config_for_node_count`) resolves to.

**Why the inserted class count and not the node count.** The tier is keyed on a rough
node count; the cap cannot be, because composition re-expands shared subtrees: a glyph is
2.3 legalized nodes per inserted class, the chrome scene is 840 (a 390,815-node tree that
hash-conses to 465 classes). Sized by nodes at the same ratio, chrome would get the
ceiling — and at the ceiling its extraction is 42% dearer in `dag_cost` and 39% larger
(the sweep's chrome rows, `cap100000-*`). Sized by inserted classes it keeps the floor it
had, byte-identical.

### Calibration

Per-inserted-class ratios 6, 7, 8, 10, 12 (floor 5,000, ceiling 50,000), 95 DejaVu glyphs
at tile 16, against the flat 5,000:

| ratio | glyphs on `ClassCap` in round 1 | Σ `dag_cost` | Σ bytes | Σ guarded | spills | Σ compile (95 glyphs) | max per glyph | peak heap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flat 5,000 | **44** | 392,757 | 852,052 | 74,213 | 1,074 | 6.6 s | 0.27 s | 3.7 MB |
| 6 | **44** (the same 44) | −10.7% | −1.2% | 51,395 | 69 | 9.7 s | 0.46 s | 11.6 MB |
| **7** | **0** | **−16.7%** | −0.3% | 51,347 | 21 | 19.7 s | 1.3 s | 14.8 MB |
| **8 (shipped)** | **0** | **−16.7%** | −0.3% | 51,395 | 20 | 25.5 s | 2.5 s | 16.1 MB |
| 10 | 0 | −16.7% | −0.8% | 51,395 | 19 | 43.4 s | 4.8 s | 19.7 MB |
| 12 | 0 | −16.4% | −1.0% | 51,503 | 26 | 47.2 s | 5.7 s | 22.7 MB |
| flat 50,000 | 0 | −10.8% (14 glyphs worse) | −7.3% | 38,260 | 86 | 1,662 s | 86 s | 23.1 MB |

Compile times are under the harness's counting allocator on a shared host (1-minute load
5–18); ratios between rows of the same table are the claim, the seconds are a sign.

The threshold is 7 (the smallest ratio at which no DEV glyph stops on the cap in its
first round, which is the pick rule the sweep was run under); 7 and 8 extract identical
terms. **8 is shipped**: the transition at 6 → 7 is a cliff — the same 44 glyphs, none
of them marginal — so a ratio sitting on it leaves a held-out font's outlines one
seventh from clipping, and the seventh costs 30% more compile time on the raised
glyphs only.

The application budget never bound on any arm of the sweep, including the flat 100,000
(chrome reaches 142,134 applications at 85,977 classes; cellgrid 165,997 at 90,816): the
ratio of applications to classes at stop stays under 2 at every cap, and the 40-per-class
budget remains the tail's deterministic terminator, not an operating point. The
safety ceiling never fired.

### Why the ceiling is pinned at the floor

With the ceiling at 50,000 the rule raises `'8'` (3,405 inserted classes) to 27,240, the
extractor prefers a different fusion of its quadratic solver, and `disc >= 0` at the
waist's tangency — exact zero at a shared extremum, decided by one rounding of
`Y·slope + c` against two — lands on the other side: a half-covered smear along the waist
at 13, 15, 17, 19 and 21 px, 15 texels where FreeType has no ink within a texel
(`freetype_oracle.rs`, the optimized arm added by this change), 37 texels where the
optimized arena differs from the raw one by ~0.5 (`kernel_glyph_optimize`). The
production tiles (16 and 32 px) are clean (cross-form |Δ| ≤ 6.4e-6). This is the `'8'`
defect that the raw-arm oracle found and that deleting the second optimizer removed
(docs/plans/2026-09-08-macro-tier-is-arena-native.md), back by the route the oracle's own
comment predicted. The knife edge belongs to the glyph kernel (`quad_tangency_winding.rs`)
and the fix does too; until it lands, `CLASSICAL_CLASS_CEILING = CLASSICAL_CLASS_FLOOR`
and the rule is inert. The enable is one constant (`CLASSICAL_CLASS_CEILING_CALIBRATED`),
and the optimized-arm oracle is the gate that says whether it may be flipped.

### What it costs, and what it buys (measured with the ceiling at 50,000)

Against the flat 5,000 on the DEV corpus (deterministic columns; `2026-09-08-class-cap-sweep.md`):

- **glyphs (both tiles — the tile scales a constant, the 190 rows are pairwise
  identical in every deterministic column):** 44 of 95 raised (to 5,448–31,952), Σ
  `dag_cost` −16.7%, Σ guarded entries −31%, spill slots 1,074 → 20, Σ bytes −0.3%, no
  glyph dearer in `dag_cost`, 18 larger in bytes; Σ compile per 95-glyph warm **6.6 s →
  25.5 s (+290%)** under the counting allocator, all of it saturation on the 44 raised
  glyphs (the 51 at the floor are byte-identical and cost what they did).
- **cell grid, psychedelic, the twelve `shader_bench` ports, chrome:** at most 465
  inserted classes → the floor → byte-identical to before.

The +5% bound this document's successor put on the glyph warm is exceeded by a factor of
~60. The raise was directed anyway ("the cap is smaller than the input; the rules never
run"); it is stated at the top of the PR rather than hidden in a ratio — and it is not
what blocked the raise; the `'8'` waist is.

### The reading the sweep gives the research question

Quality does not improve with budget; it improves with *enough* budget and then gets
worse. On the 44 raised glyphs, the flat 50,000 (5× to 50× what they need) extracts
11% cheaper than the floor where 8 per inserted class extracts 18% cheaper — 14 mid-size
glyphs (`(`, `)`, `D`, `P`, `J`, `f`, `h`, `j`, `l`, `n`, `r`, `t`, `u`, `y`) are 15–20%
dearer at 50,000 than at 5,000. On chrome every raise past the floor is a loss (+42%
`dag_cost` at 100,000, and at 100,000 the tree objective wins the two-objective race).
On the shaders +8%. Extraction cannot rank the larger space; that is the net's job, and
the budget is not the lever past the input's own frontier — which is exactly where the
new rule stops.
