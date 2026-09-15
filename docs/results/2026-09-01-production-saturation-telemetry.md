> **Retracted/Superseded (2026-09-07), ledger L054.** The "what the class cap costs" figures (median 8.66%, p90 13.22%, max 15.07% vs a 4x-lifted reference) are tree-DP costs whose sign the chrome clock contradicted (12x more classes = +15% ns/px, L072); the stop-reason counts, the 8,446-application regime and the "more saturation extracts worse code" rows (L053, L055, L056) stand. Verdict and rationale: `docs/results/2026-09-07-claims-ledger.md` (PR #1207); the corrected benchmark and re-validation order: `docs/plans/2026-09-07-benchmark-correction.md`.

# Production saturation telemetry: does the budget bind on real kernels? (2026-09-01)

Two rounds. **Round 1** (commit `b2d1e48e`, base `origin/main` 453d2a6e) measured 173/193 kernels
with a stop reason *inferred* from reference runs; its data is kept as
`2026-09-01-production-saturation-telemetry-round1-inferred.csv`. **Round 2** (this PR's harness
commits rebased onto `origin/main` 7d7eabfa) re-measured all **193/193** kernels with the stop
reason **read off the typed `SaturationResult::stop`** the loop itself sets; its data is
`2026-09-01-production-saturation-telemetry.csv` and every number below is Round 2's unless
labelled otherwise. Both were run in the dedicated worktree `.claude/worktrees/saturation-telemetry`.
This is the follow-up the integration audit
(`docs/results/2026-09-01-integration-audit.md`, open question 1) named as "the first
measurement to take before arguing about budgets": `optimize_runtime_arena_uncached`
(`pixelflow-search/src/runtime.rs:106-137`) computes a `SaturationResult` and drops it, so
nobody has known whether a real core-term kernel quiesces or is cut off by the iteration cap,
the class cap, or the 200ms wall-clock ceiling.

## Flat answer

**Yes, production's budget binds on real kernels — via the class cap, not the iteration cap —
and what the cap costs is a steady ~9% on most glyphs (never more than 15%), 0% on the
cell grid, and up to 47% on the rows the 200ms clock cuts.** Across every kernel core-term
actually compiles (the 623-node packed cell-grid program at three geometries and all 190
printable-ASCII glyph kernels at both bake densities), the typed stop reason says:

| stop (read from `SaturationResult::stop`) | rows | share |
|---|---:|---:|
| `Quiesced` — a full rule sweep ran to completion with zero unions | 12 | 6.2% |
| `ClassCap` — the 5,000-e-class budget truncated the sweep | 132 | **68.4%** |
| `Timeout` — the 200ms clock expired first (machine-dependent) | 49 | 25.4% |
| `IterationCeiling` — 100 rounds completed | 0 | 0% |

The class-cap share is a **lower bound**: every `Timeout` row is a race between the clock and
the cap on a loaded host (this run's 1-minute load never dropped below 4 — see Machine state),
and three rows that were `ClassCap` in Round 1 flipped to `Timeout` in Round 2 and three the
other way, on identical inputs. On a quiet machine some of the 49 would be `ClassCap`; none would
be `Quiesced` or `IterationCeiling`.

**Truncation cost.** The structural cost of the class cap — production's extracted latency-prior
cost vs. the same run with the class cap lifted 4× (20,000) and no clock, on the 132 `ClassCap`
rows — is **median 8.66%, p90 13.22%, max 15.07%** (107/132 rows non-zero; the worst is `&`,
`U+0026`, at 15.07% at both densities). That figure is itself a lower bound: the lifted reference
stops `ClassCap` on 128 of those 132 rows, so its cost is not the saturated optimum either (see
"Loss vs lifted is a lower bound"). The clock's cost — production vs. the same class cap with
no clock, which is **0% by construction on every `ClassCap` and `Quiesced` row** (132/132 and
12/12 rows reproduce the unclocked run's signature exactly) — is, on the 49 `Timeout` rows,
**median 11.01%, p90 35.51%, max 47.17%** (33/49 non-zero; `glyph16:U+0078`, the lowercase
`x`, and its density-2 twin are the worst at either density). The cell-grid kernel — the one
kernel every core-term session compiles at startup and on every resize — is class-cap-bound at
every geometry and shows **0% loss** against both references: extraction already finds the
answer the cap-lifted run finds. **24/193 (12.4%) rows show more saturation producing a *worse*
extracted cost** (20 of them a lifted run strictly worse than production; not a bug — the static
latency-prior DP is not monotone in graph size; see Anomalies) — a genuine caveat for anyone
treating "more budget" as strictly good.

## Coverage: 193/193

The full production population is 3 cell-grid geometries + 190 glyph kernels (95 printable-ASCII
chars × 2 bake densities) = 193. Round 2 measured all 193 in one pass (54.5 min wall clock,
`finished in 3272.15s`), including the 20 density-2.0 glyphs `U+006B` 'k' through `U+007E` '~'
that Round 1 never reached. The per-kernel 1200 s harness ceiling on the generous runs never
bound ("loss unmeasured: 0"), so every row is a complete production/reference/lifted triple.
The 20 rows new in Round 2 behave exactly like their density-1.0 twins: same node count
(95/95 twin pairs have identical node counts), same typed stop on 20/20, and identical extracted
cost on 19/20 (`k`, `U+006B`, differs by a handful of constant-fold applications because the baked
density constants differ).

## Per-kernel-class summary (Round 2, typed stops)

| group | n | quiesced | class cap | timeout (machine-dependent) | median loss vs ref | p90 loss vs ref | max loss vs ref | median loss vs lifted | p90 loss vs lifted | max loss vs lifted | median apps | max apps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cellgrid | 3 | 0 | 3 | 0 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 13,286 | 13,286 |
| glyph16 (density 1.0) | 95 | 6 | 64 | 25 | 0.00% | 15.83% | 47.17% | 8.64% | 17.34% | 47.17% | 8,164 | 40,606 |
| glyph32 (density 2.0) | 95 | 6 | 65 | 24 | 0.00% | 15.91% | 47.17% | 8.89% | 21.37% | 47.17% | 8,164 | 40,605 |
| **ALL** | **193** | **12** | **132** | **49** | **0.00%** | **15.83%** | **47.17%** | **8.63%** | **17.34%** | **47.17%** | **8,446** | **40,606** |

By stop reason, which is the split that means something:

| production stop | n | loss vs ref (same cap, no clock) | loss vs lifted (4× cap, no clock) |
|---|---:|---|---|
| `Quiesced` | 12 | 0% on all 10 defined (2 are the 1-node space glyph, 0/0) | 0% on all 10 defined |
| `ClassCap` | 132 | 0% on 132/132 — by construction | **median 8.66%, p90 13.22%, max 15.07%** (107 non-zero) |
| `Timeout` | 49 | **median 11.01%, p90 35.51%, max 47.17%** (33 non-zero) | median 11.01%, p90 36.09%, max 47.17% |

"loss vs ref" = `cost_production / cost_(4×-iterations, same class cap, no clock) − 1`; "loss vs
lifted" = against a reference that also lifts the class cap 4×. Both generous runs share a
1200 s per-kernel harness ceiling that never bound. The two space-character rows (a 1-node
kernel with production cost 0) have an undefined (0/0) loss and are excluded from the
percentile columns, not counted as 0% or as an outlier.

**Read the two loss columns as measuring different things.** "loss vs ref" can only be non-zero
on a `Timeout` row: a `ClassCap` row's reference hits the same 5,000-class budget at the same
round and extracts the same arena (132/132 rows have `cost == ref_cost` and, stronger, the same
`(iters, classes, apps, cost)` signature), and a `Quiesced` row's reference has nothing left to
do (12/12 identical signatures). The ALL-row "median 0.00% / p90 15.83%" is therefore a
Timeout-row statistic diluted by 142 structural zeros. The column that measures what the class
cap costs is "loss vs lifted" on the `ClassCap` rows — the headline of this document. Round 1
led with "median 0%, p90 ~16-19%, worst case 47.2%"; that was the diluted column.

## Loss vs lifted is a lower bound

The lifted reference is not an unbounded run: it lifts the class cap 4× (to 20,000) and the
iteration cap 4×, with no clock. Its typed `lifted_stop` is `ClassCap` on **175/193 rows** — 128
of the 132 `ClassCap` rows and 47 of the 49 `Timeout` rows — meaning the "reference" was itself
cut off by its own class budget before quiescing, at anywhere from 12,944 to 19,925 classes
(the budget scan refuses an action whose *estimated* new nodes would cross the cap, so a
truncated run can sit well under 20,000). On those rows the lifted cost is a budget-truncated
answer, not the saturated optimum under this rule set, and the true class-cap truncation cost is
**at least** the reported "loss vs lifted", possibly more. Only on the 18 rows whose lifted run
reports `Quiesced` (all 12 `Quiesced` rows, 4 `ClassCap` rows — `+` and `<` at both densities,
all at 0% loss — and 2 `Timeout` rows, `>` at both densities) is the lifted cost the fully
saturated answer. The Anomalies section below is the other side of the same coin: because
extraction is not monotone in graph size, a larger budget is not guaranteed to lower the bound
either.

This is also where Round 1's inference was most wrong. Round 1 called a lifted run `Quiesced`
whenever its class count was not near the lifted cap, and `NearLiftedCap` otherwise; the typed
loop says **39 of Round 1's 56 "lifted quiesced" rows were budget-truncated** (`ClassCap`, at
12,944-17,976 classes, identical lifted signatures in both rounds — the runs were the same, only
the label was invented). Round 1's "56 rows where the lifted cost is the saturated optimum" is
18.

### Round 1 stop labels were inferred, and are suspect

Round 1's 173-row table (commit `b2d1e48e`, now `69c0dc34` after the rebase; data kept as
`2026-09-01-production-saturation-telemetry-round1-inferred.csv`) did not read its stop reason
off the loop that decided to stop. At that commit production's `SaturationStats`/`SaturationResult`
(`pixelflow-search/src/egraph/{graph,saturate}.rs`) carried only `iterations`, `total_unions`,
and a `saturated: bool` that conflated "converged" with "hit the class cap while doing so"
(`saturate.rs`, `stats.iterations < max_iterations || stats.total_unions == 0`). The Round 1
harness therefore *inferred* a label from three runs: if the reference (4× iterations, same class
cap) stopped at the same round as production *and* lifting the class cap changed the outcome,
"class cap"; if production used fewer rounds than its iteration cap and neither of those held,
"timeout"; else "quiesced". Three reasons that inference is not trustworthy row by row:

1. **"Timeout" is a residual bucket.** Anything that fails the class-cap signature lands there.
   The signature itself is not equivalent to cap-bound-ness: 5 of the 117 ClassCap rows have a
   lifted run that *quiesces* without changing the outcome, and 5 of the 45 Timeout rows have a
   lifted run that hits `NearLiftedCap` — both directions of the "lifting changes the outcome"
   test fire on rows the other label would claim.
2. **The reference runs were timed on a loaded machine.** The same-round comparison that decides
   "class cap" vs "timeout" was taken while this machine's 1-minute load ran between 7.7 and
   181 (see Machine state). A production run that would have reached the cap at round *k* on an
   idle machine can be cut by the 200ms ceiling at round *k−1* under contention, and the
   inference then records it as Timeout with no way to tell the two apart.
3. **The production flag it leaned on was wrong in the same direction.** A sweep the class cap
   truncates mid-round with zero unions recorded reports `saturated == true`; Round 1's harness
   used that flag as a tie-breaker. `5b6f36b2` (pre-rebase `c96e0b04`) fixes the classification
   in production (a budget-truncated sweep is `ClassCap`, never quiescence) and
   `tests/saturation_stop.rs` pins all four reasons.

This PR now **types the stop reason in production** (`af821180` + `5b6f36b2`, pre-rebase
`a3d789c7` + `c96e0b04`: `pub enum SaturationStop { Quiesced, ClassCap, IterationCeiling,
Timeout }`, carried as `pub stop` on both `SaturationStats` and `SaturationResult`, re-exported
from `pixelflow_search::egraph`, fed by `ApplyResult::truncated`), and the harness reads it
(`79ad9c93`, `4fc8cdd0`; pre-rebase `81c52fd2`, `2d5940cc`) instead of inferring it — the three
`classify*` functions are gone, and the two generous runs survive only as the truncation-loss
measurement. Round 2 below re-measures all 193 kernels with the typed field.

### Round 2: what changed when the label was read instead of inferred

Round 2 re-ran all 193 kernels through the same harness with `stop` read off the type, on the
same machine, still loaded (Machine state). On the 173 rows both rounds measured:

| | count |
|---|---:|
| production stop label identical in both rounds | **167/173** (114 `ClassCap`, 42 `Timeout`, 11 `Quiesced`) |
| `ClassCap` (Round 1, inferred) → `Timeout` (Round 2, typed) | 3 — `glyph16:U+003E`, `U+004E`, `U+0076` |
| `Timeout` (Round 1) → `ClassCap` (Round 2) | 3 — `glyph32:U+0031`, `U+0041`, `U+0069` |
| Round 1 `ClassCap` labels wrong *for their own run* | **2/117** — `glyph16:U+0076` (208.6 ms, 19,266 apps vs the unclocked reference's 19,427) and `glyph32:U+003C` (200.5 ms, 35,831 vs 40,605): the clock had cut both short of the same-cap reference and the inference still said `ClassCap` |
| Round 1 `Quiesced` labels wrong | 0/11 |
| Round 1 lifted-run `Quiesced` labels wrong | **39/56** (see "Loss vs lifted is a lower bound") |
| Round 2 typed `ClassCap` rows whose production signature equals the unclocked same-cap reference's | 132/132 |
| Round 2 typed `Timeout` rows whose production signature differs from it | 49/49 |
| Round 2 typed `Quiesced` rows whose lifted run also quiesces | 12/12 |

So the typed field is consistent with the reference runs on every row, in both directions, which
the inference was not. Of the 6 production labels that differ between rounds, 4 are the clock
landing differently on the same input under different load (different `apps`, `elapsed_ms` on
both sides of 200 ms), and 2 are the inference errors above. The stop census moved from
11/117/45 (Round 1, 173 rows) to 12/132/49 (Round 2, 193 rows) — 6.2% / 68.4% / 25.4% — with
the 20 new rows contributing 1/15/4.

Deterministic quantities reproduce between rounds exactly where the clock was not involved:
on the 128 rows that were non-`Timeout` in Round 1 and non-`Timeout` in Round 2, every one of
`nodes, tier, iters, classes, apps, unions, journal_unions, cost, dp_cost, ext_nodes` and all
generous-run counts match to the digit. The only non-`Timeout` Round 1 rows with a mismatch are
the three that became `Timeout` in Round 2 (`glyph16:U+003E` 40,709 → 39,398 apps; `U+004E`
8 → 7 rounds; `U+0076`, whose Round 1 run was already clock-cut) plus `glyph32:U+003C`
(35,831 → 40,605 apps, Round 1's clock-cut run reaching the cap this time). Of Round 1's 45
`Timeout` rows, 34 retraced the identical production trajectory in Round 2 and 38 of the 42 that
stayed `Timeout` report the identical loss.

## Cell-grid kernel (all 3 geometries measured)

623 reachable nodes at **every** geometry (80×24 density 1.0, 80×24 density 2.0, 120×40
density 2.0) — confirmed by dumping the real `CellGridPackedProgram::compile` arena
(`pixelflow-core/src/lattice/cell_grid.rs`, pinned test precedent at `cell_grid.rs:1427-1446`)
at core-term's actual startup/resize geometries (`cell_width_px: 10, cell_height_px: 16` from
`core-term/src/config.rs:205-206`, `ATLAS_CAPACITY = 128` from `core-term/src/terminal_app.rs:57`).
Node count does not depend on grid dimensions, only on the packed-kernel's fixed structure — the
per-cell loop is data, not unrolled arena nodes.

All three land in the classical tier (>50 nodes) and are class-cap-bound: 7 rounds, settling at
4,768 e-classes (under the 5,000 cap — the cap-bound signature here is the trajectory diverging
once the cap is lifted to 20,000, not classes sitting exactly at 5,000), ~13,283-13,286
applications, 65-67ms wall-clock (comfortably under the 200ms budget — this kernel was never
`Timeout`). **0% loss vs. both references** at every geometry: extraction already finds the
cheapest form the cap-lifted run finds. This is the one kernel every core-term session compiles
at startup and recompiles on every resize, so it is the most consequential row in this table —
and its answer is that the cap technically binds but the shipped output is not worse for it.

## Glyph kernels (all 190 measured)

Font: `NotoSansMono-Regular.ttf`, sha1 `4999024f5b6037cb4c98c5d82cd1228acebb65d7` — byte-identical
between `pixelflow-graphics/assets/NotoSansMono-Regular.ttf` (used by the dumper) and
`assets/font/Noto_Sans_Mono/static/NotoSansMono-Regular.ttf` (core-term's actual bundled asset;
`core-term/src/terminal_app.rs:54` names the same filename core-term loads). All 95 printable-ASCII
characters compiled with `Font::glyph_kernel_scaled`, the exact call `GlyphAtlas::warm` makes
(`pixelflow-graphics/src/fonts/atlas.rs:168-184`), matching core-term's real startup warm
(density 1.0, `terminal_app.rs:201-205`) and its post-`WindowCreated` HiDPI rewarm (density 2.0,
`terminal_app.rs:239-245`). No glyph was missing from the font at either density.

Node counts range from 1 (the space character — a literal zero-content leaf) to 12,056 (`@`,
`U+0040` — the single largest kernel measured, whose production run gets **zero rounds**: the
initial arena already exceeds the 5,000-class cap before a single rewrite fires, so `iterations`,
`applications`, and `total_unions` are all 0, its typed stop is `ClassCap`, and the reported cost
is the raw unoptimized arena's cost). Median glyph size 2,364 nodes. **Node count is
density-independent**: all 95 characters have identical node counts in their `glyph16:*` and
`glyph32:*` rows — the bake density is a runtime affine-scale parameter substituted after arena
construction, not baked into the arena's shape. The trajectories are nearly but not exactly
density-independent: of the 70 twin pairs where neither run was clock-cut, 64 have identical
`(iters, classes, apps, cost)`, and the other 6 differ by a few classes or constant-fold
applications because the density constants that *are* in the arena differ.

The dominant stop reason is the class cap (64/95 density-1.0, 65/95 density-2.0), followed by
the 200ms clock (25/95, 24/95), with true quiescence rare (6/95 each: space, `-`, `=`, `[`, `]`,
`|` — the straight-segment glyphs). Class-cap loss is tightly clustered: median 8.89% / 9.12%
(density 1.0 / 2.0), p90 13.22%, max 15.07% (`&`, then `8` at 14.89% and `9` at 14.54%). Clock
loss is heavily right-skewed: 16 of the 49 `Timeout` rows lose nothing, but the worst cases —
small, punctuation-heavy glyphs whose segment count produces many symmetric equivalent rewrites
(`x`, `X`, `*`, `"`, `#`) — lose 30-47%. `x`/`X` (`U+0078`/`U+0058`, both 343 nodes) are the
single worst kernels measured at either density.

## Anomalies: more saturation sometimes extracts *worse* code

24/193 rows (12.4%; 16 `ClassCap`, 8 `Timeout`) flag a non-fatal anomaly: `prod.cost < refr.cost`
or `prod.cost < lifted.cost` or `refr.cost < lifted.cost` — i.e., a run that saturated *less*
nonetheless extracted a *cheaper* arena than a run that saturated *more*. 20 of the 24 are a
lifted run strictly worse than production. This is not evidence of nondeterminism (a separate,
*fatal* check — comparing full trajectory signatures whenever production stopped without the
clock's involvement — never fired across 193 rows, and 128 of 128 clock-free rows reproduced
Round 1 to the digit) — it is the static latency-prior DP extraction genuinely finding a worse
answer in a larger e-graph, which is possible whenever the DP's local per-e-class choice isn't
provably monotone under further merges. Example: `U+0056` (`V`, both densities), production cost
2,586, reference (same cap, no clock) also 2,586, but the class-cap-lifted run costs 2,618 — 1.2%
*worse* despite ~5,000 additional e-classes and tens of thousands of additional rule
applications to work with. This is a real caveat for the Guide research program: "run it longer"
is not a safe default assumption for extraction quality under the current static cost model.

## Effective B in the Guide's units

For the 10 non-trivial rows that quiesce (the space glyph's two rows quiesce at 0
applications), quiescence under the same-cap reference arrives at **15,080-21,494** rule
applications (median 21,364) — on the rare kernel that genuinely finishes, it takes on the order
of 20,000 applications, far beyond anything the Guide registration's budgets name. Across all
193 rows, production's own `apps` (rule applications actually fired before its stop, whatever
the reason) has median **8,446**, p90 **22,050**, max **40,606**. Neither of these is directly
comparable to the Guide registration's synthetic budgets (`origin/claude/phase3-guide`,
`docs/plans/2026-09-01-phase3-registration.md`: at **B=100 applications the median synthetic
classical expression is 48.5% worse than its own 4B state**) — that corpus is not real core-term
kernels, and the ~85× gap between B=100 and production's median 8,446 applications is itself a
finding: real kernels are far larger, in applications terms, than the synthetic corpus the
Guide experiment was calibrated against. Read together: production already spends ~85× the
synthetic B on a real kernel and still leaves a median 8.66% (≥) on the table at the class cap,
so on real kernels **the Guide's quality headroom is the ~9-15% the cap costs on the 132
cap-bound glyphs and the up-to-47% the clock costs on the 49 clock-cut ones, while on the cell
grid and the quiesced rows — where output already equals the lifted answer — the only thing left
to buy is compile latency.**

## Anytime cost at 25/50/100/200/400 applications: skipped, confirmed unavailable on main

The task asked for this if pixelflow-search on `main` exposes an application-denominated
saturation runner. It does not: `grep -rn saturate_guided --include='*.rs' .` on `origin/main`
returns nothing, and `pixelflow-search/src/egraph/` has no `anytime.rs`
(`docs/results/2026-09-01-integration-audit.md` §1 confirms the same via its own independent
trace). `anytime.rs` and an application-denominated runner exist only on
`origin/claude/phase3-guide` (confirmed present there, not merged to `main`). Per the task's own
instruction, this is skipped rather than measured against a branch this worktree does not build
from.

## Winding-kernel Dwrt macro-time e-graph: reachable, and it never runs

The glyph winding segment's coverage kernel (`pixelflow-graphics/src/fonts/ttf_curve_analytical.rs:106-129`,
the one `kernel_value!` in the codebase using `DX`/`DY`) has a *second* saturation site at
macro-expansion time: `differentiate_in_optimizer`
(`pixelflow-compiler/src/ir_bridge.rs:661-732`), budget 100 iterations / 10,000 classes / 500ms
(the hardcoded `EGraph::saturate()` default, **not** `config_for_node_count`-tiered). Driving the
exact front-end pipeline the `kernel_value!` proc-macro runs (`parser::parse` → `sema::analyze` →
`optimize::optimize`, `pixelflow-compiler/src/lib.rs:223-238`) on the winding kernel's literal
closure source, from a `#[cfg(test)]` module added to `ir_bridge.rs` (private items reachable,
no visibility change), reproduces the identical pre-Dwrt-resolution arena (39 nodes after the
first, algebra-only e-graph) that `ast_to_runtime_arena` would feed to
`differentiate_in_optimizer`.

**Zero saturation happens.** `differentiate_in_optimizer`'s own `representable` guard
(`ir_bridge.rs:687-694`) rejects the arena before constructing an `EGraph` at all: the kernel's
`in_y = (Y >= y_min) & (Y < y_max)` compiles to an `OpKind::BitAnd` node, and `BitAnd` has no
e-graph `Op` (`pixelflow-search/src/egraph/ops.rs`: `OpKind::BitAnd | OpKind::BitOr => None`).
So this kernel's `Dwrt` survives macro expansion unresolved and falls through to
`ast_to_runtime_arena`'s documented fallback (`:733`): the arena is emitted unchanged, and the
runtime `lower_dwrt` **symbolic pass** — not an e-graph, a direct differentiation pass,
`runtime.rs:120` — resolves it before the *runtime* tier's e-graph saturates the composed glyph
arena. That runtime e-graph is exactly the one already measured per-glyph above (stage 1): this
kernel's Dwrt resolution has no saturation budget of its own to report, and the winding-kernel
row of this measurement is "N/A, folds into the glyph rows."

## Machine state

Before the dump phase (2026-09-01T19:58:31Z): load average 7.71/6.30/6.99, AC power, battery
100% charged, no `cargo`/`rustc` processes visibly contending (grep hits were coincidental
substring matches in unrelated MCP process argv, not builds). During the run's final, extended
stall (2026-09-01T21:25:30Z): load average spiked to as high as **181.65** (peak observed
sample) before settling back to 11.75/41.52/60.23, `pgrep -fl cargo|rustc` returned 16 live
processes — many concurrent agent sessions on this shared machine building/testing
simultaneously (unrelated `phase3-round2`, `saturation-telemetry-flag`, and other worktrees were
all independently active during this run's window). AC power and full battery held throughout.
**Every row flagged `machine_dependent=true` (the 45 `Timeout` rows) should be read as an upper
bound shaped partly by this contention, not a clean measurement of the 200ms budget's intrinsic
bite** — a rerun on an idle machine would likely show fewer Timeout rows and more of them
reclassified as ClassCap (the two are not mutually exclusive at the moment either binds; whichever
trips first is what's recorded).

**Round 2 machine state.** The re-run was gated on a 1-minute load average below 4 and polled
`uptime` once a minute for the bounded 40-minute budget the task allowed (63 polls in all across
three launches — the first two launchers died with their sessions' shells at 23 and 19 polls; the
third, detached, ran the remaining 21); the load never dropped below 4.6 (sibling agent sessions
were building throughout), so per the rule the run went ahead and is **labelled loaded**: load
averages 6.95/7.80/8.35 at start, 6.31/6.81/8.88 at end, 1-minute samples between 5.2 and 15.5
during the 54.5 minutes. Far quieter than Round 1's 7.7-181, but not quiet — which is why the 49
`Timeout` rows are a machine-dependent upper bound and the 132 `ClassCap` rows a lower bound.
`ClassCap` rows took a median 51 ms (p90 149 ms, max 194 ms) of the 200 ms budget; `Timeout`
rows were cut at 200.1-454 ms (the deadline is checked every 1,024 evaluations inside a scan).

## Integrity note: what happened to this branch, by commit

The history below is `git reflog --date=iso` in this worktree
(`.claude/worktrees/saturation-telemetry`), 2026-09-01, local time (-0700), recorded so JP can
audit it. It replaces an earlier draft of this note that described the 13:39 cherry-pick as
another session's action; the reflog does not support that reading, and it is withdrawn.

| when | reflog line | what it was |
|---|---|---|
| 12:57:58 | `rebase (start)` / `rebase (finish)` → `453d2a6e` | branch rebased onto `origin/main`; the Round 1 harness was an uncommitted `runtime.rs` diff and the Round 1 measurement was running against it |
| 13:39:03 | `reset: moving to HEAD`; `cherry-pick: feat(pixelflow-search): feature-flagged saturation telemetry (JSONL per compile)` → `7438e0af` | after the orchestrating session asked for a typed stop reason, this branch's own agent (resumed on a later message — not a sibling session) cherry-picked the whole feature commit `586d9cf0` from `origin/claude/saturation-telemetry-flag`, which rewrote the uncommitted harness in place while the measurement was running |
| 13:58:37 / 13:58:50 | `reset: moving to HEAD`; `reset: moving to 453d2a6e` | the cherry-pick backed out; the Round 1 harness diff reapplied as a patch, byte-identical to what the in-flight measurement was compiled from, so Round 1's rows are unaffected. A `git stash` taken here ("external-reset-recovery: leftover working changes") was later dropped; it survives as unreachable commit `3c8b9fea` (`git fsck --unreachable --no-reflogs`) and nothing in the tree depends on it |
| 14:29:54 | `commit` → `b2d1e48e` | Round 1 committed: harness, this document, the 173-row CSV; no production file touched |
| 14:39:42, 14:42:19 | `commit` → `63b5a125`, `10a82193` | first spelling of the typed stop (`Converged / ClassLimit / IterationLimit / Timeout`) |
| 14:48:55 | `reset: moving to b2d1e48e` | those two discarded in favour of the flag branch's reviewed hunks |
| 14:49:05 | `commit` → `81c52fd2` | harness reads the stop off the type; `classify` / `classify_reference` / `classify_lifted` deleted |
| 14:51:01 | `commit` → `a3d789c7` | **production**: the `graph.rs` / `mod.rs` / `saturate.rs` hunks of `origin/claude/saturation-telemetry-flag` that add the stop-reason type and set it at each `break` — nothing else from that branch |
| 14:51:01 | `commit` → `c96e0b04` | **production**: a budget-truncated zero-union sweep is `ClassCap`, not quiescence (`ApplyResult::truncated`); type renamed to `SaturationStop { Quiesced, ClassCap, IterationCeiling, Timeout }`, field `stop`; `tests/saturation_stop.rs` pins all four |
| 14:55:31 | `commit` → `2d5940cc` | both harnesses (`runtime.rs`, `ir_bridge.rs`) adapted to the rename |
| 2026-09-01 19:35 | `rebase` onto `origin/main` `7d7eabfa`; `commit` (this document, Round 2 CSV) | the verifier session: `b2d1e48e`/`81c52fd2`/`a3d789c7`/`c96e0b04`/`2d5940cc` became `69c0dc34`/`79ad9c93`/`af821180`/`5b6f36b2`/`4fc8cdd0`, content unchanged; Round 2 data and this text on top |

The typed stop reason is in this PR deliberately, on the orchestrating session's ruling: under
CLAUDE.md's "when you extend a type's meaning, extend its type", giving `SaturationResult` the
stop reason its own loop already decides is type-completeness, not telemetry machinery, and it
is the one production change this PR carries (`af821180`, `5b6f36b2`, reviewable or droppable
independently of the measurement). The feature-flagged per-compile JSONL telemetry of
`claude/saturation-telemetry-flag` (PR #1083) is not brought in; that branch is untouched. No
`git stash` is used on this repo from here on.

## Reproduction

Three tests, each documented with its own run command in its `#[ignore]` message:

```bash
# Stage 1a: dump the real production glyph arenas (pixelflow-graphics)
PIXELFLOW_TELEMETRY_DIR=<dir> cargo test -p pixelflow-graphics --release \
  --test production_glyph_arena_dump -- --ignored --nocapture

# Stage 1b: dump the real production cell-grid arenas (pixelflow-core)
PIXELFLOW_TELEMETRY_DIR=<dir> cargo test -p pixelflow-core --release -- \
  --ignored dump_production_cell_grid_arenas --nocapture

# Stage 1c: replay production's saturation on every dumped arena (pixelflow-search)
env -u PIXELFLOW_NNUE_WEIGHTS \
  PIXELFLOW_TELEMETRY_DIR=<dir> PIXELFLOW_TELEMETRY_OUT=<tsv-path> \
  cargo test -p pixelflow-search --release -- \
  --ignored production_saturation_telemetry --nocapture --test-threads=1

# Stage 2: the winding-kernel Dwrt macro-time e-graph (pixelflow-compiler)
env -u PIXELFLOW_NNUE_WEIGHTS \
  cargo test -p pixelflow-compiler --release --lib -- \
  --ignored winding_kernel_dwrt_egraph_telemetry --nocapture
```

`PIXELFLOW_TELEMETRY_REF_MULT` (default 4) tunes the generous-run multiplier and
`PIXELFLOW_TELEMETRY_KERNEL_CEILING_S` (default 1200 s = 20 min) is the per-kernel wall-clock
ceiling shared by the two generous runs, which otherwise carry no e-graph time budget. A generous
run the ceiling cuts is reported, never skipped: its stop reads `Timeout` off the type, the
row's loss against it is `NA`, the row is named in the run's "loss unmeasured" line, and the
row still counts toward the stop-reason census. (The Round 1 harness spelled the ceiling
`PIXELFLOW_TELEMETRY_REF_CEILING_S`, default 600 s, and panicked on it; the Stage 2 winding-kernel
test in `ir_bridge.rs` keeps that name and behaviour.) The full per-kernel data is in
`docs/results/2026-09-01-production-saturation-telemetry.csv` (Round 2, 193 rows, header
included, `stop`/`ref_stop`/`lifted_stop` read from the type; its `.meta.txt` records the load);
Round 1's 173 inferred-label rows are kept as
`2026-09-01-production-saturation-telemetry-round1-inferred.csv` for the comparison above.

## Verification (Round 2)

Done by the verifying session on 2026-09-02, after rebasing onto `origin/main` 7d7eabfa (linear,
no conflicts; the five main commits in between touch none of this PR's hunks).

**The harness is production's call.** `production_telemetry::run` in `runtime.rs` was diffed
line-for-line against `optimize_runtime_arena_uncached` (`runtime.rs:106-137`): same rule set
(`EGraph::with_rules(all_rules())`), same lowering order (`lower_dwrt_owned` first), same tier
function (`config_for_node_count(reachable_count(..))`, tiers blitz 20/500/10 ms, rapid
50/2,000/50 ms, classical 100/5,000/200 ms in `saturate.rs`), same entry
(`saturate_with_full_budget` with the tier's three limits), same extraction-policy resolution
(`env_extraction_policy()` → `policy.extraction` → `choices_to_arena`, asserted `Static` with
`PIXELFLOW_NNUE_WEIGHTS` unset). The only difference is that the harness keeps the
`SaturationResult` production drops.

**The stop is read, not inferred.** `classify`, `classify_reference`, `classify_lifted` are gone
from the tree (`grep -rn 'fn classify' pixelflow-search pixelflow-compiler` → nothing); the
harness records `result.stop` for the production run and for both generous runs, and the
`cap_lift_changed` column (the old inference's signal) is now a raw column that agrees with the
typed `ClassCap` on 132/132 rows and with `Quiesced` on 12/12 — reported, not used. The winding
kernel's Dwrt test in `ir_bridge.rs` reads `SaturationStats::stop` the same way.

**Public API surface.** `git diff origin/main...HEAD` adds exactly: `pub enum SaturationStop`
(`graph.rs`, re-exported from `pixelflow_search::egraph`), `pub stop: SaturationStop` on
`SaturationStats` and on `SaturationResult`, and `pub truncated: bool` on `ApplyResult` (the
per-scan fact the loop needs to tell a capped zero-union sweep from a quiesced one).
`SaturationStats` loses its `Default` derive (no default stop reason is honest; nothing used it).
Everything else in the diff is `#[cfg(test)]` (`runtime.rs`, `ir_bridge.rs`, `cell_grid.rs`'s
existing `mod tests`), a test-only integration file (`pixelflow-graphics/tests/`), or docs. The
production hunks are byte-identical to `origin/claude/saturation-telemetry-flag`'s
`graph.rs`/`mod.rs`/`saturate.rs`/`tests/saturation_stop.rs` (`git diff HEAD origin/claude/saturation-telemetry-flag -- <those>` is
empty apart from test renames main has since made), so that PR's rebase onto this is a
duplicate-delete.

**Gates** (post-rebase, `30dee06d`+docs, host load 9-13): `cargo fmt --all -- --check` clean;
`cargo clippy --workspace --all-targets -- -D warnings` clean; `cargo test --workspace` all
green (0 failed; the three pre-existing numerical failures Round 1 reported are gone on this
base); `cargo check -p pixelflow-ir --no-default-features` clean; `cargo check -p
pixelflow-search --no-default-features` clean — the pre-existing `no_std` failure
(`ExprNnue::from_bytes` unavailable without `std`) was fixed on `main` by #1053
(`436d3af8`, "gate the NNUE weights opt-in behind the std feature"), which this rebase picks
up, so the branch and `origin/main` agree.

**Reproduction of Round 1.** Every deterministic column reproduces to the digit on all 128 rows
that were clock-free in both rounds (see "Round 2: what changed"); the 6 label changes and 4
count mismatches are all on rows the 200 ms clock touched in one round or the other, on a host
whose load never met the < 4 gate.

**Against the synthetic registration figure.** The Guide registration reports the median
synthetic classical expression 48.5% worse at B=100 applications than at 4B; real core-term
kernels fire a median 8,446 applications before production stops them and lose a median 8.66%
(≥, on the 132 cap-bound rows) — so on real kernels the Guide's value is quality where the cap
binds with loss (the 9-15% on cap-bound glyphs, up to 47% on clock-cut ones) and compile latency
everywhere else (the cell grid and the quiesced rows already extract the lifted answer).

**Not changed.** Auto-merge is not armed on PR #1087; JP reads this first. The per-scan
`truncated: bool` cannot say *which* budget truncated a sweep when the clock has also expired by
the end of it — the loop then reports `Timeout`; in this run every `Timeout` row's production
trajectory genuinely differed from the unclocked reference (49/49), so no row was mislabelled
that way here, but a two-valued scan-stop would make the type exact.

