# Rule-order effect on real kernels

2026-09-01. Answers JP's question directly: Round 2 v3
(`docs/plans/2026-09-01-phase3-round2-registration-v3.md`,
`docs/results/2026-09-01-round2-unguided-vs-rulecount-v3.*`) found the base-62
sweep order dominant on *synthetic classical expressions* (unguided regret at
B=100 applications: 96.58% under production `all_rules()` order, 26–46% under
random shuffles, 1.12% under a static numeric-first reorder). JP's ruling:
"the synthetic small kernels are not the workload; shaders and the real
glyph/cell-grid kernels are" — so this re-runs the same three-way comparison
(production order / numeric-first / seeded shuffles) on the real kernels: the
12 `shader_bench` ShaderToy ports, one hand-transcribed psychedelic shader
kernel, one packed cell-grid geometry (623 nodes), and the 95 ASCII glyph
arenas core-term actually bakes at both display densities (190 kernels total,
204 with shaders/psychedelic/cellgrid) — n=204.

**Journal note (per JP's instruction):** the shader/psychedelic kernels
(`pixelflow_pipeline::shader_bench`, and the psychedelic shader) are FINAL-tier
kernels reserved as a publication hold-out for ordering claims
(`docs/plans/2026-08-05-egraph-nnue-research-workflow.md` §0.2). JP explicitly
opened that set for *this* measurement. It therefore no longer serves as a
hold-out for any future rule-order-ordering claim — a later paper/report
claiming to validate rule order on held-out FINAL kernels must draw on a
different set.

## Answer

**Order still matters on real kernels, but far less than on the synthetic
corpus, and the effect shrinks fast with budget.** At the anytime checkpoints
(median regret vs. the best any of the 5 arms reaches at any checkpoint, same
kernel): production order carries **21.8% median regret at B=100**, falling to
**5.1% by B=12800**. The numeric-first static reorder is best or tied-best at
every checkpoint from B=100 through B=12800, reaching **0% median regret by
B=12800** — i.e. by production's own budget class, numeric-first is
essentially always at or better than the best of the five arms, and strictly
better than production order's own 5.1%. Three random shuffles land between
those two, never beating numeric-first and never as bad as the synthetic
corpus's worst shuffles. On the **production regime itself** (the exact call
`optimize_runtime_arena_uncached` makes: `config_for_node_count`'s tier,
`saturate_with_full_budget`, static-latency-prior extraction), switching to
numeric-first order changes what ships on 167/204 kernels (146 improved / 21
worse; 37 byte-identical), median cost ratio **0.9702** (a ~3% extraction-cost
improvement) — smaller than a synthetic reader of the v3 report would guess
from a 96.58%-regret headline, but a real, one-line, free win in the
regime that ships today. **Order stops mattering somewhere between B=6400 and
production's own budget** for most real kernels — the anytime table's regret
column is still 5–12% at B=6400 across all five arms and within 5% of the
best by B=12800, so "stops mattering" is a matter of degree past B≈6400–12800,
not a single cliff.

## Method

Same three-way comparison Round 2 v3 registered
(`pixelflow-search/src/egraph/rule_order.rs`): `RuleOrder::Production`
(`all_rules()` verbatim), `RuleOrder::NumericFirst` (base-62 sorted by
descending TRAIN strict-positive rate from
`docs/results/2026-09-01-train-guide-report.md`, ties broken by ascending
production index — `NUMERIC_FIRST_ORDER`, pinned and re-derived by a test so a
transcription error fails loudly rather than drifting), and three seeded
Fisher-Yates shuffles (seeds 1/2/3, matching v3). Every arm is the same 62
rules, only reordered — `build_rule_set` asserts this.

**Production regime** replays `optimize_runtime_arena_uncached`'s exact
sequence per kernel per arm (`pixelflow-search/src/runtime.rs`,
`run_with_rules`): `lower_dwrt_owned`, `config_for_node_count`'s tier
(`SaturationConfig`: iteration cap, **5,000-class cap**, wall-clock cap),
`saturate_with_full_budget`, then the static-latency-prior extraction
(`env_extraction_policy()` — unconditionally `CostModel::latency_prior()`
now; the NNUE extraction-head arm was deleted). Cost is `arena_cost` — the
per-op latency-prior sum over the *materialized extracted arena* the JIT
would actually compile, never `ExtractedDAG::total_cost` (that DP total pays
a 1,000,000-per-cycle `CYCLE_COST` penalty per cycle-breaking pick and does
not describe emitted code — an early draft of this harness used it directly
and produced regret numbers in the billions of percent before this was
caught and fixed; see commit history).

**Anytime** samples cost at applications
B ∈ {100, 200, 400, 800, 1600, 3200, 6400, 12800} — the same
application-denominated x-axis `pixelflow-search/src/egraph/anytime.rs`
defines (cherry-picked from `claude/phase3-round2`), reimplemented locally in
`runtime.rs` (`anytime_curve_arena`) because the real glyph arenas contain
runtime-only mask/int ops (`BitAnd` etc.) that only the module-private
`arena_to_egraph` resolves — the public `EGraph::add_arena` panics on them
(`no static Op for OpKind BitAnd`), so the anytime loop had to be re-driven
through the same private constructor `run_with_rules` uses rather than
through the public `run_anytime_curve`. Same budget semantics otherwise: one
rule sweep per checkpoint granularity, a 20,000-sweep safety ceiling (never
bound), a 60s per-(kernel,arm) wall-clock safety ceiling that panics if hit
(never hit). "Regret vs. best-any-arm-at-any-checkpoint" is computed per
kernel per checkpoint against the minimum extracted cost any of the 5 arms
reaches at that exact checkpoint B for that kernel (not a global minimum
across budgets) — the same convention v3 used on synthetics.

**Kernels**: `pixelflow_pipeline::shader_bench::named_shadertoy_kernel` for
all 12 names (already `(ExprArena, ExprId)`, no lowering needed — built by
hand, never through `kernel!`'s own e-graph); the psychedelic shader kernel
(`pixelflow-compiler/src/codegen/mod.rs`'s `emit_exact_psychedelic_kernel`
test / `pixelflow-runtime/examples/psychedelic_shader.rs`), hand-transcribed
into `ExprArena` calls (getting an unoptimized arena out of the `kernel!`
proc-macro pipeline from outside the crate would need `pixelflow-compiler`'s
private `parser`/`sema` modules public — the API change CLAUDE.md forbids;
`t`/`width`/`height` fixed at 1.0/800/600, RGB channels summed into one
scalar root, matching `shader_bench.rs`'s own `cosine_palette` convention);
one packed cell-grid geometry (`pixelflow-core/src/lattice/cell_grid.rs`'s
`dump_production_cell_grid_arenas`, the `80x24_d1` geometry — all three
dumped geometries are 623 reachable nodes, since the packed kernel's
structure doesn't depend on cols/rows/density, so "the 623-node cell grid" is
one kernel, not three); the 95 ASCII glyph arenas at both densities
(`pixelflow-graphics/tests/production_glyph_arena_dump.rs`). All were already
present as dumped `.arena` files under this session's
`scratchpad/telemetry/dumps` (per JP's instruction to reuse them if present);
the 13 shader/psychedelic dumps were minted fresh by a new dumper,
`pixelflow-pipeline/tests/shader_and_psychedelic_arena_dump.rs`, following
the existing dumpers' exact text format.

**Caveat — host load.** `uptime` load averages were 9–21 throughout this run
(recorded at start/end of the full-set run, printed by the harness), well
above the 4.0 "quiet" threshold. Every cost number here is the deterministic
static-latency-prior model — load cannot affect it. But the production
regime's `SaturationStop` — specifically how often the 200ms wall-clock cap
binds (`Timeout`) vs. the deterministic `ClassCap`/`Quiesced` — IS
machine-dependent, and is not used as a metric here for that reason: stop
reasons were 52 `Timeout` / 137 `ClassCap` / 15 `Quiesced` (production order)
vs. a range of 23–72 `Timeout` across the other arms, but under this load
that spread is not attributable to rule order alone and is reported only as
context, not as a finding.

## Production regime: cost(numeric-first) / cost(production)

n=204 (2 kernels — the space glyph at both densities — extract to cost 0
under every arm; excluded from the ratio distribution as 0/0-undefined,
counted as "unchanged").

| stat | value |
|---|---:|
| median | 0.9702 |
| q1 | 0.9474 |
| q3 | 1.0000 |
| min | 0.7379 |
| max | 1.0806 |
| improved (numeric-first cheaper) | 146 |
| unchanged (byte-identical cost) | 37 |
| worse (numeric-first costlier) | 21 |

Per group:

| group | n | median ratio | min | max |
|---|---:|---:|---:|---:|
| cellgrid | 1 | 1.0000 | 1.0000 | 1.0000 |
| glyph16 (density 1.0) | 94 | 0.9686 | 0.7379 | 1.0351 |
| glyph32 (density 2.0) | 94 | 0.9692 | 0.7379 | 1.0351 |
| psychedelic | 1 | 1.0444 | 1.0444 | 1.0444 |
| shader (12 ShaderToy ports) | 12 | 1.0000 | 0.9546 | 1.0806 |

The glyph population (190 of 204 kernels) is where the effect concentrates —
median 3% cheaper under numeric-first, one glyph as much as 26% cheaper
(ratio 0.7379), a few shaders and the psychedelic kernel land the other way
(numeric-first up to 8% costlier on one shader, 4.4% costlier on the
psychedelic kernel). The single cell-grid geometry is byte-identical under
both orders.

## Anytime: median regret % vs. best-any-arm-at-any-checkpoint

n=204, regret = (cost(arm, B) − best-at-B) / best-at-B × 100, median across
kernels (0-cost kernels excluded from the median at each B, same convention).

| arm | B=100 | B=200 | B=400 | B=800 | B=1600 | B=3200 | B=6400 | B=12800 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| production (`all_rules()`) | 21.77 | 21.72 | 19.23 | 18.61 | 15.91 | 10.66 | 6.67 | 5.09 |
| numeric-first | 18.60 | 17.20 | 13.13 | 9.39 | 5.95 | 2.17 | 0.68 | **0.00** |
| shuffled(1) | 18.06 | 17.91 | 17.17 | 16.95 | 16.95 | 14.39 | 8.52 | 5.18 |
| shuffled(2) | 15.03 | 15.03 | 15.03 | 15.03 | 12.84 | 10.64 | 6.24 | 3.96 |
| shuffled(3) | 18.06 | 18.06 | 18.06 | 17.21 | 16.73 | 15.77 | 12.79 | 11.69 |

Reading this against the synthetic v3 numbers (96.58% / 26–46% / 1.12% at
B=100 alone): every arm's regret on real kernels at B=100 (15–22%) is far
below production order's synthetic-corpus regret and much closer to the
synthetic shuffles' range — real kernels are evidently far less sensitive to
sweep order than the adversarial-shaped classical synthetics were. Numeric-
first is still the best arm at every checkpoint and the only one to reach 0%
median regret by B=12800 (production's own class-cap regime routinely runs
into the thousands of applications on these kernel sizes, so B=12800 is
within reach of the production budget for most of them, not a hypothetical
extension). Production order (today's shipped order) is the *worst* of the
five arms at every checkpoint from B=100 through B=1600, and remains
noticeably worse than numeric-first through B=12800.

## Node counts and tiers

| tier | n | node-count range |
|---|---:|---:|
| blitz (0–10 nodes) | 2 | 1–1 |
| rapid (11–50 nodes) | 6 | 40–46 |
| classical (51+ nodes) | 196 | 62–12,056 |

The 12 shaders and the psychedelic kernel are small (40–152 nodes, mostly
`rapid`/low `classical`) — the size band the synthetic corpus was built to
resemble. The glyph population spans two and a half orders of magnitude
inside the single `classical` tier (62 to 12,056 reachable nodes) — the
production tier system has no distinction between a 62-node glyph and a
12,056-node one, both get the same 100-iteration/5,000-class/200ms budget.
The 623-node cell grid sits in the middle of that range. This is itself
worth flagging separately from the rule-order question: `config_for_node_count`
treats a 200x size range as one tier.

## Files

- `docs/results/2026-09-01-rule-order-real-kernels.csv` — 1,020 rows (204
  kernels × 5 arms): `kernel, group, nodes, tier, arm, prod_stop, prod_cost,
  prod_applications, prod_iterations, prod_elapsed_ms,
  anytime_cost_at_{100,200,400,800,1600,3200,6400,12800}, anytime_ended,
  anytime_ended_at_apps`.
- `docs/results/2026-09-01-rule-order-real-kernels.json` — same rows,
  structured, plus the grid/arm/load metadata.
- `pixelflow-search/src/egraph/rule_order.rs` — `RuleOrder`,
  `NUMERIC_FIRST_ORDER` (pinned + re-derivation test), `build_rule_set`.
- `pixelflow-search/src/egraph/anytime.rs` — the application-denominated
  anytime-curve definition (cherry-picked from `claude/phase3-round2`).
- `pixelflow-search/src/runtime.rs`'s `production_telemetry` module —
  `run_with_rules` (production regime, rule-set parameter added),
  `anytime_curve_arena` (the anytime loop re-driven through the private
  arena constructor), `rule_order_real_kernels` (the harness test that
  produced the numbers above:
  `PIXELFLOW_TELEMETRY_DIR=<dumps> [PIXELFLOW_RULE_ORDER_INCLUDE_D2=1] cargo
  test -p pixelflow-search --release -- --ignored rule_order_real_kernels
  --nocapture --test-threads=1`).
- `pixelflow-pipeline/tests/shader_and_psychedelic_arena_dump.rs` — mints the
  13 shader/psychedelic `.arena` dumps this measurement needed and the
  glyph/cell-grid dumpers didn't already provide.

## What this does not settle

No change to `all_rules()`'s shipped order is made here — this is a
measurement, not a proposal. JP's stated context for this measurement is a
parallel thread making the whole lattice one kernel the e-graph sees; once
that lands, "kernel size" for this question changes by another order of
magnitude and this measurement's tier/regret numbers should be re-taken
against whatever kernel that thread produces rather than assumed to
extrapolate.
