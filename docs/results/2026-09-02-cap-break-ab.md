# The cap break costs extraction quality on 140 of 204 real kernels

**Answer.** Ending the saturation run at the first class-cap-truncated sweep —
the control-flow change PR #1083 shipped alongside its `ScanStop`
classification — leaves a **more expensive extracted arena on 140 of the 204
real kernels (68.6%)**, unchanged on 48, and cheaper on 16; the median capped
kernel pays 1.003× and the worst pays 1.305×.

The 68.6% is not a coincidence: PR #1087 measured that the class cap binds on
68.4% of real kernels (132/193). The break decides the outcome for exactly the
kernels the cap touches, and for those kernels it decides it the wrong way.

## What was compared

Two arms of `EGraph::saturate_with_limits`
(`pixelflow-search/src/egraph/graph.rs`), differing in **one line of control
flow** and nothing else:

| Arm | `ScanStop::ClassCap` sweep |
|---|---|
| **A** — `main` as merged (82961fe3, PR #1083) | `stop = ClassCap; break;` |
| **B** — this measurement | `stop = ClassCap;` — the loop continues |

`ScanStop::Deadline` still breaks in both arms: a wall-clock ceiling is a hard
ceiling. The `ScanStop` classification introduced by #1083 is untouched — a run
that ends capped still reports `ClassCap`. Arm B additionally re-arms `stop` to
`IterationCeiling` after a full, productive sweep, so `stop` always names the
**last** sweep rather than the highest-water-mark of any earlier one; that is
the same "read the reason off the loop, never infer it" discipline #1083
established.

The separation being tested is *classification vs. termination*. #1083 was
right that a truncated sweep must not read as quiescence. It did not follow
that the run must end there: the sweep's unions are committed and the rebuild
has run, so the next sweep meets a **different** graph and can make progress on
different classes.

## Regime

The exact production call, replayed per kernel: `lower_dwrt_owned` →
`arena_to_egraph` → `config_for_node_count` tier →
`saturate_with_full_budget` → `env_extraction_policy()` (the static latency
prior) → `choices_to_arena`. The quality metric is the **latency-prior cost of
the extracted arena** — the per-op table summed over every reachable operation
once, i.e. the code the JIT would actually emit — not `ExtractedDAG::total_cost`,
which carries a cycle-breaking penalty that does not describe emitted code.

Corpus: the 204 real kernels of
`docs/results/2026-09-01-rule-order-real-kernels.md` — 12 `shader_bench`
ShaderToy ports, the psychedelic shader, the 623-node packed cell grid, and 190
ASCII glyph arenas at two densities. Tier split: 196 classical, 6 rapid, 2
blitz.

## Result — production regime (the budget that ships)

`cost(A) / cost(B)`, per kernel; >1 means the merged behavior is worse.

| | median | q1 | q3 | min | max |
|---|---|---|---|---|---|
| all 204 | **1.0034** | 1.0000 | 1.0206 | 0.7822 | 1.3046 |

| outcome | kernels |
|---|---|
| **A worse than B** | **140** |
| equal | 48 |
| A better than B | 16 |

By group:

| group | n | A worse | equal | A better | median | q3 | max |
|---|---|---|---|---|---|---|---|
| shader | 12 | 4 | 6 | 2 | 1.0000 | 1.0085 | 1.0968 |
| psychedelic | 1 | 0 | 1 | 0 | 1.0000 | 1.0000 | 1.0000 |
| cellgrid | 1 | 0 | 1 | 0 | 1.0000 | 1.0000 | 1.0000 |
| glyph16 | 95 | 67 | 20 | 8 | 1.0036 | 1.0204 | 1.2506 |
| glyph32 | 95 | 69 | 20 | 6 | 1.0038 | 1.0232 | 1.3046 |

By input size (reachable nodes) — the effect is concentrated where the cap
actually binds, and is largest in the mid-size band:

| nodes | n | median | A worse |
|---|---|---|---|
| ≤50 | 8 | 1.0000 | 2 |
| 51–200 | 35 | 1.0000 | 9 |
| 201–600 | 28 | **1.0654** | 20 |
| >600 | 133 | 1.0038 | 109 |

Summed over the whole corpus: 1,036,487 (A) vs 1,030,393 (B) latency-prior
cycles, A/B = 1.0059.

Stop reasons — the mechanism, stated plainly:

| stop | A | B |
|---|---|---|
| `ClassCap` | 183 | 2 |
| `Timeout` | 6 | 187 |
| `Quiesced` | 15 | 15 |

Arm B does not escape a budget; it **trades** one for another. Removing the
break lets the run spend the wall clock it was already allotted instead of
returning early, and the wall clock is what then ends it. Median iterations go
2 → 12 and median rule applications 5,374 → 15,202 inside the *same* 200 ms
classical ceiling.

The twelve kernels the merged behavior hurts most:

| kernel | group | nodes | A | B | A/B |
|---|---|---|---|---|---|
| glyph32:U+003C | glyph32 | 295 | 514 | 394 | 1.305 |
| glyph32:U+003E | glyph32 | 295 | 519 | 399 | 1.301 |
| glyph16:U+003E | glyph16 | 295 | 519 | 415 | 1.251 |
| glyph32:U+0027 | glyph32 | 127 | 171 | 137 | 1.248 |
| glyph16:U+004B | glyph16 | 617 | 1047 | 910 | 1.151 |
| glyph32:U+004B | glyph32 | 617 | 1047 | 910 | 1.151 |
| glyph32:U+002A | glyph32 | 559 | 1143 | 1003 | 1.140 |
| glyph16:U+0059 | glyph16 | 263 | 437 | 393 | 1.112 |
| glyph32:U+0059 | glyph32 | 263 | 437 | 393 | 1.112 |
| shader:smooth_min_scene | shader | 43 | 136 | 124 | 1.097 |
| glyph16:U+003C | glyph16 | 295 | 514 | 470 | 1.094 |
| glyph16:U+0076 | glyph16 | 1115 | 1760 | 1625 | 1.083 |

And the six where continuing hurts — arm B is not free, and `shader:metaballs`
is the one real regression in the corpus:

| kernel | group | nodes | A | B | A/B |
|---|---|---|---|---|---|
| shader:metaballs | shader | 62 | 158 | 202 | 0.782 |
| glyph16:U+0048 | glyph16 | 247 | 314 | 328 | 0.957 |
| shader:mandelbrot_distance | shader | 152 | 576 | 585 | 0.985 |
| glyph16:U+0068 | glyph16 | 2931 | 5045 | 5114 | 0.987 |
| glyph32:U+0068 | glyph32 | 2931 | 5045 | 5114 | 0.987 |
| glyph16:U+0044 | glyph16 | 2606 | 4506 | 4559 | 0.988 |

More search is not monotonically better under a *fixed* extraction cost model:
extra e-classes can move the DP's minimum onto a different, slightly worse
representative. That is a real cost of arm B and it is 16 kernels against 140.

## Result — clock-lifted regime (the same caps, 5 s deadline)

Production's 10/50/200 ms ceilings are load-dependent, so the production regime
alone cannot separate the control-flow change from the host. This regime keeps
the tier's own iteration and class caps and lifts only the wall clock to 5 s.
Arm A is unaffected by the lift — its clock-lifted cost equals its production
cost on **all 204** kernels, and its slowest saturation is 314 ms — so this
column is arm A's deterministic result compared against an arm B given room to
finish.

| | median | q1 | q3 | min | max |
|---|---|---|---|---|---|
| all 204 | 1.0038 | 1.0000 | 1.0300 | 0.7149 | 1.5700 |

A worse on **146**, equal on 42, better on 16. Corpus total 1,036,487 (A) vs
1,027,622 (B), A/B = 1.0086. Per group: glyph16 71 worse / 17 equal / 7 better,
glyph32 identical, shader 4 / 6 / 2, cellgrid and psychedelic unchanged.

The important column here is the stop reason:

| stop | A | B (5 s) |
|---|---|---|
| `ClassCap` | 189 | **142** |
| `Timeout` | 0 | 47 |
| `Quiesced` | 15 | 15 |

Given room, arm B still ends `ClassCap` on 142 of 204 runs. That is the whole
point: **removing the break does not hide the cap.** The run keeps sweeping,
keeps meeting the cap, keeps reporting it — and arrives at a cheaper program
than the one that stopped at the first capped sweep. #1083's classification is
doing exactly the job it was written for; only its `break` was load-bearing on
the result.

Arm B's median clock-lifted saturation is 1,463 ms and 47 kernels reach the 5 s
harness ceiling, which is why this is a supplementary regime and not the
shipping one.

## Repeatability

Each arm was run twice over the full corpus.

| arm | kernels whose cost changed between runs |
|---|---|
| A | **0 / 204** |
| B | 9 / 204 (median run-to-run ratio 1.0000) |

Arm A is bit-exact because it is cap-bound, not clock-bound. Arm B is
clock-bound on 187/204 kernels and therefore load-sensitive — the nine moving
rows are that sensitivity, and none of them flip the verdict. Note the
direction: a **more** loaded host does **less** work inside 200 ms, so a loaded
measurement *understates* arm B. The numbers above are a lower bound on B's
advantage.

## Wall clock — context, never a metric

Saturation wall clock over the 204 kernels: 7,528 ms (A) vs 37,567 ms (B). Arm
B spends budget arm A was leaving on the table; per kernel it stays inside the
tier's own ceiling (10/50/200 ms), which is why the stop reason moves to
`Timeout` rather than the compile taking longer than the budget permits.

### The cost lands on compile time, and it is large

Removing the break makes saturation spend the budget `config_for_node_count`
already allots it, so every capped kernel now costs its tier's full ceiling
(10/50/200 ms) instead of returning after two or three sweeps. Measured on the
debug-profile test binaries, each arm run on its own:

| binary | arm A | arm B |
|---|---|---|
| `pixelflow-compiler` lib (84 tests) | **3.00 s** | **120.02 s** |
| `pixelflow-graphics` lib (162 tests) | 18.35 s | 25.06 s |

The compiler tier is the AOT `kernel!` path, so this is *build* time for
anything that expands the macro. The 40× is not contention: arm B reproduced
120.0 s both inside the full `cargo test --workspace` run and on its own, while
arm A finished the same 84 tests in 3.0 s.

Framed correctly, this is not a regression against the designed budget — the
budget policy always said a classical kernel may spend 200 ms, and arm A was
simply leaving most of it unspent. But it is a real change in wall-clock build
cost and it is the reason this measurement is being handed up rather than
merged on the quality number alone.

**This run was loaded** — host load averages 9.2–21.3 throughout (12 cores,
shared machine, other agents active). Load is recorded in the run logs and in
`prod_elapsed_ms`; it is why arm A was re-run to confirm bit-exactness and why
arm B was run twice.

## Reproduction

```
PIXELFLOW_TELEMETRY_DIR=<arena dumps>  \
PIXELFLOW_CAP_ARM=A                    \
PIXELFLOW_CAP_GENEROUS_MS=1            \
PIXELFLOW_CAP_OUT=/tmp/cap_A.csv       \
  cargo test -p pixelflow-search --release -- --ignored cap_break_ab --nocapture --test-threads=1
```

then apply the arm-B patch to `graph.rs` and repeat with `PIXELFLOW_CAP_ARM=B`.
The harness is `pixelflow-search/src/runtime/cap_break_ab.rs`; its `load_arena`
/ `arena_cost` / `run` helpers are PR #1101's, reused rather than rebuilt. The
arena dumps come from #1101's three dumpers (`cell_grid.rs`,
`production_glyph_arena_dump.rs`, `shader_and_psychedelic_arena_dump.rs`).
When #1101 lands, this module's three helpers should collapse into its
`production_telemetry` module — one definition, imported, not restated.

Per-kernel rows: `2026-09-02-cap-break-ab.csv`.
