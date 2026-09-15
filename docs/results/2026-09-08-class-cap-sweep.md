# The class-cap sweep: the classical cap is sized by the input, past that budget makes extraction worse, and the raise is blocked by the `'8'` tangency

**Not shipped — blocked, and pinned.** The rule this sweep calibrates is in the tree: the classical class cap is `clamp(8 × inserted_classes, CLASSICAL_CLASS_FLOOR, CLASSICAL_CLASS_CEILING)` (`SaturationConfig::classical_for`, `pixelflow-search/src/egraph/saturate.rs`), the application budget 40 per class of the resolved cap and the safety ceiling 1.5 ms per application (300 s at the floor). **The ceiling is pinned at the floor (5,000), which makes the rule inert and production byte-identical to before.** Raising it to the calibrated 50,000 raises 44 of the 95 DejaVu glyphs (to 5,448–31,952 classes) and they extract **−16.7% Σ `dag_cost`**, −31% guarded entries, 1,074 → 20 spill slots, −0.3% bytes, none dearer, for a 95-glyph warm of **25.5 s where it was 6.6 s (+290%)** — and it puts a half-covered smear along the `'8'` waist at 13, 15, 17, 19 and 21 px where FreeType has no ink (15 texels), because the raised `'8'` extracts a different fusion of its quadratic solver and `disc >= 0` at the waist's tangency lands on the other side of exact zero. That is the `'8'` defect the FreeType oracle found in 2026-09 (`freetype_oracle.rs`: "a new fusion choice could put one there again — which is exactly what this assertion is now for"), back by the route its own comment predicted; `kernel_glyph_optimize` (raw vs optimized, 37 divergent texels, all on `'8'`) sees it too. The knife edge is the glyph kernel's (`quad_tangency_winding.rs` measures a grazing residual of 0.688 where zero is correct), not the optimizer's — any change of fusion can land on it, and this raise did. **The prerequisite is the tangency fix in the quadratic winding kernel; after it, `CLASSICAL_CLASS_CEILING` becomes `CLASSICAL_CLASS_CEILING_CALIBRATED` (50,000) and nothing else changes.** The gap that let the raw-arm oracle miss this is closed: `freetype_oracle.rs` now runs the same check on the optimized arena at the bake's lattice (`our_optimized_ink_is_never_more_than_a_texel_from_freetype_s`), pinned at zero orphans, so no future extraction change can put ink on `'8'`'s waist without a red job. `docs/plans/2026-09-01-production-budget-determinism.md` carries the revision and the block.

**The gate (part 1):** raising the cap past 10,000 no longer changes the extraction objective silently. `shared_dag_dp_pass` held one dense bitset per live class (live²/8 bytes) and #1224 gated it at `SHARED_DAG_PASS_CLASS_LIMIT = 10_000` on the *whole* e-graph's class count, above which the tree objective was returned under the same type. The pass now holds each class's reach set in whichever of two forms is smaller (members as `u32`s, or the bitset), unions by stamping so a candidate costs its children's set sizes rather than the graph's, and runs under `SHARED_DAG_PASS_BYTE_BUDGET` (256 MiB, never approached: the largest reach total on any arm is 3.5 MB at 15,172 live classes on the cell grid at 100,000, where the dense form would have been 29 MB). A pass that crosses the budget is abandoned and `ExtractedDAG::report` / `Optimized::extraction` / the `saturation-telemetry` record say `tree_only`; every row below carries its objective. Choices are unchanged: all 210 shipped kernels compile to identical bytes, `dag_cost`, guards and pictures against the origin/main extractor (`rows/base` vs `rows/mine`, this sweep's first check), and `shared_pass_matches_the_dense_reference_on_a_saturated_graph` holds the pass to the dense pass it replaced.

## Method

Every row is one kernel compiled through the production path's three calls (`optimize_runtime_arena` / the same pipeline with the arm's optimizer in the saturation slot, `relink`, `emit::compile`) by `egraph_off_on run --class-cap N --app-cap M` (flat arms) or `--classes-per-inserted R --cap-floor 5000 --cap-ceiling 50000` (proportional arms), `--no-clock --no-probe`. The saturation columns are the optimizer's own `saturation-telemetry` record for that compile, read back from stderr: stop reason, rounds, applications, classes at stop, live (root-reachable) classes, extraction objective, reach-set bytes. `compile ms` is `optimize_ms + emit_ms` under the harness's counting allocator (`alloc_probe`; +10–20% on the compile times of the unmodified binary) on a shared aarch64 host whose 1-minute load ran 5–18 during the flat arms and up to 99 during the proportional arms and the shipped rows (other sessions' builds and this one's gates) — **the seconds are a sign, the ratios between arms are the claim, and no clock was taken at load < 8**. `peak MB` is the peak net heap growth of the compile. The corpus is the benchmark correction's DEV set — DejaVu Sans Mono, 95 printable ASCII at tile 16 and tile 32; cell grid 80×24 @2×; psychedelic; the twelve `shader_bench` ports — plus chrome (held out; run once per arm, reported separately below). The same-form scalar oracle (`eval_scalar` of the emitted arena against the JIT, 256 points) reports **0 NaN mismatches on every row of every arm** (max |Δ| ≤ 6.1e-6); production-path rows are asserted byte-identical to `Manifold::compile` (205/205).

The two glyph tiles are pairwise identical in every deterministic column at every arm (95/95 at 5,000, 10,000, 20,000 and under the shipped rule): the tile scales a constant. Six of the twelve shaders are rapid-tier (≤ 50 nodes) and so have a production cap of 2,000 that every flat arm, the 5,000 baseline included, is already above; their production rows are in `rows/mine`. The space glyph is blitz-tier. The arms keep each kernel's own tier's round cap.

## Where the cap stops clipping the input

| arm | glyphs on `ClassCap` in round 1 (of 95) | stop reasons | rounds med / max |
|---|---:|---|---:|
| flat 5,000 | **44** | class_cap 89, quiesced 6 | 2 / 7 |
| flat 10,000 | 33 | class_cap 87, quiesced 8 | 2 / 28 |
| flat 20,000 | 4 (`%`, `&`, `8`, `@`) | class_cap 85, quiesced 10 | 3 / 51 |
| flat 50,000 | 0 | class_cap 85, quiesced 10 | 2 / 91 |
| 6 per inserted class | **44** (the same 44) | | |
| **7 per inserted class** | **0** | | |
| 8 (shipped), 10, 12 per inserted class | 0 | class_cap 89, quiesced 6 | 2 / 7 |

The first round of the production rule set grows every glyph to between 6× and 7× its inserted (hash-consed) class count — a cliff, not a slope: at 6 the same 44 glyphs clip, none of them marginally, at 7 none do. The flat caps clip in order of input size (need = 10,000 for 827–1,444 inserted classes, 20,000 for 1,707–2,844, 50,000 for 3,405–3,994), which is why a flat number cannot be right for both `U+0025` (3,994 inserted) and `K` (≈150 inserted, which at 50,000 runs seven rounds to 37,540 classes in 77 s for a 649 → 492 `dag_cost` that the shipped rule gets in 93 ms at the floor). The node count is the wrong key for the same rule (`capx` arms): chrome's `Kernel` is a 390,815-node tree that inserts to 465 classes, so 4 per node hands it the 100,000 ceiling and +42% `dag_cost`, while 8 per inserted class keeps it at the floor. Even 12 per inserted class moves chrome to 5,580 and +1.4% bytes.

The application budget never bound on any arm at the plan's 40-per-class ratio, and bound only once with the cap held at 200,000 (one glyph at 20,000, one shader at 50,000): applications per class at stop stay under 2 at every cap (chrome 142,134 at 85,977 classes; cell grid 165,997 at 90,816). The two application arms are otherwise identical, so the budget dimensions separate cleanly: the class cap is the only one that binds. The safety ceiling never fired on any completed arm, including the flat 100,000 on chrome (3.6 s), the cell grid (0.5 s) and the shaders (139 s for twelve). The glyph families at 100,000 were still in flight when this was written — 26 of 95 glyph16 rows after 16 minutes, the first 26 stopping on the class cap in rounds 2–7 at 49,902–93,819 classes and taking up to 93 s each; they are left out of the tables — and the flat 50,000 already costs 1,662 s per 95 glyphs, with the small straight glyphs (`Y` 86 s, `K` 77 s, `V` 71 s) saturating seven rounds to 35,000 classes. With the application cap held at 200,000 the 50,000 arm stops 4 of 95 glyphs on the application budget instead of the class cap and extracts the identical term for every glyph (Σ bytes and `dag_cost` equal to the digit).

## Does quality improve with budget? Per family

`Δ` is against the flat 5,000; negative is better. Deterministic columns.

| family | arm | Σ bytes | Σ dag_cost | Σ guarded / schedule | spills | kernels dearer in dag_cost | Σ compile | peak heap |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| glyph16 (95) | flat 10,000 | +1.0% | −4.8% | 51,909 / 23,823 | 76 | 20 | 31 s | 6.0 MB |
| | flat 20,000 | −1.0% | −15.4% | 50,466 / 23,498 | 42 | 4 | 180 s | 10.6 MB |
| | flat 50,000 | −7.3% | −10.8% | 38,260 / 22,145 | 86 | **14** | 1,662 s | 23.1 MB |
| | **8 per inserted (shipped)** | −0.3% | **−16.7%** | 51,395 / 23,633 | 20 | **0** | 25.5 s | 16.1 MB |
| | (baseline 5,000) | 852,052 | 392,757 | 74,213 / 35,334 | 1,074 | | 6.6 s | 3.7 MB |
| shader (12) | flat 10,000 | −1.0% | +1.6% | 647 / 1,408 | 0 | 3 | 1.6 s | 4.6 MB |
| | flat 20,000 | +1.3% | +0.6% | 664 / 1,400 | 0 | 5 | 6.7 s | 9.3 MB |
| | flat 50,000 | +4.1% | +7.8% | | 0 | 7 | 22 s | 22 MB |
| | flat 100,000 | +4.5% | +9.0% | | 0 | 7 | 139 s | 44 MB |
| | shipped | 0 | 0 | | | 0 | 0.25 s | 2.3 MB |
| cellgrid (1) | 10,000 → 100,000 | 2576 → 2560 | 428 → 427 | 0 / 217 | 7 → 5 | 0 | 19 ms → 470 ms | 2.3 → 43 MB |
| psychedelic (1) | 10,000 → 100,000 | 2368 → 2352 | 825 → 829 | 29 / 127 | 0 | (+4 at ≥ 20,000) | 20 ms → 512 ms | 2.3 → 43 MB |

**Glyphs.** Budget helps exactly as far as the input's own first round, and then hurts. The 44 raised glyphs go from Σ 367,346 to 301,803 `dag_cost` (−17.8%) under the shipped rule and to 326,262 (−11.2%) at the flat 50,000: 14 mid-size glyphs (`(`, `)`, `D`, `P`, `J`, `f`, `h`, `j`, `l`, `n`, `r`, `t`, `u`, `y`; 538–990 inserted classes) are 15–20% dearer at 50,000, where they run a third round to 45,000 classes, than at their proportional 4,300–7,900 cap where they stop in round 2. Against that, the 51 floor glyphs at the flat 50,000 do find shorter code (Σ bytes −7.3% overall is theirs; guarded entries 74,213 → 38,260) at 1,583 s for 51 glyphs — 30 s each for a hundred `dag_cost` units, which the shipped rule declines. Bytes and `dag_cost` disagree at 50,000 and agree under the shipped rule; `dag_cost` is the extractor's objective and bytes are the emitter's, and the row prologue guard count (the glyph family's first metric) moves with `dag_cost` here.

**Shaders, cell grid, psychedelic.** Never clipped by the floor (≤ 1,308 live classes at 5,000), so every raise is budget past the input's frontier, and it costs: the shaders lose 8–9% `dag_cost` and 4–5% bytes at 50,000–100,000, seven of twelve dearer, for 90–550× the compile time; the cell grid and psychedelic move by a byte and a unit. The shipped rule leaves all of them at the floor, unchanged.

**Chrome (held out, one run per arm).** The sweep reproduces the production-scale plan's finding and puts a number on it at every cap: `chrome_packed` 3,424 B / 1,668 `dag_cost` at 5,000 → +5.1% / +0.3% at 10,000 → +4.7% / +0.8% at 20,000 → +16.8% / +32.7% at 50,000 → **+39.3% / +42.3% at 100,000**, compile 177 ms → 3.6 s, peak 35 → 64 MB; at 50,000 and 100,000 the sharing-aware pass runs to completion and at 100,000 the tree objective wins the two-objective race (`tree_cheaper`) — the extractor's objective itself cannot rank a space that large. Chrome inserts to 465 classes and keeps the floor under the shipped rule, byte-identical; the row is reported, not decided on.

**The reading.** Extraction quality is not monotone in budget. It improves while the cap clips the input's first rewrite round (the glyph family, where the flat cap was smaller than the input) and worsens once the graph holds more variants than the additive prior can rank (chrome, the shaders, and the glyphs themselves past ~8× their inserted size). The budget was the limit on the glyphs, and only up to the input's frontier; past it, the limit is extraction — the net's job, and not one more budget can buy.

## What the raise would ship, and what it costs — measured, then pinned off

`SaturationConfig::classical_for(inserted)`; `Budget::Production` resolves through `config_for_input(InputSize { nodes, classes })`; `Optimizer::run` reads the inserted count off the e-graph on entry (the input has been inserted and nothing rewritten). These rows were taken with the ceiling at 50,000 (`rows/ship`, production path, asserted byte-identical to `Manifold::compile`, 205/205), against the same binary's production rows before the change; with the ceiling pinned at the floor, production is the "before" column of every row:

| family | n | raised | Σ bytes | Σ dag_cost | Σ guarded | spills | dearer | Σ compile (sign, load 40–99) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| glyph16 | 95 | 44 | 852,052 → 849,652 (−0.3%) | 392,757 → 327,214 (−16.7%) | 74,213 → 51,395 | 1,074 → 20 | 0 (18 larger in bytes) | 7.9 s → 30.2 s (3.8×) |
| glyph32 | 95 | 44 | 852,100 → 849,764 (−0.3%) | 392,762 → 327,274 (−16.7%) | 74,216 → 51,395 | 1,075 → 20 | 0 | 6.7 s → 28.5 s (4.3×) |
| bench (32 pt A/O/S, + O wide) | 4 | 3 | +0.5% / −4.7% | −18.0% / −20.1% | | | 0 | 12–18× |
| cellgrid, psychedelic, shader ×12, chrome ×2 | 17 | 0 | identical | identical | identical | identical | 0 | 1.07–1.23× (load) |

The +290% on the glyph warm is the cost the directive buys; it is 44 glyphs at 0.1–2.5 s each, and it is entirely saturation (`sat_wall_ms` ≈ `optimize_ms`; emission is unchanged at ~8 ms per glyph and the extraction pass's reach sets are ≤ 1.4 MB). **The clock sign**, taken once the host quietened: alternated flat-5,000 / 8-per-inserted-class glyph16 warms, one process per arm, two rounds, 1-minute load 8.0–8.4 (at the protocol's boundary, two sweep processes still running): round 1 **6,547 ms → 26,246 ms (×4.01)**, round 2 **6,628 ms → 25,613 ms (×3.86)**, saturation 5.5 s → 25 s of it; the deterministic columns of all four runs are identical to the rows above.

**What blocked it.** At the production tiles (16 and 32 px) the raised `'8'` is clean — cross-form |Δ| against the raw arena ≤ 6.4e-6 on every glyph, pictures differ only in the last bit (87 of 190 hashes, FMA rounding). At 13, 15, 17, 19 and 21 px it is not: `kernel_glyph_optimize` reports 37 texels on `'8'` where the optimized arena has ~0.5 coverage and the raw arena 0, and the FreeType oracle's new optimized arm finds 15 of them with no ink within a texel in FreeType's rendering — the waist smear. Both tests are green with the ceiling pinned at the floor. The fix belongs to the kernel: `disc >= 0` at a shared extremum is exact zero, one rounding of `Y·slope + c` (fused) against two (raw) decides it, and the sweep changes which fusion the extractor prefers for `'8'` because it changes what the e-graph holds. Five earlier attempts at that fix are recorded in `freetype_oracle.rs`; it is the correctness stream's, and this sweep's raise waits on it.

## Generated tables

Everything below is `egraph_off_on cap-sweep` over the row files (per-kernel rows in the `.csv`, cells and movement in the `.json`).

Arms: flat class caps 5,000 / 10,000 / 20,000 / 50,000 / 100,000 with the application cap at the plan's 40 per class of cap and, separately, held at 200,000; per-inserted-class caps 6 / 7 / 8 / 10 / 12 (`caph<R>-5000-50000`), the calibrated rule being `caph8` (pinned off in the tree — see the prose); and per-node caps 4 / 6 / 8 (`capx<R>-5000-100000`), the arm that showed the node count is the wrong key. The flat 100,000 arms on the glyph families were still in flight when this was generated (26 of 95 glyph16 rows and 14 of 95 glyph32 rows after 16 minutes; the small straight glyphs take minutes each there) and are left out of the tables.

## Per family, per arm

| class | arm | cap (min–max) | apps (min–max) | n | stop | cap in round 1 | rounds med / max | apps med / max | classes med / max | live max | objective | Σ nodes | Σ bytes (vs base) | Σ dag_cost (vs base) | Σ guarded/schedule | Σ spills | oracle NaN / max abs | Σ compile ms | peak MB |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cellgrid | cap5000-app200000 | 5000 | 200000 | 1 | class_cap 1/1 | 0 | 3 / 3 | 6826 / 6826 | 4634 / 4634 | 1322 | shared 1/1 | 117 | 2576 (+0.0%) | 428 (+0.0%) | 0/217 | 7 | 0 / 0.000e0 | 19 | 2.3 |
| chrome | cap5000-app200000 | 5000 | 200000 | 1 | class_cap 1/1 | 0 | 2 / 2 | 5502 / 5502 | 4944 / 4944 | 1351 | shared 1/1 | 398 | 3424 (+0.0%) | 1668 (+0.0%) | 614/385 | 22 | 0 / 0.000e0 | 184 | 34.9 |
| chrome_channel | cap5000-app200000 | 5000 | 200000 | 1 | class_cap 1/1 | 0 | 2 / 2 | 5626 / 5626 | 4903 / 4903 | 1208 | shared 1/1 | 300 | 2640 (+0.0%) | 1418 (+0.0%) | 433/288 | 23 | 0 / 1.779e-5 | 61 | 11.9 |
| glyph16 | cap5000-app200000 | 5000 | 200000 | 95 | class_cap 89/95, quiesced 6/95 | 44 | 2 / 7 | 6691 / 18244 | 4718 / 5000 | 4175 | shared 44/95, tree_cheaper 51/95 | 89222 | 852052 (+0.0%) | 392757 (+0.0%) | 74213/35334 | 1074 | 0 / 3.427e-6 | 6565 | 3.7 |
| glyph32 | cap5000-app200000 | 5000 | 200000 | 95 | class_cap 89/95, quiesced 6/95 | 44 | 2 / 7 | 6691 / 18246 | 4718 / 5000 | 4175 | shared 44/95, tree_cheaper 51/95 | 89225 | 852100 (+0.0%) | 392762 (+0.0%) | 74216/35336 | 1075 | 0 / 5.066e-6 | 6591 | 3.7 |
| psychedelic | cap5000-app200000 | 5000 | 200000 | 1 | class_cap 1/1 | 0 | 3 / 3 | 7328 / 7328 | 4670 / 4670 | 1263 | tree_cheaper 1/1 | 112 | 2368 (+0.0%) | 825 (+0.0%) | 29/127 | 0 | 0 / 0.000e0 | 21 | 2.3 |
| shader | cap5000-app200000 | 5000 | 200000 | 12 | class_cap 9/12, quiesced 3/12 | 0 | 4.5 / 9 | 7408 / 30052 | 3094.5 / 4611 | 1308 | shared 3/12, tree_cheaper 9/12 | 636 | 15072 (+0.0%) | 4618 (+0.0%) | 652/1392 | 0 | 0 / 6.109e-6 | 303 | 2.3 |
| cellgrid | caph10-5000-50000-app40x | 5000 | 200000 | 1 | class_cap 1/1 | 0 | 3 / 3 | 6826 / 6826 | 4634 / 4634 | 1322 | shared 1/1 | 117 | 2576 (+0.0%) | 428 (+0.0%) | 0/217 | 7 | 0 / 0.000e0 | 20 | 2.3 |
| chrome | caph10-5000-50000-app40x | 5000 | 200000 | 1 | class_cap 1/1 | 0 | 2 / 2 | 5502 / 5502 | 4944 / 4944 | 1351 | shared 1/1 | 398 | 3424 (+0.0%) | 1668 (+0.0%) | 614/385 | 22 | 0 / 0.000e0 | 184 | 34.9 |
| chrome_channel | caph10-5000-50000-app40x | 5000 | 200000 | 1 | class_cap 1/1 | 0 | 2 / 2 | 5626 / 5626 | 4903 / 4903 | 1208 | shared 1/1 | 300 | 2640 (+0.0%) | 1418 (+0.0%) | 433/288 | 23 | 0 / 1.779e-5 | 63 | 11.9 |
| glyph16 | caph10-5000-50000-app40x | 5000–39940 | 200000–1597600 | 95 | class_cap 89/95, quiesced 6/95 | 0 | 2 / 7 | 13534 / 49513 | 5055 / 38813 | 9175 | shared 46/95, tree_cheaper 49/95 | 76404 | 845300 (-0.8%) | 327327 (-16.7%) | 51395/23633 | 19 | 0 / 3.438e-6 | 43365 | 19.7 |
| psychedelic | caph10-5000-50000-app40x | 5000 | 200000 | 1 | class_cap 1/1 | 0 | 3 / 3 | 7328 / 7328 | 4670 / 4670 | 1263 | tree_cheaper 1/1 | 112 | 2368 (+0.0%) | 825 (+0.0%) | 29/127 | 0 | 0 / 0.000e0 | 21 | 2.3 |
| shader | caph10-5000-50000-app40x | 5000 | 200000 | 12 | class_cap 9/12, quiesced 3/12 | 0 | 4.5 / 9 | 7408 / 30052 | 3094.5 / 4611 | 1308 | shared 3/12, tree_cheaper 9/12 | 636 | 15072 (+0.0%) | 4618 (+0.0%) | 652/1392 | 0 | 0 / 6.109e-6 | 308 | 2.3 |
| cellgrid | caph12-5000-50000-app40x | 5000 | 200000 | 1 | class_cap 1/1 | 0 | 3 / 3 | 6826 / 6826 | 4634 / 4634 | 1322 | shared 1/1 | 117 | 2576 (+0.0%) | 428 (+0.0%) | 0/217 | 7 | 0 / 0.000e0 | 20 | 2.3 |
| chrome | caph12-5000-50000-app40x | 5580 | 223200 | 1 | class_cap 1/1 | 0 | 2 / 2 | 6275 / 6275 | 5499 / 5499 | 1367 | shared 1/1 | 403 | 3472 (+1.4%) | 1686 (+1.1%) | 619/388 | 26 | 0 / 0.000e0 | 182 | 34.9 |
| chrome_channel | caph12-5000-50000-app40x | 5000 | 200000 | 1 | class_cap 1/1 | 0 | 2 / 2 | 5626 / 5626 | 4903 / 4903 | 1208 | shared 1/1 | 300 | 2640 (+0.0%) | 1418 (+0.0%) | 433/288 | 23 | 0 / 1.779e-5 | 63 | 11.9 |
| glyph16 | caph12-5000-50000-app40x | 5000–47928 | 200000–1917120 | 95 | class_cap 89/95, quiesced 6/95 | 0 | 2 / 7 | 15234 / 59344 | 6020 / 46088 | 11956 | shared 43/95, tree_cheaper 52/95 | 76458 | 843460 (-1.0%) | 328345 (-16.4%) | 51503/23669 | 26 | 0 / 3.397e-6 | 47188 | 22.7 |
| psychedelic | caph12-5000-50000-app40x | 5000 | 200000 | 1 | class_cap 1/1 | 0 | 3 / 3 | 7328 / 7328 | 4670 / 4670 | 1263 | tree_cheaper 1/1 | 112 | 2368 (+0.0%) | 825 (+0.0%) | 29/127 | 0 | 0 / 0.000e0 | 21 | 2.3 |
| shader | caph12-5000-50000-app40x | 5000 | 200000 | 12 | class_cap 9/12, quiesced 3/12 | 0 | 4.5 / 9 | 7408 / 30052 | 3094.5 / 4611 | 1308 | shared 3/12, tree_cheaper 9/12 | 636 | 15072 (+0.0%) | 4618 (+0.0%) | 652/1392 | 0 | 0 / 6.109e-6 | 301 | 2.3 |
| glyph16 | caph6-5000-50000-app40x | 5000–23964 | 200000–958560 | 95 | class_cap 89/95, quiesced 6/95 | 44 | 2 / 7 | 9522 / 19197 | 4718 / 23343 | 8986 | shared 48/95, tree_cheaper 47/95 | 78365 | 841988 (-1.2%) | 350726 (-10.7%) | 51395/23658 | 69 | 0 / 3.397e-6 | 9732 | 11.6 |
| glyph16 | caph7-5000-50000-app40x | 5000–27958 | 200000–1118320 | 95 | class_cap 89/95, quiesced 6/95 | 0 | 2 / 7 | 12657 / 29030 | 4718 / 27735 | 8750 | shared 44/95, tree_cheaper 51/95 | 76312 | 849508 (-0.3%) | 326999 (-16.7%) | 51347/23617 | 21 | 0 / 3.438e-6 | 19690 | 14.8 |
| cellgrid | caph8-5000-50000-app40x | 5000 | 200000 | 1 | class_cap 1/1 | 0 | 3 / 3 | 6826 / 6826 | 4634 / 4634 | 1322 | shared 1/1 | 117 | 2576 (+0.0%) | 428 (+0.0%) | 0/217 | 7 | 0 / 0.000e0 | 19 | 2.3 |
| chrome | caph8-5000-50000-app40x | 5000 | 200000 | 1 | class_cap 1/1 | 0 | 2 / 2 | 5502 / 5502 | 4944 / 4944 | 1351 | shared 1/1 | 398 | 3424 (+0.0%) | 1668 (+0.0%) | 614/385 | 22 | 0 / 0.000e0 | 182 | 34.9 |
| chrome_channel | caph8-5000-50000-app40x | 5000 | 200000 | 1 | class_cap 1/1 | 0 | 2 / 2 | 5626 / 5626 | 4903 / 4903 | 1208 | shared 1/1 | 300 | 2640 (+0.0%) | 1418 (+0.0%) | 433/288 | 23 | 0 / 1.779e-5 | 62 | 11.9 |
| glyph16 | caph8-5000-50000-app40x | 5000–31952 | 200000–1278080 | 95 | class_cap 89/95, quiesced 6/95 | 0 | 2 / 7 | 13534 / 36580 | 4718 / 31499 | 8769 | shared 47/95, tree_cheaper 48/95 | 76382 | 849652 (-0.3%) | 327214 (-16.7%) | 51395/23633 | 20 | 0 / 3.438e-6 | 25483 | 16.1 |
| psychedelic | caph8-5000-50000-app40x | 5000 | 200000 | 1 | class_cap 1/1 | 0 | 3 / 3 | 7328 / 7328 | 4670 / 4670 | 1263 | tree_cheaper 1/1 | 112 | 2368 (+0.0%) | 825 (+0.0%) | 29/127 | 0 | 0 / 0.000e0 | 21 | 2.3 |
| shader | caph8-5000-50000-app40x | 5000 | 200000 | 12 | class_cap 9/12, quiesced 3/12 | 0 | 4.5 / 9 | 7408 / 30052 | 3094.5 / 4611 | 1308 | shared 3/12, tree_cheaper 9/12 | 636 | 15072 (+0.0%) | 4618 (+0.0%) | 652/1392 | 0 | 0 / 6.109e-6 | 299 | 2.3 |
| cellgrid | capx4-5000-100000-app40x | 5000 | 200000 | 1 | class_cap 1/1 | 0 | 3 / 3 | 6826 / 6826 | 4634 / 4634 | 1322 | shared 1/1 | 117 | 2576 (+0.0%) | 428 (+0.0%) | 0/217 | 7 | 0 / 0.000e0 | 20 | 2.3 |
| chrome | capx4-5000-100000-app40x | 100000 | 4000000 | 1 | class_cap 1/1 | 0 | 3 / 3 | 142134 / 142134 | 85977 / 85977 | 16021 | tree_cheaper 1/1 | 568 | 4768 (+39.3%) | 2374 (+42.3%) | 888/547 | 57 | 0 / 0.000e0 | 3577 | 64.3 |
| chrome_channel | capx4-5000-100000-app40x | 100000 | 4000000 | 1 | class_cap 1/1 | 0 | 3 / 3 | 137787 / 137787 | 84136 / 84136 | 15535 | tree_cheaper 1/1 | 405 | 3456 (+30.9%) | 1860 (+31.2%) | 596/389 | 41 | 0 / 1.800e-5 | 3469 | 49.8 |
| glyph16 | capx4-5000-100000-app40x | 5000–35164 | 200000–1406560 | 95 | class_cap 89/95, quiesced 6/95 | 0 | 2 / 7 | 13534 / 42393 | 4718 / 34421 | 9086 | shared 46/95, tree_cheaper 49/95 | 76423 | 847396 (-0.5%) | 327323 (-16.7%) | 51389/23631 | 20 | 0 / 3.994e-6 | 31763 | 18.9 |
| psychedelic | capx4-5000-100000-app40x | 5000 | 200000 | 1 | class_cap 1/1 | 0 | 3 / 3 | 7328 / 7328 | 4670 / 4670 | 1263 | tree_cheaper 1/1 | 112 | 2368 (+0.0%) | 825 (+0.0%) | 29/127 | 0 | 0 / 0.000e0 | 22 | 2.3 |
| shader | capx4-5000-100000-app40x | 5000 | 200000 | 12 | class_cap 9/12, quiesced 3/12 | 0 | 4.5 / 9 | 7408 / 30052 | 3094.5 / 4611 | 1308 | shared 3/12, tree_cheaper 9/12 | 636 | 15072 (+0.0%) | 4618 (+0.0%) | 652/1392 | 0 | 0 / 6.109e-6 | 301 | 2.3 |
| cellgrid | capx6-5000-100000-app40x | 5000 | 200000 | 1 | class_cap 1/1 | 0 | 3 / 3 | 6826 / 6826 | 4634 / 4634 | 1322 | shared 1/1 | 117 | 2576 (+0.0%) | 428 (+0.0%) | 0/217 | 7 | 0 / 0.000e0 | 19 | 2.3 |
| chrome | capx6-5000-100000-app40x | 100000 | 4000000 | 1 | class_cap 1/1 | 0 | 3 / 3 | 142134 / 142134 | 85977 / 85977 | 16021 | tree_cheaper 1/1 | 568 | 4768 (+39.3%) | 2374 (+42.3%) | 888/547 | 57 | 0 / 0.000e0 | 3711 | 64.3 |
| chrome_channel | capx6-5000-100000-app40x | 100000 | 4000000 | 1 | class_cap 1/1 | 0 | 3 / 3 | 137787 / 137787 | 84136 / 84136 | 15535 | tree_cheaper 1/1 | 405 | 3456 (+30.9%) | 1860 (+31.2%) | 596/389 | 41 | 0 / 1.800e-5 | 3511 | 49.8 |
| glyph16 | capx6-5000-100000-app40x | 5000–52746 | 200000–2109840 | 95 | class_cap 89/95, quiesced 6/95 | 0 | 2 / 7 | 16090 / 65695 | 6321 / 50183 | 13371 | shared 43/95, tree_cheaper 52/95 | 75993 | 835300 (-2.0%) | 327618 (-16.6%) | 51470/23658 | 21 | 0 / 3.397e-6 | 53655 | 23.5 |
| psychedelic | capx6-5000-100000-app40x | 7476 | 299040 | 1 | class_cap 1/1 | 0 | 3 / 3 | 10172 / 10172 | 6470 / 6470 | 1572 | tree_cheaper 1/1 | 111 | 2368 (+0.0%) | 826 (+0.1%) | 29/127 | 0 | 0 / 0.000e0 | 24 | 2.9 |
| shader | capx6-5000-100000-app40x | 5000 | 200000 | 12 | class_cap 9/12, quiesced 3/12 | 0 | 4.5 / 9 | 7408 / 30052 | 3094.5 / 4611 | 1308 | shared 3/12, tree_cheaper 9/12 | 636 | 15072 (+0.0%) | 4618 (+0.0%) | 652/1392 | 0 | 0 / 6.109e-6 | 297 | 2.3 |
| cellgrid | capx8-5000-100000-app40x | 5336 | 213440 | 1 | class_cap 1/1 | 0 | 3 / 3 | 7308 / 7308 | 4898 / 4898 | 1366 | shared 1/1 | 117 | 2576 (+0.0%) | 428 (+0.0%) | 0/217 | 6 | 0 / 0.000e0 | 20 | 2.4 |
| chrome | capx8-5000-100000-app40x | 100000 | 4000000 | 1 | class_cap 1/1 | 0 | 3 / 3 | 142134 / 142134 | 85977 / 85977 | 16021 | tree_cheaper 1/1 | 568 | 4768 (+39.3%) | 2374 (+42.3%) | 888/547 | 57 | 0 / 0.000e0 | 3569 | 64.3 |
| chrome_channel | capx8-5000-100000-app40x | 100000 | 4000000 | 1 | class_cap 1/1 | 0 | 3 / 3 | 137787 / 137787 | 84136 / 84136 | 15535 | tree_cheaper 1/1 | 405 | 3456 (+30.9%) | 1860 (+31.2%) | 596/389 | 41 | 0 / 1.800e-5 | 3663 | 49.8 |
| glyph16 | capx8-5000-100000-app40x | 5000–70328 | 200000–2813120 | 95 | class_cap 89/95, quiesced 6/95 | 0 | 2 / 7 | 17275 / 91131 | 8116 / 63701 | 17203 | shared 34/95, tree_cheaper 61/95 | 76716 | 793444 (-6.9%) | 332471 (-15.3%) | 46932/23104 | 33 | 0 / 3.394e-6 | 66706 | 31.3 |
| psychedelic | capx8-5000-100000-app40x | 9968 | 398720 | 1 | class_cap 1/1 | 0 | 3 / 3 | 11307 / 11307 | 7131 / 7131 | 1727 | tree_cheaper 1/1 | 111 | 2368 (+0.0%) | 826 (+0.1%) | 29/127 | 0 | 0 / 0.000e0 | 26 | 3.4 |
| shader | capx8-5000-100000-app40x | 5000 | 200000 | 12 | class_cap 9/12, quiesced 3/12 | 0 | 4.5 / 9 | 7408 / 30052 | 3094.5 / 4611 | 1308 | shared 3/12, tree_cheaper 9/12 | 636 | 15072 (+0.0%) | 4618 (+0.0%) | 652/1392 | 0 | 0 / 6.109e-6 | 300 | 2.3 |
| cellgrid | cap10000-app200000 | 10000 | 200000 | 1 | class_cap 1/1 | 0 | 3 / 3 | 12749 / 12749 | 9059 / 9059 | 2057 | shared 1/1 | 117 | 2576 (+0.0%) | 428 (+0.0%) | 0/217 | 6 | 0 / 0.000e0 | 28 | 4.7 |
| chrome | cap10000-app200000 | 10000 | 200000 | 1 | class_cap 1/1 | 0 | 2 / 2 | 11528 / 11528 | 9495 / 9495 | 2933 | shared 1/1 | 404 | 3600 (+5.1%) | 1673 (+0.3%) | 612/388 | 28 | 0 / 0.000e0 | 198 | 34.9 |
| chrome_channel | cap10000-app200000 | 10000 | 200000 | 1 | class_cap 1/1 | 0 | 2 / 2 | 11432 / 11432 | 9413 / 9413 | 2850 | shared 1/1 | 294 | 2704 (+2.4%) | 1384 (-2.4%) | 404/281 | 24 | 0 / 1.788e-5 | 73 | 12.0 |
| glyph16 | cap10000-app200000 | 10000 | 200000 | 95 | class_cap 87/95, quiesced 8/95 | 33 | 2 / 28 | 13771 / 161545 | 8777 / 9988 | 5462 | shared 48/95, tree_cheaper 47/95 | 84619 | 860564 (+1.0%) | 373944 (-4.8%) | 51909/23823 | 76 | 0 / 3.368e-6 | 31469 | 6.0 |
| glyph32 | cap10000-app200000 | 10000 | 200000 | 95 | class_cap 87/95, quiesced 8/95 | 33 | 2 / 28 | 13771 / 161545 | 8777 / 9988 | 5462 | shared 48/95, tree_cheaper 47/95 | 84621 | 860596 (+1.0%) | 373944 (-4.8%) | 51909/23823 | 76 | 0 / 5.245e-6 | 31527 | 6.0 |
| psychedelic | cap10000-app200000 | 10000 | 200000 | 1 | class_cap 1/1 | 0 | 3 / 3 | 11318 / 11318 | 7141 / 7141 | 1732 | tree_cheaper 1/1 | 111 | 2368 (+0.0%) | 826 (+0.1%) | 29/127 | 0 | 0 / 0.000e0 | 26 | 3.4 |
| shader | cap10000-app200000 | 10000 | 200000 | 12 | class_cap 8/12, quiesced 4/12 | 0 | 5 / 15 | 16504.5 / 69343 | 6322 / 8852 | 2044 | shared 3/12, tree_cheaper 9/12 | 651 | 14928 (-1.0%) | 4691 (+1.6%) | 647/1408 | 0 | 0 / 6.109e-6 | 1580 | 4.6 |
| cellgrid | cap10000-app400000 | 10000 | 400000 | 1 | class_cap 1/1 | 0 | 3 / 3 | 12749 / 12749 | 9059 / 9059 | 2057 | shared 1/1 | 117 | 2576 (+0.0%) | 428 (+0.0%) | 0/217 | 6 | 0 / 0.000e0 | 28 | 4.7 |
| chrome | cap10000-app400000 | 10000 | 400000 | 1 | class_cap 1/1 | 0 | 2 / 2 | 11528 / 11528 | 9495 / 9495 | 2933 | shared 1/1 | 404 | 3600 (+5.1%) | 1673 (+0.3%) | 612/388 | 28 | 0 / 0.000e0 | 200 | 34.9 |
| chrome_channel | cap10000-app400000 | 10000 | 400000 | 1 | class_cap 1/1 | 0 | 2 / 2 | 11432 / 11432 | 9413 / 9413 | 2850 | shared 1/1 | 294 | 2704 (+2.4%) | 1384 (-2.4%) | 404/281 | 24 | 0 / 1.788e-5 | 73 | 12.0 |
| glyph16 | cap10000-app400000 | 10000 | 400000 | 95 | class_cap 87/95, quiesced 8/95 | 33 | 2 / 28 | 13771 / 161545 | 8777 / 9988 | 5462 | shared 48/95, tree_cheaper 47/95 | 84619 | 860564 (+1.0%) | 373944 (-4.8%) | 51909/23823 | 76 | 0 / 3.368e-6 | 31341 | 6.0 |
| glyph32 | cap10000-app400000 | 10000 | 400000 | 95 | class_cap 87/95, quiesced 8/95 | 33 | 2 / 28 | 13771 / 161545 | 8777 / 9988 | 5462 | shared 48/95, tree_cheaper 47/95 | 84621 | 860596 (+1.0%) | 373944 (-4.8%) | 51909/23823 | 76 | 0 / 5.245e-6 | 31290 | 6.0 |
| psychedelic | cap10000-app400000 | 10000 | 400000 | 1 | class_cap 1/1 | 0 | 3 / 3 | 11318 / 11318 | 7141 / 7141 | 1732 | tree_cheaper 1/1 | 111 | 2368 (+0.0%) | 826 (+0.1%) | 29/127 | 0 | 0 / 0.000e0 | 27 | 3.4 |
| shader | cap10000-app400000 | 10000 | 400000 | 12 | class_cap 8/12, quiesced 4/12 | 0 | 5 / 15 | 16504.5 / 69343 | 6322 / 8852 | 2044 | shared 3/12, tree_cheaper 9/12 | 651 | 14928 (-1.0%) | 4691 (+1.6%) | 647/1408 | 0 | 0 / 6.109e-6 | 1559 | 4.6 |
| cellgrid | cap20000-app200000 | 20000 | 200000 | 1 | class_cap 1/1 | 0 | 4 / 4 | 29932 / 29932 | 18425 / 18425 | 3998 | shared 1/1 | 117 | 2560 (-0.6%) | 427 (-0.2%) | 0/217 | 5 | 0 / 0.000e0 | 81 | 9.2 |
| chrome | cap20000-app200000 | 20000 | 200000 | 1 | class_cap 1/1 | 0 | 2 / 2 | 22744 / 22744 | 17886 / 17886 | 4054 | shared 1/1 | 403 | 3584 (+4.7%) | 1681 (+0.8%) | 604/384 | 30 | 0 / 0.000e0 | 215 | 34.9 |
| chrome_channel | cap20000-app200000 | 20000 | 200000 | 1 | class_cap 1/1 | 0 | 2 / 2 | 21309 / 21309 | 17003 / 17003 | 3829 | shared 1/1 | 294 | 2816 (+6.7%) | 1391 (-1.9%) | 416/280 | 34 | 0 / 1.770e-5 | 88 | 16.4 |
| glyph16 | cap20000-app200000 | 20000 | 200000 | 95 | application_budget 1/95, class_cap 84/95, quiesced 10/95 | 4 | 3 / 31 | 26760 / 200000 | 16560 / 19746 | 8358 | shared 47/95, tree_cheaper 48/95 | 77043 | 843700 (-1.0%) | 332119 (-15.4%) | 50466/23498 | 42 | 0 / 3.438e-6 | 245636 | 10.6 |
| glyph32 | cap20000-app200000 | 20000 | 200000 | 95 | application_budget 1/95, class_cap 84/95, quiesced 10/95 | 4 | 3 / 31 | 26760 / 200000 | 16560 / 19746 | 8358 | shared 47/95, tree_cheaper 48/95 | 77043 | 843716 (-1.0%) | 332115 (-15.4%) | 50466/23498 | 42 | 0 / 5.066e-6 | 174056 | 10.6 |
| psychedelic | cap20000-app200000 | 20000 | 200000 | 1 | class_cap 1/1 | 0 | 4 / 4 | 34631 / 34631 | 17907 / 17907 | 3694 | tree_cheaper 1/1 | 111 | 2352 (-0.7%) | 829 (+0.5%) | 29/127 | 0 | 0 / 0.000e0 | 76 | 9.0 |
| shader | cap20000-app200000 | 20000 | 200000 | 12 | class_cap 7/12, quiesced 5/12 | 0 | 6 / 18 | 30981 / 166339 | 9817 / 18774 | 4694 | shared 1/12, tree_cheaper 11/12 | 644 | 15264 (+1.3%) | 4645 (+0.6%) | 664/1400 | 0 | 0 / 6.109e-6 | 6769 | 9.3 |
| cellgrid | cap20000-app800000 | 20000 | 800000 | 1 | class_cap 1/1 | 0 | 4 / 4 | 29932 / 29932 | 18425 / 18425 | 3998 | shared 1/1 | 117 | 2560 (-0.6%) | 427 (-0.2%) | 0/217 | 5 | 0 / 0.000e0 | 82 | 9.2 |
| chrome | cap20000-app800000 | 20000 | 800000 | 1 | class_cap 1/1 | 0 | 2 / 2 | 22744 / 22744 | 17886 / 17886 | 4054 | shared 1/1 | 403 | 3584 (+4.7%) | 1681 (+0.8%) | 604/384 | 30 | 0 / 0.000e0 | 217 | 34.9 |
| chrome_channel | cap20000-app800000 | 20000 | 800000 | 1 | class_cap 1/1 | 0 | 2 / 2 | 21309 / 21309 | 17003 / 17003 | 3829 | shared 1/1 | 294 | 2816 (+6.7%) | 1391 (-1.9%) | 416/280 | 34 | 0 / 1.770e-5 | 88 | 16.4 |
| glyph16 | cap20000-app800000 | 20000 | 800000 | 95 | class_cap 85/95, quiesced 10/95 | 4 | 3 / 51 | 26760 / 426407 | 16560 / 19746 | 8358 | shared 47/95, tree_cheaper 48/95 | 77043 | 843700 (-1.0%) | 332119 (-15.4%) | 50466/23498 | 42 | 0 / 3.438e-6 | 180149 | 10.6 |
| glyph32 | cap20000-app800000 | 20000 | 800000 | 95 | class_cap 85/95, quiesced 10/95 | 4 | 3 / 51 | 26760 / 426407 | 16560 / 19746 | 8358 | shared 47/95, tree_cheaper 48/95 | 77043 | 843716 (-1.0%) | 332115 (-15.4%) | 50466/23498 | 42 | 0 / 5.066e-6 | 181666 | 10.6 |
| psychedelic | cap20000-app800000 | 20000 | 800000 | 1 | class_cap 1/1 | 0 | 4 / 4 | 34631 / 34631 | 17907 / 17907 | 3694 | tree_cheaper 1/1 | 111 | 2352 (-0.7%) | 829 (+0.5%) | 29/127 | 0 | 0 / 0.000e0 | 76 | 9.0 |
| shader | cap20000-app800000 | 20000 | 800000 | 12 | class_cap 7/12, quiesced 5/12 | 0 | 6 / 18 | 30981 / 166339 | 9817 / 18774 | 4694 | shared 1/12, tree_cheaper 11/12 | 644 | 15264 (+1.3%) | 4645 (+0.6%) | 664/1400 | 0 | 0 / 6.109e-6 | 6727 | 9.3 |
| cellgrid | cap50000-app200000 | 50000 | 200000 | 1 | class_cap 1/1 | 0 | 4 / 4 | 70275 / 70275 | 44441 / 44441 | 8233 | shared 1/1 | 117 | 2560 (-0.6%) | 427 (-0.2%) | 0/217 | 5 | 0 / 0.000e0 | 142 | 21.7 |
| chrome | cap50000-app200000 | 50000 | 200000 | 1 | class_cap 1/1 | 0 | 3 / 3 | 78903 / 78903 | 46074 / 46074 | 10256 | shared 1/1 | 518 | 4000 (+16.8%) | 2214 (+32.7%) | 824/512 | 33 | 0 / 0.000e0 | 2479 | 43.2 |
| chrome_channel | cap50000-app200000 | 50000 | 200000 | 1 | class_cap 1/1 | 0 | 3 / 3 | 77581 / 77581 | 45777 / 45777 | 10205 | shared 1/1 | 428 | 3232 (+22.4%) | 2002 (+41.2%) | 657/425 | 34 | 0 / 1.848e-5 | 2551 | 29.0 |
| glyph16 | cap50000-app200000 | 50000 | 200000 | 95 | application_budget 4/95, class_cap 81/95, quiesced 10/95 | 0 | 3 / 31 | 82466 / 200000 | 39811 / 48083 | 12932 | shared 47/95, tree_cheaper 48/95 | 80889 | 790196 (-7.3%) | 350192 (-10.8%) | 38260/22145 | 86 | 0 / 3.379e-6 | 1861425 | 23.1 |
| glyph32 | cap50000-app200000 | 50000 | 200000 | 95 | application_budget 4/95, class_cap 81/95, quiesced 10/95 | 0 | 3 / 31 | 82466 / 200000 | 39811 / 48083 | 12932 | shared 47/95, tree_cheaper 48/95 | 80893 | 790228 (-7.3%) | 350194 (-10.8%) | 38260/22145 | 86 | 0 / 5.126e-6 | 1586043 | 23.1 |
| psychedelic | cap50000-app200000 | 50000 | 200000 | 1 | class_cap 1/1 | 0 | 5 / 5 | 103075 / 103075 | 47130 / 47130 | 7712 | tree_cheaper 1/1 | 111 | 2352 (-0.7%) | 829 (+0.5%) | 29/127 | 0 | 0 / 0.000e0 | 249 | 21.6 |
| shader | cap50000-app200000 | 50000 | 200000 | 12 | application_budget 1/12, class_cap 6/12, quiesced 5/12 | 0 | 6.5 / 20 | 69623.5 / 200000 | 18599 / 44384 | 10173 | shared 2/12, tree_cheaper 10/12 | 723 | 15696 (+4.1%) | 4964 (+7.5%) | 720/1458 | 0 | 0 / 6.109e-6 | 20376 | 21.8 |
| cellgrid | cap50000-app2000000 | 50000 | 2000000 | 1 | class_cap 1/1 | 0 | 4 / 4 | 70275 / 70275 | 44441 / 44441 | 8233 | shared 1/1 | 117 | 2560 (-0.6%) | 427 (-0.2%) | 0/217 | 5 | 0 / 0.000e0 | 144 | 21.7 |
| chrome | cap50000-app2000000 | 50000 | 2000000 | 1 | class_cap 1/1 | 0 | 3 / 3 | 78903 / 78903 | 46074 / 46074 | 10256 | shared 1/1 | 518 | 4000 (+16.8%) | 2214 (+32.7%) | 824/512 | 33 | 0 / 0.000e0 | 2462 | 43.2 |
| chrome_channel | cap50000-app2000000 | 50000 | 2000000 | 1 | class_cap 1/1 | 0 | 3 / 3 | 77581 / 77581 | 45777 / 45777 | 10205 | shared 1/1 | 428 | 3232 (+22.4%) | 2002 (+41.2%) | 657/425 | 34 | 0 / 1.848e-5 | 2508 | 29.0 |
| glyph16 | cap50000-app2000000 | 50000 | 2000000 | 95 | class_cap 85/95, quiesced 10/95 | 0 | 3 / 91 | 82466 / 1449603 | 39811 / 48083 | 12932 | shared 47/95, tree_cheaper 48/95 | 80889 | 790196 (-7.3%) | 350192 (-10.8%) | 38260/22145 | 86 | 0 / 3.379e-6 | 1662068 | 23.1 |
| glyph32 | cap50000-app2000000 | 50000 | 2000000 | 95 | class_cap 85/95, quiesced 10/95 | 0 | 3 / 91 | 82466 / 1449603 | 39811 / 48083 | 12932 | shared 47/95, tree_cheaper 48/95 | 80893 | 790228 (-7.3%) | 350194 (-10.8%) | 38260/22145 | 86 | 0 / 5.126e-6 | 1952508 | 23.1 |
| psychedelic | cap50000-app2000000 | 50000 | 2000000 | 1 | class_cap 1/1 | 0 | 5 / 5 | 103075 / 103075 | 47130 / 47130 | 7712 | tree_cheaper 1/1 | 111 | 2352 (-0.7%) | 829 (+0.5%) | 29/127 | 0 | 0 / 0.000e0 | 246 | 21.6 |
| shader | cap50000-app2000000 | 50000 | 2000000 | 12 | class_cap 7/12, quiesced 5/12 | 0 | 6.5 / 30 | 69623.5 / 561412 | 22403.5 / 44384 | 10173 | shared 3/12, tree_cheaper 9/12 | 723 | 15696 (+4.1%) | 4976 (+7.8%) | 720/1457 | 0 | 0 / 6.109e-6 | 22207 | 21.8 |
| cellgrid | cap100000-app200000 | 100000 | 200000 | 1 | class_cap 1/1 | 0 | 5 / 5 | 165997 / 165997 | 90816 / 90816 | 15172 | shared 1/1 | 117 | 2560 (-0.6%) | 427 (-0.2%) | 0/217 | 5 | 0 / 0.000e0 | 471 | 43.1 |
| chrome | cap100000-app200000 | 100000 | 200000 | 1 | class_cap 1/1 | 0 | 3 / 3 | 142134 / 142134 | 85977 / 85977 | 16021 | tree_cheaper 1/1 | 568 | 4768 (+39.3%) | 2374 (+42.3%) | 888/547 | 57 | 0 / 0.000e0 | 3632 | 64.3 |
| chrome_channel | cap100000-app200000 | 100000 | 200000 | 1 | class_cap 1/1 | 0 | 3 / 3 | 137787 / 137787 | 84136 / 84136 | 15535 | tree_cheaper 1/1 | 405 | 3456 (+30.9%) | 1860 (+31.2%) | 596/389 | 41 | 0 / 1.800e-5 | 3499 | 49.8 |
| psychedelic | cap100000-app200000 | 100000 | 200000 | 1 | class_cap 1/1 | 0 | 5 / 5 | 170242 / 170242 | 92102 / 92102 | 20207 | tree_cheaper 1/1 | 111 | 2352 (-0.7%) | 829 (+0.5%) | 29/127 | 0 | 0 / 0.000e0 | 513 | 43.5 |
| shader | cap100000-app200000 | 100000 | 200000 | 12 | application_budget 3/12, class_cap 4/12, quiesced 5/12 | 0 | 7.5 / 20 | 159427.5 / 200000 | 25260 / 96399 | 18772 | shared 1/12, tree_cheaper 11/12 | 728 | 15776 (+4.7%) | 4998 (+8.2%) | 726/1474 | 0 | 0 / 6.109e-6 | 43353 | 44.1 |
| cellgrid | cap100000-app4000000 | 100000 | 4000000 | 1 | class_cap 1/1 | 0 | 5 / 5 | 165997 / 165997 | 90816 / 90816 | 15172 | shared 1/1 | 117 | 2560 (-0.6%) | 427 (-0.2%) | 0/217 | 5 | 0 / 0.000e0 | 454 | 43.1 |
| chrome | cap100000-app4000000 | 100000 | 4000000 | 1 | class_cap 1/1 | 0 | 3 / 3 | 142134 / 142134 | 85977 / 85977 | 16021 | tree_cheaper 1/1 | 568 | 4768 (+39.3%) | 2374 (+42.3%) | 888/547 | 57 | 0 / 0.000e0 | 3624 | 64.3 |
| chrome_channel | cap100000-app4000000 | 100000 | 4000000 | 1 | class_cap 1/1 | 0 | 3 / 3 | 137787 / 137787 | 84136 / 84136 | 15535 | tree_cheaper 1/1 | 405 | 3456 (+30.9%) | 1860 (+31.2%) | 596/389 | 41 | 0 / 1.800e-5 | 3555 | 49.8 |
| psychedelic | cap100000-app4000000 | 100000 | 4000000 | 1 | class_cap 1/1 | 0 | 5 / 5 | 170242 / 170242 | 92102 / 92102 | 20207 | tree_cheaper 1/1 | 111 | 2352 (-0.7%) | 829 (+0.5%) | 29/127 | 0 | 0 / 0.000e0 | 388 | 43.5 |
| shader | cap100000-app4000000 | 100000 | 4000000 | 12 | class_cap 7/12, quiesced 5/12 | 0 | 7.5 / 44 | 159427.5 / 1515517 | 41118.5 / 96399 | 18772 | shared 2/12, tree_cheaper 10/12 | 724 | 15744 (+4.5%) | 5032 (+9.0%) | 726/1479 | 0 | 0 / 6.109e-6 | 138836 | 44.1 |

## Kernels that moved against the baseline cap

For each arm, the kernels whose emitted bytes or `dag_cost` differ from the same kernel at the baseline (smallest) cap and the same application arm where one exists, else the baseline's only arm. `+` is worse.

### caph10-5000-50000-app40x: 34 better, 14 worse, 63 unchanged

| kernel | Δ bytes | Δ dag_cost |
|---|---:|---:|
| glyph16_U0074 | +48 | +8 |
| glyph16_U006C | +0 | -1 |
| glyph16_U0029 | +128 | -216 |
| glyph16_U006A | -80 | -9 |
| glyph16_U0066 | -96 | -19 |
| glyph16_U0028 | -32 | -285 |
| glyph16_U0075 | +112 | -448 |
| glyph16_U0079 | -48 | -290 |
| glyph16_U004A | -16 | -453 |
| glyph16_U0044 | -96 | -401 |
| glyph16_U0072 | -48 | -543 |
| glyph16_U0050 | -176 | -425 |
| glyph16_U0068 | -64 | -541 |
| glyph16_U006E | -176 | -537 |
| glyph16_U0039 | +912 | -2025 |
| glyph16_U0053 | +928 | -2067 |
| glyph16_U0073 | +832 | -1981 |
| glyph16_U0033 | +640 | -1957 |
| glyph16_U0065 | +176 | -1530 |
| glyph16_U0036 | +624 | -2000 |
| glyph16_U0032 | +176 | -1590 |
| glyph16_U0035 | +64 | -1555 |
| glyph16_U0042 | +32 | -1564 |
| glyph16_U007E | -16 | -1564 |
| glyph16_U0055 | -80 | -1551 |
| glyph16_U0043 | -112 | -1550 |
| glyph16_U0064 | -144 | -1540 |
| glyph16_U0063 | -176 | -1542 |
| glyph16_U0070 | -208 | -1539 |
| glyph16_U006D | -304 | -1530 |
| glyph16_U0047 | -368 | -1480 |
| glyph16_U0062 | -368 | -1522 |
| glyph16_U0051 | -320 | -1571 |
| glyph16_U0052 | -512 | -1399 |
| glyph16_U0071 | -448 | -1501 |
| glyph16_U003F | -400 | -1572 |
| glyph16_U006F | -560 | -1423 |
| glyph16_U0026 | +864 | -2851 |
| glyph16_U0024 | +208 | -2261 |
| glyph16_U0030 | -336 | -1897 |
| glyph16_U004F | -720 | -1521 |
| glyph16_U0061 | -368 | -1917 |
| glyph16_U0067 | -400 | -2036 |
| glyph16_U007B | -1040 | -1596 |
| glyph16_U007D | -1360 | -1589 |
| glyph16_U0038 | -1104 | -2520 |
| glyph16_U0025 | -816 | -3091 |
| glyph16_U0040 | -1504 | -2438 |

### caph12-5000-50000-app40x: 28 better, 21 worse, 62 unchanged

| kernel | Δ bytes | Δ dag_cost |
|---|---:|---:|
| chrome_packed | +48 | +18 |
| glyph16_U0074 | +0 | +13 |
| glyph16_U0066 | +32 | -26 |
| glyph16_U006C | -48 | +7 |
| glyph16_U006A | -48 | +1 |
| glyph16_U0029 | +128 | -216 |
| glyph16_U0028 | +128 | -279 |
| glyph16_U0044 | +64 | -425 |
| glyph16_U0079 | -96 | -291 |
| glyph16_U0075 | +48 | -436 |
| glyph16_U004A | +0 | -443 |
| glyph16_U0068 | -64 | -519 |
| glyph16_U0072 | -64 | -521 |
| glyph16_U0050 | -160 | -449 |
| glyph16_U006E | -176 | -521 |
| glyph16_U0073 | +1184 | -1984 |
| glyph16_U0039 | +1072 | -1981 |
| glyph16_U0035 | +416 | -1561 |
| glyph16_U0033 | +592 | -1887 |
| glyph16_U0047 | +96 | -1402 |
| glyph16_U0065 | +96 | -1527 |
| glyph16_U0042 | +64 | -1540 |
| glyph16_U0032 | +48 | -1599 |
| glyph16_U0043 | -64 | -1556 |
| glyph16_U0064 | -128 | -1537 |
| glyph16_U007E | -112 | -1558 |
| glyph16_U0070 | -160 | -1536 |
| glyph16_U0055 | -144 | -1556 |
| glyph16_U0061 | +160 | -1861 |
| glyph16_U0067 | +304 | -2030 |
| glyph16_U0063 | -224 | -1556 |
| glyph16_U0062 | -304 | -1488 |
| glyph16_U0071 | -368 | -1445 |
| glyph16_U0026 | +896 | -2810 |
| glyph16_U0052 | -512 | -1404 |
| glyph16_U0051 | -384 | -1568 |
| glyph16_U006D | -400 | -1567 |
| glyph16_U0024 | +256 | -2246 |
| glyph16_U006F | -672 | -1424 |
| glyph16_U003F | -576 | -1552 |
| glyph16_U004F | -768 | -1487 |
| glyph16_U007B | -704 | -1595 |
| glyph16_U0030 | -432 | -1868 |
| glyph16_U0053 | -512 | -1982 |
| glyph16_U007D | -912 | -1586 |
| glyph16_U0036 | -688 | -1952 |
| glyph16_U0040 | -1008 | -2459 |
| glyph16_U0038 | -2112 | -2288 |
| glyph16_U0025 | -2336 | -2915 |

### caph6-5000-50000-app40x: 32 better, 10 worse, 53 unchanged

| kernel | Δ bytes | Δ dag_cost |
|---|---:|---:|
| glyph16_U0028 | +0 | -40 |
| glyph16_U0030 | +992 | -1059 |
| glyph16_U0050 | -96 | -164 |
| glyph16_U004A | -96 | -173 |
| glyph16_U0075 | -96 | -178 |
| glyph16_U0044 | -144 | -153 |
| glyph16_U0068 | -160 | -217 |
| glyph16_U006E | -160 | -217 |
| glyph16_U0072 | -192 | -249 |
| glyph16_U0051 | +208 | -956 |
| glyph16_U006F | +48 | -848 |
| glyph16_U0033 | +432 | -1241 |
| glyph16_U0035 | +128 | -973 |
| glyph16_U0039 | +496 | -1351 |
| glyph16_U0053 | +432 | -1354 |
| glyph16_U007E | +112 | -1075 |
| glyph16_U0073 | +272 | -1250 |
| glyph16_U0065 | -16 | -1015 |
| glyph16_U0055 | -16 | -1091 |
| glyph16_U0064 | -96 | -1022 |
| glyph16_U0036 | +128 | -1339 |
| glyph16_U006D | -208 | -1022 |
| glyph16_U0042 | -176 | -1064 |
| glyph16_U0070 | -224 | -1092 |
| glyph16_U0032 | -240 | -1105 |
| glyph16_U003F | -320 | -1074 |
| glyph16_U0063 | -368 | -1047 |
| glyph16_U004F | -448 | -983 |
| glyph16_U0043 | -384 | -1060 |
| glyph16_U0024 | -16 | -1463 |
| glyph16_U0061 | -320 | -1173 |
| glyph16_U0071 | -480 | -1019 |
| glyph16_U0062 | -576 | -1107 |
| glyph16_U0052 | -752 | -984 |
| glyph16_U0047 | -704 | -1044 |
| glyph16_U0067 | -576 | -1321 |
| glyph16_U0026 | -400 | -1861 |
| glyph16_U0038 | -704 | -1640 |
| glyph16_U0040 | -784 | -1739 |
| glyph16_U007B | -1520 | -1090 |
| glyph16_U007D | -1632 | -1095 |
| glyph16_U0025 | -1408 | -2083 |

### caph7-5000-50000-app40x: 27 better, 17 worse, 51 unchanged

| kernel | Δ bytes | Δ dag_cost |
|---|---:|---:|
| glyph16_U0029 | +64 | -233 |
| glyph16_U0028 | +96 | -292 |
| glyph16_U0075 | +80 | -462 |
| glyph16_U0079 | -80 | -303 |
| glyph16_U0044 | +0 | -394 |
| glyph16_U004A | +48 | -454 |
| glyph16_U0050 | +0 | -418 |
| glyph16_U0072 | -16 | -537 |
| glyph16_U0068 | -32 | -537 |
| glyph16_U006E | -80 | -527 |
| glyph16_U0039 | +992 | -2021 |
| glyph16_U0033 | +832 | -1970 |
| glyph16_U0073 | +816 | -2000 |
| glyph16_U0053 | +896 | -2096 |
| glyph16_U0065 | +272 | -1543 |
| glyph16_U0036 | +720 | -2015 |
| glyph16_U0035 | +192 | -1564 |
| glyph16_U0042 | +144 | -1550 |
| glyph16_U0064 | +112 | -1530 |
| glyph16_U0032 | +144 | -1596 |
| glyph16_U007E | +96 | -1578 |
| glyph16_U0070 | +16 | -1547 |
| glyph16_U0055 | -16 | -1538 |
| glyph16_U0043 | -48 | -1559 |
| glyph16_U0063 | -96 | -1551 |
| glyph16_U006D | -128 | -1532 |
| glyph16_U0062 | -144 | -1565 |
| glyph16_U0071 | -240 | -1514 |
| glyph16_U003F | -256 | -1554 |
| glyph16_U0047 | -288 | -1523 |
| glyph16_U0052 | -464 | -1388 |
| glyph16_U0051 | -320 | -1568 |
| glyph16_U0026 | +912 | -2856 |
| glyph16_U006F | -640 | -1427 |
| glyph16_U004F | -640 | -1483 |
| glyph16_U0030 | -304 | -1839 |
| glyph16_U0061 | -288 | -1914 |
| glyph16_U0024 | +0 | -2271 |
| glyph16_U0067 | -400 | -2038 |
| glyph16_U007B | -1184 | -1598 |
| glyph16_U007D | -1312 | -1584 |
| glyph16_U0040 | -656 | -2694 |
| glyph16_U0025 | -304 | -3075 |
| glyph16_U0038 | -1040 | -2520 |

### caph8-5000-50000-app40x: 26 better, 18 worse, 67 unchanged

| kernel | Δ bytes | Δ dag_cost |
|---|---:|---:|
| glyph16_U0028 | +144 | -280 |
| glyph16_U0029 | +64 | -234 |
| glyph16_U0075 | +112 | -449 |
| glyph16_U0079 | -48 | -290 |
| glyph16_U004A | +64 | -443 |
| glyph16_U0044 | -48 | -397 |
| glyph16_U0050 | -64 | -421 |
| glyph16_U0068 | -32 | -537 |
| glyph16_U0072 | -32 | -539 |
| glyph16_U006E | -80 | -527 |
| glyph16_U0065 | +640 | -1524 |
| glyph16_U0039 | +992 | -2021 |
| glyph16_U0033 | +832 | -1970 |
| glyph16_U0073 | +848 | -2002 |
| glyph16_U0053 | +928 | -2091 |
| glyph16_U0036 | +688 | -2021 |
| glyph16_U0035 | +176 | -1561 |
| glyph16_U0042 | +128 | -1553 |
| glyph16_U007E | +80 | -1573 |
| glyph16_U0055 | +16 | -1529 |
| glyph16_U0032 | +64 | -1580 |
| glyph16_U0064 | -80 | -1541 |
| glyph16_U006D | -144 | -1536 |
| glyph16_U0043 | -144 | -1569 |
| glyph16_U0070 | -176 | -1558 |
| glyph16_U0063 | -192 | -1561 |
| glyph16_U0047 | -336 | -1510 |
| glyph16_U003F | -320 | -1557 |
| glyph16_U0062 | -336 | -1575 |
| glyph16_U0026 | +928 | -2846 |
| glyph16_U0052 | -528 | -1396 |
| glyph16_U0071 | -416 | -1524 |
| glyph16_U006F | -528 | -1425 |
| glyph16_U0051 | -400 | -1592 |
| glyph16_U004F | -560 | -1510 |
| glyph16_U0024 | +176 | -2270 |
| glyph16_U0061 | -288 | -1914 |
| glyph16_U0030 | -320 | -1896 |
| glyph16_U0040 | +112 | -2423 |
| glyph16_U0067 | -400 | -2037 |
| glyph16_U007B | -1248 | -1587 |
| glyph16_U007D | -1280 | -1573 |
| glyph16_U0025 | -352 | -3078 |
| glyph16_U0038 | -1040 | -2523 |

### capx4-5000-100000-app40x: 26 better, 20 worse, 65 unchanged

| kernel | Δ bytes | Δ dag_cost |
|---|---:|---:|
| chrome_packed | +1344 | +706 |
| chrome_R | +816 | +442 |
| glyph16_U0029 | +128 | -218 |
| glyph16_U0028 | +32 | -281 |
| glyph16_U0075 | +112 | -449 |
| glyph16_U0079 | -48 | -291 |
| glyph16_U004A | +64 | -443 |
| glyph16_U0044 | -96 | -405 |
| glyph16_U0050 | -96 | -428 |
| glyph16_U0072 | -32 | -540 |
| glyph16_U0068 | -48 | -538 |
| glyph16_U006E | -96 | -528 |
| glyph16_U0039 | +976 | -2023 |
| glyph16_U0053 | +928 | -2077 |
| glyph16_U0073 | +832 | -1991 |
| glyph16_U0033 | +752 | -1968 |
| glyph16_U0065 | +208 | -1541 |
| glyph16_U0036 | +672 | -2016 |
| glyph16_U0042 | +128 | -1545 |
| glyph16_U0064 | +112 | -1532 |
| glyph16_U0035 | +128 | -1563 |
| glyph16_U0055 | +16 | -1520 |
| glyph16_U0032 | +64 | -1590 |
| glyph16_U007E | +48 | -1574 |
| glyph16_U0043 | -32 | -1531 |
| glyph16_U0070 | -144 | -1551 |
| glyph16_U0071 | -208 | -1495 |
| glyph16_U0063 | -224 | -1531 |
| glyph16_U006D | -240 | -1540 |
| glyph16_U0047 | -336 | -1522 |
| glyph16_U0062 | -320 | -1562 |
| glyph16_U003F | -336 | -1567 |
| glyph16_U0026 | +928 | -2846 |
| glyph16_U0052 | -528 | -1397 |
| glyph16_U006F | -512 | -1416 |
| glyph16_U0051 | -432 | -1587 |
| glyph16_U0024 | +208 | -2267 |
| glyph16_U0030 | -304 | -1891 |
| glyph16_U0061 | -288 | -1916 |
| glyph16_U004F | -736 | -1536 |
| glyph16_U007B | -1232 | -1572 |
| glyph16_U007D | -1296 | -1557 |
| glyph16_U0067 | -1104 | -1806 |
| glyph16_U0025 | -384 | -3069 |
| glyph16_U0040 | -816 | -2697 |
| glyph16_U0038 | -1104 | -2517 |

### capx6-5000-100000-app40x: 39 better, 12 worse, 60 unchanged

| kernel | Δ bytes | Δ dag_cost |
|---|---:|---:|
| chrome_packed | +1344 | +706 |
| chrome_R | +816 | +442 |
| psychedelic_packed | +0 | +1 |
| glyph16_U0074 | -80 | +0 |
| glyph16_U0029 | +112 | -210 |
| glyph16_U0028 | +128 | -272 |
| glyph16_U0066 | -112 | -50 |
| glyph16_U006C | -176 | -13 |
| glyph16_U006A | -224 | -41 |
| glyph16_U0044 | +64 | -418 |
| glyph16_U0050 | -176 | -452 |
| glyph16_U0075 | -192 | -496 |
| glyph16_U0072 | -192 | -539 |
| glyph16_U0079 | -416 | -345 |
| glyph16_U006E | -224 | -560 |
| glyph16_U0068 | -224 | -572 |
| glyph16_U004A | -496 | -521 |
| glyph16_U0073 | +592 | -1976 |
| glyph16_U0071 | +32 | -1432 |
| glyph16_U0032 | +48 | -1576 |
| glyph16_U0042 | -16 | -1555 |
| glyph16_U0067 | +320 | -1911 |
| glyph16_U0065 | -64 | -1548 |
| glyph16_U0051 | -64 | -1579 |
| glyph16_U0062 | -192 | -1462 |
| glyph16_U0047 | -224 | -1443 |
| glyph16_U004F | -224 | -1502 |
| glyph16_U007E | -176 | -1561 |
| glyph16_U0063 | -256 | -1579 |
| glyph16_U0061 | -16 | -1855 |
| glyph16_U0026 | +816 | -2799 |
| glyph16_U0070 | -416 | -1604 |
| glyph16_U0030 | -352 | -1738 |
| glyph16_U0052 | -720 | -1422 |
| glyph16_U0035 | -480 | -1668 |
| glyph16_U0043 | -592 | -1644 |
| glyph16_U0064 | -656 | -1617 |
| glyph16_U0025 | +448 | -2751 |
| glyph16_U0039 | -480 | -1878 |
| glyph16_U003F | -848 | -1519 |
| glyph16_U0053 | -576 | -1958 |
| glyph16_U0036 | -624 | -1919 |
| glyph16_U0055 | -864 | -1702 |
| glyph16_U006F | -1072 | -1494 |
| glyph16_U0024 | -336 | -2300 |
| glyph16_U0033 | -816 | -1873 |
| glyph16_U006D | -1296 | -1526 |
| glyph16_U007B | -1536 | -1698 |
| glyph16_U007D | -1632 | -1691 |
| glyph16_U0038 | -1008 | -2359 |
| glyph16_U0040 | -1264 | -2511 |

### capx8-5000-100000-app40x: 43 better, 8 worse, 60 unchanged

| kernel | Δ bytes | Δ dag_cost |
|---|---:|---:|
| chrome_packed | +1344 | +706 |
| chrome_R | +816 | +442 |
| glyph16_U006A | +0 | +145 |
| glyph16_U0074 | +16 | +120 |
| glyph16_U0066 | +0 | +70 |
| psychedelic_packed | +0 | +1 |
| glyph16_U0029 | +64 | -83 |
| glyph16_U0079 | -16 | -109 |
| glyph16_U006C | -240 | +34 |
| glyph16_U0028 | -48 | -173 |
| glyph16_U0075 | -176 | -387 |
| glyph16_U0050 | -160 | -418 |
| glyph16_U0044 | -352 | -344 |
| glyph16_U0072 | -272 | -553 |
| glyph16_U004A | -544 | -385 |
| glyph16_U0068 | -608 | -551 |
| glyph16_U006E | -704 | -570 |
| glyph16_U0051 | -144 | -1343 |
| glyph16_U0061 | -496 | -1449 |
| glyph16_U0052 | -736 | -1327 |
| glyph16_U0055 | -656 | -1459 |
| glyph16_U0039 | -144 | -2041 |
| glyph16_U0035 | -896 | -1668 |
| glyph16_U0065 | -928 | -1712 |
| glyph16_U0063 | -1088 | -1605 |
| glyph16_U0043 | -1104 | -1598 |
| glyph16_U0033 | -688 | -2042 |
| glyph16_U0071 | -1184 | -1563 |
| glyph16_U0040 | -560 | -2199 |
| glyph16_U0062 | -1200 | -1590 |
| glyph16_U0053 | -528 | -2282 |
| glyph16_U007E | -1072 | -1750 |
| glyph16_U0070 | -1152 | -1703 |
| glyph16_U0064 | -1184 | -1683 |
| glyph16_U0036 | -736 | -2158 |
| glyph16_U004F | -1408 | -1506 |
| glyph16_U0067 | -912 | -2055 |
| glyph16_U0030 | -1104 | -1868 |
| glyph16_U006D | -2560 | -827 |
| glyph16_U006F | -2592 | -901 |
| glyph16_U0047 | -2560 | -1138 |
| glyph16_U0042 | -2544 | -1394 |
| glyph16_U007D | -2976 | -1104 |
| glyph16_U003F | -3056 | -1078 |
| glyph16_U0032 | -2816 | -1461 |
| glyph16_U007B | -3104 | -1231 |
| glyph16_U0038 | -1888 | -2567 |
| glyph16_U0073 | -2880 | -1662 |
| glyph16_U0024 | -3904 | -1550 |
| glyph16_U0025 | -2560 | -3216 |
| glyph16_U0026 | -4208 | -2352 |

### cap10000-app200000: 88 better, 75 worse, 43 unchanged

| kernel | Δ bytes | Δ dag_cost |
|---|---:|---:|
| glyph16_U0039 | +2080 | +407 |
| glyph32_U0039 | +2080 | +407 |
| glyph16_U0053 | +1856 | +477 |
| glyph32_U0053 | +1856 | +477 |
| glyph16_U0030 | +1920 | +397 |
| glyph32_U0030 | +1920 | +397 |
| glyph16_U0036 | +1824 | +383 |
| glyph32_U0036 | +1824 | +383 |
| glyph16_U0073 | +1648 | +408 |
| glyph32_U0073 | +1648 | +408 |
| glyph16_U0033 | +1568 | +387 |
| glyph32_U0033 | +1568 | +387 |
| glyph16_U0067 | +1504 | +450 |
| glyph32_U0067 | +1504 | +450 |
| glyph16_U0026 | +944 | +300 |
| glyph32_U0026 | +944 | +300 |
| glyph16_U0061 | +768 | +309 |
| glyph32_U0061 | +768 | +309 |
| glyph16_U0040 | +560 | +331 |
| glyph16_U0038 | +416 | +472 |
| glyph32_U0038 | +416 | +472 |
| glyph32_U0040 | +544 | +326 |
| glyph16_U0051 | +928 | -401 |
| glyph32_U0051 | +928 | -401 |
| glyph16_U0078 | +240 | +10 |
| glyph32_U0078 | +240 | +10 |
| glyph16_U0023 | +192 | +20 |
| glyph32_U0023 | +192 | +20 |
| glyph16_U0058 | +192 | -4 |
| glyph32_U0058 | +192 | -4 |
| chrome_packed | +176 | +5 |
| glyph16_U002A | +96 | +32 |
| glyph32_U002A | +96 | +32 |
| glyph16_U006A | -32 | +130 |
| glyph32_U006A | -32 | +130 |
| shader_mandelbrot_distance | +48 | +40 |
| glyph16_U0074 | -48 | +109 |
| glyph32_U0074 | -48 | +109 |
| glyph16_U0066 | +0 | +54 |
| glyph32_U0066 | +0 | +54 |
| glyph16_U004B | +80 | -37 |
| glyph16_U006B | +80 | -37 |
| glyph32_U004B | +80 | -37 |
| glyph32_U006B | +80 | -37 |
| chrome_R | +64 | -34 |
| psychedelic_packed | +0 | +1 |
| glyph16_U007C | +0 | -3 |
| glyph32_U007C | +0 | -3 |
| glyph16_U003E | +80 | -97 |
| glyph32_U003E | +80 | -97 |
| glyph16_U0041 | +0 | -24 |
| glyph16_U0057 | +16 | -40 |
| glyph32_U0041 | +0 | -24 |
| glyph32_U0057 | +16 | -40 |
| shader_metaballs | -16 | -15 |
| glyph16_U003C | +64 | -97 |
| glyph16_U0059 | +0 | -33 |
| glyph32_U003C | +64 | -97 |
| glyph32_U0059 | +0 | -33 |
| glyph16_U0069 | -32 | -5 |
| glyph32_U0069 | -32 | -5 |
| shader_julia_set | -112 | +74 |
| glyph16_U004D | +0 | -41 |
| glyph32_U004D | +0 | -41 |
| glyph16_U0034 | -16 | -34 |
| glyph32_U0034 | -16 | -34 |
| glyph16_U005E | -32 | -32 |
| glyph32_U005E | -32 | -32 |
| glyph16_U002C | -32 | -38 |
| glyph32_U002C | -32 | -38 |
| glyph16_U0024 | -560 | +485 |
| glyph32_U0024 | -560 | +485 |
| glyph16_U002F | -48 | -38 |
| glyph16_U005C | -48 | -38 |
| glyph32_U002F | -48 | -38 |
| glyph32_U005C | -48 | -38 |
| glyph16_U007A | -16 | -71 |
| glyph32_U007A | -16 | -71 |
| glyph16_U0029 | +128 | -216 |
| glyph16_U0060 | -48 | -40 |
| glyph16_U0077 | -48 | -40 |
| glyph32_U0029 | +128 | -216 |
| glyph32_U0060 | -48 | -40 |
| glyph32_U0077 | -48 | -40 |
| shader_domain_warp_fbm | -64 | -26 |
| glyph16_U0031 | -64 | -38 |
| glyph32_U0031 | -64 | -38 |
| glyph16_U0037 | -64 | -50 |
| glyph32_U0037 | -64 | -50 |
| glyph16_U004E | -80 | -50 |
| glyph32_U004E | -80 | -50 |
| glyph16_U0076 | -32 | -100 |
| glyph32_U0076 | -32 | -100 |
| glyph16_U005A | -64 | -75 |
| glyph32_U005A | -64 | -75 |
| glyph16_U0028 | +128 | -279 |
| glyph32_U0028 | +128 | -279 |
| glyph16_U006C | -208 | +54 |
| glyph32_U006C | -208 | +54 |
| glyph16_U003B | -80 | -79 |
| glyph32_U003B | -80 | -79 |
| glyph16_U0021 | -112 | -87 |
| glyph32_U0021 | -112 | -87 |
| glyph16_U0056 | -144 | -135 |
| glyph32_U0056 | -144 | -135 |
| glyph16_U0075 | +112 | -448 |
| glyph32_U0075 | +112 | -448 |
| glyph16_U0079 | -112 | -292 |
| glyph32_U0079 | -112 | -292 |
| glyph16_U004A | +0 | -450 |
| glyph32_U004A | +0 | -450 |
| glyph16_U0044 | -96 | -409 |
| glyph32_U0044 | -96 | -409 |
| glyph16_U0050 | -96 | -423 |
| glyph32_U0050 | -96 | -423 |
| glyph16_U0072 | -32 | -534 |
| glyph32_U0072 | -32 | -534 |
| glyph16_U0035 | +224 | -811 |
| glyph32_U0035 | +224 | -811 |
| glyph16_U0068 | -64 | -541 |
| glyph32_U0068 | -64 | -541 |
| glyph16_U006E | -176 | -538 |
| glyph32_U006E | -176 | -538 |
| glyph16_U0065 | +96 | -811 |
| glyph32_U0065 | +96 | -811 |
| glyph16_U0064 | +48 | -862 |
| glyph32_U0064 | +48 | -862 |
| glyph16_U007E | +112 | -945 |
| glyph32_U007E | +112 | -945 |
| glyph16_U006D | +0 | -864 |
| glyph32_U006D | +0 | -864 |
| glyph16_U0070 | +16 | -888 |
| glyph32_U0070 | +16 | -888 |
| glyph16_U004F | -48 | -859 |
| glyph32_U004F | -48 | -859 |
| glyph16_U003F | -144 | -765 |
| glyph32_U003F | -144 | -765 |
| glyph16_U0042 | -96 | -914 |
| glyph32_U0042 | -96 | -914 |
| glyph16_U0032 | -80 | -969 |
| glyph32_U0032 | -80 | -969 |
| glyph16_U0055 | -48 | -1023 |
| glyph32_U0055 | -48 | -1023 |
| glyph16_U006F | -192 | -956 |
| glyph32_U006F | -192 | -956 |
| glyph16_U0063 | -288 | -889 |
| glyph32_U0063 | -288 | -889 |
| glyph16_U0025 | -1280 | +79 |
| glyph32_U0025 | -1280 | +79 |
| glyph16_U0043 | -304 | -910 |
| glyph32_U0043 | -304 | -910 |
| glyph16_U0071 | -384 | -861 |
| glyph32_U0071 | -384 | -861 |
| glyph16_U0062 | -400 | -909 |
| glyph32_U0062 | -400 | -909 |
| glyph16_U0047 | -496 | -814 |
| glyph32_U0047 | -496 | -814 |
| glyph16_U0052 | -464 | -1388 |
| glyph32_U0052 | -464 | -1388 |
| glyph16_U007D | -1408 | -872 |
| glyph32_U007D | -1408 | -872 |
| glyph16_U007B | -1392 | -903 |
| glyph32_U007B | -1392 | -903 |

### cap10000-app400000: 88 better, 75 worse, 43 unchanged

| kernel | Δ bytes | Δ dag_cost |
|---|---:|---:|
| glyph16_U0039 | +2080 | +407 |
| glyph32_U0039 | +2080 | +407 |
| glyph16_U0053 | +1856 | +477 |
| glyph32_U0053 | +1856 | +477 |
| glyph16_U0030 | +1920 | +397 |
| glyph32_U0030 | +1920 | +397 |
| glyph16_U0036 | +1824 | +383 |
| glyph32_U0036 | +1824 | +383 |
| glyph16_U0073 | +1648 | +408 |
| glyph32_U0073 | +1648 | +408 |
| glyph16_U0033 | +1568 | +387 |
| glyph32_U0033 | +1568 | +387 |
| glyph16_U0067 | +1504 | +450 |
| glyph32_U0067 | +1504 | +450 |
| glyph16_U0026 | +944 | +300 |
| glyph32_U0026 | +944 | +300 |
| glyph16_U0061 | +768 | +309 |
| glyph32_U0061 | +768 | +309 |
| glyph16_U0040 | +560 | +331 |
| glyph16_U0038 | +416 | +472 |
| glyph32_U0038 | +416 | +472 |
| glyph32_U0040 | +544 | +326 |
| glyph16_U0051 | +928 | -401 |
| glyph32_U0051 | +928 | -401 |
| glyph16_U0078 | +240 | +10 |
| glyph32_U0078 | +240 | +10 |
| glyph16_U0023 | +192 | +20 |
| glyph32_U0023 | +192 | +20 |
| glyph16_U0058 | +192 | -4 |
| glyph32_U0058 | +192 | -4 |
| chrome_packed | +176 | +5 |
| glyph16_U002A | +96 | +32 |
| glyph32_U002A | +96 | +32 |
| glyph16_U006A | -32 | +130 |
| glyph32_U006A | -32 | +130 |
| shader_mandelbrot_distance | +48 | +40 |
| glyph16_U0074 | -48 | +109 |
| glyph32_U0074 | -48 | +109 |
| glyph16_U0066 | +0 | +54 |
| glyph32_U0066 | +0 | +54 |
| glyph16_U004B | +80 | -37 |
| glyph16_U006B | +80 | -37 |
| glyph32_U004B | +80 | -37 |
| glyph32_U006B | +80 | -37 |
| chrome_R | +64 | -34 |
| psychedelic_packed | +0 | +1 |
| glyph16_U007C | +0 | -3 |
| glyph32_U007C | +0 | -3 |
| glyph16_U003E | +80 | -97 |
| glyph32_U003E | +80 | -97 |
| glyph16_U0041 | +0 | -24 |
| glyph16_U0057 | +16 | -40 |
| glyph32_U0041 | +0 | -24 |
| glyph32_U0057 | +16 | -40 |
| shader_metaballs | -16 | -15 |
| glyph16_U003C | +64 | -97 |
| glyph16_U0059 | +0 | -33 |
| glyph32_U003C | +64 | -97 |
| glyph32_U0059 | +0 | -33 |
| glyph16_U0069 | -32 | -5 |
| glyph32_U0069 | -32 | -5 |
| shader_julia_set | -112 | +74 |
| glyph16_U004D | +0 | -41 |
| glyph32_U004D | +0 | -41 |
| glyph16_U0034 | -16 | -34 |
| glyph32_U0034 | -16 | -34 |
| glyph16_U005E | -32 | -32 |
| glyph32_U005E | -32 | -32 |
| glyph16_U002C | -32 | -38 |
| glyph32_U002C | -32 | -38 |
| glyph16_U0024 | -560 | +485 |
| glyph32_U0024 | -560 | +485 |
| glyph16_U002F | -48 | -38 |
| glyph16_U005C | -48 | -38 |
| glyph32_U002F | -48 | -38 |
| glyph32_U005C | -48 | -38 |
| glyph16_U007A | -16 | -71 |
| glyph32_U007A | -16 | -71 |
| glyph16_U0029 | +128 | -216 |
| glyph16_U0060 | -48 | -40 |
| glyph16_U0077 | -48 | -40 |
| glyph32_U0029 | +128 | -216 |
| glyph32_U0060 | -48 | -40 |
| glyph32_U0077 | -48 | -40 |
| shader_domain_warp_fbm | -64 | -26 |
| glyph16_U0031 | -64 | -38 |
| glyph32_U0031 | -64 | -38 |
| glyph16_U0037 | -64 | -50 |
| glyph32_U0037 | -64 | -50 |
| glyph16_U004E | -80 | -50 |
| glyph32_U004E | -80 | -50 |
| glyph16_U0076 | -32 | -100 |
| glyph32_U0076 | -32 | -100 |
| glyph16_U005A | -64 | -75 |
| glyph32_U005A | -64 | -75 |
| glyph16_U0028 | +128 | -279 |
| glyph32_U0028 | +128 | -279 |
| glyph16_U006C | -208 | +54 |
| glyph32_U006C | -208 | +54 |
| glyph16_U003B | -80 | -79 |
| glyph32_U003B | -80 | -79 |
| glyph16_U0021 | -112 | -87 |
| glyph32_U0021 | -112 | -87 |
| glyph16_U0056 | -144 | -135 |
| glyph32_U0056 | -144 | -135 |
| glyph16_U0075 | +112 | -448 |
| glyph32_U0075 | +112 | -448 |
| glyph16_U0079 | -112 | -292 |
| glyph32_U0079 | -112 | -292 |
| glyph16_U004A | +0 | -450 |
| glyph32_U004A | +0 | -450 |
| glyph16_U0044 | -96 | -409 |
| glyph32_U0044 | -96 | -409 |
| glyph16_U0050 | -96 | -423 |
| glyph32_U0050 | -96 | -423 |
| glyph16_U0072 | -32 | -534 |
| glyph32_U0072 | -32 | -534 |
| glyph16_U0035 | +224 | -811 |
| glyph32_U0035 | +224 | -811 |
| glyph16_U0068 | -64 | -541 |
| glyph32_U0068 | -64 | -541 |
| glyph16_U006E | -176 | -538 |
| glyph32_U006E | -176 | -538 |
| glyph16_U0065 | +96 | -811 |
| glyph32_U0065 | +96 | -811 |
| glyph16_U0064 | +48 | -862 |
| glyph32_U0064 | +48 | -862 |
| glyph16_U007E | +112 | -945 |
| glyph32_U007E | +112 | -945 |
| glyph16_U006D | +0 | -864 |
| glyph32_U006D | +0 | -864 |
| glyph16_U0070 | +16 | -888 |
| glyph32_U0070 | +16 | -888 |
| glyph16_U004F | -48 | -859 |
| glyph32_U004F | -48 | -859 |
| glyph16_U003F | -144 | -765 |
| glyph32_U003F | -144 | -765 |
| glyph16_U0042 | -96 | -914 |
| glyph32_U0042 | -96 | -914 |
| glyph16_U0032 | -80 | -969 |
| glyph32_U0032 | -80 | -969 |
| glyph16_U0055 | -48 | -1023 |
| glyph32_U0055 | -48 | -1023 |
| glyph16_U006F | -192 | -956 |
| glyph32_U006F | -192 | -956 |
| glyph16_U0063 | -288 | -889 |
| glyph32_U0063 | -288 | -889 |
| glyph16_U0025 | -1280 | +79 |
| glyph32_U0025 | -1280 | +79 |
| glyph16_U0043 | -304 | -910 |
| glyph32_U0043 | -304 | -910 |
| glyph16_U0071 | -384 | -861 |
| glyph32_U0071 | -384 | -861 |
| glyph16_U0062 | -400 | -909 |
| glyph32_U0062 | -400 | -909 |
| glyph16_U0047 | -496 | -814 |
| glyph32_U0047 | -496 | -814 |
| glyph16_U0052 | -464 | -1388 |
| glyph32_U0052 | -464 | -1388 |
| glyph16_U007D | -1408 | -872 |
| glyph32_U007D | -1408 | -872 |
| glyph16_U007B | -1392 | -903 |
| glyph32_U007B | -1392 | -903 |

### cap20000-app200000: 110 better, 55 worse, 41 unchanged

| kernel | Δ bytes | Δ dag_cost |
|---|---:|---:|
| glyph16_U006C | +80 | +399 |
| glyph32_U006C | +80 | +399 |
| glyph16_U0066 | +48 | +337 |
| glyph32_U0066 | +48 | +337 |
| glyph16_U0044 | +320 | -22 |
| glyph32_U0044 | +320 | -22 |
| shader_mandelbrot_distance | +208 | +77 |
| glyph16_U002A | +320 | -47 |
| glyph32_U002A | +320 | -47 |
| glyph16_U0023 | +288 | -38 |
| glyph32_U0023 | +288 | -38 |
| chrome_packed | +160 | +13 |
| chrome_R | +176 | -27 |
| glyph16_U0074 | -112 | +137 |
| glyph32_U0074 | -112 | +137 |
| shader_domain_warp_fbm | +16 | -11 |
| glyph16_U007C | +0 | -3 |
| glyph32_U007C | +0 | -3 |
| psychedelic_packed | -16 | +4 |
| shader_smooth_min_scene | -16 | +0 |
| cellgrid_80x24_d2 | -16 | -1 |
| shader_julia_set | +0 | -20 |
| shader_metaballs | -16 | -19 |
| glyph16_U0069 | -32 | -5 |
| glyph32_U0069 | -32 | -5 |
| glyph16_U006A | -176 | +128 |
| glyph32_U006A | -176 | +128 |
| glyph16_U004B | +80 | -137 |
| glyph16_U006B | +80 | -137 |
| glyph32_U004B | +80 | -137 |
| glyph32_U006B | +80 | -137 |
| glyph16_U003E | +48 | -109 |
| glyph32_U003E | +48 | -109 |
| glyph16_U002C | -32 | -38 |
| glyph32_U002C | -32 | -38 |
| glyph16_U0058 | +128 | -204 |
| glyph32_U0058 | +128 | -204 |
| glyph16_U003C | +32 | -109 |
| glyph32_U003C | +32 | -109 |
| glyph16_U002F | -48 | -38 |
| glyph16_U005C | -48 | -38 |
| glyph32_U002F | -48 | -38 |
| glyph32_U005C | -48 | -38 |
| glyph16_U0060 | -48 | -40 |
| glyph32_U0060 | -48 | -40 |
| glyph16_U004D | +0 | -121 |
| glyph32_U004D | +0 | -121 |
| glyph16_U007A | -48 | -79 |
| glyph32_U007A | -48 | -79 |
| glyph16_U0078 | +96 | -225 |
| glyph32_U0078 | +96 | -225 |
| glyph16_U005E | -48 | -92 |
| glyph32_U005E | -48 | -92 |
| glyph16_U0034 | -64 | -82 |
| glyph32_U0034 | -64 | -82 |
| glyph16_U004A | +0 | -150 |
| glyph16_U004E | -96 | -54 |
| glyph32_U004E | -96 | -54 |
| glyph16_U0037 | -96 | -58 |
| glyph32_U0037 | -96 | -58 |
| glyph16_U003B | -80 | -79 |
| glyph32_U003B | -80 | -79 |
| glyph16_U0059 | -32 | -129 |
| glyph32_U0059 | -32 | -129 |
| glyph32_U004A | -16 | -150 |
| glyph16_U0057 | +32 | -204 |
| glyph32_U0057 | +32 | -204 |
| glyph16_U005A | -96 | -83 |
| glyph32_U005A | -96 | -83 |
| glyph16_U0076 | -64 | -116 |
| glyph32_U0076 | -64 | -116 |
| glyph16_U0031 | -112 | -86 |
| glyph32_U0031 | -112 | -86 |
| glyph16_U0050 | +32 | -276 |
| glyph32_U0050 | +32 | -276 |
| glyph16_U0077 | -48 | -208 |
| glyph32_U0077 | -48 | -208 |
| glyph16_U0056 | -176 | -147 |
| glyph32_U0056 | -176 | -147 |
| glyph16_U0075 | -48 | -309 |
| glyph32_U0075 | -48 | -309 |
| glyph16_U0021 | -256 | -106 |
| glyph32_U0021 | -256 | -106 |
| glyph16_U0041 | -224 | -230 |
| glyph32_U0041 | -224 | -230 |
| glyph16_U0072 | -128 | -430 |
| glyph32_U0072 | -128 | -430 |
| glyph16_U0068 | -240 | -411 |
| glyph32_U0068 | -240 | -411 |
| glyph16_U0040 | +752 | -1450 |
| glyph32_U0040 | +736 | -1459 |
| glyph16_U006E | -320 | -447 |
| glyph32_U006E | -320 | -447 |
| glyph16_U0029 | -784 | -66 |
| glyph32_U0029 | -784 | -66 |
| glyph16_U0039 | +1072 | -2011 |
| glyph32_U0039 | +1072 | -2011 |
| glyph16_U0028 | -912 | -132 |
| glyph32_U0028 | -912 | -132 |
| glyph16_U0053 | +976 | -2087 |
| glyph32_U0053 | +976 | -2087 |
| glyph16_U0033 | +832 | -1969 |
| glyph16_U0073 | +864 | -2001 |
| glyph32_U0033 | +832 | -1969 |
| glyph32_U0073 | +864 | -2001 |
| glyph16_U0079 | -1008 | -179 |
| glyph32_U0079 | -1008 | -179 |
| glyph16_U0038 | +272 | -1521 |
| glyph32_U0038 | +272 | -1521 |
| glyph16_U0036 | +688 | -2020 |
| glyph32_U0036 | +688 | -2020 |
| glyph16_U0065 | +96 | -1536 |
| glyph32_U0065 | +96 | -1536 |
| glyph16_U0032 | +96 | -1598 |
| glyph32_U0032 | +96 | -1598 |
| glyph16_U0035 | +48 | -1553 |
| glyph32_U0035 | +48 | -1553 |
| glyph16_U0025 | -512 | -1009 |
| glyph32_U0025 | -512 | -1009 |
| glyph16_U0043 | +0 | -1543 |
| glyph32_U0043 | +0 | -1543 |
| glyph16_U0042 | +0 | -1565 |
| glyph32_U0042 | +0 | -1565 |
| glyph16_U0064 | -96 | -1541 |
| glyph32_U0064 | -96 | -1541 |
| glyph16_U0055 | -96 | -1547 |
| glyph32_U0055 | -96 | -1547 |
| glyph16_U0063 | -144 | -1544 |
| glyph32_U0063 | -144 | -1544 |
| glyph16_U0070 | -160 | -1540 |
| glyph16_U007E | -128 | -1572 |
| glyph32_U0070 | -160 | -1540 |
| glyph32_U007E | -128 | -1572 |
| glyph16_U0062 | -256 | -1512 |
| glyph32_U0062 | -256 | -1512 |
| glyph16_U0071 | -320 | -1462 |
| glyph32_U0071 | -320 | -1462 |
| glyph16_U0051 | -288 | -1558 |
| glyph32_U0051 | -288 | -1558 |
| glyph16_U006D | -352 | -1545 |
| glyph32_U006D | -352 | -1545 |
| glyph16_U0047 | -480 | -1472 |
| glyph32_U0047 | -480 | -1472 |
| glyph16_U003F | -400 | -1573 |
| glyph32_U003F | -400 | -1573 |
| glyph16_U006F | -640 | -1411 |
| glyph32_U006F | -640 | -1411 |
| glyph16_U0061 | -288 | -1914 |
| glyph32_U0061 | -288 | -1914 |
| glyph16_U0030 | -320 | -1896 |
| glyph32_U0030 | -320 | -1896 |
| glyph16_U0024 | +0 | -2271 |
| glyph32_U0024 | +0 | -2271 |
| glyph16_U004F | -768 | -1524 |
| glyph32_U004F | -768 | -1524 |
| glyph16_U0026 | -736 | -1694 |
| glyph32_U0026 | -736 | -1694 |
| glyph16_U007B | -1040 | -1592 |
| glyph32_U007B | -1040 | -1592 |
| glyph16_U0067 | -592 | -2055 |
| glyph32_U0067 | -592 | -2055 |
| glyph16_U0052 | -1264 | -1501 |
| glyph32_U0052 | -1264 | -1501 |
| glyph16_U007D | -1376 | -1588 |
| glyph32_U007D | -1376 | -1588 |

### cap20000-app800000: 110 better, 55 worse, 41 unchanged

| kernel | Δ bytes | Δ dag_cost |
|---|---:|---:|
| glyph16_U006C | +80 | +399 |
| glyph32_U006C | +80 | +399 |
| glyph16_U0066 | +48 | +337 |
| glyph32_U0066 | +48 | +337 |
| glyph16_U0044 | +320 | -22 |
| glyph32_U0044 | +320 | -22 |
| shader_mandelbrot_distance | +208 | +77 |
| glyph16_U002A | +320 | -47 |
| glyph32_U002A | +320 | -47 |
| glyph16_U0023 | +288 | -38 |
| glyph32_U0023 | +288 | -38 |
| chrome_packed | +160 | +13 |
| chrome_R | +176 | -27 |
| glyph16_U0074 | -112 | +137 |
| glyph32_U0074 | -112 | +137 |
| shader_domain_warp_fbm | +16 | -11 |
| glyph16_U007C | +0 | -3 |
| glyph32_U007C | +0 | -3 |
| psychedelic_packed | -16 | +4 |
| shader_smooth_min_scene | -16 | +0 |
| cellgrid_80x24_d2 | -16 | -1 |
| shader_julia_set | +0 | -20 |
| shader_metaballs | -16 | -19 |
| glyph16_U0069 | -32 | -5 |
| glyph32_U0069 | -32 | -5 |
| glyph16_U006A | -176 | +128 |
| glyph32_U006A | -176 | +128 |
| glyph16_U004B | +80 | -137 |
| glyph16_U006B | +80 | -137 |
| glyph32_U004B | +80 | -137 |
| glyph32_U006B | +80 | -137 |
| glyph16_U003E | +48 | -109 |
| glyph32_U003E | +48 | -109 |
| glyph16_U002C | -32 | -38 |
| glyph32_U002C | -32 | -38 |
| glyph16_U0058 | +128 | -204 |
| glyph32_U0058 | +128 | -204 |
| glyph16_U003C | +32 | -109 |
| glyph32_U003C | +32 | -109 |
| glyph16_U002F | -48 | -38 |
| glyph16_U005C | -48 | -38 |
| glyph32_U002F | -48 | -38 |
| glyph32_U005C | -48 | -38 |
| glyph16_U0060 | -48 | -40 |
| glyph32_U0060 | -48 | -40 |
| glyph16_U004D | +0 | -121 |
| glyph32_U004D | +0 | -121 |
| glyph16_U007A | -48 | -79 |
| glyph32_U007A | -48 | -79 |
| glyph16_U0078 | +96 | -225 |
| glyph32_U0078 | +96 | -225 |
| glyph16_U005E | -48 | -92 |
| glyph32_U005E | -48 | -92 |
| glyph16_U0034 | -64 | -82 |
| glyph32_U0034 | -64 | -82 |
| glyph16_U004A | +0 | -150 |
| glyph16_U004E | -96 | -54 |
| glyph32_U004E | -96 | -54 |
| glyph16_U0037 | -96 | -58 |
| glyph32_U0037 | -96 | -58 |
| glyph16_U003B | -80 | -79 |
| glyph32_U003B | -80 | -79 |
| glyph16_U0059 | -32 | -129 |
| glyph32_U0059 | -32 | -129 |
| glyph32_U004A | -16 | -150 |
| glyph16_U0057 | +32 | -204 |
| glyph32_U0057 | +32 | -204 |
| glyph16_U005A | -96 | -83 |
| glyph32_U005A | -96 | -83 |
| glyph16_U0076 | -64 | -116 |
| glyph32_U0076 | -64 | -116 |
| glyph16_U0031 | -112 | -86 |
| glyph32_U0031 | -112 | -86 |
| glyph16_U0050 | +32 | -276 |
| glyph32_U0050 | +32 | -276 |
| glyph16_U0077 | -48 | -208 |
| glyph32_U0077 | -48 | -208 |
| glyph16_U0056 | -176 | -147 |
| glyph32_U0056 | -176 | -147 |
| glyph16_U0075 | -48 | -309 |
| glyph32_U0075 | -48 | -309 |
| glyph16_U0021 | -256 | -106 |
| glyph32_U0021 | -256 | -106 |
| glyph16_U0041 | -224 | -230 |
| glyph32_U0041 | -224 | -230 |
| glyph16_U0072 | -128 | -430 |
| glyph32_U0072 | -128 | -430 |
| glyph16_U0068 | -240 | -411 |
| glyph32_U0068 | -240 | -411 |
| glyph16_U0040 | +752 | -1450 |
| glyph32_U0040 | +736 | -1459 |
| glyph16_U006E | -320 | -447 |
| glyph32_U006E | -320 | -447 |
| glyph16_U0029 | -784 | -66 |
| glyph32_U0029 | -784 | -66 |
| glyph16_U0039 | +1072 | -2011 |
| glyph32_U0039 | +1072 | -2011 |
| glyph16_U0028 | -912 | -132 |
| glyph32_U0028 | -912 | -132 |
| glyph16_U0053 | +976 | -2087 |
| glyph32_U0053 | +976 | -2087 |
| glyph16_U0033 | +832 | -1969 |
| glyph16_U0073 | +864 | -2001 |
| glyph32_U0033 | +832 | -1969 |
| glyph32_U0073 | +864 | -2001 |
| glyph16_U0079 | -1008 | -179 |
| glyph32_U0079 | -1008 | -179 |
| glyph16_U0038 | +272 | -1521 |
| glyph32_U0038 | +272 | -1521 |
| glyph16_U0036 | +688 | -2020 |
| glyph32_U0036 | +688 | -2020 |
| glyph16_U0065 | +96 | -1536 |
| glyph32_U0065 | +96 | -1536 |
| glyph16_U0032 | +96 | -1598 |
| glyph32_U0032 | +96 | -1598 |
| glyph16_U0035 | +48 | -1553 |
| glyph32_U0035 | +48 | -1553 |
| glyph16_U0025 | -512 | -1009 |
| glyph32_U0025 | -512 | -1009 |
| glyph16_U0043 | +0 | -1543 |
| glyph32_U0043 | +0 | -1543 |
| glyph16_U0042 | +0 | -1565 |
| glyph32_U0042 | +0 | -1565 |
| glyph16_U0064 | -96 | -1541 |
| glyph32_U0064 | -96 | -1541 |
| glyph16_U0055 | -96 | -1547 |
| glyph32_U0055 | -96 | -1547 |
| glyph16_U0063 | -144 | -1544 |
| glyph32_U0063 | -144 | -1544 |
| glyph16_U0070 | -160 | -1540 |
| glyph16_U007E | -128 | -1572 |
| glyph32_U0070 | -160 | -1540 |
| glyph32_U007E | -128 | -1572 |
| glyph16_U0062 | -256 | -1512 |
| glyph32_U0062 | -256 | -1512 |
| glyph16_U0071 | -320 | -1462 |
| glyph32_U0071 | -320 | -1462 |
| glyph16_U0051 | -288 | -1558 |
| glyph32_U0051 | -288 | -1558 |
| glyph16_U006D | -352 | -1545 |
| glyph32_U006D | -352 | -1545 |
| glyph16_U0047 | -480 | -1472 |
| glyph32_U0047 | -480 | -1472 |
| glyph16_U003F | -400 | -1573 |
| glyph32_U003F | -400 | -1573 |
| glyph16_U006F | -640 | -1411 |
| glyph32_U006F | -640 | -1411 |
| glyph16_U0061 | -288 | -1914 |
| glyph32_U0061 | -288 | -1914 |
| glyph16_U0030 | -320 | -1896 |
| glyph32_U0030 | -320 | -1896 |
| glyph16_U0024 | +0 | -2271 |
| glyph32_U0024 | +0 | -2271 |
| glyph16_U004F | -768 | -1524 |
| glyph32_U004F | -768 | -1524 |
| glyph16_U0026 | -736 | -1694 |
| glyph32_U0026 | -736 | -1694 |
| glyph16_U007B | -1040 | -1592 |
| glyph32_U007B | -1040 | -1592 |
| glyph16_U0067 | -592 | -2055 |
| glyph32_U0067 | -592 | -2055 |
| glyph16_U0052 | -1264 | -1501 |
| glyph32_U0052 | -1264 | -1501 |
| glyph16_U007D | -1376 | -1588 |
| glyph32_U007D | -1376 | -1588 |

### cap50000-app200000: 110 better, 52 worse, 44 unchanged

| kernel | Δ bytes | Δ dag_cost |
|---|---:|---:|
| glyph16_U006A | +1264 | +704 |
| glyph32_U006A | +1264 | +704 |
| glyph16_U0068 | +880 | +540 |
| glyph32_U0068 | +880 | +540 |
| glyph16_U0072 | +720 | +669 |
| glyph32_U0072 | +720 | +669 |
| glyph16_U0066 | +704 | +679 |
| glyph32_U0066 | +704 | +679 |
| glyph16_U0029 | +624 | +709 |
| glyph32_U0029 | +624 | +709 |
| glyph16_U0050 | +528 | +707 |
| glyph32_U0050 | +528 | +707 |
| glyph16_U0075 | +592 | +612 |
| glyph32_U0075 | +592 | +612 |
| chrome_R | +592 | +584 |
| chrome_packed | +576 | +546 |
| glyph16_U0028 | +480 | +641 |
| glyph32_U0028 | +480 | +641 |
| glyph16_U0044 | +432 | +619 |
| glyph32_U0044 | +432 | +619 |
| glyph16_U004A | +224 | +574 |
| glyph32_U004A | +224 | +572 |
| glyph32_U0074 | +368 | +311 |
| glyph16_U0074 | +368 | +306 |
| glyph16_U006E | +208 | +455 |
| glyph32_U006E | +208 | +455 |
| shader_julia_set | +304 | +187 |
| glyph16_U006C | +0 | +471 |
| glyph32_U006C | +0 | +471 |
| glyph16_U0079 | +240 | +120 |
| glyph32_U0079 | +240 | +120 |
| shader_mandelbrot_distance | +208 | +77 |
| shader_metaballs | +144 | +100 |
| psychedelic_packed | -16 | +4 |
| glyph16_U002A | +224 | -237 |
| glyph32_U002A | +224 | -237 |
| cellgrid_80x24_d2 | -16 | -1 |
| glyph16_U0069 | -32 | -5 |
| glyph32_U0069 | -32 | -5 |
| shader_domain_warp_fbm | -32 | -18 |
| glyph16_U003E | +48 | -109 |
| glyph32_U003E | +48 | -109 |
| glyph16_U002C | -32 | -38 |
| glyph32_U002C | -32 | -38 |
| glyph16_U003C | +32 | -109 |
| glyph32_U003C | +32 | -109 |
| glyph16_U002F | -48 | -38 |
| glyph16_U005C | -48 | -38 |
| glyph32_U002F | -48 | -38 |
| glyph32_U005C | -48 | -38 |
| glyph16_U0023 | +144 | -237 |
| glyph32_U0023 | +144 | -237 |
| glyph16_U004B | +48 | -157 |
| glyph32_U004B | +48 | -157 |
| glyph16_U0060 | -64 | -48 |
| glyph32_U0060 | -64 | -48 |
| glyph16_U006B | +32 | -157 |
| glyph32_U006B | +32 | -157 |
| glyph16_U007A | -48 | -79 |
| glyph32_U007A | -48 | -79 |
| glyph16_U0059 | -16 | -124 |
| glyph32_U0059 | -16 | -124 |
| glyph16_U0034 | -64 | -82 |
| glyph32_U0034 | -64 | -82 |
| glyph16_U004E | -96 | -54 |
| glyph32_U004E | -96 | -54 |
| glyph16_U0037 | -96 | -58 |
| glyph32_U0037 | -96 | -58 |
| glyph16_U0058 | +80 | -236 |
| glyph32_U0058 | +80 | -236 |
| glyph16_U003B | -80 | -79 |
| glyph32_U003B | -80 | -79 |
| glyph16_U004D | -32 | -137 |
| glyph32_U004D | -32 | -137 |
| glyph16_U005A | -96 | -83 |
| glyph32_U005A | -96 | -83 |
| glyph16_U0076 | -64 | -116 |
| glyph32_U0076 | -64 | -116 |
| glyph16_U0031 | -112 | -86 |
| glyph32_U0031 | -112 | -86 |
| glyph16_U005E | -96 | -108 |
| glyph32_U005E | -96 | -108 |
| glyph16_U0078 | +48 | -253 |
| glyph32_U0078 | +48 | -253 |
| glyph16_U0056 | -176 | -147 |
| glyph32_U0056 | -176 | -147 |
| glyph16_U0077 | -96 | -232 |
| glyph32_U0077 | -96 | -232 |
| glyph16_U0057 | -96 | -258 |
| glyph32_U0057 | -96 | -258 |
| glyph16_U0021 | -256 | -106 |
| glyph32_U0021 | -256 | -106 |
| glyph16_U0041 | -224 | -230 |
| glyph32_U0041 | -224 | -230 |
| glyph16_U0039 | -1072 | -790 |
| glyph32_U0039 | -1072 | -790 |
| glyph16_U0071 | -1456 | -811 |
| glyph32_U0071 | -1456 | -811 |
| glyph16_U0052 | -1344 | -966 |
| glyph32_U0052 | -1344 | -966 |
| glyph16_U0062 | -1520 | -888 |
| glyph32_U0062 | -1520 | -888 |
| glyph16_U0025 | +416 | -2834 |
| glyph32_U0025 | +416 | -2834 |
| glyph16_U0033 | -1680 | -946 |
| glyph32_U0033 | -1680 | -946 |
| glyph16_U0051 | -1712 | -975 |
| glyph32_U0051 | -1712 | -975 |
| glyph16_U0061 | -2208 | -673 |
| glyph32_U0061 | -2208 | -673 |
| glyph16_U0055 | -1824 | -1067 |
| glyph32_U0055 | -1824 | -1067 |
| glyph16_U0042 | -1616 | -1343 |
| glyph32_U0042 | -1616 | -1343 |
| glyph16_U004F | -1888 | -1175 |
| glyph32_U004F | -1888 | -1175 |
| glyph16_U0063 | -1936 | -1144 |
| glyph32_U0063 | -1936 | -1144 |
| glyph16_U0035 | -1856 | -1290 |
| glyph32_U0035 | -1856 | -1290 |
| glyph16_U006D | -2240 | -919 |
| glyph32_U006D | -2240 | -919 |
| glyph16_U0043 | -2016 | -1173 |
| glyph32_U0043 | -2016 | -1173 |
| glyph16_U0047 | -2064 | -1186 |
| glyph32_U0047 | -2064 | -1186 |
| glyph16_U0040 | -736 | -2559 |
| glyph32_U0040 | -752 | -2565 |
| glyph16_U0064 | -1952 | -1436 |
| glyph32_U0064 | -1952 | -1436 |
| glyph16_U0065 | -1936 | -1497 |
| glyph32_U0065 | -1936 | -1497 |
| glyph16_U006F | -2400 | -1164 |
| glyph32_U006F | -2400 | -1164 |
| glyph16_U0032 | -2144 | -1432 |
| glyph32_U0032 | -2144 | -1432 |
| glyph16_U0030 | -2608 | -1010 |
| glyph32_U0030 | -2608 | -1010 |
| glyph16_U003F | -2784 | -1041 |
| glyph32_U003F | -2784 | -1041 |
| glyph16_U0038 | -1312 | -2524 |
| glyph32_U0038 | -1312 | -2524 |
| glyph16_U0073 | -2416 | -1470 |
| glyph32_U0073 | -2416 | -1470 |
| glyph16_U007D | -2768 | -1157 |
| glyph32_U007D | -2768 | -1157 |
| glyph16_U0026 | -912 | -3014 |
| glyph32_U0026 | -912 | -3014 |
| glyph16_U0070 | -2480 | -1471 |
| glyph32_U0070 | -2480 | -1471 |
| glyph16_U007B | -2784 | -1246 |
| glyph32_U007B | -2784 | -1246 |
| glyph16_U007E | -2480 | -1622 |
| glyph32_U007E | -2480 | -1622 |
| glyph16_U0036 | -2624 | -1502 |
| glyph32_U0036 | -2624 | -1502 |
| glyph16_U0067 | -3088 | -1223 |
| glyph32_U0067 | -3088 | -1223 |
| glyph16_U0053 | -2832 | -1636 |
| glyph32_U0053 | -2832 | -1636 |
| glyph16_U0024 | -3632 | -1546 |
| glyph32_U0024 | -3632 | -1546 |

### cap50000-app2000000: 110 better, 53 worse, 43 unchanged

| kernel | Δ bytes | Δ dag_cost |
|---|---:|---:|
| glyph16_U006A | +1264 | +704 |
| glyph32_U006A | +1264 | +704 |
| glyph16_U0068 | +880 | +540 |
| glyph32_U0068 | +880 | +540 |
| glyph16_U0072 | +720 | +669 |
| glyph32_U0072 | +720 | +669 |
| glyph16_U0066 | +704 | +679 |
| glyph32_U0066 | +704 | +679 |
| glyph16_U0029 | +624 | +709 |
| glyph32_U0029 | +624 | +709 |
| glyph16_U0050 | +528 | +707 |
| glyph32_U0050 | +528 | +707 |
| glyph16_U0075 | +592 | +612 |
| glyph32_U0075 | +592 | +612 |
| chrome_R | +592 | +584 |
| chrome_packed | +576 | +546 |
| glyph16_U0028 | +480 | +641 |
| glyph32_U0028 | +480 | +641 |
| glyph16_U0044 | +432 | +619 |
| glyph32_U0044 | +432 | +619 |
| glyph16_U004A | +224 | +574 |
| glyph32_U004A | +224 | +572 |
| glyph32_U0074 | +368 | +311 |
| glyph16_U0074 | +368 | +306 |
| glyph16_U006E | +208 | +455 |
| glyph32_U006E | +208 | +455 |
| shader_julia_set | +304 | +187 |
| glyph16_U006C | +0 | +471 |
| glyph32_U006C | +0 | +471 |
| glyph16_U0079 | +240 | +120 |
| glyph32_U0079 | +240 | +120 |
| shader_mandelbrot_distance | +208 | +77 |
| shader_metaballs | +144 | +100 |
| shader_smooth_min_scene | +0 | +12 |
| psychedelic_packed | -16 | +4 |
| glyph16_U002A | +224 | -237 |
| glyph32_U002A | +224 | -237 |
| cellgrid_80x24_d2 | -16 | -1 |
| glyph16_U0069 | -32 | -5 |
| glyph32_U0069 | -32 | -5 |
| shader_domain_warp_fbm | -32 | -18 |
| glyph16_U003E | +48 | -109 |
| glyph32_U003E | +48 | -109 |
| glyph16_U002C | -32 | -38 |
| glyph32_U002C | -32 | -38 |
| glyph16_U003C | +32 | -109 |
| glyph32_U003C | +32 | -109 |
| glyph16_U002F | -48 | -38 |
| glyph16_U005C | -48 | -38 |
| glyph32_U002F | -48 | -38 |
| glyph32_U005C | -48 | -38 |
| glyph16_U0023 | +144 | -237 |
| glyph32_U0023 | +144 | -237 |
| glyph16_U004B | +48 | -157 |
| glyph32_U004B | +48 | -157 |
| glyph16_U0060 | -64 | -48 |
| glyph32_U0060 | -64 | -48 |
| glyph16_U006B | +32 | -157 |
| glyph32_U006B | +32 | -157 |
| glyph16_U007A | -48 | -79 |
| glyph32_U007A | -48 | -79 |
| glyph16_U0059 | -16 | -124 |
| glyph32_U0059 | -16 | -124 |
| glyph16_U0034 | -64 | -82 |
| glyph32_U0034 | -64 | -82 |
| glyph16_U004E | -96 | -54 |
| glyph32_U004E | -96 | -54 |
| glyph16_U0037 | -96 | -58 |
| glyph32_U0037 | -96 | -58 |
| glyph16_U0058 | +80 | -236 |
| glyph32_U0058 | +80 | -236 |
| glyph16_U003B | -80 | -79 |
| glyph32_U003B | -80 | -79 |
| glyph16_U004D | -32 | -137 |
| glyph32_U004D | -32 | -137 |
| glyph16_U005A | -96 | -83 |
| glyph32_U005A | -96 | -83 |
| glyph16_U0076 | -64 | -116 |
| glyph32_U0076 | -64 | -116 |
| glyph16_U0031 | -112 | -86 |
| glyph32_U0031 | -112 | -86 |
| glyph16_U005E | -96 | -108 |
| glyph32_U005E | -96 | -108 |
| glyph16_U0078 | +48 | -253 |
| glyph32_U0078 | +48 | -253 |
| glyph16_U0056 | -176 | -147 |
| glyph32_U0056 | -176 | -147 |
| glyph16_U0077 | -96 | -232 |
| glyph32_U0077 | -96 | -232 |
| glyph16_U0057 | -96 | -258 |
| glyph32_U0057 | -96 | -258 |
| glyph16_U0021 | -256 | -106 |
| glyph32_U0021 | -256 | -106 |
| glyph16_U0041 | -224 | -230 |
| glyph32_U0041 | -224 | -230 |
| glyph16_U0039 | -1072 | -790 |
| glyph32_U0039 | -1072 | -790 |
| glyph16_U0071 | -1456 | -811 |
| glyph32_U0071 | -1456 | -811 |
| glyph16_U0052 | -1344 | -966 |
| glyph32_U0052 | -1344 | -966 |
| glyph16_U0062 | -1520 | -888 |
| glyph32_U0062 | -1520 | -888 |
| glyph16_U0025 | +416 | -2834 |
| glyph32_U0025 | +416 | -2834 |
| glyph16_U0033 | -1680 | -946 |
| glyph32_U0033 | -1680 | -946 |
| glyph16_U0051 | -1712 | -975 |
| glyph32_U0051 | -1712 | -975 |
| glyph16_U0061 | -2208 | -673 |
| glyph32_U0061 | -2208 | -673 |
| glyph16_U0055 | -1824 | -1067 |
| glyph32_U0055 | -1824 | -1067 |
| glyph16_U0042 | -1616 | -1343 |
| glyph32_U0042 | -1616 | -1343 |
| glyph16_U004F | -1888 | -1175 |
| glyph32_U004F | -1888 | -1175 |
| glyph16_U0063 | -1936 | -1144 |
| glyph32_U0063 | -1936 | -1144 |
| glyph16_U0035 | -1856 | -1290 |
| glyph32_U0035 | -1856 | -1290 |
| glyph16_U006D | -2240 | -919 |
| glyph32_U006D | -2240 | -919 |
| glyph16_U0043 | -2016 | -1173 |
| glyph32_U0043 | -2016 | -1173 |
| glyph16_U0047 | -2064 | -1186 |
| glyph32_U0047 | -2064 | -1186 |
| glyph16_U0040 | -736 | -2559 |
| glyph32_U0040 | -752 | -2565 |
| glyph16_U0064 | -1952 | -1436 |
| glyph32_U0064 | -1952 | -1436 |
| glyph16_U0065 | -1936 | -1497 |
| glyph32_U0065 | -1936 | -1497 |
| glyph16_U006F | -2400 | -1164 |
| glyph32_U006F | -2400 | -1164 |
| glyph16_U0032 | -2144 | -1432 |
| glyph32_U0032 | -2144 | -1432 |
| glyph16_U0030 | -2608 | -1010 |
| glyph32_U0030 | -2608 | -1010 |
| glyph16_U003F | -2784 | -1041 |
| glyph32_U003F | -2784 | -1041 |
| glyph16_U0038 | -1312 | -2524 |
| glyph32_U0038 | -1312 | -2524 |
| glyph16_U0073 | -2416 | -1470 |
| glyph32_U0073 | -2416 | -1470 |
| glyph16_U007D | -2768 | -1157 |
| glyph32_U007D | -2768 | -1157 |
| glyph16_U0026 | -912 | -3014 |
| glyph32_U0026 | -912 | -3014 |
| glyph16_U0070 | -2480 | -1471 |
| glyph32_U0070 | -2480 | -1471 |
| glyph16_U007B | -2784 | -1246 |
| glyph32_U007B | -2784 | -1246 |
| glyph16_U007E | -2480 | -1622 |
| glyph32_U007E | -2480 | -1622 |
| glyph16_U0036 | -2624 | -1502 |
| glyph32_U0036 | -2624 | -1502 |
| glyph16_U0067 | -3088 | -1223 |
| glyph32_U0067 | -3088 | -1223 |
| glyph16_U0053 | -2832 | -1636 |
| glyph32_U0053 | -2832 | -1636 |
| glyph16_U0024 | -3632 | -1546 |
| glyph32_U0024 | -3632 | -1546 |

### cap100000-app200000: 2 better, 6 worse, 8 unchanged

| kernel | Δ bytes | Δ dag_cost |
|---|---:|---:|
| chrome_packed | +1344 | +706 |
| chrome_R | +816 | +442 |
| shader_julia_set | +304 | +231 |
| shader_metaballs | +224 | +93 |
| shader_mandelbrot_distance | +208 | +77 |
| psychedelic_packed | -16 | +4 |
| cellgrid_80x24_d2 | -16 | -1 |
| shader_domain_warp_fbm | -32 | -21 |

### cap100000-app4000000: 2 better, 7 worse, 7 unchanged

| kernel | Δ bytes | Δ dag_cost |
|---|---:|---:|
| chrome_packed | +1344 | +706 |
| chrome_R | +816 | +442 |
| shader_julia_set | +304 | +231 |
| shader_metaballs | +192 | +115 |
| shader_mandelbrot_distance | +208 | +77 |
| shader_smooth_min_scene | +0 | +12 |
| psychedelic_packed | -16 | +4 |
| cellgrid_80x24_d2 | -16 | -1 |
| shader_domain_warp_fbm | -32 | -21 |

