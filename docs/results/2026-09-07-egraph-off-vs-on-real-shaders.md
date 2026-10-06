# Saturation on vs off, on every shipped kernel

The "F: no e-graph" column of [`2026-09-06-egraph-at-production-scale.md`](../plans/2026-09-06-egraph-at-production-scale.md) §7, measured. Every kernel is compiled through the production path (`optimize_runtime_arena` → `relink` → `emit::compile`, the three calls `jit_cache::compile` makes; buffer-free kernels are asserted byte-identical to `Manifold::compile`) twice: `PIXELFLOW_SATURATION=off` runs the `Identity` path (`LowerDwrt`, `ExpandReduce`, no saturation), `on` is production. Two in-harness arms beside them: `cse-only` (the production optimizer at zero rewrite rounds — insert, extract — so hash-consing's share is separated from the rules') and `with-select-hoist` (production's rules plus the three `SelectHoistUnary` rules, which are **not** in `all_rules()`). Deterministic columns are the claim; the clock, when taken, is a sign. Per-kernel rows: the `.csv`/`.json` beside this file.

Host: aarch64 (NEON, 4 lanes), macOS, load average 8–43 throughout (other sessions); **no clock taken** on any class but glyph32 — every other column here is deterministic. The glyph32 clock was taken 2026-09-08 at load 6.3–7.1 (`uptime` 1-min average, below the 8 this doc's protocol requires), alternated `off`/`on`/`off`/`on`, one process per arm, 95 glyphs each, `--no-probe`, median of 7 samples per glyph; both rounds' `bytes`/`dag_cost` are identical to the deterministic rows (asserted by `diff` when the files are merged; the clocked rows are `…-rows/glyph32-clock-{off,on}{1,2}.jsonl`, merged after the deterministic ones so the census columns come from the latter).
**chrome (both kernels) has no `off` row**: with saturation off the emitter panics — `CBZ/CBNZ branch offset out of range (±1MB)` (`pixelflow-codegen/src/emit/aarch64.rs:849`) at a 247,789-entry body schedule (chrome_packed; chrome_R: 84,315). The constructed `Kernel` is a tree of 335,411 reachable nodes (`Kernel` composition does not hash-cons); the e-graph's insertion hash-consing takes it to 465 and the rules to 398. Without the e-graph the chrome scene does not compile on this target. *(2026-10-05: the limit is gone — a guard's branch is `cbnz w16, .+8; b target`, so its arm may be ±128 MiB — and this row is a record of the day it was not.)*
**glyph32, bench and the shader ports against the headlines (added 2026-09-08).** None of the three moves the first two findings: chrome is its own class, and the split stays hash-consing-first wherever the kernel is built by `Kernel` composition — glyph32 is −41.1% bytes / −49.6% dag_cost from hash-consing and −4.2% / −2.1% from the rules (its rows are byte-identical to glyph16's on 94 of 95 glyphs: the tile scales a constant, not the shape), bench −42.4% / −50.3% then −2.5% / +0.1%, bench_wide −45.6% / −52.0% then −1.0% / −1.9%. The twelve `shader_bench` ports are the exception that explains the rule: they arrive as an arena (`Kernel::from_parts`), already hash-consed, so insertion has almost nothing to do (−3.0% / −3.0%) and the rules are the whole effect (−0.5% bytes, −4.8% dag_cost) — the same few percent as on every other class, now without a 40% hash-consing win in front of it; "overwhelmingly hash-consing" is a fact about `Kernel` composition re-expanding shared subtrees, not about the rules being weak. The third finding does change: "no kernel regressed" was true of 196 kernels and is false of 208 — seven ports appear in the table below. Five grow in bytes while dag_cost falls (`star_sdf` 544 → 640, +17.6% / −11.8%; `smooth_min_scene` +3.8% / −5.0%; `domain_warp_fbm` +1.9% / −6.5%; `plasma` +1.2% / −5.0%; `kaleidoscope_fold` +0.8% / −1.2%) and two grow in dag_cost at unchanged bytes (`julia_set` 710 → 717, +1.0%; `smoothstep_vignette` 183 → 194, +6.0%). Every one has cse ≤ off, so the growth sits on the rules' side of the split, and it is small in absolute terms (≤ 112 bytes, ≤ 11 dag units). The two dag_cost losses are the extractor's rather than a rewrite's: the input form is still in the graph (`smoothstep_vignette` quiesced at 283 classes, uncapped), so a dearer DAG was chosen, not forced — `julia_set`'s on-extraction has more nodes than its input (123 vs 122) — and `fma-fusion` carries the strict credit on both. The fourth finding holds: `SelectHoistUnary` fires in 29 of 208 kernels now (`mandelbrot_distance` is the addition: 9 applications, 2 tight) and still changes 0 bytes. Two shader rows deserve a line of their own: `mandelbrot_distance` is the largest shader win (−32.3% bytes, −30.5% dag_cost, 2 spills → 0, 114,945 → 0 trip-weighted memory ops — the class's entire −99.3% mem-ops column), and `domain_warp_fbm` is the first kernel where the cross-form divergence is visible in pixels — same-form |Δ| exactly 0 in both modes, cross-form 0.406 (0.774 relative): the rewrites re-rounded the inside of a chaotic warp, which is the algebraic contract and not a JIT defect (6 of the 12 shader pictures are bit-identical off vs on). Provenance: the shader rows were taken after the rebase onto #1214 (`git_sha` 81da1f01; 9cd61117 on the rest), and the rebuilt binary reproduced the psychedelic and cellgrid rows column for column.
**The glyph32 clock (Σ over 95 glyphs, ns per bake call, 32×32 tile).** Round 1: `off` 625,112 → `on` 508,551 (−18.6%); round 2: 645,859 → 511,409 (−20.8%); per-kernel `on`/`off` ratio median 0.63–0.64, p10 0.48, p90 0.92–0.94, 90 of 95 glyphs faster with saturation in both rounds; per-glyph IQR/median 2.5–3.3%. The sign agrees with every static column (Σ bytes −43.6%, Σ dag_cost −50.6%, Σ mem ops −65.5%) and the magnitude is smaller than all of them, as a schedule-free cost model predicts: the clock counts what the out-of-order core overlaps, the columns count what was emitted.

## Verdict per shader class

`Δ` is on relative to off; negative is saturation winning. `mem ops` is the trip-weighted dynamic memory-op count (`dyn_memory_ops`). `cse` is the zero-round arm: the gap between `off` and `cse` is hash-consing, between `cse` and `on` is the rules.

| class | n | Σ bytes off → cse → on | Σ dag_cost off → cse → on | Σ mem ops off → on | median clock Δ | picture identical | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| bench | 3 | 73360 → 42256 → 41184 (-43.9%) | 37249 → 18513 → 18527 (-50.3%) | 891017 → 302199 (-66.1%) | - | 0/3 | helps (static) |
| bench_wide | 1 | 28208 → 15344 → 15184 (-46.2%) | 14365 → 6901 → 6770 (-52.9%) | 6242609 → 1751202 (-71.9%) | - | 0/1 | helps (static) |
| cellgrid | 1 | 8864 → 2512 → 2576 (-70.9%) | 1879 → 450 → 428 (-77.2%) | 42154099 → 5227011 (-87.6%) | - | 1/1 | helps (static) |
| glyph16 | 95 | 1500068 → 883060 → 846068 (-43.6%) | 745471 → 375881 → 368113 (-50.6%) | 2596166 → 970018 (-62.6%) | - | 1/95 | helps (static) |
| glyph32 | 95 | 1500068 → 883060 → 846100 (-43.6%) | 745471 → 375881 → 368113 (-50.6%) | 10359990 → 3570226 (-65.5%) | -36.0% | 1/95 | helps (bytes+clock) |
| psychedelic | 1 | 10496 → 2176 → 2368 (-77.4%) | 4948 → 860 → 825 (-83.3%) | 36333378 → 4324 (-100.0%) | - | 0/1 | helps (static) |
| shader | 12 | 15616 → 15152 → 15072 (-3.5%) | 5007 → 4859 → 4624 (-7.6%) | 115716 → 771 (-99.3%) | - | 6/12 | helps (static) |

## Kernels with no `off` row: the un-saturated arena does not compile

Run with `--skip` in `off` mode because the emitter panics on them (the notes above quote the panic). The columns are the `on` row and, where present, the `cse-only` arm.

| kernel | extent | nodes in → cse → on | dag_cost in → cse → on | bytes cse → on | schedule on | guarded/selects on | spills on | saturation (apps / rounds / classes / stop) | compile on (ms) |
|---|---|---|---:|---:|---:|---:|---:|---|---:|
| chrome_R | 1920×1080 | 114633 → 351 → 300 | 371499 → 1726 → 1418 | 2608 → 2640 | 288 | 433/9 | 23 | 5669 apps / 2 it / 4894 cls / ClassCap | 74.0 |
| chrome_packed | 1920×1080 | 335411 → 465 → 398 | 1090401 → 2031 → 1668 | 3456 → 3424 | 385 | 614/17 | 22 | 5531 apps / 2 it / 4940 cls / ClassCap | 222.9 |

## The headline kernels

| kernel | extent | nodes in → off → cse → on | bytes off → cse → on | dag_cost off → cse → on | schedule off → on | guarded/selects off → on | spills off → on | mem ops off → on | saturation (apps / rounds / classes / stop) | compile off → on (ms; saturation share) |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| bench_A_linear | 40×45 | 271 → 271 → 157 → 169 | 2128 → 1744 → 1968 (-7.5%) | 823 → 618 → 670 (-18.6%) | 191 → 71 | 242/7 → 77/7 | 0 → 0 | 0 → 3522 (+0.0%) | 8114 apps / 3 it / 3006 cls / ClassCap | 2.7 → 51.8 (95%) |
| bench_O_quadratic | 40×45 | 1799 → 3655 → 1536 → 1482 | 28208 → 15344 → 15184 (-46.2%) | 14365 → 6901 → 6770 (-52.9%) | 3503 → 401 | 9730/209 → 979/49 | 226 → 0 | 390359 → 117702 (-69.8%) | 3328 apps / 1 it / 4868 cls / ClassCap | 39.1 → 34.7 (55%) |
| bench_O_quadratic_wide | 640×45 | 1799 → 3655 → 1536 → 1482 | 28208 → 15344 → 15184 (-46.2%) | 14365 → 6901 → 6770 (-52.9%) | 3503 → 401 | 9730/209 → 979/49 | 226 → 0 | 6242609 → 1751202 (-71.9%) | 3328 apps / 1 it / 4868 cls / ClassCap | 38.7 → 33.6 (56%) |
| bench_S_complex | 40×45 | 2815 → 5623 → 2463 → 2453 | 43024 → 25168 → 24032 (-44.1%) | 22061 → 10994 → 11087 (-49.7%) | 5359 → 652 | 14850/317 → 1580/77 | 291 → 0 | 500658 → 180975 (-63.9%) | 2485 apps / 1 it / 5000 cls / ClassCap | 56.1 → 42.4 (52%) |
| cellgrid_80x24_d2 | 1600×768 | 667 → 667 → 123 → 117 | 8864 → 2512 → 2576 (-70.9%) | 1879 → 450 → 428 (-77.2%) | 779 → 217 | 0/0 → 0/0 | 27 → 7 | 42154099 → 5227011 (-87.6%) | 6841 apps / 3 it / 4641 cls / ClassCap | 1.6 → 25.2 (97%) |
| psychedelic_packed | 1920×1080 | 1246 → 1246 → 112 → 112 | 10496 → 2176 → 2368 (-77.4%) | 4948 → 860 → 825 (-83.3%) | 1072 → 127 | 174/6 → 29/1 | 11 → 0 | 36333378 → 4324 (-100.0%) | 7319 apps / 3 it / 4676 cls / ClassCap | 4.9 → 24.7 (96%) |
| shader_cosine_palette | 256×256 | 40 → 40 → 34 → 26 | 1264 → 1248 → 1248 (-1.3%) | 328 → 328 → 296 (-9.8%) | 146 → 133 | 99/3 → 99/3 | 0 → 0 | 0 → 0 (+0.0%) | 2614 apps / 3 it / 1767 cls / ClassCap | 1.6 → 8.6 (82%) |
| shader_domain_warp_fbm | 256×256 | 84 → 84 → 67 → 63 | 1680 → 1632 → 1712 (+1.9%) | 489 → 489 → 457 (-6.5%) | 205 → 186 | 116/4 → 116/4 | 0 → 0 | 0 → 0 (+0.0%) | 6977 apps / 3 it / 4592 cls / ClassCap | 1.5 → 24.2 (94%) |
| shader_gyroid_slice | 256×256 | 40 → 40 → 38 → 33 | 2768 → 2768 → 2768 (+0.0%) | 952 → 952 → 932 (-2.1%) | 168 → 163 | 124/4 → 124/4 | 0 → 0 | 514 → 514 (+0.0%) | 8618 apps / 9 it / 775 cls / Quiesced | 2.1 → 17.7 (88%) |
| shader_julia_set | 256×256 | 122 → 122 → 110 → 123 | 2064 → 2016 → 2064 (+0.0%) | 710 → 660 → 717 (+1.0%) | 208 → 210 | 40/14 → 58/14 | 0 → 0 | 0 → 0 (+0.0%) | 14956 apps / 5 it / 4533 cls / ClassCap | 5.5 → 38.6 (90%) |
| shader_kaleidoscope_fold | 256×256 | 46 → 46 → 43 → 40 | 2016 → 2016 → 2032 (+0.8%) | 567 → 567 → 560 (-1.2%) | 242 → 238 | 156/10 → 156/10 | 0 → 0 | 257 → 257 (+0.0%) | 601 apps / 5 it / 131 cls / Quiesced | 3.0 → 5.5 (42%) |
| shader_mandelbrot_distance | 256×256 | 152 → 152 → 128 → 96 | 2080 → 1824 → 1408 (-32.3%) | 745 → 657 → 518 (-30.5%) | 175 → 141 | 44/19 → 12/19 | 2 → 0 | 114945 → 0 (-100.0%) | 15392 apps / 4 it / 3675 cls / ClassCap | 5.1 → 37.7 (90%) |
| shader_metaballs | 256×256 | 62 → 62 → 52 → 50 | 496 → 480 → 480 (-3.2%) | 168 → 168 → 155 (-7.7%) | 51 → 43 | 0/1 → 0/1 | 0 → 0 | 0 → 0 (+0.0%) | 10090 apps / 4 it / 2584 cls / ClassCap | 0.5 → 54.6 (98%) |
| shader_plasma | 256×256 | 37 → 37 → 35 → 29 | 1360 → 1360 → 1376 (+1.2%) | 378 → 378 → 359 (-5.0%) | 129 → 122 | 87/3 → 87/3 | 0 → 0 | 0 → 0 (+0.0%) | 3298 apps / 4 it / 1822 cls / ClassCap | 1.0 → 11.9 (88%) |
| shader_smooth_min_scene | 256×256 | 43 → 43 → 40 → 37 | 416 → 416 → 432 (+3.8%) | 141 → 141 → 134 (-5.0%) | 34 → 32 | 0/0 → 0/0 | 0 → 0 | 0 → 0 (+0.0%) | 2757 apps / 3 it / 1215 cls / ClassCap | 0.3 → 7.1 (93%) |
| shader_smoothstep_vignette | 256×256 | 64 → 64 → 50 → 49 | 544 → 496 → 544 (+0.0%) | 183 → 183 → 194 (+6.0%) | 56 → 41 | 0/0 → 0/0 | 0 → 0 | 0 → 0 (+0.0%) | 1596 apps / 5 it / 283 cls / Quiesced | 0.4 → 5.3 (84%) |
| shader_star_sdf | 256×256 | 66 → 66 → 58 → 58 | 544 → 528 → 640 (+17.6%) | 195 → 195 → 172 (-11.8%) | 62 → 56 | 0/1 → 0/1 | 0 → 0 | 0 → 0 (+0.0%) | 7997 apps / 4 it / 3828 cls / ClassCap | 0.6 → 20.5 (95%) |
| shader_torus_slice | 256×256 | 38 → 38 → 34 → 31 | 384 → 368 → 368 (-4.2%) | 151 → 141 → 130 (-13.9%) | 30 → 25 | 0/0 → 0/0 | 0 → 0 | 0 → 0 (+0.0%) | 4693 apps / 5 it / 1281 cls / ClassCap | 0.3 → 10.3 (96%) |

## Where saturation makes the kernel worse, and which rule

Sorted by bytes Δ. Rule column: the on-extraction's rules from the provenance journal, `strict/tight/fired` — strict credits an application only when its own e-node was chosen (`EpisodeLabels::compute_strict`), tight is `derivation_ancestors_tight`.

| kernel | bytes off → cse → on | dag_cost off → cse → on | mem ops Δ | rules (strict/tight/fired) |
|---|---:|---:|---:|---|
| shader_star_sdf | 544 → 528 → 640 (+17.6%) | 195 → 195 → 172 (-11.8%) | +0.0% | `fma-fusion` 7/110/3053, `commutative(Add)` 0/82/1418, `associative(Add)` 0/50/1209, `reverse-associative(Add)` 0/43/1353, `commutative(Mul)` 0/25/308 |
| shader_smooth_min_scene | 416 → 416 → 432 (+3.8%) | 141 → 141 → 134 (-5.0%) | +0.0% | `fma-fusion` 2/48/820, `commutative(Add)` 0/36/331, `associative(Add)` 0/35/455, `commutative(Mul)` 0/30/109, `reverse-associative(Mul)` 1/28/121 |
| shader_domain_warp_fbm | 1680 → 1632 → 1712 (+1.9%) | 489 → 489 → 457 (-6.5%) | +0.0% | `fma-fusion` 10/226/1643, `commutative(Add)` 0/115/1193, `associative(Add)` 3/95/757, `reverse-associative(Add)` 3/80/840, `commutative(Mul)` 0/30/549 |
| shader_plasma | 1360 → 1360 → 1376 (+1.2%) | 378 → 378 → 359 (-5.0%) | +0.0% | `fma-fusion` 6/429/923, `commutative(Add)` 0/425/890, `reverse-associative(Add)` 1/168/471, `associative(Add)` 1/114/294, `commutative(Mul)` 0/59/204 |
| shader_kaleidoscope_fold | 2016 → 2016 → 2032 (+0.8%) | 567 → 567 → 560 (-1.2%) | +0.0% | `commutative(Mul)` 0/13/270, `fma-fusion` 3/7/40, `commutative(Add)` 0/4/44, `associative(Mul)` 0/2/87, `commutative(Max)` 0/2/18 |
| shader_julia_set | 2064 → 2016 → 2064 (+0.0%) | 710 → 660 → 717 (+1.0%) | +0.0% | `fma-fusion` 9/746/3267, `commutative(Add)` 0/404/2858, `associative(Add)` 0/273/1225, `commutative(Mul)` 0/224/1899, `reverse-associative(Add)` 2/203/1435 |
| shader_smoothstep_vignette | 544 → 496 → 544 (+0.0%) | 183 → 183 → 194 (+6.0%) | +0.0% | `fma-fusion` 7/22/265, `commutative(Mul)` 0/19/309, `commutative(Add)` 0/16/260, `constant-fold` 5/8/68, `canonicalize(Sub)` 2/5/40 |

### The largest wins, for the same rule attribution

| kernel | bytes off → cse → on | dag_cost off → cse → on | rules (strict/tight/fired) |
|---|---:|---:|---|
| psychedelic_packed | 10496 → 2176 → 2368 (-77.4%) | 4948 → 860 → 825 (-83.3%) | `fma-fusion` 14/371/1194, `commutative(Mul)` 0/360/1329, `associative(Mul)` 7/226/768, `reverse-associative(Mul)` 1/190/918, `distribute` 4/142/515 |
| cellgrid_80x24_d2 | 8864 → 2512 → 2576 (-70.9%) | 1879 → 450 → 428 (-77.2%) | `fma-fusion` 20/214/1515, `commutative(Add)` 0/137/861, `associative(Add)` 1/72/631, `reverse-associative(Add)` 0/58/614, `commutative(Mul)` 0/47/757 |
| glyph16_U0033 | 43408 → 24256 → 22832 (-47.4%) | 21916 → 10737 → 10873 (-50.4%) | `commutative(Mul)` 0/565/661, `commutative(Add)` 0/435/436, `associative(Add)` 42/173/173, `distribute` 40/155/155, `doubling` 0/134/218 |
| glyph32_U0033 | 43408 → 24256 → 22832 (-47.4%) | 21916 → 10737 → 10873 (-50.4%) | `commutative(Mul)` 0/565/661, `commutative(Add)` 0/435/436, `associative(Add)` 42/173/173, `distribute` 40/155/155, `doubling` 0/134/218 |
| glyph16_U0038 | 56288 → 30512 → 29664 (-47.3%) | 28621 → 13849 → 13645 (-52.3%) | `commutative(Mul)` 0/725/853, `commutative(Add)` 0/550/551, `canonicalize(Sub)` 0/132/132, `commutative(Max)` 0/94/94, `commutative(Min)` 0/65/65 |
| glyph32_U0038 | 56288 → 30512 → 29664 (-47.3%) | 28621 → 13849 → 13645 (-52.3%) | `commutative(Mul)` 0/725/853, `commutative(Add)` 0/550/551, `canonicalize(Sub)` 0/132/132, `commutative(Max)` 0/94/94, `commutative(Min)` 0/65/65 |
| glyph16_U0061 | 40544 → 22960 → 21376 (-47.3%) | 20428 → 10044 → 10230 (-49.9%) | `commutative(Mul)` 0/530/619, `commutative(Add)` 0/416/416, `associative(Add)` 52/221/221, `doubling` 1/164/279, `distribute` 50/144/145 |
| glyph32_U0061 | 40544 → 22960 → 21376 (-47.3%) | 20428 → 10044 → 10230 (-49.9%) | `commutative(Mul)` 0/530/619, `commutative(Add)` 0/416/416, `associative(Add)` 52/221/221, `doubling` 1/164/279, `distribute` 50/144/145 |
| glyph16_U0028 | 14528 → 8640 → 7712 (-46.9%) | 7237 → 3607 → 3145 (-56.5%) | `commutative(Mul)` 0/618/1648, `commutative(Add)` 0/360/699, `fma-fusion` 40/344/867, `reverse-associative(Mul)` 7/196/635, `associative(Mul)` 28/130/336 |
| glyph32_U0028 | 14528 → 8640 → 7712 (-46.9%) | 7237 → 3607 → 3145 (-56.5%) | `commutative(Mul)` 0/618/1648, `commutative(Add)` 0/360/699, `fma-fusion` 40/344/867, `reverse-associative(Mul)` 7/196/635, `associative(Mul)` 28/130/336 |

## Which rules are load-bearing on real shaders (all kernels, production run)

| rule | kernels where load-bearing | Σ strict | Σ tight | Σ fired |
|---|---:|---:|---:|---:|
| `fma-fusion` | 204 | 5451 | 71804 | 318899 |
| `constant-fold` | 203 | 4720 | 8369 | 49070 |
| `canonicalize(Div)` | 198 | 3530 | 3796 | 6159 |
| `canonicalize(Sub)` | 205 | 3256 | 8882 | 14260 |
| `factor` | 194 | 3208 | 10190 | 78669 |
| `associative(Add)` | 201 | 3041 | 28164 | 210276 |
| `associative(Mul)` | 198 | 2593 | 28406 | 85134 |
| `distribute` | 202 | 1806 | 16426 | 76304 |
| `reverse-associative(Mul)` | 198 | 1434 | 12800 | 97546 |
| `doubling` | 197 | 199 | 10019 | 33018 |
| `reverse-associative(Add)` | 197 | 121 | 22294 | 230871 |
| `diff-of-squares` | 2 | 6 | 9 | 45 |
| `halving` | 30 | 4 | 268 | 6246 |
| `annihilator` | 193 | 0 | 1363 | 27111 |
| `associative(Max)` | 1 | 0 | 4 | 8 |
| `associative(Min)` | 1 | 0 | 3 | 18 |
| `cancellation(Sub)` | 79 | 0 | 152 | 503 |
| `commutative(Add)` | 206 | 0 | 65912 | 228713 |
| `commutative(Max)` | 204 | 0 | 5457 | 10128 |
| `commutative(Min)` | 206 | 0 | 4300 | 9247 |
| `commutative(Mul)` | 206 | 0 | 57242 | 154280 |
| `cos-angle-addition` | 2 | 0 | 5 | 21 |
| `exp-homomorphism` | 1 | 0 | 5 | 21 |
| `idempotent(Min)` | 1 | 0 | 1 | 6 |
| `identity(Add)` | 197 | 0 | 3275 | 80030 |

## `SelectHoistUnary` (`select-hoist-neg|abs|sqrt`)

**Not in production.** The three rules live in `round2_rules::experimental_rules()` and are not part of `all_rules()`, so they fire zero times in every production compile above by construction. The `with-select-hoist` arm adds them to production's rules: fired in **29 of 208** measured kernels (195 applications); tight-load-bearing in **29** (188); strict 0.

| kernel | fired | tight | strict | bytes on → +hoist | dag_cost on → +hoist | guarded/selects on → +hoist | schedule on → +hoist |
|---|---:|---:|---:|---:|---:|---:|---:|
| glyph16_U0028 | 8 | 8 | 0 | 7712 → 7712 | 3145 → 3145 | 490/25 → 490/25 | 210 → 210 |
| glyph16_U0044 | 8 | 8 | 0 | 8288 → 8288 | 3264 → 3264 | 516/27 → 516/27 | 229 → 229 |
| glyph16_U004A | 8 | 8 | 0 | 8688 → 8688 | 3386 → 3386 | 545/29 → 545/29 | 247 → 247 |
| glyph16_U0050 | 8 | 8 | 0 | 8512 → 8512 | 3306 → 3306 | 524/28 → 524/28 | 237 → 237 |
| glyph16_U0068 | 8 | 8 | 0 | 8944 → 8944 | 3405 → 3405 | 545/30 → 545/30 | 253 → 253 |
| glyph16_U006E | 8 | 8 | 0 | 8816 → 8816 | 3401 → 3401 | 545/30 → 545/30 | 253 → 253 |
| glyph16_U0072 | 8 | 8 | 0 | 8800 → 8800 | 3386 → 3386 | 532/29 → 532/29 | 245 → 245 |
| glyph16_U0075 | 8 | 8 | 0 | 9152 → 9152 | 3423 → 3423 | 552/30 → 552/30 | 254 → 254 |
| glyph32_U0028 | 8 | 8 | 0 | 7712 → 7712 | 3145 → 3145 | 490/25 → 490/25 | 210 → 210 |
| glyph32_U0044 | 8 | 8 | 0 | 8288 → 8288 | 3264 → 3264 | 516/27 → 516/27 | 229 → 229 |
| glyph32_U004A | 8 | 8 | 0 | 8704 → 8704 | 3386 → 3386 | 545/29 → 545/29 | 247 → 247 |
| glyph32_U0050 | 8 | 8 | 0 | 8512 → 8512 | 3306 → 3306 | 524/28 → 524/28 | 237 → 237 |
| glyph32_U0068 | 8 | 8 | 0 | 8944 → 8944 | 3405 → 3405 | 545/30 → 545/30 | 253 → 253 |
| glyph32_U006E | 8 | 8 | 0 | 8816 → 8816 | 3401 → 3401 | 545/30 → 545/30 | 253 → 253 |
| glyph32_U0072 | 8 | 8 | 0 | 8800 → 8800 | 3386 → 3386 | 532/29 → 532/29 | 245 → 245 |
| glyph32_U0075 | 8 | 8 | 0 | 9152 → 9152 | 3423 → 3423 | 552/30 → 552/30 | 254 → 254 |
| glyph16_U0029 | 7 | 7 | 0 | 7072 → 7072 | 2914 → 2914 | 448/23 → 448/23 | 197 → 197 |
| glyph32_U0029 | 7 | 7 | 0 | 7072 → 7072 | 2914 → 2914 | 448/23 → 448/23 | 197 → 197 |
| glyph16_U0079 | 6 | 6 | 0 | 7296 → 7296 | 2840 → 2840 | 437/24 → 437/24 | 210 → 210 |
| glyph32_U0079 | 6 | 6 | 0 | 7296 → 7296 | 2840 → 2840 | 437/24 → 437/24 | 210 → 210 |
| glyph16_U0066 | 4 | 4 | 0 | 4992 → 4992 | 1959 → 1959 | 298/20 → 298/20 | 166 → 166 |
| glyph16_U006A | 4 | 4 | 0 | 4992 → 4992 | 1964 → 1964 | 293/19 → 293/19 | 161 → 161 |
| glyph16_U006C | 4 | 4 | 0 | 4720 → 4720 | 1867 → 1867 | 278/17 → 278/17 | 146 → 146 |
| glyph16_U0074 | 4 | 4 | 0 | 4992 → 4992 | 1960 → 1960 | 297/20 → 297/20 | 165 → 165 |
| glyph32_U0066 | 4 | 4 | 0 | 4992 → 4992 | 1959 → 1959 | 298/20 → 298/20 | 166 → 166 |
| glyph32_U006A | 4 | 4 | 0 | 4992 → 4992 | 1964 → 1964 | 293/19 → 293/19 | 161 → 161 |
| glyph32_U006C | 4 | 4 | 0 | 4720 → 4720 | 1867 → 1867 | 278/17 → 278/17 | 146 → 146 |
| glyph32_U0074 | 4 | 4 | 0 | 4992 → 4992 | 1960 → 1960 | 297/20 → 297/20 | 165 → 165 |
| shader_mandelbrot_distance | 9 | 2 | 0 | 1408 → 1408 | 518 → 518 | 12/19 → 12/19 | 141 → 141 |

Adding the rule changed the emitted bytes of **0** kernels; Σ guarded-value delta (+hoist − on) over all measured kernels: **+0**.

## Correctness

Same-form: `eval_scalar` of the emitted arena vs the JIT at 256 sampled pixels (a difference here is a JIT bug). Cross-form: `eval_scalar` of the legalized arena as constructed vs the JIT of the on-extraction (what the rewrites moved; divergence at singularities is the algebraic contract, not a defect).

- same-form NaN mismatches over all kernels: on **0**, off **0**; worst same-form |Δ|: on `shader_kaleidoscope_fold` 6.109476089477539e-6 (0 NaN), off `shader_kaleidoscope_fold` 6.109476089477539e-6
- worst cross-form |Δ| (on): `shader_domain_warp_fbm` 4.061430096626282e-1 (0 NaN)
- full-extent output bit-identical off vs on: **9 of 208** kernels
- packed kernels (on): `cellgrid_80x24_d2`: same-form 0 mismatching of 256 sampled pixels (max byte Δ 0), cross-form 0 (max byte Δ 0); `chrome_packed`: same-form 0 mismatching of 256 sampled pixels (max byte Δ 0), cross-form 58 (max byte Δ 1); `psychedelic_packed`: same-form 0 mismatching of 256 sampled pixels (max byte Δ 0), cross-form 0 (max byte Δ 0)
- instrument = production path: `Manifold::compile` bytes differed for 0 kernels []; probe extraction differed from production for 0 kernels []

## Saturation's share of compile time

`optimize_ms` is `optimize_runtime_arena` (legalize + saturate + extract); `emit_ms` is `emit::compile`. Wall clock at the load stated above — a ratio, not a number.

| class | Σ compile off (ms) | Σ compile on (ms) | Σ saturation (on − off optimize) | share of on |
|---|---:|---:|---:|---:|
| bench | 97.8 | 128.9 | 90.1 | 70% |
| bench_wide | 38.7 | 33.6 | 18.8 | 56% |
| cellgrid | 1.6 | 25.2 | 24.4 | 97% |
| glyph16 | 1782.9 | 5818.8 | 4990.9 | 86% |
| glyph32 | 1944.7 | 5751.8 | 4887.8 | 85% |
| psychedelic | 4.9 | 24.7 | 23.6 | 96% |
| shader | 22.0 | 242.2 | 220.9 | 91% |
| chrome_R (on only) | - | 74.0 | 70.8 | - |
| chrome_packed (on only) | - | 222.9 | 217.5 | - |
