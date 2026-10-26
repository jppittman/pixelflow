# Production glyph path baseline, 2026-10-20: the A0 numbers

**Date:** 2026-10-20
**Verified against:** `298d770c11a3a7642d738d3df1b177b8b2bc3cc3` (main with the plan merged, #1435).
The runs were taken on the plan's branch head `73cd3e5b`, tracked files clean; its tree is
`298d770c`'s (`a3c5f933`), so the figures are this commit's.
**Status:** Measurement record, no code changed. It is step **A0** of
[a bound is a value](../language/2026-10-20-a-bound-is-a-value.md) §6: production's glyph path,
re-baselined before anything changes, so that B2's "**1** program per tile extent" and every
other later number is read against it. Point-in-time; each figure is on the tree above.
**Host:** KVM guest, Intel Xeon @ 2.10 GHz, 4 vCPUs (1 thread per core), 16 GiB, Linux 6.18.
`isa::detect()` picks **AVX-512** here (no `PIXELFLOW_ISA`; the host has `avx512f` and
`avx512dq`); AVX2 is measured with `PIXELFLOW_ISA=avx2`. Every run prints its tier; no run is
labelled by the variable alone.

## 1. What was measured, and what could not be

**The path.** Production bakes a glyph through `GlyphAtlas::warm` -> `GlyphAtlas::uv`
(`pixelflow-graphics/src/fonts/atlas.rs:196`) -> `Glyph::bake` (`fonts/loop_blinn.rs`,
`Manifold::compile` + `bind` + `Lattice::collapse`), on a `Glyph` that `loop_blinn::run` built.
It does not touch `FontPrograms`. Every figure below is on that path.

**The font.** core-term's `NotoSansMono-Regular.ttf` is a Git LFS pointer in this checkout (131
bytes). The real file is `/home/user/scratch/fonts/noto_lfs.ttf` (405,892 bytes, sha256
`11b5d661e57865bce89d3f103a654cfa4faf9347abebac252e30742be1e657b2`, which is the pointer's own `oid`),
and nothing tracked was changed to use it. **Noto has no glyph for U+2328 (⌨)**; a production
atlas draws it as the blank tile. The ⌨ rows therefore use the tree's committed fallback,
`DejaVuSansMono-Fallback.ttf`, the font `tests/glyph_area_edge_cases.rs` bakes ⌨ from.

**Which of the three named examples could produce a number.**

| example | as it exists | result |
|---|---|---|
| `glyph_compile_report` (both modes) | `FONT_PATH` is a compile-time `concat!(CARGO_MANIFEST_DIR, ...)` of the LFS pointer, no override | **panics** at `Font::parse` (`glyph_compile_report.rs:121`). Run; exit 134. |
| `glyph_phase_split` | same | **panics** at `glyph_phase_split.rs:102`. Run; exit 134. |
| `frame_budget` | `PIXELFLOW_FRAME_BUDGET_FONT` override; refuses the pointer loudly | runs, **zoom mode** only (§3.5). Frame mode measures render time, which is not an A0 quantity; not run. |

So the first two ran as **patched copies**, in scratch and uncommitted, each with exactly two edits
(diffs at `/home/user/scratch/bound-value/a0/src/a0_glyph_compile_report.diff` and
`.../a0_glyph_phase_split.diff`): `FONT_PATH` points at the real file, and `DENSITIES` is one
density per process (`A0_DENSITY=1|2`). The second edit is about coldness, not convenience:
`optimize_runtime_arena`'s saturation cache is keyed on structure and serves every shape, so a
second density in the same process pays no saturation (§3.5 measures this). Their numbers are the
**cross-check** in §3.4, not the headline.

**Why a probe exists.** None of the three examples reports programs, saturations, per-program
defs, or the ⌨ glyph, and none separates compile from bake on `Glyph::bake`. The task's exception
applies: a measurement-only binary, `a0_warm_probe` (`/home/user/scratch/bound-value/a0/src/`,
not committed), public API only:

- `warm`: `GlyphAtlas::new` + `GlyphAtlas::warm(' '..='~')` itself, timed; then `jit_cache::entry_count()`
  before and after (programs); then `jit_cache::compile` on each glyph's kernel, which is a hit, and
  de-duplication by the returned `Arc` (distinct programs and their `as_bytes().len()`, a hit asserted
  by the entry count not moving); then a second atlas warmed in the same process (every program
  cached), which is **everything except compile**.
- `split` and `keyboard`: `GlyphAtlas::uv`'s body restated (`glyph_kernel_scaled`, the `+0.5`
  pixel-centre contramap, `PIXEL_CENTER` being `pub(crate)`), with `Glyph::bake` taken into its two
  public halves, `Glyph::bound` (`Manifold::compile` + `bind`: key, saturate, extract, emit) timed as
  **compile** and `Lattice::collapse` timed as **bake**. `warm`'s production-call total and `split`'s
  sum agree (110 vs 104 ms at AVX-512 tile 16), which is the check that the restatement is the path.

**Defs.** `arena_to_schedule` is `pub(crate)`, so a schedule's def count is not reported by any
public API (the "27,029 defs" of the plan's §2 came from a temporary print). The probe replays what
`jit_cache::compile` does (`optimize_runtime_arena`, `relink`, `emit::compile`), **asserts the
replayed code is byte-identical to the cached program** (it was, in every run), and counts defs by
the rule `arena_to_schedule` states: one def per reachable non-`Write` node of the arena
`passes::legalize` returns (an empty fold does not reach its body), plus one `Context` def per
uniform block. `ops` below is `EmitTraffic`'s scheduled-operation count, the independent public
figure; it runs 4-21 below defs, as constants rematerialised in place are not counted there.

## 2. Commands

Build (separate target directories under scratch; the release profile and rustflags are the
workspace's, `opt-level = 3`, `panic = "abort"`, `-C llvm-args=-fp-contract=fast`):

```text
# the three real examples, from the workspace, with telemetry on the optimizer
cd /home/user/pixelflow
CARGO_TARGET_DIR=/home/user/scratch/bound-value/a0/target-pipeline \
  cargo build --release --offline --locked -p pixelflow-pipeline \
  --features pixelflow-search/saturation-telemetry \
  --example glyph_compile_report --example glyph_phase_split --example frame_budget

# the probe, and the two patched copies, as scratch packages over path dependencies
cd /home/user/scratch/bound-value/a0/probe     # and .../probe-ex for the copies
RUSTFLAGS="-C llvm-args=-fp-contract=fast" CARGO_TARGET_DIR=/home/user/scratch/bound-value/a0/target \
  cargo build --release --offline --bin a0_warm_probe
```

`pixelflow-pipeline` declares no `saturation-telemetry` feature, so the dependency-feature form
`pixelflow-search/saturation-telemetry` is what "with `--features saturation-telemetry`" means for
it; the scratch packages enable `pixelflow-{core,codegen,search}/saturation-telemetry`. Telemetry is
on for **every** timing below: one appended JSONL record per saturation, seven per run. It was not
measured off.

Run: one **cold process** per (mode, tier, tile, repeat), sequential, three repeats; the figures
are the **median of 3** with the range beside it. Each run's sink is its own file, so a run's
saturation count is its line count.

```text
PIXELFLOW_ISA={avx512|avx2} PIXELFLOW_SATURATION_TELEMETRY=<run>.jsonl \
  a0_warm_probe {warm|split|keyboard|retile} <font> <density 1|2|1.125> [hex | density2]
PIXELFLOW_ISA=... A0_DENSITY={1|2} PIXELFLOW_GLYPH_REPORT_MODE={kernels|warm} a0_glyph_compile_report
PIXELFLOW_ISA=... A0_DENSITY={1|2} a0_glyph_phase_split
PIXELFLOW_ISA=... PIXELFLOW_FRAME_BUDGET_FONT=/home/user/scratch/fonts/noto_lfs.ttf \
  PIXELFLOW_FRAME_BUDGET_MODE=zoom frame_budget
```

Density 1 and 2 are core-term's (`GlyphAtlas::new(16.0, density, 128)`), tile 16 and tile 32.
Raw outputs: `/home/user/scratch/bound-value/a0/{runs,runs-ex,examples-as-is}/`; the summaries
`summarize.py`, `summarize_ex.py` and the runners `run_matrix.sh`, `run_ex.sh` sit beside them. The
first run of a batch is often the slowest (144.7, 132.6, 130.8 ms in the tables' ranges); on a
4-vCPU guest read the ranges, not the third digit.

## 3. Results

### 3.1 Cold `GlyphAtlas::warm(' '..='~')`, NotoSansMono-Regular, cell 16 pt

95 characters, every one with a kernel (the space is `Glyph::empty()`, not a skipped slot). Median
of 3 cold processes.

| | AVX-512 tile 16 | AVX-512 tile 32 | AVX2 tile 16 | AVX2 tile 32 |
|---|---|---|---|---|
| **distinct compiled programs** (JIT cache entries; = distinct `Arc`s) | **7** | **7** | **7** | **7** |
| **saturations** (telemetry records; 0 declined) | **7** | **7** | **7** | **7** |
| &nbsp;&nbsp;stopped by `class_cap` (5000) / `quiesced` | 6 / 1 | 6 / 1 | 6 / 1 | 6 / 1 |
| rule applications over the seven | 37,663 | 37,663 | 37,663 | 37,663 |
| **defs per program** min / median / max | 29 / 202 / 322 | 29 / 202 / 322 | 29 / 202 / 322 | 29 / 202 / 322 |
| &nbsp;&nbsp;without the blank program | 201 / 202 / 322 | same | same | same |
| scheduled ops (`EmitTraffic`) min / median / max | 25 / 181 / 303 | same | same | same |
| **code bytes**, the seven programs together | **16,856** | **16,928** | **17,052** | **17,148** |
| &nbsp;&nbsp;per program min - max | 412 - 4,084 | 416 - 4,136 | 400 - 4,012 | 400 - 4,108 |
| &nbsp;&nbsp;sum over the 95 glyphs of their program's bytes (what 95 unshared compiles would hold) | 242,516 | 243,064 | 247,000 | 247,544 |
| frame bytes: loop programs / largest | 1,344 / 3,008 | 1,344 / 3,136 | 1,344 / 2,176 | 1,344 / 2,272 |
| **cold `atlas.warm`**, ms (the production call) | **110.3** (107.8-144.7) | **109.8** (102.7-119.0) | **108.4** (103.4-132.6) | **101.9** (96.2-130.8) |
| &nbsp;&nbsp;same, `glyph_compile_report` copy | 102.1 (101.1-122.8) | 101.8 (93.6-113.8) | 110.6 (106.8-125.0) | 103.5 (103.3-105.3) |
| **`split`: kernel build** (outline -> pieces -> table -> kernel), ms | 53.6 | 59.9 | 61.9 | 51.7 |
| **`split`: compile** (`Glyph::bound`), ms | **47.9** | **56.0** | **48.9** | **44.7** |
| &nbsp;&nbsp;of it, saturation wall (telemetry `wall_clock_us`) | 39.5 | 47.1 | 40.5 | 37.0 |
| **`split`: bake** (`Lattice::collapse`, 95 tiles), ms | **0.70** | **1.0** | **0.4** | **1.0** |
| `split` total | 104.2 | 119.9 | 113.1 | 99.1 |
| second warm, every program cached (everything but compile), ms | 62.1 (55.0-80.1) | 56.3 (54.2-85.7) | 57.1 (50.9-58.5) | 56.4 (51.8-74.4) |

Medians of separate fields need not sum. A compile that hits the cache costs 14-22 us (median over
the 88 glyphs that share a program), and the six non-blank programs cost 6.6-14.3 ms each to mint (the blank
0.5-0.7 ms; run 1 of each cell). `compile - saturation` is 7.7-8.9 ms: seven emits, the key, and the 88 hits.

**The seven programs** (Noto, tile 16; tile 32 differs by at most 60 bytes per program):

| fold trip count (bucket) | glyphs | defs | ops | code B AVX-512 / AVX2 | frame B AVX-512 / AVX2 | spill slots AVX-512 / AVX2 | surviving folds |
|---|---|---|---|---|---|---|---|
| none (`Glyph::empty()`, the space) | 1 | 29 | 25 | 412 / 400 | 64 / 192 | 7 / 9 | 3 |
| 2 | 6 | 322 | 303 | 4,084 / 4,012 | 3,008 / 2,176 | 57 / 71 | 3 (unrolled: no glyph fold survives; `nodes_out` 238 against 142, §3.4) |
| 4 | 8 | 201 | 181 | 2,468 / 2,524 | 1,344 / 1,344 | 31 / 46 | 4 |
| 8 | 17 | 201 | 181 | 2,468 / 2,524 | 1,344 / 1,344 | 31 / 46 | 4 |
| 16 | 22 | 202 | 181 | 2,472 / 2,528 | 1,344 / 1,344 | 31 / 46 | 4 |
| 32 | 30 | 202 | 181 | 2,476 / 2,532 | 1,344 / 1,344 | 31 / 46 | 4 |
| 64 | 11 | 202 | 181 | 2,476 / 2,532 | 1,344 / 1,344 | 31 / 46 | 4 |

"Surviving folds" is the lattice's row, column and lane folds (3) plus the glyph's own fold (4th)
where it is kept as a loop. The trip-count-2 bucket is **unrolled** (3 folds), which is why it is
larger; every other bucket keeps the loop. The six bucket programs are one input (154 inserted
classes each) differing only in the fold's trip count, and each saturates alone. The same
seven programs, with the same defs and the same code bytes, come out of the DejaVu fallback's
ASCII (its glyphs fall in different buckets: 8 / 12 / 27 / 15 / 28 / 4 over 2...64); the code is a
function of the bucket and the tier, the glyph is data.

### 3.2 The keyboard glyph U+2328 through `Glyph::bake`, tile 16 and 32

DejaVuSansMono-Fallback, because Noto has no ⌨ (§1). One cold process per row, median of 3. One
saturation (154 inserted classes, `class_cap`, 6,670 applications), one new program.

| | AVX-512 tile 16 | AVX-512 tile 32 | AVX2 tile 16 | AVX2 tile 32 |
|---|---|---|---|---|
| outline segments / **pieces** (`GlyphRows::pieces`) | 347 / **262** | 347 / 262 | 347 / 262 | 347 / 262 |
| fold trip count (bucketed) | 512 | 512 | 512 | 512 |
| **defs** / scheduled ops | **310** / 289 | 310 / 289 | 310 / 289 | 310 / 289 |
| surviving folds | 4 (loop kept) | 4 | 4 | 4 |
| **code bytes** | **3,692** | 3,696 | 3,684 | 3,684 |
| **frame bytes** / spill slots | **1,472** / 33 | 1,472 / 33 | 1,440 / 49 | 1,440 / 49 |
| **compile** (`Glyph::bound`), ms | **11.0** (10.9-11.7) | 8.2 (8.0-11.2) | 8.4 (7.7-8.9) | 12.2 (9.8-12.6) |
| &nbsp;&nbsp;of it, saturation wall | 9.1 | 6.7 | 6.8 | 10.3 |
| kernel build, ms | 1.06 | 0.74 | 0.78 | 1.10 |
| **bake** (`Lattice::collapse`), ms | **0.035** (0.033-0.043) | 0.085 (0.084-0.107) | 0.045 (0.039-0.067) | 0.148 (0.137-0.180) |
| re-compile once cached, ms | 0.024 | 0.020 | 0.018 | 0.028 |

The plan's "347 pieces" is the outline's **segment** count; the monotone pieces `loop_blinn::pieces`
makes of it, which is what `GlyphRows::pieces()` and the table's rows count, are **262**, at every
size tried (5, 16, 32, 64 px). The plan's "27,029 defs, 75 KiB frame" row is the `FontPrograms`
program for ⌨, as the 2026-09-10 note already says of it
([a surviving reduce is a loop](../codegen/2026-09-10-a-surviving-reduce-is-a-loop.md), "Measured
2026-10-16": `Glyph::bake` keeps its fold, 206-314 defs); **the production path for the same glyph is 310 defs, 3.7 KB of code, and a
1.4 KiB frame**, the glyph's fold kept as a loop. The ⌨ program has more defs than an ASCII loop
program (310 against 202), and a larger trip count (512 against 4-64); the DejaVu ASCII programs are
also 202, so it is not the font. Not isolated here: whether the trip count or something particular
to ⌨ is the cause.

### 3.3 Where the cold warm goes (ASCII, tile 16 and 32)

About half is the host building 95 kernels (outline -> monotone pieces -> table -> kernel), about
half is compiling seven programs, and the bake is under 1%. The cached re-warm (57 ms) is the first
half plus the 95 key lookups. This is the part ruling 5 ("the derivation should stay in rust")
leaves where it is, so no later step moves it: it is the floor under any "a glyph is a call"
number.

### 3.4 The examples, as they exist or as patched

`frame_budget` zoom mode, unpatched, with the font override (median of 3; `face_ms` is a
`GlyphAtlas::new` + warm of printable ASCII; the run also builds the cell-grid program):

| step | tile | AVX-512 `face_ms` | AVX2 `face_ms` | programs added by the face | saturations added |
|---|---|---|---|---|---|
| cold | 16 | 98.6 (95.3-101.0) | 100.5 (93.2-117.1) | 7 | 7 |
| zoom (cell 11x18) | 18 | 69.5 (68.8-71.1) | 70.9 (65.8-101.0) | 7 | 0 |
| revisit (cell 10x16) | 16 | 50.8 (50.5-53.3) | 52.5 (51.6-57.5) | 0 | 0 |

(The run's whole telemetry file is 9 records, equal to the cold step's 9 added programs: 7 from the
face, 1 for the cell-grid program, 1 for the first frame. The zoom step's zero is §3.5's direct
measurement, not an inference from this file alone.) The patched copies, each over one cold process per
density, 95 kernels compiled **one by one with no program sharing** (`compile_as_baked`, not the
JIT cache, so they price every glyph's extraction and emit, not production's seven compiles):

| | AVX-512 t16 | AVX-512 t32 | AVX2 t16 | AVX2 t32 |
|---|---|---|---|---|
| `glyph_compile_report` kernels: sum of 95 compiles, ms | 238.1 | 290.9 | 239.4 | 257.3 |
| &nbsp;&nbsp;sum of code bytes / distinct code hashes | 236,064 / 7 | 236,420 / 7 | 240,532 / 7 | 240,884 / 7 |
| `glyph_phase_split`: optimize ms / emit ms | 182.2 / 59.5 | 173.5 / 59.4 | 191.8 / 64.4 | 181.8 / 61.0 |
| &nbsp;&nbsp;`nodes_out` by program (loop / trip 2 / blank) | 142 / 238 / 15 | same | same | same |
| &nbsp;&nbsp;distinct programs "exactly" / bucketed | 7 / 7 | 7 / 7 | 7 / 7 | 7 / 7 |

(Their kernels are `glyph.kernel()` without the atlas's `+0.5` contramap, so their bytes differ
from §3.1's; the seven-way partition is the same.)

### 3.5 What a zoom costs: a new tile extent re-mints every program and saturates none

`retile`: a cold warm at density 1, then a warm at another density in the same process.

| | AVX-512 | AVX2 |
|---|---|---|
| tile 16 -> 18 (density 1.125): programs added, saturations added, second warm ms | 7, **0**, 72.6 (69.1-98.7) | 7, **0**, 83.7 (71.1-98.7) |
| tile 16 -> 32 (density 2): programs added, saturations added, second warm ms | 7, **0**, 76.8 (66.8-93.4) | 7, **0**, 71.6 (69.6-91.1) |

The JIT cache keys on structure **and shape**; the optimizer's saturation cache keys on structure
alone. A different tile extent therefore costs 7 emits (the second warm is ~70-85 ms against the
~105 ms cold, of which ~50 ms is the kernel build) and no saturation. This is why the cold tile-32
row of §3.1 had to be its own process, and what "A alone does not deliver ruling 6 on the production
path" (plan §6) refers to: the extent is in the key.

## 4. What this says about the plan's own figures

1. **Programs: 7 at both tiles, not 6 and 7.** BACKLOG H1 records "95 -> 6 programs at tile 16, 7 at
   tile 32" (a September measurement, with cold warm 3.7 s and 12.0 s). On this tree the partition is
   six trip-count buckets, 2 to 64, plus the blank, at *both* tiles; the piece counts do not depend
   on the tile. Hypothesis, unverified: H1 counted the blank at one tile and not the other, or its
   derivation differed. Whatever it was, B2's gate "**1** program per tile extent" is read against
   **7**, and against 6 non-blank buckets that each saturate alone.
2. **Cold warm is ~0.1 s, not seconds.** H1's 3.7 s and 12.0 s are not reproduced: roughly 35 and 115
   times what this tree takes. Not bisected: what moved it (the loop-kept glyph fold, the deleted guard
   clustering and the e-graph speed work all landed in between). The compile here is 45-56 ms: 7
   saturations, 6 of which stop at the 5000-class cap.
3. **⌨ is 262 pieces, 310 defs, 1.4 KiB of frame on production's path.** The 27,029-def, 75 KiB
   case is `FontPrograms`; the NEON 64 KiB reach (#1414) is not a production-path limit at
   the sizes measured: 1,472 B against 64 KiB.
4. **The plan's `FontPrograms` contrast was not measured.** No example measures it (it lives in
   `src` and tests only); the plan's own figure (29 programs, 528 `one_piece` copies, the-language-is-kernel
   `:1186-1188`) is quoted there and is not re-derived here.

## 5. What each number is for

| number | where it is read (plan sections) | what a later step compares |
|---|---|---|
| 7 programs per (font, tile) = 6 buckets + blank; 7 saturations; 6 at the class cap | plan §2 rows 3-4; plan §6 **B2** gate ("**1** program per tile extent"); plan §3.5 | B2 programs: 1 per tile extent; saturations after: not per bucket |
| +7 programs, 0 saturations per tile change (this note's §3.5) | plan §6 "A alone does not deliver ruling 6"; ruling 6 | after B2 a zoom adds 1 program, not 7 |
| defs 201-202 per loop program, 322 unrolled, 29 blank; ⌨ 310 | plan §2 row 2 | the size of "one program" after B2, per tier |
| code 16.9 KB over seven programs (2.4-2.5 KB per loop program; 242 KB if counted per glyph) | plan §6 B2 gate | one program's bytes against seven |
| frame 1,344 B per loop program, 3.1 KiB at most; ⌨ 1,472 B | plan §2 row 2 (75 KiB, NEON reach) | production's frame is KiB-scale; the plan's row is the `FontPrograms` path |
| compile 45-56 ms, saturation 37-47 ms of it; bake 0.4-1.0 ms | plan §6 B2 gate ("A0's numbers after") | compile and bake after, tier by tier |
| cold warm ~100-110 ms; cached re-warm ~55-62 ms | ruling 5 (derivation stays host Rust) | the re-warm is the floor under B2/B3 |

## 6. Not measured, and limits

- **`FontPrograms`**: no example measures it (§4.4).
- **U+2328 in the production font**: absent from Noto; ⌨ is on the DejaVu fallback. Noto's heaviest
  glyph at 16 px is U+A66E (189 pieces, per `frame_budget`), unmeasured here.
- **`frame_budget` frame mode and `heaviest` content**: render time and the cell-grid scene, not A0.
- **aarch64 / NEON**: this host is x86-64; both x86 tiers are covered, NEON is not.
- **Telemetry cost**: on for every timing; one append per saturation, never measured off.
- **Defs**: replayed and counted (see top), not read from the schedule; `ops` is the independent public
  figure. The replay's bytes matched production's in every run.
- **Noise**: 4-vCPU KVM guest, medians of 3, ranges shown; the timings are good to the ~10% those
  ranges allow and the counts (programs, saturations, defs, bytes, frames) were identical in all
  three runs of every cell, and across the two fonts for ASCII.
- The probe binary gained its `retile` mode after the first matrix ran; the other modes are unchanged.
