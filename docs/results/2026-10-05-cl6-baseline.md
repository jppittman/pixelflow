# CL6 baseline, 2026-10-05

What the emitter does with a loop nest, and what that costs in heap, measured
on `28ddbeaf` before the CL6 series (the allocation tail and the single
emission buffer of
[emit should just emit](../plans/2026-09-12-emit-should-just-emit.md)) moves
any of it. Every later CL of the series is gated on the byte rows, the golden
and the pin below being equal, and judged on the memory figures against these.

Host: Intel Xeon @ 2.80 GHz, 4 cores, 15 GB, Linux x86-64, with `avx512f` and
`avx512dq`, so both x86 tiers run natively (`PIXELFLOW_ISA=avx2` for the
narrower). Release builds throughout; the golden and the pin are host-independent
and need no particular CPU.

## Why the instruments came first

`byte_probe`'s nine rows are all point-shaped: no `Reduce` survives into the
emitter under them, so the nest scoping builds, the frame the allocator lays
out over it, the roots parked for the scopes inside and the labels naming its
loops are never exercised. A refactor of any of them can move every byte of a
real kernel and leave all nine lines identical.

That was checked rather than argued. Making the driver's fold slots first-wins
instead of last-wins (a deliberate un-aliasing of two sibling folds' slots,
changing no length and no count) and running the probe on both tiers:

| | original 9 rows | six new rows (compile and production columns) |
|---|---|---|
| AVX-512 and AVX2 | **identical** | five of six change `fnv`; lengths, spills and parks do not |

The sixth, `glyph_like_wL`, is the control: one batch wide, so it has no
remainder fold and no sibling to alias. The golden and the pin below fail on
the same mutation. The mutation is not in the tree; it is a `.rev()` on the two
`(0..nest.fold_count())` iterators that build `fold_map` and `binder_map` in
`compile_via_backend`.

## What S0 added

| instrument | where | what it holds |
|---|---|---|
| six sibling-fold rows | `pixelflow-codegen/examples/byte_probe.rs`, kernels in `pixelflow-codegen/tests/support/sibling_rows.rs` | the old row's line, then `CompileResult` and every scope of `EmitTraffic`; diffed by the byte gate |
| a three-backend byte golden | `emit::tests::sibling_folds`, `the_sibling_fold_rows_emit_the_recorded_bytes_on_every_backend` | `(length, FNV-1a 64)` per row per backend |
| the slot pin | `sibling_column_folds_share_a_reduce_and_its_slots` | the later fold's slots are the earlier's |
| `alloc_probe` | `pixelflow-codegen/src/alloc_probe.rs` (`cfg(test)`) | bytes requested, allocations and peak, inside a `measure` scope on one thread |
| the sibling-scopes measurement | `sibling_scopes_allocation` (`#[ignore]`) | F = 16 to 1024 sibling folds, per stage |
| the N-glyph probe | `pixelflow-pipeline/examples/glyph_program_report.rs` | 4 to 94 glyphs as one program, per stage |

The kernels, in `tests/support/sibling_rows.rs`, are included by both the
probe and the golden, so the two measure one definition of each row. Each has
a `SUM` (or `MIN`) fold that varies with the column, so under a width with a
remainder it is carved into the main and the remainder column folds:

| row | kernel |
|---|---|
| `glyph_like_w1`, `_wL`, `_w37` | one fold over five pieces of a clamped, smoothstepped edge, with a row-invariant term and call-invariant constants read from the scopes outside; at widths 1, one batch (`L` = lanes of the tier) and 37 |
| `two_sibling_folds_w37` | two independent folds on one binder slot (a `SUM` over 6 and a `MIN` over 4), added |
| `parked_roots_w37` | one fold over 2,048 terms, each reading a row-invariant product and the constant it came from: 4,096 roots parked |
| `guarded_if_in_fold_w37` | a lane-varying `If` in a fold's body, each arm several transcendentals deep, so each is worth a branch |

Folds in the nest, besides the body (the row fold, the column folds and the
user's): `glyph_like_w1` and `_w37` have five, because the column fold is two
(a main and a remainder) and each carries a copy of the user's fold, so the
two copies are one `Reduce` carved twice. At width 1 the main fold is empty
(`[0, 0)`) and is still emitted, so a one-sample-wide kernel has siblings too.
`glyph_like_wL` has three folds and no sibling. `two_sibling_folds_w37` has
seven, two `Reduce` ids each carved twice; `parked_roots_w37` and
`guarded_if_in_fold_w37` have five. (The tables below count scopes, the body
included: 6, 4, 6, 8, 6, 6.)

## The byte rows

`cargo run --release -p pixelflow-codegen --example byte_probe`, once per
tier. Each new row prints the old line and then, indented, the compile's
counts and every scope's traffic; below is the digest, and the one block that
shows the shape (`glyph_like_w37`, AVX-512):

```text
glyph_like_w37           len=992    fnv=66e9f1ebf718d0cd spills=1 hoisted=13 jit_len=972    jit_fnv=269351306240f08a jit_branches=0/0/0
    compile: spill_count=1 spill_bytes=384 hoisted_values=13 max_regs=32 vector_bytes=64 pool=32
    traffic: carried=13 trailing=59 branches=0/0/0 scopes=6
    scaffold: instructions=0 loads_transient=0 loads_kept=0 remats=0 stores=0 writes=0 bytes=25
    scope 0: trips=1 instructions=12 loads_transient=0 loads_kept=0 remats=0 stores=0 writes=0 bytes=280
    scope 1: trips=3 instructions=3 loads_transient=0 loads_kept=0 remats=0 stores=0 writes=0 bytes=160
    scope 2: trips=6 instructions=6 loads_transient=1 loads_kept=0 remats=0 stores=1 writes=1 bytes=181
    scope 3: trips=30 instructions=8 loads_transient=0 loads_kept=0 remats=0 stores=0 writes=0 bytes=48
    scope 4: trips=3 instructions=6 loads_transient=1 loads_kept=0 remats=0 stores=1 writes=1 bytes=191
    scope 5: trips=15 instructions=8 loads_transient=0 loads_kept=0 remats=0 stores=0 writes=0 bytes=48
```

`compile` is the emitter over the arena as written; production is
`jit_cache::compile` (optimize, link, emit). `parked` is `hoisted_values`;
`m` is the frame's `spill_bytes`; `pool` is `max_regs`. `trips` are one call's
runs of each scope; scope 0 is the body, then the folds in nest order.

#### AVX-512

| row | `compile` len | fnv | production len | fnv | carried | parked | `m` B | pool | scopes | branches (compile / production) |
|---|--:|---|--:|---|--:|--:|--:|--:|--:|---|
| `glyph_like_w1` | 984 | `04d391d2df13c4d6` | 964 | `6d8eb50504687ffd` | 13 | 13 | 384 | 32 | 6 | 0/0/0 / 0/0/0 |
| `glyph_like_wL` | 664 | `f22ace44c2fda4c8` | 644 | `e071b92dabd8569d` | 13 | 13 | 256 | 32 | 4 | 0/0/0 / 0/0/0 |
| `glyph_like_w37` | 992 | `66e9f1ebf718d0cd` | 972 | `269351306240f08a` | 13 | 13 | 384 | 32 | 6 | 0/0/0 / 0/0/0 |
| `two_sibling_folds_w37` | 1056 | `662f9c26c2bbcafc` | 1092 | `d191d09588b84087` | 10 | 10 | 448 | 32 | 8 | 0/0/0 / 0/0/0 |
| `parked_roots_w37` | 278032 | `282aa77769f9e2ff` | 278000 | `1757b360e5b10d2f` | 23 | 4100 | 8640 | 32 | 6 | 0/0/0 / 0/0/0 |
| `guarded_if_in_fold_w37` | 2932 | `3943e837c9f115d3` | 3048 | `538e1fb8fdec599a` | 23 | 33 | 512 | 32 | 6 | 6/8/238 / 6/8/240 |

| row | trips per scope | bytes per scope (body first) | scaffold B | trailing B |
|---|---|---|--:|--:|
| `glyph_like_w1` | 1, 3, 3, 15, 0, 0 | 280, 152, 191, 48, 186, 59 | 25 | 43 |
| `glyph_like_wL` | 1, 3, 3, 15 | 280, 90, 181, 48 | 25 | 40 |
| `glyph_like_w37` | 1, 3, 6, 30, 3, 15 | 280, 160, 181, 48, 191, 48 | 25 | 59 |
| `two_sibling_folds_w37` | 1, 3, 6, 36, 24, 3, 18, 12 | 220, 154, 269, 12, 18, 279, 12, 18 | 25 | 49 |
| `parked_roots_w37` | 1, 3, 6, 18, 3, 9 | 43108, 57343, 195, 84472, 205, 84472 | 25 | 8212 |
| `guarded_if_in_fold_w37` | 1, 3, 6, 24, 3, 12 | 719, 224, 212, 691, 222, 691 | 25 | 148 |

#### AVX2

| row | `compile` len | fnv | production len | fnv | carried | parked | `m` B | pool | scopes | branches (compile / production) |
|---|--:|---|--:|---|--:|--:|--:|--:|--:|---|
| `glyph_like_w1` | 1016 | `46ec89671d0d59d7` | 932 | `12179306e928c52f` | 6 | 13 | 224 | 16 | 6 | 0/0/0 / 0/0/0 |
| `glyph_like_wL` | 728 | `4379d55663a9294e` | 644 | `8b593b945ef79c0c` | 6 | 13 | 160 | 16 | 4 | 0/0/0 / 0/0/0 |
| `glyph_like_w37` | 1056 | `f4f28a978e99b9ec` | 1004 | `16d1159df65e0531` | 6 | 13 | 224 | 16 | 6 | 0/0/0 / 0/0/0 |
| `two_sibling_folds_w37` | 1056 | `15c0a9e0e3472c74` | 1044 | `92bd795645d653f4` | 6 | 10 | 256 | 16 | 8 | 0/0/0 / 0/0/0 |
| `parked_roots_w37` | 247328 | `919cb6c0efe53a92` | 247296 | `e80af6a7b3a81fec` | 7 | 4100 | 4320 | 16 | 6 | 0/0/0 / 0/0/0 |
| `guarded_if_in_fold_w37` | 3012 | `90101b60330eb1ce` | 3096 | `a700acf0bceb99a4` | 7 | 33 | 288 | 16 | 6 | 6/8/238 / 6/8/240 |

| row | trips per scope | bytes per scope (body first) | scaffold B | trailing B |
|---|---|---|--:|--:|
| `glyph_like_w1` | 1, 3, 3, 15, 0, 0 | 255, 195, 209, 40, 197, 50 | 25 | 45 |
| `glyph_like_wL` | 1, 3, 3, 15 | 255, 147, 207, 40 | 25 | 54 |
| `glyph_like_w37` | 1, 3, 12, 60, 3, 15 | 295, 173, 197, 40, 233, 40 | 25 | 53 |
| `two_sibling_folds_w37` | 1, 3, 12, 72, 48, 3, 18, 12 | 233, 153, 249, 10, 15, 285, 10, 15 | 25 | 61 |
| `parked_roots_w37` | 1, 3, 12, 36, 3, 9 | 39049, 51343, 168, 74160, 204, 74160 | 25 | 8219 |
| `guarded_if_in_fold_w37` | 1, 3, 12, 48, 3, 12 | 697, 188, 183, 770, 219, 770 | 25 | 160 |

At width 1 the empty main fold shows as the two scopes with zero trips. The
`wL` width is the tier's own lane count, so a run on the other tier differs in
which rows have a remainder and in every number; the names do not.

The first nine lines of both tiers are unchanged from `e64a64ff`'s, which is
`28ddbeaf` on x86 by #1316 and #1317's own gates (below).

## The byte golden

`emit::tests::sibling_folds::GOLDEN`: `(length, FNV-1a 64 of the code bytes)`
for the same six rows on all three backends, each legalized at its own lane
count (AVX2 8, AVX-512 16, aarch64 4: stated once in the test and checked
against each backend's register file) and emitted with the default context.
Only bytes are generated, never run, so each backend emits on any host: this
is the check that puts aarch64's bytes under CI on a Linux x86 runner.
Recorded at `28ddbeaf`.

| row | AVX2 | AVX-512 | aarch64 |
|---|---|---|---|
| `glyph_like_w1` | 1016 `46ec89671d0d59d7` | 984 `04d391d2df13c4d6` | 592 `9eb350d1994f17af` |
| `glyph_like_wL` | 728 `4379d55663a9294e` | 664 `f22ace44c2fda4c8` | 400 `aa96d96596a05551` |
| `glyph_like_w37` | 1056 `f4f28a978e99b9ec` | 992 `66e9f1ebf718d0cd` | 608 `14a21aabd82fe2e8` |
| `two_sibling_folds_w37` | 1056 `15c0a9e0e3472c74` | 1056 `662f9c26c2bbcafc` | 656 `0e7016ed19878723` |
| `parked_roots_w37` | 247328 `919cb6c0efe53a92` | 278032 `282aa77769f9e2ff` | 188496 `578f9987a3386dcc` |
| `guarded_if_in_fold_w37` | 3012 `90101b60330eb1ce` | 2932 `3943e837c9f115d3` | 2144 `92246f5ac70b7ef7` |

The AVX2 and AVX-512 columns equal the probe's `compile` column on the
matching tier, to the hash: the probe legalizes at the host's lanes and the
golden at stated ones, and they are the same bytes.

Host-independence was checked, not assumed: the golden, the pin and the
allocator tests pass unchanged in dev and release, with `PIXELFLOW_ISA` unset,
`avx2` and `avx512`, and with `RUST_TEST_THREADS` default and `1`. Nothing in
it reads `isa::detect`, `PIXELFLOW_ISA`, `jit_vector_bytes` or the CPU. It
never edits itself: **an intentional byte change re-baselines `GOLDEN` in a
commit of its own**, and a failure prints the whole recomputed table to paste.

## The slot pin

`sibling_column_folds_share_a_reduce_and_its_slots`, on `glyph_like` at width
37 (AVX2, 8 lanes: a main fold and a remainder fold):

- the nest has exactly one `Reduce` carved into two folds, under different
  parents, neither inside the other (here folds 2 and 4, under 1 and 3);
- the driver, run unmodified under a backend that forwards and records every
  displacement it is handed, addresses the **later** fold's accumulator slot
  `m + 2j·vb` from both parents (and its binder slot `m + (2j + 1)·vb` where
  the allocator did not carry it) and the **earlier** fold's own slots from
  neither anywhere in the compile, at the floor pool (where both are in
  memory) and at the whole pool.

This is `fold_map`/`binder_map`'s `collect()` keeping the last fold for a
repeated `Reduce` id. It is accidental and byte-visible (it decides every
displacement above the first fold slot), and the series reproduces it by
`slot_loop` or changes it deliberately and separately.

## `alloc_probe`

A `#[global_allocator]` in the lib's test binary (`cfg(test)`, whole-file
gated), counting only the calling thread's allocations inside an explicit
`measure(|| ..)`:

- the tally is a thread-local `Cell` of a `Copy` struct, `const`-initialized
  and without a destructor, so reading it neither allocates nor registers a
  thread-exit callback (the two things an allocator cannot do), and an access
  from a thread already tearing down counts nothing instead of panicking;
- the tests that run concurrently on other threads are neither counted nor
  disturbed: it is per thread, and a scope that panics stops counting through
  a drop guard;
- `realloc` is the trait's default (a new block, a copy, a free), so a growing
  `Vec` is counted as the pair it is and the figure does not depend on whether
  the system allocator could extend in place; `alloc_zeroed` forwards to the
  system's, so a zeroed block stays lazily mapped and the probe does not touch
  its pages;
- `requested` is the sum of the layout sizes, which is the bytes asked for and
  an upper bound on the bytes zeroed; `peak` is the largest **net growth** of
  the live heap over the scope's start, the same definition
  `pixelflow_pipeline::alloc_probe` gives the bins.

One test measures a known allocation to the byte (`4096` requested, one
allocation, peak `4096`) while four other threads allocate and free 64 KiB
blocks throughout, started before the scope and joined after it; two more pin
the peak as a high-water mark and a panicking scope's cleanup.

A free of a block that predates the scope lowers the running figure like any
other free, so a scope that consumes a large input can under-report its peak by
up to that input's size. The `scope` and `allocate` stages below consume their
input (cloned outside, freed inside) and are read with that in mind.

## Heap, by stage

### `sibling_scopes`: F sibling folds, 8 values each

`cargo test --release -p pixelflow-codegen --lib sibling_scopes_allocation -- --ignored --nocapture`.
A lattice kernel with F independent surviving folds on one binder slot, each
over eight values that vary with the column, seeded by a constant of its own
and summed pairwise; one batch wide (no remainder), so F folds are F + 2
scopes. Deterministic: two runs agree to the byte.

Requested bytes / allocations / peak net bytes:

| F | defs | scopes | `schedule` (lowering) | `scope` (`from_schedule`) | `allocate` (`allocate_nest`) | `emit` (`compile_schedule`) |
|--:|--:|--:|---|---|---|---|
| 16 | 185 | 19 | 184,564 / 1,514 / 137,180 | 244,800 / 889 / 40,907 | 930,059 / 4,777 / 236,163 | 1,485,835 / 6,242 / 268,196 |
| 64 | 713 | 67 | 732,404 / 5,837 / 545,372 | 2,212,216 / 3,042 / 161,147 | 11,132,571 / 39,527 / 3,104,749 | 16,494,187 / 44,563 / 3,151,213 |
| 256 | 2,825 | 259 | 2,924,084 / 23,121 / 2,178,140 | 28,842,888 / 11,573 / 642,171 | 166,438,891 / 494,502 / 47,606,327 | 238,328,739 / 513,698 / 47,791,031 |
| 1024 | 11,273 | 1,027 | 11,688,500 / 92,243 / 8,709,212 | 435,401,568 / 45,594 / 2,566,011 | 2,610,687,507 / 7,368,527 / 753,975,457 | 3,707,843,715 / 7,444,197 / 754,713,121 |

`emit` is scoping, allocation and emission over the given schedule; its peak
is the frame-sized tables all alive together.

Growth per 4x of F (the control is lowering, which is linear):

| | 16 to 64 | 64 to 256 | 256 to 1024 |
|---|--:|--:|--:|
| `schedule` requested | 3.97 | 3.99 | 4.00 |
| `scope` requested | 9.04 | 13.04 | 15.10 |
| `allocate` requested | 11.97 | 14.95 | 15.69 |
| `allocate` peak | 13.15 | 15.33 | 15.84 |
| `emit` requested | 11.10 | 14.45 | 15.56 |
| `emit` peak | 11.75 | 15.17 | 15.79 |

Scoping and allocation are quadratic in the folds, approaching the 16x a
per-fold table the size of the whole program gives. At 1,024 folds the nest
asks the heap for 3.5 GiB and holds 720 MiB at once; at 256, 227 MiB and
46 MiB; at 64, 16 MiB and 3 MiB.

### The N-glyph rows

`pixelflow-pipeline/examples/glyph_program_report.rs`: the first N inked
glyphs of the printable range at core-term's 16 pt cell height, each the
production `Font::glyph_kernel_scaled` kernel at texel centres, a unit
(`Kernel::by_ref`), under the balanced `if id < k` tree over one uniform,
compiled as **one program** through `jit_cache::compile`. One count per
process, because the optimizer memoizes and `VmHWM` only rises; three
repeats, medians (the deterministic columns agreed on every repeat; the
memory figures varied by under 6%).

```text
for n in 4 8 16 32 94; do PIXELFLOW_PROBE_FONT=<noto.ttf> glyph_program_report $n; done
PIXELFLOW_PROBE_FONT=<noto.ttf> glyph_program_report 32 17   # a tile with a remainder
```

The font is a Git LFS object (a 131-byte pointer in a checkout without LFS),
so it is read at run time from `PIXELFLOW_PROBE_FONT`; the example refuses a
pointer file by name. The numbers below are the asset core-term ships, Noto
Sans Mono Regular: the file used has the pointer's own `oid`
(`sha256:11b5d661e57865bce89d3f103a654cfa4faf9347abebac252e30742be1e657b2`,
405,892 bytes). Tile 16 is a multiple of the lane count of both x86 tiers, so there is no
remainder column fold and each glyph's fold is carved once; tile 17 has one
and carves each glyph's fold into the main and the remainder fold, doubling
the folds. 94 is every inked glyph in `' '..='~'`; it completed in under two
seconds, so it is here.

Stages: `optimize` is `optimize_runtime_arena` cold; `compile` is
`jit_cache::compile` with the optimizer's answer memoized, so the canonical
key, the link and the emit (**the production bytes**); `emit` is
`emit::compile` alone over the optimized arena, for the counts only
`CompileResult` carries (not the production bytes: it skips the link's
renumbering). Each stage's peak is net growth over its own start, so `compile`
and `emit` start with the optimized arena already resident.

#### AVX-512

| N | tile | scopes | parked | frame B | code B | fnv | guards/arms/entries | optimize ms | compile ms | emit ms |
|--:|--:|--:|--:|--:|--:|---|---|--:|--:|--:|
| 4 | 16 | 7 | 83 | 2304 | 10808 | `f80b1bf2dec59b43` | 15/18/748 | 216 | 13 | 2 |
| 8 | 16 | 10 | 189 | 2176 | 21128 | `93934f5e61a3e725` | 32/39/2294 | 345 | 27 | 5 |
| 16 | 16 | 16 | 400 | 3968 | 49264 | `f41e54ab6051798e` | 66/81/5961 | 452 | 53 | 11 |
| **32** | **16** | 32 | 624 | 8128 | 91032 | `e2758f0ef47f0ff3` | 130/161/11151 | 377 | 99 | 25 |
| 94 | 16 | 91 | 1639 | 23488 | 261204 | `27132f5830c607dc` | 381/474/36268 | 564 | 345 | 156 |
| 32 | 17 | 62 | 624 | 8256 | 162952 | `befb8f87a71dee6e` | 260/322/22302 | 484 | 125 | 44 |
| 94 | 17 | 180 | 1639 | 23616 | 474308 | `cca3063edcabd17a` | 762/948/72536 | 556 | 519 | 283 |

| N | tile | optimize: requested / peak MiB | compile: requested / peak MiB | emit: requested / peak MiB | VmHWM after compile MiB |
|--:|--:|--|--|--|--:|
| 4 | 16 | 106.0 / 6.8 | 17.6 / 3.2 | 3.1 / 0.6 | 13.4 |
| 8 | 16 | 169.9 / 9.9 | 36.9 / 3.4 | 7.6 / 1.3 | 18.1 |
| 16 | 16 | 229.2 / 13.9 | 78.6 / 4.6 | 20.0 / 3.6 | 23.2 |
| **32** | **16** | 285.9 / 14.6 | **170.3 / 9.6** | 53.7 / 7.7 | **30.7** |
| 94 | 16 | 504.6 / 16.9 | 726.7 / 79.5 | 387.8 / 75.1 | 89.0 |
| 32 | 17 | 285.9 / 14.0 | 212.7 / 14.5 | 96.2 / 12.6 | 35.1 |
| 94 | 17 | 504.6 / 16.8 | 1083.1 / 147.8 | 744.2 / 143.5 | 137.2 |

#### AVX2

| N | tile | scopes | parked | frame B | code B | fnv | guards/arms/entries | optimize ms | compile ms | emit ms |
|--:|--:|--:|--:|--:|--:|---|---|--:|--:|--:|
| 4 | 16 | 7 | 83 | 1088 | 9832 | `97be8f8a58da24be` | 15/18/748 | 194 | 11 | 2 |
| 8 | 16 | 10 | 189 | 1344 | 22056 | `b1a3af9faf7187c5` | 32/39/2294 | 300 | 25 | 5 |
| 16 | 16 | 16 | 400 | 2688 | 45472 | `21a5c5373385cde8` | 66/81/5961 | 378 | 45 | 11 |
| **32** | **16** | 32 | 624 | 4768 | 83944 | `d0b7d1e36c532810` | 130/161/11151 | 474 | 104 | 25 |
| 94 | 16 | 91 | 1639 | 12448 | 238916 | `ae08af4f49e58426` | 381/474/36268 | 585 | 358 | 148 |
| 32 | 17 | 62 | 624 | 4832 | 150904 | `ffecf8e506b46b29` | 260/322/22302 | 426 | 126 | 63 |
| 94 | 17 | 180 | 1639 | 12512 | 435860 | `9a82cb6db5474ac7` | 762/948/72536 | 564 | 570 | 270 |

| N | tile | optimize: requested / peak MiB | compile: requested / peak MiB | emit: requested / peak MiB | VmHWM after compile MiB |
|--:|--:|--|--|--|--:|
| 4 | 16 | 106.0 / 7.6 | 17.6 / 3.2 | 3.0 / 0.6 | 13.4 |
| 8 | 16 | 169.9 / 9.8 | 36.6 / 3.4 | 7.3 / 1.3 | 17.9 |
| 16 | 16 | 229.2 / 13.8 | 77.8 / 4.6 | 19.2 / 3.6 | 23.1 |
| **32** | **16** | 285.9 / 14.5 | **168.6 / 9.5** | 52.1 / 7.7 | **30.8** |
| 94 | 16 | 504.6 / 16.8 | 722.0 / 79.5 | 383.1 / 75.2 | 90.6 |
| 32 | 17 | 285.9 / 14.0 | 209.7 / 14.4 | 93.1 / 12.6 | 34.9 |
| 94 | 17 | 504.6 / 16.8 | 1074.5 / 147.9 | 735.6 / 143.5 | 137.3 |

**Peak resident set** is `VmHWM` from `/proc/self/status`, read after the
`compile` stage, not `/usr/bin/time -v`, which this host does not have. It is
the kernel's resident high-water mark, which is what `time -v` prints as the
maximum resident set size, and `getrusage`'s `ru_maxrss` read from outside
agrees: 13.4 and 30.5 MiB for 4 and 32 glyphs (the whole process, the `emit`
stage included), and 97.5 MiB for 94, where the `emit` stage that runs after
the read lifts it from 92.7. **Heap
peak and bytes requested** come from `pixelflow_pipeline::alloc_probe` in the
example (`alloc_probe` in the codegen crate is `cfg(test)`, and an example is
a separate crate), counting every thread; the same three relaxed atomics are
on in every row, so times are comparable with each other and a few percent
above an uncounted build's.

Not printed, because an example cannot reach it without widening visibility
or reimplementing `jit_cache::compile`'s link step: the stages inside the
emitter (scoping, allocation, frames, emission) for the glyph programs. The
`sibling_scopes` table splits them for a fixture with the same shape.

## What this says about CL6b

**The 32-glyph row peaks at 9.6 MiB of heap growth in the production compile
(AVX-512; 9.5 MiB on AVX2), and the process at 30.7 MiB resident.** It asks
the heap for 170 MiB in all (the optimizer before it, 286 MiB). With a
remainder column fold (tile 17: 62 scopes) the compile peaks at 14.5 MiB and
the process at 35 MiB. The largest row measured, every inked glyph with a
remainder (94 glyphs, 180 scopes), peaks at 148 MiB of heap and 137 MiB
resident, and requests 1,083 MiB (1.06 GiB) over the compile.

The decision rule these numbers are read against was set while planning CL6
and is not in the tree, so it is stated here: 6b (about 2,300 mechanical
lines renaming every dense array's key to a scope-local id) is deferred when
peak is under about 1 GB on the 32-glyph row *and* the structural guarantee
(arrays sized by the scope, by construction) is not wanted. The 32-glyph row
is two orders of magnitude under that bar, at `28ddbeaf`, before 6a, with no
6b.

Our reading, which is the owner's to take or leave:

- **On memory alone, 6b does not clear the bar for the production glyph
  programs.** The "5 GB zeroing and 0.6 GB resident" that planning derived
  from reading the code are not what these programs ask: the worst row asks
  about 1.06 GiB and holds 148 MiB. The rule above defers 6b when peak is
  under 1 GB on the 32-glyph row *and* the structural guarantee is not
  wanted; the first conjunct holds today by 100x, and the second is a
  judgement about design, not measurement.
- **The scaling planning feared is real, and it is not these programs'.**
  Scoping and allocation are quadratic in the surviving folds (growth 15.1x
  to 15.8x per 4x between 256 and 1,024 folds), and a nest of 1,024 folds
  asks for 3.5 GiB and holds 720 MiB: the planning figures are about right at
  that scale. A 32-glyph font is 32 to 62 folds; a 94-glyph font 91 to 180.
  What the structural guarantee buys is that the next order of magnitude of
  folds does not cost 16x, not that today's font is in trouble.
- **Not measured.** The earlier "units font" rows (guard-structure baseline,
  2026-10-03: 599 pieces inlined at N=32, 959,948 B) are a different program:
  the pieces are straight-line terms and no fold survives, so it has three
  scopes of tens of thousands of defs each, not dozens of folds. It was built
  by a test-only `kernel!` copy that an example cannot include without a new
  dependency, so no figure here is for it. If the planning numbers came from
  it, they are for a shape this baseline does not cover.

## The gate, on this change

`gate.sh` (the byte probe, `glyph_compile_report`, the two pixelflow-graphics
byte tests and `glyph_branches`, on both tiers) run on this change and diffed
against `e64a64ff`, which is `28ddbeaf` on x86 by #1316's and #1317's own
gates:

| output | AVX-512 | AVX2 |
|---|---|---|
| `byte_probe`, the original nine rows | identical | identical |
| `byte_probe`, new rows | six rows (above), appended | six rows, appended |
| `glyph_compile_report` (191 lines) | identical | identical |
| `graphics_lib`, `glyph_branches` | identical | identical |
| stderr of both probes | identical | identical |

Tests and examples only: no `src/` line outside a `cfg(test)` module changed,
no visibility widened, no dependency added. Nothing in the diff changes an
emitted byte, and the gate says so.

## Reproducing

```text
cargo run --release -p pixelflow-codegen --example byte_probe                       # host tier
PIXELFLOW_ISA=avx2 cargo run --release -p pixelflow-codegen --example byte_probe    # the narrower
cargo test --release -p pixelflow-codegen --lib -- sibling_folds alloc_probe
cargo test --release -p pixelflow-codegen --lib sibling_scopes_allocation -- --ignored --nocapture
PIXELFLOW_PROBE_FONT=<noto.ttf> cargo run --release -p pixelflow-pipeline --example glyph_program_report -- 32
```
