# Guard structure baseline, 2026-10-03

What the emitter branches over, measured on the tree before the
`emit-should-just-emit` series moves any of it
(`docs/plans/2026-09-12-emit-should-just-emit.md`). Every later step of that
series is gated on these numbers being equal, per fixture, not on a total.

## What is counted

`EmitTraffic::branches` (`pixelflow-codegen/src/emit/traffic.rs`,
`BranchTraffic`), also on `CompiledKernel::branches()` for a compile that
went through `jit_cache::compile`:

| field | meaning |
|---|---|
| `guards` | `If`s with at least one arm under a branch |
| `arms_branched` | arms under a branch (a guard buys one or both of its two) |
| `arm_entries` | schedule entries under a branch, summed over those arms (an entry inside two nested arms counts in each) |

It is read off the tables the emitter branches on (`Allocation::if_guards`),
so it is what was emitted by construction, and nothing reads it back. A
guarded `If` and a blended one produce the same picture, so no pixel golden
sees a branch lost or gained. `guards` alone is a weak gate: chrome with
`cluster_if_arms` switched off still reads 3 guards, but 4 arms instead of 6.

## The table

| fixture | tier | guards | arms branched | entries |
|---|---|---:|---:|---:|
| `chrome_sphere_on_checker`, 1920x1080 | AVX-512 | 3 | 6 | 719 |
| `chrome_sphere_on_checker`, 1920x1080 | AVX2 | 3 | 6 | 719 |
| `sphere_silhouette_on_sky`, 1920x1080 | both | 0 | 0 | 0 |
| units font, N=4 glyphs at 16 px | AVX-512 | 77 | 80 | 15,886 |
| units font, N=8 | AVX-512 | 172 | 179 | 44,674 |
| units font, N=16 | AVX-512 | 263 | 278 | 79,708 |
| units font, N=32 | AVX-512 | 662 | 693 | 239,876 |

The units font rows are `id_tree` over the first N printable glyphs of Noto
Sans at 16 px, every glyph a unit with its pieces summed (the kernel the
whole-font plan emits); N=4, 8 and 16 were re-read today through
`CompiledKernel::branches` and agree with the earlier probe to the entry.

Emitted size and hash, the byte gate (`pixelflow_codegen::fnv1a64` over
`code_bytes()`):

| fixture | AVX-512 | AVX2 |
|---|---|---|
| chrome | 6,224 B `74b7beb5086e6818` | 6,192 B `1503af84be6a9e42` |
| silhouette | 1,132 B `1814649c472a84f7` | 1,212 B `49a82bb2932d271c` |
| units N=4 / 8 / 16 / 32 | 113,264 / 254,176 / 377,216 / 970,796 B | |
| units N=4 / 8 / 16 hash | `606f56720f5425e3` / `1f9c7e2545b6d81c` / `cb6474598a6bc925` | |

## What losing the branches costs

Measured with `cluster_if_arms` switched off (a probe, not in the tree):

| fixture | structure | time |
|---|---|---|
| chrome | 3 guards / 4 arms / 292 entries (was 3 / 6 / 719) | 3.5x and 3.2x slower on AVX-512 (1T, 4T); 2.6x on AVX2 |
| units font | 662 guards -> 0 | 1.9x to 3.7x slower per cell |
| 95-kernel atlas report | 112 guards -> 12; 94 of 95 hashes change | no measurable production cost |

Same pixels in every case. This is why the series is gated on the triple and
on time as well as on bytes.

## Pinned in CI

| pin | what it holds |
|---|---|
| `render::packed::tests::the_chrome_sphere_keeps_its_branches` | chrome, through `jit_cache::compile`: (3 guards, 6 arms) on every tier |
| `render::packed::tests::the_sphere_silhouette_branches_over_its_one_costly_arm` | silhouette: (1 guard, 1 arm) since the layout switch; (0, 0, 0) before it |
| `tests/glyph_branches.rs` | a glyph's coverage mask earns no branch (a branch there measured 3.6x slower) |
| `emit::tests::a_chrome_shaped_kernel_keeps_its_branches` | the same shape at the scale of one channel: (3, 6, 70), with a silhouette-shaped control (0, 0, 0) |

Entries are recorded here, not pinned: the count moves with every rewrite
rule, while an arm gained or lost is the failure the pins exist to catch.

## After the switch (layout in production, 2026-10-03)

`program::layout` chooses every scope's order and tables; `cluster_if_arms` is
no longer called. Same fixtures, same host, base and after built from the same
tree and run alternately (median of 3 rounds; the silhouette, whose bytes
moved by 64, median of 6 rounds of 61 frames).

**Structure (`EmitTraffic::branches`)**

| fixture | base | after |
|---|---|---|
| chrome 1080p, both tiers | (3, 6, 719), 6,224 B / 6,192 B | **identical bytes**, `74b7beb5086e6818` / `1503af84be6a9e42` |
| units font N=4 / 8 / 16 / 32 | (77, 80, 15,886) / (172, 179, 44,674) / (263, 278, 79,708) / (662, 693, 239,876) | **the same four triples** |
| units font bytes N=4 / 8 / 16 / 32, AVX-512 | 113,264 / 254,176 / 377,216 / 970,796 | 111,664 / 250,880 / 373,168 / 959,948 (-1.4% to -1.1%) |
| units font compile, N=32 | ~135 s (AVX-512) | **2.06 s** AVX-512, 1.89 s AVX2 |
| sphere silhouette | (0, 0, 0), 1,132 B / 1,212 B | (1, 1, 37), 1,196 B / 1,276 B |
| 95-kernel glyph table | 224 arms; 176 of 190 rows change | 576 arms |
| `byte_probe` | | sizes and spills identical on every row; hashes differ on `both_regions` (both tiers) and `row_invariant` (AVX2) |

The structure is what clustering produced wherever clustering produced it, and
the three additions are the ones the shadow predicted: the 352 glyph arms and
the silhouette's one.

**Time** (ns per pixel or texel; ratio = after / base, so under 1 is faster)

| fixture | AVX-512 | AVX2 |
|---|---|---|
| chrome 1T / 4T | 0.98 / 0.88 (same bytes: noise) | 0.98 / 0.98 |
| sphere silhouette 1T / 4T | 1.05 / 1.04 | 0.96 / 0.93 |
| glyph `@` `8` `O` at 16 px | 191 -> 35, 191 -> 30, 96 -> 18 | 284 -> 26, 282 -> 20, 143 -> 13 |
| glyph `@` `8` `O` at 32 px | 191 -> 15, 204 -> 12, 97 -> 8 | 281 -> 22, 280 -> 17, 140 -> 11 |
| all 95 glyphs at 16 px / 32 px | 74 -> 16 / 72 -> 7.0 | 103 -> 10.8 / 103 -> 9.6 |

The silhouette is the one fixture that moved the wrong way on one tier, by an
amount (4-5%) inside this host's run-to-run spread (the base's 1T runs span
1.10 to 1.44 ns/px) and of the opposite sign on AVX2. Its single new branch
guards 37 entries behind a spatially coherent mask. The glyph rows are the
352 fold-owning arms skipping a loop over a glyph's pieces outside its
bounding box, 5 to 16x faster per texel.

## How the switch was checked before it was made

The layout ran in shadow first: on every compile in a debug build (and in
release under the `layout-shadow` feature) the old analysis, run on the order
the layout chose, had to find exactly the runs the layout said, keep every arm
it guarded on the order it was given, and leave a scope unmoved when it
refused nothing for its order. It never disagreed, over the 95-kernel glyph
table on both tiers, chrome, the silhouette and the units font at N=4/8/16/32.
To make the order check non-vacuous it was also run on *unclustered*
schedules (clustering off in a scratch build), where it found the 537 arms
(N=4/8/16) and 693 arms (N=32) the search produced, from schedules in which
the old analysis found none, and chrome's 4 became 6.

The 352 extra glyph arms are all fold-owning arms the search left unguarded
because hoisting breaks their order: 2,216 to 35,549 cycles over 19 to 38
entries, against 201 to 497 cycles for the 224 arms emitted before. None under
200 cycles is realized, so no coverage-mask arm (a handful of ops, 3.6x slower
guarded) is.
