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
| `render::packed::tests::the_sphere_silhouette_earns_no_branch` | silhouette: (0, 0, 0) |
| `tests/glyph_branches.rs` | a glyph's coverage mask earns no branch (a branch there measured 3.6x slower) |
| `emit::tests::a_chrome_shaped_kernel_keeps_its_branches` | the same shape at the scale of one channel: (3, 6, 70), with a silhouette-shaped control (0, 0, 0) |

Entries are recorded here, not pinned: the count moves with every rewrite
rule, while an arm gained or lost is the failure the pins exist to catch.

## What the layout stage will change (measured in shadow, 2026-10-03)

`program::layout` chooses the order from ownership instead of repairing it, and
the old analysis checks it on every compile in a debug build (and under the
`layout-shadow` feature in release): it must find, in the laid-out order,
exactly the runs the layout says, keep every arm it guarded itself, and leave a
scope unmoved when it refused nothing for its order. It never disagreed. What
the layout *realizes*, against what is emitted today:

| fixture | emitted today | layout, on today's (clustered) schedule | layout, on the unclustered schedule |
|---|---|---|---|
| chrome (final scope) | 6 arms | 6 arms, order unmoved | 6 arms (clustering off: 4) |
| sphere silhouette | 0 arms | **1 arm** | 1 arm |
| units font N=4+8+16 | 537 arms | 537 arms, order unmoved | **537 arms** (clustering off: 0) |
| units font N=32 | 693 arms | | **693 arms** (clustering off: 0) |
| 95-kernel glyph table, both tiers | 224 arms | **576 arms** | 576 arms (clustering off: 24) |

The last column is the point: from a schedule nothing has repaired, the layout
finds exactly the arms `cluster_if_arms` produced, in the one pass, and the
units font at N=32 compiles in 3.0 s with the search off, against 135 s with it.

Two rows move what is emitted, so the switch is a measured change and not a
refactor:

- **The 95-kernel glyph table, 224 -> 576 arms.** All 352 additions are arms
  clustering left unguarded because their values are not one run after
  hoisting, and every one is a fold-owning arm: 2,216 to 35,549 cycles over 19
  to 38 entries (the old arms are 201 to 497 cycles over 52 to 130). No arm
  under 200 cycles is realized, so no coverage-mask arm (a handful of ops, 3.6x
  slower guarded) is. They skip a loop over a glyph's pieces when the pixel is
  outside the bounding box. Whether that wins is the glyph ns/texel gate's to
  say.
- **The sphere silhouette, 0 -> 1 arm.** One arm is over the bound and refused
  today for its order. The pin `the_sphere_silhouette_earns_no_branch` holds
  today's answer; it flips with the switch, and ns/px on this fixture decides
  whether the new answer stays.
