# A uniform read is one load

## Metadata
- **Author**: JP (direction), Claude (draft)
- **Status**: `Draft`
- **Created**: 2026-09-16
- **Verified against**: `bc53ea22` (main with #1268, the re-land of the surviving fold)
- **Continues**: [a-surviving-reduce-is-a-loop](2026-09-10-a-surviving-reduce-is-a-loop.md)
  (its "nested fold loops" remainder) and the emitter half of
  [one-name-bound-later](2026-09-10-one-name-bound-later.md).

**Decision it records (JP, 2026-09-16):**

> *"I think that the advantages of not having memory reads in the language
> outweigh denoting this. I think the proper place for gather is behind other
> kernels/refs … I don't think it's gonna be possible for pixelflow to be slow
> off of wearing out the ALU."*

So: nothing in this plan changes what the language can say. The glyph's
program is too big and too slow to compile for reasons that live in two
places — a fold that is still unrolled, and an emitter that spells a
lane-uniform read as a per-lane gather — and neither is a fact the algebra
needs to carry.

---

## 1. Why, with the numbers

`8` at tile 32, main at `04d537ba` (before #1268), the JIT for the AVX-512
tier, disassembled with `objdump -D -b binary -m i386:x86-64` and
histogrammed by mnemonic. 48,452 instructions, 354 KB, 2.1 s to compile in
release. Where they go:

| what | instructions | per piece (34) |
|---|---|---|
| table reads (`vgatherdps` + `kmovw` + `vcvttps2dq`) | 4,830 | 47 reads |
| address arithmetic on those reads (`vrndscaleps`, `vminps`/`vmaxps`, the `vmulps`/`vaddps` of `row·22 + col`) | ~13,000 | ~380 |
| constant materialisation (`mov` + `vbroadcastss`) | ~21,000 | ~620 |
| the Loop–Blinn arithmetic itself (`vfmadd`, `vsubps`, `vcmpps`, `vpternlogd`, `vandps`) | ~3,000 | ~90 |

The geometry is 6% of the program. Same shape at the SSE2 tier (573 KB,
508 spilled values), where each read is additionally 13 instructions of
scalar loads and inserts: 1,610 `vpextrd`/4 = 1,610 reads, 4,830
`vinsertps`/3 = the same 1,610, 3,220 `vroundps` = two per read, and so on —
the counts are internally consistent, so they are the reads.

Three facts, each with a different owner:

1. **The reads exist 34 times because the fold was unrolled.** After
   `expand_reduce` substitutes a literal for the binder, `expand_gather`
   builds `clamp(⌊row⌋)·width + clamp(⌊col⌋)` around it, and nothing after
   extraction folds anything — the pipeline unrolls *last* on purpose, so
   the e-graph's `ConstantFold` never sees these. `expand_reduce`'s own doc
   ([`pixelflow-ir/src/passes.rs`](../../pixelflow-ir/src/passes.rs)) says
   "the emitter folds their addresses to immediates". It does not, and never
   did.
2. **The read is emitted as a per-lane gather.** Every table read's row index
   depends only on the fold binder, so it is the same in every lane. The
   emitter has one lowering for `RawGather`
   ([`emit_gather_scalar`](../../pixelflow-codegen/src/emit/x86_64.rs) on
   SSE2, two halves of it on AVX2, `vgatherdps` on AVX-512, four scalar loads
   on aarch64), and one for a lane-uniform value
   ([`emit_uniform_load`](../../pixelflow-codegen/src/emit/x86_64.rs): a
   base-pointer `mov` and a `vbroadcastss`) that only `Uniform` leaves reach.
3. **Reads are never hoisted.** `plan_collapse_hoist`
   ([`pixelflow-codegen/src/emit/mod.rs`](../../pixelflow-codegen/src/emit/mod.rs))
   excludes gathers and everything computed from one, on the comment
   "winding kernels are gather-free". That was true before
   [glyph-as-a-fold-execution](2026-09-09-glyph-as-a-fold-execution.md) made
   the glyph two folds over a table. Now every coefficient read, and the
   `sqrt` of every piece's gradient norm, re-executes per batch of pixels
   though none of it depends on X or Y.

### 1a. The reading this plan corrects

The first reading of these numbers (2026-09-16, in review) proposed two
things, and JP refused both: a constant-fold-and-CSE stage after `legalize`,
and a `Row(buf, i, col)` node whose lowering would carry no floor and no
clamp because a fold binder is an integer in range by type.

Both mistook a *symptom of unrolling* for a property of reads. The address
arithmetic is a handful of vector ops per read; under a loop it runs per trip
and costs nothing the machine cannot spare. A post-extraction pass exists only
to clean up after lowerings that ignore what they were handed, and a second
optimizer after the e-graph is exactly the shape
[a-kept-structure-is-control-flow](2026-09-10-a-kept-structure-is-control-flow.md)
argues against. A memory-shaped node in the algebra, to save arithmetic,
leaks the thing [one-name-bound-later](2026-09-10-one-name-bound-later.md)
is removing. Neither survives contact with the decision above, and this
section is here so the next reader of the histogram does not re-derive them.

What does survive is the *runtime* reading: under loops, fact 2 is the
dominant cost of a trip. 47 reads at 13–52 instructions each is ~80× the
arithmetic around them, and it is memory-shaped work — loads, inserts, a
gather's dependent chain — not ALU.

---

## 2. What the language does not say, and why that is right

A `Kernel` has no read. A table is a `DiscreteManifold` whose `.kernel()`
is an opaque `Kernel`; reading it is `Kernel::at`, coordinate substitution;
`Buffer` and `Gather` are what `DiscreteManifold::kernel_for` commits to at
construction, and [one-name-bound-later](2026-09-10-one-name-bound-later.md)
says that commitment should move behind a name so extraction can choose the
lowering. Nothing here changes that direction or depends on it.

Two properties the language *does* already carry are what the emitter
should be reading:

- **Variance.** [`pixelflow-ir/src/variance.rs`](../../pixelflow-ir/src/variance.rs)
  computes, per node, whether it varies with X, with Y, or with neither. A
  batch's lanes differ only in X, so "X-invariant" *is* "the same in every
  lane". A read whose index is X-invariant denotes one value, broadcast.
- **A fold's binder is in-range by construction.** `Fold::range()` bounds
  it. A read of a bound table at such an index cannot fault. That is the
  only property hoisting needs, and it is the one `Uniform` loads are
  hoisted on today.

Both are facts the emitter can read off the DAG it is handed. Neither asks
the front end to say anything new.

---

## 3. The three lowerings, and where each lives

### 3.1 Nested fold loops — the compile-time story

Owner: [a-surviving-reduce-is-a-loop](2026-09-10-a-surviving-reduce-is-a-loop.md),
its stated remainder. `extract_folds` carves one level, so the distance fold
is a loop whose body still contains the winding fold unrolled 34 times. The
34 copies of fact 1's arithmetic are those copies. With both folds loops,
the glyph program is two bodies of roughly a thousand entries regardless of
piece count, `expand_reduce` unrolls nothing on the production path, and the
compile stops scaling with the outline.

This is the whole of the compile-time story. The 2 s was `emit` at
~O(n^1.6) in an instruction count that was 34× larger than the program;
make the program the body and the count is the body's.

What to measure: `glyph_compile_report` in warm mode before and after, and
the histogram above on `8`@32 — the read count per glyph should fall from
1,610 to the ~47 of one body, and the `vrndscaleps`/`vminps`/`vmaxps` rows
with it.

### 3.2 A lane-uniform read is one load — the runtime story

Owner: the emitter, per ISA. Where the schedule reaches a `RawGather` whose
index value is X-invariant (variance is already computed for hoisting; this
is the same table read one more time), emit it as `emit_uniform_load` does:

- index a literal (the unrolled case that remains, and every fold that
  `ExpandNestedReduce` still unrolls): `vbroadcastss dst, [base + 4·k]`, the
  address an immediate;
- index a fold binder or anything else X-invariant: one lane of the index
  into a GPR (`vmovd`/`vcvttss2si`), then `vbroadcastss dst, [base + gpr·4]`.

The per-lane sequence stays for what it is for: an index that varies across
lanes, which is what `DiscreteManifold::kernel_for` produces when a
tabulated kernel is read at `(X, Y)`. This is instruction selection keyed on
an existing analysis, in `x86_64.rs`, `avx2.rs`, `avx512.rs` and
`aarch64.rs`; the driver in `mod.rs` needs to hand the backend the variance
bit, or split `ScheduledOp::Gather` into the two cases at
`arena_to_schedule` time. The second is cleaner: the case is decided once,
where the DAG is read, and every backend gets it as a different op rather
than a flag ("fold before you dispatch").

`ISA matrix` covers every tier; the goldens are the gate, since a broadcast
of the right value is bit-identical to a gather of it.

### 3.3 X-invariant reads hoist — the other runtime story

Owner: `plan_collapse_hoist` in `mod.rs`. Delete the gather exclusion and the
`contains_gather` walk that implements it. The comment's reason was
speculation moving a read out of a select-guard arm; a read of a bound table
at an in-range index cannot fault, so it hoists on the same grounds a
`Uniform` load already does. What a fold's body reads through its binder
hoists to the fold's own scope head, not out of the fold — the existing
scope machinery from 2b already places values per scope.

This also retires H4's "hoist binder-only work out of the pixel loop" as a
separate item: it was this, seen from the other side.

### 3.4 Duplicate reads — not a lowering

47 reads per piece for 22 columns is the two bodies' copies of the same
column never meeting. Under loops that is 47 loads per trip, all L1 hits;
it is not worth a pass. H3 (the hash-consed arena) folds them by
construction when it lands, and that is the only mechanism this plan wants
for it.

---

## 4. What this plan does not do

- **No pass after the e-graph.** Legalize-last stays; nothing folds or
  merges after extraction. If a lowering produces work it could have known
  was dead, the fix is in the lowering's inputs (the variance bit) or in the
  structure it destroyed (the loop), never in a stage behind it.
- **No new node.** No `Row`, no typed index, no integer binder. A fold's
  binder stays a `Var`; the range that makes its reads safe stays on the
  `Fold`.
- **No change to `Gather`'s meaning.** `buf[clamp(⌊y⌋)][clamp(⌊x⌋)]`, as
  `DiscreteManifold::eval` computes it. The floor and clamps survive under a
  loop as a few vector ops per trip, and that is fine.
- **No bucketing changes.** #1270's padding is orthogonal; a padded row is
  a read like any other.

---

## 5. Order, and how each step is measured

| step | owner | gate | number that should move |
|---|---|---|---|
| nested fold loops | surviving-reduce plan | goldens; `run_is_a_glyph`; `font_rasterization_regression` at every ISA level | reads per glyph 1,610 → ~47; compile time; atlas bytes |
| uniform read → broadcast load | emitter, per ISA | goldens at every level; `avx512_evex_proof` for the new encodings | `vgatherdps`/`vpextrd`/`vinsertps` rows → 0 on glyph programs; collapse time |
| reads hoist | `plan_collapse_hoist` | goldens; `traffic` counts (`loads_kept` moves from body to row/frame scope) | collapse time; body instruction count |

The disassembly histogram is the instrument for all three: compile the
glyph through `compile_as_baked` (as
[`glyph_compile_report`](../../pixelflow-pipeline/examples/glyph_compile_report.rs)
does), write `code.as_bytes()` to a file, and count mnemonics. It is a
twenty-line scratch example, not a tool to keep; the numbers in §1 are what
it printed.

---

## 6. Open questions

- **Do both folds bind the same `Var`?** If the winding and distance bodies
  use the same binder index, the e-graph already hash-conses their shared
  column reads into one e-class, and the duplication in §3.4 arises only at
  unrolling; if not, it arises earlier. Either way §3.4's answer is H3, but
  the count the histogram reports after 3.1 depends on which.
- **How does `variance` classify a read indexed by a binder inside its
  scope?** The hoist in 3.3 needs "X-invariant" to be true of such a read
  while it is still binder-dependent — invariant across the batch, varying
  across trips. If variance is a flat lattice over the coordinate axes, the
  binder needs a bit of its own (the scope machinery from 2b may already
  supply this).
- **Which loop is outer?** With both folds loops, a glyph is `for piece {
  for batch { … } }` or `for batch { for piece { … } }`. The reads are
  invariant across batches and vary across pieces, so the first order makes
  every read a scope-head load and the second makes it a per-trip broadcast
  from L1. [glyph-as-a-fold-execution](2026-09-09-glyph-as-a-fold-execution.md)
  §S2 is the same question from the domain side. Not decided here; 3.2 is
  correct under either.
