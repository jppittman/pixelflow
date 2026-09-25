# One pipeline

## Metadata
- **Author**: JP (direction), Claude (draft)
- **Status**: `Proposed`. Amended 2026-09-25 by
  [the-language-is-kernel](2026-09-25-the-language-is-kernel.md): the front end
  (§3.2), the build-time entry, CL4–CL5, Q1 and Q4 are superseded there.
  After JP's ruling that there are no tables, this plan's tables backed by
  uniforms are superseded too: §1.3, A8, CL8 and Q2. So are its program per
  control-point count and its frame (§1.6, Q3). A font is one program per zoom
  level, choosing among glyphs with `if` (the-language-is-kernel §1.6–§1.8).
- **Created**: 2026-09-24; revised 2026-09-25
- **Verified against**: `8b7b75a`. The inputs were:
  - four maps (optimizer, front end, backend and artifact, bound buffers);
  - one measurement of today's glyph fold (**F**) against N instances written
    at construction over scalar uniforms (**U_band**);
  - two reviews of the first draft;
  - after JP's corrections of 2026-09-25, three reports, each checked by an
    independent skeptic:
    - where unrolling lives;
    - where the lattice's loops differ from folds;
    - an experiment that unrolls F through the e-graph's own rules (**U_g**);
  - three reviews of the second draft (alignment, facts, design), one of
    which measured the extractor's two arms.

  NotoSansMono and DejaVuSansMono were used, at 16 and 32 px, on AVX-512 and
  `PIXELFLOW_ISA=avx2`. Every scratch file is deleted, and nothing was
  committed.
- **Amends**:
  - [macro-tier-is-arena-native](2026-09-08-macro-tier-is-arena-native.md)
    steps 4–5. `ENode::Param` and the macro's `Saturate` are deleted, not
    completed.
  - [one-name-bound-later](2026-09-10-one-name-bound-later.md), executed for
    the glyph's table (§1.3).
- **Continues**:
  - [composition-is-linking](2026-09-09-composition-is-linking.md)
  - [the-isa-is-decided-at-startup](2026-09-22-the-isa-is-decided-at-startup.md)
  - [schedule-cost-model-denotation](2026-09-01-schedule-cost-model-denotation.md).
    §1.4's under-pricing is its first measured customer.
  - [demand-is-a-dag-property](2026-09-07-demand-is-a-dag-property.md). §1.4
    measures D1 as the next speed step.
- **Depends on, and does not do**:
  - [exprarena-on-dag](2026-09-09-exprarena-on-dag.md) Stage D. Every
    `Kernel` holds its graph twice (§3.2).
- **Leaves to an unwritten successor plan**: the gaps L3–L7 (§1.5), the
  frame's two remaining buffers (§1.3, Q3), and Q6.

**Convention.** **F** marks a fact: read in code or git history, or measured
at `8b7b75a`. **I** marks an inference.

**JP's words (verbatim):**

> *"It's literally supposed to be the same syntax. The same backend. You just
> run the compilation as a macro when you compile the program."* And:
> *"Don't reimplement the jit machinery. We're not maintaining this twice.
> … No separate rule set. Literally all the exact same code."*
>
> *"Bound buffers need to go. That's not supposed to be a thing."*
>
> *"We don't have uniform arrays.. we have ranges and folds."* And: *"And we
> have other kernels."*
>
> *"It has no runtime allocation of memory. It's declarative/functional."*
> And: *"the lattice is part of the schedule."*
>
> *"The 'atlas' becomes the kernel for that number of control points,
> everything else is a uniform."*
>
> 2026-09-25:
>
> *"I'm not anti unrolling. I'm anti bespoke unrolling pass. I thought we had
> unrolling in the egraph. The one that doubles loop bodies and halves their
> iterations?"*
>
> *"And I'm also anti, 'the schedule loops are different than folds'."*
>
> *"And what do you mean the table becomes like a uniform. It is a uniform
> no? The rule is that ranges have to be constant. The programs runtime
> knowable at compile time (we may relax this at some point and allow
> uniform iterations and allow runtimes that are polynomial functions of
> input, but that's like no where near the docket for now)."*

**Decisions those words settle:**

- **Unrolling is the e-graph's `HalveFold` family.** It is never a pass and
  never done at construction.
- **Every range is constant.**
- **One program per control-point count: exact N, not buckets** ("the kernel
  for that number of control points").
- **A bundled glyph's control points are uniforms** (rodata at build time),
  not constants ("everything else is a uniform").
- **The build-time entry is the macro.** An earlier revision said a build
  script, reasoning that a compiling proc macro is a second front end. That
  held only while production was written on the builder; with `kernel!` as the
  one language, the macro runs the one parser and `P` (the-language-is-kernel
  §1.8).

**What the corrections established (F, measured independently three
times).** Unroll F's own e-graph with the rules already in it, and the result
is U, which is 2.0–2.2× faster than F on AVX-512 at 16 px and 4.5–10.8× at
32 px or on AVX2. No construction-time unrolling and no bespoke pass are
involved. The one integral in the fold's body closes once, and `HalveFold`
copies the closed form.

Production misses it for two reasons, one of budget and one of price. The
fixes are one more phase of the same rule set and one price for every fold
(§1.4). Nobody chooses between "fold" and "unrolled": the author writes the
fold, and extraction decides.

---

## 1. The denotation

### 1.1 One function

```text
P : Kernel × LatticeShape × Isa → (Bytes, Link)

P(k, s, t) = emit_t ∘ legalize_t ∘ extract_s ∘ saturate_R ∘ insert ∘ canonical ∘ expand_refs  (k)

saturate_R = folds_R ∘ main_(R∖folds) ∘ closing_R        one rule set, three phases
```

- **One rule set, one vocabulary.** `R` is today's `RuleSet::runtime()`, which
  becomes the only set. Its phases are subsets of it, not other sets.
  - F: `closing_R` exists today (`graph.rs:1426-1450`).
  - `folds_R` moves the fold rules out of the main phase and runs them after
    it (§1.4).
- **All three phases are shape-free** and cached by structure
  (`runtime.rs:305-312`).
- **Extraction reads `s`.** It prices every fold by its trips (§1.4).
- **Legalization and emission read `t`.**
  - Legalization reads it through `pack`'s lane count (`native_register_file`
    → `detect()`, `emit/mod.rs:1163`).
  - Emission reads `s` as well: fold bounds, remainder arms, carry ranking
    (§4).
- **Gap L4 (Q6).** `legalize_t` builds the lattice's folds after
  extraction, where no rule reaches them. The target moves them before the
  fold phase:

  ```text
  P(k, s, t) = emit_t ∘ extract ∘ folds_R ∘ lattice_(s,t) ∘ main ∘ closing ∘ insert ∘ canonical ∘ expand_refs  (k)
  ```

  There the fold phase reads `s`, and reads `t` through `L`. §1.5 says why
  it does not exist yet.
- **The JIT is `P` run at runtime; the build-time compile is `P` run at
  build time.** Its bytes are embedded, and at load they go into the same
  cache under the same key (§3.7, A4).

**The law CI enforces:**

```text
∀ k, s, t.   bytes(P_build(k, s, t)) = bytes(P_run(k, s, t))
```

This holds by construction exactly when `P` depends on nothing but
`(k, s, t)`. **F: today it depends on seven other things.**

1. **The tier is read inside the emitter.** Both `compile_native` and
   `native_register_file` call `isa::detect()` (`emit/mod.rs:3875`, `:3887`).
2. **The bytes are mapped inside the emitter**, by
   `ExecutableCode::from_code` (`emit/mod.rs:4278`).
3. **The macro saturates, then the JIT saturates again**
   (`pixelflow-compiler/src/lib.rs:163-165`; `jit_cache.rs:145`). The same
   kernel gives different bytes in 4 of 7 cases. The runtime optimizer is
   idempotent; the macro hands it a different starting term.
4. **A process-global `KernelStore` resolves `Guard` arms and `Ref`s**
   (`emit/mod.rs:4060-4066`; `passes.rs:318`; `variance.rs:312`).
   - No production code builds a `Guard` or calls `by_ref`.
   - A `Ref`'s key digests minted identities (`key.rs:112-118`).
5. **`PIXELFLOW_SATURATION`.** Under `saturation-switch` it skips saturation
   (`runtime.rs:123`, `:141`, `:151-157`, `:388-424`). Its only consumer
   went in #1235, and CI's `test` job runs `--all-features`
   (`rust.yaml:523`, `:530`).
6. **Two silent fallbacks, and a third that is silent in effect.**
   - A declined term is emitted as given (`jit_cache.rs:145-149`).
   - A failed post-extraction `resolve` becomes `None` through `.ok()`
     (`runtime.rs:259`).
   - Any integral that reaches `resolve` is quadratured
     (`passes.rs:598-617`, `:672-686`) into N weighted samples. That is an
     approximation, and it has already shipped once as "point-sampled,
     aliased glyphs" (`loop_blinn.rs:333-337`).
7. **`canonical` numbers nodes in arena order** (`key.rs:157-163`). That is
   sound: it misses sharing and never shares wrongly. The build side and the
   run side construct identically.

`P` also repeats stages. The `Ref` guard is written three times
(`jit_cache.rs:112-119`, `runtime.rs:165`, `passes.rs:2257-2264`), and
`resolve` runs twice (`runtime.rs:259`; `passes.rs:106`). M11 leaves one of
each.

**F: what is already pure.**

- **The optimizer on one host.** Its output is bit-identical across 69
  processes.
  - Across hosts it is an **I**: `.cargo/config.toml:18-20` sets
    `-fp-contract=fast`, and no compile-time float site is known to be
    affected.
  - CL3's cross-runner check settles it.
- **The build profile.** Debug and release builds emit identical bytes.
- **x86 placement.** The code is position-independent, with 0 relocations.
- **Cross-emission.** Every backend emits on every host, but only from a
  `pub(crate)` test (`emit/mod.rs:5178-5215`). C1 changes that.

### 1.2 Compose, then compile: a stage, not a tier

```text
Kernel ──at, select, over, area, arithmetic──▶ Kernel     open: no optimizer; Dwrt, folds and integrals kept
Kernel ──────────────P(·, s, t)──────────────▶ program    closed: the only place anything is optimized
```

- **A `kernel!` expansion and a combinator produce the same thing: an open
  value.** Neither optimizes. F: `kernel!` has no production call site; its
  one user outside its crate is `pixelflow-graphics/tests/optimization_fuzz.rs`.
- **The chain rule survives composition because nothing optimizes an open
  value.** That is why `DwrtFree` exists (`pixelflow-compiler/src/lib.rs:167-196`).
  It becomes a fact about `P`'s signature, and `derivative_under_warp.rs` stays
  as the test of the stage.
- **With one optimizer that runs only at compile, there is no tier.**
- **The same holds for unrolling.** A fold is written once and stays a fold
  in every open value. Whether it runs as a loop or as copies is decided
  inside `P`, by extraction. The two exceptions today are:
  - `pack`, the lattice's strip-mine, which is a pass (L2);
  - `quadrature`, an approximation that M16 makes loud.

### 1.3 Parameters are uniforms; a table is another kernel

```text
⟦P(k, s, t)⟧ : f32^m → (L_s → f32)        m fixed by the program
```

- **`L_s` is the lattice.** It is part of the program (§4), not an argument.
- **Every parameter is a scalar uniform.** The block `f32^m` holds them.
- **A table is another kernel** (*"we have other kernels"*), a kernel on a
  finite lattice. The glyph's table already is one: a kernel on
  `Fin 10 × Fin N`, read by `at` at `(k, i)`
  (`DiscreteManifold::new(rows).kernel()`, `loop_blinn.rs:379`, `:592-593`).
  This plan changes two things about it:
  - **what backs it.** Its samples become 10N scalar uniforms instead of a
    bound buffer;
  - **its read point.** An index built from the fold's binder, `Const`, `Mul`
    and `Add`, which sema proves affine and in range. The-language-is-kernel
    §1.6 decides a child expression over a typed field, and says why.
- **Every range is constant, so every read point's range is known at
  construction.** A read outside the table is refused there, and nothing is
  clamped at runtime.
- **A fully unrolled fold's read points are constants.** Each read is then a
  scalar uniform at a static offset.
- **Named plainly.** A family of constant size, read only at points affine in
  constant-range binders, is what GLSL ES 1.00 calls a uniform array indexed
  by constant-index expressions.
  - "No runtime length, no data-dependent index" is the restriction JP
    stated.
  - Whether "we don't have uniform arrays" still excludes it is Q2.
  - If it does, the glyph cannot leave `Gather` except by instances written
    at construction, and those fail to close from N ≥ 37 (Appendix A).

**What it costs.** The-language-is-kernel §1.6: with the index a child
expression, every existing walk (substitution, variance, `FactorFold`,
placement) sees its binder, and the emitter lowers it as today's broadcast
load. A typed index field would have hidden the binder from all of them and
produced plausible pixels and the wrong glyph.

**What it replaces, for the glyph (F).**

- The glyph's table is a `Buffer` read with `Gather`:
  - bound through `Manifold::bind` (`manifold.rs:419`);
  - carried in a side table (`kernel.rs:218-231`);
  - copied on every `bind(&[])` (`manifold.rs:427`);
  - indexed by an integer in an `f32` lane, guarded by
    `EXACT_F32_INDEX = 1 << 24` (`manifold.rs:315-323`, `:457`).
- Even at a constant index, each read costs something. On U_g for `'8'` it
  runs an index add, a `vcvttss2si` and a `vbroadcastss`, then a spill:
  about 2,500 of 4,347 per-call instructions, which is under 2% of the call.
- **The reason for the change is the type, not the speed.**

**What it does not replace: the frame's two reads, which are indexed by
data.**

- **The cell grid's `cells.at(⌊x·cells_per_point⌋, row)`**
  (`lattice/cell_grid.rs:440-441`, `:456`).
  - A fold over cells makes this read binder-indexed.
  - Under JP's constant-range rule, the cell metric is then compile-time,
    so a zoom recompiles, once per level, through the cache.
  - That overrides `cell_grid.rs:18-36`'s reason for making the metric a
    uniform, and needs JP's confirmation (Q3).
- **The atlas's bilinear taps at `u0 + lx·density`**
  (`cell_grid.rs:458`, `:469-472`).
  - `u0` is cell data, so a fold over cells does not remove this read.
  - Without an atlas, which glyph a cell shows is cell data too. The
    font's tables are small (66,880 B for DejaVu's 94 ASCII glyphs,
    padded); what is data is which table.
  - So the frame needs one of two things: one data-indexed read (glyph →
    table, or the memo), or a dispatch over every glyph's program. Q3.

### 1.4 The glyph: one fold; unrolling is extraction's

```text
P_N(c, b)(X, Y) = [(X, Y) ∈ box(b)] · snap(min(|Σ_{i<N} t(T(·, i))(X, Y)|, 1))
t(p)(X, Y)      = select(rows_lo(p) < Y < rows_hi(p),  σ(p)·area(χ_p).at(X, S(p)·Y),  0)
T : Fin 10 × Fin N → f32, backed by 10N uniforms;   b: 4 uniforms
```

The author writes this once. It is today's program with its table backed by
uniforms, and `N` is the glyph's exact piece count.

**U is one extraction of this fold (F).** Three harnesses built production F
and saturated its own e-graph: first the closing phase, then the fold rules
with `ConstantFold`, each to a fixpoint.

- **The root class holds a term with no range fold and no integral.** Every
  table read in it is at a constant index: 160 of 160 for `'A'`, 640 of 640
  for `'8'`.
- **Nothing is closed twice.**
  - `ArcMoment` fires once.
  - Halving copies only closed classes: `Copying::ClosedOnly`
    (`fold_rules.rs:301`), through `representative`, the first
    non-integral node (`:357-363`).
  - "Close once, instance N" is what the fold already does. The first
    draft's fix (b) is withdrawn (Appendix A).
- **Accuracy:** every form is within 8.3e-7 of production F.
- **The phase goes after main, not before.** A design review measured both
  orders. They give the same U. But a fold that extraction keeps loses the
  main phase's algebra if the fold phase runs first: F's executed cost rises
  9.8% for `'8'`, from 8,220,416 to 9,023,232. Main run again afterwards
  recovers nothing: it hits the cap in 2 rounds.

**What U_g buys.** Warm µs, medians. F varies ±40% between runs and U_g up to
30%, so read ratios:

| tier, px | glyph (N / bucket) | F | U_g | F at exact N | U_g at exact N |
|---|---|---|---|---|---|
| AVX-512, 16 | A (12/16) | 11.3 | 5.5 | 8.5 | 4.2 |
| | 8 (43/64) | 33.6 | 15.5 | 22.5 | 10.1 |
| AVX-512, 32 | A | 46.1 | 10.3 | 35.1 | 8.0 |
| | 8 | 191 | 37.0 | 130 | 24.8 |
| AVX2, 16 | A | 12.7 | 2.67 | 9.6 | 2.13 |
| | 8 | 51.3 | 9.2 | 34.7 | 6.3 |
| AVX2, 32 | 8 | 206 | 19.0 | 133 | 13.0 |

- **Code:** F is 2.88 KB. U_g is about 1.75 KB per piece: 29 / 57 / 112 KB
  at 16 / 32 / 64 pieces.
- **Emit:** F takes 0.8–1.0 ms. U_g takes 13 / 31–43 / 100–153 ms.
- **Exact N against padding:** 1.1–1.55× faster, growing with the padding.

**Why production does not get there (F).**

1. **Budget.**
   - Production stops at `ClassCap` (5,000) in round 5, after 6,478
     applications. `HalveFold` fires 5 times, and strides reach only 4.
   - The unroll alone needs 5,662 / 10,724 / 20,843 classes (16 / 32 / 64
     pieces). An e-graph keeps every intermediate stride.
   - The cap's ceiling equals its floor (`saturate.rs:227`, `:243`), and a
     glyph inserts about 102 classes whatever its size.
2. **Pricing.**
   - The arbiter chooses between the extractor's two arms on the `dag`
     column (`extract.rs:1976`). `dag` adds each node's own cost with no
     trip multiplier (`:1880`; `SharedPricer`, `:2781-2790`).
   - `evals` prices a node that reads a binder as per-sample, and "the
     fold's trip count is not in the weight at all" (`variance.rs:494-502`).
   - So a fold's body costs one trip, and F is under-priced 12× (`'A'`) and
     38× (`'8'`).
   - `cost.rs:409-411` says "the DP multiplies". Only the tree DP does
     (`extract.rs:1868-1878`), and it is not the one that decides.

**The fixes: one more phase, one price.**

- **The fold phase (M15).**
  - It runs after main, with the fold rules taken out of main. It uses the
    fold rules plus the closing family, to a fixpoint.
  - It admits whole folds, smallest predicted full-unroll growth first,
    while the total fits under `HARD_CLASS_LIMIT` (100k, `graph.rs:512`).
  - A fold that is not admitted is not halved.
  - The growth bound is `4 · trips · spine`, where the spine is the body
    classes whose variance fact holds the binder. **I:** the 4 is the
    measured 3.1–3.3× over the unrolled term.
  - Per application, growth is already predicted exactly
    (`graph.rs:2138`, asserted at `:2049-2076`).
  - This is deterministic, and it is a size limit. **F:** a 10-character
    run needs 26,149 classes and a 25-character run 62,844. **I:** a
    94-glyph `text()` run needs about 230–280k, so admission decides there.
    `text()` has no production caller.
- **One price for every fold (M14).**
  - The arbiter prices each arm's settled term by what codegen executes,
    with one rule for kernel and lattice folds alike. That rule is `evals`
    with kernel folds placed in the nest:
    - a node that reads no binder counts `evals(variance)`;
    - any other node counts the executions of the innermost fold whose
      binder it reads, times that fold's trips;
    - each class is visited once per fold instance that reaches it.
  - **Why once per instance.** A text run's one body class sits under 9–21
    folds (`loop_blinn.rs:331-338`, `:381-392`), and codegen emits the body
    inside each (`emit/mod.rs:3378-3423`). Visiting each class once
    undercounts it 5.7–14.7×.
  - **Measured.**
    - On the phased graph, this price picks the tree arm's term, which is
      U_g exactly: fold-free, no integral, every read constant. It is 2.05×
      cheaper.
    - On today's production graph it still picks F, so M14 alone is
      glyph-neutral.
  - **Where the price is inexact.** It counts operations: it prices both arms
    of a guarded select (modelled 1.9–2.1×, measured 2.1–5.6×) and ignores
    spills.
  - **Why the price is not per class inside the DP.** `PeelFold` keeps a
    body class, and `HalveFold` nests it under a new fold
    (`fold_rules.rs:530`, `:578-579`), so one class sits under folds of
    different lengths. That is schedule-cost-model-denotation's
    `(class, level)` DP, and this plan does not build it.
  - **A guard this needs.** On production graphs the tree arm's term keeps
    an open integral: 1 each for `'A'`, `'8'` and two runs. The arbiter
    must never choose a term that holds one (CL9).
- **No price for code in this plan.** The design review measured the
  recommended per-call fetch price unsound:
  - the idle-minus-warm fit varies 2.7× with pixel size, which does not
    change bytes, so it measures executed work, not fetch;
  - how many calls a residency serves belongs to the schedule;
  - a host-measured constant would break build = run.

  Admission already bounds a kernel's unroll at about 100k classes, which is
  about 300 pieces and about 0.5 MB (**I**). Every program measured up to
  272 KB ran faster unrolled: warm always, and idle for 87–93 of 93 ASCII
  glyphs and 10 of 10 large ones (U_band). Reopen this when a production
  kernel's unrolled code outgrows L2, or a measurement shows unrolled code
  losing idle.

**What unrolling does not fix (F, per-address counts, `'8'`, AVX-512,
32 px).**

- **Row-tier work is 79% of U_g's executed instructions:** 109 of 138.5 per
  pixel.
  - It is each piece's `(i, Y)` work, run for every piece on every row.
  - Placement lifted it above the band select, into the row fold, where
    nothing guards it.
  - The band select does jump: 12,311 of 12,314 exclusive values are
    guarded, against 0 of 107 in F. But the arms it skips run on only 2–4
    of 64 batches.
- **The band condition depends only on `(i, Y)`**, so its guard belongs at
  row scope, around the piece's row work too.
  - That is D1: placement that reads demand (an-integral-is-a-fold §4;
    demand-is-a-dag-property).
  - For `'A'` at 16 px, the row-coherent ideal is about 22 vector
    instructions per pixel. U_g runs 82.

### 1.5 Schedule loops are folds

**Where this is already true (F).**

- After `legalize`, the lattice is three `Reduce`s over `Monoid::SEQ` around
  a `Write` (`lattice.rs:1-27`, `:126-130`, `:229-244`).
- The emitter treats every fold alike: `extract_folds`, `place_roots`,
  `plan_carries`, `FoldReads` and guards. F's traffic string
  `62x1, 6x32, 21x64, 105x4096` reads as the body, the row fold, the column
  fold, and the piece fold inside it, all `Scope::Fold`.
- The one exception is an execution rule: a fold whose binder a `Write`
  names as its lane runs by lanes (`emit/mod.rs:18-28`).

**Where it is not.**

| # | where (F) | the difference | this plan | open |
|---|---|---|---|---|
| L1 | `variance.rs:486-523` vs `extract.rs:1880`, `:2208` | a lattice fold's trips are priced through `evals`; a kernel fold's are dropped on the scale that decides | **closed** by M14: one rule, `evals` with kernel folds in the nest | — |
| L2 | `pack` (`lattice.rs:198-245`), `HalveFold` (`fold_rules.rs:539-581`), `expand_reduce` (`passes.rs:469-580`) | one chunking law, three implementations | **deletes** `expand_reduce`, a bespoke copy of `HalveFold`+`PeelFold` with its own substitution, on no production path (M13) | `pack` stays a pass. A rule cannot run after extraction (`insert.rs:118` refuses `Write`), and `HalveFold` = `StripMine(2)` needs `PeelFold`'s guard re-keyed (`fold_rules.rs:512-514` refuses `[0,2)`) plus identity elimination. Q6 |
| L3 | `arena.rs:370-390` | `Write { row, col, lane }` names binders as fields, so no substitution rewrites them; `combiner_op` has no SEQ arm (`fold_rules.rs:739-751`) | — | `Write` addressed by §1.3's affine read point, one representation for a binder-derived address; SEQ's combiner. Q6 |
| L4 | `passes.rs:108-112`; `insert.rs:54-59`, `:118` | the lattice is built after extraction | — | Q6. No mechanism binds the coordinates inside a saturated graph: `collapse` substitutes `X := x0+i+l` (`lattice.rs:113-124`), and the e-graph's one substitution rebuilds a representative per class (`fold_rules.rs:45-55`) |
| L5 | `scene.rs:223-257`; `manifold.rs:670-693` | stripes: the row fold's outer strip-mine is a Rust loop run by threads | — | the row fold, strip-mined and executed by threads as a lane fold is by lanes |
| L6 | `cell_grid.rs:440-441` | cells are not a fold | — | a fold over cells; zoom recompiles (Q3) |
| L7 | `atlas.rs:169-221` | the atlas is filled by a host loop over glyphs | CL11 writes by collapse into the slots | the loop itself |

**Why L4 is not "move `collapse` before saturation" (F).**

- The saturation cache key has no shape, and the atlas measured 7
  saturations against 1.
- `pack` needs `L`, so saturation would read the tier, and `pack` panics on
  a fold `collapse` did not build (`lattice.rs:211-216`).
- `collapse` refuses `Dwrt` and interval folds (`lattice.rs:141-182`).
- So the target in §1.1 inserts the lattice after the shape-free phases,
  per shape, and needs the missing mechanism above.

**Stated plainly:** this plan closes L1, deletes one of L2's three
implementations, and leaves L2's `pack` and L3–L7 open.

### 1.6 A frame is a schedule

**The target (JP): no atlas.** A frame is a fold over cells whose body
invokes `P_N` with that glyph's uniforms and the cell's origin.

- Every pixel is independent: the exact area is per pixel, with no
  cross-pixel prefix sum.
- So batches, rows and cells are independent.
- Multi-core, SIMD and ILP (halving the batch fold interleaves two batches)
  are all fold rewrites or fold execution rules.
- Implied IPC today is about 1.47 (instruction count over time).

**Today's memo is not a plain tabulation (F).**

- The atlas read is bilinear resampling (`cell_grid.rs:463-471`). **I:** an
  atlas-free frame changes pixels wherever density ≠ 1.
- The atlas is written by `copy_from_slice` from a temporary plane
  (`atlas.rs:212-218`), copied whole by `Arc::make_mut` while a frame holds
  it, and grown by the host.

**A per-cell loop in host Rust** would be a second invocation path, a second
copy of the ctx layout, and a loop nest outside the compiler. That is why L5
and L6 are the successor's, not host code here.

---

## 2. Why it diverged

**The pattern.** Each split rested on a statement that was true on the day
it was written about the *other* side's state. When the other side changed,
the reason expired and the split stayed. Under the pipeline splits, a stage
(compose versus compile) was mistaken for a tier (macro versus runtime).
Under the loop splits, when a loop is built was mistaken for what it is.

| divergence | introduced | stated reason, true on the day | what made it false |
|---|---|---|---|
| two optimizers | #974 (`eeec0ce5`) | "an arena reaching a backend unoptimized is never what anyone wanted" | it made the macro's own pass redundant, and that pass was never deleted |
| mask and integer ops only in the runtime vocabulary | `268d447f`; `d049a07e` #980 | registering them globally "broke the density-dependent AA ramp" | `d68edb8c` #1206: `DwrtFree` |
| the split named `Vocabulary` | `f0a4c4a7` #1146 | "Naming the vocabulary makes the choice visible at each insertion" | it named an accident |
| `ENode::Param` | `d68edb8c` #1206 | the macro saturates before its arguments are known | true only while the macro saturates |
| `DwrtFree` | `d68edb8c` #1206 | saturating an open term resolves `Dwrt` too early | the cause was optimizing an open term at all |
| extraction priced at POINT | `6336a0c2` #1097 | the macro has no shape | a macro that produces a value needs no extraction |
| `RuleSet::production` vs `runtime` | `077c1641` #1235; widened in #1294 | "`kernel!` has no syntax that builds a fold" | the extra rules match only `Reduce`; 7 of 7 macro-writable terms gave identical output under either set |
| `kernel_raw!` | #1206 | to benchmark an exact form | since #974 every arena is optimized at compile |
| `saturation-switch` | `c577565e` #1210 | the `egraph_off_on` harness | the harness went in #1235 |
| `detect()` inside the emitter | `3d018928` | the JIT was the only caller | a build-time caller |
| `Saturate` beside `optimize_runtime_arena` | `f0a4c4a7` #1146 | "name the endomorphism" | no production caller, and the two have drifted |
| `lower.rs` restating `Kernel` | #1206 | the macro lowers to an arena | fract, hypot and clamp defined twice; the retired-axis refusal four times |
| `Buffer` and `Gather` for parameters | `49a5ba1e`, "interpreter scope only… pending a design check-in" | the reference interpreter | the interpreter went in #1235 |
| `MAX_BOUND_BUFFERS = 4` | `3cc56272` #1175 | the ctx array lives on the stack | F: a `CachedText` of 5 distinct glyphs panics |
| `expand_reduce` | `eeec0ce5` #974 | a caller that wants an unrolled form | `HalveFold` reaches the identical shape in the graph (`fold_rules.rs:21-25`); since #1268 (`bc53ea22`, the re-land of #1252) no production path calls it |
| `dag += own`, trip-blind | `eb7a1c5d` #1117 | "this type carries neither the binder's trip count nor where codegen places its fold" | since #1268 a fold survives extraction on purpose, so its trips are its price |
| a flat class cap | `e70737ef` #1229 | the ceiling pinned at the floor until a sweep is re-run | the fold rules' growth scales with trips, not with inserted classes |

---

## 3. The subtraction list

"Replaced by" names existing code, never new parallel code. Additions have
their own list (§3.7).

### 3.1 The optimizer: one entry, one rule set, one unroller

| # | delete | where (F) | replaced by | what moves |
|---|---|---|---|---|
| M1 | `macro_tier()`, `DwrtFree`, `Saturate::macro_tier` | `pixelflow-compiler/src/lib.rs:163-196`; `saturate_pass.rs:43-61` | nothing: `kernel!` builds a value | the raw pairs in `kernel_macro.rs:196-225`; the bytes of macro-built test kernels |
| M2 | `kernel_raw!` | `lib.rs:131-151` | `kernel!`, identical after M1 | `derivative_under_warp.rs:50-81`'s pairs, replaced by the-language-is-kernel B6's equivalence gate |
| M3 | `Tier`, its prefix, its test, the telemetry's `"tier"` field | `tier.rs:20-50`, `:73`; `telemetry.rs:67`, `:130`, `:288-312` | nothing | lands with M6 |
| M4 | the `runtime()`/`production()` split; the dead subsets `core_rules()` and `transcendental_rules()` | `rules.rs:174`, `:193`; `math/mod.rs:135-148` | one set; `all_rules()` returns all 69 and promises no order | note (a) |
| M5 | `Vocabulary`; `insert`'s `vocab` argument (~70 sites); the `mask_and`/`mask_or` hatch | `ops.rs:340-388`; `insert.rs:111-115`; `fold_rules.rs:734-738` | one vocabulary | note (b) |
| M6 | `saturate_pass.rs` | `:21-129` | `optimize_runtime_arena` | `saturation_worth.rs` moves onto it |
| M7 | `Optimizer::production()`'s POINT default | `optimizer.rs:377-410` | the shape is required | research passes POINT |
| M8 | stale docs | Appendix B | — | — |
| M9 | `saturation-switch` and everything it gates | `pixelflow-search/Cargo.toml:61-69`; `runtime.rs:123`, `:141`, `:151-157`, `:388-424`, `:444` | nothing | CI's `--all-features` loses a byte-changing input |
| M10 | `Optimize`, `Rewritten`, `Identity`, `Then`, `pipeline!`; the pass wrappers | `optimize.rs` (304 lines); `passes.rs:2234-2340` | the functions they wrap | two research files call them directly |
| M11 | repeated stages | the `Ref` guards; the second `resolve` | one of each in `P` | — |
| M12 | silent fallbacks | `runtime.rs:259`; `jit_cache.rs:145-149` | a `CompileError`; a reported decline, pinned at zero for production kernels | — |
| M13 | `expand_reduce`, `expand_reduce_owned`, `ExpandReduce` | `passes.rs:469-491`, `:520-580` | the fold phase, through the one entry | its callers: `halve_fold_jit.rs:124`, `:205`; `corpus_gaps.rs:768`; `extraction_witnesses.rs:272`; `saturation_worth.rs:58` |
| M14 | the arbiter's trip-blind comparison | `extract.rs:1970-1976`, `:1880` | executed cost of each settled term, with one rule for every fold (§1.4) | kernels with a surviving fold may re-extract; the CL records which |
| M15 | the fold rules inside the main phase, under a flat cap | `rules.rs:193-198`; `saturate.rs:222-243` | the fold phase after main, admitting whole folds under `HARD_CLASS_LIMIT` (§1.4) | — |
| M16 | `quadrature` as a silent production fallback | `passes.rs:598-617`, `:672-686` | a `CompileError` when an integral reaches `resolve` on the production path | `quadrature` and passes' `Substitution` stay for research and the area oracles. `∫f = Σ wₖ f(pₖ)` approximates, so no rule may union it |

**Note (a), M4.**
- It moves the tests that pin the split: `fold_rules.rs:1549-1556`,
  `integral.rs:~1590-1598`, and `ir_insert.rs:70`, `:86` and `:160`.
- It moves the precondition at `nnue/guide/linear.rs:527-536`.
- It changes the fingerprints of about 40 research sites.
- The inflation study names its own 62 rules.

**Note (b), M5.**
- About 15 research binaries and tests pass `Templates`. Among them are
  `production_budget_determinism.rs:300`, `saturation_stop.rs:43`,
  `labeler.rs:390`, `:699`, `anytime.rs:209`, `prod_kernel_jit.rs:63`,
  `saturation_ceiling_env.rs:48`, `fold_exactness.rs:81`,
  `growth_telemetry_determinism.rs:71` and `optimizer_equivalence.rs:71`.
- Corpora that `Templates` rejected are re-pinned.

### 3.2 The front end

**Superseded by [the-language-is-kernel](2026-09-25-the-language-is-kernel.md).**
An earlier revision of this section made the `Kernel` builder the front end and
`kernel!` sugar or nothing. JP ruled the opposite: `kernel!` is the language, its
one parser is the only front end, and the builder is not a surface. What this
plan still owns from the front end:

| # | delete | where (F) | replaced by |
|---|---|---|---|
| F1 | `lower.rs`'s second copies of `fract`, `hypot`, `clamp` and the `Dwrt` encoding | `lower.rs:187-208` | lowering calls `pixelflow-ir`'s one set of definitions (the-language-is-kernel B5) |
| F4 | the `Param` family, about 120 references in 40 files, including the dead Dag-side `substitute_params` | `arena.rs:58`, `:1276+`, `:1357-1361`; `expr.rs:280-330` | parameters are uniforms or structural (the-language-is-kernel §1.4) |
| F7 | `pixelflow-compiler → pixelflow-codegen` | — | unused since `fa4ed36d` (#1172). The edge to `pixelflow-search` stays: the macro runs `P` for declared instances |
| F8 | stale docs | Appendix B | — |
| F9 | `EmitStyle`, a third op-to-syntax table | `traits.rs:7-22`; `kind.rs:778`; `ir lib.rs:131`; search `ops.rs:23-24` | its one research consumer prints `OpKind`'s name |

Every `Kernel` holds its graph twice (`kernel.rs:218-221`) and rebuilds `legacy`
on every combinator (`:258-270`); that is exprarena-on-dag Stage D's to remove.

### 3.3 The artifact: one function, run at two times

| # | move or delete | where (F) | becomes |
|---|---|---|---|
| C1 | move the `detect()` calls inside the emitter and `legalize` | `emit/mod.rs:1163`, `:3875`, `:3887` | `EmitCtx { max_regs, isa }`; `jit_cache` calls `detect()` once. `EmitCtx` loses `derive(Default)` |
| C2 | move the mmap inside `compile_via_backend` | `emit/mod.rs:4278` | the caller maps; `EmitCtx::compile` = assemble + `from_code` |
| C3 | move `jit_cache::compile`'s body | `jit_cache.rs:144-155` | `program` (A3) |
| C4 | delete `Guard` | `arena.rs:758-769`; `emit/mod.rs:4060-4066`; about 20 files | nothing. If the demand track needs a guard node, its arms live in the arena |

**`emit::compile` stays outside the law, by name.** It is the kept raw
research entry (`emit/mod.rs:3916`, 22 files). CLAUDE.md's "never obtained
unoptimized" is corrected to "through `jit_cache` or `program`".

**How the bytes load (I).** They load through `ExecutableCode::from_code`.
That needs no second loader and no object-format emitter.

### 3.4 Parameters are uniforms

**Every use of a buffer, classified.** The classes are: (i) a tabulation the
schedule made; (ii) data the host supplies; (iii) research or test plumbing.

| use | where (F) | class | becomes |
|---|---|---|---|
| U1 glyph control points | `loop_blinn.rs:379`, read at `:592`, folded over at `:384` | (ii) | a table kernel backed by 10N uniforms (§1.3). **This plan** |
| U2 atlas | `atlas.rs:42`, `:175-219`; read bilinearly at `cell_grid.rs:463-471` | (i), written like (ii) | stays a buffer (Q3) |
| U3 cell data | `terminal_app.rs:421`, `:502-506`; a new `Vec` every frame, against the no-runtime-allocation rule | (ii) | stays a buffer (Q3) |
| U4 `CachedGlyph`, `CachedText` | `cache.rs:94`, `:168-169`, `:196-214`, `:486-501` | (i) | deleted. No production consumer; `benches/font_rendering.rs:7`, `:104-125` moves. **This plan** |
| U5 `BilinearSampler`, `texture()` | `lattice/mod.rs:372-611` | (iii), (i) | goes with U2 |
| U6 research formats | the arena dump's `buf`/`B` lines; the `collapse_bench` corpus `B`/`D` lines; `collapse_cost.rs:141-217`; `corpus_gaps.rs:220-268`; `training/corpus.rs` tag 7 | (iii) | versions bumped |

**This plan deletes:**

| # | delete | where (F) |
|---|---|---|
| B1 | `binding.rs` (165 lines) and its re-export; `bind_by_identity`; `UniformBlock::entries`; the `BindingTable` comment and test import | `ir lib.rs:123-124`; search `runtime.rs:557-573`; `manifold.rs:148-156`; `lattice/mod.rs:480`; `round2_rules.rs:1303` |
| B3 | `Glyph::bound`, `Glyph::bake` | `loop_blinn.rs:204-217`. Their callers, production `atlas.rs:196` among them, move to `Lattice::eval_at` or `Lattice::bake`. `freetype_oracle.rs:268` is a named CI contract, and `rust.yaml:601-606` changes with it |
| B4 | the glyph's table, `row_at`, and the glyph's side-table entry | `loop_blinn.rs:379`, `:592` |
| B6 | the per-bind copy `Arc::new(data.to_vec())` and `Manifold.carried` | `manifold.rs:293`, `:427` |
| U4 | `CachedGlyph`, `CachedText`, and with them the 5-glyph panic | `cache.rs` |

**Deletions that wait for the successor**, once U2 and U3 leave:
- B5: the side table, if nothing else seeds it;
- B7: `bind`, `BoundManifold`'s buffer fields, `CellGridBuffers`, and the
  positional `frame(params, cells, atlas)`;
- B8: the `kernel_for`/`kernel` split;
- B9: `MAX_BOUND_BUFFERS`;
- B10: `ExprNode::Buffer`, `Gather`, and the per-ISA gather encoders.

**Stated plainly:** after this plan, the frame kernel's cells and atlas are
the only bound buffers left.

### 3.5 The glyph

| # | change | where (F) |
|---|---|---|
| G1 | the table becomes a kernel backed by uniforms, read at the fold's binder | `loop_blinn.rs:379`, `:592` |
| G2 | one box mint: 4 uniforms where there are 8 | `Glyph::over`, `Glyph::kernel`; `loop_blinn.rs:279-287` |
| G3 | exact N: delete `bucketed_trip_count` and `padding_row` | `loop_blinn.rs:361-366`, `:716` |

`run`'s fold stays. It is what extraction unrolls or keeps.

### 3.6 Widths (noted, out of scope)

These control-plane widths have no profiler reason, against CLAUDE.md's
64-bit rule:
- `UniformId(u16)`, `dense_slot → u16`, `ScheduledOp::Uniform(_, u16)`,
  `emit_uniform_load(offset: u16)`;
- `UniformIdentity(u32)`, `BufferId(u16)`, `BufferIdentity(u32)`;
- `LatticeShape([u32; _])`, `Range<u32>`, `ExprId(u32)`.

A read point's affine map is new, and it is `i64` from the start (A8). A
glyph at N=189 has 1,894 uniform slots. That fits in u16, and "it fits" is
not a reason.

### 3.7 What must be added

None of these is a second implementation of anything.

- **A3.** `pixelflow_codegen::program(kernel, shape, isa) -> Result<(Vec<u8>, KeyBytes, LinkCounts), CompileError>`.
  It returns the key's bytes, not `Canonical`, whose tables hold
  process-minted identities.
- **A4.** `unsafe fn jit_cache::preload(kernel, shape, isa, bytes)`.
  - It computes the key and uses the same `CACHE` and `from_code`.
  - It is `unsafe` because mapping bytes as code is. A7 checks the contract.
- **A5.** The build-time caller: the macro, for declared instances
  (the-language-is-kernel Phase E).
- **A6.** `vzeroupper` before `ret` on the x86 tiers.
  - F: the epilogue today is `add rsp, 0x300; ret`.
  - F: a raw call costs 155–170 ns without it and 6–9 ns with it.
- **A7.** The gates (§5).
- **A8.** Uniform families, read at an index affine in range binders
  (the-language-is-kernel §1.6, B3), with a positional setter, since F:
  `UniformBlock::offset` searches linearly (`manifold.rs:113-125`).
- **A9.** The fold phase and its admission (M15).
- **A10.** The arbiter's executed cost (M14).

**Not added:**
- a runtime length, or an index that depends on data;
- a bespoke unroller;
- a second closing mechanism;
- a price for code;
- a second loader;
- an object-format emitter;
- a layout type;
- a `cfg`.

---

## 4. The extent is baked

**Decision.** The lattice's extent is part of the program: `P` takes `s`. A
shape unknown at build time compiles on first use, through the same `P`. JP's
constant-range rule requires it.

**Evidence (F).** Emitted bytes for `'8'` at different shapes:

| comparison | tier | result |
|---|---|---|
| [16,16] vs [16,1] | AVX-512 | 1 byte differs |
| [16,16] vs [17,16] | AVX-512 | 2,884 → 4,648 B: a remainder arm |
| [32,32] vs [64,32] | AVX-512 | 2,530 of 2,884 bytes differ, from the same extracted term |

The allocator ranks carries by trip counts (`regalloc.rs:1790-1810`), and
fold bounds are constants in the code (`variance.rs:434-437`).

**Proxy for a runtime extent (F):**

| variant, as a ratio to baked [n,n] | AVX-512 | AVX2 |
|---|---|---|
| an [n,1] program per row | 0.76–1.26× | 0.99–1.42× |
| an [L,1] program per batch | 0.87–1.49× | 1.10–1.90× |

Performance does not argue for baking; the language does.

---

## 5. Migration order

Every CL is green on its own and says what it deletes, moves or adds. A CL
that deletes a file cited by a live plan updates the citation, because
`check-doc-paths.sh` fails on a missing path.

**Gate policy.** `byte_probe.rs:20-23` (#1252) records: *"a pinned hash
would fail on every intentional change … a gate that cries wolf gets
deleted."* This plan adds no committed digests. Every gate compares two
computations at the same commit:

- **Byte-neutral CLs** record a `byte_probe` before/after diff, on both x86
  tiers once C1 makes the tier explicit.
- **The law** (CL3) compares build-time bytes against runtime bytes in one
  job.
- **The three copies of `fnv1a64`** (`byte_probe.rs`,
  `demand_move_byte_check.rs:33-40`, `glyph_compile_report.rs:64`) become
  one in CL0.

**CL0: deletions that need no argument.**
- B1, M9, F7, F9, F4's dead Dag half, the dead rule subsets, and one
  `fnv1a64`.
- `byte_probe` shows nothing moves.

**CL1: A6, `vzeroupper`.**
- Every x86 program changes: one instruction before each `ret`, plus the
  pool anchor's `rel32` and the pool's padding (`x86_64.rs:77-92`,
  `:130-135`).
- The test is structural.
- The per-call cost is re-measured.

**CL2: `P` is a function.** C1–C3 and A3. `byte_probe` shows nothing moves.

**CL3: the law.** A7, with A4, and A5 for a test crate. Four checks:
1. **Build equals run.** Across kernels × shapes × `Isa::ALL`: a `kernel!`
   value, a scene, a `by_ref` composition, and a glyph at three sizes.
2. **Across hosts.** Each runner uploads its `program()` digests, and one
   job compares them at the same commit.
3. **Preloaded code is used.** There are zero saturations after preload
   (`saturation_count`, `runtime.rs:267`; `entry_count`,
   `jit_cache.rs:186`).
4. **Scope.** `check-provenance-journal-scope.sh` covers the macro crate's
   dependencies.

**CL4–CL5: superseded.** The front end's migration is
[the-language-is-kernel](2026-09-25-the-language-is-kernel.md)'s Phases A–D. M1–M3
and M6 land there, in B5, with the macro tier's saturation.

**CL6: one rule set, one vocabulary, one entry.**
- M4, M5, M7, M10–M12 and M16.
- Re-pins are argued in the CL.
- A lint modelled on `cfg-encapsulation` fails on `RuleSet::new` or
  `Optimizer` construction outside research modules.
- M16's gate is that no production kernel quadratures, which
  `glyph_is_closed` already implies for glyphs.

**CL7: C4, `Guard` deleted.**

**CL8: the glyph's table is a kernel backed by uniforms (Q2).**
- A8, G1–G3, B3, B4, B6 and U4.
- **Gates:**
  - A test halves and peels a fold over a table read and asserts the
    shifted read points.
  - An exhaustive `match` over `ExprNode`'s leaves, `ENode`'s leaves and
    the substitution walk.
  - Pixels within 1e-6 of F, the area oracles, and the goldens.
- **Bytes move.** A family read replaces `Gather` over a `Buffer`; inside a
  kept fold it is the same broadcast-load shape.
- `byte_probe` records the change, and the CL records per-glyph time
  against F.

**CL9: one price for every fold (A10, M14).**
- **Gates:**
  - Production extraction on an unrollable fold is pinned. Today
    `a_table_reading_fold_unrolls` (`fold_rules.rs:1701-1718`) asserts
    through the tree-only `extract`, so no test pins what production does.
  - `glyph_is_closed`.
  - The arbiter never chooses a term with an open integral.
- The tree arm is what finds the unrolled term, so its doc stops calling it
  a control (`extract.rs:2011-2015`).
- The CL records every production kernel whose extraction changes.
- It is glyph-neutral (measured): the production graph does not yet hold the
  unrolled term.

**CL10: the fold phase (A9, M13, M15).**
- **Gates:**
  - `glyph_is_closed`, the oracles, the goldens, and output within 1e-6 of
    F.
  - **New:** the production term for every single glyph is fold-free.
  - **New:** the 94-glyph `text()` run is pinned to what admission decides,
    with its code size recorded.
  - A test for the possible retry gap: a `HalveFold` that declined before
    its integrand closed must fire once it has, even when the closure is
    deeper than `DIRTY_TRACKING_MAX_DEPTH = 2` (`graph.rs:249`).
- **The CL records:**
  - per-glyph time against §1.4's U_g column;
  - code and emit time per N;
  - saturation time and class counts.

**CL11: the atlas is written by collapse into its slots.**
- `collapse_rows` at the atlas's pitch (`manifold.rs:584-610`).
- It deletes the temporary plane and its copy. Pixels are identical.

**The parallel track.** The payoff after CL10 lies here, measured in §1.4:

- **D1: placement that reads demand.** It puts a piece's row work inside its
  row-scope band guard. That is 79% of U_g's executed work.
- **X1: arms emitted as blocks.** U_g's emit takes 100–153 ms at 64 pieces
  and is superlinear in N.
- **The successor plan:** L2's `pack`, L3–L7, U2, U3, B5, B7–B10, Q3 and
  Q6.

---

## 6. What fonts get

Figures are F on NotoSansMono, release, medians. F varies ±40% between runs,
and the U forms up to 30%, so read ratios.

**Per glyph, warm, AVX-512 (µs).** §1.4 has the full table.

| | F (today) | after CL10: U_g at exact N |
|---|---|---|
| `'A'`, 16 px | 11.3 | 4.2 |
| `'8'`, 32 px | 191 | 24.8 |

**Instructions executed per pixel, AVX-512:**

| | F | U_g at exact N |
|---|---|---|
| `'A'`, 16 px | 185 | 61 |
| `'8'`, 32 px | 718 | 98 |

**Compile.**

| | F (today) | after CL10, JIT | built at build time |
|---|---|---|---|
| programs for ASCII | 6 buckets, 17 KB | one per N: 35 (Noto), 28 (DejaVu); about 1.75 KB per piece | the same bytes |
| first glyph of a program | saturation 21–35 ms (at the cap), extract ~3 ms, emit ~1 ms | fold phase 9–62 ms (exact N) + extract + emit 9–127 ms | a copy and an `mprotect` |
| later glyphs | cache hit, 10–16 µs | the same | the same |

At `opt-level 0` a build step runs 4–5× slower. So the build-time crates want
`[profile.*.build-override] opt-level = 3` (the macro and its dependencies
build at opt-level 0 by default).

**A frame with no atlas** (ms, 1 / 4 threads). This was measured with a host
loop over cells and U_band. It is a bound on the direction, not on this plan:

| grid, px, text | F | U_band, AVX-512 | U_band, AVX2 |
|---|---|---|---|
| 80×24, 16, claude | 26.4 / 7.0 | 6.9 / 2.8 | 4.1 / 1.6 |
| 200×60, 32, claude | 413 / 122 | 84.4 / 22.9 | 50.3 / 20.0 |

At 200×60 and 32 px nothing meets 16.7 ms. **I:** that needs D1, or the
atlas kept.

---

## 7. Open questions for JP

**Q1. Superseded:** `kernel!` is the language (the-language-is-kernel).

**Q2. Is a table kernel backed by uniforms, read only at binder-affine
points, inside "we don't have uniform arrays"?**

- It is what GLSL ES 1.00 calls a uniform array indexed by constant-index
  expressions: constant size, no runtime length, no index that depends on
  data.
- **Recommendation: yes.** It is "a uniform" under your constant-range rule,
  and it is "another kernel".
- If no, CL8 does not happen and the glyph keeps its bound buffer.
  Instances written at construction fail to close from N ≥ 37.

**Q3. The frame.** Under your constant-range rule, cells become a fold and a
zoom recompiles, once per level, through the cache. That overrides
`cell_grid.rs:18-36`. Confirm.

Then a frame without an atlas needs one of:
- **(a) one data-indexed read:** glyph → its table offset, from cell data;
- **(b) a dispatch over every glyph's program:** a `select` per glyph id,
  which jumps because a cell's id is uniform across its batches, at the
  cost of every glyph's code in the frame program.

The atlas memo (E2) keeps a data-indexed read too; E1 means no memo.

**Recommendation:** keep the atlas (E2) until D1 brings 200×60 at 32 px
under 16.7 ms, then measure (a) against (b). This belongs to the successor
plan.

**Q4. Superseded:** the syntax's surface is the-language-is-kernel Q2.

**Q5. Public API sign-offs.**
- **Additions:**
  - `program` (A3);
  - `unsafe jit_cache::preload` (A4);
  - `EmitCtx.isa` (C1);
  - uniform families and their read (A8, answered by JP: "it is a
    uniform"; spelled in the-language-is-kernel §1.6);
- **Changes:**
  - `Optimizer::production(shape)` (M7);
  - one `RuleSet`, with `all_rules()` returning 69 (M4).
- **Removals:**
  - `kernel_raw!` and `Scalar`;
  - `Glyph::bound`/`bake` and `CachedGlyph`/`CachedText`;
  - `core_rules`/`transcendental_rules`;
  - `expand_reduce`/`ExpandReduce`.

**Q6. The lattice's folds in the e-graph (L2–L4).**
- It needs a mechanism that does not exist: binding the coordinates inside
  an already-saturated graph, per shape.
  - One route is to union `Var(0) ≡ x0+i+l` on a per-shape clone, which
    changes every class's variance fact.
  - The other is to re-saturate a copy, which gives up the shape-free
    cache.
- It also needs `Write` addressed by the affine read point, and a price
  that stops `HalveFold` unrolling a row fold, which copies the whole
  kernel.
- **Recommendation:** make it the successor plan's first question. Until
  then, what makes a schedule loop different from a fold is when it is
  built, plus the host loops L5–L7.

---

## Appendix A. Construction-time unrolling (withdrawn)

The first draft wrote a glyph as N instances at construction and proposed
fix (b) to close them. **F:**

- **N separately written integrals do not all close from N ≥ 37.** Open
  integrals, out of those written: 3 of 76 at N=38, 7 of 80 at N=40, and 47
  of 128 at N=64. Coverage is off by up to 0.49, because each separately
  written integral pays for its own derivation under a shared class cap.
- **Written once and halved by `HalveFold`, the integral closes once** and is
  copied closed (§1.4).
- **Fix (b) would have needed:**
  - a second read-back extractor;
  - unions outside any rule;
  - `canonical` walking in post-order.

The construction-time U_band's measurements stay in `op_measured.md` as a
reference.

---

## Appendix B. Stale docs (M8, F8)

- **CLAUDE.md:**
  - `:176` says "no iteration binder. A fixed-count iteration is unrolled
    at construction." That is the framing JP rejects: `Fold` has a binder,
    and unrolling belongs to `HalveFold`;
  - `env_extraction_policy()`;
  - "`kernel_raw!` to skip optimization";
  - "The macro tier does not resolve `Dwrt`";
  - the Compiler Pipeline section;
  - `:187` "optimizes";
  - `:209` the `bind` diagram;
  - `:559` "macro-tier";
  - `:566` "a gather over its bound buffer";
  - "never obtained unoptimized".
- **Pricing and unrolling:**
  - `cost.rs:409-411`, "the DP multiplies". Only the tree DP does.
  - `kernel.rs:724`, "the backend unrolls".
  - `passes.rs:476`, "the address folds to an immediate".
  - `halve_fold_jit.rs:103-113`, "production always runs ExpandReduce last".
  - `optimizer.rs:133-135`, "10 classes per inserted class, 5,000–50,000".
    The code has 8, with ceiling = floor.
  - The `saturation-switch` comment naming `ExpandReduce`.
  - `extract.rs:2011-2015`, the tree arm as a control.
  - schedule-cost-model-denotation §5.1, "no rewrite changes a level".
  - a-kept-structure-is-control-flow §2.
- **READMEs:**
  - `README.md:18`, `:52`, `:68-70`;
  - `pixelflow-core/README.md:7`, `:27`, `:50-52`;
  - `pixelflow-graphics/README.md:23`, `:137`;
  - `pixelflow-search/README.md:95`.
- **Agents:**
  - `.claude/agents/algebraist.md:25`;
  - `.claude/agents/pixelflow-core.md:12`, `:25-29`.
- **Code docs:**
  - `runtime.rs:3-7`, `:19-22`;
  - `optimizer.rs:168`, `:221-226`, `:330-334`, `:399-404`, `:419`;
  - `egraph/mod.rs:15`;
  - `saturate.rs:170-172`;
  - `ops.rs:357-358`;
  - `lattice/mod.rs:326-331`;
  - `jit_cache.rs:4-7`, `:19-21`;
  - `fold_rules.rs:490-491`;
  - `optimize.rs:7-13`, `:95`;
  - `ir lib.rs:119`;
  - `fold.rs:282`;
  - `kernel.rs:4`, `:353`, `:742`;
  - `telemetry.rs:27`, `:109-113`;
  - `passes.rs:2268-2276`;
  - `glyph_optimization_cost.rs:4`;
  - `text_kernel_cost.rs:12`;
  - `halve_fold_jit.rs:105`;
  - `compiled_kernel.rs:54`.
- **Manifests:**
  - `pixelflow-compiler/Cargo.toml` (`saturation-telemetry`);
  - `core-term/Cargo.toml`;
  - `pixelflow-search/Cargo.toml`.
- **The front end:**
  - `ast.rs:43-48`;
  - `lib.rs:68-70`;
  - `symbol.rs:9-19`;
  - `sema.rs:33`;
  - `emit.rs:142-151`.
