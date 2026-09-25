# The language is `kernel!`

## Metadata
- **Author**: JP (direction), Claude (draft)
- **Status**: `Proposed`. Revised the same day after JP's rulings on Q1:
  there are no tables, the control points are uniforms, a glyph's program
  is the kernel for its number of control points, and `select` is renamed
  `if` (§1.6, §1.7). Phase A is in progress.
- **Created**: 2026-09-25
- **Verified against**: `8b7b75a`, in two rounds.
  - A four-way inventory of every use of the `Kernel` builder, a
    completeness critic, a sketch of the glyph in the extended syntax, and
    two adversarial reviews of the sketch.
  - Each measured its claims with scratch crates outside the repo; all are
    deleted, and the tree was never touched.
- **Continues**: [one-pipeline](2026-09-24-one-pipeline.md). That plan owns
  the pipeline `P`, the law `bytes(P_build) = bytes(P_run)`, unrolling as
  extraction's choice, and the price of a fold. This plan owns the front end,
  and supersedes one-pipeline's §3.2.
- **Amends**:
  - [kernel-with-a-lattice](2026-09-06-kernel-with-a-lattice.md), where
    "kernel! is the kernel" held for three days;
  - [composition-is-linking](2026-09-09-composition-is-linking.md), where
    composition is a kernel-typed argument, not a method.

**Convention.** **F** marks a fact, read in code or git history or measured.
**I** marks an inference.

**JP's words (verbatim):**

> *"Kernel and kernel jit should be the same syntax same parser."*
>
> *"The builder isn't supposed to be a thing. How did that get out? Is it
> literally the IR?"*
>
> *"It's literally supposed to be the same syntax. The same backend. You just
> run the compilation as a macro when you compile the program."*
>
> *"It is a uniform no? The rule is that ranges have to be constant. The
> programs runtime knowable at compile time (we may relax this at some point
> and allow uniform iterations and allow runtimes that are polynomial
> functions of input, but that's like no where near the docket for now)."*
>
> *"The 'atlas' becomes the kernel for that number of control points,
> everything else is a uniform."*
>
> On Q1: *"I think I want one program per font per zoom level. I want to
> [avoid] table indexing because it sounds like where we've historically
> fucked up. No, just rely on select and bounding. That forms a bsp
> automatically. We'll make computing the programs fast, and focus on the
> caching later. For now, recompile fonts during zoom. There are legit
> problems there, but it's not worth fucking up the language to solve them.
> Good enough is easy."* And: *"We shouldn't have tables at all."* And: *"You
> might rename select if…"* And, on a draft that wrote the control points in
> as constants: *"No, make the coordinates of the control points uniforms."*

---

## 0. How the builder got out (F, from git)

`pixelflow_ir::Kernel`, re-exported as `pixelflow_core::Kernel`, is the IR
with a fluent coat.

- **What it is.** It is `Arc<KernelData { rooted: Rooted<ExprData>, env,
  legacy: (ExprArena, ExprId), buffers }>` (`kernel.rs:214-231`).
- **Its surface.** 56 of its 66 public methods are node constructors, e.g.
  `add` is `self.combine(rhs, OpKind::Add)`. The other 10 are plumbing, with
  raw IR in and out (`from_parts`, `parts`, `linked_parts`, `from_rooted`,
  `dag`, `rooted`).

**The history.**

| when | commit | what happened |
|---|---|---|
| 2026-07-23 | `d264abff` | `Kernel` arrives as "the surface graphics will build glyphs on", already with the whole op set |
| 2026-07-24 | `48dbe484` | the reduction binder gets its "front door" on the builder (`sum_over`), not in the syntax, and "completes the scalar surface the real programs need" there. From then on every feature lands on the builder first: `over`/`Monoid`, uniforms, `by_ref`, `with_buffer_data`, `area` |
| 2026-09-06 | `69b005c5` #1180 | `kernel!`, `kernel_jit!` and `kernel_value!` become one macro, "kernel! is the kernel". The ahead-of-time backend is deleted instead of being pointed at the JIT's |
| 2026-09-09 | `077c1641` #1235 | the glyph needs a fold over a table, and `kernel!` has no fold syntax, so the glyph moves onto the builder. Every production kernel follows. `rules.rs:177-190` records the same gap as the reason for a second rule set |
| since | — | CLAUDE.md codifies the builder as the interface (lines 17, 19, 173, 215 and 520 onward). Every later session, including the one that drafted one-pipeline, read it as the sanctioned surface |

**Where it stands (F).** 31 production source files in 7 crates build
kernels with the builder. `kernel!` has one user outside its own crate, a
test. The syntax lost to the builder because every feature was given to the
builder first. That is the thing this plan reverses.

---

## 1. The denotation

### 1.1 One language, one parser, one pipeline

```text
kernel! block ──parse──▶ AST ──sema──▶ typed AST ──lower──▶ template(structural params)
template ──instantiate(structural values)──▶ program ──P(·, s, t)──▶ bytes
```

- **One parser:** `pixelflow-compiler`'s, run inside the macro. Nothing
  parses at runtime.
- **One lowering.** It calls `pixelflow-ir`'s one set of definitions. There
  is no second copy of `fract`, `hypot`, `clamp` or the derivative encoding;
  today `lower.rs:187-208` restates `kernel.rs:516`, `:587` and `:656`.
- **One pipeline:** `P`, owned by one-pipeline.
- **Binding time decides where `P` runs, not a tier.**
  - A program whose structural parameters and shape are declared at build
    time is optimized at expansion, by the one optimizer (Phase E).
  - Anything bound at runtime goes through the same `P` in the JIT: a font
    loaded at runtime, a new zoom level, a new shape, or a kernel-typed
    argument.
- **The builder is not a surface.** `Kernel` stays as the opaque value a
  `kernel!` entry returns and `Lattice::bake` takes. Its fluent constructors
  leave the public API (Phase D).

### 1.2 A `kernel!` block

A block of items, parsed as a `syn::File`:

- `struct` records of `f32` fields;
- `const` items;
- `fn` items.

A `pub fn` is an **entry**: the macro emits a host function for it. A private
`fn` is a **helper**, and lowering inlines it (β-reduction). `|a: T, …| e` is
sugar for a block with one entry.

- **`X` and `Y` appear only in entries. Helpers take coordinates as
  arguments.** Application is contramap: `f(X + 0.5, Y + 0.5)` is today's
  `.at(X + ½, Y + ½)`. There is no `.at` method. A helper therefore cannot
  read an unshifted `X` by accident.
- **Recursion, loops with state, `mut` and assignment are refused.** The
  language is a DAG with bounded folds.

### 1.3 Types (in sema; the IR keeps its lanes)

| type | meaning | IR |
|---|---|---|
| `f32` | a value | an `f32` lane |
| `bool` | a mask; comparisons produce it; `&` and `\|` combine it | an all-ones or all-zero lane, `OpKind::mask(bool)` |
| `usize` | a fold binder or a structural count | `Var(REDUCE_BINDER_BASE + slot)`; converted by an explicit `i as f32` |
| records | named `f32` fields | flattened at lowering |
| `[R; N]`, `N` structural | a family of `N` records of scalar uniforms, iterated at instantiation and never indexed (§1.6) | `N`·width scalar `Uniform`s at static slots |
| `impl Fn(f32, f32) -> f32` | a kernel-typed parameter (Phase D) | a hole spliced at instantiation |
| `u32` bits | packed words (Phase D) | `Bits` ops |

**Masks are typed.** Today `X.select(Y, 7.0)` compiles and blends a number as
a mask. F: probe p16 gives 5. After this plan it is a type error.

### 1.4 Binding times

| parameter | example | binding | in the key? |
|---|---|---|---|
| structural | `const N: usize`, a zoom level's tile extent | at instantiation; each value is its own program | yes |
| uniform | a piece's ten coordinates, the box, `fg`, `bg`, the origin | per call, through the entry's `Args` record | no |
| kernel-typed | `k: impl Fn(f32, f32) -> f32` | at runtime; composed, then `P` | the composed program's |

- **Everything that is not structural is a uniform, and a uniform is a
  scalar.** An `f32` argument no longer folds into a constant because of its
  type at the call site; today the call-site type decides (`emit.rs:79-100`).
- **The control points are uniforms** (JP). A glyph's program is the kernel
  for its number of pieces `N`, and the glyph itself is the uniform values.
  Two glyphs with the same `N` share one program, whatever font they came
  from. A new `N` or a new tile extent is a new program: recompiled, cached
  by key.
- **Each entry has an `Args` record.** A compiled program is bound from
  `&Args`, and "every argument supplied" is a type rather than a runtime
  assert. It replaces `Uniform` handles, `UniformBlock::set`'s linear search
  (`manifold.rs:113-125`), and the refusal at `packed.rs:185-196`.

### 1.5 Folds and integrals

| spelling | denotation | IR |
|---|---|---|
| `(a..b).map(\|i\| e).sum()`, `.product()`, `.any(\|i\| m)`, `.all(\|i\| m)`, `.fold(f32::INFINITY, f32::min)`, `.fold(f32::NEG_INFINITY, f32::max)` | ⊕ over `i ∈ [a, b)`; the identity if empty | `Reduce(Fold::Range(RangeFold{monoid, binder, a..b}), e)` |
| `integral(lo..hi, \|u\| e)` | ∫ from lo to hi of e du; lo and hi constant `f32` | `Reduce(Fold::Interval(…), e)` |
| `area(\|u, v\| e)`, a prelude function | `integral(-0.5..0.5, \|v\| integral(-0.5..0.5, \|u\| e))` | two interval folds, as `Kernel::area` builds them (`kernel.rs:778-792`) |
| `monotone_root(δ, step, bend)`, an intrinsic | τ(δ) | `integral::monotone_root`, the one definition (`integral.rs:325-360`) |

- **Ranges are constant** (JP): `a` and `b` are expressions over literals and
  structural parameters.
- **Binder slots** are assigned inside-out, the lowest slot free in the body,
  as `bind_fresh` does (`kernel.rs:797-826`).
- **Unrolling is the e-graph's** (`HalveFold`, `PeelFold`, `EmptyFold`;
  one-pipeline §1.4). The syntax never unrolls.
- **A fold's body reads nothing indexed by its binder except the binder's own
  arithmetic.** There are no tables to read (§1.6).

### 1.6 There are no tables; `if` and bounding make the tree

JP: *"We shouldn't have tables at all."* Table indexing is where this
codebase has gone wrong before:
- the `f32`-lane index and its `EXACT_F32_INDEX` guard (`manifold.rs:315-323`);
- the bound buffer, its per-bind copy (`manifold.rs:427`), and the
  `MAX_BOUND_BUFFERS` panic;
- the glyph's own table.

So the language has none:

- **No uniform arrays, no buffers, no `Gather` read by a program's author.**
  Data enters a program in one of two ways:
  - as a scalar uniform, per call;
  - as a structural count (§1.4), which fixes how many uniforms there are.
- **A family `pieces: [Row; N]` is not a table.** It is `10·N` scalar
  uniforms at static slots. `pieces.map(|p| …)` iterates the family's
  *structure* at instantiation: the program holds `N` copies of the body,
  each over its own ten uniforms, and no index exists at runtime. That is
  what "the kernel for that number of control points" means.
  - It is not a bespoke unrolling pass: there is no fold to unroll. The
    e-graph's `HalveFold` unrolls folds over ranges; a family has none.
  - The measured form of this program is `U_band` (one-pipeline §1.4,
    Appendix A): the same `N` instances over scalar uniforms.
- **Choosing among alternatives is `if`.** A tree of `if`s over a uniform
  (`if id < k { … } else { … }`) is a binary space partition over it. So is
  a tree of bounding tests over space: a glyph's box, a piece's band.
  - The partition falls out of `if` and bounding, and nobody builds it as a
    structure.
  - A mask that is uniform across a batch takes one arm, which is a jump.
    Only a mask that varies by lane blends.
- **`Select` is renamed `If`**, in the IR, the e-graph, the emitter and the
  docs. CLAUDE.md needs a section, "Select contains an if", to explain what
  the name hides. The emitter was built on the misreading, blend by default
  with a branch bought per select (`emit/guards.rs`), and that cost 73% of a
  glyph bake (docs/BACKLOG.md X1) and the slowness of runs. The name is the
  bug (A6).

### 1.7 A glyph is the kernel for its number of control points

```rust
kernel! {
    /// One oriented monotone arc piece.
    pub struct Row {
        pub x0: f32, pub e0x: f32, pub e1x: f32,
        pub y0: f32, pub e0y: f32, pub e1y: f32,
        pub sigma: f32, pub s: f32,
        pub lo: f32, pub hi: f32,
    }
    pub struct Bounds { pub x0: f32, pub y0: f32, pub x1: f32, pub y1: f32 }

    const PIXEL_CENTER: f32 = 0.5;
    const COVERAGE_SNAP: f32 = 1.0 / 1024.0;
    const NEARLY_ONE: f32 = 1.0 - COVERAGE_SNAP;

    fn coverage(f: f32) -> f32 {
        let c = f.abs().min(1.0);
        if c >= NEARLY_ONE { 1.0 } else if c <= COVERAGE_SNAP { 0.0 } else { c }
    }

    fn indicator(m: bool) -> f32 { if m { 1.0 } else { 0.0 } }

    /// χ: the region left of the arc, within its band.
    fn left_of_the_arc(p: Row, x: f32, y: f32) -> f32 {
        let b = p.e0y.max(0.0);
        let bx = p.e0x.max(0.0);
        let a = p.e1y.max(0.0) - b;
        let ax = p.e1x.max(0.0) - bx;
        let t = monotone_root(y - p.y0, b, a);
        let x_at_t = p.x0 + t * (bx + bx + ax * t);
        indicator(0.0 <= t) * indicator(t < 1.0) * indicator(x < x_at_t)
    }

    /// σ·∫∫χ over the pixel about (x, S·y), cut to the rows the piece reaches.
    /// Closed once with `p` abstract; each of the N copies instantiates the
    /// closed form over its own uniforms (§1.8).
    fn piece_term(p: Row, x: f32, y: f32) -> f32 {
        let term = p.sigma * area(|u, v| left_of_the_arc(p, x + u, p.s * y + v));
        if (y > p.lo) & (y < p.hi) { term } else { 0.0 }
    }

    fn inside(b: Bounds, x: f32, y: f32) -> bool {
        (x >= b.x0) & (x <= b.x1) & (y >= b.y0) & (y <= b.y1)
    }

    /// The glyph with N pieces. Texel (i, j) holds coverage at (i+½, j+½).
    /// The pieces and the box are uniforms; N is the program.
    pub fn glyph<const N: usize>(pieces: [Row; N], bounds: Bounds) -> f32 {
        let (x, y) = (X + PIXEL_CENTER, Y + PIXEL_CENTER);
        let f: f32 = pieces.map(|p| piece_term(p, x, y)).sum();
        if inside(bounds, x, y) { coverage(f) } else { 0.0 }
    }
}
```

**The spelling still to choose (Q2)** is `pieces.map(…).sum()`: iteration
over a family's structure at instantiation. The program it produces has no
fold, no table and no index.

**What stays host Rust.** None of this is program; it produces the
per-call uniforms and the count `N`.
- font parsing, compound glyphs and mirroring (`ttf.rs`);
- the f64 monotone split (`monotone.rs`);
- `piece_row`'s rounding, with its debug asserts;
- layout.

**A cell is one call.** A cell calls `glyph::<N>` with its glyph's pieces,
box, colours and origin as uniforms. The loop over cells is host schedule
until scheduling moves into the compiler. A call costs 6–9 ns once A2 lands
(F, measured with `vzeroupper`), so 12k cells is about 0.1 ms of calls plus
the uniform writes.

**Selection is `if` and bounding, not a lookup.** The host chooses which
program a cell calls by its glyph's `N`, and passes that glyph's uniforms.
Inside the program every choice is an `if`: the box, each piece's band. A
mask uniform across a batch takes one arm, which is a jump, so the tree of
`if`s partitions space by itself.

**A zoom level recompiles.** The tile extent is structural, so a new pixel
size is a new set of programs, one per `N` the font uses: 35 for Noto's
ASCII, 89 for the whole font (F). Caching comes later (JP). An empty glyph
stays distinct from a missing one, as the atlas's slot layout does today
(`atlas.rs:163-200`).

### 1.8 What it takes, and where it lands

Three problems, none of which touches the language:

1. **Every piece is its own integral.** A glyph has up to 189 pieces (F,
   Noto). Integrals written separately stop closing past about 37 in one
   e-graph, because each pays for its own derivation under a shared class
   cap (F, one-pipeline Appendix A).
   - **Fix: a helper is a unit of optimization.** `piece_term`'s integral is
     closed once, with its parameters abstract, and each of the `N` copies
     instantiates the closed form over its own uniforms.
   - This is "close once, instance N", with the function marking what is
     shared, so nothing has to recognize it.
   - It is sound because no rule matches a parameter specially: a derivation
     over free parameters instantiates to a derivation over any values (F:
     no rule matches `ENode::Uniform`; one-pipeline Appendix A).
2. **Size.** At about 100 classes a piece, the largest glyph is about 19k
   classes, under `HARD_CLASS_LIMIT` (100k, `graph.rs:512`). Code is about
   1.5 KB a piece (F: `U_band`, 279 KB at N = 189), and about 1.1 MB for
   the 35 programs of Noto's ASCII (F).
3. **Zoom latency.** One program compiles in 7 ms at N = 8 and 4.4 s at
   N = 189 (F, `U_band`'s emit, superlinear in N). So a zoom level takes
   seconds on one thread.
   - **Fix:** the programs are independent, so compile them in parallel, and
     emit arms as blocks (X1) to remove the superlinear emit.
   - "We'll make computing the programs fast, and focus on the caching
     later" (JP).

**What the macro emits.**
- Records become host structs (`#[repr(C)]`).
- Each entry becomes a host function that instantiates the lowered template
  with its structural values and returns the opaque `Kernel`.
- The template is a replay of `ExprArena` pushes, as `emit.rs` emits today.
- No optimization runs at expansion unless the instance is declared (Phase
  E).

---

## 2. Decisions

The evidence and JP's rulings settle these. JP can overturn any.

| # | decision | resolution |
|---|---|---|
| D1 | binding times | §1.4: structural (counts and extents), uniform (every number, per call), or kernel-typed; `Args` records |
| D2 | what the macro compiles | the JIT template always; declared instances optimized at expansion (Phase E); `macro_tier`, `Templates`, `ENode::Param` and `kernel_raw!` deleted (one-pipeline M1–M5) |
| D3 | tables | **none** (JP). A family of records is `N` scalar uniforms iterated at instantiation; choice is `if` (§1.6) |
| D4 | binders | `usize` in sema; slots inside-out; a kernel-typed argument's binders are renamed away from those live at its hole |
| D5 | `.at` | application is contramap (§1.2) |
| D6 | functions across blocks or crates | inlined within a block; across blocks only as kernel-typed arguments at runtime. A proc macro sees only its own tokens |
| D7 | records and tuples | flattened in the front end; record returns (`-> Rgba`) with one `if` on the packed word, as `packed.rs` relies on (Phase D) |
| D8 | masks and bits | types in sema only |
| D9 | the frame | a cell calls the kernel for its glyph's `N` with the glyph's uniforms; a zoom recompiles; caching later (JP, §1.7) |
| D10 | `CachedGlyph`, `CachedText`, the atlas, `BilinearSampler` | deleted as the frame moves to per-cell calls. Caching returns later as its own design (JP) |
| D11 | loop-carried iteration | refused in the syntax. The two fractal benches (`shader_bench`) stay on the IR as compiler research |
| D12 | binder-indexed immediates | refused; the packer names its four channels |
| D13 | where production kernels live | above pixelflow-core. The cell grid is terminal-shaped and leaves core (CLAUDE.md: no terminal logic in PixelFlow) |
| D14 | `kernel_raw!` | deleted. Every JIT compile optimizes (`jit_cache.rs:145`), so its promise never reached machine code |
| D15 | spellings | §1.5's folds; `if` is the only choice; `DX(e)`/`DY(e)` as today; the `.select`, `.lt`, … aliases go once the five CI-contract bodies are rewritten, keeping their names |
| D16 | public surface | an opaque `Kernel`; `Uniform`, `Scalar`, `Monoid` and `Bits` leave. `__macro` narrows to what expansions name. Only compiler crates depend on `pixelflow-ir`, enforced by CI |
| D17 | the second parser | `training/factored.rs`'s `parse_kernel_code_arena` and its printer are deleted with the corpus tool that uses them, or routed through the one parser if that tool is still needed |
| D18 | `Select` | renamed `If` everywhere (§1.6; JP) |
| D19 | helpers | a helper is a unit of optimization: an integral in it is closed once with its parameters abstract, then instantiated per copy (§1.8) |

---

## 3. Migration

Every CL is green on its own. Byte-neutral CLs record a `byte_probe` diff,
and no digests are committed (one-pipeline §5, gate policy).

### Phase A: foundations

- **A1. The one parser's bugs (F, measured).**
  - `{ let X = Y; X }` evaluates X.
  - Nested `let`s leak: `locals` is one flat map (`lower.rs:69`,
    `:259-265`).
  - An out-of-scope local is accepted.
  - Literals round twice, f64 then f32 (`lower.rs:98-103`).
  - The grammar doc is stale.
- **A2. `vzeroupper` before `ret` on x86** (one-pipeline A6). Measured
  155–170 ns per call without it and 6–9 ns with it.
- **A3. `canonical` independent of insertion order.** It walks post-order
  from the root and hash-conses structurally equal subterms. Recompiling a
  font at a zoom level it has seen then hits the cache, whatever order the
  glyphs were built in.
- **A6. `Select` renamed `If`** (D18). A mechanical rename, with CLAUDE.md's
  "Select contains an if" section retitled.
- **A4. The uniform chain at 64 bits:** `UniformId`, `dense_slot`,
  `ScheduledOp::Uniform` and `emit_uniform_load`'s offset. A glyph at
  N = 189 has 1,894 uniforms, and nothing bounds `N`; today `declare_uniform`
  asserts below `u16::MAX` (`arena.rs:700-704`). `UniformBlock::set`'s linear
  search (`manifold.rs:113-125`) goes with it: the `Args` record is
  positional.
- **Deprioritized.** 64-bit fold ends (A5) have no driver in this plan.

### Phase B: the syntax grows the font's constructs

- **B1.** The items block, `if` as the only choice, typed masks, `const`
  items and helper `fn`s.
- **B2.** Folds over constant ranges, and the binder type.
- **B3.** Binding times and `Args`, records, structural counts, and families
  of records iterated at instantiation.
- **B4.** `integral`, `area` and `monotone_root`.
- **B5.** Lowering calls `pixelflow-ir`'s definitions, and `lower.rs`'s
  copies go.
- **B6.** Helpers as optimization units (D19): a helper's integral is closed
  once and instantiated.
- **B7.** The equivalence gate: one glyph built by `kernel!` and by the
  builder gives the same pixels over ASCII at 7, 16 and 32 px.

### Phase C: the font is written in `kernel!`

- **C1.** The §1.7 block: one program per `N` per tile extent, the glyph
  its uniforms.
  - The gates are `glyph_is_closed`, `glyph_exact_area`,
    `glyph_area_edge_cases`, `freetype_oracle` and the goldens.
  - Re-baselined pins go in their own commit.
- **C2.** The frame calls `glyph::<N>` per cell. The atlas,
  `CachedGlyph`/`CachedText` and `BilinearSampler` go (D10). A zoom
  recompiles.
  - Measure the frame against today's before switching: 80×24 and 200×60,
    at 16 and 32 px.
- **C3.** The glyph's tests move onto `kernel!`.
- **C4.** `text()` and `run` (Q3).

### Phase D: the builder goes internal

- **D-a.** Kernel-typed parameters, with the capture-avoiding splice (D4).
- **D-b.** Record returns, `u32` bits, and the packed frame.
- **D-c.** Scenes, ML and the runtime examples move onto `kernel!`.
- **D-d.** The fluent constructors leave `Kernel`. Graphics and runtime drop
  `pixelflow-ir`. A CI check fails any crate outside the compiler crates that
  depends on it.
- **D-e.** The CLAUDE.md edits (Q5).

### Phase E: AOT

- **E1.** Declared instances (a font at declared pixel sizes), optimized at
  expansion and preloaded.
  - The build-override opt-level goes to 3. Measured: an optimize costs
    about 190 ms at opt-level 0 and 37–40 ms in release.
  - A CI check that each preloaded arena equals the runtime optimizer's
    output.

### The parallel track

- X1, arms emitted as blocks. It is urgent now, because a font program is
  mostly `if`s.
- D1 placement, which reads demand.
- The fold phase and the price of a fold (one-pipeline M13–M15).

---

## 4. Open questions for JP

**Q1. Answered (JP):** no tables; the control points are uniforms; the
program is the kernel for a glyph's number of control points; `if` and
bounding; recompile on zoom; caching later (§1.6–§1.8).

**Q2. Syntax vetoes.** The syntax choices are yours:
- the items block with `pub fn` entries;
- `(0..N).map(|i| …).sum()`;
- iterating a family at instantiation (`pieces.map(…)`);
- `if` as the only choice;
- `const` parameters as structural;
- `integral` and `area`;
- `DX(e)`.

**Q3. `text()` and `run`** (no production caller). **Recommendation:**
delete them. A string of glyphs is a sequence of per-cell calls, the same
path as the terminal.

**Q4. AOT's first user.** **Recommendation:** a bundled font at declared
pixel sizes: its `N`s and tile extents are known at build time.

**Q5. CLAUDE.md.** These lines codify the builder or construction-time
unrolling: 17, 19, 173, 176, 209–217, 244–249, 520–529, 558 and 566–568.
The "Select contains an if" section is retitled by A6. The file is yours;
Phase D proposes the edits.

---

## Appendix. Found along the way

- **A panic in font loading (F).** `Font::glyph_scaled_by_id` panics at
  `ttf.rs:507` on some glyph ids that no cmap entry reaches (NotoSansMono,
  iterating ids from 0). It is a slice index where a `None` belongs.
- **A silent wrap (F).** `Param(u8)` (`arena.rs:58`, `lower.rs:39`) wraps past
  256 parameters. B3 deletes it.
- **A second parser (F).** `pixelflow-pipeline/src/training/factored.rs:542`
  parses kernel code for `validate_corpus` (D17).
