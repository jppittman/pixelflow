# The language is `kernel!`

## Metadata
- **Author**: JP (direction), Claude (draft)
- **Status**: `Proposed`. Phase A is decision-independent and in progress.
  Phase D waits on Q1.
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
  - Anything bound at runtime goes through the same `P` in the JIT: a new
    `N`, a new shape, or a kernel-typed argument.
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
| `[R; N]`, `[f32; M]` | a uniform family (§1.6) | a family of scalar uniforms |
| `impl Fn(f32, f32) -> f32` | a kernel-typed parameter (Phase D) | a hole spliced at instantiation |
| `u32` bits | packed words (Phase D) | `Bits` ops |

**Masks are typed.** Today `X.select(Y, 7.0)` compiles and blends a number as
a mask. F: probe p16 gives 5. After this plan it is a type error.

### 1.4 Binding times

| parameter | example | binding | in the key? |
|---|---|---|---|
| structural | `const N: usize` | at instantiation; each value is its own program | yes |
| uniform | `s: f32`, `b: Bounds`, `pieces: [Row; N]` | per call, through the entry's `Args` record | no |
| kernel-typed | `k: impl Fn(f32, f32) -> f32` | at runtime; composed, then `P` | the composed program's |

- **Everything that is not structural is a uniform.** An `f32` argument no
  longer folds into a constant because of its type at the call site; today
  the call-site type decides (`emit.rs:79-100`).
- **Structural parameters may be `f32` as well as `usize`.** They are keyed by
  their bits, which is how a test pins a warp by a constant offset
  (`glyph_area_edge_cases.rs:79-89`).
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

**Ranges are constant** (JP): `a` and `b` are expressions over literals and
structural parameters.

- **Binder slots** are assigned inside-out, the lowest slot free in the body,
  as `bind_fresh` does (`kernel.rs:797-826`).
- **Unrolling is the e-graph's** (`HalveFold`, `PeelFold`, `EmptyFold`;
  one-pipeline §1.4). The syntax never unrolls.

### 1.6 Tables are uniforms

A uniform family `pieces: [Row; N]` is 10N scalar uniforms. A read
`pieces[i].sigma` is element `10·i + 6`.

- **What it is, named plainly.** A family of constant size, read only at
  points affine in constant-range binders, is what GLSL ES 1.00 calls a
  uniform array indexed by constant-index expressions: no runtime length, and
  no index that depends on data.
- **Why it is allowed.** JP: "It is a uniform no? The rule is that ranges have
  to be constant."

**The index is a child expression, not a typed field.** It is built from
`Var(binder)`, `Const`, `Mul` and `Add`: the shape `Gather` has today, with a
family where the `Buffer` was.

- Sema proves that it is affine in binders and in range, and that it stays
  below 2²⁴, so it is exact in an `f32` lane. The range is known because
  every range is constant.
- **Why not a typed `AffineIndex` field (F).** Every walk that tracks binders
  reads variance from children:
  - leaf variance is constant (`node.rs:183`, `variance.rs:367`);
  - `HalveFold` and `PeelFold` skip classes whose variance lacks the binder
    (`fold_rules.rs:58-65`, `:291`);
  - `FactorFold` splits factors by the same fact;
  - codegen places values by it (`emit/mod.rs:2811-2821`).
- **So a typed index would be invisible to all of them.** Halving would then
  read the same row twice and peeling would leave a binder unbound: silently
  wrong pixels.
- **The child expression is correct by construction** under every existing
  walk, and the emitter lowers exactly this shape, a broadcast load
  (`emit/mod.rs:2472-2483`).
- A typed index can come back once the IR's binders carry an index type. It
  has to arrive with that, not before.

A fully unrolled fold's indices are constants, so each read is a scalar
uniform at a static offset.

### 1.7 The glyph

```rust
kernel! {
    /// One oriented monotone arc: one row of the piece table.
    pub struct Row {
        pub x0: f32, pub e0x: f32, pub e1x: f32,
        pub y0: f32, pub e0y: f32, pub e1y: f32,
        pub sigma: f32, pub s: f32,
        pub lo: f32, pub hi: f32,
    }
    pub struct Bounds { pub x0: f32, pub y0: f32, pub x1: f32, pub y1: f32 }

    pub const COVERAGE_SNAP: f32 = 1.0 / 1024.0;
    const NEARLY_ONE: f32 = 1.0 - COVERAGE_SNAP;
    const PIXEL_CENTER: f32 = 0.5;

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
    fn piece_term(p: Row, x: f32, y: f32) -> f32 {
        let term = p.sigma * area(|u, v| left_of_the_arc(p, x + u, p.s * y + v));
        if (y > p.lo) & (y < p.hi) { term } else { 0.0 }
    }

    fn inside(b: Bounds, x: f32, y: f32) -> bool {
        (x >= b.x0) & (x <= b.x1) & (y >= b.y0) & (y <= b.y1)
    }

    fn coverage(f: f32) -> f32 {
        let c = f.abs().min(1.0);
        if c >= NEARLY_ONE { 1.0 } else if c <= COVERAGE_SNAP { 0.0 } else { c }
    }

    /// Texel (i, j) holds coverage at (i+½, j+½).
    pub fn texel<const N: usize>(pieces: [Row; N], bounds: Bounds) -> f32 {
        let x = X + PIXEL_CENTER;
        let y = Y + PIXEL_CENTER;
        let f = (0..N).map(|i| piece_term(pieces[i], x, y)).sum();
        if inside(bounds, x, y) { coverage(f) } else { 0.0 }
    }
}
```

**Measured against production (F, from the sketch's probe).** The fold lowers
to the same DAG, checked by an order-independent structural hash, for a
square and for an O of four quadratics, with and without the half-pixel
shift. The program shrinks from 99 to 87 nodes and from 8 to 4 uniforms. The
difference is one-pipeline's G2: production mints the box twice.

**Where it is not the same (F, all 94 inked ASCII glyphs at 7, 16 and 32 px).**

- **The table read.** `Gather(Buffer, …)` becomes a family read, which is a
  different node with the same load shape.
- **Exact N against padded.** 9 / 29 / 35 texels differ, by at most 1.19e-7.
  No u8 texel differs. It gives 28 to 35 programs per size where padding
  gives 6.
- **The key.** It changes even for an identical DAG, because `canonical`
  follows insertion order (A3 fixes that).

**What stays host Rust.** None of this is program; it produces the uniform
values.

- font parsing, compound glyphs and mirroring (`ttf.rs`);
- the f64 monotone split (`monotone.rs`);
- `piece_row`'s rounding, together with its debug asserts;
- the box's host union;
- layout.

**An empty glyph stays distinct from a missing one.** Today a space bakes
into its own atlas slot (`atlas.rs:163-200`), and the atlas golden pins that
layout.

### 1.8 What the macro emits

**The JIT case, always.**

- Records become host structs (`#[repr(C)]`, `Record::write`), so host and
  kernel share one layout definition.
- Each entry becomes a host function: `texel(pieces: &[Row], bounds: Bounds)
  -> Kernel`.
- That function instantiates the lowered template, writing `N =
  pieces.len()` into fold ends and family lengths, and returns the opaque
  `Kernel`.
- The template is a replay of `ExprArena` pushes, as `emit.rs` emits today.
  It is instantiated in O(template) for any `N`.
- No optimization runs at expansion, because optimization is priced against
  a shape.

**The AOT case (Phase E).** An entry names its instances, `(N, shape)`
pairs. The same expansion runs the one optimizer on each and emits the
optimized arenas, which are preloaded into the JIT cache under the same key.

- **No machine code at expansion yet.** The tier is chosen at startup, and a
  proc macro is not told the target.
- **Preloading optimized arenas** keeps the law without per-arch byte tables.

---

## 2. Decisions

The evidence settles these; each has a reason, and JP can overturn any.

| # | decision | resolution |
|---|---|---|
| D1 | binding times | §1.4: structural, uniform, or kernel-typed, by declaration; `Args` records |
| D2 | what the macro compiles | JIT template always; declared instances optimized at expansion (Phase E); `macro_tier`, `Templates`, `ENode::Param` and `kernel_raw!` deleted (one-pipeline M1–M5) |
| D3 | tables | uniform families with a child-expression index (§1.6) |
| D4 | binders | `usize` in sema; slots inside-out; a kernel-typed argument's binders are renamed away from those live at its hole, since a fixed slot would alias an argument's own fold |
| D5 | `.at` | application is contramap (§1.2) |
| D6 | functions across blocks or crates | inlined within a block; across blocks only as kernel-typed arguments at runtime. A proc macro sees only its own tokens |
| D7 | records and tuples | flattened in the front end; record returns (`-> Rgba`) with one select on the packed word, as `packed.rs` relies on (Phase D) |
| D8 | masks and bits | types in sema only |
| D10 | `CachedGlyph`, `CachedText` | deleted: no production consumer |
| D11 | loop-carried iteration | refused in the syntax. The two fractal benches (`shader_bench`) stay on the IR as compiler research |
| D12 | binder-indexed immediates | refused; the packer names its four channels |
| D13 | where production kernels live | above pixelflow-core. The cell grid is terminal-shaped and leaves core (CLAUDE.md: no terminal logic in PixelFlow) |
| D14 | `kernel_raw!` | deleted. Every JIT compile optimizes (`jit_cache.rs:145`), so its promise never reached machine code |
| D15 | spellings | §1.5's folds; `if` is the only select; `DX(e)`/`DY(e)` as today; the `.select`, `.lt`, … aliases go once the five CI-contract bodies are rewritten, keeping their names |
| D16 | public surface | an opaque `Kernel`; `Uniform`, `Scalar`, `Monoid` and `Bits` leave. `__macro` narrows to what expansions name. Only compiler crates depend on `pixelflow-ir`, enforced by CI |
| D17 | the second parser | `training/factored.rs`'s `parse_kernel_code_arena` and its printer are deleted with the corpus tool that uses them, or routed through the one parser if that tool is still needed |

---

## 3. Migration

Every CL is green on its own. Byte-neutral CLs record a `byte_probe` diff,
and no digests are committed (one-pipeline §5, gate policy).

### Phase A: foundations, independent of every open question

- **A1. The one parser's bugs (F, measured).**
  - `{ let X = Y; X }` evaluates X, because lowering matches `"X"` before
    locals.
  - Nested `let`s leak: `locals` is one flat map (`lower.rs:69`,
    `:259-265`).
  - An out-of-scope local is accepted.
  - Literals round twice, f64 then f32 (`lower.rs:98-103`).
  - The parser's grammar doc is stale.
- **A2. `vzeroupper` before `ret` on x86** (one-pipeline A6). It measured
  155–170 ns per call without it and 6–9 ns with it.
- **A3. `canonical` independent of insertion order.** It walks post-order
  from the root and hash-conses structurally equal subterms. Then two
  constructions of one DAG key alike, and §1.7's equivalence becomes a CI
  check.
- **A4. The uniform chain at 64 bits:** `UniformId`, `dense_slot`,
  `ScheduledOp::Uniform` and `emit_uniform_load`'s offset. The encoders
  narrow where the hardware does. A glyph at N = 189 needs 1,894 slots, and a
  table of cells needs far more; today `declare_uniform` asserts below
  `u16::MAX` (`arena.rs:700-704`).
- **A5. Fold ends at 64 bits** (`RangeFold`'s `u32`, `fold.rs:384-385`).

### Phase B: the syntax grows the glyph's constructs

- **B1.** The items block, `if` as select, typed masks, `const` items and
  helper `fn`s.
- **B2.** Folds over constant ranges, and the binder type.
- **B3.** Binding times and `Args`, records, and uniform families.
- **B4.** `integral`, `area` and `monotone_root`.
- **B5.** Lowering calls `pixelflow-ir`'s definitions, and `lower.rs`'s
  copies go.
- **B6.** The equivalence gate: a `kernel!` glyph and the builder's glyph
  give one key and identical pixels over ASCII at 7, 16 and 32 px.

### Phase C: the glyph is written in `kernel!`

- **C1.** `loop_blinn.rs` becomes the §1.7 block. It uses exact N and one
  box, and keeps "no ink" distinct from "no glyph".
  - The gates are `glyph_is_closed`, `glyph_exact_area`,
    `glyph_area_edge_cases`, `freetype_oracle` and the goldens.
  - Re-baselined pins go in their own commit.
- **C2.** The glyph's tests move onto `kernel!` (the sketch's bucket B, 15
  files).
- **C3.** `text()` and `run` (Q3).

### Phase D: the builder goes internal (waits on Q1)

- **D-a.** Kernel-typed parameters, with the capture-avoiding splice (D4).
- **D-b.** Record returns, `u32` bits, and the packed frame.
- **D-c.** Scenes, the cell grid, ML and the runtime examples move onto
  `kernel!`.
- **D-d.** The fluent constructors leave `Kernel`. Graphics and runtime drop
  `pixelflow-ir`. A CI check fails any crate outside the compiler crates
  that depends on it.
- **D-e.** The CLAUDE.md edits (Q5).

### Phase E: AOT

- **E1.** Declared instances, optimized at expansion and preloaded.
  - The build-override opt-level goes to 3. Measured: an optimize costs
    about 190 ms at opt-level 0 and 37–40 ms in release.
  - A CI check that each preloaded arena equals the runtime optimizer's
    output.
- **E2.** A bundled font at declared sizes, as the first user.

### The parallel track (one-pipeline)

- the fold phase and the price of a fold (one-pipeline M13–M15);
- D1 placement, which reads demand;
- X1, arms emitted as blocks.

---

## 4. Open questions for JP

**Q1. The frame (blocks Phase D).** The cell grid reads data-dependent
indices: `cells[⌊x·cells_per_point⌋]` and the atlas at `u0 + lx·density`,
where `u0` is itself cell data (`cell_grid.rs:440-472`).

- **What your rule settles.** Under constant ranges the cells become a fold,
  so a zoom recompiles, once per level, through the cache. That overrides
  `cell_grid.rs:18-36`'s reason for making the metric a uniform.
- **What is left: the glyph a cell shows is data.** The frame needs one of:
  - a data-indexed read (the atlas as the schedule's memo, or glyph →
    table);
  - one program padded to `N_max`;
  - one program per `N`, dispatched per region.
- **Recommendation:** keep the atlas as the schedule's memo, one declared
  tabulation read, until D1 and the scheduler make an atlas-free frame fit
  the budget.

**Q2. Syntax vetoes.** The syntax choices are yours:

- the items block with `pub fn` entries;
- `(0..N).map(|i| …).sum()`;
- `[Row; N]` tables;
- `if` as the only select;
- `const N` as structural;
- `integral` and `area`;
- `DX(e)`.

**Q3. `text()` and `run`** (no production caller). A run written as one
fold over all its pieces is exact but 2.6–7.7× slower at 3 or more
characters until the emitter jumps a select inside a fold (X1). Today's
shape, one fold per character over ragged spans, needs an IR addition the
syntax cannot say. **Recommendation:** port to one fold and let X1 recover
the speed, rather than keep a program that exists only on the builder.

**Q4. AOT's first user.** No production kernel is bound at build time
today. The font is loaded by path, and the tile extent is the runtime cell
height (`terminal_app.rs:67`, `:268`). **Recommendation:** a bundled font at
declared sizes. Confirm that "compiled as a macro" means declared
instances.

**Q5. CLAUDE.md.** Lines 17, 19, 173, 176, 209–217, 244–249, 520–529, 558
and 566–568 codify the builder or construction-time unrolling. The file is
yours, and Phase D proposes the edits.

---

## Appendix. Found along the way

- **A panic in font loading (F).** `Font::glyph_scaled_by_id` panics at
  `ttf.rs:507` on some glyph ids that no cmap entry reaches (NotoSansMono,
  iterating ids from 0). It is a slice index where a `None` belongs.
- **Silent parameter wrap (F).** `Param(u8)` (`arena.rs:58`, `lower.rs:39`)
  wraps silently past 256 parameters. B3 deletes it, because parameters
  become uniforms.
- **A second parser (F).** `pixelflow-pipeline/src/training/factored.rs:542`
  parses kernel code for `validate_corpus` (D17).
