# The language is `kernel!`

## Metadata
- **Author**: JP (direction), Claude (draft)
- **Status**: `Proposed`. Revised the same day after JP's rulings on Q1:
  one program per font per zoom level, there are no tables, the control
  points are uniforms, and `select` is renamed `if` (§1.6, §1.7). Revised
  again 2026-10-01: no arrays (below). Phases A and B are done but B7; Phase C waits on O1–O4.
  2026-10-01: JP answered O1 and O2 "Yes and yes". Both are built: D-a,
  the O2 half (`feat(compiler): kernel-typed parameters, applied as
  contramap`), and O1, each glyph its own optimization unit in `P`
  (`feat(search): a named kernel is its own optimization unit`; §4, O1).
- **No arrays** (2026-10-01). JP: *"Why do we have any arrays at all?"*
  and *"No arrays at all.. please go read recent docs about how this ought
  to work."*
  - **Where the plan went wrong (F, git): `75b7e8f3`.** JP had corrected
    one thing in `10b76d53`: the control points are uniforms, not
    constants. `75b7e8f3` changed two more. It replaced the font program
    with one program per control-point count `N`, which Q1 had offered JP
    (`739e8cb0`: "one program per `N`, dispatched per region") and JP had
    not chosen. And it brought back, as a "family" `[Row; N]`, the uniform
    array `10b76d53` had dropped. It read JP's atlas sentence the way
    one-pipeline already had ("One program per control-point count",
    written before the Q1 ruling). B3's second half then built the family
    (`66c1f4a1`, `8b07cec1`).
  - `10b76d53` was not array-free either: it had structural lists
    (`Glyph { pieces: [Row] }`, `Font { glyphs: [(u32, Glyph)] }`,
    `FONT.by_id`). Its font program and its `if id < k` tree were right.
  - **This revision** restores one program per font per zoom level with no
    collection type anywhere. A piece is one instance of one entry over its
    own ten uniforms, a glyph is its pieces summed under its box, the font
    is its glyphs under an `if id < k` tree, and host Rust composes them
    (§1.3, §1.6–§1.8). The families are superseded (B3) and deleted
    (`refactor(compiler): kernel! has no collection types`). What the documents do not settle is listed as open,
    not decided (§4, O1–O4).
- **`.at` is back, as first-class contramap** (2026-10-06; D5 overturned).
  JP: *"I want first class contramap support."* D5 answered `.at` with
  helper application, which is contramap on *functions*: a helper written
  over its coordinates, or a kernel-typed parameter. It is not contramap on
  *fields*. Every expression is already a field, and a field bound to a
  name — a `let`, a helper's parameter, a kernel's application — could not
  be observed anywhere but the sample without being rewritten as a function
  first. `f.at(x, y)` is that observation, for any `f32` or `bool`, and
  lowers to `ExprArena::warp`, the one arena-level definition a kernel's
  application goes through too. §1.2's rule is unchanged — `X` and `Y`
  appear only in entries — because `.at` reads neither: it rebinds them in
  its receiver. Pinned in `pixelflow-compiler/tests/at_is_contramap.rs`.
- **Integrals deleted** (2026-09-29). JP: *"just do b. delete all the
  integral stuff. other languages don't try this. probably for good
  reason."* The language has no integral: §1.5's `integral`, `area` and
  `monotone_root` rows, B4's integral half, B6 and D19 are withdrawn, and a
  glyph writes each piece's area in closed form (§1.7).
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
  is no second copy of `fract`, `hypot`, `clamp` or the derivative encoding
  (`pixelflow_ir::library`), of the binder's choice and rename
  (`ExprArena::close_over`) or of a range's bound (`Fold::admits`). Until
  B5, `lower.rs` restated each.
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
  arguments.** Application is contramap: `f(X + 0.5, Y + 0.5)` is `f` at
  `(X + ½, Y + ½)`. A helper therefore cannot read an unshifted `X` by
  accident.
- **Every expression is a field, and `.at` observes one elsewhere.**
  `d.at(x, y)` is the field `d` at `(x, y)` — contramap on a *value*, where
  application is contramap on a function: `let d = …; d.at(X + 1.0, Y) - d`
  is the neighbour's difference without rewriting `d` as a helper. It keeps
  its receiver's type (a mask stays a mask), and reads no coordinate of its
  own, so it is allowed in a helper as in an entry. A derivative observed
  elsewhere is the derivative of the warped field, the chain rule, as
  `Kernel::at` has it; under a translation that is the pointwise reading
  too (D5, overturned 2026-10-06).
- **Recursion, loops with state, `mut` and assignment are refused.** The
  language is a DAG with bounded folds.

### 1.3 Types (in sema; the IR keeps its lanes)

| type | meaning | IR |
|---|---|---|
| `f32` | a value | an `f32` lane |
| `bool` | a mask; comparisons produce it; `&` and `\|` combine it | an all-ones or all-zero lane, `OpKind::mask(bool)` |
| `usize` | a fold binder or a structural count | `Var(REDUCE_BINDER_BASE + slot)`; converted by an explicit `i as f32` |
| records | named `f32` fields | flattened at lowering |
| `impl Fn(f32, f32) -> f32` | a kernel-typed parameter: a kernel the host passes at run time, applied `k(x, y)` (Phase D-a, done) | none of its own: each application splices the argument's term, `k[X := x, Y := y]` (`ExprArena::apply`) |
| `u32` bits | packed words (Phase D) | `Bits` ops |

**No collection types** (JP, 2026-10-01: *"No arrays at all."*). The
language has no arrays, families, lists or tables. A count of things is not
a type. It is how many instances the host composed (§1.7).

**Masks are typed.** Today `X.select(Y, 7.0)` compiles and blends a number as
a mask. F: probe p16 gives 5. After this plan it is a type error.

### 1.4 Binding times

| parameter | example | binding | in the key? |
|---|---|---|---|
| structural | `const N: usize`, a zoom level's tile extent; the font's shape, meaning which glyphs and how many pieces each (I) | at instantiation, or by what the host composes; each value is its own program | yes |
| uniform | a piece's ten coordinates and a glyph's box, written once per font and zoom; a cell's glyph id, origin, `fg` and `bg`, per call | through the program's block: an entry's `Args` record, or for the composed font, O3 | no |
| kernel-typed | `k: impl Fn(f32, f32) -> f32` | at runtime; composed, then `P` | the composed program's |

- **A kernel-typed argument is admitted when the host function is called**
  (F, D-a). The host function takes it as a `&Kernel`, and builds its
  program then: it runs lowering's steps, the IR calls the macro makes at
  expansion for any other entry. The composed program declares the entry's
  own uniforms first, in `AnalyzedKernel::parameters` order, then each
  argument's, in parameter order and that argument's own declaration
  order, read or not; an instance passed twice is declared once, where it
  first appears. That is `Kernel`'s rule for a composition's operands
  (B3a). Its key is the composed term's, and nothing combines the
  template's key with the argument's.
  - **O3 reads off this order.** Where an instance's slots land is its
    position in the composition: for `glyph(ink, bounds)` with `ink` a
    balanced `sum2` tree of `one_piece` instances, the box is slots `0..4`
    and piece `k` is slots `4 + 10k .. 14 + 10k` (F, pinned by
    `kernel_copy.rs`). So the walk that composes the font can return each
    instance's offset as a prefix sum over what it composed. Not built
    (O3).
  - An entry that takes a kernel has no `Args` record: its block declares
    the argument's uniforms too, so a record of the entry's own would
    rebind only a program whose argument declares none. Binding a
    composed program is O3.

- **Everything that is not structural is a uniform, and a uniform is a
  scalar.** An `f32` argument no longer folds into a constant because of its
  type at the call site; today the call-site type decides (`emit.rs:79-100`).
- **The control points are uniforms** (JP), **and the program is the font's
  at one zoom level** (JP, Q1).
  - Each piece is an instance of one entry over its own ten uniforms.
    Identity is by instance, so two pieces are two factors of the block
    (uniform-slot-identity §3).
  - The control points and the boxes are written once per font and zoom. A
    cell writes about six uniforms (§1.7).
  - A new font or a new tile extent is a new program: recompiled, cached by
    key.
- **Each entry has an `Args` record.** A compiled program is bound from
  `&Args`, and "every argument supplied" is a type rather than a runtime
  assert. It replaces `Uniform` handles, `UniformBlock::set`'s linear search
  (`manifold.rs:113-125`), and the refusal at `packed.rs:185-196`.
  - **Which program is not yet a type (F, B3's first half).** `write_into`
    streams the values by position into any `UniformBlock`
    (`set_declared`), and the one check is the count (`ArityMismatch`). One
    entry's `Args` written into a block of another program that declares as
    many scalars binds without a word and draws plausible wrong pixels, a
    confusion the identity-keyed `set` could not make. The follow-up ties a
    block to the entry it was compiled from: a block typed by the entry
    (`UniformBlock<A>`, made by compiling that entry's kernel), or a
    per-entry token the kernel carries and `write_into` checks.
  - **A composed program has no entry of its own.** Binding the font, made
    from many entries' instances, is O3.

### 1.5 Folds

| spelling | denotation | IR |
|---|---|---|
| `(a..b).map(\|i\| e).sum()`, `.product()`, `.any(\|i\| m)`, `.all(\|i\| m)`, `.fold(f32::INFINITY, f32::min)`, `.fold(f32::NEG_INFINITY, f32::max)` | ⊕ over `i ∈ [a, b)`; the identity if empty | `Reduce(Fold{monoid, binder, a..b}, e)` |

**There are no integrals.** JP, 2026-09-29: *"just do b. delete all the
integral stuff. other languages don't try this. probably for good
reason."* The table had three more rows, landed by B4 (`4b7ea263`):
`integral(lo..hi, |u| e)` over an interval fold, `area(|u, v| e)` as two
of them over the pixel, and the intrinsic `monotone_root(δ, step, bend)`. They were deleted with everything
that gave them meaning — the IR's interval domain (`Fold::Interval`), the
builder's `Kernel::area`, the e-graph's integration rules and their closing
phase, and the quadrature that legalized what the rules left open. The
e-graph derived each piece's closed form by rewriting, so a glyph was
correct only as far as its saturation budget reached: under the flat class
cap a `kernel!` glyph of 189 pieces had every integral quadratured, its
coverage off by up to 0.92, and no test failed. A glyph writes the closed
form itself (§1.7), and so defines its own `monotone_root`, a helper like
any other.

- **Ranges are constant** (JP): `a` and `b` are expressions over literals and
  structural parameters.
- **Binder slots** are assigned inside-out, the lowest slot free in the body,
  by `ExprArena::close_over`, which `Kernel::over` and `kernel!`'s lowering
  both close a fold through (B5).
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

- **No arrays, no uniform arrays, no buffers, no `Gather` read by a
  program's author** (§1.3). Data enters a program in one of two ways:
  - as a scalar uniform in the program's block, written per call or once
    per font and zoom;
  - as structure (§1.4): the tile extent, and the font's shape, which fixes
    how many instances the host composes and so how many uniforms there
    are.
- **Choosing among alternatives is `if`.** A tree of `if`s over a uniform
  (`if id < k { … } else { … }`) is a binary space partition over it. So is
  a tree of bounding tests over space: a glyph's box, a piece's band.
  - **The worked example is the font (§1.7).** A cell's glyph id is one
    uniform. The font is its glyphs under a balanced tree of `if id < k`,
    so choosing a glyph takes about log₂ G tests for G glyphs. Inside the
    chosen glyph, its box and each piece's band, `(y > lo) & (y < hi)`, cut
    space. A piece's term is exactly zero outside its band (F,
    `fonts/loop_blinn.rs`'s `piece_term`: "The cut to the rows is an
    identity"), so that cut changes no bit.
  - The partition falls out of `if` and bounding, and nobody builds it as a
    structure. No table maps an id to a glyph, and no host lookup chooses a
    program.
  - A mask that is uniform across a batch takes one arm, which is a jump.
    Only a mask that varies by lane blends. The id is the same for the
    whole call, and a band reads `y` alone, which is uniform over a batch
    (the same doc). So their arms are jumps. The emitter does this once X1
    lands (below).
- **`Select` is renamed `If`**, in the IR, the e-graph, the emitter and the
  docs. CLAUDE.md needs a section, "Select contains an if", to explain what
  the name hides. The emitter was built on the misreading, blend by default
  with a branch bought per select (`emit/guards.rs`), and that cost 73% of a
  glyph bake (docs/BACKLOG.md X1) and the slowness of runs. The name is the
  bug (A6).

### 1.7 A font is one program per zoom level

```rust
kernel! {
    /// One oriented monotone arc piece: ten uniforms.
    pub struct Row {
        pub x0: f32, pub e0x: f32, pub e1x: f32,
        pub y0: f32, pub e0y: f32, pub e1y: f32,
        pub sigma: f32, pub s: f32,
        pub lo: f32, pub hi: f32,
    }
    /// A glyph's box: the outline's bounding box, four uniforms. Coverage
    /// reaches half a pixel past it, which `inside` reaches too, so the
    /// host passes the outline's own box.
    pub struct Bounds { pub x0: f32, pub y0: f32, pub x1: f32, pub y1: f32 }

    const PIXEL_CENTER: f32 = 0.5;
    const PIXEL_HALF: f32 = 0.5;
    const ONE_THIRD: f32 = 1.0 / 3.0;
    const ROOT_FLOOR: f32 = 1.0 / 1_267_650_600_228_229_401_496_703_205_376.0;
    const COVERAGE_SNAP: f32 = 1.0 / 1024.0;
    const NEARLY_ONE: f32 = 1.0 - COVERAGE_SNAP;

    fn coverage(f: f32) -> f32 {
        let c = f.abs().min(1.0);
        if c >= NEARLY_ONE { 1.0 } else if c <= COVERAGE_SNAP { 0.0 } else { c }
    }

    /// `τ(δ) = δ / max(step + √max(step² + bend·δ, 0), ROOT_FLOOR)`: the
    /// parameter at which the rise `t·(2·step + bend·t)` reaches the height
    /// `δ`, the reciprocal exact — `fonts/loop_blinn.rs`'s
    /// `Rise::monotone_root`, which carries its law.
    fn monotone_root(delta: f32, step: f32, bend: f32) -> f32 {
        delta * (1.0 / (step + (step * step + bend * delta).max(0.0).sqrt()).max(ROOT_FLOOR))
    }

    /// `2·step + bend·s`: a rise `q(t) = t·(2·step + bend·t)` is `t` times
    /// it at `s = t`, and climbs `(t − s)` times it at `s + t` from `s` to `t`.
    fn slope_through(step: f32, bend: f32, s: f32) -> f32 {
        step + step + bend * s
    }

    /// The area of the pixel about (x, y) left of the arc, within its band,
    /// in closed form (`fonts/loop_blinn.rs`, `RisingArc::pixel_area`, has
    /// the derivation).
    fn piece_area(p: Row, x: f32, y: f32) -> f32 {
        let b = p.e0y.max(0.0);
        let bx = p.e0x.max(0.0);
        let a = p.e1y.max(0.0) - b;
        let ax = p.e1x.max(0.0) - bx;
        let across = x - p.x0;
        let up = y - p.y0;
        let left = across - PIXEL_HALF;
        let right = across + PIXEL_HALF;
        // Where the arc enters and leaves the pixel's rows, and where it
        // reaches the pixel's left and right edges.
        let t0 = monotone_root(up - PIXEL_HALF, b, a).clamp(0.0, 1.0);
        let t1 = monotone_root(up + PIXEL_HALF, b, a).clamp(0.0, 1.0);
        let t_left = monotone_root(left, bx, ax).clamp(t0, t1);
        let t_right = monotone_root(right, bx, ax).clamp(t0, t1);
        let right_of_the_pixel = (t1 - t_right) * slope_through(b, a, t_right + t1);
        let x_left = t_left * slope_through(bx, ax, t_left) - left;
        let x_right = t_right * slope_through(bx, ax, t_right) - left;
        let rise = (t_right - t_left) * slope_through(b, a, t_left + t_right);
        let trapezoid = PIXEL_HALF * (x_left + x_right) * rise;
        let w = t_right - t_left;
        let bow = ONE_THIRD * (bx * a - b * ax) * (w * w * w);
        right_of_the_pixel + (trapezoid + bow)
    }

    /// σ·A over the pixel about (x, S·y), cut to the rows the piece reaches.
    fn piece_term(p: Row, x: f32, y: f32) -> f32 {
        let term = p.sigma * piece_area(p, x, p.s * y);
        if (y > p.lo) & (y < p.hi) { term } else { 0.0 }
    }

    /// Whether the pixel about (x, y) can meet the outline: its centre
    /// within half a pixel of the outline's box. Every piece's term is
    /// exactly 0 farther out.
    fn inside(b: Bounds, x: f32, y: f32) -> bool {
        (x >= b.x0 - PIXEL_HALF) & (x <= b.x1 + PIXEL_HALF)
            & (y >= b.y0 - PIXEL_HALF) & (y <= b.y1 + PIXEL_HALF)
    }

    /// One piece's term at the sample, over its own ten uniforms. The host
    /// composes one instance per piece.
    pub fn one_piece(p: Row) -> f32 {
        piece_term(p, X, Y)
    }

    /// Two kernels summed at the sample: the operation of the monoid a
    /// glyph's ink is, which the host folds its pieces' instances with — a
    /// balanced tree, by index.
    pub fn sum2(a: impl Fn(f32, f32) -> f32, b: impl Fn(f32, f32) -> f32) -> f32 {
        a(X, Y) + b(X, Y)
    }

    /// One glyph. Texel (i, j) holds coverage at (i+½, j+½). `ink` is the
    /// sum of the glyph's pieces, composed by the host, and `ink(x, y)`
    /// reads it at the pixel's centre: application is contramap (§1.2).
    /// `ink` is kernel-typed (Phase D-a).
    pub fn glyph(ink: impl Fn(f32, f32) -> f32, bounds: Bounds) -> f32 {
        let (x, y) = (X + PIXEL_CENTER, Y + PIXEL_CENTER);
        if inside(bounds, x, y) { coverage(ink(x, y)) } else { 0.0 }
    }
}
```

**The atlas becomes the font program.** JP: *"The 'atlas' becomes the kernel
for that number of control points, everything else is a uniform."* The atlas holds one font at one tile
size (F, `atlas.rs:46-60`: "An atlas is bound to ONE font", rebuilt "on
cell-size and density changes"). So the kernel that replaces it is the
font's at one zoom level (I), which is what JP's Q1 answer says.

**What the language defines** is the block: a piece's record and its term
in closed form, a glyph's box test, and coverage. Nothing in it is a
collection, and nothing in it names a count.

**What the host composes** is the font. The font is runtime data (§1.1: a
font loaded at runtime), and the language has no collection to hold it
(§1.3). So the walk over the parsed font is host Rust, and it composes:
- **a piece:** one instance of `one_piece` over its own ten uniforms;
- **a glyph:** `glyph(ink, bounds)` over its box's four uniforms, with
  `ink` the sum of its pieces' instances, a balanced tree of `sum2`;
- **the font:** its glyphs under a balanced tree of
  `if id < k { lower } else { upper }`.
  - **I:** the host numbers the font's glyphs `0..G`, halves the range, and
    `k` is the first id of the upper half. `k` is a count, so it is
    structural.
  - `id` is one uniform, and every node reads it. That needs one id
    instance read at every node: an entry instantiated per node with an
    `id` parameter would mint one id per node (identity is by instance,
    §1.4), and no document says how a composition reuses one (O3).
  - Composing instances is the host walking data, not unrolling: there is
    no fold over pieces, and each instance reads its own uniforms.

That is the whole font program. The tree is the partition that `if` and
bounding make (§1.6). It is not a table, and the host chooses no program.

**The composition surface (O2) is built** (D-a, `f0599991`). The host
composes a glyph from `kernel!` entries alone: one `one_piece` instance
per piece, summed by `sum2` as a balanced tree, and `glyph(ink, bounds)`
over the sum. Nothing of the builder is on that path (F,
`kernel_copy.rs`): the rows are the font's data, and the box is the
outline's own, which `inside` reaches half a pixel past. That reach was
the builder's, in `Support`'s dilated box, until review found the test
borrowing it. Measured: with the outline's box and no reach in the block,
'A' at 16 px draws texel (3, 2) as 0 where the builder draws 0.19. The id
tree is not built, and its shape is still the font's (C1).

**Two things about it are open:**
- **The units (O1).** The font is too big for one e-graph or one emit
  (§1.8). So each glyph is its own unit, and the id tree links them.
  **Built** (O1): a glyph composed by name (`by_ref`) is saturated and
  extracted by itself and linked into the tree after extraction. Emission
  is still one program, and superlinear (§4, O1).
- **Binding (O3).** The host writes each instance's uniforms into the
  composed program's block. No document says how it finds their slots.

**Today's copy (F).** The block above is expanded from
`pixelflow-compiler/tests/common/section_1_7.rs`, and
`fonts/loop_blinn/kernel_copy.rs` pins it against the builder twice.
- `a_piece_is_one_term_through_either_definition`: `one_piece` and the
  builder's `piece_term` are one canonical key.
- `real_glyphs_composed_in_the_language_draw_the_builders_pixels`: DejaVu's
  A, O, S, g, 8 and Q at 16 px, composed as above, draw the builder
  `glyph`'s pixels to twice the closed form's error bound. They differ by
  at most 1.9·10⁻⁶ on AVX-512 and on AVX2, against a bound of 1.6·10⁻⁵ at
  its smallest.

**A piece's term is its closed form, not an integral** (JP, 2026-09-29:
*"delete all the integral stuff. other languages don't try this. probably
for good reason."*).
- The block first wrote it as
  `p.sigma * area(|u, v| left_of_the_arc(p, x + u, p.s * y + v))` and left
  the calculus to the e-graph.
- The e-graph's rules stopped closing under the class cap once a glyph had
  a few dozen pieces (§1.5), and one-point quadrature legalized whatever was
  left, with nothing to notice.
- `piece_area` is the area itself. The builder's glyph
  (`fonts/loop_blinn.rs`) was rewritten to the same closed form first.
- The integral was then deleted from the language, the IR and the e-graph
  (§1.5). `monotone_root`, the intrinsic the block used to call, went with
  it, and the block now defines it.

**What stays host Rust.** None of this is program. It produces the font's
uniforms and the program's shape:
- font parsing, compound glyphs and mirroring (`ttf.rs`);
- the f64 monotone split (`monotone.rs`);
- `piece_row`'s rounding, with its debug asserts;
- layout;
- the walk over the parsed font that composes the program (above).

**A cell is one call.**
- A cell writes its glyph id, its origin `(x0, y0)` (collapse-is-a-fold
  §2.1), `fg` and `bg`, about six uniforms (I), and calls the font program
  over its tile.
- The control points stay as written for the zoom level.
- The colour blend and the pack belong to the packed frame (Phase D-b).
- The loop over cells is host schedule until scheduling moves into the
  compiler.
- A call costs 6–9 ns once A2 lands (F, measured with `vzeroupper`), so
  12k cells is about 0.1 ms of calls.

**A zoom level recompiles the font.**
- The tile extent is structural, so a new pixel size is a new program.
- **I:** the host rescales the outlines and rewrites the font's uniforms. A
  scaling `at` will not do: the closed form bakes in the pixel, so a glyph
  kernel is correct under translation only (CLAUDE.md, "Glyph coverage").
- Caching comes later (JP).
- An empty glyph stays distinct from a missing one, as the atlas's slot
  layout does today (`atlas.rs:163-200`).

### 1.8 What it takes, and where it lands

Two problems, neither of which touches the language. A third, that every
piece was its own integral, went with the integral (§1.5).

1. **Size (O1).** One font program is too big for one e-graph or one emit.
   - Noto's ASCII has 1,625 pieces (F).
   - A 32-piece glyph inserts 3,324 classes (F, `6c588c1b`), about 100 a
     piece. The font inlined inserts **169,263** classes before any rule
     fires (F, O1's measurement; the inference here was about 160k). That
     is over `HARD_CLASS_LIMIT` (100k, `graph.rs`) and three times the
     classical ceiling (50k, `6c588c1b`).
   - Emission is superlinear. One glyph's program emits in 7 ms at 8 pieces
     and 4.4 s at 189 (F, `U_band`, measured at `8b7b75a`).
   - Code is about 1.5 KB a piece (F: `U_band`, 279 KB at 189 pieces).
     ASCII's 94 inked glyphs, each emitted alone, are 2.58 MB (F, O1).
   - **Fix: each glyph is its own unit of optimization, and the id tree
     links them**, as `10b76d53` had it. **Built** (O1). The largest Noto
     ASCII glyph is `@`, 56 pieces, and inserts 5,796 classes (F, O1); the
     189-piece glyph above is `U_band`, not an ASCII glyph.
2. **Zoom latency.** A zoom recompiles every glyph.
   - (I, measured on B3's family `glyph::<64>`) a 64-piece glyph bakes in 806–897 ms under the unpinned cap (C2),
     and one of 189 pieces emits in 4.4 s. So a zoom level takes seconds
     on one thread.
   - **Fix:** glyphs are independent units, so compile them in parallel,
     and emit arms as blocks (X1) to remove the superlinear emit.
     - Built for optimization (O1): units saturate and extract in
       parallel, and a glyph structure saturates once per process, so a
       second zoom level optimizes the ASCII font in 0.8–1.4 s with no
       saturation (F). Emission is still one program, and is the cost.
   - "We'll make computing the programs fast, and focus on the caching
     later" (JP).

**What the macro emits.**
- Records become host structs (`#[repr(C)]`).
- Each entry becomes a host function that instantiates the lowered template
  with its structural values and returns the opaque `Kernel`.
- The template is a replay of `ExprArena` pushes, as `emit.rs` emits today.
- An entry that takes a kernel cannot be lowered before its argument
  exists. Its host function is lowering's steps instead, emitted as the
  statements that take them: the same IR calls, run when it is called
  (`emit::Staged`, D-a).
- No optimization runs at expansion unless the instance is declared (Phase
  E).

---

## 2. Decisions

The evidence and JP's rulings settle these. JP can overturn any.

| # | decision | resolution |
|---|---|---|
| D1 | binding times | §1.4: structural (counts and extents), uniform (every number: a cell's per call, the font's once per font and zoom), or kernel-typed; `Args` records |
| D2 | what the macro compiles | the JIT template always; declared instances optimized at expansion (Phase E); `macro_tier`, `Templates`, `ENode::Param` and `kernel_raw!` deleted (one-pipeline M1–M5) |
| D3 | tables and arrays | **none** (JP: no tables; 2026-10-01, no arrays). No collection type: data enters as scalar uniforms, a count is how many instances the host composed, and choice is `if` (§1.3, §1.6) |
| D4 | binders | `usize` in sema; slots inside-out; a kernel-typed argument's binders are renamed away from those live at its hole |
| D5 | `.at` | ~~application is contramap (§1.2)~~ **Overturned 2026-10-06** (JP: first-class contramap). Application is contramap on a function; `f.at(x, y)` is contramap on a field, any `f32` or `bool`, through `ExprArena::warp` (§1.2) |
| D6 | functions across blocks or crates | inlined within a block; across blocks only as kernel-typed arguments at runtime (D-a, done). A proc macro sees only its own tokens |
| D7 | records and tuples | flattened in the front end; record returns (`-> Rgba`) with one `if` on the packed word, as `packed.rs` relies on (Phase D) |
| D8 | masks and bits | types in sema only |
| D9 | the frame | one font program per zoom level (JP, Q1). A cell is one call writing its glyph id, origin, `fg` and `bg`; the glyph is chosen inside the program by the `if id < k` tree; a zoom recompiles; caching later (JP, §1.7) |
| D10 | `CachedGlyph`, `CachedText`, the atlas, `BilinearSampler` | become the font program, and are deleted as the frame moves to per-cell calls, after C2's measurement (O4). Caching returns later as its own design (JP) |
| D11 | loop-carried iteration | refused in the syntax. The two fractal benches (`shader_bench`) stay on the IR as compiler research |
| D12 | binder-indexed immediates | refused; the packer names its four channels |
| D13 | where production kernels live | above pixelflow-core. The cell grid is terminal-shaped and leaves core (CLAUDE.md: no terminal logic in PixelFlow) |
| D14 | `kernel_raw!` | deleted. Every JIT compile optimizes (`jit_cache.rs:145`), so its promise never reached machine code |
| D15 | spellings | §1.5's folds; `if` is the only choice; `DX(e)`/`DY(e)` as today; the `.select`, `.lt`, … aliases go once the five CI-contract bodies are rewritten, keeping their names |
| D16 | public surface | an opaque `Kernel`; `Uniform`, `Scalar`, `Monoid` and `Bits` leave. `__macro` narrows to what expansions name (since D-a that includes `ExprArena::admit`, `apply`, `open_fold`, `close_fold` and `compact`, and `OpenFold::index`). Only compiler crates depend on `pixelflow-ir`, enforced by CI |
| D17 | the second parser | `training/factored.rs`'s `parse_kernel_code_arena` and its printer are deleted with the corpus tool that uses them, or routed through the one parser if that tool is still needed |
| D18 | `Select` | renamed `If` everywhere (§1.6; JP) |
| D19 | helpers | **withdrawn** with the integrals (§1.5): it existed to close a helper's integral once, with its parameters abstract, and a helper holds none |

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
  `ScheduledOp::Uniform` and `emit_uniform_load`'s offset. **Done** in
  `9c7e7397`: `declare_uniform`'s assertion below `u16::MAX` is gone, and
  `UniformBlock::set`'s linear search is an index.
  - A font program holds about 16k uniforms for Noto's ASCII alone (I:
    1,625 pieces at ten each, plus four per glyph's box).
  - Nothing bounds a font's size.
- **Deprioritized.** 64-bit fold ends (A5) have no driver in this plan.
  Likewise the caps A4 leaves beside the uniform chain, so the remaining
  widths stay visible: `push_nary` and the key's `Nary` child count at
  `u16::MAX`, `Fold`'s `u32` ends, `BufferId(u16)` /
  `BufferIdentity(u32)` (§1.6: buffers are leaving),
  `ScheduledOp::Context(u16)`, `ExprId(u32)` and `Binder(u8)`.

### Phase B: the syntax grows the font's constructs

- **B1.** The items block, `if` as the only choice, typed masks, `const`
  items and helper `fn`s. **Done** in `2903b1f0`.
- **B2.** Folds over constant ranges, and the binder type. **Done** in
  `92cf54df`.
- **B3.** Binding times and `Args`, records, structural counts, and tuple
  `let`s. **Done**: records, binding times and `Args` in `c2b8b1ea`; tuple
  `let`s in `66c1f4a1` (B3's second half).
  - **Families are superseded** (JP, 2026-10-01: *"No arrays at all."*).
    `66c1f4a1` and `8b07cec1` also built families of records iterated at
    instantiation: the `[R; N]` parameter (`Ty::Family`), the family
    template, and the iteration's marker uniform.
  - Deleted (`refactor(compiler): kernel! has no collection types`), with
    their tests, `a_family_is_its_copies.rs` and
    `family_args_allocate_nothing.rs`.
  - The marker also extended `Uniform`'s meaning without extending its
    type. It is a uniform declared for the iteration alone, with a NaN
    default, which emission recognizes by identity (`lower.rs`,
    `Iteration`). It survives a splice and not a rewrite.
- **B4.** `integral`, `area` and `monotone_root`. **Done** in `4b7ea263`,
  with review follow-ups in `cc0dc47b`, and **deleted** with the integral
  (2026-09-29, §1.5): the syntax, its `sema` and lowering, the reserved
  names and `F32Scope` (which existed so a bound could be evaluated like an
  `f32` const) are gone, and so is `ExprArena::close_over`'s refusal of a
  fold its caller declined.
- **B5.** Lowering calls `pixelflow-ir`'s definitions, and `lower.rs`'s
  copies go. **Done** in `85815a0f`: `library`'s `fract`, `hypot`, `clamp`
  and `derivative`, written once over the sites a term is built in and
  built through by `Kernel`'s methods, lowering and the integrals' closed
  forms; `Axis`; `ExprArena::close_over` and `Placeholder`; the 2²⁴ bound
  in `RangeFold` (docs/BACKLOG.md C8); `IntervalFold::pixel`. Keys and
  bytes unchanged. (`RangeFold` is `Fold` again, and `IntervalFold` is
  gone, since the integral's deletion.)
- **B6.** **Withdrawn** with the integrals (§1.5, D19). It was: helpers as
  optimization units, a helper's integral closed once and instantiated.
  What it found about templates stands for whatever next optimizes one:
  - A helper's template is closed over its parameters through
    `ExprArena::splice_with`.
  - Its findings about a family's template are superseded with the
    families (B3). Those were the template over shared terms and an element,
    the iteration lowered as `body + marker`, and nesting innermost first.
  - A template's inputs are `Uniform` leaves, and a uniform's variance is
    constant on the lattice (`variance.rs`, "a uniform is here"). An input
    stands for any term the copies share, `x = X + ½` included. That is
    sound for every rewrite conditioned on binder-invariance, since an input
    reads no binder its template holds (pinned by
    `an_input_reads_no_binder_its_template_holds`), but extraction would
    price `x`-dependent work as per-call. Before extracting in a template,
    seed each input's variance from the term it stands for, or give a
    template's input a leaf meaning of its own.
- **B7.** The equivalence gate: one glyph built by `kernel!` and by the
  builder gives the same pixels over ASCII at 7, 16 and 32 px.

### Phase C: the font is written in `kernel!`

- **C1.** The §1.7 block and the host's walk: one font program per zoom
  level. It is built first for Noto's ASCII at one size: per-glyph units
  (O1), the id tree, and one call per cell.
  - It waits on O1's link, and on D-a or O2's interim answer.
  - The gates are `glyph_exact_area`, `glyph_area_edge_cases`,
    `freetype_oracle`, `glyph_optimizes_estimate_free` and the goldens.
    (`glyph_is_closed` also pinned that no glyph held an integral, which
    the IR can no longer express; the rest of it is
    `glyph_optimizes_estimate_free`.)
  - Re-baselined pins go in their own commit. The atlas's bilinear read
    goes, so pixels move wherever density ≠ 1 (one-pipeline §1.6).
- **C2.** The frame calls the font program per cell. The atlas,
  `CachedGlyph`/`CachedText` and `BilinearSampler` become the font program
  and go (D10). A zoom recompiles.
  - **Gate: measure before deleting (O4).** Measure the frame against
    today's at 80×24 and 200×60, at 16 and 32 px, on both x86 tiers, and
    measure a zoom's compile. If 200×60 at 32 px misses 16.7 ms, that goes
    to JP before the atlas is deleted.
  - The unpinned classical cap (`6c588c1b`) costs a glyph of copies. B3's
    family `glyph::<64>` bakes in 376 → 897 ms on AVX-512 (415 → 806 ms on
    AVX2), with code +15–20%, for an extraction within 1.2e-7 of the
    fold's. **I:** a glyph composed of 64 `one_piece` instances is the same
    sum. The font program pays that once per glyph per zoom; measure it
    with the frame.
- **C3.** The glyph's tests move onto `kernel!`.
- **C4.** `text()` and `run` (Q3).

### Phase D: the builder goes internal

- **D-a.** Kernel-typed parameters, with the capture-avoiding splice (D4).
  **Done** in `f0599991`. See O2's answer for what was built.
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

**Q1. Answered (JP):** one program per font per zoom level; no tables; the
control points are uniforms; `if` and bounding; recompile on zoom; caching
later (§1.6–§1.8). And, 2026-10-01: no arrays at all (§1.3). What the
answers leave open is O1–O4.

**Open for JP (2026-10-01).** The documents do not settle these four. Each
is recorded as open, not decided. The recommendations are inferences (I).

**O1. The font is too big for one e-graph or one emit.** **Answered (JP,
2026-10-01): yes** — each glyph is its own optimization unit, linked by the
id tree, built into `P` before C1. Built (`feat(search): a named kernel is its own optimization unit`); below the evidence.
- **Evidence.**
  - §1.8: Noto's ASCII is 1,625 pieces (F), so about 160k classes are
    inserted before any rule fires (I, from 3,324 for a 32-piece glyph, F;
    measured since: 169,263). `HARD_CLASS_LIMIT` is 100k (`graph.rs`), and
    the classical ceiling is 50k (`6c588c1b`).
  - Emission is superlinear: 7 ms at 8 pieces and 4.4 s at 189 (F).
  - `10b76d53` §1.8 reached the same fix: each glyph is its own unit,
    linked by the `if` tree.
  - Composition only inlines. `P` as one-pipeline §1.1 denotes it begins
    with `expand_refs`, and composition-is-linking's title is "the linker
    only inlines".
- **The recommendation JP accepted.** Make the unit a property of `P`:
  saturate and extract each glyph by itself, in parallel, with the id tree
  the only term across units; build that link as its own CL before C1; and
  measure in C1 whether the units are emitted as one program (which needs
  X1 for a linear emit) or as separate code joined by calls.
- **What was built (`feat(search): a named kernel is its own optimization unit`).**
  - **The denotation.** `P(k, s, t) = emit_t ∘ legalize_t ∘ L_s(k)`, with
    `L_s(t) = link(Ô_s(t), r ↦ L_s(body r))` and `Ô_s = extract_s ∘
    saturate ∘ insert°`, where `insert°` holds each unit as a leaf. **Law
    U:** `⟦L_s(k)⟧ = ⟦expand_refs(k)⟧`, by induction over the units: a
    unit leaf carries its referent's variance, the one fact a rule reads off
    a leaf; a unit is closed over the coordinates, so its context cannot
    change what it reads (a fold around it may share its slot, and the
    inner fold shadows, as every pass already respects); and the link
    substitutes equals for equals. The proof and the mechanism are in
    `pixelflow_search::runtime`'s module docs.
  - **What marks a unit: a `Ref`** (`Kernel::by_ref`). It already denotes
    "this kernel, named by content", no shipped kernel makes one, and the
    host's walk chooses the granularity: it names each glyph, and the
    pieces stay inlined, which is where a glyph's CSE is.
    - Rejected: a new node or a flag on `Ref` (two names for one thing);
      one unit per piece (1,625 units of about 100 classes, losing each
      glyph's CSE); and reading the units off the DAG as the arms of every
      `If` whose mask is frame-uniform. That last needs no marker and
      keeps the compile key a function of `k`, but it imposes a boundary on
      every program with a uniform `If`, chosen by a structural heuristic
      rather than by the program, which moves the bytes of kernels that
      asked for nothing; and the opaque leaf is needed anyway where the
      term around the font is not an `If` (D-b's blend), so it would be a
      second marker beside the first.
  - **How a unit is optimized:** by the same stages, through the same
    structure-keyed cache, so a glyph that recurs across fonts, zoom levels
    or programs saturates once. That cache now holds one in-flight slot per
    structure, so concurrent units never saturate one structure twice.
    Units run on scoped workers pulling from one index; the result lands by
    index and the link walks the term's own order, so the worker count
    cannot reach the program (pinned node for node at 1 and 4 workers).
  - **How the term around the units treats one:** as an opaque leaf
    (`ENode::Ref { key, variance }`), admitted by the unit walk
    (`EGraph::admit_unit`, crate-private; every other path still declines
    a reference), priced 0. That price is right for a form that mentions a
    unit twice (the link splices it once) and blind to a rewrite that
    changes how often one is evaluated; the font's outer term is an id
    tree and offers no such choice. A unit is still hoisted out of a fold
    it does not read, on its variance (pinned).
  - **When units are linked:** after extraction, before legalization, by
    one walk shared with `expand_refs` (`passes::link`, the one new public
    function: the runtime tier links optimized bodies through it rather
    than through a copy). `lower_dwrt` runs once, on the linked program.
    Emission stays one program.
  - **A reference a `Dwrt` reaches is no unit:** it is linked as written
    before insertion, so the chain rule runs in the graph as before
    (pinned: `∂(x²)/∂x` through a name).
  - **A decline narrows.** A unit the e-graph cannot hold is linked as
    written while the rest optimize, and saturation telemetry records every
    decline (`record_decline`). The compile falls back to the *expanded*
    arena when nothing optimized, so a `Ref` never reaches the emitter.
  - **Shape** enters as before: every unit and the term around them are
    extracted at the program's shape, per call; saturation is shape-free.
  - **The compile key names where the units are.** `body` and
    `body.by_ref()` expand alike and are different programs, so the key of
    a program of units appends its own canonical bytes, in which a `Ref` is
    its referent's identity. Without a `Ref` the key is unchanged.
- **Measured (F; release, AVX-512, 4 threads; Noto Sans Mono ASCII, the
  §1.7 composition: one `one_piece` instance per piece over its ten
  uniforms, summed under each glyph's box, each glyph a unit, under a
  balanced id tree; 16 px).**
  - 94 inked glyphs, 1,625 pieces, 35 distinct piece counts; the largest
    glyph is `@`, 56 pieces.
  - Inlined, the program inserts 169,263 classes and stops on the class
    cap after 0 rounds: no rule fires anywhere in the font, and its
    optimization (2.6 s) hands back the 169,263-node input.
  - With units: **36 saturations** (35 glyph structures, since the control
    points are uniforms, and the id tree), each under its own cap: the
    largest, `@`, inserts 5,796 classes against a cap of 46,368 and stops
    there after 2 rounds in 718 ms; the id tree inserts 374 and quiesces.
    The 36 saturations sum to 5.9 s of CPU; **the whole font optimizes in
    2.1–2.5 s wall clock** over two runs (walk, 36 saturations, 95
    extractions, link).
    The link gives 164,338 nodes against 169,263 inlined.
  - A second zoom level in the same process (32 px) saturates nothing and
    takes 0.8–1.4 s: extractions and the link.
  - Emission as one program, through the whole compile (the units'
    saturations already cached by the font's, the inlined subsets
    saturating their own), units against inlined: 4 glyphs (70 pieces)
    113,168 B in 0.64 s against 114,240 B in 1.28 s; 8 glyphs (157)
    253,984 B in 4.0 s against 227,360 B in 4.3 s; 16 glyphs (232)
    376,832 B in 12.3 s against 326,520 B in 13.5 s; 32 glyphs (599
    pieces) with units, 970,028 B in 166 s. The 94 glyphs each emitted
    alone are 2.58 MB in 2.7 s, the slowest `@` at 0.36 s.
  - So units take the font from "nothing fires" to every glyph saturated
    under its cap in 2.5 s, and **emitting the font as one program is
    C1's problem**: it grows about cubically in the pieces (the design
    measured 49 minutes for all 94 at `d8c39481` in a scratch crate), while
    the same code emitted glyph by glyph is under 3 s. X1, or units emitted
    as calls, is the lever; this CL decides neither.
  - Bytes: at 8 and 16 glyphs the units program is 12–15% larger than the
    inlined one (1% smaller at 4). The inlined runs there stopped on the class cap after one
    round; each unit ran two. The latency prior minimizes cycles, not
    bytes, so this is not a measured loss of speed; time per cell is C2's
    measurement (O4).
- **What is lost.** Rewriting across a unit's boundary: constants,
  algebra, and CSE of equal-but-not-identical terms between a glyph and
  its context, and hoisting part of a glyph. For the font the term around
  the glyphs is the id tree and a cell runs one glyph, so nothing crossing
  glyphs is on an executed path (I). Identical subterms still share: the
  link splices into a hash-consed arena.
- **Where the design was not followed, and why: the compile key names
  units by value.** An adversarial review asked for a structural key that
  walks through the units (a `refs` link table beside `buffers` and
  `uniforms`, relinked like them). Without it:
  - (a) a font rebuilt over fresh uniforms is a new compile-cache entry
    (about 2.6 MB of code) where the same structure without units would
    hit, and the cache never evicts;
  - (b) a preloaded program (Phase E) cannot hit a run-time compile of a
    program of units, because a build and a run mint different identities.
  - It was kept because the store already grows the same way: `by_ref`
    interns each glyph by value, and the store never evicts either, so a
    structural compile key alone would not stop a rebuilt font from
    growing the process. Both are the caching JP put later. An
    over-specific key misses sharing and never shares wrongly. **Before
    E1** a program of units needs the structural key (or E1 declares
    instances without units).
- **Left for others.**
  - **D-a / C1: one application rule.** Units survive only if a glyph
    reaches `P` as a reference. `Kernel::at` expands a reference at
    construction, so a kernel-typed argument applied at exactly `(X, Y)`
    must be spliced as it stands, not expanded, or the font reaches `P`
    inlined and is emitted unoptimized with no error. Whichever of D-a and
    C1 lands second adds the pin: a unit composed through the
    application reaches `P` as a leaf, one telemetry record per unit.
  - **C1's gate:** the font's largest saturation runs at least one round.
    Today zero rounds is silent.
  - **Who marks a unit after D-d** — `by_ref` on the opaque `Kernel`, or
    the language (every kernel-typed argument, or an attribute on an
    entry) — is JP's.
  - composition-is-linking's L4 (inlining as an e-graph rule) is not built
    and, for units, not wanted until C1 measures calls.

**O2. The composition surface.** **Answered (JP, 2026-10-01): yes.** D-a
is built before C1 (`f0599991`), so the host composes `kernel!` entries
and the builder never becomes the font's surface.
- **What was built.**
  - `impl Fn(f32, f32) -> f32` is a parameter type of an entry or a helper,
    and every other spelling of a function is refused. A body applies a
    kernel, `k(x, y)`, or passes it by name to a helper. Anything else is a
    spanned error naming this plan: arithmetic on one, a `let`, an `if`'s
    arm, a return, a record's field, `as f32`.
  - The block is Rust. `sema` refuses what rustc's move checker would of an
    `impl Fn`: a kernel passed on twice, used after it is passed on, or
    passed on inside a fold's body (E0382, E0507). The check follows the
    flow, as rustc's does: an `if`'s arms are two paths, so a kernel one
    arm passes on is the other arm's to apply or pass on, and after the
    `if` it is moved if either arm moved it. The language has no loop, so
    that join is all of rustc's rule (pinned against rustc,
    `rustc_is_the_oracle.rs`).
  - The closure form takes no kernel, spelled any way: bare, in
    parentheses, or through a macro's `$t:ty`. rustc refuses `impl Trait`
    in a closure's parameters (E0562).
  - The host function takes a `&Kernel`. It only reads the argument, and a
    borrow passes one instance twice (`sum2(&a, &a)`). Owning it would
    reuse the argument's arena only by building on top of it, which would
    declare its uniforms before the entry's own.
  - Lowering is one walk over a `Site`. `Expansion` builds an arena at
    expansion; its argument type is uninhabited, so it cannot apply one.
    `Staged` emits each step as the statement that takes it, and the host
    function runs them when it is called.
  - The IR gained the steps that run then: `ExprArena::admit` and `apply`
    beside `Kernel::at`, `open_fold`/`close_fold` beside `close_over` (also
    the macro's own fold, so the sequence has one definition), `compact`,
    and `free_index`, a scoped walk that replaces `free_var_at_or_above`
    for `by_ref` too. `apply` asserts the arena declares the argument's
    uniforms, so an argument applied where it was not admitted cannot
    declare them in read order.
  - Application is `Kernel::at`'s term, `k[X := u, Y := v]`, through the
    arena's own `substitute_vars_with`, and pinned as one key with `at` for
    an argument that names no other kernel. A `DX` inside an argument
    survives the application and follows the warp: `k = DX(X·X)` applied
    at `(2X, Y)` is 24 at x = 3, as `.at` gives, where resolving it first
    would give 12 (pinned, with `DX` outside the application pinned
    separately).
  - At `(X, Y)` the argument is spliced as it stands, so a `Ref` in it
    stays a name. A glyph held by `by_ref`, which is how O1 marks a unit,
    is still a unit when summed (pinned). Under a warp a `Ref` is
    expanded, as `at` expands one. So an argument that names another
    kernel, applied at the sample, is `at`'s program under another key: a
    cache entry each. Inside a fold, the fold around it cannot see the
    referent's own fold and may take the same binder; once the name is
    expanded the referent's fold shadows it, which is binding, not capture
    (pinned: 123 at (3, 5), by name and as itself, as Rust's sum).
  - **D4, as built (pending JP).** D4's row says an argument's binders
    are renamed away from those live at its hole. What is built keeps them
    apart with no rename: the folds around a hole are still open when the
    argument is spliced in, and close after it, past its binders, which is
    `close_over`'s one rule. The meaning is D4's: nothing is captured
    (132 at (3, 5), rustc's value; captured would be 156). The program is
    the builder's `Kernel::over` around `at` (one key, pinned). Whether the
    row should say so instead is JP's.
  - An argument must be closed and read no table, or the host function
    panics.
- **Measured (F, release, AVX-512, one thread, real Noto).**
  - A 189-piece glyph (Noto id 2436) composes in 94–102 ms: `one_piece`
    ×189 4.4 ms, the `sum2` tree 67–75 ms, `glyph` 20–26 ms. 19,495
    nodes, 1,894 uniforms.
  - Noto's ASCII, 94 glyphs and 1,625 pieces with each glyph its own
    program (O1), composes in 427–499 ms. The largest glyph is '@', at 56
    pieces and 18–21 ms. 170,007 nodes, 16,626 uniforms.
  - Again after review, on a shared host, with `apply`'s check that the
    arena admitted the argument and without it (the glyph 14 runs each
    way, the ASCII 10):
    the 189-piece glyph 83–124 ms (median 102) and 84–124 ms (median
    105); the ASCII 429–544 ms (median 472) and 433–574 ms (median 485).
    The check costs nothing that run-to-run noise does not hide. `inside`'s
    half-pixel reach adds four nodes a glyph: 19,499 and 170,383.
  - Composition copies at each level of the tree, so it is O(n log n).
    `uniform_slot_for` searches linearly, which is quadratic in uniforms:
    negligible per glyph, but on the zoom path once a whole font is one
    program (C1, about 16k uniforms).
- **Left open.**
  - A mask at the host boundary. A `bool` entry's kernel passed where
    `-> f32` is declared reads all-ones as NaN and draws plausible pixels.
    Refusing `-> bool` entries would not close it: the builder makes masks
    too, until D-d, and closure-form `bool` kernels are pinned against
    `any_over`/`all_over` (`fold_is_kernel_over.rs`). Nor would `admit`
    refusing a root in `OpKind::is_bitwise_domain`: `If` is in it, and a
    piece's term is an `If` of numbers, while an `If` of masks is a mask.
    The fix is a type, a mask kernel the host cannot pass as a `&Kernel`,
    which is the general case `Bits`'s doc tracks.
  - The id tree's threshold. `k` is computed by halving a font loaded at
    run time, and a const generic is fixed when rustc compiles, so
    `split::<k>` cannot be called for a runtime font. Whether `k` becomes a
    uniform written once per font, or something else, is C1's, with O3.
  - One `id` instance shared by every node of the tree is a convention of
    the host walk. It is not a type (O3).
- **Evidence (before the answer).**
  - The language composes across blocks only through kernel-typed
    arguments (D6). Those were D-a, then not built.
  - The builder composes today, and §1.1 says it is not a surface.
  - The walk is host work because the font is runtime data (§1.1, §1.7). A
    bundled font's walk could run at build time instead (Phase E).
- **Recommendation.**
  - Build D-a before C1, so the host composes `kernel!` entries and the
    builder never becomes the font's surface.
  - If C2's measurement (O4) must come first, a builder prototype outside
    the tree is enough to measure, and is then deleted.
  - Keep the walk at runtime. Phase E declares a bundled font's instances
    later (Q4).

**O3. Binding a composed program.**
- **Evidence.**
  - D-a fixes where an instance's slots land: the entry's own first, then
    each argument's, in parameter order (§1.4). So the walk can return
    offsets as prefix sums. A glyph's box is at `0..4` and piece `k` is at
    `4 + 10k` (F, `kernel_copy.rs`).
  - An entry's `Args` streams its values by position (`set_declared`), and
    the only check is the count (§1.4, "Which program is not yet a type").
  - A composed program's slots are the flattening of every instance's.
    Uniform-slot-identity §3 calls the link step that flattening.
  - `Uniform` handles, which bind by identity, leave the public surface
    (D16).
  - No document says how the host finds each piece's ten slots in the
    font's block, or how a cell writes its six without rewriting the font's
    16k (A4).
- **Recommendation.**
  - The walk that composes the font returns, with the program, where each
    instance's slots landed.
  - The font's values are written once per font and zoom, and a cell's per
    call.
  - Tie the block to the composed program by type (§1.4's follow-up)
    before C2. One font's values written into another font's block bind
    silently and draw plausible wrong pixels.

**O4. The frame budget.**
- **Evidence.**
  - One-pipeline §6 measured a frame with no atlas: a host loop over cells,
    calling per-`N` `U_band` programs. At 200×60 and 32 px it took
    84.4 / 22.9 ms (1 / 4 threads) on AVX-512 and 50.3 / 20.0 ms on AVX2,
    against 16.7 ms. 80×24 at 16 px fit (6.9 / 2.8 and 4.1 / 1.6 ms).
  - It was measured at `8b7b75a`, before the closed form (`33c6509b`), and
    it calls itself "a bound on the direction, not on this plan".
  - One-pipeline's Q3 recommended keeping the atlas until D1. JP's atlas
    sentence and Q1 ruling replace it with the font program (D10).
- **Recommendation.**
  - Measure the font program before deleting the atlas (C2's gate).
  - If 200×60 at 32 px misses 16.7 ms, bring JP the numbers and the two
    levers, caching (JP's "later") or D1 placement, before anything is
    deleted.

**Q2. Syntax vetoes.** The syntax choices are yours:
- the items block with `pub fn` entries;
- `(0..N).map(|i| …).sum()`;
- `if` as the only choice;
- `const` parameters as structural;
- `DX(e)`;
- `k: impl Fn(f32, f32) -> f32`, a `&Kernel` on the host side, and `sum2`
  as a glyph's ink. A kernel is moved when it is passed, as rustc moves it.
  The alternative is `&impl Fn(f32, f32) -> f32`, which rustc copies, so
  one kernel could be passed to two helpers.

**Q3. `text()` and `run`** (no production caller). **Recommendation:**
delete them. A string of glyphs is a sequence of per-cell calls, the same
path as the terminal.

**Q4. AOT's first user.** **Recommendation:** a bundled font at declared
pixel sizes: its glyphs and tile extents are known at build time.

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
- **A fold's index is exact only to 2²⁴ (F, B2).** It is an `f32` lane, and
  `RangeFold` accepted any `u32` end, so a builder fold past 2²⁴ summed the
  wrong terms without a word, while `kernel!` refused such a bound at
  lowering. B5 moved the refusal into `RangeFold` (`RangeFold::admits`),
  which both front ends build through (docs/BACKLOG.md, C8); it is
  `Fold::admits` since `RangeFold` folded back into `Fold` with the
  integral's deletion.
- **The integral closed only under budget (F).** A glyph written as the
  integral it is was correct only as far as saturation reached: the rules
  that derived each piece's closed form shared the flat class cap, and
  quadrature legalized, silently, whatever they left open. A `kernel!`
  glyph at 189 pieces had every integral quadratured, coverage off by up
  to 0.92, with no test failing. JP: *"just do b. delete all the
  integral stuff. other languages don't try this. probably for good
  reason."* The glyph writes its closed form (§1.7), and the integral is
  deleted (§1.5).
