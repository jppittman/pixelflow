> **Live plan; three cited paths moved (annotated 2026-09-09).** The uniform leaf
> landed as U0 (#1183) and this document is still the reference for it. Since it was
> written, `pixelflow-compiler/src/ir_bridge.rs` was split into `lower.rs` and
> `emit.rs` (#1206), and `pixelflow-compiler/src/jit_backend.rs`,
> `codegen/emitter.rs`, `jit_manifold.rs` and `pixelflow-graphics/src/animation.rs`
> were deleted. The stage table's T2/T5/T8 rows name those paths; the stages
> themselves are unaffected.

# Uniforms: a scalar parameter that is invariant without being known

## Metadata
- **Author**: JP (direction), Claude (draft)
- **Status**: Draft
- **Created**: 2026-09-06
- **Reviewers**:
- **Decision it records (JP, 2026-09-06):** *"My intent was actually for
  `Param(i)` to lower to a uniform, not a constant. The only thing that has to
  be constant are loop bounds."* And the framing for what a uniform *is*:
  *"This is supposed to be the JIT's version of the stuff in the combinator
  structs."*

---

## 1. Overview

### 1.1 Problem Statement

A `kernel!` in the combinator tier is a Rust struct. Its scalar parameters are
fields:

```rust
let circle = kernel!(|cx: f32, cy: f32, r: f32| { ... });
let c = circle(0.0, 0.0, 1.0);   // a struct { cx, cy, r }, evaluated by Manifold::eval
```

Those fields are *runtime values*. Monomorphization keeps them in the struct,
LLVM hoists whatever depends only on them out of the pixel loop, and moving the
circle is a field write. Nothing recompiles.

In the JIT tier the same parameters are folded to constants. `Param(i)` is a
placeholder in the macro-emitted template arena, and every builder call runs
`substitute_params`, which replaces it with `Const(v)`
(`pixelflow-compiler/src/jit_backend.rs`, `codegen/emitter.rs`). The emitter
panics if a `Param` reaches it. So in the tier that is meant to *replace* the
combinator tier for evaluation (docs/plans/2026-09-06-kernel-with-a-lattice.md),
moving the circle is a different kernel: a new arena, a new saturation, a new
JIT compile, and a new JIT-cache entry keyed on the value. The
[JIT cache](../../pixelflow-codegen/src/jit_cache.rs) documents exactly this: a
window resize recompiles "by decision".

That decision is fine for a glyph's curve endpoints, which really are the
kernel. It is wrong for a scene transform, a cursor position, a blink phase, or
a light direction, which are the kernel's *arguments*. The JIT tier has no way
to say "this scalar is invariant across the lattice but not known until the
call". That distinction, and not the folding itself, is the missing type.

### 1.2 Goals

- A **uniform** leaf in the IR: a scalar that is invariant across the lattice,
  supplied at call time, and never constant-folded.
- **Identity by instance**, so that composition merges the same uniform read
  from twenty places into one slot, while two instances of the same builder
  stay two slots. This is the same property `BufferIdentity` gives buffers.
- The **choice** between fold and uniform is made at the call site, by type,
  and every existing call site keeps its current meaning (fold).
- Structurally identical kernels that differ only in their uniform instances
  **share machine code**. A thousand circles is one compile.
- Uniform-only subexpressions are computed **once per call**, not per batch:
  `r * r` from a uniform `r` costs nothing in the loop.
- Loop bounds, lattice shapes, and buffer extents **stay static**, enforced by
  the types that already enforce it.

### 1.3 Non-Goals

- Uniforms in the typed combinator tier (`kernel!` struct fields of a new
  type). That tier already has runtime fields; this design gives the JIT tier
  parity with it. Whether the `kernel!` signature should later accept a uniform
  kind is an open question (§7), not part of this work.
- Non-scalar uniforms (vectors, matrices). A matrix is nine uniforms; the
  block layout in §3 makes that free, and a typed wrapper can come later.
- Dynamic loop bounds, dynamic lattice extents, dynamic buffer shapes. A
  uniform can be a *gather index*; it can never be an *extent*.
- Changing how the macro numbers `Param(i)`. The placeholder stays; only what
  it is substituted with changes.

---

## 2. Background

### 2.1 Current State

**Param is a placeholder, not a constant.** Every non-test site that touches
`ExprNode::Param` does one of three things:

| Behavior | Where |
|---|---|
| Panics: "substitute params first" | `pixelflow-codegen/src/emit/mod.rs` (emitter), `pixelflow-ir/src/eval.rs` (oracle, PointCheck), `pixelflow-search/src/nnue/factored.rs` (edge walker) |
| Declines | `pixelflow-search/src/egraph/insert.rs` returns `Declined::Param` |
| Passes through as an opaque leaf | `arena.rs` (splice, canonical key), `passes.rs` (copy; derivative is `0`), `term_arena.rs`, `jit_cache.rs` (key is the *index*, not a value) |

Macro-time saturation never sees a `Param` at all: `ir_bridge.rs` encodes
`Param(i)` as `Var(16 + i)` (`PARAM_VAR_BASE`) before insertion and decodes it
after, so the e-graph optimizes around an opaque variable. Only `variance.rs`
is pessimistic, classifying `Param` as `Variance::ALL`.

In other words, the compiler is already uniform-shaped up to two seams: the
variance class, and the substitution at the builder. What it lacks is
identity.

**Why identity is the problem.** `Param(u8)` is an index local to one
template. `Kernel::add`, `Kernel::at`, `Kernel::sum` compose by
`ExprArena::splice`, which copies a fragment into the host arena. Two
fragments each built from `circle(...)` both contain `Param(0)`, and after
splicing the host has two `Param(0)` leaves that mean different circles. The
same fragment spliced twice (`k.add(&k)`) also yields two `Param(0)` leaves,
and those mean the *same* circle. An index cannot tell these apart.

**Buffers solved this already.** `BufferId` is a slot index into one arena's
table; `BufferIdentity` is a process-unique provenance token minted once and
copied into every declaration that names that memory. `splice` merges buffer
tables by identity, so reading the same atlas from twenty places binds one
pointer, and two atlases of equal extent stay two. The e-graph's
`ENode::Buffer` carries the full decl so hash-consing is by identity too. At
frame time the cell grid attributes each declared slot to a live pointer by
identity (`SlotReads` in `pixelflow-core/src/lattice/cell_grid.rs`) and fills
the `*const *const f32` context table the kernel was compiled against.

**Per-call scalars already exist in the ABI.** The collapse kernel receives
`x0, y0, z, w` in the first four vector registers; the emitter's
`partition_by_scope` treats anything invariant in X and Y as per-call and
hoists it into the once-per-call prologue (`HoistCtx::Prologue`). `z` and `w`
are, in effect, two uniforms with fixed names.

**The glyph pipeline is the counterexample.** A glyph is thousands of
`kernel_value!` curve-segment fragments with their endpoints folded in, fused
into one arena and baked once. Those endpoints are the kernel; folding them is
what lets saturation simplify the fused expression. This design must leave
that path exactly as it is.

### 2.2 Prior Art

- **GPU uniform buffers.** A shader is compiled once; per-draw scalars live in
  a buffer bound to the pipeline; the compiler hoists uniform-only arithmetic
  to a preamble. The block layout in §3 is a UBO with the linker choosing
  offsets.
- **`BufferIdentity`** (this repo): identity is provenance, minted, never
  inferred from shape.
- **The combinator tier** (this repo): a struct instance *is* the identity of
  its parameters. Composition nests structs; DAG sharing shares instances.
  Everything below is that fact, stated for arenas.

---

## 3. Design

### 3.1 Denotation

A `kernel!` with scalar parameters denotes a function

```
K : P × L → Field        P = ℝ^n  (the struct), L = the lattice
```

Composing kernels forms the product of their parameter spaces. A scene of two
circles is `(ℝ³ × ℝ³) × L → Field`. Reading one circle from two places does not
enlarge the product; it is the same factor twice. A fused JIT kernel is the
function on the **flattened product** of every instance's parameters in the
composition. A **uniform block** is a point in that product; a **slot** is a
projection; the **link step** is the flattening, which chooses each factor's
offset.

Three consequences, each of which is a rule below:

1. Identity is the *factor*, i.e. the instance, not the name and not the index.
   Two instances of the same builder are two factors.
2. The block's layout is a function of the composition, not of the instances'
   identities. Two compositions with the same shape have the same layout and
   the same code.
3. A uniform is invariant on `L`, so its variance is `CONST`, and it is
   *unknown* on `P`, so it is never a `Const`. Today `Const` means both; this
   design splits them.

### 3.2 Architecture

```
kernel_value! template ───► Param(i) placeholders          (unchanged)
                                 │
       builder call: circle(Uniform, Uniform, f32)
                                 │ substitute_params(&[Scalar])
                                 ▼
             fragment arena: Uniform(slot) ─ table[slot] = UniformDecl { id }
                                 │                    Const(1.0)
       compose: add / at / sum ──┤ splice merges tables by UniformIdentity
                                 ▼
             fused arena ── uniform table: [decl_a, decl_b, …]   (identity order = insertion)
                                 │
       compile ──────────────────┤ link: canonical traversal ⇒ offset per identity
                                 │       variance(Uniform) = CONST ⇒ per-call prologue
                                 │       emit: broadcast-load [block + 4·offset]
                                 ▼
             Compiled { code, link: [(UniformIdentity, offset)], … }
                                 │
       per call ─────────────────┤ UniformBlock::set(handle, v)   (Result; unknown handle is an error)
                                 ▼
             call_collapse(ctx = [block_ptr, buffer_ptrs…], tile, origin)
```

### 3.3 Interfaces

**The IR leaf and its table.** Mirrors `Buffer` exactly.

```rust
// pixelflow-ir/src/arena.rs

/// Provenance of a uniform: minted once per instance, copied into every
/// declaration that names that instance. Two declarations with the same
/// identity are the same slot after a splice; nothing else can collide.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct UniformIdentity(u32);
impl UniformIdentity { pub fn mint() -> Self; }   // same counter discipline as BufferIdentity

/// Declaration of a uniform: identity plus the value the kernel has for it
/// when nothing has been bound. The default is part of the IR so that
/// `Lattice::bake` and the scalar oracle are total without a block.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct UniformDecl { pub id: UniformIdentity, pub default: f32 }

/// Slot index into one arena's uniform table. Not an identity.
pub struct UniformId(u16);

pub enum ExprNode {
    Var(u8), Const(f32), Param(u8),
    Buffer(BufferId),
    /// Lattice-invariant scalar supplied per call. Never folded.
    Uniform(UniformId),
    …
}

impl ExprArena {
    pub fn declare_uniform(&mut self, decl: UniformDecl) -> UniformId;
    pub fn push_uniform(&mut self, id: UniformId) -> ExprId;
    pub fn uniforms(&self) -> &[UniformDecl];
    /// `Param(i)` → `Const` or `Uniform`, per `params[i]`.
    pub fn substitute_params(&mut self, root: ExprId, params: &[Scalar]) -> ExprId;
}
```

`ExprNode` stays within its 16-byte static assertion: `Uniform(u16)` is the
same shape as `Buffer(BufferId)`.

**The front-end handle.** This is the struct field, as a value.

```rust
// pixelflow-ir/src/kernel.rs

/// A named scalar argument of a kernel. Creating one mints an identity; the
/// handle is the only way to set the value later, so a kernel's arguments
/// are exactly the handles its author kept.
#[derive(Clone, Copy, Debug)]
pub struct Uniform { id: UniformIdentity, default: f32 }

impl Uniform {
    pub fn new(default: f32) -> Self;
    /// The leaf, as a fragment: composes like any `Kernel`.
    pub fn kernel(self) -> Kernel;
}

/// What a builder accepts for a scalar parameter. The type decides fold vs.
/// uniform; `f32` keeps every existing call site folding.
#[derive(Clone, Copy, Debug)]
pub enum Scalar { Const(f32), Uniform(Uniform) }
impl From<f32> for Scalar   { … }
impl From<Uniform> for Scalar { … }
```

Builder closures emitted by `kernel_value!` change signature from
`move |cx: f32, …|` to `move |cx: impl Into<Scalar>, …|`. `circle(0.0, 0.0,
1.0)` still folds; `circle(cx, cy, 1.0)` with `cx: Uniform` makes the first
two arguments slots and folds the radius. Manifold-parameter and `Param`
numbering in `ir_bridge.rs` are untouched.

**The compiled program and the block.** Values are an *argument of the call*,
not state on the program. Stripes on separate threads read one block
immutably; nothing is ambient.

```rust
// pixelflow-core/src/lattice/mod.rs

impl Lattice {
    /// Compile once. Uniform-only arithmetic lands in the per-call prologue.
    pub fn compile(&self, kernel: &Kernel) -> Compiled;
    /// Unchanged: bakes with every uniform at its default. Total.
    pub fn bake(&self, kernel: &Kernel) -> DiscreteManifold;
}

pub struct Compiled { /* code, link table, lattice shape */ }

impl Compiled {
    /// A block with every uniform at its default, laid out per the link table.
    pub fn block(&self) -> UniformBlock;
    pub fn bake(&self, block: &UniformBlock) -> DiscreteManifold;
}

pub struct UniformBlock { values: Vec<f32>, link: Arc<LinkTable> }

impl UniformBlock {
    /// Error, not silence, when the handle is not one of this program's
    /// arguments: that is a composition mistake and the pixels would be
    /// plausible.
    pub fn set(&mut self, u: Uniform, v: f32) -> Result<(), UnknownUniform>;
}
```

`CellGridProgram::frame` grows the same way: it takes a block beside `cells`
and `atlas`.

**Link order.** `compile` walks the reachable subgraph from the root in the
same canonical order `jit_cache.rs` already uses for its key (ascending node
id, ids remapped dense) and assigns offsets by *first occurrence*. The link
table maps identity to offset. The JIT-cache key records uniforms by their
dense offset, never by identity, so two compositions with the same shape hit
the same code and differ only in the block they bind. The same canonicalization
is applied to `Buffer` leaves at the same time, which retires the cache's
"kernels that read bound memory are compiled fresh" exception. That exception
was a gap this design would otherwise inherit, since every uniform kernel is a
bound kernel.

**ABI.** The block is `ctx[0]`; buffer base pointers follow at `ctx[1..]`. One
entry, fixed position, so the `extern "C"` signatures and `MAX_SLOTS` bookkeeping
change by a constant. A `Uniform(slot)` at offset `o` lowers to a scalar
broadcast load from `[ctx[0] + 4·o]`: `vbroadcastss` on AVX2 and AVX-512,
`movss` + `shufps` on SSE2, `ld1r` on NEON. Because its variance is `CONST`
the load is emitted in the per-call prologue; the allocator may treat the value
as rematerializable from memory rather than spilling it, since the reload is
one instruction with no store.

**Variance.** `ExprNode::Uniform(_) => Variance::CONST` in
`compute_arena_variance`. This one line is what turns `r * r`, a rotation's
`sin`/`cos`, and a projection matrix's products into prologue work.

**E-graph.** `ENode::Uniform(UniformIdentity)`, hash-consed by identity like
`Buffer`. `insert.rs` stops declining. No rule matches it as a `Const`, so
`ConstantFold` and `fold_is_platform_specific` never see it; extraction
redeclares the decl (identity and default) into the output arena. Macro-time
saturation keeps its `Var(16+i)` encoding, since at that point the fold/uniform
choice has not been made.

**Oracle.** `eval_scalar` and `PointCheck` take a `&[f32]` block in link
order, defaulting to the decls' defaults. JIT-versus-oracle tests bind the
same block to both.

**Derivatives.** `∂u/∂x = 0`. `passes.rs` already returns `Const(0.0)` for
`Param`; `Uniform` takes the same arm.

### 3.4 What stays static, and what enforces it

| Thing | Stays | Enforced by |
|---|---|---|
| Reduce extent | `u32` | `Kernel::over(monoid, extent: u32, …)`; the arena's extent child is pushed from that `u32` as `Const`. `expand_reduce` (`passes.rs`, `const_val`) panics on anything else. |
| Lattice shape | static | `LatticeShape` is a JIT-cache key and a compile input. |
| Buffer extents | static | `BufferDecl { width, height }` is part of the IR. |
| Gather index | may depend on a uniform | It is a value, not a shape; the existing clamp handles it. |

The extent slot is still a `Const` *by convention* inside
`Nary(Reduce, [Const, Const, Const, body])`, which CLAUDE.md already flags.
Giving the extent its own node is the right fix and is out of scope here;
`expand_reduce`'s existing panic is the guard until then, and T6 pins it
against a `Uniform` specifically.

### 3.5 Data Flow, per frame

1. Consumer keeps the `Uniform` handles it created when building the scene.
2. `let mut block = compiled.block();` once, or reused across frames.
3. `block.set(cx, t.cos())?` for whatever moved.
4. `compiled.bake(&block)` per stripe, on worker threads, all sharing `&block`.

No arena is touched, no saturation runs, no compile happens. Step 4 is the
same collapse call as today with one more pointer.

### 3.6 Error Handling

| Failure | Handling |
|---|---|
| `set` with a handle the program does not contain | `Err(UnknownUniform)`. Never ignored: the wrong scene would render plausibly. |
| `splice` sees two decls with one identity and different defaults | Cannot happen: the default lives in the `Copy` handle that minted the identity. `splice` asserts equality anyway, as it does for buffer extents. |
| Identity counter exhausted | Panic, via the same `fetch_update` discipline as `BufferIdentity`; wrapping would alias two live uniforms. |
| A `Uniform` reaches an extent | `expand_reduce` panics (`const_val`), pinned by T6. Unreachable through `Kernel::over`; the panic is for rewrites. |
| Block bound to a program compiled from a different link table | `UniformBlock` carries an `Arc<LinkTable>`; `bake` asserts pointer equality. |

---

## 4. Implementation Plan

### 4.1 Task Breakdown

| Task | File(s) | Deps | Estimate |
|---|---|---|---|
| T1: `UniformIdentity`, `UniformDecl`, `UniformId`, `ExprNode::Uniform`, table, `declare`/`push`, splice merge by identity, canonical form | `pixelflow-ir/src/arena.rs`, `term.rs`, `term_arena.rs` | None | M |
| T2: `Scalar`, `Uniform` handle, `substitute_params(&[Scalar])`; `kernel_value!` builders take `impl Into<Scalar>` | `pixelflow-ir/src/kernel.rs`, `pixelflow-compiler/src/jit_backend.rs`, `codegen/emitter.rs` | T1 | S |
| T3: variance `CONST`; derivative `0`; oracle takes a block | `pixelflow-ir/src/variance.rs`, `passes.rs`, `eval.rs` | T1 | S |
| T4: `ENode::Uniform`, insert instead of decline, extraction redeclares; template/oracle match arms | `pixelflow-search/src/egraph/{node,insert,template,graph}.rs`, `math/oracle.rs`, `runtime.rs` | T1 | M |
| T5: link step, emitter `ScheduledOp::Uniform` → broadcast load per backend, `ctx[0]` ABI, cache key by dense offset (buffers too) | `pixelflow-codegen/src/emit/{mod,x86_64,avx2,avx512,aarch64}.rs`, `jit_cache.rs`, `jit_manifold.rs` | T1, T3 | L |
| T6: test pinning that `expand_reduce` refuses a `Uniform` in the extent slot | `pixelflow-ir/src/passes.rs` | T1 | S |
| T7: `Lattice::compile`, `Compiled`, `UniformBlock`; `CellGridProgram::frame` takes a block; `MAX_SLOTS + 1` | `pixelflow-core/src/lattice/{mod,cell_grid}.rs` | T5 | M |
| T8: a consumer that moves: one animated scene in `pixelflow-graphics` driven by handles, with the cache asserting one compile | `pixelflow-graphics/src/animation.rs` or `render/scene.rs` | T7 | S |

### 4.2 Parallelization

```
T1 ──┬──▶ T2 ──────────────┐
     ├──▶ T3 ──▶ T5 ──▶ T7 ──▶ T8
     ├──▶ T4 ───────────────┘
     └──▶ T6
```

T2, T3, T4, T6 are independent once T1 lands. T5 is the long pole.

### 4.3 Risk Assessment

- **Lost folding where a caller wanted it.** Mitigated by the type: `f32`
  still folds, and no call site changes meaning. The glyph path is untouched.
- **Register pressure from hoisted uniforms.** Every prologue value lives in a
  hoist slot or a register. A scene with hundreds of uniforms could pressure
  the pool. Mitigation: rematerialize from the block instead of spilling (one
  load, no store); measure with `xtask isa-matrix` before deciding it matters.
- **Cache-key change for buffers.** Canonicalizing `Buffer` in the key is a
  behavior change for kernels that were never cached. A kernel whose buffer
  *extents* differ must still miss; extents stay in the key. Pin with a test.
- **Link order and construction garbage.** Dead nodes from `substitute_params`
  rebuilds must not perturb offsets. The canonical traversal is over the
  reachable subgraph only, which is the existing rule for the cache key.
- **Two tiers, one meaning.** `Lattice::bake` (defaults) and `Compiled::bake`
  (block) must agree bit-for-bit when the block holds the defaults. Pinned by
  a test, not by inspection.

---

## 5. Testing Strategy

### 5.1 Unit Tests

- **Arena (T1):** splice merges equal identities to one slot; distinct
  identities stay distinct; `k.add(&k)` yields one slot; canonical form is
  independent of identity values and of construction garbage.
- **Substitution (T2):** `&[Scalar::Const(1.0), Scalar::Uniform(u)]` yields one
  `Const` leaf and one `Uniform` leaf; `f32` arguments produce a byte-identical
  arena to today's.
- **Variance (T3):** a `Uniform`-only subexpression is classified `CONST` and
  `partition_by_scope` places it in the per-call region. Assert the region, not
  the timing.
- **E-graph (T4):** insertion succeeds; two `Uniform` leaves with one identity
  are one e-class; `ConstantFold` never folds `Uniform + Const`; extraction
  round-trips the decl.
- **Emitter (T5):** each backend's broadcast-load encoding under the existing
  disassembly coverage; the JIT cache hits for two compositions of equal shape
  and distinct instances; misses when buffer extents differ.
- **Extent (T6):** an arena hand-built with a `Uniform` in the extent slot is
  refused by `expand_reduce`.

### 5.2 Integration Tests

- **JIT versus oracle with a bound block**, across several values, without
  recompiling between them: `Compiled` is built once and the test asserts the
  cache's compile count.
- **Defaults agree:** `Lattice::bake(k)` equals `compiled.bake(&compiled.block())`
  bit-for-bit.
- **Cell grid:** `CellGridProgram::frame` with a block whose one uniform moves
  the cursor; frames differ where the cursor is and nowhere else.
- **ISA matrix:** `xtask isa-matrix --smoke` covers the new encodings per
  level, since the output is per-level machine code.

---

## 6. Alternatives Considered

| Alternative | Pros | Cons | Why Not |
|---|---|---|---|
| Status quo: substitute, rely on the JIT cache | Nothing to build | A compile per distinct value; per-frame motion is per-frame compilation | The cache is a memo, not an ABI |
| Identity by **name** (`"cx"`) | Readable | Two circles both have a `cx` | Same reason `BufferIdentity` rejects extents: a coincidence is not a fact |
| Identity by **fragment-local index**, i.e. today's `Param(u8)` | Exists | Collides on splice, both ways (§2.1) | Cannot express "same instance" |
| **Extend the `Var` index space** (`Var(16+i)` all the way to codegen) | The macro already does it for saturation | Magic ranges; `Var` would mean four things | CLAUDE.md names this exact smell |
| **Encode a uniform as a `Gather`** from a 1×n buffer at a constant index | Zero new IR, zero new ABI | A uniform is not memory with extents; the e-graph would see a gather; clamps and gather cost in the prologue; no distinct leaf to refuse folding on | Extends the meaning without extending the type |
| **One block pointer per instance** | No link step | `ctx` grows per instance; a thousand circles is a thousand pointers | Flattening is the denotation (§3.1) |
| **Values inside the handle** (`Arc<AtomicU32>`, read at call) | `set` is global across programs | Ambient mutable state read by stripe threads; a value is no longer an argument | Determinism and thread safety are cheaper as a function argument |
| **Uniforms in the `kernel!` typed tier now** | Full parity | That tier's fields are already runtime; its eval path is leaving as a consumer API | Do the tier that lacks it |

---

## 7. Open Questions

- [ ] Should `z` and `w` become ordinary uniforms? They are per-call scalars
      with fixed names today, passed in `xmm2`/`xmm3`. Folding them into the
      block would simplify the collapse ABI to `(ctx, tile, x0, y0)`. Separate
      change; noted because this design makes it possible.
- [ ] Should `kernel!` (typed tier) accept a `Uniform` parameter kind, so a
      struct field literally *is* the handle? It would make the two tiers
      spell the same thing the same way. Deferred until the typed tier's role
      after docs/plans/2026-09-06-kernel-with-a-lattice.md is settled.
- [ ] Rematerialize-from-block versus spill: policy in the allocator, or a
      cost-model row? Measure first.
- [ ] Give the Reduce extent its own node (retire the `Const`-by-convention).
      Out of scope here; T6 guards until then.
