# One name, bound later

## Metadata
- **Author**: JP (direction), Claude (draft)
- **Status**: `Draft`
- **Created**: 2026-09-10
- **Verified against**: `6f3eb619e314304149db65d71bafbe7c096cfd15`
- **Supersedes**: the framing of L6 in
  [2026-09-09-composition-is-linking.md](2026-09-09-composition-is-linking.md)
  §4 ("`Gather` leaves the algebra"). That section's conclusion survives; its
  reason was too narrow.

**Decisions it records (JP, 2026-09-09 / 2026-09-10):**

> *"We should be getting rid of gather. I don't know why winding kernels are
> gathered through... I had a glyph of fold over our table. You should not
> know that you are doing a gather."*

> *"x, y, z, and w are uniforms. Does that make sense? like, uniforms are just
> how we spell extra parameters to the kernel function. You know, x, y, z, and
> w are privileged, but they're just uniforms."*

---

## 1. The thesis

**A kernel is a function, and everything it does not contain is a parameter.
The language has five spellings for that and should have one.**

| spelling | who binds it | how often |
|---|---|---|
| `Var(0..2)` — X, Y | the collapse | per sample |
| `Var(4..8)` — reduction binders | the enclosing fold | per term |
| `Uniform(id)` | the block, else the declared default | per call |
| `Buffer(id)` + `OpKind::Gather` | `Manifold::bind` | per bind |
| `Ref(key)` | the linker (`passes::expand_refs`) | at link |

These differ in **binding time**, not in kind. Nothing else about them is
different: each names something the graph does not hold, each is resolved by
somebody else later, and each is opaque to the passes until it is.

## 2. The evidence that this is already half-true

**Z and W were reclassified, not deleted.** `Kernel::from_parts` refuses
`Var(2)`/`Var(3)` with the reason spelled out: *"a lattice has 2 axes and a
per-call scalar is a Uniform, not an axis of extent 1."* That is exactly the
claim above, already applied to two of the four coordinates. Nobody then asked
why X and Y are a different *kind* of thing rather than two parameters the
collapse happens to bind per sample.

**`Variance` is already the parameter-dependence set.** One bit per `Var(0..8)`
(`pixelflow-ir/src/variance.rs`), `CONST` for a uniform. The analysis already
treats coordinates, binders and uniforms as one space; only the node types
disagree.

**CLAUDE.md already flags the residue**, and calls it a bug class rather than
a wart:

> `Var(u8)` means a coordinate axis or a reduce binder depending on magic
> ranges — it used to mean a manifold-param slot as well, and that third
> meaning went out with the macro parameter that needed it.

**`ExprData` puts two of them side by side.** After the #1249 merge the enum
reads `… Buffer(BufferId), Uniform(UniformId), Op(OpKind), Reduce(Fold),
Ref(KernelKey)`. `Buffer` and `Ref` are both leaves naming an external thing;
one of them is redundant, and which one is the subject of §4.

## 3. Where the leak shows, concretely

The glyph author never wrote a gather. `pixelflow-graphics/src/fonts/loop_blinn.rs`
reads its coefficient table like this:

```rust
fn row_at<'a>(table: &'a Kernel, i: &'a Kernel) -> impl Fn(usize) -> Kernel + 'a {
    move |k| table.at(&Kernel::constant(k as f32), i)
}
```

`Kernel::at` — contramap the coordinates of a named kernel. Function
application. The memory op appears one layer down, in
`DiscreteManifold::kernel_for` (`pixelflow-core/src/lattice/mod.rs`), which
builds `push_gather(buf, x, y)`.

The distribution says the rest: **pixelflow-graphics holds 1 reference to
`Gather`; the compiler internals hold 152** (ir 55, codegen 59, search 31,
core 6, compiler 1 — counted at the sha above). Nobody writing kernels uses
it. It exists so that passes can have cases for it:

- `plan_collapse_hoist` in `pixelflow-codegen/src/emit/mod.rs` refuses to
  hoist anything whose sub-DAG contains a `Gather`, with a stated reason that
  has expired — *"keeping memory reads exactly where the per-batch kernel had
  them costs nothing today (winding kernels are gather-free)"*. Since S1 made
  the glyph a fold over a table, winding kernels are gather-*driven*, so the
  blanket refusal disables the prologue hoist for exactly the kernels that
  most need it. **This is ask B's real cause** (see
  [2026-09-09-a-glyph-is-a-circle.md](2026-09-09-a-glyph-is-a-circle.md) §B),
  and patching the refusal would entrench the leak rather than fix it.
- `passes::lower_dwrt` carries a table-read special case: a gather whose index
  does not move with the differentiation variable has derivative 0.
- `MAX_BOUND_BUFFERS = 4` (`pixelflow-core/src/lattice/manifold.rs`) is a limit
  on how many symbols one program may name.

Each is a pass knowing about memory. Under §1 none of them needs to: the
question becomes "are this application's index arguments invariant in the
scope", which `Variance` already answers, and `d/dX` of `table.at(const,
binder)` is 0 by the ordinary chain rule through `at` because neither index
mentions X.

## 4. What the unified thing is

A name, plus what binds it and when. Sketch, not a signature:

```text
Name { id, bound_at: Sample | Term(binder) | Call | Bind | Link }
```

The two properties that have to survive:

- **A pass may ask when a name is bound and must not care what it names.**
  That is what makes `contains_gather` unstatable rather than merely
  discouraged.
- **The sampling semantics live on the referent, not on an op.** A
  `DiscreteManifold` already denotes nearest-neighbour-with-clamp
  (`index(collapse(f)) = f`, CLAUDE.md's representable-functor law). Reading
  one is applying it. `Gather`'s documented "floor, clamp to the declared
  extents, index row-major" is that denotation restated in the op table, which
  is where it drifts from.

**`Gather` leaving the algebra is a consequence of this, not the goal.** The
goal is that there is one kind of name.

## 5. What this does *not* change

- Codegen still emits a gather instruction, a per-sample coordinate register,
  and a uniform load. Binding time is a property of the language; the
  instruction selected for it is codegen's business, and that is the whole
  point.
- The uniform default stays out of the compiled code's identity.
  `pixelflow-ir/src/key.rs`'s `canonical` puts only the *slot* in the key —
  *"the default is the block's business, not the code's"* — while `Buffer`
  puts extents in, because the address arithmetic was folded against them.
  Those are two different binding times behaving correctly, and a unified name
  must keep the difference.
- `Select` semantics, the floating-point contract, and the fold's denotation
  are untouched.

## 6. Order

This is a denotation, not a schedule. It has no stage list yet, deliberately —
what it needs first is agreement that the five are one, because that is what
decides whether L6 is "delete `Gather`" (a rename) or "there is one kind of
name" (a language change with `Gather`'s disappearance falling out).

Two things are worth knowing before anyone commits to it:

1. **Does a surviving `Ref` need the allocator boundary?** A tabulated read is
   a leaf that emits a load — no second function, no coordinate ABI. If that
   holds, the cheap half of L5 lands here and de-risks the expensive half. If
   it does not, this is blocked behind L5 proper.
2. **What does the binding-time property cost to compute?** `Variance` gets it
   for the three `Var` cases for free. `Buffer` and `Ref` are leaves, so their
   binding time is a constant of the node. The likely answer is "nothing",
   which is worth confirming before designing around it.
