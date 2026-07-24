# Totality Is the Root Axiom: Domains, Binders, and Why the Cost Model Exists

**Date:** 2026-07-24
**Status:** Design of record (axiom layer)
**Sits above:** `REDUCTIONS_AND_FOLDS.md`, `ML_AND_LINEAR_ALGEBRA.md`,
`ML_AUTODIFF_PIPELINE.md`, `KERNELS_AND_LATTICES.md` — this note states the
constraint those all descend from.
**Feeds:** the kernel-unification plan (`docs/plans/2026-07-20-kernel-unification.md`).

## The one axiom

**The kernel language is total. It is not Turing-complete. Every program's cost
is a closed-form function of static extents, computable without running it.**

This is not a constraint we tolerate to get performance. It is the property that
makes the whole system exist:

- A **cost model** is a function from program to a number, computed *without
  running the program*. That function can only be total if the language has no
  unbounded control flow — a data-dependent `while` has no static cost. So
  "estimatable by the cost model" and "not Turing-complete" are the same
  statement.
- A **shader** is valid iff per-pixel work is bounded. Same property.
- `bake` **always terminates**. Same property.

Totality is the axiom. Everything below is a consequence of it, and the design
mistakes to avoid are all the same mistake: admitting a construct whose cost is
not a static function of extents.

## Think denotationally, not in vectors

The programs we care about — an ambient-occlusion shader, spherical-harmonic
projection, a transformer's forward pass — read as *data structures* (normals,
coefficient vectors, matrices) only if you take the operational view. Denote
them instead — ask what function they *mean* — and the vectors dissolve:

- **AO** means an integral: `AO(x) = ∫_Ω V(x,ω)·cosθ dω`. The normal isn't a
  vector; a normal field is `∇(sdf)`, already the tuple of `Dwrt` projections.
  What's new is the integral, not the vector.
- **Spherical harmonics.** A coefficient is `λ(l,m). ⟨f, Y_lm⟩` — an integral
  `∫_{S²} f·Y_lm`. The "coefficient vector" is *another kernel*, indexed by the
  discrete domain `(l,m)` instead of by screen space. You sample it; you don't
  store it. Real SH are Cartesian polynomials — no complex numbers in the
  meaning.
- **Transformers.** `Q` means `λ(i,d). q(i,d)` — a kernel over two index
  domains. Matmul is `Σ_d q(i,d)·k(j,d)`, an integral over the shared index.
  Softmax's normalizer is `Σ_j`. There is no matrix in the meaning; there are
  functions of indices and integrals over them.

The common structure under all three is **not** vectors or matrices. It is one
binder: a big-operator that folds a body over a bounded domain and eliminates
that dimension. Every "vector" is either a gradient (already `Dwrt`) or a kernel
over a discrete domain.

## What was actually missing: discrete domains + a reduction binder

Two constructs, and *only* two:

1. **Index domains are coordinate domains.** We already believe an image is a
   function of coordinates you never materialize. Extend it: a tensor is a
   function of *indices* you never materialize. The only new thing is admitting
   **bounded discrete domains** (`(l,m)`, feature-index `d`, sample-index `j`)
   alongside continuous `X/Y/Z/W`. A matrix is a kernel over a product of two
   discrete domains; a coefficient "vector" is a kernel over one. Storage does
   not change — still pull-sampled. (We had discrete fields before and removed
   them; this brings them back as the typed reification boundary — see below.)

2. **A reduction binder `⊕_{i∈D}`, parametrized by a monoid.** This is the one
   genuinely new symbolic construct: a quantifier that introduces a bound
   variable ranging over domain `D` and denotes the fold of a body under monoid
   `⊕`. Sum-monoid → integrals, contraction/matmul, SH projection, the AO
   hemisphere. Max-monoid → softmax's stabilizer. Contraction, attention,
   projection, occlusion are all this one binder at different monoids over
   different domains. It is `Let` for a *ranged* variable that gets summed out
   instead of substituted.

### The "shape check" is variable capture, not a type system for kernels

Einstein-summation semantics: a repeated bound index means contract. "Do the
matmul dimensions line up" is literally "is this the same bound variable ranging
over the same domain." Dimension-checking falls out of scoping. The continuous
kernel algebra stays **untyped** (no type variables, no inference, no traits) —
this is the eDSL / lambda-calculus layer, and it never gets a type system.

## Kernels are untyped; fields are typed

A **kernel** is the continuous, composable, algebraic thing — untyped, because
it has no shape until pinned to a domain.

A **field** is a kernel pinned to a discrete domain and reified. Typing
crystallizes exactly there, and nowhere earlier:

- A field's type is `(domain descriptor, element kind)` — a 16×16 grid of f32,
  a length-`L` band, an `N×d` feature field.
- Checking is structural equality of domain descriptors. No inference, no
  polymorphism.
- Extents are concrete literals known at `bake`, so this is **not** dependent
  typing — the scary part of typing shapes only bites when extents are symbolic,
  and here they never are (we monomorphize on them regardless).

So "kernels are never typed" and "type the discrete fields" are not in tension:
a field is not a kernel. **Types live at the discrete reification boundary; the
continuous algebra stays untyped.** Typing fields is easy *because* they are
ground and monomorphic.

## One discipline, four faces

Totality is the single rule — *every domain and every iteration count is a
static natural number* — showing up in four places:

| Face | The static nat |
|---|---|
| discrete domains | extent known at bake |
| reduction binder `⊕_{i∈D}` | `|D|` static |
| bounded iteration (the "fixpoint") | trip count static |
| typed fields | ground extents |

These are not four features to design. They are one constraint checked in four
places.

## Bounded iteration, never a general fixpoint

A total language cannot have iterate-to-convergence — it has no static bound.
What it has is **bounded iteration = primitive recursion over a static count**
(`iterate[N]`). Every real program is already this: ray-march is `N` steps,
Legendre recurrence is band `L`, transformer depth is fixed layers, Newton
refinement is fixed iters. None are true fixpoints; all are static unrolls. The
boundedness is not a restriction on the operator — it is the reason the operator
is allowed to exist.

## The lattice dissolves into the typed field — *not* into the fixpoint operator

The `Lattice` was two things: **(a)** a bounded discrete domain (the sample
grid) and **(b)** memoized storage (glyph cache, ping-pong buffers, "compute
once"). Bounded iteration is *dynamics* — it is neither (a) nor (b), so it does
**not** replace the lattice. What replaces the lattice is the **typed discrete
field**: a domain you can reify a kernel onto, with materialization as an
attribute (lazy vs. stored; ping-pong = two stored fields you swap).

The tell that this is the right cut: the lattice was *already* leaning on
totality — the only reason glyph bake always finishes is that a lattice is
finite and per-point work is bounded. Making it a typed field makes the property
it silently relied on into the property it is defined by. `collapse` / `bake`
become "reify a kernel on a discrete domain"; the tabulation machinery survives
as the *implementation* of reify, but stops being a distinct concept.

Do not conflate the two axes. The fixpoint operator replaces hand-written
iteration loops (ray-marchers, recurrences). The typed field replaces the
lattice. They are orthogonal.

## The payoff: the type system and the cost model are one artifact

Under totality, a field's type *is* its domain extents, and the cost model is a
function *of* those extents:

- `cost(⊕_{i∈D} body) = |D| · cost(body)`
- `cost(field) = |domain| · cost(sample)`

The numbers the type carries are exactly the numbers the cost model multiplies.
So the cost model is the program **re-denoted into the cost semiring** — the
same structural walk over the arena, `+` for sequencing, `×` for iterating over
a domain of size `|D|` — and totality is precisely what guarantees that walk
yields a closed form instead of diverging. **Type-check and cost-estimate are
one pass.** Once the discrete fields are typed, most of the cost model is built,
because every extent it needs has already been collected.

## Reverse-mode AD is the adjoint of the binder

Forward `Dwrt` is right for AA (one output, screen-space seed). Training wants
reverse mode. The transpose of a contraction is a contraction with free and
bound indices swapped — **reduction's adjoint is broadcast; broadcast's adjoint
is reduction.** So reverse-mode backprop through `⊕_{i∈D}` is a *symbolic
transformation on the binder*, not a separate autodiff engine beside `Dwrt`.
Both fall out of the same calculus once the integral binder exists, and both
respect the same static-bound discipline (the adjoint of a bounded fold is a
bounded fold). This is what makes "shader language *and* ML library" one
language. Detail lives in `ML_AUTODIFF_PIPELINE.md`; the axiom is that the
adjoint stays total.

## Complex / quaternion / polar are library, not primitives

None touch the binder. Polar is a warp of a continuous domain (already
`warp`/`.at()` + sin/cos). Complex-mul is a pointwise body; a quaternion is four
pointwise components. They are bodies and domain-changes — derived, two-line
library kernels. Baking them into the op set would be the wrong instinct. (If
the optimizer should fuse complex-mul into FMAs, that is an e-graph rewrite over
the body, not a new primitive type.)

## Two honest caveats

- **Total ≠ cheap.** Totality buys *finite and computable*, not *small*. Nested
  reductions multiply extents; a big unroll is a big number. The cost model's
  job is to *surface* that — a program can be perfectly total and estimated as
  "over budget." Totality gives a number; the budget decides if the number is
  acceptable.
- **Dynamic behavior survives; only dynamic *bounds* die.** Data-dependent loop
  counts (ray-march until you hit the surface) become **static max + early-out**:
  iterate up to `N`, mask off finished lanes. Behavior is data-dependent; cost
  is the static worst case `N`, which is what the model charges. This is the
  same trick GPUs already play, so almost no expressiveness is lost.

## Consequences for the design (the things not to redesign later)

1. Do not add a general `fix` / `while` / unbounded recursion. Ever. It deletes
   the cost model.
2. Do not type the continuous kernel algebra. Types live only at the field
   (reification) boundary.
3. Do not add vectors/complex/quaternions as language primitives. They are
   library over pointwise bodies and gradients.
4. Do not treat the lattice as replaceable by the iteration operator. The
   lattice becomes the typed discrete field; iteration is a separate, orthogonal
   construct.
5. Every eliminator that could iterate (`⊕_{i∈D}`, `iterate[N]`, field reify)
   carries its bound as a static nat, so the cost walk is closed-form.
6. Build the cost model as a re-denotation of the arena into the cost semiring,
   reusing the extents the field types already carry.
