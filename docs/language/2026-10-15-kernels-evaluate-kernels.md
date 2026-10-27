# Kernels evaluate kernels

**Status:** accepted (JP, 2026-10-14 and 2026-10-15). Every ruling below is
binding and overrides any earlier plan, design, skill or comment where they
disagree. The earlier text is history, not a second opinion; where it is a
plan, it is marked superseded at the top and left otherwise untouched.

## 1. `If` has one shape

JP, 2026-10-15:

> I don't want to handle divergence in our cpu version. Select/If always
> always branches. We might expose the other to the user if you really want,
> but this is just too much code for our purposes.

> If is: if all(m) { run true } if !any(m) { run false } run select(m, true,
> false). That's always been what if is supposed to be. I feel like we don't
> need any other version ... I think it always looks like that.

The guards are read as `any(m)` and `!all(m)`, so that on a batch whose mask
is mixed both arms have run before the select (with `all`/`!any` a mixed
batch would run neither). So an `If(m, a, b)` compiles to exactly:

```
if any(m)  { a's own block }
if !all(m) { b's own block }
select(m, a, b)
```

- An arm's **own block** is the nodes only that arm reads. An arm that owns
  nothing (a constant, a value the condition or the other arm also reads)
  has no block and therefore no guard: `(x² + y²).lt(1).select(green, black)`
  is `mul, fma, compare, select`, no branch. `m.select(reflective_shader,
  background)` guards the whole shader, so a batch outside the sphere never
  evaluates it.
- There is **no second shape**: no uniform-only path that skips the select,
  no blend-only path, no mispredict pricing, no decision about *whether* a
  block with work deserves its guard. Any such machinery is deleted on sight.
- The guards read `m` at runtime, per batch. Nothing in the compiler
  estimates how often `m` is true.
- The **domain split** (a mask that implies an index range emits two
  specialized programs, the region's and the complement's) is retired with the
  other shapes. JP, 2026-10-15: "Domain split retire ... let's retire it if we
  don't need it right now." `mask_support.rs`'s range analysis goes with it.

### Cost

JP, 2026-10-15: "You don't know how often m is true. Or the branching factor.
Average them and move on. ... Or do max(false, true)."

`cost(If(m, a, b)) = cost(m) + (cost(a) + cost(b)) / 2 + cost(select)`.
Average, not max: with no information it is the unbiased guess, and max would
push extraction away from `If` toward computing both arms. The cost is for
extraction choosing among equivalent forms; it never changes which batches
run which arm. Demand weighting, arm probability and projected regions are
deleted; `docs/archive/plans/2026-10-12-placement-reads-demand.md` is superseded.

## 2. Kernels evaluate kernels; there are no memory reads

JP, 2026-10-15:

> No reading adjacent pixels. There's no memory reads. There's kernel evals.
> ... A bound buffer is a kernel, but you never get to see the memory mid
> write.

> Actually, we don't have bound buffers. That's a bad name. We have discrete
> kernels. It's a type error. And they can be backed by memory because
> referentially transparent.

The language has kernels, and a kernel evaluates kernels. A **discrete
kernel** is a kernel on a lattice, `Lattice → ℝ`. It is a value: referentially
transparent, so an implementation may tabulate it in memory, and the law
`index(collapse(f)) = f` is what makes the table *be* the function rather
than back it. There is no write in the language: `collapse` yields a whole
value, so no intermediate state is observable. Memoization (baking, the JIT
cache, a glyph cache) is semantically invisible.

### Writing is application

JP, 2026-10-15:

> The scheduler has the ability to turn kernel programs into blocks of
> memory. And those blocks of memory, those lattices, are still kernels ...
> materializing a kernel gives you a discrete manifold. It's closed in that
> way. ... I want all blocks of memory to act as discrete manifolds or
> kernels. ... the ability to write is kind of outside the language. It's in
> the application, the unification of a program and the schedule is what
> gives you the ability to write. And then after you've written, you get
> something back that has the same interface as the program itself ...
> something that can be sampled at discrete points.

```
apply : Program × Schedule → DiscreteManifold
DiscreteManifold = Kernel<L>         (a kernel whose domain is the lattice L)
index(apply(f, s)) = f  on the lattice
```

Neither the language nor a backend has a write. Applying a program to a
schedule produces a block of memory, and producing it *is* the write;
`collapse` is that application. What comes back has the program's interface,
so materialization is closed: a baked glyph, the cell table and the frame are
the same kind of thing as the programs that made them. Every block of memory
a kernel is handed is a discrete manifold; there is no untyped buffer, and the
ABI's `out` pointer is the tabulation in progress. `Write`/`Seq` nodes are the
CPU scheduler's mechanics for performing an application, inside its own
chain: not a language effect, not a concept.

**Tabulation is trivial.** JP, 2026-10-15: "bake is just tabulation, and
tabulation is like trivial ... once you have a discrete manifold, it's
referentially transparent with the tabulation." So nothing chooses it: not
the user as a scheduling decision, not extraction as a rewrite. A `bake` call
is sugar in the graphics library for executing a schedule, not a language
construct. And "let's not have derivatives of discrete kernels for now":
`Dwrt` of a discrete kernel is refused by name.

### Everything is a kernel: the types

JP, 2026-10-15:

> I never wanna make a distinction between a memory read and a discrete
> function. ... The way I intend to do memory reads is to pass discrete
> kernels that happen to be tabulated. The language doesn't have arrays
> ever. ... I use kernel and function interchangeably.

> In the original pixelflow everything was a manifold. Even a regular float.
> It's arrows all the way down in a real lambda calculus language. So 5 is a
> kernel with an infinite domain and a range of, well, just 5.

A kernel is a function, and the language is the typed lambda calculus, so
the types are base sorts and arrows and nothing else:

```
S ::= Real | Int | Mask              base sorts: what one lane holds
D ::= S | D × D | Lattice(w, h)      domains: R² is Real × Real, Z² is Int × Int,
                                     a lattice is a bounded Z²
T ::= S | D → T                      a kernel is a value of arrow type
```

- A continuous kernel is `R² → Real`. A **discrete kernel** is `L → Real`
  for a lattice `L`. A comparison is a kernel `R² → Mask`. The floor of a
  coordinate is a kernel `R² → Int`, which is exactly what a discrete kernel
  can be applied to. They differ in their domain and in nothing else; there
  is no subtype relation, and the earlier "`DiscreteManifold ⊂ Kernel`" meant
  this and said it badly.
- **A constant is a kernel.** `5` is `λ_. 5 : D → Real` for every `D`: the
  global element `1 → Real` composed with the unique `D → 1`. There is no
  separate "scalar" sort of value in the language; a scalar is a kernel that
  does not read its argument.
- **Variance is factorization.** A kernel is *uniform* when it factors
  through `D → 1` (a constant), *row-invariant* when it factors through the
  projection onto one axis, and lane-varying otherwise. The CPU backend's
  choice of broadcast, contiguous load or gather for evaluating a discrete
  kernel is reading which projection its index factors through. The analysis
  the backend already runs is this factorization, not a separate concept.
- **There is no array.** The grammar has no `Array` type former, only
  `L → S`. A table in memory is a representation of a value of that type
  (`Vector α n ≅ Fin n → α`, which is `index(collapse(f)) = f`), and a
  "memory read" is application. The tabulation may or may not have happened;
  the type is the same.
- **The lift is precomposition.** `nearest(D) = D ∘ floor` with
  `floor : R² → Z²`; bilinear is four precompositions with weights. `at` is
  `contramap`, so no new construct is needed. What happens outside the
  lattice is decided by the `R² → Z²` function (clamp to the edge, or a mask
  to zero), not by `D`, which is total on its lattice and says nothing about
  outside.
- **A per-frame table is a parameter of arrow type**, `(L → Real) → R² → Real`,
  as `Uniform` is a parameter of base type. Not a memory concept.
- **`Dwrt` is defined on `Real → _` only.** Differentiating a discrete kernel
  is a type error; differentiating `D ∘ floor` is differentiating a step
  function, zero almost everywhere, and is refused by name for now.
- **The extent is a value, not a type.** Lean would write `Fin w → Fin h → ℝ`
  and carry `w` in the type; Rust can only do that with const generics, which
  would forbid runtime-sized frames. So the *sort* of the domain is a type
  (`R²`, `Z²`, `L`) and the extent of `L` travels with the kernel as a value;
  applying a discrete kernel over the wrong extent is a bake-time refusal by
  name. Every ML-shaped mistake (a table read at a real coordinate, a
  derivative through a table, a wrong-rank composition) is then a compile
  error, and only "wrong size" reaches runtime.

What this asks of the IR, in order (proposed 2026-10-15, not yet ruled):

1. **Base sorts on arena nodes.** Today one `f32` lane is a real, an integer
   and a mask by convention, and `OpKind::is_bitwise_domain()`, the folder's
   non-finite exemption and the `OpKind::mask(bool)` convention recover at
   runtime what a sort would make unbuildable. With `Real`/`Int`/`Mask` on
   every node, `Lt` returns `Mask`, `If` and `select` demand one,
   `BitAnd` is `Mask × Mask` or `Int × Int`, `floor`/`TruncToInt` are the only
   bridges `Real → Int`, and rewrite rules are sort-preserving. The discrete
   kernel's index is typed for free and the plan's "structural integrality
   guard" is never written.
2. **Arrow sorts on binders.** The arena is first-order today; arrows live in
   the Rust `Kernel<D>` wrapper. `Var`, `Uniform` and a discrete-kernel
   parameter are binders of types `Real`, `Real` and `L → Real`; a `Ref` is
   application of an arrow. This is the open work of Linear JP-22 ("one kind of name",
   "composition is linking") stated in these words.
3. **Predicate domains, later.** A domain that is a set with a predicate
   (`{x : ℝ // x ≠ 0}`; `|x| < 2²⁰` for `sin`) lives naturally in the e-graph's
   class analysis, seeded by the domain declared on the inputs and
   propagated through ops, with rewrites conditional on it: `sqrt(x·x) → x`
   on `x ≥ 0`, the `asin` guard dropped on `[−1, 1]`, a mask the domain
   already implies making its `If` unconditional. Kept decidable by keeping
   the fact lattice small (sign, interval, a finite set of exclusions).

Consequences:

- **"Bound buffer", `BoundManifold`, `bind(&[(id, buf)])` are not concepts.**
  A discrete kernel is a `Kernel` and composes like one. The ABI still hands
  the compiled code the tables it reads, as it hands it the output; that is
  the backend's business, not the language's.
- **Discrete and continuous kernels have different domains.** JP: "It's a
  type error." Evaluating `L → Real` at an `R²` coordinate does not
  typecheck. Lifting is explicit precomposition with an `R² → Z²` function
  that names the sampling (`nearest()`, `bilinear()`, ...), so the opinion
  lives in the type. A discrete kernel composed with an integer translation
  (`Z² → Z²`) stays discrete.
- **`Gather`, `Broadcast` and a plain load are not language ops.** They are
  three realizations of one thing, "evaluate discrete kernel `D` at
  `(u, v)`", and the CPU scheduler picks one from the variance it already
  computes: uniform `(u, v)` → broadcast; the lattice position plus a
  translation → contiguous vector load; lane-varying → gather. The IR has no
  gather; the backend does.
- **No arm can fault.** Under §1 an arm runs on inactive lanes of a mixed
  batch with garbage inputs. Arithmetic on garbage is NaN and the select
  discards it; a discrete kernel's domain is its lattice, so its evaluation
  outside the lattice is defined by the type, not by whatever memory lies
  past a table.

## 3. Premises for the schedule (rulings so far, design pending)

JP, 2026-10-14, deciding what is part of the schedule by imagining GPU and
TPU backends beside CPU SIMD:

- "we're a pull renderer. Each pixel is going to be computed independently.
  Our gpu backend is one pixel per thread." The lattice is the GPU's grid;
  lane packing, row loops and stripes exist only where we vectorize across
  pixels ourselves (CPU, TPU's vector unit). The lattice's traversal is the
  schedule's, and wrapping it into the kernel as folds in pixelflow-core
  (`collapse` is a fold) bakes the CPU schedule into a target-agnostic crate.
- "Yeah it's a schedule decision, but I'd put a backend dependent register
  allocator in the CPU scheduler." On a GPU the vendor compiler allocates;
  we have no allocator there. So each backend is one module exposing one
  function, `(kernel, lattice) → code`, with its own `Pass` chain inside;
  the CPU module's chain is Nest → Own → Layout → Select → Allocate →
  Assemble and owns register allocation, rematerialization and carries.
- The kernel's own folds (`sum_over`) are per pixel on every target:
  algorithm, not schedule.
- Materialization (which sub-kernels to tabulate) is schedule, per backend.

The full table (algorithm / schedule / backend, per concept, with where each
lives today) and the module layout are the next note.

## 4. What this supersedes

- CLAUDE.md "`If` contains an if" (uniform takes an arm, mixed blends):
  rewritten to §1.
- `docs/archive/plans/2026-10-12-placement-reads-demand.md`: superseded by §1's cost.
- `docs/compiler/2026-09-07-demand-is-a-dag-property.md`: its denotation of
  demand stands as history; its cost consequences do not.
- Ontology entries for `If`, arm, demand, placement, root, park, lane,
  gather, bound buffer: rewritten to §1 and §2.
- Every doc, comment or skill that says "bound buffer", "bind", "mixed-lane
  path" or treats `Gather` as a language op.
