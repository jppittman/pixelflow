# Denotation

**Status:** the denotation of record (JP, 2026-10-15: "denotation first").
**Verified against:** `2a99d933594230a6d9cf94c21d78e5dd134b4785`.

This page says what each thing in pixelflow *means*, as a mathematical
object, and the laws that meaning obeys. It is short on purpose. Three other
kinds of document surround it and are not it:

- The **ontology** (`.claude/skills/ontology/`) says, per term, what the code
  treats it as, what it is not, what follows, and where it lives. It cites
  this page for the meaning and records where the code is behind it.
- The **plans** (`docs/plans/`) say how a meaning was reached and what was
  built toward it. Where a plan and this page disagree, the plan is history,
  and `docs/language/2026-10-15-kernels-evaluate-kernels.md` is the ruling that
  settled most of the disagreements.
- **CLAUDE.md** says how to work here. Its "Floating point at the edges" is
  the one piece of denotation kept there, because every session needs it.

The rule for all of them: the meaning is chosen first, in the type system
where the type system can hold it, and the implementation is obliged to it
(CLAUDE.md, "Denote before you build"). A meaning that lives only in a
comment is an invariant something will eventually break.

---

## 1. The language

**A kernel is a function.** The language is the simply typed lambda calculus
over three base sorts. A kernel is a value of arrow type, and nothing else is
a value.

```
S ::= Real | Int | Mask              base sorts: what one point of the range holds
D ::= S | D × D | Lattice(w, h)      domains: R² = Real × Real, Z² = Int × Int,
                                     a lattice L is a bounded Z²
T ::= S | D → T                      a kernel is a value of arrow type
```

- A continuous kernel is `R² → Real`. A **discrete kernel** is `L → Real`.
  A comparison is `R² → Mask`. `floor` is `Real → Int`. They differ in their
  domain and in nothing else; there is no subtype relation among them.
- **A constant is a kernel.** `5` is `λ_. 5 : D → Real` for every `D`: the
  global element `1 → Real` composed with the unique `D → 1`. There is no
  scalar sort of value; a scalar is a kernel that does not read its argument.
  JP: "5 is a kernel with an infinite domain and a range of just 5."
- **There is no array.** The grammar has no `Array` type former. A table in
  memory is a representation of a value of type `L → S`
  (`Vector α n ≅ Fin n → α`), and reading it is application. Whether the
  tabulation has happened is not visible in the type.
- **Kernel and function are the same word.** JP: "I use kernel and function
  interchangeably." A "memory read" and "evaluating a discrete function" are
  one thing, so the first is not a concept.
- **The extent is a value.** The sort of a domain (`R²`, `Z²`, `L`) is a
  type; the extent of `L` travels with the kernel as a value and is part of
  the compile key. Applying a discrete kernel over the wrong extent is
  refused by name at bake. Rust can carry `w, h` in the type only with const
  generics, which would forbid runtime-sized frames.

**Totality is the root axiom.** The language is strongly normalizing: no
`fix`, no `while`, no recursion, no `mut`, and no trip count or extent that is
a function of a coordinate or of a lattice binder. A fold's bounds are values
placed by their variance ([a bound is a
value](language/2026-10-20-a-bound-is-a-value.md)), each fixed before the
lattice is traversed. Every program's cost is a closed-form function of values
known before the traversal, computable without running it; nothing evaluates it
per call, and the static model prices a fold whose range is not a known
constant at an estimate (`Fold::ESTIMATED_TRIPS`). "Estimatable by the cost
model", "a valid shader" and "`bake` terminates" are one property. Total is not
cheap: totality gives a number, and a value-bounded program is priced at the
estimate, never reported over budget.

**The program is a DAG with one binder.** Terms are hash-consed in an arena
(`ExprArena`, the sole IR): two uses of one term are one node, and the graph
is acyclic by how things are named, as a git history is, with no occurs
check. The only construct that binds is the fold (§3). Coordinates are bound
by the lattice (§5), not by the term.

**Composition never evaluates.** Composition substitutes and links; it does
not optimize, inline or compute. "A kernel is a function, and everything it
does not contain is a parameter." It becomes numbers exactly once, when it
is compiled at a lattice's shape and collapsed.

Decided in: `language/2026-10-15-kernels-evaluate-kernels.md` §2,
`language/2026-07-24-totality-and-the-cost-model.md`,
`language/2026-09-25-the-language-is-kernel.md`.

## 2. Application, names and parameters

**Application is precomposition.** `k.at(g) = k ∘ g`, implemented by
substituting into the `Var` leaves. A coordinate frame lives here and
nowhere else: pixel centre is `f ∘ (+½)`, a placed glyph `f ∘ (−pos)`, DPI
`f ∘ (·s)`. The lattice has no origin (§5).

**A derivative survives application.** `Dwrt` is a term, so `(Dwrt f) ∘ g`
differentiates the warped function: the chain rule holds under a warp
because nothing resolved the derivative before the warp reached it
(`DX(X·X)` at `(2X, Y)` is 24 at x = 3, not 12). `Dwrt` is defined on
`Real → _` only. A name has no derivative: a `Ref` must be linked before it
is differentiated.

**A reference is a kernel.** `Ref(key)` means "evaluate that kernel here",
named by content (`canonical(arena, root)`): two kernels with one key are
one kernel, and two uses of one name are one node. Composition by name is
linking, and a reference is an optimization unit, saturated and extracted
by itself and linked back after extraction:

```
L_s(t) = link( Ô_s(t), r ↦ L_s(body r) )
⟦L_s(k)⟧ = ⟦expand_refs(k)⟧                              (Law U)
```

where `Ô_s = extract_s ∘ saturate ∘ insert ∘ canonical`. The linker only
inlines; a call is not built.

**Everything a kernel does not contain is a parameter, and parameters
differ in binding time, not in kind.** A coordinate is bound per sample by
the lattice; a fold's index per term by its binder; a `Uniform` per call; a
reference at link. A `Uniform` is a parameter of base type; a kernel-typed
parameter is a parameter of arrow type, `(L → Real) → R² → Real`, so a
per-frame table is not a memory concept. "A scalar that is the same at every
sample is an argument, not an axis."

**A uniform is a point in a product.** A kernel with parameters is
`K : P × L → Real`, `P = ℝⁿ`; composition forms the product of parameter
spaces, a uniform block is a point in the flattened product and a slot is a
projection. A uniform is invariant on `L` (its variance is `CONST`) and
unknown on `P` (it is never a `Const`, never folded). Its identity is the
instance, not the name and not the index. A uniform may bound a fold, as a
per-call count ([a bound is a
value](language/2026-10-20-a-bound-is-a-value.md) §1.2); it is never an
extent, and the lattice's extent stays in the program and the key (§5).

Decided in: `archive/plans/2026-09-06-lattice-is-the-index.md`,
`language/2026-09-09-composition-is-linking.md`, `language/2026-09-10-one-name-bound-later.md`,
`archive/plans/2026-09-06-uniform-slot-identity.md`,
`pixelflow-compiler/tests/derivative_under_warp.rs` (the guard).

## 3. The fold

**The fold is the one binder.**

```
⟦Reduce{⊕, i ∈ [lo, hi), body}⟧ = ⊕_{i ∈ [lo, hi)} ⟦body⟧[i]
```

`⊕` is a monoid (`+`, `×`, `min`, `max`, `∧`, `∨`); each bound is a literal,
which the node holds, or a term, a child after the body, fixed before the
lattice is traversed and reading no coordinate and no lattice binder; the
equation holds where every bound is a count, an integer in `0..=2²⁴`, and a
fold over `+`, `×`, `min` or `max` with a bound that is none (a fraction, a
negative, NaN, an infinity) is NaN, while `∧`, `∨` take literal bounds only;
the monoid, the binder and the literal bounds are the node's identity, so
hash-consing is on exactly the fold's meaning. `⊕` over a set has no
accumulator and no order: that absence is what lets the e-graph reassociate
it, and iteration is introduced by lowering, never by the term.

```
⊕_{[lo,hi)} f = ⊕_{[lo,hi−1)} f ⊕ f(hi−1)                 peel
⊕_{[lo,lo)} f = identity(⊕)                                empty
⊕_{[lo,hi)} f = ⊕_{[lo,m)} f ⊕ ⊕_{[m,hi)} f                split
⊕_i (c ⊗ f) = c ⊗ ⊕_i f        if i ∉ var(c) and ⊗ distributes over ⊕
d(⊕_i f) = ⊕_i d(f)            for ⊕ = Σ only (linearity)
```

- **A surviving fold is a loop.** Peel, split and halve are e-graph rules,
  so whether a fold runs as a loop or as copies is extraction's choice,
  decided by cost; no pass unrolls, and nothing unrolls at construction.
  The schedule is the first layer where "previous iteration" means anything.
- **The binder shrinks variance.** It is the only thing in the language that
  does; a bound index is not free in the result.
- **A fold costs `len · cost(body) + (len − 1) · cost(⊕)`**, and its size for
  budgeting is the same product. This is totality cashed out: the number the
  type carries is the number the cost model multiplies.
- **An integral is a fold over a continuous domain**, `∫_{[lo,hi)} body du`,
  the same binder under the length measure instead of the counting measure.
  It is **deferred, not refused**: the IR, the e-graph and `kernel!` carry no
  integral today (deleted 2026-09-29; JP: "I liked the integral, it was just
  hard and I didn't need it yet"), and where a quantity is an integral the
  closed form is written down and summed: glyph coverage is
  `min(|Σ_p σ_p·A_p|, 1)` over a set of monotone arcs, each piece's area
  `A_p` in closed form, every term zero outside its piece's box. When the
  binder returns it returns with the fold's laws: every bound fixed before the
  lattice is traversed, no indefinite integral, no bound that is a function of
  a coordinate, no accumulation from one pixel into the next, and `at` never
  touching a bound index.
- **A matrix is a discrete kernel, and a linear operator is a kernel of
  kernels.** `M : Lattice(n, m) → Real`. An operator on kernels with kernel
  `k` is `λf. λj. Σ_i k(j, i)·f(i)`: a fold of a product of two
  applications. `Σ_d q(i, d)·k(j, d)` is a contraction over a shared bound
  index, and dimension agreement is variable capture, not a shape system.
  Whether the schedule turns a contraction into a blocked GEMM is a
  factorization of its domain (§7); the language has no `matmul` and never
  will. JP, 2026-10-15: "Matmul is an implementation detail and we're a pull
  renderer." An inner product on a continuous domain is an integral, which the
  IR does not carry yet; so the Hilbert space on offer today is the one on
  discrete kernels (sums) and on bases whose inner product has a closed form
  (the spherical-harmonic feature maps in `pixelflow-ml`), and the
  continuous one arrives with the integral.

Decided in: `archive/plans/2026-09-09-a-fold-is-a-node.md`,
`codegen/2026-09-10-a-surviving-reduce-is-a-loop.md`,
`archive/plans/2026-09-23-an-integral-is-a-fold.md`, `archive/plans/2026-09-23-a-glyph-is-a-formula.md`,
`language/2026-07-24-totality-and-the-cost-model.md`.

## 4. Mask and the conditional

**A mask is a bit pattern, not a number.** All-ones for true, all-zeros for
false; the result of every comparison; `Mask` in the grammar. `1.0` is not
true: `mask & 1.0` blends `7.0` against `9.0` into `4.5`. All-ones read as a
number is NaN, which is why the folder's non-finite guard must exempt the
bitwise domain; the guard and the convention are what the `Mask` sort
replaces (§9).

**`If` has one shape.**

```
If(m, a, b)  =  if any(m)  { a's own block }
                if !all(m) { b's own block }
                select(m, a, b)
```

The guards are read at runtime per batch; a mixed batch runs both arms, and
the select is a bitwise blend. An arm's own block is the nodes only that arm
reads; an arm that owns nothing has no block and no guard. Both arms stay
live until the select, so `If` is dispatch, not a fold. There is no
uniform-only path, no blend-only path, no mispredict pricing, no decision
about whether a block deserves its guard, and nothing estimates how often
`m` is true.

```
cost(If(m, a, b)) = cost(m) + (cost(a) + cost(b)) / 2 + cost(select)
```

The cost chooses among equivalent forms; it never changes which batches run
which arm. **No arm can fault**: on inactive lanes an arm runs on garbage,
arithmetic on garbage is NaN and the select discards it; a discrete kernel's
value outside its lattice is defined by its type (§6), not by memory. A
value an arm defines is read by that `If`'s select, so a definition
dominates every read that *observes* it, not every syntactic read, and no
phi is inserted for it.

Decided in: `language/2026-10-15-kernels-evaluate-kernels.md` §1,
`archive/plans/2026-10-08-selection-is-a-phase.md` §0.12–§0.13, CLAUDE.md
"Floating point at the edges".

## 5. The lattice and evaluation

Three objects and one verb:

```
Kernel ──compile(extent)──▶ Manifold ──collapse──▶ DiscreteManifold
```

**A lattice is an extent and nothing else.** Its points are the indices
`[0, w) × [0, h)`. It is the representable functor's index: not a coordinate
frame (a frame is a contramap, §2), not a runtime argument, not a home for
uniforms, not a union of ranges. An axis that never varies is a `Uniform`,
not an axis of extent 1. Its extent is part of the program and of the
compile key, so a resize is a recompile. That stays so under [a bound is a
value](language/2026-10-20-a-bound-is-a-value.md), which leaves the lattice's own
folds constant (its §4: deferred, not argued). A sub-lattice is an index range,
the one thing that lives on the domain side.

**Collapse is tabulation, and the table is the function.**

```
collapse : (L → a) → F a            tabulate
index    : F a → (L → a)            the DiscreteManifold read as a kernel
index(collapse(f)) = f              on L
```

The law holds without a side condition only because the lattice has no
origin: with one, `index ∘ collapse = f ∘ shift`, and the lattice stops
being the representing object. The law is pinned by a test that fails if a
coordinate is put back on the index.

**Writing is application.** Neither the language nor a backend has a write.

```
apply : Program × Schedule → DiscreteManifold
DiscreteManifold = Kernel<L>
index(apply(f, s)) = f              on L
```

Applying a program to a schedule produces a block of memory, and producing
it *is* the write; what comes back has the program's interface, so
materialization is closed: a baked glyph, the cell table and the frame are
the same kind of thing as the programs that made them. Tabulation is
trivial and nothing chooses it, neither the user nor extraction; `bake` is
sugar for executing a schedule. `collapse` yields a whole value, so no
intermediate state is observable, and memoization (baking, the JIT cache, a
glyph cache) is semantically invisible. `Write`/`Seq` are the CPU
scheduler's mechanics inside its own pass chain, not a language effect.

**Evaluation is whole-domain.** One kernel is compiled for the whole lattice
and the loop nest is inside the emitted code; the ABI is `fn(ctx, out,
pitch)` once per stripe, and nothing crosses it as a vector. There is no
per-point entry, no per-batch entry and no interpreter. A single point is a
contramap over a 1×1 extent.

Decided in: `archive/plans/2026-09-06-kernel-with-a-lattice.md`,
`archive/plans/2026-09-06-lattice-is-the-index.md`, `archive/plans/2026-09-16-collapse-is-a-fold.md`
(denotation; its traversal is the CPU backend's, §7),
`language/2026-10-15-kernels-evaluate-kernels.md` §2.

## 6. Discrete kernels

**A discrete kernel is `L → Real`**, a value, referentially transparent, so
an implementation may hold it in memory; the law above is what makes the
table *be* the function. "Bound buffer", `bind`, `BoundManifold` and
"memory read" are not concepts; a consumer cannot observe whether
evaluating a kernel runs arithmetic or reads a table.

- **Discrete and continuous have different domains.** Evaluating `L → Real`
  at an `R²` coordinate does not typecheck. JP: "It's a type error."
- **The lift is precomposition.** `nearest(D) = D ∘ floor` with
  `floor : R² → Z²`; bilinear is four precompositions with weights. The
  sampling is named at the lift, not carried by the referent. What happens
  outside the lattice is decided by the `R² → Z²` function (clamp to the
  edge, or a mask to zero), not by `D`, which is total on its lattice and
  says nothing about outside. A discrete kernel composed with an integer
  translation (`Z² → Z²`) stays discrete.
- **`Gather`, `Broadcast` and a plain load are not language ops.** They are
  three realizations of "evaluate `D` at `(u, v)`", chosen by the backend
  from which projection of the domain the index factors through (§7):
  uniform index → broadcast; the lattice position plus a translation →
  contiguous load; lane-varying → gather. `Gather(collapse(f), p) → f(p)` is
  not its own rule; it is `index(collapse(f)) = f`.
- **No `Dwrt` of a discrete kernel.** Differentiating `D ∘ floor` is
  differentiating a step function; refused by name for now.

Decided in: `language/2026-10-15-kernels-evaluate-kernels.md` §2.

## 7. Variance is factorization; the schedule is a factorization of the domain

**Variance is which projection a kernel factors through.** For a value `v`
over the iteration space `Ω` (the lattice axes and every fold's index),
`var(v) ⊆ B` is the set of binders it depends on, and the law is

```
b ∉ var(v)  ⟹  ⟦v⟧ is constant along b
```

the Boolean shadow of the forward derivative. A kernel `f : D → T` is
*uniform* when it factors through `D → 1` (a constant), *row-invariant*
when it factors through the projection onto one axis, and *varying*
otherwise; `Const ⊑ Uniform ⊑ Varying` is the order on the projections. An
e-class's variance is the intersection over its nodes, `var(C) = ⋂_{n∈C}
var(n)`, a greatest fixpoint. The analysis the compiler runs is this
factorization, not a separate concept.

**Every optimization is factoring.** CSE, loop-invariant code motion,
broadcast, tiling and register allocation are one operation, "identify a
subexpression that depends on fewer variables than its context and compute
it at the appropriate scope", at different granularities. Extraction is
factoring; the fold's binder is the one construct that shrinks the set a
term depends on. Hoisting out of the lattice's rows and hoisting out of a
fold are the same question asked at each level.

**The schedule is a factorization of the domain.** A lattice is a product,
and a backend chooses how to factor it:

```
L ≅ Rows × Batches × Lanes          CPU SIMD
L ≅ Threads                         GPU, one pixel per thread
L ≅ Tiles × Vector                  TPU
```

`apply(f, s)` evaluates each sub-arrow of `f` at the coarsest projection it
factors through: a term uniform in `Lanes` is computed once per batch, one
uniform in `Batches` once per row, one uniform in `L` once per frame. The
hoisting that was the whole 2.3× over LLVM is this rule. Which factors
exist, and whether rows and lanes exist at all, is the backend's: the
lattice's traversal is the schedule's, not the kernel's, and wrapping it
into the kernel as folds in a target-agnostic crate bakes the CPU schedule
into the language. The kernel's own folds are per point on every target:
algorithm, not schedule. Materialization, which sub-kernels to tabulate, is
schedule, per backend.

Each backend is one module exposing one function, `(kernel, lattice) →
code`, with its own `Pass` chain; the CPU module's is Nest → Own → Layout →
Select → Allocate → Assemble, and it alone owns register allocation,
rematerialization and carries. On a GPU the vendor compiler allocates. The
full table of concept → algorithm / schedule / backend is the schedule note,
`codegen/2026-10-15-a-schedule-is-a-factorization.md`.

Decided in: `archive/designs/lattice-scheduling-types.md` (right about factoring,
wrong about the mechanism), `archive/plans/2026-09-23-an-integral-is-a-fold.md` §1
(the variance law), `language/2026-10-15-kernels-evaluate-kernels.md` §2–§3.

## 8. The optimizer and the as-if contract

**The optimizer is a function.** `P : Kernel × LatticeShape × Isa → code`,
`P(k, s, t) = emit_t ∘ L_s(k)` with `L_s` from §2, run once, at compile:
`Kernel → Kernel` by `at`, `select`, folds and arithmetic is open and never
optimized; `Kernel → program` by `P` is closed and is the only place
anything is optimized. "There is no tier": the JIT is `P` run at runtime
and a build-time compile is `P` run at build time with the same key.
Optimization is a deterministic function of the arena, the shape and the
optimizer configuration (rule set and order, cost table, budget, guide and
reranker identity), and the compile cache key is exactly that triple. The
budget is denominated in rule applications, deterministic in the input;
wall clock is a fail-loud ceiling that panics the build, never a budget
dimension and never a reason to emit a worse kernel.

**The five laws** (`archive/plans/2026-09-02-optimizer-api.md`, as audited):

1. **Soundness.** Every rule preserves denotation on its documented domain,
   so an e-class *is* a semantic equivalence class and `add` is a
   homomorphism from the term algebra onto the e-graph quotient. Holds; as
   strong as the oracle that checks it.
2. **Monotonicity.** Saturation only adds equalities; the partition at
   t+1 refines t. Budget truncation stops early and can never make the graph
   unsound, so `Budget` is a quality dial, never a correctness dial. Holds.
3. **Round-trip.** The extracted term is well-founded, every node it names
   was in the graph, and it evaluates to what the input evaluated to. Holds.
4. **Guide neutrality.** For any ordering policy and any budget the
   extracted result denotes the same function; a guide changes cost and
   compile time, never meaning. Follows from 1–3; a guide only orders and
   truncates.
5. **Cost and extraction.** A non-additive cost enters as a `Reranker` over
   whole extractions, never as a pass. The first half, "an additive cost
   makes `extract_dag` an argmin", is **violated**: extraction runs a tree
   DP and a shared-DAG DP (#1116) and keeps the cheaper by true DAG cost,
   but both choose greedily bottom-up, so neither is an argmin and no
   sharing is charged for. Said so in `extract.rs`; nothing may assume
   otherwise.

Because rules are equalities of meaning, the optimizer may produce a
different answer than the unoptimized term wherever no value was promised
(`min(1.0, NaN)` on x86, under installed commutativity); it would be a
miscompile the moment anything promised one. Constant folding runs on the
build host, so it declines every row of the platform-divergence table.
`Recip` and `Rsqrt` are estimates and never fold; `MulAdd` rounds once on
every tier and does.

**The as-if contract.** A compiler owes the program's observable behaviour
and nothing about how it produces it. A test is a law about the contract
checked against an independent oracle, and the contract has five items:
values (within the documented per-op tolerance, at points where the `f64`
oracle is well-conditioned, with range hard and NaN outside a documented
domain), refusals by name, no crash, no bit-level determinism (two
compilations agree only as item 1 says, between tiers, between optimized
and raw, between machines), and the ABI and public API. Never asserted:
instruction counts, loads, stores, spills, what was hoisted, which
instruction or register was chosen, or the shape of any intermediate
structure. A same-form check cannot see a shared-definition bug; only an
external bound can. "Precision is on the table; range is not."

Decided in: `archive/plans/2026-09-02-optimizer-api.md`,
`language/2026-09-24-one-pipeline.md` §1, `archive/plans/2026-09-01-production-budget-determinism.md`,
`process/2026-10-13-tests-check-the-as-if-contract.md`, CLAUDE.md.

## 9. What the types ask of the IR (proposed, not yet ruled)

1. **Base sorts on arena nodes.** `Real`/`Int`/`Mask` on every node: `Lt`
   returns `Mask`, `If` demands one, `BitAnd` is `Mask × Mask` or
   `Int × Int`, `floor`/`TruncToInt` are the only bridges `Real → Int`, and
   rules are sort-preserving. A constant already has its sort (`Lit::Real`,
   `Lit::Mask`); `OpKind::is_bitwise_domain()` and the folder's non-finite
   exemption are the runtime recovery of what the sort on nodes makes
   unbuildable, and the mask that leaks through a `bool` entry's `&Kernel` is
   the open bug the sort closes.
2. **Arrow sorts on binders.** `Var`, `Uniform` and a kernel-typed parameter
   are binders of types `Real`, `Real` and `L → Real`; a `Ref` is
   application of an arrow. BACKLOG N1–N3 in these words.
3. **Predicate domains, later.** `{x : ℝ // x ≠ 0}`, `|x| < 2²⁰` for `sin`:
   an e-graph class analysis seeded from declared input domains, with
   rewrites conditional on it.

## 10. The backend

**A backend is a functor** from the term category to a machine's: it
preserves composition, so the code for a compound op is the code for its
parts, glued, and it chooses representation. Its phases are typed:

```
select   : scoped schedule → machine function over values
allocate : machine function over values → machine function over registers and slots
assemble : assembly program → binary          stateless; encode is total and makes no choices
```

- **Values are names; registers are resources.** A value has one definition
  and any number of reads and is `Copy`; a register is a token the allocator
  alone leases, not `Copy`, not `Clone`. Two instructions that differ only
  in physical registers are the same instruction. A register the machine or
  ABI requires is a constraint on a value; any other is the allocator's
  choice, with no exception.
- **Allocation is total.** At every point the map from live value to
  location is injective, every constraint holds, and every inserted
  instruction preserves each value from definition to every read. Running
  out is `Err(BudgetExceeded)`; failing to allocate is a panic naming the
  instruction, "a selection bug, never a fact about a kernel".
- **A lane is what one IR lane value is on this machine**, the backend's
  associated type: a vector register on AVX2 and NEON, a vector or an
  opmask on AVX-512, with no conversion as a concept. The machine cannot
  tell a mask from a number, so "a mask is not a number" is the IR's to
  answer (§9), never the backend's.
- **The ISA tier is a property of the process.** Decided once, at startup,
  by CPUID: AVX2+FMA is the floor and AVX-512 is taken where the host has it
  on x86-64; NEON on aarch64; a host below the floor is refused naming the
  missing feature. The one width in the workspace is the tier's, and nothing
  outside the emitters names a lane. "Take what the hardware gives, and
  never hand-roll something worse": edge IEEE conformance is not on offer,
  and portable code does not depend on a platform-divergent row.

Decided in: `archive/designs/assembler-as-functor.md`,
`archive/plans/2026-10-08-selection-is-a-phase.md`,
`codegen/2026-09-22-the-isa-is-decided-at-startup.md`,
`archive/plans/2026-09-12-emit-should-just-emit.md`.

## 11. Not concepts

Each of these was once a thing here. The date is when it stopped.

| not a concept | what it is instead | since |
|---|---|---|
| array, `Array` type former, uniform array, table backed by uniforms | `L → S`, a discrete kernel | 2026-10-01, 2026-10-15 |
| memory read, bound buffer, `bind`, `BoundManifold`, `Buffer` leaf | application of a discrete kernel | 2026-10-15 |
| write, `Write`/`Seq` as language effects | `apply`; CPU scheduler mechanics | 2026-10-15 |
| `Gather`/`Broadcast`/load as language ops | backend realizations of application | 2026-10-15 |
| a second shape of `If`, `Guard`, a domain split, mispredict or coherence pricing | one shape, averaged cost | 2026-10-05, 2026-10-15 |
| demand, placement that reads demand, park | ownership and the one placement rule | 2026-10-15 |
| a scalar value distinct from a kernel | `λ_. c` | 2026-10-15 |
| `area`, `Fold::Interval`, quadrature (the integral is deferred, not refused) | a fold of a closed form, until the continuous binder returns | 2026-09-29 |
| a bespoke unrolling pass (`ExpandReduce`), unrolling at construction | extraction's choice | 2026-10-03 |
| `iterate[N]`, `Fix`, `while`, recursion, `mut` | nothing: totality | 2026-07 |
| a per-batch or per-point entry, an interpreter, `eval_scalar` | compile at a shape and collapse | 2026-09-06 |
| `Field`, the combinator tier, a scalar fallback, SSE2 | the JIT's tier | 2026-09-22 |
| a lattice origin or coordinate frame, Z and W axes, `Union` of ranges | a contramap on the kernel; a `Uniform`; one index range | 2026-09-06, 2026-09-09 |
| variance as a type parameter (`Varying`/`Uniform` wrappers) | a computed factorization | 2026-09 |
| a learned extraction cost model | the latency prior, with a `Reranker` seam | 2026-09-01 |
| a lattice traversal baked into the IR as folds | the CPU backend's own pass chain | 2026-10-15 |
| `matmul` as a primitive | a fold over a shared bound index | never was |

## 12. Where each decision lives

| section | decided in |
|---|---|
| 1 language, types | kernels-evaluate-kernels §2; totality-and-the-cost-model; the-language-is-kernel |
| 2 application, names | lattice-is-the-index; composition-is-linking; one-name-bound-later; uniform-slot-identity |
| 3 fold | a-fold-is-a-node; a-surviving-reduce-is-a-loop; an-integral-is-a-fold; a-glyph-is-a-formula |
| 4 mask, If | kernels-evaluate-kernels §1; selection-is-a-phase §0.12–§0.13; CLAUDE.md |
| 5 lattice, apply | kernel-with-a-lattice; lattice-is-the-index; collapse-is-a-fold; kernels-evaluate-kernels §2 |
| 6 discrete kernels | kernels-evaluate-kernels §2 |
| 7 factorization, schedule | lattice-scheduling-types; an-integral-is-a-fold §1; kernels-evaluate-kernels §3; the schedule note |
| 8 optimizer, as-if | optimizer-api; one-pipeline §1; production-budget-determinism; tests-check-the-as-if-contract |
| 10 backend | assembler-as-functor; selection-is-a-phase; the-isa-is-decided-at-startup |
