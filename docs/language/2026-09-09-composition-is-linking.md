# Composition is linking, and the linker only inlines

**Date:** 2026-09-09
**Status:** L1–L3 executed — the reference, the store, shared expansion,
and one composition site converted and measured (§2.1, §7). §5's first cost
has its gate; the inlining rule (L4) waits on a kernel that composes
identical referents, because §7 found the glyph is not one. **Superseded in part by `docs/language/2026-10-15-kernels-evaluate-kernels.md` (2026-10-15):** the language has no buffer read, no `bind` and no `Gather` node, since a discrete kernel is evaluated at coordinates and the backend alone chooses broadcast, load or gather, so the `BufferIdentity`/`Manifold::bind` link step and `MAX_BOUND_BUFFERS` describe the backend's ABI rather than the language.

**Amended 2026-10-01: a reference is an optimization unit**
([the-language-is-kernel](2026-09-25-the-language-is-kernel.md) §4, O1,
answered yes by JP). The runtime tier saturates and extracts a referent by
itself, holds it in the referring term as an opaque leaf carrying its
variance, and links the optimized body back in after extraction
(`passes::link`, `pixelflow_search::runtime`). The title still holds — the
linker only inlines; what it inlines is now each body as optimized — and
L4 is not built: under O1 no rule opens a unit, so inline-or-call is not a
choice the e-graph makes. Whether a unit is emitted inline or called is
C1's measurement there, which is §5.2's question.
**Author:** JP (the framing and both design decisions), Claude (draft)
**Supersedes the framing of:**
[../archive/plans/2026-09-09-the-graph-differentiates.md](../archive/plans/2026-09-09-the-graph-differentiates.md)
§4–§6, which are consequences of this rather than findings of their own.

---

## The observation

`a.add(&b)` splices `b`'s arena into the result. Every combinator does. That
is **static linking**, unconditionally, with no alternative — and most of
what was fought on 2026-09-08/09 is that one fact seen from different sides:

- 4.7 million arena nodes to *construct* a 26-character string, because 26
  glyphs were inlined pairwise
- `Kernel::sum` carrying a special case against exactly that, whose own
  comment says the real fix is a hash-consed shared store
- `min_of` hand-rolled as a balanced tree for the same reason
- `Union` existing at all — a caller hand-placing summands because the
  language could not share them
- and `Gather`: the **one** dynamic-link node the language has, restricted
  to tabulated `f32`, with its own bespoke symbol table (`BufferIdentity`),
  link step (`Manifold::bind`), and unresolved-symbol error (the panic when
  a declared slot has nothing bound)

`Kernel` is already `Arc<KernelData>`, so a shared reference to a compiled
unit exists and cloning one is cheap. What is missing is that **composition
cannot hold a reference**. There is no node meaning *evaluate that kernel
here*. A buffer read is that node, degenerately, which is why it needed a
parallel mechanism to exist at all and why nothing else in the language can
be shared by reference.

The expression-template era had both modes because `Arc<Manifold>` *was* a
manifold: ownership or pointer, your choice, and composition did not care.
The JIT-first move kept the inlining half.

## 1. A reference to a kernel is a kernel

The law from
[the-graph-differentiates](../archive/plans/2026-09-09-the-graph-differentiates.md) §4 — a
consumer cannot observe whether evaluating a kernel runs arithmetic or reads
a table — is this, restated. A reference is a kernel; so is a tabulation; so
is an expression. `circle + baked_glyph` is ordinary `+` because both sides
are the same kind of thing.

## 2. Identity is the compile cache's key, and they are not two things

**Decision (JP, 2026-09-09).** A referent is named the way `jit_cache` names
one: `canonical(arena, root)` (`pixelflow-ir/src/key.rs`) walks the
reachable subgraph in post-order from the root, hash-consing structurally
equal subterms, and encodes each node — a structural content hash that does
not depend on the order the arena was built in. Two kernels with that key are
the same kernel.

So one content-addressed store serves three purposes that are currently
three mechanisms:

| use | today |
|---|---|
| compiled-code memo | `jit_cache`, keyed on `canonical` + shape |
| tabulation memo | `DiscreteManifold` + `BufferIdentity` + `bind` |
| reference identity | does not exist |

And it settles what a *tabulated* kernel's identity is, which is the
question that makes `BufferIdentity` look necessary. It is not a hash of the
buffer's megabytes. A tabulation is `collapse(f)` at a shape, so its
identity is `(key(f), shape)` — **the key of the code that produced it**.
That is why §5 of the previous plan (a kernel retains its code, `pub(crate)`)
is load-bearing rather than a convenience: without the code there is nothing
to name the tabulation *by*, and `BufferIdentity` is the fresh-minted
placeholder that stands in for the name we threw away.

### 2.1 Corrected in execution: structure identity is not value identity

**Decision 2 as literally stated is wrong, and the store's first test found
it.** `canonical(...).key` is *deliberately* blind to which memory a slot
binds — buffers and uniforms are numbered by dense slot, and that blindness
is precisely what lets a thousand circles over a thousand atlases share one
compiled region. So two `DiscreteManifold::kernel()`s over two different
tables have **identical** canonical bytes. A reference store keyed on those
bytes alone handed back the other kernel, and `Manifold::bind` panicked on
a slot nothing was bound to.

The compile cache survives that blindness because it returns the link
*beside* the code (`Linked`) and lets the caller bind. A reference store
cannot: `resolve` must return the very kernel, tabulations and all.

So there is one canonical walk and **two keys from it**. `jit_cache` keys on
the shape bytes alone, because it wants *structure* identity — the same
program for every binding. `KernelKey` digests the shape bytes and both link
tables, because a reference wants *value* identity — this kernel, over this
memory. The substance of the decision survives (one walk, one notion of
"the same"); the letter — "they are not two things" — is exactly one bit
wrong: they are one thing with two projections.

Also settled in execution: `KernelKey` is 64 bits, because `ExprNode` is
held to 16 bytes by a static assertion and a 128-bit leaf would grow every
arena in the process by half. The store keeps the full `Canonical` beside
each entry and compares it on every intern, so a collision is a loud panic
rather than a wrong referent. And the hash is hand-rolled (FNV-1a with a
SplitMix64 finalizer) rather than `DefaultHasher`, whose algorithm is
explicitly unstable across Rust releases — an identity that moves with the
toolchain is not an identity.

## 3. Inlining is an e-graph rule

**Decision (JP, 2026-09-09).** Not a scheduling annotation invented for the
purpose, and not a caller's choice: the e-graph already exists to choose
between equivalent forms by cost, and inline-or-call is exactly that choice.

```
Ref(k)  ⟷  body(k)
```

Static linking is the left-to-right direction taken; dynamic linking is it
declined. The cost model decides, which is the same answer this codebase
gives for `Select` lowering and for guard placement.

**This subsumes the materialization rewrite.** The previous plan proposed
`Gather(collapse(f), p) → f(p)` as its own rule, deriving it from
`index(collapse(f)) = f`. It is not its own rule — it is this one, at a
reference that happens to carry a cached tabulation. One rule, and the law
falls out rather than being installed.

## 4. What this makes of the earlier findings

Not separate defects; consequences.

- **`Gather` leaves the algebra** (§4 there) because a reference is a
  kernel, so nothing needs a second way to name memory.
- **`Dwrt` over a `Gather` has no rule** (§3 there) because you cannot
  differentiate a name. Give the reference a resolvable referent and you
  differentiate the referent. The special-case rule that section first
  proposed — *a gather whose index mentions no coordinate differentiates to
  zero* — was standing in for a link step.
- **`MAX_BOUND_BUFFERS = 4`** is a limit on how many symbols one program may
  name. A content-addressed store has no such number.
- **Providing a range requires providing a select** (§6 there) is unaffected
  and still wanted: it is about the *meaning* of a bounded kernel, not about
  how it is linked.

## 5. Two costs, and neither is measured

This is why this document is a denotation and not a plan.

### 5.1 A rewrite rule fires eagerly

An ordinary rewrite adds `body(k)` to `Ref(k)`'s e-class during saturation.
The graph then holds the inlined form whether or not extraction picks it —
which is the 4.7-million-node blowup arriving through the front door. "The
e-graph decides" and "the e-graph explores every inlining and then decides"
are different propositions and only the first is affordable.

So the rule has to be gated rather than free. The machinery exists — the
saturation Guide (docs/archive/plans/2026-08-31-guide-design-revision.md) chooses
which rule applications to spend budget on, and the `Reranker` seam sits
over extraction — but nothing has been asked to gate a rule by the size of
what it would introduce. **This is the first question to answer**, because
if inlining cannot be gated the whole design collapses back to static
linking with extra ceremony.

Note the measurement already in hand from the reduce, which is the same
shape of question: expanding a glyph's reduce before saturation costs 3.8×
on `A`, 5.9× on `O`, and 8.7× on `8` (87,007 → 758,059 nodes). Inlining is
that, at every composition boundary.

*Status.* The gate exists now: growth per rule application is predicted
exactly before the application is made (`EGraph::predicted_growth`,
asserted equal to the measured delta on 10,819 applications), and the
budget is spent against it. What is not yet built is the rule itself. And
§7 changes what the rule is *for*: those reduce numbers were quadratic
copies, not the unroll.

### 5.2 A surviving reference needs a calling convention

Codegen emits one flat function per kernel. If extraction *keeps* a `Ref`,
that is a call: a second function, an ABI for passing coordinates and
receiving a batch, and a register-allocation boundary the current allocator
does not model. Dynamic linking stops being free precisely here, and the
cost is real enough that it may be what decides how often the cost model
prefers a call.

## 6. What is not in question

- `Kernel` staying `Arc<KernelData>`. The reference mechanism is already
  there; only composition's inability to hold one is at issue.
- Combinators keeping `&self`. An earlier draft of this work proposed making
  them consume so tabulations could move rather than be shared — that
  hard-codes static linking, which is the thing being removed.
- The data travelling with the value, so a consumer never carries a binding
  beside a kernel. That is right under either linkage model and is the one
  piece of the previous plan's §4 that can land on its own.

## 7. L3, executed: where a reference paid, and what that decides

L3 was scoped as "use `by_ref` at the composition sites that blew up —
`text()`'s per-glyph merge, `Kernel::sum`'s fold, `min_of`." None of them
was where the cost was. `text()` merges *outlines* and calls `glyph()` once,
so a string is one reduce over one table and construction was already
linear: 12 ms for a glyph, 199 ms for 26 characters, unchanged by anything
below.

The cost was inside `glyph()`. `coverage()` asks, per piece that another
contour's ink may cover, whether that piece separates ink from no-ink —
`|w − own|` against `|w − own + dir|` — and each question reads the *whole
winding sum*. Composition splices, so a glyph with `m` such pieces held `m`
copies of the `Reduce` node, and `expand_reduce` unrolled every copy:
`m · n · body` nodes, quadratic in the piece count. Measured on
`text()` at 32 px, legalized arena reachable from the root:

| chars | pieces | by value | by reference |
|---|---|---|---|
| 1 | 40 | 654,823 | 30,535 |
| 2 | 73 | 2,712,515 | 58,339 |
| 4 | 132 | 7,640,375 | 105,111 |
| 8 | 252 | — | 194,045 |
| 26 | 613 | — | 442,089 |

By value the per-piece cost climbs 16k → 37k → 58k; by reference it is
~720 at every size. (`pixelflow-graphics/examples/text_kernel_cost.rs` is
the ruler.)

Two changes, and the second is the one that generalizes:

- `glyph()` hands `coverage` the winding **by reference**. One node, however
  many questions are asked of it.
- **Two uses of one name are one node.** `expand_refs` splices a referent
  once per key and points every later `Ref` at that subgraph. Without this
  the linker pastes `m` copies back in and the quadratic returns through
  the front door — a name that expanded per use would buy nothing over a
  splice. Pinned by `every_use_of_a_name_shares_one_referent`.

The same ruler then found the legalize *time* still quadratic with the node
count linear, in two passes that have nothing to do with references:
`differentiate` allocated and scanned an arena-sized table per `Dwrt` (a
glyph holds a few per antialiased edge), and `expand_reduce`'s substitution
allocated an arena-sized memo per unrolled term — and `None` for
`Option<ExprId>` is not the zero pattern, so each was written in full. Both
are keyed or body-sized now. Legalize of the 26-character string: 86 s →
1.15 s, linear (20 ms for one glyph, 40 pieces). The CI timeouts on
`kernel_glyph_optimize` (>600 s → 0.7 s) and `loop_blinn_winding` were the
quadratic copies reaching saturation.

### What this decides for L4

The construction-side case for a reference is fully delivered by L3 plus
shared expansion, and it needed no e-graph. So L4 does not buy *that*.

What reaches saturation is now ~720 nodes per piece, of which the unrolled
winding is ~100 and the per-piece distance terms ~600 — `m` *distinct*
fragments over different constants, which no reference can share. The next
lever on a glyph's saturation cost is therefore S1b of
[glyph-as-a-fold-execution](../graphics/2026-09-09-glyph-as-a-fold-execution.md) — the
distance fold as a second table-reading reduce — not L4. (S1b landed; its
status note there records that it did *not* shrink what saturation sees —
a table read is a subtree where a constant was a leaf — and what it bought
instead: a 5,990-node arena for any outline, and bakes 2–5× faster at
terminal sizes.)

L4's case is the one §3 states: a scene that composes many *identical*
kernels (an atlas of cached glyphs, a repeated primitive), where whether to
inline is a real choice with a real cost on each side. The growth gate it
needs exists (§5.1); the rule waits on a kernel that composes that way.

One wart, recorded rather than fixed: "the linker only inlines" holds for
every compile path, but `Kernel::parts()` hands out the *unlinked*
fragment, and five measurement consumers that took it straight into
`lower_dwrt_owned` or `arena.buffers()` each had to learn to link first
(a name has no derivative and declares no buffer). That is the right
division today — `parts()` is the fragment, and the pipeline's first step is
the link — but it is five copies of one line.

*Status.* Closed: `Kernel::linked_parts()` is `parts()` after `expand_refs`,
in one call, and the five call sites use it; `parts()` itself is unchanged
and still the right thing for a caller (`legalize`, `optimize_runtime_arena`,
`jit_cache::compile`) that links as its own first step.

## 8. `Kernel::at` expands, and this plan never said so (JP, 2026-09-12)

§3 says inlining is an e-graph rule. §5.2 says a surviving reference needs a
calling convention. Both assume a `Ref` *reaches* the IR.

**It usually does not.** `Kernel::at` — the eDSL's composition operator, and
the only way to read a tabulated kernel — calls `linked()`, which runs
`expand_refs_owned` whenever the fragment holds a `Ref` at all. That happens
at **construction**, in the eDSL, before an arena exists for anyone to
rewrite. Its own doc gives the reason:

> a `Ref` is opaque to substitution — it has no `Var` to rewrite here, only a
> name — so leaving one in place would sample the referent at the outer
> coordinates and produce plausible, wrong pixels. Expand first.

The reason is correct; the remedy is one of two available. Expanding makes
the coordinates agree by rewriting the body. **Naming the composite makes
them agree by construction**, and keeps the name.

So there are two unconditional expansions in this system, not one:

| | when | denoted |
|---|---|---|
| `passes::expand_refs` | every compile entry point | yes — L4 |
| **`Kernel::at` → `linked()`** | **at construction, in the eDSL** | **no — until now** |

The second is the worse of the two, because it happens before the IR exists.
No later stage can revisit a choice made by the front end.

### The requirement

> *"`at` needs the same machinery as the rest of the linking stuff. We need
> references that have the option, but not the requirement, to expand."* — JP

### The shape

`Ref(k).at(cx, cy)` denotes the composite `k ∘ (cx, cy)`. Name it rather than
build it:

```rust
Apply { inner: KernelKey, cx: ExprId, cy: ExprId }
```

One node, two children, one name — the shape `Guard { mask, on, off }` already
takes. Its key derives structurally from `key(k)` and the canonical forms of
`cx`/`cy`, so **naming the composite requires no expansion**: content
addressing composes, which is the same property that makes the graph acyclic.

Then one node has three lowerings, decided at extraction rather than at
construction:

- **expand** — substitute into the referent. What `at` does today, now a choice.
- **call** — §5.2's convention, with the coordinates as arguments (L5).
- **tabulate** — a read of a materialized composite (L6).

The same pattern as `Select`/`Guard` and `Reduce`/loop: equivalent forms in
the graph, priced by extraction.

> **2026-10-05:** the `Guard { mask, on, off }` node this section cites as
> precedent was retired (nothing produced it, and its emitter equalled `If`
> only for a batch-uniform mask; commits "retire the guard-arm scope", "retire ExprNode::Guard"). The `Apply` design does not
> depend on it: `KernelKey` stays, as `Ref` still names a kernel by it.

> **2026-10-26 (JP-110): the node is built, with one lowering.** `ExprNode::Apply(KernelKey, cx, cy)`
> exists (`ExprData::Apply` on the `Kernel` DAG, `Shape::Apply` in the term
> language). `Kernel::at` and `ExprArena::warp` no longer call `linked()` /
> `expand_refs`: substitution carries a `Ref` through a warp as an `Apply` at the
> substituted coordinates (a `Ref` is the identity warp, and `Apply(k, X, Y)` is
> pushed as `Ref(k)`, so a meaning has one node). Its key is derived from the
> parts, tag 14 in `pixelflow-ir/src/key.rs`: the referent's key and the two
> warp children's canonical ids, so naming the composite needs no expansion, and
> `jit_cache::compile` still keys the kernel with every name expanded
> (`expand_refs` handles an `Apply`), so the compile cache stays one key per
> program. Variance is the referent's with each coordinate it reads replaced by
> its warp's.
>
> Only **expand** is implemented: `passes::expand_refs` (every compile entry)
> and `passes::expand_applies` (the runtime tier, before the unit walk — a unit
> is closed and read at its own coordinates, which an `Apply` is not). The
> e-graph declines an `Apply` (`Declined::Apply`), so an inlined-versus-referenced
> comparison (JP-19) is made by what survives construction, and extraction does
> not yet price the choice. **call** (L5) and **tabulate** (L6) remain open.

### What this reorders

**L6 is not independent, and its row in the backlog reads as though it is.**
A table is a `Ref`; reading it is `at`; `at` expands. So a tabulated `Ref`
cannot survive being read, and "a tabulated kernel is a `Ref` with a cached
tabulation" is not expressible until `at` stops expanding.

Ordering: **L4 → `Apply` → L5 / L6.**

This also narrows the blast radius of "memory leaks into the algebra". The
eDSL surface is already clean — a table is an opaque `Kernel`, and reading it
is coordinate substitution; there is no load, no index operator, no table
type. What leaks is one constructor: `DiscreteManifold::kernel_for` builds a
fragment holding `ExprNode::Buffer` and `OpKind::Gather`, committing to the
tabulated lowering at construction, where nothing downstream can revisit it.

## 9. L4/L5 measured: the inline-everything curve (JP-19, 2026-10-26)

`pixelflow-pipeline/examples/ref_reuse_cost.rs` composes `K` copies of one
identical unit (an 8-trip fold with a `sqrt`) at distinct offsets, joined by
`min` — §3's case for L4 — and collapses a 128×128 frame. Every copy is
inlined. When this was measured `Kernel::at` expanded a reference at
construction; since JP-110 each copy survives construction as an `Apply`
(§8), but expand is its only lowering, so the compile below is unchanged.

| copies | linked nodes | code bytes | collapse µs |
|---|---|---|---|
| 1 | 27 | 388 | 38 |
| 2 | 55 | 556 | 70 |
| 4 | 111 | 924 | 141 |
| 8 | 223 | 1672 | 270 |
| 16 | 447 | 3156 | 547 |

Everything is linear: ~28 nodes, ~185 bytes and ~34 µs per additional copy
(compile milliseconds are not tabulated; they are dominated by structure-keyed
cache hits and are noise at this size). So:

- **Nothing shares today.** Identical referents at different offsets are
  different programs once `at` has substituted, so the e-graph has nothing to
  find and a surviving `Ref` has something to save.
- **What a call can buy is code size, not time.** A copy costs ~185 bytes and
  ~2 ns per pixel here; a call site (arguments, call, result, the spill of
  whatever the allocator holds across it — §5.2) is the same order as the
  body. For a body this small a call loses on execution and saves only a few
  KB. The case is a body whose bytes dominate (a glyph-scale unit, the
  thousands of bytes), which this ruler does not yet contain.
- **The prerequisite is `Apply` (§8), not the call.** Without it the
  comparison cannot be built: the only composition that keeps a name is one
  at the *same* coordinates, which `expand_refs`' per-key sharing already
  collapses to one node. (JP-110 has since built the node; the call and
  tabulate lowerings it would choose between remain open.)

No cost-model change follows from this. The pricing question that does —
how a surviving unit's cost ranks when symbolic in uniforms — is JP-14's
ruling (a-bound-is-a-value §5.1) and is not decided here.

### 9.1 Across unit sizes, and the break-even (JP-19, 2026-10-27)

The ruler now sweeps the unit: fold bodies (`loopN`, N trips: a loop, so bytes
stay small and time grows) and straight-line bodies (`wideN`, N distinct
`sqrt` terms: bytes and time both grow, the shape of a glyph-scale unit).
Marginal cost of one more inlined copy, from the 1-copy and 16-copy scenes:

| unit | bytes / copy | ns / batch / copy |
|---|---|---|
| loop2 | 185 | 9 |
| loop8 | 185 | 31 |
| loop32 | 227 | 130 |
| loop128 | 247 | 576 |
| wide16 | 2,505 | 79 |
| wide64 | 10,560 | 356 |

(`ns / batch` is per `jit_vector_bytes()` of lanes. The 8-trip fold above is
unrolled, hence loop2 = loop8 in bytes.)

No call lowering exists (L5), so the call side is a parameter, not a
measurement. With `c` the call's cost per batch (frame, arguments, result, the
spill of what the allocator holds across it — §5.2) and `s` its call-site
bytes, K copies of a unit of `B` bytes and `T` ns per batch:

- bytes saved by calling: `(K−1)·B − K·s`; positive whenever `B > s·K/(K−1)`.
- execution lost: `c / T` per copy, relative.

Reading the table with `c` of order 10 ns and `s` of order 50 bytes (stated
assumptions, not measured):

- **Loop units** gain nothing worth a call: they are ~200 bytes already,
  within a few call sites of `s`, and `c/T` is 100% at loop2 and 2% at
  loop128 — it is the *trip count*, not the call, that makes them slow.
- **Straight-line units** are where a call pays: `wide64` saves ~10 KB per
  extra copy for a `c/T` of ~3%. That is the only regime in which preserving a
  reference changes anything measurable, and it is a code-size and
  instruction-cache argument, not an execution one.
- The choice is therefore a function of `B` (bytes, known after emission) and
  `K`, not of the e-graph's cost: extraction prices time, and time does not
  separate the two. This supports §5.2's reading that the decision is
  codegen's (a unit's emitted size against its call-site count), after
  extraction, rather than a rewrite.

What this leaves open is `c` and `s`, which need the call lowering, and
compile time, which this ruler does not report: an e-graph that sees `K`
inlined copies of a `wide` unit pays for them in saturation, and a preserved
unit saturates once (`pixelflow_search::runtime`'s units). That, not
execution, is the likelier win and is measurable today by compiling the
scene with its `Apply`s held as units; today they are expanded first.

### 9.2 Compile time: the budget caps the cost and the quality pays (JP-19, 2026-10-27)

JP's constraint: sharing across plain composition globally is whole-program
optimization (fat LTO) — compile time explodes — so sharing belongs *after* a
kernel is inlined, scoped to that body. The ruler now reports cold compile
(`cold_compile_ms`: every row's unit carries a fresh literal, so no saturation-
or JIT-cache hit) and, built with `--features pixelflow-search/saturation-telemetry`,
the saturation record of each compile.

The finding is not the one fat LTO predicts. Compile time does **not**
explode with `K`; it is held down by the budget, and what falls is the search
each copy gets. Every one of the 19 saturations stopped at `class_cap`. For the
2-trip unit (16 classes inserted alone):

| copies | classes inserted | applications | wall |
|---|---|---|---|
| 1 | 16 | 16,463 | 76 ms |
| 4 | 52 | 10,020 | 27 ms |
| 16 | 194 | 5,796 | 5 ms |

Sixteen times the work gets a third of the applications, so each copy is
searched ~45× less than the same unit saturated alone — and the scene "compiles
faster" because it is optimized less. (Wide units, whose caps scale with the
input, do grow: `wide64` 7 → 31 → 120 ms for 1 → 4 → 16 copies, 6 → 37 → 165 KB
of code — linear in the copies, not superlinear.)

So the inline-everything cost is paid in **quality**, not seconds, and it is
exactly the whole-program pathology JP names: the e-graph holds the copies in
one search space and the budget is split across them. A unit saturated once and
linked (`runtime`'s Law U, which a held `Apply` would reach) gets the full
budget per distinct unit and costs the outer term only its own few classes.
That, not the call, is the case for keeping the name through extraction.

Consequences for the design question:

- **Share after inlining, scoped to the inlined body** (JP): agreed by the
  numbers. Sharing across plain composition would put every copy in one
  e-graph — the 194-class row — and is where the per-copy search collapses.
  Hash-consing at link time (`link` splices into a hash-consed arena) is the
  scoped form and already exists.
- **The unit boundary is what protects the optimizer's budget**, independently
  of whether a unit is emitted inline or called. `Apply` being expanded before
  the unit walk (JP-110) forfeits it; holding the `Apply` as a unit until
  after extraction, then expanding (the "expand" lowering, late), would keep
  the budget per unit with no call at all. That is measurable with a small
  change and is the next JP-19 step; the call lowering (L5) is only about the
  bytes in §9.1.
