# Composition is linking, and the linker only inlines

**Date:** 2026-09-09
**Status:** L1–L3 executed — the reference, the store, shared expansion,
and one composition site converted and measured (§2.1, §7). §5's first cost
has its gate; the inlining rule (L4) waits on a kernel that composes
identical referents, because §7 found the glyph is not one.
**Author:** JP (the framing and both design decisions), Claude (draft)
**Supersedes the framing of:**
[2026-09-09-the-graph-differentiates.md](2026-09-09-the-graph-differentiates.md)
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
[the-graph-differentiates](2026-09-09-the-graph-differentiates.md) §4 — a
consumer cannot observe whether evaluating a kernel runs arithmetic or reads
a table — is this, restated. A reference is a kernel; so is a tabulation; so
is an expression. `circle + baked_glyph` is ordinary `+` because both sides
are the same kind of thing.

## 2. Identity is the compile cache's key, and they are not two things

**Decision (JP, 2026-09-09).** A referent is named the way `jit_cache` names
one: `canonical(arena, root)` walks the reachable subgraph in ascending id
order, remaps children densely, and encodes each node — a structural content
hash. Two kernels with that key are the same kernel.

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
saturation Guide (docs/plans/2026-08-31-guide-design-revision.md) chooses
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
[glyph-as-a-fold-execution](2026-09-09-glyph-as-a-fold-execution.md) — the
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
