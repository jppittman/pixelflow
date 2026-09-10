# The graph differentiates; legalize is the fallback

**Date:** 2026-09-09
**Status:** Findings. No code. Three defects in the runtime pipeline's
ordering, and a criterion for when `Gather` is a leaf and when it is a
scheduling decision wearing one.
**Author:** JP (direction and both corrections), Claude (draft)
**Bears on:** S1b of
[2026-09-09-glyph-as-a-fold-execution.md](2026-09-09-glyph-as-a-fold-execution.md),
which cannot compile as planned until §3 lands.

---

## The intent

The e-graph differentiates. `Dwrt` is a first-class node in it
(`egraph/ops.rs:208` resolves it, `egraph/derivative.rs` holds the rules,
`egraph/rewrite.rs:153` expands it one chain-rule step per rewrite), and
`egraph/cost.rs` prices a *surviving* `Dwrt` at `usize::MAX/4` precisely so
extraction can never select an unlowered derivative. That pricing is only
coherent if the node is expected to live in the graph and be rewritten away
there.

`passes::legalize`'s `lower_dwrt` is the **fallback** — for arenas that never
reach the e-graph, and for whatever the rules do not cover.

The runtime pipeline has it the other way round.

## 1. The fallback is promoted to primary

`pixelflow-search/src/runtime.rs:171`:

```rust
pipeline![LowerDwrt, ExpandReduce, Saturate::runtime(shape)]
```

`LowerDwrt` runs *before* saturation, so no `Dwrt` node ever reaches the
rules that exist to rewrite it. The comment justifies it: differentiation
manufactures constants, and `ConstantFold` can only cascade over constants
that exist by the time saturation runs.

That argument holds against lowering *after* the e-graph. It does not hold
against differentiating *inside* it, which is what the rules are for — the
constants are then manufactured mid-saturation, where folding cascades over
them anyway.

## 2. `ExpandReduce` runs early because it is forced, not because it is right

`egraph/ops.rs:232`:

```rust
OpKind::Buffer | OpKind::Gather | OpKind::RawGather | OpKind::Reduce | OpKind::Uniform => None
```

`op_from_kind` returns `None` for `Reduce`. Unlike `Gather` — which
`Vocabulary::resolve` special-cases to `Some(&Gather)`, opaque to every rule
but present as structure, so hash-consing still sees it — `Reduce` has no
such case. Saturate before unrolling and the e-graph declines the whole
arena, which for a post-S1a glyph means compiling with no CSE and no fusion.

So the order is forced. The stated benefit — "what saturation sees is
binder-free arithmetic it can CSE and fold across the unrolled terms" — is
real but small, since the unrolled copies differ only in a gather's row
index, and it is a rationalisation of a constraint rather than a statement
of it.

**The cost is the one measured today.** Unrolling first means saturation
sees `N` copies of one body. A glyph carries one piece per row, so a
200-piece glyph pays ~200× the e-graph work on structurally identical terms.
`loop_blinn_winding` went 277s → 750s and `kernel_glyph_optimize` sits at
907s once S1a introduced a reduce; that is the mechanism, and it was
mis-attributed to "saturation cost" at the time.

**Measured, 2026-09-09**, nodes before and after `expand_reduce` on a real
glyph at 32 px (after `lower_dwrt`, which is what saturation is handed):
`A` 8,741 → 33,557 (3.8×), `O` 43,583 → 256,379 (5.9×), `8` 87,007 →
758,059 (**8.7×**). That is the size of the graph the e-graph saturates
over, against a body it could have optimized once.

The fix is to give `Reduce` what `Gather` has: representable as structure,
nameable by no template. **It is not the one-line change this section first
implied.** `push_reduce` encodes the combiner, the binder slot and the
extent as `Const(f32)` *children* (`arena.rs:547`), and in an e-graph those
hash-cons with every other occurrence of the same value — `Const(4.0)`'s
e-class holds `Add(2.0, 2.0)` as soon as anything folds to four, so
extraction is free to hand `expand_reduce` a non-`Const` in the binder slot.
It picks the `Const` on cost today, which is a convention doing a type's
job — the case CLAUDE.md already names when it lists "`push_reduce` encodes
an `OpKind` as a `Const(f32)`". Doing it properly wants a new `ENode`
variant carrying that metadata (the `ENode::Buffer(BufferDecl)` pattern),
which reaches hashing, extraction, cost and rebuild.

**And it is sequenced after §4/§5, not before.** Making `Reduce`
representable is machinery for a body whose expense is reading a piece
table through a `Gather` — building around the construct §4 removes. The body is then one e-class, optimized once, and
`ExpandReduce` unrolls code that is already optimized. No rule can match the
node, so nothing can move across the binder — the same argument that makes
`Gather` sound today.

## 3. `Dwrt` over a `Gather` has no rule on either side

`passes.rs:626` and `:815`:

```rust
ExprNode::Buffer(_) => Err("lower_dwrt: cannot differentiate a bound-memory read"),
OpKind::RawGather  => Err("lower_dwrt: cannot differentiate a bound-memory read"),
```

The fallback **refuses categorically** — bound-memory reads, integer/bit
ops, reductions. And the rules do not cover it either: `derivative.rs` names
no `Gather` or `Buffer`, so the rewrite never fires, the `Dwrt` survives,
extraction cannot afford it, and `arena_to_schedule` panics on one that
reaches codegen. An error on one path, a panic on the other.

**This blocks S1b outright.** The distance fold is
`in_pixels(value, scale) = value / ‖∇scale‖`, and S1b's whole content is
that `scale` is built from coefficients read out of the piece table. That
puts a `Gather` under a `Dwrt`, and nothing can lower it.


**The remedy first drafted here was wrong** and is recorded because the way
it was wrong is the point. It proposed a rule — *a gather whose index does
not mention a coordinate differentiates to zero* — which is true, and is a
special case invented to replace information that had been thrown away.
Materializing is what destroyed the derivative. See §5.

## 4. Memory-backing is opaque to the consumer. That is a law.

**A consumer of a `Kernel` cannot observe whether evaluating it runs
arithmetic or reads a table.**

This codebase already states the same law once, about a different
implementation detail. From CLAUDE.md: *"SIMD is an implementation detail.
`Field` — one SIMD batch, the collapse ABI's vector — is `pub(crate)` in
pixelflow-core, and nothing outside that crate can name it, a lane, or a
width."* Memory-backing is that law's second instance, and it has not been
enforced.

So `Gather` does not survive as an op in the algebra — not for memoized
computation, and **not for genuinely external data either**. An image, the
terminal's cell contents, a font's bytes: each enters *as a kernel*, and a
caller adds it to a circle with `+` without learning that anything is backed
by anything. `Gather` becomes something the emitter does, like a register
spill. Nobody writes a spill into their expression.

An earlier draft of this section proposed a criterion — *could the program
have computed this content?* — to sort legitimate leaves from
materialization decisions. The law makes the sort unnecessary: neither kind
is nameable. That criterion was the last place this document still treated
memory-backing as something the language expresses.

What stops being consumer-facing:

| gone | why it existed |
|---|---|
| `Gather`, `Buffer` as nameable ops | the algebra was not closed over manifolds |
| `BufferIdentity`, `Manifold::bind` | a kernel held a *name*, and the data travelled separately |
| `MAX_BOUND_BUFFERS = 4` | a limit on how many names one program may mention |
| `Glyph { kernel, binding }` | there was never a second thing to carry |
| `Union::place` panicking on a summand with a slot | a summand could not declare one |
| the collapse corpus carrying buffer contents | fixtures held a name, so the numbers had to be shipped beside it |

The `width × height ≤ 2^24` bound stops being a language constraint and
becomes the emitter's problem, where it can be solved rather than
documented.

### The history this restores

In the expression-template era a `Field` **was** a manifold, and so was a
baked buffer, and so was an expression. `circle + square` and
`circle + baked_glyph` were the same operation, because the algebra was
closed over manifolds and reading memory was not an op — it was just being
one.

The JIT-first move (docs/plans/2026-07-20-kernel-unification.md) was made
for real reasons and is not in question. What it dropped is closure: a
`Kernel` became a value holding a *promise*, and `Gather` plus the binding
tables are the scar tissue over that wound. Every separate gap fought on
2026-09-09 — threading `binding` through 88 call sites, `Union::compile`
binding `&[]`, the corpus, `MAX_BOUND_BUFFERS`, `Dwrt` having no rule, e-graph
opacity — is that one gap seen six times.

## 5. The kernel keeps its code, `pub(crate)`

A `Kernel` retains the arena that defines it. The library remembers; the
consumer cannot reach in.

That is what makes §4 a fix rather than a hiding place. With the code
retained, `index(collapse(f)) = f` stops being a law in a doc comment and
becomes a **rewrite rule the e-graph can apply in both directions**:

```
Gather(collapse(f), p)  →  f(p)        read through the table to its meaning
f(p)  →  Gather(collapse(f), p)        materialize — a scheduling decision
```

Which direction to take is a schedule choice, which is where it belonged.

And §3 dissolves rather than being fixed. `Dwrt` over a tabulation is not a
missing rule; it is not a question. You never differentiate memory — you
differentiate the code the memory caches. The special case §3 first proposed
exists only because materializing threw the code away and something had to
be invented to stand in for it.

Two costs to settle before believing this:

- **Lifetime.** A kernel reading a baked buffer keeps the defining arena
  alive. For an atlas that is every glyph's outline retained behind every
  sample. Probably fine — arenas are small beside the buffers — but it
  changes what holding a `Kernel` costs.
- **Cache identity.** Two tabulations with identical data but different
  code, or identical code baked at different shapes, must key correctly.
  Today `jit_cache` keys on arena structure plus shape, with a `Buffer`
  contributing its slot and extents; if the code comes along, that key has
  to decide whether it identifies the reader or the read.

## 6. Providing a range requires providing a select

**Decision (JP, 2026-09-09), recorded with its own hesitation intact: the
API for bounding a kernel to a range should *require* the value outside it.**

Today an out-of-range read answers with floor-then-clamp-to-edge
(`passes.rs:261`, `eval.rs:1758`) — the instruction's answer, not the
denotation's. `fonts/cache.rs` writes the repair by hand and its doc says
why: a clamped read smears a descender's boundary coverage out to infinity.
That repair is a `Select` the caller had to remember. Requiring it at
construction is the same expression, moved to where it cannot be forgotten:

```text
bounded(range, inside, outside)  ≡  in(range).select(inside, outside)
```

Three things follow, and they are the argument:

- **Totality is structural.** A bounded kernel is total by construction.
  There is no undefined region, no convention, and nothing for a consumer
  to remember — which is CLAUDE.md's *when you extend a type's meaning,
  extend its type*, applied to the meaning "defined only here".
- **Clamping is a guess, and it is usually wrong.** The library cannot know
  what lies outside; the caller can. `cache.rs` is the existing proof that
  the hardware's guess needed overriding.
- **It makes the out-of-range case visible to the compiler.** A clamp is
  invisible and unremovable. A select is a `Select`, so demand analysis can
  see the region, guard it, or eliminate it where the range is statically
  known. The construct that costs an instruction is the one the optimizer
  can remove; the free one is the one it cannot.

Where it is shaky, honestly:

- Most callers want zero outside, so it is boilerplate at every site. A
  named constructor for the common case answers this without weakening the
  requirement.
- It spends a select where the hardware offered a clamp for free. That is
  the trade above — visible and removable against invisible and permanent —
  and it should be *measured* on the atlas rather than argued.

Under §4 this is also what an off-lattice sample of a tabulated kernel
means: not the edge texel, and not a cache miss that recomputes, but
whatever the caller said was out there when they bounded it.

## Order

3 → 2 → 1 was the original order and **§3's remedy is withdrawn**; what
remains of §3 is a live blocker on S1b with no local fix. The order is now:

- **§4 + §5 together.** They are one change: the boundary that hides
  memory-backing, and the retained code that makes hiding it sound rather
  than lossy. Neither works alone — a hidden buffer whose code is gone still
  cannot be differentiated.
- **§6** rides with them, because the out-of-range answer has to come from
  somewhere the moment clamping stops being observable.
- **§2** (`Reduce` representable, so saturation stops seeing N copies)
  is independent and can land any time; it is worth ~the piece count in
  glyph compile time.
- **§1** (delete `LowerDwrt` from the runtime pipeline) is last and is
  mostly a deletion once §5 means the graph can differentiate through a
  tabulation.

S1b stays blocked until §4/§5 land. That is a real cost and it is the
honest one: the alternative was the special-case rule in §3, which would
have unblocked S1b by entrenching the thing that caused the block.
