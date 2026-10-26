# A bound is a value

**Status:** `Plan of record` (JP, 2026-10-20: "yes, write it. Then execute.").
**Created:** 2026-10-20.
**Verified against:** `6754e4031d4c58484fec04d22328517435745284`.
**Reverses:** JP 2026-10-09 ("the atlas becomes the kernel for that number of control
points, everything else is a uniform"), the basis of `FontPrograms`' one program per
piece count; and `CLAUDE.md:175` ("a fixed-count iteration is unrolled at construction;
a trip count that must change is a recompile through the shape-keyed cache"). It amends
the documents §5 lists, `2026-10-17-the-arena-is-typed.md` among them, and replaces none
of them.
**Depends on:** `2026-10-17-the-arena-is-typed.md` T3 and its §5 item 2, for B1–B3 only.
T6, with the bound retyping §5 adds to it, lifts §1.4's arithmetic refusal and deletes
the integrality conjunct; nothing in §6 waits for it.

The rulings, quoted and not re-argued (JP, 2026-10-17 to 2026-10-20):

1. *"A loop iteration count in pixelflow could have any variance no? Except being a
   function of x,y,z,w. That's explicitly not allowed."*
2. *"Why are you talking about what simd needs? Vectorization should be trivial here."*
3. A trip count is a value; constness is discovered, not declared ("the thing we steal
   from LLVM").
4. *"Any time you want to reach for an index, you must use a discrete kernel."*
5. *"The derivation should stay in rust."*
6. Per call, not per JIT compile: a font at one tile extent is one program for every
   glyph and piece count, and a run of glyphs is one program too.

## 0. The claim

A value's binding time follows its variance. A fold's follows its syntax: a literal, a
`const N: usize` that rustc evaluates in a `const` block (`StructuralRange` in
`pixelflow-compiler/src/sema.rs`, `instantiated_range` in `pixelflow-compiler/src/emit.rs`),
and the `u32` ends stored in the `Fold` node (`pixelflow-ir/src/fold.rs:289-315`). So a
count known per call is forced to be known per compile, and every count a program meets
is a program of its own. The fix: a fold's bounds are operands, placed by the machinery
that places every other value and refused only when they vary with the coordinates. An
index is application of a discrete kernel, and a fold over that kernel's lattice is
bounded by its extent, a value that travels with the argument. Therefore a font at one
tile extent is one program, and drawing a glyph is a call.

## 1. Denotation

### 1.1 The fold

```
⟦Reduce{⊕, b, lo, hi, body}⟧ = ⊕_{i ∈ [⟦lo⟧, ⟦hi⟧)} ⟦body⟧[b := i]
```

`⊕` is a monoid, `b` the binder, `lo` and `hi` terms of sort count (`Int` by type after
T6 and the bound retyping §5 adds to it; until then an integer in a `Real` lane, the
convention the binder already has, `fold.rs:318-335`). The monoid and binder are the
node's identity. A bound is a term like any other; a bound found to be a constant is
kept as node data and any other bound is a child beside the body (§3.1). `[a, b)` with
`a ≥ b` is empty, as Rust's `5..3` is; the literal spelling keeps its refusal of a
reversed range (L2).

The one rule (ruling 1): **a bound's variance shares no bit with the coordinate axes or
the lattice's own binders.** It may share an enclosing user fold's binder (a triangular
`0..i`); it may not read its own fold's binder. Refused at construction. The reason is
denotational (ruling 2): a bound fixed before the lattice is traversed means every point
runs the same loop and the cost is known before any pixel runs
(`2026-07-24-totality-and-the-cost-model.md`). The emitter's uniform `done` test
(`pixelflow-codegen/src/emit/select.rs:312`) is a consequence, never the reason.

### 1.2 The binding times that remain

A bound is placed where its variance says, by the rule that places everything
(`pixelflow-codegen/src/program/scopes.rs`, the `hoisted`/`deeper` test), with the one
addition §3.3 names for reaching the loop's latch:

| A bound that is | Its variance | Computed |
|---|---|---|
| a literal or a `const` item | known (node data, §3.1) | at expansion; a pool constant |
| a uniform, an argument's extent (`Extent(p)`, §1.3) | `Variance::CONST` | once per call, in `Scope::Body` |
| a function of an enclosing fold's binder | that binder's bit | once per enclosing trip, in that fold's scope |

The rulings' "Scope::Root" is `Scope::Body` (`pixelflow-codegen/src/program/mod.rs:207-214`).
There is no fourth row. "Structural" is not a binding time: a structural count was a
per-call value spelled so that rustc evaluated it. Ruling 1 admits every row, so a bare
per-call count is legal; decision 2 is only how `kernel!` spells it.

### 1.3 A discrete-kernel argument

There is no array (`2026-10-15-kernels-evaluate-kernels.md` §2: "a per-frame table is a
parameter of arrow type", "the extent is a value, not a type"). A glyph's pieces are one
argument `pieces : L → R`, `L` a one-dimensional lattice of extent `n`, supplied per
call. Its codomain `R` is open (decision 6): `L × Fin 10 → Real`, today's table shape and
legal in §2's grammar, or a record of ten channels, which `T ::= S | D → T` does not have
and would need a grammar extension. Ruling 4 says the fold runs over `L`, and that is the
denotation, not a convention:

```
ink(x, y) = Σ_{i : L} piece_term(pieces(i), x, y)
          = Reduce{+, i : L, 0, Extent(pieces), piece_term(pieces(i), x, y)}
```

- **The binder's domain is `L`.** `i : L`, not an integer that happens to stop at `|L|`.
  The bounds are derived from the argument and never written: `lo = 0` and
  `hi = Extent(pieces)`, a named IR leaf of `Variance::CONST` whose value is the extent
  bound with the argument at the call (row two of §1.2). Whether the index set is that
  leaf on each fold or a range node shared by sibling folds
  (`docs/archive/plans/2026-09-23-a-glyph-is-a-formula.md` §4.2, "A range is a value") is
  representation, decision 9; the denotation is the same.
- **In domain by type.** `pieces(i)` with `i : L` is in `L`. An index of any other type
  reaches `L` only through a named lift into `L` (§2's lift rule), and the lift decides
  the outside. For a `Real` codomain the outside is NaN, CLAUDE.md's documented-domain
  rule, as §1.4's is. Nothing is clamped implicitly once `lower_gather`'s clamp goes (B1).
- **A run** adds `spans : G → (start, count, box)`. The outer fold is over `G`; the inner
  fold is over `Lattice(count(g))` and reads `pieces ∘ ι_g`, where `ι_g` is the lift
  `j ↦ start(g) + j` into `L`. Its bounds `0` and `count(g)` vary with the outer binder,
  legal by ruling 1, and are leaves (§1.4). No type can carry `start(g) + count(g) ≤ |L|`,
  so the window is not in domain by type. What the type gives is a defined outside: a
  malformed span reads the lift's NaN, never memory past the table, and no bind-time host
  invariant stands in for a type.

Outline → monotone pieces → `piece_row` stays host Rust (ruling 5).

### 1.4 Out of domain

A count denotes an integer in `[0, 2²⁴]`, where the `f32` counter is exact
(`Fold::EXACT_BOUND`, `fold.rs:335`). Past it `step + 1.0` sticks and the loop never
exits; a NaN `end` exits after one trip on x86 and never on aarch64 (CLAUDE.md's `Ge`
row). What a fold with a bound outside the domain denotes depends on its sort:

| Monoid | Sort | Out of domain |
|---|---|---|
| `SUM`, `PRODUCT`, `MIN`, `MAX` | `Real` | **NaN**, CLAUDE.md's documented-domain rule: not a clamp, not a hang (§3.3 computes it) |
| `ANY`, `ALL` | `Mask` | open (decision 8). NaN is not a mask: `0x7fc00000` blends two arms into a value neither held, the `mask & 1.0` failure CLAUDE.md names. Until decided, `push_reduce` refuses these monoids with a value bound (`BoundSort`) |
| `SEQ` | the lattice's | refused with a value bound (`BoundSort`); the lattice's folds keep constant ranges (§4, "Deferred") |

After T6 an `Int`-sorted fold has no NaN either and falls under decision 8.

A count the host writes is refused where it is written, by `KnownRange::admits` turned into a
predicate on one count value: at `kernel!`'s host write of a per-call count, an entry's
`n: usize` (decision 2), by its host function and, if it has one, by its `Args` record's
`write_into` (A6), and at bind for a discrete-kernel argument's extent (B1). `2²⁴ + 1` can be refused
only there: no `f32` names it, and a uniform written as `16777217.0` is `2²⁴`, which is in
the domain. The NaN is the in-language backstop.

**Exact only for leaves until bounds are `Int`.** The guard sees the value the lane
holds. A leaf bound (a `Const`, a uniform, a binder, an `Extent`, an application of a
discrete kernel) is that value exactly; a computed bound is `f32` arithmetic and can
round into the domain: the exact sum `2²⁴ + 1` of two in-domain counts rounds to `2²⁴`,
and the fold runs one trip short with no NaN. So until T6 types bound arithmetic as
`Int`, the two user surfaces, `Kernel::over_values` and `kernel!`'s sema, refuse a value
bound that is not a leaf (`BoundArithmetic`). `push_reduce` and the e-graph do not: a
class has no leaf shape (`i` and `(i + 1) − 1` are one class), and the rules make bound
arithmetic of their own. `HalveFold` substitutes `binder := binder + s` and, by L7, turns
an inner `0..i` into `0..(i + s)`; `PeelFold` plans `hi − s`. Both stay inside the outer
fold's known range, which ends at or before 2²⁴, so they are exact, and the rounding window
cannot open there. (The alternative, `HalveFold` declining over a body whose fold bound
reads the halved binder, is simpler and loses the halve on exactly the triangular case;
this plan does not take it.) The triangular `0..i`, L8's discovered bounds and B3's
`0..count(g)` are leaves; a user's `0..i + 1` waits for T6.

### 1.5 Totality and cost

The language stays total. Every trip count is a value known before the lattice is
traversed, so a program's cost is a polynomial in values known per call; a fold over an
argument is linear in the data the caller supplied:
`|L|·cost(body) + (|L|−1)·cost(⊕) + cost(lo) + cost(hi)`. The static model prices before
the values exist, so it uses the trip count when the bound is a known constant and a
default estimate otherwise (§4, "Cost and regalloc with an unknown trip count"). The cost
is closed-form per call, no longer per compile. **No step in §6 evaluates it per call or
checks a budget against it:** the static estimate is the only price, so a value-bounded
program is never reported over budget. `docs/DENOTATION.md:62-66` is amended to say so
(§5).

### 1.6 Laws

- **L1 (constants).** A fold whose bounds are both literals denotes what it denotes today
  and emits byte-identical code. A known bound is node data, as a shift count is, so no
  `Const` is interned for it and no other literal changes position (§3.1).
- **L2 (empty).** `⟦lo⟧ ≥ ⟦hi⟧`, both in the domain, ⇒ the identity of `⊕`. A known pair
  is normalized when it is read: `known_range` has no reversed inhabitant, so
  `over_values(5, 3)` and an L8 substitution reaching `5..3` are the empty range. An
  unknown pair is decided per call by the guard's `run` test. The literal
  `Range<u32>` spelling (`Kernel::over`, `kernel!`'s `5..3`) is still refused at
  construction, as today.
- **L3 (legality).** `(var(lo) ∪ var(hi)) ∩ (COORDS ∪ {b}) = ∅`, computed by the arena's
  per-node variance transfer function run from each bound and memoized (`variance_from`,
  `pixelflow-ir/src/variance.rs`) — the one `compute_arena_variance` runs forward, so a
  `Ref` the bound reaches is resolved through the store and one it does not reach is not —
  the same function L4 extends; so construction refuses what `collapse` would otherwise
  assert. It is checked by binder
  slot, so it forbids shadowing: a fold may not rebind a slot its own bounds read, and
  `lowest_free_binder` scanning the bounds means `Kernel::over` never builds one. The
  lattice's row, column and lane binders do not exist until `collapse` substitutes X and
  Y (`pixelflow-ir/src/passes/lattice.rs`), so at construction L3 is "no X/Y bit, not the
  own binder"; after `collapse` lowering asserts it.
- **L4 (variance).** `var(Reduce) = (var(body) \ b) ∪ var(lo) ∪ var(hi)`.
- **L5 (domain).** Bounds denote integers in `[0, 2²⁴]`; outside, a `Real`-sorted fold is
  NaN, and the other sorts are §1.4's table.
- **L6 (unrolling).** `PeelFold`, `HalveFold` and `EmptyFold` fire only when both bounds
  are known (ruling 3). `EmptyFold` does not fire on class equality `find(lo) == find(hi)`:
  `[v, v)` is the identity only for an in-domain `v` and NaN otherwise (L5), and only a
  known `v` shows which.
- **L7 (substitution).** Substituting a slot into a fold substitutes into its bounds
  always (by L3 no bound reads the fold's own slot), and into its body only when the fold
  does not rebind the slot. A class reached as a bound, or under one, is rebuilt through its
  first member that keeps L3 and is no fold, because a class's fact is the meet over its
  members and its first member can read a coordinate the class does not; the rewrite
  declines when no member does. The rebuilt bound is a term, and the graph hash-conses it:
  its class can be seeded wider than the term (a fact is never repaired upward), whether the
  term lands in a class that already exists or is minted over a child that did, and L3 then
  reads that fact. The graph refuses such a fold to the plan builder, and the rewrite
  declines there too: it makes no union, and the nodes it made before that fold stay, as
  orphans that cost classes and never soundness (a lawful nested fold can be among them);
  `EGraph::add` panics only for a caller that is not a plan.
- **L8 (discovery).** Peeling an outer fold substitutes `i := Const(k)` into an inner
  bound `0..i`; the bound's class gains a `Const` member, the one normalizing rule
  (`KnownBound`, §3.2) reads it into the node, and the inner fold becomes known and
  unrollable. Nothing declared it.
- **L9 (identity).** A fold's canonical key is its header and its children's keys. A
  known bound is in the header, where today's range is; a uniform bound contributes a
  dense slot, not a value, so one program serves every `n`.

## 2. Why the current shape is wrong

Every row is one cause: the binding time of a count is attached to its syntax.

| Symptom | Measured | Traced to |
|---|---|---|
| ASCII needs **29 programs**, one per piece count, holding **528** `one_piece` copies (Σ n) | quoted from the-language-is-kernel `:1186-1188`; not re-measured ([A0](../results/2026-10-20-production-glyph-baseline.md) §4.4) | `FontPrograms` (`pixelflow-graphics/src/fonts/loop_blinn/program.rs`) has no fold; a count cannot be a uniform, so the count is the structure |
| `⌨` (347 segments, 262 pieces) is **27,029 defs**, a **75 KiB** frame past NEON's 64 KiB reach (#1414), as a `FontPrograms` program; production's `Glyph::bake` keeps the fold, at 310 defs and a 1.4 KiB frame | quoted from `docs/codegen/2026-09-10-a-surviving-reduce-is-a-loop.md:710-720`, `5574374a`, not re-measured; production's figures are A0 §3.2 | the same: one term per piece, by construction |
| production pads counts to powers of two (`bucketed_trip_count`, `padding_row`): **95 → 7 programs** at tile 16 and at tile 32, six buckets (2 to 64) and the blank; a new tile extent mints all seven again | A0 §3.1, §3.5 (BACKLOG H1's 6 and 7 are not reproduced) | the trip count is in the fold's `to_bits` (`FoldHeader::to_bits` since A2b), so in the JIT key (`pixelflow-ir/src/key.rs:445-450`) |
| each bucket **saturates separately** | A0 §3.1: 7 saturations, 6 stopped by the class cap; `pixelflow-search/src/runtime.rs`, `CACHE` keyed on canonical bytes | the same key |
| a run's program depends on every character's row offset | `loop_blinn.rs:423`, `start..start + count` | data spelled as a constant |
| `n = 0` is a program of its own, `blank()` | `program.rs:138` | an empty count cannot reach a fold |
| `kernel!` refuses `(0..4).map(\|i\| (0..i)…)` | `sema.rs:2725-2741` | the bound grammar evaluates to a `u64` at expansion |
| a structural range is `0..0` strided by `HOLE_STRIDE + k`, and its entries are never optimized | `pixelflow-compiler/src/lower.rs:283-364`, `emit.rs:133-156` | a per-instantiation value has no home but the node's bits |
| the arena plan's compile numbers are on `FontPrograms`, not production's `Glyph::bake` | `docs/archive/plans/2026-10-15-the-egraph-is-an-arena.md` | the per-count design became the workload |

On production the cost is modest. A0 measures a cold ASCII warm at about 0.1 s, half of it
the host building kernels (which ruling 5 keeps), and about 50 ms compiling the seven
programs; the bake is under 1%. The seconds-long warms and the 27,029-def program were a
September tree and `FontPrograms`. So the case for this plan is the language (rulings 1, 3
and 6) and `FontPrograms`' deletion. On production's frame time, B2 saves at most the
~50 ms per font and tile, and six of the seven emits per zoom.

The 2026-09-10 plan deferred exactly this (§5): "the key must stop carrying the range,
and the loop must compare against a loaded value rather than an immediate". JP left room
for it: "we may relax this at some point and allow uniform iterations"
(`2026-09-25-the-language-is-kernel.md:78-81`). This is that relaxation, and ruling 1 is
its edge.

## 3. The representation

### 3.1 IR (`pixelflow-ir`)

- `Fold` keeps `{ monoid, binder, stride }`. `ExprNode::Reduce` is
  `{ fold, lo: Bound, hi: Bound, body }`, `Bound::{Lit(u64), Value(ExprId)}`. A known bound
  is node data, as `ENode::Shift`'s count is (`pixelflow-search/src/egraph/node.rs:204-219`),
  and for the same reason. As a `Const` child it would be hash-consed with every equal
  literal (`pixelflow-ir/src/arena.rs:11-18`): a literal pushed after the fold, as in a mean
  `Σ_{i<n} f / n`, would return the bound's earlier id, move ahead of the `Reduce` in arena
  order and so in schedule order, and change the constant pool's order. That breaks L1 in a
  shape no byte_probe row has. `children()` yields the body, then each `Value` bound; the
  arity check counts them. The `u32` ends, a 64-bit-rule debt (`pixelflow-ir/intent.md`
  "Behind"), go with them. `Lit` is `u64`, the control-plane width; `Fold::EXACT_BOUND` is
  the domain predicate on it, a data-plane cap from the `f32` counter that widening would not
  lift (`fold.rs:325-334`), not its width.
- **Constness is discovered, never declared.** Callers pass values. `push_reduce(fold, lo,
  hi, body)`, the one constructor, reads a `Const(Lit::Real(v))` bound (after #1431;
  `Lit::Int` after T6) with `v` an in-domain integer into `Bound::Lit`, and keeps any other
  bound as `Value`. It keeps a non-integer or out-of-domain `Const` as a `Value` bound,
  whose guard makes the fold NaN (L5), so a literal a rule or the folder made never panics
  at extraction; `BoundOutOfDomain` (today's `admits` as a predicate on one count) is
  refused at the user surfaces only: `over_values` and `kernel!` (A6). `Kernel::over`'s
  literal range is refused past 2²⁴ with `KnownRange::new`'s message (a fold's range ends
  at most at 16777216 (2^24)), as it is today. It refuses, by name: a `Lit::Mask` bound
  (`BoundSort`: a mask is not a count); L3 (`BoundVaries`); a value bound on `ANY`, `ALL`
  or `SEQ` (`BoundSort`, §1.4).
  `BoundArithmetic` (§1.4) is not among them: it is refused at the user surfaces,
  `over_values` and `kernel!`'s sema, because rules make exact bound arithmetic that
  extraction rebuilds through `push_reduce`.
- `ExprArena::known_range(id)` is `Some` when both bounds are `Lit`, and returns a
  normalized range with no reversed inhabitant (`lo ≥ hi` is the empty range at `lo`).
  `len`, `is_empty`, `peel`, `peel_back`, `halve` and `admits` move to that view, and every
  reader reads only it: `program/lower.rs`'s identity arm (`:37`, `:290`), the codegen
  assert, regalloc's trips. `len()` on `u32` ends today underflows on a reversed range and
  the do-while loop would run once; the normalized view has neither. `stride ≠ 1` only with
  a known range it divides, asserted at the push (only `pack` and `HalveFold` make strides,
  both on constants).
- `close_over` discovers constness before it copies (`Bound::discover`): a bound whose term
  is a `Const` naming a count is the fold's data and not a root, and the remaining `Value`
  bounds are extra roots (`from_arena_roots`). A known bound pushes nothing, so
  `over_values` over two constants is `over` over the literals, node for node.
- `Kernel::over(monoid, Range<u32>, body)` stays the literal spelling and keeps its
  reversed-range refusal. A new `Kernel::over_values(monoid, &lo, &hi, body)` splices
  bounds with uniform order body, lo, hi, written with T4's typed handles in view;
  `pub(crate)` in A3, public in A4 (decision 7).
- `compute_arena_variance` follows L4; its "empty ⇒ CONST" arm applies to a known empty
  range only. `free_index` unions the `Value` bounds' free indices unmasked;
  `lowest_free_binder` (`arena.rs:1630`) scans them. The walkers that push only the body
  (`depth`, `has_var`, `has_degenerate`, `node_count_subtree`, `term_arena`, `expr.rs`'
  `child_ids[0]`) would skip value bounds silently, so the CL carries a grep-gated
  checklist of every `Reduce {` site. `ExprArena::substitute_vars_with` is not an L7
  substitution: it enters every fold's body and bounds, so its callers substitute only
  slots no nested fold rebinds (coordinates, retired axes, a template's metavariables), a
  precondition its doc states.
- `canonical` (A4): tag 10 and the header, a `Lit` pair at today's `to_bits` positions (both fit:
  the domain ends at 2²⁴), so a constant fold's key is unchanged; a `Value` bound writes a
  marker in its header field and appends its child's key after the body's.

### 3.2 E-graph (`pixelflow-search`)

- `ENode::Reduce { fold, lo, hi, body }` with the same `Bound` over `EClassId`.
  `children_slice` becomes an inline array with a length (body, then the `Value` bounds),
  replacing `core::slice::from_ref(body)` (`egraph/node.rs:355`). The reason ranges left the
  children (`egraph/node.rs:226-239`, "every arithmetic rule could reach a trip count")
  holds for monoid, binder, stride and a known bound, which no rule reaches, as a shift
  count; it is reversed for a value bound: arithmetic reaching a bound is how L8 happens.
- `insert` reads a `Const` bound into `Lit` as `push_reduce` does. One rule, `KnownBound`:
  a `Value` bound whose class has a `Const` member (by member scan, as `constant_of` in
  `math/algebra.rs`, `Option<Lit>` after #1431, not `const_fact`, whose move out of
  `FrozenGraph` is a pending arena-plan decision) holding an in-domain integer adds the
  `Lit` form of the fold to its class. That rule is L8. A `Uniform` bound is legal (L3) and
  unknown (L6).
- `PeelFold`/`HalveFold` fire only on a known range; Peel plans `hi − s` as a new `Lit`.
  `EmptyFold` (`fold_rules.rs:319`) fires only on a known empty range (L6). `FactorFold`
  declines a known empty range only; under a value bound it factors, accepting the
  infinite-`c` edge at a runtime `n = 0` (decision 5). Under `SUM` a `c` the graph proves
  zero (a literal `0`, or one such as `X − X`) then meets the `Mul` annihilator
  (`0·y → 0`), which removes the fold before the guard is placed, so the optimized fold
  is `0` where the fold as written is NaN: the annihilation rule's edge, in
  `FactorFold`'s doc and BACKLOG E9. `substitute`/`rebuild` (`fold_rules.rs`)
  follow L7; `rebuild`'s comment (`fold_rules.rs:600-606`) is rewritten, since its premise
  ("its class's variance excludes it") now rests on L3 and L4 adds the bounds' bits.
  `ENode::variance` follows L4 in the same CL as the IR's; `insert` and `add` refuse L3 as
  a backstop, and do not refuse bound arithmetic (§1.4): `HalveFold`'s `binder + s` in an
  inner bound is the rules' own, and exact.
- **Pricing.** A known fold's `len` is read off the node. A value-bounded fold whose
  bound's chosen form is an in-domain integer literal is priced at that count — the price
  of the literal twin `KnownBound` puts in its class — and any other at `ESTIMATED_TRIPS`
  (§4). So pricing reads the bound children's extracted picks: the DP prices a fold only
  after every child class has settled (`extract.rs`'s `Settlement`), and hands the count to
  `fold_cost` and `below` through the fold's literal twin. `below` adds its `Value` bound
  operands once: it read only the body operand before value bounds, so bound arithmetic
  would have gone unpriced with no audit failure.
- `Value` bound classes **count** toward `inserted_class_count`: the budget is a function
  of the input, and the bounds are input. A known bound makes no class, so only kernels
  with value bounds can move across the 10/50/625 tier edges; that is measured, not
  avoided.

### 3.3 Codegen, placement, regalloc (`pixelflow-codegen`)

- `ScheduledOp::Reduce { fold, lo: Bound, hi: Bound, body }`, `Bound::{Known(u64),
  Value(ValueId)}` (one generic `Bound<R>`, so `Known` is spelled `Lit`; A2b), copied from
  the arena node in `arena_to_schedule`. There is no bound
  `Const` to read or to keep out of the schedule.
- **L1's mechanism.** `Known` keeps `self.constant(..)`, a pool constant, at today's points
  (`lo` at `select.rs:273`, `end` at `:311`) and today's `trips = parent_trips * len`
  (`:287`). Nothing else is emitted for a known range.
- **Placement has one new rule: a fold reads its value bounds.** Today `operands()` treats
  `ScheduledOp::Reduce` as a leaf (`program/mod.rs:363-381`), and the carve's closure walk
  starts at the body alone (`program/scopes.rs:247`), so a bound the body does not read
  would reach neither ownership nor the loop. `operands()` and `structural_children` yield
  the `Value` bounds, so `ownership::reads` (`program/ownership.rs:116-119`) and liveness
  see them. `lo` is read in the parent scope, as the head's seed. `hi` is read at the latch:
  `end` and the `Ge` compare are emitted in the head block after the fold's frame has popped
  (`select.rs:311-312`), so `lookup(hi)` resolves in the parent, and the read is inside
  the loop, so `hi` lives to the latch as an outer value does
  (`emit/regalloc/local.rs:44-46`). The `Reduce` def's interval spans the loop: the latch's
  compare reads `hi`, so the allocator keeps it live there by its own rule, and `operands()`
  yields the bounds so the carve and ownership see them. A compare reads two registers, so a
  `hi` the allocator does not carry is reloaded at every latch. A JIT test pins a `hi` read
  only by the loop under a body that varies with X
  (`a_bound_read_only_by_the_loop_reaches_its_latch`). Otherwise placement is today's: a
  `CONST` bound lands in `Scope::Body`; a bound on an enclosing binder makes the inner
  `Reduce` vary with it, and the existing carve places it in that fold.
- **The entry guard**, for a `Real`-sorted fold whose range is not known, is built as
  arena nodes by the last pass of `pixelflow_ir::passes::legalize` (`passes/guard.rs`), so
  every guard def takes its variance from `compute_arena_variance` (the alternative, following the
  `Context` precedent, `program/lower.rs:146-148` and `:199-211`, would hand-assign the
  bounds' union). Per `Value` bound `dom = Le(0, v) & Le(v, 2²⁴) & Eq(Floor(v), v)` (a
  known bound is in the domain by construction), `run = Lt(lo, hi)`, and
  `If(dom, If(run, R, identity), NaN)`. Every compare is ordered (never `Ge`, unordered on
  x86), so a NaN bound fails `dom`. `R`'s only reader is the `If`, so `Own` gives it to the
  true arm and the existing arm machinery guards it: no edge to `exit`, no accumulator
  parameter, the allocator's single-predecessor exit and head-only parameters
  (`regalloc/local.rs:1015-1022`, `:1114`) untouched. `select.rs:266`'s assert becomes "a
  known range is non-empty".
- After `collapse`, lowering asserts that a value bound reads no lattice binder (L3 against
  the `Write`s' binders) and that a value-bounded fold has exactly one reader, an `If` whose
  true arm is the fold: the single-reader property placement rests on (pipeline-bug
  panics). `FoldHeader::new` already refuses a stride other than 1 beside a value bound, so
  no stride assert is added. A value bound the emitter holds as a mask (a comparison, or a
  bitwise and/or of two such) is refused by name, at either end and on every tier, by
  `ScheduledOp::is_mask`: a mask names no count (`BoundSort`). Without the refusal every tier
  would read the mask's bits as a float: a value `lo` is seeded `lo + 0`, and AVX-512 widens
  an opmask there (`vpmovm2d`), so all ones reads as NaN, which fails the guard's domain test,
  and all zeros as the count 0. Any other bound whose value is
  a mask, a select between two masks, is a lane read as a float on every tier alike, which
  the guard turns into NaN or a count. The predicate and the AVX-512 selector share the
  bitwise rule (`BinaryOp::on_masks`), and a test selects every op that reads a lane over a
  mask and a number and holds the opmask file to the predicate's answer.
- `Trips::{Known(u64), Estimated(u64)}` from selection, a nested fold's trips the product
  of its parent's (an estimate at `u64::MAX` when the product passes `u64`); the allocator
  reads the number through `Trips::weight`, and its sums of weights saturate, so a deep
  nest is priced at the most it can weigh and still compiles. `EmitTraffic.trips` carries
  the type, and `dynamic_memory_ops` is `None` over an estimate, so no measurement shows
  an estimate as a count. `Display` prints a known count
  as the bare number and an estimate as `~n`, so byte_probe's `trips=` text
  (`examples/byte_probe.rs:306-307`) is unchanged and A4's gate compares like with like.

### 3.4 `kernel!` (`pixelflow-compiler`)

A bound is a `Ty::Usize` expression typed in the enclosing scope (`type_of_fold` types
its names before it pushes the fold's scope): a literal, a `usize` const, an enclosing
index, or an entry parameter `n: usize` of the items form (decision 2), a uniform.
An argument's extent arrives with B5's surface. X and Y are `Ty::F32` with no `as usize`,
so sema refuses L3 by type, the IR's `BoundVaries` behind it; every name in a bound is
typed before its arithmetic is judged, and a non-leaf value bound (`0..i + 1`) is refused
until T6 (§1.4). `lower_fold` resolves `lo`/`hi` before `open_fold`, which takes them,
and `Site::close_fold` hands them to the IR. A known bound keeps the expansion-time
checks it has (reversed, past 2²⁴), and a known end beside a value is checked alone
against 2²⁴. A count the host writes is refused past 2²⁴ by the entry's host function and,
if it has one, its `Args` record (§1.4); a helper and the closure form take no count. The structural
path went with `<const N: usize>`, its only producer, in A5: `Count::Structural`,
`StructuralRange`, `instantiated_range`, `FoldRange`, `Holes`, `OPEN_RANGE`,
`HOLE_STRIDE`, `ExpansionFold`, the lowering's `Param` hole and the macro tier's
template exemption are deleted, and `kernel!` refuses a generic on any `fn`.

### 3.5 The glyph (`pixelflow-graphics`)

One program per font at a tile extent: `glyph(pieces, box)`, §1.3's fold under the box
mask. `Glyph::bake` binds rows and box per call; `n = 0` is L2, the guard's `run` test.
`run` is the two-level form. Before item 2 this cannot be built: the table's extent is in
`BufferDecl` and the key (`key.rs:415-424`), in `lower_gather`'s clamp, and in
`Manifold::bind`'s length check.

## 4. Consequences, to closure

**Deleted.** `bucketed_trip_count`, `padding_row` and their pins. `FontPrograms`'
per-count programs, `program(n)`, the host `ink`/`sum2` tree, `blank()`, `UNWRITTEN`.
§3.4's structural range path and `<const N: usize>` (A5). `Param(u8)` stays: nothing in
compilation builds one, and its deletion is the-language-is-kernel D2's, with
`Vocabulary::Templates`, on that plan's schedule. `Fold`'s ends and `to_bits`' range
fields as a type (a `Lit` pair keeps their bit positions; corpus formats that store a
`Value` bound bump a version). For glyphs, the bound-buffer path (`with_buffer_data`,
`row_at`); `bind`, `BoundManifold` and `MAX_BOUND_BUFFERS` stay for the cell grid until
item 2.

**Made ordinary.** A per-call count is a value in `Scope::Body`; a triangular loop is a
fold whose bound varies with its parent; an empty glyph is an empty range. Unrolling a
known fold is unchanged; an unknown one is not offered, as LLVM does not fully unroll an
unknown trip count. Specializing on a value, if a profile asks, is an optimizer decision
keyed on cost (LLVM's function specialization, loop versioning), never a binding time.

**Cost and regalloc with an unknown trip count.** One constant,
`Fold::ESTIMATED_TRIPS: u64 = 32`, in `pixelflow-ir` with its provenance (LLVM's
`BranchProbabilityInfo` weights a back edge 124:4: 32 expected trips per level), read by
`CostModel::fold_cost` (`egraph/cost.rs:413`), `below` (`egraph/extract.rs:1676-1692`)
and `select.rs`' trips. Nested unknown folds multiply. A fold's own node says whether its
range is known; a value bound is priced by its class's chosen form (§3.2), which the
extractor holds before it prices the fold, so no price site needs more of the graph than
the picks. The estimate is policy: a carry
ranking at 32 may differ from one at the real count.

**Exactness until T6.** A count is an integer in a `Real` lane by convention. The guard's
`Eq(Floor(v), v)` checks it per call for a leaf, and `BoundArithmetic` keeps every bound a
leaf (§1.4). T6, with the bound retyping §5 adds to it, makes non-integrality
unrepresentable, deletes that conjunct and lifts `BoundArithmetic`; the `[0, 2²⁴]` check
stays until T6's measurement puts the counter in an integer register.

**Deferred: the lattice's own extent.** The collapse lattice's extent is a
`Variance::CONST` count whose binding time is still its syntax: it is in the program and
the key (`docs/DENOTATION.md:247-248`), so a resize is a recompile. By ruling 1 it could
be a value. The only reason in the code is `pack`'s strip-mining by lane count
(`pixelflow-ir/src/passes/lattice.rs:212-220`, a main fold strided by the lane count and a
remainder fold), which is a SIMD reason and so, by ruling 2, not a justification. This
plan offers no denotational reason to keep it. It is a consequence of ruling 1 that this
plan does not take, and `SEQ` keeps constant ranges (§1.4) until a plan does.

## 5. What this reverses and the docs it corrects

Each row is edited by the CL named, and only when its new text is true: A1 states the rule
(no bound is a function of a coordinate, true today since every bound is a constant) with
a pointer here; A4 states value bounds as built; B1/B2 state the extent off the key and
one program per font. "Hunk" names an open PR whose diff touches or abuts the lines.

| Where | Says | Becomes | CL | Hunk |
|---|---|---|---|---|
| `CLAUDE.md:175`, `pixelflow-core/README.md:63-65` | no iteration binder; counts unrolled at construction; a changing count recompiles | the fold is the one binder; bounds are values placed by variance, never a function of a coordinate; unrolling is extraction's, on known constants | A1 rule, A4 as built | none (#1431's is `CLAUDE.md:62-72`) |
| `docs/DENOTATION.md:54-58` | an extent "is part of the compile key"; "refused by name at bake" | an argument's extent is a value; refused at bind; the lattice's own extent stays in the key (§4, "Deferred") | B1 | none |
| `docs/DENOTATION.md:60-66`, `:173` | no data-dependent trip count; cost a closed-form function of static extents; "the budget decides" | no coordinate-dependent trip count; cost closed-form in values known before traversal; no step evaluates it per call, and the static model prices with an estimate (§1.5) | A1 rule, A4 cost | none |
| `docs/DENOTATION.md:123-125`, `:140-142` | "Loop bounds, extents and ranges are never uniforms"; "the range is a static natural" | §1.1–§1.2 | A4 | none |
| `docs/DENOTATION.md:246-248` | the lattice's extent is in the key | stands, with a pointer to §4's deferral | A1 | none |
| totality doc `:97-99` | "Extents are concrete literals known at `bake` … we monomorphize on them regardless" | extents are values known per call; nothing monomorphizes on them | B1 | none |
| totality doc, "One discipline, four faces" (the paragraph and its table) and consequence 5 | every iteration count is a static natural; "extent known at bake" | fixed before the loop it bounds starts, never a coordinate; a fold's bounds each a literal, a per-call value or an enclosing index, so its count is known per call, or once per trip of the enclosing fold whose index a bound is | A1 the rule; the paragraph, the reduction-binder row and consequence 5 A4 as built; the discrete-domain and typed-field rows B1 | none |
| totality doc `:191-200` | "only dynamic *bounds* die … static max + early-out … cost is the static worst case `N`" | a per-call bound is a value, not a static max; the cost model's estimate is §1.5's | A4 | none |
| `2026-10-15-kernels-evaluate-kernels.md:175` | "applying a discrete kernel over the wrong extent is a bake-time refusal by name" | a bind-time refusal | B1 | none |
| the-language-is-kernel `:78-81`, §1.4 structural row, §1.5, §1.7, §1.6 `:296-314` | ranges constant; one program per piece count (2026-10-09); the host chooses a program by its piece count | a reversal note pointing here | A1 | none |
| the-language-is-kernel §1.3 `:179` | `usize` is "a fold binder or a structural count" | a fold binder or a count (built from literals and `usize` consts with `+ - * /`, or an entry's per-call `n: usize`) | A5 (literals and `usize` consts), A6 (a per-call count) | none |
| the-language-is-kernel §1.3 `:186` | a count "is how many instances the host composed" | a count is a value | A5 (A1's, after #1431) | none |
| the-language-is-kernel D1 `:643`, D3 `:645`, D9 `:651` | structural binding time; "a count is how many instances the host composed"; "one per piece count" | a reversal note pointing here | A1, after #1323 | #1323 `:644-650` holds `:645`; `:643`, `:651` abut it |
| `2026-10-17-the-arena-is-typed.md` §1.1 `Reduce` row, §4 T6 | `Reduce`: `⊕ : S × S → S`, body `: S`; T6 types the binder only | `Reduce` with value bounds `Real × Real × S → S`, `Int × Int × S → S` at T6; T6 retypes the bounds with the binder, lifts `BoundArithmetic`, measured. A known bound is node data, so T6 re-keys no bound literal | A3 | none |
| BACKLOG plan table; `docs/README.md` "The language" | no row, no entry for this plan | a row ("landed — / next A0, A1"); an entry | A1 | none |
| BACKLOG `:32` (typed-arena note) | item 2 re-derived after T3 | also: T6 retypes value bounds (this plan, §5) | A3 | none |
| BACKLOG E6 (`:149`) | a loop and its unrolling tie | the tie narrows to known ranges | A3 | none |
| BACKLOG C8 (`:163`) | 2²⁴ refused at construction | §1.4: refused at construction for a literal, at the host write for a count, NaN in the language | A4 (NaN), A6 (host write) | none |
| BACKLOG row 38, H1 (`:101`) | bucketing done; one program per bucket | §4: deleted; one program per tile extent | B2 | none (#1323's is `:114-119`) |
| ontology `language.md`: "Binding time" `:116`, "Structural parameter" `:144`, "Range (trip count)" `:497` | ranges constant; structural a binding time; a range value "proposed and unbuilt" (`:509`) | §1.1–§1.2; the range value is decision 9 | A4; "Structural parameter" A5 | none |
| ontology "Fold (Reduce)" `:417` | "its body is its only child" (`:423`); "Is not: Dynamically bounded" (`:437`); "costs `len × body`" (`:450`) | bounds are values; §1.5's cost | A4 | none |
| ontology "Unrolling" `:523` | unrolling is construction's | extraction's, on known ranges (L6) | A3 | none |
| ontology "Totality" `:896` | "Ranges are constant, and uniforms are never extents" (`:905`) | §1.5 | A4 | none |
| ontology "Monoid (SEQ)" `:470` | "padding a table with identity rows leaves a fold unchanged" (`:481`) | true, and no longer used: nothing pads | B2 | none (#1431's are `:560-568`, `:586-598`, `:670-684`; #1323's `:248-297`) |
| code docs: `kernel.rs:709`, `sema.rs:1855`, `loop_blinn.rs:395-402`, `program.rs:1-20`, `fold.rs:1-22`, `egraph/node.rs:226-239`, `fold_rules.rs:600-606`, compiler `intent.md`, `pixelflow-ir/intent.md:65-66` | the old rule | corrected in the CL that changes the code under them | A2–B4 | #1431 `pixelflow-ir/intent.md:53-67` holds `:65-66`: A2b rebases |

## 6. The CL sequence

One PR each; the tree green; glyph cache, cell grid and packed frame render. "byte_probe"
is `cargo run -p pixelflow-codegen --example byte_probe`, parent against head, under
`PIXELFLOW_ISA=avx512` and `=avx2`, and NEON under qemu where an emitter changes.

**Collisions.** #1431 (T1) touches, in the IR, `fold.rs`, `arena.rs`, `kernel.rs`,
`key.rs`, `expr.rs`, `kind.rs`, a new lit.rs, `passes.rs`, `term.rs`, `term_arena.rs`,
`library.rs`, `lib.rs` and `intent.md`; in search, `fold_rules.rs`, `egraph/node.rs`,
`graph.rs`, `graph/columns.rs`, `insert.rs`, `extract.rs`, `shift.rs`, `template.rs`,
`witness.rs`, `math/algebra.rs`, `math/fusion.rs`, `math/trig.rs`, `nnue/mod.rs`; in
codegen, `select.rs`, `program/lower.rs`, `program/mod.rs`; the compiler's `emit.rs`,
`lower.rs`, `sema.rs`; pipeline corpus and training formats and three bins; graphics
`cell_grid.rs`, `packed.rs`, `test_support.rs`; and the docs `CLAUDE.md:62-72`,
`DENOTATION.md:443-452`, the-language-is-kernel `:175-181`, ontology `:560-568`, `:586-598` and
`:670-684`, `pixelflow-ir/intent.md:10-18` and `:53-67`, desloppify's
`pixelflow-architecture/SKILL.md:34-41`. #1323 touches compiler `emit`/`lower`/`parser`/
`sema` and the tests `items_block.rs` and a new at_is_contramap.rs; IR `arena.rs`/`kernel.rs`; and
the docs BACKLOG `:114-119`, the-language-is-kernel `:40-45`, `:164-172`, `:644-650`,
ontology `:248-254`, `:275-282`, `:290-297`. #1432 (T2) touches `kind.rs`, `passes.rs`,
`program/lower.rs`, both `intent.md`s, `cost.rs` (`:146`, `:494`) and `ops.rs`, and
shares files with A2–A4 but no hunk. #1427 rewrites the compiler's `ast`/`parser`/`sema`/
`symbol`/`lower`/`emit`, its `intent.md` and seventeen tests, among them
`rustc_is_the_oracle.rs`, `fold_is_kernel_over.rs`, `binding_times.rs`,
`kernel_typed_parameters.rs` and `items_block.rs`, and keeps the structural path A5 deletes
(made `pub(crate)`).

Order: A0 goes now. A1 goes now, except its the-language-is-kernel D-table rows, which
follow #1323 (§5's "Hunk" column). A2a goes against main with #1431 still open
(2026-10-22), so #1431 rebases onto A2a. A2a leaves the
`fold.monoid().identity()` lines alone; the emptiness guard above two of them and a few import
lines still conflict, each resolved by keeping both sides. A2b–A4 rebase onto #1431 once
merged; if it is abandoned, they proceed against main. #1432 is
"whichever lands second rebases". A5–A7 go after #1323 and #1427, with a cut-over: when A4
lands, A5 goes against main as it then stands, and any later #1427 commits rebase onto
A5's deletion, which removes code #1427 only re-scopes.

### (A) On main, independent of the typed-arena series

- **A0. Re-baseline production.** No code. Cold `GlyphAtlas::warm(' '..='~')` at tiles
  16 and 32, and `⌨`, through `Glyph::bake` (`pixelflow-pipeline/examples/glyph_compile_report.rs`,
  `glyph_phase_split.rs`, `frame_budget.rs`) with `--features saturation-telemetry`:
  programs, saturations, defs, bytes, compile and bake time, both x86 tiers, in a dated
  `docs/results/` note. Every later number is against it. **Done:**
  [`2026-10-20-production-glyph-baseline.md`](../results/2026-10-20-production-glyph-baseline.md).
  7 programs and 7 saturations per font and tile extent, at tiles 16 and 32; 201-202 defs per
  loop program; 16.9 KB of code over the seven; compile 45-56 ms, bake 0.4-1.0 ms; a new tile
  extent adds 7 programs and 0 saturations. `⌨` (DejaVu; Noto has none) is 262 pieces, 310
  defs, a 1.4 KiB frame.
- **A1. Docs.** §5's rows marked A1, under the order above; the docs/README.md entry and
  the BACKLOG plan-table row. Gate: `scripts/check-doc-paths.sh`. **Done** except
  the-language-is-kernel's D-table rows, after #1323; §1.3's "how many instances the host
  composed" sentence (§5's `:186`) was A5's, once #1431 had landed.
- **A2a. One reader of a fold's range.** A `KnownRange` value (`fold.rs`: `lo`, `hi`,
  `stride`) owns `len`, `is_empty`, `peel`, `peel_back`, `halve` and `admits`, and `Fold`
  loses them. One reader per representation that will hold bounds hands it out:
  `ExprArena::known_range(id)`; `ENode::known_range()`, per node and not per class, since
  after `HalveFold` one class holds two `Reduce` nodes with different ranges and §3.2 prices
  the node; and `ScheduledOp::Reduce`'s copy, made in `arena_to_schedule` where the id is in
  hand (§3.3's shape with `Known` only). Each returns `Option`, `Some` for every fold until
  A3. A site writes its unknown-range answer where today's code already gives it (a rule
  declines, an emptiness test is false), and expects `Some` elsewhere. Every reader that
  reasons about a range moves to them, inventoried 2026-10-22: IR `variance.rs`,
  `passes/lattice.rs` (`collapse_shape` returns the ids it now drops), `passes.rs` tests;
  search `cost.rs`, `extract.rs`, `fold_rules.rs`, `node.rs` (`ENode::fold` goes);
  codegen `program/lower.rs`, `program/mod.rs`, `emit/select.rs`; compiler `lower.rs`
  (`Holes::range_of`; its private `known_range(lo, hi)` is renamed) and `emit.rs`; the
  pipeline's `glyph_phase_split.rs`. `fold_cost` and `below` change signature, not price
  (§3.2). What serializes or names a fold rather than reasoning about one keeps reading the
  header until A2b moves `lo`/`hi` out of `Fold`: the derived `Hash`/`Eq`/`Ord`, `Display`,
  and `to_bits`/`from_bits` (`key.rs`, the macro's tokens, the graphics test dumps, the
  pipeline corpora and bins). No byte or key changes. Gate: byte_probe empty, keys
  unchanged, and no range read on a `Fold` outside `fold.rs`, the three readers, and tests
  that build a fold by hand with no node behind it (the compiler's `Holes` tests).
- **A2b. Bounds are values, constants only.** §3.1–§3.3's representation: `Bound` in the
  arena node, the e-graph node and the schedule; constructors take literal ranges, a
  `Value` bound panics by name. The diff stays inside the constructors, the walkers that
  take every child, `canonical`/`to_bits` and the grep-gated checklist; pricing does not
  change (§3.2). A preceding commit adds a byte_probe row with a literal equal to a bound
  after its fold (the mean `Σ_{i<n} f / n`), so the parent has it. Gate: byte_probe empty in
  both columns on every row, the mean row included; canonical keys and `KernelKey` digests
  unchanged. Anything that moves is argued in the PR, with the sibling-fold rows
  (`pixelflow-codegen/tests/support/sibling_rows.rs`) as byte-level arbiter. **Done:**
  byte_probe empty in both columns on every row under `PIXELFLOW_ISA=avx512` and `=avx2`, the
  mean rows included, and the 29 font programs' keys and digests unchanged. The
  bounds are `u64` and the stored header is packed (`FoldHeader` 32 bytes, three counts and
  four bytes, against 48 unpacked), so the nodes grew by what a `Reduce` holds and no more:
  `NodeData` and `ExprData` 16 to 32 and `ENode` 24 to 48, each pinned by a `size_of` test; `ExprNode` (a view) 24 to 56, held by a tripwire raised from 32 to 64 and by nothing
  tighter. The growth was measured the
  way `EClassId`'s was (`node.rs`: native release build, interleaved parent/head pairs, one
  process each, `ru_maxrss`): peak RSS median 89.3 to 104.8 MB (+17%) on the cold compile of
  the 29 programs ASCII needs (8 pairs) and 16.2 to 17.8 MB (+10%) on the n=40 program (10
  pairs); callgrind instructions on the 29 programs +5.2% (saturation +7.8%, extraction
  +3.4%). That is the size of growth that kept `EClassId` at `u32` (+18%, +8%). JP, 2026-10-22: land
  A2b wide, and measure the stored header narrowed to `u32` ends and stride (16 bytes) in a
  follow-up CL, with the decision taken on both numbers. **Narrowed** (JP, 2026-10-23, on
  the follow-up's numbers): the header stores its counts in 32 bits and its readers return
  64, `FoldHeader` 16 bytes, `ENode` 32, `NodeData`/`ExprData` 16; peak RSS on the 29
  programs 90.0 MB before A2b, 104.8 at 64-bit storage (+16.4%), 95.2 at 32-bit (+5.8%), 12
  rotating rounds of the three binaries; instructions +5.2% and +3.9%. The width does not
  bind while `Fold::EXACT_BOUND` is 2²⁴, and the key's fields are already 32 bits; T6's
  lifting of that cap widens both together. The reason is in `FoldHeader`'s doc, as
  CLAUDE.md's control-plane rule asks.
- **A3. Value bounds in the IR and e-graph.** `over_values` (`pub(crate)`), the
  `push_reduce` refusals (`BoundVaries`, `BoundSort`), `BoundOutOfDomain` and
  `BoundArithmetic` at `over_values`, L4 in both variance functions, L7, `KnownBound` (L8), the rules
  gated on known ranges, `FactorFold` factoring under a value bound (decision 5), the price of a
  value-bounded fold in `fold_cost` and `below` (its bound's chosen literal through the fold's
  literal twin, else `ESTIMATED_TRIPS`), and `below`'s bound operands. Codegen refuses a `Value`
  bound by name. Gate:
  byte_probe empty; pins for L2 (`over_values(5, 3)` is the identity, `Kernel::over(_, 5..3, _)`
  still refused), L3 (X, Y, own binder and an X-varying `Ref` refused; enclosing binder
  accepted), `ANY`/`ALL`/`SEQ` under a value bound refused, a non-leaf bound refused at
  `over_values`, L4, L7, L8 (a peel makes the inner bound known), a triangular fold
  whose outer fold `HalveFold` halves (the rule-made `0..(i + s)` accepted), and a rule-made
  bound whose class fact fails L3 (declined: the fold is not built). **Done:**
  byte_probe empty in both columns under `PIXELFLOW_ISA=avx512` and `=avx2`; the 29 font
  programs' keys, digests and saturation telemetry, the extraction fields included,
  unchanged on every pinned kernel. A placeholder's variance is `CONST` (it is a binder-to-be, never a coordinate,
  and L3 is checked again when `close_over` renames it into a slot), so a triangular fold
  built through the builder passes L3. That answer is not "closed": a placeholder can escape
  its builder, and closedness is judged by `free_index`, as `Kernel::by_ref` does and the
  runtime tier's unit walk does for a unit it resolves. A value bound is priced by its
  class's chosen literal through the fold's literal twin, and the key refused a value bound
  until A4 gave it a marker.
  The L8 and `HalveFold` pins saturate over the fold rules alone, with no `ConstantFold`
  and no production optimizer: the key then refused a value bound, so no production path
  could carry one, and without `ConstantFold` a rule-made `0 + 1` stays the arithmetic it is,
  which the triangular pin must allow. `Saturation::total_evals` rises on every kernel
  with a fold, because `KnownBound` is evaluated on every `Reduce`; no gate reads it, but
  the anytime curves and the Guide research bins do, so a moved curve is not a regression.
  A stop reason moves in one corner, and only as a label: a sweep checks the class budget
  before each rule's scan, so `KnownBound` adds a check after `FactorFold`, and when the
  last allowed iteration's `FactorFold` commits past the budget (its growth prediction can
  under-count) the run reports `class_cap` where it reported `iteration_ceiling`. The
  graph, the applications, the iterations and the extraction are the same, and no pinned
  kernel reaches the corner.
- **A4. Value bounds in codegen.** `Bound::Value`, the operand and carve change, the
  legalized guard, the post-collapse assert, `Trips` and its `Display`, `EmitTraffic`;
  `over_values` public. The key: a `Value` marker in `FoldHeader::to_bits`'s count field
  and its `from_bits` inverse. The serializers that write a header without its children
  refuse a value bound by name today and learn the marker:
  `pixelflow-graphics/src/render/cell_grid.rs`'s and
  `pixelflow-graphics/tests/production_glyph_arena_dump.rs:171`'s `dump_arena`,
  `pixelflow-graphics/src/test_support.rs`'s `program_digest`,
  `pixelflow-pipeline/src/collapse_bench/corpus.rs`,
  `pixelflow-pipeline/src/bin/corpus_gaps.rs`, and the compiler's `arena_to_tokens`; the
  training corpus bumps its format version. The guard handles a rule-made out-of-domain
  literal: `push_reduce` keeps one as a value bound (A3), so the guard, not a panic,
  decides it. Extraction picks one member per class, so a bound class whose chosen member
  reads a coordinate the class does not vary with is refused at materialization with
  `BoundVaries`, loudly, and A4 leaves it so: constraining a bound's pick would trade that
  failure for another (a class read as a bound with no member that keeps L3 would never
  settle, and every other user of the class would fail with it). Gate: byte_probe
  empty (no row has a value bound); new JIT tests in
  `pixelflow-codegen/tests/` against scalar Rust, every tier: a uniform in the domain at
  `hi` (0, 1, 7, 1000, `2²⁴`), at `lo` beside a literal `hi` (0, 1, 7, 1000 below 1003;
  `2²⁴ − 2` and `2²⁴` below `2²⁴`), and at both ends of a pair (1000 and 1003; `2²⁴ − 2` and
  `2²⁴`; `2²⁴` twice, and one uniform at both ends); a uniform outside it (`2²⁴ + 2`, NaN,
  2.5, −1, ±∞, 10³⁰: NaN, no hang) at `hi`, at `lo` beside a literal `hi`, and at either end
  of a pair; uniforms 5 and 3 as `lo` and `hi` (the identity); a triangular
  `Σ_i Σ_{j<i}`; a uniform-bounded fold whose body reads no coordinate, so the guarded fold
  is hoisted to `Scope::Body` and the carve sees the `If`, not the `Reduce`, as its def (the
  single-reader assumption under placement), for `n` in the domain and for NaN, `+∞` and
  `2²⁴ + 2`; a uniform-bounded fold that reads `Y` alone, in the row fold, for the same
  kinds of `n`; a uniform-bounded fold whose body varies with X and whose `hi` is read only
  by the loop (§3.3's placement rule); one program and one code pointer for every `n`.
  `2²⁴ + 1` is not a case here: no `f32` names it (§1.4). **Done:** byte_probe empty in both
  columns under `PIXELFLOW_ISA=avx512` and `=avx2`; the 29 font programs' keys, digests and
  saturation telemetry, the extraction fields included, unchanged. The guard is
  `pixelflow_ir::passes::legalize`'s last pass (`passes/guard.rs`); lowering asserts after
  collapse that a value bound reads no lattice binder and that a value-bounded fold is read
  only as its guard's true arm, and refuses a value bound the emitter holds as a mask by
  name, at either end and on every tier (a mask names no count); `FoldHeader::new` already
  refuses a stride other than 1 beside a value, so no stride assert is added. A value bound
  is an operand of its fold: `lo` seeds the loop and `hi` is read at the latch, live there
  by the allocator's own rule.
  `Trips::{Known, Estimated}` each carry a number, an estimate printed `~n`, and
  `EmitTraffic::dynamic_memory_ops` is `None` over an estimate. The key's marker is a count
  field of all ones; the collapse corpus is v5 and the training corpus's schema names the
  marker; the arena dumps stay v1, because no reader parses an `R` line. The JIT tests are
  `pixelflow-codegen/tests/value_bounds_jit.rs`; NEON runs in CI's macOS job only. The
  extraction decision is the one above: loud, with `BoundVaries` pinned in `fold_rules.rs`
  for a class with no lawful member and for one whose cheapest member reads a coordinate
  beside a dearer lawful one.
- **A5. `<const N: usize>` is deleted**, and with it §3.4's structural path, which it was
  the only producer of. No production user: tests (`kernel_typed_parameters.rs`,
  `rustc_is_the_oracle.rs`, `items_block.rs`, `binding_times.rs`, and the `parser.rs`,
  `sema.rs`, `lower.rs` and `emit.rs` unit tests) and docs rewritten. It comes before A6
  because the obvious lowering of a structural bound, a `Param` leaf, is `Variance::ALL`
  (`variance.rs`'s `Param` arm) and L3 refuses it.
  **Done:** `kernel!` refuses a generic on any `fn` at parse, a `const` one naming §1.2; the
  structural path is deleted (§3.4), and `FoldTerms` lost its stride, which only the template
  set past 1; the template exemption went with it, so A7 is empty (below). The test templates
  that take no kernel (`ramp`, `rings`, `swept`, `mean_index`, and `emit.rs`'s
  `counted`) are written over `usize` consts and are offered to the macro tier at expansion;
  `emit.rs`'s `inside`, whose count nothing read, lost its generic; the two that take a
  kernel (`applied`, `mean_down`) stay `Staged`; the tests whose subject was the template are
  deleted. byte_probe empty in both columns under `PIXELFLOW_ISA=avx512` and
  `=avx2`; the 29 font programs' keys, digests and saturation telemetry, the extraction fields
  included, unchanged (gates compared to the base verified at main 7e9531e).
- **A6. `kernel!` bounds are typed expressions** (§3.4). Decision 2's spelling, an
  `n: usize` entry parameter; decision 1 is B5's. Gate: `tests/rustc_is_the_oracle.rs`
  gains a triangular case and a per-call count case (`fn rust_sum(n: usize, x: f32)`, in
  decision 2's spelling); the `Args` write refuses `2²⁴ + 1`;
  `tests/fold_is_kernel_over.rs` green; byte_probe empty.
  **Done:** a bound's names are typed where the fold is written: a literal or a `usize`
  const (`+ - * /` over them), an enclosing fold's index (a triangular `0..i`), or an
  entry parameter `n: usize` of the items form, a uniform declared in parameter order as
  `n as f32` and read as a bound or by `n as f32`; one program for every `n`. Refused at
  expansion: `X` or `Y` (L3), any other name that is not a `usize` (by type), arithmetic
  on a bound that is a value (`BoundArithmetic`, until T6), a value bound under `any` or
  `all` (decision 8), a known end past 2²⁴, and a `usize` parameter of a helper or of the
  closure form. The host function panics on a count past 2²⁴, and `Args::write_into`
  panics on one before writing anything; a raw block write that names no count makes the
  fold NaN by the guard, up to the optimizer's algebraic rules, which are not sound at NaN
  (BACKLOG E9). `rustc_is_the_oracle.rs` has the
  triangular and per-call cases, and `fold_is_kernel_over.rs` the same two against
  `over_values`. byte_probe empty in both columns under `PIXELFLOW_ISA=avx512` and
  `=avx2`; the 29 font programs' keys, digests and saturation telemetry, the extraction
  fields included, unchanged.
- **A7. Folded into A5.** The exemption (`emit.rs:154-156` at `7e9531e`) was
  `if !entry.structural.is_empty()`, a condition on the field A5 deletes, so A5 deleted it.
  It exempted only test templates, which A5 rewrote over `usize` consts, stripped of a
  generic nothing read (`emit.rs`'s `inside`), or deleted with the tests whose subject they
  were, and no production entry, so there was no production byte change to measure.

A alone does not deliver ruling 6 on the production path: the piece table's extent is
still in the key (§3.5). Ruling 6 lands with B2; decision 4's interim is refused (§8).

### (B) After T3 and item 2

- **B0.** T3 lands (the typed-arena series). T6 and the bound retyping follow on that
  series' schedule and delete §4's integrality conjunct and `BoundArithmetic`; nothing in
  B needs them, since B3's bounds are leaves.
- **B1. Item 2, re-derived against this note** (decision 3), and §1.3's denotation built:
  the binder `i : L`, its bounds `0` and `Extent(p)` (or decision 9's range node), lifts
  named. An `L → T` binder supplied per call; its extent a value off the key; `bind` checks
  the runtime extent and refuses one past 2²⁴; `lower_gather` stops baking extents and its
  implicit clamp gives way to named lifts. §5's B1 rows.
- **B2. A glyph is a call.** `glyph(pieces, box)`, one program per tile extent; the
  bucketing, padding and `row_at` indirection deleted. Gate: `tests/freetype_oracle.rs` and
  `glyph_atlas_golden.rs` within tolerance; bit-exact pins (`kernel_glyph_optimize.rs`,
  `glyph_optimization_cost.rs`) re-baselined in their own commit; A0's numbers after,
  with **1** program per tile extent. §5's B2 rows.
- **B3. A run is a call.** `spans`, the outer fold over `G` and the inner fold over
  `Lattice(count(g))` reading `pieces ∘ ι_g` (§1.3); `Glyph::over` concatenates tables
  host-side with `Glyph::empty()` its unit (`tests/run_is_a_glyph.rs`). Gate: B2's, plus
  one program for strings of different lengths, and a malformed span reads NaN. L3's walk
  resolves each `Ref` a bound reaches through the referent's whole arena, on every push; no
  production bound holds a `Ref` before this step. A referent's variance belongs to the
  referent, so memoize it per referent when B3's profile asks.
- **B4. `FontPrograms` goes** (decision 7); its tests move to B2's program.
- **B5. The `kernel!` surface for a fold over a discrete kernel** (decision 1); the glyph
  is then written in `kernel!` again. Depends on O3 (the-language-is-kernel `:215-218`):
  an entry that takes a kernel has no `Args` record and re-lowers per host call, which
  would be one lowering per glyph, the opposite of ruling 6.

## 7. What this is not

- Not a coordinate-dependent loop: a count varying with X, Y or a lattice binder is
  refused, permanently.
- Not yet a change to the lattice's own folds (`collapse`, `pack`, `Monoid::SEQ`,
  `Write`'s lane count): they keep constant ranges and the lattice's extent stays in the
  key. That is deferred, not argued (§4, "Deferred").
- Not specialization or versioning: nothing compiles a second program for a value.
- Not a SIMD argument: every lane runs the same loop because of the rule, not for it.
- Not predicate domains, and not an array: indexing is application.
- Not a change to the piece derivation, or to the `fn(ctx, out, pitch)` ABI: a count
  is a uniform slot.

## 8. Decisions for JP

1. **`kernel!` surface for a fold over a discrete kernel** (B5). S1,
   `pieces.map(|p| piece_term(p, X, Y)).sum()`: in-domain by construction, no index in the
   surface, no spelling for B3's window. S2, `(0..pieces.len()).map(|i| piece_term(pieces(i), X, Y)).sum()`:
   your wording, but `0..pieces.len() + 5` is writable and needs a lift for the outside.
2. **Decided (JP, 2026-10-22): an `n: usize` entry parameter.** How `kernel!` spells a per-call count (A6). Ruling 1 already admits a
   `Variance::CONST` bound, so a per-call count is legal; what is open is its surface, for
   example a `n: usize` entry parameter. It replaces `<const N>`, which A5 deletes either way.
3. **Item 2: taken by this plan (B1), or waited for** from the typed-arena series. (#1431, T1,
   and #1432, T2, landed 2026-10-22 on JP's word; item 2 itself is still open.)
4. **Refused: a production interim before item 2** (a table at a fixed capacity, padded
   with data). A capacity is a count in `BufferDecl` and the key (`key.rs:415-424`), which
   is the JIT-compile-time count ruling 6 rules out and the "static max" the totality doc
   held (`:196-200`) that this plan reverses. Ruling 6 lands with B2.
5. **Decided (JP, 2026-10-22): accept the edge.** `FactorFold` when non-emptiness is unknown: decline (loses `c·Σf` for a per-call
   `n`), or accept the infinite-`c`/`−∞` edge its doc declines (`fold_rules.rs:346-349`),
   under "edge-case IEEE conformance is not on offer".
6. **The pieces' codomain.** Ten channels as one `L × Fin 10 → Real` table (today's shape,
   in the grammar), or a record codomain, which needs an extension to `T ::= S | D → T`.
7. **Decided (JP, 2026-10-22): approved, all of it.** Public API. Deleting `FontPrograms`, `GlyphRows` and `blank`, changing
   `loop_blinn::run` and `Glyph::bake`, and adding `Kernel::over_values` (public from A4):
   this plan reads ruling 6 as permission for the first two; CLAUDE.md asks for it to be said.
8. **A mask-sorted fold out of domain.** `ANY` and `ALL` (and `Int` folds after T6) have no
   NaN. Either name their out-of-domain value and argue it, or keep them refused with a
   value bound (§1.4), as this plan does until you decide.
9. **The index set's representation** (B1). An `Extent(p)` leaf as each fold's upper
   bound, or a range node naming `L`'s index set that sibling folds share (a-glyph-is-a-formula
   §4.2). Same denotation; the second makes sibling folds over one table one loop by
   hash-consing.
