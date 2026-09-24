# Spike: `ENode` constant representation for the dyadic-exact fold

**Status:** Reconnaissance only — no source changes. Recommendation for whoever implements.
**Date:** 2026-08-08
**Scope:** `pixelflow-search` (primary), `pixelflow-compiler`, `pixelflow-ir` (read-only —
`dyadic.rs`/`fold_exactness.rs`/`arena.rs` were under concurrent edit by other agents during
this spike and were not modified here).

---

## TL;DR recommendation

Split `ENode::Const(u32)` into **two variants**, `ENode::Num(Dyadic)` and `ENode::Bits(u32)`
(option **a** in the brief). Keep `ENode::constant(val: f32) -> Self` as the single, unchanged,
f32-taking constructor everyone already calls — it auto-classifies via `Dyadic::from_f32`
(`Some` → `Num`, `None` for NaN/±inf → `Bits`). Keep `ENode::as_f32(&self) -> Option<f32>`
with its current signature and behavior for `Num` (now `Dyadic::to_f32`, one correctly-rounded
conversion) and `Bits` (unchanged bit-reinterpretation).

That combination means **the great majority of the census below needs zero or
compiler-mechanical changes.** Of the 72 real, live call sites found:

- **~46 are pure construction** (`ENode::constant(f32)` / `ENode::Op{..}` builders) — untouched,
  because the constructor signature doesn't change.
- **15 are shape-only matches** (`ENode::Const(_)` wildcard, "is this a constant leaf") — these
  need a second match arm (`Num(_) | Bits(_)`), which `rustc` finds for you: `ENode` is not
  `#[non_exhaustive]`, so every one of these is a compile error until fixed, not a silent bug.
- **~9 read `.as_f32()`/`.is_const()`** — zero caller-visible change, because those methods'
  signatures don't change.
- **A genuinely small set — about 10 sites — need real design thought**: the two hash-consing
  impls in `node.rs` (this is the whole point of the migration), the `ConstantFold` rewrite's
  arg-gather and result-construction in `math/algebra.rs`, the freshly-added `const_fact` side
  table in `graph.rs` (7 sub-edits, see below), and 6 sites that pattern-match `ENode::Const(bits)`
  with a **bound** variable and must decide what to do with each variant.

Full reasoning, the rejected alternative (a single fused variant), and an ordered
implementation sketch are below.

---

## 1. Measured site census

**Raw grep count** (`ENode::Const|ENode::constant\(|\.as_f32\(|\.is_const\(`, all four patterns,
`--type rust`, whole workspace) as of this spike: **76 matches in 17 files.** The brief's
estimate ("~71 in ~17 files") was close but not exact; here is the reconciliation:

| Adjustment | Count | Why |
|---|---|---|
| Raw matches | 76 | `rg` count, all 4 patterns, 17 files |
| − dead code | −2 | `pixelflow-search/src/egraph/algebra.rs` is **not declared as a module** anywhere (`egraph/mod.rs` has no `mod algebra;`/`pub mod algebra;`) — it does not compile into the crate. It duplicates `InverseAnnihilation<T>` byte-for-byte from `math/algebra.rs`. See §3. |
| − false positives | −2 | `nnue/factored.rs` (`v.is_const()`) and `egraph/deps.rs` (`child_v.is_const()`; deleted 2026-09-23) both call **`Variance::is_const()`** (`pixelflow-ir/src/variance.rs`), an unrelated method on an unrelated type. Not `ENode` at all. |
| **Live, real total** | **72** | across **16 compiled files** |

`pixelflow-pipeline` was checked and has **zero** references to `ENode` — it operates one
layer up, on `pixelflow_ir::ExprArena`/`ExprNode` (plain `f32`), which this change does not
touch. Out of scope, confirmed by grep, not asserted.

### Per-file counts (live, compiled files only)

| File | Raw sites | Construction | Read (bound) | Read (shape-only wildcard) | Notes |
|---|---:|---:|---:|---:|---|
| `pixelflow-search/src/egraph/node.rs` | 6 | 1 (ctor def) | 3 (`as_f32`, `is_const`, + PartialEq/Hash below) | 1 | **Defines the type + hash-consing.** See §2.1. |
| `pixelflow-search/src/egraph/graph.rs` | 25 | 21 (incl. `add_arena` widen, derivative helpers, tests) | 2 (`as_f32` in `add()`'s `const_fact` hook, `is_const` in `any_const_eq`) | 2 | Also owns the **union refusal valve** and new `const_fact` table — see §2.2. |
| `pixelflow-search/src/egraph/extract.rs` | 8 | 1 (test) | 1 (`choices_to_arena` narrowing — **the** extraction funnel) | 6 (incl. new `pin_shift_counts`, see §2.4) | |
| `pixelflow-search/src/egraph/deps.rs` | 8 (7 real) | 5 (tests) | 0 | 2 | Variance classification only; never reads the value. Deleted 2026-09-23 — the class variance fact replaced it. |
| `pixelflow-search/src/math/algebra.rs` | 7 | 5 (incl. `ConstantFold` result) | 2 (`ConstantFold` arg-gather, 1 test) | 0 | **`ConstantFold::apply` — the crux.** See §2.3. |
| `pixelflow-compiler/src/optimize.rs` | 4 | 3 | 1 (egraph→AST codegen) | 0 | Cross-crate: `ENode` is `pub`, matched from outside `pixelflow-search`. |
| `pixelflow-search/src/nnue/factored.rs` | 3 (2 real) | 0 | 0 | 2 | Shape/`OpKind` classification only — **no numeric value ever reaches NNUE features.** Reassuring finding, see §3. |
| `pixelflow-search/src/math/trig.rs` | 2 | 2 | 0 | 0 | Pythagorean identity → literal `1.0`. |
| `pixelflow-search/src/math/power.rs` | 2 | 1 | 1 (`eclass_const`, epsilon-compared) | 0 | See §3 (epsilon-vs-exact risk). |
| `pixelflow-search/src/egraph/saturate.rs` | 2 | 2 (tests) | 0 | 0 | |
| `pixelflow-search/src/math/mod.rs` | 2 | 1 (test-local widen) | 1 (test-local narrow) | 0 | **Duplicates the arena-boundary logic** in test code instead of calling `add_arena`/`choices_to_arena`. |
| `pixelflow-search/src/math/fusion.rs` | 1 | 0 | 1 (`is_one`, epsilon-compared, manually unpacks bits instead of using `.as_f32()`) | 0 | |
| `pixelflow-search/src/math/pict_rewrite_tests.rs` | 1 | 1 (test-local widen, same duplication as above) | 0 | 0 | |
| `pixelflow-search/src/egraph/codegen.rs` | 1 | 0 | 1 (`format_const` — string-codegen numeric literal emission) | 0 | |
| `pixelflow-search/src/egraph/cost.rs` | 1 | 0 | 0 | 1 | |
| `pixelflow-search/src/runtime.rs` | 1 | 1 | 0 | 0 | **Second, independent** arena→egraph widening implementation (memoized), parallel to `add_arena`. Must be kept in sync by hand — pre-existing risk, not created by this change. |
| **`pixelflow-search/src/egraph/algebra.rs`** | 2 | — | — | — | **DEAD CODE**, not compiled. Excluded from all totals above. Flagging for cleanup (separate task). |

Total live real sites: 1+21+2+1(node.rs shape)... — see the "Live, real total: 72" figure above;
the table's per-file breakdown sums to it (72 across the 16 compiled files, i.e. 76 raw − 2 dead
− 2 false-positive).

**Caveat on line numbers:** two other agents were actively editing `pixelflow-ir/src/arena.rs`,
`pixelflow-search/src/egraph/extract.rs`, `pixelflow-search/src/egraph/graph.rs`, and
`pixelflow-search/tests/fold_exactness.rs` during this spike (a real, unrelated correctness fix —
see §2.2). Line numbers cited below were re-verified against the working tree at time of writing
but **will drift further**; re-grep before acting. `git diff` those four files first to see
what already landed.

---

## 2. Deep dives on the sites that actually matter

### 2.1 `node.rs` — the type definition and hash-consing (the whole point)

```rust
pub enum ENode {
    Var(u8),
    Const(u32),              // stored as f32 bits
    Buffer(BufferDecl),
    Op { op: &'static dyn Op, children: Vec<EClassId> },
}

pub fn constant(val: f32) -> Self { ENode::Const(val.to_bits()) }      // ctor
pub fn as_f32(&self) -> Option<f32> { ... f32::from_bits(*bits) ... } // numeric read
pub fn is_const(&self, val: f32) -> bool { self.as_f32() == Some(val) }

impl PartialEq for ENode {
    (ENode::Const(a), ENode::Const(b)) => a == b,   // bit-exact — the hash-consing key
}
impl Hash for ENode {
    ENode::Const(bits) => { 1u8.hash(state); bits.hash(state); }
}
```

This `PartialEq`/`Hash` pair, together with `EGraph`'s `memo: HashMap<ENode, EClassId>`
(`graph.rs`), **is** hash-consing: two constants are the same e-node today iff their f32 bits
are identical. This is exactly what constraint 1 in the brief breaks — a dyadic fold result with
no f32 form has no bits to hash. This is the site the whole migration exists to fix; everything
else downstream is bookkeeping.

`ENode` is `pub`, not `#[non_exhaustive]`, and re-exported (`egraph/mod.rs:52`,
`pub use node::{EClassId, ENode};`) — and it is pattern-matched **outside the crate**, in
`pixelflow-compiler/src/optimize.rs` (imports `pixelflow_search::egraph::ENode` directly, line
~30). Any variant-shape change is a breaking change for that crate too, not just internal
`pixelflow-search` code — already counted in its 4 sites above.

### 2.2 `graph.rs` — the union refusal valve, and a just-landed `const_fact` table

The refusal valve the brief calls out (`EGraph::union`) is real and load-bearing. **While this
spike was in progress, another agent fixed a genuine bug in it** (visible via `git diff
pixelflow-search/src/egraph/graph.rs` at time of writing): the old implementation scanned
`EClass::nodes` to find a class's constant, but `rebuild()` drains `nodes` with `mem::take`
*before* performing congruence unions — so a node-scanning guard was blind at exactly the moment
congruence closure does its merging, and a contradictory merge could slip through during
rebuild. The fix adds a class-indexed side table:

```rust
/// The constant each class is known to equal, as f32 bits, indexed by class id —
/// maintained independently of `EClass::nodes` on purpose. [...] The fact must
/// outlive the nodes.
const_fact: Vec<Option<u32>>,
```

populated in `add()` (`self.const_fact.push(node.as_f32().map(f32::to_bits));`) and consulted —
not the node vector — in `union()`. **This table is a new, permanent piece of state that must
also become domain-aware.** Under the recommended design it becomes
`Vec<Option<ConstFact>>` where `ConstFact` mirrors the `Num`/`Bits` split (or two parallel
`Vec<Option<Dyadic>>` / `Vec<Option<u32>>`), touching: the field declaration, `Clone` impl,
both `new()`/`with_rules()` constructors, `add()`, and `union()`'s comparison — 7 mechanical
edits, all in one file, none individually hard.

For the refusal valve itself, the recommended semantics (see §5) are:
- Two `Num` facts compare via `Dyadic`'s derived, normalized `Eq` — exact, no more `ca != cb &&
  ca.to_bits() != cb.to_bits()` double-check needed (that dance existed only to admit ±0.0 and
  identical-NaN-bits under `f32`; `Dyadic` has one zero and no NaN by construction, so the
  double-check collapses to a single `!=`).
- Two `Bits` facts compare via raw bit equality — unchanged from today (still admits identical
  NaN/mask patterns, which is correct — an all-ones comparison mask must be unionable with
  itself).
- A `Num` fact and a `Bits` fact are **never** unioned — see §4 ("mixed reading") — refuse and
  journal it the same way, which doubles as a new diagnostic for a rewrite rule that
  accidentally crosses domains (arguably a bug worth catching, not just tolerating).

`canonical_op` (`graph.rs`, `ENode::Const(_) => pixelflow_ir::OpKind::Const`) is shape-only —
both new variants still map to `OpKind::Const`, no behavior change, no ambiguity.

### 2.3 `math/algebra.rs` — `ConstantFold::apply`: the site where exactness is actually decided

```rust
fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
    let ENode::Op { op, children } = node else { return None };
    let kind = op.kind();
    let mut args = Vec::with_capacity(children.len());
    for &child_id in children {
        let mut found_const = None;
        for child_node in egraph.nodes(child_id) {
            if let Some(val) = child_node.as_f32() { found_const = Some(val); break; }
        }
        args.push(found_const?);
    }
    if kind.fold_is_platform_specific(&args) { return None; }
    let result = match args.len() {
        1 => kind.eval_unary(args[0])?,
        2 => kind.eval_binary(args[0], args[1])?,
        3 => kind.eval_ternary(args[0], args[1], args[2])?,
        _ => return None,
    };
    if !result.is_finite() && !kind.is_bitwise_domain() { return None; }
    Some(RewriteAction::Create(ENode::constant(result)))
}
```

`OpKind::eval_unary/binary/ternary` (`pixelflow-ir/src/kind.rs`) already dispatch two
categorically different kinds of computation through the *same* `f32`-typed function:
arithmetic (`Add`/`Sub`/`Mul`/`Div`/`Min`/`Max`/...) and bit-domain ops
(`Lt/Le/Gt/Ge/Eq/Ne/Select/BitAnd/BitOr/TruncToInt/IntToFloat/IAdd/Shl/Shr`, exactly
`OpKind::is_bitwise_domain()`'s membership). The bit-domain arms already operate via
`.to_bits()`/`f32::from_bits()` internally — i.e. they are **already exact**, because bit
reinterpretation through `f32` never rounds. That is a reassuring finding: **the bit domain does
not need new arithmetic machinery in `pixelflow-ir::OpKind` at all**, as long as
`ENode::as_f32()` keeps returning the exact reinterpreted value for `Bits` — which it does under
the recommended design (`Bits(b) => Some(f32::from_bits(b))`, unchanged).

What genuinely needs new work, and is **not** just a representation change:
1. **Result routing** (line ~964, `Some(RewriteAction::Create(ENode::constant(result)))`): must
   become `if kind.is_bitwise_domain() { ENode::Bits(result.to_bits()) } else { /* dyadic path,
   see below */ }`. This alone (still computing `result: f32` exactly as today) already satisfies
   constraint 2 — a mask can never again be hash-consed as if it were a number.
2. **Exactness** (constraint 1) requires the non-bitwise branch to stop computing `result` via
   f32 arithmetic and instead gather `Dyadic` args (a new accessor, `as_dyadic(&self) ->
   Option<Dyadic>`, succeeding only on `Num`, refusing on `Bits` — correctly declining to do
   arithmetic on a bit pattern) and evaluate via new dyadic-native counterparts to
   `eval_unary/binary/ternary` in `pixelflow-ir::OpKind` (not present today — `Dyadic::add/
   sub/mul/try_div/try_sqrt` exist in `dyadic.rs` as of this spike but are all still `todo!()`,
   being implemented by a concurrent agent). **This second piece is genuinely out of this
   spike's scope** (it's fold-algorithm work, not `ENode`-representation work) but is the
   follow-up that actually buys the "no more f32-vs-algebra contradiction" property the brief's
   background section motivates. Recommend sequencing it as a separate phase (see §6).

The two duplicate `InverseAnnihilation<T>` sites (`math/algebra.rs` lines ~303/314, mirrored in
the dead `egraph/algebra.rs`) and the `Annihilator` site (line ~707) construct
`ENode::constant(T::identity())` where `identity()` is always `0.0` or `1.0` — always exactly
dyadic-representable, zero risk.

### 2.4 The arena boundary: two widen implementations, one narrow funnel, plus a live example of why `Const` shape-matching matters independent of value

**Widen** (`ExprNode::Const(f32)` from `pixelflow_ir::ExprArena` → `ENode`): there are **two
independent, hand-synchronized implementations**:
- `EGraph::add_arena` (`graph.rs`, production, non-memoized): `ExprNode::Const(v) => self.add(ENode::constant(*v))`.
- A second one in `pixelflow-search/src/runtime.rs` (memoized, used on a different call path):
  `&ExprNode::Const(val) => { let class = egraph.add(ENode::constant(val)); ... }`.

Both are pure construction, unaffected by the representation change if `ENode::constant(f32)`
keeps its signature — **but the fact that two implementations exist at all is a standing
maintenance risk** independent of this migration (worth flagging separately; not blocking).

Two more **test-local** reimplementations of the same widening logic exist:
`math/mod.rs`'s `expr_to_egraph` (`ExprNode::Const(val) => egraph.add(ENode::Const(val.to_bits()))`
— note: hand-rolled, doesn't even call `ENode::constant`) and
`math/pict_rewrite_tests.rs`'s equivalent. These need the same one-line fix and would ideally be
deleted in favor of calling the real `add_arena`, but that's a pre-existing duplication, not
something this migration must fix.

**Narrow** (`ENode` → `ExprArena`'s `ExprNode::Const(f32)`, at extraction): there is exactly
**one** production funnel, `choices_to_arena` (`extract.rs`, called by both `extract_dag` and
`extract_neural_to_arena`):

```rust
ENode::Const(bits) => {
    let expr_id = arena.push_const(f32::from_bits(*bits));
    ...
}
```

Under the new design: `Num(d) => arena.push_const(d.to_f32())` (the one designed, correctly-
rounded conversion — this is where dyadic exactness legitimately ends, because `ExprArena` is
and remains f32-only; that's fine, it's the documented contract of `Dyadic::to_f32`, not a
regression) and `Bits(b) => arena.push_const(f32::from_bits(b))` (unchanged). One function to
touch, well-isolated. `pixelflow-search/src/math/mod.rs`'s test-local `eclass_to_arena` duplicates
this narrowing too and needs the parallel fix.

**A live example of why the `Const` shape matters independent of its value** surfaced during
this spike: `extract.rs` gained a new function, `pin_shift_counts`, while this spike was in
progress (`git diff` shows it was added concurrently). It re-pins every `Shl`/`Shr` shift-count
child's extraction choice to a node that `matches!(n, ENode::Const(_))`, because the emitter
requires shift counts to be hardware immediates and a cost model could otherwise pick an
uncollapsed `Add` in the same e-class. This is a **shape-only** read (doesn't care whether the
constant is `Num` or `Bits` — a shift count is, semantically, closer to `Bits`/an integer, but
the pinning logic itself never inspects the value) — it costs one extra match arm under option
(a) (already counted in the 15 shape-only sites) and nothing under a design that keeps a single
outer `Const`-shaped pattern.

---

## 3. Risks found

1. **Result-routing correctness in `ConstantFold` is the load-bearing change; everything else is
   plumbing.** Get `is_bitwise_domain()`-based routing right at the one result-construction site
   (§2.3) and constraint 2 (no mask-as-number corruption) is satisfied immediately, independent
   of whether the dyadic-exact arithmetic follow-up ever lands. Get it wrong and this migration
   doesn't fix the bug it exists to fix.

2. **Epsilon-tolerant numeric matching against exact special values.** Two read sites compare a
   constant's `f32` value against a target with a fixed tolerance rather than exact equality:
   `math/power.rs`'s `eclass_const`/`const_eq` (`EPSILON = 1e-6`, matching `pow(x, 0/1/2/0.5/-1/
   -0.5)`) and `math/fusion.rs`'s `is_one` (`1e-10`, matching `recip(sqrt(x))` → `rsqrt(x)` only
   when the exponent is "close enough" to 1). All of the target values here (0, 1, 2, 0.5, -1,
   -0.5) are exactly dyadic *and* exactly f32-representable, so switching these two sites to
   exact `Dyadic` comparison would be strictly more correct and never break a currently-passing
   case — but it is a **behavior change** (a value that today matches "close enough" would stop
   matching if it isn't exactly one of those constants), so flag it for the implementer to decide
   deliberately rather than fold it in silently as part of "just changing the representation."

3. **"Mixed reading" is a real, not hypothetical, design question**, not just a hedge in the
   brief. See §4 — recommend resolving it explicitly (never unify) rather than leaving it
   implicit, because the natural implementation of option (b) resolves it *wrong* by default
   (silently unifies same-bit-pattern values regardless of domain provenance) unless it
   re-invents a domain tag, at which point it has stopped being simpler than option (a).

4. **The `const_fact` side table (§2.2) is new, not something this spike introduced, but it must
   not be forgotten** — it's easy to design the `Num`/`Bits` split against `node.rs` and
   `math/algebra.rs` and miss that `graph.rs` now carries a *second* place that stores "this
   class's constant, as bits" and must be updated in lockstep or the refusal valve regresses to
   the bug that was just fixed.

5. **Two hand-synchronized arena-widening implementations** (`graph.rs::add_arena` and
   `runtime.rs`'s memoized version) and **two hand-synchronized narrowing implementations in test
   code** (`math/mod.rs`, `math/pict_rewrite_tests.rs`, duplicating `extract.rs::choices_to_arena`)
   mean the same one-line fix must be applied in 4 places by hand, with no compiler help catching
   a missed one (they're free functions with their own `match`, not shared code) — **except**
   that the shape-only-match-forces-a-compile-error property (§2.1) does still apply to each of
   these individually, so a missed *arm* fails to compile; a missed *file* just silently keeps
   the old f32-bits behavior in a code path this spike didn't find. Grep for
   `ENode::Const` and `ExprNode::Const` together one more time immediately before landing the
   change, since new call sites may have appeared (four files changed under the census's feet
   during this very spike).

6. **NNUE feature extraction is unaffected — verified, not assumed.** `pixelflow-search/src/nnue/
   factored.rs`'s two real `ENode::Const` sites are both shape/`OpKind` classification
   (`ENode::Const(_) => OpKind::Const`); no numeric constant value ever flows into
   `EdgeAccumulator`/`GraphAccumulator` features at the `ENode` level. (`nnue/mod.rs` does
   epsilon-compare constants, but operates entirely on `pixelflow_ir::ExprNode`/`ExprArena`, a
   layer up, untouched by this change.) This means training/inference correctness is not at risk
   from this migration — a genuine relief given the file's size and centrality.

---

## 4. The "mixed reading" question, resolved

A dyadic `Num` and a `Bits` pattern can, for a finite value, decode to numerically identical
`f32` — e.g. `Num(Dyadic::from_f32(1.0))` and `Bits(0x3F800000)` both read as `1.0f32`. Should
they be the same e-node?

**No.** Recommend: never unify across domains, by construction (different variant ⇒ different
hash-cons key ⇒ never memo-collide) and by policy (the union refusal valve should refuse, and
journal, an explicit attempt to union a `Num` class with a `Bits` class, the same way it already
refuses two unequal `Num`s). Reasoning:

- This is exactly constraint 2's contract, extended: "a mask is not a number" must hold even
  when the mask's bit pattern happens to coincide with some number's bits. If arithmetic
  rewrites (associativity, distributivity, `ConstantFold`'s own arithmetic branch) could ever see
  a `Bits` value as if it were `Num(1.0)`, the entire point of separating the domains is
  defeated — the corruption constraint 2 exists to prevent doesn't require the bit pattern to be
  NaN, just to be read as a number when it wasn't meant as one. A `Bits` value with a **finite**
  reinterpretation (e.g. a `Select`/`BitAnd` result that happens to decode to a normal float) is
  exactly the dangerous case, since it wouldn't be caught by "refuse non-finite" guards.
- The alternative — unify when they coincide — requires deciding *which* provenance wins when a
  rewrite rule later wants to do arithmetic on the unified class, and silently allows a
  `Select`/`BitAnd`/shift result to participate in `Add`/`Mul` folding. That's a correctness
  regression relative to today's `is_bitwise_domain()` exemption logic, not a neutral choice.
  If a kernel genuinely needs to bridge domains (reinterpret a bit pattern as a number, or vice
  versa), that's what `TruncToInt`/`IntToFloat` are *for* — an explicit op, not implicit
  hash-consing.

---

## 5. Candidate shapes evaluated

### (a) Two variants — `Num(Dyadic)` / `Bits(u32)` — RECOMMENDED

- **Hash-consing key**: variant tag + payload. `Num` keys off `Dyadic`'s own normalized
  `(mantissa, exp)` — exact-by-construction per `dyadic.rs`'s module docs ("two dyadics denoting
  the same number must have identical fields"). `Bits` keys off raw `u32`, unchanged. Never
  cross-domain equal (§4).
- **Arena boundary**: widen = `Dyadic::from_f32(v).map_or_else(|| Bits(v.to_bits()), Num)` inside
  `ENode::constant`, so every one of the ~46 construction call sites is unchanged text.
  Narrow = per-variant, one function (`choices_to_arena`) plus 3 test-local duplicates (§2.4).
- **Mixed reading**: resolved by never unifying (§4) — the variant split makes "never" the
  default/free behavior rather than something to enforce with extra logic.
- **Churn**: ~46 construction sites untouched; 15 shape-only sites get a compiler-forced second
  arm (mechanical, ~15 one-line edits, zero design judgment needed per site); ~9
  `.as_f32()`/`.is_const()` reads untouched; **the real work is ~10 sites**: `node.rs`'s
  `PartialEq`/`Hash`/`as_f32`/`constant` (4, this is the actual point of the exercise),
  `math/algebra.rs`'s `ConstantFold` arg-gather + result (2, §2.3), `graph.rs`'s `const_fact`
  table (7 sub-edits in 1 file, §2.2), and the 6 bound-pattern reads that must decide per-variant
  behavior (`extract.rs::choices_to_arena`, `egraph/codegen.rs::format_const`,
  `pixelflow-compiler/src/optimize.rs`'s AST-literal site, `math/fusion.rs::is_one`,
  `math/power.rs::eclass_const`, plus the test-local duplicates in `math/mod.rs`/
  `pict_rewrite_tests.rs`).

### (b) One variant carrying both representations — REJECTED

E.g. `Const { num: Option<Dyadic>, bits: u32 }`, always populated with at least one field. Looks
appealing because the 15 shape-only sites (`Const(_)` wildcard) need zero changes — a real
advantage over (a). But it fails on the two properties that matter most:

- **Hash-consing must still pick a key**, and the honest key is "which domain is this value
  living in" — which is exactly the tag that (a) already has for free via the variant.
  Implementing (b) correctly requires adding that tag back inside the struct (`domain:
  ConstDomain` or equivalent), at which point (b) is isomorphic to (a) but with extra always-
  carried, usually-meaningless payload (every `Bits`-provenance value wastefully carries a `num`
  field that's either `None` or, worse, a `Some` nobody asked for; every non-f32-representable
  `Num` has no natural `bits` value to put there — `0`? the nearest rounding's bits, silently
  reintroducing the very rounding this migration exists to avoid?).
- **Implementing it incorrectly** (keying hash-consing off `(num, bits)` jointly, or off
  whichever field is populated, without a domain tag) **silently resolves §4's mixed-reading
  question the wrong way**: a `Num(1.0)` and a `Bits(0x3F800000)` would naturally serialize to
  the same `(Some(Dyadic::ONE), 0x3F800000)` payload and hash-cons together, exactly the
  corruption constraint 2 warns against. This is the decisive argument against (b): the design
  that looks like less churn achieves it by making the dangerous case easy to reach by accident.

### (c) Nothing better identified

A three-way split (`Num`/`Bits`/`Unrepresentable`) was considered and rejected: `Dyadic` already
*is* the "unrepresentable in f32 but exact" case (that's the entire premise of the module docs —
"`2^24 + 128/255` is exactly a dyadic but is not representable in f32"), so a third variant would
just be `Num` with extra steps, adding a distinction with no behavioral difference at any of the
72 sites.

---

## 6. Ordered implementation sketch

Phased so each phase compiles and is independently testable; phase 1 alone already fixes
constraint 2 (the more urgent, cheaper-to-fix half of the bug) without waiting on `Dyadic`'s
arithmetic (`add`/`sub`/`mul`/`try_div`/`try_sqrt`) to be finished by the concurrent
implementation effort.

1. **Land `Dyadic`** (concurrent work, not this spike) — `from_f32`/`to_f32` at minimum;
   `add`/`sub`/`mul` needed before phase 4.
2. **Split the type** (`node.rs`): `ENode::Const(u32)` → `ENode::Num(Dyadic)` /
   `ENode::Bits(u32)`. Rewrite `constant()` (auto-classify via `Dyadic::from_f32`), `as_f32()`
   (per-variant: `to_f32()` / `from_bits()`), `is_const()` (unchanged, delegates), `children()`
   (add arm), `PartialEq`/`Hash` (derive-friendly once `Dyadic: Eq + Hash`, per-variant tag +
   payload — this is the hash-consing fix, the actual point).
3. **Fix every compile error** `rustc` now reports — this mechanically finds the 15 shape-only
   sites plus `canonical_op`/`node_op_cost`/`node_deps`/etc. Expect ~15-20 one-line arm additions
   across `graph.rs`, `extract.rs`, `cost.rs`, `deps.rs` (since deleted), `factored.rs`.
4. **Fix the 6 bound-pattern read sites by hand** (§2.1's list) — each needs a real per-variant
   decision (usually: `Num(d) => ... d.to_f32() ...`, `Bits(b) => ... f32::from_bits(b) ...`,
   matching what `as_f32()` now does internally, but some sites — e.g. `codegen.rs::format_const`
   — could eventually special-case `Num` to emit an exact dyadic-derived literal instead of a
   rounded one; not required for correctness, worth a `// TODO` at minimum).
5. **Fix `ConstantFold::apply`'s result routing** (§2.3, item 1): branch on
   `kind.is_bitwise_domain()` and construct `Bits`/`Num` accordingly, **still computing via f32
   arithmetic for now** — this alone satisfies constraint 2 end-to-end and is independently
   testable/shippable.
6. **Fix `graph.rs`'s `const_fact` table** (§2.2) to carry the same domain split; re-derive the
   union refusal valve's `Num`-vs-`Num` (exact `Dyadic` compare), `Bits`-vs-`Bits` (bit compare,
   unchanged), and `Num`-vs-`Bits` (always refuse, §4) cases.
7. **Fix the 4 duplicate boundary implementations** (`runtime.rs`'s memoized widen; `math/mod.rs`
   and `math/pict_rewrite_tests.rs`'s test-local widen+narrow) — same one-line pattern as steps
   2-4, applied by hand since nothing forces finding them (§3, risk 5). Consider deleting the
   test-local duplicates in favor of calling `add_arena`/`choices_to_arena` directly, as a
   follow-up cleanup (not required for this migration).
8. **(Follow-up phase, separate from this migration's minimum bar)** Add dyadic-native
   `eval_unary`/`eval_binary`/`eval_ternary` counterparts in `pixelflow-ir::OpKind` for the
   non-bitwise-domain ops, and switch `ConstantFold`'s arithmetic branch (only) to gather `Dyadic`
   args via a new `ENode::as_dyadic() -> Option<Dyadic>` (succeeds only on `Num`) and evaluate
   exactly. This is what actually delivers constraint 1 (no more f32-vs-algebra contradiction);
   phases 1-7 deliver constraint 2 and the exact-hash-consing infrastructure it depends on.
9. Re-run the full-workspace grep from §1 (`ENode::Const|ENode::constant\(|\.as_f32\(|
   \.is_const\(` plus, now, `ENode::Num|ENode::Bits`) one more time before landing, since this
   spike found the codebase moving under it — new call sites may exist by the time this is
   implemented.

Separately, flag `pixelflow-search/src/egraph/algebra.rs` (dead code, §3 of the census table) for
deletion or wiring-up in an unrelated follow-up — it is out of scope for this migration either
way, since it never compiles.
