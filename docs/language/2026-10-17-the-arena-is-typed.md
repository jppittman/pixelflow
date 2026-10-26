# The arena is typed

**Status:** `Proposed`, implementing a ruling. The types are
`2026-10-15-kernels-evaluate-kernels.md` §2 ("Everything is a kernel: the
types"), accepted by JP there and reaffirmed 2026-10-17: "kernels are just
typed". That section's "what this asks of the IR" items 1 and 2 are the work;
this note denotes them and orders the CLs. Nothing here is a new ruling.
**Verified against:** `c5dd7abfabce58313d3004fc87d53148311cf399`.

## 0. The claim

A kernel is a term of the typed lambda calculus, and today its arena is
untyped. One `f32` lane is a real, an integer and a mask by convention, and
every consequence of that is a runtime guard defending what a type would make
unbuildable:

- `OpKind::is_bitwise_domain()` tells the folder which results are bit
  patterns, so that an all-ones mask (NaN read as a number) is not refused as
  "non-finite" (`pixelflow-search/src/math/algebra.rs:1043`).
- `OpKind::mask(bool)` is the one legal spelling of true, because `1.0` under
  a bitwise blend gives a value neither arm held (CLAUDE.md, "Floating point
  at the edges").
- The e-graph keys a constant by its bits (`ENode::Const(u32)`), so a folded
  true mask and a `Const(NaN)` with the same pattern are **one e-class**. A
  rule that fires on a NaN constant fires on a mask. Nothing has tripped it;
  nothing forbids it.
- `exp2` and `log2` lower through `Shl(biased, shift) // bitcast result` and
  `BitAnd(x, mant_mask)` on a real `x` (`pixelflow-ir/src/passes.rs:1189,
  1226`): a reinterpretation of a real as its bits and back, written nowhere.
  This is the "reinterpret is free" escape hatch CLAUDE.md names as the moment
  to pay.
- `kernel!`'s sema already has the types — `Type::{Scalar, Mask, Count,
  Record}` (`pixelflow-compiler/src/sema.rs:85`) — and its doc says the
  distinction "is enforced here because it cannot be enforced there". It can.
  The runtime tier (`Kernel::select`, `Kernel::at`, the glyph builders)
  constructs arenas with no sema in front of it, and gets no check at all.

The fix is the one CLAUDE.md prescribes: **when you extend a type's meaning,
extend its type.** The arena gets the typing judgment, construction refuses an
ill-typed node, and every guard above is deleted because the case it guarded is
unrepresentable.

## 1. The denotation

From §2, unchanged:

```
S ::= Real | Int | Mask              base types: what one lane holds
D ::= S | D × D | Lattice(w, h)      domains
T ::= S | D → T                      a kernel is a value of arrow type
```

Item 1 is `S` on every arena node. Item 2 is `T` on binders. This note is
mostly item 1; §5 says what item 2 is and why it waits.

### 1.1 Every op has a signature

The judgment is a table, generated from `op_table!` like `COUNT`, `ALL` and
`variant_name` are, so an op cannot be added without one:

| ops | signature |
|---|---|
| `Add Sub Mul Div Neg Sqrt Rsqrt Abs Min Max MulAdd Recip Floor Ceil Round`, the transcendentals, `Pow Atan2` | `Real^n → Real` |
| `Lt Le Gt Ge Eq Ne` | `Real × Real → Mask` |
| `If` | `Mask × S × S → S`, for each `S` |
| `TruncToInt` | `Real → Int` (conversion, `cvttps2dq`) |
| `IntToFloat` | `Int → Real` (conversion, `cvtdq2ps`) |
| `IAdd Shl Shr` | `Int × Int → Int` |
| `BitAnd BitOr` | `Mask × Mask → Mask` and `Int × Int → Int` |
| **`Bits`** (new) | `Real → Int`, the reinterpretation: zero instructions |
| **`Float`** (new) | `Int → Real`, its inverse: zero instructions |
| `Reduce` with `⊕` | `⊕ : S × S → S`, body `: S`, a value bound `: Real` (`Int` at T6), result `: S`; a known bound is node data |
| `Gather` | `Buffer × Real × Real → Real` (item 1; see §5 for item 2) |
| `Tuple` | a product of its children's types; root only |
| `Dwrt` | `Real → Real`, operand `Real` |
| `Const` | the literal's type: `Lit::Real(f32)`, `Lit::Int(i32)`, `Lit::Mask(bool)` |
| `Var` (coordinate) | `Real` |
| `Var` (fold binder) | `Int`, read in a `Real` body through `IntToFloat` — see §3 |
| `Uniform`, `Param` | `Real` (item 1); their declared type (item 2) |

`Floor` stays `Real → Real`: `roundps` yields a real with an integral value,
and the instruction is the meaning. The bridge to `Int` is `TruncToInt`, so
the discrete kernel's lift (§2 of the ruling, "the lift is precomposition")
spells `nearest(D) = D ∘ trunc ∘ floor`. The two new ops are the price of the
`exp2`/`log2` hatch: a reinterpretation is a node that compiles to nothing,
and once it is a node the folder, the e-graph and the derivative pass all see
it rather than guessing.

`BitAnd`/`BitOr` are overloaded in two base types and nothing else is: the
result type is the operands', and the operands agree. `If` is polymorphic in
its arms. No other op is.

### 1.2 Well-typed by construction, not by check

The arena holds the judgment two ways, and both are needed:

- **A type column**, `ExprArena::sort(id) -> Sort`, one per node, filled at
  push from the signature and the children's types. This is what a pass that
  rebuilds an arena (extraction, `link`, the `kernel!` loader) reads to stay
  typed, and what the e-graph carries per class.
- **Typed handles** on the construction API: `ExprId<S>`. `push_binary(Lt, a,
  b)` takes two `ExprId<Real>` and returns an `ExprId<Mask>`;
  `push_if(m, a, b)` takes an `ExprId<Mask>` and two `ExprId<S>`. A mask in
  arithmetic does not compile. `kernel!` emits typed pushes from the types
  sema already has, and the runtime tier (`Kernel::select` and friends) gets
  the same guarantee for free.

The untyped `ExprId(pub u32)` the e-graph and the passes use internally stays
as the column's index; the typed handle is `ExprId<S>(ExprId)` and nothing
else. (The `pub u32` and the control-plane width are BACKLOG items of their
own; this note does not widen them, and does not need to.)

The `ExprNode` size assertion (`≤ 32` bytes, `arena.rs:464`) is a tripwire,
not a budget. `Const(Lit)` is eight bytes and fits. If a step here trips it,
the assertion moves, not the design (CLAUDE.md, "the control plane is
64-bit").

### 1.3 What a pass must preserve

A rewrite is **type-preserving**: both sides of a rule have the same type, and
a rule's pattern variables are typed. The rule set (`rules.rs`, `and_or.rs`,
`fold_rules.rs`, `derivative.rs`, `shift.rs`) is audited once, per rule, in
the CL that types the e-graph:

- Commutativity and associativity for `Min`/`Max`/`Add`/`Mul` are `Real` rules
  and bind `Real` variables.
- `and_or.rs` rules are `Mask` rules (and, where the same identity holds for
  `Int`, two instances or one polymorphic rule — the audit decides each).
- `If` rules (`IfHoistUnary` and the rest) bind the condition at `Mask` and
  the arms at one `S`.
- A fold rule's `⊕` is typed by the body.

An e-class has one type; `union` of two classes of different type is a panic
naming both nodes, a rule bug and never a fact about a kernel. Extraction
redeclares the column into the output arena.

The folder folds by type: a `Mask` comparison yields `Lit::Mask`, an `Int` op
yields `Lit::Int` with wrapping semantics the ISA agrees on, a `Real` op yields
`Lit::Real` and refuses a non-finite result. The "refuse non-finite unless
bitwise" exemption is gone because a `Mask` result is never a float. The
value-aware platform rows (`fold_is_platform_specific`) are unchanged: those
are about values, not types.

The derivative pass is defined on `Real → Real` only (ruling, §2). Its
hand-written arms for `TruncToInt`/`IntToFloat`/`IAdd`/`Shl`/`Shr`/`BitAnd`/
`BitOr` (`passes.rs:1588–1641`), which exist to refuse or zero a derivative
through a bit op, become one arm: `Dwrt` of a non-`Real` node is a type error
at construction.

## 2. What it deletes

- `OpKind::is_bitwise_domain`, its test, and the one caller
  (`algebra.rs:1043`).
- `OpKind::mask(bool)` and every site that spells a mask as a float
  (`algebra.rs:1299`, `compiler/emit.rs:509`, the folder's comparison arms).
- `ENode::Const(u32)` keyed by bits; `Const(Lit)` keys by typed value, so a
  mask and a NaN are different classes.
- The `// bitcast result` and `// int lanes` comments in `expand_exp2`/
  `expand_log2`: the reinterpretations are `Bits`/`Float` nodes.
- The derivative pass's seven bit-op arms.
- `sema.rs`'s "cannot be enforced there" paragraph, and the duplicate
  mask/scalar logic it carries that the arena now owns. Sema keeps `Count`
  and `Record`, which are the front end's and not lane types.
- The convention paragraphs in CLAUDE.md ("Denote before you build" names this
  case as the example) and `pixelflow-ir/intent.md`'s Behind line for it.

## 3. The one place bytes may move

Every CL below is gated byte-identical on AVX-512, AVX2 and NEON
(`byte_probe`, as every regalloc and e-graph CL was) **except the fold
binder's type**.

Today a fold's index `k` is a `Var` read as a real lane, and the body does
real arithmetic on it directly; sema's `i as f32` "lowers to nothing". Under
the types `k : Int`, and a body that uses it as a real reads `IntToFloat(k)`.
Whether that costs an instruction depends on how the CPU backend keeps the
counter: if the loop counter is already a real lane (one `addps` per trip),
`IntToFloat` of it is the same lane and the backend folds the conversion to
nothing, and the bytes are identical; if the counter is an integer lane, the
conversion is a `cvtdq2ps` per trip that today is hidden in the counter's
own update. The CL that types the binder measures both and says which. Either
way the kernel's value is unchanged; the as-if contract holds.

The alternative, `k : Real` with "integral by convention", is the convention
this note exists to delete, so it is not taken.

## 4. The series

One PR each, tree green, the glyph cache, cell grid and packed frame render.
Each PR's gate is `byte_probe` under `PIXELFLOW_ISA=avx512`, `=avx2`, and
NEON under qemu, parent against head: **empty** for T1–T5, measured and
explained for T6.

- **T1. `Const(Lit)`, and the e-graph keys constants by typed value.**
  `Lit::{Real, Int, Mask}`; `OpKind::mask` becomes `Lit::Mask`; `ENode::Const`
  takes a `Lit`. The folder's comparison arms yield `Lit::Mask`. No signature
  table yet, so nothing is refused; but a mask is no longer a float anywhere,
  and the NaN/mask aliasing is closed. Deletes `OpKind::mask`.
- **T2. `Bits` and `Float`.** The two reinterpretations as ops that lower to
  nothing on every backend. `expand_exp2`/`expand_log2` spell them. The
  derivative pass refuses them by name (for now; T5 makes that structural).
- **T3. The signature table and the type column.** `op_table!` grows a
  signature per op; `ExprArena::sort(id)` is filled at push; an ill-typed
  push is `Err(TypeError { op, got, want })` on the fallible path and a panic
  naming the node on the infallible one. Every arena the tree constructs is
  well-typed or the tests say which one is not. The e-graph carries the type
  per class and panics on a mixed union. Rule audit (§1.3) lands here.
  Deletes `is_bitwise_domain` and the folder exemption.
- **T4. Typed handles.** `ExprId<S>` on the public construction API;
  `kernel!` emits typed pushes from sema's types; `Kernel::select`, `at`,
  `dx`, the fold constructors and the glyph builders take and return typed
  handles. Untyped construction is `pub(crate)`. Deletes sema's duplicate
  enforcement.
- **T5. `Dwrt` is `Real → Real` by type.** The seven bit-op derivative arms
  become unrepresentable and are deleted.
- **T6. The fold binder is `Int`.** `Var(binder) : Int`; the body reads it
  through `IntToFloat`; the backend's counter is measured per §3. T6 also
  retypes a fold's value bounds with the binder
  (docs/language/2026-10-20-a-bound-is-a-value.md §5): it lifts
  `BoundArithmetic` and deletes the guard's integrality conjunct, measured. A
  known bound is node data, so T6 re-keys no bound literal.

T1 and T2 are independent of each other and of nothing; T3 needs both; T4–T6
follow T3 in order.

## 5. Item 2, and why it waits

Arrow types on binders: `Var`, `Uniform`, `Param`, `Buffer` and `Ref` are five
spellings of "a name bound later" (BACKLOG N1), and under the types they are
one binder table whose entries have types `Real` (a coordinate, a uniform, a
`kernel!` parameter), `L → Real` (a discrete kernel supplied at call time: the
cell table, the frame, a glyph's piece table) and `D → T` for a `Ref`'s
referent. `Gather` is then application of an `L → Real` binder at an
`Int × Int` index, `Ref` is application of an arrow, and the discrete-kernel
series (`BoundManifold`, `bind`, `MAX_BOUND_BUFFERS`, the `Gather` case in
every pass, the bound-buffer replay in `collapse_bench`) is what falls out.

It waits on item 1 because an arrow type's domain and codomain are base types,
and because the one question that series had left — "is the lattice index
worth a type in the IR?" — is answered by T3: it is `Int`, and a `Gather`
whose index is `Real` stops compiling. The series is re-derived against this
note once T3 lands; its survey (`scratchpad/r2-discrete-kernel-plan.md` §2)
stands.

## 6. What this is not

- Not a change to any kernel's value. The as-if contract holds at every CL;
  T6 is the only one that may change an instruction, and it says which.
- Not predicate domains (§2's item 3). A domain with a predicate is an
  e-graph class analysis, later, and needs these types under it first.
- Not a width change. `ExprId(pub u32)`, `Fold`'s `u32` ends and `Binder`'s
  `u8` are BACKLOG items against the 64-bit control plane and land on their
  own; this note neither widens nor depends on them.
- Not the schedule. Where the lattice folds live (§3 of the ruling) is
  orthogonal: they are `Reduce` nodes and get a type like any other.
