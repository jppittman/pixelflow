# Terms about the other terms

### Exception (special case, carve-out, escape hatch)

- **Is:** a symptom. It is a place where a definition's consequences were
  not followed to the end. "No allocation outside the allocator, except x16"
  is not a rule with an exception. It is a symptom the diagnosis has not
  explained yet. Each refused exception forces a deeper model.
- **Is not:** a rule with a documented exception. STYLE.md's "Break Rules
  Sensibly … leave a brief `//` comment explaining *why*" takes the opposite
  stance, and this file does not follow it. Also not a branch that separates
  genuinely different things, and not a planned, enumerated gap such as
  one-pipeline's L1–L7. A planned gap has a name, a reason and an end.
- **Follows:** fix an exception by naming the carved-out case as an instance
  of the general one ("a loop header is a label", "a temp is a value", "the
  pool base is a value") and deleting the branch. If the case cannot be named
  that way, the model is too shallow: deepen the model, don't annotate the
  exception. "Each split rested on a statement that was true on the day it
  was written about the other side's state. When the other side changed, the
  reason expired and the split stayed" (one-pipeline §2).
- **Lives:** `.claude/skills/denotational-diagnosis/SKILL.md` (step 7,
  "Refuse exceptions", and the section "Signs it is not");
  `docs/plans/2026-09-01-register-allocation-escape-hatches.md` ("a list of
  invariants a person maintains that a type could maintain instead");
  `desloppify/rules/carve-out.json`. Exceptions known to be live today:
  - The constant pool's anchor, pinned in `r8`/`X17` (`POOL_BASE: PtrReg =
    PtrReg(8)` in `pixelflow-codegen/src/emit/x86_64.rs`; `X17` in
    `emit/aarch64.rs`).
  - `temps_for` (one per backend in `emit/{avx2,avx512,aarch64}.rs`, reached
    through the `RegisterFile::temps_for` field) and `Scratch::REDUCE_TEMPS`
    (`emit/regalloc/mod.rs`).
  - Effects (`Write`, `Seq`) as `Def`s that define no value.
  - The emitter's `Reduce` arm (`emit/mod.rs`) branching on
    `monoid != SEQ`, and `Fold::combine_op` (`pixelflow-ir/src/fold.rs`) as
    "one narrow door".
  - Ownership's "no arm owns a root" (`program/ownership.rs`).
  - Layout's per-arm `MISPREDICT_PENALTY_CYCLES` purchase
    (`program/layout.rs`; see If).
  - `Rules::tabulation` in `lower_dwrt` (`pixelflow-ir/src/passes.rs`; see
    Dwrt).

### A convention written in a comment

- **Is:** an invariant held in prose instead of in a type. "A convention
  written in a comment is an invariant something else will eventually
  break." "When you extend a type's meaning, extend its type."
- **Is not:** fixable cheaply after the fact. "The fix arrives as a runtime
  guard defending what a type should have made unrepresentable." It is also
  not worth a type when a wrong value would panic on the next line.
- **Follows:** prioritise by whether a wrong value would be *silently*
  representable. A domain confusion produces plausible pixels and deserves a
  type. The canonical cases are `Var(u8)`'s magic ranges, one `f32` lane
  carrying numbers, integers and masks at once, a `Const` carrying metadata,
  a label kept unique by naming convention, and `Args` bound by position.
- **Lives:** CLAUDE.md, "Denote before you build" and "Subtract before you
  add"; `desloppify/rules/invariant-in-comment.json`.

### Control plane and data plane

- **Is:** two different width policies. The control plane is every index,
  id, count, extent and bound that describes *a program*, and it is 64 bits
  wide. A narrower width needs a documented reason that came from a
  profiler. The data plane (`f32` lanes, `u32` pixels, ISA encoding fields)
  is deliberately narrow, because the hardware or the format dictates it.
- **Is not:** about memory being free ("It is about **which mistake is
  recoverable**"). Not a byte budget either: the `ExprNode` size assertion
  is "a tripwire against an accident and explicitly not a width to design
  against".
- **Follows:** a width chosen to fit the biggest program so far becomes a
  limit on what the language can say. `Fold`'s `u16` ends "capped a
  reduction at 65,535 terms … tuned on a psychedelic shader and broke on a
  glyph".
- **Lives:** CLAUDE.md, "The control plane is 64-bit"; #1328;
  `desloppify/rules/control-plane-64-bit.json`. Today these are narrower than
  64 bits with no profiler reason:
  - `ExprId(pub u32)`, `BufferId(pub u16)`, `BufferIdentity(u32)` (justified
    as "on the way out") and `Var(u8)` (`pixelflow-ir/src/arena.rs`).
  - `Fold { lo, hi, stride: u32 }`, whose own field doc cites this rule, and
    `Binder(u8)` (`pixelflow-ir/src/fold.rs`).
  - `LatticeShape([u32; _])` (`pixelflow-ir/src/variance.rs`).
  - The DAG's `Id(u32)` and `DagIdentity(u32)` (`pixelflow-ir/src/dag.rs`).
  - `EClassId(pub(crate) u32)` (`pixelflow-search/src/egraph/node.rs`).
  - `ScheduledOp::Context(u16)` (`pixelflow-codegen/src/program/mod.rs`).
  - The string `Label` (`pixelflow-codegen/src/emit/mod.rs`).

  `ExprNode`'s own doc still says the assertion "guarantees <= 16 bytes",
  while the assertion reads `<= 32`.

### Fold and dispatch (style). Homonym of the IR's Fold

- **Is:** two things code can do with a set of cases. "A **fold** leaves
  fewer live possibilities than it found … **Dispatch** does the opposite —
  it keeps every case alive." Guard clauses are the strongest fold, because
  they delete a case together with its join point. Branchless code is the
  limit: no case survives to runtime. Extracting a trait is the same move at
  type scope. Dispatch that happens once, at construction, is fine. Dispatch
  repeated at every use is what this rule warns against.
- **Is not:** the IR's `Fold`/`Reduce`. Not a licence to hand-roll a
  branchless form that is worse than the instruction already available. Not
  satisfied by `If`, which is dispatch (see If).
- **Follows:** an `else` doing double duty is usually a fold that was not
  taken. `Box<dyn Trait>` pays the dispatch on every call. Thirty walkers
  that each match a case to read one property should call one method.
- **Lives:** CLAUDE.md, "Fold before you dispatch" and "New implementation
  of an existing category → trait first"; `docs/STYLE.md` (Code Structure
  rule 3). Today `docs/STYLE.md` rule 3 and `AGENTS.md` still call the branch
  around an `If` "not a violation" and something codegen buys ("bought only
  where the skipped arm outcosts `MISPREDICT_PENALTY_CYCLES`"). That
  contradicts the If entry. `docs/STYLE.md` also cites
  `pixelflow-codegen/src/emit/guards.rs`, which was deleted in #1313; the
  analysis now lives in `program/guards.rs`.
