# The Optimizer API: denotation, surface, and the gaps behind it

**Date:** 2026-09-02
**Base:** `origin/main` @ `2e82cdc2`
**Branch:** `claude/optimizer-api`
**Status:** design — the API PR that follows this doc is mechanical and behavior-preserving; the two behavior changes it depends on are separated out below and land first.

---

## 0. What this is and why now

Six weeks of cost-model research have accumulated as branches: a Guide, an
application budget, per-rule telemetry, three label definitions, a rule
reordering. Each one reaches into `pixelflow-search` at a different place
because there is no place to reach into. There are three production entry
points into the optimizer and they have already drifted apart from each
other (§2.6). Every research branch therefore carries integration risk that
has nothing to do with the question it is asking.

The fix is not more machinery. It is **one entry point**, with the research
levers as *optional fields on it that default to `None`*. Then a Guide PR is a
`guide: Some(..)`, a schedule-cost PR is a `rerank: Some(..)`, and a labeling
PR is an `observer: Some(..)` — each measurable on its own, none of them a
correctness review.

That last clause is the load-bearing one, and it is a claim about **denotation,
not about code**: a policy that only chooses *which equalities to discover and
in what order* cannot change *what the result means*. Section 1 checks that
claim against the code rather than assuming it, because two of the five laws
turn out not to hold as stated.

Reading order: §1 is the audit (what is true today). §2 is the surface. §3 is
the gap table — the answer to "what future work sits behind this abstraction".
§4 is the migration sequence for the in-flight branches. §5 is versioning.

---

## 1. The laws, as verified

Five laws were proposed. Each is stated, then given a status against the code
at `2e82cdc2`, the test that pins it (or the test that should), and the
consequence if it breaks. **Three of five need correcting.**

| | Law | Status |
|---|---|---|
| **L1** | Soundness: `add` is a homomorphism onto the semantic quotient | **HOLDS** since #1105 (the silent hole is closed and pinned); still **unenforced** for ~1/3 of rules |
| **L2** | Monotonicity: saturation only adds equalities | **TRUE**, pinned by a budget-refinement ladder since #1105 |
| **L3** | Round-trip: `add(extract(g,c)) ∈ c` | **UNREPRESENTABLE as stated** — no membership test exists |
| **L4** | Guide neutrality: any ordering policy preserves denotation | **TRUE**, but wholly derivative of L1; untested |
| **L5** | `extract_dag` is argmin under an additive cost | **VIOLATED** — it is a heuristic, twice over |

### L1 — Soundness

> Every rewrite rule preserves denotation on its documented domain, so an
> e-class *is* a semantic equivalence class and `add` is a homomorphism from
> the term algebra onto the e-graph quotient:
> `add(op(t₁..tₙ)) = canonical(op(add(t₁)..add(tₙ)))`.

**Status when this doc was written: VIOLATED, once, silently. FIXED in
#1105.** `EGraph::add` returned `EClassId(0)` when `self.classes.len() >=
HARD_CLASS_LIMIT` (100 000) — a sentinel, not an insertion, and it did not go
in the memo:

```rust
if self.classes.len() >= HARD_CLASS_LIMIT {
    // Return a sentinel pointing at class 0. The e-graph is over
    // budget; further growth would be useless anyway.
    return EClassId(0);
}
```

The caller did not know it got a sentinel. `RewriteAction::Create`'s handler
took the returned id and called `union_counted(class_id, EClassId(0))`,
**asserting an equality that is false**: class 0 is whatever term was added
first. Homomorphism failed at exactly that point, and every subsequent
extraction could legally return class 0's term for an unrelated class.

It was unreachable in practice — the production class caps are 500/2000/5000
(§1.L2), two orders of magnitude below the limit — but "unreachable" is not
the standard. It was a hole in `add`'s *type*, papered over by a comment, and
precisely the failure mode CLAUDE.md forbids: a wrong value that is silently
representable.

**How it was fixed — the subtraction, not the `Result`.** Both options this
doc floated (`add → Result<EClassId, GraphFull>`, or a `Budget` dimension)
remove the sentinel; only the second removes the *failure mode*. `add` is the
homomorphism, so there is no correct value for it to return when it cannot
insert — any answer it gives is a class that does not hold the node. So the
limit moved to where growth is actually **decided**, and `add` became total
with no error case at all:

- `egraph::HARD_CLASS_LIMIT` is now a `pub const` and a **ceiling on every
  class budget**. `saturate_with_limits`, `apply_rule_at_index_timed` and
  `apply_rules_budgeted` each clamp their caller's cap to it, so a call site
  that passes `usize::MAX` gets the ceiling and an honest `ClassCap` stop
  instead of a graph that grows until `add` starts lying. Production caps are
  untouched, so no shipping configuration changes behavior.
- `apply_single_rule` — the one driver that takes no budget, because it
  applies one action per call — **panics** at the ceiling, naming the limit and
  the class count. Returning `false` there would be indistinguishable from
  "the rule did not match", which is the silent failure the ceiling exists to
  prevent. It has no production callers.
- The ceiling is now approximate by a bounded amount (a sweep under-estimates
  multi-node actions at 3 classes each, then commits the batch it accepted).
  That is the *point* of moving it: at the driver it guards memory, not
  meaning, so a bounded overshoot of a 100 000-class ceiling is free. The old
  placement bought exactness with a false class id.

**Pinned by** `over_budget_growth_cannot_assert_a_false_equality`
(`egraph/graph.rs` tests): fills a graph to the ceiling, checks `add` still
names a class that holds the node, then drives saturation past the ceiling and
asserts no equality appeared among terms that are not equal. Against the
pre-fix code it fails with four false unions — and a `SaturationStop::Quiesced`,
i.e. the run reported success.

**Enforcement gap — the bigger half.** L1 is only as strong as the oracle that
checks it, and the oracle does not cover a third of the rules. The only
cross-form numeric gates are `pixelflow-search/src/math/pict_rewrite_tests.rs:349`
and `math/mod.rs:332,366`. The PICT generator's vocabulary
(`OUTER_OPS` :206, `UNARY_WRAPPERS` :210, `build_shape` :226) only ever builds
`Add, Sub, Mul, Div, Neg, Abs, Sqrt, Recip, Sin, Cos, Exp`. Therefore:

- **no expression in any test contains** `Pow`, `Ln`, `Log2`, `Exp2`, `Tan`,
  `Asin`, `Atan`, `Min`, or `Max`;
- so all 11 `power_rules`, the `Ln`/`Log2`/`Exp2` rules in `exp.rs`, and the
  `Tan`/`Asin`/`Atan` parity rules **cannot fire under an oracle at all**;
- `math/{parity,trig,exp,power,fusion}.rs` contain zero `#[test]`;
- the `Pow` mentions in `latency_prior_regression.rs:41,66` are *cost*
  assertions, not numeric ones.

Also: the contract in `docs/plans/2026-08-05-egraph-nnue-research-workflow.md:289-305`
gates same-form hard and cross-form **at well-conditioned points**. The
implemented cross-form gate substitutes `is_finite` for "well-conditioned" —
a weaker predicate. That is the same substitution that let the `sin` range
bug through (CLAUDE.md, "Precision is on the table; range is not").

**The test that should pin it:** extend the PICT vocabulary to the full op set
and require every rule in `all_rules()` to be *witnessed* — a test that
saturates the corpus, collects `match_counts`, and fails if any rule never
fired. A rule nothing exercises is a rule nothing verifies.

**Consequence if L1 breaks:** every downstream law breaks with it. L4 in
particular is *nothing but* L1 plus "extraction picks a node from an e-class".
A Guide cannot be argued neutral on a graph that is not sound.

### L2 — Monotonicity

> Saturation only ever adds equalities; the partition at t+1 refines t. Budget
> truncation stops early; it can never make the graph unsound.

**Status: TRUE, and pinned since #1105.** Nothing removes an equality. The
class cap only `break`s the sweep; `rebuild_budgeted` is
partial-but-consistent; `union`'s constant-contradiction refusal under-merges
by design, which is monotone in the safe direction. The single exception was
L1's `HARD_CLASS_LIMIT` sentinel, which *added a false equality* — so L2 holds
exactly where L1 does, and both now hold everywhere.

**Pinned by** `saturation_at_any_budget_never_removes_an_equality`
(`egraph/graph.rs` tests), which is the ladder this section asked for: the same
seed saturated under `[1, 2, 8, 64, 512, 4096, HARD_CLASS_LIMIT]` classes,
asserting at each rung that a hand-made union survives and that every probe
pair equal at a smaller budget is still equal at this one — plus a guard that
some budget derived *something*, so the ladder cannot pass vacuously. This is
what makes "stop early" safe as an API concept, and it is what `Budget`
(§2.2) stands on.

**One incompleteness the test documents rather than asserts:** this `rebuild`
does no *upward* congruence closure. `union` enqueues only the merged class,
never its parents, so `a = b` does not by itself re-canonicalize `f(a)` and
`f(b)` into one class — they merge when a later sweep re-walks them, or not at
all. That is fewer equalities than an ideal congruence closure, which is the
direction L2 permits (and L1 requires), so it is a *quality* gap, not a
soundness one. Filed as #1106: it means CSE quality depends on sweep order in
a way nothing currently measures, and the first step there is a measurement
(how many merges a parent-tracking rebuild would recover) rather than a
parent list on faith.

**Consequence if it breaks:** `Budget` stops being a pure
quality/compile-time dial and becomes a correctness dial, at which point every
budget value needs its own review — which is the whole thing this API is
trying to avoid.

### L3 — Round-trip

> `add(extract(g,c))` is in `c`, and the extracted term denotes what the class
> denotes.

**Status: UNREPRESENTABLE AS STATED.** There is no membership test in the
API. `Extraction` (`extract.rs:32-90`) makes well-foundedness *structural* —
the materialization half of the law, and a good use of the type system. But
nothing anywhere asserts `add(extract(g,c))` lands back in `c`.
`runtime.rs:755` (`gather_arena_round_trips_through_the_egraph`) is a
*denotation* round-trip — buffer identity plus evaluated semantics — not a
membership one.

This law should be **restated as two laws**, because they need different
machinery:

- **L3a (materialization, holds):** the extracted term is well-founded and
  every node it names was in the graph. Structural, already pinned by
  `Extraction`'s constructor.
- **L3b (denotation, holds, tested):** the extracted term evaluates to what
  the input term evaluated to. `runtime.rs:755` is this test. It is the one
  that actually matters for the compiler.

Genuine membership (`add(extract(g,c)).find() == c.find()`) is cheap to test
in a unit test and needs no public surface: re-add into a *clone* of the graph
and compare canonical ids. It belongs in `pixelflow-search`'s test module, not
in `Optimizer`'s API. **Do not add a public membership method to satisfy a law
statement** — that is machinery growth in service of a doc.

### L4 — Guide neutrality — the load-bearing one

> For **any** ordering policy G and **any** budget B, the extracted result
> denotes the same function; a Guide changes cost and compile time, never
> meaning.

**Status: TRUE, and it is a one-line proof once L1 and L2 are in hand.** As
of #1105 both premises are pinned by tests rather than asserted by reading, so
the proof below rests on something a future change has to break loudly.

The argument, stated so a future Guide PR can cite it instead of re-deriving
it:

1. By **L1**, every equality in the graph is a semantic equality. The graph is
   a set of semantic equivalence classes at every instant.
2. By **L2**, a Guide can only cause the graph to hold *a subset* of the
   equalities an exhaustive run would hold. It cannot cause it to hold a
   *different* one — there is no operation available to it that removes or
   invents an equality.
3. Extraction picks **one node from the root's class**. Every node in that
   class denotes the root's function, by (1).
4. Therefore the extracted term denotes the root's function, for every G and
   every B. ∎

**What follows, and this is the payoff of the whole API:** a Guide PR does
**not** need to argue correctness. It does not need a numeric-equivalence
suite, a differential test against the unguided path, or a soundness review.
It needs exactly two things:

- proof that it only *orders and truncates* — it does not construct
  `RewriteAction`s of its own, does not call `union`, does not touch
  `const_fact`;
- a **quality** measurement (extracted cost, compile time) against the
  unguided arm at matched budget.

Reviewers should push back on a Guide PR that argues correctness — that
argument is already made here, and re-litigating it per policy is exactly the
per-implementation review this API exists to abolish.

**What I checked on `claude/phase3-guide`**, to confirm the current
implementation is in fact inside the fence: `GuidedSaturation::until_applications`
(`egraph/saturate.rs:407-575`) sorts candidates within a round and truncates
at the budget. It mints no nodes and performs no unions of its own. It is
neutral.

One soft spot, correctly classified as *budget, not semantics*: the dedup set
is keyed `(rule_idx, ClassContentKey)` (`egraph/candidate.rs:170-177`), where
`ClassContentKey` is the class's full sorted node-shape vector (:154-162) — so
a "permanent" skip is content-addressed and **re-arms the moment the class
changes**. That is fine: a re-fire on unchanged content is idempotent, the
union is already made. Separately, `sort_by_key(NodeShape::sort_key)`
(:125-130) uses a `DefaultHasher`, so a hash tie leaves order to
`egraph.nodes()`, which `rebuild` permutes — two structurally identical
classes can key unequal. **That costs work, never meaning.** Both belong in a
Guide PR's *cost* discussion, never its correctness one.

**The test that should pin L4, and it is cheap:** one arena, N guides
(including a deliberately adversarial random one and a "reverse the sensible
order" one), assert **identical denotation** across all N — *not* identical
cost. Cost is expected to differ; that is the point of a Guide. This test is
worth more than any per-Guide review and does not exist today.

### L5 — Cost and extraction

> If the cost model is ADDITIVE over term structure, `extract_dag` is argmin
> (exact DP). A non-additive cost must enter as a `Reranker` over whole
> extractions.

**Status: the second sentence is right; the first is VIOLATED.**
`extract_dag` (`extract.rs:1463-1560`) is not argmin, for three independent
reasons.

1. **It prices a tree, not a DAG.** Each class's score sums its children's
   `best_cost`, which is a tree cost. The "DAG" in the name is phases 2–4 —
   ref counts, sharing, toposort. **Sharing is never priced.** A shared
   subexpression is charged once per use in the objective and once in total in
   the emitted code.
2. **It is a single DFS, not a fixpoint.** A class whose child is still
   `on_stack` scores that child at `CYCLE_COST = 1_000_000` (:1466,1474,1484)
   and is never revisited. On the cyclic e-graphs that saturation *always*
   produces (commutativity alone makes cycles), the result is a heuristic
   whose quality depends on visit order.
3. ~~**The reported cost need not be the returned term's cost.**~~ **FIXED
   (#1111).** `total_cost` was read before `repair_choices_well_founded`
   mutated `best_node`, so a caller comparing `total_cost` across arms could
   be comparing numbers belonging to terms it did not receive — measured in
   #1115 as differing from the returned term on **132 of 302** kernels. Both
   reported numbers are now computed from the *repaired* choices by
   `cost_of_choices`, and there are two of them, because (1) above means one
   number cannot answer both questions a caller has:
   `ExtractedDAG::total_cost` is the **tree** cost the DP minimizes, and
   `ExtractedDAG::dag_cost` is the **DAG** cost the emitted kernel pays. A
   caller asking "what will this kernel cost?" wants the second (on
   `shader:julia_set`: ~1.4e7 against 716). `Optimized::cost` carries the same
   pair, so `Optimizer::run` no longer discards it. (1) and (2) are unchanged
   and remain #1116's business.

Plus a type smell that the API should not inherit: `Dwrt` is priced
`usize::MAX / 4` (`cost.rs:292`) — a sentinel wearing a cost's type, kept from
overflowing only by `saturating_add`. "Never select this" is a *constraint*,
not a large number; when the Dwrt tier unifies (#1085) it should be spelled as
one.

Anytime-cost monotonicity is asserted **empirically** (`anytime.rs:306-337` on
`phase3-guide`), not guaranteed by any of the above — which is the correct
posture given (1)–(3), and worth saying out loud so nobody reads that test as
a proof.

**Consequence for the surface, which is the actionable part:** `cost` in the
`Optimizer` is documented as *the objective the extractor is trying to
minimize*, **not** as "the extractor returns its minimum". `Reranker` is
therefore not only the seam for non-additive cost — it is the seam for *any*
objective that wants a guarantee, because it scores whole extractions and can
be argmin over a candidate set by construction. That is a stronger argument
for #1093's seam than the one that shipped with it.

---

## 2. The surface

One entry point. Five optional levers, all defaulting to the production
behavior that ships today. Every item below is justified by a production
caller or by a numbered gap with a named consumer; items that are neither are
listed in §2.7 as explicitly **out**.

Three of the five types **already exist** on main or on a branch. This is
mostly integration, not invention.

### 2.1 `RuleSet` and `RuleId` — NEW, and the largest piece

```rust
/// A stable identity for a rewrite rule, independent of its position in
/// any rule vector.
///
/// Positional indices are not identities: `all_rules()` returns a `Vec`,
/// and two queued changes (a reordering and a 33-rule batch) both move
/// every index in it. A trained Guide's per-rule weights, a per-rule label
/// table, and a JSON report key are all addressed by index today, and a
/// reorder silently repoints every one of them.
///
/// A `RuleId` is derived from the rule's *discriminating* name, so it
/// survives both reordering and insertion.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct RuleId(/* interned */);

/// An ordered rule vocabulary, plus the fingerprint that identifies it.
pub struct RuleSet { /* rules: Vec<Box<dyn Rewrite>> */ }

impl RuleSet {
    /// The production vocabulary. Today: `all_rules()`, 62 rules, in
    /// declaration order.
    pub fn production() -> Self;

    /// Hash over the ordered sequence of `RuleId`s. Covers content *and*
    /// order, because saturation is order-sensitive: two runs with the
    /// same rules in different orders can extract different (equally
    /// valid, differently priced) terms.
    pub fn fingerprint(&self) -> Fingerprint;

    pub fn id_of(&self, idx: usize) -> RuleId;
    pub fn index_of(&self, id: RuleId) -> Option<usize>;
}
```

**Justified by:** G5 (rule identity), G6 (schema fingerprint), G8
(configuration fingerprint). Named consumers: `LinearCandidateGuide` /
`PerRuleRateGuide` checkpoints, `gen_strict_labels`'s JSON, `RuleTemplates`.

**The blocker, and it must be fixed first.** `RuleId` **cannot be derived from
`Rewrite::name()` as it stands**, because `name()` returns the *family* name,
not the instance's. Verified counts in `all_rules()` (62 rules):

| `name()` | instances | site |
|---|---|---|
| `"commutative"` | 4 (Add, Mul, Min, Max) | `math/algebra.rs:462`, ctors :996-999 |
| `"associative"` | 4 (Add, Mul, Min, Max) | :349, ctors :1016-1019 |
| `"reverse-associative"` | 4 (Add, Mul, Min, Max) | :405, ctors :1021-1024 |
| `"odd-negation"` | 4 (Sin, Tan, Asin, Atan) | `math/parity.rs:275`, `parity_rules()` |
| `"identity"` | 2 (Add, Mul) | :643 |
| `"idempotent"` | 2 (Min, Max) | :738 |
| `"even-negation"` | 2 (Cos, Abs) | `parity.rs:276` |
| `"canonicalize"` | 2 (AddNeg, MulRecip) | `inverse_pair_rules()` |
| `"involution"` | 2 (AddNeg, MulRecip) | `inverse_pair_rules()` |
| `"cancellation"` | 2 (AddNeg, MulRecip) | `inverse_pair_rules()` |
| `"inverse-annihilation"` | 2 (AddNeg, MulRecip) | `inverse_pair_rules()` |

**30 of 62 rule instances answer to 11 names — 43 distinct names for 62
rules.**

The fix is already in the tree as a pattern to copy: `LogPower::name()`
(`math/power.rs:255-258`) discriminates on `self.log_op`, returning
`"log-power"` vs `"log2-power"`. Do the same for the eleven families —
`"commutative(mul)"`, `"odd-negation(atan)"` — a mechanical touch of ~11 impls.

Two things fall out for free:
- `EGraph::match_counts: HashMap<String, usize>` (`graph.rs:50`), and therefore
  `SaturationResult::rule_matches`, currently **collapses all four
  `Commutative`s into one number**. Every per-rule report built on it is
  wrong by aggregation today.
- A test pinning "62 rules, 62 distinct `RuleId`s" becomes possible, and is
  the guard that keeps the 33-rule batch honest.

**Precedent that this is the right shape:** the checkpoint format *already*
keys ops by name — `op_names` → `op_index`, `nnue/guide/linear.rs:196-211`,
with a length-agreement check and a loud error. Rules are the one axis that
never got that treatment. That is the entire argument for `RuleId` in one
sentence: **do for rules what the checkpoint already does for ops.**

### 2.2 `Budget` — EXISTS, split across two branches

```rust
/// What stops saturation.
///
/// Every variant is deterministic: the same arena under the same budget
/// produces the same graph on any machine, at any load. Wall-clock is
/// deliberately **not** a variant — see `hard_ceiling` below.
pub enum Budget {
    /// Today's production behavior: iteration and class caps chosen from
    /// the input's node count (`config_for_node_count`).
    Production,
    /// A fixed number of rule applications. The budget the research arms
    /// use, because it is the one that is comparable across policies.
    Applications(u64),
    Explicit { iterations: u32, classes: usize, applications: Option<u64> },
}
```

**Justified by:** the three production call sites (§2.6) and G3. Named
consumers: `optimize.rs`, `runtime.rs`, `ir_bridge.rs`, plus every research
arm on `phase3-*`.

**Already implemented, on branches, not new design:** `SaturationConfig` +
`config_for_node_count` (`egraph/saturate.rs:150-215`) *is* `Budget::Production`.
`SaturationStop` (`graph.rs:126`), `AppBudgetSaturationStats` (:165),
`saturate_until_applications` (:1024) and `ApplyResult::truncated` (:113) are
merged on `claude/rule-order` and `claude/phase3-guide` and are exactly
`Budget::Applications` plus the typed stop reason. Integration, not invention.

**The exception, and it is a real behavior change — read this before writing
the "byte-identical output" claim.** `SaturationConfig` carries
`hard_timeout` (`saturate.rs:151`), set to **10 ms / 50 ms / 200 ms** for
blitz/rapid/classical (:161,170,179). `saturate_with_limits` breaks on it
(`graph.rs:846-848`):

```rust
for _ in 0..max_iters {
    if start.elapsed() >= timeout { break; }
```

and `saturate_with_full_budget` then computes (`saturate.rs:122`):

```rust
let saturated = stats.iterations < max_iterations || stats.total_unions == 0;
```

**A timeout stop is reported as `saturated: true`.** Two consequences:

1. **Today's compiler output is machine-load-dependent.** This machine was at
   load 15.3 while this audit ran; a 10 ms blitz budget truncates differently
   run to run. So does CI.
2. **"Byte-identical output before and after" is not currently a provable
   claim**, and the before/after arena-hash fixture the API PR needs *must*
   pin `hard_timeout` to infinity for the comparison — otherwise it produces
   spurious diffs and, worse, can hide a real one behind noise.

This is the strongest single argument for the `Budget` enum. It is also a
behavior change, so **it ships in its own PR ahead of the API PR** (§4, step
0). In the new surface, wall clock survives only as:

```rust
/// A fail-loud ceiling, never a budget dimension. Exceeding it is a bug
/// in the budget, so it **panics** rather than silently truncating and
/// reporting success. Default: none.
pub fn hard_ceiling(self, d: Duration) -> Self;
```

**The class ceiling is already on the surface** (#1105), and `Budget` must
respect it rather than restate it:

```rust
/// The ceiling on every class budget: no saturation driver grows the
/// graph past this many e-classes, whatever cap its caller asks for.
/// Memory protection for call sites that pass no meaningful cap; two
/// orders of magnitude above the production caps, so no shipping
/// configuration meets it. Approximate by a bounded amount, on purpose —
/// at the driver it guards memory, not meaning (§1.L1).
pub const HARD_CLASS_LIMIT: usize = 100_000;   // pixelflow_search::egraph
```

Every `Budget` variant's class figure is clamped to it by the drivers it
lowers to, so `Budget::Explicit { classes: usize::MAX, .. }` is a legal way
to say "as many as the ceiling allows" and gets a truthful
`SaturationStop::ClassCap` when it stops. `Budget` should NOT re-clamp: one
enforcement point, at the growth decision, is the whole shape of the L1 fix.

### 2.3 `SaturationGuide` — EXISTS on `claude/phase3-guide`

```rust
/// An ordering policy over the rule applications available this round.
///
/// A guide may only **order and truncate**. It mints no nodes and performs
/// no unions, which is why any guide is sound by construction (§1.L4) and
/// a guide PR argues quality, never correctness.
pub trait SaturationGuide {
    fn score_candidates(&self, candidates: &[CandidateSummary]) -> Vec<f32>;
}
```

Verbatim from `pixelflow-search/src/nnue/guide/mod.rs:197`, with
`CandidateSummary` at :134. Production passes `None`. Drop-in.

### 2.4 `Reranker` — EXISTS on main

```rust
pub trait Reranker {
    fn score(&self, extraction: &Extraction, arena: &ExprArena) -> f64;
}
```

`extract.rs:264`, with `IncrementalExtractor` (:280) driving it and **no
implementation shipped** — the only impl is the test-only `TableReranker` at
:1749. #1093 kept exactly the right seam. G2 (the schedule-cost residual,
`docs/plans/2026-09-01-schedule-cost-model-denotation.md`) really is a
one-line injection.

Per §1.L5, this is also the seam for *any* objective wanting an argmin
guarantee, since `extract_dag` does not provide one.

### 2.5 `Observer` — HALF EXISTS

```rust
/// A sink for what saturation did. Optional: production passes `None` and
/// records nothing.
///
/// Every *credit definition* — strict, tight, output-class, return-to-go,
/// leave-one-out counterfactual — is computed by the consumer from these
/// records. None of them live in this crate. A library that ships a menu
/// of credit bounds is a library that has to review each one.
pub trait Observer {
    fn on_application(&mut self, rec: &ApplicationRecord);
}

pub struct ApplicationRecord {
    pub rule: RuleId,
    pub ordinal: u64,
    pub round: u32,
    pub minted: /* node ids */,
    pub unions: /* union events */,
    pub changed: bool,
}
```

`ApplicationRecord` exists (`egraph/provenance.rs:~160`) carrying
`rule_idx`/`step`/`match_root`. It does **not** carry minted node ids, unions
performed, or a `changed` flag — those live in `Origin` and `UnionEvent`,
reachable but not in the record. Consolidating them is mechanical.

**Blocked, and this is the one true coupling in the whole design.** The guided
loop's application budget *is the provenance log's length*
(`egraph/saturate.rs:439,455,547`):

```rust
if egraph.provenance().application_count() >= max_total_applications {
```

Provenance is not observational there — it is the budget counter. So
`Budget::Applications` and `Observer: Option` **cannot both ship** until
`EGraph` owns an application counter independent of the log. That is a small,
separable change (a `u64` on the graph) and it is step 1 of the migration.

The prize: provenance recording is **unconditional today with no production
consumer** (integration audit, #1079), and #1087 measured production at a
**median 8 446 applications per kernel** — that is how large a log every
production compile builds and discards.

### 2.6 `Optimizer` — NEW, and it earns its keep by collapsing five call sites

```rust
pub struct Optimizer {
    rules: RuleSet,
    budget: Budget,
    cost: CostModel,
    guide: Option<Box<dyn SaturationGuide>>,
    rerank: Option<Box<dyn Reranker>>,
    observer: Option<Box<dyn Observer>>,
}

impl Optimizer {
    /// The production configuration: `RuleSet::production()`,
    /// `Budget::Production`, `CostModel::latency_prior()`, no guide,
    /// no reranker, no observer. Byte-identical to what the three
    /// production call sites do today.
    pub fn production() -> Self;

    pub fn run(&mut self, arena: &ExprArena, root: ExprId) -> Optimized;
}

pub struct Optimized {
    pub arena: ExprArena,
    pub root: ExprId,
    pub stats: OptimizerStats,   // typed SaturationStop, applications, classes
}
```

**Three production call sites collapse into it**, and one of them is
currently, quietly, wrong:

| site | today | after |
|---|---|---|
| `pixelflow-compiler/src/optimize.rs:143-160` | `config_for_node_count` → `saturate_with_full_budget` → `policy.choices()` → DAG | `Optimizer::production().run(..)` |
| `pixelflow-search/src/runtime.rs:143-170` | byte-for-byte the same four steps, differing only in `.extraction()` vs `.choices()` on the same `Extraction` | same |
| `pixelflow-compiler/src/ir_bridge.rs:720-724` | **`eg.saturate()`** — hardcoded 100/10 000/500 ms, ignoring `config_for_node_count` — then **`extract(&eg, root, &CostModel::default())`**, the *tree* extractor | same |

The third one is the argument. `extraction.rs:1-10` claims "one policy, two
tiers"; there is a **third tier** (the `Dwrt` path) that uses a different
budget and **skips DAG/CSE extraction entirely**. The cost model happens to
agree (`default() == new() == latency_prior()`, `cost.rs:184-201`); the budget
and the extractor do not. Under `Optimizer::run` that divergence stops being
possible by construction. #1085 is already doing this unification by hand —
see §4.

**Two research sites converge on it too**: the anytime runner
(`anytime.rs::AnytimeStepper`) and the guided loop are both already shaped as
"advance under a budget, extract, score".

### 2.7 What is deliberately NOT in the surface

Subtract before you add. Each of these was considered and is out, with the
reason:

- **No whole-graph accumulator.** The `GraphAccumulator` VSA is a *feature
  encoder for one particular policy*. It belongs to whoever implements
  `SaturationGuide`, not to the trait. Putting it in the surface would make
  every future policy pay for one policy's representation.
- **No cost persistence, no cost-model registry.** `CostModel` is a value the
  caller constructs. There is no global, no lazily-loaded table, no
  process-wide default beyond `production()`.
- **No env-var policy.** Already effectively gone: `env_extraction_policy()`
  (`egraph/extraction.rs:74-76`) is `ExtractionPolicy::latency_prior()`, a
  constant. The **name is vestigial and should go with the API PR** — a
  function named for an environment variable it no longer reads is a comment
  that lies. (The NNUE weights env var remains where CLAUDE.md documents it,
  at proc-macro expansion, and is out of scope here.)
- **No credit/label definitions.** Strict, tight, output-class,
  return-to-go, counterfactual — all of them are functions of
  `ApplicationRecord`s and all of them live in the consuming research crate.
  The library ships the records, not the interpretations.
- **No feature schema.** Same reason. The library ships `CandidateSummary`;
  what a policy encodes from it is the policy's business (G6 puts the
  *fingerprint* in the surface, not the schema).
- **No membership predicate for L3.** §1.L3 — a test, not an API.
- **No `build_rule_set` / `RuleOrder` in the public surface.** It is a
  research harness (§4, G5). `RuleSet::production()` is the only ordering
  production names.

**Minimal-API discipline, and a hygiene item that undercuts it.**
`pixelflow-search/src/lib.rs:1-3` is:

```rust
#![allow(clippy::all)]
#![allow(warnings)]
#![allow(unused)]
```

The `clippy --workspace --all-targets -D warnings` gate is **vacuous for this
crate**. That is how `build_rule_set` became a `pub` re-export with no
non-test caller, and how `runtime.rs:26` acquired a module-level import used
only from the test module at :1982/:1991. Removing those three attributes is
not in the API PR — it will surface a large backlog — but **it should be a
tracked follow-up**, because "every new `pub` item needs a caller" is not
enforceable in a crate where the compiler has been told to be quiet.

---

## 3. The gap table

The deliverable JP asked for: what future work sits behind this abstraction,
and how trivial each one is to integrate once it lands.

| | Gap | Needs from the API | Needs outside the API | Consumer | Status today | The one-line injection |
|---|---|---|---|---|---|---|
| **G1** | **Guide** (learned saturation ordering) | `guide: Option<Box<dyn SaturationGuide>>` | the trained checkpoint; a quality measurement at matched budget | `phase3-*` research arms | trait + `CandidateSummary` **exist** on `claude/phase3-guide`; production has no `None` to pass because there is no field | `.guide(Some(LinearCandidateGuide::load(p)?))` |
| **G2** | **Schedule-cost residual** (non-additive cost) | `rerank: Option<Box<dyn Reranker>>` | the residual model itself (`2026-09-01-schedule-cost-model-denotation.md`) | codegen scheduling | trait + `IncrementalExtractor` **exist on main** (#1093); zero impls shipped | `.rerank(Some(ScheduleResidual::new(..)))` |
| **G3** | **Deterministic application budgets** | `Budget::Applications(n)` + typed `SaturationStop` in `OptimizerStats` | nothing | every research arm; production determinism | `saturate_until_applications` + `SaturationStop` **exist** on `rule-order`/`phase3-guide`; **blocked** by the provenance-as-counter coupling (§2.5) | `.budget(Budget::Applications(8_446))` |
| **G4** | **Observation / labels** | `observer: Option<Box<dyn Observer>>`; `ApplicationRecord` gains `minted`/`unions`/`changed` | every credit definition, in the research crate | `gen_strict_labels`, `tightened_labeler_rank`, `guide_headroom` | provenance is **always on** with no production consumer (#1079); median 8 446 records/kernel wasted (#1087) | `.observe(Some(&mut recorder))` |
| **G5** | **Rule identity** — **CRITICAL** | `RuleId` + `RuleSet::fingerprint()` | discriminating `name()` on 11 rule families | Guide checkpoints, per-rule tables, JSON reports | `rule_idx` is **positional**; two queued changes move every index | `RuleId` replaces `usize` in `CandidateSummary`, checkpoints, and report keys |
| **G6** | **One feature/observation schema** | `RuleSet::fingerprint()` + a schema fingerprint on the emitted records | the shared encoder, in the research crate | live loop **and** offline minter | three train/deploy skews already bit us; nothing is fingerprinted | records carry `{rules: Fingerprint, schema: Fingerprint}`; the loader refuses a mismatch, loudly |
| **G7** | **Whole-lattice kernel** (much larger arenas) | **nothing** — `Budget` and `Guide` are already what keep compile bounded | the lattice work itself | `pixelflow-core::Lattice::bake` | n/a | `.budget(Budget::Explicit{..})` — a *value*, not an API change |
| **G8** | **Configuration fingerprint in the JIT cache** | `RuleSet::fingerprint()` + `CostModel` id + `Budget` | key `jit_cache` on it | `pixelflow-codegen::jit_cache` | **NOT KEYED** — see §5 | append the optimizer fingerprint to `canonical_key`'s bytes |

**G7, answered plainly since it was asked as a question:** nothing in this
surface is sized to the kernel. `Budget` is expressed in iterations, classes,
and applications; `Guide` scores candidates; `Reranker` scores extractions;
`Observer` receives applications. A ten-times-larger arena changes the
*values* passed, not the *types*. **The whole-lattice kernel needs no API
change.** The one thing it does expose is §1.L5: at that size, `extract_dag`'s
tree-costing of a shared DAG and its single-pass cycle handling stop being
academic, and `Reranker` becomes the practical answer rather than a seam.

---

## 4. Migration

### 4.1 The G5 ordering ruling, stated plainly

Two changes are queued that both renumber `all_rules()`. The question is
whether either must wait for stable `RuleId`s. They have **different**
answers.

**The numeric-first reorder (`claude/rule-order`) is SAFE to land now.**
Verified: `build_rule_set(RuleOrder::Production)` returns `super::all_rules()`
**verbatim** (`egraph/rule_order.rs:51-57`), and `build_rule_set` has **zero
non-test callers** — the only uses are `rule_order.rs`'s own tests (:222-247)
and `runtime.rs`'s test harness (:1982,1991), reached through a module-level
import at :26 that the crate's `#![allow(warnings)]` hides. It is a research
harness. It does not change production ordering, so it renumbers nothing that
production or any checkpoint reads. **Land it.**

**The 33-rule batch is the blocker.** It changes `all_rules().len()` from 62
to 95, and that propagates:

- `NUMERIC_FIRST_ORDER: [usize; 62]` (`rule_order.rs:108`) — **fails at
  compile time**, loudly, plus its permutation test (:130) and its
  re-derivation test (:215). Good.
- `LinearCandidateGuide`'s dense `w_rule` (`nnue/guide/linear.rs:190,251`) —
  `w_rule.get(idx)` panics **only when the table is shorter**. Adding 33 rules
  makes every existing checkpoint shorter, so this one is loud too — *by luck*.
  A same-length **reorder** is silent, and that is the real hazard.
- `PerRuleRateGuide`'s `rate: Vec<f32>` (`linear.rs:~285`,
  `eval_control_guides.rs:135`) and `rule_embeds[target.rule_idx]`
  (`saturate.rs:~510`) — same shape: length-checked, not identity-checked.
- `gen_strict_labels`'s JSON key (`bin/gen_strict_labels.rs:231`);
  `guide_headroom.rs:580` and `tightened_labeler_rank.rs:679` JSON — these
  emit **name and idx**, but the name aliases (§2.1), so name alone does not
  save them.
- `RuleTemplates::build(idx, ..)` (`egraph/mod.rs:~128`).
- `CandidateKey.rule_idx`, `ApplicationRecord.rule_idx`, `UnionEvent.rule_idx`
  — in-memory only, safe.

**Ruling: land stable `RuleId`s before the 33-rule batch, not before the
reorder.** The batch's damage is mostly loud today by accident of length; a
future same-length change would be silent, and "loud by accident" is not a
property to plan around.

Also fix in passing: `egraph/mod.rs:97` still documents "40 math + 2 fusion =
42 total". The real count is 62, pinned by `math/mod.rs:457`.

### 4.2 The sequence

**Step 0 — `claude/prod-budget-determinism` (NEW PR, behavior change).**
Remove `hard_timeout` from the production budget path; make wall clock a
fail-loud `hard_ceiling` that panics. This is the one change that is *not*
behavior-preserving, and it must go first because the API PR's byte-identical
fixture cannot be built on top of a load-dependent optimizer (§2.2). Ship it
with a determinism test: the same arena, saturated N times under load, must
produce the same arena hash.

**Step 1 — application counter off the provenance log.** A `u64` on `EGraph`,
replacing `provenance().application_count()` at `saturate.rs:439,455,547`.
Unblocks G3 + G4 (§2.5). Small; can ride with step 0 or step 2.

**Step 2 — `RuleId` (G5).** Discriminating `name()` on the 11 aliased
families, following `power.rs:255`'s pattern; `RuleId`; `RuleSet`;
`fingerprint()`. Fixes `match_counts` aliasing as a side effect. Add the
"62 rules, 62 distinct ids" test.

**Step 3 — the API PR.** `Optimizer`, `Budget`, and the three optional
levers; the three production call sites collapse onto `Optimizer::production()`.
**No behavior change.** Proof obligation: extract a fixed set of arenas —
#1087's production glyph/cell-grid arenas, or the `shader_bench` kernels —
before and after, and compare **cost and arena hash**. With step 0 landed,
that comparison is meaningful.

**Step 4 — the branches rebase onto it.**

| branch / PR | verdict | what it has to change |
|---|---|---|
| `claude/saturation-telemetry` **#1087** | **land before** | the measurement that motivates `Budget::Applications`; independent |
| `claude/integration-audit` **#1079** | **land before** | docs only; motivates `Observer` |
| `claude/rule-order` (no PR yet) | **land before** — harness-only, verified safe (§4.1) | none |
| `claude/dwrt-unify` **#1085** | **land before**, then the API PR subsumes it | it is already unifying the third call site by hand; landing it first shrinks step 3 |
| `claude/saturation-telemetry-flag` **#1083** | **rebase onto** | the feature flag becomes `observer: Some(JsonlObserver)` — the flag was standing in for the optional field |
| `claude/phase3-guide` **#1084** | **rebase onto** | `SaturationGuide` moves under `.guide(..)`; drop the parallel budget plumbing (now `Budget::Applications`); `CandidateSummary.rule_idx` → `RuleId` |
| `claude/phase3-round2` **#1088** | **rebase onto** | results doc; re-key per-rule tables by `RuleId` |
| `claude/phase3-r2g` **#1096** | **rebase onto** | the R2G credit definition moves to the research crate consuming `ApplicationRecord`s |
| `claude/phase3-domain-shift` **#1091** | **rebase onto** | results doc; no code |
| `claude/phase3-label-constfold` **#1095** | **rebase onto** | a label test; re-key by `RuleId` |
| `claude/phase3-context` (no PR) | **rebase onto** | same as `phase3-guide` |

**Housekeeping blocking all of it:** `git fetch` in the main checkout is
currently **broken** by Finder-duplicated refs under
`.git/refs/remotes/origin` — `main 3`, `gh-pages 3`, and six `claude/* 2`
(`integration-audit`, `register-allocators-trait-k8rzn3`, `phase3-guide`,
`saturation-telemetry`, `phase3-round2`, `dwrt-unify`), producing
`fatal: bad object refs/remotes/origin/claude/dwrt-unify 2`. Delete them.
`phase3-guide` additionally carries 15 tracked duplicate files including four
live `src/bin/*.rs` Cargo targets — the exact hazard the workflow doc's §0.5
already flagged, and it must be swept before that branch rebases.

---

## 5. Versioning

JP's position: the kernel ABI is unstable across upgrades and will be
versioned if promised. That makes the question narrow and answerable — **what
must an optimizer configuration fingerprint cover, and does anything key on it
today?**

**What the fingerprint must cover.** The optimizer is a function
`(arena, root, shape) → code`. Everything else in that function's closure is
configuration, and all of it changes the output:

1. **`RuleSet` content and order.** Order, not just content: saturation is
   order-sensitive under a budget, so the same 62 rules in a different
   sequence can extract a different (equally valid, differently priced) term.
   This is `RuleSet::fingerprint()`.
2. **The cost table.** `CostModel::latency_prior()`'s per-op cycles are data;
   a retune changes extraction.
3. **The budget.** Under L2 a smaller budget yields a subset of the
   equalities, hence possibly a different (still correct) extraction.
4. **Guide and reranker identity**, when present — including the checkpoint's
   own hash, since two checkpoints of the same policy are different functions.

Note what is *not* in it: the guide's presence changes cost, not meaning
(§1.L4). The fingerprint exists so a **cache** does not serve one
configuration's code to another, not because correctness depends on it.

**Does anything key on it today? No — and this is gap G8.**
`pixelflow-codegen/src/jit_cache.rs` keys on exactly two things:

```rust
let Some(mut key) = canonical_key(arena, root) else { /* uncacheable */ };
key.extend_from_slice(&shape.key_bytes());
```

— the canonicalized reachable subgraph **as handed in, before optimization**,
plus `LatticeShape`. The module comment states the justification:

> Keyed on the arena *as handed in*, before optimization, plus the shape.
> Optimization is a deterministic function of those two, so equal inputs
> yield equal output and a hit skips the saturation as well as the codegen.

**That premise is false twice.** First, it is false *today*: optimization is
not a deterministic function of `(arena, shape)` while `hard_timeout` is a
budget dimension (§2.2) — it is also a function of machine load. Two
constructions of the same kernel in one process can legitimately produce
different code, and the first one wins the cache forever. Second, it becomes
false in a *new* way the moment any of G1/G2/G3 is exercised: a process that
compiles some kernels with a guide and some without, or that changes budget
between a warm-up and steady state, will serve the wrong entry.

The cache is process-local (`static CACHE: OnceLock<..>`), so this is not a
cross-version persistence bug today — it is a *within-process configuration*
bug, and exactly the one the research levers are about to create.

**The fix is one line and belongs with the API PR**, because that is when the
configuration becomes a first-class value:

```rust
key.extend_from_slice(&shape.key_bytes());
key.extend_from_slice(&optimizer.fingerprint().to_bytes());   // G8
```

and the module comment's claim gets restated honestly: *optimization is a
deterministic function of the arena, the shape, and the optimizer
configuration* — which, after step 0, it actually is.

If the JIT cache ever persists across process boundaries, the same
fingerprint is what a version check keys on, and it will already be there.

---

## Summary of what this doc commits to

- **Two behavior changes ship first**, each in its own PR: the wall-clock
  budget (§2.2, step 0) and the `add` sentinel (§1.L1). Both replace a silent
  failure with a loud one. The `add` sentinel **has shipped** (#1105): the
  limit moved to the growth decision, `add` is total, and L1/L2 are pinned.
- **`RuleId` before the 33-rule batch; the numeric-first reorder is safe now**
  (§4.1, verified).
- **The API PR is mechanical** — three production call sites collapse to one,
  three of five types already exist, byte-identical output provable once step
  0 lands.
- **Five tests did not exist and should**: a rule-witness test (L1), a budget
  refinement ladder (L2), a membership round-trip (L3a), **N-guides-same-
  denotation (L4)**, and a 62-distinct-`RuleId`s test (G5). The L4 one is
  worth more than any per-Guide correctness review, and it is about ten lines.
  **Two now exist** (#1105): the budget refinement ladder, and — in place of
  the rule-witness test, which is the *enforcement* half of L1 and still owed
  — a direct soundness test that no over-budget run can assert a false
  equality.
- **`extract_dag` is not argmin** (§1.L5). Say so in the doc comment; do not
  build anything that assumes otherwise. `Reranker` is the seam for anything
  that needs a guarantee.

---

## 6. What shipped, and what did not

Added after implementation. The sections above are the design as proposed;
this one is the diff between that and the code, so a reader does not have to
reconcile them by inspection.

### 6.1 Shipped

| Piece | Where |
|---|---|
| `Rewrite::specialization() -> Option<OpKind>` | `egraph/rewrite.rs`, overridden by the 11 aliased families |
| `rule_label`, `RuleId`, `RuleSet`, `Fingerprint` | `egraph/rules.rs` (new) |
| `Budget`, `Limits`, `Observer`, `Optimizer`, `Optimized`, `OptimizerStats` | `egraph/optimizer.rs` (new) |
| `SaturationStop`, `SaturationOutcome`, `EGraph::saturate_budgeted` | `egraph/graph.rs` |
| Unconditional application counter; `set_provenance_recording` | `egraph/graph.rs` |
| `ApplicationRecord.{rule, minted, unions}` + `changed()` | `egraph/provenance.rs` |
| All three production tiers on `Optimizer::run` | `optimize.rs`, `ir_bridge.rs`, `runtime.rs` |
| L2 / L3a / L4 / determinism / observation tests | `pixelflow-search/tests/optimizer_laws.rs` (deleted) (new) |
| G5's "62 rules, 62 ids" test | `egraph/rules.rs` |
| G8 for the runtime cache | `runtime.rs` — `pixelflow-codegen::jit_cache` still unkeyed |
| Telemetry re-pointed at `OptimizerStats` | `telemetry.rs` — `hard_timeout_us` → `max_applications` |

`RuleId` is derived from `(name, specialization)` rather than from eleven
hand-written decorated-name tables. `Commutative` answers `Some(Mul)` where
`LogPower` would have needed `"commutative(mul)"` spelled out; there is no
per-family string table to forget an arm of, and no fallback that could
silently re-alias an operator nobody enumerated. That is the subtraction §2.1
asked for, arrived at differently.

`match_counts` is re-keyed from `String` to `RuleId`, which fixes the
four-`Commutative`s-in-one-bucket aggregation and removes a `String`
allocation per match from the scan loop. Its only in-tree consumers take
`.values().sum()`, so nothing downstream had to change.

### 6.2 Not shipped, deliberately

**No `guide` field and no `SaturationGuide` trait.** The trait is trivial;
the loop that reads it is not, and it lives on `claude/phase3-guide`. A
`guide: Option<..>` field on an optimizer whose saturation loop never
consults it would accept a policy and ignore it — a silent failure, and the
one thing this codebase's rules forbid outright. So the field lands with the
loop, in #1084, and what this PR owes that PR is everything *around* the
field: the struct to hang it on, `Budget::Applications` as the matched-budget
currency to measure it at, `RuleId` to key its checkpoint by, and the L4 test
whose `POLICIES` table it extends rather than re-deriving. §2.3's claim that
the trait is "drop-in" is correct; §3's G1 "one-line injection" is one line
*plus the loop*, and the doc was optimistic to imply otherwise.

**`Reranker` is a real field and is honored**, routing through
`IncrementalExtractor`. Its bound was relaxed to `?Sized` so a
`Box<dyn Reranker>` works. G2 genuinely is one line.

**The `HARD_CLASS_LIMIT` sentinel (§1.L1) was fixed elsewhere.** It is a
behavior change of its own — `add` returning `EClassId(0)` over budget, and
`RewriteAction::Create` then asserting a false equality against it — so it
was correctly kept out of a no-behavior-change PR. #1107 landed it
independently while this one was in review, which is the sequencing §4 wanted
and leaves §1.L1 closed rather than deferred. One consequence lands here: the
`max_classes.min(HARD_CLASS_LIMIT)` clamp #1107 put at the top of
`saturate_with_limits` moves into the shared `saturate_bounded` loop, so
`saturate_budgeted` — and therefore all three production tiers — is held to it
too rather than inheriting it by accident of which entry point they used.

**`ir_bridge` needed no special case, because #1085 landed first.** §2.6
found that tier on a hardcoded 100/10 000/500 ms budget and the *tree*
extractor; #1085 moved it onto `saturate_for_extraction` plus the DAG
extractor by hand, which is exactly what §4 sequenced. So all three tiers were
already uniform when this PR arrived, and routing them through
`Optimizer::production()` is a substitution rather than a convergence. That is
why the byte-identical claim covers the `Dwrt` tier too.

`ir_bridge`'s own tests get smaller as a side effect. They carried an
`UNTIMED_CEILING` constant — a 120-second substitute for the tier's
`hard_timeout`, there because `cargo test` builds the crate unoptimized and
shares the machine, so asserting anything under `rapid`'s 50 ms would have
pinned the machine's load rather than the policy. `Budget` takes no clock, so
there is nothing left to substitute: the production configuration *is* the
untimed one, and those tests now exercise it unmodified.

**`ExtractionPolicy` went with `env_extraction_policy`.** §2.7 called for
deleting the vestigial name; the type it named had no remaining
responsibility once `Optimizer` owned the cost model and the lattice, so the
whole module went. Its two test callers now drive `Optimizer`, which is
strictly better — they test the production path instead of a parallel one.

### 6.3 Equivalence, measured

`optimize_runtime_arena` over the twelve `shader_bench` kernels, release
build, digesting the extracted arena. A digest is a stronger check than a
cost: equal arenas have equal cost under every model, while equal costs can
hide a different term.

| arm | combined digest | runs |
|---|---|---|
| `origin/main` @ `c1afd4b9` | `66efbe1a7133c5f4` | 3/3 identical |
| this branch | `66efbe1a7133c5f4` | 3/3 identical |

**All twelve kernels match individually**, not just in the fold — same
extracted node count, same digest, kernel by kernel. Timings are the same
order on both arms (main 9.5–101 ms, branch 4.2–122 ms per kernel, on a
machine at load ≈ 21 where that spread is noise).

Per-kernel, with the budget the run actually spent — read off
`OptimizerStats`, which is new surface this PR adds and the reason these
numbers are available at all:

| kernel | in | out | applications | classes | stop |
|---|---:|---:|---:|---:|---|
| cosine_palette | 40 | 26 | 2 614 | 1 767 | ClassCap |
| smooth_min_scene | 43 | 38 | 2 722 | 1 219 | ClassCap |
| mandelbrot_distance | 152 | 109 | 15 303 | 3 716 | ClassCap |
| star_sdf | 66 | 57 | 7 997 | 3 828 | ClassCap |
| gyroid_slice | 44 | 35 | 8 652 | 779 | Quiesced |
| plasma | 41 | 31 | 3 310 | 1 822 | ClassCap |
| domain_warp_fbm | 84 | 59 | 6 976 | 4 577 | ClassCap |
| kaleidoscope_fold | 46 | 40 | 601 | 131 | Quiesced |
| metaballs | 62 | 48 | 10 090 | 2 584 | ClassCap |
| julia_set | 122 | 122 | 14 870 | 4 527 | ClassCap |
| smoothstep_vignette | 64 | 45 | 1 596 | 283 | Quiesced |
| torus_slice | 42 | 37 | 4 697 | 1 291 | ClassCap |

**Not one `Timeout`.** Nine of twelve stop on the class cap and three
quiesce, which is why dropping the wall clock costs nothing here and why the
output is unchanged: on this corpus the clock was never what bound. That
independently reproduces #1087's finding (class cap binding on 68 % of its
kernels; 75 % here) on a different corpus, and it is what makes
`Budget::Production` a faithful reproduction of the production presets rather
than a redefinition of them.

Two cautions against over-reading it.

1. **This is not proof the clock can never bind** — only that it does not on
   these twelve at this load. An earlier run of the same harness against
   `origin/main` @ `6336a0c2`, *before* #1085 landed its `ScanStop` work,
   produced **five different digests in five identical runs** and truncated
   three of the twelve kernels mid-saturation. The clock is a live hazard the
   moment a kernel's sweep outruns it; removing it is what makes that
   unrepresentable rather than unlikely.
2. **Applications range 601 – 15 303**, median ≈ 5 800. That is the
   calibration table for `Budget::Applications` when a research arm needs two
   policies held to the same spend, and it is the number G3 was missing.

Wall clock survives only as `Optimizer::hard_ceiling`, which **panics**. A
ceiling is an assertion about the budget; the old `hard_timeout` truncated and
then reported `saturated: true`, which is the same failure wearing a success's
clothes.
