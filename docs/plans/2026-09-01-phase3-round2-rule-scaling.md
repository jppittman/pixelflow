# Phase 3 Round 2: does the Guide's advantage grow with the rule count?

**Date:** 2026-09-01
**Status:** DESIGN — pre-registration protocol for Round 2. The numeric margins this document
names as "fixed in Register" do not exist yet; they are computed from UNGUIDED data only and
committed in a separate Register document (`docs/plans/2026-09-0X-phase3-round2-registration.md`)
before any guided run at |R| > 62. Once that Register is committed, neither it nor this document's
statistics may be revised except to append results.
**Authority:** `docs/plans/2026-08-31-guide-design-revision.md` (§0 framing, §5 protocol);
`docs/plans/2026-09-01-phase3-registration.md` (Round 1 — FROZEN, never edited, its §9 results are
this round's starting point); JP's thesis, verbatim (2026-09-01):

> "We're gonna add LOTS of rules to this thing. We're gonna build a compiler, and the egraph is
> gonna be basically the only optimizer. It's gonna compete with compilers made by compiler
> experts using the egraph and learned heuristics to circumvent the pass ordering problem. As we
> add more rules, the latency and the performance join. The scaling problem is the fidelity
> problem."

Round 1 showed that at a fixed application budget in the classical band, a cold-start linear Guide
halves the cost of unguided saturation (median ratio 0.537 at B=100, 0.696 at B=200, DEV n=334),
and that most of that is per-rule base rates plus candidate-key deduplication. Round 1 was run at
one rule count, |R| = 62. The thesis is a statement about the *derivative* with respect to |R|:
unguided saturation's quality-at-budget should degrade as the rule library grows (the sweep spends
its budget on more matches, most of them idle), and a Guide's ordering should degrade slower — so
the Guide's advantage is *increasing* in |R|. Round 2 turns that into two falsifiable hypotheses
and a protocol that fixes every free parameter from unguided data before any guided number exists.

Binding rules carried over from Round 1, unchanged: budgets in recorded rule applications, never
wall-clock (a per-curve safety ceiling exists and PANICS when it binds); the one curve definition
is `pixelflow_search::egraph::anytime::run_anytime_curve_with`; DETERMINISTIC cost regret under
`CostModel::latency_prior()`; no timing anywhere in any metric; FINAL untouched; family-held-out
tiers; registration from UNGUIDED data only, committed before any guided run at |R| > 62.

**No production behavior changes in this workflow.** Every inflated or experimental rule set is a
harness-selectable `Vec<Box<dyn Rewrite>>` handed to `EGraph::with_rules`. Nothing is added to
`all_rules()` (62, pinned by test). Adoption of any new rule into production is JP's decision,
made after the measurement, on its own PR.

## 1. Hypotheses and the exact statistics

### 1.1 Setting

- Band: classical (> 50 nodes), the only band with registered budgets.
- Budgets: B = 100 (primary) and B = 200 (secondary) — Round 1's registered tiers, unchanged.
- Rule-count grid: |R| ∈ {62, 93, 124, 186, 248} for inflation modes (i) and (ii) (§2); mode (iii)
  is the pair {62, 62 + batch} (§3).
- For each expression e, each |R|, each arm a ∈ {unguided, guided}: an anytime curve
  `cost_a(e, |R|, t)` over the fixed grid `APP_CHECKPOINT_GRID`, sampled through the ONE curve
  runner. Cost at B is read at the grid checkpoint for B (first between-sweeps/rounds point with
  cumulative applications ≥ B; `app_actual` recorded, analysis plots against it).
- Reference (closure-aware, §4): `ref(e, |R|) = min over both arms and all checkpoints of
  cost_a(e, |R|, t)` — the best either arm reaches at any checkpoint **at that |R|**. Never
  pooled across |R| points.
- Per-expression regret: `regret_a(e, |R|) = (cost_a(e, |R|, B) − ref(e, |R|)) / ref(e, |R|)`.
  Zero-cost references follow the established convention: positive cost against a zero reference
  is infinite regret, never 0%.

### 1.2 H1 — unguided regret grows with |R|

**Statistic.** `U(|R|) = median_e regret_unguided(e, |R|)` over the classical sample at B (both
B reported; B=100 carries the verdict, B=200 is secondary, exactly as in Round 1).

**Test (pre-committed, both parts must hold).**
1. Direction: Spearman rank correlation of `U(|R|)` against |R| over the 5-point grid is
   **ρ ≥ +0.9** (five points: ρ = 1.0 is strictly monotone; 0.9 tolerates one adjacent
   inversion; anything less is not "grows with |R|").
2. Minimum effect: `U(248) − U(62) ≥ Δ1`, where **Δ1 is fixed in Register from unguided data**
   as the half-width of the 95% bootstrap confidence interval (10,000 resamples, seed 42) of the
   median unguided regret at |R| = 62 on the same sample. Rationale: unguided saturation is
   deterministic, so the only "noise" on a median is sampling noise over expressions; a shift
   smaller than the median's own CI half-width is not an effect this sample can see.

H1 is entirely an unguided measurement — the Register run *is* the H1 measurement, and H1's
verdict is recorded in the Register document before any guided run. This is deliberate: H2's
margin (§1.3) is derived from H1's curve, so H1 must be settled first.

### 1.3 H2 — guided regret grows slower (the Guide's advantage is increasing in |R|)

**Statistic.** The per-expression **cost ratio at B**, `q_e(|R|) = cost_guided(e, |R|, B) /
cost_unguided(e, |R|, B)`, and its median over the held-out classical sample, `Q(|R|)`.

Why the cost ratio and not a ratio of regrets: `q_e = (1 + regret_guided) / (1 + regret_unguided)`
against the same closure-aware reference, so it *is* the ratio of guided-to-unguided regret in
`1 + regret` form — but it is defined when a guided regret is exactly zero (Round 1's guided
median regret at B=100 was 0.38%, and 199/334 expressions reached the empirical best, so a raw
regret ratio would divide by zero on more than half the sample). It is also exactly Round 1's
Y-clause statistic, so Round 1's `Q(62)` = 0.537 / 0.696 is the anchor.

**Test (pre-committed, all three parts must hold).**
1. Direction: `Q(|R|)` is **non-increasing** in |R| across the grid (each adjacent step
   `Q(|R|_{k+1}) ≤ Q(|R|_k) + ε`, with ε = 0.005 to absorb ties at the third decimal — a
   pre-committed constant, not tuned).
2. Minimum effect: `Q(62) − Q(248) ≥ Δ2`, where **Δ2 is fixed in Register from unguided data**
   as `max(0.02, Y(248) − Y(62))`. Here `Y(|R|)` is Round 1's threshold formula applied to the
   unguided median truncation loss `L(|R|)` measured at that |R| (`L = (cost@B − cost@4B) /
   cost@4B`, median over classical; `Y = 1 − (1 + L/2) / (1 + L)`). Reading: the Guide's
   advantage must grow at least as fast as the "close half the truncation gap" yardstick grows
   with |R|. The 0.02 floor is a pre-committed constant so the test cannot be trivially satisfied
   if H1 fails (in which case `Y(248) − Y(62)` is near zero).
3. Round 1's claim re-instantiated at every point: `Q(|R|) ≤ 1 − Y(|R|)` for every |R| on the
   grid. A Guide whose advantage "grows" only because it collapsed at |R| = 62 is not evidence for
   the thesis.

**Slope ratio, reported not tested.** Fit `U(|R|)` and `G(|R|) = median_e regret_guided(e, |R|)`
by least squares against |R| and report `slope_G / slope_U`. H2's plain-language statement is
"slope ratio < 1"; the tests above are the falsifiable form (a slope ratio on five medians with
one near zero is not a stable statistic to gate on). Both slopes and their bootstrap CIs are
reported alongside.

### 1.4 What the sign of each answer means

| H1 | H2 | Reading |
|---|---|---|
| holds | holds | The thesis as stated: the scaling problem is the fidelity problem, and learned ordering is the lever. |
| holds | fails | Unguided degrades but the Guide degrades just as fast — rule count is a problem the Guide as built does not solve (capacity or label semantics, §7). |
| fails | holds | Unguided does not degrade at these |R| (the sweep is cheap enough); the Guide still wins more with more rules — a weaker, still useful result. |
| fails | fails | At this scale of |R|, rule count is not the axis; report as a null with the absolute-cost fidelity table (§4) as the round's deliverable. |

Each inflation mode gets its own verdict row; §2 says what each mode isolates.

## 2. Three inflation modes, each isolating one thing

All three modes produce a rule vector whose first 62 entries are `all_rules()` in its exact
production order (so rule indices 0..62 mean the same thing at every |R| point), followed by the
inflation. Nothing is inserted or reordered inside the prefix.

### 2.1 Mode (i): exact duplicates — the "learnable overhead" control

The inflation is copies of existing rules under new rule indices: a `DuplicateRule` wrapper that
delegates `apply`/`is_destructive`/templates to the original and reports
`name = "<original>#dup<k>"`. Same closure (a duplicate can create nothing the original cannot),
same per-match closure semantics, purely more matches per sweep. Copies are appended in cycles:
cycle 1 = every rule in index order (|R| = 124), cycle 2 (186), cycle 3 (248); the half-cycle
point |R| = 93 takes the even indices 0, 2, …, 60 (31 rules, spread across every module rather
than the first 31, which would all be algebra rules).

**CandidateKey decision.** `CandidateKey` is `(rule_idx, canonical class content)`
(`pixelflow-search/src/egraph/candidate.rs`). A duplicate is therefore a DISTINCT key from its
original, and `GuidedSaturation`'s dedup-before-scoring does NOT collapse it: the copy is
enumerated, scored, and — if the Guide does not rank it below the useful work — fired as a
recorded (idempotent) application that spends budget. Two possible semantics:

- **Keep `rule_idx` in the key (chosen).** The Guide must *learn* that the copies are worthless.
  It can: in unguided label minting the first copy in sweep order creates the node and takes the
  strict credit, every later copy re-fires idempotently and is labeled `Wasted` — so per-rule
  strict-positive rates for indices ≥ 62 are near zero, and both the per-rule control and the
  linear Guide will rank them last. This is the honest model of the real future: added rules are
  never literally duplicates, but many are near-redundant with existing ones (a composition or a
  specialization), and the Guide's job is exactly to learn which of the many ways to reach a
  node is worth paying for. Under this semantics mode (i) measures *learnable* overhead.
- **Content-only key (`canonical class content` alone).** The dedup would collapse the copies
  before scoring, the Guide would never see them, and mode (i) would measure nothing about the
  Guide at all — only about the dedup set. It is the cleaner "pure overhead" control, but it
  would also be wrong for real rules: two different rules matching the same class content are
  different candidates that create different nodes.

**Pre-committed: Round 2 keeps `rule_idx` in the key.** The duplicate arm is reported as the
"learnable overhead" control; the content-only variant is NOT run in Round 2 (recorded here as a
possible later diagnostic, not a pre-registered arm — adding it after seeing results would be a
post-hoc arm).

**What (i) predicts, stated so it cannot be re-read after the fact.** Unguided: the sweep fires
every rule in index order and every copy's applications are recorded, so at fixed B the useful
work shrinks roughly as 62/|R| — H1 is close to analytic in this mode, and a *failure* of H1 here
would indicate a harness bug, not a finding. Guided: if the Guide learns the copies are worthless
they sit at the bottom of every round's ranking and `Q(|R|)` is nearly flat — that flatness is
the "learnable" claim, and its failure would be informative (the Guide did not learn what a base
rate makes trivially learnable — pointing at the per-index capacity threat, §7.2).

### 2.2 Mode (ii): mechanical compositions — same closure, real state changes

The inflation is derived rules A∘B built from pairs of existing rules that expose LHS/RHS
templates (`Rewrite::lhs_template`/`rhs_template`; 30 of the 62 do today: 14 algebra, 1 parity,
4 trig, 4 exp, 5 power, 2 fusion). The remaining 32 (constant-fold, differentiate, and the
`RewriteAction`-driven rules without templates) are not composable by this generator and are
excluded from the pool — stated, not silent.

**Generator (deterministic).**
1. For each ordered pair (A, B) of templated rules, rename metavariables apart and attempt
   first-order syntactic unification of `B.lhs` with `A.rhs` at the root and at every proper
   subterm position p of `A.rhs`. A unifier σ binds A's metavariables to B-terms and/or B's to
   A-terms (both directions arise: distribute∘fma-fusion binds fma's `c` to `A*C`;
   fma-fusion's `Add(Mul(a,b),c)` against commutative's `Add(Y,X)` binds `Y := Mul(a,b)`, making
   the composed LHS *more specific* than A's).
2. The composed rule is `lhs = A.lhs σ`, `rhs = A.rhs[p := B.rhs] σ` — one rule whose single
   application creates what A-then-B would have created in two rounds. The intermediate `A.rhs σ`
   node is NOT created (that is the point: fewer applications to the same node), so the composed
   rule's closure is contained in the closure of {A, B}.
3. Filters (pre-committed): drop compositions whose lhs and rhs are α-equivalent (e.g.
   commutative∘commutative — identity, no state change); drop compositions whose rhs equals
   A.rhs or lhs (B was a no-op at that position); drop exact structural duplicates of an earlier
   composition in the enumeration order.
4. Order: the surviving pool is permuted by a seeded Fisher–Yates shuffle (seed 0x5EED2, fixed
   here), and the |R| grid takes prefixes of that permutation (31, 62, 124, 186 compositions), so
   grid points are nested (R_93 ⊂ R_124 ⊂ R_186 ⊂ R_248) and no module is favored by enumeration
   order. The pool size and the full ordered list (names, A, B, p) are written to
   `docs/results/…-round2-compositions.json` by the Register run. If the pool has fewer than 186
   members, the unreachable grid point(s) are dropped and the actual grid is recorded — never
   padded with depth-2 compositions or duplicates (that would mix modes).

**Validity argument.** A rewrite is a theorem over the reals, `∀ vars. lhs = rhs`. If A and B are
valid and σ is a unifier of `B.lhs` with `A.rhs|_p`, then for every assignment `A.lhs σ = A.rhs σ`
(A, instantiated) and `A.rhs|_p σ = B.lhs σ = B.rhs σ` (B, instantiated); substituting at position
p gives `A.lhs σ = A.rhs[p := B.rhs] σ`. A composition of valid rewrites is valid. The generator
is nevertheless not trusted on the argument alone: **every composed rule is oracle-checked by a
test** (§2.4) exactly as a hand-written rule would be, because the thing being validated is the
unifier/substitution *implementation*, not the theorem.

**Execution.** Composed rules are `TemplateRewrite`s (§8): a generic template rule whose `apply`
e-matches the LHS against `(class, node)` and whose action instantiates the RHS bottom-up. This is
the one piece of new e-graph machinery Round 2 needs; it is additive (a new `RewriteAction`
variant only harness rules produce) and pinned against production by the existing
`unguided_saturate_until_applications_stays_deterministic` test.

**What (ii) predicts.** Compositions cut both ways for unguided saturation: more matches per sweep
(H1's mechanism) but also shortcuts that reach a node in one application instead of two. The sign
of H1 in mode (ii) is therefore NOT analytic — it is the interesting arm. A measured *decrease* of
`U(|R|)` in mode (ii) is a genuine falsification of H1 for this mode and is reported as such.

### 2.3 Mode (iii): genuinely new rules — the fidelity arm

A first batch of 24 rules inventoried from what the language has (`OpKind`: Add Sub Mul Div Neg
Sqrt Rsqrt Abs Min Max MulAdd Recip Floor Ceil Round Sin Cos Tan Asin Acos Atan Atan2 Exp Exp2
Ln Log2 Log10 Pow, comparisons Lt Le Gt Ge Eq Ne, Select) and the 62 rules lack. All are
theorems over the reals on the documented domain of the ops involved (the algebraic-validity
contract, `docs/plans/2026-08-05-egraph-nnue-research-workflow.md` §0.4: divergence at
singularities is contract, never a soundness gap; no domain guards are added to make a valid
identity "safer"). Constant-guarded rules carry a *side condition on a literal*, which is a
matching condition, not an IEEE guard.

| # | Family | Rule (LHS → RHS) | Notes |
|---|---|---|---|
| N1 | min/max | `min(a,b) → neg(max(neg a, neg b))` | duality; enables max-only reasoning |
| N2 | min/max | `max(a,b) → neg(min(neg a, neg b))` | duality |
| N3 | min/max | `min(a, max(a,b)) → a` | absorption |
| N4 | min/max | `max(a, min(a,b)) → a` | absorption |
| N5 | min/max | `min(a,b) + c → min(a+c, b+c)` | translation distributes; also reverse direction as the useful one |
| N6 | min/max | `max(a,b) + c → max(a+c, b+c)` | as N5 |
| N7 | min/max | `k * min(a,b) → min(k*a, k*b)` for literal `k ≥ 0` | constant-guarded (sign of a literal) |
| N8 | min/max | `min(a, max(b,c)) → max(min(a,b), min(a,c))` | lattice distributivity |
| N9 | abs | `abs(x) → max(x, neg x)` | abs as a lattice op |
| N10 | abs | `max(x, neg x) → abs(x)` | reverse of N9 (the fusion direction) |
| N11 | select | `select(m, a, a) → a` | mask-independent |
| N12 | select | `select(lt(a,b), a, b) → min(a,b)` | over the reals; platform NaN rows differ by design (CLAUDE.md table) and no value is promised there |
| N13 | select | `select(lt(a,b), b, a) → max(a,b)` | as N12 |
| N14 | select | `select(m, f(a), f(b)) → f(select(m,a,b))` for unary f ∈ {Neg, Abs, Sqrt} | hoist: one f instead of two; one rule per op (3 indices) counted as N14a–c |
| N15 | compare | `lt(a,b) → gt(b,a)` | comparison flip (mask-domain result on both sides) |
| N16 | trig | `tan(x) → sin(x) * recip(cos(x))` | tan definition |
| N17 | trig | `sin(x) * recip(cos(x)) → tan(x)` | reverse of N16 |
| N18 | exp/log | `exp(x) → exp2(x * log2(e))` | hardware-native base |
| N19 | exp/log | `ln(x) → log2(x) * ln(2)` | as N18 |
| N20 | exp/log | `log10(x) → log2(x) * log10(2)` | as N18 |
| N21 | sqrt | `sqrt(x) * sqrt(y) → sqrt(x*y)` | on the domain of sqrt |
| N22 | sqrt/rsqrt | `rsqrt(x) * rsqrt(x) → recip(x)` | |
| N23 | sqrt/rsqrt | `x * rsqrt(x) → sqrt(x)` | the common normalize shape |
| N24 | recip | `recip(a) * recip(b) → recip(a*b)` | one estimate instead of two; reverse direction also registered as N24r |
| N25 | mul-add | `fma(a,b,c) → a*b + c` | un-fuse: re-exposes the sum to other rules |
| N26 | mul-add | `fma(a, 1, c) → a + c`; `fma(a, b, 0) → a*b` | identities on the fused op (2 indices) |
| N27 | neg | `neg(a + b) → neg(a) + neg(b)`; `neg(a*b) → neg(a) * b` | negation pushes through (2 indices) |
| N28 | div | `a / k → a * (1/k)` for literal `k ≠ 0`, reciprocal computed exactly at rule time | constant-guarded; note `Recip` itself is never folded (estimate), this replaces the *division* by an exact literal — algebraically valid, precision-changing, within contract |

That is 24 rule families; with the per-op indices (N14a–c, N26×2, N27×2, N24r) it is 31 rule
indices, so mode (iii) is the pair **|R| = 62 vs 93**, which also lines up with the first grid
step of modes (i)/(ii). If Foundations' oracle test rejects a rule (implementation error) it is
dropped and the actual count recorded; the pair is still 62 vs 62+batch.

Every N-rule is a `TemplateRewrite` where expressible; N7 and N28 (literal side conditions) and
N14 (op-parameterized) are hand-coded `Rewrite`s in the same harness module. All live in a module
that `all_rules()` does not reference. Production adoption of any of them is out of scope.

### 2.4 The oracle gate (every new rule, every composition)

Per `docs/plans/2026-08-05-egraph-nnue-research-workflow.md` §0.4, two checks, never conflated:

- **Same-form hard gate** — the scalar reference interpreter (`pixelflow_ir::eval_scalar`) against
  the JIT on the *same* arena. This is pixelflow-ir's existing parity suite; a rule cannot break it
  (rules do not touch lowering) and Round 2 does not re-run it per rule. It is listed so nobody
  mistakes the cross-form check for it.
- **Cross-form conditioned gate** — for each rule: instantiate LHS and RHS with the same random
  leaf assignment (metavariables → fresh `Var`s or small subterms), evaluate both via
  `eval_scalar` at a fixed set of well-conditioned points (seeded; finite, moderate magnitudes;
  intermediates checked non-singular), and require agreement within the per-op tolerance from
  `pixelflow_ir::eval::equivalence_tolerance` composed over the RHS. Disagreement at a
  well-conditioned point **fails the test** (an implementation error in the rule or the
  composition generator). Divergence at ill-conditioned points is recorded as metadata on the
  rule (count, sample points) — never an exclusion, never an alarm.

One test per rule family and one parametric test over the whole composition pool
(`compositions_pass_the_cross_form_oracle`), so an addition to either set without its oracle
check does not compile-and-pass silently.

## 3. The |R| grid

| Mode | |R| points | Inflation content |
|---|---|---|
| (i) duplicates | 62, 93, 124, 186, 248 | +31 (even indices), +62, +124, +186 copies |
| (ii) compositions | 62, 93, 124, 186, 248 | prefixes of the seeded pool: 31, 62, 124, 186 |
| (iii) new rules | 62, 93 | the §2.3 batch (31 indices) |

The |R| = 62 point is shared: it is `all_rules()` and is run ONCE per sample (its unguided curves
are the same data for every mode; its guided curve is Round 1's Guide re-evaluated on this
round's sample — see §5 on why it is re-run rather than copied).

## 4. Metric subtlety: the optimum moves under mode (iii)

Modes (i) and (ii) preserve the closure, so the best reachable cost of an expression is the same
at every |R| and only search efficiency varies. Mode (iii) enlarges the closure: a new rule can
reach a strictly cheaper form, so "regret" at |R| = 93 against a reference computed at |R| = 62
would be negative for the arm that found the new form — a search-quality number contaminated by
a fidelity change. Two requirements, both binding:

1. **Regret is closure-aware.** `ref(e, |R|)` is the best-either-arm-any-checkpoint cost AT THE
   SAME |R| (§1.1). It is never shared across |R| points, in any mode (harmless in (i)/(ii),
   essential in (iii)).
2. **Absolute cost is reported separately, so fidelity is visible.** For every (mode, |R|, arm):
   median and quartiles of absolute `latency_prior` cost at B, at 4B, and at curve end; and the
   **closure gain** `fid(e) = (ref(e, 62) − ref(e, 62+batch)) / ref(e, 62)` per expression
   (median, quartiles, share > 0). `fid` is what the new rules *buy* when search is not the
   bottleneck; regret is what search *loses* on the way there. The thesis says both matter and
   join; the report must show them as two columns, never one.

## 5. Sample

- **The H1 (unguided) sample is Round 1's:** the 400-expression size-stratified TRAIN+DEV sample
  of `docs/results/2026-09-01-phase3-unguided-baseline.{csv,json}` — sort all 4,143 TRAIN+DEV
  expressions by node count (name as tiebreak), stride 10.36 (= 4,143 / 400), take
  `entries[floor(i·stride)]` for i in 0..400, exactly as `phase3_unguided_baseline.rs` does.
  Composition: blitz 23, rapid 189, classical 188 (TRAIN 154 + DEV 34). H1 is measured on the
  188 classical expressions; unguided saturation trains on nothing, so TRAIN membership is not
  leakage for H1.
- **The H2 (guided) sample must be family-held-out.** Every Guide in this round is trained on
  TRAIN-family labels, so a guided number on a TRAIN expression is not evidence. Pre-committed:
  H2's verdict is taken on the **full DEV classical band (n = 334)** — the same set, same harness
  as Round 1's §9 result, and the set that anchors `Q(62)`. The 34 DEV-classical members of the
  400 sample are reported alongside as the continuity subset (n = 34 clears Round 1's n ≥ 30
  powered threshold, but its bootstrap CI on a median ratio is too wide to resolve a Δ2 of a few
  points; it is reported, not gated on).
- The corpus is the Round 1 corpus (`gen_bench_corpus --target 4000 --seed 42`; MD5s in the
  Round 1 registration §1); FINAL (`corpus_final.bin`) is not opened.
- Bands: classical carries every claim; blitz and rapid curves are produced by the same runs and
  reported per |R| **without any claim** (Round 1 §4: nothing to buy back there).
- Grid: `APP_CHECKPOINT_GRID` unchanged. Class cap: `config_for_node_count(node_count).max_classes`
  per expression, identical across arms and |R| points. Safety ceiling: per-curve wall-clock
  ceiling scaled by `|R| / 62` from Round 1's value so that more rules do not make an honest curve
  panic; it still PANICS when it binds. If the unguided Register run at |R| = 248 cannot finish
  the full grid within a practical ceiling, the Register may **truncate the grid** (drop the
  largest targets) — decided from unguided timing only, applied identically to every |R| point and
  every arm, and the |R| = 62 point re-run under the truncated grid so references are computed
  the same way everywhere. Truncation changes the reference (best over fewer checkpoints); it is
  recorded, and no grid decision is made after a guided run exists.

## 6. What Register may fix, from what data, and the gates

**Register (unguided only, committed before any guided run at |R| > 62) fixes:**

| Item | From |
|---|---|
| `U(|R|)` for every |R| and mode, B = 100 and 200 — the H1 curve and its verdict | unguided curves on the 188-classical sample |
| `L(|R|)`, `Y(|R|)` per |R| and mode | same unguided curves (truncation loss at B vs 4B) |
| Δ1 = bootstrap 95% CI half-width of the median unguided regret at |R| = 62 | unguided, |R| = 62 |
| Δ2 = max(0.02, Y(248) − Y(62)) per mode (mode (iii): Y(93) − Y(62)) | unguided |
| The composition pool (size, ordered list) and the new-rule batch's final index count after the oracle gate | generator + tests, no guided data |
| Grid truncation, if any (§5) | unguided timing at |R| = 248 |
| Rule-set fingerprints for every |R| point (names in index order, hashed) | the rule sets themselves |

Register may NOT fix or change: B, Y's formula, the checkpoint grid semantics, the reference
convention, the sample, the H2 statistics, ε, the 0.02 floor, seed values, or which mode is run.
Those are fixed here.

**Label and Guide protocol (binding).** At every |R| point of every mode, strict labels are
re-minted under THAT rule set from unguided saturation on TRAIN families
(`gen_strict_labels --rule-set …`), a cold-start linear Guide is trained on them
(`train_guide`, same hyperparameters as Round 1's `guide_checkpoint_strict_v1.json`, recorded),
and the per-rule control (`PerRuleRateGuide`) is built from the same run's TRAIN rates. No
checkpoint or label file is reused across |R| points or modes (§7.3). The `w_rule` table has
exactly |R| entries and the checkpoint carries the rule-set fingerprint; evaluating a checkpoint
under a rule set with a different fingerprint is a hard error.

**Accept gate (per mode).** H1's two-part test AND H2's three-part test hold on DEV classical
(n = 334) at B = 100, reported with full per-expression distributions (median, quartiles, p90,
per-expression JSONL) — never a single median without its distribution. B = 200 is reported as the
secondary result. Publication of the Round 2 claim requires the same on FINAL's classical band
(family-held-out; FINAL opened for the first time at that run; n < 30 → underpowered, no accept).

**Kill gate (per mode).** H2 part 3 fails at any |R| point on DEV (`Q(|R|) > 1 − Y(|R|)`: the
Round 1 claim itself stops holding as rules are added) after the same allowance Round 1 gives —
one clean re-mint/re-train round per |R| point to rule out a training defect — stop for that mode
and record it. The Guide as built does not scale with the rule count; that is a numeric,
publishable negative, and the honest fallback is the measured H1 curve with the absolute-cost
fidelity table.

**Honest fallback (pre-registered).** If H1 fails in modes (ii) and (iii) — unguided saturation
does not degrade at these rule counts — the deliverable is that finding plus the fidelity table
(§4.2): it says the sweep is not yet the bottleneck at |R| ≤ 248 and that the interesting regime
is further out. Nothing in this document permits extending the grid after seeing that result;
a larger grid is a new registration.

## 7. Threats

### 7.1 Per-candidate Guide cost must stay flat in |R|

The metric is applications, so Guide cost never enters a number — but if scoring cost grew with
|R| the Guide would be buying quality-at-budget with wall-clock, and the thesis is about both
joining, not one paying for the other. Two properties keep it flat, and both are measured:

- **Dedup before scoring.** Only survivors of the `CandidateKey` set are scored. Measurement:
  **scored candidates per recorded application** at B, per |R| — from `GuidedSaturation`'s
  episode counters (raw matches enumerated, keys deduped, candidates scored, applications
  recorded, rounds). Reported as a table by mode and |R|. Pre-committed expectation: flat to
  within 2× across the grid in mode (i) (the copies are scored once each per class content and
  never again); any growth is reported, and growth beyond 2× at |R| = 248 flags the Guide's
  per-round cost as a real scaling problem regardless of what H2 says.
- **Candidate-local features.** `CandidateFeatures` reads the matched class, its one-hop
  neighborhood, the rule index, and the budget fraction — none depends on |R|. Raw match
  enumeration (`find_rewrite_matches`) IS linear in |R| by construction: that is the sweep's cost,
  shared by both arms, and it is the mechanism H1 is about. The measurement above separates the
  two (raw matches / application will grow; scored / application must not).

### 7.2 Rule-index capacity

The linear Guide has one bias per rule index (`w_rule[rule_idx]`) and NO rule embedding in the
deployed path (`phase3_at_budget_eval` hands `GuidedSaturation` zero embeddings; the
per-rule term is a lookup table). Capacity per index is a scalar, so it grows with |R| for free —
but there is no weight sharing: a duplicate learns its own bias from its own labels (§2.1: near
zero, so learnable), and a composition learns its own from its own — which for a composition
fired rarely means few positives per index. Data per index is the threat, not parameter count.
Measurement: per-index positive counts and per-rule TRAIN rates at every |R| (already in
`train_guide`'s report); report the share of inflated indices with fewer than 10 strict positives.
If H2 fails and that share is large, the failure is attributed to label scarcity per index, and
the remedy (rule embeddings from templates, `SaturationHead::encode_rule_from_arena`, so that
similar rules share statistics) is named as Round 3's lever — not tried inside Round 2.

### 7.3 Compositions change which application gets strict credit

The strict label credits the application whose output node is literally on the extracted path.
When A∘B exists alongside A and B, the node B would have created is created by whichever fires
first in sweep order — hash-consing makes the second creation an idempotent re-fire. So at
inflated |R|: (a) B's strict-positive rate falls (some of its credit moves to A∘B), (b) A's rate
is unchanged (it was 0 under strict for structural A anyway), (c) A∘B's rate is whatever the
sweep order lets it earn. Labels minted at |R| = 62 are therefore *wrong* for a Guide deployed at
|R| = 124 — not noisy, wrong: they would say B is good when at |R| = 124 the sweep would let A∘B
take that node first.

**Protocol:** labels are re-minted at every |R| point of every mode from unguided saturation
under that exact rule set (same `gen_strict_labels` path, `--rule-set` selecting it), and the
Guide at |R| is trained only on labels minted at |R|. No label transport across |R| points.
The same holds for mode (i) (copies' `Wasted` labels exist only in a mint that contains the
copies) and mode (iii) (new rules' rates exist only there). The mint's rule-set fingerprint is
written into the label file header and checked by `train_guide`; a mismatch is a hard error.

A residual, stated: strict credit at inflated |R| is sweep-order dependent (which of B, A∘B
fires first is decided by index order), so the label is a property of the *unguided* sweep's
order, and a Guide that reorders may face a state the label never saw (A∘B available before B
has fired). This is the same approximation Round 1 already lives with (labels minted from one
run's order, deployed on a differently-ordered run) and is why the tightened-label refinement
(design revision §3 option 3, stage 2) remains the next lever after this round. It is not
addressed here.

### 7.4 Other threats, briefly

- **The |R| = 62 anchor is not Round 1's number verbatim.** `Q(62)` is re-measured in this
  round's harness on the same DEV set (the closure-aware reference and the shared unguided run
  are recomputed) so that every point of the H2 curve comes from one code path. It is expected to
  reproduce 0.537 / 0.696; a departure beyond the bootstrap CI is a harness regression and stops
  the round.
- **Grid sensitivity.** Round 1 found and fixed a dedup-marking defect via a guided-grid
  sensitivity check; the same `--guided-grid` check is run once per mode at |R| = 248 and must be
  a wash before results are read.
- **Compute.** Each |R| point is a mint + train + eval; 5 + 5 + 2 = 12 points, each a
  multi-hour unguided saturation on TRAIN for minting. Nothing here is a metric, but the safety
  ceiling scaling in §5 is what keeps honest curves from panicking at |R| = 248.

## 8. Harness API for Foundations

Additive everywhere; nothing in `all_rules()`, `saturate_until_applications`, or the anytime
runner changes behavior. Minimal public surface; `pub(crate)` unless a pipeline binary needs it.

**pixelflow-search — `src/math/inflate.rs` (new; not referenced by `all_rules()`).**

```rust
pub enum InflationMode { Duplicates, Compositions, NewRules }

/// Parsed from "base" | "dup:<count>" | "comp:<count>" | "new" — <count> is the TOTAL |R|.
pub struct RuleSetSpec { pub mode: Option<InflationMode>, pub total: usize }
impl RuleSetSpec {
    pub fn parse(s: &str) -> Result<Self, RuleSetSpecError>;
}

/// THE one constructor for every rule set this round runs. Prefix is `all_rules()`
/// in production order; inflation appended per §2. Errors (never pads) when the
/// requested total is unreachable.
pub fn build_rule_set(spec: &RuleSetSpec) -> Result<Vec<Box<dyn Rewrite>>, RuleSetError>;

/// Names in index order, joined and hashed — written into every label file,
/// checkpoint, and result row; mismatches are hard errors downstream.
pub fn rule_set_fingerprint(rules: &[Box<dyn Rewrite>]) -> RuleSetFingerprint;

pub struct DuplicateRule { /* inner: Box<dyn Rewrite>, copy: usize, name: String */ }
impl Rewrite for DuplicateRule { /* delegates apply/is_destructive/templates; name "<inner>#dup<k>" */ }

/// A composition A∘B (§2.2). `position` is the path into A.rhs where B.lhs unified.
pub struct Composition { pub a_idx: usize, pub b_idx: usize, pub position: Vec<u8>, pub rule: TemplateRewrite }
pub fn compose_rules(base: &[Box<dyn Rewrite>]) -> Vec<Composition>;   // filtered, enumeration order
pub fn composition_pool(base: &[Box<dyn Rewrite>], seed: u64) -> Vec<Composition>; // seeded permutation

/// The §2.3 batch — 31 indices. Never called by all_rules().
pub fn round2_candidate_rules() -> Vec<Box<dyn Rewrite>>;
```

**pixelflow-search — `src/egraph/template.rs` (new) + one `RewriteAction` variant.**

```rust
/// Generic template rule: e-matches `lhs` at (class, node), instantiates `rhs`.
pub struct TemplateRewrite { /* name: String, arena: Arc<ExprArena>, lhs: ExprId, rhs: ExprId */ }
impl TemplateRewrite {
    pub fn new(name: impl Into<String>, arena: ExprArena, lhs: ExprId, rhs: ExprId) -> Self;
    pub fn compose(a: &dyn Rewrite, b: &dyn Rewrite, position: &[u8]) -> Option<TemplateRewrite>;
}
impl Rewrite for TemplateRewrite { /* lhs_template/rhs_template return copies */ }

// in rewrite.rs
pub enum RewriteAction {
    /* existing variants unchanged */
    /// Instantiate a template RHS bottom-up over `bindings` (metavar i -> bindings[i]),
    /// then union the root with the matched class. Produced only by TemplateRewrite.
    Instantiate { template: Arc<ExprArena>, root: ExprId, bindings: Vec<EClassId> },
}
```

Execution in `graph.rs::apply_action` (adds nodes via `self.add` under the active application so
provenance credits them exactly like `Create`); the e-matcher is `pub(crate)`
(`template::ematch(egraph, arena, pattern, class) -> Vec<Bindings>`, standard top-down matching
over class node sets, metavariables bind to canonical classes, repeated metavariables require
`find`-equality).

**pixelflow-search — `src/egraph/saturate.rs` (additive).**

```rust
#[derive(Clone, Copy, Debug, Default)]
pub struct GuidedEpisodeStats {
    pub rounds: usize,
    pub raw_matches: usize,     // Σ find_rewrite_matches().len()
    pub deduped: usize,         // matches skipped by the seen-key set
    pub scored: usize,          // candidates handed to score_candidates
    pub applications: usize,    // recorded applications this episode
}
impl<'a, G: SaturationGuide> GuidedSaturation<'a, G> {
    pub fn episode_stats(&self) -> GuidedEpisodeStats;   // §7.1 measurement
}
```

**pixelflow-search — `src/math/oracle.rs` (deleted) (`#[cfg(test)]` helper + tests).**

```rust
pub(crate) struct OracleVerdict { pub agree: usize, pub disagree_well_conditioned: Vec<[f32; 4]>, pub ill_conditioned: usize }
pub(crate) fn cross_form_oracle(rule: &dyn Rewrite, seed: u64, points: usize) -> OracleVerdict;
// tests: one per §2.3 family; `compositions_pass_the_cross_form_oracle` over composition_pool(all_rules(), 0x5EED2)
// assert disagree_well_conditioned.is_empty(); ill_conditioned is printed as metadata.
```

**pixelflow-search — `nnue/guide/linear.rs`.** `LinearCandidateGuide::load` and
`PerRuleRateGuide::from_train_guide_report` gain `expected: &RuleSetFingerprint` and return
`Err(CheckpointError)` on mismatch with the checkpoint's/report's recorded fingerprint (the
checkpoint already records `num_rules`/`rule_names`; the fingerprint is derived from those).

**pixelflow-pipeline binaries.**

- `gen_strict_labels --rule-set <spec>` (default `base`, byte-identical output to today): mints
  under `build_rule_set(spec)`; writes the fingerprint into the report JSON and as the first
  line of each JSONL (`{"rule_set": "<fingerprint>", "num_rules": N}`).
- `train_guide`: reads the header, refuses a mixed-fingerprint dataset, writes the fingerprint
  into the checkpoint and report; `--out-checkpoint`/`--report-json` paths per |R| point.
- New `phase3_round2_curves --rule-set <spec> --arms unguided[,control,linear] --sample
  {baseline400,dev-classical} [--checkpoint --train-guide-report] --out-jsonl …`: one
  invocation per (mode, |R|, sample); `--arms unguided` is the Register mode (no guide loaded, no
  checkpoint argument accepted). Per-expression row: `name, origin, tier, node_count, class_cap,
  rule_set, num_rules, grid, arms: {arm: [{app_target, app_actual, cost, stop, clamped}]},
  guided_stats: {arm: GuidedEpisodeStats}`. Reuses `run_anytime_curve_with` for every arm; the
  safety ceiling is `ROUND1_CEILING * num_rules / 62` and panics when it binds.
- New `phase3_round2_aggregate --jsonl <one per (mode,|R|)> --register <path.json> [--verdict]`:
  computes `ref(e,|R|)`, regret, `U`, `L`, `Y`, `Q`, `G`, Δ1/Δ2, Spearman, bootstrap CIs, the
  §7.1 scored-per-application table, the §4 absolute-cost and closure-gain tables; `--register`
  writes the Register JSON from unguided-only rows and REFUSES to run if any input row contains a
  guided arm (Register is unguided by construction, not by discipline); `--verdict` applies the
  §1 tests against a committed Register and writes the results markdown.

## 9. Order of operations

1. Foundations: §8 API, oracle tests green, `all_rules()` count test still 62, the unguided
   determinism test still passes, `cargo check -p pixelflow-ir --no-default-features` still passes.
2. Register run (unguided only): 400-sample curves at every (mode, |R|) → composition pool JSON,
   H1 verdict, `L/Y/Δ1/Δ2`, grid decision → **commit the Register document**. Also the unguided
   DEV-classical curves (needed for `Q`), which are unguided data and may be produced here.
3. Per (mode, |R|): mint TRAIN labels → train → evaluate control + linear on DEV classical.
4. Aggregate with `--verdict`; append results to the Register document; nothing above §9 of this
   document changes.
