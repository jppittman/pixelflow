# The e-graph: what saturation is for when it never finishes

**Status:** `Draft` — an open research problem, 2026-10-15. A statement of the problem, not a
plan and not a solution; nothing in it is agreed. Companion engineering plan (representation and matching cost only;
landed in full, archived): `docs/archive/plans/2026-10-15-the-egraph-is-an-arena.md`.

Citations are tree paths. Two sources are not tracked: the 2026-10-15 profile and telemetry of
the 29 font programs, `/home/user/scratch/egraph/prof2/profile.md` (taken at 7cd96a32, an
ancestor of this tree), and the arena plan's one-time probes,
`/home/user/scratch/egraph/arena/gates/probe3b.txt`. Since the profile, the arena plan's five groups
rewrote `graph.rs`, `graph/runner.rs`, `and_or.rs` and `extract.rs`'s storage and pricing, and the shape cache's contents, each
change byte-identical in the optimizer's output and in the saturation and extraction telemetry
by that plan's gates. So the profile's counts (applications, unions, classes, matches) still
hold, while its instruction and time figures predate those changes; the plan's ledger records
the new ones. The profile's extraction figures are also dated by db126b77 (#1392), a 155-line
change to `extract.rs` that makes the tree pass average an `If`'s arms (section 1). Telemetry is reproducible with
`--features saturation-telemetry` and `$PIXELFLOW_SATURATION_TELEMETRY` (CLAUDE.md, "Why did
the e-graph pick that?"). Ledger rows `Lnnn` are `docs/results/2026-09-07-claims-ledger.csv`.
Results marked **[not in tree]** are on an unmerged branch; no problem below is stated only on
their evidence.

## 1. The setting

A kernel is lowered to an `ExprArena`, inserted into an e-graph (hash-consed, so CSE happens
at insertion), saturated with a fixed rule set, and one term is extracted per root.

**Saturation runs under a deterministic budget** (`pixelflow-search/src/egraph/saturate.rs`;
design record `docs/archive/plans/2026-09-01-production-budget-determinism.md`). Three counts bound
it: rule applications (`APPLICATIONS_PER_CLASS = 40` times the class cap), a cap on
**minted** e-class ids (`max_minted`; ids only rise, so classes unioned away still count),
and iterations. The tier is chosen from the inserted class count
(`docs/results/2026-10-14-the-tier-is-the-class-count.md`); the classical cap is
`clamp(8 x inserted, 5_000, 50_000)`. Wall clock is not a budget; it is a panic ceiling.

**Extraction** prices with the static latency prior (`CostModel::latency_prior()`) and is
greedy in its choice (`extract.rs`, doc on `extract_dag_scoped`): two passes settled in cost
order (Knuth's AND-OR Dijkstra). The two passes price `If` differently. The tree pass
averages the arms, `cost(m) + (cost(a) + cost(b)) / 2` (`below`, `extract.rs:1646-1670`), and
its own doc says this "can price an `If` below one of its arms, so the settlement is no longer
exact at an `If`". The shared pass unions both arms' reach sets (`SharedPricer::price`,
`:2193-2199`). The returned term is chosen by `shared.cost.dag < tree.cost.dag` (`:1409`), and
`Selection::price` (`:975-1012`) charges every selected class once, both arms in full,
weighted by `shape.evals` of its variance. So the CLAUDE.md `If` price applies inside the tree
pass only, and `dag` equals the flat `dag_cost` (every node priced once) only at
`LatticeShape::POINT` (`:1057-1067`). The claim audit is a `debug_assert_eq!`
(`:1550-1580`): it checks every extraction in debug and test builds, not in release
(`docs/results/2026-09-08-extraction-fixpoint.md`, `docs/results/2026-09-08-cse-mispricing.md`).

**The contract** is the as-if rule (`docs/process/2026-10-13-tests-check-the-as-if-contract.md`):
the compiled kernel owes its meaning within tolerance, and nothing about how it was produced.
No bit-level determinism of compiled output is promised (item 4). Everything below concerns
what the optimizer *chooses*, which the contract leaves free; speed is measured, never
asserted.

## 2. The measured fact: production inputs never saturate

On the production font (29 `kernel!` programs, one per piece count, ASCII at 16 px; profile
section 6):

- **28 of 29 programs stop on the class cap after exactly 2 iterations, cut mid-second
  sweep.** The 29th is the blank program (1 class). None stops on the application budget,
  the iteration cap or the ceiling. Inserted classes run 238 to 4,152; the cap is the 5,000
  floor up to 547 inserted and 8 x inserted above (n=40: 33,216); the n=2 program stops at
  4,956 of its 5,000 cap, 21x its input. Live classes at extraction are 29-37% of the cap.
- Totals over the 28: 502,052 applications committed, 314,275 unions, 134,360 live classes.
  Saturation costs 6.2-7.9 k instructions per application at every size: **the cap, not the
  program, sets the work.** Saturation is 51% of the cold optimize's instructions (the probe
  stops at `optimize_runtime_arena`, no JIT) and 45.5% of the full compile path (profile
  sections 1, 3.3).
- **Matching is a brute-force scan and almost all of it finds nothing:** 15,168,439
  (rule, live class) visits; 1,847,916 classes pass the root-shape filter; 3,623,791 nodes
  reach `rule.apply`; 502,309 actions are found. At most 3.3% of visits produce an action
  (a visit can produce more than one). Finding matches is 64% of saturation.
- **The cut decides the output.** Because the run stops mid-sweep, which applications land
  is a function of rule order and live-class scan order. It is deterministic, but it is the
  result of a truncation, not of a closure.
- **Equal members and retired nodes, after a full rebuild** (a one-time probe of the 29
  programs at `feb82335`: each saturated graph cloned and rebuilt to an empty worklist, the
  original left to extraction; the probe was not kept, its 29 lines are in
  `/home/user/scratch/egraph/arena/gates/probe3b.txt`). 28 of 29 programs hold classes with
  two or more members whose canonical forms (operands read through `find`) are equal: 5,243
  such classes in all (at most 536 in one program, n=40) and 13,469 surplus members, counted
  as members beyond one per distinct canonical form (at most 1,448, also n=40). The blank
  program has none. Retired nodes, those taken out of their class for good when their
  canonical form collided with a class they could not merge with: 0 in all 29, in the graph
  as saturation left it and again after the rebuild. Worklist items pending when saturation
  stopped: 0 in all 29.

The old real corpus agrees in kind, under a different regime: 93% of 207 real kernels stopped
on the cap, median 2 iterations, median 5,422 applications (L090, L091). Those 207 are the 190
DejaVu bakes of the deleted glyph kernel, 3 cell grids, chrome, psychedelic and the 12 shader
ports (`docs/results/2026-09-07-corpus-structural-gaps.md:9`), measured under an effective
flat 5,000 cap (the 8x rule's ceiling was pinned at the floor until 2026-09-29), before
class-level dirty tracking (#1224) and before the extraction fixpoint (96da8f5c). No tracked
results doc measures today's programs.

**Re-fires are measured only off production.** On `corpus_dev.bin` (1,512 synthetic
expressions, the unguided batched algorithm, 2026-08-30) 91.1% of recorded applications had
zero node/edge delta and 90.4% of scored candidates committed nothing
(`docs/results/2026-08-30-guide-scope-saturation-delta.md`, L030, NEVER TESTED ON REAL). The
mechanism named there is commutativity, associativity and involution rules re-matching their
own output. Dirty tracking (Linear JP-18) skips clean classes but not an already-resolved
(rule, class content) pair. The zero-growth share of the 502 k production applications is
unmeasured.

**The process is not monotone in the cap.** 33 of 139 candidate pairs have a subterm of the
smaller-budget term absent from the larger-budget graph
(`docs/results/2026-09-08-extraction-witnesses.md:94-130`): a different cap is a different
trajectory, not a prefix extension. Law L2 of `docs/archive/plans/2026-09-02-optimizer-api.md` as
written (the partition at t+1 refines t) holds. Its pin,
`saturation_at_any_budget_never_removes_an_equality` (`graph.rs:2797`), is a cross-budget
ladder over probe equalities on one seed, which a missing term does not break. The
term-superset reading, which the same plan uses at lines 227 and 832, is false at production
caps (witnesses doc, item 4: restate or repair L2).

**More saturation has been bought and measured, with mixed returns** (all on the old corpus):
raising 44 capped glyphs to 7-8x inserted cut summed `dag_cost` 16.7% (17.8% on the 44) at
3.9x warm time, but bytes only 0.3%; a flat 50,000 cap cut only 10.8% with 14 glyphs 15-20%
dearer; on chrome every raise past the floor is a loss
(`docs/results/2026-09-08-class-cap-sweep.md:5, 50, 64`; L072). The raise also put a
half-covered smear along the `'8'` waist at 13-21 px, where FreeType has no ink: the raised
`'8'` extracted a different fusion of its solver. The FreeType oracle caught it, `dag_cost`
could not, and it blocked the raise until 2026-09-29 (same doc, line 5). Hash-consing at
insertion was most of the measured benefit wherever the kernel was built by `Kernel`
composition, which re-expands shared subtrees; there, rules beyond it bought 4.2% bytes and
2.1% `dag_cost` on glyphs. The shader ports, which arrive already hash-consed, got their whole
effect from the rules (-0.5% bytes, -4.8% `dag_cost`)
(`docs/results/2026-09-07-egraph-off-vs-on-real-shaders.md:7`; its instrument is deleted).
Today's `kernel!` programs may sit in the ports' regime, so the glyph split may not transfer.
What rules beyond CSE buy on today's programs is unmeasured.

## 3. Problem 1: budget allocation

**Which applications should be spent so that the extracted kernel is cheaper, across a general
population of kernels?** This is a two-axis frontier: output cost (section 7 item 1) against
compile cost in instructions, including the cost of running the policy itself. Every claim
names which axis was held equal. "At the same budget" is ambiguous on its own: the minted-id
cap binds and the application budget does not, so a policy that mints fewer classes per
application gets more applications under the same cap and can win for that reason alone.

Today the budget is spent in rule order and scan order until the minted-id cap trips. Whether
any policy (ordering, deduplication, filtering, a learned Guide) beats that is the question;
whether there is headroom to beat it is a prior one. Spending less (CSE only, or a lower cap)
is a candidate too, and the rules × nodes filter (section 3.1) found `dag_cost` already
settled at B/8 on 40 of 94 old glyphs.

**The target is generality.** Performance on the selection set is not evidence of generality.
The font programs are DEV, not held out: core-term loads NotoSansMono-Regular
(`core-term/src/terminal_app.rs:76`), which the plan of record classes as DEV
(`docs/compiler/2026-09-07-benchmark-correction.md:39`); the shipped 8x multiplier was chosen at
the 6-to-7 cliff on DejaVu glyphs (`docs/results/2026-09-08-class-cap-sweep.md:27`); and this
document has already read their profile. `program(n)` does not depend on the face, since every
glyph value is an unwritten uniform (`pixelflow-graphics/src/fonts/loop_blinn/program.rs:240-258`),
so a new face adds at most new values of n, never new structure. Inside the font family the
only held-out axis is size: select on small n, test on large n. The held-out members are
NotoSansMono-Bold (with an open instrument gap) and `bench_scene_chrome` (frozen, with its
contamination stated) (benchmark correction, lines 45-46).

What is known: the perfect rule-granularity oracle keeps 3/7/24 of 62 rules
(blitz/rapid/classical; 3/7/22 in the DAG re-run) and filtering at rule granularity is
bit-identical to unguided on 98/100 synthetic kernels at B=100 (L032, L051); on synthetic
kernels regret in DAG units is 43.2% at B=100, 29.18% at B=800 and 0 only near B=12,800
(L051). Rule order moved synthetic regret 86x but real kernels 3%, reversed on psychedelic
(L043, L057). All pre-#1192; synthetic except L057 (204 real kernels, tree objective) and the
5 named shaders in L051.

### 3.1 What the Guide program established, and what it did not

The thesis (`docs/compiler/2026-07-07-guided-saturation-redesign.md`: a learned Guide lets the
rule library grow past what full saturation can afford) **has not been tested above the
shipped class cap.** At the shipped cap, on the old glyph kernel, a tuned bilinear
(rule, class) filter was null on every family [not in tree], because the e-graph's cost was
already settled at B/8 on 40 of 94 glyphs.

The ledger records 52 of 91 claims as not standing; of the Guide groups (D-F, 26 rows) none
that stands says a Guide helps. No learned artifact has changed a shipped kernel (L068);
production passes `guide: None`. **These negatives stand in their own scope (pre-#1192,
synthetic):** L049 (HELD) says no Guide has beaten unguided on trig kernels at B <= 200 and a
guided arm at 4B is still 1.11-1.13x; L043 is FAILED on real kernels; and the bilinear banner
(`docs/results/2026-09-02-bilinear-guide.md:1`) says guidance of any kind is a net cost on
trig kernels at B <= 200. The positive claims are inconclusive for stated reasons:

1. **Units.** Every cost before #1192 is a tree cost or the `dag_cost` of a tree-objective
   extraction. On `julia_set` the two differ 1.4e7 vs 716
   (`docs/results/2026-09-02-phase3-instrument-changes.md:49`). On `sh` the same rows give a
   median ratio of 0.90 in tree units and 1.10 in DAG units
   (`docs/results/2026-09-02-guided-regression-bisect.md:5`, L049); DEV 0.537 (tree) is
   0.6528 (DAG).
2. **Corpus.** DEV/TRAIN are one generator's output (`BwdGenerator` bands); `sh` and `bezier`
   are hand-built generator families. No shipped kernel was evaluated in Round 1 or the
   bilinear run. The synthetic corpus has no selects or compares and sharing 1.0 vs 5.2 on
   real kernels (`docs/results/2026-09-07-corpus-structural-gaps.md`).
3. **Regime.** Registered budgets B = 100 / 200 applications, against 5,573-37,139 per
   program in production today. Many curves quiesced before B (39-41 of 334 DEV classical at
   B=100, 126-128 at B=200; `docs/results/2026-09-01-phase3-at-budget-eval-report.md:31, 46`),
   mixing finished and truncated runs in one ratio.
4. **Thresholds.** Y = 16.3% / 9.0% were derived from tree-cost truncation loss (L033, L034,
   UNITS INVALID); margins M = 0.06 / 0.07 were calibrated on tree cost.
5. **Confounds.** The Round 1 ladder has no dedup-only arm and no dedup-plus-random-order arm,
   so learned scores, deduplication and order are not separable; the per-rule control (which
   also dedups) reached 0.565 against the Guide's 0.537, both in tree units (L035)
   (`docs/results/2026-09-01-phase3-at-budget-eval-report.md`).
6. **The registered checkpoint never loaded.** `guide_checkpoint_strict_v1.json` is refused by
   `LinearWeights::check_vocabulary` (missing `rule_fingerprint`; 61 rules vs 62). Both learned
   arms of the bilinear run were retrained on a re-mint whose positive rate doubled
   (0.522% -> 0.991%) while the control kept the old mint's rates; recorded as a deviation in
   `docs/results/2026-09-02-bilinear-guide.md`. The control arm's loader also resolved 27 of 52
   report rows to no live rule (L050).
7. **Tuning.** No learned model in that program was swept beyond one learning-rate grid
   (ledger section 7 item 11).

**The one Guide-family run on shipped kernels at production budgets** [not in tree:
`origin/claude/rules-by-nodes-filter`, 64e2f1a8:`docs/results/2026-09-08-rules-filter-bilinear.md`
and 14983928:`docs/results/2026-09-08-optuna-rules-filter.md`; open PR #1228; cited as
not-in-tree by the ledger, `docs/results/2026-09-07-claims-ledger.md:164-169, 186-190`]. A
bilinear rules × nodes filter on 208 kernels (190 DejaVu bakes of the old glyph kernel, bench,
the 12 shader ports, psychedelic, the cell grid), with NotoSansMono-Bold and chrome held out;
budgets B, B/2, B/4, B/8 under production's class cap; controls `UniformRandom` at the
filter's realized keep-rate and `PerRuleRate`. Verdict "NULL, on every family": median
`dag_cost` ratio 1.000-1.054 against a registered <= 0.95 (lines 11-13). The intrinsic metric
held (glyph AUC 0.879 vs 0.717). A 250-trial Optuna study left the null unchanged to three
decimals, and the registered configuration ranked 204th of 250. The mechanism (line 623):
`Identity` at B/8 equals `Identity` at B on 40 of 94 glyphs, the only distance-to-closure
measurement besides L054. It is DAG-priced (includes e93448a4, #1192) but by the pre-fixpoint
extractor (lacks 96da8f5c). Four limits: it has no linear arm, so it says nothing about linear
versus bilinear; `UniformRandom` is a random filter at matched keep-rate, not the
dedup-plus-random-order arm item 5 lacks; its line 629 names the decisive untested run (raise
the cap, let the filter keep the graph tractable, make each family's own metric primary); and
it shows the proxy problem (section 5).

**Linear versus bilinear is unsettled.** The one run
(`docs/results/2026-09-02-bilinear-guide.md`, L047, UNITS INVALID) is mixed: on `sh` bilinear
wins the post-hoc paired comparison; on DEV linear does better. It was never run where a win
could have been seen. The registered statistic is D_A(S,B) = m_A^S - m_A^DEV, the shift of an
arm's median ratio from DEV to the out-of-distribution set S (lines 50-62); D_linear -
D_bilinear > 0 favours bilinear. It returned +0.0234 (B=100) and +0.0345 (B=200), with every
guided arm 1.09-1.11x worse than unguided on `sh`. Decomposed into `sh` + DEV terms
(lines 195-203): 0.0019 + 0.0216 at B=100 (92% DEV baseline), 0.0179 + 0.0167 at B=200. Per
expression, bilinear is cheaper on `sh` 67-1-32 (B=100) and 79-3-18 (B=200), a post-hoc
comparison, not the registered statistic; on DEV it is better/tied/worse 26/78/230 at B=100
and 35/152/147 at B=200. On `bezier` it is never cheaper at B=100 (0/65/15), yet its D still
favours bilinear (+0.0193), through the DEV term. At B=200, D exceeds the maximum within-DEV
family swing on `dag_cost` (0.0215) and is inside M = 0.07, which was calibrated on tree cost.
Bilinear fired `half-angle-product` on 97 of 100 `sh` expressions against additive's 80, with
no cost movement (lines 300-311). The bilinear head induces 222 distinct rule orders over 514
contexts against the additive head's 1. `pythagorean` fires zero times in every arm, set and
budget (lines 258-266), so no functional form could learn to prefer it; and `constant-fold`
and `differentiate` both encode to the zero vector, so the model cannot tell them apart
(`docs/results/2026-09-02-bilinear-guide-training.md:131-136`). The settling run is
`docs/compiler/2026-09-07-benchmark-correction.md` section E item 6, which runs only if item 5 is
non-null; item 5 is "expected null" (lines 141-142), so under the plan of record the question
is never asked again.

## 4. Problem 2: attribution

**Which applications, and so which rules, were load-bearing for the extracted kernel?** This is
attribution for rule-set decisions (which rules to delete, keep or reorder); a label for a
learned Guide is one consumer of it. Hindsight provenance (`egraph/provenance.rs`,
`labeler.rs`, feature `provenance-journal`) gives exact *ancestry*, not exact *credit*
(`docs/archive/plans/2026-09-01-guide-return-to-go.md`). The loose, tight and strict labels are
proxies with known defects: loose credits every node in a class and every union by
membership; strict credits only minted nodes, so a rewrite whose right-hand side is a memo hit
earns nothing (`pythagorean` 0/231, `identity` 0/35,387; L040).

Measured only on synthetic corpora, in tree units: the median expression needs 38.2% of its
applications under the loose bound and 2.9% under strict (L027); leave-one-out deletion of a
single application changes the extracted cost in 7.6% of 1,095 sampled applications and is
zero in 92.4% (`docs/results/2026-09-01-counterfactual-credit.md`, L045). The rules × nodes
filter reports 1.2% strictly load-bearing on its corpus [not in tree]. **On any shipped
kernel under today's extractor, the load-bearing share is unmeasured.**

Leave-one-out credit is ill-posed under confluence: when two rewrite paths reach the same class,
deleting either changes nothing. 77 of the 1,012 leave-one-out zeros (7.6%) become positive
under multi-mask (`docs/results/2026-09-01-counterfactual-credit-multimask.md:36`); 90 of
2,525 (3.6%) in the regime run (`docs/results/2026-09-01-counterfactual-credit-regime.md:34`).
Credit depends on which set of applications is masked, so an attribution must name its mask
scope. A third difficulty is specific to production: a trajectory cut mid-second-sweep by a cap
may attribute the cut rather than the program. And the journal exists only when an observer is
attached (`Optimizer::egraph()`); an unobserved run yields a label set with zero rules,
silently (`docs/results/2026-09-02-oracle-filtered-budget-curves.md`).

## 5. Problem 3: the extraction objective

Minimum-cost DAG extraction is NP-hard in general, and the current method is greedy in what it
picks (section 1). What was measured (`docs/results/2026-09-02-extraction-gap.md:46-79`,
L064): a defect, the single DFS that was not a fixpoint, cost 0.258% pooled (worst 14.6%)
across all 302 kernels, all 68 strictly improved kernels real; it is fixed (96da8f5c). On the
89 small kernels where an exact DAG optimum was proved (it stops being computable around
100-200 reachable classes), greedy was optimal on 86, and 95% of the proved loss was tree-vs-DAG
pricing. #1192 closed 95.9% of that, and over 206 real kernels improved 55 with none worse,
structurally (`docs/results/2026-09-02-extraction-objective.md:11-13`, L067). Over the full
corpus the certified headroom past the tree optimum is 0.028%. The tree objective is the
units defect in every pre-#1192 Guide result. At production size (1,510-9,902 live classes)
the exact DAG gap is unknown.

Two changes since date the figures: the fixpoint (96da8f5c), and db126b77's tree-pass `If`
averaging (section 1), so the two passes now disagree on `If` by design. Nothing estimates how
often a mask is true (`extract.rs:1646-1670`); `If` frequency is part of this problem.

Cost: on the font programs extraction is 21.6% of the cold optimize and 19.2% of the compile
path, of which `ReachSets::union_below` alone is 13.4% of the optimize (profile sections 3.3,
5; Linear JP-18, reach sets are quadratic). These figures predate db126b77.

**There are two cost columns and they disagree.** Flat `dag_cost` prices every node once; the
evals-weighted objective weights each node by how often it is evaluated at the lattice.
`cosine_palette` rises 292 -> 521 in `dag_cost` while its objective falls 23%
(`docs/results/2026-09-08-extraction-fixpoint.md:112-128`). The fixpoint doc reports both and
privileges neither; reconciling them is open (`docs/results/2026-09-08-extraction-witnesses.md`
section 8 item 5, line 500). Which column the class-cap sweep's headline is stated in is not
settled by that doc's wording.

The objective is itself a proxy: `dag_cost` can fall while emitted bytes rise (psychedelic
2,176 -> 2,368, `docs/results/2026-09-07-egraph-off-vs-on-real-shaders.md:21`; five shader
ports), `PerRuleRate` reached 0.982 `dag_cost` on glyphs while bytes rose 6.9% and guarded
entries 62% [not in tree, rules-filter line 625], and a 33x cost-table perturbation changed 0
of 190 glyph extractions (L066). A non-additive schedule cost has a seam (`Reranker`,
`extract.rs`; no implementation ships) and a denotation, analytic table plus learned residual
(`docs/compiler/2026-09-01-schedule-cost-model-denotation.md`). Objections to reranking: (i) before
the fixpoint, the swap neighbourhood held 7 of 56 witnesses and no path to the rest
(`docs/results/2026-09-08-extraction-witnesses.md:445-458, 490-493`); after it, objective
witnesses on DEV fell 49 -> 28, CYCLE-PRICED 23 -> 0, and COORDINATED (44) is what remains
(`docs/results/2026-09-08-extraction-fixpoint.md:160-193`); realizability was not re-measured.
(ii) It keeps the DAG as the domain; that was raised against schedule-valued extraction, which
`docs/language/2026-10-15-kernels-evaluate-kernels.md` section 3 has since ruled the CPU backend's,
so the objection holds only if extraction is to choose schedules. (iii) L088 retracts "the cost
model does 95% of the work". A different learned object, a per-class scorer against a
uniform-random-among-argmin null, is registered (`docs/compiler/2026-09-08-extraction-judge-registration.md`)
and has not been run: its results section is empty. Whether any non-additive structure is
left in extraction (fold unrolled vs loop, Linear JP-15) is open.

## 6. Problem 4: matching cost (engineering, not research)

Every rule scans every live class each sweep; 64% of saturation is finding matches, and
malloc/free is 17-18% of all instructions. As profiled, the representation was the classic egg
layout (per-class heap `Vec`s, a parallel `ENode` type, three conversions) beside an `ExprArena`
that is already a hash-consed DAG; `docs/archive/plans/2026-10-15-the-egraph-is-an-arena.md`
addressed the representation (all five of its groups have landed, through the member lists and the frozen cache), kept
scan and member order, and was byte-identical on the font programs.

Under a truncating cap, **match order is part of the output**: a matcher that finds the same
matches in a different order lands a different set before the cut and emits different
kernels. An order-preserving matcher is judged by bit-identical output (section 7 item 6).
Only one that changes order (indexed or relational matching, deduplication) is an optimizer
change and needs section 7.

## 7. What any credible attack must satisfy

1. **The cost is what the kernel pays, per family.** The objective is each family's primary
   metric (benchmark correction section B.3, lines 52-59); nothing is promoted that worsens any
   family's own primary metric (lines 138-139). The deterministic proxy columns are the
   shape-weighted DAG price at the kernel's lattice (primary) and flat `dag_cost` (second),
   from the current extractor, with the open reconciliation cited (section 5); emitted bytes
   and a clock beside them. Per-kernel clock ratios under ~10% are not trusted (L076); the
   clock is taken at 1-minute load < 8 and labelled. No tree cost; no wall clock inside any
   cost number.
2. **Headroom first.** Before any policy experiment: CSE-only, shipped, and best feasible cap
   arms on a grid of caps with the shipped cap marked; an A/A noise floor; the minimum
   detectable effect computed in advance. The reference is the best-known term, never "the
   optimum": quiescence is unreachable at production size and a larger cap's graph is not a
   superset of a smaller one's.
3. **The baseline is a distribution.** K seeded permutations of rule order and scan order are
   the null; baseline and random arms are reported as distributions, and a win is claimed only
   outside them. Arms include dedup-only, dedup-plus-uniform-random-order, a context-free
   per-rule control, and an unguided arm matched on compute.
4. **A corpus admitted by two criteria.** Correctness, by an oracle that is not a same-form
   check (`libm`, FreeType coverage, range bounds); the deleted quarantine compared the JIT with
   a scalar interpreter that ran the same expansion, and CLAUDE.md "Precision is on the table;
   range is not" records why that is blind (Linear JP-29). And representativeness: the structural
   columns of benchmark correction section C (select density, sharing, op mix) within the
   shipped kernels' range.
5. **Held-out families**, split by generator or shipped family, never by expression:
   NotoSansMono-Bold and `bench_scene_chrome` (section 3), and size within the font family.
   The font programs are DEV.
6. **Pinned instruments.** Every result row records the commit, the rule-set fingerprint, the
   extractor version and the ISA tier; results under different extractors are not comparable.
   "Same input" means the same arena as inserted (the memo is order-dependent,
   `docs/results/2026-10-14-the-tier-is-the-class-count.md:19-21`). The same input and
   count-denominated budget give the same trajectory, extracted term and costs. This is a
   property of the experiment, not of compiled output (as-if contract item 4).
7. **Correct outputs.** The claim audit runs on every evaluated extraction (it is compiled out
   of release), and every evaluated output passes the as-if check against an external oracle:
   FreeType coverage for glyphs, agreement with the off/cse arm at a stated tolerance for
   scenes (benchmark correction, line 59). The `'8'` smear (section 2) is the case.
8. **Frozen, registered checkpoints that load.** If the registered artifact is refused, the run
   is void and is re-registered, not substituted.
9. **Tuned before quoted:** an Optuna sweep on the intrinsic metric, with the untuned
   configuration as trial 0, reported in the same table
   (`docs/results/2026-09-07-claims-ledger.md:151-190`).
10. **Budgets in the production regime's units** (thousands to tens of thousands of
    applications at the shipped caps, and above them), with the stop reason per kernel and
    finished runs separated from truncated ones.
11. **Statistics fixed before the run, in the units tested:** per-family paired geomean, worst
    case and regression count; the statistic itself registered, not only its margin; each arm
    beats the null before two arms are compared. The 29 font programs are one construction at
    29 values of n, not 29 independent samples, and their sums are dominated by n=40.
12. **Per-problem designs.** Extraction is evaluated on frozen saturated graphs (an end-to-end
    comparison confounds the two stages), with lower bounds where the optimum cannot be
    computed and the uniform-random-among-argmin null. A saturation policy also reports the
    best-known term per graph (`egraph/witness.rs`) beside the greedy result. An attribution
    is scored against a defined counterfactual with a stated mask scope.
13. **Tracked instruments and a gate.** The instrument is a tracked file at registration and
    stays until its result is superseded; raw rows are retained or content-hashed; benchmark
    correction section E item 0 (a presubmit job diffing the deterministic columns) lands
    before any result is admissible.

**What survives to re-run.** Nothing in section 3 can be re-run as-is. The corpora, label
mints and checkpoints are untracked (`pixelflow-pipeline/data/*` is gitignored), and no real
`.arena` dump is tracked. `gen_bench_corpus`, `gen_sh_corpus`, `gen_bezier_corpus` and
`egraph_off_on` were deleted in 077c1641; `sh_family::draw`, `bezier_family::draw` (in
`pixelflow-pipeline/src/training/`) and `BwdGenerator` (`pixelflow-search/src/nnue/mod.rs`)
survive. Benchmark correction section C (retargeting the generators) and section E item 0
never landed.

## 8. Open questions

1. What do the 29 font programs' telemetry and cost look like in a tracked results doc
   (today they exist only in the untracked profile)?
2. How far from quiescence does the cap leave each program? Is the rule set's closure on them
   finite, and how large? Which rules mint the classes that reach the cap?
3. What share of production applications are zero-growth re-fires, and what share are
   load-bearing for the extracted kernel?
4. What do rules beyond CSE buy on shipped kernels today, in each family's primary metric,
   bytes and the DAG columns?
5. With the cap raised and a policy keeping the graph tractable, does any policy reach a
   cheaper kernel by each family's own metric, at matched compile cost, on held-out families?
   At the shipped cap, does deduplication or random order move the output outside the
   baseline's order distribution?
6. Is a minted-id cap the right budget dimension, given that it counts dead classes and makes
   the process non-monotone? Allocated/live is 2.61x median, 10.07x worst (L058, counts HELD,
   recovery UNITS INVALID); a live-counted cap (#1114) bought +2.03% median for 2x
   applications and was closed unmerged (L059, UNITS INVALID).
7. What attribution is well-posed under confluence, and with what mask scope?
8. How large is the gap between the greedy shared pass and the exact DAG optimum at production
   size, and can it be bounded where it cannot be computed?
9. Linear versus bilinear scoring is unsettled and, under the plan of record, will not be
   re-asked: what is the smallest experiment meeting section 7, at a raised cap, that could
   tell them apart?
10. What corpus acceptance criterion is external to the compiler and cheap enough to gate
    generation?
