# The claims ledger: what the research arm said, and what real shaders said back

**Date:** 2026-09-07. **Companion:** `2026-09-07-claims-ledger.csv` (91 rows, one per quantitative claim since the self-play era; columns `id, group, claim, minted, corpus, instrument, metric_units, kind, real_check, verdict, note`).
**Why this exists (JP, 2026-09-07):** *"This isn't an egraph for fonts project. It's an egraph for shaders project, and we have an app with shaders and the moment we pointed it at the real shaders it sucked. ... We need to act like when our research hit the real world, it shit the bed, so we need to go back and correct."* This is the first artifact of that review: every claim, its corpus, its units, its instrument, and its verdict against a shipped shader.

## 1. The verdict

| verdict | rows | meaning |
|---|---:|---|
| **HELD** | 39 | stands on a real shader in correct units, or is a fact about the method that survives every re-unit |
| **FAILED** | 5 | tested on a real shader and contradicted |
| **UNITS INVALID** | 15 | priced in tree cost (`ExtractedDAG::total_cost` before #1192) — the kernel pays DAG cost; the number is in the wrong units even where the corpus was real |
| **INSTRUMENT DEFECT** | 18 | the instrument that produced it was later shown wrong: 41.67× then 4× timebase, null-context gathers, tile-vs-scanline inversion, overhead-biased ratio, cross-population Spearman, unreproducible artifacts, hash-against-prose gates |
| **NEVER TESTED ON REAL** | 14 | minted on a generated corpus and never taken on a shipped kernel |

**52 of 91 do not stand.** Of the 39 that do, **20** are measurements taken on real shaders (`verdict=HELD` and `kind=real`), 8 are findings that an instrument or method was broken (`kind=n/a`), and 11 are synthetic-corpus results that survive re-unit. Every figure in this section is a `verdict`/`kind` count over the companion CSV and should be recomputed from it rather than restated — an earlier revision said 21/8/6, which double-counted synthetic rows as real-shader measurements, a second said 35 / 16-8-11, a third said 37 / 17-8-12 before the 2026-09-08 decisions-doc verdict corrections and four new rows, and a fourth said 53 of 92 before L020 was reverted to NEVER TESTED ON REAL and L078 was dropped (§7). **No claim that an optimizer improvement helps a shipped shader survives on main**, with two exceptions: sharing-aware extraction (#1192, L067: 55/206 improved, 0 worse, chrome schedule 401→385 entries) and the schedule-side S3b win (L073), which is codegen, not the e-graph. L081's C1 guard change is a third real, measured improvement (nearly halving row time on shipped `O`/`S` glyph kernels) but was never merged — production is byte-for-byte unchanged, so nothing *on main* carries its number; it is excluded from this sentence on deployment, not on evidence.

By group, the failure is concentrated where the headlines were: the extraction-head/paper rows (16) are **10** defect-or-untested (6 INSTRUMENT DEFECT + 4 NEVER TESTED ON REAL — L020 moved back from FAILED to NEVER TESTED ON REAL on 2026-09-08 (round 3, §7), so it counts here again); Phase 3 Round 1/1b/2/R2G/bilinear (20) are **10** units-invalid-or-failed (9 UNITS INVALID + 1 FAILED — L047 moved from NEVER TESTED ON REAL to UNITS INVALID). The self-play era (8) is **5** timebase-defective or unrecorded, not all of it — L001–L003 are HELD, and they are findings *about* the method (the loop closes byte-exact; the extraction head was dead in production; the RL half cannot learn) rather than claims it produced.

The lattice programme's rows (17, real shaders) are **11** HELD. The six that are not divide two ways: L074–L077 and L089 are findings *about* instruments rather than claims that fell, and L082 is FAILED. (L078 was dropped 2026-09-08 (round 3) — it restated a claim its cited source does not make; see §7 Revisions. L081 was NEVER TESTED ON REAL and is now HELD — it is measured on shipped `O`/`S` glyph kernels, so that verdict was unavailable to it by this document's own definition. L089 is the serialized-critical-path half of L083, split out 2026-09-08 — see §7 Revisions.)

## 2. The five ways the method failed, each with the rows that show it

1. **Units.** Every cost number before #1192 is a tree cost. The kernel pays DAG cost; `julia_set` is 1.4e7 tree against 716 DAG. Fifteen rows are in the wrong units outright (L033–L035, L038, L042, L044–L047, L054, L057–L061), and the Guide's registered constants (B=100/200, Y=16.3%/9.0%) were derived from them and "port as-is, not re-derivable" (L034). The bisect (L049) is the load-bearing correction: the sh family's 0.90 in tree is 1.10 in DAG — every guided arm *loses* on trig kernels, and guides trained on tree labels steer toward unshared terms.
2. **Regime.** The Guide programme was registered at B=100–200 applications. Real kernels fired a median 8,446 before production stopped (L056), and 68.4% stopped on the class cap (L053) — the 200 ms clock-limited regime. Under the budget now shipped it is a median **5,422** applications with **93%** stopping on the cap (L090, L091; `2026-09-07-corpus-structural-gaps.md`), so the registered B=100 is **54×** below production, not 85×. Every quality-at-budget claim (L033–L035, L087) is about a regime production never runs in.
3. **Instrument.** The bench timebase was wrong by 41.67× before 2026-07-20 (L010) and by 4× again after #1071 (L065); the latency prior was measured with a null context pointer so `Gather` has never been priced; labels were minted in tile mode, and Horner-vs-Estrin's serialized-critical-path reading (L089) shows tile/latency ordering *inverts* against the scanline production loop, which is why the scanline reading is kept separately at L083. The paper's "parity" ratio is `(n+c)/(s+c)`, biased toward 1 (L013); its calibration Spearman is cross-population (L017); three of its rounds rest on artifacts not in the repository (L014, L015, L023). the corpus predictors do not survive being read per predictor (L075): `dyn_memory_ops` ranks at ρ=0.825/0.796 with aggregate sign accuracy 72.5%/77.6%, the ≈0.98 ranks belong to `dyn_emitted_ops`/`dyn_bytes`, and 27% is the AVX-512 sign score for the single `main → tripcount` pair — an earlier revision combined the 0.98 of one with the 27% of another and quoted a statistic nobody measured; it reported +40% on a change whose clock moved −7% (L074).
4. **Corpus.** Fourteen rows were never taken on a real shader, among them still the single most-quoted number in the program — "extraction with the static table is worth ~2× over not extracting" (L020). F landed (`2026-09-07-egraph-off-vs-on-real-shaders.md`) and is real, but its arms are `PIXELFLOW_SATURATION` off vs on (with a cse-only arm and a with-select-hoist arm beside them) — a **different decomposition** than extraction-with-the-table vs not-extracting. Its numbers are useful context — glyph32 aggregate −18.6%/−20.8%, 90/95 glyphs faster, with the cse-only arm carrying most of it (−41% of the −44% byte reduction; rules-beyond-CSE move it only ≈−4%; the pre-hash-consed ports show extraction-with-the-table worth −0.5% bytes / −4.8% dag_cost) — but nothing in F produces L020's own static/noswap ratio on a real kernel, so that specific decomposition remains unmeasured. Where real shaders were finally used elsewhere, synthetic headlines collapsed: rule order 86× → 0.97× and *reversed* on the psychedelic shader (L043/L057); the class-cap A/Bs that improved tree cost on glyphs were contradicted by the clock on chrome, where 12× more classes made the kernel 15% slower (L072 vs L054/L058/L059); scenes and shaders enter the e-graph small and the rules themselves do most of the expansion to the cap (L092). One row (L005) is a claim with no corpus, no number, and a deleted harness.
5. **Provenance.** Three rows carry their own banner that they predate a fix that "changes every number below" and were never re-run (L031, L036, L041) — earlier revisions said five, then four, before L028 was found to have in fact been re-run twice (0.35, then 0.186 on a second synthetic draw) and moved out of this list. Both of L028's re-runs are synthetic, so by this ledger's own definition its verdict is NEVER TESTED ON REAL, not HELD, and INSTRUMENT DEFECT was the wrong original label for it either way — the instrument was noisy, not broken (2026-09-08). The paper's FINAL tier was never opened (L026), and its 12 ShaderToy ports were then consumed for a rule-order decision, so it no longer holds out ordering claims.

## 3. What stands on real shaders, in correct units

- Production saturation is **class-cap bound** on real kernels (68.4%), fires a median **8,446 applications**, and the cap binds on the **spliced input** — chrome's 416,420 nodes hash-cons to ~4,900 classes, within 90 of the cap, on iteration 2 (L053, L056, L086).
- **More saturation sometimes extracts worse code** (L055), and on the one clocked real shader, 12× more classes cost 30× compile and **+15% ns/px** (L072). Congruence closure and the rebuild orphan do not explain the cap (L062, L063).
- **Extraction was minimizing the wrong quantity.** 95% of the proved greedy gap is tree-vs-DAG (L064); pricing sharing closes 95.9% of it, 55/206 real kernels improve, 0 worsen, and it holds on both production shapes (L067).
- **The cost table is not the lever on glyphs**: a 33× perturbation changes 0 of 190 extracted terms — the saturated graph holds no alternative lowering (L066).
- **No *learned* research artifact had ever changed a shipped kernel** as of 09-01: the NNUE head was opt-in and never set, the Guide library-only (L002, L068). Two non-learned ones *were* live and the audit says so — the remeasured static latency table (shipped `be4f98df` / #984, which `2026-09-01-integration-audit.md` calls "the one that did change what a user runs") and provenance recording, live on every production compile with no consumer and its cost never measured, which is what #1118 gated. An earlier revision dropped "learned" here and overstated the finding against L068's own LIVE-DEFAULT status column.
- The **schedule** is where the real wins were: S3 shipped chrome at 0.32× (L071), S3b recovered 3.6× via one select per colour, arm clustering and a mispredict bound (L073) — none of it an e-node. A curved glyph is all row prologue (L080); `text()` as a sum is O(n)/px (L079).
- **Additivity is ISA-conditional**: exactly right where AVX-512 is throughput-bound (slope 1.03), wrong by up to 1.9× where latency is exposed (L083) — and the tile-mode instrument inverts against production's loop.
- Budgets are now denominated in applications, sized from real counts (L069); `bake` beats `kernel!` 2.6–2.9× at SSE2 (L070) — a JIT-vs-LLVM fact whose e-graph share is unmeasured.

## 4. Retractions

Each of these gets a dated "Retracted/Superseded" block at the top of its results doc, pointing here — **applied by the follow-up PR #1215, not by this commit.** `2026-09-07-benchmark-correction.md` and the blocks themselves do not exist in this tree; a reader here should expect the pointer to be live only once that PR lands. Closed-branch docs are marked here only.

| row | claim | why |
|---|---|---|
| L022 | the additive objective is nearly solved by DP; residual at the noise floor | premise false in its own units until #1192; additivity is ISA-dependent |
| L088 | "the cost model does 95% of the work"; a reranker over swap neighbourhoods is next | rests on a cross-population synthetic Spearman; production-scale §7: B is not a step toward C |
| L087 | Guide framing "Y = half the truncation loss" at B=100/200 | wrong regime (54× against production's median 5,422) and wrong units |
| L033–L035 | Round 1 baseline, registered constants, "Guide saves 16%" | tree units; DAG reading is a smaller DEV win and a loss on every structured family |
| L038 | domain shift is H-null | compared two arms that both lose on the real metric |
| L042, L043 | Round 2 v2/v3 — `\|R\|` effect; order dominates 86× | order confound; 86× is ~3% on real and reverses on psychedelic |
| L044–L046 | R2G dataset gate, R2G Guide, regime-restricted R2G | tree-cost deltas |
| L047 | bilinear H_form-null | dag_cost of a TREE-objective extraction (pre-#1192), not a tree-cost delta; re-take under #1192 only if the benchmark correction's item 5 is non-null |
| L054, L058–L061 | class-cap cost 8.66%; ghost recovery 2.35%; live cap +2.03%; cap-break; congruence −0.07% | tree-DP costs on real kernels; sign contradicted by the chrome clock |
| L004, L006, L007 | self-play era ns numbers | corpus unrecorded; the 08-05 audit's H1/H2/H3. **Not the 41.67× timebase**: all three are ratios (L004 `ns ratio`, L006 `speedup`, L007 1.0669/0.6676/31×), and `2026-07-20-jit-compile-cost.md:29-33` states that a uniform scale factor leaves ratios and orderings unaffected. The verdicts stand on the other grounds |
| L013–L017 | the paper's parity rounds and calibration | overhead-biased ratio, cross-population ρ, unreproducible artifacts, confounded init |
| L031, L036, L041 | oracle-filtered curves, linear-vs-lookup AUC, tightened labeler | own banner: predates fixes, never re-run |
| L065 | latency-prior remeasure absolute numbers | 4× optimistic; table shift was sound but #1157 moved trig again — re-measure |
| L082 | a tight Y-extent gate fixes the `'8'` waist | an always-true gate fixed it equally; font thread |

## 5. What must be re-taken, in order

Cheapest and most load-bearing first. Each on the corrected corpus (real shaders, family-split held-out), deterministic columns as the claim, clock as a sign, decision rule pre-committed before the run.

1. **F — the e-graph on vs off, on every shipped shader** (L020's context, not L020 itself). *Complete* (2026-09-08): `docs/results/2026-09-07-egraph-off-vs-on-real-shaders.md` — glyph32 clock on/off median 0.63, Σ −19–21%, bytes −44%, real, 90/95 glyphs faster. But F's arms are `PIXELFLOW_SATURATION` off vs on (with a cse-only arm and a with-select-hoist arm beside them), not extraction-with-the-table vs not-extracting — a different decomposition than L020's own claim. The cse-only arm carries −41% of the −44% byte reduction; rules-beyond-CSE move it only ≈−4%; on the pre-hash-consed `shader_bench` ports, extraction with the table is worth −0.5% bytes / −4.8% dag_cost. Nothing here produces L020's static/noswap ratio on a real kernel — **L020 remains NEVER TESTED ON REAL**; its specific decomposition is still unmeasured on real and still needs its own re-take.
2. **Rule order on real kernels in DAG units** (L057). Dumps exist; deterministic; the direction (≈3%, not 86×) is expected to hold and the magnitude to move.
3. **Class-cap cost and the live-cap A/B in DAG units on real kernels** (L054, L059), against the chrome clock. Until this is done, "more classes" carries a measured contradiction.
4. **The latency prior, on a fixed instrument**: the 4× timebase, a real context so `Gather` is priced, scanline mode not tile (L065, L083). Then re-measure; #1157 changed trig after the last one.
5. **Guide-at-budget in DAG units at the real regime** (~8,000 applications, not 100) on real kernels (L034, L035, L038). Only after 1–4. Expected null; a null closes the question honestly.
6. **The paper's parity** (L013) — only if the paper is revived; otherwise it stays retracted.

**Not worth re-taking:** R2G (L044–L046: "the target was not the bottleneck" is a negative result unlikely to flip in DAG units); Round 2 v2 (superseded by v3, which itself failed on real); NNUE extraction overhead and incrementality (L021, L024: architecture deleted, #1093); the self-play era (apparatus deleted); L005 (nothing to re-take).

## 6. Rules the ledger imposes on the program

- **The synthetic corpus is never a headline.** Coverage and stress only. A claim *about what an optimizer change is worth* must name a shipped shader or it is NEVER TESTED ON REAL by definition. The rule is scoped to that class deliberately: it does not reach rows whose subject is a code property, a budget constant, or an instrument's own behaviour, which is why 11 synthetic-corpus rows are HELD (L019 and L083–L085 among them, `real_check = n/a`). Stated absolutely, the rule would contradict §1's own split and no consumer could reproduce the 39/52.
- **Units in the row.** `dag_cost` or the clock; a tree cost is not a cost.
- **Regime in the row.** Applications at production's actual stop, not a registered budget.
- **The instrument is a claim too.** Timebase, context, loop shape (scanline), and the oracle (external or same-form) are stated beside every number.
- **Artifacts in the repository.** A number whose per-kernel rows are uncommitted is a note, not a result.

## 7. Revisions (2026-09-08)

Applied from `docs/results/2026-09-08-open-thread-decisions.md`, which recorded the twelve
recommendations JP handed back decisions on. Each change below cites the decision it applies;
the CSV carries the full before/after reasoning in each row's `note`.

1. **L081** — kept HELD (was already corrected from NEVER TESTED ON REAL by the prior sweep);
   the unmerged-implementation caveat stays in `note`. No change needed (decision #1).
2. **L047** — reclassified NEVER TESTED ON REAL → **UNITS INVALID**. The 1.09–1.11× bilinear
   deltas are `dag_cost` of TREE-objective extractions minted before #1192; the compared terms
   no longer exist, independent of the corpus (decision #2).
3. **L083** — un-narrowed and actually split. It now carries only the scanline/production
   conclusion (HELD, unchanged verdict). The serialized-critical-path 1.2–4.0× number moves to
   a new row, **L089** (INSTRUMENT DEFECT): AVX-512 latency rows below degree 16 are
   chaining-overhead dominated per the source report's own warning; SSE2/AVX2 serialized rows
   are readable (decision #3).
4. **L072** — verdict unchanged (HELD); `note` already read as an out-of-domain counterexample
   rather than a contradiction, so no further edit was needed (decision #4).
5. **L028** — reclassified HELD → **NEVER TESTED ON REAL**. This ledger has five verdicts and no
   sixth for "noisy": both re-runs are on a synthetic 800-kernel draw and neither ever touched a
   shipped kernel. Claim restated as a range, ρ ≈ 0.19–0.35 across two draws of 800 (sampling
   variance); INSTRUMENT DEFECT was the wrong original label — noisy, not wrong (decision #5).
6. **L078** — `minted` corrected to a file+line citation:
   `docs/plans/2026-09-06-kernel-with-a-lattice.md:780–781` (the S4b-1 landed section), which
   reads "`collapse_bench` never goes through it \[`JitManifold`\] — it calls `emit::compile`
   directly." The sentence exists and is about `JitManifold`'s rename, not the optimizer; the
   row's own verdict (FAILED, read literally) and its code-verified `real_check` are unchanged
   (decision #6).
7. **L020** — reclassified NEVER TESTED ON REAL → **FAILED, on attribution**. F is complete
   (`docs/results/2026-09-07-egraph-off-vs-on-real-shaders.md`): real aggregate improvement is
   ≈1.6×, not the claimed ~2×, and 41 of 44 points of byte reduction is hash-consing at
   insertion, not extraction with the static table (which moves dag_cost ≈−4.8% on
   pre-hash-consed ports). §5 item 1 now reads Complete with these numbers; §2's "most-quoted
   number" sentence is rewritten to match (decision #7).
8. **L053 / L056** — `note` marks both historical: the 200 ms-timeout regime, pre-#1118 (already
   applied by the prior sweep). Two new rows added: **L090** (93% ClassCap, median 2 iterations,
   current `Budget::Production`, HELD) and **L091** (median 5,422 applications = 54× the
   registered B=100, HELD) — both sourced from `docs/results/2026-09-07-corpus-structural-gaps.md`.
   Every "85×" dependent on the retired regime already read "54× ... not 85×" going into this
   revision. One further row added, **L092** (HELD, real): scenes/shaders enter the e-graph
   small and the rules expand them 10–40× to the class cap in 2–5 iterations, superseding
   `docs/plans/2026-09-06-egraph-at-production-scale.md` §2's "~4,900 classes after
   hash-consing" framing — the dated Superseded banner for that sentence is added in #1215's
   worktree, which already carries retraction blocks on plan docs (decision #8).

**Verdict histogram, before → after:** HELD 37→39, FAILED 6→7, UNITS INVALID 14→15, INSTRUMENT
DEFECT 17→18, NEVER TESTED ON REAL 14→13, total rows 88→92. Headline: **53 of 92 do not stand**
(was 51 of 88).

9. **L020** — reclassified back **FAILED → NEVER TESTED ON REAL** (2026-09-08, round 3). Decision
   #7's "FAILED, on attribution" conflated F's own decomposition (`PIXELFLOW_SATURATION` off vs
   on, with a cse-only arm and a with-select-hoist arm beside it) with L020's claim
   (extraction-with-the-static-table vs not-extracting) — a different decomposition; nothing in F
   produces L020's static/noswap ratio (0.5440/0.5444/0.5437/0.5433) on a real kernel. F's numbers
   are kept in the row's `note` as context (glyph32 −18.6%/−20.8%, 90/95 faster, the cse-only arm
   carrying most of it), but the decomposition L020 actually names remains unmeasured on real. §2
   item 4 and §5 item 1 are rewritten to match.
10. **L078** dropped: restated a claim its cited source does not make (2026-09-08, round 3). The
    cited S4b-1 text is about `JitManifold`'s rename, not about whether runtime optimization runs
    — `compile_as_baked` optimizing before `emit::compile` is compatible with it. Removed from
    every table and count above, including §4 Retractions.

**Verdict histogram, round 3 (2026-09-08):** HELD 39→39, FAILED 7→5, UNITS INVALID 15→15,
INSTRUMENT DEFECT 18→18, NEVER TESTED ON REAL 13→14, total rows 92→91 (L078 dropped). Headline:
**52 of 91 do not stand** (was 53 of 92).

### Untuned models (added 2026-09-08)

11. **No learned-model row in this ledger had its configuration swept.** JP, 2026-09-08:
    *"none of my models worked at all until I ran optuna. We haven't run optuna once since I
    stopped following this closely."* The previous sweep harness (`optuna_unified.py` driving
    `train_unified --server <unix socket>`) was deleted with the RL loop in `fb8136e3`, and
    nothing has swept since. The rows this applies to, none of whose verdicts change here — the
    caveat is about what the *evidence* can support, not about which way a number went:

    - the extraction head: **L005, L006, L007, L011, L012, L013, L014, L015, L016, L017, L018**
      — every trained-head row, including the contrastive objective (L014), the random-init
      arm (L016) and the bootstrap learning curve (L018);
    - the Guide programme, linear: **L034, L035, L036, L037, L038**;
    - return-to-go: **L044, L045, L046**;
    - bilinear: **L047, L048** — with one qualification, below;
    - and the bilinear rules × nodes filter's null, which is not a row here. It is registered
      in `docs/plans/2026-09-08-rules-filter-bilinear-registration.md` and reported in
      `docs/results/2026-09-08-rules-filter-bilinear.md`, **neither of which is in this tree**:
      both are on the open PR #1228 (`claude/rules-by-nodes-filter`) and are unauditable from
      here until it lands.

    The qualification on the bilinear rows: "one hand-picked configuration" would be too
    strong for L047/L048. `docs/results/2026-09-02-bilinear-guide-training.md:213-237` records
    a learning-rate grid selected on a TRAIN-internal holdout, widened after an earlier grid
    selected its own boundary. That is a real, honestly-run one-dimensional search — it is not
    an Optuna sweep, and every other hyperparameter (embedding width, regularization, epochs)
    remains unswept, which is what the caveat is for. The distinction matters because a null
    from a model with a tuned learning rate is stronger evidence than a null from one without.

    A null from an untuned model is weak evidence against the *shape* it was testing: it cannot
    separate "this architecture does not carry the signal" from "this learning rate does not".
    It is still evidence — an untuned model that *wins* is at least a lower bound, and an untuned
    model that ties a control at least says the shape is not free money — but it is not the
    evidence a FAILED verdict on a shape would need.

    **The rule going forward: no learned model's extrinsic number is quoted — in this ledger or
    anywhere else — without an Optuna sweep on the intrinsic metric first, with the untuned
    configuration enqueued as trial 0 so the tuned and untuned numbers appear in the same table.**
    The first application of the rule is the rules × nodes filter,
    `docs/results/2026-09-08-optuna-rules-filter.md` — **also not in this tree**, and on the
    same open PR #1228 as the two artifacts above. Until #1228 lands, the rule is stated here
    but its first application cannot be checked from this commit.

### Round 4 (2026-09-15)

12. **L028 removed from the §4 "never re-run" retraction row.** That row still listed L028
    alongside L031/L036/L041 under "own banner: predates fixes, never re-run", contradicting
    revision 5 above and §2, both of which record that L028 *was* re-run twice (ρ 0.35 and
    0.186) and that the difference is sampling variance rather than a provenance failure. §2's
    count already read three rows; the table now matches it. L028's own verdict
    (NEVER TESTED ON REAL, on the synthetic-corpus ground) is unchanged.

13. **The extraction-head list under revision 11 completed.** It named L005–L007, L011–L013,
    L015 and L017 while skipping three plainly learned rows in the same family: L014
    (a trained contrastive objective), L016 (a random-init trained head) and L018 (a bootstrap
    learning curve). The caveat claims to cover every learned-model row, so those three are
    now in it.

14. **The blanket "no hyperparameter search" narrowed for L047/L048**, which did have a
    learning-rate grid selected on a TRAIN-internal holdout. See the qualification under
    revision 11.

15. **Three forward references marked as not-in-tree.** The two bilinear rules × nodes filter
    artifacts and the Optuna report cited as the rule's first application live only on the open
    PR #1228 and cannot be opened from this commit. Marked as such rather than left reading as
    though they were landed evidence — which is the ledger's own fifth failure mode.
