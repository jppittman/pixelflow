> **A dated audit, largely executed (annotated 2026-09-09).** Its verdicts were
> acted on: much of what it marked DELETE is gone, which is why it now names files
> that are not in the tree (`bench_extraction_3way.rs`, `unified_backward.rs`,
> `pixelflow-ml/src/nnue.rs`, the `nnue/` cost-model modules). Read it as the record
> of that decision round and for the six D-questions in §6 — of which D4
> (`pixelflow-ml`, still a workspace member with zero dependents) is still open.

# Dead code with decent ideas — keep / reuse-for-round-2 / segregate / delete

**Date:** 2026-09-01
**Base:** `origin/main` @ `83015dcd` (PRs #1083, #1084, #1085 still open; read from their branches where noted)
**Prompt (JP, verbatim):** "please scan the repo/take a gander at what already exists. I think we still have a lot of dead code with some decent ideas in it. factored.rs and so on."
**Doctrine:** `feedback_segregate-dont-delete-roadmapped` — KEEP-live / SEGREGATE (roadmapped earns a seam, not dead weight in a live path) / DELETE; plus one verdict for this pass, **REUSE-R2**: a dead or dormant piece whose *idea* is what the Guide's candidate-context design needs, with the plug-in point named.

**Where the program is, so "wanted later" can be judged.** The extraction-head program (ExprNnue value head, `IncrementalExtractor`, `PIXELFLOW_NNUE_WEIGHTS` opt-in, contrastive `VariantSet` in #1044) is closed honest-negative (`docs/paper/2026-08-egraph-nnue-parity.md`, #1072 §5.3: the static table ties it, every lever made it worse; static stays default). The Guide program is live: Round 1 (#1084) is a linear candidate-local model + per-rule control + dedup (`egraph/candidate.rs`, `nnue/guide/linear.rs`, `training/guide_linear.rs`); Round 2 is |R|-scaling; the candidate-context training design (DOWN bound-class summaries, UP parent-op sets 1-/2-hop, STATE dcost/on-best-path/budget; enumerable context cells; coverage table; rule-conditioned per-rule-balanced generation; one fixed offline model) was approved 2026-09-01 but its doc/branch (`claude/phase3-context`) is **not on origin** — the on-main design of record is `docs/plans/2026-08-31-guide-design-revision.md` §4, which already argues candidate-local over whole-graph.

**Evidence discipline.** LIVE = a non-test caller in `pixelflow-compiler/-ir/-codegen/-runtime/-graphics/core-term`, or a pipeline bin the docs/plans cite as a current harness. DORMANT = public API with tests/benches only or no callers. DEAD = unreachable. Every row cites `file:line` and the grep that produced the status. LOC are file totals; where a `#[cfg(test)]` boundary was measured, non-test LOC follow in parentheses.

---

## 1. The table

Sorted by verdict (KEEP, REUSE-R2, SEGREGATE, DELETE), then size.

| item | size | status + evidence | the decent idea | verdict |
|---|---|---|---|---|
| `pixelflow-pipeline/src/bin/bootstrap_extraction_head.rs` | 2236 | LIVE (harness) — 10 docs/plans+results refs (`grep -rl bootstrap_extraction_head docs`); sole trainer that emits the `PIXELFLOW_NNUE_WEIGHTS` checkpoint and sole caller of `unified_backward`/`episodes` | trajectory-paired (initial, final) supervision through the typed edge stream; the trainer of record for the closed negative result | KEEP |
| `pixelflow-search/src/nnue/factored.rs` — `ExprNnue` (1511–1543), `OpEmbeddings` (325–437), `CostEdge`/`EdgeTrace`/`EdgeSink` (602–703), `EdgeAccumulator` (705–742, 1178–1362), TRIF save/load (1823–1944) | 2159 | LIVE — `grep -rn "PIXELFLOW_NNUE_WEIGHTS\|IncrementalExtractor" pixelflow-compiler/src/optimize.rs` → lines 101, 973, 1421, 1524–1549 (opt-in policy selection); `grep -rln "EdgeTrace\|CostEdge" pixelflow-pipeline/src` → `unified_backward.rs`, `training/mod.rs`, `bootstrap_extraction_head.rs` (#1063) | typed edge stream: the accumulator walk is a value (`EdgeTrace::realize`), so embeddings train through the same fold the forward pass runs | KEEP (see §6 D1 on whether the opt-in itself stays) |
| `pixelflow-pipeline/src/jit_bench.rs` — `BenchSession` (371–1425) | 1915 | LIVE — `grep -rln "jit_bench::" pixelflow-pipeline/src/bin` → bench_jit_corpus, bench_jit_compile_cost, bench_extraction_3way, bootstrap_extraction_head; also `examples/measure_latency_prior.rs` | QoS pinning, DVFS burn-in, drift sentinel, tick-floor autoscaling; `LocalNs`/`SessionNs` newtypes enforce correction order | KEEP |
| `pixelflow-pipeline/src/training/quarantine.rs` (80–1308) | 1308 | LIVE — `grep -rl "training::quarantine" pixelflow-pipeline/src/bin` → gen_bench_corpus, bench_extraction_3way, bootstrap_extraction_head | one same-form JIT-vs-oracle gate shared by every label source, so a drifting duplicate predicate cannot reopen audit B8's leak | KEEP |
| `pixelflow-search/src/nnue/mod.rs` — `ExprGenerator` (73–217), `pattern_match_arena`/`substitute_template_arena` (218–396), `BwdGenerator` (397–1161) | 1262 | LIVE — `grep -rn "BwdGenerator" pixelflow-pipeline/src/bin pixelflow-search/examples` → gen_bench_corpus.rs:65,831; bootstrap_extraction_head.rs:105,1019; oracle_filtered_budget_curves.rs:131,295 | Lample & Charton 2019 backward generation: mint (expr, rewritten-expr) pairs by applying rule templates in reverse — the corpus generator the coverage table will be built on | KEEP |
| `pixelflow-pipeline/src/training/split.rs` (32–1173) | 1173 | LIVE — `grep -rl "training::split" pixelflow-pipeline/src/bin` → gen_bench_corpus, bench_extraction_3way, bootstrap_extraction_head | `Fence<DevSide/FinalSide>` — holdout side typed, so DEV/TRAIN cross-contamination is a compile error | KEEP |
| `pixelflow-pipeline/src/training/corpus.rs` (114–526) | 994 | LIVE — `grep -rl "training::corpus" pixelflow-pipeline/src/bin` → 8 bins incl. Guide-program bins `guide_headroom.rs`, `guide_scope_saturation_delta.rs:122` | versioned binary corpus with `corpus_identity`/`opcode_encoding_identity` hash guards | KEEP |
| `pixelflow-pipeline/src/training/factored.rs` (22–653) | 898 | LIVE — `grep -rl "training::factored" pixelflow-pipeline/src/bin` → validate_corpus, bench_jit_corpus, bench_extraction_3way, bootstrap_extraction_head; `grep -n BwdGenerator` → 0 hits (the memory-noted `Expr` shim debt is already paid) | `arena_to_kernel_code`/`parse_kernel_code_arena` — a human-readable round-trip for arenas | KEEP |
| `pixelflow-pipeline/src/training/mint.rs` (106–390) | 735 | LIVE — `grep -rl "training::mint" pixelflow-pipeline/src/bin` → bench_jit_corpus, bench_extraction_3way, bootstrap_extraction_head | `normalized_label_ns`: drift correction before overhead subtraction; hard-errors on a sessionless `BenchResult` | KEEP |
| `pixelflow-search/src/egraph/labeler.rs` — `EpisodeLabels`, `run_episode` consumers | 536 | LIVE — `grep -rln "labeler::\|EpisodeLabels" pixelflow-pipeline/src pixelflow-search/examples` → guide_headroom.rs, guide_scope_saturation_delta.rs, training/episodes.rs, oracle_filtered_budget_curves.rs, rule_report.rs | hindsight labeling — the e-graph as an audit log; load-bearing vs wasted firings, no critic | KEEP (Guide substrate) |
| `pixelflow-search/src/egraph/provenance.rs` — `Provenance`, `derivation_ancestors` | 507 | LIVE — `grep -c provenance pixelflow-search/src/egraph/graph.rs` → 51 (production saturate loop journals every union) | stable `ENodeId` + append-only origin/application/union journals that survive `rebuild_budgeted` churn | KEEP |
| `pixelflow-pipeline/src/bin/profile_extraction.rs` | 417 | LIVE (gate) — cited as the acceptance test by `pixelflow-search/src/egraph/extract.rs:140` ("Confirmed by this crate's determinism harness (`profile_extraction`'s digest)"); touched by open #1085 | determinism digest over chosen forms + predicted costs: a "performance-only" change that moves the digest is a semantics regression | KEEP (it gates any extraction change, not just the closed program) |
| `pixelflow-pipeline/examples/measure_latency_prior.rs` | 378 | LIVE — `grep -n measure_latency_prior pixelflow-search/src/egraph/cost.rs` → line 62 ("Protocol: …") | latency-chain slope at K=8/K=32 cancels call overhead; measures the JIT's own lowered form | KEEP |
| `pixelflow-search/src/egraph/extract.rs` — `IncrementalExtractor`/`try_swap`/`extract_neural_to_arena` (345–673) | 329 of 2588 | LIVE — `pixelflow-search/src/egraph/extraction.rs:56` (`ExtractionPolicy::Nnue` → `IncrementalExtractor::new(model, 8)`), reached from `pixelflow-compiler/src/optimize.rs:973` | cycle-checked incremental cost-swap extraction | KEEP |
| `pixelflow-pipeline/src/training/structural.rs` — `FenceKey`/`QuotientNode` (44–178) | 178 | LIVE — `grep -rl "training::structural" pixelflow-pipeline/src/bin` → gen_bench_corpus, bootstrap_extraction_head | literal-blind, OpKind-identity structural key: `X*2.0` and `X*3.0` cannot land on opposite sides of a fence | KEEP — and it is the ready-made **coverage-cell dedup key** for Round 2 (no port needed) |
| `pixelflow-pipeline/src/bin/validate_corpus.rs`, `bench_jit_profile.rs` | 160, 97 | LIVE / cheap utility — validate_corpus is the mint step feeding gen_bench_corpus (self-documented `cargo run` header); bench_jit_profile is a pprof harness with only a Cargo.toml self-ref | — | KEEP |
| `pixelflow-search/src/egraph/extraction.rs` | 128 | LIVE — the one policy enum both `kernel!` and `optimize_runtime_arena` go through | one policy, one env-var gate, no duplicate opt-in path | KEEP |
| `EdgeAccumulator::remove_*` (anti-row A6) | — | LIVE API, dormant justification — A6's "~2×" payoff was for the extraction head's hot path, which lost | — | KEEP as documented (harmless); note the justification is gone |
| `pixelflow-pipeline/src/training/unified_backward.rs` — `backward_value` (514) → `backward_expr_proj_and_backbone` (574) → `backward_edge_tower_from_hidden` (623) → `backward_through_accumulator` (672) | 1841 (964) | DORMANT-in-practice — `grep -rln "unified_backward" pixelflow-pipeline/src/bin` → only bootstrap_extraction_head.rs:119,1463 (the closed program) | staged analytical gradients that differentiate the *typed edge stream* (`EdgeTrace::realize`) into op embeddings — the forward fold and its adjoint are the same object | **REUSE-R2** → the backward for training op embeddings through the UP-cell binding (§2 R1) |
| `pixelflow-search/src/nnue/guide/accumulator.rs` — `GraphAccumulator` (47–345) + `shift_by`/`shift1` in `guide/mod.rs:51–63` | 934 (346) + 15 | DORMANT-LIBRARY, self-declared — `guide/mod.rs:11` "This module is inert today"; `accumulator.rs:13` `#![allow(dead_code)]`; `grep -rn GraphAccumulator` outside `nnue/guide/` → doc comments only; J10 ROADMAP-ADMITTED (`2026-08-17-cost-model-domain.md:43`) | four K=32 sections: marginal parent/child sums + 1-hop `E[p] ⊙ shift(E[c])` + 2-hop `E[gp] ⊙ shift(E[p]) ⊙ shift²(E[c])` — a 1-round GNN over op adjacency | **REUSE-R2** — the *binding primitive*, not the running-sum state, at candidate scope = the UP cells (§2 R1). `2026-08-31` §4 already made the whole-graph→candidate-local argument; #1084 built `CandidateFeatures` without it |
| `pixelflow-pipeline/src/training/episodes.rs` — `run_episode` (160), budget randomization (230–240) | 635 (473) | DORMANT-in-practice — `grep -rl "training::episodes" pixelflow-pipeline/src/bin` → only bootstrap_extraction_head.rs:111 | hash-seeded, size-scaled budgets: `budget_mult = 3 + (hash%8)`, `node_budget = (50+mult·nodes).min(2000)`, `epoch_budget = 10 + (hash·7 % 51)` — diversity in saturation depth per seed, deterministic | **REUSE-R2** → the coverage table's `budget_fraction` cell sampler, re-denominated in applications per `anytime.rs` (§2 R3) |
| `pixelflow-ml/src/graphics.rs` — `ShFeatureMap::project` (177), `RandomFourierFeature` (64), `LinearAttention` (217) | 371 (293) | DORMANT-LIBRARY — `grep -rln "pixelflow_ml" --include='*.rs' .` → only itself; the four pipeline example headers claim `cargo run -p pixelflow-ml` for examples that live in pixelflow-pipeline; last touched 2026-07-02 (rename-only) | linear attention ≡ SH-basis projection; an SH/RFF feature map is a trig-and-polynomial kernel family with known structure | **REUSE-R2 (idea only)** → an OOD corpus band `sh` in `gen_bench_corpus` expressed as `BwdGenerator` arena templates (§2 R4). The `Field`-level code itself cannot be called by the generator; it stays where it is |
| `pixelflow-search/src/nnue/guide/scoring.rs` — `encode_rule_from_arena` (320), `bilinear_score` (355) | ~60 | DORMANT — same `#![allow(dead_code)]` (scoring.rs:11), tests-only callers | `[LHS ‖ RHS ‖ LHS−RHS ‖ LHS⊙RHS]` rule-pair encoding + a (context, rule) bilinear interaction — the rung above a linear model with a per-rule intercept | **REUSE-R2** → the nonlinear rung after `linear.rs`; #1084's scoring.rs already adds a `candidate_w1`/`candidate_proj_w` tower (branch scoring.rs:97–103) that this bilinear can sit on (§2 R2) |
| `OpEmbeddings::init_with_latency_prior` / `ExprNnue::new_with_latency_prior` (`factored.rs:364–437`, 1588) | ~55 | DORMANT — `grep -rn init_with_latency_prior` outside factored.rs → cost.rs doc comments (34, 215), extract.rs tests (1940, 2215, 2302), `prod_kernel_jit.rs:62` (test), `judge_weights_load.rs:30` (test); `bootstrap_extraction_head.rs:910` uses `new_random` | seed embedding dim 0 with the static table so an untrained op already carries a sane cost prior | **REUSE-R2** → init for the candidate-context op embeddings in `train_guide` (§2 R1) |
| `pixelflow-search/src/nnue/guide/scoring.rs` — whole-graph tower `forward_graph` (193), `compute_graph_embed` (243), `mask_score_all_rules_graph` (281) | ~130 | DORMANT — as above; #1084's `guide/mod.rs:27–35` keeps it explicitly as "a live roadmap seam for a future whole-graph Judge, not a Guide-scoring code path" | accumulator → shared trunk → embedding | SEGREGATE — already done: private module, `allow(dead_code)` as the declared seam, wrapper types removed on #1084. No action |
| `docs/plans/2026-02-25-unified-training-{design,plan}.md` | 254 + 1016 | STALE — describe `train_online`/`collect_guide_data`/Python critic self-play (design.md:5–6, plan.md:5); CLAUDE.md: that loop "was removed in July 2026"; last touched 2026-03-01 | — | SEGREGATE → `docs/plans/archive/` with a one-line pointer to the 07-07 post-mortem |
| `docs/GNN_REWRITE_GUIDANCE_VISION.md` (2026-01-30), `docs/designs/nnue-training-pipeline.md` (2026-01-25) | 403, 379 | STALE — vision doc cites `pixelflow-ml/src/nnue.rs` and 401,408 HalfEP features (lines 19, 23); design doc predates the arena IR ("Replace MCTS with Curriculum Learning") | — | SEGREGATE → `docs/archive/` (same move) |
| PR #994 `claude/macos-release-signing-pipeline` | +? | JP's; blocked on five Apple secrets that do not exist (#1086 §"Stalled") | — | SEGREGATE — JP's call; not part of this inventory |
| PR #1044 `claude/round2b-contrastive` — `egraph/variants.rs` 281, `training/variant_set.rs` 711, `bin/mint_variant_sets.rs` 476, `bin/train_contrastive.rs` 1030, `training/stats.rs` 192 | 2690 (never on main) | Confirmed regression — paper §5.3 (branch `claude/workshop-writeup`, line 536ff): end-to-end 1.0082, 95% CI [1.0031, 1.0133]; domain doc retirement note (`2026-08-17-cost-model-domain.md:343`): "J6/J8 fall out if Round-2 contrastive training is abandoned" | `VariantSet` — ranking within an e-class, the only metric that discriminated models | DELETE = close the PR; cherry-pick only its 2 `docs/results/journal.jsonl` lines so #1072's Round 2b trace survives (§3, §5) |
| `pixelflow-pipeline/examples/{op_cost_stress,egraph_choices,ilp_benchmark,critical_path_test}.rs` | 83+215+165+173 = 636 | DEAD — `grep -c pixelflow` → 1 each (the stale header `cargo run -p pixelflow-ml --example …`); scalar `f32` chains with `use std::time::Instant` only; last touched 2026-04-13/06-10; 0 docs refs; superseded by `measure_latency_prior.rs` (2026-08-17, cited by cost.rs:62) | — | DELETE |
| `CostModel::{save_toml,load_toml,load_or_default,from_map,to_map}` (`pixelflow-search/src/egraph/cost.rs:325–503`) + `PIXELFLOW_COST_MODEL` / `$HOME/.config/pixelflow/cost_model.toml` / `pixelflow-ml/data/learned_cost_model.toml` probing + `pixelflow-core/src/bin/calibrate_costs.rs` | 184 + ~190 tests + 322 | DEAD — `grep -rn "load_or_default\|save_toml\|load_toml\|from_map(" --include='*.rs'` outside cost.rs → only `calibrate_costs.rs:198,225` (its own private `save_toml` over its own `CostModelData`); `load_or_default` has **zero** callers; `pixelflow-ml/data/` does not exist; calibrate_costs last touched 2026-06-11, cited only by `docs/COMPILER_ANALYSIS.md:242` as "(NEW)" | measured per-op override table (the idea now lives in `measure_latency_prior.rs` + the pinned `latency_prior_cycles()`) | DELETE — a `HOME`-dependent silent cost-model override with no caller is exactly the "runtime guard defending a comment" the codebase forbids; see §6 D3 (two mutation-test audits, incl. just-merged #1051, spent effort here) |
| `pixelflow-ml/NNUE_TRAINING.md` | 367 | DEAD — lines 342–348 point at `pixelflow-ml/src/nnue.rs`, `src/training/egraph.rs`, `examples/guided_training.rs`, `data/*.bin`: none exist (`ls pixelflow-ml/src` → `graphics.rs`, `lib.rs`); dated 2026-01-31 | — | DELETE |
| `pixelflow-search/src/nnue/mod.rs:42–68` — legacy `MAX_DEPTH = 8`, `DEPTH_LIMITED_MAGIC`, `DEPTH_LIMITED_VERSION`, five empty "HalfEP/Dense/Network/Training/Binpack" section banners | ~25 | DEAD — `grep -rn "DEPTH_LIMITED_\|nnue::MAX_DEPTH"` → 0 callers; `factored.rs:80` defines its own `MAX_DEPTH = 192` | — | DELETE |

Not rows, but checked: `pixelflow-search/src/egraph/candidate.rs` (446), `anytime.rs` (366), `nnue/guide/linear.rs` (473), `training/guide_linear.rs` (343) exist only on #1084 — in-flight, not dead; they are the Round-1 candidate-scope seam the reuse rows plug into. `math/inflate.rs`, `oracle.rs`, `egraph/template.rs` on `claude/phase3-round2` are WIP harness, KEEP on their branch. J11's duplicate saturation loop (`saturate.rs:91,104` vs `graph.rs:818,835`) is still duplicated on main; #1085 deletes it.

---

## 2. REUSE-R2 seams

Per JP: encapsulate, tightest contract. Each seam exposes the idea at candidate scope and retires the whole-graph path it came from.

**R1 — UP cells = the binding primitive at candidate scope** (`accumulator.rs` + `unified_backward.rs` + `init_with_latency_prior`).
- Contract: one `pub(crate) fn bind_neighborhood(emb: &OpEmbeddings, match_op: OpKind, parents: &[OpKind], grandparents: &[(OpKind, OpKind)]) -> [f32; 2*K]` in `nnue/guide/` — section 0 = Σ `E[p] ⊙ shift(E[match])` (1-hop), section 1 = Σ `E[gp] ⊙ shift(E[p]) ⊙ shift²(E[match])` (2-hop). It is `add_edge_at_depth`/`add_2hop_edge` (`accumulator.rs:102,249`) with the accumulator struct's running-sum state, `remove_*` mirrors, `normalized()`, and the 4 scalar budgets subtracted.
- Consumer: `CandidateSummary` (#1084 `guide/mod.rs:138`) / `CandidateFeatures::observe` (#1084 `egraph/candidate.rs`), which already walks the one-hop neighborhood — it gains one field.
- Gradient: `backward_through_accumulator` (`unified_backward.rs:672`) already differentiates a Hadamard-with-shift fold through `EdgeTrace`; the UP cell's adjoint is the same function over a `CandidateEdgeTrace` of ≤ 2 hops. Do not re-derive it.
- Init: seed `OpEmbeddings` dim 0 with `init_with_latency_prior` (`factored.rs:385`) in `train_guide` before the first SGD step, so the STATE cell's "dcost under the table" and the UP cell's op identity share a scale from step 0.
- Retire: once `bind_neighborhood` exists, `GraphAccumulator` (346 non-test LOC + 590 test LOC) has no consumer left on the Guide side; the "future whole-graph Judge" role #1084's doc reserves for it is the closed extraction-head question — decision D2.

**R2 — the nonlinear rung = bilinear over (candidate tower, rule encoding)** (`scoring.rs:320–378`).
- Contract: `SaturationHead::score(candidate_embed: &[f32; EMBED_DIM], rule: &[f32; 4*EMBED_DIM]) -> f32` = `bilinear_score` as written, fed by #1084's `candidate_w1`/`candidate_proj_w` tower instead of `compute_graph_embed`. `encode_rule_from_arena` stays byte-for-byte: `[LHS ‖ RHS ‖ LHS−RHS ‖ LHS⊙RHS]` is already the rule-conditioned encoding the design's per-rule-balanced generation wants a model to condition on.
- Gate: only after `linear.rs`'s `LinearCandidateGuide` vs `PerRuleRateGuide` shows a real gap (that is the purpose of the control arm, `linear.rs:26–44`). A bilinear that cannot beat a per-rule intercept is the extraction-head story again.

**R3 — coverage-table budget cells = `run_episode`'s sampler, re-denominated** (`episodes.rs:230–240`).
- Contract: `fn budget_cell(seed_hash: u64, expr_nodes: usize) -> ApplicationBudget` — the same hash-seeded, size-scaled draw, but returning an application count on `anytime.rs`'s geometric grid, never epochs or node caps (applications are the registered x-axis; `anytime.rs:1–25`). Used by the per-rule-balanced generator to spread each rule's candidates across `budget_fraction` cells instead of letting them pile up at the start of saturation.
- Retire: `run_episode` itself (JIT-benchmarks initial and final cost — the closed program's labels) is not needed; the Guide trains on static-table `dcost`.

**R4 — the `sh` OOD band** (`pixelflow-ml/src/graphics.rs`).
- Contract: a `BwdGenConfig` band (`gen_bench_corpus.rs:824`) whose templates are the SH-basis polynomials of `ShFeatureMap::project` (`graphics.rs:177`: the nine `l ≤ 2` terms in x, y, z) and the `cos(ωx+φ)` RFF form (`graphics.rs:64`), written as arena templates. Nothing from the crate is linked; the idea is the family. The crate itself has no live path paying for it and stays a workspace member until JP decides D4.

---

## 3. DELETE now

Dead + superseded + no roadmap. **On-main total ≈ 1,357 LOC of Rust + 367 lines of stale doc**; plus one PR closed (2,690 LOC that never lands).

| what | LOC | why |
|---|---|---|
| `pixelflow-pipeline/examples/op_cost_stress.rs`, `egraph_choices.rs`, `ilp_benchmark.rs`, `critical_path_test.rs` | 636 | scalar `f32` loops that touch no pixelflow crate; headers point at a crate they do not live in; `measure_latency_prior.rs` measures the JIT's own emitted form and is the protocol cost.rs cites |
| `CostModel::{save_toml,load_toml,load_or_default,from_map,to_map}` + their tests (`cost.rs:325–503`, test fns matching `load_or_default_*`/`load_toml_*`/`save_toml_*`/`unique_temp_dir`/`run_child`) | 184 + ~190 | zero callers; `load_or_default` probes `$HOME` and a nonexistent `pixelflow-ml/data/` and would silently swap the production cost model if a file appeared; the table is pinned in `latency_prior_cycles()` and re-derived by `measure_latency_prior.rs` |
| `pixelflow-core/src/bin/calibrate_costs.rs` (+ its `[[bin]]` in `pixelflow-core/Cargo.toml:25`) | 322 | the only writer of the file `load_or_default` reads; measures `Field` ops at 3 GHz-assumed cycles, superseded by the JIT latency-chain protocol |
| `pixelflow-search/src/nnue/mod.rs:42–68` legacy constants + empty section banners | ~25 | shadowed by `factored.rs:80`; no callers |
| `pixelflow-ml/NNUE_TRAINING.md` | 367 (doc) | describes files that do not exist |
| PR #1044 `claude/round2b-contrastive` | 2690 (branch) | measured regression; J6/J8 retire by the domain doc's own clause; would need a third port to land. Keep the 2 journal lines (cherry-pick), close the branch |

Not deleted, deliberately: `EdgeAccumulator::remove_*` (A6 says keep the API), `GraphAccumulator` (D2), `pixelflow-ml` crate (D4).

---

## 4. SEGREGATE

| what | where it goes |
|---|---|
| `nnue/guide/scoring.rs` whole-graph tower + `accumulator.rs` | already segregated: private module, `#![allow(dead_code)]` as the declared seam, wrapper types removed on #1084. Once R1 lands, the seam's roadmap holder (J10) should be re-read — see D2 |
| `docs/plans/2026-02-25-unified-training-design.md`, `-plan.md` | `docs/plans/archive/` (git mv), one header line: "Superseded 2026-07-07; system deleted July 2026" |
| `docs/GNN_REWRITE_GUIDANCE_VISION.md`, `docs/designs/nnue-training-pipeline.md` | `docs/archive/` (git mv), same header |
| `docs/COMPILER_ANALYSIS.md` (2026-07-24, cites `calibrate_costs` as "NEW") | if D3 deletes calibrate_costs, strike §"calibrate_costs" (lines ~217–279) in the same PR |
| PR #994 signing pipeline | JP's; stays open or closes on JP's call, outside this inventory |

---

## 5. Open PRs disposition

| PR | disposition |
|---|---|
| **#1044** round2b-contrastive | **Close.** Cherry-pick its 2-line `docs/results/journal.jsonl` change onto main first so #1072 §5.3's Round 2b number has a journal trace. #1086 offered merge-as-record; the record is the paper + journal, not 2.7k LOC behind a `training` feature |
| **#1049** graph.rs test renames (draft, +113) | #1086: "close or fold" — the cost.rs backlog item it corrects was closed by #1027 and now #1051. Close, or fold the one stale-note fix into whichever PR next touches `docs/bugs/2026-08-28-test-quality-audit-followup.md` |
| **#1051** cost.rs mutation gaps | **Merged** 2026-09-01 21:40. Note it added tests to the persistence block §3 proposes deleting (D3) |
| **#1054** x86_64.rs mutation gaps | Merge after the one #1086 caveat (rerun mutants on `emit_vpextrd_to_gpr`/`emit_vmovss_load_scaled` or soften the "0 real gaps" line) |
| **#1050** regalloc mutation tests | Already **closed** (#1055/#1068 deleted the graph-coloring allocator). Moot, as predicted |
| **#994** macOS signing | JP's |
| Context: **#1053** first (std-gate the NNUE opt-in), then **#1085** (J11 dedup), then **#1084** / **#1083** — per #1086's landing order; all three rewrite `saturate.rs` and only merge cleanly because none has landed | |

---

## 6. Decisions for JP

- **D1 — Does the `PIXELFLOW_NNUE_WEIGHTS` opt-in stay in the compiler?** The program is closed honest-negative; the opt-in is one `match` arm in `extraction.rs:56` plus `IncrementalExtractor` (329 LOC) and costs nothing at default. Keeping it keeps the negative result reproducible and keeps `profile_extraction`'s digest gate meaningful. Recommendation: keep, unchanged.
- **D2 — `GraphAccumulator` after R1.** #1084's doc reserves it for "a future whole-graph Judge" — that is the extraction-head question, which is closed. Once `bind_neighborhood` exists, the honest verdict is SEGREGATE→DELETE (346 + 590 test LOC) and J10 is retired by its own clause ("deleted whole if Phase 3 is cancelled" — Phase 3 was not cancelled, it moved to candidate scope). Recommendation: delete in the R1 PR, not before.
- **D3 — cost.rs persistence + `calibrate_costs`.** ≈700 LOC with zero callers and a `$HOME` probe. #1051 (merged today) and the 08-22 audit both spent mutation-testing effort on `load_or_default`; deleting it retires those tests too. Recommendation: delete; the sunk tests are not a reason to keep a silent override.
- **D4 — `pixelflow-ml` crate.** Zero dependents, 293 non-test LOC of `Field`-level attention/SH code, compiles in every `cargo test --workspace`. R4 needs the idea, not the crate. Options: keep as-is (cheap), or `git mv` to `docs/archive/pixelflow-ml/` and drop the workspace member. Recommendation: keep until R4 is written, then decide with the templates in hand.
- **D5 — #1044 close vs merge-as-record.** §5 recommends close + journal cherry-pick. If you prefer #1072 to cite merged code, merge behind `training` and mark `variant_set.rs` `#![allow(dead_code)]` with a J6 retirement note — but it will need a third port first.
- **D6 — archive vs delete for the four stale docs.** §4 says archive (git mv). If you would rather the tree not carry Jan–Feb design docs at all, `git rm` is defensible: the 07-07 post-mortem already summarizes them.
