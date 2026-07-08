# Guided Saturation Redesign

**Date:** 2026-07-07
**Status:** Plan — supersedes the RL training loop (`train_unified` REINFORCE + transformer critic)
**Builds on:** the 2026-07-07 four-agent audit of the training stack (summarized below),
docs/EGRAPH_SEARCH_INTEGRATION.md, docs/NNUE_TRAINING_RECIPE.md

## The thesis (unchanged)

Stockfish for algebra: an NNUE evaluates e-graph states like board positions; applying
rewrite rules is making moves; search + eval navigates a rewrite space far too large to
saturate. The payoff is **rule-library scale**: with a learned guide, the library can grow
to hundreds of rules (symbolic calculus included — `Dwrt` elimination is already a rewrite)
where full saturation is hopeless. The question the system answers:

> Can rule filtering keep the e-graph within budget while still reaching (nearly)
> the cheapest extraction — and thereby *enable* a much larger rule library?

## What the audit found (2026-07-07)

- **The machinery works.** Loop closes, PFTJ0002 byte-exact across Rust/Python, backprop
  finite-difference-verified, everything compiles.
- **The NNUE was dead in production**: stale `TRIC`-format weights silently rejected by the
  `TRID`-only loader; every `kernel!` compile fell back to a zero model (extraction = no-op).
  Fixed: default is now an explicit, documented no-op; `PIXELFLOW_NNUE_WEIGHTS` opt-in
  hard-fails on bad weights.
- **The RL half cannot learn in its current form** (see "Why the RL apparatus was unsound"):
  deterministic threshold policy under a REINFORCE estimator, no exploration (approved-only
  replay = one-way rule ratchet), survivorship bias (crashed trajectories censored), and the
  critic paradox — advantages → 0 as the critic improves, then batch normalization amplifies
  the residual noise to unit variance. Documented once in critic.py:45 and still open one hop
  away.
- **The policy has no consumer**: production only calls `saturate()`; A*/MCTS never integrated.
- **No recorded result**: the Feb 3-way (HCE vs Judge vs Guided) showed an overfit Judge
  beating HCE, but the numbers were never written down and the harness + HCE were deleted in
  the April squash. The best recorded number (kernel! 4.9% over LLVM, Apr 2) conflates e-graph
  rewrites with NNUE extraction.
- ~2.5k LOC confirmed dead (replay.rs, gen_es.rs, model.rs, nnue/training.rs, window.rs,
  cost_builder.rs, …); swept 2026-07-07.

## Why the redesign is shaped this way

The domain is a **deterministic, single-player, monotone, budget-bounded** game — the
AlphaTensor/AlphaDev setting, not the chess setting, and not solitaire (no hidden
information; uncertainty is computational, not aleatoric). Two structural gifts chess never
gets:

1. **Observed credit.** The e-graph is an audit log. After an episode, the winning
   extraction's derivation DAG (node origins + union causes) gives every fired match an
   exact, transitive, observed label: load-bearing or wasted. Enabling chains ("A enables B")
   are edges in that DAG. This replaces the critic/REINFORCE stack — which exists to
   *estimate* credit that chess cannot observe — with supervised targets.
2. **Cheap rollback.** A monotone arena e-graph snapshots by length and rolls back by
   truncation (+ a union-find journal). Speculative lookahead costs near-zero vs. MCTS
   cloning. Shallow search + strong eval — Stockfish's actual recipe — becomes affordable.

The residual sequential element provenance cannot see — chains never completed because
budget expired — is an **exploration** problem, handled by epsilon-random match firing
during collection and by iterating collect → label → retrain (the self-play *shape*
survives; every training target becomes an observed fact).

## Architecture

| Role | Module | Targets (all supervised) |
|---|---|---|
| **Judge** (position eval) | NNUE over e-graph state: predicts best-achievable extraction cost (log-ns) | Hindsight outcomes: best cost the episode achieved from each visited state |
| **Judge** (extraction) | Same net, expression mode (existing extraction head) | (expr, measured-ns) pairs from jit_bench corpus |
| **Guide** (move ordering) | Match scorer (repurposed mask head) | Provenance participation; later, search-chosen moves (AlphaZero policy-improvement, no adversary needed) |
| **Search** | Budget-bounded best-first over matches; Guide = prior, Judge = leaf eval; arena rollback for lookahead | — |
| **Loop** | collect (guide + ε) → hindsight-label → retrain → collect | — |

**Cut in every branch of the future:** transformer critic (all Python), REINFORCE gradient
path, advantage normalization, critic step-token trajectory machinery.
**Kept:** corpus generation, jit_bench ground truth (median-of-samples JIT wall-clock),
bootstrap_extraction_head, both NNUE heads (retargeted), the iterate-and-improve loop shape.

## Phases and decision gates

Each phase ends in a falsifiable, *recorded* result. Stop at any gate.

**Phase 0 — Cleanup (done / in flight).** Loud NNUE failure (done); dead-code sweep (done);
fail-loud extraction fixes (`unwrap_or(0)` missing-choice family); RL apparatus cut.

**Phase 1 — Provenance.** Node origins (which rule application created each e-node), union
journal (which rule caused each merge), hindsight labeler (transitive ancestors of the
winning extraction). Bonus: "explain this optimization" derivation traces (egg-style proof
production). Pure compiler infrastructure — valuable even if all ML dies.

**Phase 2 — Judge offline.** Retrain on corpus → valid TRID weights → run the deferred
extraction bench: **NNUE vs latency prior vs no-swap**, JIT-benched, numbers committed to
docs/. *Gate:* if NNUE ≤ latency prior, port the prior into the static `CostModel` as the
default and keep NNUE opt-in only.

**Phase 3 — Greedy guided saturation (the thesis test).** Guide trained on provenance
labels; budget-bounded greedy expansion; first calculus/trig rule batch. Run **the**
experiment: big-library guided vs small-library full saturation, same budget, JIT-benched.
*Gate:* if guided can't match full saturation even greedily, record it and stop — the thesis
is answered.

**Phase 4 — Search (conditional).** Only if Phase 3 shows under-credited chains (oracle
runs contain deep derivations guided runs never find): arena rollback, shallow beam
lookahead, search-as-policy-target.

## Why the RL apparatus was unsound (experimental-design summary)

1. **Observational study analyzed as an RCT** — REINFORCE requires sampled actions;
   the policy was a deterministic threshold. No sampling = no counterfactual = the
   log-prob gradient estimates nothing.
2. **Selection bias in collection** — approved-only replay (rules below threshold stop
   generating data forever) + censored failures (crashed trajectories dropped, so
   catastrophic actions leave no negative signal).
3. **Advantage collapse + renormalization** — with a deterministic policy the critic can
   in principle predict returns exactly; advantages decay to critic noise, which
   `normalize_advantages` rescales to full gradient strength. The better the critic,
   the purer the noise trained on.
4. **Unit mismatch** — critic sees per-epoch summaries (no rule identity); one epoch
   advantage is broadcast to every approved rule. Group-level measurement, individual-level
   treatment.

None of these are bugs; they are properties of the estimator chosen. The redesign avoids
the estimator rather than patching it.
