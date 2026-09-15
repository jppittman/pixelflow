> **Retracted/Superseded (2026-09-07), ledger L047, L048.** The training-quality comparison (AUC / PR-AUC) is on a synthetic corpus and was never taken on a shipped shader; the model class is kept as an instrument only. Verdict and rationale: `docs/results/2026-09-07-claims-ledger.md` (PR #1207); the corrected benchmark and re-validation order: `docs/plans/2026-09-07-benchmark-correction.md`.

# Training the bilinear Guide: the model class, its gradients, and its held-out ranking quality

**Date:** 2026-09-02
**Registration:** `docs/plans/2026-09-02-bilinear-guide-registration.md` (frozen; not
edited by this document).
**Scope:** this is the **training** half of that registration's §9 protocol — the backward
pass, the deployed `SaturationGuide`, the cold-start fit, the mandatory skew test. It runs
**no guided saturation** and reports **no `D_A(S, B)`**. The §3.3 gates are untouched and
unclaimed; nothing here may be read as evidence for or against H_repr.

---

## 0. What was actually asked, and what the answer is

Every Phase 3 Guide measurement this week ran `LinearCandidateGuide`, whose score is
additively separable — `s(r, x) = w_rule[r] + g(x)` — so no context can reorder two rules
against each other. The registration §1 establishes that as a fact about the functional
form. What did not exist was a *trained* implementation of the other functional form.

Now it does. On the same labels, the same features, the same class weighting and the same
cold start:

| held-out DEV (family-held-out, never trained on, never used to select anything) | additive | **bilinear** | constant-predictor floor |
|---|---:|---:|---:|
| AUC-ROC (pooled) | 0.9451 | **0.9558** | 0.5 |
| PR-AUC (average precision) | 0.6837 | **0.7624** | 0.1183 |
| AUC within expression (the move-ordering question), macro over 116 expressions | 0.9342 | **0.9524** | 0.5 |

Both arms are trained by this branch's own code on this branch's own dataset (§2 explains
why the 2026-09-01 numbers cannot be reused). The bilinear arm is better on all three, and
the trained model's **rule-by-context second difference is nonzero** — the quantity the
additive arm pins at exactly zero by construction — so the capacity is not merely present
in the functional form, training reaches it.

**None of that is the registered claim.** Held-out classification quality on strict labels
is not `D_linear − D_bilinear` on `sh` at B=100. A better classifier of the load-bearing
bit is a necessary condition for the guided win, not evidence of it, and §8 of the
registration already names the case where the label itself is the bottleneck. The guided
evaluation is the next task and its gates are unchanged.

---

## 1. What was built

| piece | where |
|---|---|
| backward pass over the whole candidate-scoring path, plus the SGD step | `pixelflow-search/src/nnue/guide/scoring/backward.rs` |
| `BilinearCandidateGuide` (deploy) + `BilinearTrainer` (train) + `BilinearWeights` | `pixelflow-search/src/nnue/guide/bilinear.rs` |
| checkpoint schema, record→candidate encoding | `pixelflow-pipeline/src/training/guide_bilinear.rs` |
| the trainer | `pixelflow-pipeline/src/bin/train_guide_bilinear.rs` |
| the mandatory skew test | `pixelflow-pipeline/src/bin/skew_test_bilinear_guide.rs` |

### 1.1 Gradient validation

Every trained tensor is pinned by a **central-difference check against the forward pass
itself**, in the pattern `nnue::factored`'s `numerical_gradient_check_embeddings`
established, on a fixture whose weights are index-asymmetric so a transposed index
produces a different number rather than the same one. `interaction` and `trunk_w` are
square, so `m[i]·dt[j]` and `m[j]·dt[i]` are both well-typed and both plausible in a loss
curve; those two get explicitly *paired* off-diagonal probes, `(i,j)` and `(j,i)`.

The first version of that fixture **passed vacuously**, and the way it was caught is worth
recording. With `SaturationHead::randomize`'s He initialisation and realistic input scales,
only 5 of the mask MLP's 16 hidden units are open; a slightly different fixture closed all
16, `d_g` became identically zero, every gradient below it became zero — and a central
difference of a locally-constant function is *also* zero, so `assert_close(0, 0)` passed
for every tensor. `every_relu_on_the_path_should_be_open_at_the_fixture_point` now asserts
the fixture is live before any gradient is compared, and the per-row non-vacuity check on
the four scalar rows asserts each one actually receives gradient.

That failure mode is not confined to the test. If training closes all 16 mask units,
`mask_features` collapses to the constant `mask_mlp_b2`, the bilinear score becomes
`(constᵀW + bᵀ)r`, and **the model is additively separable** — the head would silently
become the very thing it is being compared against. Two defences:

- `SaturationHead::warm_relu_biases` offsets every ReLU bias positive at cold start (an
  initialisation, not a constraint: training may still close a unit that should close).
- `train_guide_bilinear` measures the trained second difference and **refuses to write a
  report** if it is zero.

### 1.2 What is trained and what is frozen

**Trained:** `candidate_w1/b1`, `trunk_w/b`, `candidate_proj_w/b`, `mask_mlp_w1/b1`,
`mask_mlp_w2/b2`, `interaction`, `mask_bias_proj`, and `rule_proj_w/b`.

**Frozen: the op embeddings**, at `OpEmbeddings::new_with_latency_prior(seed)`. Stated,
not defaulted into:

- They carry the latency-prior initialisation from #1063 and are a crate-shared asset,
  not this head's private parameters.
- *Redundancy.* The candidate tower's first layer is a full linear map `K → HIDDEN_DIM`
  applied to the pooled embedding, so training `E` and training `candidate_w1` move the
  same function through the same subspace. Adding `E` buys no expressiveness and costs a
  second, divergent copy of a shared object inside this checkpoint.
- *Confounding.* The additive arm has no op embedding — it has one scalar per op. Letting
  the bilinear arm re-learn a 50×32 embedding would be a capacity difference on the
  *context* side, and the registration licenses a difference only in the functional form.

**The rule embedding is trained through its encoder, not as a free table.** §4 registers
the `[LHS | RHS | LHS−RHS | LHS⊙RHS]` concatenation, so `rule_proj_w/b` carry the gradient
and `r_j = P(c_j)` is derived. §1.3 is why that is not a handicap.

### 1.3 The rule encoding, and one recorded limitation

The tower that used to produce `z_LHS`/`z_RHS` left with the extraction head, so the side
embeddings are **depth-bound poolings of each rule's own templates**: each template node's
op embedding is bound to its depth with `shift_by` — the same VSA permutation
`GraphAccumulator` already uses — and summed `1/√n`.

Depth binding is load-bearing, and this is measured rather than assumed. A **flat** bag of
template ops is topology-blind, and this vocabulary contains pairs that differ only in
topology: `associative(Add)` vs `reverse-associative(Add)`, `exp-ln-cancel` vs
`ln-exp-cancel`, `exp2-log2-cancel` vs `log2-exp2-cancel`. Flat pooling gave each pair one
embedding — one score in every context, at any weights — and the concatenation matrix had
**rank 57 of 62**. Depth binding separates all three pairs and lifts it to **rank 60 of
62**. In the trained model `associative(Add)` and `reverse-associative(Add)` now carry
distinct scores (mean predicted 0.000166 vs 0.000192) and their DEV-measured rates do
differ in the same direction (0.00388 vs 0.01163) — a distinction the flat encoding could
not have expressed at any weights. Both are still far below the measured rates, which is a
statement about how hard those two rules are to tell apart from context, not about the
encoding.

Rank matters because `P` is a free `128 × 32` map: if the `c_j` are linearly independent,
some `P` realizes *any* per-rule embedding assignment, so training the rule embedding
through its encoder is exactly as expressive as a free per-rule table — and generalizes to
a rule the table has never seen.

**The remaining limitation, recorded rather than absorbed.** Two rules —
`constant-fold` and `differentiate` — define **no template on either side**, so the
registered encoding has nothing to encode for them and gives both the zero vector. They are
therefore indistinguishable to this model: one score, in every context, at any weights.
`constant-fold` is not a rare rule (289 DEV firings, measured strict rate 0.3668, mean
predicted 0.3483), so this is a real cost, not a technicality: whatever the model has
learned about `constant-fold` it is also saying about `differentiate`. It is a property of the *registered arm definition*, not a
difference introduced during training, and it is pinned:
`the_rule_encoder_should_separate_every_rule_that_has_a_template` fails if any rule *with* a
template ever collides.

---

## 2. The dataset had to be re-minted, and both arms were retrained on it

The task asked for the same strict-v1 dataset and recipe the additive model used. That
turned out to be impossible on this branch, and the reason is worth stating precisely
because it changes what the additive numbers in §0 are.

- The 2026-09-01 `strict_labels_*.jsonl` predates the RuleId-keyed row schema: its rows
  carry `rule_idx` and no `rule_id`. This branch's `Record` requires `rule_id`, so
  **neither** trainer can read those files.
- The frozen `guide_checkpoint_strict_v1.json` names 61 rules; `RuleSet::production()` on
  this branch has 62. It is already off-vocabulary here and
  `LinearWeights::check_vocabulary` refuses it — correctly.
- Re-minting from the identical corpus (`corpus_train.bin` MD5
  `0ed6cf16abcbc006cd7a3ee2365b15b4`, `corpus_dev.bin`
  `3026133ebba066eeca10f658da554400`, both verified) does **not** reproduce the old
  dataset: per-expression applications fall from ~1591 to ~757 and the pooled strict-positive
  rate rises from 0.52% to 0.99%. That is the instrument, not noise — #1118 binds the
  application budget mid-scan, #1117 changed the reported cost, #1120 fixed
  `rebuild_budgeted`'s orphaned e-nodes, and the rule set grew.

So: one dataset, minted on this branch
(`docs/results/2026-09-02-strict-label-dataset-remint.json`), and **both** arms trained on
it by this branch's own binaries. That is what makes §0 a functional-form comparison. The
frozen strict-v1 and R2G checkpoints are **untouched** — the additive arm here is a new
file at `guide_checkpoint_strict_remint.json`, written for this comparison only, and it is
*not* the registration's frozen `LinearCandidateGuide` arm.

| | 2026-09-01 mint | this re-mint |
|---|---:|---:|
| TRAIN expressions / families | 415 / 223 | 385 / 223 |
| TRAIN applications | 660,404 | 291,346 |
| TRAIN pooled positive rate | 0.522% | 0.991% |
| DEV expressions / families | 131 / 54 | 126 / 54 |
| DEV applications | 155,231 | 92,589 |

After excluding `dedup_repeat` rows — which both arms exclude, because the live guided loop
removes seen keys *before* scoring and so never ranks them — TRAIN is 22,697 candidates at
12.41% positive and DEV is 6,838 at 11.83%. `pos_weight = 7.060` (inverse class frequency,
`neg/pos` on the TRAIN split actually used), identical in definition for both arms.

---

## 3. What is held equal between the arms, and the two things that are not

| held equal | how |
|---|---|
| label source | the same `strict_labels_{train,dev}_v2.jsonl` (TRAIN FNV-1a `11aade71b4549318`, DEV `1c4d32940a8ff286`) |
| candidate features | the same five context quantities off the same `CandidateSummary`: the neighborhood op multiset (pooled here, counted there) plus `budget_fraction`, `ln(1+match_class_node_count)`, `ln(1+|neighborhood_ops|)`, `ln(1+expr_node_count)` |
| dedup filtering | `dedup_repeat` excluded, counted, reported |
| class weighting | `pos_weight = negatives / positives` on the TRAIN split used |
| loss | weighted BCE-with-logits, the same stable form |
| cold start | no warm start from any prior checkpoint |
| TRAIN-only | DEV never touches a gradient and never selects a hyperparameter |

**Feature parity is a deliberate change to the tower.** `forward_candidate` previously read
only the pooled neighborhood and `budget_fraction`; `CANDIDATE_INPUT_DIM` is now `K + 4` so
it reads all four of the additive model's scalars. Without that, the bilinear arm would
have differed from the additive arm in the *feature set* as well, and a loss could be
blamed on two missing scalars instead of on the interaction. This is recorded here per §4's
requirement that any other difference be named.

Two things could not be held equal:

1. **Gradient clipping is on the step, not on `dLoss/dz`.** `train_guide` clips `dLoss/dz`
   to ±20, the right primitive for a one-layer convex model whose update is `lr · dz · x`.
   This head is four layers with a bilinear top, where the same `dz` produces wildly
   different weight-space steps depending on the activations. `--max-grad-norm 1.0` bounds
   the Euclidean norm of the whole accumulated gradient instead. The objective is
   byte-identical; only the optimiser is bounded. Without it the head diverges to NaN
   within a few hundred steps at any usable rate — and it *fails loudly*
   (`BilinearTrainer::apply` refuses a non-finite gradient) rather than writing a
   checkpoint of NaNs.
2. **The learning rate was selected on a TRAIN-internal holdout** (20% of TRAIN families,
   43 of them; family granularity, the same fence the corpus manifest uses). Registration
   §9: "Model selection happens on a TRAIN-internal holdout or it does not happen." The
   additive model's `lr = 0.01` was never tuned against this architecture.

   Selection runs the **full 30 epochs** per candidate rate, not a short proxy: with an
   inverse-time decayed rate, the ranking after 5 epochs and after 30 are different
   questions, and a selection run of a different length ranks a different model than the
   one that gets built. (On this grid the two lengths happen to agree on `lr = 0.1`; the
   default was changed anyway, because agreeing is not the same as being the right
   comparison.)

   | lr | holdout AUC | holdout PR-AUC |
   |---:|---:|---:|
   | 0.3 | 0.9540 | 0.7474 |
   | **0.1** | **0.9741** | **0.8793** |
   | 0.06 | 0.9722 | 0.8753 |
   | 0.03 | 0.9659 | 0.8335 |
   | 0.01 | 0.9516 | 0.7390 |
   | 0.003 | 0.9450 | 0.7021 |
   | 0.001 | 0.9432 | 0.6949 |

   The optimum is interior, which is the property that makes the grid honest — an earlier
   grid topping out at 0.03 selected its own boundary, which is a sign the grid is too
   narrow rather than an answer, so the grid was widened (a TRAIN-only decision; DEV was
   not consulted).

   **What the selection cost, reported because omitting it would be selecting on DEV.**
   The `lr = 0.03` fit reaches DEV PR-AUC **0.8111**, better than the selected
   `lr = 0.1`'s 0.7624. That number exists only because an earlier, narrower grid produced
   that model and it was scored on DEV. The holdout says 0.1 and the holdout is what is
   allowed to choose, so 0.1 is the checkpoint. Going back for 0.03 now would be picking a
   hyperparameter by held-out performance, which is exactly the thing §9 forbids and
   exactly how a comparison becomes worthless.

   `--epochs 30` was fixed a priori to match `train_guide`'s default and was **not** tuned.
   The DEV curve is not monotone and epoch 29 is not its best point (PR-AUC by epoch: 0.686,
   0.746, 0.812, 0.801, 0.728, 0.835, 0.762 at epochs 0/5/10/15/20/25/29), so the reported
   0.7624 is the last-epoch model, not a best-epoch pick — no early stopping, because early
   stopping on DEV is selection on DEV. A rate this large keeps the last epochs noisy;
   a longer decay or a smaller final rate is the obvious next thing to try on the
   TRAIN-internal holdout, and would only be expected to raise this number.

---

## 4. The mandatory skew test (registration §9)

The bar is: for ≥ 1000 DEV records, the trainer's forward and the deployed
`score_candidates` agree to ≤ 1e-6. Both halves pass **exactly**, at 0.0.

Where this head's skew can actually live is not where the additive head's could, and the
test is built for the real risk rather than copied for the appearance of one. `train_guide`
and `LinearCandidateGuide` are two independent transcriptions of one formula, and
`skew_test_linear_guide` checks the transcriptions agree. `BilinearTrainer` and
`BilinearCandidateGuide` deliberately share `SaturationHead::score_candidate` — a
hand-written second copy of a four-layer forward pass with a bilinear top would be a new bug
surface, not a control. What can differ is:

- **the checkpoint boundary** — ~16,000 floats flattened by a visitor and read back into a
  freshly zeroed head; a tensor missing from that order, a transposed row-major assumption,
  or a float that does not survive JSON all deploy silently as a different model;
- **the derived rule embeddings** — the checkpoint deliberately does *not* store them
  (a derived value in a checkpoint is a second copy that can disagree with the first), so
  both sides recompute `rule_proj(concat(templates))` from templates they rebuild
  themselves;
- **the op embeddings** — frozen, carried in the checkpoint, reloaded by `OpKind::all()`
  order, where an off-by-one shifts every op and still produces plausible scores.

Because both sides read the same file, `skew_test_bilinear_guide` structurally cannot see a
bug in the **writing** path. That direction is checked in the only place an in-memory
trained head exists: `train_guide_bilinear` compares its own post-training
`BilinearTrainer::score` against the guide it builds from the file it just wrote, over 5,000
DEV records, and hard-fails before writing a report. Neither substitutes for the other; both
are reported.

Supporting checks, all in `cargo test`: a checkpoint that round-trips through JSON
bit-exactly; a refusal on an edited weight (content hash), on a foreign schema identity, on
a foreign rule fingerprint, on a truncated parameter vector, and on a wrong-length
op-embedding block; and `every_trained_parameter_should_be_reachable_from_the_flat_vector`,
which perturbs the whole flat vector and requires every trained tensor to move — the
round-trip test alone cannot catch a tensor that both sides skip identically.

---

## 5. No production behaviour change (registration §10)

Production leaves `guide: None`. Proved, not asserted: the 206-kernel production extraction
digest (`production_extraction_digest`, #1121, over `/private/tmp/classcap_corpus`) is
**byte-identical** before and after every change in this branch —
`9abdc78d6e7b5518eaacb72f719f3630` both times, 206 rows.

---

## 6. Artifacts

| file | what |
|---|---|
| `docs/results/2026-09-02-strict-label-dataset-remint.json` | the re-minted dataset's own report |
| `docs/results/2026-09-02-train-guide-additive-remint-report.{json,md}` | the additive arm on that dataset |
| `docs/results/2026-09-02-train-guide-bilinear-report.{json,md}` | the bilinear arm, including the full per-rule table, the lr grid, and the training curve |
| `docs/results/2026-09-02-skew-test-bilinear-guide.json` | the committed skew artifact |
| `pixelflow-pipeline/data/guide_checkpoint_bilinear_v1.json` | the trained checkpoint (gitignored, like every other checkpoint) |

## 7. What this does not say

- It does not claim H_repr, H_capacity, H_form-null, or H_worse. No `D_A(S, B)` was
  computed, no guided saturation was run, no `dag_cost` ratio was measured.
- It does not compare against the registration's **frozen** `LinearCandidateGuide` arm.
  That checkpoint is off-vocabulary on this branch and untouched; the additive numbers here
  are a same-code, same-data retrain, which is what a functional-form comparison needs and
  is *not* what §4 names as the frozen arm.
- Better held-out classification of the strict bit is a necessary condition for a guided
  win, not evidence of one. Registration §8 already names the case where the label is the
  bottleneck, and a classifier that fits it better does not escape that.
