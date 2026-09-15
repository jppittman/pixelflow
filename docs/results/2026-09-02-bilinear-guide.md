> **Retracted/Superseded (2026-09-07), ledger L047.** The H_form-null verdict was taken with a tree-objective extractor on generator families; it is re-taken under #1192 only if the linear Guide is non-null at the real regime (re-validation item 5), and the finding that guidance of any kind is a net cost on trig-structured kernels at B <= 200 is the reason to expect a null. Verdict and rationale: `docs/results/2026-09-07-claims-ledger.md` (PR #1207); the corrected benchmark and re-validation order: `docs/plans/2026-09-07-benchmark-correction.md`.

# The bilinear Guide: does a rule-by-context interaction buy a domain-conditional advantage?

**Date:** 2026-09-02  
**Registration:** `docs/plans/2026-09-02-bilinear-guide-registration.md` (frozen; §11 deliberately left unedited — this document is the appendix it names)  
**Rows:** `docs/results/2026-09-02-bilinear-guide.jsonl`  
**Cost:** `ExtractedDAG::dag_cost` under `CostModel::latency_prior()`. No wall clock in any number.  
**Grid (every arm):** [25, 50, 100, 200, 400, 800] recorded rule applications.

**Additive arm:** `pixelflow-pipeline/data/guide_checkpoint_strict_remint.json`.  
**Registration §4's frozen additive checkpoint** (`pixelflow-pipeline/data/guide_checkpoint_strict_v1.json`): REFUSED, not deployed: guide checkpoint pixelflow-pipeline/data/guide_checkpoint_strict_v1.json: missing field `rule_fingerprint` at line 258 column 1

## Verdict

> B=100: H_form-null — D_linear − D_bilinear on `sh` is +0.0234, inside M_100=0.06; the rule-by-context interaction buys nothing under domain shift, so the functional form was not the bottleneck.
>
> B=200: H_form-null — D_linear − D_bilinear on `sh` is +0.0345, inside M_200=0.07; the rule-by-context interaction buys nothing under domain shift, so the functional form was not the bottleneck.
>

| B | D_linear − D_bilinear on `sh` | M_B | `bezier` gap | m_bilinear^DEV | m_linear^DEV | powered | verdict |
|---:|---:|---:|---:|---:|---:|:--|:--|
| 100 | **+0.0234** | 0.06 | +0.0193 | 0.6746 | 0.6530 | yes | **H_form-null** |
| 200 | **+0.0345** | 0.07 | +0.0252 | 0.8170 | 0.8003 | yes | **H_form-null** |

## Per set and tier: cost ratio vs unguided-at-B

`dag_cost_arm@B / dag_cost_unguided@B`, per expression.

| set | n | B | arm | min | q1 | median | q3 | p90 | improved | = | worse | regret% (median) | reaches best |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dev | 334 | 100 | control | 0.2623 | 0.5735 | **0.6680** | 0.7420 | 0.8005 | 327 | 5 | 2 | 1.26 | 71 |
| dev | 334 | 100 | linear | 0.2623 | 0.5669 | **0.6530** | 0.7270 | 0.7867 | 327 | 6 | 1 | 0.38 | 95 |
| dev | 334 | 100 | bilinear | 0.2623 | 0.5843 | **0.6746** | 0.7682 | 0.9755 | 326 | 5 | 3 | 1.05 | 66 |
| sh | 100 | 100 | control | 0.9486 | 1.0263 | **1.0978** | 1.1844 | 1.2568 | 17 | 0 | 83 | 12.33 | 6 |
| sh | 100 | 100 | linear | 0.9014 | 1.0310 | **1.1063** | 1.2070 | 1.3268 | 17 | 0 | 83 | 14.69 | 14 |
| sh | 100 | 100 | bilinear | 0.9547 | 1.0496 | **1.1044** | 1.2096 | 1.2673 | 5 | 0 | 95 | 13.37 | 2 |
| bezier | 80 | 100 | control | 0.7286 | 0.7978 | **0.7978** | 0.8569 | 0.9765 | 80 | 0 | 0 | 7.58 | 35 |
| bezier | 80 | 100 | linear | 0.7286 | 0.7978 | **0.7978** | 0.8569 | 0.9765 | 80 | 0 | 0 | 7.58 | 35 |
| bezier | 80 | 100 | bilinear | 0.7978 | 0.7978 | **0.8000** | 0.8569 | 0.9765 | 80 | 0 | 0 | 7.58 | 20 |
| dev | 334 | 200 | control | 0.4502 | 0.6946 | **0.8061** | 1.0000 | 1.0000 | 231 | 92 | 11 | 0.29 | 145 |
| dev | 334 | 200 | linear | 0.4463 | 0.6741 | **0.8003** | 1.0000 | 1.0000 | 241 | 80 | 13 | 0.00 | 181 |
| dev | 334 | 200 | bilinear | 0.4463 | 0.6742 | **0.8170** | 1.0000 | 1.0000 | 240 | 77 | 17 | 0.18 | 141 |
| sh | 100 | 200 | control | 0.9498 | 1.0194 | **1.0921** | 1.1844 | 1.2573 | 17 | 0 | 83 | 11.94 | 5 |
| sh | 100 | 200 | linear | 1.0062 | 1.0984 | **1.1234** | 1.2349 | 1.3268 | 0 | 0 | 100 | 18.10 | 0 |
| sh | 100 | 200 | bilinear | 0.9559 | 1.0375 | **1.1055** | 1.2096 | 1.2673 | 11 | 1 | 88 | 13.18 | 1 |
| bezier | 80 | 200 | control | 0.7846 | 0.7857 | **0.7857** | 0.9026 | 1.0000 | 60 | 20 | 0 | 0.00 | 61 |
| bezier | 80 | 200 | linear | 0.7857 | 0.7857 | **0.8701** | 0.9423 | 1.0000 | 60 | 20 | 0 | 0.00 | 46 |
| bezier | 80 | 200 | bilinear | 0.8452 | 0.8452 | **0.8615** | 0.9026 | 1.0000 | 60 | 20 | 0 | 7.58 | 20 |

## The registered statistic `D_A(S, B) = m_A^S(B) − m_A^DEV(B)`

`m_A^DEV` is measured in THIS run on `dag_cost` for every arm (registration's instrument note); no round-1 constant is reused.

| set | B | m_control | m_linear | m_bilinear | D_control | D_linear | D_bilinear | D_linear − D_bilinear | M_B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dev | 100 | 0.6680 | 0.6530 | 0.6746 | +0.0000 | +0.0000 | +0.0000 | **+0.0000** | 0.06 |
| sh | 100 | 1.0978 | 1.1063 | 1.1044 | +0.4297 | +0.4532 | +0.4298 | **+0.0234** | 0.06 |
| bezier | 100 | 0.7978 | 0.7978 | 0.8000 | +0.1297 | +0.1447 | +0.1254 | **+0.0193** | 0.06 |
| dev | 200 | 0.8061 | 0.8003 | 0.8170 | +0.0000 | +0.0000 | +0.0000 | **+0.0000** | 0.07 |
| sh | 200 | 1.0921 | 1.1234 | 1.1055 | +0.2859 | +0.3230 | +0.2885 | **+0.0345** | 0.07 |
| bezier | 200 | 0.7857 | 0.8701 | 0.8615 | -0.0204 | +0.0698 | +0.0446 | **+0.0252** | 0.07 |

### Head to head: bilinear vs linear, per expression at B

| set | B | bilinear < linear | = | bilinear > linear |
|---|---:|---:|---:|---:|
| dev | 100 | 26 | 78 | 230 |
| sh | 100 | 67 | 1 | 32 |
| bezier | 100 | 0 | 65 | 15 |
| dev | 200 | 35 | 152 | 147 |
| sh | 200 | 79 | 3 | 18 |
| bezier | 200 | 15 | 39 | 26 |

### §3.2 power check: within-DEV family swing on this instrument

Max over the 8 DEV families of `|D_linear^f − D_bilinear^f|`. Exceeding the inherited `M_B` means the round is underpowered at that tier and no verdict is claimed; the inherited margin is never raised to match.

| B | max family swing | M_B | powered |
|---:|---:|---:|:--|
| 100 | 0.0274 | 0.06 | yes |
| 200 | 0.0215 | 0.07 | yes |

## Diagnostic 1 — do the arms fire the trig identities on `sh`?

Firings within the first B applications, pooled over the set. `exprs` = expressions where the rule fired at all; `fired` = total applications; `sp` = strict-positive.


**B = 100**

| set | arm | sin-angle-addition | cos-angle-addition | half-angle-product | pythagorean | 
|---|---|---:|---:|---:|---:|
| dev | unguided | 0 exprs / 0 fired / 0 sp | 2 exprs / 2 fired / 0 sp | 0 exprs / 0 fired / 0 sp | 0 exprs / 0 fired / 0 sp | 
| dev | control | 26 exprs / 38 fired / 1 sp | 29 exprs / 54 fired / 4 sp | 0 exprs / 0 fired / 0 sp | 0 exprs / 0 fired / 0 sp | 
| dev | linear | 11 exprs / 16 fired / 0 sp | 23 exprs / 30 fired / 2 sp | 0 exprs / 0 fired / 0 sp | 0 exprs / 0 fired / 0 sp | 
| dev | bilinear | 20 exprs / 35 fired / 1 sp | 33 exprs / 66 fired / 2 sp | 0 exprs / 0 fired / 0 sp | 0 exprs / 0 fired / 0 sp | 
| sh | unguided | 0 exprs / 0 fired / 0 sp | 0 exprs / 0 fired / 0 sp | 0 exprs / 0 fired / 0 sp | 0 exprs / 0 fired / 0 sp | 
| sh | control | 97 exprs / 281 fired / 0 sp | 53 exprs / 106 fired / 0 sp | 97 exprs / 506 fired / 146 sp | 0 exprs / 0 fired / 0 sp | 
| sh | linear | 14 exprs / 18 fired / 0 sp | 4 exprs / 7 fired / 0 sp | 80 exprs / 173 fired / 115 sp | 0 exprs / 0 fired / 0 sp | 
| sh | bilinear | 97 exprs / 333 fired / 0 sp | 36 exprs / 72 fired / 0 sp | 97 exprs / 569 fired / 149 sp | 0 exprs / 0 fired / 0 sp | 
| bezier | unguided | 0 exprs / 0 fired / 0 sp | 0 exprs / 0 fired / 0 sp | 0 exprs / 0 fired / 0 sp | 0 exprs / 0 fired / 0 sp | 
| bezier | control | 0 exprs / 0 fired / 0 sp | 0 exprs / 0 fired / 0 sp | 0 exprs / 0 fired / 0 sp | 0 exprs / 0 fired / 0 sp | 
| bezier | linear | 0 exprs / 0 fired / 0 sp | 0 exprs / 0 fired / 0 sp | 0 exprs / 0 fired / 0 sp | 0 exprs / 0 fired / 0 sp | 
| bezier | bilinear | 0 exprs / 0 fired / 0 sp | 0 exprs / 0 fired / 0 sp | 0 exprs / 0 fired / 0 sp | 0 exprs / 0 fired / 0 sp | 

**B = 200**

| set | arm | sin-angle-addition | cos-angle-addition | half-angle-product | pythagorean | 
|---|---|---:|---:|---:|---:|
| dev | unguided | 10 exprs / 11 fired / 0 sp | 14 exprs / 18 fired / 0 sp | 0 exprs / 0 fired / 0 sp | 0 exprs / 0 fired / 0 sp | 
| dev | control | 40 exprs / 101 fired / 1 sp | 41 exprs / 115 fired / 5 sp | 0 exprs / 0 fired / 0 sp | 0 exprs / 0 fired / 0 sp | 
| dev | linear | 26 exprs / 48 fired / 0 sp | 46 exprs / 102 fired / 2 sp | 0 exprs / 0 fired / 0 sp | 0 exprs / 0 fired / 0 sp | 
| dev | bilinear | 31 exprs / 71 fired / 1 sp | 48 exprs / 121 fired / 4 sp | 0 exprs / 0 fired / 0 sp | 0 exprs / 0 fired / 0 sp | 
| sh | unguided | 0 exprs / 0 fired / 0 sp | 0 exprs / 0 fired / 0 sp | 0 exprs / 0 fired / 0 sp | 0 exprs / 0 fired / 0 sp | 
| sh | control | 97 exprs / 471 fired / 0 sp | 53 exprs / 169 fired / 0 sp | 97 exprs / 897 fired / 146 sp | 0 exprs / 0 fired / 0 sp | 
| sh | linear | 62 exprs / 109 fired / 0 sp | 4 exprs / 11 fired / 0 sp | 97 exprs / 275 fired / 149 sp | 0 exprs / 0 fired / 0 sp | 
| sh | bilinear | 97 exprs / 342 fired / 0 sp | 53 exprs / 92 fired / 0 sp | 97 exprs / 777 fired / 149 sp | 0 exprs / 0 fired / 0 sp | 
| bezier | unguided | 0 exprs / 0 fired / 0 sp | 0 exprs / 0 fired / 0 sp | 0 exprs / 0 fired / 0 sp | 0 exprs / 0 fired / 0 sp | 
| bezier | control | 0 exprs / 0 fired / 0 sp | 0 exprs / 0 fired / 0 sp | 0 exprs / 0 fired / 0 sp | 0 exprs / 0 fired / 0 sp | 
| bezier | linear | 0 exprs / 0 fired / 0 sp | 0 exprs / 0 fired / 0 sp | 0 exprs / 0 fired / 0 sp | 0 exprs / 0 fired / 0 sp | 
| bezier | bilinear | 0 exprs / 0 fired / 0 sp | 0 exprs / 0 fired / 0 sp | 0 exprs / 0 fired / 0 sp | 0 exprs / 0 fired / 0 sp | 

## Diagnostic 2 — does the bilinear model's induced rule ORDER differ by context?

Each expression's first really-scored candidate is held fixed as the context `x`; every rule's identity and embedding is substituted in turn, and the model's ranking of the whole vocabulary at that context is read off. Registration §1: for the additive class this order is the same list at every context, by construction; for the bilinear class it is free to vary. `rank` is 1-based over the whole rule vocabulary (1 = scored highest).

| set | model | contexts | distinct orders | mean score spread | sin-angle-addition mean rank (best) | cos-angle-addition mean rank (best) | half-angle-product mean rank (best) | pythagorean mean rank (best) | 
|---|---|---:|---:|---:|---:|---:|---:|---:|
| dev | bilinear | 334 | 173 | 42.2383 | 11.7 (4) | 7.5 (2) | 13.5 (7) | 2.6 (2) | 
| dev | linear | 334 | 1 | 9.3830 | 40.0 (40) | 18.0 (18) | 26.0 (26) | 38.0 (38) | 
| sh | bilinear | 100 | 48 | 53.6867 | 6.5 (4) | 2.9 (2) | 9.5 (6) | 4.5 (3) | 
| sh | linear | 100 | 1 | 9.3830 | 40.0 (40) | 18.0 (18) | 26.0 (26) | 38.0 (38) | 
| bezier | bilinear | 80 | 3 | 76.8979 | 4.0 (4) | 2.0 (2) | 7.0 (7) | 8.0 (8) | 
| bezier | linear | 80 | 1 | 9.3830 | 40.0 (40) | 18.0 (18) | 26.0 (26) | 38.0 (38) | 

## Context

```json
{
  "arms": [
    "unguided",
    "control",
    "linear",
    "bilinear"
  ],
  "bilinear_checkpoint": "pixelflow-pipeline/data/guide_checkpoint_bilinear_v1.json",
  "budget": "recorded rule applications; no wall clock in any metric",
  "cost": "ExtractedDAG::dag_cost (ChoiceCost::dag, #1117) under CostModel::latency_prior()",
  "deviation_from_registration_4": "The registration's frozen `LinearCandidateGuide` arm (guide_checkpoint_strict_v1.json) is off-vocabulary on this branch and is refused by LinearWeights::check_vocabulary; the additive arm that ran is the same-recipe model retrained on the same re-minted dataset as the bilinear arm. Recorded, not worked around.",
  "frozen_linear_arm_status": "REFUSED, not deployed: guide checkpoint pixelflow-pipeline/data/guide_checkpoint_strict_v1.json: missing field `rule_fingerprint` at line 258 column 1",
  "frozen_linear_checkpoint": "pixelflow-pipeline/data/guide_checkpoint_strict_v1.json",
  "grid": [
    25,
    50,
    100,
    200,
    400,
    800
  ],
  "inherited": [
    "docs/plans/2026-09-01-phase3-registration.md",
    "docs/plans/2026-09-01-phase3-round1b-domain-shift-registration.md"
  ],
  "linear_checkpoint": "pixelflow-pipeline/data/guide_checkpoint_strict_remint.json",
  "registration": "docs/plans/2026-09-02-bilinear-guide-registration.md",
  "source_rev": "b0bcc898932850746c21b7d89c33f0034a3f4d2e",
  "train_guide_report": "docs/results/2026-09-01-train-guide-report.json",
  "uptime_at_end": "19:02  up 52 days,  9:35, 2 users, load averages: 4.83 6.49 5.82",
  "uptime_at_start": "19:02  up 52 days,  9:35, 2 users, load averages: 4.83 6.49 5.82"
}
```

---

# Reading

*Everything above this line is written by `phase3_bilinear_eval` from the row
file. Everything below is the interpretation, and it is separated so a re-run
cannot quietly rewrite a conclusion.*

## 1. The bilinear arm ties the additive one. H_form-null, both tiers.

**`D_linear − D_bilinear` on `sh` is +0.0234 at B=100 against M_100 = 0.06, and
+0.0345 at B=200 against M_200 = 0.07 — inside the frozen margin at both tiers, on
a round the §3.2 power check passes (family swing 0.0274 / 0.0215).** Registration
§3.3's conjunct 1 fails; H_repr is rejected, H_capacity and H_worse are rejected,
and the registered reading of §8 applies: **the functional form was not the
bottleneck.**

The absolute numbers are blunter than the statistic. On `sh` at B=100 the two model
classes land 0.002 apart — `m_linear^sh` 1.1063, `m_bilinear^sh` 1.1044. Decomposing
the registered gap shows where it comes from:

```
B=100:  D_linear − D_bilinear = (1.1063 − 1.1044) + (0.6746 − 0.6530) = 0.0019 + 0.0216
B=200:  D_linear − D_bilinear = (1.1234 − 1.1055) + (0.8170 − 0.8003) = 0.0179 + 0.0167
```

At B=100 **92% of the bilinear's registered advantage under shift is its worse DEV
baseline, not its better `sh` behaviour.** `D` is defined that way on purpose — it
is a shift in an arm's own advantage — but a reader who saw only `+0.0234` would
take it for a `sh` win, and it is mostly a DEV loss (median ratio 0.6746 against the
additive's 0.6530; head-to-head the bilinear is worse on 230 of 334 DEV
expressions). Even the sign is not evidence for the claim.

## 2. What this rules out, and what makes it a sharper null than round 1b's

Round 1b's H_null admitted two explanations (registration §2): either rule value is
genuinely context-independent, or the measured model class had no capacity to use
context. This round removes the second, and diagnostic 2 removes it *empirically*
rather than by construction:

| model | contexts scored | distinct induced rule orders |
|---|---:|---:|
| additive (`LinearCandidateGuide`) | 514 | **1** |
| bilinear (`BilinearCandidateGuide`) | 514 | **222** (173 DEV / 48 `sh` / 3 `bezier`, 2 shared) |

The additive model produces exactly one ranking of the 62-rule vocabulary across
every context in DEV, `sh` and `bezier` — the theorem in registration §1, observed.
The deployed bilinear head produces 222 distinct ones. It is not a collapsed head
scoring an additive function (the failure mode `backward.rs`'s three defences exist
to catch); the interaction is live at inference on real candidates, and it still
buys nothing past the margin.

And the orders it produces are **domain-separated**, which is the specific thing
the question asked for. Recomputed from the row file
(`docs/results/2026-09-02-bilinear-guide.jsonl`, whole-vocabulary Kendall tau
between induced orders, 60 sampled pairs per cell):

| | mean tau |
|---|---:|
| bilinear, two `sh` contexts | 0.926 |
| bilinear, two DEV contexts | 0.868 |
| bilinear, an `sh` context vs a DEV context | **0.685** |
| additive, any two contexts anywhere | **1.000** |

47 of the 48 distinct `sh` orders never occur anywhere in DEV, and none of the
three `bezier` orders do. A worked pair — `dev_sh_00003` against
`dev_b09_f01_00249`, tau 0.442 under the bilinear model and 1.000 under the
additive one:

```
sh  : even-negation(Abs)  cos-angle-addition  even-negation(Cos)  sin-angle-addition  power-combine  half-angle-product ...
DEV : fma-fusion          pythagorean         power-recip         recip-sqrt          even-negation(Abs)  power-rsqrt   ...
```

So the answer to "does the induced rule order actually differ between an `sh`
expression and a DEV expression" is **yes, substantially, and along the domain
axis** — and the cost at budget still does not move. That is the result. The
mechanism the hypothesis called for is present and running; the outcome it
predicted is not.

So the model-class explanation is off the table. What remains, exactly as §8
pre-committed:

1. **The features.** `neighborhood_ops` is a one-hop bag of the matched class's
   child `OpKind`s. Diagnostic 2 shows what the head learned to do with it, and it
   is not the hypothesis: under the bilinear model `pythagorean` has mean rank
   **2.6 of 62 on DEV** and **4.5 on `sh`** — it ranks the identity *higher on the
   distribution where it can never match* than on the one built to bait it. The
   conditioning is real and is on something other than domain.
2. **The label.** The strict load-bearing bit is minted credit on one trajectory.
   `pythagorean` fires **zero times, in every arm, on every set, at every budget**
   (514 expressions) — the same blind spot referred out in
   `docs/results/2026-09-01-strict-label-constant-output-blindspot.md`. A target
   that is identically zero for the rule the question is about cannot teach any
   functional form to prefer it anywhere.

This does **not** license "context does not matter". Two model classes over one
feature set and one label is a test of whether *these* features and *this* label
carry it.

## 3. Two preconditions the report has to state before the verdict is read

**(a) On `sh`, every guided arm is now worse than unguided.** Median ratio 1.09–1.11
at both tiers; 83–95 of 100 expressions worse; the control, the additive and the
bilinear all lose. Round 1b measured ~0.90 here. That is the instrument, not a
contradiction — round 1b ran on tree cost, before #1117 (`dag_cost`), #1118
(mid-scan application budget) and #1120 (`rebuild_budgeted`'s orphaned e-nodes), and
`docs/results/2026-09-02-phase3-instrument-changes.md` is why no round-1 constant is
reused as a reference here. But it means the primary set's `D` is a difference
between two arms that are **both** losing to doing nothing, and the honest summary
of `sh` at these budgets is that guidance of any kind is a net cost on it.

**(b) The registration's frozen additive arm could not be deployed.** §4 names
`guide_checkpoint_strict_v1.json` "exactly as round 1 trained it". That file names 61
rules; `RuleSet::production()` on this branch has 62, and `LinearWeights::check_vocabulary`
refuses it — correctly, and the refusal is recorded verbatim in the report's
`frozen_linear_arm_status`. The additive arm that ran is the same-recipe model
retrained by this branch's own binary on the same re-minted dataset as the bilinear
arm (`guide_checkpoint_strict_remint.json`,
`docs/results/2026-09-02-train-guide-additive-remint-report.md`). That is the
*stricter* reading of §4's binding constraint — "the only licensed difference
between the two arms is the functional form" — since a round-1 checkpoint would have
differed in label mint as well. It is a deviation from the letter of §4 and is
reported as one. The control arm's per-rule rates are still the registered frozen
`2026-09-01-train-guide-report.json`.

## 4. What the bilinear arm did do, since it is not nothing

- It fires `half-angle-product` — round 1b's best-paying trig rule — on **97 of 100**
  `sh` expressions within the first 100 applications, 569 firings / 149 strict
  positive, against the additive arm's 80 expressions / 173 firings / 115 positive.
- Head to head per expression on `sh` it beats the additive arm **67–1–32** at B=100
  and **79–3–18** at B=200.
- And the median ratio moves 0.002. The rule ordering changed a great deal; the cost
  at budget did not. That gap — many more of the "right" firings, no cost movement —
  is the same shape as the re-run oracle's finding that rule-granularity hindsight
  filtering is indistinguishable from unguided through B=400
  (`docs/results/2026-09-02-oracle-filtered-budget-curves.md`). It is further
  evidence that whatever is left at these budgets is not reachable by reordering
  rules.

## 5. Gates

| gate | status |
|---|---|
| skew test, bit-exact (§9) | PASS — `docs/results/2026-09-02-skew-test-bilinear-guide.json`, max abs diff 0.0 over 5,000 DEV records, both directions |
| cost read from `dag_cost` (§7) | yes — `ChoiceCost::dag` only; `total_cost` is never read by this harness |
| no timing in any metric (§7) | yes — wall clock appears only as a panicking safety ceiling |
| TRAIN fence (§5) | PASS — 334 DEV + 180 OOD entries probed against 3,359 TRAIN structures, 0 collisions |
| FINAL untouched (§5) | yes — `corpus_final.bin` is never opened |
| n ≥ 30 per set (§7) | DEV 334, `sh` 100, `bezier` 80 |
| §3.2 power check | PASS at both tiers (0.0274 ≤ 0.06; 0.0215 ≤ 0.07) |
| production digest (§10) | byte-identical, 206 rows — see the run record in `journal.jsonl` |
| kill gate (§7) | not fired: the bilinear arm beats unguided-at-B on DEV at both tiers (0.6746 / 0.8170) |

Registration §11 is deliberately left unedited — a frozen registration is not
amended after its run. This document is the appendix it points to.
