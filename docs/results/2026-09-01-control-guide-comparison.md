> **Retracted/Superseded (2026-09-07), ledger L036.** This doc's own banner says the 2026-09-02 review fixes change every weight and metric below, and it was never re-run; the AUC / PR-AUC comparison is withdrawn. Verdict and rationale: `docs/results/2026-09-07-claims-ledger.md` (PR #1207); the corrected benchmark and re-validation order: `docs/plans/2026-09-07-benchmark-correction.md`.

# Control-arm comparison: linear Guide vs per-rule rate lookup

> **Predates the 2026-09-02 review fixes; a re-run is required.** Both guides are now
> scored on first-seen candidates only, the DEV split is bound to the training report by
> content hash, `average_precision` groups tied scores (the per-rule control is nothing
> but large tie groups, so its PR-AUC moves most), and a negative AUC gap is reported as
> a control win instead of as a real gap. Nothing here has been edited to match.

> **Instrument change (2026-09-02 forward-port).** Two things about how a
> Phase 3 anytime curve is measured changed with the port, so a re-run does
> not reproduce the numbers below even if nothing else changed: the
> application budget now binds **mid-scan** rather than between rule sweeps
> (`app_actual == app_target` exactly, no overshoot), and the reported cost
> is the **DAG** cost the emitted kernel pays rather than the extraction DP's
> tree total (#1117). Full statement:
> [docs/results/2026-09-02-phase3-instrument-changes.md](2026-09-02-phase3-instrument-changes.md).

DEV split: `pixelflow-pipeline/data/strict_labels_dev.jsonl` (155231 samples).

| guide | DEV AUC-ROC | DEV PR-AUC |
|---|---:|---:|
| linear model (candidate-local features) | 0.9902 | 0.4595 |
| PerRuleRateGuide (rule_idx only, control) | 0.9368 | 0.1083 |

**Gap**: AUC +0.0535, PR-AUC +0.3511.

REAL GAP (AUC gap +0.0535 >= 0.02): the linear model meaningfully outranks a per-rule lookup table — candidate-local features are carrying signal beyond each rule's base rate. PR-AUC gap is +0.3511.
