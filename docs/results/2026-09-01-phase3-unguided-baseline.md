> **Retracted/Superseded (2026-09-07), ledger L033.** Every loss percentage below is a tree cost (`ExtractedDAG::total_cost` before #1192) where the kernel pays DAG cost, on a generated corpus whose median kernel has no select, no sharing and 32 nodes; the budget tiers and Y it fixed are withdrawn with it. Verdict and rationale: `docs/results/2026-09-07-claims-ledger.md` (PR #1207); the corrected benchmark and re-validation order: `docs/plans/2026-09-07-benchmark-correction.md`.

# Phase 3 unguided anytime baseline: truncation loss lives in the classical band (2026-09-01)

> **Instrument change (2026-09-02 forward-port).** Two things about how a
> Phase 3 anytime curve is measured changed with the port, so a re-run does
> not reproduce the numbers below even if nothing else changed: the
> application budget now binds **mid-scan** rather than between rule sweeps
> (`app_actual == app_target` exactly, no overshoot), and the reported cost
> is the **DAG** cost the emitted kernel pays rather than the extraction DP's
> tree total (#1117). Full statement:
> [docs/results/2026-09-02-phase3-instrument-changes.md](2026-09-02-phase3-instrument-changes.md).

Reproduce:
```
cargo run --release -p pixelflow-pipeline --features training --bin gen_bench_corpus -- --target 4000
cargo run --release -p pixelflow-pipeline --features training --bin phase3_unguided_baseline -- \
    --samples 400 \
    --out-csv docs/results/2026-09-01-phase3-unguided-baseline.csv \
    --out-json docs/results/2026-09-01-phase3-unguided-baseline.json
```

This is the measurement the Phase 3 registration
(`docs/plans/2026-09-01-phase3-registration.md`) fixes its budget tiers B and improvement
threshold Y from — unguided data only, produced and committed before any Guide training
exists. Full method (work axis, grid, class-cap semantics, fail-loud wall-clock ceiling) is in
the registration's §1 and in `pixelflow-search/src/egraph/anytime.rs`'s module doc; this
report records the run and the headline numbers.

- Corpus: `gen_bench_corpus --target 4000 --seed 42` → TRAIN 3,359 + DEV 784 + FINAL 129
  (FINAL never opened). Regeneration reproduced the prior session's corpus byte-for-byte
  (identical MD5s), so every number here is replayable from the seed.
- Sample: 400 of 4,143 TRAIN+DEV expressions, size-stratified (stride 10.36) —
  blitz n=23, rapid n=189, classical n=188.
- 399/400 runs quiesced on their own; 1 classical run exhausted the 204,800-application grid;
  0 hit the class cap or sweep ceiling. Applications to end of run: median 259, p95 30,866,
  max 206,648 (classical median 2,686 — the heavy tail is entirely classical).

## Truncation loss (the number the registration is built on)

loss% = (cost@B − cost@4B)/cost@4B, per expression, static latency-prior cost units:

| Band | Result |
|---|---|
| **classical** (n=188) | **B=100: median 48.5%, 175/188 positive, p90 5.9e4%. B=200: median 21.9%, 131/188 positive.** Median 0 from B=400 up (87/188 still positive at 400). |
| rapid (n=189) | Median 0.000 at every grid B; largest positive share 34/189 (18%) at B=50, p90 2.6%. |
| blitz (n=23) | Zero loss everywhere; median quiescence at 13 applications, below the first grid point (25). |

Full per-(scope, B) tables — p90/max/mean, live-at-B counts, B/2 columns — are in the JSON;
per-expression curves in the CSV.

## Reading

- Truncation demonstrably loses quality **only above ~50 nodes**, and there it loses a lot:
  at B=100 the median classical expression is half again worse than its own 4B state. This is
  the falsifiable headroom the Phase 3 Guide experiment targets (registered B=100 primary,
  B=200 secondary, classical band only).
- On blitz/rapid the anticipated shallow-kernel result holds: production-scale budgets
  already reach 4B quality and a Guide has nothing to buy back — the per-band
  stop-the-presses condition fires for those bands only, and no claim is registered on them.
- The one grid-exhausted expression is a reminder the classical tail is heavy
  (hundreds of thousands of applications); the class cap never bound on this sample, so the
  loss numbers are pure budget effects, not memory-cap artifacts.
