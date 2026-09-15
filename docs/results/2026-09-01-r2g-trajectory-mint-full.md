> **Retracted/Superseded (2026-09-07), ledger L044.** The return labels minted here are differences of tree costs (pre-#1192); the mint is kept as an instrument, its gate is not a result. Verdict and rationale: `docs/results/2026-09-07-claims-ledger.md` (PR #1207); the corrected benchmark and re-validation order: `docs/plans/2026-09-07-benchmark-correction.md`.

# R2G trajectory mint

> **Instrument change (2026-09-02 forward-port).** Two things about how a
> Phase 3 anytime curve is measured changed with the port, so a re-run does
> not reproduce the numbers below even if nothing else changed: the
> application budget now binds **mid-scan** rather than between rule sweeps
> (`app_actual == app_target` exactly, no overshoot), and the reported cost
> is the **DAG** cost the emitted kernel pays rather than the extraction DP's
> tree total (#1117). Full statement:
> [docs/results/2026-09-02-phase3-instrument-changes.md](2026-09-02-phase3-instrument-changes.md).

Registered budget B=100 (primary), budget ladder [100, 200, 400, 800, 1600, 3200]. `--max-expr-nodes 0` (`0` = no filter) / `--max-classes 5000` (resolved from `config_for_node_count` when `--max-classes` is unset). `--train-limit 0` `--dev-limit 0` `--n-rand 6` `--mix "1/4,1/4,1/2"`. Mint wall-clock: 1306.1s.

## train

- expressions: 3356 (3 zero-best excluded, 0 oversized-skipped, 0 wallclock-skipped)
- trajectories: 40272
- applications (JSONL rows): 18565784
- return spread at B=100: Q1 0.0000  median 0.0000  Q3 0.0430
- zero-spread expressions: 1875 / 3356 (55.9%), 55.9% of records
- dataset gate (>50% zero-spread): FIRED

| policy | median return_b100 | n |
|---|---:|---:|
| mix:strict-v1:1/2:1 | 0.0000 | 3356 |
| mix:strict-v1:1/4:1 | 0.0000 | 3356 |
| mix:strict-v1:1/4:2 | 0.0000 | 3356 |
| per-rule | 0.0000 | 3356 |
| random:1 | 0.0000 | 3356 |
| random:2 | 0.0000 | 3356 |
| random:3 | 0.0000 | 3356 |
| random:4 | 0.0000 | 3356 |
| random:5 | 0.0000 | 3356 |
| random:6 | 0.0000 | 3356 |
| strict-v1 | 0.0000 | 3356 |
| unguided | 0.0000 | 3356 |

## dev

- expressions: 783 (1 zero-best excluded, 0 oversized-skipped, 0 wallclock-skipped)
- trajectories: 9396
- applications (JSONL rows): 3544477
- return spread at B=100: Q1 0.0000  median 0.0000  Q3 0.0115
- zero-spread expressions: 470 / 783 (60.0%), 60.0% of records
- dataset gate (>50% zero-spread): FIRED

| policy | median return_b100 | n |
|---|---:|---:|
| mix:strict-v1:1/2:1 | 0.0000 | 783 |
| mix:strict-v1:1/4:1 | 0.0000 | 783 |
| mix:strict-v1:1/4:2 | 0.0000 | 783 |
| per-rule | 0.0000 | 783 |
| random:1 | 0.0000 | 783 |
| random:2 | 0.0000 | 783 |
| random:3 | 0.0000 | 783 |
| random:4 | 0.0000 | 783 |
| random:5 | 0.0000 | 783 |
| random:6 | 0.0000 | 783 |
| strict-v1 | 0.0000 | 783 |
| unguided | 0.0000 | 783 |

## sh

- expressions: 100 (0 zero-best excluded, 0 oversized-skipped, 0 wallclock-skipped)
- trajectories: 1200
- applications (JSONL rows): 3652201
- return spread at B=100: Q1 0.0127  median 0.0731  Q3 0.1096
- zero-spread expressions: 0 / 100 (0.0%), 0.0% of records
- dataset gate (>50% zero-spread): not fired

| policy | median return_b100 | n |
|---|---:|---:|
| strict-v1 | 0.0401 | 100 |
| mix:strict-v1:1/4:2 | 0.0402 | 100 |
| per-rule | 0.0441 | 100 |
| mix:strict-v1:1/2:1 | 0.0452 | 100 |
| mix:strict-v1:1/4:1 | 0.0452 | 100 |
| random:6 | 0.0487 | 100 |
| random:1 | 0.0507 | 100 |
| random:4 | 0.0509 | 100 |
| random:5 | 0.0531 | 100 |
| random:2 | 0.0541 | 100 |
| random:3 | 0.0544 | 100 |
| unguided | 0.1170 | 100 |

## bezier

- expressions: 80 (0 zero-best excluded, 0 oversized-skipped, 0 wallclock-skipped)
- trajectories: 960
- applications (JSONL rows): 2504034
- return spread at B=100: Q1 0.0751  median 0.1001  Q3 0.1216
- zero-spread expressions: 20 / 80 (25.0%), 25.0% of records
- dataset gate (>50% zero-spread): not fired

| policy | median return_b100 | n |
|---|---:|---:|
| mix:strict-v1:1/4:2 | 0.4993 | 80 |
| random:2 | 0.4993 | 80 |
| strict-v1 | 0.4993 | 80 |
| mix:strict-v1:1/2:1 | 0.5573 | 80 |
| mix:strict-v1:1/4:1 | 0.5573 | 80 |
| per-rule | 0.5573 | 80 |
| random:1 | 0.5573 | 80 |
| random:3 | 0.5573 | 80 |
| random:4 | 0.5573 | 80 |
| random:5 | 0.5573 | 80 |
| random:6 | 0.5573 | 80 |
| unguided | 0.6209 | 80 |
