> **Retracted/Superseded (2026-09-07), ledger L043.** The 86x order headline (96.58 / 1.12) is a synthetic tree-cost number: on real kernels it is ~3% (numeric-first 0.9702 median) and it reverses on the psychedelic shader; re-taken in DAG units as re-validation item 2. Verdict and rationale: `docs/results/2026-09-07-claims-ledger.md` (PR #1207); the corrected benchmark and re-validation order: `docs/plans/2026-09-07-benchmark-correction.md`.

> **Instrument change (2026-09-02 forward-port).** Two things about how a
> Phase 3 anytime curve is measured changed with the port, so a re-run does
> not reproduce the numbers below even if nothing else changed: the
> application budget now binds **mid-scan** rather than between rule sweeps
> (`app_actual == app_target` exactly, no overshoot), and the reported cost
> is the **DAG** cost the emitted kernel pays rather than the extraction DP's
> tree total (#1117). Full statement:
> [docs/results/2026-09-02-phase3-instrument-changes.md](2026-09-02-phase3-instrument-changes.md).

## SS3 - the |R| effect, order held fixed

`Delta_U(p) = U(p) - U(OrderMatchedBase(seed, |p|))`, seed = `0x20260901` (DEFAULT_INTERLEAVE_SEED). Classical band (n=188).

**Mode (i)**

| rule set | \|R\| | matched-base ref | U(p)@100 | U(matched)@100 | Delta_U@100 | differing@100 | U(p)@200 | U(matched)@200 | Delta_U@200 | differing@200 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `dup:93` | 93 | `base:matched:0x20260901:93` | 43.76 | 60.55 | -16.79 | 125 | 6.69 | 24.82 | -18.13 | 148 |
| `dup:124` | 124 | `base:matched:0x20260901:124` | 41.12 | 43.88 | -2.76 | 161 | 25.44 | 16.89 | 8.55 | 150 |
| `dup:186` | 186 | `base:matched:0x20260901:186` | 15.44 | 13.85 | 1.59 | 158 | 9.47 | 4.70 | 4.76 | 171 |
| `dup:248` | 248 | `base:matched:0x20260901:248` | 38.67 | 4.33 | 34.34 | 181 | 27.02 | 3.36 | 23.66 | 162 |

Spearman rho(Delta_U, |R|): B=100 = 1.000, B=200 = 0.800. Delta_U at max |R|: 34.34% (B=100), 23.66% (B=200).

Delta1(v3) at `dup:93` (95% bootstrap CI half-width of paired median Delta_U): B=100 = 0.05 pts (median 0.00, CI [-0.09, 0.00]), B=200 = 4.82 pts (median -4.21).

**H1(v3) verdict (i):** direction HOLDS (rho >= 0.9), effect @100 HOLDS (|Delta_U(max)| >= Delta1), effect @200 HOLDS.

**Mode (ii)**

| rule set | \|R\| | matched-base ref | U(p)@100 | U(matched)@100 | Delta_U@100 | differing@100 | U(p)@200 | U(matched)@200 | Delta_U@200 | differing@200 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `comp:93` | 93 | `base:matched:0x20260901:93` | 60.55 | 60.55 | 0.00 | 5 | 25.60 | 24.82 | 0.78 | 5 |
| `comp:124` | 124 | `base:matched:0x20260901:124` | 41.69 | 43.88 | -2.19 | 17 | 21.22 | 16.89 | 4.33 | 33 |

Spearman rho(Delta_U, |R|): B=100 = -1.000, B=200 = 1.000. Delta_U at max |R|: -2.19% (B=100), 4.33% (B=200).

Delta1(v3) at `comp:93` (95% bootstrap CI half-width of paired median Delta_U): B=100 = 0.00 pts (median 0.00, CI [0.00, 0.00]), B=200 = 0.00 pts (median 0.00).

**H1(v3) verdict (ii):** direction FAILS (rho >= 0.9), effect @100 FAILS (|Delta_U(max)| >= Delta1), effect @200 HOLDS.

**Mode (iii)**

| rule set | \|R\| | matched-base ref | U(p)@100 | U(matched)@100 | Delta_U@100 | differing@100 | U(p)@200 | U(matched)@200 | Delta_U@200 | differing@200 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `new:95` | 95 | `base:matched:0x20260901:95` | 33.11 | 48.25 | -15.14 | 186 | 52.06 | 27.52 | 24.54 | 184 |

Spearman rho(Delta_U, |R|): B=100 = undefined, B=200 = undefined. Delta_U at max |R|: -15.14% (B=100), 24.54% (B=200).

Delta1(v3) at `new:95` (95% bootstrap CI half-width of paired median Delta_U): B=100 = 11.45 pts (median -0.18, CI [-21.22, 1.67]), B=200 = 16.74 pts (median 6.12).

**H1(v3) verdict (iii):** direction FAILS (rho >= 0.9), effect @100 FAILS (|Delta_U(max)| >= Delta1), effect @200 HOLDS.

## SS4 - the order effect on its own (all rule sets |R|=62)

**classical**

| rule set | U@100 | L@100 | differing-from-base@100 | U@200 | L@200 | differing-from-base@200 |
|---|---:|---:|---:|---:|---:|---:|
| `base` | 96.58 | 48.467 | 0 | 40.49 | 21.922 | 0 |
| `base:shuffled:1` | 43.74 | 13.672 | 175 | 23.57 | 14.957 | 150 |
| `base:shuffled:2` | 46.19 | 18.585 | 186 | 25.70 | 7.934 | 143 |
| `base:shuffled:3` | 26.28 | 13.448 | 186 | 1.49 | 0.499 | 151 |
| `base:static:numeric-first` | 1.12 | 0.599 | 186 | 0.44 | 0.002 | 140 |

**rapid (reported, no claim)**

| rule set | U@100 | L@100 | differing-from-base@100 | U@200 | L@200 | differing-from-base@200 |
|---|---:|---:|---:|---:|---:|---:|
| `base` | 0.00 | 0.000 | 0 | 0.00 | 0.000 | 0 |
| `base:shuffled:1` | 0.00 | 0.000 | 22 | 0.00 | 0.000 | 7 |
| `base:shuffled:2` | 0.00 | 0.000 | 20 | 0.00 | 0.000 | 5 |
| `base:shuffled:3` | 0.00 | 0.000 | 28 | 0.00 | 0.000 | 6 |
| `base:static:numeric-first` | 0.00 | 0.000 | 23 | 0.00 | 0.000 | 3 |

**blitz (reported, no claim)**

| rule set | U@100 | L@100 | differing-from-base@100 | U@200 | L@200 | differing-from-base@200 |
|---|---:|---:|---:|---:|---:|---:|
| `base` | 0.00 | 0.000 | 0 | 0.00 | 0.000 | 0 |
| `base:shuffled:1` | 0.00 | 0.000 | 0 | 0.00 | 0.000 | 0 |
| `base:shuffled:2` | 0.00 | 0.000 | 0 | 0.00 | 0.000 | 0 |
| `base:shuffled:3` | 0.00 | 0.000 | 0 | 0.00 | 0.000 | 0 |
| `base:static:numeric-first` | 0.00 | 0.000 | 0 | 0.00 | 0.000 | 0 |

## Seed sensitivity of an inflated point

Registered seed (`0x20260901`) plus two additional interleave seeds (`1`, `2`), classical band.

**dup:124**

| seed variant | U@100 | U@200 |
|---|---:|---:|
| `dup:124` | 41.12 | 25.44 |
| `dup:124:interleave:1` | 37.80 | 32.37 |
| `dup:124:interleave:2` | 50.33 | 22.62 |

Spread (range) across the 3 seeds: B=100 = 12.53 pts, B=200 = 9.76 pts.

**comp:93**

| seed variant | U@100 | U@200 |
|---|---:|---:|
| `comp:93` | 60.55 | 25.60 |
| `comp:93:interleave:1` | 39.95 | 18.77 |
| `comp:93:interleave:2` | 38.70 | 13.07 |

Spread (range) across the 3 seeds: B=100 = 21.85 pts, B=200 = 12.53 pts.

## Registration extras (SS2 / SS5.3-SS5.6 of the v3 registration)

Every rule set this registration's numbers touch, classical band (n=188). `B in sweeps` = B / median apps_per_sweep (one-sweep probe per expression). SS7.1 flat <=> evals/app at B <= 2x `base`'s (62.41 @100, 79.08 @200).

| rule set | \|R\| | fingerprint | aps med | B100 sweeps | B200 sweeps | U@100 | L@100 | Y@100 | U@200 | L@200 | Y@200 | ev/app@100 | x base | flat@100 | ev/app@200 | x base | flat@200 | differing@100/@200 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---|
| `base` | 62 | `e99af8402beaff5d` | 100 | 1.01 | 2.01 | 96.58 | 48.467 | 16.32 | 40.49 | 21.922 | 8.99 | 31.20 | 1.00 | yes | 39.54 | 1.00 | yes | 0 / 0 |
| `base:matched:0x20260901:93` | 62 | `ab09ee08705f96aa` | 84 | 1.18 | 2.37 | 60.55 | 24.384 | 9.80 | 24.82 | 0.160 | 0.08 | 59.36 | 1.90 | yes | 63.11 | 1.60 | yes | 159 / 149 |
| `base:matched:0x20260901:124` | 62 | `5c6917e6bde09de4` | 82 | 1.21 | 2.42 | 43.88 | 16.767 | 7.18 | 16.89 | 1.763 | 0.87 | 40.29 | 1.29 | yes | 49.82 | 1.26 | yes | 184 / 146 |
| `base:matched:0x20260901:186` | 62 | `4c185b36d078890c` | 86 | 1.16 | 2.33 | 13.85 | 1.692 | 0.83 | 4.70 | 0.776 | 0.38 | 79.36 | 2.54 | no | 77.94 | 1.97 | yes | 177 / 159 |
| `base:matched:0x20260901:248` | 62 | `2baca95771460159` | 85 | 1.18 | 2.35 | 4.33 | 0.623 | 0.31 | 3.36 | 0.477 | 0.24 | 144.96 | 4.65 | no | 121.30 | 3.07 | no | 185 / 165 |
| `base:matched:0x20260901:95` | 62 | `51d425eb78ad5821` | 85 | 1.18 | 2.35 | 48.25 | 17.706 | 7.52 | 27.52 | 16.837 | 7.21 | 67.98 | 2.18 | no | 69.84 | 1.77 | yes | 178 / 143 |
| `base:shuffled:1` | 62 | `0b5612ba8d0abf82` | 88 | 1.14 | 2.29 | 43.74 | 13.672 | 6.01 | 23.57 | 14.957 | 6.51 | 58.72 | 1.88 | yes | 59.58 | 1.51 | yes | 175 / 150 |
| `base:shuffled:2` | 62 | `02c5ed25ec0daff4` | 85 | 1.18 | 2.35 | 46.19 | 18.585 | 7.84 | 25.70 | 7.934 | 3.68 | 22.20 | 0.71 | yes | 41.41 | 1.05 | yes | 186 / 143 |
| `base:shuffled:3` | 62 | `185ebacfa932f651` | 85 | 1.18 | 2.35 | 26.28 | 13.448 | 5.93 | 1.49 | 0.499 | 0.25 | 130.67 | 4.19 | no | 115.81 | 2.93 | no | 186 / 151 |
| `base:static:numeric-first` | 62 | `9e6d66598d997f37` | 88 | 1.14 | 2.29 | 1.12 | 0.599 | 0.30 | 0.44 | 0.002 | 0.00 | 46.76 | 1.50 | yes | 57.55 | 1.46 | yes | 186 / 140 |
| `dup:93` | 93 | `83e610e33e782a68` | 142 | 0.70 | 1.41 | 43.76 | 20.910 | 8.65 | 6.69 | 0.181 | 0.09 | 75.37 | 2.42 | no | 79.54 | 2.01 | no | 171 / 147 |
| `dup:124` | 124 | `b207aa331bb625ab` | 300 | 0.33 | 0.67 | 41.12 | 15.597 | 6.75 | 25.44 | 5.088 | 2.42 | 78.07 | 2.50 | no | 76.37 | 1.93 | yes | 186 / 157 |
| `dup:186` | 186 | `3a00c565900b48e6` | 502 | 0.20 | 0.40 | 15.44 | 2.565 | 1.25 | 9.47 | 1.047 | 0.52 | 91.81 | 2.94 | no | 92.67 | 2.34 | no | 179 / 178 |
| `dup:248` | 248 | `43c43d764ef7f76b` | 996 | 0.10 | 0.20 | 38.67 | 10.996 | 4.95 | 27.02 | 6.997 | 3.27 | 78.85 | 2.53 | no | 74.75 | 1.89 | yes | 186 / 160 |
| `comp:93` | 93 | `904ceec9b110e89e` | 87 | 1.15 | 2.30 | 60.55 | 24.604 | 9.87 | 25.60 | 0.175 | 0.09 | 85.77 | 2.75 | no | 92.42 | 2.34 | no | 159 / 151 |
| `comp:124` | 124 | `a7600e5942f0baa5` | 90 | 1.12 | 2.23 | 41.69 | 14.371 | 6.28 | 21.22 | 2.684 | 1.31 | 100.60 | 3.22 | no | 102.79 | 2.60 | no | 184 / 148 |
| `new:95` | 95 | `113cca49c99cc850` | 292 | 0.34 | 0.69 | 33.11 | -0.000 | -0.00 | 52.06 | 0.000 | 0.00 | 69.33 | 2.22 | no | 61.98 | 1.57 | yes | 188 / 186 |
| `dup:124:interleave:1` | 124 | `6333431113f09bde` | 266 | 0.38 | 0.75 | 37.80 | 7.564 | 3.52 | 32.37 | 13.682 | 6.02 | 97.46 | 3.12 | no | 89.30 | 2.26 | no | 186 / 178 |
| `dup:124:interleave:2` | 124 | `5ca26c601faaf74c` | 293 | 0.34 | 0.68 | 50.33 | 27.605 | 10.82 | 22.62 | 1.650 | 0.81 | 61.67 | 1.98 | yes | 69.46 | 1.76 | yes | 187 / 157 |
| `comp:93:interleave:1` | 93 | `71851d821506a78b` | 86 | 1.16 | 2.31 | 39.95 | 18.647 | 7.86 | 18.77 | 1.038 | 0.51 | 94.87 | 3.04 | no | 99.45 | 2.51 | no | 186 / 150 |
| `comp:93:interleave:2` | 93 | `437971d00006a362` | 90 | 1.12 | 2.23 | 38.70 | 23.676 | 9.57 | 13.07 | 2.008 | 0.98 | 128.14 | 4.11 | no | 123.49 | 3.12 | no | 184 / 165 |

Delta1 under v1's definition (95% bootstrap CI half-width of median U at `base`, 10000 resamples, seed 42): B=100 = 28.23 pts (median 96.58, CI [73.12, 129.58]); B=200 = 12.34 pts (median 40.49, CI [25.70, 50.39]).

**Delta_U(max) vs Delta1 (v1's definition)**

| mode | B | Delta_U(max) | Delta1(v1) | ratio | clears? |
|---|---:|---:|---:|---:|---|
| (i) | 100 | 34.34 | 28.23 | +1.22 | yes |
| (i) | 200 | 23.66 | 12.34 | +1.92 | yes |
| (ii) | 100 | -2.19 | 28.23 | -0.08 | no |
| (ii) | 200 | 4.33 | 12.34 | +0.35 | no |
| (iii) | 100 | -15.14 | 28.23 | -0.54 | no |
| (iii) | 200 | 24.54 | 12.34 | +1.99 | yes |

**Paired Delta_Y(p) = Y(p) - Y(OrderMatchedBase) and Delta2(v3) = max(0.02, Delta_Y at |R|max)**

| mode | rule set | \|R\| | Y(p)@100 | Y(matched)@100 | Delta_Y@100 | Y(p)@200 | Y(matched)@200 | Delta_Y@200 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| (i) | `dup:93` | 93 | 8.65 | 9.80 | -1.15 | 0.09 | 0.08 | 0.01 |
| (i) | `dup:124` | 124 | 6.75 | 7.18 | -0.43 | 2.42 | 0.87 | 1.55 |
| (i) | `dup:186` | 186 | 1.25 | 0.83 | 0.42 | 0.52 | 0.38 | 0.13 |
| (i) | `dup:248` | 248 | 4.95 | 0.31 | 4.64 | 3.27 | 0.24 | 3.03 |
| (ii) | `comp:93` | 93 | 9.87 | 9.80 | 0.07 | 0.09 | 0.08 | 0.01 |
| (ii) | `comp:124` | 124 | 6.28 | 7.18 | -0.90 | 1.31 | 0.87 | 0.44 |
| (iii) | `new:95` | 95 | -0.00 | 7.52 | -7.52 | 0.00 | 7.21 | -7.21 |

| mode | at | Delta2(v3)@100 | Delta2(v3)@200 |
|---|---|---:|---:|
| (i) | `dup:248` | 0.0464 | 0.0303 |
| (ii) | `comp:124` | 0.0200 (floor) | 0.0200 (floor) |
| (iii) | `new:95` | 0.0200 (floor) | 0.0200 (floor) |

## Summary

**The |R| effect, order held fixed, is small and — where it is measurable at all — negative,
not positive; the order effect dwarfs it.** Mode (i) (pure duplicates, closure identical to
`base` by construction) is the only mode with enough grid points (4) to read a trend: `Delta_U`
grows monotonically and strongly with `|R|` (Spearman rho = 1.000 at B=100, 0.800 at B=200,
clearing Delta1(v3) by a wide margin at both budgets), but the sign is *increasing regret* —
`Delta_U` rises from -16.8% at `dup:93` to +34.3% at `dup:248` (B=100). Since mode (i) cannot
change what the search can reach (`DuplicateRule` delegates verbatim to its inner rule), this
is not a capacity effect: it is budget dilution — a fixed application-count budget spends a
shrinking fraction of its applications on the productive 62 rules as redundant duplicate slots
are added to every sweep, holding those 62 rules' relative sweep position fixed. Modes (ii)
(2 points: `comp:93`, `comp:124`) and (iii) (1 point: `new:95`) don't have enough grid points to
read a trend at all — mode (ii)'s two-point Spearman rho is definitionally +-1 regardless of
effect size, and its `Delta_U` values are small (-2.2% to +4.3%) and inconsistent in sign
between B=100 and B=200; mode (iii) has one point, so rho is undefined by construction.

**The order effect, measured on the same 62 rules with no rule added or removed, is 2-3x
larger than mode (i)'s |R| effect at its biggest, and runs the opposite direction.** `base`
(production order) sits at U=96.6%/40.5% (B=100/200) — worse than every other order tested.
The three independently-shuffled seeds land at U=26-46% (B=100) / 1.5-25.7% (B=200), a 50-70
point improvement over production order from nothing but a different permutation of the
identical 62 rules. `StaticReorder(NumericFirst)` — the production quick-win candidate, rules
sorted by descending TRAIN strict-positive rate — does far better still: U=1.1%/0.4%, a ~95
and ~40 point drop from `base` at B=100/200 respectively, and better than every random shuffle
tested too. Seed sensitivity on an inflated point (`dup:124`, `comp:93` under the registered
seed plus two more) shows a real but bounded spread — 9.8-21.9 points across three seeds — so
individual `Delta_U` values in SS3 (anchored to one seed) carry that much seed noise, but every
seed tested is far below `base`'s unshuffled value, so the order effect's *existence and
direction* is not seed-fragile even though its exact magnitude at a given point is.

**Reading.** With order held fixed, the |R| effect is not the driver of v2's raw `U(|R|)`
finding — v2 §6b's confound argument holds: order, not rule count, is what moved `U`. Where
`|R|` has a measurable effect at all (mode i only), it is small relative to the order effect
and points the wrong way for a "more rules helps" story — it is a dilution cost, not a capacity
gain. The order effect is the real, large, and reproducible-across-seeds finding here, and
`StaticReorder(NumericFirst)` — the zero-runtime-cost production quick-win the orchestrator
flagged — is the single best order measured, beating even random shuffles.
