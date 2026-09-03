# Re-measuring the latency prior on current `main`

**Date:** 2026-09-02
**Machine:** Apple M2 Max (`Mac14,6`), 12 cores, aarch64 NEON JIT
**Source rev:** `ddb51a91`
**Protocol:** `pixelflow-pipeline/examples/measure_latency_prior.rs`, 7 independent runs,
each gated on 1-minute load average < 4 (observed 3.37–3.97)
**Outputs:** this file, `.csv`, `.json`

## Why re-measure

`pixelflow-search/src/egraph/cost.rs`'s per-op cycle table was last measured on
2026-08-14 (`e4388c54`, "Round 1"). Since then the code generator was substantially
rewritten: #1071 collapse-JIT unification, #1082 one kernel ABI / one compile entry,
#1092 and #1119 the loop nest reaching the compile pipeline and the allocator,
#1075–#1077 x86 memory operands, MulAdd shapes, and the ABI as a type.

Extraction is argmin over that table. If the coefficients are stale, every cost
number this repo reports is in stale units — so the table had to be refreshed
*before* anything changes the extraction objective, or the two effects could never
be told apart.

## Headline

**The table largely survived, and three entries were pricing a function that no
longer exists.**

- 24 of the 29 measured ops came back within their own cross-run spread. Those
  numbers still describe this machine.
- `Sin`, `Cos` and `Tan` are **+34% to +37%** — not drift, a different lowering.
- `Sqrt` (15 → 13) and `Select` (4 → 3) moved past their spread by smaller margins.
- Re-extracting all **206 real production kernels** under old vs new coefficients:
  **0 kernels change their extracted term, and 0 regress.**

Two defects were found in the instrument itself, described in §4. One of them
stopped the program from running at all on current `main`.

## 1. The drift table

Cross-run spread is (max − min) / median over the 7 runs — the only honest noise
estimate available on a shared box. An op is called MOVED only when its drift
exceeds both that spread and 5%, *and* the rounded table entry actually changes.

| op | OLD | NEW (median) | ratio | drift | spread | verdict |
|---|---:|---:|---:|---:|---:|---|
| `Cos` | 75 | 102.96 | 1.373 | +37.3% | 6.1% | **MOVED → 103** |
| `Sin` | 70 | 95.04 | 1.358 | +35.8% | 5.6% | **MOVED → 95** |
| `Tan` | 87 | 116.78 | 1.342 | +34.2% | 6.2% | **MOVED → 117** |
| `Select` | 4 | 3.27 | 0.818 | −18.2% | 11.0% | **MOVED → 3** |
| `Sqrt` | 15 | 13.40 | 0.893 | −10.7% | 3.2% | **MOVED → 13** |
| `Neg` | 3 | 2.64 | 0.880 | −12.0% | 115.9% | noise |
| `Recip` | 16 | 14.35 | 0.897 | −10.3% | 11.7% | noise (spread > drift) |
| `Mul` | 5 | 5.50 | 1.100 | +10.0% | 11.5% | noise (spread > drift) |
| `Max` | 3 | 2.70 | 0.900 | −10.0% | 5.6% | rounds to 3, unchanged |
| `Abs` | 3 | 2.72 | 0.907 | −9.3% | 112.5% | noise |
| `MulAdd` | 5 | 5.42 | 1.084 | +8.4% | 8.1% | held at Mul parity |
| `Min` | 3 | 2.78 | 0.927 | −7.3% | 45.0% | noise |
| `Floor` | 4 | 3.77 | 0.943 | −5.8% | 6.9% | noise |
| `Exp` | 75 | 71.58 | 0.954 | −4.6% | 5.2% | noise |
| `Ceil` | 4 | 3.82 | 0.955 | −4.5% | 20.9% | noise |
| `Log10` | 134 | 128.15 | 0.956 | −4.4% | 6.0% | noise |
| `Log2` | 122 | 116.80 | 0.957 | −4.3% | 5.6% | noise |
| `Exp2` | 69 | 66.13 | 0.958 | −4.2% | 4.8% | noise |
| `Ln` | 128 | 122.71 | 0.959 | −4.1% | 8.9% | noise |
| `Asin` | 103 | 99.24 | 0.963 | −3.7% | 11.0% | noise |
| `Acos` | 103 | 99.39 | 0.965 | −3.5% | 7.6% | noise |
| `Pow` | 196 | 189.27 | 0.966 | −3.4% | 5.6% | noise |
| `Rsqrt` | 21 | 20.32 | 0.968 | −3.2% | 6.5% | noise |
| `Round` | 4 | 3.89 | 0.973 | −2.7% | 40.6% | noise |
| `Div` | 11 | 10.90 | 0.991 | −0.9% | 9.5% | noise |
| `Atan2` | 79 | 79.55 | 1.007 | +0.7% | 5.9% | noise |
| `Atan` | 79 | 79.50 | 1.006 | +0.6% | 8.5% | noise |
| `Sub` | 4 | 4.02 | 1.005 | +0.5% | 10.2% | noise |
| `Add` | 4 | 4.00 | — | — | — | anchor |

The ops Round 1 corrected are the reassuring part of this table: `Pow` −3.4%,
`Rsqrt` −3.2%, `Recip` −10.3%, `Div` −0.9%, and the log/exp family within 5%.
**Round 1's headline corrections are confirmed, not overturned.**

## 2. The trig row is a different function, not drift

`Sin`, `Cos` and `Tan` are the only ops that moved materially, and they are exactly
the three that share `expand_sin`'s range reduction. The ops that do *not* use it
did not move:

| uses the trig reduction | | | does not | | |
|---|---:|---:|---|---:|---:|
| `Sin` | 70 → 95.04 | **+35.8%** | `Atan` | 79 → 79.50 | +0.6% |
| `Cos` | 75 → 102.96 | **+37.3%** | `Atan2` | 79 → 79.55 | +0.7% |
| `Tan` | 87 → 116.78 | **+34.2%** | `Asin` | 103 → 99.24 | −3.7% |
| | | | `Acos` | 103 → 99.39 | −3.5% |

That is a mechanistic fingerprint, not a noise pattern. The cause is #992, which
gave `sin` a Cody-Waite range reduction that survives large arguments. It landed at
`084d3efc`, **62 seconds before** Round 1's own measurement commit `e4388c54` — so
Round 1 almost certainly timed the pre-fix reduction and then committed a table
describing code the tree no longer contained. A correct reduction costs about 25
cycles more than the one it replaced; the table now says so.

Round 1's internal-consistency identity still holds under the new numbers, which is
independent evidence the measurement is coherent:

```
Log2 116.80  +  Mul 5.50  +  Exp2 66.13  =  188.43     vs  Pow measured 189.27
```

## 3. What the refresh changes in production: nothing that moves a term

All 206 real-kernel `.arena` dumps (`cellgrid`, `glyph`, `shader`, `psychedelic`)
were saturated **once each** and then extracted twice — once under the Round 1
table, once under the refreshed one. Saturation takes no cost model on the default
path, so extracting twice from one e-graph removes saturation nondeterminism and
makes every difference attributable to the table alone.

| | |
|---|---|
| kernels | 206 |
| extracted term changed | **0** |
| worse under the new table | **0** |
| better under the new table | 0 |

The refresh moves *reported cost* but not *chosen term*:

| | change in reported `dag_cost` (same term, re-priced) |
|---|---|
| median | −3.13% |
| p25 / p75 | −3.37% / −3.01% |
| min / max | −5.67% / +33.90% |
| priced lower / higher | 198 / 6 |

By category: `glyph` −3.13% (n=190), `shader` −0.94% (n=12), `cellgrid` −0.93%
(n=3), `psychedelic` +9.79% (n=1 — the one trig-heavy kernel, which is exactly
where a +35% `Sin` should show up).

**No kernel regresses, so the hold condition does not fire.**

### Why zero, and why that is a real result rather than a broken probe

A "0 changed" result is the shape a broken measurement takes, so it was checked
three ways before being believed:

1. **The two tables really do price differently** — 204 of 206 kernels report
   different costs under the two models.
2. **Positive control.** Re-running with a deliberately perverse snapshot
   (`Sqrt = 4000`, `Sin = 4000`) moved `glyph16:U+0021` from 3,125 to 102,750 — a
   33× cost swing — and the extracted term *still* did not change. Extraction
   cannot avoid these ops on this corpus because the saturated e-graph contains no
   alternative lowering to switch to. The insensitivity is a property of the
   corpus and the rule set, not of the probe.
3. **The detector is unit-tested** (`cost_table_ab::tests`): `term_signature`
   discriminates `Mul` from `Add` and is blind to node push order, and
   `arena_dag_cost` follows the table it is handed and prices a shared subterm
   once.

## 4. Defects found in the instrument

### 4.1 The program did not run on current `main` (fixed here)

`measure_latency_prior` aborted every time, deterministically, in five consecutive
runs:

```
jit_bench plausibility failure (harness bug, not data): raw 1.8208ns for a
47-op expression is below the 2.3500ns floor (0.05ns/op)
```

`MIN_NS_PER_OP` prices ops the machine *issues*, but `op_count` counts nodes in the
arena the caller hands in. Those stopped being the same number when the backend
learned to fuse (#1076): a `Mul` feeding an `Add` is two arena nodes and one
`FMLA`. The source count is an **upper** bound on issued ops where the floor needs
a **lower** one, so the guard was unsound in the false-positive direction and
rejected correct measurements.

It was verified to be a false positive, not a codegen bug, before the guard was
touched: the offending kernel computes the right answer (`jit == eval_scalar ==
19.58`). The floor now takes `op_count.div_ceil(2)` — fusion retires at most one
node per surviving node — which costs the guard nothing it was built for, since an
early `ret` or a mis-scaled timebase lands near zero. `jit_bench.rs` already
anticipated this exact failure in a comment on `var_free_arenas_still_execute_their_ops`.

This is a shared harness used by training label minting, so the change is called
out for review rather than buried.

### 4.2 `BenchMode::Latency` absolute nanoseconds are 4× optimistic (NOT fixed here)

The example prints:

```
# anchors: Add slope 0.220ns/stage ... => est. clock 13.66GHz
```

13.66GHz is impossible; this machine runs at ~3.5GHz. `measure_exec_code` divides
by `INPUT_TUPLES` (64 lane-evals) while the latency loop issues `INPUT_VECTORS`
(16) calls, each computing 4 lanes **in the same SIMD instruction stream**. Four
lanes share one dependency chain, so dividing a chain latency by the lane count
understates it by exactly `LANES`. Round 1 measured an `Add` slope of 0.87ns/stage
(≈3 cycles at 3.4GHz, correct); this run measures 0.220ns — the same number
divided by 4. The unit changed under #1071, which replaced scalar broadcasting with
real 4-lane batches and silently redefined "eval" from *one evaluation* to *one
lane*.

**This does not invalidate the table**, because the table is normalized to
`Add = 4` and a uniform scale factor cancels in `4 · lat_ns / add_slope`. It is
recorded because it does invalidate any absolute-time reading, and because this
repo has been burned by a timebase bug before. Fixing it changes the units of every
minted training label, which is a larger change deserving its own review — **filed
as a referral, not done here.**

### 4.3 The table's memory entries are unmeasured, and unmeasurable by this protocol

21 of the 50 entries are never measured: the six comparisons (3 each), the
integer/bit ops (1 each), `Tuple`/`Buffer`/`Reduce` (0), `Dwrt` (the 1000
sentinel), and — the ones worth naming — **`Gather` and `RawGather` at 10 apiece**.

`measure_exec_code` calls every kernel with a **NULL context pointer** and a
single-batch tile, and gathers read their buffer bases out of that context
register, so a probe containing one would dereference null. The loop-nest and ABI
work (#1075, #1092, #1119) therefore *cannot* show up in what this table charges
for loads and gathers: those entries are guesses, and no re-run of this protocol
can refresh them. That is a gap, not a null result.

### 4.4 Smaller protocol notes

- **The table's "cycles" are not cycles.** `Add` is pinned at 4 while the measured
  `FADD` slope is ~3 real cycles, so every entry is in units of ¾ of a cycle.
  Harmless for argmin (a global scale cancels), but `latency_prior_cycles` is also
  consumed by `OpEmbeddings::init_with_latency_prior` as if the numbers were cycles.
- **`Select` is the weakest measurement in the set.** Its stage is
  `select(ge(acc,0), mul(acc,c0), mul(acc,c1))` and the protocol subtracts only the
  `Mul` slope, on the stated assumption that the compare overlaps it. That is an
  assumption about the scheduler, not a measurement, and it is why `Select` moved
  on the narrowest margin (drift 18.2% against 11.0% spread).
- **`MulAdd` is measured at 5.42 but pinned to `Mul` parity at 5** — a hand-tune
  inside a table documented as measured. Left as-is; noted so it is not mistaken
  for a measurement.

## 5. Machine state

All 7 runs were gated on 1-minute load average < 4 and completed without a sentinel
regime change (the `BenchSession` sentinel aborts on >50% clock drift, so a run that
finishes has self-certified its own timebase).

```
runA  load 3.97   sentinel 2.665ns    runE  load 3.37   sentinel 2.340ns
runB  load 3.89   sentinel 2.299ns    runF  load 3.37   sentinel 2.279ns
runC  load 3.89   sentinel 2.808ns    runG  load 3.37   sentinel 2.238ns
runD  load 3.37   sentinel 2.279ns
```

An earlier attempt at higher load was correctly refused by the harness
(`sentinel regime change ... +50.9% > ±50%` at expression index 64), which is the
guard working as designed. Note that on this host the 1-minute load average is a
poor proxy for CPU contention: it is dominated by `fileproviderd` syncing the
iCloud-backed `Documents/` tree, with CPU 70–80% idle throughout.

## 6. Verdict

**Land the refreshed table.** Five entries changed, three of them because the
function being priced changed underneath the old measurement. No real kernel
changes its extracted term and none regresses, so the refresh is safe by the
stated hold criterion.

**The objective phase should build on the NEW coefficients.** Not because the
extraction output moves — it does not, on this corpus — but because the reported
cost does, by a median −3.13% and by +33.9% on the trig-heavy kernel. Any
before/after the objective work reports would otherwise be contaminated by a
known-stale `Sin`/`Cos`/`Tan` price, and attribution is the entire reason this was
sequenced first.
