# Open-PR sweep, close-out: one blocker, and it is a clock

**Date:** 2026-09-08
**Author:** Claude (scheduled sweep)
**Supersedes nothing.** Extends
[2026-09-08-open-pr-triage.md](2026-09-08-open-pr-triage.md) and
[2026-09-08-open-thread-decisions.md](2026-09-08-open-thread-decisions.md)
with the state at the end of the run and the one complication neither
records.

---

## 1. The finding: #1213 is red on a timeout, and the timeout hides the suite

`Presubmit Tests` run 34258806351 on `14d5fa05`, failing identically on
`ubuntu-latest` and `macos-latest`. Every other job in the run is green —
Clippy, Rustfmt, Feature matrix, ISA matrix, Behavior contracts, the four
metadata jobs. Only `Run workspace tests` fails, and it fails on the clock:

```
Summary [1103.243s] 1314/2395 tests run: 1312 passed (9 slow), 2 timed out, 16 skipped

TIMEOUT [600.012s]  pixelflow-graphics::kernel_glyph_optimize
                    optimized_glyph_matches_raw_within_reassociation_noise
TIMEOUT [600.011s]  pixelflow-graphics::loop_blinn_winding
                    a_glyph_is_exactly_zero_outside_its_support
SLOW    [454.742s]  pixelflow-graphics::loop_blinn_winding glyphs_wind_like_the_oracle
SLOW    [ 67.135s]  pixelflow-graphics::loop_blinn_winding synthetic_outlines_wind_like_the_oracle
```

Two tests hit nextest's 600 s limit; a third passes 145 s under it.

**The part worth naming is the second-order effect.** nextest cancels on
first failure, so `1081/2395` tests never ran — 45% of the suite. The branch
is not "red on two tests". It is *unmeasured on nearly half of them*, and
the two visible failures are the only reason anyone knows. A correctness
regression anywhere in that unrun 45% is currently invisible, and would stay
invisible until the clock is fixed, because the clock fails first every time.

That is a CI-design observation rather than a fact about this branch:
**a timeout and a fail-fast compose into a mask.** Any suite where the
slowest test is near the limit will report its slowness and conceal
everything scheduled after it.

Reproduced locally on this host, independently of CI: on the branch head
merged with `main`, `pixelflow-graphics::text_union_identity` took
**4282 s** and a single test from it took **3998 s** on its own. Same
signal, an order of magnitude past the CI limit.

### What this is not

Not a flake — it reproduces on two platforms and locally. Not an
infrastructure failure — the runners were healthy and every other job on the
same commits passed. Not a correctness failure *that has been observed*; the
two timing-out tests never reached an assertion.

### What must not be done about it

Raising the nextest timeout, marking the tests `#[ignore]`, or splitting them
until each fits. All three make the job green while the suite gets slower,
and CLAUDE.md's rule is explicit that a test is never skipped, disabled or
quarantined to reach green. The 45% that does not run would still not run.

The fix is that the glyph pipeline gets faster, which is the branch's own
subject.

## 2. Why this sweep did not fix it

`claude/g1-loop-blinn-glyph-iznybv` was under active development throughout
the run. Measured from `edacdb93`, the head this sweep first fetched, to
`fae39167`: 28 commits, of which **12 are the branch author's own** and the
rest arrived through their own `main` merge (`cc6d0aed`). The last two —
`bee78133`, `fae39167` — landed *after* the failing CI run, and `bee78133`
touches `fonts/cache.rs` and `kernel_glyph_optimize.rs`, one of the two
timing-out tests. Whether it moves the clock is not yet known: no CI run
exists for `fae39167` at the time of writing, and the commit's stated subject
is the API (a kernel carrying its own tabulations), not performance.

This is recorded because the sweep learned it the expensive way. Mid-run it
resolved #1213's merge conflict and fixed the compile break the merge
exposed — `egraph_off_on` pushing a `Glyph` where a `Kernel` was expected,
after `Font::glyph_kernel_scaled`'s signature changed under a bin that
arrived from `main`. That work was locally verified (workspace check, clippy,
rustfmt, three crates' tests) and then **discarded unpushed**, because a
re-fetch showed the branch author had fixed the same break 40 minutes earlier
and differently: `ad372cbe` takes the kernel alone, and `bee78133` then
deletes `Glyph.binding` outright so a kernel carries its own tabulations. The
sweep's version would have been both redundant and wrong about the direction
the API was moving.

**The rule that follows:** re-fetch immediately before pushing to a branch
this session does not own, not merely before starting work on it. A sweep's
snapshot of a branch is only valid for as long as nobody else is typing, and
on this repository somebody usually is.

## 3. State at close

| PR | mergeable | CI | open threads |
|---|---|---|---|
| #1206 | clean | moving (pushed during the run) | 0 |
| #1207 | clean | green | 8 |
| #1209 | clean | green | 0 |
| #1213 | clean, 0 behind | **red — §1** | actively worked |
| #1215 | clean | green | 10 |
| #1224 | — | — | opened mid-run |
| #994 | clean | draft by intent | 0 |

Merged by this sweep: #1054, #1086, #1154, #1188, #1199, #1217, #1218,
#1219, #1220, #1221, #1222, #1223.

`main` was red on arrival and is now green and gated:
`pixelflow-pipeline/src/bin/corpus_gaps.rs` had no `[[bin]]` entry, so Cargo
auto-discovered it without `required-features` and it broke the Feature
matrix under `--no-default-features`. Fixed, and
`scripts/check-bin-declarations.sh` now refuses the class in ~6 s, per
CLAUDE.md's rule that a gap in CI is a check to write rather than a caveat to
attach.

The 18 threads on #1207 and #1215 are research verdicts — what a claim
concludes, where a threshold sits — not defects. Each has a written
recommendation in `2026-09-08-open-thread-decisions.md`; each awaits an
accept or reject.

## 4. The condition, honestly

The sweep's brief was that every open PR be rebased, thread-free and green.
Rebased: yes, all seven. Green: six of seven, with §1 the exception. Threads:
18 remain, all decisions rather than work.

It is worth recording that the brief is **not reachable by one pass** on this
repository as it currently runs. A new PR (#1224) opened during the sweep;
three branches took pushes; and each push to a docs PR draws a fresh review
round, so #1215 went 7 threads → 4 → 10 without anything being wrong with it.
A condition quantified over "all open PRs" is a fixed point, and there is no
fixed point while other sessions are writing. What a sweep can deliver is a
cut: every PR examined against a named commit, every finding either fixed or
handed over with a recommendation. That is what the three documents in this
series are.

Related, and the reason this file exists at all: the claims ledger's fifth
failure mode is provenance, and the sweep hit it four times in one run —
an audit backlog, `L057`'s status, three ledger rows and its own triage
table all went stale, two of them because of merges the sweep itself made.
Every one of those documents records *when* it was written and none records
*which tree it was verified against*. This one names commits throughout for
that reason.
