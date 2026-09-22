# Post-submit quality pipeline

Three workflows run against every push to `main`. Together they answer: did
this commit break tests, make them flaky, or slow the hot paths — and if it
broke them, who backs it out?

```
push to main
  ├── Postsubmit Flake Detection ──(conclusion: failure)──▶ Automatic Revert
  │     test:       5× per (OS, suite); flaky ⇒ issue, consistent ⇒ fail
  │     isa-matrix: 1× per ISA level; any failure ⇒ fail, no flaky path
  └── Benchmark Regression Check
        Criterion vs gh-pages baseline; >25% ⇒ issue + commit comment
```

## Flake detection (`postsubmit-flake-detection.yaml`)

Two independent jobs, two different policies, because they're answering "is
this failure real?" at different granularities.

### `test`: 5 iterations, flaky vs. consistent

Each (OS × suite) job builds the tests once, then runs the suite **5 times**
with nextest (per-test 10-minute cap from `.config/nextest.toml`). The five
iterations classify the commit:

| Result | Meaning | Action |
|---|---|---|
| 5/5 pass | healthy | nothing |
| 1–4/5 fail | **flaky** | file/update a `flaky-test` issue; workflow stays green |
| 0/5 pass (or build fails) | **consistently broken** | workflow fails → automatic revert |

The distinction is the point: a flake reverted is a lie recorded (the next
commit "fixes" it by luck), and a hard breakage merely issue-filed rots on
main. The `flake-report` job runs with `continue-on-error` so an issue-filing
hiccup can never masquerade as a postsubmit failure and trigger a revert.

### `isa-matrix`: single run, no flaky path

Builds and lints once, then runs `cargo test --workspace` once per x86-64 ISA
tier (AVX2, AVX-512) this host's CPU supports — deliberately no retry, no
5-iteration classification. A test that only fails at one ISA level is
exactly as real a break as one that fails everywhere; retrying it would hide
the nondeterminism or ISA-specific bug it's surfacing, not resolve it. Any
failure fails the job outright, which fails the workflow and triggers the
automatic revert — including on a run where `test` above was merely flaky:
`flake-report`'s issue then says the revert was triggered/proposed because of
`isa-matrix`, not the flake, so it never claims a commit "was not reverted"
when it may in fact be reverted (or the revert PR may conflict or go
unmerged -- `automatic-revert.yaml` hasn't even run yet when this issue is
filed, since it only starts once this whole workflow concludes).

## Automatic revert (`automatic-revert.yaml`)

Fires on a failed flake-detection run. Creates a tracking issue and a revert
PR (branch `revert-failed-postsubmit-<sha7>`), with guards:

- **Dedup** — one revert branch per failing SHA, however many runs report it.
- **Loop guard** — if the failing commit is itself an automated revert,
  re-reverting would re-land the original bad commit; it escalates to an
  issue instead.
- **Merge commits** — reverted against the first parent (`-m 1`).
- **Conflicts** — a revert that doesn't apply cleanly becomes an issue
  asking for a manual revert, not a broken branch.

Reverts are proposed as PRs, not pushed to `main` directly: presubmit still
gets to check that the revert itself builds.

## Benchmark regression (`benchmark_regression.yaml`)

Runs `cargo bench --workspace --benches -- --output-format bencher` and feeds
the result to `benchmark-action/github-action-benchmark`, which keeps the
baseline series on the `gh-pages` branch. A benchmark more than **25%** slower
than baseline gets a commit comment and one open `performance-regression`
issue (subsequent regressions comment on the existing issue rather than
piling up new ones). Runs are serialized by a concurrency group so baselines
land in commit order.

Perf regressions do **not** auto-revert: hosted-runner noise makes a
threshold breach evidence, not proof. The issue is the escalation path.

For the same reason this job is **postsubmit-only** and deliberately does not
run on pull requests. One hosted-runner sample against a single baseline point
alerts on noisy neighbors as readily as on code, and a presubmit check that
fails for reasons the author cannot act on is one that gets ignored. What makes
the signal actionable — attribution to a specific commit, and the tracking
issue — only exists once the commit is on `main`.

`--benches` only works because every `[lib]` in the workspace sets
`bench = false` and every `[[bench]]` target is Criterion with
`harness = false` — the libtest harness rejects `--output-format bencher`.
Keep new crates on that convention or the benchmark job will fail at
flag-parse time.
