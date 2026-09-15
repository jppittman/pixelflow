# Test quality control follow-up — 2026-09-04

> **Mutation counts superseded (2026-09-08).** Every `cargo mutants` number
> below — the 25/10/12/3 first sweep, the 25/20/2/3 re-run, and the "0 real
> gaps" conclusion — was taken against `guards.rs` as of this pass's base
> (`1673e84`). #1177 (S3b) and #1183 (uniforms) have since rewritten the
> file: it gained `cluster_select_arms`, a `Uniform` arm in the operand
> walk, and — the part that reaches these tests — a cost gate, so an arm
> costing `<= MISPREDICT_PENALTY_CYCLES` now forms no guard at all. The
> mutant population is therefore a different one and these tallies do not
> describe the current tree.
>
> The *finding* stands and was re-confirmed on 2026-09-08: `guards.rs` still
> has no `#[test]` of its own on `main`. The three tests were repaired
> against the new contract (their arms use `Rsqrt`, 21 cycles, because a
> `Neg`'s 3 do not clear the gate) and a fourth was added pinning the gate's
> boundary at `Recip`'s exactly-16. Hand mutation check after the repair:
> `end = last + 1` → `last`, the gate's `<=` → `<`, and
> `MISPREDICT_PENALTY_CYCLES` 16 → 15 are each killed. That is four
> targeted mutants, not a sweep — **a full `cargo mutants` re-run against
> the current file is still owed** before the "0 real gaps" claim is
> restated anywhere.

Scope: scheduled continuation of the test-quality-audit series
(`docs/bugs/*-test-quality-audit-followup.md`). The 2026-09-01 pass's
backlog items 1–3 (`pixelflow-search/egraph/cost.rs` and `graph.rs`,
`pixelflow-codegen/src/emit/*`) all pointed at stale, unmerged draft PRs
from earlier passes (#1049, #1050, #1051, #1054) rather than untouched
code — none of those PRs are on a branch this session created, and three
are `mergeable_state: behind` or `dirty`. Rebasing someone else's stale
branch is out of scope for a routine pass (consistent with every prior
pass's judgement call here); left untouched. `#1054` in particular already
has an open, unaddressed review comment from a previous Claude pass asking
for the mutation numbers to be re-verified before merge — noted below for
whoever picks it up next.

Picked a fresh, previously-unexamined file instead:
`pixelflow-codegen/src/emit/guards.rs` (`analyze_select_guards`, the
Select short-circuit-guard schedule analysis). Confirmed via `grep` that
every other file in `emit/` already has direct `#[test]` coverage;
`guards.rs` and `coverage.rs` did not. `coverage.rs` is a handful of
`OpKind` const arrays already exercised element-by-element by
`emit::tests::backend_op_coverage` — nothing to add there. `guards.rs` had
real, untested logic: only exercised indirectly through whole-kernel
compile-and-execute tests (`emit::tests::sched::sched_select_guards`,
`select_guard_driver`) that check a guard formed at all, never its exact
boundaries.

`cargo-mutants` v27.1.0 (not present in this environment — installed via
`cargo install cargo-mutants --locked`, consistent with every prior pass).

## `pixelflow-codegen/src/emit/guards.rs`

First sweep, scoped to the file, default test command (so the existing
whole-kernel guard tests run alongside anything new):
`cargo mutants -p pixelflow-codegen --file pixelflow-codegen/src/emit/guards.rs -j 4`:
**25 mutants, 10 missed, 12 caught, 3 unviable.**

### Fixed: 3 new tests, all against the `pub(crate)` function directly

`analyze_select_guards` takes `&[Def]` and returns `Vec<SelectGuard>` —
both `pub(crate)` types already re-exported for exactly this purpose
(`regalloc.rs`'s own test module builds `Def`/`ScheduledOp` schedules by
hand with an identical `fn def(value: u32, op: ScheduledOp) -> Def`
helper). Added the same helper here and three tests that hand-build a
schedule rather than going through `ExprArena` — indices need to be
pinned exactly, which arena-derived schedules don't give control over:

- `range_the_true_arm_when_only_it_is_exclusive` / the false-arm mirror —
  a `Select` where one arm is just the mask reused (contributes nothing
  exclusive) and the other has a private `Var`→`Neg` chain. Asserts the
  exact `true_range`/`false_range` tuples and `select_idx`, not just "a
  guard formed" (which the existing whole-kernel tests already checked).
  This alone killed 8 of the 10 missed mutants: the range end's `+ 1`
  (both arms), the top-level `||`/`&&` guard-formation check and both its
  `!=`/`==` operands, and two of the four `mask_idx < start` comparison
  mutants (the `==` and `>` replacements on the false-arm side — the
  true-arm side's `==`/`>` mutants were apparently already caught by the
  pre-existing whole-kernel tests, since only one `<` mutant per branch
  showed up as missed going in).
- `ignore_a_false_operand_missing_from_the_schedule` — a regression test
  for the one gap that needed a purpose-built case: an operand reachable
  only through a `ValueId` the schedule never defines (a "hole" — the
  file's own module doc says arena-composed kernels splicing arbitrary
  fragments can produce these). The dense `vid_to_sched_idx` lookup
  initializes such slots to the sentinel `usize::MAX`; the filter that is
  supposed to drop them (`if idx == usize::MAX { None } else { Some(idx) }`)
  had a real mutant survive (`==` → `!=`) because nothing exercised the
  hole case at all. Flipped, the sentinel gets treated as a real schedule
  index, and range-end arithmetic (`+ 1`) overflows in debug builds.

### 2 documented equivalent mutants — not fixed, and why

Both remaining misses are `replace < with <=` at the two symmetric
`mask_idx < start` checks (lines 224, 244). `start` is always the minimum
schedule index among nodes in `{true,false}_exclusive`, a set built as
`{true,false}_deps` minus `mask_deps` — and `mask_vid` is unconditionally
a member of `mask_deps` (it's the root `transitive_deps` is called on).
So the mask's own schedule position can never itself be a member of
`true_indices`/`false_indices`, which means `mask_idx == start` is
unreachable: every schedule index is unique to one `Def`, and the mask's
index is provably excluded from the set `start` is drawn from. `<` and
`<=` therefore accept exactly the same inputs here — a genuine equivalent
mutant, the same shape as the `depth_cost` `>`/`>=` case documented in the
2026-08-22 pass on `cost.rs`.

### Re-verified

`cargo mutants -p pixelflow-codegen --file pixelflow-codegen/src/emit/guards.rs -j 4`:
**25 mutants, 20 caught, 2 missed (both documented above), 3 unviable, 0
real gaps.**

## Verified

- `cargo test -p pixelflow-codegen --lib guards::`: 3 passed, 0 failed
  (the 3 new tests; no pre-existing tests lived in this file).
- `cargo test -p pixelflow-codegen` (all targets incl. doctests): passed,
  0 failed.
- `cargo test --workspace`: passed, 0 failed.
- `cargo clippy -p pixelflow-codegen --lib --tests -- -D warnings`: clean.
- `cargo fmt -p pixelflow-codegen -- --check`: clean.
- `cargo mutants -p pixelflow-codegen --file pixelflow-codegen/src/emit/guards.rs -j 4`:
  25 mutants, 20 caught, 2 documented-equivalent, 0 real gaps.

## Recommended next steps (not done here)

1. PR #1054 (`pixelflow-codegen/src/emit/x86_64.rs` mutation gaps) has an
   open review comment from 2026-09-02 asking for the mutation-coverage
   claim to be re-verified against the file's current state (it's moved
   under two encoder refactors since the branch was opened) before
   merging. Worth either landing that verification or closing the PR if
   it's superseded.
2. PRs #1049 (`egraph/graph.rs` STYLE.md naming) and #1051 (`egraph/cost.rs`
   — believed superseded by 2026-08-22's in-tree fix, worth confirming and
   closing) remain open, unmerged, and `mergeable_state: behind`. A
   decision (rebase-and-land or close) from whoever owns that series would
   let future passes stop re-flagging them.
3. `pixelflow-codegen/src/emit/mod.rs` (5,066 lines), `regalloc.rs` (2,387),
   `aarch64.rs` (2,487), `avx2.rs`/`avx512.rs` (~1,100 each), and
   `executable.rs` (898) all already have in-file `#[test]` modules per
   this pass's `grep` sweep, but none have had a `cargo mutants` pass
   recorded against them in this series — unlike `x86_64.rs` (#1054) and
   now `guards.rs`, "has tests" hasn't been confirmed to mean "the tests
   catch mutants" for any of them yet.
4. `pixelflow-core/src/backend/arm.rs`'s NEON impls — still open per every
   pass since 2026-08-15; needs an aarch64 host, still unavailable here.
