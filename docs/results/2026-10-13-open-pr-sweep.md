# Open PR sweep — 2026-10-13

Point-in-time evidence, filed per `docs/README.md`'s classification of
`results/`. The PR states and CI runs below are as of 2026-10-13; the same
findings are posted as comments on each PR, which is where they stay current.

Two open pull requests, both drafts, both authored by the repo owner. Neither
has an unresolved review comment; both had drifted 51 commits behind `main`
and have been brought current. Neither is superseded, obsolete, or
non-salvageable, and neither is a trivial change, so neither was merged —
each carries a decision that is the owner's to make, stated below.

`main` was at `85a39ba` when both PRs were re-verified green.

**A note on "rebased":** `main` advanced to `879cf649` (#1374,
`feat(desloppify): --out journals every reply`) mid-sweep, leaving both PRs
1 commit behind. I first declined to re-update them, on the grounds that the
commit touches only `desloppify/` (5 files under `desloppify/src` and
`desloppify/tests`), is disjoint from both PRs (#994 is `.github/`,
`assets/`, `scripts/`; #1323 is `pixelflow-compiler/`, `pixelflow-ir/`), and
that both still merged cleanly per `git merge-tree` — so re-running would
spend two presubmits and two macOS slots to verify against an unrelated
crate.

**That was the wrong call, and the distinction is worth recording.** "Merges
cleanly" and "rebased" are not the same property. A clean merge says only
that no two edits collide textually; being up to date says the PR's green CI
was produced *against the code that will actually be its base*. A disjoint
diff makes a semantic break unlikely, not impossible — and this very sweep
turned up the case that proves the gap: #1339 deleted
`pixelflow-compiler/tests/common/section_1_7.rs`, and `git merge-tree`
reported no conflict with #1323 regardless, because a dangling `include!`
is not a textual collision. The merge check cannot see that class of break;
only building the merged tree can. So both PRs were brought to 0 behind
`879cf649` — #994 at `690425a5`, #1323 at `c82f8dda` — and re-verified
there.

The cost objection still stands on its own terms, but it is an argument for
making the check cheaper, not for skipping it. The place to spend it is
`scripts/ci-change-scope.sh`, which already knows how to classify a diff and
skip the jobs it cannot affect.

| PR | Rebased | Review threads | CI | Disposition |
|---|---|---|---|---|
| [#994](https://github.com/jppittman/pixelflow/pull/994) macOS signed+notarized DMG release | 0 behind (`690425a5`) | none | green | **Decide: merge inert, or close and reopen when signing is on the roadmap** |
| [#1323](https://github.com/jppittman/pixelflow/pull/1323) `.at` is first-class contramap | 0 behind (`c82f8dda`) | none | green | **Decide: the two questions the PR already asks** |

---

## A complication worth knowing about first

The session's clone was **shallow** (depth ~50). Before unshallowing,
`git merge-base` reported *no common ancestor* between `main` and either PR
branch, and `git rev-list --count` claimed #994 was 1,438 commits ahead with
8 commits on GitHub. Every one of those numbers was an artifact of the
truncated history, not a real divergence.

This is a trap for any future automated sweep: a shallow clone makes a
perfectly ordinary branch look like it has been orphaned by a history
rewrite, which is exactly the signal that would justify abandoning a PR as
non-salvageable. `git rev-parse --is-shallow-repository` is the check, and
`git fetch --unshallow` is the fix. Both PRs share a real merge base,
`b364a440` (#1322, 2026-10-06).

---

## #994 — macOS signed+notarized DMG release pipeline

**What it is.** Purely additive: 3 files, +249/−0 — a tag-triggered
(`v*.*.*`) release workflow, `assets/entitlements.plist`, and
`scripts/make-dmg.sh`. One real commit plus seven "merge main" commits
accumulated over two months of sweeps.

**Brought current and green.** Was 51 behind; updated via the PR's own
base-branch merge, now 0 behind `main` and still exactly those 3 added files.
Full presubmit re-run on the new head `061c391b`
([run 38411685746](https://github.com/jppittman/pixelflow/actions/runs/38411685746)):
**22 of 22 checks green, zero failures**, including both `Test on
ubuntu-latest` and `Test on macos-latest`.

**The load-bearing claim still holds.** The PR grants exactly one
hardened-runtime entitlement, `com.apple.security.cs.allow-unsigned-executable-memory`,
and deliberately not `allow-jit`. Re-verified against `85a39ba`:

- `pixelflow-codegen/src/emit/executable/macos.rs` allocates with plain
  `mmap(MAP_PRIVATE | MAP_ANON)` and flips permissions with
  `mprotect(PROT_READ | PROT_EXEC)`.
- `MAP_JIT` and `pthread_jit_write_protect_np` appear in **no** `.rs` file
  in the workspace.
- `xtask bundle` still exists (`xtask/src/main.rs:46`) and still writes
  `CoreTerm.app` at the workspace root (`main.rs:251`), which is where the
  workflow's relative `CoreTerm.app` references resolve.

So the entitlement set is still the correct minimal one. Had the JIT moved
back to `MAP_JIT` in the interim, this would have shipped a binary that
crashes on its first glyph bake under a hardened runtime. It did not.

**What blocks it is not code.** The five App Store Connect / Developer ID
secrets (`MACOS_CERTIFICATE_P12_BASE64`, `MACOS_CERTIFICATE_PASSWORD`,
`AC_API_KEY_P8_BASE64`, `AC_API_KEY_ID`, `AC_API_ISSUER_ID`) can only be
added by a repository admin, and the `codesign`/`notarytool`/`stapler` steps
cannot be exercised at all without them.

**Recommendation — stop re-verifying this one.** It has been swept at least
twice before (comments of 2026-09-15), each sweep reaching the same
conclusion and one of them noting that the rebase "was not worth its cost."
That is the actual finding: a draft whose only blocker is a credential nobody
in the loop can supply will keep consuming a full presubmit, including a
~25-minute macOS slot, on every sweep, forever, and produce the same answer.
Two ways out, both better than the status quo:

1. **Merge it inert.** The workflow triggers on `v*.*.*` tags and nothing
   else, so on `main` with no tag pushed it never runs. Merging costs
   nothing operationally, removes the PR from the queue, and the pipeline is
   then simply there whenever the secrets appear. The risk is honest and
   small: the first real tag push exercises never-before-run signing steps,
   and the failure mode is a failed release run, not a broken `main`.
2. **Close it.** The branch is additive-only and so will keep merging
   cleanly; reopening or re-cherry-picking later loses nothing. Correct
   choice if signed releases are not on the roadmap this quarter.

Either way the recurring cost goes away. Leaving it open as a draft is the
one option that keeps paying it.

---

## #1323 — `.at` is first-class contramap on any field

**What it is.** +460/−40 across 9 files in `pixelflow-compiler` and
`pixelflow-ir`, plus a 219-line new test. It makes `f.at(u, v)` observe any
`f32`- or `bool`-valued field at another point, denoted
`⟦f.at(u, v)⟧(x, y) = ⟦f⟧(⟦u⟧(x, y), ⟦v⟧(x, y))`, and routes both the new
syntax and kernel-typed application through a single new
`ExprArena::warp` — one contramap in the arena where there were two.

**Brought current and green.** Was 51 behind; updated via the PR's own
base-branch merge to head `a2387764`, now 0 behind `main`. Full presubmit
([run 38413933342](https://github.com/jppittman/pixelflow/actions/runs/38413933342)):
**22 of 22 checks green, zero failures** — including `Test on ubuntu-latest`,
`Test on macos-latest`, `ISA matrix`, `Feature matrix` and `Clippy`. The
merge was also built locally beforehand (clean, exit 0) on a tree whose hash
is byte-identical to the pushed head (`77a84188`), so the local build and the
CI result are about the same content.

**Not superseded.** This was the main thing to establish, given 51 commits
landed since. It is not:

- `ExprArena::warp` does not exist anywhere on `main`.
- D5 — the decision this PR overturns — is still the live entry in the
  decision table on `main`: `| D5 | .at | application is contramap (§1.2) |`.
- The intervening commits are codegen (#1324–#1373: the typed pass pipeline,
  ownership, layout, variance) plus the font-program work (#1338–#1340).
  None of them touch `.at` lowering.

So the question this PR asks is still open, and the PR is still the answer to
it.

**No semantic conflict with the font work, though there nearly was.**
#1339 ("the font is one kernel! program per piece count") *deleted*
`pixelflow-compiler/tests/common/section_1_7.rs` along with its
`macro_rules!` wrapper and its `include!`, and #1323 edits
`pixelflow-compiler/tests/items_block.rs`. Git merges the two without
conflict, which is not by itself reassuring — a deleted `include!` target is
exactly the kind of break a clean textual merge hides. Checked directly:
#1323's only change to `items_block.rs` is a doc comment, and nothing
anywhere in the tree still references `section_1_7`. Clean.

**The one real behavioral change checks out.** `ExprArena::apply` moved from
`linked_parts()` (expand `Ref`s in the *source* arena, then splice) to
`parts()` + `warp` (splice, then expand in the *destination* arena). Those
are only equivalent if ref expansion does not depend on which arena holds
the node — and it does not: `passes::expand_refs` resolves each `Ref` key
through `expanded_referent(key)`, a global store lookup, not a lookup in the
containing arena. That is what makes the PR's "one contramap, not two" claim
structurally true rather than merely true of the cases its tests cover. It
also matters more now than when the PR was written, because #1339 moved the
production font programs onto `kernel!`, so glyph baking goes through this
path.

**The merged plan doc reads coherently.** `main` changed
`docs/plans/2026-09-25-the-language-is-kernel.md` in five commits since the
merge base. After merging, D5 is struck through with its reversal, the dated
metadata entry quoting the request is present, §1.2 is extended, and main's
own new text ("application is contramap", lines 15 and 451) is consistent
with the extension rather than contradicted by it.

**Two decisions, both the owner's, both already stated in the PR.** These are
why this was not merged:

1. **`ExprArena::warp` is new public API in `pixelflow-ir`.** It has to be
   public for the same reason `apply`, `admit` and `close_fold` are: the
   staged lowering site emits `__arena.warp(...)` into the caller's crate.
   CLAUDE.md requires explicit permission to widen public surface, so this
   needs a yes.
2. **`Kernel::at` still runs its own substitution over the DAG.** A test
   pins it to the same program as `warp`, but they remain two
   implementations of one idea, and CLAUDE.md calls a copy "a future
   divergence." The PR argues for deferring the unification because it
   touches the glyph-composition hot path. That deferral is defensible, but
   it is a deliberate acceptance of a duplicate definition and should be
   recorded as such — ideally as a backlog entry, so it is not rediscovered
   later as a bug.

A third observation, offered rather than asked: the PR's own test plan
reports three pre-existing `clippy::must_use_candidate` hits on
`pixelflow-ir/src/arena.rs:986`, `dag.rs:226` and `dag.rs:579` that a newer
stable than CI's raises on unmodified `main`. That is a gap in the gate, not
a defect in this PR — CI pins an older stable and so cannot see them. Per
CLAUDE.md's own rule ("a gap in CI is a check to write, not a caveat to
attach"), the fix is a job or a toolchain bump, in a change of its own.

---

## Summary

- **Superseded by later work:** none.
- **Obsolete:** none.
- **Non-salvageable:** none. (#994 looked orphaned only because of the
  shallow clone.)
- **Recommended for closure:** #994 *if* signed macOS releases are not on
  the roadmap this quarter — and if they are, merge it inert instead. The
  recommendation is to pick one of the two and stop sweeping it.
- **Blocked on the owner, not on work:** #1323, on two questions the PR
  already asks; #994, on five secrets only an admin can add.
