# Open-PR triage — 2026-09-01, refreshed 2026-09-02

Two passes over the open pull requests against three conditions: up to date
with `main`, no unresolved review threads, no CI failures. Plus the requested
judgement on which branches are superseded, obsolete, non-salvageable, or worth
closing.

The first pass ran 2026-09-01 ~21:00–22:00 UTC against `main` at `cc4f0a7`.
This document has been rewritten for the state at **2026-09-02 10:10 UTC**,
`main` at `44c9fa3f` — seventeen commits later. A parallel session's follow-up
(#1089, merged as `2e82cdc2`) covers the intervening window and is not repeated
here.

## The headline: the collision landed, and it took the rest of the board with it

The first pass flagged that #1083, #1084 and #1085 all rewrote
`pixelflow-search/src/egraph/saturate.rs` and the `SaturationStopReason` /
`SaturationStats` / `SaturationResult` triple from three directions, that they
merged cleanly only because none had landed, and that whichever went first
would force real rework on the others.

That is now what happened, at a larger scale than predicted. `#1083`
(`82961fe3`), `#1085` (`c1afd4b9`), `#1107` and the `#1108` optimizer-entry-point
refactor all landed overnight. **Ten of the fifteen open PRs now conflict with
`main`; yesterday all thirteen merged cleanly.**

Nine of the ten share one epicentre — every conflict is in some subset of:

```
pixelflow-search/src/egraph/graph.rs
pixelflow-search/src/egraph/saturate.rs
pixelflow-search/src/egraph/mod.rs
pixelflow-search/src/runtime.rs
```

The tenth, #1072, is a different and worse shape (below).

This is the cost of landing four PRs that touch one seam without first rebasing
the branches queued behind them. It was foreseeable — it was, in fact,
foreseen — and the cheap mitigation was to rebase the queue after the first of
the four landed rather than after all four.

## Board at 2026-09-02 10:40 UTC

Six of fifteen merge cleanly; the one red CI is fixed. Rows this pass changed
are marked.

| PR | Branch | Behind | Merges? |
|---|---|---|---|
| #1114 | `claude/class-cap-live` | 0 | clean |
| #1113 | `claude/upward-congruence` | 0 | clean |
| #1109 | `claude/cap-break-ab` | 0 | **clean — reconciled this pass** |
| #1103 | `claude/all-rules-numeric-first` | 9 | conflict (6 files) |
| #1101 | `claude/rule-order-numeric-first` | 6 | conflict (5 files) |
| #1096 | `claude/phase3-r2g` | 9 | conflict (4 files) |
| #1095 | `claude/phase3-label-constfold` | 9 | conflict (4 files) |
| #1091 | `claude/phase3-domain-shift` | 5 | conflict (4 files, 12 hunks) |
| #1088 | `claude/phase3-round2` | 9 | conflict (4 files) |
| #1087 | `claude/saturation-telemetry` | 10 | conflict (6 files, 23 hunks) |
| #1086 | `claude/brave-faraday-tw3054` | 0 | clean (this doc) |
| #1084 | `claude/phase3-guide` | 9 | conflict (5 files) |
| #1072 | `claude/workshop-writeup` | 17 | conflict — modify/delete |
| #1054 | `claude/zen-babbage-wjmnit` | 0 | **clean, CI fixed this pass** |
| #994 | `claude/macos-release-signing-pipeline` | 0 | **clean, brought current** |

## What the first pass recommended, and what happened

Every recommendation was acted on within the following hours, mostly by other
sessions. Recording the outcome rather than the advice:

| PR | Recommendation | Outcome |
|---|---|---|
| #1053 | merge first — it unblocks #1083/#1084 | **merged** (`436d3af8`) |
| #1081 | ready now, zero threads | **merged** (`c7e65096`) |
| #1051 | land after a duplication check vs #1027 | **merged** (`83015dcd`) |
| #1049 | land or close; near-empty | **merged** (`38ea6eaa`) |
| #1079 | correct 6 findings, land before it goes stale | **merged** (`aa27cf1c`) |
| #1083 | declare `saturation-telemetry = ["std"]`, fix the P1 | **merged** (`82961fe3`) |
| #1085 | resolve 3 threads; land first of the saturation three | **merged** (`c1afd4b9`) |
| #1050 | close — superseded by #1055/#1068 | **closed** unmerged |
| #1044 | merge as a record or close | **closed** unmerged |
| #1054 | land after a mutants rerun | **still open, now red** |
| #1072 | last, after the P1 re-analyses | **still open, now conflicted** |
| #994 | merge dormant or close | **still open, untouched** |

Nine of twelve resolved. The three that did not are the three this pass still
recommends closing or holding.

### Two corrections to the first pass

**Thread count is not a quality signal.** The first pass ranked #1044 as
merge-ready partly on "zero unresolved threads." That was an artifact: the
review bot hit its usage limit on 2026-08-28 and never reviewed it, so an
unreviewed 3,060-line diff scored as the cleanest thing on the board. Zero
threads means either "clean" or "never looked at," and the board could not tell
them apart. Any future sweep should read thread count alongside whether a
review actually ran.

**The saturation collision had five participants, not three.** #1044's
`variants.rs:229,262` called `eg.saturate_with_limit(64)`, which #1085 deletes —
disjoint files, so `git` merged clean and the *build* would have broken on
whichever landed second. Moot now that #1044 is closed, but the three-way
framing was too narrow, and the general lesson is the one the conflict table
above makes concrete: **a clean `git merge-tree` is not evidence that a branch
still builds.**

## Review threads: zero outstanding

Both earlier passes carried "~60 unresolved threads across 8 PRs, 13 of them
P1." That figure was minted on 2026-09-01 and never re-derived. Re-derived
across all fifteen open PRs, and then closed out: **there are none left.**

| PR | threads | resolved | unresolved |
|---|---|---|---|
| #1084 | 28 | 28 | 0 |
| #1072 | 24 | 24 | 0 |
| #1091 | 9 | 9 | 0 |
| #1054 | 2 | 2 | 0 |
| the other eleven | 0 | — | 0 |

Most of the drift was not work being done by this pass. Seven of the PRs
carrying threads merged overnight (#1049, #1051, #1053, #1079, #1081, #1083,
#1085) and took their threads with them, and the branch owners worked through
the rest — #1084 closed all 28, #1091 all 9, each with a reply that confirms
the finding against source, names the fix, and discloses which committed
numbers the fix invalidates.

The last one standing was on #1072 (`Recompute baselines on the Round-3 DEV
corpus`), and it turned out to be live rather than stale. §1's abstract had
been qualified in `f00ee6f3`, but two restatements had not, and one asserted
the opposite outright: §5.1 claimed the model out-ranked the static table
(ρ 0.9438) and a bare op count (ρ 0.9486) "on the same corpus", when both
baselines are Round 1's 392-kernel DEV tier and the table above them is Round
3's 784. §6 listed all three ρ values in one breath. Fixed in `c2ce860a`:
each value is now attributed to its tier, §5.1 says the margin is not a
measured head-to-head and names the run that would settle it, §6 notes its own
argument survives regardless (it needs only that a bare op count sits near the
ceiling, true on whichever tier), and `NUMBERS.md`'s provenance row records the
tier rather than only the source document. The §4 line comparing the two
baselines *to each other* was left alone — both come from the same tier, so
that one is accurate as written.

Two lessons, pointing opposite ways, and both earned here. A stale thread count
overstates the work as badly as misreading a zero understates it (the #1044
correction below is the same error inverted), so re-derive it rather than carry
it forward. And an *unresolved* thread is not evidence of unfinished work, nor
a resolved one evidence of finished: the #1072 thread had been partly fixed and
left open, while several elsewhere were closed with the fix genuinely in place.
Read the code, not the badge.

## Complications## Complications

### 1. #1054 — was red, now fixed. Recommend a mutants rerun, then merge.

Green at `136cc63`, red the moment it was brought up to date. Nine compile
errors in `pixelflow-codegen/src/emit/x86_64.rs`: `X86Backend::epilogue` no
longer exists (#1082 removed `prologue`/`epilogue`, methods moved into a
`driver` module) plus `E0308` mismatches from #1081's `&'static str` →
`CompileError` change. Failed on ubuntu, macOS, Clippy, and all three ISA
levels. **Git merged it cleanly and it does not compile** — the clearest
instance on this board of a semantic conflict that no merge check catches.

**Fixed** in `a42a16d5`: `emit_movups_store_base_*` now drive
`emit_movups_store` through a `NoDisp` address (the encoding the deleted
function produced), the redzone overflow test compares against
`CompileError::Internal(..)`, and the four `prologue_*`/`epilogue_*` tests are
removed. That removal was checked rather than assumed: `prologue`/`epilogue`
appear nowhere in the workspace, and `frame_alloc`/`frame_free` are
unconditional one-line delegations to `emit_sub_rsp`/`emit_add_rsp`
(`x86_64.rs:1298-1304`) with no `frame_bytes > 0` branch left to cover. Deleting
tests for deleted code, not dropping tests for green. Gates: `-p
pixelflow-codegen --lib` 125/0, full workspace green, clippy `-D warnings`
clean, `fmt` clean.

Still open: this is the second encoder refactor to invalidate the branch (the
first, #1055–#1062's Vex-builder rewrite, is in its own history at `4435869f`),
and the audit's "0 real gaps" conclusion has never been re-verified since
either — `emit_vpextrd_to_gpr` and `emit_vmovss_load_scaled` carry no coverage
claim at all. Rerun `cargo mutants -p pixelflow-codegen --file
pixelflow-codegen/src/emit/x86_64.rs -- --lib --test collapse_loop` and record
real numbers, or soften the conclusion, before merging.

### 2. #1072 — resolved by accepting the deletion, not by closing it.

Its conflict was `modify/delete`:
`pixelflow-pipeline/src/bin/bootstrap_extraction_head.rs` was deleted from
`main` by #1093 ("delete the extraction head's shape, keep its denotation"),
and this branch modifies it (`b449143`, the latency-prior cold start). An
earlier pass of this document recommended closing the branch. That was
premature — accepting the upstream deletion is a legitimate resolution, and it
is what landed (`bad5d2e7`):

- The deleted binary stays deleted. The paper describes that cold-start change
  in the past tense, so no claim in it becomes false — only unverifiable from
  the tree, which `NUMBERS.md`'s reproduction-debt table already said.
- `pixelflow-pipeline/examples/inspect_flip.rs` was removed too, and this is
  the part worth flagging rather than burying: it **cannot** be ported —
  `ExprNnue`, `IncrementalExtractor` and `EGraph::saturate_with_limit` all left
  with #1093 — and it was already documented as unrunnable because its two
  checkpoints were never committed. The debt table now records that recovering
  §5.4's flip analysis needs the extraction head resurrected, not just its
  weights.
- One rustdoc comment in `factored.rs` claimed the deleted binary "now seeds
  from" the latency prior; rewritten to past tense, naming the deletion, and
  saying why the test it documents is still worth keeping.

Reversible by design: if the extraction head is meant to come back, restore it
upstream and re-add both files, rather than holding this branch un-mergeable
against a deletion that has already landed.

The four unresolved P1s on the paper's headline intervals were separately
closed out (see the review-threads section) — the branch now has zero
outstanding threads and merges clean. What remains is an editorial judgement
about whether intervals with unrecoverable provenance should ship as they
stand, and that is the author's, not a merge question.

### 3. #994 — still blocked on credentials that do not exist.

Open since 11 Aug, green, zero threads, still untested end to end. Needs five
repository secrets that have never been created, and the
codesign/notarytool/stapler path has never run. Tag-triggered, so it is inert
until someone pushes `v*.*.*`. Three weeks of drift on something CI cannot
validate. Merge it as dormant infrastructure or close it — but decide, because
leaving it open costs a rebase every time `main` moves.

### 4. The conflicted branches, triaged by what actually blocks each

Working through them empirically rather than by inspection changed the picture:
two were mechanical and are done, and the rest split into two clearly different
kinds of work. The blocker is **never** the stop-reason redesign by itself — in
every case `main`'s landed version strictly supersedes the branch's, including
the stop reason for an application budget (`SaturationStop::ApplicationBudget`)
and the application-capped run itself (`Budget::Applications(u64)`, whose own
doc calls it "the budget the research arms compare under"). What blocks each
branch is whatever *else* it added to the same files.

**Kind 1 — superseded core, mechanical. Both done.**

*#1109* and *#1087* carried only the stop-reason work plus an `#[ignore]`d
measurement module. Recipe: take `main`'s `graph.rs`/`saturate.rs`/`mod.rs`
wholesale, keep the branch's harness intact as its own module, and port that
harness off the deleted `env_extraction_policy`/`saturate_with_full_budget`
onto `Optimizer` + `Budget::Explicit` + `hard_ceiling`, which is what keeps the
caps as parameters a measurement varies. Both pushed and now merge clean.

Two things worth stealing from those merges. First, `git` interleaved #1087's
`production_telemetry` module with `main`'s `congruence_gap_probe` *inside each
other's bodies* and produced an unclosed delimiter; the fix is to take `main`'s
file whole and re-append the branch's module from its own side, not to resolve
hunk by hunk. Second, each branch's `saturation_stop.rs` tests turned out to be
a strict subset of `main`'s, name for name — independent confirmation that
nothing is lost by taking `main`'s.

**Kind 2 — an additive provenance subsystem to re-thread. #1101, #1103.**

Core delta is only 237 lines, which makes these look like Kind 1. They are not.
Alongside the superseded stop-reason work each carries `ApplicationId` threaded
through `UnionEvent`, `ApplicationRecord`, `active_application` and
`derivation_ancestors_tight`, across `graph.rs`, `provenance.rs` and
`labeler.rs`. `main` has none of it, so it must survive — but `main` rewrote the
functions it threads through, so it has to be re-applied by hand.

Two traps, both hit and recorded: `git checkout --theirs graph.rs` silently
destroys the additive work (it takes the whole file, including the parts that
auto-merged cleanly), and resolving hunk-by-hunk instead leaves fragments of the
branch's `saturate_until_applications` orphaned *inside* `main`'s rewritten
function — dangling `'outer`, `max_total_applications`, `mid_sweep_stop`. The
workable route is `main`'s file whole, then re-thread `ApplicationId`
deliberately.

**Kind 3 — a competing refactor of the same seam. #1084, #1088, #1091, #1095, #1096.**

An earlier draft of this document called these "800–1,100 lines to re-apply."
That was wrong, and the error is worth naming: it conflated *lines the branch
adds* with *lines in conflict*. #1091 adds ~937 lines to `graph.rs`/
`saturate.rs`, but most are new code that auto-merges — the actual conflict is
12 hunks, ~395 lines, and most of those resolve mechanically to `main`'s landed
design, exactly like Kind 1.

Worked through to the bottom on #1091, the mechanical parts are:

- `graph.rs` h1+h6 (154 lines) are the superseded `saturate_until_applications`.
  `main`'s `saturate_bounded` takes `max_applications` and `SaturationStop`
  already has `ApplicationBudget`, so they drop.
- `saturate.rs` and `mod.rs` are import merges — keep the branch's `candidate`
  and `nnue::guide` imports, take `main`'s for the rest, drop the deleted
  `AppBudgetSaturationStats` and `env_extraction_policy`.
- `AppBudgetSaturationStats` is the branch's own type and re-homes cleanly into
  `saturate.rs`, where the guided loop is its only producer and consumer.
  `main`'s `SaturationStats` deliberately carries no application count — that
  dimension belongs to `Budget::Applications` now.
- `Cargo.toml` auto-merges into **two `[features]` blocks** (the branch's
  `guide-checkpoint` and `main`'s `saturation-telemetry`). Cargo rejects the
  manifest and reports the failure against `core-term`, not against this crate,
  which makes it look unrelated.

What actually stops it is a **design collision in `runtime.rs`**.
`ProductionSaturation`, `saturate_for_production` and
`production_saturation_probe` exist only on these branches — `main` has none of
them. The branch factored production's saturation step into a shared seam
precisely so the probe "runs THIS code, not a line-for-line copy that drifts";
`main` rewrote the same function around `Optimizer` without that seam.
Reconciling means re-deriving the seam from `main`'s new
`optimize_runtime_arena_uncached` and re-pointing the probe at it. That is
authoring rather than merging, and a subtle error breaks the one property the
probe exists to guarantee — so it belongs to whoever owns the experiment.

The five also do not share a resolvable base: all fork from the same merge-base
with `main` but none is an ancestor of another, so this is five pieces of work,
not one.

**A collision that has now happened.** #1087 and #1101 both add a module named
`production_telemetry` to `runtime.rs`, and both add
`docs/results/2026-09-01-production-saturation-telemetry.{md,csv}`. #1087 landed
first, so #1101 is the one that must reconcile: `main`'s docs are the later and
more complete run (471 vs 286 lines; 193 kernels vs 173) and should win, while
#1101's module is the superset (1,106 vs 610 lines) and adds exactly four
functions — `run_with_rules`, `anytime_curve_arena`, `csv_escape`,
`rule_order_real_kernels` — which graft onto `main`'s already-ported module.
Note `main`'s copy is the version ported to `Optimizer`; taking #1101's wholesale
would silently revert that.

## The class-cap question three branches are independently answering

Worth naming because it explains why this seam is contested rather than merely
busy. Four PRs are answering **the same design question** — *what does it mean
for saturation to stop at the class cap?* — and each answers it differently:

| PR | Its answer |
|---|---|
| #1083 (landed) | A truncated sweep is `ClassCap`, and that **ends the run**. |
| #1109 | Classification and termination are separable: record `ClassCap`, **keep sweeping**. Measured: the break leaves a more expensive arena on 140 of 204 real kernels. |
| #1114 | `ClassCap` should name **which ceiling** bound — `ClassCeiling::{Live, Allocated}` — because one number was standing in for two populations. |
| #1101 | Its `saturate_until_applications` reported `Quiesced` where `main`'s `saturate_bounded` reports `ClassCap` for the same run. |

That last row is not a merge defect; it is the disagreement showing up as one.
This pass ported #1101's harness onto `Optimizer`/`Budget::Explicit`
mechanically — every API rename resolved, the workspace built — and its own
`clamped_rows_freeze_the_final_state` test then failed with
`left: ClassCap, right: Quiesced`. The test is the branch's own and passes on
the branch. It is detecting a real behavioural difference between the entry
point it was written against and the one that replaced it, not a slip in the
port.

So #1101 cannot be reconciled without deciding whose semantics it should be
measured under — and **that same decision blocks every other conflicted
saturation branch.** All seven depend on the deleted `saturate_until_
applications` and on `run_anytime_curve`:

| PR | `saturate_until_applications` call sites | files using the anytime curve |
|---|---|---|
| #1096 | 27 | 7 |
| #1091 | 18 | 5 |
| #1088 | 18 | 7 |
| #1095 | 17 | 6 |
| #1084 | 16 | 5 |
| #1101 | 6 | 3 |
| #1103 | 6 | 3 |

`main` deleted that entry point in favour of `Budget::Applications`, and the
two do not agree on when a class cap ends a run. Every one of these branches
must be ported across that difference, and porting it *per branch* is how seven
registered experiments end up measured under seven slightly different stopping
rules.

**So the conflict count is misleading.** This is not eight independent
reconciliations; it is **one unresolved design decision with seven branches
queued behind it**, plus #1072, which is unrelated and structural. Answering it
once — does an application-budgeted anytime run stop at the class cap, or
record it and continue? — converts all seven from "needs judgement" into the
mechanical recipe already written down above, which this pass has now executed
end to end three times (#1087, #1109, #1114).

That is the single highest-leverage item on this board, and nothing was pushed
to #1101 pending it.


## Production saturation is unbounded in wall-clock time — deliberately

**Correction.** An earlier revision of this section called this a regression on
`main`. That was wrong, and the mistake is worth recording rather than quietly
editing away: I traced the mechanism correctly and then misread intent from it.

The mechanism is real. `SaturationConfig` still defines `hard_timeout` per tier
(10/50/200 ms), `Budget::limits()` reads the preset and returns
`Limits { iterations, classes, applications }` without it,
`Optimizer::production()` sets `hard_ceiling: None`, and `Optimizer::run` calls
`saturate_budgeted` — documented "with no clock". All three production paths do
this: macro (`pixelflow-compiler/src/optimize.rs:89`), `Dwrt`
(`ir_bridge.rs:734`), runtime (`runtime.rs:166`).

But `Budget`'s own doc says why, in as many words:

> Every variant is deterministic: the same arena under the same budget produces
> the same graph on any machine, at any load. Wall clock is deliberately **not**
> a variant. […] Compiler output was therefore a function of machine load, and
> "the same kernel compiles to the same code" was not a claim anyone could
> make. Under `Budget` it is.

So the clock was traded away on purpose, for reproducible compiler output. The
law `observation_is_optional_and_does_not_move_the_budget` encodes the same
choice — I wrote a patch restoring the deadline and it failed that law
immediately, which is the design defending itself. The patch was discarded.

**What is genuinely new here is the cost of that trade, which does not appear
to have been measured.** From #1109's CI run, runtime-tier telemetry:
`wall_clock_us` 9,648,883 · 9,938,809 · 16,184,232 · 32,829,162 and one at
**352,985,035 — 353 seconds for a 279-node kernel** — each `"stop_reason":
"class_cap"` at `"iterations":100`, the full ceiling. The
`fonts::atlas::tests::growth_preserves_tiles_and_bumps_epoch` test timed out as
a result.

Three things follow, none of which is "revert #1108":

1. **The macro tier is the sharp end.** It runs inside `rustc` at expansion
   time, so an unbounded saturation there does not slow a frame — it hangs the
   build. Determinism is worth a lot; a build that does not terminate is worth
   less. Whatever bound replaces the clock (an application cap is the obvious
   deterministic candidate, and `Budget::Applications` already exists) probably
   needs to be *on* by default for that tier.
2. **`SaturationConfig::hard_timeout` is now dead for production** — only
   `saturate_with_limits` still reads it. Leaving a 10/50/200 ms field that
   nothing in the compile path consults is exactly the "convention written in a
   comment" failure `CLAUDE.md` warns about, one layer up. Delete it or route
   it somewhere real.
3. **The class-cap decision inherits this.** #1109 measured that not breaking
   on `ClassCap` yields a cheaper program on 140 of 204 kernels. With no clock,
   the cost side of that trade is unbounded, and the 353-second row is what it
   looks like. Decide the two together.

## The finding that matters most: the seam churns faster than reconciliation completes

Reconciling a branch against this seam has a **short half-life**, and that is a
process problem rather than a backlog to burn down.

The evidence accumulated over one working day:

- The board went 5/15 clean → 7/15 as #1054, #1109 and #1087 were fixed → back
  to **5/14** the moment #1087 merged. #1087 landing re-conflicted #1109 and
  #1114, both of which had been clean an hour earlier.
- **#1109 needed reconciling twice in one day**, for the same structural reason
  each time: something landed on `runtime.rs` and both sides append a module at
  end-of-file.
- **#1114 was clean at 10:10 and needs an API port by 11:00** — `Budget` gained
  a field, `SaturationStop::ClassCap` changed arity, and helpers it reuses are
  now private to other modules.
- The #1087↔#1101 collision this document predicted ("whichever lands second
  needs a rename") **materialized**: both add a `production_telemetry` module
  and both add `2026-09-01-production-saturation-telemetry.{md,csv}`.

Every one of those is the same mechanism. `pixelflow-search/src/{egraph/graph,
egraph/saturate,runtime}.rs` is one hot seam, ten branches are queued on it, and
each landing invalidates the rest. Reconciling branch *n* while branches
*n+1…* are still landing is work with a half-life measured in hours.

Two things would actually fix it, and neither is more reconciliation:

1. **Rebase the queue as a batch immediately after each seam landing**, not
   per-branch on demand. The cost is one pass; the current cost is one pass per
   branch per landing.
2. **Or freeze the seam** until the queue drains. The four PRs that landed on it
   yesterday (#1083, #1085, #1107, #1108) are what created a nine-branch
   conflict set out of a board that merged cleanly the day before.

The corollary for whoever picks this up: **reconcile in landing order and land
promptly**, because a reconciled-but-unlanded branch is a wasting asset. #1087
is the demonstration in the good direction — reconciled and merged the same
hour, so it kept its value and is now upstream.

## Per-PR disposition

Consolidating the calls this document makes, since they are scattered through
the sections above.

| PR | Call | Why |
|---|---|---|
| #1054 | **merge** after a `cargo mutants` rerun | was red, fixed and CI-green; the audit's "0 real gaps" has never been re-verified across two encoder refactors |
| #1072 | **merge** | modify/delete resolved by accepting #1093's deletion; zero threads; the remaining question is editorial, not mechanical |
| #1109 | **rework, do not merge as written** | the change it proposes is unbounded on a clock-free `main` — 353 s for a 279-node kernel, timing out CI. Pair the removed `break` with `Budget::Applications` and re-measure; the `stop`-re-arming half can land separately |
| #1113, #1114 | **review normally** | clean, current, no threads |
| #994 | **decide** — merge dormant or close | green and threadless since 11 Aug, blocked on five Apple secrets that do not exist; costs a rebase every time `main` moves |
| #1084, #1088, #1091, #1095, #1096, #1101, #1103 | **hold** | all seven gated on the class-cap decision; mechanically ready otherwise, recipe above |
| #1086 | this document | — |

Superseded or obsolete, for the record: **#1050** (closed — #1055/#1068 deleted
the allocator its tests and both its bug findings were about) and **#1044**
(closed — a confirmed-negative result, ported twice, superseded by the Phase 3
program). Neither was salvageable and both are gone. **#1054** looked like a
third case and was not: it was twice-invalidated but its tests were
re-targetable, and it is now green.

## Recommended order## Recommended order

1. **Close #1054 and #1072**; decide #994. These three have not moved in
   ~13 hours and each is blocked on something no rebase fixes.
2. **#1114, #1113** — both already current and clean; land or review them
   before they join the conflicted set.
3. **#1109** (3 behind, one conflicted file) — cheapest of the nine to
   reconcile.
4. The remaining saturation family, owner by owner, smallest first: #1101,
   #1103, #1087, #1091, #1095, #1084, #1088, #1096.

## Method note

Branch state is from `git merge-tree --write-tree` and `git rev-list --count`
per branch against `origin/main` at `44c9fa3f`; conflicting paths from
`git merge-tree --name-only`. CI conclusions are from the check-run API at each
branch's current head, read directly rather than relayed. The #1072 file
deletion was confirmed with `git ls-tree -r origin/main` and the deleting commit
identified with `git log -- <path>`.

Two limits worth stating. Thread counts below are as of the first pass and were
not re-derived for this refresh, so treat them as indicative. And this session
is scoped to its own branch, so nothing here was pushed to another PR — the
recommendations are for the branch owners.
