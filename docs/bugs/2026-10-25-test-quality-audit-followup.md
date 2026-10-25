# Test quality control follow-up — 2026-10-25

Scope: scheduled continuation of the test-quality-audit series
(`docs/bugs/2026-10-10-test-quality-audit-followup.md`). Since that pass, 15
commits landed the "a bound is a value" series (A0–A6, #1437–#1450): fold
bounds may now be a per-call value, not only a literal, in the IR, the
e-graph, codegen and `kernel!`. Picked the series' new, self-contained files
over the 10-10 backlog's carried-forward items (`emit/mod.rs`, `aarch64.rs`,
`pixelflow-core/src/backend/arm.rs`, `executable.rs`'s `macos.rs`, the
core-term mock tests, `select.rs` line 223) because every one of those is
either an aarch64/macOS-only surface this x86_64 sandbox cannot exercise, or a
golden-hash fixture this pass chose not to build unverified against NEON
hardware — same reasoning as 10-10, still true today.

## Targets

New, untested-by-mutation files from the series:
`pixelflow-ir/src/passes/guard.rs` (410 lines — the pass that wraps a
value-bounded fold in its domain guard) and `pixelflow-ir/src/lit.rs` (161
lines — the typed-literal `Lit::Real`/`Lit::Mask`, from the same period's
"a constant is a typed literal" commit, #1431). Both already carry their own
`mod tests` (white-box, same relationship `register_file.rs`'s tests had to
their file in the 10-10 pass — reached only through the file's own
`pub(crate)` surface, not private internals a public accessor should front).
Also swept `desloppify/src/review/module.rs` (171 lines, new in the same
window's desloppify refactor, #1442/#1444/#1445) — no tests at all, reached
only as `pub(super)` plumbing from `plan.rs`/`identify.rs`/`synthesize.rs`.

## `../process/STYLE.md` compliance

Every test name in `guard.rs` and `lit.rs` already reads as an "it should"
sentence (`a_kernel_with_no_value_bound_moves_back_untouched`,
`a_true_mask_is_not_the_nan_with_its_bits`) — no renames needed. Both files'
tests reach only `ExprArena`'s push/node API and the pass's own entry point,
never a private field. `module.rs`'s functions are `pub(super)`, narrower
than `pub(crate)` — per this task's own instruction, not "committed to as
part of an API," so no white-box tests were added there; see below.

## Mutation testing: `cargo-mutants` v27.1.0

Installed fresh (consistent with every prior pass).

### `pixelflow-ir/src/passes/guard.rs` + `pixelflow-ir/src/lit.rs`

34 mutants, first sweep: **22 caught, 6 unviable, 4 missed, 2 timeouts.**

**`guard.rs:84`, real gap, closed.** `guard()`'s domain match has
`(Some(low), Some(high)) if low != high => { in_domain(low) & in_domain(high) }`,
falling through to the single-term arm when `low == high` (same `ExprId`,
e.g. `Bound::Value(u)` used for both ends) to avoid wrapping one test in a
redundant self-`BitAnd`. Deleting the guard survived because the existing
`two_value_bounds_are_each_in_the_domain` test's `low == high` half only
asserted `count_of(domain, Le) == 2` — and the arena interns by content, so
`in_domain(low)` and `in_domain(high)` on the same id produce the *same two*
`Le` nodes either way; the mutant's extra `BitAnd(x, x)` wrapper is invisible
to a node-count check that doesn't count `BitAnd`. Added
`a_value_bound_shared_by_lo_and_hi_is_not_joined_to_itself`, asserting
`count_of(domain, OpKind::BitAnd) == 2` (`in_domain`'s own two joins, no
outer one) — catches it.

**`guard.rs:67`, confirmed equivalent.** `has_a_value_bound` replaced with
`true` survives because `guard()` re-derives the same answer from the node's
real `lo`/`hi` regardless of what the (now-always-true) caller believed: the
`(None, None) => return folded` arm still fires for a literal-only fold, so
the observable output — root id, arena length — is identical either way.
Same shape as 10-10's line-199 finding (`select.rs`'s dead rematerializable
branch): the check is an optimization to skip a call, not a correctness
gate, so there is nothing to test that isn't testing dead code.

**`lit.rs:88`, both `eq` mutants, timeout — explained, and now also caught
directly.** Neither was closed by a new assertion catching it as a *failure*;
cargo-mutants still reports `TIMEOUT` for both. Tracked it down by hand:
applying `fn eq(&self, other: &Self) -> bool { false }` and running the
crate's full suite single-threaded (`cargo test -p pixelflow-ir --lib --
--test-threads=1`) shows the hang is in an unrelated test,
`passes::dwrt_tests::deep_chain_does_not_overflow_the_stack` — a deep chain's
stack-safety depends on the arena's interning collapsing structurally-equal
nodes back to one id; break `Eq` and every node in the chain looks unique,
so the thing the test holds bounded stops being bounded. `cargo-mutants`
classifies a mutant by whether the *whole test binary* exits in time, so one
hanging test anywhere in the crate pins the verdict to `TIMEOUT` no matter
how fast a different test fails — the same "timeout is concerning
independent of mutation testing" shape the 2026-07-20 pass flagged in
`actor-scheduler`'s `kubelet.rs`/`spsc.rs`. Added
`a_repeated_literal_interns_to_one_node_and_a_distinct_one_to_another`
anyway: it exercises the identical root cause (the same two `eq` mutants) but
fails in milliseconds with a legible message, rather than making whoever
runs this crate's tests next wait out a stack-safety test's hang to learn
the same thing. Confirmed by hand-applying the mutation and running `cargo
test -p pixelflow-ir --lib lit::` alone: `FAILED` immediately, no timeout.

**`lit.rs:97`, `Hash::hash` → no-op, missed, not fixed.** A constant hash
for every `Lit` degrades every hash-keyed interning table (`hash-memo`
feature) to one bucket — slower, not wrong, since `Eq` still distinguishes
within the bucket. Same "Debug/Display `fmt`, low priority" bucket the
2026-07-20 pass used for non-correctness-affecting mutants: not fixed.

**`lit.rs:104`, `PartialOrd::partial_cmp` → `None`, missed, confirmed
unreachable.** `impl Ord for Lit` requires `PartialOrd` as a supertrait but
its own `cmp` doesn't call `partial_cmp` (it recomputes `self.key().cmp(...)`
directly), and nothing in the workspace calls `<`/`<=`/`.partial_cmp()` on a
`Lit` — checked with `rg` across every crate. `Dag`'s alternate
(non-`hash-memo`) interning backend needs `Ord` on its key type (`impl<T:
Ord + Clone> Key for T {}`, `dag.rs:448`) but that's `Ord::cmp`, independent
of this impl. `partial_cmp` exists only to satisfy the trait bound, is never
evaluated, and there is nothing to test that isn't testing an unreachable
line. Worth a second look only if something starts calling `Lit`'s ordering
operators directly.

### `desloppify/src/review/module.rs`

30 mutants: **26 caught, 3 unviable, 1 missed** —
`shape`'s budget check, `lines + length > budget` mutated to
`lines * length > budget`. Real: for two files whose sizes multiply to
something far under `budget` but sum to something over it (the common case —
most files are nowhere near `budget` lines alone), the product check never
omits anything, so a module's outline would run arbitrarily far past its
line budget. Not fixed here: `module.rs`'s functions are `pub(super)`, not
`pub(crate)` — plumbing reached only from `plan.rs`/`identify.rs`/
`synthesize.rs` within the same module, with no public accessor and nothing
in this crate's intent documenting it as a key abstraction the way
`register_file.rs`'s `RegisterFile` or `guard.rs`'s pass are documented.
Per this task's own rule against testing functions not committed to as part
of an API, a white-box test was not added. The sanctioned fix is through the
crate's actual public surface — `desloppify::review::{Session, plan}`,
exercised end-to-end by the golden-transcript test
(`desloppify/tests/transcript.rs`) — which means adding a new fixture layout
to that test's one shared tree (a module whose shown-lines total lands
exactly on `MAX_SHAPE_LINES` before a trailing small file, so `+` and `*`
disagree) and regenerating `tests/golden/transcript.txt`
(`TRANSCRIPT_OUT=...`). Left to a pass with room to review the regenerated
golden by hand rather than trusting a mechanical diff on a 171-line budget
function in a tool that isn't part of the rendering pipeline.

## Verified

- `cargo test -p pixelflow-ir --lib`: pass (312 tests, including the two new
  ones).
- `cargo test --workspace --lib --bins`: pass.
- `cargo fmt --all -- --check`: clean (after `cargo fmt -p pixelflow-ir`).
- `cargo clippy -p pixelflow-ir --all-targets --all-features -- -D warnings`:
  3 pre-existing errors (`arena.rs:1142`, `dag.rs:226`, `dag.rs:579`, all
  `must_use_candidate`), identical with and without this change (confirmed
  by `git stash`) — a toolchain-drift finding, not something this change
  introduced or fixes; `rust-toolchain.toml` pins `channel = "stable"` with
  no date, so a newer `stable` than whatever CI last ran under can turn on a
  lint a prior green CI run never saw. Out of scope here; worth its own pass
  if it also reproduces on a clean checkout of `main`.
- `cargo mutants -p pixelflow-ir -f pixelflow-ir/src/passes/guard.rs -f pixelflow-ir/src/lit.rs`:
  34 mutants, 23 caught, 6 unviable, 2 timeouts (both explained above), 3
  missed (97 and 104, confirmed non-correctness/unreachable above; 67,
  confirmed equivalent above) — down from 4 missed before the `guard.rs:84`
  fix.
- `cargo mutants -p desloppify -f desloppify/src/review/module.rs`: 30
  mutants, 26 caught, 3 unviable, 1 missed (line 159, real, deferred above).

## Recommended next steps (not done here)

Carried forward from 10-10 (all still blocked on hardware/host this sandbox
doesn't have), plus:

1. `pixelflow-codegen/src/emit/mod.rs` (7,289+ lines) — untouched by any
   pass.
2. `pixelflow-codegen/src/emit/aarch64.rs`, `pixelflow-core/src/backend/arm.rs`,
   `executable.rs`'s `macos.rs` — need aarch64/macOS hosts.
3. `core-term/tests/message_cuj_tests.rs` and the `ParserActor`/
   `TerminalApp` section of `core-term/tests/actor_roundtrip_tests.rs` — test
   mock reimplementations, not `core_term`'s real types.
4. `select.rs` line 223 (`&&` vs `||` across a scope's guards) — needs a
   two-`If`-one-scope fixture and new golden hashes verified against real
   AVX-512/NEON hardware.
5. **New**: `desloppify/src/review/module.rs`'s `shape` budget check (`+`
   vs `*`, confirmed real above) — needs a new fixture layout in
   `desloppify/tests/transcript.rs`'s shared tree and a regenerated
   `tests/golden/transcript.txt`, reviewed by hand.
6. **New**: the toolchain-drift clippy failure on `arena.rs:1142`,
   `dag.rs:226`, `dag.rs:579` (`must_use_candidate`, reproduces on a clean
   checkout with this sandbox's `stable` toolchain) — worth checking whether
   it also reproduces in CI's own runner, and adding the three `#[must_use]`
   attributes if so.
