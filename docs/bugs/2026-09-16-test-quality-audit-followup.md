# Test quality control follow-up — 2026-09-16

Scope: scheduled continuation of the test-quality-audit series
(`docs/bugs/2026-09-10-test-quality-audit-followup.md`'s backlog item 1).
Of that item's two untouched files, picked `pixelflow-codegen/src/emit/regalloc.rs`
(2,985 lines, 364 mutants) over `mod.rs` (5,460 lines, 275 mutants) — smaller by
mutant count despite the larger line count, and it is one coherent algorithm
(`RegisterAllocator`/`LinearScan`, this codebase's own worked example of the
"new implementation of an existing category → trait first" rule in CLAUDE.md),
where `mod.rs` is orchestration glue. `aarch64.rs` (737 mutants, runtime-untestable
from this x86_64 sandbox at the execution level) and `pixelflow-core/src/backend/arm.rs`
remain carried forward, same as every prior pass.

Also did a supplementary STYLE.md naming/scope pass over `core-term`'s test
suite, prompted by this run's task description explicitly calling out test
naming and scope discipline. `core-term` had its own dedicated
mutation-testing pass three days ago (`caf99b7`, #1259 — 1374 mutants, not
part of this doc series but same methodology), so it was not re-swept here.

## Rebased onto the H6 allocator — read the counts below as history

This pass was measured against `regalloc.rs` as it stood at `93d48c8`
(2,985 lines). Between then and this branch's merge of `main`, #1283 (H6
step 5, "the emitter executes the lattice's folds") rewrote the allocator's
scope model. **The sweep numbers in this document describe a file that no
longer exists in that form, and must not be read as a current statement
about `regalloc.rs`'s mutation coverage.** A fresh sweep against the H6
allocator is carried forward as a backlog item; it is the only thing that
can say what today's gaps are.

What the merge required, and what it says about each test:

- **Three tests deleted as obsolete.** `an_input_register_inside_the_pool_is_refused`,
  `fixed_registers_disjoint_from_scratch_and_inputs_pass_checked` and
  `a_fixed_register_aliasing_an_input_is_refused` all exercised
  `RegisterFile::inputs`, which #1283 deleted outright: the collapse ABI is
  `fn(ctx, out, pitch)` and passes no vectors, so there are no coordinate
  input registers left to collide with. The behaviour is gone, not untested.
- **Four fixtures ported to the scope model.** `a_root_the_body_never_reads_is_never_carried`
  (now `a_root_the_fold_never_reads_is_never_carried`),
  `fold_opening_at_matches_the_position_as_well_as_the_parent`, and the two
  `destination`/`result register` tests were written against
  `ScopedSchedule.regions` and `Scope::Region`, which became `body` plus
  `folds`. The first gained a control root in the same allocation, so the
  zero-use filter is pinned against a root the same budget *does* carry.
- **Seventeen `Var`-as-coordinate leaves rewritten to `leaf()`.** A `Var`
  reaching an allocation now names a fold binder and nothing else; the
  allocator panics on one no enclosing fold binds. These fixtures predate
  that rule and were building schedules it refuses.
- **Four pressure fixtures rebuilt on `RegisterFile::MIN_SCRATCH`.** They
  hardcoded "seven fillers, exactly the pool", which held only while the
  leaf sat in an input register outside the pool. Post-H6 the leaf occupies
  a pool register of its own, so the set that fills the pool is the leaf
  plus `MIN_SCRATCH - 1` fillers. Their "spacer with no pool interaction"
  also had to change: every value-defining op now needs a register, so the
  spacer is a `Seq`, which defines no value and takes none.

Because a rebuilt pressure fixture can very easily pass for the wrong
reason, each was re-verified by mutation rather than by inspection — the
same bar the rest of this document is written to. Against the restored
tree, all 259 `pixelflow-codegen` lib tests pass, and:

| mutation at `regalloc.rs`'s keep contest | caught by |
|---|---|
| `new_rank > rank(occupant)` → `>=` | `a_tie_with_the_evicted_occupant_does_not_keep_the_new_definition` (**only** this test) |
| `new_rank > rank(occupant)` → `<` | the two `destination` tests, the constant test, and `constants_are_rematerialized_rather_than_spilled` |
| the rematerialized-constant exclusion disabled | `a_constant_that_loses_its_keep_contest_is_never_given_a_register` (**only** this test) |
| the demotion queue's `!keeps` guard disabled | `a_destination_that_forces_an_eviction_but_reads_later_than_the_occupant_is_spilled_next` and the tie test |

Each rebuilt fixture is still the unique detector of the mutant it was
written for, which is what makes the port a port rather than a rewrite.

## Mutation testing: `cargo-mutants` v27.1.0

Not present in this environment (consistent with every prior pass) —
installed via `cargo install cargo-mutants --locked`.

### `pixelflow-codegen/src/emit/regalloc.rs`

First sweep: **364 mutants, 174 caught, 83 missed, 99 unviable, 8 timeouts.**

The 99 unviable mutants are cargo-mutants' own bucket (fails to compile —
mostly `const fn` bodies where a mutation produces a non-const-evaluable
expression) and needed no action. The 8 timeouts are mutations to a loop
counter's `+=` (turning it into `*=` or a no-op), which make the mutated
build hang rather than fail a specific assertion — a hang is a very
observable regression in its own right (nobody ships a register allocator
that doesn't terminate), so these were left alone rather than chased into
becoming clean "caught" results.

The 83 missed fell into groups, closed over several rounds with re-sweeps
between them to catch tests that looked right but didn't actually distinguish
the mutation (see the methodology note below):

- **`RegSet`/`GprSet`/`MaskSet`** (41 mutants) — the file's three bitset
  types (vector registers, GPRs, AVX-512 mask registers) had **no direct unit
  test at all**; every existing test reached `of`/`union`/`without`/`contains`/
  `len`/`is_empty`/`take`/`iter` only incidentally through a full allocation.
  Added one test per operation. A second sweep found four of these tests
  didn't actually kill their target: a union test using disjoint operands
  can't distinguish `|` from `^` (they agree when nothing overlaps), and a
  `contains` boundary test one below a set's *chosen* highest member can't
  distinguish `<` from `<=` unless the check is at the type's own limit (32
  registers, 8 for `MaskSet`) — the value an `<=` mutant lets through into an
  out-of-range shift. Fixed both classes.
- **`RegisterFile::checked`'s fixed-register alias loop** (4 mutants) — this
  inner loop (checking each `fixed` register against every coordinate input
  for aliasing) was reached by exactly one existing test, and that test's
  fixed register already fails the *earlier* scratch-overlap assert, so the
  alias loop's body and both loop counters' `+=` never actually ran under any
  test. Added a clean-pass test and an alias-triggering `should_panic` case
  that can only fail via this specific check.
- **`record`/`guarded_arms`/`Reservations`/`Pass::rank`/`Pass::place`** (12
  mutants) — these scan-internals were exercised only indirectly through
  whole-allocation tests, never directly. Added direct unit tests against
  each (`record`, `guarded_arms`, `Pass::new`/`rank`/`place` are all reachable
  without running a full `allocate_nest`).
- **`select_guards`, and the guarded-arm keep/revert boundary** (2 mutants) —
  `Allocation::select_guards` had no test confirming it returns the real
  analysis rather than a stand-in; a mutant replacing its body with an empty
  `Vec` survived. Also pinned two edge cases in how a spilled operand read
  inside a guarded arm decides whether re-promoting it is worth a revert at
  the arm's end: a read landing *exactly* at the arm's end (too late to be
  worth it) and a value defined *at* the arm's own start (arm-internal from
  birth, so never worth a revert regardless).
- **The destination keep-or-demote contest** (6 mutants) — a definition that
  forces an eviction to make room for itself doesn't automatically keep the
  freed register; it has to win a distance contest against the value it
  evicted, and that contest was previously only tested obliquely. Added
  direct cases for: a constant losing the contest outright (never given a
  register at all, not even a spill-eventually), a destination that reads
  later than the occupant it evicted (spilled), one that reads sooner
  (kept), and an exact tie (goes to the occupant — `keeps` is strict `>`).
- **`fold_opening_at`, root-carrying and result-register reservation** (4
  mutants) — a root the body never reads is never a carry candidate
  regardless of budget; `fold_opening_at` must match both the parent scope
  *and* the position, not either alone; and a root already resident (via its
  own destination, or via being carried) needs no separate result-register
  reservation even though other parts of that three-way `&&` guard hold.

Final sweep: **364 mutants, 248 caught, 7 missed, 99 unviable, 10 timeouts.**

The remaining 7 missed are genuine equivalent mutants, verified by
construction (and, for the harder ones, by manually re-applying the mutation
and confirming no reachable test input distinguishes it) rather than by
failure to find a test:

- `RegisterFile::capped:494:49` (`< → <=`) — `if n < M {M} else {n}` and
  `if n <= M {M} else {n}` agree at every input, including the boundary
  `n == M` (both yield `M`).
- `LinearScan::scan:2313:20` (`delete !` in a guard-mask hard-exclusion) — the
  same mask is independently recorded as "read at this position" via the
  `sites[]` that seed `Pass::new`'s ranking, which already gives it
  distance-0 regardless of this separate exclusion.
- `LinearScan::scan:2432:28` (`< → <=`, arm-end bound) — `end` is always
  `last_exclusive_index + 1 ≤ select_idx < dag.len()` by construction of
  `guards::range()`, so the `<=` arm is unreachable.
- `LinearScan::scan:2492:39` (`== → !=`, `read_here` merge) — same root cause
  as 2313: anything read by the *current* instruction is already distance-0
  via `Pass::new`'s construction, making this tier redundant for every
  reachable case.
- `LinearScan::scan:2561:44/2561:40` (three mutants, the schedule's literal
  last instruction's own eviction contest) — reaching this boundary needs the
  last instruction to lose a multi-candidate contest, which can't happen:
  anything not its operand has already expired by then, and an instruction's
  operand count (≤ 3) never fills a register file's floor-enforced pool
  (≥ `MIN_SCRATCH` = 7), so the destination always finds a free slot first.

### A methodology note worth keeping

Two "destination contest" tests initially passed but silently didn't catch
their target mutant, both for the same reason: checking a value's placement
*one instruction after* its definition is unsound when that next instruction
needs its own destination, since an unrelated eviction there can spill the
value regardless of the mutation under test. The fix was to insert an inert
spacer instruction and check placement there instead. This was only caught by
manually re-applying each mutation and re-running the targeted test rather
than trusting "this assertion should distinguish it" reasoning — worth
treating any "one step later" placement assertion in this file with
suspicion, and worth doing generally: a test that passes against both the
real code and the mutant it was written for isn't testing anything.

## STYLE.md compliance: test naming and scope (`core-term`)

Ten test names across `keys.rs`, `io/pty_tests.rs`, and `term/tests.rs`
described a scenario or the mechanism under test rather than the expected
outcome (e.g. `pty_spawn_successful`, `initiate_copy_no_selection`,
`lf_at_bottom_of_partial_scrolling_region_no_origin_mode`). Renamed each to a
complete "it should ..." sentence (e.g.
`spawned_pty_relays_the_childs_stdout_to_the_master_fd`,
`initiate_copy_returns_none_when_no_selection_is_active`,
`lf_scrolls_the_partial_scrolling_region_when_the_cursor_is_at_its_bottom_with_origin_mode_off`);
no test bodies or assertions changed.

Also reviewed `term/screen.rs`'s embedded `mod tests` for reaching past
`Screen`'s own encapsulation (poking private `scrollback_limit`/`scroll_top`/
`scroll_bot`/`tabs` fields directly) — initially flagged as a scope
violation, but on inspection every one of these pokes is either test *setup*
(building a scenario the public constructor can't express, e.g. a caller-chosen
scrollback limit) or an assertion alongside a real call to the method under
test (`set_tabstop`/`clear_tabstops`/`get_next_tabstop` are all actually
invoked; the private `tabs` vector is read only to set up or confirm state,
never used *instead of* calling the real method). This is the same pattern
`regalloc.rs`'s own tests use throughout (`def`/`alloc` helpers poke internal
fixtures directly), and it's normal same-file white-box unit testing, not a
violation of "test the public API" — left unchanged.

Two files were flagged as a genuine, larger scope issue and are **not fixed
here**, left as backlog: `core-term/tests/message_cuj_tests.rs` (all 15
tests) and the `ParserActor`/`TerminalApp` half of
`core-term/tests/actor_roundtrip_tests.rs` (13 tests) exercise hand-rolled
mock reimplementations of `core_term`'s parser/key-translator/actor types
rather than the real ones — both files say so themselves in their own header
comments. They currently test `actor_scheduler`'s generic channel mechanics
under a `core-term`-flavored costume, not anything about `core-term`'s actual
contract. Fixing this properly means either rewriting ~28 tests against the
real `AnsiProcessor`/`TerminalApp`/actor types or deleting the file and
relying on `actor-scheduler`'s own suite plus `ansi_parser_message_tests.rs`
(which already does correctly exercise the real parser) — too large a change
to make unreviewed in a test-quality pass; see "Recommended next steps".

## Verified

- `cargo test -p pixelflow-codegen --lib`: 262 passed, 0 failed.
- `cargo test -p pixelflow-codegen` (all targets incl. doctests): pass.
- `cargo clippy -p pixelflow-codegen --lib --tests -- -D warnings`: clean.
- `cargo fmt -p pixelflow-codegen -- --check`: clean.
- `cargo test -p core-term --lib`: 576 passed, 0 failed (renamed tests
  confirmed still passing under their new names).
- `cargo test --workspace`: pass.
- `cargo mutants -p pixelflow-codegen -f pixelflow-codegen/src/emit/regalloc.rs`:
  364 mutants, 7 missed (all confirmed equivalent), 248 caught, 99 unviable,
  10 timeouts.

## Recommended next steps (not done here)

Backlog carried forward from 2026-09-10, minus the item closed above:

1. `pixelflow-codegen/src/emit/mod.rs` (5,460 lines, 275 mutants) —
   untouched by any pass.
2. `pixelflow-codegen/src/emit/aarch64.rs` (2,601 lines, 737 mutants) —
   untestable at the runtime-execution level from this x86_64 sandbox, though
   its encoding-only assertions could still be mutation-tested.
3. `pixelflow-core/src/backend/arm.rs`'s NEON impls — still untestable from
   every x86_64 sandbox this series has run in; needs an aarch64 host.
4. `pixelflow-codegen/src/emit/executable.rs`'s `macos.rs` submodule —
   untouched (no macOS host in this sandbox).
5. **New**: `core-term/tests/message_cuj_tests.rs` (15 tests) and the
   `ParserActor`/`TerminalApp` section of `core-term/tests/actor_roundtrip_tests.rs`
   (13 tests) test mock reimplementations, not `core_term`'s real types — see
   above. Needs a real design decision (rewrite against real types, or
   delete and confirm `actor-scheduler`'s own suite plus
   `ansi_parser_message_tests.rs` already cover what's real here), not a
   mechanical fix.
