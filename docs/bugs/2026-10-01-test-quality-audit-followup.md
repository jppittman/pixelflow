# Test quality control follow-up — 2026-10-01

Scope: scheduled continuation of the test-quality-audit series
(`docs/bugs/2026-09-16-test-quality-audit-followup.md`'s backlog item 1).
Re-verified the backlog against the live tree before picking a target:
`pixelflow-codegen/src/emit/mod.rs` had grown to 8,266 lines (from 5,460 at
the last check) but carries only **332 mutants** by `cargo mutants --list`,
against `aarch64.rs`'s **779** (grown from 737) despite `aarch64.rs` being
the smaller file by line count — mutant count, not line count, is this
series' own tractability measure, and by that measure `mod.rs` is the
cheaper sweep, so it was picked over `aarch64.rs` again this pass.
`pixelflow-core/src/backend/arm.rs`, carried forward by every prior pass,
**no longer exists**: the module was deleted in `3d01892` (#1286,
docs/BACKLOG.md's C7) as part of "the ISA is decided at startup" — core lost
its `backend` module entirely, so this backlog item is obsolete rather than
done. `executable/macos.rs` is still blocked (no macOS host in this sandbox)
and carried forward again.

Also did a STYLE.md naming/scope pass over every core-term test added since
`250822e` (`test: mutation-testing gap closure for the register allocator,
plus a core-term naming pass (#1276)`) — this branch's shallow clone does not
reach the `caf99b7`/#1259 commit the task's own backlog note named, but
`250822e` is this series' most recent core-term-touching commit under a
different SHA, so it is the right baseline to diff against.

## STYLE.md compliance: test naming

### `pixelflow-codegen/src/emit/mod.rs`

Sixteen existing test names described a scenario or mechanism rather than an
outcome: `resolve_binary_no_spills`, `resolve_binary_left_spilled`,
`resolve_binary_both_spilled`, `resolve_muladd_fmla_path`,
`resolve_muladd_decomposed_both_ab_spilled`,
`resolve_muladd_decomposed_all_three_spilled`, `resolve_var_is_nop`,
`resolve_const`, `arena_to_schedule_filters_unreachable`,
`arena_compile_simple`, `arena_compile_with_constant`,
`arena_compile_with_spills`, `transcendental_in_expression`,
`sched_no_spill_is_correct`, `sched_spills_and_is_correct`,
`sched_select_guards`. Renamed each to a complete "it should ..." sentence,
e.g. `resolve_binary_no_spills` →
`resolving_a_binary_op_with_no_spilled_operands_reloads_nothing`,
`arena_compile_with_spills` →
`compiling_a_wide_arena_expression_under_a_tight_register_budget_spills_and_still_computes_correctly`.
No test body or assertion changed; every renamed test re-verified passing
under its new name. Names like `avx2_backend_covers_required_ops`,
`one_spill_takes_one_slot` and `dwrt_compiles_to_analytic_derivative` were
left alone: each already states the outcome as its own noun phrase, the same
pattern prior passes accepted (`push_reduce_should_panic_when_...`-style
variance is explicitly licensed by STYLE.md's own second example).

### `core-term`

Every test added since `250822e` was already compliant
(`a_missing_file_is_the_defaults`,
`a_mouse_drag_selects_and_becomes_the_primary_selection`,
`decoding_refuses_characters_outside_the_alphabet`, etc.) except two files:

- `core-term/src/term/emulator/osc_handler.rs`'s `osc_52_ignores_garbage`
  named the mechanism, not the outcome. Renamed to
  `osc_52_with_a_malformed_or_non_utf8_payload_produces_no_action`.
- `core-term/tests/print_text_equivalence.rs`'s eleven scenario-table tests
  (`short_text`, `text_that_exactly_fills_a_line`,
  `text_that_wraps_and_scrolls`, `text_starting_mid_line`,
  `text_with_autowrap_off`, `text_overwriting_wide_characters`,
  `text_mixed_with_wide_and_combining_characters`,
  `text_under_a_non_identity_charset`,
  `text_inside_a_scroll_region_with_origin_mode`, `text_with_attributes`,
  `text_on_the_alternate_screen`) all named their input scenario and left
  unstated the one outcome the whole file exists to pin — that `print_text`
  run as whole text runs equals the same bytes fed one character at a time.
  Renamed each to append that outcome, e.g. `short_text` →
  `short_text_prints_the_same_whether_fed_as_a_run_or_one_character_at_a_time`.

No scope violation found: every new or renamed test reaches only its own
file's public/crate-visible surface (`resolve_operands`, `ScheduledOp`,
`Label`, `Assembly`, `Loc`, `Binding` in `mod.rs`'s own `emit` module;
`load_config_from`, `encode`/`decode` in `core-term`'s own modules) — the
same "internal, but central to the crate's own design" relationship the
09-16 pass established for `regalloc.rs`'s `def`/`alloc` fixtures.

## Mutation testing: `cargo-mutants` v27.1.0 (`pixelflow-codegen/src/emit/mod.rs`)

First sweep: **332 mutants, 217 caught, 29 missed, 84 unviable, 2 timeouts.**

The 84 unviable needed no action (cargo-mutants' own "does not compile"
bucket). The 29 missed, closed over several rounds with re-sweeps between
them (a test that looked right but didn't actually distinguish the mutation
was caught twice this way — see the methodology note):

- **`Label`, `Assembly`/`AsmProgram`, `Loc`, `Binding`, `PtrReg`** (15
  mutants across `as_str`, `Display`, `Debug`, `with_capacity`,
  `AsmProgram`'s own `AsmInsn::emit_into`, `SourceOperand`/`StoreTarget` for
  both `Loc` and `Binding`, and `PtrReg::raw`) — every full-compile test
  reaches these only incidentally, through whatever instructions a schedule
  happens to need, so none of them pinned what the types themselves promise.
  Added a new `assembler_core` test module (18 tests) exercising each
  directly: a label's name round-trips through `as_str`/`Display`/`Debug`
  and panics past `Label::CAPACITY`; `Assembly::with_capacity` actually
  reserves; a hand-assembled two-pass program (a `jmp` to a label bound after
  a skipped `ret`) resolves its `rel32` from the right offset, including
  through `AsmProgram`'s own `AsmInsn::emit_into` rather than only its
  `assemble` method; `Loc`/`Binding`'s `StoreTarget`/`SourceOperand` impls
  agree with their own `storage`/`as_loc`/`as_slot` for every variant,
  including the trait-dispatched `source_storage` as a route distinct from
  the inherent `as_storage` it forwards to.
- **`operand_sources`'s `MulAdd` special case** (3 mutants: deleting the
  `!resident[0] && !resident[1]` guarded arm, replacing its guard with
  `true`, and `&&` → `||`) and **`resolve_operands`'s own, separate
  `a_spilled && b_spilled` decompose check** (1 mutant, `&&` → `||`) — the
  three existing `resolve_muladd_*` tests only ever had *both* multiplicands
  resident or *both* spilled, so the "exactly one spilled" case these
  mutants change was never built. Added two tests:
  `resolving_a_muladd_with_only_c_spilled_reloads_it_straight_into_the_destination`
  (`a`/`b` resident, `c` spilled — pins the FMLA path's `dst`-direct reload)
  and
  `resolving_a_muladd_with_only_a_spilled_still_fuses_and_reloads_a_to_its_own_register`
  (`a` spilled alone — pins that the decomposed path needs *both*
  multiplicands spilled, not either one, and that `a`'s reload goes to its
  own reservation rather than colliding with `c`'s `setup_mov` into `dst`).
- **`IsaBackend::test_ge`'s default body** (1 mutant, replaced with `()`) —
  the fold loop's trip test on every backend but AVX-512, which overrides
  it; this host's own tier *is* AVX-512, so no test built through
  `EmitCtx::compile` ever reaches the default. Added
  `the_default_trip_test_runs_a_surviving_reduce_to_the_right_answer_on_avx2`,
  which drives a `Reduce`-bearing schedule straight through
  `avx2::driver::Avx2Backend` and `compile_via_backend`
  (`every_backend_emits_from_this_host`'s own technique for reaching one
  backend regardless of host tier) and checks the summed result.
- **Two sibling `ScheduledOp::Guard`s' own slots** (2 of the 6
  `guard_slot`/`guard_slot_base` mutants, the ones indexing by the guard's
  own `k`) — a single guard's slot arithmetic degenerates to "the one slot"
  under any base/stride, so no existing guard fixture could tell `guard_slot
  = base + k * vector_bytes` from a broken form unless it had *two* guards
  whose slots must land at different addresses. Added
  `two_sibling_guards_in_one_scope_each_land_in_their_own_slot`: two
  independent `push_guard`s (not two raw `Select`s — the first attempt at
  this fixture used `OpKind::Select` and never produced a `ScheduledOp::Guard`
  at all, see the methodology note) with four distinct constants as their
  arms, summed and checked against all four flag combinations. It kills the
  `guard_slot` multiply mutants (`k as u32 * vector_bytes` → `/`/`+`); it
  does *not* kill the `guard_slot_base`/`park_base` mutants in the same
  cluster, because this fixture's pre-guard frame size `m` is 0, which makes
  `m + X` and `m * X` agree — see "Recommended next steps" for what a
  fixture that does catch those needs.

Final sweep: **332 mutants, 227 caught, 19 missed, 83 unviable, 3 timeouts.**

Two of the three timeouts are new, both genuine: `IsaBackend::test_ge`
replaced with `()` now times out (confirmed by hand — the trip test's
comparison never runs, so `branch_if_arm_is_dead` reads whatever garbage was
in the register and the loop does not reliably exit), and the pre-existing
`branch_starts`/`fold_slot` timeouts are unchanged. Consistent with every
prior pass's convention, a hang is left as a timeout rather than chased into
a clean "caught" result — it is itself an observable regression (nobody
ships a compiler whose emitted loop sometimes does not terminate).

The remaining 19 missed are **real, not equivalent**, except two verified
equivalent by construction (below) — this pass runs out of budget before
closing them, and they are carried forward as a scoped, named backlog item
rather than claimed closed:

- `pixelflow-codegen/src/emit/mod.rs:1816:22` (`!= → ==`, `ptr_into`'s
  `q != p`) and `:1971:28` (`!= → ==`, the schedule's "moves" reconciliation
  `src != r`) — **genuine equivalent mutants**, verified by construction.
  `RegisterFile`'s only eviction path (`LinearScan::split_out` →
  `out_of_register`) demotes a value to `Spilled` or `Remat`, never to a
  *different* register of the same class — there is no code path that
  records a placement transition from one `Reg`/`Ptr` straight to another.
  Every backend's `emit_resolve` (and `ptr_into`'s own `Slot` arm) returns
  its own `target` argument exactly when resolving a `Spilled`/`Remat`
  value. Chained, a value's binding can never read as "resident in a
  different register than this point expects" at either of these two call
  sites, so both conditions are false on every schedule the allocator can
  build. Confirmed empirically too, not just by the argument: both were
  instrumented with an `eprintln!` and the whole 310-test suite run against
  it — neither fired once.
- `pixelflow-codegen/src/emit/mod.rs:1721:28` (`< → <=`, whether a guarded
  arm's range ends exactly at the schedule's own length) — real: reaching it
  needs a guarded `Select` whose arm is the very last entry of a *fold's*
  schedule (a `Reduce`'s body schedule has no trailing `Write` the way a
  kernel's top level does, so it is the one shape where an arm can run all
  the way to `sched_len`). No current fixture nests a guard inside a fold's
  body at all.
- `:1856:36`, `:1869:36` (`delete !`, the `Reg`/`Ptr` arms of the
  preloaded-value head-reconciliation's `.find(|at| !in_register(at))`) and
  `:1923:20` (`delete !`, `hand_off`'s `!resident_throughout` for the `Ptr`
  class) — real: all three need a value an enclosing scope parked whose
  register assignment actually differs between a fold's tail and the next
  iteration's head, i.e. a multi-iteration loop under enough pressure that a
  *parked* root's placement is not simply "always in the same register" or
  "always spilled". The nested-fold pressure fixtures in this file
  (`a_folds_spill_slots_do_not_alias_its_parents` and kin) build pressure
  inside the fold, not on a value crossing its own back edge.
- `:2950:38`, `:2950:43` (`schedule_variance`'s `*idx < Variance::VARIABLES`
  guard, replaced with `true` and with `<=`) — real only at
  `idx == Variance::VARIABLES` (64): a `Var` naming a binder slot ≥ 64. No
  fixture in this file nests folds anywhere near 64 deep, and building one
  on purpose to kill two mutants is likely not worth the frame-layout
  machinery it would exercise along the way; flagged rather than attempted.
- `:3096:5` (`pending_reads` replaced wholesale with `Default::default()`)
  — real: `FoldReads` only prices an arm's *fold* cost, so a scope with a
  guarded `Select` but no sibling fold never calls into the part this
  mutant changes meaningfully. Needs a fixture with both a surviving
  `Reduce` and a guarded `Select` in the *same* scope, priced against each
  other.
- `:3998:36` (`extract_guards`' `next_id` bump, `+` → `*`) — real: this
  mutant degenerates to a no-op only when `m` (a `ValueId` max) is 0, same
  as the `guard_slot_base` cluster below; needs two guard *arms* whose
  schedules are large enough that an `m * 1`-vs-`m + 1` difference actually
  produces two arms sharing `ValueId`s.
- `:4027:86` (the `GUARD_ARM_NO_ORIGIN` sentinel pair, `u16::MAX - 1` vs
  `/ 1`) — likely equivalent (the two sentinel values are never compared to
  each other, only matched against a real arena's uniform ids, which never
  reach this high), but not proven by construction the way the two confirmed
  equivalents above were, so listed here rather than claimed.
- `:4124:45`, `:4125:49` (`fold_slot`'s `2 * j + root` arithmetic) and
  `:4135:73`, `:4164:29`, `:4165:49`, `:4175:65` (`guard_slot_base`,
  `guard_slot`'s additive term, `park_base`) — all six are the same shape as
  the two `guard_slot` mutants this pass *did* close: harmless when the
  pre-fold/pre-guard frame size `m` is 0, which every fixture in this file
  that reaches a fold or a guard at all happens to have (the two-sibling-fold
  fixtures spill inside the fold, not before it; `two_sibling_guards`'s body
  is two `Eq` masks and nothing else). Closing these needs a fixture
  combining what this pass's `two_sibling_guards` fixture has (two guards,
  or two folds, sharing one scope) with what the pre-existing spill fixtures
  have (register pressure in the *enclosing* scope, before the first guard
  or fold, so `m > 0`) — numerically checked by execution, the way
  `two_sibling_guards` is, not just by inspecting the allocation.
- `:4240:49` (`compile_via_backend`'s `def.value == vid` match guard,
  finding which `Reduce` def a fold's trip count comes from) and `:4262:49`
  (the trip-count multiply) — **a different class of gap**: both belong to
  `EmitTraffic`'s `trips` field, cost-model telemetry that is read back from
  `CompileResult::traffic`, not consulted by the emitted code at all. Every
  test in this file (including the ones this pass added) checks a kernel's
  *computed value*, which `trips` cannot affect — these two need a test that
  reads `result.traffic.trips` directly against a known fold-nest shape, a
  kind of assertion nothing in this file currently makes.

### A methodology note worth keeping

Two things worth keeping for the next pass over this file:

**The two unrelated things both called a guard.** `pixelflow-codegen`'s emit
driver has two separate mechanisms with the word "guard" in their names: the
`Select` short-circuit optimization (`select_guards`/`cluster_select_arms`/
`branch_if_arm_is_dead`, wrapping an ordinary `ScheduledOp::Ternary(Select,
..)` with a skip branch, no schedule-shape change) and `ScheduledOp::Guard`
(the "G2" mechanism, `extract_guards`/`schedule_guard_arm`, resolving two
*separate* kernels by `KernelKey` into their own scopes). A fixture built
the first way — a raw-arena `OpKind::Select` with exclusive-and-contiguous
arms — never produces a `ScheduledOp::Guard`, however elaborate, and
`nest.guard_count()` stays 0 regardless. The first attempt at this pass's
sibling-guards fixture used exactly that construction and asserted
`guard_count() == 2`, which failed outright rather than quietly passing for
the wrong reason — the cheap way to be wrong, since the assertion caught it
before any numeric check could lie. The working version built through
`arena.push_guard`/`KernelStore::intern`, the way `guard_arms`'s own
fixtures do.

**Verify "equivalent" by running the instrumentation, not just by reading
the code.** The `out_of_register`/`emit_resolve` argument for `:1816`/`:1971`
above is a real proof, but it was checked against actual behavior before
being trusted: both lines were instrumented with an `eprintln!` and the
entire 310-test suite run with it in place. Zero hits confirmed what the
code-reading argument predicted, rather than asking the reader to take the
argument on faith — the same bar the 09-16 pass set for `regalloc.rs`'s
equivalent mutants, applied here with an added empirical check because this
file's control flow (through two different fold/guard/select mechanisms) is
easier to misread than regalloc's.

## Verified

- `cargo test -p pixelflow-codegen --lib`: 328 passed, 0 failed (up from
  262 at the 09-16 pass's regalloc.rs count; the current run of this file's
  own `emit::tests` module alone is 112, up from 84 before this pass).
- `cargo test -p pixelflow-codegen` (all targets incl. doctests): pass.
- `cargo clippy -p pixelflow-codegen --lib --tests -- -D warnings`: clean.
- `cargo fmt -p pixelflow-codegen -- --check`: clean.
- `cargo test -p core-term --lib`: pass (renamed tests confirmed passing
  under their new names: `osc_52_with_a_malformed_or_non_utf8_payload_produces_no_action`
  and the 11 renamed `print_text_equivalence` tests).
- `cargo clippy -p core-term --lib --tests -- -D warnings`: clean.
- `cargo fmt -p core-term -- --check`: clean.
- `cargo test --workspace`: pass.
- `cargo mutants -p pixelflow-codegen -f pixelflow-codegen/src/emit/mod.rs`:
  332 mutants, 19 missed (2 confirmed equivalent, 17 real and carried
  forward with specific fixture requirements above), 227 caught, 83
  unviable, 3 timeouts.

## Recommended next steps (not done here)

Backlog carried forward, re-verified against the live tree:

1. `pixelflow-codegen/src/emit/mod.rs`'s remaining 17 real (non-equivalent)
   mutation gaps, all listed with their specific fixture requirements above
   — concentrated in three shapes: (a) `emit_scope`'s multi-iteration
   live-range reconciliation for a parked root crossing a fold's own back
   edge (`:1721`, `:1856`, `:1869`, `:1923`), (b) `compile_via_backend`'s
   frame-slot address arithmetic, which degenerates to a no-op whenever the
   pre-fold/pre-guard frame size is 0 — true of every fixture in this file
   that reaches a fold or guard at all (`:3998`, `:4124`, `:4125`, `:4135`,
   `:4164`, `:4165`, `:4175`), and (c) the `EmitTraffic`/`trips` cost-model
   telemetry, which no kernel-correctness test can reach by construction
   (`:4240`, `:4262`). `:2950` (×2) and `:4027` are each plausibly
   equivalent but not proven by construction the way `:1816`/`:1971` were.
2. `pixelflow-codegen/src/emit/aarch64.rs` (3,122 lines, 779 mutants by
   `cargo mutants --list`, up from 737) — still untestable at the
   runtime-execution level from this x86_64 sandbox, though its
   encoding-only assertions (does this operand sequence emit these exact
   bytes?) remain a path nobody has tried, flagged since the 09-16 pass. By
   this series' own tractability rule it is now the more expensive of the
   two files this pass chose between, so a future pass should budget for a
   partial sweep (`--re`/`-e` scoped to the encoding functions) rather than
   the whole file.
3. `pixelflow-core/src/backend/arm.rs` — **does not exist**. Deleted with
   `pixelflow-core`'s whole `backend` module in #1286 (`3d01892`,
   docs/BACKLOG.md C7, "the ISA is decided at startup"). Drop this item from
   the backlog; there is nothing here to sweep.
4. `pixelflow-codegen/src/emit/executable/macos.rs` — still blocked, no
   macOS host in this sandbox.
5. `core-term/tests/message_cuj_tests.rs` (15 tests) and the
   `ParserActor`/`TerminalApp` section of
   `core-term/tests/actor_roundtrip_tests.rs` (13 tests) — carried forward
   unchanged from the 09-16 pass; still test mock reimplementations rather
   than `core_term`'s real types, still needs a real design decision
   (rewrite vs. delete), still too large to make unreviewed in a
   test-quality pass.
