# Test quality control follow-up — 2026-09-25

Scope: scheduled continuation of the test-quality-audit series
(`docs/bugs/2026-09-16-test-quality-audit-followup.md`'s backlog item 1,
`pixelflow-codegen/src/emit/mod.rs`), plus a STYLE.md naming/scope pass over
`core-term` tests added or changed since that pass.

## Mutation testing: `cargo-mutants` v27.1.0

Not present in this environment (consistent with every prior pass) —
already installed at `/root/.cargo/bin/cargo-mutants`.

### `pixelflow-codegen/src/emit/mod.rs`

Backlog item 1 from the 09-16 doc, picked up now that `regalloc.rs` (the
other half of that item) is closed. The file has grown since it was last
sized up: 5,460 → 8,266 lines, 275 → 332 mutants — orchestration glue over
the allocator (assembler primitives, operand-source selection, the fold/guard
schedule extraction, the emitter's own driver loop, and the frame/slot layout
`compile_via_backend` hands the emitter).

First sweep: **332 mutants, 204 caught, 43 missed, 83 unviable, 2 timeouts**,
in 25 minutes (`-j 2`; the 83 unviable are `const fn` bodies and similar
compile-time-only contexts, same bucket every prior pass has found and needed
no action).

Unlike the `regalloc.rs` pass — one coherent algorithm, missed mutants spread
fairly evenly through it — every one of these 43 missed (and both timeouts)
sits in the file's first half (line 200 through line 4262 of 8266); nothing
past `compile_via_backend` (the driver that lays out a whole nest's frame and
calls `emit_scope` once per scope) has a single missed mutant. The back half
of the file — `emit_scope`'s own body past its first ~250 lines, `resolve_operands`'s
non-`MulAdd` arms, `extract_folds`, `place_roots`, `schedule_guard_arm`, the
four backend drivers — is already fully covered by this file's own large
integration-style test suite (dozens of kernels compiled and run end to end
through `EmitCtx::compile`/`compile()`, per docs/plans/2026-09-16-collapse-is-a-fold.md
and 2026-09-12-emit-should-just-emit.md). The 43 misses are concentrated where
that suite has a genuine blind spot instead: small value-representation types
with no test of their own, and a handful of specific boundary conditions
inside otherwise well-exercised functions that no existing kernel's shape
happens to cross.

Closed, verified the same way as every prior pass — by manually re-applying
each mutation and re-running the targeted test, not by trusting cargo-mutants'
own "caught" verdict or the assertion's shape:

- **`Label`/`Assembly`/`AsmProgram`'s own composition, `PtrReg`, `Loc`,
  `Binding`** (15 mutants) — the file's small value-representation types
  (a label's name, an assembler's buffer, a register-class newtype, and the
  two "where does a value live" enums) had **no direct unit test at all**;
  every existing test reached them only incidentally through a full compile.
  Added a `primitives` module with one test per method/impl: `Label::as_str`/
  `Display`/`Debug` round-trip the name; `Assembly::with_capacity` actually
  reserves the requested bytes (`Vec::with_capacity(0)` vs `capacity` is
  invisible to a `finish()`-only check, so the test asserts `.capacity()`
  directly); `Assembly::bind`/`push`/`finish` against a trivial hand-written
  `AsmInsn`, including both of `finish`'s panics; `AsmProgram<S>`'s own
  `AsmInsn` impl (an `AsmProgram` nested as one instruction of another —
  the only path that reaches `<AsmProgram<S> as AsmInsn>::emit_into` rather
  than the inherent `assemble`), checked by actually assembling one and
  reading the bytes back; `PtrReg::raw`/`as_gpr` against a non-0/1 index, so
  the two "replace with a constant" mutants can't hide behind a coincidence;
  and `Loc`/`Binding`'s `reg`/`storage`/`as_loc`/`as_storage`/`as_slot` and
  both types' `SourceOperand`/`StoreTarget` impls, one assertion per variant
  so the match itself is pinned rather than one arm of it.
- **`operand_sources`' `MulAdd` destination contest** (3 mutants) — this
  function (which operand of an instruction is resident, reloaded into the
  destination, or reloaded into a reservation) had no direct test either,
  reached only through `resolve_operands`'s own `resolve_muladd_*` fixtures,
  none of which drove its `into_dst` decision directly. Added an
  `operand_source_selection` module covering every `ScheduledOp` arity and,
  for `MulAdd`, both forms of its guard (`!resident[0] && !resident[1]`) —
  a case where it fires, a case where one multiplicand is already resident so
  it must not, and the fallback arm (deleted, or its `&&` widened to `||`)
  each get their own case, plus `Select`'s unconditional destination and
  `reloads_wanted`'s count.
- **`schedule_variance`'s `Var` fallback** (2 mutants) — every production
  `Var` this file's own tests ever build names a lattice axis or a reduce
  binder, both far below `Variance::VARIABLES` (64), so nothing exercises the
  `idx < VARIABLES` guard's `false` side or its exact boundary (`idx ==
  VARIABLES`, the first index `Variance::from_var` itself refuses). Pinned
  both directly against `ScheduledOp::Var` values built by hand.
- **`IsaBackend::test_ge`'s default body** (1 mutant) — the fold trip test
  every backend but AVX-512 uses (AVX-512 overrides it with its own
  k-register form). This sandbox's host CPU has AVX-512, so every
  `EmitCtx::compile`/`native_schedule` `Reduce` test in this file dispatches
  to the override and never reaches the default at all — the default was
  *zero-coverage dead code from this suite's point of view*, on this specific
  kind of host, however many `Reduce` tests it runs. Closed with a test that
  drives `Avx2Backend` directly (`compile_via_backend`, not `EmitCtx::compile`)
  and **executes** the result: AVX2 machine code runs on any AVX-512 host (a
  superset ISA), so this is real coverage, not merely a compiles-without-
  panicking check. Manually reapplying the mutation here didn't produce a
  quick, clean test failure — it hung: the corrupted trip test spins the
  emitted native loop forever. That is caught, in the same sense the
  regalloc.rs pass's loop-counter timeouts are
  (`docs/bugs/2026-09-16-test-quality-audit-followup.md`, "a hang is a very
  observable regression in its own right"), and it was
  confirmed by killing the run and inspecting the CPU-bound child process
  rather than by waiting it out. NEON cannot execute on any x86_64 sandbox
  this series has run in (the aarch64.rs backlog item, carried forward again
  below), so the default's NEON call site stays covered only by compilation,
  not execution.
- **`resolve_operands`'s decomposed-vs-fused `MulAdd` boundary** (1 mutant,
  `&&` → `||`) — every existing `resolve_muladd_*` fixture spills both
  multiplicands or neither; none spills exactly one. Added the missing case
  (`a` resident, `b` spilled, `c` resident): the fused form must still be
  taken, addend reloaded straight into `dst`, not the decomposed
  `FMUL`+`FADD` form a `||` would trigger too eagerly.

Final sweep: **332 mutants, 226 caught, 20 missed, 83 unviable, 3 timeouts**.
The `test_ge` fix above moved that one mutant from "missed" to "timeout" —
this pass's fix count (22) plus the final "missed" count (20) is 42, one
under the first sweep's 43, because one further mutant
(`compile_via_backend`'s `guard_slot` closure, `4165:49 + → *`) stopped
reproducing between the two sweeps without a fixture written for it —
collateral coverage from one of the fixtures above (most plausibly
`resolve_operands_gap_tests`' `EmitCtx::default()` compile, which reaches
`compile_via_backend` on a path the pre-existing suite's other fixtures
didn't combine the same way) rather than anything targeted at that line.
Left as an observation, not claimed as a deliberate fifth closed group.

### What is left, and why it stops here

The remaining 20 missed and both **pre-existing** timeouts (`emit_scope`'s
`1719`, `compile_via_backend`'s `4152` — present in the first sweep too, and
not touched by anything in this pass) sit in two places, and both were
investigated rather than skipped on sight:

- **`emit_scope`'s early carry/reconciliation logic** (lines 1719–1971: guard
  range bookkeeping, a scope's head reconciling a value's placement against
  where the previous iteration's tail left it, and a parked root's
  keep-or-store decision) and **`pending_reads`** (line 3096, the per-scope
  cost table `guards::cluster_select_arms` schedules arms against) — reached
  by every fold/guard-bearing test in this file, but a `Default::default()`
  or an inverted comparison here changes *scheduling and instruction
  selection*, not the arithmetic result a black-box correctness test checks.
  Closing these needs an assertion on the *emitted form* (which register a
  value carries in, whether a branch was taken, an instruction count) the
  way `regalloc.rs`'s own "destination contest" and "guarded-arm" fixtures
  are written, not another numeric kernel test — the same category of work
  as that file's own pass, at a scale (dozens of fixtures, one per branch)
  this single pass did not have room for.
- **`extract_guards`' arm-id counter and `compile_via_backend`'s frame/slot
  arithmetic** (lines 3998–4262: `next_id`'s bump between arms, and the
  `fold_slot`/`guard_slot`/`park_base` address formulas) — investigated
  directly rather than left on reputation. Manually reapplying
  `park_base`'s `* vector_bytes` → `+ vector_bytes` mutation and running the
  *whole* `emit::tests` module found it silently passing even under real
  register pressure (`EmitCtx::with_max_regs(MIN_SCRATCH)`) with a kernel
  combining a guard and a sibling fold's parked leaf — because with exactly
  one guard, the formula's error is a **constant offset**: `park_base` still
  lands past every other region's slots, so nothing aliases, and the "bug"
  only wastes a few bytes of stack. The same experiment on `guard_slot_base`
  showed the opposite failure mode for the same reason: with `fold_count`
  fixed at 1 in every fixture that also carries a guard, `2 * fold_count`
  and `2 + fold_count` agree. Both formulas are genuine bugs — the algebra
  above shows `park_base` under-counts once `guard_count() >= 2` (`n *
  vector_bytes` grows faster than `n + vector_bytes`, so past the crossover
  point the linear term is no longer large enough to clear the guard slots'
  own region) — but exercising that needs **two or more simultaneous
  top-level guards** sharing a nest with a fold under enough register
  pressure to force an actual park, a fixture this pass built once, confirmed
  didn't trigger the single-guard case, and did not have room to iterate to
  the two-guard one and its own register-pressure tuning. Left as a
  precisely-scoped follow-up rather than a vague one.

One further missed mutant in this same range looks like a genuine equivalent
rather than a gap, though it was not chased to the same proof standard as the
two clusters above (re-applying it and confirming *no reachable input*
distinguishes it, as `regalloc.rs`'s documented equivalents were):
`GUARD_ARM_NO_ORIGIN`'s `u16::MAX - 1` (line 4027, mutated to `u16::MAX / 1`,
i.e. `u16::MAX` again) is a sentinel pair whose own doc comment says the only
property that matters is being "far past any arena's own uniform table" —
nothing reads the two entries expecting them to differ from each other, only
from a real `UniformId`, so collapsing them to the same value is very
plausibly unobservable. Flagged rather than claimed, and left for whoever
picks up the two clusters above to confirm or refute alongside them.

Both clusters are carried forward rather than closed here; see "Recommended
next steps."

## STYLE.md compliance: test naming and scope (`core-term`)

Audited every `#[test]` function added or changed in the seven `core-term`
commits since the 09-16 pass (`d89c34f` through `044b153`: carrying out
emulator actions from keys and the shell, bindings/paste/focus, named keys,
zoom, mouse selection/primary paste/full screen/OSC 52, loading the config
file, and deleting the per-character `Print` path) — 27 new or renamed test
functions across `core-term/src/config.rs`, `core-term/src/io/pty_tests.rs`,
`core-term/src/keys.rs`, `core-term/src/term/emulator/input_handler.rs`,
`core-term/src/term/emulator/key_translator.rs`,
`core-term/src/term/emulator/osc_handler.rs`, and `core-term/src/terminal_app.rs`
(found by diffing `d89c34f^..044b153` for `+fn`/`+    fn` lines and cross-checking
against what each file's test module actually kept).

**No naming violations found.** Every one of the 27 already reads as a
complete "it should ..." sentence without opening the body —
`a_working_directory_that_cannot_be_entered_fails_the_spawn`,
`focus_changes_are_reported_only_when_the_program_asked_for_them`,
`zoom_stops_at_its_limits`, `osc_52_refuses_to_read_the_selection_back_and_ignores_garbage`,
`the_zoom_binding_resizes_the_pty_and_the_next_frame_draws_at_the_new_size`, and
so on. `core-term/src/term/emulator/input_handler.rs` gained
`it_should_send_pasted_text_to_the_pty_unwrapped_when_bracketed_paste_mode_is_off`,
whose literal `it_should_` prefix looked at first like a one-off anomaly
against the rest of the file's sentence-style names — it turned out to be
consistent with its own pre-existing sibling,
`it_should_wrap_pasted_text_in_bracketed_paste_sequences_when_bracketed_paste_mode_is_on`
(unchanged in this window, predates 09-16), so both names stand. Nothing here
needed a rename.

**No scope violations (rules 1–2) found.** Every new test drives the module's
real public or `pub(crate)` entry point rather than a private substitute:
`TerminalEmulator::interpret_input` (input_handler.rs, key_translator.rs,
osc_handler.rs), `TerminalApp::handle_data`/`handle_management`/`handle_control`
(terminal_app.rs), `NixPty::spawn_with_config` (pty_tests.rs), and
`map_key_event_to_action` (keys.rs). `config.rs`'s four new tests call
`load_config_from` directly rather than through the production entry point
`load_config_from_file_or_defaults` — that entry point reads a fixed OS
config-directory path and cannot take a test's scratch directory, so the
tests call the same-file private helper that holds the actual load-and-merge
logic (`load_config_from_file_or_defaults` is a one-line wrapper over it).
This is the same pattern the 09-16 pass already found acceptable in
`term/screen.rs`'s tests: same-file, and the helper being called *is* the
logic under test, not a stand-in for it.

One test, `a_middle_click_pastes_the_primary_selection_and_f11_toggles_fullscreen`
(terminal_app.rs), asserts two unrelated behaviors — middle-click paste and
the F11 fullscreen toggle — in one function. The name says exactly what it
covers, so it is not a rule-3 violation, but it is a scope smell worth
naming rather than silently leaving out of this doc; splitting it is a
behavior-preserving test change beyond "rename mechanically," so it is left
alone here and not carried forward as a numbered backlog item — the name
already discharges rule 3, and a mechanical pass isn't the place to redesign
test grouping.

`ansi/tests.rs`'s 962-line diff and `term/tests.rs`'s four function deletions
were both checked and are not test-naming/scope issues: the former is a
mechanical `Vec<AnsiCommand>` → `Vec<Parsed>` helper-return-type change with
no test renamed or added, and the latter deleted four paste-handling tests
whose behavior moved to `input_handler.rs` and was retested there under the
names reviewed above.

Two files already flagged as a genuine, larger scope issue in the 09-16 pass
remain untouched and are **not fixed here** (same reasoning, still too large
for an unreviewed mechanical pass): `core-term/tests/message_cuj_tests.rs`
and the `ParserActor`/`TerminalApp` half of
`core-term/tests/actor_roundtrip_tests.rs`. See "Recommended next steps."

## Verified

- `cargo test -p pixelflow-codegen --lib`: 340 passed, 0 failed.
- `cargo test -p pixelflow-codegen` (all targets incl. doctests): pass.
- `cargo clippy -p pixelflow-codegen --lib --tests -- -D warnings`: clean.
- `cargo fmt -p pixelflow-codegen -- --check`: clean.
- `cargo test -p core-term --lib`: pass (no `core-term` production or test
  code changed this pass beyond the audit above, which found nothing to fix).
- `cargo test --workspace`: pass.
- `cargo mutants -p pixelflow-codegen -f pixelflow-codegen/src/emit/mod.rs`:
  332 mutants, 20 missed, 226 caught, 83 unviable, 3 timeouts (down from 43
  missed / 2 timeouts on the first sweep; one of the three final timeouts,
  `test_ge`, is a closed gap caught via hang rather than a clean failure —
  see above).

## Recommended next steps (not done here)

Backlog carried forward from 2026-09-16, minus the item closed above:

1. **New**: `pixelflow-codegen/src/emit/mod.rs`'s remaining 20 missed
   mutants (+ 2 pre-existing timeouts) — see "What is left, and why it stops
   here" above for the precise shape of both remaining clusters (`emit_scope`'s
   carry/reconciliation logic and `pending_reads`, needing emitted-form
   assertions rather than numeric ones; `extract_guards`/`compile_via_backend`'s
   arm-id and frame/slot arithmetic, needing a two-simultaneous-guard fixture
   under register pressure). Comparable in scope to the `regalloc.rs` pass
   itself; a dedicated follow-up, not a continuation of this one.
2. `pixelflow-codegen/src/emit/aarch64.rs` (2,601 lines, 737 mutants as of
   09-16) — untestable at the runtime-execution level from this x86_64
   sandbox, though its encoding-only assertions could still be
   mutation-tested. `mod.rs`'s `test_ge` fix above found one narrow exception
   (AVX2 code executes on an AVX-512 host) that does not extend to NEON.
3. `pixelflow-core/src/backend/arm.rs`'s NEON impls — still untestable from
   every x86_64 sandbox this series has run in; needs an aarch64 host.
4. `pixelflow-codegen/src/emit/executable.rs`'s `macos.rs` submodule —
   untouched (no macOS host in this sandbox).
5. `core-term/tests/message_cuj_tests.rs` (15 tests) and the
   `ParserActor`/`TerminalApp` section of `core-term/tests/actor_roundtrip_tests.rs`
   (13 tests) test mock reimplementations, not `core_term`'s real types —
   carried forward unchanged since 09-16 (see that doc, and "STYLE.md
   compliance" above). Needs a real design decision (rewrite against real
   types, or delete and confirm `actor-scheduler`'s own suite plus
   `ansi_parser_message_tests.rs` already cover what's real here), not a
   mechanical fix.
