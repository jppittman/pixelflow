# Test quality control follow-up — 2026-09-28

Scope: scheduled continuation of the test-quality-audit series
(`docs/bugs/2026-09-16-test-quality-audit-followup.md`). Two independent
workstreams this pass, run in parallel: closing the mock-reimplementation
scope violation the 2026-09-16 pass explicitly deferred (its backlog item 5),
and a first mutation-testing pass against `pixelflow-codegen/src/emit/mod.rs`
(its backlog item 1 — the largest file in the series never yet swept).

## `core-term`: mock-based actor tests deleted (STYLE.md "test public API")

`core-term/tests/message_cuj_tests.rs` (15 tests) and the `ParserActor`/
`TerminalApp` section of `core-term/tests/actor_roundtrip_tests.rs` (13
tests) exercised hand-rolled test doubles (`MockParserActor`,
`TestParserActor`, `MockAnsiCommand`, `TestAnsiCommand`) that reimplemented
ANSI parsing and key-to-escape translation locally in the test file, never
touching `core_term::ansi::AnsiProcessor` or
`core_term::term::emulator::key_translator`. Both files said so in their own
header comments; the 2026-09-16 pass flagged it but declined to act,
calling it "too large a change to make unreviewed."

Deleted `message_cuj_tests.rs` outright and the `ParserActor`/`TerminalApp`
section from `actor_roundtrip_tests.rs` (its `pty_writer_*` section, which
already drives the real `core_term::io::event_monitor_actor::WriterControl`/
`core_term::io::Resize` types through a probe actor, is untouched and
remains the pattern to follow). For each deleted test, confirmed real
coverage already exists against the real types, or added it:

- Real ANSI parsing (escape sequences, SGR, C0 controls, fragmented input,
  empty input, multi-batch ordering): `core-term/tests/ansi_parser_message_tests.rs`
  against the real `AnsiProcessor`/`AnsiSink`.
- Real key translation and its PTY-write action: `core-term/src/term/tests.rs`
  and `key_translator.rs`'s own unit tests, against `TerminalEmulator::interpret_input`/
  `key_translator::translate_key_input`.
- Channel-closure propagation (generic `actor-scheduler` mechanics): already
  directly tested in `lib.rs`, `mealy.rs`, `ports.rs`, `dedicated_thread.rs`
  and `spsc.rs` — not incidental, genuine assertions.
- **Genuine gap, closed**: `ActorScheduler<D,C,M>` (the `ShardedInbox`-drained
  type `core-term` actually drives, distinct from the `mealy`/`Transducer`
  substrate `dedicated_thread.rs`'s priority test covers) had no direct test
  of its own cross-lane priority order or within-lane FIFO order through real
  `handle_data`/`handle_control`/`handle_management` calls. Added
  `actor-scheduler/tests/priority_and_order.rs`
  (`the_control_lane_drains_before_management_which_drains_before_data`):
  preloads 3 Data + 3 Management + 3 Control messages, drains via the real
  `ActorScheduler::run`, and asserts the exact interleaving
  `["C0","C1","C2","M0","M1","M2","D0","D1","D2"]`.
- Two scenarios judged **not** actor-scheduler-specific contracts and left
  without a replacement: a sender dropped while a handler call is
  synchronously in-flight (single-threaded call semantics — nothing can
  interrupt a running `handle_data`, so there is no design choice to pin),
  and a raw `panic!()` inside a handler unwinding through
  `thread::spawn().join()` (ordinary Rust panic propagation, not this
  crate's guarantee — and moot in every real build profile besides, which
  set `panic = "abort"`).

## Mutation testing: `pixelflow-codegen/src/emit/mod.rs` (`cargo-mutants` v27.1.0)

8,266 lines (up from 5,460 at the last count), untouched by any prior pass in
this series. First sweep, scoped to the one file
(`cargo mutants -p pixelflow-codegen -f pixelflow-codegen/src/emit/mod.rs`):
**332 mutants tested in 32m: 42 missed, 205 caught, 83 unviable, 2 timeouts.**

The 83 unviable mutants are cargo-mutants' own bucket (fails to compile) and
needed no action, same as every prior pass in this series. The 2 timeouts
(`emit_scope:1719` and `compile_via_backend:4152`, both an operator swap
that makes the mutated build hang rather than fail a specific assertion) are
left alone rather than chased into clean "caught" results — a hang is
already an observable regression in its own right, the same call the
2026-09-16 pass made for `regalloc.rs`'s own timeouts.

Of the 42 missed, **22 fixed and hand-verified this pass** (by reapplying
the exact mutation, confirming the new test fails, then reverting — the
2026-09-16 doc's methodology note is explicit that "should catch this"
reasoning alone isn't good enough; every fix below was actually re-run
against its mutation before being counted as closed).

Closed clusters:

- **`Label`/`Assembly`/`PtrReg`/`AsmProgram` had no direct unit test at all**
  — every existing test reached them only incidentally through a full
  compile. Added `label_as_str_returns_the_name_it_was_constructed_with`,
  `label_new_panics_when_the_name_is_longer_than_its_capacity`,
  `label_display_writes_the_plain_name_without_quotes`,
  `label_debug_writes_the_name_in_quotes`,
  `assembly_with_capacity_reserves_room_without_writing_any_bytes`,
  `ptr_reg_raw_returns_the_underlying_register_index`, and
  `a_nested_asm_program_emits_its_inner_instructions_in_order` (the last
  pins `AsmProgram`'s own `AsmInsn::emit_into`, which delegates to
  `assemble` — an `AsmProgram` is itself an instruction and can be nested).
- **`Loc`/`Binding`'s accessor and `SourceOperand` conversions** — same
  pattern, reached only by reading a `Binding` a real allocation produced.
  Added direct tests for `Loc::source_storage`, `Binding::as_loc`,
  `Binding::as_storage`, `Binding::as_slot` (both the spilled and the
  register/rematerialized cases), and `Binding`'s own `SourceOperand::source_storage`.
- **`operand_sources`'s `MulAdd` destination choice** — the guard
  `!resident[0] && !resident[1]` decides fused-FMA vs. decomposed
  multiply-then-add, and was reachable only through the full
  `resolve_operands`/compile pipeline, never directly. Added three tests
  covering: both multiplicands resident (fused, addend to destination), only
  one multiplicand resident (**still** fused — this is what distinguishes
  `&&` from a wrongly-lenient `||` in the guard), and both multiplicands
  needing reload (decomposed, product to destination). Hand-verified the
  middle test alone kills all three real mutants at this site (the guard
  literal, the `&&`→`||` swap, and deletion of the fused-path match arm).
- **`resolve_operands`'s own, separate `MulAdd` decompose guard**
  (`a_spilled && b_spilled`, the register/deferred-reload-carrying twin of
  `operand_sources`' abstract version above) had the exact same gap despite
  three existing tests against the all-resident, both-spilled and
  all-three-spilled cases — none of them exercised exactly one multiplicand
  spilled. Added `resolve_operands_still_fuses_muladd_when_only_one_multiplicand_is_spilled`,
  same shape as the fix above, at this function's own level (checks the
  `ResolvedOp::FusedMulAdd` payload and reload/setup_mov details, not just
  the abstract `OperandSource`).
- **`IsaBackend::test_ge`'s default body** — every backend but AVX-512
  inherits it, delegating to `alu(Ge, ...)`; AVX2 and NEON's own
  full-pipeline fold-loop tests run on this host's *native* tier, which
  happens to be AVX-512, so the default was never reached by any of them.
  Added `test_ge_defaults_to_an_ordinary_ge_comparison`, comparing an
  explicit `Avx2Backend`'s `test_ge` output against its own `alu` output
  directly.
- **`schedule_variance`'s `Var` index boundary** — the guard
  `*idx < Variance::VARIABLES` decides whether a `Var` names one of the
  lattice's own variables or falls through to `Variance::ALL`; every
  existing compile ever exercised happens to use `Var` indices nowhere near
  `Variance::VARIABLES` (64). Added two tests pinning both sides of the
  boundary directly against a hand-built one-`Def` schedule.

**Carried forward, not fixed this pass** — 20 of the 42 first-sweep misses,
in four clusters, all diagnosed but not attempted blind (to avoid a test
that passes for the wrong reason — the exact failure mode the 2026-09-16
doc's own "methodology note" warns about):

1. **`emit_scope`'s fold-loop head/pointer reconciliation** (lines 1719
   [timeout], 1721, 1816, 1856, 1869, 1923, 1971 — 7 sites) — the "hand a
   live register-carried value to a fold's back edge" and "hand a parked
   root over to the scopes inside it" machinery
   (`docs/plans/2026-09-10-a-surviving-reduce-is-a-loop.md`'s territory).
   Reachable only when a nested fold's register allocation produces a
   specific head/tail mismatch or a specific already-resident register — a
   real scenario, but one that needs a fixture built against the
   allocator's actual output rather than a hand-picked residency array, the
   way this pass's other fixes were.
2. **`pending_reads`** (line 3096) — a one-line, zero-logic delegation to
   `guards::FoldReads::new`, same shape as the `AsmProgram::emit_into` fix
   above, but constructing its `PendingFold`/`guards::FoldReads` inputs by
   hand needs a closer read of `guards.rs` than this pass had time for.
3. **`extract_guards`** (lines 3998, 4027) — arithmetic inside the fold's
   guard-extraction pass; not yet read closely enough to say what a direct
   fixture needs.
4. **`compile_via_backend`'s frame-slot-layout arithmetic** (lines 4124,
   4125 ×2, 4135, 4152 [timeout], 4164, 4165 ×3, 4175, 4240, 4262 — 12
   sites) — the fold/guard/park slot address computation (`2 * j`,
   `m + ... * vector_bytes`, trip-count multiplication). Every existing
   full-pipeline test computes through this code and checks the *numeric
   result* is correct, which several of these arithmetic swaps apparently
   survive without corrupting for the specific fold/guard counts (often 0
   or 1) those fixtures happen to use — the fix needs a test that asserts
   on the *slot addresses themselves* for a schedule with at least two
   folds or two guards, built against `regalloc::Nest`'s real output rather
   than reconstructed by hand, to be sure of what it's asserting.

Left as this file's own backlog below.

Final numbers, this file, this pass: of the 332 mutants the first sweep
found, **227 are now caught by a verified test** (205 first-sweep-caught +
22 fixed and hand-verified this pass), 83 unviable, 2 timeouts, and **20
missed, carried forward** in the four clusters above. A fresh full re-sweep
against the fixed file was not re-run (each of the 22 fixes was instead
verified individually, per the methodology above) — the next pass against
this file should start with one, both to confirm this count and to see
whether it turns up anything new past where this pass stopped reading.

## STYLE.md compliance: test naming (`pixelflow-codegen/src/emit/mod.rs`)

Fourteen test names in this file's `mod tests` (and its `mod sched`
submodule) described a scenario or the mechanism under test rather than the
expected outcome — the same class of fix as the 2026-09-16 pass's
`core-term` naming pass. Renamed each to a complete "it should ..."
sentence, no test bodies or assertions changed (e.g. `resolve_binary_no_spills`
→ `resolve_operands_reads_both_operands_directly_when_neither_is_spilled`,
`sched_select_guards` → `compile_takes_the_correct_select_arm_under_the_guard_short_circuit`).
One stale comment referencing the old name `sched_select_guards` was updated
to match. No scope violations found in this file's own tests (no test
reaching past the file's own encapsulation into an unrelated private API);
this file's tests already poke plenty of its own private items directly
(`operand_sources`, `resolve_operands`, `schedule_variance`, `Assembly`,
etc.), which is normal same-file white-box unit testing, not a STYLE.md
violation — the same conclusion the 2026-09-16 pass reached for
`regalloc.rs` and `screen.rs`.

## Verified

- `cargo test -p pixelflow-codegen --lib`: 321 passed, 0 failed.
- `cargo test -p pixelflow-codegen` (all targets incl. doctests): pass.
- `cargo clippy -p pixelflow-codegen --lib --tests -- -D warnings`: clean.
- `cargo fmt -p pixelflow-codegen -- --check`: clean.
- `cargo test -p core-term`: 602 passed, 0 failed.
- `cargo test -p actor-scheduler`: 153 lib tests + 29 integration tests
  (`dedicated_thread` 10, `ports` 7, `priority_and_order` 1 new,
  `stress_tests` 11; `generic_type`/`simple_type` are compile-only) + 1
  doctest, all passed, 0 failed.
- `cargo clippy -p core-term --tests -- -D warnings`: clean.
- `cargo clippy -p actor-scheduler --tests -- -D warnings`: clean.
- `cargo fmt -p core-term -- --check` / `-p actor-scheduler -- --check`: clean.
- `cargo test --workspace`: pass (exit 0, no failures across every crate's
  lib/integration/doc tests).

## Recommended next steps (not done here)

Backlog carried forward from 2026-09-16, minus the item closed above:

1. `pixelflow-codegen/src/emit/mod.rs`'s `emit_scope` fold hand-off/head
   reconciliation cluster (7 sites, see above) — needs allocator-driven
   fixtures, not hand-picked residency arrays.
2. `pixelflow-codegen/src/emit/mod.rs`'s `pending_reads`, `extract_guards`,
   and `compile_via_backend` frame-slot-layout clusters (1 + 2 + 12 sites,
   see above).
3. `pixelflow-codegen/src/emit/mod.rs`, the rest of it beyond this pass's
   reading (roughly lines 1-4300 of the ~8,266; the ~4,000-line `mod tests`
   block past that was only spot-checked for naming) — a fresh full sweep
   against the fixed file will say what else is left once items 1-2 close.
4. `pixelflow-codegen/src/emit/aarch64.rs` (737 mutants at last count) —
   untestable at the runtime-execution level from this x86_64 sandbox, though
   its encoding-only assertions could still be mutation-tested.
5. `pixelflow-core/src/backend/arm.rs`'s NEON impls — still untestable from
   every x86_64 sandbox this series has run in; needs an aarch64 host.
6. `pixelflow-codegen/src/emit/executable.rs`'s `macos.rs` submodule —
   untouched (no macOS host in this sandbox).
