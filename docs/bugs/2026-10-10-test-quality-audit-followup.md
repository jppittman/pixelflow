# Test quality control follow-up — 2026-10-10

Scope: scheduled continuation of the test-quality-audit series
(`docs/bugs/2026-09-16-test-quality-audit-followup.md`). Since that pass,
33 commits (#1322–#1355, the AVX2/AVX-512 "selection" pipeline — selection
phases A1–C2) landed a new instruction-selection driver
(`pixelflow-codegen/src/emit/select.rs`) and its supporting register-file
declaration type (`pixelflow-codegen/src/emit/register_file.rs`), both
untouched by any mutation-testing pass so far. Picked these two over the
09-16 backlog's carried-forward items (`emit/mod.rs` at 7,289 lines now,
`aarch64.rs`, `pixelflow-core/src/backend/arm.rs`, `executable.rs`'s
`macos.rs`) because they are new, coherent, and the obvious next targets
by this series' own precedent of auditing a complete unit at a time; the
carried-forward items are untouched again.

## STYLE.md compliance: test naming and scope

Every test name added across the selection phase's own test files
(`golden_selected.rs`, `selection_stays_linear.rs`, `register_pressure.rs`,
`residency.rs`, `memory_ratchet.rs`, `collapse_paths.rs`, `deep_frame.rs`,
`emit_traffic.rs`, `empty_fold.rs`, `loops.rs`) already reads as an "it
should ..." sentence (`a_gather_reads_the_bound_buffer`,
`a_guarded_if_runs_the_arm_that_ran`, `legacy_traffic_is_the_recorded_table`,
and so on) — no renames needed. All of them test through `compile`/
`CompileResult`/`ScopeTraffic`, the crate's public JIT entry points and
telemetry, not through any private type reached past its own file's model.
`register_file.rs` had no tests at all, so there was no naming to audit
until this pass added them (below).

## An environment trap, found before it could cost a false negative

`cargo mutants` on this sandbox's host detects AVX-512
(`pixelflow_codegen::isa::detect() == Isa::Avx512`), and
`Pipeline::of(Isa::Avx512)` is still `Legacy` — only AVX2 has switched to
the selection pipeline by default (`docs/plans/2026-10-08-selection-is-a-phase.md`).
Every integration test that means something only under the selection
pipeline starts with `if !selection() { return }` (`tests/support/knob.rs`),
so a first `cargo mutants -f .../select.rs` run, with no
`PIXELFLOW_CODEGEN` set, silently skipped every one of those tests'
assertions — they still ran and still reported "ok", just without ever
reaching `select.rs`. That run reported 7 missed mutants, several of which
turned out to be artifacts of this: a plain `cargo test` on this host
proves nothing about `select.rs` by itself.

Re-run with `PIXELFLOW_CODEGEN=selection` — the same variable CI's
`isa-matrix` job sets on its AVX2 leg
(`.github/workflows/rust.yaml`, "AVX2 compiles with the selection
pipeline... with `PIXELFLOW_CODEGEN=selection` at AVX2") — the missed count
dropped from 7 to 3 with no code change: `emit_traffic.rs`'s existing
`a_selected_constant_is_counted_as_brought_in` (gated the same way) was
already the right test for one of the seven, just never exercised on this
host without the override. The general lesson, worth stating for whoever
runs this series' next pass from a different host: **a tier-gated test
suite needs the gate forced open before mutation-testing it, and the
"missed" count without doing so is not a measurement of the tests.**

## Mutation testing: `cargo-mutants` v27.1.0

Not present in this environment (consistent with every prior pass) —
installed via `cargo install cargo-mutants --locked`.

### `pixelflow-codegen/src/emit/register_file.rs` (126 lines, new)

`RegisterFile::new` is the one constructor for a backend's register-file
declaration (`FILE: RegisterFile = RegisterFile::new(...)`, `const`, called
once per backend in `avx2.rs`/`avx512.rs`) and documents that it refuses a
self-contradictory one: a member named twice in one file, two flags
registers, an entry argument (`ctx`/`out`/`pitch`) outside the general file
or shared between two of the three, a vector width below 16 bytes or not a
power of two. Nothing called it with bad input — every production caller
only ever declares a valid file, so every `assert!` in it had executed
only on the side that never panics, the same shape of gap this series
found and closed in `RegSet`/`GprSet`/`MaskSet` on 2026-09-16.

Added nine tests in a `mod tests` in the file itself (white-box, same
relationship `regalloc`'s own bitset tests have to their file): one per
`assert!`, each constructing a `Members`/`EntryRegisters` that violates
exactly one rule and expecting the documented panic message, plus one
confirming a valid declaration reports back what it was given through
`members()`/`entry()`/`vector_bytes()`.

First sweep (after adding the tests): **23 mutants, 7 caught, 0 missed, 16
unviable.** The 16 unviable are `cargo-mutants`' own compile-failure
bucket — mutating `distinct`/`contains`'s bodies to `true`/`false` or
rewriting their loop arithmetic breaks `const`-evaluability wherever a
production `FILE` declaration then fails to const-evaluate, and one
(`RegisterFile::entry` replaced with `Default::default()`) fails because
`EntryRegisters` has no `Default` — all genuine compile errors, not
survivors needing a test.

### `pixelflow-codegen/src/emit/select.rs` (523 lines, new)

The instruction-selection driver: walks a `ScopedSchedule`'s scopes,
selecting each `ScheduledOp` into the backend's `Lane`s, opening a fold as
a loop of blocks, and choosing a guarded branch over a lane-wise blend for
an `If` whose arm is worth skipping. It is `pub(super)`, reached only
through `compile`/`CompileResult`, and is already exercised richly by
value-correctness tests (`collapse_paths.rs`'s nine tests cover every
`ScheduledOp` variant's main path end to end, including nested and
sequential guarded `If`s) and by the golden byte-exact pins
(`golden_selected.rs`, and `emit::tests::sibling_folds::GOLDEN` in
`mod.rs`, which emits all three backends host-independently).

First sweep, `PIXELFLOW_CODEGEN=selection`: **55 mutants, 29 caught, 23
unviable, 3 missed** (down from 7 missed on the first, ungated run — see
above). The three:

| mutation | line | |
|---|---|---|
| delete `!` | 147 | `if !matches!(def.op, Const\|Outer\|Seq\|Var\|Reduce) { scheduled[...] += 1 }` — which ops get counted as a scheduled operation |
| `+=` → `*=` | 155 | the same counter, in the default-counting arm |
| `+=` → `*=`/`-=` | 199 | the same counter, in the `Const` arm's own increment |
| `&&` → `\|\|` | 223 | whether an `If` at position `at` is guarded: `guards.iter().any(\|g\| g.if_idx == at && g.has_guarded_arm())` |

(147 and 155 are the same table row in the earlier sweep's numbering —
both are the single counting site the new test targets, below.)

**147 and 155, closed.** `ScopeTraffic::instructions` is public telemetry
(`CompileResult.traffic.scopes[].instructions`), and
`selection_stays_linear.rs`'s three existing tests on it
(`the_allocator_inserts_at_most_twice_what_was_scheduled`,
`doubling_a_kernel_at_most_doubles_what_is_emitted`,
`a_row_emits_bytes_in_proportion_to_its_scheduled_operations`) are all
*upper bounds* on or relative to the count itself. An undercounted or
permanently-zero counter satisfies every one of them — 0 is "at most"
anything — which is exactly what both mutations produce (a counter that
never leaves the ops it should have counted, or one `*=`'d from its initial
0 and so stuck there). None of the three is a floor, and nothing in the
file asserted one.

Added two exact-delta tests instead of another bound, in
`selection_stays_linear.rs`:

- `one_more_scheduled_operation_moves_the_total_scheduled_count_by_one`:
  compiles a chain of `n` and `n+1` plain arithmetic ops over the two
  coordinates at the same lattice shape, and asserts the total scheduled
  count differs by exactly 1.
- `a_second_sibling_fold_moves_the_total_scheduled_count_by_two`: compiles
  one `Reduce` fold alone against two sibling folds (distinct ranges, so
  arena hash-consing cannot dedupe them into one node) combined with an
  `Add`, and asserts the total differs by exactly 2 — one for the second
  fold's `Reduce` being opened (the same counting site, in its other match
  arm), one for the `Add` joining them.

Both are *differences* between two compiles of the same lattice shape, so
the lattice's own per-call fold overhead (identical on both sides) cancels
in the subtraction and the assertion needs no magic number tied to a
tier's lane width — verified by hand on both this host's tier (AVX-512)
and, via `PIXELFLOW_ISA=avx2`, the narrower one: `11→12`/`4→6` on AVX-512,
`11→12`/`4→6` on AVX2 — identical, as expected, since neither kernel's
shape is wide enough to depend on the tier's batch width. Re-swept after
adding them: both mutations caught (and, for free, line 147's *condition*
— see below).

**199's own `+=`/`*=`/`-=`, confirmed equivalent — dead under every current
backend.** The `Const` arm's guard is `if !self.b.ends_rematerializable()`:
it counts a constant only when the instruction that materializes it is
*not* one the allocator can cheaply re-place at each read. Every constant
a `LaneOp::Const` can produce lowers, on both `avx2.rs` and `avx512.rs`, to
exactly one of `Inst::Zero`, `Inst::Ones` or `Inst::LoadConst` — and
`IsaBackend::rematerializable` classifies all three as rematerializable on
both backends. So `ends_rematerializable()` is unconditionally `true`
immediately after any `ScheduledOp::Const`, the `!` is unconditionally
`false`, and the `+= 1` at line 199 never executes for any kernel on any
backend that exists today. Confirmed by `emit_traffic.rs`'s own
`a_selected_constant_is_counted_as_brought_in`, which asserts the opposite
of what a reachable line 199 would produce
(`sum(&with_constant, instructions) == sum(&without, instructions)` —
*not* counted) and is exactly what makes the sibling mutation at line
147/198 (the `if`'s own condition, `delete !`) a real catch: flipping the
condition makes the otherwise-always-false guard always-true, which *does*
make the body reachable and *does* violate that equality. Mutating the
body's own operator while the guard stays correct changes nothing, because
the body the operator lives in never runs. No test added — there is
nothing to test that would not be testing dead code; left for whoever adds
a backend whose constant materialization is not rematerializable, at which
point the branch becomes live and needs exactly the kind of test `register_file.rs`
got this pass.

**223, a real gap — deferred.** `guards` is a `Vec<IfGuard>` per scope
(`program/mod.rs`), so a scope can hold more than one, and the `&&`/`||`
difference is observable exactly when one scope has two `If`s whose guard
status differs: with `||`, any guard anywhere in the scope having a
guarded arm makes *every* `If` in that scope take the guarded path,
whether or not that `If`'s own guard says so. Nothing in the current test
suite has that shape where it is visible. `collapse_paths.rs`'s multi-`If`
tests (`a_guarded_if_nests_in_the_arm_of_another`,
`a_guarded_if_value_is_read_after_the_next_guarded_if`) do put two or three
`If`s in one scope, but check only the computed *value* — and a guarded
branch and a lane-wise blend compute the identical value by construction
(`guarded_if`'s three-way join is exactly the blend, routed through
branches instead of masked arithmetic), so no value-level test can ever
distinguish "guarded when it shouldn't be" from correct. `rows::TABLE`'s
three `If`-bearing kernels (`guarded_if_in_fold`, `binary_ops`,
`shift_muladd_blend`) each have exactly one `If` in its scope, so even the
byte-exact `GOLDEN`/`GOLDEN_SELECTED` pins never exercise the case where
two guards in one `guards` vector disagree. Closing this needs a kernel
with two `If`s sharing one scope — one whose mask is guard-worthy (e.g.
`x < 14.0`), one that structurally cannot be (e.g. `x < y`) — and a
byte-length or instruction-count assertion, not a value one, following
`golden_selected.rs`'s pattern but at a scope this series has not built a
fixture for yet. Carried to backlog rather than built here: a new golden
entry means new per-backend hashes in two files
(`emit::tests::sibling_folds::GOLDEN` and `golden_selected.rs`'s two
tables), which this pass chose not to do unverified against real
AVX-512/NEON hardware this sandbox does not have.

## Verified

- `cargo test -p pixelflow-codegen --lib`: pass, including the nine new
  `register_file::tests`.
- `PIXELFLOW_CODEGEN=selection cargo test -p pixelflow-codegen`: pass,
  including the two new `selection_stays_linear` tests.
- `cargo test --workspace`: pass.
- `cargo clippy --workspace --all-targets -- -D warnings`: clean.
- `cargo fmt --all -- --check`: clean.
- `cargo mutants -p pixelflow-codegen -f pixelflow-codegen/src/emit/register_file.rs`:
  23 mutants, 7 caught, 0 missed, 16 unviable.
- `PIXELFLOW_CODEGEN=selection cargo mutants -p pixelflow-codegen -f pixelflow-codegen/src/emit/select.rs`:
  55 mutants, 29 caught, 23 unviable, 3 missed (199's two variants,
  confirmed equivalent above; 223, real, deferred).

## Recommended next steps (not done here)

Backlog carried forward from 2026-09-16, plus this pass's one open item:

1. `pixelflow-codegen/src/emit/mod.rs` (7,289 lines) — untouched by any
   pass, and has grown since 09-16.
2. `pixelflow-codegen/src/emit/aarch64.rs` (2,601 lines) — untestable at
   the runtime-execution level from this x86_64 sandbox.
3. `pixelflow-core/src/backend/arm.rs`'s NEON impls — needs an aarch64
   host.
4. `pixelflow-codegen/src/emit/executable.rs`'s `macos.rs` submodule — no
   macOS host in this sandbox.
5. `core-term/tests/message_cuj_tests.rs` and the `ParserActor`/
   `TerminalApp` section of `core-term/tests/actor_roundtrip_tests.rs` test
   mock reimplementations, not `core_term`'s real types.
6. **New**: `select.rs` line 223 (`&&` vs `||` across a scope's guards) —
   needs a two-`If`-one-scope, one-guardable-one-not fixture and a
   byte/instruction-count assertion on it, plus new golden hashes in
   `emit::tests::sibling_folds::GOLDEN` and `golden_selected.rs` verified
   against real AVX-512 and NEON hardware. See the reasoning above for why
   a value-level test cannot see it.
