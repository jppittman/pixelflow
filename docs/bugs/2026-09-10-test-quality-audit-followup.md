# Test quality control follow-up — 2026-09-10

Scope: scheduled continuation of the test-quality-audit series
(`docs/bugs/2026-09-07-test-quality-audit-followup.md`'s backlog item 1).
`pixelflow-codegen/src/emit/guards.rs` — that pass's other open item — was
merged since (PR #1154), closing it without a re-check needed here. Picked
the two smallest untouched files from the remaining backlog:
`pixelflow-codegen/src/emit/traffic.rs` (399 lines) and
`pixelflow-codegen/src/emit/executable.rs` (763 lines, plus its `linux.rs`
submodule). `mod.rs` (5,460 lines), `regalloc.rs` (2,985 lines) and
`aarch64.rs` (2,601 lines, runtime-untestable from this x86_64 sandbox) are
carried forward, same as every prior pass.

## STYLE.md compliance: test naming

`traffic.rs`'s existing tests were already compliant
(`every_emitted_byte_is_attributed_to_exactly_one_scope`,
`a_kernel_that_must_spill_reports_stores_and_loads`,
`the_scaffolds_traffic_does_not_move_with_the_pool`).

`executable.rs`'s `tests` module (the hand-assembled-instruction JIT tests,
gated to the plain-SSE2/NEON ABI) had six names that don't read as an "it
should ..." sentence: `jit_return_x`, `jit_add_xy`, `jit_complex_expr`,
`jit_const_05_raw`, `jit_return_x_x86`, `jit_add_xy_x86`. Renamed to name the
behavior each pins, e.g. `a_hand_assembled_kernel_with_only_a_ret_passes_x_through_unchanged`,
`a_hand_assembled_kernel_executes_chained_instructions_in_order`. The
`extent_tests` and `page_tests` modules, and `linux.rs`'s own `tests`
module, were already compliant.

No test testing a private/internal item outside its own file's model was
found: every new test below is against `traffic.rs`'s `Counting` decorator
or `executable.rs`'s `ExecutableCode`/`Extent2D`/`CodePage`, all central to
those files' own documented design (`Counting`'s module doc calls it out by
name; `CodePage`/`ExecutableCode` are the crate's public JIT-memory API). One
new test reads `ExecutableCode::capacity`, a field private to the crate but
not to the `page_tests` module that reads it — the same relationship
`MockCodePage`'s own (production) constructor already has to that field.

## Mutation testing: `cargo-mutants` v27.1.0

Not present in this environment (consistent with every prior pass) —
installed via `cargo install cargo-mutants --locked`. Every run used each
package's unrestricted default test command, per the 2026-09-01 pass's
methodology fix.

### `traffic.rs`

First sweep: **62 mutants, 29 caught, 27 missed, 6 unviable.**

All 27 were real gaps, not equivalents, falling into three groups:

- **`ScopeTraffic::memory_ops`/`EmitTraffic::dynamic_memory_ops`** (18
  mutants) — this cost-model arithmetic has no direct test at all; every
  existing test reaches it only incidentally through a full compile, which
  never asserts the formula's own shape. Added one test per function with
  distinct, non-degenerate field values chosen so every operator swap (`+`
  vs `-`/`*`, `*` vs `+`/`/`) and every whole-function replacement (`0`/`1`)
  produces a different total than the correct one.
- **`Counting`'s per-method counters** (7 mutants: `emit_plan`'s
  `Reload::Const` arm, `emit_resolve`'s `loads_kept`/`remats` arms,
  `slot_store`, `slot_load`) — the existing whole-kernel pressure test
  (`a_kernel_that_must_spill_reports_stores_and_loads`) only ever exercises
  the `Reload::FromStack` path and asserts `loads_transient + loads_kept >
  0`, an OR that a mutation on the untaken path or the unchecked counter
  slips through. Added direct unit tests against `Counting` itself, backed
  by a small `RecordingBackend` stub `IsaBackend` local to the test module,
  pinning each counter independently (a stack reload vs. a rematerialized
  constant; a spilled value vs. one already in a register).
- **`begin`, `scaffold_anchor`, `scaffold_finish`** (2 mutants) — the
  decorator's plain forwarding calls had no test at all. `begin` mattered
  most: aarch64's `begin` can genuinely return
  `CompileError::BudgetExceeded` (constant pool overflowing its 12-bit `LDR`
  offset), and `Counting::begin` silently swallowing that into `Ok(())`
  would be exactly the silent-failure class this codebase's error handling
  exists to rule out. `scaffold_anchor`/`scaffold_finish` matter on aarch64
  specifically, whose backend overrides both to seed/flush its literal pool.
  Added a `begin` test asserting the stub's `Err` comes back unchanged, and
  a forwarding-count test for the scaffold hooks.

Re-run: **62 mutants, 56 caught, 6 unviable, 0 missed.**

### `executable.rs`

First sweep: **38 mutants, 19 caught, 13 missed, 6 unviable.**

All genuine gaps:

- **`Extent2D::is_empty`** (5 mutants) — no test existed at all. Added one
  covering both zero-dimension cases and the non-empty case, which also
  discriminates `||` from `&&` and either `==` from `!=`.
- **`From<(usize, usize)>`/`From<[usize; 2]> for Extent2D`** (2 mutants) —
  untested conversions; pinned both against `Extent2D::new`.
- **`ExecutableCode::is_empty`** (1 mutant) — every existing test only
  constructs non-empty code (`from_code` itself refuses an empty buffer), so
  the zero-length case the method exists to report was never reached.
  Constructed one directly via `MockCodePage::map(..).finish(0)` — the same
  two calls `from_code`'s own default implementation is built from, minus
  the length check in front of them.
- **`CodePage::from_code`'s page-rounding arithmetic** (2 mutants) — the
  `- 1` in `(len + page_size - 1) & !(page_size - 1)` is exactly what keeps
  an exact multiple of the page size from spilling into a second page, and
  nothing observed the mapped `capacity` at that boundary (or at all).
  Added a test reading `ExecutableCode::capacity` — private to the crate,
  visible to the `page_tests` module that already sits inside it — at
  `page_size()` exactly and `page_size() + 1`.
- **`host_ret`'s hardcoded return value** (2 mutants, `page_tests::host_ret`
  itself, not production code) — every test that calls this helper only
  reads the mapped bytes back, never executes them, despite the helper's own
  doc comment claiming to be "a single `ret` for the host" whose validity
  `from_code`'s safety contract depends on. Added a test that actually
  calls the mapped code as a zero-argument function and returns cleanly —
  the property nothing else here checked.
- **`impl Drop for ExecutableCode`** (1 mutant, replacing `drop` with `()`)
  — a real, silent-leak-shaped gap: nothing in the crate observes whether
  the page is actually unmapped. First attempt (scanning `/proc/self/maps`
  for the freed address after `drop`) is unreliable: reading that file is
  itself a large-enough allocation to plausibly land on the just-freed
  single page before it's inspected, which looks identical to a real leak.
  Replaced with `mmap(..., MAP_FIXED_NOREPLACE)` at the exact freed address
  immediately after `drop`: the kernel hands the same address straight back
  if it is actually free, and refuses (`EEXIST`) if anything is still
  mapped there — the two outcomes a working vs. no-op `Drop` would produce,
  with no intervening allocation to confound them. Verified stable across 5
  repeated runs before and after adding the crate's other new tests
  (concurrent allocation activity from the rest of the suite was the
  specific risk). Added to `linux.rs`'s own `tests` module, since the
  property is platform-specific (`libc::mmap`/`MAP_FIXED_NOREPLACE`) the way
  its sibling tests already are.

Re-run: **38 mutants, 32 caught, 6 unviable, 0 missed.**

## Verified

- `cargo test -p pixelflow-codegen --lib emit::traffic::`: 12 passed, 0
  failed.
- `cargo test -p pixelflow-codegen --lib emit::executable::`: 18 passed, 0
  failed (run 5x to confirm the new `Drop` test is not flaky under repeated
  and concurrent execution).
- `cargo test -p pixelflow-codegen` (all targets incl. doctests): pass.
- `cargo test --workspace`: pass.
- `cargo clippy -p pixelflow-codegen --lib --tests -- -D warnings`: clean.
- `cargo fmt -p pixelflow-codegen -- --check`: clean.
- `cargo mutants -p pixelflow-codegen --file pixelflow-codegen/src/emit/traffic.rs`:
  62 mutants, 0 missed (56 caught, 6 unviable).
- `cargo mutants -p pixelflow-codegen --file pixelflow-codegen/src/emit/executable.rs`:
  38 mutants, 0 missed (32 caught, 6 unviable).

## Recommended next steps (not done here)

Backlog carried forward from 2026-09-07, minus the two items closed above:

1. `pixelflow-codegen/src/emit/mod.rs` (5,460 lines), `regalloc.rs` (2,985
   lines) — untouched by any pass.
2. `pixelflow-codegen/src/emit/aarch64.rs` (2,601 lines) — untestable from
   this x86_64 sandbox at the runtime-execution level `avx2.rs`/`avx512.rs`
   got, though its encoding-only assertions could still be mutation-tested.
3. `pixelflow-core/src/backend/arm.rs`'s NEON impls — still untestable from
   every x86_64 sandbox this series has run in; needs an aarch64 host.
4. `executable.rs`'s `macos.rs` submodule — untouched by this pass (no macOS
   host in this sandbox); its `linux.rs` sibling now has a `Drop`-unmapping
   test that `macos.rs` does not.
