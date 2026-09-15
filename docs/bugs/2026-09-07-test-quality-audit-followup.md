# Test quality control follow-up — 2026-09-07

Scope: scheduled continuation of the test-quality-audit series
(`docs/bugs/2026-09-01-test-quality-audit-followup.md`'s backlog item 3):
`pixelflow-codegen/src/emit/avx2.rs` and `avx512.rs`, the AVX2 and AVX-512
`IsaBackend` encoders. `x86_64.rs` was already closed by an earlier,
unmerged pass (PR #1054); `mod.rs` (5,632 lines), `regalloc.rs`,
`aarch64.rs`, `traffic.rs`, `guards.rs` (partially covered by the also
unmerged PR #1154) and `executable.rs` remain untouched and are carried
forward below.

## STYLE.md compliance: test naming

Both files' runtime test modules (`tests::runtime`, JIT-and-execute style)
had non-compliant names — none read as a complete "it should ..." sentence,
several were flatly wrong once mentally prefixed (`it should binary ops`,
`it should high register`):

`avx2.rs`: `binary_ops`, `compare_lt`, `sqrt_and_neg_abs`, `select_blend`,
`const_broadcast_and_fma`, `fma_rounds_once`, `spill_frame_roundtrip`,
`gather_from_buffer`.

`avx512.rs`: `binary_ops`, `high_register`, `sqrt_op`, `neg_abs`,
`const_broadcast`, `fma_231`, `fma_rounds_once`, `gather_from_buffer`,
`spill_frame_roundtrip`.

Renamed every one to name the function under test and the behavior it
pins (e.g. `emit_binary_matches_the_scalar_reference_for_every_arithmetic_op`,
`emit_select_blends_if_true_and_if_false_by_the_mask`), following the
convention `x86_64.rs`'s own compliant names already established
(`selects_every_required_binary_op`, `encodings_match_the_manual`). Split
`avx2.rs`'s `const_broadcast_and_fma` — two unrelated assertions sharing one
name — into two tests, matching `avx512.rs`'s existing split of the same
pair. Also fixed a leftover `"fma sw"` failure-message tag in `avx2.rs` that
mislabeled real hardware FMA (this tier requires `+fma`, see the file's own
`compile_error!`) as a software fallback — a failing assertion would have
pointed the reader at the wrong mental model.

No test testing a private/internal item was found in either file: every
runtime test calls this module's own `pub`/`pub(crate)` `emit_*` functions,
consistent with "Test Public API."

## Mutation testing: `cargo-mutants` v27.1.0

Not present in this environment (consistent with every prior pass) —
installed via `cargo install cargo-mutants --locked`.

Per the 2026-09-01 pass's methodology fix, every run used the package's
unrestricted default test command (no `-- --test X`), so the crate's full
integration-test suite runs alongside the file's own tests.

### `avx2.rs` (`RUSTFLAGS="-C target-feature=+avx2,+fma"`)

First sweep: **169 mutants, 136 caught, 21 missed, 12 unviable.**

Two real gaps, both a public function with zero callers anywhere —
production or test:

- **`is_compare`** is `pub fn` (part of this crate's public API surface —
  `emit` and `avx2` are both `pub mod`) but nothing calls it: `emit_binary`
  dispatches through the private `cmp_pred` directly instead. Unlike
  `avx512.rs`'s `is_compare`, which its own `emit_plan` uses to route
  comparisons to `emit_compare`, this one had no coverage of any kind.
  Added a direct test over every `OpKind` arm.
- **`emit_movmskps_eax`** backs both of `Avx2Backend`'s short-circuit
  guards (`emit_skip_if_all_false`/`emit_skip_if_all_true`), but nothing in
  this file or the integration suite exercises those guard paths for the
  AVX2 tier specifically, so "replace `emit_movmskps_eax` with `()`"
  survived — the emitted `eax` would be left as whatever garbage a prior
  instruction happened to leave there. Added a direct JIT-and-execute test
  pinning the exact lanewise bitmask for a comparison with both true and
  false lanes present.

Re-run after the fix: **169 mutants, 139 caught, 18 missed, 12 unviable.**
Both real gaps gone; the 18 remaining are addressed below.

### `avx512.rs` (`RUSTFLAGS="-C target-feature=+avx512f,+avx512dq"`)

First sweep: **284 mutants, 229 caught, 41 missed, 13 unviable, 1 timeout.**

Five real gaps, all but one sharing a single root cause: every existing
test's `dst`/`idx`/`base` register stayed below `zmm16`/`r8`, so this
file's EVEX high-register extension bits (`R'`/`B`/`V'` — computed as
`((reg >> 4) & 1) ^ 1` and similar) were only ever exercised on their
"low" side, where inverting `0` with `^1` and forcing it to `1` with `|1`
happen to produce the same byte:

- **`Evex::rm`'s base-register `B` bit** — every production caller in this
  file addresses memory through `rsp` or `rax` (`frame_slot`,
  `RED_ZONE_CONST`, `emit_uniform_load`'s `emit_load_ptr_from_ctx` target),
  both `< r8`. Added a test that moves the incoming pointer into `r9` (not
  `rbp`/`r13` — those hit `mem_operand`'s RIP-relative special case
  instead of the ordinary `mod=00` form, caught by an early attempt at this
  fix) and round-trips a load/store through it.
- **`emit_gather`'s `R'`/`B`/`V'` bits** — the existing gather test
  addresses through `rdi` with `zmm13`/`zmm14`. Added a second gather test
  moving the base pointer into `r9` and gathering into/from `zmm20`/`zmm21`,
  mirroring the existing `emit_binary_writes_a_high_numbered_register`
  test's `zmm20` case for the ordinary binary-op path.
- **`emit_and`** is `pub fn` ("Bitwise helpers exposed for completeness /
  future mask emulation," per its own doc comment) with no caller anywhere
  — same shape as `avx2.rs`'s `is_compare` gap above. Pinned directly
  against the private `vandps` it wraps, whose own correctness is already
  proven by the `Abs` case of `emit_unary_negates_and_takes_the_absolute_value_of_every_lane`.
- **`emit_load_ptr_from_ctx`'s `&`/`<<`** — its only caller
  (`emit_uniform_load`) always passes `dst_gpr=rax(0)`, `ctx_gpr=rdi(7)`,
  values for which this function's bit ops happen to be indistinguishable
  from a wrong substitute (`0 << n == 0 >> n`; `7 & 7 == 7 | 7`). Added a
  byte-level encoding test with non-degenerate values (`dst_gpr=5`,
  `ctx_gpr=3`) that pins the exact output bytes.
- **`AVX512_FILE`'s `scratch`/`temps_for` fields** — the struct literal
  restates them explicitly (`scratch: RegSet::range(4, 28)`,
  `temps_for: super::temps_for`) rather than inheriting from `..SSE2_FILE`,
  because they genuinely differ from SSE2's shape (12 registers vs. 28,
  and a different per-op temp-count function — SSE2's accounts for a
  select-reload temp this backend's `vpternlogd`-based select doesn't
  need). Deleting either field from the literal — which is what the mutant
  does — silently falls back to the SSE2 values, and nothing before this
  pass asserted the restatement was actually taking effect: a kernel using
  enough live registers to need AVX-512's wider pool would still compile
  and run *correctly* with only twelve registers available, just with more
  spilling than intended, which no functional test can distinguish from
  "working as designed." Added a direct field-pinning test in a new
  `driver::tests` submodule (the struct isn't testable from the outer test
  module — `AVX512_FILE` is a private `const` inside `pub(crate) mod
  driver`).

Re-run after the fixes: **284 mutants, 242 caught, 28 missed, 13 unviable,
1 timeout.** All five real gaps gone; the 28 remaining are addressed below.
(The `fixed` field of `AVX512_FILE` is *not* one of the five: unlike
`scratch`/`temps_for`, it coincides with `SSE2_FILE`'s value — both are
`&[]` — so deleting that one field from the literal is a true equivalent,
not a gap; see below.)

### Equivalent mutants (46 total across both files after the fixes above)

Every remaining "missed" mutant in both files is a `|`↔`^` or `<<`↔`>>`
swap inside a byte-packing expression that ORs several bit *fields* into
one byte — `Vex`/`Evex`'s `rrr`/`rm`/`prefix` methods, `emit_gather`'s
`p0`/`p1`/`p2` construction, `emit_load_ptr_from_ctx`'s ModRM byte,
`AVX2_FILE`/`AVX512_FILE`'s restated-but-coincidentally-identical `fixed`
field — plus `avx2.rs`'s `Vex::rrr`/`Vex::rm`'s `(self.w as u8) << 7` term,
where `self.w` is `false` on every path this file ever constructs (the
AVX2 tier never sets EVEX/VEX.W).

Each OR-chain packs fields into **disjoint, non-overlapping bit
positions** by construction (e.g. `rrr`'s prefix byte: `R` at bit 7, `X` at
bit 6, `B` at bit 5, `R'` at bit 4 — never more than one field touches any
given bit). For disjoint-support values, `a | b`, `a ^ b`, and (for a
chain) any single link swapped between the two, are byte-identical for
*every* possible input — not just the values this test suite happens to
use. This is a structural property of the encoding, not a coverage gap:
no input, real or synthetic, could ever separate the mutant from the
original. `self.w as u8 == 0` always is the same kind of fact for the one
`<<`/`>>` case that isn't itself inside a disjoint-field OR-chain — shifting
a value that is always `0` in either direction is always `0`.

This is the same class the 2026-09-01 pass documented for
`pixelflow-core`'s `x86.rs` (`from_f32_scaled`'s `Default::default()`
equivalence) and the 2026-08-26 pass documented more generally — a
mutant `cargo-mutants` cannot distinguish from the original by
construction, as opposed to one no *test* happens to distinguish. No
further test was written for any of them.

## Verified

- `RUSTFLAGS="-C target-feature=+avx2,+fma" cargo test -p pixelflow-codegen
  --lib emit::avx2`: 11 passed, 0 failed.
- `RUSTFLAGS="-C target-feature=+avx512f,+avx512dq" cargo test -p
  pixelflow-codegen --lib emit::avx512`: 14 passed, 0 failed.
- `cargo clippy -p pixelflow-codegen --tests -- -D warnings` under both
  `+avx2,+fma` and `+avx512f,+avx512dq`: clean.
- `cargo fmt -p pixelflow-codegen -- --check`: clean.
- `cargo mutants -p pixelflow-codegen --file
  pixelflow-codegen/src/emit/avx2.rs` under `+avx2,+fma`: 0 real gaps
  (18/169 missed, all equivalent — see above).
- `cargo mutants -p pixelflow-codegen --file
  pixelflow-codegen/src/emit/avx512.rs` under `+avx512f,+avx512dq`: 0 real
  gaps (28/284 missed, all equivalent — see above).

## Recommended next steps (not done here)

Backlog carried forward from 2026-09-01, minus the item closed above:

1. `pixelflow-codegen/src/emit/mod.rs` (5,632 lines), `regalloc.rs` (2,699
   lines), `aarch64.rs` (2,553 lines, untestable from this x86_64 sandbox
   at the runtime-execution level the way `avx2.rs`/`avx512.rs` are, though
   its encoding-only assertions could still be mutation-tested), `traffic.rs`,
   `executable.rs` — untouched by any pass.
2. `pixelflow-codegen/src/emit/guards.rs` — partially covered by an
   unmerged draft, PR #1154 ("close `analyze_select_guards` mutation
   gaps"), open since 2026-09-04 and not `mergeable_state: behind` as of
   this pass; worth checking whether it should be landed before a future
   pass re-covers the same ground.
3. `pixelflow-core/src/backend/arm.rs`'s NEON impls — still untestable
   from every x86_64 sandbox this series has run in; needs an aarch64
   host.
4. Two other open PRs from this same series, #1049 and #1051, referenced
   by the 2026-08-26/2026-08-30/2026-09-01 passes as
   `mergeable_state: behind` on `pixelflow-search/egraph`'s `cost.rs` and
   `graph.rs`, no longer appear in the repository's open pull request list
   as of this pass — presumably closed or merged since; not independently
   re-verified here.
