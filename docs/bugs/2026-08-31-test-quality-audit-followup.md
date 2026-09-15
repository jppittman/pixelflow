# Test quality control follow-up — 2026-08-31

> **Mutation counts superseded, and two tests deleted (2026-09-08).** The
> `cargo mutants` figures below — 388 mutants, 334 caught, 39 documented
> equivalent, "0 real gaps" — were taken against a 1647-line `x86_64.rs`.
> `main`'s is now 1874 and structurally different: #1082 deleted
> `X86Backend::prologue`/`epilogue`, and #1177/#1183 replaced the
> fixed-scratch-register model with an allocator-managed pool. The mutant
> population is a different one and these tallies do not describe the tree.
>
> **Four of this pass's thirteen tests were deleted rather than ported**,
> for two different reasons.
>
> Two characterized `emit_binary_safe`, which no longer exists, in terms of
> `X86_SCRATCH` — one reserved `Reg(10)` — now `scratch: RegSet::range(4, 12)`
> reached as `plan.scratch.temp(n)`. There is no translation of "stashes
> `right` into the scratch register" into a world with no such register.
>
> Two more pinned **unreachable internal state**, which is not coverage:
> `Vex { w: true }` (every production construction goes through `Vex::new`,
> which hardcodes `w: false`), and `x86_redzone_disp(113)`'s internal
> out-of-range error (offsets advance in `vector_bytes` steps from 0, so
> they are multiples of 16, and `frame_ready` caps red-zone mode at
> `frame_size <= 128` besides — 112 is the largest reachable offset, and
> that assertion is kept).
>
> **The remaining nine survive and were re-verified on 2026-09-08** against
> current `main`: they build, they pass, and all five functions they target
> (`emit_movups_store`, `emit_load_ptr_from_ctx`, `emit_movmskps_eax`,
> `emit_cmp_eax_imm8`, `x86_redzone_disp`) still exist. Spot mutation check
> confirming they still bind rather than passing vacuously: biasing
> `x86_redzone_disp` by 8 instead of 16 kills 2, and flipping the `movups`
> store opcode from `0x11` to `0x10` kills 6.
>
> That is a spot check, not a sweep. **A full `cargo mutants` re-run against
> the current file is still owed** before "0 real gaps" is restated anywhere.

Scope: scheduled continuation of the test-quality-audit series
(`docs/bugs/2026-08-26-test-quality-audit-followup.md` and earlier).
`main` had not moved on that pass's open backlog since `9fce6cf` at the
time this pass started. Backlog item 3 (the actor-scheduler
`backoff_unit_tests` finding) was recommended for removal last pass as
already resolved twice over — this pass drops it and instead picks up
backlog item 2: `pixelflow-codegen/src/emit/*`, flagged as never
mutation-tested since the 2026-08-08 audit first noted it. Of that
directory, `x86_64.rs` (the x86-64 SSE/VEX raw instruction encoder) stood
out for having exactly one `#[test]` against 864 lines of byte-level
machine-code emission.

`cargo-mutants` v27.1.0 (freshly installed — not present in this
environment, consistent with every prior pass).

## First pass, and a mid-flight rebase

The first pass followed the established pattern: removed two genuinely
dead `pub fn`s (`emit_xorps`/`emit_andps` — zero callers anywhere in the
repo, confirmed by grep; `Neg`/`Abs` already go through the VEX
`emit_vxorps`/`emit_vandps`), then added ~35 byte-exact unit tests closing
118 real `cargo-mutants` misses down to 0, leaving 50 documented
equivalent mutants (all `replace | with ^` inside ModRM/VEX/REX byte
construction of the form `BASE | (field << shift)` — provably equivalent
because x86 instruction encoding packs bit-disjoint fields together by
definition, so OR and XOR of operands that can never share a set bit
compute the same byte for every input).

Before that work could be reviewed, **`main` merged eight commits
(#1055–#1062) that rewrote the entire `pixelflow-codegen/src/emit`
module** — a deliberate refactor replacing the raw byte-pushing SSE/VEX
helpers with a shared `Vex`/`Digit` builder abstraction, unifying
register allocation behind one trait, and moving the collapse-loop driver
logic into the per-ISA files. `x86_64.rs` grew from 864 to 1647 lines and
gained a whole new `X86Backend` driver + general-purpose-register
assembler section. All ~35 of this pass's first-draft tests targeted
functions that refactor deleted or reshaped; none of them could be merged
as written.

Resolved by merging `main` into this branch, taking upstream's
`x86_64.rs` wholesale (discarding the first-draft tests), and starting
the mutation-testing pass over against the current file.

## Second pass: right-sized to the new architecture

A naive re-application of the first pass's method — byte-exact unit tests
for every function `cargo mutants` flags — would have meant writing
~15 more tests pinning `X86Backend`'s trait-method plumbing (`frame_alloc`
calls `emit_sub_rsp`, `latch_bounds` calls `scaffold::latch_bounds`, and
so on for a dozen one-line delegations). That is exactly the failure mode
this pass had just lived through: tests wired tightly to *which private
function calls which other private function* broke wholesale the moment
the implementation was reshaped, for no behavioral reason. Mid-session
user guidance landed on this point directly: write tests at API
boundaries, not against implementation plumbing, so they survive a
refactor instead of being collateral damage from one.

Concretely, `cargo mutants -p pixelflow-codegen --file .../x86_64.rs -- --lib`
(the filter every prior pass in this series has used, to dodge slow
crate-wide baselines) reported the whole `X86Backend`/`scaffold` driver
surface — 21 mutants across `frame_alloc`, `frame_free`, `slot_store`,
`slot_load`, `latch_bounds`, `counter_clear`, `counter_step`,
`branch_if_counter_done`, `advance_out`, `store_result`, `add_scalar`,
`emit_ret`, `body_frame_bytes`, `emit_store` — as missed. But
`pixelflow-codegen/tests/collapse_loop.rs` already exists as an 11-test
integration suite that compiles and *executes* collapse-loop kernels
(the exact driver path all 21 of those methods serve) and checks real
numeric output against a scalar oracle. It just isn't part of `--lib`, so
none of the prior mutants runs in this series — including this pass's
first sweep — ever saw it. Re-running with `-- --lib --test
collapse_loop` dropped the miss count from 93 to 61, and **every one of
those 21 driver/scaffold mutants disappeared** — proof they were never a
real gap, only invisible to the test filter. No unit test was added for
any of them; the fix was including the boundary test that already existed.

### Fixed: 15 new tests, at the file's real API boundary

What survived even with `collapse_loop.rs` included were genuine logic
bugs, each kept as close to "given these bytes in, what bytes come out"
(or, for the two driver-level boundary conditions, "given this frame
size, does the prologue/epilogue touch the stack") as this file's
structure allows:

- **`Vex::head`'s W-bit shift** (`<<` vs `>>` only diverge when `w=1`,
  which no current caller passes — so, matching this series' precedent of
  testing an encoder primitive directly when no production call site
  happens to exercise a bit, `Vex { w: true, .. }` is constructed
  directly in the test).
- **`emit_movups_store_base`'s REX.R/B computation** — all four
  high/low register combinations, which also pins the `||` in its guard
  condition.
- **`emit_load_ptr_from_ctx`**, **`emit_movmskps_eax`**,
  **`emit_cmp_eax_imm8`** — this file's byte-in/byte-out contract for
  functions the collapse-loop suite doesn't happen to exercise with the
  right register/offset combination to make their bit-manipulation
  mutants observable.
- **`x86_redzone_disp`'s negation and disp8-overflow boundary** (a new
  function this refactor introduced; boundary case chosen so the offset
  that produces exactly `-128` — still representable — is distinguished
  from the one producing `-129`).
- **`emit_binary_safe`'s aliasing decision** (also new): `dst == left ||
  dst != right` is the one condition deciding whether the two-operand SSE
  hazard needs a scratch stash. Two cases — all three registers aliased,
  and `dst` aliasing only `right` — cover all three mutated operators
  (`||`→`&&`, `==`→`!=`, `!=`→`==`).
- **`X86Backend::prologue`/`epilogue`'s `frame_bytes > 0` boundary**: a
  real gap invisible to *any* execution/value test, including
  `collapse_loop.rs` — the wrong branch (`>=`) only emits a harmless dead
  `sub/add rsp, 0` pair at `frame_bytes == 0`, never a wrong answer, so a
  value-comparison oracle structurally cannot see it. This is the one
  place in this pass where a direct construct-and-call test on
  `X86Backend` was justified: not because the trait method needed pinning
  as plumbing, but because the property under test (no dead bytes) has no
  other observable surface.

All new tests are byte-exact assertions, following this file's own
existing precedent (`selects_every_required_binary_op`,
`gpr_tests::encodings_match_the_manual`) rather than reaching into fields
— testing private free functions and `pub(crate)` types directly from
this file's own `mod tests`/nested `driver::tests` is testing the
module's real contract, not crossing an API boundary, since there is no
higher-level structure inside the file itself that would exercise the
same logic more precisely than the byte sequence itself.

### Final state: 388 mutants, 334 caught, 39 documented equivalent, 15 unviable, 0 real gaps

The 39 remaining misses are all `replace | with ^`, and every one is the
same equivalence class documented in this pass's first draft and the
2026-08-26 audit's `Round /2.0`↔`*2.0` precedent: x86 ModRM/VEX/REX bytes
are bit-disjoint fields packed together by definition, so OR and XOR of
operands that can never share a set bit are unconditionally the same
function. Confirmed by inspection at every site (`Vex::rrr`/`rr`/
`digit_rm`/`rr_gpr`/`rm_scaled4`/`head`, `emit_sse_rr`, `emit_f32_const`,
`emit_load_ptr_from_ctx`, the four rsp-relative spill emitters,
`emit_movups_store_base`'s remaining `|`, `emit_cmp_tail`,
`emit_movmskps_eax`, and the new GPR assembler's `rex_w`/`modrm_rr`); a
comment at the top of the encoding tests records this so a future pass
doesn't re-chase them.

## Verified

- `cargo test -p pixelflow-codegen` (all targets incl. doctests, and the
  `collapse_loop.rs` integration suite): passed, 0 failed.
- `cargo build --workspace`: clean.
- `cargo clippy -p pixelflow-codegen --lib --tests`: clean.
- `cargo fmt -p pixelflow-codegen -- --check`: clean.
- `cargo mutants -p pixelflow-codegen --file
  pixelflow-codegen/src/emit/x86_64.rs -- --lib --test collapse_loop`:
  388 mutants, 334 caught, 39 documented equivalent, 15 unviable, 0 real
  gaps.

## Methodology note for future passes in this series

**Widen the mutants test filter before writing new tests, not after.**
Every prior pass in this series scoped `cargo mutants` to `-- --lib` to
dodge slow crate-wide baselines (a real concern — see the `cost.rs`
precedent). But `--lib` silently excludes integration test targets
(`tests/*.rs`), and this pass found a whole 21-mutant category that only
existed because of that exclusion, not because of any actual coverage
gap. Before writing tests against a mutant report, check whether the
crate has an integration-test target plausibly covering the flagged code
(`cargo test -p <crate> --test <name>` — usually named after the feature
area) and include it in the mutants filter (`-- --lib --test <name>`) if
its own runtime is small. Only fall back to a narrower `--lib`-only
scope, as `cost.rs` did, when even that combined suite's baseline is slow
enough to threaten the pass's time budget.

## Recommended next steps (not done here)

1. `pixelflow-codegen/src/emit/` has four more files never mutation-tested
   at this granularity: `executable.rs`, `regalloc.rs`,
   `aarch64.rs` (the NEON counterpart to this pass's
   `x86_64.rs` — likely has the same "byte-exact vs. execution-only" gap
   pattern, now doubly worth checking against its own integration-test
   coverage first per the methodology note above), and `mod.rs` (huge —
   needs splitting into several scoped mutants runs).
   **`avx2.rs` and `avx512.rs` came off this list on 2026-09-08:** #1199
   ran the pass over both and recorded zero real gaps in
   `docs/bugs/2026-09-07-test-quality-audit-followup.md`. Leaving them here
   would send the next scheduled pass back over finished work.
2. `pixelflow-core/src/backend/x86.rs`'s AVX2/AVX-512 (`F32x8`/`F32x16`/
   `U32x8`/`U32x16`/`Mask8`/`Mask16`) impls and `arm.rs`'s NEON impls —
   flagged since 2026-08-26 as never unit-tested under a build that
   actually activates those ISA levels. Needs `xtask isa-matrix`-style
   target-feature setup before `cargo mutants` can even run against them.
