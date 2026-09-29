# Test quality control follow-up — 2026-09-19

Scope: scheduled continuation of the test-quality-audit series
(`docs/bugs/2026-09-10-test-quality-audit-followup.md`'s backlog item 2).
`pixelflow-codegen/src/emit/mod.rs` (7,355 lines) and `regalloc.rs` (4,243
lines) are now large enough that a full-file `cargo-mutants` sweep would be
infeasible in one pass (the same reasoning the 2026-07-20 doc used to scope
its `pixelflow-core` subset), so this pass picked the third item instead:
`pixelflow-codegen/src/emit/aarch64.rs`, flagged in every prior doc as
"untestable from an x86_64 sandbox at the runtime-execution level the way
avx2.rs/avx512.rs are, though its encoding-only assertions could still be
mutation-tested." That is exactly this file's shape: `pub mod aarch64;` in
`emit/mod.rs` is **not** `#[cfg(target_arch = "aarch64")]`-gated, so the whole
file — the `Inst` encoder, the `Aarch64Backend` `IsaBackend` driver, and the
branch/register helpers — compiles and runs its own unit tests on every host,
including this one; only *executing* the emitted machine code needs a real
aarch64 CPU, and nothing here does that.

`cargo mutants --list -p pixelflow-codegen --file
pixelflow-codegen/src/emit/aarch64.rs` reports **737 mutants** for the file
(3,052 lines before this pass), well past the ~300–400 rule of thumb. Of
those, **336** sit in `disassemble_code`/`decode_aarch64_mnemonic`/
`dump_jit_asm` — a debug-only, human-readable disassembler used for `--dump-asm`-style
inspection, not on any code-generation path — so it was excluded from this
pass's mutation scope the same way prior passes deprioritized `Debug`/`Display`
`fmt` text (2026-07-20 doc) and named the disassembler out explicitly in the
2026-09-01 doc's methodology fixes. The remaining **401 mutants** — the `Inst`
enum and its encoders, `emit_fmov_imm`/`try_encode_fmov_imm8`/
`needs_const_pool`, the shift/gather/binary emitters, the whole
`Aarch64Backend` driver (`ConstPool`, every `IsaBackend` method,
`emit_instruction_plan`), and the branch/register helpers (`movz`/`cmp`/
`mvn_w`/`add`, `B`/`BCond`/`CbzW16`/`AdrpAdd`, `DispField`) — are this pass's
scope, and are now fully closed. `aarch64/table.rs` (a separate 1,004-line
file, `aarch64`'s own submodule) and the excluded disassembler are carried
forward, along with the rest of the standing backlog.

## Rebased onto the H6 emitter — the driver counts below are history

This pass was measured against `aarch64.rs` at `d36dc79` (3,052 lines).
#1283 (H6 step 5) then deleted the collapse scaffold, and with it six of the
`IsaBackend` verbs this pass had just covered. **The 401-mutant sweep and its
"0 real gaps" result describe the pre-H6 driver and are not a current
statement about `aarch64.rs`.** A fresh sweep is carried forward.

What the merge with `main` required:

- **`branch_if_counter_done_compares_then_branches` deleted.** `Counter` is
  gone; the collapse's trip counts are folds the emitter emits as loops, so
  there is no counter to compare against.
- **Six stanzas dropped** from the driver forwarding test — `counter_clear`,
  `counter_step`, `store_result` and both `advance_out` steps — for the same
  reason, along with `Counter`/`OutStep` themselves.
- **That test renamed.** It was `every_plain_leaf_isa_backend_method_emits_its_instruction`,
  true when written and false now: the trait has 23 methods and this test
  reaches 8. `emit_write`, `test_ge`, `scope_begin`/`scope_end`, `emit_plan`,
  `emit_resolve` and `frame_ready` are a carried-forward gap, recorded in the
  test's own doc comment. `emit_write` takes a row, a column, a lane and a
  width, so it is not a plain leaf and wants its own shape of test.
- **`scaffold_anchor`/`scaffold_finish` are `anchor`/`finish`** — a rename,
  and the two tests pinning the ADRP/ADD pair and the 16-byte pool padding
  port unchanged apart from the name.
- **`PoolEntry` is `[u32; 4]`, not `u32`.** A pool entry is a whole 128-bit
  register now (a splat is the common case, the lattice's iota the other), so
  `emit_pool_entry`, `ConstPool::offset_for` and `pool_entries()` are keyed and
  asserted on four words. The dedup, offset and overflow tests are otherwise
  unchanged — what they pin did not move.

No production code changed here either, and the surviving tests still pass:
252 `pixelflow-codegen` lib tests green.


## STYLE.md compliance: test naming

`aarch64.rs`'s three existing test modules (`tests`, `label_tests`,
`xr_tests`) had a mix of compliant and non-compliant names. `label_tests` and
`xr_tests` were already fully compliant (`encodings_match_the_manual`,
`a_conditional_writes_imm19_and_keeps_its_condition`, etc. — full sentences on
their own, the accepted variant `docs/STYLE.md` and the 2026-09-07 doc both
recognize). `tests` had fourteen names that were cryptic labels rather than
sentences — a mnemonic and nothing else, the same shape the 2026-09-07 doc
flagged in `avx2.rs`/`avx512.rs` (`compare_lt`, `binary_ops`):

`fmov_imm8_common_values`, `fmov_imm8_roundtrip`,
`emit_fmov_imm_fallback_for_non_encodable`, `gather_primitive_encodings`,
`disassemble_ret`, `disassemble_fadd`, `disassemble_ushr`, `disassemble_shl`,
`disassemble_mov_vec`, `disassemble_sequence`, `disassemble_zero_const`,
`disassemble_ldr_str`, `disassemble_code_empty`, `disassemble_code_short_chunk`.

Renamed to name the behavior each pins, e.g.
`try_encode_fmov_imm8_matches_the_arm_arm_for_common_encodable_and_non_encodable_values`,
`emit_fmov_imm_falls_back_to_movz_movk_dup_for_non_encodable_values`,
`disassembly_of_a_ret_names_it_ret`,
`disassembly_names_every_instruction_in_a_multi_instruction_sequence`.
`disassemble_offsets_are_sequential` was already a complete sentence
("disassemble offsets are sequential") and was left as-is.

No test reaching into a private item outside the file's own model was found.
Every test — including the new ones below — calls `aarch64.rs`'s own
`pub`/`pub(crate)` functions and types (`Inst`, `ConstPool`, `Aarch64Backend`
via the crate-internal `IsaBackend` trait, `MaskTest`/`Counter`/`OutStep`,
`ScheduledOp`), the same "central to this file's own design" carve-out the
2026-09-07 and 2026-09-10 docs used for `AVX512_FILE` and `Counting`. The new
`driver::tests` submodule follows the exact precedent `avx512.rs`'s own
`driver::tests` set: a private submodule nested inside `driver` itself, since
`Aarch64Backend`, `ConstPool` and `AARCH64_FILE` are private to that module and
not visible from the file-level `tests` module.

## Mutation testing: `cargo-mutants` v27.1.0

Already installed from a prior pass's `cargo install cargo-mutants --locked`;
version unchanged. Every run used the package's unrestricted default test
command (no `-- --test X`), per the 2026-09-01 pass's methodology fix, and no
special `RUSTFLAGS` were needed — `aarch64.rs` compiles under the plain
`x86_64-unknown-linux-gnu` target this sandbox already uses.

### First sweep (401 mutants, scope as above)

`cargo mutants -p pixelflow-codegen --file pixelflow-codegen/src/emit/aarch64.rs
-E '(disassemble_code|decode_aarch64_mnemonic|dump_jit_asm)'`:

**119 missed, 227 caught, 51 unviable, 4 timeouts** (22 minutes).

Categorizing every missed/timeout mutant:

- **`Inst::encode`'s `BCond` arm, `emit_into`'s `Mov`/`AdrpAdd` arms, `<impl AsmInsn for AdrpAdd>`'s `label_ref`** —
  five real gaps sharing one shape: an `Inst`-wrapper code path (reached only
  when an instruction is boxed into the `Inst` enum via `.into()` and driven
  through `<Inst as AsmInsn>`, rather than through its own concrete type's
  `AsmInsn` impl directly) that no existing test ever took. `AdrpAdd`'s own
  `emit_into`/`label_ref` are pinned directly by `adrp_add_reaches_across_pages`,
  but every one of those tests pushes a bare `AdrpAdd` value into an
  `Assembly` — a call that's monomorphized against `AdrpAdd`'s own impl and
  never routes through `Inst`. Nothing had ever constructed
  `Inst::from(AdrpAdd { .. })` and driven *that*. The gap was real:
  `<impl AsmInsn for Inst>::emit_into`'s wildcard fallback
  (`_ => emit32(code, self.encode())`) calls `Inst::encode`, whose own
  `AdrpAdd` arm **panics** ("must be emitted via `emit_into` or
  `AsmProgram`") — so deleting the `AdrpAdd` arm from `emit_into` silently
  turned a working two-instruction emission into a panic, and deleting the
  `label_ref` arm would have left the `ADD`'s immediate at its unpatched `#0`
  instead of the real page offset. Similarly, `emit_into`'s `Mov` arm elides a
  move to the same register (`if dst != src`); the wildcard fallback does not
  know to, so losing that arm turns every no-op move into a wasted `ORR`.
  Added `an_adrp_add_wrapped_as_an_inst_still_reaches_its_target` (in
  `label_tests`, pushing `Item::Inst(AdrpAdd { .. }.into())` and checking the
  patched `ADD` immediate), `emit_into_elides_a_mov_to_the_same_register`, and
  `inst_encode_gives_a_branch_its_fixed_word_or_its_condition` (calling
  `Inst::encode` directly on `B`/`CbzW16`/`BCond` values, which — like
  `avx2.rs`'s `is_compare` in the 2026-09-07 pass — is `pub fn` but had no
  caller of any kind).
- **`emit_fmov_imm`'s three-instruction fallback (MOVZ+MOVK+DUP)** — the
  single biggest real gap. `emit_fmov_imm_falls_back_to_movz_movk_dup_for_non_encodable_values`
  (the prior name) only ever checked `code.len() == 12`; nothing checked the
  bytes. Every bit-packing operator in the fallback (`bits & 0xFFFF`,
  `bits >> 16`, and the `|`/`<<` in each of the three emitted words) could be
  replaced with a wrong one and the test still passed. Added
  `emit_fmov_imm_general_case_encodes_the_exact_movz_movk_dup_sequence`,
  independently recomputing each word from the documented formula and
  comparing byte-for-byte.
- **`try_encode_fmov_imm8`, `needs_const_pool`, `emit_pool_entry`,
  `emit_dup_lane0`** — `needs_const_pool` (`bits != 0 && try_encode(..).is_none()`)
  had no direct test at all; `aarch64_const_pool_appends_across_bodies`
  (`mod.rs`) calls it only with values that are already `true`, so `replace
  needs_const_pool -> bool with true` and `&&` → `||` both survived. Added
  `needs_const_pool_is_false_for_zero_and_fmov_encodable_values_and_true_otherwise`.
  `emit_pool_entry` (splats an f32 bit pattern into 16 bytes) and
  `emit_dup_lane0` had zero coverage of any kind — both `pub fn`s reachable
  only through the driver, whose own tests (before this pass) never asserted
  their output shape. Added direct tests for both.
- **`emit_ushr`** — every existing call (`disassemble_ushr`, now
  `disassembly_of_ushr_decodes_the_shift_amount_and_element_size`) uses
  `dst = src = v0`, which masks a real gap: `emit_ushr`'s bit-packing operator
  for the `src` register (`0x6F200400 | dst | (src<<5) | (immhb<<16)`)
  reduces to indistinguishable results when `src == 0`. Added
  `emit_ushr_places_the_registers_and_shift_amount_in_their_own_fields` with
  distinct nonzero `dst`/`src` (3, 7), decoding each field independently.
- **`add`'s `PtrReg` operand, the free `mvn_w`, `movz`'s shift direction** —
  three variants of the same "pub item nobody calls" shape as `avx2.rs`'s
  `is_compare` and `avx512.rs`'s `emit_and` in the 2026-09-07 pass.
  `add`'s three-way `AddOperand` dispatch (`Imm12`/`Gpr`/`PtrReg`) was only
  ever exercised with the first two; `add_accepts_a_ptr_reg_operand_the_same_way_it_accepts_the_equivalent_gpr`
  cross-checks the `PtrReg` path against the equivalent `Gpr` call rather than
  a hand-picked constant. The free `pub fn mvn_w` ("bitwise helpers exposed for
  completeness") has no caller anywhere in the crate (`Inst::mvn_w`, a
  different, unrelated constructor, is what production code actually calls);
  added `mvn_w_computes_bitwise_not_via_orn_with_wzr`. `movz`'s only existing
  calls use `#0`, masking its `<<`/`>>` bit exactly like `emit_ushr`'s `src`;
  added `movz_places_a_nonzero_immediate_at_bits_5_through_20`.
- **`temps_for`/`gpr_temps_for`** — both dispatch functions (which `OpKind`s
  need a scratch register beyond their operands) had zero direct coverage;
  only exercised incidentally through full compiles that never isolate a
  specific `ScheduledOp` variant. Added
  `temps_for_asks_for_a_temp_only_for_rsqrt_recip_gather_and_reduce` and
  `gpr_temps_for_asks_for_three_for_gather_one_for_uniform_and_none_otherwise`,
  covering every match arm including the `Reduce` one (built with
  `pixelflow_ir::fold::{Binder, Fold, Monoid}`, mirroring `mod.rs`'s own
  `a_surviving_reduce_compiles_and_runs` fixture).
- **`emit_unary`/`emit_shift_imm`/`emit_binary`** — the three op-dispatch
  functions the driver's `alu`/`ShiftImm`/`Unary` cases forward to. Each
  `-> ()` stub survived because every existing exercise of them goes through a
  full kernel compile whose length assertions (`len > 0`, `is_multiple_of(4)`)
  don't notice one specific operator's instruction going missing when
  surrounding instructions still emit bytes. Added one direct test per
  function.
- **The `Aarch64Backend` driver — untested as a unit before this pass.**
  Fourteen `IsaBackend` methods (`jump`, `emit_mov`, `emit_store`, `slot_store`/
  `slot_load`, `counter_clear`/`counter_step`, `store_result`, `advance_out`,
  `add_scalar`, `load_const`, `alu`, `emit_ret`, `scaffold_anchor`) are
  one-line forwarding calls with no test anywhere pinning that they emit
  *something* rather than nothing; the shared `emit_collapse_loop` tests in
  `mod.rs` check total scaffold length across all four backends, which does
  not attribute a missing instruction to one specific method. Added
  `every_plain_leaf_isa_backend_method_emits_its_instruction` (one call per
  method, checking length) plus dedicated tests for the two with
  Assembly/label plumbing (`jump_emits_an_unconditional_branch`,
  `branch_if_counter_done_compares_then_branches`) and the one with two
  distinguishable arms (`branch_if_arm_is_dead_reduces_with_the_arms_own_instruction_count`
  — `SelectArm::True` reduces with `UMAXV`+`FMOV`+`CBZ` (12 bytes),
  `SelectArm::False` with the extra `MVN` (16 bytes)).
  `ConstPool` (deduplication by bit pattern, the 4096-entry `LDR`-offset
  overflow, `offset_for`, `is_empty`) had no test at all; added
  `const_pool_dedups_by_bit_pattern_and_offsets_by_16_bytes` and
  `const_pool_refuses_a_4096th_distinct_entry`.
  `frame_alloc`/`frame_free` (moving `sp` in `MAX_ADD_IMM`-sized chunks) had
  no test; two of their four missed mutants (`>` → `<`/`==`) were plain
  misses, the other two (`>` → `>=`, `-=` → `/=`) were **timeouts** — an
  infinite loop, since an unsigned `remaining >= 0` never becomes false and
  `remaining /= chunk` gets stuck at 1. Added
  `frame_alloc_and_frame_free_move_the_stack_pointer_in_chunks_of_at_most_max_add_imm`,
  which closes the two plain misses; the two timeouts are addressed below
  (they are not something a passing/failing test converts — see "Equivalent
  mutants" is the wrong bucket for them too, since a hang *is* a detected
  divergence, just categorized separately by `cargo-mutants`).
  `scaffold_anchor`/`scaffold_finish` (the `AdrpAdd` anchor and the pool's
  16-byte-aligned append) had no direct test; `scaffold_finish`'s `delete !`
  mutant flips "pad while *not yet* aligned" to "pad while *already*
  aligned" — invisible unless the pool is made to start mid-alignment. Added
  `scaffold_anchor_emits_an_adrp_add_and_scaffold_finish_binds_an_empty_pool`
  and `scaffold_finish_pads_the_pool_to_a_16_byte_boundary` (the latter
  pushes one stray byte after the anchor so the pool starts unaligned, then
  checks the 16-byte boundary and the four splatted copies of the pooled
  constant).
  `begin` (seeding the pool from a schedule, and refusing one that would
  leave the trailing builtins with no headroom) had one test's worth of
  indirect coverage (`aarch64_const_pool_appends_across_bodies` in `mod.rs`)
  but nothing exercising it in isolation, nothing with a *skipped* constant
  (an FMOV-encodable one, to catch `&&` → `||`), and nothing at the
  `BUILTIN_HEADROOM` boundary (128 entries below the 4095-entry `LDR` limit),
  where `+` → `*` and `>` → `==`/`<`/`>=` all survive on a small schedule.
  Added `begin_seeds_the_pool_from_constants_that_need_it_and_skips_ones_that_dont`
  and two boundary tests pinning the exact edge (3,967 constants: accepted;
  3,968: `CompileError::BudgetExceeded`).
  `emit_instruction_plan`'s `Gather` arm multiplies the buffer slot by
  `PTR_BYTES` (8) to find its context-array offset; every gather this crate
  has ever scheduled uses slot 0 or 1, where `*`, `+` and a stray `/` by 8 all
  agree. Added `emit_instruction_plan_scales_the_gather_slot_by_pointer_size_not_adds_or_divides`,
  constructing an `InstructionPlan`/`ResolvedOp::Gather` directly with
  `regalloc::Scratch::for_test_with_classes` (the `#[cfg(test)]` constructor
  built for exactly this) and slot `2`, checking the resulting `LDR X9,
  [X0, #16]` against `Inst::ldr_x` rather than a hand-computed word.

Re-run after all of the above: **401 mutants, 43 missed, 303 caught, 51
unviable, 4 timeouts.** Zero real gaps remain; the 43 missed and 4 timeouts
are addressed next.

## Equivalent mutants (43, plus 4 timeouts)

Every remaining missed mutant is a `|` ↔ `^` (or, in two cases, `&` ↔ `|`/`^`)
swap inside a byte-packing OR-chain, the same disjoint-bitfield class the
2026-09-07 doc documented for `avx2.rs`/`avx512.rs` — **with one added
subtlety this pass had to work through by simulation rather than by
inspection.**

**`cargo-mutants` mutates the operator token in place, textually — it does not
re-parenthesize.** For a flat chain of same-precedence operators
(`A | B | C | D`, which Rust parses left-associatively as
`((A | B) | C) | D`), swapping one `|` for `^` — a *higher-precedence*
operator — does not preserve that grouping. `A | B | C ^ D` reparses as
`A | B | (C ^ D)`: the mutated operator binds only to its immediate two
text-adjacent operands, not to the whole accumulated left-hand chain. A
manual first pass over this file's mutants (recorded and then corrected
during this pass) assumed grouping was preserved and misjudged several
`emit_ushr`/`mvn_w` mutants as real gaps; running the actual mutation and then
independently simulating the *reparsed* expression in Python against 20,000
random register/immediate values per site — the same "no input, real or
synthetic, could ever separate the mutant from the original" bar the
2026-09-07 doc set — showed every one of them equivalent once the correct
grouping is used. Concretely, for `emit_ushr`'s `0x6F200400 | dst | (src<<5) |
(immhb<<16)`, mutating the *last* `|` to `^` does **not** XOR `immhb<<16`
against the whole running word (which would flip the shift-immediate's
already-set high bit and be a real, detectable bug) — it reparses as
`0x6F200400 | dst | ((src<<5) ^ (immhb<<16))`, and since `src<<5` (bits 5–9)
and `immhb<<16` (bits 16–21) are themselves disjoint, that inner XOR equals
the inner OR for every input, and the whole expression is unchanged.

With that correction, every remaining mutant falls into one of these
provably-disjoint (or otherwise provably-redundant) shapes, each confirmed by
direct 20,000-trial random simulation of the *actual* reparsed expression:

- **Disjoint-bitfield OR-chains** (the bulk): `Inst::encode`'s and
  `BCond::emit_into`'s `0x5400_0000 | condition` (condition is 4 bits, the
  constant's low nibble is `0`); `emit_fmov_imm`'s MOVI/FMOV-imm8/MOVZ/MOVK/DUP
  words (`dst`, `abc<<16`, `defgh<<5`, `lo16<<5`, `hi16<<5`, and the DUP's
  `16<<5` register-literal are each confined to bit ranges the constant
  leaves at `0`); `try_encode_fmov_imm8`'s 8-bit `imm8` construction (each of
  `a`..`h` is a single bit shifted into its own distinct position); `AdrpAdd`'s
  `emit_into`/`label_ref` (`rd`, `rd<<5`, `immlo<<29`, `immhi<<5`,
  `within_page<<10` all land in the ARM ARM's disjoint encoding fields);
  `movz`/`cmp`/`mvn_w` (`imm<<5`/`dst`, `rhs<<16`/`lhs<<5`/`31`,
  `src<<16`/`dst` respectively); `emit_ushr`/`emit_shl`'s `dst`/`src<<5`/
  `immhb<<16` (verified with the reparse correction above — `emit_shl`'s
  `immhb` occupies a bit range the constant genuinely leaves clear, unlike a
  first, incorrect reading of `emit_ushr`'s).
- **`DispField::write`'s `(existing & !mask) | field`** — `field` is
  constructed as `(words << shift) & mask` two lines above, so it is
  *structurally* confined to `mask`'s bits, and `existing & !mask` to the
  complementary bits; the two operands can never share a set bit, so `|` and
  `^` agree for every possible `existing`/`mask`/`field` (not just the ones
  this suite happens to construct).
- **`DispField::write`'s `code[at + 1]`** (`+` → `*`, i.e. `code[at]` read
  twice) — the byte this reads is `existing`'s second byte, which falls
  entirely inside both `DispField`s this crate uses (`IMM19`: bits 5–23;
  `IMM26`: bits 0–25) and is therefore unconditionally overwritten by
  `(existing & !mask) | field` two lines later, regardless of what value was
  read. A wrong read of a value that is discarded before it is used is
  invisible by construction.
- **`try_encode_fmov_imm8`'s `bits & 0x7FFF_FFFF == 0` (the "±0.0 is not
  encodable" guard)** — provably redundant, not merely untested: the function's
  *other* guard three lines later (`(bits >> 25) & 0x1F != rep5`) independently
  rejects both values this check exists for (`0` and `0x8000_0000`), because
  both have `bits[29:25] == 0` while `rep5` is forced to `0x1F` whenever
  `NOT(bit 30) == 1` — true for both. Confirmed by exhaustively checking both
  trigger values under all three operators (`&`/`|`/`^`), plus 20,000 random
  values with a zeroed low-19-bits prefix (the only inputs that reach this
  line at all) for the other function-level behavior.
- **`<impl AsmInsn for Inst>::emit_into`'s `B`/`BCond`/`CbzW16` match arms**
  (deleting one, not an operator swap) — each arm's body computes *literally*
  the same expression as `Inst::encode`'s own arm for that variant (compared
  side by side: `B`'s `0x1400_0000` vs. `0x1400_0000`; `BCond`'s
  `0x5400_0000 | self.condition as u32` vs. the identical text; `CbzW16`'s
  `0x3400_0010` vs. `0x3400_0010`), so deleting the dedicated arm and falling
  through to the wildcard's `self.encode()` produces byte-identical output
  for every value of the type, not just the ones under test.

This is the same class the 2026-09-07 doc documented for `avx2.rs`/`avx512.rs`
and the 2026-08-26 pass documented more generally — mutants `cargo-mutants`
cannot distinguish from the original by construction. No test was written
chasing any of them.

**The 4 timeouts** (`frame_alloc`/`frame_free`'s `>` → `>=` and `-=` → `/=`,
two each) are pre-existing and not something a test converts into a pass or a
clean failure: with `remaining >= 0` on an unsigned integer, the loop never
terminates; with `remaining /= chunk`, `remaining` gets stuck at `1` forever
once it drops below `MAX_ADD_IMM`. A hang *is* the detected divergence here —
the same category the `Counting::begin` test in the 2026-09-10 pass singled
out `CompileError::BudgetExceeded` for, one level further: a build that never
returns is worse than a wrong answer, and `cargo-mutants`' timeout mechanism
is what catches it. Nothing further was done for these two.

## Verified

- `cargo test -p pixelflow-codegen --lib emit::aarch64::`: 65 passed, 0
  failed.
- `cargo test -p pixelflow-codegen --lib`: 262 passed, 0 failed.
- `cargo test -p pixelflow-codegen` (all targets incl. doctests): pass.
- `cargo test --workspace`: pass (no `FAILED`/panicked lines in the full
  run).
- `cargo clippy -p pixelflow-codegen --lib --tests -- -D warnings`: clean.
- `cargo fmt -p pixelflow-codegen -- --check`: clean.
- `cargo mutants -p pixelflow-codegen --file
  pixelflow-codegen/src/emit/aarch64.rs -E
  '(disassemble_code|decode_aarch64_mnemonic|dump_jit_asm)'`: 401 mutants, 0
  real gaps (43/401 missed, all equivalent — see above; 51 unviable; 4
  timeouts, pre-existing and expected).
- Manual re-check: applying each of the 43 "equivalent" mutations by hand to
  a scratch copy of the file and re-running the specific new test confirmed
  no observable difference for the disjoint-field cases, and confirmed the
  `emit_ushr`/`mvn_w` precedence-reparse correction (an initial, incorrect
  assumption that grouping was preserved would have called several of these
  "real"; the corrected simulation, and then the actual `cargo-mutants` run,
  agreed with each other).

No production bugs were found. `needs_const_pool`'s asymmetric treatment of
`-0.0` (its bit pattern is `0x8000_0000`, nonzero, so it takes the
pool-eligible path rather than the same-as-`+0.0` fast path) was investigated
as a possible bug and rejected: `emit_fmov_imm`'s own zero fast-path check is
the identical `bits == 0`, so `needs_const_pool` is *consistent* with the
function it exists to predict for, not wrong. `-0.0` as a compile-time
constant is rare enough, and both paths produce a correct result regardless
(just, for `-0.0` specifically, one extra pool slot rather than a `MOVI`), so
this is a documented asymmetry rather than a defect.

## Recommended next steps (not done here)

Backlog carried forward from 2026-09-10, minus the item closed above:

1. `pixelflow-codegen/src/emit/mod.rs` (7,355 lines) and `regalloc.rs` (4,243
   lines) — untouched by any pass, both now large enough that a future pass
   needs the same kind of explicit sub-scoping this pass gave `aarch64.rs`
   (pick a coherent `impl` block or function family, check its
   `cargo-mutants --list` count first, document the cut).
2. `pixelflow-codegen/src/emit/aarch64/table.rs` (1,004 lines) — `aarch64.rs`'s
   own submodule (the abstract-operand-to-instruction table and its
   `Movz`/`AddI64`/`SubI64`/`CmpI64`/`Imm12` encoders), untouched by this
   pass since it is a separate file from the one named in the backlog; has
   its own `#[cfg(test)] mod tests` already
   (`add_i64_encodes_register_and_immediate`,
   `binary_and_unary_from_abstract_operands`) but was not mutation-tested
   here.
3. `pixelflow-codegen/src/emit/aarch64.rs`'s disassembler
   (`disassemble_code`/`decode_aarch64_mnemonic`, ~336 mutants,
   `dump_jit_asm`) — explicitly excluded from this pass as a debug-only,
   diagnostic-only decoder (see "Scope" above). If it is ever mutation-tested,
   most of its mutants will likely be string-literal/match-arm changes to
   individual mnemonics' text, which is a large amount of low-value surface
   compared to the rest of the file; worth a deliberate scoping decision
   (e.g. testing the decoder's *structural* correctness — offsets, word
   count, general field extraction — rather than every mnemonic string) if
   picked up.
4. `pixelflow-core/src/backend/arm.rs`'s NEON impls (39 lines) — still
   untestable from every x86_64 sandbox this series has run in; needs an
   aarch64 host.
5. `pixelflow-codegen/src/emit/executable/macos.rs` (114 lines) — still
   untestable from this sandbox; needs a macOS host. Its `linux.rs` sibling
   has had a `Drop`-unmapping test since 2026-09-10.
