# pixelflow-codegen/src/emit

**Is.** The CPU backend, a functor from a scoped schedule to a binary
(DENOTATION §10): `select` to a machine function over *values*, `allocate`
(`regalloc/`) to the same over registers and slots, `assemble` to bytes.
Values are names, registers are resources, `encode` makes no choice; a lane is
`IsaBackend::Lane`, named only by these emitters.

## Owns
- The backend: `IsaBackend` (instructions, `RegisterFile`, constant pool,
  `Lane`), impls `Avx2`, `Avx512`, `Neon`, and the `Select`, `LinearScan`,
  `Assemble` chain `native` runs on the host's tier.
- Selection: what a `ScheduledOp` becomes on a tier, every opcode and encoding
  field, as `Inst<Stage>` enums; the vector width (`RegisterFile::vector_bytes`,
  read out as `jit_vector_bytes()` through `vector_bytes(isa)`).
- Label resolution (`asm.rs`), the constant pool, executable memory (`executable.rs`).

## Never
- No name from upstream: `pixelflow_search`, `pixelflow_ir::{passes,variance,store}`,
  a `program/` submodule, `crate::pipeline`, `crate::jit_cache` (bar the
  `pub use crate::pipeline::compile`).
- No register chosen outside `regalloc/`, and an encoder names no register:
  every one arrives as an `In`, `Out` or `InOut` borrow. The frame is
  addressed as the frame (x86's frame form, NEON's `[sp, #slot]` words), and
  a field an encoding fixes (`rsp` in the frame adjust, `sp`, `xzr`, `x30`,
  `k0`) is the word, not an operand.
- `select.rs` names no register or opcode; `asm.rs` imports nothing from the crate.
- No vector crosses the ABI, `fn(ctx, out, pitch)`; the lane count leaves only
  as the `u32` `native` hands the front. No `dyn`, no `Box`: `native` picks the
  tier once. Nothing reads `EmitTraffic`. No host-bound code outside `executable.rs`.

## Contract
`mod.rs` exports `compile`, `CompiledKernel` (`as_bytes`, `call`), `EmitTraffic`,
`ScopeTraffic`, `Trips`, `CompileResult`; `pub(crate)` `native`, `vector_bytes`. Inside:
`IsaBackend`, `Function`, `Stage` (`Selected`, `Bound`), `Class` (sealed: `Vector`,
`Pointer`, `Integer`, `Opmask`, `Flags`), `Value`, `LaneOp`, `Assemble`. Encoders:
`x86_64.rs` (shared), `avx2.rs`, `avx512.rs`, `aarch64.rs`; `regalloc/` has its own intent.

## Enforced by
- `scripts/check-emit-boundary.sh` (CI job `emit-boundary`): rule E on every
  file here, rule A on `asm.rs`.
- Types: `Def` is affine and built only in `build.rs`; `Builder::finish`
  asserts `Function`'s invariants; `Class` and `File` are sealed;
  `RegisterFile::new` is `const` and refuses a contradictory declaration.
- desloppify `registers-come-from-the-allocator`, `phases-in-order`, `simd-is-codegens`.
- Tests in `pixelflow-codegen/tests/`: `stack_probe.rs`, `context_past_imm12.rs`,
  `edges_x86_64.rs`, `cross_tier.rs`, run per tier by xtask's `isa-matrix`.

## Behind
- The lattice's rows, batches and lanes are folds `pixelflow_ir::passes::lattice`
  (`collapse`, `pack`) wraps the kernel in, `native` handing the front
  `lanes_of::<B>()`: the CPU schedule is upstream of its backend (kernels-evaluate-kernels §3).
- A branch displacement past its field panics (`DispField::write`,
  `patch_rel32`) rather than returning `Err`; no plan names it.
- 11 item-level `cfg`s in `executable.rs`, its platform files and `mod.rs` are
  baselined in `scripts/cfg_encapsulation_baseline.txt`.

## Decided in
`docs/plans/`: `2026-10-08-selection-is-a-phase.md` §0, `2026-09-12-emit-should-just-emit.md`,
`2026-09-22-the-isa-is-decided-at-startup.md`, `2026-09-22-a-pointer-is-a-value.md`,
`2026-09-16-collapse-is-a-fold.md`, `2026-10-15-kernels-evaluate-kernels.md` §3.
