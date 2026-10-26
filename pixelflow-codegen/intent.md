# pixelflow-codegen

**Is.** The CPU backend: a functor from the term category to a machine's
(DENOTATION §10), one function `(kernel, lattice) → code` with its own `Pass`
chain (§7), run on the ISA tier the process chose at startup. It is the only
crate that names a lane or a width.

## Owns
- The tier: `isa::detect` reads CPUID once per process; `PIXELFLOW_ISA` only
  narrows it; `jit_vector_bytes()` is the one width in the workspace.
- The chain `Legalize → Lower → Nest → Own → Layout` (`pipeline.rs`,
  `program/`), then `Select → LinearScan → Assemble` (`emit::native`). A stage
  is a `Pass`; the first `Err` ends the chain; the backend is chosen once.
- Register allocation and carries (`emit/regalloc/`), the encoders, and
  executable memory (`mmap`/`mprotect`, W^X).
- The compile cache (`jit_cache`), keyed on the canonical expanded kernel and
  its `LatticeShape`, so a resize recompiles; it runs `pixelflow-search`
  itself. Why a compile failed is `CompileError`.

## Never
- No lane, vector width, intrinsic or `cfg(target_feature)` outside `emit/`
  and `isa/`; other crates read `jit_vector_bytes()` and nothing finer.
- No ISA from build flags, no SSE2 tier, no scalar fallback; a host below the
  floor is refused naming the feature.
- `emit/` names nothing upstream; `program/` names no register or byte.
- No vector crosses the compiled kernel's ABI: a collapse is
  `fn(ctx, out, pitch)`, one call per extent.
- No silent failure: an exhausted budget is `Err`, a selection bug a panic.

## Contract
`src/lib.rs` is the whole public surface: `jit_cache` (`compile`, `Linked`,
`entry_count`), `emit` (`compile`, `CompiledKernel`, `CompileResult`,
`EmitTraffic`, `ScopeTraffic`, `Trips`), `isa` (`Isa`, `detect`), `jit_vector_bytes`,
`CompileError`, `fnv1a64`; `Pass` and `program` are crate-private.

## Enforced by
- `scripts/check-emit-boundary.sh` (CI job "Emit boundary"); desloppify
  `simd-is-codegens`, `registers-come-from-the-allocator`, `phases-in-order`.
- `Pass` laws: `then_is_associative_on_*` in `src/pass.rs`.
- Per tier: `xtask isa-matrix --clippy --smoke`, `tests/cross_tier.rs`;
  through the API: `tests/jit_cache_sharing.rs`, `tests/trig_range_jit.rs`.

## Behind
- Rows, batches and lanes are folds `pixelflow_ir::passes::legalize` wraps the
  kernel in before `Lower`: the CPU schedule sits in a target-agnostic crate
  (2026-10-15). `Nest` should factor the lattice (schedule note §1–§2).
- No `cpu/` module: the chain's front is `pipeline.rs`, its back
  `emit::native` (schedule note §4, `Proposed`).
- Memory reads remain though the language has none: `ScheduledOp::Gather`,
  `Broadcast`, `Write`, `Seq`; `Linked::buffers` (kernels-evaluate-kernels §2).
- `emit::compile` is `pub` and emits unoptimized; "never unoptimized" holds
  for `jit_cache::compile` only.
- `ScheduledOp::Context(u16)`: a 16-bit control-plane index, no profiler cited.

## Decided in
`docs/DENOTATION.md` §7, §10; `docs/language/2026-10-15-kernels-evaluate-kernels.md`
§3; `docs/codegen/2026-10-15-a-schedule-is-a-factorization.md`;
`docs/codegen/2026-09-22-the-isa-is-decided-at-startup.md`.
