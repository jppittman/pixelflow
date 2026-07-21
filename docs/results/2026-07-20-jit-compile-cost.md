# JIT compile cost: gate G0 for per-kernel JIT (2026-07-20)

Phase 0 of the kernel-unification plan: measure how long `compile_arena_dag`
(`pixelflow-ir/src/backend/emit/mod.rs`) takes for expression arenas of ~8, 32,
128, and 512 nodes, distinguishing fresh executable-memory allocation per
compile from a reused code buffer. Gate G0: if the amortized cost exceeds
~1µs/kernel, per-leaf JIT is formally dead.

Reproduce: `cargo run --release -p pixelflow-pipeline --features training --bin bench_jit_compile_cost`

## Timer correction (affects all prior harness numbers)

While validating these results, the shared harness timer
(`pixelflow-pipeline/src/jit_bench.rs::nanos_now`) was found to return raw
`mach_absolute_time()` ticks under the assumption that ticks == nanoseconds.
That is true on Intel Macs and under Rosetta, but **not on native Apple
Silicon**, where the timebase is 125/3 — one tick = 41.67ns. Verified
empirically on this machine: `mach_timebase_info` returns `numer=125,
denom=3`, and a 100ms sleep measures 2.47M raw ticks. `nanos_now` now converts
ticks to true nanoseconds via `mach_timebase_info` (queried once).

Consequence: every nanosecond figure previously produced by this harness on
Apple Silicon (bench corpus labels, the 2026-07-08 extraction-3way absolute
ns columns, extraction-head training targets) is **under-scaled by 41.67x**.
Ratios and orderings are unaffected (uniform scale factor); absolute values
are not. Without this fix, the fresh-path number below would have read
"~119ns per compile" and G0 would have (wrongly) passed.

## Setup

- **Machine**: Apple M2 Max, macOS 26.5.2, aarch64 (NEON JIT ABI, 16KB pages).
  rustc 1.92.0, `--release` (opt-level=3, no LTO).
- **Timing**: corrected `nanos_now` (see above); 101 timed compiles per cell,
  16 warmup compiles, median reported. Two independent runs agreed within ~2%.
- **Arenas**: deterministic, exactly-sized, built by `build_kernel_arena` in
  `pixelflow-pipeline/src/bin/bench_jit_compile_cost.rs`: a circle-SDF core
  (`(x-0.5)² + (y-0.5)²` via `mul_add`) extended by a cycling op mix of
  `mul`/`add`/`sub`/`sqrt`/`mul_add`/`max` and `Lt`-guarded `select` — the
  font/shader-kernel shape. All nodes reachable from the root. Ops stay in the
  directly-emittable set (no transcendentals/gather/reduce), so the lowering
  passes are identity fast-paths and both paths below compile identical work.
- **(a) fresh** — `compile_arena_dag`: each compile mmaps a new
  `ExecutableCode` region, mprotects it to RX, invalidates icache, and
  munmaps on drop. mmap and munmap are both inside the timed window.
- **(b) reused** — `CompileWorkspace::compile_arena`
  (`pixelflow-ir/src/backend/emit/mod.rs`): the `CodeBuffer` (MAP_JIT) is
  mmap'd once; each compile pays `pthread_jit_write_protect_np` toggles +
  `sys_icache_invalidate` instead of mmap/munmap, and reuses pre-sized
  scratch vectors for the schedule.

## Results

| nodes | code bytes | fresh ns/compile | reused ns/compile | reused ns/node |
|---:|---:|---:|---:|---:|
| 8   | 40    | 4,959     | 5,042     | 630 |
| 32  | 160   | 12,250    | 11,958    | 374 |
| 128 | 628   | 83,500    | 82,917    | 648 |
| 512 | 2,504 | 1,142,792 | 1,127,375 | 2,202 |

Observations:

- **Buffer reuse buys almost nothing** (±2%, within run-to-run noise). On this
  macOS version the mmap/munmap pair is far cheaper than the ~10-20µs
  documented in `executable.rs:143-152`; codegen itself dominates at every
  size. The amortizable syscall overhead the CodeBuffer was designed to
  eliminate is not where the time goes.
- **Scaling is superlinear past ~128 nodes**: ~374-650 ns/node up to 128
  nodes, 2,200 ns/node at 512 (1.14ms total). The schedule/regalloc/emit
  pipeline, not executable-memory management, is the cost center.

## Conclusion vs gate G0

**Per-leaf JIT fails G0 decisively.** The amortized (reused-buffer) cost of
compiling even the smallest, 8-node leaf-shaped kernel is ~4.8-5.0µs — about
5x over the ~1µs/kernel threshold — and the floor is set by codegen, not by
executable-memory allocation, so no allocation-pooling scheme can rescue it.
At ~5-12µs per small kernel, JIT compilation is only viable for kernels that
are compiled once and evaluated many thousands of times (its current batch
use), not per-leaf. Any path to per-leaf viability would require a
fundamentally cheaper emitter (e.g. template/copy-and-patch codegen), not
buffer reuse.
