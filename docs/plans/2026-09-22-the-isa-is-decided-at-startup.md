# The ISA is decided at startup

## Metadata
- **Author**: JP (decision), Claude (draft)
- **Status**: `Done` — the decision and every subtraction below are in the
  tree, and the SSE2 driver's deletion (§7) followed in its own PR.
- **Created**: 2026-09-22
- **Verified against**: `735cb9de` (main with #1286 and #1289)
- **Continues**: [collapse-is-a-fold](2026-09-16-collapse-is-a-fold.md) — that
  plan took the vector out of the compiled kernel's ABI (`fn(ctx, out, pitch)`);
  this one takes it out of `pixelflow-core` entirely.

**Decision it records (JP, 2026-09-22):**

> The ISA tier is decided at process startup by CPUID, not at build time by
> `target_feature`. The SSE/128-bit x86 tier is dropped: the floor is AVX2 plus
> FMA, AVX-512 is chosen whenever the host has it, and a host below the floor
> is refused loudly with a message naming the missing feature. aarch64 stays
> NEON. `pixelflow-core` loses its SIMD width entirely: the JIT owns the one
> width, and core asks for it at runtime.

---

## 1. What was wrong

Every backend in `pixelflow-codegen` has always compiled on every host —
emission is a pure function into bytes — so the target never decided which
backends *exist*, only which one was *instantiated*. That choice was a
`cfg(target_feature)` on a type alias (`emit::Native`), and `target_feature` is
a build flag that a plain `cargo build` never sets. Three consequences:

1. **Every x86-64 host ran 128-bit code.** The SSE2 backend was the default
   on a 16-lane machine, and the AVX2 and AVX-512 backends were entered only
   by a build somebody made with `-C target-feature=+avx2,+fma` or
   `+avx512f,+avx512dq`. `core-term` shipped that way.
2. **The width was stated twice and kept in step by convention.**
   `pixelflow-codegen` had `JIT_VECTOR_BYTES`, three `cfg` arms; `pixelflow-core`
   had `Field`, a `pub(crate)` newtype over `__m128`/`__m256`/`__m512`/`float32x4_t`
   chosen by the same predicate, whose only remaining job was
   `assert_eq!(size_of::<Field>(), JIT_VECTOR_BYTES)` in `Manifold::compile`.
   They drifted once already (a build-script probe in core ANDed against the
   flag; the `JIT_VECTOR_BYTES` doc comment narrated the bug), and the
   assertion existed to catch the next time.
3. **The matrix was three builds.** `xtask isa-matrix` rebuilt the whole
   workspace once per level with `RUSTFLAGS`, in a target directory of its own
   that it wiped per level to fit on disk, and passed `--target` so the flags
   would not reach build scripts — `proc-macro2`'s once died with `SIGILL` on a
   runner without AVX-512.

## 2. The denotation

The ISA tier is a property of the *process*, not of the build: a fact the CPU
reports, read once. So it is a value, `Isa`, with three inhabitants —
`Avx2`, `Avx512`, `Neon` — and one function that produces it:

```text
detect() : () → Isa        (OnceLock; the first call asks CPUID)
```

The backend is a function of the tier (`emit::compile_native`, one `match`),
and the width is a function of the tier (`Isa::vector_bytes`, an ISA-defined
table: a `zmm` is 64 bytes by definition). Everything that used to read
`JIT_VECTOR_BYTES` reads `jit_vector_bytes()`, which is `detect().vector_bytes()`.
There is nothing left for `pixelflow-core` to agree with, so it holds no
vector at all.

## 3. The floor, and why

**x86-64: AVX2 with FMA3.** The features each tier's encoder needs, read off
the emitters and declared beside the probe in `pixelflow-codegen/src/isa/x86_64.rs`:

| tier | CPUID features | what needs them |
|---|---|---|
| AVX2 (the floor) | `avx2` | `vpaddd`/`vpslld`/`vpsrld ymm`, `vpmovzxbd ymm` (the lane iota), `vgatherdps ymm`, `vbroadcastss ymm, xmm` (register source); implies AVX for the `ymm` float arithmetic, `vcmpps`, `vroundps`, `vinsertf128`/`vextractf128`, `vmovmskps`, `vcvttps2dq`/`vcvtdq2ps` |
| | `fma` | `vfmadd231ps` — the one-rounding `MulAdd` |
| AVX-512 | `avx512f` | the `zmm` file and EVEX arithmetic, `vcmpps` into `k` with `kmovw`/`kortestw`, `vptestmd`, `vpternlogd`, `vrndscaleps`, `vrcp14ps`/`vrsqrt14ps`, the writemasked `vmovups` store, `vgatherdps zmm`, `vpmovzxbd zmm`, EVEX `vcvttss2si`/`vmovq` |
| | `avx512dq` | `vpmovm2d` (every comparison's `k`→vector widening), the EVEX float logicals `vandps`/`vorps`/`vxorps`, EVEX `vpinsrq` — DQ, not F |

**aarch64: NEON**, always; Advanced SIMD is the base architecture.

**Why no SSE2 tier.** It was the default only because the build flag was
unset, never because a host lacked more: no x86-64 CPU sold since 2013 lacks
AVX2, and none has ever had AVX2 without FMA3 (x86-64-v3 codifies the pair).
Keeping it as a fallback would mean keeping a two-rounding `MulAdd`, a
scalar-insert gather and a six-register pool alive for a machine nobody runs
this on — and every one of those was a place the tiers behaved differently
(`tests/muladd_rounding.rs`'s `not(target_feature = "fma")` arm; the P2 in
`pixelflow-pipeline/src/journal.rs` where two kernels shared one fingerprint).
A host below the floor is refused at the first `detect()`:

```text
this CPU lacks `avx2`, and the JIT has no tier below AVX2+FMA on x86-64: ...
```

`pixelflow-runtime`'s `EngineTroupe::with_config` calls `detect()` first, so
`core-term` refuses before a window opens rather than faulting on its first
frame's kernel.

## 4. The override

`PIXELFLOW_ISA=avx2|avx512|neon`, read once with the detection (the tier is a
fact about the process; a compile cache keyed on shape assumes it does not
change under it). It picks a tier the host can *also* run — `avx2` on an
AVX-512 host — which is what lets one machine execute both x86 backends. It is
refused, never downgraded, when it names a tier the host cannot execute, and
an unrecognized value panics quoting itself. Same family as
`PIXELFLOW_SATURATION_CEILING_MS`: a diagnostic, never a fallback.

## 5. What is deleted

| what | lines | why it could go |
|---|---|---|
| `pixelflow-core/src/backend/{mod,x86,arm}.rs` | 165 | the four lane newtypes existed to give `Field` a size |
| `Field`, `NativeSimd`, `PARALLELISM` in `pixelflow-core/src/lib.rs`, and the width assertion in `Manifold::compile` | ~80 | nothing to agree with |
| `JIT_VECTOR_BYTES` and its three `cfg` arms in `pixelflow-codegen/src/lib.rs` | ~50 | `jit_vector_bytes()` |
| `emit::Native` and its four `cfg` arms | ~40 | `compile_native`'s `match` |
| the `avx2,-fma` `compile_error!` in `emit/avx2.rs` | 15 | the probe refuses the host |
| the SSE2-only `cfg` gates on `emit`'s tests, and the `+avx512f`/`+avx2` gates on the per-backend runtime test modules | — | tests run on the host's tier; per-backend runtime tests run wherever the host can execute them (`skip_unless_host_runs!`) |
| `xtask`'s `run_with_rustflags`, `host_triple`, the per-level target directory | ~90 | one build |

`pixelflow-core/src/backend/fastmath.rs` moved to `pixelflow-core/src/fastmath.rs`:
`FastMathGuard` is on the render path (`pixelflow-graphics/src/render/scene.rs`)
and has nothing to do with a width.

`pixelflow-core` had no `build.rs` to delete; the probe that the old
`JIT_VECTOR_BYTES` doc narrated was already gone.

## 6. What CI does now

`xtask isa-matrix` is one `cargo test --workspace --no-run`, one `cargo clippy`
(with `--clippy`), then per level in `[avx2, avx512]` that
`is_x86_feature_detected!` allows: `PIXELFLOW_ISA=<level> cargo test` on the
smoke set (presubmit, `--smoke`: codegen, ir, core, pipeline, and graphics'
two glyph-JIT test binaries) or the workspace (postsubmit). A level the runner
cannot execute is reported `NOT RUN (host lacks avx512f)`, never silently
skipped; there is no separate "built and linted" state per level any more,
because the build is not per level.

The presubmit check is still named `ISA matrix (SSE2/AVX2/AVX-512)`: branch
protection requires that exact name, and it is renamed together with the
required-checks list, not before.

Two consequences for the plain `test` jobs: `Test on ubuntu-latest` now runs
whichever tier the runner has (AVX-512 on some Azure SKUs, AVX2 on the rest),
and `Test on macos-latest` (arm64) runs the tests that were SSE2-gated before
— `dwrt_compiles_to_analytic_derivative`, the deep-spill frame, the
builtin-vs-scalar sweeps, `lowering_tests`, `sched` — on NEON. They test the
shared driver, not an encoding, and their inputs avoid every row of CLAUDE.md's
divergence table (the one `Round` tie, `1.5`, rounds to `2` under both
ties-even and ties-away).

## 7. The follow-up: delete the SSE2 driver — done

`x86_64::driver::X86Backend` had no arm in `compile_native`; it was typechecked,
swept for op coverage and unit-tested, and never instantiated. The follow-up
removed it, in three commits:

| what | lines | note |
|---|---|---|
| the AVX2 scalar-insert gather (two 128-bit halves of `emit_gather_scalar`, four temps and a GPR) | −60 net in `avx2.rs` | replaced by `vcvttps2dq`, `vpcmpeqd`, `vgatherdps ymm, [base + ymm*4], ymm` (VEX.256.66.0F38.W0 92 /r): two temps, no GPR, pinned bytewise and executed on the host |
| `x86_64::driver` — `X86Backend`, `SSE2_FILE`, its `IsaBackend` impl | ~510 | `Convert`, `index_into`, `write_address` and `frame_slot` moved to `x86_64.rs`'s top level first: they are the store's GPR arithmetic, shared by both surviving tiers |
| the 128-bit encoders only it called — `sse_rr` and the legacy `movaps`/`addps`/…, the VEX.128 `Vex`/`VexImm`, `emit_unary`/`emit_binary`/`emit_select`/`emit_const`, `X86BinaryInsn`, `emit_gather_scalar`, the `xmm` `emit_uniform_load`/`emit_broadcast_load`, `cvttss2si_*`, `movq`/`movlhps`/`movss`/`psrldq`, `movups_*`, `emit_movmskps_eax`/`emit_cmp_eax_imm8`, the `MulAdd` stand-in | ~900 | `x86_64.rs` went from 2,935 to 1,411 lines and is the leaf-encoder module its name says |
| `resolve_operands`' two-operand invariant (the aliasing `debug_assert` and its comment) | ~25 | `operand_sources` still reloads a binary's left into `dst` — a free reload target on a three-operand ISA, not a hazard |
| the SSE arm of every "every backend" byte test, `x86_backend_covers_required_ops`, the SSE2 `MulAdd` byte pins, the `X86Backend` row of the traffic test | ~120 | `pointer_class::a_context_pointer_is_loaded_once_per_call` now compiles through the AVX2 backend at eight lanes |

The leaf encoders the AVX2 and AVX-512 files import from `x86_64.rs` (`ret`,
`mov`, the `ConstPool`, `Disp`/`Mem`/`ptr`, `MovLoadPtr`/`MovStorePtr`,
`BroadcastGprs`, `Jmp`/`Jcc`) stayed; they are x86-64, not SSE2. Each tier's
register file now states every field itself — the SysV roles were `SSE2_FILE`'s
to inherit from, and are the architecture's to restate.

`Scratch::MAX_TEMPS` is still four: it was sized for the scalar-insert gather,
the widest instruction is now a guarded `Select` at six registers, and lowering
`MAX_TEMPS` (and so `MIN_SCRATCH`) moves every carry budget, which is a
measurement of its own.
