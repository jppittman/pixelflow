> **Paths stale, findings live (annotated 2026-09-09).** The completeness check and
> the first `Instruction`/`Assembler` split landed as described. The files it cites
> for the follow-up work have since moved to `pixelflow-codegen`:
> `pixelflow-ir/src/backend/emit/lowering.rs`, `bench_hoisted_scanline.rs` and
> `pixelflow-pipeline/src/bin/bench_scanline_jit.rs` were all deleted. Its central
> finding — that `Kernel::over` has never emitted a loop on any backend because it
> always unrolls — is still true and is why the language is a DAG with no binder.

# Two-level IR, backend completeness, and where the loop binder belongs

**Date:** 2026-07-25
**Status:** Design + landed slice (completeness check + first Instruction/
Assembler split). Binder-in-IR migration is NOT implemented here — see
"What's follow-up" at the end.

## The question

Adding one feature — a bounded-loop/reduction binder — looked like it would
require hand-writing assembly in three backends. That prompted two distinct
questions from the project owner, quoted directly:

1. "Do languages ever do two levels of IR, or maybe our IR is too big? ...
   Something is wrong/missing here for this to turn up/require work in 3
   backends?"
2. "I want a worker making sure this is the only version of this, because I
   feel like this project has written a lot of assembly."

Both are answered below from the actual code, file:line, not from "multi-ISA
backends are just like this." Two-level (or staged) IR is close to universal
in real compilers — LLVM IR -> SelectionDAG/GlobalISel -> MachineIR, Cranelift
CLIF -> VCode, GCC GENERIC -> GIMPLE -> RTL, Halide's single-IR-with-staged-
lowering-passes — so "should there be staging" was never the interesting
question. The interesting question is whether this codebase already has
staging, informally, and whether the boundary sits in the right place.

**Short answer: yes to staging (it already exists and is sound); no to "one
version of this" (there are three competing loop-emission strategies, only
one of which is even used in production, and none of them are wired to what
actually renders); and the missing piece is not a backend problem at all —
the reduction binder (`Kernel::over`) has never emitted a loop, ever, on any
backend, because it always unrolls.**

## Part 1 — Is there a two-level IR, and is the boundary right?

Yes. It already exists, and it is sound as far as it goes. Every compile
entry point that matters runs the same two stages:

**Stage 1 — ISA-agnostic canonicalization.** A fixed sequence of pure
`ExprArena -> ExprArena` endomorphisms in `pixelflow-ir/src/backend/emit/lowering.rs`,
run identically regardless of target:

```
lower_dwrt_owned -> expand_reduce_owned -> expand_gather_owned -> expand_transcendentals_owned
```

(see the call sites in `compile_arena_dag_with_ctx` for both architectures,
`mod.rs:645-648` aarch64 / `mod.rs:3287-3290` x86). Each pass eliminates one
op family entirely: `Dwrt` (symbolic differentiation via chain rule),
`Reduce` (the binder — currently *always* unrolled, see Part 3), the ternary
`Gather` (lowered to index arithmetic + `RawGather`), and all transcendentals
(`Sin`/`Cos`/`Exp`/.../`Pow`, expanded to polynomial arithmetic). What
survives this stage is a small, closed arithmetic/comparison/bit-manip
subset — this is the actual IR boundary, and it is a real one: no backend
ever needs to know how to differentiate, fold, gather, or evaluate a sine.

**Stage 2 — scheduling, allocation, and per-ISA leaf encoding.**
`arena_to_schedule` (linearization — free, since the arena is already
topologically ordered) -> `regalloc::linear_scan` (Belady eviction,
ISA-agnostic) -> `analyze_select_guards` (short-circuit branch analysis for
`Select`, ISA-agnostic) -> the `IsaBackend` trait's leaf methods
(`emit_plan`/`emit_mov`/`emit_jump`/`patch_branch`/...), which is the only
part that differs per architecture. This is `emit_dag_body`
(`mod.rs:1985-2184`) plus `compile_dag_via_backend`
(`mod.rs:2192-2223`), and it is genuinely shared: `Aarch64Backend`
(`mod.rs:2225`), `X86Backend` (`mod.rs:3385`), and `Avx512Backend`
(`mod.rs:3663`) are ~15-method trait impls that only encode bytes; none of
them re-implement scheduling, allocation, or `Select` guard analysis.

So the codebase already has exactly the shape the owner was asking about —
LLVM's SelectionDAG/MachineIR split, Cranelift's CLIF/VCode split — it was
just never named or documented as a boundary. `docs/designs/
2026-07-23-jit-orthodoxy-survey.md` independently reached the same
"orthodox for the Halide/XLA/shader-compiler family" verdict by a different
route (comparing tiering/speculation machinery, not IR staging), and did not
identify this staging explicitly. This doc names it.

**Where the boundary is NOT sound: encoding has no internal structure.**
Stage 2's per-ISA leaf is not itself two-level. Each backend's `emit_binary`/
`emit_unary`/`emit_ternary` is one flat function that both *decides which ops
it supports* and *emits their bytes* in the same `match`, e.g. (before this
session's change) `x86_64.rs`:

```rust
match op {
    OpKind::Add => emit_addps(code, dst, src2),
    ...
    OpKind::BitOr => emit_vorps(code, dst, dst, src2),
    _ => panic!("x86_64 binary emit not implemented for {:?}", op),
}
```

`avx512.rs`'s equivalent (`emit_binary`, pre-existing, untouched by this
session per the out-of-scope note) implements 6 of the 15 ops this match
covers and returns `Err` for the rest — a graceful failure, but the *point*
is that "which 6" was never written down anywhere except as the literal set
of arms present in that one match. Nothing enumerated "the ops a backend
must support"; a match arm existing was the only evidence that op was
supported, and a match arm *not* existing looked identical to "this op
legitimately never reaches this backend." That is a real gap, and it is not
an ISA-differences problem — see Part 4.

This gives a precise, non-analogical answer to question 1: **the codebase
already has a two-level IR (lowering vs. scheduled-and-selected leaf
emission) and the boundary is in the right place; the missing level is
*inside* the leaf** — selection (which mnemonic does this op become) and
encoding (which bytes does that mnemonic become) are conflated into one
un-auditable match per backend, the same shape LLVM's SelectionDAG (op ->
`MachineInstr`) and Cranelift's lowering pass (CLIF -> `VCode` instruction)
keep as two explicit steps.

## Part 2 — Is "3 backends of assembly" structural, irreducible, or a bug?

All three, for three different pieces of code. Read `mod.rs`, `x86_64.rs`,
`aarch64.rs`, `avx512.rs` end to end to separate them:

### 2a. The AVX-512 op-coverage gap — a bug, now load-bearing-checked

`avx512::emit_binary` supported Add/Sub/Mul/Div/Min/Max only; comparisons
route through a separate `is_compare`/`emit_compare` pair (so they *were*
covered, contrary to a first read of `emit_binary` alone) but `IAdd`/
`BitAnd`/`BitOr` were not, and `ResolvedOp::ShiftImm` was an explicit,
permanent `Err` ("EVEX integer shift not wired yet", `mod.rs:3720-3722`,
pre-existing). Unary `Rsqrt`/`Recip`/`TruncToInt`/`IntToFloat` were also
missing. Nothing enumerated the required set, so nothing could fail loudly
when 9 ops were absent — the gap was discovered by accident (36 tests
failing the first time someone compiled with `-C target-feature=+avx512f`).
This is the completeness-check gap fixed in Part 5; it is a genuine
structural absence (no contract existed to violate), not an irreducible ISA
cost — EVEX integer shift and the three bit-manip ops are ordinary
instructions, not missing hardware.

### 2b. Three unrelated loop-emission strategies, none shared, none in
production

This is the direct answer to "make sure this is the only version of this."
It is not.

**(i) A Sethi-Ullman tree-walk emitter, x86-only, no spilling.**
`needs_arena`/`emit_arena` (`mod.rs:387-588`) is a full second code
generation strategy — recursive register-by-tree-position, not
schedule+linear-scan — that exists *only* because the x86 scanline body
predates (or was never migrated to) the shared driver. It is explicitly
documented as the exception: "the per-batch path goes through the shared
schedule/regalloc driver" (`mod.rs:3269`). Aarch64 has no equivalent — its
scanline path uses the schedule+regalloc approach throughout. So this is one
architecture (x86) carrying a whole second emitter alive for one call site.

**(ii) Both scanline-hoisted paths reimplement `emit_dag_body`'s guard logic
by hand, around a hand-rolled loop, calling neither `emit_dag_body` nor
`IsaBackend`.** `compile_scanline_hoisted` (aarch64, `mod.rs:1022-1421`) is
~250 lines that duplicate `emit_dag_body`'s Select short-circuit
guard-branch bookkeeping (`branch_starts`/`branch_ends`/`pending_patches`,
byte-for-byte the same algorithm as `mod.rs:2016-2043` and `2065-2172`) —
but calls `aarch64::emit_umaxv`/`emit_cbz_w16`/`patch_cbz_cbnz` directly
instead of `IsaBackend::emit_skip_if_all_false`/`patch_branch`, and does so
*twice inline* (once for the setup phase, once for the loop phase — compare
`mod.rs:1182-1213` against `1250-1281`, nearly identical). x86's
`compile_arena_dag_scanline` (`mod.rs:3995-4068`) hand-encodes its loop
control flow as raw opcode bytes (`0x31, 0xC0` for `xor eax, eax`, etc.) and
`compile_collapse_avx512`'s `emit_avx512_collapse_loop`
(`mod.rs:3923-3974`) does the same for AVX-512's zmm registers. Three
architectures, three independent byte-level loop implementations, zero
sharing, despite `emit_dag_body`'s own doc comment already stating the
right shape: *"The body's branches are self-relative, so a caller may
freely prepend a prologue / wrap it in a loop / append an epilogue. This is
the seam that lets both the per-batch kernel and the internal-loop collapse
driver share one body emitter"* (`mod.rs:1980-1982`). `compile_collapse_avx512`
is the one place that actually uses that seam correctly — it calls
`emit_dag_body` for the body — but even it abandons `IsaBackend::emit_jump`/
`patch_branch` for the loop's own back-edge and hand-encodes the `cmp`/`jae`/
`jmp` bytes instead, for no reason the code states (both `emit_jump` and
`patch_branch` handle backward targets fine — see
`x86_64.rs::patch_rel32` and `aarch64.rs::patch_b`/`patch_cbz_cbnz`, all of
which compute `target - pos` and accept a negative result).

**(iii) None of the above three are in the production render path.** Grep
confirms: `compile_collapse_avx512`/`CollapseKernelFn` has exactly one
caller anywhere in the workspace — its own test (`mod.rs:6098` in the
`avx512_driver` test module). `ScanlineJitManifold` (the wrapper around both
scanline paths) is used only by two benchmark binaries,
`pixelflow-pipeline/src/bin/bench_scanline_jit.rs` and
`bench_hoisted_scanline.rs` — never by `pixelflow-core` or
`pixelflow-graphics`. The actual hot path,
`pixelflow-core/src/lattice/mod.rs::Lattice::collapse` (and `fold_lanes`,
`collapse_axis0/1`), is a plain Rust `for`/`while` nest
(`lattice/mod.rs:338-403`) that calls `RealizedKernel::eval` once per
SIMD-width batch, which crosses into JIT code via one `extern "C"` call per
batch (`lattice/mod.rs:606-621`) — exactly the per-batch call boundary the
internal-loop kernels exist to eliminate ("no per-batch Rust <-> JIT boundary
crossing," `mod.rs:3878`). So the project paid for three hand-written,
per-ISA internal loops, and the code that actually renders every frame
still pays a Rust-loop-plus-extern-C-call per batch anyway.

**(iv) Both scanline tiers skip Stage 1 lowering entirely — a second,
independent completeness gap from 2a, on both architectures.** Neither
`compile_arena_dag_scanline` (x86, tree-walk) nor
`compile_arena_dag_scanline_hoisted`/`arena_to_hoisted_schedule` (aarch64)
calls `lower_dwrt_owned`/`expand_reduce_owned`/`expand_gather_owned`/
`expand_transcendentals_owned` before scheduling. This is stated outright in
the aarch64 source: *"Both forms are checked: lowered `RawGather`
(`ScheduledOp::Gather`) and the high-level ternary (this path runs no
lowering passes)"* (`mod.rs:1004-1005`). Concretely: a scanline kernel
containing `sin`, `Dwrt`, or a `Reduce` binder will panic or error on
*either* architecture's scanline path, while the exact same expression
compiles fine through the per-batch path on both. This was never caught
because — see (iii) — nothing in production exercises the scanline tier.

**Verdict on question 2:** genuine duplicate/competing machinery, but not of
one kind. (i) and the byte-level halves of (ii) should go — they duplicate a
capability (schedule + linear-scan + guarded body emission) that
`emit_dag_body` already provides correctly and more completely (it runs
lowering; the scanline paths don't). (iii)'s framing ("collapse loop") is
the *right* shape and should stay and be finished, not deleted — it is an
abandoned prototype of the correct pattern, not a competing one, and its
only real defect is bypassing `IsaBackend`'s own branch primitives for no
stated reason. None of this is an irreducible per-ISA cost; it is three
uncoordinated one-off answers to "how do I emit a loop," each written when
someone needed exactly one call site to go faster, never generalized,
never connected to what ships.

## Part 3 — Where the binder/scope belongs

The reduction binder already has an ISA-agnostic front door and an
ISA-agnostic lowering pass — `Kernel::over`/`sum_over`/`product_over`/...
(`pixelflow-ir/src/kernel.rs:500-547`) build an `OpKind::Reduce` node, and
`expand_reduce`/`unroll_reduce` (`lowering.rs:266-340`) is Stage 1, exactly
where Part 1 says it should be: one pure, ISA-agnostic pass that every
backend already runs. This part of the design (docs/designs/
REDUCTIONS_AND_FOLDS.md, "the reduction is the operation that CROSSES a
scope boundary") is fully realized in code today.

**What is not realized: `unroll_reduce` never emits a loop. It always
statically unrolls** (`lowering.rs:330-339`: `n` copies of the body chained
by the combiner op, `n` read from a `Const`). This is correct and cheap for
the binder's actual current uses — dot products, softmax over small
extents, N ~ 3-64 — but it is not a substitute for a real loop, and it is
the reason the lattice's coordinate nest was never expressed as `Reduce`:
unrolling a 1920-pixel scanline into machine code is not a viable strategy,
so the lattice loop stayed a separate, invisible-to-the-IR Rust `for` loop
in `pixelflow-core`. **The binder was never missing a backend. It was
missing a second lowering strategy for large extents — which is a Stage 1
question, not a per-ISA one**, and answering it is most of what "unify the
lattice nest into the binder" needs.

Proposed placement, concretely:

1. **Stage 1 gets a second reduction strategy, chosen by extent (or an
   explicit hint), not by backend.** Small `n` keeps unrolling exactly as
   today (`unroll_reduce`, unchanged). Large `n` — or a `Reduce` explicitly
   tagged as a scope/coordinate binder rather than an arithmetic fold —
   lowers instead to a new arena shape the scheduler understands as "emit
   this body once, with a loop-carried index and accumulator, `n` times."
   This is a **pure `ExprArena -> ExprArena` decision**, made once, in
   `lowering.rs`, identically for every backend — the same place `Dwrt`,
   `Gather`, and the small-`n` `Reduce` case are already decided.

2. **Stage 2 gets the loop-control-flow mechanism once, in `emit_dag_body`/
   `compile_dag_via_backend`, not per backend.** The body-with-a-loop-inside
   is already proven sound by `compile_collapse_avx512`: call
   `emit_dag_body` once for the loop body (it already returns
   self-relative-branch bytes designed to be embeddable), then wrap it using
   only `IsaBackend` primitives that already exist —
   `backend.emit_jump()` / `backend.patch_branch()` for the backward
   branch (both already handle negative/backward targets, per Part 2's
   citation of `patch_rel32`/`patch_b`), `backend.emit_mov()` for handing
   the accumulator/index across iterations. **Zero new per-backend assembly
   is needed for the branch and move primitives** — they are exactly what
   `IsaBackend` already declares.

3. **One genuinely new per-backend primitive is needed, and it is small:
   a scalar loop-counter compare-and-branch.** `IsaBackend` currently has
   branch primitives for two things: SIMD mask guards
   (`emit_skip_if_all_false`/`emit_skip_if_all_true`, for `Select`) and an
   unconditional jump. It has nothing for "test an integer counter against
   a bound and branch" — the one piece of control flow every hand-rolled
   loop in Part 2(ii) had to invent for itself, three times, in three
   different styles (x86 scanline: `cmp r8,rcx; jae`, forward-exit +
   backward `jmp`; aarch64 scanline: `subs x21,x21,#1; b.ne`,
   decrement-and-branch; AVX-512 collapse: identical shape to x86 scanline).
   Promoting this to two new `IsaBackend` methods (e.g.
   `emit_loop_init(code, counter_reg, bound_reg) -> Branch` for the
   early-exit-if-zero check, `emit_loop_back_edge(code, counter_reg) ->
   Branch` for the decrement/compare-and-branch) turns three ad hoc,
   already-written encodings into three ~5-line trait impls, which is
   *less* code than exists today, not more — this is "subtract before you
   add" applied to the loop mechanism itself.

4. **The one truly irreducible per-backend difference is what a
   loop-invariant value is held in, not how the loop branches.** AAPCS64
   gives 8 callee-saved NEON registers (v8-v15) free for hoisting values
   across iterations at zero cost; SysV has *zero* callee-saved XMM/YMM
   registers, so x86-64 hoisting is stack-slot-based, a different strategy,
   not just different bytes (this is exactly why
   `MAX_PERSISTENT_SLOTS` is 8 on aarch64 and 0, "not yet implemented," on
   x86-64 today — `mod.rs:692-696`). This asymmetry is real, ABI-driven, and
   will still require two different hoisting strategies after the binder
   unification — but it is orthogonal to loop *control flow*, which the
   proposal above makes fully shared. Do not let this one genuine ABI
   difference justify duplicating the branch/jump machinery too, which is
   what happened in Part 2(ii).

Net effect once this lands: `Kernel::over` at pixel-coordinate scope
*becomes* the lattice's coordinate loop — closing the "lattice dissolves
into nested scope/binder nodes" idea from `docs/designs/
lattice-scheduling-types.md` and `REDUCTIONS_AND_FOLDS.md` — and
`Lattice::collapse` can finally call the internal-loop compile entry points
that already exist (generalized beyond AVX-512-only) instead of paying a
per-batch `extern "C"` call in a Rust loop on every frame, on every backend.
That second effect is not a new task; it is 2b(iii)'s dangling wire finally
getting connected, as a consequence of the binder unification rather than a
separate project.

## Part 4 — The Instruction/Assembler split (landed slice)

Part 1 identified the real gap as *inside* the per-ISA leaf: selection and
encoding conflated into one un-auditable match. `docs/function-namespace-audit.md`
found the same family from a different angle — 152 flat `emit_*` free
functions (87 `pub`) across `x86_64.rs`/`aarch64.rs`/`mod.rs`, no shared
type, "an un-namespaced public assembler API." That audit's proposed fix
(wrap `Vec<u8>` in an `Assembler` newtype) is necessary but not sufficient
by itself — a newtype around the buffer does not make the *dispatch*
exhaustive; only splitting selection from encoding does that.

**Landed this session, one backend, one op family, as the pilot:**
`x86_64.rs` gains `X86BinaryInsn` — a closed enum of the ten binary
mnemonics this backend can emit (`AddPs`/`SubPs`/.../`CmpPs(pred)`/`PAddD`/
`AndPs`/`OrPs`) — with two functions where there was one match:

- `X86BinaryInsn::select(op: OpKind) -> Option<Self>` — still partial over
  `OpKind` (most of its variants are unary, ternary, or eliminated by
  lowering before reaching here), so it keeps a `_ => None`, but every op
  this backend supports is now named exactly once, as data, in one place a
  completeness test can enumerate directly.
- `X86BinaryInsn::encode(self, code, dst, src2)` — exhaustive over the
  *closed* `X86BinaryInsn` set, **no wildcard arm**. Adding a variant
  without teaching `encode` how to emit it is now a `rustc` compile error,
  not a missing case discovered at runtime — this is the actual mechanism
  LLVM's `MachineInstr`/Cranelift's `VCode` gets from making "instruction"
  a value instead of a function call.

`emit_binary`'s behavior and byte output are unchanged (same
`emit_addps`/`emit_vandps`/etc. calls, same argument order); confirmed by
the pre-existing test suite staying green (`cargo test -p pixelflow-ir`, see
Part 5).

**Deliberately not done in this slice**, and why:

- **Not extended to unary/ternary ops, or to `aarch64.rs`.** Same pattern
  applies cleanly (`X86UnaryInsn`, `Aarch64BinaryInsn`, ...); this is
  follow-up sized-to-review, not a research question.
- **Not applied to `avx512.rs`, and not extended to a shared enum across
  ISAs.** Two reasons. First, a shared `Instruction` enum across backends
  would be the wrong abstraction — an ARM `fadd` and an x86 `vaddps` really
  are different operations with different operand shapes (this is the one
  piece of the owner's framing this doc pushes back on: leaf *encoding*
  genuinely is per-ISA and no abstraction erases that; what was missing was
  never a shared instruction set, it was exhaustiveness *within* each
  backend's own closed set). Second, `avx512.rs` and the surrounding
  `mod.rs` cfg/dispatch logic are being actively edited concurrently by a
  separate agent doing the AVX2/AVX-512 op-coverage completion in a
  different worktree; touching that file risked a real merge conflict for
  no benefit this session, so it was left alone entirely. **The natural
  place to apply this pattern next is exactly that AVX-512 completion
  work** — filling in the missing 9 ops as new `Avx512BinaryInsn`/
  `Avx512UnaryInsn` variants (with the required-op completeness test from
  Part 5 as the acceptance check) rather than as new arms in the existing
  flat match, so the fix and the structural improvement land together
  instead of the structural improvement becoming a second migration later.

## Part 5 — The completeness check (landed, verified)

`pixelflow-ir/src/backend/emit/coverage.rs` (test-only, `#[cfg(test)]`) is
the single enumeration that did not exist before: `REQUIRED_UNARY_OPS`
(10 ops), `REQUIRED_BINARY_OPS` (15), `REQUIRED_SHIFT_OPS` (2), and
`REQUIRED_TERNARY_OPS` (`MulAdd`, `Select`, documented but hand-checked
rather than looped — each has a distinct `ResolvedOp` shape). Each list's
doc comment cites exactly which `lowering.rs` pass removes everything *not*
listed, so the contract is traceable back to Part 1's Stage 1/Stage 2 split,
not an arbitrary list.

`mod.rs`'s new `backend_op_coverage` test module sweeps these against a
real backend via `IsaBackend::emit_plan` (not the raw per-op functions, so
it doesn't care whether a given backend signals "unsupported" via `Err`
(avx512) or `panic!` (x86-64/aarch64 today) — `catch_unwind` treats both as
the same "not supported" signal) and collects every failure into one
itemized list rather than stopping at the first:

- `x86_backend_covers_required_ops` — `#[cfg(target_arch = "x86_64")]`,
  always compiles and runs on x86-64. Currently green (`X86Backend`
  already covers every required op).
- `aarch64_backend_covers_required_ops` — `#[cfg(target_arch = "aarch64")]`,
  same shape, for the architecture this session's host cannot even compile
  (the struct itself is arch-gated), so it will run wherever `cargo test`
  actually runs on aarch64 hardware.
- `avx512_backend_covers_required_ops` — `#[cfg(all(target_arch = "x86_64",
  target_feature = "avx512f"))]`, mirroring the exact cfg
  `compile_arena_dag_with_ctx` uses to select `Avx512Backend` in production
  (`mod.rs:3292-3299`). On this session's default build it does not compile
  at all — not "skipped," genuinely absent from the build — so it makes no
  claim about AVX-512 completeness today and does not block this worktree's
  green build. The moment the concurrent multi-ISA work (or anyone) builds
  with `+avx512f`, this same test starts running and will fail immediately
  with an itemized list of every missing op, instead of the 36-unrelated-
  test-failures discovery mode from before this session.

`x86_64.rs` additionally gets `selects_every_required_binary_op`, a cheaper
and more precise check specific to the Part 4 pattern: it calls
`X86BinaryInsn::select` directly against `REQUIRED_BINARY_OPS` rather than
emitting-and-catching, because once a backend adopts the selection/encoding
split, checking selection is a pure function call, not a runtime
capability probe.

**Verified to actually catch a regression**, not just to pass today:
commenting out the `OpKind::BitOr => Some(Self::OrPs)` arm in
`X86BinaryInsn::select` and re-running the suite fails with

```
X86Backend (SSE2) is missing required ops: ["binary BitOr"] (see
pixelflow-ir/src/backend/emit/coverage.rs for the full completeness
contract)
```

— by name, at the completeness test, not as a panic three stack frames
away inside an unrelated test that happened to construct a `BitOr` node.
The stub was reverted immediately after confirming this; `git diff` on the
worktree shows no trace of it.

`cargo test -p pixelflow-ir --lib backend::emit` and `cargo test
--workspace` (default build, no RUSTFLAGS) both stay green with the check
added — see the commit history on this branch for the exact run.

## What's landed vs. what's follow-up

**Landed this session:**
- This design doc.
- `coverage.rs` — the required-op enumeration, the completeness contract.
- `backend_op_coverage` test module (`mod.rs`) — sweeps `X86Backend` always,
  `Aarch64Backend`/`Avx512Backend` when the build actually compiles them.
- `X86BinaryInsn` selection/encoding split (`x86_64.rs`) — pilot of the
  Instruction/Assembler pattern, one backend, one op family, zero behavior
  change, plus its own unit test.

**Explicit follow-up, not done here, and why:**
- Extending the selection/encoding split to unary/ternary ops and to
  `aarch64.rs` — same pattern, mechanical, no open design question.
- Applying the pattern to `avx512.rs` as *how* the concurrent multi-ISA
  work fills its 9 missing ops, rather than as new arms in the existing
  flat match — a coordination point for the main session, not something to
  do unilaterally into an actively-edited file.
- The reduction-binder-as-loop lowering strategy from Part 3 (large-`n`
  `Reduce` lowering to a real loop instead of unrolling) and the two new
  `IsaBackend` loop-counter primitives it needs — a real, scoped
  implementation task, larger and riskier than this session's slice, and
  explicitly out of scope per the task brief ("do not implement the full
  binder-in-IR migration").
- Retiring the x86 Sethi-Ullman tree-walk emitter (`needs_arena`/
  `emit_arena`) and both scanline-hoisted paths' hand-duplicated guard logic
  in favor of routing everything through `emit_dag_body`, and running Stage
  1 lowering on the scanline tier so it stops silently rejecting
  transcendentals/`Dwrt`/`Reduce` that the per-batch tier accepts — real
  bug fixes (2b-iv is a correctness gap, not just a cleanliness one) but a
  separate, reviewable change from this one.
- Wiring `Lattice::collapse` to the internal-loop compile entry points once
  Part 3 lands, eliminating the per-batch `extern "C"` call boundary in
  production — the payoff of Part 3, not a separate project.
