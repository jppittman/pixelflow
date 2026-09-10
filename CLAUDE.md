# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**core-term** is a GPU-free terminal emulator built on PixelFlow, a pull-based functional graphics engine using CPU SIMD.
**pixelflow** is an eDSL built on rust isomorphic to the typed lambda calculus.
**pixelflow-graphics** is a graphic library built using the aforementioned eDSL.
**pixelflow-runtime** offers a platform agnostic runtime for applications using pixelflow rendering.
**actor-scheduler** offers a user space cooperative scheduler for actor model based libraries/applications

## Critical Constraints

- **NO TERMINAL LOGIC GOES IN PIXELFLOW.** PixelFlow is a general-purpose graphics library being extracted to its own crate/repo. Keep it terminal-agnostic.
- Exporting direct manipulation of fields from pixelflow-core is strictly forbidden. Construct compute kernels at load time and render them.
- **NO RAW LANE ARITHMETIC.** Do not perform raw operations on SIMD values without explicit direction. ALWAYS build the arena — `Kernel` values and the `kernel!` macro — and let the compiler emit the instructions.
- **SIMD is an implementation detail.** `Field` — one SIMD batch, the collapse ABI's vector — is `pub(crate)` in pixelflow-core, and nothing outside that crate can name it, a lane, or a width. Do not change that. pixelflow-core is an algebra; writing it should look like Halide, not assembly.
- **Minimal public API** - Do NOT change visibility of internal APIs without explicit permission. Keep `pub(crate)` and private items encapsulated. Compose `Kernel` values instead of exposing internals.
- **Subtract before you add.** The good version of a primitive is reached by removing machinery, not stacking it. If a type's signature already refuses the wrong shape, you don't need a macro, a lint, or a doc to forbid it — the opinion lives in the types. Reach for a new dependency or a new abstraction only after subtraction has failed.

- **Denote before you build.** Say what a thing *means* — as a mathematical object, in the type system — before writing the code that manipulates it. Design is choosing the denotation; the implementation is then obliged to it. Where this codebase is good, it already works this way: `Lattice`/`DiscreteManifold` are a representable functor whose law is written down (`index(collapse(f)) = f`), and that law is *why* a buffer can BE a manifold rather than merely back one.

  Where it is bad, the meaning lives in a comment instead of a type, and every such place has cost us a bug. One `f32` lane carries continuous values, integers, and bit patterns at once — `OpKind::is_bitwise_domain()` exists to recover at runtime what a type would have given for free, and a mask (all-ones, i.e. NaN read as a number) is one careless fold away from corruption. `Var(u8)` means a coordinate axis or a reduce binder depending on magic ranges — it used to mean a manifold-param slot as well, and that third meaning went out with the macro parameter that needed it. `push_reduce` encodes an `OpKind` as a `Const(f32)`. Each convention held right up until the optimizer grew strong enough to violate it — **a convention written in a comment is an invariant something else will eventually break.**

  So: **when you extend a type's meaning, extend its type.** The escape hatch — "reinterpret is free", "this operand must be a literal", "these coordinates are in the caller's space" — is the moment to pay. Afterwards it is a bug hunt rather than a refactor, and the fix arrives as a runtime guard defending what a type should have made unrepresentable. Prioritize by whether a wrong value would be *silently* representable: a domain confusion produces plausible pixels and deserves a type; an out-of-range index that panics on the next line does not.

### Floating point at the edges

**Rust already ships reasonably fast IEEE-754 arithmetic.** `f32::min`,
`f32::round`, ordered comparisons — correct, conformant, and quick enough for
almost everything. Reaching for pixelflow *is* the decision that "almost
everything" excludes you: the library exists because conformant-and-quick was
judged inadequate on performance grounds. So the trade is made deliberately and
in one direction — **the language gives you the instruction.** Edge-case IEEE
conformance is not on offer, and code that needs it should compute that part in
scalar `f32` where it is already available and already fast.

That is the whole contract. The tables below are reference material for what the
instructions do, not a list of defects, and nothing here should be "fixed" by
spending hot-path instructions to match scalar Rust.

Behavior every target agrees on, pinned by
`pixelflow-codegen/tests/transcendental_jit.rs`:

| Op | Behavior |
|---|---|
| `Lt`, `Le` | ordered — false for NaN |
| `Eq`, `Ne` | **exact** comparison; NaN never equal, always unequal |
| every comparison's *result* | an **all-ones** lane for true, all-zero for false — a mask, never `1.0` |
| `exp`, `exp2` | saturate past ±126 exponents rather than overflowing to `inf` |

A mask is a bit pattern, not a number, and that is a load-bearing distinction:
`Select`'s **mixed-lane path** is a bitwise blend on every backend
(`andps`/`andnps`/`orps`, `vpternlogd 0xCA`, `BSL`) and `BitAnd`/`BitOr` are
literal bitwise ops. (The blend is the path a *lane-varying* mask takes, not
what `Select` is — see "Select contains an if" below.) Spell a
true mask `1.0` and `mask & 1.0` is `0x3f800000`, which blends `7.0` against
`9.0` into `4.5` — a value neither branch held. `OpKind::mask(bool)` is the only
constructor, `OpKind::is_bitwise_domain()` marks the ops whose results are
patterns, and the folder's "refuse non-finite results" guard exempts them —
otherwise an all-ones mask reads as NaN and comparison folding switches itself
off while looking like a safety check.

Behavior that differs by target, because the instructions do:

| Op | x86 | aarch64 |
|---|---|---|
| `Min`, `Max` (NaN operand) | `(a OP b) ? a : b` → the **second** operand | `FMIN`/`FMAX` **propagate** NaN |
| `Min`, `Max` (opposite-signed zeros) | operand order → the **second** zero | `FMIN` picks `-0.0`, `FMAX` picks `+0.0` |
| `Gt`, `Ge` (NaN operand) | unordered (imm8 6/5) — **true** | `FCMGT`/`FCMGE` ordered — **false** |
| `Round` (exact tie) | nearest-**even** (imm 0x00) → `round(2.5) == 2` | `FRINTA` ties-**away** → `3` |
| `Round` (`-0.5 ≤ x ≤ -0.0`) | `-0.0` (sign preserved) | `-0.0` |
| `Recip`, `Rsqrt` | `rcpps` ~12 bits; `vrcp14ps` ~14 | `FRECPE` + one `FRECPS` step |
| `MulAdd` | **one** rounding with `+fma`, **two** without (`mulps`+`addps`) | one (`FMLA`) |
| `TruncToInt` (NaN, or `x >= 2^31`) | `cvttps2dq` → **`i32::MIN`** (integer indefinite) | `FCVTZS` **saturates**; NaN → 0 |
| `Shl`, `Shr` (count outside `0..32`) | count > 31 zeroes the **whole** destination | immediate carries into `immh` → decodes as **`.2D`**, crossing lanes |

(This table had a third column, for a combinator tier that evaluated the same ops in Rust one
SIMD batch at a time. That tier is retired — see
`docs/plans/2026-09-06-kernel-with-a-lattice.md` — so there is one answer per target.)

The `Recip` and `MulAdd` rows are the reminder that "target" is finer than
"architecture": they differ between *ISA levels of the same machine*, which is
what `cargo xtask isa-matrix` exists to keep honest. `Recip`/`Rsqrt` are
estimates — only ever guaranteed close, never equal — so no argument to them is
ever foldable. `MulAdd` is the opposite case and is *not* fold-refused: FMA is
available on every target, some just spell it as a multiply then an add, and
one rounding versus two is a last-bit precision difference inside the contract
(the emitter itself decomposes a `MulAdd` under register pressure on an FMA
target). The folder and the oracle round once (`libm::fmaf`); a differential
check bounds the product's rounding as tolerance rather than skipping the point.

Unifying any row costs instructions — x86 has no ties-away rounding mode, and
NaN or signed-zero blending is a compare plus a select — so none of them are
unified. Portable code should not depend on a single row's answer;
`OpKind::fold_is_platform_specific(args)` marks them, and is value-aware because
the divergence is: `min(1.0, 2.0)` is perfectly foldable, `min(-0.0, 0.0)` is
not. Note `==` cannot see the signed-zero case (`-0.0 == 0.0`), so compare bit
patterns — `1.0 / x` turns it into `+inf` versus `-inf`.

**The one thing this does not license: folding on the wrong machine.** Constant
folding runs on the *build host* at macro-expansion time while the constant
executes on the *target*, so folding a row from the second table bakes the build
machine's answer into a binary that runs somewhere else. That is not a
speed-for-conformance trade — it is simply the wrong value for that target — so
`ConstantFold` declines those cases. Everything else folds.

Corollary, easy to get wrong: **within a target, the optimizer may still produce
a different answer than the unoptimized code.** `Min`/`Max` are not commutative
on NaN (`min(1.0, NaN)` vs `min(NaN, 1.0)` differ on x86), yet the e-graph
installs commutativity for them. That is sound precisely because no value was
promised; it would be a miscompile the moment anything promised one.

The rule that follows from all of it: **take what the hardware gives, and never
hand-roll something worse.** The retired combinator tier's `Round` was the
worked counter-example — `(x + 0.5).floor()`, two instructions where `roundps`
is one, and not any IEEE rounding mode (`round(-1.5)` was -1 there, -2
everywhere else). Slower *and* wrong is never the trade.

### Precision is on the table; range is not

The trade above buys speed with *accuracy* — an answer close to the true one,
off in the last bits. It never licenses an answer **outside the function's
range**. `sin` returning 8.64e8 is not an imprecise sine, it is not a sine; no
budget was saved by computing it, and nothing downstream can recover from it.
So range is a hard property, asserted with no tolerance, while accuracy is a
tunable. **It is currently unasserted**: `pixelflow-ir/tests/trig_range.rs`
made the claim through the scalar interpreter and went with it when the
interpreter was deleted. The property is unchanged and the assertion needs
rebuilding on the JIT — an out-of-range `sin` would now ship green.

Where a function cannot be computed over the whole input type, it gets a
**documented domain** and returns **NaN** outside it. `sin`/`cos`/`tan` are
defined for `|x| < TRIG_DOMAIN` (2²⁰); beyond that, Cody-Waite argument
reduction stops being exact, and full-range Payne-Hanek reduction costs far
more than the polynomial it would protect. Past 2²⁴ the question is meaningless
anyway — `ulp(x)` exceeds 1 radian, so an f32 no longer names a phase.

NaN specifically, and not a clamp into `[-1, 1]`: a clamped value is a wrong
answer wearing a right answer's clothes. That is precisely how the reduction bug
survived — the JIT and the `eval_scalar` oracle (since deleted, along with
every same-form suite built on it) ran the *same* expansion, so they agreed
bit-for-bit on the garbage and every same-form equivalence test
passed, while outputs in the 1e2–1e6 range slipped under the `>1e30`
"ill-conditioned" filter and were admitted as valid training labels. A
same-form check cannot see a shared-definition bug; only an external bound can.
This is the no-silent-failures rule in a data-parallel setting, where a lane
cannot panic but a NaN does propagate.

Two consequences of that domain, both pinned by tests rather than left to drift:
`sin(-0.0)` is `+0.0` (the reduction's last step is `Sub(-0.0, -0.0)`; the
all-positive split that would preserve the sign costs 15× the drift across the
whole domain), and `asin`/`acos` must guard `|x| ≤ 1` explicitly, because the
expansion's `sqrt` is the fast one — it *has* to select 0 for a non-positive
radicand, since `rsqrt(0)` is `inf` and `0·inf` is NaN, which also silently
turns an out-of-domain argument into `atan2(x, 0) = ±π/2` unless you stop it.

Corollary for the two-tier structure: **one definition, imported, not restated.**
`sin` had four copies, each with its own range reduction — the two reachable
ones had the same bug and had drifted to different polynomials besides. The
expansion in `pixelflow-ir`'s `passes` is now the only definition; the copies
went with the tiers that held them. A copy is a future divergence.

### Philosophy

- **Pull-based rendering**: Pixels are sampled, not pushed. Nothing computes until a lattice demands it.
- **A kernel with a lattice is the evaluation API**: `Kernel` describes, `Manifold::compile` specializes it at a shape, `Lattice::collapse` tabulates it. There is no per-batch entry and no interpreter.
- **SIMD as algebra**: users write equations; the compiler owns the loop nest, the hoisting, the pack and the register allocation, and emits the assembly. Lane width is never in the vocabulary.
- **The Fixed Observer**: Camera is at origin. Movement is achieved by warping coordinate space.
- **The language is a DAG**: no iteration binder. A fixed-count iteration is unrolled at construction; a trip count that must change is a recompile through the shape-keyed cache.
- **Zero allocations** - No per-frame heap allocation (ping-pong buffer strategy).
- **Platform on main thread** - Especially macOS Cocoa (Apple requirement).

## Workspace Structure

Cargo workspace with 13 member crates:

| Crate | Purpose |
|-------|---------|
| `pixelflow-core` | Lattices, the compiled `Manifold`, `collapse`, and the cell grid. Backends: x86-64 (SSE2 baseline, AVX2/AVX-512 opt-in via `target-feature`) and aarch64 (NEON) only — no portable/scalar fallback for other architectures. Edition 2024. |
| `pixelflow-compiler` | Proc-macro front end: `kernel!` and `kernel_raw!`, parser, sema, arena lowering, then optimization as an `Optimize` value over the arena. Edition 2024. |
| `pixelflow-ir` | Shared IR. `ExprArena` (sole IR), OpKind enum, the `Kernel` value/AST. |
| `pixelflow-codegen` | Expression graphs to machine code: per-ISA emitters (x86-64, aarch64), register allocation, executable memory, the JIT compile cache (`jit_cache`, `CompiledKernel`). Runs the optimizer itself, so a compiled kernel is never obtained unoptimized. |
| `pixelflow-graphics` | Font loading (TTF, SDF), colors (`Rgba8`, `Color`), the packed frame program, analytic 3-D scenes. |
| `pixelflow-ml` | Graphics ML experiments (harmonic attention, SH feature maps). Not part of the compiler cost model. |
| `pixelflow-search` | E-graph optimization. Rewrite rules, saturation, static latency-prior extraction, rule provenance + hindsight labeling, the saturation Guide. |
| `pixelflow-pipeline` | Measurement harness. JIT bench session, corpus generation (quarantine/split/mint), Guide-program research bins. |
| `pixelflow-runtime` | Display drivers (macOS Cocoa, X11, Metal; Web WASM driver exists but can't compile yet — pixelflow-ir's JIT emitter has no wasm32 backend), input handling, vsync, render pool. |
| `actor-scheduler` | Priority channels with `troupe!` macro. Control > Management > Data lanes. |
| `actor-scheduler-macros` | Procedural macros for actor system. |
| `core-term` | Terminal application: PTY management, ANSI processing, terminal emulator, key translation. |
| `xtask` | Build tooling: macOS app bundling (`bundle-run`), codegen tasks. |

Agent context files for domain-specific knowledge live in `.claude/agents/`.

## Core Concepts

### The Manifold Abstraction

Three objects and one verb:

```text
Kernel ──Manifold::compile(extent)──▶ Manifold ──bind(&[(id, buf)])──▶ BoundManifold
       ──Lattice::collapse──▶ DiscreteManifold
```

A **kernel** is the description (an arena fragment with a root), a **manifold** is that
kernel compiled at a lattice's shape, a **lattice** is the domain, and **collapse** is the
one verb that produces numbers. Kernels compose as values: `Kernel::at` contramaps
coordinates, `Kernel::select` is the conditional, `Kernel::dx`/`dy` differentiate
symbolically, `Kernel::sum_over` and friends are bounded reductions.

### Actor Model

The actor architecture separates input from rendering:

Priority lanes: **Control > Management > Data**

Control/Management prioritize latency over throughput.
Control creates backpressure by timing out senders who are too aggressive. If the timeout exceeds a threshold, an error is returned, likely causing a crash.

### Compiler Pipeline

```
Source → Parser → Sema → Arena lowering → Optimize → Rust TokenStream
            ↓                                 ↓
       Symbol Table                E-graph + latency prior
```

Two representations, the AST and `ExprArena`, and **optimization runs on the arena**.
It used to run on the AST — `kernel!` went AST → e-graph → extracted DAG → back to a
synthesized AST → and only then to the arena the e-graph had already built and discarded.
Each boundary was a place two stages could disagree about what the language is, and three
defects were found there in one week, every one a stage accepting what a later stage
refused. See docs/plans/2026-09-08-macro-tier-is-arena-native.md.

Both macros are `expand(input, optimizer)`; the optimizer is the only difference, and
"do not optimize" is a value rather than a skipped branch — `kernel_raw!` passes
`Identity`. Both emit code that rebuilds an `ExprArena` at load time as a `Kernel` — zero
params gives a `Kernel`, N params a builder closure that folds them in as constants.

**The macro tier does not resolve `Dwrt`.** A surviving `Dwrt` is what makes the chain
rule work under composition: `Kernel::at` warps by substituting into `Var` leaves, so the
warp reaches the `Dwrt`'s operand and differentiates the warped function. Saturation
*would* resolve it (the chain rule is in the rule set, and a `Dwrt` is priced so the
extractor never keeps one), so the macro tier declines any term carrying one and the
runtime tier lowers it at bake time, after composition. Resolving derivatives at expansion
time was a miscompilation for four months, visible only under a warp, and the production
glyph kernels escaped it by coincidence — a `&` mask made the e-graph decline their arena.
`pixelflow-compiler/tests/derivative_under_warp.rs` is the guard.

The compiler uses e-graphs (equality graphs) to find optimal instruction sequences:
1. **Build e-graph** from the arena
2. **Saturate** by applying rewrite rules (associativity, FMA fusion, etc.)
3. **Extract** minimum-cost implementation using the **static latency-prior cost model**
   (`CostModel::latency_prior()` — handwritten per-op cycle estimates, the only policy;
   both tiers choose it through `env_extraction_policy()`)

A learned NNUE extraction cost model was tried (2026-07 to 2026-09) and measured a tie with
the static table on schedule-free expression kernels (the workshop paper on branch
`claude/workshop-writeup`, PR #1072, closed without merging — not in this tree; see
docs/plans/2026-09-01-schedule-cost-model-denotation.md for the citations and numbers in-repo).
That closed its *shape* — a bag-of-edges MLP predicting total cost in place of the table — not
the idea: extraction is where codegen's schedule choice will be made, and a non-additive
schedule cost belongs there as a residual over the table. The shape is deleted (history in
VCS); the seams it needs are kept — the `Reranker` trait over the swap-refinement search
(`egraph/extract.rs`, no implementation shipped), the prior-seeded `OpEmbeddings`, the typed
edge stream, and per-node variance classification (`Extraction::chosen_variance`). The
successor is denoted, not built: docs/plans/2026-09-01-schedule-cost-model-denotation.md.
The e-graph also records **rule provenance** (node origins + union journal,
`pixelflow-search/src/egraph/provenance.rs`), enabling hindsight labeling of which rule
applications were load-bearing for an extraction (`labeler.rs`) — the substrate for the
successor program, the saturation Guide (docs/plans/2026-08-31-guide-design-revision.md).

### ExprArena

`ExprArena` is the sole IR representation everywhere. The old `Expr` (Arc-based tree) is deleted. All paths use arena-based expressions: e-graph extraction, the edge walker, compiler codegen, rewrite rule templates.

## Development Workflow

### CI is the gate

**Green CI is permission to submit.** Not one signal to weigh against your own
judgement — the gate itself. A change that passes goes in; do not hold it back
for a second opinion, and do not attach caveats about what the suite might not
have covered. An advisory reviewer is advisory *because* it does not block: if
a signal is worth blocking on, it belongs in CI, where it blocks.

The corollary is where doubt goes instead. **A gap in CI is a check to write,
not a caveat to attach.** Noticing that no job can catch some class of bug is a
finding about CI's design, and the deliverable for that finding is a test, a
lint, or a job — something that fails next time, for everyone. Prose in a PR
description warns one reader, once, and then is never read again.

And the corollary to that: **when a bug ships green, the retrospective is about
the gate, not the author.** "Should have looked harder" is not a finding.
"This class of bug is invisible to every job we run" is one, and it has a fix.

A CL that touches only `docs/` and Markdown skips the build-and-test jobs
(`scripts/ci-change-scope.sh` classifies the diff; the four metadata jobs still
run). The skip is a job-level `if`, so the required checks report "skipped" and
merge; a workflow-level `paths-ignore` would leave them pending.

Shift left where it is cheap, and *measure* the cheapness rather than assuming
it. A check that costs an hour presubmit belongs in postsubmit — but a fast
fraction of it usually belongs presubmit, and finding that fraction is the
work. `xtask isa-matrix --smoke` is the worked example: per ISA level, running
only the crates whose output *is* per-level machine code costs ~50s against
~344s for the whole workspace, because the build those tests need has already
happened for the lint.

### Build Commands

```bash
cargo build                       # Auto-detects display driver
cargo build --release             # opt-level=3
cargo build --profile dist        # LTO, strip, codegen-units=1
cargo test --workspace            # All tests
cargo test -p pixelflow-core      # Single crate
cargo bench -p pixelflow-core     # Benchmarks
cargo run --release -p core-term  # Run terminal directly
cargo bundle-run            # macOS bundled app
cargo bundle-run --features profiling  # Flamegraph on exit
```

### Build Profiles

- **dev** - opt-level=0, panic=abort. The former opt-level=1/2 workaround for deeply nested expression-template types is obsolete: the JIT-first `Kernel`/`ExprArena` architecture superseded that layer (see `docs/plans/2026-07-20-kernel-unification.md`).
- **release** - opt-level=3, panic=abort
- **bench** - LTO, codegen-units=1
- **dist** - LTO, strip, codegen-units=1, panic=abort

### Workspace Lints

```toml
[workspace.lints.rust]
unused_must_use = "deny"  # Can't ignore Results with `let _ =`

[workspace.lints.clippy]
let_underscore_must_use = "deny"  # Catches `let _ = expr` on #[must_use]
must_use_candidate = "warn"       # Suggests adding #[must_use]
```

All errors must be explicitly handled. No silent failures.

### Toolchain

- **Rust stable** (configured in `rust-toolchain.toml`)
- SIMD backend auto-detected at compile time via `build.rs` and target features
- Platform features automatically selected based on OS

### SIMD Backend Selection

Priority: AVX-512 > SSE2 (x86-64), NEON (aarch64) — no scalar fallback for other architectures. Detection via `build.rs` CPU feature probing + `target_feature` flags. See `pixelflow-core/src/backend/`.

## Code Style

- **Clarity over comments** - Refactor unclear code rather than explaining it
- **Rustdoc (`///`)** for public API, **`//`** for WHY not what
- `match` over `else if` for enums
- Functions < 4 arguments (group into structs)
- No boolean arguments (use enums or separate functions)
- Named constants, no magic numbers
- **Name vs namespace** - When a function name stacks several concepts
  (`compile_arena_dag_jet` = compile + arena-dag + jet), ask: is this a *name*
  or a *namespace*? A namespace inside a name is a smell — it usually means the
  concepts want to be a module, a method on a struct, or a builder, not suffixes
  on a free function. Especially watch for an accreting family of `*_with_ctx`,
  `*_scanline`, `*_jet` variants: that's the cue to introduce the struct/builder.
- **Fold before you dispatch** - a fold leaves fewer live possibilities than
  it found: `if x > 0.0 { x *= -1.0 }` takes "any sign" down to
  "non-positive", and needs no `else` because there is nothing left to say.
  Dispatch does the opposite — it keeps every case alive, and each surviving
  case is carried by everything downstream of it. So these are not two peers
  with separate jobs: fold wherever the cases can be collapsed, and reach for
  `match` (over an enum, or a trait) only for the ones that genuinely cannot
  be. An `else` doing double duty is usually a fold that wasn't taken.

  Guard clauses and early returns are this rule at function scope, and the
  strongest fold available: a `return` doesn't collapse a case into another,
  it deletes the case outright — and it takes the join point with it, which
  is why `else` stops being *discouraged* in code written this way and starts
  being unsayable. Nothing rejoins, so nothing can attach. A function reads as
  a proof: discharge, discharge, discharge, conclude. By the last line exactly
  one case is still inhabited, and the code handling it asks no questions
  because every condition was already spent above it. Flat control flow is the
  symptom of this; eliminating cases is the reason.

  **Branchless is the limit**: no case survives to runtime at all, because one
  expression is correct for every input. It is what this codebase is made of —
  a comparison yields a mask rather than a `bool`, and the language is a DAG
  with no binder — so take it wherever the hardware offers it. `Select` is
  **not** an example of it, however much it looks like one; see "Select
  contains an if" below. What it does not license is hand-rolling a
  *worse* branchless form than the instruction already there: the retired
  `Round` expansion (`(x + 0.5).floor()`, two instructions where `roundps` is
  one, and not any IEEE rounding mode) is the worked counter-example, and
  "Floating point at the edges" above is the long version.

  **Select contains an if.** `Select(m, a, b)` *means* `if m then a else b`,
  and that is two cases, not one. Both arms stay live and everything
  downstream carries both. By this section's own taxonomy `Select` is
  **dispatch**, not a fold — it collapses no case and must not be read as if
  it did.

  So the jump is not an optimization codegen may buy; **the jump is what
  `Select` is.** A batch whose mask is uniform takes an arm — that is the
  conditional, executed. A batch whose mask varies *by lane* is the case a
  jump cannot serve, because different lanes want different arms, and the
  bitwise blend is the fallback for exactly that case. Blend is the
  lane-varying path, not the definition.

  Getting that default backwards is what produced `emit/guards.rs`: with
  blend as the definition, a branch has to be *bought* per select
  (`MISPREDICT_PENALTY_CYCLES`) and, worse, an arm is only eligible when the
  values it owns happen to be one contiguous run of a flat schedule — so
  `cluster_select_arms` permutes the schedule looking for that, in rounds,
  and it measured **73% of a glyph bake** while finding a constant 282 bytes
  (docs/BACKLOG.md, X1). Emit the arms as blocks and there is nothing to
  search for: contiguity is a consequence of building the structure rather
  than a property to be recovered after destroying it.

  The distinction is load-bearing, and getting it backwards has already cost.
  If a select's meaning carries one case, then "which values does this arm
  serve" is an artifact of codegen, and the place to compute it is next to the
  emitter, per select. That is where it was built, and it did not survive
  contact. If the meaning carries two, an arm's condition is a fact about the
  DAG, the region a value is observed over is a property to be *read* rather
  than reconstructed, and `Union`'s explicit ranges and a select's implicit
  mask are the same thing at different levels of static knowledge. See
  docs/plans/2026-09-07-demand-is-a-dag-property.md.
- **Platform `cfg` is encapsulation, not sprinkle** - a platform-predicate
  `#[cfg(...)]` (`target_os`, `target_arch`, `target_family`,
  `target_feature`, `target_pointer_width`, `target_endian`, `windows`,
  `unix`, or a bare arch/os name like `aarch64`/`x86_64`/`wasm32`) is only
  allowed to (1) gate a whole file or module — `#![cfg(...)]` at the top of a
  file, or `#[cfg(...)]` directly on a `mod foo;`/`mod foo { ... }` item — or
  (2) select a platform implementation as a single-line dispatch item in
  `mod.rs` (e.g. `#[cfg(aarch64)] pub use native::Foo as PlatformFoo;`).
  Scattering it on an individual `fn`/`struct`/`impl`/field inside an
  otherwise platform-agnostic file means the platform split is `grep -r cfg`
  instead of a file boundary, and the two halves silently drift back into
  each other's file. Put the platform-specific code in its own file and
  select it once, at the seam. Enforced by `scripts/check-cfg-encapsulation.sh`
  (CI job `cfg-encapsulation`), baselined against pre-existing violations in
  `scripts/cfg_encapsulation_baseline.txt` — new violations fail, old ones
  are tracked, not silently regenerated as debt.
- **New implementation of an existing category → trait first** - Before
  adding a second way of doing something the codebase already does one way,
  check whether that category is already a trait. If it is, implement the new
  behavior as another `impl`. If it isn't, that's the signal the first
  implementation was written before anything needed to vary — extract a trait
  around the existing implementation *first*, then add the new one as a
  second `impl`. Don't grow the second implementation as a parallel free
  function, a copy of the first, or a mode flag/enum bolted onto it. Worked
  example already in this tree: `pixelflow-codegen`'s `RegisterAllocator`
  trait with `LinearScan` as its one `impl` — a second allocator is a second
  `impl RegisterAllocator`, not a fork of `LinearScan`.

  This is "fold before you dispatch" at type scope, which is why the two rules
  don't contradict each other. A case matched at twenty call sites is live at
  all twenty; extracting a trait moves the distinction to the one point where
  the concrete type is chosen, and every use downstream goes unconditional —
  it calls a method rather than taking a branch. Dispatch that happens **once,
  at construction** is not what the fold rule is warning about; dispatch
  **repeated at every use** is. Same factoring as a guard clause, one level up:
  put the case at the boundary so the interior is straight-line.

  That also ranks the ways of varying behavior. A `match` over an enum, or a
  monomorphized generic, settles the case at one point and leaves every use
  unconditional. A `Box<dyn Trait>` does not — it pays the dispatch per call,
  at every use, which is the shape this rule is against; on a hot path it is
  usually better as one of the other two.

## Common Patterns

### Using the `kernel!` Macro

```rust
use pixelflow_compiler::kernel;
use pixelflow_core::{Kernel, Lattice};

let circle = kernel!(|cx: f32, cy: f32, r: f32| {
    let dx = X - cx;
    let dy = Y - cy;
    (dx * dx + dy * dy).sqrt() - r
});

let unit_circle: Kernel = circle(0.0, 0.0, 1.0);
let plane = Lattice::frame(64, 64).bake(&unit_circle);
```

Use `kernel_raw!` to skip optimization (for benchmarking exact expression forms).

### Composing Kernels

```rust
let warped = k.at(&Kernel::x().mul(&Kernel::constant(2.0)), &y);
let selected = mask.select(&if_true, &if_false);
let radius = Kernel::x().mul(&x).add(&y.mul(&y)).sqrt();
// A scalar that is the same at every sample is an argument, not an axis.
let breathing = radius.mul(&clock.kernel());
```

### Actor Message Sending

```rust
handle.send(Message::Control(MyControlMsg))?;    // Highest priority
handle.send(Message::Management(MyMgmtMsg))?;    // Medium
handle.send(Message::Data(MyDataMsg))?;           // Lowest (backpressure)
```

## Platform Notes

### macOS
- Cocoa MUST run on main thread
- `cargo bundle-run` creates `CoreTerm.app`
- PTY I/O: kqueue-based on dedicated thread

### Linux
- X11 via the `x11` crate (feature-gated with `display_x11`)
- Requires: `libx11-dev libxext-dev libxft-dev libfontconfig1-dev libfreetype6-dev libxkbcommon-dev`
- PTY I/O: epoll-based

## Debugging Pitfalls

- **SIMD mismatch between machines**: Check `build.rs` output, verify target features. `RUSTFLAGS="-C target-cpu=native"` to match CPU.
- **Unexpectedly slow**: May be building against a lower ISA level than the CPU supports (e.g. SSE2 baseline on an AVX2/AVX-512-capable host). Check build output and `RUSTFLAGS`/`target-cpu`; there is no separate portable-scalar tier to "fall back" to — see `xtask isa-matrix`.
- **Cocoa main thread panic**: Ensure `pixelflow_runtime::run()` called from `fn main()`, not a spawned thread.
- **"cannot find `Field`"**: intended. `Field` is `pub(crate)` in pixelflow-core. Compose `Kernel` values and collapse them; there is no per-batch value to hold.
- **Why did the e-graph pick that?**: Build with `--features saturation-telemetry` (e.g. `cargo run -p core-term --features saturation-telemetry`) and every production saturation run — macro-tier `kernel!` expansions and runtime-tier `Lattice::bake`/glyph bakes alike — appends a JSONL record (budget, stop reason, cost, wall clock) to `$PIXELFLOW_SATURATION_TELEMETRY` if set, else stderr; see `pixelflow-search/src/telemetry.rs`.
- **A kernel built differently on two machines?** It cannot, by construction, and if it ever does that is the bug: production saturation budgets are denominated in **rule applications** (`SaturationConfig::max_applications`, 20,000/80,000/200,000 blitz/rapid/classical), plus the e-class and iteration caps — all three deterministic functions of the input. Wall clock is **not** a budget dimension; it is `SaturationConfig::safety_ceiling` (30s/120s/300s), a fail-loud assertion that **panics the build** rather than silently truncating saturation and emitting a worse kernel. A panic there means the budget is wrong for that input or the host is pathologically slow — investigate it, don't raise it. `PIXELFLOW_SATURATION_CEILING_MS` (ms; `0`/`off` disables) overrides the ceiling *for diagnosis only*: it can change whether the build panics, never which kernel is emitted. See `docs/plans/2026-09-01-production-budget-determinism.md`.
- **Need rule provenance (origins, union journal, derivation ancestry, the hindsight labeler)?**: build `pixelflow-search` with `--features provenance-journal` (default OFF; `pixelflow-pipeline` and `pixelflow-search`'s own tests enable it already) — without it, `Provenance::origins`/`applications`/`unions` and friends don't exist as types, they don't just return empty; `EGraph::application_count()` (the saturation budget's denominator) stays available either way.

## Execution Notes

- **Hot paths:** the loop nest is inside the emitted code — one collapse call per stripe, not one per row or per SIMD batch
- **Glyph caching:** a glyph bakes once and reads back as a gather over its bound buffer (`fonts/cache.rs`)
- **Glyph coverage:** a winding number about a reference point, per-pixel and discriminant-free (`fonts/loop_blinn.rs`, docs/plans/2026-09-08-loop-blinn-glyph.md). Its bound is a domain-side extent because `u² − v` outside its control triangle is *wrong*, not merely slow — so the glyph is where a `Union` of index ranges earns its keep
- **Antialiasing:** symbolic derivatives — `Kernel::dx()`/`dy()`, resolved before emission. A glyph's *winding* is exact (hard masks selecting signed constants); only the distance feeding the ramp is soft, so a comparison landing on the wrong side costs a rounding rather than half a unit of coverage
- **One kernel per scene:** four channel kernels compile together, so shared geometry is emitted once

## Cost Model and the Guide (offline, supervised)

Two learned programs have had their code removed here, each with its record in the tree:
- The AlphaZero-style self-play/critic/REINFORCE loop, removed July 2026 after a four-agent
  audit found it methodologically unsound (docs/plans/2026-07-07-guided-saturation-redesign.md).
- The extraction head (learned NNUE cost model for extraction): its shape was deleted in
  September 2026 after it tied the static table on schedule-free kernels (workshop paper on
  branch `claude/workshop-writeup`, PR #1072, closed without merging — not in this tree); its
  denotation — schedule cost as the analytic table plus a learned residual that reranks
  extractions — is kept behind the `Reranker` seam and specified in
  docs/plans/2026-09-01-schedule-cost-model-denotation.md, which also carries the citations and
  numbers in-repo. Not built until codegen gives the e-graph schedules to choose.

What remains:
- The static latency prior (`CostModel::latency_prior()`) is the extraction cost model.
  `pixelflow-pipeline`'s `measure_latency_prior` example and `jit_bench` (`BenchSession`,
  median-of-samples, sentinel drift normalization) are how the table is re-derived.
- `gen_bench_corpus` minted quarantined, tier-split expression corpora for the
  Guide program's research bins. **Deleted**: its quarantine gate decided
  acceptance by comparing against the scalar interpreter, so it went with it.
  A corpus needs a new acceptance criterion before that tooling comes back.
- The saturation Guide trains on hindsight provenance labels from `pixelflow-search`'s
  `egraph::labeler` — no critic, no RL (docs/plans/2026-08-31-guide-design-revision.md).
