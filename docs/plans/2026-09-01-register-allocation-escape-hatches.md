# Register allocation outside the register allocator

**Goal: no register allocation outside the register allocator.**

Today two systems assign registers. The allocator assigns DAG values over the
pool `RegisterFile` declares. Beside it, roughly a dozen `const`s in the ISA
files assign registers by hand — invisible to the allocator, unvalidated by
`RegisterFile::checked()`, and correct only by arguments written in comments.

This is not a list of bugs. It is a list of invariants a person maintains that
a type could maintain instead — CLAUDE.md's stated failure mode, in the one
place the codebase most claims to have solved.

## What the allocator actually gets

| backend | file | inputs | pool | reload | select | **pool share** |
|---|---|---|---|---|---|---|
| SSE2 | 16 xmm | 0–3 | **4–9** | 11,12 | 13 | 6/16 |
| AVX2 | 16 ymm | 0–3 | **4–7** | 11,12 | 13 | 4/16 |
| AVX-512 | 32 zmm | 0–3 | **4–9** | 11,12 | 13 | **6/32** |
| aarch64 | 32 v | 0–3 | **16–25** | 26,27 | 28 | 10/32 |

AVX-512 has 32 vector registers and allocates six of them. That is the cost of
the escape hatches stated as a number: the pool was shrunk to leave room for
registers taken by hand, so **the allocator is already paying for them — it
just does not know their names.**

And the bill arrives somewhere specific. A pool that small has no room to hold
loop-invariant values, so the collapse loop's LICM parks every hoisted value in
memory instead. The hand-picked scratch and the hand-managed hoist slots are the
same defect, and the cost lands on LICM.

## The escape hatches

### A. Instruction-level temps — the ISA needs a scratch to express one op

| const | backend | why it exists |
|---|---|---|
| `X86_SCRATCH = Reg(10)` | sse2 | SSE2 is 2-operand destructive; `emit_binary_safe` needs a temp when `dst` aliases the RHS |
| `X86_BUILTIN_SCRATCH = [10,13,14,15]` | sse2 | transcendental expansion holds several live temps |
| `UNARY_SCRATCH = Reg(15)` | avx2, avx512 | same, VEX/EVEX tier |
| `FMOV_FALLBACK_SCRATCH = [28,29,30,31]` | aarch64 | non-encodable constant → MOVZ/MOVK/DUP needs a vector temp |
| `IDX_INT`, `GATHER_DST` | avx512, aarch64 | gather's lane-extract/insert sequence |

### B. Unmodeled register classes

| const | class | note |
|---|---|---|
| `BASE_GPR=x9`, `IDX_GPR=x10`, `VAL_GPR=x11`, `CTX_GPR=x0` | general | aarch64 gather addressing |
| `RAX`, `RDI` | general | avx512 gather |
| `SCRATCH_K = k1` | mask | avx512 compare destination |
| `r9`/`r10`/`r11`, `x5`/`x6` | general | collapse-loop counters (`x86_64::scaffold`) |

`RegisterFile` has no vocabulary for either class. They are not allocated; they
are chosen.

### C. Reload targets — class A in disguise

`reload: [Reg; 2]` and `select_reload` are *declared* and *checked*, but fixed
rather than allocated. The usual justification is a bootstrap: materialising a
spilled value needs a register, and choosing that register by allocation would
need a register.

That bootstrap is an artifact, not a law. It exists only because spilling is an
**annotation on the input schedule** rather than **operations in an output
schedule**. Rewrite the program instead — insert spill/reload ops, as LLVM's
`InlineSpiller` and most modern linear-scan allocators do — and a reload simply
*defines a value*, allocated like any other, with a very short live range. Short
live ranges are what linear scan is best at.

Which makes the real statement: **a reload result and an instruction temp are
the same thing** — a value the input DAG did not contain. Classes A and C are
one missing concept, not two.

> **Correction, 2026-09-03 — half of class C is class D.** The paragraph above
> is right about `select_reload`, which is why that one came out (see the
> landed note under step 2). It is wrong about `reload`, and tracing every
> *read* rather than trusting this section is what shows it. `reload[1]` has
> four use-sites and `reload[0]` two:
>
> | site | where | class |
> |---|---|---|
> | `resolve_operands`' `tmp_op` | inside an instruction | **C** |
> | `resolve_operands`' spilled `dst` | inside an instruction | **C** |
> | the Select guard's mask reload (two call sites) | *between* instructions, in the guard scaffold | **D** |
> | the guarded Select's spilled `dst` | same | **D** |
> | parking a hoist root in its slot | after a def, prologue mode | **D** |
> | the root's return reload | after the whole schedule | **D** |
>
> Only the first two are "class A in disguise" — a need of one instruction,
> which an instruction can declare. The rest happen at points the schedule does
> not contain, so there is no instruction to hang a reservation on, and
> `RegisterFile` has to keep a register for them.
>
> The consequence is the ordering one, and it is sharp: **closing the rest of
> class C on its own frees zero registers.** `reload` stays in the file for the
> scaffold sites either way, so the two registers come back only when class D
> lands with it. That is a second reason to do step 4, independent of
> reclaiming LICM's trip to memory — and it is why the remaining work is one
> piece rather than two.
>
> None of those scaffold sites needs a *dedicated* register, which is what makes
> the combined step tractable: each happens where a free-register set is
> perfectly well defined (the root's reload runs after every value is dead), and
> what they lack is not a register but an allocator that can see the point they
> run at.

### D. Outside the DAG

`SCAFFOLD_ACC = Reg(0)`, `SCAFFOLD_SCRATCH = Reg(1)` — the collapse loop's own
X/Y arithmetic, reusing the *input* registers after the body has run. The reuse
is legitimate; the problem is that a person established it, not the allocator.

It is outside the allocator's view because it is outside the DAG, and it is
outside the DAG because **the DAG is loop-free for the e-graph's sake and the
allocator inherited that constraint for free.** Those are different
requirements: equality saturation needs acyclicity, while liveness is a fixpoint
dataflow analysis that is perfectly well-defined over a loop nest. One object
(`Vec<Def>`) serves both today, so the loop is excluded from an analysis that
could handle it.

The two scaffold registers are the small prize. The large one is `hoist_slots`:
LICM'd values from `frame_hoist`/`row_hoist` are **unconditionally spilled to
memory**, because the scaffold hand-manages them as slots. Hoisting a value out
of a loop *into a load* is a far weaker optimisation than hoisting it into a
register.

## Three apparent collisions — two of them phantom

Three backends have a fixed register whose number equals `select_reload`, whose
doc reads *"Must be untouched by the backend's own Select emission."*

| backend | `select_reload` | named collider | real? |
|---|---|---|---|
| sse2 | `Reg(13)` | `X86_BUILTIN_SCRATCH[1]` | **no** — `emit_const` takes the array as `_scratch` and ignores it |
| aarch64 | `Reg(28)` | `FMOV_FALLBACK_SCRATCH[0]` | **no** — `emit_fmov_imm` likewise ignores it |
| avx512 | `Reg(13)` | gather `IDX_INT` | **yes** |
| aarch64 | `Reg(28)` | gather `IDX_INT` | **yes** |

Tracing each parameter to a *read* rather than trusting the constant's name is
what separates these. Two of the "reserved" arrays are passed to functions that
never look at them, and both `emit_unary`s read only slot `[0]` of a `[Reg; 4]`.

That `[Reg; 4]` is itself a fossil. At the JIT's birth (`3626343f`) `emit_unary`
genuinely read all four slots — the encoder expanded transcendental polynomials
inline. `c781dc44`, *"the last hand-written polynomials leave the assemblers"*,
moved those expansions into `pixelflow-ir`'s passes. The requirement left; the
signature did not follow. A hand-rolled "this op needs N temps" frozen at a
width that stopped being true, with nothing able to notice — which is step 2 in
miniature.

**So the fixed set is far smaller than the constants suggest**: one register on
sse2 and avx2, three on avx512, two on aarch64.

## What the honest accounting reveals

Once the phantom reservations are removed and the survivors declared, every
register is accounted for — and what is left over is provably free:

| backend | file | pool now | provably free | pool could be |
|---|---|---|---|---|
| SSE2 | 16 xmm | 6 | 2 — `xmm14,15` | 8 |
| AVX2 | 16 ymm | 4 | 4 — `ymm8,9,10,14` | 8 |
| **AVX-512** | **32 zmm** | **6** | **16** — `zmm10, zmm17–31` | **22** |
| aarch64 | 32 v | 10 | 5 — `v4–7, v31` | 15 |

(aarch64 excludes `v8–v15`: AAPCS64 callee-saves their low 64 bits, and these
are leaf kernels emitted with no prologue that preserves them.)

AVX-512 allocates six of thirty-two registers while sixteen are untouched by
anything at all. **This is a performance change, not hygiene** — and the win
does not wait for step 2.

One obstruction, and it is another missing denotation. The pool is
`scratch_base: u8` plus `scratch_count: u8` — a *contiguous range*. The free
registers are not contiguous (`{14,15}` sits above `select_reload` at 13;
avx2's are `{8,9,10,14}`), so base+count structurally cannot express the pool
that is actually available. Either renumber everything to keep the range
contiguous, or let the pool become a **set**. A representation that cannot say
the true thing rounds the true thing down.

## How they formed

One cause. The allocator's input language cannot express an operation's
*scratch requirement*:

- `ScheduledOp` variants name only value operands.
- `Def` is one value plus one op.
- `Placement` is one register per value.

So an op needing a temp takes one from outside the pool, by hand, and writes
the disjointness argument in a comment. The pool then shrinks to leave room.
Every hatch above is that same workaround, repeated per backend.

## Closing them

Four steps, each independently valuable, in dependency order.

### 1. Declare the fixed registers (cheap, standalone)

Add `fixed: [Reg; N]` to `RegisterFile` naming every register a backend
reserves for its own use. Extend `checked()` to assert `fixed` is outside the
pool *and* disjoint from `inputs`, `reload`, `select_reload`.

Structurally this closes nothing — it converts three latent collisions into
const-eval failures. **This is the step that would have caught them.** Expect
it not to compile at first: each collision then resolves as either a real bug
or a documented, deliberate exception.

### 2. Let the allocator name values its input did not contain — dissolves A *and* C

The allocator's output becomes a *new* schedule rather than an annotation on
the old one: it may introduce defs the input DAG never had — instruction temps,
and spill/reload results — each allocated like any other value.

- **Temps** dissolve class A: `X86_SCRATCH`, `UNARY_SCRATCH`,
  `X86_BUILTIN_SCRATCH`, `FMOV_FALLBACK_SCRATCH` and the gathers' vector temps
  become allocated values, and their registers rejoin the pool.
- **Reload defs** dissolve class C: `Placement::Spilled` disappears (a spilled
  value is one with a `Spill` op and `Reload` ops), and `reload`/`select_reload`
  leave `RegisterFile` entirely.

Slots stay abstract in that output — a `SlotId`, not an offset — so
`FrameLayout` remains the memory manager and gains the ability to *coalesce*
disjoint live ranges onto one slot, which it cannot do today.

Register pressure *falls* even though more values are allocated: most ops need
no temp, so the pool roughly doubles on x86 and quintuples on AVX-512.

> **Landed 2026-09-03 — the temps half.** `RegisterFile::temps_for(&ScheduledOp)
> -> u8` declares the demand; the allocator reserves a slot before it places
> the destination, excluding every operand, and `Allocation::temp(i)` hands it
> to the encoder through `InstructionPlan::temp`. `UNARY_SCRATCH` is gone from
> all three of AVX2, AVX-512 and aarch64.
>
> The pool effect is smaller than the paragraph above predicts, and worth
> recording rather than rounding up:
>
> | backend | pool before | pool after | why |
> |---|---|---|---|
> | SSE2 | 6 | 6 | still `no_temps` — see below |
> | AVX2 | 5 | 5 | its `UNARY_SCRATCH` *was* `GATHER_IDX` |
> | AVX-512 | 22 | 23 | zmm15 joins `scratch` |
> | aarch64 | 15 | 16 | v29 joins `scratch` |
>
> AVX2 gains no register because ymm15 wore two names: it appeared in `fixed`
> twice, once as the gather's index and once as the unary mask, so the sign
> mask and the select blend had been *borrowing the gather's register* — safe
> only because no single instruction is both, exactly the convention step 1
> exists to make checkable. What landed there is one owner per register, not a
> larger pool.
>
> SSE2 keeps `X86_SCRATCH` in `fixed` deliberately. Its two-operand hazard
> (`emit_binary_safe`) and decomposed `MulAdd` demand a scratch as a function
> of *which registers allocation picked* (`dst == right`), not of the op — so
> the demand cannot be stated as `temps_for` yet, and drawing a second pool
> register for the unary mask meanwhile would cost pressure and free nothing.
> That backend closes when the hazard does.
>
> One thing the design forced out into the open: a temp cannot spill. Every
> value survives a small pool by going to memory, which is why a one-register
> budget was previously legal; scratch the encoder destroys mid-instruction has
> no such escape, so the pool must hold a ternary's three operands plus the
> temp. That floor is `RegisterFile::MIN_SCRATCH = 4`, asserted in `checked()`
> and clamped in `capped()` — `EmitCtx::max_regs` can no longer reach through
> it. Found by probe, not by reasoning: `with_max_regs(1)` on a `Neg` of a
> computed value hit the "unreachable" arm on AVX2 and AVX-512.
>
> **Reload defs (class C) are not done** — `Placement::Spilled` and
> `reload`/`select_reload` are still there. That is where the paragraph above's
> "roughly doubles" actually lives.

> **Landed 2026-09-03 — `select_reload`, the first of class C.** The scratch a
> backend declares became two named roles rather than one
> (`regalloc::Scratch { temp, arm_reload }`), and `select_reload` left
> `RegisterFile`. It was the clearest case in the class: one register held out
> of *every* kernel's pool so that the rare kernel with a `Select` whose result
> and both arms were all spilled had a third reload target, chosen at emit time
> by a three-way case analysis over two fixed registers. It is now a
> per-instruction reservation the allocator makes, disjoint by construction
> from that instruction's operands, destination and temp.
>
> | backend | pool before | pool after |
> |---|---|---|
> | SSE2 | 6 | **7** |
> | AVX2 | 5 | **6** |
> | AVX-512 | 23 | **24** |
> | aarch64 | 16 | 16 — see below |
>
> `arm_reload` is reserved for *every* `Select`, not only the ones that turn out
> to need it, and that is forced rather than lazy. A value resident when its
> reader is allocated can still be evicted by a later instruction — `Placement`
> is one answer per value for its whole life — so residency read at the point of
> reservation is not final. **This is the constraint that shapes the rest of
> class C**: the fixed point between "which values spill" and "how many reload
> registers each instruction needs" is why the remaining work is a redesign of
> the allocator's output contract rather than another field, and why it is not
> attempted here.
>
> Over-reserving could have cost more spilling than the extra pool register
> buys, so it was measured on the shape that maximizes the risk — *w*
> independent `Select` chains all live at once:
>
> | live selects | before (pool 6) | after (pool 7) |
> |---|---|---|
> | 8 | 4 | 4 |
> | 12 | 9 | **8** |
> | 16 | 15 | **14** |
> | 24 | 27 | **26** |
>
> Unchanged or slightly better throughout: the register the pool gains pays for
> the register a `Select` transiently borrows.
>
> aarch64 gains nothing, and finding out why is the point of step 1's
> accounting. v28 was `select_reload` *and* the register
> `emit_skip_if_all_false`/`_true` reduce a mask into with `UMAXV`/`UMINV` —
> two roles, one register, and only one of them declared. Returning v28 to the
> pool would have let the allocator hand it to a live value that a `Select`'s
> own short-circuit guard then clobbered. It is now `GUARD_SCRATCH` in
> `fixed`, where `RegisterFile::checked` can see it. The x86 tiers have no
> equivalent: their guards go through `movmskps`/`kortest` and the flags, so
> they need no vector register at all.

> **Landed 2026-09-03 — the gathers, and class A is closed.** `Scratch` carries
> up to `MAX_TEMPS` registers instead of one, so an encoding can ask for
> several, and every gather's scratch became a per-instruction reservation. The
> gathers were the last of class A: a 256-bit AVX2 gather needs four registers
> (a 128-bit half's index and value, plus one of each to carry the high half),
> AVX-512 and SSE2 two, aarch64 one — each held out of the pool for the whole
> kernel, for an instruction most kernels do not contain at all.
>
> | backend | pool at step 1 | now | of |
> |---|---|---|---|
> | SSE2 | 6 | **9** | 16 |
> | AVX2 | 4 | **10** | 16 |
> | AVX-512 | 6 | **26** | 32 |
> | aarch64 | 10 | **17** | 32 |
>
> Every one is past the "pool could be" column in the table above — that column
> was drawn before temps were allocatable, so it still charged each backend for
> the registers its own encodings borrow. `fixed` is now **empty** on AVX2 and
> AVX-512, and holds exactly one register on SSE2 (`X86_SCRATCH`, whose demand
> is a function of which registers allocation picked rather than of the op) and
> one on aarch64 (`GUARD_SCRATCH`).
>
> `MIN_SCRATCH` rose from 4 to 6 with it, and the reason moved: it is no longer
> a ternary's three operands plus a temp but AVX2's gather — one operand plus
> four temps — with room for the destination either way.
>
> Three tests had a pool size written into them as a literal and stopped
> testing anything when the pool grew past it. `muladd_and_clamp_spilled` said
> "Sethi-Ullman number > 6" and now sizes its fillers from `pool_size()`;
> `belady_evicts_the_value_used_farthest_out` and
> `the_pool_may_straddle_a_reserved_register` are stated in terms of
> `MIN_SCRATCH`. A fourth, `a_spilled_muladd_rounds_twice_on_every_target`, was
> worse than stale: its wall was `(l − r) − (l − r)`, which `legalize` folds to
> a single constant, so there was no wall — it reached the decomposed arm only
> because the pool was smaller than three live values. Its terms are now
> `(X + i) · W`, worth exactly zero at W = 0 and not foldable, because a
> variable is the one thing the folder cannot see through.

> **Landed 2026-09-03 — SSE2's last fixed register, and the hazard it was held
> for was not real.** `fixed` is now **empty on all three x86 tiers**; aarch64's
> `GUARD_SCRATCH` is the only one left in the workspace. The temps block
> predicted "that backend closes when the hazard does" — and the hazard turned
> out not to exist.
>
> `emit_binary_safe` stashed the right operand whenever the allocator chose
> `dst == right` for a non-commutative binary, because SSE2's `dst op= right`
> would corrupt it. **The allocator never makes that choice.** A destination
> takes a pool slot with no live owner, or evicts one — and an evicted value's
> placement becomes `Spilled` for its whole life, so `resolve_operands` reads
> it back from memory rather than from the register the destination took. The
> right operand is therefore a pool register no destination can alias, an
> input register (outside the pool), or `reload[1]` (outside the pool), or —
> when both operands are spilled — `dst` itself, in which case `left` is `dst`
> too. So the case was not a demand for scratch. It was a fallback for
> something unrepresentable, and one xmm was held out of every kernel's pool
> to serve it.
>
> The register moved rather than the argument: the invariant is now stated
> where the registers are *chosen* (a `debug_assert` in `resolve_operands`,
> which fails loudly in every debug build if the allocator stops guaranteeing
> it) and tested where they are *allocated*
> (`a_destination_never_lands_on_a_resident_operand`, which asserts a
> destination never lands on a pool-resident operand and counts the contested
> pairs so it cannot pass vacuously). The three real demands on that
> register — `Neg`/`Abs`'s sign mask, the select blend, and the `movaps`/
> `mulps`/`addps` stand-in's product, this tier having no FMA — are
> `temps_for` answers like every other backend's.
>
> | backend | pool before | pool after | of |
> |---|---|---|---|
> | SSE2 | 9 | **10** | 16 |
>
> Worth measuring rather than assuming, because the ops now reserving a temp
> are `Neg`, `Abs`, `Select` and `MulAdd` — not the gathers most kernels lack —
> and a reservation under pressure evicts an occupant permanently, which is
> exactly what sank the reload-target attempt below. It does not repeat here:
>
> | kernel | before (pool 9) | after (pool 10) |
> |---|---|---|
> | wide 16 | 8 spills / 1340 B | **7** / **1302 B** |
> | wide 32 | 24 / 2876 B | **23** / **2832 B** |
> | abs 8 | 0 / 835 B | 0 / **812 B** |
> | abs 16 | 8 / 1715 B | 8 / **1452 B** (−15.3%) |
> | muladd 8 | 0 / 711 B | 0 / **680 B** |
> | muladd 16 | 8 / 1471 B | 8 / **1192 B** (−19.0%) |
> | select 8 / 12 / 16 | 2 / 6 / 10 spills | unchanged, same bytes |
>
> Same or fewer spills everywhere and never more code. The gain is largest on
> the unary and `MulAdd` shapes, where the temp is now whichever pool register
> is free instead of always xmm10 — a low register needs no REX prefix, so the
> encoding shrinks. `Select` keeps its size because its temp lands on a high
> register either way; the bytes differ, the count does not.
>
> Verified by execution, not identity, per the note below: the whole workspace
> on the SSE2 baseline, and `pixelflow-codegen` plus `pixelflow-ir` on
> `+avx2,+fma` and `+avx512f,+avx512dq` — the JIT-vs-interpreter suites
> (`spill_pressure`, `prod_kernel_jit`, `transcendental_jit`,
> `oracle_reference`) are the differential half.

> **Built and measured 2026-09-03 — frequency-weighted allocation is worse, and
> the reason names what is actually missing.** The carry policy that landed
> above picks roots by raw body-read count and caps them at
> `pool − MIN_SCRATCH`. Both are hand-picked constants, and the symptom was
> shape-sensitivity: a kernel with 6 invariants got 4 of them resident, one
> with 202 got zero, because the *choice* of carry register was made from
> whatever the producing region happened to leave free.
>
> The diagnosis was that the allocator has no cost model, only a heuristic.
> `LinearScan` evicts by Belady — farthest out in **schedule order** — which is
> optimal for cache replacement under the assumption that every miss costs the
> same. A loop nest breaks that assumption: a value read once per inner
> iteration costs one reload *per iteration*, one read in the prologue costs
> one reload, and `last_use` is a schedule index that cannot tell them apart.
>
> So `WeightedScan` prices the reload instead of measuring the distance to it.
> Every read contributes `10^depth`; eviction takes whichever resident value is
> cheapest to bring back; a root is live past its region (`Pricing::live_out`)
> so ordinary eviction keeps it. Carries stop being budgeted and simply fall
> out of the eviction rule.
>
> **It is substantially worse.** Real glyph bakes (257 kernels): carries rise
> 578 → 1021, and emitted code rises **3.01 MB → 4.12 MB, +37%**. Corroborated
> on a synthetic invariant kernel, where the mechanism is legible:
>
> | invariants | LinearScan | WeightedScan |
> |---|---|---|
> | 4 | 0 spills / 498 B | 0 / 492 B |
> | 16 | 0 / 1422 B | **13 spills** / 1937 B |
> | 48 | 0 / 3886 B | **77 spills** / 6641 B |
>
> The cause is a half-priced trade. Carrying is charged to the body — every
> inner scope gets its pool as `file.inside(carried)` — so a carried invariant
> displaces a body value. Both are read at the same depth, so they are worth
> the *same* per iteration; but the displaced one is spilled, which costs a
> store as well as a reload. Pricing the benefit of carrying without pricing
> its cost therefore trades equal-weight reads and pays store traffic on top.
>
> Two things worth keeping from it. First, **`MIN_SCRATCH` is load-bearing in a
> second way**: `LinearScan`'s `pool − MIN_SCRATCH` budget is not only policy,
> it is what keeps the body above the encoding floor, and removing it as a
> "constant to be eliminated" made the allocator run out of evictable registers
> entirely. Second, the floor is *not* a sufficient cap — carrying right up to
> it leaves the body enough registers to emit and not enough to avoid spilling,
> which is the whole 37%.
>
> What the result actually asks for is not a better weight. It is that the
> regions and the body be allocated **against one pool at once**, so the
> invariant's demand and the body's compete directly, instead of the body
> receiving whatever the regions left. That is a global allocation over the
> nest rather than a sequential one, and it is a larger change than this was.
> The code is reverted; the trait sharpening it motivated is not (see the
> commit that made `allocate_nest` the required method), and it is what a
> second allocator will plug into when one is worth shipping.

> **Methodology note.** Byte identity carried #1059–#1062 because those were
> pure refactors: the emitted code was supposed to be unchanged, so an empty
> diff was the whole proof. From here on the emitted code is *supposed* to
> change — freeing registers is the point — so an empty diff would mean the
> change did nothing. The successor technique is a **structured** diff (assert
> that the only bytes that moved are the register fields intended to move)
> backed by differential execution across all four backends.

### 3. The class axis — dissolves class B

`RegClass { Vector, General, Mask }`; `RegisterFile` keyed by class; `Placement`
carrying one. `Gpr`/`Xr` already exist as newtypes; add `KReg`. A *newtype*,
not `type KReg = u8` — a transparent alias is a comment with syntax and would
let any `u8` through.

The gathers' GPRs and `SCRATCH_K` become allocated. This is also the
prerequisite for real k-register predication on AVX-512 — masked ops instead of
blend sequences — which is where that backend should eventually go, and which
is impossible while `k1` is a hardcoded transient.

### 4. Show the allocator the loop — dissolves class D

Give the allocator the whole emitted function, scaffold included, rather than
just the loop-free body. The e-graph keeps its acyclic domain; the allocator
gets a loop nest and computes liveness over it.

`SCAFFOLD_ACC`/`SCAFFOLD_SCRATCH` stop being a hand-argued reuse and become an
allocation the allocator can prove. More importantly, `hoist_slots` stops being
an unconditional trip to memory: loop-invariant values compete for registers on
merit, which is what LICM was supposed to buy.

Until then, class D should at least be *declared* under step 1, so "nothing is
live here" is written where `checked()` can see it.

> **Landed 2026-09-03 — the allocator can see the loop, and LICM stops going to
> memory.** `allocate_nest` was three independent straight-line allocations
> against the same pool; it is one problem now. Regions are allocated
> outermost-first, and a root the innermost body reads can be **carried** — kept
> in a register for the whole of the loops inside, instead of parked in a slot.
> `RegSet::without` and `RegisterFile::inside(carried)` are the vocabulary: the
> pool a scope inside a loop sees is the file's pool minus whatever is carried
> across it, which is what allocating every region against the full pool used to
> ignore.
>
> **What the old shape actually cost, measured before changing anything.**
> `HoistCtx::Body` pinned every hoisted value to `Loc::Spill`, so every *use*
> became a `Reload::FromStack` — on every iteration of the inner loop, for a
> value that provably does not change within it. Uses tracked hoist count almost
> exactly across the suite, and the worst kernel measured hoists **48** values:
> 48 stack loads per batch iteration, for 48 constants.
>
> | kernel | hoisted | uses in body |
> |---|---|---|
> | small | 1–2 | 1–2 |
> | mid | 5, 10 | 5, 10 |
> | prod | **48** | **48** |
>
> Also measured, and worth recording because it is *not* where the cost is: the
> scaffold itself is a constant **268 bytes** on every kernel checked (twelve of
> them, 268 or 269 every time). The loop's overhead is not its size, it is the
> traffic inside it.
>
> **Result, inner-loop bytes, `n` loop-invariant terms each read once:**
>
> | n | SSE2 (pool 10) | AVX2 (pool 10) | AVX-512 (pool 26) |
> |---|---|---|---|
> | 4 | 589 → 498 | 560 → 460 | 634 → **526** |
> | 8 | 897 → 806 | 820 → 720 | 938 → **722** (−23%) |
> | 16 | 1513 → 1422 | 1340 → 1240 | 1546 → **1114 (−28%)** |
> | 48 | 3977 → 3886 | 3420 → 3320 | 3978 → **3438** |
>
> **Zero added spills at every size on every tier.** Code size understates the
> win: each byte removed is a load that was executing every iteration.
>
> The shape of that table is the point. The carry budget is
> `pool − MIN_SCRATCH`, so the 128-bit tiers saturate at four carries while
> AVX-512 gets twenty — **the backend with the most registers gains the most,
> having previously gained nothing from having them.** That is the same
> observation step 1's accounting opened with, arriving from the other end.
>
> **The safety property, and the test that states it.** A carried register is
> read by the body on every iteration, so anything inside the loop writing it is
> a miscompile that surfaces only as wrong pixels. It holds by construction —
> the body's pool is `file.inside(carried)` — but the carry is chosen from what
> the *producing* region leaves free, and that had a trap worth recording: the
> first version took the complement of `placements`, which does not include
> instruction **temps**. A temp is a pool register no `Placement` records, so
> that version could hand out a register the prologue destroys.
> `a_carried_register_is_untouched_by_everything_inside_the_loop` checks
> placements *and* temps *and* `arm_reload`, and asserts something was carried
> at all — which immediately caught its own first draft, written against a file
> whose pool is exactly `MIN_SCRATCH` and whose budget is therefore zero.
>
> **What did not land: `SCAFFOLD_ACC`/`SCAFFOLD_SCRATCH`.** They are still
> hand-chosen, and they are still safe, for a reason this work makes precise
> rather than removes: they are `Reg(0)`/`Reg(1)`, *input* registers, which
> `checked()` already holds outside the pool — so a carried pool register
> survives the scaffold untouched, and the two of them survive the body. What
> makes them safe is that every coordinate round-trips to a slot and is reloaded
> at the top of each iteration. Keeping the **coordinates** in registers is the
> other half of class D — induction variables, not invariants — and it is what
> would finally make those two an allocation rather than an argument. Separable,
> and not attempted here.

## Order of attack

1 first, alone: it is small, it is a strict improvement, and it tells us
whether the three collisions are bugs before any redesign is built on top of
them.

Then 2 — the largest single win, and it retires two of the four classes at
once.

Then 3 when k-predication is actually wanted, and 4 when LICM's trip to memory
is worth reclaiming. Neither is a prerequisite for the other; both need 2
first.

**Revised 2026-09-03.** Step 2's class-A half is done and `select_reload` with
it; what is left of 2 is `reload` and `Placement::Spilled`. Two findings move
the rest:

1. **The remainder of 2 has to land with 4, or it buys nothing** — three of
   `reload`'s use-sites are outside the DAG (see the correction under class C).
2. **It needs live-range splitting, and this was measured, not argued.**
   Eviction writes `Placement::Spilled` for a value's *whole life*, so an
   operand resident when its reader is allocated can be spilled retroactively
   by a later instruction. Every reservation that landed so far dodged this
   because its demand is a function of the *op* — `temps_for` can state it.
   A reload target's demand is a function of *which operands are in registers
   at that point*, which is not final when the reservation is made.

   Restricting eviction to values nothing has read yet does make an operand's
   residency final, and it is cheap: +1 spill across every size measured. The
   whole change was built on that and it still failed, for a reason worth
   writing down rather than rediscovering.

   **Reserving a register that is needed for one instruction costs a value its
   register for the whole program.** Under pressure no pool slot is free, so
   the reservation evicts an occupant — and eviction is permanent. Every
   spilled operand therefore causes another spill, and constants are operands:

   | wide kernel | pool 9, `reload` fixed | pool 11, targets reserved |
   |---|---|---|
   | 4 terms | **0** | 4 |
   | 8 terms | **0** | 8 |

   Two more registers do not come close to paying for that. The two fixed
   registers are cheaper than the pool ones precisely *because* they are
   outside the pool: taking one costs nothing.

   And the destination cannot be reserved for at all. `evictable` makes an
   operand's residency final from its first *read*, but a value's own
   definition is a write — so a destination allocated a register can still lose
   it later, which is exactly what `reload[0]` exists to catch.

   So the remainder is not a reservation problem. A reload has to become a
   *value with a short live range* — the "output schedule rather than an
   annotation" this plan already names — because only then does materialising
   one cost a register for an instruction instead of for a program.

### The splitting version, built and measured

Built next, since that is what the paragraph above asks for. Eviction stops
writing `Placement::Spilled` over a whole life and instead **splits**: the value
keeps its register up to the eviction, a store hands the register on there, and
later reads name a memory-resident half the input never contained. On top of
that, every reload target and the destination scratch become per-instruction
reservations, `reload` leaves `RegisterFile`, and each pool grows by two.

Splitting is what makes the reservations sound, and it is worth writing down
why: a value evicted later keeps the register it held *earlier*, so residency
read when an instruction is allocated is final, and a destination holds a
register at its own definition or never. Both were exactly what the previous
attempt could not guarantee.

Measured, same kernels as above, on the SSE2 tier:

| kernel | before (pool 9) | after (pool 11) | code size |
|---|---|---|---|
| wide 16 | 8 spills / 2127 B | 8 / 1903 B | **−10.5%** |
| wide 32 | 32 / 4480 B | 32 / 3834 B | **−14.4%** |
| anchored 12×40 | 5 / 1329 B | 4 / 1132 B | **−14.8%** |
| anchored 16×64 | 9 / 2047 B | 8 / 1640 B | **−19.9%** |

Same or fewer spills everywhere and 0.7–20% less code, the gap widening with
pressure. The whole codegen suite passes on all three x86 tiers, and the
workspace passes apart from the case below.

**It is not merged, because of one hole in guarded regions.** A `Select`'s
short-circuit arms are instructions a branch can skip, which constrains where a
split's store may go:

- A value defined *before* an arm can store before the arm's branch — that path
  always runs.
- A value defined *inside* an arm has no such point. Storing at its definition
  is right only if no *nested* guard sits between there and the eviction;
  ignoring that nesting miscompiles glyph bakes, which
  `kernel_glyph_golden.rs` (deleted) catches.
- A `Select`'s own operands cannot be split at all: the guard reads its mask
  under the name the pre-redirect analysis recorded, and the arms are read at
  the `Select`, outside both ranges.

Those three leave a residue of registers that can be neither split nor spilled
whole — the latter needs the value to be unread so far — and an instruction
inside a guarded region can find every register in that state and have nowhere
to put its scratch. It is reachable: a warm glyph cache reaches it.

Closing it means the store point has to be chosen against the *nesting* of
guard ranges rather than a flat "is this index guarded", and the `Select`
operands need the guard analysis to name values by their post-split identity.
Neither is deep, but both are the kind of thing that miscompiles quietly, and
the measurement above is the reason it is worth doing properly rather than
quickly.

> **Landed 2026-09-04 — the output is a schedule.** Two pure refactors of the
> allocator's output types, with no change to which values get registers
> (#1148). A `Placement` is now a non-empty, strictly increasing sequence of
> `Span { from: Point, at: Where }` — from that program point until the next
> span's, the value lives at `at` — and `Point` is nest-wide: `(Scope, index)`
> ordered `Region(0) < … < Body`, execution order on the first pass and
> deliberately not a trip-count model. `NestAllocation` holds **one** placement
> map for the whole nest; `carries`, the per-region maps, `RegionAllocation`,
> `HoistCtx::{Body,Prologue}::carried` and `Allocation::spilled()` are deleted.
> A root is carried iff its placement at the body's first point is `Reg(_)`.
> The answer lives in the type, not beside it.
>
> Proved rather than asserted: a harness compiled 61 kernels × 3 pool sizes and
> recorded `(len, digest, spills, frame, hoisted)`; the diff against `main` is
> **empty on SSE2, AVX2+FMA and AVX-512 for both commits** — the second more
> strongly than its spec required, which asked only for the no-hoist subset.
>
> **Two places the spec was wrong, both now pinned by tests.** `ValueId`s are
> *not* partitioned by the nest: `plan_collapse_hoist` refuses to hoist a leaf
> (nothing is saved parking a value one instruction rebuilds), so a `Const` or
> `Var` feeding both an invariant and a varying term is scheduled in *both*
> scopes and placed independently in each — one span per scope, which a single
> answer per value could not hold (`a_leaf_feeding_both_scopes_is_scheduled_in_both`,
> `a_shared_leaf_is_placed_once_per_scope`). And a carried root needs *two*
> spans, not one: the carry register is chosen from what the producing region
> leaves free — that is what makes it safe — so it is never where that region
> put the value (`a_root_is_placed_twice_and_says_for_itself_whether_it_is_carried`).
>
> `FrameLayout` stays per scope; hoist slots stay the collapse driver's.
> Unifying them means the layout owns the scaffold's coordinate slots too — a
> frame-ABI change, recorded rather than half-done.
>
> **Why this is the step before the policy, not the policy.** The two
> measured negatives above — frequency weighting (+37%: pricing one side of the
> carry trade) and whole-life splitting (−10 to −20% but a hole where the
> store went) — want the same thing: the region's roots and the body's values
> competing for **one pool in one pass**, with eviction as a *split* rather
> than a whole-life sentence, the store at the definition (which a guard cannot
> skip without skipping every read), reloads that *keep* their register, and
> reconciliation at each scope's head for whatever the back edge left elsewhere.
> That is Traub's second-chance binpacking over the nest, and it needs exactly
> this output shape. Same trait, same types; the change is `LinearScan`'s
> policy and the emitter honoring transitions it already receives typed. Not
> graph coloring: SSA interference graphs are chordal, so schedule-order greedy
> coloring is already optimal *as a coloring* — what was missing is spill
> placement, which is what splitting is.

> **Landed 2026-09-04 — eviction splits a live range, and the remaining half
> of class C's hole is closed.** (#1150.) The loser of an eviction keeps the
> register it held up to that point; its life continues in its slot; a later
> read may take a pool register back and *keep* it. Eviction ranks by
> traffic before distance — a constant (rematerialized), then a value whose
> slot is already valid (no store), then one that needs a store — with
> Belady breaking ties inside a tier and "read by this very instruction"
> above everything. The store for a spilled value goes at its **definition**,
> which a guard cannot skip without skipping every read; that is what the
> earlier attempt got wrong by storing at the eviction, and it is why the
> guarded-region hole above does not reappear. A kept range inside a
> `Select` arm, for a value defined outside it, ends where the arm does
> (`guarded_arms`, from the same `analyze_select_guards` the emitter branches
> on — now in `emit/guards.rs` so both call one function).
>
> **Measured on 259 real glyph bakes, memory operations counted at the
> emitter:**
>
> | | SSE2 `main` | SSE2 after | AVX-512 `main` | AVX-512 after |
> |---|---:|---:|---:|---:|
> | code bytes | 3 046 146 | 2 856 850 (**−6.2%**) | 2 115 416 | 1 790 097 (**−15.4%**) |
> | reloads | 22 011 | 20 199 | 14 922 | 14 283 |
> | stores | 15 101 | 16 421 | 12 568 | 12 589 |
> | memory ops | 37 112 | **36 620** | 27 490 | **26 872** |
> | frame slots (`spill_count`) | 1 822 | 3 142 | 358 | 379 |
>
> wide 32 on SSE2: 2832 → 2234 bytes (−21%), same spill count.
>
> **Two acceptance bounds the spec set were wrong, and the table is why.**
> "Spill count never higher" counted `FrameLayout::slots` — values with an
> address — and splitting is precisely the change that gives a value a
> register for most of its life and a slot for the rest, so the count rises
> (+72% on SSE2) while the traffic falls (−492 memory ops). Slots stopped
> being a traffic metric at this commit; **memory-op count is the number to
> measure from here on.** "No kernel ever larger" is not a property a
> traffic-priced heuristic can offer under a full pool: the tier that buys
> the −21% is the one that costs the worst outlier (+33 bytes, +3.0%, 9 of
> 222 combinations), and splitting under the old eviction rule is perfectly
> monotone and buys 0.0%. The bound that means something is the aggregate
> plus a worst-case cap.
>
> **A miscompile shipped to CI and was caught there, and the retrospective
> is about the gate.** The emitter patched a guard's skip-branch to the
> point *after* the reconciliation that begins at the arm's end index, so
> the uniform-mask path jumped over a reload the location table said had
> happened. Only splitting could put a reload there; only a uniform mask
> could reach it; and only the VEX tiers and NEON exposed it, because SSE2's
> `temps_for` reserves a `MulAdd` scratch the others do not and so allocates
> a different schedule for the same kernel. The failing test was
> `pixelflow-core`'s `packed_bake_is_bit_exact…`, on the macOS job — and **no
> presubmit job ran `pixelflow-core` above SSE2.** The smoke set's own rule
> ("the crates whose output is per-level machine code") already included
> it; it was simply missing. It is in the set now (`smoke: codegen+ir+core`,
> ~20s per level), the fix patches the join before the reconciliation, and
> `guarded_arm_reconciliation.rs` (deleted) builds the shape on purpose and fails on
> the parent commit. An allocator-level invariant check cannot see this
> class: the placement is self-consistent and every operand resolves; only
> the order of two emissions at one index is wrong, and the guarded path has
> to actually run.
>
> **Not done here: the third piece.** `allocate_nest` still allocates each
> region against `file.inside(carried)` and picks carries from what the
> region leaves free under the `pool − MIN_SCRATCH` budget. Two findings from
> building this feed that piece: one `reg_owner` across scopes needs
> **per-definition** lives, not per-`ValueId` (a leaf scheduled in two scopes
> is two definitions of one name); and in the collapse ABI the only values
> that cross a scope boundary are parked roots, whose body-schedule
> placeholders (`Const(0.0)`) each hold a body register for their live range
> while the emitter reads the real value from a slot — a register wasted per
> parked root that one pool over the nest removes by making the root the
> value in every scope.

> **Measured 2026-09-04 — one pool over the nest, as a flat pass, is worse
> than the budget it replaces.** Built in full after #1150 and not merged
> (kept as `wip` commit `814710c2` on the agent's worktree branch so the
> implementation and the measurement survive). Regions and body allocated
> in one forward pass over `Point` order with one `reg_owner`, roots live
> to the tail, head reconciliation as a slot load, the carry budget and
> `RegisterFile::inside` deleted. Every suite green at SSE2. Glyph corpus,
> 259 kernels, SSE2, memory ops counted at the emitter:
>
> | | bytes | memory ops | frame slots | head reloads |
> |---|---:|---:|---:|---:|
> | #1150 (carry budget) | 2 856 850 | 36 620 | 3 142 | 0 |
> | one pool, Belady over the flat index | 3 256 120 (+14.0%) | 46 221 (+26.2%) | 12 124 | 1 690 |
> | one pool, loop-carried lives un-evictable | 3 997 747 (+39.9%) | 137 774 (+276%) | 49 766 | 966 |
>
> **The denotation above was wrong where it said the frequency difference
> "lives between scopes" and a wrapping next-use would price it.** Belady's
> distance in the concatenated index space counts *static* positions; an
> eviction costs *dynamic* ones. A body value read five instructions later
> is read once per iteration; a root read three hundred flat positions later
> is read on every iteration of every loop in between. The flat pass ranks
> the root as the cheaper thing to give up, so roots spill (slots ×4 with no
> other policy in play). Making loop-carried lives un-evictable is the
> design's own correction and it overcorrects exactly as the frequency
> weight did (+37%, above): hundreds of invariant roots against a
> ten-register pool leave the body allocating against the remainder.
>
> So both directions have now been measured. Weighting reads by depth
> over-carries; a flat distance under-carries; pinning loop-carried lives
> over-carries worse. The carry budget was not only a brake — allocating
> each region against `file.inside(carried)` also gave the region's own code
> the whole pool minus a few reserved registers, which a single pool cannot
> express when region values and body values compete on a metric that is
> wrong for both.
>
> Four pieces of that commit are right independently of the pooling
> decision and are the reason to keep it reachable: per-definition lives
> (`next_read` stops at the next definition of the same `ValueId`, since a
> leaf scheduled in two scopes is two definitions of one name); the
> partitioner's `Const(0.0)` placeholder treated as neither a definition
> nor a constant (believing its op made every parked root rematerializable
> as zero; believing it a definition ended each root's life at its own
> placeholder — two separate miscompiles the suites caught); scratch
> exclusion asking where an operand *is* rather than who owns the slot
> (`expire` frees a register without recording a range, and the two
> disagree exactly for a value whose last read is the current instruction);
> and the emitter following a live-in that moves.
>
> **What this points at is not another tier.** The question the budget
> answers by constant — how many registers may roots take from the body —
> has a measurable answer: the body's own peak demand. Scan the body once
> with nothing carried, read its peak register demand `P` (residents plus
> that instruction's scratch), and the carry budget is `pool − P`: no floor
> constant, zero when the body is saturated, the whole slack when it is
> not. Choose carries from the roots the body reads most, then scan the
> *region* with those roots **pre-colored** to their carry registers, so a
> root is computed straight into the register that carries it and the
> choice no longer depends on what the region happened to leave free — the
> shape-sensitivity that gives 202 invariants zero carries today. That is
> two extra scans per nest, monotone with respect to body spilling by
> construction, and it replaces the hand constant with a measurement rather
> than a weight. Next.

> **Measured 2026-09-04 — replacing the carry budget: two more designs,
> both worse on AVX-512 wall clock; the constant stands.** Two further
> attempts after the flat pass above, both preserved on branches rather
> than merged (`measured-budget-be33ec66`, and `355f1ff1` on the agent's
> worktree branch).
>
> **3′ — budget from the body's measured peak, carries pre-colored.**
> Budget `pool − max(peak of every scope inside, encoding floor of the
> region)`; carried roots computed straight into their carry register; a
> parked root pinned to its slot in inner scopes rather than a placeholder
> holding a body register. Static memory ops fell everywhere (glyph corpus
> SSE2 −14.6%, AVX-512 −7.4%; frame slots 3 142 → 580) but **bytes rose
> +19% on AVX-512** and on every invariant kernel: the body's gain is paid
> out of the once-per-row prologue's pool, and on a 26-register tier the
> budget hands out many carries whose body saving is small.
>
> **3″ — the same, with carries priced by trip count.** The units problem
> in 3′ is real and the trip counts are *already available*: `Lattice::bake`
> passes `LatticeShape` to `jit_cache::compile`, which keys on it and gives
> it to the optimizer but never to the emitter. 3″ plumbs it through
> (`EmitCtx::with_shape`, a per-region `trips`) and accepts a carry only when
> the measured **dynamic memory ops per call** — Σ scopes (memory ops in
> scope × executions per call) — fall. Glyph corpus at real bake shapes:
>
> | | SSE2 #1150 | SSE2 3″ | AVX-512 #1150 | AVX-512 3″ |
> |---|---:|---:|---:|---:|
> | dynamic memory ops / call | 1 722 997 | 1 563 949 (**−9.2%**) | 550 293 | 513 948 (−6.6%) |
> | code bytes | 2 849 452 | 2 739 826 (−3.9%) | 1 780 782 | 2 121 876 (**+19.2%**) |
> | carries | 577 | 132 | 1 630 | 2 328 |
> | kernels worse than #1150 (dyn ops) | — | 36 / 257 | — | 14 / 256 |
> | kernels worse than own zero-carry baseline | — | **0** | — | **0** |
>
> Wall clock, `font_rendering` bench (glyph at [40,45], JIT cached, release,
> medians): SSE2 −1.6% to −6.5%, inside the spread; **AVX-512 +13% to
> +18%, outside it.** Dynamic memory ops went down and time went up.
>
> **What the verification found, and why this line stops here.**
>
> 1. **The measurement never refuses anything.** Of 11 558 carry candidates
>    on SSE2, 11 426 were clipped by the budget before the trip-count
>    pricing ran, 132 accepted, **0 refused on cost** (4 of 11 537 on
>    AVX-512). "Monotone by construction" holds against the design's own
>    zero-carry baseline and is a property of the budget, not of the
>    pricing. The 36 SSE2 regressions against #1150 are all kernels whose
>    body peak saturates the 10-register pool, so the budget is 0 with
>    candidates waiting — carries #1150's `pool − 4` took, and was right to.
> 2. **The invariant kernels show no shape sensitivity at all**: identical
>    dynamic memory ops main vs 3″ at every n, and identical carries at
>    x = 256 and x = 16, because the budget is 0 in both cases and the trip
>    count only enters a test the budget prevents from running. 3″'s carry
>    count at 84/202 is 0 where #1150's is 4 (SSE2) / 20 (AVX-512) — the
>    reverse of what 3′ measured on a different expression.
> 3. **Static memory ops do not predict wall clock on AVX-512.** −6.6% dyn
>    ops, +19% bytes, +13–18% time. The structural changes alone (zero
>    carries) are **+12% dyn ops worse** than #1150 there while −9% better on
>    SSE2; carrying is the whole AVX-512 win on that metric and the whole
>    byte cost. Whatever the time is going to — a 42 KB kernel does not fit
>    L1i; a 64-byte spill is not the same cost as a 16-byte one — the cost
>    model that decided these trades cannot see it, and tuning further
>    against it is not justified.
>
> So: the flat pass under-carries, the peak budget over-carries on the wide
> tier and under-carries on the narrow one, and pricing by trip count never
> gets to decide. `pool − MIN_SCRATCH` with roots ranked by body reads
> (#1147/#1150) remains the best-measured policy on every tier by wall clock,
> and it stays. The constant is a fitted number, and the honest statement is
> that three principled replacements each lost to it because the quantity
> they optimized is not time. The trip-count plumbing is the right
> denotation and is kept on the branch, not landed: unused plumbing is
> machinery, and the rule is subtract first.
>
> What would change this: a cost model whose prediction tracks the
> `font_rendering` wall clock across tiers — code size and vector width in
> the cost of a spill, at minimum — measured before any policy is built on
> it. That is a cost-model program, not an allocator change, and it is what
> `docs/plans/2026-09-01-schedule-cost-model-denotation.md` is for.

> **Landed 2026-09-05 — `reload` leaves the register file, and class C is
> closed.** The two fixed reload registers were the last hand-chosen
> registers in the workspace. What made them dissolvable is what #1150
> established: eviction splits a live range rather than rewriting one, so a
> value in a register when its reader is *allocated* is in a register when
> its reader is *emitted*. Residency is final, and a reload target can be a
> per-instruction reservation exactly as `temps_for` temps already are.
>
> Each of `reload`'s four roles, and what replaced it:
>
> 1. *Transient operand target* (`reload[1]`): one reservation per operand
>    that is non-resident at its reader, `Scratch::reloads`, chosen by **one
>    function** — `emit::operand_sources(op, resident)` — that the allocator
>    calls to count and the emitter calls to name, so the two cannot drift.
>    `arm_reload` turned out to be this and was unified away. The
>    dst-as-target rule is kept and stated (a `Select`'s mask, an FMA's
>    addend, a two-operand binary's left reload straight into `dst`), which
>    is why two reload slots suffice for three operands.
> 2. *Destination of a value spilled at its own definition* (`reload[0]`):
>    every definition that emits an instruction holds a pool register at its
>    definition — a value that would have lost the contest takes the loser's
>    register, stores (rule 3), and its `Reg` span ends at the next point.
>    `resolve_operands` now panics on a spilled destination, and
>    `InstructionPlan::store` is deleted from all four backends. The one
>    definition that emits nothing is a rematerialized `Const`, whose
>    `Where` at its def is `Remat` — previously a wasted `LoadConst`.
> 3. *Guard scratch* (`guard_scratch = reload[0]`): reserved on the
>    instruction a guard is emitted before — `guard_mask` when the mask is
>    non-resident there, `guard_temp` when the backend asks
>    (`RegisterFile::guard_temps`: 1 on aarch64, 0 on x86).
>    `emit_skip_if_all_*` takes `Option<Reg>`.
> 4. *Park and result resolves*: the park path is resident by construction
>    (a hoist root is non-leaf, so its definition just wrote a register) and
>    reads it directly; the scope result needs a reservation only when the
>    whole body was hoisted and its root is read from a park.
>
> `MIN_SCRATCH` is 7, derived where it is defined as the widest single
> demand: AVX2's gather at a guarded arm's head (4 temps + 1 operand + 1
> guard + 1 dst). Pools: SSE2 10→12, AVX2 10→12, AVX-512 26→28, aarch64
> 18→20 — each now exactly the registers the ABI leaves unassigned minus
> the callee-saved ones.
>
> **The guarded-arm residue is gone, and here is why.** At any instruction
> the pool holds ≥ `MIN_SCRATCH`, the instruction claims at most that many
> roles, and every other slot is free or holds a value that can be split.
> The store for a spilled value goes at its definition, and a guard only
> skips a definition by skipping every read of it, so the slot is valid on
> every path that reads the value, inside an arm as outside. Nothing is
> unevictable; no reservation can fail. Pinned by
> `a_reservation_inside_a_nested_guarded_arm_always_has_a_register`
> (nested guarded `Select`s at `MIN_SCRATCH`/+1/+3, widths × depths, all
> nine mask combinations, against the scalar oracle).
>
> Measured against `main`, SSE2 (282 kernels through the graphics suite's
> real bakes; `anchored` is a wavefront, since the e-graph reorders
> independent chains and their pressure evaporates):
>
> | | bytes | frame slots | memory ops | dynamic memory ops |
> |---|---|---|---|---|
> | graphics corpus | −6.5% | 6 237 → 3 261 | −12.0% | −13.1% |
> | anchored 12×40 / 16×64 | −9.2% / −9.6% | 125→114 / 446→410 | −8.8% / −7.5% | same |
> | wide 16 / 32 | −1.2% / −1.3% | = | 12→11 / 28→27 | −8.3% / −3.6% |
>
> AVX-512: corpus bytes −3.7%, slots 1 083 → 861, traffic flat; anchored
> 16×64 memory ops −25%. Nothing regresses on any metric on either tier.
> Wall clock (`font_rendering`, release, median of 7): SSE2 −0.1% / −7.8% /
> −2.0% on the three glyphs, AVX-512 within the run-to-run spread. The
> plan's −10 to −20% bytes prediction was not reached (−9.6% at best):
> that figure came from the prototype whose pool grew from a *smaller*
> floor, and `MIN_SCRATCH` 6→7 spends one of the two registers back.
>
> Two things the brief for this had wrong, both found by checking rather
> than reading: the park path needed no reservation, and `live_in` had to
> reach the scan — a parked root's residency is the *enclosing* scope's
> answer, while its body-schedule entry is a `Const(0.0)` placeholder the
> old scan called "resident" (the same trap the 2026-09-04 notes record).
>
> With this, every register a kernel uses is chosen by the allocator.
> Class A closed 2026-09-03, class C here, class D's invariant half
> 2026-09-03; class B (`RegClass`) and class D's coordinate half remain
> declared, not dissolved.

> **Built and measured 2026-09-05 — the cost model the previous block asked
> for, and what it turned out to be missing: rematerialization.** The
> carry-budget block ends with a condition — *"a cost model whose prediction
> tracks the `font_rendering` wall clock across tiers … measured before any
> policy is built on it"* — so the next thing built is the measurement, not a
> policy. Measured against the five allocations that existed when this was
> written; `main` below is #1150, and the class-C landing directly above
> postdates the sweep (see the last paragraph).
>
> **The harness.** `pixelflow-pipeline`'s `collapse_cost` (the crate's
> `collapse_bench` module) in three subcommands. `capture` writes a corpus
> **fixture** — `(arena, root, extent)` per kernel, one text file each, the
> arena dumpers' node encoding plus the shape — so every allocation is
> measured on byte-identical input rather than on whatever each build's kernel
> construction happened to produce. `bench` compiles each entry exactly as
> `Lattice::bake` does (`optimize_runtime_arena` at the kernel's own shape,
> then `emit::compile`), times `call_collapse` into a buffer allocated once,
> and writes one JSONL row per (kernel, pass) carrying the emitted code's
> static features beside the clock. `analyze` scores closed-form predictors
> over those rows. Compilation, saturation included, is outside the timer; so
> is the scalar tail `bake` walks after the vector groups, which is not this
> kernel.
>
> The static half is `pixelflow-codegen`'s new `emit::traffic`: a counting
> decorator over the private `IsaBackend` seam, so every byte the driver emits
> is counted by construction and a new emission path is counted the day it is
> written rather than the day someone remembers to add an increment. It counts
> **per scope** — frame prologue, row prologue, body, scaffold — because a body
> instruction runs `rows × groups` times and a prologue instruction once, and
> that weighting is the entire question. Nothing in the emitter or the
> allocator reads it.
>
> ```text
> collapse_cost capture --out <dir>                       # once, any build
> collapse_cost bench   --corpus <dir> --out <rows.jsonl> \
>     --git-ref <variant> --git-sha <sha> --passes 1      # per variant x tier
> collapse_cost analyze --rows <rows...> --out report.md  # once, over all of them
> ```
>
> **The corpus: 208 kernels with shapes.** 190 atlas glyph bakes (printable
> ASCII at both display densities, `[16,16]` and `[32,32]`), the three
> characters and the `[40,45]` lattice `font_rendering` itself bakes, and 15
> synthetic kernels in three families — `wide{n}` (balanced tree, transient
> pressure only), `anchored{w}x{d}` (`w` live ranges crossing a `d`-deep
> chain), and `invariant{n}` at a hot `[256,256]` and a cold `[64,2]` shape,
> the same static code with the trip count changed. Five allocations × two
> tiers × 7 interleaved passes = 14 560 rows. The five builds are round-robined
> rather than run in blocks, so a machine that slows down mid-sweep slows every
> variant's later passes equally.
>
> **A/A floor**, pass-to-pass interquartile spread of the same build: median
> **1.42%** on AVX-512, **1.14%** on SSE2 (p95 11.5% / 15.2% — the tail is
> small kernels on a contended four-core host). A comparison counts as resolved
> when two allocations differ by more than the mean of their own spreads.
>
> **What the harness says the five allocations cost** — geomean of the
> per-kernel ratio to `main` (#1150, the incumbent):
>
> | tier | family | main | prespill | onepool | peakbudget (3′) | tripcount (3″) |
> |---|---|---:|---:|---:|---:|---:|
> | AVX-512 | all 208 | 1.000 | 1.061 | 2.235 | 1.058 | 1.058 |
> | AVX-512 | `font_rendering` glyphs | 1.000 | 1.148 | 3.603 | **1.096** | **1.106** |
> | AVX-512 | atlas glyphs 16px | 1.000 | 1.066 | 2.319 | 1.069 | 1.070 |
> | AVX-512 | invariants, hot shape | 1.000 | 0.998 | 1.302 | 0.977 | 0.972 |
> | SSE2 | all 208 | 1.000 | 1.070 | 1.424 | **0.957** | **0.961** |
> | SSE2 | `font_rendering` glyphs | 1.000 | 1.093 | 1.464 | 1.009 | 1.013 |
> | SSE2 | atlas glyphs 16px | 1.000 | 1.078 | 1.443 | 0.953 | 0.955 |
> | SSE2 | invariants, hot shape | 1.000 | 0.999 | 1.182 | 0.966 | 0.990 |
>
> **The contradiction reproduces, on a different measurement.** The 3′/3″
> allocations are 5.8% slower than `main` on AVX-512 and 4% *faster* on SSE2,
> and on the `font_rendering` glyph shapes specifically they are +9.6% / +10.6%
> on AVX-512 against +0.9% / +1.3% on SSE2 — the same sign split the previous
> block reported at +13–18%, arrived at by timing the collapse call rather than
> `Lattice::bake`. The measurement was not the problem.
>
> **What they differ by**, summed over the corpus and trip-weighted:
>
> | tier | allocation | dyn mem ops | dyn **remats** | mem + remats | code bytes | measured |
> |---|---|---:|---:|---:|---:|---:|
> | AVX-512 | main | 1 247 492 | 335 478 | **1 582 970** | 2 441 516 | 1.000 |
> | AVX-512 | tripcount | 1 067 257 | 1 025 251 | 2 092 508 | 2 952 145 | 1.058 |
> | AVX-512 | peakbudget | 1 068 457 | 1 034 467 | 2 102 924 | 2 959 769 | 1.058 |
> | AVX-512 | prespill | 1 291 300 | 978 892 | 2 270 192 | 2 942 194 | 1.061 |
> | AVX-512 | onepool | 5 672 876 | 2 333 427 | 8 006 303 | 5 606 542 | 2.235 |
> | SSE2 | peakbudget | 3 745 706 | 2 808 348 | **6 554 054** | 3 843 141 | 0.957 |
> | SSE2 | tripcount | 3 745 706 | 2 808 348 | 6 554 054 | 3 843 141 | 0.961 |
> | SSE2 | main | 4 426 109 | 3 454 348 | 7 880 457 | 3 991 144 | 1.000 |
> | SSE2 | prespill | 4 380 039 | 5 312 449 | 9 692 488 | 4 281 020 | 1.070 |
> | SSE2 | onepool | 12 046 959 | 5 193 104 | 17 240 063 | 5 661 858 | 1.424 |
>
> Read the AVX-512 half down the `dyn mem ops` column and it orders the
> allocations **backwards**: it ranks 3″ (1.07M) as the cheapest and `main`
> (1.25M) fourth, and `main` is the fastest of the four. Read it down
> `mem + remats` and the order is exactly the measured one on **both** tiers.
> 3″ bought its 14% fewer memory operations on AVX-512 by **tripling
> rematerializations** — 335k → 1 025k — and a rematerialization is not a
> memory operation, so the quantity it optimized could not see what it was
> spending. That is the residual, and it is not subtle: it is 3× on the metric
> that was excluded.
>
> **Predictor scores.** Spearman ρ across kernels, pooled over allocations, and
> — the score that matters — how often the predictor gets the **sign** of the
> delta between two allocations of the same kernel right:
>
> | predictor | ρ AVX-512 | ρ SSE2 | sign AVX-512 | sign SSE2 |
> |---|---:|---:|---:|---:|
> | `dyn_mem_ops` (the rejected policies' objective) | 0.825 | 0.796 | 72.5% (n=1230) | 77.6% (n=1597) |
> | `dyn_mem_ops` × vector width | 0.825 | 0.796 | 72.5% | 77.6% |
> | `dyn_sched_ops` (trip-weighted DAG ops) | 0.960 | 0.979 | **no opinion** | **no opinion** |
> | **`dyn_traffic` = mem ops + remats** | 0.826 | 0.806 | **99.1%** | **97.9%** |
> | `dyn_emitted_ops` (+ sched ops) | 0.979 | 0.979 | 99.1% | 97.9% |
> | `dyn_bytes` (trip-weighted code bytes) | 0.978 | 0.973 | 97.9% | 97.9% |
> | `dyn_bytes` × L1i overflow | 0.979 | 0.973 | 97.9% | 97.9% |
> | `static_mem_ops` (no trip weighting) | 0.752 | 0.677 | 72.4% | 81.5% |
> | `static_bytes` | 0.764 | 0.712 | 97.6% | 97.7% |
>
> **`dyn_traffic` — loads + stores + rematerializations, each weighted by its
> scope's trip count — is the predictor that tracks both tiers**, and it is one
> term away from the quantity three policies were built on. Adding
> rematerialization to the metric is the whole difference: `dyn_emitted_ops`
> scores identically on the sign test because the term it adds (scheduled ops)
> is a constant per kernel, and `dyn_ops_plus_3mem` scores identically to
> `dyn_mem_ops` for the same reason.
>
> **Where the aggregate hides the finding, and the per-pair table that shows
> it.** One allocation being 2.2× slower everywhere is predicted correctly by
> anything monotone in code size, so an aggregate over all ten pairs of
> allocations is carried by the easy comparison. Broken out per pair, on
> AVX-512:
>
> | pair | n | `dyn_mem_ops` | `dyn_traffic` | `dyn_bytes` |
> |---|---:|---:|---:|---:|
> | main → onepool | 183 | 100% | 100% | 100% |
> | main → peakbudget | 135 | **28%** | **100%** | 96% |
> | main → tripcount | 134 | **27%** | **100%** | 96% |
> | main → prespill | 100 | 100% | 97% | 97% |
> | peakbudget → prespill | 119 | **32%** | 97% | 95% |
> | prespill → tripcount | 123 | **33%** | 98% | 97% |
>
> On exactly the comparisons this month's allocator work turned on,
> the memory-op metric is not uninformative — it is **anti-correlated**, right
> about a quarter of the time. A coin would have done better. That is what
> "three principled replacements each lost to a fitted constant" looks like
> when the objective is measured against the clock instead of against itself.
>
> **Two smaller results, both negative.**
>
> 1. **Pricing a spill by vector width buys almost nothing.** Within a tier the
>    width is a constant, so `dyn_mem_ops × vector_bytes` cannot change any
>    ranking; pooled across tiers it moves ρ from 0.785 to 0.808 and changes no
>    sign. "A 64-byte spill is not a 16-byte one" is true and is not the
>    explanation.
> 2. **An L1i budget term changes nothing.** `dyn_bytes × (1 + overflow of
>    32 KB)` scores identically to `dyn_bytes` on both tiers. The 42 KB kernels
>    are real, but code size is already priced by the trip weighting, and the
>    cliff adds no separation.
>
> **What is wrong with the brief this work was given.** It listed the candidate
> predictors "in order of simplicity: dynamic memory ops; dynamic memory ops ×
> vector bytes; trip-weighted instruction count; trip-weighted bytes; code
> bytes against an L1i budget" — and the winner is not on that list. Every
> candidate on it treats *memory* as the thing to count, because that is the
> vocabulary the allocator's own metric established; the term that decides is
> the one the allocator's metric was defined to exclude. The generalisable
> version: when a metric has been optimized against and lost, the missing term
> is likely to be something that metric's own definition ruled out of scope,
> so the candidate list should be built from *what the emitter emits* rather
> than from variations on the losing quantity. `emit::traffic` counts remats
> apart from loads and stores for exactly this reason — it does not assume
> which of them costs.
>
> Also wrong, though harmlessly: "trip-weighted instruction count" is not a
> predictor of allocation at all. The allocator does not change the DAG's
> scheduled-operation count — only the traffic around it — so that candidate
> predicts a delta of exactly zero on every comparison and has no opinion to be
> right or wrong about. It is in the table as `dyn_sched_ops` to record that.
>
> **The `bench` profile does not change the picture, and the harness can say
> why rather than assert it.** `main` was built at `--profile bench` (LTO,
> codegen-units=1) on both tiers and re-measured against its `release` build:
> **every one of the 208 kernels emits byte-identical code on both tiers** —
> every static feature agrees, row for row — and the clock differs by 0.25%
> (AVX-512) and 0.41% (SSE2) corpus-wide, inside the A/A floor. That is what
> should be expected: the profile changes how the *emitter* is compiled, not
> what it emits, and the timed region is the emitted kernel. Every row records
> its profile, so this stays checkable rather than assumed.
>
> **What did not get measured.** Only the two x86 tiers this host runs
> natively; aarch64/NEON is where a rematerialization *is* a memory operation
> (the constant pool), so the `dyn_traffic` term that carries this result is
> exactly the term most likely to behave differently there, and re-measuring it
> is the first thing to do. And the synthetic `wide` and `anchored` families
> turned out not to spill at all once `optimize_runtime_arena` has rescheduled
> them — they contribute low-pressure points, the glyph corpus does the
> pressure work, and that is worth fixing before those two families are leaned
> on for anything.
>
>
> **Added after the rebase onto #1158 — a sixth allocation, and the first
> out-of-sample check.** The sweep above measured the five allocations that
> existed when it ran. #1158 (`reload` leaves the register file; every tier's
> pool grows by two) landed on top of it, so the same harness was run again on
> just that pair, 4 interleaved passes per tier:
>
> | tier | allocation | dyn mem ops | dyn remats | mem + remats | code bytes | measured |
> |---|---|---:|---:|---:|---:|---:|
> | AVX-512 | #1150 | 1 247 492 | 335 478 | 1 582 970 | 2 441 516 | 1.000 |
> | AVX-512 | **#1158** | 1 203 850 | 225 159 | **1 429 009** | 2 350 963 | **0.990** |
> | SSE2 | #1150 | 4 426 109 | 3 454 348 | 7 880 457 | 3 991 144 | 1.000 |
> | SSE2 | **#1158** | 3 956 377 | 2 749 489 | **6 705 866** | 3 736 784 | **0.963** |
>
> `mem + remats` falls on both tiers and so does the clock: −1.0% on AVX-512,
> −3.7% on SSE2. Two registers back is worth more on the narrow tier, which is
> the shape #1150's own table would predict. The #1150 static totals reproduce
> the sweep's `main` row **exactly**, digit for digit, which is the check that
> the counting change the rebase required did not move any number reported
> above.
>
> **What this pair does *not* settle.** It is not a discriminating comparison
> between the two predictors, because both metrics move the same way: memory
> ops and rematerializations both fall, on both tiers, so `dyn_mem_ops` and
> `dyn_traffic` mostly agree. Where they disagree the deltas are tiny — the
> median per-kernel move is **1.92%** on AVX-512 against a 1.31% A/A spread,
> and only 69 of 208 kernels move by more than 3% — so the AVX-512 sign scores
> (`dyn_mem_ops` 91%, `dyn_traffic` 70%, n = 43 and 93) are being read off
> comparisons at the measurement floor and should not be ranked. On SSE2, where
> the pair does resolve (median move 4.96%, 136 of 208 kernels past 3%),
> `dyn_traffic` leads at 85% against 81%.
>
> The discriminating comparisons remain the ones in the tables above, where one
> metric falls while the other rises — and the conclusion those reached is
> unchanged.
>
> **What this licenses.** The `pool − MIN_SCRATCH` constant still stands: none
> of the four alternatives measured against #1150 is faster than it on both
> tiers, and #1158 keeps the constant while giving the pool two more
> registers. What has changed is that there is now a static quantity whose
> sign agrees with the clock 98–99% of the time on both tiers, measured before
> any policy was built on it — which is the precondition the carry-budget block
> set. The next policy is allowed to be built now, and the thing it should
> minimise is `Σ scopes (loads + stores + remats) × trips`, with the harness
> re-run to check rather than the metric trusted.
