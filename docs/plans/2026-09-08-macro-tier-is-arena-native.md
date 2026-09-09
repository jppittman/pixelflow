# The macro tier is arena-native

*2026-09-08*

## The claim

`kernel!` should touch two representations: the surface AST the user wrote,
and `ExprArena`. It touches five.

```text
tokens ─parser─▶ AST ─sema─▶ AST ─optimize.rs─▶ e-graph ─▶ extracted DAG ─▶ AST′
                                                                             │
                                              ir_bridge ◀────────────────────┘
                                                  │
                                                  ▼
                                             ExprArena ─┬─ e-graph (again, for Dwrt only)
                                                        └─▶ tokens rebuilding the arena
```

`AST′` is the interesting one. It is not the AST the user wrote — it is an AST
the optimizer *invented*, complete with `let` bindings it synthesized to name
shared subexpressions and opaque placeholder identifiers (`__opaque7`) standing
in for anything the e-graph could not hold. A representation nothing parses and
nothing prints, existing only so that the next stage's input type is spelled
`Expr`. Then `ir_bridge` walks it back down to the arena the e-graph had
already built and thrown away.

The e-graph then runs a *second* time inside `ir_bridge`, on the arena, under
its own separate entry point, for `Dwrt` nodes only.

## Why this is not merely ugly

Three defects were found in this subsystem in one week, and all three have the
same shape: **one stage accepts what a later stage refuses.**

| Defect | Accepted by | Refused by |
|---|---|---|
| `round`/`log10`/`pow` (#1206) | `sema` | arena lowering |
| `fract`/`hypot`/`clamp` (#1206) | the e-graph's decomposition | `sema` |
| captured identifiers | `sema` | arena lowering |

No test of a single stage can see any of them, because the stage under test is
the one that is right. That is not a testing gap to be closed by writing more
tests — it is what having five representations *costs*, paid three times. Each
boundary is a place two stages can disagree, and `AST′` doubles the boundaries
for no representational gain.

The fourth is a latent one, pinned as an executable record in
`pixelflow-compiler/tests/derivative_under_warp.rs`, and it is the subject of
the next section.

## What the measurement said

The plan that preceded this one assumed the chain-rule bug was a *fix*: teach
`Kernel::at` to re-differentiate through a coordinate warp. Before writing that,
delete the expansion-time differentiation call and see what happens:

```text
a_warped_derivative_ignores_the_warp_in_kernel      got 24, expected the pinned-wrong 12
a_warped_derivative_ignores_the_warp_in_kernel_raw  got 24, expected the pinned-wrong 12
an_unwarped_derivative_is_correct_in_both_macros    pass
a_composed_derivative_does_follow_the_warp          pass
```

24 is the chain-rule truth. **The bug is not a fix, it is a deletion.**

The reason is a one-liner once seen. `Kernel::at` contramaps coordinates by
substituting into `Var` nodes. A surviving `Dwrt(X·X, 0)` has the warp
substituted *into its operand* — `Dwrt(4X², 0)`, differentiated later, giving
`8X` — which is the chain rule, obtained for free by doing nothing. Resolving
`Dwrt` at expansion time destroys exactly the node the warp needed to reach, and
substitutes into the already-resolved `2X` instead.

This also explains the coincidence the pinning test documents: the production
glyph kernels are correct only because a `&` mask makes the e-graph decline
their arena, so their `Dwrt` nodes survive expansion and are resolved at bake
time — *the correct behavior, reached by accident, via an unrelated operator.*

So `differentiate_in_optimizer` and its support (`extract_dwrt_free`,
`encode_params_as_vars`, `PARAM_VAR_BASE`, `contains_dwrt`,
`reachable_node_count`) are not a subsystem with a bug in it. Their existence is
the bug.

### The subtraction pays a second time

`PARAM_VAR_BASE = 16` encodes `Param(i)` as `Var(16 + i)` because the e-graph
declines `Param`. CLAUDE.md, on `Var(u8)`:

> `Var(u8)` means a coordinate axis or a reduce binder depending on magic
> ranges — it used to mean a manifold-param slot as well, and that third
> meaning went out with the macro parameter that needed it.

It did not go out. It was reintroduced here, at a higher base, to get params
past a vocabulary that refuses them — the exact "convention written in a
comment" the same document warns about, one careless fold away from a param
aliasing a reduce binder. Deleting expansion-time differentiation deletes the
third meaning of `Var` along with it, for real this time.

## What this buys

- **The chain rule works**, for every kernel rather than for the ones a mask
  happened to save.
- **One optimizer, not three.** `optimize.rs`'s AST round trip, `ir_bridge`'s
  `Dwrt` entry point, and `pixelflow-search::runtime`'s pipeline become one
  `Optimize` pipeline, in the `Optimize`/`Then`/`Identity` vocabulary that
  already exists and that `optimizer_laws.rs` (deleted) L5 already tests.
- **`kernel_raw!` becomes a value**, not a branch that declines to call a
  function — which is what `Identity`'s doc comment already claims it is for.
- **Two stage boundaries disappear**, and with them the class of defect that
  produced three bugs in a week.
- **~1400 lines of `optimize.rs` and ~200 of `ir_bridge.rs` delete.**

## What it costs, honestly

Derivatives expand at bake time instead of build time. That is a real cost and
it is bounded: `optimize_runtime_arena` is memoized on a canonical key of the
arena, the lattice shape and the optimizer fingerprint, so it is once per
distinct kernel shape per process, not once per frame — and `LowerDwrt` runs in
that pipeline regardless, so for every kernel whose `Dwrt` survives today
(which is all three production glyph kernels) the cost is already being paid.

The saving on the other side is a whole e-graph saturation per `Dwrt`-carrying
`kernel!` expansion, deleted from every build.

## `Param` is an opaque leaf, and the e-graph should say so

Steps 4 and 5 hinge on one thing, and it is worth stating separately because
the same workaround appears three times in this tree.

`insert` declines `Shape::Param`:

> A macro-parameter slot. Valid only before kernel compilation, so one here
> means the term reached the e-graph without being specialized.

True of the *runtime* tier, where an unsubstituted param at bake time is a bug.
False of the *macro* tier, where an unsubstituted param is the entire point of a
builder — `circle(cx, 0.0, 1.0)` exists so that `cx` is not known yet. So every
macro-side caller smuggles params past the gate, each in its own spelling:

| Where | Smuggled as |
|---|---|
| `ir_bridge::encode_params_as_vars` | `Var(16 + i)`, decoded after extraction |
| `optimize.rs` | an opaque *identifier*, held in `opaque_exprs` and restored |
| (and `Var`'s retired third meaning, reintroduced by the first) |  |

Three encodings of one idea, and the idea already has two worked examples
sitting next to it. `ENode::Buffer` and `ENode::Uniform` are opaque leaves,
hash-consed by identity, matched by no rule, never folded. `ENode::Uniform`'s
doc states the semantics exactly:

> No rule matches it as a `Const`, so it is never folded; its gain in the
> e-graph is CSE of the arithmetic that depends on it alone.

That is a `Param`, word for word. So the subtraction is to stop encoding and
add the leaf: `ENode::Param(u8)`, alongside `Uniform`, accepted under
`Vocabulary::Templates` and still declined under `Vocabulary::Runtime` — which
is `Vocabulary` doing the job it exists for, deciding what a graph under it may
hold. The runtime tier's guard survives as a *type-level* fact rather than a
comment, and all three encodings delete.

It touches `ENode`'s 23 match sites across 11 files, which is why it is its own
change rather than a paragraph of this one.

## Steps

Each step is independently green; CI is the gate.

**This change — the deletion.**

1. **`OpKind::variant_name()`**, generated by `op_table!`. Collapses
   `opkind_to_tokens`'s 40 arms and its `_ => panic!("Unsupported OpKind")`
   into three lines. This is the fourth independently-maintained copy of the op
   table, and the panic arm refuses ops the arena and codegen both handle.

2. **Delete expansion-time differentiation.** `differentiate_in_optimizer` and
   its support, the `PARAM_VAR_BASE` encoding, and the in-file test modules
   whose subject they are. Flip `derivative_under_warp.rs` from `WRONG_*` to
   `TRUTH_*` and delete its "this file pins a bug" notice.

3. **Split `ir_bridge.rs`**: AST→arena lowering and arena→tokens emission are
   two jobs sharing a file only because the file is named after neither.

**Follow-up — the arena-native pipeline.**

4. **`ENode::Param`**, per the section above.

5. **Make the macro tier's optimizer an `Optimize` value and delete
   `optimize.rs`.** `kernel!` is `Saturate` under `Vocabulary::Templates`;
   `kernel_raw!` is `Identity`. Both run on the arena, after lowering. Step 4
   is what makes this possible without a fourth param encoding.

## The gate

Step 2 changes when differentiation happens, which is exactly the kind of
change that should not rest on a reviewer noticing. What is being deleted is
one of *two* implementations of the same calculus, and `ir_bridge.rs`'s
`expansion_derivative_tests` existed to check them against each other —
differentially, at the arena level, through private functions, because that
was the only place both tiers were visible at once.

With one tier left there is nothing to cross-check, so those tests delete with
their subject. But the property they were really defending is not
"two implementations agree" — it is **"a derivative built by the macro denotes
the derivative"**, and that one has a public witness:
`derivative_under_warp.rs` already carries
`a_composed_derivative_does_follow_the_warp`, the same derivative built from
`Kernel` values.

So the pinning test becomes the regression test. Once the macro cases assert
`TRUTH_AT_3` they are asserting equality with the composed control, which is
the tier-equivalence claim stated over the public surface instead of over two
private functions. It is broadened to carry what the deleted tests carried: a
scalar param inside the differentiand, and a differentiand whose derivative is
not a constant.

---

## Amendment (2026-09-08): what execution changed about the plan

Three things the plan got wrong or did not know, recorded because two of them
are the interesting part of the change.

### The macro tier cannot saturate a `Dwrt`, and that costs 12%

The plan assumed steps 2 and 5 were independent: delete expansion-time
differentiation, then move saturation onto the arena. They are not. Saturation
*is* a differentiator — the chain rule is in the production rule set, and a
`Dwrt` node is priced so the extractor never keeps one — so moving the e-graph
onto an arena that still holds `Dwrt` resolves it, which is exactly the
miscompilation step 2 deleted. Measured, not reasoned: dropping the guard puts
`derivative_under_warp.rs` straight back to 12 where the chain rule says 24.

So the macro tier declines any term carrying a `Dwrt` and lets the runtime tier
optimize it at bake time, after `LowerDwrt`, in the order `runtime.rs` argues
for. That is correct and it is not free. For `'8'` at 17 px the bake-time arena
goes from 2964 nodes to 3318 — **12% larger** — because the glyph's fragments
now reach one saturation unfused where they used to reach two.

Correctness does not trade against 12%, so this ships. But 12% is not the
floor, and the floor is reachable: what these kernels want is saturation with
the *derivative rules withheld*, which fuses without resolving. The search
crate does not expose a rule-set knob today, so it is denoted here rather than
built.

### Two pinned defects moved, and neither was fixed

`kernel_glyph_optimize.rs` pinned 29 texels where the optimized and raw glyph
arenas disagree on `'8'`'s waist. It is now 0, and the honest reading is not
"the tangency defect is fixed" — it is still a knife edge, and
`quad_tangency_winding` and `freetype_oracle` still hold it under test. What
changed is that there is only one optimizer. Two independent sets of fusion
decisions used to compound, and the *compounding* is what straddled the edge.
The test goes back to asserting emptiness, which is the guard it was written to
be.

`quad_tangency_winding.rs` (deleted) pinned the grazing residual at 0.745; it now
measures 0.688. Also not a fix — the defect is the same size in the only unit
that matters (most of a crossing, where zero is correct). This one is worth
recording for a second reason: the file's doc claimed it was "deliberately free
of the e-graph", and it never was. `AnalyticalQuad::kernel()` is built by
`kernel!`, so the macro tier had already chosen an association before the test
ran. A number that moves when the optimizer is perturbed is measuring the
extraction, which is precisely the trap that file's own "approach 1" records.
The claim is corrected there.

### `OpKind` was cheaper to extend than expected

Adding a variant broke exactly four matches, all inside `kind.rs` — `op_table!`
working as designed. One of the four was a sweep that skipped "leaves" by
listing six names; the seventh leaf did not land in the list and arrived as a
nonsense `push_ternary(Param, ..)`. It asks `arity() == 0` now.

### Sharing a pass means sharing what the pass assumed

`Saturate` hard-coded `Tier::Runtime` into its telemetry record, which was
true while it had one caller. The macro tier adopting it inherited that label
silently — and the label is not bookkeeping: the two tiers write to *different
sinks*, because a macro-tier saturation runs inside rustc and anything it puts
on stderr is parsed by cargo's `--message-format=json` and forwarded as a
compiler message. The record is valid JSON but not diagnostic-shaped, so a bare
line corrupts that stream for every consumer. `telemetry.rs` carries a
paragraph saying exactly this, with "confirmed empirically" attached; the
prefix that prevents it was chosen for that reason and this change would have
removed it by accident.

So `Tier` moves out of the feature-gated `telemetry` module and becomes an
ordinary type the crate always compiles, and `Saturate` carries one —
`Saturate::runtime(shape)` and `Saturate::macro_tier()`, the two tier
configurations finally sitting next to each other. The tier is a property of
the pass, not an argument its one call site happened to remember.

The finding underneath is about the gate, not the change: **nothing in CI
exercises this feature beyond compiling it**, so no job could have caught
this, and a human reading the diff is not a gate. `Tier::stderr_prefix` now
owns the prefix and `tier.rs` asserts the invariant it exists for — a
macro-tier record must not lead with `{`, a runtime one must. The test is
unconditional (the feature gates the *sink*, not the tier), and it was
mutation-checked: restoring the empty prefix fails it.

That also folded the sink, which had a two-arm `match` to decide between a
prefixed `write!` and a bare `write_all`. With the prefix empty for the
runtime tier, one write serves both.

### Correction (2026-09-08, later): one of them *was* fixed

The section above says two pinned defects moved and neither was fixed. That
was right about `kernel_glyph_optimize.rs` and `quad_tangency_winding.rs` (deleted),
which compare this code to itself, and wrong as a conclusion — because it was
drawn before running the one test that compares it to something else.

`freetype_oracle.rs` pinned **4 texels where pixelflow lays down ink and
FreeType finds none** — a spurious half-covered smear outside `'8'`'s waist,
described in that file as "a REAL, live rendering defect on `main`". It is
now **0**, and the file asserts zero.

That is a real fix, and it is the same mechanism the earlier section described
without recognising what it implied. Two optimizers meant two independent sets
of fusion decisions compounding, and the compound landed on the wrong side of
`disc >= 0` at the tangency. Deleting the AST tier removes the compounding, so
no glyph in the corpus lands on the edge any more.

Five deliberate attempts to fix this failed (they are catalogued in
`quad_tangency_winding.rs`, and the fifth is refuted most decisively: the ramp
creates the imbalance it exists to remove). None of them was this. It fell out
of removing a representation.

The scope is worth stating precisely, because "fixed" can mean too much here:

- **Fixed:** no glyph in the oracle corpus — 9 glyphs × 11 sizes, 7 px to
  48 px — puts ink where FreeType finds none.
- **Not fixed:** the knife edge. `disc >= 0` is still exact zero at a shared
  extremum, and `quad_tangency_winding.rs` (deleted) still measures a grazing residual
  of 0.688 where zero is correct. A future fusion choice could put a glyph
  back on it; the now-zero assertion in `freetype_oracle.rs` is what would
  catch that.

The methodological point is the one that file already made: *every check that
compares this code to itself agreed with the bug*. The self-comparisons going
quiet was not evidence of a fix, and I was right to refuse to read it as one.
The external instrument is what settled it, and it should have been run before
the conclusion was written rather than after.
