# An integral is a fold

## Metadata
- **Author**: JP (direction), Claude (draft)
- **Status**: `Draft`. The denotation was proposed 2026-09-23 and nothing is
  built. §2's node shape (a `Cell` domain on `Fold`) is Claude's refinement of
  JP's choice of a typed node, and is open for JP to confirm (§9).
- **Created**: 2026-09-23
- **Verified against**: `d2a42c9`. Every `file:line` below was re-read there,
  by two independent maps and three fact-checks.
- **Amends**: [a-glyph-is-a-formula](2026-09-23-a-glyph-is-a-formula.md)
  §4.1, §4.3 and §6 (the corrections are listed in §6 below).
- **Continues**:
  - [demand-is-a-dag-property](2026-09-07-demand-is-a-dag-property.md) §1–§2
  - [one-conditional-three-lowerings](2026-09-08-one-conditional-three-lowerings.md) §6
  - [collapse-is-a-fold](2026-09-16-collapse-is-a-fold.md)

**Decisions it records (JP, 2026-09-23):**

> *"What's the difference between demand and variance? Integration should
> need whatever hoisting needed. It's the same thing. A sum/integral is a
> loop in math notation."*
>
> *"Demand is for the register allocator and variance is for the
> factorization? These are somehow related."*
>
> Offered either an opcode per axis or a typed node, JP chose the typed node,
> `Area{axis, body}`.

---

## 1. Two facts about a value

Let `B` be a kernel's binders: the lattice axes `X` and `Y`, plus every fold's
index. Let `Ω` be the iteration space, with one coordinate per binder. A value
`v` denotes `⟦v⟧ : Ω → V`.

**Variance** is `var(v) ⊆ B`, the loops `v` depends on.

- **Law:** if `b ∉ var(v)`, then `⟦v⟧` is constant along `b`.
- **Direction:** forward, from the leaves.
  - A leaf contributes its own binder.
  - An operation takes the union of its children's variance.
  - A fold removes the index it binds (`variance.rs:383-385`).
- **Meaning:** it is the Boolean shadow of the forward derivative. `b ∉ var(v)`
  says `∂v/∂b ≡ 0`, and it says so for any change in `b`, not only an
  infinitesimal one.

**Demand** is `dem(v) ⊆ Ω`, the points at which `v` is read.

- **Law:** if `ω ∉ dem(v)`, then replacing `⟦v⟧(ω)` with anything leaves the
  output at `ω` unchanged.
- **Direction:** backward, from the root.
  - `dem(root) = Ω`.
  - `Select(m, a, b)` passes `dem ∧ m` to `a` and `dem ∧ ¬m` to `b`.
  - Every other edge passes demand through unchanged.
  - A value with several consumers joins their demands by `∨`
    (`passes/demand.rs:1-15`).
- **Meaning:** it is the Boolean shadow of the adjoint, `∂out/∂v`. The select's
  mask is the adjoint of the select with respect to its arm.

The two are duals. Variance is read off a value's **producers**, and demand is
read off its **consumers**. Together they say what `v` costs. The evaluations
`v` actually needs are

```text
E(v) = π_var(v)( dem(v) )
```

That is the demanded points projected onto `v`'s own axes. **Hoisting** shrinks
the work by the projection: `v` is never recomputed along an axis it does not
vary on. **Guarding** shrinks it by the restriction: `v` is never computed
where nothing reads it.

### What reads which

| consumer | reads | today |
|---|---|---|
| factoring a rule's side condition: `⊕_i(c ⊗ f) = c ⊗ ⊕_i f` iff `i ∉ var(c)` | var, of the **class** | no class fact exists. `rebuild_body` walks one representative per rule firing (`egraph/fold_rules.rs:288-327`) |
| hoisting: evaluate in the innermost scope that binds some bit of var | var, of the **chosen** term | `place_roots` over `schedule_variance` (`emit/mod.rs:2904-2974`, `:2746-2819`) |
| pricing: `cost(op) · |E(v)|` | var of the chosen term; dem taken as `Ω` | `LatticeShape::evals` (`variance.rs:719-736`) in `cost_of_choices` (`extract.rs:1869-1873`) |
| skipping: a guard around a region whose literal is uniformly false over a batch | dem | per-select **exclusivity**, not demand (`guards.rs:520-723`) |
| liveness: the register allocator's fact | dem, projected onto the schedule instead of onto `Ω` | linear scan's own intervals; guard arms enter through `guarded_arms`/`guard_sites` (`regalloc.rs:2657,2685`) |

So JP's reading is right as a division of labour:

- **Variance is for the factorization.** It is also for hoisting, which is the
  same transformation. Hoisting moves the *evaluation* of `c` out of the loop.
  Factoring also moves the *operation* out, so one multiply replaces `len` of
  them.
- **Demand is for skipping and for liveness.** The allocator's live ranges are
  demand read along time instead of along the iteration space.

### Why they are not one pass

Variance depends only on what lies below `v`. Every member of an e-class
denotes the same function, so all members have the same variance. The class's
variance is therefore

```text
var(C) = ⋂_{n ∈ C} var(n)
```

computed as a greatest fixpoint from `ALL`, since a cycle must not certify
invariance. It is an e-graph fact, and it is the one the rewrite rules need.

Demand depends on what lies above `v`, and extraction has not yet chosen that.
The join over every parent e-node is a sound over-approximation, but it is
never exact before extraction. So demand is a fact about the extracted DAG.
The e-graph needs variance; codegen needs both.

## 2. An integral is a fold

`⟦Reduce { fold, body }⟧ = ⊕_{k ∈ D} ⟦body⟧[i := k]` (`fold.rs:153`). The
monoid `Σ` already lists integration among its uses (`fold.rs:46`). The pixel
integral is that fold over a different domain: length measure on a cell, where
`Σ` uses counting measure on a range.

```text
Fold { monoid, index }         index = Range { binder, lo, hi, stride }     today's Fold
                                      | Cell(axis)                          the pixel along one lattice axis

⟦Reduce { Fold { SUM, Cell(a) }, body }⟧(ℓ) = ∫_{ℓ_a − ½}^{ℓ_a + ½} ⟦body⟧(ℓ[a := s]) ds
```

This is JP's `Area{axis, body}` given `Reduce`'s own type rather than a sibling
node. As a sibling it would be a second implementation of "a fold over an
index" (CLAUDE.md, "trait first"). It would also force every rule of §3 to be
written twice or put behind a trait, when the only difference is the index's
domain.

`area[dX ∧ dY](k)` is two folds, `Cell(X)` around `Cell(Y)`. That is Fubini on
the square, the same way `Σ_{i,j}` is two sums. `Kernel::area()` is the sugar
for it. Nothing needs a 2-form node.

### The index is the lattice axis, bound after composition

A `Range` fold binds a fresh index (`Binder`, `fold.rs:96-112`). A `Cell` fold
binds the lattice axis itself, and it binds it **after** `Kernel::at` has
substituted into the body. That is what makes the measure the *screen* pixel,
as a-glyph-is-a-formula §4.1 requires:

- `(area k).at(σ)` is `area (k.at(σ))`. This is the area of the screen pixel
  under the warped shape.
- It is **not** `∫_{cell(σ(ℓ))} k`, the area of the warped pixel.

Binding at construction, with the body's `X` replaced by a fresh binder at
`Kernel::area()` time, would give the second meaning. `at` would then reach
only the cell's centre.

`Dwrt` already works this way (CLAUDE.md, "The macro tier does not resolve
`Dwrt`"). Its consequence should be stated rather than left in a comment:

- A kernel carrying a `Dwrt` or a `Cell` fold denotes `Warp → (Ω → V)`, not
  `Ω → V`.
- A rule that is **natural in the warp** holds before composition. Linearity,
  interchange and Fubini are examples.
- A rule that holds **only at the identity warp** is sound only after
  composition. Factoring by `X ∉ var(c)` and the basis integrals are examples:
  `X`-invariance is not stable under `at`.
- So `Cell` folds resolve only in the runtime tier. The macro tier declines
  them by vocabulary (`insert.rs:124-127`, `ops.rs:356-386`), not by a second
  `DwrtFree` walk.

### The cell is centred

The lattice samples at `X = x₀ + col + lane` (`passes/lattice.rs:116`). The
atlas samples texel centres with `.at(X+½, Y+½)` (`fonts/atlas.rs:181-187`).
The cell is therefore `[X − ½, X + ½)`, and three things follow:

- The midpoint of the cell is the point sample every kernel computes today.
- `LowerArea`, the fallback for a `Cell` fold no rule resolved, is the deletion
  of the fold. It is bit-exact against the status quo.
- a-glyph-is-a-formula §1 and §4.1 write `[x₀, x₀+1)`, but their own formulas
  are centred: `((Z+½)³ − (Z−½)³)/3`, and "midpoint = point sample". The
  corner cell is the typo.

### Variance of a `Cell` fold

- A `Range` fold removes its binder: `var(body) ∖ {b}`.
- A `Cell(a)` fold keeps `var(body)`. The cell moves with `a`, so the result
  varies along `a` exactly when the body does. When `a ∉ var(body)`, the
  integral equals the body, which is the constant rule of §3.

### Why not `Area { integrand, form: [ValueId; k] }`

This is a-glyph-is-a-formula §4.1's shape. It contradicts that section's own
semantics in three ways:

1. `Kernel::at` substitutes every `Var` leaf (`expr.rs:236-275`), so a form
   operand is pulled back. That is exactly what §4.1 says `at` must not do.
2. A `Var(0)` operand makes every `Area` vary along `X` (`variance.rs:367-371`).
3. `Var(i)` inside a rule template is a metavariable (`rewrite.rs:36-40`).

The `Const(f32)` axis that `Dwrt` uses avoids all three, but only by encoding a
type as a float. That is decoded `as u8` at three sites (`derivative.rs:56`,
`passes.rs:709-712`, `graph.rs:3019`), which is the smell `Fold` was created to
remove (`fold.rs:1-14`). `Dwrt` should get the same typed axis in its own CL.
That change touches the public `Kernel::dwrt(u8)`, so it needs JP's permission.

## 3. The rules are loop transformations

Every rule `area` needs is a rule `Σ` has always needed. Each is one statement
over `Reduce`, whatever its index's domain.

| rule | over a fold `⊕_{i∈D}` | `Σ` over a range | `∫` over `Cell(a)` | side condition |
|---|---|---|---|---|
| factoring | `⊕_i (c ⊗ f) = c ⊗ ⊕_i f` | `Σ(c·f) = c·Σf`; `min(c+f) = c + min f` | `∫ c·f = c·∫ f` | `i ∉ var(c)`, and `⊗` distributes over `⊕` |
| constant | `⊕_i c = c^{⊕|D|}` | `len·c` | `c`, since the cell has length 1 | `i ∉ var(c)` |
| linearity | `⊕_i (f ⊕ g) = ⊕f ⊕ ⊕g` | yes | yes | `⊕` commutative |
| interchange | `⊕_i ⊕_j f = ⊕_j ⊕_i f` | yes | `∫ Σ_p = Σ_p ∫`: the glyph's sum over pieces | neither domain depends on the other's index |
| select | `⊕_i select(m, u, w) = select(m, ⊕u, ⊕w)` | yes | yes: the glyph's band factor is a mask | `i ∉ var(m)` |
| narrowing | an indicator in `i` restricts `D` | only for constant bounds; a data bound is a dynamic trip count, which is refused and becomes a recompile | the clipped interval is two clamps, closed by an antiderivative, so no clipped-cell node exists | the bounds satisfy `i ∉ var` |
| basis | — | — | `∫ a = a`; `∫ a² = a² + 1/12`; `∫ clamp(k·a + c, 0, 1)` through `G` of §1 | `k, c` satisfy `a ∉ var` |

The half-plane, conic and Taylor rules of a-glyph-is-a-formula §4.1 carry over
unchanged, now over a `Cell` fold.

**The one analysis they all need** is `i ∉ var(C)` for an e-class `C`. It is a
per-class fact, `var(C) = ⋂ var(n)`:

- It is set on `add` from the node's transfer function.
- It is intersected on `union`.
- It is maintained the way `const_fact` already is (`graph.rs:100-108, :688,
  :741-777`).

A stale parent fact over-approximates. Without upward repair, a rule can miss
an opportunity but can never fire wrongly. This one fact replaces two things:

- `DepsAnalysis` (`egraph/deps.rs`, 532 lines, no callers). Its
  `Variance::meet` is a popcount minimum, not `⋂`: `X.meet(Y) = X`, where `⋂`
  gives `∅` (`variance.rs:151-169`).
- `rebuild_body`'s per-firing walk, which reads the first representative only.

Factoring is **not in the tree today**, for either domain. The runtime fold
rules are Peel, Halve and Empty (`egraph/rules.rs:189-191`).

The payoff differs by domain:

- **For `∫`, the payoff is reachability.** An unfactored `Cell` fold over a
  per-piece body has no closed form. It lowers to its midpoint, which is a
  point-sampled, aliased edge.
- **For `Σ`, the payoff waits on pricing.** The DAG objective, which decides,
  adds each class's own cost once, with no trip count (`extract.rs:1890`;
  `SharedPricer` at `:2784-2791`). So `Σ(c·f)` and `c·Σf` each cost one `Mul`
  and tie. The tie goes to the tree objective, which does multiply by
  `fold_body_multiple` (`extract.rs:1983-1986`, `:2211-2216`). See §9.

## 4. What demand is for, and where it stands

### Skipping

A region is the set of values whose demand implies a literal `ℓ`, at a scope
where `ℓ` is uniform over a batch. A jump over the region when `ℓ` is false is
the `Select` that "contains an if" (CLAUDE.md).

**The code does not compute this.** It computes per-select *exclusivity*: each
select's arm cone, closed over the consumers visible in one scope
(`closed_exclusive`, `guards.rs:520-723`). The two differ:

- A value read by the true arms of two selects on the same mask has demand
  `m`. It is exclusive to neither.
- The glyph has that shape: the winding and the distance fold are each masked
  by `inside` (`fonts/loop_blinn.rs:268-270, 808, 892`).
- The telemetry for the winding box reads `(exclusive, demand-exclusive, guarded) = (1, 3, 0)`
  (results [2026-09-23 §4](../results/2026-09-23-freetype-comparison.md)).

The DNF demand pass exists, but decides nothing:

- `demand_of_arena` is test-only.
- `demand_of` is reached only through `PIXELFLOW_GUARD_TELEMETRY`
  (`guards.rs:463-469`).

### Liveness

Liveness is demand along the schedule. It is the register allocator's own
backward fact, and demand does not replace it. What demand adds is which
iterations a guarded region's values are live on, and today the allocator gets
that from the guard arms.

### Pricing

The ideal count is `|E(v)|`. Statically, `dem = Ω`, so `|E(v)| = ∏_{b ∈ var(v)} |dom b|`.
The count the nest realizes is the product of trip counts over every scope on
`v`'s chain, and that can exceed the ideal:

- A glyph piece's band height `(t₁ − t₀)_p` has `var = {Y, p}`, so it ideally
  runs `h · n_p` times.
- Fold `p` sits inside the batch loop because its body reads `X`. So the
  factor actually runs `⌈w/L⌉ · h · n_p` times (`emit/mod.rs:2946`;
  collapse-is-a-fold §2.3).
- The gap is loop fission and interchange, which collapse-is-a-fold §4 defers.
- Until then, a price is honest only if it counts the nest.
- `evals` counts neither. It prices any binder bit as the innermost axis,
  `w · h` (`variance.rs:713-722`), and its note that "binders are distributed
  before the e-graph sees them" is stale.

### Placement

A value may sit at any scope between two bounds:

- **Outward:** no further than the outermost scope that binds every bit of its
  variance.
- **Inward:** no deeper than the common ancestor of its consumers.

Speculating a value out of an arm is always sound, because every op is total
(NaN, never a trap; gathers are clamped, `passes.rs:430-444`). So choosing the
scope is purely a cost question. Today `place_roots` always takes the outer
bound and reads no demand. A value hoisted out of an arm is therefore computed
whether or not the arm runs.

**None of §4 is on `area`'s path.** It is on the glyph's: the box-as-branch of
a-glyph-is-a-formula §4.3.

## 5. What the code says today

**Invariance is computed at seven sites:**

1. `compute_arena_variance` (`variance.rs:339`). In production it runs for
   `lower_dwrt`'s tabulation rule only (`passes.rs:788`). It also serves
   `unroll_reduce`, which is off the production path.
2. `node_variance` in the extractor (`extract.rs:1729-1759`).
3. `schedule_variance` in codegen (`emit/mod.rs:2746-2819`). It differs from #1
   only on `Guard`, which `collapse` refuses anyway (`passes/lattice.rs:156`).
4. `rebuild_body`'s `varies` flag (`fold_rules.rs:288-327`).
5. `Substitution` in `unroll_reduce` (`passes.rs:510-525`). Research-only.
6. `DepsAnalysis` (`egraph/deps.rs`). Dead.
7. `compute_dag_variance` (`variance.rs:432`). Dead.

**Exclusivity and the jump.** Jumps are decided by `closed_exclusive` together
with the `OUTSIDE` sentinel (`guards.rs:562-569`) and a price:

- The price, `arm_cycles`, is compared against `MISPREDICT_PENALTY_CYCLES = 16`
  (`guards.rs:399-415`).
- `arm_cycles` prices a fold at `cost(Reduce) · len`, and the table says
  `Reduce => 0` (`egraph/cost.rs:144`).
- The extractor's tree objective prices the same fold at `len · body`
  (`cost.rs:356-359`).
- One fact, two answers.

## 6. Corrections to a-glyph-is-a-formula

1. **`Dwrt` is not priced prohibitively.**
   - It costs 1000 per evaluation (`cost.rs:134`), and `node_op_cost` says why
     the sentinel was removed: extraction must be able to *keep* a `Dwrt` and
     hand it to `LowerDwrt` (`cost.rs:316-333`).
   - The "`Dwrt` survived extraction" assertion is inside a `#[cfg(test)]`
     module (`runtime.rs:1653, :1804`).
   - A `Cell` fold takes the same kind of price: finite, strictly above every
     rule's right-hand side, and pinned per rule. It is a legalization price
     and never an accuracy knob.
2. **"The latency prior prices a node the same wherever it is placed" is false
   of extraction.**
   - `evals` weights every node by its variance (`extract.rs:1869-1873`).
   - What is missing is a binder's trip count (§4).
   - `Extraction::chosen_variance` is a four-bucket histogram with one test
     caller (`extract.rs:147, :4301`), so it is not the pricing seam.
   - CLAUDE.md keeps it as a seam for the schedule cost model, and it stays.
   - Build step 4 becomes: price the nest, when fission gives the extractor a
     choice of nest.
3. **The node shape** is §2's `Cell` fold, not `Area { integrand, form }`.
4. **The cell is centred** (§2).
5. **"Its polynomial arm is demanded under `0 < u < 1`" (§4.1) and "per batch
   to the right of a piece, the constant arm" (§6) do not exist.**
   - `clamp` is `max` then `min` (`kernel.rs:651-653`); it has no arm and should
     have none, because a five-op body is below any branch's price.
   - Every perimeter-shaped saving in §6 rests on the box tree of §4.3–§4.4.
6. **§4.3's diagnosis, and results [2026-09-23 §4](../results/2026-09-23-freetype-comparison.md)'s, is wrong.**
   - The per-glyph box is refused by the mispredict gate, not by the root rule.
     Its arm is priced `0 · len = 0 ≤ 16` (`guards.rs:402`, `cost.rs:144`).
   - The rule "an arm may own a root when it owns every scope that reads it" is
     still needed, but it is not what blocks today.
   - Even with folds priced, exclusivity cannot own the winding, because two
     selects share its mask. Demand can (§4).

## 7. Subtract first

Each item's fact-check is recorded in the session's two maps. All are
workspace-internal.

| # | delete | evidence | when |
|---|---|---|---|
| S1 | `find_hoistable_arena_nodes` / `_out_of` and their NOTE (`variance.rs:497-638`), plus the test (`:1162-1197`) | no callers; the NOTE cites `plan_collapse_hoist`, which does not exist | now |
| S2 | `compute_dag_variance` and its re-export (`lib.rs:44`) | no callers | now |
| S3 | `PartialOrd, Ord` on `Demand` | its stated reason, the demand-sorted schedule, was refuted in `demand.rs:52-79`; nothing orders a `Demand` | now |
| S4 | the hand-overs keyed on `parked` (`emit/mod.rs:2108-2118, 2212-2226`), the `fold_map` park (`:3897`), and the `nested_body` binding (not its `claimed` write) | `place_roots` is the only writer of roots, and `stays_put` excludes `Reduce` and `Guard` | now |
| S5 | stale docs: `evals` (`variance.rs:705-716`); `variance.rs:4-5, :333-337, :640-646, :660`; `passes/lattice.rs:5-7`; `extract.rs:1872-1873, :1908-1916, :2194-2198`; `cost.rs:143, :342-343`; `jit_cache.rs:131-134`; `kind.rs:188-196`; `fold_rules.rs:21-25` | each names a mechanism that is gone or never ran | now |
| S6 | `DepsAnalysis`, `find_hoistable`, `var_deps`, the re-export (`egraph/mod.rs:76`), and `Variance::meet` | no callers | **with** the class fact of §3, which replaces them in place. Deleting first and re-adding in the next CL is churn. |
| S7 | the three `LowerDwrt` call sites outside `legalize` (`runtime.rs:156`, `:208`) | `emit::compile` always runs `legalize` (`emit/mod.rs:1132` → `passes.rs:99`), which lowers `Dwrt` first | **verify** before deleting; `extract_for`'s lowering may feed the cache key |

Not on this list:

- **The demand telemetry** stays until demand decides something. It is the only
  instrument that measures the exclusivity-versus-demand gap.
- **G1/G2 `Guard`** (unreachable: `collapse` panics on it, `passes/lattice.rs:156`)
  is decided in the demand-regions CL, with its replacement in view.

## 8. Build order

1. **Subtract** S1–S5.
2. **The class variance fact and the factoring rule**, in `RuleSet::runtime()`
   beside `fold_rules()`.
   - S6 goes in the same CL.
   - `rebuild_body` reads the fact instead of walking.
   - Gate: kernels built once through the `Kernel` API and once as a scalar
     `f64` Rust closure, compared texel by texel. No pixelflow evaluator
     judges pixelflow (CLAUDE.md, "a same-form check cannot see a
     shared-definition bug").
3. **The `Cell` index**.
   - `Fold` gains the domain; `Kernel::area()` is the only public addition.
   - `LowerArea` deletes a surviving `Cell` fold, in `legalize` before
     `lower_dwrt_owned`.
   - `collapse` panics on a reachable `Cell` fold, as it does on `Dwrt`.
   - `Cell` resolves only under `Vocabulary::Runtime`.
   - Gate: a twin of `derivative_under_warp.rs`, where the integrand is warped
     and the measure is not.
4. **Basis, select and narrowing rules**, which give `A_p` exact.
   - Gate: a quadrature oracle in scalar `f64`.
   - The tolerance is relative to the terms' magnitude, not `f32` rounding
     alone: at `X ≈ 1000`, `X² + 1/12` has already lost its twelfth.
5. **Half-plane, conic and Taylor** (a-glyph-is-a-formula §4.1, §7).
6. **The glyph** (a-glyph-is-a-formula §6).

**The demand track** runs in parallel and does not gate the build order above:

- **D0.** A test that a guarded arm holding a fold `W` cannot leave a sibling
  fold reading `W`'s accumulator stale. `select_arms` builds consumers from
  `operands`, which treats `Reduce` as a leaf (`regalloc.rs:3239-3262`), so a
  fold body's read of `W` is invisible to the arm analysis. This is inferred,
  and a reproduction is running.
- **D1.** Regions of equal demand replace `closed_exclusive` and `OUTSIDE`.
- **D2.** A fold is priced in `arm_cycles` with the extractor's fold price.

## 9. Open questions

- **`Cell` on `Fold`, or a sibling `Area` node** behind a shared binder trait.
  Recommended: `Cell` on `Fold`, since every rule of §3 is then written once.
  About 39 files match a `Reduce`-shaped node. Most read `fold.binder()`,
  `len()` or `range()` and must refuse a `Cell`; that is where it is resolved
  before codegen. *For JP.*
- **A binder's trip count in pricing.** After `PeelFold`, one body class sits
  under folds of different lengths (`extract.rs:2200-2210`), so the DAG
  objective cannot carry one count per class. The candidates:
  - the maximum length over the folds binding that slot, which is an upper
    bound;
  - pricing the nest only once fission exists.

  Until one lands, `Σ` factoring is cost-neutral in the deciding objective.
- **Approximate rules in an exact e-graph.** A Taylor union equates terms that
  differ by `O(h^{n+1})`. Fix `n` per compile, put it in the cache key, and have
  the rule's doc state that its union is equality up to that remainder.
- **Integrands in the mask domain.** `∫ Lt(…)` integrates an all-ones bit
  pattern. `is_bitwise_domain` includes `Select` (`kind.rs:920-938`), so the
  domain cannot be read off the root op. Refuse at construction what can be
  refused, and denote the indicator as `select(m, 1, 0)`.
