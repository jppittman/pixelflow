# An integral is a fold

## Metadata
- **Author**: JP (direction), Claude (draft)
- **Status**: `Draft`. The denotation was proposed 2026-09-23 and nothing is
  built. §2's node shape was decided by JP the same day: an integral is a
  fold whose domain is continuous, with no cell and no axis.
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
> *"I want integrals built on fold."*
>
> *"We shouldn't really have cell(axis)? … I think explicit mention of axes is
> like idk, unnecessary specificity."* And: *"I think I did the X Y Z W thing
> way before the jit and now, they're little more than conveniences."*
>
> On §2 as it now reads: *"yes, drop cell."*
>
> *"Do the integration like the derivatives. Put the rules in the egraph.
> Then, ensure none hit the emitter in the legalize passes."*

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
monoid `Σ` already lists integration among its uses (`fold.rs:46`). The
integral is that fold with a different measure:

```text
enum Fold = Range(RangeFold { monoid, binder, lo, hi, stride })    integers, counting measure    runs as a loop
          | Interval(IntervalFold { binder, lo, hi })              reals, length measure         closed or legalized

⟦Reduce { Fold::Interval({ u, [lo, hi) }), body }⟧ = ∫_lo^hi ⟦body⟧[u := s] ds
```

An enum rather than one struct with a domain field, because that is where
the difference can be refused instead of checked: a range accessor
(`len`, `stride`, `peel`) on an interval is a compile error, since a caller
must match `Fold::Range` to reach one, and a non-`Σ` integral is
unrepresentable, since `IntervalFold` has no monoid to hold one.

The binder is the same kind of thing in both: a fresh index the body reads
(`Binder`, `fold.rs:96-112`). The two domains differ in one respect only, the
measure, and that has one consequence. A range can run as a loop, and an
interval cannot. A continuous fold is either closed by a rule (§3) or
approximated by quadrature, meaning a discrete fold standing in for it.

The interval's bounds are `f32`. They are data-plane values: quadrature
substitutes them into the body as constants.

### The pixel is two intervals

```text
Kernel::area(k) = ∫_{u_y ∈ [−½, ½)} ∫_{u_x ∈ [−½, ½)} k.at(X + u_x, Y + u_y)
```

`X` and `Y` appear in this constructor and nowhere else. The IR sees two
ordinary folds, each binding a fresh index. Which coordinate an interval
perturbs is a fact about its body, not a field of the fold.

The lattice's coordinates are conveniences for naming the lattice's binders:
`collapse` rebinds them to `x₀ + col + lane` and `y₀ + row`
(`passes/lattice.rs:108-118`). The integral does not need them named.

Fubini needs no node of its own. The pixel is *built* as two folds, and
swapping them is interchange (§3), the same rule that swaps two sums.

### Bound at construction; `at` is precomposition

Like every other fold, the integral binds when it is built. `Kernel::at` then
substitutes the coordinates of the result. The order of composition says which
integral the author means:

- `area(k.at(σ))` is the screen pixel under the warped shape. This is what the
  glyph wants: `area` is the last thing before `collapse`.
- `area(k).at(σ)` is the warped pixel, the unit cell of `k`'s own space around
  `σ(ℓ)`.

Both are sayable, and neither is a special case. They agree for a translation,
such as the atlas's `.at(X+½, Y+½)` (`fonts/atlas.rs:181-187`). They differ
once `σ` scales or bends.

This replaces a-glyph-is-a-formula §4.1's rule that "`at` never substitutes the
measure". That rule would have made the integral late-bound like `Dwrt`, with
two costs:

- A kernel carrying an integral would denote `Warp → (Ω → V)`.
- Every rule whose side condition is an invariance would be sound only after
  composition, in the runtime tier.

Bound at construction, every side condition in §3 is about the bound index
`u`, and `at` never touches a bound index. `σ` is built from `X` and `Y`, so
`u ∉ var(c)` survives any warp. The integral's rules hold in both tiers.

### The pixel is centred

The lattice samples at `X = x₀ + col + lane` (`passes/lattice.rs:116`). The
interval is `[−½, ½)` about that sample, and three things follow:

- The midpoint of the pixel is the point sample every kernel computes today.
- The one-point quadrature, `u := 0`, is the fallback for an integral no rule
  closed. It is bit-exact against the status quo.
- a-glyph-is-a-formula §1 and §4.1 write `[x₀, x₀+1)`, but their own formulas
  are centred: `((Z+½)³ − (Z−½)³)/3`, and "midpoint = point sample". The
  corner cell is the typo.

More quadrature points, midpoint or Gauss–Legendre, would be a `Range` fold
with weights. That is an accuracy budget, chosen in one place.

### Variance

An interval fold removes its binder exactly as a range does: `var(body) ∖ {u}`.
The pixel integral of `k.at(X + u_x, Y + u_y)` varies along `X` exactly when
`k` does, because the body reads `X + u_x`.

### Legalization is quadrature

Codegen executes loops, and a continuous fold is not a loop.

- `legalize` replaces a surviving interval fold by its quadrature before
  `collapse`. Today that is the one-point rule: substitute the interval's
  midpoint for the binder, and multiply by the interval's length, which is 1
  for the pixel.
- `collapse` panics on a reachable interval fold, as it does on a `Dwrt`.
- Extraction prices a surviving interval fold above every rule's right-hand
  side, so a closed form wins whenever one was derived. That price is a
  legalization price and never an accuracy knob.

### Why not `Area { integrand, form: [ValueId; k] }`

That is a-glyph-is-a-formula §4.1's shape. Its form operands are the binder
spelled as a value, and a value is the wrong thing to spell it as:

- `Kernel::at` substitutes every coordinate leaf (`expr.rs:236-275`), so the
  form is pulled back.
- A `Var(0)` operand makes every integral vary along `X`
  (`variance.rs:367-371`).
- `Var(i)` inside a rule template is a metavariable (`rewrite.rs:36-40`).

A binder has none of these problems: it is what the form was trying to be.

`Dwrt` spells its axis as a `Const(f32)`, decoded `as u8` at three sites
(`derivative.rs:56`, `passes.rs:709-712`, `graph.rs:3019`). That is the smell
`Fold` was created to remove (`fold.rs:1-14`). Whether `Dwrt` should take a
binder the same way is its own question. It touches the public
`Kernel::dwrt(u8)`, so it needs JP's permission.

## 3. The rules are loop transformations

**Integration works the way differentiation does.**
- The author writes the integral, `area(k)`, just as they write `k.dx()`.
- Rewrite rules in the e-graph do the calculus, just as `ChainRule` does for
  `Dwrt` (`pixelflow-search/src/egraph/derivative.rs`).
  - The fold rules live in `fold_rules.rs`.
  - The integral-specific rules live in the `integral` module beside
    `derivative.rs` (`pixelflow-search/src/egraph/integral.rs`). The closed
    forms they write read an interval's ends, so they live beside the
    interval, in `pixelflow-ir/src/integral.rs`, as its quadrature does.
- Whatever the rules leave unclosed is lowered in `legalize`, before
  `collapse`, by `passes::resolve`: quadrature first, then `lower_dwrt`.
- No integral reaches the emitter. A panic in `collapse` and in
  `arena_to_schedule` enforces that, and so does a test that every glyph's
  extracted arena holds no interval fold.

Every rule `area` needs is a rule `Σ` has always needed. Each is one statement
over `Reduce`, whatever its domain.

| rule | over a fold `⊕_{i∈D}` | `Σ` over a range | `∫` over an interval | side condition |
|---|---|---|---|---|
| factoring | `⊕_i (c ⊗ f) = c ⊗ ⊕_i f` | `Σ(c·f) = c·Σf`; `min(c+f) = c + min f` | `∫ c·f = c·∫ f`: the glyph's `Y`-only band factor leaves the inner `u_x` integral | `i ∉ var(c)`, and `⊗` distributes over `⊕` |
| constant | `⊕_i c = c^{⊕|D|}` | `len·c` | `(hi − lo)·c`, which is `c` for the pixel | `i ∉ var(c)` |
| linearity | `⊕_i (f ⊕ g) = ⊕f ⊕ ⊕g` | yes | yes | `⊕` commutative |
| interchange | `⊕_i ⊕_j f = ⊕_j ⊕_i f` | yes | `∫ Σ_p = Σ_p ∫`, the glyph's sum over pieces; and `∫_{u_y} ∫_{u_x} = ∫_{u_x} ∫_{u_y}` | neither domain depends on the other's index |
| select | `⊕_i select(m, a, b) = select(m, ⊕a, ⊕b)` | yes | yes: the glyph's band factor is a mask | `i ∉ var(m)` |
| narrowing | an indicator in `i` restricts `D` | only for constant bounds; a data bound is a dynamic trip count, which is refused and becomes a recompile | an interval has no trip count, so its bounds may be values. The clipped length is two clamps, closed by an antiderivative, so no clipped node survives | the indicator's bounds satisfy `i ∉ var` |
| moments | — | — | `∫_lo^hi u^n du = (hi^{n+1} − lo^{n+1})/(n+1)`; over the pixel, `∫1 = 1`, `∫u = 0`, `∫u² = 1/12`. Also `∫ clamp(k·u + c, 0, 1)` through `G` of a-glyph-is-a-formula §1 | `k, c` satisfy `u ∉ var` |

No rule mentions a coordinate. `∫_u (X + u)² = X² + 1/12` falls out of
expansion, linearity, factoring and the moments. The half-plane, conic and
Taylor rules of a-glyph-is-a-formula §4.1 carry over as rules over interval
folds.

**The one analysis they all need** is `i ∉ var(C)` for an e-class `C`. It is a
per-class fact, `var(C) = ⋂ var(n)`:

- It is set on `add` from the node's transfer function.
- It is intersected on `union`.
- It is maintained the way `const_fact` already is (`graph.rs:100-108, :688,
  :741-777`).

A stale parent fact over-approximates. Without upward repair, a rule can miss
an opportunity but can never fire wrongly. This one fact replaces two things:

- `DepsAnalysis` (`egraph/deps.rs`, 532 lines, no callers; deleted in step 2). Its
  `Variance::meet` is a popcount minimum, not `⋂`: `X.meet(Y) = X`, where `⋂`
  gives `∅` (`variance.rs:151-169`).
- `rebuild_body`'s per-firing walk, which reads the first representative only.

Factoring is **not in the tree today**, for either domain. The runtime fold
rules are Peel, Halve and Empty (`egraph/rules.rs:189-191`).

The payoff differs by domain:

- **For `∫`, the payoff is reachability.** An unfactored interval fold over a
  per-piece body has no closed form. It lowers to its one-point quadrature,
  which is a point-sampled, aliased edge.
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
6. `DepsAnalysis` (`egraph/deps.rs`). Dead, and deleted in step 2.
7. `compute_dag_variance` (`variance.rs:432`). Dead.

**Exclusivity and the jump.** Jumps are decided by `closed_exclusive` together
with the `OUTSIDE` sentinel (`guards.rs:562-569`) and a price:

- The price, `arm_cycles`, is compared against `MISPREDICT_PENALTY_CYCLES = 16`
  (`guards.rs:399-415`).
- `arm_cycles` prices a fold at `cost(Reduce) · len`, and the table says
  `Reduce => 0` (`egraph/cost.rs:144`).
- The extractor's tree objective prices the same fold at `len · body`
  (`cost.rs:356-359`).
- One fact, two answers. **Resolved by D2:** there is now one formula,
  `CostModel::fold_cost` (`len · body + (len − 1) · combine`). The extractor's
  node price is `fold_cost(fold, 0)`, and the guard analysis prices an arm's
  loop with the same function.

## 6. Corrections to a-glyph-is-a-formula

1. **`Dwrt` is not priced prohibitively.**
   - It costs 1000 per evaluation (`cost.rs:134`), and `node_op_cost` says why
     the sentinel was removed: extraction must be able to *keep* a `Dwrt` and
     hand it to `LowerDwrt` (`cost.rs:316-333`).
   - The "`Dwrt` survived extraction" assertion is inside a `#[cfg(test)]`
     module (`runtime.rs:1653, :1804`).
   - A surviving interval fold takes the same kind of price: finite, strictly
     above every rule's right-hand side, and pinned per rule. It is a
     legalization price and never an accuracy knob.
2. **"The latency prior prices a node the same wherever it is placed" is false
   of extraction.**
   - `evals` weights every node by its variance (`extract.rs:1869-1873`).
   - What is missing is a binder's trip count (§4).
   - `Extraction::chosen_variance` is a four-bucket histogram with one test
     caller (`extract.rs:147, :4301`), so it is not the pricing seam.
   - CLAUDE.md keeps it as a seam for the schedule cost model, and it stays.
   - Build step 4 becomes: price the nest, when fission gives the extractor a
     choice of nest.
3. **The node shape** is §2's interval fold, not `Area { integrand, form }`.
   The form is the binder.
4. **The pixel is centred** (§2).
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

7. **"`at` never substitutes the measure" is withdrawn** (§2).
   - The integral binds at construction, and `at` is plain precomposition.
   - `area(k.at(σ))` is the screen pixel under the warped shape;
     `area(k).at(σ)` is the warped pixel.
   - The integral's rules then hold in both tiers.

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
3. **The interval domain.**
   - `Fold` gains `Interval`, and `Kernel::area()` is the only public addition.
   - `legalize` replaces a surviving interval fold by its one-point quadrature
     before `collapse`, and `collapse` panics on a reachable one.
   - `PeelFold`, `HalveFold` and `EmptyFold` decompose ranges, so they decline
     an interval. `FactorFold` applies to both domains.
   - Gate:
     - An unclosed `area(k)` collapses to `k` at the pixel centre, checked
       against an `f64` closure.
     - `area(k.at(σ))` and `area(k).at(σ)` stay distinct under a scaling `σ`.
       Step 3 alone pins that they are distinct terms. Step 4's moments pin
       that they integrate different pixels.
4. **Basis, select and narrowing rules**, which give `A_p` exact.
   - Gate: a quadrature oracle in scalar `f64`.
   - The tolerance is relative to the terms' magnitude, not `f32` rounding
     alone: at `X ≈ 1000`, `X² + 1/12` has already lost its twelfth.
   - **Built** as the chord needs it and no further: `FactorFold` made n-ary,
     `NarrowInterval` and `ClampMoment`, and a closing phase that runs that
     family plus `ConstantFold` to a fixpoint before the full rule set, only
     on a graph that holds an integral. The constant, linearity, select,
     interchange and power-moment rules close nothing the chord needs and
     wait for a kernel that does. Every fold and integration rule answers
     with one `RewriteAction::Plan`, replacing the three fold-specific
     actions. `PeelFold` and `HalveFold` decline to copy an integral no rule
     closed. The oracle is `pixelflow-core/tests/area_oracle.rs`: polygon
     clipping in `f64` against the compiled closed form, and a count of
     the integrals extraction left unclosed.
   - **Reviewed** against a second, unrelated `f64` reference
     (`pixelflow-core/tests/area_adversarial.rs`: the chord's row coverage
     integrated piecewise-exactly; literal slopes; coefficients `3`, `−0.1`,
     `−3`; one-sided and redundant cuts; bands `[P, Q]` other than `[0, 1]`;
     whole-row collapses with lane-varying arms; an interval far from zero).
     It found one miscompile. A slope the e-graph can prove zero — a literal
     `k = 0`, or `[x < a]` — makes `ClampMoment`'s sweep `d` provably zero,
     and the algebra's `x·recip(x) = 1` and `(x·a)/a = x`, sound for every
     `x` but zero, then merged the quotient `N/d` with arbitrary classes: the
     chord's area extracted as the constant `0`, with no integral left, so
     the closure pin passed. The `Select` around the quotient cannot prevent
     that, because the e-graph reasons about the quotient's class whatever
     consumes it; the divisor is now `select(narrow, 1, d)`, which no rule
     can prove zero. A literal `k = 0` then left `h·∫ 1` — the constant
     rule, not built — which quadrature computes exactly.
   - **Adversarially reviewed**, with three changes:
     - `NarrowInterval` keeps the factors its variable does not reach
       *outside* the integral it builds. Read at the reparametrized point,
       they were the whole integrand whenever the product held nothing
       else, and `∫ C` is the constant rule's. That was the literal-`k = 0`
       chord's `h·∫ 1`; it now closes, and its pin says so.
     - The closing phase spends the run's rounds rather than a round cap of
       its own. A caller that allowed `n` rounds was told of up to `2n`,
       which `run_anytime_curve`'s per-call subtraction would underflow on.
     - `FactorFold`'s floating-point note: several `Add` factors out of a
       `min`/`max` re-associate, so only one comes out bit for bit.

     Inert without an integral, measured: the glyph bakes of
     `glyph_atlas_golden` and `font_rasterization_regression` write
     saturation telemetry (rounds, applications, unions, classes, stop,
     extracted cost) identical record for record at `df2907f` and after
     this step — the n-ary `FactorFold` included.
5. **Half-plane, conic and Taylor** (a-glyph-is-a-formula §4.1, §7).
6. **The glyph** (a-glyph-is-a-formula §6).

**The demand track** runs in parallel and does not gate the build order above:

- **D0. Reproduced and fixed** (`fix(codegen): a fold is a consumer of what
  its body reads`). `select_arms` built consumers from `operands`, which treats
  a `Reduce` as a leaf (`regalloc.rs:3239-3262`), so a fold body's reads were
  invisible to the guard analysis. That caused two miscompiles:
  - A guard skipped a fold whose accumulator a sibling fold still read. 176 of
    256 texels were wrong.
  - Clustering moved a fold's input to after the fold. 76 of 256 texels were
    wrong, even when no guard fired.

  Each fold now carries edges to the parent-scope values its body reads
  (`FoldReads`). `pixelflow-core/tests/guard_sibling_fold.rs` pins the fix with
  16 kernels checked against `f64` references. Glyph output is bit-identical
  before and after.
- **D1.** Regions of equal demand replace `closed_exclusive` and `OUTSIDE`.
- **D2. Done** (`perf(codegen): a fold in a guarded arm is priced by its
  trips`).
  - On HELLO at 20 px, each glyph's distance select is now guarded. It skips
    its 8–32-trip loop on every batch outside the glyph's box. `uncached_HELLO`
    ran 12–19% faster, and glyph output is bit-identical.
  - The winding selects are still not guarded, as predicted. Two selects read
    the winding fold, so exclusivity cannot claim it. That is D1's job, or step
    6's, since after step 6 a glyph has a single fold.

## 9. Open questions

- **Algebra rules that are unsound at zero.** `InverseAnnihilation::<MulRecip>`
  (`x·recip(x) = 1`) and `Cancellation::<MulRecip>` (`(x·a)/a = x`) carry no
  `x ≠ 0` side condition. No kernel divided by a provably zero class before
  closed forms did; `mean_of_clamp` now never does. Whether the rules should
  instead refuse a divisor whose class holds, or may come to hold, zero is
  open — a const fact can say "is zero" but not "is never zero".

- ~~A cell on `Fold`, or a sibling `Area` node.~~ **Decided (JP): an integral
  is a fold over a continuous domain, with no cell and no axis.** About 39
  files match a `Reduce`-shaped node. Most read `len()`, `range()` or
  `stride()`, which have no meaning on an interval. Those sites must refuse an
  interval, or never see one, since quadrature removes it before codegen.
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
