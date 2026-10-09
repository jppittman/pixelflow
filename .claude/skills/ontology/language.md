# The language and its IR

### Kernel

- **Is:** a function of the coordinate and of its arguments, held as an
  **open value**: an arena term with a root that nothing has optimized. "A
  kernel is a function, and everything it does not contain is a parameter."
  It becomes numbers exactly once, when it is compiled at a lattice's shape
  and collapsed. The Rust `Kernel` is the opaque value a `kernel!` entry
  returns and that `Manifold::compile` and `Lattice::bake` take.
- **Is not:** compiled code (that is a Manifold). Not optimized: "A
  `kernel!` expansion and a combinator produce the same thing: an open
  value. Neither optimizes." Not a surface or a builder: "The builder is not
  a surface". JP, 2026-09-25: "The builder isn't supposed to be a thing."
  Not evaluated at a point or per batch. Not a container for its own loop
  structure, trip counts or pruning; writing those in is "the author doing
  the compiler's job … That is blaming the user". It knows nothing about
  lanes, rows, memory, colour or terminals. Superseded meanings:
  - ZST expression-template types checked by rustc (inverted 2026-07-23).
  - "kernel! is the kernel" (2026-09-06, lasted three days).
  - The fluent builder as the surface graphics builds on (2026-07-20/23,
    #1235, CLAUDE.md "Composing Kernels"), reversed on 2026-09-25: "The
    syntax lost to the builder because every feature was given to the
    builder first."
- **Follows:**
  - Composition substitutes and links, and never evaluates or optimizes.
  - So a `Dwrt` survives composition, and the chain rule holds under a warp.
  - Nothing downstream may assume a composed arena is already inlined.
  - The only way to numbers is compile → bind → collapse.
  - The fluent constructors leave the public API (Phase D-d), and only
    compiler crates may depend on `pixelflow-ir` (D16, enforced in CI).
- **Lives:** `pixelflow_ir::Kernel` (`pixelflow-ir/src/kernel.rs`),
  re-exported by `pixelflow_core` (`pixelflow-core/src/lib.rs`). Decided in
  `docs/plans/2026-09-25-the-language-is-kernel.md` §0–§1.1,
  `2026-09-24-one-pipeline.md` §1.2, and `2026-09-06-kernel-with-a-lattice.md`
  "The shape". Today these contradict the entry:
  - `select`, `at`, `over`, `sum_over`, `dwrt` and `by_ref` are still public.
  - `kernel.rs` calls itself the "JIT-first surface" and says "Composition
    is arena splicing".
  - CLAUDE.md still teaches `Kernel::at`/`select` as the composition API.
  - Every `Kernel` holds its graph twice (`rooted` and `legacy`). Removing
    that is exprarena-on-dag Stage D.
  - A `kernel!` entry's `Kernel` comes back already saturated (see kernel!),
    so it is not an unoptimized open value.
  - `pixelflow-graphics` depends on `pixelflow-ir` directly (its
    `Cargo.toml`; `ExprArena` in `fonts/loop_blinn.rs`,
    `render/cell_grid.rs`, `render/packed.rs`), and no CI job checks D16.

### kernel! (the language)

- **Is:** the one language. It has one syntax and one parser (the
  compiler's, run inside the macro), and one lowering that calls
  `pixelflow-ir`'s single definitions. A program is a block of items: `struct`
  records of `f32` fields, `const` items and `fn` items. `|a: T, …| e` is
  sugar for a block with one entry.
- **Is not:**
  - A tier or an optimizer: nothing is optimized at expansion.
  - Sugar over the builder. That was one-pipeline §3.2's first revision,
    reversed by JP on 2026-09-25.
  - A language with recursion, loops with state, `mut`, assignment, or
    collection types. All of these are refused.
  - Parsed a second time at run time (`training/factored.rs`'s parser goes,
    D17).
  - Retired macros: `kernel_jit!`/`kernel_value!` (merged in S4b-2) and
    `kernel_raw!` (to be deleted, D14: "Every JIT compile optimizes, so its
    promise never reached machine code").
  - Families and the `integral`/`area`/`monotone_root` syntax were built and
    then deleted (d8c39481, 5df0ece2).
- **Follows:** lowering holds no copy of fract, hypot, clamp, the
  derivative, the binder's choice and rename (`close_over`), or a range's
  bound (`Fold::admits`) (B5). Production kernels move onto `kernel!` (C1,
  D-c).
- **Lives:** `pixelflow-compiler/src/` (`lib.rs`, `parser.rs`, `sema.rs`,
  `lower.rs`, `emit.rs`); the-language-is-kernel §1.1–§1.2. Today
  `kernel_raw!`, `macro_tier()` and `DwrtFree` are still in `lib.rs`, and
  `kernel!` is `expand(input, &mut macro_tier())`: it saturates at expansion
  unless the term carries a `Dwrt` or structural parameters. That
  contradicts "nothing is optimized at expansion". `parse_expr` is still in
  `pixelflow-pipeline/src/training/factored.rs`.

### Entry, helper, record, Args

- **Is:**
  - An **entry** is a `pub fn`. The macro emits a host function for it that
    instantiates the lowered template with its structural values and
    returns the `Kernel`.
  - A **helper** is a private `fn`. Lowering inlines it (β-reduction), and
    it takes coordinates as arguments.
  - A **record** is a `struct` of named `f32` fields. It is flattened at
    lowering into uniforms in declaration order.
  - **Args** is each entry's record of its non-structural arguments. A
    compiled program is bound from `&Args`, so "every argument supplied" is
    a type, not a runtime assert.
- **Is not:** a helper as an optimization unit (D19, withdrawn 2026-09-29:
  "a helper holds none" of the integrals it existed for). Helpers are not
  callable across blocks; across blocks a kernel crosses only as a
  kernel-typed argument (D6). A record is not a collection and not an IR
  type. Args is not yet a guarantee of *which* program it binds: today it
  is bound by identity through `UniformBlock::set(Uniform, f32)`, which is
  superseded.
- **Follows:** X and Y appear only in entries, so a helper cannot read an
  unshifted X by accident. `write_into` streams values by position and
  checks only the count, so one entry's Args written into another program of
  the same arity "binds without a word and draws plausible wrong pixels".
  The block must therefore be typed by its entry (`UniformBlock<A>`) or
  carry a per-entry token. A composed program has no entry of its own, so it
  needs this before C2 (O3).
- **Lives:** generated by `pixelflow-compiler/src/emit.rs` (`write_into`
  returns `Result<(), ArityMismatch>`);
  `UniformBlock::set_declared`/`ArityMismatch`
  (`pixelflow-core/src/lattice/manifold.rs`); the-language-is-kernel §1.2,
  §1.4, §1.8, D6, D7, D19. Today `UniformBlock` is untyped, so the
  positional binding above is live.

### Binding time (a name bound later)

- **Is:** the one axis on which a kernel's parameters differ. There are five
  spellings today, which "differ in **binding time**, not in kind":
  - `Var` X/Y, bound per sample by collapse.
  - `Var` binders, bound per term by their fold.
  - `Uniform`, bound per call by the block.
  - `Buffer`+`Gather`, bound per bind.
  - `Ref`, bound at link.

  An author sees three binding times: **structural**, **uniform** and
  **kernel-typed**.
- **Is not:** kinds of thing that passes may tell apart. "A pass may ask
  when a name is bound and must not care what it names." That is what makes
  `contains_gather` unstatable rather than merely discouraged. Not decided
  by the call site's Rust type: uniform-slot-identity's "an `f32` folds, a
  `Uniform` becomes a slot" was superseded on 2026-09-25 ("An f32 argument
  no longer folds into a constant because of its type"). Not a tier.
- **Follows:** `Gather` leaving the algebra is a consequence of this, not
  the goal. Invariance is variance. A structural change recompiles and a
  uniform change never does. A uniform's default stays out of the key, while
  a buffer's extents are in it. The instruction chosen for a binding time
  (gather, broadcast, uniform load) is codegen's business.
- **Lives:** `ExprNode::{Var, Uniform, Buffer, Ref}`
  (`pixelflow-ir/src/arena.rs`); `Variance` (`pixelflow-ir/src/variance.rs`).
  Decided in `2026-09-10-one-name-bound-later.md` §1–§5 (status Draft) and
  the-language-is-kernel §1.4, D1. `pixelflow-compiler/tests/binding_times.rs`.

### Structural parameter

- **Is:** a parameter bound at instantiation, where each value is a
  different program: "`const N: usize`, a zoom level's tile extent; the
  font's shape, meaning which glyphs and how many pieces each". Fold ranges
  are expressions over literals and structural parameters.
- **Is not:** a type for a count ("A count of things is not a type. It is
  how many instances the host composed"). Not a uniform. Not variable at run
  time. JP may later allow "uniform iterations", which are "no where near
  the docket".
- **Follows:** every range is constant, so every read point's range is known
  at construction, so a new tile extent or zoom recompiles and is cached by
  key. A font's glyphs are drawn by one program per piece count, chosen by
  the count, a structural parameter (C1, 2026-10-09); the id tree whose
  threshold `k` raised "structural or uniform" is superseded.
- **Lives:** the-language-is-kernel §1.4–§1.5, §1.7, O2; one-pipeline
  decisions (JP, 2026-09-25). In code, a structural parameter read as a value
  (`N as f32`) is lowered to a `Param(k)` hole the host function fills
  (`count` in `pixelflow-compiler/src/lower.rs`), so it rides on the leaf
  the Param entry says has no job left.

### Uniform (the parameter)

- **Is:** a scalar `f32` slot in the program's block, invariant across the
  lattice and supplied per call. "Everything that is not structural is a
  uniform, and a uniform is a scalar." "A uniform is invariant on L, so its
  variance is CONST, and it is *unknown* on P, so it is never a Const."
  Identity is by instance: "two pieces are two factors of the block". A
  block is a point in the product of every instance's parameters. A slot is
  a projection of it, and linking is the flattening that chooses each
  factor's offset.
- **Is not:**
  - A constant. JP: "make the coordinates of the control points uniforms".
  - An extent: "A uniform can be a gather index; it can never be an extent."
  - An axis. Z and W and time became uniforms (L1).
  - An array, family or table. JP, 2026-10-01: "No arrays at all".
  - Identified by name or by a fragment-local index.
  - A `Gather` from a 1×n buffer.
  - State on the program: the block is an argument of the call.
  - Part of the compile key, by value or by default.
  - An iteration marker or a template input. B3's NaN-default marker
    "extended Uniform's meaning without extending its type".
  - The adjective "uniform" (next entry).
  - Superseded meanings: time on the W coordinate (kernel-with-a-lattice S3
    finding 5); "the call-site type decides fold vs uniform" (2026-09-06);
    "a table backed by 10N uniforms" and uniform families read at an affine
    index (one-pipeline §1.3/A8). The families were built (66c1f4a1) and
    deleted (d8c39481).
- **Follows:**
  - Uniform-only arithmetic runs once per call.
  - Compositions with equal structure share code and differ only in their
    blocks ("a thousand circles is one compile").
  - The e-graph hash-conses a uniform by identity and never folds it. Its
    gain in the e-graph is CSE. L4 measured one pixel moving one step.
  - `∂u/∂x = 0`.
  - Setting an unknown handle is an `Err`, because silence would give
    plausible pixels.
  - Defaults are in the IR, so a bake without a block is total.
  - A range implied by a mask that depends on a uniform is known at bind
    time.
  - A font holds about 16k uniforms, so the uniform chain is 64-bit.
  - `uniform_slot_for`'s linear search is quadratic on the zoom path, so how
    the host finds each instance's slots is open (O3).
- **Lives:** `ExprNode::Uniform(UniformId)`, `UniformId(pub u64)`,
  `UniformIdentity(u64)`, `UniformDecl`, `ExprArena::uniform_slot_for`
  (`pixelflow-ir/src/arena.rs`); `pixelflow_ir::Uniform`
  (`pixelflow-ir/src/kernel.rs`); `UniformBlock`
  (`pixelflow-core/src/lattice/manifold.rs`); `ScheduledOp::Uniform(base,
  offset)` (`pixelflow-codegen/src/program/mod.rs`). Decided in
  `2026-09-06-uniform-slot-identity.md` §1–§3,
  `2026-09-06-lattice-is-the-index.md` L1, and the-language-is-kernel §1.4,
  §1.6, B3, B6. Today the table in `variance.rs` says "Const | Compile-time
  constant — a uniform is here". `CONST` means lattice-invariant, not known
  at compile time.

### Uniform (adjective: lane-, batch-, row-, frame-uniform). Homonym

- **Is:** invariant over a scope. Two different facts share the word.
  - **Static** uniformity: the value's variance lacks a binder's bit. "Lane-
    uniform is 'does not depend on `l`'." Row- and frame-uniform are the
    same idea for other binders.
  - **Dynamic** uniformity: every lane of one batch's mask happens to agree.
    That is data (mask coherence) and is tested at run time.
- **Is not:** the `Uniform` leaf, which is only one way to be frame-uniform.
  Not static coherence ("Whether a mask is coherent … is data, and no
  static analysis can know it"). Not implied by placement: a value placed
  inside the lane scope is not thereby lane-varying.
- **Follows:** a lane-uniform `Gather` is emitted as a `Broadcast`. A
  frame-uniform value is placed in the body, a row-uniform one in the row
  fold. An `If` whose mask is uniform over a batch takes a jump. A
  row-uniform guard belongs at row scope. The retired axis bits 2–3 read as
  frame-uniform, which is why they are refused.
- **Lives:** `Variance` (`pixelflow-ir/src/variance.rs`);
  `ScheduledOp::{Broadcast, Lanes}` (`pixelflow-codegen/src/program/mod.rs`).
  `2026-09-16-collapse-is-a-fold.md` §2.2 and step 6; CLAUDE.md "`If`
  contains an if".

### Kernel-typed parameter

- **Is:** `k: impl Fn(f32, f32) -> f32`, "a kernel the host passes at run
  time, applied `k(x, y)`". Each application splices the argument's term,
  `k[X := x, Y := y]` (`ExprArena::apply`). The host function takes
  `&Kernel` and builds its program when called.
- **Is not:** usable in arithmetic, `let`-bound, returned, used as an `if`
  arm, or stored in a record field; each is a spanned error. Not copyable:
  it is moved, as rustc moves it (E0382/E0507). Not allowed in the closure
  form. Not a `.at` method.
- **Follows:**
  - A composed program declares the entry's uniforms first, then each
    argument's, so slot offsets are prefix sums (O3).
  - An argument applied at exactly (X, Y) is spliced as it stands, so a
    `Ref` inside it stays a unit. Under a warp it is expanded and gets a
    different cache key.
  - Inside a fold, binders are kept apart by closing the surrounding folds
    after the splice. No rename is done: "that is binding, not capture"
    (D4's rename is pending JP).
  - A mask kernel passed as `&Kernel` reads all-ones as NaN (see Mask).
- **Lives:** `ExprArena::admit`/`apply` (implemented in
  `pixelflow-ir/src/kernel.rs`), `Staged` (`pixelflow-compiler/src/emit.rs`);
  `pixelflow-compiler/tests/kernel_typed_parameters.rs`; the-language-is-kernel
  §1.3–§1.4, O2 (D-a, f0599991). This amends composition-is-linking:
  composition is a kernel-typed argument, not a method.

### Application (contramap, `at`)

- **Is:** precomposition with a coordinate map, `k.at(g) = k ∘ g`,
  implemented by substituting into `Var` leaves. "A **contramap** is where a
  coordinate frame lives": pixel centre `f ∘ (+½)`, placed glyph
  `f ∘ (−pos)`, DPI `f ∘ (·s)`. In `kernel!` it is application:
  `f(X + 0.5, Y + 0.5)`.
- **Is not:** a lattice origin. Not evaluation. Not a resolver of
  derivatives: a `Dwrt` inside survives and follows the warp, so
  `DX(X·X)` at `(2X, Y)` is 24 at x = 3, not 12. Not able to reach through a
  `Ref` (collapse refuses a reachable `Ref`). Not a method in the target
  syntax ("There is no `.at` method").
- **Follows:**
  - The lattice can be index-only.
  - Once a frame offset is in the arena, the optimizer may reassociate it.
    L2 measured 849 texels moving by up to 3.6e-5, and no same-form check
    could see it.
  - A glyph kernel bakes the ±½ pixel into its own frame, so it is correct
    under translation only. A scaling `at` covers a pixel of the wrong
    size.
  - `at` expands every `Ref` at construction, so a unit survives only when
    applied at exactly (X, Y). `Apply { inner: KernelKey, cx, cy }`, with
    expand, call or tabulate chosen at extraction, is unbuilt
    (composition-is-linking §8).
- **Lives:** `Kernel::at` and `ExprArena::apply` (`pixelflow-ir/src/kernel.rs`),
  `ExprArena::substitute_vars_with` (`pixelflow-ir/src/arena.rs`). Decided in
  `2026-09-06-lattice-is-the-index.md` "The shape" and §9.2, and
  the-language-is-kernel §1.2, D5. Today the pixel-centre ½ is not a
  contramap on the frame path: `PlaneRegion::rows` puts it in the band's
  origin (see Pixel).

### Coordinate (X, Y)

- **Is:** the two free variables a lattice binds per sample: X = `Var(0)`,
  the column, and Y = `Var(1)`, the row (`COORD_AXES = 2`). JP: "x, y, z,
  and w are privileged, but they're just uniforms" — conveniences for naming
  the lattice's binders. Collapse substitutes `X := x0 + col + lane` and
  `Y := y0 + row`. After that, no coordinate `Var` exists, only binders.
- **Is not:** a per-frame scalar. Time is a uniform: the 2025 rule "If a
  value changes per-frame, it belongs as a coordinate" was retracted in L1.
  Not four axes (Z and W are retired). Not an axiom codegen recognises by
  range ("`Var(0)` is lane-varying" went with the scaffold). Not a
  precoloured register.
- **Follows:** a warp is a substitution into `Var(0)`/`Var(1)`. `lower_dwrt`
  runs before collapse, so derivatives are taken before X is substituted.
  `Var(2)` and `Var(3)` are reserved and never reissued.
- **Lives:** `pixelflow_ir::arena::{COORD_AXES, Axis, RETIRED_COORD_AXES}`
  (`pixelflow-ir/src/arena.rs`), `Variance::{X, Y}`
  (`pixelflow-ir/src/variance.rs`). Decided in lattice-is-the-index §1 and
  §9.1, and collapse-is-a-fold §2.1. Today `variance.rs`'s header still
  describes bit 0 as "pixel column — varies per pixel".

### Var

- **Is:** an index leaf whose meaning is decided by its range:
  - `0`, `1`: the coordinates.
  - `2`, `3`: retired Z and W, reserved.
  - From `REDUCE_BINDER_BASE` (4): fold binders.
  - Past `Variance::VARIABLES`: fold placeholders during construction.
  - Rewrite-template metavariables.

  This is CLAUDE.md's canonical example of meaning living in a comment.
- **Is not:** one type. Not a manifold-param slot (`Var(8+k)`, `Var(128+k)`,
  `Var(192+s)`, deleted in S4b-2). Not a macro parameter (`Var(16+i)`,
  `PARAM_VAR_BASE`, deleted 2026-09-08: "It did not go out. It was
  reintroduced here, at a higher base").
- **Follows:**
  - Every reader decodes the range.
  - A retired axis is "refused, not unrepresentable". It would read as
    frame-uniform and hoist silently, so `emit::compile` refuses it
    unconditionally, and only on nodes reachable from the root.
  - Each meaning removed shrinks the hazard: the binder became `Binder`.
  - Once collapse has run, the axiom ranges have no reason to exist.
- **Lives:** `ExprNode::Var(u8)` (`pixelflow-ir/src/arena.rs`), whose doc
  still lists "a parameter placeholder"; `ENode::Var(u8)`
  (`pixelflow-search/src/egraph/node.rs`), whose doc still says
  "0=X, 1=Y, 2=Z, 3=W". Decided in lattice-is-the-index §9.1 and
  `2026-09-08-macro-tier-is-arena-native.md`. CLAUDE.md names two meanings
  (an axis or a binder); the code's own doc lists three.

### Param

- **Is:** an unbound scalar slot of a builder template. `Param(i)` exists
  only before the builder is called. In the e-graph it is an opaque leaf:
  no rule matches it and its derivative is 0. "`Param` keeps its meaning
  exactly … an unbound slot at bake time is still a bug and
  `Declined::Param` still means a builder was never called."
- **Is not:** an identity (`Param(u8)` "collides on splice, both ways"). Not
  a `Var` in a magic range. Not something the emitter accepts. Not a
  carrier of a loop body's per-iteration value. Superseded: "Parameters …
  baked into the kernel at build time. Different values = different kernel"
  (2025, retracted by JP on 2026-09-06: "My intent was actually for
  `Param(i)` to lower to a uniform, not a constant").
- **Follows:** with "everything not structural is a uniform" (2026-09-25),
  Param's job is gone. `ENode::Param` and the macro tier's saturation are
  "deleted, not completed" (one-pipeline M1–M5, F4).
- **Lives:** `ExprNode::Param(u8)` (`pixelflow-ir/src/arena.rs`),
  `ENode::Param(u8)` (`pixelflow-search/src/egraph/node.rs`),
  `Declined::Param` (`pixelflow-search/src/egraph/insert.rs`). Decided in
  macro-tier-is-arena-native and `2026-09-10-a-surviving-reduce-is-a-loop.md`
  2a″. Today it is still present, `Vocabulary::Templates` still admits it,
  and it has a new job: the template hole for a structural parameter read
  as a value (`count` in `pixelflow-compiler/src/lower.rs`), which refuses
  past 256 rather than wrapping.

### Binder

- **Is:** the index a fold binds. It is a scope ("Scopes are binders.
  Coordinates are bound by the lattice nest, reduction indices by
  `Kernel::over` — the same kind of thing, so they share the bitset"). "The
  binder is the only thing in the language that *shrinks* a variance set."
  It is stored as a slot, so "an index outside the binder space is
  unrepresentable". In sema it has type `usize`, converted by an explicit
  `i as f32`. The lattice's rows, batches and lanes are binders too.
- **Is not:**
  - A coordinate. Substituting a literal below the base would replace every
    X and give plausible but wrong pixels, which is why the binder gets a
    type and the trip count does not.
  - An accumulator.
  - Unique per fold: sibling folds may bind the same slot and share one
    `Var` node.
  - A register reserved for the loop's life (collapse step 1, retracted in
    step 2½).
  - A `Lane` leaf.
  - Kept apart by a depth counter (BinderScope became a set shared across
    threads).
  - Superseded: four binder slots in a `u8` variance (now 60).
- **Follows:**
  - Slots are assigned inside-out by `close_over`.
  - Placeholders sit past `Variance::VARIABLES`, or an outer rename captures
    an inner index (`sum_i sum_j f(i,j)` would become `sum_i sum_j
    f(j,j)`).
  - Anything keyed by a binder's `Var` identity aliases sibling folds. A
    binder slot keyed that way was one slot for two folds, and a label
    derived that way would be wrong too.
  - A binder is the loop counter: broadcast, stepped by 1.0, exact to 2²⁴.
  - The binder and the accumulator are roots the allocator places.
- **Lives:** `pixelflow_ir::fold::{Binder, Placeholder}`
  (`pixelflow-ir/src/fold.rs`), `arena::{REDUCE_BINDER_BASE,
  REDUCE_BINDERS}` (both `pub(crate)`), `ExprArena::close_over`
  (`pixelflow-ir/src/arena.rs`), `BinderScope` (`pixelflow-ir/src/kernel.rs`).
  Decided in `2026-09-09-a-fold-is-a-node.md` §3–§5, collapse-is-a-fold
  steps 2½ and 3, and the-language-is-kernel §1.3, §1.5, D4.

### Fold (Reduce)

- **Is:** `⟦Reduce{fold, body}⟧ = ⊕_{k ∈ fold.range()} ⟦body⟧[binder := k]`,
  meaning ⊕ over a set under a monoid. "A reduction … means the same thing
  computed as one term peeled off a shorter reduction, as a split into two,
  or as a fully unrolled chain." It is the one node that binds. Its monoid,
  binder and range are its identity, and its body is its only child. It is
  spelled `(a..b).map(|i| e).sum()`, `.product()`, `.any`, `.all`, or
  `.fold(±INF, min/max)`. "A fold is written once and stays a fold in every
  open value. Whether it runs as a loop or as copies is decided inside P, by
  extraction." A fold that survives is a loop.
- **Is not:**
  - Ordered, or carrying an accumulator, at the IR: "⊕ over a set has no
    accumulator; that absence is exactly what lets the e-graph reassociate
    it."
  - An op: there is no `Op` for a rule to name.
  - `Nary(Reduce, [Const(op), Const(var), Const(n), body])`: those children
    hash-consed with every literal of the same value.
  - Loop-carried iteration (refused, D11).
  - Unrolled by a pass or at construction.
  - Dynamically bounded.
  - An integral over a continuous domain: `Fold::Interval` was decided
    2026-09-23 and deleted 2026-09-29.
  - The style sense of "fold".
  - Superseded: "**a `Reduce` is not a loop** … emitted as `len()` copies"
    (a-fold-is-a-node §9, retracted 2026-09-10); `expand_reduce` (deleted
    2026-10-03); `ExprNode::Acc` (built and reverted: "A node every consumer
    refuses, including its target, is not a node in the language").
- **Follows:**
  - The empty fold is the monoid's identity.
  - Peel and halve are e-graph rules, so unrolling is extraction's choice.
  - A fold costs `len × body`, and its size for budgeting is also
    `len × body`.
  - `Dwrt` through a fold is linearity, which holds for Σ only.
  - Nested folds are nested loops.
  - A fold invariant in its enclosing binder is hoisted and runs once.
  - The lattice is three more folds.
  - Walkers must ask `children_slice` rather than match a case: "Thirty
    walkers matched a case to read a property", and a new variant was
    silently skipped.
- **Lives:** `ExprNode::Reduce` (`pixelflow-ir/src/arena.rs`),
  `pixelflow_ir::fold::Fold { monoid, binder, lo, hi, stride }`
  (`pixelflow-ir/src/fold.rs`), `ENode::Reduce`
  (`pixelflow-search/src/egraph/node.rs`), `ScheduledOp::Reduce`,
  `Scope::Fold` (`pixelflow-codegen/src/program/mod.rs`);
  `pixelflow-search/src/egraph/fold_rules.rs`. Decided in a-fold-is-a-node
  §2, §5, §6; a-surviving-reduce-is-a-loop; collapse-is-a-fold; one-pipeline
  decisions. Today CLAUDE.md's "The language is a DAG: no iteration binder.
  A fixed-count iteration is unrolled at construction" is stale
  (one-pipeline Appendix B).

### Monoid (SEQ)

- **Is:** a fold's combiner and its identity, `Monoid(OpKind)`: SUM (+, 0),
  PRODUCT (×, 1), MIN (+∞), MAX (−∞), ∨ and ∧ on masks. `Monoid::SEQ` is the
  unit monoid: "combine is sequencing, identity is nothing". Read
  monadically, a SEQ fold is `for_` and `Seq` is `>>`. ⊕ stays on the fold,
  because "unrolling by halving needs ⊕ first-class, and that is decisive".
- **Is not:** an `OpKind` encoded as a `Const` (the old `push_reduce`). Not
  an opcode codegen asks for: "codegen never asks an algebra for an opcode,
  because the IR spent it during lowering." A SEQ fold is not a value
  monoid; it is an effect.
- **Follows:** padding a table with identity rows leaves a fold unchanged.
  Stride-2 halving needs associativity only, while halve-and-offset also
  needs commutativity. Once lowered, the accumulate is an ordinary node and
  may fuse into an FMA. A lane fold over a value monoid is a horizontal
  reduction, refused until something needs one. SEQ has no `combiner_op`
  arm, and `Write` has no affine address, so fold rules cannot rewrite the
  lattice's folds (L3, Q6).
- **Lives:** `pixelflow-ir/src/fold.rs` (`Monoid`, `Monoid::SEQ`,
  `Fold::to_bits`/`from_bits`); `OpKind::Seq` (`pixelflow-ir/src/kind.rs`).
  Decided in a-fold-is-a-node §3–§4 and a-surviving-reduce-is-a-loop 2a″.
  Today `Fold::combine_op()` is a public "narrow door" the emitter's
  `Reduce` arm calls (`backend.alu(…, fold.combine_op(), …)` in
  `pixelflow-codegen/src/emit/mod.rs`), and its doc cites 2a″, which decided
  the opposite. The arm also branches on `monoid != SEQ` to skip seed and
  combine (as does `emit/regalloc.rs`). Both contradict this entry.

### Range (trip count)

- **Is:** a fold's half-open `[lo, hi)` with a stride, constant at
  construction. Its ends are expressions over literals and structural
  parameters. "The rule is that ranges have to be constant" (JP,
  2026-09-25). The range is the half that carries the load: peeling over a
  range leaves the body unchanged and therefore shared.
- **Is not:** a uniform or a runtime value ("a dynamic bound on `Fold` …
  Refused"; "a data bound is a dynamic trip count, which is refused and
  becomes a recompile"). Not implicitly zero-based. Not past 2²⁴, where an
  `f32` index stops being exact. Not a value of its own in the IR yet: "A
  range is a value", which would let sibling reductions over one range be
  one loop with one accumulator each, is proposed and unbuilt
  (a-glyph-is-a-formula §4.2).
- **Follows:** the trip count is in the kernel's key, so 34 pieces and 11
  pieces are two compiles. Constructors refuse a range past 2²⁴
  (`Fold::admits`). The allocator ranks carries by static trip counts.
  `bucketed_trip_count`'s power-of-two padding existed only because of
  this, and goes (G3).
- **Lives:** `Fold { lo, hi, stride }`, `Fold::admits`, `Fold::EXACT_BOUND`
  (`pixelflow-ir/src/fold.rs`). Decided in a-fold-is-a-node §3,
  a-surviving-reduce-is-a-loop §1 and §5, and the-language-is-kernel §1.5.
  Today the ends are `u32`, against the 64-bit rule, and
  `bucketed_trip_count` still pads in
  `pixelflow-graphics/src/fonts/loop_blinn.rs`.

### Unrolling

- **Is:** one extraction of a fold, reached by the e-graph's own rules.
  "Unrolling is the e-graph's `HalveFold` family. It is never a pass and
  never done at construction." HalveFold: "halve the trip count and double
  the body — `[lo,hi) step s` → `step 2s`, body `b ⊕ b[i := i+s]`".
  PeelFold peels once on an odd count, and EmptyFold is the base case. "A
  loop is a run-length-encoded unroll." "Nobody chooses between 'fold' and
  'unrolled': the author writes the fold, and extraction decides."
- **Is not:** a pass. JP: "I'm not anti unrolling. I'm anti bespoke
  unrolling pass." Not done at construction: one-pipeline Appendix A
  withdrew that after N separately written integrals failed to close from
  N ≥ 37. Not a re-bracketing of the range, which "buys nothing, because it
  is the same work in the same order". Not instance composition, which is
  the host walking data.
- **Follows:** halving costs log n rule applications where peeling costs n,
  and applications are the budget's currency. Unrolling needs a budget (the
  fold phase) and a price (trips). Admission bounds an unroll at about 100k
  classes. Whether a glyph emits a loop or straight-line code currently
  turns on a tie (see Tie-break). `pack(L)` is the same chunking law run as
  a pass (L2, open: "one chunking law, three implementations").
- **Lives:** `pixelflow-search/src/egraph/fold_rules.rs` (`HalveFold`,
  `PeelFold`, `EmptyFold`, `FactorFold`); `Fold::{peel, peel_back, halve}`
  (`pixelflow-ir/src/fold.rs`); `pixelflow-codegen/tests/halve_fold_jit.rs`.
  Decided in one-pipeline decisions (2026-09-25),
  a-surviving-reduce-is-a-loop 2a″ and E1b.

### Mask

- **Is:** a bit pattern: all-ones for true, all-zeros for false. It is the
  result of every comparison. In `kernel!` sema it is the type `bool`:
  comparisons produce it, and `&` and `|` combine it.
- **Is not:** a number. `1.0` is not true: `mask & 1.0` blends `7.0` and
  `9.0` into `4.5`. All-ones read as a number is NaN. Not something an
  arithmetic rule reasons about. Not hash-consed with a number: it is
  `Bits`, never `Num`. Not what a skipped arm's park holds. Not an IR type
  (D8: types in sema only). Superseded: mask ops kept out of the global
  vocabulary because registering them broke the AA ramp (2026-07-28). That
  reason expired with DwrtFree; the cause was optimizing an open term at all
  (one-pipeline §2).
- **Follows:**
  - `OpKind::mask(bool)` is the only constructor.
  - `is_bitwise_domain()` marks the ops whose results are patterns, and the
    folder's refuse-non-finite guard exempts them.
  - `If`'s lane-varying path is a bitwise blend.
  - Through a commuted `Min`/`Max`, a NaN mask became the gather index
    `i32::MIN`.
  - A mask that implies an index range yields a domain split.
  - `X.select(Y, 7.0)` becomes a type error.
  - Past sema a mask is a lane again, so a `bool` entry's kernel passed
    where `-> f32` is expected reads all-ones as NaN. Refusing bitwise roots
    does not close this, because `If` of numbers is in that domain. "The fix
    is a type, a mask kernel the host cannot pass as a `&Kernel`" (open).
- **At the machine:** a mask is a lane value, and how a lane value is held
  is the backend's associated type, `IsaBackend::Lane`. On AVX2 and NEON it is
  a vector register: those ISAs have no mask register, and their instructions
  take none. On AVX-512 a lane is in a vector register or in `k`: `vcmpps`
  writes `k`, and a writemask (`vblendmps zmm{k}`), `kand`/`kor` and
  `kortest` read it there. Only that backend knows a lane can be in `k`; the
  driver passes lanes through without asking. Where a lane is read in the
  other file, AVX-512's selection picks an instruction that reads it where it
  is (`vpternlogd` for a blend on a vector condition, `vpmovm2d` before an
  instruction with no `k` form). That is an ordinary instruction choice, not
  a conversion, and it needs no IR type. The machine cannot tell a mask from
  a number on NEON or AVX2, so the domain question ("a mask is not a number")
  is the IR's to answer, not the backend's.
- **Lives:** `OpKind::mask`, `OpKind::is_bitwise_domain`
  (`pixelflow-ir/src/kind.rs`); sema's `bool` (`pixelflow-compiler/src/sema.rs`);
  CLAUDE.md "Floating point at the edges"; the-language-is-kernel §1.3, D8,
  O2. Today `OpKind::mask` returns an `f32`, and `Bits`
  (`pixelflow-ir/src/kernel.rs`) types only the int conversion boundary: its
  own doc says "Comparison masks are also bit patterns and still travel as
  `Kernel`". Both contradict "not a number". At the machine, AVX-512 treats every mask
  as a vector: each compare is `vcmpps` into `k1` then `vpmovm2d` (an
  AVX-512DQ instruction) into a zmm, each guard `vptestmd` back into `k1` then
  `kortestw`, and `k1` is the only mask register the allocator knows
  (`pixelflow-codegen/src/emit/avx512.rs`) — which contradicts the paragraph
  above.

### If

- **Is:** dispatch: `if m then a else b`, two cases, both arms defined at
  every index. "Choosing among alternatives is `if`." It has one meaning and
  three lowerings, chosen by what is statically known about `m` and tried in
  this order:
  1. A domain split, when `m` implies an index range.
  2. A jump, when `m` is uniform over a batch (hoist and guard).
  3. A bitwise blend, when `m` varies by lane.

  A mask built only from uniforms is lowering 2's best case, not lowering
  3's.
- **Is not:**
  - A blend.
  - A fold.
  - Branchless by meaning.
  - A branch bought per `If`.
  - A narrowed domain ("A restricted region never restricts the output
    domain").
  - `Select`: renamed in D18, "The name is the bug". `Kernel::select`
    survives as an alias, and aliases go (D15).
  - Superseded: the style docs' branchless, one-case denotation (until
    #1208); demand-is-a-dag-property's "Select stays a blend"; the 07-20/
    07-28 reading of the guard as a purchase over a default blend;
    emit-should-just-emit's "Guard and If denote the same function", which
    is false for a lane-varying mask (Guard retired 2026-10-05).
- **Follows:**
  - A uniform mask takes an arm (a jump). The bitwise blend is the path for
    a mask that varies by lane.
  - The arms are blocks.
  - The choice of lowering "changes what is computed, never what is
    selected".
  - Arms are total (gathers clamp), so demand is an optimization and
    widening is always sound.
  - An arm's condition is a fact about the DAG, so ownership is read off
    the DAG in one pass, and contiguity follows from layout.
    `cluster_if_arms` was 73% of a glyph bake and is deleted (#1313).
  - `Union`, `IndexRange` used as a derived region, and `Support` are hand
    spellings of lowering 1, to be subtracted.
  - A tree of `if`s over a uniform, or over space, is a BSP nobody builds.
  - `IfHoistUnary` pulls shared work out of arms, against guarding, and no
    cost term opposes it (D7).
- **Lives:** `OpKind::If` (`pixelflow-ir/src/kind.rs`);
  `pixelflow-codegen/src/program/{ownership,layout,guards}.rs`; CLAUDE.md
  "`If` contains an if"; `2026-09-08-one-conditional-three-lowerings.md`
  §1, §9 (JP: "Select contains an if"); the-language-is-kernel §1.6.
  Lowering 1's analysis exists (D1, `pixelflow-ir/src/mask_support.rs`); the
  split itself, the bind-time tier, the interval tier and general guards are
  unbuilt (D2–D5). Open: layout grants a block only to an arm priced above
  `MISPREDICT_PENALTY_CYCLES` (`arm.cycles > MISPREDICT_PENALTY_CYCLES` in
  `program/layout.rs`) and merges cheaper arms into the surrounding region.
  That is the per-`If` purchase CLAUDE.md calls the backwards default, but
  #1313, the latest decision, kept it as a bound-gated pass over the
  finished DAG. Whether the bound decides *whether* `If` jumps, or only
  whether a cheap arm is worth separating, has not been decided.

### Constant (Const) and its two domains

- **Is:** a literal leaf known at construction, keyed by its bit pattern
  (−0.0 ≠ 0.0, and a NaN equals itself). It is two things that must never
  be unified: `Num`, an exact element of ℝ (`Dyadic`), and `Bits`, a
  pattern such as a mask or a shift result.
- **Is not:**
  - A uniform: "Today Const means both; this design splits them."
  - A carrier for metadata. push_reduce's Const-encoded op, binder and
    extent was the defect a-fold-is-a-node removed.
  - An f32 the e-graph may trust: "Constant folding in f32 computes
    f32-truths … the algebraic rewrites compute R-truths … their
    conjunction puts two UNEQUAL constants in one e-class."
- **Follows:**
  - ConstantFold routes results by `is_bitwise_domain`.
  - The union valve: Num/Num unions exactly, Bits/Bits by bits, Num/Bits
    always refused.
  - Exact dyadic folding declines rather than rounds, so the contradiction
    above cannot be constructed.
  - Crossing domains needs an explicit op (`TruncToInt`/`IntToFloat`).
  - In codegen a constant is a value (see Constant pool).
- **Lives:** `ExprNode::Const(f32)` and the private `NodeData::Const(u32)`
  (`pixelflow-ir/src/arena.rs`), `ENode::Const(u32)`
  (`pixelflow-search/src/egraph/node.rs`), `pixelflow_ir::dyadic::Dyadic`
  (`pixelflow-ir/src/dyadic.rs`); decided in
  `docs/plans/2026-08-08-egraph-constant-domain-spike.md` §4 and §6. Today
  there is still one variant "stored as f32 bits". The split is unbuilt, so
  the code stopped at the first link. `library::derivative`
  (`pixelflow-ir/src/library.rs`) still passes `Dwrt`'s axis as a `Const`
  operand.

### Dwrt (derivative)

- **Is:** the symbolic derivative node `Dwrt(e, axis)`: "derivatives are
  ordinary expressions compiled by the same backend". It stays unresolved
  while a kernel is open, and is lowered when the program is compiled,
  before collapse substitutes X.
- **Is not:**
  - Resolvable at expansion: "Resolving derivatives at expansion time was a
    miscompilation for four months" (12 where the truth is 24).
  - A jet or dual-number domain (Jet2/Jet3 retired).
  - Differentiable through a name (`Ref`), through memory (`Gather`), or by
    default through a fold, since linearity holds for Σ only.
  - Priced prohibitively: it costs 1000, so extraction may keep one for
    `LowerDwrt`.
  - Superseded: 07-20's "the e-graph ChainRule … at macro-expansion time;
    `lower_dwrt` is the fallback tier".
- **Follows:**
  - A warp substituted into a surviving `Dwrt`'s operand gives the chain
    rule for free.
  - Under one pipeline, nothing optimizes an open value, so `DwrtFree` stops
    being a guard and becomes a fact about P's signature.
  - The order is `lower_dwrt`, then `expand_transcendentals` (d sin = cos),
    then `collapse`. `collapse` refuses a reachable `Dwrt`.
  - The derivative of a select is a select of the derivatives.
  - Moving the axis off a `Const(f32)` touches the public `Kernel::dwrt(u8)`
    and needs JP's permission.
- **Lives:** `OpKind::Dwrt` (`pixelflow-ir/src/kind.rs`),
  `passes::lower_dwrt` (`pixelflow-ir/src/passes.rs`), `library::derivative`
  (`pixelflow-ir/src/library.rs`), `ChainRule`
  (`pixelflow-search/src/egraph/derivative.rs`), and the guard
  `pixelflow-compiler/tests/derivative_under_warp.rs`. Decided in
  macro-tier-is-arena-native, `2026-09-09-the-graph-differentiates.md`, and
  one-pipeline §1.2. Today `Rules::tabulation` (`passes.rs`), "a tabulation
  is a constant wherever its index is", is the remedy the-graph-differentiates
  §3 withdrew ("a special case invented to replace information that had been
  thrown away"). It was re-added to unblock S1b. That is an exception
  waiting for tables to leave the language.

### Ref, unit and link (composition)

- **Is:** composition by name. `Ref(KernelKey)` is a leaf meaning "evaluate
  that kernel here", named by content: "A reference to a kernel is a
  kernel." Since 2026-10-01 a reference is also an **optimization unit**. It
  is saturated and extracted by itself, held in the referring term as an
  opaque leaf that carries its variance (priced 0), and **linked** back in
  after extraction: `L_s(t) = link(Ô_s(t), r ↦ L_s(body r))`, under Law U,
  `⟦L_s(k)⟧ = ⟦expand_refs(k)⟧`. Splicing is the other form of
  composition, static linking: "Splicing is the mechanism … the operation is
  inlining".
- **Is not:**
  - Neutral "arena splicing": "Naming it after the memcpy hid the fact that
    a policy decision was being taken at all."
  - A call: L5 is unbuilt, and "the linker only inlines".
  - Reachable by substitution.
  - An open term: `by_ref` refuses a free binder.
  - A choice the e-graph makes: L4, `Ref(k) ⟷ body(k)`, is not built, and
    "under O1 no rule opens a unit".
  - An emission boundary: emission is still one flat program.
  - A unit once a `Dwrt` reaches it.
  - A unit inferred from frame-uniform `If` arms (rejected as a heuristic).
- **Follows:**
  - "The graph is acyclic the way a git history is: by how things are
    named." No occurs check is needed, and separate compilation is refused.
  - Two uses of one name are one node (`expand_refs` splices a referent
    once per key). By value, a piece cost 16k–58k nodes; by reference,
    about 720.
  - Units saturate once per structure, in parallel.
  - Rewriting across a unit's boundary is lost.
  - A `Ref`'s key digests minted identities, so build and run keys differ
    (see Compile cache key).
  - A loop body is "a reference to a DAG section", so peeling a fold whose
    body is a `Ref` needs L4 and L5.
- **Lives:** `ExprNode::Ref(KernelKey)` (`pixelflow-ir/src/arena.rs`),
  `Kernel::by_ref` (`pixelflow-ir/src/kernel.rs`), `KernelStore`
  (`pixelflow-ir/src/store.rs`), `passes::{expand_refs, link}`
  (`pixelflow-ir/src/passes.rs`), the unit leaf `ENode::Ref` and
  `EGraph::admit_unit` (`pixelflow-search/src/egraph/{node,graph}.rs`),
  `pixelflow_search::runtime` (`Program`, `Unit`;
  `pixelflow-search/src/runtime.rs`). Decided in
  `2026-09-09-composition-is-linking.md` §1–§3 (amended 2026-10-01) and the
  the-language-is-kernel O1 (86552ba9). Today the process-global
  `KernelStore` (a `static STORE: OnceLock<Mutex<…>>`) is one of P's
  impurities.

### Identity

- **Is:** provenance, minted and never inferred from shape. "Identity is
  provenance … You get one by minting it, and copy it into every declaration
  that names that memory." For uniforms, "Identity is the factor, i.e. the
  instance, not the name and not the index". For kernels, one canonical walk
  yields two projections:
  - **Structure identity** is the JIT cache key: the same program for every
    binding.
  - **Value identity** is `KernelKey`: "this kernel, over this memory".

  For rules, `RuleId` is derived from (name, specialization).
- **Is not:** a slot or index ("two arenas each call their own first uniform
  slot 0"). Not a name ("Two circles both have a cx"). Not an extent ("two
  atlases of equal size are a coincidence, not a fact"). Not one key serving
  both projections (composition-is-linking decision 2 was "exactly one bit
  wrong"). Not a `DefaultHasher` digest ("an identity that moves with the
  toolchain is not an identity"). Not a string with an id formatted in. Not
  a positional rule index.
- **Follows:** splicing merges tables by identity. ENode leaves that carry
  identity hash-cons by it, and no rule matches them. A tabulation's
  identity is `(key(f), shape)`. Minting refuses to wrap (`try_update`).
  The store keeps the full `Canonical` beside `KernelKey` and panics on a
  collision.
- **Lives:** `BufferIdentity(u32)`, `UniformIdentity(u64)`
  (`pixelflow-ir/src/arena.rs`); `KernelKey(u64)`, `canonical`, `Canonical`
  (`pixelflow-ir/src/key.rs`); `RuleId(u64)`
  (`pixelflow-search/src/egraph/rules.rs`). Decided in composition-is-linking
  §2 and §2.1, uniform-slot-identity §2.1 and §3.1, and
  `2026-09-02-optimizer-api.md` §6.1. Today `KernelKey`'s doc justifies 64
  bits by a 16-byte `ExprNode` cap that is now `<= 32`. The reason is
  stale; the width is correct.

### Table, Buffer, Gather, Broadcast

- **Is:** in the language, a table is nothing. JP: "We shouldn't have tables
  at all." Data enters a program only as a scalar uniform or as structure.
  In the IR today, a `Buffer` is bound memory with static extents and a
  minted identity. `Gather(buf, x, y) = buf[clamp ⌊y⌋][clamp ⌊x⌋]` reads it,
  with the base pointer bound per call, and is "the **one** dynamic-link node
  the language has". A `Broadcast` is a gather whose index lacks the lane
  binder's bit: one scalar load, splatted across lanes.
- **Is not:**
  - Something an author writes: "You should not know that you are doing a
    gather."
  - A parameter. JP: "Bound buffers need to go. That's not supposed to be a
    thing."
  - A GLSL-style uniform array, a family `[Row; N]`, or "another kernel … on
    a finite lattice" backed by uniforms (one-pipeline §1.3/Q2, superseded
    2026-09-25 and 2026-10-01).
  - Differentiable: you differentiate the code the memory caches.
  - Identified by its extents.
  - A reason to refuse hoisting. The 07-28 "Gathers never hoist" was
    deleted with the scaffold.
  - Superseded: composition-is-linking L6, "a tabulated kernel is a Ref
    with a cached tabulation", is subsumed by binding time (N1).
- **Follows:**
  - No f32-lane index guard, no bound buffer, and no `MAX_BOUND_BUFFERS = 4`
    panic ("a limit on how many symbols one program may name").
  - Choice is `if` plus bounding. "A fold's body reads nothing indexed by
    its binder except the binder's own arithmetic."
  - Once the frame draws from the font's programs (C2), the frame's cells
    and the atlas are the last buffers.
  - Broadcast and Gather are split once, in `arena_to_schedule`, by the lane
    bit: "decided once, where the DAG is read, not flagged per backend". The
    base is a `Context` pointer operand.
  - Where a tabulation is sampled off its lattice, JP's 2026-09-09 decision
    stands: "the API for bounding a kernel to a range should *require* the
    value outside it". Clamp-to-edge is "the instruction's answer, not the
    denotation's".
- **Lives:** `ExprNode::Buffer(BufferId)`, `BufferDecl`
  (`pixelflow-ir/src/arena.rs`), `OpKind::{Gather, RawGather}`
  (`pixelflow-ir/src/kind.rs`), `passes::expand_gather`,
  `ScheduledOp::{Gather, Broadcast}` (`pixelflow-codegen/src/program/mod.rs`),
  `Manifold::bind`, `MAX_BOUND_BUFFERS`
  (`pixelflow-core/src/lattice/manifold.rs`). Decided in the-language-is-kernel
  §1.6, D3 (2026-09-25, 2026-10-01), one-name-bound-later §3–§4, and
  collapse-is-a-fold step 6. Today the production glyph still folds over a
  `DiscreteManifold` table (`DiscreteManifold::new(rows, PIECE_ROW_COLS, …)`
  in `pixelflow-graphics/src/fonts/loop_blinn.rs`), and a table can now
  travel inside the kernel itself (`Kernel::with_buffer_data`, held by
  `Manifold`'s `carried` field). The table-free glyph is production since
  C1 (`FontPrograms`, `fonts/loop_blinn/program.rs`); the frame reads it from
  C2.

### Library (vs primitive)

- **Is:** a composite defined once over primitives, in
  `pixelflow_ir::library`, over the sealed `Terms` trait, so both front ends
  build through one definition: fract, hypot, clamp, derivative.
  `clamp = min(max(x, lo), hi)`, and "bounds with lo > hi give hi, as the
  composition says".
- **Is not:** an op that every backend and the e-graph decompose on their
  own. "Clamp's copies disagreed on lo > hi." Superseded: `expand_clamp`
  sorting the bounds (07-20).
- **Follows:** one definition reaches lowering and kernels alike, pinned by
  canonical key. "There is no second copy of fract, hypot, clamp."
- **Lives:** `pixelflow-ir/src/library.rs`;
  `pixelflow-compiler/tests/the_library_is_the_builders.rs`; the 2026-07-20
  root axiom; the-language-is-kernel §1.1.

### Transcendental expansion

- **Is:** `sin`, `cos`, `exp` and the rest have no instruction on any
  target, so the polynomial expansion in `passes` *is* what the opcode
  means: "The expansion is the denotation of the opcode, not an
  optimization over it." There is one definition, imported.
- **Is not:** a backend's business. Not an approximation of a function
  defined elsewhere. Not restatable: `sin` had four copies, and two shared
  one bug. Not defined outside its documented domain: `sin`, `cos` and `tan`
  return NaN for `|x| ≥ 2²⁰`.
- **Follows:** precision is a property of `passes`, while range is a hard
  property. The expansion runs after `lower_dwrt`. The e-graph runs before
  this lowering, so it cannot fuse the expansion's multiplies (an open gap).
- **Lives:** `pixelflow_ir::passes::expand_transcendentals`, `TRIG_DOMAIN`
  (`pixelflow-ir/src/passes.rs`); CLAUDE.md "Precision is on the table;
  range is not". Today the `passes.rs` module doc says both that the
  expansions "deliberately avoid `MulAdd` and `If`" and, two lines later,
  "They may use `If`".

### Totality

- **Is:** the root axiom: "The kernel language is total. It is not
  Turing-complete. Every program's cost is a closed-form function of static
  extents." Restated as strongly normalizing (2026-08-31).
- **Is not:** a language with `while`, `fix`, recursion, `mut`, or dynamic
  trip counts (`Fix` was removed: "it could not be given a static extent").
  Superseded: `iterate[N]` (07-20 P10).
- **Follows:** the cost model can be total. Saturation need not seek a
  fixpoint, so the optimizer is budget-only. Ranges are constant, and
  uniforms are never extents. Complex numbers, quaternions, polar
  coordinates, fract, hypot and clamp are library code.
- **Lives:** `docs/designs/2026-07-24-totality-and-the-cost-model.md`;
  `docs/plans/2026-07-20-kernel-unification.md` "Root axiom";
  the-language-is-kernel §1.5 (JP: "ranges have to be constant … we may
  relax this at some point").

### ExprArena (rooted term)

- **Is:** the sole IR: "a `Dag` of expression data, plus two identity
  tables, plus a choice of who may name a node". Construction interns: "two
  pushes of the same value — same shape, same children — return the same
  `ExprId`." An expression is a rooted term, a pair (arena, root): "Neither
  half means anything alone."
- **Is not:** a tree (the `Arc` `Expr` is deleted). Not append-only with a
  fresh id per push: "What a caller must not assume any more is that a
  `push_*` call returns a fresh id." Not visible to consumers. Not a place
  for a back edge.
- **Follows:** passes are endomorphisms on (arena, root). Construction
  garbage is unreachable, so every walk that matters (cache key,
  retired-axis guard, link order) asks only what is reachable from the root:
  "The garbage is a cost, not a contract." A name family like `*_arena` is a
  namespace smell. `Nary`'s raw offsets must not be published.
- **Lives:** `pixelflow-ir/src/arena.rs`, `pixelflow-ir/src/dag.rs` (`Dag`,
  `Node<'a, T>`, `Rooted<T>`, and the `pub(crate)` `Builder`). Decided in
  `2026-09-09-exprarena-on-dag.md` §1–§3 and `2026-08-17-cost-model-domain.md`
  J1. Today `ExprId(pub u32)` is public and forgeable. Retiring it for
  `Node`/`Rooted` is Stage D, not started, and `ExprId` is narrower than the
  64-bit rule. `ExprArena`, `ExprNode` and `pub mod arena` are re-exported
  publicly from `pixelflow-ir/src/lib.rs`, so the arena is visible to
  consumers.

### OpKind numbering

- **Is:** private to `pixelflow-ir`. Code that needs an op as bytes uses
  `OpKind::marshal`, and persisted bytes owe a format version.
- **Is not:** a dense index space to transmute from. Sparse discriminants
  made `from_index(17)` undefined behaviour. Not an encoding to store in a
  `Const`.
- **Follows:** per-op tables are `OpMap<T>`, and `index()`/`from_index` are
  `pub(crate)`.
- **Lives:** `pixelflow-ir/src/kind.rs`;
  `docs/designs/opkind-numbering-is-private.md`.

### Retired: Field, the combinator tier, the integral

- **Is:**
  - `Field`: nothing. It was the SIMD batch type of the per-batch `Manifold`
    ABI, deleted with the-isa-is-decided-at-startup ("Looking for Field:
    there is none").
  - The combinator tier (ZST combinators implementing per-batch
    `eval`, `Lower`, `HasIr`, `realize`): deleted, net −31k lines. "We don't
    want two languages."
  - The integral (`Fold::Interval`, `Kernel::area`, quadrature): deleted
    2026-09-29. JP: "just do b. delete all the integral stuff. other
    languages don't try this. probably for good reason."
- **Is not:** a fallback, a reference semantics, or a second language.
  Not something whose correctness rides on a saturation budget: a 189-piece
  glyph "had every integral quadratured, its coverage off by up to 0.92,
  and no test failed".
- **Follows:** nothing outside codegen's emitters names a lane. A closed form
  is written down (see Coverage).
- **Lives:** nowhere. Today `.claude/agents/{language-mechanic,numerics,
  pixelflow-core,pixelflow-graphics,pixelflow-ml}.md` still describe
  `Field`, `Jet2`, `Manifold` impls, `ColorCube`, `Baked` and `execute()`.
