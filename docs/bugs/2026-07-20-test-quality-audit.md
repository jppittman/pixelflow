# Test quality control pass — 2026-07-20

Scope: (1) static audit of whether tests exercise only public API, per
docs/STYLE.md's "Test Public API" rule, across all 11 library crates; (2)
`cargo-mutants` mutation testing on `actor-scheduler` (full crate, 236 mutants)
and a core-algebra subset of `pixelflow-core` (algebra.rs, dual.rs, ops/trig.rs,
ops/compare.rs, mask.rs, combinators/select.rs — 375 mutants; a full-crate run
was 4026 mutants, infeasible in one pass).

## Fixed and pushed (branch `claude/nifty-maxwell-d4x2zr`)

1. **Deleted 5 dead files**: `pixelflow-core/src/backend/{x86,arm,scalar,wasm,fastmath}.rs`.
   None were declared via `mod` anywhere in pixelflow-core — confirmed by
   `cargo check` before and after removal, both clean. They were stale
   duplicates left over from the extraction of backend code into
   `pixelflow-ir` (the real, live modules, re-exported via
   `pub use pixelflow_ir::backend::*;` in `pixelflow-core/src/backend/mod.rs`).
   Their tests (e.g. an `avx512_log2` test) looked legitimate and reviewed but
   silently never compiled or ran — worse than a brittle test, a total no-op.
2. **`pixelflow-runtime/src/platform/macos/window.rs`**: `window_creation` test
   now calls the existing public `MacWindow::size()` accessor instead of
   reading private `current_width`/`current_height` fields directly.
3. **`actor-scheduler/src/params.rs`**: added `bounds_are_ordered_and_contain_defaults`.
   `SchedulerParams::bounds()` had *zero* test coverage — cargo-mutants found
   all 8 mutants on its return value survived. New test checks `lo < hi` for
   every one of the 10 tuned dimensions and that the default value falls
   within bounds.
4. **`pixelflow-core/src/mask.rs`**: added a full unit-test module. This file
   had no `#[cfg(test)]` module at all, and mutation testing confirmed it
   empirically — **every single method** (`all_true`, `all_false`, `any`,
   `all`, `none`, `select`, `select_opt`, `to_field`, `from_field`, `bitand`,
   `bitor`, `not`, both `From` impls) survived being replaced with a no-op/
   `Default::default()` stub. `Mask` is the primitive behind every conditional
   in the public algebra (CLAUDE.md: "conditionals are performed using Select
   or postfix `.select`"), so this was the most significant coverage gap found.
   Re-running cargo-mutants scoped to this file after the fix: **21/24 caught**
   (2 unviable, 1 remaining miss is an equivalent mutant — `all_false()`'s body
   is definitionally identical to `Mask`'s derived `Default`, not a real gap).

All new/changed tests verified passing (`cargo test -p actor-scheduler --lib`,
`cargo test -p pixelflow-core --lib mask::`) and `cargo check --workspace` still
clean.

## Investigated, judged not to need a fix

- **`pixelflow-core/src/lattice/tests.rs`** — test fixtures (`XPlusY`,
  `ZTimes100`) call `x.raw_add(y)` / `.raw_mul(...)`. Initially looked like a
  CLAUDE.md "no raw ops" violation, but `raw_add`/`raw_mul` are `pub(crate)`
  `Numeric` trait methods, and hand-writing a `Manifold::eval` from concrete,
  already-evaluated `Field` values is the sanctioned low-level pattern the
  crate's own combinators use internally — there's no other way to combine two
  concrete `Field`s outside the AST-composition layer. Not a violation.
- **`actor-scheduler/src/registry.rs:81` `PodSlot::mark_restarting`** — the
  mutant deleting its body survives because `SlotState::Connected` and
  `SlotState::Restarting` are handled by the *same* match arm in `reconnect()`
  — there is currently no public-API-observable difference between the two
  states. Not a testing gap so much as a design question worth a human look:
  why does `Restarting` exist as a distinct variant if it's behaviorally
  identical to `Connected` today? Adding a test would require exposing new
  introspection API just to make the mutant killable, which wasn't done.

## Static audit: tests exercising private internals instead of public API

### pixelflow-graphics — systemic violation (not fixed, needs a design call)
- `src/spatial_bsp.rs`: ~20 tests reach directly into private fields
  `bsp.interiors[...]` (private `InteriorNode` fields: axis/threshold/children)
  with no public accessor exposing per-node structure. Lines: 503, 517, 537,
  1063, 1084, 1101, 1122-1273, 1340-1367, 1471-1526, 1529-1579, 1582-1673,
  1676-1721. These check real structural invariants (valid indices, reachable
  leaves, distinct children) that arguably deserve *some* verification — the
  fix isn't obviously "delete them," it's a judgment call between adding a
  test-only introspection API vs. accepting this as a documented rule-break
  (docs/STYLE.md's Flexibility clause) vs. rewriting as property tests over
  `eval()` output only.
- `src/subdiv/mod.rs:848` `tile_scale_values` calls private `tile_scale()`
  directly; the same behavior is already covered via the public `eigen_patch()`
  at line 802 — this one looks safely removable.

### actor-scheduler — kubelet.rs cluster (not fixed)
- `src/kubelet.rs:840-857` constructs the private `ManagedPod` struct and calls
  private `within_budget()` directly (no public constructor exists).
- `src/kubelet.rs:772-776, 869-881` read the private field `kubelet.poll_interval`
  directly (no public accessor).
- `src/kubelet.rs:897-905` reaches into private `kubelet.pods[...]` fields.
- `src/lib.rs:1462-1604` `backoff_unit_tests` calls private `backoff_with_jitter`
  / `send_with_backoff` directly — see mutation findings below: these tests are
  *also* weak, despite (or because of) testing private internals.

### core-term (audit incomplete — 3 confirmed findings, 5 files unaudited)
- `src/io/event_monitor_actor/writer.rs:218-230,244-263,267-297,300-308` —
  tests construct the private `PtyWriter` via raw struct literal and assert on
  private fields (`pending`, `pending_resize`), calling `handle_data`/
  `handle_control`/`handle_management` directly instead of through the actor
  message-passing surface. Equivalent behavior is already covered publicly in
  `mod.rs`.
- `src/terminal_app.rs:898,932-938` — `create_test_app()` calls the private
  `TerminalApp::new_registered` / constructs private `TerminalAppParamsRegistered`
  directly, bypassing the real public constructor `spawn_terminal_app`; also
  calls private `find_font_path()`.
- `src/term/screen.rs:1183` — `create_test_screen_with_scrollback` writes
  directly into the private `scrollback_limit` field (no public setter exists).
- **Not audited** (sub-agent didn't return in time): `term/emulator/mouse.rs`,
  `term/emulator/key_translator.rs`, `term/emulator/input_handler.rs`,
  `term/layout.rs`, `term/unicode.rs`.

### pixelflow-core / pixelflow-ir (minor, not fixed)
- `pixelflow-ir/src/backend/emit/mod.rs:4436-4481,4096-4133` call private
  `arena_to_schedule`/`needs_arena` with no public equivalent in-file — a
  compiler/JIT-backend internals test, closer to the STYLE.md carve-out for
  "the only reasonable seam," but worth a second look.
- `pixelflow-ir/src/backend/x86.rs:2121-2139` tests the internal `F32x16` lane
  type directly rather than through `Field::log2` — the kind of "hints at
  lanes" issue CLAUDE.md warns against, though low severity and standard for
  JIT backend unit tests.

### pixelflow-runtime — window.rs (window_creation fixed above; rest left as-is)
- `metal_layer_config`/`resize_state` reach into private fields/raw NSWindow
  handles; no public equivalent exists for Metal layer config, and
  `resize_state` has an inline comment explaining why it checks OS ground
  truth instead of our own `poll_resize()` tracking. Left as-is — a
  justified rule-break per STYLE.md's Flexibility clause.

### pixelflow-compiler (minor)
- `src/optimize.rs:1543-1607` bypasses the public `optimize()` entry point to
  call private `optimize_expr_with_model` + construct `Extraction::Nnue`
  directly, testing the opt-in-only NNUE path (CLAUDE.md: not the default),
  with weak assertions ("just verify it's well-formed").

### Compliant, no violations found
pixelflow-search, pixelflow-pipeline, pixelflow-ml, actor-scheduler's
sharded/registry/spsc/lifecycle/params/service modules, the bulk of
pixelflow-core/pixelflow-ir, the bulk of pixelflow-graphics, and (as far as
audited) core-term's ansi/*, keys.rs, io/event_monitor_actor/mod.rs,
io/pty_tests.rs, io/waker.rs, term/core_tests.rs.

## Mutation testing: actor-scheduler (236 mutants, 14 min)
**45 missed / 108 caught / 60 unviable / 23 timeout.** Beyond the fixed
`params.rs::bounds()`:
- `lib.rs` `backoff_with_jitter` (private, jitter math) + `send_with_backoff`
  (private, retry/backoff comparisons): ~15 missed/timeout mutants on
  comparison operators and arithmetic. Same functions flagged above as
  private-API test violations — `backoff_unit_tests` both breaks the
  "test public API" rule *and* fails to catch real behavior changes in the
  code it exists solely to test.
- `lib.rs` `handle_wake`: 5 missed mutants on `||`/`==` conditions gating which
  lane gets woken (lines 938, 971-973, 983).
- `lib.rs` `drain_all_with_timeout`: 2 missed `delete !` mutants (915, 916) — a
  negated condition whose both directions aren't verified.
- `sharded.rs` `ShardedInbox::drain`: 8 missed mutants across the round-robin
  index computation (`%`, `+=`) and the per-shard/total limit break condition
  (`||`, `>=`, `!`) — lines 119, 134, 138, 140. Existing tests check aggregate
  outcomes (`round_robin_fairness`, `burst_limit_respected`,
  `per_shard_limit_enforced_independently_of_total`) but not exact
  shard-rotation-across-calls or the precise total/per-shard boundary
  interplay.
- `kubelet.rs` `poll_pods`: 3 missed `+=`→`*=`/`-=` on a counter (481, 485,
  496), plus 7 timeouts (`poll_pods` bounds/`&&`, `within_budget`) — the
  already-flagged private tests are additionally weak; several mutants there
  cause the test to *hang* (timeout) rather than fail, worth a look
  independent of mutation testing.
- `spsc.rs` `try_send`/`try_recv`: 1 missed + 4 timeouts on core ring-buffer
  arithmetic/comparisons — SPSC correctness is concurrency-critical; timeouts
  specifically are concerning since a subtly-wrong mutant hangs rather than
  fails cleanly.
- Debug/Display `fmt` impls (error.rs, lib.rs, service.rs): 5 missed, low
  priority — standard practice not to assert on exact Display/Debug text.

## Mutation testing: pixelflow-core algebra subset (375 mutants, 18 min)
**281 missed / 53 caught / 40 unviable / 1 timeout** before the mask.rs fix
(mask.rs's 22 then became 1/24 after the fix — see above). Breaking down the
rest:
- **`ops/trig.rs` (92 missed)** — real, high-priority finding. `inv_pi`,
  `inv_two_pi`, and `range_reduce_pi` (the private range-reduction helpers
  behind `sin`/`cos`/etc.) have *every* mutation survive: replacing the
  constants with 0.0/1.0/-1.0, and every `+`/`-`/`*`/`/` in the reduction
  arithmetic. This means trig correctness outside the already-reduced
  principal range has essentially no verification — a badly wrong range
  reduction would currently go undetected.
- **`ops/compare.rs` (29 missed)** — `smoothstep_sigmoid`, `SoftGt`, `SoftLt`,
  `SoftSelect` (differentiable/soft comparison combinators used with `Jet2`
  autodiff) have no numerical correctness tests at all; also `Le`/`Eq`/`Ne`
  `Manifold::eval` bodies. Not fixed here — deriving correct expected values
  for the soft/autodiff variants needs care I didn't want to rush.
- **`algebra.rs` (97 missed) / `dual.rs` (26 missed)** — largely **not** a
  real gap: most of these are inside
  `#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]` blocks
  (the scalar-fallback `impl Transcendental for f32`, and
  `impl<N, A: Transcendental> Transcendental for Dual<N, A>`, which requires a
  concrete `Transcendental` impl that only exists on that same excluded-here
  cfg). On this x86_64 CI/dev machine, that code doesn't even compile in, so
  any mutation to it is trivially invisible — not a test-quality problem, but
  it does mean **the scalar-fallback numeric path has zero verification
  signal from any mutation run (or likely any test run) done on x86_64/aarch64
  hardware.** If that fallback path matters (e.g. for wasm32 or other
  architectures), it needs its own CI leg to get any signal at all.
- **`combinators/select.rs` (16 missed)** — a `delete !` mutant survives across
  many monomorphized `Select<C,T,F>::eval` impls for different tuple/array
  arities (lines 265, 303, 344, 377, 412, 448, 528, 557, 587, 618), plus two
  `FieldCondition::eval_mask` no-op mutants for `Or`/`Field`/`Abs`. Suggests
  the negation logic in `Select` is only exercised for a subset of the
  supported arities.

## Recommended next steps (not done here — judgment calls or more time needed)
1. Decide `spatial_bsp.rs`'s fate: add test-only introspection, rewrite as
   property tests over `eval()`, or explicitly document the rule-break.
2. Same decision for `kubelet.rs`'s `ManagedPod`/`poll_interval` tests — likely
   fixable by testing through `Kubelet::run()`/the public builder instead.
3. Strengthen or replace `actor-scheduler`'s `backoff_unit_tests` — it violates
   the public-API rule *and* the mutation run shows it's not even catching bugs
   in the private functions it exists to test.
4. Write real correctness tests for `ops/trig.rs`'s range reduction and
   `ops/compare.rs`'s soft-comparison combinators — the single biggest
   remaining coverage gap found in this pass, on core public-facing math.
5. Finish the core-term audit for `term/emulator/{mouse,key_translator,input_handler}.rs`,
   `term/layout.rs`, `term/unicode.rs` (the sub-agent auditing these never
   returned).
6. Consider whether the scalar-fallback (`f32: Transcendental`,
   `Dual<N, A: Transcendental>`) code path needs a dedicated CI leg, since
   normal x86_64/aarch64 test/mutation runs give it zero signal.
