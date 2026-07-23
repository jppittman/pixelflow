# Test quality control pass — 2026-07-23 (follow-up)

Continuation of `docs/bugs/2026-07-20-test-quality-audit.md` and the Jul-22
trig/compare fix. Scope this pass: (1) a fresh static audit of "tests exercise
only public API" (docs/STYLE.md "Testing") across all 12 crates, run via 4
parallel sub-agents since the Jul-20 pass's core-term audit was left
incomplete; (2) a targeted `cargo-mutants` re-run on `actor-scheduler`
against the specific functions the Jul-20 pass flagged as still weak
(`send_with_backoff`, `drain_all_with_timeout`, `handle_wake`); (3) fixing
what was safely fixable from both.

A full fresh mutation run across `actor-scheduler` (236 mutants) was started
but aborted partway through: `mod backoff_unit_tests`'s and
`mod drain_all_targeted_tests`'s "Kills: ..." comments show a full pass
already happened between Jul-20 and now, so a from-scratch re-run would have
mostly reproduced already-known results at real cost (~15 min/236 mutants on
this 4-core box). Re-running function-scoped (`-F`) against exactly the
functions the Jul-20 report named as unresolved was ~10x cheaper and gave
the same signal.

## Fixed and pushed (this session, branch `claude/nifty-maxwell-m4dtim`)

1. **`pixelflow-core/src/mask.rs`**: `mod tests`' `lane0()` helper read
   `Field`'s raw memory layout via `unsafe { *(&f as *const Field as *const
   f32) }`, assuming lane 0 sits at byte offset 0. Every other in-crate test
   already uses the existing `pub(crate) Field::store()` accessor for this;
   switched `lane0()` to use it too. Removes the only `unsafe` block in this
   test module; verified `cargo test -p pixelflow-core --lib mask::` still
   passes (10/10).

2. **`actor-scheduler/src/lib.rs`**: added 3 tests killing 5 of the 15
   mutants a targeted `cargo-mutants -F` run found still surviving in
   `send_with_backoff`/`drain_all_with_timeout`/`handle_wake` (full mutant
   list and per-test rationale below). Verified via `cargo test -p
   actor-scheduler --lib` (102/102 pass) and a scoped mutants re-run
   confirming all 5 targeted mutants now caught.

## Corrections to the raw static audit (false positives — not fixed, don't re-flag)

The 4 sub-agents doing the static sweep were instructed to flag any test
touching a non-`pub` item, then I hand-verified each flag against the
surrounding design before deciding to act. Three flagged items turned out to
be justified, not violations, on closer reading — worth recording so a
future pass doesn't re-flag them:

- **`pixelflow-core/src/lattice/tests.rs`** (`XPlusY`, `ZTimes100` calling
  `x.raw_add(y)` / `.raw_mul(...)`) — already investigated and dismissed in
  the Jul-20 report; re-confirmed here. `Add<L,R>::eval` in
  `ops/binary.rs:92-104` (the crate's own production `+` combinator) does
  the exact same thing (`self.0.eval(p).raw_add(self.1.eval(p))`) — raw ops
  on already-evaluated `Field` values is the only way to combine concrete
  Fields outside the AST-composition layer, and it's `pub(crate)`, not
  publicly exposed. Not a CLAUDE.md violation.
- **`pixelflow-compiler/src/optimize.rs:1561-1625`** (3 tests calling
  private `optimize_expr_with_model` + constructing `Extraction::Nnue(&model)`
  directly instead of the public `optimize()`/`optimize_with_model()`) — the
  public `optimize_with_model()` reads extraction strategy from
  `get_extraction()` (env-var gated per CLAUDE.md's NNUE-is-opt-in design),
  with no parameter to force a specific seeded model. These tests need a
  *deterministic* NNUE instance to assert against, which the public API
  intentionally doesn't expose a knob for. Not a fixable style violation
  without changing public API surface (which needs explicit permission per
  CLAUDE.md's "minimal public API" rule) — it's inherent to what's being
  tested (the opt-in NNUE path itself, not the default pipeline).
- **`pixelflow-pipeline/src/training/corpus.rs:469,484`** (2 tests calling
  private `read_corpus_bytes` instead of public `read_corpus(path)`) —
  `read_corpus(path)` is a two-line wrapper (`std::fs::read(path)?;
  read_corpus_bytes(&data)`); splitting I/O from parsing and unit-testing the
  parser on in-memory byte slices instead of writing temp files for every
  malformed-input case is standard, good practice, not a rule violation.

`pixelflow-core/tests/test_log2.rs`'s `unsafe { *(&result as *const Field as
*const f32) }` (flagged by the pixelflow-core audit) was investigated and
left as-is for a different reason: it's an integration test (separate crate,
external to `pixelflow-core`), so it cannot see the crate's `pub(crate)
Field::store()` at all. Checked whether `materialize()` (the crate's
documented public value-extraction entry point) could replace it — no,
`materialize<M,V>` requires `V: ops::Vector<Component = Field>` (RGBA-style
vector output), and this test evaluates scalar `Field`-returning manifolds.
**There is currently no public API for extracting a scalar f32 out of a
`Field` from outside the crate at all** — this is a real gap, not a test
smell, and not something to patch around unilaterally by changing public
API surface. Worth a human decision: either accept `unsafe` pointer-cast as
the sanctioned pattern for scalar-output integration tests, or add a small
`pub fn eval_scalar`-style test utility.

## Static audit: consolidated findings (still open, not fixed this pass)

Full per-file detail lives in the 4 sub-agent reports (not checked in —
available in this session's transcript if needed); summarized here.

### pixelflow-graphics — `spatial_bsp.rs` (same finding as Jul-20, still open)
~22 of ~55 tests index the private `interiors: Arc<[InteriorNode]>` field
directly (`bsp.interiors[0].axis`/`.threshold`) to assert on internal
tree-routing structure; only `interior_count()`/`leaf_count()` are public.
`subdiv/mod.rs:848` (`tile_scale_values`) calls private `tile_scale()`
directly — this one looks safely deletable, already covered by the public
`eigen_patch()` test. Unchanged from Jul-20's assessment: needs a design
call (test-only introspection API vs. property-testing `eval()` output vs.
documented rule-break), not a mechanical fix.

### actor-scheduler — `kubelet.rs` and `backoff_unit_tests` (same as Jul-20, still open)
- `kubelet.rs:772-776,869-881` read private field `poll_interval`;
  `:885-915` reads private field `pods`; `:839-857` constructs the private
  `ManagedPod` and calls private `within_budget()` with no public entry
  point at all (only reachable through `Kubelet::run()`'s real control
  loop). Same recommendation as Jul-20: add public accessors if this state
  should be observable, or extract `within_budget`'s restart-frequency-gate
  logic onto an independently-testable public type.
- `mod backoff_unit_tests` (lib.rs:1547-1689) still calls private
  `backoff_with_jitter`/`send_with_backoff` directly rather than through
  `ActorHandle::send`. Its mutation coverage of `backoff_with_jitter`
  specifically is now good (comments show deliberate mutant-killing tests
  already added between Jul-20 and now); the style violation itself
  (private-fn calls) is unresolved.

### core-term (Jul-20's audit was incomplete; now finished)
Confirms and completes Jul-20's partial finding. 7 violations across 6
files, 11/18 files clean:
- `term/unicode.rs:211-217` — asserts on private `GLOBAL_LOCALE_INITIALIZER` static.
- `term/screen.rs:1165-1185` — `create_test_screen_with_scrollback` writes directly to private `scrollback_limit` field (no public setter exists; the code's own inline comments show this was already a known compromise).
- `term/emulator/key_translator.rs:159-313` (7 tests) — call `pub(super) translate_key_input()` instead of the public `TerminalEmulator::interpret_input`.
- `term/emulator/input_handler.rs:198-380` (6 tests) — call private dispatchers `process_user_input_action`/`process_control_event`; **the code already has a `TODO: Refactor to use the true public API` comment acknowledging this** (line 212-213).
- `terminal_app.rs:967-976,1113-1118` — `create_test_app()` constructs the private `TerminalAppParamsRegistered` and calls private `new_registered()` (doc comment: "internal - use spawn_terminal_app instead"). Investigated: this one is defensible — `new_registered` gives synchronous, in-process construction needed to call `handle_control`/inspect `app.emulator` directly, which the actor-spawning `spawn_terminal_app` doesn't support. Likely not worth "fixing" without changing what the tests need to observe.
- `io/event_monitor_actor/writer.rs:211-308` — constructs `PtyWriter` via private fields, bypassing the real actor lifecycle; the code self-annotates as "bypassing the troupe".
- Confirmed clean (new since Jul-20): `term/emulator/mouse.rs`, `term/layout.rs`, `ansi/*`, `io/event_monitor_actor/mod.rs`, `io/pty_tests.rs`, `io/waker.rs`, `keys.rs`, `surface/manifold.rs`, `term/core_tests.rs`, `term/tests.rs`.

### pixelflow-ir (minor, not previously itemized this precisely)
- `backend/emit/mod.rs:4093-4493` (6 call sites, 4 tests) — call private
  `arena_to_schedule`/`needs_arena`/`arena_to_uses` directly; already
  covered indirectly by the public `compile_arena_dag*` tests later in the
  same file — candidate for deletion rather than fixing.
- `backend/emit/aarch64.rs:2736,2747` — `emit_ushr`/`emit_shl` are called
  directly by tests despite every sibling encoder function in the file being
  `pub`; looks like a visibility oversight, not deliberate — a `pub`
  add here (not attempted, needs the crate owner's sign-off per CLAUDE.md's
  visibility rule) would resolve it cleanly.

### pixelflow-pipeline (minor)
`training/unified_backward.rs:1780` calls private one-line `sigmoid()` to
compute an expected value — trivial, low severity, easy fix (inline the
formula in the test) but not done this pass.

### pixelflow-runtime (same as Jul-20 — `window_creation` already fixed then)
`window.rs:356-418`'s `metal_layer_config`/`resize_state` still reach
`pub(crate)` fields for raw Metal/NSWindow inspection; `resize_state` has an
inline comment explaining why it checks OS ground truth over the tracked
`poll_resize()`. Left as-is, same as Jul-20 — justified per STYLE.md's
Flexibility clause.

### Compliant, no violations found
pixelflow-search (all internals genuinely `pub` — tooling crate, not an
app with a narrow façade), pixelflow-ml, actor-scheduler's
sharded/registry/spsc/lifecycle/params/service modules, pixelflow-compiler's
lexer/parser/annotate/sema/symbol/codegen (only `optimize.rs` flagged, and
that flag was a false positive — see above), the bulk of pixelflow-core and
pixelflow-ir, the bulk of pixelflow-graphics.

## Mutation testing: actor-scheduler, targeted re-run (33 mutants, 3 min)

Scoped to exactly the functions Jul-20 flagged as still weak:
`send_with_backoff`, `drain_all_with_timeout`, `handle_wake`.
**Before this pass: 15 missed / 14 caught / 2 unviable / 2 timeout.**

Fixed (5 mutants, 3 new tests — see commits):
- `drain_all_with_timeout`'s `all_done` computation (lib.rs:915-916): a
  `delete !` mutant on either the control-term or management-term negation
  survived because every existing test kept all three lanes (control/mgmt/
  data) proportional, so a spuriously-early `all_done=true` never diverged
  from the correct answer within the test's queued message counts. Added
  `drain_all_requires_control_lane_fully_drained` and
  `_management_lane_fully_drained`, which flood a single lane past the
  batch size while leaving the others empty — this makes a wrong-negation
  `all_done` observable as silently-dropped queued messages.
- `handle_wake`'s `half_control` calculation (lib.rs:940, `control_burst_limit
  / 2`): no test distinguished `/2` from `%2` or `*2` because no test queued
  enough control messages to make the split's midpoint observable. Added
  `handle_wake_splits_control_burst_in_half_before_management`, which queues
  exactly `control_burst_limit` control messages plus one management message
  and asserts the management message's position in a shared log matches the
  exact index the `/2` split predicts (index 5, for burst limit 10).

Verified via a scoped `cargo-mutants -F` re-run isolated to these mutants:
5/5 now caught.

**Still missing (10 mutants), deliberately not attempted this pass:**
- `send_with_backoff`'s spin→yield→sleep phase-transition arithmetic
  (lib.rs:710,713,726: `<` vs `<=`/`==`, `+` vs `*`, `-` vs `+` on the
  attempt-count thresholds and `sleep_attempt` calculation). These are
  killable in principle via wall-clock assertions (e.g., asserting a
  permanently-full channel times out only after accumulating a minimum real
  sleep duration, which a wrong `sleep_attempt` calculation would shortcut
  to near-zero) — but timing-based assertions on a concurrent retry loop
  are a real flakiness risk on a loaded CI runner, and designing bounds
  wide enough to be robust while still tight enough to catch the mutants
  needs more care than a single automated pass should spend unsupervised.
  Two of these mutants (710, 713 with `<`→`>`) already show up as `TIMEOUT`
  rather than `MISSED` in the mutants output — meaning a wrong comparison
  there causes an actual hang, which is itself worth a maintainer's
  attention independent of test coverage.
- `handle_wake`'s `more_work`/`status` boolean chain (lib.rs:973-975,985:
  three `||`→`&&` sites in the 4-term `more_work` disjunction, plus one
  `||`→`&&` and one `==`→`!=` in the final `status` computation). Killing
  each `||` site independently requires precisely engineering which of the
  four lanes (`control1`/`mgmt`/`control2`/`data`) reports `More` in
  isolation at the right point in a single `handle_wake` call — control1 and
  control2 in particular are coupled (both drain the same control queue in
  one call), making "control1 More, control2 not-More" a specific queue-size
  needle to thread. Attempted the `half_control` case above precisely
  because its target was unambiguous; these four are a similar shape of
  problem but higher-risk to get subtly wrong under time pressure.

## Recommended next steps (carried over / updated)

1. `spatial_bsp.rs` and `kubelet.rs` design calls — unchanged from Jul-20,
   still the two largest structural findings.
2. Finish killing `handle_wake`'s remaining `||`/`==` mutants and
   `send_with_backoff`'s phase-transition arithmetic — needs either careful
   queue-size engineering (handle_wake) or a maintainer decision on whether
   timing-based assertions are acceptable here (send_with_backoff). The two
   comparison-direction mutants that currently `TIMEOUT` instead of `MISSED`
   (lib.rs:710,713) are worth investigating on their own even outside a
   test-coverage lens — a wrong comparison there hangs the retry loop.
3. `pixelflow-ir/src/backend/emit/aarch64.rs:551,560` — make `emit_ushr`/
   `emit_shl` `pub fn` to match every sibling encoder (needs sign-off per
   CLAUDE.md's visibility-change rule, but looks like a straightforward
   oversight fix).
4. Decide `pixelflow-core/tests/test_log2.rs`'s scalar-extraction gap: is
   `unsafe` pointer-cast the accepted pattern for scalar-output integration
   tests, or should a small public test-utility (`eval_scalar`-style) be
   added?
5. `pixelflow-pipeline/src/training/unified_backward.rs:1780` — trivial fix,
   inline the sigmoid formula instead of calling the private helper.
6. Scalar-fallback (`f32: Transcendental`, `Dual<N, A: Transcendental>`) CI
   leg — still no signal from any x86_64/aarch64 test or mutation run, per
   Jul-20. Unaddressed; would need its own CI target (e.g. wasm32) to get
   any coverage signal at all.
