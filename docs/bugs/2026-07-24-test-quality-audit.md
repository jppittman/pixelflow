# Test quality control pass — 2026-07-24

Follow-up to docs/bugs/2026-07-20-test-quality-audit.md's "Recommended next
steps." Scope this round: (1) finish the core-term static audit that a prior
sub-agent didn't complete in time (5 files); (2) close out actor-scheduler's
`handle_wake`/`drain_all_with_timeout` mutation-testing gaps flagged in the
2026-07-20 pass, since `backoff_unit_tests`'s private-API violation was
already investigated then and found impractical to fix without exposing new
test-only public API (see below).

## Fixed and pushed (branch `claude/nifty-maxwell-jamwww`)

1. **`core-term/src/term/emulator/input_handler.rs`**: all 6 tests called the
   `pub(super)` `process_user_input_action`/`process_control_event` directly.
   Both are 1:1 wrappers of `TerminalEmulator::interpret_input` (the crate's
   real public entry point — see `mod.rs:138`), so this was a mechanical,
   zero-risk fix: route every call through
   `emu.interpret_input(EmulatorInput::User(..)/::Control(..))` instead.
2. **`actor-scheduler/src/lib.rs`**: added a `handle_wake_targeted_tests`
   module (4 new tests) and 2 new tests in `drain_all_targeted_tests`,
   entirely through the public `ActorScheduler`/`ActorHandle` surface
   (`new_with_params`, `send`, `poll_once`, `run`) — no private-function
   calls. Verified against `cargo-mutants` scoped to `handle_wake` +
   `drain_all_with_timeout` (20 mutants): **9 missed → 2 missed** (was 45
   missed/108 caught workspace-wide in 2026-07-20's actor-scheduler run, this
   pass only re-scoped the two functions the 07-20 report specifically
   called out). New tests, what they kill, and the technique:
   - `more_work_is_true_when_only_first_control_pass_hits_burst_limit` —
     kills the `||`→`&&` mutant on `more_work`'s first operator (was missed).
     Uses `poll_once` + an actor whose `park` records the `SystemStatus` it
     receives — `park`'s argument is an observable proxy for the otherwise
     private `more_work` boolean, so no threading/timing needed.
   - `half_control_division_not_multiplication` /
     `half_control_division_not_modulo` — kill the `/`→`*` and `/`→`%`
     mutants on `half_control = (control_burst_limit / 2).max(1)` (both
     missed). Same `park`-recording technique, two message counts (9 and 5
     against a burst limit of 16) chosen so the correct division and each
     wrong operator diverge in whether `control1` reports `More`.
   - `busy_park_hint_forces_one_extra_wake_before_blocking` — kills the
     `||`→`&&` and `==`→`!=` mutants on
     `more_work || returned_hint == ActorStatus::Busy` (both missed). Unlike
     `more_work`, this status is never returned to a caller — its only
     observable effect is whether `run_inner` blocks on the next doorbell
     `recv()` or loops immediately. Made deterministic (no busy-spin risk)
     by having `park` return `Busy` on its first call only and `Idle`
     forever after: correct code calls `park` exactly twice before
     blocking; either mutant calls it once. Bounded assertion, no magnitude
     thresholds.
   - `drain_all_finishes_control_only_backlog_past_first_batch` /
     `drain_all_finishes_management_only_backlog_past_first_batch` — kill
     the `delete !` mutants on the control and mgmt terms of
     `all_done = !More(control) && !More(mgmt) && !More(data)` (both
     missed). Queue `Shutdown` before the scheduler starts, then 25
     control-only (or mgmt-only) messages against the default batch size of
     10; asserts the full 25 get processed rather than just the first
     batch.

## Investigated, judged not worth pursuing further

- **`handle_wake`'s remaining 2 missed mutants** (`||`→`&&` on the 2nd and
  3rd operators joining `more_work`'s four `matches!(..., More)` terms).
  Discovered empirically (by hand-mutating and re-running under `--nocapture`
  with a debug print) that `&&` binds tighter than `||` in Rust: replacing
  one `||` in a `t1||t2||t3||t4` chain doesn't just flip that connective in
  a flat left-to-right tree — it *reshapes the grouping*, e.g. `t1 || t2 &&
  t3 || t4` parses as `t1 || (t2 && t3) || t4`, not `(t1 || t2) && t3 ||
  t4`. Killing these precisely needs the second (`mgmt`) or third
  (`control2`) term to independently be `More` while the *other* three terms
  are not — and `mgmt`/`data` can never report `More` from a single upfront
  burst (their burst limit is `buffer_size * multiplier`, multiplier ≥ 1, so
  it can never be smaller than the channel's own capacity; genuinely hitting
  it needs either a concurrently-draining producer or multiple shards
  splitting the per-shard budget). Deferred as a real but low-severity gap:
  both are defensive terms in a 4-way OR already covered by the 1st/4th
  terms and the two now-fixed `half_control`/`more_work`-first-term cases;
  chasing them further means multi-producer sharding tests whose complexity
  is disproportionate to what's a purely cosmetic mis-grouping bug class.
- **`actor-scheduler`'s `backoff_unit_tests`** (flagged in 2026-07-20 as
  both a private-API violation and mutation-weak): confirmed `send_with_backoff`
  and `backoff_with_jitter` *are* reachable through the public
  `ActorHandle::send`/`ActorBuilder::new_with_params`, but only by filling a
  bounded channel to force the retry path — and the precise Ok/Timeout
  boundary cases the existing tests check (`backoff == max` vs `> max`, the
  attempt=0 case) require observing a specific *unstarted* backoff attempt,
  which isn't reachable without racing a real sleep/timeout through the
  public surface. Left as-is per docs/STYLE.md's Flexibility clause — this
  is timing-sensitive internal arithmetic where a private-function unit test
  is more honest than a flaky public-API timing test would be.
- **`mouse.rs`/`key_translator.rs`** (core-term): both construct
  `DecPrivateModes` via struct literal and call `pub(crate)`
  `encode_mouse_event`/`pub(super) translate_key_input` directly. A public
  equivalent exists (`TerminalEmulator::encode_mouse_event`,
  `interpret_input` for key events), but reaching the same mode state
  through it requires driving the real ANSI parser (`CSI ?1006h` etc.)
  instead of a one-line struct literal — disproportionate setup for what
  are pure, deterministic byte-encoders. Judgment call, leaning toward
  Flexibility-clause acceptance, consistent with 2026-07-20's treatment of
  `pixelflow-runtime`'s `resize_state`/`metal_layer_config`.
- **`input_handler.rs`'s `bracketed_paste_mode` field read** (line 229) and
  **`unicode.rs`'s `GLOBAL_LOCALE_INITIALIZER.get().is_some()` assertion**:
  both read private state with no public accessor. Minor, would need new
  test-only public API to fix "the right way" — left as documented,
  low-severity gaps rather than bugs.
- **`layout.rs`**: NOT a violation — every `Layout` field is `pub`, so
  struct-literal construction in tests is itself valid public-API usage.

## Recommended next steps (not done here)
1. `spatial_bsp.rs` and `kubelet.rs`'s private-API test clusters
   (docs/bugs/2026-07-20-test-quality-audit.md items 1–2) are still open —
   neither was touched this round; both need a design call (test-only
   introspection API vs. property tests vs. documented rule-break) rather
   than a mechanical fix.
2. `ops/trig.rs`/`ops/compare.rs` correctness tests were completed
   2026-07-22 (commit `297e8d2`) — that item from 07-20's list is done.
3. If `handle_wake`'s last 2 mutants are ever worth closing, the path is
   multi-producer sharding (`ActorBuilder::add_producer` × 2 feeding the
   same mgmt or control lane) so a single drain call's `per_shard` budget
   splits below the total queued, rather than trying to hit a burst limit
   with a single producer.
