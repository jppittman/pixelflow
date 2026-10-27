# desloppify/src/rate_limit

**Is.** A function from the calls a client has made and the instants it asks
about them to a wait: before each call to a provider, how long it sleeps, or
that it does not go (desloppify/README.md, "Rate limiting"). Time is an
input. desloppify is not a pixelflow crate, so this cites its README, not
DENOTATION.

## Owns
- The pacing decision: `RateLimiter::wait` and the slot it books. Every `Ok`
  books a slot, so concurrent callers queue behind one another and are never
  told to go at once beyond the budget.
- The budget's two representations: a fixed one (`TokenBucketConfig`, a bucket
  that starts full) and a discovered one (`AdaptiveConfig`, AIMD on the call
  rate: halve on a throttle at most once per cooldown, grow while pacing binds,
  `Retry-After` as a floor on the wait). Only `Signal::Throttled` moves the
  rate; `Signal::Failed` is paced and changes nothing.
- Refusal: a wait past `max_wait` is `Err`: the caller's failure for a retry,
  `Exhausted` for a first attempt.

## Never
- No limiter reads the time except through the `Clock` it was built with:
  `Instant::now` appears once in this directory, in `SystemClock`.
- No limiter sleeps, spawns or does I/O; it returns a `Duration` and the
  caller waits (`agent/rig.rs` does).
- No knowledge of a provider: nothing here names rig, tokio, HTTP or a status
  code. Reading a failure as a `Signal` is the caller's `Classify`.
- The implementations are not nameable: `TokenBucket` and `Adaptive` are
  `pub(super)`; callers hold `impl RateLimiter` from `token_bucket()` and
  `adaptive()`.
- No unretryable failure is asked about: a 401 never reaches `wait`.

## Contract
`mod.rs` is the contract and holds no algorithm: `RateLimiter`, `Clock`,
`SystemClock`, `Signal`, `Classify`, `Exhausted`, `BoxError`,
`TokenBucketConfig`, `AdaptiveConfig`, and the constructors `token_bucket` and
`adaptive`. `token_bucket.rs` and `adaptive.rs` are the two impls.

## Enforced by
- Types: `Clock` is the only way an impl gets an instant, and the impls are
  private, so a limiter reading the wall clock or a caller reaching a field
  does not compile outside `rate_limit/`.
- `desloppify/tests/rate_limit.rs`: both limiters driven through `wait` with a
  manual `Clock`, one test per behavior (bucket queueing and refill, a refused
  retry returning its failure, a refused first attempt as `Exhausted`, one
  halving per cooldown, growth only while calls queue, floor, ceiling,
  `Retry-After`, an outage paced without a cut).
- desloppify `mod-rs-is-a-contract`, `interface-lives-in-mod-rs`,
  `behavior-through-the-trait`, `effects-are-returned`.

## Behind
- Nothing tests the concurrency line of the contract (slots booked in call
  order); `tests/rate_limit.rs` is single-threaded.
- `adaptive` asserts its config; `token_bucket` validates nothing, so a zero
  `refill` divides by zero in `wait`. No plan or Linear issue names either.

## Decided in
`desloppify/README.md` "Rate limiting"; introduced in #1319 (`f93dc7d9`). No
plan under `docs/plans/` names this module.
