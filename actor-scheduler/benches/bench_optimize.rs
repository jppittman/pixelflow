//! Bayesian optimization of scheduler parameters.
//!
//! Measures a weighted combination of latency and throughput across all three
//! lanes, then uses a Gaussian Process surrogate with Expected Improvement
//! to search for strictly better configurations.
//!
//! The cost function encodes two kinds of knowledge:
//!
//! 1. **Measured metrics** — latency, throughput, fairness under synthetic load
//! 2. **Domain constraints** — invariants that follow from the system's role
//!    as a real-time terminal scheduler at 155 FPS
//!
//! The domain constraints use multiplicative penalties so the optimizer cannot
//! trade a constraint violation for a metric improvement. A configuration that
//! drops frames on first backoff will never beat one that doesn't, regardless
//! of how much throughput it gains.
//!
//! Run with: `cargo bench -p actor-scheduler --bench bench_optimize`

use actor_scheduler::{
    Actor, ActorBuilder, ActorScheduler, ActorStatus, HandlerError, HandlerResult, Message,
    SchedulerParams, SystemStatus,
};
use std::io::Write;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
    mpsc,
};
use std::thread;
use std::time::{Duration, Instant};

fn flush() {
    std::io::stdout().flush().unwrap();
}

// ═══════════════════════════════════════════════════════════════════════════
// Domain constraint penalties
// ═══════════════════════════════════════════════════════════════════════════
//
// These encode knowledge about the target system (a 155 FPS terminal emulator)
// that raw performance metrics cannot capture. Each penalty is >= 0, with 0
// meaning "no violation." They feed into a multiplicative cost scaling factor
// so the optimizer cannot compensate for a constraint violation with raw
// throughput gains.

/// Penalty for `min_backoff` exceeding the frame budget.
///
/// At 155 FPS, one frame ≈ 6.45ms. If the first backoff sleep exceeds this,
/// the sender drops at least one frame on the *first sign* of contention.
/// This is the most visible failure mode — a keystroke vanishes.
///
/// Returns 0.0 when min_backoff ≤ frame_budget, scales linearly above.
fn frame_budget_penalty(params: &SchedulerParams) -> f64 {
    const FRAME_BUDGET_US: f64 = 6_450.0; // 155 FPS
    let min_backoff_us = params.min_backoff.as_micros() as f64;
    let ratio = min_backoff_us / FRAME_BUDGET_US;
    if ratio <= 1.0 { 0.0 } else { ratio - 1.0 }
}

/// Penalty for the total degradation window implied by the exponential
/// backoff cascade from `min_backoff` to `max_backoff`.
///
/// # The cascade reasoning
///
/// Exponential backoff doubles each attempt: min, 2·min, 4·min, … 2^n·min.
/// The sender gives up (returns `SendError::Timeout`) when `2^n·min > max`.
///
/// If you observe a sender sleeping for duration `d`, the *previous* sleep
/// was `d/2`, and before that `d/4`, all the way back to `min_backoff`.
/// Each sleep happened because the channel was STILL full after waking.
///
/// The total wall-clock time the system has been continuously overloaded
/// before the sender gives up is the geometric sum:
///
///   total = min · (2^(n+1) − 1) ≈ 2 · max_backoff
///
/// # What the status quo ante tells you
///
/// Consider a sender at the max_backoff level (e.g., 28.5s). The previous
/// sleep was 14.2s. During those 14.2 seconds:
///
///   1. The **receiver** was running the whole time, draining as fast as it can
///   2. **All other senders** are in the same backoff cascade — they're sleeping
///      too, so incoming pressure is near zero
///   3. Despite minimal incoming load AND 14.2 seconds of continuous draining,
///      the channel was STILL full when this sender woke up
///
/// This is a proof by contradiction: a healthy receiver drains a channel of
/// N messages in microseconds. If it couldn't drain during 14+ seconds of
/// near-zero sender pressure, the receiver is either dead, deadlocked, or so
/// catastrophically behind that no amount of waiting will help.
///
/// Sleeping for 2× the previous duration is asking "will the fundamentally
/// broken thing fix itself if I just wait twice as long?" Each doubling has
/// strictly decreasing probability of success. `max_backoff` isn't "retry
/// patience" — it's "how long do you let a corpse lie before declaring death."
///
/// With the *original* params (1ms min, 5s max):
///   n = ⌊log₂(5_000_000/1_000)⌋ = 12 doublings
///   total ≈ 1ms · (2¹³ − 1) ≈ 8.2 seconds
///
/// Returns 0.0 when total ≤ threshold, scales linearly above.
fn degradation_window_penalty(params: &SchedulerParams) -> f64 {
    const MAX_ACCEPTABLE_DEGRADATION_S: f64 = 12.0;
    let min_us = params.min_backoff.as_micros() as f64;
    let max_us = params.max_backoff.as_micros() as f64;
    if min_us <= 0.0 || max_us <= 0.0 {
        return 5.0;
    }

    let n_doublings = (max_us / min_us).log2().ceil();
    let total_us = min_us * (2.0f64.powf(n_doublings + 1.0) - 1.0);
    let total_s = total_us / 1_000_000.0;

    let ratio = total_s / MAX_ACCEPTABLE_DEGRADATION_S;
    if ratio <= 1.0 { 0.0 } else { ratio - 1.0 }
}

/// Penalty for narrow jitter range (defeats thundering herd prevention).
///
/// Jitter exists to spread out concurrent retrying senders. If the jitter
/// window is only 6% (e.g., sleep = 45-51% of base), multiple senders that
/// backed off simultaneously will ALL retry at nearly the same instant,
/// creating a synchronized stampede that re-fills the channel immediately.
///
/// A minimum effective range of ~20% provides meaningful spread. Below that,
/// the penalty ramps linearly.
fn jitter_effectiveness_penalty(params: &SchedulerParams) -> f64 {
    const MIN_EFFECTIVE_RANGE_PCT: f64 = 20.0;
    let range = params.jitter_range_pct as f64;
    if range >= MIN_EFFECTIVE_RANGE_PCT {
        0.0
    } else {
        (MIN_EFFECTIVE_RANGE_PCT - range) / MIN_EFFECTIVE_RANGE_PCT
    }
}

/// Penalty for buffer sizes that delay backpressure detection.
///
/// The scheduler's backpressure signal is "channel full." With a buffer of 32,
/// 32 messages must accumulate before any sender sees contention. With 250,
/// 250 must accumulate — meaning the receiver is 250 messages behind before
/// the system even knows there's a problem.
///
/// For control messages at ~1μs processing time, 32 messages = 32μs detection
/// latency. 250 messages = 250μs. For a system that values fast failure
/// detection, smaller buffers are strictly better *at the margin* — the
/// penalty only applies when detection latency exceeds a threshold.
fn backpressure_delay_penalty(params: &SchedulerParams) -> f64 {
    // Generous threshold: up to 128 messages before penalty kicks in
    const BUFFER_SOFT_LIMIT: f64 = 128.0;
    let size = params.control_mgmt_buffer_size as f64;
    if size <= BUFFER_SOFT_LIMIT {
        0.0
    } else {
        (size - BUFFER_SOFT_LIMIT) / BUFFER_SOFT_LIMIT
    }
}

/// Compute the aggregate domain penalty multiplier.
///
/// Returns a value >= 1.0. The cost is: `raw_cost * penalty_multiplier`.
/// Each individual penalty adds to the multiplier, so a penalty of 1.0
/// doubles the cost, 2.0 triples it, etc. This makes it impossible for
/// raw metric improvements to compensate for constraint violations.
fn domain_penalty_multiplier(params: &SchedulerParams) -> f64 {
    let frame = frame_budget_penalty(params);
    let degradation = degradation_window_penalty(params);
    let jitter = jitter_effectiveness_penalty(params);
    let backpressure = backpressure_delay_penalty(params);

    // Weight the penalties by severity for the terminal use-case.
    // Frame drops are the most user-visible failure mode.
    1.0 + 2.0 * frame + 1.5 * degradation + 0.5 * jitter + 0.3 * backpressure
}

// ═══════════════════════════════════════════════════════════════════════════
// Metric weights
// ═══════════════════════════════════════════════════════════════════════════

struct CostWeights {
    control_latency: f64,
    management_latency: f64,
    data_throughput: f64,
    control_throughput: f64,
    mixed_throughput: f64,
    fairness: f64,
    latency_under_load: f64,
    burst_recovery: f64,
}

const WEIGHTS: CostWeights = CostWeights {
    control_latency: 0.15,
    management_latency: 0.05,
    data_throughput: 0.15,
    control_throughput: 0.10,
    mixed_throughput: 0.10,
    fairness: 0.15,
    latency_under_load: 0.20, // Most important: latency during real workloads
    burst_recovery: 0.10,
};

/// Raw measurements from a single evaluation.
#[derive(Debug, Clone)]
struct Measurements {
    control_latency_ns: f64,
    management_latency_ns: f64,
    data_throughput_msgs_per_sec: f64,
    control_throughput_msgs_per_sec: f64,
    mixed_throughput_msgs_per_sec: f64,
    fairness_ratio: f64,
    /// Control latency while data is continuously flowing (ns).
    latency_under_load_ns: f64,
    /// Time (ns) for control latency to return to baseline after a data burst.
    burst_recovery_ns: f64,
}

/// Evaluate a parameter configuration by running all benchmark scenarios.
fn evaluate(params: &SchedulerParams) -> Measurements {
    Measurements {
        control_latency_ns: measure_control_latency(params),
        management_latency_ns: measure_management_latency(params),
        data_throughput_msgs_per_sec: measure_data_throughput(params),
        control_throughput_msgs_per_sec: measure_control_throughput(params),
        mixed_throughput_msgs_per_sec: measure_mixed_throughput(params),
        fairness_ratio: measure_fairness_under_flood(params),
        latency_under_load_ns: measure_latency_under_load(params),
        burst_recovery_ns: measure_burst_recovery(params),
    }
}

/// Compute the scalar cost: `raw_metric_cost * domain_penalty_multiplier`.
fn cost(m: &Measurements, b: &Measurements, params: &SchedulerParams) -> f64 {
    // Raw metric ratios (lower is better for all)
    let ctrl_lat = m.control_latency_ns / b.control_latency_ns;
    let mgmt_lat = m.management_latency_ns / b.management_latency_ns;
    let data_tput = b.data_throughput_msgs_per_sec / m.data_throughput_msgs_per_sec.max(1.0);
    let ctrl_tput = b.control_throughput_msgs_per_sec / m.control_throughput_msgs_per_sec.max(1.0);
    let mixed_tput = b.mixed_throughput_msgs_per_sec / m.mixed_throughput_msgs_per_sec.max(1.0);
    let fairness = b.fairness_ratio / m.fairness_ratio.max(0.01);
    let lat_load = m.latency_under_load_ns / b.latency_under_load_ns.max(1.0);
    let burst_rec = m.burst_recovery_ns / b.burst_recovery_ns.max(1.0);

    let raw = WEIGHTS.control_latency * ctrl_lat
        + WEIGHTS.management_latency * mgmt_lat
        + WEIGHTS.data_throughput * data_tput
        + WEIGHTS.control_throughput * ctrl_tput
        + WEIGHTS.mixed_throughput * mixed_tput
        + WEIGHTS.fairness * fairness
        + WEIGHTS.latency_under_load * lat_load
        + WEIGHTS.burst_recovery * burst_rec;

    // Multiply by domain penalties — constraint violations cannot be offset
    // by metric improvements, only by fixing the violation.
    raw * domain_penalty_multiplier(params)
}

// ═══════════════════════════════════════════════════════════════════════════
// Measurement scenarios
// ═══════════════════════════════════════════════════════════════════════════

struct LatencyActor {
    response_tx: mpsc::Sender<()>,
}

impl Actor<(), (), ()> for LatencyActor {
    fn handle_data(&mut self, _: ()) -> HandlerResult {
        Ok(())
    }
    fn handle_control(&mut self, _: ()) -> HandlerResult {
        self.response_tx.send(()).ok();
        Ok(())
    }
    fn handle_management(&mut self, _: ()) -> HandlerResult {
        self.response_tx.send(()).ok();
        Ok(())
    }
    fn park(&mut self, h: SystemStatus) -> Result<ActorStatus, HandlerError> {
        Ok(match h {
            SystemStatus::Idle => ActorStatus::Idle,
            SystemStatus::Busy => ActorStatus::Busy,
        })
    }
}

fn measure_control_latency(params: &SchedulerParams) -> f64 {
    let (response_tx, response_rx) = mpsc::channel();
    let (tx, mut rx) =
        ActorScheduler::new_with_params(params.default_data_burst_limit, 64, *params);
    let h = thread::spawn(move || {
        let mut a = LatencyActor { response_tx };
        rx.run(&mut a);
    });
    thread::sleep(Duration::from_millis(1));

    // Warmup
    for _ in 0..10 {
        tx.send(Message::Control(())).unwrap();
        response_rx.recv().unwrap();
    }

    let rounds = 50;
    let mut lats = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let t = Instant::now();
        tx.send(Message::Control(())).unwrap();
        response_rx.recv().unwrap();
        lats.push(t.elapsed().as_nanos() as f64);
    }
    tx.send(Message::Shutdown).unwrap();
    h.join().unwrap();
    lats.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    lats[lats.len() / 2]
}

fn measure_management_latency(params: &SchedulerParams) -> f64 {
    let (response_tx, response_rx) = mpsc::channel();
    let (tx, mut rx) =
        ActorScheduler::new_with_params(params.default_data_burst_limit, 64, *params);
    let h = thread::spawn(move || {
        let mut a = LatencyActor { response_tx };
        rx.run(&mut a);
    });
    thread::sleep(Duration::from_millis(1));

    for _ in 0..10 {
        tx.send(Message::Management(())).unwrap();
        response_rx.recv().unwrap();
    }

    let rounds = 50;
    let mut lats = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let t = Instant::now();
        tx.send(Message::Management(())).unwrap();
        response_rx.recv().unwrap();
        lats.push(t.elapsed().as_nanos() as f64);
    }
    tx.send(Message::Shutdown).unwrap();
    h.join().unwrap();
    lats.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    lats[lats.len() / 2]
}

struct CountingActor {
    data_count: Arc<AtomicUsize>,
    control_count: Arc<AtomicUsize>,
    mgmt_count: Arc<AtomicUsize>,
}

impl Actor<i32, (), ()> for CountingActor {
    fn handle_data(&mut self, _: i32) -> HandlerResult {
        self.data_count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
    fn handle_control(&mut self, _: ()) -> HandlerResult {
        self.control_count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
    fn handle_management(&mut self, _: ()) -> HandlerResult {
        self.mgmt_count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
    fn park(&mut self, h: SystemStatus) -> Result<ActorStatus, HandlerError> {
        Ok(match h {
            SystemStatus::Idle => ActorStatus::Idle,
            SystemStatus::Busy => ActorStatus::Busy,
        })
    }
}

fn new_counting() -> (
    Arc<AtomicUsize>,
    Arc<AtomicUsize>,
    Arc<AtomicUsize>,
    CountingActor,
) {
    let d = Arc::new(AtomicUsize::new(0));
    let c = Arc::new(AtomicUsize::new(0));
    let m = Arc::new(AtomicUsize::new(0));
    let actor = CountingActor {
        data_count: d.clone(),
        control_count: c.clone(),
        mgmt_count: m.clone(),
    };
    (d, c, m, actor)
}

fn measure_data_throughput(params: &SchedulerParams) -> f64 {
    let n = 5_000;
    let (dc, _, _, mut actor) = new_counting();
    let (tx, mut rx) =
        ActorScheduler::new_with_params(params.default_data_burst_limit, 512, *params);
    let h = thread::spawn(move || rx.run(&mut actor));
    let t = Instant::now();
    for i in 0..n {
        tx.send(Message::Data(i)).unwrap();
    }
    tx.send(Message::Shutdown).unwrap();
    h.join().unwrap();
    dc.load(Ordering::Relaxed) as f64 / t.elapsed().as_secs_f64()
}

fn measure_control_throughput(params: &SchedulerParams) -> f64 {
    let n = 2_000;
    let (_, cc, _, mut actor) = new_counting();
    let (tx, mut rx) =
        ActorScheduler::new_with_params(params.default_data_burst_limit, 64, *params);
    let h = thread::spawn(move || rx.run(&mut actor));
    let t = Instant::now();
    for _ in 0..n {
        tx.send(Message::Control(())).unwrap();
    }
    tx.send(Message::Shutdown).unwrap();
    h.join().unwrap();
    cc.load(Ordering::Relaxed) as f64 / t.elapsed().as_secs_f64()
}

fn measure_mixed_throughput(params: &SchedulerParams) -> f64 {
    let per = 1_500;
    let (dc, cc, mc, mut actor) = new_counting();
    let (tx, mut rx) =
        ActorScheduler::new_with_params(params.default_data_burst_limit, 512, *params);
    let h = thread::spawn(move || rx.run(&mut actor));
    let t = Instant::now();
    for i in 0..per {
        tx.send(Message::Data(i)).unwrap();
        tx.send(Message::Control(())).unwrap();
        tx.send(Message::Management(())).unwrap();
    }
    tx.send(Message::Shutdown).unwrap();
    h.join().unwrap();
    let total =
        dc.load(Ordering::Relaxed) + cc.load(Ordering::Relaxed) + mc.load(Ordering::Relaxed);
    total as f64 / t.elapsed().as_secs_f64()
}

fn measure_fairness_under_flood(params: &SchedulerParams) -> f64 {
    let data_target = 100i32;
    let (dc, _, _, mut actor) = new_counting();
    let stop = Arc::new(AtomicBool::new(false));
    let mut builder = ActorBuilder::<i32, (), ()>::new(128, None);
    let tx = builder.add_producer();
    let tx_f = builder.add_producer();
    let mut rx = builder.build_with_burst(
        params.default_data_burst_limit,
        actor_scheduler::ShutdownMode::default(),
    );
    let h = thread::spawn(move || rx.run(&mut actor));

    let sf = stop.clone();
    let flooder = thread::spawn(move || {
        while !sf.load(Ordering::Relaxed) {
            tx_f.send(Message::Control(())).ok();
        }
    });

    thread::sleep(Duration::from_millis(2));
    for i in 0..data_target {
        tx.send(Message::Data(i)).ok();
    }
    thread::sleep(Duration::from_millis(15));

    stop.store(true, Ordering::Relaxed);
    flooder.join().unwrap();
    thread::sleep(Duration::from_millis(5));
    let processed = dc.load(Ordering::Relaxed);
    drop(tx);
    h.join().unwrap();
    processed as f64 / data_target as f64
}

/// Measure control latency while data is continuously flowing.
///
/// This simulates the realistic terminal scenario: PTY output is streaming
/// (data lane) while the user types (control lane). The scheduler must
/// deliver control messages promptly despite ongoing data pressure.
fn measure_latency_under_load(params: &SchedulerParams) -> f64 {
    let (response_tx, response_rx) = mpsc::channel();
    let data_count = Arc::new(AtomicUsize::new(0));
    let dc = data_count.clone();
    let stop = Arc::new(AtomicBool::new(false));
    let sf = stop.clone();

    // Actor that responds to control and counts data
    struct LoadedLatencyActor {
        response_tx: mpsc::Sender<()>,
        data_count: Arc<AtomicUsize>,
    }
    impl Actor<i32, (), ()> for LoadedLatencyActor {
        fn handle_data(&mut self, _: i32) -> HandlerResult {
            self.data_count.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
        fn handle_control(&mut self, _: ()) -> HandlerResult {
            self.response_tx.send(()).ok();
            Ok(())
        }
        fn handle_management(&mut self, _: ()) -> HandlerResult {
            Ok(())
        }
        fn park(&mut self, h: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(match h {
                SystemStatus::Idle => ActorStatus::Idle,
                SystemStatus::Busy => ActorStatus::Busy,
            })
        }
    }

    let mut builder = ActorBuilder::<i32, (), ()>::new(256, None);
    let tx = builder.add_producer();
    let tx_data = builder.add_producer();
    let mut rx = builder.build_with_burst(
        params.default_data_burst_limit,
        actor_scheduler::ShutdownMode::default(),
    );
    let h = thread::spawn(move || {
        let mut a = LoadedLatencyActor {
            response_tx,
            data_count: dc,
        };
        rx.run(&mut a);
    });

    // Start continuous data sender
    let data_sender = thread::spawn(move || {
        let mut i = 0i32;
        while !sf.load(Ordering::Relaxed) {
            tx_data.send(Message::Data(i)).ok();
            i = i.wrapping_add(1);
        }
    });

    // Let data flow stabilize
    thread::sleep(Duration::from_millis(5));

    // Warmup
    for _ in 0..5 {
        tx.send(Message::Control(())).unwrap();
        response_rx.recv().unwrap();
    }

    // Measure control latency under data load
    let rounds = 30;
    let mut lats = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let t = Instant::now();
        tx.send(Message::Control(())).unwrap();
        response_rx.recv().unwrap();
        lats.push(t.elapsed().as_nanos() as f64);
        // Small gap between probes so we're not flooding control too
        thread::sleep(Duration::from_micros(100));
    }

    // Teardown
    stop.store(true, Ordering::Relaxed);
    data_sender.join().unwrap();
    tx.send(Message::Shutdown).unwrap();
    h.join().unwrap();

    lats.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    lats[lats.len() / 2]
}

/// Measure burst recovery: how quickly control latency returns to baseline
/// after a large data burst.
///
/// Simulates: `ls` dumps a screenful of output (data burst), then user
/// presses a key (control). How long until the key is processed?
fn measure_burst_recovery(params: &SchedulerParams) -> f64 {
    let (response_tx, response_rx) = mpsc::channel();

    struct RecoveryActor {
        response_tx: mpsc::Sender<()>,
    }
    impl Actor<i32, (), ()> for RecoveryActor {
        fn handle_data(&mut self, _: i32) -> HandlerResult {
            Ok(())
        }
        fn handle_control(&mut self, _: ()) -> HandlerResult {
            self.response_tx.send(()).ok();
            Ok(())
        }
        fn handle_management(&mut self, _: ()) -> HandlerResult {
            Ok(())
        }
        fn park(&mut self, h: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(match h {
                SystemStatus::Idle => ActorStatus::Idle,
                SystemStatus::Busy => ActorStatus::Busy,
            })
        }
    }

    let (tx, mut rx) =
        ActorScheduler::new_with_params(params.default_data_burst_limit, 512, *params);
    let h = thread::spawn(move || {
        let mut a = RecoveryActor { response_tx };
        rx.run(&mut a);
    });
    thread::sleep(Duration::from_millis(1));

    // Establish baseline
    for _ in 0..10 {
        tx.send(Message::Control(())).unwrap();
        response_rx.recv().unwrap();
    }

    // Measure: burst 2000 data messages, then immediately send a control
    // and time how long until it's processed.
    let trials = 10;
    let mut lats = Vec::with_capacity(trials);
    for trial in 0..trials {
        // Send burst
        for i in 0..2000 {
            tx.send(Message::Data(i)).unwrap();
        }
        // Immediately send control probe and time it
        let t = Instant::now();
        tx.send(Message::Control(())).unwrap();
        response_rx.recv().unwrap();
        lats.push(t.elapsed().as_nanos() as f64);

        // Let queue drain before next trial
        if trial < trials - 1 {
            thread::sleep(Duration::from_millis(10));
        }
    }

    tx.send(Message::Shutdown).unwrap();
    h.join().unwrap();

    lats.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    lats[lats.len() / 2]
}

// ═══════════════════════════════════════════════════════════════════════════
// Gaussian Process with RBF kernel
// ═══════════════════════════════════════════════════════════════════════════

const NDIM: usize = 10;

struct GaussianProcess {
    xs: Vec<[f64; NDIM]>,
    ys: Vec<f64>,
    length_scales: [f64; NDIM],
    signal_var: f64,
    noise_var: f64,
    chol: Vec<f64>,
    alpha: Vec<f64>,
}

impl GaussianProcess {
    fn new() -> Self {
        let bounds = SchedulerParams::bounds();
        let mut ls = [0.0; NDIM];
        for (i, (lo, hi)) in bounds.iter().enumerate() {
            ls[i] = (hi - lo) / 3.0;
        }
        Self {
            xs: Vec::new(),
            ys: Vec::new(),
            length_scales: ls,
            signal_var: 1.0,
            noise_var: 0.01,
            chol: Vec::new(),
            alpha: Vec::new(),
        }
    }

    fn kernel(&self, a: &[f64; NDIM], b: &[f64; NDIM]) -> f64 {
        let mut d = 0.0;
        for i in 0..NDIM {
            let x = (a[i] - b[i]) / self.length_scales[i];
            d += x * x;
        }
        self.signal_var * (-0.5 * d).exp()
    }

    fn observe(&mut self, x: [f64; NDIM], y: f64) {
        self.xs.push(x);
        self.ys.push(y);
        self.refit();
    }

    fn refit(&mut self) {
        let n = self.xs.len();
        let mut k = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..=i {
                let v = self.kernel(&self.xs[i], &self.xs[j]);
                k[i * n + j] = v;
                k[j * n + i] = v;
            }
            k[i * n + i] += self.noise_var;
        }

        let mut l = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..=i {
                let mut s = 0.0;
                for kk in 0..j {
                    s += l[i * n + kk] * l[j * n + kk];
                }
                l[i * n + j] = if i == j {
                    let d = k[i * n + i] - s;
                    if d > 0.0 { d.sqrt() } else { 1e-10 }
                } else {
                    (k[i * n + j] - s) / l[j * n + j]
                };
            }
        }

        let mut z = vec![0.0; n];
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..i {
                s += l[i * n + j] * z[j];
            }
            z[i] = (self.ys[i] - s) / l[i * n + i];
        }

        let mut alpha = vec![0.0; n];
        for i in (0..n).rev() {
            let mut s = 0.0;
            for j in (i + 1)..n {
                s += l[j * n + i] * alpha[j];
            }
            alpha[i] = (z[i] - s) / l[i * n + i];
        }

        self.chol = l;
        self.alpha = alpha;
    }

    fn predict(&self, x: &[f64; NDIM]) -> (f64, f64) {
        let n = self.xs.len();
        if n == 0 {
            return (0.0, self.signal_var);
        }

        let mut ks = vec![0.0; n];
        for i in 0..n {
            ks[i] = self.kernel(x, &self.xs[i]);
        }

        let mean: f64 = ks.iter().zip(self.alpha.iter()).map(|(a, b)| a * b).sum();

        let mut v = vec![0.0; n];
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..i {
                s += self.chol[i * n + j] * v[j];
            }
            v[i] = (ks[i] - s) / self.chol[i * n + i];
        }
        let vsq: f64 = v.iter().map(|vi| vi * vi).sum();
        (mean, (self.signal_var - vsq).max(1e-10))
    }
}

fn expected_improvement(mean: f64, var: f64, f_best: f64) -> f64 {
    let sigma = var.sqrt();
    if sigma < 1e-12 {
        return 0.0;
    }
    let z = (f_best - mean) / sigma;
    let phi = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let big_phi = 0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2));
    (f_best - mean) * big_phi + sigma * phi
}

fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    sign * (1.0 - poly * (-x * x).exp())
}

// ═══════════════════════════════════════════════════════════════════════════
// Deterministic xorshift64 PRNG
// ═══════════════════════════════════════════════════════════════════════════

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn uniform_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }
}

fn latin_hypercube(n: usize, rng: &mut Rng) -> Vec<[f64; NDIM]> {
    let bounds = SchedulerParams::bounds();
    let mut result = vec![[0.0; NDIM]; n];
    for dim in 0..NDIM {
        let (lo, hi) = bounds[dim];
        let mut perm: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = (rng.next_u64() as usize) % (i + 1);
            perm.swap(i, j);
        }
        let step = (hi - lo) / n as f64;
        for (i, &pi) in perm.iter().enumerate() {
            result[i][dim] = lo + step * (pi as f64 + rng.uniform());
        }
    }
    result
}

// ═══════════════════════════════════════════════════════════════════════════
// Optimizer
// ═══════════════════════════════════════════════════════════════════════════

fn run_optimization() {
    let initial_samples = 15;
    let bo_iterations = 25;
    let acq_candidates = 300;

    println!("=== Bayesian Scheduler Parameter Optimization ===");
    println!("    (with domain-constraint-aware cost function)\n");
    flush();

    // 1. Baseline
    println!("Measuring baseline (current defaults)...");
    flush();
    let bp = SchedulerParams::default();
    let baseline = evaluate(&bp);
    let bc = cost(&baseline, &baseline, &bp);
    let bp_penalty = domain_penalty_multiplier(&bp);
    println!("Baseline measurements:");
    print_measurements(&baseline);
    println!("Baseline domain penalty multiplier: {:.3}x", bp_penalty);
    println!("Baseline cost: {:.4} (normalized to {:.4})\n", bc, bc);
    flush();

    print_domain_analysis(&bp);

    let mut gp = GaussianProcess::new();
    let mut rng = Rng::new(42);
    let mut best_cost = bc;
    let mut best_params = bp;
    let mut best_meas = baseline.clone();

    gp.observe(bp.to_vec(), bc);

    // 2. LHS exploration
    println!("\nPhase 1: Latin Hypercube exploration ({initial_samples} samples)...");
    flush();
    let lhs = latin_hypercube(initial_samples, &mut rng);

    for (i, pt) in lhs.iter().enumerate() {
        let p = SchedulerParams::from_vec(pt);
        if p.min_backoff > p.max_backoff || p.jitter_min_pct + p.jitter_range_pct > 100 {
            continue;
        }
        let m = evaluate(&p);
        let c = cost(&m, &baseline, &p);
        let pen = domain_penalty_multiplier(&p);
        let tag = if c < best_cost { " *BEST*" } else { "" };
        println!(
            "  [{:2}/{initial_samples}] cost={c:.4} (penalty={pen:.2}x){tag}",
            i + 1
        );
        flush();
        if c < best_cost {
            best_cost = c;
            best_params = p;
            best_meas = m;
        }
        gp.observe(*pt, c);
    }

    println!("\nAfter exploration: best cost = {best_cost:.4}\n");
    flush();

    // 3. BO loop
    println!("Phase 2: Bayesian optimization ({bo_iterations} iterations)...");
    flush();
    let bounds = SchedulerParams::bounds();

    for iter in 0..bo_iterations {
        let f_best = *gp
            .ys
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let mut best_ei = -1.0f64;
        let mut best_cand = [0.0; NDIM];

        for _ in 0..acq_candidates {
            let mut cand = [0.0; NDIM];
            for d in 0..NDIM {
                cand[d] = rng.uniform_range(bounds[d].0, bounds[d].1);
            }
            let (mu, var) = gp.predict(&cand);
            let ei = expected_improvement(mu, var, f_best);
            if ei > best_ei {
                best_ei = ei;
                best_cand = cand;
            }
        }

        let p = SchedulerParams::from_vec(&best_cand);
        if p.min_backoff > p.max_backoff || p.jitter_min_pct + p.jitter_range_pct > 100 {
            gp.observe(best_cand, best_cost * 2.0);
            continue;
        }

        let m = evaluate(&p);
        let c = cost(&m, &baseline, &p);
        let pen = domain_penalty_multiplier(&p);
        let tag = if c < best_cost { " *BEST*" } else { "" };
        println!(
            "  BO [{:2}/{bo_iterations}] cost={c:.4} (penalty={pen:.2}x) EI={best_ei:.6}{tag}",
            iter + 1
        );
        flush();
        if c < best_cost {
            best_cost = c;
            best_params = p;
            best_meas = m;
        }
        gp.observe(best_cand, c);
    }

    // 4. Report
    println!("\n{}", "=".repeat(60));
    println!("                       RESULTS");
    println!("{}\n", "=".repeat(60));

    println!("Baseline cost:  {bc:.4}");
    println!("Optimized cost: {best_cost:.4}");
    println!("Improvement:    {:.1}%\n", (1.0 - best_cost / bc) * 100.0);

    println!("--- Baseline measurements ---");
    print_measurements(&baseline);
    println!("\n--- Optimized measurements ---");
    print_measurements(&best_meas);

    println!("\n--- Domain constraint analysis (optimized) ---");
    print_domain_analysis(&best_params);

    println!("\n--- Optimized parameters ---");
    let v = best_params.to_vec();
    let dv = bp.to_vec();
    for (i, name) in SchedulerParams::NAMES.iter().enumerate() {
        let pct = if dv[i] > 0.0 {
            ((v[i] - dv[i]) / dv[i]) * 100.0
        } else {
            0.0
        };
        println!(
            "  {name:20} = {:>12.1}  (was {:>10.1}, {pct:+.1}%)",
            v[i], dv[i]
        );
    }

    println!("\n--- Copy-paste for SchedulerParams::DEFAULT ---");
    println!("pub const DEFAULT: Self = Self {{");
    println!(
        "    control_mgmt_buffer_size: {},",
        best_params.control_mgmt_buffer_size
    );
    println!(
        "    control_burst_multiplier: {},",
        best_params.control_burst_multiplier
    );
    println!(
        "    management_burst_multiplier: {},",
        best_params.management_burst_multiplier
    );
    println!(
        "    default_data_burst_limit: {},",
        best_params.default_data_burst_limit
    );
    println!("    spin_attempts: {},", best_params.spin_attempts);
    println!("    yield_attempts: {},", best_params.yield_attempts);
    println!(
        "    min_backoff: Duration::from_micros({}),",
        best_params.min_backoff.as_micros()
    );
    println!(
        "    max_backoff: Duration::from_micros({}),",
        best_params.max_backoff.as_micros()
    );
    println!("    jitter_min_pct: {},", best_params.jitter_min_pct);
    println!("    jitter_range_pct: {},", best_params.jitter_range_pct);
    println!("}};");
    flush();
}

fn print_measurements(m: &Measurements) {
    println!("  Control latency:       {:>10.0} ns", m.control_latency_ns);
    println!(
        "  Management latency:    {:>10.0} ns",
        m.management_latency_ns
    );
    println!(
        "  Latency under load:    {:>10.0} ns",
        m.latency_under_load_ns
    );
    println!("  Burst recovery:        {:>10.0} ns", m.burst_recovery_ns);
    println!(
        "  Data throughput:       {:>10.0} msg/s",
        m.data_throughput_msgs_per_sec
    );
    println!(
        "  Control throughput:    {:>10.0} msg/s",
        m.control_throughput_msgs_per_sec
    );
    println!(
        "  Mixed throughput:      {:>10.0} msg/s",
        m.mixed_throughput_msgs_per_sec
    );
    println!(
        "  Fairness ratio:        {:>9.2}%",
        m.fairness_ratio * 100.0
    );
}

fn print_domain_analysis(params: &SchedulerParams) {
    let min_us = params.min_backoff.as_micros() as f64;
    let max_us = params.max_backoff.as_micros() as f64;

    // Backoff cascade analysis
    let n_doublings = if min_us > 0.0 {
        (max_us / min_us).log2().ceil() as u32
    } else {
        0
    };
    let total_degradation_us = if min_us > 0.0 {
        min_us * (2.0f64.powi(n_doublings as i32 + 1) - 1.0)
    } else {
        0.0
    };
    let total_degradation_s = total_degradation_us / 1_000_000.0;

    println!("  Backoff cascade:");
    println!(
        "    min_backoff:           {:>10.0} us ({:.1} ms)",
        min_us,
        min_us / 1000.0
    );
    println!(
        "    max_backoff:           {:>10.0} us ({:.1} s)",
        max_us,
        max_us / 1_000_000.0
    );
    println!("    doublings to timeout:  {:>10}", n_doublings);
    println!(
        "    total degradation:     {:>10.1} s (if sender hits timeout)",
        total_degradation_s
    );
    println!("    frames lost on 1st backoff: {:>5.1}", min_us / 6_450.0);

    // Jitter analysis
    let jitter_lo = params.jitter_min_pct;
    let jitter_hi = params.jitter_min_pct + params.jitter_range_pct;
    println!(
        "  Jitter window:           {}%-{}% of backoff",
        jitter_lo, jitter_hi
    );

    // Buffer/burst analysis
    let ctrl_burst = params.control_burst_limit();
    let mgmt_burst = params.management_burst_limit();
    println!(
        "  Control buffer:          {} slots, {} msgs/wake",
        params.control_mgmt_buffer_size, ctrl_burst
    );
    println!("  Management burst:        {} msgs/wake", mgmt_burst);
    println!(
        "  Data burst:              {} msgs/wake",
        params.default_data_burst_limit
    );

    // Penalty summary
    println!("  Penalties:");
    println!(
        "    frame_budget:          {:.3}",
        frame_budget_penalty(params)
    );
    println!(
        "    degradation_window:    {:.3}",
        degradation_window_penalty(params)
    );
    println!(
        "    jitter_effectiveness:  {:.3}",
        jitter_effectiveness_penalty(params)
    );
    println!(
        "    backpressure_delay:    {:.3}",
        backpressure_delay_penalty(params)
    );
    println!(
        "    TOTAL multiplier:      {:.3}x",
        domain_penalty_multiplier(params)
    );
}

fn main() {
    run_optimization();
}
