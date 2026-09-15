//! Wall clock for collapse bodies at real shapes, beside the static features
//! of the code that produced it.
//!
//! # Why this exists
//!
//! Three register-allocator policies were built and rejected in September 2026
//! on quantities that turned out not to be time: static memory operations,
//! then dynamic memory operations per call, then code bytes
//! (`docs/plans/2026-09-01-register-allocation-escape-hatches.md`, the
//! 2026-09-04 blocks). The sharpest of them cut dynamic memory operations by
//! 6.6% on AVX-512 and ran **13–18% slower** on the glyph bench, outside the
//! run-to-run spread, while being flat-to-faster on SSE2. The conclusion
//! recorded there is that no further policy should be built until a static
//! cost model's prediction is shown to track wall clock across tiers.
//!
//! This module is the measurement that claim has to be checked against. It
//! takes a corpus of kernels **with the shapes they are baked at**, compiles
//! each one exactly as `Lattice::bake` would, times the emitted collapse
//! kernel, and records the static features of the emitted code — per scope,
//! so a trip count can weight them — as one row per (kernel, tier, git ref).
//! Five allocations of the same kernel set across two tiers is then a dataset
//! a predictor can be scored on: see [`predict`].
//!
//! # What is timed, and what is not
//!
//! One `call_collapse` into a buffer allocated once, per sample. Compilation
//! (including e-graph saturation) happens before the timer starts, and the
//! scalar tail `Lattice::bake` walks after the vector groups is not part of
//! the kernel. So the number is the emitted code's own cost at that shape —
//! which is the thing an allocator's cost model is supposed to predict, with
//! the per-bake allocation and tail arithmetic that would otherwise dilute it
//! held out.
//!
//! # Measurement discipline
//!
//! The same shape `jit_bench` uses, one level up: the process is pinned to a
//! core where the OS allows it, each sample repeats the call until it clears a
//! clock-granularity floor, the reported figure is a median with its
//! interquartile range beside it, and a fixed **sentinel** kernel is
//! re-measured through the run so slow clock drift becomes a correction
//! (`SentinelContext::normalization`) rather than a discarded session. Kernels
//! are visited in a fixed pseudo-random order so collection position does not
//! correlate with family, and rows are written in name order so two runs
//! diff.

pub mod corpus;
pub mod predict;
pub mod row;

use std::path::Path;
use std::sync::Arc;

use pixelflow_codegen::emit::executable::{ExecutableCode, Point4, TileSlice};
use pixelflow_codegen::emit::{CompileResult, compile};
use pixelflow_ir::LatticeShape;
use pixelflow_ir::arena::{ExprArena, ExprId};

use crate::jit_bench::{LocalNs, SentinelContext};
use corpus::{CollapseKernel, Trips};
pub use row::Stat;
use row::{Measurement, Row, ScopeRow, StaticFeatures};

/// Lanes in one batch for this build's vector width.
pub const LANES: usize = pixelflow_codegen::JIT_VECTOR_BYTES / 4;

/// Timed samples per kernel. The reported figure is the median; the brief's
/// floor is 7, and the extra samples cost microseconds.
const SAMPLES: usize = 15;

/// Untimed calls before the samples, to warm icache and the branch predictor.
const WARMUP_CALLS: usize = 32;

/// A sample must accumulate at least this long. Well past clock granularity
/// (`CLOCK_MONOTONIC_RAW` resolves nanoseconds) — the floor is set by
/// *dispersion*, not resolution: a sample averaging over more calls absorbs
/// the scheduler's interruptions instead of reporting them, and the
/// pass-to-pass spread of the whole corpus is what it was tuned against.
const MIN_SAMPLE_NS: u64 = 1_500_000;

/// Upper bound on that autoscale, so a pathologically cheap kernel cannot
/// spin. At 20ns a call this is still under a second per sample.
const MAX_CALLS_PER_SAMPLE: usize = 1 << 22;

/// How many kernels between sentinel re-measurements.
///
/// Small, and paired with [`SENTINEL_WINDOW`], because a drift correction is
/// itself a measurement: dividing 25 kernels by one sentinel reading injects
/// that reading's own error into all 25. Measured on this corpus, a coarse
/// sentinel made the corrected numbers *less* reproducible than the raw ones
/// (3.9% pass-to-pass against 2.5%), which is the correction doing harm.
const SENTINEL_INTERVAL: usize = 8;

/// Sentinel readings the drift factor is the median of. An odd window, so the
/// median is one reading rather than an average of two.
const SENTINEL_WINDOW: usize = 3;

/// The drift sentinel's shape — small, fixed, and unrelated to anything in the
/// corpus, so its cost moves only when the machine does.
const SENTINEL_EXTENT: [u32; 2] = [256, 32];

/// Which ISA level this binary was built for, read off the target features the
/// backend selection itself keys on.
#[must_use]
pub fn tier() -> &'static str {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
        "avx512"
    }
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        not(target_feature = "avx512f")
    ))]
    {
        "avx2"
    }
    #[cfg(all(
        target_arch = "x86_64",
        not(target_feature = "avx2"),
        not(target_feature = "avx512f")
    ))]
    {
        "sse2"
    }
    #[cfg(target_arch = "aarch64")]
    {
        "neon"
    }
}

/// Compile a corpus kernel the way `Lattice::bake` does — runtime
/// optimization at the kernel's own shape, then emit — and hand back the
/// emitter's report.
///
/// # Panics
/// If the kernel does not compile. A corpus entry that cannot be baked is a
/// corpus bug, and continuing past it would silently change which kernels the
/// two sides of a comparison share.
#[must_use]
pub fn compile_as_baked(arena: &ExprArena, root: ExprId, extent: [u32; 2]) -> CompileResult {
    let shape = LatticeShape::new(extent);
    let optimized = pixelflow_search::runtime::optimize_runtime_arena(arena, root, shape);
    let (arena, root) = optimized
        .as_deref()
        .map(|(a, r)| (a, *r))
        .unwrap_or((arena, root));
    compile(arena, root).expect("corpus kernel failed to compile")
}

/// A measurement run: owns the sentinel calibration and the output buffer.
pub struct CollapseSession {
    calibration_ns: f64,
    recent_sentinels: Vec<f64>,
    sentinel: Sentinel,
    since_sentinel: usize,
}

struct Sentinel {
    code: ExecutableCode,
    buffer: Vec<f32>,
    trips: Trips,
    bytes: u32,
    /// Kept so [`Sentinel::measure`] can bind its context the same way
    /// [`CollapseSession::measure`] binds any other kernel's — empty for
    /// this arena today, but a special-cased hardcoded slot count would be
    /// a second definition of the layout `compile_as_baked` already emits.
    arena: ExprArena,
}

impl CollapseSession {
    /// Open a session: pin the thread, build the sentinel, and take the
    /// opening calibration every later measurement is expressed in.
    ///
    /// # Panics
    /// If the sentinel kernel does not compile or measures zero.
    #[must_use]
    pub fn open() -> Self {
        pin_to_a_core();
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let y = arena.push_var(1);
        let xx = arena.push_binary(pixelflow_ir::OpKind::Mul, x, x);
        let yy = arena.push_binary(pixelflow_ir::OpKind::Mul, y, y);
        let sum = arena.push_binary(pixelflow_ir::OpKind::Add, xx, yy);
        let root = arena.push_unary(pixelflow_ir::OpKind::Sqrt, sum);
        let result = compile_as_baked(&arena, root, SENTINEL_EXTENT);
        let trips = Trips::of(SENTINEL_EXTENT, LANES as u32);
        let mut sentinel = Sentinel {
            bytes: result.code.len() as u32,
            code: result.code,
            buffer: output_buffer(trips),
            trips,
            arena,
        };
        // Burn in before calibrating: the first milliseconds of a process run
        // at whatever clock the machine happened to be at.
        for _ in 0..3 {
            sentinel.measure();
        }
        let calibration_ns = sentinel.measure();
        assert!(
            calibration_ns > 0.0,
            "sentinel calibrated at 0ns — the clock or the kernel is not what it claims"
        );
        Self {
            calibration_ns,
            recent_sentinels: vec![calibration_ns; SENTINEL_WINDOW],
            sentinel,
            since_sentinel: 0,
        }
    }

    /// The median of the last [`SENTINEL_WINDOW`] readings.
    fn local_sentinel_ns(&self) -> f64 {
        let mut window = self.recent_sentinels.clone();
        window.sort_by(f64::total_cmp);
        window[window.len() / 2]
    }

    /// The drift-correction context every label minted right now carries.
    #[must_use]
    pub fn context(&self) -> SentinelContext {
        SentinelContext {
            local_sentinel_ns: self.local_sentinel_ns(),
            calibration_ns: self.calibration_ns,
        }
    }

    fn maybe_resample_sentinel(&mut self) {
        if self.since_sentinel < SENTINEL_INTERVAL {
            self.since_sentinel += 1;
            return;
        }
        self.since_sentinel = 0;
        self.recent_sentinels.remove(0);
        self.recent_sentinels.push(self.sentinel.measure());
    }

    /// Compile and time one kernel, returning the row to write.
    ///
    /// # Panics
    /// If the kernel does not compile, or its extent is too narrow to fill a
    /// batch at this tier.
    pub fn measure(&mut self, kernel: &CollapseKernel, pass: u32) -> Row {
        self.maybe_resample_sentinel();
        let trips = Trips::of(kernel.extent, LANES as u32);
        let result = compile_as_baked(&kernel.arena, kernel.root, kernel.extent);
        let mut buffer = output_buffer(trips);
        let (buffers, uniforms) = dummy_context(&kernel.name, &kernel.arena, &kernel.buffer_data);
        let slots = context_slots(&buffers, &uniforms);
        let timing = time_kernel(&result.code, &mut buffer, trips, slots.as_ptr());
        let drift = self.context().normalization();
        Row {
            schema: row::SCHEMA.to_string(),
            git_ref: String::new(),
            git_sha: String::new(),
            profile: String::new(),
            tier: tier().to_string(),
            pass,
            kernel: kernel.name.clone(),
            family: kernel.family.clone(),
            extent: kernel.extent,
            lanes: LANES as u32,
            rows: trips.rows,
            groups: trips.groups,
            measured: Measurement {
                ns_median: timing.median,
                ns_min: timing.min,
                ns_iqr: timing.iqr,
                ns_median_drift_corrected: LocalNs::new(timing.median).normalize(drift).get(),
                drift: drift.get(),
                sentinel_calibration_ns: self.calibration_ns,
                sentinel_bytes: self.sentinel.bytes,
                samples: SAMPLES as u32,
                calls_per_sample: timing.calls as u64,
            },
            statics: features_of(&result, trips),
        }
    }
}

/// The static half of a row: the emitter's counts, plus what they derive.
#[must_use]
pub fn features_of(result: &CompileResult, trips: Trips) -> StaticFeatures {
    let t = &result.traffic;
    let scope = |s: &pixelflow_codegen::emit::traffic::ScopeTraffic| ScopeRow {
        bytes: s.bytes,
        instructions: s.instructions,
        loads_transient: s.loads_transient,
        loads_kept: s.loads_kept,
        remats: s.remats,
        stores: s.stores,
    };
    let (rows, groups) = (trips.rows, trips.groups);
    let body_trips = rows * groups;
    StaticFeatures {
        bytes_total: result.code.len() as u32,
        frame: scope(&t.frame),
        row: scope(&t.row),
        body: scope(&t.body),
        scaffold: scope(&t.scaffold),
        spill_slots: result.spill_count,
        frame_bytes: result.spill_bytes,
        hoisted: result.hoisted_values,
        carried: t.carried,
        pool: u32::from(t.pool),
        vector_bytes: t.vector_bytes,
        dyn_memory_ops: t.dynamic_memory_ops(rows, groups),
        dyn_instructions: u64::from(t.frame.instructions)
            + u64::from(t.row.instructions) * rows
            + u64::from(t.body.instructions) * body_trips,
        dyn_bytes: u64::from(t.frame.bytes)
            + u64::from(t.row.bytes) * rows
            + u64::from(t.body.bytes) * body_trips,
    }
}

fn output_buffer(trips: Trips) -> Vec<f32> {
    vec![0.0f32; (trips.rows * trips.groups) as usize * LANES]
}

/// Memory for every buffer slot `arena` declares, sized to the slot's own
/// extent and bound to `buffer_data`'s captured contents, plus the uniform
/// block (one `f32` per declared argument, at its default; a single
/// `CORPUS_ARG` slot when the arena declares none, so a kernel with no
/// argument still reads a real block rather than one omitted array entry
/// away from a null deref).
///
/// Collapse cost is *not* independent of a buffer's values, only of its
/// *identity*: `emit_skip_if_all_false`/`emit_skip_if_all_true`
/// (`pixelflow-codegen/src/emit/mod.rs`) branch at runtime on whether a
/// `Select` guard's mask has any lane set, and a zero-filled glyph piece
/// table makes every crossing-span mask uniformly false — a control-flow
/// path production never takes. So a slot binds `buffer_data`'s real
/// contents whenever capture provided them; a slot with none is the
/// exception, and falls back to zeros loudly — naming `kernel_name` and the
/// slot — rather than silently reintroducing the artifact this replaced.
fn dummy_context(
    kernel_name: &str,
    arena: &ExprArena,
    buffer_data: &[Option<Arc<Vec<f32>>>],
) -> (Vec<Vec<f32>>, Vec<f32>) {
    let buffers: Vec<Vec<f32>> = arena
        .buffers()
        .iter()
        .enumerate()
        .map(|(slot, decl)| {
            let expected = decl.width as usize * decl.height as usize;
            match buffer_data.get(slot).and_then(Option::as_ref) {
                Some(data) => {
                    assert_eq!(
                        data.len(),
                        expected,
                        "{kernel_name}: buffer slot {slot} captured {} value(s), the declared \
                         {}x{} extent wants {expected}",
                        data.len(),
                        decl.width,
                        decl.height
                    );
                    data.as_ref().clone()
                }
                None => {
                    eprintln!(
                        "collapse_bench: {kernel_name}: buffer slot {slot} ({}x{}) has no \
                         captured contents; binding zeros — this replay will not exercise the \
                         same guard decisions production's real data does",
                        decl.width, decl.height
                    );
                    vec![0.0f32; expected]
                }
            }
        })
        .collect();
    let uniforms: Vec<f32> = if arena.uniforms().is_empty() {
        vec![corpus::CORPUS_ARG]
    } else {
        arena.uniforms().iter().map(|u| u.default).collect()
    };
    (buffers, uniforms)
}

/// The context pointer table `dummy_context`'s memory is bound through: one
/// base pointer per buffer slot, then the uniform block's — exactly the
/// layout `compile_as_baked`'s emitted code reads (an `ExprNode::Uniform`'s
/// context slot is `arena.buffers().len()`, the entry right after the last
/// buffer). Borrows `buffers`/`uniforms`, so the returned pointers are valid
/// exactly as long as they are.
fn context_slots(buffers: &[Vec<f32>], uniforms: &[f32]) -> Vec<*const f32> {
    let mut slots: Vec<*const f32> = buffers.iter().map(Vec::as_ptr).collect();
    slots.push(uniforms.as_ptr());
    slots
}

/// What one kernel's timed samples came to, all ns per call.
struct Timing {
    median: f64,
    iqr: f64,
    min: f64,
    calls: usize,
}

fn time_kernel(
    code: &ExecutableCode,
    buffer: &mut [f32],
    trips: Trips,
    ctx: *const *const f32,
) -> Timing {
    let mut calls = 1usize;
    loop {
        let elapsed = run_calls(code, buffer, trips, WARMUP_CALLS.max(calls), ctx);
        if elapsed >= MIN_SAMPLE_NS || calls >= MAX_CALLS_PER_SAMPLE {
            break;
        }
        // Grow toward the floor, with a cap so one slow sample cannot
        // overshoot by orders of magnitude.
        let want = (MIN_SAMPLE_NS as f64 / elapsed.max(1) as f64).ceil() as usize;
        calls = (calls * want.clamp(2, 16)).min(MAX_CALLS_PER_SAMPLE);
    }

    let mut per_call: Vec<f64> = (0..SAMPLES)
        .map(|_| run_calls(code, buffer, trips, calls, ctx) as f64 / calls as f64)
        .collect();
    per_call.sort_by(f64::total_cmp);
    Timing {
        median: per_call[SAMPLES / 2],
        iqr: per_call[(SAMPLES * 3) / 4] - per_call[SAMPLES / 4],
        min: per_call[0],
        calls,
    }
}

fn run_calls(
    code: &ExecutableCode,
    buffer: &mut [f32],
    trips: Trips,
    calls: usize,
    ctx: *const *const f32,
) -> u64 {
    let mut x0 = [0.0f32; LANES];
    for (i, lane) in x0.iter_mut().enumerate() {
        *lane = 0.5 + i as f32;
    }
    let origin = Point4::new(x0, [0.5f32; LANES], [0.0f32; LANES], [0.0f32; LANES]);
    let tile = TileSlice::contiguous(
        buffer.as_mut_ptr(),
        trips.groups as usize,
        trips.rows as usize,
    );
    let start = crate::jit_bench::nanos_now();
    for _ in 0..calls {
        // SAFETY: the kernel was compiled for this shape, the tile is exactly
        // `rows × groups × LANES` floats of the buffer allocated for it, and
        // `ctx` points at a live table (built by `context_slots`) with one
        // base pointer per buffer slot the kernel declared plus the uniform
        // block, in the order the kernel was compiled expecting.
        unsafe {
            code.call_collapse(ctx, tile, origin);
        }
        std::hint::black_box(&buffer);
    }
    crate::jit_bench::nanos_now().saturating_sub(start)
}

impl Sentinel {
    fn measure(&mut self) -> f64 {
        // The sentinel arena declares no buffers, so there is nothing for
        // `buffer_data` to carry.
        let (buffers, uniforms) = dummy_context("sentinel", &self.arena, &[]);
        let slots = context_slots(&buffers, &uniforms);
        time_kernel(&self.code, &mut self.buffer, self.trips, slots.as_ptr()).median
    }
}

/// Keep the run on one core where the OS offers it. Migration between cores
/// is the drift the sentinel would otherwise have to absorb.
fn pin_to_a_core() {
    #[cfg(target_os = "linux")]
    {
        // SAFETY: `set` is a valid, zeroed cpu_set_t for the duration of both
        // calls, and pid 0 names the calling thread.
        unsafe {
            let mut set: libc::cpu_set_t = core::mem::zeroed();
            libc::CPU_ZERO(&mut set);
            libc::CPU_SET(0, &mut set);
            if libc::sched_setaffinity(0, core::mem::size_of::<libc::cpu_set_t>(), &set) != 0 {
                eprintln!(
                    "collapse_bench: could not pin to a core; the sentinel is the only \
                     defence against migration drift"
                );
            }
        }
    }
    #[cfg(not(target_os = "linux"))]
    eprintln!("collapse_bench: core pinning is Linux-only here; relying on the sentinel for drift");
}

/// Run the whole corpus, `passes` times, and return every row.
///
/// Kernels are visited in a fixed pseudo-random order — the same order on
/// every ref and tier, so a comparison is paired — and every pass revisits
/// them, so the pass-to-pass spread of one kernel is this harness's own A/A
/// noise floor.
///
/// # Panics
/// If any kernel fails to compile or is too narrow for this tier's batch.
pub fn run_corpus(kernels: &[CollapseKernel], passes: u32) -> Vec<Row> {
    let order = shuffled(kernels.len());
    let mut session = CollapseSession::open();
    let mut rows = Vec::with_capacity(kernels.len() * passes as usize);
    for pass in 0..passes {
        for &i in &order {
            rows.push(session.measure(&kernels[i], pass));
        }
    }
    rows.sort_by(|a, b| (&a.kernel, a.pass).cmp(&(&b.kernel, b.pass)));
    rows
}

/// A fixed permutation of `0..n` — xorshift64 with a constant seed, so the
/// visit order is identical on every ref, tier and pass.
fn shuffled(n: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    for i in (1..n).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        order.swap(i, (state % (i as u64 + 1)) as usize);
    }
    order
}

/// Write rows as JSONL, one object per line, stamped with the build's identity.
///
/// # Panics
/// If the file cannot be written or a row cannot be serialized.
pub fn write_jsonl(path: &Path, rows: &[Row], git_ref: &str, git_sha: &str, profile: &str) {
    let mut out = String::new();
    for row in rows {
        let mut row = row.clone();
        row.git_ref = git_ref.to_string();
        row.git_sha = git_sha.to_string();
        row.profile = profile.to_string();
        out.push_str(&serde_json::to_string(&row).expect("serialize row"));
        out.push('\n');
    }
    std::fs::write(path, out).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

/// Read rows back from one or more JSONL files.
///
/// # Panics
/// If a file cannot be read or a line does not parse.
#[must_use]
pub fn read_jsonl(paths: &[std::path::PathBuf]) -> Vec<Row> {
    let mut rows = Vec::new();
    for path in paths {
        let text = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        for (i, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            rows.push(
                serde_json::from_str::<Row>(line)
                    .unwrap_or_else(|e| panic!("{}:{}: {e}", path.display(), i + 1)),
            );
        }
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_visit_order_is_a_permutation_and_does_not_move() {
        let a = shuffled(64);
        let b = shuffled(64);
        assert_eq!(a, b, "the visit order must be the same on every run");
        let mut sorted = a.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..64).collect::<Vec<_>>());
    }

    #[test]
    fn a_synthetic_kernel_compiles_and_reports_traffic() {
        let kernels = corpus::synthetic();
        let kernel = kernels
            .iter()
            .find(|k| k.name.starts_with("invariant16_hot"))
            .expect("the corpus holds invariant16_hot");
        let result = compile_as_baked(&kernel.arena, kernel.root, kernel.extent);
        let trips = Trips::of(kernel.extent, LANES as u32);
        let statics = features_of(&result, trips);
        assert!(statics.bytes_total > 0);
        assert!(
            statics.body.instructions > 0,
            "the body of a kernel that reads X cannot be empty"
        );
        assert!(
            statics.dyn_memory_ops >= u64::from(statics.body.memory_ops()),
            "one body iteration is a lower bound on the call's memory traffic"
        );
    }

    #[test]
    fn hoisting_puts_work_in_a_prologue() {
        let kernels = corpus::synthetic();
        let kernel = kernels
            .iter()
            .find(|k| k.name.starts_with("invariant48_hot"))
            .expect("the corpus holds invariant48_hot");
        let result = compile_as_baked(&kernel.arena, kernel.root, kernel.extent);
        assert!(
            result.hoisted_values > 0,
            "48 X-invariant terms and nothing hoisted: the corpus is not exercising LICM"
        );
        let statics = features_of(&result, Trips::of(kernel.extent, LANES as u32));
        assert!(
            statics.frame.instructions + statics.row.instructions > 0,
            "hoisted values but empty prologues"
        );
    }
}
