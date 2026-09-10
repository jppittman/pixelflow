//! The numeric quarantine: one same-form JIT-vs-oracle gate for every label
//! source.
//!
//! Every expression that is about to become a *label* — a corpus tier entry in
//! `gen_bench_corpus`, or a live draw in a trainer — is first cross-checked against the
//! independent scalar oracle on a seeded 64-point magnitude sweep. This is a
//! SAME-ARENA check: both tiers evaluate the identical expression, so a
//! disagreement is never an algebraic-rewrite judgment call. It is a
//! miscompile (emitter or lowering bug) or an oracle regression.
//!
//! # Why this is a module and not a function in the corpus binary
//!
//! It used to live inside `gen_bench_corpus.rs`, which is a `[[bin]]` and
//! therefore exports nothing. The live synthetic stream — the *dominant* label
//! source at default flags — was consequently never numerically checked at all
//! (audit B8). A second copy of the predicate would drift from this one, and a
//! drifting gate is worse than a missing one, so there is exactly one
//! definition here and both callers import it.
//!
//! # The error model
//!
//! Acceptance is decided by [`pixelflow_ir::DifferentialCheck`], which carries
//! a **compositional per-point error bound** through the evaluation: each node
//! gets an absolute radius, seeded at the estimate instructions
//! (`Recip`/`Rsqrt`) and widened by each op's own rounding. The JIT lane is
//! accepted iff it lands inside `value ± radius`.
//!
//! The predecessor folded a max-over-ops tolerance (`Tolerance::loosest`) into
//! one whole-expression allowance, which has no compositional justification:
//! `recip`'s ~12-bit estimate error, amplified through `sin(K·Y²)` at a large
//! argument, exceeds any per-op row while every step stayed inside contract.
//! That fold rejected 14.4% of a real corpus as "compiler bugs". Every
//! tolerance constant now lives in `pixelflow-ir` beside the walk that applies
//! it — none are defined here, and none belong in either caller.
//!
//! Conditioning is likewise one number: [`PointCheck::is_well_conditioned`]
//! reports whether a point determines its answer well enough for a CROSS-form
//! comparison. The magnitude threshold and the cancellation-ratio heuristic it
//! replaced are both deleted, not kept as second opinions — the ratio rule
//! condemned `x · 1e-4` (a perfectly determined computation) on every point,
//! which is precisely how a broken extraction hid inside "metadata".

use std::io::Write;
use std::path::Path;

use pixelflow_codegen::emit::compile;
use pixelflow_ir::binding::BindingTable;
use pixelflow_ir::{DifferentialCheck, ExprData, MaskVerdict, OpKind, PointVerdict, Term};

use crate::training::factored::term_to_kernel_code;

/// Check-grid size. 64 seeded points spanning signs and magnitudes
/// (1e-4..1e4, plus pinned specials including signed zero).
pub const QUARANTINE_POINTS: usize = 64;

/// Seed for the xorshift-generated portion of the check grid. Deterministic:
/// every run (and every re-run of an investigation) sees the same grid, so a
/// sidecar record naming grid index 37 reconstructs exactly.
const QUARANTINE_GRID_SEED: u64 = 0x00C0_FFEE_5EED_2026;

/// Maximum tolerated JIT-vs-oracle mismatch rate. The quarantine compares the
/// SAME arena through both tiers under a compositional bound, so a mismatch is
/// a miscompile or an oracle regression — even 1% of those means the compiler
/// is broken (plan 0.4).
pub const MAX_MISMATCH_RATE: f64 = 0.01;

/// Maximum tolerated total exclusion rate (mismatches plus compile failures
/// and unquarantinable shapes). Above this the caller must refuse to mint
/// labels rather than mint a set with a biased hole where the hard
/// expressions were (audit M2 on censoring).
pub const MAX_EXCLUSION_RATE: f64 = 0.05;

/// The seeded magnitude-sweep check grid.
///
/// Deliberately a type rather than a bare array: building it is cheap but not
/// free, and both callers must hoist it out of their expression loop.
#[derive(Clone, Debug)]
pub struct QuarantineGrid {
    points: [[f32; 2]; QUARANTINE_POINTS],
}

impl Default for QuarantineGrid {
    fn default() -> Self {
        Self::new()
    }
}

impl QuarantineGrid {
    /// Build the grid. Points 0..4 are pinned specials (the legacy test point,
    /// signed zeros and ±1, the magnitude extremes, small integers); the rest
    /// are xorshift64 draws of sign × mantissa [1,2) × 10^{-4..=4}.
    #[must_use]
    pub fn new() -> Self {
        let mut points = [[0.0f32; 2]; QUARANTINE_POINTS];
        points[0] = [0.5, 0.7];
        points[1] = [0.0, -0.0];
        points[2] = [1e-4, -1e-4];
        points[3] = [1.0, 2.0];
        let mut state = QUARANTINE_GRID_SEED;
        for point in points.iter_mut().skip(4) {
            for lane in point.iter_mut() {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let sign = if state & 1 == 0 { 1.0f32 } else { -1.0f32 };
                let mantissa = 1.0 + ((state >> 8) & 0xFF) as f32 / 256.0;
                let exponent = ((state >> 16) % 9) as i32 - 4;
                *lane = sign * mantissa * 10.0f32.powi(exponent);
            }
        }
        Self { points }
    }

    /// A grid whose every index is `point`. Test-only: it exists so a test can
    /// pin a grid-wide property (every point unbounded, say) that the seeded
    /// sweep deliberately does not have.
    #[cfg(test)]
    fn uniform(point: [f32; 2]) -> Self {
        Self {
            points: [point; QUARANTINE_POINTS],
        }
    }

    /// The grid points, in index order — the indices sidecar records name.
    #[must_use]
    pub fn points(&self) -> &[[f32; 2]; QUARANTINE_POINTS] {
        &self.points
    }
}

use pixelflow_codegen::JIT_VECTOR_BYTES;
use pixelflow_codegen::emit::executable::{Point4, TileSlice};

const LANES: usize = JIT_VECTOR_BYTES / core::mem::size_of::<f32>();

// ── Exclusions ───────────────────────────────────────────────────────────────

/// Why an expression may not become a label.
#[derive(Debug)]
pub enum Exclusion {
    /// The JIT refused the expression; an entry that cannot compile can never
    /// be labeled, so it is excluded and counted (audit M2: loudly, with a
    /// sidecar record — never a silent `continue`).
    CompileFailed {
        /// The compiler's own error text.
        detail: String,
    },
    /// The oracle cannot evaluate this expression shape (params, buffers,
    /// reductions, memory ops), so no independent cross-check is possible.
    OracleUnsupported {
        /// Which shape, and what to do about it.
        detail: String,
    },
    /// Every grid point hit platform-divergent op inputs (CLAUDE.md contract
    /// tables), leaving nothing checkable — unverifiable, so excluded.
    NoCheckablePoints,
    /// Points survived the divergence screen, but not one of them produced a
    /// *bounded* verdict: every one composed a non-finite radius (the interval
    /// crossed a pole or a domain edge) or landed near a mask threshold. The
    /// JIT was therefore never accepted OR rejected anywhere on the grid, so an
    /// arbitrarily wrong backend is indistinguishable from a conforming one on
    /// this expression (Codex 3744586366). Admitting it as a label would be a
    /// vacuous pass, so it is excluded and counted.
    NoBoundedPoints,
    /// The JIT lane fell outside the composed error bound at some point.
    Mismatch {
        /// The grid point.
        point: [f32; 2],
        /// The oracle's value.
        expected: f32,
        /// The JIT's lane.
        got: f32,
        /// Absolute bound a conforming backend was allowed.
        bound: f32,
        /// [`bound`](Self::Mismatch::bound) relative to `expected` — the
        /// point's conditioning, for the journal.
        relative_bound: f32,
    },
    /// A mask-valued root produced the OPPOSITE valid mask at a point whose
    /// deciding comparison sits well clear of its threshold. Near-threshold
    /// disagreement is contract and lands in [`Conditioning`]; this one is a
    /// wrong answer (Codex R2).
    MaskMiscompile {
        /// The grid point.
        point: [f32; 2],
        /// The oracle's mask lane.
        expected: f32,
        /// The JIT's mask lane.
        got: f32,
    },
    /// A mask-valued root produced a lane that is neither `0x00000000` nor
    /// `0xFFFFFFFF` — a broken mask corrupts every downstream bitwise
    /// consumer, so no proximity and no tolerance excuses it.
    InvalidMaskPattern {
        /// The grid point.
        point: [f32; 2],
        /// The oracle's lane.
        expected: f32,
        /// The JIT's lane.
        got: f32,
        /// Whether the JIT lane was a valid mask pattern.
        got_is_mask: bool,
        /// Whether the oracle lane was a valid mask pattern.
        want_is_mask: bool,
    },
}

impl Exclusion {
    /// Stable slug for the sidecar's `reason` field. Both callers write the
    /// same vocabulary, so one analysis script reads both sidecars.
    #[must_use]
    pub fn reason(&self) -> &'static str {
        match self {
            Exclusion::CompileFailed { .. } => "compile_failed",
            Exclusion::OracleUnsupported { .. } => "oracle_unsupported",
            Exclusion::NoCheckablePoints => "no_checkable_points",
            Exclusion::NoBoundedPoints => "no_bounded_points",
            Exclusion::Mismatch { .. } => "numeric_mismatch",
            Exclusion::MaskMiscompile { .. } => "mask_miscompile",
            Exclusion::InvalidMaskPattern { .. } => "invalid_mask_pattern",
        }
    }

    /// Whether this exclusion accuses the compiler rather than the corpus.
    /// These are the ones [`MAX_MISMATCH_RATE`] governs.
    #[must_use]
    pub fn is_miscompile(&self) -> bool {
        matches!(
            self,
            Exclusion::Mismatch { .. }
                | Exclusion::MaskMiscompile { .. }
                | Exclusion::InvalidMaskPattern { .. }
        )
    }

    /// The sidecar JSON record for this exclusion. Floats are spelled as
    /// strings so NaN/inf survive the trip, and mask lanes as hex because a
    /// broken mask is a bit pattern, not a number.
    #[must_use]
    pub fn to_record(&self, name: &str, term: Term<'_>) -> serde_json::Value {
        match self {
            Exclusion::CompileFailed { detail } => serde_json::json!({
                "name": name,
                "reason": self.reason(),
                "expression": term_to_kernel_code(term),
                "detail": detail,
            }),
            Exclusion::OracleUnsupported { detail } => serde_json::json!({
                "name": name,
                "reason": self.reason(),
                "detail": detail,
            }),
            Exclusion::NoCheckablePoints | Exclusion::NoBoundedPoints => serde_json::json!({
                "name": name,
                "reason": self.reason(),
                "expression": term_to_kernel_code(term),
            }),
            Exclusion::Mismatch {
                point,
                expected,
                got,
                bound,
                relative_bound,
            } => serde_json::json!({
                "name": name,
                "reason": self.reason(),
                "expression": term_to_kernel_code(term),
                "point": point,
                "expected": format!("{expected:?}"),
                "got": format!("{got:?}"),
                "bound": format!("{bound:?}"),
                "relative_bound": format!("{relative_bound:?}"),
            }),
            Exclusion::MaskMiscompile {
                point,
                expected,
                got,
            } => serde_json::json!({
                "name": name,
                "reason": self.reason(),
                "expression": term_to_kernel_code(term),
                "point": point,
                "expected_bits": format!("{:#010x}", expected.to_bits()),
                "got_bits": format!("{:#010x}", got.to_bits()),
            }),
            Exclusion::InvalidMaskPattern {
                point,
                expected,
                got,
                got_is_mask,
                want_is_mask,
            } => serde_json::json!({
                "name": name,
                "reason": self.reason(),
                "expression": term_to_kernel_code(term),
                "point": point,
                "expected_bits": format!("{:#010x}", expected.to_bits()),
                "got_bits": format!("{:#010x}", got.to_bits()),
                "got_is_mask": got_is_mask,
                "want_is_mask": want_is_mask,
            }),
        }
    }
}

/// Per-expression conditioning survey of the check grid (plan 0.4): which grid
/// indices are unusable for downstream CROSS-form comparisons, and why. Never
/// grounds an exclusion — the same-form gate stays hard everywhere.
#[derive(Default, Debug)]
pub struct Conditioning {
    /// Indices where the point's evaluation path reached an op the language
    /// refuses to promise a value for (the CLAUDE.md contract tables). The
    /// walk is `Select`-aware: an unchosen branch is not on the path, so
    /// `Select(Lt(x,x), Round(-0.25), x)` stays fully checkable (Codex R3).
    pub divergent_points: Vec<usize>,
    /// Indices where the composed bound was not finite (the interval crossed
    /// a pole or a domain edge). Nothing is promised there — not a pass and
    /// not a failure.
    pub unbounded_points: Vec<usize>,
    /// Indices that do not determine their answer tightly enough to compare
    /// two algebraically-equal FORMS against each other
    /// ([`PointCheck::is_well_conditioned`]).
    pub ill_conditioned_points: Vec<usize>,
    /// Mask-valued roots only: indices where JIT and oracle produced opposite
    /// VALID masks with the deciding comparison's operands inside their own
    /// composed error — the language promises no verdict there. Metadata.
    pub mask_disagreement_points: Vec<usize>,
    /// Largest relative error bound observed at any non-divergent point. The
    /// single number to journal as this expression's conditioning.
    pub worst_relative_bound: f32,
    /// How many grid points produced a *bounded* verdict — a point where the
    /// JIT lane was actually accepted or rejected. This is the denominator of
    /// everything the quarantine claims to have verified: a pass over zero
    /// bounded points verifies nothing (Codex 3744586366).
    pub bounded_points: usize,
    /// How many grid points were surveyed at all. Zero when the expression
    /// never reached the grid (screening or compile failure).
    pub grid_points: usize,
}

impl Conditioning {
    /// Whether the whole grid was clean: no divergence, no unbounded points,
    /// no ill-conditioning, no mask disagreement.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.divergent_points.is_empty()
            && self.unbounded_points.is_empty()
            && self.ill_conditioned_points.is_empty()
            && self.mask_disagreement_points.is_empty()
    }

    /// Fraction of the surveyed grid that produced a bounded verdict — how much
    /// of this expression the quarantine actually verified. `None` when the
    /// grid was never surveyed (screening or compile failure), because a
    /// fraction over an empty denominator is not a number.
    #[must_use]
    pub fn bounded_fraction(&self) -> Option<f64> {
        (self.grid_points > 0).then(|| self.bounded_points as f64 / self.grid_points as f64)
    }

    /// The informational sidecar record. Grid indices, not values: the grid is
    /// seed-pinned, so indices reconstruct exactly.
    #[must_use]
    pub fn to_record(&self, name: &str) -> serde_json::Value {
        serde_json::json!({
            "name": name,
            "reason": "conditioning_metadata",
            "divergent_points": self.divergent_points,
            "unbounded_points": self.unbounded_points,
            "ill_conditioned_points": self.ill_conditioned_points,
            "mask_disagreement_points": self.mask_disagreement_points,
            "worst_relative_bound": format!("{:?}", self.worst_relative_bound),
            "bounded_points": self.bounded_points,
            "grid_points": self.grid_points,
        })
    }
}

/// One expression's quarantine outcome: the exclusion (if any) and the
/// conditioning survey that accompanies every verdict, passing or not.
#[derive(Debug)]
pub struct Verdict {
    /// `None` means the expression may become a label.
    pub exclusion: Option<Exclusion>,
    /// The grid survey. Populated even on an exclusion.
    pub conditioning: Conditioning,
}

/// Screen an expression for oracle support.
///
/// [`DifferentialCheck::at`] *panics* on `Param`, a bare `Buffer`, and
/// `Nary`/`Reduce` rather than screening them — loudly, by design — so this
/// must run first. `Gather`/`RawGather` are supported by the check itself but
/// refused here anyway: both callers run with an empty [`BindingTable`], so
/// there is no buffer to read. A `Uniform` is fine — an empty table reads its
/// declared default.
pub fn screen_for_oracle(term: Term<'_>) -> Result<(), String> {
    for node in term.root().descendants() {
        match *node {
            ExprData::Param(i) => return Err(format!("Param({i}) — substitute params first")),
            ExprData::Buffer(b) => return Err(format!("Buffer({}) — no binding table here", b.0)),
            // A `Uniform` needs no block: unbound, it reads the default its
            // declaration carries, which is exactly the value a caller who
            // never set it would get. A `Buffer` has no such fallback, which
            // is why that one is still refused.
            ExprData::Var(i) if usize::from(i) >= pixelflow_ir::COORD_AXES => {
                // Past the axes is two different things: the retired Z/W, and
                // a reduction binder that escaped its `Reduce`. Naming the
                // wrong one sends the reader to the wrong file.
                let why = if i < 4 {
                    "a retired coordinate axis; a per-call scalar is a Uniform"
                } else {
                    "a reduction index outside a Reduce"
                };
                return Err(format!("Var({i}) — {why}"));
            }
            ExprData::Op(op) => {
                // Arity is the DAG's, so an n-ary node is spotted by its
                // child count rather than by a variant. `Reduce` is the only
                // one built with more than three; `Tuple` is refused below
                // whatever its arity.
                if node.child_count() > 3 {
                    return Err(format!(
                        "{op:?} with {} children — a reduction rebinds its body per \
                         iteration; expand_reduce first",
                        node.child_count()
                    ));
                }
                if matches!(
                    op,
                    OpKind::Gather | OpKind::RawGather | OpKind::Dwrt | OpKind::Tuple
                ) {
                    return Err(format!("{op:?} — not quarantinable"));
                }
            }
            ExprData::Var(_) | ExprData::Const(_) | ExprData::Uniform(_) => {}
        }
    }
    Ok(())
}

/// Cross-check one expression: JIT (compiled once, evaluated per point)
/// against the scalar oracle on `grid`, under the compositional per-point
/// error bound.
///
/// The [`DifferentialCheck`] is prepared ONCE per expression and reused across
/// the grid — `new` clones and rebuilds the arena, so preparing it per point
/// would pay that 64 times.
#[must_use]
pub fn quarantine_verdict(term: Term<'_>, grid: &QuarantineGrid) -> Verdict {
    let mut conditioning = Conditioning::default();

    if let Err(detail) = screen_for_oracle(term) {
        return Verdict {
            exclusion: Some(Exclusion::OracleUnsupported { detail }),
            conditioning,
        };
    }

    let compiled = match compile(term) {
        Ok(c) => c,
        Err(e) => {
            return Verdict {
                exclusion: Some(Exclusion::CompileFailed {
                    detail: e.to_string(),
                }),
                conditioning,
            };
        }
    };

    let check = DifferentialCheck::new(term);
    let mask_root = check.root_is_mask_valued();
    let bindings = BindingTable::empty();
    // The kernel's arguments at their declared defaults — the same values the
    // empty binding table gives the oracle, so both sides read one block.
    // The screen refuses buffers, so the block is the whole context.
    let block: Vec<f32> = term.env().uniforms.iter().map(|d| d.default).collect();
    let ctx: [*const f32; 1] = [if block.is_empty() {
        core::ptr::null()
    } else {
        block.as_ptr()
    }];

    // `checkable` counts points that survived the platform-divergence screen;
    // `bounded` counts the strictly smaller set where the point actually
    // DECIDED — accepted or rejected the JIT lane. Only the second number is
    // evidence: a point whose composed radius is not finite, or whose mask sits
    // on its threshold, promises nothing and would let an arbitrarily wrong
    // backend pass (Codex 3744586366).
    let mut checkable = 0usize;
    let mut bounded = 0usize;
    // Severity-ranked worst numeric mismatch: (severity, exclusion).
    let mut worst: Option<(f32, Exclusion)> = None;
    let mut mask_invalid: Option<Exclusion> = None;
    let mut mask_miscompile: Option<Exclusion> = None;

    conditioning.grid_points = grid.points().len();
    for (chunk_idx, points) in grid.points().chunks(LANES).enumerate() {
        let mut xs = [0.0f32; LANES];
        let mut ys = [0.0f32; LANES];
        for (i, p) in points.iter().enumerate() {
            xs[i] = p[0];
            ys[i] = p[1];
        }
        // The base coordinate's last two lanes are dead: `emit::compile`
        // refuses an arena naming `Var(2)`/`Var(3)`, so nothing that got
        // this far can read them.
        let p = Point4::new(xs, ys, [0.0; LANES], [0.0; LANES]);
        let mut out = [0.0f32; LANES];
        unsafe {
            compiled
                .code
                .call_collapse(ctx.as_ptr(), TileSlice::single(out.as_mut_ptr()), p);
        }

        for (i, point) in points.iter().enumerate() {
            let idx = chunk_idx * LANES + i;
            let p = check.at(point, &bindings);
            if p.is_platform_divergent() {
                // Contract, not a defect: no backend answer is "the" answer here.
                conditioning.divergent_points.push(idx);
                continue;
            }
            checkable += 1;

            let rel = p.relative_error_bound();
            if rel > conditioning.worst_relative_bound {
                conditioning.worst_relative_bound = rel;
            }
            if !p.is_well_conditioned() {
                conditioning.ill_conditioned_points.push(idx);
            }

            let expected = p.value();
            let got = out[i];

            if mask_root {
                match p.classify_mask_root(got) {
                    MaskVerdict::Agree => bounded += 1,
                    MaskVerdict::NearThreshold => conditioning.mask_disagreement_points.push(idx),
                    MaskVerdict::Miscompile => {
                        bounded += 1;
                        if mask_miscompile.is_none() {
                            mask_miscompile = Some(Exclusion::MaskMiscompile {
                                point: *point,
                                expected,
                                got,
                            });
                        }
                    }
                    MaskVerdict::InvalidPattern {
                        got_is_mask,
                        want_is_mask,
                    } => {
                        bounded += 1;
                        if mask_invalid.is_none() {
                            mask_invalid = Some(Exclusion::InvalidMaskPattern {
                                point: *point,
                                expected,
                                got,
                                got_is_mask,
                                want_is_mask,
                            });
                        }
                    }
                }
                continue;
            }

            match p.verdict(got) {
                PointVerdict::Accept => bounded += 1,
                PointVerdict::Unbounded => conditioning.unbounded_points.push(idx),
                PointVerdict::Reject => {
                    bounded += 1;
                    // Severity for "worst point": absolute diff, with non-finite
                    // disagreement ranked above any finite one.
                    let severity = if got.is_finite() && expected.is_finite() {
                        (got - expected).abs()
                    } else {
                        f32::INFINITY
                    };
                    if worst.as_ref().is_none_or(|(prev, _)| severity > *prev) {
                        worst = Some((
                            severity,
                            Exclusion::Mismatch {
                                point: *point,
                                expected,
                                got,
                                bound: p.error_bound(),
                                relative_bound: rel,
                            },
                        ));
                    }
                }
            }
        }
    }

    conditioning.bounded_points = bounded;
    if checkable == 0 {
        return Verdict {
            exclusion: Some(Exclusion::NoCheckablePoints),
            conditioning,
        };
    }
    if bounded == 0 {
        // Points were checkABLE and none of them CHECKED anything: the JIT was
        // neither accepted nor rejected anywhere, so "passed" would mean only
        // "was never tested" (Codex 3744586366).
        return Verdict {
            exclusion: Some(Exclusion::NoBoundedPoints),
            conditioning,
        };
    }
    // An invalid mask pattern outranks an opposite-mask miscompile, which
    // outranks a numeric mismatch: that is decreasing order of "no possible
    // innocent explanation".
    let exclusion = mask_invalid
        .or(mask_miscompile)
        .or_else(|| worst.map(|(_, e)| e));
    Verdict {
        exclusion,
        conditioning,
    }
}

/// Owns the exclusion/conditioning sidecar and the end-of-run rate alarms.
///
/// The sidecar is written through the `File` directly, one line per record —
/// no `BufWriter`. The workspace compiles with `panic = "abort"`, so a panic
/// skips `Drop` and would discard anything still buffered, while the alarms
/// below panic while POINTING the reader at this sidecar for records that must
/// therefore already be on disk. Exclusions are asserted rare (<= 5%), so
/// per-record writes cost nothing.
pub struct Quarantine {
    grid: QuarantineGrid,
    log: std::fs::File,
    log_path: String,
    checked: usize,
    excluded: usize,
    mismatched: usize,
    coverage: BoundCoverage,
}

/// How much of the corpus the compositional bound actually verified.
///
/// A quarantine pass is only worth what its bounded points are worth, so this
/// travels beside the mismatch rate rather than inside a comment: a headline of
/// "zero miscompiles" over expressions whose grids never bounded anything is
/// vacuous, and these are the numbers that say which one happened (Codex
/// 3744586366).
#[derive(Clone, Copy, Debug, Default)]
pub struct BoundCoverage {
    /// Expressions excluded because not one grid point produced a bounded
    /// verdict ([`Exclusion::NoBoundedPoints`]).
    pub no_bounded_points: usize,
    /// Expressions excluded because every grid point was platform-divergent
    /// ([`Exclusion::NoCheckablePoints`]).
    pub no_checkable_points: usize,
    /// Expressions whose grid was surveyed at all — the denominator of
    /// [`mean_bounded_fraction`](Self::mean_bounded_fraction).
    pub surveyed: usize,
    /// Sum of per-expression bounded fractions over `surveyed`.
    bounded_fraction_sum: f64,
}

impl BoundCoverage {
    fn record(&mut self, conditioning: &Conditioning, exclusion: Option<&Exclusion>) {
        if let Some(f) = conditioning.bounded_fraction() {
            self.surveyed += 1;
            self.bounded_fraction_sum += f;
        }
        match exclusion {
            Some(Exclusion::NoBoundedPoints) => self.no_bounded_points += 1,
            Some(Exclusion::NoCheckablePoints) => self.no_checkable_points += 1,
            _ => {}
        }
    }

    /// Mean fraction of grid points that produced a bounded verdict, over every
    /// expression whose grid was surveyed. `None` when none was — a mean over
    /// an empty denominator is not a number.
    #[must_use]
    pub fn mean_bounded_fraction(&self) -> Option<f64> {
        (self.surveyed > 0).then(|| self.bounded_fraction_sum / self.surveyed as f64)
    }

    /// The run-summary line. Printed by [`Quarantine::finish`], and public so a
    /// caller that prints its own tally can print this one beside it.
    #[must_use]
    pub fn describe(&self) -> String {
        let mean = match self.mean_bounded_fraction() {
            Some(f) => format!("{:.1}%", f * 100.0),
            None => "n/a (no expression reached the grid)".to_string(),
        };
        format!(
            "Bound coverage: {} excluded as no_bounded_points, {} as no_checkable_points; mean \
             bounded grid fraction {mean} over {} surveyed expressions — this is the share of the \
             corpus the same-form gate actually verified",
            self.no_bounded_points, self.no_checkable_points, self.surveyed
        )
    }
}

impl Quarantine {
    /// Create (truncating) the sidecar at `log_path` and start counting.
    ///
    /// The path is a parameter, not a constant, because the corpus build and
    /// the trainer must not clobber each other's sidecar.
    #[must_use]
    pub fn new(log_path: &str) -> Self {
        if let Some(parent) = Path::new(log_path).parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent).unwrap_or_else(|e| {
                panic!(
                    "failed to create quarantine log directory {}: {e}",
                    parent.display()
                )
            });
        }
        let file = std::fs::File::create(log_path)
            .unwrap_or_else(|e| panic!("failed to create quarantine log {log_path}: {e}"));
        Self::from_file(file, log_path.to_string())
    }

    /// Wrap an already-open sidecar file — for a caller that writes its own
    /// exclusion records to the same JSONL (the trainer's `ExclusionLog`).
    /// `log_path` is used only in messages.
    #[must_use]
    pub fn from_file(log: std::fs::File, log_path: String) -> Self {
        Self {
            grid: QuarantineGrid::new(),
            log,
            log_path,
            checked: 0,
            excluded: 0,
            mismatched: 0,
            coverage: BoundCoverage::default(),
        }
    }

    /// Cross-check one expression. Returns `true` if it passed; on `false` the
    /// expression must not become a label and a sidecar line was written. A
    /// passing expression with a non-clean conditioning survey gets an
    /// informational `conditioning_metadata` line.
    pub fn check(&mut self, name: &str, term: Term<'_>) -> bool {
        self.verdict(name, term).0.is_none()
    }

    /// [`check`](Self::check), handing back the verdict itself for callers that
    /// need the *reason* rather than a bool — per-family attrition accounting
    /// in the corpus build, phase attribution in the trainer. Counting and
    /// sidecar writing happen here either way, so the two entry points can
    /// never disagree about what was checked.
    pub fn verdict(&mut self, name: &str, term: Term<'_>) -> (Option<Exclusion>, Conditioning) {
        self.checked += 1;
        let Verdict {
            exclusion,
            conditioning,
        } = quarantine_verdict(term, &self.grid);
        self.coverage.record(&conditioning, exclusion.as_ref());
        match &exclusion {
            None => {
                if !conditioning.is_clean() {
                    self.write(&conditioning.to_record(name));
                }
            }
            Some(exclusion) => {
                self.excluded += 1;
                if exclusion.is_miscompile() {
                    self.mismatched += 1;
                }
                let record = exclusion.to_record(name, term);
                self.write(&record);
            }
        }
        (exclusion, conditioning)
    }

    /// `(checked, excluded, mismatched)` so far.
    #[must_use]
    pub fn tallies(&self) -> (usize, usize, usize) {
        (self.checked, self.excluded, self.mismatched)
    }

    /// How much of what was checked was actually verified — the companion of
    /// [`tallies`](Self::tallies), and the number that says whether a
    /// zero-mismatch tally is evidence or vacuum.
    #[must_use]
    pub fn coverage(&self) -> BoundCoverage {
        self.coverage
    }

    fn write(&mut self, line: &serde_json::Value) {
        writeln!(self.log, "{line}")
            .unwrap_or_else(|e| panic!("failed to write quarantine log {}: {e}", self.log_path));
    }

    /// Number of expressions checked so far.
    #[must_use]
    pub fn checked(&self) -> usize {
        self.checked
    }

    /// Number excluded so far.
    #[must_use]
    pub fn excluded(&self) -> usize {
        self.excluded
    }

    /// Number excluded for a miscompile (numeric or mask) so far.
    #[must_use]
    pub fn mismatched(&self) -> usize {
        self.mismatched
    }

    /// The sidecar path, for messages.
    #[must_use]
    pub fn log_path(&self) -> &str {
        &self.log_path
    }

    /// `(mismatch_rate, exclusion_rate)` over everything checked.
    ///
    /// # Panics
    ///
    /// Panics if nothing was checked — a rate over an empty denominator is not
    /// a number, and reporting one would be the silent failure this project
    /// forbids.
    #[must_use]
    pub fn rates(&self) -> (f64, f64) {
        assert!(
            self.checked > 0,
            "quarantine rates requested before anything was checked"
        );
        (
            self.mismatched as f64 / self.checked as f64,
            self.excluded as f64 / self.checked as f64,
        )
    }

    /// Print the tally and panic if either rate alarm trips. Every sidecar
    /// record was already written unbuffered, so there is nothing to flush.
    ///
    /// # Panics
    ///
    /// Panics if the mismatch rate exceeds [`MAX_MISMATCH_RATE`] or the total
    /// exclusion rate exceeds [`MAX_EXCLUSION_RATE`], and if nothing was
    /// checked at all.
    pub fn finish(self) {
        let (mismatch_rate, exclusion_rate) = self.rates();
        println!(
            "Quarantine: {} checked, {} excluded ({:.3}%), of which {} JIT-vs-oracle miscompiles \
             ({:.3}%) — sidecar: {}",
            self.checked,
            self.excluded,
            exclusion_rate * 100.0,
            self.mismatched,
            mismatch_rate * 100.0,
            self.log_path
        );
        // The mismatch rate above is only worth its denominator: print what
        // fraction of the corpus the bound actually decided, on its own line.
        println!("{}", self.coverage.describe());
        assert!(
            mismatch_rate <= MAX_MISMATCH_RATE,
            "quarantine mismatch rate {:.2}% exceeds {:.0}% (plan 0.4, audit H1): the JIT and the \
             scalar oracle disagree on the SAME arena beyond the composed error bound — a \
             miscompile/lowering bug or an oracle regression, never corpus noise; see {}",
            mismatch_rate * 100.0,
            MAX_MISMATCH_RATE * 100.0,
            self.log_path
        );
        assert!(
            exclusion_rate <= MAX_EXCLUSION_RATE,
            "quarantine exclusion rate {:.2}% exceeds {:.0}% (audit M2): the label set is being \
             censored — too many expressions fail to compile or cannot be cross-checked; see {}",
            exclusion_rate * 100.0,
            MAX_EXCLUSION_RATE * 100.0,
            self.log_path
        );
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_ir::{Environment, ExprBuilder, ExprData, Rooted};

    /// A finished graph plus the environment its leaves index — what every
    /// entry point here takes, as one value.
    fn built(
        f: impl FnOnce(&mut ExprBuilder) -> pixelflow_ir::ExprRef,
    ) -> (Rooted<ExprData>, Environment) {
        let mut b = ExprBuilder::new();
        let root = f(&mut b);
        b.finish(&[root])
    }

    fn recip_pow16() -> (Rooted<ExprData>, Environment) {
        // recip(x)^16 — the Codex R1 shape: the ~12-bit estimate's error is
        // squared four times, so the max-over-ops fold's 5e-4 allowance is
        // exceeded by a perfectly conforming backend.
        built(|a| {
            let x = a.push_var(0);
            let r = a.push_unary(OpKind::Recip, x);
            let r2 = a.push_binary(OpKind::Mul, r, r);
            let r4 = a.push_binary(OpKind::Mul, r2, r2);
            let r8 = a.push_binary(OpKind::Mul, r4, r4);
            a.push_binary(OpKind::Mul, r8, r8)
        })
    }

    fn amplified_sin() -> (Rooted<ExprData>, Environment) {
        // sin(recip(x) * y * y) — the smoke run's B1 shape: a reciprocal
        // estimate fed into a large trig argument, where the relative error
        // reaches order 1 while every step stays inside contract.
        built(|a| {
            let x = a.push_var(0);
            let y = a.push_var(1);
            let k = a.push_unary(OpKind::Recip, x);
            let yy = a.push_binary(OpKind::Mul, y, y);
            let arg = a.push_binary(OpKind::Mul, k, yy);
            a.push_unary(OpKind::Sin, arg)
        })
    }

    #[test]
    fn grid_is_deterministic_and_sweeps_magnitudes() {
        let a = QuarantineGrid::new();
        let b = QuarantineGrid::new();
        for (pa, pb) in a.points().iter().zip(b.points().iter()) {
            for (la, lb) in pa.iter().zip(pb.iter()) {
                assert_eq!(la.to_bits(), lb.to_bits());
            }
        }
        // Signed zero and the magnitude extremes are pinned.
        assert_eq!(a.points()[1][0].to_bits(), 0.0f32.to_bits());
        assert_eq!(a.points()[1][1].to_bits(), (-0.0f32).to_bits());
        assert_eq!(a.points()[2][0], 1e-4);
        for point in &a.points()[4..] {
            for lane in point {
                assert!(lane.is_finite());
                let mag = lane.abs();
                assert!((9e-5..=2.1e4).contains(&mag), "lane {lane} outside sweep");
            }
        }
    }

    #[test]
    fn screen_rejects_oracle_unsupported_shapes() {
        let (expr, env) = built(|a| a.push_param(0));
        let err =
            screen_for_oracle(Term::new(expr.entry(), &env)).expect_err("params are unsupported");
        assert!(err.contains("Param"), "got: {err}");

        let (expr, env) = built(|a| {
            let x = a.push_var(0);
            a.push_nary(OpKind::Tuple, &[x])
        });
        let err =
            screen_for_oracle(Term::new(expr.entry(), &env)).expect_err("Tuple is unsupported");
        assert!(err.contains("Tuple"), "got: {err}");

        // A reduction: n-ary by child count, which is where arity lives now.
        let (expr, env) = built(|a| {
            let y = a.push_var(4);
            a.push_reduce(OpKind::Add, 4, 3, y)
        });
        let err = screen_for_oracle(Term::new(expr.entry(), &env))
            .expect_err("an unexpanded reduction is unsupported");
        assert!(err.contains("children"), "got: {err}");
    }

    #[test]
    fn screen_accepts_plain_arithmetic() {
        let (expr, env) = recip_pow16();
        screen_for_oracle(Term::new(expr.entry(), &env))
            .expect("plain arithmetic is quarantinable");
    }

    fn int_reinterpret_of_recip() -> (Rooted<ExprData>, Environment) {
        // `IntToFloat(Recip(x))` reinterprets an ESTIMATE's bit pattern, so the
        // composed radius is unbounded at every point (a bit pattern has no
        // radius) AND the two tiers' estimates differ in the low bits, which
        // the reinterpretation blows up into different numbers. That is the
        // vacuous-pass shape: nothing is ever accepted or rejected.
        built(|a| {
            let x = a.push_var(0);
            let r = a.push_unary(OpKind::Recip, x);
            a.push_unary(OpKind::IntToFloat, r)
        })
    }

    #[test]
    fn exclusion_reasons_are_stable_slugs() {
        assert_eq!(Exclusion::NoCheckablePoints.reason(), "no_checkable_points");
        assert!(!Exclusion::NoCheckablePoints.is_miscompile());
        assert_eq!(Exclusion::NoBoundedPoints.reason(), "no_bounded_points");
        assert!(
            !Exclusion::NoBoundedPoints.is_miscompile(),
            "an unverifiable expression accuses the corpus, not the compiler"
        );
        let m = Exclusion::Mismatch {
            point: [0.0; 2],
            expected: 1.0,
            got: 2.0,
            bound: 0.0,
            relative_bound: 0.0,
        };
        assert_eq!(m.reason(), "numeric_mismatch");
        assert!(m.is_miscompile());
        let mm = Exclusion::MaskMiscompile {
            point: [0.0; 2],
            expected: 0.0,
            got: f32::from_bits(0xFFFF_FFFF),
        };
        assert_eq!(mm.reason(), "mask_miscompile");
        assert!(mm.is_miscompile());
    }

    #[test]
    fn conditioning_record_names_every_bucket() {
        let mut c = Conditioning::default();
        c.divergent_points.push(3);
        c.unbounded_points.push(4);
        c.ill_conditioned_points.push(5);
        c.mask_disagreement_points.push(6);
        c.worst_relative_bound = 0.25;
        assert!(!c.is_clean());
        let rec = c.to_record("k");
        assert_eq!(rec["divergent_points"][0], 3);
        assert_eq!(rec["unbounded_points"][0], 4);
        assert_eq!(rec["ill_conditioned_points"][0], 5);
        assert_eq!(rec["mask_disagreement_points"][0], 6);
        assert_eq!(rec["reason"], "conditioning_metadata");
    }

    #[test]
    fn bounded_fraction_needs_a_surveyed_grid() {
        let never_surveyed = Conditioning::default();
        assert_eq!(
            never_surveyed.bounded_fraction(),
            None,
            "a fraction over an empty grid is not a number"
        );
        let c = Conditioning {
            bounded_points: 16,
            grid_points: 64,
            ..Conditioning::default()
        };
        assert_eq!(c.bounded_fraction(), Some(0.25));
        let rec = c.to_record("k");
        assert_eq!(rec["bounded_points"], 16);
        assert_eq!(rec["grid_points"], 64);
    }

    #[test]
    fn bound_coverage_counts_vacuous_grids_separately() {
        let mut cov = BoundCoverage::default();
        // Verified: half the grid bounded.
        cov.record(
            &Conditioning {
                bounded_points: 32,
                grid_points: 64,
                ..Conditioning::default()
            },
            None,
        );
        // Vacuous: surveyed, nothing bounded.
        cov.record(
            &Conditioning {
                bounded_points: 0,
                grid_points: 64,
                ..Conditioning::default()
            },
            Some(&Exclusion::NoBoundedPoints),
        );
        // Wholly divergent: surveyed, nothing checkable.
        cov.record(
            &Conditioning {
                bounded_points: 0,
                grid_points: 64,
                divergent_points: (0..64).collect(),
                ..Conditioning::default()
            },
            Some(&Exclusion::NoCheckablePoints),
        );
        // Never reached the grid: excluded from the mean, not counted as 0%.
        cov.record(
            &Conditioning::default(),
            Some(&Exclusion::CompileFailed {
                detail: "nope".into(),
            }),
        );
        assert_eq!(cov.no_bounded_points, 1);
        assert_eq!(cov.no_checkable_points, 1);
        assert_eq!(cov.surveyed, 3);
        assert_eq!(cov.mean_bounded_fraction(), Some(0.5 / 3.0));
        let line = cov.describe();
        assert!(line.contains("1 excluded as no_bounded_points"), "{line}");
        assert!(line.contains("16.7%"), "{line}");
    }

    // ── The compositional bound, end to end (JIT required) ───────────────────

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn amplified_estimate_error_is_not_a_miscompile() {
        // Codex R1 / smoke B1: BOTH of these are algebraically valid kernels a
        // conforming backend computes with amplified — but bounded — estimate
        // error. The max-over-ops fold called them compiler bugs; the
        // compositional bound must admit them.
        let grid = QuarantineGrid::new();
        for (label, (expr, env)) in [
            ("recip(x)^16", recip_pow16()),
            ("sin(recip(x)*y*y)", amplified_sin()),
        ] {
            let v = quarantine_verdict(Term::new(expr.entry(), &env), &grid);
            assert!(
                v.exclusion.is_none(),
                "{label} must pass the same-form gate; got {:?}",
                v.exclusion
            );
        }
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn a_grid_that_bounds_nothing_is_excluded_not_admitted() {
        // Codex 3744586366: before the fix every point of this expression was
        // recorded as metadata, `checkable` stayed at 64, and the expression
        // was admitted as a label — while the JIT had been neither accepted nor
        // rejected anywhere on the grid.
        let sweep = QuarantineGrid::new();
        let (expr, env) = int_reinterpret_of_recip();
        // Find a point this shape cannot bound, then make the whole grid that
        // point: the seeded sweep deliberately mixes bounded and unbounded
        // points, and the case under test is the all-unbounded grid.
        let probe = quarantine_verdict(Term::new(expr.entry(), &env), &sweep);
        let &idx =
            probe.conditioning.unbounded_points.first().expect(
                "IntToFloat(Recip(x)) reinterprets an estimate — some point must be unbounded",
            );
        let grid = QuarantineGrid::uniform(sweep.points()[idx]);
        let v = quarantine_verdict(Term::new(expr.entry(), &env), &grid);
        assert_eq!(
            v.conditioning.bounded_points, 0,
            "shape precondition: this grid must bound nothing"
        );
        assert!(
            matches!(v.exclusion, Some(Exclusion::NoBoundedPoints)),
            "an all-unbounded grid must be excluded, not waved through; got {:?}",
            v.exclusion
        );
        assert_eq!(v.conditioning.grid_points, QUARANTINE_POINTS);
        assert_eq!(v.conditioning.bounded_fraction(), Some(0.0));
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn a_passing_expression_verified_something() {
        // The invariant the fix buys: a pass is now evidence. Every admitted
        // expression decided at least one point, and the survey adds up to the
        // grid.
        let grid = QuarantineGrid::new();
        for (label, (expr, env)) in [
            ("recip(x)^16", recip_pow16()),
            ("sin(recip(x)*y*y)", amplified_sin()),
        ] {
            let v = quarantine_verdict(Term::new(expr.entry(), &env), &grid);
            assert!(v.exclusion.is_none(), "{label}: {:?}", v.exclusion);
            assert!(
                v.conditioning.bounded_points > 0,
                "{label} passed without deciding a single point"
            );
            let c = &v.conditioning;
            assert_eq!(
                c.bounded_points
                    + c.divergent_points.len()
                    + c.unbounded_points.len()
                    + c.mask_disagreement_points.len(),
                QUARANTINE_POINTS,
                "{label}: every grid point lands in exactly one bucket"
            );
        }
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn amplification_shows_up_as_conditioning_not_exclusion() {
        // The same amplification that must not exclude the expression MUST be
        // visible: an amplifying expression reports a wide relative bound, and
        // the points where it exceeds the cross-form threshold are listed.
        let grid = QuarantineGrid::new();
        let (expr, env) = amplified_sin();
        let v = quarantine_verdict(Term::new(expr.entry(), &env), &grid);
        assert!(v.exclusion.is_none());
        assert!(
            !v.conditioning.ill_conditioned_points.is_empty(),
            "sin at a large, estimate-derived argument must be ill-conditioned somewhere"
        );
        assert!(
            v.conditioning.worst_relative_bound > 1e-3,
            "worst relative bound {} should reflect the amplification",
            v.conditioning.worst_relative_bound
        );
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn benign_low_gain_multiply_stays_well_conditioned() {
        // Item 8 / Codex 3741889906: the deleted ratio heuristic condemned
        // `x * 1e-4` on every point because the output is far below the
        // largest intermediate. Its propagated bound is tiny, so every point
        // must stay eligible — otherwise a broken extraction hides as
        // "metadata".
        let (expr, env) = built(|a| {
            let x = a.push_var(0);
            let k = a.push_const(1e-4);
            a.push_binary(OpKind::Mul, x, k)
        });
        let grid = QuarantineGrid::new();
        let v = quarantine_verdict(Term::new(expr.entry(), &env), &grid);
        assert!(v.exclusion.is_none(), "x*1e-4 must pass");
        assert!(
            v.conditioning.ill_conditioned_points.is_empty(),
            "x*1e-4 is perfectly determined at every point; flagged {:?}",
            v.conditioning.ill_conditioned_points
        );
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn select_with_a_divergent_unchosen_branch_stays_checkable() {
        // Codex R3: `Select(Lt(x, x), Round(-0.25), x)` is portably `x` —
        // `Lt(x, x)` is false everywhere, so `Round` is never on the path. The
        // old whole-arena divergence walker skipped every grid point and
        // reported NoCheckablePoints.
        let (expr, env) = built(|a| {
            let x = a.push_var(0);
            let cond = a.push_binary(OpKind::Lt, x, x);
            let quarter = a.push_const(-0.25);
            let rounded = a.push_unary(OpKind::Round, quarter);
            a.push_ternary(OpKind::Select, cond, rounded, x)
        });
        let grid = QuarantineGrid::new();
        let v = quarantine_verdict(Term::new(expr.entry(), &env), &grid);
        assert!(
            v.exclusion.is_none(),
            "the unchosen branch's divergence must not exclude the expression; got {:?}",
            v.exclusion
        );
        assert!(
            v.conditioning.divergent_points.len() < QUARANTINE_POINTS,
            "at least some points must remain checkable"
        );
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn mask_root_passes_without_false_ill_conditioning() {
        // An all-ones TRUE lane READS as NaN, so a numeric conditioning test
        // would flag every true point. The mask path compares bit patterns and
        // records nothing away from ties.
        let (expr, env) = built(|a| {
            let x = a.push_var(0);
            let y = a.push_var(1);
            a.push_binary(OpKind::Lt, x, y)
        });
        let grid = QuarantineGrid::new();
        let v = quarantine_verdict(Term::new(expr.entry(), &env), &grid);
        assert!(v.exclusion.is_none(), "Lt(x, y) must pass");
        assert!(
            v.conditioning.ill_conditioned_points.is_empty(),
            "mask lanes must never be flagged ill-conditioned for reading as NaN"
        );
        assert!(v.conditioning.mask_disagreement_points.is_empty());
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn divergent_points_are_metadata_never_exclusions() {
        // min(X, Y) hits opposite-signed zeros at grid point 1.
        let (expr, env) = built(|a| {
            let x = a.push_var(0);
            let y = a.push_var(1);
            a.push_binary(OpKind::Min, x, y)
        });
        let grid = QuarantineGrid::new();
        let v = quarantine_verdict(Term::new(expr.entry(), &env), &grid);
        assert!(
            v.exclusion.is_none(),
            "min(x, y) must pass the same-form gate"
        );
        assert!(
            v.conditioning.divergent_points.contains(&1),
            "min on (0.0, -0.0) must be recorded as divergent"
        );
        assert!(!v.conditioning.divergent_points.contains(&0));
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn quarantine_sidecar_records_are_written_per_expression() {
        let dir = std::env::temp_dir().join(format!("pxq_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("tmp dir");
        let path = dir.join("q.jsonl");
        let path_str = path.to_string_lossy().to_string();
        let mut q = Quarantine::new(&path_str);

        let (expr, env) = amplified_sin();
        assert!(
            q.check("amplified", Term::new(expr.entry(), &env)),
            "must pass the gate"
        );

        // An unquarantinable shape is excluded and recorded.
        let (expr, env) = built(|a| a.push_param(0));
        assert!(!q.check("param", Term::new(expr.entry(), &env)));
        assert_eq!(q.excluded(), 1);
        assert_eq!(q.mismatched(), 0, "a screen refusal is not a miscompile");

        let text = std::fs::read_to_string(&path).expect("sidecar readable");
        let reasons: Vec<String> = text
            .lines()
            .map(|l| {
                let v: serde_json::Value = serde_json::from_str(l).expect("valid JSON");
                v["reason"].as_str().expect("reason slug").to_string()
            })
            .collect();
        assert!(reasons.contains(&"conditioning_metadata".to_string()));
        assert!(reasons.contains(&"oracle_unsupported".to_string()));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    #[should_panic(expected = "before anything was checked")]
    fn rates_refuse_an_empty_denominator() {
        let dir = std::env::temp_dir().join(format!("pxq_empty_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("tmp dir");
        let path = dir.join("q.jsonl").to_string_lossy().to_string();
        let q = Quarantine::new(&path);
        let _ = q.rates();
    }
}
