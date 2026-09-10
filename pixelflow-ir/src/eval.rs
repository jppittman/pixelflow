//! Differential-testing oracle: a scalar, memoized walk over an expression
//! graph.
//!
//! **This is not an execution tier.** PixelFlow is JIT-only: every rendered
//! pixel comes from emitted machine code. This module exists so tests can ask
//! "what should that machine code have produced?" without hand-computing the
//! answer, and it is compiled out of the shipped crate — it lives behind the
//! `oracle` feature, off by default, enabled only by `[dev-dependencies]`.
//!
//! It is a *reference walker over the one semantics*, not a second semantics.
//! [`eval_scalar`] runs the same `expand_transcendentals` lowering the emitter
//! runs, then interprets the result, so "the interpreter disagrees with the
//! JIT" always means the emitter is wrong about an arithmetic primitive — never
//! that the two disagree about what `sin` means. That property is what makes
//! the oracle worth keeping: it is how the select-guard and clamp-ordering
//! miscompiles were found.
//!
//! `Gather` semantics deliberately mirror `DiscreteManifold::eval` in
//! `pixelflow-core`: floor each index, clamp to `[0, extent - 1]`, read
//! row-major. The round-trip test asserts that equivalence.

use crate::binding::BindingTable;
use crate::dag::{Node, Rooted, SideTable};
use crate::decl::{COORD_AXES, REDUCE_BINDER_BASE, RETIRED_COORD_AXES, UniformId};
use crate::expr::{Environment, ExprData, Term};
use crate::kind::OpKind;
// `alloc`, not `std`: the oracle must stay buildable with the `std` feature
// off (the crate is the bootloader target's floor), and a memoized DAG walk
// genuinely needs heap storage — `alloc` is already the crate baseline.
use alloc::vec;
use alloc::vec::Vec;

/// Evaluate the subtree rooted at `root` at scalar coordinates `vars`
/// (`[X, Y]`), reading bound buffers through `bindings`.
///
/// # Panics
///
/// Panics on a node the reference interpreter does not handle: a `Param`
/// (substitute first), a bare `Buffer` outside a `Gather`, an `Nary`, or an
/// op with no scalar evaluation. These are programming errors, not inputs.
#[must_use]
pub fn eval_scalar(term: Term<'_>, vars: &[f32; COORD_AXES], bindings: &BindingTable<'_>) -> f32 {
    // A transcendental in this language IS the expansion the compiler emits, so
    // the reference interpreter evaluates that expansion rather than the host's
    // `f32::sin`. Three reasons, one answer:
    //
    //   * one definition — the interpreter and the JIT cannot disagree about
    //     `sin`, which is exactly how the known 0.5116-vs-0.8415 parity class
    //     of bug arises;
    //   * no dependency — the expansion is arithmetic, so no libm;
    //   * no_std — `f32::sin` does not exist in `core`, and this crate has to
    //     run without an operating system.
    //
    // The graph is a DAG and the expansions share subexpressions heavily, so
    // the walk memoizes per node — a naive tree walk re-evaluates every shared
    // node once per reference, which is exponential in nesting depth for
    // transcendental-heavy expressions. Nodes under a reduction binder are
    // excluded: their value changes with the binding, so caching them across
    // iterations would return a stale term.
    let expanded = crate::passes::expand_transcendentals(term);
    let term = Term::new(expanded.entry(), term.env());
    let variance = crate::variance::compute_dag_variance(term.dag());
    let memo = core::cell::RefCell::new(term.dag().side_table(None));
    Env {
        term,
        vars,
        bindings,
        reduce_vars: [0.0; 4],
        variance: &variance,
        memo: &memo,
    }
    .eval(term.root())
}

/// How closely a JIT-computed lane must match [`eval_scalar`]'s answer for the
/// same `(term, coordinates)` before the divergence is a bug.
///
/// The oracle and the JIT share one semantics (the same lowering), so most ops
/// agree bit-for-bit — but a handful of instructions are *estimates* or
/// target-rounded, and an optimized/extracted form may associate float
/// arithmetic differently than the reference graph. A single global tolerance
/// either masks real miscompiles (0.2 absolute hides a dropped Newton step —
/// audit H1) or rejects correct hardware (a 12-bit `rcpps` can never match an
/// exact `1.0 / x`). [`equivalence_tolerance`] gives the per-op allowance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Tolerance {
    /// The result is a lane **bit pattern** (a mask, an integer-domain lane) or
    /// is produced by exact/correctly-rounded instructions on every backend:
    /// the tiers must agree bit-for-bit. "Close" is meaningless for a pattern —
    /// an all-ones mask off by one bit is not almost a mask, and NaN payloads
    /// are not interchangeable here (an all-ones lane *reads* as NaN, so a
    /// NaN-tolerant comparison would accept `0x7fc00000` where the hardware
    /// wrote `0xFFFFFFFF`).
    BitExact,
    /// `|got − want| ≤ abs` or `|got − want| ≤ rel·|want|`. Identical bit
    /// patterns and NaN-vs-NaN (any payload) always pass — infinities of the
    /// same sign compare equal by bits, and both tiers producing NaN is
    /// agreement about the domain, not about a payload.
    Numeric { rel: f32, abs: f32 },
}

impl Tolerance {
    /// Whether `got` (the JIT's lane) is acceptably close to `want` (the
    /// oracle's answer) under this tolerance.
    #[must_use]
    pub fn accepts(self, got: f32, want: f32) -> bool {
        match self {
            Self::BitExact => got.to_bits() == want.to_bits(),
            Self::Numeric { rel, abs } => {
                if got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()) {
                    return true;
                }
                // Non-finite and not bit-identical is disagreement, full stop —
                // `rel · ∞ = ∞` would otherwise accept anything against an
                // infinite reference.
                if !got.is_finite() || !want.is_finite() {
                    return false;
                }
                let diff = (got - want).abs();
                diff <= abs || diff <= rel * want.abs()
            }
        }
    }

    /// The looser of two tolerances.
    ///
    /// **This is not a whole-expression acceptance rule, and must not be used
    /// as one.** Folding the per-op table over a whole graph asserts that a
    /// composition drifts no more than its worst single op, which is false as
    /// soon as an inexact value is reused: `recip(x)^16` multiplies the
    /// estimate's relative width by 16, and `sin(K·Y²)` multiplies it by the
    /// argument. Both are correct kernels that a max-over-ops fold reports as
    /// miscompiles — that fold panicked on ~14% of a real corpus. Whole-
    /// expression acceptance belongs to [`DifferentialCheck`], which composes
    /// the bound along the evaluation path at the point being checked.
    ///
    /// What remains legitimate is comparing two *single-op* allowances, which
    /// is what the per-op conformance tests do.
    #[must_use]
    pub fn loosest(self, other: Self) -> Self {
        match (self, other) {
            (Self::BitExact, t) | (t, Self::BitExact) => t,
            (Self::Numeric { rel: r1, abs: a1 }, Self::Numeric { rel: r2, abs: a2 }) => {
                Self::Numeric {
                    rel: r1.max(r2),
                    abs: a1.max(a2),
                }
            }
        }
    }
}

/// Whether `bits` is one of the two lane patterns a mask may hold: all-zeros
/// (false) or all-ones (true). Everything else — a NaN payload like
/// `0x7fc00000`, a numeric `1.0` — is *not almost a mask*; it is a broken one,
/// because `Select`/`BitAnd`/`BitOr` blend per-bit and a partial pattern
/// manufactures values neither branch held (see [`OpKind::mask`]).
#[must_use]
pub fn is_valid_mask(bits: u32) -> bool {
    bits == 0x0000_0000 || bits == 0xFFFF_FFFF
}

/// Whether the value produced at `root` is a **mask lane** (all-ones or
/// all-zeros), as opposed to a number or a general integer bit pattern.
///
/// Comparisons always produce masks; `BitAnd`/`BitOr` produce one iff both
/// operands are masks; `Select` produces one iff both *branches* are masks.
/// Note this is narrower than [`OpKind::is_bitwise_domain`]: `TruncToInt`,
/// `IAdd`, `Shl`, … are bit-pattern-domain but their results are arbitrary
/// integer patterns, which `BitExact` already covers — demanding all-ones of
/// `trunc_to_int(5.0)` would be wrong.
///
/// A `Const` counts as a mask **iff its bit pattern is one**
/// ([`is_valid_mask`]) — and `0.0` is `0x00000000`, the false mask, so the
/// everyday `Select(cond, Lt(a, b), 0.0)` is mask-valued. Treating that
/// constant as "unknown intent" is not the conservative choice it looks like:
/// it degrades the root to the numeric path, whose NaN-vs-NaN rule then accepts
/// a `0x7fc00000` lane against a required `0xFFFFFFFF`. A non-mask constant
/// like `1.0` still makes the root numeric, which is correct — blending `1.0`
/// through a mask really does produce a number.
///
/// Callers use this to pick the acceptance path for a differential check:
/// mask-valued root ⇒ [`compare_mask_root`]; otherwise fold
/// [`equivalence_tolerance`] with [`Tolerance::loosest`].
///
/// Memoized per node, for the same reason [`eval_scalar`]'s walk is: the graph
/// is a **DAG**, and mask-valuedness composes over shared children, so a naive
/// recursion re-derives every shared node once per reference — exponential in
/// nesting depth, not linear in node count. `BitAnd(n, n)` chains are the
/// realistic shape (e-graph extraction shares aggressively), and unmemoized
/// they cost `2^depth`: a **29-node** graph measured ~1.2s, a 40-deep one is
/// weeks. That is an unbounded hang inside a correctness gate, which is worse
/// than any wrong answer it could have given.
#[must_use]
pub fn is_mask_valued(root: Node<'_, ExprData>) -> bool {
    /// Mask-valuedness of a node whose children are already resolved, or
    /// `None` for the composing shapes that need their children first.
    fn leaf_verdict(node: Node<'_, ExprData>) -> Option<bool> {
        match (*node, node.child_count()) {
            (ExprData::Op(OpKind::BitAnd | OpKind::BitOr), 2)
            | (ExprData::Op(OpKind::Select), 3) => None,
            (ExprData::Op(op), 2) => Some(matches!(
                op,
                OpKind::Lt | OpKind::Le | OpKind::Gt | OpKind::Ge | OpKind::Eq | OpKind::Ne
            )),
            // A literal all-zeros / all-ones lane IS a mask; anything else is a
            // number. See the doc comment for why this is the safe direction.
            (ExprData::Const(bits), _) => Some(is_valid_mask(bits)),
            _ => Some(false),
        }
    }

    /// The two operands that decide a composing node's verdict. `Select`'s
    /// condition is a stencil, not part of the output domain, so it is
    /// deliberately not among them.
    fn deciders<'a>(node: Node<'a, ExprData>) -> (Node<'a, ExprData>, Node<'a, ExprData>) {
        let mut kids = node.children();
        if node.child_count() == 3 {
            let _cond = kids.next();
        }
        let a = kids.next().expect("a composing node has two deciders");
        let b = kids.next().expect("a composing node has two deciders");
        (a, b)
    }

    enum Task<'a> {
        Visit(Node<'a, ExprData>),
        Emit(Node<'a, ExprData>),
    }

    let mut memo = root.dag().side_table(None);
    let mut stack = vec![Task::Visit(root)];
    while let Some(task) = stack.pop() {
        match task {
            Task::Visit(node) => {
                if memo[node].is_some() {
                    continue;
                }
                if let Some(verdict) = leaf_verdict(node) {
                    memo[node] = Some(verdict);
                    continue;
                }
                let (a, b) = deciders(node);
                stack.push(Task::Emit(node));
                stack.push(Task::Visit(a));
                stack.push(Task::Visit(b));
            }
            Task::Emit(node) => {
                let (a, b) = deciders(node);
                let resolved = |child: Node<'_, ExprData>| {
                    memo[child].expect("post-order walk resolves children before their parent")
                };
                memo[node] = Some(resolved(a) && resolved(b));
            }
        }
    }
    memo[root].expect("the root is always resolved by the walk")
}

/// Outcome of checking a mask-valued root lane (JIT `got` vs oracle `want`).
///
/// Three-way on purpose — the two failure shapes mean different things and
/// must not be conflated:
///
/// - [`InvalidPattern`](Self::InvalidPattern) is a **hard bug** regardless of
///   any child tolerance: some tier wrote a lane that is not a mask at all, so
///   every downstream bitwise consumer (`Select`, `BitAnd`) is corrupted.
/// - [`Disagree`](Self::Disagree) is two tiers each producing a *valid* mask
///   with opposite verdicts. That is the near-threshold case: when the
///   comparison's operands sit within the children's numeric tolerance of each
///   other, the language promises no particular verdict (per-op drift may land
///   the operands on opposite sides). Callers record it as ill-conditioned
///   metadata, never as a run-killing miscompile.
/// - [`Agree`](Self::Agree) is bit-exact agreement between two valid masks —
///   the only passing outcome.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskComparison {
    /// Both lanes are valid masks and bit-identical.
    Agree,
    /// Both lanes are valid masks with opposite verdicts — near-threshold
    /// contract behavior, classify via the operands' numeric tolerance.
    Disagree,
    /// At least one lane is not `0x00000000`/`0xFFFFFFFF` — a miscompile, no
    /// tolerance applies. The flags say which side is broken.
    InvalidPattern {
        /// Whether the JIT lane was a valid mask pattern.
        got_is_mask: bool,
        /// Whether the oracle lane was a valid mask pattern.
        want_is_mask: bool,
    },
}

/// Root-aware acceptance for a mask-valued root: both lanes must be valid mask
/// patterns ([`is_valid_mask`]) and equality between them is bit-exact.
///
/// This is the check that [`Tolerance::loosest`] folding cannot express — a
/// numeric child (say `Lt(Sin(x), y)`) loosens the whole-expression tolerance
/// to `Numeric`, whose NaN-vs-NaN allowance would wave through a `0x7fc00000`
/// where the hardware owed `0xFFFFFFFF`. Route the root's lane here whenever
/// [`is_mask_valued`] holds; see [`MaskComparison`] for the three outcomes.
#[must_use]
pub fn compare_mask_root(got: f32, want: f32) -> MaskComparison {
    let got_is_mask = is_valid_mask(got.to_bits());
    let want_is_mask = is_valid_mask(want.to_bits());
    if !got_is_mask || !want_is_mask {
        return MaskComparison::InvalidPattern {
            got_is_mask,
            want_is_mask,
        };
    }
    if got.to_bits() == want.to_bits() {
        MaskComparison::Agree
    } else {
        MaskComparison::Disagree
    }
}

/// Whether `v` is an input on which `TruncToInt` is **platform-divergent** —
/// NaN or outside `[-2³¹, 2³¹)` — so a differential gate must classify the
/// point ill-conditioned rather than compare it.
///
/// The divergence is real hardware, not a tier bug, so nothing here is to fix:
/// x86 `cvttps2dq` writes the "integer indefinite" `0x80000000` for every
/// invalid input, aarch64 `FCVTZS` saturates (`0x7FFFFFFF` / `0x80000000`) and
/// sends NaN to `0`, and the oracle's `eval_unary` uses Rust's saturating
/// `as i32` (NaN → `0`). Same-form oracle-vs-JIT comparison at such a point
/// would report a valid backend as a miscompile. Emulating per-target
/// semantics in the oracle is explicitly not on offer — the language promises
/// no value here, exactly like `Min`/`Max` on NaN.
///
/// Negative overflow (`v < -2³¹`) happens to agree on every current target
/// (all produce `0x80000000`), but that is coincidence, not contract, so it is
/// flagged too. `v == -2³¹` is exactly representable and in-range: not
/// flagged. `TruncToInt` is the only float→int op in the `OpKind` inventory
/// (`IntToFloat` is int→float and exact everywhere), so the predicate covers
/// the whole class.
#[must_use]
pub fn trunc_input_is_divergent(v: f32) -> bool {
    /// `2³¹` is exactly representable in f32; the greatest in-range value is
    /// the f32 below it, and the least is `-2³¹` itself.
    const TWO_POW_31: f32 = 2_147_483_648.0;
    // NaN fails `contains` on its own, but spelling it out keeps the doc's
    // "NaN or outside [-2³¹, 2³¹)" and this line saying the same thing.
    v.is_nan() || !(-TWO_POW_31..TWO_POW_31).contains(&v)
}

/// Per-op allowance for JIT-vs-oracle equivalence checks — how far a *correct*
/// backend may drift from the oracle on **one instruction**.
///
/// **This table is not an expression-level gate.** Composition amplifies, and
/// by an input-dependent factor, so a differential check over a whole graph
/// belongs to [`DifferentialCheck`] — which seeds from the same estimate widths
/// documented below and propagates them along the evaluation path. What the
/// table is still good for is what it says: the per-op conformance rows, and
/// the documentation of where each op's inexactness comes from.
///
/// **Caller protocol for platform-divergent inputs:** the table cannot describe
/// inputs where the language promises nothing at all (`Min`/`Max` on NaN or
/// opposite-signed zeros, `Gt`/`Ge` on NaN, `Round` ties, NaN/out-of-range
/// `TruncToInt` — the CLAUDE.md contract tables). [`op_is_divergent_at`] is the
/// predicate for those, and [`PointCheck::is_platform_divergent`] applies it
/// along the path a point actually takes.
///
/// Rationale per class:
/// - **Bit-pattern ops** (`is_bitwise_domain`) and exact instructions
///   (`Neg`/`Abs` are sign-bit ops; `Floor`/`Ceil`/`Round`/`Min`/`Max` are
///   single exact instructions; loads load): [`Tolerance::BitExact`].
/// - **Correctly-rounded arithmetic** (`Add`/`Sub`/`Mul`/`Div`/`Sqrt`): each
///   instruction is IEEE-exact and identical to the oracle's scalar op, but an
///   extracted/rewritten form may associate differently, so a few ulps of
///   headroom (`rel 1e-5`) rather than bit equality.
/// - **`MulAdd`**: one rounding in the oracle (`libm::fmaf`, matching
///   `OpKind::eval_ternary` and every FMA target); two where a target
///   multiplies then adds. Away from catastrophic cancellation they differ
///   by at most an ulp, and [`PointCheck`]'s radius carries the product's
///   rounding so the cancelling inputs are bounded rather than skipped.
/// - **`Recip`/`Rsqrt`**: hardware *estimates*. The loosest backend is
///   SSE2/AVX2 `rcpps`/`vrcpps` at ~12 bits (max relative error `1.5·2⁻¹²`
///   ≈ 3.7e-4); `vrcp14ps` is ~14 bits and aarch64's `FRECPE`+`FRECPS`
///   refinement is tighter still. The oracle computes the exact value, so the
///   allowance is the estimate's documented error band.
/// - **Transcendentals**: both tiers evaluate the same polynomial expansion,
///   but the atan family routes through `Recip` and future expansions may fuse
///   through FMA, so these get the empirical band already pinned by
///   `tests/transcendental_jit.rs` (`rel 1e-3` / `abs 1e-4`).
///
/// # Panics
///
/// Panics on ops that must never reach a correctness comparison: `Dwrt` (must
/// be rewritten away before any backend) and `Tuple` (structural, not a lane
/// value). A new `OpKind` with no row here is a compile error, not a default.
#[must_use]
pub fn equivalence_tolerance(op: OpKind) -> Tolerance {
    /// See "Correctly-rounded arithmetic" above.
    const EXACT_ARITH: Tolerance = Tolerance::Numeric {
        rel: 1e-5,
        abs: 1e-6,
    };
    /// See "`Recip`/`Rsqrt`" above: 12-bit `rcpps` floor plus margin.
    const ESTIMATE: Tolerance = Tolerance::Numeric {
        rel: 5e-4,
        abs: 1e-6,
    };
    /// See "Transcendentals" above.
    const TRANSCENDENTAL: Tolerance = Tolerance::Numeric {
        rel: 1e-3,
        abs: 1e-4,
    };

    match op {
        // Leaves and loads: values pass through untouched.
        OpKind::Var
        | OpKind::Const
        | OpKind::Buffer
        | OpKind::Uniform
        | OpKind::Param
        | OpKind::Gather
        | OpKind::RawGather => Tolerance::BitExact,
        // Bit-pattern domain: masks, blends, and the integer primitives.
        OpKind::Lt
        | OpKind::Le
        | OpKind::Gt
        | OpKind::Ge
        | OpKind::Eq
        | OpKind::Ne
        | OpKind::Select
        | OpKind::BitAnd
        | OpKind::BitOr
        | OpKind::TruncToInt
        | OpKind::IntToFloat
        | OpKind::IAdd
        | OpKind::Shl
        | OpKind::Shr => Tolerance::BitExact,
        // Exact single instructions on every backend.
        OpKind::Neg
        | OpKind::Abs
        | OpKind::Floor
        | OpKind::Ceil
        | OpKind::Round
        | OpKind::Min
        | OpKind::Max => Tolerance::BitExact,
        // The fold itself combines with an exact monoid op in the same order
        // the unrolled lowering does; the body's ops carry their own rows.
        OpKind::Reduce => Tolerance::BitExact,
        OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div | OpKind::Sqrt | OpKind::MulAdd => {
            EXACT_ARITH
        }
        OpKind::Recip | OpKind::Rsqrt => ESTIMATE,
        OpKind::Sin
        | OpKind::Cos
        | OpKind::Tan
        | OpKind::Asin
        | OpKind::Acos
        | OpKind::Atan
        | OpKind::Atan2
        | OpKind::Exp
        | OpKind::Exp2
        | OpKind::Ln
        | OpKind::Log2
        | OpKind::Log10
        | OpKind::Pow => TRANSCENDENTAL,
        OpKind::Dwrt => panic!(
            "equivalence_tolerance: dwrt must be rewritten away before any backend — \
             a graph reaching a correctness check with one is a compile bug"
        ),
        OpKind::Tuple => panic!(
            "equivalence_tolerance: tuple is structural, not a lane value — \
             it has no scalar result to compare"
        ),
    }
}

// ═══════════════════════ compositional error bound ═══════════════════════════
//
// Why the per-op table above is not, on its own, an acceptance rule.
//
// `equivalence_tolerance` says how far ONE instruction may drift. Folding it
// over an expression with `Tolerance::loosest` (max-over-ops) asserts that the
// whole expression drifts no more than its worst single op — which is false the
// moment an inexact value is used again. Two measurements, one from a corpus
// run and one from external review, say the same thing:
//
//   * `recip(x)^16` at `x = -1`: `rcpps` is a ~12-bit estimate, so the JIT holds
//     `-1 · (1 ± 3.7e-4)`; four squarings multiply that relative width by 16,
//     giving ~5.9e-3 against a folded allowance of 5e-4. The kernel is correct
//     and the fold calls it a miscompile.
//   * `sin(K · Y²)` with `K` from a `Recip`: the same ~4e-6 seed measured on
//     NEON is harmless at argument 0.83, is 3.6e-3 at 8.3e3, and is a relative
//     error of 1.0 at 8.3e7 — the argument multiplies the seed. Folding sees an
//     expression-independent 1e-3 at every one of those points.
//
// Error does not fold, it *composes*, and it composes differently at every
// input. So the bound below is computed per point, alongside the value, by
// carrying an interval through the walk: each node gets a radius, seeded at the
// estimate instructions and widened by each op's own rounding, and the JIT lane
// is accepted iff it lands inside `value ± radius`. Cancellation and
// amplification both come out of that for free — `a - b` keeps its operands'
// absolute radii while `|value|` collapses, so the *relative* bound explodes,
// which is precisely the "ill-conditioned" signal a ratio heuristic tries and
// fails to guess at.

/// Half an ulp of `f32`: the relative width one correctly-rounded IEEE-754
/// single-precision operation adds to an already-inexact operand.
const ROUND_REL: f32 = f32::EPSILON * 0.5;

/// Relative width of the reciprocal **estimate** instructions, the only place
/// this crate's arithmetic is inexact by design. The loosest first-class
/// backend is SSE2/AVX2 `rcpps`/`rsqrtps` at ~12 bits — a maximum relative
/// error of `1.5·2⁻¹²` — while `vrcp14ps` (~14 bits) and aarch64's
/// `FRECPE` + one `FRECPS` step (measured ~4e-6) are tighter. A portable bound
/// must cover the loosest, so this is the seed for `Recip` and `Rsqrt`.
///
/// The cost is sensitivity on the tighter machines: on aarch64 the seed is ~90x
/// wider than the hardware's actual error, and through an amplifying expression
/// that gap is amplified too. Narrowing it to the host's own instruction would
/// buy that sensitivity back — and make the same corpus accept on one machine
/// and reject on another, which is a worse property for a shared gate than a
/// conservative bound. Widening never rejects a valid kernel; narrowing can.
const ESTIMATE_REL: f32 = 1.5 / 4096.0;

/// Relative bound at which a point stops being usable for comparing two
/// algebraically-equal *forms* against each other: below it the bound still
/// pins roughly three decimal digits, which is the same resolution the
/// transcendental row of [`equivalence_tolerance`] claims.
const WELL_CONDITIONED_REL: f32 = 1e-3;

/// Absolute floor under [`WELL_CONDITIONED_REL`], so a result that is
/// legitimately near zero is not declared ill-conditioned by a relative test
/// that divides by it. It gates *eligibility only* — [`PointCheck::verdict`]
/// never uses it, so a wrong answer near zero is still a rejection.
const WELL_CONDITIONED_ABS: f32 = 1e-6;

/// Absolute slack in [`PointCheck::verdict`] that absorbs a backend flushing
/// denormals to zero. Below the smallest normal `f32` the two tiers are not
/// arguing about arithmetic.
const DENORMAL_SLACK: f32 = f32::MIN_POSITIVE;

/// Whether `op` applied to these concrete argument *values* lands on one of the
/// rows where the language promises nothing, so a differential gate must skip
/// the point rather than compare it.
///
/// Two adjustments to [`OpKind::fold_is_platform_specific`], and both matter:
///
/// - `Recip`/`Rsqrt` flag there *unconditionally* (an estimate is never equal
///   to the exact value the folder computes), but for a runtime check the
///   estimate width is the contract — [`ESTIMATE_REL`] — so they are checked,
///   not skipped. Skipping them would discard most of a real corpus.
/// - `TruncToInt` has no row there because folding it is not the issue; at
///   runtime NaN/out-of-range float→int is target-defined (x86's integer
///   indefinite vs aarch64 saturation vs the oracle's Rust `as i32`), so
///   [`trunc_input_is_divergent`] supplies the missing row.
///
/// This is the whole divergence policy; a differential gate should call
/// [`PointCheck::is_platform_divergent`] rather than re-deriving it, since only
/// the walk knows which `Select` branch the point actually took.
#[must_use]
pub fn op_is_divergent_at(op: OpKind, args: &[f32]) -> bool {
    if op == OpKind::TruncToInt {
        return trunc_input_is_divergent(args[0]);
    }
    !matches!(op, OpKind::Recip | OpKind::Rsqrt) && op.fold_is_platform_specific(args)
}

/// The all-ones lane: the pattern a true comparison produces on every backend.
const MASK_TRUE: f32 = f32::from_bits(0xFFFF_FFFF);

/// The all-zeros lane: the pattern a false comparison produces.
const MASK_FALSE: f32 = 0.0;

/// The pattern a comparison produces when it answers the other way. Only
/// meaningful on a valid mask lane, which is the only place it is called from.
fn opposite_mask(value: f32) -> f32 {
    debug_assert!(is_valid_mask(value.to_bits()));
    if value.to_bits() == 0 {
        MASK_TRUE
    } else {
        MASK_FALSE
    }
}

/// One node's contribution to the walk: the oracle's value, the absolute radius
/// a conforming backend may differ by, and the "no promise" flags.
///
/// # The indeterminacy invariant
///
/// `indeterminate` says "a comparison on this point's path may legally answer
/// either way, and this node's value depends on which way it went". Two
/// carriers describe that freedom, and **exactly one of them is live at a
/// time**:
///
/// - `alt.is_some()` — the node has exactly two legal answers, both exact, and
///   `alt` is the other one; `radius` is `0`. This is the only honest form for
///   a lane pattern (a radius cannot describe an all-ones lane, which reads as
///   NaN) and the sharpest one for its numeric conversions.
/// - `alt.is_none()` — **`radius` covers every value a conforming backend may
///   produce**, including the ones reached by resolving the comparison the
///   other way.
///
/// [`over_choices`] is what maintains the second half, and it is not a nicety:
/// [`PointCheck::verdict`] and [`PointCheck::is_well_conditioned`] read
/// `radius`, so a numeric node that carried indeterminacy as a *flag* on a zero
/// radius would report a legally-opposite backend answer as a miscompile. That
/// was the `IntToFloat(Lt(Recip(x), k))` false positive: `-1.0` and `0.0` are
/// both legal there, and nothing downstream read the flag.
#[derive(Clone, Copy, Debug)]
struct NodeBound {
    value: f32,
    /// Absolute error bound, `>= 0`; `INFINITY` means unbounded (the interval
    /// crossed a pole, a domain edge, or a bit-pattern op with inexact input).
    radius: f32,
    /// A platform-divergent op was reached along the path this point actually
    /// takes — `Select`'s unchosen branch is deliberately not on that path.
    divergent: bool,
    /// The node's *other* legal value, when it has exactly two and neither
    /// carries error: a comparison whose operands straddle within their radii
    /// (the two lane patterns), or anything exact computed from one. See the
    /// type's invariant.
    alt: Option<f32>,
    /// This node's value depends on a comparison that may legally answer either
    /// way. Described exactly by `alt` when there are two answers, covered
    /// conservatively by `radius` otherwise.
    indeterminate: bool,
}

impl NodeBound {
    /// A value both tiers compute identically: same inputs, same instruction.
    const fn exact(value: f32) -> Self {
        Self {
            value,
            radius: 0.0,
            divergent: false,
            alt: None,
            indeterminate: false,
        }
    }

    /// Build a bound, normalizing the radius. Every arithmetic rule below can
    /// produce a NaN radius from an infinite operand (`0 · ∞`, `∞ − ∞`), and a
    /// NaN radius silently accepts nothing and rejects nothing — the honest
    /// reading is "unbounded".
    fn new(value: f32, radius: f32, divergent: bool, indeterminate: bool) -> Self {
        Self {
            value,
            radius: if radius.is_nan() {
                f32::INFINITY
            } else {
                radius.max(0.0)
            },
            divergent,
            alt: None,
            indeterminate,
        }
    }

    /// A node with exactly two legal answers and no error bar on either:
    /// `value` is what the oracle computed, `other` is what the opposite
    /// resolution of an indeterminate comparison produces.
    fn choice(value: f32, other: f32, divergent: bool) -> Self {
        Self {
            value,
            radius: 0.0,
            divergent,
            alt: Some(other),
            indeterminate: true,
        }
    }

    /// This bound with one of its legal values pinned: the freedom is spent, so
    /// `alt` is cleared, but the fact that something upstream was unpinned
    /// stays on the flag.
    fn pinned(self, choice: f32) -> Self {
        debug_assert!(
            self.alt.is_none() || self.radius == 0.0,
            "a pattern with a choice must carry no radius"
        );
        Self {
            value: choice,
            alt: None,
            ..self
        }
    }

    /// Lower end of this node's interval.
    fn lo(self) -> f32 {
        self.value - self.radius
    }

    /// Upper end of this node's interval.
    fn hi(self) -> f32 {
        self.value + self.radius
    }
}

/// The most arguments any bound rule takes; `over_choices` enumerates `2^N`
/// resolutions, so this is also what keeps that enumeration small.
const MAX_BOUND_ARGS: usize = 3;

/// Re-run an op's error rule once per legal resolution of its unpinned pattern
/// operands, then hull the answers into one bound.
///
/// **This is the mask→number boundary**, and the whole reason it exists is that
/// the boundary used to lose information. `Lt(Recip(x), k)` near the threshold
/// is a pattern with `radius == 0` and `indeterminate == true`; feeding it to
/// `IntToFloat` hit the exact-in/exact-out fast path, which copied the flag and
/// kept the zero radius. Nothing numeric reads the flag, so a backend that
/// picked the *other* legal mask — and therefore returned `0.0` where the
/// oracle had `-1.0` — was reported as a same-form miscompile.
///
/// Carrying the answers rather than un-bounding the point is deliberate:
/// `IntToFloat` of the two masks is `-1.0` or `0.0` and nothing else, so the
/// point stays checkable (a backend answering `5.0` — or `-0.5` — is still a
/// miscompile) instead of degrading to metadata. Boundedness is precious: an
/// expression whose every point is unbounded passes vacuously.
///
/// # Panics
///
/// Panics if handed more than [`MAX_BOUND_ARGS`] operands.
fn over_choices(args: &[NodeBound], rule: &dyn Fn(&[NodeBound]) -> NodeBound) -> NodeBound {
    assert!(
        args.len() <= MAX_BOUND_ARGS,
        "over_choices: {} operands exceeds MAX_BOUND_ARGS",
        args.len()
    );
    if args.iter().all(|a| a.alt.is_none()) {
        return rule(args);
    }

    let mut corners = [NodeBound::exact(0.0); 1 << MAX_BOUND_ARGS];
    let mut count = 0usize;
    let mut pinned = [NodeBound::exact(0.0); MAX_BOUND_ARGS];
    // Corner 0 takes every operand's own value, so it IS the oracle's answer;
    // the hull below reads it as the nominal.
    for combo in 0..(1usize << args.len()) {
        let mut representable = true;
        for (i, arg) in args.iter().enumerate() {
            let take_alt = combo & (1 << i) != 0;
            pinned[i] = match (take_alt, arg.alt) {
                (false, _) => arg.pinned(arg.value),
                (true, Some(other)) => arg.pinned(other),
                // This operand has no second choice; the corner duplicates one
                // already enumerated.
                (true, None) => {
                    representable = false;
                    break;
                }
            };
        }
        if !representable {
            continue;
        }
        corners[count] = rule(&pinned[..args.len()]);
        count += 1;
    }
    hull_of_choices(&corners[..count])
}

/// Collapse the per-resolution answers of [`over_choices`] into one bound.
/// `corners[0]` is the nominal (every operand at its own value).
fn hull_of_choices(corners: &[NodeBound]) -> NodeBound {
    let nominal = *corners
        .first()
        .expect("over_choices always enumerates the nominal corner");
    let divergent = corners.iter().any(|c| c.divergent);

    // A corner may be an unpinned pattern in its own right (a comparison whose
    // pinned operands still straddle), so its second answer is a candidate too.
    let mut candidates = [(0.0f32, 0.0f32); 2 * (1 << MAX_BOUND_ARGS)];
    let mut count = 0usize;
    for corner in corners {
        candidates[count] = (corner.value, corner.radius);
        count += 1;
        if let Some(other) = corner.alt {
            candidates[count] = (other, 0.0);
            count += 1;
        }
    }
    let candidates = &candidates[..count];

    // Two exact answers stay *two answers*, not the interval between them: a
    // backend landing anywhere in the middle computed something neither legal
    // resolution produces, and an interval would wave that through. It is also
    // the only honest form for a mask — the all-ones lane reads as NaN, and
    // `NaN - 0.0` is not an error bar.
    let other = candidates
        .iter()
        .map(|(v, _)| *v)
        .find(|v| v.to_bits() != nominal.value.to_bits());
    let exact = candidates.iter().all(|(_, r)| *r == 0.0);
    match other {
        // Every resolution agrees: the freedom never reached this node.
        None if exact => {
            return NodeBound {
                divergent,
                ..nominal
            };
        }
        Some(other) if exact => {
            let two_valued = candidates.iter().all(|(v, _)| {
                v.to_bits() == nominal.value.to_bits() || v.to_bits() == other.to_bits()
            });
            if two_valued {
                return NodeBound::choice(nominal.value, other, divergent);
            }
        }
        _ => {}
    }

    // Three or more answers, or answers that carry their own error bars: the
    // radius has to reach every candidate's whole interval. Written as an
    // explicit loop because `f32::max` *ignores* a NaN operand, which would
    // drop exactly the candidate that escaped the domain.
    let mut radius = 0.0f32;
    for (value, corner_radius) in candidates {
        let reach = (value - nominal.value).abs() + corner_radius;
        if reach.is_nan() {
            radius = f32::INFINITY;
            break;
        }
        if reach > radius {
            radius = reach;
        }
    }
    NodeBound::new(nominal.value, radius, divergent, true)
}

/// Radius about `value` of the interval `[lo, hi]`. A NaN endpoint means the
/// interval escaped the op's domain, which is unbounded, not zero.
fn radius_of(lo: f32, hi: f32, value: f32) -> f32 {
    if lo.is_nan() || hi.is_nan() || value.is_nan() {
        return f32::INFINITY;
    }
    (hi - value).max(value - lo).max(0.0)
}

/// What a uniform evaluates to: the block's value for its slot, or the
/// declared default when the table binds none — the same rule a bake without
/// a block follows, so the oracle and the JIT agree on it by construction.
fn uniform_value(env: &Environment, bindings: &BindingTable<'_>, u: UniformId) -> f32 {
    bindings.uniform(u).unwrap_or(env.uniform(u).default)
}

/// The children whose *values* feed a node. Identical to `Node::children`
/// except that a `Buffer` leaf — which is a name, not a value — is dropped from
/// `Gather`/`RawGather`, exactly as [`eval_scalar`] handles them.
///
/// # Panics
///
/// Panics on the node shapes the bound walk does not model: `Param` (substitute
/// first), a bare `Buffer`, and `Reduce` (a fold rebinds its body per
/// iteration, so a flat per-node memo cannot represent it — expand the reduce
/// first).
fn value_children<'a>(node: Node<'a, ExprData>) -> ([Option<Node<'a, ExprData>>; 3], usize) {
    fn take<'a>(
        kids: &mut impl Iterator<Item = Node<'a, ExprData>>,
        n: usize,
    ) -> ([Option<Node<'a, ExprData>>; 3], usize) {
        let mut out = [None; 3];
        for slot in out.iter_mut().take(n) {
            *slot = kids.next();
        }
        (out, n)
    }

    let mut kids = node.children();
    match (*node, node.child_count()) {
        (ExprData::Var(_) | ExprData::Const(_) | ExprData::Uniform(_), _) => ([None; 3], 0),
        (ExprData::Param(p), _) => panic!("PointCheck: Param({p}) — substitute params first"),
        (ExprData::Buffer(b), _) => panic!(
            "PointCheck: bare Buffer({}) is not a value; read it through Gather",
            b.0
        ),
        // The buffer leaf is the first child of either gather form; skip it.
        (ExprData::Op(OpKind::RawGather), 2) => {
            let _buf = kids.next();
            take(&mut kids, 1)
        }
        (ExprData::Op(OpKind::Gather), 3) => {
            let _buf = kids.next();
            take(&mut kids, 2)
        }
        (ExprData::Op(_), n @ 1..=3) => take(&mut kids, n),
        (ExprData::Op(op), n) => panic!(
            "PointCheck: {op:?} with {n} children — a reduction rebinds its body per \
             iteration, which a per-node bound cannot represent; expand_reduce first"
        ),
    }
}

/// A prepared differential check for one term: the transcendental expansion and
/// the root's mask-valuedness are computed once, then [`DifferentialCheck::at`]
/// is called per grid point.
///
/// Splitting it this way is not a micro-optimization — the expansion rebuilds
/// the whole graph, and a gate that ran it per point would pay that on every
/// one of its check points.
pub struct DifferentialCheck {
    expanded: Rooted<ExprData>,
    env: Environment,
    mask_root: bool,
}

impl DifferentialCheck {
    /// Prepare a check. `term` is what the JIT compiles; the expansion
    /// performed here is the same one the emitter performs, so the walk and the
    /// machine code are interpreting one semantics.
    #[must_use]
    pub fn new(term: Term<'_>) -> Self {
        // Mask-valuedness is structural and read off the UNEXPANDED graph:
        // that is the graph whose root defines the output's domain.
        let mask_root = is_mask_valued(term.root());
        Self {
            expanded: crate::passes::expand_transcendentals(term),
            env: term.env().clone(),
            mask_root,
        }
    }

    /// The expanded term this check evaluates.
    fn term(&self) -> Term<'_> {
        Term::new(self.expanded.entry(), &self.env)
    }

    /// Whether the root produces a mask lane. Decides which acceptance method
    /// applies: [`PointCheck::classify_mask_root`] when true,
    /// [`PointCheck::verdict`] when false.
    #[must_use]
    pub fn root_is_mask_valued(&self) -> bool {
        self.mask_root
    }

    /// Evaluate at one point, carrying the composed error bound.
    ///
    /// The walk is iterative and memoized per node for the reason every walk in
    /// this module is: the graph is a **DAG** with heavy sharing, and a
    /// recursive re-derivation costs `2^depth`.
    ///
    /// # Panics
    ///
    /// Panics on the shapes [`value_children`] refuses (`Param`, bare `Buffer`,
    /// a reduction), and on an op with no scalar evaluation (lower it first).
    #[must_use]
    pub fn at(&self, vars: &[f32; COORD_AXES], bindings: &BindingTable<'_>) -> PointCheck {
        enum Task<'a> {
            Visit(Node<'a, ExprData>),
            Emit(Node<'a, ExprData>),
        }

        let term = self.term();
        let mut bounds = term.dag().side_table(None);
        let mut stack = vec![Task::Visit(term.root())];
        while let Some(task) = stack.pop() {
            match task {
                Task::Visit(node) => {
                    if bounds[node].is_some() {
                        continue;
                    }
                    let (kids, n) = value_children(node);
                    stack.push(Task::Emit(node));
                    for kid in kids.into_iter().take(n).flatten() {
                        stack.push(Task::Visit(kid));
                    }
                }
                Task::Emit(node) => {
                    if bounds[node].is_some() {
                        continue;
                    }
                    let bound = self.node_bound(node, vars, bindings, &bounds);
                    bounds[node] = Some(bound);
                }
            }
        }

        let root = bounds[term.root()].expect("the walk resolves the root");
        PointCheck {
            value: root.value,
            radius: root.radius,
            divergent: root.divergent,
            alt: root.alt,
            indeterminate: root.indeterminate,
            mask_root: self.mask_root,
        }
    }

    /// Value + bound for one node, given its already-resolved children.
    fn node_bound(
        &self,
        node: Node<'_, ExprData>,
        vars: &[f32; COORD_AXES],
        bindings: &BindingTable<'_>,
        bounds: &SideTable<Option<NodeBound>>,
    ) -> NodeBound {
        let kids: Vec<Node<'_, ExprData>> = node.children().collect();
        let child = |i: usize| -> NodeBound {
            bounds[kids[i]].expect("post-order resolves children before their parent")
        };
        match (*node, kids.len()) {
            (ExprData::Var(i), _) => {
                let i = i as usize;
                assert!(
                    i < COORD_AXES,
                    "PointCheck: Var({i}) — {}",
                    if RETIRED_COORD_AXES.contains(&(i as u8)) {
                        "a retired coordinate axis; a per-call scalar is a Uniform"
                    } else {
                        "a reduction index outside a Reduce"
                    }
                );
                NodeBound::exact(vars[i])
            }
            (ExprData::Const(bits), _) => NodeBound::exact(f32::from_bits(bits)),
            (ExprData::Uniform(u), _) => NodeBound::exact(uniform_value(&self.env, bindings, u)),
            (ExprData::Op(OpKind::RawGather), 2) => {
                self.gather_bound(kids[0], &[child(1)], bindings, GatherKind::Raw)
            }
            (ExprData::Op(OpKind::Gather), 3) => self.gather_bound(
                kids[0],
                &[child(1), child(2)],
                bindings,
                GatherKind::Clamped,
            ),
            (ExprData::Op(OpKind::Select), 3) => select_bound(child(0), child(1), child(2)),
            (ExprData::Op(op), 1) => unary_bound(op, child(0)),
            (ExprData::Op(op), 2) => binary_bound(op, child(0), child(1)),
            (ExprData::Op(op), 3) => ternary_bound(op, child(0), child(1), child(2)),
            (other, n) => {
                panic!("PointCheck: {other:?} with {n} children has no bound (screen it out first)")
            }
        }
    }

    /// A memory read is exact — unless the composed error on the index could
    /// select a different cell, in which case the value is not bounded at all.
    fn gather_bound(
        &self,
        buf: Node<'_, ExprData>,
        idx: &[NodeBound],
        bindings: &BindingTable<'_>,
        kind: GatherKind,
    ) -> NodeBound {
        over_choices(idx, &|pinned| {
            self.gather_bound_pinned(buf, pinned, bindings, kind)
        })
    }

    /// [`gather_bound`](Self::gather_bound) with every index resolved to one
    /// legal value.
    fn gather_bound_pinned(
        &self,
        buf: Node<'_, ExprData>,
        idx: &[NodeBound],
        bindings: &BindingTable<'_>,
        kind: GatherKind,
    ) -> NodeBound {
        let id = match *buf {
            ExprData::Buffer(id) => id,
            other => {
                panic!("PointCheck: Gather's buffer child must be a Buffer leaf, got {other:?}")
            }
        };
        let decl = self.env.buffer(id);
        let data = bindings.slot(id);
        let divergent = idx.iter().any(|b| b.divergent);

        let cell = |pick: fn(NodeBound) -> f32| -> usize {
            match kind {
                GatherKind::Raw => libm::floorf(pick(idx[0])) as usize,
                GatherKind::Clamped => {
                    let max_x = decl.width.saturating_sub(1) as i64;
                    let max_y = decl.height.saturating_sub(1) as i64;
                    let xi = (libm::floorf(pick(idx[0])) as i64).clamp(0, max_x);
                    let yi = (libm::floorf(pick(idx[1])) as i64).clamp(0, max_y);
                    yi as usize * decl.width as usize + xi as usize
                }
            }
        };

        let value = data[cell(|b| b.value)];
        // The read itself adds nothing; the question is whether the index is
        // pinned. If the index interval spans two cells, the backend may read
        // either, and no radius describes "some other element of the buffer".
        let pinned = cell(NodeBound::lo) == cell(NodeBound::hi);
        NodeBound::new(
            value,
            if pinned { 0.0 } else { f32::INFINITY },
            divergent,
            idx.iter().any(|b| b.indeterminate),
        )
    }
}

/// Which index convention a memory read uses.
#[derive(Clone, Copy)]
enum GatherKind {
    /// `Gather`: floor + clamp per axis, row-major.
    Clamped,
    /// `RawGather`: an already-clamped linear index.
    Raw,
}

/// Evaluate `op`, panicking with the same message [`eval_scalar`] uses.
fn eval_unary_or_panic(op: OpKind, x: f32) -> f32 {
    op.eval_unary(x).unwrap_or_else(|| {
        panic!(
            "PointCheck: no scalar eval for unary {op:?} — lower it first (expand_transcendentals)"
        )
    })
}

fn unary_bound(op: OpKind, a: NodeBound) -> NodeBound {
    over_choices(&[a], &|args| unary_bound_pinned(op, args[0]))
}

fn unary_bound_pinned(op: OpKind, a: NodeBound) -> NodeBound {
    let value = eval_unary_or_panic(op, a.value);
    let divergent = a.divergent || op_is_divergent_at(op, &[a.value]);
    // Exact in, exact out: with identical inputs both tiers run the same
    // instruction and round identically. Only the estimates break it.
    //
    // "Identical inputs" is what `over_choices` guarantees before calling here:
    // an operand that is an unpinned mask arrives pinned to one resolution, and
    // the other resolution is a second call whose answer gets hulled in. That
    // is what stops this fast path from converting indeterminacy into a zero
    // radius on a numeric node.
    if a.radius == 0.0 && !matches!(op, OpKind::Recip | OpKind::Rsqrt) {
        return NodeBound::new(value, 0.0, divergent, a.indeterminate);
    }
    let radius = match op {
        // Sign manipulation moves the interval, never widens it.
        OpKind::Neg | OpKind::Abs => a.radius,
        // Monotone and exact: map the endpoints, add nothing.
        OpKind::Floor | OpKind::Ceil | OpKind::Round => {
            let lo = eval_unary_or_panic(op, a.lo());
            let hi = eval_unary_or_panic(op, a.hi());
            radius_of(lo, hi, value)
        }
        // Monotone and correctly rounded.
        OpKind::Sqrt => {
            let lo = eval_unary_or_panic(op, a.lo());
            let hi = eval_unary_or_panic(op, a.hi());
            radius_of(lo, hi, value) + value.abs() * ROUND_REL
        }
        // The estimate seed, on top of the input's own width. `1/x` is monotone
        // away from the pole, so an interval containing zero is unbounded.
        OpKind::Recip => {
            if a.lo() <= 0.0 && a.hi() >= 0.0 {
                f32::INFINITY
            } else {
                let (lo, hi) = (1.0 / a.hi(), 1.0 / a.lo());
                radius_of(lo, hi, value) + value.abs() * ESTIMATE_REL
            }
        }
        OpKind::Rsqrt => {
            if a.lo() <= 0.0 {
                f32::INFINITY
            } else {
                let (lo, hi) = (1.0 / libm::sqrtf(a.hi()), 1.0 / libm::sqrtf(a.lo()));
                radius_of(lo, hi, value) + value.abs() * ESTIMATE_REL
            }
        }
        // Bit-pattern reinterpretations: an inexact input has no bit pattern.
        OpKind::TruncToInt => {
            let lo = eval_unary_or_panic(op, a.lo());
            let hi = eval_unary_or_panic(op, a.hi());
            if lo.to_bits() == hi.to_bits() {
                0.0
            } else {
                f32::INFINITY
            }
        }
        OpKind::IntToFloat => f32::INFINITY,
        other => panic!(
            "PointCheck: no error rule for unary {other:?} — \
             every op the emitter can emit needs one"
        ),
    };
    NodeBound::new(value, radius, divergent, a.indeterminate)
}

/// Evaluate `op`, panicking with the same message [`eval_scalar`] uses.
fn eval_binary_or_panic(op: OpKind, x: f32, y: f32) -> f32 {
    op.eval_binary(x, y).unwrap_or_else(|| {
        panic!(
            "PointCheck: no scalar eval for binary {op:?} — lower it first (expand_transcendentals)"
        )
    })
}

fn binary_bound(op: OpKind, a: NodeBound, b: NodeBound) -> NodeBound {
    over_choices(&[a, b], &|args| binary_bound_pinned(op, args[0], args[1]))
}

fn binary_bound_pinned(op: OpKind, a: NodeBound, b: NodeBound) -> NodeBound {
    let value = eval_binary_or_panic(op, a.value, b.value);
    let divergent = a.divergent || b.divergent || op_is_divergent_at(op, &[a.value, b.value]);
    let widest = a.radius.max(b.radius);

    // A comparison's result is a pattern, not a number: it has no radius. What
    // it has is a *verdict*, and the verdict is only owed when the operands are
    // further apart than their own accumulated error. Strict `<`, so operands
    // both known exactly (radius 0) are always determinate — which is what lets
    // a caller call an opposite-but-valid mask a miscompile rather than
    // excusing every disagreement as near-threshold.
    if matches!(
        op,
        OpKind::Lt | OpKind::Le | OpKind::Gt | OpKind::Ge | OpKind::Eq | OpKind::Ne
    ) {
        let window = a.radius + b.radius;
        let separation = (a.value - b.value).abs();
        // An unbounded window straddles by definition, whatever the separation
        // says. Reaching the comparison below first would be a hole: with
        // `Lt(Recip(0.0), 0.0)` the reciprocal's value AND radius are both
        // `inf`, so `separation` is `inf` too and `inf < inf` is *false* — the
        // mask would come back exact and determinate at a point where the
        // bound explicitly promises nothing, and the gates would then reject a
        // backend that legally chose the opposite answer.
        let straddles = !window.is_finite()
            || if separation.is_nan() {
                // A NaN operand with a finite window: `Lt`/`Le`/`Eq`/`Ne` are
                // exact on NaN on every backend and `Gt`/`Ge`'s split is
                // already flagged divergent, so the verdict is owed.
                false
            } else {
                separation < window
            };
        if straddles {
            // The verdict is a free choice, so record what the other choice
            // would produce. A pattern has no radius; `alt` is the carrier.
            return NodeBound::choice(value, opposite_mask(value), divergent);
        }
        return NodeBound::new(value, 0.0, divergent, a.indeterminate || b.indeterminate);
    }

    if matches!(op, OpKind::BitAnd | OpKind::BitOr) {
        // Blending two patterns is exact; blending two *numbers* that carry
        // error is not describable as a radius on the result's bits.
        return NodeBound::new(
            value,
            if widest == 0.0 { 0.0 } else { f32::INFINITY },
            divergent,
            a.indeterminate || b.indeterminate,
        );
    }

    // Exact in, exact out (see `unary_bound`).
    if widest == 0.0 {
        return NodeBound::new(value, 0.0, divergent, a.indeterminate || b.indeterminate);
    }

    let radius = match op {
        // Absolute errors add. `|value|` does NOT — which is the whole
        // cancellation story: subtract two close numbers and the radius stays
        // while the value collapses, so the relative bound blows up on its own.
        OpKind::Add | OpKind::Sub => a.radius + b.radius + value.abs() * ROUND_REL,
        OpKind::Mul => {
            a.value.abs() * b.radius
                + b.value.abs() * a.radius
                + a.radius * b.radius
                + value.abs() * ROUND_REL
        }
        OpKind::Div => {
            // Smallest the divisor can be; if it can reach zero the quotient is
            // unbounded, full stop.
            let floor = b.value.abs() - b.radius;
            if floor <= 0.0 {
                f32::INFINITY
            } else {
                (a.radius + value.abs() * b.radius) / floor + value.abs() * ROUND_REL
            }
        }
        // The result IS one of the operands. Disjoint intervals pin which one,
        // so the answer just inherits its radius; once they overlap either may
        // be selected and the bound has to reach across to the other.
        OpKind::Min | OpKind::Max => {
            if a.hi() < b.lo() || b.hi() < a.lo() {
                if value.to_bits() == a.value.to_bits() {
                    a.radius
                } else {
                    b.radius
                }
            } else {
                ((a.value - value).abs() + a.radius).max((b.value - value).abs() + b.radius)
            }
        }
        // Integer-domain binaries on inexact lanes: no pattern is determined.
        OpKind::IAdd | OpKind::Shl | OpKind::Shr => f32::INFINITY,
        other => panic!(
            "PointCheck: no error rule for binary {other:?} — \
             every op the emitter can emit needs one"
        ),
    };
    NodeBound::new(value, radius, divergent, a.indeterminate || b.indeterminate)
}

/// `Select` is the one node whose bound is **path-dependent**, and getting that
/// wrong is how a portable expression gets thrown away: in
/// `Select(Lt(x, x), Round(-0.25), x)` the `Round` never executes — the
/// condition is false at every point — yet a walker that flags divergence
/// "anywhere in the graph" marks every grid point divergent and reports the
/// expression as having nothing checkable. So divergence, radius and
/// indeterminacy all propagate from the condition plus the branch this point
/// actually takes.
///
/// The exception is a condition that is itself indeterminate: then either
/// branch could be blended, and the bound has to cover both. That comes from
/// two places, and both are live: [`over_choices`] re-runs this rule once per
/// legal resolution when the condition carries an `alt` (which also keeps a
/// mask-valued `Select` a *pattern* rather than collapsing it to an infinite
/// radius), and the `cond.indeterminate` widening below covers a condition that
/// inherited its freedom from further up.
fn select_bound(cond: NodeBound, t: NodeBound, f: NodeBound) -> NodeBound {
    over_choices(&[cond, t, f], &|args| {
        select_bound_pinned(args[0], args[1], args[2])
    })
}

fn select_bound_pinned(cond: NodeBound, t: NodeBound, f: NodeBound) -> NodeBound {
    let value = OpKind::Select
        .eval_ternary(cond.value, t.value, f.value)
        .expect("Select evaluates on three scalars");
    let bits = cond.value.to_bits();
    // A non-canonical condition blends bits from both branches — the result is
    // not either branch's value, so nothing about either branch bounds it.
    let taken = match bits {
        0xFFFF_FFFF => t,
        0x0000_0000 => f,
        _ => {
            return NodeBound::new(
                value,
                if t.radius == 0.0 && f.radius == 0.0 && !cond.indeterminate {
                    0.0
                } else {
                    f32::INFINITY
                },
                cond.divergent,
                cond.indeterminate || t.indeterminate || f.indeterminate,
            );
        }
    };
    let radius = if cond.indeterminate {
        ((t.value - value).abs() + t.radius).max((f.value - value).abs() + f.radius)
    } else {
        taken.radius
    };
    NodeBound::new(
        value,
        radius,
        cond.divergent || taken.divergent,
        cond.indeterminate || taken.indeterminate,
    )
}

fn ternary_bound(op: OpKind, a: NodeBound, b: NodeBound, c: NodeBound) -> NodeBound {
    over_choices(&[a, b, c], &|args| {
        ternary_bound_pinned(op, args[0], args[1], args[2])
    })
}

fn ternary_bound_pinned(op: OpKind, a: NodeBound, b: NodeBound, c: NodeBound) -> NodeBound {
    let value = op
        .eval_ternary(a.value, b.value, c.value)
        .unwrap_or_else(|| panic!("PointCheck: no scalar eval for ternary {op:?}"));
    let divergent = a.divergent
        || b.divergent
        || c.divergent
        || op_is_divergent_at(op, &[a.value, b.value, c.value]);
    let radius = match op {
        // One rounding fused, two split (CLAUDE.md's MulAdd row): the two
        // answers differ by at most the rounding of the product, so that gap is
        // part of the bound even when every input is exact. This is the whole
        // policy for MulAdd — it is a tolerance, never a divergence.
        OpKind::MulAdd => {
            a.value.abs() * b.radius
                + b.value.abs() * a.radius
                + a.radius * b.radius
                + c.radius
                + (a.value * b.value).abs() * ROUND_REL
                + value.abs() * ROUND_REL
        }
        other => panic!(
            "PointCheck: no error rule for ternary {other:?} — \
             every op the emitter can emit needs one"
        ),
    };
    NodeBound::new(
        value,
        radius,
        divergent,
        a.indeterminate || b.indeterminate || c.indeterminate,
    )
}

/// The oracle's answer at one point, together with the error a conforming
/// backend is allowed to have accumulated getting there.
///
/// Produced by [`DifferentialCheck::at`]. A gate reads it in this order:
///
/// 1. [`is_platform_divergent`](Self::is_platform_divergent) — skip the point;
///    the language promises no value here (CLAUDE.md's contract tables).
/// 2. [`root_is_mask_valued`](Self::root_is_mask_valued) — pick the acceptance
///    method: [`classify_mask_root`](Self::classify_mask_root) for a mask,
///    [`verdict`](Self::verdict) for a number.
/// 3. [`is_well_conditioned`](Self::is_well_conditioned) — only for
///    *cross-form* comparison, where two algebraically-equal graphs are
///    compared to each other rather than each to its own oracle.
#[derive(Clone, Copy, Debug)]
pub struct PointCheck {
    value: f32,
    radius: f32,
    divergent: bool,
    /// The other answer this root may legally produce, when it has exactly two
    /// — the two lane patterns for a mask root, or their exact conversions for
    /// a numeric one (`IntToFloat(Lt(..))` is `-1.0` or `0.0`). See
    /// [`NodeBound`]'s indeterminacy invariant; [`PointCheck::verdict`] and
    /// [`PointCheck::is_well_conditioned`] are the numeric readers.
    alt: Option<f32>,
    indeterminate: bool,
    mask_root: bool,
}

/// Outcome of checking a numeric root's JIT lane against its composed bound.
///
/// Three-way rather than a `bool` on purpose: "the bound is not finite" is not
/// a pass and not a failure, and collapsing it into either is how a real
/// miscompile hides (as a pass) or a valid kernel gets thrown away (as a
/// failure). It is metadata — record it, do not score it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointVerdict {
    /// The JIT lane lies inside `value ± bound`.
    Accept,
    /// It does not. Within a single form this is a miscompile.
    Reject,
    /// The composed bound is not finite: the interval crossed a pole, a domain
    /// edge, or a bit-pattern op with inexact input. Nothing is promised here.
    Unbounded,
}

impl PointCheck {
    /// The oracle's value — the same number [`eval_scalar`] returns for this
    /// `(term, point)`. Callers do not need to evaluate twice.
    #[must_use]
    pub fn value(self) -> f32 {
        self.value
    }

    /// The absolute error bound composed along this point's evaluation path.
    /// `INFINITY` means unbounded.
    ///
    /// When the point has two legal answers (an indeterminate comparison
    /// consumed numerically — see [`NodeBound`]'s indeterminacy invariant) this
    /// is the distance to the farther one: the **outer** bound, so that a
    /// caller reading it as `value ± bound` cannot conclude that a legal answer
    /// is a miscompile. It is deliberately looser than
    /// [`verdict`](Self::verdict), which knows the two answers exactly and
    /// rejects the gap between them — a gate must use `verdict`.
    #[must_use]
    pub fn error_bound(self) -> f32 {
        // A mask root is judged by pattern, never by distance — the gap between
        // the two lanes is `NaN`, and reporting that as this point's
        // conditioning would say "unbounded" about a root whose acceptance test
        // never consults a bound. `mask_is_indeterminate` is the reading there.
        let Some(alt) = self.alt.filter(|_| !self.mask_root) else {
            return self.radius;
        };
        let spread = (alt - self.value).abs();
        if spread.is_nan() {
            return f32::INFINITY;
        }
        spread.max(self.radius)
    }

    /// [`error_bound`](Self::error_bound) relative to the value: the number to
    /// record as this point's conditioning. `INFINITY` when the value is zero
    /// (or non-finite) and the bound is not.
    #[must_use]
    pub fn relative_error_bound(self) -> f32 {
        let bound = self.error_bound();
        if bound == 0.0 {
            return 0.0;
        }
        let scale = self.value.abs();
        if scale > 0.0 && scale.is_finite() {
            bound / scale
        } else {
            f32::INFINITY
        }
    }

    /// Whether a platform-divergent op was reached along the path this point
    /// takes. `Select`'s unchosen branch is not on that path — see
    /// [`select_bound`]'s reasoning — so a portable expression with a
    /// divergent op in dead code stays checkable.
    #[must_use]
    pub fn is_platform_divergent(self) -> bool {
        self.divergent
    }

    /// Whether the root produces a mask lane (mirrors
    /// [`DifferentialCheck::root_is_mask_valued`]).
    #[must_use]
    pub fn root_is_mask_valued(self) -> bool {
        self.mask_root
    }

    /// Whether this point determines its answer well enough to compare two
    /// algebraically-equal **forms** against each other.
    ///
    /// This is the principled replacement for magnitude and cancellation
    /// heuristics alike. A ratio rule like "output smaller than 1e-2 × the
    /// largest intermediate ⇒ ill-conditioned" has no error model behind it: it
    /// condemns `x · 1e-4`, which is perfectly determined, and thereby hands a
    /// genuinely broken extraction (`x · 2e-4`) a place to hide. The composed
    /// bound answers the actual question — `x · 1e-4` carries a relative bound
    /// near `2⁻²⁴` and stays eligible, while `sin(K·Y²)` at a large argument
    /// carries a relative bound of order 1 and drops out.
    ///
    /// For a mask root the question is not magnitude but whether the deciding
    /// comparison is pinned, so this reports the negation of
    /// [`mask_is_indeterminate`](Self::mask_is_indeterminate).
    #[must_use]
    pub fn is_well_conditioned(self) -> bool {
        if self.mask_root {
            return !self.indeterminate;
        }
        // Two legal answers is not a determined answer, whatever the radius
        // says: `alt` carries that freedom instead of `radius` (see
        // [`NodeBound`]'s indeterminacy invariant), so the relative test below
        // would read `0` and call the point perfectly conditioned.
        if self.alt.is_some() {
            return false;
        }
        if !self.radius.is_finite() || !self.value.is_finite() {
            return false;
        }
        self.radius <= WELL_CONDITIONED_REL * self.value.abs()
            || self.radius <= WELL_CONDITIONED_ABS
    }

    /// Whether the mask this root produces is genuinely near its threshold:
    /// some comparison on the taken path has operands closer together than
    /// their own accumulated error, so either verdict is a legal answer.
    ///
    /// **This is what makes an opposite-mask disagreement classifiable.**
    /// Without it, a JIT bug that inverts `Lt(0.5, 0.7)` — operands 0.2 apart,
    /// bounds around 1e-8 — is indistinguishable from a coin flip at the
    /// boundary, and gates excuse it as contract behavior.
    ///
    /// A **numeric** root can carry the same freedom, and this is not where it
    /// is read: `IntToFloat(Lt(Recip(x), k))` may legally be `-1.0` or `0.0`,
    /// and [`verdict`](Self::verdict) accepts either (see [`NodeBound`]'s
    /// indeterminacy invariant). Do not gate a numeric root on this flag — it
    /// says only that *something* upstream was unpinned, which is not the same
    /// as this root's answer being unpinned.
    #[must_use]
    pub fn mask_is_indeterminate(self) -> bool {
        self.indeterminate
    }

    /// Accept or reject a numeric root's JIT lane.
    ///
    /// # Panics
    ///
    /// Panics on a mask-valued root. The numeric path treats NaN-vs-NaN as
    /// agreement, and an all-ones mask *reads* as NaN, so routing a mask here
    /// would accept a `0x7fc00000` lane where the hardware owed `0xFFFFFFFF` —
    /// the exact hole [`classify_mask_root`](Self::classify_mask_root) exists
    /// to close. Dispatch on
    /// [`root_is_mask_valued`](Self::root_is_mask_valued).
    #[must_use]
    pub fn verdict(self, got: f32) -> PointVerdict {
        assert!(
            !self.mask_root,
            "PointCheck::verdict on a mask-valued root — use classify_mask_root; \
             the numeric path's NaN-vs-NaN rule would accept a broken mask pattern"
        );
        if got.to_bits() == self.value.to_bits() {
            return PointVerdict::Accept;
        }
        // Two legal answers and nothing in between — `IntToFloat(Lt(...))` at a
        // straddling threshold is `-1.0` or `0.0`, never `-0.5`. Accepting the
        // other answer is what stops a conforming backend from being called a
        // miscompile; refusing the gap is what keeps the point checkable.
        if let Some(alt) = self.alt {
            let agrees = |want: f32| {
                want.to_bits() == got.to_bits()
                    || (want.is_nan() && got.is_nan())
                    || (got - want).abs() <= DENORMAL_SLACK
            };
            return if agrees(self.value) || agrees(alt) {
                PointVerdict::Accept
            } else {
                PointVerdict::Reject
            };
        }
        if !self.radius.is_finite() {
            return PointVerdict::Unbounded;
        }
        // Both tiers landing on NaN is agreement about the domain. One of them
        // doing so alone is not, and no radius bridges it.
        if got.is_nan() || self.value.is_nan() {
            return if got.is_nan() && self.value.is_nan() {
                PointVerdict::Accept
            } else {
                PointVerdict::Reject
            };
        }
        if !got.is_finite() || !self.value.is_finite() {
            // Not bit-identical and one side is infinite: `radius` cannot
            // describe the gap, so this is disagreement, not slack.
            return PointVerdict::Reject;
        }
        if (got - self.value).abs() <= self.radius + DENORMAL_SLACK {
            PointVerdict::Accept
        } else {
            PointVerdict::Reject
        }
    }

    /// Classify a mask-valued root's JIT lane, using this point's threshold
    /// proximity to tell contract behavior from a bug.
    ///
    /// # Panics
    ///
    /// Panics on a numeric root — its lane is a number, and demanding
    /// all-ones/all-zeros of `trunc_to_int(5.0)` would be wrong.
    #[must_use]
    pub fn classify_mask_root(self, got: f32) -> MaskVerdict {
        assert!(
            self.mask_root,
            "PointCheck::classify_mask_root on a numeric root — use verdict"
        );
        match compare_mask_root(got, self.value) {
            MaskComparison::Agree => MaskVerdict::Agree,
            MaskComparison::InvalidPattern {
                got_is_mask,
                want_is_mask,
            } => MaskVerdict::InvalidPattern {
                got_is_mask,
                want_is_mask,
            },
            MaskComparison::Disagree => {
                if self.indeterminate {
                    MaskVerdict::NearThreshold
                } else {
                    MaskVerdict::Miscompile
                }
            }
        }
    }
}

/// Outcome of checking a mask-valued root, with proximity taken into account.
///
/// [`MaskComparison`] answers the question the *bits* can answer; this answers
/// the one a gate actually asks. The split that matters is the last two: two
/// valid masks with opposite verdicts are contract behavior only when the
/// deciding comparison's operands sit within their accumulated error of each
/// other. Away from the threshold, opposite masks are a wrong answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskVerdict {
    /// Both lanes are valid masks and bit-identical.
    Agree,
    /// Valid masks, opposite verdicts, operands within their composed error of
    /// each other: the language promises no particular verdict. Metadata.
    NearThreshold,
    /// Valid masks, opposite verdicts, operands well clear of the threshold:
    /// one tier computed the wrong answer.
    Miscompile,
    /// At least one lane is not `0x00000000`/`0xFFFFFFFF` — a broken pattern
    /// corrupts every downstream bitwise consumer. No proximity excuses it.
    InvalidPattern {
        /// Whether the JIT lane was a valid mask pattern.
        got_is_mask: bool,
        /// Whether the oracle lane was a valid mask pattern.
        want_is_mask: bool,
    },
}

/// The immutable evaluation environment threaded through the recursion: the
/// term, the coordinate values, the buffer bindings, and the current binding
/// of each reduction index (`Var(REDUCE_BINDER_BASE..)`). Grouping them keeps
/// the recursive helpers to a single node argument.
#[derive(Clone, Copy)]
struct Env<'a> {
    term: Term<'a>,
    vars: &'a [f32; COORD_AXES],
    bindings: &'a BindingTable<'a>,
    /// Values bound to the reduction indices from
    /// [`REDUCE_BINDER_BASE`](crate::decl::REDUCE_BINDER_BASE) by enclosing
    /// folds.
    reduce_vars: [f32; 4],
    variance: &'a SideTable<crate::variance::Variance>,
    memo: &'a core::cell::RefCell<SideTable<Option<f32>>>,
}

impl<'a> Env<'a> {
    fn eval(&self, node: Node<'a, ExprData>) -> f32 {
        let is_memoizable = !self.variance[node].depends_on_binder();
        if is_memoizable && let Some(val) = self.memo.borrow()[node] {
            return val;
        }
        let kids: Vec<Node<'a, ExprData>> = node.children().collect();
        let val = match (*node, kids.len()) {
            (ExprData::Var(i), _) => {
                let i = i as usize;
                // Three ranges, and the middle one holds nothing: coordinates
                // below COORD_AXES, then the reserved retired axes, then the
                // binders from REDUCE_BINDER_BASE. Reading a retired index as
                // a binder is off the end of the coordinates and short of the
                // slots, so say so rather than subtract past zero.
                assert!(
                    !RETIRED_COORD_AXES.contains(&(i as u8)),
                    "eval_scalar: Var({i}) was a retired coordinate axis; a \
                     lattice has {COORD_AXES} axes and a per-call scalar is a Uniform",
                );
                if i < COORD_AXES {
                    self.vars[i]
                } else {
                    self.reduce_vars[i - REDUCE_BINDER_BASE as usize]
                }
            }
            (ExprData::Const(bits), _) => f32::from_bits(bits),
            (ExprData::Uniform(u), _) => uniform_value(self.term.env(), self.bindings, u),
            (ExprData::Param(p), _) => {
                panic!("eval_scalar: Param({p}) — substitute params first")
            }
            (ExprData::Buffer(b), _) => panic!(
                "eval_scalar: bare Buffer({}) is not a value; read it through Gather",
                b.0
            ),
            (ExprData::Op(OpKind::RawGather), 2) => self.raw_gather(kids[0], kids[1]),
            (ExprData::Op(OpKind::Gather), 3) => self.gather(kids[0], kids[1], kids[2]),
            (ExprData::Op(OpKind::Reduce), 4) => self.reduce(kids[0], kids[1], kids[2], kids[3]),
            (ExprData::Op(op), 1) => {
                let x = self.eval(kids[0]);
                op.eval_unary(x).unwrap_or_else(|| {
                    panic!(
                        "eval_scalar: no scalar eval for unary {op:?} — \
                         lower it first (expand_transcendentals)"
                    )
                })
            }
            (ExprData::Op(op), 2) => {
                let x = self.eval(kids[0]);
                let y = self.eval(kids[1]);
                op.eval_binary(x, y).unwrap_or_else(|| {
                    panic!(
                        "eval_scalar: no scalar eval for binary {op:?} — \
                         lower it first (expand_transcendentals)"
                    )
                })
            }
            (ExprData::Op(op), 3) => {
                let x = self.eval(kids[0]);
                let y = self.eval(kids[1]);
                let z = self.eval(kids[2]);
                op.eval_ternary(x, y, z)
                    .unwrap_or_else(|| panic!("eval_scalar: no scalar eval for ternary {op:?}"))
            }
            (ExprData::Op(op), n) => {
                panic!("eval_scalar: {op:?} with {n} children is unsupported")
            }
        };
        if is_memoizable {
            self.memo.borrow_mut()[node] = Some(val);
        }
        val
    }

    /// Read one bound buffer at floored, clamped, row-major indices. This IS the
    /// reference definition of `Gather`.
    fn gather(&self, buf: Node<'a, ExprData>, x: Node<'a, ExprData>, y: Node<'a, ExprData>) -> f32 {
        let id = match *buf {
            ExprData::Buffer(id) => id,
            other => panic!("Gather's first child must be a Buffer leaf, got {other:?}"),
        };
        let decl = self.term.env().buffer(id);
        let data = self.bindings.slot(id);

        let xf = self.eval(x);
        let yf = self.eval(y);

        // Nearest-neighbor: floor then clamp to the declared extents.
        let max_x = decl.width.saturating_sub(1) as i64;
        let max_y = decl.height.saturating_sub(1) as i64;
        let xi = (libm::floorf(xf) as i64).clamp(0, max_x);
        let yi = (libm::floorf(yf) as i64).clamp(0, max_y);

        let idx = yi as usize * decl.width as usize + xi as usize;
        data[idx]
    }

    /// Read a bound buffer at an already-computed linear index (the lowered form
    /// of `Gather`). The index is trusted to be in bounds — the lowering clamped
    /// it — so this just truncates and indexes; an out-of-bounds index is a
    /// broken lowering and panics via the slice bounds check.
    fn raw_gather(&self, buf: Node<'a, ExprData>, idx: Node<'a, ExprData>) -> f32 {
        let id = match *buf {
            ExprData::Buffer(id) => id,
            other => panic!("RawGather's first child must be a Buffer leaf, got {other:?}"),
        };
        let data = self.bindings.slot(id);
        let index = self.eval(idx);
        data[libm::floorf(index) as usize]
    }

    /// Fold `body` over the reduction index `Var(reduce_var)` = `0..extent`,
    /// combining terms with the monoid named by the `combiner` child. This is
    /// the reference definition that the unrolled `expand_reduce` form must
    /// match. `combiner`, `reduce_var`, and `extent` are `Const` children.
    fn reduce(
        &self,
        combiner: Node<'a, ExprData>,
        reduce_var: Node<'a, ExprData>,
        extent: Node<'a, ExprData>,
        body: Node<'a, ExprData>,
    ) -> f32 {
        let op = OpKind::from_index(const_of(combiner, "combiner") as usize)
            .expect("reduce combiner must be a valid OpKind index");
        let var_idx = const_of(reduce_var, "var index") as usize;
        let n = const_of(extent, "extent") as usize;
        let base = REDUCE_BINDER_BASE as usize;
        let slots = base + self.reduce_vars.len();
        assert!(
            (base..slots).contains(&var_idx),
            "reduce index Var({var_idx}) out of range (must be {base}..{slots})"
        );
        let slot = var_idx - base;

        let mut acc = op
            .monoid_identity()
            .unwrap_or_else(|| panic!("reduce combiner {op:?} is not a monoid"));
        for k in 0..n {
            let mut child = *self;
            child.reduce_vars[slot] = k as f32;
            let term = child.eval(body);
            // Combine under the monoid op (Add/Mul/Min/Max).
            acc = op
                .eval_binary(acc, term)
                .expect("monoid combiner evaluates on two scalars");
        }
        acc
    }
}

/// Read a `Const` child that encodes an integer parameter.
fn const_of(node: Node<'_, ExprData>, what: &str) -> f32 {
    node.as_f32()
        .unwrap_or_else(|| panic!("reduce {what} must be a Const, got {:?}", *node))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decl::{BufferDecl, BufferIdentity};
    use crate::expr::{ExprBuilder, ExprRef};
    use alloc::vec;

    /// A frozen graph plus its environment, with every node a test wants to
    /// name kept as an entry point.
    ///
    /// A `Term` borrows both halves, so they have to outlive it — which is
    /// what this owns. `entry(i)` is the i-th node passed to [`freeze`].
    struct Graph {
        rooted: Rooted<ExprData>,
        env: Environment,
    }

    impl Graph {
        fn term(&self, i: usize) -> Term<'_> {
            Term::new(self.rooted.entry_at(i), &self.env)
        }

        fn node(&self, i: usize) -> Node<'_, ExprData> {
            self.rooted.entry_at(i)
        }

        fn eval(&self, i: usize, vars: [f32; 2], bindings: &BindingTable<'_>) -> f32 {
            eval_scalar(self.term(i), &vars, bindings)
        }

        fn check(&self, i: usize) -> DifferentialCheck {
            DifferentialCheck::new(self.term(i))
        }
    }

    fn freeze(b: ExprBuilder, roots: &[ExprRef]) -> Graph {
        let (rooted, env) = b.finish(roots);
        Graph { rooted, env }
    }

    /// A lattice-invariant leaf holding `default` — what a scalar that used
    /// to ride a retired axis is now.
    fn uniform_leaf(b: &mut ExprBuilder, default: f32) -> ExprRef {
        let slot = b.declare_uniform(crate::Uniform::new(default).decl());
        b.push_uniform(slot)
    }

    fn buffer(b: &mut ExprBuilder, width: u32, height: u32) -> crate::decl::BufferId {
        b.declare_buffer(BufferDecl {
            id: BufferIdentity::mint(),
            width,
            height,
        })
    }

    /// Reference re-implementation of `DiscreteManifold::eval`'s index math, so
    /// the test asserts equivalence against an independent expression of the
    /// same rule rather than against the interpreter itself.
    fn discrete_eval(buf: &[f32], width: usize, height: usize, x: f32, y: f32) -> f32 {
        let xi = (libm::floorf(x) as i64).clamp(0, width as i64 - 1) as usize;
        let yi = (libm::floorf(y) as i64).clamp(0, height as i64 - 1) as usize;
        buf[yi * width + xi]
    }

    #[test]
    fn gather_round_trips_discrete_manifold() {
        // 4x3 buffer of distinct values.
        let width = 4usize;
        let height = 3usize;
        let buf: vec::Vec<f32> = (0..(width * height)).map(|i| i as f32 * 10.0).collect();

        let mut b = ExprBuilder::new();
        let slot = buffer(&mut b, width as u32, height as u32);
        let x = b.push_var(0);
        let y = b.push_var(1);
        let gather = b.push_gather(slot, x, y);
        let g = freeze(b, &[gather]);

        let bindings = BindingTable::bind(&g.env, &[buf.as_slice()]).unwrap();

        // Every in-range cell, plus out-of-range coords that must clamp.
        let coords = [
            (0.0, 0.0),
            (1.0, 0.0),
            (3.0, 2.0),
            (2.0, 1.0),
            (-5.0, -5.0),   // clamp to (0,0)
            (100.0, 100.0), // clamp to (3,2)
            (1.9, 0.9),     // floor to (1,0)
        ];
        for (cx, cy) in coords {
            let got = g.eval(0, [cx, cy], &bindings);
            let want = discrete_eval(&buf, width, height, cx, cy);
            assert_eq!(got, want, "gather at ({cx}, {cy})");
        }
    }

    #[test]
    fn lowering_preserves_gather_semantics() {
        // expand_gather must produce an index expression that evaluates
        // identically to the high-level Gather.
        use crate::passes::expand_gather;

        let width = 5usize;
        let height = 4usize;
        let buf: vec::Vec<f32> = (0..(width * height)).map(|i| i as f32 + 0.5).collect();

        let mut b = ExprBuilder::new();
        let slot = buffer(&mut b, width as u32, height as u32);
        // Gather with non-trivial index expressions: (X*2, Y+1).
        let x = b.push_var(0);
        let y = b.push_var(1);
        let two = b.push_const(2.0);
        let one = b.push_const(1.0);
        let xx = b.push_binary(OpKind::Mul, x, two);
        let yy = b.push_binary(OpKind::Add, y, one);
        let gather = b.push_gather(slot, xx, yy);
        let g = freeze(b, &[gather]);

        // The environment is preserved by lowering, so the same binding works.
        let lowered = expand_gather(g.term(0));
        let lowered_term = Term::new(lowered.entry(), &g.env);

        // The lowered form must contain a RawGather and no high-level Gather.
        let reachable: vec::Vec<(ExprData, usize)> = lowered_term
            .root()
            .descendants()
            .map(|n| (*n, n.child_count()))
            .collect();
        assert!(
            reachable
                .iter()
                .any(|(d, n)| *d == ExprData::Op(OpKind::RawGather) && *n == 2)
        );
        assert!(
            !reachable
                .iter()
                .any(|(d, n)| *d == ExprData::Op(OpKind::Gather) && *n == 3)
        );

        let bindings = BindingTable::bind(&g.env, &[buf.as_slice()]).unwrap();

        // Sweep coords including fractional and out-of-range values.
        for xi in [-2.0f32, 0.0, 0.7, 1.0, 2.0, 3.0, 10.0] {
            for yi in [-1.0f32, 0.0, 0.4, 1.0, 2.0, 3.0, 8.0] {
                let vars = [xi, yi];
                let hi = g.eval(0, vars, &bindings);
                let lo = eval_scalar(lowered_term, &vars, &bindings);
                assert_eq!(hi, lo, "gather vs lowered at ({xi}, {yi})");
            }
        }
    }

    #[test]
    fn gather_composes_with_arithmetic() {
        // out = buffer[X, 0] * 2 + 1, indices computed by an expression.
        let buf = vec![5.0f32, 6.0, 7.0, 8.0];
        let mut b = ExprBuilder::new();
        let slot = buffer(&mut b, 4, 1);
        let x = b.push_var(0);
        let zero = b.push_const(0.0);
        let gather = b.push_gather(slot, x, zero);
        let two = b.push_const(2.0);
        let one = b.push_const(1.0);
        let scaled = b.push_binary(OpKind::Mul, gather, two);
        let root = b.push_binary(OpKind::Add, scaled, one);
        let g = freeze(b, &[root]);

        let bindings = BindingTable::bind(&g.env, &[buf.as_slice()]).unwrap();
        // X = 2 -> buffer[2] = 7 -> 7*2 + 1 = 15
        assert_eq!(g.eval(0, [2.0, 0.0], &bindings), 15.0);
    }

    #[test]
    fn reduce_sum_of_squares() {
        // sum_{i=0}^{3} (i+1)^2 = 1 + 4 + 9 + 16 = 30, folded over Var(4).
        let mut b = ExprBuilder::new();
        let i = b.push_var(4);
        let one = b.push_const(1.0);
        let ip1 = b.push_binary(OpKind::Add, i, one);
        let sq = b.push_binary(OpKind::Mul, ip1, ip1);
        let root = b.push_reduce(OpKind::Add, 4, 4, sq);
        let g = freeze(b, &[root]);
        assert_eq!(g.eval(0, [0.0; 2], &BindingTable::empty()), 30.0);
    }

    #[test]
    fn reduce_max_and_mul() {
        // max_{i=0..4} i = 3 ; Reduce over 0..4 of (i+1) = 1*2*3*4 = 24.
        let mut b = ExprBuilder::new();
        let i = b.push_var(4);
        let max_root = b.push_reduce(OpKind::Max, 4, 4, i);
        let one = b.push_const(1.0);
        let ip1 = b.push_binary(OpKind::Add, i, one);
        let mul_root = b.push_reduce(OpKind::Mul, 4, 4, ip1);
        let g = freeze(b, &[max_root, mul_root]);

        let bindings = BindingTable::empty();
        assert_eq!(g.eval(0, [0.0; 2], &bindings), 3.0);
        assert_eq!(g.eval(1, [0.0; 2], &bindings), 24.0);
    }

    #[test]
    fn fold_an_empty_reduce_domain_to_the_combiners_monoid_identity() {
        // extent=0 skips `reduce`'s loop entirely, so the result IS
        // `OpKind::monoid_identity()` — the only place that private value is
        // observable through the public eval_scalar/push_reduce surface.
        fn empty_reduce(combiner: OpKind) -> f32 {
            let mut b = ExprBuilder::new();
            let body = b.push_var(4);
            let root = b.push_reduce(combiner, 4, 0, body);
            let g = freeze(b, &[root]);
            g.eval(0, [0.0; 2], &BindingTable::empty())
        }

        assert_eq!(empty_reduce(OpKind::Add), 0.0);
        assert_eq!(empty_reduce(OpKind::Mul), 1.0);
        assert_eq!(empty_reduce(OpKind::Min), f32::INFINITY);
        assert_eq!(empty_reduce(OpKind::Max), f32::NEG_INFINITY);
        assert_eq!(empty_reduce(OpKind::BitOr).to_bits(), 0u32);
        assert_eq!(
            empty_reduce(OpKind::BitAnd).to_bits(),
            u32::MAX,
            "BitAnd's monoid identity is the all-ones bit pattern (never read as a number)"
        );
    }

    #[test]
    fn reduce_lowering_preserves_semantics() {
        // Σ over i of (X + i), lowered by expand_reduce, must equal the fold.
        use crate::passes::expand_reduce;
        let mut b = ExprBuilder::new();
        let x = b.push_var(0);
        let i = b.push_var(4);
        let body = b.push_binary(OpKind::Add, x, i);
        let root = b.push_reduce(OpKind::Add, 4, 5, body); // Σ_{i=0}^{4}(X+i)
        let g = freeze(b, &[root]);

        let lowered = expand_reduce(g.term(0));
        let lowered_term = Term::new(lowered.entry(), &g.env);
        assert!(
            !lowered_term
                .root()
                .descendants()
                .any(|n| *n == ExprData::Op(OpKind::Reduce)),
            "no Reduce node remains reachable from the new root"
        );

        let bindings = BindingTable::empty();
        for xv in [-2.0f32, 0.0, 3.5, 10.0] {
            let want = g.eval(0, [xv, 0.0], &bindings);
            let got = eval_scalar(lowered_term, &[xv, 0.0], &bindings);
            assert_eq!(want, got, "reduce lowering at X={xv}");
            // Σ_{i=0}^{4}(X+i) = 5X + 10.
            assert_eq!(want, 5.0 * xv + 10.0);
        }
    }

    #[test]
    fn reduce_matmul_dot_over_gather() {
        // out = Σ_i W(i,0) * input(i,0), the matmul kernel body for one output.
        let w = alloc::vec![2.0f32, 3.0, 4.0]; // W column
        let inp = alloc::vec![10.0f32, 20.0, 30.0];
        let mut b = ExprBuilder::new();
        let wb = buffer(&mut b, 3, 1);
        let ib = buffer(&mut b, 3, 1);
        let i = b.push_var(4);
        let zero = b.push_const(0.0);
        let wg = b.push_gather(wb, i, zero);
        let ig = b.push_gather(ib, i, zero);
        let prod = b.push_binary(OpKind::Mul, wg, ig);
        let root = b.push_reduce(OpKind::Add, 4, 3, prod);
        let g = freeze(b, &[root]);

        let bindings = BindingTable::bind(&g.env, &[w.as_slice(), inp.as_slice()]).unwrap();
        // 2*10 + 3*20 + 4*30 = 20 + 60 + 120 = 200.
        assert_eq!(g.eval(0, [0.0; 2], &bindings), 200.0);
    }

    const ALL_ONES: f32 = f32::from_bits(u32::MAX);
    const QNAN: f32 = f32::from_bits(0x7fc0_0000);

    #[test]
    fn valid_mask_is_exactly_the_two_patterns() {
        assert!(is_valid_mask(0x0000_0000));
        assert!(is_valid_mask(0xFFFF_FFFF));
        assert!(!is_valid_mask(0x7fc0_0000)); // quiet-NaN payload is not a mask
        assert!(!is_valid_mask(0x3f80_0000)); // 1.0 is not a mask
        assert!(!is_valid_mask(0xFFFF_FFFE)); // one bit shy of all-ones
        assert!(!is_valid_mask(0x0000_0001));
    }

    #[test]
    fn mask_root_rejects_nan_payload_that_numeric_tolerance_accepts() {
        // The hole this API closes: a numeric child tolerance folded over a
        // comparison root accepts ANY NaN payload against the required
        // all-ones mask...
        let numeric = Tolerance::Numeric {
            rel: 1e-3,
            abs: 1e-4,
        };
        assert!(numeric.accepts(QNAN, ALL_ONES));
        // ...while the root-aware path calls it what it is: a broken lane.
        assert_eq!(
            compare_mask_root(QNAN, ALL_ONES),
            MaskComparison::InvalidPattern {
                got_is_mask: false,
                want_is_mask: true,
            }
        );
        // An invalid ORACLE lane is just as much a bug, and the flags say so.
        assert_eq!(
            compare_mask_root(0.0, 1.0),
            MaskComparison::InvalidPattern {
                got_is_mask: true,
                want_is_mask: false,
            }
        );
    }

    #[test]
    fn valid_masks_classify_three_ways() {
        assert_eq!(compare_mask_root(ALL_ONES, ALL_ONES), MaskComparison::Agree);
        assert_eq!(compare_mask_root(0.0, 0.0), MaskComparison::Agree);
        // Two VALID masks with opposite verdicts: near-threshold contract
        // behavior, distinguishable from InvalidPattern so callers can file it
        // as ill-conditioned metadata rather than a miscompile.
        assert_eq!(compare_mask_root(0.0, ALL_ONES), MaskComparison::Disagree);
        assert_eq!(compare_mask_root(ALL_ONES, 0.0), MaskComparison::Disagree);
    }

    #[test]
    fn mask_valued_walk_is_linear_on_a_shared_dag() {
        // The graph is a DAG. `BitAnd(n, n)` chains — the shape e-graph
        // extraction produces when it shares aggressively — cost 2^depth to a
        // naive recursion: 29 nodes measured ~1.2s unmemoized, and 60 nodes
        // would never return. Memoized, this is instant, and a regression
        // shows up as a hung test rather than a wrong answer.
        let mut b = ExprBuilder::new();
        let x = b.push_var(0);
        let y = b.push_var(1);
        let mut masked = b.push_binary(OpKind::Lt, x, y);
        let mut numeric = b.push_binary(OpKind::Add, x, y);
        for _ in 0..60 {
            masked = b.push_binary(OpKind::BitAnd, masked, masked);
            // Same shape, but one leaf is arithmetic, so the whole chain must
            // resolve false — the memo must not short-circuit into "true".
            numeric = b.push_binary(OpKind::BitOr, numeric, numeric);
        }
        // Select shares its branches too, and its CONDITION is never visited:
        // a numeric condition under mask branches stays mask-valued.
        let mut sel = b.push_binary(OpKind::Ge, x, y);
        for _ in 0..60 {
            sel = b.push_ternary(OpKind::Select, numeric, sel, sel);
        }
        let g = freeze(b, &[masked, numeric, sel]);

        assert!(is_mask_valued(g.node(0)));
        assert!(!is_mask_valued(g.node(1)));
        assert!(is_mask_valued(g.node(2)));
    }

    #[test]
    fn mask_valued_root_detection() {
        let mut b = ExprBuilder::new();
        let x = b.push_var(0);
        let y = b.push_var(1);
        // Numeric child under a comparison root — the motivating case
        // Lt(Sin(x), y). Sin isn't scalar-evaluable pre-lowering, but
        // mask-valued-ness is structural, so any numeric child exercises it.
        let sinx = b.push_unary(OpKind::Sin, x);
        let cmp = b.push_binary(OpKind::Lt, sinx, y);
        let ge = b.push_binary(OpKind::Ge, x, y);
        let add = b.push_binary(OpKind::Add, x, y);
        let both = b.push_binary(OpKind::BitAnd, cmp, ge);
        let tainted = b.push_binary(OpKind::BitOr, cmp, add);
        let sel_masks = b.push_ternary(OpKind::Select, cmp, ge, both);
        let sel_numeric = b.push_ternary(OpKind::Select, cmp, x, y);
        let ti = b.push_unary(OpKind::TruncToInt, x);
        let g = freeze(
            b,
            &[cmp, ge, add, both, tainted, sel_masks, sel_numeric, ti],
        );

        assert!(is_mask_valued(g.node(0)));
        // Every comparison is a mask; arithmetic is not.
        assert!(is_mask_valued(g.node(1)));
        assert!(!is_mask_valued(g.node(2)));
        // BitAnd/BitOr of masks is a mask; of anything else, unknown => false.
        assert!(is_mask_valued(g.node(3)));
        assert!(!is_mask_valued(g.node(4)));
        // Select is mask-valued iff both BRANCHES are; the condition is a
        // stencil, not part of the output domain.
        assert!(is_mask_valued(g.node(5)));
        assert!(!is_mask_valued(g.node(6)));
        // TruncToInt is bitwise-DOMAIN but not mask-VALUED: its result is an
        // arbitrary integer pattern and must not be forced to all-ones/zeros.
        assert!(OpKind::TruncToInt.is_bitwise_domain());
        assert!(!is_mask_valued(g.node(7)));
    }

    #[test]
    fn oracle_comparison_root_emits_valid_masks() {
        // End-to-end: eval_scalar's lane for a comparison root passes the
        // root-aware gate against itself and IS a valid pattern.
        let mut b = ExprBuilder::new();
        let x = b.push_var(0);
        let y = b.push_var(1);
        let root = b.push_binary(OpKind::Lt, x, y);
        let g = freeze(b, &[root]);
        let bindings = BindingTable::empty();
        for (xv, yv, verdict) in [(1.0f32, 2.0f32, true), (2.0, 1.0, false)] {
            let lane = g.eval(0, [xv, yv], &bindings);
            assert!(is_valid_mask(lane.to_bits()));
            assert_eq!(lane.to_bits() == u32::MAX, verdict);
            assert_eq!(compare_mask_root(lane, lane), MaskComparison::Agree);
        }
    }

    #[test]
    fn trunc_divergence_predicate_boundaries() {
        const TWO_POW_31: f32 = 2_147_483_648.0;
        // Divergent: NaN, infinities, and everything at/above 2^31 or below
        // -2^31 — where cvttps2dq's 0x80000000 parts ways with saturation.
        assert!(trunc_input_is_divergent(f32::NAN));
        assert!(trunc_input_is_divergent(f32::INFINITY));
        assert!(trunc_input_is_divergent(f32::NEG_INFINITY));
        assert!(trunc_input_is_divergent(TWO_POW_31));
        assert!(trunc_input_is_divergent(3.0e9));
        assert!(trunc_input_is_divergent(-3.0e9));
        // Largest f32 strictly below 2^31 is in range and convergent.
        let below = f32::from_bits(TWO_POW_31.to_bits() - 1);
        assert_eq!(below, 2_147_483_520.0);
        assert!(!trunc_input_is_divergent(below));
        // -2^31 is exactly representable and exactly i32::MIN: in range.
        assert!(!trunc_input_is_divergent(-TWO_POW_31));
        // First f32 below -2^31 is out of range.
        let past = f32::from_bits((-TWO_POW_31).to_bits() + 1);
        assert!(past < -TWO_POW_31);
        assert!(trunc_input_is_divergent(past));
        // The everyday domain is untouched.
        for v in [0.0f32, -0.0, 1.5, -1.5, 65_536.0, -65_536.0] {
            assert!(!trunc_input_is_divergent(v));
        }
        // On a convergent input the oracle's TruncToInt row stays honest.
        assert_eq!(
            OpKind::TruncToInt.eval_unary(5.75).map(f32::to_bits),
            Some(5u32)
        );
    }

    // ───────────────── compositional bound ─────────────────

    /// `recip(x)` squared `2^k` times: the amplification case, and the shape a
    /// max-over-ops fold cannot describe.
    fn recip_pow2(squarings: usize) -> Graph {
        let mut b = ExprBuilder::new();
        let x = b.push_var(0);
        let mut e = b.push_unary(OpKind::Recip, x);
        for _ in 0..squarings {
            e = b.push_binary(OpKind::Mul, e, e);
        }
        freeze(b, &[e])
    }

    #[test]
    fn estimate_error_amplifies_through_repeated_squaring() {
        // `recip(x)^16` at x = -1. `rcpps` is a ~12-bit estimate, so the JIT
        // holds -1·(1 ± 3.7e-4); four squarings multiply that relative width by
        // 16. The measured JIT lane 0.9961009 is 3.9e-3 away from the exact
        // 1.0 — eight times the folded 5e-4 allowance that used to gate it, and
        // well inside the composed bound.
        let g = recip_pow2(4);
        let point = g.check(0).at(&[-1.0, 0.0], &BindingTable::empty());

        assert_eq!(point.value(), 1.0);
        assert!(!point.is_platform_divergent());
        // 16 x the estimate width, give or take the rounding terms.
        let bound = point.error_bound();
        assert!(
            (4.0e-3..1.0e-2).contains(&bound),
            "bound {bound} should be ~16x the 3.66e-4 estimate width"
        );
        assert!(bound > 8.0 * 5e-4, "must exceed the old folded allowance");

        // The valid hardware answer passes...
        assert_eq!(point.verdict(0.996_100_9), PointVerdict::Accept);
        // ...and a genuinely wrong one still does not.
        assert_eq!(point.verdict(0.9), PointVerdict::Reject);
        assert_eq!(point.verdict(1.05), PointVerdict::Reject);
        // Same-form acceptance (above) is the hard gate and it passes.
        // Cross-form ELIGIBILITY is stricter — a 0.59% relative bound no longer
        // pins three digits — so this point is metadata for a cross-form
        // comparison rather than an input to one. Two different questions, two
        // different answers, neither of them an exclusion.
        assert!(!point.is_well_conditioned());
        assert!((0.005..0.007).contains(&point.relative_error_bound()));
    }

    #[test]
    fn bound_grows_with_the_number_of_squarings() {
        // The property that distinguishes a composed bound from a folded one:
        // it depends on the expression, not just on its worst op.
        let mut prev = 0.0f32;
        for k in 0..5 {
            let g = recip_pow2(k);
            let bound = g
                .check(0)
                .at(&[-1.0, 0.0], &BindingTable::empty())
                .error_bound();
            assert!(bound > prev, "k={k}: {bound} must exceed {prev}");
            prev = bound;
        }
    }

    /// `sin(recip(x) * y * y)` — an estimate seeded at `Recip` and multiplied
    /// by the argument before a transcendental.
    fn sin_of_recip_scaled_square() -> Graph {
        let mut b = ExprBuilder::new();
        let x = b.push_var(0);
        let y = b.push_var(1);
        let k = b.push_unary(OpKind::Recip, x);
        let y2 = b.push_binary(OpKind::Mul, y, y);
        let arg = b.push_binary(OpKind::Mul, k, y2);
        let root = b.push_unary(OpKind::Sin, arg);
        freeze(b, &[root])
    }

    #[test]
    fn amplification_is_input_dependent_not_expression_dependent() {
        // One expression, two points. The smoke run measured relative error 0
        // at argument 0.83, 3.6e-3 at 8.3e3, and 1.0 at 8.3e7 — which is why a
        // single per-expression tolerance cannot be right at both ends.
        let g = sin_of_recip_scaled_square();
        let check = g.check(0);
        let b = BindingTable::empty();

        // Small argument: the estimate's width never gets multiplied up, so the
        // point stays usable and a wrong answer is still caught.
        let near = check.at(&[1.0, 0.911], &b);
        assert!(!near.is_platform_divergent());
        assert!(
            near.is_well_conditioned(),
            "rel bound {} at argument ~0.83",
            near.relative_error_bound()
        );
        assert_eq!(near.verdict(near.value() + 0.25), PointVerdict::Reject);

        // Large argument, still inside `TRIG_DOMAIN`: the same seed is now
        // multiplied by ~1e6, scrambling the phase by several radians, so no
        // backend's answer is pinned. The gate must not call this a miscompile.
        // Stay under 2^20 — past the domain the answer is NaN by construction
        // (see `sin_past_its_domain_is_exactly_nan_not_amplified`), which is a
        // different property and would mask this one.
        let far = check.at(&[1.0, 1000.0], &b);
        assert!(
            far.relative_error_bound() > 1.0,
            "rel bound {} at argument ~1e6",
            far.relative_error_bound()
        );
        assert!(!far.is_well_conditioned());
        assert_ne!(far.verdict(-0.5), PointVerdict::Reject);
        assert_ne!(far.verdict(0.5), PointVerdict::Reject);
    }

    /// Past `TRIG_DOMAIN` the lowering returns NaN rather than a reduced value,
    /// so oracle and JIT agree *exactly* and the error bound is legitimately
    /// zero. Pinned separately because a zero bound here is correct while the
    /// same reading in-domain would mean the walk had dropped incoming error —
    /// telling those two apart by eye is what made the CI failure that
    /// motivated this test look like a regression.
    #[test]
    fn sin_past_its_domain_is_exactly_nan_not_amplified() {
        let g = sin_of_recip_scaled_square();
        // y = 9110 puts the argument at ~8.3e7, far past 2^20.
        let beyond = g.check(0).at(&[1.0, 9110.0], &BindingTable::empty());
        assert!(
            beyond.value().is_nan(),
            "expected the guarded NaN, got {}",
            beyond.value()
        );
        assert_eq!(beyond.relative_error_bound(), 0.0);
    }

    #[test]
    fn indeterminacy_survives_conversion_to_a_number() {
        // `IntToFloat(Lt(Recip(x), k))` at the threshold. The comparison is a
        // pattern with radius 0 and `indeterminate == true`; `IntToFloat` used
        // to take the exact-in exact-out fast path, copying the flag onto a
        // ZERO radius that nothing numeric reads. A backend choosing the other
        // legal mask returns 0.0 where the oracle has -1.0 (or the reverse),
        // and the hard bug gate called that a same-form miscompile.
        let mut b = ExprBuilder::new();
        let x = b.push_var(0);
        let k = b.push_var(1);
        let r = b.push_unary(OpKind::Recip, x);
        let cmp = b.push_binary(OpKind::Lt, r, k);
        let root = b.push_unary(OpKind::IntToFloat, cmp);
        let g = freeze(b, &[root]);

        let check = g.check(0);
        // The root is a NUMBER, so the numeric path is what has to know.
        assert!(!check.root_is_mask_valued());

        // x = k = 1: recip(1) is 1 to within the estimate's width, so both
        // verdicts are legal and both conversions are legal answers.
        let at = check.at(&[1.0, 1.0], &BindingTable::empty());
        assert!(at.mask_is_indeterminate());
        assert_eq!(at.value(), 0.0, "lt(1,1) is false, and int_to_float(0) = 0");
        assert_eq!(at.verdict(0.0), PointVerdict::Accept);
        // THE BUG: the opposite legal mask converts to -1.0. Not a miscompile.
        assert_eq!(at.verdict(-1.0), PointVerdict::Accept);
        // A genuinely wrong value still is one — including one that sits
        // between the two legal answers, which an interval bound would have
        // waved through.
        assert_eq!(at.verdict(-0.5), PointVerdict::Reject);
        assert_eq!(at.verdict(5.0), PointVerdict::Reject);
        // Two legal answers is never a point to compare two FORMS at, and the
        // survey must not describe it as a zero-error point either.
        assert!(!at.is_well_conditioned());
        assert_eq!(at.error_bound(), 1.0, "the two answers are 0.0 and -1.0");
        assert_eq!(at.relative_error_bound(), f32::INFINITY);

        // Away from the threshold nothing is excused: recip(0.5) = 2 is a whole
        // unit clear of 1, so the mask — and its conversion — are pinned.
        let clear = check.at(&[0.5, 1.0], &BindingTable::empty());
        assert!(!clear.mask_is_indeterminate());
        assert_eq!(clear.value(), 0.0, "2 < 1 is false");
        assert_eq!(clear.error_bound(), 0.0);
        assert_eq!(clear.verdict(0.0), PointVerdict::Accept);
        assert_eq!(clear.verdict(-1.0), PointVerdict::Reject);
        assert!(clear.is_well_conditioned());
    }

    #[test]
    fn indeterminate_mask_arithmetic_covers_both_answers() {
        // The same leak one op further out: the converted mask is an ordinary
        // number, so it composes. `1 + int_to_float(lt(recip(x), k))` is 1.0 or
        // 0.0 at the threshold, and a bound of zero on either would report the
        // other as a miscompile.
        let mut b = ExprBuilder::new();
        let x = b.push_var(0);
        let k = b.push_var(1);
        let r = b.push_unary(OpKind::Recip, x);
        let cmp = b.push_binary(OpKind::Lt, r, k);
        let converted = b.push_unary(OpKind::IntToFloat, cmp);
        let one = b.push_const(1.0);
        let root = b.push_binary(OpKind::Add, one, converted);
        let g = freeze(b, &[root]);

        let point = g.check(0).at(&[1.0, 1.0], &BindingTable::empty());
        assert_eq!(point.value(), 1.0);
        assert_eq!(point.verdict(1.0), PointVerdict::Accept);
        assert_eq!(point.verdict(0.0), PointVerdict::Accept);
        assert_eq!(point.verdict(0.5), PointVerdict::Reject);
        assert_eq!(point.verdict(2.0), PointVerdict::Reject);
        assert!(!point.is_well_conditioned());
    }

    #[test]
    fn a_determinate_mask_conversion_keeps_its_zero_bound() {
        // The widening must not leak into expressions that never straddle:
        // `int_to_float(lt(x, y))` on exact operands is exactly 0 or -1, and a
        // backend returning the other one IS a miscompile.
        let mut b = ExprBuilder::new();
        let x = b.push_var(0);
        let y = b.push_var(1);
        let cmp = b.push_binary(OpKind::Lt, x, y);
        let root = b.push_unary(OpKind::IntToFloat, cmp);
        let g = freeze(b, &[root]);

        let point = g.check(0).at(&[0.5, 0.7], &BindingTable::empty());
        assert!(!point.mask_is_indeterminate());
        assert_eq!(point.value(), -1.0, "0.5 < 0.7 is all-ones, which is -1");
        assert_eq!(point.error_bound(), 0.0);
        assert_eq!(point.verdict(-1.0), PointVerdict::Accept);
        assert_eq!(point.verdict(0.0), PointVerdict::Reject);
        assert!(point.is_well_conditioned());
    }

    #[test]
    fn an_indeterminate_mask_selecting_masks_stays_a_pattern() {
        // A mask-valued root must not be turned into a number by the widening:
        // `Select(Lt(recip(x), k), Lt(a, b), 0.0)` is still a pattern, and
        // `classify_mask_root` — not a radius — is what judges it.
        let mut b = ExprBuilder::new();
        let x = b.push_var(0);
        let k = b.push_var(1);
        // Two lattice-invariant leaves — the shape Z and W used to have.
        let a = uniform_leaf(&mut b, 1.0);
        let bb = uniform_leaf(&mut b, 2.0);
        let r = b.push_unary(OpKind::Recip, x);
        let cond = b.push_binary(OpKind::Lt, r, k);
        let branch = b.push_binary(OpKind::Lt, a, bb);
        let zero = b.push_const(0.0);
        let root = b.push_ternary(OpKind::Select, cond, branch, zero);
        let g = freeze(b, &[root]);

        let check = g.check(0);
        assert!(check.root_is_mask_valued());
        // cond straddles; the true branch is all-ones, the false branch zero,
        // so the two resolutions disagree and either lane is legal.
        let point = check.at(&[1.0, 1.0], &BindingTable::empty());
        assert!(point.mask_is_indeterminate());
        assert_eq!(
            point.classify_mask_root(ALL_ONES),
            MaskVerdict::NearThreshold
        );
        assert_eq!(point.classify_mask_root(0.0), MaskVerdict::Agree);
        // Proximity never excuses a broken pattern.
        assert_eq!(
            point.classify_mask_root(QNAN),
            MaskVerdict::InvalidPattern {
                got_is_mask: false,
                want_is_mask: true,
            }
        );
    }

    #[test]
    fn low_gain_arithmetic_stays_well_conditioned() {
        // Supersedes the ratio heuristic ("output < 1e-2 x the largest
        // intermediate => ill-conditioned"), which condemned `x * 1e-4` at
        // every point and thereby gave a broken `x * 2e-4` somewhere to hide.
        // Exact inputs through a correctly-rounded op compose to a ZERO bound,
        // so the point is maximally usable and the broken form is rejected.
        let mut b = ExprBuilder::new();
        let x = b.push_var(0);
        let small = b.push_const(1e-4);
        let root = b.push_binary(OpKind::Mul, x, small);
        let g = freeze(b, &[root]);
        let check = g.check(0);
        for xv in [1.0f32, -3.5, 1e3, 1e-3] {
            let point = check.at(&[xv, 0.0], &BindingTable::empty());
            assert_eq!(point.error_bound(), 0.0);
            assert!(point.is_well_conditioned(), "x={xv}");
            assert_eq!(point.verdict(xv * 1e-4), PointVerdict::Accept);
            assert_eq!(point.verdict(xv * 2e-4), PointVerdict::Reject, "x={xv}");
        }
    }

    #[test]
    fn cancellation_shows_up_as_an_exploded_relative_bound() {
        // `recip(x) - 1` at x = 1: the absolute bound is the estimate's width
        // and survives the subtraction untouched, while the value collapses to
        // zero — so the RELATIVE bound is infinite. That is the conditioning
        // signal, derived rather than guessed.
        let mut b = ExprBuilder::new();
        let x = b.push_var(0);
        let r = b.push_unary(OpKind::Recip, x);
        let one = b.push_const(1.0);
        let root = b.push_binary(OpKind::Sub, r, one);
        let g = freeze(b, &[root]);
        let point = g.check(0).at(&[1.0, 0.0], &BindingTable::empty());
        assert_eq!(point.value(), 0.0);
        assert!(point.error_bound() > 1e-4);
        assert_eq!(point.relative_error_bound(), f32::INFINITY);
        assert!(!point.is_well_conditioned());
    }

    #[test]
    fn divergence_respects_the_chosen_select_branch() {
        // `Select(Lt(x, x), Round(-0.25), x)` is portably `x` — the condition
        // is false everywhere, so the platform-divergent `Round` never runs. A
        // walker that flags divergence "anywhere in the graph" skips every grid
        // point and reports NoCheckablePoints.
        let mut b = ExprBuilder::new();
        let x = b.push_var(0);
        let cond = b.push_binary(OpKind::Lt, x, x);
        let quarter = b.push_const(-0.25);
        let rounded = b.push_unary(OpKind::Round, quarter);
        let root = b.push_ternary(OpKind::Select, cond, rounded, x);
        // The op is still divergent when the point actually reaches it: same
        // branches, a condition that is true.
        let taken = b.push_binary(OpKind::Le, x, x);
        let reached = b.push_ternary(OpKind::Select, taken, rounded, x);
        let g = freeze(b, &[root, reached]);

        let check = g.check(0);
        let bindings = BindingTable::empty();
        for xv in [-2.0f32, -0.25, 0.0, 1.0, 7.5] {
            let point = check.at(&[xv, 0.0], &bindings);
            assert!(!point.is_platform_divergent(), "x={xv}");
            assert_eq!(point.value(), xv);
            assert_eq!(point.verdict(xv), PointVerdict::Accept);
        }

        let point = g.check(1).at(&[1.0, 0.0], &bindings);
        assert!(point.is_platform_divergent());
    }

    #[test]
    fn mask_disagreement_needs_proximity_to_be_excused() {
        // `Lt(0.5, 0.7)` has operands 0.2 apart with zero accumulated error. A
        // JIT returning the opposite canonical mask is a miscompile, not a
        // boundary coin flip, and the classification must say so.
        let mut b = ExprBuilder::new();
        let x = b.push_var(0);
        let y = b.push_var(1);
        let root = b.push_binary(OpKind::Lt, x, y);
        let g = freeze(b, &[root]);
        let check = g.check(0);
        assert!(check.root_is_mask_valued());

        let point = check.at(&[0.5, 0.7], &BindingTable::empty());
        assert!(!point.mask_is_indeterminate());
        assert_eq!(point.classify_mask_root(ALL_ONES), MaskVerdict::Agree);
        assert_eq!(point.classify_mask_root(0.0), MaskVerdict::Miscompile);
        assert_eq!(
            point.classify_mask_root(QNAN),
            MaskVerdict::InvalidPattern {
                got_is_mask: false,
                want_is_mask: true,
            }
        );
    }

    #[test]
    fn mask_disagreement_at_the_threshold_is_contract() {
        // `Lt(recip(x), y)` at x = y = 1: the operands are equal, but the left
        // one carries the estimate's width, so both verdicts are legal answers.
        let mut b = ExprBuilder::new();
        let x = b.push_var(0);
        let y = b.push_var(1);
        let r = b.push_unary(OpKind::Recip, x);
        let root = b.push_binary(OpKind::Lt, r, y);
        let g = freeze(b, &[root]);
        let point = g.check(0).at(&[1.0, 1.0], &BindingTable::empty());

        assert!(point.mask_is_indeterminate());
        assert!(!point.is_well_conditioned());
        assert_eq!(
            point.classify_mask_root(ALL_ONES),
            MaskVerdict::NearThreshold
        );
        // A broken PATTERN is never excused by proximity.
        assert_eq!(
            point.classify_mask_root(QNAN),
            MaskVerdict::InvalidPattern {
                got_is_mask: false,
                want_is_mask: true,
            }
        );
    }

    #[test]
    fn canonical_mask_constants_are_mask_valued() {
        // `Select(cond, Lt(x + y, z), 0.0)` used to degrade to the numeric path
        // because of the constant branch — and the numeric path's NaN-vs-NaN
        // rule then accepted a broken 0x7fc00000 against a required all-ones
        // mask. `0.0` IS `0x00000000`, the false mask.
        let mut b = ExprBuilder::new();
        let x = b.push_var(0);
        let y = b.push_var(1);
        let z = uniform_leaf(&mut b, 9.0);
        let w = uniform_leaf(&mut b, 0.0);
        let cond = b.push_binary(OpKind::Lt, w, x);
        let sum = b.push_binary(OpKind::Add, x, y);
        let cmp = b.push_binary(OpKind::Lt, sum, z);

        let zero = b.push_const(0.0);
        let masked = b.push_ternary(OpKind::Select, cond, cmp, zero);
        let ones = b.push_const(ALL_ONES);
        let all_ones_branch = b.push_ternary(OpKind::Select, cond, cmp, ones);
        // A non-mask constant really does make the result a number.
        let one = b.push_const(1.0);
        let numeric = b.push_ternary(OpKind::Select, cond, cmp, one);
        // -0.0 is 0x80000000, not a mask.
        let neg_zero = b.push_const(-0.0);
        let signed = b.push_ternary(OpKind::Select, cond, cmp, neg_zero);
        let g = freeze(b, &[masked, all_ones_branch, numeric, signed]);

        assert!(is_mask_valued(g.node(0)));
        assert!(is_mask_valued(g.node(1)));
        assert!(!is_mask_valued(g.node(2)));
        assert!(!is_mask_valued(g.node(3)));

        // And the end-to-end consequence: the mask path now gates the root, so
        // the broken pattern is caught instead of waved through.
        let point = g.check(0).at(
            &[1.0, 2.0], // cond true, x+y = 3 < 9 => all-ones
            &BindingTable::empty(),
        );
        assert!(point.root_is_mask_valued());
        assert_eq!(point.value().to_bits(), u32::MAX);
        assert_eq!(
            point.classify_mask_root(QNAN),
            MaskVerdict::InvalidPattern {
                got_is_mask: false,
                want_is_mask: true,
            }
        );
    }

    #[test]
    #[should_panic(expected = "would accept a broken mask pattern")]
    fn numeric_verdict_refuses_a_mask_root() {
        let mut b = ExprBuilder::new();
        let x = b.push_var(0);
        let y = b.push_var(1);
        let root = b.push_binary(OpKind::Lt, x, y);
        let g = freeze(b, &[root]);
        let _unreachable = g
            .check(0)
            .at(&[0.0; 2], &BindingTable::empty())
            .verdict(0.0);
    }

    #[test]
    #[should_panic(expected = "on a numeric root")]
    fn mask_classification_refuses_a_numeric_root() {
        let mut b = ExprBuilder::new();
        let x = b.push_var(0);
        let y = b.push_var(1);
        let root = b.push_binary(OpKind::Add, x, y);
        let g = freeze(b, &[root]);
        let _unreachable = g
            .check(0)
            .at(&[0.0; 2], &BindingTable::empty())
            .classify_mask_root(0.0);
    }

    #[test]
    fn point_value_agrees_with_eval_scalar() {
        // The bound walk must not become a second semantics: its value is the
        // oracle's value, node for node.
        let mut b = ExprBuilder::new();
        let x = b.push_var(0);
        let y = b.push_var(1);
        let r = b.push_unary(OpKind::Recip, x);
        let s = b.push_unary(OpKind::Sin, y);
        let m = b.push_binary(OpKind::Mul, r, s);
        let c = b.push_const(2.0);
        let fma = b.push_ternary(OpKind::MulAdd, m, c, y);
        let root = b.push_binary(OpKind::Max, fma, m);
        let g = freeze(b, &[root]);

        let check = g.check(0);
        let bindings = BindingTable::empty();
        for xv in [0.5f32, -2.0, 7.25] {
            for yv in [0.1f32, -1.75, 3.0] {
                let vars = [xv, yv];
                let point = check.at(&vars, &bindings);
                assert_eq!(
                    point.value().to_bits(),
                    g.eval(0, vars, &bindings).to_bits(),
                    "at ({xv}, {yv})"
                );
            }
        }
    }

    #[test]
    fn divergence_policy_checks_estimates_and_flags_trunc() {
        // The two adjustments to `fold_is_platform_specific` that make it a
        // runtime divergence policy rather than a folding policy.
        assert!(!op_is_divergent_at(OpKind::Recip, &[3.0]));
        assert!(!op_is_divergent_at(OpKind::Rsqrt, &[3.0]));
        assert!(OpKind::Recip.fold_is_platform_specific(&[3.0]));

        assert!(op_is_divergent_at(OpKind::TruncToInt, &[f32::NAN]));
        assert!(op_is_divergent_at(OpKind::TruncToInt, &[3.0e9]));
        assert!(!op_is_divergent_at(OpKind::TruncToInt, &[5.75]));

        // The rows that carry over unchanged.
        assert!(op_is_divergent_at(OpKind::Min, &[f32::NAN, 1.0]));
        assert!(op_is_divergent_at(OpKind::Round, &[2.5]));
        assert!(!op_is_divergent_at(OpKind::Add, &[1.0, 2.0]));
    }

    #[test]
    fn every_op_the_emitter_can_emit_has_an_error_rule() {
        // A missing rule is a PANIC inside a correctness gate, so this sweeps
        // the whole inventory rather than trusting the match arms to be
        // exhaustive by inspection. Transcendentals go through their expansion
        // (which is where the integer-domain primitives show up), and every
        // operand is inexact so the "exact in, exact out" shortcut cannot hide
        // a missing arm.
        let bindings = BindingTable::empty();
        for op in OpKind::all() {
            // Leaves, memory, the reduction binder, and the two ops that must
            // be rewritten away carry their own (tested) refusals. A leaf is
            // exactly an op of arity zero, so ask that rather than listing
            // them: the list was six names, and the seventh leaf added to the
            // op table did not land in it, arriving here as a nonsense
            // `push_ternary(Param, ..)` instead.
            if op.arity() == 0
                || matches!(
                    op,
                    OpKind::Gather | OpKind::RawGather | OpKind::Reduce | OpKind::Dwrt
                )
            {
                continue;
            }
            let mut b = ExprBuilder::new();
            let x = b.push_var(0);
            // An estimate seeds a nonzero radius into every operand.
            let inexact = b.push_unary(OpKind::Recip, x);
            let root = match op.arity() {
                1 => b.push_unary(op, inexact),
                2 => b.push_binary(op, inexact, inexact),
                _ => b.push_ternary(op, inexact, inexact, inexact),
            };
            let g = freeze(b, &[root]);
            let point = g.check(0).at(&[2.0, 2.0], &bindings);
            assert!(point.error_bound() >= 0.0, "{op:?}");
        }
    }

    #[test]
    fn bound_walk_is_linear_on_a_shared_dag() {
        // Same hazard as `is_mask_valued`: a shared DAG re-derived recursively
        // costs 2^depth. A regression hangs the suite rather than lying.
        let mut b = ExprBuilder::new();
        let x = b.push_var(0);
        let mut e = b.push_unary(OpKind::Recip, x);
        for _ in 0..60 {
            e = b.push_binary(OpKind::Add, e, e);
        }
        let g = freeze(b, &[e]);
        let point = g.check(0).at(&[2.0, 0.0], &BindingTable::empty());
        assert_eq!(point.value(), 0.5 * (1u64 << 60) as f32);
        assert!(point.error_bound() > 0.0);
    }

    #[test]
    fn bind_rejects_wrong_length() {
        let mut b = ExprBuilder::new();
        let _ = buffer(&mut b, 4, 2);
        let x = b.push_var(0);
        let g = freeze(b, &[x]);
        let short = vec![0.0f32; 7]; // needs 8
        let err = BindingTable::bind(&g.env, &[short.as_slice()]).unwrap_err();
        assert!(matches!(
            err,
            crate::binding::BindError::Length {
                slot: 0,
                expected: 8,
                actual: 7
            }
        ));
    }

    #[test]
    fn bind_rejects_wrong_count() {
        let mut b = ExprBuilder::new();
        let _ = buffer(&mut b, 2, 2);
        let x = b.push_var(0);
        let g = freeze(b, &[x]);
        let err = BindingTable::bind(&g.env, &[]).unwrap_err();
        assert!(matches!(
            err,
            crate::binding::BindError::Count {
                declared: 1,
                supplied: 0
            }
        ));
    }
}
