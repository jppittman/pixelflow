//! What an integral over an interval closes to: the right-hand sides the
//! e-graph's integration rules write in place of a fold they close.
//!
//! An [`IntervalFold`] keeps its ends to itself, and these are the formulas
//! that read them — so they live beside it, the way its quadrature does, and
//! `pixelflow-search`'s integration rules decide only *when* one applies
//! (docs/plans/2026-09-23-an-integral-is-a-fold.md §3). Each builder writes
//! its closed form into an [`ExprArena`] over operands its caller pushed
//! there; a rule's template is that arena, its operands the rule's
//! metavariables.
//!
//! Every formula is an identity over ℝ. What each one does in `f32` is on
//! its own doc. No formula divides by a value that can be zero, even in an
//! arm a `Select` discards — the e-graph would prove it zero, and then prove
//! false equalities of the quotient ([`mean_of_clamp`], "The divisor") —
//! and none lets a quotient decide a saturated or degenerate case: the mean
//! of a clamp decides those by comparison first, and a monotone root
//! divides only by a denominator floored at a positive literal, so its
//! degenerate cases are quotients far outside `[0, 1]` that a clamp then
//! saturates ([`IntervalFold::arc_moment`]).
//!
//! [`mean_of_clamp`] and [`monotone_root`] are written once, here. A caller
//! that needs the mean of a clamp over a span — [`IntervalFold::clamp_moment`]
//! today, a glyph's pixel/half-plane coverage tomorrow — or the parameter at
//! which a monotone arc reaches a height calls them rather than restating
//! them (CLAUDE.md, "one definition, imported, not restated").

use crate::arena::{ExprArena, ExprId, ExprNode};
use crate::fold::IntervalFold;
use crate::kind::OpKind;

/// A span whose clamp is averaged over no wider than this fraction of the
/// band is averaged by its midpoint instead, by [`mean_of_clamp`].
///
/// `2⁻²⁰`: the midpoint of a 1-Lipschitz function with one kink of slope
/// change 1 errs by at most an eighth of the span, so the degenerate arm is
/// off by at most `2⁻²³` of the band — below `f32`'s own resolution of a
/// value in it. The span's quotient form needs a nonzero span, and within
/// this width it would be dividing a difference of nearly equal values by
/// another; the arm replaces both at once.
pub const DEGENERATE_SPAN: f32 = 1.0 / 1_048_576.0;

/// `[lower, upper]`, the closed band a clamp holds its argument in: finite,
/// with `lower < upper`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Band {
    lower: f32,
    upper: f32,
}

impl Band {
    /// The band `[lower, upper]`, or `None` unless both are finite and
    /// `lower < upper` — a clamp with `lower >= upper` is a constant, not a
    /// band, and its moment is not this one.
    #[must_use]
    pub fn new(lower: f32, upper: f32) -> Option<Self> {
        (lower.is_finite() && upper.is_finite() && lower < upper).then_some(Self { lower, upper })
    }
}

/// `slope·u + offset`: an integrand's argument, affine in the integration
/// variable `u`, with `u` free in neither operand.
#[derive(Clone, Copy, Debug)]
pub struct Affine {
    /// The coefficient of `u`.
    pub slope: ExprId,
    /// The rest.
    pub offset: ExprId,
}

/// Where a product of indicators is nonzero: `lower < u < upper`, either end
/// absent when no factor bounds that side.
///
/// Strict or not makes no difference to an integral — the ends are points,
/// and a point has no length — so a cut does not say which.
#[derive(Clone, Copy, Debug)]
pub struct Cut {
    /// The largest lower bound, if any factor gives one.
    pub lower: Option<ExprId>,
    /// The smallest upper bound, if any factor gives one.
    pub upper: Option<ExprId>,
}

/// An integral narrowed to a cut, as a reparametrization of the same
/// interval: `∫_lo^hi [cut]·f(u) du = scale · ∫_lo^hi f(point) du`.
#[derive(Clone, Copy, Debug)]
pub struct Narrowing {
    /// The Jacobian `s = h/(hi − lo)`, `h` the cut's length inside the
    /// interval.
    pub scale: ExprId,
    /// `m + s·(u − c)`: the point in the cut that `u` in the interval maps
    /// to, `m` the cut's midpoint and `c` the interval's.
    pub point: ExprId,
}

/// The values an integrand's argument takes across an interval: `from` at
/// its lower end, `to` at its upper end, `centre` at its midpoint.
#[derive(Clone, Copy, Debug)]
pub struct Sweep {
    /// The argument at the interval's lower end.
    pub from: ExprId,
    /// The argument at the interval's upper end.
    pub to: ExprId,
    /// The argument at the interval's midpoint.
    pub centre: ExprId,
}

/// What a certificate floors a control polygon's step at: `max(step,
/// STEP_FLOOR)` is a step no smaller than this whatever the table holds,
/// which is what makes a [`Rise`] rise ([`IntervalFold::arc_moment`],
/// "Certificates").
pub const STEP_FLOOR: f32 = 0.0;

/// What [`monotone_root`] floors its radicand at: `√max(r, RADICAND_FLOOR)`.
///
/// Part of the root's definition, not a tunable: below zero `√` is NaN, and
/// above it the floor would move the root wherever the radicand is small
/// but real.
pub const RADICAND_FLOOR: f32 = 0.0;

/// `2⁻¹⁰⁰`: the floor an author writes under a [`monotone_root`]'s
/// denominator, and the largest one [`RootFloor`] admits.
///
/// Where the floor is active the root is not the rise's inverse, and the
/// heights that happens at are at most `floor` from the arc's start; so an
/// integral over the root is exact up to a height of `2⁻¹⁰⁰`, far below
/// anything an `f32` pixel resolves. A larger floor would make that a
/// visible band — the rule that integrates it would then be false.
pub const ROOT_FLOOR: f32 = 1.0 / 1_267_650_600_228_229_401_496_703_205_376.0;

/// A floor for a [`monotone_root`]'s denominator: a literal in
/// `[f32::MIN_POSITIVE, ROOT_FLOOR]`.
///
/// Positive, so the quotient never divides by zero — not even where the
/// e-graph could prove the rest of the denominator is (see
/// [`mean_of_clamp`], "The divisor"); at most [`ROOT_FLOOR`], so the root it
/// floors is the rise's inverse everywhere but a height of `2⁻¹⁰⁰`.
///
/// **Normal**, not merely positive. A subnormal floor is zero wherever a
/// kernel runs with denormals-are-zero — which the renderer's workers do
/// (`FastMathGuard`) — and there `δ/max(0, floor)` is `0/0`. From `2⁻¹²⁸`
/// down its reciprocal also overflows `f32`, so the exact `1/floor` the root
/// multiplies by cannot be a finite literal. Measured before the bound: a
/// vertical line of literal columns under the subnormal floor `2⁻¹³⁶` read
/// `1` for `0` along the pixel edge it starts on
/// (`pixelflow-core/tests/arc_adversarial.rs`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RootFloor(f32);

impl RootFloor {
    /// The floor `value`, or `None` outside `[f32::MIN_POSITIVE, ROOT_FLOOR]`.
    #[must_use]
    pub fn new(value: f32) -> Option<Self> {
        (f32::MIN_POSITIVE..=ROOT_FLOOR)
            .contains(&value)
            .then_some(Self(value))
    }

    /// The literal.
    #[must_use]
    pub fn get(self) -> f32 {
        self.0
    }
}

/// `q(t) = t·(step + step + bend·t)`: one coordinate of a quadratic Bézier
/// arc, measured from its start.
///
/// With control points `p₀, p₁, p₂`, `step = p₁ − p₀` and
/// `bend = (p₂ − p₁) − step`, so `q(1) = p₂ − p₀` and
/// `q′(t) = 2·((1 − t)·step + t·(step + bend))` — the two steps of the
/// control polygon, blended. **Certified** when `step ≥ 0` and
/// `step + bend ≥ 0`: then `q′ ≥ 0` on `[0, 1]`, and the coordinate rises
/// along the whole arc.
#[derive(Clone, Copy, Debug)]
pub struct Rise {
    /// `p₁ − p₀`, the first step.
    pub step: ExprId,
    /// `(p₂ − p₁) − (p₁ − p₀)`, the second step less the first.
    pub bend: ExprId,
}

/// A monotone arc crossing an integral's variable: the integrand
/// [`IntervalFold::arc_moment`] closes.
///
/// The variable `u` is a height `δ = u + height` above the arc's start; the
/// arc reaches it at the parameter `T = τ_y(δ)` ([`monotone_root`]), and the
/// integrand reads the arc's other coordinate there, `offset + x(T)`.
#[derive(Clone, Copy, Debug)]
pub struct MonotoneArc {
    /// `D₀`: the height at the variable's zero, `δ = u + D₀`.
    pub height: ExprId,
    /// The rising coordinate the variable is a height of.
    pub y: Rise,
    /// `c₀`: what the clamp adds to `x(T)`.
    pub offset: ExprId,
    /// The rising coordinate the clamp reads.
    pub x: Rise,
    /// The floor both roots are taken under.
    pub floor: RootFloor,
}

/// `(1/(z₁ − z₀)) ∫_{z₀}^{z₁} clamp(z, P, Q) dz` — the mean of a clamp over
/// the span an affine argument sweeps, `[P, Q]` the band.
///
/// **Law.** With `G(z) = ½·clamp(z, P, Q)² + Q·(max(z, Q) − Q) + P·(min(z, P) − P)`,
/// an antiderivative of `clamp(z, P, Q)`, the mean is `N/d` where
/// `d = z₁ − z₀` and
///
/// ```text
/// N = G(z₁) − G(z₀) = ½(b − a)(b + a) + Q·(max(z₁,Q) − max(z₀,Q)) + P·(min(z₁,P) − min(z₀,P)),
/// a = clamp(z₀, P, Q),  b = clamp(z₁, P, Q)
/// ```
///
/// (the `−Q` and `−P` of `G` cancel in the difference). Emitted as
///
/// ```text
/// narrow = |d| ≤ DEGENERATE_SPAN·(Q − P)
/// select(min(z₀,z₁) ≥ Q, Q,
///   select(max(z₀,z₁) ≤ P, P,
///     select(narrow, clamp(centre, P, Q), N / select(narrow, 1, d))))
/// ```
///
/// **The divisor.** The quotient's arm is discarded wherever `narrow`
/// holds, so what it divides by there is free — and it must not be `d`.
/// An integrand whose slope the e-graph can prove zero (a literal `0`, an
/// argument the variable does not reach, a band of provably zero height)
/// makes `d` provably `0`, and the algebra's `x·recip(x) = 1` and
/// `(x·a)/a = x` hold for every `x` but zero: applied to a zero divisor
/// they merge the quotient's class with `1` and with every `x` whose
/// product with zero the graph holds, and what is built from those classes
/// collapses — a whole chord's area extracted as the constant `0`, measured
/// (`a_literal_slope_is_its_exact_area` in
/// `pixelflow-core/tests/area_adversarial.rs`). A `Select` guarding
/// the *result* does not help, because the e-graph reasons about the
/// quotient's class whatever consumes it; the divisor itself has to be one
/// no rule can prove zero. `select(narrow, 1, d)` is `1` wherever `d` is
/// small, and `d` wherever the quotient is read.
///
/// **Floating point.**
/// - A span wholly above the band gives exactly `Q`, and wholly below it
///   exactly `P`: both are decided by comparison before any division, so
///   neither depends on `N` and `d` rounding alike — which an optimizer free
///   to re-derive `d` as, say, the slope times the interval's length would
///   not preserve.
/// - A span no wider than [`DEGENERATE_SPAN`] of the band takes the clamp at
///   its centre, off by at most an eighth of the span.
/// - Otherwise `N/d`. Each term of `N` is a monotone image of `z₁` less
///   the same image of `z₀`, times a factor (`½(b + a)`, `Q` or `P`) at most
///   `max(|P|, |Q|)` in magnitude; each difference is exact (Sterbenz) or
///   rounds once relative to itself. For a band with `0 ≤ P` every factor is
///   non-negative, so the three terms share the sign of `d`, do not cancel,
///   and the quotient is the mean over the *rounded* span to a few ulps of
///   itself. A band reaching below zero can make them cancel, and the
///   quotient is then good to a few ulps of `max(|P|, |Q|)` instead — each
///   term's rounding is at most that times `|d|`. Either way the span's own
///   rounding moves the mean by at most the larger endpoint error, `clamp`
///   being 1-Lipschitz. The one condition is that `N` and `d` are formed
///   from the same `z₀` and `z₁`; an optimizer that re-derived `d`
///   independently would add a relative error of `ulp(z)/|d|`, which is
///   what the degenerate arm's width bounds.
/// - `Q·(…)` is emitted bare when `Q` is 1, and the `P` term not at all when
///   `P` is 0: `x·1 = x` exactly, and `0·(min(z₁,0) − min(z₀,0))` is `0` for
///   every finite argument.
#[must_use]
pub fn mean_of_clamp(arena: &mut ExprArena, sweep: Sweep, band: Band) -> ExprId {
    let Sweep { from, to, centre } = sweep;
    let (p, q) = (arena.push_const(band.lower), arena.push_const(band.upper));
    let a = clamp(arena, from, [p, q]);
    let b = clamp(arena, to, [p, q]);

    let half = arena.push_const(0.5);
    let width = arena.push_binary(OpKind::Sub, b, a);
    let half_width = arena.push_binary(OpKind::Mul, half, width);
    let span = arena.push_binary(OpKind::Add, b, a);
    let square = arena.push_binary(OpKind::Mul, half_width, span);

    let above_to = arena.push_binary(OpKind::Max, to, q);
    let above_from = arena.push_binary(OpKind::Max, from, q);
    let above = arena.push_binary(OpKind::Sub, above_to, above_from);
    let above = scaled(arena, above, band.upper);
    let mut numerator = arena.push_binary(OpKind::Add, square, above);
    if band.lower != 0.0 {
        let below_to = arena.push_binary(OpKind::Min, to, p);
        let below_from = arena.push_binary(OpKind::Min, from, p);
        let below = arena.push_binary(OpKind::Sub, below_to, below_from);
        let below = scaled(arena, below, band.lower);
        numerator = arena.push_binary(OpKind::Add, numerator, below);
    }

    let d = arena.push_binary(OpKind::Sub, to, from);
    let magnitude = arena.push_unary(OpKind::Abs, d);
    let threshold = arena.push_const(DEGENERATE_SPAN * (band.upper - band.lower));
    let narrow = arena.push_binary(OpKind::Le, magnitude, threshold);
    // Never `d` itself where `d` may be zero, not even in the arm `narrow`
    // discards: see "The divisor" in the doc above.
    let one = arena.push_const(1.0);
    let divisor = arena.push_ternary(OpKind::Select, narrow, one, d);
    let general = arena.push_binary(OpKind::Div, numerator, divisor);
    let degenerate = clamp(arena, centre, [p, q]);
    let inside = arena.push_ternary(OpKind::Select, narrow, degenerate, general);

    let highest = arena.push_binary(OpKind::Max, from, to);
    let under = arena.push_binary(OpKind::Le, highest, p);
    let at_most_p = arena.push_ternary(OpKind::Select, under, p, inside);

    let lowest = arena.push_binary(OpKind::Min, from, to);
    let over = arena.push_binary(OpKind::Ge, lowest, q);
    arena.push_ternary(OpKind::Select, over, q, at_most_p)
}

/// `τ(δ) = δ / max(step + √max(step² + bend·δ, 0), floor)`: the parameter at
/// which a certified [`Rise`] reaches `δ` — the one definition, which an
/// author's integrand spells and [`IntervalFold::arc_moment`] evaluates at
/// its ends. The rule that closes the integral reads back what this writes
/// as well as an author's quotient `δ/d`, and a pin builds its integrand
/// here (`ArcMoment`, `an_arc_built_by_the_definition_closes`).
///
/// **Law.** `q(t) = δ` is `bend·t² + 2·step·t − δ = 0`, whose increasing
/// root `(√(step² + bend·δ) − step)/bend` is, rationalized, `τ(δ)` — a form
/// that holds at `bend = 0` too, and loses nothing to cancellation. For a
/// certified rise, and up to the floor below:
/// - on `[0, q(1)]`, `τ` is `q`'s inverse, so `τ(q(t)) = t`;
/// - below it `τ < 0` (a negative numerator over a positive denominator),
///   and above it `τ > 1` — past the arc's end the increasing root is past
///   `1`, and past the height a falling parabola (`bend < 0`) peaks at, the
///   radicand's floor leaves `δ/step`, which is past `1` there because the
///   peak, at `t = −step/bend ≥ 1`, is at least `step` high;
/// - so `[0 ≤ τ(δ) < 1]` is `[0 ≤ δ < q(1)]`: the arc's band.
///
/// **The floors.** The radicand's ([`RADICAND_FLOOR`]) is active only off
/// the band, where no real root exists. The denominator's, `floor`, is
/// active only where `step + √(step² + bend·δ) < floor`; inside the band the
/// true root `t*` is then at least `δ/floor`, so `δ ≤ floor·t* ≤ floor`.
/// The root is exact at every height the band holds but a sliver of
/// `floor ≤ 2⁻¹⁰⁰` at its start, and never divides by zero.
///
/// **Floating point.** The product, the sum, the square root, the
/// reciprocal and the product by it each round once, so `τ` is good to a
/// few ulps of itself away from the band's start; the parameter's absolute
/// resolution is `2⁻²⁴` near `1`, which is what an arc's length multiplies
/// ([`IntervalFold::arc_moment`]). The floor is normal ([`RootFloor`]), so
/// `1/denominator ≤ 2¹²⁶` is finite and `δ` times it is never `0·∞`.
///
/// **Emitted as `δ·(1/d)`**, the reciprocal an exact `Div`, not as `δ/d`.
/// Where `d` does not vary — the bend is zero, so the radicand drops `δ` —
/// the e-graph's `MulRecip` canonicalization turns `δ/d` into `δ·recip(d)`,
/// computed once and priced below a quotient per sample; and `recip` is an
/// *estimate* (CLAUDE.md, "Floating point at the edges"). Spelled with
/// `1/d`, the reciprocal's class holds the exact quotient too, which the
/// latency prior prices below `Recip`, so the one computed once is exact.
/// The bend need not be zero when the rule fires: measured, a line whose
/// two steps are one uniform read twice closes while `a = s − s` is still a
/// difference, the algebra proves it zero afterwards, and `δ/d` extracted a
/// `recip` — `3.9e-2` of coverage wrong at AVX2, `3.5e-3` at AVX-512
/// (`pixelflow-core/tests/arc_adversarial.rs`). All-literal, `1/d` is folded
/// by `ConstantFold` like any other quotient of finite literals, and a
/// literal zero bend drops `bend·δ` here so that happens in the closing
/// phase.
#[must_use]
pub fn monotone_root(arena: &mut ExprArena, delta: ExprId, rise: Rise, floor: RootFloor) -> ExprId {
    let square = arena.push_binary(OpKind::Mul, rise.step, rise.step);
    // A float pattern matches by `==`, so `-0.0` is a zero bend too.
    let radicand = match arena.node(rise.bend) {
        ExprNode::Const(0.0) => square,
        _ => {
            let reach = arena.push_binary(OpKind::Mul, rise.bend, delta);
            arena.push_binary(OpKind::Add, square, reach)
        }
    };
    let real = arena.push_const(RADICAND_FLOOR);
    let radicand = arena.push_binary(OpKind::Max, radicand, real);
    let root = arena.push_unary(OpKind::Sqrt, radicand);
    let denominator = arena.push_binary(OpKind::Add, rise.step, root);
    let floor = arena.push_const(floor.get());
    let denominator = arena.push_binary(OpKind::Max, denominator, floor);
    let one = arena.push_const(1.0);
    let reciprocal = arena.push_binary(OpKind::Div, one, denominator);
    arena.push_binary(OpKind::Mul, delta, reciprocal)
}

/// One end of a cut inside the interval: a value the arena computes, or an
/// end of the interval itself, known here as a literal.
#[derive(Clone, Copy)]
enum End {
    Literal(f32),
    Value(ExprId),
}

impl End {
    fn id(self, arena: &mut ExprArena) -> ExprId {
        match self {
            Self::Literal(v) => arena.push_const(v),
            Self::Value(id) => id,
        }
    }

    /// `½·end`, folded when the end is a literal (exact either way).
    fn half(self, arena: &mut ExprArena) -> ExprId {
        match self {
            Self::Literal(v) => arena.push_const(0.5 * v),
            Self::Value(id) => {
                let half = arena.push_const(0.5);
                arena.push_binary(OpKind::Mul, half, id)
            }
        }
    }
}

impl IntervalFold {
    /// `∫_lo^hi [cut] du`: the length of the cut inside the interval.
    ///
    /// **Law.** One-sided, the length is a clamp of the distance from the
    /// open end: `clamp(U − lo, 0, hi − lo)` below an upper bound `U`,
    /// `clamp(hi − L, 0, hi − lo)` above a lower bound `L`. Two-sided it is
    /// `max(clamp(U, lo, hi) − clamp(L, lo, hi), 0)`; the outer `max` is what
    /// keeps a cut with `L > U` — empty — from measuring negative. Uncut, it
    /// is the interval's length.
    ///
    /// The one-sided form is a clamp of an affine function of whatever `U`
    /// or `L` is affine in, which is what lets the integral this is nested
    /// in close by [`IntervalFold::clamp_moment`] in turn.
    ///
    /// **Floating point.** Clamps are exact; the one subtraction rounds once.
    #[must_use]
    pub fn measure(self, arena: &mut ExprArena, cut: Cut) -> ExprId {
        let zero = arena.push_const(0.0);
        let length = arena.push_const(self.length());
        match (cut.lower, cut.upper) {
            (None, None) => length,
            (None, Some(upper)) => {
                let reach = offset_by(arena, upper, self.lo());
                clamp(arena, reach, [zero, length])
            }
            (Some(lower), None) => {
                let hi = arena.push_const(self.hi());
                let reach = arena.push_binary(OpKind::Sub, hi, lower);
                clamp(arena, reach, [zero, length])
            }
            (Some(lower), Some(upper)) => {
                let (lo, hi) = (arena.push_const(self.lo()), arena.push_const(self.hi()));
                let upper = clamp(arena, upper, [lo, hi]);
                let lower = clamp(arena, lower, [lo, hi]);
                let kept = arena.push_binary(OpKind::Sub, upper, lower);
                arena.push_binary(OpKind::Max, kept, zero)
            }
        }
    }

    /// Narrow an integral to a cut, keeping the interval:
    /// `∫_lo^hi [cut]·f(u) du = s · ∫_lo^hi f(m + s·(u − c)) du`.
    ///
    /// **Law.** Clip the cut into the interval, `a′ = clamp(L, lo, hi)` and
    /// `b′ = clamp(U, lo, hi)` (an absent end is the interval's own), and
    /// let `h = max(b′ − a′, 0)`, `m = ½a′ + ½b′`, `c` the interval's
    /// midpoint and `s = h/(hi − lo)`. Then `u ↦ m + s(u − c)` maps
    /// `[lo, hi)` onto `[a′, b′)` with Jacobian `s`, which is the
    /// substitution rule. An empty cut gives `s = 0`, and `0·∫ f(m)` is `0`
    /// for any `f` finite at a point of the interval — `m` is one.
    ///
    /// Narrowing keeps the interval rather than making a new one because an
    /// interval's ends are literals and a cut's are values: the cut becomes
    /// the substitution, and the fold's domain never has to hold a value.
    ///
    /// `variable` is the integration variable's leaf in `arena`.
    ///
    /// **Floating point.** `a′` and `b′` are exact clamps; `h`, `m` and the
    /// point each round once or twice, which moves where the integrand is
    /// read by a last-bit amount. `s` is `h` times the literal
    /// `1/(hi − lo)` — `h` itself for a unit interval — and `u − c` is `u`
    /// when `c` is 0; both elisions are exact.
    #[must_use]
    pub fn narrowing(self, arena: &mut ExprArena, cut: Cut, variable: ExprId) -> Narrowing {
        let (lo, hi) = (arena.push_const(self.lo()), arena.push_const(self.hi()));
        let lower = match cut.lower {
            Some(value) => End::Value(clamp(arena, value, [lo, hi])),
            None => End::Literal(self.lo()),
        };
        let upper = match cut.upper {
            Some(value) => End::Value(clamp(arena, value, [lo, hi])),
            None => End::Literal(self.hi()),
        };
        let (a, b) = (lower.id(arena), upper.id(arena));
        let kept = arena.push_binary(OpKind::Sub, b, a);
        let zero = arena.push_const(0.0);
        let h = arena.push_binary(OpKind::Max, kept, zero);
        let (half_a, half_b) = (lower.half(arena), upper.half(arena));
        let m = arena.push_binary(OpKind::Add, half_a, half_b);
        let scale = per_unit_length(arena, h, self.length());
        let centred = offset_by(arena, variable, self.midpoint());
        let step = arena.push_binary(OpKind::Mul, scale, centred);
        let point = arena.push_binary(OpKind::Add, m, step);
        Narrowing { scale, point }
    }

    /// `∫_lo^hi clamp(k·u + c, P, Q) du`, `u` free in neither `k` nor `c`.
    ///
    /// **Law.** The argument sweeps `z₀ = k·lo + c` to `z₁ = k·hi + c`, so
    /// the integral is the interval's length times the mean of the clamp
    /// over that span — [`mean_of_clamp`], whose degenerate arm takes the
    /// clamp at `k·(lo + hi)/2 + c`, the interval's midpoint.
    ///
    /// **Floating point.** See [`mean_of_clamp`]; the three arguments round
    /// once or twice each. `k·0 + c` is emitted as `c` (exact for a finite
    /// `k`) and a unit length's product is not emitted at all.
    #[must_use]
    pub fn clamp_moment(self, arena: &mut ExprArena, integrand: Affine, band: Band) -> ExprId {
        let sweep = Sweep {
            from: affine_at(arena, integrand, self.lo()),
            to: affine_at(arena, integrand, self.hi()),
            centre: affine_at(arena, integrand, self.midpoint()),
        };
        let mean = mean_of_clamp(arena, sweep, band);
        scaled(arena, mean, self.length())
    }

    /// `∫_lo^hi [0 ≤ T < 1]·clamp(c₀ + x(T), P, Q) du`, `T = τ_y(u + D₀)`:
    /// the area a monotone arc bounds, integrated along the height it
    /// rises through.
    ///
    /// `τ_y` and `τ_x` are [`monotone_root`] over `arc.y` and `arc.x`, `x`
    /// and `y` their [`Rise`]s, `D₀`, `c₀` the arc's `height` and `offset`,
    /// and `[P, Q]` the band. With `b, a` the `y` rise's step and bend and
    /// `β, α` the `x` rise's, the integral is
    ///
    /// ```text
    /// P·Δ(t_a, p) + ½(x̂(p) + x̂(q))·Δ(p, q) + K·(q − p)³ + Q·Δ(q, t_b)
    ///
    /// t_a = clamp(τ_y(D₀ + lo), 0, 1)      t_b = clamp(τ_y(D₀ + hi), 0, 1)
    /// p   = clamp(τ_x(P − c₀), t_a, t_b)   q   = clamp(τ_x(Q − c₀), t_a, t_b)
    /// x̂(t) = c₀ + x(t)    Δ(s, t) = y(t) − y(s) = (t − s)·(2b + a·(s + t))
    /// K   = (β·a − b·α)/3
    /// ```
    ///
    /// **Law.** Where the band holds, `τ_y` is `y`'s inverse, so substitute
    /// `u + D₀ = y(t)`, `du = y′(t) dt`: the interval and the band leave
    /// `t ∈ [t_a, t_b]`. `x̂` rises, so the clamp is `P` before `p`, `x̂`
    /// between `p` and `q` — which `τ_x` finds, `x̂` being `c₀` plus a rise —
    /// and `Q` after `q`: three pieces, `P·∫y′`, `∫x̂·y′` and `Q·∫y′`. The
    /// outer two are the heights `Δ`; the middle is `∫x̂ dy` along the sub-arc
    /// from `p` to `q`, which is its chord's trapezoid plus the area between
    /// the sub-arc and its chord — `K·(q − p)³` for any sub-arc of a
    /// quadratic, whose second derivative is the same everywhere. No case
    /// split: an arc wholly left of the band (`p = q = t_b`), wholly right
    /// of it (`p = q = t_a`), a horizontal arc (`b = a = 0`, every `Δ`
    /// zero), a vertical one (`β = α = 0`, whose roots divide by the floor
    /// and saturate to `t_a` or `t_b`) and a band that misses the interval
    /// (`t_a = t_b`) each land on the formula through a clamp.
    ///
    /// **Certificates.** The caller's side conditions: both rises certified
    /// (`b ≥ 0`, `b + a ≥ 0`, and the same of `β, α`), and `u` free in `D₀`,
    /// `c₀` and every step and bend. A rise that falls somewhere makes `τ`
    /// the wrong root and `x̂` a clamp that is not three pieces, so an
    /// integrand whose steps are not certified — floored at [`STEP_FLOOR`]
    /// where the e-graph can see it — has another integral, not this one.
    /// Under them the formula is exact over ℝ but for a height of at most
    /// `floor ≤ 2⁻¹⁰⁰` at the arc's start, where [`monotone_root`] is not
    /// the inverse.
    ///
    /// **Floating point.** Every divisor is a root's `max(·, floor)`, so
    /// nothing divides by zero — or by anything a rule could prove zero —
    /// and `/3` is a product by the literal `⅓`. What rounds is where the
    /// parameters land: `t_a, t_b, p, q` resolve to about `2⁻²⁴` near `1`,
    /// and a shift of a parameter by `ε` moves a height by up to `ε·y′`, at
    /// most twice the arc's extent — so the error grows with the arc's
    /// length, and a pixel wholly inside a region no longer sums to exactly
    /// `1` the way a chord's clamp moment does. `c₀` carries the rounding of
    /// the coordinates it was computed from. Measured against exact
    /// clipping: `2⁻²²·(1 + |X| + |Y| + 2·extent)` bounds it
    /// (`pixelflow-core/tests/arc_oracle.rs`).
    ///
    /// `P·Δ(t_a, p)` is not emitted for `P = 0`, nor the product by `Q` for
    /// `Q = 1` (`x·1 = x` exactly), and `P − c₀` is `−c₀` for `P = 0`.
    #[must_use]
    pub fn arc_moment(self, arena: &mut ExprArena, arc: MonotoneArc, band: Band) -> ExprId {
        let MonotoneArc {
            height,
            y,
            offset,
            x,
            floor,
        } = arc;
        let (zero, one) = (arena.push_const(0.0), arena.push_const(1.0));
        let from = shifted(arena, height, self.lo());
        let to = shifted(arena, height, self.hi());
        let t_a = monotone_root(arena, from, y, floor);
        let t_a = clamp(arena, t_a, [zero, one]);
        let t_b = monotone_root(arena, to, y, floor);
        let t_b = clamp(arena, t_b, [zero, one]);

        let enter = less(arena, band.lower, offset);
        let leave = less(arena, band.upper, offset);
        let p = monotone_root(arena, enter, x, floor);
        let p = clamp(arena, p, [t_a, t_b]);
        let q = monotone_root(arena, leave, x, floor);
        let q = clamp(arena, q, [t_a, t_b]);

        let (x_p, x_q) = (along(arena, offset, x, p), along(arena, offset, x, q));
        let ends = arena.push_binary(OpKind::Add, x_p, x_q);
        let half = arena.push_const(0.5);
        let mean = arena.push_binary(OpKind::Mul, half, ends);
        let rise = climb(arena, y, [p, q]);
        let trapezoid = arena.push_binary(OpKind::Mul, mean, rise);

        let x_over_y = arena.push_binary(OpKind::Mul, x.step, y.bend);
        let y_over_x = arena.push_binary(OpKind::Mul, y.step, x.bend);
        let twist = arena.push_binary(OpKind::Sub, x_over_y, y_over_x);
        let third = arena.push_const(1.0 / 3.0);
        let k = arena.push_binary(OpKind::Mul, third, twist);
        let width = arena.push_binary(OpKind::Sub, q, p);
        let square = arena.push_binary(OpKind::Mul, width, width);
        let cube = arena.push_binary(OpKind::Mul, square, width);
        let bow = arena.push_binary(OpKind::Mul, k, cube);
        let inside = arena.push_binary(OpKind::Add, trapezoid, bow);

        let past = climb(arena, y, [q, t_b]);
        let past = scaled(arena, past, band.upper);
        let mut total = arena.push_binary(OpKind::Add, inside, past);
        if band.lower != 0.0 {
            let before = climb(arena, y, [t_a, p]);
            let before = scaled(arena, before, band.lower);
            total = arena.push_binary(OpKind::Add, total, before);
        }
        total
    }
}

/// `x + shift`, or `x` itself when the shift is 0.
fn shifted(arena: &mut ExprArena, x: ExprId, shift: f32) -> ExprId {
    if shift == 0.0 {
        return x;
    }
    let shift = arena.push_const(shift);
    arena.push_binary(OpKind::Add, x, shift)
}

/// `level − x`, or `−x` when the level is 0.
fn less(arena: &mut ExprArena, level: f32, x: ExprId) -> ExprId {
    if level == 0.0 {
        return arena.push_unary(OpKind::Neg, x);
    }
    let level = arena.push_const(level);
    arena.push_binary(OpKind::Sub, level, x)
}

/// `offset + t·(step + step + bend·t)`: a rise at `t`, from `offset` — the
/// same composition an author's arc is spelled in.
fn along(arena: &mut ExprArena, offset: ExprId, rise: Rise, t: ExprId) -> ExprId {
    let twice = arena.push_binary(OpKind::Add, rise.step, rise.step);
    let bent = arena.push_binary(OpKind::Mul, rise.bend, t);
    let slope = arena.push_binary(OpKind::Add, twice, bent);
    let reach = arena.push_binary(OpKind::Mul, t, slope);
    arena.push_binary(OpKind::Add, offset, reach)
}

/// `Δ(s, t) = q(t) − q(s) = (t − s)·(2·step + bend·(s + t))`: how far a rise
/// climbs between two parameters, as one product, so a zero-length span is
/// exactly zero.
fn climb(arena: &mut ExprArena, rise: Rise, [s, t]: [ExprId; 2]) -> ExprId {
    let width = arena.push_binary(OpKind::Sub, t, s);
    let twice = arena.push_binary(OpKind::Add, rise.step, rise.step);
    let span = arena.push_binary(OpKind::Add, s, t);
    let bent = arena.push_binary(OpKind::Mul, rise.bend, span);
    let slope = arena.push_binary(OpKind::Add, twice, bent);
    arena.push_binary(OpKind::Mul, width, slope)
}

/// `min(max(z, lower), upper)` — the composition `Kernel::clamp` builds, so a
/// closed form's clamp is the same term an author's is.
fn clamp(arena: &mut ExprArena, z: ExprId, [lower, upper]: [ExprId; 2]) -> ExprId {
    let floored = arena.push_binary(OpKind::Max, z, lower);
    arena.push_binary(OpKind::Min, floored, upper)
}

/// `factor·x`, or `x` itself when the factor is 1.
fn scaled(arena: &mut ExprArena, x: ExprId, factor: f32) -> ExprId {
    if factor == 1.0 {
        return x;
    }
    let factor = arena.push_const(factor);
    arena.push_binary(OpKind::Mul, factor, x)
}

/// `x / length`: `x` itself when the length is 1, else `x` times the literal
/// `1/length`, rounded once here.
///
/// A product and not a quotient: a `Div` by a literal is what the e-graph's
/// `MulRecip` canonicalization rewrites to `x·recip(length)`, and one
/// `recip` shared by every use is cheaper under the DAG objective than an
/// exact `Div` each — but `recip` is an *estimate* (CLAUDE.md, "Floating
/// point at the edges"). A literal reciprocal costs one rounding and no
/// estimate; an interval whose reciprocal length overflows keeps the
/// quotient.
fn per_unit_length(arena: &mut ExprArena, x: ExprId, length: f32) -> ExprId {
    if length == 1.0 {
        return x;
    }
    let reciprocal = 1.0 / length;
    if !reciprocal.is_finite() {
        let length = arena.push_const(length);
        return arena.push_binary(OpKind::Div, x, length);
    }
    let reciprocal = arena.push_const(reciprocal);
    arena.push_binary(OpKind::Mul, reciprocal, x)
}

/// `x − shift`, or `x` itself when the shift is 0.
fn offset_by(arena: &mut ExprArena, x: ExprId, shift: f32) -> ExprId {
    if shift == 0.0 {
        return x;
    }
    let shift = arena.push_const(shift);
    arena.push_binary(OpKind::Sub, x, shift)
}

/// `slope·at + offset`, or `offset` when `at` is 0.
fn affine_at(arena: &mut ExprArena, affine: Affine, at: f32) -> ExprId {
    if at == 0.0 {
        return affine.offset;
    }
    let at = arena.push_const(at);
    let term = arena.push_binary(OpKind::Mul, affine.slope, at);
    arena.push_binary(OpKind::Add, term, affine.offset)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The floor is `2⁻¹⁰⁰`, and nothing above it is one.** A floor must
    /// be normal — the root divides by it, and denormals-are-zero reads a
    /// subnormal as `0` — and at most `2⁻¹⁰⁰`, the height below which the
    /// root may not be the rise's inverse.
    #[test]
    fn a_root_floor_is_normal_and_at_most_two_to_the_minus_100() {
        assert_eq!(ROOT_FLOOR, 2.0f32.powi(-100));
        assert_eq!(
            RootFloor::new(ROOT_FLOOR).map(RootFloor::get),
            Some(ROOT_FLOOR)
        );
        assert!(RootFloor::new(ROOT_FLOOR / 8.0).is_some());
        assert!(RootFloor::new(f32::MIN_POSITIVE).is_some());
        let refused = [
            2.0 * ROOT_FLOOR,
            f32::MIN_POSITIVE / 2.0,
            f32::from_bits(1),
            0.0,
            -0.0,
            -ROOT_FLOOR,
            f32::NAN,
        ];
        for floor in refused {
            assert_eq!(RootFloor::new(floor), None, "{floor:e}");
        }
    }
}
