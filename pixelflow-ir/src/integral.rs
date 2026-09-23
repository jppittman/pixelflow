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
//! its own doc, and no formula divides where a saturated or degenerate case
//! could reach the quotient: those cases are decided by a comparison first,
//! and a `Select` discards whatever the quotient computed there.
//!
//! [`mean_of_clamp`] is written once, here. A caller that needs the mean of a
//! clamp over a span — [`IntervalFold::clamp_moment`] today, a glyph's
//! pixel/half-plane coverage tomorrow — calls it rather than restating it
//! (CLAUDE.md, "one definition, imported, not restated").

use crate::arena::{ExprArena, ExprId};
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
/// select(min(z₀,z₁) ≥ Q, Q,
///   select(max(z₀,z₁) ≤ P, P,
///     select(|d| ≤ DEGENERATE_SPAN·(Q − P), clamp(centre, P, Q), N/d)))
/// ```
///
/// **Floating point.**
/// - A span wholly above the band gives exactly `Q`, and wholly below it
///   exactly `P`: both are decided by comparison before any division, so
///   neither depends on `N` and `d` rounding alike — which an optimizer free
///   to re-derive `d` as, say, the slope times the interval's length would
///   not preserve.
/// - A span no wider than [`DEGENERATE_SPAN`] of the band takes the clamp at
///   its centre, off by at most an eighth of the span.
/// - Otherwise `N/d`. Each term of `N` is a difference of one monotone image
///   of `z₀` and `z₁`, so each has the sign of `d` and the three do not
///   cancel; each difference is exact (Sterbenz) or rounds once relative to
///   itself. The quotient is therefore the mean over the *rounded* span to a
///   few ulps, and the span's own rounding moves it by at most the larger
///   endpoint error, `clamp` being 1-Lipschitz. The one condition is that
///   `N` and `d` are formed from the same `z₀` and `z₁`; an optimizer that
///   re-derived `d` independently would add a relative error of
///   `ulp(z)/|d|`, which is what the degenerate arm's width bounds.
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
    let general = arena.push_binary(OpKind::Div, numerator, d);
    let degenerate = clamp(arena, centre, [p, q]);
    let magnitude = arena.push_unary(OpKind::Abs, d);
    let threshold = arena.push_const(DEGENERATE_SPAN * (band.upper - band.lower));
    let narrow = arena.push_binary(OpKind::Le, magnitude, threshold);
    let inside = arena.push_ternary(OpKind::Select, narrow, degenerate, general);

    let highest = arena.push_binary(OpKind::Max, from, to);
    let under = arena.push_binary(OpKind::Le, highest, p);
    let at_most_p = arena.push_ternary(OpKind::Select, under, p, inside);

    let lowest = arena.push_binary(OpKind::Min, from, to);
    let over = arena.push_binary(OpKind::Ge, lowest, q);
    arena.push_ternary(OpKind::Select, over, q, at_most_p)
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
