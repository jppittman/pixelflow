//! Integration as e-graph rewrites.
//!
//! Integration works the way differentiation does
//! (docs/plans/2026-09-23-an-integral-is-a-fold.md §3). The author writes the
//! integral — `k.area()`, as they write `k.dx()` — and the calculus is rules
//! in the e-graph, as the chain rule is for `Dwrt` (`derivative`). Whatever
//! the rules leave unclosed is legalized before a backend sees it:
//! `passes::resolve` replaces a surviving integral by its quadrature, and a
//! test pins that none reaches the emitter.
//!
//! ```text
//! ∫_lo^hi C·Π[Aᵢ ⋈ Bᵢ]·R du = C·s·∫_lo^hi R(m + s(u − c)) du    (narrow)
//! ∫_lo^hi clamp(k·u + c, P, Q) du = (hi − lo)·mean_{z₀..z₁} clamp  (clamp moment)
//! ∫_lo^hi [0 ≤ T < 1]·clamp(x(T) + c, P, Q) du = arc_moment      (arc moment)
//! ```
//!
//! `T` in the last a monotone quadratic arc's parameter at the height
//! `u + D₀` — the curved piece of a glyph, and its straight one — and, from
//! `fold_rules`, the one rule both domains share: `∫ c·f = c·∫ f`
//! (`FactorFold`). Nothing else is here yet: the constant, linearity,
//! select, interchange and power-moment rules of the plan's table close no
//! integral a chord or an arc needs, so each waits for the kernel that does
//! (CLAUDE.md, "subtract before you add").
//!
//! **The formulas are not written here.** An interval keeps its ends to
//! itself, so what an integral closes *to* lives beside it in
//! `pixelflow_ir::integral`, the way its quadrature does; these rules decide
//! *when* one applies and build it over the classes they matched. A rule
//! writes its closed form into a template [`ExprArena`] whose `Var(i)`
//! leaves are the classes it recognized, and splices that into a
//! [`Plan`](super::fold_rules::Plan).
//!
//! **The recognizers read classes, not spellings.** An integrand arrives as
//! the author wrote it, and saturation then multiplies its spellings —
//! `a − b` beside `a + (−b)`, `k·u + c` beside `MulAdd(k, u, c)`, operands
//! commuted. Each recognizer tries every node of a class and memoises per
//! class, so a spelling the algebra added is as recognizable as the one the
//! author wrote, and a DAG costs its size.
//!
//! **Closing comes first.** Every rule here is in
//! [`RuleSet::runtime`](super::RuleSet::runtime), and the graph runs the
//! integration family — [`FactorFold`](super::FactorFold), these three, and
//! `ConstantFold` — to a fixpoint on the freshly inserted graph before the
//! full rule set sees it, counted against the same application budget
//! (`EGraph::saturate_bounded`). A chord's derivation takes three rounds of
//! that family and nothing else, an arc's one and a confirming one;
//! interleaved with the algebra, the class cap a glyph already reaches in
//! its first round would stop it half-closed. A graph with no integral
//! skips the phase, so no other kernel's saturation changes.

use alloc::boxed::Box;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::rc::Rc;
use alloc::vec::Vec;

use pixelflow_ir::integral::{
    Affine as IrAffine, Band, Cut, MonotoneArc, RADICAND_FLOOR, Rise as IrRise, RootFloor,
    STEP_FLOOR,
};
use pixelflow_ir::{Binder, ExprArena, ExprId, Fold, OpKind};

use super::fold_rules::{
    Copying, FactorFold, Factors, HeadRef, PlanBuilder, Substitution, is_integral,
};
use super::graph::EGraph;
use super::node::{EClassId, ENode};
use super::ops;
use super::rewrite::{Rewrite, RewriteAction};
use super::rules::RuleId;

/// `∫_lo^hi C·Π[Aᵢ ⋈ Bᵢ]·R(u) du = C·s·∫_lo^hi R(m + s·(u − c)) du` —
/// narrow an integral to where its indicators hold, `C` the factors the
/// variable does not reach.
///
/// **Law.** Each indicator `[A ⋈ B]` — `Select(A ⋈ B, 1, 0)`, `⋈` one of
/// `<`, `≤`, `>`, `≥` — whose difference is affine in the variable,
/// `A − B = c·u + d`, bounds `u` by its root `t = −d/c`: from above when
/// `A − B` must be negative and `c > 0`, or positive and `c < 0`; from below
/// otherwise. A negative `c` flips the side, which is the whole of the sign
/// case. The product of the bounds is `[max Lᵢ < u < min Uⱼ]`, and the
/// integral of `R` over it is the reparametrization
/// `pixelflow_ir::IntervalFold::narrowing` writes. With nothing left in the
/// product the integral is the cut's length (`IntervalFold::measure`) —
/// one-sided, a clamp, which is what lets an integral this one is nested in
/// close by [`ClampMoment`] in turn.
///
/// **Side conditions.**
/// - `c` is a nonzero finite *literal*, read off the class constant fact.
///   A slope that is a value — a band height, a table entry — would put a
///   value in the root's denominator and a sign test on it in the rule's
///   choice of side; it is refused rather than guessed.
/// - `u ∉ var(d)` holds by construction: the recognizer admits a class
///   into `d` only when the class variance fact clears it of `u`.
/// - Only an interval: `<` and `≤` bound a real variable identically,
///   because the set they disagree on is one point, and a point has no
///   length. Over a range the same point is a whole term, so no range fold
///   is ever narrowed this way.
/// - `C` stays outside the integral rather than being read at the point,
///   which would leave `∫ C` for the constant rule — not built — wherever
///   the product holds nothing else. [`FactorFold`] takes `C` out too, but
///   only of the integral it matched, never of the one this builds.
/// - The rest `R` is copied through a representative with no integral in it
///   where the class has one ([`Copying::Anything`] otherwise): an inner
///   integral already closed is copied closed.
///
/// **Floating point.** Exact over ℝ. The bounds are exact clamps of roots
/// that round once (`c = ±1` subtracts) or twice (any other `c` multiplies
/// by the literal `1/c` — never a `Div`, which the e-graph would be free to
/// turn into a shared `recip` estimate; see `quotient`); the cut's midpoint
/// and the reparametrized point round once or twice more. Each is a
/// last-bit shift of where the integrand is read, never of the measure's
/// sign: an empty cut is `max(·, 0) = 0` exactly.
pub struct NarrowInterval;

/// `∫_lo^hi clamp(k·u + c, P, Q) du = (hi − lo)·mean_{z ∈ [z₀, z₁]} clamp(z, P, Q)`,
/// `z₀ = k·lo + c`, `z₁ = k·hi + c`.
///
/// **Law.** The clamp's argument sweeps `[z₀, z₁]` affinely as `u` crosses
/// the interval, so the integral is the length times the mean of the clamp
/// over the sweep, which its antiderivative gives in closed form
/// (`pixelflow_ir::integral::mean_of_clamp`, the one definition).
///
/// **Side conditions.** `u ∉ var(k, c)`, by the recognizer's construction;
/// `P < Q`, both finite literals — a clamp with `P ≥ Q` is the constant `Q`
/// and has no band to average over. The clamp is recognized as
/// `min(max(z, P), Q)` (what `Kernel::clamp` builds) or `max(min(z, Q), P)`,
/// which are equal when `P < Q`, with either operand order.
///
/// `k` may be zero, and provably so — an argument the variable does not
/// reach is `0·u + c` — which makes the sweep `z₁ − z₀` provably zero too.
/// So the closed form never divides by the sweep where it can be zero, not
/// even in an arm a `Select` discards: an e-graph that proves a divisor zero
/// goes on to apply `x·recip(x) = 1` and `(x·a)/a = x` to it, which merge
/// the quotient with classes it is not equal to (`mean_of_clamp`, "The
/// divisor").
///
/// **Floating point.** See `mean_of_clamp`: a sweep wholly outside the band
/// is exactly `P` or `Q`, decided by comparison before any division; a
/// sweep narrower than `DEGENERATE_SPAN` of the band takes the midpoint; any
/// other is the antiderivative's difference quotient, a few ulps from the
/// mean over the rounded sweep (of `max(|P|, |Q|)`, for a band reaching
/// below zero).
pub struct ClampMoment;

/// `∫_lo^hi C·[0 ≤ T]·[T < 1]·clamp(s·R + c, P, Q) du = C·arc_moment`, with
/// `T = τ_y(u + D₀)` a monotone arc's parameter and `R = x(T)` its other
/// coordinate — the area left of the arc, integrated along the height it
/// rises through (`pixelflow_ir::IntervalFold::arc_moment`, the one
/// definition).
///
/// **Where it comes from.** An author writes a glyph piece's crossing as a
/// graph over `y` through the arc's own parameter:
/// `T = τ_y(y − y₀)` (`pixelflow_ir::integral::monotone_root`) is where the
/// arc reaches height `y`, `[0 ≤ T < 1]` its band, and `[x < x₀ + x(T)]` the
/// crossing. Over the pixel, [`FactorFold`] takes the band out of the inner
/// integral and [`NarrowInterval`] closes what is left to
/// `clamp(x₀ + x(T) − X + ½, 0, 1)` — the length of the pixel's row left of
/// the arc, which is Green's step — and leaves this rule the outer integral:
/// the substitution `y = y(t)` and a cubic moment. A line is the arc whose
/// bend is zero, so one integrand, and this one rule, serve every piece
/// (docs/plans/2026-09-23-an-integral-is-a-fold.md §8 step 5).
///
/// **Side conditions**, each read off the classes, every node of each
/// tried:
/// - The body's factors the variable reaches are exactly two indicators
///   and a clamp. The indicators are `Select(m, 1, 0)` of `0 ≤ T` and
///   `T < 1` in any of `≤ <` and `≥ >` spellings — strictness moves a
///   point, which has no length — over one class `T`.
/// - `T` holds `D / max(b + √max(b·b + a·D, 0), k)`, operands of `+`, `·`
///   and `max` in either order, `k` a normal literal no larger than
///   `2⁻¹⁰⁰` (`RootFloor`): a larger floor moves the root over a visible
///   height, and a subnormal one is zero under denormals-are-zero.
///   The radicand's floor is the literal `0`, and nothing else.
/// - `D = u + D₀`, affine in the variable with the literal slope `1`.
/// - `b` and `a` are certified: `b` holds `max(z, k)` with a literal
///   `k ≥ STEP_FLOOR` (or is such a literal), and `a` holds `e − b` with `e`
///   certified — the control polygon's steps, floored where the e-graph can
///   see it, which is what makes `T` the rise's inverse for every value a
///   table can hold rather than for the ones a host happened to write.
/// - The clamp is one [`ClampMoment`] reads, with literal band `[P, Q]`,
///   and its argument is affine in a class `R` with a positive literal
///   slope `s`: `R` holds `T·(β + β + α·T)`, `β` and `α` certified like `b`
///   and `a`. `R` is found among the classes the argument is spelled from,
///   and the argument is then read with `R` standing for the variable — so
///   `R + c` with `c = x₀ − X + ½` is recognized however narrowing spelled
///   it.
/// - `u` reaches none of `D₀`, `c`, `b`, `a`, `β`, `α`: the recognizer
///   admits a term only when the class variance fact clears it.
/// - Factors the variable does not reach multiply the result, as
///   [`NarrowInterval`] keeps them.
///
/// Anything else declines — a rule may miss an integral, never close one
/// wrongly — and so does a fold whose class already holds a member that is
/// not an integral: a rule has closed it already (or factored it, and the
/// factored integral closes where it is), and firing again on a spelling
/// the algebra added later would only add nodes.
///
/// **Floating point.** See `arc_moment`: no case split, no divisor that can
/// be zero, and an error that grows with the arc's length —
/// `2⁻²²·(1 + |X| + |Y| + 2·extent)` measured.
pub struct ArcMoment;

impl Rewrite for NarrowInterval {
    fn name(&self) -> &str {
        "narrow-interval"
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let ENode::Reduce { fold, body } = node else {
            return None;
        };
        let Fold::Interval(interval) = *fold else {
            return None;
        };
        let binder = fold.binder();
        let mut recognizer = Recognizer::new(egraph, binder);
        // Every product spelling of the body, and the body as one factor.
        let mut spellings: Vec<Factors> = egraph
            .nodes(*body)
            .iter()
            .filter_map(|spelling| Factors::of(egraph, spelling, &ops::Mul, binder))
            .collect();
        spellings.push(Factors {
            invariant: Vec::new(),
            variant: alloc::vec![egraph.find(*body)],
        });
        let Split {
            bounds,
            outside,
            rest,
        } = spellings
            .into_iter()
            .find_map(|factors| recognizer.bounds_of(factors))?;

        let mut template = Template::default();
        let cut = template.cut(&bounds)?;
        let mut plan = PlanBuilder::default();
        // What the variable does not reach multiplies the result, never the
        // integrand: `∫ c` is the constant rule, which nothing here closes.
        let mut product: Vec<HeadRef> = outside.iter().map(|&c| HeadRef::Class(c)).collect();
        if rest.is_empty() {
            let measure = interval.measure(&mut template.arena, cut);
            let [measure] = template.splice_into(&mut plan, [measure])?;
            product.push(measure);
            let root = plan.chain(&ops::Mul, &product)?;
            return Some(RewriteAction::Plan(plan.finish(root)));
        }

        let variable = template.operand(recognizer.leaf?)?;
        let narrowing = interval.narrowing(&mut template.arena, cut, variable);
        let [scale, point] = template.splice_into(&mut plan, [narrowing.scale, narrowing.point])?;
        let factors = rest
            .iter()
            .map(|&class| {
                let at_point = Substitution {
                    binder,
                    copying: Copying::Anything,
                    leaf: |_: &mut PlanBuilder, _| point,
                };
                plan.substitute(egraph, class, at_point)
            })
            .collect::<Option<Vec<HeadRef>>>()?;
        let narrowed = plan.chain(&ops::Mul, &factors)?;
        let integral = plan.reduce(*fold, narrowed);
        let narrowed = plan.op(&ops::Mul, alloc::vec![scale, integral]);
        product.push(narrowed);
        let root = plan.chain(&ops::Mul, &product)?;
        Some(RewriteAction::Plan(plan.finish(root)))
    }
}

impl Rewrite for ClampMoment {
    fn name(&self) -> &str {
        "clamp-moment"
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let ENode::Reduce { fold, body } = node else {
            return None;
        };
        let Fold::Interval(interval) = *fold else {
            return None;
        };
        let mut recognizer = Recognizer::new(egraph, fold.binder());
        let (argument, band) = clamps(egraph, *body)
            .into_iter()
            .find_map(|(argument, band)| Some((recognizer.affine(argument)?, band)))?;

        let mut template = Template::default();
        let integrand = IrAffine {
            slope: template.emit(&argument.slope)?,
            offset: template.emit(&argument.offset)?,
        };
        let moment = interval.clamp_moment(&mut template.arena, integrand, band);
        let mut plan = PlanBuilder::default();
        let [root] = template.splice_into(&mut plan, [moment])?;
        Some(RewriteAction::Plan(plan.finish(root)))
    }
}

impl Rewrite for ArcMoment {
    fn name(&self) -> &str {
        "arc-moment"
    }

    fn apply(&self, egraph: &EGraph, id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let ENode::Reduce { fold, body } = node else {
            return None;
        };
        let Fold::Interval(interval) = *fold else {
            return None;
        };
        if egraph.nodes(id).iter().any(|member| !is_integral(member)) {
            return None;
        }
        let binder = fold.binder();
        let (arc, outside) = egraph.nodes(*body).iter().find_map(|spelling| {
            let factors = Factors::of(egraph, spelling, &ops::Mul, binder)?;
            let arc = arc_integrand(egraph, binder, &factors.variant)?;
            Some((arc, factors.invariant))
        })?;

        let mut template = Template::default();
        let rise = |template: &mut Template, [step, bend]: &[Rc<Term>; 2]| {
            Some(IrRise {
                step: template.emit(step)?,
                bend: template.emit(bend)?,
            })
        };
        let integrand = MonotoneArc {
            height: template.emit(&arc.root.height)?,
            y: rise(&mut template, &arc.root.rise)?,
            offset: template.emit(&arc.offset)?,
            x: rise(&mut template, &arc.x)?,
            floor: arc.root.floor,
        };
        let moment = interval.arc_moment(&mut template.arena, integrand, arc.band);
        let mut plan = PlanBuilder::default();
        let [moment] = template.splice_into(&mut plan, [moment])?;
        let mut product: Vec<HeadRef> = outside.iter().map(|&c| HeadRef::Class(c)).collect();
        product.push(moment);
        let root = plan.chain(&ops::Mul, &product)?;
        Some(RewriteAction::Plan(plan.finish(root)))
    }
}

/// The integration rules: [`NarrowInterval`], [`ClampMoment`] and
/// [`ArcMoment`]. Inert for a kernel with no integral in it — each matches
/// an interval fold and nothing else.
#[must_use]
pub fn integral_rules() -> Vec<Box<dyn Rewrite>> {
    alloc::vec![
        Box::new(NarrowInterval) as Box<dyn Rewrite>,
        Box::new(ClampMoment),
        Box::new(ArcMoment),
    ]
}

/// Whether `rule` belongs to the family the graph runs to a fixpoint before
/// anything else when it holds an integral: the three rules here,
/// [`FactorFold`] — `∫ c·f = c·∫ f` — and `ConstantFold`, which tidies the
/// literal arithmetic a closed form leaves.
pub(crate) fn closes_integrals(rule: RuleId) -> bool {
    [
        RuleId::of(&FactorFold),
        RuleId::of(&NarrowInterval),
        RuleId::of(&ClampMoment),
        RuleId::of(&ArcMoment),
        RuleId::of(&crate::math::algebra::ConstantFold),
    ]
    .contains(&rule)
}

/// A monotone arc's root, as [`ArcMoment`] reads it off a class `T`.
#[derive(Clone)]
struct Root {
    /// `D₀` in `D = u + D₀`.
    height: Rc<Term>,
    /// The `y` rise's `[step, bend]`, `[b, a]`.
    rise: [Rc<Term>; 2],
    /// The floor under the root's denominator.
    floor: RootFloor,
}

/// An integrand [`ArcMoment`] closes: the root the band is of, the clamp's
/// band, and what the clamp reads — `offset + x(T)`, `x` the rise
/// `[β, α]` (already scaled by the argument's slope).
struct ArcIntegrand {
    root: Root,
    offset: Rc<Term>,
    x: [Rc<Term>; 2],
    band: Band,
}

/// Which edge of a band an indicator is: `0 ≤ T` or `T < 1`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Edge {
    Start,
    End,
}

/// The factors the variable reaches, read as `[0 ≤ T]·[T < 1]·clamp(…)`:
/// which factor is the clamp is tried every way round, since a product's
/// order is the author's (or the algebra's).
fn arc_integrand(egraph: &EGraph, binder: Binder, factors: &[EClassId]) -> Option<ArcIntegrand> {
    let &[f0, f1, f2] = factors else {
        return None;
    };
    [[f0, f1, f2], [f1, f2, f0], [f2, f0, f1]]
        .into_iter()
        .find_map(|[first, second, clamp_factor]| {
            let parameter = band_of(egraph, first, second)?;
            let root = monotone_root(egraph, binder, parameter)?;
            clamps(egraph, clamp_factor)
                .into_iter()
                .find_map(|(argument, band)| {
                    let (offset, x) = arc_reading(egraph, binder, argument, parameter)?;
                    Some(ArcIntegrand {
                        root: root.clone(),
                        offset,
                        x,
                        band,
                    })
                })
        })
}

/// The class `T` two indicators are the band `[0 ≤ T < 1]` of — one each
/// edge, in either order — or `None`.
fn band_of(egraph: &EGraph, a: EClassId, b: EClassId) -> Option<EClassId> {
    let (a, b) = (edges(egraph, a), edges(egraph, b));
    a.iter().find_map(|&(edge, t)| {
        b.iter()
            .any(|&(other, u)| other != edge && u == t)
            .then_some(t)
    })
}

/// Which band edges `factor` is an indicator of: `(Start, T)` for
/// `Select(m, 1, 0)` with `m` one of `0 ≤ T`, `0 < T`, `T ≥ 0`, `T > 0`, and
/// `(End, T)` for `T < 1`, `T ≤ 1`, `1 > T`, `1 ≥ T` — the literal read off
/// the class constant fact, `T` canonical.
fn edges(egraph: &EGraph, factor: EClassId) -> Vec<(Edge, EClassId)> {
    let is = |class, value: f32| egraph.constant(class) == Some(value);
    let mut found = Vec::new();
    for node in egraph.nodes(factor) {
        let Some([mask, one, zero]) = operands(node, OpKind::Select) else {
            continue;
        };
        if !is(one, 1.0) || !is(zero, 0.0) {
            continue;
        }
        for test in egraph.nodes(mask) {
            let ENode::Op { op, children } = test else {
                continue;
            };
            let &[p, q] = children.as_slice() else {
                continue;
            };
            // `p ⋈ q` read as `low < high`: which side is the literal.
            let (low, high) = match op.kind() {
                OpKind::Lt | OpKind::Le => (p, q),
                OpKind::Gt | OpKind::Ge => (q, p),
                _ => continue,
            };
            if is(low, 0.0) {
                found.push((Edge::Start, egraph.find(high)));
            }
            if is(high, 1.0) {
                found.push((Edge::End, egraph.find(low)));
            }
        }
    }
    found
}

/// `T` as a monotone root: `D / max(b + √max(b·b + a·D, 0), k)` with
/// `D = u + D₀`, the rise `[b, a]` certified and the variable reaching
/// neither, and `k` a [`RootFloor`]. See [`ArcMoment`].
fn monotone_root(egraph: &EGraph, binder: Binder, parameter: EClassId) -> Option<Root> {
    binary_in(egraph, parameter, OpKind::Div).find_map(|[delta, denominator]| {
        let delta = egraph.find(delta);
        either_order(egraph, denominator, OpKind::Max).find_map(|[sum, floor]| {
            let floor = RootFloor::new(egraph.constant(floor)?)?;
            either_order(egraph, sum, OpKind::Add).find_map(|[step, root]| {
                let step = egraph.find(step);
                unary_in(egraph, root, OpKind::Sqrt).find_map(|radicand| {
                    let bend = floored_radicand(egraph, radicand, step, delta)?;
                    let rise = certified_rise(egraph, binder, step, bend)?;
                    let height = Recognizer::new(egraph, binder).affine(delta)?;
                    (literal_of(&height.slope) == Some(1.0)).then(|| Root {
                        height: height.offset,
                        rise,
                        floor,
                    })
                })
            })
        })
    })
}

/// The bend `a` of `max(b·b + a·D, 0)` in `class`, operands in either order,
/// given `b` and `D`.
fn floored_radicand(
    egraph: &EGraph,
    class: EClassId,
    step: EClassId,
    delta: EClassId,
) -> Option<EClassId> {
    either_order(egraph, class, OpKind::Max).find_map(|[radicand, floor]| {
        if egraph.constant(floor) != Some(RADICAND_FLOOR) {
            return None;
        }
        either_order(egraph, radicand, OpKind::Add).find_map(|[square, reach]| {
            let squares = binary_in(egraph, square, OpKind::Mul)
                .any(|[p, q]| egraph.find(p) == step && egraph.find(q) == step);
            if !squares {
                return None;
            }
            either_order(egraph, reach, OpKind::Mul)
                .find_map(|[bend, d]| (egraph.find(d) == delta).then(|| egraph.find(bend)))
        })
    })
}

/// `R`'s rise `[β, α]` when `R` holds `T·(β + β + α·T)`, operands of `·`
/// and the outer `+` in either order, the rise certified.
fn monotone_quadratic(
    egraph: &EGraph,
    binder: Binder,
    rise: EClassId,
    parameter: EClassId,
) -> Option<[Rc<Term>; 2]> {
    let is_parameter = |class| egraph.find(class) == parameter;
    either_order(egraph, rise, OpKind::Mul).find_map(|[t, slope]| {
        if !is_parameter(t) {
            return None;
        }
        either_order(egraph, slope, OpKind::Add).find_map(|[twice, bent]| {
            let step = binary_in(egraph, twice, OpKind::Add).find_map(|[p, q]| {
                let p = egraph.find(p);
                (p == egraph.find(q)).then_some(p)
            })?;
            either_order(egraph, bent, OpKind::Mul).find_map(|[bend, t]| {
                if !is_parameter(t) {
                    return None;
                }
                certified_rise(egraph, binder, step, egraph.find(bend))
            })
        })
    })
}

/// What a clamp's argument reads off the arc: `offset + x(T)` for an
/// argument `s·R + c` with `R = x(T)` a [`monotone_quadratic`] of the root
/// and `s` a positive literal, returned as `(c, [s·β, s·α])`.
fn arc_reading(
    egraph: &EGraph,
    binder: Binder,
    argument: EClassId,
    parameter: EClassId,
) -> Option<(Rc<Term>, [Rc<Term>; 2])> {
    spelled_from(egraph, binder, argument)
        .into_iter()
        .find_map(|rise| {
            let [step, bend] = monotone_quadratic(egraph, binder, rise, parameter)?;
            let form = Recognizer::in_terms_of(egraph, binder, rise).affine(argument)?;
            let scale = literal_of(&form.slope).filter(|s| *s > 0.0 && s.is_finite())?;
            let scale = Rc::new(Term::Literal(scale));
            let x = [product(Rc::clone(&scale), step), product(scale, bend)];
            Some((form.offset, x))
        })
}

/// The classes the variable reaches that `class` is spelled from through
/// `+`, `−`, negation, `·` and fused multiply-add — the candidates for what
/// an affine form of it is in terms of — `class` first.
fn spelled_from(egraph: &EGraph, binder: Binder, class: EClassId) -> Vec<EClassId> {
    let varies = |class| egraph.variance(class).depends_on(binder.var());
    let mut seen = BTreeSet::new();
    let mut order = Vec::new();
    let mut stack = alloc::vec![egraph.find(class)];
    while let Some(class) = stack.pop() {
        if !varies(class) || !seen.insert(class) {
            continue;
        }
        order.push(class);
        for node in egraph.nodes(class) {
            let ENode::Op { op, children } = node else {
                continue;
            };
            if matches!(
                op.kind(),
                OpKind::Add | OpKind::Sub | OpKind::Neg | OpKind::Mul | OpKind::MulAdd
            ) {
                stack.extend(children.iter().map(|&child| egraph.find(child)));
            }
        }
    }
    order
}

/// `[step, bend]` as terms, when they are a certified rise the variable does
/// not reach: `step` [`certified`], and `bend` holding `e − step` with `e`
/// certified, so both control-polygon steps are.
///
/// A bend whose two steps are both literals is their difference, as a
/// literal — what `ConstantFold` makes of the class a round later, read now:
/// the rule fires in the same round the steps' certificates fold, and
/// `monotone_root` drops `bend·δ` for a literal zero bend, so a line with
/// constant columns has an all-literal denominator whose reciprocal folds
/// in the closing phase.
fn certified_rise(
    egraph: &EGraph,
    binder: Binder,
    step: EClassId,
    bend: EClassId,
) -> Option<[Rc<Term>; 2]> {
    let varies = |class| egraph.variance(class).depends_on(binder.var());
    if varies(step) || varies(bend) || !certified(egraph, step) {
        return None;
    }
    let second = binary_in(egraph, bend, OpKind::Sub).find_map(|[second, first]| {
        (egraph.find(first) == step && certified(egraph, second)).then_some(second)
    })?;
    let literal_bend = egraph
        .constant(second)
        .zip(egraph.constant(step))
        .and_then(|(second, first)| folded(second - first));
    Some([
        value(egraph, step),
        literal_bend.unwrap_or_else(|| value(egraph, bend)),
    ])
}

/// Whether `class` is certified at least [`STEP_FLOOR`]: it is a literal
/// that is, or holds `max(z, k)` (either order) with such a literal `k`.
/// Any node will do — every node of a class denotes the same value.
fn certified(egraph: &EGraph, class: EClassId) -> bool {
    let at_least = |class| egraph.constant(class).is_some_and(|v| v >= STEP_FLOOR);
    at_least(class)
        || binary_in(egraph, class, OpKind::Max).any(|[p, q]| at_least(p) || at_least(q))
}

/// A class as a term: its literal when it is a finite constant, else the
/// class.
fn value(egraph: &EGraph, class: EClassId) -> Rc<Term> {
    egraph
        .constant(class)
        .and_then(folded)
        .unwrap_or_else(|| Rc::new(Term::Class(egraph.find(class))))
}

/// `node`'s operands, when it is a `kind` with `N` of them.
fn operands<const N: usize>(node: &ENode, kind: OpKind) -> Option<[EClassId; N]> {
    let ENode::Op { op, children } = node else {
        return None;
    };
    if op.kind() != kind {
        return None;
    }
    children.as_slice().try_into().ok()
}

/// The operand of every unary `kind` node in `class`.
fn unary_in(egraph: &EGraph, class: EClassId, kind: OpKind) -> impl Iterator<Item = EClassId> + '_ {
    egraph
        .nodes(class)
        .iter()
        .filter_map(move |node| operands::<1>(node, kind).map(|[x]| x))
}

/// The operands of every binary `kind` node in `class`, as spelled.
fn binary_in(
    egraph: &EGraph,
    class: EClassId,
    kind: OpKind,
) -> impl Iterator<Item = [EClassId; 2]> + '_ {
    egraph
        .nodes(class)
        .iter()
        .filter_map(move |node| operands::<2>(node, kind))
}

/// [`binary_in`] for an operator that commutes: each node's operands in
/// both orders.
fn either_order(
    egraph: &EGraph,
    class: EClassId,
    kind: OpKind,
) -> impl Iterator<Item = [EClassId; 2]> + '_ {
    binary_in(egraph, class, kind).flat_map(|[p, q]| [[p, q], [q, p]])
}

/// A term a rule will write: a literal, a class the graph holds, or
/// arithmetic over them. Built by the recognizer, emitted into a template.
#[derive(Debug)]
enum Term {
    Literal(f32),
    Class(EClassId),
    Sum(Rc<Term>, Rc<Term>),
    Difference(Rc<Term>, Rc<Term>),
    Product(Rc<Term>, Rc<Term>),
    Quotient(Rc<Term>, Rc<Term>),
    Negation(Rc<Term>),
}

/// A literal term, when `value` is a finite number — the one a constant
/// fold may produce. An overflow is kept symbolic rather than baked in, as
/// `ConstantFold` does.
fn folded(value: f32) -> Option<Rc<Term>> {
    value.is_finite().then(|| Rc::new(Term::Literal(value)))
}

fn literal_of(term: &Term) -> Option<f32> {
    match term {
        Term::Literal(value) => Some(*value),
        _ => None,
    }
}

/// `a + b`, with `0 + t = t` and literal pairs folded.
fn sum(a: Rc<Term>, b: Rc<Term>) -> Rc<Term> {
    match (literal_of(&a), literal_of(&b)) {
        (Some(x), _) if x == 0.0 => b,
        (_, Some(y)) if y == 0.0 => a,
        (Some(x), Some(y)) => folded(x + y).unwrap_or_else(|| Rc::new(Term::Sum(a, b))),
        _ => Rc::new(Term::Sum(a, b)),
    }
}

/// `a − b`, with `t − 0 = t`, `0 − t = −t` and literal pairs folded.
fn difference(a: Rc<Term>, b: Rc<Term>) -> Rc<Term> {
    match (literal_of(&a), literal_of(&b)) {
        (_, Some(y)) if y == 0.0 => a,
        (Some(x), _) if x == 0.0 => negation(b),
        (Some(x), Some(y)) => folded(x - y).unwrap_or_else(|| Rc::new(Term::Difference(a, b))),
        _ => Rc::new(Term::Difference(a, b)),
    }
}

/// `a·b`, with `1·t = t`, `0·t = 0` (exact for a finite `t`, which every
/// slope and offset of an integrand is — `Annihilator`'s own license) and
/// literal pairs folded.
fn product(a: Rc<Term>, b: Rc<Term>) -> Rc<Term> {
    match (literal_of(&a), literal_of(&b)) {
        (Some(x), _) if x == 1.0 => b,
        (_, Some(y)) if y == 1.0 => a,
        (Some(x), _) if x == 0.0 => a,
        (_, Some(y)) if y == 0.0 => b,
        (Some(x), Some(y)) => folded(x * y).unwrap_or_else(|| Rc::new(Term::Product(a, b))),
        _ => Rc::new(Term::Product(a, b)),
    }
}

/// `t/c`, `c` a nonzero literal: `t` times the literal `1/c`, rounded once
/// here — or `t` itself for `c = 1`.
///
/// A product and not a quotient, because a `Div` by a literal is what the
/// e-graph's `MulRecip` canonicalization turns into `t·recip(c)`, and a
/// `recip` shared by every copy of a substituted root is cheaper under the
/// DAG objective than one exact `Div` each — measured: the chord spelled
/// `[2·x_p(y) > 2·x]` extracted `recip(−2)`, a 12–14-bit *estimate*
/// (CLAUDE.md, "Floating point at the edges"), and lost 2⁻¹⁴ of its area.
/// The literal reciprocal costs one rounding, never an estimate's error.
/// Only a divisor whose reciprocal overflows keeps the quotient.
fn quotient(t: Rc<Term>, c: f32) -> Rc<Term> {
    match folded(1.0 / c) {
        Some(reciprocal) => product(t, reciprocal),
        None => Rc::new(Term::Quotient(t, Rc::new(Term::Literal(c)))),
    }
}

/// `−t`, with literals and double negations folded.
fn negation(t: Rc<Term>) -> Rc<Term> {
    if let Some(x) = literal_of(&t) {
        return Rc::new(Term::Literal(-x));
    }
    if let Term::Negation(inner) = &*t {
        return Rc::clone(inner);
    }
    Rc::new(Term::Negation(t))
}

/// `slope·u + offset`, `u` free in neither: the recognizer's answer for a
/// class affine in the integration variable.
#[derive(Clone, Debug)]
struct Affine {
    slope: Rc<Term>,
    offset: Rc<Term>,
}

impl Affine {
    /// The variable itself: `1·u + 0`.
    fn variable() -> Self {
        Self {
            slope: Rc::new(Term::Literal(1.0)),
            offset: Rc::new(Term::Literal(0.0)),
        }
    }

    /// A value the variable does not reach: `0·u + t`.
    fn invariant(t: Rc<Term>) -> Self {
        Self {
            slope: Rc::new(Term::Literal(0.0)),
            offset: t,
        }
    }

    fn plus(&self, other: &Self) -> Self {
        Self {
            slope: sum(Rc::clone(&self.slope), Rc::clone(&other.slope)),
            offset: sum(Rc::clone(&self.offset), Rc::clone(&other.offset)),
        }
    }

    fn minus(&self, other: &Self) -> Self {
        Self {
            slope: difference(Rc::clone(&self.slope), Rc::clone(&other.slope)),
            offset: difference(Rc::clone(&self.offset), Rc::clone(&other.offset)),
        }
    }

    fn negated(&self) -> Self {
        Self {
            slope: negation(Rc::clone(&self.slope)),
            offset: negation(Rc::clone(&self.offset)),
        }
    }

    /// `factor·(slope·u + offset)`, `factor` free of `u`.
    fn times(&self, factor: &Rc<Term>) -> Self {
        Self {
            slope: product(Rc::clone(factor), Rc::clone(&self.slope)),
            offset: product(Rc::clone(factor), Rc::clone(&self.offset)),
        }
    }
}

/// A bound an indicator factor puts on the variable: its root, and which
/// side of it the factor keeps.
enum Bound {
    Lower(Rc<Term>),
    Upper(Rc<Term>),
}

/// What a [`Recognizer`] reads classes as functions of.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Variable {
    /// The integration variable: its binder's leaf.
    Binder,
    /// A class the binder reaches, standing for the variable — a monotone
    /// arc's coordinate, which a clamp's argument is affine in although the
    /// argument is no affine function of the binder (see [`ArcMoment`]).
    /// The binder's own leaf is then just another class that varies.
    Class(EClassId),
}

/// Reads classes as functions of one variable — affine forms, and the
/// bounds indicators put on it — memoised per class, every node of a class
/// tried. The variable is the integration variable, or a class that
/// depends on it ([`Variable`]); either way a term is admitted into an
/// affine form's slope or offset only when the class variance fact clears
/// it of the integration variable.
struct Recognizer<'g> {
    egraph: &'g EGraph,
    binder: Binder,
    variable: Variable,
    /// Each class's affine form, once asked — `None` when it has none.
    memo: BTreeMap<EClassId, Option<Affine>>,
    /// Classes whose form is being computed: re-entering one is a merged
    /// class reaching itself, and that path has no form.
    open: BTreeSet<EClassId>,
    /// The variable's own leaf class, once met.
    leaf: Option<EClassId>,
}

impl<'g> Recognizer<'g> {
    fn new(egraph: &'g EGraph, binder: Binder) -> Self {
        Self::reading(egraph, binder, Variable::Binder)
    }

    /// A recognizer whose variable is `class`, standing for a function of
    /// `binder`'s variable.
    fn in_terms_of(egraph: &'g EGraph, binder: Binder, class: EClassId) -> Self {
        Self::reading(egraph, binder, Variable::Class(egraph.find(class)))
    }

    fn reading(egraph: &'g EGraph, binder: Binder, variable: Variable) -> Self {
        Self {
            egraph,
            binder,
            variable,
            memo: BTreeMap::new(),
            open: BTreeSet::new(),
            leaf: None,
        }
    }

    fn varies(&self, class: EClassId) -> bool {
        self.egraph.variance(class).depends_on(self.binder.var())
    }

    /// A class the variable does not reach, as a term: its literal when the
    /// class is a finite constant (a mask's all-ones pattern is not), else
    /// the class.
    fn value(&self, class: EClassId) -> Rc<Term> {
        value(self.egraph, class)
    }

    /// `class` as `slope·v + offset`, `v` the variable, or `None` when no
    /// node of it is affine in the variable.
    fn affine(&mut self, class: EClassId) -> Option<Affine> {
        let class = self.egraph.find(class);
        if self.variable == Variable::Class(class) {
            self.leaf = Some(class);
            return Some(Affine::variable());
        }
        if !self.varies(class) {
            return Some(Affine::invariant(self.value(class)));
        }
        if let Some(known) = self.memo.get(&class) {
            return known.clone();
        }
        if !self.open.insert(class) {
            return None;
        }
        let egraph = self.egraph;
        let found = egraph
            .nodes(class)
            .iter()
            .find_map(|node| self.spelled(class, node));
        self.open.remove(&class);
        self.memo.insert(class, found.clone());
        found
    }

    /// One node of a class as an affine form: the variable, a sum or
    /// difference of affine forms, a negation of one, or a product (or
    /// fused multiply-add) with one operand the variable does not reach.
    fn spelled(&mut self, class: EClassId, node: &ENode) -> Option<Affine> {
        match node {
            ENode::Var(v) if self.variable == Variable::Binder && *v == self.binder.var() => {
                self.leaf = Some(class);
                Some(Affine::variable())
            }
            ENode::Op { op, children } => match (op.kind(), children.as_slice()) {
                (OpKind::Add, &[a, b]) => Some(self.affine(a)?.plus(&self.affine(b)?)),
                (OpKind::Sub, &[a, b]) => Some(self.affine(a)?.minus(&self.affine(b)?)),
                (OpKind::Neg, &[a]) => Some(self.affine(a)?.negated()),
                (OpKind::Mul, &[a, b]) => self.scaled(a, b).or_else(|| self.scaled(b, a)),
                (OpKind::MulAdd, &[a, b, c]) => {
                    let product = self.scaled(a, b).or_else(|| self.scaled(b, a))?;
                    Some(product.plus(&self.affine(c)?))
                }
                _ => None,
            },
            _ => None,
        }
    }

    /// `factor·rest` when the variable does not reach `factor`.
    fn scaled(&mut self, factor: EClassId, rest: EClassId) -> Option<Affine> {
        if self.varies(factor) {
            return None;
        }
        let factor = self.value(factor);
        Some(self.affine(rest)?.times(&factor))
    }

    /// The bound `factor` puts on the variable, when it is an indicator
    /// `Select(A ⋈ B, 1, 0)` of a comparison whose difference has a literal
    /// nonzero slope. See [`NarrowInterval`] for which side.
    fn bound(&mut self, factor: EClassId) -> Option<Bound> {
        let egraph = self.egraph;
        egraph.nodes(factor).iter().find_map(|node| {
            let ENode::Op { op, children } = node else {
                return None;
            };
            let &[mask, one, zero] = children.as_slice() else {
                return None;
            };
            if op.kind() != OpKind::Select
                || egraph.constant(one) != Some(1.0)
                || egraph.constant(zero) != Some(0.0)
            {
                return None;
            }
            egraph
                .nodes(mask)
                .iter()
                .find_map(|test| self.comparison(test))
        })
    }

    /// `A ⋈ B` as a bound on the variable. `A − B = c·u + d`, so the
    /// comparison holds on one side of `t = (d_B − d_A)/c`: below it when
    /// `A − B` must be negative and `c > 0`, or positive and `c < 0`.
    fn comparison(&mut self, test: &ENode) -> Option<Bound> {
        let ENode::Op { op, children } = test else {
            return None;
        };
        let &[a, b] = children.as_slice() else {
            return None;
        };
        let negative = match op.kind() {
            OpKind::Lt | OpKind::Le => true,
            OpKind::Gt | OpKind::Ge => false,
            _ => return None,
        };
        let (a, b) = (self.affine(a)?, self.affine(b)?);
        let c = literal_of(&a.minus(&b).slope)?;
        if c == 0.0 || !c.is_finite() {
            return None;
        }
        let root = if c == -1.0 {
            difference(a.offset, b.offset)
        } else {
            quotient(difference(b.offset, a.offset), c)
        };
        Some(if negative == (c > 0.0) {
            Bound::Upper(root)
        } else {
            Bound::Lower(root)
        })
    }

    /// Split a product's factors into the bounds its indicators put on the
    /// variable, the factors it does not reach, and the rest, or `None` when
    /// no factor bounds it.
    fn bounds_of(&mut self, factors: Factors) -> Option<Split> {
        let mut bounds = Vec::new();
        let mut rest = Vec::new();
        for factor in factors.variant {
            match self.bound(factor) {
                Some(bound) => bounds.push(bound),
                None => rest.push(factor),
            }
        }
        (!bounds.is_empty()).then_some(Split {
            bounds,
            outside: factors.invariant,
            rest,
        })
    }
}

/// An integrand's product, as [`NarrowInterval`] reads it.
struct Split {
    /// What its indicators say about the variable.
    bounds: Vec<Bound>,
    /// The factors the variable does not reach, which stay outside.
    outside: Vec<EClassId>,
    /// Every other factor, read at the reparametrized point.
    rest: Vec<EClassId>,
}

/// The clamps a class holds: `(argument, band)` for each node that is
/// `min(max(z, P), Q)` or `max(min(z, Q), P)`, operands in either order,
/// `P < Q` finite literals.
fn clamps(egraph: &EGraph, class: EClassId) -> Vec<(EClassId, Band)> {
    let literal_and_other = |[x, y]: [EClassId; 2]| {
        let finite = |class| egraph.constant(class).filter(|v| v.is_finite());
        finite(x)
            .map(|v| (v, y))
            .or_else(|| finite(y).map(|v| (v, x)))
    };
    let pair = |node: &ENode, kind: OpKind| match node {
        ENode::Op { op, children } if op.kind() == kind => {
            let &[x, y] = children.as_slice() else {
                return None;
            };
            Some([x, y])
        }
        _ => None,
    };
    let mut found = Vec::new();
    for node in egraph.nodes(class) {
        for (outer, inner) in [(OpKind::Min, OpKind::Max), (OpKind::Max, OpKind::Min)] {
            let Some((outer_bound, inside)) = pair(node, outer).and_then(literal_and_other) else {
                continue;
            };
            for inner_node in egraph.nodes(inside) {
                let Some((inner_bound, argument)) =
                    pair(inner_node, inner).and_then(literal_and_other)
                else {
                    continue;
                };
                let (lower, upper) = match outer {
                    OpKind::Min => (inner_bound, outer_bound),
                    _ => (outer_bound, inner_bound),
                };
                if let Some(band) = Band::new(lower, upper) {
                    found.push((argument, band));
                }
            }
        }
    }
    found
}

/// A rule's closed form under construction: an arena whose `Var(i)` leaves
/// are `operands[i]`, the classes the rule recognized.
#[derive(Default)]
struct Template {
    arena: ExprArena,
    operands: Vec<EClassId>,
    /// Terms already emitted, by address: a recognizer's terms share
    /// subterms, and emitting one twice would walk it twice.
    emitted: BTreeMap<*const Term, ExprId>,
}

impl Template {
    /// The leaf standing for `class`, or `None` past the 256 a `Var` can
    /// name.
    fn operand(&mut self, class: EClassId) -> Option<ExprId> {
        let index = match self.operands.iter().position(|&known| known == class) {
            Some(index) => index,
            None => {
                self.operands.push(class);
                self.operands.len() - 1
            }
        };
        Some(self.arena.push_var(u8::try_from(index).ok()?))
    }

    /// Write `term` into the arena.
    fn emit(&mut self, term: &Rc<Term>) -> Option<ExprId> {
        if let Some(&id) = self.emitted.get(&Rc::as_ptr(term)) {
            return Some(id);
        }
        let id = match &**term {
            Term::Literal(value) => self.arena.push_const(*value),
            Term::Class(class) => self.operand(*class)?,
            Term::Sum(a, b) => self.binary(OpKind::Add, a, b)?,
            Term::Difference(a, b) => self.binary(OpKind::Sub, a, b)?,
            Term::Product(a, b) => self.binary(OpKind::Mul, a, b)?,
            Term::Quotient(a, b) => self.binary(OpKind::Div, a, b)?,
            Term::Negation(a) => {
                let a = self.emit(a)?;
                self.arena.push_unary(OpKind::Neg, a)
            }
        };
        self.emitted.insert(Rc::as_ptr(term), id);
        Some(id)
    }

    fn binary(&mut self, op: OpKind, a: &Rc<Term>, b: &Rc<Term>) -> Option<ExprId> {
        let (a, b) = (self.emit(a)?, self.emit(b)?);
        Some(self.arena.push_binary(op, a, b))
    }

    /// The cut the bounds make: the largest lower root and the smallest
    /// upper one.
    fn cut(&mut self, bounds: &[Bound]) -> Option<Cut> {
        let mut lower: Option<ExprId> = None;
        let mut upper: Option<ExprId> = None;
        for bound in bounds {
            let (side, root, combine) = match bound {
                Bound::Lower(root) => (&mut lower, root, OpKind::Max),
                Bound::Upper(root) => (&mut upper, root, OpKind::Min),
            };
            let root = self.emit(root)?;
            *side = Some(match *side {
                Some(so_far) => self.arena.push_binary(combine, so_far, root),
                None => root,
            });
        }
        Some(Cut { lower, upper })
    }

    /// Copy `roots` into `plan`, each `Var(i)` leaf the class
    /// `operands[i]`.
    fn splice_into<const N: usize>(
        &self,
        plan: &mut PlanBuilder,
        roots: [ExprId; N],
    ) -> Option<[HeadRef; N]> {
        plan.splice(&self.arena, &roots, &self.operands)?
            .try_into()
            .ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::unclosed_integrals;
    use pixelflow_ir::integral::ROOT_FLOOR;
    use pixelflow_ir::{Kernel, LatticeShape, Monoid};

    /// The lattice every closure pin extracts at.
    const SHAPE: LatticeShape = LatticeShape::new([8, 8]);

    fn op1(op: &'static dyn ops::Op, a: EClassId) -> ENode {
        ENode::Op {
            op,
            children: alloc::vec![a],
        }
    }

    fn op2(op: &'static dyn ops::Op, a: EClassId, b: EClassId) -> ENode {
        ENode::Op {
            op,
            children: alloc::vec![a, b],
        }
    }

    fn slot(slot: u8) -> Binder {
        Binder::from_slot(slot).expect("a live slot")
    }

    /// `∫_lo^hi` over `slot`, through `Fold::from_bits`'s documented layout —
    /// the one door an interval other than the pixel has outside
    /// `pixelflow-ir`.
    fn interval(slot: u8, lo: f32, hi: f32) -> Fold {
        let bits = 1u128 << 112
            | u128::from(slot) << 64
            | u128::from(lo.to_bits()) << 32
            | u128::from(hi.to_bits());
        Fold::from_bits(bits).expect("a finite, nonempty interval")
    }

    /// Which side of which root `test` bounds the variable on, if any, with
    /// the root read back as a literal (every root below is one).
    fn side(eg: &EGraph, test: EClassId) -> Option<(&'static str, f32)> {
        let mut recognizer = Recognizer::new(eg, slot(0));
        let node = eg.nodes(test).first()?.clone();
        let (name, root) = match recognizer.comparison(&node)? {
            Bound::Lower(root) => ("lower", root),
            Bound::Upper(root) => ("upper", root),
        };
        Some((name, literal_of(&root).expect("a literal root")))
    }

    /// **The sign of the slope picks the side.** `A ⋈ B` bounds `u` above
    /// or below its root by the comparison *and* the sign of `A − B`'s
    /// slope in `u`; a negative slope flips the side, and `<` and `≤` bound
    /// the same side (their difference is a point).
    #[test]
    fn a_comparison_bounds_the_side_its_slope_says() {
        let mut eg = EGraph::with_rules(Vec::new());
        let u = eg.add(ENode::Var(slot(0).var()));
        let three = eg.add(ENode::constant(3.0));
        let two = eg.add(ENode::constant(2.0));
        let minus_two = eg.add(ENode::constant(-2.0));
        let two_u = eg.add(op2(&ops::Mul, two, u));
        let u_times_minus_two = eg.add(op2(&ops::Mul, u, minus_two));
        let cases = [
            // u < 3, u ≤ 3: above.
            (op2(&ops::Lt, u, three), Some(("upper", 3.0))),
            (op2(&ops::Le, u, three), Some(("upper", 3.0))),
            // u > 3, u ≥ 3: below.
            (op2(&ops::Gt, u, three), Some(("lower", 3.0))),
            (op2(&ops::Ge, u, three), Some(("lower", 3.0))),
            // 3 > u is u < 3: the slope of 3 − u is −1.
            (op2(&ops::Gt, three, u), Some(("upper", 3.0))),
            // 2u < 3 is u < 1.5; u·(−2) < 3 is u > −1.5.
            (op2(&ops::Lt, two_u, three), Some(("upper", 1.5))),
            (
                op2(&ops::Lt, u_times_minus_two, three),
                Some(("lower", -1.5)),
            ),
            // u = 3 is not a half-line.
            (op2(&ops::Eq, u, three), None),
        ];
        for (test, want) in cases {
            let test = eg.add(test);
            assert_eq!(side(&eg, test), want, "{:?}", eg.nodes(test));
        }
    }

    /// **Every spelling of the same affine form.** `u − 3`, `u + (−3)`,
    /// `(−3) + u`, `MulAdd(1, u, −3)` and `−(3 − u)` against `0` all bound
    /// `u` above `3`.
    #[test]
    fn every_spelling_of_an_affine_difference_is_read() {
        let mut eg = EGraph::with_rules(Vec::new());
        let u = eg.add(ENode::Var(slot(0).var()));
        let zero = eg.add(ENode::constant(0.0));
        let one = eg.add(ENode::constant(1.0));
        let three = eg.add(ENode::constant(3.0));
        let minus_three = eg.add(ENode::constant(-3.0));
        let spellings = [
            op2(&ops::Sub, u, three),
            op2(&ops::Add, u, minus_three),
            op2(&ops::Add, minus_three, u),
            ENode::Op {
                op: &ops::MulAdd,
                children: alloc::vec![one, u, minus_three],
            },
        ];
        let mut spelled: Vec<EClassId> = spellings.into_iter().map(|n| eg.add(n)).collect();
        let three_minus_u = eg.add(op2(&ops::Sub, three, u));
        spelled.push(eg.add(op1(&ops::Neg, three_minus_u)));
        for difference in spelled {
            let test = eg.add(op2(&ops::Lt, difference, zero));
            assert_eq!(
                side(&eg, test),
                Some(("upper", 3.0)),
                "{:?}",
                eg.nodes(difference)
            );
        }
    }

    /// **The class, not one spelling.** A class whose first node is not
    /// affine — `sin(u)`, merged by hand with `u − 3` as if a rule had
    /// proved them equal — is still read through the node that is.
    #[test]
    fn a_class_is_read_through_whichever_node_is_affine() {
        let mut eg = EGraph::with_rules(Vec::new());
        let u = eg.add(ENode::Var(slot(0).var()));
        let zero = eg.add(ENode::constant(0.0));
        let three = eg.add(ENode::constant(3.0));
        let opaque = eg.add(op1(&ops::Sin, u));
        let affine = eg.add(op2(&ops::Sub, u, three));
        let merged = eg.union(opaque, affine);
        eg.rebuild();
        let test = eg.add(op2(&ops::Lt, merged, zero));
        assert_eq!(side(&eg, test), Some(("upper", 3.0)));
    }

    /// **A slope that is a value declines.** `Y·u < 3` would need `3/Y`, and
    /// a side chosen by `Y`'s sign; the rule refuses rather than guesses.
    #[test]
    fn a_slope_that_is_not_a_literal_declines() {
        let mut eg = EGraph::with_rules(Vec::new());
        let u = eg.add(ENode::Var(slot(0).var()));
        let y = eg.add(ENode::Var(1));
        let three = eg.add(ENode::constant(3.0));
        let y_u = eg.add(op2(&ops::Mul, y, u));
        let test = eg.add(op2(&ops::Lt, y_u, three));
        assert_eq!(side(&eg, test), None);
    }

    /// `Select(test, 1, 0)`, as a class.
    fn indicator(eg: &mut EGraph, test: ENode) -> EClassId {
        let (one, zero) = (eg.add(ENode::constant(1.0)), eg.add(ENode::constant(0.0)));
        let test = eg.add(test);
        eg.add(ENode::Op {
            op: &ops::Select,
            children: alloc::vec![test, one, zero],
        })
    }

    /// **Only an interval is narrowed.** `<` and `≤` agree on an interval
    /// because they differ on a point; over a range the point is a term.
    #[test]
    fn narrowing_declines_a_range_fold() {
        let mut eg = EGraph::with_rules(Vec::new());
        let u = eg.add(ENode::Var(slot(0).var()));
        let three = eg.add(ENode::constant(3.0));
        let body = indicator(&mut eg, op2(&ops::Le, u, three));
        let range = ENode::Reduce {
            fold: Fold::new(Monoid::SUM, slot(0), 0..8),
            body,
        };
        let integral = ENode::Reduce {
            fold: interval(0, 0.0, 8.0),
            body,
        };
        let (range_class, integral_class) = (eg.add(range.clone()), eg.add(integral.clone()));
        assert!(NarrowInterval.apply(&eg, range_class, &range).is_none());
        assert!(
            NarrowInterval
                .apply(&eg, integral_class, &integral)
                .is_some()
        );
    }

    /// **A clamp is read in both nestings and both operand orders**, and
    /// only as a band: `min(max(z, 0), 1)`, `min(1, max(0, z))` and
    /// `max(min(z, 1), 0)` are one band; `min(max(z, 1), 0)` — `lo > hi`,
    /// the constant `0` — is none.
    #[test]
    fn a_clamp_is_read_in_every_nesting() {
        let mut eg = EGraph::with_rules(Vec::new());
        let z = eg.add(ENode::Var(slot(0).var()));
        let (zero, one) = (eg.add(ENode::constant(0.0)), eg.add(ENode::constant(1.0)));
        let floor = eg.add(op2(&ops::Max, z, zero));
        let floor_swapped = eg.add(op2(&ops::Max, zero, z));
        let ceiling = eg.add(op2(&ops::Min, z, one));
        let band = Band::new(0.0, 1.0);
        for clamp in [
            op2(&ops::Min, floor, one),
            op2(&ops::Min, one, floor_swapped),
            op2(&ops::Max, ceiling, zero),
        ] {
            let clamp = eg.add(clamp);
            let found: Vec<_> = clamps(&eg, clamp).into_iter().map(|(_, b)| b).collect();
            assert_eq!(
                found,
                alloc::vec![band.expect("a band")],
                "{:?}",
                eg.nodes(clamp)
            );
        }
        let floor_one = eg.add(op2(&ops::Max, z, one));
        let inverted = eg.add(op2(&ops::Min, floor_one, zero));
        assert!(clamps(&eg, inverted).is_empty());
    }

    /// Unclosed integrals in the runtime tier's extraction of `kernel`.
    fn unclosed(kernel: &Kernel) -> Option<usize> {
        let (arena, root) = kernel.parts();
        unclosed_integrals(arena, root, SHAPE)
    }

    /// `[mask]`: the one place a mask becomes a number.
    fn indicator_of(mask: &Kernel) -> Kernel {
        mask.select(&Kernel::constant(1.0), &Kernel::constant(0.0))
    }

    /// The chord term a glyph piece contributes, with slope `k`:
    /// `[y ≥ 2.25]·[y < 5.5]·[x < 3.5 + (y − 2.25)·k]`.
    fn chord(k: f32) -> Kernel {
        let (x, y, c) = (Kernel::x(), Kernel::y(), Kernel::constant);
        let crossing = y.sub(&c(2.25)).mul(&c(k)).add(&c(3.5));
        c(1.0)
            .mul(&indicator_of(&y.ge(&c(2.25))))
            .mul(&indicator_of(&y.lt(&c(5.5))))
            .mul(&indicator_of(&x.lt(&crossing)))
    }

    /// **(b) The chord closes.** `area(σ·[y₀ ≤ y < y₁]·[x < x_p(y)])` —
    /// factoring, narrowing twice and one clamp moment — leaves no integral
    /// for quadrature, for a slanted, a vertical and a near-horizontal edge.
    /// With literal parameters, so every constant is a class fact the
    /// recognizers fold; `pixelflow-core`'s `area_oracle` pins the same over
    /// uniforms, with each rule's own integrand and one no rule closes, and
    /// judges the values.
    ///
    /// A *literal* zero slope is in the list on purpose. The closing phase
    /// narrows the band while `(y − 2.25)·0` still varies in the variable
    /// (only the main phase's `Annihilator` makes it `0`), and the factors
    /// the variable does not reach — the literal `σ` among them — once went
    /// *inside* the narrowed integral, which left `h·∫ 1`: the constant
    /// rule's integrand, which nothing closes.
    /// `NarrowInterval` keeps them outside. (Before the zero-divisor fix in
    /// `mean_of_clamp`, this case "closed" because saturation had collapsed
    /// the whole area to the constant `0`; see
    /// `a_zero_sweep_never_makes_an_area_constant`, which is what judges
    /// that it is not closed that way now.)
    #[test]
    fn the_area_of_a_chord_closes() {
        for k in [0.4, -1.0e6, 0.0] {
            let area = chord(k).area();
            assert_eq!(unclosed(&area), Some(0), "k = {k}");
        }
    }

    /// **A provably zero sweep proves nothing false.** A literal slope of
    /// `0` — a vertical edge, or a crossing the variable does not reach —
    /// makes the clamp moment's sweep `z₁ − z₀` provably zero. Divided by
    /// that, the algebra's `x·recip(x) = 1` and `(x·a)/a = x` (sound for
    /// every `x` but zero) merged the quotient's class with whatever `x`
    /// they were handed, and saturation went on from there to a root class
    /// holding a constant: the chord's area extracted as
    /// `σ·½·max(1 − 1, 0) = 0` at every pixel, with no integral left — so
    /// the closure pin above passed. The area of each kernel here depends on
    /// `X` and on `Y`, so after the runtime tier's whole saturation its root
    /// class must still vary along both: a class merged with a constant
    /// varies along nothing.
    #[test]
    fn a_zero_sweep_never_makes_an_area_constant() {
        use crate::egraph::RuleSet;
        use crate::egraph::optimizer::Optimizer;
        use crate::egraph::{Vocabulary, insert};
        let vertical = |crossing: Kernel| {
            let (x, y, c) = (Kernel::x(), Kernel::y(), Kernel::constant);
            indicator_of(&y.ge(&c(2.25)))
                .mul(&indicator_of(&y.lt(&c(5.5))))
                .mul(&indicator_of(&x.lt(&crossing)))
                .area()
        };
        let kernels = [
            ("k = 0", chord(0.0).area()),
            ("[x < a]", vertical(Kernel::constant(3.5))),
            (
                "[x < a] over a uniform",
                vertical(pixelflow_ir::Uniform::new(3.5).kernel()),
            ),
        ];
        for (name, kernel) in kernels {
            let mut optimizer = Optimizer::production()
                .rules(RuleSet::runtime())
                .for_lattice(SHAPE);
            let mut eg = optimizer.egraph();
            let (arena, root) = kernel.parts();
            let root = insert(arena, root, &mut eg, Vocabulary::Runtime).expect("inserts");
            let stats = optimizer.saturate_term(&mut eg, arena.len());
            for axis in [0, 1] {
                assert!(
                    eg.variance(root).depends_on(axis),
                    "{name}: after saturation the area no longer varies along axis {axis} \
                     ({stats:?})"
                );
            }
        }
    }

    /// **The closing phase is inert without an integral.** Over the runtime
    /// rule set, a graph holding none runs no closing phase at all — so it
    /// saturates exactly as before the phase existed — and a graph holding
    /// one runs exactly the family: `FactorFold`, the three rules here and
    /// `ConstantFold`, in the rule set's order.
    #[test]
    fn the_closing_phase_runs_only_on_a_graph_with_an_integral() {
        use crate::egraph::{RuleSet, Vocabulary, insert};
        let set = RuleSet::runtime();
        let graph = |kernel: &Kernel| {
            let (rules, ids) = set.shared();
            let mut eg = EGraph::with_shared_rules(rules, ids);
            let (arena, root) = kernel.parts();
            insert(arena, root, &mut eg, Vocabulary::Runtime).expect("inserts");
            eg
        };
        let plain = Kernel::x().mul(&Kernel::y()).sin();
        assert!(graph(&plain).integral_closure_rules().is_empty());
        let summed = Kernel::sum_over(4, |i| Kernel::x().add(i));
        assert!(
            graph(&summed).integral_closure_rules().is_empty(),
            "a range fold is not an integral"
        );
        let family: Vec<RuleId> = graph(&plain.area())
            .integral_closure_rules()
            .into_iter()
            .map(|idx| set.id_of(idx).expect("an index into the set"))
            .collect();
        let mut expected = alloc::vec![
            RuleId::of(&crate::math::algebra::ConstantFold),
            RuleId::of(&FactorFold),
            RuleId::of(&NarrowInterval),
            RuleId::of(&ClampMoment),
            RuleId::of(&ArcMoment),
        ];
        expected.sort_by_key(|id| set.index_of(*id));
        assert_eq!(family, expected);
    }

    /// **The closing phase spends the run's rounds, not rounds of its own.**
    /// A caller that allows `n` rounds is told of at most `n` however they
    /// divide between the phases: `run_anytime_curve` subtracts each call's
    /// rounds from what it has left, and a run that reported more than it
    /// was allowed would underflow that. The chord takes three rounds of the
    /// family to close, so every allowance here ends inside or just after
    /// the closing phase.
    #[test]
    fn the_closing_phase_spends_the_runs_rounds() {
        use crate::egraph::{RuleSet, Vocabulary, insert};
        let area = chord(0.4).area();
        let (arena, root) = area.parts();
        for allowed in 1..=4 {
            let (rules, ids) = RuleSet::runtime().shared();
            let mut eg = EGraph::with_shared_rules(rules, ids);
            insert(arena, root, &mut eg, Vocabulary::Runtime).expect("inserts");
            assert!(!eg.integral_closure_rules().is_empty());
            let stats = eg.saturate_budgeted(allowed, 50_000, None);
            assert!(
                stats.iterations <= allowed,
                "allowed {allowed} rounds, ran {stats:?}"
            );
        }
    }

    /// The integration rules are the runtime tier's, and only the runtime
    /// tier's — the macro tier has no syntax that builds a fold — and they
    /// are the closing phase's.
    #[test]
    fn the_integration_rules_are_in_the_runtime_set_only() {
        use crate::egraph::RuleSet;
        for id in [
            RuleId::of(&NarrowInterval),
            RuleId::of(&ClampMoment),
            RuleId::of(&ArcMoment),
        ] {
            assert!(RuleSet::runtime().index_of(id).is_some());
            assert!(RuleSet::production().index_of(id).is_none());
            assert!(closes_integrals(id));
        }
    }

    /// A monotone arc's columns, as the author's integrand reads them: the
    /// start, the control polygon's two steps (`p₁ − p₀` and `p₂ − p₁`), the
    /// orientation `σ` and the reflection `S`.
    #[derive(Clone, Copy)]
    struct Piece {
        start: [f32; 2],
        first: [f32; 2],
        second: [f32; 2],
        sigma: f32,
        reflect: f32,
    }

    /// A curved piece: both coordinates rise, and bend.
    const CURVE: Piece = Piece {
        start: [1.25, 2.5],
        first: [3.0, 0.5],
        second: [0.75, 2.0],
        sigma: 1.0,
        reflect: 1.0,
    };

    /// How an arc's integrand is spelled: the author's, and the ways a
    /// rewrite or another author could write the same terms.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Spelling {
        /// The plan's §8 step 5, as written.
        Author,
        /// Every commutative operand the recognizers read swapped, and the
        /// band's comparisons spelled `T ≥ 0` and `1 > T`.
        Commuted,
        /// The first `y` step read raw, with no certificate.
        Uncertified,
    }

    /// The author's integrand for `piece`:
    /// `σ·area([0 ≤ T]·[T < 1]·[x < x₀ + x(T)]).at(X, S·Y)`, `T` the arc's
    /// parameter at height `y`, each value read through `column`, the root
    /// taken under `floor`.
    fn arc_term(
        piece: Piece,
        column: impl Fn(f32) -> Kernel,
        spelling: Spelling,
        floor: f32,
    ) -> Kernel {
        let (zero, one) = (Kernel::constant(0.0), Kernel::constant(1.0));
        let certify = |step: f32| match spelling {
            Spelling::Commuted => zero.max(&column(step)),
            _ => column(step).max(&zero),
        };
        let b = match spelling {
            Spelling::Uncertified => column(piece.first[1]),
            _ => certify(piece.first[1]),
        };
        let bx = certify(piece.first[0]);
        let a = certify(piece.second[1]).sub(&b);
        let ax = certify(piece.second[0]).sub(&bx);
        let d = Kernel::y().sub(&column(piece.start[1]));
        let (square, reach) = (b.mul(&b), a.mul(&d));
        let radicand = match spelling {
            Spelling::Commuted => reach.add(&square).max(&zero),
            _ => square.add(&reach).max(&zero),
        };
        let denominator = match spelling {
            Spelling::Commuted => radicand.sqrt().add(&b),
            _ => b.add(&radicand.sqrt()),
        };
        let t = d.div(&denominator.max(&Kernel::constant(floor)));
        let slope = bx.add(&bx).add(&ax.mul(&t));
        let rise = match spelling {
            Spelling::Commuted => slope.mul(&t),
            _ => t.mul(&slope),
        };
        let xt = column(piece.start[0]).add(&rise);
        let band = match spelling {
            Spelling::Commuted => indicator_of(&t.ge(&zero)).mul(&indicator_of(&one.gt(&t))),
            _ => indicator_of(&zero.le(&t)).mul(&indicator_of(&t.lt(&one))),
        };
        let chi = band.mul(&indicator_of(&Kernel::x().lt(&xt)));
        let screen_y = column(piece.reflect).mul(&Kernel::y());
        column(piece.sigma).mul(&chi.area().at(&Kernel::x(), &screen_y))
    }

    fn literal(v: f32) -> Kernel {
        Kernel::constant(v)
    }

    fn uniform(v: f32) -> Kernel {
        pixelflow_ir::Uniform::new(v).kernel()
    }

    /// **(c) An arc's area closes.** `FactorFold`, `NarrowInterval` and
    /// `ArcMoment` leave no integral for quadrature — for a curve, a line
    /// written the same way (`second = first`, so the bend is zero), a
    /// vertical and a horizontal piece and a reflected one, each over
    /// literal columns and over uniforms. One integrand and one rule for
    /// every piece is what lets a glyph be one fold with one body.
    #[test]
    fn the_area_left_of_an_arc_closes() {
        let line = Piece {
            second: CURVE.first,
            ..CURVE
        };
        let vertical = Piece {
            first: [0.0, 1.0],
            second: [0.0, 2.0],
            ..CURVE
        };
        let horizontal = Piece {
            first: [1.0, 0.0],
            second: [2.0, 0.0],
            ..CURVE
        };
        let reflected = Piece {
            start: [1.25, -2.5],
            sigma: -1.0,
            reflect: -1.0,
            ..CURVE
        };
        let pieces = [
            ("curve", CURVE),
            ("line", line),
            ("vertical", vertical),
            ("horizontal", horizontal),
            ("reflected", reflected),
        ];
        let columns: [(&str, fn(f32) -> Kernel); 2] = [("literal", literal), ("uniform", uniform)];
        for (piece_name, piece) in pieces {
            for (column_name, column) in columns {
                let area = arc_term(piece, column, Spelling::Author, ROOT_FLOOR);
                assert_eq!(
                    unclosed(&area),
                    Some(0),
                    "{piece_name} over {column_name} columns"
                );
            }
        }
    }

    /// **Every spelling the recognizers promise.** Operands of `max`, `+`
    /// and `·` swapped, and the band read through `T ≥ 0` and `1 > T`: the
    /// same integrand, closed the same way.
    #[test]
    fn a_commuted_arc_closes() {
        let area = arc_term(CURVE, uniform, Spelling::Commuted, ROOT_FLOOR);
        assert_eq!(unclosed(&area), Some(0));
    }

    /// **An uncertified step declines.** With the first `y` step read raw,
    /// `T` is the rise's inverse only for the values a host happened to
    /// write, which the rule cannot see: it leaves the outer integral for
    /// quadrature (the inner one still narrows to its clamp).
    #[test]
    fn an_arc_without_its_certificate_keeps_its_integral() {
        let area = arc_term(CURVE, uniform, Spelling::Uncertified, ROOT_FLOOR);
        assert_eq!(unclosed(&area), Some(1));
    }

    /// **The floor is pinned.** A floor above `2⁻¹⁰⁰` moves the root over a
    /// height the formula does not account for, a zero floor divides by
    /// zero where the rise is flat, and a subnormal one is zero under
    /// denormals-are-zero: all decline. A smaller normal floor is exact too,
    /// and closes.
    #[test]
    fn an_arc_under_the_wrong_floor_keeps_its_integral() {
        for floor in [2.0 * ROOT_FLOOR, 0.0, f32::MIN_POSITIVE / 1024.0] {
            let area = arc_term(CURVE, uniform, Spelling::Author, floor);
            assert_eq!(unclosed(&area), Some(1), "floor {floor:e}");
        }
        let area = arc_term(CURVE, uniform, Spelling::Author, ROOT_FLOOR / 4.0);
        assert_eq!(unclosed(&area), Some(0));
    }

    /// **A class can stand for the variable.** `(x₀ + R − X) − (−½)` —
    /// narrowing's spelling of a clamp's argument — is read in terms of
    /// `R`, a class the binder reaches through a sine, as
    /// `1·R + ((x₀ − X) − (−½))`, the difference of nearly equal
    /// coordinates first. The binder used bare beside `R` is not affine in
    /// `R`, and neither is `R·R`.
    #[test]
    fn a_class_can_stand_for_the_variable() {
        let mut eg = EGraph::with_rules(Vec::new());
        let u = eg.add(ENode::Var(slot(0).var()));
        let (x, x0) = (eg.add(ENode::Var(0)), eg.add(ENode::Var(1)));
        let r = eg.add(op1(&ops::Sin, u));
        let minus_half = eg.add(ENode::constant(-0.5));
        let xt = eg.add(op2(&ops::Add, x0, r));
        let reach = eg.add(op2(&ops::Sub, xt, x));
        let z = eg.add(op2(&ops::Sub, reach, minus_half));
        let read = |eg: &EGraph, class| Recognizer::in_terms_of(eg, slot(0), r).affine(class);

        let form = read(&eg, z).expect("affine in R");
        assert_eq!(literal_of(&form.slope), Some(1.0));
        let Term::Difference(coordinates, half) = &*form.offset else {
            panic!("offset {:?}", form.offset);
        };
        assert_eq!(literal_of(half), Some(-0.5));
        assert!(
            matches!(&**coordinates, Term::Difference(..)),
            "{coordinates:?}"
        );
        assert!(spelled_from(&eg, slot(0), z).contains(&eg.find(r)));

        let beside = eg.add(op2(&ops::Add, z, u));
        let square = eg.add(op2(&ops::Mul, r, r));
        assert!(read(&eg, beside).is_none());
        assert!(read(&eg, square).is_none());
    }

    /// **An arc closes in one round.** Within the closing phase's first
    /// round each rule sees the ones before it: `NarrowInterval` closes the
    /// inner integral to its clamp, and `ArcMoment`, after it in the rule
    /// set, closes the outer one — so after a single round every class
    /// holding an integral also holds a member that is not one.
    #[test]
    fn an_arc_closes_in_one_round() {
        use crate::egraph::{RuleSet, Vocabulary, insert};
        let (rules, ids) = RuleSet::runtime().shared();
        let mut eg = EGraph::with_shared_rules(rules, ids);
        let area = arc_term(CURVE, uniform, Spelling::Author, ROOT_FLOOR);
        let (arena, root) = area.parts();
        insert(arena, root, &mut eg, Vocabulary::Runtime).expect("inserts");
        let stats = eg.saturate_budgeted(1, 50_000, None);
        assert_eq!(stats.iterations, 1);
        let open: Vec<EClassId> = eg
            .canonical_class_ids()
            .into_iter()
            .filter(|&class| {
                let nodes = eg.nodes(class);
                nodes.iter().any(is_integral) && nodes.iter().all(is_integral)
            })
            .collect();
        assert!(open.is_empty(), "open after one round: {open:?}");
    }

    /// **A closed integral is closed.** Once `ArcMoment` has written its
    /// right-hand side beside the fold, it declines that fold: a spelling
    /// the algebra adds later would otherwise make it write another.
    #[test]
    fn the_arc_rule_declines_a_closed_integral() {
        use crate::egraph::{Vocabulary, insert};
        const CAP: usize = 50_000;
        const ARC_MOMENT: usize = 2;
        let mut eg = EGraph::with_rules(alloc::vec![
            Box::new(FactorFold) as Box<dyn Rewrite>,
            Box::new(NarrowInterval),
            Box::new(ArcMoment),
        ]);
        let area = arc_term(CURVE, uniform, Spelling::Author, ROOT_FLOOR);
        let (arena, root) = area.parts();
        insert(arena, root, &mut eg, Vocabulary::Runtime).expect("inserts");
        let integrals = |eg: &EGraph| -> Vec<(EClassId, ENode)> {
            eg.canonical_class_ids()
                .into_iter()
                .flat_map(|class| {
                    eg.nodes(class)
                        .iter()
                        .filter(|node| is_integral(node))
                        .map(move |node| (class, node.clone()))
                        .collect::<Vec<_>>()
                })
                .collect()
        };
        let fires = |eg: &EGraph| {
            integrals(eg)
                .iter()
                .filter(|(class, node)| ArcMoment.apply(eg, *class, node).is_some())
                .count()
        };
        assert_eq!(fires(&eg), 0, "the inner integral is not narrowed yet");
        // Factoring and narrowing, to a fixpoint: the inner integral closes
        // to its clamp, which is what the arc rule reads.
        for _ in 0..8 {
            let changes =
                eg.apply_rule_at_index(0, CAP).changes + eg.apply_rule_at_index(1, CAP).changes;
            eg.rebuild();
            if changes == 0 {
                break;
            }
        }
        assert_eq!(fires(&eg), 1, "the outer integral, before it closes");
        eg.apply_rule_at_index(ARC_MOMENT, CAP);
        eg.rebuild();
        assert_eq!(fires(&eg), 0, "the outer integral, closed");
    }
}
