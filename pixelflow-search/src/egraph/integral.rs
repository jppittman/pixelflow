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
//! ```
//!
//! and, from `fold_rules`, the one rule both domains share:
//! `∫ c·f = c·∫ f` (`FactorFold`). Nothing else is here yet: the constant,
//! linearity, select, interchange and power-moment rules of the plan's table
//! close no integral the chord needs, so each waits for the kernel that
//! does (CLAUDE.md, "subtract before you add").
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
//! integration family — [`FactorFold`](super::FactorFold), these two, and
//! `ConstantFold` — to a fixpoint on the freshly inserted graph before the
//! full rule set sees it, counted against the same application budget
//! (`EGraph::saturate_bounded`). The derivation takes three rounds of that
//! family and nothing else; interleaved with the algebra, the class cap a
//! glyph already reaches in its first round would stop it half-closed. A
//! graph with no integral skips the phase, so no other kernel's saturation
//! changes.

use alloc::boxed::Box;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::rc::Rc;
use alloc::vec::Vec;

use pixelflow_ir::integral::{Affine as IrAffine, Band, Cut};
use pixelflow_ir::{Binder, ExprArena, ExprId, Fold, OpKind};

use super::fold_rules::{Copying, FactorFold, Factors, HeadRef, PlanBuilder, Substitution};
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

/// The integration rules: [`NarrowInterval`] and [`ClampMoment`]. Inert for
/// a kernel with no integral in it — each matches an interval fold and
/// nothing else.
#[must_use]
pub fn integral_rules() -> Vec<Box<dyn Rewrite>> {
    alloc::vec![
        Box::new(NarrowInterval) as Box<dyn Rewrite>,
        Box::new(ClampMoment),
    ]
}

/// Whether `rule` belongs to the family the graph runs to a fixpoint before
/// anything else when it holds an integral: the two rules here,
/// [`FactorFold`] — `∫ c·f = c·∫ f` — and `ConstantFold`, which tidies the
/// literal arithmetic a closed form leaves.
pub(crate) fn closes_integrals(rule: RuleId) -> bool {
    [
        RuleId::of(&FactorFold),
        RuleId::of(&NarrowInterval),
        RuleId::of(&ClampMoment),
        RuleId::of(&crate::math::algebra::ConstantFold),
    ]
    .contains(&rule)
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

/// Reads classes as functions of one integration variable — affine forms,
/// and the bounds indicators put on it — memoised per class, every node of
/// a class tried.
struct Recognizer<'g> {
    egraph: &'g EGraph,
    binder: Binder,
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
        Self {
            egraph,
            binder,
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
        self.egraph
            .constant(class)
            .and_then(folded)
            .unwrap_or_else(|| Rc::new(Term::Class(self.egraph.find(class))))
    }

    /// `class` as `slope·u + offset`, or `None` when no node of it is
    /// affine in the variable.
    fn affine(&mut self, class: EClassId) -> Option<Affine> {
        let class = self.egraph.find(class);
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
            ENode::Var(v) if *v == self.binder.var() => {
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
    /// one runs exactly the family: `FactorFold`, both rules here and
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
        for id in [RuleId::of(&NarrowInterval), RuleId::of(&ClampMoment)] {
            assert!(RuleSet::runtime().index_of(id).is_some());
            assert!(RuleSet::production().index_of(id).is_none());
            assert!(closes_integrals(id));
        }
    }
}
