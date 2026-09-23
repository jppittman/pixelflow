//! The decompositions of a bounded fold, and factoring, as e-graph rewrites.
//!
//! ```text
//! ⊕_{[lo,hi) step s} f  =  f(lo) ⊕ ⊕_{[lo+s,hi) step s} f              (peel)
//! ⊕_{[lo,hi) step s} f  =  ⊕_{[lo,hi) step 2s} (f ⊕ f[binder:=binder+s]) (halve)
//! ⊕_{[lo,lo) step s} f  =  identity(⊕)                                  (empty)
//! ⊕_i (c ⊗ f)           =  c ⊗ ⊕_i f,   i ∉ var(c)                      (factor)
//! ```
//!
//! Peel and empty are the rules the encoding used to make unstatable. While a
//! fold's algebra, binder and range were `Const` children, a rewrite that
//! changed the range would have had to rewrite an *e-class* holding a number —
//! the same class as any literal of that value elsewhere in the kernel. With
//! the metadata in the node's identity ([`pixelflow_ir::Fold`]) a peel changes
//! only the node, and the tail shares the original body's class.
//!
//! **Peeling is O(n) applications to unroll a length-n fold; halving is
//! O(log n).** Each `HalveFold` firing doubles the body and halves the trip
//! count, and [`RangeFold::halve`](pixelflow_ir::RangeFold::halve) declines
//! on an odd count — `PeelFold` is that remainder's epilogue, run once per
//! odd level the recursion hits (`log n` of them at most), not a fallback
//! that reverts to unrolling one term at a time. `passes::expand_reduce`
//! prefers the same decomposition, through the same two
//! [`RangeFold`](pixelflow_ir::RangeFold) methods, so a surviving fold it
//! unrolls takes the identical shape saturation would have reached inside
//! the graph. It is on
//! no production path: codegen emits a fold that survives extraction as a
//! loop (`pixelflow_ir::passes::legalize`), and `expand_reduce` unrolls one
//! only for a caller that asks.
//!
//! ## Substituting under a binder, in an e-graph
//!
//! Both rules need to rebuild `body` with its binder's leaves replaced —
//! `peel` by a literal, `halve` by an expression (`binder + stride`, since the
//! doubled body still lives inside a `Reduce` and the binder must stay live)
//! — and substitution is where e-graphs and binders meet. Two things make it
//! affordable and sound here:
//!
//! - **A representative suffices.** Every node in a class denotes the same
//!   value, and `f ≡ g ⟹ f[x:=c] ≡ g[x:=c]`, so substituting through one
//!   representative gives a term equal to the substitution of the whole class.
//!   It is less *complete* — nodes added to the class later get no substituted
//!   twin — never wrong. `ChainRule` differentiates a representative for the
//!   same reason.
//! - **The class fact is the invariance test.** A class whose variance fact
//!   (`EGraph::variance`) lacks the binder denotes a function the binder does
//!   not reach, so substituting into it is the identity: the walk names the
//!   class instead of entering it, whatever its first representative spells
//!   (`i − i`, once merged with `0`, is named rather than rebuilt). A class
//!   the fact cannot clear is walked, and a subtree under it that turns out
//!   not to mention the binder is still named rather than copied, so the plan
//!   the rule emits is only the binder-dependent spine.
//!
//! A class can reach itself (`neg(neg(x)) = x`, merged), and a substitution
//! through a cycle does not terminate. The walk detects re-entry and declines
//! the whole rewrite, which is always sound.

use alloc::boxed::Box;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use pixelflow_ir::{Binder, Fold, Monoid};

use super::graph::EGraph;
use super::node::{EClassId, ENode};
use super::ops;
use super::rewrite::{Rewrite, RewriteAction};

/// What a [`HeadNode`]'s child is: an earlier entry in the same plan, or an
/// e-class that already exists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadRef {
    /// Entry `i` of the plan, which precedes this one.
    Plan(u32),
    /// A class the substitution did not move — the sharing that makes a peel
    /// cost the binder-dependent spine and nothing else.
    Class(EClassId),
}

/// One node of a peeled term, in build order.
///
/// A *plan* rather than an [`ExprArena`](pixelflow_ir::ExprArena) template,
/// because the head is a copy of a term the graph already holds and may
/// therefore contain any op the graph holds — a `Gather`, a mask — while
/// [`ops::op_from_kind`], which is what turns an arena template into nodes,
/// deliberately resolves only the ops a *rewrite rule* may name. Carrying the
/// `&'static dyn Op` from the node it was copied from needs no resolver at
/// all, and cannot disagree with the one `insert` used.
#[derive(Clone, Debug)]
pub enum HeadNode {
    /// The literal the peeled index substituted in, by bit pattern.
    Const(u32),
    Op {
        op: &'static dyn ops::Op,
        children: Vec<HeadRef>,
    },
    Reduce {
        fold: Fold,
        body: HeadRef,
    },
}

/// `⊕_{[lo,hi) step s} f = ⊕_{[lo,hi-s) step s} f ⊕ f(hi-s)`.
///
/// From the *back*, so running it to exhaustion over a `stride`-1 fold builds
/// the same left-leaning chain `passes::expand_reduce` falls back to for an
/// odd remainder. Peeling from the front is the same value in the opposite
/// association, and the difference is not cosmetic: it measured 23–42% more
/// emitted nodes on production glyphs, because the graph then has to
/// reassociate an n-deep chain to reach the shape the cost model and the
/// fusion rules were tuned on, and spends its class budget doing it. See
/// docs/plans/2026-09-09-a-fold-is-a-node.md §9.
///
/// [`HalveFold`]'s epilogue for an odd trip count, at whatever level of the
/// halving recursion it arises — declines outright on a fold
/// [`RangeFold::halve`](pixelflow_ir::RangeFold::halve) can still shrink
/// (see its `apply`). Saturation has no notion of "the cheaper rule tries
/// first": every matching rule fires every round, so without that guard
/// this rule would peel a fold one term per
/// application in parallel with `HalveFold` halving the same fold — an `n`
/// applications-worth of independent unrolling that the extractor's cost
/// model would then have to notice and discard, right back to the O(n)
/// application count `HalveFold` exists to avoid.
pub struct PeelFold;

/// `⊕_{[lo,hi) step s} f = ⊕_{[lo,hi) step 2s} (f ⊕ f[binder := binder+s])`.
///
/// The stride-2 unroll (module doc): the trip count halves and the stride
/// doubles, re-bracketing the same left-to-right order of terms —
/// `(f₀⊕f₁) ⊕ (f₂⊕f₃) ⊕ …` — rather than reordering them, so it needs only
/// associativity and holds for every [`Monoid`] here. The alternative, halving
/// the *range* into `[lo,mid)` and `[mid,hi)`, would instead pair `f₀⊕f_{n/2}`,
/// `f₁⊕f_{1+n/2}`, … — interleaving the sequence, which additionally needs
/// commutativity to still equal the original fold, and is not what this does.
///
/// Run to exhaustion (peeling the odd remainder as it arises, via
/// [`PeelFold`]) this reaches the same fully-unrolled term peeling alone
/// does, in `⌈log₂ n⌉` applications rather than `n` — the entire reason this
/// rule exists: a 34,993-term glyph fold unrolled one term per application
/// was burning that many rule applications against a budget denominated in
/// them (CLAUDE.md, "A kernel built differently on two machines?").
pub struct HalveFold;

/// `⊕_{[lo,lo)} f = identity(⊕)` — whatever the body says.
pub struct EmptyFold;

/// `⊕_i (c ⊗ f) = c ⊗ ⊕_i f`, when `i ∉ var(c)` and `⊗` distributes over `⊕`.
///
/// **Factoring**, the loop transformation that moves an *operation* out of a
/// fold, not only the evaluation of its operand (which hoisting already
/// does): one `⊗` replaces `len` of them. The pairs it knows:
///
/// | fold `⊕` | `⊗` | law |
/// |---|---|---|
/// | [`Monoid::SUM`] | `Mul` | `Σ_i (c·f) = c·Σ_i f` |
/// | [`Monoid::MIN`] | `Add` | `min_i (c+f) = c + min_i f` |
/// | [`Monoid::MAX`] | `Add` | `max_i (c+f) = c + max_i f` |
///
/// **Side condition:** the factor's class does not depend on the fold's
/// binder, read off the class variance fact (`EGraph::variance`) rather
/// than off one representative — so `(i − i)·f`, merged with `0·f`, is
/// factorable as soon as the graph knows it. The fact over-approximates, so
/// the rule can miss a factor but cannot take a binder-dependent one.
/// Either operand may be the factor: both `c ⊗ f` and `f ⊗ c` match, and the
/// result is always spelled `c ⊗ ⊕f`, since `Mul` and `Add` are commutative.
/// One firing names one factoring — the first distributing node in the
/// body's class — which is the same representative-walk incompleteness every
/// rule here accepts, and never a soundness question.
///
/// **An empty fold declines.** `Σ_∅ (c·f) = 0` but `c·Σ_∅ f = c·0`, which is
/// NaN for an infinite `c`; `min_∅ (c+f) = +∞ = c + ∞` holds for every `c`
/// but `−∞`, and the rule does not need the case — [`EmptyFold`] closes an
/// empty fold outright.
///
/// **Floating point.** Min and max with `Add` are exact: `fl(c + f)` is
/// monotone in `f`, so `min_i fl(c + f_i) = fl(c + min_i f_i)` bit for bit,
/// except where a NaN is involved or a zero's sign is chosen — both already
/// left to the target by `Min`/`Max` themselves (CLAUDE.md, "Floating point
/// at the edges"). Sum with `Mul` is not exact: `Σ fl(c·f_i)` rounds `len`
/// products and `fl(c·Σ f_i)` rounds one, and `c = ∞` against a zero term
/// can give NaN on one side only. That is a reassociation-class difference —
/// last-bit rounding plus non-finite edge cases — inside the contract
/// CLAUDE.md sets for every algebraic rule here, the same one `Associative`
/// and `FmaFusion` already rely on.
///
/// **Match depth 2**, inside `DIRTY_TRACKING_MAX_DEPTH`: the rule reads the
/// fold's body class (depth 1) and its operands' facts (depth 2). A fact
/// only changes when its own class is unioned, which is a content change to
/// that class — exactly what the dirty tracker watches.
///
/// Runtime tier only, through [`fold_rules`]: `kernel!` has no syntax that
/// builds a fold, so the macro tier's production set never holds this rule.
pub struct FactorFold;

impl Rewrite for PeelFold {
    fn name(&self) -> &str {
        "peel-fold"
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        // A range only: peeling takes one index off a count, and an interval
        // has neither indices nor a count.
        let ENode::Reduce {
            fold: Fold::Range(fold),
            body,
        } = node
        else {
            return None;
        };
        // `HalveFold`'s epilogue only (see this rule's doc): while the fold
        // can still be halved, this rule declines so the two do not
        // independently unroll the same fold in parallel.
        if fold.halve().is_some() {
            return None;
        }
        let (rest, last) = fold.peel_back()?;
        // The combiner must be nameable as an `Op` before any work is done:
        // declining early costs one lookup, declining late costs a walk of
        // the whole body.
        combiner_op(fold.monoid())?;
        let (head, head_root) = substituted_body(egraph, *body, fold.binder(), last as f32)?;
        Some(RewriteAction::PeelFold {
            head,
            head_root,
            rest: Fold::Range(rest),
            body: *body,
        })
    }
}

impl Rewrite for HalveFold {
    fn name(&self) -> &str {
        "halve-fold"
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        // A range only, like `PeelFold`: a stride is a range's.
        let ENode::Reduce {
            fold: Fold::Range(fold),
            body,
        } = node
        else {
            return None;
        };
        let halved = fold.halve()?;
        // Same ordering `PeelFold` uses, for the same reason: the combiner
        // must be nameable before any work is done, and building the shifted
        // half is a walk of the whole body — the expensive part.
        combiner_op(fold.monoid())?;
        let (shift, shift_root) = shifted_body(egraph, *body, fold.binder(), fold.stride())?;
        Some(RewriteAction::HalveFold {
            shift,
            shift_root,
            halved: Fold::Range(halved),
            body: *body,
        })
    }
}

impl Rewrite for EmptyFold {
    fn name(&self) -> &str {
        "empty-fold"
    }

    fn apply(&self, _egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let ENode::Reduce { fold, .. } = node else {
            return None;
        };
        fold.is_empty()
            .then(|| RewriteAction::Create(ENode::constant(fold.monoid().identity())))
    }
}

impl Rewrite for FactorFold {
    fn name(&self) -> &str {
        "factor-fold"
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let ENode::Reduce { fold, body } = node else {
            return None;
        };
        if fold.is_empty() {
            return None;
        }
        let distributor = distributor(fold.monoid())?;
        let binder = fold.binder().var();
        let invariant = |class: EClassId| !egraph.variance(class).depends_on(binder);
        egraph.nodes(*body).iter().find_map(|term| {
            let ENode::Op { op, children } = term else {
                return None;
            };
            if op.kind() != distributor.kind() {
                return None;
            }
            let [left, right] = children.as_slice() else {
                return None;
            };
            let (factor, rest) = match (invariant(*left), invariant(*right)) {
                (true, _) => (*left, *right),
                (false, true) => (*right, *left),
                (false, false) => return None,
            };
            Some(RewriteAction::FactorFold(Factoring {
                distributor,
                factor,
                fold: *fold,
                rest,
            }))
        })
    }
}

/// One [`FactorFold`] firing: `⊕_{fold} (factor ⊗ rest) = factor ⊗ ⊕_{fold} rest`,
/// with the side condition — `fold`'s binder is not in `factor`'s class
/// variance — already checked.
#[derive(Clone, Copy, Debug)]
pub struct Factoring {
    /// `⊗`, the operator that distributes over `fold`'s combiner.
    pub(crate) distributor: &'static dyn ops::Op,
    /// `c`, the operand the fold's binder does not reach.
    pub(crate) factor: EClassId,
    /// The fold being factored, unchanged: same algebra, binder, range.
    pub(crate) fold: Fold,
    /// `f`, the operand that stays inside the fold.
    pub(crate) rest: EClassId,
}

/// The operator that distributes over a fold's combiner — `⊗` in
/// `⊕_i (c ⊗ f) = c ⊗ ⊕_i f` — for the algebras [`FactorFold`] knows.
///
/// `PRODUCT` has no distributor that holds everywhere (`Π_i f^c = (Π_i f)^c`
/// fails for a negative base). The mask quantifiers have theirs — `BitAnd`
/// over `ANY`, `BitOr` over `ALL` — and wait on a kernel that needs them.
fn distributor(monoid: Monoid) -> Option<&'static dyn ops::Op> {
    match monoid {
        Monoid::SUM => Some(&ops::Mul),
        Monoid::MIN | Monoid::MAX => Some(&ops::Add),
        _ => None,
    }
}

/// The fold rules: [`HalveFold`] for the bulk of a trip count, [`PeelFold`]
/// as its odd-remainder epilogue (and a fold
/// [`RangeFold::halve`](pixelflow_ir::RangeFold::halve) declines on
/// outright), [`EmptyFold`] to close out, and [`FactorFold`] to move an
/// invariant factor out of the body. Inert for a kernel with no folds in it.
#[must_use]
pub fn fold_rules() -> Vec<Box<dyn Rewrite>> {
    alloc::vec![
        Box::new(HalveFold) as Box<dyn Rewrite>,
        Box::new(PeelFold) as Box<dyn Rewrite>,
        Box::new(EmptyFold),
        Box::new(FactorFold),
    ]
}

/// The `Op` that combines a fold's terms.
///
/// Local rather than [`ops::op_from_kind`] because it must cover the two mask
/// algebras, whose ops are deliberately *not* globally registered: doing that
/// would hand them to the AOT macro tier, where a `Dwrt` travelling with a
/// mask resolves before composition and miscompiles under a warp. Here the
/// caller has already named an algebra, so there is no opcode to leak.
pub(crate) fn combiner_op(monoid: Monoid) -> Option<&'static dyn ops::Op> {
    match monoid {
        Monoid::SUM => Some(&ops::Add),
        Monoid::PRODUCT => Some(&ops::Mul),
        Monoid::MIN => Some(&ops::Min),
        Monoid::MAX => Some(&ops::Max),
        Monoid::ANY => Some(ops::mask_or()),
        Monoid::ALL => Some(ops::mask_and()),
        _ => None,
    }
}

/// What a finished class contributed: where it landed, and whether anything
/// under it mentioned the binder.
#[derive(Clone, Copy)]
struct Done {
    at: HeadRef,
    varies: bool,
}

/// Build `body` with every leaf occurrence of `binder` rebuilt by `leaf`,
/// walking one representative per e-class and naming, unwalked, every class
/// whose variance fact clears the binder.
///
/// Shared by [`substituted_body`] (`binder := value`, a literal — `PeelFold`)
/// and [`shifted_body`] (`binder := binder + stride`, an expression —
/// `HalveFold`): both are "rebuild `body` with the binder's leaves replaced,"
/// differing only in what a leaf becomes. `leaf` receives the plan being
/// built (to push onto) and the binder leaf's own class (which
/// [`shifted_body`] needs, to reference the unshifted binder in what it
/// builds; [`substituted_body`] ignores it).
///
/// Returns `None` when the walk re-enters a class it is already inside: a
/// merged class can reach itself, and a substitution through a cycle does not
/// terminate. Declining costs completeness and never soundness.
fn rebuild_body(
    egraph: &EGraph,
    body: EClassId,
    binder: Binder,
    mut leaf: impl FnMut(&mut Vec<HeadNode>, EClassId) -> HeadRef,
) -> Option<(Vec<HeadNode>, HeadRef)> {
    enum Task {
        Visit(EClassId),
        Build(EClassId),
    }

    let mut plan: Vec<HeadNode> = Vec::new();
    let mut memo: BTreeMap<EClassId, Done> = BTreeMap::new();
    let mut on_stack: BTreeSet<EClassId> = BTreeSet::new();
    let mut built: Vec<Done> = Vec::new();
    let mut tasks = alloc::vec![Task::Visit(body)];

    while let Some(task) = tasks.pop() {
        match task {
            Task::Visit(class) => {
                let class = egraph.find(class);
                if let Some(&done) = memo.get(&class) {
                    built.push(done);
                    continue;
                }
                // The binder provably does not reach this class, so the
                // substitution is the identity on it — and on every member,
                // not only the representative the walk below would read.
                if !egraph.variance(class).depends_on(binder.var()) {
                    let done = named(class);
                    memo.insert(class, done);
                    built.push(done);
                    continue;
                }
                if !on_stack.insert(class) {
                    return None;
                }
                let node = egraph.nodes(class).first()?;
                // The binder itself: the one place the substitution bites.
                if matches!(node, ENode::Var(v) if *v == binder.var()) {
                    let at = leaf(&mut plan, class);
                    let done = Done { at, varies: true };
                    on_stack.remove(&class);
                    memo.insert(class, done);
                    built.push(done);
                    continue;
                }
                if node.children_slice().is_empty() {
                    let done = named(class);
                    on_stack.remove(&class);
                    memo.insert(class, done);
                    built.push(done);
                    continue;
                }
                tasks.push(Task::Build(class));
                for &child in node.children_slice().iter().rev() {
                    tasks.push(Task::Visit(child));
                }
            }
            Task::Build(class) => {
                let node = egraph.nodes(class).first()?.clone();
                let arity = node.children_slice().len();
                let start = built.len().checked_sub(arity)?;
                let kids: Vec<Done> = built.drain(start..).collect();
                let done = if kids.iter().any(|k| k.varies) {
                    plan.push(rebuild(&node, &kids)?);
                    Done {
                        at: HeadRef::Plan(plan.len() as u32 - 1),
                        varies: true,
                    }
                } else {
                    // Nothing below moved, so neither does this: the
                    // substitution is the identity on a subtree that never
                    // mentions the binder, and naming the class is how the
                    // peel shares it rather than copying it.
                    named(class)
                };
                on_stack.remove(&class);
                memo.insert(class, done);
                built.push(done);
            }
        }
    }

    // The root is named rather than assumed to be the last entry: a body that
    // never mentions the binder plans *nothing* and its head is the body's own
    // class — `⊕_{[lo,hi)} c` peeling to `c ⊕ ⊕_{[lo+1,hi)} c`, with no copy
    // made anywhere.
    Some((plan, built.pop()?.at))
}

/// Build `body[binder := value]` as a plan — `PeelFold`'s substitution. The
/// binder is resolved to a literal and does not survive into the result,
/// which is why the peeled term is safe to place outside its `Reduce`.
fn substituted_body(
    egraph: &EGraph,
    body: EClassId,
    binder: Binder,
    value: f32,
) -> Option<(Vec<HeadNode>, HeadRef)> {
    rebuild_body(egraph, body, binder, |plan, _class| {
        plan.push(HeadNode::Const(value.to_bits()));
        HeadRef::Plan(plan.len() as u32 - 1)
    })
}

/// Build `body[binder := binder + stride]` as a plan — `HalveFold`'s
/// substitution. Every leaf occurrence of the binder is rebuilt as
/// `binder + stride`, an *expression*, not a literal: unlike
/// [`substituted_body`], the binder must stay live in the result, because
/// the doubled body [`HalveFold`] builds from this is the new body of a
/// `Reduce`, not a value that has left one.
fn shifted_body(
    egraph: &EGraph,
    body: EClassId,
    binder: Binder,
    stride: u32,
) -> Option<(Vec<HeadNode>, HeadRef)> {
    rebuild_body(egraph, body, binder, |plan, class| {
        plan.push(HeadNode::Const((stride as f32).to_bits()));
        let amount = HeadRef::Plan(plan.len() as u32 - 1);
        plan.push(HeadNode::Op {
            op: &ops::Add,
            children: alloc::vec![HeadRef::Class(class), amount],
        });
        HeadRef::Plan(plan.len() as u32 - 1)
    })
}

/// A class reused as-is.
fn named(class: EClassId) -> Done {
    Done {
        at: HeadRef::Class(class),
        varies: false,
    }
}

/// One plan node, over already-planned children.
fn rebuild(node: &ENode, kids: &[Done]) -> Option<HeadNode> {
    match node {
        // A nested fold reached here binds a slot other than the one being
        // substituted, so the substitution passes through its body. One
        // that rebinds the substituted slot is never reached: it shadows the
        // slot, so its class's variance excludes it and `rebuild_body` names
        // the class before descending. "`lowest_free_binder` never reissues
        // a live slot" is not what makes that safe — it is false across a
        // `Ref`, which `Kernel::over` does not see through.
        ENode::Reduce { fold, .. } => Some(HeadNode::Reduce {
            fold: *fold,
            body: kids.first()?.at,
        }),
        ENode::Op { op, .. } => Some(HeadNode::Op {
            op: *op,
            children: kids.iter().map(|k| k.at).collect(),
        }),
        // Leaves never reach here: `Task::Build` is scheduled only for a node
        // with children.
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egraph::CostModel;
    use crate::egraph::extract::extract;
    use crate::egraph::saturate::SaturationConfig;
    use crate::egraph::{Vocabulary, insert};
    use pixelflow_ir::{ExprArena, ExprId, ExprNode, OpKind};

    fn binder(slot: u8) -> Binder {
        Binder::from_slot(slot).expect("a live binder")
    }

    /// Saturate with the fold rules alone and extract. Isolating them keeps
    /// the assertion about *these* rules rather than about whatever the whole
    /// algebra does to the residue afterwards.
    fn unroll_by_rule(arena: &ExprArena, root: ExprId) -> (ExprArena, ExprId) {
        let mut eg = EGraph::with_rules(fold_rules());
        let class = insert(arena, root, &mut eg, Vocabulary::Runtime).expect("a fold inserts");
        SaturationConfig::compatibility(60).run(&mut eg);
        let (out, out_root, _cost) = extract(&eg, class, &CostModel::latency_prior());
        (out, out_root)
    }

    fn has_fold(arena: &ExprArena) -> bool {
        arena
            .nodes()
            .any(|(_, n)| matches!(n, ExprNode::Reduce { .. }))
    }

    /// **A peel moves the range, not the body.** `Σ_{[0,3)} X` has a body
    /// that ignores the binder, so the peeled head must be *X's own e-class*
    /// — no copy — and the tail must name the same body class the original
    /// did. This is the property the range exists for: with an extent the
    /// tail would have been `Σ_{[0,2)} X(·+1)`, a rebuilt body every time.
    #[test]
    fn a_peel_shares_the_body_it_folds() {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let root = a.push_reduce(Fold::new(Monoid::SUM, binder(0), 0..3), x);

        let mut eg = EGraph::with_rules(fold_rules());
        let class = insert(&a, root, &mut eg, Vocabulary::Runtime).expect("a fold inserts");
        let body_class = match eg.nodes(class).first() {
            Some(ENode::Reduce { body, .. }) => eg.find(*body),
            other => panic!("expected a fold, got {other:?}"),
        };

        // One round, so exactly one peel has happened.
        SaturationConfig::compatibility(1).run(&mut eg);

        let sum = eg
            .nodes(class)
            .iter()
            .find_map(|n| match n {
                ENode::Op { op, children } if op.kind() == OpKind::Add => Some(children.clone()),
                _ => None,
            })
            .expect("the class must now also hold the peeled sum");
        // `rest` on the left, the peeled term on the right: the peel takes
        // the *last* index, so the chain leans the way `expand_reduce` builds
        // it (§9 of the plan — the other association measured 23–42% worse).
        assert_eq!(
            eg.find(sum[1]),
            body_class,
            "the peeled term is the body itself: substituting a binder the \
             body never reads is the identity, and hash-consing says so"
        );
        let rest_class = eg.find(sum[0]);
        match eg.nodes(rest_class).iter().find_map(ENode::fold) {
            Some(Fold::Range(rest)) => {
                assert_eq!(rest.range(), 0..2, "the rest is the shorter range");
                assert_eq!(
                    eg.find(match eg.nodes(rest_class).first() {
                        Some(ENode::Reduce { body, .. }) => *body,
                        other => panic!("expected a fold, got {other:?}"),
                    }),
                    body_class,
                    "and it folds the same body class, unchanged"
                );
            }
            other => panic!("the rest must be a range fold, got {other:?}"),
        }
    }

    /// **`PeelFold` is the epilogue, not a competitor.** An e-graph applies
    /// every matching rule every round; without this decline, `PeelFold`
    /// would independently unroll an even fold one term per application in
    /// parallel with `HalveFold`'s halving, right back to the `n`
    /// applications `HalveFold` exists to avoid (see `PeelFold`'s doc). A
    /// fold `RangeFold::halve` still shrinks must get *no* `PeelFold` action; one
    /// it declines outright on (odd, or the one-term base case) must still
    /// get its usual peel.
    #[test]
    fn peel_fold_declines_exactly_when_halve_fold_would_apply() {
        let mut a = ExprArena::new();
        let x = a.push_var(0);

        let even = a.push_reduce(Fold::new(Monoid::SUM, binder(0), 0..8), x);
        let mut eg = EGraph::with_rules(fold_rules());
        let class = insert(&a, even, &mut eg, Vocabulary::Runtime).expect("a fold inserts");
        let ENode::Reduce { fold, body } = eg.nodes(class).first().cloned().expect("a fold") else {
            panic!("expected a fold");
        };
        assert!(
            PeelFold
                .apply(&eg, class, &ENode::Reduce { fold, body })
                .is_none(),
            "8 is even — HalveFold's job, not PeelFold's"
        );

        let odd = a.push_reduce(Fold::new(Monoid::SUM, binder(0), 0..7), x);
        let mut eg2 = EGraph::with_rules(fold_rules());
        let class2 = insert(&a, odd, &mut eg2, Vocabulary::Runtime).expect("a fold inserts");
        let ENode::Reduce {
            fold: fold2,
            body: body2,
        } = eg2.nodes(class2).first().cloned().expect("a fold")
        else {
            panic!("expected a fold");
        };
        assert!(
            PeelFold
                .apply(
                    &eg2,
                    class2,
                    &ENode::Reduce {
                        fold: fold2,
                        body: body2
                    }
                )
                .is_some(),
            "7 is odd — RangeFold::halve declines, so PeelFold is the epilogue"
        );
    }

    /// **Doubling the body, one round.** `Σ_{[0,8)} X` has a body that
    /// ignores the binder, so — mirroring `a_peel_shares_the_body_it_folds`
    /// — the shifted half must be *X's own e-class*, no copy, and the
    /// doubled body must combine the original body with it, original first.
    #[test]
    fn halve_doubles_the_body_sharing_it_when_the_binder_is_unused() {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let root = a.push_reduce(Fold::new(Monoid::SUM, binder(0), 0..8), x);

        let mut eg = EGraph::with_rules(fold_rules());
        let class = insert(&a, root, &mut eg, Vocabulary::Runtime).expect("a fold inserts");
        let body_class = match eg.nodes(class).first() {
            Some(ENode::Reduce { body, .. }) => eg.find(*body),
            other => panic!("expected a fold, got {other:?}"),
        };

        // One round: 8 is even, so only `HalveFold` fires on the root (`x`
        // never reaches the binder, so `PeelFold`'s own gate is moot here —
        // `HalveFold` is simply the only rule that matches a Reduce at all).
        SaturationConfig::compatibility(1).run(&mut eg);

        // `class` now holds *two* `Reduce` nodes — the original (stride 1)
        // and the one `HalveFold` just built — so the doubled one has to be
        // picked out by its stride rather than by `find_map(ENode::fold)`,
        // which would just return whichever comes first.
        let (doubled_fold, new_body_class) = eg
            .nodes(class)
            .iter()
            .find_map(|n| match n {
                ENode::Reduce {
                    fold: Fold::Range(fold),
                    body,
                } if fold.stride() > 1 => Some((*fold, eg.find(*body))),
                _ => None,
            })
            .expect("the class must now also hold the halved fold");
        assert_eq!(doubled_fold.stride(), 2, "one halving doubles the stride");
        assert_eq!(
            doubled_fold.range(),
            0..8,
            "halve moves the stride, not the bound"
        );
        let sum = eg
            .nodes(new_body_class)
            .iter()
            .find_map(|n| match n {
                ENode::Op { op, children } if op.kind() == OpKind::Add => Some(children.clone()),
                _ => None,
            })
            .expect("the doubled body's class must hold the Add combining it with its shift");
        assert_eq!(
            eg.find(sum[0]),
            body_class,
            "the unshifted half is the body itself: a binder the body never reads shifts to \
             the identity, and hash-consing says so — `body` first, matching `b ⊕ b[binder:=binder+s]`"
        );
    }

    /// `combiner_op` is total over every [`Monoid`] constructible outside
    /// `pixelflow-ir`: `Monoid::of` (the only way to wrap an `OpKind` as a
    /// `Monoid`) is `pub(crate)` there and accepts only the six ops
    /// `OpKind::monoid_identity` names, which is exactly this match's arm
    /// list. Its `_ => None` arm — what `PeelFold`/`HalveFold` decline on —
    /// is therefore unreachable through any `Fold` this crate, or any
    /// caller of it, can build today; it exists for a `Monoid` variant that
    /// does not exist yet, the same defensive shape `PeelFold`'s equivalent
    /// check already had with no test of its own. This is the test that
    /// *is* reachable: every constructible algebra must still resolve to a
    /// combiner, so a future `Monoid` added without a matching arm here
    /// fails loudly the moment this test tries it, rather than silently
    /// falling into `_ => None` and making every fold over it inextricable.
    #[test]
    fn combiner_op_covers_every_constructible_monoid() {
        for m in [
            Monoid::SUM,
            Monoid::PRODUCT,
            Monoid::MIN,
            Monoid::MAX,
            Monoid::ANY,
            Monoid::ALL,
        ] {
            assert!(combiner_op(m).is_some(), "{m:?} has no combiner");
        }
    }

    /// The pieces of a factoring test, built directly as e-nodes so the
    /// body's shape is exactly the one written.
    struct Factorable {
        eg: EGraph,
        /// The fold's class.
        class: EClassId,
        /// The fold node itself, as a rule is handed it.
        node: ENode,
    }

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

    /// `sin(X·0.1 + i)`: reads both the lattice and the binder.
    fn wave(eg: &mut EGraph, i: EClassId) -> EClassId {
        let x = eg.add(ENode::Var(0));
        let tenth = eg.add(ENode::constant(0.1));
        let x_scaled = eg.add(op2(&ops::Mul, x, tenth));
        let phase = eg.add(op2(&ops::Add, x_scaled, i));
        eg.add(op1(&ops::Sin, phase))
    }

    /// A fold over slot 0.
    fn over(monoid: Monoid, range: core::ops::Range<u32>) -> Fold {
        Fold::new(monoid, binder(0), range)
    }

    /// `fold` over `body(i)`, with `rules` installed.
    fn folded(
        rules: Vec<Box<dyn Rewrite>>,
        fold: Fold,
        body: impl FnOnce(&mut EGraph, EClassId) -> EClassId,
    ) -> Factorable {
        let mut eg = EGraph::with_rules(rules);
        let i = eg.add(ENode::Var(fold.binder().var()));
        let body = body(&mut eg, i);
        let node = ENode::Reduce { fold, body };
        let class = eg.add(node.clone());
        Factorable { eg, class, node }
    }

    fn factor_only() -> Vec<Box<dyn Rewrite>> {
        alloc::vec![Box::new(FactorFold) as Box<dyn Rewrite>]
    }

    /// What a [`FactorFold`] firing on the fold asserts, or `None` if it
    /// declines.
    fn factoring(f: &Factorable) -> Option<Factoring> {
        match FactorFold.apply(&f.eg, f.class, &f.node)? {
            RewriteAction::FactorFold(factoring) => Some(factoring),
            other => panic!("FactorFold emitted {other:?}"),
        }
    }

    /// The firing's `(factor, rest)`, canonical.
    fn operands(f: &Factorable) -> Option<(EClassId, EClassId)> {
        factoring(f).map(|g| (f.eg.find(g.factor), f.eg.find(g.rest)))
    }

    /// Whether `class` holds `factoring`'s right-hand side,
    /// `factor ⊗ ⊕_{fold} rest`.
    fn holds(eg: &EGraph, class: EClassId, factoring: Factoring) -> bool {
        let folds_rest = |c: EClassId| {
            eg.nodes(c).iter().any(|n| {
                matches!(n, ENode::Reduce { fold, body }
                    if *fold == factoring.fold && eg.find(*body) == eg.find(factoring.rest))
            })
        };
        eg.nodes(class).iter().any(|n| match n {
            ENode::Op { op, children } if op.kind() == factoring.distributor.kind() => {
                let [factor, folded] = children.as_slice() else {
                    return false;
                };
                eg.find(*factor) == eg.find(factoring.factor) && folds_rest(*folded)
            }
            _ => false,
        })
    }

    /// **`Σ_i (Y · sin(X·0.1 + i)) = Y · Σ_i sin(X·0.1 + i)`.** `Y` is
    /// outside the binder's reach, so the fold's class gains the factored
    /// form — one `Mul` outside a fold over the narrowed body — and the fold
    /// keeps its algebra, binder and range.
    #[test]
    fn factor_fold_moves_an_invariant_factor_out_of_a_sum() {
        let mut built = None;
        let mut f = folded(factor_only(), over(Monoid::SUM, 0..8), |eg, i| {
            let y = eg.add(ENode::Var(1));
            let w = wave(eg, i);
            built = Some((y, w));
            eg.add(op2(&ops::Mul, y, w))
        });
        assert_eq!(operands(&f), built);

        let fired = factoring(&f).expect("Y is invariant");
        assert_eq!(fired.distributor.kind(), OpKind::Mul);
        assert_eq!(Some(fired.fold), f.node.fold(), "the fold is unchanged");
        let growth = f.eg.predicted_growth(&RewriteAction::FactorFold(fired));
        let before = f.eg.num_classes();
        SaturationConfig::compatibility(1).run(&mut f.eg);
        assert_eq!(
            (growth, f.eg.num_classes() - before),
            (2, 2),
            "a narrowed fold and one Mul, predicted exactly"
        );
        assert!(
            holds(&f.eg, f.class, fired),
            "the fold's class must hold Y · Σ_i sin(X·0.1 + i)"
        );
    }

    /// Either operand may be the factor: `Σ_i (f(i) · Y)` factors `Y` just
    /// as `Σ_i (Y · f(i))` does.
    #[test]
    fn factor_fold_finds_the_factor_on_either_side() {
        let mut built = None;
        let f = folded(factor_only(), over(Monoid::SUM, 0..8), |eg, i| {
            let y = eg.add(ENode::Var(1));
            let w = wave(eg, i);
            built = Some((y, w));
            eg.add(op2(&ops::Mul, w, y))
        });
        assert_eq!(operands(&f), built);
    }

    /// `min_i (Y·0.5 + cos(X + i)) = Y·0.5 + min_i cos(X + i)`, and the same
    /// for `max`: `Add` distributes over both.
    #[test]
    fn factor_fold_moves_an_invariant_offset_out_of_min_and_max() {
        for monoid in [Monoid::MIN, Monoid::MAX] {
            let mut built = None;
            let mut f = folded(factor_only(), over(monoid, 0..5), |eg, i| {
                let y = eg.add(ENode::Var(1));
                let half = eg.add(ENode::constant(0.5));
                let offset = eg.add(op2(&ops::Mul, y, half));
                let x = eg.add(ENode::Var(0));
                let phase = eg.add(op2(&ops::Add, x, i));
                let cos = eg.add(op1(&ops::Cos, phase));
                built = Some((offset, cos));
                eg.add(op2(&ops::Add, offset, cos))
            });
            assert_eq!(operands(&f), built, "{monoid:?}");
            let fired = factoring(&f).expect("Y·0.5 is invariant");
            assert_eq!(fired.distributor.kind(), OpKind::Add, "{monoid:?}");
            SaturationConfig::compatibility(1).run(&mut f.eg);
            assert!(
                holds(&f.eg, f.class, fired),
                "{monoid:?}: the fold's class must hold Y·0.5 + ⊕_i cos(X + i)"
            );
        }
    }

    /// **The side condition declines.** `Σ_i (i · sin(X·0.1 + i))`: both
    /// operands read the binder, so neither is a factor.
    #[test]
    fn factor_fold_declines_when_both_operands_read_the_binder() {
        let f = folded(factor_only(), over(Monoid::SUM, 0..8), |eg, i| {
            let w = wave(eg, i);
            eg.add(op2(&ops::Mul, i, w))
        });
        assert_eq!(operands(&f), None);
    }

    /// **An empty fold declines.** `Σ_∅ (Y·f) = 0`, while `Y · Σ_∅ f = Y·0`
    /// is NaN at an infinite `Y`; `EmptyFold` closes it instead.
    #[test]
    fn factor_fold_declines_an_empty_fold() {
        let empty = over(Monoid::SUM, 3..3);
        assert!(empty.is_empty(), "precondition");
        let f = folded(factor_only(), empty, |eg, i| {
            let y = eg.add(ENode::Var(1));
            let w = wave(eg, i);
            eg.add(op2(&ops::Mul, y, w))
        });
        assert_eq!(operands(&f), None);
    }

    /// **Only a distributing pair.** `Π_i (Y · f)` is not `Y · Π_i f` (it is
    /// `Y^len · Π_i f`), and `Σ_i (Y + f)` is not `Y + Σ_i f`.
    #[test]
    fn factor_fold_declines_an_operator_that_does_not_distribute() {
        let product = folded(factor_only(), over(Monoid::PRODUCT, 0..8), |eg, i| {
            let y = eg.add(ENode::Var(1));
            let w = wave(eg, i);
            eg.add(op2(&ops::Mul, y, w))
        });
        assert_eq!(operands(&product), None, "Mul does not distribute over Π");

        let sum_of_sums = folded(factor_only(), over(Monoid::SUM, 0..8), |eg, i| {
            let y = eg.add(ENode::Var(1));
            let w = wave(eg, i);
            eg.add(op2(&ops::Add, y, w))
        });
        assert_eq!(
            operands(&sum_of_sums),
            None,
            "Add does not distribute over Σ"
        );
    }

    /// **The class, not a representative.** `(i − i) · f(i)` reads the binder
    /// in every node it was built from, but once the graph knows `i − i = 0`
    /// the operand's class is constant, and the rule reads the class.
    #[test]
    fn factor_fold_reads_the_class_fact_not_the_representative() {
        let mut zeroish = None;
        let mut f = folded(factor_only(), over(Monoid::SUM, 0..8), |eg, i| {
            let i_minus_i = eg.add(op2(&ops::Sub, i, i));
            let w = wave(eg, i);
            zeroish = Some(i_minus_i);
            eg.add(op2(&ops::Mul, i_minus_i, w))
        });
        assert_eq!(operands(&f), None, "before the graph knows, it declines");

        let zero = f.eg.add(ENode::constant(0.0));
        f.eg.union(zeroish.expect("built"), zero);
        f.eg.rebuild();
        let (factor, _rest) = operands(&f).expect("after, the class is invariant");
        assert_eq!(factor, f.eg.find(zero));
    }

    /// **`rebuild_body` names what the fact clears.** Peeling
    /// `Σ_{[0,3)} ((i − i) + X)` once the graph knows `i − i = 0`: the body's
    /// first representative still spells the binder, but its class does not
    /// depend on it, so the substitution is the identity and the peeled head
    /// is the body's own class — nothing copied, nothing planned.
    #[test]
    fn a_peel_names_a_class_the_fact_clears() {
        let mut zeroish = None;
        let mut f = folded(fold_rules(), over(Monoid::SUM, 0..3), |eg, i| {
            let i_minus_i = eg.add(op2(&ops::Sub, i, i));
            let x = eg.add(ENode::Var(0));
            zeroish = Some(i_minus_i);
            eg.add(op2(&ops::Add, i_minus_i, x))
        });
        let zero = f.eg.add(ENode::constant(0.0));
        f.eg.union(zeroish.expect("built"), zero);
        f.eg.rebuild();

        let ENode::Reduce { body, .. } = f.node else {
            panic!("a fold")
        };
        match PeelFold.apply(&f.eg, f.class, &f.node) {
            Some(RewriteAction::PeelFold {
                head, head_root, ..
            }) => {
                assert!(head.is_empty(), "nothing to copy: {head:?}");
                assert_eq!(head_root, HeadRef::Class(f.eg.find(body)));
            }
            other => panic!("expected a peel, got {other:?}"),
        }
    }

    /// `∫_lo^hi` over `slot`, through the one public door an interval has
    /// outside `pixelflow-ir` besides `Kernel::area`: `Fold::from_bits`, with
    /// the layout its doc gives (tag 1 at bit 112, slot at 64, endpoint bits
    /// at 32 and 0).
    fn interval(slot: u8, lo: f32, hi: f32) -> Fold {
        let bits = 1u128 << 112
            | u128::from(slot) << 64
            | u128::from(lo.to_bits()) << 32
            | u128::from(hi.to_bits());
        Fold::from_bits(bits).expect("a finite, nonempty interval")
    }

    /// `(Y·X).area()` in an e-graph: its root class and both folds, outer
    /// (`u_y`) first.
    fn area_of_y_times_x(rules: Vec<Box<dyn Rewrite>>) -> (EGraph, [(EClassId, ENode); 2]) {
        let kernel = pixelflow_ir::Kernel::y()
            .mul(&pixelflow_ir::Kernel::x())
            .area();
        let (arena, root) = kernel.parts();
        let mut eg = EGraph::with_rules(rules);
        let outer_class = insert(arena, root, &mut eg, Vocabulary::Runtime).expect("inserts");
        let outer = eg.nodes(outer_class).first().expect("a class").clone();
        let ENode::Reduce {
            body: inner_class, ..
        } = outer
        else {
            panic!("area's root is a fold, got {outer:?}")
        };
        let inner_class = eg.find(inner_class);
        let inner = eg.nodes(inner_class).first().expect("a class").clone();
        (eg, [(outer_class, outer), (inner_class, inner)])
    }

    /// **(d) The range rules decline an integral.** Peeling takes an index
    /// off a count, halving doubles a stride, and an empty domain is an
    /// identity — an interval has no index, no stride, and is never empty —
    /// so all three decline on both of `area`'s folds, by pattern: they
    /// match `Fold::Range` and nothing else.
    #[test]
    fn the_range_rules_decline_an_interval() {
        let (eg, folds) = area_of_y_times_x(fold_rules());
        for (class, node) in &folds {
            assert!(
                matches!(
                    node,
                    ENode::Reduce {
                        fold: Fold::Interval(_),
                        ..
                    }
                ),
                "area builds intervals: {node:?}"
            );
            assert!(
                PeelFold.apply(&eg, *class, node).is_none(),
                "peel: {node:?}"
            );
            assert!(
                HalveFold.apply(&eg, *class, node).is_none(),
                "halve: {node:?}"
            );
            assert!(
                EmptyFold.apply(&eg, *class, node).is_none(),
                "empty: {node:?}"
            );
        }
    }

    /// **(d) Factoring is the one fold rule that holds for both domains.**
    /// `(Y·X).area()`'s inner fold is `∫_{u_x} (Y + u_y)·(X + u_x)`: the left
    /// factor does not read `u_x`, so `∫ c·f = c·∫ f` fires, with `c` the
    /// class of `Y + u_y` — read off the class variance fact, like a range's.
    #[test]
    fn factor_fold_factors_an_integral() {
        let (eg, [(_, outer), (inner_class, inner)]) = area_of_y_times_x(factor_only());
        let (
            ENode::Reduce {
                fold: outer_fold, ..
            },
            ENode::Reduce {
                fold: inner_fold,
                body,
            },
        ) = (&outer, &inner)
        else {
            panic!("two folds")
        };
        let fired = match FactorFold.apply(&eg, inner_class, &inner) {
            Some(RewriteAction::FactorFold(factoring)) => factoring,
            other => panic!("expected a factoring, got {other:?}"),
        };
        assert_eq!(fired.distributor.kind(), OpKind::Mul);
        assert_eq!(fired.fold, *inner_fold, "the integral is unchanged");
        let factor_variance = eg.variance(fired.factor);
        assert!(
            !factor_variance.depends_on(inner_fold.binder().var()),
            "the factor must not read the integral's own binder"
        );
        assert!(
            factor_variance.depends_on(1) && factor_variance.depends_on(outer_fold.binder().var()),
            "and it is the `Y + u_y` operand, which reads Y and the outer binder"
        );
        assert!(
            eg.nodes(*body).iter().any(|n| matches!(n,
                ENode::Op { op, children } if op.kind() == OpKind::Mul
                    && children.contains(&fired.factor) && children.contains(&fired.rest))),
            "factor and rest are the operands of the integrand's product"
        );
    }

    /// **A peel stops at a fold that rebinds its slot.** `Σ_{i<3} (∫_{i ∈
    /// [-½,½)} i·X + i)`: the integral rebinds slot 0 inside a sum over slot
    /// 0 — the shape `expand_refs` produces when a sum is built over an
    /// integral named by reference, since `Kernel::over` chooses a slot
    /// without seeing through a `Ref`. The inner binder shadows: the
    /// integral does not read the sum's index, so peeling names its class
    /// and copies nothing of it. Rebuilding it would substitute the peeled
    /// index for the integral's own variable — plausible, wrong pixels.
    #[test]
    fn a_peel_stops_at_a_fold_that_rebinds_its_slot() {
        let mut integral = None;
        let f = folded(fold_rules(), over(Monoid::SUM, 0..3), |eg, i| {
            let x = eg.add(ENode::Var(0));
            let ix = eg.add(op2(&ops::Mul, i, x));
            let shadowing = eg.add(ENode::Reduce {
                fold: interval(0, -0.5, 0.5),
                body: ix,
            });
            integral = Some(shadowing);
            eg.add(op2(&ops::Add, shadowing, i))
        });
        let integral = f.eg.find(integral.expect("built"));
        match PeelFold.apply(&f.eg, f.class, &f.node) {
            Some(RewriteAction::PeelFold { head, .. }) => {
                assert!(
                    !head.iter().any(|n| matches!(n, HeadNode::Reduce { .. })),
                    "the integral must be named, not rebuilt: {head:?}"
                );
                assert!(
                    head.iter()
                        .any(|n| matches!(n, HeadNode::Op { children, .. }
                        if children.contains(&HeadRef::Class(integral)))),
                    "the peeled term reads the integral's own class: {head:?}"
                );
            }
            other => panic!("expected a peel, got {other:?}"),
        }
    }

    /// **An integral no rule closed loses to a closed form.** A surviving
    /// interval is priced like a surviving `Dwrt` — keepable, so extraction
    /// can hand it to legalization, and dearer than any right-hand side a
    /// rule derives. Union it with a constant, as a closing rule would, and
    /// extraction takes the constant.
    #[test]
    fn a_closed_form_beats_a_surviving_integral() {
        let mut eg = EGraph::with_rules(Vec::new());
        let x = eg.add(ENode::Var(0));
        let u = eg.add(ENode::Var(binder(0).var()));
        let body = eg.add(op2(&ops::Add, x, u));
        let integral = eg.add(ENode::Reduce {
            fold: interval(0, -0.5, 0.5),
            body,
        });
        let (kept, kept_root, _) = extract(&eg, integral, &CostModel::latency_prior());
        assert!(
            matches!(
                kept.node(kept_root),
                ExprNode::Reduce {
                    fold: Fold::Interval(_),
                    ..
                }
            ),
            "alone, the integral is kept for legalization"
        );

        let closed = eg.add(ENode::constant(7.0));
        eg.union(integral, closed);
        eg.rebuild();
        let (out, out_root, _) = extract(&eg, integral, &CostModel::latency_prior());
        assert!(
            matches!(out.node(out_root), ExprNode::Const(v) if v == 7.0),
            "a closed form must win: got {:?}",
            out.node(out_root)
        );
    }

    /// A range and an interval over the same binder and body are different
    /// folds — `[0,1)` summed and `[0,1)` integrated agree on nothing in
    /// general — so hash-consing keeps them apart.
    #[test]
    fn a_range_and_an_interval_are_different_nodes() {
        let mut eg = EGraph::with_rules(Vec::new());
        let u = eg.add(ENode::Var(binder(0).var()));
        let sum = eg.add(ENode::Reduce {
            fold: Fold::new(Monoid::SUM, binder(0), 0..1),
            body: u,
        });
        let integral = eg.add(ENode::Reduce {
            fold: interval(0, 0.0, 1.0),
            body: u,
        });
        assert_ne!(eg.find(sum), eg.find(integral));
    }

    /// `FactorFold` is the runtime tier's and only the runtime tier's.
    #[test]
    fn factor_fold_is_in_the_runtime_set_only() {
        use crate::egraph::{RuleId, RuleSet};
        let id = RuleId::of(&FactorFold);
        assert!(RuleSet::runtime().index_of(id).is_some());
        assert!(RuleSet::production().index_of(id).is_none());
    }
}

#[cfg(test)]
mod production_shape_tests {
    use super::*;
    use crate::egraph::extract::extract;
    use crate::egraph::saturate::SaturationConfig;
    use crate::egraph::{CostModel, Vocabulary, insert};
    use pixelflow_ir::arena::{BufferDecl, BufferIdentity};
    use pixelflow_ir::{ExprArena, ExprNode, OpKind};

    fn binder(slot: u8) -> Binder {
        Binder::from_slot(slot).expect("a live binder")
    }

    fn has_fold(arena: &ExprArena) -> bool {
        arena
            .nodes()
            .any(|(_, n)| matches!(n, ExprNode::Reduce { .. }))
    }

    /// A glyph's winding is `Σ_i table[i]`-shaped: the body reads a *bound
    /// buffer* at the binder. That is the production shape, and it is the one
    /// the toy tests above do not cover.
    ///
    /// Extent 40 rather than 4, too: a fold long enough that peeling it is
    /// real work is where "the graph unrolls it" stops being obvious.
    #[test]
    fn a_table_reading_fold_unrolls() {
        let mut a = ExprArena::new();
        let buf = a.declare_buffer(BufferDecl {
            id: BufferIdentity::mint(),
            width: 64,
            height: 1,
        });
        let i = a.push_var(binder(0).var());
        let zero = a.push_const(0.0);
        let read = a.push_gather(buf, i, zero);
        let x = a.push_var(0);
        let body = a.push_binary(OpKind::Mul, read, x);
        let root = a.push_reduce(Fold::new(Monoid::SUM, binder(0), 0..40), body);

        let mut eg = EGraph::with_rules(fold_rules());
        let class = insert(&a, root, &mut eg, Vocabulary::Runtime).expect("a fold inserts");
        let stats = SaturationConfig::compatibility(200).run(&mut eg);
        let (out, out_root, _cost) = extract(&eg, class, &CostModel::latency_prior());

        assert!(
            !has_fold(&out),
            "40 terms over a bound table must unroll, not survive \
             (saturation {stats:?}); extracted: {} nodes",
            out.len()
        );
        assert!(out.len() > 40, "and the unrolled form is 40 terms wide");
    }

    /// **The same fold, priced at a real lattice.**
    ///
    /// Every other extraction test in this crate runs at
    /// [`LatticeShape::POINT`], where `evals` is 1 for every node and a
    /// node's weighted cost is just its op cost. A frame makes the weights
    /// differ by orders of magnitude between scopes, and `extract_dag_scoped`
    /// audits the DP's claim against the recomputed price of the term it
    /// names — so a weighting the two disagree about is invisible at POINT
    /// and fires here.
    ///
    /// This is the gap that let the glyph-scale claim/price mismatch reach
    /// CI with the whole crate green: production bakes at a frame, and
    /// nothing here did.
    #[test]
    fn a_table_reading_fold_prices_consistently_at_a_frame() {
        use crate::egraph::extract::extract_dag_scoped;
        use pixelflow_ir::LatticeShape;

        let mut a = ExprArena::new();
        let buf = a.declare_buffer(BufferDecl {
            id: BufferIdentity::mint(),
            width: 64,
            height: 1,
        });
        let i = a.push_var(binder(0).var());
        let zero = a.push_const(0.0);
        let read = a.push_gather(buf, i, zero);
        let x = a.push_var(0);
        let body = a.push_binary(OpKind::Mul, read, x);
        let root = a.push_reduce(Fold::new(Monoid::SUM, binder(0), 0..40), body);

        let mut eg = EGraph::with_rules(fold_rules());
        let class = insert(&a, root, &mut eg, Vocabulary::Runtime).expect("a fold inserts");
        SaturationConfig::compatibility(200).run(&mut eg);

        // The audit inside is the assertion: it is a `debug_assert`, and it
        // panics when the objective and the price come apart.
        for shape in [
            LatticeShape::POINT,
            LatticeShape::new([16, 16]),
            LatticeShape::new([256, 256]),
        ] {
            let _ = extract_dag_scoped(&eg, class, &CostModel::latency_prior(), shape);
        }
    }
}
