//! The decompositions of a bounded fold, and factoring, as e-graph rewrites.
//!
//! ```text
//! ⊕_{[lo,hi) step s} f  =  f(lo) ⊕ ⊕_{[lo+s,hi) step s} f              (peel)
//! ⊕_{[lo,hi) step s} f  =  ⊕_{[lo,hi) step 2s} (f ⊕ f[binder:=binder+s]) (halve)
//! ⊕_{[lo,lo) step s} f  =  identity(⊕)                                  (empty)
//! ⊕_i (c₁ ⊗ … ⊗ cₖ ⊗ f) =  (c₁ ⊗ … ⊗ cₖ) ⊗ ⊕_i f,   i ∉ var(cⱼ)       (factor)
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
//! count, and [`Fold::halve`](pixelflow_ir::Fold::halve) declines
//! on an odd count — `PeelFold` is that remainder's epilogue, run once per
//! odd level the recursion hits (`log n` of them at most), not a fallback
//! that reverts to unrolling one term at a time. `passes::expand_reduce`
//! prefers the same decomposition, through the same two
//! [`Fold`](pixelflow_ir::Fold) methods, so a surviving fold it
//! unrolls takes the identical shape saturation would have reached inside
//! the graph. It is on
//! no production path: codegen emits a fold that survives extraction as a
//! loop (`pixelflow_ir::passes::legalize`), and `expand_reduce` unrolls one
//! only for a caller that asks.
//!
//! ## A right-hand side is a plan
//!
//! Every rule here answers with a [`Plan`]: the nodes its right-hand side
//! adds, in build order, over classes the graph already holds. A plan
//! rather than an [`ExprArena`] template because a rebuilt body is a copy
//! of a term the graph holds, and may contain any op the graph holds (a
//! `Gather`, a mask),
//! while a template resolves its ops through [`ops::op_from_kind`], which
//! deliberately admits only what a *rule* may name. One action, replayed by
//! one function in the graph, so predicting a rule's growth and applying it
//! run the same code.
//!
//! ## Substituting under a binder, in an e-graph
//!
//! Peel and halve need to rebuild `body` with its binder's leaves replaced —
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
//!   same reason. The representative is the class's first node.
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

use pixelflow_ir::{Binder, ExprArena, ExprId, ExprNode, Fold, Monoid};

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

/// One node of a [`Plan`], in build order.
///
/// Carrying the `&'static dyn Op` from the node it was copied from needs no
/// resolver at all, and cannot disagree with the one `insert` used.
#[derive(Clone, Debug)]
pub enum HeadNode {
    /// A literal, by bit pattern.
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

/// A rule's right-hand side: the nodes it adds, in build order, and which of
/// them — or which class the graph already holds — the matched class is
/// unioned with. See the module doc for why a plan and not a template.
#[derive(Clone, Debug)]
pub struct Plan {
    /// The nodes to add, each over earlier entries and existing classes.
    pub nodes: Vec<HeadNode>,
    /// The right-hand side's root.
    pub root: HeadRef,
}

/// A [`Plan`] under construction.
#[derive(Default)]
struct PlanBuilder {
    nodes: Vec<HeadNode>,
}

/// A substitution under a binder: which binder, and what each of its leaves
/// becomes.
struct Substitution<L> {
    /// The binder whose leaves are replaced.
    binder: Binder,
    /// What a leaf becomes, given the builder to push onto and the leaf's
    /// own class.
    leaf: L,
}

impl PlanBuilder {
    fn push(&mut self, node: HeadNode) -> HeadRef {
        self.nodes.push(node);
        HeadRef::Plan(self.nodes.len() as u32 - 1)
    }

    /// A literal.
    fn constant(&mut self, value: f32) -> HeadRef {
        self.push(HeadNode::Const(value.to_bits()))
    }

    /// An operation over earlier entries or existing classes.
    fn op(&mut self, op: &'static dyn ops::Op, children: Vec<HeadRef>) -> HeadRef {
        self.push(HeadNode::Op { op, children })
    }

    /// A fold over `body`.
    fn reduce(&mut self, fold: Fold, body: HeadRef) -> HeadRef {
        self.push(HeadNode::Reduce { fold, body })
    }

    /// The finished plan, rooted at `root`.
    fn finish(self, root: HeadRef) -> Plan {
        Plan {
            nodes: self.nodes,
            root,
        }
    }

    /// Build `class` with every leaf occurrence of the substitution's
    /// binder rebuilt by its `leaf`, walking one representative per e-class
    /// and naming, unwalked, every class whose variance fact clears the
    /// binder.
    ///
    /// Shared by both rules that substitute under a binder: `PeelFold`
    /// (`binder := value`, a literal) and `HalveFold` (`binder := binder +
    /// stride`), which differ only in what a leaf becomes. `leaf` receives
    /// this builder (to push onto) and the binder leaf's own class (which
    /// `HalveFold` needs, to reference the unshifted binder in what it
    /// builds).
    ///
    /// Returns `None` when the walk re-enters a class it is already inside —
    /// a merged class can reach itself, and a substitution through a cycle
    /// does not terminate. Declining costs completeness and never soundness.
    fn substitute<L: FnMut(&mut Self, EClassId) -> HeadRef>(
        &mut self,
        egraph: &EGraph,
        class: EClassId,
        mut substitution: Substitution<L>,
    ) -> Option<HeadRef> {
        let Substitution { binder, .. } = substitution;
        enum Task {
            Visit(EClassId),
            Build(EClassId),
        }

        let mut memo: BTreeMap<EClassId, Done> = BTreeMap::new();
        let mut on_stack: BTreeSet<EClassId> = BTreeSet::new();
        let mut built: Vec<Done> = Vec::new();
        let mut tasks = alloc::vec![Task::Visit(class)];

        while let Some(task) = tasks.pop() {
            match task {
                Task::Visit(class) => {
                    let class = egraph.find(class);
                    if let Some(&done) = memo.get(&class) {
                        built.push(done);
                        continue;
                    }
                    // The binder provably does not reach this class, so the
                    // substitution is the identity on it — and on every
                    // member, not only the representative the walk below
                    // would read.
                    if !egraph.variance(class).depends_on(binder.var()) {
                        let done = named(class);
                        memo.insert(class, done);
                        built.push(done);
                        continue;
                    }
                    if !on_stack.insert(class) {
                        return None;
                    }
                    let node = representative(egraph, class)?;
                    // The binder itself: the one place the substitution bites.
                    if matches!(node, ENode::Var(v) if *v == binder.var()) {
                        let at = (substitution.leaf)(self, class);
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
                    let node = representative(egraph, class)?.clone();
                    let arity = node.children_slice().len();
                    let start = built.len().checked_sub(arity)?;
                    let kids: Vec<Done> = built.drain(start..).collect();
                    let done = if kids.iter().any(|k| k.varies) {
                        let at = self.push(rebuild(&node, &kids)?);
                        Done { at, varies: true }
                    } else {
                        // Nothing below moved, so neither does this: the
                        // substitution is the identity on a subtree that
                        // never mentions the binder, and naming the class is
                        // how the rewrite shares it rather than copying it.
                        named(class)
                    };
                    on_stack.remove(&class);
                    memo.insert(class, done);
                    built.push(done);
                }
            }
        }

        // The root is named rather than assumed to be the last entry: a body
        // that never mentions the binder plans *nothing* and its result is
        // the body's own class — `⊕_{[lo,hi)} c` peeling to
        // `c ⊕ ⊕_{[lo+1,hi)} c`, with no copy made anywhere.
        Some(built.pop()?.at)
    }
}

/// The node a substitution rebuilds `class` through: its first. See the
/// module doc.
fn representative(egraph: &EGraph, class: EClassId) -> Option<&ENode> {
    egraph.nodes(class).first()
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
/// [`Fold::halve`](pixelflow_ir::Fold::halve) can still shrink
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

/// `⊕_i (c ⊗ f) = c ⊗ ⊕_i f`, when `i ∉ var(c)` and `⊗` distributes over
/// `⊕`.
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
/// `c` is one operand of the body's `⊗`, as it stands: `Σ_i (Y·(X·f(i)))`
/// takes `Y` out in one firing and `X` in a later one, out of the fold the
/// first firing built. A body invariant as a whole keeps its right operand
/// inside, so there is always something left to fold.
///
/// **Side condition:** `c`'s class does not depend on the fold's binder,
/// read off the class variance fact (`EGraph::variance`) rather than off one
/// representative — so `(i − i)·f`, merged with `0·f`, is factorable as soon
/// as the graph knows it. The fact over-approximates, so the rule can miss a
/// factor but cannot take a binder-dependent one. The body class's every
/// `⊗` spelling is tried, and the first with an invariant operand wins.
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
/// can give NaN on one side only. That is a reassociation-class difference — last-bit rounding
/// plus non-finite edge cases — inside the contract CLAUDE.md sets for every
/// algebraic rule here, the same one `Associative` and `FmaFusion` already
/// rely on.
///
/// **Match depth 2**, inside `DIRTY_TRACKING_MAX_DEPTH`: the rule reads the
/// fold's body class (depth 1) and its operands' facts (depth 2). A fact only changes when its own class is
/// unioned, which is a content change to that class — exactly what the
/// dirty tracker watches.
///
/// Runtime tier only, through [`fold_rules`]: the macro tier's set,
/// [`RuleSet::production`](super::RuleSet::production), holds no fold rule,
/// so a `kernel!` fold is factored where it is unrolled, at bake time.
pub struct FactorFold;

impl Rewrite for PeelFold {
    fn name(&self) -> &str {
        "peel-fold"
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let ENode::Reduce { fold, body } = node else {
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
        let combiner = combiner_op(fold.monoid())?;
        let mut plan = PlanBuilder::default();
        let head = plan.substitute(
            egraph,
            *body,
            Substitution {
                binder: fold.binder(),
                leaf: |plan: &mut PlanBuilder, _class| plan.constant(last as f32),
            },
        )?;
        let rest = plan.reduce(rest, HeadRef::Class(*body));
        // `rest` first: the peel takes the *last* index, so the accumulator
        // is on the left and the chain leans the way `expand_reduce` builds
        // it.
        let root = plan.op(combiner, alloc::vec![rest, head]);
        Some(RewriteAction::Plan(plan.finish(root)))
    }
}

impl Rewrite for HalveFold {
    fn name(&self) -> &str {
        "halve-fold"
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let ENode::Reduce { fold, body } = node else {
            return None;
        };
        let halved = fold.halve()?;
        // Same ordering `PeelFold` uses, for the same reason: the combiner
        // must be nameable before any work is done, and building the shifted
        // half is a walk of the whole body — the expensive part.
        let combiner = combiner_op(fold.monoid())?;
        let stride = fold.stride();
        let mut plan = PlanBuilder::default();
        // Every leaf occurrence of the binder is rebuilt as `binder +
        // stride`, an *expression*, not a literal: unlike a peel, the binder
        // must stay live in the result, because the doubled body is the new
        // body of a `Reduce`, not a value that has left one.
        let shifted = plan.substitute(
            egraph,
            *body,
            Substitution {
                binder: fold.binder(),
                leaf: |plan: &mut PlanBuilder, class| {
                    let amount = plan.constant(stride as f32);
                    plan.op(&ops::Add, alloc::vec![HeadRef::Class(class), amount])
                },
            },
        )?;
        // `body` first: `b ⊕ b[binder := binder+s]`, the unshifted (original
        // left-to-right order) half on the left.
        let doubled = plan.op(combiner, alloc::vec![HeadRef::Class(*body), shifted]);
        let root = plan.reduce(halved, doubled);
        Some(RewriteAction::Plan(plan.finish(root)))
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
        let binder = fold.binder();
        let [factor, rest] = egraph
            .nodes(*body)
            .iter()
            .find_map(|spelling| factored(egraph, spelling, distributor, binder))?;
        let mut plan = PlanBuilder::default();
        let folded = plan.reduce(*fold, HeadRef::Class(rest));
        let root = plan.op(distributor, alloc::vec![HeadRef::Class(factor), folded]);
        Some(RewriteAction::Plan(plan.finish(root)))
    }
}

/// `spelling`, a `⊗` node, as `[c, f]`: the operand the binder does not
/// reach, to come out of the fold, and the one to stay in. `None` when
/// `spelling` is not a `⊗` node, or the binder reaches both operands. Both
/// canonical.
fn factored(
    egraph: &EGraph,
    spelling: &ENode,
    distributor: &'static dyn ops::Op,
    binder: Binder,
) -> Option<[EClassId; 2]> {
    let [left, right] = binary(spelling, distributor)?.map(|class| egraph.find(class));
    let reaches = |class: EClassId| egraph.variance(class).depends_on(binder.var());
    match (reaches(left), reaches(right)) {
        (false, _) => Some([left, right]),
        (true, false) => Some([right, left]),
        (true, true) => None,
    }
}

/// `node`'s two operands, when it is a binary `op`.
fn binary(node: &ENode, op: &'static dyn ops::Op) -> Option<[EClassId; 2]> {
    let ENode::Op { op: own, children } = node else {
        return None;
    };
    if own.kind() != op.kind() {
        return None;
    }
    let &[left, right] = children.as_slice() else {
        return None;
    };
    Some([left, right])
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
/// [`Fold::halve`](pixelflow_ir::Fold::halve) declines on
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
        // slot, so its class's variance excludes it and `substitute` names
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
            Some(rest) => {
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
            None => panic!("the rest must be a fold"),
        }
    }

    /// **`PeelFold` is the epilogue, not a competitor.** An e-graph applies
    /// every matching rule every round; without this decline, `PeelFold`
    /// would independently unroll an even fold one term per application in
    /// parallel with `HalveFold`'s halving, right back to the `n`
    /// applications `HalveFold` exists to avoid (see `PeelFold`'s doc). A
    /// fold `Fold::halve` still shrinks must get *no* `PeelFold` action; one
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
            "7 is odd — Fold::halve declines, so PeelFold is the epilogue"
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
                ENode::Reduce { fold, body } if fold.stride() > 1 => Some((*fold, eg.find(*body))),
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

    /// What a [`FactorFold`] firing asserts, read back out of its plan:
    /// `factor ⊗ ⊕_{fold} rest`, each operand a class.
    #[derive(Clone, Debug)]
    struct Factoring {
        plan: Plan,
        distributor: OpKind,
        factor: EClassId,
        fold: Fold,
        rest: EClassId,
    }

    /// Decode a factoring plan: its root is `factor ⊗ reduce`, `reduce` the
    /// planned fold over the rest.
    fn decode(plan: Plan) -> Factoring {
        let planned = |r: HeadRef| match r {
            HeadRef::Plan(i) => plan.nodes[i as usize].clone(),
            HeadRef::Class(class) => panic!("expected a planned node, got class {class:?}"),
        };
        let class = |r: HeadRef| match r {
            HeadRef::Class(class) => class,
            HeadRef::Plan(_) => panic!("expected a class, got a planned node: {plan:?}"),
        };
        let HeadNode::Op { op, children } = planned(plan.root) else {
            panic!("the root is the distributor: {plan:?}")
        };
        let [factor, folded] = children.as_slice() else {
            panic!("a binary distributor: {plan:?}")
        };
        let HeadNode::Reduce { fold, body } = planned(*folded) else {
            panic!("the distributor's right operand is the fold: {plan:?}")
        };
        Factoring {
            distributor: op.kind(),
            factor: class(*factor),
            fold,
            rest: class(body),
            plan,
        }
    }

    /// What a [`FactorFold`] firing on the fold asserts, or `None` if it
    /// declines.
    fn factoring(f: &Factorable) -> Option<Factoring> {
        match FactorFold.apply(&f.eg, f.class, &f.node)? {
            RewriteAction::Plan(plan) => Some(decode(plan)),
            other => panic!("FactorFold emitted {other:?}"),
        }
    }

    /// The firing's `(factor, rest)`, canonical.
    fn operands(f: &Factorable) -> Option<(EClassId, EClassId)> {
        factoring(f).map(|g| (f.eg.find(g.factor), f.eg.find(g.rest)))
    }

    /// Whether `class` holds `factoring`'s right-hand side,
    /// `factor ⊗ ⊕_{fold} rest`, for a factoring of one class by one class.
    fn holds(eg: &EGraph, class: EClassId, factoring: &Factoring) -> bool {
        let folds_rest = |c: EClassId| {
            eg.nodes(c).iter().any(|n| {
                matches!(n, ENode::Reduce { fold, body }
                    if *fold == factoring.fold && eg.find(*body) == eg.find(factoring.rest))
            })
        };
        eg.nodes(class).iter().any(|n| match n {
            ENode::Op { op, children } if op.kind() == factoring.distributor => {
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
        assert_eq!(fired.distributor, OpKind::Mul);
        assert_eq!(Some(fired.fold), f.node.fold(), "the fold is unchanged");
        let growth =
            f.eg.predicted_growth(&RewriteAction::Plan(fired.plan.clone()));
        let before = f.eg.num_classes();
        SaturationConfig::compatibility(1).run(&mut f.eg);
        assert_eq!(
            (growth, f.eg.num_classes() - before),
            (2, 2),
            "a narrowed fold and one Mul, predicted exactly"
        );
        assert!(
            holds(&f.eg, f.class, &fired),
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
            assert_eq!(fired.distributor, OpKind::Add, "{monoid:?}");
            SaturationConfig::compatibility(1).run(&mut f.eg);
            assert!(
                holds(&f.eg, f.class, &fired),
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
            Some(RewriteAction::Plan(plan)) => {
                // Two nodes and nothing copied: the fold over the rest of the
                // range, and the combiner adding the body's own class to it.
                assert_eq!(plan.nodes.len(), 2, "nothing to copy: {plan:?}");
                assert!(
                    matches!(&plan.nodes[1], HeadNode::Op { children, .. }
                        if children[1] == HeadRef::Class(f.eg.find(body))),
                    "the peeled term is the body's own class: {plan:?}"
                );
            }
            other => panic!("expected a peel, got {other:?}"),
        }
    }

    /// **A peel stops at a fold that rebinds its slot.** `Σ_{i<3}
    /// (max_{i<5} i·X + i)`: the inner fold rebinds slot 0 inside a sum over
    /// slot 0 — the shape `expand_refs` produces when a sum is built over a
    /// fold named by reference, since `Kernel::over` chooses a slot without
    /// seeing through a `Ref`. The inner binder shadows: the max does not
    /// read the sum's index, so peeling names its class and copies nothing of
    /// it. Rebuilding it would substitute the peeled index for the inner
    /// fold's own — plausible, wrong pixels.
    #[test]
    fn a_peel_stops_at_a_fold_that_rebinds_its_slot() {
        let mut shadowing = None;
        let inner = Fold::new(Monoid::MAX, binder(0), 0..5);
        let f = folded(fold_rules(), over(Monoid::SUM, 0..3), |eg, i| {
            let x = eg.add(ENode::Var(0));
            let ix = eg.add(op2(&ops::Mul, i, x));
            let max = eg.add(ENode::Reduce {
                fold: inner,
                body: ix,
            });
            shadowing = Some(max);
            eg.add(op2(&ops::Add, max, i))
        });
        let shadowing = f.eg.find(shadowing.expect("built"));
        match PeelFold.apply(&f.eg, f.class, &f.node) {
            Some(RewriteAction::Plan(plan)) => {
                let head = &plan.nodes;
                assert!(
                    !head
                        .iter()
                        .any(|n| matches!(n, HeadNode::Reduce { fold, .. } if *fold == inner)),
                    "the inner fold must be named, not rebuilt: {head:?}"
                );
                assert!(
                    head.iter()
                        .any(|n| matches!(n, HeadNode::Op { children, .. }
                        if children.contains(&HeadRef::Class(shadowing)))),
                    "the peeled term reads the inner fold's own class: {head:?}"
                );
            }
            other => panic!("expected a peel, got {other:?}"),
        }
    }

    /// `FactorFold` is the runtime tier's and only the runtime tier's.
    #[test]
    fn factor_fold_is_in_the_runtime_set_only() {
        use crate::egraph::{RuleId, RuleSet};
        let id = RuleId::of(&FactorFold);
        assert!(RuleSet::runtime().index_of(id).is_some());
        assert!(RuleSet::production().index_of(id).is_none());
    }

    /// **One operand per firing.** `Σ_i (Y · (X · sin(X·0.1 + i)))` takes
    /// `Y` out and keeps `X · sin(…)` inside, as it stands; `X` is the next
    /// firing's, on the fold this one builds.
    #[test]
    fn factor_fold_takes_one_operand_per_firing() {
        let mut built = None;
        let f = folded(factor_only(), over(Monoid::SUM, 0..8), |eg, i| {
            let (x, y) = (eg.add(ENode::Var(0)), eg.add(ENode::Var(1)));
            let w = wave(eg, i);
            let xw = eg.add(op2(&ops::Mul, x, w));
            built = Some((y, xw));
            eg.add(op2(&ops::Mul, y, xw))
        });
        assert_eq!(operands(&f), built);
    }

    /// **A body invariant as a whole keeps its right operand inside**:
    /// `Σ_i (Y·X) = Y · Σ_i X`.
    #[test]
    fn a_wholly_invariant_body_keeps_its_right_operand_inside() {
        let mut parts = None;
        let f = folded(factor_only(), over(Monoid::SUM, 0..8), |eg, _i| {
            let (x, y) = (eg.add(ENode::Var(0)), eg.add(ENode::Var(1)));
            parts = Some((y, x));
            eg.add(op2(&ops::Mul, y, x))
        });
        assert_eq!(operands(&f), parts);
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
