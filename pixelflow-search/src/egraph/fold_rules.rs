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
//! ## A right-hand side is a plan
//!
//! Every rule here — and every integration rule in `egraph::integral` —
//! answers with a [`Plan`]: the nodes its right-hand side adds, in build
//! order, over classes the graph already holds. A plan rather than an
//! [`ExprArena`] template because a rebuilt body is a copy of a term the
//! graph holds, and may contain any op the graph holds (a `Gather`, a mask),
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
//!   same reason. The representative is the class's first node that is not an
//!   integral, when it has one: a class an integration rule closed holds the
//!   closed form beside the fold, and copying the fold would copy the work of
//!   closing it too.
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
pub(crate) struct PlanBuilder {
    nodes: Vec<HeadNode>,
}

/// Whether a substitution may copy an integral no rule has closed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Copying {
    /// Decline rather than copy one: `PeelFold` and `HalveFold`, which copy
    /// a range fold's body once per term. An unclosed integrand copied 63
    /// times is 63 integrals to close again; declining leaves one, which the
    /// integration rules close where it is.
    ClosedOnly,
    /// Copy one where the class has no other representative:
    /// `NarrowInterval`, whose copy is one integral that re-closes in one
    /// step.
    Anything,
}

/// A substitution under a binder: which binder, what each of its leaves
/// becomes, and whether an unclosed integral may be copied on the way.
pub(crate) struct Substitution<L> {
    /// The binder whose leaves are replaced.
    pub(crate) binder: Binder,
    /// Whether the copy may hold an integral no rule has closed.
    pub(crate) copying: Copying,
    /// What a leaf becomes, given the builder to push onto and the leaf's
    /// own class.
    pub(crate) leaf: L,
}

impl PlanBuilder {
    fn push(&mut self, node: HeadNode) -> HeadRef {
        self.nodes.push(node);
        HeadRef::Plan(self.nodes.len() as u32 - 1)
    }

    /// A literal.
    pub(crate) fn constant(&mut self, value: f32) -> HeadRef {
        self.push(HeadNode::Const(value.to_bits()))
    }

    /// An operation over earlier entries or existing classes.
    pub(crate) fn op(&mut self, op: &'static dyn ops::Op, children: Vec<HeadRef>) -> HeadRef {
        self.push(HeadNode::Op { op, children })
    }

    /// A fold over `body`.
    pub(crate) fn reduce(&mut self, fold: Fold, body: HeadRef) -> HeadRef {
        self.push(HeadNode::Reduce { fold, body })
    }

    /// `t₁ ⊗ t₂ ⊗ … ⊗ tₙ`, left-leaning — the terms themselves, unchanged,
    /// when there is one. `None` for none.
    pub(crate) fn chain(&mut self, op: &'static dyn ops::Op, terms: &[HeadRef]) -> Option<HeadRef> {
        let (&first, rest) = terms.split_first()?;
        Some(
            rest.iter()
                .fold(first, |acc, &term| self.op(op, alloc::vec![acc, term])),
        )
    }

    /// The finished plan, rooted at `root`.
    pub(crate) fn finish(self, root: HeadRef) -> Plan {
        Plan {
            nodes: self.nodes,
            root,
        }
    }

    /// Copy `roots` of a rule template into this plan: each `Var(i)` leaf is
    /// `operands[i]`, a class the graph holds, and every other node is built.
    ///
    /// A template's ops resolve through [`ops::op_from_kind`] — the set a
    /// rule may name — so a node outside it declines the whole copy, as does
    /// a leaf a rule template cannot hold (a buffer, a uniform, a fold).
    /// Nodes shared between roots, or within one, are copied once.
    pub(crate) fn splice(
        &mut self,
        template: &ExprArena,
        roots: &[ExprId],
        operands: &[EClassId],
    ) -> Option<Vec<HeadRef>> {
        let mut copied: BTreeMap<ExprId, HeadRef> = BTreeMap::new();
        let mut spliced = Vec::with_capacity(roots.len());
        for &root in roots {
            let mut stack = alloc::vec![(root, false)];
            while let Some((id, expanded)) = stack.pop() {
                if copied.contains_key(&id) {
                    continue;
                }
                if !expanded {
                    stack.push((id, true));
                    stack.extend(template.children(id).map(|child| (child, false)));
                    continue;
                }
                let at = match template.node(id) {
                    ExprNode::Var(operand) => HeadRef::Class(*operands.get(operand as usize)?),
                    ExprNode::Const(value) => self.constant(value),
                    ExprNode::Unary(kind, ..)
                    | ExprNode::Binary(kind, ..)
                    | ExprNode::Ternary(kind, ..) => {
                        let op = ops::op_from_kind(kind)?;
                        let children = template
                            .children(id)
                            .map(|child| copied.get(&child).copied())
                            .collect::<Option<Vec<_>>>()?;
                        self.op(op, children)
                    }
                    _ => return None,
                };
                copied.insert(id, at);
            }
            spliced.push(*copied.get(&root)?);
        }
        Some(spliced)
    }

    /// Build `class` with every leaf occurrence of the substitution's
    /// binder rebuilt by its `leaf`, walking one representative per e-class
    /// and naming, unwalked, every class whose variance fact clears the
    /// binder.
    ///
    /// Shared by every rule that substitutes under a binder: `PeelFold`
    /// (`binder := value`, a literal), `HalveFold` (`binder := binder +
    /// stride`) and `NarrowInterval` (`binder := m + s·(binder − c)`), which
    /// differ only in what a leaf becomes. `leaf` receives this builder (to
    /// push onto) and the binder leaf's own class (which `HalveFold` needs,
    /// to reference the unshifted binder in what it builds).
    ///
    /// Returns `None` when the walk re-enters a class it is already inside —
    /// a merged class can reach itself, and a substitution through a cycle
    /// does not terminate — and, under [`Copying::ClosedOnly`], when the
    /// copy would hold an integral. Declining costs completeness and never
    /// soundness.
    pub(crate) fn substitute<L: FnMut(&mut Self, EClassId) -> HeadRef>(
        &mut self,
        egraph: &EGraph,
        class: EClassId,
        mut substitution: Substitution<L>,
    ) -> Option<HeadRef> {
        let Substitution {
            binder, copying, ..
        } = substitution;
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
                    if copying == Copying::ClosedOnly && is_integral(node) {
                        return None;
                    }
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

/// The node a substitution rebuilds `class` through: its first node that is
/// not an integral, or its first node when every one is. See the module doc.
fn representative(egraph: &EGraph, class: EClassId) -> Option<&ENode> {
    let nodes = egraph.nodes(class);
    nodes
        .iter()
        .find(|node| !is_integral(node))
        .or_else(|| nodes.first())
}

/// Whether `node` is a fold over an interval — an integral.
pub(crate) fn is_integral(node: &ENode) -> bool {
    matches!(
        node,
        ENode::Reduce {
            fold: Fold::Interval(_),
            ..
        }
    )
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
///
/// **Declines when the peeled term would copy an integral** no rule has
/// closed ([`Copying::ClosedOnly`]): see `HalveFold`.
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
///
/// **Declines when the shifted body would copy an integral** no rule has
/// closed ([`Copying::ClosedOnly`]). A glyph's sum over 64 pieces, each an
/// integral, unrolled with its integrands unclosed would hand the
/// integration rules 63 copies to close; declined, it hands them one, and a
/// class they closed is copied through its closed form (module doc).
pub struct HalveFold;

/// `⊕_{[lo,lo)} f = identity(⊕)` — whatever the body says.
pub struct EmptyFold;

/// `⊕_i (c₁ ⊗ … ⊗ cₖ ⊗ f) = (c₁ ⊗ … ⊗ cₖ) ⊗ ⊕_i f`, when `i ∉ var(cⱼ)` and
/// `⊗` distributes over `⊕`.
///
/// **Factoring**, the loop transformation that moves an *operation* out of a
/// fold, not only the evaluation of its operand (which hoisting already
/// does): one `⊗` replaces `len` of them. The pairs it knows:
///
/// | fold `⊕` | `⊗` | law |
/// |---|---|---|
/// | [`Monoid::SUM`], ranges and integrals | `Mul` | `Σ_i (c·f) = c·Σ_i f`, `∫ c·f = c·∫ f` |
/// | [`Monoid::MIN`] | `Add` | `min_i (c+f) = c + min_i f` |
/// | [`Monoid::MAX`] | `Add` | `max_i (c+f) = c + max_i f` |
///
/// **N-ary.** The body's `⊗` tree is flattened — through every `⊗` node of
/// a class the binder reaches, stopping at each class it does not — and
/// every invariant factor comes out in one firing, as the product of all of
/// them. `σ·[band]·[edge]` over a pixel is the case that needs it: the
/// invariant factors are the product's first two, and pulling one per
/// firing would take a round each and an `Associative` rewrite between.
/// A body that is invariant as a whole keeps its last factor inside, which
/// is the binary rule's answer on a binary body.
///
/// **Side condition:** each factor's class does not depend on the fold's
/// binder, read off the class variance fact (`EGraph::variance`) rather
/// than off one representative — so `(i − i)·f`, merged with `0·f`, is
/// factorable as soon as the graph knows it. The fact over-approximates, so
/// the rule can miss a factor but cannot take a binder-dependent one. The
/// body class's every `⊗` spelling is tried, first with an invariant factor
/// wins; a nested class is flattened through its first `⊗` node — the
/// representative-walk incompleteness every rule here accepts, never a
/// soundness question.
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
/// products and `fl(c·Σ f_i)` rounds one, re-associating the factors rounds
/// differently again, and `c = ∞` against a zero term can give NaN on one
/// side only. That is a reassociation-class difference — last-bit rounding
/// plus non-finite edge cases — inside the contract CLAUDE.md sets for every
/// algebraic rule here, the same one `Associative` and `FmaFusion` already
/// rely on. For an integral the one-point quadrature it would otherwise meet
/// makes the same trade.
///
/// **Match depth 2**, inside `DIRTY_TRACKING_MAX_DEPTH`, for the product at
/// the body's top: the rule reads the fold's body class (depth 1) and its
/// operands' facts (depth 2). A fact only changes when its own class is
/// unioned, which is a content change to that class — exactly what the
/// dirty tracker watches. A deeper flattening can see further than the
/// tracker; a change it misses is an opportunity missed, never a wrong
/// factoring.
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
        let combiner = combiner_op(fold.monoid())?;
        let mut plan = PlanBuilder::default();
        let head = plan.substitute(
            egraph,
            *body,
            Substitution {
                binder: fold.binder(),
                copying: Copying::ClosedOnly,
                leaf: |plan: &mut PlanBuilder, _class| plan.constant(last as f32),
            },
        )?;
        let rest = plan.reduce(Fold::Range(rest), HeadRef::Class(*body));
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
                copying: Copying::ClosedOnly,
                leaf: |plan: &mut PlanBuilder, class| {
                    let amount = plan.constant(stride as f32);
                    plan.op(&ops::Add, alloc::vec![HeadRef::Class(class), amount])
                },
            },
        )?;
        // `body` first: `b ⊕ b[binder := binder+s]`, the unshifted (original
        // left-to-right order) half on the left.
        let doubled = plan.op(combiner, alloc::vec![HeadRef::Class(*body), shifted]);
        let root = plan.reduce(Fold::Range(halved), doubled);
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
        let factors = egraph.nodes(*body).iter().find_map(|spelling| {
            let factors = Factors::of(egraph, spelling, distributor, binder)?;
            factors.split()
        })?;
        let mut plan = PlanBuilder::default();
        let invariant: Vec<HeadRef> = factors
            .invariant
            .iter()
            .map(|&c| HeadRef::Class(c))
            .collect();
        let variant: Vec<HeadRef> = factors.variant.iter().map(|&c| HeadRef::Class(c)).collect();
        let factor = plan.chain(distributor, &invariant)?;
        let rest = plan.chain(distributor, &variant)?;
        let folded = plan.reduce(*fold, rest);
        let root = plan.op(distributor, alloc::vec![factor, folded]);
        Some(RewriteAction::Plan(plan.finish(root)))
    }
}

/// A product's factors, in left-to-right order, flattened through the `⊗`
/// nodes of every class a binder reaches and split by whether it reaches
/// them.
#[derive(Clone, Debug, Default)]
pub(crate) struct Factors {
    /// The factors the binder does not reach.
    pub(crate) invariant: Vec<EClassId>,
    /// The factors it does.
    pub(crate) variant: Vec<EClassId>,
}

impl Factors {
    /// The factors of `spelling`, a `⊗` node, or `None` when it is not one.
    ///
    /// A class the binder does not reach is a factor as it stands; one it
    /// does is flattened through its first `⊗` node, or is a factor when it
    /// has none. A class met again inside its own flattening — a merged
    /// class that reaches itself — is a factor as it stands, so the walk
    /// terminates.
    pub(crate) fn of(
        egraph: &EGraph,
        spelling: &ENode,
        distributor: &'static dyn ops::Op,
        binder: Binder,
    ) -> Option<Self> {
        let [left, right] = binary(spelling, distributor)?;
        let mut factors = Self::default();
        let mut open: BTreeSet<EClassId> = BTreeSet::new();
        let mut stack = alloc::vec![right, left];
        while let Some(class) = stack.pop() {
            let class = egraph.find(class);
            if !egraph.variance(class).depends_on(binder.var()) {
                factors.invariant.push(class);
                continue;
            }
            let product = egraph
                .nodes(class)
                .iter()
                .find_map(|node| binary(node, distributor));
            match product {
                Some([l, r]) if open.insert(class) => stack.extend([r, l]),
                _ => factors.variant.push(class),
            }
        }
        Some(factors)
    }

    /// The factoring these factors give — invariant ones out, the rest in —
    /// or `None` when nothing is invariant. A body invariant as a whole
    /// keeps its last factor inside, so there is always something to fold.
    fn split(mut self) -> Option<Self> {
        if self.variant.is_empty() {
            self.variant.push(self.invariant.pop()?);
        }
        (!self.invariant.is_empty()).then_some(self)
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

    /// What a [`FactorFold`] firing asserts, read back out of its plan:
    /// `factor ⊗ ⊕_{fold} rest`, each operand a class or a planned product.
    #[derive(Clone, Debug)]
    struct Factoring {
        plan: Plan,
        distributor: OpKind,
        factor: HeadRef,
        fold: Fold,
        rest: HeadRef,
    }

    impl Factoring {
        /// The factor's class, when it is one class rather than a product
        /// the plan builds.
        fn factor_class(&self) -> EClassId {
            match self.factor {
                HeadRef::Class(class) => class,
                other => panic!("the factor is a planned product: {other:?}"),
            }
        }

        /// The rest's class, likewise.
        fn rest_class(&self) -> EClassId {
            match self.rest {
                HeadRef::Class(class) => class,
                other => panic!("the rest is a planned product: {other:?}"),
            }
        }
    }

    /// Decode a factoring plan: its root is `factor ⊗ reduce`, `reduce` the
    /// planned fold over the rest.
    fn decode(plan: Plan) -> Factoring {
        let planned = |r: HeadRef| match r {
            HeadRef::Plan(i) => plan.nodes[i as usize].clone(),
            HeadRef::Class(class) => panic!("expected a planned node, got class {class:?}"),
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
            factor: *factor,
            fold,
            rest: body,
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
        factoring(f).map(|g| (f.eg.find(g.factor_class()), f.eg.find(g.rest_class())))
    }

    /// Whether `class` holds `factoring`'s right-hand side,
    /// `factor ⊗ ⊕_{fold} rest`, for a factoring of one class by one class.
    fn holds(eg: &EGraph, class: EClassId, factoring: &Factoring) -> bool {
        let folds_rest = |c: EClassId| {
            eg.nodes(c).iter().any(|n| {
                matches!(n, ENode::Reduce { fold, body }
                    if *fold == factoring.fold && eg.find(*body) == eg.find(factoring.rest_class()))
            })
        };
        eg.nodes(class).iter().any(|n| match n {
            ENode::Op { op, children } if op.kind() == factoring.distributor => {
                let [factor, folded] = children.as_slice() else {
                    return false;
                };
                eg.find(*factor) == eg.find(factoring.factor_class()) && folds_rest(*folded)
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
            Some(RewriteAction::Plan(plan)) => decode(plan),
            other => panic!("expected a factoring, got {other:?}"),
        };
        assert_eq!(fired.distributor, OpKind::Mul);
        assert_eq!(fired.fold, *inner_fold, "the integral is unchanged");
        let (factor, rest) = (fired.factor_class(), fired.rest_class());
        let factor_variance = eg.variance(factor);
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
                    && children.contains(&factor) && children.contains(&rest))),
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
            Some(RewriteAction::Plan(plan)) => {
                let head = &plan.nodes;
                assert!(
                    !head.iter().any(|n| matches!(
                        n,
                        HeadNode::Reduce {
                            fold: Fold::Interval(_),
                            ..
                        }
                    )),
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

    /// **N-ary: every invariant factor in one firing.**
    /// `Σ_i (Y · (X · sin(X·0.1 + i)))` factors to `(Y·X) · Σ_i sin(…)` — the
    /// nested product is flattened, both invariant factors come out, and the
    /// plan builds the one `Mul` joining them. The binary rule took `Y` alone
    /// and left `X` for another round.
    #[test]
    fn factor_fold_pulls_every_invariant_factor_in_one_firing() {
        let mut parts = None;
        let mut f = folded(factor_only(), over(Monoid::SUM, 0..8), |eg, i| {
            let (x, y) = (eg.add(ENode::Var(0)), eg.add(ENode::Var(1)));
            let w = wave(eg, i);
            let xw = eg.add(op2(&ops::Mul, x, w));
            parts = Some((x, y, w));
            eg.add(op2(&ops::Mul, y, xw))
        });
        let (x, y, w) = parts.expect("built");
        let fired = factoring(&f).expect("Y and X are invariant");
        assert_eq!(
            fired.rest_class(),
            f.eg.find(w),
            "only the wave stays inside"
        );
        let HeadRef::Plan(product) = fired.factor else {
            panic!("the factor is the planned product of both: {fired:?}")
        };
        assert!(
            matches!(&fired.plan.nodes[product as usize], HeadNode::Op { op, children }
                if op.kind() == OpKind::Mul
                    && children == &[HeadRef::Class(f.eg.find(y)), HeadRef::Class(f.eg.find(x))]),
            "Y·X, in the product's order: {fired:?}"
        );
        SaturationConfig::compatibility(1).run(&mut f.eg);
        let factored = f.eg.nodes(f.class).iter().any(|n| {
            matches!(n, ENode::Op { op, children }
                if op.kind() == OpKind::Mul
                    && f.eg.nodes(children[1]).iter().any(|m| matches!(m,
                        ENode::Reduce { body, .. } if f.eg.find(*body) == f.eg.find(w))))
        });
        assert!(
            factored,
            "the fold's class must hold (Y·X) · Σ_i sin(X·0.1 + i)"
        );
    }

    /// **A body invariant as a whole keeps its last factor inside**, which is
    /// the binary rule's answer on a binary body: `Σ_i (Y·X) = Y · Σ_i X`.
    #[test]
    fn a_wholly_invariant_body_keeps_its_last_factor_inside() {
        let mut parts = None;
        let f = folded(factor_only(), over(Monoid::SUM, 0..8), |eg, _i| {
            let (x, y) = (eg.add(ENode::Var(0)), eg.add(ENode::Var(1)));
            parts = Some((y, x));
            eg.add(op2(&ops::Mul, y, x))
        });
        assert_eq!(operands(&f), parts);
    }

    /// `Σ_{i < n} ∫_{u ∈ [0,1)} (i + u) du`, the sum binding slot 0 and the
    /// integral slot 1: a range fold whose body *is* an integral that reads
    /// the range's index. Returns the graph, the sum's class and node, and
    /// the integral's class.
    fn a_sum_of_integrals(n: u32) -> (EGraph, EClassId, ENode, EClassId) {
        let mut eg = EGraph::with_rules(fold_rules());
        let i = eg.add(ENode::Var(binder(0).var()));
        let u = eg.add(ENode::Var(binder(1).var()));
        let body = eg.add(op2(&ops::Add, i, u));
        let integral = eg.add(ENode::Reduce {
            fold: interval(1, 0.0, 1.0),
            body,
        });
        let sum = ENode::Reduce {
            fold: over(Monoid::SUM, 0..n),
            body: integral,
        };
        let class = eg.add(sum.clone());
        (eg, class, sum, integral)
    }

    /// **(E4) A range fold does not copy an integral no rule has closed.**
    /// Halving (an even count) and peeling (an odd one) would each copy the
    /// unclosed integrand once per term; both decline, leaving one integral
    /// for the integration rules to close where it is.
    #[test]
    fn peel_and_halve_decline_to_copy_an_unclosed_integral() {
        let (eg, class, sum, _) = a_sum_of_integrals(4);
        assert!(HalveFold.apply(&eg, class, &sum).is_none(), "halve");
        let (eg, class, sum, _) = a_sum_of_integrals(3);
        assert!(PeelFold.apply(&eg, class, &sum).is_none(), "peel");
    }

    /// **(E4) A closed integral is copied through its closed form.** Once the
    /// integral's class also holds `i + ½` — what closing it gives — halving
    /// copies that, and no interval fold is in the plan.
    #[test]
    fn halve_copies_a_closed_integral_through_its_closed_form() {
        let (mut eg, class, sum, integral) = a_sum_of_integrals(4);
        let i = eg.add(ENode::Var(binder(0).var()));
        let half = eg.add(ENode::constant(0.5));
        let closed = eg.add(op2(&ops::Add, i, half));
        eg.union(integral, closed);
        eg.rebuild();
        match HalveFold.apply(&eg, class, &sum) {
            Some(RewriteAction::Plan(plan)) => assert!(
                !plan.nodes.iter().any(|n| matches!(
                    n,
                    HeadNode::Reduce {
                        fold: Fold::Interval(_),
                        ..
                    }
                )),
                "the shifted body copies the closed form, not the integral: {plan:?}"
            ),
            other => panic!("expected a halving, got {other:?}"),
        }
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
