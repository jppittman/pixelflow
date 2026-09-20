//! The decompositions of a bounded fold, as e-graph rewrites.
//!
//! ```text
//! ⊕_{[lo,hi) step s} f  =  f(lo) ⊕ ⊕_{[lo+s,hi) step s} f              (peel)
//! ⊕_{[lo,hi) step s} f  =  ⊕_{[lo,hi) step 2s} (f ⊕ f[binder:=binder+s]) (halve)
//! ⊕_{[lo,lo) step s} f  =  identity(⊕)                                  (empty)
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
//! count, and [`Fold::halve`] declines on an odd count — `PeelFold` is that
//! remainder's epilogue, run once per odd level the recursion hits (`log n`
//! of them at most), not a fallback that reverts to unrolling one term at a
//! time. `passes::expand_reduce` — the legalizer that runs when a fold
//! survives extraction, because codegen has no iteration binder — prefers the
//! same decomposition, through the same two [`Fold`] methods, so a fold that
//! survives saturation unrolls into the identical shape one that didn't have
//! to would have reached inside the graph.
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
//! - **Hash-consing is the invariance test.** A subtree that does not mention
//!   the binder rebuilds to the identical node, which `EGraph::add` returns the
//!   existing class for. The rule short-circuits those into metavariables
//!   anyway, so the template it emits is only the binder-dependent spine.
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
/// [`Fold::halve`] can still shrink (see its `apply`). Saturation has no
/// notion of "the cheaper rule tries first": every matching rule fires every
/// round, so without that guard this rule would peel a fold one term per
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
        combiner_op(fold.monoid())?;
        let (head, head_root) = substituted_body(egraph, *body, fold.binder(), last as f32)?;
        Some(RewriteAction::PeelFold {
            head,
            head_root,
            rest,
            body: *body,
        })
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
        combiner_op(fold.monoid())?;
        let (shift, shift_root) = shifted_body(egraph, *body, fold.binder(), fold.stride())?;
        Some(RewriteAction::HalveFold {
            shift,
            shift_root,
            halved,
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

/// The fold decompositions: [`HalveFold`] for the bulk of a trip count,
/// [`PeelFold`] as its odd-remainder epilogue (and a fold [`Fold::halve`]
/// declines on outright), [`EmptyFold`] to close out. Inert for a kernel
/// with no folds in it.
#[must_use]
pub fn fold_rules() -> Vec<Box<dyn Rewrite>> {
    alloc::vec![
        Box::new(HalveFold) as Box<dyn Rewrite>,
        Box::new(PeelFold) as Box<dyn Rewrite>,
        Box::new(EmptyFold),
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
/// walking one representative per e-class.
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
        // A nested fold binds a slot of its own — `lowest_free_binder` never
        // reissues a live one — so the outer substitution passes through its
        // body without capture.
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
            .nodes_raw()
            .iter()
            .any(|n| matches!(n, ExprNode::Reduce { .. }))
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
            .nodes_raw()
            .iter()
            .any(|n| matches!(n, ExprNode::Reduce { .. }))
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
