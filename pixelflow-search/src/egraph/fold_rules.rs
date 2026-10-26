//! The decompositions of a bounded fold, factoring, and the discovery of a
//! range, as e-graph rewrites.
//!
//! ```text
//! ⊕_{[lo,hi) step s} f  =  f(lo) ⊕ ⊕_{[lo+s,hi) step s} f              (peel)
//! ⊕_{[lo,hi) step s} f  =  ⊕_{[lo,hi) step 2s} (f ⊕ f[binder:=binder+s]) (halve)
//! ⊕_{[lo,lo) step s} f  =  identity(⊕)                                  (empty)
//! ⊕_i (c₁ ⊗ … ⊗ cₖ ⊗ f) =  (c₁ ⊗ … ⊗ cₖ) ⊗ ⊕_i f,   i ∉ var(cⱼ)       (factor)
//! ⊕_{[lo,hi)} f         =  ⊕_{[n,m)} f,   lo ≡ n, hi ≡ m              (discover)
//! ```
//!
//! **A range is known when both its bounds are literals, and peel, halve and
//! empty read only a known range** (L6 of
//! docs/language/2026-10-20-a-bound-is-a-value.md). A bound that is a value is
//! a class, and a class gains a literal member when a rule reaches it: peel
//! substituting `i := k` into an inner `0..i` is how one does. Discovery is
//! the rule that notices — [`KnownBound`] puts the fold over the literals in
//! the fold's class, and the three rules above then see it. Nothing declared
//! the range known.
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
//! count, and [`KnownRange::halve`](pixelflow_ir::KnownRange::halve) declines
//! on an odd count — `PeelFold` is that remainder's epilogue, run once per
//! odd level the recursion hits (`log n` of them at most), not a fallback
//! that reverts to unrolling one term at a time. No lowering pass unrolls a
//! fold outside the graph: codegen emits one that survives extraction as a
//! loop (`pixelflow_ir::passes::legalize` leaves it standing).
//!
//! ## A right-hand side is a plan
//!
//! Every rule here answers with a [`Plan`]: the nodes its right-hand side
//! adds, in build order, over classes the graph already holds. A plan
//! rather than an `ExprArena` template because a rebuilt body is a copy
//! of a term the graph holds, and may contain any op the graph holds (a
//! `Gather`, a mask),
//! while a template may name only what [`template_may_name`](super::ops::template_may_name) admits,
//! which is deliberately what a *rule* may name. One action, replayed by
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
//!   same reason. The representative is the class's first node, except in a
//!   nested fold's value bound, or anywhere under one, where it is the first
//!   that keeps L3 of docs/language/2026-10-20-a-bound-is-a-value.md and is no
//!   fold (a class's fact is the meet over its members, so its first member
//!   can read a coordinate the class does not), and the rewrite declines when
//!   none is. It declines too when the class fact of a bound it rebuilds
//!   fails L3: the term is lawful, but a fact is seeded at birth and never
//!   repaired upward, so the bound can land in a class that already exists, or
//!   be minted over a child that did, seeded wider than the term.
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

use pixelflow_ir::{Binder, Bound, Fold, KnownRange, Lit, Monoid, OpKind, ShiftCount};

use super::graph::EGraph;
use super::node::{EClassId, ENode};
use super::rewrite::{Rewrite, RewriteAction, RootShapes};
use super::shift::ShiftDirection;

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
/// Carries the `OpKind` of the node it was copied from, so a plan can hold any
/// op the graph holds (`Gather`, the masks), not only those a template may name.
#[derive(Clone, Debug)]
pub enum HeadNode {
    /// A literal.
    Const(Lit),
    Op {
        op: OpKind,
        children: Vec<HeadRef>,
    },
    Reduce {
        fold: Fold,
        lo: Bound<HeadRef>,
        hi: Bound<HeadRef>,
        body: HeadRef,
    },
    /// A shift, its count carried as the parameter it is.
    Shift {
        direction: ShiftDirection,
        count: ShiftCount,
        operand: HeadRef,
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
        self.push(HeadNode::Const(Lit::Real(value)))
    }

    /// An operation over earlier entries or existing classes.
    fn op(&mut self, op: OpKind, children: Vec<HeadRef>) -> HeadRef {
        self.push(HeadNode::Op { op, children })
    }

    /// A fold over `body`, from `lo` to `hi`. Four arguments because they
    /// are the node's four fields, as `ENode::reduce`'s are.
    fn reduce(
        &mut self,
        fold: Fold,
        lo: Bound<HeadRef>,
        hi: Bound<HeadRef>,
        body: HeadRef,
    ) -> HeadRef {
        self.push(HeadNode::Reduce { fold, lo, hi, body })
    }

    /// The finished plan, rooted at `root`.
    fn finish(self, root: HeadRef) -> Plan {
        Plan {
            nodes: self.nodes,
            root,
        }
    }

    /// Build `class` with every leaf occurrence of the substitution's
    /// binder rebuilt by its `leaf`, walking one member per e-class and site
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
    /// **Which member** depends on where the class sits ([`Site`]). Under a
    /// body, the class's first member: every member denotes the same value
    /// and a body may read anything. As a nested fold's value bound, the first
    /// member that keeps L3 of docs/language/2026-10-20-a-bound-is-a-value.md:
    /// reads no coordinate and not its fold's own binder, judged by
    /// `Fold::bound_may_read` on the member's own variance over its children's
    /// facts, and that is no fold. The class's fact is the meet over its
    /// members, so a member can read a coordinate the class does not vary with
    /// (`((i + X) + 1) − X` beside `i + 1`), and rebuilding through it would
    /// hand the graph a bound that varies, which `EGraph::add` refuses by
    /// panicking inside saturation. The same holds of everything a bound is
    /// made of, so the walk keeps the site for every class under a bound.
    ///
    /// The substitution itself adds no refused bit: a leaf becomes a literal or
    /// the substituted binder plus its stride, and that binder is neither a
    /// coordinate nor a nested fold's own (a fold that rebinds it is named
    /// before it is entered), so a node this plan mints is as lawful as the
    /// member it was read from. What the walk cannot see is the fact the graph
    /// will read for it. A class's fact is seeded at birth and never repaired
    /// upward (`EGraph::variance` is what `add` reads), so a bound's class can
    /// be seeded wider than the bound as a term: because the bound resolves to
    /// a class that already exists, or because the plan mints a class for it
    /// over a child that did. The walk does not replay the memo to find out.
    /// The plan is built node by node, and the graph asks of each fold whether
    /// the class fact of one of its value bounds fails L3: if so the rewrite is
    /// declined there (`plan_shape`), before `add`'s backstop could panic.
    ///
    /// Returns `None` when the walk re-enters a class it is already inside —
    /// a merged class can reach itself, and a substitution through a cycle
    /// does not terminate — and when a class in or under a value bound has no
    /// member that keeps L3 and is no fold (a fold under a bound is not
    /// entered: its body would need the law lifted by its own binder).
    /// Declining costs completeness and never soundness.
    fn substitute<L: FnMut(&mut Self, EClassId) -> HeadRef>(
        &mut self,
        egraph: &EGraph,
        class: EClassId,
        mut substitution: Substitution<L>,
    ) -> Option<HeadRef> {
        let Substitution { binder, .. } = substitution;
        enum Task {
            Visit(EClassId, Site),
            Build(EClassId, Site),
        }

        let mut memo = Memo::default();
        let mut on_stack: BTreeSet<EClassId> = BTreeSet::new();
        let mut built: Vec<Done> = Vec::new();
        let mut tasks = alloc::vec![Task::Visit(class, Site::Body)];

        while let Some(task) = tasks.pop() {
            match task {
                Task::Visit(class, site) => {
                    let class = egraph.find(class);
                    if let Some(done) = memo.get(class, site) {
                        built.push(done);
                        continue;
                    }
                    // The binder provably does not reach this class, so the
                    // substitution is the identity on it — and on every
                    // member, not only the one the walk below would read.
                    if !egraph.variance(class).depends_on(binder.var()) {
                        let done = named(class);
                        memo.insert(class, site, done);
                        built.push(done);
                        continue;
                    }
                    if !on_stack.insert(class) {
                        return None;
                    }
                    let node = *site.member(egraph, class)?;
                    // The binder itself: the one place the substitution bites.
                    if matches!(node, ENode::Var(v) if v == binder.var()) {
                        let at = (substitution.leaf)(self, class);
                        let done = Done { at, varies: true };
                        on_stack.remove(&class);
                        memo.insert(class, site, done);
                        built.push(done);
                        continue;
                    }
                    if node.children_slice().is_empty() {
                        let done = named(class);
                        on_stack.remove(&class);
                        memo.insert(class, site, done);
                        built.push(done);
                        continue;
                    }
                    tasks.push(Task::Build(class, site));
                    let children = node.children_slice();
                    for (position, &child) in children.iter().enumerate().rev() {
                        let child_site = site.of_child(&node, position);
                        tasks.push(Task::Visit(child, child_site));
                    }
                }
                Task::Build(class, site) => {
                    // The member `Visit` chose: the choice is a function of the
                    // graph, which does not change during a walk.
                    let node = *site.member(egraph, class)?;
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
                    memo.insert(class, site, done);
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

/// Where [`PlanBuilder::substitute`] is rebuilding a class, which decides the
/// member it reads.
#[derive(Clone, Copy)]
enum Site {
    /// A fold's body, or what is under one: any member does, and the first is
    /// read. Every site a fold over literals has.
    Body,
    /// A value bound of the nested fold `fold`, or what is under one: the
    /// member must keep L3 for that fold and be no fold. The end is not part
    /// of the site: the law is the same at both (`Fold::bound_may_read`).
    Bound { fold: Fold },
}

impl Site {
    /// The member of `class` the walk reads at this site: for a body its
    /// first, for a bound its first that keeps L3 and is no fold. `None` when
    /// a bound's class has none.
    ///
    /// A fold is skipped under a bound because the walk does not enter one
    /// there: its body would need the law lifted by its own binder. Skipping
    /// it reads the next member, where declining the rewrite on it would not.
    fn member(self, egraph: &EGraph, class: EClassId) -> Option<&ENode> {
        match self {
            Site::Body => egraph.nodes(class).next(),
            Site::Bound { fold } => egraph.nodes(class).find(|node| {
                !matches!(node, ENode::Reduce { .. })
                    && fold.bound_may_read(node.variance(
                        |key| egraph.ref_variance(key),
                        |child| egraph.variance(child),
                    ))
            }),
        }
    }

    /// The site of `node`'s child `position`. A fold's body stays where the
    /// fold is, and its value bounds are bounds; every other node's children
    /// are where it is.
    ///
    /// Total because [`Self::member`] never reads a fold at a bound: a fold
    /// is only ever asked here from a body.
    fn of_child(self, node: &ENode, position: usize) -> Site {
        match (node, position) {
            (ENode::Reduce { header, .. }, 1..) => Site::Bound {
                fold: header.fold(),
            },
            _ => self,
        }
    }
}

/// What the walk has finished, by class and site. A class under a body is
/// keyed by itself alone, as a walk that had no sites did; the map a bound's
/// classes are in is separate, so a fold over literals never builds a key
/// wider than a class.
#[derive(Default)]
struct Memo {
    body: BTreeMap<EClassId, Done>,
    bound: BTreeMap<(EClassId, Fold), Done>,
}

impl Memo {
    fn get(&self, class: EClassId, site: Site) -> Option<Done> {
        match site {
            Site::Body => self.body.get(&class),
            Site::Bound { fold } => self.bound.get(&(class, fold)),
        }
        .copied()
    }

    fn insert(&mut self, class: EClassId, site: Site, done: Done) {
        match site {
            Site::Body => self.body.insert(class, done),
            Site::Bound { fold } => self.bound.insert((class, fold), done),
        };
    }
}

/// `⊕_{[lo,hi) step s} f = ⊕_{[lo,hi-s) step s} f ⊕ f(hi-s)`.
///
/// From the *back*, so running it to exhaustion over a `stride`-1 fold builds
/// a left-leaning chain. Peeling from the front is the same value in the
/// opposite association, and the difference is not cosmetic: it measured
/// 23–42% more emitted nodes on production glyphs, because the graph then has
/// to reassociate an n-deep chain to reach the shape the cost model and the
/// fusion rules were tuned on, and spends its class budget doing it. See
/// docs/archive/plans/2026-09-09-a-fold-is-a-node.md §9.
///
/// [`HalveFold`]'s epilogue for an odd trip count, at whatever level of the
/// halving recursion it arises — declines outright on a fold
/// [`KnownRange::halve`](pixelflow_ir::KnownRange::halve) can still shrink
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
///
/// A nested fold whose value bound reads the binder is shifted with it: the
/// copy's bound is `i + s`, bound arithmetic that only the rules make
/// (docs/language/2026-10-20-a-bound-is-a-value.md §1.4). It stays inside the
/// known range being halved, which ends at or before 2²⁴, so it is exact. The
/// rule builds it only through a member of the bound's class that already keeps
/// L3, so the shifted bound reads what that member read, the outer index plus
/// its stride, and a bound class with no such member declines the rewrite. The
/// graph declines it too where the class fact of the shifted bound fails L3: a
/// fact is seeded at birth and never repaired upward, so a lawful term can
/// land in a class seeded wider than it, or be minted over a child that did,
/// and `EGraph::add` reads the class. The graph asks whether it would refuse
/// the fold before making it, so the rewrite is declined there and `add`'s
/// backstop is not reached with a bound that varies.
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
/// **A fold known to be empty declines.** `Σ_∅ (c·f) = 0` but `c·Σ_∅ f = c·0`,
/// which is NaN for an infinite `c`, and `min_∅ (c+f) = +∞ = c + ∞` for every
/// `c` but `−∞`; [`EmptyFold`] closes a known empty fold outright.
///
/// **A fold whose range is not known factors**, accepting that edge at a
/// runtime empty range under "edge-case IEEE conformance is not on offer"
/// (CLAUDE.md; decision 5 of docs/language/2026-10-20-a-bound-is-a-value.md):
/// declining would lose `c·Σ f` for every per-call count. At a runtime `n = 0`
/// the factored form is `c ⊗ identity`, and that differs from the fold's own
/// identity for:
///
/// - an infinite `c`;
/// - a NaN `c`, where `NaN·0` is NaN;
/// - a negative `c`, where `c·(+0)` is `-0.0`;
/// - for `MIN` and `MAX`, a NaN `c`.
///
/// **Under [`Monoid::SUM`], a `c` the graph proves zero takes its fold with
/// it**: a literal `0`, or a class merged with one, such as `X − X`. The `Mul`
/// annihilator (`0·y → 0`, `OpKind::annihilator`) rewrites `c · Σ f` to `0`, so
/// no fold is left for the guard `passes::legalize` builds to wrap, and a bound
/// that is no count (NaN, a fraction, past 2²⁴) gives `0` where the fold as
/// written is NaN. For a finite body the answers agree in the domain. That is
/// the annihilation rule's edge, not this rule's (BACKLOG E9).
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

/// `⊕_{[lo,hi)} f = ⊕_{[n_lo, n_hi)} f` where the fold's value bounds' classes
/// hold literals naming counts: **L8, discovery**, the one normalizing rule
/// (docs/language/2026-10-20-a-bound-is-a-value.md §3.2). A value bound whose
/// class gains a `Const` member — `PeelFold` substituting `i := k` into an
/// inner `0..i` is how one does — is that count, and the fold over it joins the
/// fold's class. The known form is the one `PeelFold`, `HalveFold` and
/// `EmptyFold` read (L6), so the inner fold becomes unrollable; nothing
/// declared it.
///
/// It fires only when the read makes the whole range known: a fold with one
/// bound known and one a value is no more unrollable or priceable than before,
/// and would spend a minted id and an application on a node no rule reads. A
/// literal naming no count — a fraction, a negative, past 2²⁴ — is no count,
/// and a mask is no count of any sort: the rule declines on both (L5 of
/// docs/language/2026-10-20-a-bound-is-a-value.md: such a fold is NaN, by the
/// guard `passes::legalize` ends with). A pair it makes known runs forwards (L2
/// of the same plan: `[k, j)` with `k > j` is the empty range at `k`), which
/// [`EmptyFold`] then closes.
///
/// The member is found by scan, as `math::algebra`'s `constant_of` finds one,
/// not through the class's constant fact.
///
/// **Match depth 1**, inside `DIRTY_TRACKING_MAX_DEPTH`: it reads the members
/// of the fold's bound classes. A `Const` joins a bound class by a union, a
/// content change to that class, which the dirty tracker watches.
///
/// The fold it adds is `ENode::literal_twin`'s, the one function that makes a
/// fold's literal form, and the extractor prices a value-bounded fold as it.
/// One construction buys this much: given the same literal for each bound
/// class, the form this rule puts in the class and the one a price reads are
/// the same fold. Which literal each finds is its own: this rule scans the
/// bound class's members, a price reads the form chosen for the class.
///
/// Last in [`fold_rules`], so every rule before it keeps its `RuleIdx`, and
/// inert on a fold over literals: it declines before reading the graph.
///
/// Inert is not absent. The rule is evaluated on every fold, so
/// `Saturation::total_evals` rises on every kernel that has one: no gate reads
/// it, but the anytime curves and the research bins that fit them do, and a
/// moved curve there is this rule's evaluations and not a regression. A sweep
/// also checks the class budget before each rule's scan, so this rule adds one
/// check, after [`FactorFold`]. A run over literals differs from one without
/// the rule, then, in that counter and in one label: when the last allowed
/// iteration's `FactorFold` commits past the budget (its growth prediction can
/// under-count, `EGraph::predicted_growth`), the run reports `class_cap` where
/// it reported `iteration_ceiling`. The graph, the applications, the
/// iterations and the extraction are the same.
pub struct KnownBound;

impl Rewrite for PeelFold {
    fn name(&self) -> &str {
        "peel-fold"
    }

    fn root_shapes(&self) -> RootShapes {
        RootShapes::kind(OpKind::Reduce)
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let (fold, _, _, body) = node.as_reduce()?;
        // A range that is not known cannot be peeled: decline.
        let range = node.known_range()?;
        // `HalveFold`'s epilogue only (see this rule's doc): while the fold
        // can still be halved, this rule declines so the two do not
        // independently unroll the same fold in parallel.
        if range.halve().is_some() {
            return None;
        }
        let (rest, last) = range.peel_back()?;
        // The fold must have a combiner before any work is done: declining
        // early costs one lookup, declining late costs a walk of the whole
        // body.
        let combiner = combiner_op(fold.monoid())?;
        let mut plan = PlanBuilder::default();
        let head = plan.substitute(
            egraph,
            body,
            Substitution {
                binder: fold.binder(),
                leaf: |plan: &mut PlanBuilder, _class| plan.constant(last as f32),
            },
        )?;
        let (rest_fold, lo, hi) = fold.over(rest);
        let rest = plan.reduce(rest_fold, lo, hi, HeadRef::Class(body));
        // `rest` first: the peel takes the *last* index, so the accumulator
        // is on the left and the chain leans left.
        let root = plan.op(combiner, alloc::vec![rest, head]);
        Some(RewriteAction::Plan(plan.finish(root)))
    }
}

impl Rewrite for HalveFold {
    fn name(&self) -> &str {
        "halve-fold"
    }

    fn root_shapes(&self) -> RootShapes {
        RootShapes::kind(OpKind::Reduce)
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let (fold, _, _, body) = node.as_reduce()?;
        // A range that is not known cannot be halved: decline.
        let range = node.known_range()?;
        let halved = range.halve()?;
        // Same ordering `PeelFold` uses, for the same reason: the combiner
        // must be nameable before any work is done, and building the shifted
        // half is a walk of the whole body — the expensive part.
        let combiner = combiner_op(fold.monoid())?;
        let stride = range.stride();
        let mut plan = PlanBuilder::default();
        // Every leaf occurrence of the binder is rebuilt as `binder +
        // stride`, an *expression*, not a literal: unlike a peel, the binder
        // must stay live in the result, because the doubled body is the new
        // body of a `Reduce`, not a value that has left one.
        let shifted = plan.substitute(
            egraph,
            body,
            Substitution {
                binder: fold.binder(),
                leaf: |plan: &mut PlanBuilder, class| {
                    let amount = plan.constant(stride as f32);
                    plan.op(OpKind::Add, alloc::vec![HeadRef::Class(class), amount])
                },
            },
        )?;
        // `body` first: `b ⊕ b[binder := binder+s]`, the unshifted (original
        // left-to-right order) half on the left.
        let doubled = plan.op(combiner, alloc::vec![HeadRef::Class(body), shifted]);
        let (halved_fold, lo, hi) = fold.over(halved);
        let root = plan.reduce(halved_fold, lo, hi, doubled);
        Some(RewriteAction::Plan(plan.finish(root)))
    }
}

impl Rewrite for EmptyFold {
    fn name(&self) -> &str {
        "empty-fold"
    }

    fn root_shapes(&self) -> RootShapes {
        RootShapes::kind(OpKind::Reduce)
    }

    fn apply(&self, _egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let ENode::Reduce { header, .. } = node else {
            return None;
        };
        node.known_range()
            .is_some_and(KnownRange::is_empty)
            .then(|| RewriteAction::Create(ENode::Const(header.fold().monoid().identity())))
    }
}

impl Rewrite for FactorFold {
    fn name(&self) -> &str {
        "factor-fold"
    }

    fn root_shapes(&self) -> RootShapes {
        RootShapes::kind(OpKind::Reduce)
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let (fold, lo, hi, body) = node.as_reduce()?;
        // A range known to be empty is `EmptyFold`'s. One that is not known
        // factors: at a runtime empty range the factored form is
        // `c ⊗ identity`, the edge decision 5 accepts (doc above).
        if node.known_range().is_some_and(KnownRange::is_empty) {
            return None;
        }
        let distributor = distributor(fold.monoid())?;
        let binder = fold.binder();
        let [factor, rest] = egraph
            .nodes(body)
            .find_map(|spelling| factored(egraph, spelling, distributor, binder))?;
        let mut plan = PlanBuilder::default();
        // The factored fold keeps the node's own bounds, each as the class it
        // is when it is a value.
        let folded = plan.reduce(
            fold,
            lo.map(HeadRef::Class),
            hi.map(HeadRef::Class),
            HeadRef::Class(rest),
        );
        let root = plan.op(distributor, alloc::vec![HeadRef::Class(factor), folded]);
        Some(RewriteAction::Plan(plan.finish(root)))
    }
}

impl Rewrite for KnownBound {
    fn name(&self) -> &str {
        "known-bound"
    }

    fn root_shapes(&self) -> RootShapes {
        RootShapes::kind(OpKind::Reduce)
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        // A fold over literals has nothing to discover: `literal_twin`
        // declines it on its header alone.
        let literal = |class: EClassId| egraph.nodes(class).find_map(ENode::literal);
        node.literal_twin(literal).map(RewriteAction::Create)
    }
}

/// `spelling`, a `⊗` node, as `[c, f]`: the operand the binder does not
/// reach, to come out of the fold, and the one to stay in. `None` when
/// `spelling` is not a `⊗` node, or the binder reaches both operands. Both
/// canonical.
fn factored(
    egraph: &EGraph,
    spelling: &ENode,
    distributor: OpKind,
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
fn binary(node: &ENode, op: OpKind) -> Option<[EClassId; 2]> {
    let &ENode::Op { op: own, children } = node else {
        return None;
    };
    if own != op {
        return None;
    }
    let [left, right] = children[..] else {
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
fn distributor(monoid: Monoid) -> Option<OpKind> {
    match monoid {
        Monoid::SUM => Some(OpKind::Mul),
        Monoid::MIN | Monoid::MAX => Some(OpKind::Add),
        _ => None,
    }
}

/// The fold rules: [`HalveFold`] for the bulk of a trip count, [`PeelFold`]
/// as its odd-remainder epilogue (and a fold
/// [`KnownRange::halve`](pixelflow_ir::KnownRange::halve) declines on
/// outright), [`EmptyFold`] to close out, [`FactorFold`] to move an
/// invariant factor out of the body, and [`KnownBound`] to make a range known
/// when the graph discovers a bound's constant (L8). Inert for a kernel with
/// no folds in it.
#[must_use]
pub fn fold_rules() -> Vec<Box<dyn Rewrite>> {
    alloc::vec![
        Box::new(HalveFold) as Box<dyn Rewrite>,
        Box::new(PeelFold) as Box<dyn Rewrite>,
        Box::new(EmptyFold),
        Box::new(FactorFold),
        Box::new(KnownBound),
    ]
}

/// The op that combines a fold's terms, the mask algebras (`ANY`, `ALL`)
/// included.
///
/// `None` for an algebra with no value-level combiner, `SEQ` among them: its
/// combine is "then", not a value operation. Pricing reads the `None` as the
/// unpriceable sentinel, and `PeelFold` and `HalveFold` decline on it.
pub(crate) fn combiner_op(monoid: Monoid) -> Option<OpKind> {
    match monoid {
        Monoid::SUM => Some(OpKind::Add),
        Monoid::PRODUCT => Some(OpKind::Mul),
        Monoid::MIN => Some(OpKind::Min),
        Monoid::MAX => Some(OpKind::Max),
        Monoid::ANY => Some(OpKind::BitOr),
        Monoid::ALL => Some(OpKind::BitAnd),
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
        // A nested fold reached here does not rebind the slot being
        // substituted, so the substitution passes through its body and its
        // value bounds alike (L7 of docs/language/2026-10-20-a-bound-is-a-value.md).
        // One that rebinds the slot is never reached: its class's variance is
        // `(var(body) \ slot) ∪ var(lo) ∪ var(hi)` (L4 of the same plan), and
        // no bound reads its own fold's slot (L3), so the
        // fact lacks the slot and `substitute` names the class before
        // descending. "`lowest_free_binder` never reissues a live slot" is
        // not what makes that safe — it is false across a `Ref`, which
        // `Kernel::over` does not see through.
        ENode::Reduce { header, .. } => {
            // `substitute` plans every child of the node, so `kids` is the
            // body and then each value bound.
            let (body, values) = kids.split_first()?;
            let (lo, hi) = header.attach(values.iter().map(|k| k.at));
            Some(HeadNode::Reduce {
                fold: header.fold(),
                lo,
                hi,
                body: body.at,
            })
        }
        ENode::Op { op, .. } => Some(HeadNode::Op {
            op: *op,
            children: kids.iter().map(|k| k.at).collect(),
        }),
        ENode::Shift {
            direction, count, ..
        } => Some(HeadNode::Shift {
            direction: *direction,
            count: *count,
            operand: kids.first()?.at,
        }),
        // Leaves never reach here: `Task::Build` is scheduled only for a node
        // with children.
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egraph::saturate::SaturationConfig;
    use crate::egraph::{CostModel, extract};
    use crate::egraph::{RuleSet, Saturation};
    use crate::egraph::{Vocabulary, insert};
    use pixelflow_ir::arena::{UniformDecl, UniformIdentity};
    use pixelflow_ir::fold::KNOWN_RANGE;
    use pixelflow_ir::variance::compute_arena_variance;
    use pixelflow_ir::{ExprArena, ExprId, ExprNode, OpKind, Variance};

    fn binder(slot: u8) -> Binder {
        Binder::from_slot(slot).expect("a live binder")
    }

    /// The upper bound of every `fold` reachable from `root`.
    fn nested_bounds(arena: &ExprArena, root: ExprId, fold: Fold) -> Vec<Bound<ExprId>> {
        let mut bounds = Vec::new();
        let (mut seen, mut stack) = (BTreeSet::new(), alloc::vec![root]);
        while let Some(id) = stack.pop() {
            if !seen.insert(id) {
                continue;
            }
            if let ExprNode::Reduce {
                fold: found, hi, ..
            } = arena.node(id)
                && found == fold
            {
                bounds.push(hi);
            }
            stack.extend(arena.children(id));
        }
        bounds
    }

    /// `Σ_{i<n} Σ_{j<i} X·j`, built with `push_reduce`: the inner `sum(0)`
    /// runs over `0..i`, a value bound reading the outer `sum(1)`'s index (L3:
    /// an enclosing binder is a bound's to read), and the outer over `0..n`.
    fn tri(n: u64) -> (ExprArena, ExprId) {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let j = a.push_var(binder(0).var());
        let i = a.push_var(binder(1).var());
        let body = a.push_binary(OpKind::Mul, x, j);
        let inner = a.push_reduce(
            Fold::new(Monoid::SUM, binder(0)),
            Bound::Lit(0),
            Bound::Value(i),
            body,
        );
        let outer = a.push_reduce(
            Fold::new(Monoid::SUM, binder(1)),
            Bound::Lit(0),
            Bound::Lit(n),
            inner,
        );
        (a, outer)
    }

    /// A uniform leaf: one number for the whole lattice, any fold's to count by.
    fn a_uniform(eg: &mut EGraph) -> EClassId {
        eg.add(ENode::Uniform(UniformDecl {
            id: UniformIdentity::mint(),
            default: 6.0,
        }))
    }

    /// The range of `node`, when it is a fold.
    fn reduce_range(node: &ENode) -> Option<KnownRange> {
        matches!(node, ENode::Reduce { .. }).then(|| node.known_range().expect(KNOWN_RANGE))
    }

    /// The body class of a fold node.
    fn body_of(node: &ENode) -> EClassId {
        let (_, _, _, body) = node.as_reduce().expect("a fold");
        body
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
        let root = a.push_reduce(
            Fold::new(Monoid::SUM, binder(0)),
            Bound::Lit(0),
            Bound::Lit(3),
            x,
        );

        let mut eg = Saturation::over(RuleSet::new(fold_rules()));
        let class = insert(&a, root, &mut eg, Vocabulary::Runtime).expect("a fold inserts");
        let body_class = match eg.nodes(class).next() {
            Some(node @ ENode::Reduce { .. }) => eg.find(body_of(node)),
            other => panic!("expected a fold, got {other:?}"),
        };

        // One round, so exactly one peel has happened.
        SaturationConfig::compatibility(1).run(&mut eg);

        let sum = eg
            .nodes(class)
            .find_map(|n| match n {
                ENode::Op { op, children } if *op == OpKind::Add => Some(*children),
                _ => None,
            })
            .expect("the class must now also hold the peeled sum");
        // `rest` on the left, the peeled term on the right: the peel takes
        // the *last* index, so the chain leans left (§9 of the plan — the
        // other association measured 23–42% worse).
        assert_eq!(
            eg.find(sum[1]),
            body_class,
            "the peeled term is the body itself: substituting a binder the \
             body never reads is the identity, and hash-consing says so"
        );
        let rest_class = eg.find(sum[0]);
        match eg.nodes(rest_class).find_map(reduce_range) {
            Some(rest) => {
                assert_eq!(rest.range(), 0..2, "the rest is the shorter range");
                assert_eq!(
                    eg.find(match eg.nodes(rest_class).next() {
                        Some(node @ ENode::Reduce { .. }) => body_of(node),
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
    /// fold `KnownRange::halve` still shrinks must get *no* `PeelFold` action; one
    /// it declines outright on (odd, or the one-term base case) must still
    /// get its usual peel.
    #[test]
    fn peel_fold_declines_exactly_when_halve_fold_would_apply() {
        let mut a = ExprArena::new();
        let x = a.push_var(0);

        let even = a.push_reduce(
            Fold::new(Monoid::SUM, binder(0)),
            Bound::Lit(0),
            Bound::Lit(8),
            x,
        );
        let mut eg = Saturation::over(RuleSet::new(fold_rules()));
        let class = insert(&a, even, &mut eg, Vocabulary::Runtime).expect("a fold inserts");
        let node = eg.nodes(class).next().cloned().expect("a fold");
        assert!(
            PeelFold.apply(&eg, class, &node).is_none(),
            "8 is even — HalveFold's job, not PeelFold's"
        );

        let odd = a.push_reduce(
            Fold::new(Monoid::SUM, binder(0)),
            Bound::Lit(0),
            Bound::Lit(7),
            x,
        );
        let mut eg2 = Saturation::over(RuleSet::new(fold_rules()));
        let class2 = insert(&a, odd, &mut eg2, Vocabulary::Runtime).expect("a fold inserts");
        let node2 = eg2.nodes(class2).next().cloned().expect("a fold");
        assert!(
            PeelFold.apply(&eg2, class2, &node2).is_some(),
            "7 is odd — KnownRange::halve declines, so PeelFold is the epilogue"
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
        let root = a.push_reduce(
            Fold::new(Monoid::SUM, binder(0)),
            Bound::Lit(0),
            Bound::Lit(8),
            x,
        );

        let mut eg = Saturation::over(RuleSet::new(fold_rules()));
        let class = insert(&a, root, &mut eg, Vocabulary::Runtime).expect("a fold inserts");
        let body_class = match eg.nodes(class).next() {
            Some(node @ ENode::Reduce { .. }) => eg.find(body_of(node)),
            other => panic!("expected a fold, got {other:?}"),
        };

        // One round: 8 is even, so only `HalveFold` fires on the root (`x`
        // never reaches the binder, so `PeelFold`'s own gate is moot here —
        // `HalveFold` is simply the only rule that matches a Reduce at all).
        SaturationConfig::compatibility(1).run(&mut eg);

        // `class` now holds *two* `Reduce` nodes — the original (stride 1)
        // and the one `HalveFold` just built — so the doubled one has to be
        // picked out by its stride rather than by the first `Reduce` it
        // finds, which would just return whichever comes first.
        let (doubled_range, new_body_class) = eg
            .nodes(class)
            .find_map(|n| match n {
                ENode::Reduce { .. } => reduce_range(n)
                    .filter(|range| range.stride() > 1)
                    .map(|range| (range, eg.find(body_of(n)))),
                _ => None,
            })
            .expect("the class must now also hold the halved fold");
        assert_eq!(doubled_range.stride(), 2, "one halving doubles the stride");
        assert_eq!(
            doubled_range.range(),
            0..8,
            "halve moves the stride, not the bound"
        );
        let sum = eg
            .nodes(new_body_class)
            .find_map(|n| match n {
                ENode::Op { op, children } if *op == OpKind::Add => Some(*children),
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

    /// **A shift in a fold body is copied with its direction and count.** The
    /// body reads the binder, so a peel and a halving both rebuild the shift
    /// over the substituted operand: the copy is a shift node of the same
    /// direction by the same count, over a new operand class, never the
    /// original operand and never another count. Nothing else notices a wrong
    /// count here: it is as valid a [`ShiftCount`] as the right one.
    #[test]
    fn a_shift_in_a_fold_body_is_copied_with_its_direction_and_count() {
        const COUNT: u32 = 3;
        for direction in [ShiftDirection::Left, ShiftDirection::Right] {
            for (range, rule) in [
                (0..3, &PeelFold as &dyn Rewrite),
                (0..4, &HalveFold as &dyn Rewrite),
            ] {
                let mut a = ExprArena::new();
                let (x, i) = (a.push_var(0), a.push_var(binder(0).var()));
                let count = a.push_const(COUNT as f32);
                let sum = a.push_binary(OpKind::Add, x, i);
                let shifted = a.push_binary(direction.op(), sum, count);
                let root = a.push_reduce(
                    Fold::new(Monoid::SUM, binder(0)),
                    Bound::Lit(range.start),
                    Bound::Lit(range.end),
                    shifted,
                );

                let mut eg = Saturation::over(RuleSet::new(fold_rules()));
                let class = insert(&a, root, &mut eg, Vocabulary::Runtime).expect("a fold inserts");
                let node = *eg.nodes(class).next().expect("a fold");

                let Some(RewriteAction::Plan(plan)) = rule.apply(&eg, class, &node) else {
                    panic!("{direction:?} {range:?}: {} did not plan", rule.name());
                };
                let copies: Vec<_> = plan
                    .nodes
                    .iter()
                    .filter_map(|n| match n {
                        HeadNode::Shift {
                            direction,
                            count,
                            operand,
                        } => Some((*direction, count.bits(), *operand)),
                        _ => None,
                    })
                    .collect();
                assert!(
                    matches!(
                        copies.as_slice(),
                        [(d, COUNT, HeadRef::Plan(_))] if *d == direction
                    ),
                    "{direction:?} {range:?}: {} planned {copies:?}",
                    rule.name()
                );

                // Applied: every shift the graph now holds is that shift, over
                // two operands — the original's and the copy's.
                SaturationConfig::compatibility(1).run(&mut eg);
                let mut operands = BTreeSet::new();
                for class in eg.class_ids() {
                    for node in eg.nodes(class) {
                        if let ENode::Shift {
                            direction: d,
                            count,
                            operand,
                        } = node
                        {
                            assert_eq!((*d, count.bits()), (direction, COUNT));
                            operands.insert(eg.find(*operand));
                        }
                    }
                }
                assert_eq!(
                    operands.len(),
                    2,
                    "{direction:?} {range:?}: the original shift and its copy"
                );
            }
        }
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
        eg: Saturation,
        /// The fold's class.
        class: EClassId,
        /// The fold node itself, as a rule is handed it.
        node: ENode,
    }

    /// `sin(X·0.1 + i)`: reads both the lattice and the binder.
    fn wave(eg: &mut EGraph, i: EClassId) -> EClassId {
        let x = eg.add(ENode::Var(0));
        let tenth = eg.add(ENode::constant(0.1));
        let x_scaled = eg.add(ENode::apply(OpKind::Mul, &[x, tenth]));
        let phase = eg.add(ENode::apply(OpKind::Add, &[x_scaled, i]));
        eg.add(ENode::apply(OpKind::Sin, &[phase]))
    }

    /// A fold over slot 0, and the range it runs over.
    fn over(monoid: Monoid, range: core::ops::Range<u64>) -> (Fold, core::ops::Range<u64>) {
        (Fold::new(monoid, binder(0)), range)
    }

    /// `fold` over `body(i)`, with `rules` installed.
    fn folded(
        rules: Vec<Box<dyn Rewrite>>,
        (fold, range): (Fold, core::ops::Range<u64>),
        body: impl FnOnce(&mut EGraph, EClassId) -> EClassId,
    ) -> Factorable {
        let mut eg = Saturation::over(RuleSet::new(rules));
        let i = eg.add(ENode::Var(fold.binder().var()));
        let body = body(&mut eg, i);
        let node = ENode::reduce(fold, Bound::Lit(range.start), Bound::Lit(range.end), body);
        let class = eg.add(node);
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
        lo: Bound<EClassId>,
        hi: Bound<EClassId>,
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
        let HeadNode::Reduce { fold, lo, hi, body } = planned(*folded) else {
            panic!("the distributor's right operand is the fold: {plan:?}")
        };
        Factoring {
            distributor: op,
            factor: class(*factor),
            fold,
            lo: lo.map(class),
            hi: hi.map(class),
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
            eg.nodes(c).any(|n| {
                n.as_reduce().is_some_and(|(fold, lo, hi, body)| {
                    (fold, lo, hi) == (factoring.fold, factoring.lo, factoring.hi)
                        && eg.find(body) == eg.find(factoring.rest)
                })
            })
        };
        eg.nodes(class).any(|n| match n {
            ENode::Op { op, children } if *op == factoring.distributor => {
                let [factor, folded] = children[..] else {
                    return false;
                };
                eg.find(factor) == eg.find(factoring.factor) && folds_rest(folded)
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
            eg.add(ENode::apply(OpKind::Mul, &[y, w]))
        });
        assert_eq!(operands(&f), built);

        let fired = factoring(&f).expect("Y is invariant");
        assert_eq!(fired.distributor, OpKind::Mul);
        let (fold, lo, hi, _) = f.node.as_reduce().expect("the fold's own node is a Reduce");
        assert_eq!(
            (fired.fold, fired.lo, fired.hi),
            (fold, lo, hi),
            "the fold is unchanged: its algebra, binder and bounds"
        );
        let growth = f.eg.predicted_growth(
            &RewriteAction::Plan(fired.plan.clone()),
            &mut Default::default(),
        );
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
            eg.add(ENode::apply(OpKind::Mul, &[w, y]))
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
                let offset = eg.add(ENode::apply(OpKind::Mul, &[y, half]));
                let x = eg.add(ENode::Var(0));
                let phase = eg.add(ENode::apply(OpKind::Add, &[x, i]));
                let cos = eg.add(ENode::apply(OpKind::Cos, &[phase]));
                built = Some((offset, cos));
                eg.add(ENode::apply(OpKind::Add, &[offset, cos]))
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
            eg.add(ENode::apply(OpKind::Mul, &[i, w]))
        });
        assert_eq!(operands(&f), None);
    }

    /// **A fold known to be empty declines.** `Σ_∅ (Y·f) = 0`, while
    /// `Y · Σ_∅ f = Y·0` is NaN at an infinite `Y`; `EmptyFold` closes it
    /// instead.
    #[test]
    fn factor_fold_declines_an_empty_fold() {
        let f = folded(factor_only(), over(Monoid::SUM, 3..3), |eg, i| {
            let y = eg.add(ENode::Var(1));
            let w = wave(eg, i);
            eg.add(ENode::apply(OpKind::Mul, &[y, w]))
        });
        assert!(
            f.node.known_range().expect(KNOWN_RANGE).is_empty(),
            "precondition"
        );
        assert_eq!(operands(&f), None);
    }

    /// **Only a distributing pair.** `Π_i (Y · f)` is not `Y · Π_i f` (it is
    /// `Y^len · Π_i f`), and `Σ_i (Y + f)` is not `Y + Σ_i f`.
    #[test]
    fn factor_fold_declines_an_operator_that_does_not_distribute() {
        let product = folded(factor_only(), over(Monoid::PRODUCT, 0..8), |eg, i| {
            let y = eg.add(ENode::Var(1));
            let w = wave(eg, i);
            eg.add(ENode::apply(OpKind::Mul, &[y, w]))
        });
        assert_eq!(operands(&product), None, "Mul does not distribute over Π");

        let sum_of_sums = folded(factor_only(), over(Monoid::SUM, 0..8), |eg, i| {
            let y = eg.add(ENode::Var(1));
            let w = wave(eg, i);
            eg.add(ENode::apply(OpKind::Add, &[y, w]))
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
            let i_minus_i = eg.add(ENode::apply(OpKind::Sub, &[i, i]));
            let w = wave(eg, i);
            zeroish = Some(i_minus_i);
            eg.add(ENode::apply(OpKind::Mul, &[i_minus_i, w]))
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
            let i_minus_i = eg.add(ENode::apply(OpKind::Sub, &[i, i]));
            let x = eg.add(ENode::Var(0));
            zeroish = Some(i_minus_i);
            eg.add(ENode::apply(OpKind::Add, &[i_minus_i, x]))
        });
        let zero = f.eg.add(ENode::constant(0.0));
        f.eg.union(zeroish.expect("built"), zero);
        f.eg.rebuild();

        let body = body_of(&f.node);
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
        let inner = Fold::new(Monoid::MAX, binder(0));
        let f = folded(fold_rules(), over(Monoid::SUM, 0..3), |eg, i| {
            let x = eg.add(ENode::Var(0));
            let ix = eg.add(ENode::apply(OpKind::Mul, &[i, x]));
            let max = eg.add(ENode::reduce(inner, Bound::Lit(0), Bound::Lit(5), ix));
            shadowing = Some(max);
            eg.add(ENode::apply(OpKind::Add, &[max, i]))
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
            let xw = eg.add(ENode::apply(OpKind::Mul, &[x, w]));
            built = Some((y, xw));
            eg.add(ENode::apply(OpKind::Mul, &[y, xw]))
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
            eg.add(ENode::apply(OpKind::Mul, &[y, x]))
        });
        assert_eq!(operands(&f), parts);
    }

    /// **A fold under a value bound factors** (decision 5 of the plan).
    /// `Σ_{i<u} (Y·sin(X·0.1 + i))` takes `Y` out and keeps the bound `u`, a
    /// class, in the factored fold: the range is not known, and the rule does
    /// not need it to be.
    #[test]
    fn factor_fold_factors_under_a_value_bound() {
        let fold = Fold::new(Monoid::SUM, binder(0));
        let mut eg = Saturation::over(RuleSet::new(factor_only()));
        let i = eg.add(ENode::Var(fold.binder().var()));
        let y = eg.add(ENode::Var(1));
        let w = wave(&mut eg, i);
        let body = eg.add(ENode::apply(OpKind::Mul, &[y, w]));
        let u = a_uniform(&mut eg);
        let node = ENode::reduce(fold, Bound::Lit(0), Bound::Value(u), body);
        let class = eg.add(node);
        let mut f = Factorable { eg, class, node };

        assert_eq!(f.node.known_range(), None, "precondition");
        let fired = factoring(&f).expect("Y is invariant, and the range need not be known");
        assert_eq!(f.eg.find(fired.factor), y);
        assert_eq!(f.eg.find(fired.rest), w);
        assert_eq!(
            (fired.lo, fired.hi),
            (Bound::Lit(0), Bound::Value(u)),
            "the factored fold keeps the node's own bounds"
        );
        SaturationConfig::compatibility(1).run(&mut f.eg);
        assert!(
            holds(&f.eg, f.class, &fired),
            "the fold's class must hold Y · Σ_{{i<u}} sin(X·0.1 + i)"
        );
    }

    /// **A peel substitutes into a nested fold's bound** (L7). `Σ_{i<3} Σ_{j<i}
    /// X·j` peels to the sum of `Σ_{i<2} …` and the last term, whose inner
    /// fold's bound is the peeled index: `Σ_{j<2} X·j`, the body (which never
    /// reads `i`) named.
    #[test]
    fn a_peel_substitutes_into_a_nested_folds_bound() {
        let (a, root) = tri(3);
        let mut eg = Saturation::over(RuleSet::new(fold_rules()));
        let class = insert(&a, root, &mut eg, Vocabulary::Runtime).expect("a fold inserts");
        let outer = *eg.nodes(class).next().expect("a fold");

        let Some(RewriteAction::Plan(plan)) = PeelFold.apply(&eg, class, &outer) else {
            panic!("a fold over 0..3 peels");
        };
        let inner = plan.nodes.iter().find_map(|node| match node {
            HeadNode::Reduce { fold, lo, hi, body }
                if *fold == Fold::new(Monoid::SUM, binder(0)) =>
            {
                Some((*lo, *hi, *body))
            }
            _ => None,
        });
        let Some((Bound::Lit(0), Bound::Value(HeadRef::Plan(k)), HeadRef::Class(_))) = inner else {
            panic!("the inner fold keeps its slot, its lower bound and its body: {plan:?}");
        };
        assert!(
            matches!(plan.nodes[k as usize], HeadNode::Const(Lit::Real(last)) if last == 2.0),
            "the bound is the peeled index, 2: {plan:?}"
        );
    }

    /// **A peel names a rebinding fold with a value bound** instead of
    /// entering it (L7). `Σ_{i<3} (max_{i<u} i·X + i)`: the inner fold rebinds
    /// slot 0 inside a sum over slot 0, and its bound is a uniform. Its fact is
    /// its body's less its own index, unioned with the bound's (L4), so it lacks
    /// the slot the peel substitutes, and the walk names its class.
    #[test]
    fn a_peel_names_a_rebinding_fold_with_a_value_bound() {
        let mut shadowing = None;
        let inner = Fold::new(Monoid::MAX, binder(0));
        let f = folded(fold_rules(), over(Monoid::SUM, 0..3), |eg, i| {
            let x = eg.add(ENode::Var(0));
            let ix = eg.add(ENode::apply(OpKind::Mul, &[i, x]));
            let u = a_uniform(eg);
            let max = eg.add(ENode::reduce(inner, Bound::Lit(0), Bound::Value(u), ix));
            shadowing = Some(max);
            eg.add(ENode::apply(OpKind::Add, &[max, i]))
        });
        let shadowing = f.eg.find(shadowing.expect("built"));
        let Some(RewriteAction::Plan(plan)) = PeelFold.apply(&f.eg, f.class, &f.node) else {
            panic!("a fold over 0..3 peels");
        };
        assert!(
            !plan
                .nodes
                .iter()
                .any(|n| matches!(n, HeadNode::Reduce { fold, .. } if *fold == inner)),
            "the inner fold must be named, not rebuilt: {plan:?}"
        );
        assert!(
            plan.nodes
                .iter()
                .any(|n| matches!(n, HeadNode::Op { children, .. }
                if children.contains(&HeadRef::Class(shadowing)))),
            "the peeled term reads the inner fold's own class: {plan:?}"
        );
    }

    /// **A range becomes known by discovery** (L8). After one peel of
    /// `Σ_{i<3} Σ_{j<i} X·j` the peeled head holds `Σ_{j<k} X·j` with `k` the
    /// class of the literal 2; `KnownBound` puts the fold over `0..2` in its
    /// class, where `PeelFold`, `HalveFold` and `EmptyFold` can read it.
    #[test]
    fn a_peel_makes_an_inner_bound_known() {
        let (a, root) = tri(3);
        let mut eg = Saturation::over(RuleSet::new(fold_rules()));
        insert(&a, root, &mut eg, Vocabulary::Runtime).expect("a fold inserts");
        SaturationConfig::compatibility(2).run(&mut eg);

        let inner = Fold::new(Monoid::SUM, binder(0));
        let known = eg.class_ids().any(|class| {
            let bounds: Vec<_> = eg
                .nodes(class)
                .filter_map(ENode::as_reduce)
                .filter(|&(fold, ..)| fold == inner)
                .map(|(_, lo, hi, _)| (lo, hi))
                .collect();
            bounds.iter().any(|&(_, hi)| matches!(hi, Bound::Value(_)))
                && bounds.contains(&(Bound::Lit(0), Bound::Lit(2)))
        });
        assert!(
            known,
            "a class holds the peeled head both as a value bound and over 0..2"
        );
    }

    /// **A literal joining a bound's class wakes the rule.** `Σ_{j<u}` with `u`
    /// a uniform is left alone; once the graph learns `u = 4` (a union, which
    /// is a content change to `u`'s class, one level below the fold's), the
    /// next round puts the fold over `0..4` in the fold's class. Dirty tracking
    /// reads the change through the fold's child: the rule's match depth is 1.
    #[test]
    fn a_literal_joining_a_bound_class_makes_the_range_known() {
        let fold = Fold::new(Monoid::SUM, binder(0));
        let mut eg = Saturation::over(RuleSet::new(fold_rules()));
        let x = eg.add(ENode::Var(0));
        let j = eg.add(ENode::Var(fold.binder().var()));
        let body = eg.add(ENode::apply(OpKind::Mul, &[x, j]));
        let u = a_uniform(&mut eg);
        let class = eg.add(ENode::reduce(fold, Bound::Lit(0), Bound::Value(u), body));
        let known = |eg: &Saturation| {
            eg.nodes(class)
                .filter_map(ENode::as_reduce)
                .any(|(_, lo, hi, _)| (lo, hi) == (Bound::Lit(0), Bound::Lit(4)))
        };

        SaturationConfig::compatibility(2).run(&mut eg);
        assert!(!known(&eg), "nothing is known of u yet");

        let four = eg.add(ENode::constant(4.0));
        eg.union(u, four);
        eg.rebuild();
        SaturationConfig::compatibility(1).run(&mut eg);
        assert!(known(&eg), "u = 4 makes the fold's range 0..4");
    }

    /// A fold whose bound classes are `lo` and `hi`, over `X·j`, for
    /// `KnownBound` to be asked of.
    fn bounded_by(lo: Option<Lit>, hi: Lit) -> (Saturation, ENode, EClassId) {
        let fold = Fold::new(Monoid::SUM, binder(0));
        let mut eg = Saturation::over(RuleSet::new(fold_rules()));
        let x = eg.add(ENode::Var(0));
        let j = eg.add(ENode::Var(fold.binder().var()));
        let body = eg.add(ENode::apply(OpKind::Mul, &[x, j]));
        let lo = lo.map_or(Bound::Lit(0), |lit| Bound::Value(eg.add(ENode::Const(lit))));
        let hi = Bound::Value(eg.add(ENode::Const(hi)));
        let node = ENode::reduce(fold, lo, hi, body);
        let class = eg.add(node);
        (eg, node, class)
    }

    /// `KnownBound` declines a fold over literals on its header alone.
    #[test]
    fn known_bound_declines_a_literal_fold() {
        let f = folded(fold_rules(), over(Monoid::SUM, 0..4), |eg, i| {
            let x = eg.add(ENode::Var(0));
            eg.add(ENode::apply(OpKind::Mul, &[x, i]))
        });
        assert!(KnownBound.apply(&f.eg, f.class, &f.node).is_none());
    }

    /// **A bound whose class holds a count makes the range known.** `Σ_{j<k}`
    /// over a `k` whose class holds the literal 2 is the fold over `0..2`.
    #[test]
    fn known_bound_makes_a_range_known_from_a_class_holding_a_count() {
        let (eg, node, class) = bounded_by(None, Lit::Real(2.0));
        let (fold, _, _, body) = node.as_reduce().expect("a fold");
        match KnownBound.apply(&eg, class, &node) {
            Some(RewriteAction::Create(twin)) => assert_eq!(
                twin,
                ENode::reduce(fold, Bound::Lit(0), Bound::Lit(2), body)
            ),
            other => panic!("expected the fold over 0..2, got {other:?}"),
        }
    }

    /// A literal that is no count is no count: a fraction, and a count past the
    /// domain, decline (L5: such a fold is NaN, by the guard `passes::legalize`
    /// ends with).
    #[test]
    fn known_bound_declines_a_count_out_of_domain() {
        for no_count in [2.5, 16_777_218.0] {
            let (eg, node, class) = bounded_by(None, Lit::Real(no_count));
            assert!(
                KnownBound.apply(&eg, class, &node).is_none(),
                "{no_count} names no count"
            );
        }
    }

    /// A mask is no count of any sort.
    #[test]
    fn known_bound_declines_a_mask() {
        let (eg, node, class) = bounded_by(None, Lit::Mask(true));
        assert!(KnownBound.apply(&eg, class, &node).is_none());
    }

    /// **A pair the read makes known runs forwards** (L2): `[5, 3)` is the
    /// empty range at 5, which `EmptyFold` then closes.
    #[test]
    fn known_bound_runs_a_discovered_pair_forwards() {
        let (eg, node, class) = bounded_by(Some(Lit::Real(5.0)), Lit::Real(3.0));
        let (fold, _, _, body) = node.as_reduce().expect("a fold");
        let Some(RewriteAction::Create(twin)) = KnownBound.apply(&eg, class, &node) else {
            panic!("a pair of counts makes the range known");
        };
        assert_eq!(
            twin,
            ENode::reduce(fold, Bound::Lit(5), Bound::Lit(5), body)
        );
        assert!(matches!(
            EmptyFold.apply(&eg, class, &twin),
            Some(RewriteAction::Create(ENode::Const(_)))
        ));
    }

    /// **A range that is not known is not unrolled** (L6): `PeelFold`,
    /// `HalveFold` and `EmptyFold` decline a fold whose bound is a uniform.
    #[test]
    fn an_unknown_range_is_not_unrolled() {
        let fold = Fold::new(Monoid::SUM, binder(0));
        let mut eg = Saturation::over(RuleSet::new(fold_rules()));
        let x = eg.add(ENode::Var(0));
        let j = eg.add(ENode::Var(fold.binder().var()));
        let body = eg.add(ENode::apply(OpKind::Mul, &[x, j]));
        let u = a_uniform(&mut eg);
        let node = ENode::reduce(fold, Bound::Lit(0), Bound::Value(u), body);
        let class = eg.add(node);
        for rule in [&PeelFold as &dyn Rewrite, &HalveFold, &EmptyFold] {
            assert!(
                rule.apply(&eg, class, &node).is_none(),
                "{} read a range the fold does not have",
                rule.name()
            );
        }
        assert!(KnownBound.apply(&eg, class, &node).is_none());
    }

    /// **The gate's `HalveFold` pin: a triangular fold halves with a bound the
    /// rule made.** `Σ_{i<4} Σ_{j<i} X·j` halves to a stride-2 sum over `i` of
    /// `Σ_{j<i} X·j + Σ_{j<i+1} X·j`: the shifted copy's bound is `i + 1`, an
    /// expression no caller wrote (L7 reaches the nested bound, and `add`'s
    /// backstop accepts it: it reads the outer index and nothing else). The
    /// graph saturates, and extraction gives back a closed term.
    ///
    /// The expected term, with no `ConstantFold` here, is the triangular fold
    /// itself. Only the peel's `i := 0` gives an inner bound that is a literal
    /// (the empty fold `0..0`); a rule-made `0 + 1` stays the arithmetic it is,
    /// so the other inner folds stay loops priced at the estimate and the
    /// unrolling is no cheaper. So the term holds an inner fold (slot 0) whose
    /// bound is a value reading the outer binder alone, or a literal where a
    /// peel did make one known.
    #[test]
    fn a_triangular_fold_halves_with_a_rule_made_bound() {
        let (a, root) = tri(4);
        let mut eg = Saturation::over(RuleSet::new(fold_rules()));
        let class = insert(&a, root, &mut eg, Vocabulary::Runtime).expect("a fold inserts");
        let outer = *eg.nodes(class).next().expect("a fold");
        let i = eg.add(ENode::Var(binder(1).var()));

        let Some(RewriteAction::Plan(plan)) = HalveFold.apply(&eg, class, &outer) else {
            panic!("a fold over 0..4 halves");
        };
        let inner = Fold::new(Monoid::SUM, binder(0));
        let bound = plan.nodes.iter().find_map(|node| match node {
            HeadNode::Reduce {
                fold,
                hi: Bound::Value(HeadRef::Plan(k)),
                ..
            } if *fold == inner => Some(*k),
            _ => None,
        });
        let Some(k) = bound else {
            panic!("the shifted inner fold's bound is a planned value: {plan:?}");
        };
        let HeadNode::Op {
            op: OpKind::Add,
            children,
        } = &plan.nodes[k as usize]
        else {
            panic!("the bound is the outer index plus the stride: {plan:?}");
        };
        let [HeadRef::Class(index), HeadRef::Plan(s)] = children[..] else {
            panic!("the bound adds a planned stride to a class: {plan:?}");
        };
        assert_eq!(eg.find(index), eg.find(i), "the class of the outer index");
        assert!(
            matches!(plan.nodes[s as usize], HeadNode::Const(Lit::Real(stride)) if stride == 1.0),
            "the stride is 1: {plan:?}"
        );

        SaturationConfig::compatibility(50).run(&mut eg);
        let (out, out_root, _) = extract(&eg, class, &CostModel::latency_prior());

        let variance = compute_arena_variance(&out);
        let outer_index = Variance::from_var(binder(1).var());
        let inner_bounds = nested_bounds(&out, out_root, inner);
        assert!(
            !inner_bounds.is_empty(),
            "the extracted term holds an inner fold: {}",
            out.display(out_root)
        );
        for hi in inner_bounds {
            match hi {
                Bound::Lit(_) => {}
                Bound::Value(bound) => assert!(
                    variance[bound.0 as usize].without(outer_index).is_const(),
                    "an inner bound reads the outer index and nothing else: {}",
                    out.display(out_root)
                ),
            }
        }
        assert_eq!(
            out.free_index(out_root),
            None,
            "the term is closed: every index it reads is bound"
        );
    }

    /// What `over_a_joined_bound` hands back.
    struct Joined {
        eg: Saturation,
        /// The outer fold's class and its node.
        outer: (EClassId, ENode),
        /// The class the two spellings were unioned into.
        bound: EClassId,
    }

    /// `Σ_{i<n} Σ_{j<O} X·j` by hand, where `O` is the class two spellings were
    /// unioned into, as an algebraic rule would: `spell` builds them from `X`,
    /// `Y`, the outer index and `1`. The union is the one thing here no rule
    /// did, and it is what makes the bound class's fact (the meet) smaller than
    /// its first member's variance.
    fn over_a_joined_bound(
        rules: Vec<Box<dyn Rewrite>>,
        n: u64,
        spell: impl FnOnce(&mut Saturation, [EClassId; 4]) -> [EClassId; 2],
    ) -> Joined {
        let mut eg = Saturation::over(RuleSet::new(rules));
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        let i = eg.add(ENode::Var(binder(1).var()));
        let one = eg.add(ENode::constant(1.0));
        let [first, second] = spell(&mut eg, [x, y, i, one]);
        let bound = eg.union(first, second);
        eg.rebuild();
        let j = eg.add(ENode::Var(binder(0).var()));
        let body = eg.add(ENode::apply(OpKind::Mul, &[x, j]));
        let inner = eg.add(ENode::reduce(
            Fold::new(Monoid::SUM, binder(0)),
            Bound::Lit(0),
            Bound::Value(bound),
            body,
        ));
        let class = eg.add(ENode::reduce(
            Fold::new(Monoid::SUM, binder(1)),
            Bound::Lit(0),
            Bound::Lit(n),
            inner,
        ));
        let node = *eg.nodes(class).next().expect("a fold");
        Joined {
            eg,
            outer: (class, node),
            bound,
        }
    }

    /// The fold rules, less every one but `rule`.
    fn alone(rule: &dyn Rewrite) -> Vec<Box<dyn Rewrite>> {
        fold_rules()
            .into_iter()
            .filter(|each| each.name() == rule.name())
            .collect()
    }

    /// The two minimal cases of L7 over a bound: `HalveFold` at `n = 4` and
    /// `PeelFold` at `n = 1`, each the only rule of its graph.
    const MINIMAL: [(&dyn Rewrite, u64); 2] = [(&HalveFold, 4), (&PeelFold, 1)];

    /// `((i + X) + 1) − X` beside `i + 1`, the first older than the second: the
    /// class's first member reads `X`, and the class does not (it denotes
    /// `i + 1`, an equation the algebra derives).
    fn a_varying_spelling_beside_a_lawful_one(
        eg: &mut Saturation,
        [x, _, i, one]: [EClassId; 4],
    ) -> [EClassId; 2] {
        let ix = eg.add(ENode::apply(OpKind::Add, &[i, x]));
        let ix1 = eg.add(ENode::apply(OpKind::Add, &[ix, one]));
        let varying = eg.add(ENode::apply(OpKind::Sub, &[ix1, x]));
        let lawful = eg.add(ENode::apply(OpKind::Add, &[i, one]));
        [varying, lawful]
    }

    /// `i + X` beside `i + Y`: each member reads a coordinate and the class
    /// reads neither (their meet is the outer index), so no member keeps L3.
    fn two_spellings_that_each_read_a_coordinate(
        eg: &mut Saturation,
        [x, y, i, _]: [EClassId; 4],
    ) -> [EClassId; 2] {
        let by_x = eg.add(ENode::apply(OpKind::Add, &[i, x]));
        let by_y = eg.add(ENode::apply(OpKind::Add, &[i, y]));
        [by_x, by_y]
    }

    /// **L7 rebuilds a nested bound through a member that keeps L3** (L7 of
    /// docs/language/2026-10-20-a-bound-is-a-value.md). A bound class's fact is
    /// the meet over its members, so its first member can read a coordinate the
    /// class does not: here `((i + X) + 1) − X` first and `i + 1` second.
    /// Rebuilt through the first, the bound reads `X` and `EGraph::add` panics
    /// inside saturation with `BoundVaries`. `HalveFold` at `n = 4` and
    /// `PeelFold` at `n = 1` are the two minimal cases, each alone in its rule
    /// set: both plan the bound through `i + 1`, and the graph takes the plan.
    #[test]
    fn a_nested_bound_is_rebuilt_through_the_member_that_keeps_l3() {
        for (rule, n) in MINIMAL {
            let mut f = over_a_joined_bound(alone(rule), n, a_varying_spelling_beside_a_lawful_one);
            let (class, outer) = f.outer;
            assert!(
                matches!(
                    f.eg.nodes(f.bound).next(),
                    Some(ENode::Op {
                        op: OpKind::Sub,
                        ..
                    })
                ),
                "the premise: the bound class's first member is the varying spelling"
            );
            assert_eq!(
                f.eg.variance(f.bound),
                Variance::from_var(binder(1).var()),
                "the premise: the class's fact is the outer index alone"
            );
            let Some(RewriteAction::Plan(plan)) = rule.apply(&f.eg, class, &outer) else {
                panic!(
                    "{} at n = {n}: a member of the bound keeps L3, so it plans",
                    rule.name()
                );
            };
            assert!(
                !plan.nodes.iter().any(|node| matches!(
                    node,
                    HeadNode::Op {
                        op: OpKind::Sub,
                        ..
                    }
                )),
                "n = {n}: the varying spelling was rebuilt: {plan:?}"
            );

            // The graph takes the plan: this is the call that panicked.
            SaturationConfig::compatibility(1).run(&mut f.eg);
            assert_eq!(
                value_bounded_inner_folds(&f.eg),
                2,
                "{} at n = {n}: the graph took the plan, so the nested fold was rebuilt \
                 (a plan through the varying spelling is declined, and builds none)",
                rule.name()
            );
            let rebuilt: Vec<_> = f
                .eg
                .class_ids()
                .flat_map(|class| f.eg.nodes(class).filter_map(ENode::as_reduce))
                .filter_map(|(_, _, hi, _)| match hi {
                    Bound::Value(bound) if f.eg.find(bound) != f.eg.find(f.bound) => Some(bound),
                    _ => None,
                })
                .collect();
            assert!(
                !rebuilt.is_empty()
                    && rebuilt
                        .iter()
                        .all(|&bound| f.eg.nodes(bound).all(|node| !matches!(
                            node,
                            ENode::Op {
                                op: OpKind::Sub,
                                ..
                            }
                        ))),
                "{} at n = {n}: the rebuilt bound is spelled `i + k`, not through `Sub`",
                rule.name()
            );
        }
    }

    /// `Σ_{k<1} i` beside `i · 1`, the fold the older: the same value (the
    /// runtime rules derive the first equation), and both keep L3.
    fn a_fold_beside_a_lawful_spelling(
        eg: &mut Saturation,
        [_, _, i, one]: [EClassId; 4],
    ) -> [EClassId; 2] {
        let fold = eg.add(ENode::reduce(
            Fold::new(Monoid::SUM, binder(2)),
            Bound::Lit(0),
            Bound::Lit(1),
            i,
        ));
        let times_one = eg.add(ENode::apply(OpKind::Mul, &[i, one]));
        [fold, times_one]
    }

    /// **A fold in a bound's class is not a member the walk reads** (L7).
    /// Saturation can put a fold into a bound's class (the runtime rules derive
    /// `Σ_{k<1} i = i`), and a fold under a bound would need the law lifted by
    /// its own binder, so the walk reads the next member that keeps L3: here
    /// `i · 1`. The plan holds the outer fold and the nested fold's shifted
    /// copy, and no fold for the bound.
    #[test]
    fn a_fold_in_a_bound_class_is_skipped_for_the_next_member() {
        let mut f = over_a_joined_bound(alone(&HalveFold), 4, a_fold_beside_a_lawful_spelling);
        let (class, outer) = f.outer;
        assert!(
            matches!(f.eg.nodes(f.bound).next(), Some(ENode::Reduce { .. })),
            "the premise: the bound class's first member is the fold"
        );
        let first = f.eg.nodes(f.bound).next().expect("a member");
        let reads = first.variance(|key| f.eg.ref_variance(key), |child| f.eg.variance(child));
        assert!(
            Fold::new(Monoid::SUM, binder(0)).bound_may_read(reads),
            "the premise: and the fold keeps L3 itself, so only the skip passes it over"
        );
        let Some(RewriteAction::Plan(plan)) = HalveFold.apply(&f.eg, class, &outer) else {
            panic!("the bound class has a member that is no fold and keeps L3, so it plans");
        };
        let folds = plan
            .nodes
            .iter()
            .filter(|node| matches!(node, HeadNode::Reduce { .. }))
            .count();
        assert_eq!(
            folds, 2,
            "the bound was rebuilt through `i · 1`, not the fold: {plan:?}"
        );

        SaturationConfig::compatibility(1).run(&mut f.eg);
        assert_eq!(
            value_bounded_inner_folds(&f.eg),
            2,
            "the graph took the plan and built the shifted nested fold"
        );
    }

    /// **A nested bound with no member that keeps L3 declines** (L7): the rule
    /// builds nothing rather than a bound `EGraph::add` would refuse. The
    /// class holds `i + X` and `i + Y`; each member reads a coordinate, the
    /// class's fact reads neither, and the fact is what the fold was admitted
    /// by. Declining costs completeness, never soundness.
    #[test]
    fn a_nested_bound_with_no_member_that_keeps_l3_declines() {
        for (rule, n) in MINIMAL {
            let mut f =
                over_a_joined_bound(alone(rule), n, two_spellings_that_each_read_a_coordinate);
            let (class, outer) = f.outer;
            assert_eq!(
                f.eg.variance(f.bound),
                Variance::from_var(binder(1).var()),
                "the premise: the class's fact is the outer index alone"
            );
            assert!(
                rule.apply(&f.eg, class, &outer).is_none(),
                "{} at n = {n} plans a bound no member of the class keeps",
                rule.name()
            );
            // And a whole run over it applies nothing and does not panic.
            SaturationConfig::compatibility(2).run(&mut f.eg);
        }
    }

    /// **The hole L7 leaves is loud.** Extraction picks one member per class by
    /// cost and does not know a bound's pick must keep L3, so a bound class
    /// with no member that does (here `i + X` and `i + Y`) is materialized
    /// through one that reads a coordinate, and `ExprArena::push_reduce` refuses
    /// the fold with `BoundVaries`. Nothing is silent, and it is left so:
    /// constraining a bound's pick would trade this failure for another, since
    /// a class read as a bound with no member that keeps L3 would never settle
    /// and every other user of the class would fail with it.
    #[test]
    #[should_panic(expected = "BoundVaries")]
    fn extracting_a_bound_class_with_no_member_that_keeps_l3_is_refused_by_name() {
        let f = over_a_joined_bound(alloc::vec![], 4, two_spellings_that_each_read_a_coordinate);
        let (class, _) = f.outer;
        let _refused = extract(&f.eg, class, &CostModel::latency_prior());
    }

    /// `i²` beside `((i · i) + X) − X`, the same value (an equation the algebra
    /// derives): the first is lawful and the dearer, the second reads `X` and
    /// is the cheaper.
    fn a_lawful_spelling_dearer_than_one_that_reads_a_coordinate(
        eg: &mut Saturation,
        [x, _, i, _]: [EClassId; 4],
    ) -> [EClassId; 2] {
        let two = eg.add(ENode::constant(2.0));
        let squared = eg.add(ENode::apply(OpKind::Pow, &[i, two]));
        let ii = eg.add(ENode::apply(OpKind::Mul, &[i, i]));
        let iix = eg.add(ENode::apply(OpKind::Add, &[ii, x]));
        let spelled = eg.add(ENode::apply(OpKind::Sub, &[iix, x]));
        [squared, spelled]
    }

    /// **The hole is wider than a class with no lawful member.** The pick is by
    /// cost, so a class that holds a lawful member (`i²`) is materialized
    /// through a cheaper one that reads a coordinate (`((i · i) + X) − X`), and
    /// the fold is refused with `BoundVaries` all the same. The class's fact is
    /// the outer index alone, which is why the fold was admitted. Saturation
    /// can leave the cheap spelling standing or rewrite it away; this graph has
    /// no rules, so nothing does. The cheaper spelling carries an extra
    /// subtraction to cancel the coordinate, which no kernel or profile shows
    /// winning, so the pick is left unconstrained and the failure loud.
    #[test]
    #[should_panic(expected = "BoundVaries")]
    fn extracting_a_bound_class_whose_cheapest_member_reads_a_coordinate_is_refused_by_name() {
        let f = over_a_joined_bound(
            alloc::vec![],
            3,
            a_lawful_spelling_dearer_than_one_that_reads_a_coordinate,
        );
        assert_eq!(
            f.eg.variance(f.bound),
            Variance::from_var(binder(1).var()),
            "the premise: the class's fact is the outer index alone"
        );
        assert!(
            has_a_member_that_keeps_l3(&f.eg, f.bound),
            "the premise: a member of the class keeps L3"
        );
        let (class, _) = f.outer;
        let _refused = extract(&f.eg, class, &CostModel::latency_prior());
    }

    /// `Σ_{i<8} (junk + Σ_{j<bound} X·j)`: the nested fold's bound the index
    /// itself (what `Kernel::over_values` builds) or the index plus one (bound
    /// arithmetic), and `junk = ((i + X) + 1) − X` before or after the nested
    /// fold. The runtime rules derive `junk = i + 1` and union it with the
    /// bound's own `i + 1`, which puts a coordinate-reading member first in
    /// the bound's class.
    fn a_triangular_sum_beside(bound: TriangularBound, junk: JunkAt) -> (ExprArena, ExprId) {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let i = a.push_var(binder(1).var());
        let j = a.push_var(binder(0).var());
        let one = a.push_const(1.0);
        let ix = a.push_binary(OpKind::Add, i, x);
        let ix1 = a.push_binary(OpKind::Add, ix, one);
        let varying = a.push_binary(OpKind::Sub, ix1, x);
        let hi = match bound {
            TriangularBound::TheIndex => i,
            TriangularBound::TheIndexPlusOne => a.push_binary(OpKind::Add, i, one),
        };
        let body = a.push_binary(OpKind::Mul, x, j);
        let nested = a.push_reduce(
            Fold::new(Monoid::SUM, binder(0)),
            Bound::Lit(0),
            Bound::Value(hi),
            body,
        );
        let sum = match junk {
            JunkAt::Before => a.push_binary(OpKind::Add, varying, nested),
            JunkAt::After => a.push_binary(OpKind::Add, nested, varying),
        };
        let outer = a.push_reduce(
            Fold::new(Monoid::SUM, binder(1)),
            Bound::Lit(0),
            Bound::Lit(8),
            sum,
        );
        (a, outer)
    }

    #[derive(Clone, Copy, Debug)]
    enum TriangularBound {
        TheIndex,
        TheIndexPlusOne,
    }

    #[derive(Clone, Copy, Debug)]
    enum JunkAt {
        Before,
        After,
    }

    /// **The runtime rules saturate a triangular fold that has a coordinate-
    /// reading spelling in its bound class, and extract a lawful term** (L7). No
    /// hand-made union: the production rules derive `((i + X) + 1) − X = i + 1`,
    /// the older spelling becomes the class's first member, and `HalveFold`'s
    /// second level substituted `i := i + 2` into it, which panicked with
    /// `BoundVaries`. Every extracted bound reads the outer index and nothing
    /// else, and the term is closed.
    #[test]
    fn the_runtime_rules_saturate_a_triangular_fold_with_a_varying_spelling() {
        let inner = Fold::new(Monoid::SUM, binder(0));
        let outer_index = Variance::from_var(binder(1).var());
        for bound in [TriangularBound::TheIndex, TriangularBound::TheIndexPlusOne] {
            for junk in [JunkAt::Before, JunkAt::After] {
                let (a, root) = a_triangular_sum_beside(bound, junk);
                let mut eg = Saturation::over(RuleSet::runtime());
                let class = insert(&a, root, &mut eg, Vocabulary::Runtime).expect("inserts");
                SaturationConfig::compatibility(12).run(&mut eg);
                let made: usize = eg
                    .class_ids()
                    .flat_map(|class| eg.nodes(class).filter_map(ENode::as_reduce))
                    .filter(|&(fold, _, hi, _)| fold == inner && matches!(hi, Bound::Value(_)))
                    .count();
                // The fold the kernel holds, its first shifted copy, and the
                // second level's copies, whose rebuild was the panic.
                assert!(
                    made > 3,
                    "{bound:?} {junk:?}: the rules halved through the nested bound ({made} folds)"
                );

                let (out, out_root, _) = extract(&eg, class, &CostModel::latency_prior());
                let variance = compute_arena_variance(&out);
                for hi in nested_bounds(&out, out_root, inner) {
                    if let Bound::Value(bound_term) = hi {
                        assert!(
                            variance[bound_term.0 as usize]
                                .without(outer_index)
                                .is_const(),
                            "{bound:?} {junk:?}: an extracted bound reads more than the outer \
                             index: {}",
                            out.display(out_root)
                        );
                    }
                }
                assert_eq!(out.free_index(out_root), None, "{bound:?} {junk:?}: closed");
            }
        }
    }

    /// Where a rebuilt bound lands, and how that class went stale.
    struct Stale {
        /// The nested fold's value bound.
        bound: EClassId,
        /// The class the rule's rebuilt bound hash-conses into: its fact reads
        /// `X` (it was seeded before `X` was shown not to matter, and a fact is
        /// never repaired upward), and one of its members does not.
        landing: EClassId,
    }

    /// `P = ((i + X) − X) + 1`, then `(i + X) − X` joined to `i` and `P` to
    /// `((i + X) + 1) − X`, both equations the algebra derives. The second puts
    /// `P` on the rebuild's worklist, which re-keys its node as `i + 1`; its
    /// fact stays the `{X, i}` it was seeded with. The nested bound is the
    /// index itself, and a halved copy's bound `i + 1` lands in `P`.
    fn stale_index_plus_one(eg: &mut Saturation) -> Stale {
        let x = eg.add(ENode::Var(0));
        let i = eg.add(ENode::Var(binder(1).var()));
        let one = eg.add(ENode::constant(1.0));
        let ix = eg.add(ENode::apply(OpKind::Add, &[i, x]));
        let k = eg.add(ENode::apply(OpKind::Sub, &[ix, x]));
        let p = eg.add(ENode::apply(OpKind::Add, &[k, one]));
        let ix1 = eg.add(ENode::apply(OpKind::Add, &[ix, one]));
        let q = eg.add(ENode::apply(OpKind::Sub, &[ix1, x]));
        eg.union(k, i);
        eg.rebuild();
        eg.union(p, q);
        eg.rebuild();
        Stale {
            bound: i,
            landing: p,
        }
    }

    /// [`stale_index_plus_one`] with `P` holding both spellings of the sum
    /// before `K = i` is known, so neither `i + 1` nor `1 + i` is born fresh
    /// afterwards: the commutativity the runtime set has cannot repair `P` by
    /// minting a lawful twin of it.
    fn stale_either_order(eg: &mut Saturation) -> Stale {
        let x = eg.add(ENode::Var(0));
        let i = eg.add(ENode::Var(binder(1).var()));
        let one = eg.add(ENode::constant(1.0));
        let ix = eg.add(ENode::apply(OpKind::Add, &[i, x]));
        let k = eg.add(ENode::apply(OpKind::Sub, &[ix, x]));
        let p = eg.add(ENode::apply(OpKind::Add, &[k, one]));
        let p_commuted = eg.add(ENode::apply(OpKind::Add, &[one, k]));
        eg.union(p, p_commuted);
        eg.rebuild();
        let ix1 = eg.add(ENode::apply(OpKind::Add, &[ix, one]));
        let q = eg.add(ENode::apply(OpKind::Sub, &[ix1, x]));
        eg.union(k, i);
        eg.rebuild();
        eg.union(p, q);
        eg.rebuild();
        Stale {
            bound: i,
            landing: p,
        }
    }

    /// `P = (X − X) + 1`, then `X − X` joined to `0`: `P`'s node is `0 + 1` and
    /// its fact still reads `X`. The nested bound is `i + 1`, and a peeled
    /// copy's bound, `0 + 1`, lands in `P`.
    fn stale_zero_plus_one(eg: &mut Saturation) -> Stale {
        let x = eg.add(ENode::Var(0));
        let i = eg.add(ENode::Var(binder(1).var()));
        let one = eg.add(ENode::constant(1.0));
        let x_minus_x = eg.add(ENode::apply(OpKind::Sub, &[x, x]));
        let p = eg.add(ENode::apply(OpKind::Add, &[x_minus_x, one]));
        let zero = eg.add(ENode::constant(0.0));
        eg.union(x_minus_x, zero);
        eg.rebuild();
        let bound = eg.add(ENode::apply(OpKind::Add, &[i, one]));
        Stale { bound, landing: p }
    }

    /// The bound [`stale_index_plus_one`] and [`stale_zero_plus_one`] have
    /// before they go stale, in a graph with no `P` at all: the control.
    fn fresh_index(eg: &mut Saturation) -> Stale {
        let i = eg.add(ENode::Var(binder(1).var()));
        Stale {
            bound: i,
            landing: i,
        }
    }

    fn fresh_index_plus_one(eg: &mut Saturation) -> Stale {
        let i = eg.add(ENode::Var(binder(1).var()));
        let one = eg.add(ENode::constant(1.0));
        let bound = eg.add(ENode::apply(OpKind::Add, &[i, one]));
        Stale {
            bound,
            landing: bound,
        }
    }

    /// `Σ_{i<n} Σ_{j<bound} X·j`, the outer fold's class.
    fn nested_under(eg: &mut Saturation, bound: EClassId, n: u64) -> EClassId {
        let x = eg.add(ENode::Var(0));
        let j = eg.add(ENode::Var(binder(0).var()));
        let body = eg.add(ENode::apply(OpKind::Mul, &[x, j]));
        let inner = eg.add(ENode::reduce(
            Fold::new(Monoid::SUM, binder(0)),
            Bound::Lit(0),
            Bound::Value(bound),
            body,
        ));
        eg.add(ENode::reduce(
            Fold::new(Monoid::SUM, binder(1)),
            Bound::Lit(0),
            Bound::Lit(n),
            inner,
        ))
    }

    /// How many folds over the nested fold's header the graph holds with a
    /// value upper bound.
    fn value_bounded_inner_folds(eg: &Saturation) -> usize {
        let inner = Fold::new(Monoid::SUM, binder(0));
        eg.class_ids()
            .flat_map(|class| eg.nodes(class).filter_map(ENode::as_reduce))
            .filter(|&(fold, _, hi, _)| fold == inner && matches!(hi, Bound::Value(_)))
            .count()
    }

    /// Whether every value bound of every fold the graph holds keeps L3 on its
    /// class's own fact: the backstop's condition, asked of the whole graph.
    fn every_bound_keeps_l3(eg: &Saturation) -> bool {
        eg.class_ids()
            .flat_map(|class| eg.nodes(class).filter_map(ENode::as_reduce))
            .all(|(fold, lo, hi, _)| {
                [lo, hi].into_iter().all(|bound| match bound {
                    Bound::Lit(_) => true,
                    Bound::Value(class) => fold.bound_may_read(eg.variance(class)),
                })
            })
    }

    /// Whether some member of `class` keeps L3 for the nested fold's upper
    /// bound, on its own variance over its children's facts.
    fn has_a_member_that_keeps_l3(eg: &Saturation, class: EClassId) -> bool {
        let inner = Fold::new(Monoid::SUM, binder(0));
        eg.nodes(class).any(|node| {
            let reads = node.variance(|key| eg.ref_variance(key), |child| eg.variance(child));
            inner.bound_may_read(reads)
        })
    }

    /// `i * 2` over [`stale_index_plus_one`]'s bound (the index), with that
    /// recipe's stale class `P` beside it: a halved copy's bound is
    /// `(i + 1) * 2`, a class the plan mints over `i + 1`, which lands in `P`.
    fn stale_double_index(eg: &mut Saturation) -> Stale {
        let stale = stale_index_plus_one(eg);
        let two = eg.add(ENode::constant(2.0));
        let bound = eg.add(ENode::apply(OpKind::Mul, &[stale.bound, two]));
        Stale {
            bound,
            landing: stale.landing,
        }
    }

    /// [`stale_double_index`]'s bound with no `P` at all: the control.
    fn fresh_double_index(eg: &mut Saturation) -> Stale {
        let i = eg.add(ENode::Var(binder(1).var()));
        let two = eg.add(ENode::constant(2.0));
        let bound = eg.add(ENode::apply(OpKind::Mul, &[i, two]));
        Stale {
            bound,
            landing: bound,
        }
    }

    /// **A rebuilt bound whose class fact fails L3 is declined, not built** (L7
    /// of docs/language/2026-10-20-a-bound-is-a-value.md). The rule picked a
    /// member of the bound's class that keeps L3 and rebuilt it, and the term
    /// it rebuilt is lawful (`i + 1`, `0 + 1`, `(i + 1) * 2`); but a class's
    /// fact is seeded at birth and never repaired upward, so the term can land
    /// in a class that already holds it, seeded wider than the term, or be
    /// minted over a child that did. `EGraph::add` reads the fact and panicked
    /// with `BoundVaries`. The graph now refuses the node to the plan builder,
    /// which declines the rewrite: nothing is unioned and no fold is built.
    ///
    /// Three cases, each the only rule of its graph: `HalveFold` at `n = 4`
    /// with the bound landing in `P`, `PeelFold` at `n = 1` likewise, and
    /// `HalveFold` with a bound the plan mints over a child that landed in `P`
    /// (the `(i + 1) * 2`). Each has a control with no stale class, where the
    /// same rule builds the nested fold, so the decline is the staleness and
    /// not a rule that never fires. What the run shows: the control's rule
    /// unions its plan and builds the nested fold; the stale graph's unions
    /// nothing (the plan was declined) and builds no fold, and every bound it
    /// holds keeps L3.
    #[test]
    fn a_rebuilt_bound_that_lands_in_a_stale_class_is_declined() {
        type Recipe = fn(&mut Saturation) -> Stale;
        let cases: [(&dyn Rewrite, u64, Recipe, Recipe); 3] = [
            (&HalveFold, 4, stale_index_plus_one, fresh_index),
            (&PeelFold, 1, stale_zero_plus_one, fresh_index_plus_one),
            (&HalveFold, 4, stale_double_index, fresh_double_index),
        ];
        for (rule, n, stale, control) in cases {
            let mut eg = Saturation::over(RuleSet::new(alone(rule)));
            let made = control(&mut eg);
            nested_under(&mut eg, made.bound, n);
            let ran = SaturationConfig::compatibility(1).run(&mut eg);
            let built = value_bounded_inner_folds(&eg);
            assert_eq!(
                built,
                2,
                "{} at n = {n}: the control: with no stale class the rule rebuilds the nested fold",
                rule.name()
            );
            assert!(
                ran.total_unions > 0,
                "{} at n = {n}: the control: the rule unions its plan",
                rule.name()
            );

            let mut eg = Saturation::over(RuleSet::new(alone(rule)));
            let made = stale(&mut eg);
            assert!(
                eg.variance(made.landing).depends_on_x(),
                "{} at n = {n}, the premise: the class the bound lands in reads X",
                rule.name()
            );
            assert!(
                has_a_member_that_keeps_l3(&eg, made.landing),
                "{} at n = {n}, the premise: and a member of it does not",
                rule.name()
            );
            let outer = nested_under(&mut eg, made.bound, n);
            let node = *eg.nodes(outer).next().expect("a fold");
            assert!(
                matches!(rule.apply(&eg, outer, &node), Some(RewriteAction::Plan(_))),
                "{} at n = {n}: a member of the bound's class keeps L3, so the rule plans",
                rule.name()
            );

            // The graph takes the plan: this is the call that panicked.
            let ran = SaturationConfig::compatibility(3).run(&mut eg);
            assert_eq!(
                ran.total_unions,
                0,
                "{} at n = {n}: the plan was declined, so nothing was unioned",
                rule.name()
            );
            assert_eq!(
                value_bounded_inner_folds(&eg),
                1,
                "{} at n = {n}: the declined plan built no fold",
                rule.name()
            );
            assert!(
                every_bound_keeps_l3(&eg),
                "{} at n = {n}: a bound in the graph varies",
                rule.name()
            );
        }
    }

    /// **The runtime rules saturate a graph holding a stale class a rebuilt
    /// bound lands in** (L7). The full set, commutativity and constant folding
    /// among it, repairs some staleness by minting a lawful twin of the stale
    /// class (`i + 1` beside `P`), so [`stale_either_order`] holds both
    /// spellings of `P`'s sum first and no twin is born: a halved copy's bound,
    /// `i + 1`, lands in `P` whatever the extent (`n` = 2, 4, 8), and the graph
    /// declines it at each halving the outer fold offers. A control over the
    /// bare index, with no `P`, halves through the same bound, so it holds more
    /// value-bounded inner folds than the stale graph does, and `P`'s fact still
    /// reads `X` afterwards. The first iteration is where it panicked; six leave
    /// room for the control to halve its way down, at a cost the full set's
    /// growth keeps small. The extracted term is closed, and every bound reads
    /// the outer index and nothing else.
    #[test]
    fn the_runtime_rules_saturate_a_graph_with_a_stale_bound_class() {
        let inner = Fold::new(Monoid::SUM, binder(0));
        let outer_index = Variance::from_var(binder(1).var());
        for n in [2, 4, 8] {
            let mut control = Saturation::over(RuleSet::runtime());
            let fresh = fresh_index(&mut control);
            nested_under(&mut control, fresh.bound, n);
            SaturationConfig::compatibility(6).run(&mut control);

            let mut eg = Saturation::over(RuleSet::runtime());
            let stale = stale_either_order(&mut eg);
            assert!(
                eg.variance(stale.landing).depends_on_x()
                    && has_a_member_that_keeps_l3(&eg, stale.landing),
                "n = {n}, the premise: the class reads X and a member of it does not"
            );
            let one = eg.add(ENode::constant(1.0));
            let shifted = eg.add(ENode::apply(OpKind::Add, &[stale.bound, one]));
            assert_eq!(
                eg.find(shifted),
                eg.find(stale.landing),
                "n = {n}, the premise: a halved copy's bound `i + 1` lands in P"
            );
            let class = nested_under(&mut eg, stale.bound, n);
            SaturationConfig::compatibility(6).run(&mut eg);
            assert!(
                value_bounded_inner_folds(&eg) < value_bounded_inner_folds(&control),
                "n = {n}: the halving was declined, so fewer inner folds than the control's \
                 ({} against {})",
                value_bounded_inner_folds(&eg),
                value_bounded_inner_folds(&control)
            );
            assert!(
                eg.variance(stale.landing).depends_on_x(),
                "n = {n}: P's fact still reads X: nothing repaired it"
            );
            assert!(
                every_bound_keeps_l3(&eg),
                "n = {n}: a bound in the graph varies"
            );

            let (out, out_root, _) = extract(&eg, class, &CostModel::latency_prior());
            let variance = compute_arena_variance(&out);
            let bounds = nested_bounds(&out, out_root, inner);
            assert!(
                !bounds.is_empty(),
                "n = {n}: the extracted term holds an inner fold: {}",
                out.display(out_root)
            );
            for hi in bounds {
                if let Bound::Value(bound) = hi {
                    assert!(
                        variance[bound.0 as usize].without(outer_index).is_const(),
                        "n = {n}: an extracted bound reads more than the outer index: {}",
                        out.display(out_root)
                    );
                }
            }
            assert_eq!(out.free_index(out_root), None, "n = {n}: closed");
        }
    }

    /// `KnownBound` is appended last, so every rule before it keeps its
    /// position in the runtime set, and with it the `RuleIdx` a recorded
    /// application names.
    #[test]
    fn known_bound_is_the_last_fold_rule() {
        let rules = fold_rules();
        assert_eq!(rules.last().expect("fold rules").name(), "known-bound");

        let mut before = crate::egraph::all_rules();
        before.extend([
            Box::new(HalveFold) as Box<dyn Rewrite>,
            Box::new(PeelFold),
            Box::new(EmptyFold),
            Box::new(FactorFold),
        ]);
        let before = RuleSet::new(before);
        let runtime = RuleSet::runtime();
        assert_eq!(runtime.len(), before.len() + 1);
        assert_eq!(&runtime.ids()[..before.len()], before.ids());
        assert_eq!(
            runtime.id_of(before.len()),
            Some(crate::egraph::RuleId::of(&KnownBound))
        );
    }
}

#[cfg(test)]
mod production_shape_tests {
    use super::*;
    use crate::egraph::extract::extract_dag;
    use crate::egraph::saturate::SaturationConfig;
    use crate::egraph::{CostFunction, CostModel, ENode, Vocabulary, insert};
    use crate::egraph::{RuleSet, Saturation};
    use pixelflow_ir::arena::{BufferDecl, BufferIdentity};
    use pixelflow_ir::{ExprArena, ExprNode, OpKind};

    fn binder(slot: u8) -> Binder {
        Binder::from_slot(slot).expect("a live binder")
    }

    /// What a loop costs [`LoopsCostMore`] beyond the latency prior: more than
    /// any unrolling these tests build.
    const LOOP_PENALTY: usize = 1 << 24;

    /// The latency prior with every `Reduce` priced out, so extraction takes
    /// an unrolled form whenever the class holds one.
    ///
    /// Under the plain prior this crate's table-reading fold is cheaper as a
    /// loop than unrolled (the unrolled form pays the gather and the multiply
    /// at every term), so whether the graph *can* unroll the fold cannot be
    /// asked of the production prior: its answer is the loop, which
    /// `production_keeps_the_loop_over_a_table_fold` pins. Pricing loops out
    /// is how extraction is made to report the graph's content. It mirrors
    /// `pixelflow-codegen/tests/halve_fold_jit.rs`'s device of the same name,
    /// which cannot share this one while `CostModel` has no loop price and no
    /// public materializer takes a `CostFunction`; they fold together when
    /// that is decided. See also docs/BACKLOG.md E6.
    struct LoopsCostMore(CostModel);

    impl CostFunction for LoopsCostMore {
        fn node_cost(&self, node: &ENode, parent: Option<OpKind>) -> usize {
            let own = self.0.node_cost(node, parent);
            match node {
                ENode::Reduce { .. } => own.saturating_add(LOOP_PENALTY),
                _ => own,
            }
        }
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
    /// Returns the saturated graph, the fold's class, and the saturation
    /// stats.
    fn saturated_table_fold() -> (Saturation, crate::egraph::EClassId, impl core::fmt::Debug) {
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
        let root = a.push_reduce(
            Fold::new(Monoid::SUM, binder(0)),
            Bound::Lit(0),
            Bound::Lit(40),
            body,
        );

        let mut eg = Saturation::over(RuleSet::new(fold_rules()));
        let class = insert(&a, root, &mut eg, Vocabulary::Runtime).expect("a fold inserts");
        let stats = SaturationConfig::compatibility(200).run(&mut eg);
        (eg, class, stats)
    }

    /// The saturated graph holds a `Reduce`-free form of the 40-term table
    /// fold: with loops priced out, extraction returns it. This pins the
    /// graph's content, not what production extracts.
    #[test]
    fn a_table_reading_fold_unrolls() {
        let (eg, class, stats) = saturated_table_fold();
        let dag = extract_dag(&eg, class, &LoopsCostMore(CostModel::latency_prior()));
        let (out, _out_root) = dag.to_arena(&eg);

        assert!(
            !has_fold(&out),
            "with loops priced out, the 40-term table fold must unroll \
             (saturation {stats:?}); extracted: {} nodes",
            out.len()
        );
        assert!(out.len() > 40, "and the unrolled form is 40 terms wide");
    }

    /// What production does with the same graph: under the latency prior the
    /// loop is the cheaper term, so the extractor keeps it. A prior change
    /// that makes production unroll table folds (the glyph's shape) fails
    /// here rather than moving every kernel's bytes unannounced.
    #[test]
    fn production_keeps_the_loop_over_a_table_fold() {
        let (eg, class, stats) = saturated_table_fold();
        let dag = extract_dag(&eg, class, &CostModel::latency_prior());
        let (out, _out_root) = dag.to_arena(&eg);

        assert!(
            has_fold(&out),
            "under the latency prior the loop is cheaper than its unrolling \
             (saturation {stats:?}); extracted: {} nodes",
            out.len()
        );
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
        let root = a.push_reduce(
            Fold::new(Monoid::SUM, binder(0)),
            Bound::Lit(0),
            Bound::Lit(40),
            body,
        );

        let mut eg = Saturation::over(RuleSet::new(fold_rules()));
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
