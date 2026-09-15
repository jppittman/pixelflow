//! Inserting a term into an e-graph.
//!
//! One function, over any [`Ir`]. It replaces two hand-rolled arena→e-graph
//! conversions (`EGraph::add_arena` and `runtime::arena_to_egraph`) that
//! existed for the *same* IR and disagreed about panicking, about which ops
//! were representable, and about whether unreachable nodes were inserted —
//! see docs/plans/2026-09-04-ir-as-a-trait.md §2.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use pixelflow_ir::{Children, Ir, Shape};

use super::graph::EGraph;
use super::node::{EClassId, ENode};
use super::ops::Vocabulary;

/// Why a term could not be inserted.
///
/// Declining is the only failure mode: a caller that meets one compiles the
/// term unoptimized, which is always available because optimization is never
/// required for correctness. The previous `add_arena` panicked on each of
/// these instead, so its callers had to pre-screen the term to avoid aborting
/// the build.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Declined {
    /// An op this [`Vocabulary`] may not hold.
    Op(pixelflow_ir::OpKind),
    /// A macro-parameter slot, under a [`Vocabulary`] that may not hold one.
    ///
    /// Only [`Vocabulary::Runtime`] declines it: a `Param` at bake time means
    /// a builder was never called, and the term reached compilation without
    /// being specialized. The macro tier holds params natively as
    /// [`ENode::Param`](super::node::ENode::Param), because an unbound slot
    /// is what a builder *is*.
    Param(u8),
    /// A kernel named by content. `passes::expand_refs` runs before saturation
    /// in every pipeline, so one here is a pipeline-order bug rather than a
    /// term the e-graph could learn to hold: a reference has no structure to
    /// rewrite, and inlining it inside saturation is a rule that does not
    /// exist yet (docs/plans/2026-09-09-composition-is-linking.md §3).
    Ref(pixelflow_ir::KernelKey),
}

/// Insert the subgraph reachable from `root` into `egraph`, returning the
/// e-class the root lands in.
///
/// Memoized on `I::Ref` (on top of the e-graph's own hash-consing by node
/// shape) so a DAG-shared term is walked once per node, not once per
/// reference. Iterative: term depth is unbounded in principle (`Dwrt`
/// chain-rule expansion, deep composition), so this must not blow the Rust
/// stack.
///
/// `Buffer` and `Uniform` leaves insert as themselves, carrying their
/// declarations, and under [`Vocabulary::Templates`] so does `Param`. Under
/// [`Vocabulary::Runtime`] a `Param` declines: by bake time a builder should
/// have substituted it, so one surviving is a term that was never
/// specialized. Which vocabulary may hold which leaf is the job `Vocabulary`
/// exists for, and saying it there is what stopped the macro tier from
/// smuggling params past this gate disguised as `Var`s. A `Ref` declines
/// under either, since its body is not in this term at all —
/// `passes::expand_refs` puts it there, and runs first.
///
/// **Reachable-only.** A term representation may hold nodes no longer reached
/// from `root` — an arena accumulates construction garbage — and inserting
/// those would spend `max_classes`, a budget dimension, on nodes the budget's
/// own `node_count` never counted.
pub fn insert<I: Ir>(
    term: &I,
    root: I::Ref,
    egraph: &mut EGraph,
    vocab: Vocabulary,
) -> Result<EClassId, Declined> {
    enum Task<R> {
        /// Resolve children first, then build this node.
        Visit(R),
        /// Children are on the result stack; pop and build.
        Complete(R),
    }

    let mut memo: BTreeMap<I::Ref, EClassId> = BTreeMap::new();
    let mut tasks: Vec<Task<I::Ref>> = alloc::vec![Task::Visit(root)];
    let mut built: Vec<EClassId> = Vec::new();

    while let Some(task) = tasks.pop() {
        match task {
            Task::Visit(r) => {
                if let Some(&class) = memo.get(&r) {
                    built.push(class);
                    continue;
                }
                let class = match term.project(r) {
                    Shape::Var(i) => egraph.add(ENode::Var(i)),
                    Shape::Const(v) => egraph.add(ENode::constant(v)),
                    Shape::Param(i) => match vocab {
                        Vocabulary::Templates => egraph.add(ENode::Param(i)),
                        Vocabulary::Runtime => return Err(Declined::Param(i)),
                    },
                    Shape::Ref(key) => return Err(Declined::Ref(key)),
                    Shape::Buffer(decl) => egraph.add(ENode::Buffer(decl)),
                    Shape::Uniform(decl) => egraph.add(ENode::Uniform(decl)),
                    Shape::Op(kind, children) => {
                        // Resolve before descending so an unrepresentable op
                        // is reported at the node that carries it.
                        if vocab.resolve(kind).is_none() {
                            return Err(Declined::Op(kind));
                        }
                        tasks.push(Task::Complete(r));
                        // Reverse, so children pop in operand order.
                        for i in (0..children.len()).rev() {
                            let child = children.get(i).expect("index < len");
                            tasks.push(Task::Visit(child));
                        }
                        continue;
                    }
                    // No vocabulary check: a fold is not an op, so there is
                    // no `Op` for a rule to name and nothing for a vocabulary
                    // to admit or refuse. It enters as itself, under either.
                    Shape::Reduce { body, .. } => {
                        tasks.push(Task::Complete(r));
                        tasks.push(Task::Visit(body));
                        continue;
                    }
                };
                memo.insert(r, class);
                built.push(class);
            }
            Task::Complete(r) => {
                if let Some(&class) = memo.get(&r) {
                    built.push(class);
                    continue;
                }
                let class = match term.project(r) {
                    Shape::Op(kind, children) => {
                        let op = vocab
                            .resolve(kind)
                            .expect("vocabulary already checked in Visit");
                        let start = built.len() - children.len();
                        let operands: Vec<EClassId> = built.drain(start..).collect();
                        egraph.add(ENode::Op {
                            op,
                            children: operands,
                        })
                    }
                    Shape::Reduce { fold, .. } => {
                        let body = built.pop().expect("the body was visited first");
                        egraph.add(ENode::Reduce { fold, body })
                    }
                    _ => unreachable!("Complete is scheduled only for compound shapes"),
                };
                memo.insert(r, class);
                built.push(class);
            }
        }
    }

    Ok(built.pop().expect("insert: root produced no e-class"))
}

/// Count the nodes reachable from `root` — the rough size measure saturation
/// budgets key on.
///
/// Shares [`insert`]'s reachability so a budget cannot be picked from one node
/// set while the graph is built from another.
pub fn reachable_count<I: Ir>(term: &I, root: I::Ref) -> usize {
    let mut seen: BTreeSet<I::Ref> = BTreeSet::new();
    let mut stack = alloc::vec![root];
    while let Some(r) = stack.pop() {
        if !seen.insert(r) {
            continue;
        }
        match term.project(r) {
            Shape::Op(_, children) => match children {
                Children::Many(s) => stack.extend_from_slice(s),
                other => {
                    for i in 0..other.len() {
                        stack.push(other.get(i).expect("index < len"));
                    }
                }
            },
            Shape::Reduce { body, .. } => stack.push(body),
            _ => {}
        }
    }
    seen.len()
}
