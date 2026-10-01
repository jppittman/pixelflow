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
    /// A kernel named by content, which the graph was not told is a unit.
    ///
    /// The runtime tier holds a reference as an opaque leaf when its unit
    /// walk has admitted it (`EGraph::admit_unit`): the unit is optimized by
    /// itself and linked after extraction
    /// (docs/plans/2026-09-25-the-language-is-kernel.md §4, O1). Anything
    /// else — the macro tier, a research tool, a reference no walk admitted —
    /// meets a name with no structure to rewrite and no variance to read,
    /// and declines it: inlining inside saturation is a rule that does not
    /// exist (docs/plans/2026-09-09-composition-is-linking.md §3).
    Ref(pixelflow_ir::KernelKey),
    /// A `Guard` — the hard lowering of an `If`
    /// (docs/plans/2026-09-12-emit-should-just-emit.md). Declined for a
    /// reason specific to this stage (G1), not a standing one: extraction
    /// has no price for choosing a `Guard` over the `If` it is equal to,
    /// so there is nothing yet for the e-graph to gain by holding one.
    /// `Guard`'s arms name kernels the same way a `Ref` does, and are
    /// unrepresentable as e-graph structure for the same reason: nothing
    /// here can rewrite inside a name. G3 is what gives extraction a price
    /// and this decline something to change.
    Guard,
    /// A `Write` — the store the lattice's folds wrap a kernel in
    /// (docs/plans/2026-09-16-collapse-is-a-fold.md §2.4). Declined for a
    /// standing reason: an effect is not a value, so no rule may rewrite
    /// it, and it is built by the legalize passes *after* extraction, so a
    /// term carrying one into saturation skipped the pipeline.
    Write,
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
/// smuggling params past this gate disguised as `Var`s. A `Ref` inserts as
/// an opaque [`ENode::Ref`] leaf under [`Vocabulary::Runtime`] when the graph
/// has admitted it as a unit (`EGraph::admit_unit`, which carries the
/// variance its body is not here to give), and declines otherwise.
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
                    Shape::Ref(key) => match (vocab, egraph.unit_variance(key)) {
                        (Vocabulary::Runtime, Some(variance)) => {
                            egraph.add(ENode::Ref { key, variance })
                        }
                        _ => return Err(Declined::Ref(key)),
                    },
                    Shape::Guard { .. } => return Err(Declined::Guard),
                    Shape::Write { .. } => return Err(Declined::Write),
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
            Shape::Guard { mask, .. } => stack.push(mask),
            Shape::Write { value, .. } => stack.push(value),
            _ => {}
        }
    }
    seen.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_ir::{ExprArena, ExprId, Kernel, KernelKey, KernelStore, OpKind, Variance};

    /// `Ref(named)·Y` over a fresh arena, the reference, and the key it names.
    fn named_times_y() -> (ExprArena, ExprId, ExprId, KernelKey) {
        let named = Kernel::x().mul(&Kernel::constant(5.5));
        let key = KernelStore::intern(&named);
        let mut arena = ExprArena::new();
        let reference = arena.push_ref(key);
        let y = arena.push_var(1);
        let root = arena.push_binary(OpKind::Mul, reference, y);
        (arena, root, reference, key)
    }

    /// A unit the graph admitted inserts as one opaque leaf, carrying the
    /// variance it was admitted with — so the product over it varies with X
    /// as well as Y, which nothing could have learned from the name.
    #[test]
    fn an_admitted_unit_is_a_leaf_carrying_its_variance() {
        let (arena, root, reference, key) = named_times_y();
        let mut eg = EGraph::new();
        eg.admit_unit(key, Variance::X);
        let product = insert(&arena, root, &mut eg, Vocabulary::Runtime).expect("admitted");
        let leaf = insert(&arena, reference, &mut eg, Vocabulary::Runtime).expect("admitted");
        assert!(
            eg.nodes(leaf).iter().any(|n| matches!(
                n,
                ENode::Ref { key: k, variance } if *k == key && *variance == Variance::X
            )),
            "the leaf names the unit and carries its variance"
        );
        assert_eq!(eg.variance(product), Variance::X.union(Variance::Y));
    }

    /// Admission is the runtime tier's: under the macro tier's vocabulary a
    /// reference still declines, admitted or not, as it does in a graph no
    /// one told about it.
    #[test]
    fn only_the_runtime_vocabulary_holds_a_unit() {
        let (arena, root, _, key) = named_times_y();
        let mut admitted = EGraph::new();
        admitted.admit_unit(key, Variance::X);
        assert_eq!(
            insert(&arena, root, &mut admitted, Vocabulary::Templates),
            Err(Declined::Ref(key))
        );
        let mut untold = EGraph::new();
        assert_eq!(
            insert(&arena, root, &mut untold, Vocabulary::Runtime),
            Err(Declined::Ref(key))
        );
    }

    /// A unit is optimized out of its context, which is sound only for a
    /// closed term: one reading a binder is refused at the door.
    #[test]
    #[should_panic(expected = "a unit must be")]
    fn an_open_unit_is_refused() {
        let key = KernelStore::intern(&Kernel::y().add(&Kernel::constant(0.75)));
        EGraph::new().admit_unit(key, Variance::Y.union(Variance::BINDERS));
    }
}
