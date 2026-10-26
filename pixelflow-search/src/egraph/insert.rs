//! Inserting a term into an e-graph.
//!
//! One function, over any [`Ir`]. It replaces two hand-rolled arena→e-graph
//! conversions (`EGraph::add_arena` and `runtime::arena_to_egraph`) that
//! existed for the *same* IR and disagreed about panicking, about which ops
//! were representable, and about whether unreachable nodes were inserted —
//! see docs/archive/plans/2026-09-04-ir-as-a-trait.md §2.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use pixelflow_ir::fold::{Bound, FoldHeader};
use pixelflow_ir::{Children, Ir, Lit, Shape};

use super::graph::EGraph;
use super::node::{EClassId, ENode};
use super::ops::Vocabulary;
use super::shift::{self, ShiftDirection};

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
    /// An op applied to a number of operands other than its arity.
    ///
    /// An arena's n-ary builder (`push_nary`) checks no arity, so a term can
    /// carry one; a node cannot, and building it would abort the build for a
    /// term the caller could have compiled unoptimized.
    Arity {
        /// The operation.
        op: pixelflow_ir::OpKind,
        /// How many operands the term gave it.
        operands: usize,
    },
    /// A shift whose count is not a `Const` holding a
    /// [`ShiftCount`](pixelflow_ir::ShiftCount).
    ///
    /// A count outside that range has a different answer on each target, and
    /// a count that is not a literal is not a parameter of the shift but an
    /// operand, which a shift node cannot hold. The term compiles
    /// unoptimized, and the emitter refuses it there: a literal that is not a
    /// `ShiftCount` as such, a count that is not a literal as "must be a
    /// Const".
    ShiftCount(ShiftDirection),
    /// An unbound scalar slot, under a [`Vocabulary`] that may not hold one.
    ///
    /// Only [`Vocabulary::Runtime`] declines it: the term then compiles as
    /// written, and the JIT emitter panics on the slot. [`Vocabulary::Templates`]
    /// still holds one as [`ENode::Param`]: nothing has deleted it
    /// (the-language-is-kernel D2).
    Param(u8),
    /// A kernel named by content, which the graph was not told is a unit.
    ///
    /// The runtime tier holds a reference as an opaque leaf when its unit
    /// walk has admitted it (`EGraph::admit_unit`): the unit is optimized by
    /// itself and linked after extraction
    /// (docs/language/2026-09-25-the-language-is-kernel.md §4, O1). Anything
    /// else — the macro tier, a research tool, a reference no walk admitted —
    /// meets a name with no structure to rewrite and no variance to read,
    /// and declines it: inlining inside saturation is a rule that does not
    /// exist (docs/language/2026-09-09-composition-is-linking.md §3).
    Ref(pixelflow_ir::KernelKey),
    /// A `Write` — the store the lattice's folds wrap a kernel in
    /// (docs/archive/plans/2026-09-16-collapse-is-a-fold.md §2.4). Declined for a
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
/// [`Vocabulary::Runtime`] a `Param` declines: the term then compiles as
/// written, and the JIT emitter panics on the slot. Which vocabulary may hold
/// which leaf is the
/// job `Vocabulary` exists for. A `Ref` inserts as an opaque [`ENode::Ref`]
/// leaf under [`Vocabulary::Runtime`] when the graph has admitted it as a
/// unit (`EGraph::admit_unit`, which carries the variance its body is not
/// here to give), and declines otherwise.
///
/// **Reachable-only.** A term representation may hold nodes no longer reached
/// from `root` — an arena accumulates construction garbage — and inserting
/// those would spend `max_minted`, a budget dimension, on nodes the budget's
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
                    Shape::Const(v) => egraph.add(ENode::Const(v)),
                    Shape::Param(i) => match vocab {
                        Vocabulary::Templates => egraph.add(ENode::Param(i)),
                        Vocabulary::Runtime => return Err(Declined::Param(i)),
                    },
                    Shape::Ref(key) => match (vocab, egraph.unit_variance(key)) {
                        (Vocabulary::Runtime, Some(_)) => egraph.add(ENode::Ref { key }),
                        _ => return Err(Declined::Ref(key)),
                    },
                    Shape::Write { .. } => return Err(Declined::Write),
                    Shape::Buffer(decl) => egraph.add(ENode::Buffer(decl)),
                    Shape::Uniform(decl) => egraph.add(ENode::Uniform(decl)),
                    Shape::Op(kind, children) => {
                        // Check before descending so an unrepresentable op
                        // is reported at the node that carries it.
                        if !vocab.admits(kind) {
                            return Err(Declined::Op(kind));
                        }
                        if children.len() != kind.arity() {
                            return Err(Declined::Arity {
                                op: kind,
                                operands: children.len(),
                            });
                        }
                        tasks.push(Task::Complete(r));
                        // A shift's count is the shift's own parameter, not
                        // a term to visit: only its operand is.
                        if let Some(direction) = ShiftDirection::of(kind) {
                            let Some((operand, _)) = shift::spelled(term, children) else {
                                return Err(Declined::ShiftCount(direction));
                            };
                            tasks.push(Task::Visit(operand));
                            continue;
                        }
                        // Reverse, so children pop in operand order.
                        for i in (0..children.len()).rev() {
                            let child = children.get(i).expect("index < len");
                            tasks.push(Task::Visit(child));
                        }
                        continue;
                    }
                    // No vocabulary check: a fold is not an op, so there is
                    // nothing for a vocabulary to admit or refuse. It enters
                    // as itself, under either.
                    Shape::Reduce { fold, body, lo, hi } => {
                        // Constness discovered as `push_reduce` discovers it: a
                        // value bound whose term is a literal naming a count
                        // is the node's data, neither visited nor made a
                        // class. Refused before anything is built.
                        let (lo, hi) = Bound::discover(lo, hi, |r| literal(term, r))
                            .unwrap_or_else(|refusal| panic!("{refusal}: {fold:?}"));
                        tasks.push(Task::Complete(r));
                        // Reverse, so children pop in `children()` order: the
                        // body, then `lo`'s, then `hi`'s.
                        for child in [hi.value(), lo.value()].into_iter().flatten() {
                            tasks.push(Task::Visit(child));
                        }
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
                    Shape::Op(kind, children) => match ShiftDirection::of(kind) {
                        Some(direction) => {
                            let (_, count) = shift::spelled(term, children)
                                .expect("the count was checked when the shift was visited");
                            let operand = built.pop().expect("the operand was visited first");
                            egraph.add(ENode::Shift {
                                direction,
                                count,
                                operand,
                            })
                        }
                        None => {
                            let start = built.len() - children.len();
                            let node = ENode::apply(kind, &built[start..]);
                            built.truncate(start);
                            egraph.add(node)
                        }
                    },
                    Shape::Reduce { fold, lo, hi, .. } => {
                        let (lo, hi) = Bound::discover(lo, hi, |r| literal(term, r))
                            .expect("a fold's bounds were discovered when it was visited");
                        // The header refuses a value bound its fold cannot
                        // take before `built` is touched.
                        let header = FoldHeader::new(fold, lo.erased(), hi.erased());
                        let start = built.len() - header.arity();
                        let (lo, hi) = header.attach(built[start + 1..].iter().copied());
                        let node = ENode::reduce(fold, lo, hi, built[start]);
                        built.truncate(start);
                        egraph.add(node)
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

/// The literal `r` is in `term`, if it is one: how a fold's value bound is read
/// for the count it may name (`Bound::discover`).
fn literal<I: Ir>(term: &I, r: I::Ref) -> Option<Lit> {
    match term.project(r) {
        Shape::Const(lit) => Some(lit),
        _ => None,
    }
}

/// The classes [`insert`] makes from `root`, without making them — the size
/// measure saturation budgets key on.
///
/// Shares [`insert`]'s reachability so a budget cannot be picked from one node
/// set while the graph is built from another. That is not the count of nodes
/// in the term: a shift's count is read, not visited, so its literal is
/// counted only if something else reaches it. A measure of the term itself
/// is a walk of the term.
pub fn inserted_class_count<I: Ir>(term: &I, root: I::Ref) -> usize {
    let mut seen: BTreeSet<I::Ref> = BTreeSet::new();
    let mut stack = alloc::vec![root];
    while let Some(r) = stack.pop() {
        if !seen.insert(r) {
            continue;
        }
        match term.project(r) {
            // A shift's count is read, not visited, as in `insert`.
            Shape::Op(kind, children) if ShiftDirection::of(kind).is_some() => {
                match shift::spelled(term, children) {
                    Some((operand, _)) => stack.push(operand),
                    None => stack.extend((0..children.len()).filter_map(|i| children.get(i))),
                }
            }
            Shape::Op(_, children) => match children {
                Children::Many(s) => stack.extend_from_slice(s),
                other => {
                    for i in 0..other.len() {
                        stack.push(other.get(i).expect("index < len"));
                    }
                }
            },
            // A value bound's class counts, as `insert` makes it; one read as
            // a literal makes none. A mask, which `insert` refuses, counts as
            // the class it would be: this is a measure, not a check.
            Shape::Reduce { body, lo, hi, .. } => {
                stack.push(body);
                let (lo, hi) = Bound::discover(lo, hi, |r| literal(term, r)).unwrap_or((lo, hi));
                stack.extend(lo.value());
                stack.extend(hi.value());
            }
            Shape::Write { value, .. } => stack.push(value),
            _ => {}
        }
    }
    seen.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_ir::arena::{UniformDecl, UniformIdentity};
    use pixelflow_ir::fold::{Binder, Fold, Monoid};
    use pixelflow_ir::{
        Bound, ExprArena, ExprId, Kernel, KernelKey, KernelStore, Lit, OpKind, Variance,
    };

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

    /// A unit the graph admitted inserts as one opaque leaf, its key and
    /// nothing else, and varies as it was admitted — so the product over it
    /// varies with X as well as Y, which nothing could have learned from the
    /// name.
    #[test]
    fn an_admitted_unit_is_a_leaf_varying_as_it_was_admitted() {
        let (arena, root, reference, key) = named_times_y();
        let mut eg = EGraph::new();
        eg.admit_unit(key, Variance::X);
        let product = insert(&arena, root, &mut eg, Vocabulary::Runtime).expect("admitted");
        let leaf = insert(&arena, reference, &mut eg, Vocabulary::Runtime).expect("admitted");
        assert!(
            eg.nodes(leaf)
                .any(|n| matches!(n, ENode::Ref { key: k } if *k == key)),
            "the leaf names the unit"
        );
        assert_eq!(eg.variance(leaf), Variance::X);
        assert_eq!(eg.variance(product), Variance::X.union(Variance::Y));
    }

    /// A `Ref` is only a key, so a graph that was never told what the key
    /// varies as has nothing to give it: adding one is refused rather than
    /// answered with a guess.
    #[test]
    #[should_panic(expected = "which no unit walk admitted")]
    fn a_ref_to_an_unadmitted_unit_is_refused() {
        let (_, _, _, key) = named_times_y();
        EGraph::new().add(ENode::Ref { key });
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

    /// `push_nary` takes any op and any count. A term that gives an op the
    /// wrong number of operands is declined, naming both, and leaves the
    /// graph untouched, where building the node would abort.
    #[test]
    fn an_op_with_the_wrong_operand_count_is_declined_not_built() {
        for (op, operands) in [
            (OpKind::Add, 1),
            (OpKind::Add, 3),
            (OpKind::Neg, 2),
            (OpKind::MulAdd, 4),
        ] {
            let mut arena = ExprArena::new();
            let x = arena.push_var(0);
            let root = arena.push_nary(op, &alloc::vec![x; operands]);
            let mut eg = EGraph::new();
            assert_eq!(
                insert(&arena, root, &mut eg, Vocabulary::Runtime),
                Err(Declined::Arity { op, operands }),
            );
            assert_eq!(eg.num_classes(), 0, "{op:?}: nothing was built");
        }
    }

    /// `Shl(x, 3)` over the arena's literal count inserts as one shift node
    /// over `x`'s class, and no class is made for the count: two classes,
    /// not three. A literal that is also an ordinary operand elsewhere is
    /// made there, once, and the shift's count does not mention it.
    #[test]
    fn a_shift_inserts_its_count_as_a_parameter_not_a_class() {
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let three = arena.push_const(3.0);
        let shifted = arena.push_binary(OpKind::Shl, x, three);
        let mut eg = EGraph::new();
        let class = insert(&arena, shifted, &mut eg, Vocabulary::Runtime).expect("a valid shift");
        assert_eq!(eg.num_classes(), 2);
        assert_eq!(inserted_class_count(&arena, shifted), 2);
        assert!(matches!(
            eg.nodes(class).collect::<Vec<_>>().as_slice(),
            [ENode::Shift {
                direction: ShiftDirection::Left,
                count,
                ..
            }] if count.bits() == 3
        ));

        let scaled = arena.push_binary(OpKind::Mul, x, three);
        let both = arena.push_binary(OpKind::Add, shifted, scaled);
        let mut eg = EGraph::new();
        insert(&arena, both, &mut eg, Vocabulary::Runtime).expect("valid");
        assert_eq!(
            eg.num_classes(),
            5,
            "x, the shift, the literal (made by the Mul), the Mul, the Add"
        );
        assert_eq!(inserted_class_count(&arena, both), 5);
    }

    /// Two shifts of one operand by one count are one node; by two counts,
    /// two.
    #[test]
    fn a_shift_is_hash_consed_on_its_count() {
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let (two, three) = (arena.push_const(2.0), arena.push_const(3.0));
        let a = arena.push_binary(OpKind::Shr, x, two);
        let b = arena.push_binary(OpKind::Shr, x, three);
        let root = arena.push_binary(OpKind::Add, a, b);
        let mut eg = EGraph::new();
        insert(&arena, root, &mut eg, Vocabulary::Runtime).expect("valid");
        assert_eq!(eg.num_classes(), 4, "x, two shifts, the sum");
    }

    /// `-0.0` is the integer 0, so a shift by it is a shift by the count 0:
    /// the same node as a shift by `+0.0`, not a declined one.
    #[test]
    fn a_shift_by_negative_zero_is_a_shift_by_the_count_zero() {
        for op in [OpKind::Shl, OpKind::Shr] {
            let mut arena = ExprArena::new();
            let x = arena.push_var(0);
            let (negative, positive) = (arena.push_const(-0.0), arena.push_const(0.0));
            let by_negative = arena.push_binary(op, x, negative);
            let by_positive = arena.push_binary(op, x, positive);
            let mut eg = EGraph::new();
            let class =
                insert(&arena, by_negative, &mut eg, Vocabulary::Runtime).expect("-0.0 is a count");
            assert_eq!(
                eg.num_classes(),
                2,
                "x and the shift, no class for the count"
            );
            assert!(matches!(
                eg.nodes(class).collect::<Vec<_>>().as_slice(),
                [ENode::Shift { count, .. }] if count.bits() == 0
            ));
            let same =
                insert(&arena, by_positive, &mut eg, Vocabulary::Runtime).expect("+0.0 is a count");
            assert_eq!(same, class, "{op:?}: one node, whichever zero spelled it");
            assert_eq!(eg.num_classes(), 2);
        }
    }

    /// A count that is not a `Const` holding a `ShiftCount` is
    /// declined, naming the direction, before anything is built: a graph cannot
    /// hold what the emitter would refuse, and the term compiles unoptimized.
    #[test]
    fn a_shift_by_anything_but_a_valid_literal_is_declined() {
        for op in [OpKind::Shl, OpKind::Shr] {
            let direction = ShiftDirection::of(op).expect("a shift");
            for count in [-1.0, 0.5, 32.0, 33.0, 256.0, f32::NAN, f32::INFINITY] {
                let mut arena = ExprArena::new();
                let x = arena.push_var(0);
                let count_node = arena.push_const(count);
                let root = arena.push_binary(op, x, count_node);
                let mut eg = EGraph::new();
                assert_eq!(
                    insert(&arena, root, &mut eg, Vocabulary::Runtime),
                    Err(Declined::ShiftCount(direction)),
                    "{op:?} by {count}"
                );
                assert_eq!(eg.num_classes(), 0, "nothing was built");
            }

            // A count that is arithmetic, or a leaf that is not a literal, is
            // an operand and not a parameter.
            for count_node in [
                {
                    let mut arena = ExprArena::new();
                    let x = arena.push_var(0);
                    let (a, b) = (arena.push_const(1.0), arena.push_const(2.0));
                    let sum = arena.push_binary(OpKind::Add, a, b);
                    (arena, x, sum)
                },
                {
                    let mut arena = ExprArena::new();
                    let x = arena.push_var(0);
                    let y = arena.push_var(1);
                    (arena, x, y)
                },
            ] {
                let (mut arena, x, count) = count_node;
                let root = arena.push_binary(op, x, count);
                let mut eg = EGraph::new();
                assert_eq!(
                    insert(&arena, root, &mut eg, Vocabulary::Runtime),
                    Err(Declined::ShiftCount(direction)),
                );
                assert_eq!(eg.num_classes(), 0, "nothing was built");
            }
        }
    }

    /// A `Tuple` has no fixed arity, so no vocabulary holds one: it is
    /// declined as an op before its operand count is asked about.
    #[test]
    fn a_tuple_is_declined_by_every_vocabulary() {
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let root = arena.push_nary(OpKind::Tuple, &[x, x]);
        for vocabulary in [Vocabulary::Templates, Vocabulary::Runtime] {
            assert_eq!(
                insert(&arena, root, &mut EGraph::new(), vocabulary),
                Err(Declined::Op(OpKind::Tuple)),
            );
        }
    }

    /// A unit is optimized out of its context, which is sound only for a
    /// closed term: one reading a binder is refused at the door.
    #[test]
    #[should_panic(expected = "a unit must be")]
    fn an_open_unit_is_refused() {
        let key = KernelStore::intern(&Kernel::y().add(&Kernel::constant(0.75)));
        EGraph::new().admit_unit(key, Variance::Y.union(Variance::BINDERS));
    }

    /// A fold whose `lo` is a term, the one at ref 2: a constant, a uniform or
    /// a mask. The arena reads a constant bound into the fold when it pushes
    /// it and refuses a mask, so a term whose bound is still a constant is a
    /// test's own, as another representation's might be.
    enum ValueBounded {
        Constant,
        Uniform(UniformDecl),
        Mask,
    }

    impl Ir for ValueBounded {
        type Ref = u8;

        fn project(&self, r: u8) -> Shape<'_, u8> {
            let binder = Binder::from_slot(0).expect("a binder");
            match r {
                0 => Shape::Var(0),
                2 => match self {
                    Self::Constant => Shape::Const(Lit::Real(1.0)),
                    Self::Uniform(decl) => Shape::Uniform(*decl),
                    Self::Mask => Shape::Const(Lit::Mask(true)),
                },
                _ => Shape::Reduce {
                    fold: Fold::new(Monoid::SUM, binder),
                    lo: Bound::Value(2),
                    hi: Bound::Lit(4),
                    body: 0,
                },
            }
        }

        fn embed(&mut self, _: Shape<'_, u8>) -> u8 {
            unreachable!("insert reads a term and never writes one")
        }
    }

    fn a_uniform() -> ValueBounded {
        ValueBounded::Uniform(UniformDecl {
            id: UniformIdentity::mint(),
            default: 3.0,
        })
    }

    /// The fold `class` holds, and the class of its body.
    fn the_fold(eg: &EGraph, class: EClassId) -> (ENode, EClassId) {
        let node = eg
            .nodes(class)
            .find(|node| node.as_reduce().is_some())
            .expect("the class holds a fold");
        let (_, _, _, body) = node.as_reduce().expect("a fold");
        (*node, body)
    }

    /// Constness is discovered at the fold, as the arena's `push_reduce`
    /// discovers it: a value bound whose term is a literal naming a count is
    /// the node's data, and no class is made for it. One that is not a literal
    /// stays a child, and its class counts.
    #[test]
    fn insert_reads_a_literal_bound_into_the_node() {
        let fold = Fold::new(Monoid::SUM, Binder::from_slot(0).expect("a binder"));

        let mut eg = EGraph::new();
        let class = insert(&ValueBounded::Constant, 1, &mut eg, Vocabulary::Runtime)
            .expect("a fold inserts");
        let (node, body) = the_fold(&eg, class);
        assert_eq!(
            node,
            ENode::reduce(fold, Bound::Lit(1), Bound::Lit(4), body),
            "the fold over the literal, as the arena builds it"
        );
        assert_eq!(inserted_class_count(&ValueBounded::Constant, 1), 2);
        assert_eq!(eg.num_classes(), 2, "the fold and its body");

        let uniform = a_uniform();
        let mut eg = EGraph::new();
        let class = insert(&uniform, 1, &mut eg, Vocabulary::Runtime).expect("a fold inserts");
        let (node, body) = the_fold(&eg, class);
        let (_, lo, hi, _) = node.as_reduce().expect("a fold");
        assert!(matches!(lo, Bound::Value(_)), "a uniform is no literal");
        assert_eq!(hi, Bound::Lit(4));
        assert_eq!(node.children_slice().len(), 2);
        assert_ne!(node.children_slice()[1], body);
        assert_eq!(inserted_class_count(&uniform, 1), 3);
        assert_eq!(eg.num_classes(), 3, "the fold, its body and its bound");
    }

    /// A mask names no count, whatever the fold: refused before anything is
    /// built, as `push_reduce` refuses one.
    #[test]
    #[should_panic(expected = "BoundSort")]
    fn insert_refuses_a_mask_bound() {
        let _ = insert(
            &ValueBounded::Mask,
            1,
            &mut EGraph::new(),
            Vocabulary::Runtime,
        );
    }

    /// A key names one referent, so it has one variance: admitting it again
    /// as the same is idempotent, and as another is refused rather than left
    /// to give the graph two `Ref` nodes that disagree about one unit.
    #[test]
    fn a_unit_is_admitted_with_one_variance() {
        let key = KernelStore::intern(&Kernel::x().add(&Kernel::constant(0.75)));
        let mut eg = EGraph::new();
        eg.admit_unit(key, Variance::X);
        eg.admit_unit(key, Variance::X);
        assert_eq!(eg.unit_variance(key), Some(Variance::X));
    }

    #[test]
    #[should_panic(expected = "is admitted again as")]
    fn a_unit_admitted_with_two_variances_is_refused() {
        let key = KernelStore::intern(&Kernel::x().add(&Kernel::constant(0.75)));
        let mut eg = EGraph::new();
        eg.admit_unit(key, Variance::X);
        eg.admit_unit(key, Variance::Y);
    }
}
