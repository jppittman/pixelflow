//! IR-to-IR transforms: legalization.
//!
//! Four passes, each `Term -> Rooted`, each turning ops no backend can emit
//! into ops every backend can:
//!
//! | pass | consumes | produces |
//! |---|---|---|
//! | [`lower_dwrt`] | `Dwrt` | arithmetic, and *re-introduces* transcendentals |
//! | [`expand_reduce`] | `Reduce` | the combiner applied over unrolled copies |
//! | [`expand_gather`] | `Gather` | index arithmetic + `RawGather` |
//! | [`expand_transcendentals`] | `Sin`..`Pow` | arithmetic + bit-manip atoms |
//!
//! The order in that table is the order they must run: differentiating a `sin`
//! produces a `cos`, so `lower_dwrt` has to go before the pass that expands
//! them. Every pass is idempotent and has an identity fast-path, so running
//! one that has nothing to do costs a copy of the reachable subgraph and no
//! analysis.
//!
//! Each pass builds a **fresh** graph rather than appending to the one it was
//! handed. That is what the underlying `Dag` offers — build once, freeze,
//! read — and it is also what the append-only arena these passes used to
//! mutate was doing in effect: every one of them rebuilt the reachable
//! subgraph and left the original behind as garbage. The garbage is now simply
//! not created.
//!
//! **Nothing here knows what it is lowering *for*.** There is no `cfg` in this
//! module beyond `#[cfg(test)]`, and no import outside `crate::{dag, decl,
//! expr, kind, variance}`. The legal set happens to be uniform across the
//! backends today; if it stops being uniform, that belongs in a target
//! description these passes consult, not in a `cfg` here.
//!
//! On transcendentals specifically: `sin`, `cos`, `atan` have no single
//! instruction on any target — they are *always* a polynomial — so they are not
//! a backend's business. Expanding them here means no emitter ever contains
//! transcendental assembly, the polynomial has one home, and precision is a
//! property of this code rather than of whichever backend you landed on.
//!
//! They may use `Select`. [`legalize`] runs `lower_dwrt` *before*
//! `expand_transcendentals` — the chain rule manufactures `Sin`/`Cos` nodes
//! that the transcendental pass must still lower — so an expansion is only ever
//! evaluated, never differentiated, and derivatives are taken against the
//! symbolic rules in `diff_node` instead. (`lower_dwrt` carries a `Select` rule
//! regardless: it blends the branch derivatives on the primal mask.)
//!
//! Nothing re-fuses `mul`+`add` into `MulAdd` afterwards — see `horner_step`.

use alloc::vec::Vec;

use crate::dag::{Builder, Id, Node, Rooted, SideTable};
use crate::expr::{ExprBuilderExt, ExprData, Term};
use crate::kind::OpKind;
use crate::variance::Variance;

/// Run every legalization pass, in the one order they compose in.
///
/// This is the whole pipeline. It was previously four calls copied into each
/// compile entry, which is how two since-deleted entries came to run none of
/// them, and how the deleted `CompileWorkspace` came to run none of them *and*
/// skip the guard that refuses a surviving `Dwrt`. An order that has to be
/// retyped is an order that can be forgotten.
///
/// The environment is unchanged — every pass copies `Buffer`/`Uniform` leaves
/// slot-for-slot — so the caller keeps the one it passed in.
///
/// # Errors
///
/// Propagates [`lower_dwrt`]'s error for expressions with no derivative rule —
/// bound-memory reads, integer/bit ops, reductions.
pub fn legalize(term: Term<'_>) -> Result<Rooted<ExprData>, &'static str> {
    let env = term.env();
    // `lower_dwrt` first: differentiating a `sin` manufactures a `cos`, so it
    // has to precede the pass that expands them.
    let d = lower_dwrt(term)?;
    let r = expand_reduce(Term::new(d.entry(), env));
    let g = expand_gather(Term::new(r.entry(), env));
    Ok(expand_transcendentals(Term::new(g.entry(), env)))
}

/// Whether `op` is a unary transcendental this pass expands.
fn is_transcendental_unary(op: OpKind) -> bool {
    matches!(
        op,
        OpKind::Sin
            | OpKind::Cos
            | OpKind::Tan
            | OpKind::Exp
            | OpKind::Exp2
            | OpKind::Ln
            | OpKind::Log2
            | OpKind::Log10
            | OpKind::Atan
            | OpKind::Asin
            | OpKind::Acos
    )
}

/// Whether `op` is a binary transcendental this pass expands.
fn is_transcendental_binary(op: OpKind) -> bool {
    matches!(op, OpKind::Atan2 | OpKind::Pow)
}

/// Whether a node applies `op` to exactly `arity` children.
fn is_op(node: Node<'_, ExprData>, op: OpKind, arity: usize) -> bool {
    *node == ExprData::Op(op) && node.child_count() == arity
}

// ────────────────────────────── the rebuild spine ─────────────────────────────

/// The graph a pass is building, plus where each source node landed in it.
///
/// Every pass is "copy the reachable subgraph, replacing some nodes", and this
/// is the state that shape needs: the destination builder, and the map a
/// lowering hook uses to find the already-copied children of the node it is
/// looking at.
struct Lowering {
    out: Builder<ExprData>,
    map: SideTable<Option<Id>>,
}

impl Lowering {
    /// Where `node` landed. Children are copied before parents, so this is
    /// populated for every child of the node a hook is looking at.
    fn copied(&self, node: Node<'_, ExprData>) -> Id {
        self.map[node].expect("child copied before parent")
    }

    /// The already-copied children of `node`, in operand order.
    fn kids(&self, node: Node<'_, ExprData>) -> Vec<Id> {
        node.children().map(|c| self.copied(c)).collect()
    }
}

/// Uninhabited error type for the infallible [`rebuild`] wrapper.
enum Never {}

/// Post-order rebuild of the subgraph reachable from `term`'s root, one
/// lowering pass.
///
/// For each node (children first), `lower` may return `Some(id)` to replace it
/// — building whatever it likes out of the already-copied children — or `None`
/// to keep it as a plain structural copy. Shared subexpressions are rebuilt
/// once, so a DAG stays a DAG.
fn try_rebuild<'a, E, F>(term: Term<'a>, mut lower: F) -> Result<Rooted<ExprData>, E>
where
    F: FnMut(&mut Lowering, Node<'a, ExprData>) -> Result<Option<Id>, E>,
{
    let root = term.root();
    let mut cx = Lowering {
        out: Builder::with_capacity(term.dag().len(), 0),
        map: term.dag().side_table(None),
    };
    let mut work = alloc::vec![(root, false)];

    while let Some((node, expanded)) = work.pop() {
        if cx.map[node].is_some() {
            continue;
        }
        if !expanded {
            work.push((node, true));
            for child in node.children() {
                if cx.map[child].is_none() {
                    work.push((child, false));
                }
            }
            continue;
        }
        let id = match lower(&mut cx, node)? {
            Some(new) => new,
            None => {
                let kids = cx.kids(node);
                cx.out.push_unique(*node, &kids)
            }
        };
        cx.map[node] = Some(id);
    }

    let new_root = cx.map[root].expect("root lowered");
    Ok(cx.out.finish(&[new_root]))
}

/// [`try_rebuild`] for a hook that cannot fail.
fn rebuild<'a, F>(term: Term<'a>, mut lower: F) -> Rooted<ExprData>
where
    F: FnMut(&mut Lowering, Node<'a, ExprData>) -> Option<Id>,
{
    match try_rebuild::<Never, _>(term, |cx, node| Ok(lower(cx, node))) {
        Ok(r) => r,
        Err(never) => match never {},
    }
}

/// The identity of every pass: the reachable subgraph, copied.
fn copy_only(term: Term<'_>) -> Rooted<ExprData> {
    // Ascending index order over the reachable nodes, NOT a DFS copy. A DAG's
    // index order is already topological, so this reproduces the input's
    // relative node order exactly — which is what makes the fast path a true
    // no-op. Downstream, that order IS the schedule (`pixelflow-codegen`'s
    // `term_to_schedule` walks it), so a DFS copy silently re-orders every
    // transcendental-free kernel's schedule and perturbs its register
    // allocation: `emit::tests::sched::sched_spills_and_is_correct` stopped
    // spilling. The arena-era fast path returned the arena untouched and said
    // so in a comment; this is that invariant, kept by construction.
    let dag = term.dag();
    let mut reachable = dag.side_table(false);
    for n in term.root().descendants() {
        reachable[n] = true;
    }
    let mut out = Builder::with_capacity(dag.len(), 0);
    let mut map = dag.side_table(None);
    for node in dag.iter() {
        if !reachable[node] {
            continue;
        }
        let kids: Vec<Id> = node
            .children()
            .map(|c| map[c].expect("copy_only: child copied before parent"))
            .collect();
        map[node] = Some(out.push_unique(*node, &kids));
    }
    let root = map[term.root()].expect("copy_only: the root is reachable from itself");
    out.finish(&[root])
}

/// Whether any node reachable from `term`'s root satisfies `pred` — every
/// pass's identity fast-path.
fn reaches(term: Term<'_>, pred: impl Fn(Node<'_, ExprData>) -> bool) -> bool {
    term.root().descendants().any(pred)
}

/// Bottom-up "does this subgraph still contain a node satisfying `pred`".
///
/// Computed once per pass rather than per candidate: asking `descendants()` at
/// every `Dwrt`/`Reduce` would be quadratic, and both passes ask at every one.
fn carries(term: Term<'_>, pred: impl Fn(Node<'_, ExprData>) -> bool) -> SideTable<bool> {
    let mut table = term.dag().side_table(false);
    for node in term.dag().iter() {
        table[node] = pred(node) || node.children().any(|c| table[c]);
    }
    table
}

// ──────────────────────────── Transcendental lowering ─────────────────────────

/// Expand every transcendental node reachable from `term`'s root into a
/// primitive arithmetic subgraph. Non-transcendental nodes are copied
/// unchanged.
#[must_use]
pub fn expand_transcendentals(term: Term<'_>) -> Rooted<ExprData> {
    if !reaches(term, is_transcendental) {
        return copy_only(term);
    }
    rebuild(term, |cx, node| {
        let op = node.op()?;
        match node.child_count() {
            1 if is_transcendental_unary(op) => {
                let child = node.children().next().expect("arity checked");
                let a = cx.copied(child);
                Some(expand_unary(&mut cx.out, op, a))
            }
            2 if is_transcendental_binary(op) => {
                let kids = cx.kids(node);
                Some(expand_binary(&mut cx.out, op, kids[0], kids[1]))
            }
            _ => None,
        }
    })
}

fn is_transcendental(node: Node<'_, ExprData>) -> bool {
    match node.op() {
        Some(op) if node.child_count() == 1 => is_transcendental_unary(op),
        Some(op) if node.child_count() == 2 => is_transcendental_binary(op),
        _ => false,
    }
}

// ─────────────────────────────── Gather lowering ──────────────────────────────

/// Lower every high-level `Gather(buffer, x, y)` reachable from `term`'s root
/// into index arithmetic plus a primitive [`OpKind::RawGather`].
///
/// The index expression is byte-for-byte the one `DiscreteManifold::eval`
/// computes — `clamp(floor(idx), 0, extent-1)` per axis, then
/// `yi * width + xi` — so the emitter only ever sees ops it already supports
/// (`Floor`, `Min`, `Max`, `Mul`, `Add`) plus the single `RawGather`
/// primitive. This is the analogue of [`expand_transcendentals`] for memory
/// reads.
#[must_use]
pub fn expand_gather(term: Term<'_>) -> Rooted<ExprData> {
    if !reaches(term, |n| is_op(n, OpKind::Gather, 3)) {
        return copy_only(term);
    }
    rebuild(term, |cx, node| {
        if !is_op(node, OpKind::Gather, 3) {
            return None;
        }
        let buf_node = node.children().next().expect("Gather has three children");
        let decl = match *buf_node {
            ExprData::Buffer(id) => term.env().buffer(id),
            other => panic!("expand_gather: first child must be a Buffer leaf, got {other:?}"),
        };
        let kids = cx.kids(node);
        Some(lower_gather(&mut cx.out, kids[0], kids[1], kids[2], decl))
    })
}

/// Build the index arithmetic for one gather and wrap it in a `RawGather`.
///
/// Produces `RawGather(buf, clamp(floor(y),0,h-1) * width + clamp(floor(x),0,w-1))`,
/// matching `DiscreteManifold::eval`.
fn lower_gather(
    out: &mut Builder<ExprData>,
    buf: Id,
    x: Id,
    y: Id,
    decl: crate::decl::BufferDecl,
) -> Id {
    let zero = out.push_const(0.0);
    let max_x = out.push_const(decl.width.saturating_sub(1) as f32);
    let max_y = out.push_const(decl.height.saturating_sub(1) as f32);
    let width = out.push_const(decl.width as f32);

    // xi = clamp(floor(x), 0, width-1); yi = clamp(floor(y), 0, height-1),
    // written as the min/max composition clamp denotes — there is no `Clamp`
    // primitive to lower to.
    let fx = out.push_unary(OpKind::Floor, x);
    let xi_lo = out.push_binary(OpKind::Max, fx, zero);
    let xi = out.push_binary(OpKind::Min, xi_lo, max_x);
    let fy = out.push_unary(OpKind::Floor, y);
    let yi_lo = out.push_binary(OpKind::Max, fy, zero);
    let yi = out.push_binary(OpKind::Min, yi_lo, max_y);

    // idx = yi * width + xi  (float; exact for indices < 2^24, as in DiscreteManifold)
    let row = out.push_binary(OpKind::Mul, yi, width);
    let idx = out.push_binary(OpKind::Add, row, xi);

    out.push_binary(OpKind::RawGather, buf, idx)
}

// ─────────────────────────────── Reduce lowering ──────────────────────────────

/// Unroll every `Reduce` reachable from `term`'s root into an explicit
/// accumulation tree.
///
/// `Reduce([combiner, var, extent, body])` becomes
/// `combiner(body[var:=0], combiner(body[var:=1], … body[var:=N-1]))` — N
/// inlined copies of `body` with the reduction index substituted as a `Const`.
/// Because the extent is static, each copy's gather indices become constant, so
/// the emitter folds their addresses to immediates: the fold compiles to a
/// flat, call-free, unrolled kernel.
///
/// Nested folds take one round per level of nesting: the innermost is the one
/// whose body no longer contains a `Reduce`, and unrolling it makes the fold
/// enclosing it innermost in turn. (A round is a full copy, and the nesting
/// limit is the four binder slots, so this is bounded by construction.)
///
/// # Panics
///
/// Panics if a reduction's combiner, index or extent operand is not a `Const`
/// — a uniform is a *value*, never a trip count, and reading a slot index as
/// one would unroll a plausible, wrong number of times.
#[must_use]
pub fn expand_reduce(term: Term<'_>) -> Rooted<ExprData> {
    let is_reduce = |n: Node<'_, ExprData>| *n == ExprData::Op(OpKind::Reduce);
    if !reaches(term, is_reduce) {
        return copy_only(term);
    }

    let mut current = copy_only(term);
    loop {
        let next = {
            let round = Term::new(current.entry(), term.env());
            if !reaches(round, is_reduce) {
                break;
            }
            let innermost = carries(round, is_reduce);
            let variance = crate::variance::compute_dag_variance(round.dag());
            rebuild(round, |cx, node| {
                if !is_reduce(node) {
                    return None;
                }
                let kids: Vec<Node<'_, ExprData>> = node.children().collect();
                debug_assert_eq!(kids.len(), 4, "Reduce has 4 children");
                if innermost[kids[3]] {
                    // A fold still nested inside this one: leave this `Reduce`
                    // alone for now, and let the next round see a body that
                    // has already been unrolled.
                    return None;
                }
                Some(unroll_reduce(cx, &kids, &variance))
            })
        };
        current = next;
    }
    current
}

/// Build the unrolled accumulation for one reduction whose children are
/// already copied. Reads `combiner`/`var`/`extent` from their `Const` nodes,
/// then folds `extent` substituted copies of `body` under the combiner monoid.
fn unroll_reduce(
    cx: &mut Lowering,
    kids: &[Node<'_, ExprData>],
    variance: &SideTable<Variance>,
) -> Id {
    let combiner_op = OpKind::from_index(const_val(kids[0], "reduce combiner") as usize)
        .expect("reduce combiner must be a valid OpKind index");
    let var_idx = const_val(kids[1], "reduce var") as u8;
    let n = const_val(kids[2], "reduce extent") as usize;
    let body = kids[3];

    // Empty domain folds to the monoid identity.
    if n == 0 {
        let id = combiner_op
            .monoid_identity()
            .expect("reduce combiner is a monoid");
        return cx.out.push_const(id);
    }

    // acc = body[var:=0]; then acc = combiner(acc, body[var:=k]) for k in 1..N.
    let mut acc = Substitution::new(var_idx, 0.0, variance).apply(cx, body);
    for k in 1..n {
        let next = Substitution::new(var_idx, k as f32, variance).apply(cx, body);
        acc = cx.out.push_binary(combiner_op, acc, next);
    }
    acc
}

/// Read the value of a `Const` node (reduction metadata).
fn const_val(node: Node<'_, ExprData>, what: &str) -> f32 {
    node.as_f32()
        .unwrap_or_else(|| panic!("{what} must be a Const, got {:?}", *node))
}

/// One unrolled term of a fold: the body with the bound index replaced by a
/// literal step.
///
/// The variance table is what makes this cheap. A subtree that does not depend
/// on the index would be rebuilt unchanged, so it is not rebuilt at all — the
/// copy already made for it is reused and all N terms share it. That is the
/// rewrite `⊕_i (f(i) · c) = c · ⊕_i f(i)` obtained by declining to duplicate
/// `c`.
struct Substitution<'a> {
    /// The index being replaced, and the step to replace it with.
    var: u8,
    value: f32,
    /// Variance for every node of the graph being lowered.
    variance: &'a SideTable<Variance>,
    /// Rebuilt nodes, so a shared subtree is rebuilt once and stays shared.
    memo: Option<SideTable<Option<Id>>>,
}

impl<'a> Substitution<'a> {
    fn new(var: u8, value: f32, variance: &'a SideTable<Variance>) -> Self {
        Self {
            var,
            value,
            variance,
            memo: None,
        }
    }

    fn apply(&mut self, cx: &mut Lowering, root: Node<'_, ExprData>) -> Id {
        let memo = self.memo.get_or_insert_with(|| root.dag().side_table(None));
        let mut work = alloc::vec![(root, false)];
        while let Some((node, expanded)) = work.pop() {
            if memo[node].is_some() {
                continue;
            }
            if self.variance[node].is_invariant_in(self.var) {
                memo[node] = Some(cx.copied(node));
                continue;
            }
            if !expanded {
                work.push((node, true));
                for child in node.children() {
                    if memo[child].is_none() {
                        work.push((child, false));
                    }
                }
                continue;
            }
            let id = match *node {
                ExprData::Var(i) if i == self.var => cx.out.push_const(self.value),
                data => {
                    let kids: Vec<Id> = node
                        .children()
                        .map(|c| memo[c].expect("child substituted before parent"))
                        .collect();
                    cx.out.push_unique(data, &kids)
                }
            };
            memo[node] = Some(id);
        }
        memo[root].expect("the body's root was substituted")
    }
}

// ─────────────────────────────── Dwrt lowering ───────────────────────────────

/// Rewrite every `Dwrt(expr, var)` reachable from `term`'s root into the
/// analytic derivative subgraph of `expr` with respect to coordinate `var`.
///
/// This is the runtime peer of the e-graph `ChainRule` (pixelflow-search):
/// same algebra, applied directly to the graph with no e-graph dependency.
/// Derivatives of piecewise ops (`Min`/`Max`/`Select`/`Abs`) are a mask on the
/// primal values selecting between branch derivatives.
///
/// Runs *before* [`expand_transcendentals`] (its rules produce `Sin`/`Cos`/
/// `Exp` etc., which that pass then lowers) and processes innermost `Dwrt`
/// first, so nested derivatives (`Dwrt(Dwrt(e, 0), 0)`) differentiate an
/// already-`Dwrt`-free subgraph. "Innermost first" is a round per level of
/// nesting, for the reason [`expand_reduce`] gives.
///
/// # Errors
///
/// Errors loudly on any op with no derivative rule (bound-memory reads,
/// integer/bit ops, reductions) rather than silently miscompiling.
pub fn lower_dwrt(term: Term<'_>) -> Result<Rooted<ExprData>, &'static str> {
    let is_dwrt = |n: Node<'_, ExprData>| *n == ExprData::Op(OpKind::Dwrt);
    if !reaches(term, is_dwrt) {
        return Ok(copy_only(term));
    }

    let mut current = copy_only(term);
    loop {
        let next = {
            let round = Term::new(current.entry(), term.env());
            if !reaches(round, is_dwrt) {
                break;
            }
            let nested = carries(round, is_dwrt);
            try_rebuild(round, |cx, node| {
                if !is_dwrt(node) {
                    return Ok(None);
                }
                let kids: Vec<Node<'_, ExprData>> = node.children().collect();
                if kids.len() != 2 {
                    return Err("lower_dwrt: malformed Dwrt node (must be Binary(expr, var))");
                }
                if nested[kids[0]] {
                    // An inner derivative first: this round leaves the outer one
                    // in place and the next differentiates its lowered operand.
                    return Ok(None);
                }
                let Some(var) = kids[1].as_f32() else {
                    return Err("lower_dwrt: Dwrt's variable operand must be a Const");
                };
                differentiate(cx, kids[0], var as u8).map(Some)
            })?
        };
        current = next;
    }
    Ok(current)
}

/// A derivative under construction: either a literal this pass folded, or a
/// node in the graph being built.
///
/// Most leaf derivatives are `0` or `1`, and the peephole rules below fold
/// them — which keeps the lowered graph near the size the e-graph `ChainRule`
/// plus algebraic cleanup would produce, without pulling an optimizer into
/// pixelflow-ir. Carrying the literal in the type rather than pushing a
/// `Const` and reading it back is what lets those folds cascade: a builder
/// hands out nodes, not values.
#[derive(Clone, Copy)]
enum Val {
    Const(f32),
    Node(Id),
}

impl Val {
    const ZERO: Self = Self::Const(0.0);

    fn is_zero(self) -> bool {
        matches!(self, Self::Const(v) if v == 0.0)
    }

    fn is_one(self) -> bool {
        matches!(self, Self::Const(v) if v == 1.0)
    }

    fn id(self, out: &mut Builder<ExprData>) -> Id {
        match self {
            Self::Const(v) => out.push_const(v),
            Self::Node(id) => id,
        }
    }
}

/// Build `∂(expr)/∂(Var(var))`, sharing the primal subgraph with the copy the
/// enclosing rebuild already made. Memoized per node, so a DAG differentiates
/// once per shared subexpression (forward-mode on the DAG).
///
/// Fully iterative — no recursion over expression depth, so arbitrarily deep
/// kernels cannot overflow the stack. Two passes: (1) mark the nodes whose
/// derivative a rule actually consumes (lazy per op: `Select` masks and
/// comparison operands are never differentiated), walking an explicit stack;
/// (2) compute marked derivatives in the DAG's own order, which is children
/// before parents.
fn differentiate(cx: &mut Lowering, expr: Node<'_, ExprData>, var: u8) -> Result<Id, &'static str> {
    let dag = expr.dag();

    // Pass 1: mark derivative-needed nodes.
    let mut need = dag.side_table(false);
    let mut stack = alloc::vec![expr];
    while let Some(node) = stack.pop() {
        if core::mem::replace(&mut need[node], true) {
            continue;
        }
        push_deriv_children(node, &mut stack);
    }

    // Pass 2: bottom-up compute in topological order.
    let mut memo = dag.side_table(None);
    for node in dag.iter() {
        if !need[node] {
            continue;
        }
        let d = diff_node(cx, node, var, &memo)?;
        memo[node] = Some(d);
    }
    let d = memo[expr].expect("derivative of the root was computed");
    Ok(d.id(&mut cx.out))
}

/// Which children's derivatives the rule for `node` consumes. Must stay in
/// lockstep with [`diff_node`]: a child pushed here is differentiated eagerly;
/// a child omitted here must not be read from the memo there. Ops with no
/// rule push nothing — [`diff_node`] raises the error for the node itself.
fn push_deriv_children<'a>(node: Node<'a, ExprData>, stack: &mut Vec<Node<'a, ExprData>>) {
    let Some(op) = node.op() else { return };
    let kids: Vec<Node<'a, ExprData>> = node.children().collect();
    match (op, kids.len()) {
        // d = 0 without touching the operand, and no rule at all: either way
        // the operand's derivative is never read.
        (OpKind::Floor | OpKind::Ceil | OpKind::Round, 1) => {}
        (OpKind::TruncToInt | OpKind::IntToFloat, 1) => {}
        (_, 1) => stack.push(kids[0]),
        (
            OpKind::Add
            | OpKind::Sub
            | OpKind::Mul
            | OpKind::Div
            | OpKind::Min
            | OpKind::Max
            | OpKind::Atan2
            | OpKind::Pow,
            2,
        ) => {
            stack.push(kids[0]);
            stack.push(kids[1]);
        }
        (OpKind::MulAdd, 3) => stack.extend(kids),
        // The mask is never differentiated.
        (OpKind::Select, 3) => stack.extend(kids.into_iter().skip(1)),
        // Masks, integer ops, reductions: nothing to descend into.
        _ => {}
    }
}

fn diff_node(
    cx: &mut Lowering,
    node: Node<'_, ExprData>,
    var: u8,
    memo: &SideTable<Option<Val>>,
) -> Result<Val, &'static str> {
    let data = *node;
    let op = match data {
        ExprData::Var(i) => {
            return Ok(Val::Const(if i == var { 1.0 } else { 0.0 }));
        }
        // Constants, scalar params (baked before evaluation) and uniforms
        // (invariant across the lattice) are coordinate-independent.
        ExprData::Const(_) | ExprData::Param(_) | ExprData::Uniform(_) => return Ok(Val::ZERO),
        ExprData::Buffer(_) => return Err("lower_dwrt: cannot differentiate a bound-memory read"),
        ExprData::Op(op) => op,
    };

    let kids: Vec<Node<'_, ExprData>> = node.children().collect();
    // The primal operands, as nodes in the graph being built.
    let p = |cx: &Lowering, i: usize| cx.copied(kids[i]);
    let d = |i: usize| memo[kids[i]].expect("child derivative marked and computed before parent");

    match (op, kids.len()) {
        // ───────────────────────────── unary ─────────────────────────────
        // Step functions: zero derivative almost everywhere. The operand is
        // never marked in pass 1, so the memo must not be read here.
        (OpKind::Floor | OpKind::Ceil | OpKind::Round, 1) => Ok(Val::ZERO),
        (OpKind::TruncToInt | OpKind::IntToFloat, 1) => {
            Err("lower_dwrt: cannot differentiate integer/bit-manipulation ops")
        }
        (OpKind::Neg, 1) => Ok(d_neg(cx, d(0))),
        // d(√u) = 0.5·rsqrt(u)·u'.
        (OpKind::Sqrt, 1) => {
            let a = p(cx, 0);
            let half = cx.out.push_const(0.5);
            let rs = cx.out.push_unary(OpKind::Rsqrt, a);
            let factor = cx.out.push_binary(OpKind::Mul, half, rs);
            Ok(d_mul(cx, Val::Node(factor), d(0)))
        }
        // d(u^-1/2) = -0.5·u^-3/2·u' = -0.5·rsqrt(u)·recip(u)·u'.
        (OpKind::Rsqrt, 1) => {
            let a = p(cx, 0);
            let neg_half = cx.out.push_const(-0.5);
            let rs = cx.out.push_unary(OpKind::Rsqrt, a);
            let rc = cx.out.push_unary(OpKind::Recip, a);
            let t = cx.out.push_binary(OpKind::Mul, rs, rc);
            let factor = cx.out.push_binary(OpKind::Mul, neg_half, t);
            Ok(d_mul(cx, Val::Node(factor), d(0)))
        }
        // d(1/u) = -u' / u².
        (OpKind::Recip, 1) => {
            let a = p(cx, 0);
            let ndu = d_neg(cx, d(0)).id(&mut cx.out);
            let u2 = cx.out.push_binary(OpKind::Mul, a, a);
            Ok(Val::Node(cx.out.push_binary(OpKind::Div, ndu, u2)))
        }
        // d(|u|) = (u/|u|)·u' (NaN at 0, deliberately: the sign is undefined).
        (OpKind::Abs, 1) => {
            let a = p(cx, 0);
            let au = cx.out.push_unary(OpKind::Abs, a);
            let sign = cx.out.push_binary(OpKind::Div, a, au);
            Ok(d_mul(cx, Val::Node(sign), d(0)))
        }
        (OpKind::Sin, 1) => {
            let a = p(cx, 0);
            let c = cx.out.push_unary(OpKind::Cos, a);
            Ok(d_mul(cx, Val::Node(c), d(0)))
        }
        (OpKind::Cos, 1) => {
            let a = p(cx, 0);
            let s = cx.out.push_unary(OpKind::Sin, a);
            let ns = cx.out.push_unary(OpKind::Neg, s);
            Ok(d_mul(cx, Val::Node(ns), d(0)))
        }
        // d(tan u) = u' / cos²(u).
        (OpKind::Tan, 1) => {
            let a = p(cx, 0);
            let du = d(0).id(&mut cx.out);
            let c = cx.out.push_unary(OpKind::Cos, a);
            let c2 = cx.out.push_binary(OpKind::Mul, c, c);
            Ok(Val::Node(cx.out.push_binary(OpKind::Div, du, c2)))
        }
        // d(asin u) = u' / √(1 − u²).
        (OpKind::Asin, 1) => {
            let a = p(cx, 0);
            let du = d(0).id(&mut cx.out);
            let s = sqrt_one_minus_sq(&mut cx.out, a);
            Ok(Val::Node(cx.out.push_binary(OpKind::Div, du, s)))
        }
        // d(acos u) = −u' / √(1 − u²).
        (OpKind::Acos, 1) => {
            let a = p(cx, 0);
            let du = d(0).id(&mut cx.out);
            let s = sqrt_one_minus_sq(&mut cx.out, a);
            let q = cx.out.push_binary(OpKind::Div, du, s);
            Ok(Val::Node(cx.out.push_unary(OpKind::Neg, q)))
        }
        // d(atan u) = u' / (1 + u²).
        (OpKind::Atan, 1) => {
            let a = p(cx, 0);
            let du = d(0).id(&mut cx.out);
            let one = cx.out.push_const(1.0);
            let u2 = cx.out.push_binary(OpKind::Mul, a, a);
            let den = cx.out.push_binary(OpKind::Add, one, u2);
            Ok(Val::Node(cx.out.push_binary(OpKind::Div, du, den)))
        }
        (OpKind::Exp, 1) => {
            let a = p(cx, 0);
            let e = cx.out.push_unary(OpKind::Exp, a);
            Ok(d_mul(cx, Val::Node(e), d(0)))
        }
        // d(2^u) = 2^u·ln2·u'.
        (OpKind::Exp2, 1) => {
            let a = p(cx, 0);
            let e = cx.out.push_unary(OpKind::Exp2, a);
            let ln2 = cx.out.push_const(core::f32::consts::LN_2);
            let factor = cx.out.push_binary(OpKind::Mul, e, ln2);
            Ok(d_mul(cx, Val::Node(factor), d(0)))
        }
        // d(ln u) = u' / u.
        (OpKind::Ln, 1) => {
            let a = p(cx, 0);
            let du = d(0).id(&mut cx.out);
            Ok(Val::Node(cx.out.push_binary(OpKind::Div, du, a)))
        }
        // d(log2 u) = u' / (u·ln2).
        (OpKind::Log2, 1) => {
            let a = p(cx, 0);
            let du = d(0).id(&mut cx.out);
            let ln2 = cx.out.push_const(core::f32::consts::LN_2);
            let den = cx.out.push_binary(OpKind::Mul, a, ln2);
            Ok(Val::Node(cx.out.push_binary(OpKind::Div, du, den)))
        }
        // d(log10 u) = u' / (u·ln10).
        (OpKind::Log10, 1) => {
            let a = p(cx, 0);
            let du = d(0).id(&mut cx.out);
            let ln10 = cx.out.push_const(core::f32::consts::LN_10);
            let den = cx.out.push_binary(OpKind::Mul, a, ln10);
            Ok(Val::Node(cx.out.push_binary(OpKind::Div, du, den)))
        }
        (_, 1) => Err("lower_dwrt: no derivative rule for this unary op"),

        // ───────────────────────────── binary ────────────────────────────
        (OpKind::Add, 2) => Ok(d_add(cx, d(0), d(1))),
        (OpKind::Sub, 2) => Ok(d_sub(cx, d(0), d(1))),
        // Product rule.
        (OpKind::Mul, 2) => {
            let (a, b) = (p(cx, 0), p(cx, 1));
            let t1 = d_mul(cx, d(0), Val::Node(b));
            let t2 = d_mul(cx, Val::Node(a), d(1));
            Ok(d_add(cx, t1, t2))
        }
        // Quotient rule: (a'b − ab') / b².
        (OpKind::Div, 2) => {
            let (a, b) = (p(cx, 0), p(cx, 1));
            let t1 = d_mul(cx, d(0), Val::Node(b));
            let t2 = d_mul(cx, Val::Node(a), d(1));
            let num = d_sub(cx, t1, t2);
            if num.is_zero() {
                return Ok(num);
            }
            let num = num.id(&mut cx.out);
            let den = cx.out.push_binary(OpKind::Mul, b, b);
            Ok(Val::Node(cx.out.push_binary(OpKind::Div, num, den)))
        }
        // Piecewise: the derivative of the branch the primal takes.
        (OpKind::Min | OpKind::Max, 2) => {
            let (a, b) = (p(cx, 0), p(cx, 1));
            let cmp = if op == OpKind::Min {
                OpKind::Lt
            } else {
                OpKind::Gt
            };
            let mask = cx.out.push_binary(cmp, a, b);
            let (da, db) = (d(0).id(&mut cx.out), d(1).id(&mut cx.out));
            Ok(Val::Node(cx.out.push_ternary(OpKind::Select, mask, da, db)))
        }
        // Masks are step functions: zero derivative almost everywhere.
        (OpKind::Lt | OpKind::Le | OpKind::Gt | OpKind::Ge | OpKind::Eq | OpKind::Ne, 2) => {
            Ok(Val::ZERO)
        }
        // d(atan2(y, x)) = (x·y' − y·x') / (x² + y²).
        (OpKind::Atan2, 2) => {
            let (y, x) = (p(cx, 0), p(cx, 1));
            let t1 = d_mul(cx, Val::Node(x), d(0));
            let t2 = d_mul(cx, Val::Node(y), d(1));
            let num = d_sub(cx, t1, t2);
            if num.is_zero() {
                return Ok(num);
            }
            let num = num.id(&mut cx.out);
            let y2 = cx.out.push_binary(OpKind::Mul, y, y);
            let x2 = cx.out.push_binary(OpKind::Mul, x, x);
            let den = cx.out.push_binary(OpKind::Add, x2, y2);
            Ok(Val::Node(cx.out.push_binary(OpKind::Div, num, den)))
        }
        // d(f^g) = f^g · (g'·ln f + g·f'/f).
        (OpKind::Pow, 2) => {
            let (f_, g) = (p(cx, 0), p(cx, 1));
            let lnf = cx.out.push_unary(OpKind::Ln, f_);
            let t1 = d_mul(cx, d(1), Val::Node(lnf));
            let g_over_f = cx.out.push_binary(OpKind::Div, g, f_);
            let t2 = d_mul(cx, Val::Node(g_over_f), d(0));
            let inner = d_add(cx, t1, t2);
            if inner.is_zero() {
                return Ok(inner);
            }
            let inner = inner.id(&mut cx.out);
            let pw = cx.out.push_binary(OpKind::Pow, f_, g);
            Ok(Val::Node(cx.out.push_binary(OpKind::Mul, pw, inner)))
        }
        (OpKind::Dwrt, 2) => Err("lower_dwrt: nested Dwrt survived lowering (internal invariant)"),
        (OpKind::RawGather, 2) => Err("lower_dwrt: cannot differentiate a bound-memory read"),
        (OpKind::IAdd | OpKind::Shl | OpKind::Shr | OpKind::BitAnd | OpKind::BitOr, 2) => {
            Err("lower_dwrt: cannot differentiate integer/bit-manipulation ops")
        }
        (_, 2) => Err("lower_dwrt: no derivative rule for this binary op"),

        // ──────────────────────────── ternary ────────────────────────────
        // d(a·b + c) = a'·b + a·b' + c'.
        (OpKind::MulAdd, 3) => {
            let (a, b) = (p(cx, 0), p(cx, 1));
            let t1 = d_mul(cx, d(0), Val::Node(b));
            let t2 = d_mul(cx, Val::Node(a), d(1));
            let prod = d_add(cx, t1, t2);
            Ok(d_add(cx, prod, d(2)))
        }
        // Blend the branch derivatives on the primal mask.
        (OpKind::Select, 3) => {
            let mask = p(cx, 0);
            let (db, dc) = (d(1).id(&mut cx.out), d(2).id(&mut cx.out));
            Ok(Val::Node(cx.out.push_ternary(OpKind::Select, mask, db, dc)))
        }
        (OpKind::Gather, 3) => Err("lower_dwrt: cannot differentiate a bound-memory read"),
        (_, 3) => Err("lower_dwrt: no derivative rule for this ternary op"),

        // ────────────────────────────── n-ary ────────────────────────────
        _ => Err("lower_dwrt: cannot differentiate an Nary op (Reduce/Tuple)"),
    }
}

/// `√(1 − u²)` — shared by the asin/acos rules.
fn sqrt_one_minus_sq(out: &mut Builder<ExprData>, u: Id) -> Id {
    let one = out.push_const(1.0);
    let u2 = out.push_binary(OpKind::Mul, u, u);
    let t = out.push_binary(OpKind::Sub, one, u2);
    out.push_unary(OpKind::Sqrt, t)
}

// Peephole constructors for derivative arithmetic. See [`Val`].

/// `a + b`, folding the additive identity.
fn d_add(cx: &mut Lowering, a: Val, b: Val) -> Val {
    if a.is_zero() {
        return b;
    }
    if b.is_zero() {
        return a;
    }
    let (a, b) = (a.id(&mut cx.out), b.id(&mut cx.out));
    Val::Node(cx.out.push_binary(OpKind::Add, a, b))
}

/// `a − b`, folding zeros.
fn d_sub(cx: &mut Lowering, a: Val, b: Val) -> Val {
    if b.is_zero() {
        return a;
    }
    if a.is_zero() {
        return d_neg(cx, b);
    }
    let (a, b) = (a.id(&mut cx.out), b.id(&mut cx.out));
    Val::Node(cx.out.push_binary(OpKind::Sub, a, b))
}

/// `a · b`, folding the annihilator and identity.
fn d_mul(cx: &mut Lowering, a: Val, b: Val) -> Val {
    if a.is_zero() || b.is_zero() {
        return Val::ZERO;
    }
    if a.is_one() {
        return b;
    }
    if b.is_one() {
        return a;
    }
    let (a, b) = (a.id(&mut cx.out), b.id(&mut cx.out));
    Val::Node(cx.out.push_binary(OpKind::Mul, a, b))
}

/// `−a`, folding zero.
fn d_neg(cx: &mut Lowering, a: Val) -> Val {
    if a.is_zero() {
        return a;
    }
    let a = a.id(&mut cx.out);
    Val::Node(cx.out.push_unary(OpKind::Neg, a))
}

// ────────────────────────── Transcendental expansions ─────────────────────────

/// Expand a single transcendental unary op applied to (already-lowered) `arg`.
fn expand_unary(out: &mut Builder<ExprData>, op: OpKind, arg: Id) -> Id {
    match op {
        OpKind::Sin => expand_sin(out, arg),
        // cos(x) = sin(x + π/2), with the π/2 applied to the *reduced*
        // argument (see `expand_sin_phase`).
        OpKind::Cos => expand_sin_phase(out, arg, core::f32::consts::FRAC_PI_2),
        // tan(x) = sin(x) / cos(x). Expand both so neither reaches a backend.
        OpKind::Tan => {
            let s = expand_sin(out, arg);
            let c = expand_sin_phase(out, arg, core::f32::consts::FRAC_PI_2);
            out.push_binary(OpKind::Div, s, c)
        }
        OpKind::Exp2 => expand_exp2(out, arg),
        // exp(x) = 2^(x·log2 e)
        OpKind::Exp => {
            let log2e = out.push_const(core::f32::consts::LOG2_E);
            let scaled = out.push_binary(OpKind::Mul, arg, log2e);
            expand_exp2(out, scaled)
        }
        OpKind::Log2 => expand_log2(out, arg),
        // ln(x) = log2(x)·ln 2
        OpKind::Ln => {
            let l = expand_log2(out, arg);
            let ln2 = out.push_const(core::f32::consts::LN_2);
            out.push_binary(OpKind::Mul, l, ln2)
        }
        // log10(x) = log2(x)·log10 2
        OpKind::Log10 => {
            let l = expand_log2(out, arg);
            let log10_2 = out.push_const(core::f32::consts::LOG10_2);
            out.push_binary(OpKind::Mul, l, log10_2)
        }
        // atan(x) = atan2(x, 1)
        OpKind::Atan => {
            let one = out.push_const(1.0);
            expand_atan2(out, arg, one)
        }
        // asin(x) = atan2(x, sqrt(1 - x²))
        OpKind::Asin => {
            let s = sqrt_one_minus_sq(out, arg);
            expand_atan2(out, arg, s)
        }
        // acos(x) = atan2(sqrt(1 - x²), x)
        OpKind::Acos => {
            let s = sqrt_one_minus_sq(out, arg);
            expand_atan2(out, s, arg)
        }
        _ => unreachable!("expand_unary called on non-transcendental {op:?}"),
    }
}

/// Expand a binary transcendental applied to (already-lowered) `a`, `b`.
fn expand_binary(out: &mut Builder<ExprData>, op: OpKind, a: Id, b: Id) -> Id {
    match op {
        OpKind::Atan2 => expand_atan2(out, a, b),
        // pow(a, b) = 2^(b·log2 a) — the same identity the backends' `pow`
        // builtins each implemented by calling their own log2/exp2 bodies.
        // Expanding here is what lets those bodies leave the assemblers.
        OpKind::Pow => {
            let l = expand_log2(out, a);
            let scaled = out.push_binary(OpKind::Mul, b, l);
            expand_exp2(out, scaled)
        }
        _ => unreachable!("expand_binary called on non-transcendental {op:?}"),
    }
}

/// Degree-7 odd **minimax** coefficients for `atan(t)` on `t ∈ [-1, 1]`.
///
/// Max error 8.7e-5. The Taylor coefficients for the same degree
/// (`1, -1/3, 1/5, -1/7`) are 6.2e-2 at `|t| = 1` — 704× worse for exactly the
/// same four multiplies and three adds, because Taylor spends its accuracy
/// budget at the origin while this interval's error is dominated by the
/// endpoint. Fitting the interval instead of the point is free: `atan2` expands
/// to 27 ops either way.
///
/// The fit is over `[0, 1]`, but both `atan` and an odd polynomial are odd, so
/// the error is antisymmetric and the bound carries to `[-1, 1]` unchanged.
pub const ATAN_MINIMAX: [f32; 4] = [0.999_268_04, -0.321_431_33, 0.146_614_41, -0.039_132_48];

/// `atan2(y, x)` (four-quadrant) as a primitive subgraph.
///
/// Reduces to a ratio in [-1,1] (swapping y/x when |y|>|x|), a degree-7 odd
/// polynomial for atan on that interval, then quadrant fix-ups via `Select` on
/// comparison masks.
fn expand_atan2(out: &mut Builder<ExprData>, y: Id, x: Id) -> Id {
    let pi = out.push_const(core::f32::consts::PI);
    let half_pi = out.push_const(core::f32::consts::FRAC_PI_2);
    let zero = out.push_const(0.0);

    let abs_x = out.push_unary(OpKind::Abs, x);
    let abs_y = out.push_unary(OpKind::Abs, y);

    // swap = |y| > |x|; ratio = swap ? x/y : y/x  (keeps |ratio| <= 1).
    let swap = out.push_binary(OpKind::Gt, abs_y, abs_x);
    let recip_y = out.push_unary(OpKind::Recip, y);
    let recip_x = out.push_unary(OpKind::Recip, x);
    let x_over_y = out.push_binary(OpKind::Mul, x, recip_y);
    let y_over_x = out.push_binary(OpKind::Mul, y, recip_x);
    let ratio = out.push_ternary(OpKind::Select, swap, x_over_y, y_over_x);

    // atan(ratio) on [-1,1]: ratio · Horner(c7,c5,c3,c1)(ratio²).
    let r2 = out.push_binary(OpKind::Mul, ratio, ratio);
    let mut p = out.push_const(ATAN_MINIMAX[ATAN_MINIMAX.len() - 1]);
    for &c in ATAN_MINIMAX.iter().rev().skip(1) {
        let c = out.push_const(c);
        p = horner_step(out, p, r2, c);
    }
    let atan_small = out.push_binary(OpKind::Mul, ratio, p);

    // If swapped, result is ±π/2 − atan_small (sign from ratio).
    let ratio_nonneg = out.push_binary(OpKind::Ge, ratio, zero);
    let neg_half_pi = out.push_unary(OpKind::Neg, half_pi);
    let signed_half = out.push_ternary(OpKind::Select, ratio_nonneg, half_pi, neg_half_pi);
    let swapped_val = out.push_binary(OpKind::Sub, signed_half, atan_small);
    let atan_val = out.push_ternary(OpKind::Select, swap, swapped_val, atan_small);

    // Quadrant fix-up: if x < 0, add ±π (sign from y).
    let x_neg = out.push_binary(OpKind::Lt, x, zero);
    let y_neg = out.push_binary(OpKind::Lt, y, zero);
    let neg_pi = out.push_unary(OpKind::Neg, pi);
    let adjust = out.push_ternary(OpKind::Select, y_neg, neg_pi, pi);
    let adjusted = out.push_binary(OpKind::Add, atan_val, adjust);
    out.push_ternary(OpKind::Select, x_neg, adjusted, atan_val)
}

/// `2^x` as a primitive subgraph.
///
/// Split `x = xi + xf` (xi integer, xf ∈ [0,1)); approximate `2^xf` by a
/// degree-5 minimax polynomial; reconstruct `2^xi` by writing the IEEE-754
/// exponent field directly: `2^xi = bitcast((int(xi) + 127) << 23)`. Built from
/// the bit-manip primitives (`TruncToInt`/`IntToFloat`/`IAdd`/`Shl`) — these are
/// the float↔int conversions a backend cannot avoid for exp/log.
fn expand_exp2(out: &mut Builder<ExprData>, arg_x: Id) -> Id {
    // Clamp to a safe exponent range to avoid int overflow / inf.
    let lo = out.push_const(-EXP2_CLAMP);
    let hi = out.push_const(EXP2_CLAMP);
    let x = out.push_binary(OpKind::Max, arg_x, lo);
    let x = out.push_binary(OpKind::Min, x, hi);

    // xi = floor(x), xf = x - xi
    let xi = out.push_unary(OpKind::Floor, x);
    let xf = out.push_binary(OpKind::Sub, x, xi);

    // 2^xf ≈ Horner([`EXP2_POLY`]) at xf, highest degree down.
    let mut p = out.push_const(EXP2_POLY[EXP2_POLY.len() - 1]);
    for &c in EXP2_POLY.iter().rev().skip(1) {
        let c = out.push_const(c);
        p = horner_step(out, p, xf, c);
    }

    // 2^xi = bitcast((int(xi) + 127) << 23).
    let xi_int = out.push_unary(OpKind::TruncToInt, xi);
    let bias = out.push_const(f32::from_bits(127)); // integer 127 as lane bits
    let biased = out.push_binary(OpKind::IAdd, xi_int, bias);
    // Shift amount is read by value (`v as u32 as u8`), so it is a plain 23.0.
    let shift = out.push_const(23.0);
    let pow2i = out.push_binary(OpKind::Shl, biased, shift); // bitcast result

    // 2^x = 2^xf · 2^xi
    let val = out.push_binary(OpKind::Mul, p, pow2i);

    // Outside domain (NaN input), return NaN. Check original arg_x, not clamped x.
    let is_not_nan = out.push_binary(OpKind::Eq, arg_x, arg_x);
    let nan = out.push_const(f32::NAN);
    out.push_ternary(OpKind::Select, is_not_nan, val, nan)
}

/// `log2(x)` as a primitive subgraph (documented domain x > 0).
///
/// Cephes `log2f` algorithm. `log2(x) = e + log2(m)` where `e` is the unbiased
/// exponent and `m ∈ [1,2)` is the mantissa. Extract `e` by shifting the
/// exponent field down; rebuild `m` by masking the mantissa bits and OR-ing in
/// exponent bias 127 (= 1.0). When `m ≥ √2`, halve `m` and bump `e` so the
/// polynomial argument `t = m − 1` stays in `[√2/2 − 1, √2 − 1]` — a degree-4
/// polynomial on the full `[1,2)` peaks at ~0.1 absolute error near `m → 2`.
/// Then `ln(1+t) = t − t²/2 + t³·P(t)` (degree-8 minimax `P`), scaled to base 2
/// via the split constant `log2 e = 1 + LOG2EA` to avoid the rounding from one
/// full-width multiply. Accurate to ~1 ulp over the reduced range.
fn expand_log2(out: &mut Builder<ExprData>, x: Id) -> Id {
    // Reinterpret x's bits as int (free) and extract exponent: e = (bits >> 23) - 127.
    // Shift amount read by value -> plain 23.0.
    let shift23 = out.push_const(23.0);
    let exp_field = out.push_binary(OpKind::Shr, x, shift23); // int lanes
    let exp_f = out.push_unary(OpKind::IntToFloat, exp_field);
    let bias = out.push_const(127.0);
    let e = out.push_binary(OpKind::Sub, exp_f, bias);

    // Mantissa m = bitcast((bits & 0x007FFFFF) | 0x3F800000) ∈ [1, 2).
    let mant_mask = out.push_const(f32::from_bits(0x007F_FFFF));
    let one_bits = out.push_const(f32::from_bits(0x3F80_0000));
    let mant = out.push_binary(OpKind::BitAnd, x, mant_mask);
    let m = out.push_binary(OpKind::BitOr, mant, one_bits);

    // Range-reduce: if m ≥ √2 { m /= 2; e += 1 } so t = m − 1 ∈ [−0.293, 0.414].
    let sqrt2 = out.push_const(core::f32::consts::SQRT_2);
    let reduce = out.push_binary(OpKind::Ge, m, sqrt2);
    let half = out.push_const(0.5);
    let m_halved = out.push_binary(OpKind::Mul, m, half);
    let m = out.push_ternary(OpKind::Select, reduce, m_halved, m);
    let one = out.push_const(1.0);
    let e_bumped = out.push_binary(OpKind::Add, e, one);
    let e = out.push_ternary(OpKind::Select, reduce, e_bumped, e);

    let t = out.push_binary(OpKind::Sub, m, one);

    // P(t): Cephes lnf/log2f degree-8 minimax numerator for
    // (ln(1+t) − t + t²/2) / t³ on the reduced range.
    let mut p = out.push_const(LOG2_POLY[LOG2_POLY.len() - 1]);
    for &c in LOG2_POLY.iter().rev().skip(1) {
        let c = out.push_const(c);
        p = horner_step(out, p, t, c);
    }

    // y = t³·P(t) − t²/2, so ln(1+t) = t + y.
    let t2 = out.push_binary(OpKind::Mul, t, t);
    let t3 = out.push_binary(OpKind::Mul, t2, t);
    let t3p = out.push_binary(OpKind::Mul, t3, p);
    let half_t2 = out.push_binary(OpKind::Mul, t2, half);
    let y = out.push_binary(OpKind::Sub, t3p, half_t2);

    // log2(m) = (t + y)·log2(e), with log2(e) split as 1 + LOG2EA and the
    // pieces summed smallest-first (Cephes ordering) to keep full precision:
    // e + t + y + y·LOG2EA + t·LOG2EA.
    let log2ea = out.push_const(LOG2_E_MINUS_1);
    let y_ea = out.push_binary(OpKind::Mul, y, log2ea);
    let t_ea = out.push_binary(OpKind::Mul, t, log2ea);
    let z = out.push_binary(OpKind::Add, y_ea, t_ea);
    let z = out.push_binary(OpKind::Add, z, y);
    let z = out.push_binary(OpKind::Add, z, t);
    let val = out.push_binary(OpKind::Add, z, e);

    // Documented domain: x > 0.0. Outside domain (x <= 0 or x is NaN), return NaN.
    // Lt(zero, x) is 0 < x (ordered comparison, false for NaN and false for x <= 0).
    let zero = out.push_const(0.0);
    let in_domain = out.push_binary(OpKind::Lt, zero, x);
    let nan = out.push_const(f32::NAN);
    out.push_ternary(OpKind::Select, in_domain, val, nan)
}

/// Largest `|x|` for which `sin`/`cos`/`tan` return a value. Beyond it they
/// return NaN — see [`expand_sin_phase`] for why the boundary is here.
pub const TRIG_DOMAIN: f32 = 1_048_576.0; // 2^20

/// `2π` split so that `k·term` is *exact* for every `k` the domain admits.
///
/// `TAU_HI` is 25·2⁻², `TAU_MID` is 17·2⁻⁹ — 5-bit significands, so the
/// products stay exact until `|k|` reaches 2²⁴/25 ≈ 671089 and 2²⁴/17 ≈ 986895
/// respectively. [`TRIG_DOMAIN`] needs only `|k| ≤ 166887`, a 4× margin.
/// `TAU_LO` carries the remainder at full precision; the three together
/// represent `2π` to within 6.6e-13, so the reduction drifts by at most
/// `166887 · 6.6e-13 ≈ 1.1e-7` radians at the domain edge.
pub const TAU_HI: f32 = 6.25;
pub const TAU_MID: f32 = 0.033_203_125;
pub const TAU_LO: f32 = -1.781_782e-5;

/// Degree-11 odd Chebyshev coefficients for `sin(π·t)` on `t ∈ [-1, 1]`.
///
/// Degree 7 in Taylor coefficients is accurate to 7.5e-2 at the interval ends
/// — two digits — which would make the reduction work below pointless, since
/// the polynomial would then own the entire error budget.
/// Degree 9 is accurate to 6e-6 but peaks at 1.0000029: it *returns values
/// outside sin's range*, which is the defect being fixed here, so it is not an
/// option. Degree 11 is the first that both stays inside `[-1, 1]` and is
/// accurate to 6e-7, near the f32 ulp of a result near 1.
pub const SIN_CHEB: [f32; 6] = [
    3.141_591_3,
    -5.167_677_4,
    2.549_879_3,
    -0.598_278_8,
    0.080_476_06,
    -0.005_990_654,
];

/// `2^f` on `f ∈ [0, 1)`, degree-5 minimax, ascending degree.
///
/// THE definition of the exp2 polynomial. `pixelflow-core`'s backends evaluate
/// this same table so the JIT tier and the `eval_scalar` oracle cannot disagree
/// about what `exp2` is — the divergence this replaced had the backends on a
/// degree-4 fit with different coefficients entirely, which made `exp`, `ln`,
/// `log10` and `pow` compute measurably different functions depending on which
/// tier ran them.
pub const EXP2_POLY: [f32; 6] = [
    1.0,
    core::f32::consts::LN_2,
    0.240_226_5,
    0.055_504_11,
    0.009_618_129,
    0.001_333_355_8,
];

/// Exponent range `exp2` saturates to, rather than overflowing to `inf`.
///
/// `2^n` is built as `bitcast((int(n) + 127) << 23)`, so an unclamped `n` past
/// this walks out of the exponent field and produces a value that is not a
/// power of two at all. CLAUDE.md lists the saturation as behavior every target
/// agrees on; clamping here is what makes that true.
pub const EXP2_CLAMP: f32 = 126.0;

/// Cephes `log2f` degree-8 minimax for `(ln(1+t) − t + t²/2) / t³` on the
/// √2-centered reduced range, ascending degree.
///
/// THE definition of the log2 polynomial, shared with `pixelflow-core`'s
/// backends for the reason [`EXP2_POLY`] gives.
pub const LOG2_POLY: [f32; 9] = [
    3.333_333e-1,
    -2.499_999_4e-1,
    2.000_071_5e-1,
    -1.666_805_8e-1,
    1.424_932_3e-1,
    -1.242_014_1e-1,
    1.167_699_9e-1,
    -1.151_461e-1,
    7.037_683_6e-2,
];

/// `log2(e) − 1`. Cephes splits `log2(e)` this way and sums the pieces
/// smallest-first to keep full precision; see [`LOG2_POLY`]'s use.
pub const LOG2_E_MINUS_1: f32 = 0.442_695_04;

/// `sin(x)` as a primitive subgraph (Chebyshev, matching the runtime path).
fn expand_sin(out: &mut Builder<ExprData>, x: Id) -> Id {
    expand_sin_phase(out, x, 0.0)
}

/// `sin(x + phase)` for a constant `phase`, as a primitive subgraph.
///
/// Range-reduce to `[-π, π]`, normalize to `[-1, 1]`, then [`SIN_CHEB`].
///
/// # Why the argument reduction is three terms and not one
///
/// The obvious reduction — `xx = x − k·2π` with `2π` a single f32 — is wrong
/// for large `x`, and wrong in the worst way: `k·2π` rounds to a multiple of
/// `ulp(x)`, so `xx` inherits an error that grows with `|x|` until the reduced
/// argument leaves `[-π, π]` entirely. Past that point `t` leaves `[-1, 1]`,
/// the polynomial is evaluated outside the interval it was fit on, and the
/// result diverges: `|sin|` first exceeds 1 near `x ≈ 1.4e7` and reaches `inf`
/// by `x ≈ 2.6e13`. Splitting `2π` into [`TAU_HI`]/[`TAU_MID`]/[`TAU_LO`]
/// (Cody-Waite) keeps each `k·term` exact, so the cancellation happens against
/// the true product rather than a rounded one.
///
/// # Why there is a domain limit rather than more terms
///
/// Cody-Waite buys accuracy only while `k` is exactly representable and the
/// products stay exact; extending it to the whole f32 range means Payne-Hanek
/// — a multi-word integer multiply by the bits of `1/2π` — which costs far
/// more than the polynomial it protects and is not worth it in a per-pixel
/// kernel. Past `|x| = 2²⁴` the question stops being meaningful anyway:
/// `ulp(x)` there exceeds 1 radian, so an f32 argument no longer resolves the
/// phase it is asking about.
///
/// So the reduction is honest over [`TRIG_DOMAIN`] (worst case 1.5e-6) and
/// returns NaN outside it. NaN and not a clamp into `[-1, 1]`: a clamped value
/// is a wrong answer that looks like a right one, which is exactly how this
/// defect survived — the JIT and the `eval_scalar` oracle run this same
/// expansion, so they agreed bit-for-bit on the garbage and every same-form
/// equivalence test passed.
///
/// `phase` is folded in *after* reduction rather than added to `x` up front:
/// `cos` is `sin(x + π/2)`, and at the top of the domain `ulp(x)` is 0.0625,
/// so adding π/2 to `x` there would lose most of the shift before reduction
/// ever ran.
///
/// # `sin(-0.0)` is `+0.0`, deliberately
///
/// [`TAU_LO`] is negative, so at `k = 0` the term `k·TAU_LO` is `-0.0` and the
/// last reduction step is `Sub(-0.0, -0.0)`, which is `+0.0`. Reordering the
/// three subtractions does not help — the sign dies wherever the negative
/// constant sits. Making all three positive requires `TAU_MID = 2⁻⁵`, which
/// leaves a coarser remainder and multiplies the reduction's drift by 15
/// (1.7e-6 vs 1.1e-7 at the domain edge) — that would roughly double the
/// function's total error, everywhere, to buy the sign of zero at one point.
/// Not worth it, and squarely the trade CLAUDE.md's "Floating point at the
/// edges" describes: edge-case IEEE conformance is not on offer.
fn expand_sin_phase(out: &mut Builder<ExprData>, x: Id, phase: f32) -> Id {
    use core::f32::consts::{PI, TAU};

    let shift = |out: &mut Builder<ExprData>, v: Id| {
        if phase == 0.0 {
            return v;
        }
        let p = out.push_const(phase);
        out.push_binary(OpKind::Add, v, p)
    };

    // k = floor((x + phase)/2π + 0.5) — the multiple of 2π nearest the
    // argument. An off-by-one in k can only happen when the argument sits on a
    // period boundary, where the two candidate reductions are ±π: the same
    // point, and sin agrees at both.
    let arg = shift(out, x);
    let two_pi_inv = out.push_const(1.0 / TAU);
    let half = out.push_const(0.5);
    let u = out.push_binary(OpKind::Mul, arg, two_pi_inv);
    let u = out.push_binary(OpKind::Add, u, half);
    let k = out.push_unary(OpKind::Floor, u);

    // xx = x − k·2π, in three exact pieces, then the phase back in.
    let hi = out.push_const(TAU_HI);
    let mid = out.push_const(TAU_MID);
    let lo = out.push_const(TAU_LO);
    let k_hi = out.push_binary(OpKind::Mul, k, hi);
    let k_mid = out.push_binary(OpKind::Mul, k, mid);
    let k_lo = out.push_binary(OpKind::Mul, k, lo);
    let xx = out.push_binary(OpKind::Sub, x, k_hi);
    let xx = out.push_binary(OpKind::Sub, xx, k_mid);
    let xx = out.push_binary(OpKind::Sub, xx, k_lo);
    let xx = shift(out, xx);

    // t = xx / π ∈ [-1, 1]. Reduction error can push |t| to ~1.03 at the
    // domain edge; SIN_CHEB still holds |p| ≤ 1 out to |t| = 1.3.
    let pi_inv = out.push_const(1.0 / PI);
    let t = out.push_binary(OpKind::Mul, xx, pi_inv);
    let t2 = out.push_binary(OpKind::Mul, t, t);

    // Horner in t², expanded as mul+add.
    let mut p = out.push_const(SIN_CHEB[SIN_CHEB.len() - 1]);
    for &c in SIN_CHEB.iter().rev().skip(1) {
        let c = out.push_const(c);
        p = horner_step(out, p, t2, c);
    }
    let s = out.push_binary(OpKind::Mul, t, p);

    // Outside the domain, NaN. Guarded on the *unshifted* x so sin, cos and
    // the two halves of tan all agree about where the answer stops existing.
    // NaN itself is unguarded: |NaN| < limit is false, so it propagates.
    let limit = out.push_const(TRIG_DOMAIN);
    let abs_x = out.push_unary(OpKind::Abs, x);
    let in_domain = out.push_binary(OpKind::Lt, abs_x, limit);
    let nan = out.push_const(f32::NAN);
    out.push_ternary(OpKind::Select, in_domain, s, nan)
}

/// `acc·x + add` as one `MulAdd` node.
///
/// Nothing downstream would fuse this for us: [`legalize`] is the last thing to
/// touch the graph, and the only thing that becomes an FMA instruction is an
/// `OpKind::MulAdd` node already in it. So a Horner chain emitted as
/// `Add(Mul(..))` reaches the emitter unfused — 5 mul + 5 add for `sin`, where
/// 5 `MulAdd`s do — and the e-graph cannot recover it either, since it runs
/// *before* this lowering and has no multiplies to fuse yet.
///
/// Fusing here rather than anywhere later is what keeps the trig identities
/// intact. Saturating after expansion would fuse these multiplies, but by then
/// there is no `Sin` node left for `AngleAddition` or `Parity` to match, and
/// the general rewriter turned loose on an expanded polynomial can reassociate
/// Horner form — which is chosen for accuracy, not just for op count.
///
/// # The cost, taken deliberately
///
/// Unfused mul+add rounds twice everywhere, so the `eval_scalar` oracle and
/// every backend agreed bit-for-bit. `MulAdd` rounds **once** where an FMA
/// instruction exists (x86 with `+fma`, aarch64 `FMLA`) and **twice** where it
/// does not (SSE2 baseline: `mulps` + `addps`) — CLAUDE.md, "Floating point at
/// the edges". So the SSE2 tier now differs from the FMA tiers by up to a
/// rounding per Horner step.
///
/// That is a *precision* difference, which the codebase's own rule puts on the
/// table; it is not a range difference, which is not. One rounding is never
/// less accurate than two, so the FMA tiers move toward the true value, not
/// away from it, and the polynomial's range guarantees (`|sin| ≤ 1`, the
/// `TRIG_DOMAIN` NaN edge) are unaffected — they come from the reduction and
/// the `Select`, neither of which is a Horner step.
fn horner_step(out: &mut Builder<ExprData>, acc: Id, x: Id, add: Id) -> Id {
    out.push_ternary(OpKind::MulAdd, acc, x, add)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binding::BindingTable;
    use crate::decl::{BufferDecl, BufferIdentity, UniformDecl, UniformIdentity};
    use crate::eval::eval_scalar;
    use crate::expr::{Environment, ExprBuilder, ExprRef};
    use alloc::vec::Vec;

    /// A frozen graph plus its environment, with every node a test wants to
    /// name kept as an entry point. A `Term` borrows both halves, so they have
    /// to outlive it — which is what this owns.
    struct Graph {
        rooted: Rooted<ExprData>,
        env: Environment,
    }

    impl Graph {
        fn term(&self, i: usize) -> Term<'_> {
            Term::new(self.rooted.entry_at(i), &self.env)
        }

        fn eval(&self, i: usize, pt: &[f32; 2]) -> f32 {
            eval_scalar(self.term(i), pt, &BindingTable::empty())
        }

        /// Evaluate a lowered graph against this one's environment — every
        /// pass here preserves the declaration slots.
        fn eval_lowered(&self, out: &Rooted<ExprData>, pt: &[f32; 2]) -> f32 {
            eval_scalar(
                Term::new(out.entry(), &self.env),
                pt,
                &BindingTable::empty(),
            )
        }
    }

    fn freeze(b: ExprBuilder, roots: &[ExprRef]) -> Graph {
        let (rooted, env) = b.finish(roots);
        Graph { rooted, env }
    }

    fn buffer(b: &mut ExprBuilder, width: u32, height: u32) -> crate::decl::BufferId {
        b.declare_buffer(BufferDecl {
            id: BufferIdentity::mint(),
            width,
            height,
        })
    }

    /// Build `Dwrt(expr, var)` over whatever `build` produces, lower it, and
    /// assert no `Dwrt` survives.
    fn lowered_derivative(build: impl FnOnce(&mut ExprBuilder) -> ExprRef, var: u8) -> Graph {
        let mut b = ExprBuilder::new();
        let e = build(&mut b);
        let v = b.push_const(f32::from(var));
        let root = b.push_binary(OpKind::Dwrt, e, v);
        let g = freeze(b, &[root]);
        let out = lower_dwrt(g.term(0)).expect("lower_dwrt");
        assert!(
            !out.entry()
                .descendants()
                .any(|n| *n == ExprData::Op(OpKind::Dwrt)),
            "lowered derivative still contains a reachable Dwrt",
        );
        Graph {
            rooted: out,
            env: g.env,
        }
    }

    fn assert_close(got: f32, want: f32, pt: &[f32; 2]) {
        assert_close_rel(got, want, pt, 1e-3);
    }

    /// Relative tolerance the caller chooses.
    ///
    /// Derivatives of transcendentals need a looser bound than exact
    /// arithmetic: the interpreter evaluates the language's own polynomial
    /// expansion (not the host libm — see `eval_scalar`), so comparing against
    /// `f32::cos` measures the derivative rule *and* the approximation's own
    /// error. That error is the language's actual answer, and pinning it here
    /// is the point: the tolerance documents the approximation instead of
    /// hiding it behind an exact host function.
    fn assert_close_rel(got: f32, want: f32, pt: &[f32; 2], rel: f32) {
        let tol = rel * want.abs().max(1.0);
        assert!(
            (got - want).abs() <= tol,
            "at {pt:?}: got {got}, want {want} (tol {tol})"
        );
    }

    // ───────────────────────────── Dwrt ─────────────────────────────

    #[test]
    fn differentiate_a_variable_to_one_for_itself_and_zero_for_the_others() {
        let g = lowered_derivative(|b| b.push_var(0), 0);
        assert_close(g.eval(0, &[3.0, 5.0]), 1.0, &[3.0, 5.0]);

        let g = lowered_derivative(|b| b.push_var(1), 0);
        assert_close(g.eval(0, &[3.0, 5.0]), 0.0, &[3.0, 5.0]);
    }

    #[test]
    fn compose_the_sqrt_rule_with_the_chain_rule_over_a_sum_of_squares() {
        // d/dx √(x² + y²) = x / √(x² + y²) — the font-SDF core.
        let g = lowered_derivative(
            |b| {
                let x = b.push_var(0);
                let y = b.push_var(1);
                let x2 = b.push_binary(OpKind::Mul, x, x);
                let y2 = b.push_binary(OpKind::Mul, y, y);
                let sum = b.push_binary(OpKind::Add, x2, y2);
                b.push_unary(OpKind::Sqrt, sum)
            },
            0,
        );
        for p in &[[3.0f32, 4.0], [1.0, 1.0], [-2.0, 5.0]] {
            let want = p[0] / (p[0] * p[0] + p[1] * p[1]).sqrt();
            assert_close(g.eval(0, p), want, p);
        }
    }

    #[test]
    fn take_the_derivative_of_whichever_branch_min_and_max_select() {
        // d/dx min(x·2, y·3) is 2 where x·2 < y·3, else 0 (and dually for max).
        for (op, at_small_x, at_large_x) in [(OpKind::Min, 2.0, 0.0), (OpKind::Max, 0.0, 2.0)] {
            let g = lowered_derivative(
                |b| {
                    let x = b.push_var(0);
                    let y = b.push_var(1);
                    let two = b.push_const(2.0);
                    let three = b.push_const(3.0);
                    let x2 = b.push_binary(OpKind::Mul, x, two);
                    let y3 = b.push_binary(OpKind::Mul, y, three);
                    b.push_binary(op, x2, y3)
                },
                0,
            );
            assert_close(g.eval(0, &[1.0, 5.0]), at_small_x, &[1.0, 5.0]);
            assert_close(g.eval(0, &[9.0, 1.0]), at_large_x, &[9.0, 1.0]);
        }
    }

    #[test]
    fn blend_the_branches_derivatives_by_the_same_mask_select_used() {
        // d/dx select(y > 0, x·x, x·5) = 2x above the axis, 5 below.
        let g = lowered_derivative(
            |b| {
                let x = b.push_var(0);
                let y = b.push_var(1);
                let zero = b.push_const(0.0);
                let five = b.push_const(5.0);
                let mask = b.push_binary(OpKind::Gt, y, zero);
                let xx = b.push_binary(OpKind::Mul, x, x);
                let x5 = b.push_binary(OpKind::Mul, x, five);
                b.push_ternary(OpKind::Select, mask, xx, x5)
            },
            0,
        );
        assert_close(g.eval(0, &[3.0, 1.0]), 6.0, &[3.0, 1.0]);
        assert_close(g.eval(0, &[3.0, -1.0]), 5.0, &[3.0, -1.0]);
    }

    #[test]
    fn give_a_clamped_expression_zero_derivative_outside_its_bounds() {
        // d/dx clamp(x·x, 0, 10): 2x inside, 0 once saturated. `clamp` is
        // library, so this is the min/max composition and the derivative comes
        // from the min/max rules — no clamp-specific rule exists any more.
        let g = lowered_derivative(
            |b| {
                let x = b.push_var(0);
                let zero = b.push_const(0.0);
                let ten = b.push_const(10.0);
                let xx = b.push_binary(OpKind::Mul, x, x);
                let floored = b.push_binary(OpKind::Max, xx, zero);
                b.push_binary(OpKind::Min, floored, ten)
            },
            0,
        );
        assert_close(g.eval(0, &[2.0, 0.0]), 4.0, &[2.0, 0.0]);
        assert_close(g.eval(0, &[5.0, 0.0]), 0.0, &[5.0, 0.0]);
    }

    #[test]
    fn differentiate_mul_add_by_the_product_rule_plus_the_addends_derivative() {
        // d/dx (x·y + x) = y + 1.
        let g = lowered_derivative(
            |b| {
                let x = b.push_var(0);
                let y = b.push_var(1);
                b.push_ternary(OpKind::MulAdd, x, y, x)
            },
            0,
        );
        for p in &[[2.0f32, 3.0], [-1.0, 7.0]] {
            assert_close(g.eval(0, p), p[1] + 1.0, p);
        }
    }

    #[test]
    fn differentiate_sin_exp_and_ln_to_their_own_rules_under_composition() {
        // d/dx sin(x) = cos(x); d/dx exp(x·x) = 2x·exp(x²); d/dx ln(x) = 1/x.
        //
        // The expected value is built as an expression and evaluated the same
        // way, NOT taken from the host libm. That isolates what this test is
        // for: the derivative *rule*. `cos` in this language is the expansion
        // `sin(x + π/2)`, whose polynomial degrades as the shifted argument
        // approaches π — comparing against `f32::cos` would charge that
        // approximation error to the chain rule and force a tolerance loose
        // enough to hide a real rule bug.
        let pts = [[0.7f32, 0.0], [1.3, 0.0]];

        // Entry 0 is the derivative's input, entry 1 the oracle expression.
        let mut b = ExprBuilder::new();
        let x = b.push_var(0);
        let s = b.push_unary(OpKind::Sin, x);
        let zero = b.push_const(0.0);
        let dwrt = b.push_binary(OpKind::Dwrt, s, zero);
        let expected_cos = b.push_unary(OpKind::Cos, x);
        let g = freeze(b, &[dwrt, expected_cos]);
        let out = lower_dwrt(g.term(0)).expect("lower_dwrt");
        for p in &pts {
            assert_close(g.eval_lowered(&out, p), g.eval(1, p), p);
        }

        // 2x·exp(x²), in the language.
        let mut b = ExprBuilder::new();
        let x = b.push_var(0);
        let xx = b.push_binary(OpKind::Mul, x, x);
        let e = b.push_unary(OpKind::Exp, xx);
        let zero = b.push_const(0.0);
        let dwrt = b.push_binary(OpKind::Dwrt, e, zero);
        let two = b.push_const(2.0);
        let two_x = b.push_binary(OpKind::Mul, two, x);
        let exp_xx = b.push_unary(OpKind::Exp, xx);
        let expected = b.push_binary(OpKind::Mul, two_x, exp_xx);
        let g = freeze(b, &[dwrt, expected]);
        let out = lower_dwrt(g.term(0)).expect("lower_dwrt");
        for p in &pts {
            assert_close(g.eval_lowered(&out, p), g.eval(1, p), p);
        }

        let g = lowered_derivative(
            |b| {
                let x = b.push_var(0);
                b.push_unary(OpKind::Ln, x)
            },
            0,
        );
        for p in &pts {
            assert_close(g.eval(0, p), 1.0 / p[0], p);
        }
    }

    #[test]
    fn nested_dwrt_is_second_derivative() {
        // d²/dx² (x·x·x) = 6x, via Dwrt(Dwrt(x³, 0), 0). The inner derivative
        // resolves first, and the outer one differentiates its lowered form —
        // which is why `lower_dwrt` iterates rather than making one pass.
        let mut b = ExprBuilder::new();
        let x = b.push_var(0);
        let xx = b.push_binary(OpKind::Mul, x, x);
        let xxx = b.push_binary(OpKind::Mul, xx, x);
        let v0 = b.push_const(0.0);
        let d1 = b.push_binary(OpKind::Dwrt, xxx, v0);
        let root = b.push_binary(OpKind::Dwrt, d1, v0);
        let g = freeze(b, &[root]);
        let out = lower_dwrt(g.term(0)).expect("lower_dwrt");
        assert!(
            !out.entry()
                .descendants()
                .any(|n| *n == ExprData::Op(OpKind::Dwrt))
        );
        for p in &[[2.0f32, 0.0], [-1.5, 0.0]] {
            assert_close(g.eval_lowered(&out, p), 6.0 * p[0], p);
        }
    }

    #[test]
    fn shared_subgraph_differentiates_once() {
        // A DAG: s = x·y used twice. The derivative must stay a DAG (no
        // exponential blowup) and be correct: d/dx (s·s) = 2·s·y.
        let g = lowered_derivative(
            |b| {
                let x = b.push_var(0);
                let y = b.push_var(1);
                let s = b.push_binary(OpKind::Mul, x, y);
                b.push_binary(OpKind::Mul, s, s)
            },
            0,
        );
        let p = [3.0f32, 2.0];
        assert_close(g.eval(0, &p), 2.0 * (p[0] * p[1]) * p[1], &p);
    }

    #[test]
    fn no_dwrt_is_identity() {
        let mut b = ExprBuilder::new();
        let x = b.push_var(0);
        let y = b.push_var(1);
        let e = b.push_binary(OpKind::Add, x, y);
        let g = freeze(b, &[e]);
        let out = lower_dwrt(g.term(0)).expect("lower_dwrt");
        assert_eq!(out.len(), g.term(0).root().node_count());
        assert!(out.entry().subtree_eq(g.term(0).root()));
    }

    /// A pathologically deep expression must lower without stack overflow —
    /// the rebuild, the differentiation and every traversal they use are
    /// iterative.
    #[test]
    fn deep_chain_does_not_overflow_the_stack() {
        let mut b = ExprBuilder::new();
        let x = b.push_var(0);
        let one = b.push_const(1.0);
        let mut e = x;
        for i in 0..100_000u32 {
            e = match i % 3 {
                0 => b.push_binary(OpKind::Add, e, one),
                1 => b.push_binary(OpKind::Mul, e, x),
                _ => b.push_unary(OpKind::Sqrt, e),
            };
        }
        let v0 = b.push_const(0.0);
        let root = b.push_binary(OpKind::Dwrt, e, v0);
        let g = freeze(b, &[root]);
        let out = lower_dwrt(g.term(0)).expect("lower_dwrt");
        assert!(out.len() > 100_000);
    }

    #[test]
    fn unsupported_op_errors_loudly() {
        // Differentiating a Reduce has no rule: the pass must refuse.
        let mut b = ExprBuilder::new();
        let body = b.push_var(4);
        let red = b.push_reduce(OpKind::Add, 4, 4, body);
        let v0 = b.push_const(0.0);
        let root = b.push_binary(OpKind::Dwrt, red, v0);
        let g = freeze(b, &[root]);
        assert!(lower_dwrt(g.term(0)).is_err());
    }

    #[test]
    fn flip_the_sign_for_neg_and_square_the_denominator_for_recip() {
        // d/dx -(x·x) = -2x.
        let g = lowered_derivative(
            |b| {
                let x = b.push_var(0);
                let xx = b.push_binary(OpKind::Mul, x, x);
                b.push_unary(OpKind::Neg, xx)
            },
            0,
        );
        for p in &[[3.0f32, 0.0], [-2.0, 0.0]] {
            assert_close(g.eval(0, p), -2.0 * p[0], p);
        }

        // d/dx (1/x) = -1/x².
        let g = lowered_derivative(
            |b| {
                let x = b.push_var(0);
                b.push_unary(OpKind::Recip, x)
            },
            0,
        );
        for p in &[[2.0f32, 0.0], [-4.0, 0.0]] {
            assert_close(g.eval(0, p), -1.0 / (p[0] * p[0]), p);
        }
    }

    #[test]
    fn differentiate_abs_to_the_sign_of_its_operand() {
        // d/dx |x| = x/|x| — +1 above zero, -1 below.
        let g = lowered_derivative(
            |b| {
                let x = b.push_var(0);
                b.push_unary(OpKind::Abs, x)
            },
            0,
        );
        assert_close(g.eval(0, &[3.0, 0.0]), 1.0, &[3.0, 0.0]);
        assert_close(g.eval(0, &[-3.0, 0.0]), -1.0, &[-3.0, 0.0]);
    }

    #[test]
    fn differentiate_rsqrt_to_minus_half_x_to_the_negative_three_halves() {
        // d/dx x^(-1/2) = -0.5 · x^(-3/2).
        let g = lowered_derivative(
            |b| {
                let x = b.push_var(0);
                b.push_unary(OpKind::Rsqrt, x)
            },
            0,
        );
        for xv in [4.0f32, 9.0] {
            let want = -0.5 * xv.powf(-1.5);
            let p = [xv, 0.0];
            assert_close(g.eval(0, &p), want, &p);
        }
    }

    #[test]
    fn match_the_closed_forms_for_the_remaining_trig_and_inverse_trig_rules() {
        // Expected values for the transcendental cases are built as expressions
        // and evaluated the same way as the derivative under test, NOT taken
        // from host libm — see
        // `differentiate_sin_exp_and_ln_to_their_own_rules_under_composition`.
        let pt = [0.4f32, 0.0];

        // d/dx cos(x) = -sin(x).
        let mut b = ExprBuilder::new();
        let x = b.push_var(0);
        let e = b.push_unary(OpKind::Cos, x);
        let zero = b.push_const(0.0);
        let dwrt = b.push_binary(OpKind::Dwrt, e, zero);
        let sinx = b.push_unary(OpKind::Sin, x);
        let expected = b.push_unary(OpKind::Neg, sinx);
        let g = freeze(b, &[dwrt, expected]);
        let out = lower_dwrt(g.term(0)).expect("lower_dwrt");
        assert_close(g.eval_lowered(&out, &pt), g.eval(1, &pt), &pt);

        // d/dx tan(x) = 1/cos²(x).
        let mut b = ExprBuilder::new();
        let x = b.push_var(0);
        let e = b.push_unary(OpKind::Tan, x);
        let zero = b.push_const(0.0);
        let dwrt = b.push_binary(OpKind::Dwrt, e, zero);
        let cosx = b.push_unary(OpKind::Cos, x);
        let cos2 = b.push_binary(OpKind::Mul, cosx, cosx);
        let one = b.push_const(1.0);
        let expected = b.push_binary(OpKind::Div, one, cos2);
        let g = freeze(b, &[dwrt, expected]);
        let out = lower_dwrt(g.term(0)).expect("lower_dwrt");
        assert_close(g.eval_lowered(&out, &pt), g.eval(1, &pt), &pt);

        // d/dx asin(x) = 1/√(1-x²); d/dx acos(x) = -that. √ and arithmetic
        // are exact in this interpreter, so a closed form is fine here.
        let want = 1.0 / (1.0 - pt[0] * pt[0]).sqrt();
        let g = lowered_derivative(
            |b| {
                let x = b.push_var(0);
                b.push_unary(OpKind::Asin, x)
            },
            0,
        );
        assert_close(g.eval(0, &pt), want, &pt);
        let g = lowered_derivative(
            |b| {
                let x = b.push_var(0);
                b.push_unary(OpKind::Acos, x)
            },
            0,
        );
        assert_close(g.eval(0, &pt), -want, &pt);

        // d/dx atan(x) = 1/(1+x²) — pure arithmetic, no transcendental in
        // the derivative expression itself.
        let g = lowered_derivative(
            |b| {
                let x = b.push_var(0);
                b.push_unary(OpKind::Atan, x)
            },
            0,
        );
        for xv in [0.5f32, 2.0, -3.0] {
            let p = [xv, 0.0];
            assert_close(g.eval(0, &p), 1.0 / (1.0 + xv * xv), &p);
        }
    }

    #[test]
    fn carry_the_right_constants_in_the_base_two_and_base_ten_exp_and_log_rules() {
        // d/dx 2^x = 2^x · ln2.
        let mut b = ExprBuilder::new();
        let x = b.push_var(0);
        let e = b.push_unary(OpKind::Exp2, x);
        let zero = b.push_const(0.0);
        let dwrt = b.push_binary(OpKind::Dwrt, e, zero);
        let exp2x = b.push_unary(OpKind::Exp2, x);
        let ln2 = b.push_const(core::f32::consts::LN_2);
        let expected = b.push_binary(OpKind::Mul, exp2x, ln2);
        let g = freeze(b, &[dwrt, expected]);
        let out = lower_dwrt(g.term(0)).expect("lower_dwrt");
        for xv in [0.3f32, 2.0, -1.0] {
            let p = [xv, 0.0];
            assert_close(g.eval_lowered(&out, &p), g.eval(1, &p), &p);
        }

        // d/dx log2(x) = 1/(x·ln2) — pure arithmetic given ln2 is a constant.
        for (op, base_ln) in [
            (OpKind::Log2, core::f32::consts::LN_2),
            (OpKind::Log10, core::f32::consts::LN_10),
        ] {
            let g = lowered_derivative(
                |b| {
                    let x = b.push_var(0);
                    b.push_unary(op, x)
                },
                0,
            );
            for xv in [0.5f32, 3.0] {
                let p = [xv, 0.0];
                assert_close(g.eval(0, &p), 1.0 / (xv * base_ln), &p);
            }
        }
    }

    #[test]
    fn negate_subs_right_derivative_and_follow_the_quotient_rule_for_div() {
        // d/dx (x·x - x) = 2x - 1. Both operands must depend on x: with a
        // constant-in-x right operand `db` is zero, and `da - db` and
        // `da + db` agree — the sign of Sub's right term would go unpinned.
        let g = lowered_derivative(
            |b| {
                let x = b.push_var(0);
                let xx = b.push_binary(OpKind::Mul, x, x);
                b.push_binary(OpKind::Sub, xx, x)
            },
            0,
        );
        let p = [3.0f32, 5.0];
        assert_close(g.eval(0, &p), 2.0 * p[0] - 1.0, &p);

        // d/dx (x·x / (x + y)) = (2x(x+y) - x²)/(x+y)² — the full quotient
        // rule. Same reason: an x-independent denominator makes `db` zero and
        // collapses the rule to `da / b`, so the `-a·db` term could be
        // deleted outright and this would still pass.
        let g = lowered_derivative(
            |b| {
                let x = b.push_var(0);
                let y = b.push_var(1);
                let xx = b.push_binary(OpKind::Mul, x, x);
                let denom = b.push_binary(OpKind::Add, x, y);
                b.push_binary(OpKind::Div, xx, denom)
            },
            0,
        );
        let p = [3.0f32, 2.0];
        let (xv, den) = (p[0], p[0] + p[1]);
        assert_close(g.eval(0, &p), (2.0 * xv * den - xv * xv) / (den * den), &p);
    }

    #[test]
    fn differentiate_atan2_and_pow_through_both_of_their_operands() {
        // d/dX atan2(Y, X) = (X·dY - Y·dX)/(X²+Y²) = -Y/(X²+Y²), since Y does
        // not depend on X. `Atan2`'s children are (y, x), matching `f32::atan2`.
        let g = lowered_derivative(
            |b| {
                let x = b.push_var(0);
                let y = b.push_var(1);
                b.push_binary(OpKind::Atan2, y, x)
            },
            0,
        );
        let p = [3.0f32, 4.0];
        assert_close(g.eval(0, &p), -p[1] / (p[0] * p[0] + p[1] * p[1]), &p);

        // Both Atan2 children depending on X, so the `x·dy` half of
        // (x·dy - y·dx)/(x²+y²) is exercised too — with `dy == 0` above, that
        // whole term could be deleted and the assertion would not notice.
        // d/dX atan2(X², X) = (X·2X - X²)/(X⁴ + X²) = X²/(X⁴ + X²).
        let g = lowered_derivative(
            |b| {
                let x = b.push_var(0);
                let xx = b.push_binary(OpKind::Mul, x, x);
                b.push_binary(OpKind::Atan2, xx, x)
            },
            0,
        );
        let p = [3.0f32, 0.0];
        let xv = p[0];
        assert_close(g.eval(0, &p), (xv * xv) / (xv * xv * xv * xv + xv * xv), &p);

        // d/dx x³ (constant exponent) = 3x², the ordinary power rule falling
        // out of Pow's general f^g·(g'·ln f + g·f'/f) formula.
        let g = lowered_derivative(
            |b| {
                let x = b.push_var(0);
                let three = b.push_const(3.0);
                b.push_binary(OpKind::Pow, x, three)
            },
            0,
        );
        let p = [2.0f32, 0.0];
        assert_close(g.eval(0, &p), 3.0 * p[0] * p[0], &p);

        // A constant exponent leaves `dg` zero, so the general rule's
        // `g'·ln(f)` term is unexercised above and could be deleted. With the
        // exponent varying too: d/dx x^x = x^x·(ln x + 1).
        let g = lowered_derivative(
            |b| {
                let x = b.push_var(0);
                b.push_binary(OpKind::Pow, x, x)
            },
            0,
        );
        let xv = 2.0f32;
        let p = [xv, 0.0];
        let want = libm::powf(xv, xv) * (libm::logf(xv) + 1.0);
        // Pow expands through exp/ln polynomial fits, so hold this to the
        // same looser relative tolerance the expansions are checked at.
        assert_close_rel(g.eval(0, &p), want, &p, 3e-2);
    }

    #[test]
    fn multiply_every_unary_rule_by_its_operands_derivative() {
        // Each rule above applies its op to `Var(0)` directly, where the chain
        // rule's `da` factor is exactly 1 — so a rule that dropped or miswired
        // `da` would still pass every one of them. Here each op wraps `x·x`,
        // whose derivative is `2x`, which makes that factor observable.
        //
        // The oracle is the same rule evaluated one level up: `d/du f(u)` at
        // `u = x²`, times `2x`. That deliberately does not re-derive f' by
        // hand — this test is about the chain rule's factor, and reusing the
        // rule for `f'` keeps a polynomial's accuracy out of the comparison.
        //
        // `x = 0.6` puts `x² = 0.36` inside every domain at once: within
        // [-1, 1] for Asin/Acos, strictly positive for the logs, and nonzero
        // for Recip/Rsqrt.
        const X: f32 = 0.6;
        let outer = [X, 0.0];
        let inner = [X * X, 0.0];

        for op in [
            OpKind::Sin,
            OpKind::Cos,
            OpKind::Tan,
            OpKind::Asin,
            OpKind::Acos,
            OpKind::Atan,
            OpKind::Exp,
            OpKind::Exp2,
            OpKind::Ln,
            OpKind::Log2,
            OpKind::Log10,
            OpKind::Sqrt,
            OpKind::Rsqrt,
            OpKind::Recip,
            OpKind::Neg,
            OpKind::Abs,
        ] {
            // d/dx f(x²)
            let composed = lowered_derivative(
                |b| {
                    let x = b.push_var(0);
                    let xx = b.push_binary(OpKind::Mul, x, x);
                    b.push_unary(op, xx)
                },
                0,
            );
            let got = composed.eval(0, &outer);

            // f'(u) at u = x², from the same rule with a unit-derivative
            // operand — the case the tests above already cover.
            let bare = lowered_derivative(
                |b| {
                    let u = b.push_var(0);
                    b.push_unary(op, u)
                },
                0,
            );
            let want = bare.eval(0, &inner) * 2.0 * X;

            // Guard against a vacuous comparison: if `f'(x²)·2x` happened to
            // land on zero, dropping `da` entirely would also produce zero.
            assert!(
                want.abs() > 1e-3,
                "{op:?}: oracle {want} is too near zero at x={X} to distinguish \
                 a present chain-rule factor from a missing one"
            );
            assert_close(got, want, &outer);
        }
    }

    #[test]
    fn differentiate_a_raw_comparison_of_any_kind_to_zero() {
        // A bare comparison (not wrapped in a Select) is a step function:
        // zero derivative, and — unlike an op with no rule at all —
        // `lower_dwrt` must succeed rather than error.
        //
        // All six are separate alternatives in `diff_node`'s and
        // `push_deriv_children`'s grouped matches, so covering only `Lt` would
        // let a dropped or misrouted arm for any of the other five through.
        for op in [
            OpKind::Lt,
            OpKind::Le,
            OpKind::Gt,
            OpKind::Ge,
            OpKind::Eq,
            OpKind::Ne,
        ] {
            let g = lowered_derivative(
                |b| {
                    let x = b.push_var(0);
                    let y = b.push_var(1);
                    b.push_binary(op, x, y)
                },
                0,
            );
            // Both orderings and equality, so no arm can pass by accident of
            // the operands it was handed.
            for p in &[[1.0f32, 2.0], [2.0f32, 1.0], [1.0f32, 1.0]] {
                assert_close(g.eval(0, p), 0.0, p);
            }
        }
    }

    /// `is_err()` alone can't tell a specific "no rule for this op" message
    /// apart from the generic per-arity fallback, since both are `Err`. Assert
    /// the exact message everywhere below so a deleted specific-op arm — which
    /// falls through to the generic one — is observable.
    #[test]
    fn lower_dwrt_refuses_integer_domain_and_raw_memory_ops() {
        const BOUND_MEMORY: &str = "lower_dwrt: cannot differentiate a bound-memory read";
        const INT_BIT: &str = "lower_dwrt: cannot differentiate integer/bit-manipulation ops";

        /// Wrap a Gather (itself undifferentiable) in `op`. If
        /// `push_deriv_children`'s integer/bit arm wrongly marked the operand
        /// as needing a derivative, the child's `BOUND_MEMORY` error would
        /// surface instead of the arm's own message.
        fn over_a_gather(op: OpKind) -> Result<Rooted<ExprData>, &'static str> {
            let mut b = ExprBuilder::new();
            let buf = buffer(&mut b, 2, 1);
            let gx = b.push_var(0);
            let zero = b.push_const(0.0);
            let g = b.push_gather(buf, gx, zero);
            let e = b.push_unary(op, g);
            let v0 = b.push_const(0.0);
            let root = b.push_binary(OpKind::Dwrt, e, v0);
            let graph = freeze(b, &[root]);
            lower_dwrt(graph.term(0))
        }

        for op in [OpKind::TruncToInt, OpKind::IntToFloat] {
            assert_eq!(over_a_gather(op).err(), Some(INT_BIT), "for {op:?}");
        }

        // IAdd/Shl/Shr/BitAnd/BitOr: integer/bit-manipulation primitives, at
        // the binary-op level. Each is its own alternative in the grouped arm,
        // so testing only `IAdd` would let any of the other four fall through
        // to the generic per-arity fallback — a different message, or worse, a
        // derivative — while this test still passed.
        for op in [
            OpKind::IAdd,
            OpKind::Shl,
            OpKind::Shr,
            OpKind::BitAnd,
            OpKind::BitOr,
        ] {
            let mut b = ExprBuilder::new();
            let x = b.push_var(0);
            let y = b.push_var(1);
            let e = b.push_binary(op, x, y);
            let v0 = b.push_const(0.0);
            let root = b.push_binary(OpKind::Dwrt, e, v0);
            let g = freeze(b, &[root]);
            assert_eq!(lower_dwrt(g.term(0)).err(), Some(INT_BIT), "for {op:?}");
        }

        // A bare Gather (bound-memory read) cannot be differentiated, and
        // neither can its lowered RawGather form.
        let mut b = ExprBuilder::new();
        let buf = buffer(&mut b, 2, 1);
        let gx = b.push_var(0);
        let zero = b.push_const(0.0);
        let gather = b.push_gather(buf, gx, zero);
        let v0 = b.push_const(0.0);
        let root = b.push_binary(OpKind::Dwrt, gather, v0);
        let g = freeze(b, &[root, gather]);
        assert_eq!(lower_dwrt(g.term(0)).err(), Some(BOUND_MEMORY));

        let raw = expand_gather(g.term(1));
        let mut b = ExprBuilder::new();
        let spliced = b.splice(Term::new(raw.entry(), &g.env));
        let v0 = b.push_const(0.0);
        let root = b.push_binary(OpKind::Dwrt, spliced, v0);
        let g2 = freeze(b, &[root]);
        assert_eq!(lower_dwrt(g2.term(0)).err(), Some(BOUND_MEMORY));
    }

    #[test]
    fn differentiate_floor_ceil_and_round_to_zero_without_touching_their_operand() {
        // Floor/Ceil/Round are step functions: zero derivative, and — unlike
        // every other unary rule — the rule never reads the operand's own
        // derivative. Wrapping a Gather (itself undifferentiable) proves
        // that: if `push_deriv_children` wrongly marked the operand as
        // needing a derivative, the Gather's error would surface and this
        // would fail to lower at all instead of yielding 0.
        for op in [OpKind::Floor, OpKind::Ceil, OpKind::Round] {
            let g = lowered_derivative(
                |b| {
                    let buf = buffer(b, 2, 1);
                    let gx = b.push_var(0);
                    let zero = b.push_const(0.0);
                    let gathered = b.push_gather(buf, gx, zero);
                    b.push_unary(op, gathered)
                },
                0,
            );
            let p = [0.0f32, 0.0];
            let bound = BindingTable::bind(&g.env, &[&[1.0f32, 2.0][..]]).expect("bind");
            assert_close(
                eval_scalar(Term::new(g.rooted.entry(), &g.env), &p, &bound),
                0.0,
                &p,
            );
        }
    }

    #[test]
    fn lower_dwrt_refuses_a_malformed_dwrt_shape() {
        // `Dwrt` is only well-formed as a binary node; any other arity is a
        // malformed node the pass must refuse outright, not silently
        // reinterpret.
        const MALFORMED: &str = "lower_dwrt: malformed Dwrt node (must be Binary(expr, var))";

        let mut b = ExprBuilder::new();
        let x = b.push_var(0);
        let root = b.push_unary(OpKind::Dwrt, x);
        let g = freeze(b, &[root]);
        assert_eq!(lower_dwrt(g.term(0)).err(), Some(MALFORMED));

        let mut b = ExprBuilder::new();
        let x = b.push_var(0);
        let y = b.push_var(1);
        let z = b.push_const(0.0);
        let root = b.push_ternary(OpKind::Dwrt, x, y, z);
        let g = freeze(b, &[root]);
        assert_eq!(lower_dwrt(g.term(0)).err(), Some(MALFORMED));

        let mut b = ExprBuilder::new();
        let x = b.push_var(0);
        let y = b.push_var(1);
        let root = b.push_nary(OpKind::Dwrt, &[x, y, x, y]);
        let g = freeze(b, &[root]);
        assert_eq!(lower_dwrt(g.term(0)).err(), Some(MALFORMED));
    }

    #[test]
    fn lower_dwrt_refuses_a_non_constant_variable_operand() {
        let mut b = ExprBuilder::new();
        let x = b.push_var(0);
        let root = b.push_binary(OpKind::Dwrt, x, x);
        let g = freeze(b, &[root]);
        assert_eq!(
            lower_dwrt(g.term(0)).err(),
            Some("lower_dwrt: Dwrt's variable operand must be a Const")
        );
    }

    // ──────────────────────────── Reduce ────────────────────────────

    /// A uniform is a value, never an extent. `Kernel::over` takes a `u32`
    /// so this cannot be built through the API; a graph can still be
    /// hand-built (or rewritten) into it, and then the unroll must refuse
    /// rather than read a slot index as a trip count.
    #[test]
    #[should_panic(expected = "reduce extent must be a Const")]
    fn a_uniform_in_the_extent_slot_is_refused() {
        let mut b = ExprBuilder::new();
        let u = b.declare_uniform(UniformDecl {
            id: UniformIdentity::mint(),
            default: 4.0,
        });
        let combiner = b.push_const(OpKind::Add.index() as f32);
        let rvar = b.push_const(4.0);
        let extent = b.push_uniform(u);
        let body = b.push_var(4);
        let red = b.push_nary(OpKind::Reduce, &[combiner, rvar, extent, body]);
        let g = freeze(b, &[red]);
        let _refused = expand_reduce(g.term(0));
    }

    /// The other side of the same rule: a uniform is a perfectly good *value*
    /// under the binder — invariant in the index, so shared by every term.
    #[test]
    fn a_uniform_under_a_binder_is_shared_by_every_unrolled_term() {
        let decl = UniformDecl {
            id: UniformIdentity::mint(),
            default: 10.0,
        };
        let mut b = ExprBuilder::new();
        let u = b.declare_uniform(decl);
        let uval = b.push_uniform(u);
        let i = b.push_var(4);
        let body = b.push_binary(OpKind::Add, i, uval);
        let red = b.push_reduce(OpKind::Add, 4, 3, body);
        let g = freeze(b, &[red]);

        let out = expand_reduce(g.term(0));
        let term = Term::new(out.entry(), &g.env);
        // Σ_{i<3} (i + u) = 3 + 3u.
        assert_eq!(eval_scalar(term, &[0.0; 2], &BindingTable::empty()), 33.0);
        let bound = BindingTable::empty()
            .bind_uniforms(&g.env, &[(decl.id, 1.0)])
            .expect("declared");
        assert_eq!(eval_scalar(term, &[0.0; 2], &bound), 6.0);

        // Over the reachable subgraph: the invariant leaf is shared, not
        // copied once per term.
        let uniform_leaves = term
            .root()
            .descendants()
            .filter(|n| matches!(**n, ExprData::Uniform(_)))
            .count();
        assert_eq!(uniform_leaves, 1);
    }

    #[test]
    fn nested_reductions_unroll_innermost_first() {
        // Σ_{i<3} Σ_{j<2} (i·10 + j) = Σ_i (2i·10 + 1) = 60 + 3 = 63.
        let mut b = ExprBuilder::new();
        let i = b.push_var(4);
        let j = b.push_var(5);
        let ten = b.push_const(10.0);
        let scaled = b.push_binary(OpKind::Mul, i, ten);
        let body = b.push_binary(OpKind::Add, scaled, j);
        let inner = b.push_reduce(OpKind::Add, 5, 2, body);
        let outer = b.push_reduce(OpKind::Add, 4, 3, inner);
        let g = freeze(b, &[outer]);

        assert_eq!(g.eval(0, &[0.0; 2]), 63.0);
        let out = expand_reduce(g.term(0));
        assert!(
            !out.entry()
                .descendants()
                .any(|n| *n == ExprData::Op(OpKind::Reduce)),
            "both binders must be gone"
        );
        assert_eq!(g.eval_lowered(&out, &[0.0; 2]), 63.0);
    }

    // ─────────────────────────── legalize ───────────────────────────

    #[test]
    fn legalize_lowers_gather() {
        let mut b = ExprBuilder::new();
        let buf = buffer(&mut b, 4, 1);
        let x = b.push_var(0);
        let zero = b.push_const(0.0);
        let root = b.push_gather(buf, x, zero);
        let g = freeze(b, &[root]);

        let out = legalize(g.term(0)).expect("legalize");
        assert!(
            !out.entry()
                .descendants()
                .any(|n| *n == ExprData::Op(OpKind::Gather) && n.child_count() == 3),
            "legalize left a high-level Gather reachable"
        );

        let buf_data = [10.0f32, 20.0, 30.0, 40.0];
        let bindings = BindingTable::bind(&g.env, &[&buf_data[..]]).unwrap();
        assert_eq!(
            eval_scalar(Term::new(out.entry(), &g.env), &[2.0, 0.0], &bindings),
            30.0
        );
    }

    #[test]
    fn legalize_lowers_reduce_transcendentals_and_dwrt_together() {
        // (Σ_{i<3} i) + d/dX[sin(X)], exercising `expand_reduce`, `lower_dwrt`,
        // and the `expand_transcendentals` pass that `lower_dwrt`'s own output
        // (a `Cos`) feeds into — all three passes `legalize` composes, on one
        // graph. `Dwrt` can only wrap what it can differentiate (`lower_dwrt`
        // has no rule for a raw `Reduce`, see `unsupported_op_errors_loudly`),
        // so the reduction and the derivative are independent subtrees joined
        // by `Add` rather than one nested inside the other.
        let mut b = ExprBuilder::new();
        let i = b.push_var(4);
        let red = b.push_reduce(OpKind::Add, 4, 3, i); // Σ_{i<3} i = 0+1+2 = 3
        let x = b.push_var(0);
        let s = b.push_unary(OpKind::Sin, x);
        let v0 = b.push_const(0.0);
        let dwrt_sin = b.push_binary(OpKind::Dwrt, s, v0); // d/dX sin(X) = cos(X)
        let root = b.push_binary(OpKind::Add, red, dwrt_sin);
        let g = freeze(b, &[root]);

        let out = legalize(g.term(0)).expect("legalize");
        for node in out.entry().descendants() {
            assert!(
                *node != ExprData::Op(OpKind::Reduce) && *node != ExprData::Op(OpKind::Dwrt),
                "legalize left a {:?} reachable",
                *node
            );
            // Every transcendental, not just the input `Sin`: `lower_dwrt`
            // replaces that `Sin` with a `Cos`, so naming one op would let a
            // `legalize` that skipped its final expansion pass slip through.
            // The value check below cannot catch it either — `eval_scalar`
            // expands transcendentals itself.
            assert!(
                !is_transcendental(node),
                "legalize left a backend-illegal {:?}",
                *node
            );
        }

        // 3 + cos(X), at X = 0.5.
        let want = 3.0 + libm::cosf(0.5);
        let pt = [0.5f32, 0.0];
        assert_close_rel(g.eval_lowered(&out, &pt), want, &pt, 3e-2);
    }

    #[test]
    fn a_pass_with_nothing_to_do_copies_the_reachable_subgraph_and_nothing_else() {
        // The identity fast-path: construction garbage does not survive, and
        // what does survive is structurally the same expression.
        let mut b = ExprBuilder::new();
        let _garbage = b.push_const(99.0);
        let x = b.push_var(0);
        let y = b.push_var(1);
        let root = b.push_binary(OpKind::Add, x, y);
        let g = freeze(b, &[root]);

        for out in [
            expand_transcendentals(g.term(0)),
            expand_gather(g.term(0)),
            expand_reduce(g.term(0)),
            lower_dwrt(g.term(0)).expect("nothing to lower"),
        ] {
            assert_eq!(out.len(), 3, "the unreachable Const does not travel");
            assert!(out.entry().subtree_eq(g.term(0).root()));
        }
    }

    #[test]
    fn n_ary_children_survive_a_rebuild_in_order() {
        // A `Tuple`'s children are positional, and a rebuild that reordered or
        // truncated them would still produce a well-formed graph.
        let mut b = ExprBuilder::new();
        let x = b.push_var(0);
        let y = b.push_var(1);
        let i = b.push_var(4);
        let root = b.push_nary(OpKind::Tuple, &[x, y, i]);
        let g = freeze(b, &[root]);

        // Any pass runs every reachable node through the structural copy;
        // `expand_transcendentals` is the simplest and has nothing to lower.
        let out = expand_transcendentals(g.term(0));
        let kids: Vec<ExprData> = out.entry().children().map(|c| *c).collect();
        assert_eq!(kids, [ExprData::Var(0), ExprData::Var(1), ExprData::Var(4)]);
    }

    #[test]
    fn transcendentals_evaluate_close_to_host_libm() {
        // No transcendental has a scalar `eval_unary`/`eval_binary` arm (see
        // `kind.rs`) — evaluating one at all requires `expand_transcendentals`
        // to have lowered it to arithmetic first. This evaluates each expansion
        // directly across a spread of magnitudes and signs (range reduction and
        // quadrant selection inside the expansions branch on both), checked
        // against host libm, at a tolerance sized per expansion rather than one
        // loose bound for all of them.
        //
        // `LIBM_TOL` is for the expansions that genuinely need it — `exp`,
        // `exp2`, and `atan`, whose fits carry real approximation error by
        // design (`ATAN_MINIMAX` is documented at ~8.7e-5).
        //
        // `TRIG_TOL` is separate because `SIN_CHEB` is six odd coefficients —
        // a degree-11 fit — and the module documents ~1.5e-6 for sin/cos.
        // Measured against libm at exactly the points below, the worst error
        // is 4.2e-7 in this test's own metric, so 3e-2 was roughly 70,000x
        // looser than the implementation: at `x = 0.1`, where
        // `assert_close_rel`'s `max(|want|, 1)` floor makes the bound a plain
        // absolute 0.03, a `sin` that returned 0.13 would have passed. 1e-5
        // keeps ~24x headroom over the measurement and still sits above the
        // documented accuracy, which leaves room for the ISA levels where FMA
        // contraction shifts the last bits, while staying tight enough that a
        // wrong coefficient, sign, or branch cannot hide.
        type Reference = fn(f32) -> f32;

        const LIBM_TOL: f32 = 3e-2;
        const TRIG_TOL: f32 = 1e-5;

        fn unary_at(op: OpKind, x: f32) -> f32 {
            let mut b = ExprBuilder::new();
            let v = b.push_var(0);
            let e = b.push_unary(op, v);
            freeze(b, &[e]).eval(0, &[x, 0.0])
        }

        fn binary_at(op: OpKind, x: f32, y: f32) -> f32 {
            let mut b = ExprBuilder::new();
            let a = b.push_var(0);
            let c = b.push_var(1);
            let e = b.push_binary(op, a, c);
            freeze(b, &[e]).eval(0, &[x, y])
        }

        let periodic_pts = [-100.0f32, -7.0, -0.6, 0.1, 0.6, 2.5, 7.0, 100.0];
        let unary: [(OpKind, Reference); 3] = [
            (OpKind::Sin, libm::sinf),
            (OpKind::Cos, libm::cosf),
            (OpKind::Tan, libm::tanf),
        ];
        for (op, reference) in unary {
            for &x in &periodic_pts {
                let pt = [x, 0.0];
                assert_close_rel(unary_at(op, x), reference(x), &pt, TRIG_TOL);
            }
        }

        // The exponentials are checked purely relatively, on their own points.
        // `assert_close_rel`'s `want.abs().max(1.0)` floor turns into a plain
        // absolute 0.03 once the reference falls below 1, which is most of the
        // negative half-line here: `expf(-100)` is ~3.8e-44, so returning zero
        // would pass. And at +100 `expf` is `inf`, making `|got - inf| <= inf`
        // accept anything finite. A relative-only comparison over a range
        // where both sides stay finite and nonzero keeps every point binding.
        let exp_pts = [-20.0f32, -7.0, -0.6, 0.1, 0.6, 2.5, 7.0, 20.0];
        for (op, reference) in [
            (OpKind::Exp, libm::expf as Reference),
            (OpKind::Exp2, libm::exp2f as Reference),
        ] {
            for &x in &exp_pts {
                let got = unary_at(op, x);
                let want = reference(x);
                assert!(
                    want.is_finite() && want > 0.0,
                    "{op:?} oracle at {x} is {want}: a relative check needs a finite, \
                     nonzero reference"
                );
                let rel_err = (got - want).abs() / want;
                assert!(
                    rel_err <= LIBM_TOL,
                    "at {x}: {op:?} got {got}, want {want} (relative error {rel_err} > {LIBM_TOL})"
                );
            }
        }

        // Atan is unbounded; Asin/Acos are domain-restricted to [-1, 1].
        for &x in &[-100.0f32, -1.7, -0.3, 0.3, 1.7, 100.0] {
            let pt = [x, 0.0];
            assert_close_rel(unary_at(OpKind::Atan, x), libm::atanf(x), &pt, LIBM_TOL);
        }
        for &x in &[-0.9f32, -0.5, -0.1, 0.1, 0.5, 0.9] {
            let pt = [x, 0.0];
            assert_close_rel(unary_at(OpKind::Asin, x), libm::asinf(x), &pt, LIBM_TOL);
            assert_close_rel(unary_at(OpKind::Acos, x), libm::acosf(x), &pt, LIBM_TOL);
        }

        // Ln/Log2/Log10's Cephes-style minimax fit is far tighter than the
        // trig/exp expansions above (worst case a few times 1e-7 relative,
        // vs. the percent-level `LIBM_TOL` those need), so hold it to its
        // own much narrower tolerance — loose enough for f32 rounding, tight
        // enough that a wrong Horner coefficient cannot hide inside it.
        const LOG_TOL: f32 = 3e-5;
        // The mantissa extraction reduces every input to the SAME fixed
        // range regardless of magnitude (`t ∈ [-0.293, 0.414]`, see
        // `expand_log2`), so a few magnitudes spanning decades sample almost
        // the same handful of `t` values — not enough to reliably land near
        // a Horner coefficient's worst point. Sweep the mantissa densely
        // (plus a couple of magnitudes to touch the exponent path, and the
        // range-reduction threshold itself at √2, each coefficient's most
        // sensitive point) instead.
        let mut log_pts: Vec<f32> = (0..256).map(|k| 1.0 + k as f32 * (0.999 / 256.0)).collect();
        log_pts.extend([1e-3f32, 10.0, 1e6, core::f32::consts::SQRT_2]);
        let logs: [(OpKind, Reference); 3] = [
            (OpKind::Ln, libm::logf),
            (OpKind::Log2, libm::log2f),
            (OpKind::Log10, libm::log10f),
        ];
        for (op, reference) in logs {
            for &x in &log_pts {
                let pt = [x, 0.0];
                assert_close_rel(unary_at(op, x), reference(x), &pt, LOG_TOL);
            }
        }

        // Atan2 over all four quadrants plus the axis-aligned cases.
        for (y, x) in [
            (3.0f32, 4.0),
            (3.0, -4.0),
            (-3.0, 4.0),
            (-3.0, -4.0),
            (0.0, -1.0), // pi
            (1.0, 0.0),  // pi/2
        ] {
            let pt = [y, x];
            assert_close_rel(
                binary_at(OpKind::Atan2, y, x),
                libm::atan2f(y, x),
                &pt,
                LIBM_TOL,
            );
        }

        // Pow needs a positive base (it lowers through log2/exp2).
        for (base, exp) in [(2.0f32, 3.3), (0.5, 2.0), (10.0, -1.5)] {
            let pt = [base, exp];
            assert_close_rel(
                binary_at(OpKind::Pow, base, exp),
                libm::powf(base, exp),
                &pt,
                LIBM_TOL,
            );
        }
    }
}

// ─────────────────────────── Passes as optimizers ────────────────────────────
//
// The two passes the runtime tier runs before saturation, as
// [`Optimize`](crate::optimize::Optimize) values so the tier can spell its
// pipeline as a composition instead of hand-sequenced calls whose order only a
// comment enforces.
//
// Each reports `Unchanged` where it would otherwise copy the graph to say
// "nothing to do" — the copy was pure waste, and the type has a way to decline
// it.

use crate::optimize::{Optimize, Rewritten};

/// Resolve `Dwrt` (symbolic differentiation) into ordinary arithmetic.
///
/// Runs BEFORE saturation, and the order matters: differentiation manufactures
/// constants — the winding kernels' `d = X − f(Y)` gives `DX(d) = 1` and, for
/// a straight edge, a constant `DY(d)`, making the whole gradient magnitude
/// `√(DX²+DY²)` a compile-time number — and constant folding can only cascade
/// over constants that exist by the time it runs. Lowering after the e-graph
/// leaves those folds permanently on the table, because nothing folds
/// post-extraction.
///
/// Declines on a genuinely non-differentiable op, which stops the pipeline:
/// the term compiles unoptimized and the compile entry's own `lower_dwrt`
/// reports the same error loudly, at the layer that can name it.
#[derive(Clone, Copy, Debug, Default)]
pub struct LowerDwrt;

impl Optimize for LowerDwrt {
    fn optimize(&mut self, term: Term<'_>) -> Rewritten {
        if !reaches(term, |n| *n == ExprData::Op(OpKind::Dwrt)) {
            return Rewritten::Unchanged;
        }
        match lower_dwrt(term) {
            Ok(rooted) => Rewritten::Changed(rooted, term.env().clone()),
            Err(_) => Rewritten::Declined,
        }
    }
}

/// Unroll every `Reduce` into its terms.
///
/// The extents are static, so the binder disappears into N terms sharing their
/// index-invariant subtrees, and what saturation then sees is binder-free
/// arithmetic it can CSE and fold across those terms — rather than rewriting
/// under a binder.
#[derive(Clone, Copy, Debug, Default)]
pub struct ExpandReduce;

impl Optimize for ExpandReduce {
    fn optimize(&mut self, term: Term<'_>) -> Rewritten {
        if !reaches(term, |n| *n == ExprData::Op(OpKind::Reduce)) {
            return Rewritten::Unchanged;
        }
        Rewritten::Changed(expand_reduce(term), term.env().clone())
    }
}
