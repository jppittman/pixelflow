//! Reference scalar interpreter for an [`ExprArena`].
//!
//! This walks the arena node-by-node at scalar `f32` coordinates and is the
//! ground-truth semantics the JIT and SIMD backends must match. It is the
//! first path that can *execute* `Gather` (bound-memory reads), via a
//! [`BindingTable`]; production SIMD execution arrives with the JIT in M2
//! (see `KERNELS_AND_LATTICES.md`).
//!
//! `Gather` semantics deliberately mirror `DiscreteManifold::eval` in
//! `pixelflow-core`: floor each index, clamp to `[0, extent - 1]`, read
//! row-major. The round-trip test asserts that equivalence.

use crate::arena::{ExprArena, ExprId, ExprNode};
use crate::binding::BindingTable;
use crate::kind::OpKind;

/// Evaluate the subtree rooted at `root` at scalar coordinates `vars`
/// (`[X, Y, Z, W]`), reading bound buffers through `bindings`.
///
/// # Panics
///
/// Panics on a node the reference interpreter does not handle: a `Param`
/// (substitute first), a bare `Buffer` outside a `Gather`, an `Nary`, or an
/// op with no scalar evaluation. These are programming errors, not inputs.
#[must_use]
pub fn eval_scalar(
    arena: &ExprArena,
    root: ExprId,
    vars: &[f32; 4],
    bindings: &BindingTable<'_>,
) -> f32 {
    Env {
        arena,
        vars,
        bindings,
        reduce_vars: [0.0; 4],
    }
    .eval(root)
}

/// The immutable evaluation environment threaded through the recursion: the
/// arena, the coordinate values, the buffer bindings, and the current binding
/// of each reduction index (`Var(4..8)`). Grouping them keeps the recursive
/// helpers to a single `ExprId` argument.
#[derive(Clone, Copy)]
struct Env<'a> {
    arena: &'a ExprArena,
    vars: &'a [f32; 4],
    bindings: &'a BindingTable<'a>,
    /// Values bound to reduction indices `Var(4)..Var(8)` by enclosing folds.
    reduce_vars: [f32; 4],
}

impl Env<'_> {
    fn eval(&self, id: ExprId) -> f32 {
        match self.arena.node(id) {
            ExprNode::Var(i) => {
                let i = *i as usize;
                // 0..4 are coordinates; 4..8 are reduction indices.
                if i < 4 {
                    self.vars[i]
                } else {
                    self.reduce_vars[i - 4]
                }
            }
            ExprNode::Const(v) => *v,
            ExprNode::Param(p) => panic!("eval_scalar: Param({p}) — substitute params first"),
            ExprNode::Buffer(b) => panic!(
                "eval_scalar: bare Buffer({}) is not a value; read it through Gather",
                b.0
            ),
            ExprNode::Unary(op, a) => {
                let x = self.eval(*a);
                op.eval_unary(x)
                    .unwrap_or_else(|| panic!("eval_scalar: no scalar eval for unary {op:?}"))
            }
            ExprNode::Binary(OpKind::RawGather, buf, idx) => self.raw_gather(*buf, *idx),
            ExprNode::Binary(op, a, b) => {
                let x = self.eval(*a);
                let y = self.eval(*b);
                op.eval_binary(x, y)
                    .unwrap_or_else(|| panic!("eval_scalar: no scalar eval for binary {op:?}"))
            }
            ExprNode::Ternary(OpKind::Gather, buf, x, y) => self.gather(*buf, *x, *y),
            ExprNode::Ternary(op, a, b, c) => {
                let x = self.eval(*a);
                let y = self.eval(*b);
                let z = self.eval(*c);
                op.eval_ternary(x, y, z)
                    .unwrap_or_else(|| panic!("eval_scalar: no scalar eval for ternary {op:?}"))
            }
            ExprNode::Nary(OpKind::Reduce, start, len) => {
                assert_eq!(*len, 4, "Reduce must have 4 children");
                let ch = self.arena.nary_children_slice(*start, *len);
                self.reduce(ch[0], ch[1], ch[2], ch[3])
            }
            ExprNode::Nary(op, _, _) => panic!("eval_scalar: Nary({op:?}) unsupported"),
        }
    }

    /// Read one bound buffer at floored, clamped, row-major indices. This IS the
    /// reference definition of `Gather`.
    fn gather(&self, buf: ExprId, x: ExprId, y: ExprId) -> f32 {
        let id = match self.arena.node(buf) {
            ExprNode::Buffer(id) => *id,
            other => panic!("Gather's first child must be a Buffer leaf, got {other:?}"),
        };
        let decl = self.arena.buffer_decl(id);
        let data = self.bindings.slot(id);

        let xf = self.eval(x);
        let yf = self.eval(y);

        // Nearest-neighbor: floor then clamp to the declared extents.
        let max_x = decl.width.saturating_sub(1) as i64;
        let max_y = decl.height.saturating_sub(1) as i64;
        let xi = (libm::floorf(xf) as i64).clamp(0, max_x);
        let yi = (libm::floorf(yf) as i64).clamp(0, max_y);

        let idx = yi as usize * decl.width as usize + xi as usize;
        data[idx]
    }

    /// Read a bound buffer at an already-computed linear index (the lowered form
    /// of `Gather`). The index is trusted to be in bounds — the lowering clamped
    /// it — so this just truncates and indexes; an out-of-bounds index is a
    /// broken lowering and panics via the slice bounds check.
    fn raw_gather(&self, buf: ExprId, idx: ExprId) -> f32 {
        let id = match self.arena.node(buf) {
            ExprNode::Buffer(id) => *id,
            other => panic!("RawGather's first child must be a Buffer leaf, got {other:?}"),
        };
        let data = self.bindings.slot(id);
        let index = self.eval(idx);
        data[libm::floorf(index) as usize]
    }

    /// Fold `body` over the reduction index `Var(reduce_var)` = `0..extent`,
    /// combining terms with the monoid named by the `combiner` child. This is
    /// the reference definition that the unrolled `expand_reduce` form must
    /// match. `combiner`, `reduce_var`, and `extent` are `Const` children.
    fn reduce(&self, combiner: ExprId, reduce_var: ExprId, extent: ExprId, body: ExprId) -> f32 {
        let op = OpKind::from_index(self.const_of(combiner, "reduce combiner") as usize)
            .expect("reduce combiner must be a valid OpKind index");
        let var_idx = self.const_of(reduce_var, "reduce var index") as usize;
        let n = self.const_of(extent, "reduce extent") as usize;
        assert!(
            (4..8).contains(&var_idx),
            "reduce index Var({var_idx}) out of range (must be 4..8)"
        );
        let slot = var_idx - 4;

        let mut acc = op
            .monoid_identity()
            .unwrap_or_else(|| panic!("reduce combiner {op:?} is not a monoid"));
        for k in 0..n {
            let mut child = *self;
            child.reduce_vars[slot] = k as f32;
            let term = child.eval(body);
            // Combine under the monoid op (Add/Mul/Min/Max).
            acc = op
                .eval_binary(acc, term)
                .expect("monoid combiner evaluates on two scalars");
        }
        acc
    }

    /// Read a `Const` child that encodes an integer parameter.
    fn const_of(&self, id: ExprId, what: &str) -> f32 {
        match self.arena.node(id) {
            ExprNode::Const(v) => *v,
            other => panic!("reduce {what} must be a Const, got {other:?}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::BufferDecl;
    use alloc::vec;

    /// Reference re-implementation of `DiscreteManifold::eval`'s index math, so
    /// the test asserts equivalence against an independent expression of the
    /// same rule rather than against the interpreter itself.
    fn discrete_eval(buf: &[f32], width: usize, height: usize, x: f32, y: f32) -> f32 {
        let xi = (libm::floorf(x) as i64).clamp(0, width as i64 - 1) as usize;
        let yi = (libm::floorf(y) as i64).clamp(0, height as i64 - 1) as usize;
        buf[yi * width + xi]
    }

    #[test]
    fn gather_round_trips_discrete_manifold() {
        // 4x3 buffer of distinct values.
        let width = 4usize;
        let height = 3usize;
        let buf: vec::Vec<f32> = (0..(width * height)).map(|i| i as f32 * 10.0).collect();

        let mut arena = ExprArena::new();
        let b = arena.declare_buffer(BufferDecl {
            width: width as u32,
            height: height as u32,
        });
        let x = arena.push_var(0);
        let y = arena.push_var(1);
        let gather = arena.push_gather(b, x, y);

        let bindings = BindingTable::bind(&arena, &[buf.as_slice()]).unwrap();

        // Every in-range cell, plus out-of-range coords that must clamp.
        let coords = [
            (0.0, 0.0),
            (1.0, 0.0),
            (3.0, 2.0),
            (2.0, 1.0),
            (-5.0, -5.0),   // clamp to (0,0)
            (100.0, 100.0), // clamp to (3,2)
            (1.9, 0.9),     // floor to (1,0)
        ];
        for (cx, cy) in coords {
            let got = eval_scalar(&arena, gather, &[cx, cy, 0.0, 0.0], &bindings);
            let want = discrete_eval(&buf, width, height, cx, cy);
            assert_eq!(got, want, "gather at ({cx}, {cy})");
        }
    }

    #[test]
    fn lowering_preserves_gather_semantics() {
        // The crux of M2 slice 1: expand_gather must produce an index
        // expression that evaluates identically to the high-level Gather.
        use crate::backend::emit::lowering::expand_gather;

        let width = 5usize;
        let height = 4usize;
        let buf: vec::Vec<f32> = (0..(width * height)).map(|i| i as f32 + 0.5).collect();

        let mut arena = ExprArena::new();
        let b = arena.declare_buffer(BufferDecl {
            width: width as u32,
            height: height as u32,
        });
        // Gather with non-trivial index expressions: (X*2, Y+1).
        let x = arena.push_var(0);
        let y = arena.push_var(1);
        let two = arena.push_const(2.0);
        let one = arena.push_const(1.0);
        let xx = arena.push_binary(OpKind::Mul, x, two);
        let yy = arena.push_binary(OpKind::Add, y, one);
        let gather = arena.push_gather(b, xx, yy);

        // Lower a clone; the buffer table is preserved, so the same binding works.
        let mut lowered_arena = arena.clone();
        let lowered_root = expand_gather(&mut lowered_arena, gather);

        // The lowered form REACHABLE from the new root must contain a
        // RawGather and no high-level Gather. (The arena is append-only, so the
        // original Gather remains as unreachable garbage in nodes_raw().)
        let mut reachable = alloc::vec::Vec::new();
        let mut stack = alloc::vec![lowered_root];
        while let Some(id) = stack.pop() {
            reachable.push(lowered_arena.node(id).clone());
            for c in lowered_arena.children(id) {
                stack.push(c);
            }
        }
        assert!(
            reachable
                .iter()
                .any(|n| matches!(n, ExprNode::Binary(OpKind::RawGather, _, _)))
        );
        assert!(
            !reachable
                .iter()
                .any(|n| matches!(n, ExprNode::Ternary(OpKind::Gather, _, _, _)))
        );

        let bindings = BindingTable::bind(&arena, &[buf.as_slice()]).unwrap();
        let lowered_bindings = BindingTable::bind(&lowered_arena, &[buf.as_slice()]).unwrap();

        // Sweep coords including fractional and out-of-range values.
        for xi in [-2.0f32, 0.0, 0.7, 1.0, 2.0, 3.0, 10.0] {
            for yi in [-1.0f32, 0.0, 0.4, 1.0, 2.0, 3.0, 8.0] {
                let vars = [xi, yi, 0.0, 0.0];
                let hi = eval_scalar(&arena, gather, &vars, &bindings);
                let lo = eval_scalar(&lowered_arena, lowered_root, &vars, &lowered_bindings);
                assert_eq!(hi, lo, "gather vs lowered at ({xi}, {yi})");
            }
        }
    }

    #[test]
    fn gather_composes_with_arithmetic() {
        // out = buffer[X, 0] * 2 + 1, indices computed by an expression.
        let buf = vec![5.0f32, 6.0, 7.0, 8.0];
        let mut arena = ExprArena::new();
        let b = arena.declare_buffer(BufferDecl {
            width: 4,
            height: 1,
        });
        let x = arena.push_var(0);
        let zero = arena.push_const(0.0);
        let gather = arena.push_gather(b, x, zero);
        let two = arena.push_const(2.0);
        let one = arena.push_const(1.0);
        let scaled = arena.push_binary(OpKind::Mul, gather, two);
        let root = arena.push_binary(OpKind::Add, scaled, one);

        let bindings = BindingTable::bind(&arena, &[buf.as_slice()]).unwrap();
        // X = 2 -> buffer[2] = 7 -> 7*2 + 1 = 15
        let got = eval_scalar(&arena, root, &[2.0, 0.0, 0.0, 0.0], &bindings);
        assert_eq!(got, 15.0);
    }

    #[test]
    fn reduce_sum_of_squares() {
        // sum_{i=0}^{3} (i+1)^2 = 1 + 4 + 9 + 16 = 30, folded over Var(4).
        let mut arena = ExprArena::new();
        let i = arena.push_var(4);
        let one = arena.push_const(1.0);
        let ip1 = arena.push_binary(OpKind::Add, i, one);
        let sq = arena.push_binary(OpKind::Mul, ip1, ip1);
        let root = arena.push_reduce(OpKind::Add, 4, 4, sq);

        let bindings = BindingTable::empty();
        assert_eq!(eval_scalar(&arena, root, &[0.0; 4], &bindings), 30.0);
    }

    #[test]
    fn reduce_max_and_mul() {
        // max_{i=0..4} i = 3 ; prod_{i=1..4}(via body i+1) = 2*3*4 = 24.
        let mut arena = ExprArena::new();
        let i = arena.push_var(4);
        let max_root = arena.push_reduce(OpKind::Max, 4, 4, i);
        let bindings = BindingTable::empty();
        assert_eq!(eval_scalar(&arena, max_root, &[0.0; 4], &bindings), 3.0);

        let one = arena.push_const(1.0);
        let ip1 = arena.push_binary(OpKind::Add, i, one);
        // product over i=1..4 of (i+1): i=1->2, 2->3, 3->4  => start at i=0 -> 1
        // Reduce over 0..4 of (i+1) = 1*2*3*4 = 24.
        let mul_root = arena.push_reduce(OpKind::Mul, 4, 4, ip1);
        assert_eq!(eval_scalar(&arena, mul_root, &[0.0; 4], &bindings), 24.0);
    }

    #[test]
    fn reduce_lowering_preserves_semantics() {
        // Σ over i of (X + i), lowered by expand_reduce, must equal the fold.
        use crate::backend::emit::lowering::expand_reduce;
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let i = arena.push_var(4);
        let body = arena.push_binary(OpKind::Add, x, i);
        let root = arena.push_reduce(OpKind::Add, 4, 5, body); // Σ_{i=0}^{4}(X+i)

        let mut lowered = arena.clone();
        let lroot = expand_reduce(&mut lowered, root);
        // No Reduce node remains reachable from the new root.
        let mut stack = alloc::vec![lroot];
        while let Some(id) = stack.pop() {
            assert!(!matches!(
                lowered.node(id),
                ExprNode::Nary(OpKind::Reduce, _, _)
            ));
            for c in lowered.children(id) {
                stack.push(c);
            }
        }
        let b = BindingTable::empty();
        for xv in [-2.0f32, 0.0, 3.5, 10.0] {
            let want = eval_scalar(&arena, root, &[xv, 0.0, 0.0, 0.0], &b);
            let got = eval_scalar(&lowered, lroot, &[xv, 0.0, 0.0, 0.0], &b);
            assert_eq!(want, got, "reduce lowering at X={xv}");
            // Σ_{i=0}^{4}(X+i) = 5X + 10.
            assert_eq!(want, 5.0 * xv + 10.0);
        }
    }

    #[test]
    fn reduce_matmul_dot_over_gather() {
        // out = Σ_i W(i,0) * input(i,0), the matmul kernel body for one output.
        use crate::arena::BufferDecl;
        let w = alloc::vec![2.0f32, 3.0, 4.0]; // W column
        let inp = alloc::vec![10.0f32, 20.0, 30.0];
        let mut arena = ExprArena::new();
        let wb = arena.declare_buffer(BufferDecl {
            width: 3,
            height: 1,
        });
        let ib = arena.declare_buffer(BufferDecl {
            width: 3,
            height: 1,
        });
        let i = arena.push_var(4);
        let zero = arena.push_const(0.0);
        let wg = arena.push_gather(wb, i, zero);
        let ig = arena.push_gather(ib, i, zero);
        let prod = arena.push_binary(OpKind::Mul, wg, ig);
        let root = arena.push_reduce(OpKind::Add, 4, 3, prod);

        let bindings = BindingTable::bind(&arena, &[w.as_slice(), inp.as_slice()]).unwrap();
        // 2*10 + 3*20 + 4*30 = 20 + 60 + 120 = 200.
        assert_eq!(eval_scalar(&arena, root, &[0.0; 4], &bindings), 200.0);
    }

    #[test]
    fn bind_rejects_wrong_length() {
        let mut arena = ExprArena::new();
        let _ = arena.declare_buffer(BufferDecl {
            width: 4,
            height: 2,
        });
        let short = vec![0.0f32; 7]; // needs 8
        let err = BindingTable::bind(&arena, &[short.as_slice()]).unwrap_err();
        assert!(matches!(
            err,
            crate::binding::BindError::Length {
                slot: 0,
                expected: 8,
                actual: 7
            }
        ));
    }

    #[test]
    fn bind_rejects_wrong_count() {
        let mut arena = ExprArena::new();
        let _ = arena.declare_buffer(BufferDecl {
            width: 2,
            height: 2,
        });
        let err = BindingTable::bind(&arena, &[]).unwrap_err();
        assert!(matches!(
            err,
            crate::binding::BindError::Count {
                declared: 1,
                supplied: 0
            }
        ));
    }
}
