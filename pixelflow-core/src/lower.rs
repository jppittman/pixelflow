//! [`Lower`] impls for the combinator language: the functor from ZST
//! expression trees to the [`ExprArena`] IR, one impl per generator, beside
//! the `Manifold` impls that define evaluation. Because `kernel!` output is a
//! composition of these generators, everything the macros build lowers by
//! composition — nothing outside the language ever constructs IR.
//!
//! Impls returning `None` mark values with no IR form (opaque evaluators,
//! non-uniform runtime vectors, context shapes not yet modeled); verbs like
//! `Lattice::realize` fall back to generic evaluation for those trees.

use pixelflow_ir::arena::{ExprArena, ExprId};
use pixelflow_ir::{Lower, LowerEnv, OpKind};

use crate::Field;
use crate::combinators::binding::{B0, B1, Let, UInt, UTerm, Var};
use crate::combinators::context::{A0, A1, A2, A3, CtxVar, WithContext};
use crate::combinators::{At, Select};
use crate::ops::binary::{
    Add, Atan2, Div, Hypot, Mul, MulAdd, MulRecip, MulRsqrt, Pow, Sub,
};
use crate::ops::compare::{Ge, Gt, Le, Lt, SoftGt, SoftLt, SoftSelect};
use crate::ops::derivative::{DxOf, DxxOf, DxyOf, DyOf, DyyOf, DzOf, ValOf};
use crate::ops::logic::{And, Or};
use crate::ops::unary::{
    Abs, Acos, Asin, Atan, Ceil, Cos, Exp, Exp2, Floor, Fract, Ln, Log2, Max, Min, Neg, Recip,
    Round, Rsqrt, Sin, Sqrt, Tan,
};
use crate::variables::{W, X, Y, Z};

// ───────────────────────────── leaves ────────────────────────────────────────

impl Lower for X {
    fn lower(&self, arena: &mut ExprArena, _env: &mut LowerEnv) -> Option<ExprId> {
        Some(arena.push_var(0))
    }
}
impl Lower for Y {
    fn lower(&self, arena: &mut ExprArena, _env: &mut LowerEnv) -> Option<ExprId> {
        Some(arena.push_var(1))
    }
}
impl Lower for Z {
    fn lower(&self, arena: &mut ExprArena, _env: &mut LowerEnv) -> Option<ExprId> {
        Some(arena.push_var(2))
    }
}
impl Lower for W {
    fn lower(&self, arena: &mut ExprArena, _env: &mut LowerEnv) -> Option<ExprId> {
        Some(arena.push_var(3))
    }
}

/// A `Field` used as a manifold is a runtime vector, not necessarily uniform
/// across lanes — it has no expression form.
impl Lower for Field {
    fn lower(&self, _arena: &mut ExprArena, _env: &mut LowerEnv) -> Option<ExprId> {
        None
    }
}

// ───────────────────────────── operators ─────────────────────────────────────

macro_rules! lower_unary {
    ($($ty:ident => $op:ident),* $(,)?) => {$(
        impl<M: Lower> Lower for $ty<M> {
            fn lower(&self, arena: &mut ExprArena, env: &mut LowerEnv) -> Option<ExprId> {
                let a = self.0.lower(arena, env)?;
                Some(arena.push_unary(OpKind::$op, a))
            }
        }
    )*};
}

lower_unary! {
    Sqrt => Sqrt, Neg => Neg, Abs => Abs, Floor => Floor, Rsqrt => Rsqrt,
    Sin => Sin, Cos => Cos, Log2 => Log2, Exp2 => Exp2, Exp => Exp,
    Recip => Recip, Ceil => Ceil, Round => Round, Fract => Fract, Tan => Tan,
    Asin => Asin, Acos => Acos, Atan => Atan, Ln => Ln,
}

macro_rules! lower_binary {
    ($($ty:ident => $op:ident),* $(,)?) => {$(
        impl<L: Lower, R: Lower> Lower for $ty<L, R> {
            fn lower(&self, arena: &mut ExprArena, env: &mut LowerEnv) -> Option<ExprId> {
                let l = self.0.lower(arena, env)?;
                let r = self.1.lower(arena, env)?;
                Some(arena.push_binary(OpKind::$op, l, r))
            }
        }
    )*};
}

lower_binary! {
    Add => Add, Sub => Sub, Mul => Mul, Div => Div,
    Min => Min, Max => Max,
    Lt => Lt, Le => Le, Gt => Gt, Ge => Ge,
    And => BitAnd, Or => BitOr,
    Atan2 => Atan2, Pow => Pow, Hypot => Hypot,
}

impl<A: Lower, B: Lower, C: Lower> Lower for MulAdd<A, B, C> {
    fn lower(&self, arena: &mut ExprArena, env: &mut LowerEnv) -> Option<ExprId> {
        let a = self.0.lower(arena, env)?;
        let b = self.1.lower(arena, env)?;
        let c = self.2.lower(arena, env)?;
        Some(arena.push_ternary(OpKind::MulAdd, a, b, c))
    }
}

/// `inner * recip` with the reciprocal precomputed — a plain multiply in IR.
impl<M: Lower> Lower for MulRecip<M> {
    fn lower(&self, arena: &mut ExprArena, env: &mut LowerEnv) -> Option<ExprId> {
        let inner = self.inner.lower(arena, env)?;
        let r = arena.push_const(self.recip);
        Some(arena.push_binary(OpKind::Mul, inner, r))
    }
}

/// `l * rsqrt(r)`.
impl<L: Lower, R: Lower> Lower for MulRsqrt<L, R> {
    fn lower(&self, arena: &mut ExprArena, env: &mut LowerEnv) -> Option<ExprId> {
        let l = self.0.lower(arena, env)?;
        let r = self.1.lower(arena, env)?;
        let rs = arena.push_unary(OpKind::Rsqrt, r);
        Some(arena.push_binary(OpKind::Mul, l, rs))
    }
}

impl<C: Lower, T: Lower, F: Lower> Lower for Select<C, T, F> {
    fn lower(&self, arena: &mut ExprArena, env: &mut LowerEnv) -> Option<ExprId> {
        let c = self.cond.lower(arena, env)?;
        let t = self.if_true.lower(arena, env)?;
        let f = self.if_false.lower(arena, env)?;
        Some(arena.push_ternary(OpKind::Select, c, t, f))
    }
}

/// The soft comparisons are smooth approximations with tuning baked into
/// their evaluators; they have no canonical expression form yet.
macro_rules! lower_none_binary {
    ($($ty:ident),* $(,)?) => {$(
        impl<L, R> Lower for $ty<L, R> {
            fn lower(&self, _arena: &mut ExprArena, _env: &mut LowerEnv) -> Option<ExprId> {
                None
            }
        }
    )*};
}
lower_none_binary!(SoftLt, SoftGt);

impl<C, T, F> Lower for SoftSelect<C, T, F> {
    fn lower(&self, _arena: &mut ExprArena, _env: &mut LowerEnv) -> Option<ExprId> {
        None
    }
}

// ───────────────────────── derivative projections ────────────────────────────

fn dwrt(arena: &mut ExprArena, expr: ExprId, var: u8) -> ExprId {
    let v = arena.push_const(f32::from(var));
    arena.push_binary(OpKind::Dwrt, expr, v)
}

impl<M: Lower> Lower for ValOf<M> {
    fn lower(&self, arena: &mut ExprArena, env: &mut LowerEnv) -> Option<ExprId> {
        // Value projection: every arena expression is already value-space.
        self.0.lower(arena, env)
    }
}

macro_rules! lower_first_derivative {
    ($($ty:ident => $var:literal),* $(,)?) => {$(
        impl<M: Lower> Lower for $ty<M> {
            fn lower(&self, arena: &mut ExprArena, env: &mut LowerEnv) -> Option<ExprId> {
                let m = self.0.lower(arena, env)?;
                Some(dwrt(arena, m, $var))
            }
        }
    )*};
}
lower_first_derivative!(DxOf => 0, DyOf => 1, DzOf => 2);

macro_rules! lower_second_derivative {
    ($($ty:ident => ($a:literal, $b:literal)),* $(,)?) => {$(
        impl<M: Lower> Lower for $ty<M> {
            fn lower(&self, arena: &mut ExprArena, env: &mut LowerEnv) -> Option<ExprId> {
                let m = self.0.lower(arena, env)?;
                let first = dwrt(arena, m, $a);
                Some(dwrt(arena, first, $b))
            }
        }
    )*};
}
lower_second_derivative!(DxxOf => (0, 0), DxyOf => (0, 1), DyyOf => (1, 1));

// ───────────────────────────── binding forms ─────────────────────────────────

/// Runtime value of a type-level binary number (`UTerm`/`B0`/`B1`/`UInt`).
pub trait TypeNat {
    /// The number this type denotes.
    const VALUE: usize;
}
impl TypeNat for UTerm {
    const VALUE: usize = 0;
}
impl TypeNat for B0 {
    const VALUE: usize = 0;
}
impl TypeNat for B1 {
    const VALUE: usize = 1;
}
impl<U: TypeNat, B: TypeNat> TypeNat for UInt<U, B> {
    const VALUE: usize = U::VALUE * 2 + B::VALUE;
}

impl<Val: Lower, Body: Lower> Lower for Let<Val, Body> {
    fn lower(&self, arena: &mut ExprArena, env: &mut LowerEnv) -> Option<ExprId> {
        let bound = self.val.lower(arena, env)?;
        env.bind(bound);
        let body = self.body.lower(arena, env);
        env.unbind();
        body
    }
}

impl<N: TypeNat> Lower for Var<N> {
    fn lower(&self, _arena: &mut ExprArena, env: &mut LowerEnv) -> Option<ExprId> {
        env.resolve(N::VALUE)
    }
}

/// Scalar context (`A0` array): `CtxVar<A0, I>` becomes `Param(I)`, and the
/// enclosing [`WithContext`] bakes the stored values via `substitute_params`.
/// Nested contexts substitute innermost-first, which matches the evaluator's
/// head-of-domain shadowing.
impl<const I: usize> Lower for CtxVar<A0, I> {
    fn lower(&self, arena: &mut ExprArena, _env: &mut LowerEnv) -> Option<ExprId> {
        u8::try_from(I).ok().map(|i| arena.push_param(i))
    }
}

/// Non-scalar context arrays are not modeled in the IR yet.
macro_rules! lower_none_ctxvar {
    ($($pos:ident),* $(,)?) => {$(
        impl<const I: usize> Lower for CtxVar<$pos, I> {
            fn lower(&self, _arena: &mut ExprArena, _env: &mut LowerEnv) -> Option<ExprId> {
                None
            }
        }
    )*};
}
lower_none_ctxvar!(A1, A2, A3);

impl<B: Lower> Lower for WithContext<(), B> {
    fn lower(&self, arena: &mut ExprArena, env: &mut LowerEnv) -> Option<ExprId> {
        self.body.lower(arena, env)
    }
}

impl<const N: usize, B: Lower> Lower for WithContext<([f32; N],), B> {
    fn lower(&self, arena: &mut ExprArena, env: &mut LowerEnv) -> Option<ExprId> {
        let body = self.body.lower(arena, env)?;
        Some(arena.substitute_params(body, &self.ctx.0))
    }
}

/// The lane value of a splat `Field`, or `None` if the lanes differ (a
/// genuinely vectorized runtime value has no scalar expression form). The
/// kernel macros build context arrays by splatting scalar params, so this
/// recovers exactly what was bound; the check keeps a hand-constructed
/// non-uniform context honest by declining instead of miscompiling.
fn uniform_lane(f: Field) -> Option<f32> {
    let mut lanes = [0.0f32; crate::PARALLELISM];
    f.store(&mut lanes);
    let v = lanes[0];
    lanes.iter().all(|&x| x == v).then_some(v)
}

/// The shape the kernel macros actually emit: scalar params splatted into a
/// `Field` context array. Bake each back to a `Const` via the uniform-lane
/// check, then substitute exactly as the `[f32; N]` case.
impl<const N: usize, B: Lower> Lower for WithContext<([Field; N],), B> {
    fn lower(&self, arena: &mut ExprArena, env: &mut LowerEnv) -> Option<ExprId> {
        let mut vals = [0.0f32; N];
        for (slot, field) in vals.iter_mut().zip(self.ctx.0.iter()) {
            *slot = uniform_lane(*field)?;
        }
        let body = self.body.lower(arena, env)?;
        Some(arena.substitute_params(body, &vals))
    }
}

// ───────────────────────────── composition ───────────────────────────────────

/// Contramap: lower the coordinate expressions and the inner manifold, then
/// substitute the inner's coordinate variables — `At` is precomposition, and
/// in the arena that is graph surgery, not evaluation.
impl<Cx: Lower, Cy: Lower, Cz: Lower, Cw: Lower, M: Lower> Lower for At<Cx, Cy, Cz, Cw, M> {
    fn lower(&self, arena: &mut ExprArena, env: &mut LowerEnv) -> Option<ExprId> {
        let cx = self.x.lower(arena, env)?;
        let cy = self.y.lower(arena, env)?;
        let cz = self.z.lower(arena, env)?;
        let cw = self.w.lower(arena, env)?;
        let inner = self.inner.lower(arena, env)?;
        Some(arena.substitute_vars_with(inner, &[(0, cx), (1, cy), (2, cz), (3, cw)]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_ir::binding::BindingTable;
    use pixelflow_ir::eval::eval_scalar;

    fn lower_expr(m: &impl Lower) -> (ExprArena, ExprId) {
        let mut arena = ExprArena::new();
        let mut env = LowerEnv::default();
        let root = m.lower(&mut arena, &mut env).expect("expression lowers");
        (arena, root)
    }

    fn eval(arena: &ExprArena, root: ExprId, p: [f32; 4]) -> f32 {
        eval_scalar(arena, root, &p, &BindingTable::empty())
    }

    #[test]
    fn arithmetic_tree_lowers_and_matches() {
        // (X - 1.0) * (Y + 2.0)
        let expr = Mul(Sub(X, 1.0f32), Add(Y, 2.0f32));
        let (arena, root) = lower_expr(&expr);
        assert_eq!(eval(&arena, root, [3.0, 4.0, 0.0, 0.0]), 2.0 * 6.0);
    }

    #[test]
    fn select_and_masks_lower() {
        // (X >= 0 & X <= 1) ? Y : 0
        let cond = And(Ge(X, 0.0f32), Le(X, 1.0f32));
        let expr = Select {
            cond,
            if_true: Y,
            if_false: 0.0f32,
        };
        let (arena, root) = lower_expr(&expr);
        assert_eq!(eval(&arena, root, [0.5, 7.0, 0.0, 0.0]), 7.0);
        assert_eq!(eval(&arena, root, [1.5, 7.0, 0.0, 0.0]), 0.0);
    }

    #[test]
    fn with_context_bakes_params() {
        // WithContext(([a, b],), CtxVar0 * X + CtxVar1)
        let body = Add(Mul(CtxVar::<A0, 0>::new(), X), CtxVar::<A0, 1>::new());
        let expr = WithContext::new(([3.0f32, 10.0f32],), body);
        let (arena, root) = lower_expr(&expr);
        assert_eq!(eval(&arena, root, [2.0, 0.0, 0.0, 0.0]), 16.0);
    }

    #[test]
    fn at_is_graph_surgery() {
        // (X * Y) at (X + 1, 2 * Y, Z, W)
        let warped = At {
            inner: Mul(X, Y),
            x: Add(X, 1.0f32),
            y: Mul(2.0f32, Y),
            z: Z,
            w: W,
        };
        let (arena, root) = lower_expr(&warped);
        assert_eq!(eval(&arena, root, [3.0, 4.0, 0.0, 0.0]), 4.0 * 8.0);
    }

    #[test]
    fn projections_lower_to_dwrt() {
        use pixelflow_ir::backend::emit::lowering::lower_dwrt_owned;
        // DX(X * X) — lower, then run the runtime calculus: d/dx x² = 2x.
        let expr = DxOf(Mul(X, X));
        let (arena, root) = lower_expr(&expr);
        let (out, out_root) = lower_dwrt_owned(&arena, root).expect("calculus");
        assert_eq!(eval(&out, out_root, [3.0, 0.0, 0.0, 0.0]), 6.0);
    }

    #[test]
    fn let_bindings_resolve_de_bruijn() {
        use crate::combinators::binding::{N0, N1};
        // let a = X + 1; let b = a * 2; a + b  → (x+1) + (x+1)*2
        let expr = Let {
            val: Add(X, 1.0f32),
            body: Let {
                val: Mul(Var::<N0>::new(), 2.0f32),
                body: Add(Var::<N1>::new(), Var::<N0>::new()),
            },
        };
        let (arena, root) = lower_expr(&expr);
        assert_eq!(eval(&arena, root, [4.0, 0.0, 0.0, 0.0]), 5.0 + 10.0);
    }

    #[test]
    fn opaque_values_decline() {
        let mut arena = ExprArena::new();
        let mut env = LowerEnv::default();
        assert!(Field::from(1.0).lower(&mut arena, &mut env).is_none());
        // A tree containing an opaque leaf declines as a whole.
        assert!(Add(X, Field::from(1.0)).lower(&mut arena, &mut env).is_none());
    }
}
