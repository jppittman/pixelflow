//! Symbolic-differentiation lowering (`lower_dwrt`) — rule-by-rule validation
//! against analytic ground truth, error paths, and interpreter/JIT agreement.
//!
//! The analytic sample vectors resurrect the ones the deleted emit-time jet
//! mode was validated against (commit 173c0f56: sum-of-squares, product,
//! circle SDF, neg + statically-zero partial), extended to cover every
//! derivative rule `lower_dwrt` implements.

use pixelflow_ir::backend::emit::lowering::{LowerError, lower_dwrt_owned};
use pixelflow_ir::{BindingTable, ExprArena, ExprId, ExprNode, OpKind, eval_scalar};

// ─────────────────────────────── harness ──────────────────────────────────

/// Push `Dwrt(e, Const(var))`.
fn dwrt(a: &mut ExprArena, e: ExprId, var: u8) -> ExprId {
    let v = a.push_const(var as f32);
    a.push_binary(OpKind::Dwrt, e, v)
}

/// Evaluate through the reference interpreter (which lowers Dwrt itself).
fn ev(arena: &ExprArena, root: ExprId, x: f32, y: f32) -> f32 {
    eval_scalar(arena, root, &[x, y, 0.0, 0.0], &BindingTable::empty())
}

fn close(got: f32, want: f32, tag: &str) {
    let tol = 1e-4 + 1e-4 * want.abs();
    assert!(
        (got - want).abs() <= tol || (got.is_nan() && want.is_nan()),
        "{tag}: got {got}, want {want}"
    );
}

/// Sample points from the deleted jet suite (173c0f56).
const PTS: &[(f32, f32)] = &[(3.0, 4.0), (1.0, 2.0), (-2.0, 0.5), (0.7, -1.3)];

/// Unwrap the error of a lowering that must fail (`ExprArena` has no `Debug`,
/// so `expect_err` is unavailable).
fn must_err(res: Result<(ExprArena, ExprId), LowerError>, why: &str) -> LowerError {
    match res {
        Ok(_) => panic!("{why}"),
        Err(e) => e,
    }
}

/// Assert nothing reachable from `root` is a `Dwrt` after lowering.
fn assert_dwrt_free(arena: &ExprArena, root: ExprId) {
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        assert!(
            !matches!(arena.node(id), ExprNode::Binary(OpKind::Dwrt, _, _)),
            "Dwrt survived lowering"
        );
        for c in arena.children(id) {
            stack.push(c);
        }
    }
}

/// Build `Dwrt(f(builder), var)`, lower it, assert Dwrt-free, and return
/// `(arena, root)` for evaluation.
fn lowered<F: FnOnce(&mut ExprArena) -> ExprId>(build: F, var: u8) -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let f = build(&mut a);
    let root = dwrt(&mut a, f, var);
    let (arena, root) = lower_dwrt_owned(&a, root).expect("lowering failed");
    assert_dwrt_free(&arena, root);
    (arena, root)
}

// ─────────────────── resurrected analytic vectors (173c0f56) ────────────────

#[test]
fn dwrt_sum_of_squares() {
    // f = X² + Y² ; ∂x = 2x, ∂y = 2y
    let build = |a: &mut ExprArena| {
        let x = a.push_var(0);
        let y = a.push_var(1);
        let xx = a.push_binary(OpKind::Mul, x, x);
        let yy = a.push_binary(OpKind::Mul, y, y);
        a.push_binary(OpKind::Add, xx, yy)
    };
    let (ax, rx) = lowered(build, 0);
    let (ay, ry) = lowered(build, 1);
    for &(px, py) in PTS {
        close(ev(&ax, rx, px, py), 2.0 * px, "d(x²+y²)/dx");
        close(ev(&ay, ry, px, py), 2.0 * py, "d(x²+y²)/dy");
    }
}

#[test]
fn dwrt_product_cross_term() {
    // f = X·Y ; ∂x = y, ∂y = x
    let build = |a: &mut ExprArena| {
        let x = a.push_var(0);
        let y = a.push_var(1);
        a.push_binary(OpKind::Mul, x, y)
    };
    let (ax, rx) = lowered(build, 0);
    let (ay, ry) = lowered(build, 1);
    for &(px, py) in PTS {
        close(ev(&ax, rx, px, py), py, "d(xy)/dx");
        close(ev(&ay, ry, px, py), px, "d(xy)/dy");
    }
}

#[test]
fn dwrt_circle_sdf() {
    // f = √(X²+Y²) − 1.5 ; ∂x = x/√(x²+y²), ∂y = y/√(x²+y²)
    let build = |a: &mut ExprArena| {
        let x = a.push_var(0);
        let y = a.push_var(1);
        let xx = a.push_binary(OpKind::Mul, x, x);
        let yy = a.push_binary(OpKind::Mul, y, y);
        let sum = a.push_binary(OpKind::Add, xx, yy);
        let d = a.push_unary(OpKind::Sqrt, sum);
        let r = a.push_const(1.5);
        a.push_binary(OpKind::Sub, d, r)
    };
    let (ax, rx) = lowered(build, 0);
    let (ay, ry) = lowered(build, 1);
    for &(px, py) in PTS {
        let d = (px * px + py * py).sqrt();
        close(ev(&ax, rx, px, py), px / d, "circle ∂x");
        close(ev(&ay, ry, px, py), py / d, "circle ∂y");
    }
}

#[test]
fn dwrt_neg_and_statically_zero_partial() {
    // f = -X ; ∂x = -1, ∂y = 0
    let build = |a: &mut ExprArena| {
        let x = a.push_var(0);
        a.push_unary(OpKind::Neg, x)
    };
    let (ax, rx) = lowered(build, 0);
    let (ay, ry) = lowered(build, 1);
    for &(px, py) in PTS {
        close(ev(&ax, rx, px, py), -1.0, "d(-x)/dx");
        close(ev(&ay, ry, px, py), 0.0, "d(-x)/dy");
    }
}

// ───────────────────────────── unary rules ────────────────────────────────

/// Differentiate a unary op of X at `x` and compare to `want(x)`.
fn check_unary(op: OpKind, xs: &[f32], want: impl Fn(f32) -> f32) {
    let (a, r) = lowered(
        |a| {
            let x = a.push_var(0);
            a.push_unary(op, x)
        },
        0,
    );
    for &x in xs {
        close(ev(&a, r, x, 0.0), want(x), &format!("d({op:?})/dx at {x}"));
    }
}

#[test]
fn dwrt_trig_rules() {
    let xs = [0.3f32, 1.0, -0.7, 2.1];
    check_unary(OpKind::Sin, &xs, f32::cos);
    check_unary(OpKind::Cos, &xs, |x| -x.sin());
    check_unary(OpKind::Tan, &xs, |x| 1.0 / (x.cos() * x.cos()));
    check_unary(OpKind::Atan, &xs, |x| 1.0 / (1.0 + x * x));
    let small = [0.0f32, 0.4, -0.8, 0.65];
    check_unary(OpKind::Asin, &small, |x| 1.0 / (1.0 - x * x).sqrt());
    check_unary(OpKind::Acos, &small, |x| -1.0 / (1.0 - x * x).sqrt());
}

#[test]
fn dwrt_exp_log_rules() {
    let xs = [0.3f32, 1.0, -0.7, 2.1];
    check_unary(OpKind::Exp, &xs, f32::exp);
    check_unary(OpKind::Exp2, &xs, |x| x.exp2() * core::f32::consts::LN_2);
    let pos = [0.5f32, 1.0, 2.7, 9.0];
    check_unary(OpKind::Ln, &pos, |x| 1.0 / x);
    check_unary(OpKind::Log2, &pos, |x| 1.0 / (x * core::f32::consts::LN_2));
    check_unary(OpKind::Log10, &pos, |x| {
        1.0 / (x * core::f32::consts::LN_10)
    });
}

#[test]
fn dwrt_sqrt_family_rules() {
    let pos = [0.5f32, 1.0, 2.7, 9.0];
    check_unary(OpKind::Sqrt, &pos, |x| 0.5 / x.sqrt());
    check_unary(OpKind::Rsqrt, &pos, |x| -0.5 * x.powf(-1.5));
    let xs = [0.3f32, 1.0, -0.7, 2.1];
    check_unary(OpKind::Recip, &xs, |x| -1.0 / (x * x));
}

#[test]
fn dwrt_piecewise_unary_rules() {
    // Points away from the kinks/steps (derivatives are a.e.).
    let xs = [0.3f32, 1.4, -0.7, 2.6, -2.2];
    check_unary(OpKind::Abs, &xs, f32::signum);
    check_unary(OpKind::Floor, &xs, |_| 0.0);
    check_unary(OpKind::Ceil, &xs, |_| 0.0);
    check_unary(OpKind::Round, &xs, |_| 0.0);
    check_unary(OpKind::Fract, &xs, |_| 1.0);
}

// ───────────────────────────── binary rules ───────────────────────────────

#[test]
fn dwrt_quotient_rule() {
    // f = X/Y ; ∂x = 1/y, ∂y = -x/y²
    let build = |a: &mut ExprArena| {
        let x = a.push_var(0);
        let y = a.push_var(1);
        a.push_binary(OpKind::Div, x, y)
    };
    let (ax, rx) = lowered(build, 0);
    let (ay, ry) = lowered(build, 1);
    for &(px, py) in PTS {
        close(ev(&ax, rx, px, py), 1.0 / py, "d(x/y)/dx");
        close(ev(&ay, ry, px, py), -px / (py * py), "d(x/y)/dy");
    }
}

#[test]
fn dwrt_min_max_select_active_branch() {
    // f = min(X², Y) ; ∂x = 2x where x² < y, else 0. Max symmetric.
    let build_min = |a: &mut ExprArena| {
        let x = a.push_var(0);
        let y = a.push_var(1);
        let xx = a.push_binary(OpKind::Mul, x, x);
        a.push_binary(OpKind::Min, xx, y)
    };
    let build_max = |a: &mut ExprArena| {
        let x = a.push_var(0);
        let y = a.push_var(1);
        let xx = a.push_binary(OpKind::Mul, x, x);
        a.push_binary(OpKind::Max, xx, y)
    };
    let (amin, rmin) = lowered(build_min, 0);
    let (amax, rmax) = lowered(build_max, 0);
    for &(px, py) in &[(0.5f32, 3.0f32), (2.0, 1.0), (-1.5, 4.0), (-3.0, 0.5)] {
        let min_want = if px * px < py { 2.0 * px } else { 0.0 };
        let max_want = if px * px > py { 2.0 * px } else { 0.0 };
        close(ev(&amin, rmin, px, py), min_want, "d(min(x²,y))/dx");
        close(ev(&amax, rmax, px, py), max_want, "d(max(x²,y))/dx");
    }
}

#[test]
fn dwrt_pow_rule() {
    // f = X^2.5 ; ∂x = 2.5·x^1.5  (x > 0)
    let (a, r) = lowered(
        |a| {
            let x = a.push_var(0);
            let e = a.push_const(2.5);
            a.push_binary(OpKind::Pow, x, e)
        },
        0,
    );
    for &x in &[0.5f32, 1.0, 2.0, 4.0] {
        close(ev(&a, r, x, 0.0), 2.5 * x.powf(1.5), "d(x^2.5)/dx");
    }

    // f = 2^X via Pow ; ∂x = 2^x · ln 2
    let (a, r) = lowered(
        |a| {
            let b = a.push_const(2.0);
            let x = a.push_var(0);
            a.push_binary(OpKind::Pow, b, x)
        },
        0,
    );
    for &x in &[0.0f32, 1.0, -1.5, 2.5] {
        close(
            ev(&a, r, x, 0.0),
            x.exp2() * core::f32::consts::LN_2,
            "d(2^x)/dx",
        );
    }
}

#[test]
fn dwrt_hypot_rule() {
    // f = hypot(X, Y) ; ∂x = x/hypot, ∂y = y/hypot
    let build = |a: &mut ExprArena| {
        let x = a.push_var(0);
        let y = a.push_var(1);
        a.push_binary(OpKind::Hypot, x, y)
    };
    let (ax, rx) = lowered(build, 0);
    let (ay, ry) = lowered(build, 1);
    for &(px, py) in PTS {
        let h = px.hypot(py);
        close(ev(&ax, rx, px, py), px / h, "d(hypot)/dx");
        close(ev(&ay, ry, px, py), py / h, "d(hypot)/dy");
    }
}

#[test]
fn dwrt_atan2_rule() {
    // Node order Binary(Atan2, a, b) evaluates a.atan2(b) = angle of (b, a):
    // a plays "y", b plays "x". ∂/∂a = b/(a²+b²), ∂/∂b = -a/(a²+b²).
    let build = |a: &mut ExprArena| {
        let x = a.push_var(0); // "y" operand
        let y = a.push_var(1); // "x" operand
        a.push_binary(OpKind::Atan2, x, y)
    };
    let (ax, rx) = lowered(build, 0);
    let (ay, ry) = lowered(build, 1);
    for &(px, py) in PTS {
        let d = px * px + py * py;
        close(ev(&ax, rx, px, py), py / d, "d(atan2)/d first");
        close(ev(&ay, ry, px, py), -px / d, "d(atan2)/d second");
    }
}

#[test]
fn dwrt_comparisons_are_zero() {
    for op in [
        OpKind::Lt,
        OpKind::Le,
        OpKind::Gt,
        OpKind::Ge,
        OpKind::Eq,
        OpKind::Ne,
    ] {
        let (a, r) = lowered(
            |a| {
                let x = a.push_var(0);
                let y = a.push_var(1);
                a.push_binary(op, x, y)
            },
            0,
        );
        for &(px, py) in PTS {
            close(ev(&a, r, px, py), 0.0, &format!("d({op:?})/dx"));
        }
    }
}

// ───────────────────────────── ternary rules ──────────────────────────────

#[test]
fn dwrt_mul_add_rule() {
    // f = X·Y + X ; ∂x = y + 1, ∂y = x
    let build = |a: &mut ExprArena| {
        let x = a.push_var(0);
        let y = a.push_var(1);
        a.push_ternary(OpKind::MulAdd, x, y, x)
    };
    let (ax, rx) = lowered(build, 0);
    let (ay, ry) = lowered(build, 1);
    for &(px, py) in PTS {
        close(ev(&ax, rx, px, py), py + 1.0, "d(xy+x)/dx");
        close(ev(&ay, ry, px, py), px, "d(xy+x)/dy");
    }
}

#[test]
fn dwrt_select_follows_branch() {
    // f = select(X > Y, X², 3Y) ; ∂x = 2x where x>y else 0
    let (a, r) = lowered(
        |a| {
            let x = a.push_var(0);
            let y = a.push_var(1);
            let c = a.push_binary(OpKind::Gt, x, y);
            let xx = a.push_binary(OpKind::Mul, x, x);
            let three = a.push_const(3.0);
            let ty = a.push_binary(OpKind::Mul, three, y);
            a.push_ternary(OpKind::Select, c, xx, ty)
        },
        0,
    );
    for &(px, py) in PTS {
        let want = if px > py { 2.0 * px } else { 0.0 };
        close(ev(&a, r, px, py), want, "d(select)/dx");
    }
}

#[test]
fn dwrt_clamp_follows_branch() {
    // f = clamp(X, -1, 1) ; ∂x = 1 inside, 0 clamped
    let (a, r) = lowered(
        |a| {
            let x = a.push_var(0);
            let lo = a.push_const(-1.0);
            let hi = a.push_const(1.0);
            a.push_ternary(OpKind::Clamp, x, lo, hi)
        },
        0,
    );
    for &(x, want) in &[(0.3f32, 1.0f32), (-0.9, 1.0), (1.7, 0.0), (-4.0, 0.0)] {
        close(ev(&a, r, x, 0.0), want, "d(clamp)/dx");
    }
}

// ─────────────────────── structure: DAG, nesting, memory ───────────────────

#[test]
fn dwrt_nested_second_derivatives() {
    // ∂²(x³)/∂x² = 6x
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let xx = a.push_binary(OpKind::Mul, x, x);
    let xxx = a.push_binary(OpKind::Mul, xx, x);
    let d1 = dwrt(&mut a, xxx, 0);
    let d2 = dwrt(&mut a, d1, 0);
    let (arena, root) = lower_dwrt_owned(&a, d2).expect("nested lowering failed");
    assert_dwrt_free(&arena, root);
    for &x in &[0.5f32, 1.0, -2.0, 3.0] {
        close(ev(&arena, root, x, 0.0), 6.0 * x, "d²(x³)/dx²");
    }

    // Mixed partial ∂²(x²y)/∂x∂y = 2x
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let xx = a.push_binary(OpKind::Mul, x, x);
    let f = a.push_binary(OpKind::Mul, xx, y);
    let dx = dwrt(&mut a, f, 0);
    let dxy = dwrt(&mut a, dx, 1);
    let (arena, root) = lower_dwrt_owned(&a, dxy).expect("mixed partial failed");
    assert_dwrt_free(&arena, root);
    for &(px, py) in PTS {
        close(ev(&arena, root, px, py), 2.0 * px, "d²(x²y)/dxdy");
    }
}

#[test]
fn dwrt_shared_subexpression_stays_shared() {
    // g = X·Y used twice: f = g·g. ∂x = 2·g·y. The derivative walk must
    // reference the existing g node, not duplicate its subtree per use.
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let g = a.push_binary(OpKind::Mul, x, y);
    let f = a.push_binary(OpKind::Mul, g, g);
    let root = dwrt(&mut a, f, 0);
    let (arena, root) = lower_dwrt_owned(&a, root).expect("lowering failed");
    assert_dwrt_free(&arena, root);
    for &(px, py) in PTS {
        close(ev(&arena, root, px, py), 2.0 * px * py * py, "d((xy)²)/dx");
    }
}

#[test]
fn dwrt_gather_is_piecewise_constant() {
    // A buffer read is a hard step in its index — derivative 0 by convention.
    use pixelflow_ir::arena::BufferDecl;
    let mut a = ExprArena::new();
    let b = a.declare_buffer(BufferDecl {
        width: 4,
        height: 1,
    });
    let x = a.push_var(0);
    let zero = a.push_const(0.0);
    let g = a.push_gather(b, x, zero);
    let root = dwrt(&mut a, g, 0);
    let (arena, root) = lower_dwrt_owned(&a, root).expect("gather lowering failed");
    assert_dwrt_free(&arena, root);
    let buf = [1.0f32, 5.0, 9.0, 13.0];
    let bindings = BindingTable::bind(&arena, &[&buf]).unwrap();
    for &x in &[0.2f32, 1.7, 3.0] {
        let got = eval_scalar(&arena, root, &[x, 0.0, 0.0, 0.0], &bindings);
        close(got, 0.0, "d(gather)/dx");
    }
}

// ─────────────────────────────── error paths ──────────────────────────────

#[test]
fn dwrt_unsupported_op_errors_loudly_naming_the_op() {
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let band = a.push_binary(OpKind::BitAnd, x, y);
    let root = dwrt(&mut a, band, 0);
    let err = must_err(
        lower_dwrt_owned(&a, root),
        "bitand must have no derivative rule",
    );
    assert_eq!(err, LowerError::UnsupportedOp { op: OpKind::BitAnd });
    let msg = format!("{err}");
    assert!(msg.contains("bitand"), "error must name the op: {msg}");
    assert!(err.as_static_str().contains("bitand"));

    // Reduce likewise has no rule.
    let mut a = ExprArena::new();
    let i = a.push_var(4);
    let red = a.push_reduce(OpKind::Add, 4, 3, i);
    let root = dwrt(&mut a, red, 0);
    let err = must_err(
        lower_dwrt_owned(&a, root),
        "reduce must have no derivative rule",
    );
    assert_eq!(err, LowerError::UnsupportedOp { op: OpKind::Reduce });
}

#[test]
fn dwrt_depth_bound_errors_loudly() {
    // A pathologically deep chain under Dwrt must produce DepthExceeded, not a
    // stack overflow.
    let mut a = ExprArena::new();
    let mut e = a.push_var(0);
    for _ in 0..600 {
        e = a.push_unary(OpKind::Neg, e);
    }
    let root = dwrt(&mut a, e, 0);
    let err = must_err(
        lower_dwrt_owned(&a, root),
        "deep chain must exceed the bound",
    );
    assert!(
        matches!(err, LowerError::DepthExceeded { .. }),
        "want DepthExceeded, got {err:?}"
    );
}

#[test]
fn dwrt_var_operand_validation() {
    // Non-const var operand.
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let root = a.push_binary(OpKind::Dwrt, x, y);
    let err = must_err(lower_dwrt_owned(&a, root), "non-const var must be rejected");
    assert_eq!(err, LowerError::DwrtVarNotConst);

    // Out-of-range var index.
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let v = a.push_const(7.0);
    let root = a.push_binary(OpKind::Dwrt, x, v);
    let err = must_err(lower_dwrt_owned(&a, root), "var 7 must be rejected");
    assert_eq!(err, LowerError::DwrtVarOutOfRange { value: 7.0 });
}

// ──────────────── font-ramp differential + interpreter/JIT parity ───────────

/// The font-ramp-shaped coverage expression: the exact op set the font path
/// exercises (sub, mul, div, comparison, select gating, sqrt, clamp) with the
/// gradient magnitude computed from `Dwrt` nodes over a shared subexpression.
///
///   g   = X − (2Y + 0.5)/(Y + 3)
///   f   = select(Y > 0, g, g·0.5)
///   cov = clamp(0.5 − f/√(DX(f)² + DY(f)²), 0, 1)
fn font_ramp_coverage(a: &mut ExprArena) -> ExprId {
    let x = a.push_var(0);
    let y = a.push_var(1);
    let two = a.push_const(2.0);
    let half = a.push_const(0.5);
    let three = a.push_const(3.0);
    let zero = a.push_const(0.0);
    let one = a.push_const(1.0);

    let num = {
        let t = a.push_binary(OpKind::Mul, two, y);
        a.push_binary(OpKind::Add, t, half)
    };
    let den = a.push_binary(OpKind::Add, y, three);
    let ramp = a.push_binary(OpKind::Div, num, den);
    let g = a.push_binary(OpKind::Sub, x, ramp);

    let gate = a.push_binary(OpKind::Gt, y, zero);
    let g_half = a.push_binary(OpKind::Mul, g, half);
    let f = a.push_ternary(OpKind::Select, gate, g, g_half);

    let fx = dwrt(a, f, 0);
    let fy = dwrt(a, f, 1);
    let fx2 = a.push_binary(OpKind::Mul, fx, fx);
    let fy2 = a.push_binary(OpKind::Mul, fy, fy);
    let mag2 = a.push_binary(OpKind::Add, fx2, fy2);
    let mag = a.push_unary(OpKind::Sqrt, mag2);

    let ratio = a.push_binary(OpKind::Div, f, mag);
    let dist = a.push_binary(OpKind::Sub, half, ratio);
    a.push_ternary(OpKind::Clamp, dist, zero, one)
}

/// Analytic reference for [`font_ramp_coverage`].
fn font_ramp_reference(x: f32, y: f32) -> f32 {
    let g = x - (2.0 * y + 0.5) / (y + 3.0);
    // d(ramp)/dy = (2(y+3) − (2y+0.5))/(y+3)² = 5.5/(y+3)²
    let ramp_dy = 5.5 / ((y + 3.0) * (y + 3.0));
    let (f, fx, fy) = if y > 0.0 {
        (g, 1.0, -ramp_dy)
    } else {
        (0.5 * g, 0.5, -0.5 * ramp_dy)
    };
    let mag = (fx * fx + fy * fy).sqrt();
    (0.5 - f / mag).clamp(0.0, 1.0)
}

/// Grid avoiding the select boundary (y = 0) and the pole (y = -3).
const GRID_X: &[f32] = &[-1.5, -0.4, 0.0, 0.3, 1.2, 2.0];
const GRID_Y: &[f32] = &[-1.5, -0.6, 0.4, 1.1, 2.3];

#[test]
fn font_ramp_interpreter_matches_analytic() {
    let mut a = ExprArena::new();
    let root = font_ramp_coverage(&mut a);
    for &y in GRID_Y {
        for &x in GRID_X {
            let got = ev(&a, root, x, y);
            let want = font_ramp_reference(x, y);
            close(got, want, &format!("font ramp at ({x}, {y})"));
        }
    }
}

/// Interpreter == JIT on the differential font-ramp arena: both run the same
/// `lower_dwrt` precondition, so they must agree to float tolerance.
#[test]
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", not(target_feature = "avx512f"))
))]
fn font_ramp_jit_matches_interpreter() {
    use pixelflow_ir::backend::emit::{compile_arena_dag, executable};

    let mut a = ExprArena::new();
    let root = font_ramp_coverage(&mut a);
    let compiled = compile_arena_dag(&a, root).expect("JIT compile of Dwrt arena failed");

    let jit_eval = |x: f32, y: f32| -> f32 {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use core::arch::aarch64::*;
            let f: executable::KernelFn = compiled.code.as_fn();
            let out = f(
                vdupq_n_f32(x),
                vdupq_n_f32(y),
                vdupq_n_f32(0.0),
                vdupq_n_f32(0.0),
            );
            vgetq_lane_f32(out, 0)
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use core::arch::x86_64::*;
            let f: executable::KernelFn = compiled.code.as_fn();
            let out = f(
                _mm_set1_ps(x),
                _mm_set1_ps(y),
                _mm_set1_ps(0.0),
                _mm_set1_ps(0.0),
            );
            _mm_cvtss_f32(out)
        }
    };

    for &y in GRID_Y {
        for &x in GRID_X {
            let interp = ev(&a, root, x, y);
            let jit = jit_eval(x, y);
            let tol = 1e-5 + 1e-5 * interp.abs();
            assert!(
                (jit - interp).abs() <= tol,
                "interpreter/JIT disagree at ({x}, {y}): interp={interp} jit={jit}"
            );
        }
    }
}
