//! W0-C (audit H1/H2): the scalar oracle as an independent correctness
//! reference for the JIT, plus the per-op equivalence tolerances.
//!
//! Part 1 pins the target-agreed semantics table (CLAUDE.md "Floating point at
//! the edges") on the oracle itself: ordered `Lt`/`Le`, exact `Eq`/`Ne`,
//! all-ones/all-zero mask lanes, bitwise `Select`/`BitAnd`/`BitOr`, and
//! exp/exp2 saturation.
//!
//! Part 2 is the property sweep the audit asked for: representative arenas
//! (arithmetic, comparisons+select, transcendentals, fma, estimate ops)
//! evaluated by the JIT and by `eval_scalar` across magnitude extremes
//! (1e-4..1e4, negatives, ±0.0), asserted within `equivalence_tolerance` —
//! per-op allowances, not a global absolute band. Inputs where the language
//! promises nothing (`fold_is_platform_specific`) are skipped, which is the
//! documented caller protocol for these tolerances.

use pixelflow_codegen::jit_cache;
use pixelflow_ir::{
    BindingTable, ExprArena, ExprId, ExprNode, OpKind, Tolerance, equivalence_tolerance,
    eval_scalar,
};

// ── JIT invocation at this build's width ─────────────────────────────────────

const LANES: usize = pixelflow_codegen::JIT_VECTOR_BYTES / 4;

// ── Sweep machinery ──────────────────────────────────────────────────────────

/// Magnitude extremes per the audit's H1 fix: 1e-4..1e4, negatives, ±0.0.
const MAGS: [f32; 14] = [
    -1e4, -1e2, -3.7, -1.0, -0.5, -1e-4, -0.0, 0.0, 1e-4, 0.5, 1.0, 3.7, 1e2, 1e4,
];

/// One call's worth of sweep: the values of the kernel's two arguments, which
/// are fixed for the whole call, and the coordinates to sample under them.
///
/// A sweep over four inputs used to be one flat point list, because Z and W
/// were coordinates and varied per lane. They are uniforms now, so the sweep
/// is nested: a block, then the lattice under it.
struct Sweep {
    block: [f32; 2],
    coords: Vec<[f32; 2]>,
}

/// The coordinate half of the grid: every (x, y) pair of magnitude extremes.
fn coord_grid() -> Vec<[f32; 2]> {
    let mut pts = Vec::new();
    for &x in &MAGS {
        for &y in &MAGS {
            pts.push([x, y]);
        }
    }
    pts
}

fn grid_x() -> Vec<Sweep> {
    vec![Sweep {
        block: [0.0, 0.0],
        coords: MAGS.iter().map(|&x| [x, 0.6]).collect(),
    }]
}

fn grid_xy() -> Vec<Sweep> {
    vec![Sweep {
        block: [0.6, -0.3],
        coords: coord_grid(),
    }]
}

fn grid_xyzw() -> Vec<Sweep> {
    let mut sweeps = Vec::new();
    for &z in &MAGS {
        for &w in &MAGS {
            sweeps.push(Sweep {
                block: [z, w],
                coords: coord_grid(),
            });
        }
    }
    sweeps
}

/// The two lattice-invariant arguments a sweep arena may declare, in link
/// order — what `Var(2)` and `Var(3)` became.
struct Args {
    z: pixelflow_ir::Uniform,
    w: pixelflow_ir::Uniform,
}

/// Declare both arguments in `a` and return their leaves.
fn args(a: &mut ExprArena) -> (Args, ExprId, ExprId) {
    let (z, w) = (
        pixelflow_ir::Uniform::new(0.0),
        pixelflow_ir::Uniform::new(0.0),
    );
    let (zs, ws) = (a.declare_uniform(z.decl()), a.declare_uniform(w.decl()));
    let (zl, wl) = (a.push_uniform(zs), a.push_uniform(ws));
    (Args { z, w }, zl, wl)
}

/// The intended caller pattern: fold per-op allowances over the nodes reachable
/// from the root into one whole-expression tolerance.
fn expression_tolerance(arena: &ExprArena, root: ExprId) -> Tolerance {
    let mut tol = Tolerance::BitExact;
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        let op = match arena.node(id) {
            ExprNode::Var(_) => OpKind::Var,
            ExprNode::Const(_) => OpKind::Const,
            ExprNode::Buffer(_) => OpKind::Buffer,
            ExprNode::Uniform(_) => OpKind::Uniform,
            ExprNode::Param(p) => panic!("expression_tolerance: unbound Param({p})"),
            ExprNode::Unary(op, _)
            | ExprNode::Binary(op, _, _)
            | ExprNode::Ternary(op, _, _, _)
            | ExprNode::Nary(op, _, _) => *op,
        };
        tol = tol.loosest(equivalence_tolerance(op));
        for c in arena.children(id) {
            stack.push(c);
        }
    }
    tol
}

/// What a sweep is *of*: the expression, and the arguments it declares.
struct Subject<'a> {
    name: &'a str,
    arena: &'a ExprArena,
    root: ExprId,
    /// The two lattice-invariant arguments, when the arena declares them.
    declared: Option<&'a Args>,
}

/// Compile once, then compare JIT vectors against the oracle across all
/// non-skipped points in SIMD batches, under the expression's folded tolerance.
fn assert_jit_matches_oracle(
    subject: Subject<'_>,
    sweeps: &[Sweep],
    skip: impl Fn(&[f32; 4]) -> bool,
) {
    let Subject {
        name,
        arena,
        root,
        declared,
    } = subject;
    let tol = expression_tolerance(arena, root);
    let k = pixelflow_ir::Kernel::from_parts(arena.clone(), root);
    let jit = jit_cache::compile(&k, pixelflow_ir::LatticeShape::POINT)
        .unwrap_or_else(|e| panic!("{name}: kernel failed to compile on this backend: {e}"))
        .kernel;
    let mut checked = 0usize;
    for sweep in sweeps {
        let block = sweep.block;
        // The oracle reads the same block the code does: by identity, not by
        // the order either side happens to hold it in.
        let bindings = match declared {
            Some(a) => BindingTable::empty()
                .bind_uniforms(
                    arena,
                    &[(a.z.identity(), block[0]), (a.w.identity(), block[1])],
                )
                .expect("both arguments are declared in this arena"),
            None => BindingTable::empty(),
        };
        let ctx: [*const f32; 1] = [block.as_ptr()];
        let valid: Vec<[f32; 2]> = sweep
            .coords
            .iter()
            .copied()
            .filter(|c| !skip(&[c[0], c[1], block[0], block[1]]))
            .collect();
        for chunk in valid.chunks(LANES) {
            let mut x = [0.0f32; LANES];
            let mut y = [0.0f32; LANES];
            for (i, c) in chunk.iter().enumerate() {
                x[i] = c[0];
                y[i] = c[1];
            }
            // SAFETY: `ctx` holds this kernel's block — one `f32` per declared
            // argument, in link order — alive for the call, and the vector
            // width is this build's.
            let res = unsafe {
                jit.call_bound(
                    ctx.as_ptr(),
                    pixelflow_codegen::Point4::new(x, y, [0.0; LANES], [0.0; LANES]),
                )
            };
            for (i, c) in chunk.iter().enumerate() {
                let got = res[i];
                let want = eval_scalar(arena, root, c, &bindings);
                let p = [c[0], c[1], block[0], block[1]];
                assert!(
                    tol.accepts(got, want),
                    "{name}{p:?}: JIT {got} ({:#010x}) vs oracle {want} ({:#010x}) exceeds {tol:?}",
                    got.to_bits(),
                    want.to_bits(),
                );
                checked += 1;
            }
        }
    }
    // A sweep that skipped everything verified nothing — fail loudly.
    assert!(checked > 0, "{name}: sweep skipped every point");
}

// ── Part 1: pinned target-agreed semantics, on the oracle itself ─────────────

const T: u32 = u32::MAX; // an all-ones mask lane
const F: u32 = 0;

fn eval1(arena: &ExprArena, root: ExprId, x: f32, y: f32) -> f32 {
    eval_scalar(arena, root, &[x, y], &BindingTable::empty())
}

#[test]
fn lt_le_are_ordered_and_results_are_masks() {
    let nan = f32::NAN;
    for op in [OpKind::Lt, OpKind::Le] {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let root = a.push_binary(op, x, y);
        // Ordered: false for any NaN operand, on every target.
        for (px, py) in [(nan, 1.0), (1.0, nan), (nan, nan)] {
            assert_eq!(eval1(&a, root, px, py).to_bits(), F, "{op:?}({px}, {py})");
        }
        // And the true/false results are bit patterns, never 1.0/0.0-as-number.
        assert_eq!(eval1(&a, root, 1.0, 2.0).to_bits(), T, "{op:?}(1, 2)");
        assert_eq!(eval1(&a, root, 2.0, 1.0).to_bits(), F, "{op:?}(2, 1)");
    }
}

#[test]
fn eq_ne_are_exact_and_nan_aware() {
    let nan = f32::NAN;
    let near = f32::from_bits(0.5f32.to_bits() + 1);

    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let eq = a.push_binary(OpKind::Eq, x, y);
    let ne = a.push_binary(OpKind::Ne, x, y);

    // Exact: one-ulp neighbours are NOT equal (no epsilon smearing).
    assert_eq!(eval1(&a, eq, 0.5, near).to_bits(), F);
    assert_eq!(eval1(&a, ne, 0.5, near).to_bits(), T);
    assert_eq!(eval1(&a, eq, 0.5, 0.5).to_bits(), T);

    // NaN: never equal, always unequal — agreed by every target.
    assert_eq!(eval1(&a, eq, nan, nan).to_bits(), F);
    assert_eq!(eval1(&a, ne, nan, nan).to_bits(), T);
    assert_eq!(eval1(&a, eq, nan, 1.0).to_bits(), F);
    assert_eq!(eval1(&a, ne, 1.0, nan).to_bits(), T);
}

/// Gt/Ge on NaN diverge by target (x86 unordered-true, aarch64 ordered-false),
/// so the language promises nothing and `fold_is_platform_specific` flags the
/// inputs. The oracle still has to answer *something*; its documented single
/// choice is x86's unordered predicate, and this pins that choice.
#[test]
fn gt_ge_pin_the_documented_unordered_choice() {
    let nan = f32::NAN;
    for op in [OpKind::Gt, OpKind::Ge] {
        assert!(op.fold_is_platform_specific(&[nan, 1.0]));
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let root = a.push_binary(op, x, y);
        for (px, py) in [(nan, 1.0), (1.0, nan), (nan, nan)] {
            assert_eq!(eval1(&a, root, px, py).to_bits(), T, "{op:?}({px}, {py})");
        }
        // Away from NaN both targets agree and the oracle must too.
        assert_eq!(eval1(&a, root, 2.0, 1.0).to_bits(), T);
        assert_eq!(eval1(&a, root, 1.0, 2.0).to_bits(), F);
    }
}

#[test]
fn select_is_a_bitwise_blend_not_a_truthy_branch() {
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let t = a.push_const(7.0);
    let f = a.push_const(9.0);

    // Canonical mask from a comparison: picks each branch exactly.
    let m = a.push_binary(OpKind::Lt, x, y);
    let sel = a.push_ternary(OpKind::Select, m, t, f);
    assert_eq!(eval1(&a, sel, 1.0, 2.0), 7.0);
    assert_eq!(eval1(&a, sel, 2.0, 1.0), 9.0);

    // Non-canonical "mask" 1.0: the blend is bitwise, so the result is the
    // bit formula's value — NOT 7.0. Spelling true as 1.0 is the mask bug.
    let fake = a.push_const(1.0);
    let sel_fake = a.push_ternary(OpKind::Select, fake, t, f);
    let m_bits = 1.0f32.to_bits();
    let expect = (m_bits & 7.0f32.to_bits()) | (!m_bits & 9.0f32.to_bits());
    let got = eval1(&a, sel_fake, 0.0, 0.0);
    assert_eq!(got.to_bits(), expect);
    assert_ne!(got, 7.0, "a truthy select would have returned the branch");
}

#[test]
fn bitand_bitor_operate_on_lane_bits() {
    let px = f32::from_bits(0xDEAD_BEEF);
    let py = f32::from_bits(0x0F0F_0F0F);
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let and = a.push_binary(OpKind::BitAnd, x, y);
    let or = a.push_binary(OpKind::BitOr, x, y);
    assert_eq!(eval1(&a, and, px, py).to_bits(), 0xDEAD_BEEF & 0x0F0F_0F0F);
    assert_eq!(eval1(&a, or, px, py).to_bits(), 0xDEAD_BEEF | 0x0F0F_0F0F);
}

#[test]
fn exp_and_exp2_saturate_rather_than_overflow() {
    for op in [OpKind::Exp, OpKind::Exp2] {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let root = a.push_unary(op, x);
        // Past a ±126 exponent std overflows to inf; the expansion clamps.
        let hi = eval1(&a, root, 200.0, 0.0);
        assert!(hi.is_finite(), "{op:?}(200) must saturate, got {hi}");
        assert!(hi > 1e30, "{op:?}(200) saturates high, got {hi}");
        let lo = eval1(&a, root, -200.0, 0.0);
        assert!(lo.is_finite() && lo > 0.0, "{op:?}(-200) got {lo}");
        assert!(lo < 1e-30, "{op:?}(-200) saturates low, got {lo}");
    }
}

/// An op with no scalar semantics is a loud panic, never a default value.
#[test]
#[should_panic(expected = "no scalar eval")]
fn unsupported_op_panics_with_the_op_name() {
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let zero = a.push_const(0.0);
    let root = a.push_binary(OpKind::Dwrt, x, zero);
    // Bound (not `let _`) so the workspace's must-use lints hold; the call is
    // expected to panic before the value exists.
    let _unreachable = eval_scalar(&a, root, &[1.0; 2], &BindingTable::empty());
}

// ── The tolerance table itself ───────────────────────────────────────────────

#[test]
fn mask_and_pattern_ops_demand_bit_equality() {
    for op in [
        OpKind::Lt,
        OpKind::Le,
        OpKind::Gt,
        OpKind::Ge,
        OpKind::Eq,
        OpKind::Ne,
        OpKind::Select,
        OpKind::BitAnd,
        OpKind::BitOr,
        OpKind::TruncToInt,
        OpKind::IntToFloat,
        OpKind::IAdd,
        OpKind::Shl,
        OpKind::Shr,
    ] {
        assert!(op.is_bitwise_domain(), "{op:?} classification drifted");
        assert_eq!(equivalence_tolerance(op), Tolerance::BitExact, "{op:?}");
    }
}

#[test]
fn estimate_ops_get_the_hardware_estimate_band() {
    for op in [OpKind::Recip, OpKind::Rsqrt] {
        match equivalence_tolerance(op) {
            Tolerance::Numeric { rel, .. } => {
                // Must cover the loosest backend: SSE2/AVX2 rcpps at ~12 bits
                // (max relative error 1.5 * 2^-12).
                assert!(rel >= 1.5 * (0.5f32).powi(12), "{op:?} rel {rel} too tight");
                assert!(rel <= 1e-2, "{op:?} rel {rel} loose enough to hide bugs");
            }
            t => panic!("{op:?} is an estimate, got {t:?}"),
        }
    }
}

#[test]
fn accepts_is_bit_aware_and_nan_aware() {
    let exact = Tolerance::BitExact;
    let near = f32::from_bits(1.0f32.to_bits() + 1);
    assert!(exact.accepts(1.0, 1.0));
    assert!(!exact.accepts(1.0, near));
    // Distinct NaN payloads are NOT interchangeable bit patterns: an all-ones
    // mask reads as NaN, and accepting any NaN would accept a broken mask.
    assert!(!exact.accepts(f32::NAN, f32::from_bits(u32::MAX)));

    let num = Tolerance::Numeric {
        rel: 1e-5,
        abs: 1e-6,
    };
    assert!(num.accepts(1.0, near));
    assert!(num.accepts(f32::NAN, f32::from_bits(0x7fc0_0001))); // both NaN
    assert!(num.accepts(f32::INFINITY, f32::INFINITY)); // same bits
    assert!(!num.accepts(f32::INFINITY, f32::NEG_INFINITY));
    assert!(!num.accepts(1.1, 1.0));
    // Signed zeros agree numerically (0.0 == -0.0) even though bits differ.
    assert!(num.accepts(0.0, -0.0));
}

#[test]
fn loosest_folds_toward_the_weaker_bound() {
    let a = Tolerance::Numeric {
        rel: 1e-5,
        abs: 1e-6,
    };
    let b = Tolerance::Numeric {
        rel: 1e-3,
        abs: 1e-7,
    };
    assert_eq!(Tolerance::BitExact.loosest(a), a);
    assert_eq!(a.loosest(Tolerance::BitExact), a);
    assert_eq!(
        a.loosest(b),
        Tolerance::Numeric {
            rel: 1e-3,
            abs: 1e-6
        }
    );
    assert_eq!(
        Tolerance::BitExact.loosest(Tolerance::BitExact),
        Tolerance::BitExact
    );
}

#[test]
#[should_panic(expected = "dwrt must be rewritten away")]
fn dwrt_has_no_equivalence_tolerance() {
    // Bound (not `let _`) so the workspace's must-use lints hold; the call is
    // expected to panic before the value exists.
    let _unreachable = equivalence_tolerance(OpKind::Dwrt);
}

// ── Part 2: property sweeps, JIT vs oracle under per-op tolerances ───────────

#[test]
fn jit_matches_oracle_arithmetic() {
    // ((X*Y + Z) - W) / (X + 2.5) — exact correctly-rounded ops only.
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let (declared, z, w) = args(&mut a);
    let xy = a.push_binary(OpKind::Mul, x, y);
    let s = a.push_binary(OpKind::Add, xy, z);
    let d = a.push_binary(OpKind::Sub, s, w);
    let c = a.push_const(2.5);
    let den = a.push_binary(OpKind::Add, x, c);
    let root = a.push_binary(OpKind::Div, d, den);
    assert_jit_matches_oracle(
        Subject {
            name: "arith",
            arena: &a,
            root,
            declared: Some(&declared),
        },
        &grid_xyzw(),
        |_| false,
    );
}

#[test]
fn jit_matches_oracle_exact_unaries_and_min_max() {
    // sqrt(|X|) + (-ceil(Y)) + min(|Z|, |W| + 1). The min operands are shaped
    // so no sweep point produces the (±0.0, ∓0.0) pair the language declines
    // to promise: |Z| is +0.0 at worst and |W|+1 is at least 1.
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let (declared, z, w) = args(&mut a);
    let sq = {
        let ax = a.push_unary(OpKind::Abs, x);
        a.push_unary(OpKind::Sqrt, ax)
    };
    let nc = {
        let cy = a.push_unary(OpKind::Ceil, y);
        a.push_unary(OpKind::Neg, cy)
    };
    let mn = {
        let az = a.push_unary(OpKind::Abs, z);
        let aw = a.push_unary(OpKind::Abs, w);
        let one = a.push_const(1.0);
        let aw1 = a.push_binary(OpKind::Add, aw, one);
        a.push_binary(OpKind::Min, az, aw1)
    };
    let t = a.push_binary(OpKind::Add, sq, nc);
    let root = a.push_binary(OpKind::Add, t, mn);
    assert_jit_matches_oracle(
        Subject {
            name: "exact_unaries",
            arena: &a,
            root,
            declared: Some(&declared),
        },
        &grid_xyzw(),
        // The two large addends — `-ceil(Y)` and the `Min` — cancel at some
        // grid points, and there the comparison stops being about the ops
        // this test names. The JIT's kernel is optimized and the oracle
        // evaluates the arena's own association; reassociating `+` is a valid
        // rewrite of the algebra (CLAUDE.md, "Floating point at the edges"),
        // so the two sides run different, equally correct, execution
        // histories. When the addends are 1e4 and the result is O(1), one ulp
        // of an addend is ~1e-3 and any relative tolerance on the *result* is
        // a tolerance on the execution history instead: at
        // `[-3.7, 10000, -10000, -10000]` the optimized form returns
        // `sqrt(3.7)` to the last bit and the oracle's association returns
        // 1.9238281. Skip the cancelling points; every op here is exercised
        // by the rest of the grid, where nothing cancels.
        |p| {
            let (big_neg, min_term) = (-p[1].ceil(), p[2].abs().min(p[3].abs() + 1.0));
            let largest = big_neg.abs().max(min_term.abs());
            largest > 1.0 && (big_neg + min_term).abs() <= largest * 1e-3
        },
    );
}

#[test]
fn jit_matches_oracle_min_max_where_promised() {
    // Direct Min/Max over the raw grid, skipping the inputs the language
    // declines to promise (NaN operands, opposite-signed zeros) — the
    // documented caller protocol for BitExact ops with divergent rows.
    for op in [OpKind::Min, OpKind::Max] {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let root = a.push_binary(op, x, y);
        assert_jit_matches_oracle(
            Subject {
                name: op.name(),
                arena: &a,
                root,
                declared: None,
            },
            &grid_xy(),
            |p| op.fold_is_platform_specific(&[p[0], p[1]]),
        );
    }
}

#[test]
fn jit_matches_oracle_round_away_from_ties() {
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let root = a.push_unary(OpKind::Round, x);
    let pts = vec![Sweep {
        block: [0.0, 0.0],
        coords: MAGS
            .iter()
            .chain(&[2.4, 2.6, -2.4, -2.6, 7.0, -7.0, 0.4999, 1234.56])
            .map(|&x| [x, 0.0])
            .collect(),
    }];
    assert_jit_matches_oracle(
        Subject {
            name: "round",
            arena: &a,
            root,
            declared: None,
        },
        &pts,
        |p| {
            // Ties break differently per target; (-0.5, -0.0] loses its sign in
            // the combinator tier. Both flagged; both skipped.
            OpKind::Round.fold_is_platform_specific(&[p[0]])
        },
    );
}

#[test]
fn jit_matches_oracle_comparisons_and_select() {
    // select(lt(X,Y) & ge(Z,W), X - Y, Z * W): mask plumbing end to end.
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let (declared, z, w) = args(&mut a);
    let lt = a.push_binary(OpKind::Lt, x, y);
    let ge = a.push_binary(OpKind::Ge, z, w);
    let m = a.push_binary(OpKind::BitAnd, lt, ge);
    let tb = a.push_binary(OpKind::Sub, x, y);
    let fb = a.push_binary(OpKind::Mul, z, w);
    let root = a.push_ternary(OpKind::Select, m, tb, fb);
    // The grid holds no NaN, so no comparison lands on a divergent row.
    assert_jit_matches_oracle(
        Subject {
            name: "cmp_select",
            arena: &a,
            root,
            declared: Some(&declared),
        },
        &grid_xyzw(),
        |_| false,
    );
}

#[test]
fn jit_matches_oracle_mask_root_bit_for_bit() {
    // A mask-valued root: lt(X,Y) | gt(Z,W). Folded tolerance is BitExact, so
    // this asserts the JIT writes the same all-ones/all-zero lanes the oracle
    // does — the mask contract, through compiled code.
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let (declared, z, w) = args(&mut a);
    let lt = a.push_binary(OpKind::Lt, x, y);
    let gt = a.push_binary(OpKind::Gt, z, w);
    let root = a.push_binary(OpKind::BitOr, lt, gt);
    assert_eq!(expression_tolerance(&a, root), Tolerance::BitExact);
    assert_jit_matches_oracle(
        Subject {
            name: "mask_root",
            arena: &a,
            root,
            declared: Some(&declared),
        },
        &grid_xyzw(),
        |_| false,
    );
}

#[test]
fn jit_matches_oracle_fma() {
    // The oracle rounds once (`libm::fmaf`); an FMA target agrees bit for
    // bit and an SSE2 target is one product-rounding away, inside
    // `EXACT_ARITH`. No point is skipped: rounding is tolerance, not
    // divergence.
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let (declared, z, _w) = args(&mut a);
    let root = a.push_ternary(OpKind::MulAdd, x, y, z);
    // The addend is the kernel's argument: one value per call, every (x, y)
    // pair swept under it.
    let sweeps: Vec<Sweep> = MAGS
        .iter()
        .map(|&z| Sweep {
            block: [z, 0.0],
            coords: coord_grid(),
        })
        .collect();
    assert_jit_matches_oracle(
        Subject {
            name: "fma",
            arena: &a,
            root,
            declared: Some(&declared),
        },
        &sweeps,
        |_| false,
    );
}

#[test]
fn jit_matches_oracle_recip_rsqrt() {
    for op in [OpKind::Recip, OpKind::Rsqrt] {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let root = a.push_unary(op, x);
        // ±0 → ±inf agrees by bits; negative rsqrt → NaN in both tiers; the
        // finite points must land inside the estimate band vs the exact oracle.
        assert_jit_matches_oracle(
            Subject {
                name: op.name(),
                arena: &a,
                root,
                declared: None,
            },
            &grid_x(),
            |_| false,
        );
    }
}

#[test]
fn jit_matches_oracle_int_primitives() {
    // int_to_float(trunc_to_int(X)) is trunc-toward-zero; every sweep value is
    // in i32 range, where the tiers are bit-identical.
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let ti = a.push_unary(OpKind::TruncToInt, x);
    let root = a.push_unary(OpKind::IntToFloat, ti);
    assert_jit_matches_oracle(
        Subject {
            name: "trunc_int",
            arena: &a,
            root,
            declared: None,
        },
        &grid_x(),
        |_| false,
    );
}

#[test]
fn jit_matches_oracle_transcendentals_unary() {
    // Both tiers evaluate the same expansion, so agreement here means the
    // backend encodes the expansion's primitives correctly — including the
    // TruncToInt/IAdd/Shl integer path inside exp2/log2.
    for op in [
        OpKind::Sin,
        OpKind::Cos,
        OpKind::Tan,
        OpKind::Exp,
        OpKind::Exp2,
        OpKind::Ln,
        OpKind::Log2,
        OpKind::Log10,
        OpKind::Asin,
        OpKind::Acos,
        OpKind::Atan,
    ] {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let root = a.push_unary(op, x);
        assert_jit_matches_oracle(
            Subject {
                name: op.name(),
                arena: &a,
                root,
                declared: None,
            },
            &grid_x(),
            |_| false,
        );
    }
}

#[test]
fn jit_matches_oracle_transcendentals_binary() {
    for op in [OpKind::Atan2, OpKind::Pow] {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let root = a.push_binary(op, x, y);
        assert_jit_matches_oracle(
            Subject {
                name: op.name(),
                arena: &a,
                root,
                declared: None,
            },
            &grid_xy(),
            |_| false,
        );
    }
}
