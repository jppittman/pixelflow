//! exp/log kernels through the JIT, checked numerically at this build's width.
//!
//! These lower (`expand_transcendentals`) to the integer-domain primitives
//! TruncToInt/IntToFloat/IAdd/Shl/Shr — ops a backend can silently lack while
//! every other test stays green. That is how Avx512Backend once shipped
//! refusing `ShiftImm`: `test_log2` passed on the retired combinator tier,
//! which computed its own log2/exp and never reached the JIT, the full
//! workspace passed under `+avx512f`, and a `Kernel::log2` still failed to
//! compile. The completeness sweep catches the emission gap; this catches the
//! semantics — the bytes must also be right.

use pixelflow_codegen::jit_cache;
use pixelflow_ir::{ExprArena, Kernel, Lit};

/// One point of a kernel compiled at [`pixelflow_ir::LatticeShape::POINT`],
/// read back through the origin block.
fn eval_point(jit: &pixelflow_codegen::emit::CompiledKernel, x: f32, y: f32) -> f32 {
    let mut out = [0.0f32; 1];
    let origin = [x, y];
    // SAFETY: every kernel this file compiles declares no buffer and no
    // uniform, so `ctx[0]` — the uniform-block slot — is unread; `ctx[1]` is
    // the origin block, and `out` holds the one sample a single-point
    // lattice writes.
    let ctx: [*const f32; 2] = [core::ptr::null(), origin.as_ptr()];
    unsafe {
        jit.call(ctx.as_ptr(), out.as_mut_ptr(), 1);
    }
    out[0]
}

fn eval_points(jit: &pixelflow_codegen::emit::CompiledKernel, inputs: &[(f32, f32)]) -> Vec<f32> {
    inputs.iter().map(|&(x, y)| eval_point(jit, x, y)).collect()
}

fn check(name: &str, k: &Kernel, inputs: &[f32], reference: impl Fn(f32) -> f32) {
    let jit = jit_cache::compile(k, pixelflow_ir::LatticeShape::POINT)
        .unwrap_or_else(|e| panic!("{name}: kernel failed to compile on this backend: {e}"))
        .kernel;
    for &x in inputs {
        let got = eval_point(&jit, x, 0.0);
        let want = reference(x);
        let rel = ((got - want) / want.abs().max(1e-6)).abs();
        assert!(
            rel <= 1e-3 || (got - want).abs() <= 1e-4,
            "{name}({x}): got {got}, want {want} (rel {rel:.2e})"
        );
    }
}

#[test]
fn exp_log_kernels_compile_and_agree_with_std() {
    // Log domains can range wide; exp inputs stay where e^x is a finite f32.
    // The expansion clamps its exponent to ±126 by design (expand_exp2,
    // lowering.rs) — the JIT saturates where std overflows to inf, so inputs
    // past ~88.7 (ln(f32::MAX)) compare a clamp against an inf, which is a
    // contract difference, not an encoding bug.
    let logs = [0.5f32, 1.0, 2.0, 8.0, 100.0, 4096.0];
    let exps = [-10.0f32, -1.0, 0.0, 0.5, 1.0, 8.0, 80.0];
    check("log2", &Kernel::x().log2(), &logs, f32::log2);
    check("ln", &Kernel::x().ln(), &logs, f32::ln);
    check("exp", &Kernel::x().exp(), &exps, f32::exp);
    check("exp2", &Kernel::x().exp2(), &exps, f32::exp2);
}

// ── Floating-point edge cases: the documented contract, not IEEE ─────────────
//
// The language's FP contract (CLAUDE.md, "Floating point at the edges") is
// *hardware semantics*, deliberately not scalar Rust's: NaN comparisons,
// Min/Max NaN handling and Round's tie-breaking follow whatever the SIMD
// instruction does, because matching `f32::min`/`f32::round` would cost extra
// instructions in the hot path for cases the language does not promise.
//
// What the contract DOES require is that the two tiers agree: the JIT and
// `OpKind::eval_*` (the differential oracle, and the e-graph's constant
// folder) must produce the same answer, or a folded expression differs from
// the same expression computed at runtime. These tests pin that agreement at
// whatever width the build selects — they are the reason the oracle was
// changed to match the hardware rather than the encoders changed to match
// scalar Rust.

/// Every lane of a binary kernel, JIT vs oracle, on edge-case inputs; `agree`
/// is what counts as the same answer from the two tiers.
fn assert_tiers_agree_binary(
    name: &str,
    k: &Kernel,
    op: pixelflow_ir::OpKind,
    agree: impl Fn(f32, f32) -> bool,
) {
    let jit = jit_cache::compile(k, pixelflow_ir::LatticeShape::POINT)
        .unwrap_or_else(|e| panic!("{name}: {e}"))
        .kernel;
    let nan = f32::NAN;
    let inputs: Vec<(f32, f32)> = [
        (1.0f32, nan),
        (nan, 1.0f32),
        (nan, nan),
        (2.5, 2.5),
        (-0.0, 0.0),
        (1.0, 2.0),
        (2.0, 1.0),
        (f32::INFINITY, 1.0),
    ]
    .into_iter()
    .filter(|&(x, y)| !op.fold_is_platform_specific(&[x, y]))
    .collect();

    let results = eval_points(&jit, &inputs);
    for (&(x, y), got) in inputs.iter().zip(results) {
        // The oracle answers with a typed literal and the JIT wrote a lane, so
        // the comparison is in the lane the literal tabulates to: a real's bits,
        // a mask's all-ones or all-zero (`Lit::bits`). `Lit::real` would refuse
        // the comparisons' masks, which is the very answer this test pins.
        let want = op
            .eval_binary(Lit::Real(x), Lit::Real(y))
            .map(|lit| f32::from_bits(lit.bits()))
            .expect("oracle covers this op");
        assert!(
            agree(got, want),
            "{name}({x}, {y}): JIT gave {got} ({:#010x}), oracle gave {want} ({:#010x}) — \
             tiers must agree",
            got.to_bits(),
            want.to_bits()
        );
    }
}

/// Bit-exact, not `==`: `-0.0 == 0.0` would hide a signed-zero disagreement,
/// which is observable downstream as `1.0/x` = ∓inf. Any NaN is as good as
/// any other, since `Min`/`Max` hand back a NaN operand, not a mask.
fn same_bits_or_both_nan(got: f32, want: f32) -> bool {
    got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan())
}

#[test]
fn min_max_nan_handling_agrees_between_tiers() {
    use pixelflow_ir::OpKind;
    assert_tiers_agree_binary(
        "min",
        &Kernel::x().min(&Kernel::y()),
        OpKind::Min,
        same_bits_or_both_nan,
    );
    assert_tiers_agree_binary(
        "max",
        &Kernel::x().max(&Kernel::y()),
        OpKind::Max,
        same_bits_or_both_nan,
    );
}

/// Gt/Ge are the unordered predicates (true for NaN); Lt/Le are ordered. The
/// asymmetry is the hardware's, and the oracle records it.
///
/// Compared bit-for-bit against the raw comparison — no `select` wrapper and no
/// NaN escape hatch — because the *bit pattern* is the contract, not merely
/// which branch it would pick. An all-ones lane is a NaN, so a NaN-tolerant
/// comparison would accept `f32::NAN` where the hardware writes `0xFFFFFFFF`
/// and both would still "agree": a mask lane is all-ones or all-zero in both
/// tiers or it is not a mask.
#[test]
fn nan_comparisons_agree_between_tiers() {
    use pixelflow_ir::OpKind;
    for (name, op, k) in [
        ("gt", OpKind::Gt, Kernel::x().gt(&Kernel::y())),
        ("ge", OpKind::Ge, Kernel::x().ge(&Kernel::y())),
        ("lt", OpKind::Lt, Kernel::x().lt(&Kernel::y())),
        ("le", OpKind::Le, Kernel::x().le(&Kernel::y())),
    ] {
        assert_tiers_agree_binary(name, &k, op, |got, want| got.to_bits() == want.to_bits());
    }
}

/// A folded comparison must be substitutable for an executed one.
///
/// `If` and `BitAnd` are bitwise on every backend, so a mask lane that is
/// merely *truthy* corrupts them: with `1.0` (`0x3f800000`) standing in for
/// true, `runtime_mask & 1.0` is `0x3f800000`, and blending `7.0` against `9.0`
/// through that pattern yields `4.5` — a value neither branch held. This is the
/// end-to-end shape: one operand of the `&` is a constant the folder produced,
/// the other is computed at runtime.
#[test]
fn a_folded_mask_blends_like_a_computed_one() {
    use pixelflow_ir::OpKind;

    // The folder's answer for `0.5 != nextafter(0.5)` — true, and (since the
    // epsilon fix) reachable for ordinary finite values.
    let near = f32::from_bits(0.5f32.to_bits() + 1);
    let folded = OpKind::Ne
        .eval_binary(Lit::Real(0.5), Lit::Real(near))
        .expect("oracle covers Ne");
    assert_eq!(
        folded,
        Lit::Mask(true),
        "the folder's answer is a mask, not a float"
    );
    // The literal, as the folder would leave it in an arena: a mask constant,
    // with no float spelling anywhere.
    let mut arena = ExprArena::new();
    let root = arena.push_lit(folded);
    let folded = Kernel::from_parts(arena, root);

    let k = Kernel::x()
        .gt(&Kernel::constant(0.0))
        .and(&folded)
        .select(&Kernel::constant(7.0), &Kernel::constant(9.0));
    let jit = jit_cache::compile(&k, pixelflow_ir::LatticeShape::POINT)
        .expect("mask kernel compiles")
        .kernel;

    assert_eq!(
        eval_point(&jit, 1.0, 0.0),
        7.0,
        "mask true must select if_true exactly"
    );
    assert_eq!(
        eval_point(&jit, -1.0, 0.0),
        9.0,
        "mask false must select if_false exactly"
    );
}

#[test]
fn round_agrees_between_tiers_away_from_ties() {
    use pixelflow_ir::OpKind;
    let rounded = Kernel::x().round();
    let jit = jit_cache::compile(&rounded, pixelflow_ir::LatticeShape::POINT)
        .expect("round compiles")
        .kernel;
    // Non-ties only. At a tie the two targets disagree by design (x86
    // nearest-even, aarch64 FRINTA ties-away), so there is no answer to
    // assert — see `tie_result_is_platform_specific`.
    assert!(OpKind::Round.fold_is_platform_specific(&[2.5]));
    for x in [2.4f32, 2.6, -2.4, -2.6, 0.2, 7.0, -7.0] {
        let got = eval_point(&jit, x, 0.0);
        let want = OpKind::Round
            .eval_unary(Lit::Real(x))
            .and_then(Lit::real)
            .expect("oracle covers Round");
        assert_eq!(got, want, "round({x}): JIT {got} vs oracle {want}");
    }
}

/// Which ops carry the platform-specific marker. The marker's *effect* on
/// constant folding is covered where the folder lives
/// (`pixelflow-search::math::algebra`, `platform_specific_fold_tests`) — this
/// only pins the classification, which is what the emitters and the contract
/// table in CLAUDE.md are written against.
#[test]
fn platform_specific_ops_are_classified() {
    use pixelflow_ir::OpKind;
    for op in [OpKind::Min, OpKind::Max, OpKind::Gt, OpKind::Ge] {
        assert!(
            op.fold_is_platform_specific(&[f32::NAN, 1.0]),
            "{op:?} diverges between x86 and aarch64 on NaN"
        );
    }
    // The ones both hardwares agree on stay foldable.
    for op in [OpKind::Lt, OpKind::Le, OpKind::Eq, OpKind::Ne, OpKind::Add] {
        assert!(
            !op.fold_is_platform_specific(&[f32::NAN, 1.0]),
            "{op:?} agrees across targets and should remain foldable"
        );
    }

    // Estimate instructions: no argument is safe, because an estimate is only
    // guaranteed close and the three targets are not close to each other.
    for op in [OpKind::Recip, OpKind::Rsqrt] {
        for x in [2.0f32, 3.0, 0.5, 1.0] {
            assert!(
                op.fold_is_platform_specific(&[x]),
                "{op:?}({x}) is an estimate and must never fold"
            );
        }
    }

    // MulAdd always folds. One rounding versus two is precision, not a
    // different value, even on the inputs where the two forms disagree.
    let fused_differs = [1.000_000_1f32, 4097.0, 4097.0];
    assert!(
        !OpKind::MulAdd.fold_is_platform_specific(&fused_differs),
        "a rounding difference is inside the contract and must not block a fold"
    );
    let [a, b, c] = fused_differs.map(Lit::Real);
    assert_eq!(
        OpKind::MulAdd.eval_ternary(a, b, c),
        Some(Lit::Real(
            fused_differs[0].mul_add(fused_differs[1], fused_differs[2])
        )),
        "the folder's answer is the single-rounding one, whatever profile built it"
    );
}

/// `Eq`/`Ne` are exact in every emitter, so the oracle must be too. The old
/// `(x-y).abs() < f32::EPSILON` form folded 0.5 and its next representable
/// neighbour to "equal" while the hardware said "not equal" — a fold-vs-runtime
/// miscompile on ordinary finite values, with no NaN or platform difference
/// involved.
#[test]
fn eq_ne_are_exact_not_epsilon() {
    use pixelflow_ir::OpKind;

    let near = f32::from_bits(0.5f32.to_bits() + 1);
    assert_ne!(0.5f32, near, "the two values really are distinct f32s");
    assert!(
        (0.5f32 - near).abs() < f32::EPSILON,
        "and closer than EPSILON"
    );

    let (t, f) = (Lit::Mask(true), Lit::Mask(false));
    let (half, near) = (Lit::Real(0.5), Lit::Real(near));
    assert_eq!(OpKind::Eq.eval_binary(half, near), Some(f));
    assert_eq!(OpKind::Ne.eval_binary(half, near), Some(t));
    assert_eq!(OpKind::Eq.eval_binary(half, half), Some(t));

    // NaN: never equal, always unequal — agreed by every emitter.
    let nan = Lit::Real(f32::NAN);
    assert_eq!(OpKind::Eq.eval_binary(nan, nan), Some(f));
    assert_eq!(OpKind::Ne.eval_binary(nan, nan), Some(t));

    // `Kernel` exposes no eq/ne constructor, so there is no JIT side to compare
    // against here — the folder IS the consumer that was wrong, and these
    // assertions are what pins it.
}

/// An INVALID float->int conversion has three answers, so the folder must
/// decline it: x86 `cvttps2dq` produces the "integer indefinite"
/// `0x8000_0000` for every invalid input, aarch64 `FCVTZS` saturates (NaN to
/// 0), and Rust's `as i32` — which `eval_unary` uses — saturates like aarch64.
/// So `+inf` would fold to `i32::MAX`'s bit pattern but execute as `i32::MIN`
/// on x86: optimizing would change the program's output.
#[test]
fn invalid_trunc_to_int_is_platform_specific() {
    for x in [
        f32::INFINITY,
        f32::NAN,
        3e9,
        // 2^31 is the first f32 ABOVE i32::MAX. `i32::MAX as f32` rounds up to
        // exactly this value, so a naive `x > i32::MAX as f32` bound would
        // wrongly admit it.
        2_147_483_648.0,
    ] {
        assert!(
            pixelflow_ir::OpKind::TruncToInt.fold_is_platform_specific(&[x]),
            "invalid conversion of {x} must not fold"
        );
    }

    // Value-aware: a conversion that IS in range agrees on every target and
    // still folds. i32::MIN is exactly representable and is a valid input.
    // The negative overflow direction agrees by arithmetic accident: x86's
    // integer-indefinite pattern IS i32::MIN, which is also what aarch64 and
    // Rust saturate to. So these must still fold -- refusing them would cost
    // ~19% of the f32 space for nothing.
    for x in [
        0.0,
        -0.0,
        1.9,
        -1.9,
        2_147_483_520.0,
        -2_147_483_648.0,
        -3e9,
        -2_147_483_904.0,
        f32::NEG_INFINITY,
    ] {
        assert!(
            !pixelflow_ir::OpKind::TruncToInt.fold_is_platform_specific(&[x]),
            "in-range conversion of {x} should still fold"
        );
    }
}

/// A shift count outside `0..32` has three answers, so the folder must decline
/// it: the folder reduces mod 32, x86 (V)PSLLD/(V)PSRLD zero the entire
/// destination for any count > 31, and aarch64's immediate carries into
/// `immh` and silently re-encodes the instruction as a `.2D` (64-bit element)
/// shift that crosses the 32-bit lane boundary.
#[test]
fn out_of_range_shift_counts_are_platform_specific() {
    for op in [pixelflow_ir::OpKind::Shl, pixelflow_ir::OpKind::Shr] {
        for count in [
            32.0,
            33.0,
            40.0,
            64.0,
            -1.0,
            -0.0001,
            2.5,
            f32::NAN,
            f32::INFINITY,
        ] {
            assert!(
                op.fold_is_platform_specific(&[1.0, count]),
                "{op:?} by {count} must not fold"
            );
        }
        // In-range integral counts agree on every target and still fold,
        // `-0.0` among them: it is the integer 0.
        for count in [0.0, -0.0, 1.0, 8.0, 23.0, 31.0] {
            assert!(
                !op.fold_is_platform_specific(&[1.0, count]),
                "{op:?} by {count} should still fold"
            );
        }
    }
}

/// `X` shifted by a `Const` of `count`, as the arena spells it.
fn shift_of_x(
    op: pixelflow_ir::OpKind,
    count: f32,
) -> (pixelflow_ir::ExprArena, pixelflow_ir::ExprId) {
    let mut a = pixelflow_ir::ExprArena::new();
    let x = a.push_var(0);
    let c = a.push_const(count);
    let root = a.push_binary(op, x, c);
    (a, root)
}

fn compile_shift(op: pixelflow_ir::OpKind, count: f32) -> bool {
    let (a, root) = shift_of_x(op, count);
    pixelflow_codegen::emit::compile(&a, root, pixelflow_ir::LatticeShape::POINT).is_ok()
}

/// A shift count is refused where the `Const` narrows to the encoder's `u8`,
/// not after — `256.0 as u32 as u8` is `0`, so a downstream check would see a
/// legal identity shift where the kernel asked for something no target can do.
#[test]
#[should_panic(expected = "is not a ShiftCount (an integer literal in 0..32")]
fn a_shift_count_that_aliases_to_zero_is_still_refused() {
    let _ = compile_shift(pixelflow_ir::OpKind::Shl, 256.0);
}

/// `-0.0` is the integer 0, which the folder folds and every target shifts
/// by, so the emitter takes it as the count 0. Proved on the code's output,
/// not on the compile succeeding: through the emitter's own entry and the
/// optimizing one, the lane comes back unchanged, which is what the folder's
/// oracle gives. A shift by 1 is the control that the harness reads a shift
/// at all.
#[test]
fn a_negative_zero_shift_count_is_the_count_zero() {
    use pixelflow_ir::{LatticeShape, OpKind};

    let lanes = [1.0f32, -2.5, f32::from_bits(0xDEAD_BEEF), f32::MAX];
    let assert_unmoved =
        |route: &str, op: OpKind, jit: &pixelflow_codegen::emit::CompiledKernel| {
            for x in lanes {
                let got = eval_point(jit, x, 0.0);
                let oracle = op
                    .eval_binary(Lit::Real(x), Lit::Real(-0.0))
                    .and_then(Lit::real)
                    .expect("the oracle covers the shifts");
                assert_eq!(
                    got.to_bits(),
                    x.to_bits(),
                    "{route}: {op:?} by -0.0 moved {x}"
                );
                assert_eq!(
                    got.to_bits(),
                    oracle.to_bits(),
                    "{route}: {op:?} by -0.0 on {x}: JIT and oracle disagree"
                );
            }
        };

    for op in [OpKind::Shl, OpKind::Shr] {
        let (a, root) = shift_of_x(op, -0.0);
        let emitted = pixelflow_codegen::emit::compile(&a, root, LatticeShape::POINT)
            .expect("a count of -0.0 compiles")
            .code;
        assert_unmoved("emit::compile", op, &emitted);

        let optimized = jit_cache::compile(&Kernel::from_parts(a, root), LatticeShape::POINT)
            .expect("a count of -0.0 compiles")
            .kernel;
        assert_unmoved("jit_cache::compile", op, &optimized);
    }

    let (a, root) = shift_of_x(OpKind::Shl, 1.0);
    let by_one = pixelflow_codegen::emit::compile(&a, root, LatticeShape::POINT)
        .expect("a count of 1 compiles")
        .code;
    assert_eq!(
        eval_point(&by_one, 1.0, 0.0).to_bits(),
        1.0f32.to_bits() << 1,
        "the control shift moved the lane"
    );
}

/// The same guard covers the counts that do not alias.
#[test]
#[should_panic(expected = "is not a ShiftCount (an integer literal in 0..32")]
fn a_lane_crossing_shift_count_is_refused() {
    let _ = compile_shift(pixelflow_ir::OpKind::Shl, 32.0);
}

/// The same shift through the optimizing entry, which every kernel is
/// compiled by: a count the e-graph declines is not repaired on the way, so
/// the emitter's own refusal is the one that fires.
fn jit_compile_shift(count: impl FnOnce(&mut pixelflow_ir::ExprArena) -> pixelflow_ir::ExprId) {
    let mut a = pixelflow_ir::ExprArena::new();
    let x = a.push_var(0);
    let c = count(&mut a);
    let root = a.push_binary(pixelflow_ir::OpKind::Shl, x, c);
    let kernel = Kernel::from_parts(a, root);
    jit_cache::compile(&kernel, pixelflow_ir::LatticeShape::POINT)
        .expect("the emitter refuses by panic, not by error");
}

#[test]
#[should_panic(expected = "is not a ShiftCount (an integer literal in 0..32")]
fn the_optimizing_entry_still_refuses_a_lane_crossing_count() {
    jit_compile_shift(|a| a.push_const(32.0));
}

#[test]
#[should_panic(expected = "is not a ShiftCount (an integer literal in 0..32")]
fn the_optimizing_entry_still_refuses_a_fractional_count() {
    jit_compile_shift(|a| a.push_const(0.5));
}

/// A count of 0 is the identity and every target agrees on it, so it must
/// COMPILE — `fold_is_platform_specific` classifies it as portable, and the
/// encoder has to honour that. aarch64 `USHR` cannot encode `#0`, so `Shr`
/// lowers to a move rather than being refused.
#[test]
fn a_zero_shift_count_compiles_as_the_identity() {
    for op in [pixelflow_ir::OpKind::Shl, pixelflow_ir::OpKind::Shr] {
        assert!(
            compile_shift(op, 0.0),
            "{op:?} by 0 is the identity and must compile"
        );
    }
}

// ── exp / exp2 saturation (CLAUDE.md, "Floating point at the edges") ─────────
//
// The contract: exp and exp2 saturate past ±126 exponents rather than
// overflowing to inf. The expansion (`expand_exp2`, pixelflow-ir passes) clamps
// the base-2 argument to [-EXP2_CLAMP, EXP2_CLAMP] = [-126, 126] before building
// 2^n from the exponent field, so the saturated values are exact:
//   high side  2^126  = bits 0x7E80_0000 (exponent field 253)
//   low side   2^-126 = bits 0x0080_0000 (exponent field 1, f32::MIN_POSITIVE)
// The low side saturates to the smallest normal, not to 0.
// `exp(x)` is `exp2(x·log2 e)`, so its saturation point is the same clamp on
// the base-2 exponent: x ≈ ±87.34 in natural units.

const TWO_POW_126: f32 = f32::from_bits(253 << 23);
const TWO_POW_NEG_126: f32 = f32::from_bits(1 << 23);

/// Relative tolerance for in-range exp/exp2 against `f32::exp`/`f32::exp2`.
///
/// This is the implementation's precision, not an IEEE claim. `EXP2_POLY` holds
/// the Taylor coefficients of 2^f (1/k!·ln2^k), not the minimax its rustdoc
/// names, so the degree-5 truncation error is the Lagrange remainder
/// t^6/6! ≈ 1.54e-4 relative at t = ln2 (the top of the fractional range). A
/// full-domain sweep measured the worst case at 8.9e-5 (exp, where `x·log2 e`
/// adds one f32 rounding at |exponent| ≤ 126). 2e-4 bounds that with margin,
/// and is far below any coefficient or clamp error, which move results by ≥1e-2.
const EXP_REL_TOL: f32 = 2e-4;

/// Compile `k` once at a point lattice and evaluate it at each `x`.
fn eval_at(name: &str, k: &Kernel, xs: &[f32]) -> Vec<f32> {
    let jit = jit_cache::compile(k, pixelflow_ir::LatticeShape::POINT)
        .unwrap_or_else(|e| panic!("{name}: kernel failed to compile on this backend: {e}"))
        .kernel;
    xs.iter().map(|&x| eval_point(&jit, x, 0.0)).collect()
}

#[test]
fn exp2_saturates_to_two_to_the_126_past_the_clamp() {
    // 126.0 is the clamp itself: in range and saturated agree on 2^126.
    let xs = [126.0, 126.5, 127.0, 150.0, 1000.0, f32::MAX, f32::INFINITY];
    for (&x, got) in xs.iter().zip(eval_at("exp2", &Kernel::x().exp2(), &xs)) {
        assert_eq!(
            got.to_bits(),
            TWO_POW_126.to_bits(),
            "exp2({x}) must saturate to 2^126, got {got}"
        );
    }
}

#[test]
fn exp2_saturates_to_two_to_the_minus_126_past_the_clamp() {
    // The low side saturates to the smallest normal, not to 0.
    let xs = [
        -126.0,
        -126.5,
        -127.0,
        -150.0,
        -1000.0,
        f32::MIN,
        f32::NEG_INFINITY,
    ];
    for (&x, got) in xs.iter().zip(eval_at("exp2", &Kernel::x().exp2(), &xs)) {
        assert_eq!(
            got.to_bits(),
            TWO_POW_NEG_126.to_bits(),
            "exp2({x}) must saturate to 2^-126, got {got}"
        );
    }
}

#[test]
fn exp_saturates_past_the_clamp_on_its_base_two_exponent() {
    // exp(k·ln 2) = 2^k, so these are the exp2 boundaries in natural units.
    let ks = [127.0f64, 150.0, 1000.0];
    let high: Vec<f32> = ks
        .iter()
        .map(|k| (k * core::f64::consts::LN_2) as f32)
        .chain([1000.0, f32::INFINITY])
        .collect();
    for (&x, got) in high.iter().zip(eval_at("exp", &Kernel::x().exp(), &high)) {
        assert_eq!(
            got.to_bits(),
            TWO_POW_126.to_bits(),
            "exp({x}) must saturate to 2^126, got {got}"
        );
    }

    let low: Vec<f32> = ks
        .iter()
        .map(|k| (-k * core::f64::consts::LN_2) as f32)
        .chain([-1000.0, f32::NEG_INFINITY])
        .collect();
    for (&x, got) in low.iter().zip(eval_at("exp", &Kernel::x().exp(), &low)) {
        assert_eq!(
            got.to_bits(),
            TWO_POW_NEG_126.to_bits(),
            "exp({x}) must saturate to 2^-126, got {got}"
        );
    }
}

/// Saturation is range, not precision: the outputs are finite and positive
/// everywhere the input is not NaN, and a NaN input stays NaN (the expansion's
/// documented domain guard).
#[test]
fn exp_and_exp2_stay_in_range_and_pass_nan_through() {
    let xs = [
        f32::NAN,
        f32::MIN,
        -1e30,
        -1000.0,
        -88.0,
        0.0,
        88.0,
        1000.0,
        f32::MAX,
        f32::INFINITY,
    ];
    for (name, k) in [("exp", Kernel::x().exp()), ("exp2", Kernel::x().exp2())] {
        for (&x, got) in xs.iter().zip(eval_at(name, &k, &xs)) {
            if x.is_nan() {
                assert!(got.is_nan(), "{name}(NaN) must be NaN, got {got}");
            } else {
                assert!(
                    got.is_finite() && got > 0.0,
                    "{name}({x}) left the range (0, inf): got {got}"
                );
            }
        }
    }
}

/// Inside the clamp, the saturating kernels still agree with the Rust oracle,
/// right up to the boundary on both sides.
#[test]
fn exp2_agrees_with_std_up_to_the_clamp() {
    let xs = [
        -126.0f32, -125.9, -125.0, -64.5, -1.0, -0.0, 0.5, 64.5, 124.9, 125.0, 125.9, 126.0,
    ];
    for (&x, got) in xs.iter().zip(eval_at("exp2", &Kernel::x().exp2(), &xs)) {
        let want = x.exp2();
        let rel = ((got - want) / want).abs();
        assert!(
            rel <= EXP_REL_TOL,
            "exp2({x}): got {got}, want {want} (rel {rel:.2e})"
        );
    }
}

/// Same boundary for `exp`, with the argument chosen so `x·log2 e` lands at and
/// just inside ±125 and ±126 (x = 86.0 gives ~124.07; x = 87.3 gives ~125.95).
#[test]
fn exp_agrees_with_std_up_to_the_clamp() {
    let xs = [
        -87.3f32, -86.0, -40.0, -1.0, 0.0, 0.25, 1.0, 40.0, 86.0, 87.3,
    ];
    for (&x, got) in xs.iter().zip(eval_at("exp", &Kernel::x().exp(), &xs)) {
        let want = x.exp();
        let rel = ((got - want) / want).abs();
        assert!(
            rel <= EXP_REL_TOL,
            "exp({x}): got {got}, want {want} (rel {rel:.2e})"
        );
    }
}
