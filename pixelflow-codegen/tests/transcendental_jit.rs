//! exp/log kernels through the JIT, checked numerically at this build's width.
//!
//! These lower (`expand_transcendentals`) to the integer-domain primitives
//! TruncToInt/IntToFloat/IAdd/Shl/Shr — ops a backend can silently lack while
//! every other test stays green, because `Field`'s combinator tier computes
//! its own log2/exp via native intrinsics and never consults the JIT. That is
//! exactly how Avx512Backend shipped refusing `ShiftImm`: `test_log2` passed
//! (combinator tier), the full workspace passed under `+avx512f`, and a
//! `Kernel::log2` still failed to compile. The completeness sweep catches the
//! emission gap; this catches the semantics — the bytes must also be right.

use pixelflow_codegen::jit_cache;
use pixelflow_ir::Kernel;

const LANES: usize = pixelflow_codegen::JIT_VECTOR_BYTES / 4;

fn eval_points_1d(jit: &pixelflow_codegen::CompiledKernel, inputs: &[f32]) -> Vec<f32> {
    let mut outputs = Vec::with_capacity(inputs.len());
    for chunk in inputs.chunks(LANES) {
        let mut xs = [0.0f32; LANES];
        for (i, &x) in chunk.iter().enumerate() {
            xs[i] = x;
        }
        let res = unsafe {
            jit.call(pixelflow_codegen::Point4::new(
                xs,
                [0.0; LANES],
                [0.0; LANES],
                [0.0; LANES],
            ))
        };
        outputs.extend_from_slice(&res[..chunk.len()]);
    }
    outputs
}

fn eval_points_2d(jit: &pixelflow_codegen::CompiledKernel, inputs: &[(f32, f32)]) -> Vec<f32> {
    let mut outputs = Vec::with_capacity(inputs.len());
    for chunk in inputs.chunks(LANES) {
        let mut xs = [0.0f32; LANES];
        let mut ys = [0.0f32; LANES];
        for (i, &(x, y)) in chunk.iter().enumerate() {
            xs[i] = x;
            ys[i] = y;
        }
        let res = unsafe {
            jit.call(pixelflow_codegen::Point4::new(
                xs,
                ys,
                [0.0; LANES],
                [0.0; LANES],
            ))
        };
        outputs.extend_from_slice(&res[..chunk.len()]);
    }
    outputs
}

fn check(name: &str, k: &Kernel, inputs: &[f32], reference: impl Fn(f32) -> f32) {
    let jit = jit_cache::compile(k, pixelflow_ir::LatticeShape::POINT)
        .unwrap_or_else(|e| panic!("{name}: kernel failed to compile on this backend: {e}"))
        .kernel;
    let results = eval_points_1d(&jit, inputs);
    for (&x, got) in inputs.iter().zip(results) {
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

/// `|sin| ≤ 1` and `|cos| ≤ 1` through *emitted machine code*, not the oracle.
///
/// Trig uses range reduction with modular arithmetic (`MulAdd` + `TruncToInt` +
/// `IntToFloat`), which overflows past ~1e7 without float64 intermediates. The
/// pipeline's domain gate clamps inputs to `TRIG_DOMAIN = 1e6`; anything past
/// that gets NaN from the JIT, so we test:
///
/// 1. bounded in `[-1, 1]` inside the domain, and accurate against libc
/// 2. NaN (or inf-guarded) outside the domain — never wrong finite answers.
#[test]
fn sin_cos_stay_bounded_and_nan_outside_domain() {
    // The contract: within TRIG_DOMAIN, sin/cos are valid and bounded.
    const TRIG_DOMAIN: f32 = 1e6;

    for (name, k) in [("sin", Kernel::x().sin()), ("cos", Kernel::x().cos())] {
        let jit = jit_cache::compile(&k, pixelflow_ir::LatticeShape::POINT)
            .unwrap_or_else(|e| panic!("{name}: failed to compile on this backend: {e}"))
            .kernel;

        // One argument per binade across the whole finite f32 range, plus the
        // magnitudes from the pipeline smoke run that first showed the defect.
        let mut args: Vec<f32> = (-20i32..=127)
            .flat_map(|e| {
                let b = (2.0f64).powi(e) as f32;
                [b, -b, b * 1.3, b * 1.7]
            })
            .collect();
        args.extend_from_slice(&[0.0, 1.72e7, 8.64e8, 1.4e7, 2.61e13, 1e30, f32::MAX]);

        let results = eval_points_1d(&jit, &args);
        for (&x, got) in args.iter().zip(results) {
            assert!(
                got.is_nan() || got.abs() <= 1.0,
                "{name}({x:e}) = {got:e} — outside [-1, 1]",
            );
            // Inside the domain the answer must also exist and be right; a
            // kernel that returned NaN everywhere would pass the bound above.
            if x.abs() < TRIG_DOMAIN {
                let want = if name == "sin" {
                    (x as f64).sin()
                } else {
                    (x as f64).cos()
                } as f32;
                assert!(
                    (got - want).abs() <= 4e-6,
                    "{name}({x:e}) = {got}, want {want}",
                );
            } else {
                assert!(
                    got.is_nan(),
                    "{name}({x:e}) = {got} — should be NaN outside TRIG_DOMAIN",
                );
            }
        }
    }
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

/// Every lane of a binary kernel, JIT vs oracle, on edge-case inputs.
fn assert_tiers_agree_binary(name: &str, k: &Kernel, op: pixelflow_ir::OpKind) {
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

    let results = eval_points_2d(&jit, &inputs);
    for (&(x, y), got) in inputs.iter().zip(results) {
        let want = op.eval_binary(x, y).expect("oracle covers this op");
        // Bit-exact, not `==`: `-0.0 == 0.0` would hide a signed-zero
        // disagreement, which is observable downstream as `1.0/x` = ∓inf.
        assert!(
            got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()),
            "{name}({x}, {y}): JIT gave {got}, oracle gave {want} — tiers must agree"
        );
    }
}

#[test]
fn min_max_nan_handling_agrees_between_tiers() {
    use pixelflow_ir::OpKind;
    assert_tiers_agree_binary("min", &Kernel::x().min(&Kernel::y()), OpKind::Min);
    assert_tiers_agree_binary("max", &Kernel::x().max(&Kernel::y()), OpKind::Max);
}

/// Gt/Ge are the unordered predicates (true for NaN); Lt/Le are ordered. The
/// asymmetry is the hardware's, and the oracle records it.
///
/// Compared bit-for-bit against the raw comparison — no `select` wrapper and no
/// NaN escape hatch — because the *bit pattern* is the contract, not merely
/// which branch it would pick. An all-ones lane is a NaN, so a NaN-tolerant
/// comparison would accept `f32::NAN` where the hardware writes `0xFFFFFFFF`
/// and both would still "agree".
#[test]
fn nan_comparisons_agree_between_tiers() {
    use pixelflow_ir::OpKind;
    let nan = f32::NAN;
    for (name, op, k) in [
        ("gt", OpKind::Gt, Kernel::x().gt(&Kernel::y())),
        ("ge", OpKind::Ge, Kernel::x().ge(&Kernel::y())),
        ("lt", OpKind::Lt, Kernel::x().lt(&Kernel::y())),
        ("le", OpKind::Le, Kernel::x().le(&Kernel::y())),
    ] {
        let jit = jit_cache::compile(&k, pixelflow_ir::LatticeShape::POINT)
            .unwrap_or_else(|e| panic!("{name}: {e}"))
            .kernel;
        let inputs: Vec<(f32, f32)> = [
            (1.0f32, nan),
            (nan, 1.0f32),
            (nan, nan),
            (1.0, 2.0),
            (2.0, 1.0),
            (2.5, 2.5),
            (-0.0, 0.0),
        ]
        .into_iter()
        .filter(|&(x, y)| !op.fold_is_platform_specific(&[x, y]))
        .collect();

        let results = eval_points_2d(&jit, &inputs);
        for (&(x, y), got) in inputs.iter().zip(results) {
            let want = op.eval_binary(x, y).expect("oracle covers this op");
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "{name}({x}, {y}): JIT wrote {:#010x}, oracle {:#010x} — a mask \
                 lane is all-ones or all-zero in both tiers or it is not a mask",
                got.to_bits(),
                want.to_bits()
            );
        }
    }
}

/// A folded comparison must be substitutable for an executed one.
///
/// `Select` and `BitAnd` are bitwise on every backend, so a mask lane that is
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
    let folded = OpKind::Ne.eval_binary(0.5, near).expect("oracle covers Ne");

    let k = Kernel::x()
        .gt(&Kernel::constant(0.0))
        .and(&Kernel::constant(folded))
        .select(&Kernel::constant(7.0), &Kernel::constant(9.0));
    let jit = jit_cache::compile(&k, pixelflow_ir::LatticeShape::POINT)
        .expect("mask kernel compiles")
        .kernel;

    let res = eval_points_1d(&jit, &[1.0, -1.0]);
    assert_eq!(res[0], 7.0, "mask true must select if_true exactly");
    assert_eq!(res[1], 9.0, "mask false must select if_false exactly");
}

#[test]
fn round_agrees_between_tiers_away_from_ties() {
    use pixelflow_ir::OpKind;
    let rounded = Kernel::x().round();
    let jit = jit_cache::compile(&rounded, pixelflow_ir::LatticeShape::POINT)
        .expect("round compiles")
        .kernel;
    // Non-ties only. At a tie the three tiers disagree by design (x86
    // nearest-even, aarch64 FRINTA ties-away, combinator `(x+0.5).floor()`), so
    // there is no answer to assert — see `tie_result_is_platform_specific`.
    assert!(OpKind::Round.fold_is_platform_specific(&[2.5]));
    let inputs = [2.4f32, 2.6, -2.4, -2.6, 0.2, 7.0, -7.0];
    let results = eval_points_1d(&jit, &inputs);
    for (&x, got) in inputs.iter().zip(results) {
        let want = OpKind::Round.eval_unary(x).expect("oracle covers Round");
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
    assert_eq!(
        OpKind::MulAdd.eval_ternary(fused_differs[0], fused_differs[1], fused_differs[2]),
        Some(fused_differs[0].mul_add(fused_differs[1], fused_differs[2])),
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

    let t = OpKind::mask(true);
    let f = OpKind::mask(false);
    assert_eq!(
        OpKind::Eq.eval_binary(0.5, near).map(f32::to_bits),
        Some(f.to_bits())
    );
    assert_eq!(
        OpKind::Ne.eval_binary(0.5, near).map(f32::to_bits),
        Some(t.to_bits())
    );
    assert_eq!(
        OpKind::Eq.eval_binary(0.5, 0.5).map(f32::to_bits),
        Some(t.to_bits())
    );

    // NaN: never equal, always unequal — agreed by every emitter.
    let nan = f32::NAN;
    assert_eq!(
        OpKind::Eq.eval_binary(nan, nan).map(f32::to_bits),
        Some(f.to_bits())
    );
    assert_eq!(
        OpKind::Ne.eval_binary(nan, nan).map(f32::to_bits),
        Some(t.to_bits())
    );

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
        // In-range integral counts agree on every target and still fold.
        for count in [0.0, 1.0, 8.0, 23.0, 31.0] {
            assert!(
                !op.fold_is_platform_specific(&[1.0, count]),
                "{op:?} by {count} should still fold"
            );
        }
    }
}

fn compile_shift(op: pixelflow_ir::OpKind, count: f32) -> bool {
    let mut a = pixelflow_ir::ExprBuilder::new();
    let x = a.push_var(0);
    let c = a.push_const(count);
    let root = a.push_binary(op, x, c);
    let (rooted, env) = a.finish(&[root]);
    pixelflow_codegen::emit::compile(pixelflow_ir::Term::new(rooted.entry(), &env)).is_ok()
}

/// A shift count is refused where the `Const` narrows to the encoder's `u8`,
/// not after — `256.0 as u32 as u8` is `0`, so a downstream check would see a
/// legal identity shift where the kernel asked for something no target can do.
#[test]
#[should_panic(expected = "is not an integer in 0..32")]
fn a_shift_count_that_aliases_to_zero_is_still_refused() {
    let _ = compile_shift(pixelflow_ir::OpKind::Shl, 256.0);
}

/// The same guard covers the counts that do not alias.
#[test]
#[should_panic(expected = "is not an integer in 0..32")]
fn a_lane_crossing_shift_count_is_refused() {
    let _ = compile_shift(pixelflow_ir::OpKind::Shl, 32.0);
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
