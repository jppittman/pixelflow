//! The reduction binder `⊕_{i ∈ D}` on the `Kernel` surface.
//!
//! One monoid-parametrized fold over a bounded discrete domain covers the
//! programs the language was missing: contraction (matmul), softmax's
//! stabilizer and normalizer, spherical-harmonic projection, and the ambient
//! -occlusion sample sum. The tests below build those shapes and check them
//! against closed forms, plus the two evaluation tiers (IR interpreter and
//! JIT) against each other.

use pixelflow_ir::binding::BindingTable;
use pixelflow_ir::{Kernel, Monoid, eval_scalar};

/// Evaluate through the IR interpreter — the language's reference semantics.
fn interp(k: &Kernel, x: f32, y: f32) -> f32 {
    eval_scalar(k.term(), &[x, y], &BindingTable::empty())
}

#[test]
fn sum_over_folds_a_bounded_domain() {
    // Σ_{i<5} i = 10, independent of the coordinates.
    let k = Kernel::sum_over(5, |i| i.clone());
    assert_eq!(interp(&k, 0.0, 0.0), 10.0);

    // Σ_{i<4} (i + X) = 6 + 4X — the body closes over outer coordinates.
    let k = Kernel::sum_over(4, |i| i.add(&Kernel::x()));
    assert_eq!(interp(&k, 2.0, 0.0), 6.0 + 8.0);

    // An empty domain folds to the monoid identity.
    assert_eq!(interp(&Kernel::sum_over(0, |i| i.clone()), 0.0, 0.0), 0.0);
}

#[test]
fn over_is_the_primitive_and_the_named_folds_are_helpers() {
    // Every named fold is `over` at a different monoid — swapping the algebra
    // is the only difference between them.
    let body = |i: &Kernel| i.add(&Kernel::constant(1.0));
    for (monoid, helper) in [
        (Monoid::SUM, Kernel::sum_over(4, body)),
        (Monoid::PRODUCT, Kernel::product_over(4, body)),
        (Monoid::MAX, Kernel::max_over(4, body)),
        (Monoid::MIN, Kernel::min_over(4, body)),
    ] {
        let primitive = Kernel::over(monoid, 4, body);
        assert_eq!(
            interp(&primitive, 0.0, 0.0),
            interp(&helper, 0.0, 0.0),
            "{monoid:?} helper must agree with `over`"
        );
    }
}

#[test]
fn mask_monoids_are_bounded_quantifiers() {
    // ∃_{i<4} (i > X) — true iff some index exceeds X.
    let any = Kernel::any_over(4, |i| i.gt(&Kernel::x()));
    // A mask is "set" when nonzero; select turns it back into a number.
    let to_num = |m: &Kernel| m.select(&Kernel::constant(1.0), &Kernel::constant(0.0));

    assert_eq!(interp(&to_num(&any), 2.0, 0.0), 1.0, "3 > 2, so ∃ holds");
    assert_eq!(interp(&to_num(&any), 9.0, 0.0), 0.0, "no index exceeds 9");

    // ∀_{i<4} (i >= X) — true iff every index is at least X.
    let all = Kernel::all_over(4, |i| i.ge(&Kernel::x()));
    assert_eq!(interp(&to_num(&all), 0.0, 0.0), 1.0, "every i >= 0");
    assert_eq!(interp(&to_num(&all), 1.0, 0.0), 0.0, "i = 0 fails i >= 1");

    // Empty domains give each quantifier's identity: ∃ is vacuously false,
    // ∀ vacuously true.
    let none = Kernel::any_over(0, |i| i.ge(&Kernel::constant(0.0)));
    let every = Kernel::all_over(0, |i| i.lt(&Kernel::constant(0.0)));
    assert_eq!(interp(&to_num(&none), 0.0, 0.0), 0.0);
    assert_eq!(interp(&to_num(&every), 0.0, 0.0), 1.0);
}

#[test]
fn monoids_beyond_sum() {
    // Π_{i<4} (i + 1) = 24
    let one = Kernel::constant(1.0);
    assert_eq!(
        interp(&Kernel::product_over(4, |i| i.add(&one)), 0.0, 0.0),
        24.0
    );

    // max_{i<5} (i · X) with X = -1 is 0 (attained at i = 0).
    let k = Kernel::max_over(5, |i| i.mul(&Kernel::x()));
    assert_eq!(interp(&k, -1.0, 0.0), 0.0);
    assert_eq!(interp(&k, 2.0, 0.0), 8.0);

    // min over the same body is symmetric.
    let k = Kernel::min_over(5, |i| i.mul(&Kernel::x()));
    assert_eq!(interp(&k, -1.0, 0.0), -4.0);

    // Empty domains fold to each monoid's identity.
    assert_eq!(
        interp(&Kernel::product_over(0, |i| i.clone()), 0.0, 0.0),
        1.0
    );
    assert_eq!(
        interp(&Kernel::max_over(0, |i| i.clone()), 0.0, 0.0),
        f32::NEG_INFINITY
    );
}

#[test]
fn nested_binders_do_not_capture() {
    // Σ_{i<3} Σ_{j<4} (i·10 + j): the inner index must stay distinct from the
    // outer one. Σ_i Σ_j (10i + j) = 4·10·(0+1+2) + 3·(0+1+2+3) = 120 + 18.
    let ten = Kernel::constant(10.0);
    let k = Kernel::sum_over(3, |i| {
        let scaled = i.mul(&ten);
        Kernel::sum_over(4, move |j| scaled.add(j))
    });
    assert_eq!(interp(&k, 0.0, 0.0), 138.0);

    // Three deep, with the innermost body reading all three indices:
    // Σ_{a<2} Σ_{b<2} Σ_{c<2} (4a + 2b + c) = 4·(0..7 summed) = 28.
    let k = Kernel::sum_over(2, |a| {
        let a4 = a.mul(&Kernel::constant(4.0));
        Kernel::sum_over(2, move |b| {
            let ab = a4.add(&b.mul(&Kernel::constant(2.0)));
            Kernel::sum_over(2, move |c| ab.add(c))
        })
    });
    assert_eq!(interp(&k, 0.0, 0.0), 28.0);
}

/// Unrolling must not duplicate work that does not vary with the index.
///
/// `⊕_i (f(i) · c) = c · ⊕_i f(i)` when `deps(c) ∩ {i} = {}`
/// (docs/designs/REDUCTIONS_AND_FOLDS.md:109). Unrolling gets this for free by
/// *not copying* `c` into each of the N terms — so the rewrite shows up as one
/// `Sin` node in the lowered graph rather than N of them. Before the variance
/// analysis could see binders, every reduction index read as depending on
/// everything, nothing in a body was ever invariant, and `Σ_{i<16} i·sin(Y)+X`
/// emitted 5882 bytes instead of 962.
#[test]
fn unrolling_shares_index_invariant_work() {
    use pixelflow_ir::passes;
    use pixelflow_ir::{ExprData, OpKind};

    // Count only nodes the root reaches: `expand_reduce` rebuilds in place and
    // leaves the pre-rebuild nodes behind, so the whole graph over-counts.
    let count_sin = |k: &Kernel| {
        let lowered = passes::expand_reduce(k.term());
        lowered
            .entry()
            .descendants()
            .filter(|n| **n == ExprData::Op(OpKind::Sin))
            .count()
    };

    // sin(Y) is invariant in the index: one copy serves all 16 terms.
    let invariant = Kernel::sum_over(16, |i| i.mul(&Kernel::y().sin()));
    assert_eq!(
        count_sin(&invariant),
        1,
        "sin(Y) must be shared, not copied"
    );

    // sin(X + i) genuinely varies with the index: N copies is correct, and the
    // sharing must not be so eager that it collapses distinct terms.
    let varying = Kernel::sum_over(16, |i| Kernel::x().add(i).sin());
    assert_eq!(count_sin(&varying), 16, "sin(X+i) differs per term");

    // Semantics are unchanged by the sharing: Σ_{i<4} i·sin(0) = 0, and
    // Σ_{i<4} (i + Y) = 6 + 4Y, checked at Y = 2 → 14.
    assert_eq!(
        interp(&Kernel::sum_over(4, |i| i.add(&Kernel::y())), 0.0, 2.0),
        14.0
    );
}

/// Placeholder claims are process-wide, so two threads building nested binders
/// at the same time must not be handed the same index. They release out of
/// order — thread A finishes its outer fold while B is still inside its own —
/// which is exactly the interleaving a depth counter gets wrong: it would
/// re-issue a placeholder that is still live and silently turn `Σ_i Σ_j f(i,j)`
/// into `Σ_i Σ_j f(j,j)`. This test failed roughly one run in three before the
/// claim became a set.
#[test]
fn concurrently_built_binders_do_not_capture() {
    // Σ_{i<3} Σ_{j<4} (10i + j) = 138, the two-deep case above, built from many
    // threads at once with the nesting held open long enough to interleave.
    let build_and_check = || {
        for _ in 0..2_000 {
            let k = Kernel::sum_over(3, |i| {
                let scaled = i.mul(&Kernel::constant(10.0));
                Kernel::sum_over(4, move |j| scaled.add(j))
            });
            assert_eq!(interp(&k, 0.0, 0.0), 138.0);
        }
    };

    std::thread::scope(|s| {
        for _ in 0..8 {
            let _ = s.spawn(build_and_check);
        }
    });
}

#[test]
fn contraction_is_a_sum_over_a_shared_index() {
    // A matmul row·column: Σ_d q(d)·k(d) with q(d) = d + X, k(d) = 2d.
    // With X = 1: Σ_{d<4} (d+1)(2d) = 2(0 + 2 + 6 + 12) = 40.
    let two = Kernel::constant(2.0);
    let dot = Kernel::sum_over(4, move |d| {
        let q = d.add(&Kernel::x());
        let kv = d.mul(&two);
        q.mul(&kv)
    });
    assert_eq!(interp(&dot, 1.0, 0.0), 40.0);
}

#[test]
fn softmax_shape_normalizes() {
    // The softmax of a bounded score row, evaluated at one position: the
    // max-stabilizer and the sum-normalizer are both this binder, and the
    // whole thing is a pure expression of the coordinates.
    //
    // scores(j) = j · X. p(Y) = exp(s(Y) - m) / Σ_j exp(s(j) - m).
    const N: u32 = 4;
    let scores = |j: &Kernel, x: &Kernel| j.mul(x);

    let x = Kernel::x();
    let m = Kernel::max_over(N, {
        let x = x.clone();
        move |j| scores(j, &x)
    });
    let denom = Kernel::sum_over(N, {
        let (x, m) = (x.clone(), m.clone());
        move |j| scores(j, &x).sub(&m).exp()
    });
    let numer = scores(&Kernel::y(), &x).sub(&m).exp();
    let p = numer.div(&denom);

    // Probabilities over the row must sum to 1 and be ordered by score.
    let total: f32 = (0..N).map(|j| interp(&p, 1.0, j as f32)).sum();
    assert!((total - 1.0).abs() < 1e-5, "softmax row sums to {total}");
    for j in 1..N {
        assert!(
            interp(&p, 1.0, j as f32) > interp(&p, 1.0, (j - 1) as f32),
            "softmax must increase with score at j={j}"
        );
    }
    // Uniform scores (X = 0) give the uniform distribution.
    assert!((interp(&p, 0.0, 2.0) - 0.25).abs() < 1e-5);
}

#[test]
fn occlusion_shaped_accumulation_over_samples() {
    // The AO shape: average a bounded set of samples of a field, each taken at
    // an offset coordinate — `at` warps inside the binder's body.
    // field(x, y) = x, sampled at x + i  =>  (1/N) Σ_{i<N} (X + i) = X + 1.5
    const N: u32 = 4;
    let field = Kernel::x();
    let ao = Kernel::sum_over(N, move |i| field.at(&Kernel::x().add(i), &Kernel::y()))
        .div(&Kernel::constant(N as f32));

    assert_eq!(interp(&ao, 10.0, 0.0), 11.5);
}

/// The JIT unrolls reductions (`expand_reduce`) while the interpreter folds
/// them natively; both must denote the same function.
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
mod jit {
    use super::*;

    // CompiledKernel::call's ABI tracks the build's selected width (SSE2/AVX2;
    // this module is gated off avx512f above), so the splat/extract pair must
    // match: __m256 under +avx2, else __m128.
    #[cfg(target_feature = "avx2")]
    fn jit_eval(k: &Kernel, x: f32, y: f32) -> f32 {
        use core::arch::x86_64::*;
        let jit = pixelflow_codegen::jit_cache::compile(k, pixelflow_ir::LatticeShape::POINT)
            .expect("reduction JIT compile")
            .kernel;
        unsafe {
            _mm256_cvtss_f32(jit.call(pixelflow_codegen::Point4::new(
                _mm256_set1_ps(x),
                _mm256_set1_ps(y),
                _mm256_set1_ps(0.0),
                _mm256_set1_ps(0.0),
            )))
        }
    }

    #[cfg(not(target_feature = "avx2"))]
    fn jit_eval(k: &Kernel, x: f32, y: f32) -> f32 {
        use core::arch::x86_64::*;
        let jit = pixelflow_codegen::jit_cache::compile(k, pixelflow_ir::LatticeShape::POINT)
            .expect("reduction JIT compile")
            .kernel;
        unsafe {
            _mm_cvtss_f32(jit.call(pixelflow_codegen::Point4::new(
                _mm_set1_ps(x),
                _mm_set1_ps(y),
                _mm_set1_ps(0.0),
                _mm_set1_ps(0.0),
            )))
        }
    }

    fn assert_tiers_agree(k: &Kernel, label: &str) {
        for &(x, y) in &[(0.0f32, 0.0f32), (1.0, 2.0), (-1.5, 0.5), (3.0, 1.0)] {
            let (want, got) = (interp(k, x, y), jit_eval(k, x, y));
            assert!(
                (want - got).abs() < 1e-4,
                "{label}: JIT {got} != interpreter {want} at ({x}, {y})"
            );
        }
    }

    #[test]
    fn unrolled_jit_matches_the_interpreter() {
        assert_tiers_agree(&Kernel::sum_over(6, |i| i.mul(&Kernel::x())), "sum");
        assert_tiers_agree(&Kernel::max_over(5, |i| i.sub(&Kernel::y())), "max");
        assert_tiers_agree(
            &Kernel::product_over(3, |i| i.add(&Kernel::constant(1.0))),
            "product",
        );

        // Nested: Σ_{i<3} Σ_{j<3} (i·X + j·Y)
        let nested = Kernel::sum_over(3, |i| {
            let ix = i.mul(&Kernel::x());
            Kernel::sum_over(3, move |j| ix.add(&j.mul(&Kernel::y())))
        });
        assert_tiers_agree(&nested, "nested");
    }

    /// Masks are represented differently by the two tiers (the interpreter
    /// compares to `1.0`/`0.0`, the JIT to all-set/all-clear lanes), so the
    /// quantifier monoids are the place a tier divergence would surface. Both
    /// must agree once the mask is selected back into a number.
    #[test]
    fn quantifier_monoids_agree_across_tiers() {
        let to_num = |m: Kernel| m.select(&Kernel::constant(1.0), &Kernel::constant(0.0));

        assert_tiers_agree(&to_num(Kernel::any_over(4, |i| i.gt(&Kernel::x()))), "any");
        assert_tiers_agree(&to_num(Kernel::all_over(4, |i| i.ge(&Kernel::x()))), "all");
        // Empty domains must produce each quantifier's identity in both tiers.
        assert_tiers_agree(&to_num(Kernel::any_over(0, |i| i.clone())), "any-empty");
        assert_tiers_agree(&to_num(Kernel::all_over(0, |i| i.clone())), "all-empty");
    }
}
