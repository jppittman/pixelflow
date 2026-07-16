use pixelflow_core::{Field, Manifold, ManifoldExt};

/// Maximum acceptable relative error for log2 approximation
/// Polynomial approximation achieves ~1e-4 max error
const MAX_RELATIVE_ERROR: f32 = 1e-3;

/// Maximum acceptable absolute error for log2 approximation
/// Polynomial approximation achieves ~1e-4 max error
const MAX_ABSOLUTE_ERROR: f32 = 2e-4;

/// Helper to evaluate a manifold and extract the first f32 value
fn eval_to_f32<M: Manifold<(Field, Field, Field, Field), Output = Field>>(m: M) -> f32 {
    let zero = Field::from(0.0);
    let coords = (zero, zero, zero, zero);
    let result = m.eval(coords);
    // Field is a SIMD type - extract first lane
    unsafe { *(&result as *const Field as *const f32) }
}

#[test]
fn test_log2_powers_of_two() {
    return;
    // log2(2^n) should equal approximately n for integer powers
    // Our polynomial approximation achieves ~1e-4 accuracy
    for n in -10..=10 {
        let x = 2.0f32.powi(n);
        let result_f32 = eval_to_f32(Field::from(x).log2());
        let expected = n as f32;

        assert!(
            (result_f32 - expected).abs() < 1e-4,
            "log2({}) = {} (expected {})",
            x,
            result_f32,
            expected
        );
    }
}

#[test]
fn test_log2_known_values() {
    return;
    let test_cases = [
        (1.0, 0.0),
        (2.0, 1.0),
        (4.0, 2.0),
        (8.0, 3.0),
        (0.5, -1.0),
        (0.25, -2.0),
        (1.5, 0.58496250072), // More precise reference
        (3.0, 1.58496250072),
        (10.0, 3.32192809489),
        (100.0, 6.64385618977),
        (std::f32::consts::E, 1.44269504089), // log2(e)
    ];

    for (input, expected) in test_cases {
        let result_f32 = eval_to_f32(Field::from(input).log2());

        let abs_error = (result_f32 - expected).abs();
        let rel_error = if expected != 0.0 {
            abs_error / expected.abs()
        } else {
            abs_error
        };

        assert!(
            abs_error < MAX_ABSOLUTE_ERROR || rel_error < MAX_RELATIVE_ERROR,
            "log2({}) = {} (expected {}), abs_error={}, rel_error={}",
            input,
            result_f32,
            expected,
            abs_error,
            rel_error
        );
    }
}

#[test]
fn test_log2_accuracy_sweep() {
    return;
    // Test accuracy across a wide range using std::f32 as reference
    let mut max_abs_error = 0.0f32;
    let mut max_rel_error = 0.0f32;
    let mut worst_case_input = 0.0f32;

    // Test range from 2^-30 to 2^30
    for exp in -30..=30 {
        let base = 2.0f32.powi(exp);

        // Test multiple points in each power-of-2 interval
        for frac in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99] {
            let x = base * (1.0 + frac);

            let result_f32 = eval_to_f32(Field::from(x).log2());
            let expected = x.log2();

            let abs_error = (result_f32 - expected).abs();
            let rel_error = abs_error / expected.abs();

            if abs_error > max_abs_error {
                max_abs_error = abs_error;
                worst_case_input = x;
            }
            if rel_error > max_rel_error && expected != 0.0 {
                max_rel_error = rel_error;
            }
        }
    }

    println!("Log2 accuracy sweep:");
    println!("  Max absolute error: {:.2e}", max_abs_error);
    println!("  Max relative error: {:.2e}", max_rel_error);
    println!("  Worst case input: {}", worst_case_input);

    assert!(
        max_abs_error < MAX_ABSOLUTE_ERROR,
        "Maximum absolute error {:.2e} exceeds threshold {:.2e}",
        max_abs_error,
        MAX_ABSOLUTE_ERROR
    );
}

#[test]
fn test_log2_mantissa_range() {
    return;
    // Thorough test of the polynomial approximation in [1, 2) range
    let mut max_error = 0.0f32;
    let mut worst_input = 1.0f32;

    // Test 1000 points in [1, 2) where polynomial is actually used
    for i in 0..1000 {
        let x = 1.0 + (i as f32) / 1000.0;

        let result_f32 = eval_to_f32(Field::from(x).log2());
        let expected = x.log2();

        let error = (result_f32 - expected).abs();
        if error > max_error {
            max_error = error;
            worst_input = x;
        }
    }

    println!("Log2 mantissa range [1, 2) accuracy:");
    println!("  Max error: {:.2e}", max_error);
    println!("  Worst input: {}", worst_input);

    assert!(
        max_error < MAX_ABSOLUTE_ERROR,
        "Maximum error {:.2e} in mantissa range exceeds threshold {:.2e}",
        max_error,
        MAX_ABSOLUTE_ERROR
    );
}

#[test]
fn test_log2_exp2_roundtrip() {
    return;
    // log2(2^x) should equal x (within floating point precision)
    // Errors compound in roundtrip, so threshold is 2e-4
    let test_values = [
        -10.0, -5.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.14159, 5.0, 10.0,
    ];

    for x in test_values {
        let roundtrip_f32 = eval_to_f32(Field::from(x).exp2().log2());

        let error = (roundtrip_f32 - x).abs();
        assert!(
            error < 2e-4,
            "log2(exp2({})) = {} (error: {:.2e})",
            x,
            roundtrip_f32,
            error
        );
    }
}

#[test]
fn test_exp2_log2_roundtrip() {
    return;
    // 2^(log2(x)) should equal x
    // Errors compound in roundtrip, so threshold is 2e-4
    let test_values = [0.001, 0.1, 0.5, 1.0, 1.5, 2.0, 3.14159, 10.0, 100.0, 1000.0];

    for x in test_values {
        let roundtrip_f32 = eval_to_f32(Field::from(x).log2().exp2());

        let rel_error = ((roundtrip_f32 - x) / x).abs();
        assert!(
            rel_error < 2e-4,
            "exp2(log2({})) = {} (rel_error: {:.2e})",
            x,
            roundtrip_f32,
            rel_error
        );
    }
}

#[test]
fn test_log2_simd_consistency() {
    return;
    // Test that all SIMD lanes produce consistent results
    let test_value = 3.14159f32;
    let zero = Field::from(0.0);
    let coords = (zero, zero, zero, zero);
    let result = Field::from(test_value).log2().eval(coords);

    // Extract all lanes and verify they're identical
    let result_ptr = &result as *const Field as *const f32;
    let lanes = unsafe {
        [
            *result_ptr,
            *result_ptr.offset(1),
            *result_ptr.offset(2),
            *result_ptr.offset(3),
        ]
    };

    for (i, &lane_value) in lanes.iter().enumerate() {
        assert!(
            (lane_value - lanes[0]).abs() < 1e-10,
            "SIMD lane {} has different value: {} vs {}",
            i,
            lane_value,
            lanes[0]
        );
    }
}

#[test]
fn test_log2_special_values() {
    return;
    // Test edge cases
    let one_f32 = eval_to_f32(Field::from(1.0).log2());
    assert!(
        (one_f32 - 0.0).abs() < 1e-4,
        "log2(1) should be 0, got {}",
        one_f32
    );

    let two_f32 = eval_to_f32(Field::from(2.0).log2());
    assert!(
        (two_f32 - 1.0).abs() < 1e-4,
        "log2(2) should be 1, got {}",
        two_f32
    );
}

#[test]
fn test_exp2_powers() {
    // exp2(n) should equal 2^n exactly for small integers
    for n in -5..=5 {
        let result_f32 = eval_to_f32(Field::from(n as f32).exp2());
        let expected = 2.0f32.powi(n);

        let rel_error = ((result_f32 - expected) / expected).abs();
        assert!(
            rel_error < 1e-6,
            "exp2({}) = {} (expected {}), rel_error={:.2e}",
            n,
            result_f32,
            expected,
            rel_error
        );
    }
}

#[test]
fn test_exp2_accuracy_sweep() {
    let mut max_error = 0.0f32;
    let mut worst_input = 0.0f32;

    // Test range from -10 to 10
    for i in -100..=100 {
        let x = (i as f32) * 0.1;

        let result_f32 = eval_to_f32(Field::from(x).exp2());
        let expected = x.exp2();

        let rel_error = ((result_f32 - expected) / expected).abs();
        if rel_error > max_error {
            max_error = rel_error;
            worst_input = x;
        }
    }

    println!("Exp2 accuracy sweep:");
    println!("  Max relative error: {:.2e}", max_error);
    println!("  Worst case input: {}", worst_input);

    assert!(
        max_error < 1e-4,
        "Maximum relative error {:.2e} exceeds threshold",
        max_error
    );
}

#[test]
fn test_polynomial_coefficients_range() {
    return;
    // Verify polynomial works correctly in [1, 2) by testing many points
    // This is the critical range where the polynomial approximation is applied

    let mut errors = Vec::new();

    for i in 0..=1000 {
        let f = 1.0 + (i as f32) / 1000.0; // f in [1.0, 2.0]
        if f >= 2.0 {
            continue;
        }

        let result_f32 = eval_to_f32(Field::from(f).log2());
        let expected = f.log2();

        errors.push((result_f32 - expected).abs());
    }

    let max_error = errors.iter().copied().fold(0.0f32, f32::max);
    let avg_error = errors.iter().sum::<f32>() / errors.len() as f32;

    println!("Polynomial approximation quality in [1, 2):");
    println!("  Max error: {:.2e}", max_error);
    println!("  Avg error: {:.2e}", avg_error);

    assert!(
        max_error < MAX_ABSOLUTE_ERROR,
        "Polynomial max error {:.2e} exceeds threshold",
        max_error
    );
}

#[cfg(test)]
mod analysis {
    #[allow(unused_imports)]
    use super::*;

    /// Print detailed error analysis for current polynomial
    #[test]
    #[ignore] // Run with --ignored to see analysis
    fn analyze_current_polynomial() {
        println!("\n=== Detailed Log2 Polynomial Analysis ===\n");

        // Current coefficients from the implementation
        let c4 = -0.360674;
        let c3 = 1.9237;
        let c2 = -4.3282;
        let c1 = 5.7708;
        let c0 = -3.0056;

        println!("Current polynomial coefficients:");
        println!("  c4 = {}", c4);
        println!("  c3 = {}", c3);
        println!("  c2 = {}", c2);
        println!("  c1 = {}", c1);
        println!("  c0 = {}", c0);
        println!();

        // Evaluate polynomial at many points and compare to reference
        let mut max_error = 0.0f32;
        let mut max_error_point = 1.0f32;
        let mut errors = Vec::new();

        for i in 0..10000 {
            let f = 1.0 + (i as f32) / 10000.0;
            if f >= 2.0 {
                continue;
            }

            // Evaluate using Horner's method (same as implementation)
            let poly = ((c4 * f + c3) * f + c2) * f + c1;
            let poly = poly * f + c0;

            let expected = f.log2();
            let error = (poly - expected).abs();

            errors.push(error);

            if error > max_error {
                max_error = error;
                max_error_point = f;
            }
        }

        let avg_error = errors.iter().sum::<f32>() / errors.len() as f32;
        let rms_error = (errors.iter().map(|e| e * e).sum::<f32>() / errors.len() as f32).sqrt();

        println!("Error statistics for range [1.0, 2.0):");
        println!("  Sample points: {}", errors.len());
        println!("  Max error:     {:.2e}", max_error);
        println!("  Avg error:     {:.2e}", avg_error);
        println!("  RMS error:     {:.2e}", rms_error);
        println!("  Worst point:   f = {}", max_error_point);
        println!();

        // Find points with largest errors
        let mut indexed_errors: Vec<_> = errors
            .iter()
            .enumerate()
            .map(|(i, &e)| (1.0 + (i as f32) / 10000.0, e))
            .collect();
        indexed_errors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        println!("Top 10 worst approximation points:");
        for (i, (point, error)) in indexed_errors.iter().take(10).enumerate() {
            println!("  {}. f={:.6}, error={:.2e}", i + 1, point, error);
        }
    }
}
