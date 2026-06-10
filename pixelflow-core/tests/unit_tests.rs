//! Comprehensive unit tests for pixelflow-core.
//!
//! These tests target 80%+ coverage of the library.

extern crate std;

use std::prelude::v1::*;

use pixelflow_core::jet::Jet2;
use pixelflow_core::{
    // Operations
    // Core types
    Field,
    Manifold,
    ManifoldExt,
    Max,
    Min,
    PARALLELISM,
    W,
    // Variables
    X,
    Y,
    Z,
    // Combinators
    combinators::{Fix, Map, Select},
    // Materialize
    materialize,
    scale,
    variables::Axis,
};

// ============================================================================
// Test Helpers
// ============================================================================

/// Asserts that a Field is approximately equal to a float value across all lanes.
fn assert_field_approx_eq(field: Field, expected: f32) {
    // Construct AST to check equality
    // Field ops return AST nodes, so we need to eval them to get the result mask.
    let expected_field = Field::from(expected);
    let diff = (field - expected_field).abs();
    // Loosened epsilon to 1e-2 to account for rcp/rsqrt approximations (approx 12-bit precision)
    // especially when scaled by larger inputs (e.g. scale combinator).
    let is_small = diff.lt(Field::from(1e-2));

    // Evaluate at dummy coordinates (Field ignores them)
    let zero = Field::from(0.0);
    let coords = (zero, zero, zero, zero);
    let mask = is_small.eval(coords);

    // Check if difference is small enough
    if !mask.all() {
        // If not, materialize to show details
        #[derive(Clone, Copy)]
        struct Wrapper(Field);
        impl pixelflow_core::ops::Vector for Wrapper {
            type Component = Field;
            fn get(&self, _axis: Axis) -> Field {
                self.0
            }
        }
        impl Manifold for Wrapper {
            type Output = Wrapper;
            fn eval(&self, _: (Field, Field, Field, Field)) -> Wrapper {
                *self
            }
        }

        let m = Wrapper(field);
        let mut out = vec![0.0f32; PARALLELISM * 4];
        materialize(&m, 0.0, 0.0, &mut out);

        let mut actual_values = Vec::new();
        for i in 0..PARALLELISM {
            // Materialize outputs interleaved RGBA. Wrapper puts Field in all channels.
            // So index i*4 is the value for lane i.
            actual_values.push(out[i * 4]);
        }

        panic!(
            "Field assertion failed.\nExpected: approx {}\nActual (lanes): {:?}",
            expected, actual_values
        );
    }
}

// ============================================================================
// Field Tests
// ============================================================================

mod field_tests {
    use super::*;

    #[test]
    fn field_from_f32_should_broadcast_value() {
        let f: Field = 42.0f32.into();
        assert_field_approx_eq(f, 42.0);
    }

    #[test]
    fn field_from_i32_should_broadcast_value() {
        let f: Field = 42i32.into();
        assert_field_approx_eq(f, 42.0);
    }

    #[test]
    fn field_arithmetic_should_compute_correctly() {
        let a: Field = 2.0f32.into();
        let b: Field = 3.0f32.into();
        let zero = Field::from(0.0);
        let coords = (zero, zero, zero, zero);

        assert_field_approx_eq((a + b).eval(coords), 5.0);
        assert_field_approx_eq((a - b).eval(coords), -1.0);
        assert_field_approx_eq((a * b).eval(coords), 6.0);
        assert_field_approx_eq((a / b).eval(coords), 2.0 / 3.0);
    }

    #[test]
    fn field_bitwise_should_compute_correctly() {
        let a: Field = 1.0f32.into();
        let b: Field = 2.0f32.into();

        // 1.0 & 2.0 -> 0.0 (bitwise representation)
        assert_field_approx_eq(a & b, 0.0);
    }

    #[test]
    fn field_min_max_should_select_extremes() {
        let a: Field = 5.0f32.into();
        let b: Field = 3.0f32.into();
        // min/max return Min/Max combinators - eval them to get Field
        let zero = Field::from(0.0);
        let coords = (zero, zero, zero, zero);
        assert_field_approx_eq(a.min(b).eval(coords), 3.0);
        assert_field_approx_eq(a.max(b).eval(coords), 5.0);
    }

    // Robustness tests
    #[test]
    fn field_sqrt_should_handle_zero() {
        // sqrt(0) -> NaN if not handled carefully when using rsqrt * self.
        // Field::sqrt returns Field directly (internal method exposed via pub fn sqrt)
        // Wait, unit_tests.rs checks `X.sqrt()`. That uses ManifoldExt which creates AST.
        // But `Field::sqrt()` (method) is also available.
        // Let's test the AST node evaluation which is what users use.
        let expr = Field::from(0.0).sqrt(); // This calls ManifoldExt::sqrt -> Sqrt<Field>
        let zero = Field::from(0.0);
        let coords = (zero, zero, zero, zero);

        assert_field_approx_eq(expr.eval(coords), 0.0);
    }

    #[test]
    fn field_sqrt_should_handle_negative() {
        let expr = Field::from(-1.0).sqrt();
        let zero = Field::from(0.0);
        let coords = (zero, zero, zero, zero);

        assert_field_approx_eq(expr.eval(coords), 0.0);
    }
}

// ============================================================================
// Variable Tests
// ============================================================================

mod variable_tests {
    use super::*;

    #[test]
    fn coordinate_variables_should_evaluate_to_inputs() {
        let coords = (
            Field::from(5.0),
            Field::from(3.0),
            Field::from(1.0),
            Field::from(7.0),
        );

        let x_res = X.eval(coords);
        let y_res = Y.eval(coords);
        let z_res = Z.eval(coords);
        let w_res = W.eval(coords);

        assert_field_approx_eq(x_res, 5.0);
        assert_field_approx_eq(y_res, 3.0);
        assert_field_approx_eq(z_res, 1.0);
        assert_field_approx_eq(w_res, 7.0);
    }

    #[test]
    fn axis_enums_should_be_distinct() {
        assert_eq!(Axis::X, Axis::X);
        assert_ne!(Axis::X, Axis::Y);
    }
}

// ============================================================================
// Manifold Implementation Tests
// ============================================================================

mod manifold_tests {
    use super::*;

    #[test]
    fn constant_types_should_eval_to_constant() {
        let zero = Field::from(0.0);
        let coords = (zero, zero, zero, zero);

        let c_f32 = 42.0f32.eval(coords);
        assert_field_approx_eq(c_f32, 42.0);

        let c_i32 = 42i32.eval(coords);
        assert_field_approx_eq(c_i32, 42.0);
    }

    #[test]
    fn scale_combinator_should_scale_coordinates() {
        // scale(X, 2.0) evals X at x/2.0
        let scaled = scale(X, 2.0);
        let x = Field::from(10.0);
        let zero = Field::from(0.0);
        let coords = (x, zero, zero, zero);

        assert_field_approx_eq(scaled.eval(coords), 5.0);
    }
}

// ============================================================================
// Binary Operations Tests
// ============================================================================

mod binary_ops_tests {
    use super::*;

    #[test]
    fn manifold_operators_should_compute_correctly() {
        let x = Field::from(10.0);
        let y = Field::from(2.0);
        let zero = Field::from(0.0);
        let coords = (x, y, zero, zero);

        assert_field_approx_eq((X + Y).eval(coords), 12.0);
        assert_field_approx_eq((X - Y).eval(coords), 8.0);
        assert_field_approx_eq((X * Y).eval(coords), 20.0);
        assert_field_approx_eq((X / Y).eval(coords), 5.0);
    }
}

// ============================================================================
// Unary Operations Tests
// ============================================================================

mod unary_ops_tests {
    use super::*;

    #[test]
    fn unary_operators_should_compute_correctly() {
        let x = Field::from(4.0);
        let neg_x = Field::from(-4.0);
        let y = Field::from(5.0);
        let zero = Field::from(0.0);

        let c_x = (x, zero, zero, zero);
        let c_neg = (neg_x, zero, zero, zero);
        let c_xy = (x, y, zero, zero);

        assert_field_approx_eq(X.sqrt().eval(c_x), 2.0);
        assert_field_approx_eq(X.abs().eval(c_neg), 4.0);
        assert_field_approx_eq(Max(X, Y).eval(c_xy), 5.0);
        assert_field_approx_eq(Min(X, Y).eval(c_xy), 4.0);
    }
}

// ============================================================================
// Select Combinator Tests
// ============================================================================

mod select_tests {
    use super::*;

    #[test]
    fn select_should_choose_based_on_condition() {
        let coords_pos = (
            Field::from(5.0),
            Field::from(10.0),
            Field::from(20.0),
            Field::from(0.0),
        );
        let coords_neg = (
            Field::from(-1.0),
            Field::from(10.0),
            Field::from(20.0),
            Field::from(0.0),
        );

        let sel_pos = Select {
            cond: X.gt(0.0f32),
            if_true: Y,
            if_false: Z,
        };

        // 5.0 > 0 -> True -> Y (10.0)
        assert_field_approx_eq(sel_pos.eval(coords_pos), 10.0);

        // -1.0 > 0 -> False -> Z (20.0)
        assert_field_approx_eq(sel_pos.eval(coords_neg), 20.0);
    }

    // Robustness: Short-circuiting
    #[derive(Clone, Copy)]
    struct Panics;
    type Field4 = (Field, Field, Field, Field);
    impl Manifold<Field4> for Panics {
        type Output = Field;
        fn eval(&self, _p: Field4) -> Field {
            panic!("Manifold evaluated when it should have been short-circuited!");
        }
    }

    #[derive(Clone, Copy)]
    struct Safe(f32);
    impl Manifold<Field4> for Safe {
        type Output = Field;
        fn eval(&self, _p: Field4) -> Field {
            Field::from(self.0)
        }
    }

    #[test]
    fn select_should_short_circuit_true_branch() {
        let select = Select {
            cond: X.gt(X), // Always false
            if_true: Panics,
            if_false: Safe(42.0),
        };
        let zero = Field::from(0.0);
        let coords = (zero, zero, zero, zero);
        assert_field_approx_eq(select.eval(coords), 42.0);
    }

    #[test]
    fn select_should_short_circuit_false_branch() {
        let select = Select {
            cond: X.ge(X), // Always true
            if_true: Safe(42.0),
            if_false: Panics,
        };
        let zero = Field::from(0.0);
        let coords = (zero, zero, zero, zero);
        assert_field_approx_eq(select.eval(coords), 42.0);
    }

    #[test]
    fn select_should_blend_mixed_mask() {
        if PARALLELISM < 2 {
            return;
        }

        // X = sequential(0.0) -> [0, 1, 2, 3...]
        // cond: X < 1.0. True for lane 0, False for lane 1+.
        let s = Select {
            cond: X.lt(1.0f32),
            if_true: Safe(10.0),
            if_false: Safe(20.0),
        };

        let x_seq = Field::sequential(0.0);
        let zero = Field::from(0.0);
        let coords = (x_seq, zero, zero, zero);
        let result = s.eval(coords);

        // Helper to inspect lanes
        #[derive(Clone, Copy)]
        struct Res(Field);
        impl pixelflow_core::ops::Vector for Res {
            type Component = Field;
            fn get(&self, _axis: Axis) -> Field {
                self.0
            }
        }
        impl Manifold for Res {
            type Output = Res;
            fn eval(&self, _: (Field, Field, Field, Field)) -> Res {
                *self
            }
        }

        let m = Res(result);
        let mut out = vec![0.0f32; PARALLELISM * 4];
        materialize(&m, 0.0, 0.0, &mut out);

        let lane0 = out[0];
        let lane1 = out[4];

        assert!((lane0 - 10.0).abs() < 1e-5, "Lane 0 should be 10.0");
        assert!((lane1 - 20.0).abs() < 1e-5, "Lane 1 should be 20.0");
    }

    #[test]
    fn select_comparisons_should_respect_equality_boundary() {
        let zero = Field::from(0.0);
        let coords = (zero, zero, zero, zero);

        // Gt: 0 > 0 -> False
        let sel_gt = Select {
            cond: X.gt(0.0f32),
            if_true: Field::from(1.0),
            if_false: Field::from(2.0),
        };
        assert_field_approx_eq(sel_gt.eval(coords), 2.0);

        // Ge: 0 >= 0 -> True
        let sel_ge = Select {
            cond: X.ge(0.0f32),
            if_true: Field::from(1.0),
            if_false: Field::from(2.0),
        };
        assert_field_approx_eq(sel_ge.eval(coords), 1.0);

        // Lt: 0 < 0 -> False
        let sel_lt = Select {
            cond: X.lt(0.0f32),
            if_true: Field::from(1.0),
            if_false: Field::from(2.0),
        };
        assert_field_approx_eq(sel_lt.eval(coords), 2.0);

        // Le: 0 <= 0 -> True
        let sel_le = Select {
            cond: X.le(0.0f32),
            if_true: Field::from(1.0),
            if_false: Field::from(2.0),
        };
        assert_field_approx_eq(sel_le.eval(coords), 1.0);
    }
}

// ============================================================================
// Map Combinator Tests
// ============================================================================

mod map_tests {
    use super::*;

    #[test]
    fn map_should_transform_coordinates() {
        // Substitute X with X+X
        let doubled = Map::new(X, X + X);
        let x = Field::from(5.0);
        let zero = Field::from(0.0);
        let coords = (x, zero, zero, zero);

        // Input x=5, Map transforms coords to x=10, then evals X (which returns current x)
        assert_field_approx_eq(doubled.eval(coords), 10.0);
    }

    #[test]
    fn map_clamp_should_restrict_range() {
        let clamped = X.map(X.max(0.0f32).min(1.0f32));
        let zero = Field::from(0.0);

        let coords_half = (Field::from(0.5), zero, zero, zero);
        let coords_low = (Field::from(-0.5), zero, zero, zero);
        let coords_high = (Field::from(1.5), zero, zero, zero);

        assert_field_approx_eq(clamped.eval(coords_half), 0.5);
        assert_field_approx_eq(clamped.eval(coords_low), 0.0);
        assert_field_approx_eq(clamped.eval(coords_high), 1.0);
    }
}

// ============================================================================
// Fix Combinator Tests
// ============================================================================

mod fix_tests {
    use super::*;

    #[test]
    fn fix_combinator_should_iterate() {
        // Iterate: start at 0, add 1 each step, stop at 5
        let fix = Fix {
            seed: 0.0f32,
            step: W + 1.0f32,
            done: W.ge(5.0f32),
        };
        let zero = Field::from(0.0);
        let coords = (zero, zero, zero, zero);
        assert_field_approx_eq(fix.eval(coords), 5.0);
    }
}

// ============================================================================
// Jet2 Tests
// ============================================================================

mod jet2_tests {
    use super::*;

    #[test]
    fn jet2_derivatives_should_be_correct() {
        // Test x^2 + y
        let expr = X * X + Y;

        let x_jet = Jet2::x(Field::from(5.0));
        let y_jet = Jet2::y(Field::from(3.0));
        let zero = Jet2::constant(Field::from(0.0));
        let coords = (x_jet, y_jet, zero, zero);

        let result = expr.eval(coords);

        assert_field_approx_eq(result.val, 28.0);
        assert_field_approx_eq(result.dx, 10.0); // 2x
        assert_field_approx_eq(result.dy, 1.0); // 1
    }
}

// ============================================================================
// Complex Expression Tests
// ============================================================================

mod complex_expr_tests {
    use super::*;

    #[test]
    fn circle_sdf_should_compute_distance() {
        let dist = (X * X + Y * Y).sqrt();
        let x = Field::from(3.0);
        let y = Field::from(4.0);
        let zero = Field::from(0.0);
        let coords = (x, y, zero, zero);

        assert_field_approx_eq(dist.eval(coords), 5.0);
    }
}

// ============================================================================
// Default Trait Tests
// ============================================================================

mod default_tests {
    use super::*;

    #[test]
    fn field_default_should_be_zero() {
        let f: Field = Default::default();
        assert_field_approx_eq(f, 0.0);
    }
}

// ============================================================================
// Clone and Copy Tests
// ============================================================================

mod clone_copy_tests {
    use super::*;

    #[test]
    fn field_copy_should_work() {
        let a = Field::from(1.0);
        let b = a; // Copy
        assert_field_approx_eq(b, 1.0);
    }

    #[test]
    fn axis_clone_should_work() {
        let axis = Axis::X;
        let axis2 = axis;
        assert_eq!(axis, axis2);
    }
}

// ============================================================================
// Debug Trait Tests
// ============================================================================

mod debug_tests {
    use super::*;
    use std::fmt::Write;

    #[test]
    fn field_debug_should_produce_output() {
        let f = Field::from(42.0);
        let mut s = String::new();
        write!(s, "{:?}", f).unwrap();
        assert!(!s.is_empty());
    }
}

// ============================================================================
// Hash Tests
// ============================================================================

mod hash_tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn axis_should_be_hashable() {
        let mut set = HashSet::new();
        set.insert(Axis::X);
        assert!(set.contains(&Axis::X));
    }
}

// ============================================================================
// BNot Tests
// ============================================================================

mod bnot_tests {
    use super::*;
    use pixelflow_core::ops::logic::BNot;

    #[test]
    fn bnot_should_invert_logic() {
        let not_x = BNot(X.gt(0.0f32));

        let x = Field::from(5.0);
        let zero = Field::from(0.0);
        let coords = (x, zero, zero, zero);

        // 5.0 > 0.0 is True. Not True is False (0.0).
        let result = not_x.eval(coords);
        // Result is Field mask (0.0 or -1.0/NaN/etc depending on implementation?)
        assert_field_approx_eq(result, 0.0);

        // Try false case
        let x_neg = Field::from(-5.0);
        let coords_neg = (x_neg, zero, zero, zero);

        // -5 > 0 is False. Not False is True.
        // Use select to verify mask logic
        let sel = Select {
            cond: not_x,
            if_true: Field::from(1.0),
            if_false: Field::from(0.0),
        };
        assert_field_approx_eq(sel.eval(coords_neg), 1.0);
    }
}
