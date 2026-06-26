use pixelflow_core::Jet2; // Use Alias
use pixelflow_core::{Field, ManifoldExt};

#[test]
fn test_sqrt_derivative() {
    // Test at x=4.0
    let x_val = Field::from(4.0);
    let x_jet = Jet2::x(x_val);

    // Use inherent method jet_sqrt
    let result = x_jet.jet_sqrt();

    // Value check: sqrt(4) = 2
    // Use ManifoldExt::constant() to evaluate AST nodes from Field ops
    let val_diff = (result.val() - Field::from(2.0)).constant().abs();
    // Relaxed tolerance 1e-3
    assert!(
        val_diff.lt(Field::from(1e-3)).all(),
        "Value mismatch: expected 2.0, got {:?}",
        result.val()
    );

    // Derivative check: 1/(2*sqrt(4)) = 0.25
    let dx_diff = (result.dx() - Field::from(0.25)).constant().abs();
    assert!(
        dx_diff.lt(Field::from(1e-3)).all(),
        "Derivative mismatch: expected 0.25, got {:?}",
        result.dx()
    );
}

#[test]
fn test_sqrt_derivative_zero() {
    // Check behavior at 0.0
    let x_val = Field::from(0.0);
    let x_jet = Jet2::x(x_val);

    let result = x_jet.jet_sqrt();

    // Value check: sqrt(0) = 0
    let val_diff = (result.val() - Field::from(0.0)).constant().abs();
    assert!(
        val_diff.lt(Field::from(1e-3)).all(),
        "Value mismatch at 0: expected 0.0, got {:?}",
        result.val()
    );
}
