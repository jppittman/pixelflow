use pixelflow_core::jet::Jet2;
use pixelflow_core::{ManifoldCompat, ManifoldExt, X, Y};

#[test]
#[ignore = "Needs internal Field access for lane extraction"]
fn test_jet2_automatic_gradient() {
    // Expression: x² + y
    let expr = X * X + Y;

    // Evaluate at (5, 3) with jets
    let x_jet = Jet2::x(5.0.into());
    let y_jet = Jet2::y(3.0.into());
    let zero = Jet2::constant(0.0.into());

    let _result = expr.eval_raw(x_jet, y_jet, zero, zero);

    // Fields are accessed directly: result.val, result.dx, result.dy
    // But we can't extract scalar values without internal store()
    // This test needs to be restructured to work with the new API
}

#[test]
#[ignore = "Needs internal Field access for lane extraction"]
fn test_jet2_product_rule() {
    let expr = X * Y;

    let x_jet = Jet2::x(3.0.into());
    let y_jet = Jet2::y(4.0.into());
    let zero = Jet2::constant(0.0.into());

    let _result = expr.eval_raw(x_jet, y_jet, zero, zero);
    // result.val, result.dx, result.dy are Fields
}

#[test]
#[ignore = "Needs internal Field access for lane extraction"]
fn test_jet2_chain_rule_sqrt() {
    let expr = X.sqrt();

    let x_jet = Jet2::x(16.0.into());
    let zero = Jet2::constant(0.0.into());

    let _result = expr.eval_raw(x_jet, zero, zero, zero);
}

#[test]
#[ignore = "Needs internal Field access for lane extraction"]
fn test_jet2_circle_normal() {
    // Compute distance from origin - the SDF of a circle centered at origin
    // We use just the sqrt(x² + y²) part to get the autodiff gradients.
    // The constant radius subtraction happens after evaluation to keep Jet2 compatibility.
    let dist = (X * X + Y * Y).sqrt();

    let x_jet = Jet2::x(50.0.into());
    let y_jet = Jet2::y(50.0.into());
    let zero = Jet2::constant(0.0.into());

    // Evaluate with jets to get automatic gradients (distance and partial derivatives)
    let dist_result = dist.eval_raw(x_jet, y_jet, zero, zero);
    // For a circle SDF, subtract radius after: sdf = dist - radius
    let _circle_sdf = dist_result - Jet2::constant(100.0.into());
}
