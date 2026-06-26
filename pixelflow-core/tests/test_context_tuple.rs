//! Prototype test for WithContext flat tuple approach
//!
//! Note: This is a COMPILATION TEST to prove flat context tuples
//! avoid trait solver explosion that occurs with nested Let.

use pixelflow_core::{WithContext, X, Y};

#[test]
fn test_with_context_5_params_compiles() {
    // THE KEY TEST: This compiles with 5 params where nested Let fails!
    //
    // WithContext creates: WithContext<(V0, V1, V2, V3, V4), Body>
    // Instead of: Let<V0, Let<V1, Let<V2, Let<V3, Let<V4, Body>>>>>
    //
    // Trait bounds are FLAT, not recursive - trait solver is happy!

    let v0 = X + 1.0f32;
    let v1 = Y + 2.0f32;
    let v2 = X * Y;
    let v3 = X - Y;
    let v4 = X / 2.0f32;

    // For now, body is just a constant (CtxVar arithmetic needs more impls)
    let body = X; // Placeholder body

    let _kernel = WithContext::new((v0, v1, v2, v3, v4), body);

    // Success! If this compiles, we've proven the flat approach works.
    // TODO: Implement CtxVar arithmetic to actually use the bound values
}

#[test]
fn test_with_context_constructs() {
    // Just verify it constructs correctly
    let v0 = X;
    let v1 = Y;
    let body = X;

    let _kernel = WithContext::new((v0, v1), body);
    // Type is: WithContext<(X, Y), X> - flat structure!
}
