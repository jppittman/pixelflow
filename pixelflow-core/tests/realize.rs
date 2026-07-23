//! `Lattice::realize`: the consumer-facing compilation verb.
//!
//! A combinator kernel — the ordinary `kernel!` output, never touched by any
//! IR code — realizes through the `Lower` boundary: the tree lowers by
//! composition, JIT-compiles once (through the global cache), and tabulates
//! identically to the generic `collapse` path. The consumer wrote zero IR.

use pixelflow_compiler::kernel;
use pixelflow_core::{Lattice, Manifold};

/// Realize and collapse must agree texel-for-texel (up to backend numerics:
/// the combinator's approximate-rsqrt sqrt vs the JIT's exact one).
fn assert_paths_agree<M>(m: &M, name: &str)
where
    M: Manifold<
            (
                pixelflow_core::Field,
                pixelflow_core::Field,
                pixelflow_core::Field,
                pixelflow_core::Field,
            ),
            Output = pixelflow_core::Field,
        > + pixelflow_core::Lower,
{
    let lattice = Lattice {
        extent: [16, 16, 1, 1],
        origin: [0.5, 0.5, 0.0, 0.0],
    };
    let generic = lattice.collapse(m);
    let realized = lattice.realize(m);

    for (i, (a, b)) in generic
        .buffer()
        .iter()
        .zip(realized.buffer().iter())
        .enumerate()
    {
        let diff = f32::abs(a - b);
        // Wide enough for the combinator backend's approximate-rsqrt sqrt
        // (~5e-4 relative, amplified near zero crossings); the JIT emits
        // exact hardware sqrt. Tightens when the combinator emitter dies.
        let tol = 2e-3 + 2e-3 * f32::abs(*a);
        assert!(
            diff <= tol,
            "{name} texel {i}: collapse {a} vs realize {b} (diff {diff})"
        );
    }
}

#[test]
fn realize_matches_collapse_for_combinator_kernels() {
    // Params + sqrt + literals: exercises WithContext/CtxVar baking,
    // arithmetic generators, and the unary chain.
    let circle = kernel!(|cx: f32, cy: f32, r: f32| {
        ((X - cx) * (X - cx) + (Y - cy) * (Y - cy)).sqrt() - r
    })(8.0, 8.0, 5.0);
    assert_paths_agree(&circle, "circle_sdf");

    // Piecewise: comparisons, mask AND, select.
    let band = kernel!(|lo: f32, hi: f32| {
        ((Y >= lo) & (Y < hi)).select(X, 0.0)
    })(4.0, 12.0);
    assert_paths_agree(&band, "band_select");
}

#[test]
fn realize_engages_the_jit_for_lowerable_trees() {
    // A structurally unique kernel (constant chosen to not collide with any
    // other test) must add exactly one entry to the global compile cache the
    // first time it realizes, and none the second time.
    let k = kernel!(|| X * 977.125 + Y)();
    let lattice = Lattice {
        extent: [8, 8, 1, 1],
        origin: [0.5, 0.5, 0.0, 0.0],
    };

    // Tests share the process-global cache and run in parallel, so only
    // monotonic facts are assertable: realizing a structurally unique kernel
    // must grow the cache (it compiled), and repeated realizes agree
    // bit-for-bit. Exact hit behavior is pinned by the jit_cache unit tests.
    let before = pixelflow_ir::jit_cache::entry_count();
    let first = lattice.realize(&k);
    let mid = pixelflow_ir::jit_cache::entry_count();
    let second = lattice.realize(&k);

    assert!(mid > before, "first realize must JIT-compile the kernel");
    assert_eq!(first.buffer(), second.buffer());
}

#[test]
fn realize_falls_back_when_a_node_declines() {
    // A tree with an opaque leaf (a runtime Field constant) cannot lower;
    // realize must silently take the generic path and still be correct.
    use pixelflow_core::{Add, Field, X};
    let opaque = Add(X, Field::from(3.0));
    let lattice = Lattice {
        extent: [4, 4, 1, 1],
        origin: [0.5, 0.5, 0.0, 0.0],
    };
    // (That the tree declines to lower at all is pinned by the
    // opaque_values_decline unit test; cache-count assertions are racy
    // across parallel tests sharing the global cache.)
    let out = lattice.realize(&opaque);
    assert_eq!(out.buffer()[0], 0.5 + 3.0);
}
