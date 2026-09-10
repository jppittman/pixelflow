//! A thousand structurally identical kernels differing only in their uniform
//! instances is **one** compile.
//!
//! This is the whole reason a uniform is keyed by dense offset rather than
//! by identity: the code is a function of the composition's shape, and the
//! instances are a property of the block. Alone in its own binary because
//! the assertion is on the cache's process-global entry count, which every
//! other test that compiles a kernel perturbs.

#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use std::sync::Arc;

use pixelflow_codegen::jit_cache::{compile, entry_count};
use pixelflow_ir::decl::{UniformDecl, UniformIdentity};
use pixelflow_ir::expr::ExprBuilder;
use pixelflow_ir::kind::OpKind;
use pixelflow_ir::{Kernel, LatticeShape};

/// `(x − cx)·r + cy` over three fresh instances, declared in one of two
/// orders so the link — not the declaration order — is what is shared.
fn circle(declared_in_order: bool) -> Kernel {
    let decl = |default| UniformDecl {
        id: UniformIdentity::mint(),
        default,
    };
    let (cx, cy, r) = (decl(0.0), decl(0.0), decl(1.0));
    let mut a = ExprBuilder::new();
    let (scx, scy, sr) = if declared_in_order {
        (
            a.declare_uniform(cx),
            a.declare_uniform(cy),
            a.declare_uniform(r),
        )
    } else {
        let sr = a.declare_uniform(r);
        let scy = a.declare_uniform(cy);
        (a.declare_uniform(cx), scy, sr)
    };
    let x = a.push_var(0);
    let ucx = a.push_uniform(scx);
    let ur = a.push_uniform(sr);
    let ucy = a.push_uniform(scy);
    let d = a.push_binary(OpKind::Sub, x, ucx);
    let scaled = a.push_binary(OpKind::Mul, d, ur);
    let root = a.push_binary(OpKind::Add, scaled, ucy);
    let (rooted, env) = a.finish(&[root]);
    Kernel::from_rooted(rooted, env)
}

#[test]
fn a_thousand_circles_is_one_compile() {
    const SHAPE: LatticeShape = LatticeShape::new([64, 64]);
    let before = entry_count();
    let k = circle(true);
    let first = compile(&k, SHAPE).expect("compile").kernel;
    let after_first = entry_count();
    assert_eq!(after_first - before, 1, "the first circle compiles once");
    for i in 1..1000 {
        let k = circle(i % 2 == 0);
        let linked = compile(&k, SHAPE).expect("compile");
        assert!(
            Arc::ptr_eq(&first, &linked.kernel),
            "circle {i} did not share the first one's code"
        );
    }
    assert_eq!(
        entry_count(),
        after_first,
        "999 more circles, differing only in their uniform instances, must not add a cache entry"
    );
}
