//! A thousand structurally identical kernels differing only in their uniform
//! instances is **one** compile.
//!
//! This is the whole reason a uniform is keyed by dense offset rather than
//! by identity: the code is a function of the composition's shape, and the
//! instances are a property of the block. In its own binary because the
//! assertions are on process-global counts — the JIT cache's entries, the
//! optimizer's saturations — which every other test that compiles a kernel
//! perturbs; and the tests here take one lock, since two of them in one
//! process would perturb each other the same way.

#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use std::sync::{Arc, Mutex};

use pixelflow_codegen::jit_cache::{compile, entry_count};
use pixelflow_ir::arena::{ExprArena, UniformDecl, UniformIdentity};
use pixelflow_ir::kind::OpKind;
use pixelflow_ir::{Kernel, LatticeShape};

/// The process-global counts below are read across several compiles, so the
/// tests of this binary run one at a time.
static SERIAL: Mutex<()> = Mutex::new(());

/// `(x − cx)·r + cy` over three fresh instances, declared in one of two
/// orders so the link — not the declaration order — is what is shared.
fn circle(declared_in_order: bool) -> Kernel {
    let decl = |default| UniformDecl {
        id: UniformIdentity::mint(),
        default,
    };
    let (cx, cy, r) = (decl(0.0), decl(0.0), decl(1.0));
    let mut a = ExprArena::new();
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
    Kernel::from_parts(a, root)
}

/// `circle` with one more term, so the two tests in this binary never race
/// on one structure's first saturation.
fn shifted_circle(declared_in_order: bool) -> Kernel {
    let circle = circle(declared_in_order);
    let (arena, root) = circle.parts();
    let mut a = arena.clone();
    let one = a.push_const(1.0);
    let root = a.push_binary(OpKind::Add, root, one);
    Kernel::from_parts(a, root)
}

/// One structure saturates once, whatever shape it is compiled at and
/// whichever instances it names. The e-graph is shape-free and name-free;
/// only the extraction, priced by the lattice, runs per shape — so a
/// resized frame or a second band height costs an emit, not a saturation.
#[test]
fn one_saturation_per_structure_across_shapes_and_compositions() {
    use pixelflow_search::runtime::saturation_count;

    let _serial = SERIAL.lock().expect("serial");
    let before = saturation_count();
    let _ = compile(&shifted_circle(true), LatticeShape::new([32, 32])).expect("compile");
    assert_eq!(saturation_count() - before, 1, "the first shape saturates");
    for shape in [[33, 32], [32, 33], [48, 8]] {
        let _ = compile(&shifted_circle(false), LatticeShape::new(shape)).expect("compile");
    }
    assert_eq!(
        saturation_count() - before,
        1,
        "another shape or composition of a saturated structure is an extraction, not a saturation"
    );
}

#[test]
fn a_thousand_circles_is_one_compile() {
    const SHAPE: LatticeShape = LatticeShape::new([64, 64]);
    let _serial = SERIAL.lock().expect("serial");
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
