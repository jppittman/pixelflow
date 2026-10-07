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
use pixelflow_ir::arena::{ExprArena, UniformDecl, UniformId, UniformIdentity};
use pixelflow_ir::kind::OpKind;
use pixelflow_ir::{Kernel, LatticeShape};

/// The process-global counts below are read across several compiles, so the
/// tests of this binary run one at a time.
static SERIAL: Mutex<()> = Mutex::new(());

/// Three fresh instances — `cx`, `cy`, `r` — minted in that order whatever
/// order they are then declared in.
fn instances() -> [UniformDecl; 3] {
    let decl = |default| UniformDecl {
        id: UniformIdentity::mint(),
        default,
    };
    [decl(0.0), decl(0.0), decl(1.0)]
}

/// `(x − cx)·r + cy` over three fresh instances declared as `cx`, `cy`, `r`.
fn circle_declared_in_order() -> Kernel {
    let [cx, cy, r] = instances();
    let mut a = ExprArena::new();
    let scx = a.declare_uniform(cx);
    let scy = a.declare_uniform(cy);
    let sr = a.declare_uniform(r);
    circle_over(a, [scx, scy, sr])
}

/// [`circle_declared_in_order`] with the instances declared as `r`, `cy`,
/// `cx`, so the link — not the declaration order — is what is shared.
fn circle_declared_reversed() -> Kernel {
    let [cx, cy, r] = instances();
    let mut a = ExprArena::new();
    let sr = a.declare_uniform(r);
    let scy = a.declare_uniform(cy);
    let scx = a.declare_uniform(cx);
    circle_over(a, [scx, scy, sr])
}

/// `(x − cx)·r + cy` over the three declared slots `[cx, cy, r]`.
fn circle_over(mut a: ExprArena, [scx, scy, sr]: [UniformId; 3]) -> Kernel {
    let x = a.push_var(0);
    let ucx = a.push_uniform(scx);
    let ur = a.push_uniform(sr);
    let ucy = a.push_uniform(scy);
    let d = a.push_binary(OpKind::Sub, x, ucx);
    let scaled = a.push_binary(OpKind::Mul, d, ur);
    let root = a.push_binary(OpKind::Add, scaled, ucy);
    Kernel::from_parts(a, root)
}

/// `base` with one more term, so the two tests in this binary never race
/// on one structure's first saturation.
fn shifted(base: &Kernel) -> Kernel {
    let (arena, root) = base.parts();
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
    let _ = compile(
        &shifted(&circle_declared_in_order()),
        LatticeShape::new([32, 32]),
    )
    .expect("compile");
    assert_eq!(saturation_count() - before, 1, "the first shape saturates");
    for shape in [[33, 32], [32, 33], [48, 8]] {
        let _ = compile(
            &shifted(&circle_declared_reversed()),
            LatticeShape::new(shape),
        )
        .expect("compile");
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
    let k = circle_declared_in_order();
    let first = compile(&k, SHAPE).expect("compile").kernel;
    let after_first = entry_count();
    assert_eq!(after_first - before, 1, "the first circle compiles once");
    for i in 1..1000 {
        let k = if i % 2 == 0 {
            circle_declared_in_order()
        } else {
            circle_declared_reversed()
        };
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

/// `(x·s − 6.125)·y` over a fresh scale `s`: a unit's body, of a structure
/// no other test here builds.
fn unit_body() -> Kernel {
    let mut a = ExprArena::new();
    let s = a.declare_uniform(UniformDecl {
        id: UniformIdentity::mint(),
        default: 2.0,
    });
    let x = a.push_var(0);
    let y = a.push_var(1);
    let us = a.push_uniform(s);
    let c = a.push_const(6.125);
    let xs = a.push_binary(OpKind::Mul, x, us);
    let shifted = a.push_binary(OpKind::Sub, xs, c);
    let root = a.push_binary(OpKind::Mul, shifted, y);
    Kernel::from_parts(a, root)
}

/// **A unit shared by two programs saturates once.** A named kernel is
/// optimized by itself, through the same structure-keyed cache as any term
/// (docs/plans/2026-09-25-the-language-is-kernel.md §4, O1), so a glyph that
/// recurs — in another program, or over another font's uniforms — pays only
/// for the term around it.
///
/// - The first program saturates its unit and the choice around it: 2.
/// - A second program around the *same* unit saturates only its own term: 1.
/// - A third around a unit of the same structure over fresh uniforms
///   saturates its own term again — the term around a unit keys it by value,
///   which is its identity — and not the unit: 1.
#[test]
fn a_unit_shared_by_two_programs_saturates_once() {
    use pixelflow_search::runtime::saturation_count;

    const SHAPE: LatticeShape = LatticeShape::new([16, 16]);
    let _serial = SERIAL.lock().expect("serial");
    let unit = unit_body().by_ref();
    let x = Kernel::x();

    let before = saturation_count();
    let chosen = Kernel::y()
        .lt(&Kernel::constant(3.0))
        .select(&unit, &Kernel::constant(0.0));
    let _ = compile(&chosen, SHAPE).expect("compile");
    assert_eq!(
        saturation_count() - before,
        2,
        "the unit and the term around it"
    );

    let before = saturation_count();
    let _ = compile(&unit.mul(&x), SHAPE).expect("compile");
    assert_eq!(
        saturation_count() - before,
        1,
        "the unit is saturated already; only the new term around it is not"
    );

    let before = saturation_count();
    let _ = compile(&unit_body().by_ref().mul(&x), SHAPE).expect("compile");
    assert_eq!(
        saturation_count() - before,
        1,
        "the term around a fresh unit saturates; the unit, a structure already saturated, does not"
    );
}
