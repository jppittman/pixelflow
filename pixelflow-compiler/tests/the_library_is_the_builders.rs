//! What the IR defines, `kernel!` builds through, and so does the builder:
//! a library method or a derivative written in the syntax is the one
//! `Kernel`'s method builds — the same arena, by
//! `pixelflow_ir::key::canonical` — because both call
//! `pixelflow_ir::library`'s one definition of it
//! (docs/plans/2026-09-25-the-language-is-kernel.md §1.1, B5).
//!
//! `fract`, `hypot` and `clamp` are library, not primitives; `DX`, `DY`
//! and the Hessian family are `Dwrt` chains, one per axis. Each used to be
//! written twice, in `kernel!`'s lowering and in `Kernel`'s method, and a
//! copy is a future divergence (CLAUDE.md). `kernel_raw!` keeps the lowered
//! shape, so each comparison is the front end's word and not the
//! optimizer's. `fold_is_kernel_over.rs` pins the folds the same way.

use pixelflow_compiler::kernel_raw;
use pixelflow_core::Kernel;
use pixelflow_ir::key::canonical;

/// Whether two kernels are one program: the same canonical key bytes, what
/// the JIT cache keys on. The tables beside them say which argument each
/// slot binds, and two instances of one argument are two identities.
fn assert_same_program(written: &Kernel, built: &Kernel) {
    let (written_arena, written_root) = written.parts();
    let (built_arena, built_root) = built.parts();
    assert_eq!(
        canonical(written_arena, written_root).key,
        canonical(built_arena, built_root).key,
        "written: {}\nbuilt:   {}",
        written_arena.display(written_root),
        built_arena.display(built_root),
    );
}

fn x() -> Kernel {
    Kernel::x()
}

fn y() -> Kernel {
    Kernel::y()
}

fn k(value: f32) -> Kernel {
    Kernel::constant(value)
}

#[test]
fn fract_is_kernel_fract() {
    let written = kernel_raw!(|| (X * 0.25).fract());
    let built = x().mul(&k(0.25)).fract();
    assert_same_program(&written, &built);
}

#[test]
fn hypot_is_kernel_hypot() {
    let written = kernel_raw!(|| (X - 3.0).hypot(Y * 0.5));
    let built = x().sub(&k(3.0)).hypot(&y().mul(&k(0.5)));
    assert_same_program(&written, &built);
}

/// A bound read from a parameter is a uniform in both: the key numbers a
/// uniform by where it is first read, so one argument in the same place is
/// one program.
#[test]
fn clamp_is_kernel_clamp() {
    let written = kernel_raw!(|lo: f32| X.clamp(lo, Y * 2.0))(0.25);
    let lo = pixelflow_ir::Uniform::new(0.25).kernel();
    let built = x().clamp(&lo, &y().mul(&k(2.0)));
    assert_same_program(&written, &built);
}

/// `V(e)` is `e`: every arena expression is already a value.
#[test]
fn v_is_the_value() {
    let written = kernel_raw!(|| V(X * Y));
    let built = x().mul(&y());
    assert_same_program(&written, &built);
}

#[test]
fn dx_is_kernel_dx() {
    let written = kernel_raw!(|| DX(X * X * Y));
    let built = x().mul(&x()).mul(&y()).dx();
    assert_same_program(&written, &built);
}

#[test]
fn dy_is_kernel_dy() {
    let written = kernel_raw!(|| DY(X * X * Y));
    let built = x().mul(&x()).mul(&y()).dy();
    assert_same_program(&written, &built);
}

/// The Hessian family is the derivative taken twice, innermost first:
/// `DXY(e)` is `e.dx().dy()`, not `e.dy().dx()` — the same function, and a
/// different term.
#[test]
fn the_hessian_is_the_derivative_twice() {
    let e = || x().mul(&x()).mul(&y()).mul(&y());
    assert_same_program(&kernel_raw!(|| DXX(X * X * Y * Y)), &e().dx().dx());
    assert_same_program(&kernel_raw!(|| DXY(X * X * Y * Y)), &e().dx().dy());
    assert_same_program(&kernel_raw!(|| DYY(X * X * Y * Y)), &e().dy().dy());

    let written = kernel_raw!(|| DXY(X * X * Y * Y));
    let swapped = e().dy().dx();
    let (arena, root) = written.parts();
    let (swapped_arena, swapped_root) = swapped.parts();
    assert_ne!(
        canonical(arena, root).key,
        canonical(swapped_arena, swapped_root).key,
        "DXY differentiates by X first"
    );
}

/// `Kernel::dwrt` by axis index is `dx`/`dy`: one numbering of the axes.
#[test]
fn dwrt_by_index_is_dx_and_dy() {
    let e = x().mul(&y()).sin();
    assert_same_program(&e.dwrt(0), &e.dx());
    assert_same_program(&e.dwrt(1), &e.dy());
}

kernel_raw! {
    /// A library method in a helper.
    fn radius(x: f32, y: f32) -> f32 {
        x.hypot(y)
    }

    /// Its derivative, through two more library methods.
    pub fn slope() -> f32 {
        DX(radius(X - 8.0, Y).clamp(1.0, 5.0).fract())
    }
}

/// The definitions compose as terms: a derivative of a library method, in
/// a helper, is the builder's derivative of its method.
#[test]
fn a_derivative_of_a_library_method_in_a_helper_is_the_builders() {
    let built = x()
        .sub(&k(8.0))
        .hypot(&y())
        .clamp(&k(1.0), &k(5.0))
        .fract()
        .dx();
    assert_same_program(&slope(), &built);
}
