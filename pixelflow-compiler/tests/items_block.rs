//! The items form of `kernel!`: a block of `const`s, helpers and entries
//! (docs/plans/2026-09-25-the-language-is-kernel.md §1.2, Phase B1).
//!
//! What a body *means* is pinned against rustc in `rustc_is_the_oracle.rs`.
//! This file pins the block's shape: several entries sharing a helper, a
//! helper applied to shifted coordinates, an entry's parameters bound as a
//! builder's are, and `if` and `.select` lowering to one node.

use pixelflow_compiler::{kernel, kernel_raw};
use pixelflow_core::{Kernel, Lattice, Manifold, Uniform};
use pixelflow_ir::key::canonical;

/// The sample every entry is read at: `X = 3`, `Y = 5`.
const AT: (f32, f32) = (3.0, 5.0);

fn bake(k: &Kernel) -> f32 {
    Lattice::eval_at(k, AT.0, AT.1)
}

kernel! {
    /// Shared by every entry; inlined at each call.
    fn sq(x: f32) -> f32 { x * x }

    /// A helper of the coordinates: an entry passes `X` and `Y` to it, or
    /// anything else — application is contramap.
    fn radius2_at(x: f32, y: f32, cx: f32, cy: f32) -> f32 { sq(x - cx) + sq(y - cy) }

    /// A mask is an ordinary argument of a helper.
    fn nearer(m: bool, a: f32, b: f32) -> f32 { if m { a } else { b } }

    pub fn radius2(cx: f32, cy: f32) -> f32 { radius2_at(X, Y, cx, cy) }

    pub fn ring(cx: f32, cy: f32, r: f32) -> f32 { radius2_at(X, Y, cx, cy) - sq(r) }

    /// The same helper over shifted coordinates: today's `.at(X + 1, Y - 1)`.
    pub fn shifted_radius2(cx: f32, cy: f32) -> f32 { radius2_at(X + 1.0, Y - 1.0, cx, cy) }

    pub fn nearest_axis() -> f32 { nearer(X.abs() < Y.abs(), X, Y) }

    /// An entry may be a mask.
    pub fn inside(r: f32) -> bool { radius2_at(X, Y, 0.0, 0.0) < sq(r) }
}

/// Several entries, one helper: each entry is its own `Kernel`, and the
/// helper is inlined into each.
#[test]
fn entries_share_a_helper() {
    // (3 - 1)² + (5 - 2)² = 13.
    assert_eq!(bake(&radius2(1.0, 2.0)), 13.0);
    assert_eq!(bake(&ring(1.0, 2.0, 2.0)), 9.0);
    assert_eq!(bake(&nearest_axis()), 3.0, "|X| < |Y| picks X");
}

/// A helper takes its coordinates as arguments, so applying it to a
/// shifted coordinate warps it: `(4 - 1)² + (4 - 2)²`.
#[test]
fn applying_a_helper_to_a_shifted_coordinate_warps_it() {
    assert_eq!(bake(&shifted_radius2(1.0, 2.0)), 13.0);
    assert_eq!(bake(&shifted_radius2(0.0, 0.0)), 32.0);
}

/// An entry's parameters are bound exactly as a builder's are: an `f32`
/// folds, a `Uniform` handle is an argument of the compiled kernel.
#[test]
fn an_entrys_parameters_bind_as_a_builders_do() {
    let folded = radius2(1.0, 2.0);
    assert!(folded.parts().0.uniforms().is_empty());

    let cx = Uniform::new(1.0);
    let k = radius2(cx, 2.0);
    assert_eq!(k.parts().0.uniforms(), &[cx.decl()]);
    assert_eq!(bake(&k), 13.0, "default cx = 1");

    let program = Manifold::compile(&k, [1, 1]);
    let mut block = program.block();
    block.set(cx, 3.0).expect("cx is the argument");
    let moved = program.bind(&[]).with_uniforms(&block).eval_at(AT.0, AT.1);
    assert_eq!(moved, 9.0, "(3 − 3)² + (5 − 2)²");
}

/// A `bool` entry is a mask `Kernel`, and composes as one.
#[test]
fn an_entry_may_return_a_mask() {
    let one = Kernel::constant(1.0);
    let zero = Kernel::constant(0.0);
    assert_eq!(bake(&inside(6.0).select(&one, &zero)), 1.0, "9 + 25 < 36");
    assert_eq!(bake(&inside(5.0).select(&one, &zero)), 0.0, "9 + 25 ≥ 25");
}

/// `if m { a } else { b }` and `m.select(a, b)` are one node: the arenas
/// they lower to have the same canonical form. `kernel_raw!` keeps the
/// lowered shape, so this is the front end's word and not the optimizer's.
#[test]
fn if_and_select_lower_to_the_same_arena() {
    let spelled_if = kernel_raw!(|| if X < Y { X * 2.0 } else { Y });
    let spelled_select = kernel_raw!(|| (X.lt(Y)).select(X * 2.0, Y));
    let (arena_if, root_if) = spelled_if.parts();
    let (arena_select, root_select) = spelled_select.parts();
    assert_eq!(
        canonical(arena_if, root_if),
        canonical(arena_select, root_select)
    );
    assert_eq!(bake(&spelled_if), 6.0);
    assert_eq!(bake(&spelled_select), 6.0);
}
