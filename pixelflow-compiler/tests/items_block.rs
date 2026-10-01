//! The items form of `kernel!`: a block of `const`s, helpers and entries
//! (docs/plans/2026-09-25-the-language-is-kernel.md §1.2, Phase B1).
//!
//! What a body *means* is pinned against rustc in `rustc_is_the_oracle.rs`.
//! This file pins the block's shape: several entries sharing a helper, a
//! helper applied to shifted coordinates, an entry's parameters as its
//! uniforms, and `if` and `.select` lowering to one node. Records and the
//! binding times are `binding_times.rs`'s.

use pixelflow_compiler::{kernel, kernel_raw};
use pixelflow_core::{Kernel, Lattice, Manifold};
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

/// An entry's parameters are its uniforms (plan §1.4): the kernel declares
/// one per parameter, the call's values its defaults, and a program compiled
/// once is rebound per call from the entry's `Args` record. This used to
/// pin the call-site-type rule — an `f32` folded, a `Uniform` handle bound
/// — and moved its argument through the handle; the handle is gone, and the
/// assertion is kept, through `Args`.
#[test]
fn an_entrys_parameters_are_uniforms_rebound_through_args() {
    let k = radius2(1.0, 2.0);
    assert_eq!(k.uniforms().len(), 2, "cx and cy");
    assert_eq!(bake(&k), 13.0, "default cx = 1");

    let program = Manifold::compile(&k, [1, 1]);
    let mut block = program.block();
    Radius2Args { cx: 3.0, cy: 2.0 }
        .write_into(&mut block)
        .expect("radius2's arguments");
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

/// A block written where no prelude is in scope: a record, a `pub const`, a
/// helper, a tuple `let`, a fold over a structural count, the `Args`
/// record, the closure form, and an entry that takes a kernel — whose host
/// function runs lowering's steps, a fold's and a library method's and a
/// derivative's among them. The expansion names every item by path,
/// so it expands in a `#[no_implicit_prelude]` module as anywhere else —
/// which a bare `Some`, or a method called through a prelude trait, would
/// not.
#[no_implicit_prelude]
mod without_a_prelude {
    ::pixelflow_compiler::kernel! {
        /// A point with a weight.
        pub struct Mass { pub x: f32, pub w: f32 }

        /// Half.
        pub const HALF: f32 = 0.5;

        fn weighed(m: Mass, r: f32) -> f32 { m.x * r + m.w }

        /// The weighed point plus each fold index, plus a half where `v`
        /// is past `X`.
        pub fn swept<const N: usize>(m: Mass, v: f32, r: f32) -> f32 {
            let (a, b) = (r, X);
            (0..N).map(|i| weighed(m, a) + (i as f32)).sum::<f32>()
                + if b < v { HALF } else { 0.0 }
        }

        /// A kernel applied in a fold over a structural count, weighed.
        pub fn applied<const N: usize>(k: impl Fn(f32, f32) -> f32, m: Mass) -> f32 {
            weighed(m, (0..N).map(|i| k(X + (i as f32), Y)).sum::<f32>()) + X.fract() + DX(Y)
        }
    }

    /// The closure form, scaled.
    pub fn scaled(v: f32, r: f32) -> ::pixelflow_core::Kernel {
        let scaled = ::pixelflow_compiler::kernel!(|v: f32, r: f32| v * r + X);
        scaled(v, r)
    }
}

/// The block without a prelude means what it says, and its program rebinds
/// from its `Args`.
#[test]
fn a_block_expands_where_no_prelude_is_in_scope() {
    use without_a_prelude::{HALF, Mass, SweptArgs, scaled, swept};
    let m = Mass { x: 1.0, w: 2.0 };
    let (v, r) = (4.0, 0.5);
    let x = 0.5;
    let by_rust: f32 =
        (0..3).map(|i| (m.x * r + m.w) + i as f32).sum::<f32>() + if x < v { HALF } else { 0.0 };
    let kernel = swept::<3>(m, v, r);
    assert_eq!(Lattice::eval_at(&kernel, x, 0.0), by_rust);

    let lattice = Lattice::frame(4, 2);
    let program = Manifold::compile(&kernel, lattice.extent);
    let mut block = program.block();
    let args = SweptArgs::<3> {
        m: Mass { x: -1.0, w: 4.0 },
        v: 0.125,
        r: 2.0,
    };
    args.write_into(&mut block).expect("swept's own arguments");
    let rebound = lattice.collapse(&program.bind(&[]).with_uniforms(&block));
    let baked = lattice.bake(&swept::<3>(args.m, args.v, args.r));
    assert_eq!(rebound.buffer(), baked.buffer());

    assert_eq!(Lattice::eval_at(&scaled(1.0, 2.0), 0.5, 0.0), 2.5);

    // Σ_i (x + i + y) over three, times the weight, plus fract(x) and ∂Y/∂X.
    let k = ::pixelflow_compiler::kernel!(|| X + Y);
    let by_rust: f32 = m.x * (0..3).map(|i| x + i as f32 + 1.0).sum::<f32>() + m.w + x;
    let applied = without_a_prelude::applied::<3>(&k, m);
    assert_eq!(Lattice::eval_at(&applied, x, 1.0), by_rust);
}
