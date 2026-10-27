//! The items form of `kernel!`: a block of `const`s, helpers and entries
//! (docs/language/2026-09-25-the-language-is-kernel.md §1.2, Phase B1).
//!
//! What a body *means* is pinned against rustc in `rustc_is_the_oracle.rs`.
//! This file pins the block's shape: several entries sharing a helper, a
//! helper applied to shifted coordinates, an entry's parameters as its
//! uniforms, and `if` and `.select` lowering to one node. Records and the
//! binding times are `binding_times.rs`'s.

use pixelflow_compiler::{kernel, kernel_raw};
use pixelflow_core::{Kernel, Lattice, Manifold};

mod common;
use common::{AT, assert_same_program, bake};

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

    /// The same helper over shifted coordinates: `radius2(cx, cy)` at
    /// `(X + 1, Y - 1)`. A function is warped by applying it; a field, by
    /// `.at` (`at_is_contramap.rs`).
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
/// once is rebound per call from the entry's `Args` record.
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
    assert_same_program(&spelled_if, &spelled_select);
    assert_eq!(bake(&spelled_if), 6.0);
    assert_eq!(bake(&spelled_select), 6.0);
}

/// A block written where no prelude is in scope: a record, a `pub const`, a
/// helper, a tuple `let`, a fold over a `usize` const, a count parameter, the
/// `Args` record, the closure form, and an entry that takes a kernel — whose host
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

        /// How many terms.
        const N: usize = 3;

        fn weighed(m: Mass, r: f32) -> f32 { m.x * r + m.w }

        /// The weighed point plus each fold index, plus a half where `v`
        /// is past `X`.
        pub fn swept(m: Mass, v: f32, r: f32) -> f32 {
            let (a, b) = (r, X);
            (0..N).map(|i| weighed(m, a) + (i as f32)).sum::<f32>()
                + if b < v { HALF } else { 0.0 }
        }

        /// The weighed point summed over a count the host passes.
        pub fn counted(m: Mass, n: usize) -> f32 {
            (0..n).map(|i| weighed(m, i as f32)).sum::<f32>()
        }

        /// A kernel applied in a fold over a `usize` const, weighed.
        pub fn applied(k: impl Fn(f32, f32) -> f32, m: Mass) -> f32 {
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
    use without_a_prelude::{CountedArgs, HALF, Mass, SweptArgs, counted, scaled, swept};
    let m = Mass { x: 1.0, w: 2.0 };
    let (v, r) = (4.0, 0.5);
    let x = 0.5;
    let by_rust: f32 =
        (0..3).map(|i| (m.x * r + m.w) + i as f32).sum::<f32>() + if x < v { HALF } else { 0.0 };
    let kernel = swept(m, v, r);
    assert_eq!(Lattice::eval_at(&kernel, x, 0.0), by_rust);

    let lattice = Lattice::frame(4, 2);
    let program = Manifold::compile(&kernel, lattice.extent);
    let mut block = program.block();
    let args = SweptArgs {
        m: Mass { x: -1.0, w: 4.0 },
        v: 0.125,
        r: 2.0,
    };
    args.write_into(&mut block).expect("swept's own arguments");
    let rebound = lattice.collapse(&program.bind(&[]).with_uniforms(&block));
    let baked = lattice.bake(&swept(args.m, args.v, args.r));
    assert_eq!(rebound.buffer(), baked.buffer());

    // Σ_{i<4} (i·m.x + m.w), rebound from its `Args` at another count.
    let by_rust: f32 = (0..4).map(|i| (i as f32) * m.x + m.w).sum();
    assert_eq!(Lattice::eval_at(&counted(m, 4), x, 0.0), by_rust);
    let program = Manifold::compile(&counted(m, 0), lattice.extent);
    let mut block = program.block();
    CountedArgs { m, n: 4 }
        .write_into(&mut block)
        .expect("counted's own arguments");
    let rebound = lattice.collapse(&program.bind(&[]).with_uniforms(&block));
    assert_eq!(rebound.buffer(), lattice.bake(&counted(m, 4)).buffer());

    assert_eq!(Lattice::eval_at(&scaled(1.0, 2.0), 0.5, 0.0), 2.5);

    // Σ_i (x + i + y) over three, times the weight, plus fract(x) and ∂Y/∂X.
    let k = ::pixelflow_compiler::kernel!(|| X + Y);
    let by_rust: f32 = m.x * (0..3).map(|i| x + i as f32 + 1.0).sum::<f32>() + m.w + x;
    let applied = without_a_prelude::applied(&k, m);
    assert_eq!(Lattice::eval_at(&applied, x, 1.0), by_rust);
}

mod digit_named_entries {
    ::pixelflow_compiler::kernel! {
        pub fn _1(a: f32) -> f32 { a + X }
        pub fn _2d(a: f32) -> f32 { a }
        pub fn __3(a: f32) -> f32 { a }
    }
}

/// An entry whose name has no letter before its first digit still has an
/// `Args` record, spelled with a leading underscore so it is an identifier.
#[test]
fn an_entry_named_by_a_digit_has_an_args_record() {
    use digit_named_entries::{__3, _1, _1Args, _2d, _2dArgs, _3Args};

    assert_eq!(bake(&_1(2.0)), 5.0, "2 + X");
    assert_eq!(bake(&_2d(2.0)), 2.0);
    assert_eq!(bake(&__3(2.0)), 2.0);

    let program = Manifold::compile(&_1(2.0), [1, 1]);
    let mut block = program.block();
    _1Args { a: 4.0 }
        .write_into(&mut block)
        .expect("_1's arguments");
    let moved = program.bind(&[]).with_uniforms(&block).eval_at(AT.0, AT.1);
    assert_eq!(moved, 7.0, "4 + X");
    let _ = (_2dArgs { a: 0.0 }, _3Args { a: 0.0 });
}

/// A `macro_rules` `$t:ty` reaches `kernel!` as a type inside an invisible
/// group, and rustc reads the type inside, so every position that names a
/// type does too: a parameter, a return, a field, a `const`, a `let`, a
/// count parameter, a cast and a fold's `::<f32>`.
mod typed_by_a_fragment {
    use super::*;

    macro_rules! spelled_by_fragments {
        ($real:ty, $count:ty) => {
            ::pixelflow_compiler::kernel! {
                const SCALE: $real = 2.0;
                const FOUR: $count = 4;
                pub struct Pair { pub a: $real, pub b: $real }
                fn twice(v: $real) -> $real { v * 2.0 }
                pub fn spelled(n: $count, p: Pair, w: $real) -> $real {
                    let sum: $real = (0..n).map(|i| X * (i as $real)).sum::<$real>();
                    twice(sum) + p.a * p.b * w * SCALE + (0..FOUR).map(|i| i as f32).sum::<f32>()
                }
            }
            pub fn closure(v: $real) -> Kernel {
                let k = ::pixelflow_compiler::kernel!(|v: $real| {
                    let doubled: $real = v * 2.0;
                    doubled + X
                });
                k(v)
            }
        };
    }

    spelled_by_fragments!(f32, usize);

    #[test]
    fn every_position_reads_the_type_inside_the_group() {
        // X = 3: Σ_{i<3} 3i = 9, doubled 18, + 2·5·7·2 = 140, + (0+1+2+3) = 6.
        let k = spelled(3, Pair { a: 2.0, b: 5.0 }, 7.0);
        assert_eq!(bake(&k), 18.0 + 140.0 + 6.0);
        assert_eq!(bake(&closure(4.0)), 8.0 + 3.0);

        // Rebound from its `Args`, the program is the one built at those values.
        let program = Manifold::compile(&k, [1, 1]);
        let mut block = program.block();
        SpelledArgs {
            n: 3,
            p: Pair { a: 3.0, b: 1.0 },
            w: 2.0,
        }
        .write_into(&mut block)
        .expect("spelled's own arguments");
        let rebound = program.bind(&[]).with_uniforms(&block).eval_at(AT.0, AT.1);
        assert_eq!(rebound, bake(&spelled(3, Pair { a: 3.0, b: 1.0 }, 2.0)));
    }
}

kernel! {
    fn r#half(x: f32) -> f32 { x * 0.5 }

    pub fn halved() -> f32 { half(X) }
}

/// rustc reads `r#half` and `half` as one name, and so does the macro: the
/// helper spelled raw is the helper called plain.
#[test]
fn a_raw_spelling_is_the_name_it_spells() {
    assert_eq!(bake(&halved()), 1.5);
}
