//! Records, binding times and `Args`
//! (docs/plans/2026-09-25-the-language-is-kernel.md §1.3, §1.4; B3).
//!
//! An entry's `const N: usize` generics are **structural**: each value is its
//! own program. Everything else is a **uniform**: an `f32` parameter is one,
//! a record parameter one per field, and a call's values are the kernel's
//! arguments, never its constants. So every call of an entry at one
//! structural instantiation is one program — one canonical key, one compiled
//! region — and a program compiled once is rebound per call from the entry's
//! `Args` record, into a block the caller keeps; so is a program the entry's
//! kernel was composed into, beside kernels with no arguments of their own.
//!
//! What a record and a count *mean* is pinned against rustc in
//! `rustc_is_the_oracle.rs`; the refusals are `sema`'s and the parser's unit
//! tests. This file pins the binding: keys, positions and rebinding.

use pixelflow_compiler::{kernel, kernel_raw};
use pixelflow_core::{ArityMismatch, Kernel, Lattice, Manifold, Uniform};
use pixelflow_ir::key::canonical;

kernel! {
    /// An axis-aligned box.
    pub struct Bounds { pub x0: f32, pub y0: f32, pub x1: f32, pub y1: f32 }

    fn inside(b: Bounds, x: f32, y: f32) -> bool {
        (x >= b.x0) & (x <= b.x1) & (y >= b.y0) & (y <= b.y1)
    }

    /// `fg` inside the box, `bg` outside: a record parameter and two scalars.
    pub fn boxed(b: Bounds, fg: f32, bg: f32) -> f32 {
        let c = b;
        if inside(c, X, Y) { fg } else { bg }
    }

    /// A body that never reads its first parameter.
    pub fn skips(unread: f32, a: f32, c: f32) -> f32 { a * 10.0 + c }

    /// The mean of `X·i` over `i ∈ [0, N)`, scaled: `N` is the program.
    pub fn ramp<const N: usize>(scale: f32) -> f32 {
        (0..N).map(|i| X * (i as f32)).sum::<f32>() * scale / (N as f32)
    }

    /// `N` rings about a box's corner, `fg` on `bg`: the macro doc's
    /// example, compiled, so the doc cannot drift from the language. A
    /// record is its block's: this one could not name `Bounds` from another.
    pub fn rings<const N: usize>(b: Bounds, fg: f32, bg: f32) -> f32 {
        let dx = X - b.x0;
        let dy = Y - b.y0;
        let r = (dx * dx + dy * dy).sqrt();
        let within = (X <= b.x1) & (Y <= b.y1);
        let n: f32 = (0..N).map(|i| if r < (i as f32) + 1.0 { 1.0 } else { 0.0 }).sum();
        if within { fg * n / (N as f32) } else { bg }
    }
}

/// The lattice the rebinding tests collapse over: small, and not a whole
/// SIMD batch wide, so a row's remainder is exercised too.
const FRAME: (usize, usize) = (11, 7);

const UNIT: Bounds = Bounds {
    x0: 2.0,
    y0: 1.0,
    x1: 6.0,
    y1: 4.0,
};

/// The key the JIT cache compiles under: the canonical form's shape bytes,
/// which number a kernel's arguments by slot and hold neither their
/// identities nor their values.
fn program_key(k: &Kernel) -> Vec<u8> {
    let (arena, root) = k.parts();
    canonical(arena, root).key
}

/// The address of the code `k` compiles to at `FRAME`'s shape: one compiled
/// region per program, shared by every kernel of it.
fn code_at_frame(k: &Kernel) -> *const u8 {
    Manifold::compile(k, [FRAME.0 as u32, FRAME.1 as u32])
        .code_bytes()
        .as_ptr()
}

/// A record parameter is one uniform per field, in field order, before the
/// parameters after it: the call's values, in declaration order.
#[test]
fn a_record_parameter_is_one_uniform_per_field_in_declaration_order() {
    let k = boxed(UNIT, 1.0, 0.25);
    let defaults: Vec<f32> = k.uniforms().iter().map(|u| u.default).collect();
    assert_eq!(defaults, [2.0, 1.0, 6.0, 4.0, 1.0, 0.25]);
    assert_eq!(Lattice::eval_at(&k, 3.0, 2.0), 1.0, "inside the box");
    assert_eq!(Lattice::eval_at(&k, 7.0, 2.0), 0.25, "right of it");
}

/// Every call of an entry is one program: two calls with different values
/// share a canonical key and a compiled region. Each structural value is a
/// program of its own.
#[test]
fn every_call_of_an_entry_at_one_instantiation_is_one_program() {
    let a = boxed(UNIT, 1.0, 0.0);
    let b = boxed(
        Bounds {
            x0: -3.0,
            y0: 0.5,
            x1: 9.0,
            y1: 2.5,
        },
        0.5,
        0.75,
    );
    assert_eq!(program_key(&a), program_key(&b));
    assert_eq!(code_at_frame(&a), code_at_frame(&b), "one compiled region");

    assert_eq!(program_key(&ramp::<4>(1.0)), program_key(&ramp::<4>(2.5)));
    assert_eq!(
        code_at_frame(&ramp::<4>(1.0)),
        code_at_frame(&ramp::<4>(2.5))
    );
    assert_ne!(
        program_key(&ramp::<4>(1.0)),
        program_key(&ramp::<5>(1.0)),
        "a structural value is its own program"
    );
}

kernel_raw! {
    const TWO: usize = 2;
    const FIVE: usize = 5;

    /// Two ranges over structural parameters, one inside the other — `M`
    /// rows of `N` terms — and `N` as a value.
    pub fn grid<const M: usize, const N: usize>() -> f32 {
        (0..M)
            .map(|i| (0..N).map(|j| X * (i as f32) + Y * (j as f32)).sum::<f32>())
            .sum::<f32>()
            + (N as f32)
    }

    /// `grid::<2, 5>`, written over ranges known at expansion.
    pub fn grid_two_by_five() -> f32 {
        (0..TWO)
            .map(|i| (0..FIVE).map(|j| X * (i as f32) + Y * (j as f32)).sum::<f32>())
            .sum::<f32>()
            + (FIVE as f32)
    }
}

/// Each distinct range over structural parameters is a hole of its own. A
/// template with two, instantiated at `M = 2, N = 5`, is the program written
/// over `0..2` and `0..5`; at `M = 5, N = 2` it is another. Both are
/// `kernel_raw!`, so neither key is an optimizer's. Two ranges sharing one
/// hole would compile a program over one of them twice, with plausible
/// pixels.
#[test]
fn each_structural_range_is_a_hole_of_its_own() {
    assert_eq!(
        program_key(&grid::<2, 5>()),
        program_key(&grid_two_by_five())
    );
    assert_ne!(
        program_key(&grid::<5, 2>()),
        program_key(&grid::<2, 5>()),
        "the rows and the terms are not interchangeable"
    );
}

/// Structural parameters spelled as the emitted range's bounds once were,
/// in a module of their own so that the lint their spelling trips is
/// allowed for them alone.
#[allow(non_upper_case_globals)]
mod lower_case {
    use pixelflow_compiler::kernel;

    kernel! {
        /// A sum over `[lo, hi)`: the names the emission binds are its own.
        pub fn span<const lo: usize, const hi: usize>() -> f32 {
            (lo..hi).map(|i| i as f32).sum::<f32>()
        }
    }
}

/// An open fold's range is evaluated in code the entry's own names cannot
/// reach: a structural parameter named `lo` or `hi` is the range's bound,
/// not a pattern the evaluation's binding reads as it.
#[test]
fn a_structural_parameter_named_as_the_emissions_locals_is_its_own() {
    assert_eq!(
        Lattice::eval_at(&lower_case::span::<2, 5>(), 0.0, 0.0),
        9.0,
        "2 + 3 + 4"
    );
}

/// A program compiled once, rebound from two different `Args`, gives the
/// pixels of baking each call: the call's values and the `Args` record's are
/// one binding.
#[test]
fn a_program_compiled_once_is_rebound_from_args() {
    let lattice = Lattice::frame(FRAME.0, FRAME.1);
    let program = Manifold::compile(&boxed(Bounds::default(), 0.0, 0.0), lattice.extent);
    let bound = program.bind(&[]);
    let calls = [
        BoxedArgs {
            b: UNIT,
            fg: 1.0,
            bg: 0.25,
        },
        BoxedArgs {
            b: Bounds {
                x0: 0.0,
                y0: 3.0,
                x1: 9.0,
                y1: 6.0,
            },
            fg: -2.0,
            bg: 5.0,
        },
    ];
    let mut block = program.block();
    for args in calls {
        args.write_into(&mut block).expect("boxed's own arguments");
        let rebound = lattice.collapse(&bound.clone().with_uniforms(&block));
        let baked = lattice.bake(&boxed(args.b, args.fg, args.bg));
        assert_eq!(rebound.buffer(), baked.buffer(), "{args:?}");
    }

    let program = Manifold::compile(&ramp::<4>(0.0), lattice.extent);
    let mut block = program.block();
    RampArgs::<4> { scale: 2.0 }
        .write_into(&mut block)
        .expect("ramp's own argument");
    let rebound = lattice.collapse(&program.bind(&[]).with_uniforms(&block));
    assert_eq!(rebound.buffer(), lattice.bake(&ramp::<4>(2.0)).buffer());
}

/// A parameter the body never reads is still declared, so the arguments
/// after it keep their positions: its value is taken and dropped, not given
/// to the next one.
#[test]
fn an_unread_parameter_binds_positionally() {
    let k = skips(99.0, 1.0, 3.0);
    assert_eq!(k.uniforms().len(), 3, "the unread parameter is declared");
    assert_eq!(Lattice::eval_at(&k, 0.0, 0.0), 13.0);

    let program = Manifold::compile(&k, [1, 1]);
    assert_eq!(program.uniforms().len(), 2, "the code reads `a` and `c`");
    let mut block = program.block();
    SkipsArgs {
        unread: 42.0,
        a: 2.0,
        c: 5.0,
    }
    .write_into(&mut block)
    .expect("skips's own arguments");
    assert_eq!(
        program.bind(&[]).with_uniforms(&block).eval_at(0.0, 0.0),
        25.0,
        "a = 2 and c = 5; a shifted binding would give 42 · 10 + 2"
    );
}

/// `Args` written for a program of another arity is a typed error, as are
/// values of the wrong count and an entry's `Args` for a composition that
/// declares other arguments beside the entry's; none truncates or pads, and
/// nothing is written.
#[test]
fn an_arity_mismatch_is_an_error() {
    let program = Manifold::compile(&boxed(UNIT, 1.0, 0.0), [1, 1]);
    let mut block = program.block();
    let skips_args = SkipsArgs {
        unread: 0.0,
        a: 1.0,
        c: 2.0,
    };
    assert_eq!(
        skips_args.write_into(&mut block),
        Err(ArityMismatch {
            declared: 6,
            supplied: 3
        })
    );
    assert_eq!(
        block.set_declared([1.0; 7]),
        Err(ArityMismatch {
            declared: 6,
            supplied: 7
        })
    );
    assert_eq!(block.values(), program.block().values(), "nothing written");

    let beside = Uniform::new(1.0).kernel().add(&boxed(UNIT, 1.0, 0.0));
    let program = Manifold::compile(&beside, [1, 1]);
    let boxed_args = BoxedArgs {
        b: UNIT,
        fg: 1.0,
        bg: 0.0,
    };
    assert_eq!(
        boxed_args.write_into(&mut program.block()),
        Err(ArityMismatch {
            declared: 7,
            supplied: 6
        }),
        "the uniform's argument is declared first, then boxed's six"
    );
}

/// A kernel the entry's is composed into, beside kernels with no arguments
/// of their own, declares the entry's arguments in the entry's order — as
/// the right operand of an arithmetic node or a warp's coordinate, which a
/// composition splices in rather than copies — so the entry's `Args`
/// rebinds it: the pixels are those of baking the same composition of the
/// call. An unread parameter keeps its place there too.
#[test]
fn the_entrys_args_rebind_a_kernel_it_is_composed_into() {
    let lattice = Lattice::frame(FRAME.0, FRAME.1);
    let compositions: [fn(&Kernel) -> Kernel; 3] = [
        |k| Kernel::constant(0.5).add(k),
        |k| Kernel::x().mul(k),
        |k| Kernel::x().add(&Kernel::y()).at(k, &Kernel::y()),
    ];
    let args = BoxedArgs {
        b: Bounds {
            x0: 1.0,
            y0: 2.0,
            x1: 7.0,
            y1: 5.0,
        },
        fg: 3.0,
        bg: -0.5,
    };
    for (i, compose) in compositions.iter().enumerate() {
        let program = Manifold::compile(&compose(&boxed(UNIT, 1.0, 0.25)), lattice.extent);
        let mut block = program.block();
        args.write_into(&mut block)
            .expect("the entry's arguments, in its order");
        let rebound = lattice.collapse(&program.bind(&[]).with_uniforms(&block));
        let baked = lattice.bake(&compose(&boxed(args.b, args.fg, args.bg)));
        assert_eq!(rebound.buffer(), baked.buffer(), "composition {i}");
    }

    let composed = Kernel::constant(1.0).add(&skips(99.0, 1.0, 3.0));
    assert_eq!(composed.uniforms().len(), 3, "the unread parameter joins");
    let program = Manifold::compile(&composed, [1, 1]);
    let mut block = program.block();
    SkipsArgs {
        unread: 42.0,
        a: 2.0,
        c: 5.0,
    }
    .write_into(&mut block)
    .expect("skips's own arguments");
    assert_eq!(
        program.bind(&[]).with_uniforms(&block).eval_at(0.0, 0.0),
        26.0,
        "1 + a·10 + c with a = 2 and c = 5"
    );
}

/// How far a quotient may sit from its value, relative, at every ISA tier.
///
/// The optimizer may divide through `Recip`, which is an estimate whose
/// accuracy is the tier's (CLAUDE.md, "Floating point at the edges"):
/// `rcpps`, AVX2's, is good to about 1.5·2⁻¹² and gives `3 / 4` as
/// `0.7498169`; AVX-512's `vrcp14ps` happens to be exact on `1/4`; NEON's
/// `FRECPE` plus one `FRECPS` step is closer than `rcpps`. A quotient pinned
/// bit for bit is a pin on one tier. 2⁻¹¹ admits every tier's estimate and
/// is far inside the gap between one ring count and the next.
const RECIP_RELATIVE_TOLERANCE: f32 = 1.0 / 2048.0;

/// The macro doc's binding-times example: one call baked, and the program
/// compiled once and rebound from `Args` with another call's values, which
/// gives that call's pixels — exactly, since both run one program at one
/// tier.
#[test]
fn the_macro_docs_example_bakes_and_rebinds() {
    let lattice = Lattice::frame(FRAME.0, FRAME.1);
    let b = Bounds {
        x0: 1.0,
        y0: 1.0,
        x1: 8.0,
        y1: 5.0,
    };
    let once = lattice.bake(&rings::<4>(b, 1.0, 0.0));
    let corner = once.buffer()[0];
    assert!(
        (corner - 0.75).abs() <= 0.75 * RECIP_RELATIVE_TOLERANCE,
        "(0, 0) is inside, √2 from the corner: within three of the four rings, so 3/4 \
         up to the tier's reciprocal; got {corner}"
    );

    let program = Manifold::compile(&rings::<4>(b, 1.0, 0.0), lattice.extent);
    let mut block = program.block();
    RingsArgs::<4> {
        b,
        fg: 0.5,
        bg: 0.25,
    }
    .write_into(&mut block)
    .expect("rings' own arguments");
    let again = lattice.collapse(&program.bind(&[]).with_uniforms(&block));
    assert_eq!(
        again.buffer(),
        lattice.bake(&rings::<4>(b, 0.5, 0.25)).buffer()
    );
}
