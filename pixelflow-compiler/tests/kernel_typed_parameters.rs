//! Kernel-typed parameters: a `kernel!` entry that takes a kernel the host
//! passes at run time, `k: impl Fn(f32, f32) -> f32`, and applies it,
//! `k(x, y)` (docs/plans/2026-09-25-the-language-is-kernel.md §1.3, §1.4,
//! D4, D6, Phase D-a).
//!
//! What an application *means* is pinned against rustc in
//! `rustc_is_the_oracle.rs`: a Rust closure passed for the same parameter of
//! the same tokens gives the same value. This file pins what the composed
//! program *is*:
//!
//! - **contramap**: `k(u, v)` is `k.at(u, v)`, one program by canonical key,
//!   and the program an entry builds is the one lowering builds with `k` a
//!   helper — so a composition through the syntax and the same composition
//!   through the builder share a JIT cache entry;
//! - **capture avoidance** (D4): an argument holding its own fold, applied
//!   inside a fold whose index the coordinates read, keeps its binder, and
//!   the fold around it takes another;
//! - **binding** (§1.4): the entry's own uniforms first, in declaration
//!   order, then each argument's, in parameter order and the argument's own
//!   order, read or not, and an instance passed twice declared once — the
//!   order a positional binding of the composed program reads (O3);
//! - **admission**: an argument is closed and reads no table.

use pixelflow_compiler::{kernel, kernel_raw};
use pixelflow_core::{DiscreteManifold, Kernel, Lattice, Uniform};
use pixelflow_ir::key::canonical;
use pixelflow_ir::{ExprArena, ExprNode, OpKind};

/// The sample the values are read at: `X = 3`, `Y = 5`.
const AT: (f32, f32) = (3.0, 5.0);

fn bake(k: &Kernel) -> f32 {
    Lattice::eval_at(k, AT.0, AT.1)
}

/// Whether two kernels are one program: the same canonical key.
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

fn constant(v: f32) -> Kernel {
    Kernel::constant(v)
}

kernel! {
    /// The monoid's operation: two kernels summed at the sample.
    pub fn sum2(a: impl Fn(f32, f32) -> f32, b: impl Fn(f32, f32) -> f32) -> f32 {
        a(X, Y) + b(X, Y)
    }

    /// `k` under a warp, and under another.
    pub fn warped(k: impl Fn(f32, f32) -> f32) -> f32 {
        k(X + 1.0, Y * 2.0) - k(Y, X)
    }

    /// D4's pin: an argument applied inside a fold whose index the
    /// coordinates read.
    pub fn shifted_sum(k: impl Fn(f32, f32) -> f32) -> f32 {
        (0..3).map(|i| k(X + (i as f32), Y)).sum()
    }

    /// The same, the fold and the application a helper's, inlined.
    fn column_sum(k: impl Fn(f32, f32) -> f32, x: f32, y: f32) -> f32 {
        (0..3).map(|i| k(x + (i as f32), y)).sum()
    }

    pub fn shifted_sum_by_helper(k: impl Fn(f32, f32) -> f32) -> f32 {
        column_sum(k, X, Y)
    }

    /// The entry's own uniform first, then the arguments', in parameter
    /// order: `r`, then `a`'s, then `b`'s, whatever order the body applies
    /// them in.
    pub fn weighted(a: impl Fn(f32, f32) -> f32, r: f32, b: impl Fn(f32, f32) -> f32) -> f32 {
        b(X, Y) * r + a(X, Y)
    }

    /// An argument applied under a structural count's fold.
    pub fn mean_down<const N: usize>(k: impl Fn(f32, f32) -> f32) -> f32 {
        (0..N).map(|i| k(X, Y + (i as f32))).sum::<f32>() / (N as f32)
    }

    /// An argument that is never applied: its uniforms are declared all the
    /// same, so a positional binding does not shift when one goes unread.
    pub fn first(a: impl Fn(f32, f32) -> f32, b: impl Fn(f32, f32) -> f32) -> f32 {
        a(X, Y)
    }

    /// `DX` of an application under a warp: the chain rule sees through
    /// it.
    pub fn stretched(k: impl Fn(f32, f32) -> f32) -> f32 {
        DX(k(X * 2.0, Y))
    }

    /// An application under a warp, and nothing else: a derivative is the
    /// argument's own, inside it.
    pub fn warp2(k: impl Fn(f32, f32) -> f32) -> f32 {
        k(X * 2.0, Y)
    }

    /// An argument applied at the sample, inside a fold whose index the
    /// body reads beside it.
    pub fn beside_the_index(k: impl Fn(f32, f32) -> f32) -> f32 {
        (0..3).map(|i| k(X, Y) + X * (i as f32)).sum()
    }
}

// The helper form of `shifted_sum`: the argument written as a helper of the
// coordinates, inlined at its application. It is the program the entry
// builds, by the denotation's law, so it is the one key both share.
kernel_raw! {
    fn k(x: f32, y: f32) -> f32 {
        (0..4).map(|j| x * (j as f32) + y).sum()
    }
    pub fn shifted_sum_inline() -> f32 {
        (0..3).map(|i| k(X + (i as f32), Y)).sum()
    }
}

/// `Σ_j (X·j + Y)` over `j ∈ [0, 4)`, a kernel holding its own fold: at
/// `(x, y)` it is `6x + 4y`.
fn folded() -> Kernel {
    kernel_raw!(|| (0..4).map(|j| X * (j as f32) + Y).sum())
}

// ─────────────────────────── contramap ───────────────────────────

/// At the sample, an application is the argument itself: `sum2(a, b)` is
/// `a + b`, the builder's sum, one program.
#[test]
fn a_sum_of_two_kernels_is_their_sum() {
    let a = kernel!(|| X * Y);
    let b = kernel!(|r: f32| X - r)(0.5);
    let summed = sum2(&a, &b);
    assert_eq!(bake(&summed), 15.0 + 2.5);
    assert_same_program(&summed, &a.add(&b));
}

/// One instance passed for both parameters is one argument: its value
/// twice, its uniforms declared once.
#[test]
fn one_kernel_passed_twice_is_one_argument() {
    let a = kernel!(|r: f32| X * r)(2.0);
    let doubled = sum2(&a, &a);
    assert_eq!(bake(&doubled), 12.0);
    assert_eq!(doubled.uniforms(), a.uniforms());
}

/// Under a warp an application is `Kernel::at`: the same program.
#[test]
fn an_application_is_at() {
    let k = kernel!(|s: f32| X * 3.0 - Y * s)(1.0);
    let written = warped(&k);
    let built = k
        .at(&x().add(&constant(1.0)), &y().mul(&constant(2.0)))
        .sub(&k.at(&y(), &x()));
    assert_same_program(&written, &built);
    // (4·3 − 10) − (5·3 − 3).
    assert_eq!(bake(&written), -10.0);
}

/// D4. An argument holding its own fold, applied inside a fold whose index
/// the coordinates read, is the same program three ways — the entry, the
/// builder's `Kernel::over` around `at`, and the argument written as a
/// helper — and captures nothing: `Σ_i (6(X + i) + 4Y)` is 132 at (3, 5),
/// where the argument's fold binding the outer index gives 156.
#[test]
fn an_argument_holding_a_fold_inside_a_fold_captures_nothing() {
    let k = folded();
    let written = shifted_sum(&k);
    let built = Kernel::sum_over(3, |i| k.at(&x().add(i), &y()));
    let inline = shifted_sum_inline();
    assert_same_program(&written, &built);
    assert_same_program(&written, &inline);
    assert_eq!(bake(&written), 132.0);
    assert_same_program(&shifted_sum_by_helper(&k), &written);
}

/// Each fold binds its own slot: the argument's, inside, the one its
/// closing chose, and the entry's, around it, the next.
#[test]
fn the_fold_around_an_application_binds_past_the_arguments() {
    let k = folded();
    let written = shifted_sum(&k);
    let (arena, root) = written.parts();
    let ExprNode::Reduce { fold: outer, .. } = arena.node(root) else {
        panic!("the entry's fold, got {}", arena.display(root));
    };
    let (k_arena, k_root) = k.parts();
    let ExprNode::Reduce { fold: inner, .. } = k_arena.node(k_root) else {
        panic!("the argument's fold, got {}", k_arena.display(k_root));
    };
    assert_eq!(inner.binder().slot(), 0);
    assert_eq!(outer.binder().slot(), 1);
}

/// An argument with two nested folds keeps both, and the fold around its
/// application takes a third slot: the value is rustc's for the same sums.
#[test]
fn an_argument_with_nested_folds_keeps_them() {
    let k = kernel_raw!(|| (0..2)
        .map(|i| (0..3).map(|j| X * (i as f32) + Y * (j as f32)).sum::<f32>())
        .sum());
    let rust_k = |x: f32, y: f32| -> f32 {
        (0..2)
            .map(|i| (0..3).map(|j| x * (i as f32) + y * (j as f32)).sum::<f32>())
            .sum()
    };
    let rust: f32 = (0..3).map(|i| rust_k(AT.0 + (i as f32), AT.1)).sum();
    let written = shifted_sum(&k);
    assert_eq!(bake(&written), rust);
    assert_same_program(&written, &Kernel::sum_over(3, |i| k.at(&x().add(i), &y())));
}

/// A derivative of an application is the chain rule through the warp:
/// `DX(k(2X, Y))` with `k = X·X` is `DX((2X)²)`, `8X`, 24 at `X = 3` — as
/// `.at` then `.dx` gives.
#[test]
fn a_derivative_of_an_application_follows_the_warp() {
    let k = kernel!(|| X * X);
    let written = stretched(&k);
    assert_eq!(bake(&written), 24.0);
    assert_eq!(bake(&k.at(&x().mul(&constant(2.0)), &y()).dx()), 24.0);
}

/// A derivative inside an argument survives the application and is taken
/// in the sample too: `k = DX(X·X)` applied at `(2X, Y)` is `DX((2X)²)`,
/// `8X`, 24 at `X = 3`, and `k.at(2X, Y)`'s program — where resolving the
/// derivative first, `2x`, and reading it at `2X` would give 12. (The
/// derivative outside the application, above, cannot tell the two apart:
/// both give 24.)
#[test]
fn a_derivative_inside_an_argument_follows_the_warp() {
    let k = kernel!(|| DX(X * X));
    let written = warp2(&k);
    assert_eq!(bake(&written), 24.0);
    assert_same_program(&written, &k.at(&x().mul(&constant(2.0)), &y()));
    // Resolved before the warp: `2x` read at `x = 2·3`.
    let resolved_first = 2.0 * (2.0 * AT.0);
    assert_eq!(resolved_first, 12.0);
    assert_ne!(bake(&written), resolved_first);
}

/// A structural count's fold around an application: each count is its own
/// program, built when the host function is instantiated and called.
#[test]
fn a_structural_entry_applies_its_argument() {
    let k = kernel!(|| X + Y);
    // (8 + 9 + 10 + 11) / 4.
    assert_eq!(bake(&mean_down::<4>(&k)), 9.5);
    assert_eq!(bake(&mean_down::<1>(&k)), 8.0);
}

// ─────────────────────────── binding ───────────────────────────

/// A kernel declaring `decls` in order, reading them last first: the order
/// a walk would meet them in is not the order they are declared in.
fn read_backwards(decls: &[pixelflow_ir::arena::UniformDecl]) -> Kernel {
    let mut arena = ExprArena::new();
    let slots: Vec<_> = decls.iter().map(|d| arena.declare_uniform(*d)).collect();
    let mut root = arena.push_const(0.0);
    for &slot in slots.iter().rev() {
        let leaf = arena.push_uniform(slot);
        root = arena.push_binary(OpKind::Add, root, leaf);
    }
    Kernel::from_parts(arena, root)
}

/// The entry's own uniforms, then each argument's, in parameter order — not
/// the order the body applies them in — each in the argument's own
/// declaration order, read or not.
#[test]
fn the_entrys_uniforms_come_first_then_each_arguments_in_order() {
    let [a0, b0, b1, b2] = [1.0, 2.0, 3.0, 4.0].map(Uniform::new);
    let a = a0.kernel().mul(&x());
    // `b2` is declared and read by nothing.
    let b = {
        let read = read_backwards(&[b0.decl(), b1.decl()]);
        let mut arena = ExprArena::new();
        let (read_arena, read_root) = read.parts();
        for u in [b0, b1, b2] {
            arena.declare_uniform(u.decl());
        }
        let root = arena.splice(read_arena, read_root);
        Kernel::from_parts(arena, root)
    };
    assert_eq!(b.uniforms(), [b0.decl(), b1.decl(), b2.decl()]);

    let composed = weighted(&a, 0.5, &b);
    let defaults: Vec<f32> = composed.uniforms().iter().map(|u| u.default).collect();
    assert_eq!(defaults, [0.5, 1.0, 2.0, 3.0, 4.0]);
    assert_eq!(
        &composed.uniforms()[1..],
        [a0, b0, b1, b2].map(Uniform::decl)
    );
    // (2 + 3) · 0.5 + 3.
    assert_eq!(bake(&composed), 5.5);
}

/// An argument never applied declares its uniforms all the same, and one
/// passed for two parameters declares them once, where it first appears.
#[test]
fn an_unapplied_argument_declares_and_a_shared_one_declares_once() {
    let [a0, b0] = [1.0, 2.0].map(Uniform::new);
    let (a, b) = (a0.kernel(), b0.kernel());
    assert_eq!(first(&a, &b).uniforms(), [a0.decl(), b0.decl()]);
    let shared = weighted(&a, 0.5, &a);
    let defaults: Vec<f32> = shared.uniforms().iter().map(|u| u.default).collect();
    assert_eq!(defaults, [0.5, 1.0]);
}

/// The program an entry builds holds the program and nothing else: what a
/// substitution replaced, and a uniform nothing reads, are dropped, as
/// emission drops them from an arena built at expansion.
#[test]
fn a_composed_kernel_holds_only_its_program() {
    let k = kernel!(|s: f32| X * 3.0 - Y * s)(1.0);
    for composed in [warped(&k), shifted_sum(&folded()), first(&k, &k)] {
        let (arena, root) = composed.parts();
        let mut reached = vec![false; arena.len()];
        let mut stack = vec![root];
        while let Some(id) = stack.pop() {
            if !std::mem::replace(&mut reached[id.0 as usize], true) {
                stack.extend(arena.children(id));
            }
        }
        assert!(reached.iter().all(|&r| r), "{}", arena.display(root));
    }
}

// ─────────────────────────── admission ───────────────────────────

/// At the sample an argument is spliced as it stands, so a name in it stays
/// a name: a glyph held by reference (O1 of the plan) is still one when it
/// is summed, and draws what it names.
#[test]
fn a_name_summed_at_the_sample_stays_a_name() {
    let a = kernel!(|| X * Y);
    let b = kernel!(|| X - Y);
    let named = sum2(&a.by_ref(), &b.by_ref());
    let (arena, root) = named.parts();
    let names = arena
        .nodes()
        .filter(|(_, n)| matches!(n, ExprNode::Ref(_)))
        .count();
    assert_eq!(names, 2, "{}", arena.display(root));
    assert_eq!(bake(&named), bake(&sum2(&a, &b)));
}

/// A name applied at the sample inside a fold stays a name, so the fold
/// around it cannot see the referent's own fold and may choose its binder:
/// once the name is expanded the referent's fold shadows it, which is
/// binding, not capture. `Σ_i (k(X, Y) + X·i)` with `k = 6X + 4Y`, a fold
/// of its own, is 123 at (3, 5) held by name or as itself, as Rust's sum
/// is. The two are one program under two keys: `at` would expand the name.
#[test]
fn a_name_applied_inside_a_fold_is_what_it_names() {
    let k = folded();
    let rust_k = |x: f32, y: f32| -> f32 { (0..4).map(|j| x * (j as f32) + y).sum() };
    let rust: f32 = (0..3).map(|i| rust_k(AT.0, AT.1) + AT.0 * (i as f32)).sum();
    assert_eq!(rust, 123.0);
    let named = beside_the_index(&k.by_ref());
    let (arena, root) = named.parts();
    assert!(
        arena.nodes().any(|(_, n)| matches!(n, ExprNode::Ref(_))),
        "the name is kept: {}",
        arena.display(root)
    );
    assert_eq!(bake(&named), rust);
    assert_eq!(bake(&beside_the_index(&k)), rust);
}

/// An argument reads no table: the language has none (D3).
#[test]
#[should_panic(expected = "reads a table")]
fn an_argument_reading_a_table_is_refused() {
    let table = DiscreteManifold::new(vec![1.0], 1, 1).kernel();
    let _refused = sum2(&table, &x());
}

/// An argument is closed: a `Kernel::over` body passed on from inside its
/// closure still holds the fold's placeholder, which a fold around the
/// application would otherwise bind.
#[test]
#[should_panic(expected = "an index no fold in it binds")]
fn an_open_argument_is_refused() {
    let _refused = Kernel::sum_over(2, |i| shifted_sum(&i.mul(&x())));
}
