//! Sema's contract (`sema/mod.rs`): `analyze` over what `parse` gives,
//! asserting the `AnalyzedKernel` or the refusal text.

use crate::parser::parse;
use crate::sema::{AnalyzedKernel, analyze};
use pixelflow_ir::passes::lattice::KERNEL_BINDER_CAPACITY;
use proc_macro2::TokenStream;
use quote::quote;

/// `sema`'s refusal of `input`, as text.
fn refusal(input: TokenStream) -> String {
    let def = parse(input).expect("the input parses");
    analyze(def).expect_err("sema refuses it").to_string()
}

fn accepted(input: TokenStream) -> AnalyzedKernel {
    let def = parse(input).expect("the input parses");
    analyze(def).expect("sema accepts it")
}

/// A body that names `Z` or `W` is a compile error that says where the
/// value goes instead — not a capture from the caller's scope, which is
/// what an unknown name would otherwise become.
#[test]
fn a_body_naming_a_retired_axis_is_refused_with_the_uniform_note() {
    for body in [quote! { || X + Z }, quote! { || X * W }] {
        let text = refusal(body);
        assert!(
            text.contains("no longer a coordinate")
                && text.contains("every parameter is a uniform"),
            "the message must point at uniforms, got: {text}"
        );
    }
}

/// A parameter may still be *called* Z: the refusal is about the axis a
/// lattice does not have, not about the letter.
#[test]
fn a_parameter_named_z_is_ordinary() {
    accepted(quote! { |Z: f32| X + Z });
}

#[test]
fn a_kernel_over_the_axes_and_a_parameter_is_accepted() {
    accepted(quote! { |r: f32| X * X + Y * Y - r });
}

/// A name nothing in the kernel binds is refused here, with its span.
/// Accepting it as a capture from the caller's scope would be one stage
/// accepting what a later one refuses: arena lowering has no node for a
/// captured Rust binding. The way out is a parameter, which is what the
/// message says.
#[test]
fn an_unknown_name_is_refused_not_captured() {
    let text = refusal(quote! { |r: f32| X * X + captured_from_env });
    assert!(
        text.contains("cannot find `captured_from_env`") && text.contains("parameter"),
        "the message must name the identifier and the way out, got: {text}"
    );
}

/// A `let` inside a block goes out of scope where the block ends, and a
/// use after it is an error, as rustc makes it one.
#[test]
fn a_local_used_after_its_block_ends_is_refused() {
    let err = refusal(quote! {
        || {
            {
                let a = X;
                a
            };
            a
        }
    });
    assert!(err.contains("cannot find `a`"), "got: {err}");
}

/// The caller's locals are not in scope in the helper: a helper naming
/// one the caller happens to bind is an error, not dynamic scoping.
#[test]
fn a_helper_does_not_see_the_callers_locals() {
    let err = refusal(quote! {
        fn leak() -> f32 { a }
        pub fn f() -> f32 { let a = X; leak() }
    });
    assert!(err.contains("cannot find `a`"), "got: {err}");
}

/// `let X` and `let Y` are refused, as a parameter named X or Y is: a
/// binding that shadowed a coordinate would mean one thing to one stage
/// and another to the next.
#[test]
fn a_let_named_after_a_coordinate_is_refused() {
    for input in [
        quote! { || { let X = Y; X } },
        quote! { || { let Y: f32 = X; Y } },
        quote! { |r: f32| r + { let X = r; X } },
    ] {
        let err = refusal(input);
        assert!(
            err.contains("shadows the intrinsic coordinate"),
            "got: {err}"
        );
    }
}

/// A parameter shadowed by a `let` in an inner block is visible again
/// after the block: closing the inner scope restores what it shadowed.
#[test]
fn a_parameter_shadowed_in_an_inner_block_is_in_scope_after_it() {
    accepted(quote! { |r: f32| ({ let r = X; r }) + r });
}

/// A `let`'s initializer sees the binding it is about to shadow.
#[test]
fn a_let_initializer_sees_the_binding_it_shadows() {
    accepted(quote! { |r: f32| { let r = r * 2.0; let a = X; let a = a + r; a } });
}

#[test]
fn a_parameter_named_after_a_coordinate_is_refused() {
    let err = refusal(quote! { |X: f32| X * X }); // X shadows the intrinsic
    assert!(err.contains("shadows the intrinsic"), "got: {err}");
}

/// The binding is printed as its author spelled it; the intrinsic it
/// shadows is printed as the language names it.
#[test]
fn a_raw_spelled_coordinate_binding_names_the_intrinsic_plainly() {
    let err = refusal(quote! { || { let r#X = 1.0; X } });
    assert!(
        err.contains("`r#X` shadows the intrinsic coordinate variable `X`"),
        "got: {err}"
    );
}

/// rustc reads `r#a` and `a` as one name, and so does the language: a
/// raw spelling finds what the plain one bound, and the other way round.
#[test]
fn a_raw_spelling_is_the_name_it_spells() {
    accepted(quote! { fn r#half(x: f32) -> f32 { x } pub fn f() -> f32 { half(X) } });
    accepted(quote! { fn half(x: f32) -> f32 { x } pub fn f() -> f32 { r#half(X) } });
    accepted(quote! { |r#a: f32| a + X });
    accepted(quote! { || { let r#a = X; a } });
    accepted(quote! { const r#k: f32 = 2.0; pub fn f() -> f32 { X * k } });
}

/// Nor does a raw spelling make a second name to walk around a refusal
/// of the first: rustc would refuse the host item the macro emits for it.
#[test]
fn a_raw_spelling_is_not_another_name() {
    let cases: [(TokenStream, &str); 4] = [
        (
            quote! { pub fn f() -> f32 { X } pub fn r#f() -> f32 { Y } },
            "is defined twice in this `kernel!` block",
        ),
        (quote! { |a: f32, r#a: f32| a }, "duplicate parameter"),
        (
            quote! { const N: usize = 2; pub fn f(r#N: f32) -> f32 { X } },
            "shadows the `const",
        ),
        (
            quote! { fn h(x: f32) -> f32 { x } pub fn f(r#h: f32) -> f32 { X } },
            "shadows the `fn",
        ),
    ];
    for (input, expected) in cases {
        let err = refusal(input);
        assert!(err.contains(expected), "expected `{expected}`, got: {err}");
    }
}

#[test]
fn a_let_in_a_block_is_in_scope_for_the_rest_of_it() {
    accepted(quote! {
        |cx: f32| {
            let dx = X - cx;
            dx * dx
        }
    });
}

#[test]
fn a_parameter_declared_twice_is_refused() {
    let err = refusal(quote! { |r: f32, r: f32| X - r });
    assert!(err.contains("duplicate parameter"));
}

// ───────────────────────────── types ─────────────────────────────

/// `X.select(Y, 7.0)` would blend a number as a mask and give 5 at
/// (2, 1): a value neither arm held. It is a type error, at the receiver.
#[test]
fn a_number_used_as_a_mask_is_a_type_error() {
    let err = refusal(quote! { || X.select(Y, 7.0) });
    assert!(
        err.contains("expected `bool`, found `f32`") && err.contains("mask"),
        "got: {err}"
    );
    let err = refusal(quote! { || if X { Y } else { 7.0 } });
    assert!(
        err.contains("expected `bool`, found `f32`") && err.contains("condition"),
        "got: {err}"
    );
    let err = refusal(quote! { || X & Y });
    assert!(
        err.contains("expected `bool`, found `f32`") && err.contains("`&`"),
        "got: {err}"
    );
}

/// The reverse confusion: a mask where a number is needed.
#[test]
fn a_mask_used_as_a_number_is_a_type_error() {
    let cases: [(TokenStream, &str); 5] = [
        (quote! { || X + (X < Y) }, "arithmetic"),
        (quote! { || -(X < Y) }, "negates"),
        (quote! { || (X < Y).sqrt() }, "`.sqrt` takes"),
        (quote! { || (X < Y) < Y }, "comparison"),
        (quote! { || DX(X < Y) }, "projection"),
    ];
    for (input, expected) in cases {
        let err = refusal(input);
        assert!(
            err.contains("expected `f32`, found `bool`") && err.contains(expected),
            "expected `{expected}`, got: {err}"
        );
    }
}

/// The arms of a choice agree, whichever spelling chooses.
#[test]
fn the_arms_of_a_choice_have_one_type() {
    let err = refusal(quote! { || if X < Y { X } else { X < Y } });
    assert!(err.contains("both arms of an `if`"), "got: {err}");
    let err = refusal(quote! { || (X < Y).select(X < Y, X) });
    assert!(err.contains("both arms of `.select`"), "got: {err}");
    // A choice between masks is a mask.
    accepted(quote! { || if X < Y { X < 2.0 } else { Y < 2.0 } });
    accepted(quote! { || (X < Y).select(X < 2.0, Y < 2.0) });
}

/// A `let` annotation is checked, and a `bool` local is a `bool`.
#[test]
fn a_let_annotation_is_checked() {
    accepted(quote! { || { let m: bool = X < Y; let v: f32 = X; if m { v } else { Y } } });
    let err = refusal(quote! { || { let m: f32 = X < Y; m } });
    assert!(
        err.contains("expected `f32`, found `bool`") && err.contains("`m` is declared"),
        "got: {err}"
    );
    let err = refusal(quote! { || { let m: u32 = X; m } });
    assert!(err.contains("`f32` or a `bool`"), "got: {err}");
}

/// A parameter's declared type is honored: an `f32` is a number, and
/// there is no other type for an entry's parameters to be.
#[test]
fn a_parameter_type_is_honored() {
    let err = refusal(quote! { |n: i32| X + n });
    assert!(err.contains("`f32` or a `bool`"), "got: {err}");
    let err = refusal(quote! { |m: bool| if m { X } else { Y } });
    assert!(
        err.contains("an entry's parameter is an `f32`"),
        "got: {err}"
    );
    // A helper may take a mask.
    accepted(quote! {
        fn pick(m: bool, a: f32, b: f32) -> f32 { if m { a } else { b } }
        pub fn f() -> f32 { pick(X < Y, X, Y) }
    });
}

// ───────────────────────── items and helpers ─────────────────────────

/// What a `const` cannot be: a coordinate, a comparison, a call, a cycle,
/// a `bool`.
#[test]
fn a_const_that_is_not_constant_is_refused() {
    let cases: [(TokenStream, &str); 6] = [
        (
            quote! { const A: f32 = X; pub fn f() -> f32 { A } },
            "cannot find a `const` named `X`",
        ),
        (
            quote! { const A: f32 = 1.0 < 2.0; pub fn f() -> f32 { A } },
            "evaluated at expansion",
        ),
        (
            quote! { const A: f32 = (2.0).sqrt(); pub fn f() -> f32 { A } },
            "evaluated at expansion",
        ),
        (
            quote! { const A: f32 = B + 1.0; const B: f32 = A * 2.0; pub fn f() -> f32 { A } },
            "depends on itself",
        ),
        (
            quote! { const A: bool = 1.0; pub fn f() -> f32 { X } },
            "a `const` in a `kernel!` block is an `f32`",
        ),
        (
            quote! { const A: f32 = 1.0; const A: f32 = 2.0; pub fn f() -> f32 { X } },
            "defined twice",
        ),
    ];
    for (input, expected) in cases {
        let err = refusal(input);
        assert!(err.contains(expected), "expected `{expected}`, got: {err}");
    }
}

/// A helper is a function of its arguments: it may not read a coordinate.
#[test]
fn a_coordinate_in_a_helper_is_refused() {
    let err = refusal(quote! {
        fn shifted() -> f32 { X + 0.5 }
        pub fn f() -> f32 { shifted() }
    });
    assert!(
        err.contains("`X` in a helper") && err.contains("take the coordinate as a parameter"),
        "got: {err}"
    );
}

/// Recursion, direct, mutual and through a fold's body, is refused at
/// the call that closes the cycle.
#[test]
fn recursion_is_refused() {
    let err = refusal(quote! {
        fn f(x: f32) -> f32 { f(x) }
        pub fn g() -> f32 { f(X) }
    });
    assert!(
        err.contains("recursion is refused: the language is a DAG")
            && err.contains("`f` calls `f`"),
        "got: {err}"
    );
    let err = refusal(quote! {
        fn a(x: f32) -> f32 { b(x) + 1.0 }
        fn b(x: f32) -> f32 { c(x) * 2.0 }
        fn c(x: f32) -> f32 { a(x) }
        pub fn g() -> f32 { a(X) }
    });
    assert!(
        err.contains("recursion is refused") && err.contains("`a` calls `b` calls `c` calls `a`"),
        "got: {err}"
    );
    // Through a fold's body: a call there is a call, and lowering would
    // inline it without end.
    let err = refusal(quote! {
        fn f(x: f32) -> f32 { (0..2).map(|i| f(x)).sum() }
        pub fn g() -> f32 { f(X) }
    });
    assert!(
        err.contains("recursion is refused") && err.contains("`f` calls `f`"),
        "got: {err}"
    );
}

/// A call is checked at its arity and its parameters' types; an entry
/// is not callable; an unknown function is not captured.
#[test]
fn a_call_is_checked() {
    let cases: [(TokenStream, &str); 5] = [
        (
            quote! { fn h(x: f32) -> f32 { x } pub fn f() -> f32 { h(X, Y) } },
            "`h` takes 1 argument, but 2 were supplied",
        ),
        (
            quote! { fn h(x: f32) -> f32 { x } pub fn f() -> f32 { h(X < Y) } },
            "parameter 1 of `h` is a `f32`",
        ),
        (
            quote! { pub fn e() -> f32 { X } pub fn f() -> f32 { e() } },
            "`e` is an entry, not a helper",
        ),
        (
            quote! { pub fn f() -> f32 { outside(X) } },
            "cannot find a function `outside`",
        ),
        (
            quote! { fn h(x: f32) -> f32 { x } pub fn f() -> f32 { h } },
            "`h` is a function, not a value",
        ),
    ];
    for (input, expected) in cases {
        let err = refusal(input);
        assert!(err.contains(expected), "expected `{expected}`, got: {err}");
    }
}

/// A `fn` returns what it declares.
#[test]
fn a_return_type_is_checked() {
    let err = refusal(quote! { pub fn f() -> bool { X } });
    assert!(
        err.contains("expected `bool`, found `f32`") && err.contains("`f` declares"),
        "got: {err}"
    );
    accepted(quote! { pub fn f() -> bool { X < Y } });
}

/// Nothing shadows an item.
#[test]
fn an_item_is_not_shadowed() {
    let err = refusal(quote! {
        const R: f32 = 1.0;
        pub fn f(R: f32) -> f32 { R }
    });
    assert!(err.contains("shadows the `const R`"), "got: {err}");
    let err = refusal(quote! {
        fn h(x: f32) -> f32 { x }
        pub fn f() -> f32 { let h = X; h }
    });
    assert!(err.contains("shadows the `fn h`"), "got: {err}");
    let err = refusal(quote! {
        fn X(x: f32) -> f32 { x }
        pub fn f() -> f32 { X(Y) }
    });
    assert!(err.contains("intrinsic coordinate"), "got: {err}");
}

/// A block with no entry would expand to nothing.
#[test]
fn a_block_with_no_entry_is_refused() {
    let err = refusal(quote! { fn h(x: f32) -> f32 { x } });
    assert!(err.contains("no entry emits nothing"), "got: {err}");
}

/// A block of helpers that recurse is refused for the recursion, not for
/// having no entry: the cycle is the diagnostic that says what to fix.
#[test]
fn recursion_is_refused_before_a_missing_entry() {
    for input in [
        quote! { fn h(x: f32) -> f32 { h(x) } },
        quote! {
            fn a(x: f32) -> f32 { b(x) }
            fn b(x: f32) -> f32 { a(x) }
        },
    ] {
        let err = refusal(input);
        assert!(err.contains("recursion is refused"), "got: {err}");
    }
}

// ─────────────────────── folds and the binder type ───────────────────────

/// Each fold types as its monoid's terms: a sum, a product, a minimum
/// and a maximum of `f32`s are an `f32`, and `any` and `all` of `bool`s
/// are a `bool`.
#[test]
fn every_fold_types_as_its_terms() {
    accepted(quote! { || (0..4).map(|i| X * (i as f32)).sum() });
    accepted(quote! { || (0..4).map(|i| X + i as f32).product() });
    accepted(quote! { || (0..4).map(|i| X - i as f32).fold(f32::INFINITY, f32::min) });
    accepted(quote! { || (0..4).map(|i| X - i as f32).fold(f32::NEG_INFINITY, f32::max) });
    accepted(quote! { pub fn f() -> bool { (0..4).any(|i| X < i as f32) } });
    accepted(quote! { pub fn f() -> bool { (0..4).all(|i| X < i as f32) } });
    let cases: [(TokenStream, &str); 3] = [
        (
            quote! { || (0..4).map(|i| X < i as f32).sum() },
            "a sum, a product, a minimum or a maximum combines `f32`s",
        ),
        (
            quote! { || (0..4).any(|i| X + i as f32) },
            "`any` and `all` combine `bool`s",
        ),
        (
            quote! { || (0..4).any(|i| X < i as f32) + 1.0 },
            "arithmetic takes `f32`s",
        ),
    ];
    for (input, expected) in cases {
        let err = refusal(input);
        assert!(
            err.contains("mismatched types") && err.contains(expected),
            "expected `{expected}`, got: {err}"
        );
    }
}

/// A bound must be a count: a known `usize` or an unchanged count parameter.
/// Other types and expressions are refused.
#[test]
fn a_range_bound_that_is_not_a_count_is_refused() {
    let cases: [(TokenStream, &str); 8] = [
        (
            quote! { |n: f32| (0..n).map(|i| i as f32).sum() },
            "`n` is a `f32`",
        ),
        (
            quote! { || (0..X).map(|i| i as f32).sum() },
            "`X` is a `f32`",
        ),
        (
            quote! { || { let n = 4.0; (0..n).map(|i| i as f32).sum() } },
            "`n` is a `f32`",
        ),
        (
            quote! { const R: f32 = 4.0; pub fn f() -> f32 { (0..R).map(|i| i as f32).sum() } },
            "`R` is a `f32`",
        ),
        (
            quote! { || (0..4.0).map(|i| i as f32).sum() },
            "expected `usize`, found a float literal",
        ),
        (
            quote! { || (0..X.floor()).map(|i| i as f32).sum() },
            "a range's bound is a count",
        ),
        (
            quote! { || (0..(1 < 2)).map(|i| i as f32).sum() },
            "a range's bound is a count",
        ),
        (
            quote! { || (0..-1).map(|i| i as f32).sum() },
            "cannot apply unary operator `-` to type `usize`",
        ),
    ];
    for (input, expected) in cases {
        let err = refusal(input);
        assert!(err.contains(expected), "expected `{expected}`, got: {err}");
    }
}

/// A user fold is rectangular: an enclosing fold's index is no bound, at
/// either end, alone or under arithmetic, while an inner fold over a constant
/// range may still read the outer index in its body (§4 of
/// docs/language/2026-10-25-kernel-staging-and-rectangular-folds.md).
#[test]
fn an_enclosing_fold_index_is_no_range_bound() {
    for input in [
        quote! { || (0..4).map(|i| (0..i).map(|j| j as f32).sum::<f32>()).sum() },
        quote! { || (0..4).map(|i| (i..4).map(|j| j as f32).sum::<f32>()).sum() },
        quote! { || (0..4).map(|i| (0..i + 1).map(|j| j as f32).sum::<f32>()).sum() },
        quote! { || (0..4).map(|i| (0..i).map(|i| i as f32).sum::<f32>()).sum() },
        quote! {
            pub fn f(n: usize) -> f32 {
                (0..n).map(|i| (0..i).map(|j| j as f32).sum::<f32>()).sum()
            }
        },
    ] {
        let err = refusal(input);
        assert!(
            err.contains("fold index used as a range bound")
                && err.contains("constant or an unchanged uniform count"),
            "got: {err}"
        );
    }
    accepted(quote! {
        || (0..4).map(|i| (0..3).map(|j| (i as f32) + (j as f32)).sum::<f32>()).sum()
    });
    accepted(quote! {
        pub fn f(n: usize) -> f32 {
            (0..4).map(|i| (0..n).map(|j| (i as f32) * (j as f32)).sum::<f32>()).sum()
        }
    });
}

/// A range runs forwards; a reversed one is refused, saying it would be
/// the empty fold. The empty range itself is a fold like any other.
#[test]
fn a_reversed_range_is_refused_and_an_empty_one_is_not() {
    let err = refusal(quote! { || (5..3).map(|i| i as f32).sum() });
    assert!(
        err.contains("`5..3` runs backwards") && err.contains("identity"),
        "got: {err}"
    );
    accepted(quote! { || (3..3).map(|i| i as f32).sum() });
}

/// A bound is held in 64 bits here; the IR's index is an `f32` lane,
/// exact to 2²⁴, and its fold ends are `u32`. A bound past the lane is
/// refused — past `u32` too, naming the plan's A5 — and 2²⁴ itself is
/// the last bound accepted.
#[test]
fn a_bound_past_the_exact_index_is_refused() {
    for past in [
        quote! { || (0..16777217).map(|i| i as f32).sum() },
        quote! { || (16777217..16777217).map(|i| i as f32).sum() },
        quote! { || (0..4294967296).map(|i| i as f32).sum() },
    ] {
        let err = refusal(past);
        assert!(err.contains("past 16777216 (2^24)"), "got: {err}");
    }
    accepted(quote! { || (16777215..16777216).map(|i| i as f32).sum() });
}

/// Folds nest as deep as collapse leaves binders, and no deeper: one more is
/// refused, not a panic.
#[test]
fn folds_nest_as_deep_as_collapse_has_binders() {
    let fold = |body: TokenStream| quote! { (0..1).map(|i| #body).sum::<f32>() };
    let nest = |depth: usize| (0..depth).fold(quote! { X }, |body, _| fold(body));
    let (deepest, one_deeper) = (
        nest(KERNEL_BINDER_CAPACITY),
        nest(KERNEL_BINDER_CAPACITY + 1),
    );
    accepted(quote! { || #deepest });
    let err = refusal(quote! { || #one_deeper });
    assert!(err.contains("collapse reserves 3"), "got: {err}");
}

/// A range's bounds are built from `usize` consts, at the largest value a
/// `usize` holds too.
#[test]
fn a_range_from_usize_consts_is_accepted() {
    accepted(quote! {
        const LO: usize = 7 / 2;
        const N: usize = LO * 4 - 1;
        pub const MAX: usize = 18446744073709551615;
        pub fn f() -> f32 { (LO..LO + N).map(|i| i as f32).sum() + (N as f32) }
    });
}

/// A `usize` const is checked as rustc checks one: an overflow, a
/// division by zero, a negation, a float and an `f32` const are errors.
#[test]
fn a_usize_const_is_checked_as_rustc_checks_it() {
    let cases: [(TokenStream, &str); 6] = [
        (
            quote! { const N: usize = 3 - 4; pub fn f() -> f32 { X } },
            "attempt to compute `3_usize - 4_usize`, which would overflow",
        ),
        (
            quote! { const N: usize = 18446744073709551615 + 1; pub fn f() -> f32 { X } },
            "which would overflow",
        ),
        (
            quote! { const N: usize = 4 / 0; pub fn f() -> f32 { X } },
            "attempt to divide `4_usize` by zero",
        ),
        (
            quote! { const N: usize = -1; pub fn f() -> f32 { X } },
            "cannot apply unary operator `-` to type `usize`",
        ),
        (
            quote! { const N: usize = 2.0; pub fn f() -> f32 { X } },
            "expected `usize`, found a float literal",
        ),
        (
            quote! { const A: f32 = 1.0; const N: usize = A; pub fn f() -> f32 { X } },
            "`A` is an `f32` const, where a `usize` is expected",
        ),
    ];
    for (input, expected) in cases {
        let err = refusal(input);
        assert!(err.contains(expected), "expected `{expected}`, got: {err}");
    }
    let err = refusal(quote! {
        const N: usize = 18446744073709551616;
        pub fn f() -> f32 { X }
    });
    assert!(err.contains("out of range for `usize`"), "got: {err}");
}

/// An `f32` const names a `usize` const only through `as f32`.
#[test]
fn an_f32_const_cannot_name_a_usize_const_without_as() {
    let err = refusal(quote! {
        const N: usize = 4;
        const A: f32 = N;
        pub fn f() -> f32 { X * A }
    });
    assert!(err.contains("`N` is a `usize`"), "got: {err}");
}

/// A fold's index is a `usize`, and a `usize` is not a value: arithmetic
/// on one, a comparison of one, and one where an `f32` is expected are
/// type errors at the name, which say `i as f32`. So is a `usize` const.
#[test]
fn a_usize_where_a_value_is_expected_is_a_type_error() {
    let cases: [(TokenStream, &str); 9] = [
        (quote! { || (0..4).map(|i| X * i).sum() }, "`i`"),
        (quote! { || (0..4).map(|i| (i + 1) as f32).sum() }, "`i`"),
        (quote! { || (0..4).any(|i| i < 2) }, "`i`"),
        (quote! { || (0..4).map(|i| i).sum() }, "`i`"),
        (quote! { || (0..4).map(|i| i.sqrt()).sum() }, "`i`"),
        (quote! { || (0..4).map(|i| DX(i)).sum() }, "`i`"),
        (
            quote! { || (0..4).map(|i| { let j = i; j as f32 }).sum() },
            "`i`",
        ),
        (
            quote! {
                fn h(x: f32) -> f32 { x }
                pub fn f() -> f32 { (0..4).map(|i| h(i)).sum() }
            },
            "`i`",
        ),
        (
            quote! { const N: usize = 4; pub fn f() -> f32 { X * N } },
            "`N`",
        ),
    ];
    for (input, name) in cases {
        let err = refusal(input);
        assert!(
            err.contains(&format!("mismatched types: {name} is a `usize`"))
                && err.contains("as f32"),
            "expected {name} as a `usize`, got: {err}"
        );
    }
}

/// `as f32` converts a `usize`, named, and nothing else. (A target other
/// than `f32` is the parser's refusal.)
#[test]
fn as_f32_converts_a_usize_and_nothing_else() {
    accepted(quote! { || (0..4).map(|i| (i) as f32).sum() });
    accepted(quote! { const N: usize = 4; pub fn f() -> f32 { X * (N as f32) } });
    let cases: [(TokenStream, &str); 4] = [
        (quote! { || X as f32 }, "this expression's type is `f32`"),
        (quote! { || (X < Y) as f32 }, "a mask"),
        (
            quote! { || (0..4).map(|i| (i as f32) as f32).sum() },
            "this expression's type is `f32`",
        ),
        (
            quote! { const A: f32 = 1.0; pub fn f() -> f32 { A as f32 } },
            "this expression's type is `f32`",
        ),
    ];
    for (input, expected) in cases {
        let err = refusal(input);
        assert!(err.contains(expected), "expected `{expected}`, got: {err}");
    }
}

/// A `usize` is a count, not a value: no parameter, `let` or `fn`
/// return is declared one.
#[test]
fn a_usize_is_not_a_declared_value_type() {
    let cases: [(TokenStream, &str); 3] = [
        (
            quote! { |n: usize| X },
            "a count parameter `n: usize` belongs to an item entry",
        ),
        (
            quote! { pub fn f() -> usize { 4 } },
            "a kernel `fn` returns an `f32` or a `bool`",
        ),
        (
            quote! { || { let n: usize = 4; X } },
            "a `let` in a kernel body is an `f32` or a `bool`",
        ),
    ];
    for (input, expected) in cases {
        let err = refusal(input);
        assert!(err.contains(expected), "expected `{expected}`, got: {err}");
    }
    accepted(quote! { pub fn f(n: usize) -> f32 { X * (n as f32) } });
}

/// A fold's index is scoped as a closure's parameter is: it shadows a
/// binding, an enclosing index among them, and nothing past the fold's
/// body sees it. It shadows no item.
#[test]
fn a_fold_index_is_scoped_to_the_folds_body() {
    accepted(quote! { |r: f32| (0..4).map(|r| r as f32).sum() + r });
    accepted(quote! { || (0..2).map(|i| (0..3).map(|i| i as f32).sum::<f32>()).sum() });
    let err = refusal(quote! { || (0..4).map(|i| i as f32).sum() + i as f32 });
    assert!(err.contains("cannot find `i`"), "got: {err}");
    let err = refusal(quote! { || (0..4).map(|X| X).sum() });
    assert!(
        err.contains("a fold's index `X` shadows the intrinsic"),
        "got: {err}"
    );
    let err = refusal(quote! {
        const N: usize = 4;
        pub fn f() -> f32 { (0..N).map(|N| X).sum() }
    });
    assert!(err.contains("shadows the `const N`"), "got: {err}");
}

// ───────────────────────────── records ─────────────────────────────

/// A block's record, for the cases below: `Row { x0, s }`.
fn with_row(body: TokenStream) -> TokenStream {
    quote! {
        pub struct Row { pub x0: f32, pub s: f32 }
        #body
    }
}

/// A record is a parameter's type, of an entry or a helper; a field is
/// read by name; `let q = p;` aliases, annotated or not; and a record is
/// passed on by name, through parentheses too.
#[test]
fn a_record_is_named_aliased_passed_and_read_by_field() {
    accepted(with_row(quote! {
        fn scaled(p: Row, x: f32) -> f32 { p.s * (x - p.x0) }
        pub fn f(p: Row, r: f32) -> f32 {
            let q = p;
            let t: Row = (q);
            scaled(t, X) + (q).x0 * r
        }
    }));
}

/// A record where a value is expected — in arithmetic, a comparison, a
/// method, a fold, a statement — or computed, chosen or returned, is
/// refused, naming Phase D (D7).
#[test]
fn a_record_used_as_a_value_or_computed_is_refused_naming_phase_d() {
    let cases: [(TokenStream, &str); 9] = [
        (
            quote! { pub fn f(p: Row) -> f32 { p + 1.0 } },
            "the record `Row` where a value is expected",
        ),
        (
            quote! { pub fn f(p: Row, q: Row) -> bool { p == q } },
            "the record `Row` where a value is expected",
        ),
        (
            quote! { pub fn f(p: Row) -> f32 { p.sqrt() } },
            "the record `Row` where a value is expected",
        ),
        (
            quote! { pub fn f(p: Row) -> f32 { (0..2).map(|i| p).sum() } },
            "the record `Row` where a value is expected",
        ),
        (
            quote! { pub fn f(p: Row) -> f32 { p; X } },
            "the record `Row` where a value is expected",
        ),
        (
            quote! { pub fn f(p: Row, q: Row) -> f32 { (if X < Y { p } else { q }).x0 } },
            "an `if` choosing between records",
        ),
        (
            quote! { pub fn f(p: Row) -> f32 { ({ p }).x0 } },
            "a record computed by an expression",
        ),
        (
            quote! {
                fn h(p: Row) -> f32 { p.x0 }
                pub fn f(p: Row) -> f32 { h(p.clone()) }
            },
            "a record computed by an expression",
        ),
        (
            quote! { pub fn f(p: Row) -> Row { p } },
            "a `fn` returning the record `Row`",
        ),
    ];
    for (body, expected) in cases {
        let err = refusal(with_row(body));
        assert!(
            err.contains(expected) && err.contains("D7"),
            "expected `{expected}` and `D7`, got: {err}"
        );
    }
}

/// A field is one of the record's, and only a record has fields; a
/// record argument is the helper's record.
#[test]
fn a_field_is_one_of_the_records_and_a_record_is_its_own_type() {
    let err = refusal(with_row(quote! { pub fn f(p: Row) -> f32 { p.y0 } }));
    assert!(
        err.contains("no field `y0` on the record `Row`") && err.contains("`x0`, `s`"),
        "got: {err}"
    );
    let err = refusal(with_row(quote! { pub fn f(r: f32) -> f32 { r.x0 } }));
    assert!(err.contains("a field of a `f32`"), "got: {err}");
    let err = refusal(quote! {
        pub struct Row { pub x0: f32 }
        pub struct Col { pub x0: f32 }
        fn h(p: Row) -> f32 { p.x0 }
        pub fn f(c: Col) -> f32 { h(c) }
    });
    assert!(err.contains("expected `Row`, found `Col`"), "got: {err}");
    let err = refusal(with_row(quote! {
        fn h(p: Row) -> f32 { p.x0 }
        pub fn f() -> f32 { h(X) }
    }));
    assert!(err.contains("expected `Row`, found `f32`"), "got: {err}");
}

/// A record is named `f32` fields: a field of another record, an array,
/// a tuple or any other type is refused at the field, naming §1.3; a
/// field declared twice is refused too.
#[test]
fn a_record_field_that_is_not_an_f32_is_refused_naming_its_section() {
    let cases: [(TokenStream, &str); 5] = [
        (
            quote! { pub struct Pair { a: Row } },
            "`a` is the record `Row`",
        ),
        (quote! { pub struct Pair { a: [f32; 2] } }, "an array"),
        (quote! { pub struct Pair { a: (f32, f32) } }, "a tuple"),
        (quote! { pub struct Pair { a: bool } }, "a `bool`"),
        (
            quote! { pub struct Pair { a: T } },
            "a type the language does not have",
        ),
    ];
    for (record, expected) in cases {
        let err = refusal(with_row(quote! { #record pub fn f() -> f32 { X } }));
        assert!(
            err.contains(expected) && err.contains("§1.3"),
            "expected `{expected}` and `§1.3`, got: {err}"
        );
    }
    let err = refusal(quote! { struct Pair { a: f32, a: f32 } pub fn f() -> f32 { X } });
    assert!(err.contains("declared twice"), "got: {err}");
}

/// An entry's `Args` record takes a host type name, which no record and
/// no other entry's `Args` may take.
#[test]
fn an_args_record_name_taken_twice_is_refused() {
    let err = refusal(quote! {
        pub struct CircleArgs { pub r: f32 }
        pub fn circle(r: f32) -> f32 { X - r }
    });
    assert!(
        err.contains("`circle`'s `Args` record, `CircleArgs`, is also the record"),
        "got: {err}"
    );
    let err = refusal(quote! {
        pub fn a_b(r: f32) -> f32 { X - r }
        pub fn a__b(r: f32) -> f32 { X + r }
    });
    assert!(err.contains("the `Args` record of `a_b`"), "got: {err}");
    // An entry with no parameters has no `Args` record to collide.
    accepted(quote! {
        pub struct PlainArgs { pub r: f32 }
        pub fn plain() -> f32 { X }
    });
}

// ───────────────────────────── tuple lets ─────────────────────────────

/// A tuple `let` types every expression where the statement stands,
/// before any of its names binds — so `let (a, b) = (b, a);` swaps, as
/// Rust's does — and then binds each name to its expression's type: a
/// mask, a value, a record aliased.
#[test]
fn a_tuple_let_types_each_expression_before_binding_any_name() {
    accepted(quote! { || { let (a, b) = (X, Y); let (a, b) = (b, a); a - b } });
    accepted(quote! { || { let (m, w) = (X < Y, X); if m { w } else { 0.0 } } });
    accepted(with_row(quote! {
        pub fn f(p: Row, r: f32) -> f32 { let (q, s) = (p, r); q.x0 * s }
    }));
    let cases: [(TokenStream, &str); 3] = [
        (
            quote! { || { let (b, c) = (X < Y, b); X } },
            "cannot find `b`",
        ),
        (
            quote! { || { let (a, b): (f32, f32) = (X, X < Y); a } },
            "`b` is declared as a `f32`",
        ),
        (
            quote! { || { let (X, b) = (Y, Y); b } },
            "shadows the intrinsic coordinate",
        ),
    ];
    for (input, expected) in cases {
        let err = refusal(input);
        assert!(err.contains(expected), "expected `{expected}`, got: {err}");
    }
}

// ───────────────────────────── kernels ─────────────────────────────

/// A body over a kernel-typed parameter, and helpers to pass one to.
fn with_kernel(body: TokenStream) -> TokenStream {
    quote! {
        fn at(k: impl Fn(f32, f32) -> f32, x: f32) -> f32 { k(x, x) }
        fn plain(x: f32) -> f32 { x }
        #body
    }
}

/// A kernel-typed parameter is `impl Fn(f32, f32) -> f32`, an entry's
/// or a helper's; a body applies it at two `f32`s, as often as it likes
/// and inside a fold, and passes it on by name, once, after applying it.
#[test]
fn a_kernel_is_applied_and_passed_on() {
    accepted(with_kernel(quote! {
        pub fn f(k: impl Fn(f32, f32) -> f32, r: f32) -> f32 {
            k(X, Y) * r + k(Y + 1.0, X) + (0..3).map(|i| k(X, i as f32)).sum::<f32>()
        }
        pub fn g(k: impl Fn(f32, f32) -> f32) -> f32 { at(k, X) }
        pub fn h(k: impl Fn(f32, f32) -> f32) -> f32 { k(X, Y) + at(k, Y) }
    }));
}

/// Every other spelling of a function is refused, naming the one the
/// language has, what differs, and the phase.
#[test]
fn a_function_spelled_otherwise_is_refused() {
    let cases: [(TokenStream, &str); 8] = [
        (quote! { k: impl Fn(f32) -> f32 }, "two coordinates"),
        (quote! { k: impl Fn(f32, f32) -> bool }, "`-> f32`"),
        (quote! { k: impl Fn(f32, f32) }, "`-> f32`"),
        (
            quote! { k: impl FnMut(f32, f32) -> f32 },
            "mutates or consumes",
        ),
        (quote! { k: impl Fn(f32, f32) -> f32 + Copy }, "one bound"),
        (quote! { k: &impl Fn(f32, f32) -> f32 }, "not by reference"),
        (quote! { k: &dyn Fn(f32, f32) -> f32 }, "not by reference"),
        (quote! { k: fn(f32, f32) -> f32 }, "not a function pointer"),
    ];
    for (param, expected) in cases {
        let err = refusal(quote! { pub fn f(#param) -> f32 { X } });
        assert!(
            err.contains(expected) && err.contains("impl Fn(f32, f32) -> f32"),
            "`{param}`: expected `{expected}`, got: {err}"
        );
        assert!(err.contains("Phase D-a"), "names the plan's phase: {err}");
    }
}

/// A kernel is not a value, wherever a value is expected: each refusal
/// says so and names the phase.
#[test]
fn a_kernel_is_not_a_value() {
    let cases: [TokenStream; 9] = [
        quote! { pub fn f(k: impl Fn(f32, f32) -> f32) -> f32 { k + 1.0 } },
        quote! { pub fn f(k: impl Fn(f32, f32) -> f32) -> bool { k < X } },
        quote! { pub fn f(k: impl Fn(f32, f32) -> f32) -> f32 { k.sqrt() } },
        quote! { pub fn f(k: impl Fn(f32, f32) -> f32) -> f32 { k } },
        quote! { pub fn f(k: impl Fn(f32, f32) -> f32) -> f32 { let j = k; X } },
        quote! { pub fn f(k: impl Fn(f32, f32) -> f32) -> f32 { k; X } },
        quote! { pub fn f(k: impl Fn(f32, f32) -> f32) -> f32 { plain(k) } },
        quote! { pub fn f(k: impl Fn(f32, f32) -> f32) -> f32 { (0..2).map(|i| k).sum() } },
        quote! { pub fn f(k: impl Fn(f32, f32) -> f32) -> f32 { if X < Y { k } else { k } } },
    ];
    for input in cases {
        let text = input.to_string();
        let err = refusal(with_kernel(input));
        assert!(
            err.contains("a kernel where a value is expected") && err.contains("Phase D-a"),
            "`{text}`: got {err}"
        );
    }
}

/// A kernel is a parameter and nothing else: as a record's field, a
/// return type, a `let`'s annotation or the operand of `as f32` it is
/// refused, naming the phase.
#[test]
fn a_kernel_is_a_parameter_and_nothing_else() {
    let cases: [(TokenStream, &str); 4] = [
        (
            quote! { pub struct R { k: impl Fn(f32, f32) -> f32 } pub fn f() -> f32 { X } },
            "never a field",
        ),
        (
            quote! { pub fn f() -> impl Fn(f32, f32) -> f32 { X } },
            "a `fn` returning a function",
        ),
        (
            quote! {
                pub fn f(k: impl Fn(f32, f32) -> f32) -> f32 {
                    let j: impl Fn(f32, f32) -> f32 = k;
                    X
                }
            },
            "a `let` of a function",
        ),
        (
            quote! { pub fn f(k: impl Fn(f32, f32) -> f32) -> f32 { k as f32 } },
            "the operand of `as f32`",
        ),
    ];
    for (input, expected) in cases {
        let err = refusal(input);
        assert!(err.contains(expected), "expected `{expected}`, got: {err}");
        assert!(err.contains("Phase D-a"), "names the plan's phase: {err}");
    }
}

/// An application is a kernel's, at two `f32`s; a name that is not a
/// kernel is not applied.
#[test]
fn an_application_is_a_kernels_at_two_values() {
    let cases: [(TokenStream, &str); 4] = [
        (
            quote! { pub fn f(k: impl Fn(f32, f32) -> f32) -> f32 { k(X) } },
            "applied at the two coordinates",
        ),
        (
            quote! { pub fn f(k: impl Fn(f32, f32) -> f32) -> f32 { k(X, Y, X) } },
            "applied at the two coordinates",
        ),
        (
            quote! { pub fn f(k: impl Fn(f32, f32) -> f32) -> f32 { k(X < Y, Y) } },
            "applied at two `f32` coordinates",
        ),
        (
            quote! { pub fn f(r: f32) -> f32 { r(X, Y) } },
            "only a kernel is applied",
        ),
    ];
    for (input, expected) in cases {
        let err = refusal(input);
        assert!(err.contains(expected), "expected `{expected}`, got: {err}");
    }
}

/// A kernel is passed on by name, to a kernel-typed parameter.
#[test]
fn a_kernel_is_passed_on_by_name_to_a_kernel_typed_parameter() {
    let cases: [(TokenStream, &str); 2] = [
        (
            quote! { pub fn f(k: impl Fn(f32, f32) -> f32) -> f32 { at({ k }, X) } },
            "passed by name",
        ),
        (
            quote! { pub fn f(k: impl Fn(f32, f32) -> f32) -> f32 { at(X, X) } },
            "expected `impl Fn(f32, f32) -> f32`, found `f32`",
        ),
    ];
    for (input, expected) in cases {
        let err = refusal(with_kernel(input));
        assert!(err.contains(expected), "expected `{expected}`, got: {err}");
    }
}

/// rustc moves an `impl Fn` it passes, and a block is Rust: a kernel is
/// passed on at most once, used not at all after, and never from inside
/// a fold's body, a closure `.map` calls again (E0382, E0507).
#[test]
fn a_kernel_passed_on_moves_as_rustc_moves_it() {
    let cases: [(TokenStream, &str); 4] = [
        (
            quote! { pub fn f(k: impl Fn(f32, f32) -> f32) -> f32 { at(k, X) + at(k, Y) } },
            "E0382",
        ),
        (
            quote! { pub fn f(k: impl Fn(f32, f32) -> f32) -> f32 { at(k, X) + k(X, Y) } },
            "E0382",
        ),
        (
            quote! { pub fn f(k: impl Fn(f32, f32) -> f32) -> f32 { k(at(k, X), Y) } },
            "E0382",
        ),
        (
            quote! {
                pub fn f(k: impl Fn(f32, f32) -> f32) -> f32 {
                    (0..2).map(|i| at(k, i as f32)).sum()
                }
            },
            "E0507",
        ),
    ];
    for (input, expected) in cases {
        let err = refusal(with_kernel(input));
        assert!(err.contains(expected), "expected `{expected}`, got: {err}");
    }
}

/// An `if`'s arms are two paths, as rustc's move check follows them: a
/// kernel passed on in one arm is the other's to use or pass on, and
/// after the `if` it is moved if either arm moved it (E0382).
#[test]
fn a_kernel_passed_on_in_one_arm_is_the_others() {
    accepted(with_kernel(quote! {
        pub fn f(k: impl Fn(f32, f32) -> f32) -> f32 {
            if X < Y { at(k, X) } else { k(X, Y) }
        }
        pub fn g(k: impl Fn(f32, f32) -> f32) -> f32 {
            if X < Y { at(k, X) } else { at(k, Y) }
        }
        pub fn h(k: impl Fn(f32, f32) -> f32) -> f32 {
            if X < Y { k(X, Y) } else if X < 0.0 { at(k, Y) } else { at(k, X) }
        }
    }));
    let maybe_moved = [
        quote! {
            pub fn f(k: impl Fn(f32, f32) -> f32) -> f32 {
                (if X < Y { at(k, X) } else { 0.0 }) + k(X, Y)
            }
        },
        quote! {
            pub fn f(k: impl Fn(f32, f32) -> f32) -> f32 {
                (if X < Y { 0.0 } else { at(k, X) }) + k(X, Y)
            }
        },
        quote! {
            pub fn f(k: impl Fn(f32, f32) -> f32) -> f32 {
                (if X < Y { 0.0 } else if X < 0.0 { at(k, X) } else { 1.0 }) + at(k, Y)
            }
        },
    ];
    for input in maybe_moved {
        let err = refusal(with_kernel(input));
        assert!(err.contains("E0382"), "got: {err}");
    }
}

/// A parameter, `let` or index named after a projection is refused: a
/// call is resolved by its name, and `DX(x, y)` must not mean a kernel
/// to one stage and a derivative to another.
#[test]
fn a_binding_named_after_a_projection_is_refused() {
    for input in [
        quote! { pub fn f(DX: impl Fn(f32, f32) -> f32) -> f32 { DX(X, Y) } },
        quote! { pub fn f(V: f32) -> f32 { V } },
        quote! { || { let DY = X; DY } },
        quote! { || (0..2).map(|DXX| DXX as f32).sum() },
    ] {
        let err = refusal(input);
        assert!(err.contains("is named after the projection"), "got: {err}");
    }
}

/// A call to a projection takes the one argument a projection takes:
/// `DX(x, y)` is no application of a kernel.
#[test]
fn a_projection_takes_one_argument() {
    for input in [quote! { || DX(X, Y) }, quote! { || V() }] {
        let err = refusal(input);
        assert!(err.contains("takes 1 argument, but"), "got: {err}");
    }
}
