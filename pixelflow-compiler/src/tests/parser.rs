//! The parser's contract (`parser/mod.rs`): the `KernelDef` a token stream
//! spells, or the refusal at the token that spells nothing.

use crate::ast::{
    BinaryOp, ClosureEntry, Expr, FoldExpr, KernelDef, LetStmt, Literal, LiteralExpr, Reduction,
    Role, Stmt, UnaryOp,
};
use crate::parser::parse;
use proc_macro2::TokenStream;
use quote::quote;
use syn::Type;

/// The closure sugar's one entry.
fn entry(def: &KernelDef) -> &ClosureEntry {
    let KernelDef::Closure(entry) = def else {
        panic!("the closure form parses to a closure entry");
    };
    entry
}

/// The body `|| X #op X` parses to.
fn binary(op: &TokenStream) -> Expr {
    let def = parse(quote!(|| X #op X)).unwrap_or_else(|e| panic!("`{op}`: {e}"));
    entry(&def).body.clone()
}

#[test]
fn every_supported_binary_operator_parses_to_its_own_variant() {
    let cases: &[(TokenStream, BinaryOp)] = &[
        (quote!(+), BinaryOp::Add),
        (quote!(-), BinaryOp::Sub),
        (quote!(*), BinaryOp::Mul),
        (quote!(/), BinaryOp::Div),
        (quote!(<), BinaryOp::Lt),
        (quote!(<=), BinaryOp::Le),
        (quote!(>), BinaryOp::Gt),
        (quote!(>=), BinaryOp::Ge),
        (quote!(==), BinaryOp::Eq),
        (quote!(!=), BinaryOp::Ne),
        (quote!(&), BinaryOp::BitAnd),
        (quote!(|), BinaryOp::BitOr),
    ];
    for (op, expected) in cases {
        let Expr::Binary(binary) = binary(op) else {
            panic!("`X {op} X` is a binary expression");
        };
        assert_eq!(binary.op, *expected, "{op}");
    }
}

/// `%` has no IR op and `+=` assigns: neither is a kernel operator, and each
/// is refused at the operator, naming it.
#[test]
fn an_unsupported_binary_operator_is_refused_at_the_operator() {
    let cases: &[(TokenStream, &str)] = &[
        (quote!(+=), "`+=` assigns"),
        (quote!(%), "`%` has no IR op"),
        (quote!(^), "unsupported binary operator `^`"),
        (quote!(&&), "unsupported binary operator `&&`"),
    ];
    for (op, claim) in cases {
        let refusal = parse(quote!(|| X #op X))
            .err()
            .unwrap_or_else(|| panic!("`X {op} X` is refused"))
            .to_string();
        assert!(refusal.contains(claim), "`{op}`: {refusal}");
    }
}

/// `!` has no IR op, so it is not a kernel operator either.
#[test]
fn negation_is_the_one_unary_operator() {
    let def = parse(quote!(|| -X)).unwrap();
    let Expr::Unary(unary) = &entry(&def).body else {
        panic!("`-X` is a unary expression");
    };
    assert_eq!(unary.op, UnaryOp::Neg);
    let refusal = parse(quote!(|| !X)).unwrap_err().to_string();
    assert!(refusal.contains("`!` has no IR op"), "{refusal}");
}

/// An entry's `Args` record is its name in UpperCamelCase, then `Args`;
/// a raw identifier is named by what it spells.
#[test]
fn an_entrys_args_record_is_its_name_in_upper_camel_case() {
    for (entry, args) in [
        ("circle", "CircleArgs"),
        ("shifted_radius2", "ShiftedRadius2Args"),
        ("a__b_", "ABArgs"),
        ("r#type", "TypeArgs"),
        ("_1", "_1Args"),
        ("__3", "_3Args"),
    ] {
        let input: TokenStream = format!("pub fn {entry}() -> f32 {{ X }}")
            .parse()
            .expect("an items block");
        let def = parse(input).unwrap_or_else(|e| panic!("{entry}: {e}"));
        let [item] = def.fns() else {
            panic!("{entry}: one fn item");
        };
        assert_eq!(item.args_record().to_string(), args, "{entry}");
    }
}

#[test]
fn the_closure_sugar_is_one_entry_with_an_inferred_return_type() {
    let input = quote! { |r: f32| X * X + Y * Y - r };
    let kernel = parse(input).unwrap();
    let entry = entry(&kernel);
    assert_eq!(entry.params.len(), 1);
    assert_eq!(entry.params[0].name.to_string(), "r");
}

#[test]
fn a_block_body_parses_its_lets_and_final_expression() {
    let input = quote! {
        |cx: f32, cy: f32| {
            let dx = X - cx;
            let dy = Y - cy;
            dx * dx + dy * dy
        }
    };
    let kernel = parse(input).unwrap();
    match &entry(&kernel).body {
        Expr::Block(block) => {
            assert_eq!(block.stmts.len(), 2); // two let statements
            assert!(block.expr.is_some()); // final expression
        }
        _ => panic!("expected block expression"),
    }
}

/// A parameter's declared type is kept for `sema` to check; the parser
/// does not judge it.
#[test]
fn a_parameter_keeps_the_type_it_was_declared_with() {
    let input = quote! { |cx: f32, m: bool| X * cx };
    let kernel = parse(input).unwrap();
    let entry = entry(&kernel);
    assert_eq!(entry.params.len(), 2);
    assert_eq!(entry.params[0].name.to_string(), "cx");
    assert_eq!(entry.params[1].name.to_string(), "m");
    for (param, want) in entry.params.iter().zip(["f32", "bool"]) {
        let syn::Type::Path(path) = &*param.ty else {
            panic!("expected a path type for {}", param.name);
        };
        assert_eq!(path.path.segments[0].ident.to_string(), want);
    }
}

/// A multi-segment path (qself-free but `len() != 1`) is refused as the
/// path it is, not silently truncated to an `Ident` of its first segment;
/// so is a call through one.
#[test]
fn a_path_from_outside_the_block_is_refused_by_name() {
    let err = refusal(quote! { || std::f32::consts::PI });
    assert!(
        err.contains("a path in a kernel body: `std :: f32 :: consts :: PI`")
            && err.contains("not captured"),
        "got: {err}"
    );
    let err = refusal(quote! { || f32::sqrt(X) });
    assert!(err.contains("a qualified call"), "got: {err}");
}

/// A value with parts built or taken apart in a body, anywhere but a
/// tuple `let`, is refused where it is written: a record's pattern
/// naming §1.3, since a record is read by field; a tuple as a value and
/// a tuple's field naming Phase D (D7), since a tuple is taken apart
/// where it is written and so is never one; a record literal naming
/// Phase D (D7).
#[test]
fn a_record_pattern_a_tuple_value_and_a_record_literal_are_refused_naming_their_phase() {
    let cases: [(TokenStream, &str, &str); 4] = [
        (
            quote! { || { let Row { x0, .. } = p; x0 } },
            "a pattern",
            "§1.3",
        ),
        (quote! { || (X, Y) }, "a tuple in a kernel body", "D7"),
        (quote! { |p: f32| p.0 }, "a tuple's field", "D7"),
        (quote! { || Row { x0: X }.x0 }, "a record literal", "D7"),
    ];
    for (input, expected, phase) in cases {
        let err = refusal(input);
        assert!(
            err.contains(expected) && err.contains(phase),
            "expected `{expected}` and `{phase}`, got: {err}"
        );
    }
}

/// `p.x0` parses as a field of whatever `p` is; which record, and
/// whether it has the field, are `sema`'s questions.
#[test]
fn a_named_field_parses_as_a_field_read() {
    let def = parse(quote! { |p: f32| (p).x0 + 1.0 }).expect("parses");
    let Expr::Binary(sum) = &entry(&def).body else {
        panic!("expected the sum, got {:?}", entry(&def).body);
    };
    let Expr::Field(field) = &*sum.lhs else {
        panic!("expected a field, got {:?}", sum.lhs);
    };
    assert_eq!(field.member.to_string(), "x0");
    assert_eq!(
        field.base.named().map(ToString::to_string),
        Some("p".into())
    );
}

/// A range anywhere but a fold's names the fold's spellings.
#[test]
fn a_range_is_refused_naming_the_fold() {
    for input in [
        quote! { || X + (0.0..1.0) },
        quote! { || DX(0.0..1.0) },
        quote! { || { let r = 0..4; X } },
    ] {
        let err = refusal(input);
        assert!(
            err.contains("a range in a kernel body") && err.contains("(a..b).map(|i| e).sum()"),
            "got: {err}"
        );
    }
}

/// Every other Rust expression is refused at the token, named, with
/// nothing passed through.
#[test]
fn any_other_expression_is_refused_at_parse() {
    for other in [
        quote! { || unsafe { X } },
        quote! { || match X { _ => Y } },
        quote! { || &X },
        quote! { || X? },
    ] {
        let err = refusal(other);
        assert!(
            err.contains("unsupported expression in a kernel body: `"),
            "got: {err}"
        );
    }
}

/// A `let`'s annotation is kept beside its name, for `sema` to check.
#[test]
fn a_let_keeps_its_name_and_its_type_annotation() {
    let input = quote! {
        || {
            let dx: f32 = X;
            dx
        }
    };
    let kernel = parse(input).unwrap();
    match &entry(&kernel).body {
        Expr::Block(block) => {
            assert_eq!(block.stmts.len(), 1);
            match &block.stmts[0] {
                Stmt::Let(lets) => {
                    let let_stmt = &lets[0];
                    assert_eq!(let_stmt.name.to_string(), "dx");
                    let ty = let_stmt.ty.as_ref().expect("expected a type annotation");
                    if let Type::Path(type_path) = ty {
                        assert_eq!(type_path.path.segments[0].ident.to_string(), "f32");
                    } else {
                        panic!("expected path type");
                    }
                }
                _ => panic!("expected let statement"),
            }
        }
        _ => panic!("expected block expression"),
    }
}

#[test]
fn a_terminal_statement_with_a_trailing_semicolon_is_not_the_blocks_final_expression() {
    let input = quote! { || { X; } };
    let kernel = parse(input).unwrap();
    match &entry(&kernel).body {
        Expr::Block(block) => {
            assert_eq!(
                block.stmts.len(),
                1,
                "the semicolon-terminated X is a statement"
            );
            assert!(
                block.expr.is_none(),
                "a block ending in `expr;` has no final expression"
            );
        }
        _ => panic!("expected block expression"),
    }
}

/// What the literal body `|| <lit>` denotes as parsed, or the parser's
/// refusal.
fn parsed_literal(lit: TokenStream) -> Result<LiteralExpr, String> {
    let def = parse(quote! { || #lit }).map_err(|e| e.to_string())?;
    match &entry(&def).body {
        Expr::Literal(literal) => Ok(literal.clone()),
        other => panic!("expected a literal, got {other:?}"),
    }
}

/// The value the literal body `|| <lit>` denotes where a value is
/// expected, or the refusal: the parser's, or the value position's.
fn literal(lit: TokenStream) -> Result<f32, String> {
    parsed_literal(lit)?.f32_value().map_err(|e| e.to_string())
}

/// The parser's refusal of `input`, as text.
fn refusal(input: TokenStream) -> String {
    parse(input)
        .expect_err("the input is refused at parse")
        .to_string()
}

/// `let … else` is refused, not parsed with its `else` branch dropped.
#[test]
fn let_else_is_refused_rather_than_dropped() {
    let input = quote! {
        || {
            let a = X else { return; };
            a
        }
    };
    let err = refusal(input);
    assert!(err.contains("`let ... else`"), "got: {err}");
}

/// A float literal rounds once, straight to `f32`, as rustc rounds it.
/// This literal is just above the midpoint between `1.0` and `1 + 2⁻²³`,
/// by far less than an `f64` ulp: through `f64` it lands exactly on the
/// midpoint and the cast ties to even, `1.0`.
#[test]
#[allow(clippy::excessive_precision)] // The digits past f32's precision are the witness.
fn a_float_literal_rounds_once_to_f32() {
    let once = 1.00000005960464477539062500001_f32;
    let twice = 1.00000005960464477539062500001_f64 as f32;
    assert_eq!(once.to_bits(), 0x3f80_0001);
    assert_eq!(twice.to_bits(), 0x3f80_0000);

    let got = literal(quote!(1.00000005960464477539062500001)).expect("in range");
    assert_eq!(got.to_bits(), once.to_bits());
    assert_eq!(literal(quote!(0.1)), Ok(0.1_f32));
    assert_eq!(literal(quote!(2.5f32)), Ok(2.5));
    assert_eq!(literal(quote!(1_000.25)), Ok(1000.25));
    // The suffix makes an integer a float literal to rustc: rounded once.
    assert_eq!(literal(quote!(16777217f32)), Ok(16_777_216.0));
}

/// A `let` binds a plain name; `mut`, `ref` and `@` are refused, not
/// silently dropped.
#[test]
fn a_let_binds_a_plain_name() {
    for decorated in [
        quote! { || { let mut a = X; a } },
        quote! { || { let ref a = X; a } },
        quote! { || { let a @ 1.0..=2.0 = X; a } },
    ] {
        let err = refusal(decorated);
        assert!(err.contains("plain name"), "got: {err}");
    }
}

#[test]
fn a_float_literal_past_f32s_range_is_refused() {
    let err = literal(quote!(1e39)).expect_err("rounds to infinity");
    assert!(err.contains("out of range for `f32`"), "got: {err}");
    // The largest finite `f32` is in range.
    assert_eq!(literal(quote!(3.4028235e38)), Ok(f32::MAX));
}

/// An integer literal is exact or refused where a value is expected.
/// Every integer up to 2²⁴ is an `f32`; 2²⁴ + 1 is the first that is not.
#[test]
fn an_integer_literal_is_exact_or_refused() {
    assert_eq!(literal(quote!(0)), Ok(0.0));
    assert_eq!(literal(quote!(16777216)), Ok(16_777_216.0));
    // One significant bit, however large: exactly an `f32`.
    assert_eq!(literal(quote!(1099511627776)), Ok(1_099_511_627_776.0));
    assert_eq!(literal(quote!(0x10)), Ok(16.0));

    for inexact in [quote!(16777217), quote!(4294967295)] {
        let err = literal(inexact).expect_err("an f32 cannot hold it");
        assert!(
            err.contains("not exactly representable as an `f32`"),
            "got: {err}"
        );
    }
    // Past `u128` no type holds it, and the parser says so.
    let err = literal(quote!(1000000000000000000000000000000000000000000)).expect_err("past u128");
    assert!(err.contains("integer literal is too large"), "got: {err}");
}

/// An integer is kept exactly as written, since its type is its
/// position's: `16777217` is no `f32`, and is a `usize`. A float is never
/// a count.
#[test]
fn an_integer_literal_is_kept_exact_for_its_position() {
    let int = parsed_literal(quote!(16777217)).expect("parses");
    assert_eq!(int.value, Literal::Int(16_777_217));
    assert_eq!(int.usize_value().expect("a count"), 16_777_217);
    assert!(int.f32_value().is_err());

    let past_usize = parsed_literal(quote!(18446744073709551616)).expect("parses");
    let err = past_usize.usize_value().expect_err("2^64").to_string();
    assert!(err.contains("out of range for `usize`"), "got: {err}");

    let float = parsed_literal(quote!(4.0)).expect("parses");
    let err = float.usize_value().expect_err("a float").to_string();
    assert!(err.contains("expected `usize`"), "got: {err}");
}

/// A suffix naming another type is refused rather than silently dropped.
#[test]
fn a_literal_suffixed_with_a_type_other_than_f32_is_refused() {
    for foreign in [quote!(2.5f64), quote!(3u8), quote!(3i32)] {
        let err = literal(foreign).expect_err("a kernel value is an f32");
        assert!(err.contains("literal in a kernel body"), "got: {err}");
    }
}

/// Runs `work` on a thread with `stack` bytes, so that a test of the bound
/// cannot take the test process down with it.
fn on_a_stack<T: Send + 'static>(stack: usize, work: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::Builder::new()
        .stack_size(stack)
        .spawn(work)
        .unwrap()
        .join()
        .unwrap()
}

/// Every chain shape the parser converts level by level while `syn`
/// parses it flat: `terms` deep exactly, in a closure body.
fn chains(terms: usize) -> [(&'static str, String); 4] {
    [
        ("a sum", vec!["X"; terms].join(" + ")),
        ("a method chain", format!("X{}", ".abs()".repeat(terms - 1))),
        ("a field chain", format!("X{}", ".x".repeat(terms - 1))),
        (
            "an else-if chain",
            format!("{}{{ 1.0 }}", "if X < 1.0 { 1.0 } else ".repeat(terms - 1)),
        ),
    ]
}

fn parse_body(body: &str) -> Result<(), String> {
    let input: TokenStream = format!("|| {body}").parse().unwrap();
    parse(input).map(|_| ()).map_err(|e| e.to_string())
}

/// A macro that generates a body writes a long left-leaning sum, and the
/// parser recurses once per term, so what a term costs in stack is how
/// long a body can be. The bound's whole claim is that a body at it
/// parses on rustc's own 8 MiB thread stack, so this runs on exactly
/// that one: a frame that grows past the bound's margin (1.7x) aborts
/// here rather than in a user's build.
#[test]
fn a_chain_at_the_bound_parses_within_a_compiler_threads_stack() {
    const RUSTC_THREAD_STACK: usize = 8 * 1024 * 1024;
    for (shape, body) in chains(1024 - 1) {
        let parsed = on_a_stack(RUSTC_THREAD_STACK, move || parse_body(&body));
        assert_eq!(parsed, Ok(()), "{shape}");
    }
}

/// Past the bound the program is refused where it is, not crashed, and
/// the refusal costs the same however far past: it runs on rustc's own
/// stack, 20 times the bound out (`syn`'s own recursive parse and drop of
/// a chain overflow at about 25 times in a dev build, which this crate
/// cannot hold), because a refusal that walks the remaining chain (a span
/// of the whole subtree did) overflows there while the same test on a
/// larger stack passes.
#[test]
fn a_chain_past_the_bound_is_refused_not_a_crash() {
    const RUSTC_THREAD_STACK: usize = 8 * 1024 * 1024;
    for terms in [1024 + 1, 1024 * 20] {
        for (shape, body) in chains(terms) {
            let refused = on_a_stack(RUSTC_THREAD_STACK, move || parse_body(&body));
            let message = refused.expect_err(shape);
            assert!(
                message.contains("nested deeper than 1024"),
                "{shape}, {terms} terms: {message}"
            );
        }
    }
}

#[test]
fn a_literal_that_is_not_a_number_is_refused() {
    for other in [quote!(true), quote!("text"), quote!('c')] {
        let err = literal(other).expect_err("not a number");
        assert!(err.contains("only numeric literals"), "got: {err}");
    }
}

// ───────────────────────── the items form ─────────────────────────

/// A block of items: `const`s, helpers and entries, told apart by `pub`.
#[test]
fn an_items_block_parses_consts_helpers_and_entries() {
    let input = quote! {
        const R: f32 = 1.0;
        pub const TWO_R: f32 = R * 2.0;
        /// A helper: private.
        fn sq(x: f32) -> f32 { x * x }
        pub fn circle(cx: f32) -> f32 { sq(X - cx) - R }
    };
    let def = parse(input).unwrap();
    assert!(matches!(def, KernelDef::Items { .. }));
    assert_eq!(def.consts().len(), 2);
    assert!(matches!(def.consts()[0].vis, syn::Visibility::Inherited));
    assert!(matches!(def.consts()[1].vis, syn::Visibility::Public(_)));
    assert_eq!(def.fns().len(), 2);
    assert_eq!(def.fns()[0].role(), Role::Helper);
    assert_eq!(def.fns()[0].attrs.len(), 1, "the doc comment is kept");
    assert_eq!(def.fns()[1].role(), Role::Entry);
    assert_eq!(def.fns()[1].params.len(), 1);
}

/// `if c { a } else { b }` and an `else if` chain parse as the choice.
#[test]
fn an_if_with_its_else_parses_as_the_choice() {
    let input = quote! { || if X < Y { X } else if X < 2.0 { 2.0 } else { Y } };
    let def = parse(input).unwrap();
    let Expr::If(outer) = &entry(&def).body else {
        panic!("expected an if, got {:?}", entry(&def).body);
    };
    assert!(matches!(*outer.cond, Expr::Binary(_)));
    assert!(
        matches!(*outer.else_branch, Expr::If(_)),
        "`else if` is an `if` in the else position"
    );
}

/// An `if` without an `else` has no value to be, so it is refused where
/// it stands.
#[test]
fn an_if_without_an_else_is_refused() {
    let err = refusal(quote! { || if X < Y { X } });
    assert!(err.contains("without an `else`"), "got: {err}");
    let err = refusal(quote! { || if let a = X { a } else { Y } });
    assert!(err.contains("`if let`"), "got: {err}");
}

/// The constructs a DAG has no meaning for are refused with a span, not
/// passed through for a later stage to refuse without one.
#[test]
fn loops_assignment_return_and_closures_are_refused() {
    let cases: [(TokenStream, &str); 7] = [
        (quote! { || loop { X } }, "a loop"),
        (quote! { || { while X < Y { X }; X } }, "a loop"),
        (quote! { || { for i in 0..3 { X }; X } }, "a loop"),
        (quote! { || { let a = X; a = Y; a } }, "assignment"),
        (quote! { || { let a = X; a += Y; a } }, "assigns"),
        (quote! { || { return X; } }, "`return`"),
        (quote! { || DX(|x: f32| x) }, "a closure"),
    ];
    for (input, expected) in cases {
        let err = refusal(input);
        assert!(err.contains(expected), "expected `{expected}`, got: {err}");
    }
}

/// What the parser would otherwise drop is refused: a turbofish on a call
/// (`convert_path` refuses one on a name) and a block's label.
#[test]
fn a_turbofish_on_a_call_and_a_block_label_are_refused_rather_than_dropped() {
    let cases: [(TokenStream, &str); 2] = [
        (
            quote! { fn h(a: f32) -> f32 { a } pub fn f() -> f32 { h::<u8>(X) } },
            "type arguments on a call",
        ),
        (quote! { || 'l: { X } }, "a block label"),
    ];
    for (input, expected) in cases {
        let err = refusal(input);
        assert!(err.contains(expected), "expected `{expected}`, got: {err}");
    }
}

/// An empty `where` and a parameterless fold closure print no tokens, so a
/// refusal spanned at the clause or at the parameter list would carry the
/// whole invocation's span. These are refused at the `where` keyword and
/// at the closure's opening bar. The span itself is not observable here:
/// proc-macro2 outside a running macro keeps no locations, and the crate
/// does not take its `span-locations` feature for one test. It holds by
/// construction, `new_spanned` over a token the input always has, and
/// this pins that the inputs are refused at the parser with their text.
#[test]
fn a_refusal_over_nothing_is_spanned_at_a_token_that_exists() {
    let cases: [(TokenStream, &str); 4] = [
        (
            quote! { pub fn e() -> f32 where { X } },
            "a `where` clause on",
        ),
        (
            quote! { pub struct R where { x: f32 } pub fn f(r: R) -> f32 { X } },
            "a `where` clause on",
        ),
        (
            quote! { fn h() -> f32 where { X } pub fn f() -> f32 { h() } },
            "a `where` clause on",
        ),
        (
            quote! { || (0..3).map(|| X).sum::<f32>() },
            "a fold's closure takes one parameter",
        ),
    ];
    for (input, expected) in cases {
        let shown = input.to_string();
        let err = refusal(input);
        assert!(
            err.contains(expected),
            "expected `{expected}`, got: {err} for {shown}"
        );
    }
}

/// `%` and `!` have no IR op; each is refused by name at the operator.
#[test]
fn rem_and_not_are_refused_by_name() {
    let err = refusal(quote! { || X % Y });
    assert!(err.contains("`%` has no IR op"), "got: {err}");
    let err = refusal(quote! { || !(X < Y) });
    assert!(err.contains("`!` has no IR op"), "got: {err}");
}

/// A record parses with its visibility, its doc comments and its fields
/// in declaration order, each field's own visibility and docs kept.
#[test]
fn a_record_parses_with_its_fields_and_docs() {
    let def = parse(quote! {
        /// One arc piece.
        pub struct Row {
            /// Where it starts.
            pub x0: f32,
            sigma: f32,
        }
        pub fn f(p: Row) -> f32 { p.x0 }
    })
    .expect("parses");
    let [row] = def.records() else {
        panic!("one record, got {:?}", def.records());
    };
    assert_eq!(row.name.to_string(), "Row");
    assert!(matches!(row.vis, syn::Visibility::Public(_)));
    assert_eq!(row.attrs.len(), 1, "the record's doc comment is kept");
    let names: Vec<String> = row.fields.iter().map(|f| f.name.to_string()).collect();
    assert_eq!(names, ["x0", "sigma"]);
    assert_eq!(
        row.fields[0].attrs.len(),
        1,
        "a field's doc comment is kept"
    );
    assert!(matches!(row.fields[1].vis, syn::Visibility::Inherited));
}

/// A record is named `f32` fields: a tuple struct, a unit struct and a
/// generic record are refused where they are written, naming §1.3; any
/// other item names what the block holds.
#[test]
fn a_record_that_is_not_named_fields_is_refused_naming_its_section() {
    let cases: [(TokenStream, &str); 3] = [
        (quote! { struct Row(f32, f32); }, "a tuple struct"),
        (quote! { struct Row; }, "a unit struct"),
        (quote! { struct Row<T> { x: T } }, "a generic record"),
    ];
    for (input, expected) in cases {
        let err = refusal(quote! { #input pub fn f() -> f32 { X } });
        assert!(
            err.contains(expected) && err.contains("§1.3"),
            "expected `{expected}` and `§1.3`, got: {err}"
        );
    }
    let err = refusal(quote! {
        enum Row { A }
        pub fn f() -> f32 { X }
    });
    assert!(
        err.contains("a `enum`") && err.contains("records (`struct`s of `f32` fields)"),
        "got: {err}"
    );
    for attr in [quote!(#[repr(packed)]), quote!(#[cfg(any())])] {
        let err = refusal(quote! {
            #attr
            struct Row { x: f32 }
            pub fn f() -> f32 { X }
        });
        assert!(
            err.contains("a `repr` or a `cfg` on a record"),
            "got: {err}"
        );
        let err = refusal(quote! {
            struct Row { #attr x: f32 }
            pub fn f() -> f32 { X }
        });
        assert!(
            err.contains("a `repr` or a `cfg` on a record"),
            "got: {err}"
        );
    }
}

/// A record's other attributes, and its fields', are kept for its host
/// twin: rustc reads them there.
#[test]
fn a_records_attributes_are_kept() {
    let def = parse(quote! {
        /// A row.
        #[allow(dead_code)]
        #[cfg_attr(any(), derive(Eq))]
        pub struct Row { #[allow(unused)] pub x0: f32 }
        pub fn f() -> f32 { X }
    })
    .expect("parses");
    let [row] = def.records() else {
        panic!("one record, got {}", def.records().len());
    };
    let paths: Vec<String> = row
        .attrs
        .iter()
        .map(|attr| attr.path().get_ident().expect("a plain path").to_string())
        .collect();
    assert_eq!(paths, ["doc", "allow", "cfg_attr"]);
    assert_eq!(row.fields[0].attrs.len(), 1, "the field's `allow`");
}

// ───────────────────────────── folds ─────────────────────────────

/// The fold the body `|| <expr>` parses to.
fn fold(input: TokenStream) -> FoldExpr {
    let def = parse(quote! { || #input }).expect("the fold parses");
    match &entry(&def).body {
        Expr::Fold(fold) => fold.clone(),
        other => panic!("expected a fold, got {other:?}"),
    }
}

/// Each spelling of a fold names its monoid; the closure's parameter is
/// the binder and its body the fold's body.
#[test]
fn every_fold_spelling_parses_to_its_reduction() {
    let cases: [(TokenStream, Reduction); 8] = [
        (quote! { (0..4).map(|i| X).sum() }, Reduction::Sum),
        (quote! { (0..4).map(|i| X).sum::<f32>() }, Reduction::Sum),
        (quote! { (0..4).map(|i| X).product() }, Reduction::Product),
        (
            quote! { (0..4).map(|i| X).product::<f32>() },
            Reduction::Product,
        ),
        (
            quote! { (0..4).map(|i| X).fold(f32::INFINITY, f32::min) },
            Reduction::Min,
        ),
        (
            quote! { (0..4).map(|i| X).fold(f32::NEG_INFINITY, f32::max) },
            Reduction::Max,
        ),
        (quote! { (0..4).any(|i| X < Y) }, Reduction::Any),
        (quote! { (0..4).all(|i| X < Y) }, Reduction::All),
    ];
    for (input, want) in cases {
        let parsed = fold(input);
        assert_eq!(parsed.reduction, want);
        assert_eq!(parsed.binder.to_string(), "i");
    }
    let parsed = fold(quote! { (LO..LO + N).map(|k| X).sum() });
    assert!(matches!(*parsed.range.lo, Expr::Ident(_)));
    assert!(matches!(*parsed.range.hi, Expr::Binary(_)));
}

/// A fold spelled any other way is refused at parse, naming what the
/// language accepts: a `.fold` that is not a min or a max, a method that
/// ends no fold, a range that is inclusive or open, and a closure that is
/// not `|i| body`.
#[test]
fn a_fold_spelled_any_other_way_is_refused() {
    let cases: [(TokenStream, &str); 19] = [
        (quote! { || (0..4).map(|i| X) }, "an iterator, not a value"),
        (quote! { || (0..4).map(|i| X).min() }, "does not end a fold"),
        (
            quote! { || (0..4).map(|i| X).fold(0.0, |a, b| a + b) },
            "names no monoid",
        ),
        (
            quote! { || (0..4).map(|i| X).fold(f32::INFINITY, f32::max) },
            "names no monoid",
        ),
        (
            quote! { || (0..4).map(|i| X).fold(std::f32::INFINITY, f32::min) },
            "names no monoid",
        ),
        (
            quote! { || (0..4).map(|i| X).fold(f32::INFINITY) },
            "names no monoid",
        ),
        (quote! { || (0..4).sum() }, "`.sum` on a range"),
        (
            quote! { || (0..4).rev().map(|i| X).sum() },
            "`.rev` on a range",
        ),
        (quote! { || (0..=4).map(|i| X).sum() }, "an inclusive range"),
        (quote! { || (0..).map(|i| X).sum() }, "both of its bounds"),
        (
            quote! { || (0..4).map(|i: usize| X).sum() },
            "`usize`, always",
        ),
        (quote! { || (0..4).map(|i, j| X).sum() }, "one parameter"),
        (quote! { || (0..4).map(|_| X).sum() }, "a plain name"),
        (quote! { || (0..4).map(move |i| X).sum() }, "unqualified"),
        (
            quote! { || (0..4).map(|i| -> f32 { X }).sum() },
            "declares no return type",
        ),
        (
            quote! { || (0..4).map(|i| X).sum::<f64>() },
            "over anything but `f32`s",
        ),
        (
            quote! { || (0..4).map(|i| X).sum(1.0) },
            "takes no arguments",
        ),
        (quote! { || (0..4).map(X).sum() }, "the closure `|i| body`"),
        (
            quote! { || (0..4).any::<f32>(|i| X < Y) },
            "takes no type arguments",
        ),
    ];
    for (input, expected) in cases {
        let err = refusal(input);
        assert!(err.contains(expected), "expected `{expected}`, got: {err}");
    }
    // The refusal of a `.fold` names the two it accepts.
    let err = refusal(quote! { || (0..4).map(|i| X).fold(0.0, f32::max) });
    assert!(
        err.contains("f32::INFINITY, f32::min") && err.contains("f32::NEG_INFINITY, f32::max"),
        "got: {err}"
    );
}

/// A closure is the body of a fold and nothing else: anywhere else it is
/// refused, naming the fold's spellings, and the phase that brings a
/// function as an argument.
#[test]
fn a_closure_outside_a_fold_is_refused() {
    for input in [
        quote! { || X.max(|i| i) },
        quote! { || { let f = |x: f32| x; X } },
        quote! { || (0..4).map(|i| X).sum() + (|j| Y) },
        quote! { || DX(|u| u) },
    ] {
        let err = refusal(input);
        assert!(
            err.contains("a closure in a kernel body")
                && err.contains("the body of a fold, and nothing else")
                && err.contains("(a..b).map(|i| e).sum()")
                && err.contains("Phase D"),
            "got: {err}"
        );
    }
}

/// The language has no collection types (§1.3, §1.6), and every spelling
/// of one is refused where it is written, saying so: a parameter typed
/// as an array or a slice, by value or by reference; an array literal;
/// indexing and slicing; `.into_iter()` and `.iter()`; and a fold's
/// spelling over anything but a range — where the closure's own refusal
/// would say less.
#[test]
fn a_collection_spelled_any_way_is_refused_naming_its_sections() {
    let cases: [(TokenStream, &str); 14] = [
        (quote! { |v: [f32; 2]| X }, "a collection-typed parameter"),
        (
            quote! { pub struct Row { pub x0: f32 } pub fn f(pieces: [Row; 4]) -> f32 { X } },
            "a collection-typed parameter",
        ),
        (
            quote! { fn h(v: &[f32]) -> f32 { 1.0 } pub fn f() -> f32 { X } },
            "a collection-typed parameter",
        ),
        (quote! { |v: [f32]| X }, "a collection-typed parameter"),
        (quote! { || [X, Y] }, "an array"),
        (quote! { || { let v = [X; 2]; X } }, "an array"),
        (quote! { || X[0] }, "indexing"),
        (quote! { |v: f32| v[1..3] }, "indexing"),
        (
            quote! { |v: f32| v.into_iter().map(|e| e * X).sum() },
            "`.into_iter()`",
        ),
        (quote! { |v: f32| v.iter().any(|e| e < X) }, "`.iter()`"),
        (
            quote! { |pieces: f32| pieces.map(|p| p * X).sum() },
            "`.map` over `pieces`, which is not a range,",
        ),
        (
            quote! { |pieces: f32| pieces.any(|p| p < X) },
            "`.any` over `pieces`, which is not a range,",
        ),
        (
            quote! { || X.all(|p| p < Y) },
            "`.all` over `X`, which is not a range,",
        ),
        (
            quote! { || (X + Y).map(|e| e).sum() },
            "which is not a range,",
        ),
    ];
    for (input, expected) in cases {
        let err = refusal(input);
        assert!(
            err.contains(expected)
                && err.contains("the language has no collection types")
                && err.contains("§1.3, §1.6"),
            "expected `{expected}`, got: {err}"
        );
    }
}

/// A collection reaching `kernel!` through a macro's `$t:ty` fragment, a
/// type inside an invisible group, is a collection as it is spelled
/// bare.
#[test]
fn a_collection_inside_a_macros_type_fragment_is_refused_as_one() {
    let fragment = proc_macro2::Group::new(proc_macro2::Delimiter::None, quote! { [f32; 2] });
    let err = refusal(quote! { |v: #fragment| X });
    assert!(
        err.contains("a collection-typed parameter"),
        "expected the collection's refusal, got: {err}"
    );
}

// ───────────────────────────── tuple lets ─────────────────────────────

/// The one tuple `let` of `|| { input; X }`, flattened.
fn tuple_let(input: TokenStream) -> Vec<LetStmt> {
    let def = parse(quote! { || { #input; X } }).expect("parses");
    let Expr::Block(block) = &entry(&def).body else {
        panic!("a block");
    };
    let [Stmt::Let(lets)] = block.stmts.as_slice() else {
        panic!("one tuple `let`, got {:?}", block.stmts);
    };
    lets.clone()
}

/// `let (a, b) = (e1, e2);` is flattened: each name bound to its own
/// expression, with its own annotation when the tuple type gives one,
/// in one statement that binds them all at once (D7's front half).
#[test]
fn a_tuple_let_binds_each_name_to_its_expression() {
    let lets = tuple_let(quote! { let (a, b) = (X, Y + 1.0) });
    let names: Vec<String> = lets.iter().map(|l| l.name.to_string()).collect();
    assert_eq!(names, ["a", "b"]);
    assert!(matches!(lets[0].init, Expr::Ident(_)));
    assert!(matches!(lets[1].init, Expr::Binary(_)));
    assert!(lets.iter().all(|l| l.ty.is_none()));

    let lets = tuple_let(quote! { let (m, w): (bool, f32) = ((X < Y), X) });
    let types: Vec<String> = lets
        .iter()
        .map(|l| {
            let ty = l.ty.as_ref().expect("annotated");
            quote!(#ty).to_string()
        })
        .collect();
    assert_eq!(types, ["bool", "f32"]);

    assert_eq!(
        tuple_let(quote! { let (a,) = ((X,)) }).len(),
        1,
        "a tuple of one"
    );
}

/// A tuple is written in one place, the tuple `let` that takes it
/// apart; anywhere else it is a tuple value, refused naming Phase D
/// (D7), and so is a `let` whose value is not a tuple of expressions.
#[test]
fn a_tuple_anywhere_but_a_destructuring_let_is_refused() {
    let cases: [(TokenStream, &str); 6] = [
        (quote! { || (X, Y) }, "a tuple in a kernel body"),
        (
            quote! { || { let t = (X, Y); X } },
            "a tuple in a kernel body",
        ),
        (quote! { || X.max((X, Y)) }, "a tuple in a kernel body"),
        (
            quote! { || { let (a, b) = (X, Y); (b, a) } },
            "a tuple in a kernel body",
        ),
        (
            quote! { || { let (a, b) = t; a } },
            "the value of a tuple `let` is a tuple of expressions",
        ),
        (
            quote! { || { let (a, b) = if X < Y { (X, Y) } else { (Y, X) }; a } },
            "the value of a tuple `let` is a tuple of expressions",
        ),
    ];
    for (input, expected) in cases {
        let err = refusal(input);
        assert!(
            err.contains(expected) && err.contains("Phase D") && err.contains("D7"),
            "expected `{expected}` naming Phase D, got: {err}"
        );
    }
}

/// A tuple `let` binds plain names, as many as its tuple has elements,
/// each once: a mismatched arity (rustc's E0308), a pattern within the
/// pattern (a tuple value, Phase D), `_`, `..`, a name bound twice
/// (E0416), no names at all, and an annotation that is not a tuple of
/// as many types are each refused where they are written.
#[test]
fn a_tuple_let_of_another_shape_is_refused() {
    let cases: [(TokenStream, &str); 9] = [
        (
            quote! { let (a, b) = (X, Y, X) },
            "expected a tuple with 2 elements, found one with 3 elements",
        ),
        (
            quote! { let (a, b, c) = (X, Y) },
            "expected a tuple with 3 elements, found one with 2 elements",
        ),
        (
            quote! { let ((a, b), c) = ((X, Y), X) },
            "a pattern within a tuple `let`",
        ),
        (
            quote! { let ((a), b) = (X, Y) },
            "a pattern within a tuple `let`",
        ),
        (quote! { let (a, _) = (X, Y) }, "`_` in a tuple `let`"),
        (
            quote! { let (a, ..) = (X, Y) },
            "a pattern in a kernel `let`",
        ),
        (
            quote! { let (a, a) = (X, Y) },
            "identifier `a` is bound more than once in the same pattern",
        ),
        (quote! { let () = () }, "binds nothing"),
        (
            quote! { let (a, b): (f32, f32, f32) = (X, Y) },
            "the annotation of a tuple `let` of 2 names is a tuple of 2 types",
        ),
    ];
    for (input, expected) in cases {
        let err = refusal(quote! { || { #input; X } });
        assert!(err.contains(expected), "expected `{expected}`, got: {err}");
    }
    let err = refusal(quote! { || { let ((a, b), c) = ((X, Y), X); a } });
    assert!(err.contains("Phase D (D7"), "got: {err}");
}

/// Type arguments name a type a method would be generic over, and no
/// method in a kernel body is generic; they are refused, not dropped.
#[test]
fn type_arguments_on_a_method_are_refused() {
    let err = refusal(quote! { || X.sqrt::<f32>() });
    assert!(err.contains("type arguments on a method"), "got: {err}");
}

/// `i as f32` parses as a cast of its operand; which operands a cast
/// means anything of is `sema`'s question, since it needs their types.
#[test]
fn a_cast_parses_with_its_operand() {
    let def = parse(quote! { || (i) as f32 }).expect("parses");
    let Expr::Cast(cast) = &entry(&def).body else {
        panic!("expected a cast, got {:?}", entry(&def).body);
    };
    assert_eq!(cast.named().map(ToString::to_string), Some("i".to_string()));
    let def = parse(quote! { || (X + 1.0) as f32 }).expect("parses");
    let Expr::Cast(cast) = &entry(&def).body else {
        panic!("expected a cast");
    };
    assert!(cast.named().is_none());
}

/// The one conversion is to `f32`; a cast to any other type is refused
/// at the type, so the AST never holds one.
#[test]
fn a_cast_to_anything_but_f32_is_refused() {
    for input in [
        quote! { || (0..4).map(|i| i as u32).sum() },
        quote! { || X as f64 },
        quote! { || X as usize },
        quote! { || X as std::primitive::f32 },
    ] {
        let err = refusal(input);
        assert!(
            err.contains("`as` converts a `usize` to an `f32`, and to nothing else"),
            "got: {err}"
        );
    }
}

/// A `fn` signature carries nothing the language cannot honor: an
/// neither an entry nor a helper takes generics; counts are entry parameters.
#[test]
fn a_fn_signature_is_plain() {
    let err = refusal(quote! {
        fn h<const N: usize>(x: f32) -> f32 { x }
        pub fn f() -> f32 { X }
    });
    assert!(
        err.contains("a `const` generic") && err.contains("n: usize"),
        "got: {err}"
    );
    let cases: [(TokenStream, &str); 7] = [
        (
            quote! { pub fn f<T>() -> f32 { X } },
            "nothing in a kernel is generic",
        ),
        (
            quote! { pub fn f<'a>() -> f32 { X } },
            "nothing in a kernel is generic",
        ),
        (
            quote! { pub fn f<const N: u32>() -> f32 { X } },
            "a `const` generic",
        ),
        (
            quote! { pub fn f<const N: usize = 4>() -> f32 { X } },
            "a `const` generic",
        ),
        (quote! { pub fn f() { X } }, "declares no return type"),
        (quote! { pub fn f(mut x: f32) -> f32 { x } }, "plain name"),
        (quote! { const fn f() -> f32 { X } }, "`const fn`"),
    ];
    for (input, expected) in cases {
        let err = refusal(input);
        assert!(err.contains(expected), "expected `{expected}`, got: {err}");
    }
}

/// A kernel-typed parameter is parsed, as any typed parameter is, and
/// `sema` decides whether its type is the language's one function type.
/// The closure form takes none: rustc refuses `impl Trait` in a
/// closure's parameters (E0562).
#[test]
fn a_kernel_typed_parameter_is_an_entrys_and_not_a_closures() {
    let def = parse(quote! {
        pub fn f(k: impl Fn(f32, f32) -> f32, r: f32) -> f32 { k(X, Y) * r }
    })
    .expect("an entry's kernel-typed parameter parses");
    assert!(def.fns()[0].params[0].is_kernel());
    assert!(!def.fns()[0].params[1].is_kernel());
    // However it is spelled: bare, in parentheses, through a macro's
    // `$t:ty` (an invisible group), or as some other function.
    let grouped = proc_macro2::Group::new(
        proc_macro2::Delimiter::None,
        quote! { impl Fn(f32, f32) -> f32 },
    );
    for closure in [
        quote! { |k: impl Fn(f32, f32) -> f32| k(X, Y) },
        quote! { |k: (impl Fn(f32, f32) -> f32)| k(X, Y) },
        quote! { |k: #grouped| k(X, Y) },
        quote! { |k: &dyn Fn(f32, f32) -> f32| k(X, Y) },
        quote! { |k: fn(f32, f32) -> f32| k(X, Y) },
    ] {
        let err = refusal(closure.clone());
        assert!(
            err.contains("E0562") && err.contains("Phase D-a"),
            "{closure}: got {err}"
        );
    }
}

// ───────────────────────────── methods ─────────────────────────────

#[test]
fn an_unknown_method_is_refused_by_name() {
    let err = refusal(quote! { |r: f32| X.unknownmethod() });
    assert!(err.contains("unknown method"));
}

#[test]
fn typo_suggestion_for_method_matches_case_insensitively() {
    // "SQRT" differs from "sqrt" in every char position case-sensitively,
    // so only the case-insensitive fallback catches it.
    let err = refusal(quote! { || X.SQRT() });
    assert!(err.contains("did you mean 'sqrt'"), "{err}");
}

#[test]
fn a_method_name_near_a_method_gets_a_suggestion() {
    // One character from `sqrt`; only that a suggestion is offered is
    // asserted, so the vocabulary may grow another near name.
    let err = refusal(quote! { || X.sqrx() });
    assert!(err.contains("did you mean"), "{err}");
}

#[test]
fn a_method_name_of_a_methods_length_but_far_from_it_gets_no_suggestion() {
    // `qqqq` is the length of `sqrt` and differs from it in three
    // characters, past MAX_TYPO_CHAR_DIFF: matching length alone must not
    // be enough to suggest one.
    let err = refusal(quote! { || X.qqqq() });
    assert!(!err.contains("did you mean"), "{err}");
}

#[test]
fn a_method_name_far_from_every_length_gets_no_suggestion() {
    // Far from every method however the vocabulary grows.
    let err = refusal(quote! { || X.qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq() });
    assert!(!err.contains("did you mean"), "{err}");
}

/// A retired method is refused by name, with the way out.
#[test]
fn a_retired_method_is_refused() {
    for input in [quote! { || X.constant() }, quote! { || X.collapse() }] {
        let err = refusal(input);
        assert!(
            err.contains("sampled a manifold-typed parameter"),
            "got: {err}"
        );
    }
}
