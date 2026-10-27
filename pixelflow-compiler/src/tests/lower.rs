//! Lowering's contract (`lower/mod.rs`): `Expansion::lower` over what `parse`
//! then `analyze` give, asserting the `Lowered` term or the refusal text.

use crate::ast::Role;
use crate::lower::{Expansion, Lowered};
use crate::parser::parse;
use pixelflow_ir::arena::ExprNode;
use pixelflow_ir::arena::KERNEL_BINDER_CAPACITY;
use quote::quote;

/// `sema` counts the folds one body nests; a helper's fold inlined into
/// them nests deeper than any body shows, and lowering refuses that one,
/// where collapse would run out of binders.
#[test]
fn a_helper_fold_nests_inside_the_callers_and_is_refused_past_collapse_capacity() {
    use proc_macro2::TokenStream;
    let fold = |body: TokenStream| quote! { (0..1).map(|i| #body).sum::<f32>() };
    let nest =
        |depth: usize, innermost: TokenStream| (0..depth).fold(innermost, |body, _| fold(body));
    let program = |depth: usize| {
        let nested = nest(depth, quote! { deepest() });
        quote! {
            fn deepest() -> f32 { (0..1).map(|i| 1.0).sum::<f32>() }
            pub fn f() -> f32 { #nested }
        }
    };
    let through_sema = |input| {
        let analyzed = crate::sema::analyze(parse(input).expect("parses")).expect("analyzes");
        let entry = analyzed.def.fns().iter().find(|f| f.role() == Role::Entry);
        Expansion::lower(entry.expect("one entry").callable(), &analyzed)
    };
    let Ok(Lowered { arena, root, .. }) = through_sema(program(KERNEL_BINDER_CAPACITY - 1)) else {
        panic!("as deep as collapse has binders lowers");
    };
    let ExprNode::Reduce { fold: outer, .. } = arena.node(root) else {
        panic!("expected a fold");
    };
    assert_eq!(
        usize::from(outer.binder().slot()),
        KERNEL_BINDER_CAPACITY - 1
    );
    let Err(err) = through_sema(program(KERNEL_BINDER_CAPACITY)) else {
        panic!("one binder deeper than collapse admits");
    };
    let err = err.to_string();
    assert!(err.contains("collapse reserves 3"), "got: {err}");
}
