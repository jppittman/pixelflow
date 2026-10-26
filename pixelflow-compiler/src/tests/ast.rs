//! The vocabulary's contract (`ast/mod.rs`): the one reading of a bare type
//! name, and Rust's lexical scoping over any binding.

use crate::ast::{Name, Scopes, bare_name, is_f32};
use proc_macro2::Span;
use syn::{Ident, Type};

/// The one reading of a bare type name: through a macro's invisible
/// group, and for nothing a path spells with more than its name.
#[test]
fn a_bare_name_is_seen_through_a_group_and_nothing_longer() {
    let fragment = proc_macro2::Group::new(proc_macro2::Delimiter::None, quote::quote!(f32));
    let grouped: Type = syn::parse_quote!(#fragment);
    assert!(matches!(grouped, Type::Group(_)));
    assert!(is_f32(&grouped));
    assert_eq!(bare_name(&syn::parse_quote!(usize)).unwrap(), "usize");
    for refused in [
        syn::parse_quote!(::f32),
        syn::parse_quote!(std::primitive::f32),
        syn::parse_quote!(<f32>::Item),
        syn::parse_quote!(Vec<f32>),
        syn::parse_quote!((f32)),
        syn::parse_quote!(&f32),
    ] {
        let refused: Type = refused;
        assert_eq!(bare_name(&refused), None, "{}", quote::quote!(#refused));
    }
}

fn name(text: &str) -> Name {
    Name::of(&Ident::new(text, Span::call_site()))
}

#[test]
fn a_lookup_sees_the_innermost_binding_in_scope() {
    let mut scopes = Scopes::default();
    scopes.bind(name("a"), 0);
    scopes.push_scope();
    scopes.bind(name("a"), 1);
    scopes.push_scope();
    assert_eq!(scopes.lookup(&name("a")), Some(&1));
    scopes.bind(name("a"), 2);
    scopes.bind(name("a"), 3);
    assert_eq!(scopes.lookup(&name("a")), Some(&3));
    scopes.pop_scope();
    assert_eq!(scopes.lookup(&name("a")), Some(&1));
    scopes.pop_scope();
    assert_eq!(scopes.lookup(&name("a")), Some(&0));
}
