//! Model 1, the declared type's surface: the one function type, and the one
//! reading of "this type is the bare name N".

use syn::{Ident, Type};

/// What a declared type is as a function.
///
/// The language has one function type, `impl Fn(f32, f32) -> f32`: a kernel
/// supplied at run time, a function of the two coordinates (plan §1.3,
/// §1.4, Phase D-a). The one classification of a type as that: `sema` types
/// a parameter by it and refuses every other spelling of a function, naming
/// this one, and lowering and emission find an entry's kernel-typed
/// parameters by it, so no two stages can disagree about which they are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FunctionType {
    /// Not a function.
    NotOne,
    /// `impl Fn(f32, f32) -> f32`.
    Kernel,
    /// A function spelled some other way: what differs from the one
    /// spelling.
    Misspelled(&'static str),
}

impl FunctionType {
    /// The classification of `ty`.
    pub(crate) fn of(ty: &Type) -> Self {
        match ty {
            Type::Paren(inner) => Self::of(&inner.elem),
            Type::Group(inner) => Self::of(&inner.elem),
            Type::ImplTrait(bounds) => Self::of_bounds(&bounds.bounds),
            Type::TraitObject(_) => Self::Misspelled(
                "`impl`, not `dyn`: a kernel-typed parameter is `impl Fn(f32, f32) -> f32`, \
                 and the host passes a `&Kernel` for it",
            ),
            Type::BareFn(_) => Self::Misspelled(
                "`impl Fn`, not a function pointer: a kernel is a program built at run time, \
                 not a Rust function",
            ),
            Type::Reference(reference) => match Self::of(&reference.elem) {
                Self::NotOne => Self::NotOne,
                Self::Kernel | Self::Misspelled(_) => Self::Misspelled(
                    "by value, `impl Fn(f32, f32) -> f32`, not by reference: the host \
                     function passes a `&Kernel` for it",
                ),
            },
            _ => Self::NotOne,
        }
    }

    /// `impl B`: a kernel when `B` is `Fn(f32, f32) -> f32`, alone.
    fn of_bounds(
        bounds: &syn::punctuated::Punctuated<syn::TypeParamBound, syn::Token![+]>,
    ) -> Self {
        let mut bounds = bounds.iter();
        let (Some(syn::TypeParamBound::Trait(bound)), None) = (bounds.next(), bounds.next()) else {
            return Self::Misspelled(
                "one bound, `Fn(f32, f32) -> f32`, and no other: a kernel is a function of \
                 the two coordinates and nothing more",
            );
        };
        if bound.lifetimes.is_some() || !matches!(bound.modifier, syn::TraitBoundModifier::None) {
            return Self::Misspelled("`Fn(f32, f32) -> f32`, with no lifetimes and no `?`");
        }
        let segment = match bound.path.segments.iter().collect::<Vec<_>>().as_slice() {
            [segment] if bound.path.leading_colon.is_none() => *segment,
            _ => return Self::Misspelled("`Fn`, by its own name"),
        };
        match segment.ident.to_string().as_str() {
            "Fn" => {}
            "FnMut" | "FnOnce" => {
                return Self::Misspelled(
                    "`Fn`: a kernel is applied, and nothing a body does mutates or consumes it",
                );
            }
            _ => {
                return Self::Misspelled(
                    "an `impl` type is a kernel, `impl Fn(f32, f32) -> f32`, and nothing else",
                );
            }
        }
        let syn::PathArguments::Parenthesized(signature) = &segment.arguments else {
            return Self::Misspelled("`Fn(f32, f32) -> f32`, its signature written out");
        };
        let takes_the_coordinates =
            signature.inputs.len() == 2 && signature.inputs.iter().all(is_f32);
        if !takes_the_coordinates {
            return Self::Misspelled(
                "`Fn(f32, f32)`: a kernel is a function of the two coordinates, applied as \
                 `k(x, y)`",
            );
        }
        match &signature.output {
            syn::ReturnType::Type(_, output) if is_f32(output) => Self::Kernel,
            _ => Self::Misspelled(
                "`-> f32`: a kernel's value is an `f32`; a mask is a `bool` a body computes, \
                 never a kernel's value",
            ),
        }
    }
}

/// The one name a declared type is, by itself: `f32`, `usize`, a record's.
///
/// The one reading of "this type is the bare name N", which every stage that
/// classifies a declaration builds on. A macro's `$t:ty` fragment arrives as
/// a type inside an invisible group, and rustc reads the type inside, so this
/// does too: a spelling is the same type in every position. A path with a
/// leading `::`, a qualified self or generic arguments names nothing bare.
pub(crate) fn bare_name(ty: &Type) -> Option<&Ident> {
    match ty {
        Type::Group(inner) => bare_name(&inner.elem),
        Type::Path(path) if path.qself.is_none() => path.path.get_ident(),
        _ => None,
    }
}

/// Whether `ty` is exactly `f32`: the one type a fold's terms, a kernel's
/// coordinates and value, and the one conversion's target, may name.
pub(crate) fn is_f32(ty: &Type) -> bool {
    bare_name(ty).is_some_and(|name| name == "f32")
}
