//! A fold, `(a..b).map(|i| e).sum()` and its five other spellings: the range,
//! the binder, the monoid, and the refusal of every method on a range or a
//! mapped range that is not one of them.

use super::block::plain_name;
use super::expr::{Nesting, convert_expr};
use super::items::no_collection_types;
use crate::ast::{CastExpr, Expr, FoldExpr, Name, RangeExpr, Reduction, is_f32};
use syn::Pat;
use syn::spanned::Spanned;

/// Every spelling of a fold, as a refusal of any other names them.
pub(super) const FOLD_SPELLINGS: &str = "`(a..b).map(|i| e).sum()`, `.product()`, \
     `.fold(f32::INFINITY, f32::min)` or `.fold(f32::NEG_INFINITY, f32::max)`, and \
     `(a..b).any(|i| m)` or `.all(|i| m)`";

/// The two `.fold`s that name a monoid, by its identity and its operation.
/// A `.fold` with any other arguments is a loop with state in disguise, or a
/// monoid whose identity is not the one written, so it is refused.
const FOLD_MONOIDS: [(Reduction, [&str; 2], [&str; 2]); 2] = [
    (Reduction::Min, ["f32", "INFINITY"], ["f32", "min"]),
    (Reduction::Max, ["f32", "NEG_INFINITY"], ["f32", "max"]),
];

/// A method call apart from its receiver: all a fold reads of the call
/// itself, owned so the closure that is its body moves rather than copies.
pub(super) struct MethodTail {
    method: syn::Ident,
    turbofish: Option<syn::AngleBracketedGenericArguments>,
    args: syn::punctuated::Punctuated<syn::Expr, syn::Token![,]>,
}

impl MethodTail {
    /// The call's method, type arguments and arguments, moved out of it; the
    /// call is read no further, since a fold's receiver is a range.
    pub(super) fn take(call: &mut syn::ExprMethodCall) -> Self {
        Self {
            method: call.method.clone(),
            turbofish: call.turbofish.take(),
            args: std::mem::take(&mut call.args),
        }
    }
}

/// The range a method is called on, through its parentheses: the `(a..b)`
/// of `(a..b).any(…)`. Taken out of the receiver, which is left without it;
/// any other receiver is untouched.
pub(super) fn take_range(receiver: &mut syn::Expr) -> Option<syn::ExprRange> {
    match receiver {
        syn::Expr::Paren(paren) => take_range(&mut paren.expr),
        syn::Expr::Range(range) => Some(syn::ExprRange {
            attrs: std::mem::take(&mut range.attrs),
            start: range.start.take(),
            limits: range.limits,
            end: range.end.take(),
        }),
        _ => None,
    }
}

/// `(a..b).map(f)`: the range and the `map` call a fold's terminal method is
/// called on.
pub(super) struct MappedRange {
    range: syn::ExprRange,
    map: MethodTail,
}

/// A [`MappedRange`], taken out of the receiver as [`take_range`] does.
pub(super) fn take_mapped_range(receiver: &mut syn::Expr) -> Option<MappedRange> {
    let syn::Expr::MethodCall(map) = receiver else {
        return None;
    };
    if RangeMethod::of(&map.method) != Some(RangeMethod::Map) {
        return None;
    }
    let range = take_range(&mut map.receiver)?;
    Some(MappedRange {
        range,
        map: MethodTail::take(map),
    })
}

/// The methods that iterate, by name: a fold's `.map`, `.any` and `.all`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RangeMethod {
    /// `.map(|i| e)`: an iterator, which a terminal method ends.
    Map,
    /// `.any(|i| m)`: a fold by itself.
    Any,
    /// `.all(|i| m)`: a fold by itself.
    All,
}

impl RangeMethod {
    /// The one reading of an iterating method's name.
    fn of(method: &syn::Ident) -> Option<Self> {
        match Name::of(method).to_string().as_str() {
            "map" => Some(Self::Map),
            "any" => Some(Self::Any),
            "all" => Some(Self::All),
            _ => None,
        }
    }
}

/// The method that ends `(a..b).map(|i| e)`, by name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FoldTerminal {
    Sum,
    Product,
    Fold,
}

impl FoldTerminal {
    /// The one reading of a terminal method's name.
    fn of(method: &syn::Ident) -> Option<Self> {
        match Name::of(method).to_string().as_str() {
            "sum" => Some(Self::Sum),
            "product" => Some(Self::Product),
            "fold" => Some(Self::Fold),
            _ => None,
        }
    }
}

/// The methods that make a collection something to iterate: an array's or a
/// slice's, by value or by reference.
const COLLECTION_METHODS: [&str; 2] = ["into_iter", "iter"];

/// The refusal of a collection iterated: one of [`COLLECTION_METHODS`], or
/// one of [`RangeMethod`]'s `map`, `any` and `all` with a closure called on anything but a
/// range — and not a closure out of place, which the closure's own refusal
/// would say. The one iteration is a fold, and it ranges over integers.
pub(super) fn a_collection_iterated(call: &syn::ExprMethodCall) -> Option<syn::Error> {
    if COLLECTION_METHODS.iter().any(|name| call.method == name) {
        return Some(no_collection_types(
            call,
            &format!("`.{}()`", call.method),
            &format!("the one iteration is a fold, over a range: {FOLD_SPELLINGS}"),
        ));
    }
    RangeMethod::of(&call.method)?;
    if !call
        .args
        .iter()
        .any(|arg| matches!(arg, syn::Expr::Closure(_)))
    {
        return None;
    }
    let receiver = &call.receiver;
    Some(no_collection_types(
        receiver,
        &format!(
            "`.{}` over `{}`, which is not a range,",
            call.method,
            quote::quote!(#receiver)
        ),
        &format!("a fold iterates a constant range: {FOLD_SPELLINGS}"),
    ))
}

/// The one place a tuple is written: a `let` taking it apart.
pub(super) const TUPLE_LET: &str = "`let (a, b) = (e1, e2);` binds each name to its expression";

/// A method called on a range itself: `.any(|i| m)` and `.all(|i| m)` are
/// folds, and nothing else is a value.
pub(super) fn convert_range_method(
    range: syn::ExprRange,
    call: MethodTail,
    nesting: Nesting,
) -> syn::Result<Expr> {
    let reduction = match RangeMethod::of(&call.method) {
        Some(RangeMethod::Any) => Reduction::Any,
        Some(RangeMethod::All) => Reduction::All,
        Some(RangeMethod::Map) => {
            return Err(syn::Error::new_spanned(
                &call.method,
                format!(
                    "a mapped range is an iterator, not a value\n\
                     \n\
                     note: a fold ends in the reduction that combines its terms: {FOLD_SPELLINGS}"
                ),
            ));
        }
        None => {
            return Err(syn::Error::new(
                call.method.span(),
                format!(
                    "`.{}` on a range\n\
                     \n\
                     note: a range is the domain of a fold: {FOLD_SPELLINGS}",
                    call.method
                ),
            ));
        }
    };
    refuse_turbofish(&call)?;
    let span = call.method.span();
    let (binder, body) = the_closure(call, nesting)?;
    Ok(Expr::Fold(FoldExpr {
        reduction,
        range: convert_range(range, nesting)?,
        binder,
        body: Box::new(body),
        span,
    }))
}

/// The method that ends `(a..b).map(|i| e)`: the reduction the fold combines
/// its terms under.
pub(super) fn convert_fold_terminal(
    MappedRange { range, map }: MappedRange,
    call: MethodTail,
    nesting: Nesting,
) -> syn::Result<Expr> {
    let reduction = match FoldTerminal::of(&call.method) {
        Some(FoldTerminal::Sum) => sum_or_product(&call, Reduction::Sum)?,
        Some(FoldTerminal::Product) => sum_or_product(&call, Reduction::Product)?,
        Some(FoldTerminal::Fold) => fold_monoid(&call)?,
        None => {
            return Err(syn::Error::new(
                call.method.span(),
                format!(
                    "`.{}` does not end a fold\n\
                     \n\
                     note: a fold is spelled {FOLD_SPELLINGS}",
                    call.method
                ),
            ));
        }
    };
    refuse_turbofish(&map)?;
    let (binder, body) = the_closure(map, nesting)?;
    Ok(Expr::Fold(FoldExpr {
        reduction,
        range: convert_range(range, nesting)?,
        binder,
        body: Box::new(body),
        span: call.method.span(),
    }))
}

/// `.sum()` and `.product()` take no arguments, and a type argument only if
/// it is the one type the terms have: `.sum::<f32>()`, which rustc needs
/// wherever nothing else says what the sum is.
fn sum_or_product(call: &MethodTail, reduction: Reduction) -> syn::Result<Reduction> {
    if let Some(arguments) = &call.turbofish {
        let names_f32 = match arguments.args.iter().collect::<Vec<_>>().as_slice() {
            [syn::GenericArgument::Type(ty)] => is_f32(ty),
            _ => false,
        };
        if !names_f32 {
            return Err(syn::Error::new_spanned(
                arguments,
                format!(
                    "`.{}` over anything but `f32`s\n\
                     \n\
                     note: a fold's terms are `f32`s; the one type argument it takes is \
                     `::<f32>`",
                    call.method
                ),
            ));
        }
    }
    if let Some(arg) = call.args.first() {
        return Err(syn::Error::new_spanned(
            arg,
            format!("`.{}()` takes no arguments", call.method),
        ));
    }
    Ok(reduction)
}

/// `operand as f32`. The conversion's target is always `f32`, so any other
/// is refused here and the AST does not hold one; what may be converted — a
/// `usize`, by name — is `sema`'s question, since it needs the name's type.
pub(super) fn convert_cast(expr_cast: syn::ExprCast, nesting: Nesting) -> syn::Result<Expr> {
    if !is_f32(&expr_cast.ty) {
        return Err(syn::Error::new_spanned(
            &expr_cast.ty,
            "`as` converts a `usize` to an `f32`, and to nothing else\n\
             \n\
             note: every value in a kernel body is an `f32` or a `bool`, and a `usize` (a \
             fold's index or a `usize` const) becomes one only as `i as f32`\n\
             help: write `as f32`",
        ));
    }
    Ok(Expr::Cast(CastExpr {
        span: expr_cast.as_token.span,
        operand: Box::new(convert_expr(*expr_cast.expr, nesting)?),
    }))
}

/// `.fold(identity, operation)` names a monoid only as one of
/// [`FOLD_MONOIDS`]; any other `.fold` is refused, naming them.
fn fold_monoid(call: &MethodTail) -> syn::Result<Reduction> {
    refuse_turbofish(call)?;
    let refusal = || {
        syn::Error::new(
            call.method.span(),
            "this `.fold` names no monoid the language has\n\
             \n\
             note: a `.fold` is `.fold(f32::INFINITY, f32::min)` or \
             `.fold(f32::NEG_INFINITY, f32::max)`: the minimum and the maximum, each with \
             its identity\n\
             note: a sum is `.sum()` and a product `.product()`; any other fold carries \
             state from one term to the next, and the language is a DAG",
        )
    };
    let [identity, operation] = call.args.iter().collect::<Vec<_>>()[..] else {
        return Err(refusal());
    };
    FOLD_MONOIDS
        .iter()
        .find(|(_, id, op)| is_path(identity, id) && is_path(operation, op))
        .map(|(reduction, ..)| *reduction)
        .ok_or_else(refusal)
}

/// Whether `expr` is exactly the path `segments`: `f32::INFINITY`.
fn is_path(expr: &syn::Expr, segments: &[&str; 2]) -> bool {
    let syn::Expr::Path(path) = expr else {
        return false;
    };
    path.qself.is_none()
        && path.path.leading_colon.is_none()
        && path.path.segments.len() == segments.len()
        && path
            .path
            .segments
            .iter()
            .zip(segments)
            .all(|(segment, name)| segment.arguments.is_empty() && segment.ident == name)
}

/// A method of a fold that takes no type argument.
fn refuse_turbofish(call: &MethodTail) -> syn::Result<()> {
    match &call.turbofish {
        Some(arguments) => Err(syn::Error::new_spanned(
            arguments,
            format!(
                "`.{}` takes no type arguments in a kernel body",
                call.method
            ),
        )),
        None => Ok(()),
    }
}

/// The one argument of a fold's `.map`, `.any` or `.all`: a closure whose
/// parameter is the fold's index and whose body is the fold's body.
fn the_closure(call: MethodTail, nesting: Nesting) -> syn::Result<(syn::Ident, Expr)> {
    let args = call.args.into_iter().collect::<Vec<_>>();
    let Ok([syn::Expr::Closure(closure)]) = <[syn::Expr; 1]>::try_from(args) else {
        return Err(syn::Error::new(
            call.method.span(),
            format!(
                "`.{}` takes one argument, the closure `|i| body`\n\
                 \n\
                 note: its parameter is the fold's index and its body is what the fold \
                 combines",
                call.method
            ),
        ));
    };
    let qualified = !closure.attrs.is_empty()
        || closure.lifetimes.is_some()
        || closure.constness.is_some()
        || closure.movability.is_some()
        || closure.asyncness.is_some()
        || closure.capture.is_some();
    if qualified {
        return Err(syn::Error::new_spanned(
            closure,
            "a fold's closure is `|i| body`, unqualified\n\
             \n\
             note: it is a term of the fold, not a value: nothing is captured, moved or awaited",
        ));
    }
    if let syn::ReturnType::Type(_, ty) = &closure.output {
        return Err(syn::Error::new_spanned(
            ty,
            "a fold's closure declares no return type: its body is a term of the fold, an \
             `f32` for a sum, product, min or max, a `bool` for `any` or `all`",
        ));
    }
    let [param] = closure.inputs.iter().collect::<Vec<_>>()[..] else {
        return Err(syn::Error::new_spanned(
            // `|| body` has no parameter tokens to span: the opening bar is
            // the closure's first token that always exists.
            closure.or1_token,
            "a fold's closure takes one parameter, the index it ranges over",
        ));
    };
    let binder = match param {
        Pat::Ident(pat_ident) => plain_name(pat_ident, param)?,
        Pat::Type(typed) => {
            return Err(syn::Error::new_spanned(
                &typed.ty,
                "a fold's index is a `usize`, always; write the plain name, `|i|`",
            ));
        }
        other => {
            return Err(syn::Error::new_spanned(
                other,
                "a fold's index is a plain name: `|i|`",
            ));
        }
    };
    Ok((binder, convert_expr(*closure.body, nesting)?))
}

/// `a..b`: both bounds, half-open. Whether they are constant, and run
/// forwards, is `sema`'s question: it evaluates them.
fn convert_range(range: syn::ExprRange, nesting: Nesting) -> syn::Result<RangeExpr> {
    if let syn::RangeLimits::Closed(dots) = &range.limits {
        return Err(syn::Error::new_spanned(
            dots,
            "an inclusive range: a fold ranges over a half-open `a..b`\n\
             \n\
             help: write the end one past the last index",
        ));
    }
    let span = range.limits.span();
    match (range.start, range.end) {
        (Some(lo), Some(hi)) => Ok(RangeExpr {
            lo: Box::new(convert_expr(*lo, nesting)?),
            hi: Box::new(convert_expr(*hi, nesting)?),
            span,
        }),
        (start, end) => Err(syn::Error::new_spanned(
            syn::ExprRange {
                attrs: range.attrs,
                start,
                limits: range.limits,
                end,
            },
            "a fold's range names both of its bounds: `a..b`",
        )),
    }
}
