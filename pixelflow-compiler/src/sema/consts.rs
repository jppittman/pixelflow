//! Counts and `const`s: a `usize` expression checked as rustc's const
//! evaluator checks it, a fold's bounds, and every `const`'s value.

use crate::ast::{BinaryOp, ConstItem, Expr, Name, RangeExpr, UnaryOp};
use pixelflow_ir::passes::lattice::{COLLAPSE_BINDERS, KERNEL_BINDER_CAPACITY};
use pixelflow_ir::{Binder, Fold, KnownRange};
use proc_macro2::{Span, TokenStream};
use quote::quote;
use std::collections::HashMap;
use syn::Ident;

use super::analyze::{ConstValue, Ty};
use super::body::a_count_is_not_a_value;

/// A count: a `usize` evaluated at expansion or read as a value from an
/// enclosing fold or entry.
#[derive(Debug, Clone)]
pub(crate) enum Count {
    /// Evaluated here, each operation checked.
    Known(u64),
    /// A count bound by a fold or an entry parameter, read as a value.
    Name(Ident),
}

impl Count {
    pub(crate) fn name(&self) -> Option<&Ident> {
        match self {
            Count::Name(name) => Some(name),
            Count::Known(_) => None,
        }
    }
}

/// One of a count's operations on two known counts, `None` where rustc's
/// const evaluator would refuse it.
type CheckedOp = fn(u64, u64) -> Option<u64>;

/// Integer arithmetic over counts: what a `usize` const's initializer and a
/// fold's bounds are built from. Each operation on two known counts is
/// checked here, as rustc's const evaluator checks it, so a count that would
/// wrap is an error rather than a different count.
///
/// A trait because its scopes differ only in what a name means: a `const`
/// being evaluated may name another, evaluated on demand; a fold's bound
/// names a `usize` const, already evaluated, or a value count.
/// Anything else is not constant.
trait UsizeScope {
    /// The value of `name` in a `usize` expression, or why it has none.
    fn usize_named(&mut self, name: &Ident) -> syn::Result<Count>;

    /// The refusal of something a `usize` expression is not built from.
    fn not_a_count(&self, span: Span) -> syn::Error;

    /// An integer literal, a name, `+ - * /` and parentheses.
    fn eval_usize(&mut self, expr: &Expr) -> syn::Result<Count> {
        let binary = match expr {
            Expr::Literal(literal) => return literal.usize_value().map(Count::Known),
            Expr::Ident(name) => return self.usize_named(name),
            Expr::Paren(inner) => return self.eval_usize(inner),
            // Constant, but no count: rustc's refusal of the same tokens.
            Expr::Unary(unary) => match unary.op {
                UnaryOp::Neg => {
                    return Err(syn::Error::new(
                        unary.span,
                        "cannot apply unary operator `-` to type `usize`\n\
                         \n\
                         note: unsigned values cannot be negated; a range's bounds and a \
                         `usize` const are counts",
                    ));
                }
            },
            Expr::Binary(binary) => binary,
            other => return Err(self.not_a_count(other.span())),
        };
        let lhs = self.eval_usize(&binary.lhs)?;
        let rhs = self.eval_usize(&binary.rhs)?;
        let (checked, symbol): (CheckedOp, TokenStream) = match binary.op {
            BinaryOp::Add => (u64::checked_add, quote!(+)),
            BinaryOp::Sub => (u64::checked_sub, quote!(-)),
            BinaryOp::Mul => (u64::checked_mul, quote!(*)),
            BinaryOp::Div => (u64::checked_div, quote!(/)),
            BinaryOp::Lt
            | BinaryOp::Le
            | BinaryOp::Gt
            | BinaryOp::Ge
            | BinaryOp::Eq
            | BinaryOp::Ne
            | BinaryOp::BitAnd
            | BinaryOp::BitOr => return Err(self.not_a_count(binary.span)),
        };
        if let Some(name) = lhs.name().or_else(|| rhs.name()) {
            return Err(syn::Error::new(
                name.span(),
                format!(
                    "`{name}` is a value bound, and arithmetic on it is not a `usize` bound; convert it with `{name} as f32` in the fold body"
                ),
            ));
        }
        let (Count::Known(lhs), Count::Known(rhs)) = (&lhs, &rhs) else {
            unreachable!("a value bound in count arithmetic was refused above");
        };
        if binary.op == BinaryOp::Div && *rhs == 0 {
            return Err(syn::Error::new(
                binary.span,
                format!("attempt to divide `{lhs}_usize` by zero"),
            ));
        }
        checked(*lhs, *rhs).map(Count::Known).ok_or_else(|| {
            syn::Error::new(
                binary.span,
                format!(
                    "attempt to compute `{lhs}_usize {symbol} {rhs}_usize`, which would overflow"
                ),
            )
        })
    }
}

/// The names in a bound, before count arithmetic is evaluated. The analyzer
/// checks their types in the fold's enclosing scope.
pub(crate) fn bound_names(bound: &Expr) -> Vec<&Ident> {
    match bound {
        Expr::Ident(name) => vec![name],
        Expr::Paren(inner) => bound_names(inner),
        Expr::Unary(unary) => bound_names(&unary.operand),
        Expr::Binary(binary) => {
            let mut names = bound_names(&binary.lhs);
            names.extend(bound_names(&binary.rhs));
            names
        }
        _ => Vec::new(),
    }
}

/// A fold's bounds: known or value counts, and — when both are
/// known here — the range running forwards.
///
/// The one evaluation of them, and the one admission: a range is refused
/// here when it runs backwards, or when the IR cannot hold it: its index is
/// an `f32` lane, exact only to [`Fold::EXACT_BOUND`]. Value bounds are
/// checked by the IR guard.
///
/// A reversed known
/// range is refused rather than read as the empty fold it would be in Rust:
/// an empty range is written `a..a`, and `b..a` is almost always a slip
/// (clippy's `reversed_empty_ranges` is deny-by-default for the same
/// reason), and the IR's own `Fold` refuses one.
///
/// # Errors
///
/// A bound that is not a permitted count, an overflow, or a known range that
/// runs backwards or reaches past [`Fold::EXACT_BOUND`].
pub(crate) fn range_bounds(range: &RangeExpr, scope: RangeScope<'_>) -> syn::Result<[Count; 2]> {
    let mut scope = scope;
    let lo = scope.eval_usize(&range.lo)?;
    let hi = scope.eval_usize(&range.hi)?;
    let (Count::Known(first), Count::Known(last)) = (&lo, &hi) else {
        for (end, count) in [("first", &lo), ("last", &hi)] {
            if let Count::Known(value) = count {
                if *value > Fold::EXACT_BOUND {
                    return Err(syn::Error::new(
                        range.span,
                        format!(
                            "the {end} fold bound {value} reaches past {} (2^24): a fold's index is an exact `f32` only through that count",
                            Fold::EXACT_BOUND
                        ),
                    ));
                }
            }
        }
        return Ok([lo, hi]);
    };
    if first > last {
        return Err(syn::Error::new(
            range.span,
            format!(
                "the range `{first}..{last}` runs backwards\n\
                 \n\
                 note: `{first}..{last}` holds no index, so a fold over it would be its monoid's \
                 identity; the empty range is written `a..a`, and a reversed one is refused as \
                 the slip it usually is"
            ),
        ));
    }
    if !KnownRange::admits(&(*first..*last)) {
        return Err({
            let bound = Fold::EXACT_BOUND;
            syn::Error::new(
                range.span,
                format!(
                    "the range `{first}..{last}` reaches past {bound} (2^24): a fold's index is an \
                     `f32` lane, which names every integer only that far, so past it indices \
                     would round together and the fold would not be the one written \
                     (`Fold::EXACT_BOUND`)"
                ),
            )
        });
    }
    Ok([lo, hi])
}

/// The refusal of a fold opened with every user binder available to collapse
/// already taken.
///
/// `sema` makes it for the folds one body nests; lowering makes it again
/// where an inlined helper's fold nests inside a caller's, the one depth
/// `sema` does not see.
pub(crate) fn folds_nested_too_deep(span: Span) -> syn::Error {
    syn::Error::new(
        span,
        format!(
            "folds nested more than {} deep: collapse reserves {} of the IR's {} binder \
             slots for its row, column, and lane folds",
            KERNEL_BINDER_CAPACITY,
            COLLAPSE_BINDERS,
            Binder::COUNT,
        ),
    )
}

/// What a fold's bound may name: a `usize` const or an entry count
/// parameter.
#[derive(Clone, Copy)]
pub(crate) struct RangeScope<'a> {
    pub(crate) consts: &'a HashMap<Name, ConstValue>,
}

impl UsizeScope for RangeScope<'_> {
    fn usize_named(&mut self, name: &Ident) -> syn::Result<Count> {
        match self.consts.get(&Name::of(name)) {
            Some(ConstValue::Usize(value)) => Ok(Count::Known(*value)),
            Some(ConstValue::F32(_)) => Err(syn::Error::new(
                name.span(),
                format!(
                    "`{name}` is an `f32` const, and a range's bounds are `usize`s\n\
                     \n\
                     help: declare the count as `const {name}: usize`"
                ),
            )),
            None => Ok(Count::Name(name.clone())),
        }
    }

    fn not_a_count(&self, span: Span) -> syn::Error {
        syn::Error::new(
            span,
            "a range's bound is a count\n\
             \n\
             note: a bound is built from integer literals and `usize` consts, \
             or is an unchanged count parameter; \
             arithmetic on known counts uses `+ - * /` and parentheses",
        )
    }
}

/// Every `const`'s value.
///
/// A `const` is evaluated here, at expansion, and gets the value rustc gives
/// the same tokens. An `f32` const is evaluated per operation in `f32`,
/// since rustc evaluates an `f32` `const` in `f32` too. The discipline is
/// exactly the one rustc's const evaluator has: each operation rounds once,
/// in `f32`, and a product and a sum are never contracted into one rounding
/// — which matters because this crate is built with `-fp-contract=fast`
/// (`.cargo/config.toml`), and an FMA gives `b * c + d` a value two
/// roundings never reach. Its literals were rounded once by the parser, and
/// nothing here rounds again. A `usize` const is evaluated in 64-bit
/// integers, each operation checked ([`UsizeScope`]). A const may name
/// another declared anywhere in the block; a cycle is refused.
pub(super) fn evaluate_consts(
    consts: &[ConstItem],
    types: &HashMap<Name, Ty>,
) -> syn::Result<HashMap<Name, ConstValue>> {
    let mut evaluator = ConstEvaluator {
        items: consts.iter().map(|c| (Name::of(&c.name), c)).collect(),
        types,
        values: HashMap::with_capacity(consts.len()),
        in_progress: Vec::new(),
    };
    for c in consts {
        evaluator.value_of(&c.name)?;
    }
    Ok(evaluator.values)
}

struct ConstEvaluator<'a> {
    items: HashMap<Name, &'a ConstItem>,
    /// Each const's declared type, which decides how it is evaluated.
    types: &'a HashMap<Name, Ty>,
    values: HashMap<Name, ConstValue>,
    /// The consts whose initializers are being evaluated, outermost first:
    /// naming one of them again is a cycle.
    in_progress: Vec<Name>,
}

impl UsizeScope for ConstEvaluator<'_> {
    fn usize_named(&mut self, name: &Ident) -> syn::Result<Count> {
        match self.value_of(name)? {
            ConstValue::Usize(value) => Ok(Count::Known(value)),
            ConstValue::F32(_) => Err(syn::Error::new(
                name.span(),
                format!(
                    "`{name}` is an `f32` const, where a `usize` is expected\n\
                     \n\
                     note: a `usize` const is built from integers and other `usize` consts"
                ),
            )),
        }
    }

    fn not_a_count(&self, span: Span) -> syn::Error {
        Self::not_constant(span)
    }
}

impl ConstEvaluator<'_> {
    /// `f32` arithmetic at expansion: what an `f32` const's initializer is
    /// built from — literals, `f32` consts, a `usize` const `as f32`,
    /// `+ - * /`, unary `-` and parentheses — each operation in `f32`, as
    /// rustc's const evaluator does it (see [`evaluate_consts`]).
    fn eval_f32(&mut self, expr: &Expr) -> syn::Result<f32> {
        match expr {
            Expr::Literal(literal) => literal.f32_value(),
            Expr::Ident(name) => self.f32_named(name),
            Expr::Paren(inner) => self.eval_f32(inner),
            // Rust's own `as`, which rounds to the nearest `f32` as rustc's
            // does: exact up to 2²⁴. A `const` cannot name a value
            // parameter — they are an entry's — so a count this does not
            // know is refused rather than assumed away.
            Expr::Cast(cast) => {
                let Some(name) = cast.named() else {
                    return Err(Self::not_constant(cast.span));
                };
                match self.usize_named(name)? {
                    Count::Known(count) => Ok(count as f32),
                    Count::Name(_) => Err(Self::not_constant(cast.span)),
                }
            }
            Expr::Unary(unary) => match unary.op {
                UnaryOp::Neg => Ok(-self.eval_f32(&unary.operand)?),
            },
            Expr::Binary(binary) => {
                let lhs = self.eval_f32(&binary.lhs)?;
                let rhs = self.eval_f32(&binary.rhs)?;
                match binary.op {
                    BinaryOp::Add => Ok(lhs + rhs),
                    BinaryOp::Sub => Ok(lhs - rhs),
                    BinaryOp::Mul => Ok(lhs * rhs),
                    BinaryOp::Div => Ok(lhs / rhs),
                    BinaryOp::Lt
                    | BinaryOp::Le
                    | BinaryOp::Gt
                    | BinaryOp::Ge
                    | BinaryOp::Eq
                    | BinaryOp::Ne
                    | BinaryOp::BitAnd
                    | BinaryOp::BitOr => Err(Self::not_constant(binary.span)),
                }
            }
            other => Err(Self::not_constant(other.span())),
        }
    }

    /// The `f32` const `name`'s value; a `usize` const is a count, and is a
    /// value only as `name as f32`.
    fn f32_named(&mut self, name: &Ident) -> syn::Result<f32> {
        match self.value_of(name)? {
            ConstValue::F32(value) => Ok(value),
            ConstValue::Usize(_) => Err(a_count_is_not_a_value(name)),
        }
    }

    /// The value of the const `name`, evaluating it on first demand.
    fn value_of(&mut self, name: &Ident) -> syn::Result<ConstValue> {
        let key = Name::of(name);
        if let Some(&value) = self.values.get(&key) {
            return Ok(value);
        }
        let Some(item) = self.items.get(&key).copied() else {
            return Err(syn::Error::new(
                name.span(),
                format!(
                    "cannot find a `const` named `{key}`\n\
                     \n\
                     note: a `const` initializer is evaluated at expansion, so it names \
                     only literals and other `const`s of this block"
                ),
            ));
        };
        if self.in_progress.contains(&key) {
            let cycle = self
                .in_progress
                .iter()
                .map(Name::to_string)
                .collect::<Vec<_>>()
                .join("` → `");
            return Err(syn::Error::new(
                name.span(),
                format!(
                    "`{key}` depends on itself\n\
                     \n\
                     note: `{cycle}` → `{key}`"
                ),
            ));
        }
        self.in_progress.push(key.clone());
        let value = match self.types.get(&key) {
            Some(Ty::Usize) => ConstValue::Usize(self.known_usize(&item.init)?),
            Some(Ty::F32) => ConstValue::F32(self.eval_f32(&item.init)?),
            Some(Ty::Bool | Ty::Record(_) | Ty::Kernel) | None => {
                return Err(syn::Error::new_spanned(
                    &item.ty,
                    "a `const` in a `kernel!` block is an `f32` or a `usize`",
                ));
            }
        };
        self.in_progress.pop();
        self.values.insert(key, value);
        Ok(value)
    }

    /// A `usize` initializer's value, known here. A value count
    /// leaves a count unknown, and a `const` item cannot name one — they are
    /// an entry's — so one that is not known is refused rather than assumed
    /// away.
    fn known_usize(&mut self, expr: &Expr) -> syn::Result<u64> {
        match self.eval_usize(expr)? {
            Count::Known(value) => Ok(value),
            Count::Name(_) => Err(Self::not_constant(expr.span())),
        }
    }

    fn not_constant(span: Span) -> syn::Error {
        syn::Error::new(
            span,
            "a `const` initializer is evaluated at expansion\n\
             \n\
             note: an `f32` const is built from literals, other `f32` consts, a `usize` \
             const `as f32`, `+ - * /`, unary `-` and parentheses, each operation in `f32`\n\
             note: a `usize` const is built from integer literals, other `usize` consts, \
             `+ - * /` and parentheses, each operation checked",
        )
    }
}
