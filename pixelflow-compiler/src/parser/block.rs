//! A block and what it holds: `let` statements, a plain name or a tuple of
//! them, the final expression, and the number a literal denotes.

use super::expr::{Nesting, convert_expr};
use super::fold::TUPLE_LET;
use crate::ast::{BlockExpr, Expr, LetStmt, Literal, PLAN, Stmt, same_name};
use syn::{Pat, Type};

/// A `{ .. }` in expression position. A block is a value, so a label has
/// nothing to jump to.
pub(super) fn convert_labeled_block(
    expr_block: syn::ExprBlock,
    nesting: Nesting,
) -> syn::Result<Expr> {
    match expr_block.label {
        Some(label) => Err(syn::Error::new_spanned(
            label,
            "a block label in a kernel body\n\
             \n\
             note: a block is a value, the last expression in it; there is no `break` \
             to name it, so a label has no use",
        )),
        None => convert_block(expr_block.block, nesting).map(Expr::Block),
    }
}

/// Convert a syn::Block to our BlockExpr.
pub(super) fn convert_block(block: syn::Block, nesting: Nesting) -> syn::Result<BlockExpr> {
    let span = block.brace_token.span.join();
    let count = block.stmts.len();
    let mut stmts = Vec::with_capacity(count);
    let mut final_expr = None;

    for (i, stmt) in block.stmts.into_iter().enumerate() {
        let is_last = i + 1 == count;

        match stmt {
            syn::Stmt::Local(local) => stmts.push(convert_let(local, nesting)?),

            syn::Stmt::Expr(expr, semi) => {
                let converted = convert_expr(expr, nesting)?;
                if is_last && semi.is_none() {
                    final_expr = Some(Box::new(converted));
                } else {
                    stmts.push(Stmt::Expr(converted));
                }
            }

            syn::Stmt::Item(item) => {
                return Err(syn::Error::new_spanned(
                    item,
                    "item definitions are not allowed inside a kernel body\n\
                     \n\
                     note: a body holds let bindings and expressions\n\
                     \n\
                     help: declare `const`s and helper `fn`s at the top of the kernel! block:\n\
                     help:   kernel! {\n\
                     help:       fn helper(x: f32) -> f32 { x * 2.0 }\n\
                     help:       pub fn entry() -> f32 { helper(X) }\n\
                     help:   }",
                ));
            }

            syn::Stmt::Macro(mac) => {
                return Err(syn::Error::new_spanned(
                    mac,
                    "macro invocations are not allowed inside kernel! blocks\n\
                     \n\
                     note: kernel! needs to analyze the expression at compile time\n\
                     \n\
                     help: expand the macro outside the kernel! or use equivalent expressions:\n\
                     help:   let value = some_macro!();\n\
                     help:   let my_kernel = kernel!(|| value * X);",
                ));
            }
        }
    }

    Ok(BlockExpr {
        stmts,
        expr: final_expr,
        span,
    })
}

/// A `let`: a plain name bound to its expression, or `let (a, b) = (e1,
/// e2);`, each name bound to its own expression (D7's front half).
fn convert_let(local: syn::Local, nesting: Nesting) -> syn::Result<Stmt> {
    let (pattern, annotation) = match &local.pat {
        Pat::Type(pat_type) => (&*pat_type.pat, Some(&*pat_type.ty)),
        pattern => (pattern, None),
    };
    let init = local.init.ok_or_else(|| {
        syn::Error::new_spanned(
            &local.pat,
            "let binding must have an initializer\n\
             \n\
             help: provide a value for this binding:\n\
             help:   let dx = X - cx;",
        )
    })?;
    // A plain name always matches, and so does a tuple of them, so an `else`
    // branch could never run: accepting one would compile a branch nobody
    // can reach, silently.
    if let Some((else_token, _)) = &init.diverge {
        return Err(syn::Error::new_spanned(
            else_token,
            "`let ... else` is not supported in a kernel body\n\
             \n\
             note: a kernel `let` binds a plain name, which always matches, \
             so the `else` branch could never run\n\
             \n\
             help: remove the `else` branch",
        ));
    }
    let bindings = match pattern {
        Pat::Ident(pat_ident) => vec![Binding {
            name: plain_name(pat_ident, &local.pat)?,
            ty: annotation.cloned(),
            value: *init.expr,
        }],
        Pat::Tuple(names) => tuple_let(names, annotation, *init.expr)?,
        pattern => return Err(refuse_let_pattern(pattern)),
    };
    bindings
        .into_iter()
        .map(|binding| {
            Ok(LetStmt {
                name: binding.name,
                ty: binding.ty,
                init: convert_expr(binding.value, nesting)?,
            })
        })
        .collect::<syn::Result<Vec<_>>>()
        .map(Stmt::Let)
}

/// A name a `let` binds, with the annotation and the expression it is bound
/// to, before the expression is converted.
struct Binding {
    name: syn::Ident,
    ty: Option<Type>,
    value: syn::Expr,
}

/// `let (a, b) = (e1, e2);`: plain names, as many as the tuple of
/// expressions has elements, each bound to its own, all at once. A tuple is
/// taken apart where it is written and nowhere else, so nothing of it is
/// left to be a value (D7's front half); a pattern within the pattern, a
/// `_`, and a value that is not a tuple of expressions are refused.
fn tuple_let(
    pattern: &syn::PatTuple,
    annotation: Option<&Type>,
    mut init: syn::Expr,
) -> syn::Result<Vec<Binding>> {
    let mut names: Vec<syn::Ident> = Vec::with_capacity(pattern.elems.len());
    for element in &pattern.elems {
        let name = match element {
            Pat::Ident(pat_ident) => plain_name(pat_ident, element)?,
            Pat::Wild(wild) => {
                return Err(syn::Error::new_spanned(
                    wild,
                    "`_` in a tuple `let`\n\
                     \n\
                     note: a kernel body has no effects, so a value nothing reads need not be \
                     written: drop it from both tuples",
                ));
            }
            Pat::Tuple(_) | Pat::Paren(_) => {
                return Err(syn::Error::new_spanned(
                    element,
                    format!(
                        "a pattern within a tuple `let`\n\
                         \n\
                         note: a tuple inside a tuple is a tuple value, and a tuple as a value \
                         is Phase D (D7 of {PLAN})\n\
                         help: bind the names flat, `let (a, b, c) = (e1, e2, e3);`"
                    ),
                ));
            }
            other => return Err(refuse_let_pattern(other)),
        };
        if names.iter().any(|bound| same_name(bound, &name)) {
            return Err(syn::Error::new(
                name.span(),
                format!(
                    "identifier `{name}` is bound more than once in the same pattern\n\
                     \n\
                     note: rustc refuses it too (E0416)"
                ),
            ));
        }
        names.push(name);
    }
    if names.is_empty() {
        return Err(syn::Error::new_spanned(
            pattern,
            "a tuple `let` of no names binds nothing\n\
             \n\
             note: a kernel body has no unit value to take apart",
        ));
    }
    let Some(tuple) = tuple_expression(&mut init) else {
        return Err(syn::Error::new_spanned(
            &init,
            format!(
                "the value of a tuple `let` is a tuple of expressions, `(e1, e2)`\n\
                 \n\
                 note: a tuple is taken apart where it is written; a tuple computed — \
                 returned, passed or chosen — is a tuple value, which is Phase D (D7 of \
                 {PLAN})"
            ),
        ));
    };
    if tuple.elems.len() != names.len() {
        let found = tuple.elems.len();
        return Err(syn::Error::new_spanned(
            &init,
            format!(
                "mismatched types: expected a tuple with {} elements, found one with {} \
                 elements\n\
                 \n\
                 note: `let ({}) = …` binds each name to one expression of the tuple",
                names.len(),
                found,
                names
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        ));
    }
    let values = std::mem::take(&mut tuple.elems);
    let types = element_types(annotation, names.len())?;
    Ok(names
        .into_iter()
        .zip(values)
        .zip(types)
        .map(|((name, value), ty)| Binding { name, ty, value })
        .collect())
}

/// The tuple an expression is, through parentheses, or `None` for anything
/// else.
fn tuple_expression(expr: &mut syn::Expr) -> Option<&mut syn::ExprTuple> {
    match expr {
        syn::Expr::Paren(paren) => tuple_expression(&mut paren.expr),
        syn::Expr::Tuple(tuple) => Some(tuple),
        _ => None,
    }
}

/// Each name's annotation in `let (a, b): (f32, f32) = …`: the tuple
/// type's elements, one per name, or none at all when there is no
/// annotation.
fn element_types(annotation: Option<&Type>, arity: usize) -> syn::Result<Vec<Option<Type>>> {
    let Some(annotation) = annotation else {
        return Ok(vec![None; arity]);
    };
    match annotation {
        Type::Tuple(tuple) if tuple.elems.len() == arity => {
            Ok(tuple.elems.iter().cloned().map(Some).collect())
        }
        _ => Err(syn::Error::new_spanned(
            annotation,
            format!(
                "the annotation of a tuple `let` of {arity} names is a tuple of {arity} types, \
                 `(f32, f32)`"
            ),
        )),
    }
}

/// A `let` whose pattern is neither a name nor a tuple of names — `let Row
/// { x0, .. } = …`, `let [a, b] = …` — takes a value apart that the language
/// reads otherwise: a record by field.
fn refuse_let_pattern(pattern: &Pat) -> syn::Error {
    syn::Error::new_spanned(
        pattern,
        format!(
            "a pattern in a kernel `let`\n\
             \n\
             note: a kernel `let` binds a plain name — an `f32`, a `bool`, or a record, which \
             it aliases (`let q = p;`) and whose fields are read by name (`q.x0`) — or takes \
             a tuple apart where it is written, {TUPLE_LET}\n\
             note: a record's fields are read by name, not taken apart by a pattern \
             (§1.3 of {PLAN})"
        ),
    )
}

/// The one type suffix a numeric literal in a kernel body may carry: every
/// value there is an `f32`.
const F32_SUFFIX: &str = "f32";

/// The number a literal denotes, or a spanned refusal saying why it names
/// none.
pub(super) fn literal_value(lit: &syn::Lit) -> syn::Result<Literal> {
    match lit {
        syn::Lit::Float(float) => float_value(float).map(Literal::F32),
        syn::Lit::Int(int) => int_value(int),
        other => Err(syn::Error::new_spanned(
            other,
            "only numeric literals are allowed in a kernel body\n\
             \n\
             note: every value in a kernel body is an `f32`; a `bool` comes from a comparison",
        )),
    }
}

/// A float literal rounds once, to the nearest `f32` — the value rustc gives
/// the same literal.
///
/// Parsing through `f64` first would round twice, and the two disagree
/// wherever the `f64` lands exactly on the midpoint between two `f32`s and
/// the cast breaks the tie to even: `1.00000005960464477539062500001` is
/// `1 + 2⁻²³` rounded once and `1.0` rounded twice.
fn float_value(float: &syn::LitFloat) -> syn::Result<f32> {
    refuse_a_foreign_suffix(float.suffix(), float)?;
    rounded_once(float.base10_digits(), float)
}

/// Digits rounded once to the nearest `f32` (`str::parse` is correctly
/// rounded). An infinity is refused, as rustc refuses it
/// (`overflowing_literals` is deny-by-default): it is not what anyone wrote.
fn rounded_once(digits: &str, literal: &impl quote::ToTokens) -> syn::Result<f32> {
    let value: f32 = digits
        .parse()
        .map_err(|err| syn::Error::new_spanned(literal, err))?;
    if value.is_infinite() {
        return Err(syn::Error::new_spanned(
            literal,
            format!(
                "literal out of range for `f32`\n\
                 \n\
                 note: `{digits}` rounds to infinity; the largest `f32` is {max:e}",
                max = f32::MAX
            ),
        ));
    }
    Ok(value)
}

/// An integer literal denotes an exact integer, and it is kept as one: which
/// type it is — an `f32` where a value is expected, a `usize` in a range's
/// bounds or a `usize` const — depends on where it stands, and
/// [`LiteralExpr::f32_value`](crate::ast::LiteralExpr::f32_value) and [`LiteralExpr::usize_value`](crate::ast::LiteralExpr::usize_value) answer for
/// each. Past `u128` it is refused, as rustc refuses it: no type in the
/// language holds it.
///
/// `16777217f32` is a float literal to rustc (the suffix makes it one), and
/// it rounds once like `16777217.0`, so that is what it does here too.
fn int_value(int: &syn::LitInt) -> syn::Result<Literal> {
    refuse_a_foreign_suffix(int.suffix(), int)?;
    if int.suffix() == F32_SUFFIX {
        return rounded_once(int.base10_digits(), int).map(Literal::F32);
    }
    int.base10_parse::<u128>().map(Literal::Int).map_err(|_| {
        syn::Error::new_spanned(
            int,
            format!(
                "integer literal is too large\n\
                 \n\
                 note: `{}` exceeds the largest integer a literal may be, {}",
                int.base10_digits(),
                u128::MAX
            ),
        )
    })
}

/// The plain name a `let` or a parameter binds. `mut`, `ref` and an `@`
/// subpattern are refused rather than dropped: nothing in a kernel body
/// assigns, borrows or matches, so each would be a promise the body cannot
/// keep, and a subpattern can refute a `let`, which rustc refuses too.
pub(super) fn plain_name(pat_ident: &syn::PatIdent, pat: &Pat) -> syn::Result<syn::Ident> {
    let decorated =
        pat_ident.by_ref.is_some() || pat_ident.mutability.is_some() || pat_ident.subpat.is_some();
    if decorated {
        return Err(syn::Error::new_spanned(
            pat,
            "a `let` or a parameter in a kernel binds a plain name\n\
             \n\
             note: `mut`, `ref` and `@` are refused rather than ignored: nothing in a \
             kernel body assigns, borrows or matches",
        ));
    }
    Ok(pat_ident.ident.clone())
}

/// A suffix naming a type other than `f32` is refused rather than dropped:
/// `2.5f64` or `3u8` asks for a type a kernel body does not have. A count is
/// written unsuffixed: an integer takes its type from where it stands.
fn refuse_a_foreign_suffix(suffix: &str, literal: &impl quote::ToTokens) -> syn::Result<()> {
    match suffix {
        "" | F32_SUFFIX => Ok(()),
        other => Err(syn::Error::new_spanned(
            literal,
            format!(
                "a `{other}` literal in a kernel body\n\
                 \n\
                 note: every value in a kernel body is an `f32`; an unsuffixed integer in a \
                 range's bounds or a `usize` const is a count\n\
                 \n\
                 help: remove the suffix, or write `{F32_SUFFIX}`"
            ),
        )),
    }
}
