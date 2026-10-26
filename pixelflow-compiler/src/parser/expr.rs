//! A body's expressions: the one dispatcher every nesting level recurses
//! through, each syntactic form's conversion as its own arm, the method and
//! operator tables, and the refusal of every form the language has not.

use super::MAX_NESTING;
use super::block::{convert_block, convert_labeled_block, literal_value};
use super::fold::{
    FOLD_SPELLINGS, MethodTail, TUPLE_LET, a_collection_iterated, convert_cast,
    convert_fold_terminal, convert_range_method, take_mapped_range, take_range,
};
use super::items::no_collection_types;
use crate::ast::{
    Arity, BinaryExpr, BinaryOp, CallExpr, Expr, FieldExpr, IfExpr, LiteralExpr, Method,
    MethodCallExpr, Name, PLAN, ProjectExpr, Projection, UnaryExpr, UnaryOp, arity_mismatch,
};
use proc_macro2::Span;
use syn::spanned::Spanned;

/// How deep in a body's expression the conversion is: the depth of the
/// `convert_expr` that is running, handed down by value so a level that is
/// finished takes its count with it.
#[derive(Clone, Copy)]
pub(super) struct Nesting(usize);

impl Nesting {
    /// Outside every expression.
    pub(super) const ROOT: Self = Self(0);

    /// One level below this, or the refusal of an expression past
    /// [`MAX_NESTING`].
    fn deeper(self, expr: &syn::Expr) -> syn::Result<Self> {
        match self.0 < MAX_NESTING {
            true => Ok(Self(self.0 + 1)),
            false => Err(refuse_nesting(expr)),
        }
    }
}

/// The refusal of an expression nested past [`MAX_NESTING`]. Cold, and
/// pointed at the node's own head token ([`head_span`]): the refused
/// expression is the whole unconverted rest of the chain, and the span of a
/// subtree is a recursive walk of all of it, which overflows rustc's stack on
/// a chain far enough past the bound.
#[cold]
#[inline(never)]
fn refuse_nesting(expr: &syn::Expr) -> syn::Error {
    syn::Error::new(
        head_span(expr),
        format!(
            "expression nested deeper than {MAX_NESTING}\n\
             \n\
             note: each level of a body's nesting is a level of the compiler's own stack\n\
             help: bind the parts with `let`, or move one into a helper `fn`"
        ),
    )
}

/// The span of the token that makes `expr` the shape it is (its operator,
/// method name, keyword or delimiter), in constant time. Never `Spanned::span`
/// of the node, which walks every descendant.
fn head_span(expr: &syn::Expr) -> Span {
    match expr {
        syn::Expr::Binary(binary) => binary.op.span(),
        syn::Expr::Unary(unary) => unary.op.span(),
        syn::Expr::MethodCall(method) => method.method.span(),
        syn::Expr::Call(call) => call.paren_token.span.join(),
        syn::Expr::If(expr_if) => expr_if.if_token.span,
        syn::Expr::Paren(paren) => paren.paren_token.span.join(),
        syn::Expr::Field(field) => field.member.span(),
        syn::Expr::Cast(cast) => cast.as_token.span,
        syn::Expr::Block(block) => block.block.brace_token.span.join(),
        _ => Span::call_site(),
    }
}

/// Drop the unconverted suffix of a rejected chain without recursing through
/// its boxes. At the nesting limit that suffix may contain tens of thousands
/// of `else if`, binary, method or field nodes. Letting Rust drop it as one
/// `syn::Expr` overflows the compiler thread even though the parser has
/// already produced its refusal.
fn drop_rejected_chain(expr: syn::Expr) {
    let mut pending = vec![expr];
    while let Some(expr) = pending.pop() {
        match expr {
            syn::Expr::Binary(binary) => {
                pending.push(*binary.left);
                pending.push(*binary.right);
            }
            syn::Expr::MethodCall(call) => {
                pending.push(*call.receiver);
                pending.extend(call.args);
            }
            syn::Expr::Field(field) => pending.push(*field.base),
            syn::Expr::If(expr_if) => {
                pending.push(*expr_if.cond);
                if let Some((_, otherwise)) = expr_if.else_branch {
                    pending.push(*otherwise);
                }
            }
            syn::Expr::Unary(unary) => pending.push(*unary.expr),
            syn::Expr::Paren(paren) => pending.push(*paren.expr),
            syn::Expr::Cast(cast) => pending.push(*cast.expr),
            syn::Expr::Call(call) => {
                pending.push(*call.func);
                pending.extend(call.args);
            }
            other => drop(other),
        }
    }
}

/// Convert syn::Expr to our AST Expr.
///
/// Only a dispatcher: this is the function every nesting level of a body
/// recurses through, so what a level costs is this frame plus its arm's, and
/// each arm is a function of its own with its own temporaries. A refusal's
/// text is built in [`refuse_expression`], off the recursive path.
pub(super) fn convert_expr(expr: syn::Expr, nesting: Nesting) -> syn::Result<Expr> {
    let nesting = match nesting.deeper(&expr) {
        Ok(nesting) => nesting,
        Err(refusal) => {
            drop_rejected_chain(expr);
            return Err(refusal);
        }
    };
    match expr {
        syn::Expr::Path(expr_path) => convert_path(expr_path),
        syn::Expr::Lit(expr_lit) => convert_lit(expr_lit),
        syn::Expr::Binary(expr_binary) => convert_binary(expr_binary, nesting),
        syn::Expr::Unary(expr_unary) => convert_unary(expr_unary, nesting),
        syn::Expr::MethodCall(expr_method) => convert_method_call(expr_method, nesting),
        syn::Expr::Call(expr_call) => convert_call(expr_call, nesting),
        syn::Expr::If(expr_if) => convert_if(expr_if, nesting),
        syn::Expr::Paren(expr_paren) => convert_paren(expr_paren, nesting),
        syn::Expr::Field(expr_field) => convert_field(expr_field, nesting),
        syn::Expr::Cast(expr_cast) => convert_cast(expr_cast, nesting),
        syn::Expr::Block(expr_block) => convert_labeled_block(expr_block, nesting),
        refused => Err(refuse_expression(&refused)),
    }
}

/// A name: `X`, `cx`. A path of any other shape names something outside the
/// block.
fn convert_path(expr_path: syn::ExprPath) -> syn::Result<Expr> {
    if expr_path.path.segments.len() == 1 && expr_path.qself.is_none() {
        let segment = &expr_path.path.segments[0];
        if segment.arguments.is_empty() {
            return Ok(Expr::Ident(segment.ident.clone()));
        }
    }
    Err(syn::Error::new_spanned(
        &expr_path,
        format!(
            "a path in a kernel body: `{}`\n\
             \n\
             note: a kernel body sees X, Y (in an entry), its parameters, the block's \
             `const`s, and the `let` bindings in scope; a name from the enclosing Rust \
             scope is not captured\n\
             help: declare the value as a `const` of this block, or as a parameter of \
             this kernel",
            quote::quote!(#expr_path)
        ),
    ))
}

fn convert_lit(expr_lit: syn::ExprLit) -> syn::Result<Expr> {
    Ok(Expr::Literal(LiteralExpr {
        value: literal_value(&expr_lit.lit)?,
        span: expr_lit.lit.span(),
    }))
}

fn convert_binary(expr_binary: syn::ExprBinary, nesting: Nesting) -> syn::Result<Expr> {
    let op = binary_op(&expr_binary.op)?;
    let span = expr_binary.op.span();
    let lhs = convert_expr(*expr_binary.left, nesting)?;
    let rhs = convert_expr(*expr_binary.right, nesting)?;
    Ok(Expr::Binary(BinaryExpr {
        op,
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
        span,
    }))
}

fn convert_unary(expr_unary: syn::ExprUnary, nesting: Nesting) -> syn::Result<Expr> {
    let op = unary_op(&expr_unary.op)?;
    let span = expr_unary.op.span();
    let operand = convert_expr(*expr_unary.expr, nesting)?;
    Ok(Expr::Unary(UnaryExpr {
        op,
        operand: Box::new(operand),
        span,
    }))
}

fn convert_paren(expr_paren: syn::ExprParen, nesting: Nesting) -> syn::Result<Expr> {
    let inner = convert_expr(*expr_paren.expr, nesting)?;
    Ok(Expr::Paren(Box::new(inner)))
}

/// The refusal of an expression the language has no form for, naming what it
/// found. Cold and never inlined: a `format!` in [`convert_expr`] would put
/// its temporaries in the frame of every level of every body.
#[cold]
#[inline(never)]
fn refuse_expression(expr: &syn::Expr) -> syn::Error {
    match expr {
        // A value with parts, built in a body. A record enters a kernel as a
        // parameter and is read by field; a tuple is taken apart where it is
        // written, by a `let` (D7's front half), and a tuple or a record
        // built as a value — passed, returned, chosen — is Phase D (D7).
        syn::Expr::Tuple(expr_tuple) => syn::Error::new(
            expr_tuple.paren_token.span.join(),
            format!(
                "a tuple in a kernel body\n\
                 \n\
                 note: every value in a kernel body is an `f32` or a `bool`; a tuple is taken \
                 apart where it is written, by a `let`: {TUPLE_LET}\n\
                 note: a tuple as a value — passed, returned or chosen — is Phase D (D7 of \
                 {PLAN})"
            ),
        ),

        // There are no collection types (§1.3, §1.6): nothing is read by
        // index, a slice is a range of indices, and nothing builds an array.
        syn::Expr::Index(expr_index) => no_collection_types(
            expr_index,
            "indexing",
            "a value is an `f32` or a `bool` and a record is read by field, `p.x0`; nothing \
             is read by index, and a slice is a range of indices",
        ),
        syn::Expr::Array(_) | syn::Expr::Repeat(_) => no_collection_types(
            expr,
            "an array",
            "several values are several parameters, or several `let`s",
        ),

        syn::Expr::Struct(expr_struct) => syn::Error::new_spanned(
            &expr_struct.path,
            format!(
                "a record literal in a kernel body\n\
                 \n\
                 note: a record enters a kernel as a parameter, one uniform per field, and a \
                 body reads its fields, `p.x0`\n\
                 note: building a record, and returning one, are Phase D (D7 of {PLAN})"
            ),
        ),

        syn::Expr::Range(expr_range) => syn::Error::new_spanned(
            expr_range,
            format!(
                "a range in a kernel body\n\
                 \n\
                 note: a range is the domain of a fold: {FOLD_SPELLINGS}"
            ),
        ),

        // Iteration with state has no denotation in a DAG. A bounded
        // reduction is a fold over a constant range.
        syn::Expr::Loop(_) | syn::Expr::While(_) | syn::Expr::ForLoop(_) => {
            syn::Error::new_spanned(
                expr,
                format!(
                    "a loop in a kernel body\n\
                     \n\
                     note: the language is a DAG: nothing iterates with state\n\
                     note: a bounded reduction over a constant range is a fold: {FOLD_SPELLINGS}"
                ),
            )
        }

        syn::Expr::Assign(_) => syn::Error::new_spanned(
            expr,
            "assignment in a kernel body\n\
             \n\
             note: a kernel body binds a name once, with `let`; nothing is mutable",
        ),

        syn::Expr::Return(_) => syn::Error::new_spanned(
            expr,
            "`return` in a kernel body\n\
             \n\
             note: a body is an expression; its value is its final expression",
        ),

        syn::Expr::Closure(_) => syn::Error::new_spanned(
            expr,
            format!(
                "a closure in a kernel body\n\
                 \n\
                 note: a closure is the body of a fold, and nothing else: {FOLD_SPELLINGS}\n\
                 note: a function is passed as a kernel, to an entry's or a helper's \
                 kernel-typed parameter, `k: impl Fn(f32, f32) -> f32`, and applied by name, \
                 `k(x, y)` (Phase D-a of {PLAN}); a private `fn` in the block is a helper, \
                 called by name"
            ),
        ),

        // `syn::Expr` is `#[non_exhaustive]`, so the arm stays; it refuses at
        // the token, naming the syntax, and nothing is passed through.
        other => syn::Error::new_spanned(
            other,
            format!(
                "unsupported expression in a kernel body: `{}`",
                quote::quote!(#other)
            ),
        ),
    }
}

/// `p.x0`: a named field. Which record `p` is, and whether it has the field,
/// are `sema`'s questions. A tuple's field, `p.0`, is refused: a record's
/// fields have names.
fn convert_field(expr_field: syn::ExprField, nesting: Nesting) -> syn::Result<Expr> {
    let member = match expr_field.member {
        syn::Member::Named(member) => member,
        syn::Member::Unnamed(index) => {
            return Err(syn::Error::new_spanned(
                index,
                format!(
                    "a tuple's field in a kernel body\n\
                     \n\
                     note: a record's fields have names, `p.x0`; a tuple is taken apart by a \
                     `let`, {TUPLE_LET}\n\
                     note: a tuple as a value is Phase D (D7 of {PLAN})"
                ),
            ));
        }
    };
    Ok(Expr::Field(FieldExpr {
        base: Box::new(convert_expr(*expr_field.base, nesting)?),
        member,
    }))
}

/// `if c { a } else { b }`, and `else if` chains. The `else` is required —
/// there is no unit for an `if` without one to be — and `if let` is not a
/// choice between two values.
fn convert_if(expr_if: syn::ExprIf, nesting: Nesting) -> syn::Result<Expr> {
    let mut current = expr_if;
    let mut depth = nesting;
    let mut arms = Vec::new();
    let tail = loop {
        let span = current.if_token.span;
        let Some((_, else_expr)) = current.else_branch else {
            return Err(syn::Error::new(
                span,
                "an `if` without an `else`\n\
                 \n\
                 note: an `if` is the choice between two values, so it needs both; a kernel \
                 body has no unit for the missing arm to be",
            ));
        };
        let cond = match *current.cond {
            syn::Expr::Let(pattern) => {
                drop_rejected_chain(*else_expr);
                return Err(syn::Error::new_spanned(
                    pattern,
                    "`if let` in a kernel body\n\
                     \n\
                     note: an `if` chooses between two values by a `bool`; nothing here is a pattern",
                ));
            }
            cond => match convert_expr(cond, depth) {
                Ok(cond) => cond,
                Err(error) => {
                    drop_rejected_chain(*else_expr);
                    return Err(error);
                }
            },
        };
        let then_branch = match convert_block(current.then_branch, depth) {
            Ok(branch) => branch,
            Err(error) => {
                drop_rejected_chain(*else_expr);
                return Err(error);
            }
        };
        arms.push((cond, then_branch, span));
        match *else_expr {
            syn::Expr::If(next) => {
                let next_expr = syn::Expr::If(next);
                depth = match depth.deeper(&next_expr) {
                    Ok(depth) => depth,
                    Err(error) => {
                        drop_rejected_chain(next_expr);
                        return Err(error);
                    }
                };
                let syn::Expr::If(next) = next_expr else {
                    unreachable!("the else branch was an if expression");
                };
                current = next;
            }
            tail => break convert_expr(tail, depth)?,
        }
    };
    Ok(arms
        .into_iter()
        .rev()
        .fold(tail, |else_branch, (cond, then_branch, span)| {
            Expr::If(IfExpr {
                cond: Box::new(cond),
                then_branch,
                else_branch: Box::new(else_branch),
                span,
            })
        }))
}

/// `receiver.method(args)`: a fold's terminal, a method of the language, or
/// a refusal of any other name.
fn convert_method_call(
    mut expr_method: syn::ExprMethodCall,
    nesting: Nesting,
) -> syn::Result<Expr> {
    if let Some(range) = take_range(&mut expr_method.receiver) {
        return convert_range_method(range, MethodTail::take(&mut expr_method), nesting);
    }
    if let Some(mapped) = take_mapped_range(&mut expr_method.receiver) {
        return convert_fold_terminal(mapped, MethodTail::take(&mut expr_method), nesting);
    }
    // Built before the receiver is converted, returned after it: the
    // receiver's own refusal, the innermost, comes first, as rustc's
    // does (`(0..4).rev().map(…)` is about `.rev`).
    let collection = a_collection_iterated(&expr_method);
    if let Some(arguments) = &expr_method.turbofish {
        return Err(syn::Error::new_spanned(
            arguments,
            "type arguments on a method in a kernel body\n\
             \n\
             note: every value in a kernel body is an `f32` or a `bool`; no method is \
             generic",
        ));
    }
    let receiver = convert_expr(*expr_method.receiver, nesting)?;
    if let Some(refusal) = collection {
        return Err(refusal);
    }
    let args = expr_method
        .args
        .into_iter()
        .map(|arg| convert_expr(arg, nesting))
        .collect::<syn::Result<Vec<_>>>()?;
    let method = resolve_method(&expr_method.method, args.len())?;
    Ok(Expr::MethodCall(MethodCallExpr {
        receiver: Box::new(receiver),
        name: expr_method.method,
        method,
        args,
    }))
}

/// Methods of `Kernel`, not of the language: a body composes by applying a
/// kernel-typed parameter, `k(x, y)`, and the host composes `Kernel` values.
const RETIRED_METHODS: [&str; 2] = ["constant", "collapse"];

/// Maximum per-character difference for a same-length method name to be
/// suggested as a typo fix (e.g. `sqrtt` -> `sqrt`).
const MAX_TYPO_CHAR_DIFF: usize = 2;

/// What `name` denotes called with `arity` arguments, or the refusal of it:
/// the one place a method's spelling is read.
fn resolve_method(name: &syn::Ident, arity: usize) -> syn::Result<Method> {
    let text = Name::of(name).to_string();
    if RETIRED_METHODS.contains(&text.as_str()) {
        return Err(syn::Error::new(
            name.span(),
            format!(
                "`.{text}()` inside a kernel body sampled a manifold-typed \
                 parameter, and there are none\n\
                 help: every expression is already a field: use it as it stands, or \
                 observe it elsewhere with `.at(x, y)`"
            ),
        ));
    }
    Method::resolve(&text, arity).ok_or_else(|| unknown_method(name, arity))
}

/// The refusal of a method name nothing advertises, or of a known one at
/// the wrong arity.
fn unknown_method(name: &syn::Ident, arity: usize) -> syn::Error {
    let text = Name::of(name).to_string();

    // A recognized name at the wrong arity is not an unknown name: the typo
    // search below would find the very name just declared unknown
    // (`unknown method 'sqrt'; did you mean 'sqrt'?`). Say which arity the
    // name takes.
    if let Some((_, want)) = Method::advertised().find(|(known, _)| *known == text) {
        return arity_mismatch(name.span(), &text, Arity { want, got: arity });
    }

    let suggestion = Method::advertised().map(|(known, _)| known).find(|known| {
        known.to_lowercase() == text.to_lowercase()
            || (known.len() == text.len()
                && known
                    .chars()
                    .zip(text.chars())
                    .filter(|(a, b)| a != b)
                    .count()
                    <= MAX_TYPO_CHAR_DIFF)
    });

    let msg = match suggestion {
        Some(similar) => format!(
            "unknown method '{text}'\n\
             help: did you mean '{similar}'?"
        ),
        None => format!(
            "unknown method '{text}'\n\
             note: common methods: sqrt, abs, sin, cos, exp, min, max, clone\n\
             help: see Kernel's method surface for what is available"
        ),
    };

    syn::Error::new(name.span(), msg)
}

/// A free call by a single name: a projection, resolved here, or a helper or
/// a kernel, which only `sema` can tell apart (it takes the scope).
fn convert_call(expr_call: syn::ExprCall, nesting: Nesting) -> syn::Result<Expr> {
    let syn::Expr::Path(ref path) = *expr_call.func else {
        return Err(refuse_a_qualified_call(&expr_call));
    };
    if path.path.segments.len() != 1 || path.qself.is_some() {
        return Err(refuse_a_qualified_call(&expr_call));
    }
    let segment = &path.path.segments[0];
    if !segment.arguments.is_empty() {
        return Err(refuse_type_arguments_on_a_call(segment));
    }
    let func = segment.ident.clone();
    let args = expr_call
        .args
        .into_iter()
        .map(|arg| convert_expr(arg, nesting))
        .collect::<syn::Result<Vec<_>>>()?;
    // A projection takes one argument. At any other arity the name stays a
    // call, so `sema` refuses it after the binding checks: a kernel parameter
    // named `DX` and applied at two coordinates is a binding named after a
    // projection, and that refusal says what to rename.
    let Some(projection) = Projection::from_name(&Name::of(&func)) else {
        return Ok(Expr::Call(CallExpr { func, args }));
    };
    Ok(match <[Expr; 1]>::try_from(args) {
        Ok([arg]) => Expr::Project(ProjectExpr {
            projection,
            arg: Box::new(arg),
            span: func.span(),
        }),
        Err(args) => Expr::Call(CallExpr { func, args }),
    })
}

/// A call names its callee and nothing else: the types of a helper are its
/// signature's.
fn refuse_type_arguments_on_a_call(segment: &syn::PathSegment) -> syn::Error {
    syn::Error::new_spanned(
        &segment.arguments,
        format!(
            "type arguments on a call in a kernel body: `{}`\n\
             \n\
             note: a helper's parameter and return types are written once, in its \
             `fn` signature; a call passes its arguments and names no type",
            quote::quote!(#segment)
        ),
    )
}

/// A qualified callee names something outside the block.
fn refuse_a_qualified_call(expr_call: &syn::ExprCall) -> syn::Error {
    syn::Error::new_spanned(
        &expr_call.func,
        format!(
            "a qualified call in a kernel body: `{}`\n\
             \n\
             note: a body calls the block's helpers (private `fn`s) and the \
             projections V, DX, DY, DXX, DXY, DYY, by name; an operation on a value is \
             a method, `x.sqrt()`",
            quote::quote!(#expr_call)
        ),
    )
}

/// A binary operator, or the refusal that names it. `%` has no IR op, and a
/// compound assignment assigns.
fn binary_op(op: &syn::BinOp) -> syn::Result<BinaryOp> {
    let message = match op {
        syn::BinOp::Add(_) => return Ok(BinaryOp::Add),
        syn::BinOp::Sub(_) => return Ok(BinaryOp::Sub),
        syn::BinOp::Mul(_) => return Ok(BinaryOp::Mul),
        syn::BinOp::Div(_) => return Ok(BinaryOp::Div),
        syn::BinOp::Lt(_) => return Ok(BinaryOp::Lt),
        syn::BinOp::Le(_) => return Ok(BinaryOp::Le),
        syn::BinOp::Gt(_) => return Ok(BinaryOp::Gt),
        syn::BinOp::Ge(_) => return Ok(BinaryOp::Ge),
        syn::BinOp::Eq(_) => return Ok(BinaryOp::Eq),
        syn::BinOp::Ne(_) => return Ok(BinaryOp::Ne),
        syn::BinOp::BitAnd(_) => return Ok(BinaryOp::BitAnd),
        syn::BinOp::BitOr(_) => return Ok(BinaryOp::BitOr),
        syn::BinOp::AddAssign(_)
        | syn::BinOp::SubAssign(_)
        | syn::BinOp::MulAssign(_)
        | syn::BinOp::DivAssign(_)
        | syn::BinOp::RemAssign(_)
        | syn::BinOp::BitXorAssign(_)
        | syn::BinOp::BitAndAssign(_)
        | syn::BinOp::BitOrAssign(_)
        | syn::BinOp::ShlAssign(_)
        | syn::BinOp::ShrAssign(_) => format!(
            "`{}` assigns, and a kernel body assigns nothing\n\
             \n\
             note: a kernel body binds a name once, with `let`; nothing is mutable",
            quote::quote!(#op)
        ),
        syn::BinOp::Rem(_) => "`%` has no IR op\n\
             \n\
             help: `a - (a / b).floor() * b` is the remainder with the floor's sign"
            .to_string(),
        _ => format!(
            "unsupported binary operator `{}`\n\
             \n\
             note: the kernel! macro supports these binary operators:\n\
             note:   arithmetic: + - * /\n\
             note:   comparison: < <= > >= == != (a `bool`)\n\
             note:   on `bool`s: & |",
            quote::quote!(#op)
        ),
    };
    Err(syn::Error::new_spanned(op, message))
}

/// A unary operator, or the refusal that names it. `!` has no IR op.
fn unary_op(op: &syn::UnOp) -> syn::Result<UnaryOp> {
    let message = match op {
        syn::UnOp::Neg(_) => return Ok(UnaryOp::Neg),
        syn::UnOp::Not(_) => "`!` has no IR op yet\n\
             \n\
             help: write the comparison that means the negation, or swap the arms of the `if`"
            .to_string(),
        _ => format!(
            "unsupported unary operator `{}`\n\
             \n\
             note: the kernel! macro supports `-` (negation); for other unary operations, \
             use method calls like .abs()",
            quote::quote!(#op)
        ),
    };
    Err(syn::Error::new_spanned(op, message))
}
