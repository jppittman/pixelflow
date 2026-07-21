//! # Expression Annotation Pass
//!
//! Walks the AST and resolves each literal's `Var` binding index, writing it
//! into the literal's `var_index` field. This is a pure functional pass over the
//! single [`Expr`] representation — there is no separate annotated tree.
//!
//! ## Design
//!
//! The annotation pass threads context through return values (functional state):
//! ```text
//! annotate(expr, ctx) -> (Expr, ctx')
//! ```
//!
//! This avoids mutation while still tracking state (the literal counter).
//!
//! ## Why Annotate?
//!
//! In Jet domains (Jet3, Jet2), literals cannot be inlined as actual Jet values
//! because they'd break the ZST expression tree. Instead, literals become
//! `Var<N>` references bound via `Let`. The annotation pass assigns each literal
//! its Var index (stored in [`LiteralExpr::var_index`]).

use crate::ast::{
    BinaryExpr, BinaryOp, BlockExpr, CallExpr, Expr, LetStmt, LiteralExpr, MethodCallExpr, Stmt,
    TupleExpr, UnaryExpr,
};
use std::collections::HashMap;
use syn::Lit;

/// Context threaded through the annotation pass.
#[derive(Clone)]
pub struct AnnotationCtx {
    /// Next literal index to assign (0, 1, 2, ...).
    /// These are "collection order" indices, inverted at emit time.
    pub next_literal: usize,
}

impl AnnotationCtx {
    pub fn new() -> Self {
        Self { next_literal: 0 }
    }
}

/// Which type "space" a literal lives in.
///
/// Kernel bodies that use the derivative projections `V`/`DX`/`DY`/`DZ` mix
/// two spaces: **domain space** (values of the kernel's scalar/domain type,
/// e.g. `Jet2`) and **projected space** (`Field` values produced by the
/// projections). The projections are the boundary: domain-space in,
/// Field-space out.
///
/// Literals are polymorphic at the source level, but codegen must pick a
/// concrete type: domain-space literals are lifted to the kernel's scalar
/// type (via the `A0` context array), while projected-space literals must
/// evaluate to `Field` — otherwise e-graph rewrites that introduce fresh
/// literals into Field-space math (e.g. `a/b + c -> mul_add(a, 1/b, c)`)
/// emit ill-typed `Jet2 op Field` expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiteralSpace {
    /// Lifted to the kernel's domain scalar type (the default).
    Domain,
    /// Field-space: the literal participates in post-`V`/`DX`/`DY` math and
    /// is emitted re-projected (`V(CtxVar...)`) so it evaluates to `Field`.
    Projected,
}

/// Collected literal for Let binding generation.
#[derive(Debug, Clone)]
pub struct CollectedLiteral {
    pub index: usize,
    pub lit: Lit,
    /// The inferred type space this literal must be emitted in.
    pub space: LiteralSpace,
}

/// Annotate an expression tree, resolving literal Var indices in place.
///
/// This is a pure function — context flows through return values, and the input
/// is cloned (the returned [`Expr`] has `var_index` populated on its literals).
pub fn annotate(expr: &Expr, ctx: AnnotationCtx) -> (Expr, AnnotationCtx, Vec<CollectedLiteral>) {
    let mut literals = Vec::new();
    let (annotated, final_ctx) = annotate_expr(expr, ctx, &mut literals);

    // Infer each literal's type space from the annotated tree (literals now
    // carry their collection index) and record it for codegen.
    let spaces = infer_literal_spaces(&annotated, literals.len());
    for lit in &mut literals {
        lit.space = spaces[lit.index];
    }

    (annotated, final_ctx, literals)
}

// ============================================================================
// Literal Space Inference
// ============================================================================

/// Internal lattice for bottom-up space resolution.
///
/// `Unknown` means "not constrained by any leaf yet" (literals, opaque calls,
/// parameters). `Domain` and `Projected` correspond to [`LiteralSpace`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Space {
    Unknown,
    Domain,
    Projected,
}

/// Join two operand spaces at a type-homogeneous operator.
///
/// `Unknown` is the identity. A `Domain`/`Projected` mix would already be
/// ill-typed in the source (there are no mixed `Jet op Field` operators), so
/// the choice is arbitrary — the generated code fails to compile loudly
/// either way. `Projected` wins so a fresh literal at least stays with the
/// Field-space operand that most likely constrained the rewrite.
fn join(a: Space, b: Space) -> Space {
    match (a, b) {
        (Space::Unknown, s) | (s, Space::Unknown) => s,
        (Space::Projected, _) | (_, Space::Projected) => Space::Projected,
        (Space::Domain, Space::Domain) => Space::Domain,
    }
}

/// Is this call one of the derivative projections (domain-space in,
/// Field-space out)?
fn is_projection(func: &str) -> bool {
    matches!(func, "V" | "DX" | "DY" | "DZ" | "DXX" | "DXY" | "DYY")
}

/// Infer the [`LiteralSpace`] for every collected literal in an annotated
/// expression.
///
/// Bottom-up pass: leaves with known space (coordinates → `Domain`,
/// `V`/`DX`/`DY`/`DZ` projections → `Projected`, locals → the space of their
/// init) propagate through type-homogeneous operators. Wherever an operator
/// resolves to a known space, that space is pushed down onto any literals in
/// its still-unresolved operands. Literals never constrained by a
/// Field-space sibling default to `Domain` — exactly the pre-existing
/// behavior, so kernels without projections are unaffected.
fn infer_literal_spaces(expr: &Expr, literal_count: usize) -> Vec<LiteralSpace> {
    let mut inference = SpaceInference {
        spaces: vec![None; literal_count],
        locals: HashMap::new(),
    };
    inference.resolve(expr);
    inference
        .spaces
        .into_iter()
        .map(|s| s.unwrap_or(LiteralSpace::Domain))
        .collect()
}

struct SpaceInference {
    /// Per-collection-index space assignment (None = still unconstrained).
    spaces: Vec<Option<LiteralSpace>>,
    /// Space of each let-bound local, keyed by name.
    locals: HashMap<String, Space>,
}

impl SpaceInference {
    /// Resolve the space of `expr`, assigning spaces to literals wherever an
    /// operator join produces a known space.
    fn resolve(&mut self, expr: &Expr) -> Space {
        match expr {
            Expr::Literal(_) => Space::Unknown,

            Expr::Ident(ident) => {
                let name = ident.name.to_string();
                match name.as_str() {
                    // Coordinate variables evaluate to the domain scalar type.
                    "X" | "Y" | "Z" | "W" => Space::Domain,
                    // Locals carry the space of their init expression.
                    // Anything else (scalar params, manifold params) is
                    // unconstraining: scalar params are domain-lifted, but
                    // `Unknown` + the `Domain` default yields the same
                    // emission, without guessing about manifold outputs.
                    _ => self.locals.get(&name).copied().unwrap_or(Space::Unknown),
                }
            }

            Expr::Call(call) => {
                let func = call.func.to_string();
                // Resolve args for their internal literals either way.
                for arg in &call.args {
                    self.resolve(arg);
                }
                if is_projection(&func) {
                    // Projection boundary: argument is domain-space (its
                    // literals keep the Domain default), result is Field.
                    Space::Projected
                } else {
                    Space::Unknown
                }
            }

            Expr::Binary(binary) => {
                let l = self.resolve(&binary.lhs);
                let r = self.resolve(&binary.rhs);
                let joined = join(l, r);
                if joined != Space::Unknown {
                    self.force(&binary.lhs, joined);
                    self.force(&binary.rhs, joined);
                }
                match binary.op {
                    BinaryOp::Add
                    | BinaryOp::Sub
                    | BinaryOp::Mul
                    | BinaryOp::Div
                    | BinaryOp::Rem => joined,
                    // Comparisons and mask combinators constrain their
                    // operands (joined above) but produce a mask, which does
                    // not carry either value space outward.
                    BinaryOp::Lt
                    | BinaryOp::Le
                    | BinaryOp::Gt
                    | BinaryOp::Ge
                    | BinaryOp::Eq
                    | BinaryOp::Ne
                    | BinaryOp::BitAnd
                    | BinaryOp::BitOr => Space::Unknown,
                }
            }

            Expr::Unary(unary) => self.resolve(&unary.operand),

            Expr::MethodCall(call) => {
                let method = call.method.to_string();
                match method.as_str() {
                    // `.clone()` preserves the receiver's space.
                    "clone" => self.resolve(&call.receiver),

                    // `select(mask, a, b)`: the mask is an independent
                    // island; the value space is the join of the branches.
                    "select" => {
                        self.resolve(&call.receiver);
                        let mut joined = Space::Unknown;
                        for arg in &call.args {
                            joined = join(joined, self.resolve(arg));
                        }
                        if joined != Space::Unknown {
                            for arg in &call.args {
                                self.force(arg, joined);
                            }
                        }
                        joined
                    }

                    // `.at()` re-maps the domain; coordinates keep their own
                    // spaces and the output is the inner manifold's, which we
                    // cannot see. Resolve children, propagate nothing.
                    "at" => {
                        self.resolve(&call.receiver);
                        for arg in &call.args {
                            self.resolve(arg);
                        }
                        Space::Unknown
                    }

                    // Comparison methods: operands share a space, output is
                    // a mask.
                    "lt" | "le" | "gt" | "ge" | "eq" | "ne" => {
                        self.join_and_force(call);
                        Space::Unknown
                    }

                    // Everything else (sqrt, abs, min, max, mul_add, clamp,
                    // trig, ...) is type-homogeneous: receiver, args, and
                    // output all share one space.
                    _ => self.join_and_force(call),
                }
            }

            Expr::Paren(inner) => self.resolve(inner),

            Expr::Block(block) => {
                for stmt in &block.stmts {
                    match stmt {
                        Stmt::Let(let_stmt) => {
                            let space = self.resolve(&let_stmt.init);
                            self.locals.insert(let_stmt.name.to_string(), space);
                        }
                        Stmt::Expr(e) => {
                            self.resolve(e);
                        }
                    }
                }
                match &block.expr {
                    Some(e) => self.resolve(e),
                    None => Space::Unknown,
                }
            }

            Expr::Tuple(tuple) => {
                for elem in &tuple.elems {
                    self.resolve(elem);
                }
                Space::Unknown
            }

            Expr::Verbatim(_) => Space::Unknown,
        }
    }

    /// Join receiver + args of a type-homogeneous method call and force the
    /// joined space onto unresolved operands.
    fn join_and_force(&mut self, call: &MethodCallExpr) -> Space {
        let mut joined = self.resolve(&call.receiver);
        for arg in &call.args {
            joined = join(joined, self.resolve(arg));
        }
        if joined != Space::Unknown {
            self.force(&call.receiver, joined);
            for arg in &call.args {
                self.force(arg, joined);
            }
        }
        joined
    }

    /// Push a resolved space down into `expr`, assigning it to any literal
    /// that is not yet constrained. Recursion stays within type-homogeneous
    /// structure: it does not cross projection calls (`V(...)` internals are
    /// domain-space), `.at()` coordinate lists, opaque calls, or blocks
    /// (locals were resolved in order already). Assignments are first-wins,
    /// so literals already fixed by an inner join are untouched.
    fn force(&mut self, expr: &Expr, space: Space) {
        let assign = match space {
            Space::Domain => LiteralSpace::Domain,
            Space::Projected => LiteralSpace::Projected,
            Space::Unknown => return,
        };
        match expr {
            Expr::Literal(lit) => {
                if let Some(idx) = lit.var_index {
                    if let Some(slot) = self.spaces.get_mut(idx) {
                        if slot.is_none() {
                            *slot = Some(assign);
                        }
                    }
                }
            }
            Expr::Binary(binary) => {
                self.force(&binary.lhs, space);
                self.force(&binary.rhs, space);
            }
            Expr::Unary(unary) => self.force(&unary.operand, space),
            Expr::Paren(inner) => self.force(inner, space),
            Expr::Tuple(tuple) => {
                for elem in &tuple.elems {
                    self.force(elem, space);
                }
            }
            Expr::MethodCall(call) => match call.method.to_string().as_str() {
                "at" => {}
                "select" => {
                    for arg in &call.args {
                        self.force(arg, space);
                    }
                }
                _ => {
                    self.force(&call.receiver, space);
                    for arg in &call.args {
                        self.force(arg, space);
                    }
                }
            },
            // Boundaries: projections/opaque calls, idents, blocks, verbatim.
            Expr::Call(_) | Expr::Ident(_) | Expr::Block(_) | Expr::Verbatim(_) => {}
        }
    }
}

fn annotate_expr(
    expr: &Expr,
    ctx: AnnotationCtx,
    literals: &mut Vec<CollectedLiteral>,
) -> (Expr, AnnotationCtx) {
    match expr {
        Expr::Ident(ident) => (Expr::Ident(ident.clone()), ctx),

        Expr::Literal(lit_expr) => {
            // Always assign a collection-order index to literals.
            // This makes literals into expression tree nodes (Var<N>), enabling:
            // 1. `0.1 + expr` to work (both sides are expression trees)
            // 2. FMA pattern matching (MulAdd recognition)
            // 3. Uniform treatment across Field and Jet domains
            //
            // The actual Var<N> index is computed at emit time by inverting:
            // Var index = (literal_count - 1) - collection_index
            // This ensures the last literal bound (innermost Let) is at N0.
            let collection_index = ctx.next_literal;
            literals.push(CollectedLiteral {
                index: collection_index,
                lit: lit_expr.lit.clone(),
                // Refined by space inference in `annotate` once the whole
                // tree is annotated.
                space: LiteralSpace::Domain,
            });
            let new_ctx = AnnotationCtx {
                next_literal: ctx.next_literal + 1,
            };
            (
                Expr::Literal(LiteralExpr {
                    lit: lit_expr.lit.clone(),
                    span: lit_expr.span,
                    var_index: Some(collection_index),
                }),
                new_ctx,
            )
        }

        Expr::Binary(binary) => {
            let (lhs, ctx1) = annotate_expr(&binary.lhs, ctx, literals);
            let (rhs, ctx2) = annotate_expr(&binary.rhs, ctx1, literals);
            (
                Expr::Binary(BinaryExpr {
                    op: binary.op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                    span: binary.span,
                }),
                ctx2,
            )
        }

        Expr::Unary(unary) => {
            let (operand, ctx1) = annotate_expr(&unary.operand, ctx, literals);
            (
                Expr::Unary(UnaryExpr {
                    op: unary.op,
                    operand: Box::new(operand),
                    span: unary.span,
                }),
                ctx1,
            )
        }

        Expr::MethodCall(call) => {
            let (receiver, mut ctx1) = annotate_expr(&call.receiver, ctx, literals);
            let mut args = Vec::with_capacity(call.args.len());
            for arg in &call.args {
                let (annotated_arg, new_ctx) = annotate_expr(arg, ctx1, literals);
                args.push(annotated_arg);
                ctx1 = new_ctx;
            }
            (
                Expr::MethodCall(MethodCallExpr {
                    receiver: Box::new(receiver),
                    method: call.method.clone(),
                    args,
                    span: call.span,
                }),
                ctx1,
            )
        }

        Expr::Tuple(tuple) => {
            let mut ctx1 = ctx;
            let mut elems = Vec::with_capacity(tuple.elems.len());
            for elem in &tuple.elems {
                let (annotated_elem, new_ctx) = annotate_expr(elem, ctx1, literals);
                elems.push(annotated_elem);
                ctx1 = new_ctx;
            }
            (
                Expr::Tuple(TupleExpr {
                    elems,
                    span: tuple.span,
                }),
                ctx1,
            )
        }

        Expr::Block(block) => {
            let (annotated_block, ctx1) = annotate_block(block, ctx, literals);
            (Expr::Block(annotated_block), ctx1)
        }

        Expr::Paren(inner) => {
            let (annotated, ctx1) = annotate_expr(inner, ctx, literals);
            (Expr::Paren(Box::new(annotated)), ctx1)
        }

        Expr::Call(call) => {
            // Annotate all arguments (this is where manifold param rewriting happens)
            let mut ctx1 = ctx;
            let mut args = Vec::with_capacity(call.args.len());
            for arg in &call.args {
                let (annotated_arg, new_ctx) = annotate_expr(arg, ctx1, literals);
                args.push(annotated_arg);
                ctx1 = new_ctx;
            }
            (
                Expr::Call(CallExpr {
                    func: call.func.clone(),
                    args,
                    span: call.span,
                }),
                ctx1,
            )
        }

        Expr::Verbatim(syn_expr) => (Expr::Verbatim(syn_expr.clone()), ctx),
    }
}

fn annotate_block(
    block: &BlockExpr,
    ctx: AnnotationCtx,
    literals: &mut Vec<CollectedLiteral>,
) -> (BlockExpr, AnnotationCtx) {
    let mut current_ctx = ctx;
    let mut stmts = Vec::with_capacity(block.stmts.len());

    for stmt in &block.stmts {
        let (annotated_stmt, new_ctx) = annotate_stmt(stmt, current_ctx, literals);
        stmts.push(annotated_stmt);
        current_ctx = new_ctx;
    }

    let (final_expr, final_ctx) = match &block.expr {
        Some(e) => {
            let (annotated, ctx1) = annotate_expr(e, current_ctx, literals);
            (Some(Box::new(annotated)), ctx1)
        }
        None => (None, current_ctx),
    };

    (
        BlockExpr {
            stmts,
            expr: final_expr,
            span: block.span,
        },
        final_ctx,
    )
}

fn annotate_stmt(
    stmt: &Stmt,
    ctx: AnnotationCtx,
    literals: &mut Vec<CollectedLiteral>,
) -> (Stmt, AnnotationCtx) {
    match stmt {
        Stmt::Let(let_stmt) => {
            let (init, ctx1) = annotate_expr(&let_stmt.init, ctx, literals);
            (
                Stmt::Let(Box::new(LetStmt {
                    name: let_stmt.name.clone(),
                    ty: let_stmt.ty.clone(),
                    init,
                    span: let_stmt.span,
                })),
                ctx1,
            )
        }
        Stmt::Expr(expr) => {
            let (annotated, ctx1) = annotate_expr(expr, ctx, literals);
            (Stmt::Expr(annotated), ctx1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;
    use quote::quote;

    #[test]
    fn annotate_no_literals() {
        let input = quote! { || X + Y };
        let kernel = parse(input).unwrap();
        let ctx = AnnotationCtx::new();
        let (_, _, literals) = annotate(&kernel.body, ctx);
        assert!(literals.is_empty());
    }

    #[test]
    fn annotate_single_literal() {
        let input = quote! { || X + 1.0 };
        let kernel = parse(input).unwrap();
        let ctx = AnnotationCtx::new();
        let (annotated, _, literals) = annotate(&kernel.body, ctx);

        // Literals are always collected now (for both Field and Jet modes)
        assert_eq!(literals.len(), 1);
        assert_eq!(literals[0].index, 0); // collection order index

        // Check the annotated tree has the var_index
        if let Expr::Binary(binary) = annotated {
            if let Expr::Literal(lit) = &*binary.rhs {
                assert_eq!(lit.var_index, Some(0));
            } else {
                panic!("expected literal");
            }
        } else {
            panic!("expected binary");
        }
    }

    #[test]
    fn annotate_multiple_literals() {
        let input = quote! { || 1.0 / X.sqrt() + 2.0 };
        let kernel = parse(input).unwrap();
        let ctx = AnnotationCtx::new();
        let (_, _, literals) = annotate(&kernel.body, ctx);

        assert_eq!(literals.len(), 2);
        // Collection-order indices: first literal → 0, second → 1
        // The actual Var<N> index is computed at emit time by inverting
        assert_eq!(literals[0].index, 0); // collection order
        assert_eq!(literals[1].index, 1); // collection order
    }
}
