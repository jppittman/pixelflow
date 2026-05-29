//! Bridge between macro AST and pixelflow-ir.
//!
//! This module handles conversions between:
//! 1. Macro AST → arena IR
//! 2. arena IR → runtime construction code
//!
//! The IR becomes the canonical representation, with AST only used during parsing.

use crate::ast::{BinaryOp, Expr, UnaryOp};
use pixelflow_ir::OpKind;
use pixelflow_ir::arena::{ExprArena, ExprId};
use proc_macro2::TokenStream;
use quote::quote;
use std::collections::HashMap;
use syn::Lit;

// ============================================================================
// AST → Arena IR Conversion
// ============================================================================

/// Build a `param_name → index` map from an analyzed kernel.
///
/// Index is declaration order: first scalar param = 0, second = 1, etc.
/// Only scalar params are included — manifold params cannot be constant-folded.
pub fn scalar_param_indices(analyzed: &crate::sema::AnalyzedKernel) -> HashMap<String, u8> {
    analyzed
        .def
        .params
        .iter()
        .enumerate()
        .filter_map(|(i, p)| match &p.kind {
            crate::ast::ParamKind::Scalar(_) => Some((p.name.to_string(), i as u8)),
            crate::ast::ParamKind::Manifold => None,
        })
        .collect()
}

/// Convert macro AST to an arena-allocated IR.
///
/// Mirrors [`ast_to_ir`] exactly but pushes nodes into `arena` instead of
/// heap-allocating [`Arc`] wrappers. Children are recursed first so that
/// parent nodes always reference already-interned [`ExprId`]s.
///
/// `param_indices` maps parameter names to their declaration-order index (0-based).
/// Parameter identifiers are emitted as arena `Param(i)` nodes.
pub fn ast_to_arena(
    expr: &Expr,
    param_indices: &HashMap<String, u8>,
    arena: &mut ExprArena,
) -> Result<ExprId, String> {
    let mut locals: HashMap<String, ExprId> = HashMap::new();
    ast_to_arena_inner(expr, param_indices, &mut locals, arena)
}

/// Translate an AST node into the arena, resolving `let`-bound locals via
/// `locals`. The optimizer emits `let`-bindings (a [`Expr::Block`]) for shared
/// subexpressions; each binding maps to a single [`ExprId`], so the arena
/// faithfully preserves the discovered CSE as a DAG rather than duplicating
/// subtrees.
fn ast_to_arena_inner(
    expr: &Expr,
    param_indices: &HashMap<String, u8>,
    locals: &mut HashMap<String, ExprId>,
    arena: &mut ExprArena,
) -> Result<ExprId, String> {
    match expr {
        Expr::Ident(ident) => {
            let name = ident.name.to_string();
            match name.as_str() {
                "X" => Ok(arena.push_var(0)),
                "Y" => Ok(arena.push_var(1)),
                "Z" => Ok(arena.push_var(2)),
                "W" => Ok(arena.push_var(3)),
                _ => {
                    if let Some(&id) = locals.get(&name) {
                        Ok(id)
                    } else if let Some(&idx) = param_indices.get(&name) {
                        Ok(arena.push_param(idx))
                    } else {
                        Err(format!("Unknown identifier: {}", name))
                    }
                }
            }
        }

        Expr::Literal(lit) => {
            if let Some(val) = extract_f64_from_lit(&lit.lit) {
                Ok(arena.push_const(val as f32))
            } else {
                Err(format!("Non-numeric literal"))
            }
        }

        Expr::Binary(binary) => {
            let lhs = ast_to_arena_inner(&binary.lhs, param_indices, locals, arena)?;
            let rhs = ast_to_arena_inner(&binary.rhs, param_indices, locals, arena)?;

            let op = match binary.op {
                BinaryOp::Add => OpKind::Add,
                BinaryOp::Sub => OpKind::Sub,
                BinaryOp::Mul => OpKind::Mul,
                BinaryOp::Div => OpKind::Div,
                BinaryOp::Lt => OpKind::Lt,
                BinaryOp::Le => OpKind::Le,
                BinaryOp::Gt => OpKind::Gt,
                BinaryOp::Ge => OpKind::Ge,
                BinaryOp::Eq => OpKind::Eq,
                BinaryOp::Ne => OpKind::Ne,
                _ => return Err(format!("Unsupported binary op: {:?}", binary.op)),
            };

            Ok(arena.push_binary(op, lhs, rhs))
        }

        Expr::Unary(unary) => {
            let operand = ast_to_arena_inner(&unary.operand, param_indices, locals, arena)?;

            let op = match unary.op {
                UnaryOp::Neg => OpKind::Neg,
                UnaryOp::Not => return Err(format!("Unsupported unary op: Not")),
            };

            Ok(arena.push_unary(op, operand))
        }

        Expr::MethodCall(call) => {
            let method = call.method.to_string();
            let receiver = ast_to_arena_inner(&call.receiver, param_indices, locals, arena)?;

            match (method.as_str(), call.args.len()) {
                // Unary methods - primitives
                ("sqrt", 0) => Ok(arena.push_unary(OpKind::Sqrt, receiver)),
                ("abs", 0) => Ok(arena.push_unary(OpKind::Abs, receiver)),
                ("neg", 0) => Ok(arena.push_unary(OpKind::Neg, receiver)),
                ("floor", 0) => Ok(arena.push_unary(OpKind::Floor, receiver)),
                ("ceil", 0) => Ok(arena.push_unary(OpKind::Ceil, receiver)),
                ("recip", 0) => Ok(arena.push_unary(OpKind::Recip, receiver)),
                ("rsqrt", 0) => Ok(arena.push_unary(OpKind::Rsqrt, receiver)),

                // Unary methods - transcendentals (lowered before JIT)
                ("sin", 0) => Ok(arena.push_unary(OpKind::Sin, receiver)),
                ("cos", 0) => Ok(arena.push_unary(OpKind::Cos, receiver)),
                ("tan", 0) => Ok(arena.push_unary(OpKind::Tan, receiver)),
                ("exp", 0) => Ok(arena.push_unary(OpKind::Exp, receiver)),
                ("exp2", 0) => Ok(arena.push_unary(OpKind::Exp2, receiver)),
                ("ln", 0) => Ok(arena.push_unary(OpKind::Ln, receiver)),
                ("log2", 0) => Ok(arena.push_unary(OpKind::Log2, receiver)),

                // Unary methods - inverse trigonometric
                ("atan", 0) => Ok(arena.push_unary(OpKind::Atan, receiver)),
                ("asin", 0) => Ok(arena.push_unary(OpKind::Asin, receiver)),
                ("acos", 0) => Ok(arena.push_unary(OpKind::Acos, receiver)),

                // Binary methods
                ("min", 1) => {
                    let arg = ast_to_arena_inner(&call.args[0], param_indices, locals, arena)?;
                    Ok(arena.push_binary(OpKind::Min, receiver, arg))
                }
                ("max", 1) => {
                    let arg = ast_to_arena_inner(&call.args[0], param_indices, locals, arena)?;
                    Ok(arena.push_binary(OpKind::Max, receiver, arg))
                }
                ("atan2", 1) => {
                    let arg = ast_to_arena_inner(&call.args[0], param_indices, locals, arena)?;
                    Ok(arena.push_binary(OpKind::Atan2, receiver, arg))
                }

                // Ternary methods
                ("mul_add", 2) => {
                    let b = ast_to_arena_inner(&call.args[0], param_indices, locals, arena)?;
                    let c = ast_to_arena_inner(&call.args[1], param_indices, locals, arena)?;
                    Ok(arena.push_ternary(OpKind::MulAdd, receiver, b, c))
                }
                ("select", 2) => {
                    let if_true = ast_to_arena_inner(&call.args[0], param_indices, locals, arena)?;
                    let if_false = ast_to_arena_inner(&call.args[1], param_indices, locals, arena)?;
                    Ok(arena.push_ternary(OpKind::Select, receiver, if_true, if_false))
                }
                ("clamp", 2) => {
                    let lo = ast_to_arena_inner(&call.args[0], param_indices, locals, arena)?;
                    let hi = ast_to_arena_inner(&call.args[1], param_indices, locals, arena)?;
                    Ok(arena.push_ternary(OpKind::Clamp, receiver, lo, hi))
                }

                // Comparison methods (emitted by e-graph extraction)
                ("lt", 1) => {
                    let a = ast_to_arena_inner(&call.args[0], param_indices, locals, arena)?;
                    Ok(arena.push_binary(OpKind::Lt, receiver, a))
                }
                ("le", 1) => {
                    let a = ast_to_arena_inner(&call.args[0], param_indices, locals, arena)?;
                    Ok(arena.push_binary(OpKind::Le, receiver, a))
                }
                ("gt", 1) => {
                    let a = ast_to_arena_inner(&call.args[0], param_indices, locals, arena)?;
                    Ok(arena.push_binary(OpKind::Gt, receiver, a))
                }
                ("ge", 1) => {
                    let a = ast_to_arena_inner(&call.args[0], param_indices, locals, arena)?;
                    Ok(arena.push_binary(OpKind::Ge, receiver, a))
                }
                ("eq", 1) => {
                    let a = ast_to_arena_inner(&call.args[0], param_indices, locals, arena)?;
                    Ok(arena.push_binary(OpKind::Eq, receiver, a))
                }
                ("ne", 1) => {
                    let a = ast_to_arena_inner(&call.args[0], param_indices, locals, arena)?;
                    Ok(arena.push_binary(OpKind::Ne, receiver, a))
                }

                _ => Err(format!("Unsupported method: {}", method)),
            }
        }

        // Parentheses are transparent - just recurse into the inner expression
        Expr::Paren(inner) => ast_to_arena_inner(inner, param_indices, locals, arena),

        // Blocks carry the optimizer's CSE: each `let __n = <expr>;` binds a
        // shared subexpression to a single arena node, and the final expression
        // references those bindings by name.
        Expr::Block(block) => {
            for stmt in &block.stmts {
                match stmt {
                    crate::ast::Stmt::Let(let_stmt) => {
                        let id = ast_to_arena_inner(&let_stmt.init, param_indices, locals, arena)?;
                        locals.insert(let_stmt.name.to_string(), id);
                    }
                    // A non-binding statement has no value to thread; evaluate
                    // it so any nested error surfaces, then discard the id.
                    crate::ast::Stmt::Expr(e) => {
                        let _ = ast_to_arena_inner(e, param_indices, locals, arena)?;
                    }
                }
            }
            match &block.expr {
                Some(final_expr) => ast_to_arena_inner(final_expr, param_indices, locals, arena),
                None => Err(format!("Block has no final expression")),
            }
        }

        _ => Err(format!("Unsupported expression type")),
    }
}

/// Generate runtime arena-construction code from macro AST.
pub fn ast_to_runtime_arena(
    expr: &Expr,
    param_indices: &HashMap<String, u8>,
) -> Result<TokenStream, String> {
    let mut arena = ExprArena::new();
    let root = ast_to_arena(expr, param_indices, &mut arena)?;
    let nodes = arena.nodes_raw();
    let nary_children = arena.nary_children_raw();

    let node_tokens: Vec<TokenStream> = nodes
        .iter()
        .map(|node| match node {
            pixelflow_ir::arena::ExprNode::Var(i) => {
                quote! { ::pixelflow_ir::arena::ExprNode::Var(#i) }
            }
            pixelflow_ir::arena::ExprNode::Const(v) => {
                quote! { ::pixelflow_ir::arena::ExprNode::Const(#v) }
            }
            pixelflow_ir::arena::ExprNode::Param(i) => {
                quote! { ::pixelflow_ir::arena::ExprNode::Param(#i) }
            }
            pixelflow_ir::arena::ExprNode::Unary(op, child) => {
                let op_code = opkind_to_tokens(*op);
                let child = child.0;
                quote! { ::pixelflow_ir::arena::ExprNode::Unary(#op_code, ::pixelflow_ir::arena::ExprId(#child)) }
            }
            pixelflow_ir::arena::ExprNode::Binary(op, a, b) => {
                let op_code = opkind_to_tokens(*op);
                let a = a.0;
                let b = b.0;
                quote! { ::pixelflow_ir::arena::ExprNode::Binary(#op_code, ::pixelflow_ir::arena::ExprId(#a), ::pixelflow_ir::arena::ExprId(#b)) }
            }
            pixelflow_ir::arena::ExprNode::Ternary(op, a, b, c) => {
                let op_code = opkind_to_tokens(*op);
                let a = a.0;
                let b = b.0;
                let c = c.0;
                quote! { ::pixelflow_ir::arena::ExprNode::Ternary(#op_code, ::pixelflow_ir::arena::ExprId(#a), ::pixelflow_ir::arena::ExprId(#b), ::pixelflow_ir::arena::ExprId(#c)) }
            }
            pixelflow_ir::arena::ExprNode::Nary(op, start, len) => {
                let op_code = opkind_to_tokens(*op);
                quote! { ::pixelflow_ir::arena::ExprNode::Nary(#op_code, #start, #len) }
            }
        })
        .collect();

    let child_tokens: Vec<TokenStream> = nary_children
        .iter()
        .map(|id| {
            let id = id.0;
            quote! { ::pixelflow_ir::arena::ExprId(#id) }
        })
        .collect();

    let root = root.0;
    Ok(quote! {{
        let __nodes = vec![#(#node_tokens),*];
        let __nary_children = vec![#(#child_tokens),*];
        let __arena = ::pixelflow_ir::arena::ExprArena::from_raw(__nodes, __nary_children);
        (__arena, ::pixelflow_ir::arena::ExprId(#root))
    }})
}

/// Extract f64 from a syn::Lit.
fn extract_f64_from_lit(lit: &Lit) -> Option<f64> {
    match lit {
        Lit::Float(f) => f.base10_parse::<f64>().ok(),
        Lit::Int(i) => i.base10_parse::<i64>().ok().map(|v| v as f64),
        _ => None,
    }
}

/// Map OpKind to its token representation.
fn opkind_to_tokens(kind: OpKind) -> TokenStream {
    match kind {
        OpKind::Add => quote! { ::pixelflow_ir::OpKind::Add },
        OpKind::Sub => quote! { ::pixelflow_ir::OpKind::Sub },
        OpKind::Mul => quote! { ::pixelflow_ir::OpKind::Mul },
        OpKind::Div => quote! { ::pixelflow_ir::OpKind::Div },
        OpKind::Neg => quote! { ::pixelflow_ir::OpKind::Neg },
        OpKind::Sqrt => quote! { ::pixelflow_ir::OpKind::Sqrt },
        OpKind::Rsqrt => quote! { ::pixelflow_ir::OpKind::Rsqrt },
        OpKind::Recip => quote! { ::pixelflow_ir::OpKind::Recip },
        OpKind::Abs => quote! { ::pixelflow_ir::OpKind::Abs },
        OpKind::Min => quote! { ::pixelflow_ir::OpKind::Min },
        OpKind::Max => quote! { ::pixelflow_ir::OpKind::Max },
        OpKind::MulAdd => quote! { ::pixelflow_ir::OpKind::MulAdd },
        OpKind::Sin => quote! { ::pixelflow_ir::OpKind::Sin },
        OpKind::Cos => quote! { ::pixelflow_ir::OpKind::Cos },
        OpKind::Atan => quote! { ::pixelflow_ir::OpKind::Atan },
        OpKind::Asin => quote! { ::pixelflow_ir::OpKind::Asin },
        OpKind::Acos => quote! { ::pixelflow_ir::OpKind::Acos },
        OpKind::Atan2 => quote! { ::pixelflow_ir::OpKind::Atan2 },
        OpKind::Tan => quote! { ::pixelflow_ir::OpKind::Tan },
        OpKind::Exp => quote! { ::pixelflow_ir::OpKind::Exp },
        OpKind::Exp2 => quote! { ::pixelflow_ir::OpKind::Exp2 },
        OpKind::Ln => quote! { ::pixelflow_ir::OpKind::Ln },
        OpKind::Log2 => quote! { ::pixelflow_ir::OpKind::Log2 },
        OpKind::Log10 => quote! { ::pixelflow_ir::OpKind::Log10 },
        OpKind::Pow => quote! { ::pixelflow_ir::OpKind::Pow },
        OpKind::Hypot => quote! { ::pixelflow_ir::OpKind::Hypot },
        OpKind::Floor => quote! { ::pixelflow_ir::OpKind::Floor },
        OpKind::Ceil => quote! { ::pixelflow_ir::OpKind::Ceil },
        OpKind::Round => quote! { ::pixelflow_ir::OpKind::Round },
        OpKind::Fract => quote! { ::pixelflow_ir::OpKind::Fract },
        OpKind::Lt => quote! { ::pixelflow_ir::OpKind::Lt },
        OpKind::Le => quote! { ::pixelflow_ir::OpKind::Le },
        OpKind::Gt => quote! { ::pixelflow_ir::OpKind::Gt },
        OpKind::Ge => quote! { ::pixelflow_ir::OpKind::Ge },
        OpKind::Eq => quote! { ::pixelflow_ir::OpKind::Eq },
        OpKind::Ne => quote! { ::pixelflow_ir::OpKind::Ne },
        OpKind::Select => quote! { ::pixelflow_ir::OpKind::Select },
        OpKind::Clamp => quote! { ::pixelflow_ir::OpKind::Clamp },
        _ => panic!("Unsupported OpKind for JIT: {:?}", kind),
    }
}
