//! Bridge between macro AST and pixelflow-ir.
//!
//! This module handles conversions between:
//! 1. Macro AST → IR (ast_to_ir)
//! 2. IR → E-graph (ir_to_egraph)
//! 3. E-graph → IR (egraph_to_ir)
//! 4. IR → Type-level code (ir_to_code)
//!
//! The IR becomes the canonical representation, with AST only used during parsing.

use crate::ast::{BinaryExpr, BinaryOp, Expr, LiteralExpr, UnaryOp};
use pixelflow_ir::{Expr as IR, OpKind};
use pixelflow_search::egraph::{EClassId, EGraph, ENode, ExprTree, Leaf, ops};
use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use std::collections::HashMap;
use syn::{Ident, Lit};

// ============================================================================
// AST → IR Conversion
// ============================================================================

/// Build a `param_name → index` map from an analyzed kernel.
///
/// Index is declaration order: first scalar param = 0, second = 1, etc.
/// Only scalar params are included — manifold params cannot be constant-folded.
pub fn scalar_param_indices(
    analyzed: &crate::sema::AnalyzedKernel,
) -> HashMap<String, u8> {
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

/// Convert macro AST to IR.
///
/// This flattens the high-level parsing structure (with source spans, etc.)
/// into the clean IR representation used for optimization.
///
/// `param_indices` maps parameter names to their declaration-order index (0-based).
/// Parameter identifiers are emitted as [`IR::Param(i)`] for later substitution.
pub fn ast_to_ir(expr: &Expr, param_indices: &HashMap<String, u8>) -> Result<IR, String> {
    match expr {
        Expr::Ident(ident) => {
            // Map coordinate variables to their indices
            let name = ident.name.to_string();
            match name.as_str() {
                "X" => Ok(IR::Var(0)),
                "Y" => Ok(IR::Var(1)),
                "Z" => Ok(IR::Var(2)),
                "W" => Ok(IR::Var(3)),
                _ => {
                    if let Some(&idx) = param_indices.get(&name) {
                        Ok(IR::Param(idx))
                    } else {
                        Err(format!("Unknown identifier: {}", name))
                    }
                }
            }
        }

        Expr::Literal(lit) => {
            if let Some(val) = extract_f64_from_lit(&lit.lit) {
                Ok(IR::Const(val as f32))
            } else {
                Err(format!("Non-numeric literal"))
            }
        }

        Expr::Binary(binary) => {
            let lhs = Box::new(ast_to_ir(&binary.lhs, param_indices)?);
            let rhs = Box::new(ast_to_ir(&binary.rhs, param_indices)?);

            let op = match binary.op {
                BinaryOp::Add => OpKind::Add,
                BinaryOp::Sub => OpKind::Sub,
                BinaryOp::Mul => OpKind::Mul,
                BinaryOp::Div => OpKind::Div,
                _ => return Err(format!("Unsupported binary op: {:?}", binary.op)),
            };

            Ok(IR::Binary(op, lhs, rhs))
        }

        Expr::Unary(unary) => {
            let operand = Box::new(ast_to_ir(&unary.operand, param_indices)?);

            let op = match unary.op {
                UnaryOp::Neg => OpKind::Neg,
                UnaryOp::Not => return Err(format!("Unsupported unary op: Not")),
            };

            Ok(IR::Unary(op, operand))
        }

        Expr::MethodCall(call) => {
            let method = call.method.to_string();
            let receiver = Box::new(ast_to_ir(&call.receiver, param_indices)?);

            match (method.as_str(), call.args.len()) {
                // Unary methods - primitives
                ("sqrt", 0) => Ok(IR::Unary(OpKind::Sqrt, receiver)),
                ("abs", 0) => Ok(IR::Unary(OpKind::Abs, receiver)),
                ("neg", 0) => Ok(IR::Unary(OpKind::Neg, receiver)),
                ("floor", 0) => Ok(IR::Unary(OpKind::Floor, receiver)),
                ("ceil", 0) => Ok(IR::Unary(OpKind::Ceil, receiver)),
                ("recip", 0) => Ok(IR::Unary(OpKind::Recip, receiver)),
                ("rsqrt", 0) => Ok(IR::Unary(OpKind::Rsqrt, receiver)),

                // Unary methods - transcendentals (lowered before JIT)
                ("sin", 0) => Ok(IR::Unary(OpKind::Sin, receiver)),
                ("cos", 0) => Ok(IR::Unary(OpKind::Cos, receiver)),
                ("tan", 0) => Ok(IR::Unary(OpKind::Tan, receiver)),
                ("exp", 0) => Ok(IR::Unary(OpKind::Exp, receiver)),
                ("exp2", 0) => Ok(IR::Unary(OpKind::Exp2, receiver)),
                ("ln", 0) => Ok(IR::Unary(OpKind::Ln, receiver)),
                ("log2", 0) => Ok(IR::Unary(OpKind::Log2, receiver)),

                // Unary methods - inverse trigonometric
                ("atan", 0) => Ok(IR::Unary(OpKind::Atan, receiver)),
                ("asin", 0) => Ok(IR::Unary(OpKind::Asin, receiver)),
                ("acos", 0) => Ok(IR::Unary(OpKind::Acos, receiver)),

                // Binary methods
                ("min", 1) => {
                    let arg = Box::new(ast_to_ir(&call.args[0], param_indices)?);
                    Ok(IR::Binary(OpKind::Min, receiver, arg))
                }
                ("max", 1) => {
                    let arg = Box::new(ast_to_ir(&call.args[0], param_indices)?);
                    Ok(IR::Binary(OpKind::Max, receiver, arg))
                }
                ("atan2", 1) => {
                    let arg = Box::new(ast_to_ir(&call.args[0], param_indices)?);
                    Ok(IR::Binary(OpKind::Atan2, receiver, arg))
                }

                // Ternary methods
                ("mul_add", 2) => {
                    let b = Box::new(ast_to_ir(&call.args[0], param_indices)?);
                    let c = Box::new(ast_to_ir(&call.args[1], param_indices)?);
                    Ok(IR::Ternary(OpKind::MulAdd, receiver, b, c))
                }

                _ => Err(format!("Unsupported method: {}", method)),
            }
        }

        // Parentheses are transparent - just recurse into the inner expression
        Expr::Paren(inner) => ast_to_ir(inner, param_indices),

        _ => Err(format!("Unsupported expression type")),
    }
}

/// Extract f64 from a syn::Lit.
fn extract_f64_from_lit(lit: &Lit) -> Option<f64> {
    match lit {
        Lit::Float(f) => f.base10_parse::<f64>().ok(),
        Lit::Int(i) => i.base10_parse::<i64>().ok().map(|v| v as f64),
        _ => None,
    }
}

// ============================================================================
// IR → E-graph Conversion (Flattening)
// ============================================================================

/// Context for flattening IR trees into E-graph.
pub struct IRToEGraphContext {
    pub egraph: EGraph,
}

impl IRToEGraphContext {
    pub fn new() -> Self {
        Self {
            egraph: EGraph::with_rules(crate::optimize::standard_rules()),
        }
    }

    /// Flatten an IR tree into the E-graph, returning the root e-class ID.
    pub fn ir_to_egraph(&mut self, ir: &IR) -> EClassId {
        match ir {
            IR::Var(idx) => self.egraph.add(ENode::Var(*idx)),

            IR::Const(val) => self.egraph.add(ENode::constant(*val)),

            IR::Param(i) => panic!(
                "Expr::Param({}) reached e-graph optimizer — call substitute_params before optimization",
                i
            ),

            IR::Unary(op, child) => {
                let child_id = self.ir_to_egraph(child);
                let op_ref = opkind_to_op(*op);
                self.egraph.add(ENode::Op {
                    op: op_ref,
                    children: vec![child_id],
                })
            }

            IR::Binary(op, lhs, rhs) => {
                let lhs_id = self.ir_to_egraph(lhs);
                let rhs_id = self.ir_to_egraph(rhs);
                let op_ref = opkind_to_op(*op);
                self.egraph.add(ENode::Op {
                    op: op_ref,
                    children: vec![lhs_id, rhs_id],
                })
            }

            IR::Ternary(op, a, b, c) => {
                let a_id = self.ir_to_egraph(a);
                let b_id = self.ir_to_egraph(b);
                let c_id = self.ir_to_egraph(c);
                let op_ref = opkind_to_op(*op);
                self.egraph.add(ENode::Op {
                    op: op_ref,
                    children: vec![a_id, b_id, c_id],
                })
            }

            IR::Nary(op, children) => {
                let child_ids: Vec<EClassId> = children
                    .iter()
                    .map(|child| self.ir_to_egraph(child))
                    .collect();
                let op_ref = opkind_to_op(*op);
                self.egraph.add(ENode::Op {
                    op: op_ref,
                    children: child_ids,
                })
            }
        }
    }
}

/// Map OpKind to a static Op trait object reference.
fn opkind_to_op(kind: OpKind) -> &'static dyn ops::Op {
    match kind {
        OpKind::Add => &ops::Add,
        OpKind::Sub => &ops::Sub,
        OpKind::Mul => &ops::Mul,
        OpKind::Div => &ops::Div,
        OpKind::Neg => &ops::Neg,
        OpKind::Sqrt => &ops::Sqrt,
        OpKind::Rsqrt => &ops::Rsqrt,
        OpKind::Recip => &ops::Recip,
        OpKind::Abs => &ops::Abs,
        OpKind::Min => &ops::Min,
        OpKind::Max => &ops::Max,
        OpKind::MulAdd => &ops::MulAdd,
        _ => panic!("Unsupported OpKind: {:?}", kind),
    }
}

// ============================================================================
// E-graph → IR Conversion (Extraction)
// ============================================================================

/// Convert an extracted ExprTree back to IR.
pub fn egraph_to_ir(tree: &ExprTree) -> IR {
    match tree {
        ExprTree::Leaf(Leaf::Var(idx)) => IR::Var(*idx),

        ExprTree::Leaf(Leaf::Const(val)) => IR::Const(*val),

        ExprTree::Op { op, children } => {
            let name = op.name();

            // Map op name back to OpKind
            let kind = match name {
                "add" => OpKind::Add,
                "sub" => OpKind::Sub,
                "mul" => OpKind::Mul,
                "div" => OpKind::Div,
                "neg" => OpKind::Neg,
                "sqrt" => OpKind::Sqrt,
                "rsqrt" => OpKind::Rsqrt,
                "recip" => OpKind::Recip,
                "abs" => OpKind::Abs,
                "min" => OpKind::Min,
                "max" => OpKind::Max,
                "mul_add" => OpKind::MulAdd,
                _ => panic!("Unknown op: {}", name),
            };

            // Convert children
            let child_irs: Vec<IR> = children.iter().map(|c| egraph_to_ir(c)).collect();

            match child_irs.len() {
                1 => IR::Unary(kind, Box::new(child_irs[0].clone())),
                2 => IR::Binary(
                    kind,
                    Box::new(child_irs[0].clone()),
                    Box::new(child_irs[1].clone()),
                ),
                3 => IR::Ternary(
                    kind,
                    Box::new(child_irs[0].clone()),
                    Box::new(child_irs[1].clone()),
                    Box::new(child_irs[2].clone()),
                ),
                _ => IR::Nary(kind, child_irs),
            }
        }
    }
}

// ============================================================================
// IR → Type-Level Code Generation
// ============================================================================

/// Generate runtime Expr constructor code from IR.
///
/// This emits Rust code that, when executed, builds the same IR tree at runtime.
/// Used by kernel_jit! to defer compilation to runtime (JIT).
pub fn ir_to_runtime_expr(ir: &IR) -> TokenStream {
    match ir {
        IR::Var(idx) => {
            quote! { ::pixelflow_ir::Expr::Var(#idx) }
        }

        IR::Const(val) => {
            quote! { ::pixelflow_ir::Expr::Const(#val) }
        }

        IR::Param(idx) => {
            quote! { ::pixelflow_ir::Expr::Param(#idx) }
        }

        IR::Unary(op, child) => {
            let child_code = ir_to_runtime_expr(child);
            let op_code = opkind_to_tokens(*op);
            quote! {
                ::pixelflow_ir::Expr::Unary(#op_code, Box::new(#child_code))
            }
        }

        IR::Binary(op, lhs, rhs) => {
            let lhs_code = ir_to_runtime_expr(lhs);
            let rhs_code = ir_to_runtime_expr(rhs);
            let op_code = opkind_to_tokens(*op);
            quote! {
                ::pixelflow_ir::Expr::Binary(#op_code, Box::new(#lhs_code), Box::new(#rhs_code))
            }
        }

        IR::Ternary(op, a, b, c) => {
            let a_code = ir_to_runtime_expr(a);
            let b_code = ir_to_runtime_expr(b);
            let c_code = ir_to_runtime_expr(c);
            let op_code = opkind_to_tokens(*op);
            quote! {
                ::pixelflow_ir::Expr::Ternary(
                    #op_code,
                    Box::new(#a_code),
                    Box::new(#b_code),
                    Box::new(#c_code),
                )
            }
        }

        IR::Nary(op, children) => {
            let child_codes: Vec<_> = children.iter().map(ir_to_runtime_expr).collect();
            let op_code = opkind_to_tokens(*op);
            quote! {
                ::pixelflow_ir::Expr::Nary(#op_code, vec![#(#child_codes),*])
            }
        }
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
        OpKind::Floor => quote! { ::pixelflow_ir::OpKind::Floor },
        OpKind::Ceil => quote! { ::pixelflow_ir::OpKind::Ceil },
        OpKind::Select => quote! { ::pixelflow_ir::OpKind::Select },
        OpKind::Clamp => quote! { ::pixelflow_ir::OpKind::Clamp },
        _ => panic!("Unsupported OpKind for JIT: {:?}", kind),
    }
}

/// Generate type-level code from IR.
///
/// This emits the type-level AST that will be monomorphized by rustc.
pub fn ir_to_code(ir: &IR) -> TokenStream {
    match ir {
        IR::Var(idx) => {
            // Map variable indices to coordinate variables
            match idx {
                0 => quote! { X },
                1 => quote! { Y },
                2 => quote! { Z },
                3 => quote! { W },
                _ => {
                    let var_name = format_ident!("v{}", idx);
                    quote! { #var_name }
                }
            }
        }

        IR::Const(val) => {
            quote! { #val }
        }

        IR::Param(i) => panic!(
            "Expr::Param({}) reached type-level codegen — Param nodes are only for kernel_jit!, not kernel!",
            i
        ),

        IR::Unary(op, child) => {
            let child_code = ir_to_code(child);
            match op {
                OpKind::Neg => quote! { Neg::new(#child_code) },
                OpKind::Sqrt => quote! { (#child_code).sqrt() },
                OpKind::Abs => quote! { (#child_code).abs() },
                OpKind::Rsqrt => quote! { (#child_code).rsqrt() },
                OpKind::Recip => quote! { (#child_code).recip() },
                _ => panic!("Unsupported unary op: {:?}", op),
            }
        }

        IR::Binary(op, lhs, rhs) => {
            let lhs_code = ir_to_code(lhs);
            let rhs_code = ir_to_code(rhs);
            match op {
                OpKind::Add => quote! { (#lhs_code) + (#rhs_code) },
                OpKind::Sub => quote! { (#lhs_code) - (#rhs_code) },
                OpKind::Mul => quote! { (#lhs_code) * (#rhs_code) },
                OpKind::Div => quote! { (#lhs_code) / (#rhs_code) },
                OpKind::Min => quote! { (#lhs_code).min(#rhs_code) },
                OpKind::Max => quote! { (#lhs_code).max(#rhs_code) },
                _ => panic!("Unsupported binary op: {:?}", op),
            }
        }

        IR::Ternary(op, a, b, c) => {
            let a_code = ir_to_code(a);
            let b_code = ir_to_code(b);
            let c_code = ir_to_code(c);
            match op {
                OpKind::MulAdd => quote! { (#a_code).mul_add(#b_code, #c_code) },
                _ => panic!("Unsupported ternary op: {:?}", op),
            }
        }

        IR::Nary(_, _children) => {
            panic!("N-ary ops not yet supported in codegen")
        }
    }
}
