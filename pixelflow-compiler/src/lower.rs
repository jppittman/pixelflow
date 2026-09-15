//! Macro AST → `ExprArena`.
//!
//! The front end's one lowering step: the surface syntax a user wrote becomes
//! the IR everything downstream speaks. `let` bindings resolve to the
//! [`ExprId`] they name, so the arena is a DAG and a shared subexpression is
//! one node; operators and DSL methods resolve through [`OpKind`], so the op
//! table is not restated here.
//!
//! Emission — arena to the `TokenStream` that rebuilds it — is [`crate::emit`].

use crate::ast::{BinaryOp, Expr, UnaryOp};
use pixelflow_ir::OpKind;
use pixelflow_ir::arena::{ExprArena, ExprId};
use std::collections::HashMap;
use syn::Lit;

/// DSL method calls that denote a fixed composition of primitive ops rather
/// than a single [`OpKind`] — `(name, arg_count)`, `arg_count` excluding the
/// receiver.
///
/// Lowering builds the composition; this list is the one place that says
/// which names and arities exist, so `sema`'s validation and lowering's
/// dispatch cannot silently drift on which library methods a kernel body may
/// call. They did once, in both directions at once — see
/// `every_advertised_method_compiles` in the crate root.
pub(crate) const LIBRARY_METHODS: &[(&str, usize)] = &[("fract", 0), ("hypot", 1), ("clamp", 2)];

/// Build a `param_name → index` map over the params of a kernel.
///
/// Indices are dense in declaration order: each becomes a `Param(i)` arena
/// node, substituted by `substitute_params` with the builder closure's
/// arguments in the same order.
pub fn param_indices(analyzed: &crate::sema::AnalyzedKernel) -> HashMap<String, u8> {
    analyzed
        .def
        .params
        .iter()
        .enumerate()
        .map(|(i, p)| (p.name.to_string(), i as u8))
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
    let mut lowering = Lowering {
        param_indices,
        locals: HashMap::new(),
        arena,
    };
    lowering.lower(expr)
}

/// State threaded through the AST → arena walk: parameter names (fixed for
/// the whole kernel), `let`-bound locals (grows as blocks are walked), and
/// the arena nodes are pushed into.
struct Lowering<'a> {
    param_indices: &'a HashMap<String, u8>,
    locals: HashMap<String, ExprId>,
    arena: &'a mut ExprArena,
}

impl Lowering<'_> {
    /// Translate an AST node into the arena, resolving `let`-bound locals via
    /// `self.locals`. The optimizer emits `let`-bindings (a [`Expr::Block`]) for
    /// shared subexpressions; each binding maps to a single [`ExprId`], so the
    /// arena faithfully preserves the discovered CSE as a DAG rather than
    /// duplicating subtrees.
    fn lower(&mut self, expr: &Expr) -> Result<ExprId, String> {
        match expr {
            Expr::Ident(ident) => {
                let name = ident.name.to_string();
                match name.as_str() {
                    "X" => Ok(self.arena.push_var(0)),
                    "Y" => Ok(self.arena.push_var(1)),
                    _ => {
                        if let Some(&id) = self.locals.get(&name) {
                            Ok(id)
                        } else if let Some(&idx) = self.param_indices.get(&name) {
                            Ok(self.arena.push_param(idx))
                        } else {
                            Err(format!("Unknown identifier: {}", name))
                        }
                    }
                }
            }

            Expr::Literal(lit) => {
                if let Some(val) = extract_f64_from_lit(&lit.lit) {
                    Ok(self.arena.push_const(val as f32))
                } else {
                    Err("Non-numeric literal".to_string())
                }
            }

            Expr::Binary(binary) => {
                let lhs = self.lower(&binary.lhs)?;
                let rhs = self.lower(&binary.rhs)?;

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
                    // Mask combination: comparison results are canonical masks in
                    // both tiers (all-ones SIMD lanes in the JIT, 1.0/0.0 in the
                    // interpreter), so bitwise AND/OR is logical AND/OR exactly.
                    BinaryOp::BitAnd => OpKind::BitAnd,
                    BinaryOp::BitOr => OpKind::BitOr,
                    _ => return Err(format!("Unsupported binary op: {:?}", binary.op)),
                };

                Ok(self.arena.push_binary(op, lhs, rhs))
            }

            Expr::Unary(unary) => {
                let operand = self.lower(&unary.operand)?;

                let op = match unary.op {
                    UnaryOp::Neg => OpKind::Neg,
                    UnaryOp::Not => return Err("Unsupported unary op: Not".to_string()),
                };

                Ok(self.arena.push_unary(op, operand))
            }

            Expr::MethodCall(call) => {
                let method = call.method.to_string();

                // `.at(x, y)` warped a manifold-typed macro param at a
                // call site. There are no manifold params: a kernel composes
                // `Kernel` values, and `Kernel::at` is the warp.
                if method == "at" {
                    return Err(
                        ".at() inside a kernel body samples a manifold param, and there are none; \
                         compose Kernel values with Kernel::at instead"
                            .to_string(),
                    );
                }

                let receiver = self.lower(&call.receiver)?;
                let arg_count = call.args.len();

                // Arena expressions are values; `.clone()` (needed by the
                // combinator backend for non-Copy trees) is the identity here,
                // so one kernel body compiles under both backends.
                if method == "clone" && arg_count == 0 {
                    return Ok(receiver);
                }

                // Primitive ops: one `OpKind` per (name, arity), read from
                // the single table `OpKind::from_method_call` resolves
                // against — not re-listed here as a second copy that could
                // silently drift from it (see `LIBRARY_METHODS` below for
                // the one part of this dispatch that table doesn't cover).
                if let Some(op) = OpKind::from_method_call(&method, arg_count) {
                    let mut args = Vec::with_capacity(arg_count);
                    for arg in &call.args {
                        args.push(self.lower(arg)?);
                    }
                    return Ok(match *args.as_slice() {
                        [] => self.arena.push_unary(op, receiver),
                        [a] => self.arena.push_binary(op, receiver, a),
                        [a, b] => self.arena.push_ternary(op, receiver, a, b),
                        _ => unreachable!(
                            "OpKind::from_method_call only resolves ops of arity 1..=3"
                        ),
                    });
                }

                match (method.as_str(), arg_count) {
                    // `fract(x) = x - floor(x)`.
                    ("fract", 0) => {
                        let f = self.arena.push_unary(OpKind::Floor, receiver);
                        Ok(self.arena.push_binary(OpKind::Sub, receiver, f))
                    }
                    // `hypot(x, y) = sqrt(x² + y²)`.
                    ("hypot", 1) => {
                        let arg = self.lower(&call.args[0])?;
                        let xx = self.arena.push_binary(OpKind::Mul, receiver, receiver);
                        let yy = self.arena.push_binary(OpKind::Mul, arg, arg);
                        let sum = self.arena.push_binary(OpKind::Add, xx, yy);
                        Ok(self.arena.push_unary(OpKind::Sqrt, sum))
                    }
                    // `clamp` is library, not a primitive: it denotes
                    // `min(max(x, lo), hi)` and is built as that composition.
                    ("clamp", 2) => {
                        let lo = self.lower(&call.args[0])?;
                        let hi = self.lower(&call.args[1])?;
                        let floored = self.arena.push_binary(OpKind::Max, receiver, lo);
                        Ok(self.arena.push_binary(OpKind::Min, floored, hi))
                    }

                    _ => Err(format!("Unsupported method: {}", method)),
                }
            }

            // Derivative projections (V/DX/DY and the Hessian family) map to
            // `Dwrt` nodes: the runtime `lower_dwrt` pass (pixelflow-ir) rewrites
            // them into chain-rule arithmetic before codegen, replacing the
            // combinator backend's Jet2/Jet3 forward-mode evaluation. `V` is the
            // identity — every arena expression is already value-space.
            Expr::Call(call) => {
                let func = call.func.to_string();
                if call.args.len() != 1 {
                    return Err(format!(
                        "Unsupported call: {}/{} (projections take one argument)",
                        func,
                        call.args.len()
                    ));
                }
                let inner = self.lower(&call.args[0])?;
                match func.as_str() {
                    "V" => Ok(inner),
                    "DX" => Ok(push_dwrt(self.arena, inner, 0)),
                    "DY" => Ok(push_dwrt(self.arena, inner, 1)),
                    "DZ" => Err(
                        "`DZ` is no longer a coordinate: a lattice has two axes, X and Y"
                            .to_string(),
                    ),
                    "DXX" => {
                        let d = push_dwrt(self.arena, inner, 0);
                        Ok(push_dwrt(self.arena, d, 0))
                    }
                    "DXY" => {
                        let d = push_dwrt(self.arena, inner, 0);
                        Ok(push_dwrt(self.arena, d, 1))
                    }
                    "DYY" => {
                        let d = push_dwrt(self.arena, inner, 1);
                        Ok(push_dwrt(self.arena, d, 1))
                    }
                    _ => Err(format!("Unsupported call: {}", func)),
                }
            }

            // Parentheses are transparent - just recurse into the inner expression
            Expr::Paren(inner) => self.lower(inner),

            // Blocks carry the optimizer's CSE: each `let __n = <expr>;` binds a
            // shared subexpression to a single arena node, and the final expression
            // references those bindings by name.
            Expr::Block(block) => {
                for stmt in &block.stmts {
                    match stmt {
                        crate::ast::Stmt::Let(let_stmt) => {
                            let id = self.lower(&let_stmt.init)?;
                            self.locals.insert(let_stmt.name.to_string(), id);
                        }
                        // A non-binding statement has no value to thread; evaluate
                        // it so any nested error surfaces, then discard the id.
                        crate::ast::Stmt::Expr(e) => {
                            let _ = self.lower(e)?;
                        }
                    }
                }
                match &block.expr {
                    Some(final_expr) => self.lower(final_expr),
                    None => Err("Block has no final expression".to_string()),
                }
            }

            // The parser's catch-all: syntax the DSL has no node for, kept
            // whole so the error can name it. A captured local lands here —
            // `let s = 2.0; kernel!(|| X * s)` — and "unsupported expression
            // type" was a poor way to say "`s` is not in scope inside a
            // kernel body".
            Expr::Verbatim(e) => Err(format!(
                "unsupported expression in a kernel body: `{}`",
                quote::quote!(#e)
            )),

            _ => Err("Unsupported expression type".to_string()),
        }
    }
}

/// Push `Dwrt(expr, var)` — the variable index rides as a `Const` operand,
/// matching the encoding the e-graph `ChainRule` and `lower_dwrt` read.
fn push_dwrt(arena: &mut ExprArena, expr: ExprId, var: u8) -> ExprId {
    let v = arena.push_const(var as f32);
    arena.push_binary(OpKind::Dwrt, expr, v)
}

/// Extract f64 from a syn::Lit.
fn extract_f64_from_lit(lit: &Lit) -> Option<f64> {
    match lit {
        Lit::Float(f) => f.base10_parse::<f64>().ok(),
        Lit::Int(i) => i.base10_parse::<i64>().ok().map(|v| v as f64),
        _ => None,
    }
}
