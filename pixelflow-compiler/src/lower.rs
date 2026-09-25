//! Macro AST → `ExprArena`.
//!
//! The front end's one lowering step: the surface syntax a user wrote becomes
//! the IR everything downstream speaks. `let` bindings resolve to the
//! [`ExprId`] they name, so the arena is a DAG and a shared subexpression is
//! one node; operators and DSL methods resolve through [`OpKind`], so the op
//! table is not restated here.
//!
//! Emission — arena to the `TokenStream` that rebuilds it — is [`crate::emit`].

use crate::ast::{BinaryOp, BlockExpr, Expr, Stmt, UnaryOp};
use crate::symbol::Scopes;
use pixelflow_ir::OpKind;
use pixelflow_ir::arena::{ExprArena, ExprId};
use std::collections::HashMap;

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
        locals: Scopes::default(),
        arena,
    };
    lowering.lower(expr)
}

/// State threaded through the AST → arena walk: parameter names (fixed for
/// the whole kernel), `let`-bound locals (one scope per block being walked,
/// with Rust's lexical scoping), and the arena nodes are pushed into.
struct Lowering<'a> {
    param_indices: &'a HashMap<String, u8>,
    locals: Scopes<ExprId>,
    arena: &'a mut ExprArena,
}

impl Lowering<'_> {
    /// Translate an AST node into the arena, resolving `let`-bound locals via
    /// `self.locals`. Each binding maps to a single [`ExprId`], so a local
    /// used twice is one node and the arena is a DAG rather than duplicated
    /// subtrees.
    fn lower(&mut self, expr: &Expr) -> Result<ExprId, String> {
        match expr {
            Expr::Ident(ident) => self.resolve(&ident.name.to_string()),

            Expr::Literal(lit) => Ok(self.arena.push_const(lit.value)),

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

            // A block's `let`s live in a scope of their own, which ends with it.
            Expr::Block(block) => {
                self.locals.push_scope();
                let value = self.lower_block_contents(block);
                self.locals.pop_scope();
                value
            }

            // The parser's catch-all: syntax the DSL has no node for, kept
            // whole so the error can name it. An unbound name never lands
            // here: sema refuses it, with a span, before lowering runs.
            Expr::Verbatim(e) => Err(format!(
                "unsupported expression in a kernel body: `{}`",
                quote::quote!(#e)
            )),

            _ => Err("Unsupported expression type".to_string()),
        }
    }

    /// The node a name refers to: the innermost `let` binding of it in scope,
    /// else a coordinate, else a parameter.
    ///
    /// Bindings come first because that is what lexical scoping means. `sema`
    /// refuses a `let` named X or Y, so today the order only decides a
    /// local against a parameter it shadows; but matching `"X"` before the
    /// locals was how `{ let X = Y; X }` came to read the coordinate, and a
    /// lowering that is right only because an earlier stage refused its
    /// input is one refactor away from that bug again.
    fn resolve(&mut self, name: &str) -> Result<ExprId, String> {
        if let Some(&id) = self.locals.lookup(name) {
            return Ok(id);
        }
        // The same order sema documents: a binding, then a parameter, then
        // the coordinates. Sema refuses a parameter named X or Y, so the two
        // stages agree without one relying on the other's refusal.
        if let Some(&idx) = self.param_indices.get(name) {
            return Ok(self.arena.push_param(idx));
        }
        match name {
            "X" => Ok(self.arena.push_var(0)),
            "Y" => Ok(self.arena.push_var(1)),
            _ => Err(format!("Unknown identifier: {name}")),
        }
    }

    /// A block's statements in order, then its value, in the scope the
    /// caller opened for it.
    fn lower_block_contents(&mut self, block: &BlockExpr) -> Result<ExprId, String> {
        for stmt in &block.stmts {
            match stmt {
                // The initializer is lowered before the binding exists, so it
                // sees whatever the name meant before: `let a = a + 1.0;`.
                Stmt::Let(let_stmt) => {
                    let id = self.lower(&let_stmt.init)?;
                    self.locals.bind(let_stmt.name.to_string(), id);
                }
                // A non-binding statement has no value to thread; lower it so
                // any nested error surfaces, then discard the id.
                Stmt::Expr(e) => {
                    self.lower(e)?;
                }
            }
        }
        match &block.expr {
            Some(final_expr) => self.lower(final_expr),
            None => Err("Block has no final expression".to_string()),
        }
    }
}

/// Push `Dwrt(expr, var)` — the variable index rides as a `Const` operand,
/// matching the encoding the e-graph `ChainRule` and `lower_dwrt` read.
fn push_dwrt(arena: &mut ExprArena, expr: ExprId, var: u8) -> ExprId {
    let v = arena.push_const(var as f32);
    arena.push_binary(OpKind::Dwrt, expr, v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;
    use pixelflow_ir::arena::ExprNode;
    use quote::quote;

    /// Lower a body straight from the parser, with no `sema` in front: what
    /// lowering itself makes of a name, whatever an earlier stage would
    /// refuse.
    fn lower_unanalyzed(input: proc_macro2::TokenStream) -> (ExprArena, ExprId) {
        let unanalyzed = crate::sema::AnalyzedKernel {
            def: parse(input).expect("the body parses"),
        };
        let params = param_indices(&unanalyzed);
        let mut arena = ExprArena::new();
        let root =
            ast_to_arena(&unanalyzed.def.body, &params, &mut arena).expect("the body lowers");
        (arena, root)
    }

    /// Probe p5, at the stage that got it wrong. `sema` now refuses
    /// `let X`, but lowering resolves a name through its scopes before the
    /// coordinates regardless: `{ let X = Y; X }` is the local, which is `Y`.
    /// It was `Var(0)`.
    #[test]
    fn a_binding_is_resolved_before_a_coordinate_of_the_same_name() {
        let (arena, root) = lower_unanalyzed(quote! { || { let X = Y; X } });
        assert!(
            matches!(arena.node(root), ExprNode::Var(1)),
            "`X` is the local bound to Y, got {:?}",
            arena.node(root)
        );
    }

    /// A local shadows the parameter it is named after, and only inside its
    /// block.
    #[test]
    fn a_binding_shadows_a_parameter_only_inside_its_block() {
        let (arena, root) = lower_unanalyzed(quote! { |r: f32| ({ let r = X; r }) + r });
        let ExprNode::Binary(OpKind::Add, inner, outer) = arena.node(root) else {
            panic!("expected the sum, got {:?}", arena.node(root));
        };
        assert!(
            matches!(arena.node(inner), ExprNode::Var(0)),
            "inner `r` is X"
        );
        assert!(
            matches!(arena.node(outer), ExprNode::Param(0)),
            "outer `r` is the parameter"
        );
    }

    /// A literal lowers to the value the parser gave it, bit for bit.
    #[test]
    fn a_literal_lowers_to_the_value_the_parser_rounded_once() {
        let (arena, root) = lower_unanalyzed(quote! { || 1.00000005960464477539062500001 });
        let ExprNode::Const(value) = arena.node(root) else {
            panic!("expected a constant, got {:?}", arena.node(root));
        };
        assert_eq!(value.to_bits(), 0x3f80_0001);
    }
}
