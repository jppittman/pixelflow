//! Macro AST → `ExprArena`.
//!
//! The front end's one lowering step: the surface syntax a user wrote becomes
//! the IR everything downstream speaks. `let` bindings resolve to the
//! [`ExprId`] they name, so the arena is a DAG and a shared subexpression is
//! one node; operators and DSL methods resolve through [`OpKind`], so the op
//! table is not restated here.
//!
//! A helper is inlined at each call — β-reduction. Its arguments are lowered
//! in the caller's scope, once each, and its body is lowered in a scope of
//! its own where its parameters name those nodes: the callee sees its
//! parameters, the block's `const`s and nothing of the caller's, which is
//! what lexical scoping means. A `const` lowers to the value `sema` gave it.
//! An `if` lowers to [`OpKind::If`], the same node `.select` does.
//!
//! Emission — arena to the `TokenStream` that rebuilds it — is [`crate::emit`].

use crate::ast::{BinaryOp, BlockExpr, Expr, FnItem, Role, Stmt, UnaryOp};
use crate::sema::AnalyzedKernel;
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

/// A derivative projection, called as `DX(e)`.
///
/// One definition of the names: `sema` accepts a call by asking here, and
/// lowering builds the `Dwrt` chain by matching on the variant, so the two
/// cannot disagree about which projections exist.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Projection {
    /// `V(e)`: the value itself. Every arena expression is already
    /// value-space, so it is the identity.
    Value,
    Dx,
    Dy,
    Dxx,
    Dxy,
    Dyy,
}

impl Projection {
    /// The projection a name denotes, if any.
    pub(crate) fn from_name(name: &str) -> Option<Self> {
        match name {
            "V" => Some(Self::Value),
            "DX" => Some(Self::Dx),
            "DY" => Some(Self::Dy),
            "DXX" => Some(Self::Dxx),
            "DXY" => Some(Self::Dxy),
            "DYY" => Some(Self::Dyy),
            _ => None,
        }
    }
}

/// The coordinate axes, as `Dwrt` names them.
const AXIS_X: u8 = 0;
const AXIS_Y: u8 = 1;

/// Build a `param_name → index` map over the params of an entry.
///
/// Indices are dense in declaration order: each becomes a `Param(i)` arena
/// node, substituted by `substitute_params` with the host function's
/// arguments in the same order. `Param` holds a `u8`, so more parameters
/// than it counts is a refusal, not a wrap.
fn param_indices(entry: &FnItem) -> Result<HashMap<String, u8>, String> {
    entry
        .params
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let index = u8::try_from(i).map_err(|_| {
                format!(
                    "`{}` declares {} parameters; an entry takes at most {}",
                    entry.name,
                    entry.params.len(),
                    usize::from(u8::MAX) + 1
                )
            })?;
            Ok((p.name.to_string(), index))
        })
        .collect()
}

/// Lower an entry's body into `arena`, inlining the block's helpers and
/// folding its `const`s. Children are recursed first so that parent nodes
/// always reference already-interned [`ExprId`]s.
pub fn lower_entry(
    entry: &FnItem,
    analyzed: &AnalyzedKernel,
    arena: &mut ExprArena,
) -> Result<ExprId, String> {
    let helpers = analyzed
        .def
        .fns
        .iter()
        .filter(|f| f.role() == Role::Helper)
        .map(|f| (f.name.to_string(), f))
        .collect();
    let mut lowering = Lowering {
        program: Program {
            consts: &analyzed.consts,
            helpers,
        },
        frame: Frame {
            role: entry.role(),
            params: param_indices(entry)?,
            locals: Scopes::default(),
        },
        arena,
    };
    lowering.lower(&entry.body)
}

/// The block's items: what every body can name besides its own scope.
struct Program<'a> {
    consts: &'a HashMap<String, f32>,
    helpers: HashMap<String, &'a FnItem>,
}

/// The function being lowered: an entry's parameters by index, and the
/// `let`-bound locals of the blocks being walked (one scope per block, with
/// Rust's lexical scoping). An inlined helper's parameters are locals of its
/// own frame, bound to the argument nodes.
struct Frame {
    role: Role,
    params: HashMap<String, u8>,
    locals: Scopes<ExprId>,
}

/// State threaded through the AST → arena walk.
struct Lowering<'a> {
    program: Program<'a>,
    frame: Frame,
    arena: &'a mut ExprArena,
}

impl Lowering<'_> {
    /// Translate an AST node into the arena, resolving `let`-bound locals via
    /// the frame's scopes. Each binding maps to a single [`ExprId`], so a
    /// local used twice is one node and the arena is a DAG rather than
    /// duplicated subtrees.
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
                    // A `bool` is a canonical mask lane — all-ones or all-zero
                    // — so bitwise AND/OR is logical AND/OR exactly. `sema`
                    // types both operands as `bool`.
                    BinaryOp::BitAnd => OpKind::BitAnd,
                    BinaryOp::BitOr => OpKind::BitOr,
                };

                Ok(self.arena.push_binary(op, lhs, rhs))
            }

            Expr::Unary(unary) => {
                let operand = self.lower(&unary.operand)?;
                let op = match unary.op {
                    UnaryOp::Neg => OpKind::Neg,
                };
                Ok(self.arena.push_unary(op, operand))
            }

            Expr::MethodCall(call) => {
                let method = call.method.to_string();
                let receiver = self.lower(&call.receiver)?;
                let arg_count = call.args.len();

                // Arena expressions are values, so `.clone()` is the identity.
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

            // A helper is inlined; a projection becomes a `Dwrt` chain.
            Expr::Call(call) => {
                let func = call.func.to_string();
                if let Some(helper) = self.program.helpers.get(&func).copied() {
                    return self.inline(helper, &call.args);
                }
                self.lower_projection(&func, &call.args)
            }

            // The choice, the same node `.select` lowers to: `If(m, a, b)`
            // is `if m then a else b`.
            Expr::If(choice) => {
                let cond = self.lower(&choice.cond)?;
                let then = self.lower_block(&choice.then_branch)?;
                let otherwise = self.lower(&choice.else_branch)?;
                Ok(self.arena.push_ternary(OpKind::If, cond, then, otherwise))
            }

            // Parentheses are transparent - just recurse into the inner expression
            Expr::Paren(inner) => self.lower(inner),

            Expr::Block(block) => self.lower_block(block),
        }
    }

    /// Derivative projections (V/DX/DY and the Hessian family) map to
    /// `Dwrt` nodes: the runtime `lower_dwrt` pass (pixelflow-ir) rewrites
    /// them into chain-rule arithmetic before codegen.
    fn lower_projection(&mut self, func: &str, args: &[Expr]) -> Result<ExprId, String> {
        let Some(projection) = Projection::from_name(func) else {
            return Err(format!("Unsupported call: {func}"));
        };
        let [arg] = args else {
            return Err(format!(
                "Unsupported call: {func}/{} (projections take one argument)",
                args.len()
            ));
        };
        let inner = self.lower(arg)?;
        Ok(match projection {
            Projection::Value => inner,
            Projection::Dx => push_dwrt(self.arena, inner, AXIS_X),
            Projection::Dy => push_dwrt(self.arena, inner, AXIS_Y),
            Projection::Dxx => {
                let d = push_dwrt(self.arena, inner, AXIS_X);
                push_dwrt(self.arena, d, AXIS_X)
            }
            Projection::Dxy => {
                let d = push_dwrt(self.arena, inner, AXIS_X);
                push_dwrt(self.arena, d, AXIS_Y)
            }
            Projection::Dyy => {
                let d = push_dwrt(self.arena, inner, AXIS_Y);
                push_dwrt(self.arena, d, AXIS_Y)
            }
        })
    }

    /// β-reduction: `helper(args)` is the helper's body with each parameter
    /// bound to its argument's node.
    ///
    /// The arguments are lowered in the caller's frame, once each, so an
    /// argument used twice in the body is one node. The body is lowered in a
    /// frame of its own: the helper's parameters are its base scope, and
    /// nothing of the caller's — no local, no entry parameter — is visible.
    fn inline(&mut self, helper: &FnItem, args: &[Expr]) -> Result<ExprId, String> {
        if args.len() != helper.params.len() {
            return Err(format!(
                "`{}` takes {} arguments, but {} were supplied",
                helper.name,
                helper.params.len(),
                args.len()
            ));
        }
        let mut locals = Scopes::default();
        for (param, arg) in helper.params.iter().zip(args) {
            let id = self.lower(arg)?;
            locals.bind(param.name.to_string(), id);
        }
        let callee = Frame {
            role: Role::Helper,
            params: HashMap::new(),
            locals,
        };
        let caller = std::mem::replace(&mut self.frame, callee);
        let body = self.lower(&helper.body);
        self.frame = caller;
        body
    }

    /// The node a name refers to: the innermost `let` binding of it in scope,
    /// else a parameter, else a `const`, else a coordinate.
    ///
    /// Bindings come first because that is what lexical scoping means. `sema`
    /// refuses a `let` named X or Y, so today the order only decides a
    /// local against a parameter it shadows; but matching `"X"` before the
    /// locals was how `{ let X = Y; X }` came to read the coordinate, and a
    /// lowering that is right only because an earlier stage refused its
    /// input is one refactor away from that bug again. For the same reason a
    /// coordinate in a helper is refused here too, not only in `sema`.
    fn resolve(&mut self, name: &str) -> Result<ExprId, String> {
        if let Some(&id) = self.frame.locals.lookup(name) {
            return Ok(id);
        }
        // The same order sema documents: a binding, then a parameter, then
        // a const, then the coordinates. Sema refuses a parameter named X or
        // Y, so the two stages agree without one relying on the other's
        // refusal.
        if let Some(&idx) = self.frame.params.get(name) {
            return Ok(self.arena.push_param(idx));
        }
        if let Some(&value) = self.program.consts.get(name) {
            return Ok(self.arena.push_const(value));
        }
        let axis = match name {
            "X" => AXIS_X,
            "Y" => AXIS_Y,
            _ => return Err(format!("Unknown identifier: {name}")),
        };
        match self.frame.role {
            Role::Entry => Ok(self.arena.push_var(axis)),
            Role::Helper => Err(format!(
                "`{name}` in a helper: a helper takes its coordinates as arguments"
            )),
        }
    }

    /// A block's `let`s live in a scope of their own, which ends with it.
    fn lower_block(&mut self, block: &BlockExpr) -> Result<ExprId, String> {
        self.frame.locals.push_scope();
        let value = self.lower_block_contents(block);
        self.frame.locals.pop_scope();
        value
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
                    self.frame.locals.bind(let_stmt.name.to_string(), id);
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

    /// Lower a block straight from the parser, with no `sema` in front: what
    /// lowering itself makes of a name, whatever an earlier stage would
    /// refuse. The `const`s are evaluated, since lowering reads their values.
    /// A refusal comes back as its text: the arena has no `Debug` to unwrap
    /// through.
    fn lower_unanalyzed(input: proc_macro2::TokenStream) -> Result<(ExprArena, ExprId), String> {
        let def = parse(input).expect("the body parses");
        let consts = def
            .consts
            .iter()
            .map(|c| {
                let Expr::Literal(lit) = &c.init else {
                    panic!("this harness holds literal consts only");
                };
                (c.name.to_string(), lit.value)
            })
            .collect();
        let unanalyzed = AnalyzedKernel { def, consts };
        let entry = unanalyzed
            .def
            .fns
            .iter()
            .find(|f| f.role() == Role::Entry)
            .expect("one entry");
        let mut arena = ExprArena::new();
        let root = lower_entry(entry, &unanalyzed, &mut arena)?;
        Ok((arena, root))
    }

    fn lowered(input: proc_macro2::TokenStream) -> (ExprArena, ExprId) {
        lower_unanalyzed(input).expect("the body lowers")
    }

    /// Probe p5, at the stage that got it wrong. `sema` now refuses
    /// `let X`, but lowering resolves a name through its scopes before the
    /// coordinates regardless: `{ let X = Y; X }` is the local, which is `Y`.
    /// It was `Var(0)`.
    #[test]
    fn a_binding_is_resolved_before_a_coordinate_of_the_same_name() {
        let (arena, root) = lowered(quote! { || { let X = Y; X } });
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
        let (arena, root) = lowered(quote! { |r: f32| ({ let r = X; r }) + r });
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
        let (arena, root) = lowered(quote! { || 1.00000005960464477539062500001 });
        let ExprNode::Const(value) = arena.node(root) else {
            panic!("expected a constant, got {:?}", arena.node(root));
        };
        assert_eq!(value.to_bits(), 0x3f80_0001);
    }

    /// `if` and `.select` are the same node.
    #[test]
    fn an_if_lowers_to_the_if_node() {
        let (arena, root) = lowered(quote! { || if X < Y { X } else { Y } });
        let ExprNode::Ternary(OpKind::If, cond, a, b) = arena.node(root) else {
            panic!("expected If, got {:?}", arena.node(root));
        };
        assert!(matches!(
            arena.node(cond),
            ExprNode::Binary(OpKind::Lt, _, _)
        ));
        assert!(matches!(arena.node(a), ExprNode::Var(0)));
        assert!(matches!(arena.node(b), ExprNode::Var(1)));
    }

    /// A `const` lowers to its value.
    #[test]
    fn a_const_lowers_to_its_value() {
        let (arena, root) = lowered(quote! {
            const R: f32 = 2.5;
            pub fn f() -> f32 { X * R }
        });
        let ExprNode::Binary(OpKind::Mul, _, r) = arena.node(root) else {
            panic!("expected the product, got {:?}", arena.node(root));
        };
        assert!(matches!(arena.node(r), ExprNode::Const(v) if v == 2.5));
    }

    /// A helper's body is lowered in a frame of its own: its parameter
    /// names the argument's node, and a caller's local of the same name as
    /// something the helper reads is not visible to it. Here `sq`'s `x` is
    /// the argument `Y`, not the caller's `let x = X`, and the argument is
    /// one node used twice.
    #[test]
    fn a_helper_is_inlined_in_its_own_scope() {
        let (arena, root) = lowered(quote! {
            fn sq(x: f32) -> f32 { x * x }
            pub fn f() -> f32 { let x = X; sq(Y) + x }
        });
        let ExprNode::Binary(OpKind::Add, call, local) = arena.node(root) else {
            panic!("expected the sum, got {:?}", arena.node(root));
        };
        let ExprNode::Binary(OpKind::Mul, a, b) = arena.node(call) else {
            panic!("expected the square, got {:?}", arena.node(call));
        };
        assert_eq!(a, b, "the argument is one node, used twice");
        assert!(
            matches!(arena.node(a), ExprNode::Var(1)),
            "`x` in `sq` is Y"
        );
        assert!(
            matches!(arena.node(local), ExprNode::Var(0)),
            "`x` in `f` is X"
        );
    }

    /// The caller's locals are not in scope in the helper, at this stage
    /// too: a helper naming one the caller happens to bind is an error, not
    /// dynamic scoping.
    #[test]
    fn a_helper_does_not_see_the_callers_locals() {
        let Err(err) = lower_unanalyzed(quote! {
            fn leak() -> f32 { a }
            pub fn f() -> f32 { let a = X; leak() }
        }) else {
            panic!("`a` is not in the helper's scope");
        };
        assert!(err.contains("Unknown identifier: a"), "got: {err}");
    }

    /// A coordinate in a helper is refused by lowering itself, not only by
    /// `sema`.
    #[test]
    fn a_coordinate_in_a_helper_is_refused_here_too() {
        let Err(err) = lower_unanalyzed(quote! {
            fn shifted() -> f32 { X }
            pub fn f() -> f32 { shifted() }
        }) else {
            panic!("a helper reads no coordinate");
        };
        assert!(err.contains("`X` in a helper"), "got: {err}");
    }

    /// Helpers calling helpers: each call is its own inlining, over its own
    /// arguments.
    #[test]
    fn helpers_call_helpers() {
        let (arena, root) = lowered(quote! {
            fn twice(x: f32) -> f32 { x + x }
            fn quad(x: f32) -> f32 { twice(twice(x)) }
            pub fn f() -> f32 { quad(X) }
        });
        let ExprNode::Binary(OpKind::Add, a, b) = arena.node(root) else {
            panic!("expected the outer sum, got {:?}", arena.node(root));
        };
        assert_eq!(a, b);
        let ExprNode::Binary(OpKind::Add, c, d) = arena.node(a) else {
            panic!("expected the inner sum, got {:?}", arena.node(a));
        };
        assert_eq!(c, d);
        assert!(matches!(arena.node(c), ExprNode::Var(0)));
    }
}
