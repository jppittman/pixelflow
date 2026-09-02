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

/// Base `Var` index for manifold-param slots in an expansion arena: a *bare*
/// reference to manifold param `k` becomes `Var(128 + k)`, and the builder
/// substitutes the slot with the argument kernel's spliced fragment at
/// construction time (all bare references share one fragment, preserving the
/// DAG). Var 0..4 are coordinates, 4..8 reduction indices, and the e-graph's
/// scalar-param encoding sits at 16+; slots are gone (substituted) before any
/// arena reaches the optimizer or a backend.
pub const MANIFOLD_SLOT_BASE: u8 = 128;

/// Base `Var` index for `.at()` call sites: each `m.at(x, y, z, w)` gets its
/// own slot `Var(192 + s)`, substituted with a *fresh* splice of `m`'s
/// fragment warped by the site's coordinate expressions
/// (`substitute_vars_with` on Var 0..4) — per-site warps cannot share a
/// fragment the way bare references do.
pub const AT_SITE_BASE: u8 = 192;

/// Maximum manifold params per kernel (slots 128..192).
pub const MAX_MANIFOLD_PARAMS: usize = 64;

/// Maximum `.at()` sites per kernel body (slots 192..=255).
pub const MAX_AT_SITES: usize = 64;

/// One `.at()` call site recorded during AST → arena conversion: which
/// manifold param it samples and the arena ids of its four coordinate
/// expressions (template-relative, so they are literals in emitted code).
pub struct AtSite {
    pub param: u8,
    pub coords: [ExprId; 4],
}

/// What the builder must compose at construction time, alongside the arena
/// template: which bare param slots are used, and every `.at()` site.
pub struct CompositionPlan {
    pub bare_params: Vec<u8>,
    pub at_sites: Vec<AtSite>,
}

impl CompositionPlan {
    pub fn is_empty(&self) -> bool {
        self.bare_params.is_empty() && self.at_sites.is_empty()
    }
}

/// Build a `param_name → index` map over the *scalar* params of a kernel.
///
/// Indices are dense over scalars in declaration order (manifold params do
/// not consume an index): they become `Param(i)` arena nodes, substituted by
/// `substitute_params` with the builder closure's scalar arguments in the
/// same dense order.
pub fn scalar_param_indices(analyzed: &crate::sema::AnalyzedKernel) -> HashMap<String, u8> {
    analyzed
        .def
        .params
        .iter()
        .filter(|p| matches!(p.kind, crate::ast::ParamKind::Scalar(_)))
        .enumerate()
        .map(|(i, p)| (p.name.to_string(), i as u8))
        .collect()
}

/// Build a `param_name → slot` map over the *manifold* params of a kernel,
/// dense in declaration order. Slot `k` appears in the arena as
/// `Var(MANIFOLD_SLOT_BASE + k)`.
pub fn manifold_param_indices(analyzed: &crate::sema::AnalyzedKernel) -> HashMap<String, u8> {
    analyzed
        .def
        .params
        .iter()
        .filter(|p| matches!(p.kind, crate::ast::ParamKind::Manifold))
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
    manifold_indices: &HashMap<String, u8>,
    arena: &mut ExprArena,
) -> Result<(ExprId, CompositionPlan), String> {
    let mut locals: HashMap<String, ExprId> = HashMap::new();
    let ctx = Ctx {
        param_indices,
        manifold_indices,
        at_sites: std::cell::RefCell::new(Vec::new()),
    };
    let root = ast_to_arena_inner(expr, &ctx, &mut locals, arena)?;

    // Bare-slot usage is read off the built arena rather than tracked during
    // the walk: any reachable-or-not Var in the bare range means the builder
    // must splice that param's fragment once. (`.at()` receivers never push
    // their bare Var — see the MethodCall arm — so this set is exact.)
    let mut bare_params: Vec<u8> = arena
        .nodes_raw()
        .iter()
        .filter_map(|n| match n {
            pixelflow_ir::arena::ExprNode::Var(i)
                if (MANIFOLD_SLOT_BASE..AT_SITE_BASE).contains(i) =>
            {
                Some(i - MANIFOLD_SLOT_BASE)
            }
            _ => None,
        })
        .collect();
    bare_params.sort_unstable();
    bare_params.dedup();

    Ok((
        root,
        CompositionPlan {
            bare_params,
            at_sites: ctx.at_sites.into_inner(),
        },
    ))
}

/// Name-resolution context for the AST → arena walk.
struct Ctx<'a> {
    param_indices: &'a HashMap<String, u8>,
    manifold_indices: &'a HashMap<String, u8>,
    /// `.at()` sites recorded as the walk encounters them; site `s` appears
    /// in the arena as `Var(AT_SITE_BASE + s)`.
    at_sites: std::cell::RefCell<Vec<AtSite>>,
}

/// Resolve `expr` as a reference to a manifold param without pushing arena
/// nodes: a direct param name, or a local bound to one (its recorded id is
/// the param's bare slot `Var`).
fn manifold_slot_of(
    expr: &Expr,
    ctx: &Ctx<'_>,
    locals: &HashMap<String, ExprId>,
    arena: &ExprArena,
) -> Option<u8> {
    let Expr::Ident(ident) = expr else {
        return None;
    };
    let name = ident.name.to_string();
    if let Some(&slot) = ctx.manifold_indices.get(&name) {
        return Some(slot);
    }
    if let Some(&id) = locals.get(&name)
        && let pixelflow_ir::arena::ExprNode::Var(i) = arena.node(id)
        && (MANIFOLD_SLOT_BASE..AT_SITE_BASE).contains(i)
    {
        return Some(i - MANIFOLD_SLOT_BASE);
    }
    None
}

/// Translate an AST node into the arena, resolving `let`-bound locals via
/// `locals`. The optimizer emits `let`-bindings (a [`Expr::Block`]) for shared
/// subexpressions; each binding maps to a single [`ExprId`], so the arena
/// faithfully preserves the discovered CSE as a DAG rather than duplicating
/// subtrees.
fn ast_to_arena_inner(
    expr: &Expr,
    ctx: &Ctx<'_>,
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
                    } else if let Some(&idx) = ctx.param_indices.get(&name) {
                        Ok(arena.push_param(idx))
                    } else if let Some(&slot) = ctx.manifold_indices.get(&name) {
                        // Manifold param: a reserved slot variable, replaced
                        // by the argument kernel's spliced fragment when the
                        // builder closure runs.
                        Ok(arena.push_var(MANIFOLD_SLOT_BASE + slot))
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
                Err("Non-numeric literal".to_string())
            }
        }

        Expr::Binary(binary) => {
            let lhs = ast_to_arena_inner(&binary.lhs, ctx, locals, arena)?;
            let rhs = ast_to_arena_inner(&binary.rhs, ctx, locals, arena)?;

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

            Ok(arena.push_binary(op, lhs, rhs))
        }

        Expr::Unary(unary) => {
            let operand = ast_to_arena_inner(&unary.operand, ctx, locals, arena)?;

            let op = match unary.op {
                UnaryOp::Neg => OpKind::Neg,
                UnaryOp::Not => return Err("Unsupported unary op: Not".to_string()),
            };

            Ok(arena.push_unary(op, operand))
        }

        Expr::MethodCall(call) => {
            let method = call.method.to_string();

            // `.at(x, y, z, w)`: sample a manifold param at warped
            // coordinates. Intercepted before receiver evaluation so the
            // receiver's bare slot Var is never pushed — the site gets its
            // own slot, substituted with a per-site warped splice.
            if method == "at" {
                let Some(param) = manifold_slot_of(&call.receiver, ctx, locals, arena) else {
                    // NOTE: a local bound to a manifold param (`let t = tex;
                    // t.at(..)`) is not resolvable here — the AST optimizer
                    // eliminates the manifold-binding let while restoring the
                    // opaque `.at()` call verbatim, so the local is dangling
                    // by the time this bridge runs. Direct receivers cover
                    // the real usage (bilinear, scene3d).
                    return Err(".at() receiver must be a manifold param".to_string());
                };
                if call.args.len() != 4 {
                    return Err(format!(
                        ".at() takes 4 coordinate arguments, got {}",
                        call.args.len()
                    ));
                }
                let mut coords = [ExprId(0); 4];
                for (i, arg) in call.args.iter().enumerate() {
                    coords[i] = ast_to_arena_inner(arg, ctx, locals, arena)?;
                }
                let mut sites = ctx.at_sites.borrow_mut();
                if sites.len() >= MAX_AT_SITES {
                    return Err(format!(
                        "kernel body has more than {} .at() sites",
                        MAX_AT_SITES
                    ));
                }
                let s = sites.len() as u8;
                sites.push(AtSite { param, coords });
                return Ok(arena.push_var(AT_SITE_BASE + s));
            }

            let receiver = ast_to_arena_inner(&call.receiver, ctx, locals, arena)?;

            match (method.as_str(), call.args.len()) {
                // Arena expressions are values; `.clone()` (needed by the
                // combinator backend for non-Copy trees) is the identity here,
                // so one kernel body compiles under both backends.
                ("clone", 0) => Ok(receiver),

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
                    let arg = ast_to_arena_inner(&call.args[0], ctx, locals, arena)?;
                    Ok(arena.push_binary(OpKind::Min, receiver, arg))
                }
                ("max", 1) => {
                    let arg = ast_to_arena_inner(&call.args[0], ctx, locals, arena)?;
                    Ok(arena.push_binary(OpKind::Max, receiver, arg))
                }
                ("atan2", 1) => {
                    let arg = ast_to_arena_inner(&call.args[0], ctx, locals, arena)?;
                    Ok(arena.push_binary(OpKind::Atan2, receiver, arg))
                }

                // Ternary methods
                ("mul_add", 2) => {
                    let b = ast_to_arena_inner(&call.args[0], ctx, locals, arena)?;
                    let c = ast_to_arena_inner(&call.args[1], ctx, locals, arena)?;
                    Ok(arena.push_ternary(OpKind::MulAdd, receiver, b, c))
                }
                ("select", 2) => {
                    let if_true = ast_to_arena_inner(&call.args[0], ctx, locals, arena)?;
                    let if_false = ast_to_arena_inner(&call.args[1], ctx, locals, arena)?;
                    Ok(arena.push_ternary(OpKind::Select, receiver, if_true, if_false))
                }
                // `clamp` is library, not a primitive: it denotes
                // `min(max(x, lo), hi)` and is built as that composition.
                ("clamp", 2) => {
                    let lo = ast_to_arena_inner(&call.args[0], ctx, locals, arena)?;
                    let hi = ast_to_arena_inner(&call.args[1], ctx, locals, arena)?;
                    let floored = arena.push_binary(OpKind::Max, receiver, lo);
                    Ok(arena.push_binary(OpKind::Min, floored, hi))
                }

                // Comparison methods (emitted by e-graph extraction)
                ("lt", 1) => {
                    let a = ast_to_arena_inner(&call.args[0], ctx, locals, arena)?;
                    Ok(arena.push_binary(OpKind::Lt, receiver, a))
                }
                ("le", 1) => {
                    let a = ast_to_arena_inner(&call.args[0], ctx, locals, arena)?;
                    Ok(arena.push_binary(OpKind::Le, receiver, a))
                }
                ("gt", 1) => {
                    let a = ast_to_arena_inner(&call.args[0], ctx, locals, arena)?;
                    Ok(arena.push_binary(OpKind::Gt, receiver, a))
                }
                ("ge", 1) => {
                    let a = ast_to_arena_inner(&call.args[0], ctx, locals, arena)?;
                    Ok(arena.push_binary(OpKind::Ge, receiver, a))
                }
                ("eq", 1) => {
                    let a = ast_to_arena_inner(&call.args[0], ctx, locals, arena)?;
                    Ok(arena.push_binary(OpKind::Eq, receiver, a))
                }
                ("ne", 1) => {
                    let a = ast_to_arena_inner(&call.args[0], ctx, locals, arena)?;
                    Ok(arena.push_binary(OpKind::Ne, receiver, a))
                }

                _ => Err(format!("Unsupported method: {}", method)),
            }
        }

        // Derivative projections (V/DX/DY/DZ and the Hessian family) map to
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
            let inner = ast_to_arena_inner(&call.args[0], ctx, locals, arena)?;
            match func.as_str() {
                "V" => Ok(inner),
                "DX" => Ok(push_dwrt(arena, inner, 0)),
                "DY" => Ok(push_dwrt(arena, inner, 1)),
                "DZ" => Ok(push_dwrt(arena, inner, 2)),
                "DXX" => {
                    let d = push_dwrt(arena, inner, 0);
                    Ok(push_dwrt(arena, d, 0))
                }
                "DXY" => {
                    let d = push_dwrt(arena, inner, 0);
                    Ok(push_dwrt(arena, d, 1))
                }
                "DYY" => {
                    let d = push_dwrt(arena, inner, 1);
                    Ok(push_dwrt(arena, d, 1))
                }
                _ => Err(format!("Unsupported call: {}", func)),
            }
        }

        // Parentheses are transparent - just recurse into the inner expression
        Expr::Paren(inner) => ast_to_arena_inner(inner, ctx, locals, arena),

        // Blocks carry the optimizer's CSE: each `let __n = <expr>;` binds a
        // shared subexpression to a single arena node, and the final expression
        // references those bindings by name.
        Expr::Block(block) => {
            for stmt in &block.stmts {
                match stmt {
                    crate::ast::Stmt::Let(let_stmt) => {
                        let id = ast_to_arena_inner(&let_stmt.init, ctx, locals, arena)?;
                        locals.insert(let_stmt.name.to_string(), id);
                    }
                    // A non-binding statement has no value to thread; evaluate
                    // it so any nested error surfaces, then discard the id.
                    crate::ast::Stmt::Expr(e) => {
                        let _ = ast_to_arena_inner(e, ctx, locals, arena)?;
                    }
                }
            }
            match &block.expr {
                Some(final_expr) => ast_to_arena_inner(final_expr, ctx, locals, arena),
                None => Err("Block has no final expression".to_string()),
            }
        }

        _ => Err("Unsupported expression type".to_string()),
    }
}

/// Generate runtime arena-construction code from macro AST.
///
/// Derivatives are eliminated here, at macro-expansion time, when possible:
/// see [`differentiate_in_optimizer`]. Kernels whose `Dwrt` nodes survive (an
/// op the e-graph cannot differentiate, or a saturation budget miss) emit the
/// `Dwrt`-carrying arena unchanged — the runtime `lower_dwrt` pass in
/// pixelflow-ir is the fallback tier and errors loudly only on genuinely
/// non-differentiable ops.
pub fn ast_to_runtime_arena(
    expr: &Expr,
    param_indices: &HashMap<String, u8>,
    manifold_indices: &HashMap<String, u8>,
) -> Result<(TokenStream, CompositionPlan), String> {
    let mut arena = ExprArena::new();
    let (mut root, plan) = ast_to_arena(expr, param_indices, manifold_indices, &mut arena)?;
    // A composing kernel skips expansion-time optimization entirely: slots
    // stand for whole expressions (so the calculus must wait for splicing),
    // and extraction would rebuild the arena, invalidating the plan's
    // template-relative `.at()` coordinate ids.
    if plan.is_empty()
        && let Some((optimized, optimized_root)) = differentiate_in_optimizer(&arena, root)
    {
        arena = optimized;
        root = optimized_root;
    }
    let nodes = arena.nodes_raw();
    let nary_children = arena.nary_children_raw();

    let node_tokens: Vec<TokenStream> = nodes
        .iter()
        .map(|node| match node {
            pixelflow_ir::arena::ExprNode::Var(i) => {
                quote! { ::pixelflow_core::__macro::ir::arena::ExprNode::Var(#i) }
            }
            // By bit pattern, not as a decimal literal: `quote`'s `f32`
            // impl goes through `Literal::f32_suffixed`, which asserts
            // `is_finite()` — and non-finite constants are ordinary here. A
            // true comparison mask is all-ones (`OpKind::mask`), which is
            // `BitAnd`'s monoid identity and therefore `all_over`'s seed, and
            // the folder now produces those. Bits also roundtrip exactly, with
            // no decimal-formatting question to get wrong.
            pixelflow_ir::arena::ExprNode::Const(v) => {
                let bits = v.to_bits();
                quote! { ::pixelflow_core::__macro::ir::arena::ExprNode::Const(f32::from_bits(#bits)) }
            }
            pixelflow_ir::arena::ExprNode::Param(i) => {
                quote! { ::pixelflow_core::__macro::ir::arena::ExprNode::Param(#i) }
            }
            // The `kernel!` macro has no buffer surface yet, so this is
            // unreachable in practice; fail loud rather than emit a node that
            // references a buffer table `from_raw` does not reconstruct.
            pixelflow_ir::arena::ExprNode::Buffer(b) => {
                panic!(
                    "kernel! produced ExprNode::Buffer({}) — lattice parameters are not wired \
                     into the compiler yet (KERNELS_AND_LATTICES.md M4)",
                    b.0
                )
            }
            pixelflow_ir::arena::ExprNode::Unary(op, child) => {
                let op_code = opkind_to_tokens(*op);
                let child = child.0;
                quote! { ::pixelflow_core::__macro::ir::arena::ExprNode::Unary(#op_code, ::pixelflow_core::__macro::ir::arena::ExprId(#child)) }
            }
            pixelflow_ir::arena::ExprNode::Binary(op, a, b) => {
                let op_code = opkind_to_tokens(*op);
                let a = a.0;
                let b = b.0;
                quote! { ::pixelflow_core::__macro::ir::arena::ExprNode::Binary(#op_code, ::pixelflow_core::__macro::ir::arena::ExprId(#a), ::pixelflow_core::__macro::ir::arena::ExprId(#b)) }
            }
            pixelflow_ir::arena::ExprNode::Ternary(op, a, b, c) => {
                let op_code = opkind_to_tokens(*op);
                let a = a.0;
                let b = b.0;
                let c = c.0;
                quote! { ::pixelflow_core::__macro::ir::arena::ExprNode::Ternary(#op_code, ::pixelflow_core::__macro::ir::arena::ExprId(#a), ::pixelflow_core::__macro::ir::arena::ExprId(#b), ::pixelflow_core::__macro::ir::arena::ExprId(#c)) }
            }
            pixelflow_ir::arena::ExprNode::Nary(op, start, len) => {
                let op_code = opkind_to_tokens(*op);
                quote! { ::pixelflow_core::__macro::ir::arena::ExprNode::Nary(#op_code, #start, #len) }
            }
        })
        .collect();

    let child_tokens: Vec<TokenStream> = nary_children
        .iter()
        .map(|id| {
            let id = id.0;
            quote! { ::pixelflow_core::__macro::ir::arena::ExprId(#id) }
        })
        .collect();

    let root = root.0;
    let tokens = quote! {{
        let __nodes = vec![#(#node_tokens),*];
        let __nary_children = vec![#(#child_tokens),*];
        let __arena = ::pixelflow_core::__macro::ir::arena::ExprArena::from_raw(__nodes, __nary_children);
        (__arena, ::pixelflow_core::__macro::ir::arena::ExprId(#root))
    }};
    Ok((tokens, plan))
}

/// Emit the construction-time composition statements for `plan`: splice each
/// bare param's fragment once, splice-and-warp a fresh fragment per `.at()`
/// site, then substitute every slot in one pass. Expects `__arena: ExprArena`
/// and `__root: ExprId` (mut) in scope; `accessors[k]` is the expression for
/// manifold param `k` (a builder-closure argument or a struct field), which
/// must implement `Lower`.
pub fn composition_stmts(plan: &CompositionPlan, accessors: &[TokenStream]) -> TokenStream {
    if plan.is_empty() {
        return quote! {};
    }

    let bare: Vec<TokenStream> = plan
        .bare_params
        .iter()
        .map(|k| {
            let acc = &accessors[*k as usize];
            let slot = MANIFOLD_SLOT_BASE + k;
            quote! {
                {
                    let __frag = ::pixelflow_core::Lower::lower(
                        &#acc,
                        &mut __arena,
                        &mut ::pixelflow_core::LowerEnv::default(),
                    )
                    .expect("kernel argument does not lower to IR");
                    __subs.push((#slot, __frag));
                }
            }
        })
        .collect();

    let sites: Vec<TokenStream> = plan
        .at_sites
        .iter()
        .enumerate()
        .map(|(s, site)| {
            let acc = &accessors[site.param as usize];
            let slot = AT_SITE_BASE + s as u8;
            let [cx, cy, cz, cw] = site.coords.map(|c| c.0);
            quote! {
                {
                    let __frag = ::pixelflow_core::Lower::lower(
                        &#acc,
                        &mut __arena,
                        &mut ::pixelflow_core::LowerEnv::default(),
                    )
                    .expect("kernel argument does not lower to IR");
                    let __warped = __arena.substitute_vars_with(
                        __frag,
                        &[
                            (0u8, ::pixelflow_core::__macro::ir::arena::ExprId(#cx)),
                            (1u8, ::pixelflow_core::__macro::ir::arena::ExprId(#cy)),
                            (2u8, ::pixelflow_core::__macro::ir::arena::ExprId(#cz)),
                            (3u8, ::pixelflow_core::__macro::ir::arena::ExprId(#cw)),
                        ],
                    );
                    __subs.push((#slot, __warped));
                }
            }
        })
        .collect();

    quote! {
        let mut __subs: ::std::vec::Vec<(u8, ::pixelflow_core::__macro::ir::arena::ExprId)> =
            ::std::vec::Vec::new();
        #( #bare )*
        #( #sites )*
        __root = __arena.substitute_vars_with(__root, &__subs);
    }
}

/// Push `Dwrt(expr, var)` — the variable index rides as a `Const` operand,
/// matching the encoding the e-graph `ChainRule` and `lower_dwrt` read.
fn push_dwrt(arena: &mut ExprArena, expr: ExprId, var: u8) -> ExprId {
    let v = arena.push_const(var as f32);
    arena.push_binary(OpKind::Dwrt, expr, v)
}

// ============================================================================
// Expansion-time differentiation (calculus in the optimizer)
// ============================================================================

/// Base `Var` index used to carry `Param(i)` leaves through the e-graph, which
/// has no Param representation: indices 0..4 are coordinates and 4..8 are
/// reduction indices, so params ride at 16+ and are mapped back after
/// extraction. To every rewrite rule they are opaque leaves — exactly the
/// semantics of an unbound scalar — and the chain rule gives them derivative
/// zero like any non-differentiation variable.
const PARAM_VAR_BASE: u8 = 16;

/// Run the AOT-tier e-graph (full rule set: derivative + algebra + fusion)
/// over a `Dwrt`-carrying expansion arena, so derivatives are expanded *and
/// simplified* at macro-expansion time and the runtime never sees the
/// calculus.
///
/// Returns `None` when there is nothing to do (no `Dwrt`) or when the result
/// still contains a `Dwrt` (unsupported op / budget miss) — the caller then
/// emits the original arena and the runtime `lower_dwrt` pass takes over.
/// A budget miss is legitimate behavior, not a failure: the output's only
/// contract is that a `Some` is `Dwrt`-free and mathematically equivalent.
fn differentiate_in_optimizer(arena: &ExprArena, root: ExprId) -> Option<(ExprArena, ExprId)> {
    use pixelflow_ir::arena::ExprNode;
    use pixelflow_search::egraph::{CostModel, EGraph, extract};

    if !contains_dwrt(arena) {
        return None;
    }

    // Manifold-param and `.at()`-site slots (`Var(128+)`) stand for whole
    // kernel expressions spliced in at construction time — differentiating
    // one as if it were an independent variable (derivative 0) would be
    // wrong. The calculus for composed kernels resolves after splicing, in
    // the runtime `lower_dwrt` tier.
    if arena
        .nodes_raw()
        .iter()
        .any(|n| matches!(n, ExprNode::Var(i) if *i >= MANIFOLD_SLOT_BASE))
    {
        return None;
    }

    // Every op must be representable in the e-graph (`add_arena` panics on
    // ops with no rewrite-rule `Op`, e.g. the bit-manip primitives). The
    // kernel surface cannot produce those today, but fall back rather than
    // panic if that ever changes.
    let representable = arena.nodes_raw().iter().all(|n| match n {
        ExprNode::Unary(op, _)
        | ExprNode::Binary(op, _, _)
        | ExprNode::Ternary(op, _, _, _)
        | ExprNode::Nary(op, _, _) => pixelflow_search::egraph::ops::op_from_kind(*op).is_some(),
        ExprNode::Buffer(_) => false,
        ExprNode::Var(_) | ExprNode::Const(_) | ExprNode::Param(_) => true,
    });
    if !representable {
        return None;
    }

    // Param(i) -> Var(16+i) for the round-trip.
    let encoded_nodes: Vec<ExprNode> = arena
        .nodes_raw()
        .iter()
        .map(|n| match n {
            ExprNode::Param(i) => {
                assert!(
                    PARAM_VAR_BASE + i < MANIFOLD_SLOT_BASE,
                    "kernel has too many scalar params to encode for the e-graph"
                );
                ExprNode::Var(PARAM_VAR_BASE + i)
            }
            other => other.clone(),
        })
        .collect();
    let encoded = ExprArena::from_raw(encoded_nodes, arena.nary_children_raw().to_vec());

    // One saturation, full rule set: differentiation and algebra rewrite
    // TOGETHER — the point of the e-graph is that there is no pass ordering,
    // so the optimizer is free to simplify the differentiand before, during,
    // or after the chain rule fires (see the `derivative` module doc in
    // pixelflow-search). Standard optimizer budget.
    let mut eg = EGraph::with_rules(crate::optimize::standard_rules());
    let root_class = eg.add_arena(&encoded, root);
    eg.saturate();

    let (out, out_root, _cost) = extract(&eg, root_class, &CostModel::default());
    if contains_dwrt(&out) {
        return None;
    }

    // Var(16+i) -> Param(i).
    let decoded_nodes: Vec<ExprNode> = out
        .nodes_raw()
        .iter()
        .map(|n| match n {
            ExprNode::Var(i) if *i >= PARAM_VAR_BASE => ExprNode::Param(i - PARAM_VAR_BASE),
            other => other.clone(),
        })
        .collect();
    let decoded = ExprArena::from_raw(decoded_nodes, out.nary_children_raw().to_vec());
    Some((decoded, out_root))
}

fn contains_dwrt(arena: &ExprArena) -> bool {
    arena.nodes_raw().iter().any(|n| {
        matches!(
            n,
            pixelflow_ir::arena::ExprNode::Binary(OpKind::Dwrt, _, _)
                | pixelflow_ir::arena::ExprNode::Unary(OpKind::Dwrt, _)
                | pixelflow_ir::arena::ExprNode::Ternary(OpKind::Dwrt, _, _, _)
        )
    })
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
        OpKind::Add => quote! { ::pixelflow_core::__macro::ir::OpKind::Add },
        OpKind::Sub => quote! { ::pixelflow_core::__macro::ir::OpKind::Sub },
        OpKind::Mul => quote! { ::pixelflow_core::__macro::ir::OpKind::Mul },
        OpKind::Div => quote! { ::pixelflow_core::__macro::ir::OpKind::Div },
        OpKind::Neg => quote! { ::pixelflow_core::__macro::ir::OpKind::Neg },
        OpKind::Sqrt => quote! { ::pixelflow_core::__macro::ir::OpKind::Sqrt },
        OpKind::Rsqrt => quote! { ::pixelflow_core::__macro::ir::OpKind::Rsqrt },
        OpKind::Recip => quote! { ::pixelflow_core::__macro::ir::OpKind::Recip },
        OpKind::Abs => quote! { ::pixelflow_core::__macro::ir::OpKind::Abs },
        OpKind::Min => quote! { ::pixelflow_core::__macro::ir::OpKind::Min },
        OpKind::Max => quote! { ::pixelflow_core::__macro::ir::OpKind::Max },
        OpKind::MulAdd => quote! { ::pixelflow_core::__macro::ir::OpKind::MulAdd },
        OpKind::Sin => quote! { ::pixelflow_core::__macro::ir::OpKind::Sin },
        OpKind::Cos => quote! { ::pixelflow_core::__macro::ir::OpKind::Cos },
        OpKind::Atan => quote! { ::pixelflow_core::__macro::ir::OpKind::Atan },
        OpKind::Asin => quote! { ::pixelflow_core::__macro::ir::OpKind::Asin },
        OpKind::Acos => quote! { ::pixelflow_core::__macro::ir::OpKind::Acos },
        OpKind::Atan2 => quote! { ::pixelflow_core::__macro::ir::OpKind::Atan2 },
        OpKind::Tan => quote! { ::pixelflow_core::__macro::ir::OpKind::Tan },
        OpKind::Exp => quote! { ::pixelflow_core::__macro::ir::OpKind::Exp },
        OpKind::Exp2 => quote! { ::pixelflow_core::__macro::ir::OpKind::Exp2 },
        OpKind::Ln => quote! { ::pixelflow_core::__macro::ir::OpKind::Ln },
        OpKind::Log2 => quote! { ::pixelflow_core::__macro::ir::OpKind::Log2 },
        OpKind::Log10 => quote! { ::pixelflow_core::__macro::ir::OpKind::Log10 },
        OpKind::Pow => quote! { ::pixelflow_core::__macro::ir::OpKind::Pow },
        OpKind::Floor => quote! { ::pixelflow_core::__macro::ir::OpKind::Floor },
        OpKind::Ceil => quote! { ::pixelflow_core::__macro::ir::OpKind::Ceil },
        OpKind::Round => quote! { ::pixelflow_core::__macro::ir::OpKind::Round },
        OpKind::Lt => quote! { ::pixelflow_core::__macro::ir::OpKind::Lt },
        OpKind::Le => quote! { ::pixelflow_core::__macro::ir::OpKind::Le },
        OpKind::Gt => quote! { ::pixelflow_core::__macro::ir::OpKind::Gt },
        OpKind::Ge => quote! { ::pixelflow_core::__macro::ir::OpKind::Ge },
        OpKind::Eq => quote! { ::pixelflow_core::__macro::ir::OpKind::Eq },
        OpKind::Ne => quote! { ::pixelflow_core::__macro::ir::OpKind::Ne },
        OpKind::Select => quote! { ::pixelflow_core::__macro::ir::OpKind::Select },
        // Mask combination (canonical masks in both tiers).
        OpKind::BitAnd => quote! { ::pixelflow_core::__macro::ir::OpKind::BitAnd },
        OpKind::BitOr => quote! { ::pixelflow_core::__macro::ir::OpKind::BitOr },
        // Lowered at runtime by pixelflow-ir's `lower_dwrt` before codegen.
        OpKind::Dwrt => quote! { ::pixelflow_core::__macro::ir::OpKind::Dwrt },
        _ => panic!("Unsupported OpKind for JIT: {:?}", kind),
    }
}

#[cfg(test)]
mod expansion_derivative_tests {
    use super::*;
    use pixelflow_ir::arena::ExprNode;
    use pixelflow_ir::binding::BindingTable;
    use pixelflow_ir::eval::eval_scalar;
    use pixelflow_ir::passes::lower_dwrt_owned;

    /// The optimizer's contract, checked differentially: whatever it returns
    /// for a `Dwrt`-carrying arena must agree numerically with the runtime
    /// `lower_dwrt` tier — two independent implementations of the same
    /// calculus checking each other. `None` (budget miss → fallback) is
    /// always legitimate and asserts nothing; a `Some` must be honest:
    /// `Dwrt`-free, params round-tripped, and mathematically equivalent.
    /// Whether the output is also *cheaper* is a cost-model concern for the
    /// bench harness (the `pixelflow-pipeline` bench bins), not a unit test.
    ///
    /// These tests build arenas by hand and call the private
    /// `differentiate_in_optimizer` directly (STYLE.md "Test Public API"
    /// exception): the claim under test is that this expansion-time pass
    /// agrees numerically with the independent runtime `lower_dwrt` tier,
    /// which is only checkable by comparing the two tiers' arena-level
    /// output directly — the public `ast_to_runtime_arena` entry point only
    /// exposes generated `TokenStream`, not the intermediate arena needed for
    /// this differential check.
    fn assert_matches_runtime_tier(a: &ExprArena, root: ExprId, params: &[f32], pts: &[[f32; 4]]) {
        let Some((out, out_root)) = differentiate_in_optimizer(a, root) else {
            return; // fallback tier's job; nothing claimed, nothing to check
        };

        assert!(
            !super::contains_dwrt(&out),
            "Some(..) must be Dwrt-free — that is the claim it makes"
        );
        assert!(
            !out.nodes_raw()
                .iter()
                .any(|n| matches!(n, ExprNode::Var(i) if *i >= PARAM_VAR_BASE)),
            "encoded param Var leaked through the round-trip undecoded"
        );

        // Reference: substitute params into the ORIGINAL arena and run the
        // runtime lowering tier on it.
        let mut reference = a.clone();
        let ref_root = reference.substitute_params(root, params);
        let (ref_arena, ref_root) =
            lower_dwrt_owned(&reference, ref_root).expect("runtime tier lowers the reference");

        let mut got = out.clone();
        let got_root = got.substitute_params(out_root, params);

        for p in pts {
            let want = eval_scalar(&ref_arena, ref_root, p, &BindingTable::empty());
            let g = eval_scalar(&got, got_root, p, &BindingTable::empty());
            let tol = 1e-3 * want.abs().max(1.0);
            assert!(
                (g - want).abs() <= tol,
                "at {p:?}: optimizer={g}, runtime tier={want} (tol {tol})"
            );
        }
    }

    /// d/dx (p0 · √(x² + y²)) — a scalar param inside the differentiand
    /// exercises the Param ↔ Var(16+) round-trip alongside the calculus.
    #[test]
    fn param_derivative_matches_runtime_tier() {
        let mut a = ExprArena::new();
        let p0 = a.push_param(0);
        let x = a.push_var(0);
        let y = a.push_var(1);
        let x2 = a.push_binary(OpKind::Mul, x, x);
        let y2 = a.push_binary(OpKind::Mul, y, y);
        let sum = a.push_binary(OpKind::Add, x2, y2);
        let dist = a.push_unary(OpKind::Sqrt, sum);
        let e = a.push_binary(OpKind::Mul, p0, dist);
        let root = push_dwrt(&mut a, e, 0);

        assert_matches_runtime_tier(
            &a,
            root,
            &[2.0],
            &[
                [3.0, 4.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [-2.0, 5.0, 0.0, 0.0],
            ],
        );
    }

    /// The piecewise font ramp: min/max over a gradient-normalized ratio,
    /// with two shared `Dwrt` sites.
    #[test]
    fn piecewise_ramp_matches_runtime_tier() {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let d = a.push_binary(OpKind::Sub, x, y);
        let dx = push_dwrt(&mut a, d, 0);
        let dy = push_dwrt(&mut a, d, 1);
        let dx2 = a.push_binary(OpKind::Mul, dx, dx);
        let dy2 = a.push_binary(OpKind::Mul, dy, dy);
        let s = a.push_binary(OpKind::Add, dx2, dy2);
        let grad = a.push_unary(OpKind::Sqrt, s);
        let ratio = a.push_binary(OpKind::Div, d, grad);
        let zero = a.push_const(0.0);
        let one = a.push_const(1.0);
        let mx = a.push_binary(OpKind::Max, ratio, zero);
        let root = a.push_binary(OpKind::Min, mx, one);

        assert_matches_runtime_tier(
            &a,
            root,
            &[],
            &[
                [2.0, 1.0, 0.0, 0.0],
                [0.5, 0.2, 0.0, 0.0],
                [-1.0, 1.0, 0.0, 0.0],
            ],
        );
    }

    /// No `Dwrt` -> nothing to do; the caller keeps the original arena (and
    /// the AST-level optimizer's existing output is not perturbed).
    #[test]
    fn differentiate_in_optimizer_returns_none_when_arena_has_no_dwrt() {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let root = a.push_binary(OpKind::Add, x, y);
        assert!(differentiate_in_optimizer(&a, root).is_none());
    }
}

/// Production saturation telemetry, stage 2 of 2 (the `Dwrt` macro-time
/// e-graph; stage 1 is `pixelflow-search/src/runtime.rs`'s
/// `production_telemetry`, docs/results/2026-09-01-production-saturation-telemetry.md).
///
/// [`differentiate_in_optimizer`] (this file, above) is production's *other*
/// saturation site: it runs at `kernel_value!` macro-expansion time, on
/// every kernel whose body contains `Dwrt` (`DX`/`DY`), with a budget that is
/// NOT `config_for_node_count`-tiered — it is the hardcoded "standard
/// optimizer budget" `eg.saturate()` = 100 iterations / 10,000 classes /
/// 500 ms (`graph.rs:811-813`), called at `:722` above with the result
/// discarded exactly like `optimize_runtime_arena_uncached` discards its
/// `SaturationResult`. core-term's one reachable user of this path is the
/// glyph winding segment's coverage kernel
/// (`pixelflow-graphics/src/fonts/ttf_curve_analytical.rs:106-129`, a
/// `kernel_value!` whose `grad` term differentiates `d` through `DX`/`DY`).
///
/// This module drives the *exact* front-end pipeline the `kernel_value!`
/// proc-macro fn runs (`lib.rs:223-238`: `parser::parse` → `sema::analyze` →
/// `optimize::optimize` — a FIRST, algebra-only e-graph, AST-level, sized by
/// `config_for_node_count` on raw AST node count, distinct from the Dwrt
/// e-graph and measured here too for completeness) on the winding kernel's
/// literal closure source, reaching the same `AnalyzedKernel` that
/// `jit_backend::emit_kernel_value` (`:306-339`) would feed to
/// `ast_to_runtime_arena` (`:468-482`) — then replays
/// `differentiate_in_optimizer`'s own body verbatim (same param encoding,
/// same rule set, same extraction) with `saturate_with_limits` standing in
/// for `saturate()` so `SaturationStats` and provenance survive instead of
/// being dropped.
///
/// There is exactly one kernel to measure here, not a sweep: every glyph
/// segment shares this one closure body (only the 7 scalar params — the
/// endpoints and slope — differ per segment, substituted after arena
/// construction), so the arena `differentiate_in_optimizer` sees is
/// structurally identical for every call site. `param_indices` are threaded
/// through as arena `Param` leaves either way (see the encode step below),
/// so the measured arena does not depend on which segment's numbers would
/// eventually be substituted.
///
/// Nothing here changes production behavior: no signature, no visibility,
/// no code outside `#[cfg(test)]`.
#[cfg(test)]
mod production_telemetry {
    use super::*;
    use pixelflow_search::egraph::{CostModel, EGraph, SaturationStats, SaturationStop, extract};
    use std::time::{Duration, Instant};

    /// Verbatim closure body from
    /// `pixelflow-graphics/src/fonts/ttf_curve_analytical.rs:106-118` (the
    /// `kernel_value!` argument, up through the closing `}` of the body —
    /// the call-site arguments on `:119-127` are runtime values, irrelevant
    /// to the arena's shape). `crate::parser::parse` expects exactly the
    /// tokens between the macro's parens, per its own doc example
    /// (`lib.rs:217`: `kernel_value!(|cx: f32, r: f32| (X - cx) * r)`).
    const WINDING_KERNEL_SRC: &str = r#"
        |x0: f32,
         y0: f32,
         dx_over_dy: f32,
         dir: f32,
         y_min: f32,
         y_max: f32,
         min_grad: f32|
         -> Field {
            let in_y = (Y >= y_min) & (Y < y_max);
            let d = X - ((Y - y0) * dx_over_dy + x0);
            let grad = (DX(d.clone()) * DX(d.clone()) + DY(d.clone()) * DY(d.clone())).sqrt();
            let coverage = (V(d) / (grad + V(min_grad)) + V(0.5))
                .max(V(0.0))
                .min(V(1.0));
            in_y.select(coverage * V(dir), V(0.0))
        }
    "#;

    fn reachable_count(arena: &ExprArena, root: ExprId) -> usize {
        let len = arena.nodes_raw().len();
        let mut seen = vec![false; len];
        let mut stack = vec![root];
        let mut n = 0usize;
        while let Some(id) = stack.pop() {
            if core::mem::replace(&mut seen[id.0 as usize], true) {
                continue;
            }
            n += 1;
            stack.extend(arena.children(id));
        }
        n
    }

    /// Latency-prior DAG cost (leaves free, each reachable op counted once) —
    /// NOT `extract`'s own returned cost, which (like `extract_dag`'s
    /// `total_cost`) folds in a 1,000,000-per-cycle `CYCLE_COST` penalty
    /// (`extract.rs:958`) that does not describe emitted code. See
    /// `runtime.rs`'s `arena_cost` for the sibling of this function.
    fn arena_cost(arena: &ExprArena, root: ExprId, costs: &CostModel) -> usize {
        use pixelflow_ir::arena::ExprNode;
        let len = arena.nodes_raw().len();
        let mut seen = vec![false; len];
        let mut stack = vec![root];
        let mut total = 0usize;
        while let Some(id) = stack.pop() {
            if core::mem::replace(&mut seen[id.0 as usize], true) {
                continue;
            }
            let kind = match arena.node(id) {
                ExprNode::Var(_) | ExprNode::Const(_) | ExprNode::Buffer(_) => None,
                ExprNode::Unary(k, _)
                | ExprNode::Binary(k, _, _)
                | ExprNode::Ternary(k, _, _, _) => Some(*k),
                other @ (ExprNode::Param(_) | ExprNode::Nary(..)) => {
                    panic!(
                        "winding-kernel Dwrt arena contains {other:?}, unexpected pre-extraction"
                    )
                }
            };
            if let Some(k) = kind {
                total += costs.cost(k);
            }
            stack.extend(arena.children(id));
        }
        total
    }

    struct DwrtRun {
        stop: SaturationStop,
        iterations: usize,
        total_unions: usize,
        classes_after: usize,
        applications: usize,
        journal_unions: usize,
        elapsed: Duration,
        cost: usize,
        dwrt_survived: bool,
    }

    /// Ops in `arena` that the e-graph cannot represent — the same test
    /// `differentiate_in_optimizer` runs at `:687-694` before ever building
    /// an `EGraph`, replicated (not called) so the caller can report *which*
    /// op blocked it rather than the production function's plain `None`.
    /// Empty means representable.
    fn unrepresentable_ops(arena: &ExprArena) -> Vec<OpKind> {
        use pixelflow_ir::arena::ExprNode;
        let mut bad: Vec<OpKind> = arena
            .nodes_raw()
            .iter()
            .filter_map(|n| match n {
                ExprNode::Unary(op, _)
                | ExprNode::Binary(op, _, _)
                | ExprNode::Ternary(op, _, _, _) => {
                    (pixelflow_search::egraph::ops::op_from_kind(*op).is_none()).then_some(*op)
                }
                ExprNode::Buffer(_) => Some(OpKind::Buffer), // ir_bridge.rs:690: Buffer => false unconditionally
                ExprNode::Var(_) | ExprNode::Const(_) | ExprNode::Param(_) => None,
                ExprNode::Nary(op, _, _) => {
                    (pixelflow_search::egraph::ops::op_from_kind(*op).is_none()).then_some(*op)
                }
            })
            .collect();
        bad.sort_by_key(|k| format!("{k:?}"));
        bad.dedup();
        bad
    }

    /// [`differentiate_in_optimizer`]'s own body (`:661-732`), replayed with
    /// budget as parameters and `saturate_with_limits` in place of
    /// `saturate()` so the discarded `SaturationStats` and the (unconditional,
    /// per `graph.rs:137,157`) provenance counters survive. Encode/decode and
    /// rule set are copied verbatim; nothing in the production function is
    /// called or modified. Returns `Err(blocking ops)` exactly where
    /// production's `representable` guard (`:687-694`) would return `None`
    /// WITHOUT ever constructing an `EGraph` — i.e. zero saturation, not a
    /// truncated one.
    fn run_dwrt_egraph(
        arena: &ExprArena,
        root: ExprId,
        max_iterations: usize,
        max_classes: usize,
        timeout: Duration,
    ) -> Result<DwrtRun, Vec<OpKind>> {
        use pixelflow_ir::arena::ExprNode;
        let bad = unrepresentable_ops(arena);
        if !bad.is_empty() {
            return Err(bad);
        }

        // ir_bridge.rs:697-712 — Param(i) -> Var(16+i) encoding.
        let encoded_nodes: Vec<ExprNode> = arena
            .nodes_raw()
            .iter()
            .map(|n| match n {
                ExprNode::Param(i) => {
                    assert!(
                        PARAM_VAR_BASE + i < MANIFOLD_SLOT_BASE,
                        "too many scalar params to encode"
                    );
                    ExprNode::Var(PARAM_VAR_BASE + i)
                }
                other => other.clone(),
            })
            .collect();
        let encoded = ExprArena::from_raw(encoded_nodes, arena.nary_children_raw().to_vec());

        // ir_bridge.rs:718-722, with saturate_with_limits standing in for saturate().
        let mut eg = EGraph::with_rules(crate::optimize::standard_rules());
        let root_class = eg.add_arena(&encoded, root);
        let started = Instant::now();
        let stats: SaturationStats = eg.saturate_with_limits(max_iterations, max_classes, timeout);
        let elapsed = started.elapsed();

        // ir_bridge.rs:724 — extract's own returned cost carries the
        // cycle-penalty caveat above; recompute from the extracted arena.
        let costs = CostModel::latency_prior();
        let (out, out_root, _extract_reported_cost) = extract(&eg, root_class, &costs);
        let dwrt_survived = contains_dwrt(&out);
        let cost = if dwrt_survived {
            // A Dwrt-carrying "extraction" is not code arena_cost can price
            // (Dwrt has no CostModel entry) — the real fallback (:733)
            // recompiles the ORIGINAL arena unresolved, at the runtime tier;
            // report the pre-extraction node count's cost as N/A via 0 and
            // let `dwrt_survived` carry the signal instead of a fabricated number.
            0
        } else {
            arena_cost(&out, out_root, &costs)
        };

        Ok(DwrtRun {
            stop: stats.stop,
            iterations: stats.iterations,
            total_unions: stats.total_unions,
            classes_after: eg.num_classes(),
            applications: eg.provenance().application_count(),
            journal_unions: eg.provenance().union_count(),
            elapsed,
            cost,
            dwrt_survived,
        })
    }

    /// Drives `parser::parse` -> `sema::analyze` -> `optimize::optimize` ->
    /// `ast_to_arena`, i.e. everything `jit_backend::emit_kernel_value`
    /// (`:306-339`) does before calling `ast_to_runtime_arena` — reaching the
    /// identical pre-Dwrt-resolution arena, without going through
    /// `ast_to_runtime_arena` itself (which would call the production
    /// `differentiate_in_optimizer` and discard the stats we need).
    #[test]
    #[ignore = "measurement: cargo test -p pixelflow-compiler --release -- --ignored winding_kernel_dwrt_egraph_telemetry --nocapture"]
    fn winding_kernel_dwrt_egraph_telemetry() {
        assert!(
            std::env::var("PIXELFLOW_NNUE_WEIGHTS").is_err(),
            "PIXELFLOW_NNUE_WEIGHTS is set; optimize::optimize's e-graph #1 would use it — unset it"
        );

        // lib.rs:224-227
        let tokens: TokenStream = WINDING_KERNEL_SRC
            .parse()
            .expect("lex winding kernel source");
        let kernel_ast = crate::parser::parse(tokens).expect("parse winding kernel source");
        // lib.rs:228-231
        let analyzed = crate::sema::analyze(kernel_ast).expect("sema winding kernel");
        // lib.rs:236 — e-graph #1 (algebra, AST-level, config_for_node_count-tiered).
        let algebra_started = Instant::now();
        let analyzed = crate::optimize::optimize(analyzed);
        let algebra_elapsed = algebra_started.elapsed();

        // jit_backend.rs:307-325 gates (must all pass, or emit_kernel_value
        // would never reach ast_to_runtime_arena for this kernel).
        assert!(
            analyzed.def.struct_decl.is_none(),
            "winding kernel must be a named-struct-free fragment"
        );
        let param_map = scalar_param_indices(&analyzed);
        let manifold_map = manifold_param_indices(&analyzed);
        assert!(
            manifold_map.is_empty(),
            "winding kernel takes no manifold params"
        );
        assert_eq!(
            param_map.len(),
            7,
            "winding kernel has 7 scalar params (x0,y0,dx_over_dy,dir,y_min,y_max,min_grad)"
        );

        // ir_bridge.rs:471-472 — the arena differentiate_in_optimizer receives.
        let mut arena = ExprArena::new();
        let (root, plan) = ast_to_arena(&analyzed.def.body, &param_map, &manifold_map, &mut arena)
            .expect("ast_to_arena on winding kernel body");
        assert!(
            plan.is_empty(),
            "winding kernel is non-composing; a non-empty plan would make ast_to_runtime_arena SKIP differentiate_in_optimizer entirely (ir_bridge.rs:477)"
        );
        assert!(
            contains_dwrt(&arena),
            "winding kernel body must contain Dwrt (DX/DY) after e-graph #1 — nothing to measure otherwise"
        );

        let node_count = reachable_count(&arena, root);
        println!(
            "winding kernel Dwrt arena: {node_count} reachable nodes (post-algebra-e-graph, pre-Dwrt-resolution)"
        );
        println!(
            "algebra e-graph (#1, AST-level) wall-clock: {:.1}ms",
            algebra_elapsed.as_secs_f64() * 1e3
        );

        // Production budget (ir_bridge.rs:718-722): NOT config_for_node_count
        // -tiered, the hardcoded EGraph::saturate() default.
        const PROD_MAX_ITERS: usize = 100;
        const PROD_MAX_CLASSES: usize = 10_000;
        const PROD_TIMEOUT_MS: u64 = 500;
        let mult: usize = std::env::var("PIXELFLOW_TELEMETRY_REF_MULT")
            .map(|s| s.parse().expect("REF_MULT must be an integer"))
            .unwrap_or(4);
        let ceiling = Duration::from_secs(
            std::env::var("PIXELFLOW_TELEMETRY_REF_CEILING_S")
                .map(|s| s.parse().expect("REF_CEILING_S must be an integer"))
                .unwrap_or(600),
        );

        // Guard first (matches differentiate_in_optimizer:687-694 exactly):
        // if the arena is not e-graph-representable, production returns
        // `None` WITHOUT constructing an `EGraph` at all — zero saturation,
        // not a truncated one. Check once; all three runs would bail
        // identically since the guard depends only on the (budget-independent)
        // arena contents.
        if let Err(bad_ops) = run_dwrt_egraph(
            &arena,
            root,
            PROD_MAX_ITERS,
            PROD_MAX_CLASSES,
            Duration::from_millis(PROD_TIMEOUT_MS),
        ) {
            println!(
                "RESULT: differentiate_in_optimizer bails at the representable guard (ir_bridge.rs:687-694) \
                 BEFORE constructing an EGraph — blocking op(s): {bad_ops:?}. Zero saturation happens for this \
                 kernel at macro-expansion time; Dwrt survives to ast_to_runtime_arena's fallback (:733), which \
                 emits the Dwrt-carrying arena unchanged, and the runtime `lower_dwrt` SYMBOLIC PASS (not an \
                 e-graph — runtime.rs:120) resolves it before the runtime tier's e-graph saturates the composed \
                 glyph arena. That runtime e-graph is the SAME one already measured per-glyph in \
                 pixelflow-search's production_telemetry (stage 1) — this kernel's Dwrt resolution has no \
                 budget of its own to report."
            );
            return;
        }
        let prod = run_dwrt_egraph(
            &arena,
            root,
            PROD_MAX_ITERS,
            PROD_MAX_CLASSES,
            Duration::from_millis(PROD_TIMEOUT_MS),
        )
        .expect("guard already checked representable");
        let refr = run_dwrt_egraph(
            &arena,
            root,
            PROD_MAX_ITERS * mult,
            PROD_MAX_CLASSES,
            ceiling,
        )
        .expect("guard already checked representable");
        assert!(
            refr.elapsed < ceiling,
            "reference run hit its {ceiling:?} safety ceiling — raise PIXELFLOW_TELEMETRY_REF_CEILING_S and re-run"
        );
        let lifted = run_dwrt_egraph(
            &arena,
            root,
            PROD_MAX_ITERS * mult,
            PROD_MAX_CLASSES * mult,
            ceiling,
        )
        .expect("guard already checked representable");
        assert!(
            lifted.elapsed < ceiling,
            "cap-lifted run hit its {ceiling:?} safety ceiling — raise PIXELFLOW_TELEMETRY_REF_CEILING_S and re-run"
        );

        assert!(
            !prod.dwrt_survived,
            "production run left Dwrt unresolved (budget miss) — the runtime lower_dwrt fallback would fire for every glyph; this changes the flat-answer conclusion, do not treat as a normal row"
        );
        assert!(
            !refr.dwrt_survived && !lifted.dwrt_survived,
            "reference/lifted run left Dwrt unresolved despite a larger budget — investigate before trusting cost numbers"
        );

        // Read off the loop's own decision (`SaturationStats::stop`),
        // never inferred from the reference runs; those exist only to price
        // the truncation.
        let stop = prod.stop;
        let loss_vs_ref = (prod.cost as f64 - refr.cost as f64) / refr.cost as f64 * 100.0;
        let loss_vs_lifted = (prod.cost as f64 - lifted.cost as f64) / lifted.cost as f64 * 100.0;

        println!(
            "prod:   stop={stop:?} iters={}/{PROD_MAX_ITERS} classes={} apps={} unions={} journal_unions={} elapsed_ms={:.1} cost={}",
            prod.iterations,
            prod.classes_after,
            prod.applications,
            prod.total_unions,
            prod.journal_unions,
            prod.elapsed.as_secs_f64() * 1e3,
            prod.cost
        );
        println!(
            "ref({mult}x iters, same class cap): iters={} classes={} apps={} elapsed_ms={:.1} cost={} loss_vs_ref={loss_vs_ref:.2}%",
            refr.iterations,
            refr.classes_after,
            refr.applications,
            refr.elapsed.as_secs_f64() * 1e3,
            refr.cost
        );
        println!(
            "lifted({mult}x iters, {mult}x classes): iters={} classes={} apps={} elapsed_ms={:.1} cost={} loss_vs_lifted={loss_vs_lifted:.2}%",
            lifted.iterations,
            lifted.classes_after,
            lifted.applications,
            lifted.elapsed.as_secs_f64() * 1e3,
            lifted.cost
        );
    }
}
