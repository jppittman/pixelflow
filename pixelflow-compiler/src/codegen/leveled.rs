//! # Leveled (BFS) Code Generation
//!
//! Emits code by evaluating expression trees level-by-level (breadth-first).
//! This approach:
//! 1. Naturally solves the nested manifold composition problem
//! 2. Exposes instruction-level parallelism (nodes at same level are independent)
//! 3. Makes register pressure explicit (each level is a "wavefront")
//! 4. **Enables uniform hoisting**: W-only expressions can be precomputed per-frame
//!
//! ## Architecture
//!
//! Given expression `(inner - r) * scale`:
//! ```text
//!         *           <- Level 2
//!        / \
//!       -   scale     <- Level 1
//!      / \
//!   inner  r          <- Level 0 (leaves)
//! ```
//!
//! Emits:
//! ```rust,ignore
//! let __l0_0 = inner.eval(__p);  // Manifold param evaluated
//! let __l0_1 = Field::from(r);   // Scalar param converted
//! let __l0_2 = Field::from(scale);
//! let __l1_0 = __l0_0 - __l0_1;  // Level 1 uses level 0 values
//! let __l2_0 = __l1_0 * __l0_2;  // Level 2 uses level 0 and 1
//! __l2_0
//! ```
//!
//! Each level only references already-evaluated values from previous levels.
//!
//! ## Dependency Tracking for Uniform Hoisting
//!
//! Each node is classified by what it depends on:
//! - **Const**: Literal values, compile-time constants
//! - **Uniform**: Depends only on W (time/frame) - can be computed once per frame
//! - **Varying**: Depends on X, Y, or Z (spatial) - must be computed per pixel
//!
//! Uniform expressions like `(W * 0.7).sin()` can be hoisted out of the pixel loop.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use std::collections::HashMap;

use crate::annotate::AnnotatedExpr;
use crate::ast::{BinaryOp, ParamKind, UnaryOp};
use crate::sema::AnalyzedKernel;
use crate::symbol::SymbolKind;

/// Dependency classification for uniform hoisting.
///
/// Forms a lattice: Const < Uniform < Varying
/// Operations propagate upward: `Uniform + Varying = Varying`
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Deps {
    /// No dependencies - literal constant
    Const,
    /// Depends only on W (time) - compute once per frame
    Uniform,
    /// Depends on X, Y, or Z - compute per pixel
    Varying,
}

impl Deps {
    /// Join two dependencies (least upper bound in lattice)
    #[inline]
    pub fn join(self, other: Self) -> Self {
        self.max(other)
    }

    /// Is this uniform (can be hoisted)?
    #[inline]
    pub fn is_uniform(&self) -> bool {
        matches!(self, Deps::Const | Deps::Uniform)
    }
}

/// A node in the leveled representation, with dependency tracking.
#[derive(Debug, Clone)]
pub struct LeveledNode {
    pub kind: LeveledNodeKind,
    pub deps: Deps,
}

/// The kind of leveled node.
#[derive(Debug, Clone)]
pub enum LeveledNodeKind {
    /// Leaf: parameter reference (scalar or manifold)
    Param { name: String, kind: ParamKind },
    /// Leaf: literal value
    Literal { value: f64 },
    /// Leaf: intrinsic (X, Y, Z, W)
    Intrinsic { name: String },
    /// Leaf: local variable reference
    Local { name: String },
    /// Binary operation referencing two prior nodes
    Binary {
        op: BinaryOp,
        left: NodeRef,
        right: NodeRef,
    },
    /// Unary operation referencing one prior node
    Unary { op: UnaryOp, operand: NodeRef },
    /// Method call on a node
    Method {
        receiver: NodeRef,
        method: String,
        args: Vec<NodeRef>,
    },
}

/// Reference to a node by (level, index within level)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeRef {
    pub level: usize,
    pub index: usize,
}

impl NodeRef {
    pub fn new(level: usize, index: usize) -> Self {
        Self { level, index }
    }

    /// Generate the variable name for this node reference
    pub fn var_name(&self) -> syn::Ident {
        format_ident!("__l{}_{}", self.level, self.index)
    }
}

/// Leveled representation of an expression tree.
pub struct LeveledExpr {
    /// Nodes organized by level (level 0 = leaves)
    pub levels: Vec<Vec<LeveledNode>>,
    /// The root node reference
    pub root: NodeRef,
}

/// Context for building a leveled expression.
struct LevelBuilder<'a> {
    analyzed: &'a AnalyzedKernel,
    /// Maps expression pointer to its node reference (for CSE)
    expr_to_ref: HashMap<usize, NodeRef>,
    /// Nodes at each level
    levels: Vec<Vec<LeveledNode>>,
}

impl<'a> LevelBuilder<'a> {
    fn new(analyzed: &'a AnalyzedKernel) -> Self {
        Self {
            analyzed,
            expr_to_ref: HashMap::new(),
            levels: Vec::new(),
        }
    }

    /// Build leveled representation from annotated expression.
    /// Returns the root node reference.
    fn build(&mut self, expr: &AnnotatedExpr) -> NodeRef {
        // First pass: compute depth of each node (DFS)
        let depths = self.compute_depths(expr);
        let max_depth = depths.values().copied().max().unwrap_or(0);

        // Initialize levels
        self.levels = vec![Vec::new(); max_depth + 1];

        // Second pass: assign nodes to levels (post-order to ensure dependencies first)
        self.assign_to_levels(expr, &depths)
    }

    /// Compute the depth of each subexpression (0 = leaf)
    fn compute_depths(&self, expr: &AnnotatedExpr) -> HashMap<usize, usize> {
        let mut depths = HashMap::new();
        self.compute_depth_recursive(expr, &mut depths);
        depths
    }

    fn compute_depth_recursive(
        &self,
        expr: &AnnotatedExpr,
        depths: &mut HashMap<usize, usize>,
    ) -> usize {
        let ptr = expr as *const _ as usize;

        let depth = match expr {
            // Leaves have depth 0
            AnnotatedExpr::Ident(_) | AnnotatedExpr::Literal(_) | AnnotatedExpr::Verbatim(_) => 0,

            // Unary: depth = child + 1
            AnnotatedExpr::Unary(unary) => {
                let child_depth = self.compute_depth_recursive(&unary.operand, depths);
                child_depth + 1
            }

            // Binary: depth = max(left, right) + 1
            AnnotatedExpr::Binary(binary) => {
                let left_depth = self.compute_depth_recursive(&binary.lhs, depths);
                let right_depth = self.compute_depth_recursive(&binary.rhs, depths);
                left_depth.max(right_depth) + 1
            }

            // Method call: depth = max(receiver, args) + 1
            AnnotatedExpr::MethodCall(call) => {
                let recv_depth = self.compute_depth_recursive(&call.receiver, depths);
                let args_depth = call
                    .args
                    .iter()
                    .map(|a| self.compute_depth_recursive(a, depths))
                    .max()
                    .unwrap_or(0);
                recv_depth.max(args_depth) + 1
            }

            // Block: depth of final expression
            AnnotatedExpr::Block(block) => {
                if let Some(ref final_expr) = block.expr {
                    self.compute_depth_recursive(final_expr, depths)
                } else {
                    0
                }
            }

            // Paren: same as inner
            AnnotatedExpr::Paren(inner) => self.compute_depth_recursive(inner, depths),

            // Tuple: max of elements + 1
            AnnotatedExpr::Tuple(tuple) => {
                let max_elem = tuple
                    .elems
                    .iter()
                    .map(|e| self.compute_depth_recursive(e, depths))
                    .max()
                    .unwrap_or(0);
                max_elem + 1
            }

            // Call: treat like method call
            AnnotatedExpr::Call(call) => {
                let args_depth = call
                    .args
                    .iter()
                    .map(|a| self.compute_depth_recursive(a, depths))
                    .max()
                    .unwrap_or(0);
                args_depth + 1
            }
        };

        depths.insert(ptr, depth);
        depth
    }

    /// Assign nodes to levels and return root reference.
    /// Also computes dependency classification for uniform hoisting.
    fn assign_to_levels(
        &mut self,
        expr: &AnnotatedExpr,
        depths: &HashMap<usize, usize>,
    ) -> NodeRef {
        let ptr = expr as *const _ as usize;

        // Check if already processed (CSE)
        if let Some(&node_ref) = self.expr_to_ref.get(&ptr) {
            return node_ref;
        }

        let depth = depths[&ptr];

        let (kind, deps) = match expr {
            AnnotatedExpr::Ident(ident) => {
                let name = ident.name.to_string();
                match self.analyzed.symbols.lookup(&name) {
                    Some(sym) => match sym.kind {
                        SymbolKind::Intrinsic => {
                            // X, Y, Z are Varying; W is Uniform
                            let deps = match name.as_str() {
                                "W" => Deps::Uniform,
                                "X" | "Y" | "Z" => Deps::Varying,
                                _ => Deps::Varying, // Conservative default
                            };
                            (LeveledNodeKind::Intrinsic { name }, deps)
                        }
                        SymbolKind::Parameter => {
                            // Look up param kind
                            let param_kind = self
                                .analyzed
                                .def
                                .params
                                .iter()
                                .find(|p| p.name.to_string() == name)
                                .map(|p| p.kind.clone())
                                .unwrap_or(ParamKind::Scalar(syn::parse_quote!(f32)));
                            // Scalar parameters are Const (captured at kernel creation)
                            // Manifold parameters need evaluation, so Varying
                            let deps = match &param_kind {
                                ParamKind::Scalar(_) => Deps::Const,
                                ParamKind::Manifold => Deps::Varying,
                            };
                            (
                                LeveledNodeKind::Param {
                                    name,
                                    kind: param_kind,
                                },
                                deps,
                            )
                        }
                        SymbolKind::ManifoldParam => (
                            LeveledNodeKind::Param {
                                name,
                                kind: ParamKind::Manifold,
                            },
                            Deps::Varying,
                        ),
                        SymbolKind::Local => {
                            // Local variables - conservatively Varying
                            (LeveledNodeKind::Local { name }, Deps::Varying)
                        }
                    },
                    None => (LeveledNodeKind::Local { name }, Deps::Varying),
                }
            }

            AnnotatedExpr::Literal(lit) => {
                // Parse the literal value from syn::Lit
                let value = match &lit.lit {
                    syn::Lit::Float(f) => f.base10_parse::<f64>().unwrap_or(0.0),
                    syn::Lit::Int(i) => i.base10_parse::<f64>().unwrap_or(0.0),
                    _ => 0.0,
                };
                (LeveledNodeKind::Literal { value }, Deps::Const)
            }

            AnnotatedExpr::Unary(unary) => {
                let operand = self.assign_to_levels(&unary.operand, depths);
                let operand_deps = self.get_deps(operand);
                (
                    LeveledNodeKind::Unary {
                        op: unary.op.clone(),
                        operand,
                    },
                    operand_deps,
                )
            }

            AnnotatedExpr::Binary(binary) => {
                let left = self.assign_to_levels(&binary.lhs, depths);
                let right = self.assign_to_levels(&binary.rhs, depths);
                let deps = self.get_deps(left).join(self.get_deps(right));
                (
                    LeveledNodeKind::Binary {
                        op: binary.op.clone(),
                        left,
                        right,
                    },
                    deps,
                )
            }

            AnnotatedExpr::MethodCall(call) => {
                let receiver = self.assign_to_levels(&call.receiver, depths);
                let args: Vec<_> = call
                    .args
                    .iter()
                    .map(|a| self.assign_to_levels(a, depths))
                    .collect();
                // Join deps from receiver and all args
                let mut deps = self.get_deps(receiver);
                for &arg in &args {
                    deps = deps.join(self.get_deps(arg));
                }
                (
                    LeveledNodeKind::Method {
                        receiver,
                        method: call.method.to_string(),
                        args,
                    },
                    deps,
                )
            }

            AnnotatedExpr::Paren(inner) => {
                // Paren is transparent - return inner's reference
                return self.assign_to_levels(inner, depths);
            }

            AnnotatedExpr::Block(block) => {
                // For now, just handle final expression
                // TODO: handle let bindings
                if let Some(ref final_expr) = block.expr {
                    return self.assign_to_levels(final_expr, depths);
                } else {
                    // Empty block - emit unit? For now, treat as literal 0
                    (LeveledNodeKind::Literal { value: 0.0 }, Deps::Const)
                }
            }

            // TODO: handle more cases
            _ => {
                // Fallback: treat as literal 0
                (LeveledNodeKind::Literal { value: 0.0 }, Deps::Const)
            }
        };

        let node = LeveledNode { kind, deps };

        // Add to appropriate level
        let index = self.levels[depth].len();
        self.levels[depth].push(node);

        let node_ref = NodeRef::new(depth, index);
        self.expr_to_ref.insert(ptr, node_ref);
        node_ref
    }

    /// Get the deps for a node reference
    fn get_deps(&self, node_ref: NodeRef) -> Deps {
        self.levels[node_ref.level][node_ref.index].deps
    }
}

/// Statistics about dependency classification for uniform hoisting.
#[derive(Debug, Default)]
pub struct DepsStats {
    pub const_nodes: usize,
    pub uniform_nodes: usize,
    pub varying_nodes: usize,
    /// Uniform nodes that are children of varying nodes (hoisting opportunities)
    pub hoistable_nodes: usize,
}

impl DepsStats {
    /// Percentage of nodes that could potentially be hoisted
    pub fn hoisting_potential(&self) -> f64 {
        let total = self.const_nodes + self.uniform_nodes + self.varying_nodes;
        if total == 0 {
            0.0
        } else {
            (self.uniform_nodes as f64 / total as f64) * 100.0
        }
    }
}

/// Analyze the deps distribution in a leveled expression.
pub fn analyze_deps(analyzed: &AnalyzedKernel, annotated: &AnnotatedExpr) -> DepsStats {
    let mut builder = LevelBuilder::new(analyzed);
    let root = builder.build(annotated);
    let root_deps = builder.get_deps(root);

    let mut stats = DepsStats::default();

    for level in &builder.levels {
        for node in level {
            match node.deps {
                Deps::Const => stats.const_nodes += 1,
                Deps::Uniform => stats.uniform_nodes += 1,
                Deps::Varying => stats.varying_nodes += 1,
            }
        }
    }

    // Count hoistable: uniform nodes that are direct children of varying nodes
    // These represent actual hoisting opportunities
    for level in &builder.levels {
        for node in level {
            if node.deps == Deps::Varying {
                match &node.kind {
                    LeveledNodeKind::Unary { operand, .. } => {
                        if builder.get_deps(*operand) == Deps::Uniform {
                            stats.hoistable_nodes += 1;
                        }
                    }
                    LeveledNodeKind::Binary { left, right, .. } => {
                        if builder.get_deps(*left) == Deps::Uniform {
                            stats.hoistable_nodes += 1;
                        }
                        if builder.get_deps(*right) == Deps::Uniform {
                            stats.hoistable_nodes += 1;
                        }
                    }
                    LeveledNodeKind::Method { receiver, args, .. } => {
                        if builder.get_deps(*receiver) == Deps::Uniform {
                            stats.hoistable_nodes += 1;
                        }
                        for &arg in args {
                            if builder.get_deps(arg) == Deps::Uniform {
                                stats.hoistable_nodes += 1;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    stats
}

/// Emit leveled code from annotated expression.
pub fn emit_leveled(
    analyzed: &AnalyzedKernel,
    annotated: &AnnotatedExpr,
    use_jet_wrapper: bool,
) -> TokenStream {
    let mut builder = LevelBuilder::new(analyzed);
    let root = builder.build(annotated);

    // Generate code for each level
    let mut stmts = Vec::new();

    for (level_idx, level) in builder.levels.iter().enumerate() {
        for (node_idx, node) in level.iter().enumerate() {
            let var_name = NodeRef::new(level_idx, node_idx).var_name();

            let value = emit_node(node, use_jet_wrapper);
            stmts.push(quote! { let #var_name = #value; });
        }
    }

    let root_var = root.var_name();

    quote! {
        {
            #(#stmts)*
            #root_var
        }
    }
}

/// Emit code for a single node
fn emit_node(node: &LeveledNode, use_jet_wrapper: bool) -> TokenStream {
    match &node.kind {
        LeveledNodeKind::Param { name, kind } => {
            let name_ident = format_ident!("{}", name);
            match kind {
                ParamKind::Manifold => {
                    // Manifold params are evaluated at their level
                    quote! { #name_ident.eval(__p) }
                }
                ParamKind::Scalar(_) => {
                    if use_jet_wrapper {
                        quote! { __ScalarType::from_f32(#name_ident) }
                    } else {
                        quote! { ::pixelflow_core::Field::from(#name_ident) }
                    }
                }
            }
        }

        LeveledNodeKind::Literal { value } => {
            let lit = *value as f32;
            if use_jet_wrapper {
                quote! { __ScalarType::from_f32(#lit) }
            } else {
                quote! { ::pixelflow_core::Field::from(#lit) }
            }
        }

        LeveledNodeKind::Intrinsic { name } => {
            let name_ident = format_ident!("{}", name);
            // Intrinsics need to be evaluated at the point
            quote! { #name_ident.eval(__p) }
        }

        LeveledNodeKind::Local { name } => {
            let name_ident = format_ident!("{}", name);
            quote! { #name_ident }
        }

        LeveledNodeKind::Binary { op, left, right } => {
            let left_var = left.var_name();
            let right_var = right.var_name();

            match op {
                BinaryOp::Add => quote! { #left_var + #right_var },
                BinaryOp::Sub => quote! { #left_var - #right_var },
                BinaryOp::Mul => quote! { #left_var * #right_var },
                BinaryOp::Div => quote! { #left_var / #right_var },
                BinaryOp::Rem => quote! { #left_var % #right_var },
                BinaryOp::Lt => quote! { #left_var.lt(#right_var) },
                BinaryOp::Le => quote! { #left_var.le(#right_var) },
                BinaryOp::Gt => quote! { #left_var.gt(#right_var) },
                BinaryOp::Ge => quote! { #left_var.ge(#right_var) },
                BinaryOp::Eq => quote! { #left_var.eq(#right_var) },
                BinaryOp::Ne => quote! { #left_var.ne(#right_var) },
                BinaryOp::BitAnd => quote! { #left_var & #right_var },
                BinaryOp::BitOr => quote! { #left_var | #right_var },
            }
        }

        LeveledNodeKind::Unary { op, operand } => {
            let operand_var = operand.var_name();

            match op {
                UnaryOp::Neg => quote! { -#operand_var },
                UnaryOp::Not => quote! { !#operand_var },
            }
        }

        LeveledNodeKind::Method {
            receiver,
            method,
            args,
        } => {
            let recv_var = receiver.var_name();
            let method_ident = format_ident!("{}", method);
            let arg_vars: Vec<_> = args.iter().map(|a| a.var_name()).collect();

            quote! { #recv_var.#method_ident(#(#arg_vars),*) }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotate::{AnnotationCtx, annotate};
    use crate::parser::parse;
    use crate::sema::analyze;
    use quote::quote;

    fn analyze_input(input: proc_macro2::TokenStream) -> DepsStats {
        let kernel = parse(input).unwrap();
        let analyzed = analyze(kernel).unwrap();
        let ctx = AnnotationCtx::new();
        let (annotated_body, _, _) = annotate(&analyzed.def.body, ctx);
        analyze_deps(&analyzed, &annotated_body)
    }

    #[test]
    fn test_node_ref_var_name() {
        let r = NodeRef::new(2, 3);
        assert_eq!(r.var_name().to_string(), "__l2_3");
    }

    #[test]
    fn test_deps_const_only() {
        // Pure constant expression: 1.0 + 2.0
        let stats = analyze_input(quote! { || 1.0 + 2.0 });
        assert!(stats.varying_nodes == 0, "Expected no varying nodes");
        assert!(stats.uniform_nodes == 0, "Expected no uniform nodes");
        assert!(stats.const_nodes > 0, "Expected const nodes");
    }

    #[test]
    fn test_deps_varying_only() {
        // Pure varying: X + Y
        let stats = analyze_input(quote! { || X + Y });
        assert!(stats.varying_nodes > 0, "Expected varying nodes");
        // X and Y are varying leaves
        assert_eq!(stats.varying_nodes, 3, "Expected 3 varying: X, Y, Add");
    }

    #[test]
    fn test_deps_uniform_only() {
        // Pure uniform: W * 2.0
        let stats = analyze_input(quote! { || W * 2.0 });
        assert!(stats.const_nodes > 0, "Expected const nodes (2.0)");
        assert!(stats.uniform_nodes > 0, "Expected uniform nodes (W, W*2.0)");
        assert_eq!(stats.varying_nodes, 0, "Expected no varying");
    }

    #[test]
    fn test_deps_mixed() {
        // Mixed: (W * 0.5).sin() + X
        // W*0.5 and sin(...) are uniform
        // X is varying
        // The addition is varying
        let stats = analyze_input(quote! { || (W * 0.5).sin() + X });
        assert!(stats.const_nodes > 0, "Expected const (0.5)");
        assert!(stats.uniform_nodes > 0, "Expected uniform (W, W*0.5, sin)");
        assert!(stats.varying_nodes > 0, "Expected varying (X, +)");
        // The hoistable count: sin(...) is uniform child of varying Add
        assert!(
            stats.hoistable_nodes > 0,
            "Expected hoistable uniform nodes"
        );
    }

    #[test]
    fn test_deps_scalar_params_are_const() {
        // Scalar parameters captured at kernel creation are Const
        let stats = analyze_input(quote! { |r: f32| X - r });
        // r is Const (scalar param)
        // X is Varying
        // X - r is Varying
        assert!(stats.const_nodes > 0, "Expected const (r)");
        assert!(stats.varying_nodes > 0, "Expected varying (X, X-r)");
    }

    #[test]
    fn test_deps_lattice_join() {
        assert_eq!(Deps::Const.join(Deps::Const), Deps::Const);
        assert_eq!(Deps::Const.join(Deps::Uniform), Deps::Uniform);
        assert_eq!(Deps::Const.join(Deps::Varying), Deps::Varying);
        assert_eq!(Deps::Uniform.join(Deps::Uniform), Deps::Uniform);
        assert_eq!(Deps::Uniform.join(Deps::Varying), Deps::Varying);
        assert_eq!(Deps::Varying.join(Deps::Varying), Deps::Varying);
    }
}
