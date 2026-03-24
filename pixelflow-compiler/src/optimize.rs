//! # AST Optimization
//!
//! Performs algebraic simplification and constant folding on the AST.
//!
//! ## Two-Pass Architecture
//!
//! **Pass 1 (Structural)**: Tree-based peephole optimization
//! - Constant folding: `1.0 + 2.0` → `3.0`
//! - Identity removal: `x + 0.0` → `x`, `x * 1.0` → `x`
//! - Zero propagation: `x * 0.0` → `0.0`
//!
//! **Pass 2 (Global)**: E-graph equality saturation
//! - Processes entire kernel expression globally (across let bindings)
//! - FMA fusion: `a * b + c` → `MulAdd(a, b, c)` when profitable
//! - Rsqrt: `1 / sqrt(y)` → `rsqrt(y)` (real instruction)
//! - Algebraic identities discovered via rewrite rules
//!
//! The global pass sees through let bindings, enabling optimizations like:
//! ```text
//! let a = X * X;
//! let b = Y * Y;
//! (a + b).sqrt()  // E-graph sees: sqrt(X*X + Y*Y)
//! ```

use crate::ast::{
    BinaryExpr, BinaryOp, BlockExpr, CallExpr, Expr, IdentExpr, LetStmt, LiteralExpr, MethodCallExpr, Stmt,
    UnaryExpr, UnaryOp,
};
use crate::cost_builder;
use crate::ir_bridge::{ast_to_ir, egraph_to_ir, ir_to_code, IRToEGraphContext};
use crate::sema::AnalyzedKernel;
use pixelflow_search::egraph::{
    CostModel, EClassId, EGraph, ENode, ExprTree, ExtractedDAG, Leaf, ops,
    Rewrite, extract_neural,
};
use pixelflow_search::nnue::ExprNnue;
use pixelflow_search::math::all_rules as search_all_rules;
use proc_macro2::Span;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};
use syn::{Ident, Lit};

// ============================================================================
// Canonical Rule Set
// ============================================================================

/// All rewrite rules for PixelFlow optimization: 40 math + 3 fusion = 43 total.
///
/// Delegates to `pixelflow_search::math::all_rules()` which is the canonical
/// source of truth for all rewrite rules.
pub fn standard_rules() -> Vec<Box<dyn Rewrite>> {
    search_all_rules()
}

/// Counter for generating unique opaque variable names.
static OPAQUE_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// NNUE model for AOT optimization, loaded from trained weights.
static NNUE_MODEL: OnceLock<ExprNnue> = OnceLock::new();

/// Get the neural cost model, initializing it from scratch (random) or loaded weights.
fn get_nnue_model() -> &'static ExprNnue {
    NNUE_MODEL.get_or_init(|| {
        // PixelflowZero: Start from total ignorance.
        // In production, this would load from a .bin file.
        ExprNnue::new_random(42)
    })
}

/// Generate a unique name for an opaque expression (unknown method call, etc.)
fn unique_opaque_name(prefix: &str) -> String {
    let id = OPAQUE_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("__{}{}", prefix, id)
}

/// Heuristic score for rewrite priority.
///
/// Higher scores = apply sooner (more promising rewrites).
///
/// Priority order:
/// 1. **Identity/Annihilator** (x+0, x*1, x*0) - eliminate operations (highest)
/// 2. **FMA fusion** (a*b+c) - enable SIMD instructions
/// 3. **RecipSqrt** (1/sqrt(x)) - special instruction
/// 4. **Canonicalization** (normalize forms) - enables other matches
/// 5. **Fusion-enabling rewrites** (distribute, etc.)
/// 6. **Everything else** (commutative, etc.) - apply last
fn heuristic_score_rewrite(egraph: &EGraph, target: &pixelflow_search::egraph::RewriteTarget) -> i64 {
    use pixelflow_search::egraph::RewriteTarget;

    // Get the rule name
    let rule_name = match egraph.rule(target.rule_idx) {
        Some(rule) => rule.name(),
        None => return 0,
    };

    // Priority scoring based on rule type
    match rule_name {
        // Tier 1: Eliminate operations (identity, annihilator)
        "identity" | "annihilator" => 1000,

        // Tier 2: Enable SIMD (FMA fusion)
        "fma_fusion" => 800,

        // Tier 3: Special instructions
        "recip_sqrt" => 700,

        // Tier 4: Canonicalization (enables other matches)
        "canonicalize" => 600,

        // Tier 5: Fusion-enabling
        "distribute" | "factor" | "involution" | "cancellation" => 500,

        // Tier 6: Everything else (commutativity, etc.)
        _ => 100,
    }
}

/// Time-controlled saturation configuration.
///
/// Controls how long e-graph saturation runs before giving up.
/// Named after chess time controls: blitz (fast), rapid (medium), classical (thorough).
struct SaturationConfig {
    /// Maximum number of rewrite iterations.
    max_iterations: usize,
    /// Hard wall-clock timeout.
    hard_timeout: Duration,
    /// Maximum e-classes before stopping (memory safety valve).
    max_classes: usize,
}

impl SaturationConfig {
    /// Blitz: fast, for trivial expressions (0-10 nodes).
    fn blitz() -> Self {
        Self {
            max_iterations: 20,
            hard_timeout: Duration::from_millis(10),
            max_classes: 500,
        }
    }

    /// Rapid: balanced, for normal expressions (11-50 nodes).
    fn rapid() -> Self {
        Self {
            max_iterations: 50,
            hard_timeout: Duration::from_millis(50),
            max_classes: 2000,
        }
    }

    /// Classical: thorough, for complex expressions (51+ nodes).
    fn classical() -> Self {
        Self {
            max_iterations: 100,
            hard_timeout: Duration::from_millis(200),
            max_classes: 5000,
        }
    }
}

/// E-graph saturation with chess-style time control.
///
/// Uses iterative saturation with time and size limits:
/// - Time budget (hard timeout): Stop immediately
/// - Size limit (max classes): Prevent memory explosion
/// - Iteration limit (max iterations): Budget control
///
/// ## Chess Time Management
///
/// Like chess engines, we use multiple limits to ensure the compiler
/// never hangs, even on expressions that cause exponential e-graph growth.
///
/// Returns true if saturated (optimal), false if limit hit (best-effort).
fn saturate_with_time_control(
    egraph: &mut EGraph,
    config: &SaturationConfig,
) -> bool {
    let start = Instant::now();

    // Iterative saturation with time and size checks
    for _iteration in 0..config.max_iterations {
        let elapsed = start.elapsed();

        // HARD LIMIT: Stop immediately
        if elapsed >= config.hard_timeout {
            return false;
        }

        // SIZE LIMIT: Stop if e-graph exploded
        if egraph.num_classes() > config.max_classes {
            return false;
        }

        // Apply one round of rewrites (all matching rules)
        let changes = egraph.apply_rules_once();

        // Saturated if no changes
        if changes == 0 {
            return true;
        }
    }

    false // Budget exhausted
}

/// Select the saturation config based on expression node count.
///
/// | Nodes | Config | Rationale |
/// |-------|--------|-----------|
/// | 0-10 | blitz | Trivial expressions need minimal optimization |
/// | 11-50 | rapid | Normal complexity, balanced approach |
/// | 51+ | classical | Complex expressions need thorough search |
fn config_for_node_count(node_count: usize) -> SaturationConfig {
    match node_count {
        0..=10 => SaturationConfig::blitz(),
        11..=50 => SaturationConfig::rapid(),
        _ => SaturationConfig::classical(),
    }
}

/// Count AST nodes (rough measure of expression complexity).
fn count_ast_nodes(expr: &Expr) -> usize {
    match expr {
        Expr::Literal(_) | Expr::Ident(_) => 1,
        Expr::Binary(b) => 1 + count_ast_nodes(&b.lhs) + count_ast_nodes(&b.rhs),
        Expr::Unary(u) => 1 + count_ast_nodes(&u.operand),
        Expr::MethodCall(c) => {
            1 + count_ast_nodes(&c.receiver)
                + c.args.iter().map(count_ast_nodes).sum::<usize>()
        }
        Expr::Call(c) => 1 + c.args.iter().map(count_ast_nodes).sum::<usize>(),
        Expr::Paren(p) => count_ast_nodes(p),
        Expr::Block(b) => {
            let stmt_nodes: usize = b.stmts.iter().map(|s| {
                match s {
                    Stmt::Let(l) => 1 + count_ast_nodes(&l.init),
                    Stmt::Expr(e) => count_ast_nodes(e),
                }
            }).sum();
            let expr_nodes = b.expr.as_ref().map(|e| count_ast_nodes(e)).unwrap_or(0);
            stmt_nodes + expr_nodes
        }
        Expr::Tuple(t) => 1 + t.elems.iter().map(count_ast_nodes).sum::<usize>(),
        Expr::Verbatim(_) => 1,
    }
}

/// Optimize an analyzed kernel using tree rewriting and e-graphs.
pub fn optimize(mut analyzed: AnalyzedKernel) -> AnalyzedKernel {
    // 1. Structural optimization (catches things inside opaque nodes)
    analyzed.def.body = optimize_expr(analyzed.def.body);

    // 2. E-Graph optimization (global rewriting & fusion)
    // Uses neural cost model for structural extraction
    optimize_with_nnue(analyzed)
}

/// Optimize an analyzed kernel using neural cost extraction.
pub fn optimize_with_nnue(mut analyzed: AnalyzedKernel) -> AnalyzedKernel {
    let nnue = get_nnue_model();
    analyzed.def.body = optimize_expr_with_nnue(analyzed.def.body, nnue);
    analyzed
}

/// Optimize a single expression using e-graph saturation and neural extraction.
fn optimize_expr_with_nnue(expr: Expr, nnue: &ExprNnue) -> Expr {
    // Treat the entire expression as a unit for global optimization
    optimize_via_nnue(&expr, nnue)
}

/// Optimize an expression via e-graph with neural extraction.
fn optimize_via_nnue(expr: &Expr, nnue: &ExprNnue) -> Expr {
    // Try to convert AST → IR
    let ir = match ast_to_ir(expr, &std::collections::HashMap::new()) {
        Ok(ir) => ir,
        Err(_) => {
            // Fallback to legacy AST-based if IR fails
            let mut ctx = EGraphContext::new();
            let root = ctx.expr_to_egraph(expr);
            let node_count = count_ast_nodes(expr);
            saturate_with_time_control(&mut ctx.egraph, &config_for_node_count(node_count));
            
            // PixelflowZero: Pick the best structural form via NNUE
            let (tree, _) = extract_neural(&ctx.egraph, root, nnue);
            return ctx.tree_to_expr(&tree);
        }
    };

    // Flatten IR → E-graph
    let mut ctx = IRToEGraphContext::new();
    let root = ctx.ir_to_egraph(&ir);

    let node_count = count_ast_nodes(expr);
    saturate_with_time_control(&mut ctx.egraph, &config_for_node_count(node_count));

    // Extract optimized E-graph → ExprTree using NNUE
    let (tree, _) = extract_neural(&ctx.egraph, root, nnue);

    // Convert ExprTree → AST
    let legacy_ctx = EGraphContext::new();
    legacy_ctx.tree_to_expr(&tree)
}

/// Check if a block must preserve its structure during optimization.
///
/// A block must preserve structure if:
/// 1. It has let bindings that are referenced in the final expression, OR
/// 2. It has opaque expressions (like method calls on captured manifolds)
///    that reference let-bound locals
///
/// The e-graph optimizer inlines variable references, so if we don't preserve
/// the block structure, the let bindings would be lost in the extracted result.
fn block_has_opaque_with_locals(block: &BlockExpr) -> bool {
    // Collect names of let-bound locals
    let local_names: std::collections::HashSet<String> = block
        .stmts
        .iter()
        .filter_map(|s| {
            if let Stmt::Let(let_stmt) = s {
                Some(let_stmt.name.to_string())
            } else {
                None
            }
        })
        .collect();

    if local_names.is_empty() {
        return false;
    }

    // If the final expression references ANY let-bound local in an opaque context,
    // we must preserve the block structure.
    // Note: Standard usage (e.g. `x + y`) is fine - the e-graph will inline it.
    if let Some(ref final_expr) = block.expr {
        if expr_has_opaque_refs(final_expr, &local_names) {
            return true;
        }
    }

    // Also check if any statement's init references locals in opaque contexts
    for stmt in &block.stmts {
        if let Stmt::Let(let_stmt) = stmt {
            if expr_has_opaque_refs(&let_stmt.init, &local_names) {
                return true;
            }
        }
    }

    false
}

/// Check if an expression has opaque sub-expressions that reference the given names.
fn expr_has_opaque_refs(expr: &Expr, local_names: &std::collections::HashSet<String>) -> bool {
    match expr {
        // Method calls on non-intrinsic receivers are opaque if they use locals
        Expr::MethodCall(call) => {
            // Check if the receiver is opaque (Verbatim) and args reference locals
            // This catches patterns like: ColorCube::default().at(red, green, blue, 1.0)
            // where ColorCube::default() is Verbatim and red/green/blue are locals
            if matches!(call.receiver.as_ref(), Expr::Verbatim(_)) {
                if call.args.iter().any(|arg| expr_references_any(arg, local_names)) {
                    return true;
                }
            }
            // Check if this is a method on a captured variable (not X, Y, Z, W)
            if let Expr::Ident(ident) = call.receiver.as_ref() {
                let name = ident.name.to_string();
                // If the receiver is a local or an external captured variable,
                // and args contain locals, this is problematic
                if !is_coordinate_intrinsic(&name) {
                    // Check if any arg references a local
                    if call.args.iter().any(|arg| expr_references_any(arg, local_names)) {
                        return true;
                    }
                }
            }
            // Recurse into receiver and args
            expr_has_opaque_refs(&call.receiver, local_names)
                || call.args.iter().any(|a| expr_has_opaque_refs(a, local_names))
        }

        // Function calls are treated as opaque because expr_to_egraph doesn't
        // map them to ENodes (it falls through to create_opaque_var).
        // Therefore, if any arg references a local, we must preserve structure.
        Expr::Call(call) => {
            // Calls are opaque. If args reference locals, the call itself is an opaque ref.
            if call.args.iter().any(|a| expr_references_any(a, local_names)) {
                return true;
            }
            // Recurse to check for nested opaque refs
            call.args.iter().any(|a| expr_has_opaque_refs(a, local_names))
        }

        // Recurse into other expression types
        Expr::Binary(b) => {
            expr_has_opaque_refs(&b.lhs, local_names) || expr_has_opaque_refs(&b.rhs, local_names)
        }
        Expr::Unary(u) => expr_has_opaque_refs(&u.operand, local_names),
        Expr::Paren(p) => expr_has_opaque_refs(p, local_names),
        Expr::Tuple(t) => t.elems.iter().any(|e| expr_has_opaque_refs(e, local_names)),
        Expr::Block(b) => {
            b.stmts.iter().any(|s| {
                if let Stmt::Let(l) = s {
                    expr_has_opaque_refs(&l.init, local_names)
                } else {
                    false
                }
            }) || b.expr.as_ref().map_or(false, |e| expr_has_opaque_refs(e, local_names))
        }

        Expr::Ident(_) | Expr::Literal(_) => false,

        // Verbatim expressions wrap syn::Expr - check if they reference locals
        Expr::Verbatim(syn_expr) => syn_expr_references_any(syn_expr, local_names),
    }
}

/// Check if an expression references any of the given names.
fn expr_references_any(expr: &Expr, names: &std::collections::HashSet<String>) -> bool {
    match expr {
        Expr::Ident(i) => names.contains(&i.name.to_string()),
        Expr::Binary(b) => {
            expr_references_any(&b.lhs, names) || expr_references_any(&b.rhs, names)
        }
        Expr::Unary(u) => expr_references_any(&u.operand, names),
        Expr::MethodCall(c) => {
            expr_references_any(&c.receiver, names)
                || c.args.iter().any(|a| expr_references_any(a, names))
        }
        Expr::Call(c) => c.args.iter().any(|a| expr_references_any(a, names)),
        Expr::Paren(p) => expr_references_any(p, names),
        Expr::Tuple(t) => t.elems.iter().any(|e| expr_references_any(e, names)),
        Expr::Block(b) => {
            b.stmts.iter().any(|s| {
                if let Stmt::Let(l) = s {
                    expr_references_any(&l.init, names)
                } else {
                    false
                }
            }) || b.expr.as_ref().map_or(false, |e| expr_references_any(e, names))
        }
        Expr::Literal(_) => false,

        // Verbatim expressions wrap syn::Expr - check if they reference any names
        Expr::Verbatim(syn_expr) => syn_expr_references_any(syn_expr, names),
    }
}

/// Check if a syn::Expr references any of the given names.
///
/// This walks the syn::Expr tree looking for identifiers that match any of the names.
/// Used for checking Verbatim expressions that wrap raw syn::Expr values.
fn syn_expr_references_any(expr: &syn::Expr, names: &std::collections::HashSet<String>) -> bool {
    use syn::Expr as SynExpr;

    match expr {
        SynExpr::Path(path) => {
            // Simple identifier like `c_x`
            if let Some(ident) = path.path.get_ident() {
                names.contains(&ident.to_string())
            } else {
                // Qualified path like `Discrete::pack` - check segments
                path.path.segments.iter().any(|seg| names.contains(&seg.ident.to_string()))
            }
        }

        SynExpr::MethodCall(call) => {
            // Recursively check receiver and arguments
            syn_expr_references_any(&call.receiver, names)
                || call.args.iter().any(|arg| syn_expr_references_any(arg, names))
        }

        SynExpr::Call(call) => {
            // Check function and arguments
            syn_expr_references_any(&call.func, names)
                || call.args.iter().any(|arg| syn_expr_references_any(arg, names))
        }

        SynExpr::Binary(bin) => {
            syn_expr_references_any(&bin.left, names)
                || syn_expr_references_any(&bin.right, names)
        }

        SynExpr::Unary(un) => syn_expr_references_any(&un.expr, names),

        SynExpr::Paren(paren) => syn_expr_references_any(&paren.expr, names),

        SynExpr::Field(field) => syn_expr_references_any(&field.base, names),

        SynExpr::Index(index) => {
            syn_expr_references_any(&index.expr, names)
                || syn_expr_references_any(&index.index, names)
        }

        SynExpr::Cast(cast) => syn_expr_references_any(&cast.expr, names),

        SynExpr::Reference(reference) => syn_expr_references_any(&reference.expr, names),

        SynExpr::Tuple(tuple) => tuple.elems.iter().any(|e| syn_expr_references_any(e, names)),

        SynExpr::Array(array) => array.elems.iter().any(|e| syn_expr_references_any(e, names)),

        SynExpr::Block(block) => {
            block.block.stmts.iter().any(|stmt| {
                match stmt {
                    syn::Stmt::Local(local) => {
                        local.init.as_ref().map_or(false, |init| {
                            syn_expr_references_any(&init.expr, names)
                        })
                    }
                    syn::Stmt::Expr(expr, _) => syn_expr_references_any(expr, names),
                    _ => false,
                }
            })
        }

        SynExpr::If(if_expr) => {
            syn_expr_references_any(&if_expr.cond, names)
                || if_expr.then_branch.stmts.iter().any(|stmt| {
                    if let syn::Stmt::Expr(expr, _) = stmt {
                        syn_expr_references_any(expr, names)
                    } else {
                        false
                    }
                })
                || if_expr.else_branch.as_ref().map_or(false, |(_, else_expr)| {
                    syn_expr_references_any(else_expr, names)
                })
        }

        // Literals don't reference variables
        SynExpr::Lit(_) => false,

        // For other expression types, conservatively return true to preserve structure
        // (better to preserve than to accidentally lose bindings)
        _ => true,
    }
}

/// Check if a name is a coordinate intrinsic (X, Y, Z, W).
fn is_coordinate_intrinsic(name: &str) -> bool {
    matches!(name, "X" | "Y" | "Z" | "W")
}

/// Optimize a block while preserving its structure.
///
/// Each let binding and the final expression are optimized independently.
fn optimize_block_preserving_structure(mut block: BlockExpr, nnue: &ExprNnue) -> Expr {
    for stmt in &mut block.stmts {
        if let Stmt::Let(let_stmt) = stmt {
            let init = std::mem::replace(
                &mut let_stmt.init,
                make_literal(0.0, Span::call_site()),
            );
            let_stmt.init = optimize_expr_with_nnue(init, nnue);
        }
    }
    if let Some(final_expr) = block.expr.take() {
        block.expr = Some(Box::new(optimize_expr_with_nnue(*final_expr, nnue)));
    }
    Expr::Block(block)
}

/// Optimize an expression via the e-graph (AST-based, legacy).
///
/// Uses chess-style time control to prevent hanging on complex expressions.
fn optimize_via_egraph(expr: &Expr, costs: &CostModel) -> Expr {
    let mut ctx = EGraphContext::new();
    let root = ctx.expr_to_egraph(expr);

    // Select time budget based on expression complexity
    let node_count = count_ast_nodes(expr);
    let config = config_for_node_count(node_count);

    // Time-controlled saturation (replaces fixed budget)
    saturate_with_time_control(&mut ctx.egraph, &config);

    let tree = ctx.egraph.extract_tree_with_costs(root, costs);
    ctx.tree_to_expr(&tree)
}

/// Optimize an expression via e-graph with DAG-aware extraction.
///
/// Unlike `optimize_via_egraph`, this tracks shared subexpressions and emits
/// let-bindings for e-classes used more than once. This enables Common
/// Subexpression Elimination (CSE) in the generated code.
///
/// # Example
///
/// For `sin(X) * sin(X) + sin(X)`:
/// - Tree extraction: `X.sin() * X.sin() + X.sin()` (3 sin calls)
/// - DAG extraction: `{ let __0 = X.sin(); __0 * __0 + __0 }` (1 sin call)
#[allow(dead_code)] // Will be used in future integration
fn optimize_via_egraph_dag(expr: &Expr, costs: &CostModel) -> Expr {
    let mut ctx = EGraphContext::new();
    let root = ctx.expr_to_egraph(expr);

    // Select time budget based on expression complexity
    let node_count = count_ast_nodes(expr);
    let config = config_for_node_count(node_count);

    // Time-controlled saturation (replaces fixed budget)
    saturate_with_time_control(&mut ctx.egraph, &config);

    // Use DAG extraction to capture sharing
    let dag = ctx.egraph.extract_dag_with_costs(root, costs);

    // Only use DAG-to-expr if there are actually shared subexpressions
    // This avoids unnecessary block wrapping for simple expressions
    if dag.shared.iter().any(|(id, _)| ctx.egraph.find(*id) != dag.root) {
        ctx.dag_to_expr(&dag)
    } else {
        // No sharing - use simpler tree extraction
        let tree = ctx.egraph.extract_tree_with_costs(root, costs);
        ctx.tree_to_expr(&tree)
    }
}

/// Optimize an expression via the IR pipeline (NEW).
///
/// This uses the unified IR representation:
/// AST → IR → E-graph → ExprTree → AST
///
/// Uses chess-style time control to prevent hanging on complex expressions.
/// Falls back to AST-based optimization if IR conversion fails.
fn optimize_via_ir(expr: &Expr, costs: &CostModel) -> Expr {
    // Try to convert AST → IR
    let ir = match ast_to_ir(expr, &std::collections::HashMap::new()) {
        Ok(ir) => ir,
        Err(_) => {
            // IR conversion failed (unsupported constructs like blocks, captured variables)
            // Fall back to legacy AST-based optimization
            return optimize_via_egraph(expr, costs);
        }
    };

    // Flatten IR → E-graph
    let mut ctx = IRToEGraphContext::new();
    let root = ctx.ir_to_egraph(&ir);

    // Select time budget based on expression complexity
    let node_count = count_ast_nodes(expr);
    let config = config_for_node_count(node_count);

    // Time-controlled saturation (replaces fixed budget)
    saturate_with_time_control(&mut ctx.egraph, &config);

    // Extract optimized E-graph → ExprTree
    let tree = ctx.egraph.extract_tree_with_costs(root, costs);

    // Convert ExprTree → AST using existing infrastructure
    // We reuse the AST generation since it already knows how to emit method calls, etc.
    let legacy_ctx = EGraphContext::new();
    legacy_ctx.tree_to_expr(&tree)
}

// ============================================================================
// E-Graph Integration (Legacy AST-based)
// ============================================================================

/// Context for converting between AST and e-graph representations.
struct EGraphContext {
    /// The e-graph being built.
    egraph: EGraph,
    /// Map from variable name to e-class ID.
    var_to_eclass: HashMap<String, EClassId>,
    /// Map from variable index to name (for extraction).
    idx_to_name: Vec<String>,
    /// Map from opaque variable names to their original expressions.
    /// Used to restore expressions that can't be represented in the e-graph.
    opaque_exprs: HashMap<String, Expr>,
}

impl EGraphContext {
    fn new() -> Self {
        Self {
            egraph: EGraph::with_rules(standard_rules()),
            var_to_eclass: HashMap::new(),
            idx_to_name: Vec::new(),
            opaque_exprs: HashMap::new(),
        }
    }

    /// Get or create an e-class for a variable.
    fn get_or_create_var(&mut self, name: &str) -> EClassId {
        if let Some(&id) = self.var_to_eclass.get(name) {
            return id;
        }

        // Assign next index
        let idx = self.idx_to_name.len() as u8;
        self.idx_to_name.push(name.to_string());

        let id = self.egraph.add(ENode::Var(idx));
        self.var_to_eclass.insert(name.to_string(), id);
        id
    }

    /// Create an opaque variable for an expression we can't optimize.
    /// The original expression is stored and will be restored during extraction.
    fn create_opaque_var(&mut self, prefix: &str, expr: &Expr) -> EClassId {
        let name = unique_opaque_name(prefix);
        self.opaque_exprs.insert(name.clone(), expr.clone());
        self.get_or_create_var(&name)
    }

    /// Check if a method is known and can be converted to ENode.
    fn is_known_method(method: &str, arg_count: usize) -> bool {
        match method {
            // Unary methods (0 args)
            "sqrt" | "rsqrt" | "recip" | "abs" | "neg"
            | "floor" | "ceil" | "round" | "fract"
            | "sin" | "cos" | "tan" | "asin" | "acos" | "atan"
            | "exp" | "exp2" | "ln" | "log2" | "log10" => arg_count == 0,

            // Binary methods (1 arg)
            "min" | "max" | "atan2" | "pow" | "hypot"
            | "lt" | "le" | "gt" | "ge" | "eq" | "ne" => arg_count == 1,

            // Ternary methods (2 args)
            "mul_add" | "select" | "clamp" => arg_count == 2,

            _ => false,
        }
    }

    /// Convert an AST expression to an e-graph, returning the root e-class.
    fn expr_to_egraph(&mut self, expr: &Expr) -> EClassId {
        match expr {
            Expr::Ident(ident) => self.get_or_create_var(&ident.name.to_string()),

            Expr::Literal(lit) => {
                if let Some(val) = extract_f64_from_lit(&lit.lit) {
                    self.egraph.add(ENode::constant(val as f32))
                } else {
                    // Non-numeric literal - preserve original
                    self.create_opaque_var("lit", expr)
                }
            }

            Expr::Binary(binary) => {
                // Check if this is a supported binary op BEFORE converting children
                // Unsupported ops are preserved as opaque expressions
                match binary.op {
                    BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div
                    | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge
                    | BinaryOp::Eq | BinaryOp::Ne => {
                        // Supported - convert children
                        let lhs = self.expr_to_egraph(&binary.lhs);
                        let rhs = self.expr_to_egraph(&binary.rhs);

                        let op: &'static dyn ops::Op = match binary.op {
                            BinaryOp::Add => &ops::Add,
                            BinaryOp::Sub => &ops::Sub,
                            BinaryOp::Mul => &ops::Mul,
                            BinaryOp::Div => &ops::Div,
                            BinaryOp::Lt => &ops::Lt,
                            BinaryOp::Le => &ops::Le,
                            BinaryOp::Gt => &ops::Gt,
                            BinaryOp::Ge => &ops::Ge,
                            BinaryOp::Eq => &ops::Eq,
                            BinaryOp::Ne => &ops::Ne,
                            _ => unreachable!(),
                        };
                        self.egraph.add(ENode::Op { op, children: vec![lhs, rhs] })
                    }
                    // For other ops (Rem, BitXor, Shl, Shr)
                    // preserve as opaque expression with original structure
                    _ => self.create_opaque_var("binop", expr),
                }
            }

            Expr::Unary(unary) => {
                match unary.op {
                    UnaryOp::Neg => {
                        let operand = self.expr_to_egraph(&unary.operand);
                        self.egraph.add(ENode::Op { op: &ops::Neg, children: vec![operand] })
                    }
                    UnaryOp::Not => {
                        // Map Not(x) to 1.0 - x (assuming boolean 0.0/1.0 logic)
                        let operand = self.expr_to_egraph(&unary.operand);
                        let one = self.egraph.add(ENode::constant(1.0));
                        self.egraph.add(ENode::Op { op: &ops::Sub, children: vec![one, operand] })
                    }
                }
            }

            Expr::MethodCall(call) => {
                let method = call.method.to_string();

                // Check if this is a known method BEFORE converting children
                // Unknown methods preserve the original expression structure
                if !Self::is_known_method(&method, call.args.len()) {
                    return self.create_opaque_var("method", expr);
                }

                let receiver = self.expr_to_egraph(&call.receiver);

                match method.as_str() {
                    // === Unary methods ===
                    "sqrt" => self.egraph.add(ENode::Op { op: &ops::Sqrt, children: vec![receiver] }),
                    "rsqrt" => self.egraph.add(ENode::Op { op: &ops::Rsqrt, children: vec![receiver] }),
                    "recip" => self.egraph.add(ENode::Op { op: &ops::Recip, children: vec![receiver] }),
                    "abs" => self.egraph.add(ENode::Op { op: &ops::Abs, children: vec![receiver] }),
                    "neg" => self.egraph.add(ENode::Op { op: &ops::Neg, children: vec![receiver] }),
                    "floor" => self.egraph.add(ENode::Op { op: &ops::Floor, children: vec![receiver] }),
                    "ceil" => self.egraph.add(ENode::Op { op: &ops::Ceil, children: vec![receiver] }),
                    "round" => self.egraph.add(ENode::Op { op: &ops::Round, children: vec![receiver] }),
                    "fract" => self.egraph.add(ENode::Op { op: &ops::Fract, children: vec![receiver] }),
                    "sin" => self.egraph.add(ENode::Op { op: &ops::Sin, children: vec![receiver] }),
                    "cos" => self.egraph.add(ENode::Op { op: &ops::Cos, children: vec![receiver] }),
                    "tan" => self.egraph.add(ENode::Op { op: &ops::Tan, children: vec![receiver] }),
                    "asin" => self.egraph.add(ENode::Op { op: &ops::Asin, children: vec![receiver] }),
                    "acos" => self.egraph.add(ENode::Op { op: &ops::Acos, children: vec![receiver] }),
                    "atan" => self.egraph.add(ENode::Op { op: &ops::Atan, children: vec![receiver] }),
                    "exp" => self.egraph.add(ENode::Op { op: &ops::Exp, children: vec![receiver] }),
                    "exp2" => self.egraph.add(ENode::Op { op: &ops::Exp2, children: vec![receiver] }),
                    "ln" => self.egraph.add(ENode::Op { op: &ops::Ln, children: vec![receiver] }),
                    "log2" => self.egraph.add(ENode::Op { op: &ops::Log2, children: vec![receiver] }),
                    "log10" => self.egraph.add(ENode::Op { op: &ops::Log10, children: vec![receiver] }),

                    // === Binary methods ===
                    "min" => {
                        let arg = self.expr_to_egraph(&call.args[0]);
                        self.egraph.add(ENode::Op { op: &ops::Min, children: vec![receiver, arg] })
                    }
                    "max" => {
                        let arg = self.expr_to_egraph(&call.args[0]);
                        self.egraph.add(ENode::Op { op: &ops::Max, children: vec![receiver, arg] })
                    }
                    "atan2" => {
                        let arg = self.expr_to_egraph(&call.args[0]);
                        self.egraph.add(ENode::Op { op: &ops::Atan2, children: vec![receiver, arg] })
                    }
                    "pow" => {
                        let arg = self.expr_to_egraph(&call.args[0]);
                        self.egraph.add(ENode::Op { op: &ops::Pow, children: vec![receiver, arg] })
                    }
                    "hypot" => {
                        let arg = self.expr_to_egraph(&call.args[0]);
                        self.egraph.add(ENode::Op { op: &ops::Hypot, children: vec![receiver, arg] })
                    }

                    // === Comparison methods ===
                    "lt" => {
                        let arg = self.expr_to_egraph(&call.args[0]);
                        self.egraph.add(ENode::Op { op: &ops::Lt, children: vec![receiver, arg] })
                    }
                    "le" => {
                        let arg = self.expr_to_egraph(&call.args[0]);
                        self.egraph.add(ENode::Op { op: &ops::Le, children: vec![receiver, arg] })
                    }
                    "gt" => {
                        let arg = self.expr_to_egraph(&call.args[0]);
                        self.egraph.add(ENode::Op { op: &ops::Gt, children: vec![receiver, arg] })
                    }
                    "ge" => {
                        let arg = self.expr_to_egraph(&call.args[0]);
                        self.egraph.add(ENode::Op { op: &ops::Ge, children: vec![receiver, arg] })
                    }
                    "eq" => {
                        let arg = self.expr_to_egraph(&call.args[0]);
                        self.egraph.add(ENode::Op { op: &ops::Eq, children: vec![receiver, arg] })
                    }
                    "ne" => {
                        let arg = self.expr_to_egraph(&call.args[0]);
                        self.egraph.add(ENode::Op { op: &ops::Ne, children: vec![receiver, arg] })
                    }

                    // === Ternary methods ===
                    "mul_add" => {
                        let b = self.expr_to_egraph(&call.args[0]);
                        let c = self.expr_to_egraph(&call.args[1]);
                        self.egraph.add(ENode::Op { op: &ops::MulAdd, children: vec![receiver, b, c] })
                    }
                    "select" => {
                        let if_true = self.expr_to_egraph(&call.args[0]);
                        let if_false = self.expr_to_egraph(&call.args[1]);
                        self.egraph.add(ENode::Op { op: &ops::Select, children: vec![receiver, if_true, if_false] })
                    }
                    "clamp" => {
                        let min_val = self.expr_to_egraph(&call.args[0]);
                        let max_val = self.expr_to_egraph(&call.args[1]);
                        self.egraph.add(ENode::Op { op: &ops::Clamp, children: vec![receiver, min_val, max_val] })
                    }

                    // Should not reach here due to is_known_method check
                    _ => unreachable!("Unknown method {} should have been handled as opaque", method),
                }
            }

            Expr::Paren(inner) => self.expr_to_egraph(inner),

            Expr::Block(block) => {
                // For blocks with let bindings, add bindings to var map
                for stmt in &block.stmts {
                    if let Stmt::Let(let_stmt) = stmt {
                        let init_id = self.expr_to_egraph(&let_stmt.init);
                        self.var_to_eclass
                            .insert(let_stmt.name.to_string(), init_id);
                    }
                }

                // Optimize the final expression
                if let Some(expr) = &block.expr {
                    self.expr_to_egraph(expr)
                } else {
                    // Empty block - return zero
                    self.egraph.add(ENode::constant(0.0))
                }
            }

            // For Call and Verbatim, treat as opaque and store original expression
            // so it can be restored during extraction
            Expr::Call(call) => {
                self.create_opaque_var(&format!("call_{}_", call.func), expr)
            }

            Expr::Verbatim(_) => {
                self.create_opaque_var("verbatim_", expr)
            }

            Expr::Tuple(tuple) => {
                let elems: Vec<_> = tuple.elems.iter().map(|e| self.expr_to_egraph(e)).collect();
                self.egraph.add(ENode::Op { op: &ops::Tuple, children: elems })
            }
        }
    }

    /// Convert an ExtractedDAG to an AST expression with let-bindings for shared subexprs.
    ///
    /// For shared e-classes (used more than once), this generates let-bindings:
    /// ```text
    /// {
    ///     let __0 = shared_expr_1;
    ///     let __1 = shared_expr_2;
    ///     root_expr_using_bindings
    /// }
    /// ```
    fn dag_to_expr(&self, dag: &ExtractedDAG) -> Expr {
        let span = Span::call_site();

        // Build a map from shared e-class indices to their let-binding names
        let mut binding_names: HashMap<usize, String> = HashMap::new();
        let mut stmts: Vec<Stmt> = Vec::new();

        // Emit let-bindings for shared e-classes in topological order
        let mut binding_idx = 0usize;
        for &class_id in &dag.schedule {
            let canonical = self.egraph.find(class_id);

            // Only bind shared classes that aren't the root
            // (the root becomes the final expression, not a binding)
            if dag.is_shared(canonical) && canonical != dag.root {
                let var_name = format!("__{}", binding_idx);

                // Build the AST for this e-class
                let expr = self.eclass_to_expr(canonical, dag, &binding_names);

                // Create let statement
                stmts.push(Stmt::Let(LetStmt {
                    name: Ident::new(&var_name, span),
                    ty: None,
                    init: expr,
                    span,
                }));

                binding_names.insert(canonical.index(), var_name);
                binding_idx += 1;
            }
        }

        // Build the root expression
        let root_expr = self.eclass_to_expr(dag.root, dag, &binding_names);

        if stmts.is_empty() {
            // No shared subexpressions, return simple expression
            root_expr
        } else {
            // Wrap in a block with let-bindings
            Expr::Block(BlockExpr {
                stmts,
                expr: Some(Box::new(root_expr)),
                span,
            })
        }
    }

    /// Build an AST expression for a single e-class, using bindings for shared subexprs.
    fn eclass_to_expr(
        &self,
        class: EClassId,
        dag: &ExtractedDAG,
        binding_names: &HashMap<usize, String>,
    ) -> Expr {
        let span = Span::call_site();
        let canonical = self.egraph.find(class);

        // If this e-class is bound to a variable, just reference it
        if let Some(name) = binding_names.get(&canonical.index()) {
            return Expr::Ident(IdentExpr {
                name: Ident::new(name, span),
                span,
            });
        }

        // Get the best node for this e-class
        let node_idx = dag.best_node_idx(canonical)
            .unwrap_or_else(|| panic!("No best node for e-class {} in DAG", canonical.index()));
        let node = &self.egraph.nodes(canonical)[node_idx];

        match node {
            ENode::Var(idx) => {
                // Try to get the variable name from our mapping
                let name = self.idx_to_name
                    .get(*idx as usize)
                    .cloned()
                    .unwrap_or_else(|| match idx {
                        0 => "X".to_string(),
                        1 => "Y".to_string(),
                        2 => "Z".to_string(),
                        3 => "W".to_string(),
                        _ => format!("__var{}", idx),
                    });

                // Check if this is an opaque variable - restore original expression
                if let Some(original) = self.opaque_exprs.get(&name) {
                    return original.clone();
                }

                Expr::Ident(IdentExpr {
                    name: Ident::new(&name, span),
                    span,
                })
            }

            ENode::Const(bits) => make_literal(f32::from_bits(*bits) as f64, span),

            ENode::Op { op, children } => {
                let name = op.name();
                let child_exprs: Vec<Expr> = children.iter()
                    .map(|&c| self.eclass_to_expr(c, dag, binding_names))
                    .collect();

                self.emit_op_as_expr(name, &child_exprs, span)
            }
        }
    }

    /// Emit an operation as an AST expression.
    fn emit_op_as_expr(&self, op_name: &str, children: &[Expr], span: Span) -> Expr {
        match (op_name, children) {
            // Binary arithmetic
            ("add", [a, b]) => Expr::Binary(BinaryExpr {
                op: BinaryOp::Add,
                lhs: Box::new(a.clone()),
                rhs: Box::new(b.clone()),
                span,
            }),
            ("sub", [a, b]) => Expr::Binary(BinaryExpr {
                op: BinaryOp::Sub,
                lhs: Box::new(a.clone()),
                rhs: Box::new(b.clone()),
                span,
            }),
            ("mul", [a, b]) => Expr::Binary(BinaryExpr {
                op: BinaryOp::Mul,
                lhs: Box::new(a.clone()),
                rhs: Box::new(b.clone()),
                span,
            }),
            ("div", [a, b]) => Expr::Binary(BinaryExpr {
                op: BinaryOp::Div,
                lhs: Box::new(a.clone()),
                rhs: Box::new(b.clone()),
                span,
            }),

            // Unary
            ("neg", [a]) => Expr::Unary(UnaryExpr {
                op: UnaryOp::Neg,
                operand: Box::new(a.clone()),
                span,
            }),
            ("recip", [a]) => Expr::Binary(BinaryExpr {
                op: BinaryOp::Div,
                lhs: Box::new(make_literal(1.0, span)),
                rhs: Box::new(a.clone()),
                span,
            }),

            // Unary methods
            ("sqrt", [a]) => self.unary_method_expr(a, "sqrt", span),
            ("rsqrt", [a]) => self.unary_method_expr(a, "rsqrt", span),
            ("abs", [a]) => self.unary_method_expr(a, "abs", span),
            ("floor", [a]) => self.unary_method_expr(a, "floor", span),
            ("ceil", [a]) => self.unary_method_expr(a, "ceil", span),
            ("round", [a]) => self.unary_method_expr(a, "round", span),
            ("fract", [a]) => self.unary_method_expr(a, "fract", span),
            ("sin", [a]) => self.unary_method_expr(a, "sin", span),
            ("cos", [a]) => self.unary_method_expr(a, "cos", span),
            ("tan", [a]) => self.unary_method_expr(a, "tan", span),
            ("asin", [a]) => self.unary_method_expr(a, "asin", span),
            ("acos", [a]) => self.unary_method_expr(a, "acos", span),
            ("atan", [a]) => self.unary_method_expr(a, "atan", span),
            ("exp", [a]) => self.unary_method_expr(a, "exp", span),
            ("exp2", [a]) => self.unary_method_expr(a, "exp2", span),
            ("ln", [a]) => self.unary_method_expr(a, "ln", span),
            ("log2", [a]) => self.unary_method_expr(a, "log2", span),
            ("log10", [a]) => self.unary_method_expr(a, "log10", span),

            // Binary methods
            ("min", [a, b]) => self.binary_method_expr(a, b, "min", span),
            ("max", [a, b]) => self.binary_method_expr(a, b, "max", span),
            ("atan2", [a, b]) => self.binary_method_expr(a, b, "atan2", span),
            ("pow", [a, b]) => self.binary_method_expr(a, b, "pow", span),
            ("hypot", [a, b]) => self.binary_method_expr(a, b, "hypot", span),

            // Comparisons
            ("lt", [a, b]) => self.binary_op_expr(a, b, BinaryOp::Lt, span),
            ("le", [a, b]) => self.binary_op_expr(a, b, BinaryOp::Le, span),
            ("gt", [a, b]) => self.binary_op_expr(a, b, BinaryOp::Gt, span),
            ("ge", [a, b]) => self.binary_op_expr(a, b, BinaryOp::Ge, span),
            ("eq", [a, b]) => self.binary_op_expr(a, b, BinaryOp::Eq, span),
            ("ne", [a, b]) => self.binary_op_expr(a, b, BinaryOp::Ne, span),

            // Ternary
            ("mul_add", [a, b, c]) => Expr::MethodCall(MethodCallExpr {
                receiver: Box::new(a.clone()),
                method: Ident::new("mul_add", span),
                args: vec![b.clone(), c.clone()],
                span,
            }),
            ("select", [a, b, c]) => Expr::MethodCall(MethodCallExpr {
                receiver: Box::new(a.clone()),
                method: Ident::new("select", span),
                args: vec![b.clone(), c.clone()],
                span,
            }),
            ("clamp", [a, b, c]) => Expr::MethodCall(MethodCallExpr {
                receiver: Box::new(a.clone()),
                method: Ident::new("clamp", span),
                args: vec![b.clone(), c.clone()],
                span,
            }),

            // Tuple
            ("tuple", elems) => Expr::Tuple(crate::ast::TupleExpr {
                elems: elems.to_vec(),
                span,
            }),

            // Unknown - try as unary or binary method
            (name, [a]) => self.unary_method_expr(a, name, span),
            (name, [a, b]) => self.binary_method_expr(a, b, name, span),
            (name, _) => panic!("Unknown operation {} with {} children", name, children.len()),
        }
    }

    fn unary_method_expr(&self, a: &Expr, name: &str, span: Span) -> Expr {
        Expr::MethodCall(MethodCallExpr {
            receiver: Box::new(a.clone()),
            method: Ident::new(name, span),
            args: vec![],
            span,
        })
    }

    fn binary_method_expr(&self, a: &Expr, b: &Expr, name: &str, span: Span) -> Expr {
        Expr::MethodCall(MethodCallExpr {
            receiver: Box::new(a.clone()),
            method: Ident::new(name, span),
            args: vec![b.clone()],
            span,
        })
    }

    fn binary_op_expr(&self, a: &Expr, b: &Expr, op: BinaryOp, span: Span) -> Expr {
        Expr::Binary(BinaryExpr {
            op,
            lhs: Box::new(a.clone()),
            rhs: Box::new(b.clone()),
            span,
        })
    }

    /// Convert an extracted expression tree back to an AST expression.
    fn tree_to_expr(&self, tree: &ExprTree) -> Expr {
        let span = Span::call_site();

        match tree {
            ExprTree::Leaf(Leaf::Var(idx)) => {
                // First, try to get the name from our variable mapping
                let name = self
                    .idx_to_name
                    .get(*idx as usize)
                    .cloned()
                    .unwrap_or_else(|| {
                        // Fallback: recognize intrinsic coordinate variable indices
                        // The IR bridge uses 0=X, 1=Y, 2=Z, 3=W convention
                        match idx {
                            0 => "X".to_string(),
                            1 => "Y".to_string(),
                            2 => "Z".to_string(),
                            3 => "W".to_string(),
                            _ => format!("__var{}", idx),
                        }
                    });

                // Check if this is an opaque variable - restore original expression
                if let Some(original) = self.opaque_exprs.get(&name) {
                    return original.clone();
                }

                Expr::Ident(IdentExpr {
                    name: Ident::new(&name, span),
                    span,
                })
            }

            ExprTree::Leaf(Leaf::Const(val)) => make_literal(*val as f64, span),

            ExprTree::Op { op, children } => {
                let name = op.name();
                match (name, children.as_slice()) {
                    // Binary arithmetic
                    ("add", [a, b]) => Expr::Binary(BinaryExpr {
                        op: BinaryOp::Add,
                        lhs: Box::new(self.tree_to_expr(a)),
                        rhs: Box::new(self.tree_to_expr(b)),
                        span,
                    }),
                    ("sub", [a, b]) => Expr::Binary(BinaryExpr {
                        op: BinaryOp::Sub,
                        lhs: Box::new(self.tree_to_expr(a)),
                        rhs: Box::new(self.tree_to_expr(b)),
                        span,
                    }),
                    ("mul", [a, b]) => Expr::Binary(BinaryExpr {
                        op: BinaryOp::Mul,
                        lhs: Box::new(self.tree_to_expr(a)),
                        rhs: Box::new(self.tree_to_expr(b)),
                        span,
                    }),
                    ("div", [a, b]) => Expr::Binary(BinaryExpr {
                        op: BinaryOp::Div,
                        lhs: Box::new(self.tree_to_expr(a)),
                        rhs: Box::new(self.tree_to_expr(b)),
                        span,
                    }),

                    // Unary
                    ("neg", [a]) => Expr::Unary(UnaryExpr {
                        op: UnaryOp::Neg,
                        operand: Box::new(self.tree_to_expr(a)),
                        span,
                    }),
                    ("sqrt", [a]) => self.unary_method(a, "sqrt", span),
                    ("rsqrt", [a]) => self.unary_method(a, "rsqrt", span),
                    ("recip", [a]) => {
                        // recip(x) = 1.0 / x - emit as division
                        Expr::Binary(BinaryExpr {
                            op: BinaryOp::Div,
                            lhs: Box::new(make_literal(1.0, span)),
                            rhs: Box::new(self.tree_to_expr(a)),
                            span,
                        })
                    }
                    ("abs", [a]) => self.unary_method(a, "abs", span),
                    ("floor", [a]) => self.unary_method(a, "floor", span),
                    ("ceil", [a]) => self.unary_method(a, "ceil", span),
                    ("round", [a]) => self.unary_method(a, "round", span),
                    ("fract", [a]) => self.unary_method(a, "fract", span),
                    ("sin", [a]) => self.unary_method(a, "sin", span),
                    ("cos", [a]) => self.unary_method(a, "cos", span),
                    ("tan", [a]) => self.unary_method(a, "tan", span),
                    ("asin", [a]) => self.unary_method(a, "asin", span),
                    ("acos", [a]) => self.unary_method(a, "acos", span),
                    ("atan", [a]) => self.unary_method(a, "atan", span),
                    ("exp", [a]) => self.unary_method(a, "exp", span),
                    ("exp2", [a]) => self.unary_method(a, "exp2", span),
                    ("ln", [a]) => self.unary_method(a, "ln", span),
                    ("log2", [a]) => self.unary_method(a, "log2", span),
                    ("log10", [a]) => self.unary_method(a, "log10", span),

                    // Binary methods
                    ("min", [a, b]) => self.binary_method(a, b, "min", span),
                    ("max", [a, b]) => self.binary_method(a, b, "max", span),
                    ("atan2", [a, b]) => self.binary_method(a, b, "atan2", span),
                    ("pow", [a, b]) => self.binary_method(a, b, "pow", span),
                    ("hypot", [a, b]) => self.binary_method(a, b, "hypot", span),

                    // Comparisons
                    ("lt", [a, b]) => self.binary_op(a, b, BinaryOp::Lt, span),
                    ("le", [a, b]) => self.binary_op(a, b, BinaryOp::Le, span),
                    ("gt", [a, b]) => self.binary_op(a, b, BinaryOp::Gt, span),
                    ("ge", [a, b]) => self.binary_op(a, b, BinaryOp::Ge, span),
                    ("eq", [a, b]) => self.binary_op(a, b, BinaryOp::Eq, span),
                    ("ne", [a, b]) => self.binary_op(a, b, BinaryOp::Ne, span),

                    // Ternary
                    ("mul_add", [a, b, c]) => Expr::MethodCall(MethodCallExpr {
                        receiver: Box::new(self.tree_to_expr(a)),
                        method: Ident::new("mul_add", span),
                        args: vec![self.tree_to_expr(b), self.tree_to_expr(c)],
                        span,
                    }),
                    ("select", [a, b, c]) => Expr::MethodCall(MethodCallExpr {
                        receiver: Box::new(self.tree_to_expr(a)),
                        method: Ident::new("select", span),
                        args: vec![self.tree_to_expr(b), self.tree_to_expr(c)],
                        span,
                    }),
                    ("clamp", [a, b, c]) => Expr::MethodCall(MethodCallExpr {
                        receiver: Box::new(self.tree_to_expr(a)),
                        method: Ident::new("clamp", span),
                        args: vec![self.tree_to_expr(b), self.tree_to_expr(c)],
                        span,
                    }),

                    // Tuple
                    ("tuple", elems) => Expr::Tuple(crate::ast::TupleExpr {
                        elems: elems.iter().map(|e| self.tree_to_expr(e)).collect(),
                        span,
                    }),

                    // Unknown operation - emit as method call if possible
                    (op_name, [a]) => self.unary_method(a, op_name, span),
                    (op_name, [a, b]) => self.binary_method(a, b, op_name, span),
                    _ => panic!("Unknown operation {} with {} children", name, children.len()),
                }
            }
        }
    }

    fn unary_method(&self, a: &ExprTree, name: &str, span: Span) -> Expr {
        Expr::MethodCall(MethodCallExpr {
            receiver: Box::new(self.tree_to_expr(a)),
            method: Ident::new(name, span),
            args: vec![],
            span,
        })
    }

    fn binary_method(&self, a: &ExprTree, b: &ExprTree, name: &str, span: Span) -> Expr {
        Expr::MethodCall(MethodCallExpr {
            receiver: Box::new(self.tree_to_expr(a)),
            method: Ident::new(name, span),
            args: vec![self.tree_to_expr(b)],
            span,
        })
    }

    fn binary_op(&self, a: &ExprTree, b: &ExprTree, op: BinaryOp, span: Span) -> Expr {
        Expr::Binary(BinaryExpr {
            op,
            lhs: Box::new(self.tree_to_expr(a)),
            rhs: Box::new(self.tree_to_expr(b)),
            span,
        })
    }
}

/// Extract f64 from a syn::Lit.
fn extract_f64_from_lit(lit: &Lit) -> Option<f64> {
    match lit {
        Lit::Float(f) => f.base10_parse::<f64>().ok(),
        Lit::Int(i) => i.base10_parse::<f64>().ok(),
        _ => None,
    }
}

fn optimize_expr(expr: Expr) -> Expr {
    match expr {
        Expr::Binary(binary) => optimize_binary(binary),
        Expr::Unary(unary) => optimize_unary(unary),
        Expr::Paren(inner) => Expr::Paren(Box::new(optimize_expr(*inner))),
        Expr::Block(block) => optimize_block(block),
        // Recursively optimize method call arguments and receiver
        Expr::MethodCall(mut call) => {
            call.receiver = Box::new(optimize_expr(*call.receiver));
            call.args = call.args.into_iter().map(optimize_expr).collect();
            Expr::MethodCall(call)
        }
        Expr::Tuple(mut tuple) => {
            tuple.elems = tuple.elems.into_iter().map(optimize_expr).collect();
            Expr::Tuple(tuple)
        }
        Expr::Call(mut call) => {
            call.args = call.args.into_iter().map(optimize_expr).collect();
            Expr::Call(call)
        }
        _ => expr,
    }
}

fn optimize_binary(mut binary: BinaryExpr) -> Expr {
    // 1. Optimize operands first
    binary.lhs = Box::new(optimize_expr(*binary.lhs));
    binary.rhs = Box::new(optimize_expr(*binary.rhs));

    // 2. Try constant folding
    if let (Some(lhs_val), Some(rhs_val)) = (extract_f64(&binary.lhs), extract_f64(&binary.rhs)) {
        if let Some(result) = fold_constants(binary.op, lhs_val, rhs_val) {
            return make_literal(result, binary.span);
        }
    }

    // 3. Try algebraic simplification
    if let Some(simplified) = simplify_algebraic(&binary) {
        return simplified;
    }

    Expr::Binary(binary)
}

fn optimize_unary(mut unary: UnaryExpr) -> Expr {
    unary.operand = Box::new(optimize_expr(*unary.operand));

    if let Some(val) = extract_f64(&unary.operand) {
        if let Some(result) = fold_unary(unary.op, val) {
            return make_literal(result, unary.span);
        }
    }

    Expr::Unary(unary)
}

fn optimize_block(mut block: BlockExpr) -> Expr {
    // Optimize statements
    for stmt in &mut block.stmts {
        if let Stmt::Let(let_stmt) = stmt {
            let_stmt.init = optimize_expr(std::mem::replace(
                &mut let_stmt.init,
                make_literal(0.0, Span::call_site()), // Dummy placeholder
            ));
        } else if let Stmt::Expr(expr) = stmt {
            *expr = optimize_expr(std::mem::replace(
                expr,
                make_literal(0.0, Span::call_site()), // Dummy placeholder
            ));
        }
    }

    // Optimize final expression
    if let Some(expr) = block.expr {
        block.expr = Some(Box::new(optimize_expr(*expr)));
    }

    Expr::Block(block)
}

// --- Helpers ---

fn extract_f64(expr: &Expr) -> Option<f64> {
    if let Expr::Literal(lit_expr) = expr {
        match &lit_expr.lit {
            Lit::Float(f) => f.base10_parse::<f64>().ok(),
            Lit::Int(i) => i.base10_parse::<f64>().ok(),
            _ => None,
        }
    } else {
        None
    }
}

fn make_literal(val: f64, span: Span) -> Expr {
    // Handle non-finite values specially - these can't be written as float literals
    if val.is_nan() {
        // Return f32::NAN as a path expression
        let path: syn::Expr = syn::parse_quote_spanned!(span=> f32::NAN);
        return Expr::Verbatim(path);
    }
    if val.is_infinite() {
        // Return f32::INFINITY or f32::NEG_INFINITY
        let path: syn::Expr = if val.is_sign_positive() {
            syn::parse_quote_spanned!(span=> f32::INFINITY)
        } else {
            syn::parse_quote_spanned!(span=> f32::NEG_INFINITY)
        };
        return Expr::Verbatim(path);
    }
    let mut s = val.to_string();
    if !s.contains('.') && !s.contains('e') && !s.contains('E') {
        s.push_str(".0");
    }
    let lit = syn::LitFloat::new(&s, span);
    Expr::Literal(LiteralExpr {
        lit: Lit::Float(lit),
        span,
    })
}

fn fold_constants(op: BinaryOp, lhs: f64, rhs: f64) -> Option<f64> {
    let result = match op {
        BinaryOp::Add => lhs + rhs,
        BinaryOp::Sub => lhs - rhs,
        BinaryOp::Mul => lhs * rhs,
        BinaryOp::Div => lhs / rhs,
        BinaryOp::Rem => lhs % rhs,
        _ => return None, // Comparisons etc. not folded to float (return bool)
    };
    // Don't fold to infinity or NaN - keep the expression form
    // so the runtime can handle it appropriately
    if result.is_finite() {
        Some(result)
    } else {
        None
    }
}

fn fold_unary(op: UnaryOp, val: f64) -> Option<f64> {
    match op {
        UnaryOp::Neg => Some(-val),
        _ => None,
    }
}

fn simplify_algebraic(binary: &BinaryExpr) -> Option<Expr> {
    let lhs_val = extract_f64(&binary.lhs);
    let rhs_val = extract_f64(&binary.rhs);

    match binary.op {
        BinaryOp::Add => {
            // x + 0 = x
            if is_zero(rhs_val) {
                return Some(*binary.lhs.clone());
            }
            // 0 + x = x
            if is_zero(lhs_val) {
                return Some(*binary.rhs.clone());
            }
        }
        BinaryOp::Sub => {
            // x - 0 = x
            if is_zero(rhs_val) {
                return Some(*binary.lhs.clone());
            }
        }
        BinaryOp::Mul => {
            // x * 1 = x
            if is_one(rhs_val) {
                return Some(*binary.lhs.clone());
            }
            // 1 * x = x
            if is_one(lhs_val) {
                return Some(*binary.rhs.clone());
            }
            // x * 0 = 0
            if is_zero(rhs_val) {
                return Some(make_literal(0.0, binary.span));
            }
            // 0 * x = 0
            if is_zero(lhs_val) {
                return Some(make_literal(0.0, binary.span));
            }
        }
        BinaryOp::Div => {
            // x / 1 = x
            if is_one(rhs_val) {
                return Some(*binary.lhs.clone());
            }
            // 0 / x = 0
            if is_zero(lhs_val) {
                return Some(make_literal(0.0, binary.span));
            }
        }
        _ => {}
    }

    None
}

fn is_zero(val: Option<f64>) -> bool {
    matches!(val, Some(v) if v.abs() < f64::EPSILON)
}

fn is_one(val: Option<f64>) -> bool {
    matches!(val, Some(v) if (v - 1.0).abs() < f64::EPSILON)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;
    use crate::sema::analyze;
    use quote::quote;

    fn optimize_code(input: proc_macro2::TokenStream) -> String {
        let kernel = parse(input).unwrap();
        let analyzed = analyze(kernel).unwrap();
        let optimized = optimize(analyzed);
        format!("{:?}", optimized.def.body)
    }

    fn optimize_code_egraph(input: proc_macro2::TokenStream, costs: &CostModel) -> String {
        let kernel = parse(input).unwrap();
        let analyzed = analyze(kernel).unwrap();
        let optimized = optimize_with_egraph(analyzed, costs);
        format!("{:?}", optimized.def.body)
    }

    #[test]
    fn test_constant_folding() {
        let input = quote! { |x: f32| x + (1.0 + 2.0) };
        let debug = optimize_code(input);
        assert!(debug.contains("LiteralExpr"));
        assert!(debug.contains("3.0"));
        assert!(!debug.contains("1.0"));
        assert!(!debug.contains("2.0"));
    }

    #[test]
    fn test_identity_add() {
        let input = quote! { |x: f32| x + 0.0 };
        let debug = optimize_code(input);
        assert!(debug.contains("IdentExpr"));
        assert!(debug.contains("x"));
        assert!(!debug.contains("BinaryExpr"));
    }

    #[test]
    fn test_zero_mul() {
        let input = quote! { |x: f32| x * 0.0 };
        let debug = optimize_code(input);
        assert!(debug.contains("LiteralExpr"));
        assert!(debug.contains("0.0"));
        assert!(!debug.contains("IdentExpr"));
    }

    #[test]
    fn test_complex_folding() {
        let input = quote! { |x: f32| (1.0 + 2.0) * x + 0.0 };
        let debug = optimize_code(input);
        assert!(debug.contains("3.0"));
        assert!(debug.contains("x"));
    }

    // ========================================================================
    // E-Graph Integration Tests
    // ========================================================================

    #[test]
    fn test_egraph_identity_add() {
        let input = quote! { |x: f32| x + 0.0 };
        let debug = optimize_code_egraph(input, &CostModel::default());
        // Should simplify to just x
        assert!(debug.contains("IdentExpr"));
        assert!(debug.contains("x"));
        assert!(!debug.contains("Add"));
    }

    #[test]
    fn test_egraph_identity_mul() {
        let input = quote! { |x: f32| x * 1.0 };
        let debug = optimize_code_egraph(input, &CostModel::default());
        // Should simplify to just x
        assert!(debug.contains("IdentExpr"));
        assert!(debug.contains("x"));
        assert!(!debug.contains("Mul"));
    }

    #[test]
    fn test_egraph_zero_mul() {
        let input = quote! { |x: f32| x * 0.0 };
        let debug = optimize_code_egraph(input, &CostModel::default());
        // Should simplify to 0.0
        eprintln!("test_egraph_zero_mul output: {}", debug);
        assert!(debug.contains("LiteralExpr"), "Expected LiteralExpr in: {}", debug);
        assert!(debug.contains("0.0") || debug.contains("0"), "Expected 0 in: {}", debug);
    }

    #[test]
    fn test_egraph_sub_self() {
        let input = quote! { |x: f32| x - x };
        let debug = optimize_code_egraph(input, &CostModel::default());
        // Should simplify to 0.0
        assert!(debug.contains("0"));
    }

    #[test]
    fn test_egraph_fma_fusion_with_fma_costs() {
        // a * b + c should become mul_add when FMA is cheap
        let input = quote! { |a: f32, b: f32, c: f32| a * b + c };
        let debug = optimize_code_egraph(input, &CostModel::with_fma());
        // With FMA costs, should extract mul_add
        assert!(debug.contains("mul_add"));
    }

    #[test]
    fn test_egraph_fma_unfused_with_expensive_fma() {
        // a * b + c should stay as mul + add when FMA is made expensive
        let input = quote! { |a: f32, b: f32, c: f32| a * b + c };

        // Create a cost model where MulAdd is artificially expensive
        let mut expensive_fma = CostModel::default();
        expensive_fma.set_cost(pixelflow_ir::OpKind::MulAdd, 20); // More than mul(5) + add(4) = 9

        let debug = optimize_code_egraph(input, &expensive_fma);
        // With expensive FMA, should prefer unfused (mul(5) + add(4) = 9 < mul_add(20))
        assert!(!debug.contains("mul_add"), "Expected unfused when MulAdd is expensive: {}", debug);
    }

    #[test]
    fn test_egraph_fma_fused_with_default_costs() {
        // With realistic default costs, FMA should be preferred (MulAdd(5) < Mul(5) + Add(4))
        let input = quote! { |a: f32, b: f32, c: f32| a * b + c };
        let debug = optimize_code_egraph(input, &CostModel::default());
        // Modern CPUs have cheap FMA, so it should be fused
        assert!(debug.contains("mul_add"), "Expected FMA fusion with default costs: {}", debug);
    }

    #[test]
    fn test_egraph_complex_expression() {
        // ((x + 0) * 1 - x) should simplify to 0
        let input = quote! { |x: f32| (x + 0.0) * 1.0 - x };
        let debug = optimize_code_egraph(input, &CostModel::default());
        // Should simplify to 0.0
        assert!(debug.contains("0"));
    }

    #[test]
    fn test_egraph_preserves_variables() {
        // Simple expression with named variables
        let input = quote! { |cx: f32, cy: f32| cx + cy };
        let debug = optimize_code_egraph(input, &CostModel::default());
        // Should preserve variable names
        assert!(debug.contains("cx"));
        assert!(debug.contains("cy"));
    }

    #[test]
    fn test_egraph_handles_sqrt() {
        let input = quote! { |x: f32| x.sqrt() };
        let debug = optimize_code_egraph(input, &CostModel::default());
        // Should preserve sqrt
        assert!(debug.contains("sqrt"));
    }

    #[test]
    fn test_egraph_div_sqrt_to_rsqrt() {
        // x / sqrt(y) should become x * rsqrt(y) via algebra:
        // x / sqrt(y) = x * (1/sqrt(y)) = x * rsqrt(y)
        let input = quote! { |x: f32, y: f32| x / y.sqrt() };
        let debug = optimize_code_egraph(input, &CostModel::with_fast_rsqrt());
        // Should use rsqrt (real instruction) instead of 1/sqrt
        assert!(debug.contains("rsqrt"), "Expected rsqrt in: {}", debug);
    }

    #[test]
    fn test_egraph_double_negation() {
        let input = quote! { |x: f32| - -x };
        let debug = optimize_code_egraph(input, &CostModel::default());
        // Should simplify to just x
        assert!(debug.contains("IdentExpr"));
        assert!(debug.contains("x"));
        assert!(!debug.contains("Neg"));
    }

    // ========================================================================
    // Cross-Binding Optimization Tests (Global Pass)
    // ========================================================================

    #[test]
    fn test_global_optimization_across_let_bindings() {
        // The global pass should see through let bindings:
        // let a = x; let b = 0.0; a + b → x
        let input = quote! { |x: f32| {
            let a = x;
            let b = 0.0;
            a + b
        }};
        let debug = optimize_code_egraph(input, &CostModel::default());
        // Should simplify to just x (no Add, no 0.0 literal in result)
        assert!(debug.contains("x"), "Expected x in output: {}", debug);
        assert!(!debug.contains("Add"), "Should eliminate addition with zero: {}", debug);
    }

    #[test]
    fn test_global_optimization_zero_multiplication() {
        // let a = x * x; let b = 0.0; a * b → 0.0
        let input = quote! { |x: f32| {
            let a = x * x;
            let b = 0.0;
            a * b
        }};
        let debug = optimize_code_egraph(input, &CostModel::default());
        // Should simplify to 0.0
        assert!(debug.contains("0"), "Expected 0 in output: {}", debug);
        assert!(!debug.contains("Mul"), "Should eliminate multiplication: {}", debug);
    }

    #[test]
    fn test_global_optimization_self_subtraction() {
        // let a = X * X + Y * Y; a - a → 0.0
        let input = quote! { || {
            let a = X * X + Y * Y;
            a - a
        }};
        let debug = optimize_code_egraph(input, &CostModel::default());
        // Should simplify to 0.0
        assert!(debug.contains("0"), "Expected 0 in output: {}", debug);
    }

    #[test]
    fn test_global_fma_across_bindings() {
        // let product = a * b; product + c → mul_add(a, b, c)
        let input = quote! { |a: f32, b: f32, c: f32| {
            let product = a * b;
            product + c
        }};
        let debug = optimize_code_egraph(input, &CostModel::with_fma());
        // Should fuse into mul_add
        assert!(debug.contains("mul_add"), "Expected FMA fusion: {}", debug);
    }

    #[test]
    fn test_discriminant_pattern() {
        // This is the problematic pattern:
        // d_dot_c² - (c_sq - r_sq) should use Neg to wrap (c_sq - r_sq)
        let input = quote! { |d: f32, c: f32, r: f32| {
            let d_sq = d * d;
            let c_sq = c * c;
            let r_sq = r * r;
            d_sq - (c_sq - r_sq)
        }};
        let debug = optimize_code_egraph(input, &CostModel::fully_optimized());
        eprintln!("Discriminant AST: {}", debug);

        // The AST should contain a Neg wrapping the inner subtraction
        // If FMA is used: mul_add(d, d, Neg(Sub(c_sq, r_sq)))
        // The key check: the output should NOT have the wrong sign pattern
        // Wrong pattern: Sub(c_sq, Neg(r_sq)) which equals c_sq + r_sq
        // Right pattern: Neg(Sub(c_sq, r_sq)) which equals -c_sq + r_sq

        // With FMA fusion, we expect: mul_add(d, d, ...)
        // And the third argument should involve a Neg wrapping the subtraction
        assert!(debug.contains("mul_add"), "Expected FMA fusion: {}", debug);

        // Check that Neg appears in the output (wrapping the inner expression)
        assert!(debug.contains("Neg") || debug.contains("neg"),
                "Expected Neg in third argument of mul_add: {}", debug);
    }

    #[test]
    fn test_discriminant_with_intrinsics() {
        // This matches the actual failing test more closely:
        // d_dot_c = X*cx + Y*cy + Z*cz
        // c_sq = cx*cx + cy*cy + cz*cz
        // r_sq = r*r
        // discriminant = d_dot_c*d_dot_c - (c_sq - r_sq)
        let input = quote! { |cx: f32, cy: f32, cz: f32, r: f32| {
            let d_dot_c = X * cx + Y * cy + Z * cz;
            let c_sq = cx * cx + cy * cy + cz * cz;
            let r_sq = r * r;
            d_dot_c * d_dot_c - (c_sq - r_sq)
        }};
        let debug = optimize_code_egraph(input, &CostModel::fully_optimized());
        eprintln!("Discriminant with intrinsics AST: {}", debug);

        // Check for FMA
        assert!(debug.contains("mul_add"), "Expected FMA fusion: {}", debug);

        // Check that Neg appears - the key correctness check
        assert!(debug.contains("Neg") || debug.contains("neg"),
                "Expected Neg in expression: {}", debug);
    }

    // ========================================================================
    // DAG Extraction Tests
    // ========================================================================

    /// Test DAG optimization with shared subexpressions.
    #[test]
    fn test_dag_optimization_shared_subexpr() {
        // sin(X) * sin(X) should emit a let-binding
        let input = quote! { || X.sin() * X.sin() };
        let kernel = parse(input).unwrap();
        let analyzed = analyze(kernel).unwrap();

        // Use neural optimizer
        let nnue = ExprNnue::new_random(42);
        let optimized = optimize_expr_with_nnue(analyzed.def.body.clone(), &nnue);

        let debug = format!("{:?}", optimized);
        eprintln!("DAG optimized sin(X)*sin(X): {}", debug);

        // The output should either:
        // 1. Have a let-binding for the shared sin(X), OR
        // 2. Reference the same subexpression (e-graph dedup)
        // For now, just verify it's well-formed
        assert!(debug.contains("sin") || debug.contains("Sin"), "Expected sin in output");
    }

    /// Test DAG optimization with triple use of shared subexpr.
    #[test]
    fn test_dag_optimization_triple_shared() {
        // sqrt(X) * sqrt(X) + sqrt(X) should emit let-binding
        let input = quote! { || X.sqrt() * X.sqrt() + X.sqrt() };
        let kernel = parse(input).unwrap();
        let analyzed = analyze(kernel).unwrap();

        let nnue = ExprNnue::new_random(42);
        let optimized = optimize_expr_with_nnue(analyzed.def.body.clone(), &nnue);

        let debug = format!("{:?}", optimized);
        eprintln!("DAG optimized sqrt(X)*sqrt(X)+sqrt(X): {}", debug);

        // Should have sqrt in output
        assert!(debug.contains("sqrt"), "Expected sqrt in output");
    }

    /// Test that simple expressions without sharing don't get wrapped in blocks.
    #[test]
    fn test_dag_optimization_no_sharing() {
        // X + Y has no sharing, should remain simple
        let input = quote! { || X + Y };
        let kernel = parse(input).unwrap();
        let analyzed = analyze(kernel).unwrap();

        let nnue = ExprNnue::new_random(42);
        let optimized = optimize_expr_with_nnue(analyzed.def.body.clone(), &nnue);

        let debug = format!("{:?}", optimized);
        eprintln!("DAG optimized X+Y: {}", debug);

        // Should NOT be wrapped in a block
        assert!(!debug.starts_with("Block"), "Simple expression should not be wrapped in block");
    }

    #[test]
    fn test_full_pipeline_discriminant() {
        use crate::codegen;

        // Full pipeline test matching actual kernel! macro
        let input = quote! { |cx: f32, cy: f32, cz: f32, r: f32| -> Jet3 {
            let d_dot_c = X * cx + Y * cy + Z * cz;
            let c_sq = cx * cx + cy * cy + cz * cz;
            let r_sq = r * r;
            d_dot_c * d_dot_c - (c_sq - r_sq)
        }};

        let kernel = parse(input).unwrap();
        let analyzed = analyze(kernel).unwrap();

        // This is what the kernel! macro does
        let optimized = optimize(analyzed);

        eprintln!("Optimized AST: {:?}", optimized.def.body);

        let output = codegen::emit(optimized);
        let output_str = output.to_string();

        eprintln!("Generated code:\n{}", output_str);

        // The key check: the output should have .neg() wrapping the inner subtraction
        // NOT: c_sq - r * r.neg() (which is c_sq + r²)
        // YES: (c_sq - r_sq).neg() (which is -c_sq + r²)

        // Check for the WRONG pattern (the bug)
        let has_wrong_pattern = output_str.contains("r . neg ( )") && !output_str.contains(") . neg ( )");
        assert!(!has_wrong_pattern, "Found wrong pattern (r.neg() without wrapping): {}", output_str);
    }
}