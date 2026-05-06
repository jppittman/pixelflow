//! # Semantic Analysis
//!
//! Analyzes the AST for semantic correctness and annotates it with symbol information.
//!
//! ## Responsibilities
//!
//! 1. **Symbol Resolution**: Match identifiers to their definitions
//! 2. **Scope Management**: Track let bindings within blocks
//! 3. **Validation**: Ensure all referenced symbols are defined
//!
//! ## Symbol Resolution Rules
//!
//! When an identifier is encountered:
//! 1. Check if it's an intrinsic (X, Y, Z, W) → leave unchanged
//! 2. Check if it's a captured parameter → transform to `self.param`
//! 3. Check if it's a local variable → leave unchanged
//! 4. Otherwise → error (undefined symbol)
//!
//! ## Output
//!
//! The semantic phase produces an `AnalyzedKernel` which includes:
//! - The original AST (possibly annotated)
//! - The populated symbol table
//! - Any resolved type information

use crate::ast::{BlockExpr, Expr, KernelDef, LetStmt, MethodCallExpr, Param, ParamKind, Stmt};
use crate::symbol::{SymbolKind, SymbolTable};
use pixelflow_ir::known_method_names;
use syn::Ident;

/// DSL-specific methods that aren't IR operations.
/// These are handled separately in the macro/runtime.
const DSL_METHODS: &[&str] = &[
    "at",       // coordinate transformation
    "constant", // collapse to Field
    "collapse", // alias for constant
    "clone",    // clone for reuse
];

/// The result of semantic analysis.
#[derive(Debug)]
pub struct AnalyzedKernel {
    /// The original kernel definition.
    pub def: KernelDef,
    /// The populated symbol table.
    pub symbols: SymbolTable,
}

/// Perform semantic analysis on a parsed kernel.
pub fn analyze(kernel: KernelDef) -> syn::Result<AnalyzedKernel> {
    // Anonymous kernels (no struct_decl) allow captured variables from environment
    let is_anonymous = kernel.struct_decl.is_none();
    let mut analyzer = SemanticAnalyzer::new(is_anonymous);

    // Register all parameters in the symbol table
    for param in &kernel.params {
        analyzer.register_parameter(param)?;
    }

    // Analyze the body expression
    analyzer.analyze_expr(&kernel.body)?;

    Ok(AnalyzedKernel {
        def: kernel,
        symbols: analyzer.symbols,
    })
}

/// The semantic analyzer state.
struct SemanticAnalyzer {
    symbols: SymbolTable,
    /// Whether this is an anonymous kernel (allows captured variables).
    is_anonymous: bool,
}

impl SemanticAnalyzer {
    fn new(is_anonymous: bool) -> Self {
        SemanticAnalyzer {
            symbols: SymbolTable::new(),
            is_anonymous,
        }
    }

    /// Register a parameter in the symbol table.
    fn register_parameter(&mut self, param: &Param) -> syn::Result<()> {
        let name = param.name.to_string();

        // Check for shadowing intrinsics (error)
        if self.symbols.is_intrinsic(&name) {
            return Err(syn::Error::new(
                param.name.span(),
                format!(
                    "parameter '{}' shadows intrinsic coordinate variable\n\
                     note: intrinsics are: X, Y, Z, W (coordinate variables)\n\
                     help: rename this parameter to something else",
                    name
                ),
            ));
        }

        // Check for duplicate parameters
        if self.symbols.lookup(&name).is_some() {
            return Err(syn::Error::new(
                param.name.span(),
                format!(
                    "duplicate parameter '{}'\n\
                     help: each parameter must have a unique name",
                    name
                ),
            ));
        }

        // Register based on parameter kind
        match &param.kind {
            ParamKind::Scalar(ty) => {
                self.symbols
                    .register_parameter(param.name.clone(), ty.clone());
            }
            ParamKind::Manifold => {
                self.symbols.register_manifold_param(param.name.clone());
            }
        }
        Ok(())
    }

    /// Analyze an expression for symbol resolution.
    fn analyze_expr(&mut self, expr: &Expr) -> syn::Result<()> {
        match expr {
            Expr::Ident(ident_expr) => {
                self.resolve_ident(&ident_expr.name)?;
            }

            Expr::Literal(_) => {
                // Literals are always valid
            }

            Expr::Binary(binary) => {
                self.analyze_expr(&binary.lhs)?;
                self.analyze_expr(&binary.rhs)?;
            }

            Expr::Unary(unary) => {
                self.analyze_expr(&unary.operand)?;
            }

            Expr::MethodCall(call) => {
                self.analyze_method_call(call)?;
            }

            Expr::Call(call) => {
                // Analyze all arguments (function name is external, not resolved here)
                for arg in &call.args {
                    self.analyze_expr(arg)?;
                }
            }

            Expr::Block(block) => {
                self.analyze_block(block)?;
            }

            Expr::Paren(inner) => {
                self.analyze_expr(inner)?;
            }

            Expr::Tuple(tuple) => {
                for elem in &tuple.elems {
                    self.analyze_expr(elem)?;
                }
            }

            Expr::Verbatim(_) => {
                // Verbatim expressions pass through without analysis
                // The Rust compiler will catch any errors
            }
        }
        Ok(())
    }

    /// Resolve an identifier reference.
    fn resolve_ident(&self, ident: &Ident) -> syn::Result<SymbolKind> {
        let name = ident.to_string();

        match self.symbols.lookup(&name) {
            Some(symbol) => Ok(symbol.kind),
            None => {
                // For anonymous kernels, unknown symbols are captured from environment
                // The Rust closure will handle the capture - no error needed
                if self.is_anonymous {
                    return Ok(SymbolKind::Local); // Treat as external/captured
                }

                // For named kernels, undefined symbols are errors
                let suggestion = self.find_similar_symbol(&name);
                let msg = match suggestion {
                    Some(similar) => format!(
                        "undefined symbol '{}'\n\
                         help: did you mean '{}'?\n\
                         note: available intrinsics: X, Y, Z, W",
                        name, similar
                    ),
                    None => format!(
                        "undefined symbol '{}'\n\
                         note: available intrinsics: X, Y, Z, W\n\
                         help: check spelling or add as a parameter",
                        name
                    ),
                };
                Err(syn::Error::new(ident.span(), msg))
            }
        }
    }

    /// Find a similar symbol name for typo suggestions.
    fn find_similar_symbol(&self, name: &str) -> Option<String> {
        let name_lower = name.to_lowercase();

        // Check intrinsics first (common typos)
        let intrinsics = ["X", "Y", "Z", "W"];
        for intr in intrinsics {
            if intr.to_lowercase() == name_lower {
                return Some(intr.to_string());
            }
        }

        // Check parameters and locals
        for sym_name in self.symbols.all_names() {
            // Simple similarity: same length and differs by 1-2 chars
            if sym_name.len() == name.len() {
                let diff_count = sym_name
                    .chars()
                    .zip(name.chars())
                    .filter(|(a, b)| a != b)
                    .count();
                if diff_count <= 2 {
                    return Some(sym_name);
                }
            }
            // Case-insensitive match
            if sym_name.to_lowercase() == name_lower {
                return Some(sym_name);
            }
        }

        None
    }

    /// Analyze a method call.
    fn analyze_method_call(&mut self, call: &MethodCallExpr) -> syn::Result<()> {
        // Analyze the receiver
        self.analyze_expr(&call.receiver)?;

        // Analyze arguments
        for arg in &call.args {
            self.analyze_expr(arg)?;
        }

        // Validate method name against known methods (IR ops + DSL methods)
        let method_name = call.method.to_string();
        let is_ir_method = known_method_names().any(|m| m == method_name);
        let is_dsl_method = DSL_METHODS.contains(&method_name.as_str());

        if !is_ir_method && !is_dsl_method {
            // Find similar method for suggestion - collect all known methods
            let all_methods: Vec<&str> = known_method_names()
                .chain(DSL_METHODS.iter().copied())
                .collect();

            let suggestion = all_methods
                .iter()
                .find(|&&m| {
                    let m_lower = m.to_lowercase();
                    let name_lower = method_name.to_lowercase();
                    m_lower == name_lower
                        || (m.len() == method_name.len()
                            && m.chars()
                                .zip(method_name.chars())
                                .filter(|(a, b)| a != b)
                                .count()
                                <= 2)
                })
                .copied();

            let msg = match suggestion {
                Some(similar) => format!(
                    "unknown method '{}'\n\
                     help: did you mean '{}'?",
                    method_name, similar
                ),
                None => format!(
                    "unknown method '{}'\n\
                     note: common methods: sqrt, abs, sin, cos, exp, min, max, clone\n\
                     help: see ManifoldExt trait for available methods",
                    method_name
                ),
            };

            return Err(syn::Error::new(call.method.span(), msg));
        }
        Ok(())
    }

    /// Analyze a block expression.
    fn analyze_block(&mut self, block: &BlockExpr) -> syn::Result<()> {
        // Enter a new scope
        self.symbols.push_scope();

        // Analyze each statement
        for stmt in &block.stmts {
            match stmt {
                Stmt::Let(let_stmt) => {
                    self.analyze_let(let_stmt)?;
                }
                Stmt::Expr(expr) => {
                    self.analyze_expr(expr)?;
                }
            }
        }

        // Analyze the final expression
        if let Some(expr) = &block.expr {
            self.analyze_expr(expr)?;
        }

        // Exit the scope
        self.symbols.pop_scope();

        Ok(())
    }

    /// Analyze a let statement.
    fn analyze_let(&mut self, let_stmt: &LetStmt) -> syn::Result<()> {
        // First, analyze the initializer (uses current scope)
        self.analyze_expr(&let_stmt.init)?;

        // Then register the new binding
        let name = let_stmt.name.to_string();

        // Warning: shadowing intrinsics in let is allowed but unusual
        if self.symbols.is_intrinsic(&name) {
            // Could emit a warning here in the future
        }

        self.symbols
            .register_local(let_stmt.name.clone(), let_stmt.ty.clone());

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;
    use quote::quote;

    #[test]
    fn analyze_simple_kernel() {
        let input = quote! { |r: f32| X * X + Y * Y - r };
        let kernel = parse(input).unwrap();
        let analyzed = analyze(kernel).unwrap();

        assert!(analyzed.symbols.is_parameter("r"));
        assert!(analyzed.symbols.is_intrinsic("X"));
        assert!(analyzed.symbols.is_intrinsic("Y"));
    }

    #[test]
    fn error_on_undefined_symbol() {
        // Named kernels reject undefined symbols (anonymous kernels allow captures)
        let input = quote! { struct Test = |r: f32| X * X + undefined_var };
        let kernel = parse(input).unwrap();
        let result = analyze(kernel);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("undefined symbol"));
    }

    #[test]
    fn anonymous_allows_captured_variables() {
        // Anonymous kernels allow captured variables from environment
        let input = quote! { |r: f32| X * X + captured_from_env };
        let kernel = parse(input).unwrap();
        let result = analyze(kernel);
        assert!(result.is_ok(), "Anonymous kernels should allow captured variables");
    }

    #[test]
    fn error_on_shadowing_intrinsic() {
        let input = quote! { |X: f32| X * X }; // X shadows the intrinsic
        let kernel = parse(input).unwrap();
        let result = analyze(kernel);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("shadows intrinsic"));
    }

    #[test]
    fn block_scoping() {
        let input = quote! {
            |cx: f32| {
                let dx = X - cx;
                dx * dx
            }
        };
        let kernel = parse(input).unwrap();
        let result = analyze(kernel);
        assert!(result.is_ok());
    }

    #[test]
    fn typo_suggestion_for_intrinsic() {
        // Lowercase "x" should suggest uppercase "X" (named kernel rejects typos)
        let input = quote! { struct Test = || x * x };
        let kernel = parse(input).unwrap();
        let result = analyze(kernel);

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("undefined symbol"));
        assert!(err.contains("did you mean 'X'"));
    }

    #[test]
    fn typo_suggestion_for_parameter() {
        // "radiu" should suggest "radius" (named kernel rejects typos)
        let input = quote! { struct Test = |radius: f32| X - radiu };
        let kernel = parse(input).unwrap();
        let result = analyze(kernel);

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("undefined symbol"));
        // Similar names with 1-2 char difference should be suggested
    }

    #[test]
    fn error_on_unknown_method() {
        let input = quote! { |r: f32| X.unknownmethod() };
        let kernel = parse(input).unwrap();
        let result = analyze(kernel);

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("unknown method"));
    }

    #[test]
    fn typo_suggestion_for_method() {
        // "sqrtt" should suggest "sqrt"
        let input = quote! { || X.sqrtt() };
        let kernel = parse(input).unwrap();
        let result = analyze(kernel);

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("unknown method"));
    }

    #[test]
    fn known_methods_accepted() {
        // All ManifoldExt methods should be accepted
        let input = quote! { || X.sqrt().abs().sin().cos().clone() };
        let kernel = parse(input).unwrap();
        let result = analyze(kernel);
        assert!(result.is_ok());
    }

    #[test]
    fn error_on_duplicate_parameter() {
        let input = quote! { |r: f32, r: f32| X - r };
        let kernel = parse(input).unwrap();
        let result = analyze(kernel);

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("duplicate parameter"));
    }
}
