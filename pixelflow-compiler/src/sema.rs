//! # Semantic Analysis
//!
//! Analyzes the AST for semantic correctness and annotates it with symbol information.
//!
//! ## Responsibilities
//!
//! 1. **Symbol Resolution**: Match identifiers to their definitions
//! 2. **Scope Management**: Track let bindings within blocks, with Rust's
//!    lexical scoping ([`crate::symbol::Scopes`], which lowering resolves
//!    through too)
//! 3. **Validation**: Ensure all referenced symbols are defined
//!
//! ## Symbol Resolution Rules
//!
//! An identifier resolves to the innermost binding of its name in scope:
//! 1. a `let`-bound local → a shared arena id
//! 2. a declared parameter → a `Param` folded in by the builder
//! 3. an intrinsic (X, Y) → a coordinate `Var`
//! 4. otherwise → refused. A kernel body does not capture from the caller's
//!    scope, so a name nothing here binds is an error here, with a span —
//!    not a capture that lowering then refuses without one.
//!
//! Nothing shadows X or Y — a parameter or a `let` of that name is refused —
//! so a coordinate always means the coordinate.
//!
//! ## Output
//!
//! The semantic phase produces an `AnalyzedKernel`: the AST, validated.

use crate::ast::{BlockExpr, Expr, KernelDef, LetStmt, MethodCallExpr, Param, Stmt};
use crate::lower::LIBRARY_METHODS;
use crate::symbol::{SymbolKind, SymbolTable};
use pixelflow_ir::{OpKind, known_method_names};
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
}

/// Perform semantic analysis on a parsed kernel.
pub fn analyze(kernel: KernelDef) -> syn::Result<AnalyzedKernel> {
    let mut analyzer = SemanticAnalyzer::new();

    // Register all parameters in the symbol table
    for param in &kernel.params {
        analyzer.register_parameter(param)?;
    }

    // Analyze the body expression
    analyzer.analyze_expr(&kernel.body)?;

    Ok(AnalyzedKernel { def: kernel })
}

/// The semantic analyzer state.
/// Coordinate names a `kernel!` body may not use: they named the Z and W
/// axes, which a lattice no longer has.
const RETIRED_COORDINATES: [&str; 2] = ["Z", "W"];

/// Maximum per-character difference for a same-length method name to be
/// suggested as a typo fix (e.g. `sqrtt` -> `sqrt`).
const MAX_TYPO_CHAR_DIFF: usize = 2;

struct SemanticAnalyzer {
    symbols: SymbolTable,
}

impl SemanticAnalyzer {
    fn new() -> Self {
        SemanticAnalyzer {
            symbols: SymbolTable::new(),
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
                     note: intrinsics are: X, Y (coordinate variables)\n\
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

        self.symbols
            .register_parameter(param.name.clone(), (*param.ty).clone());
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

    /// Resolve an identifier reference to the innermost binding of its name
    /// in scope, or refuse it.
    ///
    /// An unknown name used to be accepted here as a capture from the
    /// caller's scope, which lowering then refused as `Unknown identifier` —
    /// one stage accepting what a later one refuses, and the later one has no
    /// span to point with. Worse, it hid an out-of-scope local: the leaked
    /// binding lowering never popped was still there to be found, so
    /// `{ { let a = X; a }; a }` compiled.
    fn resolve_ident(&self, ident: &Ident) -> syn::Result<SymbolKind> {
        let name = ident.to_string();
        if let Some(symbol) = self.symbols.lookup(&name) {
            return Ok(symbol.kind);
        }
        // `Z` and `W` were coordinate intrinsics until a lattice became two
        // axes; say so, rather than that the name is missing.
        if let Some(axis) = RETIRED_COORDINATES.iter().find(|a| **a == name) {
            return Err(syn::Error::new(
                ident.span(),
                format!(
                    "`{axis}` is no longer a coordinate: a lattice has two axes, X and Y\n\
                     note: a scalar that is the same at every sample is a uniform, not an axis\n\
                     help: declare it as a parameter of this kernel and pass a \
                     `Uniform` handle at the call site"
                ),
            ));
        }
        Err(syn::Error::new(
            ident.span(),
            format!(
                "cannot find `{name}` in this kernel body\n\
                 note: a kernel body sees X, Y, its parameters, and the `let` bindings in \
                 scope; a `let` inside a block goes out of scope where the block ends\n\
                 note: a value from the enclosing Rust scope is not captured\n\
                 help: to use an outside value, declare it as a parameter of this kernel \
                 and pass it at the call site"
            ),
        ))
    }

    /// Analyze a method call.
    fn analyze_method_call(&mut self, call: &MethodCallExpr) -> syn::Result<()> {
        // Analyze the receiver
        self.analyze_expr(&call.receiver)?;

        // Analyze arguments
        for arg in &call.args {
            self.analyze_expr(arg)?;
        }

        // Validate method name AND arity against known methods (IR ops +
        // library compositions + DSL methods) — `OpKind::from_method_call`
        // checks arity, so `.sqrt(1)` is rejected here rather than slipping
        // through as "known" and failing later with a less specific error.
        let method_name = call.method.to_string();
        let arg_count = call.args.len();
        let is_ir_method = OpKind::from_method_call(&method_name, arg_count).is_some();
        let is_library_method = LIBRARY_METHODS.contains(&(method_name.as_str(), arg_count));
        let is_dsl_method = DSL_METHODS.contains(&method_name.as_str());

        if !is_ir_method && !is_library_method && !is_dsl_method {
            // A recognized name at the wrong arity is not an unknown name, and
            // sending it into the typo search below produced the useless
            // `unknown method 'sqrt'; did you mean 'sqrt'?` — the search found
            // the very name it had just declared unknown. Answer the question
            // the caller actually got wrong.
            if let Some(want) = Self::expected_arg_count(&method_name) {
                return Err(syn::Error::new(
                    call.method.span(),
                    format!(
                        "`{method_name}` takes {want} argument{}, but {arg_count} \
                         {} supplied",
                        if want == 1 { "" } else { "s" },
                        if arg_count == 1 { "was" } else { "were" },
                    ),
                ));
            }

            // Find similar method for suggestion - collect all known methods
            let all_methods: Vec<&str> = known_method_names()
                .chain(LIBRARY_METHODS.iter().map(|(name, _)| *name))
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
                                <= MAX_TYPO_CHAR_DIFF)
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
                     help: see Kernel's method surface for what is available",
                    method_name
                ),
            };

            return Err(syn::Error::new(call.method.span(), msg));
        }
        Ok(())
    }

    /// The argument count a known method takes, or `None` if no method has
    /// that name at any arity.
    ///
    /// Name and arity are separate questions. `OpKind::from_method_call`
    /// deliberately answers them together — that is what makes `.sqrt(1.0)` a
    /// hard error rather than something that slips through and fails later —
    /// but a *diagnostic* has to take them apart again to say which one is
    /// wrong.
    ///
    /// Asking `from_method_call` again at the op's own arity is what
    /// distinguishes a DSL method from an op that merely shares a name
    /// (`add`, `shl`), without this module needing to see the private
    /// predicate that decides it.
    fn expected_arg_count(name: &str) -> Option<usize> {
        if let Some(op) = OpKind::from_name(name) {
            let args = op.arity().checked_sub(1)?;
            if OpKind::from_method_call(name, args).is_some() {
                return Some(args);
            }
        }
        LIBRARY_METHODS
            .iter()
            .find(|(known, _)| *known == name)
            .map(|(_, count)| *count)
    }

    /// Analyze a block expression in a scope of its own.
    fn analyze_block(&mut self, block: &BlockExpr) -> syn::Result<()> {
        self.symbols.push_scope();
        let analyzed = self.analyze_block_contents(block);
        self.symbols.pop_scope();
        analyzed
    }

    /// A block's statements in order, then its value, in the scope
    /// [`Self::analyze_block`] opened.
    fn analyze_block_contents(&mut self, block: &BlockExpr) -> syn::Result<()> {
        for stmt in &block.stmts {
            match stmt {
                Stmt::Let(let_stmt) => self.analyze_let(let_stmt)?,
                Stmt::Expr(expr) => self.analyze_expr(expr)?,
            }
        }
        match &block.expr {
            Some(expr) => self.analyze_expr(expr),
            None => Ok(()),
        }
    }

    /// Analyze a let statement.
    fn analyze_let(&mut self, let_stmt: &LetStmt) -> syn::Result<()> {
        let name = let_stmt.name.to_string();

        // A `let X` would make `X` mean the local below it — and lowering
        // used to match the coordinate names before locals, so the kernel
        // silently read the coordinate instead (`{ let X = Y; X }` gave X).
        // Refused for the same reason as a parameter named X.
        if self.symbols.is_intrinsic(&name) {
            return Err(syn::Error::new(
                let_stmt.name.span(),
                format!(
                    "`let {name}` shadows the intrinsic coordinate variable `{name}`\n\
                     note: intrinsics are: X, Y (coordinate variables), and every use of one \
                     in a kernel body means the coordinate\n\
                     help: rename this binding to something else"
                ),
            ));
        }

        // The initializer is analyzed before the binding exists, so it sees
        // whatever the name meant before: `let a = a + 1.0;`.
        self.analyze_expr(&let_stmt.init)?;
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

    /// A body that names `Z` or `W` is a compile error that says where the
    /// value goes instead — not a capture from the caller's scope, which is
    /// what an unknown name would otherwise become.
    #[test]
    fn a_body_naming_a_retired_axis_is_refused_with_the_uniform_note() {
        for body in [quote! { || X + Z }, quote! { || X * W }] {
            let kernel = parse(body).unwrap();
            let err = analyze(kernel).expect_err("Z and W are not coordinates");
            let text = err.to_string();
            assert!(
                text.contains("no longer a coordinate") && text.contains("Uniform"),
                "the message must point at uniforms, got: {text}"
            );
        }
    }

    /// A parameter may still be *called* Z: the refusal is about the
    /// intrinsic that is gone, not about the letter.
    #[test]
    fn a_parameter_named_z_is_ordinary() {
        let kernel = parse(quote! { |Z: f32| X + Z }).unwrap();
        assert!(analyze(kernel).is_ok());
    }

    #[test]
    fn analyze_simple_kernel() {
        let input = quote! { |r: f32| X * X + Y * Y - r };
        let kernel = parse(input).unwrap();
        assert!(analyze(kernel).is_ok());
    }

    /// A name nothing in the kernel binds is refused here, with its span.
    ///
    /// This test used to pin the opposite — `analyze` accepted an unknown name
    /// as a capture from the caller's scope — while documenting that the
    /// kernel did not compile, because arena lowering has no node for a
    /// captured Rust binding and refused it as `Unknown identifier`. That is
    /// the shape of the `round`/`log10`/`pow` and `fract`/`hypot`/`clamp`
    /// defects: one stage accepting what a later stage refuses. It also hid
    /// an out-of-scope local (see the next test). A capture is still
    /// expressible — the emitted tokens sit in the caller's scope, so it
    /// could fold as a `Const` exactly as a parameter does — and when it is
    /// built, it is built in both stages at once. Pass it as a parameter
    /// meanwhile, which is what the message says.
    #[test]
    fn an_unknown_name_is_refused_not_captured() {
        let input = quote! { |r: f32| X * X + captured_from_env };
        let kernel = parse(input).unwrap();
        let err = analyze(kernel).expect_err("a kernel body does not capture");
        let text = err.to_string();
        assert!(
            text.contains("cannot find `captured_from_env`") && text.contains("parameter"),
            "the message must name the identifier and the way out, got: {text}"
        );
    }

    /// Probe p15. A `let` inside a block goes out of scope where the block
    /// ends, and a use after it is an error, as rustc makes it one. Before
    /// the fix this compiled and read the leaked binding: 3 at `X = 3`.
    #[test]
    fn a_local_used_after_its_block_ends_is_refused() {
        let input = quote! {
            || {
                {
                    let a = X;
                    a
                };
                a
            }
        };
        let kernel = parse(input).unwrap();
        let err = analyze(kernel).expect_err("`a` is out of scope");
        assert!(err.to_string().contains("cannot find `a`"), "got: {err}");
    }

    /// Probe p5. `let X` and `let Y` are refused, as a parameter named X or Y
    /// is. Before the fix `{ let X = Y; X }` compiled and read the
    /// coordinate X, because lowering matched the coordinate names before
    /// locals.
    #[test]
    fn a_let_named_after_a_coordinate_is_refused() {
        for input in [
            quote! { || { let X = Y; X } },
            quote! { || { let Y: f32 = X; Y } },
            quote! { |r: f32| r + { let X = r; X } },
        ] {
            let kernel = parse(input).unwrap();
            let err = analyze(kernel).expect_err("a coordinate cannot be shadowed");
            assert!(
                err.to_string().contains("shadows the intrinsic coordinate"),
                "got: {err}"
            );
        }
    }

    /// A parameter shadowed by a `let` in an inner block is visible again
    /// after the block. The table used to drop the name outright when the
    /// inner block ended; with unknown names now refused, that would have
    /// turned a correct kernel into an error.
    #[test]
    fn a_parameter_shadowed_in_an_inner_block_is_in_scope_after_it() {
        let input = quote! { |r: f32| ({ let r = X; r }) + r };
        let kernel = parse(input).unwrap();
        assert!(analyze(kernel).is_ok());
    }

    /// A `let`'s initializer sees the binding it is about to shadow.
    #[test]
    fn a_let_initializer_sees_the_binding_it_shadows() {
        let input = quote! { |r: f32| { let r = r * 2.0; let a = X; let a = a + r; a } };
        let kernel = parse(input).unwrap();
        assert!(analyze(kernel).is_ok());
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
    fn typo_suggestion_for_method_matches_case_insensitively() {
        // "SQRT" differs from "sqrt" in every char position case-sensitively,
        // so only the case-insensitive fallback catches it.
        let input = quote! { || X.SQRT() };
        let kernel = parse(input).unwrap();
        let result = analyze(kernel);

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("did you mean 'sqrt'"), "{err}");
    }

    #[test]
    fn typo_suggestion_for_method_names_the_exact_match_at_the_two_char_diff_boundary() {
        // "bba" differs from "abs" in exactly 2 chars (position 0: b vs a,
        // position 2: a vs s; position 1 matches) and differs by 3 from
        // every other same-length method name (add, sub, mul, div, neg,
        // min, max, sin, cos, tan, exp, pow, shl, shr) — an unambiguous
        // 2-char-diff match.
        let input = quote! { || X.bba() };
        let kernel = parse(input).unwrap();
        let result = analyze(kernel);

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("did you mean 'abs'"), "{err}");
    }

    #[test]
    fn typo_suggestion_for_method_is_absent_when_no_same_length_method_is_close() {
        // "qqq" shares a length with several 3-letter methods (sin, cos,
        // tan, abs, neg) but differs from every one of them in all 3 chars —
        // matching length alone must not be enough to suggest one.
        let input = quote! { || X.qqq() };
        let kernel = parse(input).unwrap();
        let result = analyze(kernel);

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(!err.contains("did you mean"), "{err}");
    }

    #[test]
    fn known_methods_accepted() {
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
