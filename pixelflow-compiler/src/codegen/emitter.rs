//! Core code emission logic for kernel compilation.

use std::collections::{HashMap, HashSet};

use proc_macro2::TokenStream;
use quote::{ToTokens, format_ident, quote};

use crate::annotate::{AnnotationCtx, annotate};
use crate::ast::{BinaryOp, Expr, ParamKind, Stmt, UnaryOp};
use crate::sema::AnalyzedKernel;
use crate::symbol::SymbolKind;

use super::struct_emitter::{Derives, StructEmitter};
use super::util::{build_array, sort_by_index, standard_imports};

/// Scan an annotated expression to find manifold params used with `.at()`.
/// Returns the set of manifold param names that need ManifoldExt trait bound.
fn find_at_manifold_params(expr: &Expr, symbols: &crate::symbol::SymbolTable) -> HashSet<String> {
    let mut result = HashSet::new();
    find_at_manifold_params_inner(expr, symbols, &mut result);
    result
}

/// Scan an annotated expression to find manifold params that have derivative operations applied.
/// Returns the set of manifold param names that need `Output: HasDerivatives` trait bound.
///
/// Derivative operations: DX(), DY(), DZ(), V() when applied to a manifold param or local
/// that binds to a manifold param.
fn find_derivative_manifold_params(
    expr: &Expr,
    symbols: &crate::symbol::SymbolTable,
) -> HashSet<String> {
    let mut result = HashSet::new();
    let mut locals_to_manifolds: HashMap<String, String> = HashMap::new();
    find_derivative_params_inner(expr, symbols, &mut result, &mut locals_to_manifolds);
    result
}

fn find_derivative_params_inner(
    expr: &Expr,
    symbols: &crate::symbol::SymbolTable,
    result: &mut HashSet<String>,
    locals_to_manifolds: &mut HashMap<String, String>,
) {
    match expr {
        Expr::Call(call) => {
            let func_str = call.func.to_string();
            // Check for derivative operations: DX, DY, DZ, V
            if matches!(func_str.as_str(), "DX" | "DY" | "DZ" | "V") {
                // Check if the argument is a manifold param or a local bound to one
                if let Some(Expr::Ident(ident_expr)) = call.args.first() {
                    let name_str = ident_expr.name.to_string();
                    // Local bound to manifold param (e.g., `let t = geometry;`)
                    if let Some(manifold_name) = locals_to_manifolds.get(&name_str) {
                        result.insert(manifold_name.clone());
                    } else if let Some(symbol) = symbols.lookup(&name_str) {
                        // Direct manifold param reference
                        if matches!(symbol.kind, SymbolKind::ManifoldParam) {
                            result.insert(name_str);
                        }
                    }
                }
            }
            // Recurse into args
            for arg in &call.args {
                find_derivative_params_inner(arg, symbols, result, locals_to_manifolds);
            }
        }
        Expr::MethodCall(call) => {
            find_derivative_params_inner(&call.receiver, symbols, result, locals_to_manifolds);
            for arg in &call.args {
                find_derivative_params_inner(arg, symbols, result, locals_to_manifolds);
            }
        }
        Expr::Binary(binary) => {
            find_derivative_params_inner(&binary.lhs, symbols, result, locals_to_manifolds);
            find_derivative_params_inner(&binary.rhs, symbols, result, locals_to_manifolds);
        }
        Expr::Unary(unary) => {
            find_derivative_params_inner(&unary.operand, symbols, result, locals_to_manifolds);
        }
        Expr::Block(block) => {
            for stmt in &block.stmts {
                match stmt {
                    Stmt::Let(let_stmt) => {
                        // Track local bindings to manifold params: `let t = geometry;`
                        if let Expr::Ident(ident_expr) = &let_stmt.init {
                            let init_name = ident_expr.name.to_string();
                            if let Some(symbol) = symbols.lookup(&init_name) {
                                if matches!(symbol.kind, SymbolKind::ManifoldParam) {
                                    let local_name = let_stmt.name.to_string();
                                    locals_to_manifolds.insert(local_name, init_name);
                                }
                            }
                        }
                        find_derivative_params_inner(
                            &let_stmt.init,
                            symbols,
                            result,
                            locals_to_manifolds,
                        );
                    }
                    Stmt::Expr(expr) => {
                        find_derivative_params_inner(expr, symbols, result, locals_to_manifolds);
                    }
                }
            }
            if let Some(expr) = &block.expr {
                find_derivative_params_inner(expr, symbols, result, locals_to_manifolds);
            }
        }
        Expr::Paren(inner) => {
            find_derivative_params_inner(inner, symbols, result, locals_to_manifolds);
        }
        Expr::Tuple(tuple) => {
            for elem in &tuple.elems {
                find_derivative_params_inner(elem, symbols, result, locals_to_manifolds);
            }
        }
        Expr::Ident(_) | Expr::Literal(_) | Expr::Verbatim(_) => {}
    }
}

/// Collect locals whose bound value (transitively) involves a manifold param.
///
/// Such locals are bound to non-Copy values — tap expressions like
/// `tex.at(...)` capture `&self.tex`, and only fully-ZST expression trees are
/// Copy. The e-graph optimizer is free to extract a form that references such
/// a local more than once, so every use must clone instead of move (the clone
/// copies coordinate ZSTs plus a reference — free).
///
/// Walks blocks in statement order so taint propagates through chains of
/// lets (`let c = tex.at(...); let s = c + ...;` taints both `c` and `s`).
fn find_manifold_tainted_locals(
    expr: &Expr,
    symbols: &crate::symbol::SymbolTable,
) -> HashSet<String> {
    let mut tainted = HashSet::new();
    collect_tainted_locals(expr, symbols, &mut tainted);
    tainted
}

fn collect_tainted_locals(
    expr: &Expr,
    symbols: &crate::symbol::SymbolTable,
    tainted: &mut HashSet<String>,
) {
    match expr {
        Expr::Block(block) => {
            for stmt in &block.stmts {
                match stmt {
                    Stmt::Let(let_stmt) => {
                        // Nested blocks inside the init may bind their own locals.
                        collect_tainted_locals(&let_stmt.init, symbols, tainted);
                        if expr_touches_manifold(&let_stmt.init, symbols, tainted) {
                            tainted.insert(let_stmt.name.to_string());
                        }
                    }
                    Stmt::Expr(e) => collect_tainted_locals(e, symbols, tainted),
                }
            }
            if let Some(e) = &block.expr {
                collect_tainted_locals(e, symbols, tainted);
            }
        }
        Expr::Binary(b) => {
            collect_tainted_locals(&b.lhs, symbols, tainted);
            collect_tainted_locals(&b.rhs, symbols, tainted);
        }
        Expr::Unary(u) => collect_tainted_locals(&u.operand, symbols, tainted),
        Expr::Paren(p) => collect_tainted_locals(p, symbols, tainted),
        Expr::MethodCall(c) => {
            collect_tainted_locals(&c.receiver, symbols, tainted);
            for arg in &c.args {
                collect_tainted_locals(arg, symbols, tainted);
            }
        }
        Expr::Call(c) => {
            for arg in &c.args {
                collect_tainted_locals(arg, symbols, tainted);
            }
        }
        Expr::Tuple(t) => {
            for elem in &t.elems {
                collect_tainted_locals(elem, symbols, tainted);
            }
        }
        Expr::Ident(_) | Expr::Literal(_) | Expr::Verbatim(_) => {}
    }
}

/// Does this expression reference a manifold param or an already-tainted
/// local anywhere?
fn expr_touches_manifold(
    expr: &Expr,
    symbols: &crate::symbol::SymbolTable,
    tainted: &HashSet<String>,
) -> bool {
    match expr {
        Expr::Ident(ident) => {
            let name = ident.name.to_string();
            if tainted.contains(&name) {
                return true;
            }
            matches!(
                symbols.lookup(&name).map(|s| &s.kind),
                Some(SymbolKind::ManifoldParam)
            )
        }
        Expr::Binary(b) => {
            expr_touches_manifold(&b.lhs, symbols, tainted)
                || expr_touches_manifold(&b.rhs, symbols, tainted)
        }
        Expr::Unary(u) => expr_touches_manifold(&u.operand, symbols, tainted),
        Expr::Paren(p) => expr_touches_manifold(p, symbols, tainted),
        Expr::MethodCall(c) => {
            expr_touches_manifold(&c.receiver, symbols, tainted)
                || c.args
                    .iter()
                    .any(|a| expr_touches_manifold(a, symbols, tainted))
        }
        Expr::Call(c) => c
            .args
            .iter()
            .any(|a| expr_touches_manifold(a, symbols, tainted)),
        Expr::Tuple(t) => t
            .elems
            .iter()
            .any(|e| expr_touches_manifold(e, symbols, tainted)),
        Expr::Block(b) => {
            b.stmts.iter().any(|s| match s {
                Stmt::Let(l) => expr_touches_manifold(&l.init, symbols, tainted),
                Stmt::Expr(e) => expr_touches_manifold(e, symbols, tainted),
            }) || b
                .expr
                .as_ref()
                .is_some_and(|e| expr_touches_manifold(e, symbols, tainted))
        }
        Expr::Literal(_) | Expr::Verbatim(_) => false,
    }
}

fn find_at_manifold_params_inner(
    expr: &Expr,
    symbols: &crate::symbol::SymbolTable,
    result: &mut HashSet<String>,
) {
    match expr {
        Expr::MethodCall(call) => {
            let method_str = call.method.to_string();
            // Check if this is a `.at()` call on a manifold param
            if method_str == "at" {
                if let Expr::Ident(ident_expr) = &*call.receiver {
                    let name_str = ident_expr.name.to_string();
                    if let Some(symbol) = symbols.lookup(&name_str) {
                        if matches!(symbol.kind, SymbolKind::ManifoldParam) {
                            result.insert(name_str);
                        }
                    }
                }
            }
            // Recurse into receiver and args
            find_at_manifold_params_inner(&call.receiver, symbols, result);
            for arg in &call.args {
                find_at_manifold_params_inner(arg, symbols, result);
            }
        }
        Expr::Binary(binary) => {
            find_at_manifold_params_inner(&binary.lhs, symbols, result);
            find_at_manifold_params_inner(&binary.rhs, symbols, result);
        }
        Expr::Unary(unary) => {
            find_at_manifold_params_inner(&unary.operand, symbols, result);
        }
        Expr::Call(call) => {
            for arg in &call.args {
                find_at_manifold_params_inner(arg, symbols, result);
            }
        }
        Expr::Block(block) => {
            for stmt in &block.stmts {
                match stmt {
                    Stmt::Let(let_stmt) => {
                        find_at_manifold_params_inner(&let_stmt.init, symbols, result);
                    }
                    Stmt::Expr(expr) => {
                        find_at_manifold_params_inner(expr, symbols, result);
                    }
                }
            }
            if let Some(expr) = &block.expr {
                find_at_manifold_params_inner(expr, symbols, result);
            }
        }
        Expr::Paren(inner) => {
            find_at_manifold_params_inner(inner, symbols, result);
        }
        Expr::Tuple(tuple) => {
            for elem in &tuple.elems {
                find_at_manifold_params_inner(elem, symbols, result);
            }
        }
        // Leaf nodes - no recursion needed
        Expr::Ident(_) | Expr::Literal(_) | Expr::Verbatim(_) => {}
    }
}

/// The code emitter state.
pub struct CodeEmitter<'a> {
    analyzed: &'a AnalyzedKernel,
    /// Maps parameter names to their (ArrayID, Index) location.
    /// ArrayID: 0=A0 (Scalars), 1=A1 (M0), 2=A2 (M1), etc.
    param_indices: HashMap<String, (u8, usize)>,
    /// Maps manifold parameter names to their generic type index (M0, M1, ...).
    manifold_indices: HashMap<String, usize>,
    /// Collected literals from annotation pass (for Let bindings in Jet mode).
    collected_literals: Vec<crate::annotate::CollectedLiteral>,
    /// Locals whose bound value involves a manifold param (non-Copy);
    /// every reference to these is emitted as a clone, never a move.
    manifold_locals: HashSet<String>,
}

impl<'a> CodeEmitter<'a> {
    pub fn new(analyzed: &'a AnalyzedKernel) -> Self {
        // Separate params into scalars and manifolds
        let mut scalar_params = Vec::with_capacity(analyzed.def.params.len());
        let mut manifold_params = Vec::with_capacity(analyzed.def.params.len());

        for param in &analyzed.def.params {
            match param.kind {
                ParamKind::Scalar(_) => scalar_params.push(param),
                ParamKind::Manifold => manifold_params.push(param),
            }
        }

        let mut param_indices = HashMap::new();
        let mut manifold_indices = HashMap::new();

        // Scalars go into A0 (Array 0)
        // Indices are assigned in reverse order of declaration (deepest first)
        // Literals will be added later, effectively extending this array
        let n_scalars = scalar_params.len();
        for (i, param) in scalar_params.iter().enumerate() {
            // Index: n-1-i (last param gets 0)
            param_indices.insert(param.name.to_string(), (0u8, n_scalars - 1 - i));
        }

        // Manifolds go into subsequent arrays (A1, A2, ...)
        // Each manifold gets its own array of size 1
        for (i, param) in manifold_params.iter().enumerate() {
            let array_id = (i + 1) as u8; // A1, A2, ...
            param_indices.insert(param.name.to_string(), (array_id, 0));
            manifold_indices.insert(param.name.to_string(), i);
        }

        // The body here is post-optimization, so this also covers optimizer-
        // introduced `__N` bindings whose value chains back to a manifold param.
        let manifold_locals = find_manifold_tainted_locals(&analyzed.def.body, &analyzed.symbols);

        CodeEmitter {
            analyzed,
            param_indices,
            manifold_indices,
            collected_literals: Vec::new(),
            manifold_locals,
        }
    }

    /// Emit the complete kernel definition.
    pub fn emit_kernel(&mut self) -> TokenStream {
        // Dispatch based on whether this is a named or anonymous kernel
        if let Some(ref decl) = self.analyzed.def.struct_decl {
            self.emit_named_kernel(decl.visibility.clone(), decl.name.clone())
        } else {
            self.emit_closure_kernel()
        }
    }

    /// Emit an anonymous kernel as a closure returning WithContext.
    ///
    /// This allows natural environment capture via Rust's closure semantics.
    ///
    /// Output pattern:
    /// ```ignore
    /// move |cx: f32, cy: f32| {
    ///     use ::pixelflow_core::{X, Y, Z, W, WithContext, CtxVar, ...};
    ///     let __expr = { X - CtxVar::<N0>::new() };
    ///     WithContext::new((cx, cy), __expr)
    /// }
    /// ```
    fn emit_closure_kernel(&mut self) -> TokenStream {
        let params = &self.analyzed.def.params;
        let std_imports = standard_imports();

        // For closure kernels, use the return type if specified.
        // The return type annotation is the user's explicit declaration of scalar type.
        // This enables `kernel!(|h: f32| -> Jet3 { h / Y })` to work correctly.
        // Default to Field if no return type is specified.
        let scalar_type = match &self.analyzed.def.return_ty {
            Some(ty) => quote! { #ty },
            None => quote! { ::pixelflow_core::Field },
        };

        // Run annotation pass to collect literals and assign Var indices
        let annotation_ctx = AnnotationCtx::new();
        let (annotated_body, _, collected_literals) =
            annotate(&self.analyzed.def.body, annotation_ctx);
        self.collected_literals = collected_literals;

        // Always adjust param indices to account for literals in context
        // Literals go into A0 (Scalars), so only adjust scalar params (ArrayID 0)
        let literal_count = self.collected_literals.len();
        for (array_id, idx) in self.param_indices.values_mut() {
            if *array_id == 0 {
                *idx += literal_count;
            }
        }

        // Transform and emit the body as a ZST expression
        let body = self.emit_annotated_expr(&annotated_body);

        // Generate the Peano type imports needed
        let peano_imports = self.emit_peano_imports();

        // Generate closure parameters with types
        let closure_params: Vec<TokenStream> = params
            .iter()
            .map(|p| {
                let name = &p.name;
                match &p.kind {
                    ParamKind::Scalar(ty) => quote! { #name: #ty },
                    ParamKind::Manifold => quote! { #name },
                }
            })
            .collect();

        // Determine if we need special handling for manifold params
        let manifold_count = self.manifold_indices.len();
        let has_scalar_params = params
            .iter()
            .any(|p| matches!(p.kind, ParamKind::Scalar(_)));
        let has_literals = !self.collected_literals.is_empty();

        // Single manifold + scalars/literals: use ManifoldBind (manifold handled separately)
        // Multiple manifolds: use Computed with pre-eval (fallback)
        // No manifolds: use plain WithContext
        let use_manifold_bind = manifold_count == 1 && (has_scalar_params || has_literals);
        let use_computed_fallback = manifold_count > 1;

        // Pre-evaluate manifolds to get concrete scalar values (for Computed fallback)
        let mut pre_eval_stmts = Vec::new();
        if use_computed_fallback {
            for param in params.iter() {
                if matches!(param.kind, ParamKind::Manifold) {
                    let name = &param.name;
                    let eval_name = format_ident!("__eval_{}", name);
                    pre_eval_stmts.push(quote! {
                        let #eval_name = #name.eval(__p);
                    });
                }
            }
        }

        // Group values into arrays by ArrayID
        // A0: Scalars (literals + scalar params)
        // A1..AN: Manifold params (only for non-ManifoldBind cases)
        let mut arrays: Vec<Vec<(usize, TokenStream)>> = vec![Vec::new(); 16]; // Max 16 arrays

        // 1. Add Literals to A0 - use scalar_type for proper type lifting
        for c in &self.collected_literals {
            let lit = &c.lit;
            let val = quote! { #scalar_type::from(#lit) };
            arrays[0].push((c.index, val));
        }

        // 2. Add Parameters to appropriate arrays
        for param in params.iter() {
            let name = &param.name;
            let (array_id, idx) = self.param_indices[&name.to_string()];

            let param_value = match &param.kind {
                ParamKind::Manifold => {
                    if use_manifold_bind {
                        // ManifoldBind handles this param - don't add to context
                        continue;
                    } else if use_computed_fallback {
                        // Computed fallback: use pre-evaluated value
                        let eval_name = format_ident!("__eval_{}", name);
                        quote! { #eval_name }
                    } else {
                        // Single manifold param with no scalars - can store directly
                        quote! { #name }
                    }
                }
                // Scalar params use scalar_type for proper type lifting (e.g., Jet3::from)
                ParamKind::Scalar(_) => quote! { #scalar_type::from(#name) },
            };

            if (array_id as usize) < arrays.len() {
                arrays[array_id as usize].push((idx, param_value));
            }
        }

        // Build the context tuple
        // Empty context is (), single array is ([...],), multi-array is ([...], [...])
        let raw_arrays: Vec<TokenStream> = arrays
            .iter()
            .filter(|a| !a.is_empty())
            .map(|vals| {
                let sorted = sort_by_index(vals.clone());
                quote! { [#(#sorted),*] }
            })
            .collect();

        let context_tuple = if raw_arrays.is_empty() {
            quote! { () }
        } else if raw_arrays.len() == 1 {
            let a = &raw_arrays[0];
            quote! { (#a,) }
        } else {
            quote! { (#(#raw_arrays),*) }
        };

        // Choose code generation strategy based on param composition
        if use_manifold_bind {
            // Single manifold + scalars/literals: use ManifoldBind
            // ManifoldBind carries the manifold type in its signature, helping type inference
            let manifold_name = params
                .iter()
                .find(|p| matches!(p.kind, ParamKind::Manifold))
                .map(|p| &p.name)
                .expect("use_manifold_bind requires exactly one manifold param");

            quote! {
                move |#(#closure_params),*| {
                    #std_imports
                    #peano_imports

                    // The body embeds the manifold by value (`ContextFree`), and
                    // `ManifoldBind` needs it too. Clone for the bind so both uses
                    // are valid for non-`Copy` (JIT-backed) inner manifolds;
                    // combinator manifolds are ZSTs, so the clone is free. UFCS
                    // (not `.clone()`) so it type-checks against the still-
                    // unresolved closure-parameter type.
                    let __manifold_bind = ::core::clone::Clone::clone(&#manifold_name);
                    let __expr = { #body };
                    let __inner_body = WithContext::new(#context_tuple, __expr);
                    ManifoldBind::new(__manifold_bind, __inner_body)
                }
            }
        } else if use_computed_fallback {
            // Multiple manifolds: use Computed with pre-eval
            // Type inference may still fail for some cases
            quote! {
                move |#(#closure_params),*| {
                    #std_imports
                    #peano_imports

                    let __expr = { #body };
                    // Pre-evaluate manifolds, then build context with evaluated values
                    Computed::new(move |__p| {
                        #(#pre_eval_stmts)*
                        WithContext::new(#context_tuple, __expr).eval(__p)
                    })
                }
            }
        } else {
            // No manifolds, or single manifold without scalars/literals
            quote! {
                move |#(#closure_params),*| {
                    #std_imports
                    #peano_imports

                    let __expr = { #body };
                    WithContext::new(#context_tuple, __expr)
                }
            }
        }
    }

    /// Emit a named kernel as a struct with Manifold impl.
    ///
    /// This creates a user-named struct that can be used in struct fields.
    /// Uses the StructEmitter builder to consolidate all 8 code paths.
    fn emit_named_kernel(&mut self, visibility: syn::Visibility, name: syn::Ident) -> TokenStream {
        let params = &self.analyzed.def.params;
        let std_imports = standard_imports();

        // Count manifold parameters for generic type generation
        let manifold_count = self.manifold_indices.len();

        // Generate generic type parameter names (M0, M1, ...)
        let generic_names: Vec<syn::Ident> = (0..manifold_count)
            .map(|i| format_ident!("M{}", i))
            .collect();

        // Generate struct fields with pub visibility
        let struct_fields: Vec<TokenStream> = params
            .iter()
            .map(|p| {
                let field_name = &p.name;
                match &p.kind {
                    ParamKind::Scalar(ty) => quote! { pub #field_name: #ty },
                    ParamKind::Manifold => {
                        let idx = self.manifold_indices[&field_name.to_string()];
                        let generic_name = &generic_names[idx];
                        quote! { pub #field_name: #generic_name }
                    }
                }
            })
            .collect();

        // Generate struct field names for construction
        let field_names: Vec<_> = params.iter().map(|p| p.name.clone()).collect();

        // Generate constructor parameters
        let constructor_params: Vec<TokenStream> = params
            .iter()
            .map(|p| {
                let field_name = &p.name;
                match &p.kind {
                    ParamKind::Scalar(ty) => quote! { #field_name: #ty },
                    ParamKind::Manifold => {
                        let idx = self.manifold_indices[&field_name.to_string()];
                        let generic_name = &generic_names[idx];
                        quote! { #field_name: #generic_name }
                    }
                }
            })
            .collect();

        // Run annotation pass to collect literals and assign Var indices
        let annotation_ctx = AnnotationCtx::new();
        let (annotated_body, _, collected_literals) =
            annotate(&self.analyzed.def.body, annotation_ctx);
        self.collected_literals = collected_literals;

        // Always adjust param indices to account for literals in context
        // Literals go into A0 (Scalars), so only adjust scalar params (ArrayID 0)
        let literal_count = self.collected_literals.len();
        for (array_id, idx) in self.param_indices.values_mut() {
            if *array_id == 0 {
                *idx += literal_count;
            }
        }

        // Transform and emit the body as a ZST expression
        let body = self.emit_annotated_expr(&annotated_body);

        // Generate the Peano type imports needed
        let peano_imports = self.emit_peano_imports();

        // Determine output type, domain type, and scalar type
        let (output_type, domain_type) = match (
            &self.analyzed.def.domain_ty,
            &self.analyzed.def.return_ty,
        ) {
            (Some(domain), Some(output)) => {
                let _type_str = quote! { #domain }.to_string();
                // panic!("DEBUG: domain type is '{}'", type_str);
                let domain_tokens = if let syn::Type::Tuple(_) = domain {
                    quote! { #domain }
                } else {
                    quote! { (#domain, #domain, #domain, #domain) }
                };
                (quote! { #output }, domain_tokens)
            }
            (None, Some(ty)) => (quote! { #ty }, quote! { (#ty, #ty, #ty, #ty) }),
            (None, None) | (Some(_), None) => (
                quote! { ::pixelflow_core::Field },
                quote! { (::pixelflow_core::Field, ::pixelflow_core::Field, ::pixelflow_core::Field, ::pixelflow_core::Field) },
            ),
        };

        // Use the Spatial trait to determine the scalar type of the domain
        let scalar_type = quote! { <#domain_type as ::pixelflow_core::Spatial>::Coord };

        // Determine derives and domain config based on parameter configuration
        let has_fixed_domain =
            self.analyzed.def.domain_ty.is_some() || self.analyzed.def.return_ty.is_some();

        // Scan for manifold params used with .at() - these need ManifoldExt bound
        // This must be computed BEFORE emit_unified_binding, which needs to skip pre-eval for these params
        let at_manifold_params = find_at_manifold_params(&annotated_body, &self.analyzed.symbols);

        // Scan for manifold params that have derivative operations (DX, DY, DZ, V) applied
        // These need `Output: HasDerivatives` trait bound
        let derivative_manifold_params =
            find_derivative_manifold_params(&annotated_body, &self.analyzed.symbols);

        // Generate the binding (passing at_manifold_params to skip pre-eval for .at() params)
        // Always use Field for literals - it's the base scalar type that all others can be constructed from
        let field_type = quote! { ::pixelflow_core::Field };
        let (manifold_eval_stmts, at_binding) =
            self.emit_unified_binding(&at_manifold_params, &field_type);

        // Build trait bounds for generic structs with fixed domain
        // All manifold params get domain-based bounds. The .at() combinator handles
        // coordinate type transformation internally.
        let mut trait_bounds: Vec<TokenStream> = params
                .iter()
                .filter_map(|p| {
                    if matches!(p.kind, ParamKind::Manifold) {
                        let name_str = p.name.to_string();
                        let idx = self.manifold_indices[&name_str];
                        let g = &generic_names[idx];

                        if derivative_manifold_params.contains(&name_str) {
                            // Derivative params: used in coordinate expressions (DX, DY, DZ, V)
                            // Output must be scalar_type (e.g., Jet3) for At combinator
                            Some(quote! { #g: ::pixelflow_core::Manifold<#domain_type, Output = #scalar_type> + ::pixelflow_core::ManifoldExpr + Send + Sync })
                        } else {
                            // Non-derivative params: used as Select branches, kernel output
                            // Output must be output_type (e.g., Field)
                            Some(quote! { #g: ::pixelflow_core::Manifold<#domain_type, Output = #output_type> + ::pixelflow_core::ManifoldExpr + Send + Sync })
                        }
                    } else {
                        None
                    }
                })
                .collect();

        // Add trait bounds for derivative params' outputs
        for name_str in &derivative_manifold_params {
            if let Some(&idx) = self.manifold_indices.get(name_str) {
                let g = &generic_names[idx];
                // Derivative extraction support
                trait_bounds.push(quote! {
                        <#g as ::pixelflow_core::Manifold<#domain_type>>::Output: ::pixelflow_core::ops::derivative::HasDerivatives + ::pixelflow_core::ops::derivative::HasDz
                    });
                // Comparison support: output must convert to Field for <, >, etc.
                trait_bounds.push(quote! {
                        <#g as ::pixelflow_core::Manifold<#domain_type>>::Output: Into<::pixelflow_core::Field> + Copy
                    });
            }
        }

        // Note: Manifold params used with .at() don't get additional trait bounds here.
        // The At combinator handles domain transformation. If a param is used ONLY with
        // .at() (like color_cube in ColorSky), its domain depends on the coordinate
        // expression types, which vary (Field for V(X), Jet3 for X*t, etc.).

        // Determine derives:
        // - Unit struct or single scalar param → CloneCopy
        // - Single manifold param → Clone (Copy handled conditionally in StructEmitter)
        // - Multiple params → Clone only (multi-field structs shouldn't derive Copy)
        let derives = if params.is_empty() || (manifold_count == 0 && params.len() == 1) {
            Derives::CloneCopy
        } else {
            Derives::Clone
        };

        // Build the emitter
        let mut emitter = StructEmitter::new(visibility, name)
            .with_generics(generic_names.clone())
            .with_derives(derives)
            .with_fields(struct_fields, field_names, constructor_params);

        // Configure domain
        if has_fixed_domain || manifold_count == 0 {
            // Fixed domain: all scalar params, or explicit domain/return type
            emitter = emitter.with_fixed_domain(domain_type, output_type, trait_bounds);
        }
        // else: generic domain (default in StructEmitter)

        // Configure eval body
        emitter = emitter.with_eval_body(
            std_imports,
            peano_imports,
            manifold_eval_stmts,
            body,
            at_binding,
        );

        emitter.build()
    }

    /// Emit imports for array-based context system.
    ///
    /// With the array-based approach, we use const generics instead of Peano numbers.
    /// The A0, A1, A2, A3 markers are already imported in standard_imports.
    fn emit_peano_imports(&self) -> TokenStream {
        // No additional imports needed - A0, A1, A2, A3 are in standard_imports
        // Const generic indices are written directly as literals
        quote! {}
    }

    /// Emit unified WithContext/CtxVar binding for params (and Let for literals).
    ///
    /// `at_manifold_params` contains names of manifold params that use `.at()`.
    /// These are NOT pre-evaluated - they're accessed via `(&self.field).at(...)` lazily.
    /// `scalar_type` is the type used for scalar/literal conversion (e.g., `Jet3::from` instead of `Field::from`).
    /// This should be the domain's scalar type (from `Spatial::Coord`), not the output type.
    fn emit_unified_binding(
        &self,
        _at_manifold_params: &HashSet<String>,
        scalar_type: &TokenStream,
    ) -> (TokenStream, TokenStream) {
        let params = &self.analyzed.def.params;

        if params.is_empty() && self.collected_literals.is_empty() {
            return (quote! {}, quote! { __expr.eval(__p) });
        }

        // Determine if we need to pre-evaluate manifold params
        // Always pre-evaluate manifolds when:
        // 1. There are multiple manifolds (need consistent evaluation order)
        // 2. There are scalar params mixed with manifolds
        let manifold_count = self.manifold_indices.len();
        let has_scalar_params = params
            .iter()
            .any(|p| matches!(p.kind, ParamKind::Scalar(_)));
        let _needs_pre_eval = manifold_count > 0 && (manifold_count > 1 || has_scalar_params);

        // NOTE: Manifold params are NO LONGER pre-evaluated.
        // They're accessed directly via (&self.name) in the expression tree.
        // This allows Rust to infer output types from usage context.
        // Only scalar params go into the context arrays now.
        let pre_eval_stmts = Vec::<TokenStream>::new();

        // Group values into arrays by ArrayID
        // A0: Scalars (literals + scalar params)
        // A1..AN: Manifold params
        let mut arrays: Vec<Vec<(usize, TokenStream)>> = vec![Vec::new(); 16]; // Max 16 arrays

        // 1. Add Literals to A0 - use scalar_type for proper type lifting
        for c in &self.collected_literals {
            let lit = &c.lit;
            let val = quote! { #scalar_type::from(#lit) };
            arrays[0].push((c.index, val));
        }

        // 2. Add scalar parameters to arrays
        // NOTE: Manifold params are NO LONGER added to context arrays.
        // They're accessed directly via (&self.name) in the expression tree.
        for param in params.iter() {
            let name = &param.name;
            let name_str = name.to_string();

            // Skip manifold params - they're accessed directly
            if matches!(param.kind, ParamKind::Manifold) {
                continue;
            }

            let (array_id, idx) = self.param_indices[&name_str];
            // Scalar params use scalar_type for proper type lifting (e.g., Jet3::from)
            let param_value = quote! { #scalar_type::from(self.#name) };

            if (array_id as usize) < arrays.len() {
                arrays[array_id as usize].push((idx, param_value));
            }
        }

        // Build the tuple of arrays
        let mut array_exprs = Vec::new();
        for array_values in arrays.iter() {
            if !array_values.is_empty() {
                // Sort by index and extract values
                let sorted_values = sort_by_index(array_values.clone());
                // Build array: ([val0, val1],)
                array_exprs.push(build_array(&sorted_values));
            }
        }

        // Generate the WithContext call
        // Note: We need a tuple of arrays. If there's only one array, we need (array,)

        let context_tuple = if array_exprs.is_empty() {
            quote! { () }
        } else {
            // Unwrap the single-element tuples from build_array to get raw arrays
            // build_array returns `([a,b],)` -> we want `[a,b]`
            // This relies on knowing build_array implementation details.
            // Let's modify the logic to construct arrays directly here.

            let raw_arrays: Vec<TokenStream> = arrays
                .iter()
                .filter(|a| !a.is_empty())
                .map(|vals| {
                    let sorted = sort_by_index(vals.clone());
                    quote! { [#(#sorted),*] }
                })
                .collect();

            if raw_arrays.len() == 1 {
                let a = &raw_arrays[0];
                quote! { (#a,) }
            } else {
                quote! { (#(#raw_arrays),*) }
            }
        };

        // Wrap in Let bindings for literals?
        // NO - literals are now in A0! We don't use nested Let bindings anymore.
        // We use the array context exclusively.

        let at_binding = quote! {
            WithContext::new(#context_tuple, __expr).eval(__p)
        };

        let stmts = if pre_eval_stmts.is_empty() {
            quote! {}
        } else {
            quote! { #(#pre_eval_stmts)* }
        };
        (stmts, at_binding)
    }

    /// Emit code for an annotated expression (pure, no mutation).
    ///
    /// Literals with var_index become Var<N> references.
    /// This is the clean functional version that works with the annotation pass.
    pub fn emit_annotated_expr(&self, expr: &Expr) -> TokenStream {
        match expr {
            Expr::Ident(ident_expr) => {
                let name = &ident_expr.name;
                let name_str = name.to_string();

                // Locals bound to manifold-tapping expressions (user lets or
                // optimizer-introduced `__N` bindings) are non-Copy: the
                // extracted form may reference them more than once, so every
                // use clones. UFCS keeps this valid for not-yet-inferred
                // types; for ZST expression values the clone is free.
                if self.manifold_locals.contains(&name_str) {
                    return quote! { ::core::clone::Clone::clone(&#name) };
                }

                match self.analyzed.symbols.lookup(&name_str) {
                    Some(symbol) => match symbol.kind {
                        SymbolKind::Intrinsic => {
                            // Intrinsics (X, Y, Z, W) emitted as-is
                            quote! { #name }
                        }
                        SymbolKind::ManifoldParam => {
                            // Manifold params: wrap in ContextFree to lift from Manifold<P>
                            // to Manifold<(Ctx, P)> for use with context variables
                            if self.analyzed.def.struct_decl.is_some() {
                                // Named kernel: emit ContextFree(&self.field_name)
                                quote! { ContextFree(&self.#name) }
                            } else {
                                // Anonymous closure: emit ContextFree(name)
                                quote! { ContextFree(#name) }
                            }
                        }
                        SymbolKind::Parameter => {
                            // Scalar parameters use CtxVar::<Ax, INDEX>::new()
                            if let Some(&(array_id, idx)) = self.param_indices.get(&name_str) {
                                let marker = match array_id {
                                    0 => quote! { A0 },
                                    1 => quote! { A1 },
                                    2 => quote! { A2 },
                                    3 => quote! { A3 },
                                    4 => quote! { A4 },
                                    5 => quote! { A5 },
                                    6 => quote! { A6 },
                                    7 => quote! { A7 },
                                    8 => quote! { A8 },
                                    9 => quote! { A9 },
                                    10 => quote! { A10 },
                                    11 => quote! { A11 },
                                    12 => quote! { A12 },
                                    13 => quote! { A13 },
                                    14 => quote! { A14 },
                                    15 => quote! { A15 },
                                    _ => panic!("Too many context arrays (max 16 supported)"),
                                };
                                quote! { CtxVar::<#marker, #idx>::new() }
                            } else {
                                quote! { #name }
                            }
                        }
                        SymbolKind::Local => {
                            // Locals emitted as-is
                            quote! { #name }
                        }
                    },
                    None => {
                        // Unknown - emit as-is
                        quote! { #name }
                    }
                }
            }

            Expr::Literal(lit) => {
                // Always emit literals as CtxVar references for ZST preservation
                // This ensures expression trees remain Copy (composed of ZST nodes)
                if let Some(var_idx) = lit.var_index {
                    // Literals go at indices in the context array
                    // Use array-based indexing: CtxVar::<A0, INDEX>::new()
                    let collected = self
                        .collected_literals
                        .iter()
                        .find(|c| c.index == var_idx)
                        .unwrap_or_else(|| {
                            panic!(
                                "literal with var_index {var_idx} was never collected \
                                 by the annotation pass — annotation and emission are \
                                 out of sync"
                            )
                        });
                    match collected.space {
                        // Domain-space: the A0 entry already has the domain
                        // scalar type; reference it directly.
                        crate::annotate::LiteralSpace::Domain => {
                            quote! { CtxVar::<A0, #var_idx>::new() }
                        }
                        // Field-space (post-V/DX/DY math): re-project the
                        // domain-lifted A0 entry back to Field, exactly like
                        // user-written `V(literal)`. Keeps the tree ZST while
                        // making optimizer-introduced literals type-stable.
                        crate::annotate::LiteralSpace::Projected => {
                            quote! { V(CtxVar::<A0, #var_idx>::new()) }
                        }
                    }
                } else {
                    // Fallback: no var_index assigned (shouldn't happen after annotation)
                    // Emit as Field::from
                    let l = &lit.lit;
                    quote! { ::pixelflow_core::Field::from(#l) }
                }
            }

            Expr::Binary(binary) => {
                let lhs = self.emit_annotated_expr(&binary.lhs);
                let rhs = self.emit_annotated_expr(&binary.rhs);

                // Always wrap binary expressions in parentheses to preserve precedence.
                // This prevents issues like `(X + val).sqrt()` becoming `X + val.sqrt()`
                // when the binary expression is used as a method receiver.
                match binary.op {
                    BinaryOp::Add => quote! { (#lhs + #rhs) },
                    BinaryOp::Sub => quote! { (#lhs - #rhs) },
                    BinaryOp::Mul => quote! { (#lhs * #rhs) },
                    BinaryOp::Div => quote! { (#lhs / #rhs) },
                    BinaryOp::Rem => quote! { (#lhs % #rhs) },
                    BinaryOp::Lt => quote! { #lhs.lt(#rhs) },
                    BinaryOp::Le => quote! { #lhs.le(#rhs) },
                    BinaryOp::Gt => quote! { #lhs.gt(#rhs) },
                    BinaryOp::Ge => quote! { #lhs.ge(#rhs) },
                    BinaryOp::Eq => quote! { #lhs.eq(#rhs) },
                    BinaryOp::Ne => quote! { #lhs.ne(#rhs) },
                    BinaryOp::BitAnd => quote! { (#lhs & #rhs) },
                    BinaryOp::BitOr => quote! { (#lhs | #rhs) },
                }
            }

            Expr::Unary(unary) => {
                let operand = self.emit_annotated_expr(&unary.operand);
                match unary.op {
                    // Parentheses are required because method call binds tighter than binary operators.
                    // Without parens, `a - b.neg()` is parsed as `a - (b.neg())`, not `(a - b).neg()`.
                    UnaryOp::Neg => quote! { (#operand).neg() },
                    UnaryOp::Not => quote! { !(#operand) },
                }
            }

            Expr::MethodCall(call) => {
                let method = &call.method;
                let method_str = method.to_string();

                // Special case: `.at()` on manifold params in named kernels
                // Use (&self.field).at(...) to borrow the manifold and call .at() on the reference.
                // This works because:
                // 1. &M: ManifoldExpr (blanket impl) gives ManifoldExt and thus .at()
                // 2. &M: Manifold<P> (blanket impl) allows At<..., &M> to be a valid Manifold
                // 3. At's generalized impl accepts coords with Into<I> outputs, so mixed
                //    Field/Jet3 coords all convert to Field via Into<Field>
                let is_named_kernel = self.analyzed.def.struct_decl.is_some();
                if is_named_kernel && method_str == "at" {
                    if let Expr::Ident(ident_expr) = &*call.receiver {
                        let name = &ident_expr.name;
                        let name_str = name.to_string();
                        if let Some(symbol) = self.analyzed.symbols.lookup(&name_str) {
                            if matches!(symbol.kind, SymbolKind::ManifoldParam) {
                                // Use emit_at_coord_arg for .at() arguments to handle literals properly
                                let args: Vec<TokenStream> = call
                                    .args
                                    .iter()
                                    .map(|a| self.emit_at_coord_arg(a))
                                    .collect();
                                // Emit (&self.field_name).at(...) - borrow and call .at() on reference
                                return quote! { (&self.#name).at(#(#args),*) };
                            }
                        }
                    }
                }

                // Normal case: emit receiver.method(args) using standard emit
                let args: Vec<TokenStream> = call
                    .args
                    .iter()
                    .map(|a| self.emit_annotated_expr(a))
                    .collect();
                let receiver = self.emit_annotated_expr(&call.receiver);
                if args.is_empty() {
                    quote! { #receiver.#method() }
                } else {
                    quote! { #receiver.#method(#(#args),*) }
                }
            }

            Expr::Call(call) => {
                // Free function call: V(m), DX(expr), etc.
                // Emit with transformed arguments (manifold params become Var<N>)
                let func = &call.func;
                let args: Vec<TokenStream> = call
                    .args
                    .iter()
                    .map(|a| self.emit_annotated_expr(a))
                    .collect();

                if args.is_empty() {
                    quote! { #func() }
                } else {
                    quote! { #func(#(#args),*) }
                }
            }

            Expr::Block(block) => {
                let stmts: Vec<TokenStream> = block
                    .stmts
                    .iter()
                    .map(|s| self.emit_annotated_stmt(s))
                    .collect();

                let final_expr = block.expr.as_ref().map(|e| self.emit_annotated_expr(e));

                match final_expr {
                    Some(expr) => quote! {
                        {
                            #(#stmts)*
                            #expr
                        }
                    },
                    None => quote! {
                        {
                            #(#stmts)*
                        }
                    },
                }
            }

            Expr::Paren(inner) => {
                let inner_code = self.emit_annotated_expr(inner);
                quote! { (#inner_code) }
            }

            Expr::Tuple(tuple) => {
                let elems: Vec<TokenStream> = tuple
                    .elems
                    .iter()
                    .map(|e| self.emit_annotated_expr(e))
                    .collect();
                quote! { (#(#elems),*) }
            }

            Expr::Verbatim(syn_expr) => syn_expr.to_token_stream(),
        }
    }

    fn emit_annotated_stmt(&self, stmt: &Stmt) -> TokenStream {
        match stmt {
            Stmt::Let(let_stmt) => {
                let name = &let_stmt.name;
                let init = self.emit_annotated_expr(&let_stmt.init);

                match &let_stmt.ty {
                    Some(ty) => quote! { let #name: #ty = #init; },
                    None => quote! { let #name = #init; },
                }
            }
            Stmt::Expr(expr) => {
                let code = self.emit_annotated_expr(expr);
                quote! { #code; }
            }
        }
    }

    /// Emit a coordinate argument for `.at()` on manifold params.
    ///
    /// For literals, wrap in ContextFree so they work with context-extended domains.
    /// f32 implements Manifold<P, Output = Field>, and ContextFree lifts that to
    /// Manifold<(Ctx, P)> by ignoring the context.
    /// For other expressions, use normal emission.
    fn emit_at_coord_arg(&self, expr: &Expr) -> TokenStream {
        match expr {
            Expr::Literal(lit_expr) => {
                // Wrap literal in ContextFree so it works with context-extended domains
                // f32: Manifold<P, Output = Field> for P: Send + Sync
                // ContextFree<f32>: Manifold<(Ctx, P)> by ignoring context
                let lit = &lit_expr.lit;
                match lit {
                    syn::Lit::Float(f) => {
                        let value = f.base10_parse::<f64>().unwrap_or(0.0);
                        quote! { ::pixelflow_core::combinators::ContextFree(#value as f32) }
                    }
                    syn::Lit::Int(i) => {
                        let value = i.base10_parse::<i64>().unwrap_or(0);
                        quote! { ::pixelflow_core::combinators::ContextFree(#value as f32) }
                    }
                    _ => self.emit_annotated_expr(expr),
                }
            }
            _ => self.emit_annotated_expr(expr),
        }
    }
}
