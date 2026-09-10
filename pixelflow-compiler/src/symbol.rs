//! # Symbol Table
//!
//! The symbol table tracks all identifiers in scope during compilation.
//!
//! ## Symbol Classes
//!
//! PixelFlow has a two-layer symbol table that mirrors the contramap pattern:
//!
//! | Class      | Binding Time        | Arena Representation | Example |
//! |------------|---------------------|----------------------|---------|
//! | Intrinsic  | Collapse time       | `Var(0..4)`          | X, Y    |
//! | Parameter  | Construction time   | `Param(i)`           | cx, r   |
//! | Local      | Expression scope    | A shared DAG handle  | dx, dy  |
//!
//! ## Intrinsic Coordinates
//!
//! The intrinsic coordinates (X, Y, Z, W) are special:
//! - They become `Var(0..4)` arena nodes
//! - They are always in scope (global namespace)
//!
//! ## Parameter Symbols
//!
//! Parameters declared in the closure syntax become the builder closure's
//! arguments: `|cx: f32, cy: f32|` produces `move |cx: f32, cy: f32| -> Kernel`,
//! and each reference in the body is a `Param(i)` arena node the builder
//! substitutes with the argument.

use proc_macro2::Span;
use std::collections::HashMap;
use syn::{Ident, Type};

/// The binding class of a symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolKind {
    /// Intrinsic coordinate variable (X, Y, Z, W).
    /// Bound at evaluation time via `eval_raw` parameters.
    Intrinsic,

    /// Captured scalar parameter from closure syntax (e.g., `r: f32`).
    /// Bound at construction time, accessed via `self.name`.
    Parameter,

    /// Local variable introduced by `let`.
    /// Scoped to the containing block.
    Local,
}

/// A symbol in the symbol table.
#[derive(Debug, Clone)]
pub struct Symbol {
    /// The identifier name.
    #[allow(dead_code)]
    pub name: Ident,
    /// What kind of symbol this is.
    pub kind: SymbolKind,
    /// The type (if known). Intrinsics have implicit types.
    #[allow(dead_code)]
    pub ty: Option<Type>,
    /// Where the symbol was defined.
    #[allow(dead_code)]
    pub span: Span,
}

/// The symbol table for a kernel compilation.
#[derive(Debug, Clone)]
pub struct SymbolTable {
    /// All symbols indexed by name.
    symbols: HashMap<String, Symbol>,
    /// Scopes for local variable shadowing (future use).
    scope_stack: Vec<Vec<String>>,
}

impl SymbolTable {
    /// Create a new symbol table with intrinsic coordinates pre-populated.
    pub fn new() -> Self {
        let mut table = SymbolTable {
            symbols: HashMap::new(),
            scope_stack: vec![Vec::new()],
        };

        // Register intrinsic coordinate variables. A lattice has two axes;
        // `Z` and `W` are refused by name in sema, with the message that
        // points at uniforms.
        for name in ["X", "Y"] {
            table.symbols.insert(
                name.to_string(),
                Symbol {
                    name: Ident::new(name, Span::call_site()),
                    kind: SymbolKind::Intrinsic,
                    ty: None, // Intrinsics are polymorphic over Numeric
                    span: Span::call_site(),
                },
            );
        }

        table
    }

    /// Register a scalar parameter symbol (e.g., `r: f32`).
    pub fn register_parameter(&mut self, name: Ident, ty: Type) {
        let key = name.to_string();
        self.symbols.insert(
            key.clone(),
            Symbol {
                name,
                kind: SymbolKind::Parameter,
                ty: Some(ty),
                span: Span::call_site(),
            },
        );
        // Add to current scope
        if let Some(scope) = self.scope_stack.last_mut() {
            scope.push(key);
        }
    }

    /// Register a local variable.
    pub fn register_local(&mut self, name: Ident, ty: Option<Type>) {
        let key = name.to_string();
        self.symbols.insert(
            key.clone(),
            Symbol {
                name,
                kind: SymbolKind::Local,
                ty,
                span: Span::call_site(),
            },
        );
        if let Some(scope) = self.scope_stack.last_mut() {
            scope.push(key);
        }
    }

    /// Look up a symbol by name.
    pub fn lookup(&self, name: &str) -> Option<&Symbol> {
        self.symbols.get(name)
    }

    /// Check if a name is an intrinsic coordinate.
    pub fn is_intrinsic(&self, name: &str) -> bool {
        self.symbols
            .get(name)
            .is_some_and(|s| s.kind == SymbolKind::Intrinsic)
    }

    /// Check if a name is a captured parameter.
    #[cfg(test)]
    pub fn is_parameter(&self, name: &str) -> bool {
        self.symbols
            .get(name)
            .is_some_and(|s| s.kind == SymbolKind::Parameter)
    }

    /// Get all symbol names.
    #[cfg(test)]
    pub fn all_names(&self) -> impl Iterator<Item = String> + '_ {
        self.symbols.keys().cloned()
    }

    /// Push a new scope (for future block scoping).
    pub fn push_scope(&mut self) {
        self.scope_stack.push(Vec::new());
    }

    /// Pop a scope and remove its symbols.
    pub fn pop_scope(&mut self) {
        if let Some(scope) = self.scope_stack.pop() {
            for name in scope {
                self.symbols.remove(&name);
            }
        }
    }
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intrinsics_are_predefined() {
        let table = SymbolTable::new();
        assert!(table.is_intrinsic("X"));
        assert!(table.is_intrinsic("Y"));
        assert!(!table.is_intrinsic("Z"));
        assert!(!table.is_intrinsic("W"));
        assert!(!table.is_intrinsic("cx"));
    }

    #[test]
    fn register_parameter_marks_name_as_parameter_and_not_intrinsic() {
        let mut table = SymbolTable::new();
        let ident = Ident::new("radius", Span::call_site());
        let ty: Type = syn::parse_quote!(f32);
        table.register_parameter(ident, ty);

        assert!(table.is_parameter("radius"));
        assert!(!table.is_intrinsic("radius"));
    }

    #[test]
    fn all_names_lists_intrinsics_and_every_registered_parameter() {
        let mut table = SymbolTable::new();
        table.register_parameter(
            Ident::new("radius", Span::call_site()),
            syn::parse_quote!(f32),
        );

        let names: std::collections::HashSet<String> = table.all_names().collect();
        let expected: std::collections::HashSet<String> =
            ["X", "Y", "radius"].iter().map(|s| s.to_string()).collect();
        assert_eq!(names, expected);
    }

    #[test]
    fn pop_scope_removes_locals_registered_since_the_matching_push_scope() {
        let mut table = SymbolTable::new();
        table.push_scope();
        table.register_local(Ident::new("dx", Span::call_site()), None);
        assert!(table.lookup("dx").is_some());

        table.pop_scope();

        assert!(table.lookup("dx").is_none());
        // The outer scope (and its intrinsics) must be untouched.
        assert!(table.is_intrinsic("X"));
    }
}
