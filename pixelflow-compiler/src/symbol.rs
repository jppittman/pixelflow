//! # Symbol Table
//!
//! The symbol table tracks all identifiers in scope during compilation.
//!
//! ## Symbol Classes
//!
//! PixelFlow has a two-layer symbol table that mirrors the contramap pattern:
//!
//! | Class      | Binding Time        | Runtime Representation | Example |
//! |------------|---------------------|------------------------|---------|
//! | Intrinsic  | Evaluation time     | `X`, `Y`, `Z`, `W`     | X, Y    |
//! | Parameter  | Construction time   | `self.name`            | cx, r   |
//! | Local      | Expression scope    | Local variable         | dx, dy  |
//!
//! ## Intrinsic Coordinates
//!
//! The intrinsic coordinates (X, Y, Z, W) are special:
//! - They're zero-sized types in `pixelflow_core::variables`
//! - They implement `Manifold` and return their respective coordinate
//! - They're always in scope (global namespace)
//!
//! ## Parameter Symbols
//!
//! Parameters declared in the closure syntax become struct fields:
//! - `|cx: f32, cy: f32|` → `struct __Kernel { cx: f32, cy: f32 }`
//! - References in the body become `self.cx`, `self.cy`

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

    /// Manifold parameter from closure syntax (e.g., `inner: kernel`).
    /// Becomes a generic type parameter `M{n}` with trait bounds.
    /// Evaluated at `__p` first, then bound via Let.
    ManifoldParam,

    /// Local variable introduced by `let`.
    /// Scoped to the containing block.
    Local,
}

/// A symbol in the symbol table.
#[derive(Debug, Clone)]
pub struct Symbol {
    /// The identifier name.
    pub name: Ident,
    /// What kind of symbol this is.
    pub kind: SymbolKind,
    /// The type (if known). Intrinsics have implicit types.
    pub ty: Option<Type>,
    /// Where the symbol was defined.
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

        // Register intrinsic coordinate variables
        // These mirror pixelflow_core::variables::{X, Y, Z, W}
        for name in ["X", "Y", "Z", "W"] {
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

    /// Register a manifold parameter symbol (e.g., `inner: kernel`).
    pub fn register_manifold_param(&mut self, name: Ident) {
        let key = name.to_string();
        self.symbols.insert(
            key.clone(),
            Symbol {
                name,
                kind: SymbolKind::ManifoldParam,
                ty: None, // Type is generic (M0, M1, etc.)
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
            .map_or(false, |s| s.kind == SymbolKind::Intrinsic)
    }

    /// Check if a name is a captured parameter.
    pub fn is_parameter(&self, name: &str) -> bool {
        self.symbols
            .get(name)
            .map_or(false, |s| s.kind == SymbolKind::Parameter)
    }

    /// Get all scalar parameter symbols (for struct generation).
    pub fn parameters(&self) -> impl Iterator<Item = &Symbol> {
        self.symbols
            .values()
            .filter(|s| s.kind == SymbolKind::Parameter)
    }

    /// Check if a name is a manifold parameter.
    pub fn is_manifold_param(&self, name: &str) -> bool {
        self.symbols
            .get(name)
            .map_or(false, |s| s.kind == SymbolKind::ManifoldParam)
    }

    /// Get all manifold parameter symbols (for generic type generation).
    pub fn manifold_params(&self) -> impl Iterator<Item = &Symbol> {
        self.symbols
            .values()
            .filter(|s| s.kind == SymbolKind::ManifoldParam)
    }

    /// Get all symbol names (for typo suggestions in error messages).
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
        assert!(table.is_intrinsic("Z"));
        assert!(table.is_intrinsic("W"));
        assert!(!table.is_intrinsic("cx"));
    }

    #[test]
    fn parameter_registration() {
        let mut table = SymbolTable::new();
        let ident = Ident::new("radius", Span::call_site());
        let ty: Type = syn::parse_quote!(f32);
        table.register_parameter(ident, ty);

        assert!(table.is_parameter("radius"));
        assert!(!table.is_intrinsic("radius"));
    }
}
