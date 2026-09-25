//! # Symbol Table
//!
//! The names a kernel body can see, and the one definition of which binding a
//! use of a name refers to.
//!
//! ## Symbol Classes
//!
//! PixelFlow has a two-layer symbol table that mirrors the contramap pattern:
//!
//! | Class      | Binding Time        | Arena Representation | Example |
//! |------------|---------------------|----------------------|---------|
//! | Intrinsic  | Collapse time       | `Var(0)`, `Var(1)`   | X, Y    |
//! | Parameter  | Construction time   | `Param(i)`           | cx, r   |
//! | Local      | Expression scope    | A shared `ExprId`    | dx, dy  |
//!
//! ## Intrinsic Coordinates
//!
//! The intrinsic coordinates X and Y are special:
//! - They become `Var(0)` and `Var(1)` arena nodes
//! - They are always in scope, and nothing shadows them: `sema` refuses a
//!   parameter or a `let` named X or Y
//!
//! ## Parameter Symbols
//!
//! Parameters declared in the closure syntax become the builder closure's
//! arguments: `|cx: f32, cy: f32|` produces `move |cx: f32, cy: f32| -> Kernel`,
//! and each reference in the body is a `Param(i)` arena node the builder
//! substitutes with the argument.
//!
//! ## Scoping
//!
//! Rust's lexical scoping, exactly. A `let` is visible from the statement after
//! it to the end of its block; it shadows every binding of its name already in
//! scope; and a use sees the innermost binding in scope. [`Scopes`] is that
//! rule, and both `sema` (over [`Symbol`]s) and lowering (over arena ids)
//! resolve names through it, so the two cannot disagree about which binding a
//! name means.
//!
//! They did disagree, and both were wrong. `sema` kept one map and deleted a
//! name outright when the block that shadowed it ended, so a parameter
//! shadowed in an inner block was gone after it; lowering kept one map and
//! never removed anything, so an inner `let` replaced an outer one for the
//! rest of the kernel.

use proc_macro2::Span;
use std::collections::HashMap;
use syn::{Ident, Type};

/// The binding class of a symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolKind {
    /// Intrinsic coordinate variable (X, Y).
    /// Bound at collapse time, as a `Var` node.
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

/// Rust's lexical scoping over bindings of `V`.
///
/// Sema binds [`Symbol`]s and lowering binds arena ids; both resolve a name
/// through this, which is what makes them agree on what it refers to.
#[derive(Debug, Clone)]
pub struct Scopes<V> {
    /// The kernel's own scope first — its coordinates and parameters — then
    /// one per enclosing block, innermost last. Never empty: the kernel's
    /// scope is not a block and is never popped.
    frames: Vec<HashMap<String, V>>,
}

impl<V> Default for Scopes<V> {
    fn default() -> Self {
        Scopes {
            frames: vec![HashMap::new()],
        }
    }
}

impl<V> Scopes<V> {
    /// Open a block's scope.
    pub fn push_scope(&mut self) {
        self.frames.push(HashMap::new());
    }

    /// Close the innermost block's scope. Its bindings go out of scope, and
    /// whatever they shadowed is visible again.
    pub fn pop_scope(&mut self) {
        assert!(
            self.frames.len() > 1,
            "popped a scope that was never pushed: the kernel's own scope is not a block"
        );
        self.frames.pop();
    }

    /// Bind `name` in the innermost scope, shadowing any binding of it
    /// already in scope — including an earlier one in the same block.
    pub fn bind(&mut self, name: String, value: V) {
        self.frames
            .last_mut()
            .expect("the kernel's own scope is never popped")
            .insert(name, value);
    }

    /// The innermost binding of `name` in scope.
    pub fn lookup(&self, name: &str) -> Option<&V> {
        self.frames.iter().rev().find_map(|frame| frame.get(name))
    }

    /// Every name bound in any scope, shadowed ones included.
    #[cfg(test)]
    fn names(&self) -> impl Iterator<Item = String> + '_ {
        self.frames.iter().flat_map(|frame| frame.keys().cloned())
    }
}

/// The symbol table for a kernel compilation.
#[derive(Debug, Clone)]
pub struct SymbolTable {
    symbols: Scopes<Symbol>,
}

impl SymbolTable {
    /// Create a new symbol table with intrinsic coordinates pre-populated.
    pub fn new() -> Self {
        let mut table = SymbolTable {
            symbols: Scopes::default(),
        };

        // Register intrinsic coordinate variables. A lattice has two axes;
        // `Z` and `W` are refused by name in sema, with the message that
        // points at uniforms.
        for name in ["X", "Y"] {
            table.symbols.bind(
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
        self.symbols.bind(
            name.to_string(),
            Symbol {
                name,
                kind: SymbolKind::Parameter,
                ty: Some(ty),
                span: Span::call_site(),
            },
        );
    }

    /// Register a local variable in the innermost scope.
    pub fn register_local(&mut self, name: Ident, ty: Option<Type>) {
        self.symbols.bind(
            name.to_string(),
            Symbol {
                name,
                kind: SymbolKind::Local,
                ty,
                span: Span::call_site(),
            },
        );
    }

    /// Look up the innermost binding of a name in scope.
    pub fn lookup(&self, name: &str) -> Option<&Symbol> {
        self.symbols.lookup(name)
    }

    /// Check if a name is an intrinsic coordinate.
    pub fn is_intrinsic(&self, name: &str) -> bool {
        self.lookup(name)
            .is_some_and(|s| s.kind == SymbolKind::Intrinsic)
    }

    /// Check if a name is a captured parameter.
    #[cfg(test)]
    pub fn is_parameter(&self, name: &str) -> bool {
        self.lookup(name)
            .is_some_and(|s| s.kind == SymbolKind::Parameter)
    }

    /// Get all symbol names.
    #[cfg(test)]
    pub fn all_names(&self) -> impl Iterator<Item = String> + '_ {
        self.symbols.names()
    }

    /// Open a block's scope.
    pub fn push_scope(&mut self) {
        self.symbols.push_scope();
    }

    /// Close the innermost block's scope, and with it every local bound in it.
    pub fn pop_scope(&mut self) {
        self.symbols.pop_scope();
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

    /// Popping a scope restores what its bindings shadowed. The table used to
    /// be one map, so a local shadowing a parameter overwrote it and the pop
    /// then deleted the name outright: the parameter was gone for the rest of
    /// the kernel.
    #[test]
    fn pop_scope_restores_the_binding_a_local_shadowed() {
        let mut table = SymbolTable::new();
        table.register_parameter(Ident::new("r", Span::call_site()), syn::parse_quote!(f32));
        table.push_scope();
        table.register_local(Ident::new("r", Span::call_site()), None);
        assert_eq!(table.lookup("r").map(|s| s.kind), Some(SymbolKind::Local));

        table.pop_scope();

        assert!(table.is_parameter("r"));
    }

    #[test]
    fn a_lookup_sees_the_innermost_binding_in_scope() {
        let mut scopes = Scopes::default();
        scopes.bind("a".to_string(), 0);
        scopes.push_scope();
        scopes.bind("a".to_string(), 1);
        scopes.push_scope();
        assert_eq!(scopes.lookup("a"), Some(&1));
        scopes.bind("a".to_string(), 2);
        scopes.bind("a".to_string(), 3);
        assert_eq!(scopes.lookup("a"), Some(&3));
        scopes.pop_scope();
        assert_eq!(scopes.lookup("a"), Some(&1));
        scopes.pop_scope();
        assert_eq!(scopes.lookup("a"), Some(&0));
    }

    #[test]
    #[should_panic(expected = "never pushed")]
    fn the_kernels_own_scope_cannot_be_popped() {
        Scopes::<u8>::default().pop_scope();
    }
}
