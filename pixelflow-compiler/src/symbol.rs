//! # Symbol Table
//!
//! The names a kernel body can see, and the one definition of which binding a
//! use of a name refers to.
//!
//! ## Symbol Classes
//!
//! | Class      | Binding Time        | Arena Representation | Example |
//! |------------|---------------------|----------------------|---------|
//! | Intrinsic  | Collapse time       | `Var(0)`, `Var(1)`   | X, Y    |
//! | Parameter  | Construction time   | `Param(i)`           | cx, r   |
//! | Const      | Expansion time      | `Const(v)`           | PI      |
//! | Local      | Expression scope    | A shared `ExprId`    | dx, dy  |
//!
//! A helper's parameters are a fourth thing at lowering — the argument's own
//! node, bound by name where the helper is inlined — but to `sema` they are
//! parameters like any other: a name with a type.
//!
//! ## Intrinsic Coordinates
//!
//! The intrinsic coordinates X and Y are special:
//! - They become `Var(0)` and `Var(1)` arena nodes
//! - They are in scope in an entry, and nothing shadows them: `sema` refuses
//!   a parameter or a `let` named X or Y, and refuses a use of one in a
//!   helper
//!
//! ## Parameter Symbols
//!
//! An entry's parameters become the host function's arguments: `|cx: f32,
//! cy: f32|` produces `move |cx: f32, cy: f32| -> Kernel`, and each reference
//! in the body is a `Param(i)` arena node the builder substitutes with the
//! argument.
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

use crate::sema::Ty;
use std::collections::HashMap;

/// The binding class of a symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolKind {
    /// Intrinsic coordinate variable (X, Y).
    /// Bound at collapse time, as a `Var` node.
    Intrinsic,

    /// A declared parameter (e.g., `r: f32`): an entry's is bound when its
    /// host function runs, a helper's where the helper is inlined.
    Parameter,

    /// A `const` item, evaluated at expansion.
    Const,

    /// Local variable introduced by `let`.
    /// Scoped to the containing block.
    Local,
}

/// A symbol in the symbol table.
#[derive(Debug, Clone, Copy)]
pub struct Symbol {
    /// What kind of symbol this is.
    pub kind: SymbolKind,
    /// The symbol's type.
    pub ty: Ty,
}

/// Rust's lexical scoping over bindings of `V`.
///
/// Sema binds [`Symbol`]s and lowering binds arena ids; both resolve a name
/// through this, which is what makes them agree on what it refers to.
#[derive(Debug, Clone)]
pub struct Scopes<V> {
    /// The function's own scope first — its coordinates, parameters and the
    /// block's items — then one per enclosing block, innermost last. Never
    /// empty: the function's scope is not a block and is never popped.
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
            "popped a scope that was never pushed: the function's own scope is not a block"
        );
        self.frames.pop();
    }

    /// Bind `name` in the innermost scope, shadowing any binding of it
    /// already in scope — including an earlier one in the same block.
    pub fn bind(&mut self, name: String, value: V) {
        self.frames
            .last_mut()
            .expect("the function's own scope is never popped")
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

/// The symbol table for one function's analysis.
#[derive(Debug, Clone)]
pub struct SymbolTable {
    symbols: Scopes<Symbol>,
}

impl SymbolTable {
    /// The coordinate intrinsics. A lattice has two axes; `Z` and `W` are
    /// refused by name in sema, with the message that points at uniforms.
    pub const COORDINATES: [&'static str; 2] = ["X", "Y"];

    /// Create a new symbol table with intrinsic coordinates pre-populated.
    pub fn new() -> Self {
        let mut table = SymbolTable {
            symbols: Scopes::default(),
        };
        for name in Self::COORDINATES {
            table.symbols.bind(
                name.to_string(),
                Symbol {
                    kind: SymbolKind::Intrinsic,
                    ty: Ty::F32,
                },
            );
        }
        table
    }

    /// Register a parameter symbol (e.g., `r: f32`).
    pub fn register_parameter(&mut self, name: &str, ty: Ty) {
        self.bind(name, SymbolKind::Parameter, ty);
    }

    /// Register a `const` item. Every const is an `f32`.
    pub fn register_const(&mut self, name: &str) {
        self.bind(name, SymbolKind::Const, Ty::F32);
    }

    /// Register a local variable in the innermost scope.
    pub fn register_local(&mut self, name: &str, ty: Ty) {
        self.bind(name, SymbolKind::Local, ty);
    }

    fn bind(&mut self, name: &str, kind: SymbolKind, ty: Ty) {
        self.symbols.bind(name.to_string(), Symbol { kind, ty });
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

    /// Check if a name is a declared parameter.
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
        table.register_parameter("radius", Ty::F32);

        assert!(table.is_parameter("radius"));
        assert!(!table.is_intrinsic("radius"));
        assert_eq!(table.lookup("radius").map(|s| s.ty), Some(Ty::F32));
    }

    #[test]
    fn all_names_lists_intrinsics_and_every_registered_parameter() {
        let mut table = SymbolTable::new();
        table.register_parameter("radius", Ty::F32);

        let names: std::collections::HashSet<String> = table.all_names().collect();
        let expected: std::collections::HashSet<String> =
            ["X", "Y", "radius"].iter().map(|s| s.to_string()).collect();
        assert_eq!(names, expected);
    }

    #[test]
    fn pop_scope_removes_locals_registered_since_the_matching_push_scope() {
        let mut table = SymbolTable::new();
        table.push_scope();
        table.register_local("dx", Ty::F32);
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
        table.register_parameter("r", Ty::F32);
        table.push_scope();
        table.register_local("r", Ty::Bool);
        assert_eq!(table.lookup("r").map(|s| s.kind), Some(SymbolKind::Local));
        assert_eq!(table.lookup("r").map(|s| s.ty), Some(Ty::Bool));

        table.pop_scope();

        assert!(table.is_parameter("r"));
        assert_eq!(table.lookup("r").map(|s| s.ty), Some(Ty::F32));
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
