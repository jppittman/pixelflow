//! Model 3, the name: an identifier as the language compares it, and what a
//! written method or projection resolves to — once, here, for every stage.

use pixelflow_ir::arena::Axis;
use pixelflow_ir::{OpKind, known_method_names};
use proc_macro2::Span;
use syn::Ident;
use syn::ext::IdentExt;

/// A name as the language compares it: an identifier without its `r#`.
///
/// rustc reads `r#a` and `a` as one name, so a stage that keys a map by the
/// written spelling would let a raw one walk around a refusal of the other
/// (a second `fn r#f` beside `f`), or miss what the other bound. Every map
/// keyed by a name, and every comparison of two, is by this, built only from
/// an identifier ([`Name::of`]) or, for the two the language spells itself,
/// by [`Name::coordinates`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct Name(Ident);

impl Name {
    /// The name an identifier spells, whichever way it is spelled.
    pub(crate) fn of(ident: &Ident) -> Self {
        Name(ident.unraw())
    }

    /// The coordinate intrinsics, by spelling. A lattice has two axes; `Z`
    /// and `W` are refused by name in sema, with the message that points at
    /// uniforms.
    const COORDINATES: [(&'static str, Axis); 2] = [("X", Axis::X), ("Y", Axis::Y)];

    /// A name the language spells itself, as plain text: `X` and `Y`.
    fn spelled(text: &str) -> Self {
        Name(Ident::new(text, Span::call_site()))
    }

    /// The axis this name is the coordinate of, if it is one.
    pub(crate) fn coordinate(&self) -> Option<Axis> {
        Self::COORDINATES
            .into_iter()
            .find(|&(spelling, _)| self == spelling)
            .map(|(_, axis)| axis)
    }

    /// The two names the language spells itself, each with its axis: `X`
    /// then `Y`.
    pub(crate) fn coordinates() -> impl Iterator<Item = (Name, Axis)> {
        Self::COORDINATES
            .into_iter()
            .map(|(spelling, axis)| (Name::spelled(spelling), axis))
    }
}

impl std::fmt::Display for Name {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl PartialEq<str> for Name {
    fn eq(&self, text: &str) -> bool {
        self.0 == text
    }
}

/// Whether two identifiers spell one name.
pub(crate) fn same_name(a: &Ident, b: &Ident) -> bool {
    Name::of(a) == Name::of(b)
}

/// A method name as the language reads it, `clone` included.
const CLONE: &str = "clone";

/// What a method call denotes: one of the three things a name can mean.
///
/// Resolved from the written name and the number of arguments by
/// [`Method::resolve`], which every stage that asks "which method is this"
/// reads, so no two of them can disagree about which methods exist or what
/// arity each takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Method {
    /// `.clone()`: the identity. An arena expression is a value, so a copy of
    /// one is itself.
    Clone,
    /// Observe an `f32` or `bool` field at two other coordinates.
    At,
    /// A primitive: one [`OpKind`], at the arity `OpKind::from_method_call`
    /// admits it.
    Op(OpKind),
    /// A fixed composition of primitives, which the IR's `library` defines.
    Library(LibraryMethod),
}

/// A method that denotes a fixed composition of primitive ops rather than a
/// single [`OpKind`]: the syntax's name for one of `pixelflow_ir::library`'s
/// definitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LibraryMethod {
    Fract,
    Hypot,
    Clamp,
}

impl LibraryMethod {
    /// Each one's spelling and the arguments it takes, not counting the
    /// receiver.
    const ALL: [(Self, &'static str, usize); 3] = [
        (Self::Fract, "fract", 0),
        (Self::Hypot, "hypot", 1),
        (Self::Clamp, "clamp", 2),
    ];
}

impl Method {
    /// The method `name` denotes when called with `arity` arguments, not
    /// counting the receiver, or `None` if no method has that name at that
    /// arity.
    ///
    /// Name and arity are resolved together, so `.sqrt(1.0)` is no method
    /// rather than a `sqrt` that something later finds malformed.
    pub(crate) fn resolve(name: &str, arity: usize) -> Option<Self> {
        if name == CLONE && arity == 0 {
            return Some(Self::Clone);
        }
        if name == "at" && arity == 2 {
            return Some(Self::At);
        }
        if let Some(op) = OpKind::from_method_call(name, arity) {
            return Some(Self::Op(op));
        }
        LibraryMethod::ALL
            .into_iter()
            .find(|&(_, spelling, arguments)| spelling == name && arguments == arity)
            .map(|(library, ..)| Self::Library(library))
    }

    /// Every method the language has, as `(name, arity)`, the arity not
    /// counting the receiver: the surface a body may call.
    pub(crate) fn advertised() -> impl Iterator<Item = (&'static str, usize)> {
        let ops = known_method_names().map(|name| {
            let op = OpKind::from_name(name)
                .expect("known_method_names() only yields names from_name parses");
            (name, op.arity() - 1)
        });
        let libraries = LibraryMethod::ALL
            .into_iter()
            .map(|(_, spelling, arguments)| (spelling, arguments));
        ops.chain(libraries).chain([(CLONE, 0), ("at", 2)])
    }
}

/// A derivative projection, called as `DX(e)`.
///
/// Resolved from the callee's name by [`Projection::from_name`], once, in
/// the parser. Nothing can shadow a projection's name — `sema` refuses every
/// item, parameter, `let` and index named after one — so a call to one with its
/// one argument is a projection wherever it stands, and no later stage asks
/// again. At any other arity it stays a [`CallExpr`](super::CallExpr), which `sema` refuses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Projection {
    /// `V(e)`: the value itself. Every arena expression is already
    /// value-space, so it is the identity.
    Value,
    Dx,
    Dy,
    Dxx,
    Dxy,
    Dyy,
}

impl Projection {
    /// The projection a name denotes, if any.
    pub(crate) fn from_name(name: &Name) -> Option<Self> {
        match name.to_string().as_str() {
            "V" => Some(Self::Value),
            "DX" => Some(Self::Dx),
            "DY" => Some(Self::Dy),
            "DXX" => Some(Self::Dxx),
            "DXY" => Some(Self::Dxy),
            "DYY" => Some(Self::Dyy),
            _ => None,
        }
    }
}
