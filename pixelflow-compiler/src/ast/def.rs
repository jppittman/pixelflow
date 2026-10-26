//! Model 2, the definition: what a `kernel!` invocation spells, and the one
//! view of a function — entry or helper, either spelling — the stages read.

use super::{Expr, FunctionType, bare_name, same_name};
use syn::ext::IdentExt;
use syn::{Ident, Type};

/// A complete kernel definition: what a `kernel!` invocation spells.
///
/// A sum of the two spellings. The closure form is one entry — parameters and
/// a body, its type inferred — and its expansion is an expression; the items
/// form is a block of records, `const`s and `fn`s, and its expansion is
/// items. The stages that treat both alike read [`Self::records`],
/// [`Self::consts`] and [`Self::fns`], which a closure has none of.
#[derive(Debug, Clone)]
pub(crate) enum KernelDef {
    /// `kernel!(|a: f32, …| e)`: the expansion is an expression — a `Kernel`
    /// with no parameters, a closure over their `f32`s with some, each one a
    /// uniform.
    Closure(ClosureEntry),
    /// `kernel! { struct …; const …; fn …; pub fn … }`: the expansion is
    /// items — a host struct per record, a host `const` per `pub const`, and
    /// per entry a host `fn` and, when it has parameters, its `Args` record.
    Items {
        /// The block's records, in declaration order: [`RecordId`] indexes
        /// this.
        records: Vec<RecordItem>,
        /// The block's `const` items, in declaration order.
        consts: Vec<ConstItem>,
        /// The block's `fn` items — entries and helpers — in declaration
        /// order.
        fns: Vec<FnItem>,
    },
}

impl KernelDef {
    /// The block's records, in declaration order.
    pub(crate) fn records(&self) -> &[RecordItem] {
        match self {
            KernelDef::Closure(_) => &[],
            KernelDef::Items { records, .. } => records,
        }
    }

    /// The block's `const` items, in declaration order.
    pub(crate) fn consts(&self) -> &[ConstItem] {
        match self {
            KernelDef::Closure(_) => &[],
            KernelDef::Items { consts, .. } => consts,
        }
    }

    /// The block's `fn` items, in declaration order.
    pub(crate) fn fns(&self) -> &[FnItem] {
        match self {
            KernelDef::Closure(_) => &[],
            KernelDef::Items { fns, .. } => fns,
        }
    }

    /// The record a declared type names, if it names one of the block's.
    ///
    /// The one resolution of a record's name: `sema` types a parameter by
    /// it, and lowering and emission lay a record parameter out by it, so no
    /// two stages can disagree about which record a type is.
    pub(crate) fn record_named(&self, ty: &Type) -> Option<RecordId> {
        let name = bare_name(ty)?;
        self.records()
            .iter()
            .position(|record| same_name(&record.name, name))
            .map(RecordId)
    }

    /// The record `id` names.
    pub(crate) fn record(&self, id: RecordId) -> &RecordItem {
        &self.records()[id.0]
    }
}

/// The closure form's one entry: `|a: f32, …| e`.
///
/// It has no name, no visibility and no declared return type, because it
/// is never a host function: its type is inferred, and it has no
/// records or helpers.
#[derive(Debug, Clone)]
pub(crate) struct ClosureEntry {
    pub(crate) params: Vec<Param>,
    /// The kernel body expression.
    pub(crate) body: Expr,
}

impl ClosureEntry {
    pub(crate) fn callable(&self) -> Callable<'_> {
        Callable {
            name: None,
            role: Role::Entry,
            params: &self.params,
            body: &self.body,
        }
    }
}

/// What `sema` and lowering read of a function: its role, parameters and
/// body. An item's [`FnItem::callable`] and
/// the closure's [`ClosureEntry::callable`] are the two ways to have one, so
/// a stage that handles an entry handles either spelling of it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Callable<'a> {
    /// The function's name; the closure form has none.
    pub(crate) name: Option<&'a Ident>,
    pub(crate) role: Role,
    pub(crate) params: &'a [Param],
    pub(crate) body: &'a Expr,
}

impl<'a> Callable<'a> {
    /// Its kernel-typed parameters in declaration order: the order their
    /// arguments are admitted in, and so the order each argument's uniforms
    /// are declared in, after the function's own
    /// ([`AnalyzedKernel::parameters`](crate::sema::AnalyzedKernel::parameters))
    /// — `Kernel`'s rule for a composition's operands, which a positional
    /// binding of the composed program reads (O3 of the plan).
    pub(crate) fn kernel_parameters(&self) -> impl Iterator<Item = &'a Ident> {
        self.params
            .iter()
            .filter(|param| param.is_kernel())
            .map(|param| &param.name)
    }
}

/// Which of the block's records a type is: an index into
/// [`KernelDef::records`], in declaration order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct RecordId(usize);

/// `struct R { a: f32, b: f32 }`: a record of named `f32` fields
/// (docs/language/2026-09-25-the-language-is-kernel.md §1.3).
///
/// Emitted as a host `#[repr(C)]` struct of the same name, fields and
/// visibility. In a body a record is its fields: an entry's record parameter
/// is one uniform per field, in field order, and a helper's is its
/// argument's fields.
#[derive(Debug, Clone)]
pub(crate) struct RecordItem {
    /// Its attributes (any but `repr` and `cfg`), re-emitted on the host
    /// struct.
    pub(crate) attrs: Vec<syn::Attribute>,
    pub(crate) vis: syn::Visibility,
    pub(crate) name: Ident,
    /// The fields, in declaration order: the order a record parameter's
    /// uniforms are declared in.
    pub(crate) fields: Vec<RecordField>,
}

/// One named field of a record. `sema` requires its type to be `f32`.
#[derive(Debug, Clone)]
pub(crate) struct RecordField {
    /// Its attributes (any but `repr` and `cfg`), re-emitted on the host
    /// struct's field.
    pub(crate) attrs: Vec<syn::Attribute>,
    pub(crate) vis: syn::Visibility,
    pub(crate) name: Ident,
    pub(crate) ty: Type,
}

/// A `const NAME: f32 = expr;` or `const NAME: usize = expr;` item,
/// evaluated at expansion.
#[derive(Debug, Clone)]
pub(crate) struct ConstItem {
    /// Doc comments, re-emitted on a `pub const`'s host twin.
    pub(crate) attrs: Vec<syn::Attribute>,
    /// `pub` makes the value a host `const` too.
    pub(crate) vis: syn::Visibility,
    pub(crate) name: Ident,
    /// The declared type; sema requires `f32` or `usize`.
    pub(crate) ty: Type,
    pub(crate) init: Expr,
}

/// A `fn` item: an entry when `pub`, a helper otherwise.
#[derive(Debug, Clone)]
pub(crate) struct FnItem {
    /// Doc comments, re-emitted on an entry's host function.
    pub(crate) attrs: Vec<syn::Attribute>,
    /// `pub` makes an entry — the macro emits a host function for it. A
    /// private `fn` is a helper, inlined at each call.
    pub(crate) vis: syn::Visibility,
    pub(crate) name: Ident,
    pub(crate) params: Vec<Param>,
    /// The declared return type.
    pub(crate) ret: Type,
    /// The kernel body expression.
    pub(crate) body: Expr,
}

/// What a `fn` item is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Role {
    /// A program: it may read `X` and `Y`, and the macro emits a host
    /// function for it.
    Entry,
    /// A function of its arguments only, inlined at each call.
    Helper,
}

impl FnItem {
    pub(crate) fn callable(&self) -> Callable<'_> {
        Callable {
            name: Some(&self.name),
            role: self.role(),
            params: &self.params,
            body: &self.body,
        }
    }

    /// An entry is any `fn` with a visibility; a helper has none.
    pub(crate) fn role(&self) -> Role {
        match self.vis {
            syn::Visibility::Inherited => Role::Helper,
            _ => Role::Entry,
        }
    }

    /// Whether a parameter is kernel-typed: the entry is then a composition,
    /// built when its host function is called (plan Phase D-a).
    pub(crate) fn takes_a_kernel(&self) -> bool {
        self.params.iter().any(Param::is_kernel)
    }

    /// Whether the entry has an `Args` record (plan §1.4): it has uniform
    /// parameters and nothing else. An entry that takes a kernel has none —
    /// its program declares the kernel's uniforms after its own, and the
    /// record could rebind only a program whose argument declares none.
    /// Binding a composed program is O3 of the plan.
    pub(crate) fn has_args_record(&self) -> bool {
        !self.params.is_empty() && !self.takes_a_kernel()
    }

    /// The name of an entry's `Args` record: the entry's name in
    /// UpperCamelCase, then `Args` — `shifted_radius` has
    /// `ShiftedRadiusArgs` (plan §1.4). Dropping the underscores can leave a
    /// digit first, which no identifier starts with, so the name then gains
    /// one leading underscore: `_2d` has `_2dArgs`, `__3` has `_3Args`.
    /// `sema` refuses one that collides, and emission names the record by it.
    pub(crate) fn args_record(&self) -> Ident {
        let camel: String = self
            .name
            .unraw()
            .to_string()
            .split('_')
            .flat_map(|word| {
                let mut letters = word.chars();
                letters
                    .next()
                    .map(|first| first.to_uppercase().chain(letters))
                    .into_iter()
                    .flatten()
            })
            .collect();
        let lead = if camel.starts_with(char::is_numeric) {
            "_"
        } else {
            ""
        };
        Ident::new(&format!("{lead}{camel}Args"), self.name.span())
    }
}

/// A declared parameter: a scalar, one of the block's records, or a kernel.
#[derive(Debug, Clone)]
pub(crate) struct Param {
    /// Parameter name.
    pub(crate) name: Ident,
    /// The declared type (`f32`, a record or `impl Fn(f32, f32) -> f32`;
    /// `bool` in a helper).
    pub(crate) ty: Box<Type>,
}

impl Param {
    /// Whether the parameter is kernel-typed, `impl Fn(f32, f32) -> f32`.
    pub(crate) fn is_kernel(&self) -> bool {
        FunctionType::of(&self.ty) == FunctionType::Kernel
    }
}
