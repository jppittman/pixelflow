//! The stage's function and its output types: `analyze`, the `Items` a body
//! sees, and `Ty`, the type a body's every name and expression has.

use crate::ast::{
    FnItem, FunctionType, KernelDef, Name, PLAN, Param, Projection, RecordField, RecordId,
    Reduction, Role, bare_name, same_name,
};
use pixelflow_ir::OpKind;
use proc_macro2::Span;
use std::collections::{HashMap, HashSet};
use syn::{Ident, Type};

use super::body::FnAnalyzer;
use super::calls::refuse_recursion;
use super::consts::evaluate_consts;

/// The type of a name or an expression in a kernel body.
///
/// Two value types, and the IR has one lane for both: a `bool` is an
/// all-ones or all-zero mask (`OpKind::mask`). The distinction is enforced
/// here because it cannot be enforced there — a mask read as a number is a
/// NaN, and a number used as a mask blends bit patterns.
///
/// The third, `usize`, is a count and not a value. A fold's index is one,
/// and so are a `usize` const and an entry's per-call count parameter. A name of
/// this type is refused wherever a value is expected, and converted only by
/// `i as f32`: the index is an `f32` lane already, so the conversion lowers
/// to nothing, but writing it is what keeps index arithmetic out of the
/// language.
///
/// The fourth, a record, is its `f32` fields under one name (§1.3). It has
/// no lane of its own: lowering flattens it, so a record is written only by
/// name — a parameter, a `let` alias, an argument passed on — and read only
/// by field. Anything that would compute one, choose one or return one is
/// Phase D (D7).
///
/// The fifth, a kernel, is a function of the two coordinates the host
/// supplies at run time: a parameter typed `impl Fn(f32, f32) -> f32`
/// (Phase D-a). It is not a value either. A body applies it, `k(x, y)` —
/// the argument evaluated at `(x, y)`, an `f32` — or passes it by name to a
/// helper's kernel-typed parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Ty {
    /// A value.
    F32,
    /// A mask: a comparison produces it, `&` and `|` combine it, an `if`
    /// chooses by it.
    Bool,
    /// A count: a fold's index, a `usize` const or an entry parameter.
    /// 64 bits, as `usize` is on every target the language compiles for.
    Usize,
    /// One of the block's records.
    Record(RecordId),
    /// A kernel-typed parameter: `impl Fn(f32, f32) -> f32`.
    Kernel,
}

impl Ty {
    /// The type a declaration names, or `None` for one the language does not
    /// have.
    pub(super) fn from_syn(ty: &Type) -> Option<Ty> {
        let name = bare_name(ty)?;
        [Ty::F32, Ty::Bool, Ty::Usize]
            .into_iter()
            .find(|primitive| name == primitive.name())
    }

    /// The value type a declaration names, `f32` or `bool`; `what` begins
    /// the refusal of any other ("a kernel parameter is").
    pub(super) fn of_a_value(ty: &Type, what: &str) -> syn::Result<Ty> {
        match Ty::from_syn(ty) {
            Some(value @ (Ty::F32 | Ty::Bool)) => Ok(value),
            Some(Ty::Usize) => Err(syn::Error::new_spanned(
                ty,
                format!(
                    "{what} an `f32` or a `bool`\n\
                     \n\
                     note: a `usize` is a count, not a value: a fold's index, a `usize` const \
                     or an entry's count parameter (`pub fn f(n: usize)`), which a \
                     body converts by `i as f32`"
                ),
            )),
            Some(Ty::Record(_) | Ty::Kernel) | None => Err(syn::Error::new_spanned(
                ty,
                format!(
                    "{what} an `f32` or a `bool`\n\
                     \n\
                     note: every value in a kernel body is one of the two"
                ),
            )),
        }
    }

    /// The type's name, as written; a record's is its own
    /// ([`Items::name_of`]), and here it is only `record`.
    pub(super) fn name(self) -> &'static str {
        match self {
            Ty::F32 => "f32",
            Ty::Bool => "bool",
            Ty::Usize => "usize",
            Ty::Record(_) => "record",
            Ty::Kernel => "impl Fn(f32, f32) -> f32",
        }
    }
}

/// What a fold's body is, and so what the fold is: the terms of a sum, a
/// product, a minimum or a maximum are `f32`s, and those of `any` and `all`
/// are `bool`s. A fold has its terms' type.
pub(super) fn term_type(reduction: Reduction) -> (Ty, &'static str) {
    match reduction {
        Reduction::Sum | Reduction::Product | Reduction::Min | Reduction::Max => (
            Ty::F32,
            "a sum, a product, a minimum or a maximum combines `f32`s",
        ),
        Reduction::Any | Reduction::All => (
            Ty::Bool,
            "`any` and `all` combine `bool`s; a comparison gives one",
        ),
    }
}

/// How an `OpKind` method types: what its receiver and arguments are, and
/// what it gives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MethodTyping {
    /// Every operand an `f32`, and an `f32` out.
    Arithmetic,
    /// Two `f32`s in, a `bool` out.
    Comparison,
    /// A `bool` receiver chooses between two operands of one type.
    Choice,
}

/// The typing of an `OpKind` a kernel body may call as a method.
///
/// An op whose result is a bit pattern ([`OpKind::is_bitwise_domain`]) is a
/// mask or an integer lane, which `f32` arithmetic would read as a number:
/// the ones the language types are named above it, and a new one the IR
/// advertises refuses to be typed as arithmetic until it is given a typing
/// here. `every_advertised_method_compiles` types every advertised op, so the
/// addition fails there, for whoever makes it.
pub(super) fn method_typing(op: OpKind) -> MethodTyping {
    match op {
        OpKind::Lt | OpKind::Le | OpKind::Gt | OpKind::Ge | OpKind::Eq | OpKind::Ne => {
            MethodTyping::Comparison
        }
        OpKind::If => MethodTyping::Choice,
        op if !op.is_bitwise_domain() => MethodTyping::Arithmetic,
        op => unreachable!("`{op:?}` yields a bit pattern and has no `MethodTyping`: give it one"),
    }
}

/// A `const`'s value, evaluated at expansion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum ConstValue {
    /// `const NAME: f32`: a value, folded into every body that names it.
    F32(f32),
    /// `const NAME: usize`: a count, for a range's bounds or `NAME as f32`.
    Usize(u64),
}

/// The result of semantic analysis.
#[derive(Debug)]
pub(crate) struct AnalyzedKernel {
    /// The original kernel definition.
    pub(crate) def: KernelDef,
    /// Every `const`'s value, evaluated at expansion.
    pub(crate) consts: HashMap<Name, ConstValue>,
}

/// An entry's parameter as its program sees it (plan §1.4): a uniform
/// scalar, or a record of them.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Parameter<'a> {
    /// An `f32`: one uniform.
    Scalar(&'a Ident),
    /// A per-call `usize`, carried by one uniform as an `f32` lane.
    Count(&'a Ident),
    /// One of the block's records, with that record's fields: one uniform per
    /// field.
    Record {
        name: &'a Ident,
        id: RecordId,
        fields: &'a [RecordField],
    },
}

/// One uniform scalar of an entry — what the host function's argument is,
/// for this call.
#[derive(Debug, Clone)]
pub(crate) enum Scalar {
    /// A parameter: `param`.
    Param(Ident),
    /// A per-call count, checked before it is converted to `f32`.
    Count(Ident),
    /// One field of a record parameter: `param.field`.
    Field(Ident, Ident),
}

impl AnalyzedKernel {
    /// An entry's uniform parameters in declaration order: an `f32` is one
    /// uniform, a record one per field (plan §1.4). The one definition of
    /// the order a program declares its own uniforms in. Lowering declares
    /// them by it and hands on the scalar behind each declaration, which is
    /// what the host function and its `Args` record supply — so a positional
    /// binding cannot disagree with the program it binds. A kernel-typed parameter is not among them: its uniforms are
    /// the argument's, declared after these ([`crate::ast::Callable::kernel_parameters`]).
    pub(crate) fn parameters<'a>(&'a self, params: &'a [Param]) -> Vec<Parameter<'a>> {
        params
            .iter()
            .filter(|param| !param.is_kernel())
            .map(
                |param| match (Ty::from_syn(&param.ty), self.def.record_named(&param.ty)) {
                    (Some(Ty::Usize), _) => Parameter::Count(&param.name),
                    (_, None) => Parameter::Scalar(&param.name),
                    (_, Some(id)) => Parameter::Record {
                        name: &param.name,
                        id,
                        fields: self.def.record(id).fields.as_slice(),
                    },
                },
            )
            .collect()
    }
}

/// Type-check a parsed block and evaluate its `const`s.
///
/// # Errors
///
/// A spanned `syn::Error` for the first name, type, call-graph, `const` or
/// fold-bound refusal listed in the module doc.
pub(crate) fn analyze(def: KernelDef) -> syn::Result<AnalyzedKernel> {
    let items = Items::collect(&def)?;
    let consts = evaluate_consts(def.consts(), &items.consts)?;
    match &def {
        KernelDef::Closure(entry) => {
            FnAnalyzer::new(entry.callable(), &items, &consts)?.check()?;
        }
        KernelDef::Items { fns, .. } => {
            for f in fns {
                FnAnalyzer::new(f.callable(), &items, &consts)?.check()?;
            }
            refuse_recursion(fns)?;
            require_an_entry(fns)?;
        }
    }
    Ok(AnalyzedKernel { def, consts })
}

/// A block whose expansion would be empty is refused rather than expanded
/// to nothing.
fn require_an_entry(fns: &[FnItem]) -> syn::Result<()> {
    let has_entry = fns.iter().any(|f| f.role() == Role::Entry);
    if !has_entry {
        return Err(syn::Error::new(
            Span::call_site(),
            "a `kernel!` block with no entry emits nothing\n\
             \n\
             note: a `pub fn` is an entry, and the macro emits a host function for each; \
             a private `fn` is a helper, inlined where it is called",
        ));
    }
    Ok(())
}

/// Coordinate names a `kernel!` body may not use: `Z` and `W` name axes a
/// lattice does not have.
pub(super) const RETIRED_COORDINATES: [&str; 2] = ["Z", "W"];

/// The projection of the axis a lattice does not have.
pub(super) const RETIRED_PROJECTION: &str = "DZ";

/// A `fn`'s declared types.
#[derive(Debug)]
pub(super) struct Signature {
    pub(super) role: Role,
    pub(super) params: Vec<Ty>,
    pub(super) ret: Ty,
}

/// The block's items by name: what every body can see besides its own
/// scope.
pub(super) struct Items<'a> {
    /// The block, whose records a type names ([`KernelDef::record_named`]).
    pub(super) def: &'a KernelDef,
    /// Every record's name, which no other item may take.
    records: HashSet<Name>,
    /// Every `const`'s declared type: `f32` or `usize`.
    pub(super) consts: HashMap<Name, Ty>,
    pub(super) fns: HashMap<Name, Signature>,
}

impl<'a> Items<'a> {
    /// Collect the items, refusing a duplicate name, a name that is a
    /// coordinate or a projection, and a declared type the language does
    /// not have.
    fn collect(def: &'a KernelDef) -> syn::Result<Self> {
        let mut items = Items {
            def,
            records: HashSet::with_capacity(def.records().len()),
            consts: HashMap::with_capacity(def.consts().len()),
            fns: HashMap::with_capacity(def.fns().len()),
        };
        for record in def.records() {
            items.refuse_a_taken_name(&record.name)?;
            items.records.insert(Name::of(&record.name));
        }
        for record in def.records() {
            items.check_fields(&record.fields)?;
        }
        for c in def.consts() {
            items.refuse_a_taken_name(&c.name)?;
            let ty = match Ty::from_syn(&c.ty) {
                Some(ty @ (Ty::F32 | Ty::Usize)) => ty,
                Some(Ty::Bool | Ty::Record(_) | Ty::Kernel) | None => {
                    return Err(syn::Error::new_spanned(
                        &c.ty,
                        "a `const` in a `kernel!` block is an `f32` or a `usize`\n\
                         \n\
                         note: it is evaluated at expansion: an `f32` is folded into every \
                         body that names it, and a `usize` is a count, a range's bound; a \
                         `bool` would be a mask constant, which nothing spells yet",
                    ));
                }
            };
            items.consts.insert(Name::of(&c.name), ty);
        }
        for f in def.fns() {
            items.refuse_a_taken_name(&f.name)?;
            let signature = items.signature(f)?;
            items.fns.insert(Name::of(&f.name), signature);
        }
        items.refuse_a_taken_args_record()?;
        Ok(items)
    }

    /// An item's name is unique in the block, and is not a coordinate, a
    /// projection or a function of the language.
    fn refuse_a_taken_name(&self, name: &Ident) -> syn::Result<()> {
        let text = name.to_string();
        let key = Name::of(name);
        if key.coordinate().is_some() {
            return Err(syn::Error::new(
                name.span(),
                format!("`{text}` is the intrinsic coordinate; an item cannot be named after it"),
            ));
        }
        if Projection::from_name(&key).is_some() {
            return Err(syn::Error::new(
                name.span(),
                format!("`{text}` is a projection; an item cannot be named after it"),
            ));
        }
        let taken = self.consts.contains_key(&key)
            || self.fns.contains_key(&key)
            || self.records.contains(&key);
        if taken {
            return Err(syn::Error::new(
                name.span(),
                format!("`{text}` is defined twice in this `kernel!` block"),
            ));
        }
        Ok(())
    }

    /// Each entry with parameters has an `Args` record (plan §1.4), a host
    /// struct beside the block's records, so its name is one no record and
    /// no other entry's `Args` takes.
    fn refuse_a_taken_args_record(&self) -> syn::Result<()> {
        let mut taken: HashMap<Name, &Ident> = HashMap::with_capacity(self.def.fns().len());
        let entries = self.def.fns().iter().filter(|f| f.role() == Role::Entry);
        for entry in entries.filter(|f| f.has_args_record()) {
            let args = Name::of(&entry.args_record());
            let clash = match taken.get(&args) {
                Some(other) => Some(format!("the `Args` record of `{other}`")),
                None => self
                    .records
                    .contains(&args)
                    .then(|| format!("the record `{args}`")),
            };
            if let Some(clash) = clash {
                return Err(syn::Error::new(
                    entry.name.span(),
                    format!(
                        "`{}`'s `Args` record, `{args}`, is also {clash}\n\
                         \n\
                         note: each entry with uniform parameters, and no kernel-typed one, has \
                         an `Args` record, named after it in UpperCamelCase (§1.4 of {PLAN})\n\
                         help: rename one of the two",
                        entry.name
                    ),
                ));
            }
            taken.insert(args, &entry.name);
        }
        Ok(())
    }

    /// Every field of a record is an `f32` (§1.3), and no two share a name.
    fn check_fields(&self, fields: &[RecordField]) -> syn::Result<()> {
        let mut seen: Vec<&Ident> = Vec::with_capacity(fields.len());
        for field in fields {
            if seen.iter().any(|&seen| same_name(seen, &field.name)) {
                return Err(syn::Error::new(
                    field.name.span(),
                    format!("field `{}` is declared twice", field.name),
                ));
            }
            seen.push(&field.name);
            if Ty::from_syn(&field.ty) == Some(Ty::F32) {
                continue;
            }
            let what = match (&field.ty, self.def.record_named(&field.ty)) {
                (_, Some(inner)) => format!("the record `{}`", self.def.record(inner).name),
                (ty, None) if FunctionType::of(ty) != FunctionType::NotOne => format!(
                    "a function: a kernel is a `fn`'s parameter, applied by name, never a \
                     field (Phase D-a of {PLAN})"
                ),
                (Type::Array(_), None) => "an array".to_string(),
                (Type::Tuple(_), None) => "a tuple".to_string(),
                (ty, None) => match Ty::from_syn(ty) {
                    Some(scalar) => format!("a `{}`", scalar.name()),
                    None => "a type the language does not have".to_string(),
                },
            };
            return Err(syn::Error::new_spanned(
                &field.ty,
                format!(
                    "a record's field is an `f32`, and `{}` is {what}\n\
                     \n\
                     note: a record is named `f32` fields and nothing else — no record inside \
                     a record, no array, no generic (§1.3 of {PLAN})",
                    field.name
                ),
            ));
        }
        Ok(())
    }

    /// The type a declaration names in this block: a scalar, or a record.
    pub(super) fn declared(&self, ty: &Type) -> Option<Ty> {
        self.def
            .record_named(ty)
            .map(Ty::Record)
            .or_else(|| Ty::from_syn(ty))
    }

    /// A type's name as a diagnostic writes it: a record's is its own.
    pub(super) fn name_of(&self, ty: Ty) -> String {
        match ty {
            Ty::Record(record) => self.def.record(record).name.to_string(),
            Ty::F32 | Ty::Bool | Ty::Usize | Ty::Kernel => ty.name().to_string(),
        }
    }

    /// A `fn`'s parameter and return types. An entry's parameters are its
    /// uniforms: `f32`s, and records of them. A helper's may be `bool`s too —
    /// a mask is an ordinary argument at an inlined call.
    fn signature(&self, f: &FnItem) -> syn::Result<Signature> {
        let role = f.role();
        let mut params = Vec::with_capacity(f.params.len());
        for param in &f.params {
            params.push(self.param_type(param, role)?);
        }
        let ret = self.return_type(&f.ret)?;
        Ok(Signature { role, params, ret })
    }

    /// What a `fn` returns: an `f32` or a `bool`. A record return is Phase
    /// D (D7): it lowers to one program per field, or to a packed word.
    fn return_type(&self, ty: &Type) -> syn::Result<Ty> {
        if FunctionType::of(ty) != FunctionType::NotOne {
            return Err(syn::Error::new_spanned(
                ty,
                format!(
                    "a `fn` returning a function\n\
                     \n\
                     note: a kernel `fn` returns an `f32` or a `bool`; a kernel is composed by \
                     the host, which passes one entry's kernel to another's kernel-typed \
                     parameter (§1.7 and Phase D-a of {PLAN})"
                ),
            ));
        }
        if let Some(record) = self.def.record_named(ty) {
            return Err(syn::Error::new_spanned(
                ty,
                format!(
                    "a `fn` returning the record `{}`\n\
                     \n\
                     note: a kernel `fn` returns an `f32` or a `bool`; a record parameter's \
                     fields are read by name, `p.x0`\n\
                     note: returning a record (`-> Rgba`) is Phase D (D7 of {PLAN})",
                    self.def.record(record).name
                ),
            ));
        }
        Ty::of_a_value(ty, "a kernel `fn` returns")
    }

    pub(super) fn param_type(&self, param: &Param, role: Role) -> syn::Result<Ty> {
        if let Some(record) = self.def.record_named(&param.ty) {
            return Ok(Ty::Record(record));
        }
        match FunctionType::of(&param.ty) {
            FunctionType::Kernel => return Ok(Ty::Kernel),
            FunctionType::Misspelled(why) => {
                return Err(syn::Error::new_spanned(
                    &param.ty,
                    format!(
                        "a kernel-typed parameter is `impl Fn(f32, f32) -> f32`, and this one is \
                         spelled otherwise\n\
                         \n\
                         note: {why}\n\
                         note: a kernel is passed by the host at run time and applied by name, \
                         `k(x, y)` (Phase D-a of {PLAN})"
                    ),
                ));
            }
            FunctionType::NotOne => {}
        }
        if let Some(name) = bare_name(&param.ty).filter(|_| Ty::from_syn(&param.ty).is_none()) {
            return Err(syn::Error::new_spanned(
                &param.ty,
                format!(
                    "a kernel parameter is an `f32` or a `bool`, or one of this block's \
                     records, and `{name}` is none of them\n\
                     \n\
                     note: a block sees only its own items: a record declared in another \
                     `kernel!` block, or a Rust struct, is not in scope (D6 of {PLAN})"
                ),
            ));
        }
        if Ty::from_syn(&param.ty) == Some(Ty::Usize) {
            return match (role, &self.def) {
                (Role::Entry, KernelDef::Items { .. }) => Ok(Ty::Usize),
                (Role::Entry, KernelDef::Closure(_)) => Err(syn::Error::new_spanned(
                    &param.ty,
                    "a count parameter `n: usize` belongs to an item entry; the closure form binds only `f32` values",
                )),
                (Role::Helper, _) => Err(syn::Error::new_spanned(
                    &param.ty,
                    "a helper's parameter cannot be a count; only an item entry takes `n: usize`",
                )),
            };
        }
        let ty = Ty::of_a_value(&param.ty, "a kernel parameter is")?;
        match (role, ty) {
            (Role::Entry, Ty::Bool) => Err(syn::Error::new_spanned(
                &param.ty,
                "an entry's parameter is an `f32` or a record\n\
                 \n\
                 note: an entry's parameters are its uniforms, bound per call: an `f32` is one, \
                 a record is one per field, and a uniform is a number\n\
                 help: take the mask's operands as parameters and compare them in the body",
            )),
            _ => Ok(ty),
        }
    }
}
