//! `ExprArena` → the `TokenStream` that rebuilds it at load time.
//!
//! The back end, and there is one. It produces a `pixelflow_core::Kernel` — an
//! arena fragment, the language's own value. Nothing is compiled at
//! macro-expansion time and nothing is compiled at construction: a `Kernel`
//! becomes machine code when a consumer compiles it at a lattice's shape and
//! collapses it (`Lattice::bake`), which is the only way a kernel turns into
//! numbers.
//!
//! Two steps, because the interesting thing happens between them: an arena is
//! lowered from the AST ([`crate::lower`]), *then* optionally rewritten, then
//! emitted. Emission takes an arena rather than an AST so that the optimizer
//! has somewhere to stand.
//!
//! What is emitted depends on how the block was spelled
//! ([`KernelDef`](crate::ast::KernelDef)). The closure form is an expression:
//! a `Kernel`, or a closure over its parameters' `f32`s returning one. The
//! items form is items: a host `#[repr(C)]` struct per record, a host `const`
//! per `pub const`, and per entry a host `fn` returning a `Kernel` and, when
//! it has uniform parameters and no kernel-typed one, its `Args` record. A
//! helper is inlined and a private `const` is folded, so neither leaves a
//! trace.
//!
//! **An entry that takes a kernel is staged** (plan Phase D-a). Its
//! argument, `k: impl Fn(f32, f32) -> f32`, is a `&Kernel` the host passes
//! when it calls the entry's host function, so the program does not exist
//! at expansion: the host function *is* lowering, run then.
//! [`Staged`](staged::Staged) is the [`Site`](crate::lower::Site) that writes
//! each of lowering's steps as the statement that takes it — the same IR
//! call [`Expansion`](crate::lower::Expansion) makes, in the same order —
//! and `k(x, y)` is one more, [`ExprArena::apply`]. No optimizer runs at
//! expansion on such an entry, as on a template: the composed program is
//! optimized when it is baked (§1.8).
//!
//! [`ExprArena::apply`]: pixelflow_ir::arena::ExprArena::apply
//!
//! **Binding times** (docs/language/2026-09-25-the-language-is-kernel.md §1.4).
//! A host function takes its parameters by their declared types, and every
//! one is a uniform: the kernel it returns declares one uniform per scalar —
//! a record parameter, one per field — in declaration order, with the call's
//! value as its default. So the call's values are the kernel's arguments,
//! never its constants: every call of an entry is one program, keyed and
//! compiled once, and a compiled program is rebound per call from the
//! entry's `Args` record. A per-call count is a uniform and may bound a fold
//! without making another program.
//!
//! # Contract
//!
//! One function, the stage's: the tokens that rebuild an analyzed block's
//! kernels at load time, or lowering's refusal. It is `pub(crate)` where it is
//! implemented (`host.rs`) and re-exported here; every other item under
//! `emit/` is the implementation and is `pub(super)` or private — the host
//! items (`host.rs`), the arena's replay and the snippets every statement is
//! spelled with (`arena.rs`), and `Staged`, the second `Site` (`staged.rs`),
//! which is emission's because it writes `::pixelflow_core::__macro` paths
//! with emission's snippets.
//!
//! ```ignore
//! /// Implemented by `emit::emit_kernel`.
//! trait Emit {
//!     /// The tokens that rebuild `analyzed`'s kernels at load time. Closure form: an expression, a
//!     /// `Kernel` or `move |p: f32, …| -> Kernel`. Items form: a `#[repr(C)]` struct per record, a
//!     /// host `const` per `pub const`, per entry a `#[must_use]` host `fn` (a kernel param as
//!     /// `&Kernel`) and, when `has_args_record`, its `Args`
//!     /// record with `write_into`. Every uniform is declared with the call's value as its default,
//!     /// in `parameters` order. `optimizer` is offered exactly the entries with no
//!     /// kernel-typed parameter; `Declined`/`Unchanged` keep the lowered arena, `Changed` is
//!     /// relinked to declaration order; an entry taking a kernel is `Staged`. No `::std` path.
//!     /// Deterministic: equal input and optimizer answer, byte-equal tokens. Err: lowering's.
//!     fn emit_kernel(analyzed: &AnalyzedKernel, optimizer: &mut dyn Optimize) -> syn::Result<TokenStream>;
//! }
//! ```

mod arena;
mod host;
mod staged;

pub(crate) use host::emit_kernel;
