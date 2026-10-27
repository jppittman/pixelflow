//! # PixelFlow Kernel Compiler Frontend
//!
//! A compiler frontend for the PixelFlow DSL, implemented as Rust proc-macros.
//!
//! ## Architecture
//!
//! ```text
//! Source (macro input)
//!     │
//!     ▼ Parser (parser/)
//! AST (ast/)
//!     │
//!     ▼ Semantic Analysis (sema/)
//! Analyzed AST
//!     │
//!     ▼ Arena lowering (lower/)
//! ExprArena
//!     │
//!     ▼ `impl Optimize`  — `kernel!` saturates, `kernel_raw!` is `Identity`
//! ExprArena
//!     │
//!     ▼ Emission (emit/)
//! Rust TokenStream that rebuilds a `Kernel` at load time
//! ```
//!
//! An entry that takes a kernel-typed parameter leaves the diagram at
//! lowering: its argument exists only when its host function is called, so
//! lowering's steps are emitted as the statements that take them, and run
//! then (`emit::Staged`, Phase D-a of
//! docs/language/2026-09-25-the-language-is-kernel.md).
//!
//! Two representations, the surface AST and the IR, and optimization runs on
//! the IR. Every further representation is a boundary where two stages can
//! disagree about what the language is, each disagreement a stage accepting
//! what a later one refuses; so no stage re-synthesizes an AST from an
//! optimized term. See docs/archive/plans/2026-09-08-macro-tier-is-arena-native.md.
//!
//! There is **one backend**, and it produces a `pixelflow_core::Kernel` — an
//! arena fragment, the language's own value. Nothing is compiled at
//! macro-expansion time and nothing is compiled at construction: a `Kernel`
//! becomes machine code when a consumer compiles it at a lattice's shape and
//! collapses it (`Lattice::bake`), which is the only way a kernel turns into
//! numbers.
//!
//! So there are two macros, and the only difference between them is the
//! [`Optimize`] value they hand the same `expand`:
//! [`kernel!`](macro@kernel) saturates, [`kernel_raw!`](macro@kernel_raw)
//! passes [`Identity`]. Not optimizing is a value here, not a branch that
//! declines to call a function, which is what that type exists to say.

mod ast;
mod emit;
mod lower;
mod parser;
mod sema;
#[cfg(test)]
mod tests;

use pixelflow_ir::OpKind;
use pixelflow_ir::arena::{ExprArena, ExprId, ExprNode};
use pixelflow_ir::optimize::{Identity, Optimize, Rewritten};
use pixelflow_search::Saturate;
use proc_macro::TokenStream;

/// The `kernel!` macro: the language, as a block of items or as a closure. An
/// entry with no kernel-typed parameter and no `Dwrt` is
/// saturated at macro-expansion time; any other entry is optimized when it is
/// baked (see *Pipeline*).
///
/// # The items form
///
/// A block of records, `const` items and `fn` items
/// (docs/language/2026-09-25-the-language-is-kernel.md §1.2):
///
/// - A `pub fn name(params) -> f32 { body }` is an **entry**: the macro
///   emits a host `pub fn name(params) -> Kernel`, taking its parameters by
///   their declared types — a kernel-typed one as a `&Kernel` — and, when
///   it has uniform parameters and no kernel-typed one, its `Args` record
///   (see *Binding times* below).
/// - A private `fn` is a **helper**: type-checked once, inlined at each call.
///   Helpers may call helpers; a cycle is refused, because the language is a
///   DAG. `X` and `Y` appear only in entries — a helper takes its
///   coordinates as arguments, so that applying it to a shifted coordinate
///   warps it.
/// - A `struct R { a: f32, b: f32 }` is a **record** (§1.3): named `f32`
///   fields, emitted as a host `#[repr(C)]` struct of the same name and
///   visibility, its attributes kept. A record is the type of an entry's or
///   a helper's parameter; a body reads a field, `p.a`, aliases a record,
///   `let q = p;`, and passes one on by name. Building, returning, choosing
///   between or computing with whole records is Phase D (D7), and refused.
/// - A `const NAME: f32 = expr;` is evaluated at expansion, per operation in
///   `f32`, from literals, other consts, `+ - * /`, unary `-` and
///   parentheses. A `const NAME: usize = expr;` is a count, evaluated from
///   integers, other `usize` consts and `+ - * /`, each operation checked. A
///   `pub const` is also emitted as a host `pub const`. A constant of a
///   program is spelled this way, and no other.
///
/// ```no_run
/// use pixelflow_compiler::kernel;
/// use pixelflow_core::{Kernel, Lattice};
///
/// kernel! {
///     pub const UNIT: f32 = 1.0;
///     const RINGS: usize = 4;
///
///     /// The distance from `(cx, cy)`; a function of its arguments.
///     fn dist(x: f32, y: f32, cx: f32, cy: f32) -> f32 {
///         let dx = x - cx;
///         let dy = y - cy;
///         (dx * dx + dy * dy).sqrt()
///     }
///
///     /// The signed distance to the circle: the entry reads `X` and `Y`.
///     pub fn circle(cx: f32, cy: f32, r: f32) -> f32 {
///         dist(X, Y, cx, cy) - r
///     }
///
///     /// One inside the unit disc, zero outside; a choice is spelled `if`.
///     pub fn disc(cx: f32, cy: f32) -> f32 {
///         if dist(X, Y, cx, cy) < UNIT { UNIT } else { 0.0 }
///     }
///
///     /// How many of the discs about the origin of radii 1 to `RINGS`
///     /// contain the sample: a fold, Σ over `i ∈ [0, RINGS)`.
///     pub fn rings() -> f32 {
///         (0..RINGS)
///             .map(|i| if dist(X, Y, 0.0, 0.0) < (i as f32) + UNIT { UNIT } else { 0.0 })
///             .sum()
///     }
/// }
///
/// let unit_circle: Kernel = circle(0.0, 0.0, UNIT);
/// let plane = Lattice::frame(64, 64).bake(&unit_circle);
/// ```
///
/// # Types
///
/// Every expression is an `f32` or a `bool`. A comparison (`<`, `<=`, `>`,
/// `>=`, `==`, `!=`) gives a `bool`; `&` and `|` combine two `bool`s; an
/// `if c { a } else { b }` chooses by one, and both arms have the same type.
/// A `bool` where an `f32` is expected, or the reverse, is a type error at
/// expansion: `X.select(Y, 7.0)` would blend a number as a mask. The IR
/// keeps one lane for both; the type lives in the front end.
///
/// # Folds
///
/// `(a..b).map(|i| e).sum()` is Σ of `e` over `i ∈ [a, b)`; `.product()`,
/// `.fold(f32::INFINITY, f32::min)` and `.fold(f32::NEG_INFINITY, f32::max)`
/// are Π, min and max, and `(a..b).any(|i| m)` and `.all(|i| m)` are ∃ and ∀
/// of `bool`s. An empty range gives the monoid's identity, up to the
/// optimizer's algebraic rules, which are not sound at NaN or infinity
/// (Linear JP-12). Each bound is a
/// `usize`: an integer, a `usize` const or `+ - * /` over them, known at
/// expansion; an enclosing fold's index, for a triangular `(0..i)`; or an
/// entry's count parameter, `n: usize`, bound per call. A bound that is a
/// name the program binds stands alone — `0..i + 1` waits for bounds typed
/// as integers — and no bound reads `X` or `Y`. The index `i` is a `usize`,
/// which a body reads only as `i as f32`: there is no arithmetic on an index
/// and nothing to index. A fold lowers to one `Reduce`, the node
/// `Kernel::over` builds, or `Kernel::over_values` for a bound that is a
/// value; unrolling it is the e-graph's choice, at bake time.
/// A closure is the body of a fold and appears nowhere else.
///
/// `if` is the choice. `.select(a, b)` lowers to the same node; D15 of the
/// plan removes the spelling.
///
/// A `let` binds a name, or takes a tuple apart where it is written:
/// `let (x, y) = (X + 0.5, Y + 0.5);` binds each name to its expression, every
/// expression read before any name binds, as Rust's does. A tuple anywhere
/// else is a value, which is Phase D (D7).
///
/// # No collection types
///
/// The language has none (§1.3, §1.6): no array, slice, list or table — as
/// a parameter, a value, or a thing to index or iterate. Several values are
/// several parameters, a record is its named fields, and the one iteration
/// is a fold, over a range. Each spelling of a collection — `[R; N]`,
/// `&[R]`, `[a, b]`, `v[k]`, `.into_iter()`, `.iter()`, `.map` over anything
/// but a range — is refused where it is written.
///
/// # The closure form
///
/// ```text
/// kernel!(|param1: f32, param2: f32, ...| expression)
/// ```
///
/// Sugar for a block with one entry, whose type is inferred, and the
/// expansion is an expression rather than an item:
///
/// - Zero params → a `Kernel` value.
/// - N params → a closure `move |p0: f32, ...| -> Kernel`, every parameter a
///   uniform, as an entry's are. It has no `Args` record: a program compiled
///   from it is rebound by position, `block.set_declared([p0, ...])`.
///
/// The closure form takes no kernel-typed parameter: rustc refuses
/// `impl Trait` in a closure's parameters (E0562).
///
/// ```no_run
/// # use pixelflow_compiler::kernel;
/// # use pixelflow_core::{Kernel, Lattice};
/// let circle = kernel!(|cx: f32, cy: f32, r: f32| {
///     let dx = X - cx;
///     let dy = Y - cy;
///     (dx * dx + dy * dy).sqrt() - r
/// });
///
/// let unit_circle: Kernel = circle(0.0, 0.0, 1.0);
/// let plane = Lattice::frame(64, 64).bake(&unit_circle);
/// ```
///
/// The five derivative projections (`DX`, `DY`, `DXX`, `DXY`, `DYY`) lower to
/// `Dwrt` chains (`V` is the value itself: `V(e)` is `e`, and lowers to no
/// node) that the macro tier never resolves: a kernel carrying one is not saturated
/// at expansion, and the runtime tier lowers it when the kernel is baked,
/// after composition, so a `Kernel::at` warp reaches the differentiated
/// operand (`tests/derivative_under_warp.rs`).
///
/// # Kernel-typed parameters
///
/// An entry's parameter typed `impl Fn(f32, f32) -> f32` is a **kernel**
/// the host passes when it calls the entry (§1.3, §1.4, Phase D-a), and the
/// host function takes a `&Kernel` for it. A body applies one, `k(x, y)` —
/// the argument at `(x, y)`, which is `k.at(x, y)`: application is
/// contramap — or passes it by name to a helper's kernel-typed parameter,
/// and does nothing else with it: arithmetic on one, a `let` of one, a
/// return of one are refused. As rustc moves an `impl Fn` it passes, a body
/// passes one on at most once on each path through it — an `if`'s arms are
/// two — then uses it no more on that path, and never inside a fold's body.
///
/// Such an entry is a composition, whose program exists only once the host
/// passes its argument, so its host function runs lowering's steps when it
/// is called: the IR calls this macro makes at expansion for any other
/// entry, then. No optimizer runs on it at expansion; the composed program
/// is optimized when it is baked. A fold around an application closes after
/// the argument is in, so the argument keeps its binders and the fold takes
/// another (D4). An argument is closed and reads no table, or the call
/// panics.
///
/// ```no_run
/// # use pixelflow_compiler::kernel;
/// kernel! {
///     pub fn sum2(a: impl Fn(f32, f32) -> f32, b: impl Fn(f32, f32) -> f32) -> f32 {
///         a(X, Y) + b(X, Y)
///     }
///
///     /// Three copies of `k`, a column apart.
///     pub fn columns(k: impl Fn(f32, f32) -> f32) -> f32 {
///         (0..3).map(|i| k(X + (i as f32), Y)).sum()
///     }
/// }
///
/// let disc = kernel!(|r: f32| (X * X + Y * Y).sqrt() - r)(4.0);
/// let both = sum2(&disc, &columns(&disc));
/// ```
///
/// # Binding times
///
/// Every value a program reads is bound at its use (§1.4):
///
/// - **Uniform**: every scalar and count parameter. An `f32` or `usize`
///   count is one uniform and a record is one per field; the kernel an entry
///   returns declares them in that order,
///   each with the call's value as its default, and baking it draws the
///   call. The value is an argument of the program, never folded into it, so
///   every call of an entry is one program: compiled once, and rebound per
///   call from the entry's `Args` record — `<Entry in UpperCamelCase>Args`,
///   its parameters as fields, written into a block the program made with
///   `write_into`, which allocates nothing once the block is the caller's
///   alone. A kernel the entry's is composed into still rebinds from it, its
///   declarations the entry's in order beside argument-free kernels; beside
///   other arguments the count differs, and `write_into` refuses. A
///   constant is a `const` item.
/// - **Kernel-typed**: an argument the host passes when it calls the entry.
///   The composed program declares the entry's own uniforms first, in
///   declaration order, then each argument's, in parameter order and that
///   argument's own order, read or not — an instance passed twice declared
///   once, where it first appears. Its key is the composed term's. It has
///   no `Args` record: binding a composed program is O3 of the plan.
///
/// ```no_run
/// use pixelflow_compiler::kernel;
/// use pixelflow_core::{Lattice, Manifold};
///
/// kernel! {
///     /// An axis-aligned box: a record.
///     pub struct Bounds { pub x0: f32, pub y0: f32, pub x1: f32, pub y1: f32 }
///
///     /// `count` rings about the box's corner, `fg` on `bg`.
///     pub fn rings(count: usize, b: Bounds, fg: f32, bg: f32) -> f32 {
///         let dx = X - b.x0;
///         let dy = Y - b.y0;
///         let r = (dx * dx + dy * dy).sqrt();
///         let within = (X <= b.x1) & (Y <= b.y1);
///         let n: f32 = (0..count).map(|i| if r < (i as f32) + 1.0 { 1.0 } else { 0.0 }).sum();
///         if within { fg * n / (count as f32) } else { bg }
///     }
/// }
///
/// let lattice = Lattice::frame(64, 64);
/// let b = Bounds { x0: 8.0, y0: 8.0, x1: 40.0, y1: 40.0 };
/// let once = lattice.bake(&rings(4, b, 1.0, 0.0));            // one call, baked
///
/// let program = Manifold::compile(&rings(4, b, 1.0, 0.0), lattice.extent);
/// let mut block = program.block();
/// RingsArgs { count: 4, b, fg: 0.5, bg: 0.25 }.write_into(&mut block)?;
/// let again = lattice.collapse(&program.bind(&[]).with_uniforms(&block));
/// # Ok::<(), pixelflow_core::ArityMismatch>(())
/// ```
///
/// # Pipeline
///
/// 1. **Parser**: items or closure syntax → AST
/// 2. **Semantic analysis**: symbol resolution, types, `const` evaluation,
///    the call graph
/// 3. **Arena lowering**: each entry's body becomes an `ExprArena`, helpers
///    inlined, its parameters declared as uniforms
/// 4. **Optimization**: e-graph saturation + latency-prior extraction, on
///    the arena. A kernel carrying a `Dwrt` declines here and is optimized
///    at bake time instead, so composition still gets the chain rule.
/// 5. **Emission**: the arena becomes code that rebuilds it at load time,
///    the call's values as its uniforms' defaults
///
/// An entry that takes a kernel skips 3 to 5 at expansion: its host
/// function is lowering's steps, emitted as the statements that take them,
/// and builds its arena when it is called (Phase D-a).
#[proc_macro]
pub fn kernel(input: TokenStream) -> TokenStream {
    expand(input, &mut macro_tier())
}

/// The `kernel_raw!` macro: like [`kernel!`](macro@kernel) but **without**
/// e-graph optimization, so the emitted arena has the shape that was written.
///
/// # Use Cases
///
/// - Benchmarking an exact expression form: `X * Y + z` against
///   `X.mul_add(Y, z)`, which `kernel!` would fuse into the same node.
/// - Debugging: what the front end built, before anything rewrote it.
///
/// # Example
///
/// ```no_run
/// use pixelflow_compiler::kernel_raw;
///
/// // Two different arenas — mul then add, against one MulAdd.
/// let unoptimized = kernel_raw!(|z: f32| X * Y + z)(0.5);
/// let explicit_fma = kernel_raw!(|z: f32| X.mul_add(Y, z))(0.5);
/// ```
#[proc_macro]
pub fn kernel_raw(input: TokenStream) -> TokenStream {
    expand(input, &mut Identity)
}

/// The macro tier's optimizer: equality saturation over the template
/// vocabulary, under the same production policy the runtime tier uses.
///
/// It deliberately does **not** include `LowerDwrt`. A `Dwrt` node left
/// intact is what makes the chain rule work under composition — `Kernel::at`
/// warps by substituting into `Var` leaves, so the warp reaches a surviving
/// `Dwrt`'s operand and differentiates the warped function. Resolving
/// derivatives here would be a miscompilation under composition, and
/// `tests/derivative_under_warp.rs` pins that. The runtime tier lowers them at
/// bake time, after composition, in `legalize`'s order.
fn macro_tier() -> impl Optimize {
    DwrtFree(Saturate::macro_tier())
}

/// Run `inner`, unless the term carries a `Dwrt`.
///
/// Saturation would resolve one — the chain rule is in the rule set, and a
/// `Dwrt` node is priced so the extractor never keeps it — which at expansion
/// time is the miscompilation `macro_tier` describes. It declines the *whole*
/// kernel rather than the `Dwrt` subterm, so a kernel carrying one reaches the
/// runtime tier unfused and is saturated once, there. Saturation with the
/// derivative rules withheld would fuse it without resolving anything; the
/// search crate exposes no such rule set.
struct DwrtFree<P>(P);

impl<P: Optimize> Optimize for DwrtFree<P> {
    fn optimize(&mut self, arena: &ExprArena, root: ExprId) -> Rewritten {
        let carries_dwrt = arena
            .nodes()
            .any(|(_, n)| matches!(n, ExprNode::Binary(OpKind::Dwrt, _, _)));
        if carries_dwrt {
            return Rewritten::Declined;
        }
        self.0.optimize(arena, root)
    }
}

/// Parse, analyze, lower, optimize, emit — the whole front end, over
/// `proc_macro2` tokens so that a test can run the pipeline the macros run.
/// The macros differ only in the optimizer they hand it.
///
/// A refusal is a span and a claim, whichever stage makes it.
fn front_end(
    input: proc_macro2::TokenStream,
    optimizer: &mut dyn Optimize,
) -> syn::Result<proc_macro2::TokenStream> {
    let analyzed = parser::parse(input).and_then(sema::analyze)?;
    emit::emit_kernel(&analyzed, optimizer)
}

/// [`front_end`] as a procedural macro: a refusal is the `compile_error!`
/// at its span.
fn expand(input: TokenStream, optimizer: &mut dyn Optimize) -> TokenStream {
    front_end(input.into(), optimizer)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}
