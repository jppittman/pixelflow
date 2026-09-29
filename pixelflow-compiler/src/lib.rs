//! # PixelFlow Kernel Compiler Frontend
//!
//! A compiler frontend for the PixelFlow DSL, implemented as Rust proc-macros.
//!
//! ## Architecture
//!
//! ```text
//! Source (macro input)
//!     │
//!     ▼ Parser (parser.rs)
//! AST (ast.rs)
//!     │
//!     ▼ Semantic Analysis (sema.rs)
//! Analyzed AST + Symbol Table
//!     │
//!     ▼ Arena lowering (lower.rs)
//! ExprArena
//!     │
//!     ▼ `impl Optimize`  — `kernel!` saturates, `kernel_raw!` is `Identity`
//! ExprArena
//!     │
//!     ▼ Emission (emit.rs)
//! Rust TokenStream that rebuilds a `Kernel` at load time
//! ```
//!
//! Two representations, the surface AST and the IR. It used to be five: the
//! optimizer ran
//! on the *AST*, so `kernel!` went AST → e-graph → extracted DAG → back to an
//! AST nothing had written (synthesized `let` bindings naming shared
//! subexpressions, opaque placeholder identifiers standing in for terms the
//! e-graph could not hold) → and only then to the arena the e-graph had
//! already built and thrown away. Each of those boundaries is a place two
//! stages can disagree about what the language is, and three such
//! disagreements were found in one week — every one of them a stage accepting
//! what a later stage refused. See
//! docs/plans/2026-09-08-macro-tier-is-arena-native.md.
//!
//! There is **one backend**, and it produces a [`Kernel`] — an arena fragment,
//! the language's own value. Nothing is compiled at macro-expansion time and
//! nothing is compiled at construction: a `Kernel` becomes machine code when a
//! consumer compiles it at a lattice's shape and collapses it
//! (`Lattice::bake`), which is the only way a kernel turns into numbers.
//!
//! So there are two macros, and the only difference between them is the
//! [`Optimize`] value they hand the same `expand`:
//! [`kernel!`](macro@kernel) saturates, [`kernel_raw!`](macro@kernel_raw)
//! passes [`Identity`]. Not optimizing is a value here, not a branch that
//! declines to call a function, which is what that type exists to say.
//!
//! [`Kernel`]: pixelflow_core::Kernel

mod ast;
mod emit;
mod lower;
mod parser;
mod sema;
mod symbol;

use pixelflow_ir::OpKind;
use pixelflow_ir::arena::{ExprArena, ExprId, ExprNode};
use pixelflow_ir::optimize::{Identity, Optimize, Rewritten};
use pixelflow_search::Saturate;
use proc_macro::TokenStream;

/// The plan that owns the constructs the front end refuses by phase: a
/// refusal of one names the phase that brings it.
pub(crate) const PLAN: &str = "docs/plans/2026-09-25-the-language-is-kernel.md";

/// The `kernel!` macro: the language, as a block of items or as a closure,
/// optimized by the e-graph at macro-expansion time.
///
/// # The items form
///
/// A block of records, `const` items and `fn` items
/// (docs/plans/2026-09-25-the-language-is-kernel.md §1.2):
///
/// - A `pub fn name(params) -> f32 { body }` is an **entry**: the macro
///   emits a host `pub fn name(params) -> Kernel`, taking its parameters by
///   their declared types, and — when it has parameters — its `Args` record
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
/// ```ignore
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
/// expansion: `X.select(Y, 7.0)` used to blend a number as a mask. The IR
/// keeps one lane for both; the type lives in the front end.
///
/// # Folds
///
/// `(a..b).map(|i| e).sum()` is Σ of `e` over `i ∈ [a, b)`; `.product()`,
/// `.fold(f32::INFINITY, f32::min)` and `.fold(f32::NEG_INFINITY, f32::max)`
/// are Π, min and max, and `(a..b).any(|i| m)` and `.all(|i| m)` are ∃ and ∀
/// of `bool`s. An empty range gives the monoid's identity. The bounds are
/// constant — integers, `usize` consts and an entry's structural parameters
/// — and the index `i` is a `usize`, which a body reads only as `i as f32`:
/// there is no arithmetic on an index and nothing to index. A fold lowers to
/// one `Reduce`, the node `Kernel::over` builds; unrolling it is the
/// e-graph's choice, at bake time. A closure is the body of a fold or of a
/// family's iteration and appears nowhere else.
///
/// `if` is the choice. `.select(a, b)` still lowers to the same node this
/// phase, and Phase B of the plan removes it.
///
/// A `let` binds a name, or takes a tuple apart where it is written:
/// `let (x, y) = (X + 0.5, Y + 0.5);` binds each name to its expression, every
/// expression read before any name binds, as Rust's does. A tuple anywhere
/// else is a value, which is Phase D (D7).
///
/// # Families
///
/// An entry's parameter may be a family, `name: [R; N]`: `N` elements of one
/// of the block's records or of `f32`s, `N` a structural parameter, an
/// integer or a `usize` const (§1.6). A family is `N` elements' uniforms at
/// static slots, element-major, and not a table: nothing indexes it,
/// measures it, passes it on, compares it or returns it. It is iterated as a
/// whole, as Rust iterates an array by value —
/// `name.into_iter().map(|p| e).sum()`, `.product()`, the two `.fold`s,
/// `name.into_iter().any(|p| m)` and `.all(|p| m)` — which is ⊕ of
/// `e[p := element k]` over the elements, the monoid's identity when there
/// are none. The program holds
/// one copy of the body per element, made when the host function is
/// instantiated: at `N = 3` it is `e[p₀] + e[p₁] + e[p₂]`, the copies
/// written out, with no fold, no binder and no index. A helper takes one
/// element; the entry's `Args` record holds the family as its array.
///
/// ```ignore
/// use pixelflow_compiler::kernel;
/// use pixelflow_core::{Lattice, Manifold};
///
/// kernel! {
///     /// A disc: its centre and its radius.
///     pub struct Disc { pub cx: f32, pub cy: f32, pub r: f32 }
///
///     fn inside(d: Disc, x: f32, y: f32) -> f32 {
///         let (dx, dy) = (x - d.cx, y - d.cy);
///         if dx * dx + dy * dy < d.r * d.r { 1.0 } else { 0.0 }
///     }
///
///     /// How many of the discs cover the sample: `N` is the program, and
///     /// the discs are its uniforms.
///     pub fn cover<const N: usize>(discs: [Disc; N]) -> f32 {
///         discs.into_iter().map(|d| inside(d, X, Y)).sum()
///     }
/// }
///
/// let lattice = Lattice::frame(8, 8);
/// let a = Disc { cx: 3.0, cy: 3.0, r: 2.0 };
/// let b = Disc { cx: 5.0, cy: 3.0, r: 2.0 };
/// let once = lattice.bake(&cover([a, b]));                      // N = 2, inferred
///
/// let program = Manifold::compile(&cover([a, b]), lattice.extent);
/// let mut block = program.block();
/// CoverArgs { discs: [b, Disc { r: 3.0, ..a }] }.write_into(&mut block)?;
/// let again = lattice.collapse(&program.bind(&[]).with_uniforms(&block));
/// ```
///
/// # The closure form
///
/// ```ignore
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
/// ```ignore
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
/// Kernels compose as values — `Kernel::at`/`sum`/`select`/arithmetic — so
/// there is no manifold-typed parameter. Derivatives (`DX`/`DY`) become
/// symbolic `Dwrt` nodes, resolved by the e-graph here when it can and by
/// codegen otherwise.
///
/// # Binding times
///
/// Every value a program reads is bound at one of two times (§1.4):
///
/// - **Structural**: an entry's `const N: usize` generics. The host function
///   is generic over them, and each value is its own program — a count in a
///   fold's range, or `N as f32`. A helper takes none; it reads its entry's
///   through an argument.
/// - **Uniform**: every parameter. An `f32` is one uniform and a record is
///   one per field; the kernel an entry returns declares them in that order,
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
///
/// ```ignore
/// use pixelflow_compiler::kernel;
/// use pixelflow_core::{Lattice, Manifold};
///
/// kernel! {
///     /// An axis-aligned box: a record.
///     pub struct Bounds { pub x0: f32, pub y0: f32, pub x1: f32, pub y1: f32 }
///
///     /// `N` rings about the box's corner, `fg` on `bg`.
///     pub fn rings<const N: usize>(b: Bounds, fg: f32, bg: f32) -> f32 {
///         let dx = X - b.x0;
///         let dy = Y - b.y0;
///         let r = (dx * dx + dy * dy).sqrt();
///         let within = (X <= b.x1) & (Y <= b.y1);
///         let n: f32 = (0..N).map(|i| if r < (i as f32) + 1.0 { 1.0 } else { 0.0 }).sum();
///         if within { fg * n / (N as f32) } else { bg }
///     }
/// }
///
/// let lattice = Lattice::frame(64, 64);
/// let b = Bounds { x0: 8.0, y0: 8.0, x1: 40.0, y1: 40.0 };
/// let once = lattice.bake(&rings::<4>(b, 1.0, 0.0));            // one call, baked
///
/// let program = Manifold::compile(&rings::<4>(b, 1.0, 0.0), lattice.extent);
/// let mut block = program.block();
/// RingsArgs::<4> { b, fg: 0.5, bg: 0.25 }.write_into(&mut block)?;
/// let again = lattice.collapse(&program.bind(&[]).with_uniforms(&block));
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
///    at bake time instead, so composition still gets the chain rule; so is
///    an entry with structural parameters or a family, a template until it
///    is instantiated.
/// 5. **Emission**: the arena becomes code that rebuilds it at load time,
///    the call's values as its uniforms' defaults, and a family's
///    iterations the copies of their bodies, made as the host function runs
#[proc_macro]
pub fn kernel(input: TokenStream) -> TokenStream {
    expand(input, &mut macro_tier())
}

/// The `kernel_raw!` macro: like [`kernel!`](macro@kernel) but **without**
/// e-graph optimization, so the emitted arena has the shape that was written.
///
/// # Use Cases
///
/// - Benchmarking an exact expression form: `X * Y + Z` against
///   `(X).mul_add(Y, Z)`, which `kernel!` would fuse into the same node.
/// - Corpus generation, where the input to the optimizer is the subject.
/// - Debugging: what the front end built, before anything rewrote it.
///
/// # Example
///
/// ```ignore
/// // Two different arenas — mul then add, against one MulAdd.
/// let unoptimized = kernel_raw!(|| X * Y + Z);
/// let explicit_fma = kernel_raw!(|| (X).mul_add(Y, Z));
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
/// derivatives here instead was a miscompilation, and `tests/
/// derivative_under_warp.rs` pins that. The runtime tier lowers them at bake
/// time, after composition, in `legalize`'s order.
fn macro_tier() -> impl Optimize {
    DwrtFree(Saturate::macro_tier())
}

/// Run `inner`, unless the term carries a `Dwrt`.
///
/// Saturation would resolve one — the chain rule is in the rule set, and a
/// `Dwrt` node is priced so the extractor never keeps it — which at expansion
/// time is exactly the miscompilation above. Measured, not assumed: dropping
/// this wrapper puts `derivative_under_warp.rs` back to 12 where the chain
/// rule says 24.
///
/// It declines the *whole* kernel rather than the `Dwrt` subterm, and that
/// costs something. For `'8'` at 17 px the bake-time arena goes from 2964
/// nodes to 3318 — 12% — because the glyph's fragments now reach the runtime
/// tier unfused and one saturation does not recover what two did. Correctness
/// is not negotiable against 12%, so this ships, but the price is real and it
/// is not the floor: what this kernel actually wants is saturation with the
/// derivative rules *withheld*, which would fuse it without resolving
/// anything. That needs a rule-set the search crate does not currently
/// expose, so it is denoted here and not built.
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

/// Parse, analyze, lower, optimize, emit — the whole front end. The macros
/// differ only in the optimizer they hand it.
fn expand(input: TokenStream, optimizer: &mut dyn Optimize) -> TokenStream {
    let tokens = proc_macro2::TokenStream::from(input);
    let analyzed = match parser::parse(tokens).and_then(sema::analyze) {
        Ok(a) => a,
        Err(e) => return e.to_compile_error().into(),
    };
    match emit::emit_kernel(&analyzed, optimizer) {
        Ok(tokens) => tokens.into(),
        Err(e) => syn::Error::new(proc_macro2::Span::call_site(), e)
            .to_compile_error()
            .into(),
    }
}

/// Every method name the front end advertises must survive both macros.
///
/// This class of bug shipped once. `sema` accepted `.round()`, `.log10()`
/// and `.pow()` — ordinary `OpKind`s, so `known_method_names()` returned
/// them — while arena lowering had no arm for any of the three; and the
/// mirror-image gap hid `fract`/`hypot`/`clamp`, whose e-graph decomposition
/// `sema` rejected the names for, leaving it unreachable from either macro.
/// Both halves are a disagreement *between pipeline stages*, so no test of a
/// single stage can see them: the stage under test is the one that is right.
///
/// Nor can sampling see them. A method is exercised only if some test
/// happens to write a kernel calling it, and `kernel_macro.rs`'s cases are
/// hand-picked — every one of those six was already covered at the SIMD
/// backend, in codegen, and on the `Kernel` value API, and still nobody had
/// written `X.round()` inside a `kernel!` body.
///
/// So run the real pipeline, both macros' versions of it, over the whole
/// advertised surface. Adding an op to `OpKind::is_dsl_method` or a name to
/// `LIBRARY_METHODS` without a path through every stage fails here, for
/// whoever adds it.
#[cfg(test)]
mod every_advertised_method_compiles {
    use crate::lower::LIBRARY_METHODS;
    use crate::sema::{MethodTyping, method_typing};
    use crate::{Identity, Optimize, emit, macro_tier, parser, sema};
    use pixelflow_ir::{OpKind, known_method_names};
    use proc_macro2::Span;
    use quote::quote;
    use syn::Ident;

    /// Which macro's pipeline to run. `kernel!` saturates the e-graph between
    /// sema and lowering; `kernel_raw!` goes straight across. The difference
    /// between them is where the `hypot`/`fract` asymmetry lived, so both are
    /// swept.
    #[derive(Clone, Copy, Debug)]
    enum Macro {
        Kernel,
        KernelRaw,
    }

    /// Expand a well-typed call of `method` through `which` macro's own
    /// pipeline — the same calls [`kernel`] and [`kernel_raw`] make — and
    /// report whether it yields code.
    ///
    /// Well-typed by `sema`'s own typing of the op: an `f32` operand is `X`,
    /// a `bool` one is `X.lt(X)`. The sweep asks the stage under test what
    /// it takes, which is fine for what this checks — that every stage has
    /// a path for every advertised name — and the typing itself is pinned
    /// by `sema`'s tests.
    fn expand(which: Macro, method: &str, arg_count: usize) -> Result<(), String> {
        let name = Ident::new(method, Span::call_site());
        let number = quote!(X);
        let mask = quote!(X.lt(X));
        let (receiver, args): (proc_macro2::TokenStream, Vec<proc_macro2::TokenStream>) =
            match OpKind::from_method_call(method, arg_count).map(method_typing) {
                Some(MethodTyping::Choice) => (mask, vec![number.clone(), number]),
                Some(MethodTyping::Comparison) | Some(MethodTyping::Arithmetic) | None => {
                    (number.clone(), vec![number; arg_count])
                }
            };
        let body = quote! { || #receiver.#name(#(#args),*) };

        let def = parser::parse(body).map_err(|e| e.to_string())?;
        let analyzed = sema::analyze(def).map_err(|e| e.to_string())?;
        let mut kernel_optimizer;
        let mut raw_optimizer;
        let optimizer: &mut dyn Optimize = match which {
            Macro::Kernel => {
                kernel_optimizer = macro_tier();
                &mut kernel_optimizer
            }
            Macro::KernelRaw => {
                raw_optimizer = Identity;
                &mut raw_optimizer
            }
        };
        emit::emit_kernel(&analyzed, optimizer).map(|_| ())
    }

    /// Every `(method, arg_count)` the front end accepts: the primitive ops
    /// `sema` validates against, plus the library compositions.
    fn advertised() -> impl Iterator<Item = (&'static str, usize)> {
        known_method_names()
            .map(|name| {
                let op = OpKind::from_name(name)
                    .expect("known_method_names() only yields names from_name parses");
                // Arity counts the receiver as the first operand.
                (name, op.arity() - 1)
            })
            .chain(LIBRARY_METHODS.iter().copied())
    }

    #[test]
    fn through_the_kernel_macros_pipeline() {
        for (method, arg_count) in advertised() {
            assert_eq!(
                expand(Macro::Kernel, method, arg_count),
                Ok(()),
                "`kernel!(|| X.{method}(..))` is advertised but does not compile"
            );
        }
    }

    #[test]
    fn through_the_kernel_raw_macros_pipeline() {
        for (method, arg_count) in advertised() {
            assert_eq!(
                expand(Macro::KernelRaw, method, arg_count),
                Ok(()),
                "`kernel_raw!(|| X.{method}(..))` is advertised but does not compile"
            );
        }
    }

    /// The converse guard: a name nothing advertises must still be refused,
    /// so the sweeps above cannot be satisfied by accepting everything.
    #[test]
    fn a_name_the_front_end_does_not_advertise_is_still_refused() {
        assert!(expand(Macro::Kernel, "not_a_real_method", 0).is_err());
        // A real op at the wrong arity is just as unadvertised.
        assert!(expand(Macro::Kernel, "sqrt", 2).is_err());
    }

    /// A recognized name at the wrong arity must say so.
    ///
    /// Refusing it is not enough, and refusing it was all the test above
    /// checked. `sema` validates name and arity together — which is what makes
    /// `.sqrt(1.0)` a hard error instead of something that fails three stages
    /// later — but the failure then fell into the typo-suggestion path, which
    /// dutifully searched for a name close to `sqrt`, found `sqrt`, and
    /// emitted `unknown method 'sqrt'; did you mean 'sqrt'?`.
    #[test]
    fn a_known_name_at_the_wrong_arity_reports_the_arity() {
        let err = expand(Macro::Kernel, "sqrt", 2).expect_err("wrong arity must fail");
        assert!(
            err.contains("takes 0 arguments"),
            "expected an arity diagnostic, got: {err}"
        );
        assert!(
            !err.contains("did you mean"),
            "a recognized name must not be sent to the typo search: {err}"
        );
    }
}
