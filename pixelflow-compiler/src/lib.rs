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

/// The `kernel!` macro: closure syntax for a [`Kernel`](pixelflow_core::Kernel),
/// optimized by the e-graph at macro-expansion time.
///
/// - Zero params → a `Kernel` value.
/// - N params → a builder closure `move |p0: f32, ...| -> Kernel` that
///   constant-folds its arguments into the fragment.
///
/// Kernels compose as values — `Kernel::at`/`sum`/`select`/arithmetic — so
/// there is no manifold-typed parameter. Derivatives (`DX`/`DY`) become
/// symbolic `Dwrt` nodes, resolved by the e-graph here when it can and by
/// codegen otherwise.
///
/// # Syntax
///
/// ```ignore
/// kernel!(|param1: f32, param2: f32, ...| expression)
/// ```
///
/// # Example
///
/// ```ignore
/// use pixelflow_compiler::kernel;
/// use pixelflow_core::{Kernel, Lattice};
///
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
/// # Parameters
///
/// A builder's arguments are anything `Into<Scalar>`, and the type at the
/// call site decides what the parameter is. An `f32` is folded into the
/// fragment as a constant, so `circle(0.0, 0.0, 1.0)` is the same kernel it
/// always was. A [`Uniform`](pixelflow_core::Uniform) handle makes the
/// parameter an *argument* of the compiled kernel instead — invariant across
/// the lattice, bound per call from a `UniformBlock`, never folded — so a
/// scene transform or a cursor position moves without a recompile:
///
/// ```ignore
/// let cx = Uniform::new(0.0);
/// let moving = circle(cx, 0.0, 1.0);   // cx is an argument; cy and r are folded
/// ```
///
/// Each `let` binding of a builder is one signature: the same binding cannot
/// be called with an `f32` and a `Uniform` in the same position.
///
/// # Pipeline
///
/// 1. **Parser**: closure syntax → AST
/// 2. **Semantic analysis**: symbol resolution, method validation
/// 3. **Arena lowering**: the AST becomes an `ExprArena`
/// 4. **Optimization**: e-graph saturation + latency-prior extraction, on
///    the arena. A kernel carrying a `Dwrt` declines here and is optimized
///    at bake time instead, so composition still gets the chain rule.
/// 5. **Emission**: the arena becomes code that rebuilds it at load time
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
            .nodes_raw()
            .iter()
            .any(|n| matches!(n, ExprNode::Binary(OpKind::Dwrt, _, _)));
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

    /// Expand `X.<method>(X, ..)` through `which` macro's own pipeline —
    /// the same calls [`kernel`] and [`kernel_raw`] make — and report
    /// whether it yields code.
    fn expand(which: Macro, method: &str, arg_count: usize) -> Result<(), String> {
        let name = Ident::new(method, Span::call_site());
        let args = (0..arg_count).map(|_| quote!(X));
        let body = quote! { || X.#name(#(#args),*) };

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
