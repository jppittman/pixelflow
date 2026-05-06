//! # Code Generation
//!
//! Emits Rust code from the analyzed AST.
//!
//! ## Architecture: ZST Expression + Let/Var Binding
//!
//! PixelFlow expressions are Copy when all components are ZST (zero-sized types).
//! The coordinate variables X, Y, Z, W are ZST, and so are Var<N> references.
//! This means expressions using Var<N> remain Copy.
//!
//! The solution is a two-layer architecture:
//!
//! 1. **ZST Expression**: Built using coordinate variables (X, Y, Z, W) and Var<N>
//! 2. **Value Struct**: Stores non-ZST captured parameters (f32 values)
//! 3. **Let/Var binding**: Nested Let wrappers extend domain with parameter values
//!
//! ## Let/Var Binding (Peano-Encoded Stack)
//!
//! Parameters are bound using nested `Let::new()` calls that extend the domain:
//! - First param → deepest binding → `Var::<N{n-1}>`
//! - Last param → shallowest binding → `Var::<N0>` (head of stack)
//!
//! This allows **unlimited parameters** (no longer limited to 2).
//!
//! ## Example Transformation
//!
//! ```text
//! // User writes:
//! kernel!(|cx: f32, cy: f32, cz: f32| X - cx + Y - cy + Z - cz)
//!
//! // Becomes:
//! struct __Kernel { cx: f32, cy: f32, cz: f32 }
//!
//! impl Manifold<Field4> for __Kernel {
//!     fn eval(&self, __p: Field4) -> Field {
//!         // ZST expression using Var<N> (Copy!)
//!         let __expr = X - Var::<N2>::new() + Y - Var::<N1>::new() + Z - Var::<N0>::new();
//!         // Nested Let bindings extend domain with parameter values
//!         Let::new(self.cx,
//!           Let::new(self.cy,
//!             Let::new(self.cz,
//!               __expr))).eval(__p)
//!     }
//! }
//! ```
//!
//! ## Module Structure
//!
//! - `util`: Shared utility functions (imports, tuple building)
//! - `binding`: Binding strategy enum and emission
//! - `struct_emitter`: Builder pattern for struct generation
//! - `emitter`: Core CodeEmitter logic

mod binding;
mod emitter;
mod leveled;
mod struct_emitter;
mod util;

use crate::sema::AnalyzedKernel;
use proc_macro2::TokenStream;

pub use emitter::CodeEmitter;

/// Emit Rust code for an analyzed kernel.
pub fn emit(analyzed: AnalyzedKernel) -> TokenStream {
    let mut emitter = CodeEmitter::new(&analyzed);
    emitter.emit_kernel()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimize::optimize;
    use crate::parser::parse;
    use crate::sema::analyze;
    use quote::quote;

    fn compile(input: TokenStream) -> TokenStream {
        let kernel = parse(input).unwrap();
        let analyzed = analyze(kernel).unwrap();
        emit(analyzed)
    }

    fn compile_with_optimize(input: TokenStream) -> TokenStream {
        let kernel = parse(input).unwrap();
        let analyzed = analyze(kernel).unwrap();
        let optimized = optimize(analyzed);
        emit(optimized)
    }

    #[test]
    fn emit_simple_kernel() {
        // Anonymous kernel - emits closure returning WithContext
        let input = quote! { |cx: f32| X - cx };
        let output = compile(input);
        let output_str = output.to_string();

        // Should be a closure
        assert!(
            output_str.contains("move | cx : f32 |"),
            "Expected closure, got: {}",
            output_str
        );
        // Should use CtxVar::<A0, 0usize> for the single parameter
        assert!(
            output_str.contains("CtxVar :: < A0 , 0usize >"),
            "Expected CtxVar::<A0, 0usize>, got: {}",
            output_str
        );
        // Should use WithContext
        assert!(
            output_str.contains("WithContext :: new"),
            "Expected WithContext::new, got: {}",
            output_str
        );
    }

    #[test]
    fn emit_two_params() {
        // Anonymous kernel with two params
        let input = quote! { |cx: f32, cy: f32| (X - cx) + (Y - cy) };
        let output = compile(input);
        let output_str = output.to_string();

        // cx → CtxVar::<A0, 1usize> (first param, highest index in tuple, but index 1 in array?)
        // Wait, param_indices logic:
        // n=2. cx: index 1. cy: index 0.
        // Array values are sorted by index.
        // param_values = [(0, cy), (1, cx)].
        // Array = [cy, cx].
        // So cy is at A0[0], cx is at A0[1].

        assert!(
            output_str.contains("CtxVar :: < A0 , 1usize >"),
            "Expected CtxVar::<A0, 1usize> for cx, got: {}",
            output_str
        );
        assert!(
            output_str.contains("CtxVar :: < A0 , 0usize >"),
            "Expected CtxVar::<A0, 0usize> for cy, got: {}",
            output_str
        );
        // Should use WithContext with array
        assert!(
            output_str.contains("WithContext :: new ((["),
            "Expected WithContext with array, got: {}",
            output_str
        );
    }

    #[test]
    fn emit_three_params() {
        let input = quote! { |a: f32, b: f32, c: f32| a + b + c };
        let output = compile(input);
        let output_str = output.to_string();

        // a → index 2, b → index 1, c → index 0
        assert!(
            output_str.contains("CtxVar :: < A0 , 2usize >"),
            "Expected CtxVar::<A0, 2usize> for a"
        );
        assert!(
            output_str.contains("CtxVar :: < A0 , 1usize >"),
            "Expected CtxVar::<A0, 1usize> for b"
        );
        assert!(
            output_str.contains("CtxVar :: < A0 , 0usize >"),
            "Expected CtxVar::<A0, 0usize> for c"
        );
    }

    #[test]
    fn emit_empty_params() {
        // Anonymous kernel with no params
        let input = quote! { || X + Y };
        let output = compile(input);
        let output_str = output.to_string();

        // Should be a closure with empty params (|| or | |)
        assert!(
            output_str.contains("||") || output_str.contains("| |"),
            "Expected no-param closure, got: {}",
            output_str
        );
        // Should use WithContext with unit
        assert!(
            output_str.contains("WithContext :: new (() , __expr)"),
            "Expected WithContext with unit, got: {}",
            output_str
        );
    }

    #[test]
    fn emit_method_calls() {
        let input = quote! { |r: f32| (X * X + Y * Y).sqrt() - r };
        let output = compile(input);
        let output_str = output.to_string();

        // r → CtxVar::<A0, 0usize>
        assert!(output_str.contains(". sqrt ()"));
        assert!(
            output_str.contains("CtxVar :: < A0 , 0usize >"),
            "Expected CtxVar::<A0, 0usize> for r"
        );
    }

    #[test]
    fn emit_manifold_param() {
        // Anonymous kernel with manifold param - still emits closure
        let input = quote! { |inner: kernel, r: f32| inner - r };
        let output = compile(input);
        let output_str = output.to_string();

        // Should be a closure (anonymous kernels always emit closures)
        assert!(
            output_str.contains("move |"),
            "Expected closure, got: {}",
            output_str
        );

        // inner → ContextFree(inner), r → CtxVar::<A0, 0usize>
        assert!(
            output_str.contains("ContextFree (inner)"),
            "Expected ContextFree for inner, got: {}",
            output_str
        );
        assert!(
            output_str.contains("CtxVar :: < A0 , 0usize >"),
            "Expected CtxVar::<A0, 0usize> for r, got: {}",
            output_str
        );

        // Should use WithContext
        assert!(
            output_str.contains("WithContext :: new"),
            "Expected WithContext::new, got: {}",
            output_str
        );
    }

    #[test]
    fn emit_multiple_manifold_params() {
        // Anonymous kernel with multiple manifold params
        let input = quote! { |a: kernel, b: kernel| a + b };
        let output = compile(input);
        let output_str = output.to_string();

        // Should be a closure
        assert!(
            output_str.contains("move |"),
            "Expected closure, got: {}",
            output_str
        );

        // a, b -> ContextFree(a), ContextFree(b)
        assert!(
            output_str.contains("ContextFree (a)") && output_str.contains("ContextFree (b)"),
            "Expected ContextFree for a and b, got: {}",
            output_str
        );

        // Should use WithContext with tuple
        assert!(
            output_str.contains("WithContext :: new"),
            "Expected WithContext, got: {}",
            output_str
        );
    }

    #[test]
    fn emit_named_kernel() {
        // Named kernel - emits struct with given name
        let input = quote! { pub struct Circle = |cx: f32, cy: f32, r: f32| -> Field {
            let dx = X - cx;
            let dy = Y - cy;
            (dx * dx + dy * dy).sqrt() - r
        }};
        let output = compile(input);
        let output_str = output.to_string();

        // Should have struct named Circle
        assert!(
            output_str.contains("pub struct Circle"),
            "Expected pub struct Circle, got: {}",
            output_str
        );
        // Should have new constructor
        assert!(
            output_str.contains("pub fn new"),
            "Expected new constructor, got: {}",
            output_str
        );
        // Should implement Manifold
        assert!(
            output_str.contains("impl :: pixelflow_core :: Manifold"),
            "Expected Manifold impl, got: {}",
            output_str
        );
        // Should preserve let bindings
        assert!(
            output_str.contains("let dx"),
            "Expected let dx binding, got: {}",
            output_str
        );
    }

    #[test]
    fn emit_named_kernel_with_domain() {
        // Named kernel with domain type annotation
        let input = quote! { pub struct Test = |x: f32| Field -> Discrete {
            let a = X + x;
            a
        }};
        let output = compile(input);
        let output_str = output.to_string();

        eprintln!("Domain kernel output:\n{}", output_str);

        // Should preserve let bindings even with domain annotation
        assert!(
            output_str.contains("let a"),
            "Expected let a binding, got: {}",
            output_str
        );
    }

    #[test]
    fn emit_named_kernel_with_verbatim() {
        // Named kernel with method call on qualified path (Verbatim)
        // This mimics: ColorCube::default().at(red, green, blue, 1.0)
        let input = quote! { pub struct Test = |t: f32| Field -> Discrete {
            let a = X + t;
            let b = Y + a;
            Foo::default().at(a, b, 1.0)
        }};
        let output = compile(input);
        let output_str = output.to_string();

        eprintln!("Verbatim kernel output:\n{}", output_str);

        // Should preserve let bindings
        assert!(
            output_str.contains("let a"),
            "Expected let a binding, got: {}",
            output_str
        );
        assert!(
            output_str.contains("let b"),
            "Expected let b binding, got: {}",
            output_str
        );
    }

    #[test]
    fn emit_complex_kernel_like_psychedelic() {
        // This test mimics the psychedelic shader structure
        let input = quote! { pub struct Test = |t: f32, width: f32, height: f32| Field -> Discrete {
            let scale = 2.0 / height;
            let half_width = width * 0.5;
            let x = (X - half_width) * scale;
            let y = (Y - half_width) * scale;
            let r_sq = x * x + y * y;
            let radial = (r_sq - 0.7).abs();
            let red = (radial + 1.0) * 0.5;
            let green = (radial - 0.3).abs();
            let blue = (t * 0.1).sin();
            Foo::default().at(red, green, blue, 1.0)
        }};
        let output = compile(input);
        let output_str = output.to_string();

        eprintln!("Complex kernel output:\n{}", output_str);

        // Should preserve all let bindings
        assert!(output_str.contains("let scale"), "Expected let scale");
        assert!(
            output_str.contains("let half_width"),
            "Expected let half_width"
        );
        assert!(output_str.contains("let x"), "Expected let x");
        assert!(output_str.contains("let y"), "Expected let y");
        assert!(output_str.contains("let r_sq"), "Expected let r_sq");
        assert!(output_str.contains("let radial"), "Expected let radial");
        assert!(output_str.contains("let red"), "Expected let red");
        assert!(output_str.contains("let green"), "Expected let green");
        assert!(output_str.contains("let blue"), "Expected let blue");
    }

    #[test]
    fn emit_exact_psychedelic_kernel() {
        // Exact copy of the psychedelic shader kernel
        let input = quote! { pub struct PsychedelicScene = |t: f32, width: f32, height: f32| Field -> Discrete {
            // Screen coordinate remapping
            let scale = 2.0 / height;
            let half_width = width * 0.5;
            let half_height = height * 0.5;
            let x = (X - half_width) * scale;
            let y = (half_height - Y) * scale;

            // Time via W coordinate
            let time = W + t;

            // Radial field
            let r_sq = x * x + y * y;
            let radial = (r_sq - 0.7).abs();

            // Swirl
            let swirl_scale = (1.0 - radial) * 5.0;
            let vx = x * swirl_scale;
            let vy = y * swirl_scale;

            // Time-based values
            let phase = time * 0.5;
            let sin_w03 = (time * 0.3).sin();
            let sin_w20 = (time * 2.0).sin();

            // Swirl computation
            let swirl = ((vx + phase).sin() + 1.0) * ((vx + phase) - (vy + phase * 0.7)).abs() * 0.2 + 0.001;

            // Radial falloff with pulsing
            let pulse = 1.0 + sin_w20 * 0.1;
            let radial_factor = (radial * -4.0 * pulse).exp();

            // Red channel
            let y_factor_r = (y * 1.0 + sin_w03 * 0.2).exp();
            let raw_r = y_factor_r * radial_factor / swirl;
            let soft_r = raw_r / (raw_r.abs() + 1.0);
            let red = (soft_r + 1.0) * 0.5;

            // Green channel
            let y_factor_g = (y * -1.0 + sin_w03 * 0.2).exp();
            let raw_g = y_factor_g * radial_factor / swirl;
            let soft_g = raw_g / (raw_g.abs() + 1.0);
            let green = (soft_g + 1.0) * 0.5;

            // Blue channel
            let y_factor_b = (y * -2.0 + sin_w03 * 0.2).exp();
            let raw_b = y_factor_b * radial_factor / swirl;
            let soft_b = raw_b / (raw_b.abs() + 1.0);
            let blue = (soft_b + 1.0) * 0.5;

            // Contramap through color cube
            ColorCube::default().at(red, green, blue, 1.0)
        }};
        let output = compile(input);
        let output_str = output.to_string();

        eprintln!("Psychedelic kernel output:\n{}", output_str);

        // Should preserve all let bindings
        assert!(output_str.contains("let scale"), "Expected let scale");
        assert!(output_str.contains("let red"), "Expected let red");
        assert!(output_str.contains("let green"), "Expected let green");
        assert!(output_str.contains("let blue"), "Expected let blue");
    }

    #[test]
    fn emit_exact_psychedelic_kernel_with_optimize() {
        // Same kernel but with optimization (this is what the actual macro does)
        let input = quote! { pub struct PsychedelicScene = |t: f32, width: f32, height: f32| Field -> Discrete {
            // Screen coordinate remapping
            let scale = 2.0 / height;
            let half_width = width * 0.5;
            let half_height = height * 0.5;
            let x = (X - half_width) * scale;
            let y = (half_height - Y) * scale;

            // Time via W coordinate
            let time = W + t;

            // Radial field
            let r_sq = x * x + y * y;
            let radial = (r_sq - 0.7).abs();

            // Swirl
            let swirl_scale = (1.0 - radial) * 5.0;
            let vx = x * swirl_scale;
            let vy = y * swirl_scale;

            // Time-based values
            let phase = time * 0.5;
            let sin_w03 = (time * 0.3).sin();
            let sin_w20 = (time * 2.0).sin();

            // Swirl computation
            let swirl = ((vx + phase).sin() + 1.0) * ((vx + phase) - (vy + phase * 0.7)).abs() * 0.2 + 0.001;

            // Radial falloff with pulsing
            let pulse = 1.0 + sin_w20 * 0.1;
            let radial_factor = (radial * -4.0 * pulse).exp();

            // Red channel
            let y_factor_r = (y * 1.0 + sin_w03 * 0.2).exp();
            let raw_r = y_factor_r * radial_factor / swirl;
            let soft_r = raw_r / (raw_r.abs() + 1.0);
            let red = (soft_r + 1.0) * 0.5;

            // Green channel
            let y_factor_g = (y * -1.0 + sin_w03 * 0.2).exp();
            let raw_g = y_factor_g * radial_factor / swirl;
            let soft_g = raw_g / (raw_g.abs() + 1.0);
            let green = (soft_g + 1.0) * 0.5;

            // Blue channel
            let y_factor_b = (y * -2.0 + sin_w03 * 0.2).exp();
            let raw_b = y_factor_b * radial_factor / swirl;
            let soft_b = raw_b / (raw_b.abs() + 1.0);
            let blue = (soft_b + 1.0) * 0.5;

            // Contramap through color cube
            ColorCube::default().at(red, green, blue, 1.0)
        }};
        let output = compile_with_optimize(input);
        let output_str = output.to_string();

        eprintln!("Psychedelic kernel with optimize output:\n{}", output_str);

        // Should preserve all let bindings
        assert!(output_str.contains("let scale"), "Expected let scale");
        assert!(output_str.contains("let red"), "Expected let red");
        assert!(output_str.contains("let green"), "Expected let green");
        assert!(output_str.contains("let blue"), "Expected let blue");
    }
}
