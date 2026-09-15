> **Superseded (2026-07-20) — the implementation plan for a macro that no longer
> exists.** See the banner on
> [`2025-02-21-kernel-jit-feature-parity-design.md`](2025-02-21-kernel-jit-feature-parity-design.md).
> `kernel_jit!`, `JitManifold`, `pixelflow-ir/src/expr.rs` and
> `pixelflow-compiler/src/ir_bridge.rs` are all deleted; the task list below is
> finished or moot. Kept for the param-baking rationale.

# kernel_jit! Feature Parity Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Bring `kernel_jit!` to full feature parity with `kernel!` — captured scalar parameters are constant-folded into the JIT'd kernel at build time, returning an `impl Manifold` with zero runtime parameter overhead.

**Architecture:** Add `Expr::Param(u8)` to the IR as an ephemeral node type; `substitute_params` folds it to `Expr::Const` before JIT compilation; `kernel_jit!` emits a builder closure that captures params as `f32`, substitutes, compiles, and returns a `JitManifold` that owns the `ExecutableCode`. No cache, no calling convention change, no emitter changes.

**Tech Stack:** Rust proc-macros (`syn`, `quote`, `proc_macro2`), `pixelflow-ir` (expr IR, JIT emitter), `pixelflow-compiler` (proc-macro crate), `pixelflow-core` (`Manifold` trait, `Field` type).

---

## Context You Must Have Before Starting

### The two macros and their intended parity

```rust
// kernel! — compile-time LLVM, params baked in
let k = kernel!(|cx: f32, r: f32| (X - cx) * r);
let manifold = k(1.0, 2.0);           // returns impl Manifold
manifold.eval((x, y, z, w));          // pixel evaluation

// kernel_jit! — after this plan, identical semantics
let k = kernel_jit!(|cx: f32, r: f32| (X - cx) * r);
let manifold = k(1.0, 2.0);           // JITs immediately, params folded as Const
manifold.eval((x, y, z, w));          // calls JIT'd machine code
```

### Parameter ordering rule (IMPORTANT — matches `kernel!`)

Parameters are indexed in **reverse declaration order** in the scalar array:
- `|cx: f32, cy: f32, r: f32|` → `cx` = index 2, `cy` = index 1, `r` = index 0
- This mirrors the `CtxVar` indexing in `kernel!` codegen (`n - 1 - i`)
- For `kernel_jit!` purposes this means: `params[0]` in the builder closure slice = last declared param

### Key files and their roles

| File | Role |
|------|------|
| `pixelflow-ir/src/expr.rs` | `Expr` enum — add `Param(u8)` here |
| `pixelflow-ir/src/lib.rs` | Re-exports — add `substitute_params` here |
| `pixelflow-compiler/src/ir_bridge.rs` | `ast_to_ir` — map param idents to `Expr::Param(i)` |
| `pixelflow-compiler/src/lib.rs` | `kernel_jit!` macro — rewrite to emit builder closure |
| `pixelflow-ir/src/jit_manifold.rs` | New file — `JitManifold` struct + `Manifold` impl |
| `pixelflow-ir/src/backend/emit/executable.rs` | Tests live here — add param round-trip test |
| `pixelflow-compiler/src/codegen/mod.rs` | Existing tests live here — add `kernel_jit!` round-trip test |

### The `Manifold` trait signature (from `pixelflow-core/src/manifold.rs`)

```rust
pub trait Manifold<P = (Field, Field, Field, Field)>: Send + Sync {
    type Output;
    fn eval(&self, p: P) -> Self::Output;
}
```

`JitManifold` will implement `Manifold<(Field, Field, Field, Field), Output = Field>`.

### `Field` type and `KernelFn`

`Field` is `#[repr(transparent)]` wrapping the platform SIMD type:
- ARM64: `float32x4_t` (NEON, 4-lane f32)
- x86-64: `__m128` (SSE2, 4-lane f32), or `__m512` (AVX-512, 16-lane f32)

`KernelFn` is defined in `pixelflow-ir/src/backend/emit/executable.rs`:
```rust
#[cfg(target_arch = "aarch64")]
pub type KernelFn = extern "C" fn(float32x4_t, float32x4_t, float32x4_t, float32x4_t) -> float32x4_t;
```

### How `ast_to_ir` currently fails on parameters

In `pixelflow-compiler/src/ir_bridge.rs` the `Expr::Ident` branch:
```rust
Expr::Ident(ident) => {
    match name.as_str() {
        "X" => Ok(IR::Var(0)),
        "Y" => Ok(IR::Var(1)),
        "Z" => Ok(IR::Var(2)),
        "W" => Ok(IR::Var(3)),
        _ => Err(format!("Unknown identifier: {}", name)),  // <-- params hit this
    }
}
```

### How `sema::analyze` tracks parameters

`AnalyzedKernel.def.params` is a `Vec<Param>` in declaration order. Each `Param` has `.name: Ident` and `.kind: ParamKind`. For scalar parameters, `kind = ParamKind::Scalar(type)`. This is the source of truth for parameter index.

### Existing test pattern to follow

From `pixelflow-ir/src/backend/emit/executable.rs`:
```rust
#[test]
#[cfg(target_arch = "aarch64")]
fn test_compile_add_xy() {
    let expr = Expr::Binary(OpKind::Add, Box::new(Expr::Var(0)), Box::new(Expr::Var(1)));
    let exec = compile(&expr).expect("compile failed");
    unsafe {
        let func: KernelFn = exec.as_fn();
        use core::arch::aarch64::*;
        let x = vdupq_n_f32(10.0);
        let y = vdupq_n_f32(32.0);
        let result = func(x, y, vdupq_n_f32(0.0), vdupq_n_f32(0.0));
        assert_eq!(vgetq_lane_f32(result, 0), 42.0);
    }
}
```

---

## Task 1: Add `Expr::Param(u8)` to the IR

**Files:**
- Modify: `pixelflow-ir/src/expr.rs`

**Step 1: Read the current `Expr` enum**

Open `pixelflow-ir/src/expr.rs` and locate the `Expr` enum. It currently has:
`Var(u8)`, `Const(f32)`, `Unary(...)`, `Binary(...)`, `Ternary(...)`, `Nary(...)`.

**Step 2: Add `Param(u8)` variant**

Add the new variant after `Const(f32)`. Include a doc comment explaining it is ephemeral:

```rust
/// Captured parameter by index. Ephemeral — must be substituted to [`Expr::Const`]
/// via [`substitute_params`] before passing to the JIT emitter.
/// Index is declaration order: first param = 0, second = 1, etc.
Param(u8),
```

**Step 3: Update `kind()`, `depth()`, `node_count()` match arms**

These methods exist on `Expr` and use exhaustive matches. Add `Param` to each:

- `kind()`: `Expr::Param(_) => OpKind::Const` (closest semantic match — it becomes a constant)
- `depth()`: `Expr::Param(_) => 0` (leaf node, same as `Var` and `Const`)
- `node_count()`: `Expr::Param(_) => 1` (single node, same as `Var` and `Const`)

**Step 4: Build to check for exhaustiveness errors**

```bash
cd /Users/jppittman/Documents/Projects/core-term/.claude/worktrees/crazy-montalcini
cargo build -p pixelflow-ir 2>&1 | head -40
```

Expected: Any match arms that don't handle `Param` will error. Fix them.

**Step 5: Commit**

```bash
git add pixelflow-ir/src/expr.rs
git commit -m "feat(pixelflow-ir): add Expr::Param(u8) ephemeral node"
```

---

## Task 2: Add `substitute_params` to `pixelflow-ir`

**Files:**
- Modify: `pixelflow-ir/src/lib.rs`

**Step 1: Write the failing test first**

At the bottom of `pixelflow-ir/src/lib.rs`, add a `#[cfg(test)]` module:

```rust
#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;

    #[test]
    fn substitute_single_param() {
        // Expr::Param(0) with params=[3.14] → Expr::Const(3.14)
        let expr = Expr::Param(0);
        let result = substitute_params(&expr, &[3.14_f32]);
        assert!(matches!(result, Expr::Const(v) if (v - 3.14).abs() < 1e-6));
    }

    #[test]
    fn substitute_params_in_binary() {
        // (X - Param(0)) with params=[1.0] → (X - Const(1.0))
        let expr = Expr::Binary(
            OpKind::Sub,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Param(0)),
        );
        let result = substitute_params(&expr, &[1.0_f32]);
        match result {
            Expr::Binary(OpKind::Sub, left, right) => {
                assert!(matches!(*left, Expr::Var(0)));
                assert!(matches!(*right, Expr::Const(v) if (v - 1.0).abs() < 1e-6));
            }
            _ => panic!("expected Binary(Sub, ...)"),
        }
    }

    #[test]
    fn substitute_multiple_params() {
        // Param(0) + Param(1) with params=[10.0, 32.0] → Const(10.0) + Const(32.0)
        let expr = Expr::Binary(
            OpKind::Add,
            Box::new(Expr::Param(0)),
            Box::new(Expr::Param(1)),
        );
        let result = substitute_params(&expr, &[10.0_f32, 32.0_f32]);
        match result {
            Expr::Binary(OpKind::Add, left, right) => {
                assert!(matches!(*left, Expr::Const(v) if (v - 10.0).abs() < 1e-6));
                assert!(matches!(*right, Expr::Const(v) if (v - 32.0).abs() < 1e-6));
            }
            _ => panic!("expected Binary(Add, ...)"),
        }
    }

    #[test]
    #[should_panic(expected = "substitute_params: param index 1 out of range")]
    fn substitute_params_panics_on_index_out_of_range() {
        let expr = Expr::Param(1); // index 1, but only 1 param provided
        substitute_params(&expr, &[1.0_f32]);
    }
}
```

**Step 2: Run the test to verify it fails**

```bash
cargo test -p pixelflow-ir substitute 2>&1 | tail -20
```

Expected: FAIL — `substitute_params` does not exist yet.

**Step 3: Implement `substitute_params` in `pixelflow-ir/src/lib.rs`**

Add this function (place it near the top of the file, after the re-exports, inside `#[cfg(feature = "alloc")]`):

```rust
#[cfg(feature = "alloc")]
/// Replaces every [`Expr::Param(i)`] node with [`Expr::Const(params[i])`].
///
/// # Panics
///
/// Panics if any `Param(i)` has `i >= params.len()`. This is always a bug in
/// the calling macro — the number of params in the expression must match the
/// slice provided.
pub fn substitute_params(expr: &Expr, params: &[f32]) -> Expr {
    match expr {
        Expr::Param(i) => {
            let i = *i as usize;
            assert!(
                i < params.len(),
                "substitute_params: param index {} out of range (have {} params)",
                i,
                params.len()
            );
            Expr::Const(params[i])
        }
        Expr::Var(i) => Expr::Var(*i),
        Expr::Const(v) => Expr::Const(*v),
        Expr::Unary(op, child) => {
            Expr::Unary(*op, Box::new(substitute_params(child, params)))
        }
        Expr::Binary(op, left, right) => Expr::Binary(
            *op,
            Box::new(substitute_params(left, params)),
            Box::new(substitute_params(right, params)),
        ),
        Expr::Ternary(op, a, b, c) => Expr::Ternary(
            *op,
            Box::new(substitute_params(a, params)),
            Box::new(substitute_params(b, params)),
            Box::new(substitute_params(c, params)),
        ),
        Expr::Nary(op, children) => Expr::Nary(
            *op,
            children.iter().map(|c| substitute_params(c, params)).collect(),
        ),
    }
}
```

**Step 4: Run tests to verify they pass**

```bash
cargo test -p pixelflow-ir substitute 2>&1 | tail -20
```

Expected: 4 tests pass.

**Step 5: Commit**

```bash
git add pixelflow-ir/src/lib.rs
git commit -m "feat(pixelflow-ir): add substitute_params for Expr::Param folding"
```

---

## Task 3: Extend `ir_bridge` to map parameters to `Expr::Param`

**Files:**
- Modify: `pixelflow-compiler/src/ir_bridge.rs`

**Step 1: Understand the current `ast_to_ir` signature**

The function currently is:
```rust
pub fn ast_to_ir(expr: &Expr) -> Result<IR, String>
```

It needs to know which identifiers are parameters and what their indices are. The analyzed kernel (`AnalyzedKernel`) has `.def.params: Vec<Param>` in declaration order.

**Step 2: Change signature to accept a parameter map**

Change `ast_to_ir` to accept a parameter index map:

```rust
pub fn ast_to_ir(
    expr: &crate::ast::Expr,
    param_indices: &std::collections::HashMap<String, u8>,
) -> Result<::pixelflow_ir::Expr, String>
```

The map is `param_name → index` where index follows declaration order (first param = 0, second = 1, etc.).

**Step 3: Update the `Ident` branch to use the map**

Replace the `_ => Err(...)` fallback:

```rust
Expr::Ident(ident) => {
    let name = ident.name.to_string();
    match name.as_str() {
        "X" => Ok(::pixelflow_ir::Expr::Var(0)),
        "Y" => Ok(::pixelflow_ir::Expr::Var(1)),
        "Z" => Ok(::pixelflow_ir::Expr::Var(2)),
        "W" => Ok(::pixelflow_ir::Expr::Var(3)),
        _ => {
            if let Some(&idx) = param_indices.get(&name) {
                Ok(::pixelflow_ir::Expr::Param(idx))
            } else {
                Err(format!("Unknown identifier: {name}"))
            }
        }
    }
}
```

**Step 4: Update all recursive calls within `ast_to_ir`**

Every recursive call to `ast_to_ir` inside the function must pass `param_indices` through. Search for all `ast_to_ir(` calls within the file and add the second argument.

**Step 5: Add a helper to build the param index map from `AnalyzedKernel`**

Add this function near `ast_to_ir`:

```rust
/// Builds a `param_name → index` map from an analyzed kernel.
/// Index is declaration order: first param = 0, second = 1, etc.
/// Only scalar params are included — manifold params cannot be folded.
pub fn scalar_param_indices(
    analyzed: &crate::sema::AnalyzedKernel,
) -> std::collections::HashMap<String, u8> {
    analyzed
        .def
        .params
        .iter()
        .enumerate()
        .filter_map(|(i, p)| match &p.kind {
            crate::ast::ParamKind::Scalar(_) => Some((p.name.to_string(), i as u8)),
            crate::ast::ParamKind::Manifold => None,
        })
        .collect()
}
```

**Step 6: Update the call site in `kernel_jit!`**

In `pixelflow-compiler/src/lib.rs`, the existing call is:
```rust
let ir = match ir_bridge::ast_to_ir(&analyzed.def.body) {
```

Change to:
```rust
let param_map = ir_bridge::scalar_param_indices(&analyzed);
let ir = match ir_bridge::ast_to_ir(&analyzed.def.body, &param_map) {
```

**Step 7: Build to check everything compiles**

```bash
cargo build -p pixelflow-compiler 2>&1 | head -40
```

Fix any remaining call sites that pass the old single-argument signature.

**Step 8: Commit**

```bash
git add pixelflow-compiler/src/ir_bridge.rs pixelflow-compiler/src/lib.rs
git commit -m "feat(pixelflow-compiler): map param idents to Expr::Param in ir_bridge"
```

---

## Task 4: Create `JitManifold`

**Files:**
- Create: `pixelflow-ir/src/jit_manifold.rs`
- Modify: `pixelflow-ir/src/lib.rs` (add `pub mod jit_manifold; pub use jit_manifold::JitManifold;`)

**Step 1: Write the failing test first**

Add to the existing test module in `pixelflow-ir/src/lib.rs`:

```rust
#[cfg(all(test, feature = "alloc", target_arch = "aarch64"))]
mod jit_manifold_tests {
    use super::*;
    use crate::backend::emit::compile;

    #[test]
    fn jit_manifold_eval_x_plus_const() {
        // Build expr: X + Const(32.0) (simulates a param already substituted)
        let expr = Expr::Binary(
            OpKind::Add,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Const(32.0)),
        );
        let code = compile(&expr).expect("compile failed");
        let m = JitManifold::new(code);

        use pixelflow_core::Field;
        // eval at X=10.0, others=0.0 → expect 42.0
        let x = Field::splat(10.0);
        let y = Field::splat(0.0);
        let z = Field::splat(0.0);
        let w = Field::splat(0.0);
        let result = m.eval((x, y, z, w));
        assert_eq!(result.extract(0), 42.0);
    }
}
```

**Step 2: Run the test to verify it fails**

```bash
cargo test -p pixelflow-ir jit_manifold 2>&1 | tail -20
```

Expected: FAIL — `JitManifold` does not exist.

**Step 3: Create `pixelflow-ir/src/jit_manifold.rs`**

```rust
use crate::backend::emit::executable::{ExecutableCode, KernelFn};
use pixelflow_core::{Field, Manifold};

/// A JIT-compiled manifold. Owns the executable code and calls it on eval.
///
/// Constructed by `kernel_jit!` after parameter substitution and JIT compilation.
/// The caller owns this value and decides its lifetime — there is no internal cache.
pub struct JitManifold {
    code: ExecutableCode,
}

impl JitManifold {
    pub fn new(code: ExecutableCode) -> Self {
        Self { code }
    }
}

impl Manifold<(Field, Field, Field, Field)> for JitManifold {
    type Output = Field;

    #[inline(always)]
    fn eval(&self, (x, y, z, w): (Field, Field, Field, Field)) -> Field {
        // SAFETY: ExecutableCode is read-only mapped memory containing a valid
        // function matching KernelFn's ABI. The JIT emitter is responsible for
        // correctness of the emitted code.
        let func: KernelFn = unsafe { self.code.as_fn() };
        Field::from_raw(func(x.into_raw(), y.into_raw(), z.into_raw(), w.into_raw()))
    }
}

// SAFETY: ExecutableCode contains read-only mapped memory with no interior mutability.
unsafe impl Send for JitManifold {}
unsafe impl Sync for JitManifold {}
```

> **Note on `Field::from_raw` / `into_raw`:** Check whether these methods exist. If `Field` is `#[repr(transparent)]` over the SIMD type, you may need `unsafe { core::mem::transmute(x) }` instead. Look at how existing tests in `executable.rs` call `KernelFn` and extract the result — match that pattern exactly.

**Step 4: Add the module to `pixelflow-ir/src/lib.rs`**

```rust
#[cfg(feature = "alloc")]
pub mod jit_manifold;
#[cfg(feature = "alloc")]
pub use jit_manifold::JitManifold;
```

**Step 5: Build to check**

```bash
cargo build -p pixelflow-ir 2>&1 | head -40
```

Fix any Field conversion issues by looking at how `executable.rs` tests cast between `Field` and the raw SIMD types.

**Step 6: Run the test**

```bash
cargo test -p pixelflow-ir jit_manifold 2>&1 | tail -20
```

Expected: PASS.

**Step 7: Commit**

```bash
git add pixelflow-ir/src/jit_manifold.rs pixelflow-ir/src/lib.rs
git commit -m "feat(pixelflow-ir): add JitManifold implementing Manifold trait"
```

---

## Task 5: Rewrite `kernel_jit!` to emit a builder closure

**Files:**
- Modify: `pixelflow-compiler/src/lib.rs` (the `kernel_jit` proc-macro function, lines ~177-260)

**Step 1: Write the failing test first**

These are integration tests using the macro. Add to `pixelflow-compiler/src/codegen/mod.rs` or a new `pixelflow-compiler/tests/kernel_jit.rs` integration test file:

```rust
// pixelflow-compiler/tests/kernel_jit.rs
use pixelflow_compiler::kernel_jit;
use pixelflow_core::{Field, Manifold};

#[test]
fn kernel_jit_no_params_returns_manifold() {
    let m = kernel_jit!(|| X + Y);
    let result = m.eval((Field::splat(10.0), Field::splat(32.0), Field::splat(0.0), Field::splat(0.0)));
    assert_eq!(result.extract(0), 42.0);
}

#[test]
fn kernel_jit_one_param_builder() {
    let builder = kernel_jit!(|offset: f32| X + offset);
    let m = builder(32.0_f32);
    let result = m.eval((Field::splat(10.0), Field::splat(0.0), Field::splat(0.0), Field::splat(0.0)));
    assert_eq!(result.extract(0), 42.0);
}

#[test]
fn kernel_jit_two_params_builder() {
    let builder = kernel_jit!(|cx: f32, r: f32| (X - cx) * r);
    let m = builder(1.0_f32, 2.0_f32);
    // X=5.0: (5.0 - 1.0) * 2.0 = 8.0
    let result = m.eval((Field::splat(5.0), Field::splat(0.0), Field::splat(0.0), Field::splat(0.0)));
    assert_eq!(result.extract(0), 8.0);
}

#[test]
fn kernel_jit_same_semantics_as_kernel() {
    // Both should produce identical outputs for the same expression
    use pixelflow_compiler::kernel;

    let jit_builder = kernel_jit!(|cx: f32| X - cx);
    let ct_builder = kernel!(|cx: f32| X - cx);

    let jit_m = jit_builder(3.0_f32);
    let ct_m = ct_builder(3.0_f32);

    for x_val in [0.0_f32, 1.0, 5.0, -2.0, 100.0] {
        let p = (Field::splat(x_val), Field::splat(0.0), Field::splat(0.0), Field::splat(0.0));
        let jit_result = jit_m.eval(p);
        let ct_result = ct_m.eval(p);
        assert!(
            (jit_result.extract(0) - ct_result.extract(0)).abs() < 1e-5,
            "mismatch at x={x_val}: jit={} ct={}",
            jit_result.extract(0),
            ct_result.extract(0)
        );
    }
}
```

**Step 2: Run the tests to verify they fail**

```bash
cargo test -p pixelflow-compiler 2>&1 | tail -30
```

Expected: FAIL — `kernel_jit!` currently returns a raw function pointer, not a builder or `JitManifold`.

**Step 3: Rewrite the `kernel_jit` proc-macro function**

In `pixelflow-compiler/src/lib.rs`, replace the body of `pub fn kernel_jit(input: TokenStream) -> TokenStream`:

```rust
#[proc_macro]
pub fn kernel_jit(input: TokenStream) -> TokenStream {
    let tokens = proc_macro2::TokenStream::from(input);

    let kernel_ast = match parser::parse(tokens) {
        Ok(ast) => ast,
        Err(e) => return e.to_compile_error().into(),
    };

    let analyzed = match sema::analyze(kernel_ast) {
        Ok(a) => a,
        Err(e) => return e.to_compile_error().into(),
    };

    // Collect scalar params in declaration order
    let scalar_params: Vec<_> = analyzed
        .def
        .params
        .iter()
        .filter(|p| matches!(p.kind, crate::ast::ParamKind::Scalar(_)))
        .collect();

    let param_map = ir_bridge::scalar_param_indices(&analyzed);

    let ir = match ir_bridge::ast_to_ir(&analyzed.def.body, &param_map) {
        Ok(ir) => ir,
        Err(e) => {
            return syn::Error::new(proc_macro2::Span::call_site(), e)
                .to_compile_error()
                .into()
        }
    };

    let expr_code = ir_bridge::ir_to_runtime_expr(&ir);

    if scalar_params.is_empty() {
        // Zero-param case: compile immediately, return JitManifold directly
        let output = quote! {
            {
                let expr = #expr_code;
                let code = ::pixelflow_ir::backend::emit::compile(&expr)
                    .expect("kernel_jit! JIT compilation failed");
                ::pixelflow_ir::JitManifold::new(code)
            }
        };
        output.into()
    } else {
        // N-param case: emit a builder closure
        // Generate: move |p0: f32, p1: f32, ...| -> JitManifold { ... }
        let param_names: Vec<proc_macro2::Ident> = scalar_params
            .iter()
            .map(|p| p.name.clone())
            .collect();
        let param_types: Vec<proc_macro2::TokenStream> = scalar_params
            .iter()
            .map(|_| quote! { f32 })
            .collect();
        // Build the &[f32] slice in declaration order (index 0 = first param)
        let param_slice = quote! { &[ #( #param_names as f32 ),* ] };

        let output = quote! {
            move | #( #param_names : #param_types ),* | -> ::pixelflow_ir::JitManifold {
                let expr = #expr_code;
                let expr = ::pixelflow_ir::substitute_params(&expr, #param_slice);
                let code = ::pixelflow_ir::backend::emit::compile(&expr)
                    .expect("kernel_jit! JIT compilation failed");
                ::pixelflow_ir::JitManifold::new(code)
            }
        };
        output.into()
    }
}
```

**Step 4: Run the tests**

```bash
cargo test -p pixelflow-compiler 2>&1 | tail -30
```

Expected: All 4 new tests pass. The existing `kernel_jit` tests (if any) may need updating since the return type changed.

**Step 5: Fix any existing tests that called the old `kernel_jit!` API**

Search for existing uses:
```bash
grep -r "kernel_jit" /Users/jppittman/Documents/Projects/core-term/.claude/worktrees/crazy-montalcini --include="*.rs" -l
```

Update any tests that expected a raw `KernelFn` to use the new `JitManifold` API.

**Step 6: Run the full workspace tests**

```bash
cargo test --workspace 2>&1 | tail -40
```

Expected: All tests pass.

**Step 7: Commit**

```bash
git add pixelflow-compiler/src/lib.rs pixelflow-compiler/tests/kernel_jit.rs
git commit -m "feat(pixelflow-compiler): rewrite kernel_jit! to emit builder closure with JitManifold"
```

---

## Task 6: Emitter integration test with substituted params

**Files:**
- Modify: `pixelflow-ir/src/backend/emit/executable.rs` (add test at bottom)

**Step 1: Write the test**

Add to the test section of `executable.rs`:

```rust
#[test]
#[cfg(target_arch = "aarch64")]
fn test_compile_with_param_substitution() {
    use crate::{substitute_params, Expr, OpKind};

    // Simulate kernel_jit!(|offset: f32| X + offset) called with offset=32.0
    // Before substitution: X + Param(0)
    let expr_with_param = Expr::Binary(
        OpKind::Add,
        Box::new(Expr::Var(0)),
        Box::new(Expr::Param(0)),
    );

    // After substitution: X + Const(32.0)
    let expr = substitute_params(&expr_with_param, &[32.0_f32]);

    let exec = compile(&expr).expect("compile failed");
    unsafe {
        let func: KernelFn = exec.as_fn();
        use core::arch::aarch64::*;
        let x = vdupq_n_f32(10.0);
        let result = func(x, vdupq_n_f32(0.0), vdupq_n_f32(0.0), vdupq_n_f32(0.0));
        assert_eq!(vgetq_lane_f32(result, 0), 42.0);
    }
}

#[test]
#[cfg(target_arch = "aarch64")]
fn test_compile_two_params_substituted() {
    use crate::{substitute_params, Expr, OpKind};

    // Simulate kernel_jit!(|cx: f32, r: f32| (X - cx) * r) called with cx=1.0, r=2.0
    // Param(0)=cx=1.0, Param(1)=r=2.0
    let expr_with_params = Expr::Binary(
        OpKind::Mul,
        Box::new(Expr::Binary(
            OpKind::Sub,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Param(0)),
        )),
        Box::new(Expr::Param(1)),
    );

    let expr = substitute_params(&expr_with_params, &[1.0_f32, 2.0_f32]);

    let exec = compile(&expr).expect("compile failed");
    unsafe {
        let func: KernelFn = exec.as_fn();
        use core::arch::aarch64::*;
        // X=5.0: (5.0 - 1.0) * 2.0 = 8.0
        let x = vdupq_n_f32(5.0);
        let result = func(x, vdupq_n_f32(0.0), vdupq_n_f32(0.0), vdupq_n_f32(0.0));
        assert_eq!(vgetq_lane_f32(result, 0), 8.0);
    }
}
```

**Step 2: Run the tests**

```bash
cargo test -p pixelflow-ir test_compile_with_param 2>&1 | tail -20
cargo test -p pixelflow-ir test_compile_two_params 2>&1 | tail -20
```

Expected: Both pass. The emitter never sees `Param` nodes — `substitute_params` handles them before the expr reaches `compile`.

**Step 3: Commit**

```bash
git add pixelflow-ir/src/backend/emit/executable.rs
git commit -m "test(pixelflow-ir): add emitter integration tests for param substitution"
```

---

## Task 7: Full workspace clean build and test

**Step 1: Clean build**

```bash
cargo build --workspace 2>&1 | tail -20
```

Expected: Zero errors, zero warnings about unused imports or dead code.

**Step 2: Full test suite**

```bash
cargo test --workspace 2>&1 | tail -40
```

Expected: All tests pass.

**Step 3: Verify `kernel_jit!` semantic parity manually**

If you have a way to run a small binary, write a quick smoke test in `pixelflow-ir/src/lib.rs` or as a doc-test:

```rust
// kernel_jit!(|cx: f32| X - cx) with cx=3.0, eval at X=10.0 → 7.0
// kernel!(|cx: f32| X - cx) with cx=3.0, eval at X=10.0 → 7.0
// Both must agree.
```

**Step 4: Final commit**

```bash
git commit --allow-empty -m "feat: kernel_jit! feature parity with kernel! complete"
```

---

## Summary of Changes

| File | Type | What Changed |
|------|------|--------------|
| `pixelflow-ir/src/expr.rs` | Modified | Added `Expr::Param(u8)` variant |
| `pixelflow-ir/src/lib.rs` | Modified | Added `substitute_params`, `JitManifold` re-export, tests |
| `pixelflow-ir/src/jit_manifold.rs` | Created | `JitManifold` struct + `Manifold` impl |
| `pixelflow-compiler/src/ir_bridge.rs` | Modified | `ast_to_ir` maps params to `Expr::Param`, `scalar_param_indices` helper |
| `pixelflow-compiler/src/lib.rs` | Modified | `kernel_jit!` emits builder closure + `JitManifold` |
| `pixelflow-compiler/tests/kernel_jit.rs` | Created | Macro round-trip tests |
| `pixelflow-ir/src/backend/emit/executable.rs` | Modified | Two param substitution integration tests |

**No changes to:**
- The JIT emitter (`emit/mod.rs`, `emit/aarch64.rs`, `emit/x86_64.rs`)
- The calling convention or `KernelFn` type
- `ExecutableCode`
- `kernel!` (compile-time macro)
