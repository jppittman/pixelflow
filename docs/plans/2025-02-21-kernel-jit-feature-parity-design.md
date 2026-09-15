> **Superseded (2026-07-20); its subject and every file it names are gone.**
> `kernel_jit!` was retired with the dual-backend framing by
> [`2026-07-20-kernel-unification.md`](2026-07-20-kernel-unification.md) — the
> param-baking scope specified here *did* land, and the `Param` semantics below are
> still the ones in force. Deleted since: `pixelflow-ir/src/expr.rs` (the Arc-tree
> `Expr`, replaced by `ExprArena`), `pixelflow-ir/src/jit_manifold.rs`
> (`JitManifold`), and `pixelflow-compiler/src/ir_bridge.rs` (split into `lower.rs`
> and `emit.rs` by #1206).

# kernel_jit! Feature Parity Design

**Date:** 2025-02-21
**Status:** Approved

## Goal

Bring `kernel_jit!` to full feature parity with `kernel!`. Both macros should have identical semantics from the caller's perspective — the only difference is LLVM (compile-time) vs the custom emitter (runtime JIT) under the hood.

```rust
// kernel! — compile-time LLVM
let k = kernel!(|cx: f32, r: f32| (X - cx) * r);
let manifold = k(1.0, 2.0);  // returns impl Manifold, params baked in via LLVM

// kernel_jit! — runtime JIT, identical semantics
let k = kernel_jit!(|cx: f32, r: f32| (X - cx) * r);
let manifold = k(1.0, 2.0);  // JITs immediately, params folded as Expr::Const
```

## Semantic Model

Parameters and coordinates are distinct by kind, not by calling convention:

- **Parameters** (`cx: f32`) — baked into the kernel at build time. Different values = different kernel. This is the "build the kernel you want" model.
- **Coordinates** (X, Y, Z, W) — the eval-time inputs that stream through every pixel call.

If a value changes per-frame, it belongs as a coordinate, not a parameter. Building a kernel with the wrong thing as a parameter is a design error, not a runtime problem to cache around.

## Caching

**No cache.** `kernel_jit!` compiles and returns an owned `JitManifold` immediately. The caller decides lifetime and caching. This is the suckless-aligned choice — don't build infrastructure the caller didn't ask for.

## Design

### Section 1: IR Changes (`pixelflow-ir`)

Add `Expr::Param(index: u8)` to `pixelflow-ir/src/expr.rs`. This node represents a captured parameter — a scalar `f32` that will be constant-folded into `Expr::Const` before JIT compilation.

`Param` is **ephemeral**: it exists only in the pre-compilation IR tree and never reaches the emitter. A new function:

```rust
fn substitute_params(expr: &Expr, params: &[f32]) -> Expr
```

walks the tree recursively, replacing `Param(i)` with `Const(params[i])`. Called at the call site (when `k(1.0, 2.0)` is invoked), before passing to `compile()`.

Panics if `params.len()` doesn't match the number of `Param` nodes — this is always a macro bug, not a user error.

### Section 2: `ir_bridge` Changes (`pixelflow-compiler`)

In `pixelflow-compiler/src/ir_bridge.rs`, extend `ast_to_ir` to handle parameter identifiers. Parameters are already tracked in order by semantic analysis — `cx` is param 0, `cy` is param 1, etc. The bridge maps each identifier to `Expr::Param(index)` instead of erroring.

### Section 3: `kernel_jit!` Output Shape (`pixelflow-compiler`)

The macro returns a builder closure matching `kernel!` exactly:

```rust
// kernel_jit!(|cx: f32, r: f32| (X - cx) * r) expands to:
move |cx: f32, r: f32| -> JitManifold {
    // Expr tree with Param nodes — generated at macro expansion time
    let expr = /* ... Expr::Param(0), Expr::Param(1) ... */;
    // Substitute params → constants
    let expr = substitute_params(&expr, &[cx, r]);
    // JIT compile
    let code = compile(&expr).expect("kernel_jit! JIT compilation failed");
    JitManifold::new(code)
}
```

`JitManifold` is a small struct in `pixelflow-ir` that:
- Owns `ExecutableCode`
- Implements `Manifold<(Field, Field, Field, Field), Output = Field>`
- Its `eval` calls the function pointer directly
- Is `Send + Sync` (read-only mapped memory)

For the zero-parameter case (`kernel_jit!(|| X + Y)`), the macro returns a `JitManifold` directly — same as current behavior, just wrapped in the new type.

### Section 4: Error Handling

Fail fast, fail loud, no silent failures.

- `compile()` returns `Result` — builder closure calls `.expect("kernel_jit! JIT compilation failed")`. Panics on failure. No fallback.
- `substitute_params` panics on param count mismatch. This is always a bug in generated code.
- `JitManifold::eval` calls the function pointer unsafely. No catch, no fallback. Wrong emitted code surfaces immediately.

### Section 5: Testing

**IR unit tests** (`pixelflow-ir`):
- `substitute_params` replaces `Param(0)` with `Const(1.0)` correctly
- `substitute_params` panics on param count mismatch

**Emitter integration tests** (`pixelflow-ir`):
- Build `Expr` with `Param` nodes, substitute, compile, verify output matches hand-calculated result

**Macro round-trip tests** (`pixelflow-compiler`):
- `kernel_jit!(|cx: f32| X + cx)` called with `cx = 1.0`, eval at `X = 2.0` → asserts `3.0`
- Mirrors existing `kernel!` tests exactly

## Files to Change

| File | Change |
|------|--------|
| `pixelflow-ir/src/expr.rs` | Add `Expr::Param(u8)` variant |
| `pixelflow-ir/src/lib.rs` | Add `substitute_params` fn, re-export |
| `pixelflow-ir/src/backend/emit/mod.rs` | No change needed (Param never reaches emitter) |
| `pixelflow-compiler/src/ir_bridge.rs` | Map param idents to `Expr::Param(i)` |
| `pixelflow-compiler/src/lib.rs` | `kernel_jit!` emits builder closure, wraps in `JitManifold` |
| `pixelflow-ir/src/jit_manifold.rs` | New file: `JitManifold` struct + `Manifold` impl |

## What Does Not Change

- The JIT emitter (`emit/mod.rs`, `emit/aarch64.rs`, `emit/x86_64.rs`) — no changes needed
- The calling convention — still `(Field, Field, Field, Field) -> Field`
- `KernelFn` type — unchanged
- `ExecutableCode` — unchanged
