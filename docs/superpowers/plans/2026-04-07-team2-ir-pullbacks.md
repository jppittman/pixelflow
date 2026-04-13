# Team 2: IR Pullback Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Register a VJP (vector-Jacobian product) pullback rule for every `OpKind` in `pixelflow-ir`, then implement `emit_backward` in `pixelflow-compiler` that walks an `ExprArena` in reverse topological order and emits the backward pass as a new `ExprArena`.

**Architecture:** Each `OpKind` gets a `pullback` match arm: `(output_grad_node, input_nodes) → Vec<input_grad_nodes>`. `emit_backward` does a standard reverse-mode backprop: topological sort, reverse traversal, gradient accumulation at nodes with multiple consumers. No new external dependencies.

**Tech Stack:** Rust stable, `pixelflow-ir`, `pixelflow-compiler`.

---

## Context You Need

Read before starting:
- `pixelflow-ir/src/kind.rs` — `OpKind` enum (41 variants, `COUNT = 41`)
- `pixelflow-ir/src/arena.rs` — `ExprArena`, `ExprId`, `ExprNode` variants
- `pixelflow-compiler/src/ir_bridge.rs` — how the compiler uses the arena

Key `ExprArena` API:
```rust
arena.push_var(i: u8) -> ExprId          // Var(i) — coordinate variable (0=X,1=Y,2=Z,3=W)
arena.push_const(v: f32) -> ExprId       // Const(v)
arena.push_unary(op, child) -> ExprId    // Unary(op, child)
arena.push_binary(op, a, b) -> ExprId   // Binary(op, a, b)
arena.push_ternary(op, a, b, c) -> ExprId // Ternary(op, a, b, c)
arena.get(id) -> &ExprNode               // read a node
arena.len() -> usize
```

`ExprNode` variants:
```rust
ExprNode::Var(u8)
ExprNode::Const(f32)
ExprNode::Param(u8)
ExprNode::Unary(OpKind, ExprId)          // one input
ExprNode::Binary(OpKind, ExprId, ExprId) // two inputs
ExprNode::Ternary(OpKind, ExprId, ExprId, ExprId) // three inputs (Select: cond, then, else)
ExprNode::Nary(OpKind, u32, u16)         // n inputs (rarely used here)
```

`OpKind` variants used in the network (and their pullback rules):
```
Add(a,b)   → (d, d)                          // d flows to both
Sub(a,b)   → (d, Neg(d))                     // negate for b
Mul(a,b)   → (Mul(d,b), Mul(d,a))           // product rule
Neg(x)     → Neg(d)
Sqrt(x)    → Mul(d, Mul(Const(0.5), Rsqrt(x)))   // d * 0.5 / sqrt(x)
Rsqrt(x)   → Mul(d, Mul(Const(-0.5), Mul(Rsqrt(x), Mul(Rsqrt(x), Rsqrt(x)))))
Abs(x)     → Mul(d, Select(Ge(x,0), Const(1.0), Const(-1.0)))  // sign(x)
Exp(x)     → Mul(d, Exp(x))                 // d * exp(x)
Ln(x)      → Mul(d, Recip(x))               // d / x = d * recip(x)
Log2(x)    → Mul(d, Mul(Const(1.0/LN2), Recip(x)))  // d / (x * ln2)
Sin(x)     → Mul(d, Cos(x))
Cos(x)     → Neg(Mul(d, Sin(x)))
Max(a,b)   → (Mul(d, Ge(a,b)), Mul(d, Lt(a,b)))  // step function gate
Min(a,b)   → (Mul(d, Le(a,b)), Mul(d, Gt(a,b)))
Select(c,t,f) → (Const(0.0), Mul(d, Select(c,Const(1.0),Const(0.0))), Mul(d, Select(c,Const(0.0),Const(1.0))))
Var        → Const(0.0)  // no gradient w.r.t. variables (they're coordinates, not params)
Const      → Const(0.0)
```

`LN2 = 0.693147180559945_f32`

---

## File Structure

| File | Change |
|------|--------|
| `pixelflow-ir/src/pullback.rs` | New file: `pullback_inputs` function |
| `pixelflow-ir/src/lib.rs` | `pub mod pullback;` + re-export |
| `pixelflow-compiler/src/backward.rs` | New file: `emit_backward` function |
| `pixelflow-compiler/src/lib.rs` | `pub mod backward;` + re-export |

---

## Task 1: Pullback rules for arithmetic ops

**Files:** Create `pixelflow-ir/src/pullback.rs`

- [ ] **Step 1: Write the failing tests**

Create `pixelflow-ir/src/pullback.rs` with this test module first:

```rust
//! VJP pullback rules for ExprArena nodes.
//!
//! Each `OpKind` maps to a pullback rule:
//! given the output gradient node and the input nodes, produce input gradient nodes.
//!
//! `pullback_inputs(op, d_out, inputs, arena)` returns one `ExprId` per input.

use crate::arena::{ExprArena, ExprId, ExprNode};
use crate::kind::OpKind;

const LN2: f32 = 0.693_147_18_f32;

/// Compute input gradients for a single node.
///
/// - `op`: the operation of the node
/// - `d_out`: ExprId of the output gradient (flowing in from above)
/// - `inputs`: slice of input ExprIds (the original forward inputs)
/// - `arena`: arena to push new gradient nodes into
///
/// Returns one gradient ExprId per input (same length as `inputs`).
///
/// # Panics
///
/// Panics if `inputs` has the wrong arity for `op`.
pub fn pullback_inputs(
    op: OpKind,
    d_out: ExprId,
    inputs: &[ExprId],
    arena: &mut ExprArena,
) -> Vec<ExprId> {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> ExprArena {
        ExprArena::new()
    }

    #[test]
    fn pullback_add_passes_grad_to_both() {
        let mut arena = setup();
        let a = arena.push_var(0);
        let b = arena.push_var(1);
        let d = arena.push_const(1.0);
        let grads = pullback_inputs(OpKind::Add, d, &[a, b], &mut arena);
        assert_eq!(grads.len(), 2);
        // Both gradients should be the same node (d itself)
        assert_eq!(grads[0], d);
        assert_eq!(grads[1], d);
    }

    #[test]
    fn pullback_sub_negates_rhs_grad() {
        let mut arena = setup();
        let a = arena.push_var(0);
        let b = arena.push_var(1);
        let d = arena.push_const(1.0);
        let grads = pullback_inputs(OpKind::Sub, d, &[a, b], &mut arena);
        assert_eq!(grads.len(), 2);
        // d_a = d, d_b = -d
        assert_eq!(grads[0], d);
        // d_b should be Neg(d)
        assert_eq!(arena.get(grads[1]), &ExprNode::Unary(OpKind::Neg, d));
    }

    #[test]
    fn pullback_mul_product_rule() {
        let mut arena = setup();
        let a = arena.push_var(0); // a
        let b = arena.push_var(1); // b
        let d = arena.push_const(1.0);
        let grads = pullback_inputs(OpKind::Mul, d, &[a, b], &mut arena);
        assert_eq!(grads.len(), 2);
        // d_a = d * b, d_b = d * a
        assert_eq!(arena.get(grads[0]), &ExprNode::Binary(OpKind::Mul, d, b));
        assert_eq!(arena.get(grads[1]), &ExprNode::Binary(OpKind::Mul, d, a));
    }

    #[test]
    fn pullback_neg_negates_grad() {
        let mut arena = setup();
        let x = arena.push_var(0);
        let d = arena.push_const(1.0);
        let grads = pullback_inputs(OpKind::Neg, d, &[x], &mut arena);
        assert_eq!(grads.len(), 1);
        assert_eq!(arena.get(grads[0]), &ExprNode::Unary(OpKind::Neg, d));
    }
}
```

- [ ] **Step 2: Add pullback.rs to pixelflow-ir/src/lib.rs**

In `pixelflow-ir/src/lib.rs`, add:

```rust
pub mod pullback;
pub use pullback::pullback_inputs;
```

- [ ] **Step 3: Run tests to confirm they fail**

```bash
cargo test -p pixelflow-ir pullback
```

Expected: compile errors or test failures on `todo!()`

- [ ] **Step 4: Implement arithmetic pullbacks**

Replace `todo!()` in `pullback_inputs` with:

```rust
pub fn pullback_inputs(
    op: OpKind,
    d_out: ExprId,
    inputs: &[ExprId],
    arena: &mut ExprArena,
) -> Vec<ExprId> {
    match op {
        OpKind::Add => {
            assert_eq!(inputs.len(), 2, "Add requires 2 inputs");
            vec![d_out, d_out]
        }
        OpKind::Sub => {
            assert_eq!(inputs.len(), 2, "Sub requires 2 inputs");
            let neg_d = arena.push_unary(OpKind::Neg, d_out);
            vec![d_out, neg_d]
        }
        OpKind::Mul => {
            assert_eq!(inputs.len(), 2, "Mul requires 2 inputs");
            let [a, b] = [inputs[0], inputs[1]];
            let d_a = arena.push_binary(OpKind::Mul, d_out, b);
            let d_b = arena.push_binary(OpKind::Mul, d_out, a);
            vec![d_a, d_b]
        }
        OpKind::Neg => {
            assert_eq!(inputs.len(), 1, "Neg requires 1 input");
            let neg_d = arena.push_unary(OpKind::Neg, d_out);
            vec![neg_d]
        }
        OpKind::Var | OpKind::Const | OpKind::Param => {
            // Leaves: no inputs to propagate to. Return empty.
            vec![]
        }
        _ => todo!("pullback_inputs: {:?} not yet implemented", op),
    }
}
```

- [ ] **Step 5: Run tests**

```bash
cargo test -p pixelflow-ir pullback
```

Expected: `pullback_add_passes_grad_to_both ... ok`, `pullback_sub_negates_rhs_grad ... ok`, `pullback_mul_product_rule ... ok`, `pullback_neg_negates_grad ... ok`

- [ ] **Step 6: Commit**

```bash
git add pixelflow-ir/src/pullback.rs pixelflow-ir/src/lib.rs
git commit -m "feat(ir): pullback rules for Add, Sub, Mul, Neg"
```

---

## Task 2: Pullback rules for unary math ops

**Files:** `pixelflow-ir/src/pullback.rs`

- [ ] **Step 1: Write failing tests**

Add to `#[cfg(test)]` in `pullback.rs`:

```rust
#[test]
fn pullback_exp() {
    let mut arena = setup();
    let x = arena.push_var(0);
    let exp_x = arena.push_unary(OpKind::Exp, x);
    let d = arena.push_const(1.0);
    let grads = pullback_inputs(OpKind::Exp, d, &[x], &mut arena);
    assert_eq!(grads.len(), 1);
    // d_x = d * exp(x)
    assert_eq!(arena.get(grads[0]), &ExprNode::Binary(OpKind::Mul, d, exp_x));
}

#[test]
fn pullback_sqrt() {
    let mut arena = setup();
    let x = arena.push_var(0);
    let d = arena.push_const(1.0);
    let grads = pullback_inputs(OpKind::Sqrt, d, &[x], &mut arena);
    assert_eq!(grads.len(), 1);
    // d_x = d * 0.5 * rsqrt(x) = d * 0.5 / sqrt(x)
    // structure: Mul(d, Mul(Const(0.5), Rsqrt(x)))
    let half = arena.push_const(0.5);
    let rsqrt_x = arena.push_unary(OpKind::Rsqrt, x);
    let half_rsqrt = arena.push_binary(OpKind::Mul, half, rsqrt_x);
    let expected = arena.push_binary(OpKind::Mul, d, half_rsqrt);
    assert_eq!(arena.get(grads[0]), arena.get(expected));
}

#[test]
fn pullback_ln() {
    let mut arena = setup();
    let x = arena.push_var(0);
    let d = arena.push_const(1.0);
    let grads = pullback_inputs(OpKind::Ln, d, &[x], &mut arena);
    assert_eq!(grads.len(), 1);
    // d_x = d * recip(x) = d / x
    let recip_x = arena.push_unary(OpKind::Recip, x);
    let expected = arena.push_binary(OpKind::Mul, d, recip_x);
    assert_eq!(arena.get(grads[0]), arena.get(expected));
}

#[test]
fn pullback_sin() {
    let mut arena = setup();
    let x = arena.push_var(0);
    let d = arena.push_const(1.0);
    let grads = pullback_inputs(OpKind::Sin, d, &[x], &mut arena);
    assert_eq!(grads.len(), 1);
    // d_x = d * cos(x)
    let cos_x = arena.push_unary(OpKind::Cos, x);
    let expected = arena.push_binary(OpKind::Mul, d, cos_x);
    assert_eq!(arena.get(grads[0]), arena.get(expected));
}

#[test]
fn pullback_cos() {
    let mut arena = setup();
    let x = arena.push_var(0);
    let d = arena.push_const(1.0);
    let grads = pullback_inputs(OpKind::Cos, d, &[x], &mut arena);
    assert_eq!(grads.len(), 1);
    // d_x = -d * sin(x) = Neg(Mul(d, Sin(x)))
    let sin_x = arena.push_unary(OpKind::Sin, x);
    let d_sin = arena.push_binary(OpKind::Mul, d, sin_x);
    let expected = arena.push_unary(OpKind::Neg, d_sin);
    assert_eq!(arena.get(grads[0]), arena.get(expected));
}
```

- [ ] **Step 2: Run to confirm failures**

```bash
cargo test -p pixelflow-ir pullback_exp pullback_sqrt pullback_ln pullback_sin pullback_cos
```

Expected: failures with `not yet implemented` panic

- [ ] **Step 3: Implement unary math pullbacks**

Add these arms to the `match op` in `pullback_inputs` (before the `_ => todo!` arm):

```rust
        OpKind::Sqrt => {
            assert_eq!(inputs.len(), 1);
            // d * 0.5 * rsqrt(x)
            let half = arena.push_const(0.5);
            let rsqrt_x = arena.push_unary(OpKind::Rsqrt, inputs[0]);
            let half_rsqrt = arena.push_binary(OpKind::Mul, half, rsqrt_x);
            let d_x = arena.push_binary(OpKind::Mul, d_out, half_rsqrt);
            vec![d_x]
        }
        OpKind::Rsqrt => {
            assert_eq!(inputs.len(), 1);
            // d/dx rsqrt(x) = -0.5 * x^(-3/2) = -0.5 * rsqrt(x)^3
            let rsqrt_x = arena.push_unary(OpKind::Rsqrt, inputs[0]);
            let rsqrt2 = arena.push_binary(OpKind::Mul, rsqrt_x, rsqrt_x);
            let rsqrt3 = arena.push_binary(OpKind::Mul, rsqrt2, rsqrt_x);
            let neg_half = arena.push_const(-0.5);
            let scale = arena.push_binary(OpKind::Mul, neg_half, rsqrt3);
            let d_x = arena.push_binary(OpKind::Mul, d_out, scale);
            vec![d_x]
        }
        OpKind::Abs => {
            assert_eq!(inputs.len(), 1);
            // sign(x): 1 if x >= 0, -1 otherwise
            let zero = arena.push_const(0.0);
            let one = arena.push_const(1.0);
            let neg_one = arena.push_const(-1.0);
            let ge_zero = arena.push_binary(OpKind::Ge, inputs[0], zero);
            let sign = arena.push_ternary(OpKind::Select, ge_zero, one, neg_one);
            let d_x = arena.push_binary(OpKind::Mul, d_out, sign);
            vec![d_x]
        }
        OpKind::Recip => {
            assert_eq!(inputs.len(), 1);
            // d/dx (1/x) = -1/x^2
            let neg_one = arena.push_const(-1.0);
            let x2 = arena.push_binary(OpKind::Mul, inputs[0], inputs[0]);
            let neg_recip2 = arena.push_binary(OpKind::Mul, neg_one,
                arena.push_unary(OpKind::Recip, x2));
            let d_x = arena.push_binary(OpKind::Mul, d_out, neg_recip2);
            vec![d_x]
        }
        OpKind::Exp => {
            assert_eq!(inputs.len(), 1);
            // d * exp(x) — need exp(x) node; push it fresh (CSE is the JIT's job)
            let exp_x = arena.push_unary(OpKind::Exp, inputs[0]);
            let d_x = arena.push_binary(OpKind::Mul, d_out, exp_x);
            vec![d_x]
        }
        OpKind::Exp2 => {
            assert_eq!(inputs.len(), 1);
            // d/dx 2^x = 2^x * ln(2)
            let exp2_x = arena.push_unary(OpKind::Exp2, inputs[0]);
            let ln2 = arena.push_const(LN2);
            let scale = arena.push_binary(OpKind::Mul, exp2_x, ln2);
            let d_x = arena.push_binary(OpKind::Mul, d_out, scale);
            vec![d_x]
        }
        OpKind::Ln => {
            assert_eq!(inputs.len(), 1);
            // d / x
            let recip_x = arena.push_unary(OpKind::Recip, inputs[0]);
            let d_x = arena.push_binary(OpKind::Mul, d_out, recip_x);
            vec![d_x]
        }
        OpKind::Log2 => {
            assert_eq!(inputs.len(), 1);
            // d / (x * ln2)
            let recip_x = arena.push_unary(OpKind::Recip, inputs[0]);
            let inv_ln2 = arena.push_const(1.0 / LN2);
            let scale = arena.push_binary(OpKind::Mul, recip_x, inv_ln2);
            let d_x = arena.push_binary(OpKind::Mul, d_out, scale);
            vec![d_x]
        }
        OpKind::Log10 => {
            assert_eq!(inputs.len(), 1);
            // d / (x * ln10)
            const LN10: f32 = 2.302_585_1_f32;
            let recip_x = arena.push_unary(OpKind::Recip, inputs[0]);
            let inv_ln10 = arena.push_const(1.0 / LN10);
            let scale = arena.push_binary(OpKind::Mul, recip_x, inv_ln10);
            let d_x = arena.push_binary(OpKind::Mul, d_out, scale);
            vec![d_x]
        }
        OpKind::Sin => {
            assert_eq!(inputs.len(), 1);
            let cos_x = arena.push_unary(OpKind::Cos, inputs[0]);
            let d_x = arena.push_binary(OpKind::Mul, d_out, cos_x);
            vec![d_x]
        }
        OpKind::Cos => {
            assert_eq!(inputs.len(), 1);
            let sin_x = arena.push_unary(OpKind::Sin, inputs[0]);
            let d_sin = arena.push_binary(OpKind::Mul, d_out, sin_x);
            let d_x = arena.push_unary(OpKind::Neg, d_sin);
            vec![d_x]
        }
        OpKind::Floor | OpKind::Ceil | OpKind::Round | OpKind::Fract => {
            assert_eq!(inputs.len(), 1);
            // Piecewise constant: gradient is zero almost everywhere.
            let zero = arena.push_const(0.0);
            vec![zero]
        }
```

- [ ] **Step 4: Run tests**

```bash
cargo test -p pixelflow-ir pullback
```

Expected: All pullback tests pass.

- [ ] **Step 5: Commit**

```bash
git add pixelflow-ir/src/pullback.rs
git commit -m "feat(ir): pullback rules for unary math ops (sqrt, exp, ln, log2, trig)"
```

---

## Task 3: Pullback rules for Max, Min, Select (ReLU support)

**Files:** `pixelflow-ir/src/pullback.rs`

- [ ] **Step 1: Write failing tests**

```rust
#[test]
fn pullback_max_relu_shape() {
    // ReLU is Max(x, 0). d_x = d * (x > 0)
    let mut arena = setup();
    let x = arena.push_var(0);
    let zero = arena.push_const(0.0);
    let d = arena.push_const(1.0);
    let grads = pullback_inputs(OpKind::Max, d, &[x, zero], &mut arena);
    assert_eq!(grads.len(), 2);
    // d_a = d * (a >= b): d * Ge(x, 0)
    let ge = arena.push_binary(OpKind::Ge, x, zero);
    let expected_da = arena.push_binary(OpKind::Mul, d, ge);
    assert_eq!(arena.get(grads[0]), arena.get(expected_da));
}

#[test]
fn pullback_select_routes_grad() {
    // Select(cond, then, else): d_then = d if cond else 0, d_else = 0 if cond else d
    let mut arena = setup();
    let cond = arena.push_var(0);
    let t = arena.push_var(1);
    let f = arena.push_var(2);
    let d = arena.push_const(1.0);
    let grads = pullback_inputs(OpKind::Select, d, &[cond, t, f], &mut arena);
    assert_eq!(grads.len(), 3);
    // d_cond = 0 (not differentiable)
    assert_eq!(arena.get(grads[0]), &ExprNode::Const(0.0));
    // d_then = Select(cond, d, 0)
    let zero = arena.push_const(0.0);
    let expected_dt = arena.push_ternary(OpKind::Select, cond, d, zero);
    assert_eq!(arena.get(grads[1]), arena.get(expected_dt));
}
```

- [ ] **Step 2: Run to confirm failures**

```bash
cargo test -p pixelflow-ir pullback_max pullback_select
```

- [ ] **Step 3: Implement**

Add to the `match op` block:

```rust
        OpKind::Max => {
            assert_eq!(inputs.len(), 2);
            let [a, b] = [inputs[0], inputs[1]];
            // d_a = d * (a >= b), d_b = d * (a < b)
            let ge_ab = arena.push_binary(OpKind::Ge, a, b);
            let lt_ab = arena.push_binary(OpKind::Lt, a, b);
            let d_a = arena.push_binary(OpKind::Mul, d_out, ge_ab);
            let d_b = arena.push_binary(OpKind::Mul, d_out, lt_ab);
            vec![d_a, d_b]
        }
        OpKind::Min => {
            assert_eq!(inputs.len(), 2);
            let [a, b] = [inputs[0], inputs[1]];
            // d_a = d * (a <= b), d_b = d * (a > b)
            let le_ab = arena.push_binary(OpKind::Le, a, b);
            let gt_ab = arena.push_binary(OpKind::Gt, a, b);
            let d_a = arena.push_binary(OpKind::Mul, d_out, le_ab);
            let d_b = arena.push_binary(OpKind::Mul, d_out, gt_ab);
            vec![d_a, d_b]
        }
        OpKind::Select => {
            assert_eq!(inputs.len(), 3);
            let [cond, t, f] = [inputs[0], inputs[1], inputs[2]];
            // cond is not differentiable
            let zero = arena.push_const(0.0);
            let d_t = arena.push_ternary(OpKind::Select, cond, d_out, zero);
            let d_f = arena.push_ternary(OpKind::Select, cond, zero, d_out);
            vec![zero, d_t, d_f]
        }
        OpKind::Clamp => {
            assert_eq!(inputs.len(), 3);
            let [x, lo, hi] = [inputs[0], inputs[1], inputs[2]];
            // Gradient passes through only when lo < x < hi
            let zero = arena.push_const(0.0);
            let ge_lo = arena.push_binary(OpKind::Ge, x, lo);
            let le_hi = arena.push_binary(OpKind::Le, x, hi);
            // in_range = ge_lo AND le_hi: use Mul(ge, le) as boolean AND
            let in_range = arena.push_binary(OpKind::Mul, ge_lo, le_hi);
            let d_x = arena.push_binary(OpKind::Mul, d_out, in_range);
            vec![d_x, zero, zero]  // d_lo = 0, d_hi = 0
        }
        OpKind::MulAdd => {
            assert_eq!(inputs.len(), 3);
            let [a, b, c] = [inputs[0], inputs[1], inputs[2]];
            // MulAdd(a, b, c) = a*b + c
            let d_a = arena.push_binary(OpKind::Mul, d_out, b);
            let d_b = arena.push_binary(OpKind::Mul, d_out, a);
            let d_c = d_out;
            vec![d_a, d_b, d_c]
        }
        // Comparison ops: not differentiable (produce masks, not floats)
        OpKind::Lt | OpKind::Le | OpKind::Gt | OpKind::Ge | OpKind::Eq | OpKind::Ne => {
            let zero = arena.push_const(0.0);
            vec![zero, zero]
        }
        OpKind::Tuple => {
            // Pass gradient to each child unchanged
            inputs.iter().map(|_| d_out).collect()
        }
        // Remaining ops (Div, Pow, Hypot, Atan2, Tan, Asin, Acos, Atan): not needed
        // for the NNUE network. Add when needed.
        _ => {
            panic!(
                "pullback_inputs: {:?} not implemented. \
                 Add a rule in pixelflow-ir/src/pullback.rs",
                op
            )
        }
```

- [ ] **Step 4: Run full pullback tests**

```bash
cargo test -p pixelflow-ir pullback
```

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add pixelflow-ir/src/pullback.rs
git commit -m "feat(ir): pullback rules for Max, Min, Select, Clamp, MulAdd"
```

---

## Task 4: emit_backward — reverse-mode pass over ExprArena

**Files:** Create `pixelflow-compiler/src/backward.rs`

- [ ] **Step 1: Write the failing test**

Create `pixelflow-compiler/src/backward.rs`:

```rust
//! Reverse-mode automatic differentiation over `ExprArena`.
//!
//! `emit_backward(arena, output_id)` appends gradient computation nodes to `arena`
//! and returns a map from each node's `ExprId` to its gradient `ExprId`.
//!
//! Algorithm:
//! 1. Topological sort (children before parents, since arena is append-only this is just
//!    forward order — nodes always reference earlier nodes).
//! 2. Assign the output gradient: grad[output_id] = Const(1.0).
//! 3. Reverse traversal: for each node (latest first), apply pullback_inputs,
//!    accumulate into each input's gradient (Add if multiple consumers).
//!
//! Returns a `Vec<Option<ExprId>>` indexed by `ExprId.0`: `result[i]` is the
//! gradient of node `i`, or `None` if it has no gradient (e.g. unreachable).

use alloc::vec;
use alloc::vec::Vec;
use pixelflow_ir::arena::{ExprArena, ExprId, ExprNode};
use pixelflow_ir::kind::OpKind;
use pixelflow_ir::pullback::pullback_inputs;

/// Emit backward pass nodes into `arena` for the subgraph rooted at `output_id`.
///
/// Returns `gradients` where `gradients[i]` is the `ExprId` of the gradient
/// of node `ExprId(i)`, or `None` if that node has no gradient.
#[must_use]
pub fn emit_backward(arena: &mut ExprArena, output_id: ExprId) -> Vec<Option<ExprId>> {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_ir::arena::ExprArena;
    use pixelflow_ir::kind::OpKind;

    #[test]
    fn backward_const_output() {
        // y = 2.0 (constant). dy/dx = 0.
        let mut arena = ExprArena::new();
        let y = arena.push_const(2.0);
        let grads = emit_backward(&mut arena, y);
        // Const has no inputs so gradient is just Const(1.0) at the output itself.
        assert!(grads[y.0 as usize].is_some());
    }

    #[test]
    fn backward_linear() {
        // y = 2.0 * x. dy/dx = 2.0.
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let two = arena.push_const(2.0);
        let y = arena.push_binary(OpKind::Mul, two, x);

        let grads = emit_backward(&mut arena, y);
        let dx_id = grads[x.0 as usize].expect("x should have a gradient");

        // Evaluate gradient at x=3: dy/dx = 2.0 regardless of x.
        // For a Mul(Const(2.0), x): d_x = d_out * Const(2.0) = 1.0 * 2.0 = 2.0
        // We verify the node structure: d_x should be Binary(Mul, d_out, two)
        // where d_out = Const(1.0).
        let d_out_id = grads[y.0 as usize].expect("output should have gradient");
        assert_eq!(arena.get(d_out_id), &ExprNode::Const(1.0));
        assert_eq!(
            arena.get(dx_id),
            &ExprNode::Binary(OpKind::Mul, d_out_id, two)
        );
    }

    #[test]
    fn backward_add_shared_input() {
        // y = x + x. dy/dx = 2.0 (gradients accumulate).
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let y = arena.push_binary(OpKind::Add, x, x);

        let grads = emit_backward(&mut arena, y);
        let dx_id = grads[x.0 as usize].expect("x should have a gradient");

        // Both uses of x contribute d_out each. Sum = d_out + d_out = 2 * d_out.
        // The gradient node should be Add(d_out, d_out).
        let d_out_id = grads[y.0 as usize].unwrap();
        assert_eq!(
            arena.get(dx_id),
            &ExprNode::Binary(OpKind::Add, d_out_id, d_out_id)
        );
    }
}
```

- [ ] **Step 2: Add backward.rs to pixelflow-compiler/src/lib.rs**

In `pixelflow-compiler/src/lib.rs`, add:

```rust
pub mod backward;
pub use backward::emit_backward;
```

- [ ] **Step 3: Run tests to confirm failures**

```bash
cargo test -p pixelflow-compiler backward
```

Expected: compile errors or `todo!()` panics.

- [ ] **Step 4: Implement emit_backward**

Replace `todo!()`:

```rust
pub fn emit_backward(arena: &mut ExprArena, output_id: ExprId) -> Vec<Option<ExprId>> {
    let n = arena.len();
    // gradient[i] = ExprId of gradient for node ExprId(i), accumulated via Add.
    let mut gradient: Vec<Option<ExprId>> = vec![None; n];

    // Seed: output gradient = Const(1.0)
    let one = arena.push_const(1.0);
    gradient[output_id.0 as usize] = Some(one);

    // ExprArena is append-only and nodes always reference earlier nodes,
    // so forward order IS topological order. Reverse it for backprop.
    for node_idx in (0..n).rev() {
        let node_id = ExprId(node_idx as u32);
        let d_out = match gradient[node_idx] {
            Some(g) => g,
            None => continue, // unreachable node
        };

        // Collect inputs and op for this node.
        let (op, inputs) = match arena.get(node_id).clone() {
            ExprNode::Var(_) | ExprNode::Const(_) | ExprNode::Param(_) => continue,
            ExprNode::Unary(op, a) => (op, vec![a]),
            ExprNode::Binary(op, a, b) => (op, vec![a, b]),
            ExprNode::Ternary(op, a, b, c) => (op, vec![a, b, c]),
            ExprNode::Nary(_, _, _) => continue, // not needed for NNUE
        };

        // Compute input gradients via pullback.
        let input_grads = pullback_inputs(op, d_out, &inputs, arena);

        // Accumulate into each input's gradient.
        for (input_id, d_input) in inputs.iter().zip(input_grads.iter()) {
            let idx = input_id.0 as usize;
            gradient[idx] = Some(match gradient[idx] {
                None => *d_input,
                Some(existing) => arena.push_binary(OpKind::Add, existing, *d_input),
            });
        }
    }

    gradient
}
```

- [ ] **Step 5: Run tests**

```bash
cargo test -p pixelflow-compiler backward
```

Expected: All backward tests pass.

- [ ] **Step 6: Commit**

```bash
git add pixelflow-compiler/src/backward.rs pixelflow-compiler/src/lib.rs
git commit -m "feat(compiler): emit_backward — reverse-mode AD over ExprArena"
```

---

## Task 5: Cleanup

**Files:** `pixelflow-ir/src/`, `pixelflow-compiler/src/`

- [ ] **Step 1: Ensure all OpKind variants are documented**

In `pixelflow-ir/src/kind.rs`, every variant group should have a `///` comment. Add where missing:

```rust
/// Unified enumeration of all IR operations.
///
/// Each variant has a corresponding pullback rule in `pixelflow_ir::pullback`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum OpKind {
    // --- Variables and Constants ---
    Var = 0,    /// Coordinate variable (0=X, 1=Y, 2=Z, 3=W)
    Const = 1,  /// Literal f32 constant
    // ... (add doc to each group)
```

- [ ] **Step 2: Check for any legacy Arc-based Expr shims**

```bash
grep -r "Arc<.*Expr" pixelflow-ir/src/ pixelflow-compiler/src/
```

If any results: read the file and delete the shim. The `ExprArena`/`ExprId` system is the only IR.

- [ ] **Step 3: Run full workspace test**

```bash
cargo test --workspace
```

Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add pixelflow-ir/src/ pixelflow-compiler/src/
git commit -m "chore(ir): doc comments on OpKind, remove any legacy Expr shims"
```
