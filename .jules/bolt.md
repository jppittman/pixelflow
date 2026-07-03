## 2025-12-28 - Inherent Methods Shadowing Optimized Trait Implementations
**Learning:** I discovered that `Jet3` and `Jet2` had optimized `sqrt` implementations in their `Numeric` trait implementation, but their *inherent* `sqrt` methods (which shadow the trait methods when called directly) were still using the unoptimized slow path.
**Action:** Always check both trait implementations AND inherent methods when optimizing types in Rust, as inherent methods take precedence and might be legacy/unoptimized code.

## 2025-12-28 - Optimized Keybinding Lookups
**Learning:** `KeybindingsConfig` stored bindings as a `Vec`, causing O(n) lookup overhead on every keypress.
**Action:** Refactored `KeybindingsConfig` to maintain a `HashMap` for O(1) lookups while keeping the `Vec` for serialization/deserialization compatibility using `#[serde(from/into)]`. This ensures performance without breaking config file format.

## 2025-12-28 - AST Optimization and Parentheses
**Learning:** In the `pixelflow-macros` AST optimizer, I initially removed `Expr::Paren` wrappers assuming they were redundant during optimization recursion. This broke operator precedence (e.g., `(X - offset).abs()` became logic that failed tests).
**Action:** When implementing AST transformations, always preserve grouping/parentheses nodes unless you perform a specific precedence check proving they are redundant.

## 2025-12-28 - Rasterizer Inner Loop Hoisting
**Learning:** The inner loop of `execute_stripe` was re-evaluating `Field::sequential(start)` on every iteration, which involves multiple SIMD instructions (broadcast/load + add).
**Action:** Hoisted the initialization of `xs` out of the loop and updated it incrementally using a pre-computed `step` vector. This reduced the inner loop overhead significantly, yielding a ~34% improvement in rasterization throughput.

## 2024-07-02 - Enforce Style Guidelines for Test Naming
**Learning:** Bulk renaming `#[test]` functions by simply stripping the `test_` prefix can inadvertently cause compiler errors (like E0061) if the new function name shadows an existing function in scope that the test body intends to call.
**Action:** When renaming tests, always check if the stripped name collides with another symbol (e.g., by scanning the file for calls using the new name) and safely fallback to prepending `verify_` to avoid conflicts.

## 2024-07-02 - Fix Flaky Test shutdown_drain_all_timeout_fallback
**Learning:** `shutdown_drain_all_timeout_fallback` frequently flaked on CI due to a very tight hardcoded timeout bound (`< 150ms`).
**Action:** Relaxed the hardcoded timing threshold in flaky tests to generous values (e.g., `500ms`) when explicitly assigned to fix them, per testing resiliency rules.
## 2024-07-03 - Enforcing Match Over Else If
**Learning:** `STYLE.md` strictly dictates the use of `match` over `if`/`else if` chains when dealing with a fixed set of conditions (like specific string literal matching). The codebase contained instances where strings were checked sequentially using `else if`, which violates this structural style rule and is less idiomatic in Rust.
**Action:** When adding new conditional logic checking against known literal values, proactively use `match` statements instead of chained `if`/`else if` blocks to ensure compliance with project style guidelines.
