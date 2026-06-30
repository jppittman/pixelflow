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
## 2025-12-28 - HashSet Instantiation in Hot Loops
**Learning:** `kqueue.rs` was instantiating a `HashSet` on every event loop iteration to track seen tokens for deduplication. This causes unnecessary heap allocations for small `nev` counts, especially since a linear search over `events_out` is sufficient and allocation-free.
**Action:** In hot event loops or loops with small bounded bounds, replace `HashSet` deduplication with linear search over a pre-existing dynamically growing `Vec` using `.iter_mut().find()` to eliminate memory allocation overhead.
