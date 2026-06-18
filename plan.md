1. **Refactor `SemanticAnalyzer::new`**
    - The `is_anonymous: bool` parameter violates the rule against using boolean arguments.
    - Create an enum `KernelType` with variants `Anonymous` and `Named`.
    - Modify `SemanticAnalyzer::new` to accept `KernelType` instead of `bool`.
    - Update `pixelflow-compiler/src/sema.rs` where `SemanticAnalyzer::new` is called.

2. **Complete Pre-Commit Steps**
    - Run `cargo fmt`.
    - Run `cargo clippy --all-targets --all-features -- -D warnings`.
    - Run `cargo test -p pixelflow-compiler`.
    - Fix any regressions.
