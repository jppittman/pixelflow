# Ah, I see! `with_fma`, `with_fast_rsqrt`, `fully_optimized` DO NOT EXIST ANYMORE!
# Or they never existed, and my previous fix on `pixelflow-compiler/src/optimize.rs`
# changed `optimize_with_egraph` to `optimize_via_egraph` because the methods were missing.
# Wait, let's look at `optimize_code_egraph` in `pixelflow-compiler/src/optimize.rs` AGAIN.
# It seems those functions WERE in `cost_builder.rs`?
# I saw `build_cost_model_with_hce` and `parse_cost_model_toml` in `cost_builder.rs`.
# If `with_fma`, `with_fast_rsqrt` and `fully_optimized` are missing, I can just implement them!
# They were probably in `pixelflow-search/src/egraph/cost.rs` before they were removed?
# Let's search if they are in `pixelflow-compiler/src/cost_builder.rs`.
