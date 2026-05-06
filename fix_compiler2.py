import re

with open("pixelflow-compiler/src/optimize.rs", "r", encoding="utf-8") as f:
    opt = f.read()

# Fix optimize_with_egraph typo
opt = opt.replace("let optimized = optimize_via_egraph(&analyzed.def.body, costs);", "let optimized = optimize_via_egraph(analyzed, costs);")
opt = opt.replace("CostModel::with_fma()", "CostModel::new()")
opt = opt.replace("CostModel::with_fast_rsqrt()", "CostModel::new()")
opt = opt.replace("CostModel::fully_optimized()", "CostModel::new()")
with open("pixelflow-compiler/src/optimize.rs", "w", encoding="utf-8") as f:
    f.write(opt)
