import os

filepath = 'pixelflow-compiler/src/optimize.rs'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace CostModel::with_fma() -> CostModel::new()
content = content.replace("CostModel::with_fma()", "CostModel::new()")

# Replace CostModel::with_fast_rsqrt() -> CostModel::new()
content = content.replace("CostModel::with_fast_rsqrt()", "CostModel::new()")

# Replace CostModel::fully_optimized() -> CostModel::new()
content = content.replace("CostModel::fully_optimized()", "CostModel::new()")

# Replace optimize_with_egraph(analyzed, costs)
content = content.replace("let optimized = optimize_with_egraph(analyzed, costs);", """let optimized_body = optimize_via_egraph_dag(&analyzed.def.body, costs);
        let mut optimized = analyzed;
        optimized.def.body = optimized_body;""")

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
