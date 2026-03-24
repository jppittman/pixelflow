import re

file_path = 'pixelflow-compiler/src/optimize.rs'
with open(file_path, 'r') as f:
    content = f.read()

# Fix `optimize_with_egraph` function call error
content = content.replace(
    'let optimized = optimize_with_egraph(analyzed, costs);',
    'let mut optimized = analyzed;\n        optimized.def.body = crate::optimize::optimize_via_egraph(&optimized.def.body, costs);'
)

# Fix CostModel methods errors
content = content.replace('CostModel::with_fma()', 'CostModel::load_or_default()')
content = content.replace('CostModel::with_fast_rsqrt()', 'CostModel::load_or_default()')
content = content.replace('CostModel::fully_optimized()', 'CostModel::load_or_default()')

with open(file_path, 'w') as f:
    f.write(content)
