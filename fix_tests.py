import re

with open('pixelflow-compiler/src/optimize.rs', 'r') as f:
    content = f.read()

content = content.replace('CostModel::with_fma()', 'CostModel::new()')
content = content.replace('CostModel::with_fast_rsqrt()', 'CostModel::new()')
content = content.replace('CostModel::fully_optimized()', 'CostModel::new()')

with open('pixelflow-compiler/src/optimize.rs', 'w') as f:
    f.write(content)
