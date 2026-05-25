import re

with open('pixelflow-compiler/src/optimize.rs', 'r') as f:
    content = f.read()

content = content.replace('let optimized = optimize_via_egraph(analyzed, costs);', 'let optimized = optimize_via_egraph(&analyzed.def.body, costs);')
content = content.replace('format!("{:?}", optimized.def.body)', 'format!("{:?}", optimized)')

with open('pixelflow-compiler/src/optimize.rs', 'w') as f:
    f.write(content)
