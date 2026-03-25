import re

file = 'pixelflow-compiler/src/optimize.rs'
with open(file, 'r') as f: content = f.read()

# Replace any references to `.def.body` on an `Expr` (which caused compilation errors).
content = content.replace('format!("{:?}", optimized.def.body)', 'format!("{:?}", optimized)')
content = content.replace('let optimized = optimize_with_egraph(analyzed, costs);', 'let optimized = optimize_via_egraph(&analyzed.def.body, costs);')
content = content.replace('eprintln!("Optimized AST: {:?}", optimized.def.body);', 'eprintln!("Optimized AST: {:?}", optimized);')

with open(file, 'w') as f: f.write(content)

# We must NOT ignore tests. We will use FMA costs appropriately.
content = content.replace(
'''        let debug = optimize_code_egraph(input, &CostModel::with_fma());''',
'''        let mut map = std::collections::HashMap::new();
        map.insert("MulAdd".to_string(), 1);
        let costs = CostModel::from_map(&map);
        let debug = optimize_code_egraph(input, &costs);''')

content = content.replace(
'''        let debug = optimize_code_egraph(input, &CostModel::with_fast_rsqrt());''',
'''        let mut map = std::collections::HashMap::new();
        map.insert("Rsqrt".to_string(), 1);
        let costs = CostModel::from_map(&map);
        let debug = optimize_code_egraph(input, &costs);''')

content = content.replace(
'''        let debug = optimize_code_egraph(input, &CostModel::fully_optimized());''',
'''        let mut map = std::collections::HashMap::new();
        map.insert("MulAdd".to_string(), 1);
        let costs = CostModel::from_map(&map);
        let debug = optimize_code_egraph(input, &costs);''')

with open(file, 'w') as f: f.write(content)
