import re

file = 'pixelflow-compiler/src/optimize.rs'
with open(file, 'r') as f: content = f.read()

# I used `let mut map = std::collections::HashMap::new(); ... CostModel::from_map(&map)` but it seems it didn't pass? Let's verify why test failed.
# `assertion failed: debug.contains("mul_add")`. Why? Because `CostModel::from_map(&map)` creates a cost model but with default costs for others?
# Wait, `CostModel::from_map(&map)` might not actually use a shallow/default model for the rest!
# Let's inspect `CostModel` in `pixelflow-search`.
