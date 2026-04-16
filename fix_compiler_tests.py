import os

filepath = 'pixelflow-compiler/src/optimize.rs'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix error 1: optimize_with_egraph to optimize_via_egraph
content = content.replace("optimize_with_egraph(analyzed, costs)", "optimize_via_egraph_dag(&analyzed.def.body, costs)")

# Let's check how CostModel is created in tests.
