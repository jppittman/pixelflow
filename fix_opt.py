import re
with open("pixelflow-compiler/src/optimize.rs", "r") as f:
    content = f.read()

content = content.replace("analyzed.def.body = Box::new(optimize_via_egraph(&analyzed.def.body, costs));", "analyzed.def.body = Box::new(optimize_via_egraph(&analyzed.def.body, costs));")
content = content.replace("analyzed.def.body = Box::new(optimize_via_egraph(&analyzed.def.body, costs));", "analyzed.def.body = Box::new(optimize_via_egraph(&analyzed.def.body, costs));")
content = content.replace("Box::new(optimize_via_egraph(&analyzed.def.body, costs))", "optimize_via_egraph(&analyzed.def.body, costs)")

with open("pixelflow-compiler/src/optimize.rs", "w") as f:
    f.write(content)
