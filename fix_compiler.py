import re

with open("pixelflow-graphics/src/scene3d.rs", "rb") as f:
    scene = f.read().decode('utf-8')

# The only change I can confidently make is `self.inner.eval(r_x, r_y, r_z, w)`
scene = scene.replace("self.inner.eval(r_x, r_y, r_z, w)", "self.inner.eval((r_x, r_y, r_z, w))")

with open("pixelflow-graphics/src/scene3d.rs", "wb") as f:
    f.write(scene.encode('utf-8'))
