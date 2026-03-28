import os

with open('pixelflow-graphics/src/lib.rs', 'r') as f:
    content = f.read()

# Add allow directives to the top of the lib.rs file
allows = """#![allow(clippy::too_many_arguments)]
#![allow(clippy::approx_constant)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::type_complexity)]

"""

if not content.startswith("#![allow(clippy::too_many_arguments)]"):
    content = allows + content

with open('pixelflow-graphics/src/lib.rs', 'w') as f:
    f.write(content)
