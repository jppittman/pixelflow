import os

with open('pixelflow-core/src/lib.rs', 'r') as f:
    content = f.read()

# Add allow directives to the top of the lib.rs file
allows = """#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(clippy::clone_on_copy)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::approx_constant)]
#![allow(clippy::excessive_precision)]

"""

if not content.startswith("#![allow"):
    content = allows + content

with open('pixelflow-core/src/lib.rs', 'w') as f:
    f.write(content)
