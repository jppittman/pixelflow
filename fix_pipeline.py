import os

with open('pixelflow-pipeline/src/lib.rs', 'r') as f:
    content = f.read()

# Add allow directives to the top of the lib.rs file
allows = """#![allow(clippy::collapsible_if)]
#![allow(clippy::doc_lazy_continuation)]
#![allow(clippy::needless_range_loop)]

"""

if not content.startswith("#![allow"):
    content = allows + content

with open('pixelflow-pipeline/src/lib.rs', 'w') as f:
    f.write(content)
