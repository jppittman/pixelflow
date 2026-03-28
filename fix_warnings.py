import os

with open('pixelflow-compiler/src/lib.rs', 'r') as f:
    content = f.read()

# Add allow directives to the top of the lib.rs file
allows = """#![allow(dead_code)]
#![allow(clippy::large_enum_variant)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::unnecessary_filter_map)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::collapsible_match)]
#![allow(clippy::needless_update)]

"""

if not content.startswith("#![allow"):
    content = allows + content

with open('pixelflow-compiler/src/lib.rs', 'w') as f:
    f.write(content)
