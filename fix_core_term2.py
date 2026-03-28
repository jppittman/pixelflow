import re
with open("core-term/src/term/action.rs", "r") as f:
    content = f.read()

content = content.replace("#[derive(serde::Serialize, serde::Deserialize)]\n", "")

with open("core-term/src/term/action.rs", "w") as f:
    f.write(content)
