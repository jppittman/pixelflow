import re
with open("core-term/src/term/action.rs", "r") as f:
    content = f.read()

# Fix unmatched angle bracket
content = content.replace("text: Option<std::borrow::Cow<'static, str>>>,", "text: Option<std::borrow::Cow<'static, str>>,")

# Add Serialize, Deserialize to UserInputAction
content = content.replace("pub enum UserInputAction {", "#[derive(serde::Serialize, serde::Deserialize)]\npub enum UserInputAction {")

with open("core-term/src/term/action.rs", "w") as f:
    f.write(content)
