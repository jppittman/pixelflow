import re

with open('./pixelflow-ir/src/backend/emit/aarch64.rs', 'r') as f:
    content = f.read()

content = content.replace("/// If `is_adrp` is true, assumes 8 bytes are reserved and patches `ADRP X17` + `ADD X17`.", "/// If `mode` is `AdrMode::Adrp`, assumes 8 bytes are reserved and patches `ADRP X17` + `ADD X17`.")

with open('./pixelflow-ir/src/backend/emit/aarch64.rs', 'w') as f:
    f.write(content)
