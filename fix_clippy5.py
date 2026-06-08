import sys

# Replace constants with built-ins to fix excessive precision clippy errors

with open('pixelflow-ir/src/backend/emit/mod.rs', 'r') as f:
    text = f.read()

# Fix too many args by creating a struct for the context
# pub fn emit_resolve(code: &mut Vec<u8>, vid: regalloc::ValueId, target: Reg, spill_offsets: &std::collections::HashMap<regalloc::ValueId, u16>, schedule: &[(regalloc::ValueId, ScheduledOp)], pool: &ConstPool) -> Reg {
# This is a bit too invasive, let's just allow clippy::too_many_arguments

if "#[allow(clippy::too_many_arguments)]\nfn emit_resolve" not in text:
    text = text.replace("fn emit_resolve(\n", "#[allow(clippy::too_many_arguments)]\nfn emit_resolve(\n")
    with open('pixelflow-ir/src/backend/emit/mod.rs', 'w') as f:
        f.write(text)
    print("Updated emit/mod.rs")
