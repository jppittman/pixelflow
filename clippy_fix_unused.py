with open("pixelflow-ir/src/backend/compounds.rs", "r") as f:
    text = f.read()

text = text.replace("let mantissa_bits = Self::from_bits(0x3F800000); // 1.0", "let _mantissa_bits = Self::from_bits(0x3F800000); // 1.0")
text = text.replace("let mantissa_mask = Self::from_bits(0x007FFFFF);", "let _mantissa_mask = Self::from_bits(0x007FFFFF);")

with open("pixelflow-ir/src/backend/compounds.rs", "w") as f:
    f.write(text)

with open("pixelflow-ir/src/backend/emit/aarch64.rs", "r") as f:
    text = f.read()

text = text.replace("pub fn emit_prologue(code: &mut Vec<u8>)", "pub fn emit_prologue(_code: &mut Vec<u8>)")
text = text.replace("pool: &mut super::ConstPool,", "_pool: &mut super::ConstPool,")

with open("pixelflow-ir/src/backend/emit/aarch64.rs", "w") as f:
    f.write(text)

with open("actor-scheduler/src/spsc.rs", "r") as f:
    text = f.read()

text = text.replace("struct Counted(u32);", "struct Counted;")
with open("actor-scheduler/src/spsc.rs", "w") as f:
    f.write(text)
