with open("pixelflow-ir/src/backend/emit/aarch64.rs", "r") as f:
    text = f.read()

text = text.replace("pub fn emit_prologue(_code: &mut Vec<u8>)", "pub fn emit_prologue(code: &mut Vec<u8>)")
text = text.replace("_pool: &mut super::ConstPool,", "pool: &mut super::ConstPool,")

with open("pixelflow-ir/src/backend/emit/aarch64.rs", "w") as f:
    f.write(text)

with open("actor-scheduler/src/spsc.rs", "r") as f:
    text = f.read()

text = text.replace("struct Counted;", "struct Counted(u32);")
with open("actor-scheduler/src/spsc.rs", "w") as f:
    f.write(text)
