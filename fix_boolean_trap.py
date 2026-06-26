import re

path1 = 'pixelflow-ir/src/backend/emit/aarch64.rs'
with open(path1, 'r') as f:
    text1 = f.read()

# Replace bool with AdrKind enum
enum_def = """
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdrKind {
    Adr,
    Adrp,
}
"""

text1 = text1.replace(
    'pub fn patch_adr_or_adrp(code: &mut [u8], adr_pos: usize, target_pos: usize, is_adrp: bool) {',
    enum_def + '\npub fn patch_adr_or_adrp(code: &mut [u8], adr_pos: usize, target_pos: usize, kind: AdrKind) {'
)

text1 = text1.replace(
    'if is_adrp {',
    'if kind == AdrKind::Adrp {'
)

text1 = text1.replace(
    'pub fn patch_adr_or_adrp(&mut self, adr_pos: usize, target_pos: usize, is_adrp: bool) {',
    'pub fn patch_adr_or_adrp(&mut self, adr_pos: usize, target_pos: usize, kind: AdrKind) {'
)

text1 = text1.replace(
    'patch_adr_or_adrp(&mut self.code, adr_pos, target_pos, is_adrp);',
    'patch_adr_or_adrp(&mut self.code, adr_pos, target_pos, kind);'
)

with open(path1, 'w') as f:
    f.write(text1)


path2 = 'pixelflow-ir/src/backend/emit/mod.rs'
with open(path2, 'r') as f:
    text2 = f.read()

text2 = text2.replace(
    'aarch64::patch_adr_or_adrp(&mut code, adr_pos, pool_start, needs_adrp);',
    'aarch64::patch_adr_or_adrp(&mut code, adr_pos, pool_start, if needs_adrp { aarch64::AdrKind::Adrp } else { aarch64::AdrKind::Adr });'
)

text2 = text2.replace(
    'aarch64::patch_adr_or_adrp(code, adr_pos, pool_start, needs_adrp);',
    'aarch64::patch_adr_or_adrp(code, adr_pos, pool_start, if needs_adrp { aarch64::AdrKind::Adrp } else { aarch64::AdrKind::Adr });'
)

with open(path2, 'w') as f:
    f.write(text2)
