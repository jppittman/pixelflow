with open('pixelflow-ir/src/backend/emit/mod.rs', 'r') as f:
    text = f.read()

# Fix the use of ExprArena in tests
search = """#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(target_arch = "aarch64")]
    use alloc::boxed::Box;"""
replace = """#[cfg(test)]
mod tests {
    use crate::ExprArena;
    use super::*;
    #[cfg(target_arch = "aarch64")]
    use alloc::boxed::Box;"""
text = text.replace(search, replace)

with open('pixelflow-ir/src/backend/emit/mod.rs', 'w') as f:
    f.write(text)

with open('pixelflow-ir/src/arena.rs', 'r') as f:
    text = f.read()

# Add Default for ExprArena
search = """impl ExprArena {
    /// Creates a new, empty arena.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            nary_children: Vec::new(),
        }
    }"""
replace = """impl Default for ExprArena {
    fn default() -> Self {
        Self::new()
    }
}

impl ExprArena {
    /// Creates a new, empty arena.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            nary_children: Vec::new(),
        }
    }"""
text = text.replace(search, replace)
with open('pixelflow-ir/src/arena.rs', 'w') as f:
    f.write(text)
