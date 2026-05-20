import os

with open('pixelflow-compiler/src/optimize.rs', 'r') as f:
    content = f.read()

search = """            if matches!(call.receiver.as_ref(), Expr::Verbatim(_)) {
                if call.args.iter().any(|arg| expr_references_any(arg, local_names)) {
                    return true;
                }
            }"""
replace = """            if matches!(call.receiver.as_ref(), Expr::Verbatim(_))
                && call.args.iter().any(|arg| expr_references_any(arg, local_names)) {
                    return true;
            }"""
content = content.replace(search, replace)

search2 = """            if is_zero(rhs_val) {
                return Some(*binary.lhs.clone());
            }"""
replace2 = """            if is_zero(rhs_val) {
                return Some(*binary.lhs.clone());
            }"""
# just format this differently maybe? No, let's look for the actual exact content
content = content.replace(
    "        BinaryOp::Sub => {\n            // x - 0 = x\n            if is_zero(rhs_val) {\n                return Some(*binary.lhs.clone());\n            }\n        }",
    "        BinaryOp::Sub if is_zero(rhs_val) => return Some(*binary.lhs.clone()),\n        BinaryOp::Sub => {}"
)

with open('pixelflow-compiler/src/optimize.rs', 'w') as f:
    f.write(content)
