import re

with open('pixelflow-ir/src/backend/emit/mod.rs', 'r') as f:
    text = f.read()

# Fix loop indices
text = text.replace("for bi in 0..branch_starts[sched_idx].len() {", "for bi in 0..branch_starts[sched_idx].len() {")
text = text.replace("for ei in 0..branch_ends[sched_idx].len() {", "for ei in 0..branch_ends[sched_idx].len() {")

with open('pixelflow-ir/src/backend/emit/mod.rs', 'w') as f:
    f.write(text)


with open('pixelflow-ir/src/variance.rs', 'r') as f:
    text = f.read()

# Fix match block
search = """            ExprNode::Unary(op, _) => match *op {
                OpKind::Sin
                | OpKind::Cos
                | OpKind::Exp
                | OpKind::Exp2
                | OpKind::Log
                | OpKind::Log2
                | OpKind::Log10
                | OpKind::Sqrt
                | OpKind::InverseSqrt
                | OpKind::Asin
                | OpKind::Acos
                | OpKind::Atan
                | OpKind::Pow
                | OpKind::Tan => 3, // Transcendentals: highest priority
                _ => 1,
            },"""
replace = """            ExprNode::Unary(OpKind::Sin, _)
            | ExprNode::Unary(OpKind::Cos, _)
            | ExprNode::Unary(OpKind::Exp, _)
            | ExprNode::Unary(OpKind::Exp2, _)
            | ExprNode::Unary(OpKind::Log, _)
            | ExprNode::Unary(OpKind::Log2, _)
            | ExprNode::Unary(OpKind::Log10, _)
            | ExprNode::Unary(OpKind::Sqrt, _)
            | ExprNode::Unary(OpKind::InverseSqrt, _)
            | ExprNode::Unary(OpKind::Asin, _)
            | ExprNode::Unary(OpKind::Acos, _)
            | ExprNode::Unary(OpKind::Atan, _)
            | ExprNode::Unary(OpKind::Pow, _)
            | ExprNode::Unary(OpKind::Tan, _) => 3, // Transcendentals: highest priority
            ExprNode::Unary(_, _) => 1,"""
text = text.replace(search, replace)
with open('pixelflow-ir/src/variance.rs', 'w') as f:
    f.write(text)


with open('pixelflow-ir/src/backend/mod.rs', 'r') as f:
    text = f.read()

# Fix Excessive Precision & Consts
text = text.replace("const LN_2: f32 = 0.6931471805599453;", "const LN_2: f32 = std::f32::consts::LN_2;")
text = text.replace("const LOG10_2: f32 = 0.30102999566398120;", "const LOG10_2: f32 = std::f32::consts::LOG10_2;")

with open('pixelflow-ir/src/backend/mod.rs', 'w') as f:
    f.write(text)


with open('pixelflow-ir/src/backend/x86.rs', 'r') as f:
    text = f.read()

text = text.replace("let c1 = _mm_set1_ps(1.6719970703125);", "let c1 = _mm_set1_ps(1.671_997_1);")
text = text.replace("let c3 = _mm_set1_ps(-0.645963541666667);", "let c3 = _mm_set1_ps(-0.645_963_55);")
text = text.replace("let c5 = _mm_set1_ps(0.079689450);", "let c5 = _mm_set1_ps(0.079_689_45);")
text = text.replace("let c7 = _mm_set1_ps(-0.0046817541);", "let c7 = _mm_set1_ps(-0.004_681_754);")

text = text.replace("let c3 = _mm_set1_ps(-0.333333333);", "let c3 = _mm_set1_ps(-0.333_333_34);")
text = text.replace("let c7 = _mm_set1_ps(-0.142857143);", "let c7 = _mm_set1_ps(-0.142_857_15);")
with open('pixelflow-ir/src/backend/x86.rs', 'w') as f:
    f.write(text)
