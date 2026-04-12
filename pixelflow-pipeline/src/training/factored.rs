//! # Expression parsing and serialization utilities for NNUE training.
//!
//! Provides parsers for two expression syntaxes:
//! - **S-expression**: `Add(Mul(Var(0), Var(1)), Var(2))` (JSONL training data format)
//! - **Kernel code**: `(X * Y) + Z` (human-readable, round-trips with `parse_kernel_code`/`expr_to_kernel_code`)

use crate::nnue::{Expr, OpKind};
use pixelflow_ir::EmitStyle;

// ============================================================================
// Expression Parsing (for loading training data)
// ============================================================================

/// Parse an expression from a string representation.
///
/// Format: `OpName(child1, child2, ...)` or `Var(n)` or `Const(value)`
pub fn parse_expr(s: &str) -> Option<Expr> {
    let s = s.trim();

    // Try parsing as Var
    if s.starts_with("Var(") && s.ends_with(')') {
        let inner = &s[4..s.len() - 1];
        let idx: u8 = inner.parse().ok()?;
        return Some(Expr::Var(idx));
    }

    // Try parsing as Const
    if s.starts_with("Const(") && s.ends_with(')') {
        let inner = &s[6..s.len() - 1];
        let val: f32 = inner.parse().ok()?;
        return Some(Expr::Const(val));
    }

    // Parse as operation
    let paren_pos = s.find('(')?;
    let op_name = &s[..paren_pos];
    let op = parse_op_kind(op_name)?;

    // Find matching closing paren
    let inner = &s[paren_pos + 1..s.len() - 1];
    let children = split_args(inner);

    match op.arity() {
        0 => None, // Should have been caught above
        1 => {
            if children.len() != 1 {
                return None;
            }
            let a = parse_expr(children[0])?;
            Some(Expr::Unary(op, Box::new(a)))
        }
        2 => {
            if children.len() != 2 {
                return None;
            }
            let a = parse_expr(children[0])?;
            let b = parse_expr(children[1])?;
            Some(Expr::Binary(op, Box::new(a), Box::new(b)))
        }
        3 => {
            if children.len() != 3 {
                return None;
            }
            let a = parse_expr(children[0])?;
            let b = parse_expr(children[1])?;
            let c = parse_expr(children[2])?;
            Some(Expr::Ternary(op, Box::new(a), Box::new(b), Box::new(c)))
        }
        _ => None,
    }
}

/// Parse operation name to OpKind.
fn parse_op_kind(name: &str) -> Option<OpKind> {
    match name.to_lowercase().as_str() {
        "add" => Some(OpKind::Add),
        "sub" => Some(OpKind::Sub),
        "mul" => Some(OpKind::Mul),
        "div" => Some(OpKind::Div),
        "neg" => Some(OpKind::Neg),
        "sqrt" => Some(OpKind::Sqrt),
        "rsqrt" => Some(OpKind::Rsqrt),
        "abs" => Some(OpKind::Abs),
        "min" => Some(OpKind::Min),
        "max" => Some(OpKind::Max),
        "muladd" | "mul_add" | "fma" => Some(OpKind::MulAdd),
        "recip" => Some(OpKind::Recip),
        "floor" => Some(OpKind::Floor),
        "ceil" => Some(OpKind::Ceil),
        "round" => Some(OpKind::Round),
        "fract" => Some(OpKind::Fract),
        "sin" => Some(OpKind::Sin),
        "cos" => Some(OpKind::Cos),
        "tan" => Some(OpKind::Tan),
        "asin" => Some(OpKind::Asin),
        "acos" => Some(OpKind::Acos),
        "atan" => Some(OpKind::Atan),
        "atan2" => Some(OpKind::Atan2),
        "exp" => Some(OpKind::Exp),
        "exp2" => Some(OpKind::Exp2),
        "ln" => Some(OpKind::Ln),
        "log2" => Some(OpKind::Log2),
        "log10" => Some(OpKind::Log10),
        "pow" => Some(OpKind::Pow),
        "hypot" => Some(OpKind::Hypot),
        _ => None,
    }
}

/// Split comma-separated arguments, respecting nested parentheses.
fn split_args(s: &str) -> Vec<&str> {
    let mut args = Vec::new();
    let mut depth = 0;
    let mut start = 0;

    for (i, c) in s.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth == 0 => {
                args.push(s[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }

    if start < s.len() {
        args.push(s[start..].trim());
    }

    args
}

// ============================================================================
// Kernel Code Parser (Functional Recursive Descent)
// ============================================================================
//
// Grammar:
//   expr     ::= additive
//   additive ::= multiplicative (('+' | '-') multiplicative)*
//   mult     ::= postfix (('*' | '/') postfix)*
//   postfix  ::= primary ('.' method)*
//   method   ::= IDENT '(' expr? ')'
//   primary  ::= '(' expr ')' | '-' postfix | VAR | NUM
//   VAR      ::= 'X' | 'Y' | 'Z' | 'W'
//   NUM      ::= float literal

/// Parser result: (parsed value, remaining input)
type ParseResult<'a, T> = Option<(T, &'a str)>;

/// Parse kernel code syntax like "(X + Y)" into Expr.
pub fn parse_kernel_code(s: &str) -> Option<Expr> {
    kc_expr(s.trim()).and_then(|(expr, rest)| rest.is_empty().then_some(expr))
}

/// Top-level: parse a complete expression
fn kc_expr(input: &str) -> ParseResult<Expr> {
    parse_additive(input.trim())
}

/// Parse additive: left-associative chain of +/-
fn parse_additive(input: &str) -> ParseResult<Expr> {
    let (mut acc, mut rest) = parse_multiplicative(input)?;

    while let Some((op, remaining)) = parse_additive_op(rest.trim_start()) {
        let (rhs, remaining) = parse_multiplicative(remaining.trim_start())?;
        acc = Expr::Binary(op, Box::new(acc), Box::new(rhs));
        rest = remaining;
    }

    Some((acc, rest))
}

fn parse_additive_op(input: &str) -> ParseResult<OpKind> {
    match input.chars().next()? {
        '+' => Some((OpKind::Add, &input[1..])),
        '-' => Some((OpKind::Sub, &input[1..])),
        _ => None,
    }
}

/// Parse multiplicative: left-associative chain of * /
fn parse_multiplicative(input: &str) -> ParseResult<Expr> {
    let (mut acc, mut rest) = parse_postfix(input)?;

    while let Some((op, remaining)) = parse_multiplicative_op(rest.trim_start()) {
        let (rhs, remaining) = parse_postfix(remaining.trim_start())?;
        acc = Expr::Binary(op, Box::new(acc), Box::new(rhs));
        rest = remaining;
    }

    Some((acc, rest))
}

fn parse_multiplicative_op(input: &str) -> ParseResult<OpKind> {
    match input.chars().next()? {
        '*' => Some((OpKind::Mul, &input[1..])),
        '/' => Some((OpKind::Div, &input[1..])),
        _ => None,
    }
}

/// Parse postfix: primary followed by method chains
fn parse_postfix(input: &str) -> ParseResult<Expr> {
    let (mut acc, mut rest) = parse_primary(input)?;

    while let Some((expr, remaining)) = parse_method_call(rest.trim_start(), acc.clone()) {
        acc = expr;
        rest = remaining;
    }

    Some((acc, rest))
}

/// Parse a method call: .method() or .method(arg) or .method(arg1, arg2)
///
/// Dispatches through `OpKind::from_name()` + `arity()` — no hand-maintained
/// op enumeration. Adding a new OpKind automatically makes it parseable here.
fn parse_method_call<'a>(input: &'a str, base: Expr) -> ParseResult<'a, Expr> {
    let input = input.strip_prefix('.')?;
    let (method_name, rest) = parse_ident(input)?;
    let rest = rest.strip_prefix('(')?;

    // Unary method: .method()
    if let Some(rest) = rest.trim_start().strip_prefix(')') {
        let op = OpKind::from_name(method_name)?;
        if op.arity() != 1 {
            return None;
        }
        return Some((Expr::Unary(op, Box::new(base)), rest));
    }

    // Parse first argument
    let (arg1, rest) = kc_expr(rest.trim_start())?;
    let rest = rest.trim_start();

    // Ternary method: .method(arg1, arg2)
    if let Some(rest) = rest.strip_prefix(',') {
        let (arg2, rest) = kc_expr(rest.trim_start())?;
        let rest = rest.trim_start().strip_prefix(')')?;
        let op = OpKind::from_name(method_name)?;
        if op.arity() != 3 {
            return None;
        }
        return Some((
            Expr::Ternary(op, Box::new(base), Box::new(arg1), Box::new(arg2)),
            rest,
        ));
    }

    // Binary method: .method(arg)
    let rest = rest.strip_prefix(')')?;
    let op = OpKind::from_name(method_name)?;
    if op.arity() != 2 {
        return None;
    }
    Some((Expr::Binary(op, Box::new(base), Box::new(arg1)), rest))
}

/// Parse primary: parens, negation, variable, or number
fn parse_primary(input: &str) -> ParseResult<Expr> {
    let input = input.trim_start();

    // Parenthesized expression
    if let Some(rest) = input.strip_prefix('(') {
        let (expr, rest) = kc_expr(rest)?;
        let rest = rest.trim_start().strip_prefix(')')?;
        return Some((expr, rest));
    }

    // Unary negation
    if let Some(rest) = input.strip_prefix('-') {
        let (expr, rest) = parse_postfix(rest.trim_start())?;
        return Some((Expr::Unary(OpKind::Neg, Box::new(expr)), rest));
    }

    // Variable or number
    parse_variable(input).or_else(|| parse_number(input))
}

/// Parse a variable: X, Y, Z, W
fn parse_variable(input: &str) -> ParseResult<Expr> {
    let (c, rest) = input.split_at(1.min(input.len()));
    match c {
        "X" => Some((Expr::Var(0), rest)),
        "Y" => Some((Expr::Var(1), rest)),
        "Z" => Some((Expr::Var(2), rest)),
        "W" => Some((Expr::Var(3), rest)),
        _ => None,
    }
}

/// Parse a numeric literal
fn parse_number(input: &str) -> ParseResult<Expr> {
    let end = input
        .char_indices()
        .find(|(_, c)| !matches!(c, '0'..='9' | '.' | '-' | 'e' | 'E' | '+'))
        .map(|(i, _)| i)
        .unwrap_or(input.len());

    if end == 0 {
        return None;
    }

    let num_str = &input[..end];
    let val: f32 = num_str.parse().ok()?;
    Some((Expr::Const(val), &input[end..]))
}

/// Parse an identifier (method name). Accepts `[a-zA-Z_][a-zA-Z0-9_]*`
/// to handle op names with digits like `atan2`, `exp2`, `log2`, `log10`.
fn parse_ident(input: &str) -> ParseResult<&str> {
    let mut chars = input.char_indices();
    // First character must be alphabetic or underscore
    match chars.next() {
        Some((_, c)) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return None,
    }
    // Remaining characters can also include digits
    let end = chars
        .find(|(_, c)| !c.is_ascii_alphanumeric() && *c != '_')
        .map(|(i, _)| i)
        .unwrap_or(input.len());

    Some((&input[..end], &input[end..]))
}

// ============================================================================
// Expr → Kernel Code Serialization
// ============================================================================

/// Convert an `Expr` to kernel code syntax (inverse of [`parse_kernel_code`]).
///
/// Dispatches formatting through [`OpKind::emit_style()`] — no op enumeration.
/// The output string, when passed to `parse_kernel_code()`, yields a semantically
/// equivalent expression.
///
/// # Panics
///
/// Panics on `Expr::Param` (must be substituted first) or `Expr::Nary`.
pub fn expr_to_kernel_code(expr: &Expr) -> String {
    match expr {
        Expr::Var(0) => "X".into(),
        Expr::Var(1) => "Y".into(),
        Expr::Var(2) => "Z".into(),
        Expr::Var(3) => "W".into(),
        Expr::Var(i) => panic!(
            "expr_to_kernel_code: variable index {} exceeds X/Y/Z/W range", i
        ),
        Expr::Const(v) => format_const_kc(*v),
        Expr::Param(i) => panic!(
            "Expr::Param({}) reached expr_to_kernel_code — call substitute_params first", i
        ),
        Expr::Unary(op, a) => {
            let a_s = expr_to_kernel_code(a);
            emit_op_kc(*op, &[a_s])
        }
        Expr::Binary(op, a, b) => {
            let a_s = expr_to_kernel_code(a);
            let b_s = expr_to_kernel_code(b);
            emit_op_kc(*op, &[a_s, b_s])
        }
        Expr::Ternary(op, a, b, c) => {
            let a_s = expr_to_kernel_code(a);
            let b_s = expr_to_kernel_code(b);
            let c_s = expr_to_kernel_code(c);
            emit_op_kc(*op, &[a_s, b_s, c_s])
        }
        Expr::Nary(op, _) => panic!(
            "expr_to_kernel_code: Nary({}) not representable in kernel code syntax", op.name()
        ),
    }
}

/// Emit an operation in kernel code syntax, dispatching through `emit_style()`.
fn emit_op_kc(op: OpKind, args: &[String]) -> String {
    match (op.emit_style(), args) {
        (EmitStyle::UnaryPrefix, [a]) => format!("(-{})", a),
        (EmitStyle::UnaryMethod, [a]) => format!("({}).{}()", a, op.name()),
        (EmitStyle::BinaryInfix(sym), [a, b]) => format!("({} {} {})", a, sym, b),
        (EmitStyle::BinaryMethod, [a, b]) => format!("({}).{}({})", a, op.name(), b),
        (EmitStyle::BinaryMethodNamed(method), [a, b]) => format!("({}).{}({})", a, method, b),
        (EmitStyle::TernaryMethod, [a, b, c]) => {
            format!("({}).{}({}, {})", a, op.name(), b, c)
        }
        (EmitStyle::Special, _) => panic!(
            "emit_op_kc: Special ops (Var/Const/Tuple) must be handled by caller, got {}",
            op.name()
        ),
        (style, args) => panic!(
            "emit_op_kc: arity mismatch for {}: {:?} expects different arg count, got {}",
            op.name(), style, args.len()
        ),
    }
}

/// Format a constant for kernel code syntax.
fn format_const_kc(v: f32) -> String {
    if !v.is_finite() {
        panic!("expr_to_kernel_code: non-finite constant {} cannot be represented", v);
    }
    if v.is_sign_negative() && v != 0.0 {
        return format!("(-{})", format_const_kc(-v));
    }
    // Rust's Display for f32 produces shortest round-trip representation
    format!("{}", v)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_expr_should_succeed_when_called() {
        let expr = parse_expr("Add(Var(0), Var(1))").expect("Expected value but got None/Err");
        assert!(matches!(expr, Expr::Binary(OpKind::Add, _, _)));

        let expr = parse_expr("Mul(Add(Var(0), Var(1)), Var(2))").expect("Expected value but got None/Err");
        assert!(matches!(expr, Expr::Binary(OpKind::Mul, _, _)));

        let expr = parse_expr("MulAdd(Var(0), Var(1), Var(2))").expect("Expected value but got None/Err");
        assert!(matches!(expr, Expr::Ternary(OpKind::MulAdd, _, _, _)));
    }

    // ========================================================================
    // Kernel Code Parser Tests
    // ========================================================================

    #[test]
    fn parse_kernel_code_variables_should_succeed_when_called() {
        assert!(matches!(parse_kernel_code("X"), Some(Expr::Var(0))));
        assert!(matches!(parse_kernel_code("Y"), Some(Expr::Var(1))));
        assert!(matches!(parse_kernel_code("Z"), Some(Expr::Var(2))));
        assert!(matches!(parse_kernel_code("W"), Some(Expr::Var(3))));
    }

    #[test]
    fn parse_kernel_code_constants_should_succeed_when_called() {
        assert!(matches!(parse_kernel_code("1.0"), Some(Expr::Const(v)) if (v - 1.0).abs() < 1e-6));
        assert!(matches!(parse_kernel_code("(4.595877)"), Some(Expr::Const(v)) if (v - 4.595877).abs() < 1e-5));
        assert!(matches!(parse_kernel_code("0.0"), Some(Expr::Const(v)) if v.abs() < 1e-6));
    }

    #[test]
    fn parse_kernel_code_binary_ops_should_succeed_when_called() {
        let expr = parse_kernel_code("(X + Y)").expect("Expected value but got None/Err");
        assert!(matches!(expr, Expr::Binary(OpKind::Add, _, _)));

        let expr = parse_kernel_code("(X - Y)").expect("Expected value but got None/Err");
        assert!(matches!(expr, Expr::Binary(OpKind::Sub, _, _)));

        let expr = parse_kernel_code("(X * Y)").expect("Expected value but got None/Err");
        assert!(matches!(expr, Expr::Binary(OpKind::Mul, _, _)));

        let expr = parse_kernel_code("(X / Y)").expect("Expected value but got None/Err");
        assert!(matches!(expr, Expr::Binary(OpKind::Div, _, _)));
    }

    #[test]
    fn parse_kernel_code_from_benchmark_cache_should_succeed_when_called() {
        let expr = parse_kernel_code("((4.595877) - Z)").expect("Expected value but got None/Err");
        assert!(matches!(expr, Expr::Binary(OpKind::Sub, _, _)));

        let expr = parse_kernel_code("((4.595877) + (-Z))").expect("Expected value but got None/Err");
        assert!(matches!(expr, Expr::Binary(OpKind::Add, _, _)));

        let expr = parse_kernel_code("((-Z) + (4.595877))").expect("Expected value but got None/Err");
        assert!(matches!(expr, Expr::Binary(OpKind::Add, _, _)));
    }

    #[test]
    fn parse_kernel_code_unary_ops_should_succeed_when_called() {
        let expr = parse_kernel_code("(-X)").expect("Expected value but got None/Err");
        assert!(matches!(expr, Expr::Unary(OpKind::Neg, _)));

        let expr = parse_kernel_code("(X).sqrt()").expect("Expected value but got None/Err");
        assert!(matches!(expr, Expr::Unary(OpKind::Sqrt, _)));

        let expr = parse_kernel_code("(X).abs()").expect("Expected value but got None/Err");
        assert!(matches!(expr, Expr::Unary(OpKind::Abs, _)));
    }

    #[test]
    fn parse_kernel_code_nested_should_succeed_when_called() {
        let expr = parse_kernel_code("((X + Y) * Z)").expect("Expected value but got None/Err");
        if let Expr::Binary(OpKind::Mul, left, right) = expr {
            assert!(matches!(*left, Expr::Binary(OpKind::Add, _, _)));
            assert!(matches!(*right, Expr::Var(2)));
        } else {
            panic!("Expected Binary(Mul, ...) got {:?}", expr);
        }
    }

    #[test]
    fn parse_kernel_code_method_chains_should_succeed_when_called() {
        let expr = parse_kernel_code("(X).sqrt()");
        assert!(expr.is_some(), "Should parse (X).sqrt()");

        let expr = parse_kernel_code("(X).min(Y)");
        assert!(expr.is_some(), "Should parse (X).min(Y)");

        let expr = parse_kernel_code("((X).abs()).abs()");
        assert!(expr.is_some(), "Should parse chained abs");

        let expr = parse_kernel_code("(((X).rsqrt()).abs())");
        assert!(expr.is_some(), "Should parse rsqrt then abs");
    }

    #[test]
    fn parse_kernel_code_complex_expressions_should_succeed_when_called() {
        let expr = parse_kernel_code("((-0.724020)).rsqrt()");
        assert!(expr.is_some(), "Should parse rsqrt of negative const: {:?}", expr);

        let expr = parse_kernel_code("((((X).rsqrt()).abs()).abs())");
        assert!(expr.is_some(), "Should parse deeply nested methods");

        let expr = parse_kernel_code("(X).min(((-Z)).max(Y))");
        assert!(expr.is_some(), "Should parse nested min/max");

        let expr = parse_kernel_code("(((-3.551370)).rsqrt() * (1.0 / W))");
        assert!(expr.is_some(), "Should parse rsqrt multiplication");
    }

    #[test]
    fn parse_actual_failures_should_succeed_when_called() {
        let expr = parse_kernel_code(
            "((((X).rsqrt()).abs()).abs()).min(((((-0.724020)).rsqrt() * (1.0 / (X).abs()))).min(W))"
        );
        assert!(expr.is_some(), "Should parse chained min: {:?}", expr);

        let expr = parse_kernel_code(
            "((W * (((Y * X)).max(X)).rsqrt()) + (W * (-(((Y).abs() * Z) - (((0.296980) * Z) + (-W))))))"
        );
        assert!(expr.is_some(), "Should parse complex expression");
    }

    // ================================================================
    // expr_to_kernel_code round-trip tests
    // ================================================================

    #[test]
    fn expr_to_kernel_code_variables_should_succeed_when_called() {
        assert_eq!(expr_to_kernel_code(&Expr::Var(0)), "X");
        assert_eq!(expr_to_kernel_code(&Expr::Var(1)), "Y");
        assert_eq!(expr_to_kernel_code(&Expr::Var(2)), "Z");
        assert_eq!(expr_to_kernel_code(&Expr::Var(3)), "W");
    }

    #[test]
    fn expr_to_kernel_code_constants_should_succeed_when_called() {
        let code = expr_to_kernel_code(&Expr::Const(3.14));
        let reparsed = parse_kernel_code(&code);
        assert!(reparsed.is_some(), "Failed to reparse constant: {}", code);
    }

    /// String-level round-trip: serialize → parse → re-serialize must be identical.
    fn assert_string_roundtrip(code: &str) {
        let parsed = parse_kernel_code(code)
            .unwrap_or_else(|| panic!("parse_kernel_code failed on: {}", code));
        let re_emitted = expr_to_kernel_code(&parsed);
        assert_eq!(code, re_emitted, "String round-trip failed");
    }

    #[test]
    fn expr_to_kernel_code_roundtrip_simple_should_succeed_when_called() {
        let expr = Expr::Binary(OpKind::Add, Box::new(Expr::Var(0)), Box::new(Expr::Var(1)));
        let code = expr_to_kernel_code(&expr);
        assert_string_roundtrip(&code);
    }

    #[test]
    fn expr_to_kernel_code_roundtrip_unary_methods_should_succeed_when_called() {
        for i in 0..OpKind::COUNT {
            let Some(op) = OpKind::from_index(i) else { continue };
            if op.arity() != 1 { continue; }
            let expr = Expr::Unary(op, Box::new(Expr::Var(0)));
            let code = expr_to_kernel_code(&expr);
            assert_string_roundtrip(&code);
        }
    }

    #[test]
    fn expr_to_kernel_code_roundtrip_binary_should_succeed_when_called() {
        for i in 0..OpKind::COUNT {
            let Some(op) = OpKind::from_index(i) else { continue };
            if op.arity() != 2 { continue; }
            let expr = Expr::Binary(op, Box::new(Expr::Var(0)), Box::new(Expr::Var(1)));
            let code = expr_to_kernel_code(&expr);
            assert_string_roundtrip(&code);
        }
    }

    #[test]
    fn expr_to_kernel_code_roundtrip_ternary_should_succeed_when_called() {
        for i in 0..OpKind::COUNT {
            let Some(op) = OpKind::from_index(i) else { continue };
            if op.arity() != 3 { continue; }
            let expr = Expr::Ternary(
                op,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Var(1)),
                Box::new(Expr::Var(2)),
            );
            let code = expr_to_kernel_code(&expr);
            assert_string_roundtrip(&code);
        }
    }

    #[test]
    fn expr_to_kernel_code_roundtrip_generated_should_succeed_when_called() {
        use pixelflow_search::nnue::{ExprGenConfig, ExprGenerator};
        let config = ExprGenConfig {
            max_depth: 6,
            leaf_prob: 0.3,
            num_vars: 4,
            include_fused: false,
        };
        let mut rng = ExprGenerator::new(12345, config);
        for i in 0..200 {
            let expr = rng.generate();
            let code = expr_to_kernel_code(&expr);
            let reparsed = parse_kernel_code(&code)
                .unwrap_or_else(|| panic!("parse failed on expr #{}: {}", i, code));
            let re_emitted = expr_to_kernel_code(&reparsed);
            assert_eq!(
                code, re_emitted,
                "Round-trip failed on expr #{}", i
            );
        }
    }
}
