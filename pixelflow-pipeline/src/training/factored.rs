//! # Expression parsing and serialization utilities for NNUE training.
//!
//! Provides parsers for two expression syntaxes:
//! - **S-expression**: `Add(Mul(Var(0), Var(1)), Var(2))` (test-only repro format)
//! - **Kernel code**: `(X * Y) + Z` (human-readable, round-trips with
//!   `parse_kernel_code_arena`/`arena_to_kernel_code`)

use std::collections::HashMap;
#[cfg(test)]
use std::sync::Arc;

use pixelflow_ir::arena::ExprNode;
use pixelflow_ir::{EmitStyle, ExprArena, ExprId, OpKind};
#[cfg(test)]
use pixelflow_search::egraph::Pattern as Expr;

// ============================================================================
// Expression Parsing (for loading training data)
// ============================================================================

/// Parse an expression from a string representation.
///
/// Test-only helper for older logged repros in `OpName(child1, child2, ...)`
/// form such as `Add(Mul(Var(0), Var(1)), Var(2))`.
#[cfg(test)]
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
            Some(Expr::Unary(op, Arc::new(a)))
        }
        2 => {
            if children.len() != 2 {
                return None;
            }
            let a = parse_expr(children[0])?;
            let b = parse_expr(children[1])?;
            Some(Expr::Binary(op, Arc::new(a), Arc::new(b)))
        }
        3 => {
            if children.len() != 3 {
                return None;
            }
            let a = parse_expr(children[0])?;
            let b = parse_expr(children[1])?;
            let c = parse_expr(children[2])?;
            Some(Expr::Ternary(op, Arc::new(a), Arc::new(b), Arc::new(c)))
        }
        _ => None,
    }
}

/// Parse operation name to OpKind.
#[cfg(test)]
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
#[cfg(test)]
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
// Kernel Code Parsing
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

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum NodeKey {
    Var(u8),
    Const(u32),
    Unary(OpKind, ExprId),
    Binary(OpKind, ExprId, ExprId),
    Ternary(OpKind, ExprId, ExprId, ExprId),
}

struct ArenaInterner {
    arena: ExprArena,
    nodes: HashMap<NodeKey, ExprId>,
}

impl ArenaInterner {
    fn new() -> Self {
        Self {
            arena: ExprArena::new(),
            nodes: HashMap::new(),
        }
    }

    fn push_key(&mut self, key: NodeKey) -> ExprId {
        if let Some(&existing) = self.nodes.get(&key) {
            return existing;
        }

        let id = match key {
            NodeKey::Var(i) => self.arena.push_var(i),
            NodeKey::Const(bits) => self.arena.push_const(f32::from_bits(bits)),
            NodeKey::Unary(op, a) => self.arena.push_unary(op, a),
            NodeKey::Binary(op, a, b) => self.arena.push_binary(op, a, b),
            NodeKey::Ternary(op, a, b, c) => self.arena.push_ternary(op, a, b, c),
        };
        self.nodes.insert(key, id);
        id
    }

    fn push_var(&mut self, index: u8) -> ExprId {
        self.push_key(NodeKey::Var(index))
    }

    fn push_const(&mut self, value: f32) -> ExprId {
        self.push_key(NodeKey::Const(value.to_bits()))
    }

    fn push_unary(&mut self, op: OpKind, a: ExprId) -> ExprId {
        self.push_key(NodeKey::Unary(op, a))
    }

    fn push_binary(&mut self, op: OpKind, a: ExprId, b: ExprId) -> ExprId {
        self.push_key(NodeKey::Binary(op, a, b))
    }

    fn push_ternary(&mut self, op: OpKind, a: ExprId, b: ExprId, c: ExprId) -> ExprId {
        self.push_key(NodeKey::Ternary(op, a, b, c))
    }

    fn finish(self, root: ExprId) -> (ExprArena, ExprId) {
        (self.arena, root)
    }
}

enum ParseOp {
    PrefixNeg,
    Binary(OpKind),
    GroupParen,
    GroupMethod { op: OpKind, commas: usize },
}

struct ArenaKernelParser<'a> {
    input: &'a str,
    pos: usize,
    interner: ArenaInterner,
}

impl<'a> ArenaKernelParser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            input,
            pos: 0,
            interner: ArenaInterner::new(),
        }
    }

    fn parse(mut self) -> Option<(ExprArena, ExprId)> {
        let root = self.parse_expr()?;
        self.skip_ws();
        (self.pos == self.input.len()).then(|| self.interner.finish(root))
    }

    fn parse_expr(&mut self) -> Option<ExprId> {
        let mut values = Vec::new();
        let mut ops = Vec::new();
        let mut expecting_operand = true;

        loop {
            self.skip_ws();

            if expecting_operand {
                match self.peek_char()? {
                    '(' => {
                        self.pos += 1;
                        ops.push(ParseOp::GroupParen);
                    }
                    '-' => {
                        self.pos += 1;
                        ops.push(ParseOp::PrefixNeg);
                    }
                    'X' | 'Y' | 'Z' | 'W' => {
                        let var = self.parse_variable_id()?;
                        values.push(self.interner.push_var(var));
                        expecting_operand = false;
                    }
                    c if c.is_ascii_digit() || c == '.' => {
                        let number = self.parse_number_literal()?;
                        values.push(self.interner.push_const(number));
                        expecting_operand = false;
                    }
                    _ => return None,
                }
                continue;
            }

            if self.consume_char('.') {
                let method = self.parse_ident_at_pos()?;
                self.skip_ws();
                if !self.consume_char('(') {
                    return None;
                }

                let op = OpKind::from_name(method)?;
                if op.arity() == 1 {
                    self.skip_ws();
                    if !self.consume_char(')') {
                        return None;
                    }
                    let base = values.pop()?;
                    values.push(self.interner.push_unary(op, base));
                } else {
                    ops.push(ParseOp::GroupMethod { op, commas: 0 });
                    expecting_operand = true;
                }
                continue;
            }

            Self::reduce_prefix_negs(&mut self.interner, &mut values, &mut ops)?;
            self.skip_ws();

            match self.peek_char() {
                Some('+') | Some('-') | Some('*') | Some('/') => {
                    let op = self.parse_binary_op()?;
                    Self::reduce_binary_ops(&mut self.interner, &mut values, &mut ops, op)?;
                    ops.push(ParseOp::Binary(op));
                    expecting_operand = true;
                }
                Some(',') => {
                    self.pos += 1;
                    Self::reduce_until_group(&mut self.interner, &mut values, &mut ops)?;
                    match ops.last_mut() {
                        Some(ParseOp::GroupMethod { commas, .. }) => {
                            *commas += 1;
                            expecting_operand = true;
                        }
                        _ => return None,
                    }
                }
                Some(')') => {
                    self.pos += 1;
                    Self::reduce_until_group(&mut self.interner, &mut values, &mut ops)?;
                    match ops.pop()? {
                        ParseOp::GroupParen => {}
                        ParseOp::GroupMethod { op, commas } => {
                            let explicit_args = commas + 1;
                            if explicit_args + 1 != op.arity() {
                                return None;
                            }
                            let value = match op.arity() {
                                2 => {
                                    let b = values.pop()?;
                                    let a = values.pop()?;
                                    self.interner.push_binary(op, a, b)
                                }
                                3 => {
                                    let c = values.pop()?;
                                    let b = values.pop()?;
                                    let a = values.pop()?;
                                    self.interner.push_ternary(op, a, b, c)
                                }
                                _ => return None,
                            };
                            values.push(value);
                        }
                        ParseOp::PrefixNeg | ParseOp::Binary(_) => return None,
                    }
                }
                None => {
                    Self::reduce_until_group(&mut self.interner, &mut values, &mut ops)?;
                    if !ops.is_empty() {
                        return None;
                    }
                    return (values.len() == 1).then(|| values.pop()).flatten();
                }
                _ => return None,
            }
        }
    }

    fn reduce_prefix_negs(
        interner: &mut ArenaInterner,
        values: &mut Vec<ExprId>,
        ops: &mut Vec<ParseOp>,
    ) -> Option<()> {
        while matches!(ops.last(), Some(ParseOp::PrefixNeg)) {
            ops.pop();
            let value = values.pop()?;
            values.push(interner.push_unary(OpKind::Neg, value));
        }
        Some(())
    }

    fn reduce_binary_ops(
        interner: &mut ArenaInterner,
        values: &mut Vec<ExprId>,
        ops: &mut Vec<ParseOp>,
        incoming: OpKind,
    ) -> Option<()> {
        loop {
            match ops.last() {
                Some(ParseOp::Binary(current))
                    if Self::precedence(*current) >= Self::precedence(incoming) =>
                {
                    let current = match ops.pop()? {
                        ParseOp::Binary(op) => op,
                        _ => unreachable!(),
                    };
                    let b = values.pop()?;
                    let a = values.pop()?;
                    values.push(interner.push_binary(current, a, b));
                }
                Some(ParseOp::PrefixNeg) => {
                    Self::reduce_prefix_negs(interner, values, ops)?;
                }
                _ => break,
            }
        }
        Some(())
    }

    fn reduce_until_group(
        interner: &mut ArenaInterner,
        values: &mut Vec<ExprId>,
        ops: &mut Vec<ParseOp>,
    ) -> Option<()> {
        loop {
            match ops.last() {
                Some(ParseOp::Binary(_)) => {
                    let op = match ops.pop()? {
                        ParseOp::Binary(op) => op,
                        _ => unreachable!(),
                    };
                    let b = values.pop()?;
                    let a = values.pop()?;
                    values.push(interner.push_binary(op, a, b));
                }
                Some(ParseOp::PrefixNeg) => {
                    Self::reduce_prefix_negs(interner, values, ops)?;
                }
                _ => break,
            }
        }
        Some(())
    }

    fn precedence(op: OpKind) -> u8 {
        match op {
            OpKind::Add | OpKind::Sub => 1,
            OpKind::Mul | OpKind::Div => 2,
            _ => 0,
        }
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek_char(), Some(c) if c.is_whitespace()) {
            self.pos += 1;
        }
    }

    fn peek_char(&self) -> Option<char> {
        self.input[self.pos..].chars().next()
    }

    fn consume_char(&mut self, expected: char) -> bool {
        if self.peek_char() == Some(expected) {
            self.pos += expected.len_utf8();
            true
        } else {
            false
        }
    }

    fn parse_variable_id(&mut self) -> Option<u8> {
        let var = match self.peek_char()? {
            'X' => 0,
            'Y' => 1,
            'Z' => 2,
            'W' => 3,
            _ => return None,
        };
        self.pos += 1;
        Some(var)
    }

    fn parse_number_literal(&mut self) -> Option<f32> {
        let start = self.pos;
        let mut seen_digit = false;
        let mut seen_dot = false;
        let mut seen_exp = false;

        while let Some(c) = self.peek_char() {
            match c {
                '0'..='9' => {
                    seen_digit = true;
                    self.pos += 1;
                }
                '.' if !seen_dot && !seen_exp => {
                    seen_dot = true;
                    self.pos += 1;
                }
                'e' | 'E' if seen_digit && !seen_exp => {
                    seen_exp = true;
                    self.pos += 1;
                    if matches!(self.peek_char(), Some('+') | Some('-')) {
                        self.pos += 1;
                    }
                }
                _ => break,
            }
        }

        (self.pos > start && seen_digit)
            .then(|| self.input[start..self.pos].parse().ok())
            .flatten()
    }

    fn parse_ident_at_pos(&mut self) -> Option<&'a str> {
        let start = self.pos;
        let (ident, _) = parse_ident(&self.input[start..])?;
        self.pos += ident.len();
        Some(ident)
    }

    fn parse_binary_op(&mut self) -> Option<OpKind> {
        let op = match self.peek_char()? {
            '+' => OpKind::Add,
            '-' => OpKind::Sub,
            '*' => OpKind::Mul,
            '/' => OpKind::Div,
            _ => return None,
        };
        self.pos += 1;
        Some(op)
    }
}

/// Parser result: (parsed value, remaining input)
#[cfg(test)]
type ParseResult<'a, T> = Option<(T, &'a str)>;

/// Parse kernel code syntax like "(X + Y)" into Expr.
#[cfg(test)]
pub fn parse_kernel_code(s: &str) -> Option<Expr> {
    kc_expr(s.trim()).and_then(|(expr, rest)| rest.is_empty().then_some(expr))
}

/// Parse kernel code directly into an [`ExprArena`] (DAG) with structural sharing.
///
/// Identical subexpressions map to the same [`ExprId`], so the returned arena is
/// a true DAG rather than a duplicated tree.  The dedup key is `ExprNode` equality:
/// two nodes are shared iff they have the same [`OpKind`] and the same child
/// [`ExprId`]s (or the same leaf value).
///
/// Returns `None` if the input fails to parse.
pub fn parse_kernel_code_arena(s: &str) -> Option<(ExprArena, ExprId)> {
    ArenaKernelParser::new(s.trim()).parse()
}

/// Top-level: parse a complete expression
#[cfg(test)]
fn kc_expr(input: &str) -> ParseResult<'_, Expr> {
    parse_additive(input.trim())
}

/// Parse additive: left-associative chain of +/-
#[cfg(test)]
fn parse_additive(input: &str) -> ParseResult<'_, Expr> {
    let (mut acc, mut rest) = parse_multiplicative(input)?;

    while let Some((op, remaining)) = parse_additive_op(rest.trim_start()) {
        let (rhs, remaining) = parse_multiplicative(remaining.trim_start())?;
        acc = Expr::Binary(op, Arc::new(acc), Arc::new(rhs));
        rest = remaining;
    }

    Some((acc, rest))
}

#[cfg(test)]
fn parse_additive_op(input: &str) -> ParseResult<'_, OpKind> {
    match input.chars().next()? {
        '+' => Some((OpKind::Add, &input[1..])),
        '-' => Some((OpKind::Sub, &input[1..])),
        _ => None,
    }
}

/// Parse multiplicative: left-associative chain of * /
#[cfg(test)]
fn parse_multiplicative(input: &str) -> ParseResult<'_, Expr> {
    let (mut acc, mut rest) = parse_postfix(input)?;

    while let Some((op, remaining)) = parse_multiplicative_op(rest.trim_start()) {
        let (rhs, remaining) = parse_postfix(remaining.trim_start())?;
        acc = Expr::Binary(op, Arc::new(acc), Arc::new(rhs));
        rest = remaining;
    }

    Some((acc, rest))
}

#[cfg(test)]
fn parse_multiplicative_op(input: &str) -> ParseResult<'_, OpKind> {
    match input.chars().next()? {
        '*' => Some((OpKind::Mul, &input[1..])),
        '/' => Some((OpKind::Div, &input[1..])),
        _ => None,
    }
}

/// Parse postfix: primary followed by method chains
#[cfg(test)]
fn parse_postfix(input: &str) -> ParseResult<'_, Expr> {
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
#[cfg(test)]
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
        return Some((Expr::Unary(op, Arc::new(base)), rest));
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
            Expr::Ternary(op, Arc::new(base), Arc::new(arg1), Arc::new(arg2)),
            rest,
        ));
    }

    // Binary method: .method(arg)
    let rest = rest.strip_prefix(')')?;
    let op = OpKind::from_name(method_name)?;
    if op.arity() != 2 {
        return None;
    }
    Some((Expr::Binary(op, Arc::new(base), Arc::new(arg1)), rest))
}

/// Parse primary: parens, negation, variable, or number
#[cfg(test)]
fn parse_primary(input: &str) -> ParseResult<'_, Expr> {
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
        return Some((Expr::Unary(OpKind::Neg, Arc::new(expr)), rest));
    }

    // Variable or number
    parse_variable(input).or_else(|| parse_number(input))
}

/// Parse a variable: X, Y, Z, W
#[cfg(test)]
fn parse_variable(input: &str) -> ParseResult<'_, Expr> {
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
#[cfg(test)]
fn parse_number(input: &str) -> ParseResult<'_, Expr> {
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
fn parse_ident(input: &str) -> Option<(&str, &str)> {
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
#[cfg(test)]
pub fn expr_to_kernel_code(expr: &Expr) -> String {
    match expr {
        Expr::Var(0) => "X".into(),
        Expr::Var(1) => "Y".into(),
        Expr::Var(2) => "Z".into(),
        Expr::Var(3) => "W".into(),
        Expr::Var(i) => panic!(
            "expr_to_kernel_code: variable index {} exceeds X/Y/Z/W range",
            i
        ),
        Expr::Const(v) => format_const_kc(*v),
        Expr::Param(i) => panic!(
            "Expr::Param({}) reached expr_to_kernel_code — call substitute_params first",
            i
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
            "expr_to_kernel_code: Nary({}) not representable in kernel code syntax",
            op.name()
        ),
    }
}

/// Convert an [`ExprArena`] subtree into kernel code syntax.
pub fn arena_to_kernel_code(arena: &ExprArena, root: ExprId) -> String {
    enum Task {
        Visit(ExprId),
        Emit { node: ExprId, arity: usize },
    }

    let mut stack = vec![Task::Visit(root)];
    let mut result_stack: Vec<String> = Vec::new();

    while let Some(task) = stack.pop() {
        match task {
            Task::Visit(id) => {
                let arity = arena.children(id).len();
                stack.push(Task::Emit { node: id, arity });
                let children: Vec<ExprId> = arena.children(id).collect();
                for child in children.into_iter().rev() {
                    stack.push(Task::Visit(child));
                }
            }
            Task::Emit { node, arity } => {
                let start = result_stack.len().saturating_sub(arity);
                let args: Vec<String> = result_stack.drain(start..).collect();
                let emitted = match arena.node(node) {
                    ExprNode::Var(0) => "X".into(),
                    ExprNode::Var(1) => "Y".into(),
                    ExprNode::Var(2) => "Z".into(),
                    ExprNode::Var(3) => "W".into(),
                    ExprNode::Var(i) => panic!(
                        "arena_to_kernel_code: variable index {} exceeds X/Y/Z/W range",
                        i
                    ),
                    ExprNode::Const(v) => format_const_kc(*v),
                    ExprNode::Param(i) => panic!(
                        "ExprNode::Param({}) reached arena_to_kernel_code — substitute params first",
                        i
                    ),
                    ExprNode::Unary(op, _)
                    | ExprNode::Binary(op, _, _)
                    | ExprNode::Ternary(op, _, _, _) => emit_op_kc(*op, &args),
                    ExprNode::Nary(op, _, _) => panic!(
                        "arena_to_kernel_code: Nary({}) not representable in kernel code syntax",
                        op.name()
                    ),
                };
                result_stack.push(emitted);
            }
        }
    }

    result_stack
        .pop()
        .unwrap_or_else(|| panic!("arena_to_kernel_code: empty result stack"))
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
            op.name(),
            style,
            args.len()
        ),
    }
}

/// Format a constant for kernel code syntax.
fn format_const_kc(v: f32) -> String {
    if !v.is_finite() {
        panic!(
            "expr_to_kernel_code: non-finite constant {} cannot be represented",
            v
        );
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
    use crate::jit_bench::benchmark_jit_arena;

    const REWRITE_BUG_INPUTS: [f32; 4] = [0.5, 0.7, 1.3, -0.2];

    fn eval_expr_scalar(expr: &Expr, vars: &[f32; 4]) -> f32 {
        match expr {
            Expr::Var(i) => vars[*i as usize],
            Expr::Const(c) => *c,
            Expr::Param(i) => panic!("Param in eval_expr_scalar: {i}"),
            Expr::Unary(op, a) => {
                let a = eval_expr_scalar(a, vars);
                op.eval_unary(a)
                    .unwrap_or_else(|| panic!("eval_unary failed for {:?}", op))
            }
            Expr::Binary(op, a, b) => {
                let a = eval_expr_scalar(a, vars);
                let b = eval_expr_scalar(b, vars);
                op.eval_binary(a, b)
                    .unwrap_or_else(|| panic!("eval_binary failed for {:?}", op))
            }
            Expr::Ternary(op, a, b, c) => {
                let a = eval_expr_scalar(a, vars);
                let b = eval_expr_scalar(b, vars);
                let c = eval_expr_scalar(c, vars);
                op.eval_ternary(a, b, c)
                    .unwrap_or_else(|| panic!("eval_ternary failed for {:?}", op))
            }
            Expr::Nary(kind, _) => panic!("Nary in eval_expr_scalar: {:?}", kind),
        }
    }

    fn logged_expr_scalar_output(src: &str) -> f32 {
        let expr = parse_expr(src).unwrap_or_else(|| panic!("parse_expr failed: {src}"));
        eval_expr_scalar(&expr, &REWRITE_BUG_INPUTS)
    }

    fn logged_expr_jit_output(src: &str) -> f32 {
        let expr = parse_expr(src).unwrap_or_else(|| panic!("parse_expr failed: {src}"));
        let kernel = expr_to_kernel_code(&expr);
        let (arena, root) = parse_kernel_code_arena(&kernel)
            .unwrap_or_else(|| panic!("parse_kernel_code_arena failed: {kernel}"));
        benchmark_jit_arena(&arena, root)
            .unwrap_or_else(|err| panic!("benchmark_jit_arena failed for {kernel}: {err:?}"))
            .output[0]
    }

    fn logged_expr_roundtrip_scalar_output(src: &str) -> f32 {
        let expr = parse_expr(src).unwrap_or_else(|| panic!("parse_expr failed: {src}"));
        let kernel = expr_to_kernel_code(&expr);
        let reparsed = parse_kernel_code(&kernel)
            .unwrap_or_else(|| panic!("parse_kernel_code failed: {kernel}"));
        eval_expr_scalar(&reparsed, &REWRITE_BUG_INPUTS)
    }

    fn assert_scalar_and_jit_close(src: &str, epsilon: f32) {
        let scalar = logged_expr_scalar_output(src);
        let jit = logged_expr_jit_output(src);
        let diff = (scalar - jit).abs();
        assert!(
            diff <= epsilon,
            "scalar/JIT mismatch\nexpr: {src}\nscalar: {scalar}\njit: {jit}\ndiff: {diff} > {epsilon}"
        );
    }

    fn collect_subexpressions<'a>(expr: &'a Expr, out: &mut Vec<&'a Expr>) {
        out.push(expr);
        match expr {
            Expr::Unary(_, a) => collect_subexpressions(a, out),
            Expr::Binary(_, a, b) => {
                collect_subexpressions(a, out);
                collect_subexpressions(b, out);
            }
            Expr::Ternary(_, a, b, c) => {
                collect_subexpressions(a, out);
                collect_subexpressions(b, out);
                collect_subexpressions(c, out);
            }
            Expr::Var(_) | Expr::Const(_) | Expr::Param(_) | Expr::Nary(_, _) => {}
        }
    }

    #[test]
    fn test_parse_kernel_code_arena_basic() {
        // Simple expression: no structural sharing expected.
        let (arena, root) = parse_kernel_code_arena("(X + Y)").unwrap();
        assert!(
            arena.len() >= 3,
            "expected at least 3 nodes (X, Y, Add); got {}",
            arena.len()
        );
        let _ = root; // root is valid
    }

    #[test]
    fn test_parse_kernel_code_arena_structural_sharing() {
        // (X + X): the two X leaves are structurally identical and should share an id.
        let (arena, _root) = parse_kernel_code_arena("(X + X)").unwrap();
        // Without sharing: 3 nodes (X, X, Add). With sharing: 2 nodes (X, Add).
        assert_eq!(
            arena.len(),
            2,
            "expected 2 unique nodes for (X + X) with sharing, got {}",
            arena.len()
        );
    }

    #[test]
    fn test_parse_kernel_code_arena_deeply_shared() {
        // ((X + Y) * (X + Y)): the (X + Y) subtree appears twice — should be shared.
        // Without sharing: 7 nodes. With sharing: 4 nodes (X, Y, Add, Mul).
        let (arena, _root) = parse_kernel_code_arena("((X + Y) * (X + Y))").unwrap();
        assert_eq!(
            arena.len(),
            4,
            "expected 4 unique nodes for ((X+Y)*(X+Y)) with sharing, got {}",
            arena.len()
        );
    }

    #[test]
    fn test_parse_kernel_code_arena_round_trip() {
        // Arena parse + arena_to_kernel_code should match the original.
        let src = "((X * Y) + (X * Y))";
        let (arena, root) = parse_kernel_code_arena(src).unwrap();
        let code = arena_to_kernel_code(&arena, root);
        // The round-tripped form should parse cleanly.
        assert!(
            parse_kernel_code(&code).is_some(),
            "arena round-trip produced un-parseable code: {}",
            code
        );
    }

    #[test]
    fn test_parse_kernel_code_arena_methods() {
        let src = "(((X).abs()).min(Y)).mul_add(Z, W)";
        let (arena, root) = parse_kernel_code_arena(src).unwrap();
        let code = arena_to_kernel_code(&arena, root);
        let (reparsed, reparsed_root) = parse_kernel_code_arena(&code).unwrap();
        assert_eq!(code, arena_to_kernel_code(&reparsed, reparsed_root));
        assert!(
            arena.len() >= 7,
            "expected method-heavy parse to build a real DAG"
        );
    }

    #[test]
    fn test_parse_expr() {
        let expr = parse_expr("Add(Var(0), Var(1))").unwrap();
        assert!(matches!(expr, Expr::Binary(OpKind::Add, _, _)));

        let expr = parse_expr("Mul(Add(Var(0), Var(1)), Var(2))").unwrap();
        assert!(matches!(expr, Expr::Binary(OpKind::Mul, _, _)));

        let expr = parse_expr("MulAdd(Var(0), Var(1), Var(2))").unwrap();
        assert!(matches!(expr, Expr::Ternary(OpKind::MulAdd, _, _, _)));
    }

    // ========================================================================
    // Kernel Code Parser Tests
    // ========================================================================

    #[test]
    fn test_parse_kernel_code_variables() {
        assert!(matches!(parse_kernel_code("X"), Some(Expr::Var(0))));
        assert!(matches!(parse_kernel_code("Y"), Some(Expr::Var(1))));
        assert!(matches!(parse_kernel_code("Z"), Some(Expr::Var(2))));
        assert!(matches!(parse_kernel_code("W"), Some(Expr::Var(3))));
    }

    #[test]
    fn test_parse_kernel_code_constants() {
        assert!(matches!(parse_kernel_code("1.0"), Some(Expr::Const(v)) if (v - 1.0).abs() < 1e-6));
        assert!(
            matches!(parse_kernel_code("(4.595877)"), Some(Expr::Const(v)) if (v - 4.595877).abs() < 1e-5)
        );
        assert!(matches!(parse_kernel_code("0.0"), Some(Expr::Const(v)) if v.abs() < 1e-6));
    }

    #[test]
    fn test_parse_kernel_code_binary_ops() {
        let expr = parse_kernel_code("(X + Y)").unwrap();
        assert!(matches!(expr, Expr::Binary(OpKind::Add, _, _)));

        let expr = parse_kernel_code("(X - Y)").unwrap();
        assert!(matches!(expr, Expr::Binary(OpKind::Sub, _, _)));

        let expr = parse_kernel_code("(X * Y)").unwrap();
        assert!(matches!(expr, Expr::Binary(OpKind::Mul, _, _)));

        let expr = parse_kernel_code("(X / Y)").unwrap();
        assert!(matches!(expr, Expr::Binary(OpKind::Div, _, _)));
    }

    #[test]
    fn test_parse_kernel_code_from_benchmark_cache() {
        let expr = parse_kernel_code("((4.595877) - Z)").unwrap();
        assert!(matches!(expr, Expr::Binary(OpKind::Sub, _, _)));

        let expr = parse_kernel_code("((4.595877) + (-Z))").unwrap();
        assert!(matches!(expr, Expr::Binary(OpKind::Add, _, _)));

        let expr = parse_kernel_code("((-Z) + (4.595877))").unwrap();
        assert!(matches!(expr, Expr::Binary(OpKind::Add, _, _)));
    }

    #[test]
    fn test_parse_kernel_code_unary_ops() {
        let expr = parse_kernel_code("(-X)").unwrap();
        assert!(matches!(expr, Expr::Unary(OpKind::Neg, _)));

        let expr = parse_kernel_code("(X).sqrt()").unwrap();
        assert!(matches!(expr, Expr::Unary(OpKind::Sqrt, _)));

        let expr = parse_kernel_code("(X).abs()").unwrap();
        assert!(matches!(expr, Expr::Unary(OpKind::Abs, _)));
    }

    #[test]
    fn test_parse_kernel_code_nested() {
        let expr = parse_kernel_code("((X + Y) * Z)").unwrap();
        if let Expr::Binary(OpKind::Mul, left, right) = expr {
            assert!(matches!(*left, Expr::Binary(OpKind::Add, _, _)));
            assert!(matches!(*right, Expr::Var(2)));
        } else {
            panic!("Expected Binary(Mul, ...) got {:?}", expr);
        }
    }

    #[test]
    fn test_parse_kernel_code_method_chains() {
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
    fn test_parse_kernel_code_complex_expressions() {
        let expr = parse_kernel_code("((-0.724020)).rsqrt()");
        assert!(
            expr.is_some(),
            "Should parse rsqrt of negative const: {:?}",
            expr
        );

        let expr = parse_kernel_code("((((X).rsqrt()).abs()).abs())");
        assert!(expr.is_some(), "Should parse deeply nested methods");

        let expr = parse_kernel_code("(X).min(((-Z)).max(Y))");
        assert!(expr.is_some(), "Should parse nested min/max");

        let expr = parse_kernel_code("(((-3.551370)).rsqrt() * (1.0 / W))");
        assert!(expr.is_some(), "Should parse rsqrt multiplication");
    }

    #[test]
    fn test_parse_actual_failures() {
        let expr = parse_kernel_code(
            "((((X).rsqrt()).abs()).abs()).min(((((-0.724020)).rsqrt() * (1.0 / (X).abs()))).min(W))",
        );
        assert!(expr.is_some(), "Should parse chained min: {:?}", expr);

        let expr = parse_kernel_code(
            "((W * (((Y * X)).max(X)).rsqrt()) + (W * (-(((Y).abs() * Z) - (((0.296980) * Z) + (-W))))))",
        );
        assert!(expr.is_some(), "Should parse complex expression");
    }

    // ================================================================
    // expr_to_kernel_code round-trip tests
    // ================================================================

    #[test]
    fn test_expr_to_kernel_code_variables() {
        assert_eq!(expr_to_kernel_code(&Expr::Var(0)), "X");
        assert_eq!(expr_to_kernel_code(&Expr::Var(1)), "Y");
        assert_eq!(expr_to_kernel_code(&Expr::Var(2)), "Z");
        assert_eq!(expr_to_kernel_code(&Expr::Var(3)), "W");
    }

    #[test]
    fn test_expr_to_kernel_code_constants() {
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
    fn test_expr_to_kernel_code_roundtrip_simple() {
        let expr = Expr::Binary(OpKind::Add, Arc::new(Expr::Var(0)), Arc::new(Expr::Var(1)));
        let code = expr_to_kernel_code(&expr);
        assert_string_roundtrip(&code);
    }

    #[test]
    fn test_expr_to_kernel_code_roundtrip_unary_methods() {
        for i in 0..OpKind::COUNT {
            let Some(op) = OpKind::from_index(i) else {
                continue;
            };
            if op.arity() != 1 {
                continue;
            }
            let expr = Expr::Unary(op, Arc::new(Expr::Var(0)));
            let code = expr_to_kernel_code(&expr);
            assert_string_roundtrip(&code);
        }
    }

    #[test]
    fn test_expr_to_kernel_code_roundtrip_binary() {
        for i in 0..OpKind::COUNT {
            let Some(op) = OpKind::from_index(i) else {
                continue;
            };
            if op.arity() != 2 {
                continue;
            }
            let expr = Expr::Binary(op, Arc::new(Expr::Var(0)), Arc::new(Expr::Var(1)));
            let code = expr_to_kernel_code(&expr);
            assert_string_roundtrip(&code);
        }
    }

    #[test]
    fn test_expr_to_kernel_code_roundtrip_ternary() {
        for i in 0..OpKind::COUNT {
            let Some(op) = OpKind::from_index(i) else {
                continue;
            };
            if op.arity() != 3 {
                continue;
            }
            let expr = Expr::Ternary(
                op,
                Arc::new(Expr::Var(0)),
                Arc::new(Expr::Var(1)),
                Arc::new(Expr::Var(2)),
            );
            let code = expr_to_kernel_code(&expr);
            assert_string_roundtrip(&code);
        }
    }

    #[test]
    fn test_expr_to_kernel_code_roundtrip_generated() {
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
            assert_eq!(code, re_emitted, "Round-trip failed on expr #{}", i);
        }
    }

    #[test]
    fn test_seed_42_t1_pair_is_close_under_scalar_semantics() {
        let initial = "log2(add(abs(neg(atan2(pow(add(abs(neg(pow(add(abs(neg(neg(Const(-0.9631642)))), Const(0.001)), Const(-1)))), Const(0.001)), Const(-1)), mul(exp(add(pow(add(abs(neg(neg(atan2(Const(0.1167655), Var(1))))), Const(0.001)), Const(-1)), add(min(Var(3), Var(1)), log2(add(abs(neg(Var(3))), Const(0.001)))))), min(min(sub(cos(neg(Const(1.5691397))), add(Const(-1.1460416), Var(2))), abs(exp(Var(0)))), log10(add(abs(min(sub(Const(0.1855638), Var(1)), exp(Var(1)))), Const(0.001)))))))), Const(0.001)))";
        let final_ = "log2(add(Const(0.001), abs(atan2(pow(add(abs(neg(pow(add(abs(Const(-0.9631642)), Const(0.001)), Const(-1)))), Const(0.001)), Const(-1)), mul(mul(min(min(sub(cos(neg(Const(1.5691397))), add(Const(-1.1460416), Var(2))), abs(exp(Var(0)))), log10(add(abs(min(sub(Const(0.1855638), Var(1)), exp(Var(1)))), Const(0.001)))), exp(pow(add(abs(atan2(Const(0.1167655), Var(1))), Const(0.001)), Const(-1)))), exp(add(min(Var(3), Var(1)), log2(add(abs(neg(Var(3))), Const(0.001))))))))))";
        let initial = logged_expr_scalar_output(initial);
        let final_ = logged_expr_scalar_output(final_);
        let diff = (initial - final_).abs();
        assert!(
            diff <= 1e-6,
            "scalar rewrite mismatch\ninitial: {initial}\nfinal: {final_}\ndiff: {diff}"
        );
    }

    #[test]
    fn test_seed_42_t1_initial_jit_matches_scalar() {
        let initial = "log2(add(abs(neg(atan2(pow(add(abs(neg(pow(add(abs(neg(neg(Const(-0.9631642)))), Const(0.001)), Const(-1)))), Const(0.001)), Const(-1)), mul(exp(add(pow(add(abs(neg(neg(atan2(Const(0.1167655), Var(1))))), Const(0.001)), Const(-1)), add(min(Var(3), Var(1)), log2(add(abs(neg(Var(3))), Const(0.001)))))), min(min(sub(cos(neg(Const(1.5691397))), add(Const(-1.1460416), Var(2))), abs(exp(Var(0)))), log10(add(abs(min(sub(Const(0.1855638), Var(1)), exp(Var(1)))), Const(0.001)))))))), Const(0.001)))";
        assert_scalar_and_jit_close(initial, 1e-3);
    }

    #[test]
    fn test_seed_42_t1_initial_roundtrip_scalar_matches_tree_scalar() {
        let initial = "log2(add(abs(neg(atan2(pow(add(abs(neg(pow(add(abs(neg(neg(Const(-0.9631642)))), Const(0.001)), Const(-1)))), Const(0.001)), Const(-1)), mul(exp(add(pow(add(abs(neg(neg(atan2(Const(0.1167655), Var(1))))), Const(0.001)), Const(-1)), add(min(Var(3), Var(1)), log2(add(abs(neg(Var(3))), Const(0.001)))))), min(min(sub(cos(neg(Const(1.5691397))), add(Const(-1.1460416), Var(2))), abs(exp(Var(0)))), log10(add(abs(min(sub(Const(0.1855638), Var(1)), exp(Var(1)))), Const(0.001)))))))), Const(0.001)))";
        let tree = logged_expr_scalar_output(initial);
        let roundtrip = logged_expr_roundtrip_scalar_output(initial);
        assert!(
            (tree - roundtrip).abs() <= 1e-6,
            "tree: {tree}, roundtrip: {roundtrip}"
        );
    }

    #[test]
    fn test_seed_42_t1_final_jit_matches_scalar() {
        let final_ = "log2(add(Const(0.001), abs(atan2(pow(add(abs(neg(pow(add(abs(Const(-0.9631642)), Const(0.001)), Const(-1)))), Const(0.001)), Const(-1)), mul(mul(min(min(sub(cos(neg(Const(1.5691397))), add(Const(-1.1460416), Var(2))), abs(exp(Var(0)))), log10(add(abs(min(sub(Const(0.1855638), Var(1)), exp(Var(1)))), Const(0.001)))), exp(pow(add(abs(atan2(Const(0.1167655), Var(1))), Const(0.001)), Const(-1)))), exp(add(min(Var(3), Var(1)), log2(add(abs(neg(Var(3))), Const(0.001))))))))))";
        assert_scalar_and_jit_close(final_, 1e-3);
    }

    #[test]
    fn test_seed_24042_t52_pair_is_close_under_scalar_semantics() {
        let initial = "div(min(add(mul(min(pow(add(abs(neg(neg(abs(neg(neg(add(mul(cos(neg(neg(Var(1)))), Const(0.52093434)), Const(-1.414685)))))))), Const(0.001)), Const(-1)), Var(1)), log10(add(abs(neg(neg(add(add(max(add(mul(Var(2), Var(2)), Var(3)), Var(1)), atan2(atan2(Var(0), Var(3)), add(Var(1), neg(Var(3))))), mul(ln(add(abs(neg(neg(neg(Const(0.12621832))))), Const(0.001))), mul_add(Var(0), cos(neg(neg(Var(3)))), pow(add(abs(neg(neg(Const(-0.20414245)))), Const(0.001)), Const(-0.5)))))))), Const(0.001)))), pow(add(abs(neg(neg(atan2(Var(2), log10(add(abs(neg(neg(tan(Var(1))))), Const(0.001))))))), Const(0.001)), Const(-0.5))), log2(add(abs(neg(neg(Var(0)))), Const(0.001)))), add(abs(neg(pow(add(abs(neg(pow(add(abs(neg(log2(add(abs(neg(mul(min(add(mul(log10(add(abs(neg(neg(Const(1.2403846)))), Const(0.001))), Var(2)), mul(log10(add(abs(neg(neg(Const(1.2403846)))), Const(0.001))), Var(3))), max(ln(add(abs(neg(neg(Const(0.2514913)))), Const(0.001))), mul(Var(0), Const(0.5072496)))), mul(pow(abs(neg(neg(log10(add(abs(neg(neg(Var(3)))), Const(0.001)))))), Const(0.5)), ln(add(abs(neg(sin(Var(2)))), Const(0.001))))))), Const(0.001))))), Const(0.001)), mul(add(add(mul(div(add(Const(0.61049294), Const(1.354384)), add(abs(neg(mul(Var(0), Var(0)))), Const(0.001))), mul(tan(Const(-1.6860065)), recip(add(abs(neg(Const(-1.7208018))), Const(0.001))))), add(mul(max(Var(3), Var(0)), cos(neg(Var(0)))), mul_add(Var(3), Var(1), Var(0)))), log10(add(abs(neg(Const(-1.1988422))), Const(0.001)))), max(abs(neg(ln(add(abs(neg(pow(add(abs(Var(1)), Const(0.001)), Var(2)))), Const(0.001))))), ln(add(abs(mul(Const(-0.47071946), pow(add(abs(neg(Const(-1.670574))), Const(0.001)), Var(0)))), Const(0.001)))))))), Const(0.001)), Const(-1)))), Const(0.001)))";
        let final_ = "div(min(add(mul(min(pow(add(abs(neg(neg(abs(neg(neg(add(mul(cos(neg(neg(Var(1)))), Const(0.52093434)), Const(-1.414685)))))))), Const(0.001)), Const(-1)), Var(1)), log10(add(abs(neg(neg(add(add(max(add(mul(Var(2), Var(2)), Var(3)), Var(1)), atan2(atan2(Var(0), Var(3)), add(Var(1), neg(Var(3))))), mul(ln(add(Const(0.12621832), Const(0.001))), mul_add(Var(0), cos(neg(neg(Var(3)))), pow(add(Const(0.20414245), Const(0.001)), Const(-0.5)))))))), Const(0.001)))), pow(add(abs(neg(neg(atan2(Var(2), log10(add(abs(neg(neg(tan(Var(1))))), Const(0.001))))))), Const(0.001)), Const(-0.5))), log2(add(abs(neg(neg(Var(0)))), Const(0.001)))), add(abs(neg(pow(add(abs(neg(pow(add(abs(neg(log2(add(abs(neg(mul(min(add(mul(Const(0.093906365), Var(2)), mul(Const(0.093906365), Var(3))), max(Const(-1.3763785), mul(Var(0), Const(0.5072496)))), mul(pow(abs(neg(neg(log10(add(abs(neg(neg(Var(3)))), Const(0.001)))))), Const(0.5)), ln(add(abs(neg(sin(Var(2)))), Const(0.001))))))), Const(0.001))))), Const(0.001)), mul(add(add(mul(div(add(Const(0.61049294), Const(1.354384)), add(abs(neg(mul(Var(0), Var(0)))), Const(0.001))), mul(Const(8.641348), recip(add(Const(1.7208018), Const(0.001))))), add(mul(max(Var(3), Var(0)), cos(neg(Var(0)))), mul_add(Var(3), Var(1), Var(0)))), log10(add(Const(1.1988422), Const(0.001)))), max(abs(neg(ln(add(abs(neg(pow(add(abs(Var(1)), Const(0.001)), Var(2)))), Const(0.001))))), ln(add(abs(mul(Const(-0.47071946), pow(add(Const(1.670574), Const(0.001)), Var(0)))), Const(0.001)))))))), Const(0.001)), Const(-1)))), Const(0.001)))";
        let initial = logged_expr_scalar_output(initial);
        let final_ = logged_expr_scalar_output(final_);
        let diff = (initial - final_).abs();
        assert!(
            diff <= 1e-3,
            "scalar rewrite mismatch\ninitial: {initial}\nfinal: {final_}\ndiff: {diff}"
        );
    }

    #[test]
    fn test_seed_24042_t52_initial_jit_matches_scalar() {
        let initial = "div(min(add(mul(min(pow(add(abs(neg(neg(abs(neg(neg(add(mul(cos(neg(neg(Var(1)))), Const(0.52093434)), Const(-1.414685)))))))), Const(0.001)), Const(-1)), Var(1)), log10(add(abs(neg(neg(add(add(max(add(mul(Var(2), Var(2)), Var(3)), Var(1)), atan2(atan2(Var(0), Var(3)), add(Var(1), neg(Var(3))))), mul(ln(add(abs(neg(neg(neg(Const(0.12621832))))), Const(0.001))), mul_add(Var(0), cos(neg(neg(Var(3)))), pow(add(abs(neg(neg(Const(-0.20414245)))), Const(0.001)), Const(-0.5)))))))), Const(0.001)))), pow(add(abs(neg(neg(atan2(Var(2), log10(add(abs(neg(neg(tan(Var(1))))), Const(0.001))))))), Const(0.001)), Const(-0.5))), log2(add(abs(neg(neg(Var(0)))), Const(0.001)))), add(abs(neg(pow(add(abs(neg(pow(add(abs(neg(log2(add(abs(neg(mul(min(add(mul(log10(add(abs(neg(neg(Const(1.2403846)))), Const(0.001))), Var(2)), mul(log10(add(abs(neg(neg(Const(1.2403846)))), Const(0.001))), Var(3))), max(ln(add(abs(neg(neg(Const(0.2514913)))), Const(0.001))), mul(Var(0), Const(0.5072496)))), mul(pow(abs(neg(neg(log10(add(abs(neg(neg(Var(3)))), Const(0.001)))))), Const(0.5)), ln(add(abs(neg(sin(Var(2)))), Const(0.001))))))), Const(0.001))))), Const(0.001)), mul(add(add(mul(div(add(Const(0.61049294), Const(1.354384)), add(abs(neg(mul(Var(0), Var(0)))), Const(0.001))), mul(tan(Const(-1.6860065)), recip(add(abs(neg(Const(-1.7208018))), Const(0.001))))), add(mul(max(Var(3), Var(0)), cos(neg(Var(0)))), mul_add(Var(3), Var(1), Var(0)))), log10(add(abs(neg(Const(-1.1988422))), Const(0.001)))), max(abs(neg(ln(add(abs(neg(pow(add(abs(Var(1)), Const(0.001)), Var(2)))), Const(0.001))))), ln(add(abs(mul(Const(-0.47071946), pow(add(abs(neg(Const(-1.670574))), Const(0.001)), Var(0)))), Const(0.001)))))))), Const(0.001)), Const(-1)))), Const(0.001)))";
        assert_scalar_and_jit_close(initial, 1e-3);
    }

    #[test]
    fn test_seed_24042_t52_initial_roundtrip_scalar_matches_tree_scalar() {
        let initial = "div(min(add(mul(min(pow(add(abs(neg(neg(abs(neg(neg(add(mul(cos(neg(neg(Var(1)))), Const(0.52093434)), Const(-1.414685)))))))), Const(0.001)), Const(-1)), Var(1)), log10(add(abs(neg(neg(add(add(max(add(mul(Var(2), Var(2)), Var(3)), Var(1)), atan2(atan2(Var(0), Var(3)), add(Var(1), neg(Var(3))))), mul(ln(add(abs(neg(neg(neg(Const(0.12621832))))), Const(0.001))), mul_add(Var(0), cos(neg(neg(Var(3)))), pow(add(abs(neg(neg(Const(-0.20414245)))), Const(0.001)), Const(-0.5)))))))), Const(0.001)))), pow(add(abs(neg(neg(atan2(Var(2), log10(add(abs(neg(neg(tan(Var(1))))), Const(0.001))))))), Const(0.001)), Const(-0.5))), log2(add(abs(neg(neg(Var(0)))), Const(0.001)))), add(abs(neg(pow(add(abs(neg(pow(add(abs(neg(log2(add(abs(neg(mul(min(add(mul(log10(add(abs(neg(neg(Const(1.2403846)))), Const(0.001))), Var(2)), mul(log10(add(abs(neg(neg(Const(1.2403846)))), Const(0.001))), Var(3))), max(ln(add(abs(neg(neg(Const(0.2514913)))), Const(0.001))), mul(Var(0), Const(0.5072496)))), mul(pow(abs(neg(neg(log10(add(abs(neg(neg(Var(3)))), Const(0.001)))))), Const(0.5)), ln(add(abs(neg(sin(Var(2)))), Const(0.001))))))), Const(0.001))))), Const(0.001)), mul(add(add(mul(div(add(Const(0.61049294), Const(1.354384)), add(abs(neg(mul(Var(0), Var(0)))), Const(0.001))), mul(tan(Const(-1.6860065)), recip(add(abs(neg(Const(-1.7208018))), Const(0.001))))), add(mul(max(Var(3), Var(0)), cos(neg(Var(0)))), mul_add(Var(3), Var(1), Var(0)))), log10(add(abs(neg(Const(-1.1988422))), Const(0.001)))), max(abs(neg(ln(add(abs(neg(pow(add(abs(Var(1)), Const(0.001)), Var(2)))), Const(0.001))))), ln(add(abs(mul(Const(-0.47071946), pow(add(abs(neg(Const(-1.670574))), Const(0.001)), Var(0)))), Const(0.001)))))))), Const(0.001)), Const(-1)))), Const(0.001)))";
        let tree = logged_expr_scalar_output(initial);
        let roundtrip = logged_expr_roundtrip_scalar_output(initial);
        assert!(
            (tree - roundtrip).abs() <= 1e-6,
            "tree: {tree}, roundtrip: {roundtrip}"
        );
    }

    #[test]
    fn test_seed_24042_t52_final_jit_matches_scalar() {
        let final_ = "div(min(add(mul(min(pow(add(abs(neg(neg(abs(neg(neg(add(mul(cos(neg(neg(Var(1)))), Const(0.52093434)), Const(-1.414685)))))))), Const(0.001)), Const(-1)), Var(1)), log10(add(abs(neg(neg(add(add(max(add(mul(Var(2), Var(2)), Var(3)), Var(1)), atan2(atan2(Var(0), Var(3)), add(Var(1), neg(Var(3))))), mul(ln(add(Const(0.12621832), Const(0.001))), mul_add(Var(0), cos(neg(neg(Var(3)))), pow(add(Const(0.20414245), Const(0.001)), Const(-0.5)))))))), Const(0.001)))), pow(add(abs(neg(neg(atan2(Var(2), log10(add(abs(neg(neg(tan(Var(1))))), Const(0.001))))))), Const(0.001)), Const(-0.5))), log2(add(abs(neg(neg(Var(0)))), Const(0.001)))), add(abs(neg(pow(add(abs(neg(pow(add(abs(neg(log2(add(abs(neg(mul(min(add(mul(Const(0.093906365), Var(2)), mul(Const(0.093906365), Var(3))), max(Const(-1.3763785), mul(Var(0), Const(0.5072496)))), mul(pow(abs(neg(neg(log10(add(abs(neg(neg(Var(3)))), Const(0.001)))))), Const(0.5)), ln(add(abs(neg(sin(Var(2)))), Const(0.001))))))), Const(0.001))))), Const(0.001)), mul(add(add(mul(div(add(Const(0.61049294), Const(1.354384)), add(abs(neg(mul(Var(0), Var(0)))), Const(0.001))), mul(Const(8.641348), recip(add(Const(1.7208018), Const(0.001))))), add(mul(max(Var(3), Var(0)), cos(neg(Var(0)))), mul_add(Var(3), Var(1), Var(0)))), log10(add(Const(1.1988422), Const(0.001)))), max(abs(neg(ln(add(abs(neg(pow(add(abs(Var(1)), Const(0.001)), Var(2)))), Const(0.001))))), ln(add(abs(mul(Const(-0.47071946), pow(add(Const(1.670574), Const(0.001)), Var(0)))), Const(0.001)))))))), Const(0.001)), Const(-1)))), Const(0.001)))";
        assert_scalar_and_jit_close(final_, 1e-3);
    }

    #[test]
    #[ignore]
    fn diagnose_seed_24042_t52_initial_subexpression_mismatches() {
        let initial = "div(min(add(mul(min(pow(add(abs(neg(neg(abs(neg(neg(add(mul(cos(neg(neg(Var(1)))), Const(0.52093434)), Const(-1.414685)))))))), Const(0.001)), Const(-1)), Var(1)), log10(add(abs(neg(neg(add(add(max(add(mul(Var(2), Var(2)), Var(3)), Var(1)), atan2(atan2(Var(0), Var(3)), add(Var(1), neg(Var(3))))), mul(ln(add(abs(neg(neg(neg(Const(0.12621832))))), Const(0.001))), mul_add(Var(0), cos(neg(neg(Var(3)))), pow(add(abs(neg(neg(Const(-0.20414245)))), Const(0.001)), Const(-0.5)))))))), Const(0.001)))), pow(add(abs(neg(neg(atan2(Var(2), log10(add(abs(neg(neg(tan(Var(1))))), Const(0.001))))))), Const(0.001)), Const(-0.5))), log2(add(abs(neg(neg(Var(0)))), Const(0.001)))), add(abs(neg(pow(add(abs(neg(pow(add(abs(neg(log2(add(abs(neg(mul(min(add(mul(log10(add(abs(neg(neg(Const(1.2403846)))), Const(0.001))), Var(2)), mul(log10(add(abs(neg(neg(Const(1.2403846)))), Const(0.001))), Var(3))), max(ln(add(abs(neg(neg(Const(0.2514913)))), Const(0.001))), mul(Var(0), Const(0.5072496)))), mul(pow(abs(neg(neg(log10(add(abs(neg(neg(Var(3)))), Const(0.001)))))), Const(0.5)), ln(add(abs(neg(sin(Var(2)))), Const(0.001))))))), Const(0.001))))), Const(0.001)), mul(add(add(mul(div(add(Const(0.61049294), Const(1.354384)), add(abs(neg(mul(Var(0), Var(0)))), Const(0.001))), mul(tan(Const(-1.6860065)), recip(add(abs(neg(Const(-1.7208018))), Const(0.001))))), add(mul(max(Var(3), Var(0)), cos(neg(Var(0)))), mul_add(Var(3), Var(1), Var(0)))), log10(add(abs(neg(Const(-1.1988422))), Const(0.001)))), max(abs(neg(ln(add(abs(neg(pow(add(abs(Var(1)), Const(0.001)), Var(2)))), Const(0.001))))), ln(add(abs(mul(Const(-0.47071946), pow(add(abs(neg(Const(-1.670574))), Const(0.001)), Var(0)))), Const(0.001)))))))), Const(0.001)), Const(-1)))), Const(0.001)))";
        let expr = parse_expr(initial).unwrap();
        let mut subs = Vec::new();
        collect_subexpressions(&expr, &mut subs);
        let mut mismatches = Vec::new();
        for sub in subs {
            let text = expr_to_kernel_code(sub);
            let scalar = eval_expr_scalar(sub, &REWRITE_BUG_INPUTS);
            let (arena, root) = parse_kernel_code_arena(&text).unwrap();
            let jit = benchmark_jit_arena(&arena, root).unwrap().output[0];
            let diff = (scalar - jit).abs();
            if diff > 1e-3 {
                mismatches.push((diff, text, scalar, jit));
            }
        }
        mismatches.sort_by(|a, b| b.0.total_cmp(&a.0));
        for (diff, text, scalar, jit) in mismatches.into_iter().take(20) {
            println!("diff={diff} scalar={scalar} jit={jit} expr={text}");
        }
    }
}
