//! Expression parsing with precedence climbing for the `bcc` C compiler.
//!
//! This module implements a complete C11 expression parser using the precedence
//! climbing algorithm (a form of Pratt parsing) across all 15 C operator
//! precedence levels. It handles:
//!
//! - **Binary operators** — 18 binary operators from comma (level 1) through
//!   multiplicative (level 13), with correct left-to-right or right-to-left
//!   associativity.
//! - **Ternary operator** — `? :` at precedence level 3 (right-to-left).
//! - **Assignment operators** — 11 assignment forms at level 2 (right-to-left).
//! - **Unary prefix operators** — `++`, `--`, `+`, `-`, `~`, `!`, `*`, `&`,
//!   `sizeof`, `_Alignof`, casts, and GCC `__extension__` / `&&label`.
//! - **Postfix operators** — function call `()`, subscript `[]`, member `.`,
//!   arrow `->`, post-increment `++`, post-decrement `--`.
//! - **Primary expressions** — literals, identifiers, parenthesized expressions,
//!   compound literals, `_Generic` selection, and GCC builtins.
//!
//! # Precedence Table (15 Levels)
//!
//! | Level | Operators                  | Associativity   |
//! |-------|----------------------------|-----------------|
//! |   1   | `,`                        | Left-to-right   |
//! |   2   | `= += -= *= /= %= <<= >>= &= ^= \|=` | Right-to-left |
//! |   3   | `?:`                       | Right-to-left   |
//! |   4   | `\|\|`                     | Left-to-right   |
//! |   5   | `&&`                       | Left-to-right   |
//! |   6   | `\|`                       | Left-to-right   |
//! |   7   | `^`                        | Left-to-right   |
//! |   8   | `&`                        | Left-to-right   |
//! |   9   | `== !=`                    | Left-to-right   |
//! |  10   | `< > <= >=`               | Left-to-right   |
//! |  11   | `<< >>`                   | Left-to-right   |
//! |  12   | `+ -`                      | Left-to-right   |
//! |  13   | `* / %`                    | Left-to-right   |
//! |  14   | Unary prefix               | Right-to-left   |
//! |  15   | Postfix                    | Left-to-right   |
//!
//! # Error Recovery
//!
//! On expression parse errors, the parser reports the error through
//! `Parser::error()` and returns `Expression::Error { span }` to allow the
//! caller to continue parsing. This enables the ≥80% error recovery rate
//! specified in AAP §0.2.1.
//!
//! # Integration Points
//!
//! - Called by `statements.rs` for expression statements, conditions, return values.
//! - Called by `declarations.rs` for initializer expressions, bit-field widths.
//! - Called by `gcc_extensions.rs` for asm operand expressions, typeof, and
//!   `__extension__` prefix.
//! - Per AAP §0.5.1 Group 2: "Expression parsing with C operator precedence
//!   and associativity (Pratt parsing or precedence climbing)."
//!
//! # Safety
//!
//! This module contains zero `unsafe` blocks. All operations use safe Rust.

use super::ast::{
    AssignmentOp, BinaryOp, CharPrefix, Expression, FloatSuffix, GenericAssociation,
    Initializer, IntSuffix, NumericBase, StringPrefix, UnaryOp,
};
use super::gcc_extensions;
use super::types;
use super::Parser;
use crate::common::intern::InternId;
use crate::frontend::lexer::token::{TokenKind, TokenValue};

// ===========================================================================
// Precedence Table
// ===========================================================================

/// Returns the binary precedence and right-associativity flag for a token kind.
///
/// The C operator precedence table has 15 levels. This function covers levels
/// 4–13 (the pure binary operators). Levels 1 (comma), 2 (assignment), and 3
/// (ternary) are handled separately in the grammar.
///
/// Returns `Some((precedence, is_right_associative))` for binary operators,
/// or `None` for tokens that are not binary operators.
fn get_binary_precedence(kind: TokenKind) -> Option<(u8, bool)> {
    match kind {
        // Level 4: Logical OR (left-to-right)
        TokenKind::PipePipe => Some((4, false)),

        // Level 5: Logical AND (left-to-right)
        TokenKind::AmpAmp => Some((5, false)),

        // Level 6: Bitwise OR (left-to-right)
        TokenKind::Pipe => Some((6, false)),

        // Level 7: Bitwise XOR (left-to-right)
        TokenKind::Caret => Some((7, false)),

        // Level 8: Bitwise AND (left-to-right)
        TokenKind::Amp => Some((8, false)),

        // Level 9: Equality (left-to-right)
        TokenKind::EqualEqual => Some((9, false)),
        TokenKind::BangEqual => Some((9, false)),

        // Level 10: Relational (left-to-right)
        TokenKind::Less => Some((10, false)),
        TokenKind::Greater => Some((10, false)),
        TokenKind::LessEqual => Some((10, false)),
        TokenKind::GreaterEqual => Some((10, false)),

        // Level 11: Shift (left-to-right)
        TokenKind::LessLess => Some((11, false)),
        TokenKind::GreaterGreater => Some((11, false)),

        // Level 12: Additive (left-to-right)
        TokenKind::Plus => Some((12, false)),
        TokenKind::Minus => Some((12, false)),

        // Level 13: Multiplicative (left-to-right)
        TokenKind::Star => Some((13, false)),
        TokenKind::Slash => Some((13, false)),
        TokenKind::Percent => Some((13, false)),

        _ => None,
    }
}

/// Maps a `TokenKind` binary operator token to the corresponding `BinaryOp`
/// AST variant. Panics if called with a non-binary-operator token kind.
fn token_to_binary_op(kind: TokenKind) -> BinaryOp {
    match kind {
        TokenKind::Plus => BinaryOp::Add,
        TokenKind::Minus => BinaryOp::Sub,
        TokenKind::Star => BinaryOp::Mul,
        TokenKind::Slash => BinaryOp::Div,
        TokenKind::Percent => BinaryOp::Mod,
        TokenKind::Amp => BinaryOp::BitwiseAnd,
        TokenKind::Pipe => BinaryOp::BitwiseOr,
        TokenKind::Caret => BinaryOp::BitwiseXor,
        TokenKind::LessLess => BinaryOp::ShiftLeft,
        TokenKind::GreaterGreater => BinaryOp::ShiftRight,
        TokenKind::AmpAmp => BinaryOp::LogicalAnd,
        TokenKind::PipePipe => BinaryOp::LogicalOr,
        TokenKind::EqualEqual => BinaryOp::Equal,
        TokenKind::BangEqual => BinaryOp::NotEqual,
        TokenKind::Less => BinaryOp::Less,
        TokenKind::Greater => BinaryOp::Greater,
        TokenKind::LessEqual => BinaryOp::LessEqual,
        TokenKind::GreaterEqual => BinaryOp::GreaterEqual,
        _ => unreachable!("token_to_binary_op called with non-binary operator: {:?}", kind),
    }
}

/// Maps a `TokenKind` assignment operator token to the corresponding
/// `AssignmentOp` AST variant. Returns `None` if the token is not an
/// assignment operator.
fn token_to_assignment_op(kind: TokenKind) -> Option<AssignmentOp> {
    match kind {
        TokenKind::Equal => Some(AssignmentOp::Assign),
        TokenKind::PlusEqual => Some(AssignmentOp::AddAssign),
        TokenKind::MinusEqual => Some(AssignmentOp::SubAssign),
        TokenKind::StarEqual => Some(AssignmentOp::MulAssign),
        TokenKind::SlashEqual => Some(AssignmentOp::DivAssign),
        TokenKind::PercentEqual => Some(AssignmentOp::ModAssign),
        TokenKind::AmpEqual => Some(AssignmentOp::AndAssign),
        TokenKind::PipeEqual => Some(AssignmentOp::OrAssign),
        TokenKind::CaretEqual => Some(AssignmentOp::XorAssign),
        TokenKind::LessLessEqual => Some(AssignmentOp::ShlAssign),
        TokenKind::GreaterGreaterEqual => Some(AssignmentOp::ShrAssign),
        _ => None,
    }
}

// ===========================================================================
// Public Entry Points
// ===========================================================================

/// Parses a full expression at comma-expression level (precedence 1).
///
/// This is the top-level expression entry point used for expression statements,
/// for-loop increment clauses, and any context where the comma operator is
/// valid as an expression separator.
///
/// If the parsed expression contains a comma operator, the result is wrapped
/// in `Expression::Comma { exprs, span }`. Otherwise, the single
/// sub-expression is returned directly.
///
/// Grammar:
/// ```text
///   expression := assignment-expression (',' assignment-expression)*
/// ```
pub(super) fn parse_expression(parser: &mut Parser<'_>) -> Expression {
    let start = parser.current_span();
    let first = parse_assignment_expression(parser);

    // Check if there is a comma operator following the first sub-expression.
    if !parser.check(TokenKind::Comma) {
        return first;
    }

    // Collect comma-separated sub-expressions.
    let mut exprs = vec![first];
    while parser.match_token(TokenKind::Comma) {
        let expr = parse_assignment_expression(parser);
        exprs.push(expr);
    }

    let span = parser.span_from(start);
    Expression::Comma { exprs, span }
}

/// Parses an assignment expression (precedence 2, right-to-left).
///
/// This is the most commonly used sub-expression entry point — function
/// arguments, array subscripts, initializer values, ternary sub-expressions,
/// and many other contexts parse at assignment-expression level.
///
/// Grammar:
/// ```text
///   assignment-expression := conditional-expression
///                          | unary-expression assignment-operator assignment-expression
/// ```
///
/// Assignment operators are right-to-left associative: `a = b = c` parses as
/// `a = (b = c)`.
pub(super) fn parse_assignment_expression(parser: &mut Parser<'_>) -> Expression {
    let start = parser.current_span();

    // Parse a conditional expression (ternary) as the left-hand side.
    let lhs = parse_conditional_expression(parser);

    // Check for assignment operators.
    if let Some(op) = token_to_assignment_op(parser.current().kind) {
        parser.advance(); // Consume the assignment operator.

        // Right-to-left: recurse at assignment level for the right-hand side.
        let rhs = parse_assignment_expression(parser);

        let span = parser.span_from(start);
        return Expression::Assignment {
            op,
            target: Box::new(lhs),
            value: Box::new(rhs),
            span,
        };
    }

    lhs
}

// ===========================================================================
// Conditional (Ternary) Expression — Precedence 3
// ===========================================================================

/// Parses a conditional (ternary) expression.
///
/// Grammar:
/// ```text
///   conditional-expression := logical-or-expression
///                           | logical-or-expression '?' expression ':' conditional-expression
/// ```
///
/// The ternary operator is right-to-left associative:
/// `a ? b : c ? d : e` parses as `a ? b : (c ? d : e)`.
fn parse_conditional_expression(parser: &mut Parser<'_>) -> Expression {
    let start = parser.current_span();
    let condition = parse_binary_expression(parser, 4); // Start at logical-OR level.

    if !parser.match_token(TokenKind::Question) {
        return condition;
    }

    // Parse the "then" branch as a full expression (includes comma operator).
    let then_expr = parse_expression(parser);

    // Expect `:`.
    if parser.expect(TokenKind::Colon).is_err() {
        let span = parser.span_from(start);
        return Expression::Error { span };
    }

    // Parse the "else" branch as a conditional expression (right-to-left).
    let else_expr = parse_conditional_expression(parser);

    let span = parser.span_from(start);
    Expression::Ternary {
        condition: Box::new(condition),
        then_expr: Box::new(then_expr),
        else_expr: Box::new(else_expr),
        span,
    }
}

// ===========================================================================
// Precedence Climbing Core — Binary Operators (Levels 4–13)
// ===========================================================================

/// Core precedence climbing loop for binary operators at levels 4–13.
///
/// The algorithm:
/// 1. Parse a unary (prefix) expression as the initial left-hand side.
/// 2. While the current token is a binary operator with precedence ≥
///    `min_precedence`:
///    a. Record the operator token and its precedence.
///    b. Advance past the operator.
///    c. Recurse: for left-to-right operators, recurse with `precedence + 1`;
///       for right-to-left operators, recurse with `precedence`.
///    d. Build a `Binary` AST node with the operator and operands.
///
/// This naturally handles left-to-right and right-to-left associativity and
/// produces a correctly structured AST for all precedence levels.
fn parse_binary_expression(parser: &mut Parser<'_>, min_precedence: u8) -> Expression {
    let start = parser.current_span();
    let mut left = parse_unary_expression(parser);

    loop {
        let kind = parser.current().kind;

        // Look up the precedence and associativity of the current token.
        let (prec, is_right_assoc) = match get_binary_precedence(kind) {
            Some(pair) => pair,
            None => break, // Not a binary operator — stop climbing.
        };

        // If the operator's precedence is below our minimum, stop.
        if prec < min_precedence {
            break;
        }

        // Consume the operator token.
        let op_kind = parser.current().kind;
        parser.advance();

        // Convert the token to an AST binary-op enum variant.
        let op = token_to_binary_op(op_kind);

        // Recurse for the right-hand operand.
        // Left-to-right: use `prec + 1` so equal-precedence ops bind to the left.
        // Right-to-left: use `prec` so equal-precedence ops bind to the right.
        let next_min = if is_right_assoc { prec } else { prec + 1 };
        let right = parse_binary_expression(parser, next_min);

        let span = parser.span_from(start);
        left = Expression::Binary {
            op,
            left: Box::new(left),
            right: Box::new(right),
            span,
        };
    }

    left
}

// ===========================================================================
// Unary Prefix Expressions — Precedence 14
// ===========================================================================

/// Parses a unary prefix expression (precedence 14, right-to-left).
///
/// Handles:
/// - `++expr`, `--expr` (pre-increment/decrement)
/// - `+expr`, `-expr` (unary plus/minus)
/// - `~expr` (bitwise NOT)
/// - `!expr` (logical NOT)
/// - `*expr` (dereference)
/// - `&expr` (address-of)
/// - `sizeof expr`, `sizeof(type)`
/// - `_Alignof(type)`
/// - `(type)expr` (cast)
/// - GCC `__extension__ expr`
/// - GCC `&&label` (label address)
///
/// Falls through to postfix expression parsing if no unary prefix is found.
fn parse_unary_expression(parser: &mut Parser<'_>) -> Expression {
    let start = parser.current_span();

    match parser.current().kind {
        // Pre-increment: ++expr
        TokenKind::PlusPlus => {
            parser.advance();
            let operand = parse_unary_expression(parser);
            let span = parser.span_from(start);
            Expression::UnaryPrefix {
                op: UnaryOp::PreIncrement,
                operand: Box::new(operand),
                span,
            }
        }

        // Pre-decrement: --expr
        TokenKind::MinusMinus => {
            parser.advance();
            let operand = parse_unary_expression(parser);
            let span = parser.span_from(start);
            Expression::UnaryPrefix {
                op: UnaryOp::PreDecrement,
                operand: Box::new(operand),
                span,
            }
        }

        // Unary plus: +expr
        TokenKind::Plus => {
            parser.advance();
            let operand = parse_cast_expression(parser);
            let span = parser.span_from(start);
            Expression::UnaryPrefix {
                op: UnaryOp::Plus,
                operand: Box::new(operand),
                span,
            }
        }

        // Unary minus (negation): -expr
        TokenKind::Minus => {
            parser.advance();
            let operand = parse_cast_expression(parser);
            let span = parser.span_from(start);
            Expression::UnaryPrefix {
                op: UnaryOp::Negate,
                operand: Box::new(operand),
                span,
            }
        }

        // Bitwise NOT: ~expr
        TokenKind::Tilde => {
            parser.advance();
            let operand = parse_cast_expression(parser);
            let span = parser.span_from(start);
            Expression::UnaryPrefix {
                op: UnaryOp::BitwiseNot,
                operand: Box::new(operand),
                span,
            }
        }

        // Logical NOT: !expr
        TokenKind::Bang => {
            parser.advance();
            let operand = parse_cast_expression(parser);
            let span = parser.span_from(start);
            Expression::UnaryPrefix {
                op: UnaryOp::LogicalNot,
                operand: Box::new(operand),
                span,
            }
        }

        // Dereference: *expr
        TokenKind::Star => {
            parser.advance();
            let operand = parse_cast_expression(parser);
            let span = parser.span_from(start);
            Expression::UnaryPrefix {
                op: UnaryOp::Dereference,
                operand: Box::new(operand),
                span,
            }
        }

        // Address-of: &expr
        TokenKind::Amp => {
            parser.advance();
            let operand = parse_cast_expression(parser);
            let span = parser.span_from(start);
            Expression::UnaryPrefix {
                op: UnaryOp::AddressOf,
                operand: Box::new(operand),
                span,
            }
        }

        // sizeof expression or sizeof(type)
        TokenKind::Sizeof => parse_sizeof(parser),

        // _Alignof(type)
        TokenKind::Alignof => parse_alignof(parser),

        // GCC __extension__ prefix
        TokenKind::GccExtension => gcc_extensions::parse_extension_prefix(parser),

        // GCC &&label address — note: `AmpAmp` is a single token `&&`
        // In expression context, `&&` followed by an identifier is a label
        // address. In binary context, `&&` is logical AND, but that case is
        // handled by the binary expression parser (level 5). The unary prefix
        // parser only sees `&&` when it appears at the start of a unary
        // expression, which is the label-address case.
        TokenKind::AmpAmp => {
            // Use lookahead(1) to check if followed by identifier for label
            // address. This is equivalent to peek() but uses the explicit
            // n-token lookahead API for clarity on the look-ahead distance.
            if parser.lookahead(1).kind == TokenKind::Identifier {
                gcc_extensions::parse_label_address(parser)
            } else {
                // Fall through to postfix — will likely cause an error.
                parse_postfix_expression(parser)
            }
        }

        // No unary prefix — fall through to postfix expression.
        // Per C grammar: unary-expression := postfix-expression | prefix-op cast-expression
        _ => parse_postfix_expression(parser),
    }
}

// ===========================================================================
// Cast Expression
// ===========================================================================

/// Parses a cast expression: `(type_name) expr` or falls through to postfix.
///
/// This is one of the trickiest disambiguations in C parsing:
/// - `(int)x` — cast expression
/// - `(x)` — parenthesized expression (if `x` is a variable)
/// - `(x)(y)` — function call if `x` is a variable, or cast if `x` is typedef
/// - `(struct Point){.x=1}` — compound literal
/// - `({ ... })` — GCC statement expression
///
/// Strategy: When we see `(`, peek at the next token. If the next token starts
/// a type specifier (type keywords or a known typedef name), consume `(`, parse
/// the type name, consume `)`, then check for `{` (compound literal) or parse
/// cast operand. If the next token is `{`, delegate to statement expression
/// handler. Otherwise, fall through to postfix expression (which handles
/// parenthesized expressions in primary).
fn parse_cast_expression(parser: &mut Parser<'_>) -> Expression {
    if parser.check(TokenKind::LeftParen) {
        // GCC statement expression: `({ ... })` — delegate to postfix/primary.
        if parser.peek().kind == TokenKind::LeftBrace {
            return parse_postfix_expression(parser);
        }

        // Check if the token after `(` starts a type name.
        // Disambiguation: If the identifier is a typedef name but is followed
        // by `=`, `++`, `--`, `[`, `.`, `->`, or a binary operator, then it's
        // a VARIABLE in a parenthesized expression, NOT a cast. Example:
        //   typedef struct list { int len; } list;
        //   struct list *list;
        //   if ((list = malloc(...)) == NULL)  // NOT a cast, it's assignment
        if is_type_start_token(parser, parser.peek())
            && !is_typedef_var_use(parser)
        {
            let start = parser.current_span();
            parser.advance(); // Consume `(`

            // Parse the type name.
            let type_name = types::parse_type_name(parser);

            // Expect `)`.
            if parser.expect(TokenKind::RightParen).is_err() {
                let span = parser.span_from(start);
                return Expression::Error { span };
            }

            // Check for compound literal: `(type){init}`
            if parser.check(TokenKind::LeftBrace) {
                let initializer = parse_compound_initializer(parser);
                let span = parser.span_from(start);
                return Expression::CompoundLiteral {
                    type_name: Box::new(type_name),
                    initializer,
                    span,
                };
            }

            // Regular cast: parse the operand as a cast-expression
            // (right-to-left, so cast of cast works: `(int)(float)x`).
            let operand = parse_cast_expression(parser);
            let span = parser.span_from(start);
            return Expression::Cast {
                type_name: Box::new(type_name),
                operand: Box::new(operand),
                span,
            };
        }
    }

    // Not a cast — fall through to unary-expression (which includes postfix).
    // Per C grammar: cast-expression := unary-expression | '(' type-name ')' cast-expression
    parse_unary_expression(parser)
}

// ===========================================================================
// Sizeof and _Alignof
// ===========================================================================

/// Parses a `sizeof` expression.
///
/// Two forms:
/// - `sizeof(type_name)` → `Expression::SizeofType`
/// - `sizeof expr` → `Expression::SizeofExpr`
///
/// Disambiguation: If `(` follows `sizeof` and the next token starts a type
/// name, we parse as `sizeof(type)`. Otherwise, we parse as `sizeof expr`
/// (the `(` may be part of a parenthesized expression).
fn parse_sizeof(parser: &mut Parser<'_>) -> Expression {
    let start = parser.current_span();
    parser.advance(); // Consume `sizeof`

    // Check for sizeof(type_name): `sizeof` `(` type-start-token
    if parser.check(TokenKind::LeftParen) && is_type_start_token(parser, parser.peek()) {
        parser.advance(); // Consume `(`

        let type_name = types::parse_type_name(parser);

        if parser.expect(TokenKind::RightParen).is_err() {
            let span = parser.span_from(start);
            return Expression::Error { span };
        }

        let span = parser.span_from(start);
        return Expression::SizeofType {
            type_name: Box::new(type_name),
            span,
        };
    }

    // sizeof expr (unary expression, not assignment expression).
    let expr = parse_unary_expression(parser);
    let span = parser.span_from(start);
    Expression::SizeofExpr {
        expr: Box::new(expr),
        span,
    }
}

/// Parses an `_Alignof` expression.
///
/// Grammar: `_Alignof(type_name)`
///
/// Unlike `sizeof`, `_Alignof` always requires parenthesized type name
/// (no expression form in C11).
fn parse_alignof(parser: &mut Parser<'_>) -> Expression {
    let start = parser.current_span();
    parser.advance(); // Consume `_Alignof`

    if parser.expect(TokenKind::LeftParen).is_err() {
        let span = parser.span_from(start);
        return Expression::Error { span };
    }

    let type_name = types::parse_type_name(parser);

    if parser.expect(TokenKind::RightParen).is_err() {
        let span = parser.span_from(start);
        return Expression::Error { span };
    }

    let span = parser.span_from(start);
    Expression::Alignof {
        type_name: Box::new(type_name),
        span,
    }
}

/// Returns `true` if the given token starts a type name.
///
/// Checks for type specifier keywords, type qualifier keywords, and typedef
/// names (identifiers that were previously declared as typedef names).
pub(super) fn is_type_start_token(parser: &Parser<'_>, token: &crate::frontend::lexer::token::Token) -> bool {
    let kind = token.kind;

    // Type specifier keywords (int, void, struct, union, enum, etc.)
    if kind.is_type_specifier() {
        return true;
    }

    // Type qualifier keywords (const, volatile, restrict, _Atomic)
    if kind.is_type_qualifier() {
        return true;
    }

    // Storage class keywords (static, extern, etc.) — valid in some type contexts.
    if kind.is_storage_class() {
        return true;
    }

    // Function specifiers.
    if matches!(kind, TokenKind::Inline | TokenKind::Noreturn) {
        return true;
    }

    // GCC __attribute__ and __extension__ can precede type specifiers.
    if matches!(kind, TokenKind::GccAttribute | TokenKind::GccExtension) {
        return true;
    }

    // __builtin_va_list is a type.
    if kind == TokenKind::BuiltinVaList {
        return true;
    }

    // Typeof is a type specifier.
    if kind == TokenKind::Typeof {
        return true;
    }

    // Check for typedef names.
    if kind == TokenKind::Identifier {
        if let TokenValue::Identifier(id) = token.value {
            return parser.is_typedef_name(id);
        }
    }

    false
}

/// Checks whether `( identifier ...` is actually a variable usage inside
/// parentheses rather than a cast expression. This handles the ambiguity
/// where a typedef name is also used as a local variable name (common in
/// code like `typedef struct list { ... } list; struct list *list;`).
///
/// If `peek()` (the token after `(`) is an Identifier that is a typedef
/// name, and the token AFTER the identifier is an operator like `=`, `++`,
/// `--`, `,`, `[`, `->`, `.`, etc., then this is clearly a variable in a
/// parenthesized expression, NOT a cast.
fn is_typedef_var_use(parser: &Parser<'_>) -> bool {
    let next = parser.peek();
    // Only applies to typedef-name identifiers.
    if next.kind != TokenKind::Identifier {
        return false;
    }
    let is_typedef = if let TokenValue::Identifier(id) = next.value {
        parser.is_typedef_name(id)
    } else {
        return false;
    };
    if !is_typedef {
        return false;
    }

    // Check the token after the identifier (lookahead(2): token at pos+2).
    let after = parser.lookahead(2);

    // Direct operator after typedef name: `(list = ...`, `(list->...`, etc.
    if matches!(
        after.kind,
        TokenKind::Equal
            | TokenKind::PlusEqual
            | TokenKind::MinusEqual
            | TokenKind::StarEqual
            | TokenKind::SlashEqual
            | TokenKind::PercentEqual
            | TokenKind::AmpEqual
            | TokenKind::PipeEqual
            | TokenKind::CaretEqual
            | TokenKind::LessLessEqual
            | TokenKind::GreaterGreaterEqual
            | TokenKind::PlusPlus
            | TokenKind::MinusMinus
            | TokenKind::LeftBracket
            | TokenKind::Arrow
            | TokenKind::Dot
    ) {
        return true;
    }

    // Pattern: `(typedef_name)` followed by a postfix operator.
    // Example: `((list)->len)` — `(list)` is parenthesized variable, not a cast,
    // because a cast operand can't start with `->` or `.`.
    if after.kind == TokenKind::RightParen {
        let after_paren = parser.lookahead(3);
        if matches!(
            after_paren.kind,
            TokenKind::Arrow
                | TokenKind::Dot
                | TokenKind::LeftBracket
                | TokenKind::PlusPlus
                | TokenKind::MinusMinus
        ) {
            return true;
        }
    }

    false
}

// ===========================================================================
// Postfix Expressions — Precedence 15
// ===========================================================================

/// Parses a postfix expression (precedence 15, left-to-right).
///
/// Starts with a primary expression and then loops over postfix operators:
/// - `(args)` — function call
/// - `[index]` — array subscript
/// - `.member` — direct member access
/// - `->member` — pointer member access
/// - `++` — post-increment
/// - `--` — post-decrement
fn parse_postfix_expression(parser: &mut Parser<'_>) -> Expression {
    let start = parser.current_span();
    let mut expr = parse_primary_expression(parser);

    loop {
        match parser.current().kind {
            // Function call: expr(args)
            TokenKind::LeftParen => {
                parser.advance(); // Consume `(`
                let args = parse_argument_list(parser);
                if parser.expect(TokenKind::RightParen).is_err() {
                    // Recovery: skip to `)` or statement boundary.
                    recover_to_paren_close(parser);
                }
                let span = parser.span_from(start);
                expr = Expression::Call {
                    callee: Box::new(expr),
                    args,
                    span,
                };
            }

            // Array subscript: expr[index]
            TokenKind::LeftBracket => {
                parser.advance(); // Consume `[`
                let index = parse_expression(parser);
                if parser.expect(TokenKind::RightBracket).is_err() {
                    // Recovery: skip to `]` or statement boundary.
                    recover_to_bracket_close(parser);
                }
                let span = parser.span_from(start);
                expr = Expression::Subscript {
                    array: Box::new(expr),
                    index: Box::new(index),
                    span,
                };
            }

            // Direct member access: expr.member
            TokenKind::Dot => {
                parser.advance(); // Consume `.`
                let member = parse_member_name(parser);
                let span = parser.span_from(start);
                expr = Expression::MemberAccess {
                    object: Box::new(expr),
                    member,
                    span,
                };
            }

            // Pointer member access: expr->member
            TokenKind::Arrow => {
                parser.advance(); // Consume `->`
                let member = parse_member_name(parser);
                let span = parser.span_from(start);
                expr = Expression::ArrowAccess {
                    pointer: Box::new(expr),
                    member,
                    span,
                };
            }

            // Post-increment: expr++
            TokenKind::PlusPlus => {
                parser.advance();
                let span = parser.span_from(start);
                expr = Expression::PostIncrement {
                    operand: Box::new(expr),
                    span,
                };
            }

            // Post-decrement: expr--
            TokenKind::MinusMinus => {
                parser.advance();
                let span = parser.span_from(start);
                expr = Expression::PostDecrement {
                    operand: Box::new(expr),
                    span,
                };
            }

            _ => break,
        }
    }

    expr
}

/// Parses a comma-separated list of assignment expressions for function call
/// arguments.
///
/// Note: arguments are parsed at assignment-expression level, NOT full
/// expression level — the comma here is an argument separator, not the
/// comma operator.
///
/// Returns an empty `Vec` for `f()` (no arguments).
fn parse_argument_list(parser: &mut Parser<'_>) -> Vec<Expression> {
    let mut args = Vec::new();

    // Handle empty argument list.
    if parser.check(TokenKind::RightParen) {
        return args;
    }

    // Parse first argument.
    args.push(parse_assignment_expression(parser));

    // Parse remaining comma-separated arguments.
    while parser.match_token(TokenKind::Comma) {
        args.push(parse_assignment_expression(parser));
    }

    args
}

/// Parses a member name after `.` or `->`.
///
/// Expects an identifier token. On error, reports the issue and returns a
/// dummy `InternId`.
fn parse_member_name(parser: &mut Parser<'_>) -> InternId {
    if let TokenKind::Identifier = parser.current().kind {
        if let TokenValue::Identifier(id) = parser.current().value {
            parser.advance();
            return id;
        }
    }

    // Also accept keywords as member names (struct members can shadow keywords).
    if parser.current().kind.is_keyword() {
        // Try to look up the keyword string as an interned identifier.
        let kw_str = parser.current().kind.as_str();
        if let Some(id) = parser.lookup_interned(kw_str) {
            parser.advance();
            return id;
        }
    }

    // Report the error at the position of the `.` or `->` token that was
    // already consumed by the caller, so the diagnostic points at the
    // access operator rather than the unexpected following token.
    let access_span = parser.previous_span();
    parser.error_at(access_span, "expected member name after '.' or '->'");
    InternId::from_raw(0)
}

// ===========================================================================
// Primary Expressions
// ===========================================================================

/// Parses a primary expression — the atomic building blocks of all expressions.
///
/// Handles:
/// - Integer literals
/// - Floating-point literals
/// - String literals (with adjacent string concatenation)
/// - Character literals
/// - Identifiers
/// - Parenthesized expressions, casts, compound literals, statement expressions
/// - `_Generic` selection
/// - GCC builtin calls
fn parse_primary_expression(parser: &mut Parser<'_>) -> Expression {
    let start = parser.current_span();

    match parser.current().kind {
        // Integer literal
        TokenKind::IntegerLiteral => parse_integer_literal(parser),

        // Floating-point literal
        TokenKind::FloatLiteral => parse_float_literal(parser),

        // String literal (with adjacent concatenation)
        TokenKind::StringLiteral => parse_string_literal(parser),

        // Character literal
        TokenKind::CharLiteral => parse_char_literal(parser),

        // Identifier
        TokenKind::Identifier => {
            if let TokenValue::Identifier(id) = parser.current().value {
                parser.advance();
                let span = parser.span_from(start);
                Expression::Identifier { name: id, span }
            } else {
                parser.advance();
                Expression::Error {
                    span: parser.span_from(start),
                }
            }
        }

        // Parenthesized expression, cast, compound literal, or statement expr.
        TokenKind::LeftParen => parse_paren_expression(parser),

        // C11 _Generic selection expression
        TokenKind::Generic => parse_generic_selection(parser),

        // GCC builtin calls with special syntax.
        TokenKind::BuiltinVaArg
        | TokenKind::BuiltinOffsetof
        | TokenKind::BuiltinVaStart
        | TokenKind::BuiltinVaEnd
        | TokenKind::BuiltinVaCopy => {
            let builtin = parser.current().kind;
            gcc_extensions::parse_builtin_call(parser, builtin)
        }

        // Other GCC builtins parsed as regular function calls.
        kind if is_regular_builtin(kind) => {
            // Parse as a regular identifier + function call.
            // The builtin keyword is used as if it were an identifier name.
            let kw_str = kind.as_str();
            let name = parser.lookup_interned(kw_str).unwrap_or(InternId::from_raw(0));
            parser.advance();
            let span = parser.span_from(start);
            Expression::Identifier { name, span }
        }

        // __builtin_va_list used as a type — treat as identifier in expr context
        TokenKind::BuiltinVaList => {
            let kw_str = parser.current().kind.as_str();
            let name = parser.lookup_interned(kw_str).unwrap_or(InternId::from_raw(0));
            parser.advance();
            let span = parser.span_from(start);
            Expression::Identifier { name, span }
        }

        // End of input or unexpected token.
        _ => {
            if parser.is_at_end() {
                parser.error("expected expression, found end of input");
            } else {
                // Use the dedicated unexpected-token diagnostic which formats
                // an "expected X, found Y" message pointing at the exact
                // source location.
                parser.unexpected_token("expression");
            }
            let span = parser.span_from(start);
            Expression::Error { span }
        }
    }
}

/// Returns `true` if the given `TokenKind` is a GCC builtin that should be
/// parsed as a regular function call (not one with special syntax).
fn is_regular_builtin(kind: TokenKind) -> bool {
    matches!(
        kind,
        TokenKind::BuiltinExpect
            | TokenKind::BuiltinUnreachable
            | TokenKind::BuiltinConstantP
            | TokenKind::BuiltinChooseExpr
            | TokenKind::BuiltinBswap16
            | TokenKind::BuiltinBswap32
            | TokenKind::BuiltinBswap64
            | TokenKind::BuiltinClz
            | TokenKind::BuiltinCtz
            | TokenKind::BuiltinPopcount
            | TokenKind::BuiltinFfs
            | TokenKind::BuiltinAbs
            | TokenKind::BuiltinFabsf
            | TokenKind::BuiltinFabs
            | TokenKind::BuiltinInf
            | TokenKind::BuiltinInff
            | TokenKind::BuiltinHugeVal
            | TokenKind::BuiltinHugeValf
            | TokenKind::BuiltinNan
            | TokenKind::BuiltinNanf
            | TokenKind::BuiltinTrap
            | TokenKind::BuiltinAlloca
            | TokenKind::BuiltinMemcpy
            | TokenKind::BuiltinMemset
            | TokenKind::BuiltinStrlen
            | TokenKind::BuiltinFrameAddress
            | TokenKind::BuiltinTypesCompatibleP
    )
}

// ===========================================================================
// Parenthesized / Cast / Compound Literal / Statement Expression
// ===========================================================================

/// Parses an expression that begins with `(` in a primary expression context.
///
/// When we reach here from `parse_primary_expression`, the `parse_cast_expression`
/// layer has already checked and rejected the cast/compound-literal interpretation.
/// Therefore, this function handles:
/// - `({ ... })` — GCC statement expression
/// - `(expr)` — parenthesized expression
///
/// If reached from a context where cast wasn't checked first (shouldn't happen
/// in normal flow), this also handles cast/compound literal as a safety net.
fn parse_paren_expression(parser: &mut Parser<'_>) -> Expression {
    let start = parser.current_span();

    // Check expression nesting depth before recursing into a parenthesized
    // expression. Each paren nesting level adds ~8 stack frames through the
    // expression parsing chain, so we use a separate, lower limit than
    // statement nesting to prevent stack overflow.
    if parser.enter_expr_nesting().is_err() {
        // Depth limit exceeded — an error diagnostic has already been emitted.
        // Skip to recovery point and return an Error expression.
        parser.synchronize();
        let span = parser.span_from(start);
        return Expression::Error { span };
    }

    // Check for GCC statement expression: `({` ... `})`
    if parser.check(TokenKind::LeftParen) && parser.peek().kind == TokenKind::LeftBrace {
        let result = gcc_extensions::parse_statement_expression(parser);
        parser.exit_expr_nesting();
        return result;
    }

    // Consume `(`
    parser.advance();

    // Safety net: if this is a type name, handle cast/compound literal.
    // This path is reached when `parse_paren_expression` is called from
    // a context that didn't go through `parse_cast_expression` first.
    //
    // Special handling for `__extension__`: it matches `is_type_specifier_start`
    // because it CAN precede type specifiers (e.g., `(__extension__ __int128)`),
    // but it can ALSO precede expressions (e.g., `(__extension__ 42)`). We
    // disambiguate by looking past the `__extension__` token(s) to see if what
    // follows is actually a type specifier.
    let is_type_start = if parser.current().kind == TokenKind::GccExtension {
        // Look ahead past __extension__ token(s) to check the real token
        let mut skip = 0;
        while parser.lookahead(skip).kind == TokenKind::GccExtension {
            skip += 1;
        }
        let after_ext = parser.lookahead(skip);
        after_ext.kind.is_type_specifier()
            || after_ext.kind.is_type_qualifier()
            || after_ext.kind.is_storage_class()
            || matches!(after_ext.kind, TokenKind::Inline | TokenKind::Noreturn | TokenKind::GccAttribute | TokenKind::BuiltinVaList)
            || (after_ext.kind == TokenKind::Identifier && {
                if let super::super::lexer::token::TokenValue::Identifier(id) = after_ext.value {
                    parser.is_typedef_name(id)
                } else {
                    false
                }
            })
    } else {
        types::is_type_specifier_start(parser)
    };
    // Disambiguation (Fix 117): If the current token is a typedef identifier
    // but the NEXT token is an assignment operator, `++`, `--`, `[`, `->`,
    // or `.`, this is actually a variable use in a parenthesized expression,
    // NOT a cast. Example: `typedef struct list list; struct list *list;`
    // then `(list = malloc(...))` is a parenthesized assignment, not a cast.
    //
    // Also handles `(list)->member` pattern: if the typedef name is followed
    // by `)` and then a postfix operator (`->`, `.`, `[`, `++`, `--`), the
    // typedef name is a variable expression, not a type in a cast.
    let is_type_start = if is_type_start {
        if parser.current().kind == TokenKind::Identifier {
            let next_kind = parser.peek().kind;
            if matches!(
                next_kind,
                TokenKind::Equal
                    | TokenKind::PlusEqual
                    | TokenKind::MinusEqual
                    | TokenKind::StarEqual
                    | TokenKind::SlashEqual
                    | TokenKind::PercentEqual
                    | TokenKind::AmpEqual
                    | TokenKind::PipeEqual
                    | TokenKind::CaretEqual
                    | TokenKind::LessLessEqual
                    | TokenKind::GreaterGreaterEqual
                    | TokenKind::PlusPlus
                    | TokenKind::MinusMinus
                    | TokenKind::LeftBracket
                    | TokenKind::Arrow
                    | TokenKind::Dot
            ) {
                false
            } else if next_kind == TokenKind::RightParen {
                // `(typedef_name)` — check what follows the closing paren.
                // If it's a postfix operator, this is `(variable)` not a cast.
                let after_paren = parser.lookahead(2);
                !matches!(
                    after_paren.kind,
                    TokenKind::Arrow
                        | TokenKind::Dot
                        | TokenKind::LeftBracket
                        | TokenKind::PlusPlus
                        | TokenKind::MinusMinus
                )
            } else {
                true
            }
        } else {
            true
        }
    } else {
        false
    };
    if is_type_start {
        let type_name = types::parse_type_name(parser);

        if parser.expect(TokenKind::RightParen).is_err() {
            parser.exit_expr_nesting();
            let span = parser.span_from(start);
            return Expression::Error { span };
        }

        // Check for compound literal: `(type){init}`
        if parser.check(TokenKind::LeftBrace) {
            let initializer = parse_compound_initializer(parser);
            parser.exit_expr_nesting();
            let span = parser.span_from(start);
            return Expression::CompoundLiteral {
                type_name: Box::new(type_name),
                initializer,
                span,
            };
        }

        // Cast expression: `(type)expr`
        let operand = parse_cast_expression(parser);
        parser.exit_expr_nesting();
        let span = parser.span_from(start);
        return Expression::Cast {
            type_name: Box::new(type_name),
            operand: Box::new(operand),
            span,
        };
    }

    // Regular parenthesized expression.
    let inner = parse_expression(parser);

    if parser.expect(TokenKind::RightParen).is_err() {
        recover_to_paren_close(parser);
    }

    // Decrement the expression nesting depth counter on exit.
    parser.exit_expr_nesting();

    let span = parser.span_from(start);
    Expression::Paren {
        inner: Box::new(inner),
        span,
    }
}

// ===========================================================================
// _Generic Selection Expression (C11)
// ===========================================================================

/// Parses a C11 `_Generic` selection expression.
///
/// Grammar:
/// ```text
///   _Generic ( assignment-expression , generic-assoc-list )
///   generic-assoc-list := generic-association
///                       | generic-assoc-list , generic-association
///   generic-association := type-name : assignment-expression
///                        | default : assignment-expression
/// ```
fn parse_generic_selection(parser: &mut Parser<'_>) -> Expression {
    let start = parser.current_span();
    parser.advance(); // Consume `_Generic`

    if parser.expect(TokenKind::LeftParen).is_err() {
        return Expression::Error {
            span: parser.span_from(start),
        };
    }

    // Parse the controlling expression.
    let controlling = parse_assignment_expression(parser);

    if parser.expect(TokenKind::Comma).is_err() {
        return Expression::Error {
            span: parser.span_from(start),
        };
    }

    // Parse the generic association list.
    let mut associations = Vec::new();
    loop {
        let assoc_start = parser.current_span();

        if parser.check(TokenKind::Default) {
            // `default : assignment-expression`
            parser.advance(); // Consume `default`
            if parser.expect(TokenKind::Colon).is_err() {
                break;
            }
            let expr = parse_assignment_expression(parser);
            let assoc_span = parser.span_from(assoc_start);
            associations.push(GenericAssociation::Default {
                expr,
                span: assoc_span,
            });
        } else {
            // `type-name : assignment-expression`
            let type_name = types::parse_type_name(parser);
            if parser.expect(TokenKind::Colon).is_err() {
                break;
            }
            let expr = parse_assignment_expression(parser);
            let assoc_span = parser.span_from(assoc_start);
            associations.push(GenericAssociation::Type {
                type_name,
                expr,
                span: assoc_span,
            });
        }

        if !parser.match_token(TokenKind::Comma) {
            break;
        }
    }

    if parser.expect(TokenKind::RightParen).is_err() {
        // Use the parser's built-in synchronize() to skip to a statement
        // boundary, then attempt to consume a trailing `)` if present.
        parser.synchronize();
        let _ = parser.match_token(TokenKind::RightParen);
    }

    let span = parser.span_from(start);
    Expression::Generic {
        controlling: Box::new(controlling),
        associations,
        span,
    }
}

// ===========================================================================
// Literal Parsing
// ===========================================================================

/// Parses an integer literal token into an AST node.
fn parse_integer_literal(parser: &mut Parser<'_>) -> Expression {
    let start = parser.current_span();

    if let TokenValue::Integer {
        value,
        suffix,
        base,
    } = parser.current().value
    {
        parser.advance();
        let span = parser.span_from(start);

        // Convert token-level suffix/base enums to AST-level enums.
        let ast_suffix = convert_int_suffix(suffix);
        let ast_base = convert_numeric_base(base);

        Expression::IntegerLiteral {
            value,
            suffix: ast_suffix,
            base: ast_base,
            span,
        }
    } else {
        parser.advance();
        Expression::Error {
            span: parser.span_from(start),
        }
    }
}

/// Parses a floating-point literal token into an AST node.
fn parse_float_literal(parser: &mut Parser<'_>) -> Expression {
    let start = parser.current_span();

    if let TokenValue::Float { value, suffix } = parser.current().value {
        parser.advance();
        let span = parser.span_from(start);

        let ast_suffix = convert_float_suffix(suffix);

        Expression::FloatLiteral {
            value,
            suffix: ast_suffix,
            span,
        }
    } else {
        parser.advance();
        Expression::Error {
            span: parser.span_from(start),
        }
    }
}

/// Parses a string literal with adjacent string concatenation.
///
/// C language feature: `"hello" " " "world"` → single string `"hello world"`.
/// Adjacent string literals are concatenated at parse time.
fn parse_string_literal(parser: &mut Parser<'_>) -> Expression {
    let start = parser.current_span();

    let mut value = String::new();
    let prefix = StringPrefix::None;

    // Consume the first string literal.
    if let TokenValue::Str(ref s) = parser.current().value {
        value.push_str(s);
    }
    parser.advance();

    // Concatenate adjacent string literals (C standard, not just preprocessor).
    while parser.check(TokenKind::StringLiteral) {
        if let TokenValue::Str(ref s) = parser.current().value {
            value.push_str(s);
        }
        parser.advance();
    }

    let span = parser.span_from(start);
    Expression::StringLiteral {
        value,
        prefix,
        span,
    }
}

/// Parses a character literal token into an AST node.
fn parse_char_literal(parser: &mut Parser<'_>) -> Expression {
    let start = parser.current_span();

    if let TokenValue::Char(ch) = parser.current().value {
        parser.advance();
        let span = parser.span_from(start);
        Expression::CharLiteral {
            value: ch,
            prefix: CharPrefix::None,
            span,
        }
    } else {
        parser.advance();
        Expression::Error {
            span: parser.span_from(start),
        }
    }
}

// ===========================================================================
// Compound Initializer (for compound literals)
// ===========================================================================

/// Parses a brace-enclosed initializer list for compound literals.
///
/// Grammar:
/// ```text
///   compound-initializer := '{' initializer-list? ','? '}'
///   initializer-list := designation? initializer
///                     | initializer-list ',' designation? initializer
///   designation := designator-list '='
///   designator := '[' constant-expression ']' | '.' IDENTIFIER
/// ```
fn parse_compound_initializer(parser: &mut Parser<'_>) -> Initializer {
    let start = parser.current_span();

    if parser.expect(TokenKind::LeftBrace).is_err() {
        return Initializer::Expression(Box::new(Expression::Error {
            span: parser.span_from(start),
        }));
    }

    let mut items = Vec::new();

    // Handle empty initializer: `{}`
    if parser.check(TokenKind::RightBrace) {
        parser.advance();
        let span = parser.span_from(start);
        return Initializer::Compound { items, span };
    }

    loop {
        let item_start = parser.current_span();
        let mut designators = Vec::new();

        // Parse designators: `.field`, `[index]`, or GCC range `[low ... high]`
        while parser.check(TokenKind::Dot) || parser.check(TokenKind::LeftBracket) {
            if parser.match_token(TokenKind::Dot) {
                // Field designator: `.field`
                if let TokenKind::Identifier = parser.current().kind {
                    if let TokenValue::Identifier(id) = parser.current().value {
                        parser.advance();
                        designators.push(super::ast::Designator::Field(id));
                    }
                } else {
                    parser.error("expected field name in designator");
                    break;
                }
            } else if parser.match_token(TokenKind::LeftBracket) {
                // Array index designator: `[expr]`
                let index_expr = parse_assignment_expression(parser);
                // Check for GCC range: `[low ... high]`
                if parser.check(TokenKind::Ellipsis) {
                    parser.advance(); // Consume `...`
                    let high_expr = parse_assignment_expression(parser);
                    designators.push(super::ast::Designator::Range(
                        Box::new(index_expr),
                        Box::new(high_expr),
                    ));
                } else {
                    designators.push(super::ast::Designator::Index(Box::new(index_expr)));
                }
                if parser.expect(TokenKind::RightBracket).is_err() {
                    break;
                }
            }
        }

        // After designators, expect `=` to separate designator from value.
        if !designators.is_empty() {
            if !parser.match_token(TokenKind::Equal) {
                parser.error("expected '=' after designator");
            }
        }

        // Parse the initializer value (can be nested compound or expression).
        let initializer = if parser.check(TokenKind::LeftBrace) {
            parse_compound_initializer(parser)
        } else {
            Initializer::Expression(Box::new(parse_assignment_expression(parser)))
        };

        let item_span = parser.span_from(item_start);
        items.push(super::ast::DesignatedInitializer {
            designators,
            initializer,
            span: item_span,
        });

        // Check for trailing comma or end of list.
        if !parser.match_token(TokenKind::Comma) {
            break;
        }

        // Allow trailing comma before `}`.
        if parser.check(TokenKind::RightBrace) {
            break;
        }
    }

    if parser.expect(TokenKind::RightBrace).is_err() {
        // Recovery: skip to `}` or statement boundary.
        while !parser.is_at_end()
            && !parser.check(TokenKind::RightBrace)
            && !parser.check(TokenKind::Semicolon)
        {
            parser.advance();
        }
        let _ = parser.match_token(TokenKind::RightBrace);
    }

    let span = parser.span_from(start);
    Initializer::Compound { items, span }
}

// ===========================================================================
// Type Conversion Helpers (Token enums → AST enums)
// ===========================================================================

/// Converts a token-level `IntSuffix` to the AST-level `IntSuffix`.
///
/// Both enum types mirror each other but live in different modules. This
/// conversion bridges the token layer and the AST layer.
fn convert_int_suffix(token_suffix: crate::frontend::lexer::token::IntSuffix) -> IntSuffix {
    use crate::frontend::lexer::token::IntSuffix as TS;
    match token_suffix {
        TS::None => IntSuffix::None,
        TS::Unsigned => IntSuffix::Unsigned,
        TS::Long => IntSuffix::Long,
        TS::ULong => IntSuffix::ULong,
        TS::LongLong => IntSuffix::LongLong,
        TS::ULongLong => IntSuffix::ULongLong,
    }
}

/// Converts a token-level `FloatSuffix` to the AST-level `FloatSuffix`.
fn convert_float_suffix(
    token_suffix: crate::frontend::lexer::token::FloatSuffix,
) -> FloatSuffix {
    use crate::frontend::lexer::token::FloatSuffix as FS;
    match token_suffix {
        FS::None => FloatSuffix::None,
        FS::Float => FloatSuffix::Float,
        FS::Long => FloatSuffix::Long,
    }
}

/// Converts a token-level `NumericBase` to the AST-level `NumericBase`.
fn convert_numeric_base(
    token_base: crate::frontend::lexer::token::NumericBase,
) -> NumericBase {
    use crate::frontend::lexer::token::NumericBase as NB;
    match token_base {
        NB::Decimal => NumericBase::Decimal,
        NB::Hexadecimal => NumericBase::Hexadecimal,
        NB::Octal => NumericBase::Octal,
        NB::Binary => NumericBase::Binary,
    }
}

// ===========================================================================
// Error Recovery Helpers
// ===========================================================================

/// Attempts to recover from a missing `)` by skipping tokens until we find
/// `)`, `;`, or `}`.
fn recover_to_paren_close(parser: &mut Parser<'_>) {
    let mut depth: u32 = 0;
    while !parser.is_at_end() {
        match parser.current().kind {
            TokenKind::LeftParen => {
                depth += 1;
                parser.advance();
            }
            TokenKind::RightParen => {
                if depth == 0 {
                    parser.advance();
                    return;
                }
                depth -= 1;
                parser.advance();
            }
            TokenKind::Semicolon | TokenKind::RightBrace => return,
            _ => {
                parser.advance();
            }
        }
    }
}

/// Attempts to recover from a missing `]` by skipping tokens until we find
/// `]`, `;`, or `}`.
fn recover_to_bracket_close(parser: &mut Parser<'_>) {
    let mut depth: u32 = 0;
    while !parser.is_at_end() {
        match parser.current().kind {
            TokenKind::LeftBracket => {
                depth += 1;
                parser.advance();
            }
            TokenKind::RightBracket => {
                if depth == 0 {
                    parser.advance();
                    return;
                }
                depth -= 1;
                parser.advance();
            }
            // Use match_any to check for statement boundary stop-tokens.
            // match_any consumes the token if it matches, which is
            // acceptable here since we are discarding tokens during recovery.
            _ if parser.match_any(&[TokenKind::Semicolon, TokenKind::RightBrace]).is_some() => return,
            _ => {
                parser.advance();
            }
        }
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::diagnostics::DiagnosticEmitter;
    use crate::common::intern::Interner;
    use crate::common::source_map::SourceSpan;
    use crate::frontend::lexer::token::{
        FloatSuffix as TFS, IntSuffix as TIS, NumericBase as TNB, Token, TokenKind, TokenValue,
    };

    // -----------------------------------------------------------------------
    // Test Helpers
    // -----------------------------------------------------------------------

    /// Creates a dummy `SourceSpan` for test tokens.
    fn dummy_span() -> SourceSpan {
        SourceSpan::dummy()
    }

    /// Creates a simple token with a given kind and no value.
    fn tok(kind: TokenKind) -> Token {
        Token::new(kind, dummy_span(), TokenValue::None)
    }

    /// Creates an identifier token.
    fn tok_id(interner: &mut Interner, name: &str) -> Token {
        let id = interner.intern(name);
        Token::new(TokenKind::Identifier, dummy_span(), TokenValue::Identifier(id))
    }

    /// Creates an integer literal token.
    fn tok_int(value: u128) -> Token {
        Token::new(
            TokenKind::IntegerLiteral,
            dummy_span(),
            TokenValue::Integer {
                value,
                suffix: TIS::None,
                base: TNB::Decimal,
            },
        )
    }

    /// Creates a float literal token.
    fn tok_float(value: f64) -> Token {
        Token::new(
            TokenKind::FloatLiteral,
            dummy_span(),
            TokenValue::Float {
                value,
                suffix: TFS::None,
            },
        )
    }

    /// Creates a string literal token.
    fn tok_str(s: &str) -> Token {
        Token::new(
            TokenKind::StringLiteral,
            dummy_span(),
            TokenValue::Str(s.to_string()),
        )
    }

    /// Creates a character literal token.
    fn tok_char(c: u32) -> Token {
        Token::new(TokenKind::CharLiteral, dummy_span(), TokenValue::Char(c))
    }

    /// Creates an EOF token.
    fn tok_eof() -> Token {
        tok(TokenKind::Eof)
    }

    /// Helper to build a token stream and parse an expression.
    fn parse_tokens(tokens: Vec<Token>) -> (Expression, DiagnosticEmitter) {
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let expr = {
            let mut parser = Parser::new(&tokens, &interner, &mut diag);
            parse_expression(&mut parser)
        };
        (expr, diag)
    }

    /// Helper to parse with a pre-populated interner.
    fn parse_with_interner(
        tokens: Vec<Token>,
        interner: &Interner,
    ) -> (Expression, DiagnosticEmitter) {
        let mut diag = DiagnosticEmitter::new();
        let expr = {
            let mut parser = Parser::new(&tokens, interner, &mut diag);
            parse_expression(&mut parser)
        };
        (expr, diag)
    }

    // -----------------------------------------------------------------------
    // Precedence Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_mul_precedence() {
        // 1 + 2 * 3 → Binary(Add, 1, Binary(Mul, 2, 3))
        let tokens = vec![tok_int(1), tok(TokenKind::Plus), tok_int(2),
                          tok(TokenKind::Star), tok_int(3), tok_eof()];
        let (expr, _) = parse_tokens(tokens);

        match expr {
            Expression::Binary { op: BinaryOp::Add, ref left, ref right, .. } => {
                assert!(matches!(**left, Expression::IntegerLiteral { value: 1, .. }));
                assert!(matches!(**right, Expression::Binary { op: BinaryOp::Mul, .. }));
            }
            _ => panic!("expected Binary(Add, ...), got {:?}", expr),
        }
    }

    #[test]
    fn test_mul_add_precedence() {
        // 1 * 2 + 3 → Binary(Add, Binary(Mul, 1, 2), 3)
        let tokens = vec![tok_int(1), tok(TokenKind::Star), tok_int(2),
                          tok(TokenKind::Plus), tok_int(3), tok_eof()];
        let (expr, _) = parse_tokens(tokens);

        match expr {
            Expression::Binary { op: BinaryOp::Add, ref left, ref right, .. } => {
                assert!(matches!(**left, Expression::Binary { op: BinaryOp::Mul, .. }));
                assert!(matches!(**right, Expression::IntegerLiteral { value: 3, .. }));
            }
            _ => panic!("expected Binary(Add, Binary(Mul, ...), 3), got {:?}", expr),
        }
    }

    #[test]
    fn test_logical_precedence() {
        // a || b && c → Binary(Or, a, Binary(And, b, c))
        let mut interner = Interner::new();
        let tokens = vec![
            tok_id(&mut interner, "a"), tok(TokenKind::PipePipe),
            tok_id(&mut interner, "b"), tok(TokenKind::AmpAmp),
            tok_id(&mut interner, "c"), tok_eof(),
        ];
        let (expr, _) = parse_with_interner(tokens, &interner);

        match expr {
            Expression::Binary { op: BinaryOp::LogicalOr, ref right, .. } => {
                assert!(matches!(**right, Expression::Binary { op: BinaryOp::LogicalAnd, .. }));
            }
            _ => panic!("expected Binary(LogicalOr, ..., Binary(LogicalAnd, ...))"),
        }
    }

    // -----------------------------------------------------------------------
    // Assignment Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_simple_assignment() {
        // x = 5
        let mut interner = Interner::new();
        let tokens = vec![
            tok_id(&mut interner, "x"), tok(TokenKind::Equal),
            tok_int(5), tok_eof(),
        ];
        let (expr, _) = parse_with_interner(tokens, &interner);

        assert!(matches!(expr, Expression::Assignment {
            op: AssignmentOp::Assign, ..
        }));
    }

    #[test]
    fn test_compound_assignment() {
        // x += 1
        let mut interner = Interner::new();
        let tokens = vec![
            tok_id(&mut interner, "x"), tok(TokenKind::PlusEqual),
            tok_int(1), tok_eof(),
        ];
        let (expr, _) = parse_with_interner(tokens, &interner);

        assert!(matches!(expr, Expression::Assignment {
            op: AssignmentOp::AddAssign, ..
        }));
    }

    #[test]
    fn test_right_assoc_assignment() {
        // a = b = c → Assignment(=, a, Assignment(=, b, c))
        let mut interner = Interner::new();
        let tokens = vec![
            tok_id(&mut interner, "a"), tok(TokenKind::Equal),
            tok_id(&mut interner, "b"), tok(TokenKind::Equal),
            tok_id(&mut interner, "c"), tok_eof(),
        ];
        let (expr, _) = parse_with_interner(tokens, &interner);

        match expr {
            Expression::Assignment { op: AssignmentOp::Assign, ref value, .. } => {
                assert!(matches!(**value, Expression::Assignment {
                    op: AssignmentOp::Assign, ..
                }));
            }
            _ => panic!("expected nested assignment"),
        }
    }

    // -----------------------------------------------------------------------
    // Ternary Test
    // -----------------------------------------------------------------------

    #[test]
    fn test_ternary_expression() {
        // a ? b : c
        let mut interner = Interner::new();
        let tokens = vec![
            tok_id(&mut interner, "a"), tok(TokenKind::Question),
            tok_id(&mut interner, "b"), tok(TokenKind::Colon),
            tok_id(&mut interner, "c"), tok_eof(),
        ];
        let (expr, _) = parse_with_interner(tokens, &interner);

        assert!(matches!(expr, Expression::Ternary { .. }));
    }

    // -----------------------------------------------------------------------
    // Unary Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_unary_prefix() {
        // -x
        let mut interner = Interner::new();
        let tokens = vec![
            tok(TokenKind::Minus), tok_id(&mut interner, "x"), tok_eof(),
        ];
        let (expr, _) = parse_with_interner(tokens, &interner);

        match expr {
            Expression::UnaryPrefix { op: UnaryOp::Negate, .. } => {}
            _ => panic!("expected UnaryPrefix(Negate, ...)"),
        }
    }

    #[test]
    fn test_double_negation() {
        // -(-x) → Negate(Negate(x))
        let mut interner = Interner::new();
        let tokens = vec![
            tok(TokenKind::Minus), tok(TokenKind::LeftParen),
            tok(TokenKind::Minus), tok_id(&mut interner, "x"),
            tok(TokenKind::RightParen), tok_eof(),
        ];
        let (expr, _) = parse_with_interner(tokens, &interner);

        match expr {
            Expression::UnaryPrefix { op: UnaryOp::Negate, ref operand, .. } => {
                match **operand {
                    Expression::Paren { ref inner, .. } => {
                        assert!(matches!(**inner, Expression::UnaryPrefix {
                            op: UnaryOp::Negate, ..
                        }));
                    }
                    _ => panic!("expected Paren wrapping negation"),
                }
            }
            _ => panic!("expected UnaryPrefix(Negate, ...)"),
        }
    }

    #[test]
    fn test_pre_increment() {
        // ++x
        let mut interner = Interner::new();
        let tokens = vec![
            tok(TokenKind::PlusPlus), tok_id(&mut interner, "x"), tok_eof(),
        ];
        let (expr, _) = parse_with_interner(tokens, &interner);

        assert!(matches!(expr, Expression::UnaryPrefix {
            op: UnaryOp::PreIncrement, ..
        }));
    }

    #[test]
    fn test_address_of_and_deref() {
        // *&x → Deref(AddressOf(x))
        let mut interner = Interner::new();
        let tokens = vec![
            tok(TokenKind::Star), tok(TokenKind::Amp),
            tok_id(&mut interner, "x"), tok_eof(),
        ];
        let (expr, _) = parse_with_interner(tokens, &interner);

        match expr {
            Expression::UnaryPrefix { op: UnaryOp::Dereference, ref operand, .. } => {
                assert!(matches!(**operand, Expression::UnaryPrefix {
                    op: UnaryOp::AddressOf, ..
                }));
            }
            _ => panic!("expected Deref(AddressOf(x))"),
        }
    }

    // -----------------------------------------------------------------------
    // Postfix Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_post_increment() {
        // x++
        let mut interner = Interner::new();
        let tokens = vec![
            tok_id(&mut interner, "x"), tok(TokenKind::PlusPlus), tok_eof(),
        ];
        let (expr, _) = parse_with_interner(tokens, &interner);

        assert!(matches!(expr, Expression::PostIncrement { .. }));
    }

    #[test]
    fn test_function_call() {
        // f(1, 2, 3)
        let mut interner = Interner::new();
        let tokens = vec![
            tok_id(&mut interner, "f"), tok(TokenKind::LeftParen),
            tok_int(1), tok(TokenKind::Comma),
            tok_int(2), tok(TokenKind::Comma),
            tok_int(3), tok(TokenKind::RightParen), tok_eof(),
        ];
        let (expr, _) = parse_with_interner(tokens, &interner);

        match expr {
            Expression::Call { ref args, .. } => {
                assert_eq!(args.len(), 3);
            }
            _ => panic!("expected Call expression"),
        }
    }

    #[test]
    fn test_empty_function_call() {
        // f()
        let mut interner = Interner::new();
        let tokens = vec![
            tok_id(&mut interner, "f"), tok(TokenKind::LeftParen),
            tok(TokenKind::RightParen), tok_eof(),
        ];
        let (expr, _) = parse_with_interner(tokens, &interner);

        match expr {
            Expression::Call { ref args, .. } => {
                assert!(args.is_empty());
            }
            _ => panic!("expected empty Call expression"),
        }
    }

    #[test]
    fn test_array_subscript() {
        // a[i]
        let mut interner = Interner::new();
        let tokens = vec![
            tok_id(&mut interner, "a"), tok(TokenKind::LeftBracket),
            tok_id(&mut interner, "i"), tok(TokenKind::RightBracket), tok_eof(),
        ];
        let (expr, _) = parse_with_interner(tokens, &interner);

        assert!(matches!(expr, Expression::Subscript { .. }));
    }

    #[test]
    fn test_member_access() {
        // s.x
        let mut interner = Interner::new();
        let tokens = vec![
            tok_id(&mut interner, "s"), tok(TokenKind::Dot),
            tok_id(&mut interner, "x"), tok_eof(),
        ];
        let (expr, _) = parse_with_interner(tokens, &interner);

        assert!(matches!(expr, Expression::MemberAccess { .. }));
    }

    #[test]
    fn test_arrow_access() {
        // p->y
        let mut interner = Interner::new();
        let tokens = vec![
            tok_id(&mut interner, "p"), tok(TokenKind::Arrow),
            tok_id(&mut interner, "y"), tok_eof(),
        ];
        let (expr, _) = parse_with_interner(tokens, &interner);

        assert!(matches!(expr, Expression::ArrowAccess { .. }));
    }

    #[test]
    fn test_chained_postfix() {
        // f(a, b)[i].x → MemberAccess(Subscript(Call(f, [a, b]), i), x)
        let mut interner = Interner::new();
        let tokens = vec![
            tok_id(&mut interner, "f"),
            tok(TokenKind::LeftParen),
            tok_id(&mut interner, "a"), tok(TokenKind::Comma),
            tok_id(&mut interner, "b"),
            tok(TokenKind::RightParen),
            tok(TokenKind::LeftBracket),
            tok_id(&mut interner, "i"),
            tok(TokenKind::RightBracket),
            tok(TokenKind::Dot),
            tok_id(&mut interner, "x"),
            tok_eof(),
        ];
        let (expr, _) = parse_with_interner(tokens, &interner);

        match expr {
            Expression::MemberAccess { ref object, .. } => {
                match **object {
                    Expression::Subscript { ref array, .. } => {
                        assert!(matches!(**array, Expression::Call { .. }));
                    }
                    _ => panic!("expected Subscript inside MemberAccess"),
                }
            }
            _ => panic!("expected MemberAccess at top level"),
        }
    }

    // -----------------------------------------------------------------------
    // Literal Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_integer_literal() {
        let tokens = vec![tok_int(42), tok_eof()];
        let (expr, _) = parse_tokens(tokens);

        match expr {
            Expression::IntegerLiteral { value: 42, .. } => {}
            _ => panic!("expected IntegerLiteral(42)"),
        }
    }

    #[test]
    fn test_float_literal() {
        let tokens = vec![tok_float(3.14), tok_eof()];
        let (expr, _) = parse_tokens(tokens);

        match expr {
            Expression::FloatLiteral { .. } => {}
            _ => panic!("expected FloatLiteral"),
        }
    }

    #[test]
    fn test_string_concatenation() {
        // "hello" " " "world" → single string "hello world"
        let tokens = vec![
            tok_str("hello"), tok_str(" "), tok_str("world"), tok_eof(),
        ];
        let (expr, _) = parse_tokens(tokens);

        match expr {
            Expression::StringLiteral { ref value, .. } => {
                assert_eq!(value, "hello world");
            }
            _ => panic!("expected StringLiteral"),
        }
    }

    #[test]
    fn test_char_literal() {
        let tokens = vec![tok_char(b'a' as u32), tok_eof()];
        let (expr, _) = parse_tokens(tokens);

        match expr {
            Expression::CharLiteral { value, .. } => {
                assert_eq!(value, b'a' as u32);
            }
            _ => panic!("expected CharLiteral"),
        }
    }

    // -----------------------------------------------------------------------
    // Comma Expression Test
    // -----------------------------------------------------------------------

    #[test]
    fn test_comma_expression() {
        // 1, 2, 3 → Comma { exprs: [1, 2, 3] }
        let tokens = vec![
            tok_int(1), tok(TokenKind::Comma),
            tok_int(2), tok(TokenKind::Comma),
            tok_int(3), tok_eof(),
        ];
        let (expr, _) = parse_tokens(tokens);

        match expr {
            Expression::Comma { ref exprs, .. } => {
                assert_eq!(exprs.len(), 3);
            }
            _ => panic!("expected Comma expression"),
        }
    }

    // -----------------------------------------------------------------------
    // Error Recovery Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_error_unexpected_eof() {
        let tokens = vec![tok_eof()];
        let (expr, diag) = parse_tokens(tokens);

        assert!(matches!(expr, Expression::Error { .. }));
        assert!(diag.has_errors());
    }

    #[test]
    fn test_parenthesized_expression() {
        // (42)
        let tokens = vec![
            tok(TokenKind::LeftParen), tok_int(42),
            tok(TokenKind::RightParen), tok_eof(),
        ];
        let (expr, _) = parse_tokens(tokens);

        match expr {
            Expression::Paren { ref inner, .. } => {
                assert!(matches!(**inner, Expression::IntegerLiteral { value: 42, .. }));
            }
            _ => panic!("expected Paren expression"),
        }
    }

    // -----------------------------------------------------------------------
    // Bitwise / Shift / Comparison Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_bitwise_or_xor_and_precedence() {
        // a | b ^ c & d → Binary(|, a, Binary(^, b, Binary(&, c, d)))
        let mut interner = Interner::new();
        let tokens = vec![
            tok_id(&mut interner, "a"), tok(TokenKind::Pipe),
            tok_id(&mut interner, "b"), tok(TokenKind::Caret),
            tok_id(&mut interner, "c"), tok(TokenKind::Amp),
            tok_id(&mut interner, "d"), tok_eof(),
        ];
        let (expr, _) = parse_with_interner(tokens, &interner);

        match expr {
            Expression::Binary { op: BinaryOp::BitwiseOr, ref right, .. } => {
                match **right {
                    Expression::Binary { op: BinaryOp::BitwiseXor, ref right, .. } => {
                        assert!(matches!(**right, Expression::Binary {
                            op: BinaryOp::BitwiseAnd, ..
                        }));
                    }
                    _ => panic!("expected BitwiseXor"),
                }
            }
            _ => panic!("expected BitwiseOr at top level"),
        }
    }

    #[test]
    fn test_shift_precedence() {
        // a + b << c → Binary(<<, Binary(+, a, b), c)
        let mut interner = Interner::new();
        let tokens = vec![
            tok_id(&mut interner, "a"), tok(TokenKind::Plus),
            tok_id(&mut interner, "b"), tok(TokenKind::LessLess),
            tok_id(&mut interner, "c"), tok_eof(),
        ];
        let (expr, _) = parse_with_interner(tokens, &interner);

        match expr {
            Expression::Binary { op: BinaryOp::ShiftLeft, ref left, .. } => {
                assert!(matches!(**left, Expression::Binary { op: BinaryOp::Add, .. }));
            }
            _ => panic!("expected ShiftLeft at top level"),
        }
    }

    #[test]
    fn test_comparison_equality_precedence() {
        // a == b < c → Binary(==, a, Binary(<, b, c))
        let mut interner = Interner::new();
        let tokens = vec![
            tok_id(&mut interner, "a"), tok(TokenKind::EqualEqual),
            tok_id(&mut interner, "b"), tok(TokenKind::Less),
            tok_id(&mut interner, "c"), tok_eof(),
        ];
        let (expr, _) = parse_with_interner(tokens, &interner);

        match expr {
            Expression::Binary { op: BinaryOp::Equal, ref right, .. } => {
                assert!(matches!(**right, Expression::Binary { op: BinaryOp::Less, .. }));
            }
            _ => panic!("expected Equal at top level"),
        }
    }

    // -----------------------------------------------------------------------
    // Shift-assign Test
    // -----------------------------------------------------------------------

    #[test]
    fn test_shift_assign() {
        // x <<= 2
        let mut interner = Interner::new();
        let tokens = vec![
            tok_id(&mut interner, "x"), tok(TokenKind::LessLessEqual),
            tok_int(2), tok_eof(),
        ];
        let (expr, _) = parse_with_interner(tokens, &interner);

        assert!(matches!(expr, Expression::Assignment {
            op: AssignmentOp::ShlAssign, ..
        }));
    }
}
