//! Expression parsing for the bcc compiler.
//! Stub implementation enabling compilation of the parser module.
//! Full implementation will be provided by the assigned agent.

use super::Parser;
use super::ast::Expression;
use crate::common::source_map::SourceSpan;
use crate::frontend::lexer::token::{TokenKind, TokenValue};

/// Parses a full expression (comma-expression level, precedence 1).
///
/// Stub: parses a single primary expression (identifier or literal).
pub(super) fn parse_expression(parser: &mut Parser<'_>) -> Expression {
    parse_assignment_expression(parser)
}

/// Parses an assignment expression (precedence 2).
///
/// This is the most common entry point for sub-expression parsing: function
/// arguments, array subscripts, initializers, asm operands, and more all
/// parse at assignment-expression level.
///
/// Stub: parses a single primary expression (identifier or literal).
pub(super) fn parse_assignment_expression(parser: &mut Parser<'_>) -> Expression {
    parse_primary(parser)
}

/// Parses a primary expression (identifier, literal, or parenthesized).
///
/// Stub: handles identifiers, integer/string/char/float literals, and
/// returns Error for anything else.
fn parse_primary(parser: &mut Parser<'_>) -> Expression {
    let span = parser.current_span();
    match parser.current().kind {
        TokenKind::Identifier => {
            if let TokenValue::Identifier(id) = parser.current().value {
                parser.advance();
                Expression::Identifier {
                    name: id,
                    span: parser.span_from(span),
                }
            } else {
                parser.advance();
                Expression::Error {
                    span: parser.span_from(span),
                }
            }
        }
        TokenKind::IntegerLiteral => {
            if let TokenValue::Integer {
                value,
                suffix,
                base,
            } = parser.current().value
            {
                parser.advance();
                Expression::IntegerLiteral {
                    value,
                    suffix: super::ast::IntSuffix::None,
                    base: super::ast::NumericBase::Decimal,
                    span: parser.span_from(span),
                }
            } else {
                parser.advance();
                Expression::Error {
                    span: parser.span_from(span),
                }
            }
        }
        TokenKind::StringLiteral => {
            if let TokenValue::Str(ref s) = parser.current().value {
                let val = s.clone();
                parser.advance();
                Expression::StringLiteral {
                    value: val,
                    prefix: super::ast::StringPrefix::None,
                    span: parser.span_from(span),
                }
            } else {
                parser.advance();
                Expression::Error {
                    span: parser.span_from(span),
                }
            }
        }
        _ => {
            parser.error("expected expression");
            Expression::Error {
                span: parser.span_from(span),
            }
        }
    }
}
