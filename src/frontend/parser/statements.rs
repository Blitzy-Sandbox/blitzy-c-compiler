//! Statement parsing for the bcc compiler.
//! Stub implementation enabling compilation of the parser module.
//! Full implementation will be provided by the assigned agent.

use super::Parser;
use super::ast::{BlockItem, Statement};
use crate::common::source_map::SourceSpan;

/// Parses a compound statement (block): `{ item1 item2 ... }`.
///
/// The opening `{` has NOT been consumed yet. This function consumes
/// both the opening `{` and the closing `}`.
///
/// Stub: Consumes balanced braces and returns an empty compound statement.
pub(super) fn parse_compound_statement(parser: &mut Parser<'_>) -> Statement {
    let span = parser.current_span();

    // Consume the opening `{`.
    if parser.check(crate::frontend::lexer::token::TokenKind::LeftBrace) {
        parser.advance();
    }

    // Skip tokens until matching `}`.
    let mut depth: u32 = 1;
    while !parser.is_at_end() && depth > 0 {
        match parser.current().kind {
            crate::frontend::lexer::token::TokenKind::LeftBrace => {
                depth += 1;
                parser.advance();
            }
            crate::frontend::lexer::token::TokenKind::RightBrace => {
                depth -= 1;
                parser.advance();
            }
            _ => {
                parser.advance();
            }
        }
    }

    let end_span = parser.span_from(span);
    Statement::Compound {
        items: Vec::new(),
        span: end_span,
    }
}
