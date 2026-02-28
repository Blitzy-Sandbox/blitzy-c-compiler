//! Declaration parsing for the bcc compiler.
//! Stub implementation enabling compilation of the parser module.
//! Full implementation will be provided by the assigned agent.

use super::Parser;
use super::ast::{Declaration, DeclSpecifiers, Declarator, DirectDeclarator, InitDeclarator, TypeSpecifier, StorageClass};
use crate::common::source_map::SourceSpan;

/// Parses a top-level external declaration (variable, function, typedef,
/// struct/union/enum definition, or _Static_assert).
///
/// Called by `Parser::parse_translation_unit()` for each top-level construct.
pub(super) fn parse_external_declaration(parser: &mut Parser<'_>) -> Declaration {
    // Stub: consume tokens until semicolon or EOF to prevent infinite loop,
    // returning an empty declaration.
    let span = parser.current_span();
    while !parser.is_at_end() && !parser.check(crate::frontend::lexer::token::TokenKind::Semicolon) {
        parser.advance();
    }
    if !parser.is_at_end() {
        parser.advance(); // consume semicolon
    }
    Declaration::Empty { span }
}
