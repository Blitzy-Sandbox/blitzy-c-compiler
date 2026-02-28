//! Type specifier parsing for the bcc compiler.
//! Stub implementation enabling compilation of the parser module.
//! Full implementation will be provided by the assigned agent.

use super::Parser;
use super::ast::{
    DeclSpecifiers, Declarator, DirectDeclarator, FunctionSpecifier, Pointer, StorageClass,
    TypeName, TypeQualifier, TypeSpecifier,
};
use crate::common::source_map::SourceSpan;
use crate::frontend::lexer::token::TokenKind;

/// Parses a type name (used in casts, sizeof, typeof, __builtin_va_arg, etc.).
///
/// Grammar:
/// ```text
///   type-name := specifier-qualifier-list abstract-declarator?
/// ```
///
/// Stub: parses a basic type specifier keyword and returns a minimal TypeName.
pub(super) fn parse_type_name(parser: &mut Parser<'_>) -> TypeName {
    let span = parser.current_span();

    let type_spec = parse_basic_type_specifier(parser);

    // Check for pointer declarator suffix (e.g., `int *`).
    let abstract_declarator = if parser.check(TokenKind::Star) {
        let mut pointers = Vec::new();
        while parser.match_token(TokenKind::Star) {
            pointers.push(Pointer {
                qualifiers: Vec::new(),
            });
        }
        Some(super::ast::AbstractDeclarator {
            pointer: pointers,
            direct: None,
            span: parser.span_from(span),
        })
    } else {
        None
    };

    let end_span = parser.span_from(span);
    TypeName {
        specifiers: DeclSpecifiers {
            storage_class: None,
            type_qualifiers: Vec::new(),
            type_specifier: type_spec,
            function_specifiers: Vec::new(),
            attributes: Vec::new(),
            span: end_span,
        },
        abstract_declarator,
        span: end_span,
    }
}

/// Parses a basic type specifier keyword.
///
/// Stub: recognizes basic C type keywords and struct/union/enum references.
fn parse_basic_type_specifier(parser: &mut Parser<'_>) -> TypeSpecifier {
    let kind = parser.current().kind;
    match kind {
        TokenKind::Void => {
            parser.advance();
            TypeSpecifier::Void
        }
        TokenKind::Char => {
            parser.advance();
            TypeSpecifier::Char
        }
        TokenKind::Short => {
            parser.advance();
            TypeSpecifier::Short
        }
        TokenKind::Int => {
            parser.advance();
            TypeSpecifier::Int
        }
        TokenKind::Long => {
            parser.advance();
            TypeSpecifier::Long
        }
        TokenKind::Float => {
            parser.advance();
            TypeSpecifier::Float
        }
        TokenKind::Double => {
            parser.advance();
            TypeSpecifier::Double
        }
        TokenKind::Signed => {
            parser.advance();
            let inner = parse_basic_type_specifier(parser);
            TypeSpecifier::Signed(Box::new(inner))
        }
        TokenKind::Unsigned => {
            parser.advance();
            let inner = parse_basic_type_specifier(parser);
            TypeSpecifier::Unsigned(Box::new(inner))
        }
        TokenKind::Bool => {
            parser.advance();
            TypeSpecifier::Bool
        }
        TokenKind::Struct => {
            parser.advance();
            // Expect tag name.
            if let crate::frontend::lexer::token::TokenValue::Identifier(id) =
                parser.current().value
            {
                let span = parser.current_span();
                parser.advance();
                TypeSpecifier::StructRef {
                    tag: id,
                    span: parser.span_from(span),
                }
            } else {
                TypeSpecifier::Error
            }
        }
        TokenKind::Union => {
            parser.advance();
            if let crate::frontend::lexer::token::TokenValue::Identifier(id) =
                parser.current().value
            {
                let span = parser.current_span();
                parser.advance();
                TypeSpecifier::UnionRef {
                    tag: id,
                    span: parser.span_from(span),
                }
            } else {
                TypeSpecifier::Error
            }
        }
        TokenKind::Enum => {
            parser.advance();
            if let crate::frontend::lexer::token::TokenValue::Identifier(id) =
                parser.current().value
            {
                let span = parser.current_span();
                parser.advance();
                TypeSpecifier::EnumRef {
                    tag: id,
                    span: parser.span_from(span),
                }
            } else {
                TypeSpecifier::Error
            }
        }
        TokenKind::Identifier => {
            // Could be a typedef name.
            if let crate::frontend::lexer::token::TokenValue::Identifier(id) =
                parser.current().value
            {
                if parser.is_typedef_name(id) {
                    let span = parser.current_span();
                    parser.advance();
                    return TypeSpecifier::TypedefName {
                        name: id,
                        span: parser.span_from(span),
                    };
                }
            }
            // Not a type — fall through to error.
            TypeSpecifier::Error
        }
        _ => TypeSpecifier::Error,
    }
}
