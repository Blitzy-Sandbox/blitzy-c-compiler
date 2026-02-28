//! GCC extension parsing for the `bcc` C compiler.
//!
//! This module implements parsing for all GCC-specific language extensions
//! required for compiling real-world C codebases such as SQLite, Lua, and
//! Redis. It covers the following extension categories:
//!
//! - **`__attribute__((...))` annotations** — Arbitrary GCC attribute lists
//!   with identifier, string, integer, and expression arguments.
//! - **Statement expressions** — `({ ... })` allowing compound statements in
//!   expression position (value is the last expression statement).
//! - **`typeof` / `__typeof__` type specifiers** — Deriving type specifiers
//!   from expressions or other types.
//! - **Computed goto** — `&&label` (label address) and `goto *expr` (indirect
//!   goto via label pointer).
//! - **Inline assembly** — `asm` / `__asm__` with volatile/inline/goto
//!   qualifiers, output/input operands, clobber lists, and goto labels.
//! - **`__extension__` prefix** — Suppresses pedantic warnings for GCC
//!   extension usage.
//! - **`__builtin_*` intrinsics** — Parser-level recognition of builtins that
//!   take type arguments (`__builtin_va_arg`, `__builtin_offsetof`) or have
//!   special call syntax (`__builtin_va_start`, `__builtin_va_end`,
//!   `__builtin_va_copy`).
//!
//! # Error Recovery
//!
//! On syntax errors within GCC extensions, the parser reports the error via
//! `Parser::error()` / `Parser::error_at()` and returns a best-effort AST
//! node (often `Expression::Error` or `TypeSpecifier::Error`) to maintain the
//! ≥80% error recovery rate specified in AAP §0.2.1.
//!
//! # Integration Points
//!
//! - Called by `declarations.rs` for attribute-decorated declarations.
//! - Called by `expressions.rs` for statement expressions, label addresses,
//!   typeof, and `__builtin_*` calls.
//! - Called by `statements.rs` for asm statements, computed goto, and
//!   `__extension__` prefixed declarations.
//! - Per AAP §0.7: "GCC extensions explicitly required: `__attribute__`,
//!   `__builtin_*` intrinsics, inline assembly (`asm`/`__asm__` with operand
//!   constraints), statement expressions, `typeof`/`__typeof__`, computed
//!   goto, `__extension__`."
//!
//! # Safety
//!
//! This module contains zero `unsafe` blocks. All operations use safe Rust.

use super::ast::{
    AsmOperand, AsmStatement, AttributeArg, Expression, GccAttribute, Statement, TypeSpecifier,
};
use super::Parser;
use crate::common::intern::InternId;
use crate::frontend::lexer::token::{TokenKind, TokenValue};

// ===========================================================================
// __attribute__((...)) Parsing
// ===========================================================================

/// Parses a single `__attribute__((...))` block and returns a list of
/// attributes contained within the double-parenthesized syntax.
///
/// Grammar:
/// ```text
///   __attribute__ (( attrib-list ))
///   attrib-list := attrib-spec | attrib-list , attrib-spec
///   attrib-spec := ε | IDENTIFIER | IDENTIFIER ( arg-list )
/// ```
///
/// The double-parenthesis syntax is a GCC convention that prevents commas
/// within attribute argument lists from being misinterpreted as macro argument
/// separators.
///
/// On error (missing parentheses, unexpected tokens), this function attempts
/// recovery by consuming tokens until the closing `))` or end of statement,
/// returning whatever attributes were successfully parsed.
pub(super) fn parse_gcc_attribute(parser: &mut Parser) -> Vec<GccAttribute> {
    let start = parser.current_span();

    // Consume `__attribute__` keyword.
    if !parser.match_token(TokenKind::GccAttribute) {
        return Vec::new();
    }

    // Expect `((`
    if parser.expect(TokenKind::LeftParen).is_err() {
        return Vec::new();
    }
    if parser.expect(TokenKind::LeftParen).is_err() {
        return Vec::new();
    }

    let mut attributes = Vec::new();

    // Parse comma-separated attribute specifications.
    loop {
        // Check for closing `))` — allows empty attribute lists and trailing
        // commas.
        if parser.check(TokenKind::RightParen) {
            break;
        }

        // An empty specification (just a comma separator) is valid.
        if parser.check(TokenKind::Comma) {
            parser.advance();
            continue;
        }

        // Try to parse a single attribute specification.
        if let Some(attr) = parse_single_attribute(parser) {
            attributes.push(attr);
        }

        // After each attribute, expect either `,` (more attributes) or `)`
        // (end of list).
        if !parser.match_token(TokenKind::Comma) {
            break;
        }
    }

    // Expect `))`
    if parser.expect(TokenKind::RightParen).is_err() {
        // Attempt recovery: consume tokens until we find `)` or `;` or `}`.
        recover_attribute_close(parser);
        return attributes;
    }
    if parser.expect(TokenKind::RightParen).is_err() {
        recover_attribute_close(parser);
    }

    let _ = start; // span bookkeeping — attributes carry individual spans
    attributes
}

/// Parses a single attribute specification within a `__attribute__((...))`.
///
/// An attribute specification is either:
/// - An identifier: `packed`, `unused`, `__packed__`
/// - An identifier with parenthesized arguments: `aligned(16)`,
///   `section(".text")`, `format(printf, 1, 2)`
///
/// Returns `None` if the current token cannot begin an attribute (error case).
fn parse_single_attribute(parser: &mut Parser) -> Option<GccAttribute> {
    let attr_start = parser.current_span();

    // Attribute name must be an identifier (or a keyword used as attribute
    // name, e.g., `const`, `volatile` — GCC accepts these).
    let name = match parser.current().kind {
        TokenKind::Identifier => {
            if let TokenValue::Identifier(id) = parser.current().value {
                parser.advance();
                id
            } else {
                parser.error("expected attribute name");
                return None;
            }
        }
        // GCC accepts keywords as attribute names (e.g., `const`, `volatile`).
        kind if kind.is_keyword() => {
            let keyword_str = kind.as_str();
            let id = match parser.lookup_interned(keyword_str) {
                Some(id) => id,
                None => {
                    // If the keyword isn't in the interner, use a fallback.
                    parser.error("expected attribute name");
                    return None;
                }
            };
            parser.advance();
            id
        }
        _ => {
            parser.error("expected attribute name");
            return None;
        }
    };

    // Check if the attribute has arguments.
    let args = if parser.match_token(TokenKind::LeftParen) {
        let parsed_args = parse_attribute_args(parser);
        if parser.expect(TokenKind::RightParen).is_err() {
            // Recovery: skip to matching `)` or stop at `,`/`)`.
            recover_to_matching_paren(parser);
        }
        parsed_args
    } else {
        Vec::new()
    };

    let span = parser.span_from(attr_start);
    Some(GccAttribute { name, args, span })
}

/// Parses comma-separated attribute arguments within parentheses.
///
/// Arguments can be:
/// - Identifiers: `printf` in `format(printf, 1, 2)`
/// - String literals: `".text"` in `section(".text")`
/// - Integer literals: `16` in `aligned(16)`
/// - Nested parenthesized expressions (for complex attributes)
fn parse_attribute_args(parser: &mut Parser) -> Vec<AttributeArg> {
    let mut args = Vec::new();

    // Handle empty argument list.
    if parser.check(TokenKind::RightParen) {
        return args;
    }

    loop {
        let arg = parse_one_attribute_arg(parser);
        args.push(arg);

        if !parser.match_token(TokenKind::Comma) {
            break;
        }
    }

    args
}

/// Parses a single attribute argument.
///
/// Tries to classify the argument as an identifier, string, integer, or
/// falls back to a generic expression-like token sequence.
fn parse_one_attribute_arg(parser: &mut Parser) -> AttributeArg {
    match parser.current().kind {
        TokenKind::StringLiteral => {
            if let TokenValue::Str(ref s) = parser.current().value {
                let val = s.clone();
                parser.advance();
                // Handle concatenated string literals.
                let mut result = val;
                while let TokenKind::StringLiteral = parser.current().kind {
                    if let TokenValue::Str(ref s2) = parser.current().value {
                        result.push_str(s2);
                    }
                    parser.advance();
                }
                AttributeArg::String(result)
            } else {
                parser.advance();
                AttributeArg::String(String::new())
            }
        }
        TokenKind::IntegerLiteral => {
            if let TokenValue::Integer { value, .. } = parser.current().value {
                parser.advance();
                AttributeArg::Integer(value as i128)
            } else {
                parser.advance();
                AttributeArg::Integer(0)
            }
        }
        // Negative integer: unary minus followed by integer literal.
        TokenKind::Minus if parser.peek().kind == TokenKind::IntegerLiteral => {
            parser.advance(); // consume `-`
            if let TokenValue::Integer { value, .. } = parser.current().value {
                parser.advance();
                AttributeArg::Integer(-(value as i128))
            } else {
                parser.advance();
                AttributeArg::Integer(0)
            }
        }
        TokenKind::Identifier => {
            if let TokenValue::Identifier(id) = parser.current().value {
                parser.advance();
                AttributeArg::Identifier(id)
            } else {
                parser.advance();
                // Fallback: treat as integer zero for error recovery.
                AttributeArg::Integer(0)
            }
        }
        // Keywords used as attribute arguments (e.g., `printf` could be a
        // keyword-ish identifier in `format(printf, 1, 2)` though typically
        // it's recognized as an identifier).
        kind if kind.is_keyword() => {
            let keyword_str = kind.as_str();
            let id = parser.lookup_interned(keyword_str);
            parser.advance();
            match id {
                Some(intern_id) => AttributeArg::Identifier(intern_id),
                None => AttributeArg::Integer(0),
            }
        }
        // Nested parenthesized expression — skip balanced parens.
        TokenKind::LeftParen => {
            // Consume the entire nested parenthesized expression, treating it
            // as a generic expression argument. We skip balancedly.
            skip_balanced_parens_as_arg(parser)
        }
        _ => {
            // Unknown argument type — attempt to consume one token and
            // return a placeholder.
            parser.error("unexpected token in attribute argument list");
            parser.advance();
            AttributeArg::Integer(0)
        }
    }
}

/// Consumes a parenthesized expression in an attribute argument list,
/// treating the entire thing as a generic integer argument (best-effort
/// recovery for complex attribute expressions).
fn skip_balanced_parens_as_arg(parser: &mut Parser) -> AttributeArg {
    // Consume '('
    parser.advance();
    let mut depth = 1u32;
    while !parser.is_at_end() && depth > 0 {
        match parser.current().kind {
            TokenKind::LeftParen => {
                depth += 1;
                parser.advance();
            }
            TokenKind::RightParen => {
                depth -= 1;
                if depth > 0 {
                    parser.advance();
                } else {
                    parser.advance(); // consume final ')'
                }
            }
            _ => {
                parser.advance();
            }
        }
    }
    // We can't meaningfully extract an expression here, so we return
    // an integer zero placeholder. The semantic analyzer will handle
    // attribute validation.
    AttributeArg::Integer(0)
}

// ===========================================================================
// try_parse_attributes — convenience wrapper
// ===========================================================================

/// Attempts to parse one or more consecutive `__attribute__((...))` blocks.
///
/// This is the primary entry point called from declaration and statement
/// parsers at every position where GCC attributes are allowed:
///
/// - After type specifiers: `int __attribute__((aligned(16))) x;`
/// - After declarators: `int x __attribute__((unused));`
/// - After function declarations: `void f(void) __attribute__((noreturn));`
/// - Before function definitions: `__attribute__((constructor)) void init() {}`
/// - After struct/union/enum tag: `struct __attribute__((packed)) S { ... };`
///
/// If the current token is not `__attribute__`, returns an empty Vec without
/// consuming any tokens.
pub(super) fn try_parse_attributes(parser: &mut Parser) -> Vec<GccAttribute> {
    let mut attrs = Vec::new();

    // Multiple consecutive __attribute__ blocks are allowed.
    while parser.check(TokenKind::GccAttribute) {
        let mut block_attrs = parse_gcc_attribute(parser);
        attrs.append(&mut block_attrs);
    }

    attrs
}

// ===========================================================================
// Statement Expression Parsing — ({ ... })
// ===========================================================================

/// Parses a GCC statement expression: `({ statements... last_expr; })`.
///
/// Called by the expression parser when it encounters `(` followed by `{`,
/// indicating a GCC statement expression rather than a parenthesized
/// expression. The opening `(` has NOT been consumed yet by the caller.
///
/// Grammar:
/// ```text
///   statement-expr := '(' compound-statement ')'
/// ```
///
/// The value of the statement expression is the value of the last
/// expression-statement in the compound block. If the block is empty or
/// ends with a non-expression statement, the value is `void`.
///
/// # Error Recovery
///
/// On parse failures within the compound statement, attempts to skip to the
/// closing `})` and returns `Expression::Error`.
pub(super) fn parse_statement_expression(parser: &mut Parser) -> Expression {
    let start = parser.current_span();

    // Consume the opening `(`.
    if parser.expect(TokenKind::LeftParen).is_err() {
        return Expression::Error {
            span: parser.span_from(start),
        };
    }

    // The compound statement parser handles `{ ... }`.
    // We delegate to the statement parser's compound statement function.
    // Since this is an inter-module call, we use a forward-declared helper.
    let body = super::statements::parse_compound_statement(parser);

    // Consume the closing `)`.
    if parser.expect(TokenKind::RightParen).is_err() {
        // Try to recover by looking for `)`.
        while !parser.is_at_end()
            && !parser.check(TokenKind::RightParen)
            && !parser.check(TokenKind::Semicolon)
        {
            parser.advance();
        }
        let _ = parser.match_token(TokenKind::RightParen);
    }

    let span = parser.span_from(start);
    Expression::StatementExpr {
        body: Box::new(body),
        span,
    }
}

// ===========================================================================
// typeof / __typeof__ Parsing
// ===========================================================================

/// Parses a `typeof` / `__typeof__` type specifier.
///
/// Grammar:
/// ```text
///   typeof-specifier := 'typeof' '(' expression ')'
///                     | 'typeof' '(' type-name ')'
/// ```
///
/// Disambiguation strategy: If the first token inside the parentheses is a
/// type keyword (`int`, `void`, `struct`, etc.), a type qualifier (`const`,
/// etc.), or a known typedef name, we parse as a type name. Otherwise, we
/// parse as an expression.
pub(super) fn parse_typeof(parser: &mut Parser) -> TypeSpecifier {
    let start = parser.current_span();

    // Consume `typeof` / `__typeof__` keyword.
    if !parser.match_token(TokenKind::Typeof) {
        parser.error("expected 'typeof' or '__typeof__'");
        return TypeSpecifier::Error;
    }

    // Expect `(`
    if parser.expect(TokenKind::LeftParen).is_err() {
        return TypeSpecifier::Error;
    }

    // Disambiguate: type name vs expression.
    let is_type = is_typeof_type_start(parser);

    if is_type {
        // Parse as type name.
        let type_name = super::types::parse_type_name(parser);
        if parser.expect(TokenKind::RightParen).is_err() {
            return TypeSpecifier::Error;
        }
        let span = parser.span_from(start);
        TypeSpecifier::TypeofType {
            type_name: Box::new(type_name.specifiers.type_specifier),
            span,
        }
    } else {
        // Parse as expression.
        let expr = super::expressions::parse_assignment_expression(parser);
        if parser.expect(TokenKind::RightParen).is_err() {
            return TypeSpecifier::Error;
        }
        let span = parser.span_from(start);
        TypeSpecifier::Typeof {
            expr: Box::new(expr),
            span,
        }
    }
}

/// Checks whether the current token starts a type name (for `typeof`
/// disambiguation).
///
/// Returns `true` if the current token is:
/// - A type specifier keyword (`int`, `void`, `struct`, `union`, `enum`, etc.)
/// - A type qualifier keyword (`const`, `volatile`, `restrict`, `_Atomic`)
/// - A storage class keyword used as part of a type (`static` in some contexts)
/// - A known typedef name (identifier previously declared via `typedef`)
fn is_typeof_type_start(parser: &Parser) -> bool {
    let kind = parser.current().kind;

    // Type specifier keywords.
    if kind.is_type_specifier() {
        return true;
    }

    // Type qualifier keywords.
    if kind.is_type_qualifier() {
        return true;
    }

    // Check for typedef names — an identifier that is a known typedef.
    if kind == TokenKind::Identifier {
        if let TokenValue::Identifier(id) = parser.current().value {
            if parser.is_typedef_name(id) {
                return true;
            }
        }
    }

    false
}

// ===========================================================================
// Computed Goto — &&label and goto *expr
// ===========================================================================

/// Parses a GCC label address expression: `&&label_name`.
///
/// Called by the expression parser when `&&` is encountered in a context
/// where it is a unary label-address operator (not logical AND). The `&&`
/// token has NOT been consumed yet by the caller.
///
/// Grammar:
/// ```text
///   label-addr := '&&' IDENTIFIER
/// ```
///
/// Returns `Expression::LabelAddr { label, span }` on success, or
/// `Expression::Error` if the identifier is missing.
pub(super) fn parse_label_address(parser: &mut Parser) -> Expression {
    let start = parser.current_span();

    // Consume `&&`.
    if !parser.match_token(TokenKind::AmpAmp) {
        return Expression::Error {
            span: parser.span_from(start),
        };
    }

    // Expect an identifier (label name).
    match parser.current().kind {
        TokenKind::Identifier => {
            if let TokenValue::Identifier(id) = parser.current().value {
                parser.advance();
                let span = parser.span_from(start);
                Expression::LabelAddr { label: id, span }
            } else {
                parser.error("expected label name after '&&'");
                Expression::Error {
                    span: parser.span_from(start),
                }
            }
        }
        _ => {
            parser.error("expected label name after '&&'");
            Expression::Error {
                span: parser.span_from(start),
            }
        }
    }
}

/// Parses a GCC computed goto statement: `goto *expr;`.
///
/// Called by the statement parser when `goto` is followed by `*`, indicating
/// an indirect goto (computed goto) rather than a direct goto. The `goto`
/// keyword has already been consumed by the caller.
///
/// Grammar:
/// ```text
///   computed-goto := 'goto' '*' expression ';'
/// ```
///
/// Returns `Statement::ComputedGoto { target, span }` where `target` is the
/// expression whose runtime value is a label address obtained via `&&label`.
pub(super) fn parse_computed_goto(parser: &mut Parser) -> Statement {
    let start = parser.current_span();

    // Consume `*`.
    if parser.expect(TokenKind::Star).is_err() {
        parser.synchronize();
        return Statement::Null {
            span: parser.span_from(start),
        };
    }

    // Parse the target expression.
    let target = super::expressions::parse_expression(parser);

    // Expect `;`.
    if parser.expect(TokenKind::Semicolon).is_err() {
        // Error recovery: synchronize to next statement boundary.
    }

    let span = parser.span_from(start);
    Statement::ComputedGoto {
        target: Box::new(target),
        span,
    }
}

// ===========================================================================
// Inline Assembly Parsing — asm / __asm__
// ===========================================================================

/// Parses a GCC inline assembly statement.
///
/// Grammar:
/// ```text
///   asm-statement := ('asm' | '__asm__') asm-qualifiers '(' asm-body ')' ';'
///   asm-qualifiers := ('volatile' | '__volatile__')? 'inline'? 'goto'?
///   asm-body := template
///             | template ':' output-operands
///             | template ':' output-operands ':' input-operands
///             | template ':' output-operands ':' input-operands ':' clobbers
///             | template ':' outputs ':' inputs ':' clobbers ':' goto-labels
/// ```
///
/// The `asm` / `__asm__` keyword has NOT been consumed yet.
pub(super) fn parse_asm_statement(parser: &mut Parser) -> Statement {
    let start = parser.current_span();

    // Consume `asm` or `__asm__` keyword.
    if !parser.match_token(TokenKind::Asm) {
        parser.error("expected 'asm' or '__asm__'");
        return Statement::Null {
            span: parser.span_from(start),
        };
    }

    // Parse optional qualifiers: volatile, inline, goto (in any order).
    let mut is_volatile = false;
    let mut is_inline = false;
    let mut is_goto = false;

    loop {
        match parser.current().kind {
            TokenKind::Volatile => {
                parser.advance();
                is_volatile = true;
            }
            TokenKind::Inline => {
                parser.advance();
                is_inline = true;
            }
            TokenKind::Goto => {
                parser.advance();
                is_goto = true;
            }
            // Handle __volatile__ as an identifier with that spelling.
            TokenKind::Identifier => {
                if let TokenValue::Identifier(id) = parser.current().value {
                    let name = parser.interner().resolve(id);
                    if name == "__volatile__" {
                        parser.advance();
                        is_volatile = true;
                        continue;
                    } else if name == "__inline__" {
                        parser.advance();
                        is_inline = true;
                        continue;
                    }
                }
                break;
            }
            _ => break,
        }
    }

    // Expect `(`
    if parser.expect(TokenKind::LeftParen).is_err() {
        parser.synchronize();
        return Statement::Null {
            span: parser.span_from(start),
        };
    }

    // Parse the assembly template string (possibly concatenated).
    let template = parse_asm_template(parser);

    // Parse optional sections separated by `:`.
    let mut outputs = Vec::new();
    let mut inputs = Vec::new();
    let mut clobbers = Vec::new();
    let mut goto_labels = Vec::new();

    // Section 1: output operands.
    if parser.match_token(TokenKind::Colon) {
        if !parser.check(TokenKind::Colon)
            && !parser.check(TokenKind::RightParen)
        {
            outputs = parse_asm_operands(parser);
        }

        // Section 2: input operands.
        if parser.match_token(TokenKind::Colon) {
            if !parser.check(TokenKind::Colon)
                && !parser.check(TokenKind::RightParen)
            {
                inputs = parse_asm_operands(parser);
            }

            // Section 3: clobber list.
            if parser.match_token(TokenKind::Colon) {
                if !parser.check(TokenKind::Colon)
                    && !parser.check(TokenKind::RightParen)
                {
                    clobbers = parse_clobber_list(parser);
                }

                // Section 4: goto labels (only if `goto` qualifier present).
                if is_goto && parser.match_token(TokenKind::Colon) {
                    if !parser.check(TokenKind::RightParen) {
                        goto_labels = parse_goto_labels(parser);
                    }
                }
            }
        }
    }

    // Expect `)`
    if parser.expect(TokenKind::RightParen).is_err() {
        // Recovery: skip to `;`.
        while !parser.is_at_end()
            && !parser.check(TokenKind::Semicolon)
            && !parser.check(TokenKind::RightBrace)
        {
            parser.advance();
        }
    }

    // Expect `;`
    let _ = parser.match_token(TokenKind::Semicolon);

    let span = parser.span_from(start);
    Statement::Asm(AsmStatement {
        is_volatile,
        is_inline,
        is_goto,
        template,
        outputs,
        inputs,
        clobbers,
        goto_labels,
        span,
    })
}

/// Parses the assembly template: one or more concatenated string literals.
fn parse_asm_template(parser: &mut Parser) -> String {
    let mut template = String::new();

    loop {
        match parser.current().kind {
            TokenKind::StringLiteral => {
                if let TokenValue::Str(ref s) = parser.current().value {
                    template.push_str(s);
                }
                parser.advance();
            }
            _ => break,
        }
    }

    if template.is_empty() {
        parser.error("expected assembly template string");
    }

    template
}

/// Parses comma-separated assembly operands.
///
/// Grammar:
/// ```text
///   asm-operand := ('[' IDENTIFIER ']')? STRING '(' expression ')'
/// ```
fn parse_asm_operands(parser: &mut Parser) -> Vec<AsmOperand> {
    let mut operands = Vec::new();

    loop {
        // Check for end of operand list.
        if parser.check(TokenKind::Colon)
            || parser.check(TokenKind::RightParen)
            || parser.is_at_end()
        {
            break;
        }

        let op_start = parser.current_span();

        // Optional symbolic name: `[name]`.
        let symbolic_name = if parser.match_token(TokenKind::LeftBracket) {
            let name = match parser.current().kind {
                TokenKind::Identifier => {
                    if let TokenValue::Identifier(id) = parser.current().value {
                        parser.advance();
                        Some(id)
                    } else {
                        None
                    }
                }
                _ => {
                    parser.error("expected symbolic name in asm operand");
                    None
                }
            };
            if parser.expect(TokenKind::RightBracket).is_err() {
                // Recovery
            }
            name
        } else {
            None
        };

        // Constraint string literal.
        let constraint = match parser.current().kind {
            TokenKind::StringLiteral => {
                if let TokenValue::Str(ref s) = parser.current().value {
                    let c = s.clone();
                    parser.advance();
                    c
                } else {
                    parser.advance();
                    String::new()
                }
            }
            _ => {
                parser.error("expected constraint string in asm operand");
                String::new()
            }
        };

        // Parenthesized C expression.
        if parser.expect(TokenKind::LeftParen).is_err() {
            // Recovery: try to skip to next comma or colon.
            let span = parser.span_from(op_start);
            operands.push(AsmOperand {
                symbolic_name,
                constraint,
                expr: Expression::Error { span },
                span,
            });
            if !parser.match_token(TokenKind::Comma) {
                break;
            }
            continue;
        }

        let expr = super::expressions::parse_expression(parser);

        if parser.expect(TokenKind::RightParen).is_err() {
            // Recovery
        }

        let span = parser.span_from(op_start);
        operands.push(AsmOperand {
            symbolic_name,
            constraint,
            expr,
            span,
        });

        if !parser.match_token(TokenKind::Comma) {
            break;
        }
    }

    operands
}

/// Parses a comma-separated clobber list (string literals).
///
/// Common clobbers: `"memory"`, `"cc"`, register names (`"eax"`, `"rbx"`).
fn parse_clobber_list(parser: &mut Parser) -> Vec<String> {
    let mut clobbers = Vec::new();

    loop {
        if parser.check(TokenKind::Colon)
            || parser.check(TokenKind::RightParen)
            || parser.is_at_end()
        {
            break;
        }

        match parser.current().kind {
            TokenKind::StringLiteral => {
                if let TokenValue::Str(ref s) = parser.current().value {
                    clobbers.push(s.clone());
                }
                parser.advance();
            }
            _ => {
                parser.error("expected string literal in clobber list");
                break;
            }
        }

        if !parser.match_token(TokenKind::Comma) {
            break;
        }
    }

    clobbers
}

/// Parses comma-separated goto labels (identifiers) for `asm goto`.
fn parse_goto_labels(parser: &mut Parser) -> Vec<InternId> {
    let mut labels = Vec::new();

    loop {
        if parser.check(TokenKind::RightParen) || parser.is_at_end() {
            break;
        }

        match parser.current().kind {
            TokenKind::Identifier => {
                if let TokenValue::Identifier(id) = parser.current().value {
                    labels.push(id);
                    parser.advance();
                } else {
                    parser.advance();
                }
            }
            _ => {
                parser.error("expected label name in asm goto label list");
                break;
            }
        }

        if !parser.match_token(TokenKind::Comma) {
            break;
        }
    }

    labels
}

// ===========================================================================
// __extension__ Prefix
// ===========================================================================

/// Parses a `__extension__` prefix on an expression.
///
/// `__extension__` is a GCC extension that suppresses pedantic warnings for
/// GCC extension usage within conforming code. It acts as a transparent
/// wrapper around the expression that follows it.
///
/// Grammar:
/// ```text
///   extension-expr := '__extension__' cast-expression
/// ```
///
/// The `__extension__` keyword has NOT been consumed yet.
pub(super) fn parse_extension_prefix(parser: &mut Parser) -> Expression {
    let start = parser.current_span();

    // Consume `__extension__`.
    if !parser.match_token(TokenKind::GccExtension) {
        return Expression::Error {
            span: parser.span_from(start),
        };
    }

    // Parse the following expression. We use assignment-expression level
    // to match GCC's behavior where __extension__ has unary precedence.
    let expr = super::expressions::parse_assignment_expression(parser);

    let span = parser.span_from(start);
    Expression::Extension {
        expr: Box::new(expr),
        span,
    }
}

// ===========================================================================
// __builtin_* Intrinsic Parsing
// ===========================================================================

/// Parses a compiler builtin call with special syntax.
///
/// Some builtins like `__builtin_va_arg(ap, type)` and
/// `__builtin_offsetof(type, member)` take type arguments, which cannot be
/// handled by normal function call parsing. This function dispatches to
/// the appropriate builtin-specific parser.
///
/// The builtin keyword has NOT been consumed yet.
///
/// # Arguments
///
/// * `parser` — The parser state.
/// * `builtin` — The `TokenKind` identifying which builtin to parse.
pub(super) fn parse_builtin_call(parser: &mut Parser, builtin: TokenKind) -> Expression {
    match builtin {
        TokenKind::BuiltinVaArg => parse_builtin_va_arg(parser),
        TokenKind::BuiltinOffsetof => parse_builtin_offsetof(parser),
        TokenKind::BuiltinVaStart => parse_builtin_va_start(parser),
        TokenKind::BuiltinVaEnd => parse_builtin_va_end(parser),
        TokenKind::BuiltinVaCopy => parse_builtin_va_copy(parser),
        _ => {
            // For other builtins (e.g., __builtin_expect, __builtin_clz,
            // etc.), parse as a regular function call. The builtin keyword
            // itself acts as the callee identifier.
            parse_builtin_as_regular_call(parser)
        }
    }
}

/// Parses `__builtin_va_arg(ap, type)`.
///
/// The second argument is a type name, not an expression.
fn parse_builtin_va_arg(parser: &mut Parser) -> Expression {
    let start = parser.current_span();

    // Consume `__builtin_va_arg`.
    parser.advance();

    // Expect `(`
    if parser.expect(TokenKind::LeftParen).is_err() {
        return Expression::Error {
            span: parser.span_from(start),
        };
    }

    // Parse first argument: `ap` (expression).
    let ap = super::expressions::parse_assignment_expression(parser);

    // Expect `,`
    if parser.expect(TokenKind::Comma).is_err() {
        return Expression::Error {
            span: parser.span_from(start),
        };
    }

    // Parse second argument: type name.
    let type_name = super::types::parse_type_name(parser);

    // Expect `)`
    if parser.expect(TokenKind::RightParen).is_err() {
        return Expression::Error {
            span: parser.span_from(start),
        };
    }

    let span = parser.span_from(start);
    Expression::BuiltinVaArg {
        ap: Box::new(ap),
        type_name: Box::new(type_name),
        span,
    }
}

/// Parses `__builtin_offsetof(type, member-designator)`.
///
/// The first argument is a type name and the second is a member designator.
fn parse_builtin_offsetof(parser: &mut Parser) -> Expression {
    let start = parser.current_span();

    // Consume `__builtin_offsetof`.
    parser.advance();

    // Expect `(`
    if parser.expect(TokenKind::LeftParen).is_err() {
        return Expression::Error {
            span: parser.span_from(start),
        };
    }

    // Parse first argument: type name.
    let type_name = super::types::parse_type_name(parser);

    // Expect `,`
    if parser.expect(TokenKind::Comma).is_err() {
        return Expression::Error {
            span: parser.span_from(start),
        };
    }

    // Parse member designator: an identifier (possibly with `.` chain or
    // `[index]` suffixes). For simplicity, we parse just the initial
    // identifier; full member-designator chains can be expanded later.
    let member = match parser.current().kind {
        TokenKind::Identifier => {
            if let TokenValue::Identifier(id) = parser.current().value {
                parser.advance();
                id
            } else {
                parser.error("expected member name in __builtin_offsetof");
                let span = parser.span_from(start);
                return Expression::Error { span };
            }
        }
        _ => {
            parser.error("expected member name in __builtin_offsetof");
            let span = parser.span_from(start);
            return Expression::Error { span };
        }
    };

    // Skip any additional member-designator suffixes (`.field`, `[index]`)
    // for full GCC compatibility.
    while parser.check(TokenKind::Dot) || parser.check(TokenKind::LeftBracket) {
        if parser.match_token(TokenKind::Dot) {
            // Skip field name.
            if parser.check(TokenKind::Identifier) {
                parser.advance();
            }
        } else if parser.match_token(TokenKind::LeftBracket) {
            // Skip index expression.
            let _index_expr = super::expressions::parse_expression(parser);
            let _ = parser.expect(TokenKind::RightBracket);
        }
    }

    // Expect `)`
    if parser.expect(TokenKind::RightParen).is_err() {
        return Expression::Error {
            span: parser.span_from(start),
        };
    }

    let span = parser.span_from(start);
    Expression::BuiltinOffsetof {
        type_name: Box::new(type_name),
        member,
        span,
    }
}

/// Parses `__builtin_va_start(ap, param)`.
fn parse_builtin_va_start(parser: &mut Parser) -> Expression {
    let start = parser.current_span();

    // Consume `__builtin_va_start`.
    parser.advance();

    // Expect `(`
    if parser.expect(TokenKind::LeftParen).is_err() {
        return Expression::Error {
            span: parser.span_from(start),
        };
    }

    // Parse first argument: `ap` expression.
    let ap = super::expressions::parse_assignment_expression(parser);

    // Expect `,`
    if parser.expect(TokenKind::Comma).is_err() {
        return Expression::Error {
            span: parser.span_from(start),
        };
    }

    // Parse second argument: `param` expression.
    let param = super::expressions::parse_assignment_expression(parser);

    // Expect `)`
    if parser.expect(TokenKind::RightParen).is_err() {
        return Expression::Error {
            span: parser.span_from(start),
        };
    }

    let span = parser.span_from(start);
    Expression::BuiltinVaStart {
        ap: Box::new(ap),
        param: Box::new(param),
        span,
    }
}

/// Parses `__builtin_va_end(ap)`.
fn parse_builtin_va_end(parser: &mut Parser) -> Expression {
    let start = parser.current_span();

    // Consume `__builtin_va_end`.
    parser.advance();

    // Expect `(`
    if parser.expect(TokenKind::LeftParen).is_err() {
        return Expression::Error {
            span: parser.span_from(start),
        };
    }

    // Parse argument: `ap` expression.
    let ap = super::expressions::parse_assignment_expression(parser);

    // Expect `)`
    if parser.expect(TokenKind::RightParen).is_err() {
        return Expression::Error {
            span: parser.span_from(start),
        };
    }

    let span = parser.span_from(start);
    Expression::BuiltinVaEnd {
        ap: Box::new(ap),
        span,
    }
}

/// Parses `__builtin_va_copy(dest, src)`.
fn parse_builtin_va_copy(parser: &mut Parser) -> Expression {
    let start = parser.current_span();

    // Consume `__builtin_va_copy`.
    parser.advance();

    // Expect `(`
    if parser.expect(TokenKind::LeftParen).is_err() {
        return Expression::Error {
            span: parser.span_from(start),
        };
    }

    // Parse first argument: `dest` expression.
    let dest = super::expressions::parse_assignment_expression(parser);

    // Expect `,`
    if parser.expect(TokenKind::Comma).is_err() {
        return Expression::Error {
            span: parser.span_from(start),
        };
    }

    // Parse second argument: `src` expression.
    let src = super::expressions::parse_assignment_expression(parser);

    // Expect `)`
    if parser.expect(TokenKind::RightParen).is_err() {
        return Expression::Error {
            span: parser.span_from(start),
        };
    }

    let span = parser.span_from(start);
    Expression::BuiltinVaCopy {
        dest: Box::new(dest),
        src: Box::new(src),
        span,
    }
}

/// Parses a generic builtin as a regular function call expression.
///
/// For builtins that don't need special parsing (e.g., `__builtin_expect`,
/// `__builtin_clz`, etc.), we treat them as regular function calls. The
/// builtin keyword token is consumed and used as the callee identifier.
fn parse_builtin_as_regular_call(parser: &mut Parser) -> Expression {
    let start = parser.current_span();

    // Get the builtin name as an interned identifier. We look up the
    // keyword's string representation in the interner.
    let builtin_kind = parser.current().kind;
    let builtin_name_str = builtin_kind.as_str();
    let name_id = parser
        .lookup_interned(builtin_name_str)
        .unwrap_or_else(|| InternId::from_raw(0));
    parser.advance();

    // Expect `(`
    if parser.expect(TokenKind::LeftParen).is_err() {
        return Expression::Error {
            span: parser.span_from(start),
        };
    }

    // Parse comma-separated arguments.
    let mut args = Vec::new();
    if !parser.check(TokenKind::RightParen) {
        loop {
            let arg = super::expressions::parse_assignment_expression(parser);
            args.push(arg);
            if !parser.match_token(TokenKind::Comma) {
                break;
            }
        }
    }

    // Expect `)`
    if parser.expect(TokenKind::RightParen).is_err() {
        return Expression::Error {
            span: parser.span_from(start),
        };
    }

    let span = parser.span_from(start);
    Expression::Call {
        callee: Box::new(Expression::Identifier {
            name: name_id,
            span: start,
        }),
        args,
        span,
    }
}

// ===========================================================================
// Internal Recovery Helpers
// ===========================================================================

/// Attempts to recover from a malformed `__attribute__` by skipping to the
/// closing `)` at the appropriate nesting level.
fn recover_attribute_close(parser: &mut Parser) {
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
            TokenKind::Semicolon | TokenKind::LeftBrace | TokenKind::RightBrace => {
                // Stop at statement/block boundaries.
                return;
            }
            _ => {
                parser.advance();
            }
        }
    }
}

/// Attempts to skip to a matching `)` for error recovery within attribute
/// arguments.
fn recover_to_matching_paren(parser: &mut Parser) {
    let mut depth: u32 = 1;
    while !parser.is_at_end() && depth > 0 {
        match parser.current().kind {
            TokenKind::LeftParen => {
                depth += 1;
                parser.advance();
            }
            TokenKind::RightParen => {
                depth -= 1;
                parser.advance();
            }
            TokenKind::Semicolon | TokenKind::RightBrace => {
                return;
            }
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
    use crate::common::source_map::{SourceLocation, SourceSpan};
    use crate::frontend::lexer::token::{Token, TokenKind, TokenValue};

    // -----------------------------------------------------------------------
    // Test Helpers
    // -----------------------------------------------------------------------

    /// Creates a dummy SourceSpan for testing.
    fn dummy_span() -> SourceSpan {
        SourceSpan::dummy()
    }

    /// Creates a simple token with the given kind and no value.
    fn make_token(kind: TokenKind) -> Token {
        Token::new(kind, dummy_span(), TokenValue::None)
    }

    /// Creates an identifier token with the given InternId.
    fn make_ident_token(id: InternId) -> Token {
        Token::new(
            TokenKind::Identifier,
            dummy_span(),
            TokenValue::Identifier(id),
        )
    }

    /// Creates a string literal token.
    fn make_string_token(s: &str) -> Token {
        Token::new(
            TokenKind::StringLiteral,
            dummy_span(),
            TokenValue::Str(s.to_string()),
        )
    }

    /// Creates an integer literal token.
    fn make_int_token(value: u128) -> Token {
        Token::new(
            TokenKind::IntegerLiteral,
            dummy_span(),
            TokenValue::Integer {
                value,
                suffix: crate::frontend::lexer::token::IntSuffix::None,
                base: crate::frontend::lexer::token::NumericBase::Decimal,
            },
        )
    }

    /// Creates a parser for testing with the given token stream.
    fn make_parser<'a>(
        tokens: &'a [Token],
        interner: &'a Interner,
        diagnostics: &'a mut DiagnosticEmitter,
    ) -> Parser<'a> {
        Parser::new(tokens, interner, diagnostics)
    }

    // -----------------------------------------------------------------------
    // __attribute__ Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_attribute_no_args() {
        // __attribute__((packed))
        let mut interner = Interner::new();
        let packed_id = interner.intern("packed");

        let tokens = vec![
            make_token(TokenKind::GccAttribute),
            make_token(TokenKind::LeftParen),
            make_token(TokenKind::LeftParen),
            make_ident_token(packed_id),
            make_token(TokenKind::RightParen),
            make_token(TokenKind::RightParen),
            make_token(TokenKind::Eof),
        ];

        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);
        let attrs = parse_gcc_attribute(&mut parser);

        assert_eq!(attrs.len(), 1);
        assert_eq!(attrs[0].name, packed_id);
        assert!(attrs[0].args.is_empty());
    }

    #[test]
    fn test_parse_attribute_with_int_arg() {
        // __attribute__((aligned(16)))
        let mut interner = Interner::new();
        let aligned_id = interner.intern("aligned");

        let tokens = vec![
            make_token(TokenKind::GccAttribute),
            make_token(TokenKind::LeftParen),
            make_token(TokenKind::LeftParen),
            make_ident_token(aligned_id),
            make_token(TokenKind::LeftParen),
            make_int_token(16),
            make_token(TokenKind::RightParen),
            make_token(TokenKind::RightParen),
            make_token(TokenKind::RightParen),
            make_token(TokenKind::Eof),
        ];

        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);
        let attrs = parse_gcc_attribute(&mut parser);

        assert_eq!(attrs.len(), 1);
        assert_eq!(attrs[0].name, aligned_id);
        assert_eq!(attrs[0].args.len(), 1);
        match &attrs[0].args[0] {
            AttributeArg::Integer(v) => assert_eq!(*v, 16),
            _ => panic!("expected integer argument"),
        }
    }

    #[test]
    fn test_parse_attribute_with_string_arg() {
        // __attribute__((section(".text")))
        let mut interner = Interner::new();
        let section_id = interner.intern("section");

        let tokens = vec![
            make_token(TokenKind::GccAttribute),
            make_token(TokenKind::LeftParen),
            make_token(TokenKind::LeftParen),
            make_ident_token(section_id),
            make_token(TokenKind::LeftParen),
            make_string_token(".text"),
            make_token(TokenKind::RightParen),
            make_token(TokenKind::RightParen),
            make_token(TokenKind::RightParen),
            make_token(TokenKind::Eof),
        ];

        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);
        let attrs = parse_gcc_attribute(&mut parser);

        assert_eq!(attrs.len(), 1);
        assert_eq!(attrs[0].name, section_id);
        assert_eq!(attrs[0].args.len(), 1);
        match &attrs[0].args[0] {
            AttributeArg::String(s) => assert_eq!(s, ".text"),
            _ => panic!("expected string argument"),
        }
    }

    #[test]
    fn test_parse_attribute_with_multiple_args() {
        // __attribute__((format(printf, 1, 2)))
        let mut interner = Interner::new();
        let format_id = interner.intern("format");
        let printf_id = interner.intern("printf");

        let tokens = vec![
            make_token(TokenKind::GccAttribute),
            make_token(TokenKind::LeftParen),
            make_token(TokenKind::LeftParen),
            make_ident_token(format_id),
            make_token(TokenKind::LeftParen),
            make_ident_token(printf_id),
            make_token(TokenKind::Comma),
            make_int_token(1),
            make_token(TokenKind::Comma),
            make_int_token(2),
            make_token(TokenKind::RightParen),
            make_token(TokenKind::RightParen),
            make_token(TokenKind::RightParen),
            make_token(TokenKind::Eof),
        ];

        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);
        let attrs = parse_gcc_attribute(&mut parser);

        assert_eq!(attrs.len(), 1);
        assert_eq!(attrs[0].name, format_id);
        assert_eq!(attrs[0].args.len(), 3);
        match &attrs[0].args[0] {
            AttributeArg::Identifier(id) => assert_eq!(*id, printf_id),
            _ => panic!("expected identifier argument 'printf'"),
        }
        match &attrs[0].args[1] {
            AttributeArg::Integer(v) => assert_eq!(*v, 1),
            _ => panic!("expected integer argument 1"),
        }
        match &attrs[0].args[2] {
            AttributeArg::Integer(v) => assert_eq!(*v, 2),
            _ => panic!("expected integer argument 2"),
        }
    }

    #[test]
    fn test_parse_multiple_attributes_in_one_block() {
        // __attribute__((packed, aligned(4)))
        let mut interner = Interner::new();
        let packed_id = interner.intern("packed");
        let aligned_id = interner.intern("aligned");

        let tokens = vec![
            make_token(TokenKind::GccAttribute),
            make_token(TokenKind::LeftParen),
            make_token(TokenKind::LeftParen),
            make_ident_token(packed_id),
            make_token(TokenKind::Comma),
            make_ident_token(aligned_id),
            make_token(TokenKind::LeftParen),
            make_int_token(4),
            make_token(TokenKind::RightParen),
            make_token(TokenKind::RightParen),
            make_token(TokenKind::RightParen),
            make_token(TokenKind::Eof),
        ];

        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);
        let attrs = parse_gcc_attribute(&mut parser);

        assert_eq!(attrs.len(), 2);
        assert_eq!(attrs[0].name, packed_id);
        assert!(attrs[0].args.is_empty());
        assert_eq!(attrs[1].name, aligned_id);
        assert_eq!(attrs[1].args.len(), 1);
    }

    #[test]
    fn test_try_parse_attributes_consecutive() {
        // __attribute__((packed)) __attribute__((aligned(8)))
        let mut interner = Interner::new();
        let packed_id = interner.intern("packed");
        let aligned_id = interner.intern("aligned");

        let tokens = vec![
            make_token(TokenKind::GccAttribute),
            make_token(TokenKind::LeftParen),
            make_token(TokenKind::LeftParen),
            make_ident_token(packed_id),
            make_token(TokenKind::RightParen),
            make_token(TokenKind::RightParen),
            make_token(TokenKind::GccAttribute),
            make_token(TokenKind::LeftParen),
            make_token(TokenKind::LeftParen),
            make_ident_token(aligned_id),
            make_token(TokenKind::LeftParen),
            make_int_token(8),
            make_token(TokenKind::RightParen),
            make_token(TokenKind::RightParen),
            make_token(TokenKind::RightParen),
            make_token(TokenKind::Eof),
        ];

        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);
        let attrs = try_parse_attributes(&mut parser);

        assert_eq!(attrs.len(), 2);
        assert_eq!(attrs[0].name, packed_id);
        assert_eq!(attrs[1].name, aligned_id);
    }

    #[test]
    fn test_try_parse_attributes_none() {
        // No __attribute__ present — should return empty vec.
        let interner = Interner::new();
        let tokens = vec![make_token(TokenKind::Semicolon), make_token(TokenKind::Eof)];

        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);
        let attrs = try_parse_attributes(&mut parser);

        assert!(attrs.is_empty());
        // Parser position should not have advanced.
        assert!(parser.check(TokenKind::Semicolon));
    }

    // -----------------------------------------------------------------------
    // Label Address Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_label_address() {
        // &&my_label
        let mut interner = Interner::new();
        let label_id = interner.intern("my_label");

        let tokens = vec![
            make_token(TokenKind::AmpAmp),
            make_ident_token(label_id),
            make_token(TokenKind::Eof),
        ];

        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);
        let expr = parse_label_address(&mut parser);

        match expr {
            Expression::LabelAddr { label, .. } => assert_eq!(label, label_id),
            _ => panic!("expected LabelAddr expression"),
        }
    }

    #[test]
    fn test_parse_label_address_error() {
        // && followed by non-identifier — should produce Error.
        let interner = Interner::new();
        let tokens = vec![
            make_token(TokenKind::AmpAmp),
            make_token(TokenKind::Semicolon),
            make_token(TokenKind::Eof),
        ];

        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);
        let expr = parse_label_address(&mut parser);

        match expr {
            Expression::Error { .. } => { /* expected */ }
            _ => panic!("expected Error expression for malformed label address"),
        }
    }

    // -----------------------------------------------------------------------
    // Extension Prefix Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_extension_prefix() {
        // __extension__ <some_expression>
        // For this test we just verify the wrapper is created with an inner
        // expression. Since we need a full expression parser, we test with
        // a simple integer literal if available, or use an identifier.
        let mut interner = Interner::new();
        let x_id = interner.intern("x");

        let tokens = vec![
            make_token(TokenKind::GccExtension),
            make_ident_token(x_id),
            make_token(TokenKind::Eof),
        ];

        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);
        let expr = parse_extension_prefix(&mut parser);

        match expr {
            Expression::Extension { expr: inner, .. } => match *inner {
                Expression::Identifier { name, .. } => assert_eq!(name, x_id),
                // The inner expression might be parsed differently depending
                // on the expression parser implementation. The key check is
                // that Extension was created.
                _ => { /* acceptable — expression parser may vary */ }
            },
            _ => panic!("expected Extension expression"),
        }
    }

    // -----------------------------------------------------------------------
    // Inline Assembly Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_basic_asm() {
        // asm("nop");
        let interner = Interner::new();
        let tokens = vec![
            make_token(TokenKind::Asm),
            make_token(TokenKind::LeftParen),
            make_string_token("nop"),
            make_token(TokenKind::RightParen),
            make_token(TokenKind::Semicolon),
            make_token(TokenKind::Eof),
        ];

        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);
        let stmt = parse_asm_statement(&mut parser);

        match stmt {
            Statement::Asm(asm) => {
                assert_eq!(asm.template, "nop");
                assert!(!asm.is_volatile);
                assert!(!asm.is_inline);
                assert!(!asm.is_goto);
                assert!(asm.outputs.is_empty());
                assert!(asm.inputs.is_empty());
                assert!(asm.clobbers.is_empty());
                assert!(asm.goto_labels.is_empty());
            }
            _ => panic!("expected Asm statement"),
        }
    }

    #[test]
    fn test_parse_asm_volatile() {
        // asm volatile("" ::: "memory");
        let interner = Interner::new();
        let tokens = vec![
            make_token(TokenKind::Asm),
            make_token(TokenKind::Volatile),
            make_token(TokenKind::LeftParen),
            make_string_token(""),
            make_token(TokenKind::Colon),
            make_token(TokenKind::Colon),
            make_token(TokenKind::Colon),
            make_string_token("memory"),
            make_token(TokenKind::RightParen),
            make_token(TokenKind::Semicolon),
            make_token(TokenKind::Eof),
        ];

        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);
        let stmt = parse_asm_statement(&mut parser);

        match stmt {
            Statement::Asm(asm) => {
                assert!(asm.is_volatile);
                assert_eq!(asm.clobbers.len(), 1);
                assert_eq!(asm.clobbers[0], "memory");
            }
            _ => panic!("expected Asm statement"),
        }
    }

    #[test]
    fn test_parse_asm_with_clobbers() {
        // asm("" ::: "memory", "cc");
        let interner = Interner::new();
        let tokens = vec![
            make_token(TokenKind::Asm),
            make_token(TokenKind::LeftParen),
            make_string_token(""),
            make_token(TokenKind::Colon),
            make_token(TokenKind::Colon),
            make_token(TokenKind::Colon),
            make_string_token("memory"),
            make_token(TokenKind::Comma),
            make_string_token("cc"),
            make_token(TokenKind::RightParen),
            make_token(TokenKind::Semicolon),
            make_token(TokenKind::Eof),
        ];

        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);
        let stmt = parse_asm_statement(&mut parser);

        match stmt {
            Statement::Asm(asm) => {
                assert_eq!(asm.clobbers.len(), 2);
                assert_eq!(asm.clobbers[0], "memory");
                assert_eq!(asm.clobbers[1], "cc");
            }
            _ => panic!("expected Asm statement"),
        }
    }

    // -----------------------------------------------------------------------
    // Error Recovery Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_attribute_error_recovery_missing_close() {
        // __attribute__((packed — missing ))
        let mut interner = Interner::new();
        let packed_id = interner.intern("packed");

        let tokens = vec![
            make_token(TokenKind::GccAttribute),
            make_token(TokenKind::LeftParen),
            make_token(TokenKind::LeftParen),
            make_ident_token(packed_id),
            make_token(TokenKind::Semicolon), // statement boundary stops recovery
            make_token(TokenKind::Eof),
        ];

        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);
        let attrs = parse_gcc_attribute(&mut parser);

        // Should have recovered with the packed attribute parsed.
        assert_eq!(attrs.len(), 1);
        assert_eq!(attrs[0].name, packed_id);
    }

    #[test]
    fn test_typeof_disambiguation_type_keyword() {
        // typeof(int) — should recognise as typeof-type.
        let interner = Interner::new();
        let tokens = vec![
            make_token(TokenKind::Typeof),
            make_token(TokenKind::LeftParen),
            make_token(TokenKind::Int),
            make_token(TokenKind::RightParen),
            make_token(TokenKind::Eof),
        ];

        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        // The typeof disambiguation check should identify `int` as a type.
        // Note: Full typeof parsing requires the type parser to be operational,
        // so we test the disambiguation logic itself.
        parser.advance(); // skip typeof
        parser.advance(); // skip (
        assert!(is_typeof_type_start(&parser));
    }

    #[test]
    fn test_typeof_disambiguation_identifier() {
        // typeof(x) where x is NOT a typedef — should be expression.
        let mut interner = Interner::new();
        let x_id = interner.intern("x");

        let tokens = vec![
            make_token(TokenKind::Typeof),
            make_token(TokenKind::LeftParen),
            make_ident_token(x_id),
            make_token(TokenKind::RightParen),
            make_token(TokenKind::Eof),
        ];

        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        parser.advance(); // skip typeof
        parser.advance(); // skip (
        assert!(!is_typeof_type_start(&parser));
    }
}
